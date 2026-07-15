from .consistency import * 
from .Equiformerv2 import *
import pickle,os,tempfile, shutil, zipfile, time, math 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from ..utils import group_descriptor_to_feats
from ..graphs.datasets import *
from .modules import * 
from .TImodel import * 
from ..utils.utils_rdkit import mol_with_atom_index_and_formal_charge
from .sampler_base import TreeInvent_Sampler_Base

class TreeInvent_Sampler_LBDD(TreeInvent_Sampler_Base):
    def __init__(self,modelname,local_rank=None,**kwargs):
        
        #super().__init__(modelname=modelname,local_rank=local_rank,**kwargs)
        self.local_rank=local_rank
        self.batchsize=FGP.batchsize*FGP.accsteps
        self.device=FGP.device
        self.jobstr='nadd+rgen+nconn'
        if not FGP.only_2d:
            self.jobstr+='+coords'
        if FGP.with_term_model:
            self.jobstr+='+term' 
        self.model=TreeInvent_Model(modelname=modelname,local_rank=local_rank,jobs=self.jobstr,**kwargs)
        return
    
    def sample_mols(self,nmols=10,savepath='./sample_mols',collectpath='./collect_mols',debug=True):
        sampled_mols=[]
        sampled_smis=[]
        sampled_valids=[]
        nmols=math.ceil(nmols/(FGP.batchsize-1))*FGP.batchsize
        nbatch=math.ceil(nmols/FGP.batchsize)
        for bid in range(nbatch):
            conf_step_id=0
            nadd_step_id=0
            rgen_step_id=0
            nconn_step_id=0
            self.sample_prepare()

            self.init_l_graphs(mode='all')

            # 0 index graph is always unused to avoid GGNN fails for empty graphs in the total batch.
            # thus it always needs to initialize the graph with only one node, its the same in ring generations.
            while torch.sum(self.l_stop_mask[1:])>0:
                self.init_nadd(mode='all')
                self.init_r_graphs(mode='all')
                self.init_nconn(mode='all')
                self.init_term(mode='all')
                self.reset_zero_indices()
                self.update_l_saturation_masks()
                self.gen_topology_constrains()
                # initialize the 0 indexed graph with only one node
                # find unstoped complex graphs
                if FGP.with_term_model:
                    l_unstop_ids=torch.where(self.l_stop_mask==1)[0]
                    unempty_ids=torch.where(self.l_rounds[l_unstop_ids]>0)[0]
                    print ('unempty_ids',unempty_ids)
                    if len(unempty_ids)>0:
                        l_unempty_ids=l_unstop_ids[unempty_ids]
                        l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds=self.select_l_graphs(l_unempty_ids)
                        l_term_masks=self.select_l_term_masks(l_unempty_ids)
                        l_non_term_ids,l_term_ids,term_likelihoods=self.model.term_step(l_feats,l_adjs_mat,l_coords,l_masks)
                        print (l_term_ids,l_unempty_ids,self.l_stop_mask)
                        self.l_stop_mask[l_unempty_ids[l_term_ids]]=0
                        self.term_likelihoods[l_unempty_ids]=term_likelihoods
                
                l_unstop_ids=torch.where(self.l_stop_mask==1)[0]
                l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds=self.select_l_graphs(l_unstop_ids)
                l_nadd_masks=self.select_l_nadd_masks(l_unstop_ids)
                
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nadd_before')
                
                # sample nadd actions for unstoped complex graphs
                if FGP.with_term_model:
                    l_add_ids,nadd_likelihoods=self.model.nadd_step(l_feats,l_adjs_mat,l_coords,l_masks)
                else:
                    l_add_ids,l_term_ids,nadd_likelihoods=self.model.nadd_step(l_feats,l_adjs_mat,l_coords,l_masks)
                    print (l_term_ids)
                    self.l_stop_mask[l_unstop_ids[l_term_ids]]=0
                
                self.nadd_likelihoods[l_unstop_ids]=nadd_likelihoods
                
                # trans indices of nadd actions to real complex graph indices
                nadd_mol_ids=l_unstop_ids[l_add_ids[0]]
                nadd_type_ids=l_add_ids[1]
                if FGP.with_term_model:
                    self.record_nadd_actions(nadd_mol_ids,nadd_type_ids,term_ids=None,savepath=f'{savepath}/{bid}')
                else:
                    self.record_nadd_actions(nadd_mol_ids,nadd_type_ids,l_term_ids,savepath=f'{savepath}/{bid}')
                # update single atom groups first
                self.r_stop_mask=self.l_stop_mask.clone().detach()    
                single_node_ids=torch.where(nadd_type_ids<len(FGP.atom_types))[0]            
                self.update_single_atom_r_graphs(nadd_mol_ids,nadd_type_ids,single_node_ids)
                self.r_stop_mask[nadd_mol_ids[single_node_ids]]=0 
                self.r_ftypes[nadd_mol_ids]=self.gid_to_gfeats[nadd_type_ids]
                self.update_r_maxatoms()
                
                #start fragment generation with rgen model
                self.record_nadd(path=f'{savepath}/{bid}')
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nadd_after')
                nadd_step_id+=1

                if torch.sum(self.l_stop_mask[1:])==0:
                    break

                print('RGEN stage'+'-'*80)
                while torch.sum(self.r_stop_mask[1:])>0:
                    self.reset_zero_indices()
                    # find unstoped ring graphs
                    r_unstop_ids=torch.where(self.r_stop_mask==1)[0]

                    # get unstoped ring graphs and complex graphs
                    l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds=self.select_l_graphs(r_unstop_ids)
                    
                    r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_round,r_maxatoms,r_focused_ids=self.select_r_graphs(r_unstop_ids)

                    # sample rgen actions
                    r_add_ids,r_conn_ids,r_term_ids,r_likelihoods=\
                                            self.model.rgen_step(l_feats,l_adjs_mat,l_coords,l_masks,
                                                                            r_feats,r_adjs_mat,r_ftypes,r_masks,r_atomnums)

                    # update the stop mask of ring graphs
                    r_invalid_ids,r_full_ids=self.model.invalid_rgen_actions(r_add_ids,r_conn_ids,r_term_ids,r_adjs_mat,r_atomnums,r_maxatoms)

                    self.r_stop_mask[r_unstop_ids[r_term_ids]]=0
                    self.r_stop_mask[r_unstop_ids[r_full_ids]]=0
                    self.r_likelihoods[r_unstop_ids[r_term_ids],self.r_rounds[r_unstop_ids[r_term_ids]]]=r_likelihoods[r_term_ids]
                    
                    # full ids are invalid actions, should not be updated by RL, thus, we didn't collect its likelihoods
                    self.init_r_graphs(ids=r_unstop_ids[r_invalid_ids],mode='failed')
                    
                    r_valid_mask=self.r_stop_mask.clone().detach()
                    r_valid_mask[r_unstop_ids[r_invalid_ids]]=0
                    r_add_valid_ids=torch.nonzero(r_valid_mask[r_unstop_ids[r_add_ids[0]]]>0)
                    r_conn_valid_ids=torch.nonzero(r_valid_mask[r_unstop_ids[r_conn_ids[0]]]>0)
                    
                    batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods\
                        =self.select_valid_r_add_actions(r_add_ids,r_add_valid_ids,r_likelihoods)
                        
                    batch=r_unstop_ids[batch]
                    self.update_r_graphs_with_add_actions(batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods)
                    
                    batch,bond_to,bond_type,bond_from,likelihoods=self.select_valid_r_conn_actions(r_conn_ids,r_conn_valid_ids,r_likelihoods)
                    batch=r_unstop_ids[batch]
                    self.update_r_graphs_with_conn_actions(batch,bond_to,bond_type,bond_from,likelihoods)
                    #print (self.r_feats[1])
                    rgen_step_id+=1
                    #self.record_rgen(path=f'{savepath}/{bid}',gen_process_path=f'{rgen_step_id}')
                self.record_rgen(path=f'{savepath}/{bid}')
                # update the complex graphs with the generated fragments
                
                print ('Nconn stage'+'-'*80)
                self.reset_zero_indices()
                
                l_full_ids=torch.where(self.l_atomnums+self.r_atomnums>FGP.max_latoms)
                self.l_stop_mask[l_full_ids]=0
                self.nconn_stop_mask=self.l_stop_mask.clone().detach()
                nconn_unstop_ids=torch.where(self.nconn_stop_mask>0)[0]
                
                l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds=self.select_l_graphs(nconn_unstop_ids)
                l_nconn_masks=self.select_l_nconn_masks(nconn_unstop_ids)
                
                r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids=self.select_r_graphs(nconn_unstop_ids)
                l_atomnums=self.l_atomnums[nconn_unstop_ids]
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nconn_before')
                nconn_ids,nconn_likelihoods=\
                    self.model.nconn_step(l_feats,l_adjs_mat,l_coords,l_masks,r_feats,r_adjs_mat,r_focused_ids,r_masks,l_atomnums,l_nconn_masks)

                # select valid node connection actions
                nconn_invalid_ids=self.model.invalid_nconn_actions(nconn_ids,l_atomnums)
                
                nconn_valid_mask=self.nconn_stop_mask.clone().detach()
                nconn_valid_mask[nconn_unstop_ids[nconn_invalid_ids]]=0
                
                nconn_valid_ids=torch.nonzero(nconn_valid_mask[nconn_unstop_ids[nconn_ids[0]]]>0).view(-1)
                
                # select valid node connection actions
                batch,bond_to,bond_type,bond_from,likelihoods=self.select_valid_nconn_actions(nconn_ids,nconn_valid_ids,nconn_likelihoods)
                batch=nconn_unstop_ids[batch]
                self.record_nconn_actions(batch,bond_to,bond_type,bond_from,likelihoods,savepath=f'{savepath}/{bid}')
                self.update_l_graphs_with_nconn_actions(batch, bond_to, bond_type, bond_from, likelihoods)
                if conf_step_id>0:
                    nconn_step_id+=1
                    
                # update the l graphs with the generated fragments
                l_update_ids=torch.where(self.l_stop_mask>0)[0]
                started_l_ids=torch.where(self.l_rounds[l_update_ids]==0)[0]
                started_l_ids=l_update_ids[started_l_ids]
                
                updated_ids=torch.unique(torch.cat((started_l_ids,batch)))
                self.update_l_graphs_with_rgraphs(updated_ids)
                self.record_nconn(path=f'{savepath}/{bid}')
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nconn_after')
                
                mols,_,_=self.l_graphs_to_mol()
                self.l_mols[1:]=mols
                print (self.l_mols)
                for i in range(FGP.batchsize-1):
                    if self.l_mols[i+1] is None:
                        self.l_stop_mask[i+1]=0

                # update the coords of the complex graphs
                if FGP.graph_3D_encoder_type!='2D':
                    print ('Conf stage'+'-'*80) 
                    self.conf_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                    self.conf_stop_mask[updated_ids]=1
                    stoped_ids=torch.where(self.l_stop_mask==0)[0]
                    self.conf_stop_mask[stoped_ids]=0

                    self.reset_zero_indices()
                    self.conf_stop_mask[0]=0
                    coords_unstop_ids=torch.where(self.conf_stop_mask>0)[0]
                    print ('coords_unstop_ids',coords_unstop_ids)
                    if len(coords_unstop_ids)==0:
                        break
                    
                    if True:
                        l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds=self.select_l_graphs(coords_unstop_ids)
                        l_mols=self.select_l_mols(coords_unstop_ids)
                        print (l_atomnums,torch.where(l_atomnums<6,1,0).any())

                        if torch.where(l_atomnums>1,1,0).any():
                            if torch.where(l_atomnums<6,1,0).any():
                                samples,samples_diff,samples_diff_before,_=self.model.sample_coords_batch(
                                        l_feats,l_atoms,l_adjs_mat,l_coords,l_masks, 
                                        g_ligand_labels=l_masks, 
                                        with_MMFF_guide=False,
                                        guide_loops=FGP.MMFF_guide_loops,
                                        guide_type=FGP.MMFF_guide_type,
                                        show_state=True,
                                        guide_model=FGP.MMFF_guide_model,
                                        rdkitmols=l_mols) 

                            else:                                
                                l_feats_dup, l_adjs_mat_dup, l_coords_dup, l_atoms_dup, l_masks_dup, l_mols_dup\
                                    =self.conf_duplications(l_feats,l_adjs_mat,l_coords,l_atoms,l_masks,l_mols,duplicates=FGP.conf_duplications)
                                n=l_coords.shape[1]
                                samples,samples_diff,samples_diff_before,energies=self.model.sample_coords_batch(
                                        l_feats_dup,l_atoms_dup,l_adjs_mat_dup,l_coords_dup,l_masks_dup, 
                                        g_ligand_labels=l_masks_dup,  
                                        with_MMFF_guide=FGP.with_MMFF_guide,
                                        guide_loops=FGP.MMFF_guide_loops,
                                        guide_type=FGP.MMFF_guide_type,
                                        show_state=True,
                                        guide_model=FGP.MMFF_guide_model,
                                        rdkitmols=l_mols_dup)
                                
                                if energies is not None:
                                    min_e_ids=energies.view(-1,FGP.conf_duplications).argmin(dim=1)
                                    samples=samples.view(-1,FGP.conf_duplications,n,3)
                                    samples=samples[torch.arange(samples.shape[0]),min_e_ids]
                                else:
                                    samples=samples.view(-1,FGP.conf_duplications,n,3) 
                                    samples=samples[:,0]

                            self.update_l_graphs_with_conf_actions(coords_unstop_ids,samples)
                        
                        conf_step_id+=1
                        self.record_conf(path=f'{savepath}/{bid}')
                        print ('STOP!'*20)

            mols,smis,valid_ids=self.l_graphs_to_mol(with_coords=True)
            sampled_mols+=mols
            sampled_smis+=smis
            sampled_valids+=valid_ids
            self.record_conf(path=f'{collectpath}-{bid}',debug=False)
            
        return mols,smis,valid_ids
    def __to_device(self,tensor):
        if self.local_rank is not None:
            return tensor.cuda(self.local_rank)
        else:
            return tensor.cuda()            
    
    def conf_duplications(self,l_feats,l_adjs_mat,l_coords,l_atoms,l_masks,l_mols,duplicates=1):
        b,n,fd=l_feats.shape
        l_feats_dup=l_feats.unsqueeze(1).repeat(1,duplicates,1,1).view(-1,n,fd)
        adjsd=l_adjs_mat.shape[-1]
        l_adjs_mat_dup=l_adjs_mat.unsqueeze(1).repeat(1,duplicates,1,1,1).view(-1,n,n,adjsd)
        l_coords_dup=l_coords.unsqueeze(1).repeat(1,duplicates,1,1).view(-1,n,3)
        l_atoms_dup=l_atoms.unsqueeze(1).repeat(1,duplicates,1).view(-1,n)
        l_masks_dup=l_masks.unsqueeze(1).repeat(1,duplicates,1).view(-1,n)
        l_mols_dup=[mol  for mol in l_mols for i in range(duplicates)]
        print (l_mols_dup)
        print (l_feats.shape,l_feats_dup.shape)
        print (l_adjs_mat.shape,l_adjs_mat_dup.shape)
        print (l_coords.shape,l_coords_dup.shape)
        print (l_atoms.shape,l_atoms_dup.shape)
        print (l_masks.shape,l_masks_dup.shape)
        
        return l_feats_dup, l_adjs_mat_dup, l_coords_dup,  l_atoms_dup, l_masks_dup, l_mols_dup
    