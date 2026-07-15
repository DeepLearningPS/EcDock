
import os, math
from .consistency import * 
from .Equiformerv2 import *
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from ..utils import group_descriptor_to_feats

from ..graphs.datasets import *

from .modules import * 
from .TImodel import * 
from .sampler_base import *

class TreeInvent_Sampler_SBDD(TreeInvent_Sampler_Base):
    def __init__(self,modelname,local_rank=None,**kwargs):
        self.local_rank=local_rank
        self.device=FGP.device
        self.jobstr='nadd+rgen+nconn+nint'
        if not FGP.only_2d:
            self.jobstr+='+coords'
        self.model=TreeInvent_Model(modelname=modelname,local_rank=local_rank,jobs='nadd+rgen+nconn+nint+coords',**kwargs)
        return
    
    def init_c_graphs(self,Datas,pocket_mol):
        self.c_feats,self.c_adjs_mat,self.c_coords,self.c_masks=\
            (self.__to_device(Datas[key]).float() for key in ["C_Feats","C_Adjs_Mat","C_Coords","C_Masks"])
        self.c_gmasks,self.c_dmasks=self.__to_device(Datas["C_GMasks"]),self.__to_device(Datas["C_DMasks"])
        self.p_groups=self.__to_device(Datas["P_Groups"]).long()
        self.p_groups_masks=self.__to_device(Datas["P_Groups_Masks"]).long()
        self.n_pgroups=self.__to_device(Datas["N_PGroups"]).long().view(-1)
        self.c_fix_masks=self.__to_device(Datas["C_Fix_Masks"]).long()
        self.c_flexible_masks=self.__to_device(Datas["C_Flexible_Masks"]).long()
        self.c_atoms=self.__to_device(Datas["C_Atoms"]).long()
        self.c_pocket_labels=self.__to_device(Datas["C_Pocket_Labels"]).long()
        self.c_ligand_labels=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_patoms+FGP.max_latoms)).long()
        self.p_groups_int_masks=self.__to_device(Datas["P_INT_Groups_Masks"]).long()
        self.p_groups_atom_int_masks=self.__to_device(Datas["P_INT_Groups_Atom_Masks"]).long()
        self.p_atom_int_masks=self.__to_device(Datas["P_INT_Atom_Masks"]).long()
        self.p_mols=[pocket_mol]*FGP.batchsize
        self.pl_adjs=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_patoms,FGP.max_latoms,len(FGP.bond_types)+1).float())
        return 
        
        
    def init_nint(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        if mode=='all':
            self.nint_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nint_likelihoods[ids]=0
        return
    
    def update_c_graphs(self):
        
        self.c_feats[:,FGP.max_patoms:]=self.l_feats
        self.c_atoms[:,FGP.max_patoms:]=self.l_atoms
        self.c_adjs_mat[:,FGP.max_patoms:,FGP.max_patoms:]=self.l_adjs_mat
        self.c_adjs_mat[:,:FGP.max_patoms,FGP.max_patoms:]=self.pl_adjs
        self.c_adjs_mat[:,FGP.max_patoms:,:FGP.max_patoms]=self.pl_adjs.transpose(1,2)
        lp_conn_adjs=(self.__to_device(torch.ones_like(self.pl_adjs[:,:,:,0]))*self.p_atom_int_masks.unsqueeze(-1)).transpose(1,2)
        lp_conn_adjs=lp_conn_adjs*self.l_masks.unsqueeze(-1)
        
        self.c_adjs_mat[:,:FGP.max_patoms,FGP.max_patoms:,-1]=lp_conn_adjs.transpose(1,2)
        self.c_adjs_mat[:,FGP.max_patoms:,:FGP.max_patoms,-1]=lp_conn_adjs
        
        self.c_coords[:,FGP.max_patoms:]=self.l_coords
        self.c_masks[:,FGP.max_patoms:]=self.l_masks
        self.c_gmasks[:,FGP.max_patoms:]=self.l_gmasks
        self.c_dmasks[:,FGP.max_patoms:]=self.l_dmasks
        
        self.c_fix_masks[:,FGP.max_patoms:]=self.l_fix_masks
        self.c_flexible_masks[:,FGP.max_patoms:]=self.l_flexible_masks
        self.c_ligand_labels[:,FGP.max_patoms:]=self.l_masks
        
        return 
    
    def select_c_graphs(self,ids,mask_mode='normal',output_mode='simple'):
        if mask_mode=='graph':
            masks=self.c_gmasks
        elif mask_mode=='coords':
            masks=self.c_dmasks
        else:
            masks=self.c_masks
        
        self.update_c_graphs()
        c_feats=self.c_feats*masks.unsqueeze(-1).float()
        c_atoms=self.c_atoms*masks
        masks2D=(masks.unsqueeze(-1)*masks.unsqueeze(-2)).unsqueeze(-1)
        c_adjs_mat=self.c_adjs_mat*masks2D
        c_coords=self.c_coords*masks.unsqueeze(-1).float()
        c_pocket_labels=self.c_pocket_labels[ids]
        c_ligand_labels=self.c_ligand_labels[ids]
        c_feats=c_feats[ids]
        c_atoms=self.c_atoms[ids]
        c_adjs_mat=c_adjs_mat[ids]
        c_coords=c_coords[ids]
        c_masks=masks[ids]
        p_groups=self.p_groups[ids]
        p_groups_masks=self.p_groups_masks[ids]
        c_flexible_masks=self.c_flexible_masks[ids]
        c_fix_masks=self.c_fix_masks[ids]
        n_pgroups=self.n_pgroups[ids]
        
        p_groups_int_masks=self.p_groups_int_masks[ids]
        
        p_groups_atom_int_masks=self.p_groups_atom_int_masks[ids]
        
        p_atom_int_masks=self.p_atom_int_masks[ids]
        
        if output_mode=='simple':
            return c_feats,c_adjs_mat,c_coords,c_masks
        elif output_mode=='middle':
            return c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_atom_int_masks
        else:
            return c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_groups_atom_int_masks,p_atom_int_masks,c_pocket_labels,c_ligand_labels
    
    def select_valid_nint_actions(self,nint_ids,valid_ids,likelihoods):
        batch,int_to,int_type,int_from=nint_ids
        batch=batch[valid_ids]
        int_to=int_to[valid_ids]
        int_type=int_type[valid_ids]
        int_from=int_from[valid_ids]
        likelihoods=likelihoods[valid_ids]
        return batch,int_to,int_type,int_from,likelihoods
    
    def update_pl_adjs_with_nint_actions(self,batch,int_to,int_type,int_from,likelihoods,focus_lgroups,focus_lgroups_mask,p_groups,p_groups_mask):
        for ix,bid in enumerate(batch):
            if bid!=0:
                n_p_atoms=torch.sum(p_groups_mask[ix,int_to[ix]]).long()
                
                #print('int_to,bid',int_to,bid)
                
                int_p_atoms=p_groups[ix,int_to[ix]][:n_p_atoms]
                
                n_l_atoms=torch.sum(focus_lgroups_mask[ix]).long()
                
                int_l_atoms=focus_lgroups[ix,:n_l_atoms]-FGP.max_patoms

                int_types=int_type[ix]+5
                
                p_grids,l_grids=torch.meshgrid(int_p_atoms, int_l_atoms, indexing='ij')
                
                self.pl_adjs[bid,p_grids,l_grids,int_types]=1
                
                self.pl_adjs[bid]=self.pl_adjs[bid]*self.p_atom_int_masks[bid].unsqueeze(-1).unsqueeze(-1)
                
        self.nint_likelihoods[batch]=likelihoods
        return
    
    def reset_zero_indices(self):
        self.l_stop_mask[0]=1
        self.r_stop_mask[0]=1
        self.init_l_graph_zeros_indices()
        self.init_r_graph_zeros_indices()
        self.pl_adjs[0]=0
        self.l_likelihoods[0]=0
        self.r_likelihoods[0]=0
        self.nadd_likelihoods[0]=0
        self.nconn_likelihoods[0]=0
        self.nint_likelihoods[0]=0
        return 
    
    
    def sample_mols(self,MG,nmols_per_complex=10,savepath='./sample_mols',collectpath='./collect_mols',debug=True):
        sampled_mols=[]
        sampled_smis=[]
        sampled_valids=[]

        sample_set,loader=self.__init_sample_loader(MG,nmols_per_complex)
        sample_bar=enumerate(loader)
        ref_lig,ref_pocket=self.record_pocket_and_reflig(MG,savepath)
        print (MG.p_natoms)
        for bid,Datas in sample_bar: 
            conf_step_id=0
            nadd_step_id=0
            rgen_step_id=0
            nconn_step_id=0
            nint_step_id=0
            
            self.sample_prepare()
            self.init_c_graphs(Datas,ref_pocket)
            self.init_l_graphs(mode='all')

            # 0 index graph is always unused to avoid GGNN fails for empty graphs in the total batch.
            # thus it always needs to initialize the graph with only one node, its the same in ring generations.
            
            while torch.sum(self.l_stop_mask[1:])>0:
                self.init_nadd(mode='all')
                self.init_r_graphs(mode='all')
                self.init_nconn(mode='all')
                self.init_nint(mode='all')
                self.update_l_saturation_masks()
                self.gen_topology_constrains()
                self.reset_zero_indices()
                all_ids=self.__to_device(torch.arange(FGP.batchsize).long())
                # initialize the 0 indexed graph with only one node
                # find unstoped complex graphs
                l_unstop_ids=torch.where(self.l_stop_mask==1)[0]
                c_feats,c_adjs_mat,c_coords,c_masks=self.select_c_graphs(l_unstop_ids,mask_mode='normal',output_mode='simple')
                l_nadd_masks=self.select_l_nadd_masks(l_unstop_ids)
                
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nadd_before')

                # sample nadd actions for unstoped complex graphs
                #print ('l_adjs',torch.sum(self.l_adjs_mat[:,:5,:5],dim=-1))
                l_add_ids,l_term_ids,nadd_likelihoods=self.model.nadd_step(c_feats,c_adjs_mat,c_coords,c_masks)
                print (l_term_ids)
                # update the stop mask of complex graphs
                self.l_stop_mask[l_unstop_ids[l_term_ids]]=0

                self.nadd_likelihoods[l_unstop_ids]=nadd_likelihoods

                # trans indices of nadd actions to real complex graph indices
                nadd_mol_ids=l_unstop_ids[l_add_ids[0]]
                nadd_type_ids=l_add_ids[1]

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
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nadd_before')
                nadd_step_id+=1

                if torch.sum(self.l_stop_mask[1:])==0:
                    break

                print('RGEN stage'+'-'*80)
                while torch.sum(self.r_stop_mask[1:])>0:
                    self.reset_zero_indices()
                    # find unstoped ring graphs
                    r_unstop_ids=torch.where(self.r_stop_mask==1)[0]

                    # get unstoped ring graphs and complex graphs
                    c_feats,c_adjs_mat,c_coords,c_masks=self.select_c_graphs(r_unstop_ids,mask_mode='normal',output_mode='simple')
                    
                    r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_round,r_maxatoms,r_focused_ids=self.select_r_graphs(r_unstop_ids)

                    # sample rgen actions
                    r_add_ids,r_conn_ids,r_term_ids,r_likelihoods=\
                                            self.model.rgen_step(c_feats,c_adjs_mat,c_coords,c_masks,
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
                 
                c_feats,c_adjs_mat,c_coords,c_masks\
                    =self.select_c_graphs(nconn_unstop_ids,mask_mode='normal',output_mode='simple')
                l_nconn_masks=self.select_l_nconn_masks(nconn_unstop_ids)

                r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids=self.select_r_graphs(nconn_unstop_ids)
                l_atomnums=self.l_atomnums[nconn_unstop_ids]
                self.record_sampler_states(savepath=f'{savepath}/{bid}',savename='nconn_before')

                nconn_ids,nconn_likelihoods=\
                    self.model.nconn_step(c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_focused_ids,r_masks,l_atomnums,l_nconn_masks)

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

                self.nint_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                self.nint_stop_mask[updated_ids]=1
                stoped_ids=torch.where(self.l_stop_mask==0)[0]
                self.nint_stop_mask[stoped_ids]=0
                
                while torch.sum(self.nint_stop_mask[1:])>0:
                    self.reset_zero_indices()
                    self.nint_stop_mask[0]=1

                    nint_unstop_ids=torch.where(self.nint_stop_mask>0)[0]
                    r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_round,r_maxatoms,r_focused_ids=self.select_r_graphs(nint_unstop_ids)
                    
                    cg_feats,cg_atoms,cg_adjs_mat,cg_coords,cg_masks,\
                        p_groups,p_group_masks,n_pgroups,p_group_int_masks,p_atom_int_masks=self.select_c_graphs(nint_unstop_ids,mask_mode='graph',output_mode='middle')
                    
                    cd_feats,cd_atoms,cd_adjs_mat,cd_coords,cd_masks,\
                        p_groups,p_group_masks,n_pgroups,p_group_int_masks,p_atom_int_masks=self.select_c_graphs(nint_unstop_ids,mask_mode='coords',output_mode='middle')

                    focus_ftypes=r_ftypes
                    focus_lgroups=self.l_groups[nint_unstop_ids,self.l_rounds[nint_unstop_ids]-1]+FGP.max_patoms
                    focus_lgroup_masks=self.l_groups_masks[nint_unstop_ids,self.l_rounds[nint_unstop_ids]-1]
                    
                    focus_lgroup_ids=self.l_rounds[nint_unstop_ids]-1
                        
                    nint_ids,nint_term_ids,nint_likelihoods=self.model.nint_step(cg_feats,cg_adjs_mat,cg_masks,
                                                                                      cd_feats,cd_adjs_mat,cd_coords,cd_masks,
                                                                                      p_groups,p_group_masks,p_group_int_masks,
                                                                                      focus_lgroups,focus_lgroup_masks,focus_ftypes,focus_lgroup_ids)
                    
                    self.nint_stop_mask[nint_unstop_ids[nint_term_ids]]=0
                    self.nint_likelihoods[nint_unstop_ids[nint_term_ids]]=nint_likelihoods[nint_term_ids]
                    self.nint_stop_mask[nint_unstop_ids[nint_term_ids]]=0
                    
                    nint_invalid_ids=self.model.invalid_nint_actions(nint_ids,n_pgroups)
                    self.nint_stop_mask[nint_unstop_ids[nint_invalid_ids]]=0
                    
                    nint_valid_masks=self.nint_stop_mask.clone().detach()
                    nint_valid_masks[nint_unstop_ids[nint_invalid_ids]]=0
                    nint_valid_ids=torch.nonzero(nint_valid_masks[nint_unstop_ids[nint_ids[0]]]>0).view(-1)
                    
                    batch,int_to,int_type,int_from,likelihoods=self.select_valid_nint_actions(nint_ids,nint_valid_ids,nint_likelihoods)
                    batch=nint_unstop_ids[batch]
                    
                    self.update_pl_adjs_with_nint_actions(batch,int_to,int_type,int_from,likelihoods,focus_lgroups,focus_lgroup_masks,p_groups,p_group_masks)
                    nint_step_id+=1
                # update the coords of the complex graphs
                if not FGP.only_2d:
                    print ('Conf stage'+'-'*80) 
                    self.conf_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                    self.conf_stop_mask[updated_ids]=1
                    stoped_ids=torch.where(self.l_stop_mask==0)[0]
                    self.conf_stop_mask[stoped_ids]=0

                    self.reset_zero_indices()
                    self.conf_stop_mask[0]=0
                    coords_unstop_ids=torch.where(self.conf_stop_mask>0)[0]
                    l_atomnums=self.l_atomnums[coords_unstop_ids]
                    
                    if len(coords_unstop_ids)==0:
                        break
                    print (coords_unstop_ids)
                    if True:
                        c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,\
                            p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_groups_atom_int_masks,p_atom_int_masks,c_pocket_labels,c_ligand_labels\
                                =self.select_c_graphs(coords_unstop_ids,mask_mode='graphs',output_mode='all')
                        print (l_atomnums,torch.where(l_atomnums<6,1,0).any())
                        c_adjs_mat[0,FGP.max_patoms:,FGP.max_patoms:]=0                        
                        l_mols=self.select_l_mols(coords_unstop_ids)
                        p_mols=self.select_p_mols(coords_unstop_ids)
                        c_mols=self.combine_c_mols(p_mols,l_mols)

                        if torch.where(l_atomnums>1,1,0).any():
                            if torch.where(l_atomnums<6,1,0).any():
                                #print ('c_adjs_mat diagonals',c_adjs_mat[:,indices,indices].shape,torch.where(torch.sum(c_adjs_mat[:,indices,indices],dim=-1)))  
                                samples,samples_diff,samples_diff_before,_=self.model.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_masks,c_flexible_masks,c_pocket_labels,c_ligand_labels,
                                    with_MMFF_guide=False,
                                    guide_loops=FGP.MMFF_guide_loops,
                                    guide_type=FGP.MMFF_guide_type,
                                    show_state=True,
                                    guide_model=FGP.MMFF_guide_model,
                                    rdkit_mols=c_mols)
                            else:
                                c_feats_dup, c_adjs_mat_dup, c_coords_dup, c_atoms_dup, c_masks_dup, c_mols_dup\
                                    =self.conf_duplications(c_feats,c_adjs_mat,c_coords,c_atoms,c_masks,c_mols,duplicates=FGP.conf_duplications)
                                n=c_coords.shape[1]
                                samples,samples_diff,samples_diff_before,energies=self.model.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_masks,c_flexible_masks,c_pocket_labels,c_ligand_labels,
                                    with_MMFF_guide=FGP.with_MMFF_guide,
                                    guide_loops=FGP.MMFF_guide_loops,
                                    guide_type=FGP.MMFF_guide_type,
                                    show_state=True,
                                    guide_model=FGP.MMFF_guide_model,
                                    rdkit_mols=c_mols)
                                
                                if energies is not None:
                                    min_e_ids=energies.view(-1,FGP.conf_duplications).argmin(dim=1)
                                    samples=samples.view(-1,FGP.conf_duplications,n,3)
                                    samples=samples[torch.arange(samples.shape[0]),min_e_ids]
                                else:
                                    samples=samples.view(-1,FGP.conf_duplications,n,3) 
                                    samples=samples[:,0]

                            clash_ids,pass_ids=self.detect_clash(samples,c_masks)
                            self.l_stop_mask[coords_unstop_ids[clash_ids]]=0

                        self.update_l_graphs_with_conf_actions(coords_unstop_ids[pass_ids],samples[pass_ids])
                    
                        conf_step_id+=1
                        self.record_conf(path=f'{savepath}/{bid}')
                        print ('STOP!'*20)

            mols,smis,valid_ids=self.l_graphs_to_mol(with_coords=True)
            sampled_mols+=mols
            sampled_smis+=smis
            sampled_valids+=valid_ids
            self.record_conf(path=f'{collectpath}-{bid}',debug=False)
                
        return mols,smis,valid_ids
    
    def __init_sample_loader(self,MG,nmols_per_complex=10):

        nmols_per_complex=math.ceil(nmols_per_complex/(FGP.batchsize-1))*FGP.batchsize
        if self.local_rank is not None:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            sampler=torch.utils.data.distributed.DistributedSampler(sample_set)
            loader=DataLoader(sample_set,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers,sampler=sampler)
        else:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            loader=DataLoader(sample_set,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers)

        return sample_set,loader
         
    def __to_device(self,tensor):
        if self.local_rank is not None:
            return tensor.cuda(self.local_rank)
        else:
            return tensor.cuda()
    
    def select_p_mols(self,ids):
        mols=[]
        for id in ids:
            mols.append(self.p_mols[id])
        return mols

    def combine_c_mols(self,p_mols,l_mols):
        assert len(l_mols)==len(p_mols), 'l_mols and p_mols should have the same length'
        c_mols=[]
        for lmol,pmol in zip(l_mols,p_mols):
            c_mols.append(Chem.CombineMols(pmol,lmol))
        return c_mols
    
    def conf_duplications(self,c_feats,c_adjs_mat,c_coords,c_atoms,c_masks,c_mols,duplicates=1):
        b,n,fd=c_feats.shape
        c_feats_dup=c_feats.unsqueeze(1).repeat(1,duplicates,1,1).view(-1,n,fd)
        adjsd=c_adjs_mat.shape[-1]
        c_adjs_mat_dup=c_adjs_mat.unsqueeze(1).repeat(1,duplicates,1,1,1).view(-1,n,n,adjsd)
        c_coords_dup=c_coords.unsqueeze(1).repeat(1,duplicates,1,1).view(-1,n,3)
        c_atoms_dup=c_atoms.unsqueeze(1).repeat(1,duplicates,1).view(-1,n)
        c_masks_dup=c_masks.unsqueeze(1).repeat(1,duplicates,1).view(-1,n)
        c_mols_dup=[mol  for mol in c_mols for i in range(duplicates)]
        return c_feats_dup, c_adjs_mat_dup, c_coords_dup, c_atoms_dup, c_masks_dup, c_mols_dup
    
    def detect_clash(self,coords,gmasks):
        gmasks_2D=(gmasks.unsqueeze(-1)*(gmasks.unsqueeze(-1).permute(0,2,1))).bool()
        pred_dismat=torch.cdist(coords,coords,compute_mode='donot_use_mm_for_euclid_dist')
        print ('pred_dismat',pred_dismat.shape)
        print ('gmasks_2D',gmasks_2D.shape)
        clash_ids=[] 
        pass_ids=[]
        b=pred_dismat.shape[0]
        for i in range(b):
            detach_masks=gmasks_2D[i][:FGP.max_patoms,FGP.max_patoms:]
            detach_cdist=pred_dismat[i][:FGP.max_patoms,FGP.max_patoms:]
            if torch.any(detach_cdist[detach_masks]<2):
                print (f'clash detected in {i}!')
                clash_ids.append(i)
            else:
                pass_ids.append(i)
        clash_ids=self.__to_device(torch.Tensor(clash_ids).long())
        pass_ids=self.__to_device(torch.Tensor(pass_ids).long())
        return clash_ids,pass_ids

    def copy_to_l_graph_wrapper(self):
        self.l_feats_wrap=self.l_feats.clone().detach()
        self.l_adjs_mat_wrap=self.l_adjs_mat.clone().detach()
        self.l_adjs_wrap=self.l_adjs.clone().detach()
        self.l_coords_wrap=self.l_coords.clone().detach()
        self.l_atoms_wrap=self.l_atoms.clone().detach()
        self.l_masks_wrap=self.l_masks.clone().detach()
        self.l_gmasks_wrap=self.l_gmasks.clone().detach()
        self.l_dmasks_wrap=self.l_dmasks.clone().detach()
        self.l_rounds_wrap=self.l_rounds.clone().detach()
        self.l_groups_wrap=self.l_groups.clone().detach()
        self.l_groups_masks_wrap=self.l_groups_masks.clone().detach()
        self.l_likelihoods_wrap=self.l_likelihoods.clone().detach()
        self.l_fix_masks_wrap=self.l_fix_masks.clone().detach()
        self.l_flexible_masks_wrap=self.l_flexible_masks.clone().detach()
        self.l_pts_wrap=self.l_pts.clone().detach()
        self.l_bond_orders_wrap=self.l_bond_orders.clone().detach()
        self.l_max_bonds_wrap=self.l_max_bonds.clone().detach()
        self.l_total_bonds_wrap=self.l_total_bonds.clone().detach()
        self.l_allowed_bonds_wrap=self.l_allowed_bonds.clone().detach()
        self.l_saturation_masks_wrap=self.l_saturation_masks.clone().detach()
        self.l_possible_conn_masks_wrap=self.l_possible_conn_masks.clone().detach()
        self.l_mols_wrap=self.l_mols.clone().detach()
    
    def rollback_l_graph_from_wrapper(self,ids):
        self.l_feats[ids]=self.l_feats_wrap[ids]
        self.l_adjs_mat[ids]=self.l_adjs_mat_wrap[ids]
        self.l_adjs[ids]=self.l_adjs_wrap[ids]
        self.l_coords[ids]=self.l_coords_wrap[ids]
        self.l_atoms[ids]=self.l_atoms_wrap[ids]
        self.l_masks[ids]=self.l_masks_wrap[ids]
        self.l_gmasks[ids]=self.l_gmasks_wrap[ids]
        self.l_dmasks[ids]=self.l_dmasks_wrap[ids]
        self.l_rounds[ids]=self.l_rounds_wrap[ids]
        self.l_groups[ids]=self.l_groups_wrap[ids]
        self.l_groups_masks[ids]=self.l_groups_masks_wrap[ids]
        self.l_likelihoods[ids]=self.l_likelihoods_wrap[ids]
        self.l_fix_masks[ids]=self.l_fix_masks_wrap[ids]
        self.l_flexible_masks[ids]=self.l_flexible_masks_wrap[ids]
        self.l_pts[ids]=self.l_pts_wrap[ids]
        self.l_bond_orders[ids]=self.l_bond_orders_wrap[ids]
        self.l_max_bonds[ids]=self.l_max_bonds_wrap[ids]
        self.l_total_bonds[ids]=self.l_total_bonds_wrap[ids]
        self.l_allowed_bonds[ids]=self.l_allowed_bonds_wrap[ids]
        self.l_saturation_masks[ids]=self.l_saturation_masks_wrap[ids]
        self.l_possible_conn_masks[ids]=self.l_possible_conn_masks_wrap[ids]
        self.l_mols[ids]=self.l_mols_wrap[ids]
        return


