import torch

#from .equiformer import * 
from .consistency import * 
#from .en_transformer import * 
from .Equiformerv2 import *

import pickle,os,tempfile, shutil, zipfile, time, math, tqdm 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from ..utils import group_descriptor_to_feats
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from ..graphs.datasets import *
from tqdm import tqdm 
from torch import distributed as dist
from .modules import * 
from .TImodel import * 
class TreeInvent_Sampler_SBDD:
    def __init__(self,modelname,local_rank=None,**kwargs):
        self.local_rank=local_rank
        self.batchsize=FGP.batchsize*FGP.accsteps
        self.device=FGP.device
        self.model=TreeInvent_Model(modelname=modelname,local_rank=local_rank,jobs='nadd+rgen+nconn+nint+coords',**kwargs)
        return
    
    def init_l_graphs(self,ids=None,mode='all'):  
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        
        if mode=='all':             
            self.l_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types))).float()
            self.l_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,FGP.max_latoms,len(FGP.bond_types)+1)).float()
            self.l_coords=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,3).float())
            self.l_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_gmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_dmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_rounds=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_groups=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).long())
            self.l_groups_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).float())
            self.pl_adjs=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_patoms,FGP.max_latoms,len(FGP.bond_types)+1).float())
            self.l_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
            self.l_fix_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_flexible_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
        else:
            self.l_feats[ids]=0
            self.l_adjs_mat[ids]=0
            self.l_atomnums[ids]=0
            self.l_atoms[ids]=0
            self.l_masks[ids]=0
            self.l_gmasks[ids]=0
            self.l_dmasks[ids]=0
            self.l_rounds[ids]=0
            self.l_groups[ids]=0
            self.l_groups_masks[ids]=0
            self.pl_adjs[ids]=0
            self.l_likelihoods[ids]=0
            self.l_fix_masks[ids]=0
            self.l_flexible_masks[ids]=0
        return 
      
    def init_r_graphs(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode=='all':
            self.r_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types)).float())
            self.r_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,FGP.max_lgroup_size,len(FGP.bond_types)+1).float())
            self.r_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.r_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_focused_ids=self.__to_device(torch.zeros(FGP.batchsize,1).long())
            self.r_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_rounds=self.__to_device(torch.zeros(FGP.batchsize)).long()
            self.r_ftypes=self.__to_device(torch.zeros(FGP.batchsize,FGP.n_group_feats)).float()
            self.r_likelihoods=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size*4).float())
            self.r_maxatoms=self.__to_device(torch.zeros(FGP.batchsize).long())
            
        else:
            
            self.r_feats[ids]=0
            self.r_adjs_mat[ids]=0
            self.r_atomnums[ids]=0
            self.r_atoms[ids]=0
            self.r_focused_ids[ids]=0
            self.r_masks[ids]=0
            self.r_rounds[ids]=0
            self.r_likelihoods[ids]=0
            if mode!='failed':
                self.r_ftypes[ids]=0
                self.r_maxatoms[ids]=0
                
        return 
    
    def init_c_graphs(self,Datas):
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
        return 
        
    def init_nconn(self,ids=None,mode='all'):
        if mode !='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode== 'all':
            self.nconn_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
            
        else:
            self.nconn_likelihoods[ids]=0
            
        return 
        
    def init_nint(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        if mode=='all':
            self.nint_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nint_likelihoods[ids]=0
        return
    
    def init_nadd(self,ids=None,mode='all'):
        if mode !='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        if mode=='all':
            self.nadd_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nadd_likelihoods[ids]=0
        return
        
    def select_l_graphs(self,ids):
        l_feats=self.l_feats[ids]
        l_adjs_mat=self.l_adjs_mat[ids]
        l_coords=self.l_coords[ids]
        l_atomnums=self.l_atomnums[ids]
        l_atoms=self.l_atoms[ids]
        l_masks=self.l_masks[ids]
        l_rounds=self.l_rounds[ids]
        return l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds
        
    def select_r_graphs(self,ids):
        
        r_feats=self.r_feats[ids]
        r_adjs_mat=self.r_adjs_mat[ids]
        r_atomnums=self.r_atomnums[ids]
        r_atoms=self.r_atoms[ids]
        r_masks=self.r_masks[ids]
        r_rounds=self.r_rounds[ids]
        r_ftypes=self.r_ftypes[ids]
        r_maxatoms=self.r_maxatoms[ids]
        r_focused_ids=self.r_focused_ids[ids]
        
        return r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids
    
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
    
    def sample_prepare(self):
        self.l_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.r_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.nint_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.gid_to_gfeats=[]
        for i in range(len(FGP.group_index_type_dict.keys())):
            self.gid_to_gfeats.append(group_descriptor_to_puzzled_group_feats(FGP.group_index_type_dict[i]))
        self.gid_to_gfeats=np.array(self.gid_to_gfeats)
        self.gid_to_gfeats=self.__to_device(torch.Tensor(self.gid_to_gfeats)).float()
        self.single_atom_node_feats=[]
        for i in range(len(FGP.atom_types)):
            self.single_atom_node_feats.append(onek_encoding_unk(FGP.atom_types[i],FGP.atom_types_for_feats)+onek_encoding_unk(0,FGP.formal_charge_types))
        self.single_atom_node_feats=self.__to_device(torch.Tensor(self.single_atom_node_feats).float())
        return
    
    def update_r_maxatoms(self):
        r_ids=torch.nonzero(self.r_ftypes[:,0]).view(-1)
        self.r_maxatoms[r_ids]=torch.sum(self.r_ftypes[r_ids,1:10],dim=1).long()
        single_ids=torch.nonzero(torch.sum(self.r_ftypes[:,1:10],dim=1)==1)
        self.r_maxatoms[single_ids]=1
        pharm_ids=torch.nonzero(torch.sum(self.r_ftypes[:,1:10],dim=1)<0)
        self.r_maxatoms[pharm_ids]=self.r_ftypes[pharm_ids,10].long()
        return 
    
    def update_single_atom_r_graphs(self,nadd_mol_ids,nadd_type_ids,single_node_ids):
        # nadd_mol_ids should correspond to all graphs
        
        self.r_feats[nadd_mol_ids[single_node_ids],0]=self.single_atom_node_feats[nadd_type_ids[single_node_ids]]
        self.r_atomnums[nadd_mol_ids[single_node_ids]]+=1
        self.r_atoms[nadd_mol_ids[single_node_ids],0]=self.__to_device(torch.Tensor(FGP.atom_types).long())[nadd_type_ids[single_node_ids]]
        self.r_masks[nadd_mol_ids[single_node_ids],:1]=1
        return 
    
    def select_valid_r_add_actions(self,r_add_ids,valid_ids,likelihoods):
        # the valid_ids should corresponds to r_add_ids
        
        (batch,bond_to,atom_type,charge,bond_type,bond_from)=r_add_ids
        
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        atom_type=atom_type[valid_ids]
        charge=charge[valid_ids]
        bond_type=bond_type[valid_ids] 
        likelihoods=likelihoods[valid_ids]
        
        return batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods
    
    def update_r_graphs_with_add_actions(self,batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods):
        # the batch should correspond to the ids to self.r_graphs
        
        self.r_feats[batch,bond_from,atom_type+2]=1
        self.r_feats[batch,bond_from,charge+len(FGP.atom_types_for_feats)]=1
        
        non_empty_r_ids=torch.nonzero(self.r_atomnums[batch]>0)
        batch_non_empty=batch[non_empty_r_ids]
        bond_to_non_empty=bond_to[non_empty_r_ids]
        bond_from_non_empty=bond_from[non_empty_r_ids]
        bond_type_non_empty=bond_type[non_empty_r_ids]
        
        self.r_atomnums[batch]+=1
        self.r_atoms[batch,bond_from]=\
            self.__to_device(torch.Tensor(FGP.atom_types).long())[atom_type]
        
        self.r_adjs_mat[batch_non_empty,
                        bond_from_non_empty,
                        bond_to_non_empty,
                        bond_type_non_empty+1]=1
        
        self.r_adjs_mat[batch_non_empty,
                        bond_to_non_empty,
                        bond_from_non_empty,
                        bond_type_non_empty+1]=1
        for b in batch:
            self.r_masks[b,:self.r_atomnums[b]]=1
            
        self.r_likelihoods[batch,self.r_rounds[batch]]=likelihoods
        self.r_rounds[batch]+=1
        return
    
    def select_valid_r_conn_actions(self,r_conn_ids,valid_ids,likelihoods):
        batch,bond_to,bond_type,bond_from=r_conn_ids
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        bond_type=bond_type[valid_ids]
        likelihoods=likelihoods[valid_ids]
        return batch,bond_to,bond_type,bond_from,likelihoods

    def update_r_graphs_with_conn_actions(self,batch,bond_to,bond_type,bond_from,likelihoods):
        #print ('batch:',batch,'bond_to:',bond_to,'bond_type:',bond_type,'bond_from:',bond_from)
        self.r_adjs_mat[batch,bond_from,bond_to,bond_type+1]=1
        self.r_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.r_likelihoods[batch,self.r_rounds[batch]]=likelihoods
        self.r_rounds[batch]+=1
        return
    
    def select_valid_nconn_actions(self,nconn_ids,valid_ids,likelihoods):
        batch,bond_to,bond_type,bond_from=nconn_ids
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        bond_type=bond_type[valid_ids]
        likelihoods=likelihoods[valid_ids]
        return batch,bond_to,bond_type,bond_from,likelihoods
    
    def update_l_graphs_with_nconn_actions(self,batch,bond_to,bond_type,bond_from,likelihoods):
        self.l_adjs_mat[batch,bond_from,bond_to,bond_type+1]=1
        self.l_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.nconn_likelihoods[batch]=likelihoods
        return 
    
    def update_l_graphs_with_rgraphs(self,ids):
        for i in ids:
            l_natoms=self.l_atomnums[i]
            r_natoms=self.r_atomnums[i]
            self.l_feats[i,l_natoms:l_natoms+r_natoms]=self.r_feats[i,:r_natoms]
            self.l_adjs_mat[i,l_natoms:l_natoms+r_natoms,l_natoms:l_natoms+r_natoms]=self.r_adjs_mat[i,:r_natoms,:r_natoms]
            self.l_atoms[i,l_natoms:l_natoms+r_natoms]=self.r_atoms[i,:r_natoms]
            self.l_groups[i,self.l_rounds[i],:r_natoms]=l_natoms+self.__to_device(torch.arange(r_natoms).long())
            self.l_groups_masks[i,self.l_rounds[i],:r_natoms]=1
            self.l_masks[i,:l_natoms+r_natoms]=1
            self.l_gmasks[i,l_natoms:l_natoms+r_natoms]=1
            self.l_atomnums[i]+=r_natoms
            self.l_rounds[i]+=1            
            
            self.l_fix_masks[i]=0
            self.l_flexible_masks[i]=0
            if FGP.l_mode=='fix-previous':
                self.l_fix_masks[i,:l_natoms]=1
                self.l_fix_masks[i,l_natoms:self.l_atomnums[i]]=1
            else:    
                self.l_fix_masks[i,:self.l_atomnums[i]]=0
                self.l_flexible_masks[i,:self.l_atomnums[i]]=1
        return 
    
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

    def update_l_graphs_with_conf_actions(self,ids,c_coords):
        self.l_coords[ids]=c_coords[:,FGP.max_patoms:]
        self.l_dmasks[ids]=self.l_gmasks[ids]
        return 
    
    def reset_zero_indices(self):
        self.l_stop_mask[0]=1
        self.r_stop_mask[0]=1
        self.__init_l_graph_zeros_indices()
        self.__init_r_graph_zeros_indices()
        self.l_likelihoods[0]=0
        self.r_likelihoods[0]=0
        self.nadd_likelihoods[0]=0
        self.nconn_likelihoods[0]=0
        self.nint_likelihoods[0]=0
        return 
    
    
    def sample_mols(self,MG,nmols_per_complex=10,savepath='./sample_mols',debug=True):
        sampled_mols=[]
        sampled_smis=[]
        sampled_valids=[]

        sample_set,loader=self.__init_sample_loader(MG,nmols_per_complex)
        sample_bar=enumerate(loader)
        self.record_pocket_and_reflig(MG,savepath)
        print (MG.p_natoms)
        for bid,Datas in sample_bar: 
            conf_step_id=0
            nadd_step_id=0
            rgen_step_id=0
            nconn_step_id=0
            nint_step_id=0
            
            self.sample_prepare()
            self.init_c_graphs(Datas)
            self.init_l_graphs(mode='all')

            # 0 index graph is always unused to avoid GGNN fails for empty graphs in the total batch.
            # thus it always needs to initialize the graph with only one node, its the same in ring generations.
            while torch.sum(self.l_stop_mask[1:])>0:
                self.init_nadd(mode='all')
                self.init_r_graphs(mode='all')
                self.init_nconn(mode='all')
                self.init_nint(mode='all')
                
                self.reset_zero_indices()
                all_ids=self.__to_device(torch.arange(FGP.batchsize).long())
                # initialize the 0 indexed graph with only one node
                # find unstoped complex graphs
                l_unstop_ids=torch.where(self.l_stop_mask==1)[0]
                c_feats,c_adjs_mat,c_coords,c_masks=self.select_c_graphs(l_unstop_ids,mask_mode='normal',output_mode='simple')
                # sample nadd actions for unstoped complex graphs
                print ('l_adjs',torch.sum(self.l_adjs_mat[:,:5,:5],dim=-1))
                l_add_ids,l_term_ids,nadd_likelihoods=self.model.nadd_step(c_feats,c_adjs_mat,c_coords,c_masks)

                
                self.l_stop_mask[l_unstop_ids[l_term_ids]]=0
                self.nadd_likelihoods[l_unstop_ids]=nadd_likelihoods
                # trans indices of nadd actions to real complex graph indices
                nadd_mol_ids=l_unstop_ids[l_add_ids[0]]
                nadd_type_ids=l_add_ids[1]
                # update single atom groups first
                self.r_stop_mask=self.l_stop_mask.clone().detach()    
                single_node_ids=torch.where(nadd_type_ids<len(FGP.atom_types))[0]            
                self.update_single_atom_r_graphs(nadd_mol_ids,nadd_type_ids,single_node_ids)
                self.r_stop_mask[nadd_mol_ids[single_node_ids]]=0 
                
                self.r_ftypes[nadd_mol_ids]=self.gid_to_gfeats[nadd_type_ids]
                self.update_r_maxatoms()
                
                #start fragment generation with rgen model
                self.record_nadd(path=f'{savepath}/{bid}')
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
                
                r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids=self.select_r_graphs(nconn_unstop_ids)
            
                l_atomnums=self.l_atomnums[nconn_unstop_ids]
                nconn_ids,nconn_likelihoods=\
                    self.model.nconn_step(c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_focused_ids,r_masks,l_atomnums)

                # select valid node connection actions
                nconn_invalid_ids=self.model.invalid_nconn_actions(nconn_ids,l_atomnums)
                
                nconn_valid_mask=self.nconn_stop_mask.clone().detach()
                nconn_valid_mask[nconn_unstop_ids[nconn_invalid_ids]]=0
                
                nconn_valid_ids=torch.nonzero(nconn_valid_mask[nconn_unstop_ids[nconn_ids[0]]]>0).view(-1)

                # select valid node connection actions
                batch,bond_to,bond_type,bond_from,likelihoods=self.select_valid_nconn_actions(nconn_ids,nconn_valid_ids,nconn_likelihoods)
                batch=nconn_unstop_ids[batch]
                
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
                print ('Nint stage'+'-'*80)
                self.nint_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                self.nint_stop_mask[updated_ids]=1
                
                
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
                print ('Conf stage'+'-'*80) 
                self.conf_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                self.conf_stop_mask[updated_ids]=1
                self.reset_zero_indices()
                self.conf_stop_mask[0]=1 
                coords_unstop_ids=torch.where(self.conf_stop_mask>0)[0]

                if len(coords_unstop_ids)==0:
                    break

                if True:
                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,\
                        p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_groups_atom_int_masks,p_atom_int_masks,c_pocket_labels,c_ligand_labels\
                            =self.select_c_graphs(coords_unstop_ids,mask_mode='graphs',output_mode='all')
                            
                    b,n,_,d=c_adjs_mat.shape
                    indices=torch.arange(n).to(self.device)
                    
                    #print ('c_adjs_mat diagonals',c_adjs_mat[:,indices,indices].shape,torch.where(torch.sum(c_adjs_mat[:,indices,indices],dim=-1)))
                    c_adjs_mat[0,FGP.max_patoms:,FGP.max_patoms:]=0

                    samples,samples_diff,samples_diff_before=self.model.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_masks,c_flexible_masks,c_pocket_labels,c_ligand_labels)
                    
                    self.update_l_graphs_with_conf_actions(coords_unstop_ids,samples)
                    
                    conf_step_id+=1
                    self.record_conf(path=f'{savepath}/{bid}')
                    print (torch.sum(self.l_adjs_mat[:,:5,:5],dim=-1),self.l_atoms)
                    print ('STOP!'*20)

            mols,smis,valid_ids=self.__l_graphs_to_mol(with_coords=True)
            sampled_mols+=mols
            sampled_smis+=smis
            sampled_valids+=valid_ids
            self.record_conf(path=f'{savepath}')
                
        return mols,smis,valid_ids
    
    def __init_sample_loader(self,MG,nmols_per_complex=10):

        nmols_per_complex=math.ceil(nmols_per_complex/(FGP.batchsize-1))*FGP.batchsize
        if self.local_rank is not None:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            sampler=torch.utils.data.distributed.DistributedSampler(sample_set)
            loader=DataLoader(sample_set,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers,sampler=sampler)
        else:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            loader=DataLoader(Train_Dataset,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers)

        return sample_set,loader
     
    def __node_feats_to_atom(self,node_feat):
        non_zero_idc=torch.nonzero(node_feat>0)
        atom_idx=non_zero_idc[0]
        atom_type=FGP.atom_types_for_feats[atom_idx]
        new_atom=Chem.Atom(atom_type)
        fc_idx=non_zero_idc[1]-len(FGP.atom_types_for_feats)
        formal_charge=FGP.formal_charge_types[fc_idx]
        new_atom.SetFormalCharge(formal_charge)
        return new_atom
                
    def __l_graphs_to_mol(self,with_coords=False):
        if with_coords: 
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:],self.l_coords[1:])
        else:
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:])
        return mols,smis,valid_ids

    def __r_graphs_to_mol(self):
        mols,smis,valid_ids=self.graph_to_mols(self.r_feats[1:],self.r_adjs_mat[1:],self.r_atomnums[1:])
        return mols,smis,valid_ids
    
    def graph_to_mols(self,feats,adjs_mat,atomnums,coords=None): 
        mols=[]
        smis=[]
        batchsize=feats.shape[0]
        graphsize=feats.shape[1]
        valid_ids=[0]*batchsize
        for i in range(batchsize):
            mol=Chem.RWMol()
            node_to_idx={}
            #print ('atomnums',atomnums[i])
            for j in range(atomnums[i]):
                atom=self.__node_feats_to_atom(feats[i,j])
                mol_idx=mol.AddAtom(atom)
                node_to_idx[j]=mol_idx
            #print ('node_to_idx',node_to_idx)
            edge_mask=self.__to_device(torch.triu(
                    torch.ones((graphsize, graphsize)),
                    diagonal=1
                    ).unsqueeze(-1)).long()
            edge_idc=torch.nonzero(edge_mask*adjs_mat[i])
            for idx1,idx2,bond_index in edge_idc:
                #    if idx1>=atomnums[i] or idx2>=atomnums[i]:
                #        print ('idx1,idx2,bond_index',i,idx1,idx2,bond_index,atomnums[i])
                try:
                    bond_type=FGP.bond_types[bond_index-1]
                    mol.AddBond(node_to_idx[idx1.item()],node_to_idx[idx2.item()],bond_type) 
                except:
                    pass
            try:
            #if True:
                mol=mol.GetMol()
                AllChem.Compute2DCoords(mol)
                #Chem.SanitizeMol(mol)
                smi=Chem.MolToSmiles(mol)
                if coords is not None:
                    mol=Change_mol_xyz(mol,coords[i].clone().detach().cpu().numpy()[:atomnums[i]])
                mols.append(mol)
                smis.append(smi)
                valid_ids[i]=1
                #print (mol,smi)
            except:
                mols.append(None)
                smis.append(None)  
            
        return mols,smis,valid_ids 

    def record_mol(self,mol,smi,savepath='./samples',savename='ligand'):
        os.system(f'mkdir -p {savepath}')
        if not os.path.exists(savepath):
            os.system(f"mkdir -p {savepath}")
            
        with open(f"{savepath}/{savename}.smi",'w') as f:
            f.write(smi)
            
        try:
            molsupp=Chem.SDWriter(f'{savepath}/{savename}.sdf')
            molsupp.write(mol)
            molsupp.close()
        except:
            MolToXYZ(mol,f'{savepath}/{savename}.xyz')
        
        if 'conf' in savename:
            MolToXYZ(mol,f'{savepath}/{savename}.xyz')
            
        try:
            Draw.MolToFile(mol,f"{savepath}/{savename}.png")
        except:
            pass
        return

    def record_pocket_and_reflig(self,MG,savepath='./samples'):
        MG_cp=copy.deepcopy(MG)
        Mass_center=MG_cp.get_pocket_mass_center()
        MG_cp.p_coords=MG_cp.p_coords-Mass_center
        MG_cp.l_coords=MG_cp.l_coords-Mass_center
        
        ref_ligmol=MG_cp.Trans_Ligand_to_Mol()
        ref_pocket=MG_cp.Trans_Pocket_to_Mol()

        os.system(f'mkdir -p {savepath}')
        try:
            molsupp=Chem.SDWriter(f'{savepath}/ref_lig.sdf')
            molsupp.write(ref_ligmol)
            molsupp.close()
        except:
            MolToXYZ(ref_ligmol,f'{savepath}/ref_lig.xyz')
    
            
        MolToXYZ(ref_pocket,f'{savepath}/ref_pocket.xyz')
        return 

    def record_nadd(self,path=f'./samples/'): 
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol()
        for i in range(len(l_mols)):
            savepath=f"{path}-{i+1}/{self.l_rounds[i+1]}"
            if not os.path.exists(savepath):
                os.system(f"mkdir -p {savepath}")
            with open(f"{savepath}/l_nadd-{self.l_rounds[i+1]}.rftype",'w') as gf:
                f=self.r_ftypes[i+1].long()
                if f[0]>0:
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}-HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'
                else:
                    if f[10]>1:
                        descriptor=f'R{f[0]}-C*-N*-O*-F*-P*-S*-Cl*-Br*-I*-Bes{f[10]}-AR*-'+\
                            f'HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'                   
                    else:
                        descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR*-'+\
                        f'HD*-HA*-Neg*-Pos*-Aro*-Hyd*-LHyd*'
                gf.write(descriptor+'\n')
            if l_mols[i] is not None:
                self.record_mol(l_mols[i],l_smis[i],savepath=savepath,savename=f'l_nadd-{self.l_rounds[i+1]}')
        return
        
    def record_rgen(self,path=f'./samples',gen_process_path=''):
        r_mols,r_smis,r_valid_ids=self.__r_graphs_to_mol()
        for i in range(len(r_mols)):
            if r_mols[i] is not None:
                savepath=f"{path}-{i+1}/{self.l_rounds[i+1]}"
                if gen_process_path !='':
                    savepath=savepath+'/'+gen_process_path
                    self.record_mol(r_mols[i],r_smis[i],savepath=savepath,savename=f'r_gen-{self.r_rounds[i+1]}')
                else:
                    self.record_mol(r_mols[i],r_smis[i],savepath=savepath,savename=f'r_gen-{self.l_rounds[i+1]}')
        return
        
    def record_nconn(self,path=f'./samples'):
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol()
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                self.record_mol(l_mols[i],l_smis[i],savepath=savepath,savename=f'l_nconn-{self.l_rounds[i+1]}')
        return
        
    def record_conf(self,path=f'./samples'):
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol(with_coords=True)
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                self.record_mol(l_mols[i],l_smis[i],savepath=savepath,savename=f'l_conf-{self.l_rounds[i+1]}')
        return
         
    def __to_device(self,tensor):
        if self.local_rank is not None:
            return tensor.cuda(self.local_rank)
        else:
            return tensor.cuda()

    def __to_model_device(self,model):
        if self.local_rank is not None:
            model.cuda(self.local_rank)
            model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model=torch.nn.parallel.DistributedDataParallel(
                                                        model,
                                                        device_ids=[self.local_rank],
                                                        output_device=self.local_rank,
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=False
                                                        )
        else:
            model.cuda()
        return model
    
    def __init_l_graph_zeros_indices(self):
        dummy_af=torch.Tensor(onek_encoding_unk(6,FGP.atom_types_for_feats)+\
                onek_encoding_unk(0,FGP.formal_charge_types))
        self.l_feats[0]=0
        self.l_adjs_mat[0]=0
        self.l_coords[0]=0
        self.l_atoms[0]=0
        self.l_masks[0]=0
        self.l_atomnums[0]=0
        
        self.l_feats[0,0]=self.__to_device(dummy_af).float()
        self.l_feats[0,1]=self.__to_device(dummy_af).float()
        self.l_adjs_mat[0,0,1,0]=1
        self.l_adjs_mat[0,1,0,0]=1
        self.l_atomnums[0]=2
        self.l_atoms[0,0]=6
        self.l_atoms[0,1]=6
        self.l_masks[0,0]=1
        self.l_masks[0,1]=1
        self.l_coords[0]=torch.randn_like(self.l_coords[0])
        return 
     
    def __init_r_graph_zeros_indices(self):
        dummy_af=torch.Tensor(onek_encoding_unk(6,FGP.atom_types_for_feats)+onek_encoding_unk(0,FGP.formal_charge_types)).float()
        self.r_feats[0]=0
        self.r_feats[0,0]=self.__to_device(dummy_af)
        self.r_feats[0,1]=self.__to_device(dummy_af)
        
        self.r_adjs_mat[0]=0
        self.r_adjs_mat[0,0,1,0]=1
        self.r_adjs_mat[0,1,0,0]=1
        
        self.r_atomnums[0]=0
        self.r_atomnums[0]=2
        
        self.r_atoms[0]=0
        self.r_atoms[0,0]=6
        self.r_atoms[0,1]=6
        
        self.r_masks[0]=0
        self.r_masks[0,0]=1
        self.r_masks[0,1]=1
        
        self.r_ftypes[0]=0
        self.r_maxatoms[0]=2
        return
              
    def __nadd_constrains_to_mask(self,nadd_ct):
        n_types=len(FGP.group_index_type_dict.keys())
        mask=self.__to_device(torch.ones(ntypes+1).long())
        for fid, group_descriptor in FGP.group_index_type_dict.items():
            Rnum,Besnum,ARnum,element_num_dict,pharm_flag_dict=group_descriptor_to_feats(group_descriptor)
            if Rnum < nadd_ct.ringnum_range[0] or Rnum > nadd_ct.ringnum_range[1]:
                mask[fid]=0
                
            for attype in enumerate(["C","N","O","F","P","S","Cl","Br","I"]):
                if element_num_dict[attype]!='*':
                    if int(element_num_dict[attype]) < nadd_ct.atnum_range_dict[attype][0] or int(element_num_dict[attype]) > nadd_ct.atnum_range_dict[attype][1]:
                        mask[fid]=0
                else:
                    mask[fid]=0
                    
            if int(Besnum) < nadd_ct.branchnum_range[0] or int(Besnum) > nadd_ct.branchnum_range[1]:
                mask[fid]=0
                
            if ARnum !='*':
                if int(ARnum) <  nadd_ct.aromnum_range[0] or int(ARnum) > nadd_ct.aromnum_range[1]:
                    mask[fid]=0
            else:
                if nadd_ct.aromnum_range[0]>0:
                    mask[fid]=0
                
            if pharm_flag_dict[pharmtype]!='*':
                for pharmtype in enumerate(["Donor","Acceptor","Neg","Pos","Aro","Hyd","LHyd"]):
                    if nadd_ct.pharm_types[pharmtype]>0:
                        if pharm_flag_dict[pharmtype]==0:
                            mask[fid]=0
                    elif nadd_ct.pharm_type[pharmtype]<0:
                        if pharm_flag_dict[pharmtype]>0:
                            mask[fid]=0
            else:
                pharm_labels=[num for num in nadd_ct.pharm_types if num>0]
                if np.sum(pharm_labels)==1:
                    if nadd_ct.pharm_types['Donor']>0:
                        for key in ['C','F','P','Cl','Br','I']:
                            if element_num_dict[key]>0:
                                mask[fid]=0
                    if nadd_ct.pharm_types['Acceptor']>0:
                        for key in ['C','N','F','P','Cl','Br','I']:
                            if element_num_dict[key]>0:
                                mask[fid]=0
                    for pharm in ['Neg','Pos','Aro','Hyd','LHyd']:
                        if nadd_ct.pharm_types[pharm]>0:
                            for key in ['C','N','O','F','P','S','Cl','Br','I']:
                                if element_num_dict[key]>0:
                                    mask[fid]=0
                else:
                    mask[fid]=0
        return mask

                
            