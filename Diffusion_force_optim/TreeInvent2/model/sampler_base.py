import torch
from .consistency import * 
from .Equiformerv2 import *

import pickle,os,tempfile, shutil, zipfile, time, math, tqdm 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from ..utils import group_descriptor_to_feats

from ..graphs.datasets import *
from tqdm import tqdm 
from torch import distributed as dist
from .modules import * 
from .TImodel import * 
from ..utils.utils_rdkit import mol_with_atom_index_and_formal_charge
#import cairosvg

class TreeInvent_Sampler_Base:
    def __init__(self,modelname,local_rank=None,**kwargs):
        self.local_rank=local_rank
        self.device=FGP.device
        self.jobstr='nadd+rgen+nconn'
        if not FGP.only_2d:
            self.jobstr+='+coords'
        #self.model=TreeInvent_Model(modelname=modelname,local_rank=local_rank,jobs=self.jobstr,**kwargs)
        self.model=None
        return

    def init_l_graphs(self,ids=None,mode='all'):  
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        
        if mode=='all':             
            self.l_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types))).float()
            self.l_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,FGP.max_latoms,len(FGP.bond_types)+1)).float()
            self.l_adjs=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,FGP.max_latoms)).long()
            self.l_coords=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,3).float())
            
            self.l_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_gmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_dmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            
            self.l_rounds=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_groups=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).long())
            self.l_groups_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).float())
            self.l_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
            
            self.l_fix_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_flexible_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_pts=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,2))
            
            self.l_bond_orders=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,FGP.max_latoms).float())
            self.l_max_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).float())
            self.l_total_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).float())
            self.l_allowed_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,4).float())
            self.l_saturation_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_possible_conn_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,4).long())
            self.l_mols=[None]*FGP.batchsize

        else:

            self.l_feats[ids]=0
            self.l_adjs_mat[ids]=0
            self.l_adjs[ids]=0
            self.l_coords[ids]=0
            
            self.l_atomnums[ids]=0
            self.l_atoms[ids]=0
            self.l_masks[ids]=0
            self.l_gmasks[ids]=0
            self.l_dmasks[ids]=0
            
            self.l_rounds[ids]=0
            self.l_groups[ids]=0
            self.l_groups_masks[ids]=0
            self.l_likelihoods[ids]=0
            
            self.l_fix_masks[ids]=0
            self.l_flexible_masks[ids]=0
            self.l_pts[ids]=0
            self.l_bond_orders[ids]=0
            self.l_max_bonds[ids]=0
            self.l_total_bonds[ids]=0
            self.l_allowed_bonds[ids]=0
            self.l_saturation_masks[ids]=1
            self.l_possible_conn_masks[ids]=1
            self.l_mols[0]=None

        return 
      
    def init_r_graphs(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode=='all':
            self.r_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types)).float())
            self.r_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,FGP.max_lgroup_size,len(FGP.bond_types)+1).float())
            self.r_adjs=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,FGP.max_lgroup_size).long())
            self.r_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.r_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_focused_ids=self.__to_device(torch.zeros(FGP.batchsize,1).long())
            self.r_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_rounds=self.__to_device(torch.zeros(FGP.batchsize)).long()
            self.r_ftypes=self.__to_device(torch.zeros(FGP.batchsize,FGP.n_group_feats)).float()
            self.r_likelihoods=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size*4).float())
            self.r_maxatoms=self.__to_device(torch.zeros(FGP.batchsize).long())

            self.r_bond_orders=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,FGP.max_latoms).float())
            self.r_max_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).float())
            self.r_total_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).float())
            self.r_allowed_bonds=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,4).float())
            self.r_saturation_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_possible_conn_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,4).long())
            self.r_mols=[None]*FGP.batchsize
        else:
            self.r_feats[ids]=0
            self.r_adjs_mat[ids]=0
            self.r_adjs[ids]=0
            self.r_atomnums[ids]=0
            self.r_atoms[ids]=0
            self.r_focused_ids[ids]=0
            self.r_masks[ids]=0
            self.r_rounds[ids]=0
            
            self.r_likelihoods[ids]=0
            if mode!='failed':
                self.r_ftypes[ids]=0
                self.r_maxatoms[ids]=0
            self.r_bond_orders[ids]=0
            self.r_max_bonds[ids]=0
            self.r_total_bonds[ids]=0
            self.r_allowed_bonds[ids]=0
            self.r_saturation_masks[ids]=1
            self.r_possible_conn_masks[ids]=1
            self.r_mols[0]=None
        return 
        
    def init_nconn(self,ids=None,mode='all'):
        if mode !='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode== 'all':
            self.nconn_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nconn_likelihoods[ids]=0
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
    
    def select_l_mols(self,ids):
        mols=[]
        for id in ids:
            mols.append(self.l_mols[id])
        return mols
    
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
    
    def select_r_mols(self,ids):
        mols=[]
        for id in ids:
            mols.append(self.r_mols[id])
        return mols
        
    def sample_prepare(self):
        self.l_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.r_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
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
        
        self.r_adjs_mat[batch_non_empty,bond_from_non_empty,bond_to_non_empty,bond_type_non_empty+1]=1
        
        self.r_adjs[batch_non_empty,bond_from_non_empty,bond_to_non_empty]=bond_type_non_empty+1
        
        self.r_adjs_mat[batch_non_empty,bond_to_non_empty,bond_from_non_empty,bond_type_non_empty+1]=1
        
        self.r_adjs[batch_non_empty,bond_to_non_empty,bond_from_non_empty]=bond_type_non_empty+1
        
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
        self.r_adjs[batch,bond_from,bond_to]=bond_type+1
        self.r_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.r_adjs[batch,bond_to,bond_from]=bond_type+1
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
        print ('update_nconn',bond_type.type(),self.l_adjs.type())
        self.l_adjs[batch,bond_from,bond_to]=bond_type+1
        self.l_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.l_adjs[batch,bond_to,bond_from]=bond_type+1
        self.nconn_likelihoods[batch]=likelihoods
        return 
    
    def update_l_graphs_with_rgraphs(self,ids):
        for i in ids:
            l_natoms=self.l_atomnums[i]
            r_natoms=self.r_atomnums[i]
            
            self.l_feats[i,l_natoms:l_natoms+r_natoms]=self.r_feats[i,:r_natoms]
            self.l_adjs_mat[i,l_natoms:l_natoms+r_natoms,l_natoms:l_natoms+r_natoms]=self.r_adjs_mat[i,:r_natoms,:r_natoms]
            self.l_adjs[i,l_natoms:l_natoms+r_natoms,l_natoms:l_natoms+r_natoms]=self.r_adjs[i,:r_natoms,:r_natoms]
            
            self.l_atoms[i,l_natoms:l_natoms+r_natoms]=self.r_atoms[i,:r_natoms]
            self.l_groups[i,self.l_rounds[i],:r_natoms]=l_natoms+self.__to_device(torch.arange(r_natoms).long())
            self.l_groups_masks[i,self.l_rounds[i],:r_natoms]=1
            self.l_masks[i,:l_natoms+r_natoms]=1
            self.l_gmasks[i,l_natoms:l_natoms+r_natoms]=1
            self.l_atomnums[i]+=r_natoms
            self.l_pts[i,self.l_rounds[i],1]=self.l_atomnums[i]
            self.l_rounds[i]+=1
            self.l_pts[i,self.l_rounds[i],0]=self.l_atomnums[i]
             
            self.l_fix_masks[i]=0
            self.l_flexible_masks[i]=0
            if FGP.l_mode=='fix-previous':
                self.l_fix_masks[i,:l_natoms]=1
                self.l_fix_masks[i,l_natoms:self.l_atomnums[i]]=1
            else:    
                self.l_fix_masks[i,:self.l_atomnums[i]]=0
                self.l_flexible_masks[i,:self.l_atomnums[i]]=1
        return 

    def update_l_graphs_with_conf_actions(self,ids,c_coords):
        self.l_coords[ids]=c_coords[:,FGP.max_patoms:]
        self.l_dmasks[ids]=self.l_gmasks[ids]
        return 
    
    def reset_zero_indices(self):
        self.l_stop_mask[0]=1
        self.r_stop_mask[0]=1
        self.init_l_graph_zeros_indices()
        self.init_r_graph_zeros_indices()
        
        self.l_likelihoods[0]=0
        self.r_likelihoods[0]=0
        self.nadd_likelihoods[0]=0
        self.nconn_likelihoods[0]=0
        return 
    
    def update_l_saturation_masks(self):
        self.l_bond_orders=torch.where(self.l_adjs==4,torch.ones_like(self.l_adjs)*1.5,self.l_adjs)
        self.l_total_bonds=torch.sum(self.l_bond_orders,dim=-1)
        self.l_max_bonds=torch.zeros_like(self.l_atoms).float()
        for key in [6,7,8,9,15,16,17,35,53]:
            self.l_max_bonds+=torch.where(self.l_atoms==key,1,0)*FGP.atom_max_bonds_table[key]

        self.l_saturation_masks=torch.where(self.l_total_bonds<=self.l_max_bonds-1,1,0)*self.l_masks
        print ('l_saturation_masks',self.l_saturation_masks)
        l_possible_conn_masks=self.__to_device(torch.zeros((FGP.batchsize,*FGP.leaf_conn_dim)).float())
        l_possible_conn_masks[:,:,0]=1
        l_possible_conn_masks[:,:,1]=2
        l_possible_conn_masks[:,:,2]=3
        l_possible_conn_masks[:,:,3]=4 # 1.5 when aromatic bond are allowed
        self.l_allowed_bonds=(self.l_max_bonds-self.l_total_bonds).unsqueeze(-1).tile(1,1,4)*self.l_saturation_masks.unsqueeze(-1)
        print (self.l_allowed_bonds.shape)
        
        self.l_possible_conn_masks=torch.where(l_possible_conn_masks<=self.l_allowed_bonds,1,0)*self.l_saturation_masks.unsqueeze(-1)
        for i in range(FGP.batchsize):
            if self.l_atomnums[i]==0:
                self.l_possible_conn_masks[i]=1
        return 
    
    def update_r_saturation_masks(self):
        self.r_bond_orders=torch.where(self.r_adjs==4,torch.ones_like(self.r_adjs)*1.5,self.r_adjs)
        self.r_total_bonds=torch.sum(self.r_bond_orders,dim=-1)
        self.r_max_bonds=torch.zeros_like(self.r_atoms).float()
        for key in [6,7,8,9,15,16,17,35,53]:
            r_max_bonds+=torch.where(self.r_atoms==key,1,0)*FGP.atom_max_bonds_table[key]
        self.r_saturation_masks=torch.where(self.r_total_bonds<=self.r_max_bonds-1,1,0)*self.r_masks
        print ('r_saturation_masks',self.r_saturation_masks)
        r_possiblle_conn_masks=self.__to_device(torch.zeros((FGP.batchsize,FGP.max_lgroup_size,4)).float())
        r_possiblle_conn_masks[:,:,0]=1
        r_possiblle_conn_masks[:,:,1]=2
        r_possiblle_conn_masks[:,:,2]=3
        r_possiblle_conn_masks[:,:,3]=4
        self.r_allowed_bonds=(self.r_max_bonds-self.r_total_bonds).unsqueeze(-1).tile(1,1,4)*self.r_saturation_masks.unsqueeze(-1)
        print (self.r_allowed_bonds.shape)
        self.r_possible_conn_masks=torch.where(r_possiblle_conn_masks<=self.r_allowed_bonds,1,0)*self.r_saturation_masks.unsqueeze(-1)
        for i in range(FGP.batchsize):
            if self.r_atomnums[i]==0:
                self.r_possible_conn_masks[i]=1
        return
            
    def __node_feats_to_atom(self,node_feat):
        non_zero_idc=torch.nonzero(node_feat>0)
        atom_idx=non_zero_idc[0]
        atom_type=FGP.atom_types_for_feats[atom_idx]
        new_atom=Chem.Atom(atom_type)
        fc_idx=non_zero_idc[1]-len(FGP.atom_types_for_feats)
        formal_charge=FGP.formal_charge_types[fc_idx]
        new_atom.SetFormalCharge(formal_charge)
        return new_atom
                
    def l_graphs_to_mol(self,with_coords=False):
        if with_coords: 
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:],self.l_coords[1:])
        else:
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:])
        return mols,smis,valid_ids

    def r_graphs_to_mol(self):
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
                
                Chem.SanitizeMol(mol)
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
        
        mol=mol_with_atom_index_and_formal_charge(mol)
        AllChem.Compute2DCoords(mol)
        try:
            Draw.MolToFile(mol,f"{savepath}/{savename}.svg")
            input_svg = f"{savepath}/{savename}.svg"
            output_png = f"{savepath}/{savename}.png"
            dpi = 300
            output_width = 800  # 
            output_height = 800  # 
            cairosvg.svg2png(
                url=input_svg, write_to=output_png,
                dpi=dpi, output_width=output_width, 
                output_height=output_height)
            
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
        return ref_ligmol,ref_pocket 

    def record_nadd(self,path=f'./samples/'): 
        l_mols,l_smis,l_valid_ids=self.l_graphs_to_mol()
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
        r_mols,r_smis,r_valid_ids=self.r_graphs_to_mol()
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
        l_mols,l_smis,l_valid_ids=self.l_graphs_to_mol()
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                self.record_mol(l_mols[i],l_smis[i],savepath=savepath,savename=f'l_nconn-{self.l_rounds[i+1]}')
        return
        
    def record_conf(self,path=f'./samples',debug=True):
        l_mols,l_smis,l_valid_ids=self.l_graphs_to_mol(with_coords=True)
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                if debug:
                    savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                else:
                    savepath=f"{path}-{i+1}"
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
    
    def init_l_graph_zeros_indices(self):
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
        self.l_mols[0]=Chem.MolFromSmiles('CC')
        return 
     
    def init_r_graph_zeros_indices(self):
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
        self.r_mols[0]=None
        return
              
    def __nadd_constrains_to_mask(self,nadd_ct):
        
        n_types=len(FGP.group_index_type_dict.keys())
        nadd_mask=self.__to_device(torch.ones(n_types+1).long())
        term_mask=self.__to_device(torch.ones(2).long())
        if nadd_ct.force_step:
            term_mask[1]=0
            nadd_mask[-1]=0
        else: 
            for fid, group_descriptor in FGP.group_index_type_dict.items():
                Rnum,Besnum,ARnum,element_num_dict,pharm_flag_dict=group_descriptor_to_feats(group_descriptor)
                #print (element_num_dict,pharm_flag_dict)
                if int(Rnum) < nadd_ct.ringnum_range[0] or int(Rnum) > nadd_ct.ringnum_range[1]:
                    nadd_mask[fid]=0
                    
                for aid,attype in enumerate(["C","N","O","F","P","S","Cl","Br","I"]):
                    if element_num_dict[attype]!='*':
                        if int(element_num_dict[attype]) < nadd_ct.atnum_range_dict[attype][0] or int(element_num_dict[attype]) > nadd_ct.atnum_range_dict[attype][1]:
                            nadd_mask[fid]=0
                    else:
                        nadd_mask[fid]=0
                        
                if int(Besnum) < nadd_ct.branchnum_range[0] or int(Besnum) > nadd_ct.branchnum_range[1]:
                    nadd_mask[fid]=0
                    
                if ARnum !='*':
                    if int(ARnum) <  nadd_ct.ar_ringnum_range[0] or int(ARnum) > nadd_ct.ar_ringnum_range[1]:
                        nadd_mask[fid]=0
                else:
                    if nadd_ct.ar_ringnum_range[0]>0:
                        nadd_mask[fid]=0
                    
                if pharm_flag_dict['HD']!='*':
                    for pid,pharmtype in enumerate(pharm_flag_dict.keys()):
                        if nadd_ct.pharm_types[pharmtype]>0:
                            if int(pharm_flag_dict[pharmtype])==0:
                                nadd_mask[fid]=0
                        elif nadd_ct.pharm_types[pharmtype]<0:
                            if int(pharm_flag_dict[pharmtype])>0:
                                nadd_mask[fid]=0
                else:
                    pharm_labels=[int(nadd_ct.pharm_types[key]) for key in  nadd_ct.pharm_types.keys() if int(nadd_ct.pharm_types[key])>0]
                    if np.sum(pharm_labels)==1:
                        if nadd_ct.pharm_types['HD']>0:
                            for key in ['C','F','P','Cl','Br','I']:
                                if int(element_num_dict[key])>0:
                                    nadd_mask[fid]=0
                        if nadd_ct.pharm_types['HA']>0:
                            for key in ['C','N','F','P','Cl','Br','I']:
                                if int(element_num_dict[key])>0:
                                    nadd_mask[fid]=0
                        for pharm in ['Neg','Pos','Aro','Hyd','LHyd']:
                            if nadd_ct.pharm_types[pharm]>0:
                                for key in ['C','N','O','F','P','S','Cl','Br','I']:
                                    if int(element_num_dict[key])>0:
                                        nadd_mask[fid]=0
                    else:
                        nadd_mask[fid]=0
        return nadd_mask,term_mask
    
    def __nconn_constrains_to_mask(self,nconn_ct,pts,possible_conn_masks,atomics):
        mask=self.__to_device(torch.zeros(FGP.leaf_conn_dim)).long()
        if len(nconn_ct.constrain_connect_node_id)>0:
            for kid,k in enumerate(nconn_ct.constrain_connect_node_id):
                pt0=pts[k,0]
                pt1=pts[k,1]
                if len(nconn_ct.constrain_connect_atom_id[kid])>0:
                    for nid,n in enumerate(range(pt0,pt1)):
                        if nid in nconn_ct.constrain_connect_atom_id_in_node[kid]:
                            if atomics[n] in nconn_ct.constrain_connect_atomic_type:
                                mask[n,nconn_ct.constrain_connect_bond_type]=1
                else:
                    for nid,n in enumerate(range(pt0,pt1)):
                        if atomics[n] in nconn_ct.constrain_connect_atomic_type:
                            mask[n,nconn_ct.constrain_connect_bond_type]=1
                if len(nconn_ct.constrain_saturation_atom_id_in_node[kid])>0:
                    for nid,n in enumerate(range(pt0,pt1)):
                        if nid in nconn_ct.constrain_saturation_atom_id_in_node[kid]:
                            mask[n]=0
        else:
            mask=self.__to_device(torch.ones(FGP.leaf_conn_dim)).long()
        #print (mask.shape)
        mask=(mask*possible_conn_masks).view(-1)
        return mask
    
    def gen_topology_constrains(self):
        self.term_masks=[]
        self.nadd_masks=[]
        self.nconn_masks=[]
        for i in range(FGP.batchsize):
            step_for_graph=self.l_rounds[i].detach().cpu().numpy()
            nadd_ct=TP.constrain_step_dict[str(step_for_graph)]["node add"]
            nconn_ct=TP.constrain_step_dict[str(step_for_graph)]["node conn"]
            nadd_mask,term_mask=self.__nadd_constrains_to_mask(nadd_ct)
            nconn_mask=self.__nconn_constrains_to_mask(nconn_ct,self.l_pts[i],self.l_possible_conn_masks[i],self.l_atoms[i])
            self.term_masks.append(term_mask)
            self.nadd_masks.append(nadd_mask)
            self.nconn_masks.append(nconn_mask)
        self.term_masks=self.__to_device(torch.stack(self.term_masks)).long()
        self.nadd_masks=self.__to_device(torch.stack(self.nadd_masks)).long()
        self.nconn_masks=self.__to_device(torch.stack(self.nconn_masks)).long()
        return 
    
    def select_l_term_masks(self,ids):
        return self.term_masks[ids]
    
    def select_l_nadd_masks(self,ids):
        return self.nadd_masks[ids]
    
    def select_l_nconn_masks(self,ids):
        return self.nconn_masks[ids]

    def __to_numpy(self,tensor):
        return tensor.cpu().detach().numpy()

    def __save_l_states(self,idx,savepath='l_states.pkl'):
        l_feats=self.__to_numpy(self.l_feats[idx])
        l_adjs_mat=self.__to_numpy(self.l_adjs_mat[idx])
        l_adjs=self.__to_numpy(self.l_adjs[idx])
        l_coords=self.__to_numpy(self.l_coords[idx])
        
        l_atomnums=self.__to_numpy(self.l_atomnums[idx])
        l_atoms=self.__to_numpy(self.l_atoms[idx])
        l_masks=self.__to_numpy(self.l_masks[idx])
        l_gmasks=self.__to_numpy(self.l_gmasks[idx])
        l_dmasks=self.__to_numpy(self.l_dmasks[idx])
        
        l_rounds=self.__to_numpy(self.l_rounds[idx])
        l_groups=self.__to_numpy(self.l_groups[idx])
        l_groups_masks=self.__to_numpy(self.l_groups_masks[idx])
        l_likelihoods=self.__to_numpy(self.l_likelihoods[idx])
        
        l_pts=self.__to_numpy(self.l_pts[idx])
        l_fix_masks=self.__to_numpy(self.l_fix_masks[idx])
        l_flexible_masks=self.__to_numpy(self.l_flexible_masks[idx])
        l_stop_mask=self.__to_numpy(self.l_stop_mask[idx])
        
        l_bond_orders=self.__to_numpy(self.l_bond_orders[idx])
        l_max_bonds=self.__to_numpy(self.l_max_bonds[idx])
        l_total_bonds=self.__to_numpy(self.l_total_bonds[idx])
        l_allowed_bonds=self.__to_numpy(self.l_allowed_bonds[idx])
        l_stauration_masks=self.__to_numpy(self.l_saturation_masks[idx])
        l_possible_conn_masks=self.__to_numpy(self.l_possible_conn_masks[idx])
        
        States_Dict={
            "L_Feats":l_feats,"L_Adjs_mat":l_adjs_mat,"L_Adjs":l_adjs,"L_Coords":l_coords,
            "L_Atomnums":l_atomnums,"L_Atoms":l_atoms,"L_Masks":l_masks,"L_GMasks":l_gmasks,"L_DMasks":l_dmasks,
            "L_Rounds":l_rounds,"L_Groups":l_groups,"L_Groups_Masks":l_groups_masks,"L_Likelihoods":l_likelihoods,
            "L_Pts":l_pts,"L_Fix_Masks":l_fix_masks,"L_Flexible_Masks":l_flexible_masks,"L_Stop_Mask":l_stop_mask,
            "L_Bond_Orders":l_bond_orders,"L_Max_Bonds":l_max_bonds,"L_Total_Bonds":l_total_bonds,
            "L_Allowed_Bonds":l_allowed_bonds,"L_Saturation_Masks":l_stauration_masks,
            "L_Possible_Conn_Masks":l_possible_conn_masks
        }
        
        with open(savepath,'wb') as f:
            pickle.dump(States_Dict,f)
        
        return
    
    def __load_l_states(self,idx,loadpath='l_states.pkl'):
        with open(loadpath,'rb') as f:
            States_Dict=pickle.load(f)
        self.l_feats[idx]=self.__to_device(torch.Tensor(States_Dict["L_Feats"]))
        self.l_adjs_mat[idx]=self.__to_device(torch.Tensor(States_Dict["L_Adjs_mat"]))
        self.l_adjs[idx]=self.__to_device(torch.Tensor(States_Dict["L_Adjs"]))
        self.l_coords[idx]=self.__to_device(torch.Tensor(States_Dict["L_Coords"]))
        
        self.l_atomnums[idx]=self.__to_device(torch.Tensor(States_Dict["L_Atomnums"]))
        self.l_atoms[idx]=self.__to_device(torch.Tensor(States_Dict["L_Atoms"]))
        self.l_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Masks"]))
        self.l_gmasks[idx]=self.__to_device(torch.Tensor(States_Dict["L_GMasks"]))
        self.l_dmasks[idx]=self.__to_device(torch.Tensor(States_Dict["L_DMasks"]))
        
        self.l_rounds[idx]=self.__to_device(torch.Tensor(States_Dict["L_Rounds"]))
        self.l_groups[idx]=self.__to_device(torch.Tensor(States_Dict["L_Groups"]))
        self.l_groups_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Groups_Masks"]))
        self.l_likelihoods[idx]=self.__to_device(torch.Tensor(States_Dict["L_Likelihoods"]))
        
        self.l_pts[idx]=self.__to_device(torch.Tensor(States_Dict["L_Pts"]))
        self.l_fix_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Fix_Masks"]))
        self.l_flexible_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Flexible_Masks"]))
        self.l_stop_mask[idx]=self.__to_device(torch.Tensor(States_Dict["L_Stop_Mask"]))
        
        self.l_bond_orders[idx]=self.__to_device(torch.Tensor(States_Dict["L_Bond_Orders"]))
        self.l_max_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["L_Max_Bonds"]))
        self.l_total_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["L_Total_Bonds"]))
        self.l_allowed_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["L_Allowed_Bonds"]))
        self.l_saturation_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Saturation_Masks"]))
        self.l_possible_conn_masks[idx]=self.__to_device(torch.Tensor(States_Dict["L_Possible_Conn_Masks"]))
        return
    
    def __save_r_states(self,idx,savepath='r_stats.pkl'):
        r_feats=self.__to_numpy(self.r_feats[idx])
        r_adjs_mat=self.__to_numpy(self.r_adjs_mat[idx])
        r_adjs=self.__to_numpy(self.r_adjs[idx])

        r_atomnums=self.__to_numpy(self.r_atomnums[idx])
        r_atoms=self.__to_numpy(self.r_atoms[idx])
        r_focused_ids=self.__to_numpy(self.r_focused_ids[idx])
        r_masks=self.__to_numpy(self.r_masks[idx])
        r_rounds=self.__to_numpy(self.r_rounds[idx])
        
        r_ftypes=self.__to_numpy(self.r_ftypes[idx])
        r_maxatoms=self.__to_numpy(self.r_maxatoms[idx])
        r_likelihoods=self.__to_numpy(self.r_likelihoods[idx])
        r_bond_orders=self.__to_numpy(self.r_bond_orders[idx])
        r_max_bonds=self.__to_numpy(self.r_max_bonds[idx])
        r_total_bonds=self.__to_numpy(self.r_total_bonds[idx])
        r_allowed_bonds=self.__to_numpy(self.r_allowed_bonds[idx])
        r_saturation_masks=self.__to_numpy(self.r_saturation_masks[idx])
        r_possible_conn_masks=self.__to_numpy(self.r_possible_conn_masks[idx])
        
        r_stop_mask=self.__to_numpy(self.r_stop_mask[idx])
        States_Dict={
            "R_Feats":r_feats,"R_Adjs_mat":r_adjs_mat,"R_Adjs":r_adjs,
            "R_Atomnums":r_atomnums,"R_Atoms":r_atoms,"R_Focused_ids":r_focused_ids,"R_Masks":r_masks,"R_Rounds":r_rounds,
            "R_Ftypes":r_ftypes,"R_Maxatoms":r_maxatoms,"R_Likelihoods":r_likelihoods,
            "R_Bond_Orders":r_bond_orders,"R_Max_Bonds":r_max_bonds,"R_Total_Bonds":r_total_bonds,
            "R_Allowed_Bonds":r_allowed_bonds,"R_Saturation_Masks":r_saturation_masks,"R_Possible_Conn_Masks":r_possible_conn_masks,
            "R_Stop_Mask":r_stop_mask
        }
        with open(savepath,'wb') as f:
            pickle.dump(States_Dict,f)
        return
    
    def __load_r_states(self,idx,loadpath='r_states.pkl'):
        with open(loadpath,'rb') as f:
            States_Dict=pickle.load(f)
        self.r_feats[idx]=self.__to_device(torch.Tensor(States_Dict["R_Feats"]))
        self.r_adjs_mat[idx]=self.__to_device(torch.Tensor(States_Dict["R_Adjs_mat"]))
        self.r_adjs[idx]=self.__to_device(torch.Tensor(States_Dict["R_Adjs"]))
        
        self.r_atomnums[idx]=self.__to_device(torch.Tensor(States_Dict["R_Atomnums"]))
        self.r_atoms[idx]=self.__to_device(torch.Tensor(States_Dict["R_Atoms"]))
        self.r_masks[idx]=self.__to_device(torch.Tensor(States_Dict["R_Masks"]))
        self.r_rounds[idx]=self.__to_device(torch.Tensor(States_Dict["R_Rounds"]))
        
        self.r_ftypes[idx]=self.__to_device(torch.Tensor(States_Dict["R_Ftypes"]))
        self.r_maxatoms[idx]=self.__to_device(torch.Tensor(States_Dict["R_Maxatoms"]))
        self.r_likelihoods[idx]=self.__to_device(torch.Tensor(States_Dict["R_Likelihoods"]))
        self.r_bond_orders[idx]=self.__to_device(torch.Tensor(States_Dict["R_Bond_Orders"]))
        self.r_max_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["R_Max_Bonds"]))
        self.r_total_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["R_Total_Bonds"]))
        self.r_allowed_bonds[idx]=self.__to_device(torch.Tensor(States_Dict["R_Allowed_Bonds"]))
        self.r_saturation_masks[idx]=self.__to_device(torch.Tensor(States_Dict["R_Saturation_Masks"]))
        self.r_possible_conn_masks[idx]=self.__to_device(torch.Tensor(States_Dict["R_Possible_Conn_Masks"]))
        self.r_stop_mask[idx]=self.__to_device(torch.Tensor(States_Dict["R_Stop_Mask"]))
        return
    

    def __save_sampler_states(self,idx,savepath,savename='sampler_states'):
        if not os.path.exists(savepath):
            os.system(f"mkdir -p {savepath}")
            
        self.__save_l_states(idx,savepath=f'{savepath}/{savename}_l_states.pkl')
        self.__save_r_states(idx,savepath=f'{savepath}/{savename}_{self.r_rounds[idx]}_r_states.pkl')
        
        return
    
    def record_sampler_states(self,savepath='./l_states',savename='sampler_states'):
        for i in range(1,FGP.batchsize):
            self.__save_sampler_states(i,savepath=f'{savepath}-{i}/{self.l_rounds[i]}',savename=savename)
            
        return
    def record_rg_states(self,savepath='./rg_states',savename='rg_states'):
        for i in range(1,FGP.batchsize):
            self.__save_r_states(i,savepath=f'{savepath}-{i}/{self.l_rounds[i]}/{savename}_{self.r_rounds[i]}_r_states.pkl')
        return
    
    def record_nadd_actions(self,nadd_add_ids,nadd_type_ids,term_ids=None,savepath='./l_states'):
        for idx,i in enumerate(nadd_add_ids):
            if i!=0:
                nadd_action={"NADD_ADD_IDS":i,"NADD_TYPE_IDS":nadd_type_ids[idx],"NADD_MASKS":self.nadd_masks[i]}
                with open(f'{savepath}-{i}/{self.l_rounds[i]}/nadd_action.pkl','wb') as f:
                    pickle.dump(nadd_action,f)
        if term_ids is not None and len(term_ids[0])>0:
            for idx,i in enumerate(term_ids[0]):
                if i!=0:
                    term_action={"Terminate":i,"Term_Mask":self.term_masks[i]}
                    with open(f'{savepath}-{i}/{self.l_rounds[i]}/term_action.pkl','wb') as f:
                        pickle.dump(term_action,f)
        return 
    
    def record_nconn_actions(self,batch,bond_to,bond_type,bond_from,likelihoods,savepath='./l_states'):
        for idx,i in enumerate(batch):
            if i!=0:
                nconn_action={"Batch":i,"Bond_to":bond_to[idx],"Bond_type":bond_type[idx],"Bond_from":bond_from[idx],"Likelihoods":likelihoods[idx],"NCONN_MASKS":self.nconn_masks[i]}
                with open(f'{savepath}-{i}/{self.l_rounds[i]}/nconn_action.pkl','wb') as f:
                    pickle.dump(nconn_action,f)
        return
            