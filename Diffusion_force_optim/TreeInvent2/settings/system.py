import numpy as np 
from rdkit import Chem
import os 

class syssetting:
    def __init__(self):
        self.max_ligand_natoms = 50
        self.max_pocket_nres=50
        self.max_ring_size = 20
        self.max_ligand_nodes=40
        self.max_rings=8

        self.ring_type_cover = 0.95
        self.possible_atom_types = [6,7,8,9,15,16,17,35,53]
        self.possible_ringatom_types = [6,7,8,16]
        self.formal_charge_types = [-2,-1,0,1,2]
        self.bond_types = [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.ring_cover_rate = 0.99
        self.n_edge_feats = len(self.bond_types)
        self.n_atom_feats = len(self.possible_atom_types)+len(self.formal_charge_types)
        self.n_node_feats=12
        self.similarity_type='Morgan' #'rdkit'
        self.similarity_radius=2
        self.ring_types_save_path=f'./datasets'
        self.dropout_pocket=0.1
        self.CA_cutoff=8
        self.MC_rate={'holo':0.45, 'apo':0.45, 'random_pert':0.01,'fixbb':0.05,'flexbb':0.04}
        self.select_MC_type=None
    

    def update(self):
        self.ring_types_save_filename = f'{self.ring_types_save_path}/descriptor_{self.ring_cover_rate}.csv'
        if os.path.exists(self.ring_types_save_filename):
            with open(self.ring_types_save_filename,'r') as f:
                self.ring_types=[line.strip().split()[0] for line in f.readlines()]
        else:
            self.ring_types=[]
        
            
        self.num_node_types=len(self.ring_types)+len(self.possible_atom_types)

        self.f_ring_add_dim=(self.max_ring_size,len(self.possible_ringatom_types),len(self.formal_charge_types),len(self.bond_types))
        self.f_ring_connect_dim=(self.max_ring_size,len(self.bond_types))
        self.f_ring_add_per_node=np.prod(self.f_ring_add_dim[1:])
        self.f_ring_connect_per_node=np.prod(self.f_ring_connect_dim[1:])
        self.f_ring_termination_dim=(1)
        
        self.f_graph_add_dim=(self.num_node_types)
        self.f_graph_termination_dim=(1)
        
        self.f_node_joint_dim=(self.max_ligand_natoms,len(self.bond_types))
        
        self.node_type_dict={}
        self.ringatom_type_dict={}
        for aid,i in enumerate(self.possible_atom_types):
            self.node_type_dict[i]=aid

        for aid,i in enumerate(self.possible_ringatom_types):
            self.ringatom_type_dict[i]=aid

        for rid,ringtype in enumerate(self.ring_types):
            self.node_type_dict[ringtype]=rid+len(self.possible_atom_types)
        #print (self.node_type_dict)

        self.node_type_reverse_dict={v:k for k,v in self.node_type_dict.items()}
        #print (self.node_type_reverse_dict)
        self.f_node_dict={}
        for key,value in self.node_type_reverse_dict.items():
            if str(value)[0]=='R':
                var=value.split('-')
                #print (var)
                rnum=int(var[0][1:])
                Cnum=int(var[1][1:])
                Nnum=int(var[2][1:])
                Onum=int(var[3][1:])
                Fnum=int(var[4][1:])
                Pnum=int(var[5][1:])
                Snum=int(var[6][1:])
                Besnum=int(var[10][3:])
                ARnum=int(var[11][2:]) 
                f=[rnum,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Besnum,ARnum]
            else:
                f=[0]*12
                aid=int(key)+1
                f[aid]=1
            self.f_node_dict[key]=f
            
        self.ringat_to_molat=np.array([self.possible_atom_types.index(i) for i in self.possible_ringatom_types])
        