from rdkit import Chem
import json 
import os 
import numpy as np 
from .topology_parms import *

Element_Table={'H':1,'B':5,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17,'Br':35,'I':53}
Table_Element={v:k for k,v in Element_Table.items()}

class GPARMAS:
    def __init__(self):
        #assert "TREEINVENT_HOME" in os.environ.keys(), "Please set TREEINVENT_HOME in your environment variables! echo 'export TREEINVENT_HOME=/path/to/TreeInvent2' >> ~/.bashrc"
        self.atom_types=[6,7,8,9,15,16,17,35,53]
        self.atom_types_for_feats=[1,5,6,7,8,9,15,16,17,35,53]
        self.atom_max_bonds_table={1:1,5:3,6:4,7:3,8:2,9:1,15:5,16:6,17:1,35:1,53:1} 
        self.ring_atom_types=[6,7,8,16]
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC,'HBOND','SALTBRIDGE','PISTACKING','PICATION','HYDROPHOBIC','HALOGEN','WATERBRIDGE','PL']
        self.formal_charge_types=[-2,-1,0,1,2]
        self.pharm_types=['Donor','Acceptor','NegIonizable','PosIonizable','Aromatic','Hydrophobe','LumpedHydrophobe']
        self.max_atoms=210
        
        self.max_latoms=60
        self.max_lgroups=60
        self.max_lgroup_size=27
        self.max_single_ring_size=8
        self.max_lbonds=70
        self.max_langles=97
        self.max_ldihedrals=140
    
        self.max_patoms=150
        self.max_pgroups=55
        self.max_pgroup_size=100
        self.max_pbonds=160
        self.max_pangles=250
        self.max_pdihedrals=250
        
        self.n_group_feats=19

        self.group_type_file='./datasets/descriptor_0.98.csv'
        
        self.batchsize=1
        self.accsteps=5
        self.pl_mask_rate=0
        self.device='cuda'
        
        self.conf_depth=6
        self.conf_edge_channels=64
        self.conf_sphere_channels=128
        self.conf_ffn_hidden_channels=256
        self.conf_attn_alpha_channels=32
        self.conf_attn_value_channels=16
        self.conf_num_heads=8
        self.conf_num_distance_basis=512
        self.conf_num_sphere_samples=128
        self.conf_max_neighbors=500
         
        self.graph_hidden_dim=256
        self.graph_message_size=256
        self.graph_message_passes=3
        self.dropout=0.0
        self.graph_depth=4
        self.only_2d=False

        self.consistency_training_steps=25
        self.sigma_min=0.002
        self.sigma_max=80.0
        self.rho=7.0
        self.sigma_data=0.5
        self.initial_timesteps=2
        self.final_timesteps=25
        self.initlr=0.0001 
        self.lr_patience=100
        self.lr_cooldown=100
        self.n_workers=0
        self.loss_weight={"ic":0.9,"xyz":0.1}

        self.p_mode='fix-all' # or 'fix-backbone','fix-nonint','relax-all'
        self.l_mode='relax-all' # or 'fix-previous'
        self.int_mode='atom-based' # or 'atom-based'
        self.ec_mode='inpaint' # or 'fix-protein'

        self.save_diff_process=False
        self.pl_mask_rate=0
        self.noise_std_for_graph=0.0

        self.with_term_model=False
        self.with_MMFF_guide=True
        self.MMFF_lr=0.001
        self.MMFF_decay = 0.5
        self.MMFF_max_decay_step = 5
        self.MMFF_patience_tol_step = 10
        self.MMFF_patience_tol_value = 0
        self.MMFF_clip = 1e+5
        self.MMFF_guide_loops = 1
        self.MMFF_constraint = 0
        self.MMFF_guide_type='asynchronous' # synchronous or asynchronous
        self.MMFF_guide_model='LFBGS'

        self.with_ctrlnet_for_nadd=False
        self.with_ctrlnet_for_rgen=False
        self.with_ctrlnet_for_nconn=False
        self.with_ctrlnet_for_nint=False
        
        self.freeze_backbone_for_nadd=True
        self.freeze_backbone_for_rgen=True
        self.freeze_backbone_for_nconn=True
        self.freeze_backbone_for_nint=True

        self.load_dict={'coords':None,'ema':None,'nadd':None,'rgen':None,'nconn':None,'nint':None,'term':None}
        self.load_type='perepoch'
        self.conf_duplications=1
        
        
    def Update(self):
        if os.path.exists(self.group_type_file):
            with open(self.group_type_file,'r') as f:
                self.group_type_index_dict={line.strip().split()[0]:lid for lid,line in enumerate(f.readlines())}
        
        self.group_index_type_dict={v:k for k,v in self.group_type_index_dict.items()}
        
        self.tree_add_dim=(len(self.group_index_type_dict.keys())+1)
        self.tree_conn_dim=(self.max_lgroups)
        self.tree_int_add_dim=(self.max_pgroups)
        self.tree_int_term_dim=(1)
        
        self.leaf_add_dim=len(self.group_index_type_dict.keys())+1
        self.r_add_dim=(self.max_lgroup_size,len(self.atom_types),len(self.formal_charge_types),4)
        self.r_conn_dim=(self.max_lgroup_size,4)
        self.r_term_dim=(1)
        self.r_gen_dim=np.prod(self.r_add_dim)+np.prod(self.r_conn_dim)+np.prod(self.r_term_dim)
        self.leaf_conn_dim=(self.max_latoms,4)
        self.leaf_term_dim=(1)
        self.leaf_int_add_dim=(self.max_pgroups,7)
        self.leaf_int_term_dim=(1)
        
class DockPARAMS:
    def __init__(self):
        #assert "DockStream_HOME" in os.environ.keys(), "Please set DockStream_HOME in your environment variables! echo 'export DockStream_HOME=/path/to/DockStream' >> ~/.bashrc"
        #self.dockstream_root_path=os.environ['DockStream_HOME']
        self.dock_input_path='.'
        self.target_pdb=''
        self.reflig_pdb=''
        self.backend='AutoDock-Vina' #"Glide"
        self.box_size=20
        self.low_threshold=-13.0
        self.high_threshold=-2.0
        self.k=0.25
        self.vina_bin_path='./'
        self.glide_keywords={}
        self.ncores=10
        self.nposes=2
        self.grid_path=''
        self.glide_flags={}
        self.glide_ver='2017'
        self.glide_keywords={}
    def Update(self):
        assert  self.backend in ['AutoDock-Vina','Glide'], "Only AutoDock-Vina and Glide are supported!"
        if self.backend=='AutoDock-Vina':
            assert "VINA_HOME" in os.environ.keys(), "Please set VINA_HOME in your environment variables! echo 'export VINA_HOME=/path/to/autodock-vina' >> ~/.bashrc"
            self.vina_bin_path=os.environ['VINA_HOME']
        elif self.backend=='Glide':
            assert "SCHRODINGER" in os.environ.keys(), "Please set SCHRODINGER in your environment variables! echo 'export SCHRODINGER=/path/to/schrodinger' >> ~/.bashrc"
            assert self.grid_path != '', "Please provide the Glide grid file path of the receptor!"
        return 
        
    
class RocsPARAMS:
    def __init__(self):
        self.cff_path=''
        self.reflig_sdf_path=''
        self.shape_w=0.5
        self.color_w=0.5
        self.sim_measure="Tanimoto"
        
    def Update(self):
        pass
    
class node_add_constrain:
    def __init__(self):
        self.ringnum_range=(0,10)
        self.ar_ringnum_range=(0,10)
        self.branchnum_range=(0,10)
        self.atnum_range_dict={'C':(0,100),'N':(0,100),'O':(0,100),'F':(0,100),'P':(0,100),'S':(0,100),'Cl':(0,100),'Br':(0,100),'I':(0,100)}
        self.total_atnum_range=(0,100)
        self.pharm_types={'Donor':False,'Acceptor':False,'NegIonizable':False,'PosIonizable':False,'Aromatic':False,'Hydrophobe':False,'LumpedHydrophobe':False}
        self.force_step=False
        self.specific_nodefile=None
        self.update()
        return

    def update(self):
        pass
        return

class node_conn_constrain:
    def __init__(self):
        self.bondnum_range=(0,10)
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.force_step=False
        self.specific_nodefile=None
        self.constrain_connect_node_id=[]
        self.update()
        return

    def update(self):
        return
 
def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_PARAMS(obj,jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,obj)
    obj.Update()
    return obj

def Define_topology_constrains_from_json(PARAMS,jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        if 'sample_constrain' in jsondict.keys():
            Loaddict2obj(jsondict['sample_constrain'],PARAMS)
            if "nadd_constrain_basic" in jsondict["sample_constrain"].keys():
                Loaddict2obj(jsondict["sample_constrain"]["nadd_constrain_basic"],PARAMS.nadd_constrain_basic)
            if "nconn_constrain_basic" in jsondict["sample_constrain"].keys():
                Loaddict2obj(jsondict["sample_constrain"]["nconn_constrain_basic"],PARAMS.nconn_constrain_basic)
            if "constrain_step_dict" in jsondict["sample_constrain"].keys():
                for key in jsondict["sample_constrain"]["constrain_step_dict"].keys():
                    if key in PARAMS.constrain_step_dict.keys():
                        if "node add" in jsondict["sample_constrain"]["constrain_step_dict"][key].keys():
                            tmp_nadd_constrain=nadd_constrain() 
                            Loaddict2obj(jsondict["sample_constrain"]["constrain_step_dict"][key]["node add"],tmp_nadd_constrain)
                            PARAMS.constrain_step_dict[key]["node add"]=tmp_nadd_constrain
                        if "node conn" in jsondict["sample_constrain"]["constrain_step_dict"][key].keys():
                            tmp_nconn_constrain=nconn_constrain()
                            Loaddict2obj(jsondict["sample_constrain"]["constrain_step_dict"][key]["node conn"],tmp_nconn_constrain)
                            PARAMS.constrain_step_dict[key]["node conn"]=tmp_nconn_constrain
    PARAMS.update()
    return 

FGP=GPARMAS()

TP=TopologyPARAMS()
TP.update()

DP=DockPARAMS()

