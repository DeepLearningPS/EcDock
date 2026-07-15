
class nadd_constrain:
    def __init__(self):
        self.ringnum_range=(0,10)
        self.ar_ringnum_range=(0,10)
        self.branchnum_range=(0,10)
        self.atnum_range_dict={'C':(0,100),'N':(0,100),'O':(0,100),'F':(0,100),'P':(0,100),'S':(0,100),'Cl':(0,100),'Br':(0,100),'I':(0,100)}
        self.total_atnum_range=(0,100)
        self.pharm_types={'HD':0,'HA':0,'Neg':0,'Pos':0,'Aro':0,'Hyd':0,'LHyd':0} # 0 for yes or no; 1 for yes; -1 for no
        self.force_step=False
        self.specific_nodefile=None
        self.specific_nodegraph=None
        self.update()
        return

    def update(self):
        pass
        return

class nconn_constrain:
    def __init__(self):
        #self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.saturation_atomid_list=[] # defined the non-anchor atomids
        self.constrain_connect_node_id=[] #defined the node to be connected
        self.constrain_connect_atom_id_in_node=[[]]
        self.constrain_stauration_atom_id_in_node=[[]]
        self.constrain_connect_bond_type=[0,1,2,3] 
        self.constrain_connect_atomic_type=[6,7,8,9,15,16,17,35,53]
        self.anchor_before=-1 #define the anchor to connect the existed nodes, only used for specific rdkit mol nodes
        return 

    def update(self):
        return
    
class TopologyPARAMS:
    def __init__(self):
        self.max_node_steps=100
        self.max_group_nodes=100
        self.temp=1.0
        self.nadd_constrain_basic=nadd_constrain()
        self.nconn_constrain_basic=nconn_constrain()
        #self.nint_constrain_basic=nint_constrain()
        self.constrain_step_dict={}
        return 
    def update(self):
        for i in range(self.max_node_steps+1):
            if str(i) not in self.constrain_step_dict.keys():
                self.constrain_step_dict[str(i)]={'node add':nadd_constrain(),'node conn':nconn_constrain()}
            else:
                if "node add" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]['node add']=self.nadd_constrain_basic()
                if "node conn" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]['node conn']=self.nconn_constrain_basic()
        for i in range(self.max_node_steps):
            if self.constrain_step_dict[str(i)]['node add'].specific_nodegraph:
                for j in range(i+1):
                    self.constrain_step_dict[str(j)]['node conn'].force_step=True
        return 

