from rdkit import Chem 

def predeal_charged_molgraph(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

class nadd_type_constrain:
    def __init__(self):
        self.max_ring_num_per_node=10
        self.min_ring_num_per_node=0
        self.max_aromatic_rings=10
        self.min_aromatic_rings=0
        self.min_branches=0
        self.max_branches=10
        self.max_anum_per_atomtype={'C':100,'N':100,'O':100,'F':100,'P':100,'S':100,'Cl':100,'Br':100,'I':100}
        self.min_anum_per_atomtype={'C':0,'N':0,'O':0,'F':0,'P':0,'S':0,'Cl':0,'Br':0,'I':0}
        self.max_heavy_atoms=100
        self.force_step=False
        self.specific_nodefile=None
        self.update()
        return

    def update(self):
        if self.specific_nodefile:
            with open(self.specific_nodefile,'rb') as f:
                self.specific_nodegraph=pickle.load(f)
                self.specific_nodegraph=predeal_charged_molgraph(self.specific_nodegraph)
                Chem.Kekulize(self.specific_nodegraph)
        else:
            self.specific_nodegraph=None
        return 

class node_conn_constrain:
    def __init__(self):
        self.saturation_atomid_list=[] # defined the non-anchor atomids
        self.constrain_connect_node_id=[] #defined the node to be connected
        self.constrain_connect_atom_id=[[]]
        self.constrain_connect_bond_type=[0,1,2] 
        self.constrain_connect_atomic_type=[6,7,8,9,15,16,17,35,53]
        self.anchor_before=-1 #define the anchor to connect the existed nodes, only used for specific rdkit mol nodes
        return 
    
class samplesetting:
    def __init__(self):
        self.max_node_steps=100
        self.max_ring_nodes=100
        self.min_node_steps=7
        self.min_mw=150
        self.min_atoms=15
        self.temp=1.0
        self.node_constrain_basic=nadd_type_constrain()
        self.node_conn_constrain_basic=node_conn_constrain()
        self.constrain_step_dict={}
        self.samplenum=1000
        self.savepath='./samples'
        self.batchsize=128
        return 
    
    def update(self):
        for i in range(self.max_node_steps+1):
            if str(i) not in self.constrain_step_dict.keys():
                self.constrain_step_dict[str(i)]={"node add":self.node_constrain_basic,"node conn":self.node_conn_constrain_basic}
            else:
                if "node add" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]["node add"]=self.node_constrain_basic
                if "node conn" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]["node conn"]=self.node_conn_constrain_basic 
        for i in range(self.max_node_steps):
            if self.constrain_step_dict[str(i)]["node add"].specific_nodegraph:
                for j in range(i+1):
                    self.constrain_step_dict[str(i)]["node add"].force_step=True
        return