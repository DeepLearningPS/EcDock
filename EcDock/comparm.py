from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
import json 

class GPARMAS:
    def __init__(self):
        self.atom_types=[1,6,7,8,9,15,16,17,35,53]
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.if_chiral=False
        self.chiral_types=[ ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_OTHER,
                            ChiralType.CHI_TETRAHEDRAL, ChiralType.CHI_ALLENE, ChiralType.CHI_SQUAREPLANAR, ChiralType.CHI_TRIGONALBIPYRAMIDAL,ChiralType.CHI_OCTAHEDRAL]
        self.max_atoms=250 #,250pdbbind2020，,posebusters:64
        self.max_protein_atoms = 256
        self.batchsize=50
        self.device='cuda'
        self.dim=(16,16)
        self.dim_head=(16,16)
        self.heads=(8,4)
        self.num_linear_att_heads=0
        self.num_degrees=2
        self.depth=6
        self.consistency_training_steps=5 #, self.final_timesteps
        self.sigma_min=0.002
        self.sigma_max=80.0  #80.0
        self.rho=7.0
        self.sigma_data=0.5
        self.initial_timesteps=2  #
        self.lr_patience=100
        self.lr_cooldown=100
        self.n_workers=20
        self.multi_step = 0
        self.final_timesteps=5  #
        self.recover=False

        self.ema_exit = False #False，CMv2
        self.with_MMFF_guide = False
        self.guide_type = 'asynchronous' #synchronous / asynchronous 
        self.opt_types = 'complex'
        self.min_type = 'LBFGS' #/SGD/AdamW/LBFGS
        self.cross_loss = 1 #，. 1,
        self.force_loss = 1 #

        self.sample_batch_size = 2 # #，，

        self.force_step = 1 #

        self.embedding3d = False #3d
        self.embedding3d_noise_pos = False #3D，

        self.glide_vina = False #glidevina（）, glide、vina，

        self.cross_distance_num = 'best' #, best/20/40, Glide and Vina setting is 5, ECDock setting is 20

        self.single_cross_distance_id = None #0，，，，None，，，id，5

        
        self.use_distance   = True #，True
        
        self.data_type = 'CrossDocked2020' #，CrossDocked2020DEKOIS2.0VSDS_DTEBV-D(), BindingNetv2_High, 
        
        self.train_data_name = 'newer_pdbbind2020' # BindingNetv2_High, newer_pdbbind2020
        #
        self.loss_weight={"ic":0.01, "xyz":1.0, "cross_distance":1.0, "ref_cross":1.0, "ref":1.0} #{"ic":0.9,"xyz":0.1}, {"ic":0.1,"xyz":0.9}，，，
        self.interaction_stype = 'interaction'  #atom/centor/distance/all/interaction/interaction_all, ，，，，
        #interaction_all：unimol4.5，，4.5，O，N，
        self.interaction_distance = 10  #，，，self.interaction_stype == 'centor/atom' 
        self.cross_distance_cutoff = 4.5 #unimol cross_distance，4.5, self.interaction_stype == 'interaction' 

        self.min_distance_atom_num = 200 # self.interaction_stype == 'ditance'
        self.atom2atom_distance = 8  #，8ai，

        self.bond_type_num = 20 #，4,8,20, ，4/88/340, equifrmer

        #self.steps_list=[150]  #, [1,2,5,10,15,25],
        #self.steps_list=[25]  #, [1,2,5,10,15,25],
        #self.steps_list=[1,2,5,10,15,25]  #, [1,2,5,10,15,25]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] #，，，15+skip25
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #self.steps_list=[1,2,3,4,5]
        #，，，, ？
        '''
        #sigmas = karras_schedule(num_timesteps, self.sigma_min, self.sigma_max, self.rho, ligand_pos.device)
        sigmas，0.002~80.0, ，sigma[0.002,80.0]
        '''
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15]
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
        #self.steps_list=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        #self.steps_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

        #，EGNN，
            
def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_GPARAMS(jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,GP)
    return 

GP=GPARMAS()
