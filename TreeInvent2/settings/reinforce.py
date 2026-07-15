
class rlsetting:
    def __init__(self):
        self.score_components=['similarities']
        self.score_weights=[1]
        self.target_smiles=['CO[C@H](C)[C@H](O)CC1CCOCC1']
        self.score_type='continuous'
        self.qsar_models_path='./qsar.pkl'
        self.max_gen_atoms=38
        self.score_thresholds=[1.0]
        self.tanimoto_k=0.7
        self.sigma=50
        self.vsigma=1.5
        self.ksigma=1.5
        self.acc_steps=1
        self.target_molfile=''
        self.temp_range=(1.0,1.2)
        self.temp_scheduler="same"
        self.unknown_fielter=False
        self.save_pic=True
        self.agent_savepath='./'
        self.prior_loadzip=''
        self.agent_loadzip=''
        
    def update(self):
        if self.target_molfile!='':
            with open(self.target_molfile,'r') as f:
                self.target_smiles=[line.strip() for line in f.readlines()]
        if len(self.score_components)>1:
            n=len(self.score_components)-len(self.score_weights)
            for i in range(n):
                self.score_weights.append(1)
        if len(self.score_components)>1:
            n=len(self.score_components)-len(self.score_thresholds)
            for i in range(n):
                self.score_thresholds.append(1)
        return
    
class ftsetting:
    def __init__(self):
        self.smi_path=''
        self.sdf_path=''
        self.converge_similarity=0.7

class clsetting:
    def __init__(self):
        self.smi_path=''
        self.sdf_path=''
        self.max_clsteps_per_iter=1000