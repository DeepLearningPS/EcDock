
class trainsetting:
    def __init__(self):
        self.dataset_path='./datasets'
        self.rearrange_molgraph_mode='fixed'
        self.epochs=100
        self.initlr=1e-4
        self.shuffle=False
        self.cut=0.95
        self.optimizer='Adam'
        self.lr_patience=500
        self.lr_cooldown=500
        self.max_grad_norm=1.5
        self.max_rel_lr=1
        self.min_rel_lr=0.01
        self.ring_seq="bfs" # or "dfs"
        self.savepath='.'
        self.dataset_sdf_path='.'
        
        #for FlexPose
        self.apo_rate=0.45
        self.rand_per_rate=0.01
        self.fix_pack_rate=0.05
        self.flex_pack_rate=0.04
        self.holo_rate=0.45
        self.protein_n_rand_pert=3
        self.protein_n_fixbb_repack=3
        self.protein_n_flexbb_repack=3
        self.pocket_rand_pert_range=30
        self.tmp_path='./tmp'
        self.savefreq=5
