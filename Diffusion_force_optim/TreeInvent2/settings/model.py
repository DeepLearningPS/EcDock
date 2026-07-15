
class modelsetting:
    def __init__(self):
        self.batchsize=2
        self.num_workers=0
        #parameters for graph neural network
        self.dropout=0.1
        self.drop_huge_data=False
        self.drop_huge_data_max_len=250
        self.clamp_rate=0
        self.clamp_max=2
        self.p_x_sca_indim=20 
        self.l_x_sca_indim=14
        self.p_edge_sca_indim=18
        self.l_edge_sca_indim=4

        self.p_x_vec_indim=24
        self.l_x_vec_indim=1
        self.p_edge_vec_indim=16
        self.l_edge_vec_indim=1

        self.p_x_sca_hidden=256
        self.p_edge_sca_hidden=128
        self.p_x_vec_hidden=128
        self.p_edge_vec_hidden=64

        self.l_x_sca_hidden=256
        self.l_edge_sca_hidden=128
        self.l_x_vec_hidden=128
        self.l_edge_vec_hidden=64

        self.c_x_sca_hidden=256
        self.c_edge_sca_hidden=128
        self.c_x_vec_hidden=128
        self.c_edge_vec_hidden=64

        self.n_head=4
        self.use_pretrain=True
        self.add_l_dismap=True
        self.conf_p_block=8
        self.conf_l_feat_block=6
        self.conf_c_block=3
        self.conf_c_block_only_coor=3
        self.graph_p_block=4
        self.graph_l_feat_block=3
        self.graph_c_block=3
        self.graph_c_block_only_coor=0


        #parameters for consistency diffusion
        self.sigma_min=0.002
        self.sigma_max=80.000
        self.sigma_data=0.5
        self.initial_timesteps=2
        self.final_timesteps=25
        self.rho=7.0
        self.loadzip=''
        self.loadtype='perepoch'
        
        self.gamma_l_coor=1
        self.gamma_CA=1
        self.gamma_CB=1
        self.gamma_SC=2

        self.MMFF_min=True
        self.MMFF_lr=5e-5
        self.MMFF_decay=0.5
        self.MMFF_max_decay_step=10
        self.MMFF_patience_tol_step=100
        self.MMFF_patience_tol_value=0
        self.MMFF_clip=1e+5
        self.MMFF_constraint=5

        # used only in FlexPose based methods

        self.l_init_sigma=5  
        self.update_coor_clamp=None

        self.coor_scale=10
        self.coords_noise_for_coordgen=0.5
        self.coords_noise_for_graphgen=0.3
        self.drop_huge_data=False
        self.drop_huge_data_max_len=250
        self.mlp_depth=4
        self.n_cycle=4
        self.MMFF_loop=1
        self.pocket_type='holo'
        self.fix_prot=True
        self.consistency_type='outter' 
        self.normal_skip_out=False
        self.sigma_emb=False