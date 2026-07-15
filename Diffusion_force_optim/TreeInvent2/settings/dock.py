
class docksetting:
    def __init__(self):
        self.dock_input_path='.'
        self.target_pdb=''
        self.reflig_pdb=''
        self.backend='AutoDock-Vina' #"Glide"
        self.box_size=20
        self.low_threshold=-13.0
        self.high_threshold=-2.0
        self.k=0.25
        self.dockstream_root_path='./'
        self.vina_bin_path='./'
        self.glide_keywords={}
        self.ncores=10
        self.nposes=2
        self.grid_path=''
        self.glide_flags={}
        self.glide_ver='2017'
        self.glide_keywords={}
    def update(self):
        pass

