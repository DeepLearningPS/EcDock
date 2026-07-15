import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.smearing import GaussianSmearing

try:
    pass
except ImportError:
    pass


from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)

from EcDock.comparm import *


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = (
    23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100
)


@registry.register_model("equiformer_v2")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions #
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        '''
        "Compute graph on the fly" 。，。
        ，。
        ，
        '''
        max_neighbors (int):    Maximum number of neighbors per atom ##
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms #，
        max_num_elements (int): Maximum atomic number  #，？one-hot。。

        num_layers (int):             Number of layers in the GNN #

        sphere_channels (int):        Number of spherical channels (one set per resolution) #（），
        '''
        ，"Number of spherical channels" 。，。
         ""  ""。

        " ，。， 
        l 。l,  l 。

        。，.
        ，。，
        
        '''

        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention #SO(2)
        num_heads (int):            Number of attention heads #
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head #， = *
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network  #
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh']) #, BN，LN

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)   #，
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)  #，。                                                                                                
        '''
        #.，，，，。
        。

        ，l_max，。。

        ，、，l_max，。l_max，。

        ？。，。，，，
        ？


        ，SO(2)  SO(3) （Special Orthogonal Group），

        #，

        SO(2)： 。 
        SO(2) ，。 
        SO(2) ，。，
        SO(2)  θ， θ 。

        SO(3)： 。 
        SO(3) ，。 
        SO(3) ，，，1 (3*3)

        '''
        grid_resolution (int):        Resolution of SO3_Grid  #SO3_Grid，，None。

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        #，，

        edge_channels (int):                Number of channels for the edge invariant features #，
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features #True
        #（）（），，

        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks #，
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        #Lm， False，。，

        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances #gaussian， ，

        attn_activation (str):      Type of activation function for SO(2) graph attention #so(2)'scaled_silu'
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer 
        #so(2)，False，Equiformer

        use_attn_renorm (bool):     Whether to re-normalize attention weights #。Ture
        ffn_activation (str):       Type of activation function for feedforward network #'scaled_silu'
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation  #gate
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. #FFNMLP，MLP，
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False. #“use_gate_act”False，S2。True

        alpha_drop (float):         Dropout rate for attention weights #0.1
        drop_path_rate (float):     Drop path rate #0.05
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks #0.0

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions #，


        #V2
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        #“max_neighbors”arg，/，。?
        
        avg_num_nodes (float):      Average number of nodes per graph    #
        avg_degree (float):         Average degree of nodes in the graph #


        #，use_energy_lin_ref  load_energy_lin_ref ，True，false，False，True
        #，OC22，？？？
        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        
        。
        ，“False”，OC22“lin_ref”。
        （），“True”，。
        
    
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    
        ，nn.Parameters。()
        ，，“use_energy_lin_ref”TrueFalse。
        use_energy_lin_refTrue，load_energy_lin_refFalse，。。

    """

    def __init__(
        self,
        num_atoms = None,      # not used
        bond_feat_dim = None,  # not used
        num_targets = None,    # not used
        use_pbc=True,        #
        regress_forces=True, #
        otf_graph=True,      #，，
        max_neighbors=500,   #,20                 #500？32？
        max_radius=5.0,      #，。12.0 #？
        max_num_elements=90, #one-hot。。   #？

        num_layers=12,       #
        sphere_channels=128, #
        attn_hidden_channels=128, #SO(2)，？，64
        num_heads=8,              #
        attn_alpha_channels=32,   #, 256/8 = 32，64
        attn_value_channels=16,   #128/8 = 16
        ffn_hidden_channels=512,  #  512/8 = 64，128
        
        norm_type='rms_norm_sh',  #，'layer_norm_sh' 。, BN，LN
        
        lmax_list=[6],            #，
        mmax_list=[2],            #，。，，
        grid_resolution=None,     #16，SO3_Grid，，None。（mmax_list）

        num_sphere_samples=128,   #，，。？

        edge_channels=128,        #，
        use_atom_edge_embedding=True, 
        #（）（），，
        
        share_atom_edge_embedding=False, #，
        use_m_share_rad=False,
        #Lm， False，。，

        distance_function="gaussian",  #gaussian， ，
        num_distance_basis=512,        #

        attn_activation='scaled_silu', #'silu'，so(2)'scaled_silu'

        use_s2_act_attn=False,        #so(2)，False，Equiformer
        use_attn_renorm=True,         #。Ture
        ffn_activation='scaled_silu', #'silu'，'scaled_silu'
        use_gate_act=False,           #gate
        use_grid_mlp=False,           #True，FFNMLP，MLP，。FFN？？
        use_sep_s2_act=True,          #“use_gate_act”False，S2。True

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0,                #0.0

        weight_init='normal',         #'uniform'，
        learn_energy = False,

        distance_resolution: float = 0.02, #Distance between distance basis functions in Angstroms

        #
        enforce_max_neighbors_strictly: bool = False,
        avg_num_nodes: Optional[float] = None,
        avg_degree: Optional[float] = None,

        #
        use_energy_lin_ref: Optional[bool] = False,
        load_energy_lin_ref: Optional[bool] = False,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error(
                "You need to install e3nn==0.4.4 to use EquiformerV2."
            )
            raise ImportError

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements + 1

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding: #
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.learn_energy = learn_energy

        self.distance_resolution = distance_resolution

        #V2
        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE
        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly  #True，4
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."


        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.device = "cuda"  # torch.cuda.current_device()

        self.grad_forces = True  #，,？
        self.num_resolutions: int = len(self.lmax_list) #：Laplace。List of maximum degree of the spherical harmonics (1 to 10)，

        #。：？？
        self.sphere_channels_all: int = (
            self.num_resolutions * self.sphere_channels # * (128)
        )

        # Weights for message initialization
        #，10，10*d，
        #。one-hot，mlp，
        #，，，，，，，one-hot+
        #，：：，，，，；：，，。5+3=8
        #，，，，，。？
        #msemse，
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms #1？，，mse，
        #，，GaussianSmearing()
        '''
        GaussianSmearing ，。、，。，
        ，。

        GaussianSmearing ，。，，。
        '''

        #，，GPU，，
        assert self.distance_function in [
            "gaussian",
        ]

        self.num_gaussians = int(max_radius / self.distance_resolution)

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                #self.num_gaussians # 12.0 / 0.02 = 600
                int(self.num_gaussians / GP.bond_type_num),  #75, #600，，75， 75 * 8() = 600,self.distance_expansion。GPU，
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions（，，MLP） (input channels and 2 hidden channels) #mlp
        ##print('int(self.distance_expansion.num_output):', int(self.distance_expansion.num_output * 8))
        #exit()
        self.edge_channels_list = [int(self.distance_expansion.num_output * GP.bond_type_num)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding #，，，GNN
        #one-hot,。，
        #
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding: #self.share_atom_edge_embeddingFalse
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        '''
        #
        Wigner D ，。 Eugene Wigner
        20，。

        Wigner D SO(3) ，。，（unitary）（irreducible），
        
        '''
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions): #，，，6，list，
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts #lm，
        self.mappingReduced = CoefficientMappingModule(
            self.lmax_list, self.mmax_list #（6,2）
        )

        # Initialize the transformations between spherical（） and grid representations
        '''
        Grid representations 。，
        、、。
        3D。，？？？
        '''
        self.SO3_grid = ModuleListInfo(
            "({}, {})".format(max(self.lmax_list), max(self.lmax_list))
        )
        for lval in range(max(self.lmax_list) + 1): #1~6
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid) # 6*6SO3_Grid

        # Edge-degree embedding，，，，
        #，
        #，，，，，one-hot，MLP
        #，=one-hot+MLP, (MLP), （）
        #= one-hot+MLP, （）。，，，，
        #，，，mlp
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding, #
            rescale_factor=self.avg_degree, #avg_degree or _AVG_DEGREE，avg_degreeNone，_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2 #EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)


        #EquiformerV2，，，，
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list), # 6
            num_channels=self.sphere_channels, # 128
        )

        # ，，，，？

        if self.learn_energy:
            self.energy_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
            )

        #
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        if self.load_energy_lin_ref: #，false
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        self.apply(self._init_weights) #，？
        self.apply(self._uniform_init_rad_func_linear_weights)

    @conditional_grad(torch.enable_grad())
    def forward(self,         
        h               = None, 
        pos             = None, 
        distance_vec    = None, 
        edge_dist       = None,
        element         = None,
        edge_type       = None, 
        edge_index      = None, 
        mask_ligand     = None, 
        sigmas          = None, 
        mask            = None, 
        batch           = None, 
        protein_max_atom_num = None, 
        ligand_max_atom_num  = None, 
        node_atom            = None,
        e_w                  =None, 
        fix_x                =None,

            ):
        self.batch_size = max(batch) + 1 #，，，natoms
        self.dtype  = pos.dtype   #float32
        self.device = pos.device  #cdua()

        atomic_numbers = element.long() #
        num_atoms      = len(atomic_numbers) #

        edge_distance     = edge_dist 
        edge_distance_vec = distance_vec

        ###print('self.batch_size:', self.batch_size)
        ###print('atomic_numbers:',atomic_numbers.shape) #torch.Size([488])
        ####print('atomic_numbers:',atomic_numbers) #
        ####print('batch flag:', batch)
        ###print('self.max_num_elements:', self.max_num_elements) #17
        ###print('max(atomic_numbers):', max(atomic_numbers)) #max(atomic_numbers): tensor(17, device='cuda:0')

        if max(atomic_numbers) > self.max_num_elements - 1:
            raise Exception(f'{max(atomic_numbers) > self.max_num_elements - 1}')

        ###print('edge_index:', edge_index.shape)
        ####print('edge_index:', edge_index[:,:50]) #
        ###print('edge_distance:', edge_distance.shape)
        ###print('edge_distance_vec:', edge_distance_vec.shape)

        '''
        atomic_numbers: torch.Size([488])
        self.max_num_elements: 17
        edge_index: torch.Size([2, 15507])
        edge_distance: torch.Size([15507])
        edge_distance_vec: torch.Size([15507, 3])
        x: torch.Size([488, 49, 128])
        offset: 128
        offset_res: 49
        edge_distance befor: torch.Size([15507])
        edge_distance after: torch.Size([15507, 600])
        '''
        
        '''
        #
        #，，
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly, #，True，。False
        )
        '''

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge #
        edge_rot_mat = self._init_edge_rot_mat(
            None, None, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations #
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        #
        # Init per node representations using an atomic number based embedding, ，
        offset = 0
        x = SO3_Embedding(
            num_atoms, #
            self.lmax_list,  #6
            self.sphere_channels, #8
            self.device,
            self.dtype,
        )

        #print('x1:', x.embedding.shape)#torch.Size([489, 49, 8]) #49？
        ##，lmax_list，，，49，8，4，，
        #4*49==196
        #exit()

        '''
        edge_index: torch.Size([2, 17829])                                                                                                                                                                               
        edge_distance: torch.Size([17829])                                                                                                                                                                               
        edge_distance_vec: torch.Size([17829, 3])                                                                                                                                                                        
        x: torch.Size([560, 49, 128])                                                                                                                                                                                    
        /home/bingxing2/wangzw/pengyq/1.12.1/pytorch/aten/src/ATen/native/cuda/Indexing.cu:975: indexSelectLargeIndex: block: [553,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
        '''

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution #
        for i in range(self.num_resolutions): #1
            '''
            offset in: 0                                                                                                                                                                                                     
            offset_res in: 0                                                                                                                                                                                                 
            x in befor: torch.Size([560, 49, 128]) 
            '''
            #print('offset in:', offset)
            ###print('offset_res in:', offset_res)
            ###print('self.max_num_elements:', self.max_num_elements) #17
            ###print('max(atomic_numbers):', max(atomic_numbers)) #max(atomic_numbers): tensor(17, device='cuda:0')

            if max(atomic_numbers.detach().cpu().tolist()) > self.max_num_elements - 1:
                raise Exception(f'{max(atomic_numbers) > self.max_num_elements - 1}')
            
            if self.num_resolutions == 1:

                #，，[560, 0, 0]， [553,0,0]
                ###print('x in befor:', x.embedding.shape)
                ###print('self.sphere_embedding:', self.sphere_embedding.weight.shape) #17 * 128 , torch.Size([17, 128]) ,
                #，17，self.sphere_embedding17 * 128，0，17，，
                #self.sphere_embedding，1
                ###print('self.sphere_embedding(atomic_numbers) befor:', self.sphere_embedding(atomic_numbers).shape) #

                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )

                ###print('x in after:', x.embedding.shape)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        #print('x2:', x.embedding.shape) #torch.Size([489, 49, 8])
        ###print('offset:', offset)
        ###print('offset_res:', offset_res)
        #
        # Edge encoding (distance and atom edge)
        ###print('edge_distance befor:', edge_distance.shape)
        edge_distance = self.distance_expansion(edge_distance) #
        ###print('edge_distance after:', edge_distance.shape)

        #
        #print('edge_type, edge_distance:', edge_type.shape, edge_distance.shape)
        #edge_type, edge_distance: torch.Size([15626, 8]) torch.Size([15626, 75])

        #edge_type, edge_distance: torch.Size([9354, 8]) torch.Size([9354, 600])

        edge_distance = self.outer_product(edge_type, edge_distance) #edge_type, edge_distance  8 * 600 = 4800，edge_distance

        #print('edge_distance1:', edge_distance.shape) #edge_distance1: torch.Size([15626, 600])
        #edge_distance: torch.Size([9354, 4800])
        #exit()
        

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding: #
            source_element = atomic_numbers[
                edge_index[0]
            ]  # Source atom atomic number
            target_element = atomic_numbers[
                edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        #print('edge_distance2:', edge_distance.shape) #edge_distance1: torch.Size([15626, 600])
        # Edge-degree embedding，。？
        
        #raise Exception('stop')
    
        '''
        atomic_numbers: torch.Size([488])
        self.max_num_elements: 17
        edge_index: torch.Size([2, 15507])
        edge_distance: torch.Size([15507])
        edge_distance_vec: torch.Size([15507, 3])
        x: torch.Size([488, 49, 128])
        offset: 128
        offset_res: 49
        edge_distance befor: torch.Size([15507])
        edge_distance after: torch.Size([15507, 600])
        '''

        ###print('edge_index2:', edge_index.shape)
        ####print('edge_index:', edge_index[:,:50]) #
        ###print('edge_distance2:', edge_distance.shape)
        ###print('edge_distance_vec2:', edge_distance_vec.shape)


        #
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )

        

        ###print('edge_degree:', edge_degree.embedding.shape)
        #raise Exception('stop')
        #x.embedding = x.embedding + edge_degree.embedding #+

        #
        try:
            #print('h:', h.shape)
            #print('x.embedding:', x.embedding.shape)
            '''
            h: torch.Size([716, 3136]) ,  torch.Size([716, 49， 64])                                                                                                                                                            
            x.embedding: torch.Size([716, 49, 128])
            '''
            x.embedding = x.embedding + edge_degree.embedding 
            #x.embedding = x.embedding + h.unsqueeze(dim=1)
            x.embedding = x.embedding + h.view(x.embedding.size())
            #print('h:', h.shape)
            #print('x.embedding:', x.embedding.shape)
            #exit()
            '''
            #，25*4，Equiformer
            h: torch.Size([969, 100])
            x.embedding: torch.Size([969, 25, 4])
            h: torch.Size([969, 100])
            x.embedding: torch.Size([969, 25, 4])
            '''
        except Exception as e:
            #print('error:', e)
            #print('x.embedding:', x.embedding.shape)
            #print('h:', h.shape)
            raise Exception(e)

        #print('x3:', x.embedding.shape) #torch.Size([489, 49, 8])
        

        ###############################################################
        # Update spherical node embeddings
        # GNN，
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch,  # for GraphDropPath
            )

        
        # Final layer norm
        x.embedding = self.norm(x.embedding)
        ###print('x.embedding output:', x.embedding.shape)

        #outputs = {"emb": x.embedding.sum(dim = 1)}
        outputs = {"emb": x.embedding.view(x.embedding.size(0), -1)} #3，2

        #print('x4:', x.embedding.shape)  #x4: torch.Size([489, 49, 8])
        ###############################################################
        # Energy estimation
        #
        ###############################################################

        if self.learn_energy:
            #print('predict energy')
            node_energy = self.energy_block(x)
            node_energy = node_energy.embedding.narrow(1, 0, 1)
            energy = torch.zeros(
                self.batch_size,
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energy.index_add_(0, batch, node_energy.view(-1))
            energy = energy / self.avg_num_nodes

            # Add the per-atom linear references to the energy.
            if self.use_energy_lin_ref and self.load_energy_lin_ref:
                # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
                # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
                # where
                #
                # E_DFT = raw DFT energy,
                # E_ref = reference energy,
                # E_mean = normalizer mean,
                # E_std = normalizer std,
                # \hat{E} = predicted energy,
                # \hat{E_DFT} = predicted DFT energy.
                #
                # We can also write this as
                # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
                # which is why we save E_ref / E_std as the linear reference.
                with torch.cuda.amp.autocast(False):
                    energy = energy.to(self.energy_lin_ref.dtype).index_add(
                        0,
                        batch,
                        self.energy_lin_ref[atomic_numbers],
                    )

            outputs = {"energy": energy}
        ###############################################################
        # Force estimation,GPU，，，edge_distancen * 600
        ###############################################################
        if self.regress_forces:
            #print('predict forces')
            forces = self.force_block(
                x, atomic_numbers, edge_distance, edge_index
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)
        

            #
            pos[mask_ligand]  = forces[mask_ligand]
            #pos = pos + forces * mask_ligand[:, None]  # only ligand positions will be updated
            outputs["forces"] = pos

            ###print('forces:', pos.shape)
            #raise Exception('stop')
            #exit()
        else:
            outputs["forces"] = None

        
        return outputs["emb"], outputs["forces"]



    def outer_product(self, *vectors):
        '''
            edge_type[n_src & n_dst] = 0   #
            edge_type[n_src & ~n_dst] = 1  #，
            edge_type[~n_src & n_dst] = 2  #，
            edge_type[~n_src & ~n_dst] = 3 #
            edge_type = F.one_hot(edge_type, num_classes=4)
        '''
        #vectors = (edge_attr, dist_feat)； edge_attr
        for index, vector in enumerate(vectors): 
            if index == 0: #
                out = vector.unsqueeze(-1) #E * D1 * 1
            else: #
                out = out * vector.unsqueeze(1) #vector.unsqueeze(1) ,E * 1 * D2
                out = out.view(out.shape[0], -1).unsqueeze(-1) # E * (D1*D2)
        return out.squeeze()


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, (torch.nn.Linear, SO3_LinearV2)):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)
