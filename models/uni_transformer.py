import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
from EcDock.comparm import *

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product
from ocp.ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from ocp.ocpmodels.models.escn.escn import eSCN
import math
from collections import defaultdict
from ordered_set import OrderedSet

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        ##print('edge_feat_dim 4?:', edge_feat_dim) #4,8
        #edge_feat_dim = 4
        
        #raise Exception('test')
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim #，，，
        #print('input_dim, hidden_dim, output_dim, kv_input_dim:', input_dim, hidden_dim, output_dim, kv_input_dim) #128 128 128
        # 128 128 128 424
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):

        #(h_in, dist_feat, edge_feat, edge_index, e_w=e_w) #edge_feat，
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        ##print('kv_input:', kv_input.shape) #kv_input: torch.Size([114023, 424])，MLPin_dim, out_dim, hidden_dim: 344 128 128
        #，x，MLPin_dim，in_dim
        #raise Exception('test')
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads) #，MLP，，
        ##print('hk_func:', k.shape)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        ##print('xk_func:',k.shape)

        v = self.xv_func(kv_input)
        ##print('xv_func:', v.shape)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]

#，
class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        #edge_attr，， edge_attr，mlp，，
        #
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True) #

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l，4，
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist) #GaussianSmearing，，
            #print('edge_attr, dist_feat:', edge_attr.shape, dist_feat.shape)
            #edge_attr, dist_feat: torch.Size([240503, 8]) torch.Size([240503, 20])
            
            dist_feat = outer_product(edge_attr, dist_feat) #=+，,edge_attr
            #print('dist_feat:', dist_feat.shape)
            #dist_feat: torch.Size([240503, 160])
            #exit()
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w) #，，
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out #，，？
        #，
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist) #distance_expansion
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w) 
            #，rel_x，
            if not fix_x: #fix_xfalse，，
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src] #dist，，
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                cutoff_mode='radius', ew_net_type='r',
                num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False,
                equiformer_args = None, equiformer = False, escn_args = None, escn = False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph ，
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]，global

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian) #
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)
        
        self.init_h_emb_layer = self._build_init_h_layer()

        self.equiformer_args = equiformer_args
        self.equiformer = equiformer
        self.escn_args = escn_args
        self.escn = escn

        if GP.embedding3d:
            self.linear_transform_dim = nn.Linear(200 + 200, 200)


            self.ligand_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = False,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = 3,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )


            self.protein_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = False,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = 3,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )
        







        if equiformer == True: #
            print('equiformer')
            self.base_block =  EquiformerV2_OC20(
                use_pbc             = self.equiformer_args.use_pbc,
                regress_forces      = self.equiformer_args.regress_forces,
                otf_graph           = self.equiformer_args.otf_graph,
                max_neighbors       = self.equiformer_args.max_neighbors,
                max_radius          = self.equiformer_args.max_radius,
                max_num_elements    = self.equiformer_args.max_num_elements,
                num_layers          = self.equiformer_args.num_layers,
                sphere_channels     = self.equiformer_args.sphere_channels,
                attn_hidden_channels        = self.equiformer_args.attn_hidden_channels,
                num_heads                   = self.equiformer_args.num_heads,
                attn_alpha_channels         = self.equiformer_args.attn_alpha_channels,
                attn_value_channels         = self.equiformer_args.attn_value_channels,
                ffn_hidden_channels         = self.equiformer_args.ffn_hidden_channels,
                norm_type                   = self.equiformer_args.norm_type,
                lmax_list                   = self.equiformer_args.lmax_list,
                mmax_list                   = self.equiformer_args.mmax_list,
                grid_resolution             = self.equiformer_args.grid_resolution,
                num_sphere_samples          = self.equiformer_args.num_sphere_samples,
                edge_channels               = self.equiformer_args.edge_channels,
                use_atom_edge_embedding     = self.equiformer_args.use_atom_edge_embedding,
                share_atom_edge_embedding   = self.equiformer_args.share_atom_edge_embedding,
                use_m_share_rad     = self.equiformer_args.use_m_share_rad, #False
                distance_function   = self.equiformer_args.distance_function,
                num_distance_basis  = self.equiformer_args.num_distance_basis,
                attn_activation     = self.equiformer_args.attn_activation,
                use_s2_act_attn     = self.equiformer_args.use_s2_act_attn,
                use_attn_renorm     = self.equiformer_args.use_attn_renorm,
                ffn_activation      = self.equiformer_args.ffn_activation,
                use_gate_act        = self.equiformer_args.use_gate_act,
                use_grid_mlp        = self.equiformer_args.use_grid_mlp,
                use_sep_s2_act      = self.equiformer_args.use_sep_s2_act,
                alpha_drop          = self.equiformer_args.alpha_drop,
                drop_path_rate      = self.equiformer_args.drop_path_rate,
                proj_drop           = self.equiformer_args.proj_drop,
                weight_init         = self.equiformer_args.weight_init,        
                )
            
        elif escn == True: #
            print('escn')
            self.base_block =  eSCN(
                use_pbc             = self.escn_args.use_pbc,
                regress_forces      = self.escn_args.regress_forces,
                otf_graph           = self.escn_args.otf_graph,
                max_neighbors       = self.escn_args.max_neighbors,
                max_num_elements    = self.escn_args.max_num_elements,
                num_layers          = self.escn_args.num_layers,

                lmax_list           = self.escn_args.lmax_list,
                mmax_list           = self.escn_args.mmax_list,           
                #grid_resolution     = self.escn_args.grid_resolution,

                sphere_channels     = self.escn_args.sphere_channels,
                hidden_channels     = self.escn_args.hidden_channels,
                edge_channels       = self.escn_args.edge_channels,
                use_grid            = self.escn_args.use_grid,
                num_sphere_samples  = self.escn_args.num_sphere_samples,
                distance_function   = self.escn_args.distance_function,
                basis_width_scalar  = self.escn_args.basis_width_scalar,
                distance_resolution = self.escn_args.distance_resolution,
                show_timing_info    = False,
        
                )
        else:
            print('egnn')
            self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn': #KNN，edge_inde，？，，
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type_8(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #GPU tensor，
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand.cpu() == 1].numpy() #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices = (edge_type == 1).nonzero().view(-1) #0
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index




    @staticmethod
    def _build_edge_type_8_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #GPU tensor，
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices = (edge_type == 1).nonzero().view(-1) #0
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index







    @staticmethod
    def _build_edge_type_20_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #GPU tensor，
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ###print('ligand_node_list:', len(ligand_node_local))
        ###print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ###print('ligand_node_global:', len(ligand_node_global))
        ###print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        ligand_node_global2local_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ##print('K:', k) #tensor(165, device='cuda:0')
            ##print('v:', v) #tensor(1396, device='cuda:0')
            ligand_node_local2global_dict[k.item()] = v
            ligand_node_global2local_dict[v.item()] = k
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            ##print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            ##print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            ##print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            ##print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            ##print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ###print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        protein_node_global2local_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
            protein_node_global2local_dict[v.item()] = k




        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices = (edge_type == 1).nonzero().view(-1) #0
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ''''''
        only_ligand_bond_type     = ligand_bond_type.clone()
        only_ligand_bond_index    = new_ligand_bond_index.clone()
        only_ligand_edge_type_dim = F.one_hot(only_ligand_bond_type, num_classes=20)

        #id，
        #print('only_ligand_bond_index.T.shape:', only_ligand_bond_index.T.shape) #torch.Size([376, 2])
        new_only_ligand_bond_index = torch.zeros(only_ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(only_ligand_bond_index.T):
            try:
                new_only_ligand_bond_index[i][0] = ligand_node_global2local_dict[bd[0].item()] #tensork，，
                new_only_ligand_bond_index[i][1] = ligand_node_global2local_dict[bd[1].item()]
            except Exception as e:
                #print('i:', i)
                #print('bd', bd)
                #print('error:', e)
                raise Exception(ligand_node_global2local_dict)
            
        new_only_ligand_bond_index = new_only_ligand_bond_index.T

        ''''''
        indices1 = (edge_type == 7).nonzero().view(-1) #0
        #  torch.index_select() ,
        indices_to_keep = indices1
        only_protein_bond_type      = torch.index_select(edge_type, 0, indices_to_keep)
        only_protein_bond_index     = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        only_protein_edge_type_dim  = F.one_hot(only_protein_bond_type, num_classes=20)

        #id，
        new_only_protein_bond_index = torch.zeros(only_protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(only_protein_bond_index.T):
            try:
                new_only_protein_bond_index[i][0] = protein_node_global2local_dict[bd[0].item()] #tensork，，
                new_only_protein_bond_index[i][1] = protein_node_global2local_dict[bd[1].item()]
            except Exception as e:
                #print('i:', i)
                #print('bd', bd)
                #print('error:', e)
                raise Exception(protein_node_global2local_dict)
        
        new_only_protein_bond_index = new_only_protein_bond_index.T


        #
        ###print('ligand_bond_type:', ligand_bond_type)
        ###print('new_ligand_bond_index:', new_ligand_bond_index)

    
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ###print('new_edge_type:', new_edge_type.shape)
        ###print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index, ligand_node_global2local_dict, protein_node_global2local_dict, only_ligand_edge_type_dim, new_only_ligand_bond_index, only_protein_edge_type_dim, new_only_protein_bond_index






    # @staticmethod #self，，self，
    def _build_edge_type_interaction_20_gpu(self, x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN
        ):
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，


        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #, ？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #-，，
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]

            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #，x，

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #[2, n*m,]
            #
            l_combinations_isring = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring).cuda()
            l_combinations_isO    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN).cuda()
            l_combinations_isN    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO).cuda()

            l2p.append(l_combinations_isring)
            l2p.append(l_combinations_isO)
            l2p.append(l_combinations_isN)

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations(x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring).cuda()
            p_combinations_isO    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO).cuda()
            p_combinations_isN    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN).cuda()

            p2l.append(p_combinations_isring)
            p2l.append(p_combinations_isO)
            p2l.append(p_combinations_isN)

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            

        l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)    
        p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 


        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #knn,(),id
        #，,edge_type,7
        #id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensork，，
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, l2p_edge_index, p2l_edge_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index






    # @staticmethod #self，，self，
    def _build_edge_type_interaction_8_gpu(self, x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN
        ):
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，


        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #, ？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #-，，
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]

            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #，x，

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #[2, n*m,]
            #
            l_combinations_isring = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring).cuda()
            l_combinations_isO    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN).cuda()
            l_combinations_isN    = self.combinations(x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO).cuda()

            l2p.append(l_combinations_isring)
            l2p.append(l_combinations_isO)
            l2p.append(l_combinations_isN)

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations(x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring).cuda()
            p_combinations_isO    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO).cuda()
            p_combinations_isN    = self.combinations(x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN).cuda()

            p2l.append(p_combinations_isring)
            p2l.append(p_combinations_isO)
            p2l.append(p_combinations_isN)

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            

        l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)    
        p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 


        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #knn,(),id
        #，,edge_type,7
        #id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensork，，
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, l2p_edge_index, p2l_edge_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index



    def combinations(self, x, atom1, atom2, atom_index1, atom_index2):
            #，
            vec1 = atom1[atom_index1]
            vec2 = atom2[atom_index2]
            #  torch.meshgrid 
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            copy_combination = combination.clone()
            
            #6ai, ，,
            dis_limit = GP.atom2atom_distance

            dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
            dis_index = dis <= dis_limit #

            combination = combination.t()[dis_index]

            '''
            if len(combination) == 0:
                #print('12')
                dis_limit = 12.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]
            '''
            
            if len(combination) == 0:
                #print('')
                dis_limit = 100000000.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]

            '''
            #6ai, ，。60，，，
            nun_limit = 60

            dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #
            dis_index = dis <= dis_limit #

            combination = combination[dis_index].t()
            '''

            return combination.t()




    # @staticmethod #self，，self，
    def _build_edge_type_interaction_8_gpu_optim(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,
        ):
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，


        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v
        

        #
        '''
        ligand_atom_isring
        ligand_atom_isO
        ligand_atom_isN

        protein_atom_isring
        protein_atom_isO
        protein_atom_isN
        '''
        l2p = [] #
        p2l = [] #
        #torch.unique(tensor, dim=0)
        #print('max(batch):', max(batch))
        #raise Exception('test0')

        for b in range(max(batch) + 1): #, ？
            ligand_atom  = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 1] #-，，
            protein_atom = protein_ligand_node_list[batch == b][mask_ligand[batch == b] == 0]

            #print('ligand_atom:', ligand_atom.shape)
            #print('protein_atom:', protein_atom.shape)
            #print('type(atom_isring):', atom_isring.dtype)
            #print('atom_isring:', atom_isring.shape)
            #print('atom_isO:', atom_isO.shape)
            #print('atom_isN:', atom_isN.shape)

            #
            ligand_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isO    = atom_isO[batch == b][mask_ligand[batch == b] == 1]
            ligand_atom_isN    = atom_isN[batch == b][mask_ligand[batch == b] == 1]

            protein_atom_isring = atom_isring[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isO = atom_isO[batch == b][mask_ligand[batch == b] == 0]
            protein_atom_isN = atom_isN[batch == b][mask_ligand[batch == b] == 0]


            ligand_cross_isring_flag = cross_isring_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_isO_flag    = cross_isO_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_isN_flag    = cross_isN_flag[batch == b][mask_ligand[batch == b] == 1]
            ligand_cross_lp_pos      = cross_lp_pos[batch == b][mask_ligand[batch == b] == 1]

            protein_cross_isring_flag = cross_isring_flag[batch == b][mask_ligand[batch == b] == 0]
            protein_cross_isO_flag    = cross_isO_flag[batch == b][mask_ligand[batch == b] == 0] 
            protein_cross_isN_flag    = cross_isN_flag[batch == b][mask_ligand[batch == b] == 0] 
            protein_cross_lp_pos      = cross_lp_pos[batch == b][mask_ligand[batch == b] == 0]

            #print('protein_cross_lp_pos:', protein_cross_lp_pos.shape)
            #print('cross_distance:', cross_distance.shape)
            #print('batch == b:', (batch == b).shape)
            #protein_cross_lp_pos: torch.Size([125, 3])                                                                                                                                           | 0/40 [00:00<?, ?it/s]
            #cross_distance: torch.Size([13, 125])
            #batch == b: torch.Size([138])       
            cross_distance_matrix = cross_distance[b] #cross_distancelist，cross_distance
            #print('cross_distance.shape:', cross_distance.shape)
            #cross_distance_matrix = cross_distance[batch == b]



            #sub_x = x[batch == b][mask_ligand[batch == b] == 0] #，x，

            #print('ligand_atom_isring:', ligand_atom_isring.shape)
            #print('ligand_atom_isO:', ligand_atom_isO.shape)
            #print('ligand_atom_isN:', ligand_atom_isN.shape)
            #print('protein_atom_isring:', protein_atom_isring.shape)
            #print('protein_atom_isO:', protein_atom_isO.shape)
            #print('protein_atom_isN:', protein_atom_isN.shape)
            #[2, n*m,]
            #
            centor = x[ligand_atom].mean(dim = 0)
            l_combinations_isring = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isring, protein_atom_isring, centor, 
                    ligand_cross_isring_flag, protein_cross_isring_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')
            l_combinations_isO    = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isO, protein_atom_isN, centor, 
                    ligand_cross_isO_flag, protein_cross_isN_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')
            l_combinations_isN    = self.combinations_optim(x, org_x, ligand_atom, protein_atom, ligand_atom_isN, protein_atom_isO, centor, 
                    ligand_cross_isN_flag, protein_cross_isO_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'ligand')

            if l_combinations_isring.size(0) != 0:
                l2p.append(l_combinations_isring.cuda())
            if l_combinations_isO.size(0) != 0:
                l2p.append(l_combinations_isO.cuda())
            if l_combinations_isN.size(0) != 0:
                l2p.append(l_combinations_isN.cuda())

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isring, ligand_atom_isring, centor, 
                    protein_cross_isring_flag, ligand_cross_isring_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            p_combinations_isO    = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isN, ligand_atom_isO, centor, 
                    protein_cross_isN_flag, ligand_cross_isO_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            p_combinations_isN    = self.combinations_optim(x, org_x, protein_atom, ligand_atom, protein_atom_isO, ligand_atom_isN, centor, 
                    protein_cross_isO_flag, ligand_cross_isN_flag, cross_distance_matrix, ligand_cross_lp_pos, protein_cross_lp_pos, flag = 'protein')
            
            if p_combinations_isring.size(0) != 0:
                p2l.append(p_combinations_isring.cuda())
            if p_combinations_isO.size(0) != 0:
                p2l.append(p_combinations_isO.cuda())
            if p_combinations_isN.size(0) != 0:
                p2l.append(p_combinations_isN.cuda())

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

        
        if len(l2p) != 0:
            l2p_edge_index = torch.unique(torch.cat(l2p, dim = -1), dim = -1)
        else:
            l2p_edge_index = torch.empty(0, 0).cuda()
        
        if len(p2l) != 0:
            p2l_edge_index = torch.unique(torch.cat(p2l, dim = -1), dim = -1) 
        else:
            p2l_edge_index = torch.empty(0, 0).cuda()



        #print('l2p[:, :3]:', l2p_edge_index[:, :3]) #[]
        #print('p2l[:, :3]:', p2l_edge_index[:, :3]) #[]

        #print('l2p:', l2p_edge_index.shape) #[]
        #print('p2l:', p2l_edge_index.shape) #[]

        #raise Exception('test')

        '''
        ligand_atom: torch.Size([34])
        protein_atom: torch.Size([400])
        type(atom_isring): torch.bool
        atom_isring: torch.Size([434])
        atom_isO: torch.Size([434])
        atom_isN: torch.Size([434])
        ligand_atom_isring: torch.Size([34])
        ligand_atom_isO: torch.Size([34])
        ligand_atom_isN: torch.Size([34])
        protein_atom_isring: torch.Size([400])
        protein_atom_isO: torch.Size([400])
        protein_atom_isN: torch.Size([400])
        l_combinations_isring: torch.Size([2, 0])
        l_combinations_isO: torch.Size([2, 364])
        l_combinations_isN: torch.Size([2, 324])
        p_combinations_isring: torch.Size([2, 0])
        p_combinations_isO: torch.Size([2, 364])
        p_combinations_isN: torch.Size([2, 324])
        l2p[:, :3]: tensor([[400, 400, 400],
                [  0,   3,   4]], device='cuda:0')
        p2l[:, :3]: tensor([[  0,   0,   0],
                [400, 405, 412]], device='cuda:0')
        l2p: torch.Size([2, 688])
        p2l: torch.Size([2, 688])
        '''


        '''
        #knn,(),id
        #，,edge_type,7
        #id
        new_protein_bond_index = torch.zeros(protein_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(protein_bond_index.T):
            new_protein_bond_index[i][0] = protein_node_local2global_dict[bd[0].item()] #tensork，，
            new_protein_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()]
        
        new_protein_bond_index = new_protein_bond_index.T
        '''
        





        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        l2p_edge_type = torch.full([l2p_edge_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([p2l_edge_index.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        if l2p_edge_index.size(0) != 0:
            new_edge_type  = torch.cat([new_edge_type, l2p_edge_type], dim = 0)  #
            new_edge_index = torch.cat([new_edge_index, l2p_edge_index], dim = 1)

        if p2l_edge_index.size(0) != 0:
            new_edge_type  = torch.cat([new_edge_type, p2l_edge_type], dim = 0)
            new_edge_index = torch.cat([new_edge_index, p2l_edge_index], dim = 1)


        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index








    # @staticmethod #self，，self，
    def _build_edge_type_interaction_8_gpu_optim_v2(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #，
        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T

        #
        new_cross_bond_index = torch.unique(new_cross_bond_index, dim = -1)
        new_cross_bond_index_reverse = torch.unique(new_cross_bond_index_reverse, dim = -1)

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        #，
        l2p_edge_type = torch.full([new_cross_bond_index.size(1)], 5, dtype = torch.int64).cuda()
        p2l_edge_type = torch.full([new_cross_bond_index_reverse.size(1)], 6, dtype = torch.int64).cuda()

        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, l2p_edge_type, p2l_edge_type], dim = 0)  #
        try:
            new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)
        except Exception as e:
            print("error:", e)
            print('new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse:', new_edge_index.shape, new_ligand_bond_index.shape, new_cross_bond_index.shape, new_cross_bond_index_reverse.shape)
            #torch.Size([2, 8000]) torch.Size([2, 56]) torch.Size([4, 24]) torch.Size([4, 24]), 1，，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index






    # @staticmethod #self，，self，
    def _build_edge_type_interaction_20_gpu_optim(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #。
        #id，iid

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #,pyg，，，，.1，，
        #，，id，，，
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #，
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, , mask,。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##0？

        #exit()

        #，
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #

        #1，，，，，? ，
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 11  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index



    def _build_edge_type_interaction_20_gpu_optim_no_interactive_gpu(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #。
        #id，iid

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #,pyg，，，，.1，，
        #，，id，，，
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #，
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, , mask,。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##0？

        #exit()

        #，
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #

        #1，，，，，? ，
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.cpu().numpy().tobytes()] = v  # ,， cpugpu
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].cpu().numpy().tobytes()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.cpu().numpy().tobytes()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].cpu().numpy().tobytes()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].cpu().numpy().tobytes()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].cpu().numpy().tobytes()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 11  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        #indices = torch.cat([indices1, indices5, indices6])
        indices = torch.cat([indices1])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        #rows_to_remove = indices.detach().cpu().tolist()
        #  torch.index_select() ,
        #indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        
        
        #
        '''
        mask = ~torch.isin(a, b)
        result = a[mask]
        '''
        all_index = torch.tensor(list(range(edge_type.size(0)))).cuda()
        mask = ~torch.isin(all_index, indices)
        indices_to_keep = all_index[mask]
        
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index




    def _build_edge_type_interaction_20_gpu_optim_no_interactive(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,

        protein_element_batch,
        protein_link_t_batch,
        protein_link_t_reverse_batch,
        ligand_element_batch,
        protein_element,
        ligand_element,
    ):

        #assert x.shape[0] == len(protein_element) + len(ligand_element) #。
        #id，iid

        ligand_atom_num_list  = []
        protein_atom_num_list = []
        ligand_atom_num_list.append(0)
        protein_atom_num_list.append(0)
        g_num = max(ligand_element_batch) + 1
        #print('g_num:', g_num)
        #print('ligand_element.shape:', ligand_element.shape) #torch.Size([26])
        #print('ligand_element_batch.shape:', ligand_element_batch.shape)
        #print('ligand_element_batch:', ligand_element_batch)

        #print('protein_element.shape:', protein_element.shape) #torch.Size([250])
        #print('protein_element_batch.shape:', protein_element_batch.shape)
        #print('protein_element_batch:', protein_element_batch)
        lg_nums = 0
        pr_nums = 0
        #print('ligand_element_batch:', ligand_element_batch)
        #print('g_num:', g_num)
        for i in range(g_num):
            nm1 = len(ligand_element[ligand_element_batch == i])
            lg_nums = lg_nums + nm1
            ligand_atom_num_list.append(lg_nums)

            nm2 = len(protein_element[protein_element_batch == i])
            pr_nums = pr_nums + nm2
            protein_atom_num_list.append(pr_nums)
        
        #print('ligand_atom_num_list:', ligand_atom_num_list) #ligand_atom_num_list: [0, 13, 13]
        #print('protein_atom_num_list:', protein_atom_num_list) #protein_atom_num_list: [0, 125, 125]
        
        #print('max(ligand_bond_index) befor:', torch.max(ligand_bond_index))

        #,pyg，，，，.1，，
        #，，id，，，
        '''
        #print('ligand_bond_index.shape:', ligand_bond_index.shape) #tensor(189, device='cuda:0')
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num):
            mask = ligand_bond_type_batch == i
            #new_ligand_bond_index[mask] = ligand_bond_index.T[mask] + ligand_atom_num_list[i]

            print('i:', i)
            print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            print('max(ligand_bond_index.T[mask]):', torch.max(ligand_bond_index.T[mask]))


        ligand_bond_index = new_ligand_bond_index.T
        #exit()
        '''

        
        #，
        #print('cross_bond_index.shape:', cross_bond_index.shape) #torch.Size([2, 582])
        #print('protein_link_t_batch.shape:', protein_link_t_batch.shape) #torch.Size([582])
        #print('protein_link_t_batch:', protein_link_t_batch) #
        
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        #print('new_cross_bond_index.shape:', new_cross_bond_index.shape) #torch.Size([582, 2])

        #
        #assert torch.allclose(cross_bond_index.T[protein_link_t_batch == 0], cross_bond_index.T[protein_link_t_batch == g_num - 1], atol=0.02)

        
        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))
        #print('cross_bond_index.T, before:', cross_bond_index.T)
        for i in range(g_num): #
            #print('i:', i)
            #print('ligand_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('protein_atom_num_list[i]:', protein_atom_num_list[i])
            mask = protein_link_t_batch == i
            #print('mask:', mask)
            #print('cross_bond_index.T[mask][:, 0]:', cross_bond_index.T[mask][:, 0])
            #print('protein_atom_num_list[i]:', ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0] + protein_atom_num_list[i]:', cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])
            #print('cross_bond_index.T[mask][:, 0].shape:', cross_bond_index.T[mask][:, 0].shape)
            #print('new_cross_bond_index[mask][:, 0].shape:', new_cross_bond_index[mask][:, 0].shape)

            tmp = new_cross_bond_index[mask]
            new_cross_bond_index[:, 0][mask] = (cross_bond_index.T[:, 0][mask] + ligand_atom_num_list[i])

            #print('new_cross_bond_index[mask][:, 0]:', new_cross_bond_index[mask][:, 0]) #0, , mask,。

    

            new_cross_bond_index[:, 1][mask] = cross_bond_index.T[:, 1][mask] + protein_atom_num_list[i]
            #print('new_cross_bond_index[mask]:', new_cross_bond_index[mask]) #0?
        cross_bond_index = new_cross_bond_index.T 

        #print('max(cross_bond_index.T[:, 0]):', torch.max(cross_bond_index.T[:, 0]))
        #print('max(cross_bond_index.T[:, 1]):', torch.max(cross_bond_index.T[:, 1]))

        #print('cross_bond_index.T, after:', cross_bond_index.T) ##0？

        #exit()

        #，
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i in range(g_num): #
            mask = protein_link_t_reverse_batch == i
            new_cross_bond_index_reverse[:, 0][mask] = cross_bond_index_reverse.T[:, 0][mask] + protein_atom_num_list[i]
            new_cross_bond_index_reverse[:, 1][mask] = cross_bond_index_reverse.T[:, 1][mask] + ligand_atom_num_list[i]

        cross_bond_index_reverse = new_cross_bond_index_reverse.T
        


        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #

        #1，，，，，? ，
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        
        #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
        #print('ligand_node_local:', ligand_node_local)
        #print('len(ligand_node_local):', len(ligand_node_local))
        #print('len(ligand_element):', len(ligand_element))
        #print('max(ligand_bond_index):', torch.max(ligand_bond_index)) #tensor(227, device='cuda:0')
        #print('ligand_bond_index.T:', ligand_bond_index.T) 
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 11  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        #indices = torch.cat([indices1, indices5, indices6])
        indices = torch.cat([indices1])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index




    # @staticmethod #self，，self，
    def _build_edge_type_interaction_20_gpu_optim_distance(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #cross_distance，，，
        #，，，cross_bond_distance， cross_bond_distance_reverse

        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 11  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)


        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index





    # @staticmethod #self，，self，
    def _build_edge_type_interaction_20_gpu_optim_distance_extend(self, x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
        batch, #mask_liagnd
        atom_isring,
        atom_isO,
        atom_isN,

        cross_isring_flag, 
        cross_isO_flag, 
        cross_isN_flag, 
        cross_lp_pos,
        cross_distance,

        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse
        ):
        #cross_distance，，，
        #，，，cross_bond_distance， cross_bond_distance_reverse

        #cross_bond_index_reverse, cross_bond_type_reverse,，
        #GPU tensor，
        #id
        #
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        new_ligand_bond_index = new_ligand_bond_index.T



        '''
        #，idid
        '''
        protein_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 0])))).cuda()
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()
        protein_node_global = protein_ligand_node_list[mask_ligand == 0] #pytorch2.0datas[index]，

        #id
        protein_node_local2global_dict = {}
        for k, v in zip(protein_node_local, protein_node_global):
            protein_node_local2global_dict[k.item()] = v


        #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
        #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
        #id
        new_cross_bond_index = torch.zeros(cross_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index.T): #N * 2
            new_cross_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index[i][1] = protein_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index = new_cross_bond_index.T


        #id_reverse
        new_cross_bond_index_reverse = torch.zeros(cross_bond_index_reverse.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(cross_bond_index_reverse.T): #N * 2
            new_cross_bond_index_reverse[i][0] = protein_node_local2global_dict[bd[0].item()] #
            new_cross_bond_index_reverse[i][1] = ligand_node_local2global_dict[bd[1].item()] #
        new_cross_bond_index_reverse = new_cross_bond_index_reverse.T


        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 11  #


        indices1 = (edge_type == 1).nonzero().view(-1) #0
        indices5 = (edge_type == 5).nonzero().view(-1) #0
        indices6 = (edge_type == 6).nonzero().view(-1) #0
        indices = torch.cat([indices1, indices5, indices6])
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        new_edge_type  = torch.cat([new_edge_type, ligand_bond_type, cross_bond_type, cross_bond_type_reverse], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index, new_cross_bond_index, new_cross_bond_index_reverse], dim = 1)

        #，12,13,14,15，2，1617，
        #，
        protein_index = protein_node_global
        #protein_pos   = x[protein_index]

        #new_cross_bond_index，3.5~4.5，<4.5，。2ai，
        #，，id
        #
        ligand_to_protein_index_dict = defaultdict(set)
        for ids_i, ids_j in new_cross_bond_index.T: #N*2
            ligand_to_protein_index_dict[ids_i].add(ids_j)

        extend_cross_bond_index = []
        # , ，
        for l_atom_index in ligand_to_protein_index_dict:
            p_atom_set = ligand_to_protein_index_dict[l_atom_index]

            #
            exit_protein_index = p_atom_set #
            #exit_protein_pos   = x[exit_protein_index]
            
            #，source_protein_index
            target_protein_index = set(protein_index) - set(exit_protein_index)
            #target_protein_pos   = x[target_protein_index]

            #exit_protein_indextarget_protein_index，2ai

            #，
            vec1 = torch.LongTensor(list(exit_protein_index)).cuda()
            vec2 = torch.LongTensor(list(target_protein_index)).cuda()
            #  torch.meshgrid 
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            
            #6ai, ，,
            dis_limit = 2.0

            dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
            dis_index = dis <= dis_limit #

            combination = combination.t()[dis_index] #k * 2
            combination = combination.t() # 2 * k
            extend_pro_atom_index = torch.unique(combination[1])


            #[l_atom_index]extend_pro_atom_index
            vec1 = torch.LongTensor([l_atom_index]).cuda()
            vec2 = extend_pro_atom_index.cuda()
            #  torch.meshgrid 
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            extend_cross_bond_index.append(combination)

        #12,13,14,15，2，1617，
        extend_cross_bond_index = torch.cat(extend_cross_bond_index, dim=-1)
        extend_cross_bond_type  = torch.full([extend_cross_bond_index.size(1)], 16, dtype = torch.int64).cuda()

        # 
        extend_cross_bond_index_reverse = extend_cross_bond_index[[1, 0], :] 
        extend_cross_bond_type_reverse  = torch.full([extend_cross_bond_index_reverse.size(1)], 17, dtype = torch.int64).cuda()

        #
        new_edge_type  = torch.cat([new_edge_type, extend_cross_bond_type, extend_cross_bond_type_reverse], dim = 0)  #
        new_edge_index = torch.cat([new_edge_index, extend_cross_bond_index, extend_cross_bond_index_reverse], dim = 1)

        edge_type_dim = F.one_hot(new_edge_type, num_classes=20) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index





    def truncate(self, arr, decimals):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        return int(arr * factor) #
        #return math.ceil(arr * factor) #
        #return np.round(arr, decimals)

    def truncate2(self, arr, decimals = 2):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        #return int(arr * factor) #
        #return math.ceil(arr * factor) #
        return np.round(arr, decimals)

    def combinations_optim(self, x, org_x, atom1, atom2, atom_index1, atom_index2, centor, 
            cross_atom_flag1, cross_atom_flag2, cross_distance, cross_ligand, cross_protein, 
            flag
            ):
            #cross_ligand_atom_flag，cross_protein_atom_flag，，O，N，, 1O, 2N, 3, 0
            #，。atom1, atom2id，atom_index1, atom_index2，bool
            #cross_ligand, cross_protein，x

            if GP.interaction_stype == 'interaction':
                #，
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]
                x1 = org_x[vec1]
                x2 = org_x[vec2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #key，indexvalue
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #，，small_cross_distance, 
                #cross_ligand_atom_flag, cross_protein_atom_flagO，N，，，O~N，
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                small_cross_protein  = cross_protein[cross_protein_atom_flag]


                #，small_cross_ligandsmall_cross_proteinx
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #list
                #print('protein_index:', protein_index)#list

                small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #，，8ai？shape = [n, m]
                
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 8ai，6ai，tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                for k in range(small_cross_distance_flag.size(0)):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #list
                #print('new_protein_index:', new_protein_index)#list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #，O,N，。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #，
                    #print('None') #None？
                    #combination = []


            elif GP.interaction_stype == 'interaction_all':
                #，，
                #4.5，4.5，o,n,，，，
                #vec1 = atom1[atom_index1]
                #vec2 = atom2[atom_index2]
                vec1 = atom1
                vec2 = atom2

                x1 = org_x[vec1]
                x2 = org_x[vec2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #key，indexvalue
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #，，small_cross_distance, 
                #cross_ligand_atom_flag, cross_protein_atom_flagO，N，，，O~N，
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                #small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                #small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                #small_cross_protein  = cross_protein[cross_protein_atom_flag]

                #
                small_cross_distance = cross_distance
                small_cross_ligand   = cross_ligand
                small_cross_protein  = cross_protein


                #，small_cross_ligandsmall_cross_proteinx
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #list
                #print('protein_index:', protein_index)#list

                small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #，，8ai？shape = [n, m]
                
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 8ai，6ai，tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                for k in range(small_cross_distance_flag.size(0)):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #list
                #print('new_protein_index:', new_protein_index)#list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #，O,N，。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #，
                    #print('None') #None？
                    #combination = []




            elif GP.interaction_stype == 'centor': # 
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]

                #6ai, ，,
                dis_limit = GP.interaction_distance  #8，

                if flag == 'ligand':
                    dis = torch.norm(centor - x[vec2], p = 2, dim = -1) #
                    dis_index = dis <= dis_limit #
                    new_vec2 = vec2[dis_index]
                    grid_x, grid_y = torch.meshgrid(vec1, new_vec2, indexing='ij') #grid_x, grid_y
                    # 
                    combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

                elif flag == 'protein':
                    dis = torch.norm(centor - x[vec1], p = 2, dim = -1) #
                    dis_index = dis <= dis_limit #
                    new_vec1 = vec1[dis_index]
                    grid_x, grid_y = torch.meshgrid(new_vec1, vec2, indexing='ij') #grid_x, grid_y
                    # 
                    combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

                # ，combination.t()，，，O, N, O, N, ，
                vec1 = torch.unique(combination[0]) # unique_consecutive，
                vec2 = torch.unique(combination[1])
                #  torch.meshgrid ， vec1, vec2id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

            elif GP.interaction_stype == 'all':
                #，
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

            elif GP.interaction_stype == 'distance':  #，，
                raise Exception('Unsupport combinations_optim, please change')
                '''
                grid_x, grid_y = torch.meshgrid(atom1, atom2, indexing='ij') #grid_x, grid_y
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]

                #，
                copy_combination = combination.clone()
                
                #，200
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                sorted_dis, sorted_indices = torch.sort(dis) # ，

                atom_num  = GP.min_distance_atom_num  #num
                dis_index = sorted_indices[:atom_num]  #

                combination = combination.t()[dis_index] ##shape = [2, n*m] -> [n*m, 2]

                #atom1, atom2，，combination，True，atom_index1, atom_index2
                new_atom_index1, new_atom_index2 = torch.zeros_like(atom_index1, dtype=torch.bool), torch.zeros_like(atom_index2, dtype=torch.bool)

                vec1 = torch.unique(combination.t()[0]) #，，
                vec2 = torch.unique(combination.t()[1])

                for i in vec1:
                    new_atom_index1[i] = True
                
                for i in vec2:
                    new_atom_index2[i] = True
                
                final_atom_index1, final_atom_index2 = new_atom_index1 & atom_index1, new_atom_index2 & atom_index2


                #，
                vec1 = atom1[final_atom_index1]
                vec2 = atom2[final_atom_index2]
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                '''


            else:
                raise Exception('Unsupport combinations_optim, please change')
                """
                vec1 = atom1[atom_index1]
                vec2 = atom2[atom_index2]

                #  torch.meshgrid ， vec1, vec2id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij') #grid_x, grid_y
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 

                #，
                copy_combination = combination.clone()
                
                #6ai, ，,
                dis_limit = GP.interaction_distance  #8，

                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]

                '''
                if len(combination) == 0:
                    #print('12')
                    dis_limit = 12.0 #，
                    combination = copy_combination.clone()
                    dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                    dis_index = dis <= dis_limit #

                    combination = combination.t()[dis_index]
                '''
                
                if len(combination) == 0:
                    #print('')
                    dis_limit = 100000000.0 #，
                    combination = copy_combination.clone()
                    dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                    dis_index = dis <= dis_limit #

                    combination = combination.t()[dis_index]

                '''
                #6ai, ，。60，，，
                nun_limit = 60

                dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #
                dis_index = dis <= dis_limit #

                combination = combination[dis_index].t()
                '''

                # ，combination.t()，，，O, N, O, N, ，
                vec1 = torch.unique(combination.t()[0]) #，，
                vec2 = torch.unique(combination.t()[1])
                #  torch.meshgrid ， vec1, vec2id
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m] 
                """
            #
            #if combination.shape[0] != 0:
                #combination = torch.unique(combination, dim = -1)
            return combination






    @staticmethod
    def _build_edge_type_82(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()
        
        '''
        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

        #KNN，，，
        #new_edge_type  = torch.cat([edge_type, ligand_bond_type], dim = 0)  #
        new_edge_type = edge_type #，，ligand_bond_typeKNN
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        #new_edge_index = torch.cat([edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index


        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_type = edge_type
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        #new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index


    @staticmethod
    def _build_edge_type_82_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #GPU tensor，
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).cuda()

    
        #id
        protein_ligand_node_list = torch.LongTensor(list(range(len(mask_ligand)))).cuda()

        ligand_node_global = protein_ligand_node_list[mask_ligand == 1] #pytorch2.0datas[index]，

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k.item()] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).cuda()
        for i, bd in enumerate(ligand_bond_index.T):
            #print('bd:', bd) #numpy bd: [0, 9], torch tensor， tensor([0, 9], device='cuda:0')
            #print('bd[0]:', bd[0]) #bd[0]: tensor(0, device='cuda:0')
            #print('bd[1]:', bd[1]) #bd[0]: tensor(9, device='cuda:0')
            #print('ligand_node_local:', ligand_node_local) #ligand_node_local: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19], device='cuda:0')
            #exit()
            #print('ligand_node_local2global_dict:', ligand_node_local2global_dict)
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0].item()] #tensork，，
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1].item()]
        
        new_ligand_bond_index = new_ligand_bond_index.T
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 1  #
        edge_type[n_src & ~n_dst]  = 5  #，
        edge_type[~n_src & n_dst]  = 6  #，
        edge_type[~n_src & ~n_dst] = 7  #


        indices = (edge_type == 1).nonzero().view(-1) #0
        #indices = (edge_type == 1).nonzero().squeeze() #0

        # 
        rows_to_remove = indices.detach().cpu().tolist()
        
        '''
        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''
        
        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_type = edge_type
        #new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        #new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=8) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index

    @staticmethod
    def _build_edge_type_4(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #4
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 0  #
        edge_type[n_src & ~n_dst]  = 1  #，
        edge_type[~n_src & n_dst]  = 2  #，
        edge_type[~n_src & ~n_dst] = 3  #


        indices = (edge_type == 0).nonzero().squeeze() #0，edge_type == 0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #new_edge_type  = torch.cat([new_edge_type, ligand_bond_type], dim = 0)  #
        new_edge_type  = torch.cat([new_edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_index = torch.cat([new_edge_index, new_ligand_bond_index], dim = 1)

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=4) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index
    





    @staticmethod
    def _build_edge_type_42(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #4
        #id
        ligand_node_local = torch.LongTensor(list(range(len(mask_ligand[mask_ligand == 1])))).numpy()

    
        #id
        protein_ligand_node_list = list(range(len(mask_ligand)))

        ligand_node_global = torch.LongTensor(protein_ligand_node_list)[mask_ligand == 1].numpy()

        ##print('ligand_node_list:', len(ligand_node_local))
        ##print('protein_ligand_node_list:', len(protein_ligand_node_list))
        ##print('ligand_node_global:', len(ligand_node_global))
        ##print('edge_index:', edge_index.shape)

        '''
        ligand_node_list: 282
        protein_ligand_node_list: 3388
        ligand_node_global: 282
        edge_index: torch.Size([2, 108416])
        '''

        #id
        ligand_node_local2global_dict = {}
        for k, v in zip(ligand_node_local, ligand_node_global):
            ligand_node_local2global_dict[k] = v
        

        #id
        new_ligand_bond_index = torch.zeros(ligand_bond_index.T.shape, dtype = torch.int64).numpy()
        for i, bd in enumerate(ligand_bond_index.T.detach().cpu().numpy()):
            ##print('bd:', bd) #bd: [0 1]
            new_ligand_bond_index[i][0] = ligand_node_local2global_dict[bd[0]]
            new_ligand_bond_index[i][1] = ligand_node_local2global_dict[bd[1]]
        
        new_ligand_bond_index = torch.from_numpy(new_ligand_bond_index.T).cuda()
        ##print('new_ligand_bond_index:', new_ligand_bond_index.shape) #torch.Size([2, 582])


        #raise Exception('test')
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.

        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst]   = 0  #
        edge_type[n_src & ~n_dst]  = 1  #，
        edge_type[~n_src & n_dst]  = 2  #，
        edge_type[~n_src & ~n_dst] = 3  #


        indices = (edge_type == 0).nonzero().squeeze() #0，edge_type == 0

        # 
        rows_to_remove = indices.detach().cpu().tolist()

        '''
        #  torch.index_select() ,
        indices_to_keep = torch.tensor(list(set(range(edge_type.size(0))) - set(rows_to_remove))).cuda()
        new_edge_type   = torch.index_select(edge_type, 0, indices_to_keep)
        new_edge_index  = torch.index_select(edge_index, 1, indices_to_keep)  #2 * N
        '''

        #
        ##print('ligand_bond_type:', ligand_bond_type)
        ##print('new_ligand_bond_index:', new_ligand_bond_index)

    
        #KNN，，，
        #new_edge_type  = torch.cat([edge_type, torch.zeros_like(ligand_bond_type, dtype = torch.int64)], dim = 0)#，0
        new_edge_type = edge_type #，，ligand_bond_typeKNN
        #new_edge_index = torch.cat([edge_index, new_ligand_bond_index], dim = 1)
        new_edge_index = edge_index

        #

        ##print('new_edge_type:', new_edge_type.shape)
        ##print('new_edge_index:', new_edge_index.shape)
        #new_edge_type: torch.Size([103536])
        #new_edge_index: torch.Size([2, 103536])

        #(edge_type,edge_index)，，idid
        #(edge_type,edge_index)，
        edge_type_dim = F.one_hot(new_edge_type, num_classes=4) #4，8,8，？
        #，
        return edge_type_dim, new_edge_index






    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst] = 0   #
        edge_type[n_src & ~n_dst] = 1  #，
        edge_type[~n_src & n_dst] = 2  #，
        edge_type[~n_src & ~n_dst] = 3 #

        
        edge_type = F.one_hot(edge_type, num_classes=20)
        return edge_type, edge_index


    @staticmethod
    def _build_edge_type_20(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch):
        #，
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1 #1
        n_dst = mask_ligand[dst] == 1

        #，，，，，？？？
        #edge_type，，0,1,2,3
        #
        edge_type[n_src & n_dst] = 0   #
        edge_type[n_src & ~n_dst] = 1  #，
        edge_type[~n_src & n_dst] = 2  #，
        edge_type[~n_src & ~n_dst] = 3 #

        
        edge_type = F.one_hot(edge_type, num_classes=20)
        return edge_type, edge_index




    def forward(self, h, x, org_x, element_all, mask_ligand, batch, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, atom_isring, atom_isO, atom_isN, 
                cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,
                coords_predict,
                complex_mol,

                protein_element_batch = None,
                protein_link_t_batch = None,
                protein_link_t_reverse_batch = None,
                ligand_element_batch = None,
                protein_element = None,
                ligand_element  = None,

                rd_x = None,

                sigmas = None, 
                protein_max_atom_num = None, ligand_max_atom_num  = None, args = None, return_all=False, fix_x=False, equiformer = False, escn = False):

        all_x = [x]
        all_h = [h]

        #print('org_h.shape:', h.shape) #torch.Size([874, 3136])
        
        '''
        #KNN，batchid,KNNbatch,，KNN，,KNN，
        #，，，。
        edge_index = self._connect_edge(org_x, mask_ligand, batch) #。，，。
        #KNNx，32
        #src, dst = edge_index

        # edge type (dim: 4)，
        edge_type, edge_index = self._build_edge_type(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch) #，。。？#？
        #？？？？？？edge_index
        src, dst = edge_index
        ##print('edge_index:', edge_index)
        ''' 
        atom_num_list = torch.tensor([600, 700, 800, 1000, 1200], dtype = torch.int64).cuda()

        for b_idx in range(self.num_blocks):
            #print('self.num_blocks:', self.num_blocks)
            
            '''
            edge_index = self._connect_edge(x, mask_ligand, batch) #。，，。
            #KNNx，32
            src, dst = edge_index

            # edge type (dim: 4)，
            edge_type = self._build_edge_type(edge_index, mask_ligand) #，。
            '''

            #，，x,org_x
            #，KNN？？？

            if GP.embedding3d:
                #edge_index = self._connect_edge(rd_x, mask_ligand, batch) #。，，。
                edge_index = self._connect_edge(rd_x, mask_ligand, batch)
            else:
                edge_index = self._connect_edge(x, mask_ligand, batch)

            if GP.embedding3d:
                '''3D'''
                '''rdkit，，，knn'''
                edge_type_, edge_index_, ligand_node_global2local_dict, protein_node_global2local_dict, only_ligand_edge_type_dim, only_ligand_bond_index, only_protein_edge_type_dim, only_protein_bond_index = self._build_edge_type_20_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch)


                #，id，，edge_index, 
                #
                #print('\n')
                '''，x, ，，，'''
                l_h             = h[mask_ligand == True]
                
                if GP.embedding3d_noise_pos:
                    l_x             = x[mask_ligand == True] 
                else:
                    l_x             = rd_x[mask_ligand == True] 
                

                #print('l_x.shape:', l_x.shape) #torch.Size([176, 3])

                #only_ligand_edge_type_dim, only_ligand_bond_index, only_protein_edge_type_dim, only_protein_bond_index
                l_element       = element_all[mask_ligand == True]
                #print('l_element.shape:', l_element.shape) #torch.Size([176])
                #print('edge_type.shape:', edge_type.shape) #torch.Size([42526, 20]) 
                l_edge_type     = only_ligand_edge_type_dim
                #print('l_edge_type.shape:', l_edge_type.shape) #torch.Size([376, 20]), ？
                l_edge_index    = only_ligand_bond_index
                #print('l_edge_index.shape:', l_edge_index.shape) #torch.Size([2, 376])
                ##print('l_edge_index:', l_edge_index) #0
                l_mask_ligand   = None #，
                l_batch         = batch[mask_ligand == True]
                #print('l_batch:', l_batch.shape) # torch.Size([176])
                l_fix_x         = None

                l_dst, l_src = l_edge_index

                
                

                if self.ew_net_type == 'global': #global，
                    l_dist = torch.norm(l_x[l_dst] - l_x[l_src], p=2, dim=-1, keepdim=True) #，，
                    #print('l_dist.shape:', l_dist.shape)
                    l_dist_feat = self.distance_expansion(l_dist) #
                    #print('l_dist_feat.shape:', l_dist_feat.shape)
                    l_logits = self.edge_pred_layer(l_dist_feat) #mlp，，
                    #print('l_logits.shape:', l_logits.shape)
                    l_e_w = torch.sigmoid(l_logits)
                else:
                    l_e_w = None
                
                #l_dist.shape: torch.Size([376, 1])
                #l_dist_feat.shape: torch.Size([376, 20])
                #l_logits.sjhape: torch.Size([376, 1])
                



                l_distance_vec = l_x[l_src] - l_x[l_dst]
                l_edge_dist    = l_distance_vec.norm(dim=-1)  #，，unimol
                #print('l_edge_dist.shape:', l_edge_dist.shape) #l_edge_dist.shape: torch.Size([376])


                l_h, _ = self.ligand_block(h = l_h, pos = l_x, distance_vec = l_distance_vec, edge_dist = l_edge_dist, element = l_element,
                                        edge_type = l_edge_type, edge_index = l_edge_index, mask_ligand = l_mask_ligand, 
                                        sigmas = sigmas, mask = None, batch = l_batch, 
                                        protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                        node_atom = None,
                                        e_w=l_e_w, fix_x=l_fix_x)
                
                #print('l_h.shape:', l_h.shape) #torch.Size([176, 200])
                

                #
                #print('\n')
                p_h             = h[mask_ligand == False]
                
                if GP.embedding3d_noise_pos:
                    p_x             = x[mask_ligand == False]
                else:
                    p_x             = rd_x[mask_ligand == False]


                #print('p_x.shape:', p_x.shape) #torch.Size([176, 3])

                #only_protein_edge_type_dim, only_protein_bond_index, only_protein_edge_type_dim, only_protein_bond_index
                p_element       = element_all[mask_ligand == False]
                #print('p_element.shape:', p_element.shape) #torch.Size([176])
                #print('edge_type.shape:', edge_type.shape) #torch.Size([42526, 20]) 
                p_edge_type     = only_protein_edge_type_dim
                #print('p_edge_type.shape:', p_edge_type.shape) #torch.Size([376, 20]), ？
                p_edge_index    = only_protein_bond_index
                #print('p_edge_index.shape:', p_edge_index.shape) #torch.Size([2, 376])
                ##print('p_edge_index:', p_edge_index) #0
                p_mask_ligand   = None #，
                p_batch         = batch[mask_ligand == False]
                #print('p_batch:', p_batch.shape) # torch.Size([176])
                p_fix_x         = None

                p_dst, p_src = p_edge_index

                
                

                if self.ew_net_type == 'global': #global，
                    p_dist = torch.norm(p_x[p_dst] - p_x[p_src], p=2, dim=-1, keepdim=True) #，，
                    #print('p_dist.shape:', p_dist.shape)
                    p_dist_feat = self.distance_expansion(p_dist) #
                    #print('p_dist_feat.shape:', p_dist_feat.shape)
                    p_logits = self.edge_pred_layer(p_dist_feat) #mlp，，
                    #print('p_logits.shape:', p_logits.shape)
                    p_e_w = torch.sigmoid(p_logits)
                else:
                    p_e_w = None


                p_distance_vec = p_x[p_src] - p_x[p_dst]
                p_edge_dist    = p_distance_vec.norm(dim=-1)  #，，unimol


                p_h, _ = self.protein_block(h = p_h, pos = p_x, distance_vec = p_distance_vec, edge_dist = p_edge_dist, element = p_element,
                                        edge_type = p_edge_type, edge_index = p_edge_index, mask_ligand = p_mask_ligand, 
                                        sigmas = sigmas, mask = None, batch = p_batch, 
                                        protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                        node_atom = None,
                                        e_w=p_e_w, fix_x=p_fix_x)



                #3d
                new_h = torch.empty(h.shape[0], l_h.shape[1] * 2).cuda()
                #print('h:', h.shape)
                #print('l_h:', l_h.shape)
                #print('h[mask_ligand == True]:', h[mask_ligand == True].shape)
                #print('mask_ligand:', mask_ligand.shape)
                '''
                h: torch.Size([1413, 200])                                                                                                                                                                             
                l_h: torch.Size([163, 200])                                                                                                                                                                            
                h[mask_ligand == True]: torch.Size([163, 200])                                                                                                                                                         
                mask_ligand: torch.Size([1413]) 
                '''
                new_h[mask_ligand == True]  = torch.cat([h[mask_ligand == True], l_h], dim = -1) #
                #RuntimeError: shape mismatch: value tensor of shape [163, 400] cannot be broadcast to indexing result of shape [163, 326]
                new_h[mask_ligand == False] = torch.cat([h[mask_ligand == False], p_h], dim = -1)


                
                #400 -> 200，
                h = self.linear_transform_dim(new_h)



            ''''''
            # # # ，edge_index（，，-），，
            # # #，
                
            #edge_type, edge_index = self._build_edge_type_8_gpu(edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch) #，。。？#？
            
            #，org_x
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu(org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, batch, atom_isring, atom_isO, atom_isN)
            
            ##coords_predictorg_x，，
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu(coords_predict, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, batch, atom_isring, atom_isO, atom_isN)
            #，，id
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim_v2(x, coords_predict, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
                    

            #unimol，
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance)



            #，，id
            #edge_type, edge_index = self._build_edge_type_interaction_8_gpu_optim_v2(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)


            #ON, ，。
            
            if GP.use_distance:
                edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                        batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,  
                        
                        protein_element_batch,
                        protein_link_t_batch,
                        protein_link_t_reverse_batch,
                        ligand_element_batch,
                        protein_element,
                        ligand_element,

                        )
            else:
                #
                edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_no_interactive_gpu(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                        batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                        cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse,  
                        
                        protein_element_batch,
                        protein_link_t_batch,
                        protein_link_t_reverse_batch,
                        ligand_element_batch,
                        protein_element,
                        ligand_element,

                        )
                
            
            
            
            
            #unimolequiformer
            #edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_distance(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
            

            #，3.5~4.5，，2ai，，，unimol，
            #，，
            #edge_type, edge_index = self._build_edge_type_interaction_20_gpu_optim_distance_extend(x, org_x, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch, 
                    #batch, atom_isring, atom_isO, atom_isN, cross_isring_flag, cross_isO_flag, cross_isN_flag, cross_lp_pos, cross_distance,
                    #cross_bond_index, cross_bond_type, cross_bond_index_reverse, cross_bond_type_reverse)
            
            #build_edge_type_interaction_8_gpuorg_x，，，32KNN
            #，.x + 
            
            src, dst = edge_index

            '''
            def _build_edge_type_interaction_8_gpu(self, edge_index, mask_ligand, ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,
            batch, #mask_liagnd
            atom_isring,
            atom_isO,
            atom_isN,
            ):
            '''


            distance_vec = x[src] - x[dst]
            edge_dist    = distance_vec.norm(dim=-1)  #，，unimol
            unimol_dist  = []


            if self.ew_net_type == 'global': #global，
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True) #，，
                dist_feat = self.distance_expansion(dist) #
                #print('dist_feat.shape:', dist_feat.shape)
                #print('dist_feat.dtype:', dist_feat.dtype) #logits.dtype: torch.float32
                logits = self.edge_pred_layer(dist_feat) #mlp，，
                #print('logits.shape:', logits.shape)
                #print('logits.dtype:', logits.dtype) #logits.dtype: torch.float32
                #exit()
                e_w = torch.sigmoid(logits)
            else:
                e_w = None
            
            #sigma_emb = self.sigma_emb_layer(sigmas.view(-1,1)).unsqueeze(1)
            #h = h + sigma_emb

            #print('h:', h.shape) #h: torch.Size([716, 3136]) ,  torch.Size([716, 49， 64]) 
            
            if self.equiformer == True:
                h, x = self.base_block(h = h, pos = x, distance_vec = distance_vec, edge_dist = edge_dist, element = element_all,
                                    edge_type = edge_type, edge_index = edge_index, mask_ligand = mask_ligand, 
                                    sigmas = sigmas, mask = None, batch = batch, 
                                    protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                    node_atom = None,
                                    e_w=e_w, fix_x=fix_x)
                
                #print('h, x:', h.shape, x.shape)
            
            elif self.escn == True:
                h, x = self.base_block(h = h, pos = x, distance_vec = distance_vec, edge_dist = edge_dist, element = element_all,
                                    edge_type = edge_type, edge_index = edge_index, mask_ligand = mask_ligand, 
                                    sigmas = sigmas, mask = None, batch = batch, 
                                    protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num, 
                                    node_atom = None,
                                    e_w=e_w, fix_x=fix_x)
                
            else:
                #
                for l_idx, layer in enumerate(self.base_block):
                    h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x) #，。？
                    

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs