import torch 
from ..comparm import * 
import pickle,os,tempfile, shutil, zipfile, time, math, time, tqdm 
from torch.utils.data import Dataset,DataLoader

from datetime import datetime
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_
from .gnn import GGNN,MLP,GraphGather
from .egnn import EGNN_Network
from .Equiformerv2 import Equiformer_encoder
from einops import rearrange

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)
    dim += value_expand_len
    #print (values.shape,indices.shape)
    return values.gather(dim, indices)

class Equi_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d3_net=Equiformer_encoder(
                    feat_dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    edge_dim=len(FGP.bond_types)+1,
                    output_dim=FGP.graph_hidden_dim,
                    num_layers=FGP.graph_depth,
                    num_heads=8,
                    ffn_hidden_channels=128,
                    edge_channels=64,
                    target="L1"
                )

        self.d3_mlp=MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                out_features=FGP.graph_hidden_dim,dropout_p=FGP.dropout
                )

        self.d3_gather = GraphGather(
                    node_features=FGP.graph_hidden_dim,
                    hidden_node_features=FGP.graph_hidden_dim,
                    out_features=FGP.graph_hidden_dim,
                    att_depth=FGP.graph_depth,
                    att_hidden_dim=FGP.graph_hidden_dim,
                    att_dropout_p=FGP.dropout,
                    emb_depth=FGP.graph_depth,
                    emb_hidden_dim=FGP.graph_hidden_dim,
                    emb_dropout_p=FGP.dropout
                )

    def forward(self,nodes,edges,coords,node_mask):
        d3_atom_emb = self.d3_net(nodes.float(),coords,edges.float(),node_mask.bool())
        d3_atom_emb_mlp = self.d3_mlp(d3_atom_emb)+d3_atom_emb
        d3_graph_emb = self.d3_gather(d3_atom_emb_mlp, d3_atom_emb, node_mask) 
        return d3_graph_emb,d3_atom_emb_mlp
    
class EGNN_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d3_egnn=EGNN_Network(depth=3,dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),edge_dim=len(FGP.bond_types)+1)
        self.d3_mlp=MLP(
                in_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                out_features=FGP.graph_hidden_dim,dropout_p=FGP.dropout
                )

        self.d3_gather = GraphGather(
                    node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    hidden_node_features=FGP.graph_hidden_dim,
                    out_features=FGP.graph_hidden_dim,
                    att_depth=FGP.graph_depth,
                    att_hidden_dim=FGP.graph_hidden_dim,
                    att_dropout_p=FGP.dropout,
                    emb_depth=FGP.graph_depth,
                    emb_hidden_dim=FGP.graph_hidden_dim,
                    emb_dropout_p=FGP.dropout
                )
    def forward(self,nodes,edges,coords,node_mask):
        d3_atom_emb,d3_coords = self.d3_egnn(nodes.float(),coords,edges.float())
        d3_atom_emb_mlp = self.d3_mlp(d3_atom_emb)
        d3_graph_emb = self.d3_gather(d3_atom_emb_mlp, d3_atom_emb, node_mask) 
        return d3_graph_emb,d3_atom_emb_mlp

class GGNN_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d2_gnn=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        ) 
        
    def forward(self,nodes,edges):
        d2_graph_emb,d2_atom_emb = self.d2_gnn(nodes.float(),edges.float())
        return d2_graph_emb,d2_atom_emb
    
class D3_Mix_Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #print 
        self.d2_gnn=GGNN_Net()
        if FGP.graph_3D_encoder_type=='egnn':
            self.d3_egnn=EGNN_Net()
        else:
            self.d3_egnn=Equi_Net()

    def forward(self,nodes,edges,coords,node_mask):
        d2_graph_emb,d2_atom_emb = self.d2_gnn(nodes.float(),edges.float())
        d3_graph_emb,d3_atom_emb = self.d3_egnn(nodes.float(), edges.float(), coords, node_mask)
        atom_emb=torch.cat((d2_atom_emb,d3_atom_emb),dim=-1)
        graph_emb=torch.cat((d2_graph_emb,d3_graph_emb),dim=-1)
        return graph_emb,atom_emb

class Node_adder_2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_molgnn=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )

        self.l_molgnn=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim*2,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )

        self.node_add_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=len(FGP.group_index_type_dict.keys()),
            dropout_p=FGP.dropout
            )

        self.node_terminate_net2= MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )
        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):
        c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],l_atom_embedding),dim=-1)
    
        graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)
        
        f_node_add1=self.node_add_net_1(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        f_node_add1=f_node_add1.view(-1,FGP.max_latoms*64)
        
        f_node_add2=torch.cat((f_node_add1,graph_embedding),dim=1)
        
        add_output=self.node_add_net_2(f_node_add2)
        
        terminate_output=self.node_terminate_net2(graph_embedding)
        
        action_output=torch.cat((add_output,terminate_output),dim=1)
        
        return action_output
 
class Node_adder_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_molgnn=D3_Mix_Block()
        self.p_molgnn=D3_Mix_Block()
        self.l_molgnn=D3_Mix_Block()

        
        self.node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim*6,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )

        self.node_add_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*6,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=len(FGP.group_index_type_dict.keys()),
            dropout_p=FGP.dropout
            )

        self.node_terminate_net2= MLP(
            in_features=FGP.graph_hidden_dim*6,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )
        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):
        c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges,complex_coords,complex_masks)
        p_graph_embedding,p_atom_embedding=self.p_molgnn(complex_nodes[:,:FGP.max_patoms],complex_edges[:,:FGP.max_patoms,:FGP.max_patoms],complex_coords[:,:FGP.max_patoms],complex_masks[:,:FGP.max_patoms])

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:],complex_coords[:,FGP.max_patoms:],complex_masks[:,FGP.max_patoms:])

        p_graph_embedding_for_atom=p_graph_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],l_atom_embedding,p_graph_embedding_for_atom),dim=-1)
        graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding,p_graph_embedding),dim=-1)
        f_node_add1=self.node_add_net_1(atom_embedding)
        f_node_add1=f_node_add1.view(-1,FGP.max_latoms*64)
        f_node_add2=torch.cat((f_node_add1,graph_embedding),dim=1)
        add_output=self.node_add_net_2(f_node_add2)
        terminate_output=self.node_terminate_net2(graph_embedding)
        action_output=torch.cat((add_output,terminate_output),dim=1)
        return action_output

class Node_adder_Equi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_molgnn=D3_Mix_Block()

        self.l_molgnn=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim*3,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )

        self.node_add_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=len(FGP.group_index_type_dict.keys()),
            dropout_p=FGP.dropout
            )

        self.node_terminate_net2= MLP(
            in_features=FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )
        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):
        c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges,complex_coords,complex_masks)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],l_atom_embedding),dim=-1)
    
        graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)
        
        f_node_add1=self.node_add_net_1(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        f_node_add1=f_node_add1.view(-1,FGP.max_latoms*64)
        
        f_node_add2=torch.cat((f_node_add1,graph_embedding),dim=1)
        
        add_output=self.node_add_net_2(f_node_add2)
        
        terminate_output=self.node_terminate_net2(graph_embedding)
        
        action_output=torch.cat((add_output,terminate_output),dim=1)
        
        return action_output

class Ring_gener_2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.l_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.r_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout)
        
        self.ring_node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim[1:]),
                dropout_p=FGP.dropout
            )

        self.ring_node_add_net_2 = MLP(
                in_features=np.prod(FGP.r_add_dim)+FGP.graph_hidden_dim*4,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim),
                dropout_p=FGP.dropout
            )

        self.ring_node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim[1:]),
            dropout_p=FGP.dropout)

        self.ring_node_connect_net_2 = MLP(
            in_features=np.prod(FGP.r_conn_dim)+FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim),
            dropout_p=FGP.dropout)

        self.ring_node_terminate_net_2 = MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
            )
        return 

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks):
        #print ('molnodes',molnodes.shape,moledges.shape)
        c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges)
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print ('ringnodes',ringnodes.shape,ringedges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        r_atom_emb=r_atom_emb*ring_masks.unsqueeze(-1).long()
        #print ('ringnode_embedding.shape',ringnode_embedding.shape)
        f_ring_node_add1=self.ring_node_add_net_1(r_atom_emb)
        f_ring_node_add1=f_ring_node_add1.view(-1,np.prod(FGP.r_add_dim))
        
        ft_embed=self.ft_emb(f_t)
        f_ring_node_add2=torch.cat((f_ring_node_add1,r_graph_emb,graph_emb,ft_embed),dim=1)
        
        add_output=self.ring_node_add_net_2(f_ring_node_add2)
        f_ring_node_connect1=self.ring_node_connect_net_1(r_atom_emb)
        
        f_ring_node_connect1=f_ring_node_connect1.view(-1,np.prod(FGP.r_conn_dim))
        f_ring_node_connect2=torch.cat((f_ring_node_connect1,r_graph_emb,graph_emb,ft_embed),dim=1)
        connect_output=self.ring_node_connect_net_2(f_ring_node_connect2)
        
        f_ring_node_terminate=torch.cat((r_graph_emb,ft_embed),dim=1)
        terminate_output=self.ring_node_terminate_net_2(f_ring_node_terminate)
        action_output=torch.cat((add_output,connect_output,terminate_output),dim=1)
        return action_output
 
class Ring_gener_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c_net=D3_Mix_Block()

        self.l_net=D3_Mix_Block()
            
        self.r_net=GGNN_Net()
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout)
        
        self.ring_node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim[1:]),
                dropout_p=FGP.dropout
            )

        self.ring_node_add_net_2 = MLP(
                in_features=np.prod(FGP.r_add_dim)+FGP.graph_hidden_dim*6,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim),
                dropout_p=FGP.dropout
            )

        self.ring_node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim[1:]),
            dropout_p=FGP.dropout)

        self.ring_node_connect_net_2 = MLP(
            in_features=np.prod(FGP.r_conn_dim)+FGP.graph_hidden_dim*6,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim),
            dropout_p=FGP.dropout)

        self.ring_node_terminate_net_2 = MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
            )
        return 

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks):
        #print ('molnodes',molnodes.shape,moledges.shape)
        c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:],complex_coords[:,FGP.max_patoms:],complex_masks[:,FGP.max_patoms:])
        atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)
        graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print ('ringnodes',ringnodes.shape,ringedges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        #print ('ringnode_embedding.shape',ringnode_embedding.shape)
        f_ring_node_add1=self.ring_node_add_net_1(r_atom_emb)
        f_ring_node_add1=f_ring_node_add1.view(-1,np.prod(FGP.r_add_dim))
        
        ft_embed=self.ft_emb(f_t)
        f_ring_node_add2=torch.cat((f_ring_node_add1,r_graph_emb,graph_emb,ft_embed),dim=1)
        
        add_output=self.ring_node_add_net_2(f_ring_node_add2)
        f_ring_node_connect1=self.ring_node_connect_net_1(r_atom_emb)
        
        f_ring_node_connect1=f_ring_node_connect1.view(-1,np.prod(FGP.r_conn_dim))
        f_ring_node_connect2=torch.cat((f_ring_node_connect1,r_graph_emb,graph_emb,ft_embed),dim=1)
        connect_output=self.ring_node_connect_net_2(f_ring_node_connect2)
        
        f_ring_node_terminate=torch.cat((r_graph_emb,ft_embed),dim=1)
        terminate_output=self.ring_node_terminate_net_2(f_ring_node_terminate)
        action_output=torch.cat((add_output,connect_output,terminate_output),dim=1)
        return action_output

class Ring_gener_Equi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_net=D3_Mix_Block()
        
        self.l_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.r_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout)
        
        self.ring_node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim[1:]),
                dropout_p=FGP.dropout
            )

        self.ring_node_add_net_2 = MLP(
                in_features=np.prod(FGP.r_add_dim)+FGP.graph_hidden_dim*5,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=np.prod(FGP.r_add_dim),
                dropout_p=FGP.dropout
            )

        self.ring_node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim[1:]),
            dropout_p=FGP.dropout)

        self.ring_node_connect_net_2 = MLP(
            in_features=np.prod(FGP.r_conn_dim)+FGP.graph_hidden_dim*5,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim),
            dropout_p=FGP.dropout)

        self.ring_node_terminate_net_2 = MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
            )
        return 

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks):
        #print ('molnodes',molnodes.shape,moledges.shape)
        c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print ('ringnodes',ringnodes.shape,ringedges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        r_atom_emb=r_atom_emb*ring_masks.unsqueeze(-1).long()
        #print ('ringnode_embedding.shape',ringnode_embedding.shape)
        f_ring_node_add1=self.ring_node_add_net_1(r_atom_emb)
        f_ring_node_add1=f_ring_node_add1.view(-1,np.prod(FGP.r_add_dim))
        
        ft_embed=self.ft_emb(f_t)
        f_ring_node_add2=torch.cat((f_ring_node_add1,r_graph_emb,graph_emb,ft_embed),dim=1)
        
        add_output=self.ring_node_add_net_2(f_ring_node_add2)
        f_ring_node_connect1=self.ring_node_connect_net_1(r_atom_emb)
        
        f_ring_node_connect1=f_ring_node_connect1.view(-1,np.prod(FGP.r_conn_dim))
        f_ring_node_connect2=torch.cat((f_ring_node_connect1,r_graph_emb,graph_emb,ft_embed),dim=1)
        connect_output=self.ring_node_connect_net_2(f_ring_node_connect2)
        
        f_ring_node_terminate=torch.cat((r_graph_emb,ft_embed),dim=1)
        terminate_output=self.ring_node_terminate_net_2(f_ring_node_terminate)
        action_output=torch.cat((add_output,connect_output,terminate_output),dim=1)
        return action_output
 
class Node_conner_2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        graph_hidden_dim=FGP.graph_hidden_dim
        graph_message_size=FGP.graph_message_size
        graph_message_passes=FGP.graph_message_passes
        graph_depth=FGP.graph_depth
        dropout=FGP.dropout
        """

        graph_hidden_dim=256
        graph_message_size=256
        graph_message_passes=3
        graph_depth=4
        dropout=0.0

        self.c_molgnn=GGNN(hidden_node_features=graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=graph_message_size,
                        message_passes=graph_message_passes,
                        hidden_dim=graph_hidden_dim,
                        module_depth=graph_depth,
                        dropout=dropout
                        )
        
        self.l_molgnn=GGNN(hidden_node_features=graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=graph_message_size,
                        message_passes=graph_message_passes,
                        hidden_dim=graph_hidden_dim,
                        module_depth=graph_depth,
                        dropout=dropout
                        )
        
        self.r_net=GGNN(hidden_node_features=graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=graph_message_size,
                        message_passes=graph_message_passes,
                        hidden_dim=graph_hidden_dim,
                        module_depth=graph_depth,
                        dropout=dropout
                        )

        self.node_connect_net_1 = MLP(
            in_features=graph_hidden_dim*2,
            hidden_layer_sizes=[graph_hidden_dim*2] * graph_depth,
            out_features=64,
            dropout_p=dropout)

        self.node_connect_net_2 = MLP(
            in_features=64*FGP.max_latoms+graph_hidden_dim*4,
            hidden_layer_sizes=[graph_hidden_dim] * graph_depth,
            out_features=np.prod(FGP.leaf_conn_dim),
            dropout_p=dropout)

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks):
        c_graph_emb,c_atom_emb=self.c_molgnn(complex_nodes.float(),complex_edges.float())
        
        l_graph_emb,l_atom_emb=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        cond_graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print (ring_nodes.shape,ring_edges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        focused_atom_embedding=batched_index_select(r_atom_emb,focused_ids.long()).squeeze(1)
        
        ligand_atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)
        #print (ligand_atom_emb.shape)
        f_node_connect1=self.node_connect_net_1(ligand_atom_emb)
        
        f_node_connect1=f_node_connect1.view(-1,FGP.max_latoms*64)
        
        f_ring_node_connect2=torch.cat((f_node_connect1,r_graph_emb,cond_graph_emb,focused_atom_embedding),dim=1)
        
        connect_output=self.node_connect_net_2(f_ring_node_connect2)
        
        return connect_output 

class Node_conner_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_molgnn=D3_Mix_Block()
        
        self.p_molgnn=D3_Mix_Block()
        
        self.l_molgnn=D3_Mix_Block()

        self.r_net=GGNN_Net()

        self.node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim*6,
            hidden_layer_sizes=[FGP.graph_hidden_dim*2] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)

        self.node_connect_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*8,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_conn_dim),
            dropout_p=FGP.dropout)

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks):
        c_graph_emb,c_atom_emb=self.c_molgnn(complex_nodes.float(),complex_edges.float(),complex_coords.float(),complex_masks)
        
        p_graph_emb,p_atom_emb=self.p_molgnn(complex_nodes[:,:FGP.max_patoms],complex_edges[:,:FGP.max_patoms,:FGP.max_patoms],complex_coords[:,:FGP.max_patoms],complex_masks[:,:FGP.max_patoms])
        
        l_graph_emb,l_atom_emb=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:],complex_coords[:,FGP.max_patoms:],complex_masks[:,FGP.max_patoms:])
        
        cond_graph_emb=torch.cat((c_graph_emb,l_graph_emb,p_graph_emb),dim=-1)
        
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        focused_atom_embedding=batched_index_select(r_atom_emb,focused_ids.long()).squeeze(1)
        
        p_graph_emb_for_atom=p_graph_emb.unsqueeze(1).tile(1,FGP.max_latoms,1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        ligand_atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb,p_graph_emb_for_atom),dim=-1)
        #print (ligand_atom_emb.shape)
        f_node_connect1=self.node_connect_net_1(ligand_atom_emb)
        
        f_node_connect1=f_node_connect1.view(-1,FGP.max_latoms*64)
        
        f_ring_node_connect2=torch.cat((f_node_connect1,r_graph_emb,cond_graph_emb,focused_atom_embedding),dim=1)
        
        connect_output=self.node_connect_net_2(f_ring_node_connect2)
        
        return connect_output

class Node_conner_Equi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_molgnn=D3_Mix_Block()
        
        self.l_molgnn=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        
        self.r_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )

        self.node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim*2] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)

        self.node_connect_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*5,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_conn_dim),
            dropout_p=FGP.dropout)

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks):
        c_graph_emb,c_atom_emb=self.c_molgnn(complex_nodes.float(),complex_edges.float(),complex_coords.float(),complex_masks)
        
        l_graph_emb,l_atom_emb=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        cond_graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print (ring_nodes.shape,ring_edges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        focused_atom_embedding=batched_index_select(r_atom_emb,focused_ids.long()).squeeze(1)
        
        ligand_atom_emb=torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)
        #print (ligand_atom_emb.shape)
        f_node_connect1=self.node_connect_net_1(ligand_atom_emb)
        
        f_node_connect1=f_node_connect1.view(-1,FGP.max_latoms*64)
        
        f_ring_node_connect2=torch.cat((f_node_connect1,r_graph_emb,cond_graph_emb,focused_atom_embedding),dim=1)
        
        connect_output=self.node_connect_net_2(f_ring_node_connect2)
        
        return connect_output

class Node_int_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_graph_net=GGNN_Net()
        self.complex_coord_net=EGNN_Net()
        
        self.pgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim*2,
            dropout_p=FGP.dropout
        )
        
        self.pgroups_gather=GraphGather(
            node_features=FGP.graph_hidden_dim*2,
            hidden_node_features=FGP.graph_hidden_dim*2,
            out_features=FGP.graph_hidden_dim,
            att_depth=FGP.graph_depth,
            att_hidden_dim=FGP.graph_hidden_dim,
            att_dropout_p=FGP.dropout,
            emb_depth=FGP.graph_depth,
            emb_hidden_dim=FGP.graph_hidden_dim,
            emb_dropout_p=FGP.dropout
        )
        
        self.lgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim*2,
            dropout_p=FGP.dropout
        )
        
        self.lgroups_gather=GraphGather(
            node_features=FGP.graph_hidden_dim*2,
            hidden_node_features=FGP.graph_hidden_dim*2,
            out_features=FGP.graph_hidden_dim,
            att_depth=FGP.graph_depth,
            att_hidden_dim=FGP.graph_hidden_dim,
            att_dropout_p=FGP.dropout,
            emb_depth=FGP.graph_depth,
            emb_hidden_dim=FGP.graph_hidden_dim,
            emb_dropout_p=FGP.dropout
        )
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout) 
        
        self.node_int_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)

        self.node_int_net_2 = MLP(
            in_features=64*FGP.max_pgroups+FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_int_add_dim),
            dropout_p=FGP.dropout)
        
        self.node_int_term_net_2=MLP(
            in_features=FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )

    def forward(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,\
            pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes):
        complex_2D_graph_emb,complex_2D_atom_emb=self.complex_graph_net(complex_2D_nodes.float(),complex_2D_edges.float())
        complex_3D_graph_emb,complex_3D_atom_emb=self.complex_coord_net(complex_3D_nodes.float(),complex_3D_edges,complex_3D_coords.float(),complex_3D_masks)
        complex_graph_emb=torch.cat((complex_2D_graph_emb,complex_3D_graph_emb),dim=-1)
        complex_atom_emb=torch.cat((complex_2D_atom_emb,complex_3D_atom_emb),dim=-1)
        complex_pgroups_emb=batched_index_select(complex_atom_emb,pgroups).squeeze(1)
        focus_lgroups_emb=batched_index_select(complex_atom_emb,focus_lgroups).squeeze(1)
        
        complex_pgroups_emb=complex_pgroups_emb*pgroups_masks.unsqueeze(-1).float()
        complex_pgroups_emb_mlp=self.pgroups_mlp(complex_pgroups_emb)*pgroups_masks.unsqueeze(-1).float()
        b=complex_pgroups_emb_mlp.shape[0]
        d=complex_pgroups_emb_mlp.shape[1]
        
        complex_pgroups_emb_mlp=rearrange(complex_pgroups_emb_mlp,'b d c h -> (b d) c h')
        complex_pgroups_emb=rearrange(complex_pgroups_emb,'b d c h -> (b d) c h')
        pgroups_masks=rearrange(pgroups_masks,'b d c-> (b d) c')
        
        complex_pgroups_emb_gather=self.pgroups_gather(complex_pgroups_emb_mlp,complex_pgroups_emb,pgroups_masks)
        complex_pgroups_emb_gather=rearrange(complex_pgroups_emb_gather,'(b d) h -> b d h',b=b)*pgroups_int_masks.unsqueeze(-1).long()
        #print ('complex_pgroups_emb_gather',complex_pgroups_emb_gather.shape) 
        
        focus_lgroups_emb_mlp=self.lgroups_mlp(focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).float()
        #print (focus_lgroups_emb.shape,focus_lgroups_emb_mlp.shape,)
        focus_lgroups_emb_gather=self.lgroups_gather(focus_lgroups_emb_mlp,focus_lgroups_emb,focus_lgroups_masks)
        #print (focus_lgroups_emb_gather.shape)
        #input_pl_pairs=torch.cat((complex_pgroups_emb_gather,focus_lgroups_emb_gather.tile(1,1,)),dim=1)
        f_node_int1=self.node_int_net_1(complex_pgroups_emb_gather)
        
        f_node_int1=f_node_int1.view(-1,FGP.max_pgroups*64)
        ft_embed=self.ft_emb(focus_ftypes)
        
        f_node_int2=torch.cat((f_node_int1,complex_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_add_output=self.node_int_net_2(f_node_int2)
        
        f_node_int3=torch.cat((complex_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_term_output=self.node_int_term_net_2(f_node_int3)
        
        int_output=torch.cat((f_node_int_add_output,f_node_int_term_output),dim=1)
        return int_output
    
class Node_int_Equi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_graph_net=GGNN_Net()
        self.complex_coord_net=Equi_Net()
        
        self.pgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim*2,
            dropout_p=FGP.dropout
        )
        
        self.pgroups_gather=GraphGather(
            node_features=FGP.graph_hidden_dim*2,
            hidden_node_features=FGP.graph_hidden_dim*2,
            out_features=FGP.graph_hidden_dim,
            att_depth=FGP.graph_depth,
            att_hidden_dim=FGP.graph_hidden_dim,
            att_dropout_p=FGP.dropout,
            emb_depth=FGP.graph_depth,
            emb_hidden_dim=FGP.graph_hidden_dim,
            emb_dropout_p=FGP.dropout
        )
        
        self.lgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim*2,
            dropout_p=FGP.dropout
        )
        
        self.lgroups_gather=GraphGather(
            node_features=FGP.graph_hidden_dim*2,
            hidden_node_features=FGP.graph_hidden_dim*2,
            out_features=FGP.graph_hidden_dim,
            att_depth=FGP.graph_depth,
            att_hidden_dim=FGP.graph_hidden_dim,
            att_dropout_p=FGP.dropout,
            emb_depth=FGP.graph_depth,
            emb_hidden_dim=FGP.graph_hidden_dim,
            emb_dropout_p=FGP.dropout
        )
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout) 
        
        self.node_int_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)

        self.node_int_net_2 = MLP(
            in_features=64*FGP.max_pgroups+FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_int_add_dim),
            dropout_p=FGP.dropout)
        
        self.node_int_term_net_2=MLP(
            in_features=FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )

    def forward(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,\
            pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes):
        complex_2D_graph_emb,complex_2D_atom_emb=self.complex_graph_net(complex_2D_nodes.float(),complex_2D_edges.float())
        complex_3D_graph_emb,complex_3D_atom_emb=self.complex_coord_net(complex_3D_nodes.float(),complex_3D_edges,complex_3D_coords.float(),complex_3D_masks)
        complex_graph_emb=torch.cat((complex_2D_graph_emb,complex_3D_graph_emb),dim=-1)
        complex_atom_emb=torch.cat((complex_2D_atom_emb,complex_3D_atom_emb),dim=-1)
        complex_pgroups_emb=batched_index_select(complex_atom_emb,pgroups).squeeze(1)
        focus_lgroups_emb=batched_index_select(complex_atom_emb,focus_lgroups).squeeze(1)
        
        complex_pgroups_emb=complex_pgroups_emb*pgroups_masks.unsqueeze(-1).float()
        complex_pgroups_emb_mlp=self.pgroups_mlp(complex_pgroups_emb)*pgroups_masks.unsqueeze(-1).float()
        b=complex_pgroups_emb_mlp.shape[0]
        d=complex_pgroups_emb_mlp.shape[1]
        
        complex_pgroups_emb_mlp=rearrange(complex_pgroups_emb_mlp,'b d c h -> (b d) c h')
        complex_pgroups_emb=rearrange(complex_pgroups_emb,'b d c h -> (b d) c h')
        pgroups_masks=rearrange(pgroups_masks,'b d c-> (b d) c')
        
        complex_pgroups_emb_gather=self.pgroups_gather(complex_pgroups_emb_mlp,complex_pgroups_emb,pgroups_masks)
        complex_pgroups_emb_gather=rearrange(complex_pgroups_emb_gather,'(b d) h -> b d h',b=b)*pgroups_int_masks.unsqueeze(-1).long()
        #print ('complex_pgroups_emb_gather',complex_pgroups_emb_gather.shape) 
        
        focus_lgroups_emb_mlp=self.lgroups_mlp(focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).float()
        #print (focus_lgroups_emb.shape,focus_lgroups_emb_mlp.shape,)
        focus_lgroups_emb_gather=self.lgroups_gather(focus_lgroups_emb_mlp,focus_lgroups_emb,focus_lgroups_masks)
        #print (focus_lgroups_emb_gather.shape)
        #input_pl_pairs=torch.cat((complex_pgroups_emb_gather,focus_lgroups_emb_gather.tile(1,1,)),dim=1)
        f_node_int1=self.node_int_net_1(complex_pgroups_emb_gather)
        
        f_node_int1=f_node_int1.view(-1,FGP.max_pgroups*64)
        ft_embed=self.ft_emb(focus_ftypes)
        
        f_node_int2=torch.cat((f_node_int1,complex_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_add_output=self.node_int_net_2(f_node_int2)
        
        f_node_int3=torch.cat((complex_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_term_output=self.node_int_term_net_2(f_node_int3)
        
        int_output=torch.cat((f_node_int_add_output,f_node_int_term_output),dim=1)
        return int_output
    
    
class Graph_terminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        graph_hidden_dim=256
        graph_message_size=256
        graph_message_passes=3
        graph_depth=4
        dropout=0.0
        self.c_molgnn=GGNN(hidden_node_features=graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=graph_message_size,
                        message_passes=graph_message_passes,
                        hidden_dim=graph_hidden_dim,
                        module_depth=graph_depth,
                        dropout=dropout
                        )

        self.l_molgnn=GGNN(hidden_node_features=graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=graph_message_size,
                        message_passes=graph_message_passes,
                        hidden_dim=graph_hidden_dim,
                        module_depth=graph_depth,
                        dropout=dropout
                        )
        
        self.node_add_net_1 = MLP(
                in_features=graph_hidden_dim*2,
                hidden_layer_sizes=[graph_hidden_dim] * graph_depth,
                out_features=64,
                dropout_p=dropout
            )

        self.node_add_net_2 = MLP(
            in_features=64*FGP.max_latoms+graph_hidden_dim*2,
            hidden_layer_sizes=[graph_hidden_dim] * graph_depth,
            out_features=len(FGP.group_index_type_dict.keys()),
            dropout_p=dropout
            )

        self.node_terminate_net2= MLP(
            in_features=graph_hidden_dim*2,
            hidden_layer_sizes=[graph_hidden_dim] * graph_depth,
            out_features=1,
            dropout_p=dropout
        )
        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):
        c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],l_atom_embedding),dim=-1)
    
        graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)
        
        f_node_add1=self.node_add_net_1(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        f_node_add1=f_node_add1.view(-1,FGP.max_latoms*64)
        
        f_node_add2=torch.cat((f_node_add1,graph_embedding),dim=1)
        
        add_output=self.node_add_net_2(f_node_add2)
        
        terminate_output=self.node_terminate_net2(graph_embedding)
        
        action_output=torch.cat((add_output,terminate_output),dim=1)
        
        return action_output  
 