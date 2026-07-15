import torch 
from ..comparm import * 

from datetime import datetime
from einops import rearrange
import torch.nn as nn 
from .modules import GGNN_Net,Equi_Net
from .gnn.modules import MLP, GraphGather
from .modules import batched_index_select
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Node_adder_ctrl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FGP.with_ctrlnet_for_nadd:
            self.c_ctrl_net = Equi_Net()
            self.c_ctrl_net_atom_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))
            self.c_ctrl_net_graph_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        self.l_molgnn=GGNN_Net()

        self.node_add_net_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )

        self.node_add_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=len(FGP.group_index_type_dict.keys()),
            dropout_p=FGP.dropout
            )

        self.node_terminate_net2= MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )
        if FGP.with_ctrlnet_for_nadd:
            self.node_add_net_ctrl_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )
        
            self.node_add_net_ctrl_zero_mlp_1 = zero_module(torch.nn.Linear(64,64))

        if FGP.freeze_backbone_for_nadd:
            #self.c_molgnn.requires_grad_(False)
            self.l_molgnn.requires_grad_(False)
            self.node_add_net_1.requires_grad_(False)
            self.node_add_net_2.requires_grad_(False)
            self.node_terminate_net2.requires_grad_(False)
        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):

        if FGP.with_ctrlnet_for_nadd:
            ctrl_graph_embedding, ctrl_atom_embedding=self.c_ctrl_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        
            ctrl_atom_embedding=self.c_ctrl_net_atom_zero_mlp(ctrl_atom_embedding)[:,FGP.max_patoms:]   

            ctrl_graph_embedding=self.c_ctrl_net_graph_zero_mlp(ctrl_graph_embedding)

        #c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        unctrl_atom_embedding=l_atom_embedding
    
        unctrl_graph_embedding=l_graph_embedding #torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)

        if FGP.with_ctrlnet_for_nadd:

            atom_embedding=ctrl_atom_embedding*0.25+unctrl_atom_embedding

            graph_embedding=ctrl_graph_embedding*0.25+unctrl_graph_embedding
        else:
            atom_embedding=unctrl_atom_embedding
            graph_embedding=unctrl_graph_embedding

        graph_f_node_add1=self.node_add_net_1(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        graph_f_node_add1=graph_f_node_add1.view(-1,FGP.max_latoms*64)

        if FGP.with_ctrlnet_for_nadd:
            ctrl_f_node_add_1=self.node_add_net_ctrl_1(ctrl_atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
            
            ctrl_f_node_add_1=self.node_add_net_ctrl_zero_mlp_1(ctrl_f_node_add_1)
            
            ctrl_f_node_add_1=ctrl_f_node_add_1.view(-1,FGP.max_latoms*64)            

            f_node_add_1=ctrl_f_node_add_1*0.5+graph_f_node_add1

            f_node_add2=torch.cat((f_node_add_1,graph_embedding),dim=1)

        else:
            f_node_nadd1=graph_f_node_add1

            f_node_add2=torch.cat((f_node_nadd1,graph_embedding),dim=1)
        
        add_output=self.node_add_net_2(f_node_add2)
        
        terminate_output=self.node_terminate_net2(unctrl_graph_embedding)
        
        action_output=torch.cat((add_output,terminate_output),dim=1)
        
        return action_output



class Ring_gener_ctrl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FGP.with_ctrlnet_for_rgen:
            self.c_ctrl_net = Equi_Net()
            self.c_ctrl_net_atom_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))
            self.c_ctrl_net_graph_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        #self.c_net=GGNN_Net()
        
        self.l_net=GGNN_Net()
        
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
                in_features=np.prod(FGP.r_add_dim)+FGP.graph_hidden_dim*3,
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
            in_features=np.prod(FGP.r_conn_dim)+FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.r_conn_dim),
            dropout_p=FGP.dropout)

        self.ring_node_terminate_net_2 = MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
            )
        
        if FGP.freeze_backbone_for_rgen:
            #self.c_net.requires_grad_(False)
            self.l_net.requires_grad_(False)
            self.r_net.requires_grad_(False)
            self.ring_node_add_net_1.requires_grad_(False)
            self.ring_node_add_net_2.requires_grad_(False)
            self.ring_node_connect_net_1.requires_grad_(False)
            self.ring_node_connect_net_2.requires_grad_(False)
            self.ring_node_terminate_net_2.requires_grad_(False)
        return 

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks):
        if FGP.with_ctrlnet_for_rgen:
            ctrl_graph_emb, ctrl_atom_emb=self.c_ctrl_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        
            ctrl_atom_emb=self.c_ctrl_net_atom_zero_mlp(ctrl_atom_emb)[:,FGP.max_patoms:]   

            ctrl_graph_emb=self.c_ctrl_net_graph_zero_mlp(ctrl_graph_emb)

        #print ('molnodes',molnodes.shape,moledges.shape)
        #c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges)
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        unctrl_atom_emb=l_atom_emb #torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        unctrl_graph_emb=l_graph_emb #torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        
        if FGP.with_ctrlnet_for_rgen:
            atom_emb=ctrl_atom_emb*0.25+unctrl_atom_emb
            graph_emb=ctrl_graph_emb*0.25+unctrl_graph_emb
        else:
            atom_emb=unctrl_atom_emb
            graph_emb=unctrl_graph_emb

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
    
class Node_connect_ctrl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FGP.with_ctrlnet_for_nconn:
            self.c_ctrl_net = Equi_Net()
            self.c_ctrl_net_atom_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))
            self.c_ctrl_net_graph_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        #self.c_net=GGNN_Net()
        
        self.l_net=GGNN_Net()
        
        self.r_net=GGNN_Net()

        self.node_connect_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim*2] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)

        self.node_connect_net_2 = MLP(
            in_features=64*FGP.max_latoms+FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_conn_dim),
            dropout_p=FGP.dropout)

        if FGP.with_ctrlnet_for_nconn:
            self.node_connect_net_ctrl_1 = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout
            )
        
            self.node_connect_net_ctrl_zero_mlp_1 = zero_module(torch.nn.Linear(64,64))
        
        if FGP.freeze_backbone_for_nconn:
            #self.c_net.requires_grad_(False)
            self.l_net.requires_grad_(False)
            self.r_net.requires_grad_(False)
            self.node_connect_net_1.requires_grad_(False)
            self.node_connect_net_2.requires_grad_(False)

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks):
        if FGP.with_ctrlnet_for_nconn:
            ctrl_graph_emb, ctrl_atom_emb=self.c_ctrl_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        
            ctrl_atom_emb=self.c_ctrl_net_atom_zero_mlp(ctrl_atom_emb)[:,FGP.max_patoms:]   

            ctrl_graph_emb=self.c_ctrl_net_graph_zero_mlp(ctrl_graph_emb)
        
        #c_graph_emb,c_atom_emb=self.c_net(complex_nodes.float(),complex_edges.float())
        
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        unctrl_cond_graph_emb=l_graph_emb #torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print (ring_nodes.shape,ring_edges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        focused_atom_embedding=batched_index_select(r_atom_emb,focused_ids.long()).squeeze(1)
        
        unctrl_ligand_atom_emb= l_atom_emb #torch.cat((c_atom_emb[:,FGP.max_patoms:],l_atom_emb),dim=-1)
        
        if FGP.with_ctrlnet_for_nconn:
            ligand_atom_emb=ctrl_atom_emb*0.25+unctrl_ligand_atom_emb
            cond_graph_emb=ctrl_graph_emb*0.25+unctrl_cond_graph_emb
        else:
            ligand_atom_emb=unctrl_ligand_atom_emb
            cond_graph_emb=unctrl_cond_graph_emb

        #print (ligand_atom_emb.shape)
        f_node_connect_1=self.node_connect_net_1(ligand_atom_emb)
        
        f_node_connect_1=f_node_connect_1.view(-1,FGP.max_latoms*64)

        if FGP.with_ctrlnet_for_nconn:
            ctrl_f_node_connect_1=self.node_connect_net_ctrl_1(ctrl_atom_emb)
            
            ctrl_f_node_connect_1=self.node_connect_net_ctrl_zero_mlp_1(ctrl_f_node_connect_1)
            
            ctrl_f_node_connect_1=ctrl_f_node_connect_1.view(-1,FGP.max_latoms*64)            

            f_node_connect_1=ctrl_f_node_connect_1*0.5+f_node_connect_1

        
        f_ring_node_connect2=torch.cat((f_node_connect_1,r_graph_emb,cond_graph_emb,focused_atom_embedding),dim=1)
        
        connect_output=self.node_connect_net_2(f_ring_node_connect2)
        
        return connect_output 
    

class Node_int_ctrl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_graph_net=GGNN_Net()
        if FGP.with_ctrlnet_for_nint:
            self.complex_ctrl_net=Equi_Net()
            self.complex_ctrl_net_graph_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        self.pgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim,
            dropout_p=FGP.dropout
        )
        
        self.pgroups_gather=GraphGather(
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
        if FGP.with_ctrlnet_for_nint:
            self.pgroups_mlp_ctrl=MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=FGP.graph_hidden_dim,
                dropout_p=FGP.dropout
            )
            self.pgroups_gather_ctrl=GraphGather(
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
            self.pgroups_gather_ctrl_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        
        self.lgroups_mlp=MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=FGP.graph_hidden_dim,
            dropout_p=FGP.dropout
        )
        
        self.lgroups_gather=GraphGather(
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
        if FGP.with_ctrlnet_for_nint:
            self.lgroups_mlp_ctrl=MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=FGP.graph_hidden_dim,
                dropout_p=FGP.dropout
            )
            self.lgroups_gather_ctrl=GraphGather(
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
            self.lgroups_gather_ctrl_zero_mlp=zero_module(torch.nn.Linear(FGP.graph_hidden_dim,FGP.graph_hidden_dim))

        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout) 
        
        self.node_int_net_1 = MLP(
            in_features=FGP.graph_hidden_dim,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=64,
            dropout_p=FGP.dropout)
        
        if FGP.with_ctrlnet_for_nint:
            self.node_int_net_1_ctrl = MLP(
                in_features=FGP.graph_hidden_dim,
                hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
                out_features=64,
                dropout_p=FGP.dropout)
            
            self.node_int_net_1_ctrl_zero_mlp=zero_module(torch.nn.Linear(64,64))
            
        self.node_int_net_2 = MLP(
            in_features=64*FGP.max_pgroups+FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=np.prod(FGP.leaf_int_add_dim),
            dropout_p=FGP.dropout)
        
        self.node_int_term_net_2=MLP(
            in_features=FGP.graph_hidden_dim*3,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )

    def forward(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,\
            pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes):
        
        complex_2D_graph_emb,complex_2D_atom_emb=self.complex_graph_net(complex_2D_nodes.float(),complex_2D_edges.float())
        
        if FGP.with_ctrlnet_for_nint:
            complex_3D_graph_emb,complex_3D_atom_emb=self.complex_ctrl_net(complex_3D_nodes.float(),complex_3D_edges,complex_3D_coords.float(),complex_3D_masks)

        #complex_graph_emb=torch.cat((complex_2D_graph_emb,complex_3D_graph_emb),dim=-1)
        #complex_atom_emb=torch.cat((complex_2D_atom_emb,complex_3D_atom_emb),dim=-1)
        complex_pgroups_emb=batched_index_select(complex_2D_atom_emb,pgroups).squeeze(1)
        focus_lgroups_emb=batched_index_select(complex_2D_atom_emb,focus_lgroups).squeeze(1)
        if FGP.with_ctrlnet_for_nint:
            ctrl_pgroups_emb=batched_index_select(complex_3D_atom_emb,pgroups).squeeze(1)
            ctrl_focus_lgroups_emb=batched_index_select(complex_3D_atom_emb,focus_lgroups).squeeze(1)

        complex_pgroups_emb=complex_pgroups_emb*pgroups_masks.unsqueeze(-1).float()
        complex_pgroups_emb_mlp=self.pgroups_mlp(complex_pgroups_emb)*pgroups_masks.unsqueeze(-1).float()
        if FGP.with_ctrlnet_for_nint:
            ctrl_pgroups_emb=ctrl_pgroups_emb*pgroups_masks.unsqueeze(-1).float()
            ctrl_pgroups_emb_mlp=self.pgroups_mlp_ctrl(ctrl_pgroups_emb)*pgroups_masks.unsqueeze(-1).float()

        b=complex_pgroups_emb_mlp.shape[0]
        d=complex_pgroups_emb_mlp.shape[1]
        
        complex_pgroups_emb_mlp=rearrange(complex_pgroups_emb_mlp,'b d c h -> (b d) c h')
        complex_pgroups_emb=rearrange(complex_pgroups_emb,'b d c h -> (b d) c h')
        #print ('complex_pgroups_emb_gather',complex_pgroups_emb_gather.shape) 

        if FGP.with_ctrlnet_for_nint:
            ctrl_pgroups_emb_mlp=rearrange(ctrl_pgroups_emb_mlp,'b d c h -> (b d) c h')
            ctrl_pgroups_emb=rearrange(ctrl_pgroups_emb,'b d c h -> (b d) c h')
        
        pgroups_masks=rearrange(pgroups_masks,'b d c-> (b d) c')
        complex_pgroups_emb_gather=self.pgroups_gather(complex_pgroups_emb_mlp,complex_pgroups_emb,pgroups_masks)
        complex_pgroups_emb_gather=rearrange(complex_pgroups_emb_gather,'(b d) h -> b d h',b=b)*pgroups_int_masks.unsqueeze(-1).long()
        if FGP.with_ctrlnet_for_nint:
            ctrl_pgroups_emb_gather=self.pgroups_gather_ctrl(ctrl_pgroups_emb_mlp,ctrl_pgroups_emb,pgroups_masks)
            ctrl_pgroups_emb_gather=self.pgroups_gather_ctrl_zero_mlp(ctrl_pgroups_emb_gather)
            ctrl_pgroups_emb_gather=rearrange(ctrl_pgroups_emb_gather,'(b d) h -> b d h',b=b)*pgroups_int_masks.unsqueeze(-1).long()        

        if FGP.with_ctrlnet_for_nint:
            pgroups_emb_gather=ctrl_pgroups_emb_gather*0.25+complex_pgroups_emb_gather
        else:
            pgroups_emb_gather=complex_pgroups_emb_gather

        focus_lgroups_emb_mlp=self.lgroups_mlp(focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).float()
        #print (focus_lgroups_emb.shape,focus_lgroups_emb_mlp.shape,)
        
        focus_lgroups_emb_gather=self.lgroups_gather(focus_lgroups_emb_mlp,focus_lgroups_emb,focus_lgroups_masks)
        
        if FGP.with_ctrlnet_for_nint:
            ctrl_focus_lgroups_emb_mlp=self.lgroups_mlp_ctrl(ctrl_focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).float()
            ctrl_focus_lgroups_emb_gather=self.lgroups_gather_ctrl(ctrl_focus_lgroups_emb_mlp,ctrl_focus_lgroups_emb,focus_lgroups_masks)
            ctrl_focus_lgroups_emb_gather=self.lgroups_gather_ctrl_zero_mlp(ctrl_focus_lgroups_emb_gather)
        
        if FGP.with_ctrlnet_for_nint:
            lgroups_emb_gather=ctrl_focus_lgroups_emb_gather*0.25+focus_lgroups_emb_gather
        else:
            lgroups_emb_gather=focus_lgroups_emb_gather

        #print (focus_lgroups_emb_gather.shape)
        #input_pl_pairs=torch.cat((complex_pgroups_emb_gather,focus_lgroups_emb_gather.tile(1,1,)),dim=1)
        unctrl_f_node_int1=self.node_int_net_1(pgroups_emb_gather)
        unctrl_f_node_int1=unctrl_f_node_int1.view(-1,FGP.max_pgroups*64)
        if FGP.with_ctrlnet_for_nint:
            ctrl_f_node_int1=self.node_int_net_1_ctrl(ctrl_pgroups_emb_gather)
            ctrl_f_node_int1=self.node_int_net_1_ctrl_zero_mlp(ctrl_f_node_int1)
            ctrl_f_node_int1=ctrl_f_node_int1.view(-1,FGP.max_pgroups*64)
        
        if FGP.with_ctrlnet_for_nint:
            f_node_int1=ctrl_f_node_int1*0.5+unctrl_f_node_int1
        else:
            f_node_int1=unctrl_f_node_int1

        ft_embed=self.ft_emb(focus_ftypes)

        if FGP.with_ctrlnet_for_nint:
            complex_ctrl_graph_emb=self.complex_ctrl_net_graph_zero_mlp(complex_3D_graph_emb)
            complex_graph_emb=complex_ctrl_graph_emb*0.25+complex_2D_graph_emb
        else:
            complex_graph_emb=complex_2D_graph_emb

        f_node_int2=torch.cat((f_node_int1,complex_graph_emb,lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_add_output=self.node_int_net_2(f_node_int2)
        
        f_node_int3=torch.cat((complex_2D_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_term_output=self.node_int_term_net_2(f_node_int3)
        
        int_output=torch.cat((f_node_int_add_output,f_node_int_term_output),dim=1)
        return int_output