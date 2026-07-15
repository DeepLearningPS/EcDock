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
from .modules import * 

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim,embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim=embed_dim
        self.query = torch.nn.Linear(input_dim, embed_dim)
        self.key = torch.nn.Linear(input_dim, embed_dim)
        self.value = torch.nn.Linear(input_dim, embed_dim)
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        #print ('queries.shape',queries.shape)
        #print ('keys.shape',keys.shape)
        #print ('values.shape',values.shape)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_dim ** 0.5)+1e-6
        #print('scores.shape',scores.shape)
        attention = self.softmax(scores)
        #print ('attention.shape',attention[0])
        weighted = torch.bmm(attention, values)
        #print ('weighted.shape',weighted.shape)
        return weighted
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        #print ('q.shape',q.shape)
        #print ('k.shape',k.shape)
        #print ('v.shape',v.shape)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o    

class Self_EncoderBlock(torch.nn.Module):
    def __init__(self,input_dim,embed_dim,num_splits):
        super().__init__()
        assert input_dim % num_splits == 0, "Embedding dimension must be 0 modulo number of heads."
        unit_input_dim=input_dim//num_splits
        self.num_splits=num_splits        
        self.self_attn=SelfAttention(input_dim=unit_input_dim,embed_dim=embed_dim)
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(FGP.dropout),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim) 
        )
        return 

    def forward (self,x):
        if self.num_splits>1:
            x=x.chunk(self.num_splits,dim=-1)
            #print ('x input.shape',[xi.shape for xi in x])
            x=torch.cat([xi.unsqueeze(-2) for xi in x],dim=-2)
            #print ('x stack.shape',x.shape)
            b,n,s,d=x.shape
            x=rearrange(x,'b n s d -> (b n) s d')
            
        x=self.self_attn(x)
        
        if self.num_splits>1:
            x=torch.mean(x,dim=-2)
            x=rearrange(x,'(b n) d -> b n d',b=b,n=n)
            #print (x.shape)
        linear_out=self.linear_net(x)
        x=x+linear_out
        return x

class MHA_EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        
        super().__init__()
        
        # Attention layer
        self.mh_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        
        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        #print ('x in MHA',x.shape)
        attn_out = self.mh_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x
 
class Node_adder_MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FGP.graph_3D_encoder:
            self.c_molgnn=D3_Mix_Block()
        else:
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

        if FGP.graph_3D_encoder:
            self.self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*6,embed_dim=FGP.graph_hidden_dim,num_splits=6)
        else:
            self.self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*4,embed_dim=FGP.graph_hidden_dim,num_splits=4)
            
        self.mha_blocks=torch.nn.ModuleList()
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.mha_blocks.append(block)
        
        self.output_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,embed_dim=len(FGP.group_index_type_dict.keys())+1,num_splits=1)

        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):
        if FGP.graph_3D_encoder:
            c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges,complex_coords,complex_masks)
        else:
            c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        c_graph_embedding_for_atom=c_graph_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)
        l_graph_embedding_for_atom=l_graph_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)
        #print (c_graph_embedding_for_atom.shape)
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],
                                  l_atom_embedding,
                                  c_graph_embedding_for_atom,
                                  l_graph_embedding_for_atom),dim=-1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
    
        #graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)
        atom_embedding=self.self_attn_block(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        for i in range(FGP.MHA_depth):
            atom_embedding=self.mha_blocks[i](atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        #print ('embed in MHA',atom_embedding.shape)
        add_output=self.output_self_attn_block(atom_embedding)
        add_output=torch.mean(add_output,dim=-2)
        return add_output

class Ring_gener_MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if FGP.graph_3D_encoder:
            self.c_net=D3_Mix_Block()
        else:
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
        if FGP.graph_3D_encoder:
            self.radd_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*6,embed_dim=FGP.graph_hidden_dim,num_splits=6)
        else:
            self.radd_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*5,embed_dim=FGP.graph_hidden_dim,num_splits=5)
            
        self.radd_mha_blocks=torch.nn.ModuleList()
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.radd_mha_blocks.append(block)
        
        self.radd_output_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,
                                                      embed_dim=np.prod(FGP.r_add_dim[1:]),
                                                      num_splits=1)
        
        if FGP.graph_3D_encoder:
            self.rconn_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*6,embed_dim=FGP.graph_hidden_dim,num_splits=6)
        else:
            self.rconn_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*5,embed_dim=FGP.graph_hidden_dim,num_splits=5)
            
        
        self.rconn_mha_blocks=torch.nn.ModuleList()
        
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.rconn_mha_blocks.append(block)
        
        self.rconn_output_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,
                                                      embed_dim=np.prod(FGP.r_conn_dim[1:]),
                                                      num_splits=1)

        self.ring_node_terminate_net_2 = MLP(
            in_features=FGP.graph_hidden_dim*2,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
            )
        return 

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks):
        #print ('molnodes',molnodes.shape,moledges.shape)
        if FGP.graph_3D_encoder:
            c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges,complex_coords,complex_masks)
        else:
            c_graph_emb,c_atom_emb=self.c_net(complex_nodes,complex_edges)
            
        l_graph_emb,l_atom_emb=self.l_net(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        #graph_emb=torch.cat((c_graph_emb,l_graph_emb),dim=-1)
        #print ('ringnodes',ringnodes.shape,ringedges.shape)
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        c_graph_emb_for_ratoms=c_graph_emb.unsqueeze(1).tile(1,FGP.max_lgroup_size,1)
        l_graph_emb_for_ratoms=l_graph_emb.unsqueeze(1).tile(1,FGP.max_lgroup_size,1)
        r_graph_emb_for_ratoms=r_graph_emb.unsqueeze(1).tile(1,FGP.max_lgroup_size,1)
        ft_embed=self.ft_emb(f_t)
        ft_embed_for_ratoms=ft_embed.unsqueeze(1).tile(1,FGP.max_lgroup_size,1)
        
        r_atom_emb=torch.cat((r_atom_emb,
                              c_graph_emb_for_ratoms,
                              l_graph_emb_for_ratoms,
                              r_graph_emb_for_ratoms,
                              ft_embed_for_ratoms
                              ),dim=-1)*ring_masks.unsqueeze(-1).long()
        #print ('r_atom_emb',r_atom_emb.shape)
        radd_atom_emb=self.radd_self_attn_block(r_atom_emb)*ring_masks.unsqueeze(-1).long()
        for i in range(FGP.MHA_depth):
            radd_atom_emb=self.radd_mha_blocks[i](radd_atom_emb)*ring_masks.unsqueeze(-1).long()
        radd_atom_emb=self.radd_output_block(radd_atom_emb)*ring_masks.unsqueeze(-1).long()
        #print (radd_atom_emb.shape)
        radd_output=torch.cat(torch.chunk(radd_atom_emb,FGP.max_lgroup_size,dim=-2),dim=-1).squeeze(1)
        #print (radd_output.shape)
        rconn_atom_emb=self.rconn_self_attn_block(r_atom_emb)*ring_masks.unsqueeze(-1).long()
        for i in range(FGP.MHA_depth):
            rconn_atom_emb=self.rconn_mha_blocks[i](rconn_atom_emb)*ring_masks.unsqueeze(-1).long()
        rconn_atom_emb=self.rconn_output_block(rconn_atom_emb)*ring_masks.unsqueeze(-1).long()
        #print (rconn_atom_emb.shape)
        rconn_output=torch.cat(torch.chunk(rconn_atom_emb,FGP.max_lgroup_size,dim=-2),dim=-1).squeeze(1)
        #print (rconn_output.shape)
        f_ring_node_terminate=torch.cat((r_graph_emb,ft_embed),dim=1)
        terminate_output=self.ring_node_terminate_net_2(f_ring_node_terminate)
        action_output=torch.cat((radd_output,rconn_output,terminate_output),dim=-1)
        return action_output 
    
class Node_conner_MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #if FGP.graph_3D_encoder:
        #    self.c_molgnn=D3_Mix_Block()
        #else:
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
        
        self.r_net=GGNN(hidden_node_features=FGP.graph_hidden_dim,
                        n_node_features=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                        n_edge_features=len(FGP.bond_types)+1,
                        message_size=FGP.graph_message_size,
                        message_passes=FGP.graph_message_passes,
                        hidden_dim=FGP.graph_hidden_dim,
                        module_depth=FGP.graph_depth,
                        dropout=FGP.dropout
                        )
        #if FGP.graph_3D_encoder:
        #    self.nconn_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*8,embed_dim=FGP.graph_hidden_dim,num_splits=8)
        #else:
        self.nconn_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*6,embed_dim=FGP.graph_hidden_dim,num_splits=6)
            
        
        self.nconn_mha_blocks=torch.nn.ModuleList()
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.nconn_mha_blocks.append(block)
        
        self.nconn_output_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,
                                                      embed_dim=4,
                                                      num_splits=1)

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks):
        #if FGP.graph_3D_encoder:
        #    c_graph_emb,c_atom_emb=self.c_molgnn(complex_nodes,complex_edges,complex_coords,complex_masks)
        #else:
        c_graph_emb,c_atom_emb=self.c_molgnn(complex_nodes,complex_edges)
            
        l_graph_emb,l_atom_emb=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        
        r_graph_emb,r_atom_emb=self.r_net(ring_nodes.float(),ring_edges.float())
        
        c_graph_emb_for_latoms=c_graph_emb.unsqueeze(1).tile(1,FGP.max_latoms,1)
        l_graph_emb_for_latoms=l_graph_emb.unsqueeze(1).tile(1,FGP.max_latoms,1)
        r_graph_emb_for_latoms=r_graph_emb.unsqueeze(1).tile(1,FGP.max_latoms,1)
        
        #print (ring_nodes.shape,ring_edges.shape)
        
        focused_atom_embedding=batched_index_select(r_atom_emb,focused_ids.long()).squeeze(1)
        focused_atom_embedding_for_latoms=focused_atom_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)
        
        ligand_atom_emb=torch.cat(( c_atom_emb[:,FGP.max_patoms:],
                                    l_atom_emb,
                                    c_graph_emb_for_latoms,
                                    l_graph_emb_for_latoms,
                                    r_graph_emb_for_latoms,
                                    focused_atom_embedding_for_latoms
                                   ),dim=-1)
        
        #print (ligand_atom_emb.shape)
        ligand_atom_emb=self.nconn_self_attn_block(ligand_atom_emb)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()

        for i in range(FGP.MHA_depth):
            ligand_atom_emb=self.nconn_mha_blocks[i](ligand_atom_emb)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        
        nconn_output=self.nconn_output_block(ligand_atom_emb)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        #print (nconn_output.shape)
        nconn_output=torch.cat(torch.chunk(nconn_output,FGP.max_latoms,dim=-2),dim=-1).squeeze(1)
        #print (nconn_output.shape)

        return nconn_output

class Node_int_MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_graph_net=GGNN_Net()
        self.complex_coord_net=Equi_Net()
        
        self.pgroups_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*2,embed_dim=FGP.graph_hidden_dim,num_splits=2) 
        
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
        
        self.lgroups_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*2,embed_dim=FGP.graph_hidden_dim,num_splits=2) 
        
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
        
        self.ft_emb=MLP(in_features=FGP.n_group_feats,
                        hidden_layer_sizes=[FGP.graph_hidden_dim]*FGP.graph_depth,
                        out_features=FGP.graph_hidden_dim,
                        dropout_p=FGP.dropout) 
        
        self.plpair_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*5,embed_dim=FGP.graph_hidden_dim,num_splits=5)
        
        self.plpair_mha_blocks=torch.nn.ModuleList()    
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.plpair_mha_blocks.append(block)
            
        self.plpair_output_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,
                                                      embed_dim=FGP.leaf_int_add_dim[1],
                                                      num_splits=1) 
        
        self.node_int_term_net=MLP(
            in_features=FGP.graph_hidden_dim*4,
            hidden_layer_sizes=[FGP.graph_hidden_dim] * FGP.graph_depth,
            out_features=1,
            dropout_p=FGP.dropout
        )

    def forward(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,\
            pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes):
        complex_2D_graph_emb,complex_2D_atom_emb=self.complex_graph_net(complex_2D_nodes.float(),complex_2D_edges.float())
        complex_3D_graph_emb,complex_3D_atom_emb=self.complex_coord_net(complex_3D_nodes.float(),complex_3D_edges,complex_3D_coords.float(),complex_3D_masks)
        complex_2D_atom_emb=complex_2D_atom_emb*complex_2D_masks.unsqueeze(-1).long()
        complex_3D_atom_emb=complex_3D_atom_emb*complex_3D_masks.unsqueeze(-1).long()
        
        complex_graph_emb=torch.cat((complex_2D_graph_emb,complex_3D_graph_emb),dim=-1)
        complex_atom_emb=torch.cat((complex_2D_atom_emb,complex_3D_atom_emb),dim=-1)
        
        complex_pgroups_emb=batched_index_select(complex_atom_emb,pgroups).squeeze(1)
        focus_lgroups_emb=batched_index_select(complex_atom_emb,focus_lgroups).squeeze(1)
        
        complex_pgroups_emb=complex_pgroups_emb*pgroups_masks.unsqueeze(-1).float()
        b=complex_pgroups_emb.shape[0]
        d=complex_pgroups_emb.shape[1]
        complex_pgroups_emb=rearrange(complex_pgroups_emb,'b d c h -> (b d) c h')
        pgroups_masks=rearrange(pgroups_masks,'b d c -> (b d) c')
        complex_pgroups_emb=self.pgroups_self_attn_block(complex_pgroups_emb)*pgroups_masks.unsqueeze(-1).long()
        complex_pgroups_emb_mlp=self.pgroups_mlp(complex_pgroups_emb)*pgroups_masks.unsqueeze(-1).float()
        complex_pgroups_emb_gather=self.pgroups_gather(complex_pgroups_emb_mlp,complex_pgroups_emb,pgroups_masks)
        #print (complex_pgroups_emb_gather.shape)
        complex_pgroups_emb_gather=rearrange(complex_pgroups_emb_gather,'(b d) h -> b d h',b=b,d=d)
        
        #print (focus_lgroups_emb.shape,focus_lgroups_masks.shape)
        focus_lgroups_emb=self.lgroups_self_attn_block(focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).long()
        #print (focus_lgroups_emb.shape)
        focus_lgroups_emb_mlp=self.lgroups_mlp(focus_lgroups_emb)*focus_lgroups_masks.unsqueeze(-1).float()
        #print (focus_lgroups_emb.shape,focus_lgroups_emb_mlp.shape,)
        focus_lgroups_emb_gather=self.lgroups_gather(focus_lgroups_emb_mlp,focus_lgroups_emb,focus_lgroups_masks)
        #print ('focus_lgroups_emb_gather',focus_lgroups_emb_gather.shape)
        focus_lgroups_emb_gather_=focus_lgroups_emb_gather.unsqueeze(1).tile(1,FGP.max_pgroups,1) 
        complex_graph_emb_=complex_graph_emb.unsqueeze(1).tile(1,FGP.max_pgroups,1)
        ft_embed=self.ft_emb(focus_ftypes)
        ft_embed_=ft_embed.unsqueeze(1).tile(1,FGP.max_pgroups,1)
        
        #print (complex_pgroups_emb_gather.shape)
        #print (focus_lgroups_emb_gather_.shape,complex_graph_emb_.shape,ft_embed_.shape)
        
        input_pl_pairs=torch.cat((complex_pgroups_emb_gather,
                                  focus_lgroups_emb_gather_,
                                  complex_graph_emb_,
                                  ft_embed_),dim=-1)
        #print ('input_pl',input_pl_pairs.shape)
        input_pl_pairs=self.plpair_self_attn_block(input_pl_pairs)*pgroups_int_masks.unsqueeze(-1).long()
        for i in range(FGP.MHA_depth):
            input_pl_pairs=self.plpair_mha_blocks[i](input_pl_pairs)*pgroups_int_masks.unsqueeze(-1).long()
        
        input_pl_pairs=input_pl_pairs*pgroups_int_masks.unsqueeze(-1).long()
        input_pl_pairs=self.plpair_output_block(input_pl_pairs)*pgroups_int_masks.unsqueeze(-1).long()
        input_pl_pairs=torch.cat(torch.chunk(input_pl_pairs,FGP.max_pgroups,dim=-2),dim=-1).squeeze(1)
        #print (complex_graph_emb.shape,focus_lgroups_emb_gather.shape,ft_embed.shape)
        f_node_int3=torch.cat((complex_graph_emb,focus_lgroups_emb_gather,ft_embed),dim=1)
        f_node_int_term_output=self.node_int_term_net(f_node_int3)
        
        int_output=torch.cat((input_pl_pairs,f_node_int_term_output),dim=1)
        return int_output

class Graph_terminator_MHA(torch.nn.Module):
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

        self.self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim*4,embed_dim=FGP.graph_hidden_dim,num_splits=4)
            
        self.mha_blocks=torch.nn.ModuleList()
        for i in range(FGP.MHA_depth):
            block=MHA_EncoderBlock(input_dim=FGP.graph_hidden_dim,dim_feedforward=FGP.graph_hidden_dim,
                                                num_heads=4,dropout=0.0)
            self.mha_blocks.append(block)
        
        self.output_self_attn_block=Self_EncoderBlock(input_dim=FGP.graph_hidden_dim,embed_dim=len(FGP.group_index_type_dict.keys())+1,num_splits=1)

        return

    def forward(self,complex_nodes,complex_edges,complex_coords,complex_masks):

        c_graph_embedding,c_atom_embedding=self.c_molgnn(complex_nodes,complex_edges)

        l_graph_embedding,l_atom_embedding=self.l_molgnn(complex_nodes[:,FGP.max_patoms:],complex_edges[:,FGP.max_patoms:,FGP.max_patoms:])
        c_graph_embedding_for_atom=c_graph_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)
        l_graph_embedding_for_atom=l_graph_embedding.unsqueeze(1).tile(1,FGP.max_latoms,1)
        #print (c_graph_embedding_for_atom.shape)
        atom_embedding=torch.cat((c_atom_embedding[:,FGP.max_patoms:],
                                  l_atom_embedding,
                                  c_graph_embedding_for_atom,
                                  l_graph_embedding_for_atom),dim=-1)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
    
        #graph_embedding=torch.cat((c_graph_embedding,l_graph_embedding),dim=-1)
        atom_embedding=self.self_attn_block(atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        for i in range(FGP.MHA_depth):
            atom_embedding=self.mha_blocks[i](atom_embedding)*complex_masks[:,FGP.max_patoms:].unsqueeze(-1).long()
        #print ('embed in MHA',atom_embedding.shape)
        add_output=self.output_self_attn_block(atom_embedding)
        add_output=torch.mean(add_output,dim=-2)
        return add_output