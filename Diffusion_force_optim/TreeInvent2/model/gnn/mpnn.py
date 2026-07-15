"""
Defines specific MPNN implementations.
"""
# load general packages and functions
from collections import namedtuple
import math
import torch

# load GraphINVENT-specific functions
from .aggregation_mpnn import AggregationMPNN
from .edge_mpnn import *
from .summation_mpnn import *
from .modules import *
class MNN(SummationMPNN):
    """
    The "message neural network" model.
    """
    def __init__(self,hidden_node_features=100,n_edge_features=4, message_size=100, message_passes=3) -> None:
        super().__init__()
        
        #FGP.modelsetting       = constants
        self.hidden_node_features = hidden_node_features
        self.n_edge_features      = n_edge_features
        self.message_size         = message_size
        self.message_passes         = message_passes
        message_weights      = torch.Tensor(message_size,
                                            hidden_node_features,
                                            n_edge_features)
        if self.device == "cuda":
            message_weights = message_weights.to("cuda", non_blocking=True)

        self.message_weights = torch.nn.Parameter(message_weights)

        self.gru             = torch.nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdev = 1.0 / math.sqrt(self.message_weights.size(1))
        self.message_weights.data.uniform_(-stdev, stdev)

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        edges_view            = edges.view(-1, 1, 1, self.n_edge_features)
        weights_for_each_edge = (edges_view * self.message_weights.unsqueeze(0)).sum(3)
        return torch.matmul(weights_for_each_edge,
                            node_neighbours.unsqueeze(-1)).squeeze()

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = torch.sum(hidden_nodes, dim=1)
        return graph_embeddings


class S2V(SummationMPNN):
    """
    The "set2vec" model.
    """
    def __init__(self,hidden_node_features=100,
                        n_edge_features=4,
                        message_size=100,
                        message_passes=3,
                        enn_hidden_dim=250,
                        enn_depth=4,
                        enn_dropout_p=0.0,
                        s2v_lstm_computations=3,
                        s2v_memory_size=100) -> None:

        super().__init__()
        self.hidden_node_features=hidden_node_features
        self.n_edge_features=n_edge_features
        self.message_size=message_size
        self.message_passes=message_passes
        self.enn_hidden_dim=enn_hidden_dim
        self.enn_depth=enn_depth
        self.enn_dropout_p=enn_dropout_p
        self.s2v_lstm_computations=s2v_lstm_computations
        self.s2v_memory_size=s2v_memory_size
        #FGP.modelsetting  = constants

        self.enn        = MLP(
            in_features=self.n_edge_features,
            hidden_layer_sizes=[self.enn_hidden_dim] * self.enn_depth,
            out_features=self.hidden_node_features * self.message_size,
            dropout_p=self.enn_dropout_p
        )

        self.gru        = torch.nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True
        )

        self.s2v        = Set2Vec(
            node_features=self.n_node_features,
            hidden_node_features=self.hidden_node_features,
            lstm_computations=self.s2v_lstm_computations,
            memory_size=self.s2v_memory_size
        )

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        enn_output = self.enn(edges)
        matrices   = enn_output.view(-1,
                                     self.message_size,
                                     self.hidden_node_features)
        msg_terms  = torch.matmul(matrices,
                                  node_neighbours.unsqueeze(-1)).squeeze(-1)
        return msg_terms

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        #output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return graph_embeddings


class AttentionS2V(AggregationMPNN):
    """
    The "set2vec with attention" model.
    """
    def __init__(self,  hidden_node_features=100,
                        n_edge_features=4,
                        message_size=100,
                        message_passes=3,
                        enn_hidden_dim=250,
                        enn_depth=4,
                        enn_dropout_p=0.0,
                        s2v_lstm_computations=3,
                        s2v_memory_size=100,
                        att_hidden_dim=250,
                        att_depth=4,
                        att_dropout_p=0.0,
                        big_positive=1e6,
                        big_negative=-1e6
                ) -> None:

        super().__init__()
        self.hidden_node_features=hidden_node_features
        self.n_edge_features=n_edge_features
        self.message_size=message_size
        self.message_passes=message_passes
        self.enn_hidden_dim=enn_hidden_dim
        self.enn_depth=enn_depth
        self.enn_dropout_p=enn_dropout_p
        self.s2v_lstm_computations=s2v_lstm_computations
        self.s2v_memory_size=s2v_memory_size
        self.att_hidden_dim=att_hidden_dim
        self.att_depth=att_depth
        self.att_dropout_p=att_dropout_p
        self.big_positive=big_positive
        self.big_negative=big_negative
        self.device=device

        self.enn        = MLP(
            in_features = self.n_edge_features,
            hidden_layer_sizes = [self.enn_hidden_dim] * self.enn_depth,
            out_features = self.hidden_node_features * self.message_size,
            dropout_p = self.enn_dropout_p
        )

        self.att_enn    = MLP(
            in_features = self.hidden_node_features + self.n_edge_features,
            hidden_layer_sizes = [self.att_hidden_dim] * self.att_depth,
            out_features = self.message_size,
            dropout_p    = self.att_dropout_p
        )

        self.gru        = torch.nn.GRUCell(
            input_size = self.message_size,
            hidden_size = self.hidden_node_features,
            bias = True
        )

        self.s2v        = Set2Vec(
            node_features = self.n_node_features,
            hidden_node_features = self.hidden_node_features,
            lstm_computations = self.s2v_lstm_computations,
            memory_size = self.s2v_memory_size,
        )

    def aggregate_message(self, nodes : torch.Tensor,
                          node_neighbours : torch.Tensor,
                          edges : torch.Tensor,
                          mask : torch.Tensor) -> torch.Tensor:
        Softmax         = torch.nn.Softmax(dim=1)
        max_node_degree = node_neighbours.shape[1]

        enn_output      = self.enn(edges)
        matrices        = enn_output.view(-1,
                                          max_node_degree,
                                          self.message_size,
                                          self.hidden_node_features
                                          )
        message_terms   = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze()

        att_enn_output  = self.att_enn(torch.cat((edges, node_neighbours), dim=2))
        energies        = att_enn_output.view(-1, max_node_degree, self.message_size)
        energy_mask     = (1 - mask).float() * self.big_negative
        weights         = Softmax(energies + energy_mask.unsqueeze(-1))

        return (weights * message_terms).sum(1)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        messages = messages + torch.zeros(self.message_size).to(messages)
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor,
                input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings


class GGNN(SummationMPNN):
    """
    The "gated-graph neural network" model.
    """
    def __init__(self,  hidden_node_features=100,
                        n_node_features=14,
                        n_edge_features=4,
                        message_size=100,
                        message_passes=3,
                        hidden_dim=250,
                        module_depth=4,
                        dropout=0.0,
                        ) -> None:
        super().__init__(hidden_node_features,n_edge_features,message_size,message_passes)

        self.hidden_node_features=hidden_node_features
        self.n_node_features=n_node_features
        self.n_edge_features=n_edge_features
        self.message_size=message_size
        self.message_passes=message_passes
        self.enn_hidden_dim=hidden_dim
        self.enn_depth=module_depth
        self.enn_dropout_p=dropout
        self.gather_width=hidden_node_features
        self.gather_att_depth=module_depth
        self.gather_att_hidden_dim = hidden_dim
        self.gather_att_dropout_p = dropout
        self.gather_emb_depth = module_depth
        self.gather_emb_hidden_dim = hidden_dim 
        self.gather_emb_dropout_p = dropout
        self.big_positive=1e6

        self.msg_nns    = torch.nn.ModuleList()
        for _ in range(self.n_edge_features):
            self.msg_nns.append(
                MLP(
                    in_features=self.hidden_node_features,
                    hidden_layer_sizes=[self.enn_hidden_dim] * self.enn_depth,
                    out_features= self.message_size,
                    dropout_p=self.enn_dropout_p,
                )
            )

        self.gru        = torch.nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True
        )

        self.gather     = GraphGather(
            node_features=self.n_node_features,
            hidden_node_features=self.hidden_node_features,
            out_features=self.gather_width,
            att_depth=self.gather_att_depth,
            att_hidden_dim=self.gather_att_hidden_dim,
            att_dropout_p=self.gather_att_dropout_p,
            emb_depth=self.gather_emb_depth,
            emb_hidden_dim=self.gather_emb_hidden_dim,
            emb_dropout_p=self.gather_emb_dropout_p,
            big_positive=self.big_positive
        )

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        edges_v               = edges.view(-1, self.n_edge_features, 1)
        node_neighbours_v     = edges_v * node_neighbours.view(-1,
                                                               1,
                                                               self.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.n_edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings


class AttentionGGNN(AggregationMPNN):
    """
    The "GGNN with attention" model.
    """
    def __init__(self, hidden_node_features,
                        n_edge_features,
                        message_size,
                        message_passes,
                        n_node_features=14,
                        hidden_dim=250,
                        module_depth=4,
                        dropout=0.0):
        super().__init__(hidden_node_features,n_edge_features,message_size,message_passes)
        
        self.hidden_node_features=hidden_node_features
        self.n_node_features=n_node_features
        self.n_edge_features=n_edge_features
        self.message_size=message_size
        self.message_passes=message_passes

        self.enn_hidden_dim=hidden_dim
        self.enn_depth=module_depth
        self.enn_dropout_p=dropout

        self.msg_depth=module_depth
        self.msg_dropout_p=dropout
        self.msg_hidden_dim=hidden_dim

        self.gather_width = hidden_node_features
        self.gather_att_depth = module_depth
        self.gather_att_hidden_dim = hidden_dim
        self.gather_att_dropout_p = dropout
        self.gather_emb_depth = module_depth
        self.gather_emb_hidden_dim = hidden_dim 
        self.gather_emb_dropout_p = dropout 
        self.big_positive = 1e6 
        self.att_hidden_dim = hidden_dim 
        self.att_depth = module_depth 
        self.att_dropout_p = dropout 
        

        #FGP.modelsetting = constants
        self.msg_nns   = torch.nn.ModuleList()
        self.att_nns   = torch.nn.ModuleList()

        for _ in range(self.n_edge_features):
            self.msg_nns.append(
                MLP(
                  in_features=self.hidden_node_features,
                  hidden_layer_sizes=[self.msg_hidden_dim] * self.msg_depth,
                  out_features=self.message_size,
                  dropout_p=self.msg_dropout_p,
                )
            )
            self.att_nns.append(
                MLP(
                  in_features=self.hidden_node_features,
                  hidden_layer_sizes=[self.att_hidden_dim] * self.att_depth,
                  out_features=self.message_size,
                  dropout_p=self.att_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True
        )

        self.gather = GraphGather(
            node_features=self.n_node_features,
            hidden_node_features=self.hidden_node_features,
            out_features=self.gather_width,
            att_depth=self.gather_att_depth,
            att_hidden_dim=self.gather_att_hidden_dim,
            att_dropout_p=self.gather_att_dropout_p,
            emb_depth=self.gather_emb_depth,
            emb_hidden_dim=self.gather_emb_hidden_dim,
            emb_dropout_p=self.gather_emb_dropout_p,
            big_positive=self.big_positive
        )

    def aggregate_message(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                          edges : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        Softmax = torch.nn.Softmax(dim=1)

        energy_mask = (mask == 0).float() * self.big_positive

        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns[i](node_neighbours)
            for i in range(self.n_edge_features)
        ]
        energies_masked_per_edge = [ edges[:, :, i].unsqueeze(-1) * self.att_nns[i](node_neighbours)
            for i in range(self.n_edge_features) ]

        embedding   = sum(embeddings_masked_per_edge)
        energies    = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention   = Softmax(energies)

        return torch.sum(attention * embedding, dim=1)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes.float())

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:

        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)

        #output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return graph_embeddings

class EMN(EdgeMPNN):
    """
    The "edge memory network" model.
    """
    def __init__(self,  hidden_node_features=100,
                        n_node_features=14,
                        n_edge_features=4,
                        message_size=100,
                        message_passes=3,
                        enn_hidden_dim=250,
                        enn_depth=4,
                        enn_dropout_p=0.0,
                        edge_emb_depth = 4, 
                        edge_emb_dropout_p = 0.0,
                        edge_emb_hidden_dim = 250,
                        edge_emb_size = 100,
                        msg_depth = 4,
                        msg_dropout_p = 0.0,
                        msg_hidden_dim = 250,
                        gather_width=100,
                        gather_att_depth=4,
                        gather_att_hidden_dim = 250,
                        gather_att_dropout_p = 0.0,
                        gather_emb_depth = 4,
                        gather_emb_hidden_dim = 250,
                        gather_emb_dropout_p = 0.0,
                        big_positive=1e6) -> None:

        super().__init__()

        self.hidden_node_features=hidden_node_features
        self.n_node_features=n_node_features
        self.n_edge_features=n_edge_features
        self.message_size=message_size
        self.message_passes=message_passes

        self.enn_hidden_dim=enn_hidden_dim
        self.enn_depth=enn_depth
        self.enn_dropout_p=enn_dropout_p

        self.msg_depth=msg_depth
        self.msg_dropout_p=msg_dropout_p
        self.msg_hidden_dim=msg_hidden_dim

        self.gather_width=gather_width
        self.gather_att_depth=gather_att_depth
        self.gather_att_hidden_dim = gather_att_hidden_dim
        self.gather_att_dropout_p = gather_att_dropout_p
        self.gather_emb_depth = gather_emb_depth
        self.gather_emb_hidden_dim = gather_emb_hidden_dim 
        self.gather_emb_dropout_p = gather_emb_dropout_p
        self.big_positive=big_positive

        self.embedding_nn = MLP(
            in_features=self.n_node_features * 2 + self.n_edge_features,
            hidden_layer_sizes=[self.edge_emb_hidden_dim] *self.edge_emb_depth,
            out_features=self.edge_emb_size,
            dropout_p=self.edge_emb_dropout_p,)

        self.emb_msg_nn   = MLP(
            in_features=self.edge_emb_size,
            hidden_layer_sizes=[self.msg_hidden_dim] * self.msg_depth,
            out_features=self.edge_emb_size,
            dropout_p=self.msg_dropout_p,
        )

        self.att_msg_nn   = MLP(
            in_features=self.edge_emb_size,
            hidden_layer_sizes=[self.att_hidden_dim] * self.att_depth,
            out_features=self.edge_emb_size,
            dropout_p=self.att_dropout_p,
        )

        self.gru          = torch.nn.GRUCell(
            input_size=self.edge_emb_size,
            hidden_size=self.edge_emb_size,
            bias=True
        )

        self.gather       = GraphGather(
            node_features=self.edge_emb_size,
            hidden_node_features=self.edge_emb_size,
            out_features=self.gather_width,
            att_depth=self.gather_att_depth,
            att_hidden_dim=self.gather_att_hidden_dim,
            att_dropout_p=self.gather_att_dropout_p,
            emb_depth=self.gather_emb_depth,
            emb_hidden_dim=self.gather_emb_hidden_dim,
            emb_dropout_p=self.gather_emb_dropout_p,
            big_positive=self.big_positive
        )


    def preprocess_edges(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                         edges : torch.Tensor) -> torch.Tensor:
        cat = torch.cat((nodes, node_neighbours, edges), dim=1)
        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges : torch.Tensor, ingoing_edge_memories : torch.Tensor,
                        ingoing_edges_mask : torch.Tensor) -> torch.Tensor:
        Softmax             = torch.nn.Softmax(dim=1)
        energy_mask         = ((1 - ingoing_edges_mask).float() * self.big_negative).unsqueeze(-1)
        cat                 = torch.cat((edges.unsqueeze(1), ingoing_edge_memories), dim=1)
        embeddings          = self.emb_msg_nn(cat)
        edge_energy         = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies            = torch.cat((edge_energy.unsqueeze(1), ing_memory_energies), dim=1)
        attention           = Softmax(energies)

        # set aggregation of set of given edge feature and ingoing edge memories
        message = (attention * embeddings).sum(dim=1)
        return self.gru(message)  # return hidden state

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings
