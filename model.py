import math
from graph_generator import *
from layers import GCNConv_dense, GCNConv_dgl
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN_DAE(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, gen_mode, sparse):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dgl(hidden_dim, nclasses))

        else:
            self.layers.append(GCNConv_dense(in_dim, hidden_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dense(hidden_dim, nclasses))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.k = k
        self.knn_metric = knn_metric
        self.i = i_
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.nnodes = features.shape[0]
        self.sparse = sparse

        if gen_mode == 0:
            self.graph_gen = FullParam(features, non_linearity, k, knn_metric, self.i, sparse).cuda()

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        if not self.sparse:
            Adj_ = symmetrize(Adj_)
            Adj_ = normalize(Adj_, self.normalization, self.sparse)
        return Adj_

    def forward(self, features, x):  # x corresponds to masked_fearures
        Adj_ = self.get_adj(features)
        if self.sparse:
            Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

    def forward(self, x):

        if self.sparse:
            Adj = self.Adj
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GCN_C_PyG(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_C_PyG, self).__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, 16) #should be hidden_channels
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, adj_t, edge_weight):
        x = self.conv1(x, adj_t, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t, edge_weight)

        return F.log_softmax(x, dim=1)


class GCN_PyG(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_PyG, self).__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, 16) #should be hidden_channels
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)


# using the ADJ directly!
class NAIVEGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, dropout_adj, sparse):
        super(NAIVEGCN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gcn_1 = nn.Linear(self.input_size, self.hidden_size)
        self.gcn_2 = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x, A):

        x = self.gcn_1(x)
        x = torch.sparse.mm(A, x)
        x = self.relu(x)

        # layer 2
        x = self.gcn_2(x)
        x = torch.sparse.mm(A, x)
        x = self.softmax(x)

        return x


class GCN_C(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_C, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

    def forward(self, x, adj_t):
        if self.sparse:
            Adj = adj_t
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(adj_t)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x
