import torch.nn as nn
from utils import *


class FullParam(nn.Module):
    def __init__(self, features, non_linearity, k, knn_metric, i, sparse):
        super(FullParam, self).__init__()

        self.non_linearity = non_linearity
        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.sparse = sparse

        if self.non_linearity == "exp":
            self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_exp(features, self.k, self.knn_metric, self.i)))
        elif self.non_linearity == "elu":
            self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))
        elif self.non_linearity == 'none':
            self.Adj = nn.Parameter(torch.from_numpy(nearest_neighbors(features, self.k, self.knn_metric)))
        else:
            raise NameError('No non-linearity has been specified')

    def forward(self, h):
        if not self.sparse:
            if self.non_linearity == "exp":
                Adj = torch.exp(self.Adj)
            elif self.non_linearity == "elu":
                Adj = F.elu(self.Adj) + 1
            elif self.non_linearity == "none":
                Adj = self.Adj
        else:
            if self.non_linearity == 'exp':
                Adj = self.Adj.coalesce()
                Adj.values = torch.exp(Adj.values())
            elif self.non_linearity == 'elu':
                Adj = self.Adj.coalesce()
                Adj.values = F.elu(Adj.values()) + 1
            elif self.non_linearity == "none":
                Adj = self.Adj
            else:
                raise NameError('Non-linearity is not supported in the sparse setup')
        return Adj