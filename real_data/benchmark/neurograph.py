import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import numpy as np
softmax = torch.nn.LogSoftmax(dim=1)

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ResidualGNNs(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, hidden, num_layers, num_classes=1, gnn_type='GCNConv'):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.num_layers = num_layers

        GNN = self._get_gnn_class(gnn_type)

        if gnn_type == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        # Upper triangle WITH diagonal: n*(n+1)/2 elements
        input_dim = int((num_features * (num_features + 1)) / 2)
        input_dim1 = input_dim + (hidden_channels * num_layers)
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), num_classes),
        )

    def _get_gnn_class(self, gnn_type):
        mapping = {
            'GCNConv': GCNConv,
            'SAGEConv': SAGEConv,
            'GINConv': lambda in_c, out_c: GINConv(nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))),
            'GATConv': GATConv,
            'GraphConv': GraphConv,
            'ChebConv': ChebConv,
            'TransformerConv': TransformerConv,
        }
        return mapping.get(gnn_type, GCNConv)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                # Extract upper triangular elements (including diagonal) using indices
                n = xx.shape[-1]
                idx = torch.triu_indices(n, n, device=xx.device)
                x = torch.stack([t[idx[0], idx[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx,batch)
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x  # Raw logits (for BCEWithLogitsLoss / MSELoss)