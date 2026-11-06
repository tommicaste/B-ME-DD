import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn.functional as F


# ------- Models -------

# Velickovic et al. use K=8 attention heads on Cora/Citeseer; at hidden_dim=64
# that reproduces their first layer exactly (8 heads x 8 features).
GAT_HEADS = 8


class LR(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x, adj=None, mask=None):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None, mask=None):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class SparseGNNRegressor(nn.Module):
    """Sparse canonical PyG GCN/SAGE/GIN/GAT (edge_index + batch). forward(data).

    conv1 / optional conv2 / global-mean-pool / linear-out, using torch_geometric
    sparse convs on a PyG Batch, so it shares the PyGGraphDataset / LightningModelPyG
    path with ResidualGNN and BrainGNN.
    """
    def __init__(self, kind: str, in_dim: int, hidden_dim: int, num_convs: int = 2,
                 activation: nn.Module = None, readout: str = 'mean'):
        super().__init__()
        self.conv1 = self._make_conv(kind, in_dim, hidden_dim)
        self.conv2 = self._make_conv(kind, hidden_dim, hidden_dim) if num_convs == 2 else None
        if activation is None:
            activation = nn.ELU() if kind.lower() == 'gat' else nn.ReLU()
        self.activation = activation
        if readout not in ('mean', 'vec'):
            raise ValueError(f"readout must be 'mean' or 'vec', got {readout!r}")
        self.readout = readout
        # Node features are the rows of a (D,D) correlation matrix, so n_nodes == in_dim.
        self.n_nodes = in_dim
        self.out = nn.Linear(hidden_dim if readout == 'mean' else in_dim * hidden_dim, 1)

    @staticmethod
    def _make_conv(kind, in_dim, out_dim):
        kind = kind.lower()
        if kind == 'gcn':
            return GCNConv(in_dim, out_dim)
        if kind == 'sage':
            return SAGEConv(in_dim, out_dim)
        if kind == 'gin':
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim)))
        if kind == 'gat':
            # Velickovic et al. multi-head attention: GAT_HEADS heads concatenated.
            # Per-head width is out_dim // GAT_HEADS so the total matches GCN/SAGE --
            # otherwise GAT would be GAT_HEADS x wider and any win would be capacity.
            if out_dim % GAT_HEADS:
                raise ValueError(f"hidden_dim {out_dim} must be divisible by "
                                 f"GAT_HEADS={GAT_HEADS}")
            return GATConv(in_dim, out_dim // GAT_HEADS, heads=GAT_HEADS, concat=True)
        raise ValueError(f"Unknown conv type: {kind}")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.activation(self.conv1(x, edge_index))
        if self.conv2 is not None:
            x = self.activation(self.conv2(x, edge_index))
        if self.readout == 'mean':
            g = global_mean_pool(x, batch)
        else:
            # mean_pool(A_norm X W) = (1/N) 1^T A_norm X W, so the adjacency enters only
            # through the column sums of A_norm -- measured cosine 0.98-0.99 across
            # variants, i.e. mean-pooling very nearly cancels the graph. Keeping every
            # node keeps that information (cosine drops to 0.49 at T=5).
            n_graphs = int(batch.max()) + 1
            if x.size(0) != n_graphs * self.n_nodes:
                raise ValueError(f"vec readout needs a constant {self.n_nodes} nodes/graph, "
                                 f"got {x.size(0)} nodes for {n_graphs} graphs")
            g = x.view(n_graphs, self.n_nodes * x.size(-1))
        return self.out(g)


# ------- Lightning Module -------

class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        task='regression',
        lr=1e-3,
        weight_decay=0.0,
        l1_lambda=0.0,
        use_adjacency=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.use_adjacency = use_adjacency

        if task == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, x, a=None):
        if self.use_adjacency:
            return self.model(x, a)
        return self.model(x)

    def _unpack(self, batch):
        if self.use_adjacency:
            x, a, y = batch
            return x, a, y
        x, y = batch
        return x, None, y

    def _l1_penalty(self):
        if self.l1_lambda == 0.0:
            return 0.0
        l1 = 0.0
        for p in self.model.parameters():
            l1 = l1 + p.abs().sum()
        return self.l1_lambda * l1

    def _compute_metrics(self, y_hat, y, prefix):
        loss = self.loss_fn(y_hat, y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.task == 'classification':
            with torch.no_grad():
                preds = (y_hat > 0).float()
                acc = (preds == y).float().mean()
            self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, a, y = self._unpack(batch)
        y = y.float()
        y_hat = self(x, a).squeeze(-1)

        base_loss = self.loss_fn(y_hat, y)
        l1 = self._l1_penalty()
        loss = base_loss + l1

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(l1, torch.Tensor):
            self.log("train_l1", l1, on_step=False, on_epoch=True)

        if self.task == 'classification':
            with torch.no_grad():
                preds = (y_hat > 0).float()
                acc = (preds == y).float().mean()
            self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, a, y = self._unpack(batch)
        y = y.float()
        y_hat = self(x, a).squeeze(-1)
        return self._compute_metrics(y_hat, y, "val")

    def test_step(self, batch, batch_idx):
        x, a, y = self._unpack(batch)
        y = y.float()
        y_hat = self(x, a).squeeze(-1)
        return self._compute_metrics(y_hat, y, "test")

    def predict_step(self, batch, batch_idx):
        x, a, _ = self._unpack(batch)
        return self(x, a).squeeze(-1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ------- Dataset -------

class GraphDataset(Dataset):
    def __init__(self, x, y, adjacency=None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.a = torch.as_tensor(adjacency, dtype=torch.float32) if adjacency is not None else None

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        if self.a is not None:
            return self.x[i], self.a[i], self.y[i]
        return self.x[i], self.y[i]


# ------- PyG Adapter Components -------

def dense_to_edge_index(adj):
    """Convert dense adjacency (n, n) to edge_index (2, num_edges)."""
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index


class PyGGraphDataset(Dataset):
    """Dataset that converts dense matrices to PyG Data objects."""

    def __init__(self, x, y, adjacency, include_edge_attr=False):
        # x: correlation matrices (N, n_rois, n_rois) - used as node features
        # adjacency: binary adjacency (N, n_rois, n_rois) - used for edge_index
        # include_edge_attr: if True, also store edge_attr (correlation values) and pos (identity)
        self.data_list = []
        n_rois = x.shape[1]
        pos = torch.eye(n_rois, dtype=torch.float32) if include_edge_attr else None
        for i in range(len(x)):
            node_features = torch.as_tensor(x[i], dtype=torch.float32)
            adj = torch.as_tensor(adjacency[i], dtype=torch.float32)
            edge_index = dense_to_edge_index(adj)
            label = torch.as_tensor(y[i], dtype=torch.float32)
            data = Data(x=node_features, edge_index=edge_index, y=label)
            if include_edge_attr:
                corr = torch.as_tensor(x[i], dtype=torch.float32)
                data.edge_attr = corr[edge_index[0], edge_index[1]].unsqueeze(-1)
                data.pos = pos
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class LightningModelPyG(pl.LightningModule):
    """Lightning wrapper for PyG models that handle Batch objects directly."""

    def __init__(
        self,
        model,
        task='regression',
        lr=1e-3,
        weight_decay=0.0,
        l1_lambda=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

        if task == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.model(batch)

    def _l1_penalty(self):
        if self.l1_lambda == 0.0:
            return 0.0
        l1 = 0.0
        for p in self.model.parameters():
            l1 = l1 + p.abs().sum()
        return self.l1_lambda * l1

    def _compute_metrics(self, y_hat, y, prefix):
        loss = self.loss_fn(y_hat, y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.task == 'classification':
            with torch.no_grad():
                preds = (y_hat > 0).float()
                acc = (preds == y).float().mean()
            self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        y = batch.y.float()
        y_hat = self(batch).squeeze(-1)

        base_loss = self.loss_fn(y_hat, y)
        l1 = self._l1_penalty()
        loss = base_loss + l1

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(l1, torch.Tensor):
            self.log("train_l1", l1, on_step=False, on_epoch=True)

        if self.task == 'classification':
            with torch.no_grad():
                preds = (y_hat > 0).float()
                acc = (preds == y).float().mean()
            self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y.float()
        y_hat = self(batch).squeeze(-1)
        return self._compute_metrics(y_hat, y, "val")

    def test_step(self, batch, batch_idx):
        y = batch.y.float()
        y_hat = self(batch).squeeze(-1)
        return self._compute_metrics(y_hat, y, "test")

    def predict_step(self, batch, batch_idx):
        """Raw per-sample scores, in loader order. save_results turns them into
        per-window AND per-subject metrics; the logged *_acc is a mean of
        per-batch means and cannot be re-aggregated by subject."""
        return self(batch).squeeze(-1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ------- BrainGNN Integration -------

from real_data.benchmark.BrainGNN_Pytorch.net.braingnn import Network as _BrainGNNNetwork
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

EPS = 1e-10


class BrainGNNWrapper(nn.Module):
    """Wraps BrainGNN Network to follow pipeline conventions (like ResidualGNNs).

    - forward(data) takes a PyG Batch
    - Returns (logits, s1, s2) — raw logits (no log_softmax) + pooling scores
    - Single output for both classification (BCEWithLogitsLoss) and regression (MSE)
    """

    def __init__(self, indim, ratio, k=8, R=200):
        super().__init__()
        # Build original network, then replace final layer
        self.net = _BrainGNNNetwork(indim=indim, ratio=ratio, nclass=2, k=k, R=R)
        self.net.fc3 = nn.Linear(self.net.dim3, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr, pos = data.edge_attr, data.pos
        net = self.net

        x = net.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = net.pool1(x, edge_index, edge_attr, batch)
        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = net.augment_adj(edge_index, edge_attr, x.size(0))

        x = net.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = net.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1, x2], dim=1)
        x = net.bn1(F.relu(net.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = net.bn2(F.relu(net.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        logits = net.fc3(x)

        num_graphs = batch.max().item() + 1
        return (
            logits,
            torch.sigmoid(score1).view(num_graphs, -1),
            torch.sigmoid(score2).view(num_graphs, -1),
        )


def _topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = (-torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean()
           - torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean())
    return res


def _consist_loss(s, device):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0], device=device)
    D = torch.eye(s.shape[0], device=device) * torch.sum(W, dim=1)
    L = D - W
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res


class LightningBrainGNN(pl.LightningModule):
    """Lightning wrapper for BrainGNN with custom regularization losses."""

    def __init__(self, model, task='regression', lr=1e-3, weight_decay=0.0,
                 ratio=0.5, lamb3=0.1, lamb5=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.ratio = ratio
        self.lamb3 = lamb3
        self.lamb5 = lamb5

        if task == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.model(batch)

    def _braingnn_loss(self, batch, prefix):
        y = batch.y.float()
        logits, s1, s2 = self(batch)
        y_hat = logits.squeeze(-1)

        base_loss = self.loss_fn(y_hat, y)
        loss_tpk1 = _topk_loss(s1, self.ratio)
        loss_tpk2 = _topk_loss(s2, self.ratio)

        loss_consist = 0
        if self.task == 'classification':
            for c in [0, 1]:
                mask = (y == c)
                if mask.any():
                    loss_consist += _consist_loss(s1[mask], y.device)

        loss = base_loss + self.lamb3 * (loss_tpk1 + loss_tpk2) + self.lamb5 * loss_consist

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.task == 'classification':
            with torch.no_grad():
                preds = (y_hat > 0).float()
                acc = (preds == y).float().mean()
            self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._braingnn_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._braingnn_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._braingnn_loss(batch, "test")

    def predict_step(self, batch, batch_idx):
        logits, _, _ = self(batch)
        return logits.squeeze(-1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
