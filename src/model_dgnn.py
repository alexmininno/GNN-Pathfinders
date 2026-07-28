import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

# ---------------------------------------------------------------------------
# GNN layer (positive-weight only, standard sum aggregation)
# ---------------------------------------------------------------------------

class GCNConvSimple(nn.Module):
    """
    A lightweight GCN layer for non-negative edge weights (raw A entries).
    Separate linear transforms for message and self-loop, then sum-aggregate.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_msg = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        w = edge_weight.view(-1)
        row, col = edge_index

        out = self.lin_self(x)
        if row.numel() > 0:
            # Applica la proiezione lineare sui Nodi (O(N)) prima di propagare (anziché sugli archi O(E))
            x_msg = self.lin_msg(x)
            msg = x_msg[row] * w.to(x_msg.dtype).view(-1, 1)  # scale by arrow count
            out.index_add_(0, col, msg)
        return out

class GNNTokenizer(nn.Module):
    """
    GNN Tokenizer using A-based (positive) edge weights.
    Includes LayerNorm to prevent sum-aggregation explosion.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConvSimple(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConvSimple(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.leaky_relu(h, 0.2)
            if i > 0:
                x = x + h
            else:
                x = h
        return x

# ---------------------------------------------------------------------------
# Distance Regressor Head
# ---------------------------------------------------------------------------

class DistanceRegressorHead(nn.Module):
    def __init__(self, hidden_channels):
        """
        Calculates the topological mutation distance between Graph A and B
        using their pooled representations and a pre-computed node-level
        difference (pooled AFTER comparison to preserve local signals).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Softplus(),  # Smooth >0 activation to prevent dead gradients
        )

    def forward(self, z_a, z_b, node_diff_pooled):
        # node_diff_pooled = mean(abs(h_a - h_b)), computed at node-level
        # before pooling so local mutation signals are preserved.
        features = torch.cat([z_a, z_b, node_diff_pooled], dim=-1)
        return self.mlp(features)

# ---------------------------------------------------------------------------
# Main DGNN Model
# ---------------------------------------------------------------------------

class DGNNSeiberg(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        num_gnn_layers=3,
        num_transformer_layers=2,
        nhead=4,
    ):
        """
        DGNN Architecture: Continuous Topological Distance Regressor.
        Fully decoupled from previous DGNN models. Built from scratch.
        """
        super(DGNNSeiberg, self).__init__()

        # 1. Base Topological Encoder parts
        self.tokenizer = GNNTokenizer(
            in_channels, hidden_channels, num_gnn_layers
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        # 2. Continuous Distance Regressor
        self.distance_regressor = DistanceRegressorHead(hidden_channels)

    def encode(self, data):
        """
        Runs GNN + Transformer and returns node embeddings.
        """
        # Strictly use ONLY the rank [N, 1]. Strip any dual_flag if it exists!
        # Apply symlog-like normalization (log1p) internally to handle high-rank scales
        x = torch.log1p(data.x[:, 0:1])
        edge_index, edge_attr, batch_idx = data.edge_index, data.edge_attr, data.batch
        
        # Edge attributes are arrow counts, which can also be extremely large.
        edge_attr = torch.log1p(edge_attr)

        # A. GNN Tokenization
        h_nodes = self.tokenizer(x, edge_index, edge_attr)

        # B. Transformer (handles global context with padding mask)
        # Sostituito il loop su CPU con l'operatore nativo PyG (Niente CUDA Sync)
        h_padded, mask = to_dense_batch(h_nodes, batch_idx)
        padding_mask = ~mask

        # MASK WORKAROUND FOR MPS BACKEND:
        # If the device is MPS, passing a boolean src_key_padding_mask causes NaNs during backprop.
        # However, passing a float additive mask is stable and prevents attention pollution.
        if x.device.type == "mps":
            float_mask = torch.where(padding_mask, float('-inf'), 0.0)
            h_transformer = self.transformer_encoder(h_padded, src_key_padding_mask=float_mask)
        else:
            h_transformer = self.transformer_encoder(h_padded, src_key_padding_mask=padding_mask)

        # Ricostruisce l'esatto ordine sequenziale scartando i token di padding in un sol colpo
        h_final = h_transformer[mask]
        h_final = torch.nan_to_num(h_final, nan=0.0)
        return h_final

    def encode_graphs(self, data_a, data_b):
        h_a = self.encode(data_a)
        h_b = self.encode(data_b)
        z_a = global_mean_pool(h_a, data_a.batch)
        z_b = global_mean_pool(h_b, data_b.batch)
        return h_a, h_b, z_a, z_b

    def forward(self, data_a, data_b):
        """
        Args:
            data_a, data_b : PyG batched graph objects

        Returns:
            dist_pred   : [B, 1] continuous distance prediction
        """
        h_a, h_b, z_a, z_b = self.encode_graphs(data_a, data_b)

        # Compare nodes FIRST, then pool.
        # This preserves the local signal: at d=1 one node lights up,
        # at d=2 two nodes light up, etc.
        node_diff = torch.abs(h_a - h_b)  # [N_total, H]
        node_diff_pooled = global_mean_pool(node_diff, data_a.batch)  # [B, H]

        dist_pred = self.distance_regressor(z_a, z_b, node_diff_pooled)

        return dist_pred
