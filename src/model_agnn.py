import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch, to_dense_adj

class HybridGPSLayer(nn.Module):
    """ A single layer combining Local GCN and Global Transformer Attention. """
    def __init__(self, channels, nhead, dropout=0.1):
        super().__init__()
        
        # 1. Local Track (Message Passing)
        self.conv = GATv2Conv(channels, channels, heads=1, concat=False)
        
        # 2. Global Track (Self-Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=nhead, 
            dropout=dropout, batch_first=True
        )
        
        # 3. Fusion & FFN
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, device_type):
        # -- Local Track (Message Passing along explicit edges) --
        h_local = self.conv(x, edge_index)
        
        # -- Global Track (Dense Self-Attention everywhere) --
        h_dense, mask = to_dense_batch(x, batch)
        
        # Construct 3D float mask to bypass PyTorch SDPA key_padding_mask bugs on MPS/CPU
        batch_size, n_max = mask.shape
        float_mask = torch.zeros(batch_size, n_max, n_max, device=x.device)
        float_mask = float_mask.masked_fill(~mask.unsqueeze(1).expand(-1, n_max, -1), float('-inf'))
        float_mask_repeated = float_mask.repeat_interleave(self.attn.num_heads, dim=0)
        
        attn_out, _ = self.attn(
            query=h_dense, key=h_dense, value=h_dense, 
            attn_mask=float_mask_repeated,
            need_weights=False
        )
        # Extract valid nodes back to sparse graph format
        h_global = attn_out[mask]
        
        # -- Hybrid Fusion: Add Original, Local, and Global --
        h_fused = self.norm1(x + h_local + self.dropout(h_global))
        
        # -- Feed Forward Network --
        out = self.norm2(h_fused + self.ffn(h_fused))
        return out


def get_enhanced_node_features(data):
    """
    Computes a 5-dimensional feature representation for Seiberg quivers:
    1. Log-scaled Group Rank
    2. Log-scaled In-Degree (incoming arrows)
    3. Log-scaled Out-Degree (outgoing arrows)
    4. Log-scaled In-Flux (incoming neighbor rank interaction)
    5. Log-scaled Out-Flux (outgoing neighbor rank interaction)
    """
    N = data.x.shape[0]
    device = data.x.device
    
    # Cast variables to float32 to avoid dtype mismatches during scatter_add_
    ranks = data.x[:, 0:1].to(torch.float32) # [N, 1]
    
    in_deg = torch.zeros(N, 1, dtype=torch.float32, device=device)
    out_deg = torch.zeros(N, 1, dtype=torch.float32, device=device)
    in_flux = torch.zeros(N, 1, dtype=torch.float32, device=device)
    out_flux = torch.zeros(N, 1, dtype=torch.float32, device=device)
    
    if data.edge_index.numel() > 0:
        u = data.edge_index[0]
        v = data.edge_index[1]
        weight = data.edge_attr.to(torch.float32) if data.edge_attr is not None else torch.ones(data.edge_index.shape[1], 1, dtype=torch.float32, device=device)
        
        # Degrees
        out_deg.scatter_add_(0, u.unsqueeze(-1), weight)
        in_deg.scatter_add_(0, v.unsqueeze(-1), weight)
        
        # Fluxes
        ranks_flat = ranks.squeeze(-1)
        out_flux.scatter_add_(0, u.unsqueeze(-1), weight * ranks_flat[v].unsqueeze(-1))
        in_flux.scatter_add_(0, v.unsqueeze(-1), weight * ranks_flat[u].unsqueeze(-1))
        
    # Scale all features with log1p
    return torch.cat([
        torch.log1p(ranks),
        torch.log1p(in_deg),
        torch.log1p(out_deg),
        torch.log1p(in_flux),
        torch.log1p(out_flux)
    ], dim=-1)


class AGNNGPS(nn.Module):
    """
    AGNN Sequence-to-Sequence model for Seiberg Duality.
    Maps (Graph A, Graph B) -> Probability distribution over nodes of Graph A.
    """
    def __init__(
        self,
        in_channels=5,
        pe_channels=8,        
        hidden_channels=128,
        num_encoder_layers=2,
        num_decoder_layers=2, # Kept in init args for compatibility, but unused
        nhead=8,
        dropout=0.20,
        use_delta_a=True
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.nhead = nhead
        self.use_delta_a = use_delta_a
        
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.pe_proj = nn.Linear(pe_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        
        # DGNN Encoders
        self.encoders = nn.ModuleList([
            HybridGPSLayer(hidden_channels, nhead, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Node-level classifier head
        # Features: dense_a, dense_b, diff_deep (b - a), diff_raw (b0 - a0), original_x_a, [optional delta_A]
        classifier_in_features = hidden_channels * 4 + in_channels + (1 if use_delta_a else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def load_state_dict(self, state_dict, strict=False):
        # Ignore missing cross-attention keys from older checkpoints
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, data_a, data_b, action_mask=None):
        device_type = data_a.x.device.type
        
        x_a = get_enhanced_node_features(data_a)
        x_b = get_enhanced_node_features(data_b)
            
        # --- INITIAL EMBEDDINGS ---
        h_a_0 = self.input_norm(self.node_proj(x_a) + self.pe_proj(data_a.pe))
        h_b_0 = self.input_norm(self.node_proj(x_b) + self.pe_proj(data_b.pe))
        
        # --- ENCODING GRAPH A ---
        h_a = h_a_0
        for layer in self.encoders:
            h_a = layer(h_a, data_a.edge_index, data_a.batch, device_type)
            
        # --- ENCODING GRAPH B ---
        h_b = h_b_0
        for layer in self.encoders:
            h_b = layer(h_b, data_b.edge_index, data_b.batch, device_type)
            
        # --- DENSE PADDING ---
        dense_a, mask_a = to_dense_batch(h_a, data_a.batch)
        dense_b, _ = to_dense_batch(h_b, data_b.batch)
        
        dense_a_0, _ = to_dense_batch(h_a_0, data_a.batch)
        dense_b_0, _ = to_dense_batch(h_b_0, data_b.batch)
        dense_feats_a, _ = to_dense_batch(x_a, data_a.batch)
        
        # --- EXACT ALIGNMENT SUBTRACTION (Crucial: No torch.abs()) ---
        diff_deep = dense_b - dense_a
        diff_raw = dense_b_0 - dense_a_0
        
        # --- ADJACENCY DIFFERENCING ---
        if self.use_delta_a:
            n_max = mask_a.shape[1]
            edge_attr_a = data_a.edge_attr.view(-1) if data_a.edge_attr is not None else None
            edge_attr_b = data_b.edge_attr.view(-1) if data_b.edge_attr is not None else None
            
            A_a = to_dense_adj(data_a.edge_index, batch=data_a.batch, edge_attr=edge_attr_a, max_num_nodes=n_max)
            A_b = to_dense_adj(data_b.edge_index, batch=data_b.batch, edge_attr=edge_attr_b, max_num_nodes=n_max)
            delta_A = torch.abs(A_b - A_a).sum(dim=-1).unsqueeze(-1)
            
            features = torch.cat([dense_a, dense_b, diff_deep, diff_raw, dense_feats_a, delta_A], dim=-1)
        else:
            features = torch.cat([dense_a, dense_b, diff_deep, diff_raw, dense_feats_a], dim=-1)
            
        logits = self.classifier(features).squeeze(-1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0) 
        
        logits = logits.masked_fill(~mask_a, -1e9)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
            
        return logits
