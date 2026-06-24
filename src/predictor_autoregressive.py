"""
predictor_autoregressive.py

Predictor for Autoregressive GPS model.
Given a pair of graphs (A, B), it computes the forward pass of the AR model
and returns the predicted probabilities/logits for the next mutation step.
"""

import argparse
import ast
import os
import sys
import torch
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_autoregressive import AutoregressiveGPS
from src.siamese_dataset import safe_laplacian_pe


class AutoregressivePredictor:
    def __init__(self, model_path, hidden_channels=128, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self.device = torch.device("mps")
        else:
            self.device = device

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # Auto-detect legacy model (without delta_A in classifier)
        classifier_weight = state_dict.get('classifier.0.weight')
        use_delta_a = True
        if classifier_weight is not None:
            in_features = classifier_weight.shape[1]
            if in_features == hidden_channels * 4 + 5: # 517 for hidden_channels=128
                use_delta_a = False
                print(f"AutoregressivePredictor: Detected legacy checkpoint (no delta_A). Running in backward compatibility mode.")

        self.model = AutoregressiveGPS(hidden_channels=hidden_channels, use_delta_a=use_delta_a).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_pyg_data(self, ranks, adj):
        """Converts raw ranks/adj to PyG Data with Laplacian PE."""
        N = len(ranks)
        MAX_F32 = 3.4e38
        
        safe_ranks = [MAX_F32 if r > MAX_F32 else (-MAX_F32 if r < -MAX_F32 else float(r)) for r in ranks]
        x = torch.tensor(safe_ranks, dtype=torch.float32).view(N, 1)
        # Second column is a dummy (dual flag placeholder — not used by AR model)
        x = torch.cat([x, torch.zeros((N, 1), dtype=torch.float32)], dim=1)

        safe_adj = [[MAX_F32 if a > MAX_F32 else (-MAX_F32 if a < -MAX_F32 else float(a)) for a in row] for row in adj]
        A = torch.tensor(safe_adj, dtype=torch.float32)
        
        edge_index = (A > 0).nonzero().t()
        if edge_index.numel() == 0:
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_attr = A[edge_index[0], edge_index[1]].view(-1, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = N
        return safe_laplacian_pe(data, k=8)

    @torch.no_grad()
    def predict_logits_batch(self, batch_a, batch_b):
        """
        Takes batched PyG Data objects and computes the logits.
        """
        logits = self.model(batch_a, batch_b)
        return logits
        
    @torch.no_grad()
    def predict(self, ranks_a, adj_a, ranks_b, adj_b):
        """
        Computes the forward pass and returns probabilities for mutation at each node.
        """
        data_a = self.get_pyg_data(ranks_a, adj_a)
        data_b = self.get_pyg_data(ranks_b, adj_b)

        batch_a = Batch.from_data_list([data_a]).to(self.device)
        batch_b = Batch.from_data_list([data_b]).to(self.device)

        logits = self.model(batch_a, batch_b).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits.cpu().tolist(),
            "probabilities": probs.cpu().tolist()
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoregressive Predictor CLI"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to AR checkpoint")
    parser.add_argument("--ranks_a", type=str, required=True, help="e.g. '[1, 2, 3]'")
    parser.add_argument("--adj_a", type=str, required=True, help="e.g. '[[0,1,0],[0,0,1],[1,0,0]]'")
    parser.add_argument("--ranks_b", type=str, required=True, help="e.g. '[1, 1, 3]'")
    parser.add_argument("--adj_b", type=str, required=True, help="e.g. '[[0,0,1],[1,0,0],[0,1,0]]'")
    args = parser.parse_args()

    predictor = AutoregressivePredictor(args.model)
    res = predictor.predict(
        ast.literal_eval(args.ranks_a),
        ast.literal_eval(args.adj_a),
        ast.literal_eval(args.ranks_b),
        ast.literal_eval(args.adj_b)
    )

    print("\n--- Autoregressive Prediction ---")
    print(f"Logits:        {res['logits']}")
    print(f"Probabilities: {res['probabilities']}")
    print("-----------------------------------")
