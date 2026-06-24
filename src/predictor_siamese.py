"""
predictor_siamese.py

Universal predictor for Siamese Seiberg GNNs.
"""

import argparse
import ast
import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SiamesePredictor:
    def __init__(self, model_path, hidden_channels=64, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self.device = torch.device("mps")
        else:
            self.device = device

        from src.model_siamese import SiameseSeiberg
        self.model = SiameseSeiberg(hidden_channels=hidden_channels)

        # print(f"Loading Siamese from {model_path} onto {self.device}")

        if os.path.exists(model_path):
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Use strict=False to handle potential minor key mismatches safely, 
            # but we expect the architecture to match our auto-detection.
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Checkpoint {model_path} not found. Using untrained weights!")

        self.model.to(self.device)
        self.model.eval()

    def _create_data_object(self, ranks, adjacency):
        """
        Converts raw ranks and adjacency to PyG Data.
        """
        N = len(ranks)
        MAX_F32 = 3.4e38
        
        safe_ranks = [MAX_F32 if r > MAX_F32 else (-MAX_F32 if r < -MAX_F32 else float(r)) for r in ranks]
        x_target = torch.tensor(safe_ranks, dtype=torch.float32).view(N, 1)
        
        dual_flag = torch.zeros((N, 1), dtype=torch.float32)
        x = torch.cat([x_target, dual_flag], dim=1)

        safe_adj = [[MAX_F32 if a > MAX_F32 else (-MAX_F32 if a < -MAX_F32 else float(a)) for a in row] for row in adjacency]
        A = torch.tensor(safe_adj, dtype=torch.float32)
        indices = (A > 0).nonzero().t()
        if indices.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = indices.long()
            edge_attr = A[indices[0], indices[1]].view(-1, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = N
        return data

    @torch.no_grad()
    def predict(self, ranks_a, adj_a, ranks_b, adj_b):
        data_a = self._create_data_object(ranks_a, adj_a)
        data_b = self._create_data_object(ranks_b, adj_b)

        batch_a = Batch.from_data_list([data_a]).to(self.device)
        batch_b = Batch.from_data_list([data_b]).to(self.device)

        # Forward pass returning dist (compatible across and V5)
        dist_pred = self.model(batch_a, batch_b)
        predicted_distance = dist_pred.item()
        
        return {
            "estimated_distance": round(predicted_distance, 2)
        }

    @torch.no_grad()
    def predict_batch(self, lists_of_ranks_a, lists_of_adj_a, ranks_b, adj_b):
        """
        Takes a list of mutant states (A candidates) and computes their distance to a single target B.
        """
        if not lists_of_ranks_a:
            return []

        data_b = self._create_data_object(ranks_b, adj_b)
        # Duplicate data_b to match the batch size of A candidates
        # Fix: Use clones to avoid in-place corruption during batch collation
        batch_b = Batch.from_data_list([data_b.clone() for _ in range(len(lists_of_ranks_a))]).to(self.device)
        
        data_a_list = [self._create_data_object(r, a) for r, a in zip(lists_of_ranks_a, lists_of_adj_a)]
        batch_a = Batch.from_data_list(data_a_list).to(self.device)
        
        dist_pred = self.model(batch_a, batch_b)
        
        return dist_pred.view(-1).cpu().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Predictor CLI")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to Siamese checkpoint (.pth). Example: --model_path $CHECKPOINTS_DIR/best_siamese.pth"
    )
    parser.add_argument("--ranks_a", type=str, required=True, help="Ranks for Graph A as a list. Example: --ranks_a '[1, 2, 3]'")
    parser.add_argument(
        "--adj_a", type=str, required=True, help="Adjacency matrix for Graph A as a list of lists. Example: --adj_a '[[0,1,0],[0,0,1],[1,0,0]]'"
    )
    parser.add_argument("--ranks_b", type=str, required=True, help="Ranks for Graph B as a list. Example: --ranks_b '[1, 1, 3]'")
    parser.add_argument(
        "--adj_b", type=str, required=True, help="Adjacency matrix for Graph B as a list of lists. Example: --adj_b '[[0,0,1],[1,0,0],[0,1,0]]'"
    )

    args = parser.parse_args()

    ranks_a = ast.literal_eval(args.ranks_a)
    adj_a = ast.literal_eval(args.adj_a)
    ranks_b = ast.literal_eval(args.ranks_b)
    adj_b = ast.literal_eval(args.adj_b)

    predictor = SiamesePredictor(args.model_path)
    result = predictor.predict(ranks_a, adj_a, ranks_b, adj_b)

    print(f"\n--- Siamese Pure Distance Prediction ---")
    print(f"Estimated Distance:   {result['estimated_distance']}")
    print("---------------------------------------------")
