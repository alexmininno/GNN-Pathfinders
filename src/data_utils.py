import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import IterableDataset, get_worker_info
import os
import os.path as osp
import random
import math
import networkx as nx

from torch_geometric.data import Data


class SeibergData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # Concatenate these flat tensors along dimension 0 when batching
        if key in ["y_delta_b", "y_delta_a", "y_adj", "y_delta_r", "y_rank"]:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def symlog_inv(y):
    """Inverse of symmetric log: x = sign(y) * (exp(|y|) - 1)"""
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)


def create_gnn_inputs(adj_matrix):
    """
    Args:
        adj_matrix (list or array): The NxN dense matrix.
                                    A[i][j] = number of arrows i -> j.
    """
    A = torch.tensor(adj_matrix, dtype=torch.float)
    indices = (A > 0).nonzero().t()
    edge_index = indices.long()
    edge_weights = A[indices[0], indices[1]]
    edge_attr = edge_weights.view(-1, 1)
    return edge_index, edge_attr


class SeibergChunkedDataset(IterableDataset):
    """
    Iterable dataset for a single pre-partitioned node-group folder.

    Expected layout (curriculum):
        root/
        ├── metadata.pt
        ├── train/chunk_0.pt ...
        └── test/chunk_0.pt ...

    The ``num_nodes`` filter is **not** applied when the dataset is loaded
    from a pre-partitioned curriculum folder (all graphs share the same node
    count by construction).  It is still available for backward compatibility
    with the legacy flat layout.
    """

    def __init__(
        self, root, split, normalization="symlog", num_nodes=None, max_depth=None
    ):
        super().__init__()
        self.split_name = split
        self.normalization = normalization
        self.num_nodes = num_nodes
        self.max_depth = max_depth
        self.split_dir = osp.join(root, split)

        if normalization not in ["zscore", "log", "none", "symlog"]:
            raise ValueError(
                "Normalization must be 'zscore', 'log', 'none', or 'symlog'"
            )

        meta_path = osp.join(root, "metadata.pt")
        if not osp.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        self.metadata = torch.load(meta_path, weights_only=False, map_location="cpu")
        self.chunk_size = self.metadata.get("chunk_size", 2000)
        self.max_nodes = self.metadata.get("n_max", 12)
        self.data_dir = osp.join(root, split)
        self.num_examples = self.metadata[f"num_{split}"]

        # Try to use O(1) indexed depth mapping if available
        depth_index_key = f"{split}_depth_index"
        self.chunk_files = []

        if max_depth is not None and depth_index_key in self.metadata:
            depth_map = self.metadata[
                depth_index_key
            ]  # dict of str(chunk_idx) -> [depths]
            for c_idx_str, depths in depth_map.items():
                if any(d <= max_depth for d in depths) or any(d == -1 for d in depths):
                    self.chunk_files.append(
                        osp.join(self.split_dir, f"chunk_{c_idx_str}.pt")
                    )
        else:
            self.chunk_files = [
                os.path.join(self.split_dir, f)
                for f in os.listdir(self.split_dir)
                if f.endswith(".pt")
            ]

        random.seed(42)
        random.shuffle(self.chunk_files)

        self.num_chunks = len(self.chunk_files)

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_chunks = self.chunk_files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_chunks = self.chunk_files[worker_id::num_workers]

        # Shuffle chunks for stochasticity per epoch
        if self.split_name == "train":
            worker_chunks_copy = worker_chunks.copy()
            random.shuffle(worker_chunks_copy)
            worker_chunks = worker_chunks_copy

        for chunk_path in worker_chunks:
            if not osp.exists(chunk_path):
                continue
            try:
                chunk_data = torch.load(
                    chunk_path, weights_only=False, map_location="cpu"
                )
            except Exception as e:
                print(f"Error loading {chunk_path}: {e}")
                continue

            # Sub-chunk shuffling for trains
            if self.split_name == "train":
                random.shuffle(chunk_data)

            while chunk_data:
                data_item = chunk_data.pop()
                data = SeibergData()
                items = data_item.items() if hasattr(data_item, "items") else data_item
                for key, item in items:
                    setattr(data, key, item)

                # Optional node-count filter (used only with legacy flat layout)
                if self.num_nodes is not None and data.num_nodes != self.num_nodes:
                    continue

                # Optional depth filter (used for curriculum learning by depth)
                # data.depth is stored natively as a python list, so its length is the duality depth.
                if self.max_depth is not None:
                    if not hasattr(data, "depth"):
                        continue  # If graph has no depth, skip it in depth-curriculum
                    if len(data.depth) > self.max_depth:
                        continue

                # --- Normalisation ---
                if self.normalization == "zscore":
                    mean = self.metadata["mean"]
                    std = self.metadata["std"]
                    t_mean = self.metadata["target_mean"]
                    t_std = self.metadata["target_std"]
                    if data.x.shape[0] > 0:
                        data.x[:, 0] = (data.x[:, 0] - mean) / std
                    if hasattr(data, "y_rank"):
                        data.y_rank = (data.y_rank - t_mean) / t_std

                elif self.normalization == "log":
                    if data.x.shape[0] > 0:
                        data.x[:, 0] = torch.log1p(data.x[:, 0].to(torch.float32)).to(
                            torch.float32
                        )
                    if hasattr(data, "y_rank"):
                        data.y_rank = torch.log1p(data.y_rank.to(torch.float32)).to(
                            torch.float32
                        )
                    if hasattr(data, "y_adj"):
                        data.y_adj = torch.log1p(data.y_adj.to(torch.float32)).to(
                            torch.float32
                        )

                elif self.normalization == "symlog":
                    if data.x.shape[0] > 0:
                        x_r_raw = data.x[:, 0].clone().to(torch.float32)
                        data.x[:, 0] = (
                            torch.sign(x_r_raw) * torch.log1p(torch.abs(x_r_raw))
                        ).to(torch.float32)

                        if hasattr(data, "y_delta_r"):
                            if isinstance(data.y_delta_r, list):
                                data.y_delta_r = torch.tensor(
                                    data.y_delta_r, dtype=torch.float32
                                )
                            else:
                                data.y_delta_r = data.y_delta_r.to(torch.float32)
                            target_r_raw = x_r_raw + data.y_delta_r.view(-1)
                            target_r_norm = torch.sign(target_r_raw) * torch.log1p(
                                torch.abs(target_r_raw)
                            )
                            data.y_delta_r = (
                                (target_r_norm - data.x[:, 0].to(torch.float32))
                                .flatten()
                                .to(torch.float32)
                            )

                    if hasattr(data, "y_delta_b"):
                        n = data.x.shape[0]
                        b_old_dense = torch.zeros((n, n))
                        row, col = data.edge_index
                        b_old_dense[row, col] = data.edge_attr.view(-1)

                        if isinstance(data.y_delta_b, list):
                            data.y_delta_b = torch.tensor(
                                data.y_delta_b, dtype=torch.float
                            ).view(n, n)
                        b_new_dense = b_old_dense + data.y_delta_b.view(n, n)

                        b_old_norm = torch.sign(b_old_dense) * torch.log1p(
                            torch.abs(b_old_dense)
                        )
                        b_new_norm = torch.sign(b_new_dense) * torch.log1p(
                            torch.abs(b_new_dense)
                        )
                        data.y_delta_b = (b_new_norm - b_old_norm).flatten()

                    # y_delta_a: normalise ΔA in the same symlog frame as edge_attr
                    # edge_attr holds raw A values (positive counts) PRE-symlog transform.
                    # So A_old is reconstructed from edge_attr before it is normalised,
                    # then both A_old and A' = A_old + ΔA are mapped through symlog.
                    if hasattr(data, "y_delta_a"):
                        n = data.x.shape[0]
                        a_old_dense = torch.zeros((n, n))
                        row, col = data.edge_index
                        a_old_dense[row, col] = data.edge_attr.view(-1).to(
                            torch.float32
                        )

                        if isinstance(data.y_delta_a, list):
                            data.y_delta_a = torch.tensor(
                                data.y_delta_a, dtype=torch.float
                            ).view(n, n)
                        a_new_dense = a_old_dense + data.y_delta_a.view(n, n)

                        a_old_norm = torch.sign(a_old_dense) * torch.log1p(
                            torch.abs(a_old_dense)
                        )
                        a_new_norm = torch.sign(a_new_dense) * torch.log1p(
                            torch.abs(a_new_dense)
                        )
                        data.y_delta_a = (a_new_norm - a_old_norm).flatten()

                    data.edge_attr = torch.sign(data.edge_attr) * torch.log1p(
                        torch.abs(data.edge_attr)
                    )

                    if hasattr(data, "y_rank"):
                        data.y_rank = (
                            torch.sign(data.y_rank)
                            * torch.log1p(torch.abs(data.y_rank))
                        ).flatten()

                else:  # "none"
                    data.x = data.x.float()
                    if hasattr(data, "y_rank"):
                        data.y_rank = data.y_rank.float()

                if hasattr(data, "y_adj"):
                    data.y_adj = data.y_adj.flatten().float()

                # Cast any float64 tensors → float32.
                # .pt files are saved with float64 by json_to_pt for
                # precision, but MPS/CUDA cannot handle float64.
                data = data.apply(
                    lambda t: t.to(torch.float32) if t.dtype == torch.float64 else t
                )

                # Strip family_id: it's a non-tensor string used only by
                # generate_dgnn_v3_dataset (which reads .pt directly via
                # torch.load). PyG's Batch.from_data_list cannot collate
                # strings, and old .pt files don't have it → KeyError.
                if hasattr(data, "family_id"):
                    del data["family_id"]

                yield data

    def __len__(self):
        return self.num_examples


def load_data(
    data_path="Databases/seiberg_local_curriculum_dataset/3",
    batch_size=32,
    train_split=0.8,
    normalization="symlog",
    num_workers=4,
    num_nodes=None,
    max_depth=None,
):
    """
    Unified data loader.

    Curriculum layout (preferred):
        data_path = ".../seiberg_local_curriculum_dataset/3"
        → folder contains metadata.pt + train/ + test/

    Legacy flat layout:
        data_path = ".../seiberg_local_dataset"
        → same structure but all node sizes mixed; use num_nodes to filter.

    Monolithic .pt file (legacy):
        data_path = "Databases/Quivers_processed_gnn.pt"
    """
    # ------------------------------------------------------------------ #
    # Disk-based path (curriculum or legacy flat)
    # ------------------------------------------------------------------ #
    if os.path.isdir(data_path):
        print(
            f"Loading chunked dataset from {data_path} " f"(workers={num_workers}) ...",
            flush=True,
        )

        train_dataset = SeibergChunkedDataset(
            data_path,
            "train",
            normalization=normalization,
            num_nodes=num_nodes,
            max_depth=max_depth,
        )
        test_dataset = SeibergChunkedDataset(
            data_path,
            "test",
            normalization=normalization,
            num_nodes=num_nodes,
            max_depth=max_depth,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        if normalization == "zscore":
            meta = train_dataset.metadata
            mean = torch.tensor(meta["mean"])
            std = torch.tensor(meta["std"])
            t_mean = torch.tensor(meta["target_mean"])
            t_std = torch.tensor(meta["target_std"])
        else:
            mean = torch.tensor(0.0)
            std = torch.tensor(1.0)
            t_mean = torch.tensor(0.0)
            t_std = torch.tensor(1.0)

        print(
            f"  Split: {len(train_dataset)} train, {len(test_dataset)} test",
            flush=True,
        )
        return train_loader, test_loader, mean, std, t_mean, t_std

    # ------------------------------------------------------------------ #
    # Legacy monolithic .pt file
    # ------------------------------------------------------------------ #
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading monolithic data from {data_path} ...")
    data_list = torch.load(data_path, weights_only=False, map_location="cpu")

    mean, std = 0.0, 1.0
    target_mean, target_std = 0.0, 1.0

    if normalization == "zscore":
        all_ranks = torch.cat([d.x[:, 0] for d in data_list])
        mean = all_ranks.mean()
        std = all_ranks.std()
        all_targets = [d.y_rank for d in data_list if hasattr(d, "y_rank")]
        if all_targets:
            all_targets = torch.cat(all_targets)
            target_mean = all_targets.mean()
            target_std = all_targets.std()

    for data in data_list:
        if normalization == "zscore":
            data.x[:, 0] = (data.x[:, 0] - mean) / std
            if hasattr(data, "y_rank"):
                data.y_rank = (data.y_rank - target_mean) / target_std
        elif normalization == "log":
            if data.x.shape[0] > 0:
                data.x[:, 0] = torch.log1p(data.x[:, 0])
            if hasattr(data, "y_rank"):
                data.y_rank = torch.log1p(data.y_rank)
            if hasattr(data, "y_adj"):
                data.y_adj = torch.log1p(data.y_adj)
        else:
            data.x = data.x.float()
            if hasattr(data, "y_rank"):
                data.y_rank = data.y_rank.float()
        if hasattr(data, "y_adj"):
            data.y_adj = data.y_adj.float()

    # Group-level split (prevent leakage)
    print("Grouping identical graphs (legacy) ...")
    groups = {}
    for data in data_list:
        edge_index = data.edge_index.t().tolist()
        edge_index.sort()
        edge_sig = tuple(tuple(e) for e in edge_index)
        x_ranks = tuple(round(x, 4) for x in data.x[:, 0].tolist())
        x_dual = (
            tuple(round(x, 4) for x in data.x[:, 1].tolist())
            if data.x.shape[1] > 1
            else ()
        )
        sig = (edge_sig, x_ranks, x_dual)
        groups.setdefault(sig, []).append(data)

    generator = torch.Generator().manual_seed(42)
    group_keys = list(groups.keys())
    perm = torch.randperm(len(group_keys), generator=generator)
    shuffled_keys = [group_keys[i] for i in perm]
    train_idx = int(len(shuffled_keys) * train_split)
    train_data = [d for k in shuffled_keys[:train_idx] for d in groups[k]]
    test_data = [d for k in shuffled_keys[train_idx:] for d in groups[k]]

    print(f"Legacy split: {len(train_data)} train, {len(test_data)} test")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, mean, std, target_mean, target_std

import hashlib

def get_raw_graph_signature(ranks, adj_matrix):
    ranks_tuple = tuple(int(round(r)) for r in ranks)
    adj_tuple = tuple(tuple(int(round(a)) for a in row) for row in adj_matrix)
    return (ranks_tuple, adj_tuple)

def get_graph_hash(ranks, adj_matrix):
    raw_sig = get_raw_graph_signature(ranks, adj_matrix)
    m = hashlib.md5()
    m.update(str(raw_sig).encode("utf-8"))
    return m.digest()

def mutate_ranks(ranks, adj_matrix, k, enforce_anomaly_free=True):
    N_f_in = sum(adj_matrix[i][k] * ranks[i] for i in range(len(ranks)))
    N_f_out = sum(adj_matrix[k][j] * ranks[j] for j in range(len(ranks)))

    if enforce_anomaly_free and N_f_in != N_f_out:
        return None

    new_rank_k = N_f_in - ranks[k]
    MAX_INT64 = 9223372036854775807
    if new_rank_k <= 0 or new_rank_k > MAX_INT64:
        return None
    new_ranks = list(ranks)
    new_ranks[k] = new_rank_k
    return new_ranks

def mutate_adjacency(adj_matrix, k):
    n = len(adj_matrix)
    new_adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i == k or j == k:
                new_adj[i][j] = adj_matrix[j][i]
            else:
                netFlow = (adj_matrix[i][j] + adj_matrix[i][k] * adj_matrix[k][j]) - (
                    adj_matrix[j][i] + adj_matrix[j][k] * adj_matrix[k][i]
                )
                new_adj[i][j] = max(0, netFlow)
    return new_adj

def is_connected(adj_matrix):
    n = len(adj_matrix)
    if n == 0:
        return True
    visited = [False] * n
    queue = [0]
    visited[0] = True
    start = 0
    while start < len(queue):
        curr = queue[start]
        start += 1
        for neighbors in range(n):
            if not visited[neighbors] and (
                adj_matrix[curr][neighbors] > 0 or adj_matrix[neighbors][curr] > 0
            ):
                visited[neighbors] = True
                queue.append(neighbors)
    return all(visited)


def _to_nx_graph(ranks, adj):
    """Convert ranks and adjacency matrix to networkx DiGraph with node attributes."""
    G = nx.DiGraph()
    n = len(ranks)
    for i in range(n):
        G.add_node(i, rank=int(ranks[i]))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0:
                G.add_edge(i, j, weight=int(adj[i][j]))
    return G


def get_wl_hash(ranks, adj):
    """
    Computes a Weisfeiler-Lehman graph hash for a quiver, which is invariant
    under node permutations (isomorphisms).
    """
    G = _to_nx_graph(ranks, adj)
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            h = nx.weisfeiler_lehman_graph_hash(G, node_attr='rank', edge_attr='weight')
    except TypeError:
        # Fallback if edge_attr is not supported in the installed networkx version
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            h = nx.weisfeiler_lehman_graph_hash(G, node_attr='rank')
    return h


def check_isomorphism(r1, a1, r2, a2):
    """Check if two quivers are isomorphic (same structure and ranks)."""
    if len(r1) != len(r2):
        return False
    G1 = _to_nx_graph(r1, a1)
    G2 = _to_nx_graph(r2, a2)
    nm = nx.algorithms.isomorphism.categorical_node_match('rank', -1)
    em = nx.algorithms.isomorphism.categorical_edge_match('weight', 0)
    return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)


def get_isomorphism_mapping(r1, a1, r2, a2):
    """
    Returns the mapping from nodes of quiver 1 to quiver 2, if they are isomorphic.
    Mapping is a dict mapping node index in G1 to node index in G2.
    Returns None if not isomorphic.
    """
    if len(r1) != len(r2):
        return None
    G1 = _to_nx_graph(r1, a1)
    G2 = _to_nx_graph(r2, a2)
    nm = nx.algorithms.isomorphism.categorical_node_match('rank', -1)
    em = nx.algorithms.isomorphism.categorical_edge_match('weight', 0)
    GM = nx.algorithms.isomorphism.DiGraphMatcher(G1, G2, node_match=nm, edge_match=em)
    if GM.is_isomorphic():
        return GM.mapping
    return None


if __name__ == "__main__":
    try:
        train_loader, _, _, _, _, _ = load_data(
            "Databases/seiberg_local_curriculum_dataset/3"
        )
        print("Loader check passed.")
    except Exception as e:
        print(f"Loader check warning (expected if DB missing): {e}")
