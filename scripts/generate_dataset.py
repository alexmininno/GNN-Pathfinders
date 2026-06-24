import os
import argparse
import json
import random
import multiprocessing as mp
import numpy as np
import torch
import shutil
import gc
import uuid
import hashlib
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from collections import defaultdict

# Configure PyTorch to use the file system instead of file descriptors
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# ---------------------------------------------------------------------------
# Global Cache for Zero-Copy Multiprocessing (Leverages Linux Fork)
# ---------------------------------------------------------------------------
_SHARED_DATA = {}


def init_apsp_worker(adj, mutations):
    """Initializes workers by giving them read access to the parent process's memory.
    Note: We no longer pass raw states to workers because PyTorch object
    construction will safely occur in the parent process."""
    global _SHARED_DATA
    _SHARED_DATA["adj"] = adj
    _SHARED_DATA["mutations"] = mutations


# ---------------------------------------------------------------------------
# Numba Compiled Kernels
# ---------------------------------------------------------------------------


@njit(nogil=True)
def mutate_ranks_kernel(ranks, adj, k, enforce_anomaly_free=True):
    n = len(ranks)
    n_f_in = np.int64(0)
    n_f_out = np.int64(0)
    LIMIT = np.int64(800000000)

    for i in range(n):
        if adj[i, k] > LIMIT or ranks[i] > LIMIT or adj[k, i] > LIMIT:
            return np.zeros(0, dtype=np.int64)

        n_f_in += adj[i, k] * ranks[i]
        n_f_out += adj[k, i] * ranks[i]

    if enforce_anomaly_free and n_f_in != n_f_out:
        return np.zeros(0, dtype=np.int64)

    new_rank_k = n_f_in - ranks[k]
    if new_rank_k <= 0:
        return np.zeros(0, dtype=np.int64)

    new_ranks = ranks.copy()
    new_ranks[k] = new_rank_k
    return new_ranks


@njit(nogil=True)
def mutate_adjacency_kernel(adj, k):
    n = adj.shape[0]
    new_adj = np.zeros((n, n), dtype=np.int64)
    LIMIT = np.int64(800000000)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i == k or j == k:
                new_adj[i, j] = adj[j, i]
            else:
                if (
                    adj[i, k] > LIMIT
                    or adj[k, j] > LIMIT
                    or adj[j, k] > LIMIT
                    or adj[k, i] > LIMIT
                ):
                    return np.zeros((0, 0), dtype=np.int64)

                netFlow = (adj[i, j] + adj[i, k] * adj[k, j]) - (
                    adj[j, i] + adj[j, k] * adj[k, i]
                )
                new_adj[i, j] = max(0, netFlow)
    return new_adj


@njit(nogil=True)
def is_connected_kernel(adj):
    n = adj.shape[0]
    if n == 0:
        return True
    visited = np.zeros(n, dtype=np.bool_)
    queue = np.zeros(n, dtype=np.int64)
    head = 0
    tail = 0

    queue[tail] = 0
    tail += 1
    visited[0] = True

    count = 1
    while head < tail:
        curr = queue[head]
        head += 1
        for neighbor in range(n):
            if not visited[neighbor] and (
                adj[curr, neighbor] > 0 or adj[neighbor, curr] > 0
            ):
                visited[neighbor] = True
                queue[tail] = neighbor
                tail += 1
                count += 1

    return count == n


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def get_raw_graph_signature(ranks, adj_matrix):
    ranks_tuple = tuple(int(round(r)) for r in ranks)
    adj_tuple = tuple(tuple(int(round(a)) for a in row) for row in adj_matrix)
    return (ranks_tuple, adj_tuple)


def load_family_metadata(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def save_family_metadata(meta_path, mapping):
    with open(meta_path, "w") as f:
        json.dump(mapping, f, indent=2)


def get_family_id(sig, mapping, name="Unknown"):
    key = json.dumps(sig, separators=(",", ":"))
    if key not in mapping:
        mapping[key] = {"uuid": str(uuid.uuid4()), "name": name}
    elif isinstance(mapping[key], str):
        mapping[key] = {"uuid": mapping[key], "name": name}
    return mapping[key]["uuid"]


def get_exact_state(ranks, adj):
    ranks_tuple = tuple(int(r) for r in ranks)
    adj_tuple = tuple(tuple(int(a) for a in row) for row in adj)
    m = hashlib.md5()
    m.update(str((ranks_tuple, adj_tuple)).encode("utf-8"))
    cert = m.digest()
    return cert, ranks, adj


def build_pyg_data(ranks, adj, dist_from_root, family_id):
    num_nodes = len(ranks)
    A = torch.tensor(adj, dtype=torch.int64)
    indices = (A > 0).nonzero().t()

    if indices.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.int64)
    else:
        edge_index = indices.long()
        edge_attr = A[indices[0], indices[1]].view(-1, 1)

    ranks_tensor = torch.tensor(ranks, dtype=torch.int64).view(-1, 1)
    dual_flag = torch.zeros((num_nodes, 1), dtype=torch.int64)
    x = torch.cat([ranks_tensor, dual_flag], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.depth = [0] * int(dist_from_root)
    data.family_id = family_id
    return data


# ---------------------------------------------------------------------------
# Worker Functions
# ---------------------------------------------------------------------------


def worker_expand_chunk(args_tuple):
    chunk, max_rank, max_arrows, enforce_anomaly_free = args_tuple
    local_edges = []
    local_new_states = {}

    for source_cert, ranks_bytes, adj_bytes, n_nodes in chunk:
        ranks = np.frombuffer(ranks_bytes, dtype=np.int64)
        adj = np.frombuffer(adj_bytes, dtype=np.int64).reshape((n_nodes, n_nodes))

        for k in range(n_nodes):
            new_ranks = mutate_ranks_kernel(ranks, adj, k, enforce_anomaly_free)
            if new_ranks.size == 0:
                continue

            new_adj = mutate_adjacency_kernel(adj, k)
            if new_adj.size == 0:
                continue

            if not is_connected_kernel(new_adj):
                continue
            if max_rank is not None and np.any(new_ranks >= max_rank):
                continue
            if max_arrows is not None and np.any(new_adj >= max_arrows):
                continue

            target_cert, exact_ranks, exact_adj = get_exact_state(new_ranks, new_adj)

            local_edges.append((source_cert, target_cert, k + 1))
            local_new_states[target_cert] = (exact_ranks.tobytes(), exact_adj.tobytes())

    return local_edges, local_new_states


def worker_extract_paths_zero_copy(args_tuple):
    # The worker returns ONLY native Python types (int, float, list),
    # completely bypassing PyTorch's inter-process mmap overhead.
    batch_sources, dists_to_generate, max_pairs, split_ratio = args_tuple

    global _SHARED_DATA
    adj_sparse = _SHARED_DATA["adj"]
    edge_mutations = _SHARED_DATA["mutations"]

    local_results = {d: [] for d in dists_to_generate}

    dist_batch, pred_batch = shortest_path(
        csgraph=adj_sparse,
        directed=False,
        indices=batch_sources,
        unweighted=True,
        return_predecessors=True,
    )

    for d in dists_to_generate:
        mask = dist_batch == d
        row_indices, col_indices = np.where(mask)

        valid_pairs = []
        for r, c in zip(row_indices, col_indices):
            i = batch_sources[r]
            j = c
            if j > i:
                valid_pairs.append((i, j, r))

        if len(valid_pairs) > 0:
            if max_pairs is not None and len(valid_pairs) > max_pairs:
                chosen_idx = np.random.choice(
                    len(valid_pairs), size=max_pairs, replace=False
                )
                valid_pairs = [valid_pairs[k] for k in chosen_idx]

            for i, j, r in valid_pairs:
                pred_array = pred_batch[r]
                path = []
                if d > 0:
                    curr = j
                    while curr != i:
                        prev = pred_array[curr]
                        if prev < 0:
                            break
                        path.append(edge_mutations[(prev, curr)])
                        curr = prev
                    path.reverse()

                split = "train" if random.random() < split_ratio else "test"
                # Send INDICES instead of Tensors
                local_results[d].append((int(i), int(j), float(d), path, split))

    return local_results


# ---------------------------------------------------------------------------
# Core Execution
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified Theories Dataset Generator (Zero-Copy Architecture - Exact Limits)"
    )
    parser.add_argument(
        "--input_db",
        type=str,
        default="Databases/BasicTheoriesData_100.json",
        help="Path to input basic theories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Databases/Theories_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing dataset folders for the selected nodes before generating.",
    )

    # Node Filtering
    parser.add_argument("--min_nodes", type=int, default=3, help="Min node count")
    parser.add_argument("--max_nodes", type=int, default=12, help="Max node count")
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific list of nodes to generate (overrides min/max)",
    )

    # Dualization Control
    parser.add_argument(
        "--bfs_depth",
        type=int,
        default=3,
        help="Max depth for the Breadth-First-Search tree generated per family.",
    )

    # Distance Constraints
    parser.add_argument(
        "--min_dist", type=int, default=-1, help="Minimum pairwise distance to save."
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=-1,
        help="Maximum pairwise distance to save (-1 for no limit, pruned by bfs_depth).",
    )
    parser.add_argument(
        "--dists",
        type=int,
        nargs="+",
        default=None,
        help="Specific list of pairwise distances to save (overrides min/max dist).",
    )

    # Filtering
    parser.add_argument(
        "--max_rank",
        type=int,
        default=None,
        help="Filter out theories with ranks >= this value",
    )
    parser.add_argument(
        "--max_arrows",
        type=int,
        default=None,
        help="Filter out theories with arrows >= this value",
    )
    parser.add_argument(
        "--relax_anomaly",
        action="store_true",
        help="Skip the anomaly-free check (N_f_in == N_f_out) during mutation; only positive ranks and connectivity are enforced",
    )

    # Execution
    parser.add_argument(
        "--chunk_size", type=int, default=5000, help="Number of pairs per file chunk"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of multiprocessing workers",
    )
    parser.add_argument(
        "--max_pairs_per_dist",
        type=int,
        default=None,
        help="Maximum number of random pairs to generate per distance bucket to prevent disk exhaustion",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    enforce_anomaly_free = not args.relax_anomaly

    with open(args.input_db, "r") as f:
        basic_theories = json.load(f)

    meta_path = os.path.join(os.path.dirname(args.input_db), "family_metadata.json")
    family_mapping = load_family_metadata(meta_path)
    print(f"[Init] Loaded {len(family_mapping)} families. Workers: {args.num_workers}")

    nodes_to_process = (
        args.nodes if args.nodes else list(range(args.min_nodes, args.max_nodes + 1))
    )
    allowed_dists = set(args.dists) if args.dists is not None else None

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    for target_nodes in nodes_to_process:
        print(f"\n===================================")
        print(f"Processing Graph Size: {target_nodes} Nodes")
        print(f"===================================")

        nodes_out_dir = os.path.join(args.output_dir, str(target_nodes))
        os.makedirs(nodes_out_dir, exist_ok=True)

        active_seeds = [t for t in basic_theories if len(t["Ranks"]) == target_nodes]

        if allowed_dists is not None:
            dists_to_generate = list(allowed_dists)
        else:
            _min = max(0, args.min_dist)
            _max = args.max_dist if args.max_dist >= 0 else args.bfs_depth
            dists_to_generate = list(range(_min, _max + 1))

        completed_dists_path = os.path.join(
            nodes_out_dir, "completed_families_dists.json"
        )
        completed_families = (
            json.load(open(completed_dists_path))
            if os.path.exists(completed_dists_path)
            else {}
        )

        if args.clear and os.path.exists(nodes_out_dir):
            for d in dists_to_generate:
                dist_dir = os.path.join(nodes_out_dir, f"dist_{d}")
                if os.path.exists(dist_dir):
                    shutil.rmtree(dist_dir)
            for fam, dists in completed_families.items():
                completed_families[fam] = [
                    x for x in dists if x not in dists_to_generate
                ]
            with open(completed_dists_path, "w") as f:
                json.dump(completed_families, f)

        for seed_idx, theory in enumerate(active_seeds):
            ranks_raw = theory["Ranks"]
            adj_raw = theory["Adjacency"]
            theory_name = theory.get("Name", "Unknown")

            raw_sig = get_raw_graph_signature(ranks_raw, adj_raw)
            base_theory_id = get_family_id(raw_sig, family_mapping, name=theory_name)

            completed_for_fam = completed_families.get(base_theory_id, [])
            missing_dists = [d for d in dists_to_generate if d not in completed_for_fam]
            if not missing_dists:
                continue

            print(
                f"  [{seed_idx+1}/{len(active_seeds)}] Family {base_theory_id} | BFS Depth {args.bfs_depth} ..."
            )

            initial_ranks = np.array(ranks_raw, dtype=np.int64)
            initial_adj = np.array(adj_raw, dtype=np.int64)
            root_cert, c_ranks, c_adj = get_exact_state(initial_ranks, initial_adj)

            visited_states = {root_cert: (c_ranks.tobytes(), c_adj.tobytes())}
            edges_list = []
            current_queue = [root_cert]

            # --- PHASE 1: BFS ---
            with mp.Pool(processes=args.num_workers) as pool_bfs:
                for depth in range(args.bfs_depth):
                    chunk_size = max(1, len(current_queue) // (args.num_workers * 2))
                    tasks = []
                    for i in range(0, len(current_queue), chunk_size):
                        batch_keys = current_queue[i : i + chunk_size]
                        chunk = [
                            (
                                k_cert,
                                visited_states[k_cert][0],
                                visited_states[k_cert][1],
                                target_nodes,
                            )
                            for k_cert in batch_keys
                        ]
                        tasks.append(
                            (
                                chunk,
                                args.max_rank,
                                args.max_arrows,
                                enforce_anomaly_free,
                            )
                        )

                    next_queue_set = set()
                    for local_edges, local_new_states in pool_bfs.imap_unordered(
                        worker_expand_chunk, tasks
                    ):
                        edges_list.extend(local_edges)
                        for t_cert, data_tuple in local_new_states.items():
                            if t_cert not in visited_states:
                                visited_states[t_cert] = data_tuple
                                next_queue_set.add(t_cert)

                    current_queue = list(next_queue_set)

            save_family_metadata(meta_path, family_mapping)

            # Data Preparation
            byte_to_idx = {b: i for i, b in enumerate(visited_states.keys())}
            n_states = len(byte_to_idx)
            if n_states == 0:
                continue

            row_idx = [byte_to_idx[u] for u, v, _ in edges_list]
            col_idx = [byte_to_idx[v] for u, v, _ in edges_list]
            data = np.ones(len(edges_list))

            edge_mutations = {}
            for u, v, mut in edges_list:
                i, j = byte_to_idx[u], byte_to_idx[v]
                edge_mutations[(i, j)] = mut
                edge_mutations[(j, i)] = mut

            adj_sparse = coo_matrix(
                (data, (row_idx, col_idx)), shape=(n_states, n_states)
            ).tocsr()
            dist_from_root_array = shortest_path(
                csgraph=adj_sparse,
                directed=False,
                indices=byte_to_idx[root_cert],
                unweighted=True,
            )
            dist_from_root_array[np.isinf(dist_from_root_array)] = 0

            idx_to_state_bytes = [None] * n_states
            for b, state_tuple in visited_states.items():
                idx_to_state_bytes[byte_to_idx[b]] = state_tuple

            # Global Buffers and Counters managed exclusively by the parent
            counts_per_dist = {d: 0 for d in dists_to_generate}
            split_buffers = {"train": defaultdict(list), "test": defaultdict(list)}

            # Recover correct chunk numbering if interrupted
            chunk_counters = {"train": defaultdict(int), "test": defaultdict(int)}
            for split in ["train", "test"]:
                for d in dists_to_generate:
                    d_dir = os.path.join(nodes_out_dir, f"dist_{d}", split)
                    if os.path.exists(d_dir):
                        files = [
                            f
                            for f in os.listdir(d_dir)
                            if f.startswith("chunk_") and f.endswith(".pt")
                        ]
                        ids = [
                            int(f.split("_")[1].split(".")[0])
                            for f in files
                            if "_" in f
                        ]
                        if ids:
                            chunk_counters[split][d] = max(ids) + 1

            all_sources = np.random.permutation(n_states)
            batch_size = 500

            apsp_tasks = []
            for start_idx in range(0, n_states, batch_size):
                end_idx = min(start_idx + batch_size, n_states)
                batch_sources = all_sources[start_idx:end_idx]
                task_args = (
                    batch_sources,
                    dists_to_generate,
                    args.max_pairs_per_dist,
                    args.split_ratio,
                )
                apsp_tasks.append(task_args)

            print(
                f"    -> Distributing path computation over {len(apsp_tasks)} batches (Zero-Copy) to workers..."
            )

            with mp.Pool(
                processes=args.num_workers,
                initializer=init_apsp_worker,
                initargs=(adj_sparse, edge_mutations),
            ) as pool_apsp:

                for local_results in pool_apsp.imap_unordered(
                    worker_extract_paths_zero_copy, apsp_tasks
                ):
                    all_dists_done = True

                    for d in dists_to_generate:
                        # If we are already done with this distance, skip it
                        if (
                            args.max_pairs_per_dist is not None
                            and counts_per_dist[d] >= args.max_pairs_per_dist
                        ):
                            continue

                        items = local_results.get(d, [])
                        if not items:
                            all_dists_done = False
                            continue

                        # Slicing Logic: take exactly what we need
                        if args.max_pairs_per_dist is not None:
                            needed = args.max_pairs_per_dist - counts_per_dist[d]
                            if len(items) > needed:
                                items = items[:needed]

                        # Build PyTorch GNN tensors directly in the Parent Thread!
                        for idx_i, idx_j, d_float, path, split in items:

                            # Extraction and construction for Graph I
                            r_bytes_i, a_bytes_i = idx_to_state_bytes[idx_i]
                            r_i = np.frombuffer(r_bytes_i, dtype=np.int64)
                            a_i = np.frombuffer(a_bytes_i, dtype=np.int64).reshape(
                                (target_nodes, target_nodes)
                            )
                            pyg_i = build_pyg_data(
                                r_i, a_i, dist_from_root_array[idx_i], base_theory_id
                            )

                            # Extraction and construction for Graph J
                            r_bytes_j, a_bytes_j = idx_to_state_bytes[idx_j]
                            r_j = np.frombuffer(r_bytes_j, dtype=np.int64)
                            a_j = np.frombuffer(a_bytes_j, dtype=np.int64).reshape(
                                (target_nodes, target_nodes)
                            )
                            pyg_j = build_pyg_data(
                                r_j, a_j, dist_from_root_array[idx_j], base_theory_id
                            )

                            split_buffers[split][d].append(
                                (pyg_i, pyg_j, d_float, path)
                            )
                            counts_per_dist[d] += 1

                            if len(split_buffers[split][d]) >= args.chunk_size:
                                dist_dir = os.path.join(
                                    nodes_out_dir, f"dist_{d}", split
                                )
                                os.makedirs(dist_dir, exist_ok=True)
                                torch.save(
                                    split_buffers[split][d],
                                    os.path.join(
                                        dist_dir,
                                        f"chunk_{chunk_counters[split][d]}_{uuid.uuid4().hex[:4]}.pt",
                                    ),
                                )
                                chunk_counters[split][d] += 1
                                split_buffers[split][d] = []

                        if (
                            args.max_pairs_per_dist is None
                            or counts_per_dist[d] < args.max_pairs_per_dist
                        ):
                            all_dists_done = False

                    # Aggressive shutdown if the exact target has been reached in every bucket
                    if all_dists_done:
                        print(
                            "    -> Exact limit reached for all distances! Immediate termination of queued workers."
                        )
                        pool_apsp.terminate()
                        break

            # Save remaining items in buffers
            for split in ["train", "test"]:
                for d, buf in split_buffers[split].items():
                    if buf:
                        dist_dir = os.path.join(nodes_out_dir, f"dist_{d}", split)
                        os.makedirs(dist_dir, exist_ok=True)
                        torch.save(
                            buf,
                            os.path.join(
                                dist_dir,
                                f"chunk_{chunk_counters[split][d]}_{uuid.uuid4().hex[:4]}.pt",
                            ),
                        )
                        chunk_counters[split][d] += 1

            completed_families.setdefault(base_theory_id, []).extend(missing_dists)
            completed_families[base_theory_id] = list(
                set(completed_families[base_theory_id])
            )
            with open(completed_dists_path, "w") as f:
                json.dump(completed_families, f)

            del (
                visited_states,
                edges_list,
                idx_to_state_bytes,
                dist_from_root_array,
                adj_sparse,
                apsp_tasks,
            )
            gc.collect()

        # Generation of Metadata
        total_train, total_test = 0, 0
        for d_name in os.listdir(nodes_out_dir):
            if d_name.startswith("dist_"):
                for split in ["train", "test"]:
                    split_dir = os.path.join(nodes_out_dir, d_name, split)
                    if os.path.exists(split_dir):
                        for f in os.listdir(split_dir):
                            if f.endswith(".pt"):
                                count = len(
                                    torch.load(
                                        os.path.join(split_dir, f),
                                        map_location="cpu",
                                        weights_only=False,
                                    )
                                )
                                if split == "train":
                                    total_train += count
                                else:
                                    total_test += count

        meta_out = {
            "node_group": target_nodes,
            "num_train": total_train,
            "num_test": total_test,
        }
        torch.save(meta_out, os.path.join(nodes_out_dir, "metadata.pt"))
        print(
            f"[{target_nodes} Nodes] Metadata saved. Train: {total_train}, Test: {total_test}"
        )

    print("\n[✔] Unified Dataset Generation Complete!")


if __name__ == "__main__":
    main()
