import gc
import torch
import os
import random
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.transforms import AddLaplacianEigenvectorPE


import numpy as np
import warnings
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

def safe_laplacian_pe(data, k=8):
    """
    Computes Laplacian Positional Encodings safely for graphs with N < k.
    PyG's AddLaplacianEigenvectorPE crashes when N < k due to a sign multiplication bug.
    """
    N = data.num_nodes
    if N <= 1:
        data.pe = torch.zeros(N, k, dtype=torch.float32)
        data.is_valid = False # Flag for filtering
        return data

    # Fallback to None if edge_attr is not suitable for weights
    edge_weight = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if data.edge_attr.dim() == 1:
            edge_weight = data.edge_attr.float()
        elif data.edge_attr.dim() == 2 and data.edge_attr.shape[1] == 1:
            edge_weight = data.edge_attr.squeeze(-1).float()

    edge_index, edge_weight = get_laplacian(
        data.edge_index, edge_weight, normalization='sym', num_nodes=N)
    
    L = to_scipy_sparse_matrix(edge_index, edge_weight, N)
    
    L_array = L.toarray()
    if not np.isfinite(L_array).all():
        data.pe = torch.zeros(N, k, dtype=torch.float32)
        data.is_valid = False
        return data

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eig_vals, eig_vecs = np.linalg.eigh(L_array)
    except Exception as e:
        # warnings.warn(f"Laplacian PE failed (N={N}): {e}")
        data.pe = torch.zeros(N, k, dtype=torch.float32)
        data.is_valid = False # Flag for filtering
        return data

    max_k = min(k + 1, N)
    eig_vecs = eig_vecs[:, 1:max_k]
    
    pe = torch.from_numpy(eig_vecs).float()
    if pe.shape[1] < k:
        padding = torch.zeros(N, k - pe.shape[1], dtype=torch.float32)
        pe = torch.cat([pe, padding], dim=1)
        
    sign = -1 + 2 * torch.randint(0, 2, (k,))
    pe = pe * sign
    
    data.pe = pe
    data.is_valid = True
    return data

def process_dgnn_chunk(chunk_data, k=8):
    """
    Strips metadata, casts to float32, and applies PE transform if provided.
    Filters out any pair where Laplacian PE failed.
    """
    keys_to_keep = {'x', 'edge_index', 'edge_attr'}
    processed_samples = []
    for i in range(len(chunk_data)):
        g_a, g_b, dist, seq = chunk_data[i]
        processed_graphs = []
        for g in (g_a, g_b):
            # Absolute first step: enforce 2D node features shape sanity check
            if hasattr(g, 'x') and g.x is not None:
                if g.x.dim() == 1:
                    g.x = g.x.unsqueeze(-1)
                
            # Strip metadata that causes PyG collate KeyErrors
            for key in list(g.keys()):
                if key not in keys_to_keep:
                    delattr(g, key)
                    
            for key in ['x', 'edge_attr']:
                if hasattr(g, key) and getattr(g, key) is not None:
                    t = getattr(g, key)
                    if t.dtype in (torch.float64, torch.int64):
                        setattr(g, key, t.to(torch.float32))
            

            
            # Add Laplacian Positional Encodings
            if k > 0:
                g = safe_laplacian_pe(g, k=k)
            
            # Ensure pe is float32
            if hasattr(g, 'pe') and getattr(g, 'pe') is not None:
                t = getattr(g, 'pe')
                if t.dtype in (torch.float64, torch.int64):
                    setattr(g, 'pe', t.to(torch.float32))
                    
            processed_graphs.append(g)
            
        # Filter out invalid graphs
        if getattr(processed_graphs[0], 'is_valid', True) and getattr(processed_graphs[1], 'is_valid', True):
            processed_samples.append((processed_graphs[0], processed_graphs[1], dist, seq))
            
    return processed_samples

def process_agnn_chunk(chunk_data, k=8):
    """
    Refines pairs for AGNN classification.
    Target is the first node index in the mutation sequence (0-indexed).
    Does NOT filter by distance; extracts seq[0] for all provided samples.
    Filters out any pair where Laplacian PE failed.
    """
    keys_to_keep = {'x', 'edge_index', 'edge_attr'}
    processed_samples = []
    
    for i in range(len(chunk_data)):
        g_a, g_b, dist, seq = chunk_data[i]
        
        # seq is 1-indexed in the .pt files, so we subtract 1 for 0-indexed target
        if len(seq) == 0 or int(seq[0]) <= 0:
            continue
            
        target_node_idx = int(seq[0]) - 1
        
        processed_graphs = []
        for g in (g_a, g_b):
            # Absolute first step: enforce 2D node features shape sanity check
            if hasattr(g, 'x') and g.x is not None:
                if g.x.dim() == 1:
                    g.x = g.x.unsqueeze(-1)
                
            # Strip metadata
            for key in list(g.keys()):
                if key not in keys_to_keep:
                    delattr(g, key)
                    
            for key in ['x', 'edge_attr']:
                if hasattr(g, key) and getattr(g, key) is not None:
                    t = getattr(g, key)
                    if t.dtype in (torch.float64, torch.int64):
                        setattr(g, key, t.to(torch.float32))
            

            
            # Add Laplacian PE
            if k > 0:
                g = safe_laplacian_pe(g, k=k)
            
            if hasattr(g, 'pe') and getattr(g, 'pe') is not None:
                t = getattr(g, 'pe')
                if t.dtype in (torch.float64, torch.int64):
                    setattr(g, 'pe', t.to(torch.float32))
                    
            processed_graphs.append(g)
            
        # Filter out invalid graphs
        if getattr(processed_graphs[0], 'is_valid', True) and getattr(processed_graphs[1], 'is_valid', True):
            processed_samples.append((processed_graphs[0], processed_graphs[1], target_node_idx, dist))
        
    return processed_samples


class DGNNIterableDataset(IterableDataset):
    """
    Offline Static DGNN Dataset.
    Loads graph pairs and provides:
    - Graph A & Graph B
    - Topological Distance
    - Action Path (node mutation sequence)
    """

    def __init__(self, data_dir, split="train", seed=42):
        super().__init__()
        self.split = split
        self.seed = seed
        self.chunk_files = []
        self.rng = random.Random(seed)
        self.update_data_dir(data_dir)

    def update_data_dir(self, data_dir):
        """Updates the directory to load chunks from, e.g., for curriculum learning"""
        split_dir = os.path.join(data_dir, self.split)
        if os.path.exists(split_dir):
            self.data_dir = data_dir
            self.chunk_files = [
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(".pt")
            ]
            # Use local RNG for shuffles to avoid messing with global state
            self.rng.shuffle(self.chunk_files)
        else:
            print(
                f"  [!] Notice: Dataset dir {data_dir} not found. Continuing with: {getattr(self, 'data_dir', 'None')}"
            )

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_chunks = self.chunk_files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_chunks = self.chunk_files[worker_id::num_workers]

        # Shuffle chunks for stochasticity per epoch
        if self.split == "train":
            worker_chunks_copy = worker_chunks.copy()
            self.rng.shuffle(worker_chunks_copy)
            worker_chunks = worker_chunks_copy

        for chunk_path in worker_chunks:
            if not os.path.exists(chunk_path):
                continue
            try:
                # Force explicit file handle closure
                with open(chunk_path, 'rb') as f:
                    chunk_data = torch.load(f, map_location="cpu", weights_only=False)
                
                chunk_data = process_dgnn_chunk(chunk_data, k=8)

            except Exception as e:
                print(f"Error loading {chunk_path}: {e}")
                continue

            if self.split == "train":
                self.rng.shuffle(chunk_data)

            while chunk_data:
                g_a, g_b, dist, seq = chunk_data.pop()
                yield g_a, g_b, torch.as_tensor(dist, dtype=torch.float32), torch.as_tensor(seq, dtype=torch.long)


def collate_dgnn(batch):
    """
    Takes a list of tuples: (graph_a, graph_b, distance, sequence)
    Instead of returning a padded sequence of tokens, it returns a 
    flat node-level target counting how many times each node was dualized.
    """
    graph_a_list = [item[0] for item in batch]
    graph_b_list = [item[1] for item in batch]
    distances = torch.stack([item[2] for item in batch])

    # sequences are 1-indexed node IDs specifying the path.
    sequences = [item[3] for item in batch]
    
    target_counts_list = []
    max_nodes = max([g.num_nodes for g in graph_a_list]) if graph_a_list else 0

    for i, seq in enumerate(sequences):
        g_nodes = graph_a_list[i].num_nodes
        # 1-indexed -> 0-indexed
        valid_steps = seq[seq > 0] - 1
        # count occurrences for each node up to g_nodes length
        counts = torch.bincount(valid_steps, minlength=g_nodes).float()
        target_counts_list.append(counts)

    # Flat targets aligning with the batched Graph A's node list
    flat_target_counts = torch.cat(target_counts_list) if target_counts_list else torch.tensor([], dtype=torch.float32)

    # Use native C++ padding
    padded_target_counts = pad_sequence(target_counts_list, batch_first=True)

    batch_a = Batch.from_data_list(graph_a_list)
    batch_b = Batch.from_data_list(graph_b_list)

    return batch_a, batch_b, distances.unsqueeze(1), flat_target_counts, padded_target_counts


def collate_agnn(batch, explosion_threshold=1e6):
    """
    Collate for AGNN classification.
    Returns (batch_a, batch_b, target_node_indices, action_mask)
    """
    from torch_geometric.utils import to_dense_batch
    
    g_a_list = [item[0] for item in batch]
    g_b_list = [item[1] for item in batch]
    targets = torch.tensor([item[2] for item in batch], dtype=torch.long)
    distances = torch.tensor([item[3] for item in batch], dtype=torch.long)
    
    batch_a = Batch.from_data_list(g_a_list)
    batch_b = Batch.from_data_list(g_b_list)
    
    # Compute action mask for batch_a fully vectorized on the active device
    N_total = batch_a.x.shape[0]
    device = batch_a.x.device
    ranks = batch_a.x[:, 0].float()
    
    n_f_in = torch.zeros(N_total, dtype=torch.float32, device=device)
    if batch_a.edge_index.numel() > 0:
        u = batch_a.edge_index[0]
        v = batch_a.edge_index[1]
        if batch_a.edge_attr is not None:
            weight = batch_a.edge_attr.float().view(-1)
        else:
            weight = torch.ones(batch_a.edge_index.shape[1], dtype=torch.float32, device=device)
        n_f_in.scatter_add_(0, v, weight * ranks[u])
        
    new_ranks = n_f_in - ranks
    valid_actions = (new_ranks > 0) & (new_ranks <= explosion_threshold)
    
    # For graphs where no action is valid, default to all True
    graph_has_valid = torch.zeros(batch_a.num_graphs, dtype=torch.bool, device=device)
    graph_has_valid.index_fill_(0, batch_a.batch[valid_actions], True)
    
    valid_actions = valid_actions | ~graph_has_valid[batch_a.batch]
    
    action_mask, _ = to_dense_batch(valid_actions, batch_a.batch, fill_value=False)
    
    return batch_a, batch_b, targets, action_mask, distances


class CurriculumMixedDataset(IterableDataset):
    def __init__(self, quotas, db_path, split="train", seed=42, max_yields=None, agnn=False, pe_channels=8):
        self.quotas = quotas
        self.db_path = db_path
        self.split = split
        self.seed = seed
        self.max_yields = max_yields
        self.agnn = agnn
        self.pe_channels = pe_channels
        
        self._populate_chunk_map()

    def _populate_chunk_map(self):
        """Scans the database directory to find all available data chunks for the current quotas."""
        self.chunk_map = {}
        for (n, d), q in self.quotas.items():
            if q <= 0: continue
            path_nd = os.path.join(self.db_path, str(n), f"dist_{int(d)}")
            split_dir = os.path.join(path_nd, self.split)
            if os.path.exists(split_dir):
                chunks = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")]
                self.chunk_map[(n, d)] = chunks

    def update_distances(self, dist_string):
        """
        Force the dataset to re-scan directories for a new set of distances.
        Useful when resuming training from a checkpoint that was at a further curriculum stage.
        """
        dist_list = [int(x) for x in str(dist_string).split(",")]
        # Keep the same nodes we were already training on
        nodes = sorted(list(set([n for (n, d) in self.quotas.keys()])))
        
        # Re-build quotas
        self.quotas = {(n, d): 1.0 for n in nodes for d in dist_list}
        
        # Re-populate chunk map
        self._populate_chunk_map()
        
        print(f"  [Dataset] Updated distances to {dist_string}. Found {sum(len(c) for c in self.chunk_map.values())} chunks.")

    def __len__(self):
        """ Returns the total number of samples across all loaded chunks. """
        if self.max_yields is not None:
            return self.max_yields
        
        # Each chunk in this dataset typically contains 5000 samples.
        # However, we can be more precise by counting chunks.
        total_chunks = sum(len(chunks) for chunks in self.chunk_map.values())
        return total_chunks * 5000 

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Determine total yields across workers
        max_yields = self.max_yields
        if max_yields is None:
            total_chunks = sum(len(chunks) for chunks in self.chunk_map.values())
            max_yields = total_chunks * 5000
            
        worker_max_yields = max_yields // num_workers + 1
        
        # Identify active keys with non-empty chunk lists
        active_keys = [k for k, chunks in self.chunk_map.items() if len(chunks) > 0]
        if not active_keys:
            return
            
        # We estimate each chunk yields about 5000 samples, but we want to ensure
        # high representation across active keys even for small yields:
        worker_chunks_to_yield = max(len(active_keys) * 2, worker_max_yields // 4000 + 2)
        
        rng = random.Random(self.seed + worker_id)
        
        # Get weights for active keys
        weights = [self.quotas[k] for k in active_keys]
        sum_weights = sum(weights)
        if sum_weights <= 0:
            return
            
        # Sample keys according to their weights
        sampled_keys = rng.choices(active_keys, weights=weights, k=worker_chunks_to_yield)
        
        # For each sampled key, choose a chunk path randomly from its available chunks
        weighted_chunk_list = []
        for key in sampled_keys:
            chunks = self.chunk_map[key]
            chunk_path = rng.choice(chunks)
            weighted_chunk_list.append((key, chunk_path))
            
        # Shuffle the chunk list to mix node sizes and distances across the epoch
        rng.shuffle(weighted_chunk_list)
        
        yielded_count = 0
        
        for key, chunk_path in weighted_chunk_list:
            if worker_max_yields is not None and yielded_count >= worker_max_yields:
                return
                
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_data = torch.load(f, map_location="cpu", weights_only=False)
                
                if self.agnn:
                    chunk_data = process_agnn_chunk(chunk_data, k=self.pe_channels)
                else:
                    chunk_data = process_dgnn_chunk(chunk_data, k=self.pe_channels)
            except Exception as e:
                print(f"  [!] Chunk load failed ({chunk_path}): {e}")
                continue
                
            rng.shuffle(chunk_data)
            
            while chunk_data:
                if worker_max_yields is not None and yielded_count >= worker_max_yields:
                    del chunk_data
                    gc.collect()
                    return
                    
                sample = chunk_data.pop()
                yielded_count += 1
                
                if self.agnn:
                    g_a, g_b, target, dist = sample
                    yield g_a, g_b, target, dist
                else:
                    g_a, g_b, dist, seq = sample
                    yield g_a, g_b, torch.as_tensor(dist, dtype=torch.float32), torch.as_tensor(seq, dtype=torch.long)
                    
            del chunk_data
            gc.collect()
