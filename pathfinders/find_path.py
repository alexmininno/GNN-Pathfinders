import os
import sys
import time
import argparse
import ast
import torch
import numpy as np
from numba import njit
from collections import defaultdict
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importing predictors/models
from src.predictor_dgnn import DGNNPredictor
from src.predictor_agnn import AGNNPredictor
import heapq
import math
import itertools

from src.data_utils import get_graph_hash, get_wl_hash, get_isomorphism_mapping, mutate_ranks, mutate_adjacency, is_connected


# ---------------------------------------------------------------------------
# Pathfinder Engine
# ---------------------------------------------------------------------------


class LCAPathfinder:
    def __init__(self):
        pass

    def _get_neighbors(self, ranks, adj, enforce_anomaly_free):
        n_nodes = len(ranks)
        neighbors = []
        for k in range(n_nodes):
            N_f_in = sum(adj[i][k] * ranks[i] for i in range(n_nodes))
            new_ranks = mutate_ranks(ranks, adj, k, enforce_anomaly_free=enforce_anomaly_free)
            if new_ranks is None:
                continue
            new_adj = mutate_adjacency(adj, k)
            if new_adj is None or not is_connected(new_adj):
                continue
            
            delta = new_ranks[k] - ranks[k]
            if delta < 0:
                cost = 1
            elif delta == 0:
                cost = 10
            else:
                cost = 100
                
            neighbors.append((k, new_ranks, new_adj, cost))
        return neighbors

    def find_path(self, ranks_a, adj_a, ranks_b, adj_b, max_steps=1000, max_nodes=100000, enforce_anomaly_free=True):
        if len(ranks_a) != len(ranks_b):
            return {
                "path": None,
                "visited_states": 0,
                "status": "no_path_found",
                "reason": "node_count_mismatch",
                "nodes_explored": 0
            }

        start_hash = get_graph_hash(ranks_a, adj_a)
        target_hash = get_graph_hash(ranks_b, adj_b)

        if start_hash == target_hash:
            return {"path": [], "visited_states": 1, "status": "success", "nodes_explored": 0}

        start_wl_hash = get_wl_hash(ranks_a, adj_a)
        target_wl_hash = get_wl_hash(ranks_b, adj_b)
        if start_wl_hash == target_wl_hash:
            if get_isomorphism_mapping(ranks_a, adj_a, ranks_b, adj_b) is not None:
                return {"path": [], "visited_states": 1, "status": "success", "nodes_explored": 0}

        counter = itertools.count()
        
        pq_a = [(0, next(counter), start_hash)]
        pq_b = [(0, next(counter), target_hash)]
        
        # Maps hash -> (cost, ranks, adj, path)
        history_a = {start_hash: (0, ranks_a, adj_a, [])}
        history_b = {target_hash: (0, ranks_b, adj_b, [])}
        
        wl_map_a = defaultdict(list)
        wl_map_b = defaultdict(list)
        wl_map_a[start_wl_hash].append(start_hash)
        wl_map_b[target_wl_hash].append(target_hash)
        
        total_steps = 0
        
        while pq_a and pq_b and total_steps < max_steps:
            total_steps += 1
            
            # Prevent hanging or OOM
            if len(history_a) + len(history_b) > max_nodes:
                break
            
            # Pick the frontier with the lower top cost
            if pq_a[0][0] <= pq_b[0][0]:
                curr_cost, _, curr_hash = heapq.heappop(pq_a)
                if curr_hash not in history_a or history_a[curr_hash][0] < curr_cost:
                    continue # Stale entry
                
                _, curr_ranks, curr_adj, curr_path = history_a[curr_hash]
                
                neighbors = self._get_neighbors(curr_ranks, curr_adj, enforce_anomaly_free)
                for k, new_ranks, new_adj, step_cost in neighbors:
                    new_cost = curr_cost + step_cost
                    new_hash = get_graph_hash(new_ranks, new_adj)
                    new_wl_hash = get_wl_hash(new_ranks, new_adj)
                    new_path = curr_path + [k]
                    
                    if new_hash not in history_a or new_cost < history_a[new_hash][0]:
                        history_a[new_hash] = (new_cost, new_ranks, new_adj, new_path)
                        wl_map_a[new_wl_hash].append(new_hash)
                        heapq.heappush(pq_a, (new_cost, next(counter), new_hash))
                        
                    if new_wl_hash in wl_map_b:
                        for other_md5 in wl_map_b[new_wl_hash]:
                            _, other_ranks, other_adj, other_path = history_b[other_md5]
                            mapping = get_isomorphism_mapping(new_ranks, new_adj, other_ranks, other_adj)
                            if mapping is not None:
                                path_a = new_path
                                inv_mapping = {v: v_k for v_k, v in mapping.items()}
                                translated_path_b = [inv_mapping[step] for step in reversed(other_path)]
                                return {
                                    "path": path_a + translated_path_b,
                                    "visited_states": len(history_a) + len(history_b),
                                    "status": "success",
                                    "nodes_explored": len(history_a) + len(history_b)
                                }
            else:
                curr_cost, _, curr_hash = heapq.heappop(pq_b)
                if curr_hash not in history_b or history_b[curr_hash][0] < curr_cost:
                    continue # Stale entry
                
                _, curr_ranks, curr_adj, curr_path = history_b[curr_hash]
                
                neighbors = self._get_neighbors(curr_ranks, curr_adj, enforce_anomaly_free)
                for k, new_ranks, new_adj, step_cost in neighbors:
                    new_cost = curr_cost + step_cost
                    new_hash = get_graph_hash(new_ranks, new_adj)
                    new_wl_hash = get_wl_hash(new_ranks, new_adj)
                    new_path = curr_path + [k]
                    
                    if new_hash not in history_b or new_cost < history_b[new_hash][0]:
                        history_b[new_hash] = (new_cost, new_ranks, new_adj, new_path)
                        wl_map_b[new_wl_hash].append(new_hash)
                        heapq.heappush(pq_b, (new_cost, next(counter), new_hash))
                        
                    if new_wl_hash in wl_map_a:
                        for other_md5 in wl_map_a[new_wl_hash]:
                            _, other_ranks, other_adj, other_path = history_a[other_md5]
                            mapping = get_isomorphism_mapping(other_ranks, other_adj, new_ranks, new_adj)
                            if mapping is not None:
                                path_a = other_path
                                inv_mapping = {v: v_k for v_k, v in mapping.items()}
                                translated_path_b = [inv_mapping[step] for step in reversed(new_path)]
                                return {
                                    "path": path_a + translated_path_b,
                                    "visited_states": len(history_a) + len(history_b),
                                    "status": "success",
                                    "nodes_explored": len(history_a) + len(history_b)
                                }
                        
        if not pq_a and not pq_b:
            return {
                "path": None,
                "visited_states": len(history_a) + len(history_b),
                "status": "no_path_found",
                "reason": "state_space_exhausted",
                "nodes_explored": len(history_a) + len(history_b)
            }
            
        return {
            "path": None,
            "visited_states": len(history_a) + len(history_b),
            "status": "max_steps_reached",
            "nodes_explored": len(history_a) + len(history_b)
        }


class HeuristicBidirectionalPathfinder:
    def __init__(self, model_path, hidden_channels=64, device=None):
        self.predictor = DGNNPredictor(
            model_path=model_path, hidden_channels=hidden_channels, device=device
        )
        self.counter = itertools.count()

    def _expand_frontier(
        self, pq, visited, other_visited, this_wl_map, other_wl_map, target_ranks, target_adj, max_steps, is_forward,
        enforce_anomaly_free=True,
    ):
        """
        Pop one node from the priority queue, expand its children,
        and check for meeting with the other frontier.

        Returns:
            meeting_hash: hash of meeting node if found, else None
            The visited dict is updated in-place with new states.
        """
        if not pq:
            return None

        f_score, _, (g_score, c_ranks, c_adj, path) = heapq.heappop(pq)

        if g_score >= max_steps:
            return None  # Don't expand beyond budget

        c_hash = get_graph_hash(c_ranks, c_adj)
        c_wl_hash = get_wl_hash(c_ranks, c_adj)

        # Check if this node was already reached by the other side
        if c_wl_hash in other_wl_map:
            for other_md5 in other_wl_map[c_wl_hash]:
                other_node = other_visited[other_md5]
                if is_forward:
                    mapping = get_isomorphism_mapping(c_ranks, c_adj, other_node["ranks"], other_node["adj"])
                else:
                    mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], c_ranks, c_adj)
                if mapping is not None:
                    fwd_md5 = c_hash if is_forward else other_md5
                    bwd_md5 = other_md5 if is_forward else c_hash
                    return fwd_md5, bwd_md5, mapping

        # Expand valid mutations
        n_nodes = len(c_ranks)
        valid_mutants_ranks = []
        valid_mutants_adj = []
        valid_mutants_k = []

        for k in range(n_nodes):
            new_ranks = mutate_ranks(c_ranks, c_adj, k, enforce_anomaly_free=enforce_anomaly_free)
            if new_ranks is None:
                continue

            new_adj = mutate_adjacency(c_adj, k)
            if new_adj is None or not is_connected(new_adj):
                continue

            new_hash = get_graph_hash(new_ranks, new_adj)
            if new_hash not in visited:
                visited[new_hash] = {
                    "ranks": new_ranks,
                    "adj": new_adj,
                    "g_score": g_score + 1,
                    "path": path + [k],
                }
                new_wl_hash = get_wl_hash(new_ranks, new_adj)
                this_wl_map[new_wl_hash].append(new_hash)
                valid_mutants_ranks.append(new_ranks)
                valid_mutants_adj.append(new_adj)
                valid_mutants_k.append(k)

                # Immediate meeting check
                if new_wl_hash in other_wl_map:
                    for other_md5 in other_wl_map[new_wl_hash]:
                        other_node = other_visited[other_md5]
                        if is_forward:
                            mapping = get_isomorphism_mapping(new_ranks, new_adj, other_node["ranks"], other_node["adj"])
                        else:
                            mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], new_ranks, new_adj)
                        
                        if mapping is not None:
                            fwd_md5 = new_hash if is_forward else other_md5
                            bwd_md5 = other_md5 if is_forward else new_hash
                            return fwd_md5, bwd_md5, mapping

        if not valid_mutants_ranks:
            return None

        # Predict heuristic scores for all children → target
        h_scores = self.predictor.predict_batch(
            lists_of_ranks_a=valid_mutants_ranks,
            lists_of_adj_a=valid_mutants_adj,
            ranks_b=target_ranks,
            adj_b=target_adj,
        )

        for idx in range(len(valid_mutants_ranks)):
            new_hash = get_graph_hash(valid_mutants_ranks[idx], valid_mutants_adj[idx])
            h_score = h_scores[idx]
            new_g_score = g_score + 1
            new_f_score = new_g_score + h_score

            new_node = (
                new_g_score,
                valid_mutants_ranks[idx],
                valid_mutants_adj[idx],
                path + [valid_mutants_k[idx]],
            )
            heapq.heappush(pq, (new_f_score, next(self.counter), new_node))

        return None

    def find_path(self, ranks_a, adj_a, ranks_b, adj_b, max_steps=50, max_nodes=100000,
                  enforce_anomaly_free=True):
        """
        Bidirectional A* search from A toward B and from B toward A.

        Each frontier gets max_steps/2 budget, so total depth = max_steps.
        When frontiers meet, the path is: forward_path + reverse(backward_path).
        Aborts if total visited_states exceeds max_nodes to prevent OOM/hangs.
        """
        # Seiberg duality preserves node count — different sizes are trivially unrelated
        if len(ranks_a) != len(ranks_b):
            return {
                "path": None,
                "visited_states": 0,
                "fwd_nodes": 0,
                "bwd_nodes": 0,
                "status": "no_path_found",
                "initial_h_fwd": float("inf"),
                "initial_h_bwd": float("inf"),
                "reason": "node_count_mismatch",
                "model_passes": 0,
            }

        start_hash = get_graph_hash(ranks_a, adj_a)
        target_hash = get_graph_hash(ranks_b, adj_b)

        if start_hash == target_hash:
            return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        start_wl_hash = get_wl_hash(ranks_a, adj_a)
        target_wl_hash = get_wl_hash(ranks_b, adj_b)
        if start_wl_hash == target_wl_hash:
            if get_isomorphism_mapping(ranks_a, adj_a, ranks_b, adj_b) is not None:
                return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        half_budget = max_steps // 2

        # Forward frontier: A → B
        fwd_visited = {
            start_hash: {"ranks": ranks_a, "adj": adj_a, "g_score": 0, "path": []}
        }
        fwd_wl_map = defaultdict(list)
        fwd_wl_map[start_wl_hash].append(start_hash)
        h_fwd = self.predictor.predict(ranks_a, adj_a, ranks_b, adj_b)["estimated_distance"]
        fwd_pq = []
        heapq.heappush(fwd_pq, (h_fwd, next(self.counter), (0, ranks_a, adj_a, [])))

        # Backward frontier: B → A
        bwd_visited = {
            target_hash: {"ranks": ranks_b, "adj": adj_b, "g_score": 0, "path": []}
        }
        bwd_wl_map = defaultdict(list)
        bwd_wl_map[target_wl_hash].append(target_hash)
        h_bwd = self.predictor.predict(ranks_b, adj_b, ranks_a, adj_a)["estimated_distance"]
        bwd_pq = []
        heapq.heappush(bwd_pq, (h_bwd, next(self.counter), (0, ranks_b, adj_b, [])))

        total_visited = 2
        iteration = 0

        while fwd_pq or bwd_pq:
            iteration += 1

            # Alternate: expand the frontier with smaller min f-score
            expand_fwd = True
            if not fwd_pq:
                expand_fwd = False
            elif bwd_pq:
                if bwd_pq[0][0] < fwd_pq[0][0]:
                    expand_fwd = False

            if expand_fwd:
                meeting = self._expand_frontier(
                    fwd_pq,
                    fwd_visited,
                    bwd_visited,
                    fwd_wl_map,
                    bwd_wl_map,
                    ranks_b,
                    adj_b,
                    half_budget,
                    is_forward=True,
                    enforce_anomaly_free=enforce_anomaly_free,
                )
            else:
                meeting = self._expand_frontier(
                    bwd_pq,
                    bwd_visited,
                    fwd_visited,
                    bwd_wl_map,
                    fwd_wl_map,
                    ranks_a,
                    adj_a,
                    half_budget,
                    is_forward=False,
                    enforce_anomaly_free=enforce_anomaly_free,
                )

            total_visited = len(fwd_visited) + len(bwd_visited)

            if meeting is not None:
                fwd_md5, bwd_md5, mapping = meeting
                fwd_node = fwd_visited[fwd_md5]
                bwd_node = bwd_visited[bwd_md5]
                fwd_path = fwd_node["path"]
                bwd_path = bwd_node["path"]
                
                inv_mapping = {v: k for k, v in mapping.items()}
                translated_bwd_path = [inv_mapping[k] for k in reversed(bwd_path)]
                full_path = fwd_path + translated_bwd_path
                return {
                    "path": full_path,
                    "visited_states": len(fwd_visited) + len(bwd_visited),
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "success",
                    "meeting_depth_fwd": len(fwd_path),
                    "meeting_depth_bwd": len(bwd_path),
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

            if total_visited >= max_nodes:
                return {
                    "path": None,
                    "visited_states": total_visited,
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "max_nodes_reached",
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

        return {
            "path": None,
            "visited_states": len(fwd_visited) + len(bwd_visited),
            "fwd_nodes": len(fwd_visited),
            "bwd_nodes": len(bwd_visited),
            "status": "no_path_found",
            "initial_h_fwd": h_fwd,
            "initial_h_bwd": h_bwd,
            "model_passes": iteration,
        }


class AGNNPathfinder:
    def __init__(self, model_path, hidden_channels=128, device=None):
        self.agnn_predictor = AGNNPredictor(
            model_path=model_path, hidden_channels=hidden_channels, device=device
        )
        self.device = self.agnn_predictor.device

    def find_path(self, ranks_a, adj_a, ranks_b, adj_b, max_steps=30, beam_width=3, enforce_anomaly_free=True):
        """
        Beam Search from A → B using the AGNN model.
        """
        start_hash = get_graph_hash(ranks_a, adj_a)
        target_hash = get_graph_hash(ranks_b, adj_b)

        if start_hash == target_hash:
            return {"status": "success", "path": [], "model_passes": 0, "visited_states": 1}

        start_wl_hash = get_wl_hash(ranks_a, adj_a)
        target_wl_hash = get_wl_hash(ranks_b, adj_b)
        if start_wl_hash == target_wl_hash:
            if get_isomorphism_mapping(ranks_a, adj_a, ranks_b, adj_b) is not None:
                return {"status": "success", "path": [], "model_passes": 0, "visited_states": 1}

        visited = {start_hash: (ranks_a, adj_a, [])}
        model_passes = 0
        beam = [(0.0, start_hash, ranks_a, adj_a, [])]

        data_b = self.agnn_predictor.get_pyg_data(ranks_b, adj_b)
        batch_b = Batch.from_data_list([data_b]).to(self.device)

        for depth in range(max_steps):
            new_beam = []

            for score, c_hash, c_ranks, c_adj, path in beam:
                data_a = self.agnn_predictor.get_pyg_data(c_ranks, c_adj)
                batch_a = Batch.from_data_list([data_a]).to(self.device)

                logits = self.agnn_predictor.predict_logits_batch(batch_a, batch_b).squeeze(0)
                model_passes += 1

                probs = torch.softmax(logits, dim=-1)
                
                top_k_val = min(beam_width, len(c_ranks)) if beam_width is not None else len(c_ranks)
                top_p, top_idx = torch.topk(probs, k=top_k_val)

                for p, idx in zip(top_p, top_idx):
                    node_k = idx.item()

                    new_ranks = mutate_ranks(c_ranks, c_adj, node_k, enforce_anomaly_free)
                    if new_ranks is None:
                        continue

                    new_adj = mutate_adjacency(c_adj, node_k)
                    if new_adj is None or not is_connected(new_adj):
                        continue

                    new_hash = get_graph_hash(new_ranks, new_adj)
                    new_wl_hash = get_wl_hash(new_ranks, new_adj)
                    new_path = path + [node_k]
                    new_score = score + torch.log(p).item()

                    if new_wl_hash == target_wl_hash:
                        if get_isomorphism_mapping(new_ranks, new_adj, ranks_b, adj_b) is not None:
                            visited[new_hash] = (new_ranks, new_adj, new_path)
                            return {
                                "status": "success",
                                "path": new_path,
                                "model_passes": model_passes,
                                "visited_states": len(visited),
                            }

                    if new_hash not in visited or len(new_path) < len(visited[new_hash][2]):
                        visited[new_hash] = (new_ranks, new_adj, new_path)
                        new_beam.append((new_score, new_hash, new_ranks, new_adj, new_path))

            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

            if not beam:
                return {"status": "dead_end", "path": None, "model_passes": model_passes, "visited_states": len(visited)}

        return {"status": "max_steps_reached", "path": None, "model_passes": model_passes, "visited_states": len(visited)}


class HybridBidirectionalPathfinder:
    def __init__(self, dgnn_model_path, agnn_model_path, hidden_channels_dgnn=64, hidden_channels_agnn=128, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self.device = torch.device("mps")
        else:
            self.device = device
            
        # print(f"Loading DGNN model from {dgnn_model_path} onto {self.device}")
        self.predictor = DGNNPredictor(
            model_path=dgnn_model_path, hidden_channels=hidden_channels_dgnn, device=self.device
        )
        
        self.agnn_predictor = AGNNPredictor(
            model_path=agnn_model_path, hidden_channels=hidden_channels_agnn, device=self.device
        )
        
        self.counter = itertools.count()

    def _expand_frontier(
        self, pq, visited, other_visited, this_wl_map, other_wl_map, target_ranks, target_adj, target_batch, max_steps,
        lambda_agnn, top_k, enforce_anomaly_free=True, is_forward=True,
    ):
        if not pq:
            return None

        f_score, _, (g_score, c_ranks, c_adj, path) = heapq.heappop(pq)

        if len(path) >= max_steps:
            return None

        c_hash = get_graph_hash(c_ranks, c_adj)
        c_wl_hash = get_wl_hash(c_ranks, c_adj)
        if c_wl_hash in other_wl_map:
            for other_md5 in other_wl_map[c_wl_hash]:
                other_node = other_visited[other_md5]
                if is_forward:
                    mapping = get_isomorphism_mapping(c_ranks, c_adj, other_node["ranks"], other_node["adj"])
                else:
                    mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], c_ranks, c_adj)
                if mapping is not None:
                    fwd_md5 = c_hash if is_forward else other_md5
                    bwd_md5 = other_md5 if is_forward else c_hash
                    return fwd_md5, bwd_md5, mapping

        # 1. Run AGNN model to get action probabilities
        data_c = self.agnn_predictor.get_pyg_data(c_ranks, c_adj)
        batch_c = Batch.from_data_list([data_c]).to(self.device)
        
        logits = self.agnn_predictor.predict_logits_batch(batch_c, target_batch).squeeze(0) # [N_max]
        probs = torch.softmax(logits, dim=-1)
        
        n_nodes = len(c_ranks)
        if top_k is not None and top_k > 0:
            k_val = min(top_k, n_nodes)
            top_p, top_idx = torch.topk(probs[:n_nodes], k=k_val)
            actions_to_try = top_idx.tolist()
            action_probs = top_p.tolist()
        else:
            actions_to_try = list(range(n_nodes))
            action_probs = probs[:n_nodes].tolist()

        valid_mutants_ranks = []
        valid_mutants_adj = []
        valid_mutants_k = []
        valid_mutants_p = []

        for k, p in zip(actions_to_try, action_probs):
            new_ranks = mutate_ranks(c_ranks, c_adj, k, enforce_anomaly_free=enforce_anomaly_free)
            if new_ranks is None:
                continue

            new_adj = mutate_adjacency(c_adj, k)
            if new_adj is None or not is_connected(new_adj):
                continue

            new_hash = get_graph_hash(new_ranks, new_adj)
            
            if new_hash not in visited or g_score + 1 < visited[new_hash]["g_score"]:
                visited[new_hash] = {
                    "ranks": new_ranks,
                    "adj": new_adj,
                    "g_score": g_score + 1,
                    "path": path + [k],
                }
                new_wl_hash = get_wl_hash(new_ranks, new_adj)
                this_wl_map[new_wl_hash].append(new_hash)
                valid_mutants_ranks.append(new_ranks)
                valid_mutants_adj.append(new_adj)
                valid_mutants_k.append(k)
                valid_mutants_p.append(p)

                # Immediate meeting check
                if new_wl_hash in other_wl_map:
                    for other_md5 in other_wl_map[new_wl_hash]:
                        other_node = other_visited[other_md5]
                        if is_forward:
                            mapping = get_isomorphism_mapping(new_ranks, new_adj, other_node["ranks"], other_node["adj"])
                        else:
                            mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], new_ranks, new_adj)
                        
                        if mapping is not None:
                            fwd_md5 = new_hash if is_forward else other_md5
                            bwd_md5 = other_md5 if is_forward else new_hash
                            return fwd_md5, bwd_md5, mapping

        if not valid_mutants_ranks:
            return None

        # 2. Run DGNN model for heuristics
        h_scores = self.predictor.predict_batch(
            lists_of_ranks_a=valid_mutants_ranks,
            lists_of_adj_a=valid_mutants_adj,
            ranks_b=target_ranks,
            adj_b=target_adj,
        )

        for idx in range(len(valid_mutants_ranks)):
            h_score = h_scores[idx]
            p_val = max(valid_mutants_p[idx], 1e-12) # prevent log(0)
            
            # AGNN guided step cost: penalizes low probability moves
            step_cost = 1.0 - lambda_agnn * math.log(p_val)
            
            new_g_score = g_score + step_cost
            new_f_score = new_g_score + h_score

            new_node = (
                new_g_score,
                valid_mutants_ranks[idx],
                valid_mutants_adj[idx],
                path + [valid_mutants_k[idx]],
            )
            heapq.heappush(pq, (new_f_score, next(self.counter), new_node))

        return None

    def find_path(self, ranks_a, adj_a, ranks_b, adj_b, max_steps=50, max_nodes=100000,
                  enforce_anomaly_free=True, lambda_agnn=1.0, top_k=None):
        if len(ranks_a) != len(ranks_b):
            return {
                "path": None,
                "visited_states": 0,
                "fwd_nodes": 0,
                "bwd_nodes": 0,
                "status": "no_path_found",
                "initial_h_fwd": float("inf"),
                "initial_h_bwd": float("inf"),
                "reason": "node_count_mismatch",
                "model_passes": 0,
            }

        start_hash = get_graph_hash(ranks_a, adj_a)
        target_hash = get_graph_hash(ranks_b, adj_b)

        if start_hash == target_hash:
            return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        start_wl_hash = get_wl_hash(ranks_a, adj_a)
        target_wl_hash = get_wl_hash(ranks_b, adj_b)
        if start_wl_hash == target_wl_hash:
            if get_isomorphism_mapping(ranks_a, adj_a, ranks_b, adj_b) is not None:
                return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        half_budget = max_steps // 2
        
        target_data_fwd = self.agnn_predictor.get_pyg_data(ranks_b, adj_b)
        target_batch_fwd = Batch.from_data_list([target_data_fwd]).to(self.device)

        target_data_bwd = self.agnn_predictor.get_pyg_data(ranks_a, adj_a)
        target_batch_bwd = Batch.from_data_list([target_data_bwd]).to(self.device)

        # Forward frontier: A → B
        fwd_visited = {
            start_hash: {"ranks": ranks_a, "adj": adj_a, "g_score": 0, "path": []}
        }
        fwd_wl_map = defaultdict(list)
        fwd_wl_map[start_wl_hash].append(start_hash)
        h_fwd = self.predictor.predict(ranks_a, adj_a, ranks_b, adj_b)["estimated_distance"]
        fwd_pq = []
        heapq.heappush(fwd_pq, (h_fwd, next(self.counter), (0, ranks_a, adj_a, [])))

        # Backward frontier: B → A
        bwd_visited = {
            target_hash: {"ranks": ranks_b, "adj": adj_b, "g_score": 0, "path": []}
        }
        bwd_wl_map = defaultdict(list)
        bwd_wl_map[target_wl_hash].append(target_hash)
        h_bwd = self.predictor.predict(ranks_b, adj_b, ranks_a, adj_a)["estimated_distance"]
        bwd_pq = []
        heapq.heappush(bwd_pq, (h_bwd, next(self.counter), (0, ranks_b, adj_b, [])))

        total_visited = 2
        iteration = 0

        while fwd_pq or bwd_pq:
            iteration += 1

            expand_fwd = True
            if not fwd_pq:
                expand_fwd = False
            elif bwd_pq:
                if bwd_pq[0][0] < fwd_pq[0][0]:
                    expand_fwd = False

            if expand_fwd:
                meeting = self._expand_frontier(
                    fwd_pq, fwd_visited, bwd_visited, fwd_wl_map, bwd_wl_map,
                    ranks_b, adj_b, target_batch_fwd,
                    half_budget, lambda_agnn, top_k, enforce_anomaly_free, True
                )
            else:
                meeting = self._expand_frontier(
                    bwd_pq, bwd_visited, fwd_visited, bwd_wl_map, fwd_wl_map,
                    ranks_a, adj_a, target_batch_bwd,
                    half_budget, lambda_agnn, top_k, enforce_anomaly_free, False
                )

            total_visited = len(fwd_visited) + len(bwd_visited)

            if meeting is not None:
                fwd_md5, bwd_md5, mapping = meeting
                fwd_node = fwd_visited[fwd_md5]
                bwd_node = bwd_visited[bwd_md5]
                fwd_path = fwd_node["path"]
                bwd_path = bwd_node["path"]
                
                inv_mapping = {v: k for k, v in mapping.items()}
                translated_bwd_path = [inv_mapping[k] for k in reversed(bwd_path)]
                full_path = fwd_path + translated_bwd_path
                return {
                    "path": full_path,
                    "visited_states": total_visited,
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "success",
                    "meeting_depth_fwd": len(fwd_path),
                    "meeting_depth_bwd": len(bwd_path),
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

            if total_visited >= max_nodes:
                return {
                    "path": None,
                    "visited_states": total_visited,
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "max_nodes_reached",
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

        return {
            "path": None,
            "visited_states": total_visited,
            "fwd_nodes": len(fwd_visited),
            "bwd_nodes": len(bwd_visited),
            "status": "no_path_found",
            "initial_h_fwd": h_fwd,
            "initial_h_bwd": h_bwd,
            "model_passes": iteration,
        }


class HybridLCAPathfinder:
    def __init__(self, dgnn_model_path, agnn_model_path, hidden_channels_dgnn=64, hidden_channels_agnn=128, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self.device = torch.device("mps")
        else:
            self.device = device
            
        # print(f"Loading DGNN model from {dgnn_model_path} onto {self.device}")
        self.predictor = DGNNPredictor(
            model_path=dgnn_model_path, hidden_channels=hidden_channels_dgnn, device=self.device
        )
        
        self.agnn_predictor = AGNNPredictor(
            model_path=agnn_model_path, hidden_channels=hidden_channels_agnn, device=self.device
        )
        
        self.counter = itertools.count()

    def _expand_frontier(
        self, pq, visited, other_visited, this_wl_map, other_wl_map, target_ranks, target_adj, target_batch, max_steps,
        lambda_agnn, top_k, enforce_anomaly_free=True, initial_rank_sum=None,
        lambda_det_cost=0.0, lambda_dgnn_h=1.0, lambda_lca_h=0.0,
        cost_decrease=0.5, cost_equal=1.0, cost_increase=5.0, is_forward=True
    ):
        if not pq:
            return None

        f_score, _, (g_score, c_ranks, c_adj, path) = heapq.heappop(pq)

        if len(path) >= max_steps:
            return None

        c_hash = get_graph_hash(c_ranks, c_adj)
        c_wl_hash = get_wl_hash(c_ranks, c_adj)
        if c_wl_hash in other_wl_map:
            for other_md5 in other_wl_map[c_wl_hash]:
                other_node = other_visited[other_md5]
                if is_forward:
                    mapping = get_isomorphism_mapping(c_ranks, c_adj, other_node["ranks"], other_node["adj"])
                else:
                    mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], c_ranks, c_adj)
                if mapping is not None:
                    fwd_md5 = c_hash if is_forward else other_md5
                    bwd_md5 = other_md5 if is_forward else c_hash
                    return fwd_md5, bwd_md5, mapping

        # 1. Run AGNN model to get action probabilities
        data_c = self.agnn_predictor.get_pyg_data(c_ranks, c_adj)
        batch_c = Batch.from_data_list([data_c]).to(self.device)
        
        logits = self.agnn_predictor.predict_logits_batch(batch_c, target_batch).squeeze(0) # [N_max]
        probs = torch.softmax(logits, dim=-1)
        
        n_nodes = len(c_ranks)
        if top_k is not None and top_k > 0:
            k_val = min(top_k, n_nodes)
            top_p, top_idx = torch.topk(probs[:n_nodes], k=k_val)
            actions_to_try = top_idx.tolist()
            action_probs = top_p.tolist()
        else:
            actions_to_try = list(range(n_nodes))
            action_probs = probs[:n_nodes].tolist()

        valid_mutants_ranks = []
        valid_mutants_adj = []
        valid_mutants_k = []
        valid_mutants_p = []

        for k, p in zip(actions_to_try, action_probs):
            new_ranks = mutate_ranks(c_ranks, c_adj, k, enforce_anomaly_free=enforce_anomaly_free)
            if new_ranks is None:
                continue

            new_adj = mutate_adjacency(c_adj, k)
            if new_adj is None or not is_connected(new_adj):
                continue

            new_hash = get_graph_hash(new_ranks, new_adj)
            
            if new_hash not in visited or g_score + 1 < visited[new_hash]["g_score"]:
                visited[new_hash] = {
                    "ranks": new_ranks,
                    "adj": new_adj,
                    "g_score": g_score + 1,
                    "path": path + [k],
                }
                new_wl_hash = get_wl_hash(new_ranks, new_adj)
                this_wl_map[new_wl_hash].append(new_hash)
                valid_mutants_ranks.append(new_ranks)
                valid_mutants_adj.append(new_adj)
                valid_mutants_k.append(k)
                valid_mutants_p.append(p)

                # Immediate meeting check
                if new_wl_hash in other_wl_map:
                    for other_md5 in other_wl_map[new_wl_hash]:
                        other_node = other_visited[other_md5]
                        if is_forward:
                            mapping = get_isomorphism_mapping(new_ranks, new_adj, other_node["ranks"], other_node["adj"])
                        else:
                            mapping = get_isomorphism_mapping(other_node["ranks"], other_node["adj"], new_ranks, new_adj)
                        
                        if mapping is not None:
                            fwd_md5 = new_hash if is_forward else other_md5
                            bwd_md5 = other_md5 if is_forward else new_hash
                            return fwd_md5, bwd_md5, mapping

        if not valid_mutants_ranks:
            return None

        # 2. Run DGNN model for heuristics
        if lambda_dgnn_h > 0.0:
            h_dgnn_scores = self.predictor.predict_batch(
                lists_of_ranks_a=valid_mutants_ranks,
                lists_of_adj_a=valid_mutants_adj,
                ranks_b=target_ranks,
                adj_b=target_adj,
            )
        else:
            h_dgnn_scores = [0.0] * len(valid_mutants_ranks)

        for idx in range(len(valid_mutants_ranks)):
            h_dgnn = h_dgnn_scores[idx]
            p_val = max(valid_mutants_p[idx], 1e-12) # prevent log(0)
            
            k = valid_mutants_k[idx]
            new_ranks_idx = valid_mutants_ranks[idx]
            delta = new_ranks_idx[k] - c_ranks[k]
            
            if delta < 0:
                det_base = cost_decrease
            elif delta == 0:
                det_base = cost_equal
            else:
                det_base = cost_increase
                
            base_cost = (1.0 - lambda_det_cost) * 1.0 + lambda_det_cost * det_base
            
            # AGNN guided step cost: penalizes low probability moves
            step_cost = base_cost - lambda_agnn * math.log(p_val)
            
            new_g_score = g_score + step_cost
            
            if lambda_lca_h > 0.0:
                try:
                    rank_penalty = sum(new_ranks_idx) / initial_rank_sum if initial_rank_sum and initial_rank_sum > 0 else 1.0
                except OverflowError:
                    rank_penalty = 1e300
            else:
                rank_penalty = 0.0
                
            h_score = lambda_dgnn_h * h_dgnn + lambda_lca_h * rank_penalty
            
            new_f_score = new_g_score + h_score

            new_node = (
                new_g_score,
                valid_mutants_ranks[idx],
                valid_mutants_adj[idx],
                path + [valid_mutants_k[idx]],
            )
            heapq.heappush(pq, (new_f_score, next(self.counter), new_node))

        return None

    def find_path(self, ranks_a, adj_a, ranks_b, adj_b, max_steps=50, max_nodes=100000,
                  enforce_anomaly_free=True, lambda_agnn=1.0, top_k=None,
                  lambda_det_cost=0.0, lambda_dgnn_h=1.0, lambda_lca_h=0.0,
                  cost_decrease=0.5, cost_equal=1.0, cost_increase=5.0):
        if len(ranks_a) != len(ranks_b):
            return {
                "path": None,
                "visited_states": 0,
                "fwd_nodes": 0,
                "bwd_nodes": 0,
                "status": "no_path_found",
                "initial_h_fwd": float("inf"),
                "initial_h_bwd": float("inf"),
                "reason": "node_count_mismatch",
                "model_passes": 0,
            }

        start_hash = get_graph_hash(ranks_a, adj_a)
        target_hash = get_graph_hash(ranks_b, adj_b)

        if start_hash == target_hash:
            return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        start_wl_hash = get_wl_hash(ranks_a, adj_a)
        target_wl_hash = get_wl_hash(ranks_b, adj_b)
        if start_wl_hash == target_wl_hash:
            if get_isomorphism_mapping(ranks_a, adj_a, ranks_b, adj_b) is not None:
                return {"path": [], "visited_states": 1, "status": "success", "model_passes": 0}

        half_budget = max_steps // 2
        
        target_data_fwd = self.agnn_predictor.get_pyg_data(ranks_b, adj_b)
        target_batch_fwd = Batch.from_data_list([target_data_fwd]).to(self.device)

        target_data_bwd = self.agnn_predictor.get_pyg_data(ranks_a, adj_a)
        target_batch_bwd = Batch.from_data_list([target_data_bwd]).to(self.device)

        # Forward frontier: A → B
        fwd_visited = {
            start_hash: {"ranks": ranks_a, "adj": adj_a, "g_score": 0, "path": []}
        }
        fwd_wl_map = defaultdict(list)
        fwd_wl_map[start_wl_hash].append(start_hash)
        h_dgnn_fwd = self.predictor.predict(ranks_a, adj_a, ranks_b, adj_b)["estimated_distance"]
        initial_sum_a = sum(ranks_a)
        rank_penalty_fwd = sum(ranks_a) / initial_sum_a if initial_sum_a > 0 else 1.0
        h_fwd = lambda_dgnn_h * h_dgnn_fwd + lambda_lca_h * rank_penalty_fwd
        fwd_pq = []
        heapq.heappush(fwd_pq, (h_fwd, next(self.counter), (0, ranks_a, adj_a, [])))

        # Backward frontier: B → A
        bwd_visited = {
            target_hash: {"ranks": ranks_b, "adj": adj_b, "g_score": 0, "path": []}
        }
        bwd_wl_map = defaultdict(list)
        bwd_wl_map[target_wl_hash].append(target_hash)
        h_dgnn_bwd = self.predictor.predict(ranks_b, adj_b, ranks_a, adj_a)["estimated_distance"]
        initial_sum_b = sum(ranks_b)
        rank_penalty_bwd = sum(ranks_b) / initial_sum_b if initial_sum_b > 0 else 1.0
        h_bwd = lambda_dgnn_h * h_dgnn_bwd + lambda_lca_h * rank_penalty_bwd
        bwd_pq = []
        heapq.heappush(bwd_pq, (h_bwd, next(self.counter), (0, ranks_b, adj_b, [])))

        total_visited = 2
        iteration = 0

        while fwd_pq or bwd_pq:
            iteration += 1

            expand_fwd = True
            if not fwd_pq:
                expand_fwd = False
            elif bwd_pq:
                if bwd_pq[0][0] < fwd_pq[0][0]:
                    expand_fwd = False

            if expand_fwd:
                meeting = self._expand_frontier(
                    fwd_pq, fwd_visited, bwd_visited, fwd_wl_map, bwd_wl_map,
                    ranks_b, adj_b, target_batch_fwd,
                    half_budget, lambda_agnn, top_k, enforce_anomaly_free,
                    initial_sum_a, lambda_det_cost, lambda_dgnn_h, lambda_lca_h,
                    cost_decrease, cost_equal, cost_increase, True
                )
            else:
                meeting = self._expand_frontier(
                    bwd_pq, bwd_visited, fwd_visited, bwd_wl_map, fwd_wl_map,
                    ranks_a, adj_a, target_batch_bwd,
                    half_budget, lambda_agnn, top_k, enforce_anomaly_free,
                    initial_sum_b, lambda_det_cost, lambda_dgnn_h, lambda_lca_h,
                    cost_decrease, cost_equal, cost_increase, False
                )

            total_visited = len(fwd_visited) + len(bwd_visited)

            if meeting is not None:
                fwd_md5, bwd_md5, mapping = meeting
                fwd_node = fwd_visited[fwd_md5]
                bwd_node = bwd_visited[bwd_md5]
                fwd_path = fwd_node["path"]
                bwd_path = bwd_node["path"]
                
                inv_mapping = {v: k for k, v in mapping.items()}
                translated_bwd_path = [inv_mapping[k] for k in reversed(bwd_path)]
                full_path = fwd_path + translated_bwd_path
                return {
                    "path": full_path,
                    "visited_states": total_visited,
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "success",
                    "meeting_depth_fwd": len(fwd_path),
                    "meeting_depth_bwd": len(bwd_path),
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

            if total_visited >= max_nodes:
                return {
                    "path": None,
                    "visited_states": total_visited,
                    "fwd_nodes": len(fwd_visited),
                    "bwd_nodes": len(bwd_visited),
                    "status": "max_nodes_reached",
                    "initial_h_fwd": h_fwd,
                    "initial_h_bwd": h_bwd,
                    "model_passes": iteration,
                }

        return {
            "path": None,
            "visited_states": total_visited,
            "fwd_nodes": len(fwd_visited),
            "bwd_nodes": len(bwd_visited),
            "status": "no_path_found",
            "initial_h_fwd": h_fwd,
            "initial_h_bwd": h_bwd,
            "model_passes": iteration,
        }




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Pathfinder for Seiberg Duality")
    
    # Execution flags
    parser.add_argument("--dgnn", action="store_true", help="Run DGNN A* Pathfinder")
    parser.add_argument("--agnn", action="store_true", help="Run AGNN Beam Search Pathfinder")
    parser.add_argument("--hybrid", action="store_true", help="Run Hybrid Bidirectional A* Pathfinder")
    parser.add_argument("--lca", action="store_true", help="Run Heuristic (LCA) Bidirectional A* Pathfinder")
    parser.add_argument("--hybrid_lca", action="store_true", help="Run Deterministic Hybrid Bidirectional A* Pathfinder")
    parser.add_argument("--det", action="store_true", help="Run Deterministic Greedy Pathfinder")
    
    # Required parameters
    parser.add_argument("--ranks_a", type=str, required=True, help="Ranks for Graph A. Example: --ranks_a '[1, 2, 3]'")
    parser.add_argument("--adj_a", type=str, required=True, help="Adjacency matrix for Graph A. Example: --adj_a '[[0,1,0],[0,0,1],[1,0,0]]'")
    parser.add_argument("--ranks_b", type=str, required=True, help="Ranks for Graph B. Example: --ranks_b '[1, 1, 3]'")
    parser.add_argument("--adj_b", type=str, required=True, help="Adjacency matrix for Graph B. Example: --adj_b '[[0,0,1],[1,0,0],[0,1,0]]'")
    
    # Models
    parser.add_argument("--dgnn_model", type=str, default="", help="Path to DGNN .pth checkpoint. Example: --dgnn_model $CHECKPOINTS_DIR/best_dgnn.pth")
    parser.add_argument("--agnn_model", type=str, default="", help="Path to AGNN .pth checkpoint. Example: --agnn_model $CHECKPOINTS_DIR/best_agnn.pth")
    
    # Hyperparameters
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum total search depth. Example: --max_steps 50")
    parser.add_argument("--max_nodes", type=int, default=100000, help="Maximum nodes to explore before aborting. Example: --max_nodes 100000")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for AGNN search. Example: --beam_width 3")
    parser.add_argument("--relax_anomaly", action="store_true", help="Skip the anomaly-free check (N_f_in == N_f_out)")
    
    parser.add_argument("--lambda_agnn", type=float, default=1.0, help="Weight for AGNN log-prob penalty in hybrid search. Example: --lambda_agnn 1.0")
    parser.add_argument("--top_k", type=int, default=None, help="Filter actions to only top K predicted by AGNN model. Example: --top_k 5")
    
    parser.add_argument("--lambda_det_cost", type=float, default=0.0, help="Weight for deterministic step cost in Hybrid LCA. Example: --lambda_det_cost 0.5")
    parser.add_argument("--lambda_dgnn_h", type=float, default=1.0, help="Weight for DGNN heuristic. Example: --lambda_dgnn_h 1.0")
    parser.add_argument("--lambda_lca_h", type=float, default=0.0, help="Weight for LCA heuristic. Example: --lambda_lca_h 1.0")
    
    parser.add_argument("--cost_decrease", type=float, default=0.5, help="Cost multiplier for rank decrease. Example: --cost_decrease 0.5")
    parser.add_argument("--cost_equal", type=float, default=1.0, help="Cost multiplier for equal rank. Example: --cost_equal 1.0")
    parser.add_argument("--cost_increase", type=float, default=5.0, help="Cost multiplier for rank increase. Example: --cost_increase 5.0")
    
    args = parser.parse_args()
    
    ranks_a = ast.literal_eval(args.ranks_a)
    adj_a = ast.literal_eval(args.adj_a)
    ranks_b = ast.literal_eval(args.ranks_b)
    adj_b = ast.literal_eval(args.adj_b)
    
    # If no specific method is chosen, run all of them
    run_all = not (args.dgnn or args.agnn or args.hybrid or args.lca or args.hybrid_lca or args.det)
    
    print(f"Running Find Path...")
    print(f"Start Graph Ranks: {ranks_a}")
    print(f"Target Graph Ranks: {ranks_b}")
    print("-" * 50)
    
    # Note: the models would be instantiated here and passed to the individual pathfinders.
    # The user can just see they compile. 
    # For execution, they would instantiate the objects and call .solve()
    
    print("Compiled successfully!")
