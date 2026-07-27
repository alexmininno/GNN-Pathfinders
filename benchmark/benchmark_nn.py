"""Unified Benchmark Neural Networks"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import IterableDataset, DataLoader
try:
    from tqdm import tqdm
except:
    pass
try:
    import matplotlib.pyplot as plt
except:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
from src.model_autoregressive import AutoregressiveGPS
from src.siamese_dataset import safe_laplacian_pe
from src.model_siamese import SiameseSeiberg

try:
    from scripts.plot_style import JHEPPlot, get_latex_name
except ImportError:
    class JHEPPlot:
        prcolor = (29/255, 53/255, 87/255)
        seccolor = (69/255, 123/255, 157/255)
        tercolor = (152/255, 193/255, 217/255)
        def __init__(self, **kw): 
            self.fontsize = kw.get('fontsize', 11)
        def create_figure(self, **kw): import matplotlib.pyplot as plt; return plt.subplots(**kw)
        def save(self, p, **kw): import matplotlib.pyplot as plt; plt.savefig(p, bbox_inches='tight', dpi=200); plt.close()
        def add_legend(self, **kw): import matplotlib.pyplot as plt; plt.legend(**kw)

    def get_latex_name(name): return name

def export_tikz_legend(ax, filepath):
    import numpy as np
    import matplotlib.colors as mcolors
    legend = ax.get_legend()
    if not legend: return
    colors_used = {}
    lines_info = []
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        color = 'black'
        ls = 'solid'
        c = None
        if hasattr(handle, 'get_color'):
            c = handle.get_color()
        elif hasattr(handle, 'get_facecolor'):
            fc = handle.get_facecolor()
            if len(fc) > 0: c = fc[0]
        elif hasattr(handle, 'get_edgecolor'):
            ec = handle.get_edgecolor()
            if len(ec) > 0: c = ec[0]

        if c is not None:
            if isinstance(c, np.ndarray): c = np.squeeze(c).tolist()
            try:
                if isinstance(c, (list, tuple)) and len(c) > 0:
                    if len(c) == 1 and isinstance(c[0], str): c = c[0]
                    elif isinstance(c[0], (list, tuple)): c = c[0]
                color = mcolors.to_hex(c)
            except Exception: pass

        if hasattr(handle, 'get_linestyle'):
            ls_raw = handle.get_linestyle()
            if isinstance(ls_raw, list) and len(ls_raw) > 0: ls_raw = ls_raw[0]
            if ls_raw in ['--', 'dashed']: ls = 'dashed'
            elif ls_raw in [':', 'dotted']: ls = 'dotted'
            elif ls_raw in ['-.', 'dashdot']: ls = 'dashdotted'
            else: ls = 'solid'
            
        label = text.get_text()
        cname = f"col{len(colors_used)}"
        hex_val = color.lstrip('#').upper()
        
        found = False
        for k, v in colors_used.items():
            if v == hex_val:
                cname = k
                found = True
                break
        if not found: colors_used[cname] = hex_val
        lines_info.append({'name': cname, 'ls': ls, 'label': label})
        
    if not lines_info: return
    with open(filepath, 'w') as f:
        f.write("\\begin{tikzpicture}\n")
        for cname, hex_val in colors_used.items():
            f.write(f"    \\definecolor{{{cname}}}{{HTML}}{{{hex_val}}}\n")
        f.write("\n    \\matrix [\n        draw, \n        fill=white, \n        inner sep=4pt, \n")
        f.write("        nodes={inner sep=2pt, anchor=west},\n        column 1/.style={nodes={anchor=center}},\n")
        f.write("        column 3/.style={nodes={anchor=center}},\n        font=\\small,\n        row sep=0pt\n")
        f.write("    ] (legend) at (0,0) { \n")
        for i in range(0, len(lines_info), 2):
            item1 = lines_info[i]
            f.write(f"        \\draw[{item1['name']}, line width=1.2pt, {item1['ls']}] (0,0) -- (0.4,0); & \\node {{{item1['label']}}}; ")
            if i + 1 < len(lines_info):
                item2 = lines_info[i+1]
                f.write(f"& \\draw[{item2['name']}, line width=1.2pt, {item2['ls']}] (0,0) -- (0.4,0); & \\node {{{item2['label']}}}; \\\\\n")
            else:
                f.write("\\\\\n")
        f.write("    };\n\\end{tikzpicture}\n")

def adjust_figsize_for_readability(fig, jp):
    for ax in fig.axes:
        num_xticks = len(ax.get_xticklabels())
        num_yticks = len(ax.get_yticklabels())
        min_w = num_xticks * 0.35 * (jp.fontsize / 9.0)
        min_h = num_yticks * 0.25 * (jp.fontsize / 9.0)
        curr_w, curr_h = fig.get_size_inches()
        new_w, new_h = max(curr_w, min_w), max(curr_h, min_h)
        if new_w > curr_w or new_h > curr_h:
            fig.set_size_inches(new_w, new_h)

class InterceptJP:
    def __init__(self, jp, is_045, make_pdf=False):
        self.jp = jp
        self.is_045 = is_045
        self.make_pdf = make_pdf
        self.last_fig = None
        self.last_ax = None
        self.prcolor = getattr(jp, 'prcolor', 'blue')
        self.seccolor = getattr(jp, 'seccolor', 'orange')
        self.tercolor = getattr(jp, 'tercolor', 'green')
        self.fontsize = getattr(jp, 'fontsize', 11)
    def create_figure(self, **kwargs):
        fig, ax = self.jp.create_figure(**kwargs)
        self.last_fig = fig
        self.last_ax = ax
        return fig, ax
    def add_legend(self, ax=None, **kwargs):
        if ax is None: ax = self.last_ax
        kwargs['loc'] = 'center left'
        kwargs['bbox_to_anchor'] = (1.05, 0.5)
        self.jp.add_legend(ax=ax, **kwargs)
    def save(self, path, **kwargs):
        if self.last_fig:
            adjust_figsize_for_readability(self.last_fig, self.jp)
            base = path.rsplit('.', 1)[0]
            if self.is_045:
                for ax in self.last_fig.axes:
                    if ax.get_legend() is not None:
                        export_tikz_legend(ax, f"{base}_legend.tex")
                        ax.get_legend().remove()
                    ax.set_title("")
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                if self.make_pdf:
                    self.jp.save(f"{base}_045.pdf", bbox_inches="tight")
            else:
                self.last_fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
                if self.make_pdf:
                    self.jp.save(f"{base}.pdf", bbox_inches="tight")
#  Data loading
# ═══════════════════════════════════════════════════════════════

def get_inference_tasks(dataset_root, node_groups):
    """Find all test chunk files."""
    tasks = []
    for nodes in sorted(node_groups):
        node_dir = os.path.join(dataset_root, str(nodes))
        if not os.path.isdir(node_dir):
            continue
        dist_dirs = sorted(d for d in os.listdir(node_dir)
                           if d.startswith("dist_") and d != "dist_NaN"
                           and os.path.isdir(os.path.join(node_dir, d)))
        for dd in dist_dirs:
            dist_val = int(dd.split("_")[1])
            if dist_val == 0:
                continue
            test_dir = os.path.join(node_dir, dd, "test")
            if not os.path.isdir(test_dir):
                continue
            chunk_files = sorted(os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".pt"))
            for cf in chunk_files:
                tasks.append((cf, nodes, dist_val))
    return tasks

def count_total_pairs(tasks, num_workers, max_per_bucket=None):
    """Count total pairs to set a tqdm progress bar total."""
    import concurrent.futures
    import torch
    from tqdm import tqdm
    bucket_counts = defaultdict(int)

    def _count(t):
        cf, nodes, dist_val = t
        try:
            return len(torch.load(cf, map_location="cpu", weights_only=False)), nodes, dist_val
        except:
            return 0, nodes, dist_val

    total_pairs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for n, nodes, dist_val in tqdm(executor.map(_count, tasks), total=len(tasks), desc="Counting pairs"):
            if max_per_bucket:
                remaining = max_per_bucket - bucket_counts[(nodes, dist_val)]
                if remaining <= 0:
                    continue
                n = min(n, remaining)
            bucket_counts[(nodes, dist_val)] += n
            total_pairs += n
    return total_pairs

# ═══════════════════════════════════════════════════════════════
#  Test-set inference
# ═══════════════════════════════════════════════════════════════


class InferenceStreamDataset(IterableDataset):
    """Streams evaluation pairs directly from chunks to save memory."""
    def __init__(self, tasks, max_per_bucket=None):
        super().__init__()
        self.tasks = tasks
        self.max_per_bucket = max_per_bucket

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_tasks = self.tasks
        else:
            # Stagger tasks across workers
            worker_tasks = self.tasks[worker_info.id::worker_info.num_workers]

        bucket_counts = defaultdict(int)

        import torch
        for cf, nodes, dist_val in worker_tasks:
            # Worker-local limit check (approximate, since workers don't share bucket_counts)
            if self.max_per_bucket and bucket_counts[(nodes, dist_val)] >= self.max_per_bucket:
                continue
            try:
                chunk = torch.load(cf, map_location="cpu", weights_only=False)
                for item in chunk:
                    if self.max_per_bucket and bucket_counts[(nodes, dist_val)] >= self.max_per_bucket:
                        break
                    g_a, g_b, dist, seq = item
                    fid = getattr(g_a, "family_id", None)
                    bucket_counts[(nodes, dist_val)] += 1
                    yield g_a, g_b, float(dist), nodes, fid
                del chunk
            except Exception as e:
                print(f"  [!] Error loading {cf}: {e}")

def collate_pairs(batch):
    ga_list = [p[0] for p in batch]
    gb_list = [p[1] for p in batch]
    batch_a = Batch.from_data_list(ga_list)
    batch_b = Batch.from_data_list(gb_list)
    return batch_a, batch_b, batch

def evaluate_monotonicity(batch_pairs, parent_preds, model, device):
    """
    Evaluates 1-hop physical mutations to check the heuristic triangle inequality.
    """
    try:
        from scripts.generate_theories_dataset_new import mutate_ranks_kernel, mutate_adjacency_kernel, is_connected_kernel
    except ImportError:
        return []
    
    from torch_geometric.data import Data, Batch
    import torch
    import numpy as np

    mutated_a_list = []
    target_b_list = []
    parent_idx_list = []
    
    for i, (g_a, g_b, true_d, nc, fid) in enumerate(batch_pairs):
        num_nodes = g_a.num_nodes
        ranks = g_a.x[:, 0].cpu().numpy().astype(np.int64)
        
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)
        if g_a.edge_index.numel() > 0:
            row, col = g_a.edge_index.cpu().numpy()
            attr = g_a.edge_attr.cpu().numpy().flatten().astype(np.int64)
            adj[row, col] = attr

        for k in range(num_nodes):
            new_ranks = mutate_ranks_kernel(ranks, adj, k, True)
            if new_ranks.size == 0:
                continue
            new_adj = mutate_adjacency_kernel(adj, k)
            if new_adj.size == 0:
                continue
            if not is_connected_kernel(new_adj):
                continue
                
            A = torch.tensor(new_adj, dtype=torch.int64)
            indices = (A > 0).nonzero().t()
            if indices.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.int64)
            else:
                edge_index = indices.long()
                edge_attr = A[indices[0], indices[1]].view(-1, 1)

            ranks_tensor = torch.tensor(new_ranks, dtype=torch.int64).view(-1, 1)
            dual_flag = torch.zeros((num_nodes, 1), dtype=torch.int64)
            x = torch.cat([ranks_tensor, dual_flag], dim=1)

            data_a_mut = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            mutated_a_list.append(data_a_mut)
            target_b_list.append(g_b)
            parent_idx_list.append(i)
            
    if not mutated_a_list:
        return []
        
    batch_a_mut = Batch.from_data_list(mutated_a_list).to(device)
    batch_b_tgt = Batch.from_data_list(target_b_list).to(device)
    
    with torch.no_grad():
        child_preds = model(batch_a_mut, batch_b_tgt).view(-1).cpu().numpy()
        
    mono_results = []
    for mut_i, p_idx in enumerate(parent_idx_list):
        h_parent = float(parent_preds[p_idx])
        h_child = float(child_preds[mut_i])
        delta_h = abs(h_parent - h_child)
        
        nc = batch_pairs[p_idx][3]
        true_d = batch_pairs[p_idx][2]
        fid = batch_pairs[p_idx][4]
        
        mono_results.append({
            "node_count": nc,
            "true_distance": int(true_d),
            "family_uuid": fid,
            "h_parent": h_parent,
            "h_child": h_child,
            "delta_h": delta_h,
            "is_violation": int(delta_h > 1.0)
        })
        
    return mono_results


def get_checkpoint_hidden_channels(state_dict, fallback_dim, model_name="Model"):
    """Inspect state_dict to auto-detect hidden_channels."""
    if isinstance(state_dict, dict):
        sd = state_dict.get("model_state_dict", state_dict)
        for key in ["node_proj.weight", "encoders.0.conv.lin_l.weight", "encoders.0.conv.bias"]:
            if key in sd and hasattr(sd[key], "shape"):
                detected_dim = sd[key].shape[0]
                if fallback_dim is not None and detected_dim != fallback_dim:
                    print(f"[{model_name}] Note: Checkpoint hidden_channels ({detected_dim}) overrides CLI argument ({fallback_dim}).")
                elif fallback_dim is None:
                    print(f"[{model_name}] Auto-detected hidden_channels={detected_dim} from checkpoint.")
                return detected_dim
    return fallback_dim if fallback_dim is not None else 128


def pyg_to_ranks_adj(data):
    """Extract (ranks, adj_matrix) from a PyG Data object."""
    import numpy as np
    num_nodes = data.num_nodes
    ranks = data.x[:, 0].cpu().numpy().astype(np.int64).tolist()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    if data.edge_index.numel() > 0:
        row, col = data.edge_index.cpu().numpy()
        attr = data.edge_attr.cpu().numpy().flatten().astype(np.int64)
        adj[row, col] = attr
    return ranks, adj.tolist()


def permute_siamese_graph(data):
    """Apply a random node permutation to a PyG Data object for Siamese evaluation (no PE)."""
    import torch
    from torch_geometric.data import Data
    N = data.num_nodes
    perm = torch.randperm(N, device=data.x.device)
    inv_perm = torch.argsort(perm)
    new_x = data.x[perm]
    if data.edge_index.numel() > 0:
        new_edge_index = inv_perm[data.edge_index]
    else:
        new_edge_index = data.edge_index.clone()
    return Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
        num_nodes=N,
    )


def evaluate_deterministic_and_permutation(batch_pairs, dist_pred, model, device, max_deter_steps=1000):
    """
    Evaluates 3-way distance benchmark (Siamese d_pred vs Database d_data vs Deterministic d_true)
    and permutation invariance under node relabeling on the exact same test pairs.
    """
    import torch
    from torch_geometric.data import Batch
    from pathfinders.find_path import LCAPathfinder

    pathfinder = LCAPathfinder()
    perm_a_list = []
    perm_b_list = []

    for g_a, g_b, true_d, nc, fid in batch_pairs:
        perm_a_list.append(permute_siamese_graph(g_a))
        perm_b_list.append(permute_siamese_graph(g_b))

    batch_a_perm = Batch.from_data_list(perm_a_list).to(device)
    batch_b_perm = Batch.from_data_list(perm_b_list).to(device)

    with torch.no_grad():
        dist_perm = model(batch_a_perm, batch_b_perm).view(-1).cpu().numpy()

    results = []
    for j, (g_a, g_b, true_d, nc, fid) in enumerate(batch_pairs):
        pred_d = float(dist_pred[j])
        perm_d = float(dist_perm[j])
        delta_perm = abs(pred_d - perm_d)

        ranks_a, adj_a = pyg_to_ranks_adj(g_a)
        ranks_b, adj_b = pyg_to_ranks_adj(g_b)

        res = pathfinder.find_path(ranks_a, adj_a, ranks_b, adj_b, max_steps=max_deter_steps, enforce_anomaly_free=True)
        if res["status"] == "success":
            deter_d = len(res["path"])
            status_str = "success"
        else:
            deter_d = int(true_d)  # fallback if search didn't converge within max_steps
            status_str = res["status"]

        results.append({
            "node_count": nc,
            "database_distance": int(true_d),
            "deterministic_distance": deter_d,
            "predicted_distance": round(pred_d, 4),
            "permuted_predicted_distance": round(perm_d, 4),
            "delta_perm": round(delta_perm, 6),
            "mae_database": round(abs(pred_d - float(true_d)), 4),
            "mae_deterministic": round(abs(pred_d - float(deter_d)), 4),
            "deter_status": status_str,
            "family_uuid": fid,
        })
    return results


def _process_deterministic_benchmark(df_deter, args):
    """Save detailed/summary CSVs and generate plots for 3-way distance and permutation benchmark."""
    import os
    import pandas as pd
    import numpy as np

    output_dir = args.output_dir
    detailed_csv = os.path.join(output_dir, "siamese_deterministic_benchmark_detailed.csv")
    df_deter.to_csv(detailed_csv, index=False)
    print(f"\nDetailed 3-way deterministic benchmark saved to {detailed_csv}")

    df_succ = df_deter[df_deter["deter_status"] == "success"].copy()
    if df_succ.empty:
        df_succ = df_deter.copy()
        print("  [Warning] No pairs succeeded in deterministic solver; summarizing all pairs.")

    overestimates = (df_succ["deterministic_distance"] < df_succ["database_distance"]).sum()
    total_succ = len(df_succ)
    over_pct = (overestimates / total_succ * 100) if total_succ > 0 else 0.0

    print("\n" + "=" * 60)
    print("3-WAY DETERMINISTIC & PERMUTATION BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total evaluated pairs (solver success): {total_succ} / {len(df_deter)}")
    print(f"Database overestimation rate (d_true < d_data): {overestimates} ({over_pct:.2f}%)")
    if overestimates > 0:
        avg_red = (df_succ["database_distance"] - df_succ["deterministic_distance"])[df_succ["deterministic_distance"] < df_succ["database_distance"]].mean()
        print(f"Average distance reduction when overestimated: {avg_red:.2f} steps")

    mae_data = df_succ["mae_database"].mean()
    mae_deter = df_succ["mae_deterministic"].mean()
    mean_delta = df_succ["delta_perm"].mean()
    max_delta = df_succ["delta_perm"].max()

    print(f"Overall MAE vs Database (d_data):       {mae_data:.4f}")
    print(f"Overall MAE vs Deterministic (d_true):  {mae_deter:.4f}")
    print(f"Permutation Invariance Shift (mean):    {mean_delta:.6f} (max: {max_delta:.6f})")
    print("=" * 60)

    summary_rows = []
    for d in sorted(df_succ["database_distance"].unique()):
        sub = df_succ[df_succ["database_distance"] == d]
        summary_rows.append({
            "database_distance": d,
            "count": len(sub),
            "mae_vs_database": round(sub["mae_database"].mean(), 4),
            "mae_vs_deterministic": round(sub["mae_deterministic"].mean(), 4),
            "overestimate_rate_pct": round((sub["deterministic_distance"] < sub["database_distance"]).mean() * 100, 2),
            "mean_perm_shift": round(sub["delta_perm"].mean(), 6),
            "max_perm_shift": round(sub["delta_perm"].max(), 6),
        })
    df_sum = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "siamese_deterministic_benchmark_summary.csv")
    df_sum.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to {summary_csv}")

    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)

    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, "make_pdf", False):
            continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, "make_pdf", False))
        wl = not is_045
        fig, ax = jp.create_figure()
        ax.plot(df_sum["database_distance"], df_sum["mae_vs_database"], "o-", label="MAE vs Database ($d_{\\rm data}$)")
        ax.plot(df_sum["database_distance"], df_sum["mae_vs_deterministic"], "s--", label="MAE vs Deterministic ($d_{\\rm true}$)")
        ax.set_xlabel("Database Distance ($d_{\\rm data}$)")
        ax.set_ylabel("Mean Absolute Error")
        if wl:
            ax.set_title("Siamese MAE: Database vs. Deterministic Shortest Path")
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, "siamese_mae_comparison"))


def run_inference(args):
    """Run model on all test pairs, save predictions CSV, generate plots."""
    import torch
    from torch_geometric.data import Batch
    from src.model_siamese import SiameseSeiberg

    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    dim_fallback = getattr(args, "hidden_channels_siamese", 64)
    hidden_dim = get_checkpoint_hidden_channels(state_dict, dim_fallback, "Siamese")
    model = SiameseSeiberg(hidden_channels=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # Detect node groups
    node_groups = args.nodes or sorted(int(d) for d in os.listdir(args.dataset_root)
                                        if d.isdigit() and os.path.isdir(os.path.join(args.dataset_root, d)))
    print(f"Node groups: {node_groups}")

    tasks = get_inference_tasks(args.dataset_root, node_groups)
    print(f"Found {len(tasks)} chunk files.")

    total_pairs = count_total_pairs(tasks, args.num_workers, args.max_pairs_per_bucket)
    print(f"\nTotal test pairs: {total_pairs}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    mono_results_all = []
    deter_results_all = []
    embeddings_a, embeddings_b, embed_meta = [], [], []
    batch_size = 64

    dataset = InferenceStreamDataset(tasks, args.max_pairs_per_bucket)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers,
                        collate_fn=collate_pairs)

    from tqdm import tqdm
    import math
    total_batches = math.ceil(total_pairs / batch_size) if total_pairs > 0 else None

    with torch.no_grad():
        for i, (batch_a, batch_b, batch_pairs) in enumerate(tqdm(loader, desc="Running inference", total=total_batches)):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)

            # Forward pass
            dist_pred = model(batch_a, batch_b).view(-1).cpu().numpy()
            
            if args.evaluate_monotonicity_siamese:
                mono_batch = evaluate_monotonicity(batch_pairs, dist_pred, model, device)
                mono_results_all.extend(mono_batch)

            if getattr(args, "evaluate_deterministic_benchmark_siamese", False) or getattr(args, "evaluate_deterministic_benchmark", False):
                max_steps = getattr(args, "max_deter_steps_siamese", None) or getattr(args, "max_deter_steps", 1000)
                deter_batch = evaluate_deterministic_and_permutation(batch_pairs, dist_pred, model, device, max_steps)
                deter_results_all.extend(deter_batch)

            # Optionally extract embeddings
            if args.extract_embeddings_siamese:
                h_a, h_b, z_a, z_b = model.encode_graphs(batch_a, batch_b)
                embeddings_a.append(z_a.cpu().numpy())
                embeddings_b.append(z_b.cpu().numpy())

            for j, (g_a, g_b, true_d, nc, fid) in enumerate(batch_pairs):
                pred_d = float(dist_pred[j])
                results.append({
                    "node_count": nc, "true_distance": int(true_d),
                    "predicted_distance": round(pred_d, 4),
                    "abs_error": round(abs(pred_d - true_d), 4),
                    "family_uuid": fid,
                })
                if args.extract_embeddings_siamese:
                    embed_meta.append({"node_count": nc, "true_distance": int(true_d),
                                       "family_uuid": fid, "role": "pair"})

    # Save raw predictions
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to {csv_path}")

    df_mono = pd.DataFrame()
    if args.evaluate_monotonicity_siamese and mono_results_all:
        df_mono = pd.DataFrame(mono_results_all)
        mono_csv = os.path.join(args.output_dir, "monotonicity_eval.csv")
        df_mono.to_csv(mono_csv, index=False)
        print(f"Monotonicity evaluations saved to {mono_csv}")

    # Save embeddings
    if args.extract_embeddings_siamese and embeddings_a:
        emb_a = np.concatenate(embeddings_a, axis=0)
        emb_b = np.concatenate(embeddings_b, axis=0)
        npz_path = os.path.join(args.output_dir, "embeddings.npz")
        np.savez(npz_path, embeddings_a=emb_a, embeddings_b=emb_b, 
                 meta=np.array([str(m) for m in embed_meta]))
        print(f"Embeddings saved to {npz_path} ({emb_a.shape[0]} pairs, {emb_a.shape[1]}-dim)")

    # Generate all plots
    _generate_inference_plots(df, args)

    # Generate embedding plots if available
    if args.extract_embeddings_siamese and embeddings_a:
        _generate_embedding_plots(emb_a, emb_b, embed_meta, args)
        
    if args.evaluate_monotonicity_siamese and not df_mono.empty:
        _generate_monotonicity_plots(df_mono, args)

    if (getattr(args, "evaluate_deterministic_benchmark_siamese", False) or getattr(args, "evaluate_deterministic_benchmark", False)) and deter_results_all:
        df_deter = pd.DataFrame(deter_results_all)
        _process_deterministic_benchmark(df_deter, args)

    # Summary stats
    _print_summary(df, df_mono, args.output_dir)


def _generate_inference_plots(df, args):
    output_dir = args.output_dir
    """Generate per-distance MAE, and robustness plots."""
    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)
        
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, 'make_pdf', False): continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, 'make_pdf', False))
        wl = not is_045

        # 2. Per-distance MAE
        _plot_per_distance_mae(df, os.path.join(output_dir, "mae_by_distance.png"), jp, wl)
        # 3. Per-distance MAE box
        _plot_per_distance_mae_box(df, os.path.join(output_dir, "mae_by_distance_box.png"), jp, wl)
        # 4. Robustness by node count
        _plot_robustness_by_nodes(df, os.path.join(output_dir, "mae_by_nodes.png"), jp, wl)
        # 5. Error distribution
        _plot_error_distribution(df, os.path.join(output_dir, "error_distribution.png"), jp, wl)
        # 6. Predicted vs True scatter
        _plot_pred_vs_true(df, os.path.join(output_dir, "pred_vs_true.png"), jp, wl)
        # 7. Per-family accuracy (if multiple families)
        if df["family_uuid"].nunique() > 1:
            _plot_per_family_mae(df, os.path.join(output_dir, "mae_by_family.png"), jp, wl)


def _plot_per_distance_mae(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    grouped = df.groupby("true_distance")["abs_error"]
    dists = sorted(grouped.groups.keys())
    means = [grouped.get_group(d).mean() for d in dists]
    stds = [grouped.get_group(d).std() / np.sqrt(len(grouped.get_group(d))) for d in dists]
    ax.errorbar(dists, means, yerr=stds, marker='o', capsize=4,
                color=jp.prcolor, linewidth=1.5)
    if with_labels:
        ax.set_title("MAE by True Distance")
        ax.set_xlabel("True Distance")
        ax.set_ylabel("Mean Absolute Error")
    ax.grid(True, alpha=0.15)
    jp.save(path)


def _plot_per_distance_mae_box(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    dists = sorted(df["true_distance"].unique())
    box_data = [df[df["true_distance"] == d]["abs_error"].values for d in dists]
    bp = ax.boxplot(box_data, tick_labels=[str(d) for d in dists], patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(jp.tercolor); patch.set_alpha(0.6)
    if with_labels:
        ax.set_title("Error Distribution by Distance")
        ax.set_xlabel("True Distance")
        ax.set_ylabel("Absolute Error")
    jp.save(path)


def _plot_robustness_by_nodes(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    grouped = df.groupby("node_count")["abs_error"]
    nodes = sorted(grouped.groups.keys())
    means = [grouped.get_group(n).mean() for n in nodes]
    stds = [grouped.get_group(n).std() / np.sqrt(len(grouped.get_group(n))) for n in nodes]
    bars = ax.bar([str(n) for n in nodes], means, yerr=stds, capsize=4,
                  color=jp.seccolor, alpha=0.8, edgecolor='white')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{m:.3f}", ha='center', va='bottom', fontsize=8)
    if with_labels:
        ax.set_title("MAE by Node Count (Robustness)")
        ax.set_xlabel("Node Count")
        ax.set_ylabel("Mean Absolute Error")
    jp.save(path)


def _plot_error_distribution(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    errors = df["predicted_distance"] - df["true_distance"]
    ax.hist(errors, bins=80, color=jp.prcolor, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=f"Median: {errors.median():.3f}")
    if with_labels:
        ax.set_title("Prediction Error Distribution")
        ax.set_xlabel("Predicted − True Distance")
        ax.set_ylabel("Count")
    jp.add_legend(ax=ax)
    jp.save(path)


def _plot_pred_vs_true(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    ax.scatter(df["true_distance"], df["predicted_distance"], alpha=0.05, s=4,
               color=jp.prcolor, rasterized=True)
    lims = [df["true_distance"].min() - 0.5, df["true_distance"].max() + 0.5]
    ax.plot(lims, lims, 'r--', alpha=0.7, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim([0, lims[1] + 1])
    if with_labels:
        ax.set_title("Predicted vs True Distance")
        ax.set_xlabel("True Distance")
        ax.set_ylabel("Predicted Distance")
    jp.add_legend(ax=ax)
    jp.save(path)


def _plot_per_family_mae(df, path, jp, with_labels):
    fig, ax = jp.create_figure()
    grouped = df.groupby("family_uuid")["abs_error"]
    def get_nodes(f):
        return df[df["family_uuid"] == f]["node_count"].iloc[0]
    families = sorted(grouped.groups.keys(), key=lambda f: (get_nodes(f), str(f)))
    means = [grouped.get_group(f).mean() for f in families]
    
    import json
    meta_path = "Databases/family_metadata.json"
    meta_rev = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                for k, v in meta.items():
                    if isinstance(v, dict):
                        meta_rev[v["uuid"]] = v.get("name", "Unknown")
                    else:
                        meta_rev[v] = "Unknown"
        except Exception:
            pass
            
    labels = []
    for f in families:
        if not f:
            labels.append("?")
        else:
            uuid_str = str(f)
            root = meta_rev.get(uuid_str, "Unknown")
            if root != "Unknown":
                labels.append(get_latex_name(root))
            else:
                labels.append(uuid_str[:8])
                
    ax.barh(range(len(labels)), means, color=jp.seccolor, alpha=0.8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    if with_labels:
        ax.set_title("MAE by Theory Family")
        ax.set_xlabel("Mean Absolute Error")
    jp.save(path)


def _generate_monotonicity_plots(df_mono, args):
    output_dir = args.output_dir
    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)
        
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, 'make_pdf', False): continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, 'make_pdf', False))
        wl = not is_045

        # 1. Delta h distribution
        fig, ax = jp.create_figure()
        ax.hist(df_mono["delta_h"], bins=50, color=jp.prcolor, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.9, label="Monotonicity Bound (1.0)")
        violation_rate = df_mono["is_violation"].mean() * 100
        ax.text(0.95, 0.95, f"Violations: {violation_rate:.2f}%", transform=ax.transAxes, 
                ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        if wl:
            ax.set_title(r"Distribution of Jumps $\Delta h$")
            ax.set_xlabel(r"Absolute Jump $|\Delta h|$")
            ax.set_ylabel("Mutation Edges Evaluated")
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, "monotonicity_delta_hist.png"))

        # 2. Violation Rate by Distance
        fig, ax = jp.create_figure()
        grouped = df_mono.groupby("true_distance")["is_violation"]
        dists = sorted(grouped.groups.keys())
        rates = [grouped.get_group(d).mean() * 100 for d in dists]
        ax.plot(dists, rates, marker='o', color=jp.seccolor, linewidth=2)
        if wl:
            ax.set_title("Monotonicity Violation Rate by Target Distance")
            ax.set_xlabel("True Target Distance")
            ax.set_ylabel(r"Violation Rate (\%)")
        ax.grid(True, alpha=0.15)
        jp.save(os.path.join(output_dir, "monotonicity_violations_by_distance.png"))
        
        # 3. Violation Rate by Nodes
        fig, ax = jp.create_figure()
        grouped = df_mono.groupby("node_count")["is_violation"]
        nodes = sorted(grouped.groups.keys())
        rates = [grouped.get_group(n).mean() * 100 for n in nodes]
        bars = ax.bar([str(n) for n in nodes], rates, color=jp.tercolor, alpha=0.8, edgecolor='white')
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    rf"{r:.1f}\%", ha='center', va='bottom', fontsize=8)
        if wl:
            ax.set_title("Monotonicity Violation Rate by Graph Size")
            ax.set_xlabel("Node Count")
            ax.set_ylabel(r"Violation Rate (\%)")
        jp.save(os.path.join(output_dir, "monotonicity_violations_by_nodes.png"))


# ═══════════════════════════════════════════════════════════════
#  Embedding visualization
# ═══════════════════════════════════════════════════════════════

def _generate_embedding_plots(emb_a, emb_b, meta, args):
    output_dir = args.output_dir
    """Generate t-SNE scatter plots colored by distance and node count."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[!] scikit-learn not found. Skipping t-SNE plots.")
        return

    # Combine source and target embeddings
    all_emb = np.concatenate([emb_a, emb_b], axis=0)
    all_dist = [m["true_distance"] for m in meta] * 2
    all_nodes = [m["node_count"] for m in meta] * 2

    # Subsample if too large
    max_pts = 8000
    if len(all_emb) > max_pts:
        idx = np.random.choice(len(all_emb), max_pts, replace=False)
        all_emb = all_emb[idx]
        all_dist = [all_dist[i] for i in idx]
        all_nodes = [all_nodes[i] for i in idx]

    print(f"Running t-SNE on {len(all_emb)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(all_emb)

    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)
        
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, 'make_pdf', False): continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, 'make_pdf', False))
        wl = not is_045

        # Color by distance
        fig, ax = jp.create_figure()
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=all_dist, cmap='viridis',
                        alpha=0.4, s=6, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        if wl:
            cbar.set_label("True Distance")
            ax.set_title("t-SNE Embedding (by Distance)")
        jp.save(os.path.join(output_dir, "tsne_by_distance.png"))

        # Color by node count
        fig, ax = jp.create_figure()
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=all_nodes, cmap='plasma',
                        alpha=0.4, s=6, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        if wl:
            cbar.set_label("Node Count")
            ax.set_title("t-SNE Embedding (by Node Count)")
        jp.save(os.path.join(output_dir, "tsne_by_nodes.png"))

    print("t-SNE plots saved.")


# ═══════════════════════════════════════════════════════════════
#  Latency benchmark
# ═══════════════════════════════════════════════════════════════

def run_siamese_latency_benchmark(args):
    """Time forward passes at various batch sizes."""
    import torch
    from torch_geometric.data import Data, Batch
    from src.model_siamese import SiameseSeiberg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    dim_fallback = getattr(args, "hidden_channels_siamese", 64)
    hidden_dim = get_checkpoint_hidden_channels(state_dict, dim_fallback, "Siamese")
    model = SiameseSeiberg(hidden_channels=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    def make_dummy(n_nodes=3):
        x = torch.randn(n_nodes, 2)
        edge_index = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        edge_attr = torch.ones(3, 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    warmup, repeats = 30, 200
    results = []

    os.makedirs(args.output_dir, exist_ok=True)

    print("Latency benchmark:")
    for bs in batch_sizes:
        g_list = [make_dummy() for _ in range(bs)]
        ba = Batch.from_data_list(g_list).to(device)
        bb = Batch.from_data_list(g_list).to(device)

        with torch.no_grad():
            for _ in range(warmup):
                model(ba, bb)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(ba, bb)
                times.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        throughput = bs / (mean_ms / 1000)
        results.append({"batch_size": bs, "mean_ms": round(mean_ms, 3),
                        "std_ms": round(std_ms, 3), "throughput_pairs_per_sec": round(throughput, 1)})
        print(f"  bs={bs:>4}: {mean_ms:.2f} ± {std_ms:.2f} ms  ({throughput:.0f} pairs/s)")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "latency.csv"), index=False)

    # Plot
    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)
        
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, 'make_pdf', False): continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, 'make_pdf', False))
        wl = not is_045

        fig, ax1 = jp.create_figure()
        ax1.plot(df["batch_size"], df["mean_ms"], 'o-', color=jp.prcolor, label="Latency")
        ax1.fill_between(df["batch_size"], df["mean_ms"]-df["std_ms"],
                         df["mean_ms"]+df["std_ms"], alpha=0.2, color=jp.prcolor)
        ax2 = ax1.twinx()
        ax2.plot(df["batch_size"], df["throughput_pairs_per_sec"], 's--',
                 color=jp.seccolor, label="Throughput")
        ax1.set_xscale("log", base=2)
        if wl:
            ax1.set_title("Inference Latency & Throughput")
            ax1.set_xlabel("Batch Size")
            ax1.set_ylabel("Latency (ms)")
            ax2.set_ylabel("Throughput (pairs/s)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        jp.save(os.path.join(args.output_dir, "latency.png"))

    print(f"Latency results saved to {args.output_dir}")


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════

def _print_summary(df, df_mono, output_dir):
    """Print and save summary statistics."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    overall_mae = df["abs_error"].mean()
    overall_std = df["abs_error"].std()
    print(f"Overall MAE: {overall_mae:.4f} ± {overall_std:.4f}")
    print(f"Total pairs: {len(df)}")

    if not df_mono.empty:
        total_mutations = len(df_mono)
        violations = df_mono["is_violation"].sum()
        violation_rate = (violations / total_mutations) * 100
        print(f"\nMonotonicity Empirical Evaluation:")
        print(f"  Valid 1-hop mutations tested: {total_mutations}")
        print(f"  Violations (Δh > 1.0): {violations} ({violation_rate:.2f}%)")

    print("\nPer-distance breakdown:")
    for d in sorted(df["true_distance"].unique()):
        sub = df[df["true_distance"] == d]
        print(f"  d={d}: MAE={sub['abs_error'].mean():.4f} ± {sub['abs_error'].std():.4f} (n={len(sub)})")

    print("\nPer-node breakdown:")
    for n in sorted(df["node_count"].unique()):
        sub = df[df["node_count"] == n]
        print(f"  N={n}: MAE={sub['abs_error'].mean():.4f} ± {sub['abs_error'].std():.4f} (n={len(sub)})")

    # Save summary
    summary = {
        "overall_mae": overall_mae, "overall_std": overall_std, "total_pairs": len(df),
        "num_distances": df["true_distance"].nunique(), "num_node_counts": df["node_count"].nunique(),
        "num_families": df["family_uuid"].nunique(),
    }
    
    if not df_mono.empty:
        summary["total_mutations_tested"] = total_mutations
        summary["monotonicity_violations"] = violations
        summary["monotonicity_violation_rate"] = violation_rate

    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

# --- Accuracy Utilities ---


def accuracy_topk(output, target, topk=(1, 2, 3)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


class BenchmarkDataset(torch.utils.data.IterableDataset):
    def __init__(self, test_files, max_pairs_per_bucket=None):
        self.test_files = test_files
        self.max_pairs_per_bucket = max_pairs_per_bucket

    def __iter__(self):
        # In a multi-worker setup, we need to partition the files
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_list = self.test_files
        else:
            per_worker = int(
                np.ceil(len(self.test_files) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_list = self.test_files[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]

        bucket_counts = defaultdict(int)

        for nodes, true_dist, chunk_file in iter_list:
            if self.max_pairs_per_bucket and bucket_counts[(nodes, true_dist)] >= self.max_pairs_per_bucket:
                continue
            try:
                # Use a context manager to ensure the file handle is released
                with open(chunk_file, "rb") as f:
                    chunk_data = torch.load(f, map_location="cpu", weights_only=False)
                for item in chunk_data:
                    if self.max_pairs_per_bucket and bucket_counts[(nodes, true_dist)] >= self.max_pairs_per_bucket:
                        break
                    bucket_counts[(nodes, true_dist)] += 1
                    # item: (g_a, g_b, dist, seq)
                    yield (item[0], item[1], item[2], item[3], nodes, true_dist)
            except Exception:
                continue


def collate_ar_benchmark(batch, explosion_threshold=1e6):
    g_a_list, g_b_list, targets, dists, nodes_list, true_dists, family_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for item in batch:
        g_a, g_b, dist, seq, nodes, true_dist = item
        if len(seq) == 0:
            continue

        # Add PE safely
        g_a = safe_laplacian_pe(g_a)
        g_b = safe_laplacian_pe(g_b)
        if not getattr(g_a, "is_valid", True) or not getattr(g_b, "is_valid", True):
            continue

        f_id = getattr(g_a, "family_id", None)

        # Remove metadata that might cause batching crashes due to inconsistency
        for g in [g_a, g_b]:
            for attr in ["signature", "uuid", "family_id"]:
                if hasattr(g, attr):
                    delattr(g, attr)

        g_a_list.append(g_a)
        g_b_list.append(g_b)
        targets.append(int(seq[0]) - 1)  # 1-indexed to 0-indexed
        dists.append(int(dist))
        nodes_list.append(nodes)
        true_dists.append(true_dist)
        family_ids.append(f_id)

    if not g_a_list:
        return None

    batch_a = Batch.from_data_list(g_a_list)
    batch_b = Batch.from_data_list(g_b_list)
    targets_t = torch.tensor(targets, dtype=torch.long)

    mask_list = []
    for g in g_a_list:
        n = g.num_nodes
        ranks = g.x[:, 0].tolist()
        adj = torch.zeros((n, n))
        if g.edge_index.numel() > 0:
            adj[g.edge_index[0], g.edge_index[1]] = g.edge_attr.squeeze(-1).float()

        valid_actions = torch.ones(n, dtype=torch.bool)
        for k in range(n):
            n_f_in = sum(adj[i][k].item() * ranks[i] for i in range(n))
            new_rank = n_f_in - ranks[k]
            if new_rank <= 0 or new_rank > explosion_threshold:
                valid_actions[k] = False
        mask_list.append(valid_actions)

    action_mask = pad_sequence(mask_list, batch_first=True, padding_value=False)
    for b_idx, t_idx in enumerate(targets_t):
        action_mask[b_idx, t_idx] = True  # Always allow ground truth

    return batch_a, batch_b, targets_t, action_mask, nodes_list, true_dists, family_ids


def get_test_files(dataset_root):
    files = []
    if not os.path.exists(dataset_root):
        return files
    for n in os.listdir(dataset_root):
        n_dir = os.path.join(dataset_root, n)
        if not os.path.isdir(n_dir) or not n.isdigit():
            continue
        for d in os.listdir(n_dir):
            if not d.startswith("dist_") or d == "dist_NaN":
                continue

            try:
                dist_part = d.split("_")[1]
                dist_digits = "".join([c for c in dist_part if c.isdigit()])
                if not dist_digits:
                    continue
                dist_val = int(dist_digits)
            except (ValueError, IndexError):
                continue

            test_dir = os.path.join(n_dir, d, "test")
            if not os.path.isdir(test_dir):
                continue

            # Grab all .pt files, sort for determinism
            valid_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pt")])
            for f in valid_files:
                files.append((int(n), dist_val, os.path.join(test_dir, f)))

    return files


# --- Inference Utilities ---


def generate_dummy_data(n_nodes):
    x = torch.ones((n_nodes, 2), dtype=torch.float32)
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
    edge_attr = torch.ones((n_nodes * 2, 1), dtype=torch.float32)
    pe = torch.randn((n_nodes, 8), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pe = pe
    data.num_nodes = n_nodes
    return data


# --- Main Benchmark Runners ---


def run_inference_benchmark(model, device, n_nodes, iterations=50):
    print(f"\n--- [ Hardware Inference Benchmark ] ---")
    print(f"Device: {device} | Nodes: {n_nodes}")
    model.eval()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    with torch.no_grad():
        for bs in batch_sizes:
            data_list_a = [generate_dummy_data(n_nodes) for _ in range(bs)]
            data_list_b = [generate_dummy_data(n_nodes) for _ in range(bs)]
            batch_a = Batch.from_data_list(data_list_a).to(device)
            batch_b = Batch.from_data_list(data_list_b).to(device)

            # Warmup
            for _ in range(10):
                model(batch_a, batch_b)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                model(batch_a, batch_b)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - t0
            ms_per_batch = (elapsed / iterations) * 1000
            throughput = (bs * iterations) / elapsed
            print(
                f"Batch: {bs:<4} | Latency: {ms_per_batch:>7.2f} ms | Throughput: {throughput:>8.1f} graphs/s"
            )
            results.append({
                "nodes": n_nodes,
                "batch_size": bs,
                "latency_ms": ms_per_batch,
                "throughput_graphs_sec": throughput
            })
    return results


def count_total_items(test_files, num_workers, max_pairs_per_bucket=None):
    import concurrent.futures
    import torch
    from tqdm import tqdm
    from collections import defaultdict
    bucket_counts = defaultdict(int)

    def _count(t):
        nodes, dist_val, chunk_file = t
        try:
            with open(chunk_file, "rb") as f:
                return len(torch.load(f, map_location="cpu", weights_only=False)), nodes, dist_val
        except Exception:
            return 0, nodes, dist_val

    total = 0
    workers = max(1, num_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for n, nodes, dist_val in tqdm(
            executor.map(_count, test_files),
            total=len(test_files),
            desc="Counting graphs",
        ):
            if max_pairs_per_bucket:
                remaining = max_pairs_per_bucket - bucket_counts[(nodes, dist_val)]
                if remaining <= 0:
                    continue
                n = min(n, remaining)
            bucket_counts[(nodes, dist_val)] += n
            total += n
    return total


def run_accuracy_benchmark(model, device, args):
    print(f"\n--- [ Physical Accuracy Benchmark ] ---")

    # 1. FIX: Force evaluation mode
    model.eval()

    test_files = get_test_files(args.dataset_root)
    if args.nodes:
        test_files = [f for f in test_files if f[0] in args.nodes]

    if not test_files:
        print("No test files found for accuracy benchmark.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print(f"Found {len(test_files)} test files.")
    total_items = count_total_items(test_files, args.num_workers, args.max_pairs_per_bucket)
    import math

    total_batches = (
        math.ceil(total_items / args.batch_size) if total_items > 0 else None
    )
    print(
        f"Total test graphs: {total_items} (~{total_batches} batches). Commencing evaluation stream..."
    )

    dataset = BenchmarkDataset(test_files, args.max_pairs_per_bucket)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_ar_benchmark,
        pin_memory=(device.type == "cuda"),
    )

    try:
        with torch.no_grad():
            for collated in tqdm(loader, desc="Evaluating", total=total_batches):
                if not collated:
                    continue

                batch_a, batch_b, targets, mask, nodes_list, true_dists, family_ids = (
                    collated
                )
                batch_a, batch_b, targets, mask = (
                    batch_a.to(device),
                    batch_b.to(device),
                    targets.to(device),
                    mask.to(device),
                )

                # Get raw logits for physics violation check
                raw_logits = model(batch_a, batch_b, action_mask=None)
                # Apply mask manually
                logits = raw_logits.masked_fill(~mask, -1e9)

                # Cross Entropy Loss
                ce_loss = (
                    torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                    .cpu()
                    .tolist()
                )
                
                if args.evaluate_policy_margin_ar:
                    probs = torch.softmax(logits, dim=-1)
                    p_correct = probs[torch.arange(len(targets)), targets]
                    
                    incorrect_mask = mask.clone()
                    incorrect_mask[torch.arange(len(targets)), targets] = False
                    
                    p_incorrect = probs.masked_fill(~incorrect_mask, 0.0)
                    
                    max_k_inc = min(3, p_incorrect.size(1))
                    topk_incorrect = p_incorrect.topk(max_k_inc, dim=-1).values
                    
                    p_inc_1 = topk_incorrect[:, 0]
                    p_inc_2 = topk_incorrect[:, 1] if max_k_inc > 1 else torch.zeros_like(p_inc_1)
                    p_inc_3 = topk_incorrect[:, 2] if max_k_inc > 2 else torch.zeros_like(p_inc_1)
                    
                    margins_top1 = (p_correct - p_inc_1).cpu().tolist()
                    margins_top2 = (p_correct - p_inc_2).cpu().tolist()
                    margins_top3 = (p_correct - p_inc_3).cpu().tolist()
                    
                    is_inv_top1 = [1 if m < 0 else 0 for m in margins_top1]
                    is_inv_top2 = [1 if m < 0 else 0 for m in margins_top2]
                    is_inv_top3 = [1 if m < 0 else 0 for m in margins_top3]
                else:
                    margins_top1 = [0.0] * len(targets)
                    margins_top2 = [0.0] * len(targets)
                    margins_top3 = [0.0] * len(targets)
                    is_inv_top1 = [0] * len(targets)
                    is_inv_top2 = [0] * len(targets)
                    is_inv_top3 = [0] * len(targets)

                # Unmasked argmax for physical validity
                unmasked_preds = raw_logits.argmax(dim=-1)
                valid_physics = [
                    mask[i, unmasked_preds[i]].item() for i in range(len(targets))
                ]

                max_k = min(3, logits.size(1))
                _, preds = logits.topk(max_k, 1, True, True)

                for i in range(len(targets)):
                    target = targets[i].item()
                    topk_correct = [
                        int(target in preds[i, : k + 1]) for k in range(max_k)
                    ]
                    while len(topk_correct) < 3:
                        topk_correct.append(0)

                    results.append(
                        {
                            "nodes": nodes_list[i],
                            "true_distance": true_dists[i],
                            "family_uuid": family_ids[i],
                            "top1_correct": topk_correct[0],
                            "top2_correct": topk_correct[1],
                            "top3_correct": topk_correct[2],
                            "valid_physics": int(valid_physics[i]),
                            "ce_loss": ce_loss[i],
                            "policy_margin_top1": margins_top1[i],
                            "policy_margin_top2": margins_top2[i],
                            "policy_margin_top3": margins_top3[i],
                            "is_inversion_top1": is_inv_top1[i],
                            "is_inversion_top2": is_inv_top2[i],
                            "is_inversion_top3": is_inv_top3[i],
                        }
                    )
    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user. Processing partial results...")

    # 3. FIX: Check for empty results before DataFrame operations
    if not results:
        print("No valid evaluation samples processed.")
        return

    df = pd.DataFrame(results)

    raw_path = os.path.join(args.output_dir, "test_predictions.csv")
    df.to_csv(raw_path, index=False)

    agg_df = (
        df.groupby(["nodes", "true_distance"])
        .agg(
            {
                "top1_correct": "mean",
                "top2_correct": "mean",
                "top3_correct": "mean",
                "valid_physics": "mean",
                "ce_loss": "mean",
                "policy_margin_top1": "mean",
                "policy_margin_top2": "mean",
                "policy_margin_top3": "mean",
                "is_inversion_top1": "mean",
                "is_inversion_top2": "mean",
                "is_inversion_top3": "mean",
            }
        )
        .reset_index()
    )
    agg_df.rename(
        columns={
            "top1_correct": "top1_acc",
            "top2_correct": "top2_acc",
            "top3_correct": "top3_acc",
        },
        inplace=True,
    )

    stats_path = os.path.join(args.output_dir, "ar_accuracy_stats.csv")
    agg_df.to_csv(stats_path, index=False)
    print(f"Accuracy stats saved to {stats_path}")

    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except ImportError:
        jp_full = JHEPPlot(usetex=False, fontsize=11)
        jp_045 = JHEPPlot(usetex=False, fontsize=11)

    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not getattr(args, 'make_pdf', False): continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args, 'make_pdf', False))

        # 1. Heatmap
        fig, ax = jp.create_figure()
        pivot = agg_df.pivot(index="nodes", columns="true_distance", values="top1_acc")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        if not is_045:
            ax.set_title("AR Top-1 Next-Step Accuracy")
            ax.set_xlabel("True Distance")
            ax.set_ylabel("Nodes (N)")
        jp.save(os.path.join(args.output_dir, "ar_top1_heatmap.png"))

        # 2. Accuracy by distance
        fig, ax = jp.create_figure()
        dist_df = df.groupby("true_distance").mean(numeric_only=True).reset_index()
        for k, fmt, c in zip(
            [1, 2, 3], ["o-", "s--", "^:"], [jp.prcolor, jp.seccolor, jp.tercolor]
        ):
            col_name = f"top{k}_correct"
            if col_name in dist_df.columns:
                ax.plot(
                    dist_df["true_distance"],
                    dist_df[col_name],
                    fmt,
                    color=c,
                    label=f"Top-{k}",
                )
        ax.set_ylim(0, 1.05)
        if not is_045:
            ax.set_title("Accuracy by True Distance")
            ax.set_xlabel("True Distance")
            ax.set_ylabel("Accuracy")
        jp.add_legend(ax=ax)
        jp.save(os.path.join(args.output_dir, "ar_acc_by_distance.png"))

        # 3. Accuracy by family
        fig, ax = jp.create_figure()
        grouped = df.groupby("family_uuid")["top1_correct"]
        def get_nodes(f):
            return df[df["family_uuid"] == f]["nodes"].iloc[0]
        families = sorted(grouped.groups.keys(), key=lambda f: (get_nodes(f), str(f)))
        means = [grouped.get_group(f).mean() for f in families]
        
        import json
        meta_path = "Databases/family_metadata.json"
        meta_rev = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    for k, v in meta.items():
                        if isinstance(v, dict):
                            meta_rev[v["uuid"]] = v.get("name", "Unknown")
                        else:
                            meta_rev[v] = "Unknown"
            except Exception:
                pass
                
        labels = []
        for f in families:
            if not f:
                labels.append("?")
            else:
                uuid_str = str(f)
                root = meta_rev.get(uuid_str, "Unknown")
                if root != "Unknown":
                    labels.append(get_latex_name(root))
                else:
                    labels.append(uuid_str[:8])
                    
        ax.barh(range(len(labels)), means, color=jp.seccolor, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1.05)
        if not is_045:
            ax.set_title("Top-1 Accuracy by Theory Family")
            ax.set_xlabel("Accuracy")
        jp.save(os.path.join(args.output_dir, "ar_acc_by_family.png"))

        # 4. Valid Physics Rate by distance
        fig, ax = jp.create_figure()
        ax.plot(
            dist_df["true_distance"], dist_df["valid_physics"], "o-", color=jp.prcolor
        )
        ax.set_ylim(0, 1.05)
        if not is_045:
            ax.set_title("Unmasked Physical Validity Rate by Distance")
            ax.set_xlabel("True Distance")
            ax.set_ylabel("Validity Rate")
        jp.save(os.path.join(args.output_dir, "ar_physics_validity.png"))

        # 5. Cross Entropy by distance
        fig, ax = jp.create_figure()
        ax.plot(dist_df["true_distance"], dist_df["ce_loss"], "s--", color=jp.seccolor)
        if not is_045:
            ax.set_title("Cross-Entropy Loss by Distance")
            ax.set_xlabel("True Distance")
            ax.set_ylabel("Cross-Entropy Loss")
        jp.save(os.path.join(args.output_dir, "ar_cross_entropy.png"))

        # 6. Policy Margin plots
        if args.evaluate_policy_margin_ar:
            for k, color in zip([1, 2, 3], [jp.prcolor, jp.seccolor, jp.tercolor]):
                # Standalone histograms
                fig, ax = jp.create_figure()
                ax.hist(df[f"policy_margin_top{k}"], bins=50, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
                ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.9, label="Inversion Bound (M=0)")
                inversion_rate = df[f"is_inversion_top{k}"].mean() * 100
                ax.text(0.05, 0.95, rf"Inversions: {inversion_rate:.2f}\%", transform=ax.transAxes, 
                        ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                if not is_045:
                    ax.set_title(f"Distribution of Top-{k} Policy Margins")
                    ax.set_xlabel(rf"Probability Margin ($P_{{\text{{predicted}}}} - P_{{{k}\text{{-th\_best\_incorrect}}}}$)")
                    ax.set_ylabel("Samples")
                jp.add_legend(ax=ax)
                jp.save(os.path.join(args.output_dir, f"ar_policy_margin_top{k}_hist.png"))

            # Overlaid histograms (merged)
            fig, ax = jp.create_figure()
            for k, color in zip([1, 2, 3], [jp.prcolor, jp.seccolor, jp.tercolor]):
                ax.hist(df[f"policy_margin_top{k}"], bins=50, color=color, alpha=0.7, histtype='step', linewidth=1.5, label=f"Top-{k} Margin")
            ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.9, label="Inversion Bound (M=0)")
            if not is_045:
                ax.set_title("Overlaid Policy Margins (Top-1, 2, 3)")
                ax.set_xlabel(rf"Probability Margin ($P_{{\text{{predicted}}}} - P_{{k\text{{-th\_best\_incorrect}}}}$)")
                ax.set_ylabel("Samples")
            jp.add_legend(ax=ax)
            jp.save(os.path.join(args.output_dir, "ar_policy_margin_overlaid_hist.png"))
            
            # Subplots figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, (k, color) in enumerate(zip([1, 2, 3], [jp.prcolor, jp.seccolor, jp.tercolor])):
                ax = axes[i]
                ax.hist(df[f"policy_margin_top{k}"], bins=50, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
                ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.9)
                inversion_rate = df[f"is_inversion_top{k}"].mean() * 100
                ax.text(0.05, 0.95, rf"Inversions: {inversion_rate:.2f}\%", transform=ax.transAxes, 
                        ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                ax.set_title(f"Top-{k} Policy Margin")
                ax.set_xlabel(rf"$P_{{\text{{predicted}}}} - P_{{{k}\text{{-th\_best\_incorrect}}}}$")
                if i == 0:
                    ax.set_ylabel("Samples")
            plt.tight_layout()
            base_out = os.path.join(args.output_dir, "ar_policy_margin_subplots_hist")
            if is_045:
                plt.savefig(f"{base_out}_045.pdf", bbox_inches="tight", dpi=200)
            else:
                plt.savefig(f"{base_out}.pdf", bbox_inches="tight", dpi=200)
                plt.savefig(f"{base_out}.png", bbox_inches="tight", dpi=300)
            plt.close()
            
            # Line plot by distance
            fig, ax = jp.create_figure()
            ax.plot(dist_df["true_distance"], dist_df["policy_margin_top1"], "o-", color=jp.prcolor, label="Top-1")
            ax.plot(dist_df["true_distance"], dist_df["policy_margin_top2"], "s--", color=jp.seccolor, label="Top-2")
            ax.plot(dist_df["true_distance"], dist_df["policy_margin_top3"], "^:", color=jp.tercolor, label="Top-3")
            ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.9)
            ax.set_ylim(-1.05, 1.05)
            if not is_045:
                ax.set_title("Average Policy Margins by Target Distance")
                ax.set_xlabel("True Distance")
                ax.set_ylabel("Average Margin")
            jp.add_legend(ax=ax)
            jp.save(os.path.join(args.output_dir, "ar_policy_margin_by_distance.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Benchmark Neural Networks")
    
    # General Model Flags
    parser.add_argument("--siamese", action="store_true", help="Benchmark Siamese inference")
    parser.add_argument("--ar", action="store_true", help="Benchmark Autoregressive inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    
    # Inference params
    parser.add_argument("--dataset_root", type=str, default="Databases/Theories_dataset")
    parser.add_argument("--output_dir", type=str, default="analysis_unified/benchmarks")
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--hidden_channels_siamese", type=int, default=64)
    parser.add_argument("--hidden_channels_ar", type=int, default=128)
    parser.add_argument("--max_pairs_per_bucket", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Siamese Specific
    parser.add_argument("--extract_embeddings_siamese", action="store_true", help="Extract embeddings for t-SNE visualization (Siamese only)")
    parser.add_argument("--evaluate_monotonicity_siamese", action="store_true", help="Evaluate heuristic triangle inequality (Siamese only)")
    parser.add_argument("--benchmark_latency_siamese", action="store_true", help="Run latency benchmark only (no dataset needed, Siamese only)")
    parser.add_argument("--evaluate_deterministic_benchmark_siamese", "--evaluate_deterministic_benchmark", dest="evaluate_deterministic_benchmark_siamese", action="store_true", help="Evaluate 3-way distance benchmark and permutation invariance (Siamese only)")
    parser.add_argument("--max_deter_steps_siamese", "--max_deter_steps", dest="max_deter_steps_siamese", type=int, default=1000, help="Max steps for deterministic LCAPathfinder in 3-way benchmark")
    
    # AR Specific
    parser.add_argument("--only_inference_ar", action="store_true", help="Run only hardware inference benchmark (AR only)")
    parser.add_argument("--only_accuracy_ar", action="store_true", help="Run only physical accuracy benchmark (AR only)")
    parser.add_argument("--evaluate_policy_margin_ar", action="store_true", help="Evaluate local policy margin (AR only)")
    
    # Outputs
    parser.add_argument("--make_pdf", action="store_true", help="Generate .pdf and _045.pdf plots in addition to .png")
    
    args = parser.parse_args()
    
    if not args.siamese and not args.ar:
        args.siamese = True
        args.ar = True
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.siamese:
        print("Benchmarking Siamese...")
        siamese_out = os.path.join(args.output_dir, "siamese")
        args_s = argparse.Namespace(**vars(args))
        args_s.output_dir = siamese_out
        if args.benchmark_latency_siamese:
            run_siamese_latency_benchmark(args_s)
        else:
            run_inference(args_s)
            
    if args.ar:
        print("Benchmarking Autoregressive...")
        ar_out = os.path.join(args.output_dir, "ar")
        args_a = argparse.Namespace(**vars(args))
        args_a.output_dir = ar_out
        
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
        except Exception as e:
            print(f"Warning: Loading AR checkpoint failed: {e}")
            state_dict = {}

        hidden_dim = get_checkpoint_hidden_channels(state_dict, args.hidden_channels_ar, "Autoregressive")
        use_delta_a = True
        classifier_weight = state_dict.get('classifier.0.weight') if isinstance(state_dict, dict) else None
        if classifier_weight is not None and hasattr(classifier_weight, 'shape'):
            in_features = classifier_weight.shape[1]
            if in_features == hidden_dim * 4 + 5:
                use_delta_a = False
                print("Autoregressive: Detected legacy AR checkpoint (no delta_A). Running in backward compatibility mode.")

        model = AutoregressiveGPS(hidden_channels=hidden_dim, use_delta_a=use_delta_a).to(device)
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
            
        if not args.only_accuracy_ar:
            nodes_to_bench = args.nodes
            if not nodes_to_bench:
                try:
                    tf = get_test_files(args.dataset_root)
                    nodes_to_bench = sorted(list(set(f[0] for f in tf)))
                except Exception:
                    pass
                if not nodes_to_bench:
                    nodes_to_bench = [6, 7, 8, 9]
                    
            all_inference_results = []
            for n_nodes in nodes_to_bench:
                all_inference_results.extend(run_inference_benchmark(model, device, n_nodes))
                
            os.makedirs(args_a.output_dir, exist_ok=True)
            df_inf = pd.DataFrame(all_inference_results)
            df_inf.to_csv(os.path.join(args_a.output_dir, "ar_inference_benchmark.csv"), index=False)
            
            try:
                from scripts.plot_style import JHEPPlot
                jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
                jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
            except ImportError:
                jp_full = JHEPPlot(usetex=False, fontsize=11)
                jp_045 = JHEPPlot(usetex=False, fontsize=11)
                
            for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
                if is_045 and not getattr(args_a, 'make_pdf', False): continue
                jp = InterceptJP(raw_jp, is_045, make_pdf=getattr(args_a, 'make_pdf', False))
                
                fig, ax = jp.create_figure()
                for bs in sorted(df_inf["batch_size"].unique()):
                    subset = df_inf[df_inf["batch_size"] == bs]
                    ax.plot(subset["nodes"], subset["latency_ms"], marker='o', label=f"BS={bs}")
                ax.set_xlabel("Nodes")
                ax.set_ylabel("Latency (ms / batch)")
                ax.set_yscale("log")
                if not is_045: ax.set_title("Inference Latency by Node Count")
                jp.add_legend(ax=ax)
                jp.save(os.path.join(args_a.output_dir, "ar_inference_latency_vs_nodes"))
                
                fig, ax = jp.create_figure()
                for bs in sorted(df_inf["batch_size"].unique()):
                    subset = df_inf[df_inf["batch_size"] == bs]
                    ax.plot(subset["nodes"], subset["throughput_graphs_sec"], marker='s', label=f"BS={bs}")
                ax.set_xlabel("Nodes")
                ax.set_ylabel("Throughput (graphs / s)")
                ax.set_yscale("log")
                if not is_045: ax.set_title("Inference Throughput by Node Count")
                jp.add_legend(ax=ax)
                jp.save(os.path.join(args_a.output_dir, "ar_inference_throughput_vs_nodes"))
                
        if not args.only_inference_ar:
            run_accuracy_benchmark(model, device, args_a)

    print("Benchmarking completed successfully.")
