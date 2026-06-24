"""
Unified Evaluate & Analyze Pathfinders
"""

import os
import sys
import json
import math
import time
import argparse
import multiprocessing as mp
from collections import defaultdict
import torch
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    pass

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathfinders.find_path import (
    HeuristicBidirectionalPathfinder,
    AutoregressivePathfinder,
    HybridBidirectionalPathfinder,
    LCAPathfinder,
    HybridLCAPathfinder
)

from src.data_utils import get_graph_hash
from scripts.data_sampler import stream_sample, pyg_to_raw

def count_valid_mutations(ranks, adj, enforce_anomaly_free=True):
    from src.data_utils import mutate_ranks
    valid = 0
    for k in range(len(ranks)):
        if mutate_ranks(ranks, adj, k, enforce_anomaly_free) is not None:
            valid += 1
    return valid

def compute_brute_force_bfs(b_mean, dist, bidirectional=False):
    if b_mean <= 1:
        return dist + 1
    if bidirectional:
        return 2 * ((b_mean**(math.ceil(dist/2) + 1) - 1) / (b_mean - 1))
    else:
        return (b_mean**(dist + 1) - 1) / (b_mean - 1)

def replay_path(ranks, adj, path, enforce_anomaly_free=True):
    from src.data_utils import mutate_ranks, mutate_adjacency
    curr_ranks, curr_adj = ranks, adj
    for k in path:
        new_ranks = mutate_ranks(curr_ranks, curr_adj, k, enforce_anomaly_free)
        if new_ranks is None:
            return None, None
        new_adj = mutate_adjacency(curr_adj, k)
        curr_ranks, curr_adj = new_ranks, new_adj
    return curr_ranks, curr_adj

# ---------------------------------------------------------------------------
# Globals for workers
# ---------------------------------------------------------------------------
_WORKERS = {}

def _worker_init(args):
    global _WORKERS
    device = torch.device("cpu")
    
    if args.siamese or args.hybrid or args.hybrid_lca:
        _WORKERS["siamese"] = HeuristicBidirectionalPathfinder(
            model_path=args.siamese_model, hidden_channels=args.hidden_channels_siamese, device=device
        )
    if args.ar or args.hybrid or args.hybrid_lca:
        _WORKERS["ar"] = AutoregressivePathfinder(
            model_path=args.ar_model, hidden_channels=args.hidden_channels_ar, device=device
        )
    if args.hybrid:
        _WORKERS["hybrid"] = HybridBidirectionalPathfinder(
            siamese_model_path=args.siamese_model, ar_model_path=args.ar_model,
            hidden_channels_siamese=args.hidden_channels_siamese, hidden_channels_ar=args.hidden_channels_ar,
            device=device
        )
    if args.lca:
        _WORKERS["lca"] = LCAPathfinder()
    if args.hybrid_lca:
        _WORKERS["hybrid_lca"] = HybridLCAPathfinder(
            siamese_model_path=args.siamese_model, ar_model_path=args.ar_model,
            hidden_channels_siamese=args.hidden_channels_siamese, hidden_channels_ar=args.hidden_channels_ar,
            device=device
        )

def _worker_eval(task):
    (
        ranks_a, adj_a, ranks_b, adj_b, true_dist, true_path, true_path_orig, uuid, args
    ) = task
    
    enforce_anomaly_free = not args.relax_anomaly
    b_a = count_valid_mutations(ranks_a, adj_a, enforce_anomaly_free=enforce_anomaly_free)
    b_b = count_valid_mutations(ranks_b, adj_b, enforce_anomaly_free=enforce_anomaly_free)
    b_mean = (b_a + b_b) / 2.0
    
    if math.isinf(true_dist):
        bf_nodes_fwd = float("inf")
        bf_nodes_bidir = float("inf")
    else:
        bf_nodes_fwd = compute_brute_force_bfs(b_mean, int(true_dist), bidirectional=False)
        bf_nodes_bidir = compute_brute_force_bfs(b_mean, int(true_dist), bidirectional=True)
        
    dynamic_max_steps = max(args.max_steps, int(true_dist) + 2) if not math.isinf(true_dist) else args.max_steps
    
    results = {
        "input": {"ranks": ranks_a, "adjacency": adj_a},
        "output": {"ranks": ranks_b, "adjacency": adj_b},
        "true_distance": "NaN" if math.isinf(true_dist) else int(true_dist),
        "true_path": true_path_orig,
        "uuid": uuid,
        "valid_branching_factor_a": b_a,
        "valid_branching_factor_b": b_b,
        "brute_force_bfs_nodes_fwd": round(bf_nodes_fwd, 1) if bf_nodes_fwd != float('inf') else None,
        "brute_force_bfs_nodes_bidir": round(bf_nodes_bidir, 1) if bf_nodes_bidir != float('inf') else None,
    }

    # Model evaluation
    def evaluate_model(model_key, fn_call, hyperparams):
        if args.__dict__.get(model_key):
            t0 = time.time()
            try:
                res = fn_call()
                t1 = time.time()
                path = res.get("path")
                
                path_is_valid = None
                if path is not None:
                    final_ranks, final_adj = replay_path(ranks_a, adj_a, path, enforce_anomaly_free=enforce_anomaly_free)
                    path_is_valid = (final_ranks is not None and get_graph_hash(final_ranks, final_adj) == get_graph_hash(ranks_b, adj_b))
                
                results[f"{model_key}_result"] = {
                    "status": res.get("status"),
                    "predicted_path": path,
                    "predicted_distance": len(path) if path is not None else None,
                    "path_matches": (path == true_path) if res.get("status") == "success" else False,
                    "path_is_valid": path_is_valid,
                    "nodes_explored": res.get("visited_states", res.get("nodes_explored")),
                    "time_seconds": round(t1 - t0, 4),
                    "model_passes": res.get("model_passes", 0),
                    "hyperparameters": hyperparams
                }
            except Exception as e:
                results[f"{model_key}_result"] = {
                    "status": "error",
                    "error": str(e),
                    "time_seconds": round(time.time() - t0, 4),
                    "nodes_explored": 0,
                    "hyperparameters": hyperparams
                }

    evaluate_model("siamese", 
        lambda: _WORKERS["siamese"].find_path(ranks_a, adj_a, ranks_b, adj_b, max_steps=dynamic_max_steps, max_nodes=args.max_nodes, enforce_anomaly_free=enforce_anomaly_free),
        {"hidden_channels": args.hidden_channels_siamese}
    )
    evaluate_model("ar",
        lambda: _WORKERS["ar"].find_path(ranks_a, adj_a, ranks_b, adj_b, max_steps=dynamic_max_steps, beam_width=args.beam_width, enforce_anomaly_free=enforce_anomaly_free),
        {"hidden_channels": args.hidden_channels_ar, "beam_width": args.beam_width}
    )
    evaluate_model("hybrid",
        lambda: _WORKERS["hybrid"].find_path(ranks_a, adj_a, ranks_b, adj_b, max_steps=dynamic_max_steps, max_nodes=args.max_nodes, enforce_anomaly_free=enforce_anomaly_free, lambda_ar=args.lambda_ar, top_k=args.top_k),
        {"lambda_ar": args.lambda_ar, "top_k": args.top_k}
    )
    evaluate_model("lca",
        lambda: _WORKERS["lca"].find_path(ranks_a, adj_a, ranks_b, adj_b, max_steps=dynamic_max_steps, enforce_anomaly_free=enforce_anomaly_free),
        {}
    )
    evaluate_model("hybrid_lca",
        lambda: _WORKERS["hybrid_lca"].find_path(
            ranks_a, adj_a, ranks_b, adj_b, max_steps=dynamic_max_steps, max_nodes=args.max_nodes, enforce_anomaly_free=enforce_anomaly_free,
            lambda_ar=args.lambda_ar, top_k=args.top_k,
            lambda_det_cost=args.lambda_det_cost, lambda_siamese_h=args.lambda_siamese_h, lambda_lca_h=args.lambda_lca_h,
            cost_decrease=args.cost_decrease, cost_equal=args.cost_equal, cost_increase=args.cost_increase
        ),
        {
            "lambda_ar": args.lambda_ar, "top_k": args.top_k,
            "lambda_det_cost": args.lambda_det_cost, "lambda_siamese_h": args.lambda_siamese_h, "lambda_lca_h": args.lambda_lca_h,
            "cost_decrease": args.cost_decrease, "cost_equal": args.cost_equal, "cost_increase": args.cost_increase
        }
    )
    
    return results

def evaluate(args):
    if not (args.siamese or args.ar or args.hybrid or args.lca or args.hybrid_lca):
        return

    enforce_anomaly_free = not args.relax_anomaly
    if args.num_workers <= 1:
        _worker_init(args)

    ctx = mp.get_context("spawn") if args.num_workers > 1 else None

    for dataset_path in args.datasets:
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        if not os.path.exists(dataset_path):
            continue
        
        node_groups = args.nodes if args.nodes else sorted([int(d) for d in os.listdir(dataset_path) if d.isdigit() and os.path.isdir(os.path.join(dataset_path, d))])
        
        counts, sampled_pairs = stream_sample(
            dataset_path, node_groups, seed=args.seed, use_all=args.all_pairs,
            fraction=args.sample_fraction, min_sample=args.min_sample, max_sample=args.max_sample,
            unrelated_only=args.unrelated_only, num_workers=args.num_workers,
            use_cache=not args.no_cache, rebuild_cache=args.rebuild_cache, distances=args.dist,
        )
        
        dataset_out_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        unique_nodes = sorted(list(set(k[0] for k in sampled_pairs.keys())))
        file_suffix = "_unrelated" if args.unrelated_only else ""
        
        for current_nodes in unique_nodes:
            out_path = os.path.join(dataset_out_dir, f"eval_{current_nodes}n{file_suffix}.json")
            existing_data = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r") as f:
                        existing_data = json.load(f)
                except Exception:
                    pass
                    
            # Map existing uuid -> dict
            existing_map = {item.get("uuid"): item for item in existing_data}
            
            node_tasks = []
            for key in sorted(sampled_pairs.keys()):
                if key[0] != current_nodes: continue
                for item in sampled_pairs[key]:
                    if len(item) == 7:
                        ranks_a, adj_a, ranks_b, adj_b, true_dist, true_path = item[:6]
                        uuid = item[6]
                    else:
                        g_a, g_b, true_dist, true_path = item
                        ranks_a, adj_a = pyg_to_raw(g_a)
                        ranks_b, adj_b = pyg_to_raw(g_b)
                        uuid = getattr(g_a, "family_id", None)
                    
                    true_path_0idx = [p - 1 for p in true_path] if true_path else []
                    node_tasks.append((ranks_a, adj_a, ranks_b, adj_b, float(true_dist), true_path_0idx, true_path, uuid, args))
                    
            if not node_tasks: continue
            
            results_list = []
            if args.num_workers <= 1:
                for task in tqdm(node_tasks, desc=f"Evaluating {current_nodes}n"):
                    results_list.append(_worker_eval(task))
            else:
                with ctx.Pool(processes=args.num_workers, initializer=_worker_init, initargs=(args,)) as pool:
                    for res in tqdm(pool.imap_unordered(_worker_eval, node_tasks), total=len(node_tasks)):
                        results_list.append(res)
                        
            # Merge with existing
            for res in results_list:
                uuid = res["uuid"]
                if uuid in existing_map:
                    for k, v in res.items():
                        if k.endswith("_result"):
                            existing_map[uuid][k] = v
                else:
                    existing_data.append(res)
                    existing_map[uuid] = res
                    
            with open(out_path, "w") as f:
                json.dump(existing_data, f, indent=2)


def _generate_theory_plots(ds, ds_data, n_nodes, output_dir, jp_full, jp_045, meta_rev, make_pdf, baseline):
    from scripts.plot_style import InterceptJP, get_latex_name
    import os
    import numpy as np
    import seaborn as sns

    print(f"\nGenerating Theory Plots for {ds} (Nodes: {n_nodes})...")
    out_dir = os.path.join(output_dir, ds, f"Nodes_{n_nodes}")
    os.makedirs(out_dir, exist_ok=True)
    
    models = ["ar_result", "siamese_result", "hybrid_result", "lca_result", "hybrid_lca_result"]
    labels = {"ar_result": "AR", "siamese_result": "Siamese", "hybrid_result": "Hybrid", "lca_result": "LCA", "hybrid_lca_result": "Hybrid LCA"}
    
    uuids_present = set()
    for e in ds_data:
        u = e.get("uuid")
        if not u:
            for m in models:
                if isinstance(e.get(m), dict) and e[m].get("uuid"):
                    u = e[m].get("uuid")
                    break
        if u: uuids_present.add(u)
    uuids_present = sorted(list(uuids_present))
    
    if not uuids_present:
        return

    theory_colors = sns.color_palette("husl", len(uuids_present))
    
    def get_short_uuid(uuid_str, min_len=8):
        return uuid_str[:min_len] if uuid_str else "Unknown"

    def get_theory_label(u):
        root = meta_rev.get(u, "Unknown")
        return get_latex_name(root) if root != "Unknown" else f"uuid {get_short_uuid(u)}"

    baseline_label = "BFS Baseline" if baseline == 'bfs' else "LCA Baseline"

    for model_key in models:
        m_out_dir = os.path.join(out_dir, model_key)
        
        # Extract data for this model
        extracted = []
        for e in ds_data:
            if isinstance(e.get(model_key), dict):
                m_data = dict(e[model_key])
                m_data["true_distance"] = e.get("true_distance")
                if not m_data.get("uuid"):
                    m_data["uuid"] = e.get("uuid")
                
                # compute efficiency ratio for the entry
                if baseline == 'bfs':
                    base_nodes = e.get("brute_force_bfs_nodes_bidir")
                else:
                    lca_dict = e.get("lca_result")
                    base_nodes = lca_dict.get("nodes_explored") if isinstance(lca_dict, dict) else None
                
                m_nodes = m_data.get("nodes_explored")
                if base_nodes is not None and m_nodes is not None and m_nodes > 0:
                    m_data["efficiency_ratio"] = base_nodes / m_nodes
                else:
                    m_data["efficiency_ratio"] = None
                    
                extracted.append(m_data)
                
        if not extracted: continue
        os.makedirs(m_out_dir, exist_ok=True)
        
        theory_data = {}
        for e in extracted:
            u = e.get("uuid")
            if not u: continue
            if u not in theory_data: theory_data[u] = []
            theory_data[u].append(e)

        for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
            if is_045 and not make_pdf:
                continue

            jp = InterceptJP(raw_jp, is_045, output_dir=output_dir, baseline_label=baseline_label, make_pdf=make_pdf)
            ncol = 2 if len(uuids_present) > 6 else 1

            dists_all = set()
            for u, data in theory_data.items():
                dists_all.update([e.get("true_distance") for e in data if e.get("true_distance") is not None])
            dists = sorted(list(dists_all))

            if not dists: continue

            # Success Rate
            fig, ax = jp.create_figure()
            has_data = False
            for i, u in enumerate(uuids_present):
                if u not in theory_data: continue
                data = theory_data[u]
                x_vals, y_vals = [], []
                for d in dists:
                    entries = [e for e in data if e.get("true_distance") == d]
                    if entries:
                        rate = sum(1 for e in entries if e.get("status") == "success") / len(entries)
                        x_vals.append(d)
                        y_vals.append(rate * 100)
                if x_vals:
                    has_data = True
                    ax.plot(x_vals, y_vals, label=get_theory_label(u), color=theory_colors[i], marker="o", linestyle="-", markersize=4)
            if has_data:
                ax.set_ylabel("Success Rate (%)")
                ax.set_xlabel("True Distance")
                ax.set_title(f"Success Rate by Theory ({labels[model_key]}, N={n_nodes})")
                jp.add_legend(ax=ax, ncol=ncol)
                jp.save(os.path.join(m_out_dir, "success_rate_by_theory"))

            # Efficiency Ratio Overlay
            fig, ax = jp.create_figure()
            has_data = False
            for i, u in enumerate(uuids_present):
                if u not in theory_data: continue
                data = theory_data[u]
                x_vals, y_vals = [], []
                for d in dists:
                    entries = [e for e in data if e.get("true_distance") == d and e.get("efficiency_ratio") is not None and e.get("status") == "success"]
                    if entries:
                        x_vals.append(d)
                        y_vals.append(np.mean([e["efficiency_ratio"] for e in entries]))
                if x_vals:
                    has_data = True
                    ax.plot(x_vals, y_vals, label=get_theory_label(u), color=theory_colors[i], marker='o', markersize=4)
            if has_data:
                ax.axhline(y=1, color='black', linestyle='--', alpha=0.6, label=baseline_label)
                ax.set_yscale("log")
                ax.set_xlabel("True Distance")
                ax.set_ylabel("Efficiency Ratio (Log)")
                ax.set_title(f"Efficiency Ratio by Theory ({labels[model_key]}, N={n_nodes})")
                jp.add_legend(ax=ax, ncol=ncol)
                jp.save(os.path.join(m_out_dir, "efficiency_ratio_by_theory"))

            # Optimality Gap Overlay
            fig, ax = jp.create_figure()
            has_data = False
            for i, u in enumerate(uuids_present):
                if u not in theory_data: continue
                data = theory_data[u]
                x_vals, y_vals = [], []
                for d in dists:
                    entries = [e for e in data if e.get("true_distance") == d and e.get("status") == "success"]
                    gaps = [e.get("predicted_distance") - d for e in entries if e.get("predicted_distance") is not None]
                    if entries and gaps:
                        x_vals.append(d)
                        y_vals.append(np.mean(gaps))
                if x_vals:
                    has_data = True
                    ax.plot(x_vals, y_vals, label=get_theory_label(u), color=theory_colors[i], marker='^', markersize=4)
            if has_data:
                ax.set_xlabel("True Distance")
                ax.set_ylabel("Mean Optimality Gap")
                ax.set_title(f"Optimality Gap by Theory ({labels[model_key]}, N={n_nodes})")
                jp.add_legend(ax=ax, ncol=ncol)
                jp.save(os.path.join(m_out_dir, "optimality_gap_by_theory"))


def analyze(args):
    from scripts.plot_style import JHEPPlot, InterceptJP
    
    all_data = []
    for root, _, files in os.walk(args.output_dir):
        for f in files:
            if f.startswith("eval_") and f.endswith(".json"):
                with open(os.path.join(root, f), "r") as json_file:
                    all_data.extend(json.load(json_file))
                    
    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data to analyze.")
        return
        
    df['nodes'] = df['input'].apply(lambda x: len(x['ranks']) if isinstance(x, dict) and 'ranks' in x else 0)
    # Reinsert dataset if not present
    if 'dataset' not in df.columns:
        df['dataset'] = "Unknown"
        
    models = ["siamese_result", "ar_result", "hybrid_result", "lca_result", "hybrid_lca_result"]
    baseline = args.baseline
    
    def get_model_metric(row, model, metric):
        if isinstance(row.get(model), dict):
            return row[model].get(metric)
        return np.nan

    # Expand model dicts
    for model in models:
        df[f'{model}_status'] = df.apply(lambda r: get_model_metric(r, model, 'status'), axis=1)
        df[f'{model}_nodes_explored'] = df.apply(lambda r: get_model_metric(r, model, 'nodes_explored'), axis=1)
        
        # Calculate efficiency ratio
        if baseline == 'bfs':
            df[f'{model}_efficiency_ratio'] = df.apply(
                lambda r: r['brute_force_bfs_nodes_bidir'] / r[f'{model}_nodes_explored'] 
                if pd.notnull(r.get('brute_force_bfs_nodes_bidir')) and pd.notnull(r[f'{model}_nodes_explored']) and r[f'{model}_nodes_explored'] > 0 else np.nan,
                axis=1
            )
        elif baseline == 'lca':
            df[f'{model}_efficiency_ratio'] = df.apply(
                lambda r: get_model_metric(r, 'lca_result', 'nodes_explored') / r[f'{model}_nodes_explored']
                if pd.notnull(get_model_metric(r, 'lca_result', 'nodes_explored')) and pd.notnull(r[f'{model}_nodes_explored']) and r[f'{model}_nodes_explored'] > 0 else np.nan,
                axis=1
            )
            
        df[f'{model}_valid_efficiency'] = np.where(df[f'{model}_status'] == 'success', df[f'{model}_efficiency_ratio'], np.nan)
        
    summary = df.groupby(['nodes']).agg(
        **{f"{m}_success_rate": (f"{m}_status", lambda x: (x == 'success').mean()) for m in models},
        **{f"{m}_efficiency_ratio_mean": (f"{m}_valid_efficiency", "mean") for m in models}
    ).reset_index()
    
    print("\n" + "="*50)
    print(f"SUMMARY STATISTICS (Baseline: {baseline})")
    print("="*50)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.output_dir, f"summary_stats_{baseline}.csv"), index=False)
    
    # Collect metadata for theory labels
    meta_rev = {}
    datasets = set(df['dataset'].dropna().unique())
    for ds in datasets:
        ds_path = os.path.join(ds, "metadata.json")
        if os.path.exists(ds_path):
            try:
                with open(ds_path, "r") as f:
                    meta = json.load(f)
                    for k, v in meta.items():
                        if isinstance(v, dict) and "uuid" in v:
                            meta_rev[v["uuid"]] = v.get("name", "Unknown")
            except Exception as e:
                print(f"Warning loading metadata from {ds}: {e}")

    # Plotting logic
    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception as e:
        print(f"Warning: Failed to instantiate JHEPPlot with usetex: {e}")
        jp_full = JHEPPlot(fontsize=11)
        jp_045 = JHEPPlot(fontsize=11)
        
    def get_color(model):
        colors = {'siamese_result': 'C0', 'ar_result': 'C1', 'hybrid_result': 'C2', 'lca_result': 'C3', 'hybrid_lca_result': 'C4'}
        return colors.get(model, 'k')
        
    def get_label(model):
        labels = {'siamese_result': 'Siamese', 'ar_result': 'AR', 'hybrid_result': 'Hybrid', 'lca_result': 'LCA', 'hybrid_lca_result': 'Hybrid LCA'}
        return labels.get(model, model)
    
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not args.make_pdf:
            continue
            
        baseline_label = "BFS Baseline" if baseline == 'bfs' else "LCA Baseline"
        jp = InterceptJP(raw_jp, is_045, output_dir=args.output_dir, baseline_label=baseline_label, make_pdf=args.make_pdf)
        
        # Plot Success Rate (aggregated over all datasets/nodes)
        fig, ax = jp.create_figure()
        has_plot = False
        for model in models:
            if summary[f'{model}_success_rate'].notna().any():
                ax.plot(summary['nodes'], summary[f'{model}_success_rate'], label=get_label(model), marker='o', color=get_color(model))
                has_plot = True
        if has_plot:
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Success Rate')
            jp.add_legend(ax=ax)
            jp.save(os.path.join(args.output_dir, f'success_rate_vs_nodes_{baseline}'))
        
        # Plot Efficiency Ratio (aggregated over all datasets/nodes)
        fig, ax = jp.create_figure()
        has_plot = False
        for model in models:
            if summary[f'{model}_efficiency_ratio_mean'].notna().any() and (model != 'lca_result' or baseline != 'lca'):
                ax.plot(summary['nodes'], summary[f'{model}_efficiency_ratio_mean'], label=get_label(model), marker='s', color=get_color(model))
                has_plot = True
        if has_plot:
            ax.set_yscale('log')
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel(f'Efficiency Ratio (vs {baseline.upper()})')
            jp.add_legend(ax=ax)
            jp.save(os.path.join(args.output_dir, f'efficiency_ratio_vs_nodes_{baseline}'))

    # Plot specific dataset/model/node/theory data
    for ds in datasets:
        ds_data = df[df['dataset'] == ds].to_dict('records')
        nodes_all = sorted(list(set([e["nodes"] for e in ds_data])))
        
        for n_nodes in nodes_all:
            n_data = [e for e in ds_data if e["nodes"] == n_nodes]
            _generate_theory_plots(ds, n_data, n_nodes, args.output_dir, jp_full, jp_045, meta_rev, args.make_pdf, args.baseline)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Flags
    parser.add_argument("--siamese", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--lca", action="store_true")
    parser.add_argument("--hybrid_lca", action="store_true")
    
    # Model Paths
    parser.add_argument("--siamese_model", type=str, default="best_siamese.pth")
    parser.add_argument("--ar_model", type=str, default="best_autoregressive.pth")
    
    # Hyperparams
    parser.add_argument("--hidden_channels_siamese", type=int, default=64)
    parser.add_argument("--hidden_channels_ar", type=int, default=128)
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--lambda_ar", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--lambda_det_cost", type=float, default=1.3)
    parser.add_argument("--lambda_siamese_h", type=float, default=1.8)
    parser.add_argument("--lambda_lca_h", type=float, default=0.0)
    parser.add_argument("--cost_decrease", type=float, default=0.3)
    parser.add_argument("--cost_equal", type=float, default=2.7)
    parser.add_argument("--cost_increase", type=float, default=3.1)
    
    # Evaluation Config
    parser.add_argument("--datasets", type=str, nargs="+", default=["Databases/Theories_dataset"])
    parser.add_argument("--output_dir", type=str, default="results_unified")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_nodes", type=int, default=100000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--dist", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_fraction", type=float, default=0.01)
    parser.add_argument("--min_sample", type=int, default=400)
    parser.add_argument("--max_sample", type=int, default=5000)
    parser.add_argument("--all_pairs", action="store_true")
    parser.add_argument("--unrelated_only", action="store_true")
    parser.add_argument("--relax_anomaly", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    
    # Analysis
    parser.add_argument("--make_analysis", action="store_true")
    parser.add_argument("--make_pdf", action="store_true", help="Generate .pdf and _045.pdf plots in addition to .png")
    parser.add_argument("--baseline", type=str, choices=['bfs', 'lca'], default='bfs')
    
    args = parser.parse_args()
    
    evaluate(args)
    if args.make_analysis:
        analyze(args)
