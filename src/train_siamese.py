"""
train_siamese.py

Training script for Siamese.
Implements a strict Continuous Distance Regressor for Heuristic Pipeline usage.
"""

import os
import sys
import argparse
import time
import math
import random
import numpy as np

# Add the project directory to the path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import resource

try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_limit = min(8192, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
except Exception:
    pass

from src.siamese_dataset import SiameseIterableDataset, collate_siamese
from src.model_siamese import SiameseSeiberg

# -------------------------------------------------------------
# Logs
# -------------------------------------------------------------
LOG_CSV_HEADER = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_dist_loss",
    "val_dist_mae",
    "lr",
    "curr_dist",
]


def resolve_log_path(logs_dir, is_resume):
    os.makedirs(logs_dir, exist_ok=True)
    import glob
    from datetime import datetime

    existing = sorted(glob.glob(os.path.join(logs_dir, "siamese_training_*.csv")))
    if is_resume and existing:
        return existing[-1]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(logs_dir, f"siamese_training_{stamp}.csv")


def save_logs(
    log_path, epoch, train_loss, val_loss, val_dist_loss, val_dist_mae, lr, curr_dist
):
    write_header = not os.path.exists(log_path)
    import csv

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_CSV_HEADER)
        writer.writerow(
            [epoch, train_loss, val_loss, val_dist_loss, val_dist_mae, lr, curr_dist]
        )


# -------------------------------------------------------------
# Arguments
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Train Siamese Seiberg")
    parser.add_argument(
        "--db",
        type=str,
        default="Databases/Theories_dataset",
        help="Path to database",
    )
    parser.add_argument(
        "--nodes", type=str, default="3", help="Number of nodes to train on, or 'mix'"
    )
    parser.add_argument(
        "--mix_stages",
        action="store_true",
        help="Mix batches uniformly randomly across all available node counts in one epoch",
    )
    parser.add_argument(
        "--sqrt_mix",
        action="store_true",
        help="Sample sqrt(N) batches per stage to balance dataset sizes.",
    )
    parser.add_argument(
        "--min_mix_nodes",
        type=int,
        default=0,
        help="Minimum nodes to include (0 for min available).",
    )
    parser.add_argument(
        "--max_mix_nodes",
        type=int,
        default=0,
        help="Maximum nodes to include (0 for max available).",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=2000,
        help="Cap total batches per epoch.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--dry_run", action="store_true", help="Run 1 epoch, 10 batches max"
    )
    parser.add_argument(
        "--checkpoint_siamese",
        type=str,
        default="checkpoint_siamese.pth",
        help="Path to save/resume Siamese checkpoint",
    )
    parser.add_argument(
        "--checkpoint_best",
        type=str,
        default="best_siamese.pth",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint_siamese"
    )
    parser.add_argument(
        "--clear_history", action="store_true", help="Clear metrics when resuming"
    )

    parser.add_argument(
        "--min_dist", type=int, default=1, help="Minimum mutation distance to include"
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=0,
        help="Maximum mutation distance to include (0 means no limit)",
    )

    # Curriculum Learning Config (Performance Based)
    parser.add_argument(
        "--curr_start_dist",
        type=float,
        default=2.0,
        help="Start at 2.0 to give target variance, preventing mode collapse.",
    )
    parser.add_argument("--curr_end_dist", type=float, default=15.0)
    parser.add_argument(
        "--curr_mae_threshold",
        type=float,
        default=0.25,
        help="Validation MAE required to advance",
    )
    parser.add_argument(
        "--curr_patience",
        type=int,
        default=3,
        help="Consecutive epochs required below threshold",
    )
    parser.add_argument(
        "--curr_step_dist",
        type=float,
        default=0.5,
        help="Float increment added to max distance when advancing",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning. If off, trains at all distances immediately.",
    )
    parser.add_argument(
        "--dist_node",
        action="store_true",
        help="Select distances only from theories with nodes >= distance.",
    )

    parser.add_argument("--device", type=str, default="auto")

    # LR Scheduler
    parser.add_argument(
        "--use_scheduler", action="store_true", help="Use CosineAnnealingLR"
    )
    parser.add_argument(
        "--scheduler_period",
        type=int,
        default=0,
        help="Period (T_max) for cosine scheduler. If 0, defaults to epochs.",
    )
    parser.add_argument(
        "--eta_min", type=float, default=1e-5, help="Min LR for cosine scheduler"
    )
    parser.add_argument(
        "--reset_lr_on_curr",
        action="store_true",
        help="Reset learning rate when curriculum advances",
    )
    parser.add_argument(
        "--reset_lr_decay",
        type=float,
        default=1.0,
        help="Factor to multiply base LR by at each reset (e.g. 0.8 means reset to 80%% of previous start LR)",
    )

    # Logging
    parser.add_argument("--save_logs", action="store_true", help="Save CSV logs")
    parser.add_argument(
        "--log_dir", type=str, default="logs_siamese", help="Log directory"
    )

    return parser.parse_args()


def get_max_available_dist(db_path, node_counts):
    max_dist = 0
    for n in node_counts:
        n_dir = os.path.join(db_path, str(n))
        if not os.path.exists(n_dir):
            continue
        for entry in os.listdir(n_dir):
            if entry.startswith("dist_"):
                try:
                    d = int(entry.split("_")[1])
                    if d > max_dist:
                        max_dist = d
                except:
                    pass
    return max_dist


@torch.no_grad()
def evaluate_model(
    model, loader, device, criterion_mse, max_batches=None, stage_name=""
):
    model.eval()

    total_dist_loss = 0
    total_samples = 0
    dist_mae_sum = 0

    b_idx = 0
    for data_a, data_b, dist_true, _, _ in loader:
        if max_batches is not None and b_idx >= max_batches:
            break

        data_a, data_b = data_a.to(device), data_b.to(device)
        dist_true = dist_true.to(device)
        batch_size = data_a.num_graphs

        dist_pred = model(data_a, data_b)

        loss_dist = criterion_mse(dist_pred, dist_true)

        total_dist_loss += loss_dist.item() * batch_size
        total_samples += batch_size

        dist_mae_sum += torch.abs(dist_pred - dist_true).sum().item()

        b_idx += 1
        if b_idx % 20 == 0:
            print(f"    [Val {stage_name}] Batch {b_idx}", end="\r")

    if b_idx > 0:
        print(f"    [Val {stage_name}] Completed {b_idx} batches.      ")

    metrics = {
        "loss_dist": total_dist_loss / total_samples if total_samples > 0 else 0,
        "dist_mae": dist_mae_sum / total_samples if total_samples > 0 else 0,
    }
    return metrics, total_dist_loss / total_samples


def update_plot(history, filename="plots/training_progress_siamese_v2.png"):
    if not history["train_loss"] or not history["val_loss"]:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    max_pts = 1000

    def downsample(data):
        if len(data) <= max_pts:
            return data
        return data[:: max(1, len(data) // max_pts)]

    h_plot = {k: downsample(v) for k, v in history.items()}

    fig = plt.figure(figsize=(14, 10))
    ax_loss = plt.subplot(2, 2, 1)
    ax_zoom = plt.subplot(2, 2, 2)
    ax_mae = plt.subplot(2, 2, 3)
    ax_mae_zoom = plt.subplot(2, 2, 4)

    if "train_loss" in h_plot and h_plot["train_loss"]:
        ax_loss.plot(
            h_plot["train_loss"],
            label="Train Loss",
            alpha=0.9,
            color="purple",
            linewidth=2,
        )
        ax_zoom.plot(
            h_plot["train_loss"],
            label="Train Loss",
            alpha=0.9,
            color="purple",
            linewidth=2,
        )

    if "val_loss" in h_plot and h_plot["val_loss"]:
        v_x = [
            (i + 1) * max(1, len(h_plot["train_loss"]) // len(h_plot["val_loss"]))
            for i in range(len(h_plot["val_loss"]))
        ]
        ax_loss.plot(
            v_x, h_plot["val_loss"], "o-", label="Val Loss", color="red", linewidth=2
        )
        ax_zoom.plot(
            v_x, h_plot["val_loss"], "o-", label="Val Loss", color="red", linewidth=2
        )

    ax_loss.set_title("Distance Regressor Loss")
    ax_loss.set_xlabel("Batches")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.grid(True)
    if ax_loss.get_legend_handles_labels()[0]:
        ax_loss.legend()

    ax_zoom.set_yscale("symlog", linthresh=1e-5)
    ax_zoom.set_title("Loss (Symmetrical Log)")
    ax_zoom.set_xlabel("Batches")
    ax_zoom.grid(True, which="both", linestyle="--", alpha=0.5)
    if ax_zoom.get_legend_handles_labels()[0]:
        ax_zoom.legend()

    if "val_dist_mae" in h_plot and h_plot["val_dist_mae"]:
        v_x = [
            (i + 1) * max(1, len(h_plot["train_loss"]) // len(h_plot["val_dist_mae"]))
            for i in range(len(h_plot["val_dist_mae"]))
        ]
        ax_mae.plot(
            v_x,
            h_plot["val_dist_mae"],
            "o-",
            label="Dist MAE",
            color="salmon",
            linewidth=2,
        )
        ax_mae_zoom.plot(
            v_x,
            h_plot["val_dist_mae"],
            "o-",
            label="Dist MAE",
            color="salmon",
            linewidth=2,
        )
    ax_mae.set_title("Prediction Error MAE")
    ax_mae.set_xlabel("Batches")
    ax_mae.set_ylabel("MAE Distance")
    ax_mae.set_ylim(bottom=0)
    ax_mae.grid(True)
    if ax_mae.get_legend_handles_labels()[0]:
        ax_mae.legend()

    ax_mae_zoom.set_yscale("symlog", linthresh=1e-2)
    ax_mae_zoom.set_title("Prediction Error MAE (Symmetrical Log)")
    ax_mae_zoom.set_xlabel("Batches")
    ax_mae_zoom.grid(True, which="both", linestyle="--", alpha=0.5)
    if ax_mae_zoom.get_legend_handles_labels()[0]:
        ax_mae_zoom.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.device in ["auto", "cuda"]
        else (
            "mps"
            if torch.backends.mps.is_available() and args.device in ["auto", "mps"]
            else "cpu"
        )
    )
    if args.device not in ["auto", "cuda", "mps"]:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    is_mixing = args.nodes == "mix" or args.mix_stages

    def get_loader_for(n, d, split):
        path_nd = os.path.join(args.db, str(n), f"dist_{int(d)}")
        if not os.path.exists(path_nd):
            return None
        ds = SiameseIterableDataset(path_nd, split=split)

        # When mixing, we hold up to 30 dataloaders in memory simultaneously.
        # If args.num_workers > 0, this spawns 30 * num_workers processes, deadlocking macOS.
        actual_workers = 0 if is_mixing else args.num_workers

        return DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=collate_siamese,
            num_workers=actual_workers,
            pin_memory=(device.type == "cuda"),
        )

    available_stages = []

    if is_mixing:
        all_nodes = sorted(
            [int(entry) for entry in os.listdir(args.db) if entry.isdigit()]
        )
        if not all_nodes:
            all_nodes = [3]  # fallback
        min_available = min(all_nodes)
        max_available = max(all_nodes)

        actual_min_mix = args.min_mix_nodes if args.min_mix_nodes > 0 else min_available
        actual_max_mix = args.max_mix_nodes if args.max_mix_nodes > 0 else max_available

        for n_nodes in all_nodes:
            if actual_min_mix <= n_nodes <= actual_max_mix:
                available_stages.append(n_nodes)

        stage_batch_probs = {}
        total_sqrt_weight = 0.0

        for n in available_stages:
            stage_path = os.path.join(args.db, str(n), "dist_1")
            train_dir = os.path.join(stage_path, "train")

            size_n = 5000
            if os.path.exists(train_dir):
                size_n = max(
                    len([f for f in os.listdir(train_dir) if f.endswith(".pt")]) * 5000,
                    100,
                )

            w = math.sqrt(size_n) if args.sqrt_mix else 1.0
            stage_batch_probs[n] = w
            total_sqrt_weight += w

        stage_quotas = {}
        for n in available_stages:
            fraction = stage_batch_probs[n] / total_sqrt_weight
            stage_quotas[n] = max(1, int(fraction * args.max_batches_per_epoch))

    else:
        n_choice = int(args.nodes)
        available_stages = [n_choice]
        stage_quotas = {n_choice: args.max_batches_per_epoch}

    max_db_dist = get_max_available_dist(args.db, available_stages)
    if args.max_dist > 0:
        max_db_dist = min(max_db_dist, args.max_dist)
    curr_end_dist = min(args.curr_end_dist, float(max_db_dist))

    # Model
    model = SiameseSeiberg(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.gnn_layers,
        num_transformer_layers=args.transformer_layers,
        nhead=args.nhead,
    ).to(device)

    # Since we have decoupled from V2a and bypassed the Transformer bug,
    # we can train the entire continuous regressor end-to-end optimally.
    use_fused = device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=use_fused)
    criterion_mse = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dist_loss": [],
        "val_dist_mae": [],
        "lr": [],
    }

    start_epoch = 0
    best_dist_mae = float("inf")
    current_max_dist = args.curr_start_dist if args.curriculum else float(curr_end_dist)
    consecutive_success = 0

    scheduler = None
    if args.use_scheduler:
        t_max = args.scheduler_period if args.scheduler_period > 0 else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.eta_min
        )

    if args.resume:
        resume_path = (
            args.checkpoint_siamese
            if os.path.exists(args.checkpoint_siamese)
            else args.checkpoint_best
        )
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # 1. ADD THESE LINES TO OVERRIDE THE LOADED LEARNING RATE
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5

            start_epoch = ckpt["epoch"]
            if not args.clear_history and "history" in ckpt:
                for k in history:
                    if k in ckpt["history"]:
                        history[k] = ckpt["history"][k]
            if "best_dist_mae" in ckpt:
                best_dist_mae = ckpt["best_dist_mae"]
            if "current_max_dist" in ckpt:
                current_max_dist = ckpt["current_max_dist"]
            if "consecutive_success" in ckpt:
                consecutive_success = ckpt["consecutive_success"]

            # 2. COMMENT OUT OR DELETE THE SCHEDULER LOADING
            # if scheduler and "scheduler_state_dict" in ckpt:
            #     scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            # 3. DISABLE THE SCHEDULER SO IT DOESN'T REVERT YOUR OVERRIDE
            scheduler = None

    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    )
    last_dist_int = -1

    log_path = None
    if args.save_logs:
        log_path = resolve_log_path(args.log_dir, args.resume)

    print("\nStarting Training (Siamese Continuous Distance Regressor)...")

    try:
        for epoch in range(start_epoch, args.epochs):
            current_max_dist = min(current_max_dist, curr_end_dist)

            # Snap to clean value to prevent floating-point drift from repeated += step
            current_max_dist = round(current_max_dist, 6)

            # Cumulative curriculum math
            frac = current_max_dist - math.floor(current_max_dist)
            if frac < 1e-9:
                # Treat as exact integer (e.g. 3.0)
                d_frontier = int(round(current_max_dist))
                active_dists = list(range(args.min_dist, d_frontier + 1))
                if len(active_dists) > 0:
                    w_each = 1.0 / len(active_dists)
                    dist_weights = {d: w_each for d in active_dists}
                else:
                    dist_weights = {}
            else:
                d_frontier = int(math.ceil(current_max_dist))
                active_dists = list(range(args.min_dist, d_frontier + 1))
                if len(active_dists) == 0:
                    dist_weights = {}
                elif len(active_dists) == 1:
                    dist_weights = {active_dists[0]: 1.0}
                else:
                    # Smooth interpolation of weights:
                    # new frontier distance grows smoothly from 0% to its uniform share
                    w_frontier = frac * (1.0 / len(active_dists))
                    w_remainder = (1.0 - w_frontier) / (len(active_dists) - 1)
                    dist_weights = {d: w_remainder for d in active_dists}
                    dist_weights[d_frontier] = w_frontier

            # Pad printable dict to show all distances up to max_db_dist for visibility
            printable_w = {
                f"dist_{k}": f"{dist_weights.get(k, 0)*100:.1f}%"
                for k in range(args.min_dist, max_db_dist + 1)
            }
            print(
                f"  [+] Cumulative Blend: max_dist={current_max_dist:.2f} | Mix: {printable_w}"
            )

            model.train()
            t_dist_loss = 0
            t_count_loss = 0
            t_samples = 0
            epoch_batches = 0
            start_time = time.time()
            last_log_time = time.time()
            batches_since_log = 0

            # Build iters per (n, d)
            iters = {}
            valid_stages_for_d = {}
            for d in dist_weights.keys():
                valid = [n for n in available_stages if n >= d]
                if args.dist_node and len(valid) == 0:
                    print(
                        f"  [WARNING] --dist_node: no stages satisfy n >= {d}. Falling back to all available stages for distance {d}."
                    )
                    valid_stages_for_d[d] = available_stages
                elif args.dist_node:
                    valid_stages_for_d[d] = valid
                else:
                    valid_stages_for_d[d] = available_stages

            for d, w in dist_weights.items():
                if (
                    w < 1e-9
                ):  # Skip distances with negligible weight (incl. fp residuals)
                    continue
                valid_n = valid_stages_for_d[d]

                total_quota_all = sum(stage_quotas[n] for n in available_stages)
                total_quota_valid = sum(stage_quotas[n] for n in valid_n)

                for n in valid_n:
                    loader = get_loader_for(n, d, "train")
                    if loader is not None:
                        scale_factor = (
                            total_quota_all / total_quota_valid
                            if total_quota_valid > 0
                            else 1.0
                        )
                        q = max(1, int(stage_quotas[n] * scale_factor * w))
                        iters[(n, d)] = {"iter": iter(loader), "quota": q, "done": 0}

            active_keys = list(iters.keys())
            total_target_batches = sum(v["quota"] for v in iters.values())

            while active_keys:
                weights = [
                    max(1, iters[k]["quota"] - iters[k]["done"]) for k in active_keys
                ]
                key = random.choices(active_keys, weights=weights, k=1)[0]
                st = iters[key]
                if st["done"] >= st["quota"]:
                    active_keys.remove(key)
                    continue

                try:
                    data_a, data_b, dist_true, _, _ = next(st["iter"])
                except StopIteration:
                    active_keys.remove(key)
                    continue

                data_a, data_b = data_a.to(device), data_b.to(device)
                dist_true = dist_true.to(device)
                batch_size = data_a.num_graphs

                optimizer.zero_grad()

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        dist_pred = model(data_a, data_b)
                        loss = criterion_mse(dist_pred, dist_true)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Clean FP32 branch for MPS/CPU
                    dist_pred = model(data_a, data_b)
                    loss = criterion_mse(dist_pred, dist_true)
                    loss.backward()
                    optimizer.step()

                t_dist_loss += loss.item() * batch_size
                t_samples += batch_size
                st["done"] += 1
                epoch_batches += 1
                batches_since_log += 1

                history["train_loss"].append(loss.item())

                if epoch_batches % 100 == 0:
                    bps = batches_since_log / max(time.time() - last_log_time, 0.001)
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | Batch {epoch_batches}/{total_target_batches} | Loss: {loss.item():.6e} | {bps:.2f} b/s"
                    )
                    last_log_time = time.time()
                    batches_since_log = 0
                    if epoch_batches % 500 == 0:
                        update_plot(history)

                if args.dry_run and epoch_batches >= 10:
                    active_stages = []
                    break

            avg_dl = t_dist_loss / max(1, t_samples)

            # Evaluate
            print(f"  [~] Starting Evaluation (Epoch {epoch+1})...")
            v_met = {"loss_dist": 0, "dist_mae": 0}
            mae_by_dist = {}
            count_by_dist = {}
            v_loss_sum = 0

            eval_keys = list(iters.keys())
            num_st = 0
            for k in eval_keys:  # keys are (n, d)
                n, d = k
                test_loader = get_loader_for(n, d, "test")
                if test_loader is None:
                    continue
                v_quota = max(10, min(100, iters[k]["quota"] // 4))
                run_m, run_l = evaluate_model(
                    model,
                    test_loader,
                    device,
                    criterion_mse,
                    max_batches=v_quota,
                    stage_name=f"N:{n} D:{d}",
                )
                v_loss_sum += run_l
                for metric in v_met:
                    v_met[metric] += run_m[metric]

                if d not in mae_by_dist:
                    mae_by_dist[d] = 0
                    count_by_dist[d] = 0
                mae_by_dist[d] += run_m["dist_mae"]
                count_by_dist[d] += 1

                num_st += 1

            if num_st > 0:
                for metric in v_met:
                    v_met[metric] /= num_st
                v_loss_sum /= num_st
                for d in mae_by_dist:
                    mae_by_dist[d] /= count_by_dist[d]
            else:
                num_st = 1  # fallback

            history["val_loss"].append(v_loss_sum)
            history["val_dist_loss"].append(v_met["loss_dist"])
            history["val_dist_mae"].append(v_met["dist_mae"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

            if scheduler:
                scheduler.step()

            # --- Automated Curriculum Logic ---
            consecutive_success = (
                consecutive_success + 1
                if v_met["dist_mae"] < args.curr_mae_threshold
                else 0
            )

            if consecutive_success >= args.curr_patience:
                if current_max_dist < curr_end_dist:
                    current_max_dist += args.curr_step_dist
                    print(
                        f"\n[CURRICULUM] Target MAE {args.curr_mae_threshold} achieved for {args.curr_patience} epochs! Advancing Max Dist to {current_max_dist:.2f}\n"
                    )

                    if args.reset_lr_on_curr:
                        advancements = round(
                            (current_max_dist - args.curr_start_dist)
                            / args.curr_step_dist
                        )
                        new_base_lr = args.lr * (args.reset_lr_decay**advancements)

                        for param_group in optimizer.param_groups:
                            param_group["lr"] = new_base_lr

                        if scheduler:
                            t_max = (
                                args.scheduler_period
                                if args.scheduler_period > 0
                                else args.epochs
                            )
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=t_max, eta_min=args.eta_min
                            )
                            print(
                                f"  [LR RESET] Learning rate reset to {new_base_lr:.6e} and scheduler restarted."
                            )
                        else:
                            print(
                                f"  [LR RESET] Learning rate reset to {new_base_lr:.6e}."
                            )

                consecutive_success = 0

            print(
                f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_dl:.6e} | Val Loss: {v_loss_sum:.6e} | Val Dist Loss: {v_met['loss_dist']:.6e} | Curr Dist: {current_max_dist:.1f}"
            )
            mae_by_dist_str = ", ".join(
                [f"d={d}: {mae_by_dist[d]:.3f}" for d in sorted(mae_by_dist.keys())]
            )
            print(
                f"  Dist MAE: {v_met['dist_mae']:.6e} | MAE by Dist: [{mae_by_dist_str}]"
            )
            print(
                f"  LR: {optimizer.param_groups[0]['lr']:.6e} | Time: {time.time()-start_time:.1f}s"
            )

            if args.save_logs:
                save_logs(
                    log_path,
                    epoch + 1,
                    avg_dl,
                    v_loss_sum,
                    v_met["loss_dist"],
                    v_met["dist_mae"],
                    optimizer.param_groups[0]["lr"],
                    current_max_dist,
                )

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dist_mae": best_dist_mae,
                "history": history,
                "current_max_dist": current_max_dist,
                "consecutive_success": consecutive_success,
            }
            if scheduler:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt, args.checkpoint_siamese)

            if v_met["dist_mae"] < best_dist_mae:
                best_dist_mae = v_met["dist_mae"]
                torch.save(ckpt, args.checkpoint_best)
                print(f"  [!] New best model saved (Dist MAE: {best_dist_mae:.4f})")

            update_plot(history)
            if args.dry_run:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dist_mae": best_dist_mae,
            "history": history,
            "current_max_dist": current_max_dist,
            "consecutive_success": consecutive_success,
        }
        if scheduler:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, args.checkpoint_siamese)


if __name__ == "__main__":
    main()
