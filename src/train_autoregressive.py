import os
import sys
import argparse
import time
import csv
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import resource
import math

# Add the project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.siamese_dataset import CurriculumMixedDataset, collate_autoregressive
from src.model_autoregressive import AutoregressiveGPS


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_topk_per_sample(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions per sample."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].sum(dim=0).float() * 100.0
            res.append(correct_k)
        return res


def plot_progress(history, plot_path):
    """Generates a progress plot from the history dictionary."""
    try:
        if not history["train_loss"]:
            return

        max_pts = 2000
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        train_full = history["train_loss"]
        if len(train_full) > max_pts:
            step = len(train_full) // max_pts
            train_pts = train_full[::step]
            x_train = range(0, len(train_full), step)
        else:
            train_pts = train_full
            x_train = range(len(train_full))

        ax1.plot(
            x_train, train_pts, label="Train Loss (Batch)", alpha=0.4, color="royalblue"
        )

        # Val Loss (Epoch-level, scaled to batch X-axis)
        if history["val_loss"]:
            val_pts = history["val_loss"]
            # Estimate batch position for validation points
            # If we have val_at_batch, use it, otherwise fallback to estimate
            if "val_at_batch" in history and len(history["val_at_batch"]) == len(
                val_pts
            ):
                x_val = history["val_at_batch"]
            else:
                x_val = [
                    i * (len(train_full) / max(len(val_pts), 1))
                    for i in range(1, len(val_pts) + 1)
                ]

            ax1.plot(x_val, val_pts, "o-", label="Val Loss (Epoch)", color="darkorange")

        ax1.set_title("Convergence (Loss)")
        ax1.set_xlabel("Batches")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Accuracies (Epoch-level)
        if history["val_top1"]:
            epochs = range(1, len(history["val_top1"]) + 1)
            
            if "val_dist_top1" in history:
                val_dist_top1 = history["val_dist_top1"]
                all_dists = set()
                for epoch_dists in val_dist_top1:
                    all_dists.update(epoch_dists.keys())
                
                # Plot distance-specific Top-1 lines with lower opacity
                for d in sorted(list(all_dists)):
                    d_curve = [ep_dists.get(d, None) for ep_dists in val_dist_top1]
                    valid_x = [ep for ep, val in zip(epochs, d_curve) if val is not None]
                    valid_y = [val for val in d_curve if val is not None]
                    if valid_x:
                        ax2.plot(valid_x, valid_y, ".-", alpha=0.3, label=f"Dist {d} Top-1", linewidth=1, markersize=4)

            # Average metrics (prominent)
            ax2.plot(epochs, history["val_top1"], "o-", label="Avg Top-1", color="seagreen", linewidth=2)
            ax2.plot(
                epochs,
                history["val_top2"],
                "s--",
                label="Avg Top-2",
                color="mediumseagreen",
                linewidth=2
            )
            ax2.plot(
                epochs, history["val_top3"], "^:", label="Avg Top-3", color="lightgreen", linewidth=2
            )
            
            # Make legend smaller and multi-column to handle potentially many distance labels
            ax2.legend(fontsize='x-small', loc='lower right', ncol=2)

        ax2.set_title("Validation Accuracy (%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")
        pass


def resolve_log_path(logs_dir, is_resume):
    os.makedirs(logs_dir, exist_ok=True)
    import glob
    from datetime import datetime
    existing = sorted(glob.glob(os.path.join(logs_dir, "train_*.csv")))
    if is_resume and existing:
        return existing[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(logs_dir, f"train_{stamp}.csv")


def parse_args():
    parser = argparse.ArgumentParser("Train Autoregressive Seiberg GPS")
    parser.add_argument("--db", type=str, default="Databases/Theories_dataset")
    parser.add_argument("--nodes", type=str, default="mix")
    parser.add_argument(
        "--dist",
        type=str,
        default="1",
        help="Distances to train on (e.g. '1' or '1,2')",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning. If off, trains at all distances immediately.",
    )
    parser.add_argument(
        "--sqrt_mix",
        action="store_true",
        help="Sample sqrt(N) batches per stage to balance dataset sizes.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=2000,
        help="Cap the number of batches per epoch",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=500,
        help="Cap the number of validation batches",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--pe_channels", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint_autoregressive.pth"
    )
    parser.add_argument(
        "--checkpoint_best", type=str, default="best_autoregressive.pth"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--clear_history", action="store_true", help="Clear metrics when resuming"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run 1 epoch, 10 batches max"
    )

    # Logging
    parser.add_argument("--save_logs", action="store_true", help="Save CSV logs")
    parser.add_argument("--log_dir", type=str, default="logs_autoregressive")

    # Curriculum Control
    parser.add_argument(
        "--curr_threshold",
        type=float,
        default=95.0,
        help="Top-1 Accuracy (%%) to advance distance",
    )
    parser.add_argument(
        "--curr_patience",
        type=int,
        default=3,
        help="Epochs above threshold required to advance",
    )
    parser.add_argument(
        "--min_dist", type=int, default=1, help="Minimum mutation distance to include"
    )
    parser.add_argument(
        "--max_dist", type=int, default=15, help="Maximum mutation distance to include"
    )
    parser.add_argument(
        "--curr_step_dist", type=int, default=1, help="Increment for mutation distance"
    )
    parser.add_argument(
        "--dist_node", action="store_true", help="Select distances only from theories with nodes >= distance."
    )

    # LR & Scheduler
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--scheduler_period", type=int, default=10)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--reset_lr_on_curr", action="store_true")
    parser.add_argument("--reset_lr_decay", type=float, default=1.0)

    return parser.parse_args()


def print_dataset_summary(dataset):
    """Prints a summary of the dataset distribution across node counts and distances."""
    print("\n" + "=" * 50)
    print("DATASET DISTRIBUTION SUMMARY")
    print("-" * 50)
    print(f"{'Nodes':<10} | {'Dist':<10} | {'Chunks':<10} | {'Est. Samples':<15}")
    print("-" * 50)
    total_est_samples = 0
    # dataset.chunk_map is a dict: (nodes, dist) -> list of chunk paths
    for (n, d), chunks in sorted(dataset.chunk_map.items()):
        count = len(chunks)
        est = count * 5000
        print(f"{n:<10} | {d:<10} | {count:<10} | {est:<15}")
        total_est_samples += est
    print("-" * 50)
    print(f"Total Estimated Training Samples: {total_est_samples}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device(args.device)

    # Increase file descriptor limit for macOS stability with multi-worker DataLoaders
    # File descriptor limit increase is no longer strictly required due to explicit handle management
    # but kept as a safety measure for high worker counts if desired.
    # try:
    #     soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    #     target_limit = min(hard, 10240)
    #     if soft < target_limit:
    #         resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
    #         print(f"  [System] File descriptor limit increased: {soft} -> {target_limit}")
    # except Exception as e:
    #     print(f"  [!] Warning: Could not increase file descriptor limit: {e}")

    print(f"Training on {device} (Conda Environment: seiberg-gnn)")

    # 1. Setup Dataset
    if args.nodes == "mix":
        node_list = sorted([int(n) for n in os.listdir(args.db) if n.isdigit()])
    else:
        node_list = [int(x) for x in args.nodes.split(",")]

    if args.curriculum:
        dist_list = [int(x) for x in args.dist.split(",")]
    else:
        # Get all distances across all nodes
        max_db_dist = 0
        for n in node_list:
            n_dir = os.path.join(args.db, str(n))
            if not os.path.exists(n_dir): continue
            for entry in os.listdir(n_dir):
                if entry.startswith("dist_"):
                    try:
                        d = int(entry.split("_")[1])
                        if d > max_db_dist: max_db_dist = d
                    except: pass
        max_train_dist = min(args.max_dist, max_db_dist) if args.max_dist > 0 else max_db_dist
        dist_list = list(range(args.min_dist, max_train_dist + 1))
        # Override args.dist so printouts look right
        args.dist = ",".join(map(str, dist_list))

    # Quotas: (n_nodes, distance) -> weight
    quotas = {}
    for n in node_list:
        w_n = 1.0
        if args.sqrt_mix:
            stage_path = os.path.join(args.db, str(n), "dist_1")
            train_dir = os.path.join(stage_path, "train")
            size_n = 5000
            if os.path.exists(train_dir):
                size_n = max(len([f for f in os.listdir(train_dir) if f.endswith(".pt")]) * 5000, 100)
            w_n = math.sqrt(size_n)
        for d in dist_list:
            if args.dist_node and n < d:
                continue
            quotas[(n, d)] = w_n

    train_max_yields = args.max_batches_per_epoch * args.batch_size if args.max_batches_per_epoch else None
    val_max_yields = args.max_val_batches * args.batch_size if args.max_val_batches else None

    train_dataset = CurriculumMixedDataset(
        quotas=quotas, db_path=args.db, split="train", autoregressive=True, max_yields=train_max_yields, pe_channels=args.pe_channels
    )
    val_dataset = CurriculumMixedDataset(
        quotas=quotas, db_path=args.db, split="test", autoregressive=True, max_yields=val_max_yields, pe_channels=args.pe_channels
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_autoregressive,
        persistent_workers=(args.num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_autoregressive,
        persistent_workers=(args.num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model, Loss, Optimizer
    model = AutoregressiveGPS(
        in_channels=5,  # enhanced node features: rank + in_deg + out_deg + in_flux + out_flux
        pe_channels=args.pe_channels,
        hidden_channels=args.hidden_channels,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )

    start_epoch = 0
    best_top1 = 0.0
    consecutive_success = 0
    epoch = 0
    avg_top1 = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_top1": [],
        "val_top2": [],
        "val_top3": [],
        "val_at_batch": [],
        "batches_per_epoch": 0,
        "val_dist_top1": [],
    }

    if args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)

        # Load scheduler state if it exists
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if not args.clear_history and "history" in ckpt:
            for k in history:
                if k in ckpt["history"]:
                    history[k] = ckpt["history"][k]
        if "dist" in ckpt:
            args.dist = ckpt["dist"]
            print(f"  [Curriculum] Resumed at Distance: {args.dist}")

            # Force the datasets to re-scan directories using the correct resumed distance
            train_dataset.update_distances(args.dist)
            val_dataset.update_distances(args.dist)

        if "consecutive_success" in ckpt:
            consecutive_success = ckpt["consecutive_success"]
        print(f"Resumed from epoch {start_epoch}")

    log_dir = args.log_dir
    plot_dir = "plots"
    log_path = None
    if args.save_logs:
        log_path = resolve_log_path(log_dir, args.resume)
    os.makedirs(plot_dir, exist_ok=True)

    live_plot_path = os.path.join(
        plot_dir,
        f"live_progress_{os.path.basename(args.checkpoint).replace('.pth','')}.png",
    )

    log_file = None
    log_writer = None
    if args.save_logs:
        write_header = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(
                ["epoch", "train_loss", "val_loss", "val_top1", "val_top2", "val_top3", "dist_top1"]
            )

    # Session Configuration Summary
    print("\n" + "=" * 50)
    print("SESSION CONFIGURATION")
    print("-" * 50)
    print("  [Model Architecture]")
    print(f"    - Hidden Channels:    {args.hidden_channels}")
    print(f"    - PE Channels:        {args.pe_channels}")
    print(f"    - Encoder Layers:     {args.num_encoder_layers}")
    print(f"    - Decoder Layers:     {args.num_decoder_layers}")
    print(f"    - Attention Heads:    {args.nhead}")
    print(f"    - Dropout:            {args.dropout}")
    print("\n  [Curriculum & Data]")
    print(f"    - Target Nodes:       {args.nodes}")
    print(f"    - Starting Distances: {args.dist}")
    print(f"    - Batch Size:         {args.batch_size}")
    print(
        f"    - Epoch Cap:          {args.max_batches_per_epoch if args.max_batches_per_epoch else 'None'}"
    )
    print(
        f"    - Val Cap:            {args.max_val_batches if args.max_val_batches else 'None'}"
    )
    print(f"    - Learning Rate:      {args.lr}")
    print(
        f"    - Scheduler:          {'CosineAnnealing' if args.use_scheduler else 'None'}"
    )
    print(
        f"    - Reset LR on Curr:   {args.reset_lr_on_curr} (Decay: {args.reset_lr_decay})"
    )
    print("\n  [Curriculum Control]")
    print(f"    - Curriculum Active:  {args.curriculum}")
    print(f"    - Sqrt Mix:           {args.sqrt_mix}")
    print(f"    - Accuracy Threshold: {args.curr_threshold}% Top-1")
    print(f"    - Patience:           {args.curr_patience} epochs")
    print(f"    - Max Distance:       {args.max_dist}")
    print(f"    - Step Distance:      {args.curr_step_dist}")
    print(f"    - Device:             {device}")
    print("-" * 50)

    # Dataset Distribution Summary
    print_dataset_summary(train_dataset)

    print(f"Starting training for {args.epochs - start_epoch} remaining epochs...")

    epoch = start_epoch
    epoch_finished = True
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_finished = False
            model.train()
            train_loss = 0.0

            # Curriculum Awareness Print
            print(f"\n--- [ Epoch {epoch+1}/{args.epochs} ] ---")
            print(f"  Curriculum: Nodes {args.nodes} | Distances {args.dist}")

            last_log_time = time.time()
            batches_since_log = 0

            # Determine batches per epoch (accounting for cap)
            try:
                total_train_batches = len(train_loader)
                if args.max_batches_per_epoch:
                    total_train_batches = min(
                        total_train_batches, args.max_batches_per_epoch
                    )
            except (TypeError, AttributeError):
                total_train_batches = (
                    args.max_batches_per_epoch if args.max_batches_per_epoch else 0
                )

            history["batches_per_epoch"] = total_train_batches

            for batch_idx, (batch_a, batch_b, targets, action_mask, batch_dists) in enumerate(
                train_loader
            ):
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                targets = targets.to(device)
                action_mask = (
                    action_mask.to(device) if action_mask is not None else None
                )

                optimizer.zero_grad()
                logits = model(batch_a, batch_b, action_mask=action_mask)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                train_loss += current_loss
                batches_since_log += 1
                history["train_loss"].append(current_loss)

                if (batch_idx + 1) % 100 == 0:
                    bps = batches_since_log / max(time.time() - last_log_time, 0.001)
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{total_train_batches} | Loss: {current_loss:.6e} | {bps:.2f} b/s"
                    )
                    last_log_time = time.time()
                    batches_since_log = 0

                if (batch_idx + 1) % 500 == 0:
                    plot_progress(history, live_plot_path)

                if args.dry_run and (batch_idx + 1) >= 10:
                    break

            avg_train_loss = train_loss / (batch_idx + 1)

            # Validation
            model.eval()
            from collections import defaultdict
            val_loss = 0.0
            v_top1, v_top2, v_top3 = 0.0, 0.0, 0.0
            total_val_samples = 0
            
            dist_top1 = defaultdict(float)
            dist_top2 = defaultdict(float)
            dist_top3 = defaultdict(float)
            dist_counts = defaultdict(int)

            try:
                total_val_batches = len(val_loader)
                if args.max_val_batches:
                    total_val_batches = min(total_val_batches, args.max_val_batches)
            except (TypeError, AttributeError):
                total_val_batches = args.max_val_batches if args.max_val_batches else 0

            print(f"  [~] Starting Evaluation (Epoch {epoch+1})...")
            v_idx = -1
            with torch.no_grad():
                for v_idx, (v_a, v_b, v_targets, v_mask, v_dists) in enumerate(val_loader):
                    v_a = v_a.to(device)
                    v_b = v_b.to(device)
                    v_targets = v_targets.to(device)
                    v_mask = v_mask.to(device) if v_mask is not None else None
                    v_dists = v_dists.to(device)

                    v_logits = model(v_a, v_b, action_mask=v_mask)
                    v_loss = criterion(v_logits, v_targets)
                    val_loss += v_loss.item()
                    
                    batch_sz = v_targets.size(0)
                    total_val_samples += batch_sz

                    t1_per_sample, t2_per_sample, t3_per_sample = accuracy_topk_per_sample(v_logits, v_targets, topk=(1, 2, 3))
                    
                    v_top1 += t1_per_sample.sum().item()
                    v_top2 += t2_per_sample.sum().item()
                    v_top3 += t3_per_sample.sum().item()
                    
                    for idx in range(batch_sz):
                        d = int(v_dists[idx].item())
                        dist_counts[d] += 1
                        dist_top1[d] += t1_per_sample[idx].item()
                        dist_top2[d] += t2_per_sample[idx].item()
                        dist_top3[d] += t3_per_sample[idx].item()

                    if (v_idx + 1) % 20 == 0:
                        print(
                            f"    [Val] Batch {v_idx+1}/{total_val_batches}", end="\r"
                        )

            print(f"    [Val] Completed {v_idx+1} batches.      ")

            avg_val_loss = val_loss / (v_idx + 1)
            avg_top1 = v_top1 / max(total_val_samples, 1)
            avg_top2 = v_top2 / max(total_val_samples, 1)
            avg_top3 = v_top3 / max(total_val_samples, 1)

            history["val_loss"].append(avg_val_loss)
            history["val_top1"].append(avg_top1)
            history["val_top2"].append(avg_top2)
            history["val_top3"].append(avg_top3)
            history["val_at_batch"].append(len(history["train_loss"]))

            print(
                f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6e} | Val Loss: {avg_val_loss:.6e} | Dist: {args.dist}"
            )
            print(
                f"  Top-1: {avg_top1:.2f}% | Top-2: {avg_top2:.2f}% | Top-3: {avg_top3:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            print("  --- Accuracy by Distance ---")
            epoch_dist_top1 = {}
            for d in sorted(dist_counts.keys()):
                cnt = dist_counts[d]
                if cnt > 0:
                    dt1 = dist_top1[d] / cnt
                    dt2 = dist_top2[d] / cnt
                    dt3 = dist_top3[d] / cnt
                    epoch_dist_top1[d] = dt1
                    print(f"    Dist {d:>2} (n={cnt:<4}): Top-1: {dt1:>5.2f}% | Top-2: {dt2:>5.2f}% | Top-3: {dt3:>5.2f}%")
            
            if "val_dist_top1" not in history:
                history["val_dist_top1"] = []
            history["val_dist_top1"].append(epoch_dist_top1)

            # --- Early Stopping Logic ---
            if 'best_val_loss' not in history:
                history['best_val_loss'] = float('inf')
                history['val_loss_patience'] = 0

            if avg_val_loss < history['best_val_loss']:
                history['best_val_loss'] = avg_val_loss
                history['val_loss_patience'] = 0
            else:
                history['val_loss_patience'] += 1

            if history['val_loss_patience'] >= 10:
                print(f"\n[!] Early Stopping triggered: val_loss diverged for 10 epochs. Halting curriculum stage.")
                break

            # --- Automated Curriculum Logic ---
            if args.curriculum:
                consecutive_success = (
                    0 if avg_top1 < args.curr_threshold else consecutive_success + 1
                )
                if consecutive_success >= args.curr_patience:
                    dists = [int(x) for x in args.dist.split(",")]
                    next_d = max(dists) + args.curr_step_dist
                    if next_d <= args.max_dist:
                        # Save specialized checkpoint for the distance just completed
                        current_max = max(dists)
                        d_ckpt_path = f"best_auto_d{current_max}.pth"
    
                        milestone_ckpt = {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                scheduler.state_dict() if scheduler else None
                            ),
                            "best_top1": best_top1,
                            "avg_top1": avg_top1,
                            "history": history,
                            "dist": args.dist,
                            "consecutive_success": consecutive_success,
                        }
                        torch.save(milestone_ckpt, d_ckpt_path)
                        print(f"  [Curriculum] Saved milestone checkpoint: {d_ckpt_path}")
    
                        args.dist = args.dist + f",{next_d}"
                        print(
                            f"\n[CURRICULUM ADVANCED] Top-1 > {args.curr_threshold}% for {args.curr_patience} epochs!"
                        )
                        print(f"  Advancing Curriculum to Distance: {args.dist}\n")
    
                        if args.reset_lr_on_curr:
                            advancements = (next_d - min(dists)) // args.curr_step_dist
                            new_lr = args.lr * (args.reset_lr_decay**advancements)
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = new_lr
                            if args.use_scheduler:
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                    optimizer, T_max=args.epochs, eta_min=args.eta_min
                                )
                                print(
                                    f"  [LR RESET] Learning rate reset to {new_lr:.2e} and scheduler restarted."
                                )
                            else:
                                print(f"  [LR RESET] Learning rate reset to {new_lr:.2e}.")
    
                        # Refresh dataset and loaders
                        dist_list = [int(x) for x in args.dist.split(",")]
                        quotas = {}
                        for n in node_list:
                            w_n = 1.0
                            if args.sqrt_mix:
                                stage_path = os.path.join(args.db, str(n), "dist_1")
                                train_dir = os.path.join(stage_path, "train")
                                size_n = 5000
                                if os.path.exists(train_dir):
                                    size_n = max(len([f for f in os.listdir(train_dir) if f.endswith(".pt")]) * 5000, 100)
                                w_n = math.sqrt(size_n)
                            for d in dist_list:
                                if args.dist_node and n < d:
                                    continue
                                quotas[(n, d)] = w_n
    
                        # Explicitly clean up old loaders to free file descriptors and terminate persistent workers.
                        # PyTorch workaround: explicitly kill persistent workers to prevent memory leaks without IPC overhead.
                        # This uses a private API (_iterator) which is more efficient than recreating the entire manager.
                        for loader in [train_loader, val_loader]:
                            if loader is not None:
                                iterator = getattr(loader, "_iterator", None)
                                if iterator is not None:
                                    try:
                                        iterator._shutdown_workers()
                                    except Exception:
                                        pass
    
                        if "train_loader" in locals():
                            del train_loader
                        if "val_loader" in locals():
                            del val_loader
                        import gc
    
                        gc.collect()
    
                        train_dataset = CurriculumMixedDataset(
                            quotas=quotas,
                            db_path=args.db,
                            split="train",
                            autoregressive=True,
                            max_yields=train_max_yields,
                            pe_channels=args.pe_channels,
                        )
                        train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_autoregressive,
                            persistent_workers=(args.num_workers > 0),
                            pin_memory=(device.type == "cuda"),
                        )
                        total_train_batches = (
                            len(train_loader)
                            if not args.max_batches_per_epoch
                            else min(len(train_loader), args.max_batches_per_epoch)
                        )
    
                        val_dataset = CurriculumMixedDataset(
                            quotas=quotas,
                            db_path=args.db,
                            split="test",
                            autoregressive=True,
                            max_yields=val_max_yields,
                            pe_channels=args.pe_channels,
                        )
                        val_loader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_autoregressive,
                            persistent_workers=(args.num_workers > 0),
                            pin_memory=(device.type == "cuda"),
                        )
    
                        consecutive_success = 0
                        # best_top1 is NOT reset here; it strictly tracks the maximum accuracy ever achieved across all distances.
                        # Specialized checkpoints (best_auto_dx.pth) handle distance-specific performance tracking.
    
                        # Print summary of the new distribution
                        print_dataset_summary(train_dataset)

            if scheduler:
                scheduler.step()

            # Logs
            if args.save_logs and log_writer is not None:
                dist_str = ";".join(f"{d}:{history['val_dist_top1'][-1][d]:.2f}" for d in sorted(history['val_dist_top1'][-1].keys()))
                log_writer.writerow(
                    [
                        epoch,
                        avg_train_loss,
                        avg_val_loss,
                        avg_top1,
                        avg_top2,
                        avg_top3,
                        dist_str
                    ]
                )
                log_file.flush()

            # Live Plot
            plot_progress(history, live_plot_path)

            # Checkpoints
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_top1": best_top1,
                "avg_top1": avg_top1,
                "history": history,
                "dist": args.dist,
                "consecutive_success": consecutive_success,
            }
            torch.save(ckpt, args.checkpoint)

            if avg_top1 > best_top1:
                best_top1 = avg_top1
                torch.save(ckpt, args.checkpoint_best)
                print(f"  [!] New best model saved (Top-1: {best_top1:.2f}%)")

            epoch_finished = True

            if args.dry_run:
                print("  [~] Dry run completed.")
                break

        if log_file is not None:
            log_file.close()

    except KeyboardInterrupt:
        print("\n" + "!" * 60)
        print("  [!] Keyboard Interrupt detected. Cleaning up...")
        print("!" * 60)

        if 'log_file' in locals() and log_file is not None:
            try:
                log_file.close()
            except:
                pass

        try:
            ckpt = {
                "epoch": epoch if epoch_finished else epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_top1": best_top1,
                "avg_top1": avg_top1,
                "history": history,
                "dist": args.dist,
                "consecutive_success": consecutive_success,
            }
            torch.save(ckpt, args.checkpoint)
            print(f"  [>] Progress safely saved to: {args.checkpoint}")
        except Exception as e:
            print(f"  [X] Failed to save checkpoint: {e}")

        print("\nTerminating training. Goodbye.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
