import os
import sys
import glob
import argparse
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    pass

from scripts.plot_style import JHEPPlot, InterceptJP

def plot_agnn_logs(log_path, output_dir, jp_full, jp_045, make_pdf):
    print(f"Reading AGNN log from {log_path}...")
    df = pd.read_csv(log_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. AGNN Loss
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not make_pdf: continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=make_pdf)
        fig, ax = jp.create_figure()
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='C0', linewidth=1.5)
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='C1', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('AGNN Model Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, 'agnn_loss'))

    # 2. AGNN Accuracy
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not make_pdf: continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=make_pdf)
        fig, ax = jp.create_figure()
        ax.plot(df['epoch'], df['val_top1'], label='Top-1 Accuracy', color='C0', linewidth=1.5)
        ax.plot(df['epoch'], df['val_top2'], label='Top-2 Accuracy', color='C1', linewidth=1.5)
        ax.plot(df['epoch'], df['val_top3'], label='Top-3 Accuracy', color='C2', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('AGNN Validation Accuracy')
        ax.grid(True, linestyle='--', alpha=0.6)
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, 'agnn_accuracy'))

def plot_dgnn_logs(log_path, output_dir, jp_full, jp_045, make_pdf):
    print(f"Reading DGNN log from {log_path}...")
    df = pd.read_csv(log_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DGNN Loss
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not make_pdf: continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=make_pdf)
        fig, ax = jp.create_figure()
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='C0', linewidth=1.5)
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='C1', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DGNN Model Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, 'dgnn_loss'))

    # 2. DGNN MAE
    for is_045, raw_jp in [(False, jp_full), (True, jp_045)]:
        if is_045 and not make_pdf: continue
        jp = InterceptJP(raw_jp, is_045, make_pdf=make_pdf)
        fig, ax = jp.create_figure()
        ax.plot(df['epoch'], df['val_dist_mae'], label='Validation MAE', color='C2', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title('DGNN Validation Distance MAE')
        ax.grid(True, linestyle='--', alpha=0.6)
        jp.add_legend(ax=ax)
        jp.save(os.path.join(output_dir, 'dgnn_mae'))

def get_latest_log(directory):
    log_files = glob.glob(os.path.join(directory, "*.csv"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training Logs")
    parser.add_argument("--dgnn", action="store_true", help="Plot DGNN logs")
    parser.add_argument("--agnn", action="store_true", help="Plot AGNN logs")
    parser.add_argument("--make_pdf", action="store_true", help="Generate .pdf and _045.pdf plots in addition to .png")
    parser.add_argument("--logs_dir_agnn", type=str, default="logs_agnn")
    parser.add_argument("--logs_dir_dgnn", type=str, default="logs_dgnn")
    parser.add_argument("--output_dir", type=str, default="analysis_unified/log_plots")
    
    args = parser.parse_args()
    
    if not args.dgnn and not args.agnn:
        args.dgnn = True
        args.agnn = True
        
    try:
        jp_full = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
        jp_045 = JHEPPlot(intextwidth=6.6155, usetex=True, fontsize=11)
    except Exception as e:
        print(f"Warning: Failed to instantiate JHEPPlot with usetex: {e}")
        jp_full = JHEPPlot(fontsize=11)
        jp_045 = JHEPPlot(fontsize=11)
        
    if args.dgnn:
        print("Plotting DGNN logs...")
        dgnn_log = get_latest_log(args.logs_dir_dgnn)
        if dgnn_log:
            plot_dgnn_logs(dgnn_log, args.output_dir, jp_full, jp_045, args.make_pdf)
        else:
            print(f"No logs found in {args.logs_dir_dgnn}")
            
    if args.agnn:
        print("Plotting AGNN logs...")
        agnn_log = get_latest_log(args.logs_dir_agnn)
        if agnn_log:
            plot_agnn_logs(agnn_log, args.output_dir, jp_full, jp_045, args.make_pdf)
        else:
            print(f"No logs found in {args.logs_dir_agnn}")
            
    print("Plotting completed successfully.")
