# Graphs Pathfinders

This repository contains the codebase for training, evaluating, and running neural pathfinders for Seiberg duality graphs.

## Setup
To set up the environment, you can use the provided `setup_env.sh` script to create a conda environment named `seiberg-gnn`:

```bash
bash setup_env.sh
conda activate seiberg-gnn
```

## Scripts Usage
Below is the `--help` output and an example usage for each script in the repository:

### `src/train_siamese.py`

```text
usage: Train Siamese Seiberg [-h] [--db DB] [--nodes NODES] [--mix_stages]
                             [--sqrt_mix] [--min_mix_nodes MIN_MIX_NODES]
                             [--max_mix_nodes MAX_MIX_NODES]
                             [--max_batches_per_epoch MAX_BATCHES_PER_EPOCH]
                             [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                             [--lr LR] [--hidden_channels HIDDEN_CHANNELS]
                             [--gnn_layers GNN_LAYERS]
                             [--transformer_layers TRANSFORMER_LAYERS]
                             [--nhead NHEAD] [--num_workers NUM_WORKERS]
                             [--dry_run]
                             [--checkpoint_siamese CHECKPOINT_SIAMESE]
                             [--checkpoint_best CHECKPOINT_BEST] [--resume]
                             [--clear_history] [--min_dist MIN_DIST]
                             [--max_dist MAX_DIST]
                             [--curr_start_dist CURR_START_DIST]
                             [--curr_end_dist CURR_END_DIST]
                             [--curr_mae_threshold CURR_MAE_THRESHOLD]
                             [--curr_patience CURR_PATIENCE]
                             [--curr_step_dist CURR_STEP_DIST] [--curriculum]
                             [--dist_node] [--device DEVICE] [--use_scheduler]
                             [--scheduler_period SCHEDULER_PERIOD]
                             [--eta_min ETA_MIN] [--reset_lr_on_curr]
                             [--reset_lr_decay RESET_LR_DECAY] [--save_logs]
                             [--log_dir LOG_DIR]

options:
  -h, --help            show this help message and exit
  --db DB               Path to database
  --nodes NODES         Number of nodes to train on, or 'mix'
  --mix_stages          Mix batches uniformly randomly across all available
                        node counts in one epoch
  --sqrt_mix            Sample sqrt(N) batches per stage to balance dataset
                        sizes.
  --min_mix_nodes MIN_MIX_NODES
                        Minimum nodes to include (0 for min available).
  --max_mix_nodes MAX_MIX_NODES
                        Maximum nodes to include (0 for max available).
  --max_batches_per_epoch MAX_BATCHES_PER_EPOCH
                        Cap total batches per epoch.
  --epochs EPOCHS       Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --hidden_channels HIDDEN_CHANNELS
  --gnn_layers GNN_LAYERS
  --transformer_layers TRANSFORMER_LAYERS
  --nhead NHEAD
  --num_workers NUM_WORKERS
  --dry_run             Run 1 epoch, 10 batches max
  --checkpoint_siamese CHECKPOINT_SIAMESE
                        Path to save/resume Siamese checkpoint
  --checkpoint_best CHECKPOINT_BEST
  --resume              Resume from checkpoint_siamese
  --clear_history       Clear metrics when resuming
  --min_dist MIN_DIST   Minimum mutation distance to include
  --max_dist MAX_DIST   Maximum mutation distance to include (0 means no
                        limit)
  --curr_start_dist CURR_START_DIST
                        Start at 2.0 to give target variance, preventing mode
                        collapse.
  --curr_end_dist CURR_END_DIST
  --curr_mae_threshold CURR_MAE_THRESHOLD
                        Validation MAE required to advance
  --curr_patience CURR_PATIENCE
                        Consecutive epochs required below threshold
  --curr_step_dist CURR_STEP_DIST
                        Float increment added to max distance when advancing
  --curriculum          Enable curriculum learning. If off, trains at all
                        distances immediately.
  --dist_node           Select distances only from theories with nodes >=
                        distance.
  --device DEVICE
  --use_scheduler       Use CosineAnnealingLR
  --scheduler_period SCHEDULER_PERIOD
                        Period (T_max) for cosine scheduler. If 0, defaults to
                        epochs.
  --eta_min ETA_MIN     Min LR for cosine scheduler
  --reset_lr_on_curr    Reset learning rate when curriculum advances
  --reset_lr_decay RESET_LR_DECAY
                        Factor to multiply base LR by at each reset (e.g. 0.8
                        means reset to 80% of previous start LR)
  --save_logs           Save CSV logs
  --log_dir LOG_DIR     Log directory

```

### `src/predictor_siamese.py`

```text
usage: predictor_siamese.py [-h] --model_path MODEL_PATH --ranks_a RANKS_A
                            --adj_a ADJ_A --ranks_b RANKS_B --adj_b ADJ_B

Siamese Predictor CLI

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to Siamese checkpoint (.pth). Example:
                        --model_path $CHECKPOINTS_DIR/best_siamese.pth
  --ranks_a RANKS_A     Ranks for Graph A as a list. Example: --ranks_a '[1,
                        2, 3]'
  --adj_a ADJ_A         Adjacency matrix for Graph A as a list of lists.
                        Example: --adj_a '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B     Ranks for Graph B as a list. Example: --ranks_b '[1,
                        1, 3]'
  --adj_b ADJ_B         Adjacency matrix for Graph B as a list of lists.
                        Example: --adj_b '[[0,0,1],[1,0,0],[0,1,0]]'

```

### `src/predictor_autoregressive.py`

```text
usage: predictor_autoregressive.py [-h] --model MODEL --ranks_a RANKS_A
                                   --adj_a ADJ_A --ranks_b RANKS_B --adj_b
                                   ADJ_B

Autoregressive Predictor CLI

options:
  -h, --help         show this help message and exit
  --model MODEL      Path to AR checkpoint
  --ranks_a RANKS_A  e.g. '[1, 2, 3]'
  --adj_a ADJ_A      e.g. '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B  e.g. '[1, 1, 3]'
  --adj_b ADJ_B      e.g. '[[0,0,1],[1,0,0],[0,1,0]]'
```

### `pathfinders/find_path.py`

```text
usage: find_path.py [-h] [--siamese] [--ar] [--hybrid] [--lca] [--hybrid_lca]
                    [--det] --ranks_a RANKS_A --adj_a ADJ_A --ranks_b RANKS_B
                    --adj_b ADJ_B [--siamese_model SIAMESE_MODEL]
                    [--ar_model AR_MODEL] [--max_steps MAX_STEPS]
                    [--max_nodes MAX_NODES] [--beam_width BEAM_WIDTH]
                    [--relax_anomaly] [--lambda_ar LAMBDA_AR] [--top_k TOP_K]
                    [--lambda_det_cost LAMBDA_DET_COST]
                    [--lambda_siamese_h LAMBDA_SIAMESE_H]
                    [--lambda_lca_h LAMBDA_LCA_H]
                    [--cost_decrease COST_DECREASE] [--cost_equal COST_EQUAL]
                    [--cost_increase COST_INCREASE]

Unified Pathfinder for Seiberg Duality

options:
  -h, --help            show this help message and exit
  --siamese             Run Siamese A* Pathfinder
  --ar                  Run Autoregressive Beam Search Pathfinder
  --hybrid              Run Hybrid Bidirectional A* Pathfinder
  --lca                 Run Heuristic (LCA) Bidirectional A* Pathfinder
  --hybrid_lca          Run Deterministic Hybrid Bidirectional A* Pathfinder
  --det                 Run Deterministic Greedy Pathfinder
  --ranks_a RANKS_A     Ranks for Graph A. Example: --ranks_a '[1, 2, 3]'
  --adj_a ADJ_A         Adjacency matrix for Graph A. Example: --adj_a
                        '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B     Ranks for Graph B. Example: --ranks_b '[1, 1, 3]'
  --adj_b ADJ_B         Adjacency matrix for Graph B. Example: --adj_b
                        '[[0,0,1],[1,0,0],[0,1,0]]'
  --siamese_model SIAMESE_MODEL
                        Path to Siamese .pth checkpoint. Example:
                        --siamese_model $CHECKPOINTS_DIR/best_siamese.pth
  --ar_model AR_MODEL   Path to AR .pth checkpoint. Example: --ar_model
                        $CHECKPOINTS_DIR/best_auto.pth
  --max_steps MAX_STEPS
                        Maximum total search depth. Example: --max_steps 50
  --max_nodes MAX_NODES
                        Maximum nodes to explore before aborting. Example:
                        --max_nodes 100000
  --beam_width BEAM_WIDTH
                        Beam width for AR search. Example: --beam_width 3
  --relax_anomaly       Skip the anomaly-free check (N_f_in == N_f_out)
  --lambda_ar LAMBDA_AR
                        Weight for AR log-prob penalty in hybrid search.
                        Example: --lambda_ar 1.0
  --top_k TOP_K         Filter actions to only top K predicted by AR model.
                        Example: --top_k 5
  --lambda_det_cost LAMBDA_DET_COST
                        Weight for deterministic step cost in Hybrid LCA.
                        Example: --lambda_det_cost 0.5
  --lambda_siamese_h LAMBDA_SIAMESE_H
                        Weight for Siamese heuristic. Example:
                        --lambda_siamese_h 1.0
  --lambda_lca_h LAMBDA_LCA_H
                        Weight for LCA heuristic. Example: --lambda_lca_h 1.0
  --cost_decrease COST_DECREASE
                        Cost multiplier for rank decrease. Example:
                        --cost_decrease 0.5
  --cost_equal COST_EQUAL
                        Cost multiplier for equal rank. Example: --cost_equal
                        1.0
  --cost_increase COST_INCREASE
                        Cost multiplier for rank increase. Example:
                        --cost_increase 5.0
```

### `analysis/evaluate_pathfinders.py`

```text
usage: evaluate_pathfinders.py [-h] [--siamese] [--ar] [--hybrid] [--lca]
                               [--hybrid_lca] [--siamese_model SIAMESE_MODEL]
                               [--ar_model AR_MODEL] [--hidden_channels_siamese HIDDEN_CHANNELS_SIAMESE]
                               [--hidden_channels_ar HIDDEN_CHANNELS_AR] [--beam_width BEAM_WIDTH]
                               [--lambda_ar LAMBDA_AR] [--top_k TOP_K] [--lambda_det_cost LAMBDA_DET_COST]
                               [--lambda_siamese_h LAMBDA_SIAMESE_H] [--lambda_lca_h LAMBDA_LCA_H]
                               [--cost_decrease COST_DECREASE] [--cost_equal COST_EQUAL]
                               [--cost_increase COST_INCREASE] [--datasets DATASETS [DATASETS ...]]
                               [--output_dir OUTPUT_DIR] [--max_steps MAX_STEPS]
                               [--max_nodes MAX_NODES] [--num_workers NUM_WORKERS]
                               [--nodes NODES [NODES ...]] [--dist DIST [DIST ...]] [--seed SEED]
                               [--sample_fraction SAMPLE_FRACTION] [--min_sample MIN_SAMPLE]
                               [--max_sample MAX_SAMPLE] [--all_pairs] [--unrelated_only]
                               [--relax_anomaly] [--no_cache] [--rebuild_cache] [--make_analysis]
                               [--make_pdf] [--baseline {bfs,lca}]

Evaluate and Analyze Pathfinders

Execution Modes:
  -h, --help            show this help message and exit
  --siamese             Evaluate Siamese A*
  --ar                  Evaluate Autoregressive
  --hybrid              Evaluate Hybrid
  --lca                 Evaluate LCA
  --hybrid_lca          Evaluate Deterministic Hybrid
  --make_analysis       Run analysis after evaluation
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
  --baseline {bfs,lca}  Baseline to use for computing efficiency ratios. Default: bfs

Model & Dataset Options:
  --siamese_model SIAMESE_MODEL
                        Path to Siamese .pth checkpoint. Default: best_siamese.pth
  --ar_model AR_MODEL   Path to AR .pth checkpoint. Default: best_autoregressive.pth
  --datasets DATASETS   Paths to datasets to evaluate. Default: Databases/Theories_dataset
  --nodes NODES         Filter evaluation to specific node counts. Example: --nodes 4 5 6
  --dist DIST           Filter evaluation to specific distances.

Evaluation & Search Options:
  --beam_width BEAM_WIDTH
                        Beam width for AR search. Default: 3
  --max_steps MAX_STEPS
                        Maximum search depth. Default: 30
  --max_nodes MAX_NODES
                        Maximum total nodes to explore. Default: 100000
  --hidden_channels_siamese CHANNELS
                        Hidden channels for Siamese model. Default: 64
  --hidden_channels_ar CHANNELS
                        Hidden channels for AR model. Default: 128
  --lambda_ar LAMBDA_AR
                        Weight for AR log-prob penalty in hybrid. Default: 1.0
  --top_k TOP_K         Filter actions to top K predicted by AR. Default: None
  --relax_anomaly       Skip anomaly-free check (N_f_in == N_f_out)

Sampling & Multiprocessing:
  --num_workers NUM_WORKERS
                        Number of parallel processes. Default: 1
  --seed SEED           Random seed for sampling. Default: 42
  --sample_fraction SAMPLE_FRACTION
                        Fraction of pairs to sample. Default: 0.01
  --min_sample MIN_SAMPLE
                        Minimum pairs to sample per bucket. Default: 400
  --max_sample MAX_SAMPLE
                        Maximum pairs to sample per bucket. Default: 5000
  --all_pairs           Evaluate all pairs (disables sampling).
  --unrelated_only      Sample only from unrelated graphs.
  --rebuild_cache       Force rebuild of dataset cache.
  --no_cache            Disable using the dataset cache.

Deterministic Hybrid Weights:
  --lambda_det_cost LAMBDA_DET_COST
                        Weight for deterministic step cost. Default: 1.3
  --lambda_siamese_h LAMBDA_SIAMESE_H
                        Weight for Siamese GNN heuristic. Default: 1.8
  --lambda_lca_h LAMBDA_LCA_H
                        Weight for LCA rank penalty in heuristic. Default: 0.0
  --cost_decrease COST_DECREASE
                        Base cost multiplier for rank decrease. Default: 0.3
  --cost_equal COST_EQUAL
                        Base cost multiplier for equal rank. Default: 2.7
  --cost_increase COST_INCREASE
                        Base cost multiplier for rank increase. Default: 3.1

Analysis Output:
  --output_dir OUTPUT_DIR
                        Directory to save results/plots. Default: results_unified
```

### `benchmark/benchmark_nn.py`

```text
usage: benchmark_nn.py [-h] [--siamese] [--ar] --checkpoint CHECKPOINT
                       [--dataset_root DATASET_ROOT] [--output_dir OUTPUT_DIR]
                       [--nodes NODES [NODES ...]]
                       [--hidden_channels_siamese HIDDEN_CHANNELS_SIAMESE]
                       [--hidden_channels_ar HIDDEN_CHANNELS_AR]
                       [--max_pairs_per_bucket MAX_PAIRS_PER_BUCKET]
                       [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                       [--extract_embeddings_siamese]
                       [--evaluate_monotonicity_siamese]
                       [--benchmark_latency_siamese] [--only_inference_ar]
                       [--only_accuracy_ar] [--evaluate_policy_margin_ar]
                       [--make_pdf]

Unified Benchmark Neural Networks

options:
  -h, --help            show this help message and exit
  --siamese             Benchmark Siamese inference
  --ar                  Benchmark Autoregressive inference
  --checkpoint CHECKPOINT
                        Path to .pth checkpoint
  --dataset_root DATASET_ROOT
  --output_dir OUTPUT_DIR
  --nodes NODES [NODES ...]
  --hidden_channels_siamese HIDDEN_CHANNELS_SIAMESE
  --hidden_channels_ar HIDDEN_CHANNELS_AR
  --max_pairs_per_bucket MAX_PAIRS_PER_BUCKET
  --num_workers NUM_WORKERS
  --batch_size BATCH_SIZE
  --extract_embeddings_siamese
                        Extract embeddings for t-SNE visualization (Siamese
                        only)
  --evaluate_monotonicity_siamese
                        Evaluate heuristic triangle inequality (Siamese only)
  --benchmark_latency_siamese
                        Run latency benchmark only (no dataset needed, Siamese
                        only)
  --only_inference_ar   Run only hardware inference benchmark (AR only)
  --only_accuracy_ar    Run only physical accuracy benchmark (AR only)
  --evaluate_policy_margin_ar
                        Evaluate local policy margin (AR only)
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
```

### `benchmark/plot_training_logs.py`

```text
usage: plot_training_logs.py [-h] [--siamese] [--ar] [--make_pdf]
                             [--logs_dir_ar LOGS_DIR_AR]
                             [--logs_dir_siamese LOGS_DIR_SIAMESE]
                             [--output_dir OUTPUT_DIR]

Plot Training Logs

options:
  -h, --help            show this help message and exit
  --siamese             Plot Siamese logs
  --ar                  Plot Autoregressive logs
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
  --logs_dir_ar LOGS_DIR_AR
  --logs_dir_siamese LOGS_DIR_SIAMESE
  --output_dir OUTPUT_DIR
```

### `scripts/plot_style.py`

This module provides reusable JHEP-style plotting utilities for LaTeX-ready PDF figures. It acts as a centralized wrapper, utilizing the `InterceptJP` class to manage specific requirements such as baseline labeling and exporting multiple plot variations (e.g., standard `.png` along with `.pdf` formats) uniformly across benchmarking scripts.

### `src/data_utils.py`

This file is a standalone data utility module that houses essential PyTorch Geometric dataset abstractions like `SeibergData` and `SeibergChunkedDataset`. It also centralizes all core graph operations used across predictors and pathfinders (e.g., `mutate_ranks`, `mutate_adjacency`, `is_connected`, `get_graph_hash`), ensuring that the `src/` core directory is fully self-contained and free of external script dependencies.

### `scripts/generate_dataset.py`

```text
usage: generate_dataset.py [-h] [--input_db INPUT_DB]
                           [--output_dir OUTPUT_DIR] [--clear]
                           [--min_nodes MIN_NODES] [--max_nodes MAX_NODES]
                           [--nodes NODES [NODES ...]] [--bfs_depth BFS_DEPTH]
                           [--min_dist MIN_DIST] [--max_dist MAX_DIST]
                           [--dists DISTS [DISTS ...]] [--max_rank MAX_RANK]
                           [--max_arrows MAX_ARROWS] [--relax_anomaly]
                           [--chunk_size CHUNK_SIZE]
                           [--split_ratio SPLIT_RATIO]
                           [--num_workers NUM_WORKERS]
                           [--max_pairs_per_dist MAX_PAIRS_PER_DIST]

Unified Theories Dataset Generator (Fast Architecture)

options:
  -h, --help            show this help message and exit
  --input_db INPUT_DB   Path to input basic theories
  --output_dir OUTPUT_DIR
                        Output directory
  --clear               Clear existing dataset folders for the selected nodes
                        before generating.
  --min_nodes MIN_NODES
                        Min node count
  --max_nodes MAX_NODES
                        Max node count
  --nodes NODES [NODES ...]
                        Specific list of nodes to generate (overrides min/max)
  --bfs_depth BFS_DEPTH
                        Max depth for the Breadth-First-Search tree generated
                        per family.
  --min_dist MIN_DIST   Minimum pairwise distance to save.
  --max_dist MAX_DIST   Maximum pairwise distance to save (-1 for no limit,
                        pruned by bfs_depth).
  --dists DISTS [DISTS ...]
                        Specific list of pairwise distances to save (overrides
                        min/max dist).
  --max_rank MAX_RANK   Filter out theories with ranks >= this value
  --max_arrows MAX_ARROWS
                        Filter out theories with arrows >= this value
  --relax_anomaly       Skip the anomaly-free check (N_f_in == N_f_out) during
                        mutation; only positive ranks and connectivity are
                        enforced
  --chunk_size CHUNK_SIZE
                        Number of pairs per file chunk
  --split_ratio SPLIT_RATIO
                        Train split ratio
  --num_workers NUM_WORKERS
  --max_pairs_per_dist MAX_PAIRS_PER_DIST
                        Maximum number of random pairs to generate per
                        distance bucket to prevent disk exhaustion

```

