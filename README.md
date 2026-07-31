# Graphs Pathfinders

This repository contains the codebase and data generation tools for tracing Seiberg dualities, as detailed in [arXiv:2607.28268 [hep-th]](https://arxiv.org/abs/2607.28268). The project applies machine learning to the problem of establishing when two supersymmetric quiver gauge theories are dual. Mathematically, this translates to finding a sequence of quiver mutations connecting two graphs.

The code generates training datasets of quiver gauge theories originating from D-branes probing toric Calabi-Yau singularities. It includes two primary graph neural network (GNN) architectures built with transformer layers: a Distance GNN (DGNN) that estimates the minimum mutations required to connect a pair of theories, and an Adviser GNN (AGNN) that assigns probabilities to the vertices most likely to be dualized next.

These networks guide heuristic search algorithms to find paths between theories. The repository implements bidirectional A* and beam search pathfinders, treating the DGNN output as the heuristic and the AGNN output as the cost function. It also includes physics-informed approaches like the Lowest Common Ancestor (LCA) pathfinder, alongside hybrid models that combine neural network policies with deterministic rules. The provided tools allow users to train new models, reproduce benchmark metrics against breadth-first search baselines, and explore the computational complexity of different duality networks.

## Setup
To set up the environment, run the provided `setup_env.sh` script to create a conda environment named `seiberg-gnn`:

```bash
bash setup_env.sh
conda activate seiberg-gnn
```

## Script usage
Below is the `--help` output and an example command for each script in the repository. When executing scripts located in subdirectories (such as `benchmark/plot_training_logs.py`), run them from the repository root or set `PYTHONPATH=.` so internal imports resolve correctly.

### `src/train_dgnn.py`

```text
usage: Train DGNN Seiberg [-h] [--db DB] [--nodes NODES] [--mix_stages]
                          [--sqrt_mix] [--min_mix_nodes MIN_MIX_NODES]
                          [--max_mix_nodes MAX_MIX_NODES]
                          [--max_batches_per_epoch MAX_BATCHES_PER_EPOCH]
                          [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                          [--lr LR] [--hidden_channels HIDDEN_CHANNELS]
                          [--gnn_layers GNN_LAYERS]
                          [--transformer_layers TRANSFORMER_LAYERS]
                          [--nhead NHEAD] [--pe_channels PE_CHANNELS]
                          [--num_workers NUM_WORKERS] [--dry_run]
                          [--checkpoint_dgnn CHECKPOINT_DGNN]
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
                             [--eta_min ETA_MIN] [--override_lr OVERRIDE_LR]
                             [--reset_lr_on_curr]
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
  --pe_channels PE_CHANNELS
                        Positional encoding channels
  --num_workers NUM_WORKERS
  --dry_run             Run 1 epoch, 10 batches max
  --checkpoint CHECKPOINT
                        Path to save/resume DGNN checkpoint
  --checkpoint_best CHECKPOINT_BEST
  --resume              Resume from checkpoint
  --clear_history       Clear metrics when resuming
  --min_dist MIN_DIST   Minimum mutation distance to include
  --max_dist MAX_DIST   Maximum mutation distance to include (0 means no
                        limit)
  --curr_start_dist CURR_START_DIST
                        Start at 2.0 to give target variance, preventing mode
                        collapse.
  --curr_end_dist CURR_END_DIST
                        End curriculum distance (0.0 means use max_db_dist
                        limit)
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
  --override_lr OVERRIDE_LR
                        Override LR when resuming
  --reset_lr_on_curr    Reset learning rate when curriculum advances
  --reset_lr_decay RESET_LR_DECAY
                        Factor to multiply base LR by at each reset (e.g. 0.8
                        means reset to 80% of previous start LR)
  --save_logs           Save CSV logs
  --log_dir LOG_DIR     Log directory
```

**Example:**

```bash
python src/train_dgnn.py \
  --nodes mix \
  --sqrt_mix \
  --epochs 250 \
  --num_workers 5 \
  --save_logs \
  --log_dir logs_dgnn \
  --dist_node \
  --checkpoint checkpoints/checkpoint_dgnn.pth \
  --checkpoint_best checkpoints/best_dgnn.pth \
  --lr 1e-4 \
  --use_scheduler \
  --scheduler_period 250 \
  --eta_min 1e-6
```

### `src/train_agnn.py`

```text
usage: Train AGNN Seiberg GPS [-h] [--db DB] [--nodes NODES]
                                        [--dist DIST] [--curriculum]
                                        [--sqrt_mix] [--epochs EPOCHS]
                                        [--batch_size BATCH_SIZE]
                                        [--max_batches_per_epoch MAX_BATCHES_PER_EPOCH]
                                        [--max_val_batches MAX_VAL_BATCHES]
                                        [--lr LR]
                                        [--weight_decay WEIGHT_DECAY]
                                        [--hidden_channels HIDDEN_CHANNELS]
                                        [--pe_channels PE_CHANNELS]
                                        [--num_encoder_layers NUM_ENCODER_LAYERS]
                                        [--num_decoder_layers NUM_DECODER_LAYERS]
                                        [--nhead NHEAD] [--dropout DROPOUT]
                                        [--num_workers NUM_WORKERS]
                                        [--device DEVICE]
                                        [--checkpoint CHECKPOINT]
                                        [--checkpoint_best CHECKPOINT_BEST]
                                        [--resume] [--clear_history]
                                        [--dry_run] [--save_logs]
                                        [--log_dir LOG_DIR]
                                        [--curr_threshold CURR_THRESHOLD]
                                        [--curr_patience CURR_PATIENCE]
                                        [--min_dist MIN_DIST]
                                        [--max_dist MAX_DIST]
                                        [--curr_step_dist CURR_STEP_DIST]
                                        [--dist_node] [--use_scheduler]
                                        [--scheduler_period SCHEDULER_PERIOD]
                                        [--eta_min ETA_MIN]
                                        [--reset_lr_on_curr]
                                        [--reset_lr_decay RESET_LR_DECAY]

options:
  -h, --help            show this help message and exit
  --db DB
  --nodes NODES
  --dist DIST           Distances to train on (e.g. '1' or '1,2')
  --curriculum          Enable curriculum learning. If off, trains at all
                        distances immediately.
  --sqrt_mix            Sample sqrt(N) batches per stage to balance dataset
                        sizes.
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --max_batches_per_epoch MAX_BATCHES_PER_EPOCH
                        Cap the number of batches per epoch
  --max_val_batches MAX_VAL_BATCHES
                        Cap the number of validation batches
  --lr LR
  --weight_decay WEIGHT_DECAY
  --hidden_channels HIDDEN_CHANNELS
  --pe_channels PE_CHANNELS
  --num_encoder_layers NUM_ENCODER_LAYERS
  --num_decoder_layers NUM_DECODER_LAYERS
  --nhead NHEAD
  --dropout DROPOUT
  --num_workers NUM_WORKERS
  --device DEVICE
  --checkpoint CHECKPOINT
  --checkpoint_best CHECKPOINT_BEST
  --resume
  --clear_history       Clear metrics when resuming
  --dry_run             Run 1 epoch, 10 batches max
  --save_logs           Save CSV logs
  --log_dir LOG_DIR
  --curr_threshold CURR_THRESHOLD
                        Top-1 Accuracy (%) to advance distance
  --curr_patience CURR_PATIENCE
                        Epochs above threshold required to advance
  --min_dist MIN_DIST   Minimum mutation distance to include
  --max_dist MAX_DIST   Maximum mutation distance to include
  --curr_step_dist CURR_STEP_DIST
                        Increment for mutation distance
  --dist_node           Select distances only from theories with nodes >=
                        distance.
  --use_scheduler
  --scheduler_period SCHEDULER_PERIOD
  --eta_min ETA_MIN
  --reset_lr_on_curr
  --reset_lr_decay RESET_LR_DECAY
```

**Example:**

```bash
python src/train_agnn.py \
  --nodes mix \
  --sqrt_mix \
  --epochs 250 \
  --batch_size 128 \
  --num_workers 3 \
  --save_logs \
  --log_dir logs_agnn \
  --dist_node \
  --use_scheduler \
  --eta_min 1e-6 \
  --checkpoint checkpoint_agnn.pth \
  --checkpoint_best best_agnn.pth 
```

### `src/predictor_dgnn.py`

```text
usage: predictor_dgnn.py [-h] --model_path MODEL_PATH --ranks_a RANKS_A
                         --adj_a ADJ_A --ranks_b RANKS_B --adj_b ADJ_B

DGNN Predictor CLI

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to DGNN checkpoint (.pth). Example:
                        --model_path $CHECKPOINTS_DIR/best_dgnn.pth
  --ranks_a RANKS_A     Ranks for Graph A as a list. Example: --ranks_a '[1,
                        2, 3]'
  --adj_a ADJ_A         Adjacency matrix for Graph A as a list of lists.
                        Example: --adj_a '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B     Ranks for Graph B as a list. Example: --ranks_b '[1,
                        1, 3]'
  --adj_b ADJ_B         Adjacency matrix for Graph B as a list of lists.
                        Example: --adj_b '[[0,0,1],[1,0,0],[0,1,0]]'
```

**Example:**

```bash
python src/predictor_dgnn.py \
  --model_path checkpoints/best_dgnn.pth \
  --ranks_a '[5, 1, 2, 1, 3]' \
  --adj_a '[[0, 3, 0, 0, 1], [0, 0, 8, 0, 0], [3, 0, 0, 0, 0], [0, 1, 4, 0, 0], [0, 0, 1, 3, 0]]' \
  --ranks_b '[1, 1, 2, 1, 3]' \
  --adj_b '[[0, 0, 3, 0, 0], [3, 0, 0, 0, 0], [0, 1, 0, 0, 2], [0, 1, 4, 0, 0], [1, 0, 0, 3, 0]]'
```

Output:
```text
--- DGNN Pure Distance Prediction ---
Estimated Distance:   0.91
---------------------------------------------
```

### `src/predictor_agnn.py`

```text
usage: predictor_agnn.py [-h] --model MODEL --ranks_a RANKS_A
                         --adj_a ADJ_A --ranks_b RANKS_B --adj_b
                         ADJ_B

AGNN Predictor CLI

options:
  -h, --help         show this help message and exit
  --model MODEL      Path to AGNN checkpoint
  --ranks_a RANKS_A  e.g. '[1, 2, 3]'
  --adj_a ADJ_A      e.g. '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B  e.g. '[1, 1, 3]'
  --adj_b ADJ_B      e.g. '[[0,0,1],[1,0,0],[0,1,0]]'
```

**Example:**

```bash
python src/predictor_agnn.py \
  --model checkpoints/best_agnn.pth \
  --ranks_a '[5, 1, 2, 1, 3]' \
  --adj_a '[[0, 3, 0, 0, 1], [0, 0, 8, 0, 0], [3, 0, 0, 0, 0], [0, 1, 4, 0, 0], [0, 0, 1, 3, 0]]' \
  --ranks_b '[1, 1, 2, 1, 3]' \
  --adj_b '[[0, 0, 3, 0, 0], [3, 0, 0, 0, 0], [0, 1, 0, 0, 2], [0, 1, 4, 0, 0], [1, 0, 0, 3, 0]]'
```

Output:
```text
--- AGNN Prediction ---
Logits:        [6.289902687072754, -0.2608375549316406, -0.6523122787475586, 0.5631594657897949, 2.945847988128662]
Probabilities: [0.9606642723083496, 0.0013728442136198282, 0.0009281238890253007, 0.0031295265071094036, 0.0339052639901638]
-----------------------------------
```

> [!NOTE]
> The `Logits` array represents the raw, unnormalized scores assigned by the AGNN to each node in the starting graph. Each index maps to a node, and higher scores indicate a stronger prediction that mutating that specific node is the optimal next step toward the target graph. The `Probabilities` array is simply the softmax normalization of these logits.

### `pathfinders/find_path.py`

```text
usage: find_path.py [-h] [--dgnn] [--agnn] [--hybrid] [--lca] [--hybrid_lca]
                    --ranks_a RANKS_A --adj_a ADJ_A --ranks_b RANKS_B
                    --adj_b ADJ_B [--dgnn_model DGNN_MODEL]
                    [--agnn_model AGNN_MODEL] [--max_steps MAX_STEPS]
                    [--max_nodes MAX_NODES] [--beam_width BEAM_WIDTH]
                    [--relax_anomaly] [--lambda_agnn LAMBDA_AGNN] [--top_k TOP_K]
                    [--lambda_det_cost LAMBDA_DET_COST]
                    [--lambda_dgnn_h LAMBDA_DGNN_H]
                    [--lambda_lca_h LAMBDA_LCA_H]
                    [--cost_decrease COST_DECREASE] [--cost_equal COST_EQUAL]
                    [--cost_increase COST_INCREASE]

Unified Pathfinder for Seiberg Duality

options:
  -h, --help            show this help message and exit
  --dgnn                Run DGNN A* Pathfinder
  --agnn                Run AGNN Beam Search Pathfinder
  --hybrid              Run Hybrid Bidirectional A* Pathfinder
  --lca                 Run Heuristic (LCA) Bidirectional A* Pathfinder
  --hybrid_lca          Run Deterministic Hybrid Bidirectional A* Pathfinder
  --ranks_a RANKS_A     Ranks for Graph A. Example: --ranks_a '[1, 2, 3]'
  --adj_a ADJ_A         Adjacency matrix for Graph A. Example: --adj_a
                        '[[0,1,0],[0,0,1],[1,0,0]]'
  --ranks_b RANKS_B     Ranks for Graph B. Example: --ranks_b '[1, 1, 3]'
  --adj_b ADJ_B         Adjacency matrix for Graph B. Example: --adj_b
                        '[[0,0,1],[1,0,0],[0,1,0]]'
  --dgnn_model DGNN_MODEL
                        Path to DGNN .pth checkpoint. Example:
                        --dgnn_model $CHECKPOINTS_DIR/best_dgnn.pth
  --agnn_model AGNN_MODEL   Path to AGNN .pth checkpoint. Example: --agnn_model
                        $CHECKPOINTS_DIR/best_agnn.pth
  --max_steps MAX_STEPS
                        Maximum total search depth. Example: --max_steps 50
  --max_nodes MAX_NODES
                        Maximum nodes to explore before aborting. Example:
                        --max_nodes 100000
  --beam_width BEAM_WIDTH
                        Beam width for AGNN search. Example: --beam_width 3
  --relax_anomaly       Skip the anomaly-free check (N_f_in == N_f_out)
  --lambda_agnn LAMBDA_AGNN
                        Weight for AGNN log-prob penalty in hybrid search.
                        Example: --lambda_agnn 1.0
  --top_k TOP_K         Filter actions to only top K predicted by AGNN model.
                        Example: --top_k 5
  --lambda_det_cost LAMBDA_DET_COST
                        Weight for deterministic step cost in Hybrid LCA.
                        Example: --lambda_det_cost 0.5
  --lambda_dgnn_h LAMBDA_DGNN_H
                        Weight for DGNN heuristic. Example:
                        --lambda_dgnn_h 1.0
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

**Example:**

```bash
python pathfinders/find_path.py \
  --ranks_a '[5, 1, 2, 1, 3]' \
  --adj_a '[[0, 3, 0, 0, 1], [0, 0, 8, 0, 0], [3, 0, 0, 0, 0], [0, 1, 4, 0, 0], [0, 0, 1, 3, 0]]' \
  --ranks_b '[1, 1, 2, 1, 3]' \
  --adj_b '[[0, 0, 3, 0, 0], [3, 0, 0, 0, 0], [0, 1, 0, 0, 2], [0, 1, 4, 0, 0], [1, 0, 0, 3, 0]]' \
  --dgnn --agnn --hybrid --lca --hybrid_lca \
  --dgnn_model checkpoints/best_dgnn.pth \
  --agnn_model checkpoints/best_agnn.pth
```

Output:
```text
Running Find Path...
Start Graph Ranks: [5, 1, 2, 1, 3]
Target Graph Ranks: [1, 1, 2, 1, 3]
--------------------------------------------------
Running Deterministic Greedy Pathfinder (LCA)...
Result: {
  "path": [
    0
  ],
  "visited_states": 3,
  "status": "success",
  "nodes_explored": 3
}

Running Heuristic (LCA) Bidirectional A* Pathfinder...
Result: {
  "path": [
    0
  ],
  "visited_states": 3,
  "status": "success",
  "nodes_explored": 3
}

Running DGNN Bidirectional A* Pathfinder...
Result: {
  "path": [
    0
  ],
  "visited_states": 3,
  "fwd_nodes": 1,
  "bwd_nodes": 2,
  "status": "success",
  "meeting_depth_fwd": 0,
  "meeting_depth_bwd": 1,
  "initial_h_fwd": 0.91,
  "initial_h_bwd": 0.82,
  "model_passes": 1
}

Running AGNN Beam Search Pathfinder...
Result: {
  "status": "success",
  "path": [
    0
  ],
  "model_passes": 1,
  "visited_states": 2
}

Running Hybrid Bidirectional A* Pathfinder...
Result: {
  "path": [
    0
  ],
  "visited_states": 3,
  "fwd_nodes": 1,
  "bwd_nodes": 2,
  "status": "success",
  "meeting_depth_fwd": 0,
  "meeting_depth_bwd": 1,
  "initial_h_fwd": 0.91,
  "initial_h_bwd": 0.82,
  "model_passes": 1
}

Running Hybrid LCA Bidirectional A* Pathfinder...
Result: {
  "path": [
    0
  ],
  "visited_states": 3,
  "fwd_nodes": 1,
  "bwd_nodes": 2,
  "status": "success",
  "meeting_depth_fwd": 0,
  "meeting_depth_bwd": 1,
  "initial_h_fwd": 0.91,
  "initial_h_bwd": 0.82,
  "model_passes": 1
}
```

### `analysis/evaluate_pathfinders.py`

```text
usage: evaluate_pathfinders.py [-h] [--dgnn] [--agnn] [--hybrid] [--lca]
                               [--hybrid_lca] [--dgnn_model DGNN_MODEL]
                               [--agnn_model AGNN_MODEL]
                               [--hidden_channels_dgnn HIDDEN_CHANNELS_DGNN]
                               [--hidden_channels_agnn HIDDEN_CHANNELS_AGNN]
                               [--beam_width BEAM_WIDTH]
                               [--lambda_agnn LAMBDA_AGNN] [--top_k TOP_K]
                               [--lambda_det_cost LAMBDA_DET_COST]
                               [--lambda_dgnn_h LAMBDA_DGNN_H]
                               [--lambda_lca_h LAMBDA_LCA_H]
                               [--cost_decrease COST_DECREASE]
                               [--cost_equal COST_EQUAL]
                               [--cost_increase COST_INCREASE]
                               [--datasets DATASETS [DATASETS ...]]
                               [--output_dir OUTPUT_DIR]
                               [--max_steps MAX_STEPS]
                               [--max_steps_lca MAX_STEPS_LCA]
                               [--max_nodes MAX_NODES]
                               [--num_workers NUM_WORKERS]
                               [--nodes NODES [NODES ...]]
                               [--dist DIST [DIST ...]] [--seed SEED]
                               [--sample_fraction SAMPLE_FRACTION]
                               [--min_sample MIN_SAMPLE]
                               [--max_sample MAX_SAMPLE] [--all_pairs]
                               [--unrelated_only] [--relax_anomaly]
                               [--no_cache] [--rebuild_cache]
                               [--make_analysis] [--make_pdf]
                               [--baseline {bfs,lca}]

options:
  -h, --help            show this help message and exit
  --dgnn
  --agnn
  --hybrid
  --lca
  --hybrid_lca
  --dgnn_model DGNN_MODEL
  --agnn_model AGNN_MODEL
  --hidden_channels_dgnn HIDDEN_CHANNELS_DGNN
  --hidden_channels_agnn HIDDEN_CHANNELS_AGNN
  --beam_width BEAM_WIDTH
  --lambda_agnn LAMBDA_AGNN
  --top_k TOP_K
  --lambda_det_cost LAMBDA_DET_COST
  --lambda_dgnn_h LAMBDA_DGNN_H
  --lambda_lca_h LAMBDA_LCA_H
  --cost_decrease COST_DECREASE
  --cost_equal COST_EQUAL
  --cost_increase COST_INCREASE
  --datasets DATASETS [DATASETS ...]
  --output_dir OUTPUT_DIR
  --max_steps MAX_STEPS
  --max_steps_lca MAX_STEPS_LCA, --max_steps_det MAX_STEPS_LCA
                        Maximum search steps for LCA / deterministic models
  --max_nodes MAX_NODES
  --num_workers NUM_WORKERS
  --nodes NODES [NODES ...]
  --dist DIST [DIST ...]
  --seed SEED
  --sample_fraction SAMPLE_FRACTION
  --min_sample MIN_SAMPLE
  --max_sample MAX_SAMPLE
  --all_pairs
  --unrelated_only
  --relax_anomaly
  --no_cache
  --rebuild_cache
  --make_analysis
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
  --baseline {bfs,lca}
```

**Example:**

```bash
python analysis/evaluate_pathfinders.py \
  --agnn_model best_agnn.pth \
  --dgnn_model best_dgnn.pth \
  --num_workers 5 \
  --relax_anomaly \
  --min_sample 1000 \
  --output_dir results/ \
  --datasets databases/Theories_dataset/
```

### `benchmark/benchmark_nn.py`

```text
usage: benchmark_nn.py [-h] [--dgnn] [--agnn] --checkpoint CHECKPOINT
                       [--dataset_root DATASET_ROOT] [--output_dir OUTPUT_DIR]
                       [--nodes NODES [NODES ...]]
                       [--hidden_channels_dgnn HIDDEN_CHANNELS_DGNN]
                       [--hidden_channels_agnn HIDDEN_CHANNELS_AGNN]
                       [--max_pairs_per_bucket MAX_PAIRS_PER_BUCKET]
                       [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                       [--extract_embeddings_dgnn]
                       [--evaluate_monotonicity_dgnn]
                       [--benchmark_latency_dgnn]
                       [--evaluate_deterministic_benchmark_dgnn]
                       [--max_deter_steps_dgnn MAX_DETER_STEPS_DGNN]
                       [--only_inference_agnn] [--only_accuracy_agnn]
                       [--evaluate_policy_margin_agnn] [--make_pdf]

Unified Benchmark Neural Networks

options:
  -h, --help            show this help message and exit
  --dgnn                Benchmark DGNN inference
  --agnn                Benchmark AGNN inference
  --checkpoint CHECKPOINT
                        Path to .pth checkpoint
  --dataset_root DATASET_ROOT
  --output_dir OUTPUT_DIR
  --nodes NODES [NODES ...]
  --hidden_channels_dgnn HIDDEN_CHANNELS_DGNN
  --hidden_channels_agnn HIDDEN_CHANNELS_AGNN
  --max_pairs_per_bucket MAX_PAIRS_PER_BUCKET
  --num_workers NUM_WORKERS
  --batch_size BATCH_SIZE
  --extract_embeddings_dgnn
                        Extract embeddings for t-SNE visualization (DGNN
                        only)
  --evaluate_monotonicity_dgnn
                        Evaluate heuristic triangle inequality (DGNN only)
  --benchmark_latency_dgnn
                        Run latency benchmark only (no dataset needed, DGNN
                        only)
  --evaluate_deterministic_benchmark_dgnn, --evaluate_deterministic_benchmark
                        Evaluate 3-way distance benchmark and permutation
                        invariance (DGNN only)
  --max_deter_steps_dgnn MAX_DETER_STEPS_DGNN, --max_deter_steps MAX_DETER_STEPS_DGNN
                        Max steps for deterministic LCAPathfinder in 3-way
                        benchmark
  --only_inference_agnn   Run only hardware inference benchmark (AGNN only)
  --only_accuracy_agnn    Run only physical accuracy benchmark (AGNN only)
  --evaluate_policy_margin_agnn
                        Evaluate local policy margin (AGNN only)
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
```

**Example:**

```bash
python benchmark/benchmark_nn.py \
  --dgnn \
  --agnn \
  --output_dir results \
  --dataset_root databases/Theories_dataset \
  --num_workers 6 \
  --checkpoint_dgnn best_dgnn.pth \
  --checkpoint_agnn best_agnn.pth \
  --evaluate_policy_margin_agnn \
  --extract_embeddings_dgnn \
  --evaluate_monotonicity_dgnn \
  --benchmark_latency_dgnn \
  --evaluate_deterministic_benchmark_dgnn
```

### `benchmark/plot_training_logs.py`

```text
usage: plot_training_logs.py [-h] [--dgnn] [--agnn] [--make_pdf]
                             [--logs_dir_agnn LOGS_DIR_AGNN]
                             [--logs_dir_dgnn LOGS_DIR_DGNN]
                             [--output_dir OUTPUT_DIR]

Plot Training Logs

options:
  -h, --help            show this help message and exit
  --dgnn                Plot DGNN logs
  --agnn                Plot AGNN logs
  --make_pdf            Generate .pdf and _045.pdf plots in addition to .png
  --logs_dir_agnn LOGS_DIR_AGNN
  --logs_dir_dgnn LOGS_DIR_DGNN
  --output_dir OUTPUT_DIR
```

**Example:**

```bash
python benchmark/plot_training_logs.py \
  --logs_dir_agnn logs/train_logs_agnn.csv \
  --logs_dir_dgnn logs/train_logs_dgnn.csv \
  --output_dir logs_plots/
```

### `scripts/plot_style.py`

This module defines JHEP-style plotting utilities for LaTeX-ready figures. It wraps matplotlib output using the `InterceptJP` class to handle baseline labeling and export multiple file formats (including `.png` and `.pdf`) across benchmarking scripts.

### `src/data_utils.py`

This module contains PyTorch Geometric dataset abstractions, including `SeibergData` and `SeibergChunkedDataset`. It also defines core graph operations used across predictors and pathfinders (such as `mutate_ranks`, `mutate_adjacency`, `is_connected`, and `get_graph_hash`), keeping the `src/` directory self-contained without external script dependencies.

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

Unified Theories Dataset Generator (Zero-Copy Architecture - Exact Limits)

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
                        Number of multiprocessing workers
  --max_pairs_per_dist MAX_PAIRS_PER_DIST
                        Maximum number of random pairs to generate per
                        distance bucket to prevent disk exhaustion
```

**Example:**

```bash
python scripts/generate_dataset.py \
  --nodes 1 2 3 4 5 6 7 8 9 10 \
  --dists 0 1 2 3 4 5 6 7 8 9 10 11 12 \
  --bfs_depth 6 \
  --max_pairs_per_dist 10000 \
  --input_db databases/BasicTheoriesData_100.json \
  --output_dir databases/Theories_dataset/ \
  --num_workers 5
```
