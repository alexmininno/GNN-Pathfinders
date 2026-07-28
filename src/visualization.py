import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def denorm_rank(val, mean, std):
    """Denormalize a rank value."""
    return int(round((val * std) + mean))


def draw_quiver(adj, ranks, ax, title, dual_idx=-1, pos=None):
    """
    Draw a single Quiver graph using NetworkX.
    """
    G = nx.DiGraph()
    rows, cols = adj.shape
    G.add_nodes_from(range(rows))

    # Edges
    edge_labels = {}
    for i in range(rows):
        for j in range(cols):
            weight = adj[i, j]
            if weight > 0:
                G.add_edge(i, j)
                if weight > 1:
                    edge_labels[(i, j)] = str(int(weight))

    if pos is None:
        pos = nx.circular_layout(G)

    # Colors
    node_colors = []
    for i in range(rows):
        if i == dual_idx:
            node_colors.append("salmon")  # Red-ish for dualized node
        else:
            node_colors.append("skyblue")

    # Labels (Ranks)
    labels = {i: str(ranks[i]) for i in range(rows)}

    # Draw
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_weight="bold")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="-|>",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        edge_color="gray",
    )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax, font_color="red"
    )

    ax.set_title(title)
    ax.axis("off")
    return pos


def visualize_example(
    batch,
    pred_ranks,
    pred_adj,
    title_prefix,
    target_mean,
    target_std,
    rank_mean,
    rank_std,
    normalization="zscore",
):
    """
    Visualize Input, True Output, and Predicted Output for a single graph.

    Robustness:
    - If a batch of graphs is passed (from DataLoader), this function
      automatically extracts the first graph (batch_index == 0) to prevent
      shape mismatch errors.
    """
    # Handle Batch object (extract first graph)
    if hasattr(batch, "batch") and batch.batch is not None and batch.batch.max() > 0:
        print("Batch detected in visualization. Extracting the first graph...")
        mask = batch.batch == 0
        num_nodes = mask.sum().item()

        # Slice inputs (x, y_rank, y_adj)
        # We create a dummy object to hold sliced data
        class SimpleData:
            pass

        single = SimpleData()
        single.x = batch.x[mask]
        single.num_nodes = num_nodes

        # Slice predictions
        pred_ranks = pred_ranks[mask]
        adj_size = num_nodes * num_nodes
        pred_adj = pred_adj[:adj_size]

        # Slice targets
        if hasattr(batch, "y_rank"):
            single.y_rank = batch.y_rank[mask]
        if hasattr(batch, "y_adj"):
            single.y_adj = batch.y_adj[:adj_size]

        # Slice Edges (for Input Quiver)
        # Filter edges where both source and dest are < num_nodes
        # (Assuming graph 0 nodes are 0..N-1)
        e_mask = (batch.edge_index[0] < num_nodes) & (batch.edge_index[1] < num_nodes)
        single.edge_index = batch.edge_index[:, e_mask]
        if batch.edge_attr is not None:
            single.edge_attr = batch.edge_attr[e_mask]
        else:
            single.edge_attr = None

        batch = single
    else:
        num_nodes = batch.num_nodes

    # --- 1. Extract Data ---
    # Input Adjacency
    input_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    src_nodes = edge_index[0].cpu().numpy()
    dst_nodes = edge_index[1].cpu().numpy()

    # Weight logic
    if edge_attr is not None and edge_attr.shape[1] > 0:
        weights = edge_attr[:, 0].cpu().numpy()
    else:
        weights = np.ones(len(src_nodes))

    for u, v, w in zip(src_nodes, dst_nodes, weights):
        input_adj[u, v] = int(w)

    # Input Ranks & Dualized Node
    raw_x = batch.x
    input_ranks_raw = raw_x[:, 0].cpu().numpy()

    # Check if dual flag exists (column 1)
    if raw_x.shape[1] > 1:
        dual_flags = raw_x[:, 1].cpu().numpy()
        dual_node_idx = np.where(dual_flags >= 0.9)[0]
        dual_node_idx = dual_node_idx[0] if len(dual_node_idx) > 0 else -1
    else:
        dual_node_idx = -1

    if normalization == "zscore":
        input_ranks = [denorm_rank(r, rank_mean, rank_std) for r in input_ranks_raw]
    elif normalization == "log":
        input_ranks = [int(round(np.expm1(r))) for r in input_ranks_raw]
    else:
        input_ranks = [int(r) for r in input_ranks_raw]

    # True Output
    if normalization == "log":
        # Edges are log-normalized in this mode
        true_adj_raw = batch.y_adj.cpu().numpy().reshape(num_nodes, num_nodes)
        true_adj = np.rint(np.expm1(true_adj_raw)).astype(int)
    else:
        true_adj = batch.y_adj.cpu().numpy().reshape(num_nodes, num_nodes).astype(int)

    true_ranks_norm = batch.y_rank.cpu().numpy().flatten()
    if normalization == "zscore":
        true_ranks = [denorm_rank(r, target_mean, target_std) for r in true_ranks_norm]
    elif normalization == "log":
        true_ranks = [int(round(np.expm1(r))) for r in true_ranks_norm]
    else:
        true_ranks = [int(r) for r in true_ranks_norm]

    # Predicted Output
    # Predicted Output
    pred_adj_block = pred_adj.cpu().detach().numpy().reshape(num_nodes, num_nodes)
    if normalization == "log":
        pred_adj_rounded = np.maximum(0, np.rint(np.expm1(pred_adj_block)).astype(int))
    else:
        pred_adj_rounded = np.maximum(0, np.rint(pred_adj_block).astype(int))

    pred_ranks_norm = pred_ranks.cpu().detach().numpy().flatten()
    if normalization == "zscore":
        pred_ranks_val = [
            denorm_rank(r, target_mean, target_std) for r in pred_ranks_norm
        ]
    elif normalization == "log":
        pred_ranks_val = [int(round(np.expm1(r))) for r in pred_ranks_norm]
    else:
        pred_ranks_val = [int(r) for r in pred_ranks_norm]

    # --- Plot ---
    print(f"\n{'='*20} {title_prefix} {'='*20}")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    dual_msg = f"(Red=Node {dual_node_idx})" if dual_node_idx != -1 else ""
    pos = draw_quiver(
        input_adj,
        input_ranks,
        axes[0],
        f"Input\n{dual_msg}",
        dual_idx=dual_node_idx,
    )

    draw_quiver(true_adj, true_ranks, axes[1], "True Dual", pos=pos)
    draw_quiver(pred_adj_rounded, pred_ranks_val, axes[2], "Predicted Dual", pos=pos)

    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_data_quiver(data, ax=None, title="Quiver Visualization"):
    """
    Plot a single Quiver graph from a PyG Data object.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on.
            If None, a new figure is created.
        title (str): Title for the plot.
    """
    # 1. Extract Ranks
    if hasattr(data, "x") and data.x is not None:
        ranks = data.x[:, 0].cpu().numpy()
        ranks = [int(round(r)) for r in ranks]
    else:
        # Fallback if no x features, though quiver usually implies ranks
        ranks = [0] * data.num_nodes

    # 2. Build Adjacency
    num_nodes = data.num_nodes
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_index = data.edge_index
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    if data.edge_attr is not None and data.edge_attr.numel() > 0:
        weights = data.edge_attr[:, 0].cpu().numpy()
    else:
        weights = np.ones(len(src))

    for u, v, w in zip(src, dst, weights):
        adj[u, v] = int(w)

    # 3. Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True
    else:
        show_plot = False

    draw_quiver(adj, ranks, ax, title)

    if show_plot:
        plt.show()


def plot_dgnn_metrics(history, save_path="plots/dgnn_v3_metrics.png"):
    """
    Plot DGNN V3 Tracking metrics:
    Dist Loss, Seq Loss, Dist MAE, Token Acc, Path Acc
    """
    epochs = range(1, len(history["train_dist_loss"]) + 1)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Dist Loss
    axs[0, 0].plot(epochs, history["train_dist_loss"], label="Train Dist Loss")
    axs[0, 0].plot(epochs, history["val_dist_loss"], label="Val Dist Loss")
    axs[0, 0].set_title("Distance MSE Loss")
    axs[0, 0].legend()

    # 2. Seq Loss
    axs[0, 1].plot(epochs, history["train_seq_loss"], label="Train Seq Loss")
    axs[0, 1].plot(epochs, history["val_seq_loss"], label="Val Seq Loss")
    axs[0, 1].set_title("Sequence CrossEntropy Loss")
    axs[0, 1].legend()

    # 3. Dist MAE
    axs[0, 2].plot(epochs, history["val_dist_mae"], label="Val Dist MAE", color="green")
    axs[0, 2].set_title("Distance MAE (Operations)")
    axs[0, 2].legend()

    # 4. Token Accuracy
    axs[1, 0].plot(
        epochs, history["val_token_acc"], label="Val Token Acc", color="purple"
    )
    axs[1, 0].set_title("Sequence Token Accuracy")
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].legend()

    # 5. Path Accuracy
    axs[1, 1].plot(
        epochs, history["val_path_acc"], label="Val Path Exact Match", color="orange"
    )
    axs[1, 1].set_title("Exact Path Accuracy")
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].legend()

    # 6. LR
    if "lr" in history and len(history["lr"]) > 0:
        axs[1, 2].plot(epochs, history["lr"], label="Learning Rate", color="red")
        axs[1, 2].set_title("Learning Rate")
        axs[1, 2].set_yscale("log")
        axs[1, 2].legend()
    else:
        axs[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
