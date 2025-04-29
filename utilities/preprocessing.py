import pandas as pd
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def load_dataset_for_stgcn(window_size=12):
    """
    Load PeMSD7 traffic dataset into a StaticGraphTemporalSignal object.

    Args:
        window_size (int): Number of time steps to use as features before predicting next step. Default is 12 (representing 1 hour with 5-min intervals)

    Returns:
        StaticGraphTemporalSignal: Temporal graph data loader
    """
    # Load velocity data (speeds at sensor stations)
    velocity_df = pd.read_csv("dataset/PeMSD7_V_228.csv", header=None)
    velocity_matrix = velocity_df.values  # Shape: (288, 12672)

    # Load adjacency matrix (data structure of the graph)
    adj_df = pd.read_csv("dataset/PeMSD7_W_228.csv", header=None)
    adj_matrix = adj_df.values  # Shape: (288, 288)

    # Create edge_index and edge_weight from adjacency matrix
    edge_indices = []
    edge_weights = []

    # Convert adjacency matrix to edge_index and edge_weight format
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:  # If there is an edge, otherwise skip
                edge_indices.append([i, j])
                edge_weights.append(adj_matrix[i, j])

    edge_index = np.array(edge_indices).T  # Shape: (2, num_edges)
    edge_weight = np.array(edge_weights)  # Shape: (num_edges,)

    # Create temporal features and targets
    num_nodes = velocity_matrix.shape[1]    # 288 nodes (columns)
    num_time_steps = velocity_matrix.shape[0]  # 12672 time steps (rows, 5-min intervals representing 44 days in total)

    # We'll create sequences where we use window_size previous time steps as features
    # and predict the next time step
    features = []
    targets = []

    # For each valid time window
    for t in range(num_time_steps - window_size):
        # Features: window_size previous time steps for all nodes
        # Shape: (num_nodes, window_size)
        feature_window = velocity_matrix[t:t+window_size, :].T
        features.append(feature_window)
        
        # Target: next time step after the window for all nodes
        # Shape: (num_nodes, 1)
        target = velocity_matrix[t+window_size, :].reshape(num_nodes, 1)
        targets.append(target)

    # Convert to StaticGraphTemporalSignal
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    return dataset
