import pandas as pd
import random
import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split


def load_dataset_for_stgcn(window_size=12, forecast_horizon=3):
    """
    Load PeMSD7 traffic dataset into a StaticGraphTemporalSignal object.

    Args:
        window_size (int): Number of time steps to use as features before predicting next step. Default is 12 (representing 1 hour with 5-min intervals)
        forecast_horizon (int): Number of time steps to predict. Default is 3 (i.e., in the next 15 minutes)

    Returns:
        StaticGraphTemporalSignal: Temporal graph data loader
    """
    # Load velocity data (speeds at sensor stations)
    velocity_df = pd.read_csv("dataset/PeMSD7_V_228.csv", header=None)
    velocity_matrix = velocity_df.values  # Shape: (12672, 228)

    # Load adjacency matrix (data structure of the graph)
    adj_df = pd.read_csv("dataset/PeMSD7_W_228.csv", header=None)
    adj_matrix = adj_df.values  # Shape: (228, 228)

    # Create edge_index and edge_weight from adjacency matrix
    edge_indices = []
    edge_weights = []

    # Convert adjacency matrix to edge_index and edge_weight format
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            edge_indices.append([i, j])
            edge_weights.append(adj_matrix[i, j])

    edge_index = np.array(edge_indices).T  # Shape: (2, num_edges)
    edge_weight = np.array(edge_weights)  # Shape: (num_edges,)

    # Compute similarity 
    similarity_weights = 1 / (edge_weight + 1e-5)

    # Normalize the weights
    similarity_weights = similarity_weights / np.sum(similarity_weights)

    # Create temporal features and targets
    num_nodes = velocity_matrix.shape[1]  # 228 nodes (columns)
    num_time_steps = velocity_matrix.shape[0]  # 12672 time steps (rows, 5-min intervals representing 44 days in total)

    # We'll create sequences where we use window_size previous time steps as features
    # and predict the next time step
    features = []
    targets = []

    # For each valid time window
    for t in range(num_time_steps - window_size - forecast_horizon + 1):
        # Features: window_size previous time steps for all nodes
        # Shape: (num_nodes, window_size)
        feature_window = velocity_matrix[t : t + window_size, :].T
        features.append(feature_window)

        # Target: next time step after the window for all nodes
        # Shape: (num_nodes, 1)
        target = velocity_matrix[t + window_size + forecast_horizon - 1, :].reshape(num_nodes, 1)
        targets.append(target)

    # Convert to StaticGraphTemporalSignal
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    return dataset


# This is DEPRECATED, but kept for reference
def split_dataset(dataset, train_ratio = 0.8, validation_split = 0.5):
    """
    Split the dataset into training and testing sets.

    Args:
        dataset (StaticGraphTemporalSignal): The dataset to split.
        train_ratio (float): The ratio of the dataset to use for training.
        validation_split (float): The ratio of the test set to use for validation.

    Returns:
        Tuple[StaticGraphTemporalSignal, StaticGraphTemporalSignal, StaticGraphTemporalSignal]: The training, testing and validation datasets.
    """

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = train_ratio)

    # Further split the test dataset into testing and validation sets
    test_dataset, val_dataset = temporal_signal_split(test_dataset, train_ratio = validation_split)

    # Return the datasets
    return train_dataset, test_dataset, val_dataset


def train_test_split(dataset):
    """
    Split the dataset into exactly half, since training will be every weekday in May
    and testing will be every weekday in June.

    Args:
        dataset (StaticGraphTemporalSignal): The dataset to split.

    Returns:
        Tuple[StaticGraphTemporalSignal, StaticGraphTemporalSignal]: The training and testing datasets.
    """

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = 0.5)

    # Return the datasets
    return train_dataset, test_dataset


def subset_data(dataset, subset_ratio):
    """
    Make a subset of the dataset for testing purposes.
    
    Args:
        dataset (StaticGraphTemporalSignal): The dataset to subset.
        subset_ratio (float): The ratio of the dataset to use for the subset.

    Returns:
        StaticGraphTemporalSignal: The subset of the dataset.
    """

    # Split the dataset into subset and remaining
    subset_dataset, remaining_dataset = temporal_signal_split(dataset, train_ratio = subset_ratio)

    # Return the subset dataset
    return subset_dataset


def train_test_subset(dataset, subset_ratio):
    """
    Split the dataset into training and testing sets, and make a subset of both sets for testing purposes.
    
    Args:
        dataset (StaticGraphTemporalSignal): The dataset to split and subset.
        subset_ratio (float): The ratio of the dataset to use for the subset.

    Returns:
        Tuple[StaticGraphTemporalSignal, StaticGraphTemporalSignal]: The training and testing datasets, after subsetting.
    """

    # Use the train_test_split function to split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(dataset)

    # Use the subset_data function to make a subset of both the training and testing sets
    train_subset = subset_data(train_dataset, subset_ratio)
    test_subset = subset_data(test_dataset, subset_ratio)

    # Return the training and testing subsets
    return train_subset, test_subset


def shuffle_dataset(dataset, randomisation_offset = 1):
    """
    Shuffle the dataset, which is likely going to be the training subset.
    
    Args:
        dataset (StaticGraphTemporalSignal): The dataset to shuffle.
        randomisation_offset (int): The offset to use for randomisation. Use the epoch number to shuffle the dataset differently each time.

    Returns:
        StaticGraphTemporalSignal: The shuffled dataset.
    """

    # Extract all of the StaticGraphTemporalSignal's attributes (copying features and targets to avoid modifying the original dataset)
    edge_index = dataset.edge_index
    edge_weight = dataset.edge_weight
    features = dataset.features.copy()
    targets = dataset.targets.copy()

    # Special number
    special_number = 6122003

    # Random seed so both the features and targets are shuffled in the same way
    np.random.seed(special_number + randomisation_offset)

    # Convert features and targets to numpy arrays
    features = np.array(features)
    targets = np.array(targets)

    # Generate a random permutation of the indices
    permutation = np.random.permutation(len(features))

    # Shuffle the features and targets using the permutation
    features = features[permutation]
    targets = targets[permutation]

    # Reconstruct the dataset
    shuffled_dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    # Print a few examples of the shuffled dataset (commented out for debugging)
    # print("Shuffled dataset examples:")
    # for i in range(6):
    #     print(f"Feature {i}: {shuffled_dataset.features[i][0]}")
    #     print(f"Target {i}: {shuffled_dataset.targets[i][0]}")
    #     print()

    # Return the shuffled dataset
    return shuffled_dataset


def get_all_velocity_data():
    """
    Load the velocity data from the PeMSD7 dataset, and return it as a numpy array.

    Returns:
        torch.Tensor: The velocity data as a 1D tensor.
    """

    # Load the velocity data
    velocity_df = pd.read_csv("dataset/PeMSD7_V_228.csv", header=None)
    velocity_matrix = velocity_df.values  # Shape: (12672, 228)

    # Flatten the velocity matrix to a 1D array
    velocity_matrix = velocity_matrix.flatten()

    # Conver to a tensor and return
    return torch.tensor(velocity_matrix, dtype=torch.float32)
