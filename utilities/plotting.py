# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: Functions for plotting and saving neural network data.

# # ========== IMPORTS ==========
import matplotlib.pyplot as plt
import os
from datetime import datetime


# # ========== PLOTTING FUNCTIONS ==========
def plot_and_save_loss(losses, num_nodes, hidden_channels, learning_rate, subset_ratio, decay, decay_step, test_number, extended_desc, folder = 'graphs', subfolder = None):
    """
    Plots training loss over epochs, then saves the plot to a specified folder.
    
    Args:
        losses (list): List of loss values for each epoch.
        num_nodes (int): Number of nodes in the graph.
        hidden_channels (int): Number of hidden channels in the model.
        learning_rate (float): Learning rate used in training.
        subset_ratio (float): Ratio of the dataset used for training.
        decay (float): Decay rate for the learning rate.
        decay_step (int): Step size for the learning rate decay.
        test_number (str): Test number for the experiment. Can be used to identify the test run and can be a string.
        extended_desc (str): Extended description to be placed at the bottom of the plot.
        folder (str): Folder where the plot will be saved. Defaults to 'graphs'.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if subfolder:
        folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    timestamp_str = datetime.now().strftime("%d-%m-%Y at %H:%M")
    filename_time = datetime.now().strftime("%d-%m-%Y-%H-%M")

    # Set up the plot
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"STGCN Training Loss (Test {test_number})")
    plt.suptitle(f"Generated on {timestamp_str}", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.ylim(0, 0.3)
	  # Convert subset ratio to percentage
    subset_ratio_percentage = subset_ratio * 100

    # Add key parameters as text at the bottom
    text_str = (
        f"Number of Nodes: {num_nodes}    "
        f"Hidden Channels: {hidden_channels}    "
        f"Learning Rate: {learning_rate}    "
        f"Subset Ratio: {subset_ratio_percentage}%   "
        f"Decay: {decay}    "
        f"Decay Step: {decay_step}    "
    )
    plt.figtext(0.5, -0.05, text_str, wrap=True, horizontalalignment='center', fontsize=9, color='gray')

    # Add extended description
    plt.figtext(0.5, -0.02, extended_desc, wrap=True, horizontalalignment='center', fontsize=8)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()

    save_path = os.path.join(folder, f"training_loss_{filename_time}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to {save_path}")
