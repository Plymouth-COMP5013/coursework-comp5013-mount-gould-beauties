# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: Functions for plotting and saving neural network data.

# # ========== IMPORTS ==========
import matplotlib.pyplot as plt
import os
from datetime import datetime


# # ========== PLOTTING FUNCTIONS ==========
def plot_and_save_loss(losses, folder='graphs', num_nodes=228, hidden_channels=16, learning_rate=0.001, subset_ratio=0.03):
    """
    Plots training loss over epochs, then saves the plot to a specified folder.
    
    Args:
		losses (list): List of loss values for each epoch.
		folder (str): Folder to save the plot. Default is 'graphs'.
		num_nodes (int): Number of nodes in the graph. Default is 228.
		hidden_channels (int): Number of hidden channels in the model. Default is 16.
		learning_rate (float): Learning rate used in training. Default is 0.001.
		subset_ratio (float): Ratio of the dataset used for training. Default is 0.03.
        
    Returns:
		None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp_str = datetime.now().strftime("%d-%m-%Y at %H:%M:%S")
    filename_time = datetime.now().strftime("%d-%m-%Y-%H-%M")

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss")
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
        f"Subset Ratio: {subset_ratio_percentage}%"
    )
    plt.figtext(0.5, 0.01, text_str, wrap=True, horizontalalignment='center', fontsize=9)

    save_path = os.path.join(folder, f"training_loss_{filename_time}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to {save_path}")
