# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: Functions for plotting and saving neural network data.

# # ========== IMPORTS ==========
import matplotlib.pyplot as plt
import os
from datetime import datetime


# # ========== PLOTTING FUNCTIONS ==========
def plot_and_save_loss(losses, folder='graphs'):
    """
    Plots training loss over epochs, then saves the plot to a specified folder.
    
    Args:
		losses (list): List of loss values for each epoch.
		folder (str): Folder to save the plot. Default is 'graphs'.
        
    Returns:
		None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss")
    plt.suptitle(f"Generated on {timestamp}", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    save_path = os.path.join(folder, f"training_loss_{filename_time}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training loss plot to {save_path}")
