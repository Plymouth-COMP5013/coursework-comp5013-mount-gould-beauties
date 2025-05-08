# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: Functions for plotting and saving neural network data.

# # ========== IMPORTS ==========
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime


# # ========== PLOTTING FUNCTIONS ==========
def plot_and_save_loss(train_losses, val_losses, num_nodes, hidden_channels, learning_rate, subset_ratio, decay, decay_step, intended_epochs, test_number, extended_desc, folder = 'graphs', subfolder = None):
    """
    Plots training loss over epochs, then saves the plot to a specified folder.
    
    Args:
        train_losses (list): List of loss values for each epoch, obtained during training.
        val_losses (list): List of validation loss values for each epoch, obtained during validation.
        num_nodes (int): Number of nodes in the graph.
        hidden_channels (int): Number of hidden channels in the model.
        learning_rate (float): Learning rate used in training.
        subset_ratio (float): Ratio of the dataset used for training.
        decay (float): Decay rate for the learning rate.
        decay_step (int): Step size for the learning rate decay.
        intended_epochs (int): Intended number of epochs for training, which may have been unfulfilled due to early stopping.
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
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='grey', label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, color='orange', label='Validation Loss', marker='o')
    plt.title(f"STGCN Training Loss (Test {test_number})")
    plt.suptitle(f"Generated on {timestamp_str}", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.grid(True)
    plt.legend(loc='upper right')
    # Convert subset ratio to percentage
    subset_ratio_percentage = subset_ratio * 100

    # Add key parameters as text at the bottom
    text_str = (
        f"Nodes: {num_nodes}    "
        f"Hidden Channels: {hidden_channels}    "
        f"Learning Rate: {learning_rate}    "
        f"Subset Ratio: {subset_ratio_percentage}%   "
        f"Decay: {decay}    "
        f"Decay Step: {decay_step}    "
        f"Intended Epochs: {intended_epochs}    "
        f"Stopped at: Epoch {len(train_losses)}    "
        f"Final Loss: {val_losses[-1]:.4f}"
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


def plot_ground_truth_and_predictions(times, ground_truth, predictions, test_number, folder = 'graphs', subfolder = None):
    """
    Plots ground truth and predictions over time, then saves the plot to a specified folder.
    
    Args:
        times (list): List of time points.
        ground_truth (list): List of ground truth values.
        predictions (list): List of predicted values.
        test_number (str): Test number for the experiment. Can be used to identify the test run and can be a string.
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

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(times, ground_truth, label="Ground Truth", linewidth=2)
    plt.plot(times, predictions, label="Predicted", linestyle='--', linewidth=2)

    # Format time on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability

    # Labels and legend
    plt.title("Speed over 24 Hours")
    plt.suptitle(f"Test {test_number} - Generated on {timestamp_str}", fontsize=10)
    plt.xlabel("Time of Day")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(folder, f"predictions_vs_ground_truth_{filename_time}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved predictions vs ground truth plot to {save_path}")