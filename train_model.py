# Author: Reef Lakin
# Last Modified: 07.05.2025
# Description: The main script for training our STGCN model on the traffic forecasting dataset.


# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, train_test_split, subset_data, shuffle_dataset, get_all_velocity_data
from utilities.plotting import plot_and_save_loss
from stgcn import STGCN
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from mechanisms.early_stopping import EarlyStopping
from mechanisms.normalisation import ZScoreNormaliser



# ========== OPTIONS ==========
# Ratio of the dataset to use for the subset to test on. Example: 0.03 = 3% of the dataset
SUBSET_RATIO = 0.1

# Learning rate for the optimizer. A good value is usually between 0.001 and 0.01
LEARNING_RATE = 0.001

# Number of hidden channels in the model. A good value is usually between 16 and 64. Higher numbers can lead to overfitting or longer training times.
HIDDEN_CHANNELS = 64

# Number of epochs to train the model. A good value is usually between 10 and 50, or lower for quick tests.
EPOCHS = 70

# Nuber of nodes in the dataset. This is usually fixed for a given dataset. Currently I'm only supporting 288 nodes.
NUM_NODES = 228

# Learning rate decay step size. A good value is usually between 3 and 8 if using aggressive decay.
STEP_SIZE = 5

# Gamme for learning rate decay. A good value is usually between 0.3 and 0.9.
GAMMA = 0.8

# Sub-folder for the graphs. If None is provided, the graphs will be saved in the highest level of the 'graphs' folder.
GRAPH_SUBFOLDER = "series_2"

# Test number for the experiment. Can be used to identify the test run and can be a string.
TEST_NUMBER = "2.5"

# Extended description to be placed at the bottom of the plot.
EXTENDED_DESC = "Another test with larger patience and slightly altered learning rate decay settings."

# Patience for early stopping (i.e., how many epochs to wait before stopping if no improvement is seen). 
PATIENCE = 20

# The number of 5-minute intervals ahead to predict. 3 means 15 minutes ahead, 6 means 30 minutes ahead, etc. DO NOT INCREASE BEYOND 4 (YET)!
FORECAST_HORIZON = 3

# Model saving options; would we like to save the model's architecture and state dictionary?
SAVE_ARCHITECTURE = True
SAVE_STATE_DICT = True



# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12, forecast_horizon=FORECAST_HORIZON)
all_velocity_values = get_all_velocity_data()
train_set, test_set = train_test_split(dataset)



# ========== SETUP ==========
model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
loss_fn = nn.MSELoss()
num_epochs = EPOCHS
early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.002)
normaliser = ZScoreNormaliser(all_velocity_values)



# ========== TRAINING LOOP ==========
model.train()
training_losses = []
validation_losses = []

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):

    # ----- Reset local variables -----
    norm_mse_loss_total = 0
    raw_mse_loss_total = 0
    all_predictions = []
    all_targets = []

    # ----- Shuffle and subset the dataset -----
    training_subset = shuffle_dataset(train_set)
    training_subset = subset_data(train_set, subset_ratio=SUBSET_RATIO)

    testing_subset = shuffle_dataset(test_set)
    testing_subset = subset_data(test_set, subset_ratio=SUBSET_RATIO)

    # ----- Training through each snapshot -----
    for time_step, snapshot in tqdm(enumerate(training_subset), desc="Training In Progress", leave=False):

        # ----- Forward pass through the STGCN model -----
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
        x = normaliser.normalise(x)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
        y_hat_single = y_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]

        # ----- Aquire the ground truth targets -----
        raw_targets = snapshot.y.view(-1)
        norm_targets = normaliser.normalise(raw_targets)

        # ----- Compute the loss -----
        norm_mse_loss = torch.mean((y_hat_single - norm_targets) ** 2)
        norm_mse_loss_total = norm_mse_loss_total + norm_mse_loss

        raw_y_hat_single = normaliser.denormalise(y_hat_single)
        raw_mse_loss = torch.mean((raw_y_hat_single - raw_targets) ** 2)
        raw_mse_loss_total = raw_mse_loss_total + raw_mse_loss

        # ----- Collect predictions and targets for later analysis -----
        all_predictions.append(y_hat_single.detach())
        all_targets.append(norm_targets.detach())

    # ----- RMSE calculation -----
    avg_raw_mse_loss = raw_mse_loss_total / (time_step + 1)
    avg_norm_mse_loss = norm_mse_loss_total / (time_step + 1)

    avg_raw_rmse_loss = torch.sqrt(avg_raw_mse_loss)
    avg_norm_mse_loss = torch.sqrt(avg_norm_mse_loss)

    training_losses.append(avg_raw_rmse_loss.item())

    # ----- Backpropagation and optimisation -----
    avg_norm_mse_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # ----- Validation begins -----
    model.eval()
    norm_mse_val_loss_total = 0
    raw_mse_val_loss_total = 0
    with torch.no_grad():

        # ----- Validation through each snapshot, as with the training loop -----
        for time_step, snapshot in tqdm(enumerate(testing_subset), desc="Validation In Progress", leave=False):

            # ----- Forward pass through the STGCN model -----
            x_val = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
            x_val = normaliser.normalise(x_val)
            y_val_hat = model(x_val, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
            y_val_hat_single = y_val_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]

            # ----- Aquire the ground truth targets -----
            raw_y_val_target = snapshot.y.view(-1)
            norm_y_val_target = normaliser.normalise(raw_y_val_target)

            # ----- Compute the validation loss -----
            norm_mse_val_loss = torch.mean((y_val_hat_single - norm_y_val_target) ** 2)
            norm_mse_val_loss_total = norm_mse_val_loss_total + norm_mse_val_loss

            raw_y_val_hat_single = normaliser.denormalise(y_val_hat_single)
            raw_mse_val_loss = torch.mean((raw_y_val_hat_single - raw_y_val_target) ** 2)
            raw_mse_val_loss_total = raw_mse_val_loss_total + raw_mse_val_loss

    # ----- RMSE calculation -----
    avg_raw_mse_val_loss = raw_mse_val_loss_total / (time_step + 1)
    avg_norm_mse_val_loss = norm_mse_val_loss_total / (time_step + 1)

    raw_rmse_val_loss = torch.sqrt(avg_raw_mse_val_loss)
    norm_rmse_val_loss = torch.sqrt(avg_norm_mse_val_loss)

    validation_losses.append(raw_rmse_val_loss.item())

    # ----- Print the training and validation loss -----
    print(f"Epoch {epoch + 1}/{num_epochs}, Raw Validation Loss: {raw_rmse_val_loss.item():.4f}, Normalised Validation Loss: {norm_rmse_val_loss.item():.4f}")

    # ----- Early stopping check -----
    if early_stopping.check(norm_rmse_val_loss.item()):
        print(f"Early stopping has been triggered at epoch {epoch + 1}")
        break
    
    # ----- Continue training if early stopping is not triggered -----
    model.train()

    # ----- Print the training loss -----
    print(f"Epoch {epoch + 1}/{num_epochs}, Raw Training Loss: {avg_raw_rmse_loss.item():.4f}, Normalised Training Loss: {avg_norm_mse_loss.item():.4f}")

    # ----- Sample (15) predictions and targets -----
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    sample_indices = torch.randperm(all_predictions.size(0))[:15]
    sampled_predictions = all_predictions[sample_indices]
    sampled_targets = all_targets[sample_indices]

    # ----- Print the sampled predictions and targets -----
    sampled_predictions = normaliser.denormalise(sampled_predictions)
    sampled_targets = normaliser.denormalise(sampled_targets)

    print(f"\nSampled Predictions and Targets for Epoch {epoch + 1}:")
    for i in range(len(sampled_predictions)):
        print(f"Sample {i + 1}: Predicted: {sampled_predictions[i].item():.2f}, Target: {sampled_targets[i].item():.2f}")



# ========== PLOT THE LOSS ==========
plot_and_save_loss(
    training_losses,
    validation_losses,
    num_nodes=NUM_NODES,
    hidden_channels=HIDDEN_CHANNELS,
    learning_rate=LEARNING_RATE,
    subset_ratio=SUBSET_RATIO,
    decay=GAMMA,
    decay_step=STEP_SIZE,
    intended_epochs=EPOCHS,
    test_number=TEST_NUMBER,
    extended_desc=EXTENDED_DESC,
    folder='graphs',
    subfolder=GRAPH_SUBFOLDER)



# ========== OLD TESTING METHOD (Currently Under Construction) ==========
# model.eval()

# mse_loss = 0
# rmse_loss = 0
# with torch.no_grad():
#     for time_step, snapshot in tqdm(enumerate(testing_subset), desc="Testing Batches", leave=False):
#         x = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
#         x = normaliser.normalise(x)
#         y_hat = model(x, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
#         y_hat_single = y_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]
#         target = normaliser.normalise(snapshot.y.view(-1)) 

#         # Compute the loss
#         mse = torch.mean((y_hat_single - target) ** 2)
#         rmse = torch.sqrt(mse)

#         # Add the loss to the total loss
#         mse_loss = mse_loss + mse
#         rmse_loss = rmse_loss + rmse

#     rmse_loss = rmse_loss.item() / (time_step + 1)
#     print(f"Test Loss: {rmse_loss:.4f}")



# ========== SAVE MODEL ==========
if SAVE_ARCHITECTURE or SAVE_STATE_DICT:
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")

    save_data = {
        'model_state_dict': model.state_dict(),
        'normalisation_mean': normaliser.mean,
        'normalisation_std': normaliser.std,
    }

    # Save the model state dictionary and normalisation values
    if SAVE_STATE_DICT:
        torch.save(save_data, f"stgcn_model_state_dict_{timestamp}.pth")

    # Save the model architecture
    if SAVE_ARCHITECTURE:
        torch.save(model, f"stgcn_model_architecture_{timestamp}.pth")