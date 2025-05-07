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
SUBSET_RATIO = 0.05

# Learning rate for the optimizer. A good value is usually between 0.001 and 0.01
LEARNING_RATE = 0.002

# Number of hidden channels in the model. A good value is usually between 16 and 64. Higher numbers can lead to overfitting or longer training times.
HIDDEN_CHANNELS = 64

# Number of epochs to train the model. A good value is usually between 10 and 50, or lower for quick tests.
EPOCHS = 10

# Nuber of nodes in the dataset. This is usually fixed for a given dataset. Currently I'm only supporting 288 nodes.
NUM_NODES = 228

# Learning rate decay step size. A good value is usually between 3 and 8 if using aggressive decay.
STEP_SIZE = 2

# Gamme for learning rate decay. A good value is usually between 0.3 and 0.9.
GAMMA = 0.8

# Sub-folder for the graphs. If None is provided, the graphs will be saved in the highest level of the 'graphs' folder.
GRAPH_SUBFOLDER = "series_2"

# Test number for the experiment. Can be used to identify the test run and can be a string.
TEST_NUMBER = "2.3"

# Extended description to be placed at the bottom of the plot.
EXTENDED_DESC = "This is after reworking the validation loss to be denormalised to the raw training speeds."

# Patience for early stopping (i.e., how many epochs to wait before stopping if no improvement is seen). 
PATIENCE = 6

# The number of 5-minute intervals ahead to predict. 3 means 15 minutes ahead, 6 means 30 minutes ahead, etc. DO NOT INCREASE BEYOND 4 (YET)!
FORECAST_HORIZON = 3

# Model saving options; would we like to save the model's architecture and state dictionary?
SAVE_ARCHITECTURE = False
SAVE_STATE_DICT = False


# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12, forecast_horizon=FORECAST_HORIZON)
# train_subet, test_subset = train_test_subset(dataset, subset_ratio=SUBSET_RATIO)
all_velocity_values = get_all_velocity_data()
train_set, test_set = train_test_split(dataset)


# ========== SETUP ==========
model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
loss_fn = nn.MSELoss()
num_epochs = EPOCHS
early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.003)
normaliser = ZScoreNormaliser(all_velocity_values)


# ========== COMPUTE NORMALISATION Values ==========

# Flatten all x and y values across the training and testing subsets
all_values_x_train = torch.cat([snapshot.x.view(-1) for snapshot in train_set])
all_values_y_train = torch.cat([snapshot.y.view(-1) for snapshot in train_set])
all_values_x_test = torch.cat([snapshot.x.view(-1) for snapshot in test_set])
all_values_y_test = torch.cat([snapshot.y.view(-1) for snapshot in test_set])

# Combine the x and y values into a single tensor
all_values_x = torch.cat((all_values_x_train, all_values_x_test))
all_values_y = torch.cat((all_values_y_train, all_values_y_test))

# Combine the x and y values into a single tensor
all_values = torch.cat((all_values_x, all_values_y))

# Compute the min and max values
min_value = all_values.min()
max_value = all_values.max()
print(f"Min value: {min_value.item()}, Max value: {max_value.item()}")


# ========== TRAINING LOOP ==========
model.train()

training_losses = []
validation_losses = []

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):

    # Reset local variables
    norm_mse_loss = 0
    norm_rmse_loss = 0
    raw_mse_loss = 0
    raw_rmse_loss = 0
    all_predictions = []
    all_targets = []

    # Split and shuffle the training dataset
    training_subset = shuffle_dataset(train_set)
    training_subset = subset_data(train_set, subset_ratio=SUBSET_RATIO)

    # Split and shuffle the testing dataset
    testing_subset = shuffle_dataset(test_set)
    testing_subset = subset_data(test_set, subset_ratio=SUBSET_RATIO)

    for time_step, snapshot in tqdm(enumerate(training_subset), desc="Training Batches", leave=False):
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
        x = normaliser.normalise(x)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
        y_hat_single = y_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]

        raw_targets = snapshot.y.view(-1)
        norm_targets = normaliser.normalise(raw_targets)

        # Compute the loss (normalised)
        norm_mse = torch.mean((y_hat_single - norm_targets) ** 2)
        norm_rmse = torch.sqrt(norm_mse)

        # Add the loss to the total loss (normalised)
        norm_mse_loss = norm_mse_loss + norm_mse
        norm_rmse_loss = norm_rmse_loss + norm_rmse

        # Denormalise the predictions and compute the loss (raw)
        raw_y_hat_single = normaliser.denormalise(y_hat_single)
        raw_mse = torch.mean((raw_y_hat_single - raw_targets) ** 2)
        raw_rmse = torch.sqrt(raw_mse)

        # Add the loss to the total loss (raw)
        raw_mse_loss = raw_mse_loss + raw_mse
        raw_rmse_loss = raw_rmse_loss + raw_rmse

        # Collect the normalised predictions and targets for sampling later
        all_predictions.append(y_hat_single.detach())
        all_targets.append(norm_targets.detach())

    norm_rmse_loss = norm_rmse_loss / (time_step + 1)
    norm_rmse_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # Append the training loss to the list
    training_losses.append(raw_rmse_loss.item())

    # Check for early stopping by using the test subset for validation
    model.eval()
    norm_mse_val_loss = 0
    norm_rmse_val_loss = 0
    raw_mse_val_loss = 0
    raw_rmse_val_loss = 0
    with torch.no_grad():
        for time_step, snapshot in enumerate(testing_subset):
            x_val = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
            x_val = normaliser.normalise(x_val)
            y_val_hat = model(x_val, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
            y_val_hat_single = y_val_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]
            y_val_target = normaliser.normalise(snapshot.y.view(-1))

            # Compute the validation loss (normalised)
            norm_mse_val = torch.mean((y_val_hat_single - y_val_target) ** 2)
            norm_rmse_val = torch.sqrt(norm_mse_val)

            # Add the validation loss to the total validation loss (normalised)
            norm_mse_val_loss = norm_mse_val_loss + norm_mse_val
            norm_rmse_val_loss = norm_rmse_val_loss + norm_rmse_val

            # Denormalise the predictions and compute the validation loss (raw)
            raw_y_val_hat_single = normaliser.denormalise(y_val_hat_single)
            raw_mse_val = torch.mean((raw_y_val_hat_single - snapshot.y.view(-1)) ** 2)
            raw_rmse_val = torch.sqrt(raw_mse_val)

            # Add the validation loss to the total validation loss (raw)
            raw_mse_val_loss = raw_mse_val_loss + raw_mse_val
            raw_rmse_val_loss = raw_rmse_val_loss + raw_rmse_val

    norm_rmse_val_loss = norm_rmse_val_loss / (time_step + 1)
    print(f"Epoch {epoch + 1}/{num_epochs}, Raw Validation Loss: {raw_rmse_val_loss.item():.4f}, Normalised Validation Loss: {norm_rmse_val_loss.item():.4f}")

    # Append the validation loss to the list
    validation_losses.append(raw_rmse_val_loss.item())

    # Check for early stopping
    if early_stopping.check(norm_rmse_val_loss.item()):
        print(f"Early stopping at epoch {epoch + 1}")
        break

    # If not stopping, continue training
    model.train()

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Randomly sample 8 indices
    sample_indices = torch.randperm(all_predictions.size(0))[:8]
    sampled_predictions = all_predictions[sample_indices]
    sampled_targets = all_targets[sample_indices]

    print(f"Epoch {epoch + 1}/{num_epochs}, Raw Training Loss: {raw_rmse_loss.item():.4f}, Normalised Training Loss: {norm_rmse_loss.item():.4f}")

    # Denormalise the sampled predictions and targets
    sampled_predictions = normaliser.denormalise(sampled_predictions)
    sampled_targets = normaliser.denormalise(sampled_targets)

    # Print the sampled predictions and targets
    for i in range(len(sampled_predictions)):
        print(f"Sample {i + 1}: Predicted: {sampled_predictions[i].item():.2f}, Target: {sampled_targets[i].item():.2f}")

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


# ========== TESTING ==========
model.eval()

mse_loss = 0
rmse_loss = 0
with torch.no_grad():
    for time_step, snapshot in tqdm(enumerate(testing_subset), desc="Testing Batches", leave=False):
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1) # [1, 12, 228, 1]
        x = normaliser.normalise(x)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight) # [1, 4, 228, 1]
        y_hat_single = y_hat[:, FORECAST_HORIZON - 1, :, :].squeeze() # [228]
        target = normaliser.normalise(snapshot.y.view(-1)) 

        # Compute the loss
        mse = torch.mean((y_hat_single - target) ** 2)
        rmse = torch.sqrt(mse)

        # Add the loss to the total loss
        mse_loss = mse_loss + mse
        rmse_loss = rmse_loss + rmse

    rmse_loss = rmse_loss.item() / (time_step + 1)
    print(f"Test Loss: {rmse_loss:.4f}")


# ========== SAVE MODEL ==========
if SAVE_ARCHITECTURE or SAVE_STATE_DICT:
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")

    save_data = {
        'model_state_dict': model.state_dict(),
        'min_val': min_value,
        'max_val': max_value
    }

    # Save the model state dictionary and normalisation values
    if SAVE_STATE_DICT:
        torch.save(save_data, f"stgcn_model_state_dict_{timestamp}.pth")

    # Save the model architecture
    if SAVE_ARCHITECTURE:
        torch.save(model, f"stgcn_model_architecture_{timestamp}.pth")