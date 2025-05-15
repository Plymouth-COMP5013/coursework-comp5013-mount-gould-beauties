# Author: Reef Lakin
# Last Modified: 15.05.2025
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
from torch.optim.lr_scheduler import LambdaLR, StepLR
from mechanisms.early_stopping import EarlyStopping
from mechanisms.normalisation import ZScoreNormaliser



# ========== LEGACY OPTIONS ==========
# The following options are legacy options that are officially deprecated and might get removed in the future.

# Subset Ratio:- Portion of the dataset to use for training and validation. Example: 0.03 = 3% of the dataset. Turn off by setting to None.
SUBSET_RATIO = None



# ========== MAIN OPTIONS ==========
# These are the main options for the STGCN training flow.

# Learning rate for the optimizer. A good value is 0.001, and will decrease with the learning rate scheduler.
LEARNING_RATE = 0.001

# Gamma for learning rate decay. A good value is between 0.7 and 0.9. Lower values than this will mean a more aggressive decay.
GAMMA = 0.6

# Learning rate decay step size. After how many epochs should the learning rate decay? Smaller values means it will decay quicker.
STEP_SIZE = 3

# The number of epochs the learning rate scheduler takes to "warm-up".
WARMUP_EPOCHS = 10

# Whether or not the model should use warmup. If False, the learning rate will decay at the STEP_SIZE interval.
USE_WARMUP = True

# Number of hidden channels in the model. A good value is usually between 16 and 64. Higher numbers have only seen worse performance and longer training times.
HIDDEN_CHANNELS = 32

# Number of epochs to train the model. A good value is around 50, but early stopping may trigger the model to stop training earlier.
EPOCHS = 50

# Mini-batch Size:- The number of time snapshots before the model's weights get backpropagated. A good value is 16-64, but experimentation is recommended.
MINIBATCH_SIZE = 32

# Number of nodes in the dataset. Currently only 228 nodes are supported.
NUM_NODES = 228

# Sub-folder for the graphs. If None is provided, the graphs will be saved in the highest level of the 'graphs' folder. Can be anything.
GRAPH_SUBFOLDER = "series_4"

# Test number for the experiment. Can be used to identify the test run on a loss plot. Doesn't have to be a number, can be anything.
TEST_NUMBER = "4.1"

# Extended description to be placed at the bottom of the plot. Describe what this test is about, maybe what you've changed. Again, can be anything.
EXTENDED_DESC = "An extended run of Test 3.9, as that was very very promising."

# Patience for early stopping (i.e., how many epochs to wait before stopping if no improvement is seen). Kills the training if validation loss doesn't improve for this many epochs.
PATIENCE = 8

# The improvement threshold for early stopping. If the validation loss doesn't improve by this amount, the training will stop.
MIN_DELTA = 0.005

# The number of 5-minute intervals ahead to predict. 3 means 15 minutes ahead, 6 means 30 minutes ahead, etc. DO NOT INCREASE BEYOND 4 (YET)!
FORECAST_HORIZON = 3

# Model saving options; would we like to save the model's architecture and state dictionary to the root directory?
SAVE_ARCHITECTURE = True
SAVE_STATE_DICT = True

# Add a relative path to a pretrained model to load it in. If None, the model will be trained from scratch.
PRETRAINED_MODEL_PATH = None



# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12, forecast_horizon=FORECAST_HORIZON)
all_velocity_values = get_all_velocity_data()
train_set, test_set = train_test_split(dataset)



# ========== LEARNING RATE FUNCTION ==========
def lr_lambda(epoch):

    # Linear warmup for the first WARMUP_EPOCHS epochs
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / WARMUP_EPOCHS 
    
    # After warmup, use the step decay
    else:
        return GAMMA ** ((epoch - WARMUP_EPOCHS) // STEP_SIZE)
    


# ========== LOAD MODEL ==========
if PRETRAINED_MODEL_PATH is not None:
    checkpoint = torch.load(PRETRAINED_MODEL_PATH)
    model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Pretrained model loaded successfully.")
else:
    model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)
    print("No pretrained model found. Training from scratch.")



# ========== SETUP ==========
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
num_epochs = EPOCHS
early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
normaliser = ZScoreNormaliser(all_velocity_values)

if USE_WARMUP:
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
else:
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)



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
    minibatch_count = 0
    minibatch_loss = 0
    optimizer.zero_grad()

    # ----- Shuffle and subset the dataset -----
    training_subset = shuffle_dataset(train_set)
    if SUBSET_RATIO is not None:
        training_subset = subset_data(training_subset, subset_ratio=SUBSET_RATIO)

    testing_subset = shuffle_dataset(test_set)
    if SUBSET_RATIO is not None:
        testing_subset = subset_data(testing_subset, subset_ratio=SUBSET_RATIO)

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
        norm_mse_loss_total += norm_mse_loss.item()
        minibatch_loss = minibatch_loss + norm_mse_loss

        raw_y_hat_single = normaliser.denormalise(y_hat_single)
        raw_mse_loss = torch.mean((raw_y_hat_single - raw_targets) ** 2)
        raw_mse_loss_total = raw_mse_loss_total + raw_mse_loss

        # ----- Collect predictions and targets for later analysis -----
        all_predictions.append(y_hat_single.detach())
        all_targets.append(norm_targets.detach())

        # ----- Backpropagation and optimisation -----
        minibatch_count += 1
        if minibatch_count == MINIBATCH_SIZE:

            minibatch_loss = minibatch_loss / MINIBATCH_SIZE
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = torch.tensor(0.0, requires_grad=True)
            minibatch_count = 0

    # ----- RMSE calculation -----
    avg_raw_mse_loss = raw_mse_loss_total / (time_step + 1)
    avg_norm_mse_loss = norm_mse_loss_total / (time_step + 1)

    avg_raw_rmse_loss = torch.sqrt(avg_raw_mse_loss)
    avg_norm_rmse_loss = torch.sqrt(avg_norm_mse_loss)

    training_losses.append(avg_raw_rmse_loss.item())

    # ----- Learning rate scheduler step -----
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

    # ----- Sample (10) predictions and targets -----
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    sample_indices = torch.randperm(all_predictions.size(0))[:10]
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
    minibatch_size=MINIBATCH_SIZE,
    decay=GAMMA,
    decay_step=STEP_SIZE,
    intended_epochs=EPOCHS,
    test_number=TEST_NUMBER,
    extended_desc=EXTENDED_DESC,
    folder='graphs',
    subfolder=GRAPH_SUBFOLDER)



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