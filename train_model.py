# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, train_test_subset, train_test_split, subset_data, shuffle_dataset, get_all_velocity_data
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
SUBSET_RATIO = 0.2

# Learning rate for the optimizer. A good value is usually between 0.001 and 0.01
LEARNING_RATE = 0.003

# Number of hidden channels in the model. A good value is usually between 16 and 64. Higher numbers can lead to overfitting or longer training times.
HIDDEN_CHANNELS = 64

# Number of epochs to train the model. A good value is usually between 10 and 50, or lower for quick tests.
EPOCHS = 60

# Nuber of nodes in the dataset. This is usually fixed for a given dataset. Currently I'm only supporting 288 nodes.
NUM_NODES = 228

# Learning rate decay step size. A good value is usually between 3 and 8 if using aggressive decay.
STEP_SIZE = 5

# Gamme for learning rate decay. A good value is usually between 0.3 and 0.9.
GAMMA = 0.7

# Sub-folder for the graphs. If None is provided, the graphs will be saved in the highest level of the 'graphs' folder.
GRAPH_SUBFOLDER = "series_2"

# Test number for the experiment. Can be used to identify the test run and can be a string.
TEST_NUMBER = "2.0"

# Extended description to be placed at the bottom of the plot.
EXTENDED_DESC = "First test with z-score standardisation. A smaller learning rate of 0.003 is used and patience is increased."

# Patience for early stopping (i.e., how many epochs to wait before stopping if no improvement is seen). 
PATIENCE = 10

# Model saving options; would we like to save the model's architecture and state dictionary?
SAVE_ARCHITECTURE = True
SAVE_STATE_DICT = True


# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12)
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
    mse_loss = 0
    rmse_loss = 0
    all_predictions = []
    all_targets = []

    # Split and shuffle the training dataset
    training_subset = shuffle_dataset(train_set)
    training_subset = subset_data(train_set, subset_ratio=SUBSET_RATIO)

    # Split and shuffle the testing dataset
    testing_subset = shuffle_dataset(test_set)
    testing_subset = subset_data(test_set, subset_ratio=SUBSET_RATIO)

    for time_step, snapshot in tqdm(enumerate(training_subset), desc="Training Batches", leave=False):
        # print(snapshot)
        # print(time_step)
        # print(snapshot.x.shape)
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
        # x = (x - min_value) / (max_value - min_value)  # Normalise x via min-max scaling
        x = normaliser.normalise(x)  # Normalise x via z-score scaling
        # print(x.shape)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
        # print(f"y_hat shape: {y_hat.shape}")         # likely [1, 1, 288, 1]
        # print(f"snapshot.y shape: {snapshot.y.shape}")  # this is the question mark
        # Take the first timestep from y_hat
        y_hat_single = y_hat[:, 0, :, :].squeeze()   # shape: [228]
        # target = (snapshot.y.view(-1) - min_value) / (max_value - min_value)  # Normalise target via min-max scaling
        target = normaliser.normalise(snapshot.y.view(-1))  # Normalise target via z-score scaling

        # Compute the loss
        mse = torch.mean((y_hat_single - target) ** 2)
        rmse = torch.sqrt(mse)

        # Add the loss to the total loss
        mse_loss = mse_loss + mse
        rmse_loss = rmse_loss + rmse

        # Collect predictions and targets for sampling later
        all_predictions.append(y_hat_single.detach())
        all_targets.append(target.detach())

    rmse_loss = rmse_loss / (time_step + 1)
    rmse_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # Append the training loss to the list
    training_losses.append(rmse_loss.item())

    # Check for early stopping by using the test subset for validation
    model.eval()
    mse_val_loss = 0
    rmse_val_loss = 0
    with torch.no_grad():
        for time_step, snapshot in enumerate(testing_subset):
            x_val = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
            # x_val = (x_val - min_value) / (max_value - min_value)
            x_val = normaliser.normalise(x_val)  # Normalise x via z-score scaling
            y_val_hat = model(x_val, snapshot.edge_index, snapshot.edge_weight)
            y_val_hat_single = y_val_hat[:, 0, :, :].squeeze()
            # y_val_target = (snapshot.y.view(-1) - min_value) / (max_value - min_value)
            y_val_target = normaliser.normalise(snapshot.y.view(-1))  # Normalise target via z-score scaling

            # Compute the validation loss
            mse_val = torch.mean((y_val_hat_single - y_val_target) ** 2)
            rmse_val = torch.sqrt(mse_val)

            # Add the validation loss to the total validation loss
            mse_val_loss = mse_val_loss + mse_val
            rmse_val_loss = rmse_val_loss + rmse_val

    rmse_val_loss = rmse_val_loss / (time_step + 1)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {rmse_val_loss.item():.4f}")
    print(f"Snapshot Count in Training Set: {training_subset.snapshot_count}/{train_set.snapshot_count}")
    print(f"Snapshot Count in Validation Set: {testing_subset.snapshot_count}/{test_set.snapshot_count}")

    # Append the validation loss to the list
    validation_losses.append(rmse_val_loss.item())

    # Check for early stopping
    if early_stopping.check(rmse_val_loss.item()):
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

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {rmse_loss.item():.4f}")

    # Denormalise the sampled predictions and targets
    # sampled_predictions = sampled_predictions * (max_value - min_value) + min_value
    # sampled_targets = sampled_targets * (max_value - min_value) + min_value
    sampled_predictions = normaliser.denormalise(sampled_predictions)
    sampled_targets = normaliser.denormalise(sampled_targets)

    # Print the sampled predictions and targets
    for i in range(len(sampled_predictions)):
        print(f"Sample {i + 1}: Predicted: {sampled_predictions[i].item():.4f}, Target: {sampled_targets[i].item():.4f}")

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


# ========== EVALUATION ==========
model.eval()

mse_loss = 0
rmse_loss = 0
with torch.no_grad():
    for time_step, snapshot in tqdm(enumerate(testing_subset), desc="Testing Batches", leave=False):
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
        # x = (x - min_value) / (max_value - min_value) # Normalise x via min-max scaling
        x = normaliser.normalise(x)  # Normalise x via z-score scaling
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
        y_hat_single = y_hat[:, 0, :, :].squeeze()
        # target = (snapshot.y.view(-1) - min_value) / (max_value - min_value) # Normalise target via min-max scaling
        target = normaliser.normalise(snapshot.y.view(-1))  # Normalise target via z-score scaling

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