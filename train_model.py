# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, train_test_subset
from utilities.plotting import plot_and_save_loss
from stgcn import STGCN
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from early_stopping import EarlyStopping


# ========== OPTIONS ==========
# Ratio of the dataset to use for the subset to test on. Example: 0.03 = 3% of the dataset
SUBSET_RATIO = 0.3

# Learning rate for the optimizer. A good value is usually between 0.001 and 0.01
LEARNING_RATE = 0.004

# Number of hidden channels in the model. A good value is usually between 16 and 64. Higher numbers can lead to overfitting or longer training times.
HIDDEN_CHANNELS = 32

# Number of epochs to train the model. A good value is usually between 10 and 50, or lower for quick tests.
EPOCHS = 50

# Nuber of nodes in the dataset. This is usually fixed for a given dataset. Currently I'm only supporting 288 nodes.
NUM_NODES = 228

# Learning rate decay step size. A good value is usually between 5 and 10.
STEP_SIZE = 2

# Gamme for learning rate decay. A good value is usually between 0.3 and 0.9.
GAMMA = 0.7

# Sub-folder for the graphs. If None is provided, the graphs will be saved in the highest level of the 'graphs' folder.
GRAPH_SUBFOLDER = "series_1"

# Test number for the experiment. Can be used to identify the test run and can be a string.
TEST_NUMBER = "1.1"

# Extended description to be placed at the bottom of the plot.
EXTENDED_DESC = "Test on 30 percent of the dataset with the new early stopping mechanic."

# Model saving options; would we like to save the model's architecture and state dictionary?
SAVE_ARCHITECTURE = True
SAVE_STATE_DICT = True


# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12)
train_subet, test_subset = train_test_subset(dataset, subset_ratio=SUBSET_RATIO)


# ========== SETUP ==========
model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
loss_fn = nn.MSELoss()
num_epochs = EPOCHS
early_stopping = EarlyStopping(patience=4, min_delta=0.01)


# ========== COMPUTE NORMALISATION Values ==========

# Flatten all x and y values across the training and testing subsets
all_values_x_train = torch.cat([snapshot.x.view(-1) for snapshot in train_subet])
all_values_y_train = torch.cat([snapshot.y.view(-1) for snapshot in train_subet])
all_values_x_test = torch.cat([snapshot.x.view(-1) for snapshot in test_subset])
all_values_y_test = torch.cat([snapshot.y.view(-1) for snapshot in test_subset])

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

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    loss = 0
    all_predictions = []
    all_targets = []
    for time_step, snapshot in tqdm(enumerate(train_subet), desc="Training Batches", leave=False):
        # print(snapshot)
        # print(time_step)
        # print(snapshot.x.shape)
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
        x = (x - min_value) / (max_value - min_value)  # Normalise x via min-max scaling
        # print(x.shape)
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
        # print(f"y_hat shape: {y_hat.shape}")         # likely [1, 1, 288, 1]
        # print(f"snapshot.y shape: {snapshot.y.shape}")  # this is the question mark
        # Take the first timestep from y_hat
        y_hat_single = y_hat[:, 0, :, :].squeeze()   # shape: [228]
        target = (snapshot.y.view(-1) - min_value) / (max_value - min_value)  # Normalise target via min-max scaling

        loss = loss + torch.mean((y_hat_single - target) ** 2)

        # Collect predictions and targets for sampling later
        all_predictions.append(y_hat_single.detach())
        all_targets.append(target.detach())

    loss = loss / (time_step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # Check for early stopping by using the test subset for validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for time_step, snapshot in enumerate(test_subset):
            x_val = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
            x_val = (x_val - min_value) / (max_value - min_value)
            y_val_hat = model(x_val, snapshot.edge_index, snapshot.edge_weight)
            y_val_hat_single = y_val_hat[:, 0, :, :].squeeze()
            y_val_target = (snapshot.y.view(-1) - min_value) / (max_value - min_value)

            val_loss += torch.mean((y_val_hat_single - y_val_target) ** 2)

    val_loss = val_loss / (time_step + 1)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss.item():.4f}")

    # Check for early stopping
    if early_stopping.check(val_loss.item()):
        print(f"Early stopping at epoch {epoch + 1}")
        break

    # If not stopping, continue training
    model.train()
        
    training_losses.append(loss.item())

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Randomly sample 8 indices
    sample_indices = torch.randperm(all_predictions.size(0))[:8]
    sampled_predictions = all_predictions[sample_indices]
    sampled_targets = all_targets[sample_indices]

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Denormalise the sampled predictions and targets
    sampled_predictions = sampled_predictions * (max_value - min_value) + min_value
    sampled_targets = sampled_targets * (max_value - min_value) + min_value

    # Print the sampled predictions and targets
    for i in range(len(sampled_predictions)):
        print(f"Sample {i + 1}: Predicted: {sampled_predictions[i].item():.4f}, Target: {sampled_targets[i].item():.4f}")

plot_and_save_loss(
    training_losses,
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

total_loss = 0
with torch.no_grad():
    for time_step, snapshot in tqdm(enumerate(test_subset), desc="Testing Batches", leave=False):
        x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
        x = (x - min_value) / (max_value - min_value) # Normalise x via min-max scaling
        y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
        y_hat_single = y_hat[:, 0, :, :].squeeze()
        target = (snapshot.y.view(-1) - min_value) / (max_value - min_value) # Normalise target via min-max scaling

        loss = loss + torch.mean((y_hat_single - target) ** 2)

    total_loss += loss.item()
    total_loss = total_loss / (time_step + 1)
    print(f"Test Loss: {total_loss:.4f}")


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