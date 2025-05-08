# ========== IMPORTS ==========
import torch
from stgcn import STGCN
from utilities.preprocessing import load_dataset_for_stgcn, train_test_split, subset_data, shuffle_dataset, get_all_velocity_data
from mechanisms.normalisation import ZScoreNormaliser
from tqdm import tqdm
from datetime import datetime
import os



# ========== SETUP Stored Model Files ==========
# Constants
NUM_NODES = 228
HIDDEN_CHANNELS = 64
FORECAST_HORIZON = 3

# File path for the .pth state dict file
file_path = 'saved_models/series_3/3_2/stgcn_model_state_dict_08-05-2025-09-15.pth'

# Unpack the state dict
checkpoint = torch.load(file_path)

# Initialise a new STGCN model
model = STGCN(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=1, num_nodes=NUM_NODES)

# Load the state dict into the model
model.load_state_dict(checkpoint['model_state_dict'])

# Grab the mean and std from the checkpoint
mean = checkpoint['normalisation_mean']
std = checkpoint['normalisation_std']

# Test description
test_description = "Full test on Model 3.4"



# ========== LOAD DATA ==========
dataset = load_dataset_for_stgcn(window_size=12, forecast_horizon=FORECAST_HORIZON)
all_velocity_values = get_all_velocity_data()
unused_train_set, test_set = train_test_split(dataset)
normaliser = ZScoreNormaliser(all_velocity_values)



# ========== TESTING ==========
model.eval()
norm_mse_val_loss_total = 0
raw_mse_val_loss_total = 0
with torch.no_grad():

	# ----- Validation through each snapshot, as with the training loop -----
	for time_step, snapshot in tqdm(enumerate(test_set), desc="Full Test Set Validation", leave=False):

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



# ========== SAVE TO FILE ==========
timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")

# Create the directory if it doesn't exist
output_dir = "test_results/full_test_set_validation"
os.makedirs(output_dir, exist_ok=True)

# Define the file path
output_file_path = os.path.join(output_dir, f"validation_results_{timestamp}.txt")

# Write the loss values to the file
with open(output_file_path, "w") as file:
	file.write(f"Test Description: {test_description}\n")
	file.write(f"Raw RMSE Validation Loss: {raw_rmse_val_loss.item():.6f}\n")
	file.write(f"Normalized RMSE Validation Loss: {norm_rmse_val_loss.item():.6f}\n")
	file.write(f"Raw MSE Validation Loss: {avg_raw_mse_val_loss.item():.6f}\n")
	file.write(f"Normalized MSE Validation Loss: {avg_norm_mse_val_loss.item():.6f}\n")

print(f"The test has concluded. This model achieved a raw RMSE validation loss of {raw_rmse_val_loss.item():.6f}. A file has been saved to {output_file_path}.")