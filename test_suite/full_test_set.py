import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ========== IMPORTS ==========
import torch
from stgcn import STGCN
from utilities.preprocessing import load_dataset_for_stgcn, train_test_split, subset_data, shuffle_dataset, get_all_velocity_data
from mechanisms.normalisation import ZScoreNormaliser
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import numpy as np
from utilities.plotting import plot_ground_truth_and_predictions



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

# Provide a base date
base_date = datetime(2000, 12, 2)



# ========== CREATE Time Truth Dictionary ==========
# Dynamically creates a dictionary to hold ground truth vs. predicted values for each time step
data = {
    i: {
		"pred_avgs": [],
        "truth_avgs": []
    }
    for i in range(1, 289)
}



# ========== LOAD DATA ==========
dataset = load_dataset_for_stgcn(window_size=12, forecast_horizon=FORECAST_HORIZON)
all_velocity_values = get_all_velocity_data()
unused_train_set, test_set = train_test_split(dataset)
normaliser = ZScoreNormaliser(all_velocity_values)



# ========== TESTING ==========
model.eval()
norm_mse_val_loss_total = 0
raw_mse_val_loss_total = 0
time_of_day_identifier = 1
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

		# ----- Gather up the predictions and ground truths -----
		preds = np.array(raw_y_val_hat_single)
		truths = np.array(raw_y_val_target)

		# ----- Find the average across all preds and truths -----
		pred_avg = np.mean(preds)
		truth_avg = np.mean(truths)

		# ----- Append the averages to the dictionary -----
		data[time_of_day_identifier]["pred_avgs"].append(pred_avg)
		data[time_of_day_identifier]["truth_avgs"].append(truth_avg)

		# ----- Deal with incrementing the time of day identifier -----
		time_of_day_identifier = time_of_day_identifier + 1
		if time_of_day_identifier > 288:
			time_of_day_identifier = 1

# ----- RMSE calculation -----
avg_raw_mse_val_loss = raw_mse_val_loss_total / (time_step + 1)
avg_norm_mse_val_loss = norm_mse_val_loss_total / (time_step + 1)

raw_rmse_val_loss = torch.sqrt(avg_raw_mse_val_loss)
norm_rmse_val_loss = torch.sqrt(avg_norm_mse_val_loss)

# ----- Generate times in 5-min intervals -----
times = [base_date + timedelta(minutes=5 * i) for i in range(288)]

# ----- Average predictions at each time step -----
avg_pred_avgs_over_time = []
for i in range(1, 289):
		pred_avgs = data[i]["pred_avgs"]
		avg_pred_avgs = np.mean(pred_avgs)
		avg_pred_avgs_over_time.append(avg_pred_avgs)

# ----- Average truths at each time step -----
avg_truth_avgs_over_time = []
for i in range(1, 289):
		truth_avgs = data[i]["truth_avgs"]
		avg_truth_avgs = np.mean(truth_avgs)
		avg_truth_avgs_over_time.append(avg_truth_avgs)

# ----- Plotting the ground truth and predictions -----
plot_ground_truth_and_predictions(
	times = times,
	ground_truth= avg_truth_avgs_over_time,
	predictions= avg_pred_avgs_over_time,
	test_number = "3.2",
	folder = 'graphs',
	subfolder = 'full_test_set_validation'
)



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
