# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, train_test_subset
from stgcn import STGCN
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12)
train_subet, test_subset = train_test_subset(dataset, subset_ratio=0.25)


# ========== SETUP ==========
model = STGCN(in_channels=1, hidden_channels=64, out_channels=1, num_nodes=228)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 30


# ========== TRAINING LOOP ==========
model.train()

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
	loss = 0
	for time, snapshot in tqdm(enumerate(train_subet), desc="Training Batches", leave=False):
		# print(snapshot)
		# print(time)
		# print(snapshot.x.shape)
		x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
		# print(x.shape)
		y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
		# print(f"y_hat shape: {y_hat.shape}")         # likely [1, 1, 288, 1]
		# print(f"snapshot.y shape: {snapshot.y.shape}")  # this is the question mark
		# Take the first timestep from y_hat
		y_hat_single = y_hat[:, 0, :, :].squeeze()   # shape: [228]
		target = snapshot.y.view(-1)                 # shape: [228]

		loss = loss + torch.mean((y_hat_single - target) ** 2)

	loss = loss / (time + 1)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


# ========== EVALUATION ==========
model.eval()

total_loss = 0
with torch.no_grad():
	for time, snapshot in tqdm(enumerate(test_subset), desc="Testing Batches", leave=False):
		x = snapshot.x.T.unsqueeze(0).unsqueeze(-1)
		y_hat = model(x, snapshot.edge_index, snapshot.edge_weight)
		y_hat_single = y_hat[:, 0, :, :].squeeze()
		target = snapshot.y.view(-1)

		loss = loss + torch.mean((y_hat_single - target) ** 2)

	total_loss += loss.item()
	total_loss = total_loss / (time + 1)
	print(f"Test Loss: {total_loss:.4f}")


# ========== SAVE MODEL ==========

# Generate a Unix timestamp for the model filename
unix_timestamp = time.time()

# Convert the timestamp to a human-readable format
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(unix_timestamp))

# Save the model state dictionary
torch.save(model.state_dict(), f"stgcn_model_{timestamp}.pth")

# Save the model architecture
torch.save(model, f"stgcn_model_architecture_{timestamp}.pth")
