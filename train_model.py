# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, split_dataset
from stgcn import train_model, evaluate_model
from stgcn import STGCN
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # optional progress bar


# ========== DATA MANAGEMENT ==========
dataset = load_dataset_for_stgcn(window_size=12)
train, test, val = split_dataset(dataset)


# ========== SETUP ==========
model = STGCN(in_channels=1, hidden_channels=64, out_channels=1, num_nodes=288)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 10


# ========== TRAINING LOOP ==========
for epoch in range(1, num_epochs + 1):
    train_loss = train_model(model, train, optimizer, loss_fn)
    val_loss = evaluate_model(model, val, loss_fn)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")