# Author: Reef Lakin
# Last Modified: 29.04.2025
# Description: A PyTorch implementation of the Spatio-Temporal Graph Convolutional Network (ST-GCN) for traffic forecasting. 
# Helpful Links:
#   https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/stgcn.py
#   https://github.com/benedekrozemberczki/pytorch_geometric_temporal
#   https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting

# ========== IMPORTS ==========
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


# ========== STGCN Model Class Definition ==========
class STGCN(torch.nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network (ST-GCN) for traffic forecasting.
    This model uses two layers of spatio-temporal convolutional layers followed by a final convolutional layer
    to reduce the output features to the desired number of output channels.

    Args:
        in_channels (int): Number of input channels (features).
        hidden_channels (int): Number of channels for the spatial convolutional layers.
        out_channels (int): Number of output channels.
        num_nodes (int): Number of nodes in the graph.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        K (int, optional): Order of Chebyshev polynomial. Default is 3.
    """


    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, kernel_size=3, K=3):
        super(STGCN, self).__init__()

        self.stconv1 = STConv(num_nodes, in_channels, hidden_channels, hidden_channels, kernel_size, K)
        self.stconv2 = STConv(num_nodes, hidden_channels, hidden_channels, hidden_channels, kernel_size, K)
        self.final_linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.stconv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.stconv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.final_linear(x)
        return x


# ========== TRAINING Method ==========
def train_model(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for snapshot in loader:
        x = snapshot.x.unsqueeze(0)  # shape: [1, T, N, F]
        y = snapshot.y.unsqueeze(0)  # shape: [1, N]
        edge_index = snapshot.edge_index
        edge_weight = snapshot.edge_weight

        # x = x.permute(0, 1, 2, 3)
        out = model(x, edge_index, edge_weight).squeeze(-1)  # shape: [1, N]

        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / loader.snapshot_count


# ========== EVALUATION Method ==========
def evaluate_model(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for snapshot in loader:
            x = snapshot.x.unsqueeze(0)
            y = snapshot.y.unsqueeze(0)
            edge_index = snapshot.edge_index
            edge_weight = snapshot.edge_weight

            # x = x.permute(0, 1, 2, 3)
            out = model(x, edge_index, edge_weight).squeeze(-1)
            loss = loss_fn(out, y)
            total_loss += loss.item()
    return total_loss / loader.snapshot_count
