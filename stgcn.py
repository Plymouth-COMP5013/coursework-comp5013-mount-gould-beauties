# Author: Reef Lakin
# Last Modified: 29.04.2025
# Description: A PyTorch implementation of the Spatio-Temporal Graph Convolutional Network (ST-GCN) for traffic forecasting. 
# Helpful Links:
#   https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/stgcn.py
#   https://github.com/benedekrozemberczki/pytorch_geometric_temporal

# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv

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
    """


    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, kernel_size=3):
        super(STGCN, self).__init__()

        # The first spatio-temporal convolutional layer
        self.st_conv1 = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=3,  # order of Chebyshev polynomial
            normalization="sym"
        )

        # The second spatio-temporal convolutional layer
        self.st_conv2 = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=3,  # order of Chebyshev polynomial
            normalization="sym"
        )

        # Final convolutional layer to reduce to output features
        self.final_conv = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )

        self.fc = torch.nn.Linear(num_nodes * out_channels, num_nodes)

    def forward(self, x, edge_index):
        # x shape: [batch_size, num_nodes, num_features, sequence_length]
        x = self.st_conv1(x, edge_index)
        x = F.relu(x)

        x = self.st_conv2(x, edge_index)
        x = F.relu(x)

        # Final conv to reduce to output features
        x = self.final_conv(x.unsqueeze(1))  # Add channel dimension: [B, 1, N, F, T]
        x = x.squeeze(1)  # Remove channel dimension

        # Flatten and apply FC layer
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
