"""
GNN model architectures for network intrusion detection
Contains encoder, decoder, and full autoencoder implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv

def get_gnn_layer(gnn_type, in_dim, out_dim):
    """Factory function to create GNN layers based on type"""
    if gnn_type == "GCN": 
        return GCNConv(in_dim, out_dim)
    elif gnn_type == "SAGE": 
        return SAGEConv(in_dim, out_dim)
    elif gnn_type == "GIN":
        return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
    elif gnn_type == "GAT": 
        return GATConv(in_dim, out_dim, heads=4, concat=False, dropout=0.1)
    else: 
        raise ValueError(f"Unsupported GNN type: {gnn_type}")

class MultiScaleGNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, gnn_type):
        super().__init__()
        self.conv1 = get_gnn_layer(gnn_type, input_dim, hidden_dim)
        self.conv2 = get_gnn_layer(gnn_type, hidden_dim, hidden_dim)
        self.conv3 = get_gnn_layer(gnn_type, hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        h3 = self.conv3(h2, edge_index)
        return h3

class ContrastiveNodeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

class EdgeDecoder(nn.Module):
    def __init__(self, latent_dim, edge_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * latent_dim, edge_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(edge_dim * 2, edge_dim)
        )
        
    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        pair = torch.cat([src, dst], dim=1)
        return self.fc(pair)

class MultiScaleGNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim, gnn_type):
        super().__init__()
        self.encoder = MultiScaleGNNEncoder(input_dim, hidden_dim, latent_dim, gnn_type)
        self.node_decoder = ContrastiveNodeDecoder(latent_dim, hidden_dim, input_dim)
        self.edge_decoder = EdgeDecoder(latent_dim, edge_dim) if edge_dim > 0 else None
        
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.node_decoder(z)
        edge_hat = self.edge_decoder(z, edge_index) if self.edge_decoder else None
        return x_hat, edge_hat, z