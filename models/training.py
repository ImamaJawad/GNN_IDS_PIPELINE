"""
Training functions for GNN autoencoder models
Handles model initialization, training loop, and evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from .architectures import MultiScaleGNNAutoencoder

def train_gnn_autoencoder(X_nodes, X_edges, edge_index, train_ips, ip_to_idx, model_params, gnn_type="GCN"):
    """
    Train GNN autoencoder model
    
    Args:
        X_nodes: Node features array
        X_edges: Edge features array  
        edge_index: Edge connectivity tensor
        train_ips: Set of training IP addresses
        ip_to_idx: Mapping from IP to node index
        model_params: Dictionary of model hyperparameters
        gnn_type: Type of GNN architecture
        
    Returns:
        Tuple of (trained_model, training_losses, scalers)
    """
    import time
    
    print(f"Training GNN Autoencoder ({gnn_type})...")
    
    # *** RUNTIME METRICS: Start training timer ***
    start_train = time.time()
    
    # *** FIX 1: Handle infinity values BEFORE scaling ***
    print(" Cleaning infinity values...")
    
    # Replace inf/-inf with NaN first
    X_nodes = np.nan_to_num(X_nodes, nan=0.0, posinf=np.nan, neginf=np.nan)
    X_edges = np.nan_to_num(X_edges, nan=0.0, posinf=np.nan, neginf=np.nan) if X_edges.shape[0] > 0 else X_edges
    
    # Fill NaN values with column medians (more robust than mean)
    for col in range(X_nodes.shape[1]):
        col_data = X_nodes[:, col]
        if np.isnan(col_data).any():
            median_val = np.nanmedian(col_data)
            if np.isnan(median_val):  # If all values were inf/nan
                median_val = 0.0
            X_nodes[:, col] = np.nan_to_num(col_data, nan=median_val)
    
    if X_edges.shape[0] > 0:
        for col in range(X_edges.shape[1]):
            col_data = X_edges[:, col]
            if np.isnan(col_data).any():
                median_val = np.nanmedian(col_data)
                if np.isnan(median_val):
                    median_val = 0.0
                X_edges[:, col] = np.nan_to_num(col_data, nan=median_val)
    
    # Clip extreme values to safe range
    X_nodes = np.clip(X_nodes, -1e6, 1e6)
    if X_edges.shape[0] > 0:
        X_edges = np.clip(X_edges, -1e6, 1e6)
    
    print(f" Data cleaning completed. Shapes: X_nodes={X_nodes.shape}, X_edges={X_edges.shape}")
    
    # Scale features (now safe from infinity values)
    scaler_node = StandardScaler()
    scaler_edge = StandardScaler()
    
    X_nodes_scaled = scaler_node.fit_transform(X_nodes)
    X_edges_scaled = scaler_edge.fit_transform(X_edges) if X_edges.shape[0] > 0 else X_edges
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_nodes_scaled, dtype=torch.float).to(device)
    edge_attr = torch.tensor(X_edges_scaled, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    
    # Create model
    model = MultiScaleGNNAutoencoder(
        X.shape[1], 
        model_params['hidden_dim'], 
        model_params['latent_dim'], 
        edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0, 
        gnn_type
    ).to(device)
    
    print(f"Model architecture: {X.shape[1]} → {model_params['hidden_dim']} → {model_params['latent_dim']}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Training node indices (only benign IPs)
    train_node_indices = [ip_to_idx[ip] for ip in train_ips if ip in ip_to_idx]
    
    # Training loop
    model.train()
    training_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    print(" Starting training...")
    for epoch in range(model_params['max_epochs']):
        optimizer.zero_grad()
        x_hat, edge_hat, z = model(X, edge_index)
        
        # Node reconstruction loss (only on training nodes)
        loss_node = F.mse_loss(x_hat[train_node_indices], X[train_node_indices])
        
        # Edge reconstruction loss
        loss_edge = F.mse_loss(edge_hat, edge_attr) if edge_hat is not None and edge_attr.shape[0] > 0 else 0
        
        # Contrastive loss for representation learning
        z_train = z[train_node_indices]
        if len(z_train) > 1:
            z_norm = F.normalize(z_train, dim=1)
            similarity_matrix = torch.mm(z_norm, z_norm.t())
            contrastive_loss = torch.mean(torch.triu(similarity_matrix, diagonal=1)**2)
        else:
            contrastive_loss = 0
        
        # Total loss
        total_loss = loss_node + 0.1 * (loss_edge if isinstance(loss_edge, torch.Tensor) else 0) + 0.01 * contrastive_loss
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        training_losses.append(total_loss.item())
        
        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= model_params['early_stopping_patience']:
            print(f" Early stopping at epoch {epoch}")
            break
            
        if epoch % 25 == 0:
            node_loss_val = loss_node.item()
            edge_loss_val = loss_edge.item() if isinstance(loss_edge, torch.Tensor) else loss_edge
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.6f} | Node: {node_loss_val:.6f} | Edge: {edge_loss_val:.6f}")
    
    # *** RUNTIME METRICS: End training timer ***
    end_train = time.time()
    
    print(f" Training completed after {len(training_losses)} epochs in {end_train - start_train:.2f}s")
    
    scalers = {'node': scaler_node, 'edge': scaler_edge}
    
    # *** RUNTIME METRICS: Store timing info in scalers for access in main.py ***
    scalers['timing'] = {
        'start_train': start_train,
        'end_train': end_train,
        'train_duration': end_train - start_train
    }
    
    return model, training_losses, scalers