"""
Graph construction functions for GNN-based network intrusion detection
Handles leakage-free graph building with node and edge feature aggregation
"""
import torch
import numpy as np

def construct_leakage_free_graph(df, node_features, edge_features, train_ips=None, mode='train'):
    """
    Construct graph without data leakage between train and test sets
    
    Args:
        df: Network flow dataframe
        node_features: List of features for node representation
        edge_features: List of features for edge representation  
        train_ips: Set of training IP addresses
        mode: 'train' or 'full' - determines which flows to use
        
    Returns:
        Tuple of (node_features, edge_features, edge_index, ip_to_idx, edge_to_flows)
    """
    print(f" Constructing leakage-free graph in {mode} mode...")
    
    # Filter flows based on mode to prevent leakage
    if mode == 'train' and train_ips is not None:
        # Training mode: only benign flows from training IPs
        df_filtered = df[
            (df['IPV4_SRC_ADDR'].isin(train_ips)) & 
            (df['IPV4_DST_ADDR'].isin(train_ips)) &
            (df['Label'] == 0)
        ].copy()
        print(f"    Training mode: Using {len(df_filtered)} BENIGN flows only")
    else:
        # Evaluation mode: all flows
        df_filtered = df.copy()
        print(f"    Evaluation mode: Using {len(df_filtered)} flows")
    
    # Create consistent node indexing using ALL IPs from original data
    all_src_ips = set(df['IPV4_SRC_ADDR'].unique())
    all_dst_ips = set(df['IPV4_DST_ADDR'].unique()) 
    all_ips = sorted(list(all_src_ips.union(all_dst_ips)))
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    
    print(f"    Graph nodes: {len(all_ips)} IPs")
    
    # Aggregate node features for each IP
    X_nodes = _aggregate_node_features(df_filtered, all_ips, node_features)
    
    # Construct edges and edge features
    X_edges, edge_index, edge_to_flows = _construct_edges(df_filtered, edge_features, ip_to_idx)
    
    print(f"    Graph edges: {edge_index.shape[1]}")
    
    return X_nodes, X_edges, edge_index, ip_to_idx, edge_to_flows

def _aggregate_node_features(df_filtered, all_ips, node_features):
    """Aggregate node features for each IP address"""
    node_data = []
    
    for ip in all_ips:
        # Get all flows involving this IP
        ip_flows = df_filtered[
            (df_filtered['IPV4_SRC_ADDR'] == ip) | 
            (df_filtered['IPV4_DST_ADDR'] == ip)
        ]
        
        if len(ip_flows) == 0:
            # No flows for this IP in filtered data - use zero features
            node_feat = np.zeros(len(node_features) * 4)
        else:
            # Aggregate features: mean, std, min, max
            feat_values = ip_flows[node_features].values
            node_feat = np.concatenate([
                np.mean(feat_values, axis=0),
                np.std(feat_values, axis=0),
                np.min(feat_values, axis=0),
                np.max(feat_values, axis=0)
            ])
        
        node_data.append(node_feat)
    
    return np.array(node_data)

def _construct_edges(df_filtered, edge_features, ip_to_idx):
    """Construct edges and aggregate edge features"""
    # Group flows by IP pairs for edge construction
    df_filtered['_pair_key'] = list(zip(df_filtered['IPV4_SRC_ADDR'], df_filtered['IPV4_DST_ADDR']))
    pair_groups = df_filtered.groupby('_pair_key')
    
    edge_list = []
    edge_features_list = []
    edge_to_flows = {}
    
    for (src_ip, dst_ip), group in pair_groups:
        if src_ip in ip_to_idx and dst_ip in ip_to_idx:
            src_idx = ip_to_idx[src_ip]
            dst_idx = ip_to_idx[dst_ip]
            
            # Add bidirectional edges
            edge_list.extend([(src_idx, dst_idx), (dst_idx, src_idx)])
            
            # Aggregate edge features: mean, sum, count
            edge_feat_values = group[edge_features].values
            edge_feat = np.concatenate([
                np.mean(edge_feat_values, axis=0),
                np.sum(edge_feat_values, axis=0),
                [len(group)]  # Flow count
            ])
            
            # Same features for both directions
            edge_features_list.extend([edge_feat, edge_feat])
            
            # Map edges to flow indices for later analysis
            edge_to_flows[(src_idx, dst_idx)] = group.index.tolist()
            edge_to_flows[(dst_idx, src_idx)] = group.index.tolist()
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.zeros((2, 0), dtype=torch.long)
    X_edges = np.array(edge_features_list) if edge_features_list else np.zeros((0, len(edge_features) * 2 + 1))
    
    return X_edges, edge_index, edge_to_flows
    