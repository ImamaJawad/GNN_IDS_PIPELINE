"""
Configuration file for GNN-based Network Intrusion Detection System
Contains dataset configurations, model parameters, and experimental settings
"""
import torch
import numpy as np
import random
import gc

# Reproducibility settings
RANDOM_SEED = 123

def reset_torch(seed=RANDOM_SEED):
    """Reset all random states for reproducible experiments"""
    gc.collect()
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset configurations
DATASET_CONFIGS = {
    "CICIDS-2018-v2": {
        "path": "/kaggle/input/2018cic-v2/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'L7_PROTO'],
        "edge_features": ['IN_BYTES', 'IN_PKTS', 'OUT_BYTES']
    },
    "UNSW-NB15-v2": {
        "path": "/kaggle/input/unsw-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'L4_SRC_PORT', 'TCP_FLAGS'],
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'IN_PKTS']
    },
    "ToN-IoT-v2": {
        "path": "/kaggle/input/ton-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'L4_SRC_PORT', 'TCP_FLAGS'],
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES']
    },
    "BoT-IoT-v2": {
        "path": "/kaggle/input/bot-v2/befb58edf3428167_MOHANAD_A4706/data/NF-BoT-IoT-v2.csv",
        "node_features": ['PROTOCOL', 'L4_SRC_PORT', 'TCP_FLAGS'],
        "edge_features": ['IN_BYTES', 'IN_PKTS']
    },
    "CICIDS-2018-v1": {
        "path": "/kaggle/input/2018-v1-cicids/88a47ba2ab64258e_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'L7_PROTO'],
        "edge_features": ['IN_BYTES', 'IN_PKTS', 'OUT_BYTES']
    },
    "UNSW-NB15-v1": {
        "path": "/kaggle/input/nf-unsw-nb15-netflowversion1/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL','L4_SRC_PORT', 'TCP_FLAGS'],
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'IN_PKTS']
    },
    "ToN-IoT-v1": {
        "path": "/kaggle/input/nf-ton-iot/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv",        
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'L4_SRC_PORT', 'TCP_FLAGS'],  # REVERT!
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES']
    },
    "BoT-IoT-v1": { 
        "path": "/kaggle/input/nf-bot-iot-kaggle1/de2c6f75dd50d933_MOHANAD_A4706/data/NF-BoT-IoT.csv",

        "node_features": ['TCP_FLAGS', 'PROTOCOL', 'L4_SRC_PORT'],  # KEEP SAME
        "edge_features": ['IN_BYTES', 'IN_PKTS', 'OUT_PKTS']  # KEEP SAME
    },
    "CICIDS-2018-v3": {
        "path": "/kaggle/input/cic-nf-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'L7_PROTO'],
        "edge_features": ['IN_BYTES', 'IN_PKTS', 'OUT_BYTES']
    },
    "UNSW-NB15-v3": {
        "path": "/kaggle/input/unsw-v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv",
        "node_features": ['L4_DST_PORT', 'PROTOCOL','L4_SRC_PORT', 'TCP_FLAGS'],
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'IN_PKTS']
    },
    "ToN-IoT-v3": {
        "path": "/kaggle/input/ton-v3/02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv",        
        "node_features": ['L4_DST_PORT', 'PROTOCOL', 'L4_SRC_PORT', 'TCP_FLAGS'],  # REVERT!
        "edge_features": ['FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES']
    },
    "BoT-IoT-v3": { 
        "path": "/kaggle/input/bot-v3/d509c9db7490cf92_NFV3DATA-A11964_A11964/data/NF-BoT-IoT-v3.csv",
        "node_features": ['TCP_FLAGS', 'PROTOCOL', 'L4_SRC_PORT'],  # KEEP SAME
        "edge_features": ['IN_BYTES', 'IN_PKTS', 'OUT_PKTS']  # KEEP SAME
    }

}

# Model hyperparameters
MODEL_PARAMS = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "dropout": 0.1,
    "max_epochs": 300,
    "early_stopping_patience": 50
}

# Experiment settings
EXPERIMENT_SETTINGS = {
    "max_total_rows": 1000000,
    "chunk_size": 20000,
    "max_filtered_rows": 500000,
    "test_size": 0.15
}