# Custom class imports:
import os
import sys
import torch
import numpy as np

sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/ml')
from models import Net3D_Dropout
from train import run_multiple_trains

default_config = {
    "epochs" : 2000,
    "lr" : 1e-4,
    "data_size" : 10000,
    "batch_size" : 100,
    "rescale_labels" : 1,
    "loss_fn" : torch.nn.MSELoss(),
    "patience" : 300,
    "min_delta" : 0,
    "model" : None,
    "no_td" : None,
    "param" : None,
    "normalize_data" : False,
    "data_path" : "/home/goncalo/Projects/thesis/scripts/Data/noisy_train/m_2/Videos",
    "generation_df_path" : "/home/goncalo/Projects/thesis/scripts/Data/noisy_train/m_2/noisy_train_simulated.pickle"
}

grid_search_dict =    {
    "param" : ["theta_e"],
    "model" : None,
    "epochs" : 500,
    "data_size" : 1500,
    "no_td" : False,
    "loss_fn" : torch.nn.GaussianNLLLoss(),
    "patience" : 100,
	"normalize_data" : False
}

configs_list = []

for dpr in np.linspace(0,1,num=5):
    grid_search_dict["model"] = Net3D_Dropout(out_dim=2, dropout_rate=dpr)
    configs_list.append(grid_search_dict)

run_multiple_trains(
    configs_list=configs_list,
    train_dir="Training",
    default_config=default_config,
    return_test_idxs=True
)
