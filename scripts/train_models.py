# Custom class imports:
import os
import sys
import torch

sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/ml')
from models import Net2D
from train import run_multiple_trains

default_config = {
    "epochs" : 2000,
    "lr" : 1e-4,
    "data_size" : 10000,
    "batch_size" : 100,
    "rescale" : 1,
    "loss_fn" : torch.nn.MSELoss(),
    "model" : None,
    "param" : None,
    "data_path" : "c:/Users/sneaky/Code/lensing/Data/train/m_2/Videos",
    "generation_df_path" : "c:/Users/sneaky/Code/lensing//Data/train/m_2/train_simulated.pickle"
}

configs_list = [
    {
        "data_size" : 15000,
        "rescale" : 1
    }
]

run_multiple_trains(
    configs_list=configs_list,
    train_dir="Training",
    default_config=default_config,
    return_test_idxs=True
)
