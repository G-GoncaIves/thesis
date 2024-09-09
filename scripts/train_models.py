# Custom class imports:
import os
import sys
import torch

sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/ml')
from models import Net3D_Positive, Net3D
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

configs_list = [
   {
        "param" : ["im_td"],
        "model" : Net3D_Positive(out_dim=1),
        "epochs" : 1000,
        "data_size" : 15000,
        "no_td" : False,
        "rescale" : 1/200,
        "patience" : 100,
	"normalize_data" : True
    },
    {
        "param" : ["im_td"],
        "model" : Net3D(out_dim=1),
        "epochs" : 1000,
        "data_size" : 15000,
        "no_td" : False,
        "rescale" : 1/200,
        "patience" : 100,
        "normalize_data" : True
    }
]

run_multiple_trains(
    configs_list=configs_list,
    train_dir="Training",
    default_config=default_config,
    return_test_idxs=True
)
