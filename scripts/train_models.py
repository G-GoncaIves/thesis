# Custom class imports:
import os
import sys
import torch

<<<<<<< HEAD
sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/ml')
from models import Net3D, Net2D, Net3D_v1, Net3D_v2, Net3D_v3
=======
sys.path.insert(1, 'c:/Users/sneaky/Code/thesis/utils/ml')
from models import Net3D_wDO
>>>>>>> c07d1f4a33202a2e02ec468c5ffed79c9262f483
from train import run_multiple_trains

default_config = {
    "epochs" : 2000,
    "lr" : 1e-4,
    "data_size" : 10000,
    "batch_size" : 100,
    "rescale" : 1,
    "loss_fn" : torch.nn.MSELoss(),
    "patience" : 300,
    "min_delta" : 0,
    "model" : None,
    "no_td" : None,
    "param" : None,
<<<<<<< HEAD
    "data_path" : "/home/goncalo/Projects/thesis/scripts/Data/noisy_train/m_2/Videos",
    "generation_df_path" : "/home/goncalo/Projects/thesis/scripts/Data/noisy_train/m_2/noisy_train_simulated.pickle"
=======
    "data_path" : "/home/sneaky/Code/lensing/Data/noisy_offset/m_2/Videos",
    "generation_df_path" : "/home/sneaky/Code/lensing/Data/noisy_offset/Data/test/m_2/test_simulated.pickle"
>>>>>>> c07d1f4a33202a2e02ec468c5ffed79c9262f483
}

configs_list = [
   {
        "param" : ["im_td"],
        "model" : Net3D_v2(out_dim=1),
        "epochs" : 1000,
        "data_size" : 15000,
<<<<<<< HEAD
        "no_td" : False,
        "rescale" : 1/200,
        "patience" : 50
=======
        "rescale" : 1,
        "param" : ["theta_e"],
        "model" : Net3D_wDO(out_dim=1)
>>>>>>> c07d1f4a33202a2e02ec468c5ffed79c9262f483
    }
]

run_multiple_trains(
    configs_list=configs_list,
    train_dir="Training",
    default_config=default_config,
    return_test_idxs=True
)
