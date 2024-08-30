# Custom class imports:
import os
import sys

sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/generation')
from generation import simulate_dataset

simulate_dataset(
    nbr=20000,
    store_dir=os.path.join(os.getcwd(), "Data").replace("\\","/"),
    overwrite=False,
    desired_multiplicity=4,
    threshold=1e6,
    visibility_ratio=0.6,
    min_visibility_len=10,
    name="noisy_train_quad",
    save_every=50
)
