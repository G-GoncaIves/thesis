# Custom class imports:
import os
import sys

sys.path.insert(1, '/home/goncalo/Projects/thesis/utils/generation')
from generation import simulate_dataset

simulate_dataset(
    nbr=50000,
    store_dir=os.path.join(os.getcwd(), "Data").replace("\\","/"),
    overwrite=False,
    desired_multiplicity=2,
    threshold=1e6,
    visibility_ratio=0.6,
    min_visibility_len=10,
    name="test",
    save_every=1
)
