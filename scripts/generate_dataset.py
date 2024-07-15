# Custom class imports:
import os
from utils.generation import simulate_dataset

simulate_dataset(
    nbr=10,
    store_dir=os.path.join(os.getcwd(), "Data").replace("\\","/"),
    overwrite=False,
    desired_multiplicity=2,
    threshold=1e6,
    visibility_ratio=0.6,
    min_visibility_len=10,
    name="test",
    save_every=1
)
