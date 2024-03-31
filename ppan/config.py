import random
from os import environ

import numpy as np
from dotenv import load_dotenv
from torch import cuda, manual_seed, device
from torchvision import disable_beta_transforms_warning

load_dotenv()

device = device("cuda" if cuda.is_available() else "cpu")

# Wandb logging settings
environ["WANDB_PROJECT"] = environ.get("PPAN_WANDB_PROJECT_NAME",
                                       "rach3-onset-detector")

# Let's try to have some reproducibility
seed: int = int(environ.get("PPAN_SEED", 42))
manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# The number of keys on a (regular) piano
num_labels = 88

# The fps of all videos we're working with
fps = 30
temporal_res = 1/fps

# Remove annoying warnings
disable_beta_transforms_warning()

# Model configuration
pretrained_model: str = environ.get("PPAN_PRETRAINED_MODEL",
                                    "MCG-NJU/videomae-small-finetuned-kinetics")
