import random

import numpy as np
from torch import cuda, manual_seed, device
from torchvision import disable_beta_transforms_warning

device = device("cuda" if cuda.is_available() else "cpu")

# Let's try to have some reproducibility
seed = 42
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
pretrained_model = "MCG-NJU/videomae-base"
