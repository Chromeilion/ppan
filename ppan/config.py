import random

import numpy as np
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lets try to have some reproducibility
seed = 45
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Remove annoying warnings
torchvision.disable_beta_transforms_warning()

# Model configuration
pretrained_model = "MCG-NJU/videomae-small-finetuned-kinetics"
main_res = (224, 224)

# Cuda and video_reader backends are much faster than pyav, however,
# torchvision must be compiled from source in order to have support for
# them. They're also only supported for Linux systems.
#
# Also, the cuda backend has a different API for some reason and doesn't
# support some very important features. Honestly I wouldn't be surprised
# if its buggy even, as there's been no update to the incomplete code in
# 2 years.
#
# See here for instructions on getting the CUDA backend working (if you
# hate yourself):
# https://github.com/pytorch/vision/tree/main/torchvision/csrc/io/decoder/gpu
torchvision.set_video_backend("video_reader")
VID_BACKEND = "video_reader"
