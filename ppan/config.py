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
d_model = 768
pretrained_encoder = "google/vit-base-patch16-224"
main_res = (448, 448)
pretrained_playing_det_vit = "google/vit-base-patch16-224-in21k"
det_res = (224, 224)
# Default sequence length for midi tokens sequences.
seq_len = 110

# Choosing a video backend automatically. cuda and video_reader are much
# faster than pyav, however, torchvision must be compiled from source in order
# to have support for them. They're also only supported for Linux systems.
#
# See here for instructions on getting it working:
# https://github.com/pytorch/vision/tree/main/torchvision/csrc/io/decoder/gpu
try:
    if device.type == "cuda":
        torchvision.set_video_backend("pyav")
        VID_BACKEND = "pyav"
    else:
        torchvision.set_video_backend("video_reader")
        VID_BACKEND = "video_reader"
except RuntimeError:
    torchvision.set_video_backend("pyav")
    VID_BACKEND = "pyav"
