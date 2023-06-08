import torch
import torchvision
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lets at least try to have some reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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
        torchvision.set_video_backend("cuda")
        VID_BACKEND = "cuda"
    else:
        torchvision.set_video_backend("video_reader")
        VID_BACKEND = "video_reader"
except RuntimeError:
    torchvision.set_video_backend("pyav")
    VID_BACKEND = "pyav"
