import random
from os import environ, PathLike
from typing import Union, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from torch import cuda, manual_seed, device

load_dotenv()

# Let's try to have some reproducibility
seed: int = int(environ.get("PPAN_SEED", 42))
manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

frame_hd5_file = "r3s_processed.hdf5"

# Useful for typing
PathLike = Union[str, bytes, PathLike]

# Automatically choose a device
device = device("cuda" if cuda.is_available() else "cpu")

# Wandb logging settings
run_name = environ.get("PPAN_WANDB_PROJECT_NAME", "ppan_run")
environ["WANDB_PROJECT"] = run_name

# Sample format used for preprocessing.
# [[midi_path, flac_path, video_path, bounding_box, whether to rotate 180,
#   random_resize]]
SAMPLE_TYPE = List[
    Tuple[PathLike, Optional[PathLike], PathLike,
          Optional[tuple[int, int, int, int]], bool, bool, PathLike]
]
TEST_TRAIN_SPLIT = tuple[list[SAMPLE_TYPE], list[SAMPLE_TYPE]]

# Default model resolution
model_crop_resolution = (122, 720)
# The resolution is set to specifically have the same number of pixels as
# the pretraining resolution.
model_resolution = (64, 784)
model_no_channels = 3
model_no_frames = 6
frame_stride = 1
frames_per_ds_sample = 6

# The number of keys on a (regular) piano
num_labels = 88

# Number of elements to keep in the eval set to make training faster.
max_eval_steps = 500

# The fps of all videos we're working with
fps = 30
temporal_res = 1/fps

# The size of a processed sample before applying final model specific
# preprocessing
processed_temporal_size = 64 * temporal_res
processed_horizontal_res = 720

PRETRAINED_MODEL_BASE = environ.get("PPAN_MODEL_BASE_LOC", "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-b/pytorch_model.bin")
PRETRAINED_MODEL_SMALL = environ.get("PPAN_MODEL_SMALL_LOC", "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth")

# Default pretraining settings
no_epochs = 10
finetune_default = {
    "architecture": "vit_s",
    "optimizer": "adamw",
    "no_epochs": no_epochs,
    "batch_size": 96,
    "lr_sgd": 0.25,
    "lr_adamw": 1e-3,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 0.05,
    "warmup_ratio": 2.5/no_epochs,
    "scheduler_type": "cosine",
    "eval_every": 250,
    "save_every": 250,
    "frames_only": False,
    "onsets_only": False,
    "spatial_jitter": False,
    "rand_erase": False,
    "rand_rotate": False,
    "rotate_180": False,
    "gaussian_noise": False,
    "grayscale": True,
    "color_jitter": False,
    "dropout": 0.,
    "drop_path": 0.,
    "momentum": 0.9,
    "grad_clip": 1.,
    "do_smoothing_frame": False,
    "do_smoothing_onset": False,
    "confidence_frame": None,
    "confidence_onset": None,
    "do_mixup": False,
    "mixup_alpha": 0.2,
    "stride": 1,
    "window_size": 6,
    "lenience": 1,
    "image_size": None
}
