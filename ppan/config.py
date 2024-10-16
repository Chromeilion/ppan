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
#model_resolution = (64, 784) # The image gets slightly stretched.
model_resolution = (64, 704)
model_no_channels = 3
model_no_frames = 6
frame_stride = 1
frames_per_ds_sample = 6

# For image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The number of keys on a (regular) piano
num_labels = 88

# Number of elements to keep in the eval set to make training faster.
max_eval_steps = 500

# The fps of all videos we're working with
fps = 30
temporal_res = 1/fps

# The size of a processed sample before applying final model specific
# preprocessing
processed_temporal_size = 7 * temporal_res
processed_horizontal_res = 720

PRETRAINED_MODEL_BASE = "MCG-NJU/videomae-base"
PRETRAINED_MODEL_SMALL = "MCG-NJU/videomae-small-finetuned-kinetics"

# Default pretraining settings
finetune_default = {
    "no_epochs": 10,
    "batch_size": 64,
    "lr_sgd": 3e-4,
    "lr_adamw": 5e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 0.03,
    "warmup_ratio": 1/10,  # 0.5 epochs of warmup
    "scheduler_type": "cosine",
    "eval_every": 250,
    "save_every": 250,
    "randaug": True,
    "temporal_jitter": True,
    "spatial_jitter": True,
    "rand_erase": False,
    "rotate_180": True,
    "dropout": 0.,
    "momentum": 0.9,
    "grad_clip": 1,
    "mask_percentage": None
}
