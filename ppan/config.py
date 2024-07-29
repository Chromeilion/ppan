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
environ["WANDB_PROJECT"] = environ.get("PPAN_WANDB_PROJECT_NAME",
                                       "rach3-onset-detector")

# Sample format used for preprocessing.
# [[midi_path, flac_path, video_path, bounding_box, whether to rotate 180,
#   random_resize]]
SAMPLE_TYPE = List[
    Tuple[PathLike, Optional[PathLike], PathLike,
          Optional[tuple[int, int, int, int]], bool, bool, PathLike]
]
TEST_TRAIN_SPLIT = tuple[list[SAMPLE_TYPE], list[SAMPLE_TYPE]]

# Default model resolution
model_crop_resolution = (112, 640)
model_resolution = (96, 512)
model_no_channels = 1

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
processed_horizontal_res = 672

# Base model config
base = {
    "image_size": model_resolution,
    "patch_size": 16,
    "num_channels": model_no_channels,
    "num_frames": 5,
    "tubelet_size": 1,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
    "use_mean_pooling": True,
    "decoder_hidden_size": 384,
    "decoder_num_hidden_layers": 4,
    "decoder_num_attention_heads": 6,
    "decoder_intermediate_size": 1536,
    "norm_pix_loss": True
}
small = {
    "image_size": model_resolution,
    "patch_size": 32,
    "attention_probs_dropout_prob": 0.0,
    "decoder_hidden_size": 384,
    "decoder_intermediate_size": 768,
    "decoder_num_attention_heads": 6,
    "decoder_num_hidden_layers": 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 384,
    "initializer_range": 0.02,
    "intermediate_size": 1536,
    "layer_norm_eps": 1e-12,
    "norm_pix_loss": True,
    "num_attention_heads": 6,
    "num_channels": model_no_channels,
    "tubelet_size": 1,
    "num_frames": 5,
    "num_hidden_layers": 12,
    "qkv_bias": True,
    "use_mean_pooling": True
}
# Default pretraining settings
pretrain_default = {
    "no_epochs": 400,
    "batch_size": 32,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.05,
    "warmup_ratio": 0.1,
    "scheduler_type": "cosine",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "mask_ratio": 0.9,
    "eval_every": 1000,
    "save_every": 1000
}
# Default pretraining settings
finetune_default = {
    "no_epochs": 40,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "scheduler_type": "cosine",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "mask_ratio": 0.9,
    "eval_every": 1000,
    "save_every": 1000
}
