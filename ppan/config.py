import random
from os import environ, PathLike
from typing import Union, List, Optional, Tuple, Any

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
model_crop_resolution = (105, 656)
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
    "tubelet_size": 5,
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
    "patch_size": 16,
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
    "tubelet_size": 5,
    "num_frames": 5,
    "num_hidden_layers": 12,
    "qkv_bias": True,
    "use_mean_pooling": True
}

PRETRAINED_MODEL_BASE = "MCG-NJU/videomae-base"


def get_model_size() -> tuple[dict[str, Any], Optional[str]]:
    size = environ.get("PPAN_MODEL_SIZE", "base")
    match size:
        case "base":
            return base, PRETRAINED_MODEL_BASE
        case "small":
            return small, None


# Default pretraining settings
pretrain_default = {
    "no_epochs": 50,
    "batch_size": 512,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.05,
    "warmup_ratio": 0.2,
    "scheduler_type": "cosine",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "mask_ratio": 0.8,
    "eval_every": 500,
    "save_every": 500,
    "grad_clip": 1
}
# Default pretraining settings
finetune_default = {
    "no_epochs": 30,
    "batch_size": 128,
    "learning_rate": 6e-4,
    "weight_decay": 6e-5,
    "warmup_ratio": 2.5/30,  # 2.5 epochs of warmup
    "scheduler_type": "cosine",
    "eval_every": 250,
    "save_every": 250,
    "randaug": True,
    "temporal_jitter": True,
    "spatial_jitter": True,
    "hidden_dropout_prob": 0.01,
    "attention_probs_dropout_prob": 0.01,
    "momentum": 0.9,
    "grad_clip": 1
}
# Two keys don't occur in the dataset, therefore we can set anything for these.
_inf = 2000
class_weights = [
    1.3952e+04, 7.9065e+04, 7.9065e+04, 8.1782e+03, 6.2410e+03, 1.8594e+03,
    2.1852e+03, 1.9196e+03, 1.8741e+03, 1.7960e+03, 8.4462e+02, 7.0917e+02,
    2.6854e+02, 4.8606e+02, 6.6064e+02, 3.1867e+02, 3.2041e+02, 1.4168e+02,
    1.4665e+02, 1.2725e+02, 1.2833e+02, 1.7252e+02, 9.3822e+01, 9.2810e+01,
    5.1570e+01, 6.8144e+01, 8.4461e+01, 5.8277e+01, 6.3203e+01, 3.9854e+01,
    4.6302e+01, 3.5338e+01, 3.5249e+01, 4.3971e+01, 2.8787e+01, 3.4078e+01,
    2.1886e+01, 2.4361e+01, 2.9290e+01, 2.2964e+01, 2.7489e+01, 1.7056e+01,
    2.5317e+01, 2.0924e+01, 1.9297e+01, 3.0272e+01, 1.9986e+01, 2.4304e+01,
    1.7109e+01, 2.4794e+01, 2.9040e+01, 2.3743e+01, 3.0071e+01, 1.8820e+01,
    3.3394e+01, 2.6925e+01, 2.9727e+01, 4.5560e+01, 3.7076e+01, 5.8921e+01,
    3.5638e+01, 6.2831e+01, 9.8621e+01, 7.9161e+01, 9.6372e+01, 6.4751e+01,
    1.3458e+02, 1.1000e+02, 1.5682e+02, 2.1842e+02, 1.8446e+02, 2.9811e+02,
    1.2011e+02, 3.6875e+02, 8.3568e+02, 4.2988e+02, 9.0955e+02, 3.9236e+02,
    1.8667e+03, 3.9433e+02, 2.7896e+03, 8.0396e+03, 5.3293e+03, 5.2710e+04,
    2.1562e+04, 4.7439e+05, _inf, _inf
 ]
