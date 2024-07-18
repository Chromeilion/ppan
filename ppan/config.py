import random
from os import environ, PathLike
from typing import Union, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from torch import cuda, manual_seed, device

load_dotenv()

device = device("cuda" if cuda.is_available() else "cpu")

# Wandb logging settings
environ["WANDB_PROJECT"] = environ.get("PPAN_WANDB_PROJECT_NAME",
                                       "rach3-onset-detector")

# [[midi_path, flac_path, video_path, bounding_box, whether to rotate 180,
#   random_resize]]
SAMPLE_TYPE = List[
    Tuple[PathLike, Optional[PathLike], PathLike,
          Optional[tuple[int, int, int, int]], bool, bool, PathLike]
]
TEST_TRAIN_SPLIT = tuple[list[SAMPLE_TYPE], list[SAMPLE_TYPE]]

# Let's try to have some reproducibility
seed: int = int(environ.get("PPAN_SEED", 42))
manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model_crop_resolution = (112, 640)
model_resolution = (96, 512)

# The number of keys on a (regular) piano
num_labels = 88

# The fps of all videos we're working with
fps = 30
temporal_res = 1/fps

# The size of a processed sample before applying final model specific
# preprocessing
processed_temporal_size = 7 * temporal_res
processed_horizontal_res = 672

# Model configuration
pretrained_model: str = environ.get("PPAN_PRETRAINED_MODEL",
                                    "MCG-NJU/videomae-small-finetuned-kinetics")

PathLike = Union[str, bytes, PathLike]

model_config = {
    "image_size": model_resolution,
    "patch_size": 16,
    "num_channels": 1,
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
