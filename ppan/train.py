from typing import Optional

from ppan.dataset import Rach3Dataset
from ppan.types import PathLike

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


def main(dataset_dir: PathLike, output: Optional[PathLike] = None, **kwargs):
    torch.manual_seed(42)
    dataset = Rach3Dataset(root=dataset_dir)
    weights = MViT_V2_S_Weights.KINETICS400_V1
    model = mvit_v2_s(weights=weights)
    loader = DataLoader(dataset, batch_size=16)

    for batch in loader:
        ...
