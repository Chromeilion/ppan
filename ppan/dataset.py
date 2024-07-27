import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypedDict

import torch
import torch.nn as nn
import torchvision.tv_tensors
from torch.utils.data import Dataset


class OutputMap(TypedDict):
    """Template for a dictionary that controls the names and presence of the
    outputs in a PPANDataset. If one of these is set to None, then it never
    gets loaded.
    """
    vid: Optional[str]
    lab: Optional[str]


class BaseVideoProcessor(nn.Module, ABC):
    """
    Base video processor abstract class. Can be used to customize the video
    processing behaviour of the PPANDataset class. Simply override the
    process_video method.
    """
    @abstractmethod
    def process_video(self, img: torch.tensor) -> torch.tensor:
        ...

    def forward(self, x) -> torch.tensor:
        return self.process_video(x)

    @staticmethod
    def _get_midpoint(x: torch.tensor) -> int:
        """Get the middle frame index of a video tensor (...TCHW format).
        """
        return x.shape[-4] // 2

    def _center_cut(self, vid):
        """Get the (temporal) center of the video according to the supplied
        padding value. Essentially, we create a new view into the video clip
        with number of frames equal to 2*pad + 1 which is centered on the
        middle frame.
        """
        if self.pad is not None:
            midpoint = self._get_midpoint(vid)
            vid = vid[midpoint-self.pad:midpoint+self.pad+1]
        return vid


class PPANDataset(Dataset):
    """
    PPAN Dataset responsible for loading videos and processing them.
    The provided dataset should have already been pre-processed with
    PPANPreProcessor.
    """
    VID_FILENAME = "clip.mp4"
    LABEL_FILENAME = "labels.pt"

    def __init__(self, samples: list[Path], video_processor: BaseVideoProcessor,
                 output_map: Optional[OutputMap] = None) -> None:
        self.samples: list[Path] = samples
        self.videos: list[Path] = [
            i / self.VID_FILENAME for i in self.samples
        ]
        self.video_processor: BaseVideoProcessor = video_processor
        self._labs: dict[int, Optional[torch.Tensor]] = defaultdict(
            lambda: None
        )
        self._midpoint: Optional[int] = None
        self.output_map: Optional[OutputMap] = output_map

    def _get_lab(self, idx) -> torch.Tensor:
        if self._labs[idx] is None:
            self._labs[idx] = torch.load(
                self.samples[idx] / self.LABEL_FILENAME,
                weights_only=True)
        return torch.squeeze(self._labs[idx]).float()

    def _get_vid(self, idx) -> torch.Tensor:
        return self.video_processor(torchvision.io.read_video(
            str(self.videos[idx]),
            pts_unit="sec",
            output_format="TCHW"
        )[0])

    def _get_midpoint(self, vid: torch.tensor) -> int:
        if self._midpoint is None:
            self._midpoint = int(vid.shape[0] // 2) + 1
        return self._midpoint

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> (tuple[torch.Tensor, torch.Tensor] |
                                   dict[str, torch.Tensor]):
        if self.output_map is not None:
            out = {}
            if self.output_map["vid"] is not None:
                out[self.output_map["vid"]] = self._get_vid(idx)
            if self.output_map["lab"] is not None:
                out[self.output_map["lab"]] = self._get_lab(idx)
            return out
        return self._get_vid(idx), self._get_lab(idx)


def get_samples(root: Path):
    return [root / i for i in sorted(os.listdir(root))]

