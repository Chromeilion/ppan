import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
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
    FRAME_PREFIX: str = "frame_"
    FRAME_EXTENSION: str = ".jpeg"
    LABEL_FILENAME: str = "labels.pt"

    def __init__(self, samples: list[Path],
                 video_processor: BaseVideoProcessor,
                 output_map: Optional[OutputMap] = None,
                 pad: Optional[int] = None) -> None:
        samples = sorted([str(i) for i in samples])
        # Store samples in a numpy array for a better memory footprint
        self.samples: npt.NDArray[bytes] = np.array(samples).astype(np.string_)
        self.no_frames = len(list(Path(samples[0]).glob(
            f"{self.FRAME_PREFIX}*{self.FRAME_EXTENSION}"))
        )
        self.pad: Optional[int] = pad
        self.frame_files: npt.NDArray[bytes] = np.array(
            self._calc_frame_files()).astype(np.string_)

        self.vid_key: Optional[str] = "vid"
        self.lab_key: Optional[str] = "lab"
        if output_map is not None:
            self.vid_key = output_map["vid"]
            self.lab_key = output_map["lab"]

        self.video_processor: BaseVideoProcessor = video_processor

    @staticmethod
    def _load_lab(filepath: str) -> torch.Tensor:
        return torch.squeeze(torch.load(
                filepath,
                weights_only=True)).float()

    def _get_lab(self, idx):
        return self._load_lab(str(Path(
            self.samples[idx].decode())/self.LABEL_FILENAME))

    def _calc_frame_files(self) -> tuple[str, ...]:
        """Get the filenames of all the frames we want to load per sample.
        """
        files: list[str] = []
        midpoint = self._get_midpoint(no_frames=self.no_frames)
        start, end = 0, self.no_frames
        if self.pad is not None:
            start = midpoint - self.pad
            end = midpoint + self.pad
        for frame_no in range(start, end):
            files.append(self.FRAME_PREFIX+str(frame_no)+self.FRAME_EXTENSION)

        return tuple(files)

    def _get_vid(self, idx) -> torch.Tensor:
        """Load a video from the saved frames in a sample
        """
        sample = self.samples[idx]
        frames = []
        for frame in self.frame_files:
            frame_data = torchvision.io.read_file(str(Path(sample.decode())/frame.decode()))
            frames.append(torchvision.io.decode_jpeg(frame_data,
                          device="cpu"))
        video = torch.stack(frames, dim=0)
        return self.video_processor(video)

    @staticmethod
    def _get_midpoint(no_frames) -> int:
        """Calculate the middle frame index in a list of frames.
        """
        return int(no_frames // 2) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> (tuple[torch.Tensor, torch.Tensor] |
                                   dict[str, torch.Tensor]):
        out = {}
        if self.vid_key is not None:
            out[self.vid_key] = self._get_vid(idx)
        if self.lab_key is not None:
            out[self.lab_key] = self._get_lab(idx)
        return out


def get_samples(root: Path):
    return [root / i for i in sorted(os.listdir(root))]

