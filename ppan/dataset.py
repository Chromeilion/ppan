import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypedDict, Any
import random

from ppan.config import frames_per_ds_sample, frame_stride, model_resolution
from ppan.utils import RunningCellMaskingGenerator
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


class OutputMap(TypedDict):
    """Template for a dictionary that controls the names and presence of the
    outputs in a PPANDataset. If one of these is set to None, then it never
    gets loaded.
    """
    vid: Optional[str]
    lab: Optional[str]
    mask: Optional[str]


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
                 shared_dict=None,
                 temporal_jitter: Optional[bool] = None,
                 mask_percentage: Optional[float] = None,
                 tubelet_size: Optional[tuple[int, int, int]] = None) -> None:
        if temporal_jitter is None:
            temporal_jitter = False
        if tubelet_size is None and mask_percentage is not None:
            raise AttributeError("Please provide the tubelet size when using "
                                 "masking")
        self.temporal_jitter = temporal_jitter

        samples = sorted([str(i) for i in samples])
        # Store samples in a numpy array to avoid refcounts blowing up the
        # memory usage.
        self.samples: npt.NDArray[bytes] = np.array(samples).astype(np.bytes_)
        self.no_frames = frames_per_ds_sample
        self.stride = frame_stride

        self.frame_idxs: npt.NDArray[int] = self._calc_frame_file_idxs()

        if mask_percentage is None:
            self.mask_key = None
            if output_map is not None:
                output_map["mask"] = None
        else:
            frame_patches = self.no_frames // tubelet_size[0]
            h_patches = model_resolution[0] // tubelet_size[1]
            w_patches = model_resolution[1] // tubelet_size[2]
            self.mask_generator = RunningCellMaskingGenerator(
                input_size=(frame_patches, h_patches, w_patches),
                mask_ratio=mask_percentage
            )
            self.mask_key: Optional[str] = "lab"

        self.vid_key: Optional[str] = "vid"
        self.lab_key: Optional[str] = "lab"

        if output_map is not None:
            self.vid_key = output_map["vid"]
            self.lab_key = output_map["lab"]
            self.mask_key = output_map["mask"]

        self.video_processor: BaseVideoProcessor = video_processor
        self.shared_dict: Optional[dict[str, Any]] = shared_dict

    @staticmethod
    def _load_lab(filepath: str) -> torch.Tensor:
        return torch.squeeze(torch.load(
            filepath,
            weights_only=True)).float()

    def _get_lab(self, idx):
        return self._load_lab(str(Path(
            self.samples[idx].decode()) / self.LABEL_FILENAME))

    @staticmethod
    def _calc_frame_file_idxs() -> npt.NDArray[int]:
        """Get the filenames of all the frames we want to load per sample.
        """
        return np.array([0, 1, 2, 3, 4, 5])

    def get_frame_files(self, idxs: npt.NDArray[int]) -> list[str]:
        files = []
        for frame_no in idxs:
            files.append(
                self.FRAME_PREFIX + str(frame_no) + self.FRAME_EXTENSION)
        return files

    def _read_file(self, filepath):
        if self.shared_dict is not None:
            if str(filepath) not in self.shared_dict:
                data = torchvision.io.read_file(filepath).numpy()
                self.shared_dict[str(filepath)] = data
            return torch.tensor(self.shared_dict[str(filepath)])
        return torchvision.io.read_file(filepath)

    def _get_vid(self, idx) -> torch.Tensor:
        """Load a video from the saved frames in a sample
        """
        sample = self.samples[idx]
        path = Path(sample.decode())

        if self.temporal_jitter:
            jitter = random.choice([0, 1])
        else:
            jitter = 0

        frames = [torchvision.io.decode_jpeg(
            self._read_file(str(path / frame))) for frame in
            self.get_frame_files(self.frame_idxs + jitter)]

        video = torch.stack(frames, dim=0)
        return self.video_processor(video)

    @staticmethod
    def _get_midpoint(no_frames) -> int:
        """Calculate the middle frame index in a list of frames.
        """
        return int(no_frames // 2) + 1

    def __len__(self):
        return len(self.samples)

    def _get_mask(self):
        mask = torch.tensor(self.mask_generator(), dtype=torch.bool)
        return mask.flatten()

    def __getitem__(self, idx) -> (tuple[torch.Tensor, torch.Tensor] |
                                   dict[str, torch.Tensor]):
        out = {}
        if self.mask_key is not None:
            out[self.mask_key] = self._get_mask()
        if self.vid_key is not None:
            out[self.vid_key] = self._get_vid(idx)
        if self.lab_key is not None:
            out[self.lab_key] = self._get_lab(idx)
        return out


def get_samples(root: Path):
    cmd = f'ls {str(root)}'
    cmd_out = os.popen(cmd).read().split("\n")
    samples = sorted([root / i for i in cmd_out if i])
    return samples


def split_samples(samples, val_perc: float = 0.05):
    """Split samples into train and validaiton sets.
    """
    n_val = round(val_perc * len(samples))
    val = random.sample(samples, n_val)
    train = set(samples) - set(val)
    return list(train), val
