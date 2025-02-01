from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypedDict, Any
import random
from dataclasses import dataclass

from ppan.midi import PPAnMidi
from ppan.config import frame_hd5_file

import  h5py
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset

device = "cpu"

class OutputMap(TypedDict):
    """Template for a dictionary that controls the names and presence of the
    outputs in a PPANDataset. If one of these is set to None, then it never
    gets loaded.
    """
    vid: Optional[str]
    frames: Optional[str]
    onsets: Optional[str]


class BaseVideoProcessor(nn.Module, ABC):
    """
    Base video processor abstract class. Can be used to customize the video
    processing behaviour of the PPANDataset class. Simply override the
    process_video method.
    """
    @abstractmethod
    @torch.compile(mode="reduce-overhead")
    def process_video(self, img: torch.tensor) -> torch.tensor:
        ...

    def forward(self, x) -> torch.tensor:
        return self.process_video(x)


class DefaultVideoProcessor(BaseVideoProcessor):
    """Default video processor doing basically nothing except a resize.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = v2.Resize((224, 224))

    def process_video(self, vid):
        return self.augmentations(vid)


@dataclass
class DatasetConfig:
    video_processor: BaseVideoProcessor
    output_map: OutputMap
    stride: int # Step size between consecutive frames in a window
    window_size: int
    lenience: int
    shared_dict: Optional[dict[str, Any]] = None
    max_samples: Optional[int] = None


class PPANDataset(Dataset):
    """
    PPAN Dataset responsible for loading videos and processing them.
    The provided dataset should have already been pre-processed with
    PPANPreProcessor.
    """
    def __init__(self, root: Path, config: DatasetConfig) -> None:
        self.root = root
        self._hd5_frame = None
        self.config = config
        self.samples = self._load_samples()
        if self.config.max_samples is not None:
            self.samples = random.choices(self.samples, k=self.config.max_samples)
        self.vid_key: Optional[str] = "vid"
        self.onsets_key: Optional[str] = "onsets"
        self.frames_key: Optional[str] = "frames"

        if config.output_map is not None:
            self.vid_key = config.output_map["vid"]
            self.onsets_key = config.output_map["onsets"]
            self.frames_key = config.output_map["frames"]

        self.video_processor: BaseVideoProcessor = config.video_processor
        self.cache: Optional[dict[str, Any]] = config.shared_dict

    @property
    def hd5_frame(self):
        if self._hd5_frame is None:
            self._hd5_frame = h5py.File(str(self.root / frame_hd5_file), "r")
        return self._hd5_frame

    def _load_samples(self) -> list[tuple[int, ProcessedSampleWrapper]]:
        videos = list(self.hd5_frame.keys())
        samples = []
        for i in videos:
            frames = self.hd5_frame[i]["frames"]
            video_wrap = ProcessedSampleWrapper(frames, i, self.root/f"{i}.midi", self.root/f"{i}_video_details.txt", self.config)
            for j in video_wrap.get_samples():
                samples += [(j, video_wrap)]

        return samples

    def _get_lab(self, idx):
        sample = self.samples[idx]
        onset = sample[1].midi_wrapper.oo_array[..., sample[0]]
        frame = sample[1].midi_wrapper.f_array[..., sample[0]]
        return {"onsets": onset, "frames": frame}

    def _get_vid(self, idx) -> torch.Tensor:
        """Load a video from the saved frames in a sample
        """
        sample = self.samples[idx]
        video = sample[1].video_wrapper.get_frame_window(sample[0]).to(device, non_blocking=True)
        return self.video_processor(video)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> (tuple[torch.Tensor, torch.Tensor] |
                                   dict[str, torch.Tensor]):
        out = {}
        if self.vid_key is not None:
            out[self.vid_key] = self._get_vid(idx)
        if self.onsets_key is None and self.frames_key is None:
            return out
        labs = self._get_lab(idx)
        if self.onsets_key is not None:
            out[self.onsets_key] = torch.tensor(labs["onsets"], device=device).to(device, non_blocking=True).float()
        if self.frames_key is not None:
            out[self.frames_key] = torch.tensor(labs["frames"], device=device).to(device, non_blocking=True).float()
        return out


class ProcessedSampleWrapper:
    """Helper for handling each individual sample. Keeps track of the frame and
    MIDI objects and provides a sampler.
    """
    MIDI_FILE = "data.midi"
    def __init__(self, frames, cache_prefix: str, midi_path: Path, details_file: Path, config: DatasetConfig) -> None:
        self.config = config
        self.video_wrapper = FramedVideoWrapper(
            details_file,
            frames,
            window_size=config.window_size,
            stride=config.stride,
            cache_prefix=cache_prefix,
            cache_dict=config.shared_dict
        )
        self.midi_wrapper: PPAnMidi = PPAnMidi(
            n_frames=self.video_wrapper.n_frames,
            temporal_res=self.video_wrapper.frametime,
            lenience=config.lenience
        ).set_midi(str(midi_path))

    def get_samples(self):
        n_frames = self.video_wrapper.n_frames
        half_window = self.config.window_size // 2
        samples = list(range(half_window, n_frames-half_window))
        return samples


class FramedVideoWrapper:
    """Convenience class that can handle automatically loading frames from a
    processed video folder.
    """
    FRAME_PREFIX: str = "frame_"
    FRAME_EXTENSION: str = ".jpeg"
    DETAILS_FILE: str = "video_details.txt"

    def __init__(self, details_file, frame_data, window_size: int, stride: int, cache_prefix: str, cache_dict: Optional[dict] = None):
        self.details_file = details_file
        self.frames = frame_data
        self.n_frames: int = frame_data.shape[0]
        self._duration = None
        self.frametime: float =  self.duration / self.n_frames

        self.frame_idxs = np.array([i * stride for i in range(-window_size//2, window_size//2)])
        self.cache: Optional[dict] = cache_dict
        self.cache_prefix = cache_prefix
        self.cache_index = torch.tensor([False for _ in range(self.frames.shape[0])], dtype=torch.bool)

    @property
    def duration(self) -> float:
        if self._duration is None:
            with open(self.details_file) as f:
                details = [i.split(": ") for i in f.readlines()]
            details = {
                key: val for key, val in details
            }
            self._duration = float(details["duration"])
        return self._duration

    def get_frames(self, frames: npt.NDArray[int]):
        if self.cache is not None:
            not_in_cache = ~self.cache_index[frames]
            if torch.any(not_in_cache):
                for i in torch.where(not_in_cache)[0]:
                    frame_data = self.frames[frames[i]]
                    self.cache[self.cache_prefix+str(int(frames[i]))] = frame_data
                    self.cache_index[frames[i]] = True

            return [torch.tensor(self.cache[self.cache_prefix+str(int(i))]) for i in frames]

        all_frames = []
        for i in range(len(frames)):
            frame_data = self.frames[frames[i]]
            all_frames.append(frame_data)

        return [torch.tensor(i) for i in all_frames]

    def get_video(self, frames: npt.NDArray[int]):
        frame_data = self.get_frames(frames)
        frames_decoded = torchvision.io.decode_jpeg(frame_data)
        video = torch.stack(frames_decoded, dim=0)
        return video

    def get_time_window(self, time: float):
        """Get a window into the video centered at a certain time.
        """
        frame = round(time / self.frametime)
        return self.get_frame_window(frame)

    def get_frame_window(self, frame: int):
        """Get a window into the video centered at a certain frame.
        """
        idxs = self.frame_idxs + frame
        return self.get_video(idxs)


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
