import itertools
import random
from typing import List, Tuple, Optional, Callable

import mido
import numpy as np
import torch
import torchvision
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import IterableDataset

from ppan.config import VID_BACKEND
from ppan.midi import PPAnMidi
from ppan.types import PathLike


class Rach3Dataset(IterableDataset):
    """
    Dataset object for training on the Rach3 dataset. Handles loading and
    preprocessing all necessary files.
    """

    def __init__(self, samples: List[Tuple[PathLike, PathLike, PathLike]],
                 seq_len: int,
                 epoch_size: Optional[int] = None,
                 frame_transform: Optional[Callable] = None,
                 video_transform: Optional[Callable] = None,
                 clip_len: Optional[int] = None,
                 sample_rate: Optional[int] = None,
                 start: Optional[float] = None,
                 end: Optional[float] = None):
        """
        Parameters
        ----------
        samples : List[Tuple[PathLike, PathLike, PathLike]]
            [midi_path, flac_path, video_path]
        seq_len : int
            maximum length of token sequence
        epoch_size : Optional[int]
            defaults to len(samples)
        frame_transform : Optional[Callable]
            Transform to be applied per frame
        video_transform : Optional[Callable]
            Transform to be applied per video
        clip_len : Optional[int]
            The amount of frames per video. Defaults to 16
        sample_rate : Optional[int]
            FPS basically, defaults to 60
        start : Optional[float]
            float from zero to one, where to start in the dataset. For example,
            0.5 is in the middle
        end : Optional[float]
            Same as start but for the end
        """
        if clip_len is None:
            clip_len = 16
        if sample_rate is None:
            sample_rate = 60
        if start is None:
            start = 0
        if end is None:
            end = 1

        self.samples = samples

        self.start = int(len(self.samples) * start)
        self.end = int(len(self.samples) * end)

        self.samples = self.samples[self.start:self.end]
        random.shuffle(self.samples)

        if epoch_size is None:
            epoch_size = len(self.samples)

        self.clip_len: int = clip_len
        self.epoch_size: int = epoch_size
        self.frame_transform: Optional[Callable] = frame_transform
        self.video_transform: Optional[Callable] = video_transform
        self.sample_rate = sample_rate

        self.tokenizer = PPAnMidi(seq_len)
        self.vocab_len = self.tokenizer.vocab_len

    def __iter__(self):
        for i in range(self.epoch_size):
            midi_path, flac_path, video_path = self.samples[i]

            video = torchvision.io.VideoReader(str(video_path), "video")

            metadata = video.get_metadata()
            video_frames = []
            sample_ratio = int(np.ceil(self.get_fps(metadata)) //
                               self.sample_rate)
            max_seek = self.get_duration(metadata) - \
                (self.clip_len * sample_ratio / self.get_fps(metadata))
            max_seek_frame = int(max_seek * self.get_fps(metadata))
            start = random.uniform(0., max_seek / 4)
            prev = start
            start_frame = int(start * self.get_fps(metadata))
            midi_file = mido.MidiFile(midi_path)

            for frame_no, frame in enumerate(itertools.islice(
                    video.seek(start), max_seek_frame - start_frame)):
                if frame_no % sample_ratio != 0:
                    continue

                current_pts = start + frame_no*(1/self.get_fps(metadata))

                frame_data = frame['data']

                if VID_BACKEND == "cuda":
                    frame_data = torch.swapaxes(frame_data, 0, 2)

                if self.frame_transform is not None:
                    frame_data = self.frame_transform(frame_data)

                video_frames.append(torch.swapaxes(frame_data, 1, 2))
                if len(video_frames) == self.clip_len:

                    stacked_frames = torch.stack(video_frames, 0)
                    if self.video_transform is not None:
                        stacked_frames = self.video_transform(stacked_frames)

                    tokens, padding_mask = self.tokenizer(
                        midi_file,
                        (prev, current_pts)
                    )
                    output = {
                        'path': str(video_path),
                        'video': stacked_frames,
                        'target': tokens,
                        'padding_mask': padding_mask,
                        'start': prev,
                        'end': current_pts
                    }
                    yield output
                    video_frames = []
                    prev = current_pts

    @staticmethod
    def get_fps(metadata):
        if VID_BACKEND in ["cuda"]:
            return metadata["video"]["fps"]
        return metadata["video"]["fps"][0]

    @staticmethod
    def get_duration(metadata):
        if VID_BACKEND in ["cuda"]:
            return metadata["video"]["duration"]
        return metadata["video"]["duration"][0]


def split_data(samples: List[Tuple[PathLike, PathLike, PathLike]],
               percentage: float):
    """
    Split the dataset into train and test sets according to percentage size of
    test.
    This method is approximate. The split is not exactly according to the given
    percentage.

    Parameters
    ----------
    percentage : float
        between zero and one, what percentage of the entire dataset the test
        set is.
    samples : List[Tuple[PathLike, PathLike, PathLike]]

    Returns
    -------
    train : List[Tuple[PathLike, PathLike, PathLike]]
    test : List[Tuple[PathLike, PathLike, PathLike]]
    """
    samples = np.array(samples)
    mask = np.random.rand(len(samples)) <= percentage
    train = samples[~mask]
    test = samples[mask]
    return train, test


def load_data(root: PathLike):
    """
    Load the dataset for use with PPAn.

    Parameters
    ----------
    root : PathLike

    Returns
    -------
    samples : List[Tuple[PathLike, PathLike, PathLike]]
        List containing: (midi_path, flac_path, video_path)
    """
    dataset = DatasetUtils(root)
    sessions = dataset.remove_noncomplete(
        subsession_list=dataset.get_sessions(),
        required=["midi.splits_list", "flac.splits_list",
                  "video.splits_list"]
    )
    # [midi_path, flac_path, video_path]
    samples: List[Tuple[PathLike, PathLike, PathLike]] = []

    for i in sessions:
        [samples.append(j) for j in zip(i.midi.splits_list,
                                        i.flac.splits_list,
                                        i.video.splits_list)]

    return samples


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = 1 / float(worker_info.num_workers)
    worker_id = worker_info.id

    dataset.start = worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, 1)
