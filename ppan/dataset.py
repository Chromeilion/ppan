from ppan.types import PathLike

from typing import List, Tuple, Optional, Callable
import random
import itertools

from rach3datautils.utils.dataset import DatasetUtils
from rach3datautils.utils.session import Session
from rach3datautils.utils.multimedia import MultimediaTools
import torchvision
import torch
from torch.utils.data import IterableDataset


class Rach3Dataset(IterableDataset):
    """
    Dataset object for training on the Rach3 dataset. Handles loading and
    preprocessing all necessary files.
    """
    def __init__(self, root: PathLike,
                 epoch_size: Optional[int] = None,
                 frame_transform: Optional[Callable] = None,
                 video_transform: Optional[Callable] = None,
                 clip_len: Optional[int] = None):
        if clip_len is None:
            clip_len = 16

        self.dataset = DatasetUtils(root)
        self.sessions = self.dataset.remove_noncomplete(
            subsession_list=self.dataset.get_sessions(),
            required=["midi.splits_list", "flac.splits_list",
                      "video.splits_list"]
        )
        # [midi_path, flac_path, video_path]
        self.samples: List[Tuple[PathLike, PathLike, PathLike]] = []

        for i in self.sessions:
            [self.samples.append(j) for j in zip(i.midi.splits_list,
                                                 i.flac.splits_list,
                                                 i.video.splits_list)]

        if epoch_size is None:
            epoch_size = len(self.samples)

        self.clip_len: int = clip_len
        self.epoch_size: int = epoch_size
        self.frame_transform: Optional[Callable] = frame_transform
        self.video_transform: Optional[Callable] = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            midi_path, flac_path, video_path = random.choice(self.samples)

            video = torchvision.io.VideoReader(str(video_path), "video")
            metadata = video.get_metadata()
            video_frames = []

            max_seek = metadata["video"]["duration"][0] - \
                (self.clip_len / metadata["video"]["fps"][0])

            start = random.uniform(0., max_seek)
            current_pts = max_seek
            for frame in itertools.islice(video.seek(start), self.clip_len):
                frame_data = frame['data']
                if self.frame_transform is not None:
                    frame_data = self.frame_transform(frame_data)
                current_pts = frame["pts"]
                video_frames.append(frame_data)

            stacked_frames = torch.stack(video_frames, 0)

            if self.video_transform is not None:
                stacked_frames = self.video_transform(stacked_frames)

            output = {
                'path': str(video_path),
                'video': stacked_frames,
                'target': 123,
                'start': start,
                'end': current_pts
            }
            yield output
