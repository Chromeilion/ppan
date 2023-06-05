from typing import List, Tuple, Optional, Callable
import random
import itertools
import io

import miditok

from ppan.types import PathLike
from ppan.config import seq_len

import numpy as np
from rach3datautils.utils.dataset import DatasetUtils
from rach3datautils.utils.session import Session
from rach3datautils.utils.multimedia import MultimediaTools
import torchvision
import torch
from torch.utils.data import IterableDataset
from miditok import Structured
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
import mido

torchvision.set_video_backend("pyav")


class Rach3Dataset(IterableDataset):
    """
    Dataset object for training on the Rach3 dataset. Handles loading and
    preprocessing all necessary files.
    """

    def __init__(self, root: PathLike,
                 epoch_size: Optional[int] = None,
                 frame_transform: Optional[Callable] = None,
                 video_transform: Optional[Callable] = None,
                 clip_len: Optional[int] = None,
                 sample_rate: Optional[int] = None,
                 start: Optional[float] = None,
                 end: Optional[float] = None):
        if clip_len is None:
            clip_len = 16
        if sample_rate is None:
            sample_rate = 60
        if start is None:
            start = 0
        if end is None: end = 1

        self.dataset = DatasetUtils(root)
        self.sessions = self.dataset.remove_noncomplete(
            subsession_list=self.dataset.get_sessions(),
            required=["midi.splits_list", "flac.splits_list",
                      "video.splits_list"]
        )
        # [midi_path, flac_path, video_path]
        self.samples: List[Tuple[PathLike, PathLike, PathLike]] = []

        self.start = int(len(self.samples) * start)
        self.end = int(len(self.samples) * end)

        self.samples = self.samples[start:end]

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
        self.sample_rate = sample_rate

        special_tokens = ["BOS", "EOS", "PAD"]
        self.tokenizer = Structured(pitch_range=(range(0, 127)),
                                    special_tokens=special_tokens)
        self.vocab_len = len(self.tokenizer)

    def __iter__(self):
        for i in range(self.epoch_size):
            midi_path, flac_path, video_path = random.choice(self.samples)

            video = torchvision.io.VideoReader(str(video_path), "video",
                                               num_threads=4)
            metadata = video.get_metadata()
            video_frames = []
            sample_ratio = int(np.ceil(metadata["video"]["fps"][0]) //
                               self.sample_rate)
            max_seek = metadata["video"]["duration"][0] - \
                       (self.clip_len * sample_ratio /
                        metadata["video"]["fps"][0])
            max_seek_frame = int(max_seek * metadata["video"]["fps"][0])
            start = random.uniform(0., max_seek / 4)
            start_frame = int(start * metadata["video"]["fps"][0])

            for frame_no, frame in enumerate(itertools.islice(
                    video.seek(start), max_seek_frame - start_frame)):
                if frame_no % sample_ratio != 0:
                    continue

                frame_data = frame['data']
                if self.frame_transform is not None:
                    frame_data = self.frame_transform(frame_data)

                current_pts = frame["pts"]
                video_frames.append(frame_data)
                if len(video_frames) == self.clip_len:

                    stacked_frames = torch.stack(video_frames, 0)

                    if self.video_transform is not None:
                        stacked_frames = self.video_transform(stacked_frames)

                    midi_file = mido.MidiFile(midi_path)
                    tokens, padding_mask = self.tokenize_midi(
                        midi_file,
                        (start, current_pts)
                    )
                    output = {
                        'path': str(video_path),
                        'video': stacked_frames,
                        'target': tokens,
                        'padding_mask': padding_mask,
                        'start': start,
                        'end': current_pts
                    }
                    yield output
                    video_frames = []
                    start = current_pts

    def tokenize_midi(self,
                      midi,
                      timestamps: Tuple[float, float]) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a section from a midi based on timestamps and tokenize it.
        This implementation is a little absurd, but It's what I came up with
        and it works.

        Parameters
        ----------
        midi : mido.MidiFile
        timestamps : Tuple[float, float]

        Returns
        -------
        tokens : torch.Tensor
        """
        new_midi = mido.MidiFile(type=0)
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        current_time = 0
        for i in midi.tracks[0]:
            if i.type == 'set_tempo':
                tempo = i.tempo

            current_time += mido.tick2second(i.time, midi.ticks_per_beat, tempo)

            if current_time > timestamps[0]:
                new_track.append(i)
            if current_time > timestamps[1]:
                break

        with io.BytesIO() as f:
            new_midi.save(file=f)
            # Make sure to go back to the start of the file after writing!
            f.seek(0)
            tokens = self.tokenizer.midi_to_tokens(MidiFile(file=f),
                                                   add_special_tokens=True)

            if len(tokens) != 0:
                tokens = list(tokens[0])

            tokens.insert(0, self.tokenizer["BOS_None"])

            if len(tokens) > seq_len-1:
                tokens = tokens[:seq_len-2]
            tokens.append(self.tokenizer["EOS_None"])

            while len(tokens) < seq_len:
                tokens.append(self.tokenizer["PAD_None"])

        tokens = torch.tensor(tokens, dtype=torch.long)

        padding_mask = torch.squeeze(
            torch.where(tokens == self.tokenizer["PAD_None"],
                        True,
                        False))

        return tokens, padding_mask


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = 1 / float(worker_info.num_workers)
    worker_id = worker_info.id

    dataset.start = worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, 1)
