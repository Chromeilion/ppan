import io
import itertools
import random
from typing import List, Tuple, Optional, Callable

import mido
import numpy as np
import torch
import torchvision
from miditok import Structured
from miditoolkit import MidiFile
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import IterableDataset

from ppan.config import seq_len, VID_BACKEND
from ppan.types import PathLike


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
        if end is None:
            end = 1

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

        special_tokens = ["BOS", "EOS", "PAD"]
        self.tokenizer = Structured(pitch_range=(range(0, 127)),
                                    special_tokens=special_tokens,
                                    nb_velocities=1)
        self.vocab_len = len(self.tokenizer)

    def __iter__(self):
        for i in range(self.epoch_size):
            midi_path, flac_path, video_path = self.samples[i]

            video = torchvision.io.VideoReader(str(video_path), "video")

            metadata = video.get_metadata()
            video_frames = []
            sample_ratio = int(np.ceil(self.get_fps(metadata)) //
                               self.sample_rate)
            max_seek = self.get_duration(metadata) - \
                       (self.clip_len * sample_ratio /
                        self.get_fps(metadata))
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

                    tokens, padding_mask = self.tokenize_midi(
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

            current_time += mido.tick2second(i.time,
                                             midi.ticks_per_beat,
                                             tempo)

            if timestamps[0] <= current_time <= timestamps[1]:
                new_track.append(i)
            elif current_time > timestamps[1]:
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
                print(len(tokens))
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

    def tokens_to_midi(self, tokens: torch.Tensor):
        tokens = torch.unsqueeze(tokens, dim=1)
        midi_list = [self.tokenizer.tokens_to_midi(
            tokens=i.cpu().detach().numpy()) for i in
            torch.unbind(tokens, dim=0)]
        return midi_list


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = 1 / float(worker_info.num_workers)
    worker_id = worker_info.id

    dataset.start = worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, 1)
