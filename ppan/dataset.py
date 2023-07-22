import random
from typing import List, Tuple, Optional, Callable, TypedDict
import itertools

import numpy as np
import torch
import torchvision
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import IterableDataset

from ppan.config import VID_BACKEND
from ppan.preprocessing.midi import PPAnMidi
from ppan.preprocessing.video import PPAnVideoPreprocesser
from ppan.types import PathLike


class VideoDatasetOutput(TypedDict):
    path: str
    video: torch.Tensor
    tgt_in: torch.Tensor
    target: torch.Tensor
    padding_mask: torch.Tensor
    start: float
    end: float


class BaseDataset(IterableDataset):
    """
    Base class for PPAn datasets.
    """
    def __init__(self,
                 samples: List[Tuple[PathLike, PathLike, PathLike]],
                 clips_per_vid: Optional[int] = None,
                 epoch_size: Optional[int] = None,
                 frame_transform: Optional[Callable] = None,
                 video_transform: Optional[Callable] = None,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 shuffle_every_loop: Optional[bool] = None,
                 temporal_res: Optional[float] = None,
                 ):
        """
        Parameters
        ----------
        samples : List[Tuple[PathLike, PathLike, PathLike]]
            [midi_path, flac_path, video_path]
        epoch_size : Optional[int]
        frame_transform : Optional[Callable]
        video_transform : Optional[Callable]
        start : Optional[float]
        end : Optional[float]
        shuffle_every_loop : Optional[bool]
            whether to shuffle the dataset every time a new generator loop is
            started.
        temporal_res : Optional[float]
            The distance in seconds between yielded frames.
        """
        if temporal_res is None:
            temporal_res = 0.05
        if shuffle_every_loop is None:
            shuffle_every_loop = False
        if start is None:
            start = 0
        if end is None:
            end = 1
        if clips_per_vid is None:
            clips_per_vid = 100

        self.temporal_res = temporal_res

        self.shuffle = shuffle_every_loop
        self.samples = samples

        self.start = int(len(self.samples) * start)
        if np.isclose(end, 1):
            self.end=None
        else:
            self.end = int(len(self.samples) * end) + 1

        self.samples = self.samples[self.start:self.end]
        random.shuffle(self.samples)

        self.clips_per = clips_per_vid

        if epoch_size is None:
            epoch_size = len(self.samples)

        self.epoch_size: int = epoch_size
        self.frame_transform: Optional[Callable] = frame_transform
        self.video_transform: Optional[Callable] = video_transform

        self.midi = PPAnMidi()
        self.vocab_len = self.midi.vocab_len

        self.video_processor = PPAnVideoPreprocesser()

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

    def __len__(self):
        return len(self.samples) * self.clips_per


class ImageVecDataset(BaseDataset):
    """
    Dataset object for training on the Rach3 dataset. Handles loading and
    preprocessing all necessary files.
    Gives images plus vector representation of current notes being played.
    """
    # We need to decode a certain amount of frames before the actual timestamp
    # we want because we'll have corrupt data otherwise.
    # This is because of the way video and specifically keyframes work.
    # If the GPU decoding returned a pts, we could seek to just keyframes and
    # use those. This would save a lot of time. But no, they haven't
    # implemented that.
    FRAMES_PER_SAMPLE = 30

    def __init__(self, samples: List[Tuple[PathLike, PathLike, PathLike]],
                 tokenizer=None,
                 *args, **kwargs):
        super().__init__(samples, *args, **kwargs)
        self.tokenizer = tokenizer

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shuffle:
            random.shuffle(self.samples)

        for i in range(self.epoch_size):
            midi_path, flac_path, video_path = self.samples[i]

            video = torchvision.io.VideoReader(str(video_path), "audio")
            metadata = video.get_metadata()
            duration = metadata["video"]["duration"]
            frametime = 1. / self.get_fps(metadata)

            self.midi.set_midi(midi_path)

            # Pick a random point to start at. Remember that we need to decode
            # an amount of frames before the start point, so we need to allow
            # for that with some space at the start.
            # Additionally, we expect a certain amount of clips per video,
            # therefore, we need to leave enough space at the end of the video
            # to be able to generate these.
            decoding_space = self.FRAMES_PER_SAMPLE * frametime
            start = random.uniform(
                0. + decoding_space,
                duration - self.temporal_res*(self.clips_per+9)
            )
            times = np.linspace(start,
                                start+(self.temporal_res*(self.clips_per+9)),
                                self.clips_per+9)

            counter = 0
            for time in times:
                # Decode some amount of frames before the one we want
                decode_start = time - decoding_space
                for _ in itertools.islice(video.seek(decode_start), None,
                                          self.FRAMES_PER_SAMPLE-1):
                    ...

                # Get the frame we want
                vid_next = next(video)
                frame_data = vid_next['data']

                # CUDA has a different API than the other backends, so gotta
                # swap some axis
                if VID_BACKEND == "cuda":
                    frame_data = torch.swapaxes(frame_data, 0, 2)
                    frame_data = torch.swapaxes(frame_data, 1, 2)

                # Crop out the section of the image that is always not a piano
                frame_data = frame_data[:, 400:, :]

                if self.frame_transform is not None:
                    frame_data = self.frame_transform(
                        frame_data, return_tensor="pt")

                note_vec = self.midi.midi_to_vec(
                    timestamps=(time, time+frametime)
                )
                notes = self.midi.note_vec_to_sentence(note_vec)

                if notes is not False:
                    if self.tokenizer is not None:
                        notes = self.tokenizer(notes, padding='max_length',
                                               return_tensors="pt")
                    counter += 1
                    yield {"pixel_values": frame_data.pixel_values[0],
                           "labels": torch.squeeze(notes.input_ids)}

                if counter >= self.clips_per:
                    break


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
