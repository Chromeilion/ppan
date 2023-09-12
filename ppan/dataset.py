import time
import os
import itertools
import random
from functools import partial
from torch.multiprocessing import Pool
from typing import List, Tuple, Optional, Callable, TypedDict
import csv

import numpy as np
import torch
import torchvision
import torchvision.utils
import torchvision.transforms.functional as f
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

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
                 shuffle_every_loop: Optional[bool] = None,
                 temporal_res: Optional[float] = None,
                 rotate: bool = False
                 ):
        """
        Parameters
        ----------
        samples : List[Tuple[PathLike, PathLike, PathLike]]
            [midi_path, flac_path, video_path]
        epoch_size : Optional[int]
        frame_transform : Optional[Callable]
        video_transform : Optional[Callable]
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

        self.temporal_res = temporal_res
        self.rotate = rotate
        self.shuffle = shuffle_every_loop
        self.samples = samples
        random.shuffle(self.samples)
        self.clips_per = clips_per_vid

        if epoch_size is None:
            epoch_size = 100000000
        self.epoch_size: int = epoch_size
        self.frame_transform: Optional[Callable] = frame_transform
        self.video_transform: Optional[Callable] = video_transform

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

    @staticmethod
    def plot_image(img_tensor: torch.Tensor):
        import matplotlib.pyplot as plt
        img_tensor_np = torch.swapaxes(img_tensor.detach().cpu(), 0, 2)
        plt.figure()
        plt.imshow(img_tensor_np)
        plt.show()


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
    FRAMES_PER_SAMPLE = 5

    def __init__(self, samples: List[Tuple[PathLike, PathLike, PathLike]],
                 tokenizer=None,
                 *args, **kwargs):
        super().__init__(samples, *args, **kwargs)
        self.tokenizer = tokenizer
        self.augment = v2.Compose([
            v2.RandomCrop(size=(1950, 1900))])

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shuffle:
            random.shuffle(self.samples)
        with Pool(4) as p:
            for _ in range(self.epoch_size):
                seeds = [random.randint(0, 10000) for _ in self.samples]
                for data in p.imap(partial(_load,
                                           FRAMES_PER_SAMPLE=self.FRAMES_PER_SAMPLE,
                                           clips_per=self.clips_per,
                                           temporal_res=self.temporal_res,
                                           frame_transform=self.frame_transform,
                                           augment=self.augment,
                                           rotate=self.rotate
                                           ),
                                   zip(self.samples, seeds)):
                    for i in data:
                        if i[1]:
                            if self.tokenizer is not None:
                                notes = self.tokenizer(i[1], padding='max_length',
                                                       return_tensors="pt")
                            if i[0].size() != (3, 448, 448):
                                continue
                            elif notes.input_ids.size()[1] > 20:
                                continue
                            yield {"pixel_values": i[0],
                                   "labels": torch.squeeze(notes.input_ids),
                                   "interpolate_pos_encoding": True}


class EvalDataset(ImageVecDataset):
    """
    Simplified dataloader focussed solely around evaluation.
    """
    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_dist_time = 0.1
        for i in self.samples:
            try:
                for j in _eval_load(
                    sample=i,
                    frame_dist_time=frame_dist_time,
                    temporal_res=self.temporal_res,
                    rotate=self.rotate,
                    tokenizer=self.tokenizer
                ):
                    yield j
            except RuntimeError:
                continue


class PlayingDataset(BaseDataset):
    FRAMES_PER_SAMPLE = 35

    def __init__(self, samples: List[Tuple[PathLike, PathLike, PathLike]],
                 tokenizer=None,
                 *args, **kwargs):
        super().__init__(samples, *args, **kwargs)
        self.tokenizer = tokenizer
        torchvision.disable_beta_transforms_warning()
        self.augment = v2.Compose([
            v2.RandomCrop(size=(1950, 1900)),
            v2.RandomHorizontalFlip(p=0.5)])

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shuffle:
            random.shuffle(self.samples)
        with Pool(4) as p:
            for _ in range(self.epoch_size):
                seeds = [random.randint(0, 10000) for i in self.samples]
                for data in p.imap(partial(_load,
                                           FRAMES_PER_SAMPLE=self.FRAMES_PER_SAMPLE,
                                           clips_per=self.clips_per,
                                           temporal_res=self.temporal_res,
                                           frame_transform=self.frame_transform,
                                           augment=self.augment,
                                           rotate=self.rotate
                                           ),
                                   zip(self.samples, seeds)):
                    for i in data:
                        if i[1]:
                            label = 1
                        else:
                            label = 0
                        yield {"pixel_values": i[0], "label": label}


def _load(sample_seed, FRAMES_PER_SAMPLE, clips_per,
          temporal_res, frame_transform, augment, rotate):
    crop = None
    if len(sample_seed[0]) == 4:
        midi_path, flac_path, video_path, crop = sample_seed[0]
    else:
        midi_path, flac_path, video_path = sample_seed[0]

    random.seed(sample_seed[1])
    np.random.seed(sample_seed[1])
    torch.manual_seed(sample_seed[1])

    video = torchvision.io.VideoReader(str(video_path))
    metadata = video.get_metadata()
    duration = BaseDataset.get_duration(metadata)
    frametime = 1. / BaseDataset.get_fps(metadata)

    frame_dist_time = 0.1
    frame_dist = int(frame_dist_time/frametime)

    midi = PPAnMidi()
    midi.set_midi(midi_path)

    # Pick a random point to start at. Remember that we need to decode
    # an amount of frames before the start point, so we need to allow
    # for that with some space at the start.
    # Additionally, we expect a certain amount of clips per video,
    # therefore, we need to leave enough space at the end of the video
    # to be able to generate these.
    start = random.uniform(
        0.,
        duration - temporal_res * clips_per - frame_dist_time*3
    )
    times = np.linspace(start,
                        start + (temporal_res * clips_per),
                        clips_per)
    return_list = []
    for time in times:
        video.seek(time)
        frame_data_prev = next(video)['data']
        for _ in itertools.islice(video, None, frame_dist - 1):
            ...
        frame_mid = next(video)
        time = frame_mid['pts']
        frame_data_mid = frame_mid['data']
        for _ in itertools.islice(video, None, frame_dist - 1):
            ...
        frame_data_next = next(video)['data']

        # CUDA has a different API than the other backends, so gotta
        # swap some axis
        if VID_BACKEND == "cuda":
            frame_data_prev = fix_cuda_axes(frame_data_prev)
            frame_data_mid = fix_cuda_axes(frame_data_mid)
            frame_data_next = fix_cuda_axes(frame_data_next)

        if crop is not None:
            frame_data_prev = do_crop(frame_data_prev, crop)
            frame_data_mid = do_crop(frame_data_mid, crop)
            frame_data_next = do_crop(frame_data_next, crop)

        if rotate:
            frame_data_prev = f.rotate(frame_data_prev, 180)
            frame_data_mid = f.rotate(frame_data_mid, 180)
            frame_data_next = f.rotate(frame_data_next, 180)

        frame_data_prev = resize_correct(frame_data_prev)
        frame_data_mid = resize_correct(frame_data_mid)
        frame_data_next = resize_correct(frame_data_next)

        img = combine_imgs(frame_data_prev, frame_data_mid, frame_data_next)

        img = augment(img)

        if frame_transform is not None:
            img = frame_transform(
                img, return_tensors="pt")

        img = torch.squeeze(img.pixel_values)
        notes_str = midi.midi_to_notes(
            timestamps=(time-frametime/2, time+frametime/2)
        )
        print(notes_str)
        return_list.append((img, notes_str))

    return return_list


def _eval_load(sample, frame_dist_time, temporal_res, rotate, tokenizer):
    crop = None
    if len(sample) == 4:
        midi_path, flac_path, video_path, crop = sample
    else:
        midi_path, flac_path, video_path = sample

    video = torchvision.io.VideoReader(str(video_path), stream='video')

    metadata = video.get_metadata()
    frametime = 1. / EvalDataset.get_fps(metadata)

    frame_dist = round(frame_dist_time / frametime)
    frameskip = max(1, round(temporal_res / frametime))

    midi = PPAnMidi()
    midi.set_midi(midi_path)
    rotlist = []
    for frame_no, frame in enumerate(video.seek(0)):
        rotlist.append(frame)
        if len(rotlist) > frame_dist * 2:
            rotlist.pop(0)
        if frame_no % frameskip != 0 or len(rotlist) < frame_dist * 2:
            continue

        current_time = rotlist[frame_dist+1]['pts'] - frame_dist * frametime

        frame_data_prev = rotlist[0]['data']
        frame_data_mid = rotlist[frame_dist+1]['data']
        frame_data_next = frame['data']
        # CUDA has a different API than the other backends, so gotta
        # swap some axis
        if VID_BACKEND == "cuda":
            frame_data_prev = fix_cuda_axes(frame_data_prev)
            frame_data_mid = fix_cuda_axes(frame_data_mid)
            frame_data_next = fix_cuda_axes(frame_data_next)

        if crop is not None:
            frame_data_prev = do_crop(frame_data_prev, crop)
            frame_data_mid = do_crop(frame_data_mid, crop)
            frame_data_next = do_crop(frame_data_next, crop)

        if rotate:
            frame_data_prev = f.rotate(frame_data_prev, 180)
            frame_data_mid = f.rotate(frame_data_mid, 180)
            frame_data_next = f.rotate(frame_data_next, 180)

        frame_data_prev = resize_correct(frame_data_prev)
        frame_data_mid = resize_correct(frame_data_mid)
        frame_data_next = resize_correct(frame_data_next)

        img = combine_imgs(frame_data_prev, frame_data_mid, frame_data_next)

        try:
            notes_str = midi.midi_to_notes(
                timestamps=(current_time-frametime/2,
                            current_time + frametime/2)
            )
        except AttributeError:
            raise RuntimeError

        if notes_str:
            if tokenizer is not None:
                notes = tokenizer(notes_str, padding='max_length',
                                  return_tensors="pt")
                notes = torch.squeeze(notes.input_ids)
            else:
                notes = notes_str
        else:
            notes = []

        yield {"pixel_values": img,
               "labels": notes,
               "midi_path": str(midi_path),
               "video_path": str(video_path),
               'time': current_time}


def combine_imgs(im1, im2, im3):
    img = torch.zeros(size=(im2.size()[0],
                            680 * 3,
                            im2.size()[2]),
                      dtype=torch.uint8).to(im2.device)

    img[:, :680, :] = im1
    img[:, 680:1360, :] = im2
    img[:, 1360:2040, :] = im3
    return img


def resize_correct(img):
    return f.resize(img,
                    size=[680, 1920],
                    antialias=False)


def fix_cuda_axes(img):
    img = torch.swapaxes(img, 0, 2)
    img = torch.swapaxes(img, 1, 2)
    return img


def do_crop(img, crop_vals):
    return f.crop(img,
                  top=crop_vals[0],
                  left=crop_vals[2],
                  height=crop_vals[1] - crop_vals[0],
                  width=crop_vals[3] - crop_vals[2])


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
        crops = [(400, 1080, 0, 1920) for _ in i.midi.splits_list]
        [samples.append(j) for j in zip(i.midi.splits_list,
                                        i.flac.splits_list,
                                        i.video.splits_list,
                                        crops)]

    return samples


def load_ytmidi(root: PathLike):
    data = []
    with open(os.path.join(root, "dataset.csv"), "r") as f:
        reader = csv.reader(f)
        [data.append(i) for i in reader]

    samples_train = []
    samples_test = []
    for i in data:
        crop = [int(j) for j in i[4:]]
        tup = (os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
               None, i[0], crop)
        if i[3] == "1":
            samples_train.append(tup)
        elif i[3] == "3":
            samples_test.append(tup)

    return samples_train, samples_test


def load_miditest(root):
    midi_root = os.path.join(root, "miditest_MIDI")
    midi_files = os.listdir(midi_root)
    midi_files = [os.path.join(midi_root, i) for i in midi_files]
    videos_root = os.path.join(root, "miditest_videos")
    videos = os.listdir(videos_root)
    videos = [os.path.join(videos_root, i) for i in videos]
    return list(zip(midi_files, [None for _ in midi_files], videos))


def show(imgs):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
