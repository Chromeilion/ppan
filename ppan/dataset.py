from pathlib import Path
import os
import itertools
import random
from typing import List, Tuple, Optional, Callable, TypedDict
import csv
import json

import numpy as np
import torch
import torchvision
import torchvision.utils
import torchvision.transforms.functional as f
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import IterableDataset
from torchvision.transforms import v2
from torchvision import tv_tensors

from ppan.config import VID_BACKEND, device
from ppan.preprocessing.midi import PPAnMidi
from ppan.types import PathLike


class VideoDatasetOutput(TypedDict):
    path: str
    video: torch.Tensor
    tgt_in: torch.Tensor
    target: torch.Tensor
    padding_mask: torch.Tensor
    start: float
    end: float


SAMPLE_TYPE = List[
    Tuple[PathLike, PathLike, PathLike, tuple[int, int, int, int]]]


class BaseDataset(IterableDataset):
    """
    Base class for PPAn datasets.
    """

    def __init__(self,
                 samples: SAMPLE_TYPE,
                 clips_per_vid: Optional[int] = None,
                 epoch_size: Optional[int] = None,
                 frame_transform: Optional[Callable] = None,
                 video_transform: Optional[Callable] = None,
                 shuffle_every_loop: Optional[bool] = None,
                 temporal_res: Optional[float] = None,
                 temporal_size: Optional[float] = None,
                 rotate: bool = False,
                 frame_dist_time: Optional[float] = None
                 ):
        """
        Parameters
        ----------
        samples : SAMPLE_TYPE
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
            temporal_res = .071
        if shuffle_every_loop is None:
            shuffle_every_loop = False
        if frame_dist_time is None:
            frame_dist_time = 0.5
        if temporal_size is None:
            temporal_size = .5

        self.frame_dist_time = frame_dist_time
        self.temporal_res = temporal_res
        self.temporal_size = temporal_size
        self.no_frames_per_clip = int(temporal_size // temporal_res)
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

    def __init__(self, samples: SAMPLE_TYPE,
                 *args, **kwargs):
        super().__init__(samples, *args, **kwargs)
        self.augment = v2.Compose([
            v2.UniformTemporalSubsample(self.no_frames_per_clip),
            v2.Grayscale(num_output_channels=3),
            v2.RandomHorizontalFlip(),
            v2.AutoAugment()
        ]
        )
        self.box_augment = v2.Compose([
            v2.RandomZoomOut(side_range=(1.1, 1.3)),
        ]
        )

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shuffle:
            random.shuffle(self.samples)
        for _ in range(self.epoch_size):
            for sample in self.samples:
                midi_path, flac_path, video_path, crop = sample

                video = torchvision.io.VideoReader(str(video_path))
                metadata = video.get_metadata()
                duration = metadata["video"]["duration"][0]
                frametime = 1. / metadata["video"]["fps"][0]

                midi = PPAnMidi()
                midi.set_midi(midi_path)

                start = random.uniform(
                    0.,
                    duration - self.temporal_res * self.clips_per -
                    self.frame_dist_time * 3
                )
                times = np.linspace(start,
                                    start + (
                                                self.temporal_res * self.clips_per),
                                    self.clips_per)
                return_list = []
                for time in times:
                    frames = []
                    for frame in itertools.takewhile(
                            lambda x: x[
                                          'pts'] <= time + self.temporal_size / 2.,
                            video.seek(time - self.temporal_size / 2.)):
                        img = frame['data']
                        if self.frame_transform is not None:
                            img = self.frame_transform(
                                img, return_tensors="pt")['pixel_values']
                        frames.append(img)

                    frames = torch.stack(frames, dim=0)

                    if self.rotate:
                        frames = f.rotate(frames, 180)

                    box = tv_tensors.BoundingBoxes(
                        torch.tensor([crop[0], crop[2], crop[1], crop[3]]),
                        format=tv_tensors.BoundingBoxFormat("XYXY"),
                        canvas_size=frames.shape[-2:]
                    )
                    box = self.box_augment(box)
                    frames = do_crop(frames, box)
                    frames = self.augment(frames)

                    if self.video_transform is not None:
                        frames = list(frames)
                        frames = torch.squeeze(self.video_transform(
                            frames, return_tensors="pt"
                        )["pixel_values"])

                    notes_vec = midi.midi_to_onset_offset_vec(
                        timestamps=(
                            time - self.temporal_res * 2,
                            time + self.temporal_res / 2)
                    )

                    yield {"pixel_values": frames,
                           "labels": notes_vec}


class EvalDataset(ImageVecDataset):
    """
    Simplified dataloader focussed solely around evaluation.
    """

    def __init__(self, samples: SAMPLE_TYPE,
                 *args, **kwargs):
        super().__init__(samples, *args, **kwargs)
        self.augment = v2.Grayscale(num_output_channels=3)

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in self.samples:
            try:
                for j in _eval_load(
                        sample=i,
                        frame_dist_time=self.frame_dist_time,
                        temporal_res=self.temporal_res,
                        rotate=self.rotate,
                        augment=self.augment
                ):
                    yield j
            except RuntimeError:
                continue


def _eval_load(sample, frame_dist_time, temporal_res, rotate,
               augment):
    crop = None
    if len(sample) == 4:
        midi_path, flac_path, video_path, crop = sample
    else:
        midi_path, flac_path, video_path = sample

    video = torchvision.io.VideoReader(str(video_path), stream='video')

    metadata = video.get_metadata()
    frametime = 1. / metadata["video"]["fps"][0]

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

        current_time = rotlist[frame_dist + 1]['pts'] - frame_dist * frametime

        frame_data_prev = rotlist[0]['data'].to(device)
        frame_data_mid = rotlist[frame_dist + 1]['data'].to(device)
        frame_data_next = frame['data'].to(device)

        stacked_frames = torch.stack([frame_data_prev, frame_data_mid,
                                      frame_data_next], dim=0)

        # CUDA has a different API than the other backends, so gotta
        # swap some axis
        if VID_BACKEND == "cuda":
            stacked_frames = fix_cuda_axes(stacked_frames)

        if rotate:
            stacked_frames = f.rotate(stacked_frames, 180)

        stacked_frames = resize_correct(stacked_frames)
        stacked_frames = augment(stacked_frames)
        img = combine_imgs(stacked_frames)

        try:
            notes_str = midi.midi_to_notes(
                timestamps=(current_time - frametime / 2,
                            current_time + frametime / 2)
            )
        except AttributeError:
            raise RuntimeError

        if notes_str:
            notes = notes_str
        else:
            notes = []
        img.share_memory_()
        yield {"pixel_values": img,
               "labels": notes,
               "midi_path": str(midi_path),
               "video_path": str(video_path),
               'time': current_time}


def combine_imgs(img):
    new_img = torch.zeros(size=(img.size()[1],
                                680 * 3,
                                img.size()[3]),
                          dtype=torch.uint8).to(img.device)

    new_img[:, :680, :] = img[0]
    new_img[:, 680:1360, :] = img[1]
    new_img[:, 1360:2040, :] = img[2]
    return new_img


def resize_correct(img):
    return f.resize(img,
                    size=[680, 1920],
                    antialias=False)


def fix_cuda_axes(img):
    img = torch.swapaxes(img, 1, 3)
    img = torch.swapaxes(img, 2, 3)
    return img


def do_crop(img, crop_vals):
    bbx = v2.ConvertBoundingBoxFormat("XYWH")(crop_vals)
    return f.crop(img, bbx[0, 0], bbx[0, 1], bbx[0, 2], bbx[0, 3])


def load_rach3(root: PathLike):
    """
    Load the dataset for use with PPAn.

    Parameters
    ----------
    root : PathLike

    Returns
    -------
    test : List[Tuple[PathLike, PathLike, PathLike]]
    train : List[Tuple[PathLike, PathLike, PathLike]]
    """
    root = Path(root)
    test, train = root / "test", root / "train"
    bbs_path = root / "rach3_bounding_boxes.json"

    with open(bbs_path, "r") as f:
        bbs = json.load(f)
    bbs = {i["session_id"]: i["box"] for i in bbs}
    return load_rach3_split(test, bbs), load_rach3_split(train, bbs)


def load_rach3_split(root: PathLike,
                     bbs: dict) -> List[Tuple[PathLike, PathLike, PathLike]]:
    """
    Load a folder containing Rach3 files (such as test or train folders)

    Parameters
    ----------
    root : PathLike
    bbs : dict

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
        bb_meta = bbs[str(i.id)][0]["box"]
        bb = (round(bb_meta["y1"]), round(bb_meta["y2"]),
              round(bb_meta["x1"]), round(bb_meta["x2"]))
        crops = [bb for _ in range(len(i.midi.splits_list))]
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
        video = os.path.join(root, f'videos/{Path(i[0]).name}')
        crop = [int(j) for j in i[4:]]
        tup = (os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
               None, video, crop)
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
