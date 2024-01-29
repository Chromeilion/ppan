import csv
import json
import os
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import nvidia.dali.fn as fn
import nvidia.dali.plugin.pytorch.fn as pfn
import torch
import torchvision.transforms.functional as functional
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from rach3datautils.utils.dataset import DatasetUtils
from rach3datautils.utils.multimedia import MultimediaTools
from torch.utils.data import IterableDataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import VideoMAEImageProcessor

from ppan.config import seed
from ppan.midi import PPAnMidi
from ppan.types import PathLike

SAMPLE_TYPE = List[List[
    Tuple[PathLike, PathLike, PathLike, tuple[int, int, int, int]]]]


class BaseDataset(IterableDataset):
    """
    Base class for PPAn datasets.
    """
    def __init__(
            self,
            datasets: SAMPLE_TYPE,
            batch_size: Optional[int] = 4,
            epoch_size: Optional[int] = None,
            frame_transform: Optional[Callable[[torch.tensor],
                                               torch.tensor]] = None,
            video_transform: Optional[VideoMAEImageProcessor] = None,
            temporal_res: Optional[float] = None,
            temporal_size: Optional[float] = None,
            dataset_max_framerate: Optional[int] = None,
            step: Optional[int] = None,
            cachefile_name: Optional[str] = None,
            max_iters_per_epoch: Optional[int] = None
    ):
        """
        Parameters
        ----------
        datasets : SAMPLE_TYPE
            [[midi_path, flac_path, video_path, bounding_box]]
        epoch_size : Optional[int]
        frame_transform : Optional[Callable]
        video_transform : Optional[Callable]
            whether to shuffle the dataset every time a new generator loop is
            started.
        temporal_res : Optional[float]
            The distance in seconds between yielded frames.
        step : Optional[int]
            The amount of frames between consecutive sequences
        """
        if temporal_res is None:
            temporal_res = .03333
        if temporal_size is None:
            temporal_size = .54
        if dataset_max_framerate is None:
            dataset_max_framerate = 30
        if epoch_size is None:
            epoch_size = 1
        if step is None:
            step = 1
        if cachefile_name is None:
            cachefile_name = "./ppan_cache.txt"

        self.max_iters_per_epoch = max_iters_per_epoch
        self.cachefile_name = cachefile_name
        self.step = step
        self.datasets = datasets
        self.temporal_res = temporal_res
        self.temporal_size = temporal_size
        dataset_frametime = 1 / dataset_max_framerate
        self.no_frames_per_clip = int(self.temporal_size // dataset_frametime)
        self.temporal_res_frames = int(self.temporal_size // self.temporal_res)
        self.epoch_size: int = epoch_size

        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self._file_list = None
        self.midi_cache = defaultdict(dict)

        augmentations = [
            v2.UniformTemporalSubsample(self.temporal_res_frames),
            v2.ToDtype(torch.float, scale=True),
        ]
        if self.video_transform is not None:
            if self.video_transform.do_normalize:
                augmentations.append(
                    v2.Normalize(mean=self.video_transform.image_mean,
                                 std=self.video_transform.image_std)
                )
        self.augment = v2.Compose(augmentations)

        self.pipes = [
            self.video_pipe(
                batch_size=batch_size, num_threads=2,
                device_id=0, seed=seed, dataset_idx=i
            ) for i in range(len(self.datasets))
        ]
        self.dali_iter = DALIGenericIterator(
            self.pipes,
            ['pixel_values', 'label', 'note_vec',
             'timestamps', 'dataset_idx'],
            reader_name="VideoReader"
        )

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __len__(self):
        if self.max_iters_per_epoch is None:
            return self.epoch_size * len(self.dali_iter)
        return self.epoch_size * min(len(self.dali_iter),
                                     self.max_iters_per_epoch)

    def __iter__(self) -> dict[str, torch.Tensor]:
        for _ in range(self.epoch_size):
            for iter_no, [vals] in enumerate(self.dali_iter):
                vals['pixel_values'] = vals['pixel_values']
                vals['pixel_values'] = self.augment(vals['pixel_values'])
                yield self.finish_processing(vals)
                if self.max_iters_per_epoch is not None:
                    if iter_no > self.max_iters_per_epoch:
                        break
            # New epoch, we need to start from zero with the iterator
            self.dali_iter.reset()

    @abstractmethod
    def finish_processing(self, vals):
        ...

    def get_midi(self, dataset_idx, sample_idx) -> PPAnMidi:
        """Get the PPaNMidi object for a sample. All objects are
        automatically cached for future use.
        """
        if sample_idx not in self.midi_cache[dataset_idx]:
            sample = self.datasets[dataset_idx][sample_idx]
            vid_len = MultimediaTools().ff_probe(sample[2])
            vid_len = float(vid_len["streams"][0]["duration"])
            midi_path = sample[0]
            midi = PPAnMidi(temporal_res=self.temporal_res, lenience=1,
                            vid_len=vid_len)
            midi.set_midi(midi_path)
            self.midi_cache[dataset_idx][sample_idx] = midi

        return self.midi_cache[dataset_idx][sample_idx]

    @staticmethod
    def plot_image(img_tensor: torch.Tensor):
        """Plot a tensor image."""
        import matplotlib.pyplot as plt
        img_tensor_np = torch.swapaxes(img_tensor.detach().cpu(), 0, 2)
        plt.figure()
        plt.imshow(img_tensor_np)
        plt.show()

    @staticmethod
    def do_crop(video, box) -> torch.tensor:
        """Apply a crop using a bounding box."""
        bbx = v2.ConvertBoundingBoxFormat("XYWH")(box)
        return functional.crop(
            video, bbx[0, 0], bbx[0, 1], bbx[0, 2], bbx[0, 3])

    def midi_pipe_pytorch(self, label, dataset_idx, timestamps):
        midi = self.get_midi(dataset_idx.item(), label[0].item())
        # If the number of frames is even, we take the average between
        # the two middle frames. This means if one is positive and one
        # negative, we get a value of 0.5 instead of 1 or 0.
        if timestamps.shape[0] % 2 == 0:
            mid = timestamps.shape[0] // 2
            time_1 = timestamps[mid]
            time_2 = timestamps[mid+1]
            notes_vec_1 = midi(
                time_1,
                device=label.device,
                dtype=torch.float
            )
            notes_vec_2 = midi(
                time_2,
                device=label.device,
                dtype=torch.float
            )
            notes_vec = (notes_vec_1 + notes_vec_2) / 2
        # If the number of frames is odd, we can just use the middle one.
        else:
            time = timestamps[timestamps.shape[0] // 2]
            notes_vec = midi(
                time,
                device=label.device,
                dtype=torch.float
            )
        return notes_vec

    def video_pipe_pytorch(self, video, label, dataset_idx):
        crop = self.datasets[dataset_idx][label][-1]
        box = tv_tensors.BoundingBoxes(
            torch.tensor([crop[0], crop[2], crop[1], crop[3]]),
            format=tv_tensors.BoundingBoxFormat("XYXY"),
            canvas_size=video.shape[-2:]
        )
        video = self.do_crop(video, box)
        video = functional.resize(
            video,
            [self.video_transform.crop_size["height"],
             self.video_transform.crop_size["width"]]
        )
        return video

    def file_list(self, dataset_idx) -> str:
        if not os.path.exists(self.cachefile_name):
                self._file_list = self._get_file_list(dataset_idx)
                with open(self.cachefile_name, "wb") as f:
                    f.writelines([str.encode(i) for i in self._file_list])
        return self.cachefile_name

    def _get_file_list(self, dataset_idx) -> str:
        pad = (self.no_frames_per_clip // 2) * self.temporal_res
        file_list = []
        for sample_idx in range(len(self.datasets[dataset_idx])):
            file_list.append(self.get_midi(
                dataset_idx, sample_idx
            ).generate_filelist_labs(
                self.datasets[dataset_idx][sample_idx][2], sample_idx, pad)
            )
        return "\n".join(file_list)

    @abstractmethod
    @pipeline_def
    def video_pipe(self, dataset_idx: int):
        ...


class ImageVecDataset(BaseDataset):
    """
    Dataset object for training on a video/midi dataset such as Rach3.
    Handles loading, preprocessing, and batching all necessary files.
    """
    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec']}

    @pipeline_def
    def video_pipe(self, dataset_idx: int):
        video, label, timestamps = fn.readers.video(
            device="gpu",
            file_list=self.file_list(dataset_idx),
            enable_timestamps=True,
            sequence_length=self.no_frames_per_clip,
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            initial_fill=2,
            name="VideoReader",
            step=self.step,
            file_list_include_preceding_frame=True
        )
        video = fn.transpose(video, perm=[0, 3, 1, 2])
        video = pfn.torch_python_function(
            video, label,
            dataset_idx,
            function=self.video_pipe_pytorch
        )
        note_vec = pfn.torch_python_function(
            label, dataset_idx,
            timestamps,
            function=self.midi_pipe_pytorch
        )
        return video, label, note_vec, timestamps, dataset_idx


class EvalDataset(BaseDataset):
    """
    For evaluating on a video/midi dataset. Loads clips sequentially and
    returns the timestamp.
    """
    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec']}


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
