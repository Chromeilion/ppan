import csv
import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union

import nvidia.dali.fn as fn
import nvidia.dali.plugin.pytorch.fn as pfn
import torch
import torchvision.transforms.functional as functional
import torchvision.tv_tensors
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

PathLike = Union[str, bytes, os.PathLike]

# [[midi_path, flac_path, video_path, bounding_box, whether to rotate 180,
#   random_resize]]
SAMPLE_TYPE = List[
    Tuple[PathLike, Optional[PathLike], PathLike,
          Optional[tuple[int, int, int, int]], bool, bool]
]


# Tensor GN pipeline, taken from:
# https://github.com/pytorch/vision/issues/6192#issuecomment-1164176231
def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 5.0

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


class BaseDataset(IterableDataset):
    """
    Base class for PPAn datasets.
    """
    def __init__(
            self,
            datasets: SAMPLE_TYPE,
            batch_size: int,
            epoch_size: Optional[int] = None,
            frame_transform: Optional[Callable[[torch.tensor],
                                               torch.tensor]] = None,
            video_transform: Optional[VideoMAEImageProcessor] = None,
            temporal_res: Optional[float] = None,
            temporal_size: Optional[float] = None,
            dataset_max_framerate: Optional[int] = None,
            step: Optional[int] = None,
            cachefile_name: Optional[str] = None,
            max_iters_per_epoch: Optional[int] = None,
            target_epoch: Optional[int] = None
    ):
        """
        Parameters
        ----------
        datasets : SAMPLE_TYPE
            [[midi_path, flac_path, video_path, bounding_box, rotate 180]]
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
            step = 2
        if cachefile_name is None:
            cachefile_name = "./ppan_cache.txt"

        self.batch_size = batch_size
        self.lenience = 1
        self.max_iters_per_epoch = max_iters_per_epoch
        self.cachefile_name = cachefile_name
        self.step = step
        self.dataset = datasets
        self.temporal_res = temporal_res
        self.temporal_size = temporal_size
        dataset_frametime = 1 / dataset_max_framerate
        self.no_frames_per_clip = int(self.temporal_size // dataset_frametime)
        self.temporal_res_frames = int(self.temporal_size // self.temporal_res)
        self.epoch_size: int = epoch_size

        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self._file_list = None
        self.midi_cache: dict[int, PPAnMidi] = {}

        self._augment = None
        self.dali_iter = self.get_iter()

        if target_epoch is not None:
            self.target_iteration = len(self) * target_epoch

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __len__(self):
        if self.max_iters_per_epoch is None:
            return self.epoch_size * len(self.dali_iter)
        return self.epoch_size * min(len(self.dali_iter),
                                     self.max_iters_per_epoch)

    def __iter__(self) -> dict[str, torch.Tensor]:
        for epoch in range(self.epoch_size):
            iter_no = 0
            for vals in self.dali_iter:
                for val in vals:
                    yield self.finish_processing(val)
                    iter_no += 1
                    if self.max_iters_per_epoch is not None:
                        if iter_no >= self.max_iters_per_epoch:
                            break
                if self.max_iters_per_epoch is not None:
                    if iter_no >= self.max_iters_per_epoch:
                        break
            # New epoch, we need to start from zero with the iterator
            self.dali_iter.reset()

    @abstractmethod
    def finish_processing(self, vals):
        ...

    @abstractmethod
    def get_video_reader(self, num_gpus, d_id) -> fn.readers.video:
        ...

    @property
    @abstractmethod
    def augmentations(self):
        ...

    def get_iter(self):
        n = int(os.environ.get("PPAN_NO_GPU", 1))
        num_threads = 4
        pipes = [
            self.video_pipe(
                batch_size=self.batch_size,
                num_threads=num_threads,
                device_id=i,
                seed=seed,
                num_gpus=n,
                d_id=i
            ) for i in range(n)
        ]
        dali_iter = DALIGenericIterator(
            pipes,
            ['pixel_values', 'label', 'note_vec',
             'timestamps'],
            reader_name="VideoReader"
        )
        return dali_iter

    def get_midi(self, sample_idx) -> PPAnMidi:
        """Get the PPaNMidi object for a sample. All objects are
        automatically cached for future use.
        """
        if sample_idx not in self.midi_cache:
            sample = self.dataset[sample_idx]
            vid_len = MultimediaTools().ff_probe(sample[2])
            vid_len = float(vid_len["streams"][0]["duration"])
            midi_path = sample[0]
            midi = PPAnMidi(temporal_res=self.temporal_res,
                            lenience=self.lenience,
                            vid_len=vid_len)
            midi.set_midi(midi_path)
            self.midi_cache[sample_idx] = midi

        return self.midi_cache[sample_idx]

    @staticmethod
    def plot_image(img_tensor: torch.Tensor):
        """Plot a tensor image."""
        import matplotlib.pyplot as plt
        img_tensor = torch.swapaxes(img_tensor, 0, 2)
        img = img_tensor.detach().cpu()
        plt.figure()
        plt.imshow(img)
        plt.show()

    @staticmethod
    def do_crop(video: torch.Tensor,
                box: torchvision.tv_tensors.TVTensor) -> torch.tensor:
        """Apply a crop using a bounding box."""
        bbx = v2.ConvertBoundingBoxFormat("XYWH")(box)
        return functional.crop(
            video, bbx[0, 0], bbx[0, 1], bbx[0, 2], bbx[0, 3])

    def midi_pipe_pytorch(self, label: torch.Tensor,
                          timestamps: torch.Tensor):
        """Load labels from the correct MIDI file.
        """
        if self.dataset[label[0].item()][0] is None:
            return torch.tensor([0], device=label.device)

        midi = self.get_midi(label[0].item())
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

    def video_pipe_pytorch(self, video: torch.Tensor,
                           label: torch.Tensor):
        sample = self.dataset[label]
        crop = sample[3]
        rotate_180 = sample[4]
        random_resize = sample[5]
        if crop is not None:
            crop = torch.tensor(crop)
            if random_resize:
                # Randomly resize the crop as an augmentation
                rand_am = torch.randint(-20, 20, [2])
                rand_am2 = torch.randint(-20, 20, [2])
                crop[[1, 3]] += rand_am
                crop[[0, 2]] -= rand_am2

            box = tv_tensors.BoundingBoxes(
                torch.tensor([crop[0], crop[2], crop[1], crop[3]]),
                format=tv_tensors.BoundingBoxFormat("XYXY"),
                canvas_size=video.shape[2:]
            )
            video = self.do_crop(video, box)

        # Rotate the video into the correct orientation.
        if rotate_180:
            video = functional.rotate(video, 180)

        # Wrap the rectangle such that it fits into a square better.
        vid_size = list(video.size())
        vid_size[-2] = vid_size[-2] * 2
        vid_size[-1] = vid_size[-1] // 2
        wrapped = torch.zeros(size=vid_size, dtype=video.dtype).to(video.device)
        wrapped[..., :video.shape[-2], :] = video[..., :vid_size[-1]]
        wrapped[..., video.shape[-2]:, :] = video[..., vid_size[-1]:vid_size[-1]*2]
        video = wrapped

        video = self._pad_to_square(video)

        # Although we apply the model transformation later, we resize
        # here in order to reduce the amount of used memory.
        video = functional.resize(
            video,
            [self.video_transform.crop_size["height"],
             self.video_transform.crop_size["width"]]
        )
        # Apply augmentations to the cropped video
        video = self.augmentations(video)

        return video

    @staticmethod
    def _pad_to_square(video: torch.Tensor):
        # Pad the video into a square shape to prevent any more
        # changes to the aspect ratio.
        shortest_dim = torch.argmin(torch.tensor(video.shape[-2:]))
        max_dim = torch.argmax(torch.tensor(video.shape[-2:]))

        amount_to_pad = (video.shape[-2:][max_dim] -
                         video.shape[-2:][shortest_dim])
        if shortest_dim == 1:
            video = v2.functional.pad(video, [amount_to_pad, 0])
        else:
            video = v2.functional.pad(video, [0, amount_to_pad])
        return video

    def file_list(self) -> str:
        if not os.path.exists(self.cachefile_name):
            self._file_list = self._get_file_list()
            with open(self.cachefile_name, "wb") as f:
                f.writelines([str.encode(i) for i in self._file_list])
        return self.cachefile_name

    def _get_file_list(self) -> str:
        pad = (self.no_frames_per_clip // 2) * self.temporal_res
        file_list = []
        for sample_idx in range(len(self.dataset)):
            file_list.append(self.get_midi(
                sample_idx
            ).generate_filelist_labs(
                self.dataset[sample_idx][2], sample_idx, pad)
            )
        return "\n".join(file_list)

    @pipeline_def()
    def video_pipe(self, num_gpus: int, d_id: int):
        video, label, timestamps = self.get_video_reader(
            num_gpus, d_id
        )
        video = fn.transpose(video, perm=[0, 3, 1, 2])
        video = pfn.torch_python_function(
            video, label,
            function=self.video_pipe_pytorch
        )
        note_vec = pfn.torch_python_function(
            label,
            timestamps,
            function=self.midi_pipe_pytorch
        )
        return video, label, note_vec, timestamps


class PPAnTrainDataset(BaseDataset):
    """
    Dataset object for training on a video/midi dataset such as Rach3.
    Handles loading, preprocessing, and batching all necessary files.
    """
    @property
    def augmentations(self):
        if self._augment is None:
            augmentations = [
                v2.UniformTemporalSubsample(self.temporal_res_frames),
                v2.RandomApply([
                    v2.ColorJitter(hue=0.1,
                                   brightness=0.2,
                                   contrast=0.1,
                                   saturation=0.1)
                ],
                    p=0.25
                ),
                v2.ToDtype(torch.float, scale=True),
            ]
            if self.video_transform is not None:
                if self.video_transform.do_normalize:
                    augmentations.append(
                        v2.Normalize(mean=self.video_transform.image_mean,
                                     std=self.video_transform.image_std)
                    )
            self._augment = v2.Compose(augmentations)
        return self._augment

    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec']}

    def get_video_reader(self, num_gpus, d_id
                         ) -> fn.readers.video:
        return fn.readers.video(
            device="gpu",
            file_list=self.file_list(),
            enable_timestamps=True,
            sequence_length=self.no_frames_per_clip,
            shard_id=d_id,
            random_shuffle=True,
            initial_fill=2,
            name=f"VideoReader",
            step=self.step,
            file_list_include_preceding_frame=True,
            num_shards=num_gpus
        )


class PPAnEvalDataset(BaseDataset):
    """
    For evaluating on a video/midi dataset. Loads clips sequentially and
    returns the timestamp.
    """
    @property
    def augmentations(self):
        if self._augment is None:
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
            self._augment = v2.Compose(augmentations)
        return self._augment

    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec'],
                'timestamps': vals['timestamps'],
                'file_idx': vals['label']}

    def get_all_video_samples(self):
        all_vids = []
        [all_vids.append(str(i[2])) for i in self.dataset]
        return all_vids

    def get_video_reader(self, num_gpus, d_id) -> fn.readers.video:
        return fn.readers.video(
            device="gpu",
            filenames=self.get_all_video_samples(),
            labels=[],
            enable_timestamps=True,
            sequence_length=self.no_frames_per_clip,
            shard_id=d_id,
            num_shards=num_gpus,
            random_shuffle=False,
            initial_fill=2,
            name=f"VideoReader",
            step=1,
            file_list_include_preceding_frame=True,
        )


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
    return (load_rach3_split(test, bbs, False),
            load_rach3_split(train, bbs, True))


def load_rach3_split(root: PathLike,
                     bbs: dict,
                     augment: bool) -> SAMPLE_TYPE:
    """
    Load a folder containing Rach3 files (such as test or train folders)

    Parameters
    ----------
    root : PathLike
    bbs : dict

    Returns
    -------
    samples : SAMPLE_TYPE
    """
    dataset = DatasetUtils(root)
    sessions = dataset.remove_noncomplete(
        subsession_list=dataset.get_sessions(),
        required=["midi.splits_list", "flac.splits_list",
                  "video.splits_list"]
    )
    samples: SAMPLE_TYPE = []
    for i in sessions:
        bb_meta = bbs[str(i.id)][0]["box"]
        bb = (round(bb_meta["y1"]), round(bb_meta["y2"]),
              round(bb_meta["x1"]), round(bb_meta["x2"]))
        crops = [bb for _ in range(len(i.midi.splits_list))]
        [samples.append(j) for j in zip(i.midi.splits_list,
                                        i.flac.splits_list,
                                        i.video.splits_list,
                                        crops,
                                        [False for _ in range(len(crops))],
                                        [augment for _ in range(len(crops))])]

    return samples


def load_pianoyt(root: PathLike):
    data = []
    with open(os.path.join(root, "dataset.csv"), "r") as f:
        reader = csv.reader(f)
        [data.append(i) for i in reader]

    samples_train = []
    samples_test = []
    for i in data:
        video = os.path.join(root, f'processed_videos/{Path(i[0]).name}')
        video = video.replace(" ", "_")
        crop = [int(j) for j in i[4:]]
        crop = [crop[0], crop[1], crop[2], crop[3]]
        tup = [os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
               None, video, crop, True, True]
        if i[3] == "1":
            samples_train.append(tup)
        elif i[3] == "3":
            tup[-1] = False
            samples_test.append(tup)

    return samples_test, samples_train


def load_miditest(root):
    midi_root = os.path.join(root, "miditest_MIDI")
    midi_files = os.listdir(midi_root)
    midi_files = [os.path.join(midi_root, i) for i in midi_files]
    videos_root = os.path.join(root, "miditest_processed_videos")
    videos = os.listdir(videos_root)
    videos = [os.path.join(videos_root, i) for i in videos]
    none = [None for _ in midi_files]
    true = [True for _ in midi_files]
    false = [False for _ in midi_files]
    return list(zip(midi_files, none, videos, none, true, false))


def load_all_data(rach3_dir: Optional[PathLike],
                  pianoyt_dir: Optional[PathLike],
                  miditest_dir: Optional[PathLike]):
    """Load Rach3, pianoYT, and Miditest and put them into train, test and
    validation splits.
    """
    test = []
    train = []
    miditest = None
    pianoyt_test = None
    rach3_test = None
    if rach3_dir is not None:
        rach3_test, rach3_train = load_rach3(rach3_dir)
        test.extend(rach3_test)
        train.extend(rach3_train)
    if pianoyt_dir is not None:
        pianoyt_test, pianoyt_train = load_pianoyt(pianoyt_dir)
        test.extend(pianoyt_test)
        train.extend(pianoyt_train)
    if miditest_dir is not None:
        miditest = load_miditest(miditest_dir)

    return test, train, miditest, pianoyt_test, rach3_test
