import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable

import nvidia.dali.fn as fn
import nvidia.dali.plugin.pytorch.fn as pfn
import torch
import torchvision
import torchvision.transforms.functional as functional
import torchvision.transforms.v2 as v2
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from rach3datautils.utils.multimedia import MultimediaTools
from torch.utils.data import IterableDataset
from torchvision import tv_tensors
from transformers import VideoMAEImageProcessor

from ppan.config import seed, SAMPLE_TYPE, processed_horizontal_res
from ppan.midi import PPAnMidi


class BaseDatasetProcessor(IterableDataset):
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
            target_epoch: Optional[int] = None,
            pretrain: Optional[bool] = None
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
            temporal_res = 1/30
        if temporal_size is None:
            temporal_size = 7/30
        if dataset_max_framerate is None:
            dataset_max_framerate = 30
        if epoch_size is None:
            epoch_size = 1
        if step is None:
            step = 1
        if cachefile_name is None:
            cachefile_name = "./ppan_cache.txt"
        if pretrain is None:
            pretrain = False

        self.pretrain = pretrain
        self.batch_size = batch_size
        self.lenience = 0
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
        r_resize_int = os.environ.get(
            "PPAN_AUG_CROP_JITTER", None)
        if r_resize_int is not None:
            r_resize_int = tuple([int(i) for i in r_resize_int.split(":")])
        self.rand_resize_interval: Optional[tuple[int, int]] = r_resize_int
        self.do_stack: bool = os.environ.get("PPAN_AUG_STACK", "True") == "True"
        self.dali_iter = self.get_iter()

        if target_epoch is not None:
            self.target_iteration = len(self) * target_epoch

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __len__(self):
        iter_len = len(self.dali_iter) * int(os.environ.get("PPAN_NO_GPU", 1))
        if self.max_iters_per_epoch is None:
            return self.epoch_size * iter_len
        return self.epoch_size * min(iter_len, self.max_iters_per_epoch)

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
                d_id=i,
                set_affinity=True
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
            vid_meta = MultimediaTools().ff_probe(sample[2])
            n_frames = int(vid_meta["streams"][0]["nb_frames"])
            midi_path = sample[0]
            midi = PPAnMidi(temporal_res=self.temporal_res,
                            lenience=self.lenience,
                            n_frames=n_frames)
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
        time = timestamps.item() + self.temporal_res_frames // 2
        notes_vec = midi(
            time,
            device=label.device,
            dtype=torch.bool
        )
        return notes_vec.float()

    def video_pipe_pytorch(self, video: torch.Tensor,
                           label: torch.Tensor):
        sample = self.dataset[label]
        crop = sample[3]
        rotate_180 = sample[4]
        random_resize = sample[5]

        # Some videos in PianoYT are the wrong resolution smh my head
        if "PianoYT" in Path(sample[0]).parts:
            if video.shape[-1] == 1920 and abs(crop[-1] - 1280) < abs(crop[-1] - 1920):
                video = v2.functional.resize(video, [720, 1280])
            elif video.shape[-1] == 1280 and abs(crop[-1] - 1280) > abs(crop[-1] - 1920):
                video = v2.functional.resize(video, [1080, 1920])
        if crop is not None:
            crop = torch.tensor(crop)
            if random_resize:
                # Randomly resize the crop as an augmentation
                rand_am = torch.randint(
                    self.rand_resize_interval[0],
                    self.rand_resize_interval[1],
                    [4]
                )
                # We either have to subtract or add depending on whether
                # it's the left edge, right edge, top edge or bottom edge.
                crop[[1, 3]] += rand_am[:2]
                crop[[0, 2]] -= rand_am[2:]

            box = tv_tensors.BoundingBoxes(
                torch.tensor([crop[0], crop[2], crop[1], crop[3]]),
                format=tv_tensors.BoundingBoxFormat("XYXY"),
                canvas_size=video.shape[2:]
            )
            video = self.do_crop(video, box)

        # Rotate the video into the correct orientation.
        if rotate_180:
            video = functional.rotate(video, 180)

        try:
            if sample[6] is not None:
                video = functional.hflip(video)
        except IndexError:
            pass

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
        pad = int((self.temporal_size // self.temporal_res) // 2)
        file_list = []
        for sample_idx in range(len(self.dataset)):
            file_list.append(self.get_midi(
                sample_idx
            ).generate_filelist_labs(
                self.dataset[sample_idx][2], sample_idx, pad)
            )
        tot = []
        [tot.extend(i.split("\n")) for i in file_list]
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
        note_vec = 0
        if not self.pretrain:
            note_vec = pfn.torch_python_function(
                label,
                timestamps,
                function=self.midi_pipe_pytorch
            )
        return video, label, note_vec, timestamps

    def _get_color_augmentation(self):
        if os.environ.get("PPAN_AUG_GREYSCALE", "True") == "False":
            if self.video_transform is not None:
                if self.video_transform.do_normalize:
                    return v2.Normalize(mean=self.video_transform.image_mean,
                                        std=self.video_transform.image_std)
        else:
            return v2.Grayscale(num_output_channels=1)


class PPAnDatasetProcessor(BaseDatasetProcessor):
    """
    Dataset object for training on a video/midi dataset such as Rach3.
    Handles loading, preprocessing, and batching all necessary files.
    """
    @property
    def augmentations(self):
        if self._augment is None:
            self._augment = torch.nn.Sequential(
                v2.Grayscale(num_output_channels=1),
                v2.Resize(max_size=processed_horizontal_res,
                          size=None)
            )
        return self._augment

    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec'],
                'sample': self.dataset[vals['label'].item()]}

    def get_video_reader(self, num_gpus, d_id) -> fn.readers.video:
        return fn.readers.video(
            device="gpu",
            file_list=self.file_list(),
            enable_frame_num=True,
            sequence_length=self.no_frames_per_clip,
            file_list_frame_num=True,
            shard_id=d_id,
            random_shuffle=True,
            file_list_include_preceding_frame=True,
            name=f"VideoReader",
            step=self.step,
            num_shards=num_gpus
        )

    def _get_color_augmentation(self):
        if os.environ.get("PPAN_AUG_GREYSCALE", "True") == "False":
            return lambda x: x


class PPAnEvalDataset(BaseDatasetProcessor):
    """
    For evaluating on a video/midi dataset. Loads clips sequentially and
    returns the timestamp.
    """
    @property
    def augmentations(self):
        if self._augment is None:
            self._augment = v2.Compose([v2.ToDtype(torch.float, scale=True),
                                        self._get_color_augmentation()])
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
            step=self.step,
            file_list_include_preceding_frame=True,
        )
