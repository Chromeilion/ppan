import os
from typing import Optional, Callable
import subprocess

import nvidia.dali.fn as fn
import nvidia.dali.plugin.pytorch.fn as pfn
import torch
import torchvision
import torchvision.transforms.functional as functional
import torchvision.transforms.v2 as v2
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.utils.data import IterableDataset
from torchvision import tv_tensors

from ppan.config import seed, processed_horizontal_res
from ppan.model import PPANVideoProcessor
from ppan.utils import Sample


class DatasetProcessor(IterableDataset):
    """
    Processor class that converts videos into cropped frames.
    """
    def __init__(
            self,
            datasets: list[Sample],
            batch_size: int,
            epoch_size: Optional[int] = None,
            frame_transform: Optional[Callable[[torch.tensor], torch.tensor]] = None,
            video_transform: Optional[PPANVideoProcessor] = None,
            temporal_res: Optional[float] = None,
            temporal_size: Optional[float] = None,
            dataset_max_framerate: Optional[int] = None,
            cachefile_name: Optional[str] = None,
            step: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        datasets : list[Sample]
        epoch_size : Optional[int]
        frame_transform : Optional[Callable]
        video_transform : Optional[Callable]
            Transformation applied to the entire video clip at once
        temporal_res : Optional[float]
            The distance in seconds between yielded frames.
        """
        if temporal_res is None:
            temporal_res = 1/30
        if temporal_size is None:
            temporal_size = 32/30
        if dataset_max_framerate is None:
            dataset_max_framerate = 30
        if epoch_size is None:
            epoch_size = 1
        if cachefile_name is None:
            cachefile_name = "./ppan_cache.txt"

        self.batch_size = batch_size
        self.cachefile_name = cachefile_name
        self.dataset: list[Sample] = datasets
        self.temporal_res = temporal_res
        self.temporal_size = temporal_size
        dataset_frametime = 1 / dataset_max_framerate
        self.no_frames_per_clip = int(self.temporal_size // dataset_frametime)
        if step is None:
            step = self.no_frames_per_clip
        self.step = step
        self.temporal_res_frames = int(self.temporal_size // self.temporal_res)
        self.epoch_size: int = epoch_size

        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self._file_list = None
        self._augment = None
        self.dali_iter = self.get_iter()

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __len__(self):
        iter_len = len(self.dali_iter) * int(os.environ.get("PPAN_NO_GPU", 1))
        return self.epoch_size * iter_len

    def __iter__(self) -> dict[str, torch.Tensor]:
        for epoch in range(self.epoch_size):
            iter_no = 0
            for vals in self.dali_iter:
                for val in vals:
                    yield self.finish_processing(val)
                    iter_no += 1

            # New epoch, we need to start from zero with the iterator
            self.dali_iter.reset()

    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'sample': [self.dataset[i] for i in vals['label']],
                'times': vals['timestamps']}

    def get_video_reader(self, num_gpus, d_id) -> fn.readers.video:
        return fn.readers.video(
            device="gpu",
            file_list=self.file_list(),
            enable_frame_num=True,
            sequence_length=self.no_frames_per_clip,
            file_list_frame_num=True,
            shard_id=d_id,
            file_list_include_preceding_frame=True,
            name=f"VideoReader",
            step=self.step,
            num_shards=num_gpus,
            pad_last_batch=True,
            pad_sequences=True,
        )

    @property
    def augmentations(self):
        if self._augment is None:
            self._augment = torch.nn.Sequential(
                v2.Resize(max_size=processed_horizontal_res,
                          size=None)
            )
        return self._augment

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
            ['pixel_values', 'label', 'timestamps'],
            reader_name="VideoReader"
        )
        return dali_iter

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

    def video_pipe_pytorch(self, video: torch.Tensor,
                           label: torch.Tensor):
        sample = self.dataset[label]
        crop = sample.bounding_box
        rotate_180 = sample.rotate_180

        # Some videos in PianoYT are the wrong resolution smh my head
        if sample.dataset == "pianoyt":
            if video.shape[-1] == 1920 and abs(crop[-1] - 1280) < abs(crop[-1] - 1920):
                video = v2.functional.resize(video, [720, 1280])
            elif video.shape[-1] == 1280 and abs(crop[-1] - 1280) > abs(crop[-1] - 1920):
                video = v2.functional.resize(video, [1080, 1920])

        if crop is not None:
            crop = torch.tensor(crop)
            box = tv_tensors.BoundingBoxes(
                torch.tensor([crop[0], crop[2], crop[1], crop[3]]),
                format=tv_tensors.BoundingBoxFormat("XYXY"),
                canvas_size=video.shape[2:]
            )
            video = self.do_crop(video, box)

        # Rotate the video into the correct orientation.
        if rotate_180:
            video = functional.rotate(video, 180)

        # Apply augmentations to the cropped video
        video = self.augmentations(video)
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video

    def file_list(self) -> str:
        if not os.path.exists(self.cachefile_name):
            self._file_list = self._get_file_list()
            with open(self.cachefile_name, "wb") as f:
                f.writelines([str.encode(i) for i in self._file_list])
        return self.cachefile_name

    def _get_file_list(self) -> str:
        file_list = []
        for sample_idx in range(len(self.dataset)):
            sample = self.dataset[sample_idx]
            n_frames = int(subprocess.run(
                f"ffprobe -v error -select_streams v:0 -count_packets "
                f"-show_entries stream=nb_read_packets -of csv=p=0 "
                f"{sample.video_path}".split(" "), capture_output=True).stdout.decode())

            file_list.append(
                f"{self.dataset[sample_idx].video_path} "
                f"{sample_idx} "
                f"0 "
                f"{n_frames}"
            )
        tot = []
        [tot.extend(i.split("\n")) for i in file_list]
        return "\n".join(file_list)

    @pipeline_def
    def video_pipe(self, num_gpus: int, d_id: int):
        video, label, timestamps = self.get_video_reader(
            num_gpus, d_id
        )
        video = fn.transpose(video, perm=[0, 3, 1, 2])
        video = pfn.torch_python_function(
            video, label,
            function=self.video_pipe_pytorch
        )
        return video, label, timestamps
