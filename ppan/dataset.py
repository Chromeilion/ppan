import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
import glob
from collections import defaultdict
import random

from ultralytics import YOLO
from tqdm import tqdm
import torch
import torchvision.tv_tensors
from rach3datautils.utils.dataset import DatasetUtils
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ppan.config import SAMPLE_TYPE, PathLike, TEST_TRAIN_SPLIT, device


class BaseDataset(Dataset):
    VID_FILENAME = "clip.mp4"
    LABEL_FILENAME = "labels.pt"

    def __init__(self, root, video_processor):
        self.root = Path(root)
        self.samples = sorted(os.listdir(self.root))
        self.videos = [os.path.join(self.root, i, self.VID_FILENAME) for i in
                       self.samples]
        self.video_processor = video_processor
        self._labs: dict[int, Optional[torch.Tensor]] = defaultdict(
            lambda: None
        )
        self._midpoint: Optional[int] = None

    def _get_lab(self, idx) -> torch.Tensor:
        if self._labs[idx] is None:
            self._labs[idx] = torch.load(os.path.join(
                self.root, self.samples[idx], self.LABEL_FILENAME),
                weights_only=True)
        return self._labs[idx]

    def _get_vid(self, idx) -> torch.Tensor:
        return self.video_processor(torchvision.io.read_video(
            self.videos[idx],
            pts_unit="sec",
            output_format="TCHW"
        )[0])

    def _get_midpoint(self, vid: torch.tensor) -> int:
        if self._midpoint is None:
            self._midpoint = int(vid.shape[0] // 2) + 1
        return self._midpoint

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._get_vid(idx), self._get_lab(idx)


class PretrainDataset(BaseDataset):
    """
    Dataset for masked pretraining.
    """
    def __init__(self, seq_length: int, mask_ratio: Optional[float] = None,
                 pad: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_ratio is None:
            mask_ratio = 0.9
        self.no_masks = round(seq_length*mask_ratio)
        self.seq_length = seq_length
        self.seq = range(seq_length)
        self.pad: Optional[int] = pad

    def __getitem__(self, idx):
        vid = self._get_vid(idx)
        if self.pad is not None:
            midpoint = self._get_midpoint(vid)
            vid = vid[midpoint-self.pad:midpoint+self.pad+1]
        bool_masked_pos = torch.zeros(
            self.seq_length,
            dtype=torch.bool
        )
        bool_masked_pos[random.sample(self.seq, self.no_masks)] = 1
        return {'pixel_values': vid,
                'bool_masked_pos': bool_masked_pos}


class PPAnTrainDataset(BaseDataset):
    """
    Dataset object for training on a video/midi dataset such as Rach3.
    Handles loading, preprocessing, and batching all necessary files.
    """
    BRIGHTNESS_JITTER = 0.01
    GAUSSIAN_MEAN = 0
    GAUSSIAN_STDDEV = 0.01

    @property
    def augmentations(self):
        if self._augment is None:
            self._augment = v2.Compose([
                v2.ToDtype(torch.float, scale=True),
                v2.RandomApply([
                    v2.ColorJitter(brightness=self.BRIGHTNESS_JITTER),
                    v2.GaussianNoise(mean=self.GAUSSIAN_MEAN,
                                     sigma=self.GAUSSIAN_STDDEV),
                ], p=0.4),
                self._get_color_augmentation()])
        return self._augment

    def finish_processing(self, vals):
        return {'pixel_values': vals['pixel_values'],
                'labels': vals['note_vec'],
                'sample': self.dataset[vals['label'].item()]}


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
    augment : bool

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


def load_pianoyt(root: PathLike) -> TEST_TRAIN_SPLIT:
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


def load_miditest(root) -> list[SAMPLE_TYPE]:
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


def load_all_data(rach3_dir: Optional[PathLike] = None,
                  pianoyt_dir: Optional[PathLike] = None,
                  miditest_dir: Optional[PathLike] = None):
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


def calculate_bounding_boxes(samples, yolo_model_checkpoint) -> list[SAMPLE_TYPE]:
    """
    Get bounding box predictions for a list of samples.
    Utilizes the Ultralytics package.

    Parameters
    ----------
    samples : list[ppan.dataset.SAMPLE_TYPE]
    yolo_model_checkpoint : PathLike

    Returns
    -------
    samples : ppan.dataset.SAMPLE_TYPE
        The same samples as passed in but with bounding boxes added.
    """
    # Load the model
    model = YOLO(yolo_model_checkpoint)

    new_samples = []
    for sample in tqdm(samples, desc="Calculating Bounding Boxes"):
        sample_preds = []
        video = sample[2]
        pred = model.predict(source=str(video), stream=True,
                             verbose=False, vid_stride=5)
        # Get predictions over the first 10 seconds.
        [sample_preds.append(next(pred)) for _ in range(5*10)]

        filtered_session_preds = [
            i for i in sample_preds if i.boxes.conf.shape[0] > 0
        ]
        best_pred = max(filtered_session_preds, key=lambda x: x.boxes.conf[0])
        bb_meta = json.loads(best_pred.tojson())[0]['box']
        bb = (round(bb_meta["y1"]), round(bb_meta["y2"]),
              round(bb_meta["x1"]), round(bb_meta["x2"]))
        new_sample = [i for i in sample]
        new_sample[3] = bb
        new_samples.append(new_sample)
    return new_samples


def load_omaps(root: PathLike) -> TEST_TRAIN_SPLIT:
    """
    Load all samples for the OMAPS dataset for use with PPAN.

    Returns
    -------
    samples : list[SAMPLE_TYPE]
    """
    test_dir = os.path.join(root, "test")
#    train_dir = os.path.join(root, "train")

    return _load_omaps_split(test_dir), None# _load_omaps_split(train_dir)


def _load_omaps_split(root: PathLike) -> list[SAMPLE_TYPE]:
    """Load a train or test split for the OMAPS dataset.
    """
    videos = [os.path.join(root, i) for i in
              sorted(glob.glob("*.mp4", root_dir=root))]
    labels = [os.path.join(root, i) for i in
              sorted(glob.glob("*.txt", root_dir=root))]
    none = [None for _ in videos]
    true = [True for _ in videos]
    false = [False for _ in videos]
    samples = list(zip(none, none, videos, none, true, false, labels))
#    samples = calculate_bounding_boxes(
#        samples,
#        "./model_weights/piano-detector-yolov8s.pt"
#    )
    return samples

