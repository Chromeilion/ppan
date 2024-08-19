"""
Misc. utility functions used across the package.
"""
import csv
import glob
import json
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from rach3datautils.utils.dataset import DatasetUtils

from ppan.config import PathLike, SAMPLE_TYPE, TEST_TRAIN_SPLIT


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


def load_omaps(root: PathLike) -> TEST_TRAIN_SPLIT:
    """
    Load all samples for the OMAPS dataset for use with PPAN.

    Returns
    -------
    samples : list[SAMPLE_TYPE]
    """
    test_dir = os.path.join(root, "test")
    train_dir = os.path.join(root, "train")

    return _load_omaps_split(test_dir), _load_omaps_split(train_dir)


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
    samples = calculate_bounding_boxes(
        samples,
        "./model_weights/piano-detector-yolov8s.pt"
    )
    return samples


# From VideoMAE:
# https://github.com/MCG-NJU/VideoMAE/blob/main/masking_generator.py
class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame,
                     dtype=bool),
            np.ones(self.num_masks_per_frame,
                    dtype=bool),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask

def patchify(video: torch.tensor, p: int, tu: int):
    """Patchify a batched video tensor of shape BTCHW.
    """
    h = video.shape[3] // p
    w = video.shape[4] // p
    t = video.shape[1] // tu
    c = video.shape[2]

    patches = video.reshape(
        shape=(video.shape[0], t, tu, c, h, p, w, p))
    patches = torch.einsum('ntuchpwq->nthwupqc', patches)
    patches = patches.reshape(patches.shape[0], -1, tu * p ** 2)
    return patches


def unpatchify(patches: torch.tensor, p: int, tu: int, t: int, h: int, w: int, c: int):
    patches = patches.reshape(shape=(patches.shape[0], t//tu, h//p, w//p, tu, p, p, c))
    patches = torch.einsum('nthwupqc->ntuchpwq', patches)
    return patches.reshape(shape=(patches.shape[0], t, c, h, w))


def count_pos_neg_samples(dataset):
    pos, neg = 0, 0
    batch_size = 512
    for sample in tqdm(DataLoader(dataset, batch_size=batch_size,
                                  num_workers=os.cpu_count()-2)):
        lab = sample["label_ids"]
        pos += torch.sum(lab > 0.0001, dim=0) / batch_size
        neg += torch.sum(lab < 0.4, dim=0) / batch_size
    return neg/pos

def freeze_pretrained_weights(model) -> None:
    """Freeze the pretrained model weights. Useful for transfer learning.
    The model should be some Huggingface pretrained model.
    """
    for param in model.base_model.parameters():
        param.requires_grad = False
