"""
Misc. utility functions used across the package.
"""
import csv
import glob
import json
import os
from pathlib import Path
from typing import Optional, Literal
from collections import OrderedDict
from dataclasses import dataclass

import torchvision.transforms.v2 as v2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rach3datautils.utils.dataset import DatasetUtils
from timm import create_model

import ppan.model_mae
from ppan.config import PathLike, TEST_TRAIN_SPLIT


ds_prefixes = Literal["r3x", "r3s", "pianoyt", "miditest"]

@dataclass
class Sample:
    midi_path: PathLike
    video_path: PathLike
    dataset: ds_prefixes
    bounding_box: Optional[tuple[int, int, int, int]] = None
    flac_path: Optional[PathLike] = None
    rotation_factor: float = 0
    rotate_180: bool = False


def load_rach3(root: PathLike, ds_prefix: ds_prefixes = "r3s"):
    """
    Load the dataset for use with PPAn.

    Parameters
    ----------
    root : PathLike
    ds_prefix : ds_prefixes

    Returns
    -------
    test : List[Tuple[PathLike, PathLike, PathLike]]
    train : List[Tuple[PathLike, PathLike, PathLike]]
    """
    root = Path(root)
    test, train = root / "test", root / "train"
    bbs_path = root / f"{ds_prefix}_bounding_boxes.json"

    with open(bbs_path, "r") as f:
        meta = json.load(f)
    bbs = {"_".join(key.split("_")[:-1]): value for key, value in meta.items()}
    rot = {"_".join(key.split("_")[:-1]): value["rot_angle"] for key, value in meta.items()}

    return (load_rach3_split(test, bbs, rot, False, ds_prefix),
            load_rach3_split(train, bbs, rot, True, ds_prefix))


def load_rach3_split(root: PathLike,
                     bbs: dict,
                     rot: dict,
                     augment: bool,
                     ds_prefix: ds_prefixes) -> list[Sample]:
    """
    Load a folder containing Rach3 fi        crops = [bb for _ in range(len(i.midi.splits_list))]
        rots = [r for _ in range(len(i.midi.splits_list))]les (such as test or train folders)

    Parameters
    ----------
    root : PathLike
    bbs : dict
    rot : dict
    augment : bool
    ds_prefix : ds_prefixes

    Returns
    -------
    samples : list[Sample]
    """
    dataset = DatasetUtils(root)
    sessions = dataset.remove_noncomplete(
        subsession_list=dataset.get_sessions(),
        required=["midi.splits_list", "flac.splits_list",
                  "video.splits_list"]
    )
    samples: list[Sample] = []
    for i in sessions:
        bb_meta = bbs[str(i.id)]
        bb = (int(bb_meta["y1"]), int(bb_meta["y2"]),
              int(bb_meta["x1"]), int(bb_meta["x2"]))
        r = float(rot[str(i.id)])
        for mid, flac, video in zip(i.midi.splits_list,
                                    i.flac.splits_list,
                                    i.video.splits_list):
            sample = Sample(
                midi_path=mid,
                flac_path=flac,
                video_path=video,
                bounding_box=bb,
                dataset=ds_prefix,
                rotation_factor=r,
                rotate_180=False
            )
            samples.append(sample)
    return samples


def load_pianoyt(root: PathLike) -> tuple[list[Sample], list[Sample]]:
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
        crop = (crop[0], crop[1], crop[2], crop[3])
        tup = [os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
               None, video, crop, True, True]
        sample = Sample(
            midi_path=os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
            video_path=video,
            bounding_box=crop,
            rotate_180=True,
            dataset="pianoyt"
        )
        if i[3] == "1":
            samples_train.append(sample)
        elif i[3] == "3":
            tup[-1] = False
            samples_test.append(sample)

    return samples_test, samples_train


def load_miditest(root) -> list[Sample]:
    midi_root = os.path.join(root, "miditest_MIDI")
    midi_files = os.listdir(midi_root)
    midi_files = [os.path.join(midi_root, i) for i in midi_files]
    videos_root = os.path.join(root, "miditest_processed_videos")
    videos = os.listdir(videos_root)
    videos = [os.path.join(videos_root, i) for i in videos]
    samples = [Sample(
        midi_path=mid,
        video_path=vid,
        rotate_180=True,
        dataset="miditest"
    ) for vid, mid in zip(videos, midi_files)]
    return samples


def load_all_data(rach3_s_dir: Optional[PathLike] = None,
                  rach3_x_dir: Optional[PathLike] = None,
                  pianoyt_dir: Optional[PathLike] = None,
                  miditest_dir: Optional[PathLike] = None):
    """Load Rach3, pianoYT, and Miditest and put them into train, test and
    validation splits.
    """
    test = []
    train = []
    miditest = None
    pianoyt_test = None
    rach3_s_test = None
    rach3_x_test = None
    if rach3_s_dir is not None:
        rach3_s_test, rach3_s_train = load_rach3(rach3_s_dir, "r3s")
        test.extend(rach3_s_test)
        train.extend(rach3_s_train)
    if rach3_x_dir is not None:
        rach3_x_test, rach3_x_train = load_rach3(rach3_x_dir, "r3x")
        test.extend(rach3_x_test)
        train.extend(rach3_x_train)
    if pianoyt_dir is not None:
        pianoyt_test, pianoyt_train = load_pianoyt(pianoyt_dir)
        test.extend(pianoyt_test)
        train.extend(pianoyt_train)
    if miditest_dir is not None:
        miditest = load_miditest(miditest_dir)

    return test, train, miditest, pianoyt_test, rach3_s_test, rach3_x_test


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


def _load_omaps_split(root: PathLike) -> list[Sample]:
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

def freeze_weights(model) -> None:
    """Freeze the pretrained model weights. Useful for transfer learning.
    The model should be some Huggingface pretrained model.
    """
    for param in model.base_model.parameters():
        param.requires_grad = False


def unfreeze_weights(model) -> None:
    """Freeze the pretrained model weights. Useful for transfer learning.
    The model should be some Huggingface pretrained model.
    """
    for param in model.base_model.parameters():
        param.requires_grad = True


# From "Asymmetric Loss For Multi-Label Classification"(ICCV, 2021)
# See: https://github.com/Alibaba-MIIL/ASL, https://arxiv.org/abs/2009.14119
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.mean()


class AspectJitter(nn.Module):
    def __init__(self, output_size, resize_interval: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if resize_interval is None:
            resize_interval = (0, 10)
        self.rand_resize_interval = resize_interval
        self.resize = v2.Resize(output_size)

    def forward(self, x):
        rand_am = torch.randint(
            self.rand_resize_interval[0],
            self.rand_resize_interval[1],
            [4]
        )
        x = x[..., rand_am[0]:x.shape[-2]-rand_am[1], rand_am[2]:x.shape[-1]-rand_am[3]]
        x = self.resize(x)
        return x


"""
Running cell masking is taken from:
 - https://github.com/OpenGVLab/VideoMAEv2/
 - https://github.com/alibaba-mmai-research/Masked-Action-Recognition

Specifically, these papers:
 - https://arxiv.org/abs/2303.16727
 - https://arxiv.org/abs/2207.11660
"""

class Cell:

    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class RunningCellMaskingGenerator:

    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio

        num_masks_per_cell = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map.flatten())
            mask = np.stack(mask, axis=0)
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Running Cell Masking with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self):
        mask = self.all_mask_maps[np.random.randint(self.cell_size)]
        return np.copy(mask)


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".
            format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_vit(config):
    """Load a VideoMAEv2 checkpoint and return the model. Based on code
    from:
    https://github.com/OpenGVLab/VideoMAEv2/
    """
    model = create_model(
        config.model,
        img_size=config.image_size,
        pretrained=False,
        all_frames=config.num_frames,
        tubelet_size=config.tubelet_size,
        drop_rate=config.dropout,
        drop_path_rate=config.drop_path,
        attn_drop_rate=config.attn_drop_rate,
        head_drop_rate=config.head_drop_rate,
        drop_block_rate=None,
        with_cp=False,
        num_classes=88*2 # Onsets and frames
    )
    checkpoint = torch.hub.load_state_dict_from_url(
        config.pretrained_encoder, map_location='cpu', check_hash=True)

    print("Load ckpt from %s" % config.model)
    checkpoint_model = None
    for model_key in config.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    for old_key in list(checkpoint_model.keys()):
        if old_key.startswith('_orig_mod.'):
            new_key = old_key[10:]
            checkpoint_model[new_key] = checkpoint_model.pop(old_key)

    state_dict = model.state_dict()
    for k in ['head_1.weight', 'head_1.bias']:
        if k in checkpoint_model and checkpoint_model[
            k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    for k in ['head_2.weight', 'head_2.bias']:
        if k in checkpoint_model and checkpoint_model[
            k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    load_state_dict(
        model, checkpoint_model)

    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    return model
