"""
Misc. utility functions used across the package.
"""
import os
from typing import Optional, Literal
from collections import OrderedDict
from dataclasses import dataclass

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from timm import create_model

import ppan.model_mae
from ppan.config import PathLike, PRETRAINED_MODEL_SMALL, PRETRAINED_MODEL_BASE


ds_prefixes = Literal["r3x", "r3s", "pianoyt", "miditest", "omaps"]

@dataclass
class Sample:
    video_path: PathLike
    dataset: ds_prefixes
    midi_path: PathLike | None = None
    bounding_box: Optional[tuple[int, int, int, int]] = None
    flac_path: Optional[PathLike] = None
    rotation_factor: float = 0
    rotate_180: bool = False
    note_intervals: PathLike = None


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

def get_backbone(config):
    """Load a VideoMAEv2 checkpoint and return the model. Based on code
    from:
    https://github.com/OpenGVLab/VideoMAEv2/
    """
    num_classes = 88*2
    if config.pretrained_encoder == "vit_s":
        pretrained_encoder = PRETRAINED_MODEL_SMALL
        config.model = "vit_small_patch16_224"
    elif config.pretrained_encoder == "vit_b":
        pretrained_encoder = PRETRAINED_MODEL_BASE
        config.model = "vit_base_patch16_224"
    else:
        pretrained_encoder = PRETRAINED_MODEL_SMALL
        config.model = "vit_small_patch16_224"

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
        num_classes=num_classes # Onsets and frames
    )
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained_encoder, map_location='cpu', check_hash=True)
    except ValueError:
        checkpoint = torch.load(pretrained_encoder, map_location='cpu')

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
