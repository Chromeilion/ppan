from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, DefaultDataCollator
from ppan.config import (PRETRAINED_MODEL_SMALL, model_no_frames,
                         model_resolution, model_crop_resolution)
from ppan.utils import AsymmetricLossOptimized, get_vit
from ppan.dataset import BaseVideoProcessor
from torchvision.tv_tensors import Video
import torchvision.transforms.v2 as v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PPANConfig(PretrainedConfig):
    model_type = "ppan"

    def __init__(
            self,
            dropout: float = 0.,
            pretrained_encoder: str = PRETRAINED_MODEL_SMALL,
            num_frames: int = model_no_frames,
            image_size: tuple[int, int] = model_resolution,
            do_smoothing: bool = False,
            confidence: float = 0.8,
            loss_gamma_neg_onset: float = 2,
            loss_gamma_pos_onset: float = 0.,
            loss_gamma_neg_frame: float = 0.5,
            loss_gamma_pos_frame: float = 0.,
            loss_clip = None,
            initializer_range: float = 0.02,
            weight_init: str = "normal",
            model: str = "vit_small_patch16_224",
            tubelet_size: int = 2,
            stochastic_depth: float = 0.,
            attn_drop_rate: float = 0.,
            head_drop_rate: float = 0.,
            model_key: str = "model|module|state_dict",
            loss_fn: Literal["asymmetric", "bce"] = "asymmetric",
            bce_loss_weight_onset: float = 4,
            bce_loss_weight_frame: float = 2,
            do_mixup: bool = False,
            mixup_alpha: float = 0.,
            **kwargs,
    ):
        self.dropout = dropout
        self.drop_path = stochastic_depth
        self.model = model
        self.tubelet_size = tubelet_size
        self.pretrained_encoder = pretrained_encoder
        self.image_size = image_size
        self.num_frames = num_frames
        self.do_smoothing = do_smoothing
        self.confidence = confidence
        self.loss_gamma_neg_frame = loss_gamma_neg_frame
        self.loss_gamma_neg_onset = loss_gamma_neg_onset
        self.loss_gamma_pos_frame = loss_gamma_pos_frame
        self.loss_gamma_pos_onset = loss_gamma_pos_onset
        self.loss_clip_frame = loss_clip
        self.loss_clip_onset = loss_clip
        self.initializer_range = initializer_range
        self.weight_init = weight_init
        self.attn_drop_rate = attn_drop_rate
        self.head_drop_rate = head_drop_rate
        self.model_key = model_key
        self.loss_fn = loss_fn
        self.bce_weight_onset = bce_loss_weight_onset
        self.bce_weight_frame = bce_loss_weight_frame
        self.do_mixup = do_mixup
        self.mixup_alpha = mixup_alpha
        super().__init__(**kwargs)


class PPANModel(PreTrainedModel):
    config_class = PPANConfig

    def __init__(self, config: PPANConfig):
        super().__init__(config)
        self.pretrained_model = get_vit(config)

        if config.loss_fn == "asymmetric":
            self.frame_loss_fn = AsymmetricLossOptimized(
                gamma_neg=config.loss_gamma_neg_frame,
                gamma_pos=config.loss_gamma_pos_frame,
                clip=config.loss_clip_frame
            )
            self.onset_loss_fn = AsymmetricLossOptimized(
                gamma_neg=config.loss_gamma_neg_onset,
                gamma_pos=config.loss_gamma_pos_onset,
                clip=config.loss_clip_onset
            )
        elif config.loss_fn == "bce":
            self.frame_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(config.bce_weight_frame)
            )
            self.onset_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(config.bce_weight_onset)
            )

    def forward(self, pixel_values: torch.Tensor, onsets=None, frames=None):
        # If no batch dimension was supplied, we add it.
        if len(pixel_values.shape) == 4:
            pixel_values = torch.unsqueeze(pixel_values, dim=0)
        if onsets is None and frames is None:
            raise AttributeError("Both onsets and frames cannot be None!!!")
        if len(pixel_values.shape) != 5:
            raise AttributeError("The supplied video clip has an unexpected "
                                 "shape!!! It should be BTCHW.")
        if pixel_values.shape[1] != self.config.num_frames:
            raise AttributeError("An incorrect number of frames was given to the "
                                 "model!!!")
        if pixel_values.shape[3] != self.config.image_size[0] or pixel_values.shape[4] != self.config.image_size[1]:
            raise AttributeError("Incorrect image size given to the model!!!")

        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        logits_onset, logits_frame = self.pretrained_model(pixel_values)

        if onsets is not None or frames is not None:
            loss = self.get_loss(logits_onset, logits_frame, onsets, frames)
            return {"logits": torch.cat([logits_onset[:, None, :], logits_frame[:, None, :]], dim=1), "loss": loss}

        return {"logits": torch.cat([logits_onset[:, None, :], logits_frame[:, None, :]], dim=1)}

    def get_loss(self, logit_onset, logit_frame, onsets, frames):
        loss = 0
        if onsets is not None:
            loss += self.get_single_loss(logit_onset, onsets, self.onset_loss_fn)
        if frames is not None:
            loss += self.get_single_loss(logit_frame, frames, self.frame_loss_fn)

        return loss


    def get_single_loss(self, logits, labels, loss_fn):
        if self.config.do_smoothing:
            labels_pos = labels > 0.5
            labels[labels_pos] = self.config.confidence
            labels[~labels_pos] = 1 - self.config.confidence

        return loss_fn(logits, labels)


class PPANVideoProcessor(BaseVideoProcessor):
    """
    Video processor for PPAN finetuning
    """
    def __init__(self, randaug: Optional[bool] = None,
                 spatial_jitter: Optional[bool] = None,
                 rand_erase: Optional[bool] = None,
                 rotate_180: Optional[bool] = None,
                 gaussian_noise: Optional[bool] = None,
                 color_jitter: Optional[bool] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if spatial_jitter is None:
            spatial_jitter = False
        if randaug is None:
            randaug = False
        if rand_erase is None:
            rand_erase = False
        if rotate_180 is None:
            rotate_180 = False
        if gaussian_noise is None:
            gaussian_noise = False
        if color_jitter is None:
            color_jitter = False

        augs = []
        augs.append(v2.Resize(model_crop_resolution))
        if randaug:
            augs.append(v2.RandAugment(
                num_ops=2, magnitude=5
            ))
        if spatial_jitter:
            augs.append(v2.ScaleJitter(
                target_size=(model_crop_resolution[1], model_crop_resolution[0]),
                scale_range=(0.9, 1.005)
            ))
            augs.append(v2.RandomCrop(size=model_crop_resolution,
                                      pad_if_needed=True))
        if color_jitter:
            augs.append(v2.RandomApply([v2.ColorJitter(brightness=0.1)],
                                       p=0.4))
        if rand_erase:
            augs.append(v2.RandomErasing())
        if rotate_180:
            augs.append(v2.RandomApply([v2.RandomRotation([180, 180])], p=0.5))

        augs.append(v2.Resize(model_resolution))
        augs.append(v2.ToDtype(torch.float32, scale=True))
        if gaussian_noise:
            augs.append(v2.RandomApply([v2.GaussianNoise()], p=0.5))
        augs.append(v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        self.augmentations = v2.Compose(augs)

    def process_video(self, vid):
        return self.augmentations(vid)


class PPANCollate:
    def __init__(self, config: PPANConfig):
        self.default_collate = DefaultDataCollator()
        self.config = config
        self.mixup = v2.MixUp(alpha=config.mixup_alpha)

    def __call__(self, *args, **kwargs):
        batch = self.default_collate(*args, **kwargs)
        if self.config.do_mixup:
            vid = Video(batch["pixel_values"])
            mixed = self.mixup(vid, batch["onsets"], batch["frames"])
            batch = {"pixel_values": mixed[0], "onsets": mixed[1], "frames": mixed[2]}

        return batch
