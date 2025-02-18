from typing import Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, DefaultDataCollator
from ppan.config import (PRETRAINED_MODEL_SMALL, model_no_frames,
                         model_resolution, model_crop_resolution)
from ppan.utils import get_backbone
from torchvision.tv_tensors import Video
import torchvision.transforms.v2 as v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Weights come from the negative / positive note ratios calculated on the
# r3x training set.
ONSET_WEIGHTS = torch.tensor([
    2.8450e+05, 8.4695e+03, 2.9427e+04, 8.6410e+03, 1.5239e+04, 2.5633e+03,
    1.9937e+03, 8.1656e+03, 1.4192e+03, 2.1675e+03, 8.6585e+02, 1.3874e+03,
    8.5560e+02, 9.1573e+02, 6.7962e+02, 2.0066e+02, 1.3420e+03, 1.8974e+02,
    5.2639e+02, 1.5399e+02, 1.1877e+02, 6.8031e+02, 8.3903e+01, 5.4673e+02,
    8.9576e+01, 4.4382e+02, 9.1870e+01, 5.8271e+01, 5.4419e+02, 5.7001e+01,
    2.2339e+02, 5.5163e+01, 5.2772e+01, 2.3671e+02, 3.8674e+01, 2.2286e+02,
    4.8456e+01, 2.4035e+02, 4.2150e+01, 3.6648e+01, 2.3440e+02, 3.9965e+01,
    1.9482e+02, 4.4059e+01, 5.6116e+01, 2.3390e+02, 4.8188e+01, 1.9646e+02,
    6.0978e+01, 1.6535e+02, 5.6925e+01, 4.3591e+01, 2.1902e+02, 4.8745e+01,
    1.5013e+02, 5.3318e+01, 6.9512e+01, 1.7958e+02, 6.0178e+01, 2.2336e+02,
    9.7330e+01, 3.0299e+02, 1.5871e+02, 1.4988e+02, 5.7312e+02, 1.8146e+02,
    4.8801e+02, 3.0818e+02, 4.1447e+02, 1.0067e+03, 7.4161e+02, 1.0139e+03,
    7.0958e+02, 1.2006e+03, 9.4892e+02, 9.9904e+02, 2.1716e+03, 1.7906e+03,
    3.1019e+03, 2.8817e+03, 5.4002e+03, 1.3819e+04, 2.7530e+04, 2.3706e+04,
    torch.inf, 7.7586e+04, 4.2673e+05, 6.5649e+04])

FRAME_WEIGHTS = torch.tensor([
    1.4224e+05, 1.1331e+03, 2.9522e+03, 1.4044e+03, 2.5541e+03, 7.3524e+02,
    6.4847e+02, 1.5346e+03, 2.8444e+02, 5.0803e+02, 2.0650e+02, 2.6420e+02,
    2.4924e+02, 1.7950e+02, 1.5809e+02, 4.1458e+01, 3.3070e+02, 4.9092e+01,
    1.3130e+02, 3.9834e+01, 3.0982e+01, 1.7526e+02, 2.2639e+01, 1.2599e+02,
    2.6813e+01, 7.8612e+01, 2.5492e+01, 1.3984e+01, 1.4580e+02, 1.6242e+01,
    5.0817e+01, 1.4438e+01, 1.3036e+01, 5.9349e+01, 9.0316e+00, 5.6589e+01,
    1.2696e+01, 4.2857e+01, 1.1550e+01, 9.1663e+00, 7.0539e+01, 1.0270e+01,
    4.1591e+01, 1.1279e+01, 1.4184e+01, 6.0006e+01, 1.0749e+01, 4.8579e+01,
    1.6003e+01, 3.8792e+01, 1.4415e+01, 9.2401e+00, 5.7225e+01, 1.1633e+01,
    3.5540e+01, 1.3557e+01, 1.8232e+01, 5.4043e+01, 1.4764e+01, 5.6118e+01,
    2.5404e+01, 8.2223e+01, 4.7504e+01, 4.2714e+01, 1.7300e+02, 4.3556e+01,
    1.1849e+02, 9.2329e+01, 2.2712e+01, 2.8103e+02, 1.6818e+02, 2.2741e+02,
    1.6705e+02, 3.0846e+02, 2.4379e+02, 2.2577e+02, 6.8270e+02, 3.7281e+02,
    7.8457e+02, 6.3132e+02, 7.3199e+02, 6.7588e+03, 7.5183e+03, 7.6531e+03,
    torch.inf, 1.1852e+04, 3.4138e+05, 7.6532e+03])

# Because some weights are extremely high, training becomes unstable.
# Therefore, I rescale everything here to be more reasonable.
FRAME_WEIGHTS = ((FRAME_WEIGHTS / FRAME_WEIGHTS.min()).clamp(1, 3) + 1).tolist()
ONSET_WEIGHTS = ((ONSET_WEIGHTS / ONSET_WEIGHTS.min()).clamp(1, 3) + 2).tolist()


class BaseVideoProcessor(nn.Module, ABC):
    """
    Base video processor abstract class. Can be used to customize the video
    processing behaviour of the PPANDataset class. Simply override the
    process_video method.
    """
    @abstractmethod
    @torch.compile(mode="reduce-overhead")
    def process_video(self, img: torch.tensor) -> torch.tensor:
        ...

    def forward(self, x) -> torch.tensor:
        return self.process_video(x)


class PPANConfig(PretrainedConfig):
    model_type = "ppan"

    def __init__(
            self,
            dropout: float = 0.,
            pretrained_encoder: str = PRETRAINED_MODEL_SMALL,
            num_frames: int = model_no_frames,
            image_size: tuple[int, int] = model_resolution,
            confidence_onset: float = 0.8,
            confidence_frame: float = 0.8,
            loss_clip = None,
            initializer_range: float = 0.02,
            weight_init: str = "normal",
            model: str = "vit_s",
            tubelet_size: int = 2,
            stochastic_depth: float = 0.,
            attn_drop_rate: float = 0.,
            head_drop_rate: float = 0.,
            model_key: str = "model|module|state_dict",
            bce_loss_weight_onset: list[float] = None,
            bce_loss_weight_frame: list[float] = None,
            do_mixup: bool = False,
            mixup_alpha: float = 0.,
            img_mean: float = IMAGENET_DEFAULT_MEAN,
            img_std: float = IMAGENET_DEFAULT_STD,
            **kwargs,
    ):
        if bce_loss_weight_onset is None:
            bce_loss_weight_onset = ONSET_WEIGHTS
        if bce_loss_weight_frame is None:
            bce_loss_weight_frame = FRAME_WEIGHTS

        self.do_smoothing_frame, self.do_smoothing_onset = False, False
        if confidence_frame is not None:
            self.do_smoothing_frame = True
        if confidence_onset is not None:
            self.do_smoothing_onset = True
        if image_size is None:
            image_size = model_resolution
        if num_frames is None:
            num_frames = model_no_frames

        self.dropout = dropout
        self.drop_path = stochastic_depth
        self.model = model
        self.tubelet_size = tubelet_size
        self.pretrained_encoder = pretrained_encoder
        self.image_size = image_size
        self.num_frames = num_frames
        self.confidence_onset = confidence_onset
        self.confidence_frame = confidence_frame
        self.loss_clip_frame = loss_clip
        self.loss_clip_onset = loss_clip
        self.initializer_range = initializer_range
        self.weight_init = weight_init
        self.attn_drop_rate = attn_drop_rate
        self.head_drop_rate = head_drop_rate
        self.model_key = model_key
        self.bce_weight_onset = bce_loss_weight_onset
        self.bce_weight_frame = bce_loss_weight_frame
        self.do_mixup = do_mixup
        self.mixup_alpha = mixup_alpha
        self.img_mean = img_mean
        self.img_std = img_std
        super().__init__(**kwargs)


class PPANModel(PreTrainedModel):
    config_class = PPANConfig

    def __init__(self, config: PPANConfig):
        super().__init__(config)
        self.pretrained_model = get_backbone(config)

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
        if len(pixel_values.shape) != 5:
            raise AttributeError("The supplied video clip has an unexpected "
                                 "shape!!! It should be BTCHW.")
        if pixel_values.shape[1] != self.config.num_frames:
            raise AttributeError("An incorrect number of frames was given to the "
                                 "model!!!")
        if pixel_values.shape[3] != self.config.image_size[0] or pixel_values.shape[4] != self.config.image_size[1]:
            raise AttributeError("Incorrect image size given to the model!!!")

        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        logits = self.pretrained_model(pixel_values)
        logits_onset = logits[:, :88]
        logits_frame = logits[:, 88:]

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
        return loss_fn(logits, labels)


class PPANVideoProcessor(BaseVideoProcessor):
    """
    Video processor for PPAN finetuning
    """
    def __init__(self, rand_rotate: Optional[bool] = None,
                 spatial_jitter: Optional[bool] = None,
                 rand_erase: Optional[bool] = None,
                 gaussian_noise: Optional[bool] = None,
                 color_jitter: Optional[bool] = None,
                 grayscale: Optional[bool] = None,
                 resolution: tuple[int, int] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if resolution is None:
            resolution = model_resolution
        if spatial_jitter is None:
            spatial_jitter = False
        if rand_erase is None:
            rand_erase = False
        if gaussian_noise is None:
            gaussian_noise = False
        if color_jitter is None:
            color_jitter = False
        if grayscale is None:
            grayscale = False
        if rand_rotate is None:
            rand_rotate = False

        augs: list = list()
        augs.append(v2.Resize(None, max_size=model_crop_resolution[1]))
        augs.append(v2.RandomCrop(model_crop_resolution, pad_if_needed=True))
        if spatial_jitter:
            augs.append(v2.ScaleJitter(
                target_size=(model_crop_resolution[1], model_crop_resolution[0]),
                scale_range=(0.96, 1.001),
                interpolation=v2.InterpolationMode.NEAREST
            ))
            augs.append(v2.RandomCrop(size=model_crop_resolution,
                                      pad_if_needed=True))
        if color_jitter:
            augs.append(v2.RandomApply([v2.ColorJitter(brightness=0.1)],
                                       p=0.4))
        if rand_rotate:
            augs.append(v2.RandomApply([v2.RandomRotation(0.2)], p=0.4))
        if grayscale:
            augs.append(v2.Grayscale(num_output_channels=3))
        if rand_erase:
            augs.append(v2.RandomErasing())

        augs.append(v2.Resize(resolution, interpolation=v2.InterpolationMode.NEAREST))
        augs.append(v2.ToDtype(torch.float32, scale=True))
        if gaussian_noise:
            augs.append(v2.RandomApply([v2.GaussianNoise()], p=0.4))
        augs.append(v2.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                 std=IMAGENET_DEFAULT_STD))
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
        if "frame" in batch and self.config.do_smoothing_frame:
            frame_p = batch["frames"] > 0.6
            frame_n = batch["frames"] < 0.4
            batch["frames"][frame_p] = self.config.confidence_frame
            batch["frames"][frame_n] = 1 - self.config.confidence_frame

        if "onsets" in batch and self.config.do_smoothing_onset:
            onset_p = batch["onsets"] > 0.6
            onset_n = batch["onsets"] < 0.4
            batch["onsets"][onset_p] = self.config.confidence_onset
            batch["onsets"][onset_n] = 1 - self.config.confidence_onset

        if self.config.do_mixup:
            vid = Video(batch["pixel_values"])
            mixed = self.mixup(vid, batch["onsets"], batch["frames"])
            batch = {"pixel_values": mixed[0], "onsets": mixed[1], "frames": mixed[2]}

        return batch
