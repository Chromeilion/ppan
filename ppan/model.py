import math
from typing import TypedDict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, VideoMAEModel
from transformers.models.vivit.modeling_vivit import VivitEmbeddings
from ppan.config import (num_labels, PRETRAINED_MODEL_SMALL,
                         model_no_frames, model_resolution,
                         model_crop_resolution, IMAGENET_MEAN, IMAGENET_STD)
from ppan.utils import AsymmetricLossOptimized
from ppan.dataset import BaseVideoProcessor
import torchvision.transforms.v2 as v2


class PPANDecoderConfig(TypedDict):
    patch_size: int
    dim_embedding: int
    nhead: int
    dropout: float
    num_layers: int
    dim_feedforward: int


class PPANConfig(PretrainedConfig):
    model_type = "ppan"

    def __init__(
            self,
            dropout: float = 0.,
            pretrained_encoder: str = PRETRAINED_MODEL_SMALL,
            num_frames: int = model_no_frames,
            image_size: tuple[int, int] = model_resolution,
            do_smoothing: bool = False,
            confidence: float = 0.7,
            loss_gamma_neg: float = 2,
            loss_gamma_pos: float = 0,
            loss_clip = None,
            initializer_range: float = 0.02,
            **kwargs,
    ):
        self.dropout = dropout
        self.pretrained_encoder = pretrained_encoder
        self.image_size = image_size
        self.num_frames = num_frames
        self.do_smoothing = do_smoothing
        self.confidence = confidence
        self.loss_gamma_neg = loss_gamma_neg
        self.loss_gamma_pos = loss_gamma_pos
        self.loss_clip = loss_clip
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class PPANModel(PreTrainedModel):
    config_class = PPANConfig

    def __init__(self, config: PPANConfig):
        super().__init__(config)
        encoderargs = {
            "pretrained_model_name_or_path": config.pretrained_encoder,
            "image_size": config.image_size,
            "num_frames": config.num_frames,
            "hidden_dropout_prob": config.dropout,
            "attention_probs_dropout_prob": config.dropout,
        }
        self.pretrained_model = VideoMAEModel.from_pretrained(
            **encoderargs
        )
        self.num_video_tokens = self.pretrained_model.base_model.embeddings.num_patches

        embedding_dim = self.pretrained_model.config.hidden_size
        self.loss_fn = AsymmetricLossOptimized(
            gamma_neg=config.loss_gamma_neg,
            gamma_pos=config.loss_gamma_pos,
            clip=config.loss_clip
        )
        self.head = nn.Linear(embedding_dim, num_labels)
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Standard fine-tuning procedure, initialize new layers with zeros.
        """
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos=None, labels=None):
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

        x = self.pretrained_model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos
        ).last_hidden_state

        x = torch.mean(x, dim=1)
        logits = self.head(x)

        if labels is not None:
            loss = self.get_loss(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"logits": logits}

    def get_loss(self, logits, labels):
        if self.config.do_smoothing:
            labels_pos = labels > 0.5
            labels[labels_pos] = self.config.confidence
            labels[~labels_pos] = 1 - self.config.confidence

        return self.loss_fn(logits, labels)


class PPANVideoProcessor(BaseVideoProcessor):
    """
    Video processor for PPAN finetuning
    """
    def __init__(self, randaug: Optional[bool] = None,
                 spatial_jitter: Optional[bool] = None,
                 rand_erase: Optional[bool] = None,
                 rotate_180: Optional[bool] = None,
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

        augs = []
        if spatial_jitter:
            augs.append(v2.RandomCrop(size=model_crop_resolution,
                                      pad_if_needed=True))
            augs.append(v2.ScaleJitter(
                target_size=(model_crop_resolution[1], model_crop_resolution[0]),
                scale_range=(0.8, 1.01)
            ))
            augs.append(v2.RandomCrop(size=model_crop_resolution,
                                      pad_if_needed=True))
        else:
            augs.append(v2.CenterCrop(size=model_crop_resolution))

        if randaug:
            augs.append(v2.RandAugment(
                num_ops=2, magnitude=5
            ))
        if rand_erase:
            augs.append(v2.RandomErasing())
        if rotate_180:
            augs.append(v2.RandomApply([v2.RandomRotation([180, 180])], p=0.5))

        augs.append(v2.Resize(model_resolution))
        augs.append(v2.ToDtype(torch.float32, scale=True))
        augs.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.augmentations = v2.Compose(augs)

    def process_video(self, vid):
        return self.augmentations(vid)


class PPanEmbeddings(VivitEmbeddings):
    """
    Modified Vivit embeddings that enable positional encoding interpolation
    for frames (and not just H/W).
    """
    def interpolate_pos_encoding(self, embeddings, height, width, frames):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        frame_patches = self.config.num_frames/self.config.tubelet_size[0]
        num_positions = (self.position_embeddings.shape[1]-1)/frame_patches

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        f0 = frames // self.config.tubelet_size[0]
        h0 = height // self.config.tubelet_size[1]
        w0 = width // self.config.tubelet_size[2]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0, f0 = h0 + 0.1, w0 + 0.1, f0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, -1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(f0 / frame_patches, h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="trilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 1, 3, 4, 2).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width, num_frames)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings