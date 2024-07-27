import os
from pathlib import Path
from typing import Optional, Union
import random

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from dotenv import load_dotenv
from transformers import (
    VideoMAEConfig,
    VideoMAEForPreTraining,
    TrainingArguments,
    Trainer
)

from ppan.config import (seed, model_resolution, model_config,
                         pretrain_default, max_eval_steps)
from ppan.dataset import BaseVideoProcessor, PPANDataset, get_samples

PathLike = Union[str, bytes, os.PathLike]


def pretrain(dataset_dir: PathLike,
             output_dir: PathLike,
             no_epochs: Optional[int] = None,
             eval_every: Optional[int] = None,
             save_every: Optional[int] = None,
             batch_size: Optional[int] = None,
             learning_rate: Optional[float] = None,
             weight_decay: Optional[float] = None,
             warmup_ratio: Optional[float] = None,
             scheduler_type: Optional[str] = None,
             checkpoint_dir: Optional[PathLike] = None,
             adam_beta1: Optional[float] = None,
             adam_beta2: Optional[float] = None,
             mask_ratio: Optional[float] = None,
             *_, **__) -> None:
    if dataset_dir is None:
        raise AttributeError("A dataset directory is required for "
                             "model pre-training.")
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        raise AttributeError("An output directory is required for "
                             "model pre-training.")
    if no_epochs is None:
        no_epochs = pretrain_default["no_epochs"]
    if batch_size is None:
        batch_size = pretrain_default["batch_size"]
    if learning_rate is None:
        learning_rate = pretrain_default["learning_rate"]
    if weight_decay is None:
        weight_decay = pretrain_default["weight_decay"]
    if warmup_ratio is None:
        warmup_ratio = pretrain_default["warmup_ratio"]
    if scheduler_type is None:
        scheduler_type = pretrain_default["scheduler_type"]
    if adam_beta1 is None:
        adam_beta1 = pretrain_default["adam_beta1"]
    if adam_beta2 is None:
        adam_beta2 = pretrain_default["adam_beta2"]
    if mask_ratio is None:
        mask_ratio = pretrain_default["mask_ratio"]
    if eval_every is None:
        eval_every = pretrain_default["eval_every"]
    if save_every is None:
        save_every = pretrain_default["save_every"]
    load_dotenv()
    config = VideoMAEConfig(**model_config)
    model = VideoMAEForPreTraining(config).train()

    # How many frames from the center we want in each clip
    pad = int(model_config["num_frames"] // 2)

    processor = PretrainProcessor(pad=pad)
    output_map = {"vid": "pixel_values", "lab": None}
    train_ds = PPANDataset(
        video_processor=processor,
        samples=get_samples(dataset_dir/"train"),
        output_map=output_map
    )
    test_ds = PPANDataset(
        video_processor=processor,
        samples=random.sample(get_samples(dataset_dir/"test"), max_eval_steps),
        output_map=output_map
    )
    training_arguments = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=no_epochs,
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        optim="adamw_torch",
        save_steps=save_every,
        dataloader_pin_memory=False,
        report_to=["wandb"],
        # Video decoding is already  multithreaded, so using all cores for
        # dataset workers is detrimental to performance.
        dataloader_num_workers=os.cpu_count()//2
    )
    trainer = PretrainTrainer(
        mask_ratio=mask_ratio,
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train(
        resume_from_checkpoint=checkpoint_dir
    )


class PretrainProcessor(BaseVideoProcessor):
    """
    Dataset for masked pretraining.
    """
    def __init__(self, pad: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad: Optional[int] = pad
        self.augmentations = torch.nn.Sequential(
            v2.RandomCrop(size=model_resolution),
            v2.Resize(model_resolution),
            v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True)
        )

    def process_video(self, vid):
        vid = self._center_cut(vid)
        vid = self.augmentations(vid)
        return vid


class PretrainTrainer(Trainer):
    """Huggingface trainer override adding masking to the training loop.
    """
    def __init__(self, mask_ratio: float, model, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.mask_ratio = mask_ratio

        self.seq_length = model.videomae.embeddings.num_patches
        self.no_masks = round(self.seq_length * self.mask_ratio)
        self.seq = range(self.seq_length)

    def _get_mask(self, batch_size):
        """Calculate mask locations
        """
        masks = []
        for i in range(batch_size):
            bool_masked_pos = torch.zeros(
                self.seq_length,
                dtype=torch.bool
            )
            bool_masked_pos[random.sample(self.seq, self.no_masks)] = 1
            masks.append(bool_masked_pos)

        return torch.stack(masks, dim=0)

    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Small override which generates and injects the patch masks into the
        input.
        """
        inputs["bool_masked_pos"] = self._get_mask(inputs["pixel_values"].shape[0])
        return super(PretrainTrainer, self).training_step(model, inputs)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[
               torch.Tensor]]:
        inputs["bool_masked_pos"] = self._get_mask(inputs["pixel_values"].shape[0])
        return super(PretrainTrainer, self).prediction_step(
            model, inputs, prediction_loss_only, ignore_keys)
