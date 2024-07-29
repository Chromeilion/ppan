import os
import random
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    VideoMAEConfig,
    VideoMAEForPreTraining,
    TrainingArguments,
    Trainer
)
from transformers.integrations import WandbCallback

from ppan.config import fps
from ppan.config import (seed, model_resolution, base,
                         pretrain_default, max_eval_steps)
from ppan.dataset import BaseVideoProcessor, PPANDataset, get_samples
from ppan.utils import TubeMaskingGenerator
from ppan.utils import patchify, unpatchify

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
    config = VideoMAEConfig(**base)
    model = VideoMAEForPreTraining(config).train()

    # How many frames from the center we want in each clip
    pad = int(base["num_frames"] // 2)

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
    progress_callback = WandbPretrainPredictionProgressCallback(
        trainer=trainer,
        val_dataset=test_ds
    )
    trainer.add_callback(progress_callback)
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
        self.mask_gen = TubeMaskingGenerator(
            (
                model.config.num_frames,
                model.config.image_size[0]//model.config.patch_size,
                model.config.image_size[1]//model.config.patch_size
            ),
            mask_ratio
        )

    def get_mask(self, batch_size):
        """Calculate mask locations
        """
        masks = []
        for i in range(batch_size):
            masks.append(torch.tensor(self.mask_gen()))

        return torch.stack(masks, dim=0)

    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Small override which generates and injects the patch masks into the
        input.
        """
        inputs["bool_masked_pos"] = self.get_mask(inputs["pixel_values"].shape[0])
        return super(PretrainTrainer, self).training_step(model, inputs)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[
               torch.Tensor]]:
        inputs["bool_masked_pos"] = self.get_mask(inputs["pixel_values"].shape[0])
        return super(PretrainTrainer, self).prediction_step(
            model, inputs, prediction_loss_only, ignore_keys)


class WandbPretrainPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.
    """

    def __init__(self, trainer, val_dataset,
                 num_samples=3, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

    Parameters:
        trainer : Trainer
            The Hugging Face Trainer instance.
        val_dataset : Dataset
            The validation dataset for generating predictions.
        num_samples : int, optional
            Number of samples to select from
            the validation dataset for generating predictions. Defaults to 3.
        freq : int, optional
            Frequency of logging. Defaults to 1.
        """
        super().__init__()
        self.trainer: Trainer = trainer
        self.sample_dataset = next(iter(DataLoader(
            val_dataset,
            batch_size=num_samples,
            shuffle=True
        )))
        self.sample_dataset["bool_masked_pos"] = trainer.get_mask(
            self.sample_dataset["pixel_values"].shape[0])
        self.freq = freq
        self.mask = trainer.get_mask(num_samples)

    def add_preds_vid(self, logits: torch.Tensor, model,
                      step: int):
        vid = self.sample_dataset["pixel_values"]
        p = model.config.patch_size
        tu = model.config.tubelet_size
        patches = patchify(vid, p, tu)
        pos_masks = 0
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if self.mask[i, j]:
                    patches[i, j, :] = logits[i, pos_masks, :]
                    pos_masks += 1
            pos_masks = 0

        reconstructed = unpatchify(patches, p, tu, vid.shape[1],
                                   vid.shape[3], vid.shape[4])

        reconstructed = v2.functional.to_dtype(
            v2.functional.grayscale_to_rgb(reconstructed),
            torch.uint8, scale=True
        )
        reconstructed = reconstructed.cpu().numpy()

        self._wandb.log(
            {"Model predictions": wandb.Video(
                reconstructed,
                "Reconstructed masked images",
                fps//2
            )
            },
            step=step
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % state.eval_steps * self.freq == 0:
            model = kwargs["model"]
            with torch.no_grad():
                sample = {
                    k: v.to(device=model.device) for k, v in
                    self.sample_dataset.items()
                }
                preds = model(
                    **sample,
                    return_dict=True
                )
            logits = preds["logits"].cpu()
            self.add_preds_vid(
                logits,
                model=model,
                step=state.global_step
            )
            super().on_evaluate(args, state, control, **kwargs)

