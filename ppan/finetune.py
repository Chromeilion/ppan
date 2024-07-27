import random
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from dotenv import load_dotenv
from transformers import (
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer
)

from ppan.config import (seed, num_labels, model_resolution,
                         max_eval_steps, finetune_default)
from ppan.config import base as model_config
from ppan.dataset import PPANDataset, BaseVideoProcessor, get_samples
from ppan.trainer_callbacks import WandbFinetuePredictionProgressCallback

PathLike = Union[str, bytes, os.PathLike]


def train(dataset_dir: PathLike,
          pretrained_checkpoint: PathLike,
          output_dir: PathLike,
          no_epochs: Optional[int] = None,
          eval_every: Optional[int] = None,
          save_every: Optional[int] = None,
          batch_size: Optional[int] = None,
          class_weights: Optional[float] = None,
          learning_rate: Optional[float] = None,
          weight_decay: Optional[float] = None,
          warmup_ratio: Optional[float] = None,
          scheduler_type: Optional[str] = None,
          checkpoint_dir: Optional[PathLike] = None,
          adam_beta1: Optional[float] = None,
          adam_beta2: Optional[float] = None,
          *_, **__):
    if dataset_dir is None:
        raise AttributeError("The dataset directory is required for "
                             "model training.")
    if output_dir is None:
        raise AttributeError("The output directory is required for "
                             "model training.")
    if no_epochs is None:
        no_epochs = finetune_default["no_epochs"]
    if batch_size is None:
        batch_size = finetune_default["batch_size"]
    if learning_rate is None:
        learning_rate = finetune_default["learning_rate"]
    if weight_decay is None:
        weight_decay = finetune_default["weight_decay"]
    if warmup_ratio is None:
        warmup_ratio = finetune_default["warmup_ratio"]
    if scheduler_type is None:
        scheduler_type = finetune_default["scheduler_type"]
    if adam_beta1 is None:
        adam_beta1 = finetune_default["adam_beta1"]
    if adam_beta2 is None:
        adam_beta2 = finetune_default["adam_beta2"]
    if eval_every is None:
        eval_every = finetune_default["eval_every"]
    if save_every is None:
        save_every = finetune_default["save_every"]

    load_dotenv()
    dataset_dir = Path(dataset_dir)
    # How many frames from the center we want in each clip
    pad = int(model_config["num_frames"] // 2)
    processor = TrainProcessor(pad=pad)
    output_map = {"vid": "pixel_values", "lab": "label_ids"}
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
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_checkpoint,
        problem_type="regression",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).train()

    training_arguments = TrainingArguments(
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=["wandb"],
        # Video decoding is already  multithreaded, so using all cores for
        # dataset workers is detrimental to performance.
        dataloader_num_workers=os.cpu_count() // 2
    )
    trainer = FinetuneTrainer(
        weight=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )
    # Instantiate the WandbPredictionProgressCallback
    # A copy of the dataset is passed so that the state isn't messed up
    # for the Trainer.
    progress_callback = WandbFinetuePredictionProgressCallback(
        trainer=trainer,
        val_dataset=test_ds
    )
    # Add the callback to the trainer
    trainer.add_callback(progress_callback)

    trainer.train(resume_from_checkpoint=checkpoint_dir)


class TrainProcessor(BaseVideoProcessor):
    """
    Video processor for PPAN finetuning
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


class FinetuneTrainer(Trainer):
    """Huggingface trainer override adding BCE loss and dynamic class weights
    to the training loop.
    """
    def __init__(self, weight: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weight is not None:
            self.weight = torch.tensor([weight])
        else:
            self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute loss using BCE
        if self.weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.weight.to(logits.device))
        else:
            pos = torch.sum(labels)
            if pos < 1:
                weight = torch.tensor(88.).to(logits.device)
            else:
                weight = (torch.numel(labels)-pos)/pos
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=weight
            )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
