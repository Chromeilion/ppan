import os
from typing import Optional, Union, Callable
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from dotenv import load_dotenv
from transformers import (
    VideoMAEConfig,
    VideoMAEForPreTraining,
    TrainingArguments,
    Trainer
)
from ppan.config import seed, model_resolution, model_config
from ppan.dataset import PretrainDataset

PathLike = Union[str, bytes, os.PathLike]


def pretrain(dataset_dir: PathLike,
             output_dir: PathLike,
             no_epochs: Optional[int] = None,
             eval_every: Optional[int] = None,
             save_every: Optional[int] = None,
             batch_size: Optional[int] = None,
             max_iters_per_epoch_test: Optional[int] = None,
             max_iters_per_epoch_train: Optional[int] = None,
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
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        raise AttributeError("The output directory is required for "
                             "model training.")
    if no_epochs is None:
        no_epochs = 1
    if batch_size is None:
        batch_size = 8
    if learning_rate is None:
        learning_rate = 1e-3
    if weight_decay is None:
        weight_decay = 0.05
    if warmup_ratio is None:
        warmup_ratio = 0.1
    if scheduler_type is None:
        scheduler_type = "cosine"
    if adam_beta1 is None:
        adam_beta1 = 0.9
    if adam_beta2 is None:
        adam_beta2 = 0.95
    load_dotenv()
    lr = learning_rate
    config = VideoMAEConfig(**model_config)
    model = VideoMAEForPreTraining(config).train()

    # How many frames from the center we want in each clip
    pad = int(model_config["num_frames"] // 2)

    processor = get_pretrain_video_processor()
    train_ds = PretrainDataset(
        video_processor=processor,
        root=dataset_dir/"train",
        seq_length=model.videomae.embeddings.num_patches,
        pad=pad
    )
    test_ds = PretrainDataset(
        video_processor=processor,
        root=dataset_dir/"test",
        seq_length=model.videomae.embeddings.num_patches,
        pad=pad
    )
    training_arguments = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=no_epochs,
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        learning_rate=lr,
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
        dataloader_num_workers=os.cpu_count()//2,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )
    trainer.train()


def get_pretrain_video_processor() -> Callable[[torch.IntTensor],
                                               torch.FloatTensor]:
    return torch.nn.Sequential(
        v2.RandomCrop(size=model_resolution),
        v2.Resize(model_resolution),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True)
    )
