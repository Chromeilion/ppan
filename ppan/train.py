import copy
import os
from typing import Optional, Union
from random import shuffle

from dotenv import load_dotenv
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments
)

from ppan.config import seed, pretrained_model, num_labels
from ppan.dataset import load_all_data, PPAnTrainDataset
from ppan.trainer import PPAnTrainer
from ppan.trainer_callbacks import (WandbPredictionProgressCallback)

PathLike = Union[str, bytes, os.PathLike]


def train(rach3_dir: PathLike,
          pianoyt_dir: PathLike,
          miditest_dir: PathLike,
          output_dir: PathLike,
          no_epochs: Optional[int] = None,
          eval_every: Optional[int] = None,
          save_every: Optional[int] = None,
          batch_size: Optional[int] = None,
          max_iters_per_epoch_test: Optional[int] = None,
          max_iters_per_epoch_train: Optional[int] = None,
          class_weights: Optional[float] = None,
          learning_rate: Optional[float] = None,
          weight_decay: Optional[float] = None,
          warmup_ratio: Optional[float] = None,
          scheduler_type: Optional[str] = None,
          checkpoint_dir: Optional[PathLike] = None,
          adam_beta1: Optional[float] = None,
          adam_beta2: Optional[float] = None,
          *_, **__):
    if rach3_dir is None:
        raise AttributeError("The dataset directory is required for "
                             "model training.")
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
        adam_beta2 = 0.999

    load_dotenv()
    no_gpu = int(os.environ.get("PPAN_NO_GPU", 1))
    # Use the linear scaling rule to calculate the lr, more info here:
    # https://arxiv.org/abs/1706.02677
    lr = (learning_rate * batch_size * no_gpu) / 256.

    test_samples, train_samples, _, _, _ = load_all_data(
        rach3_dir, pianoyt_dir, miditest_dir
    )
    # Remove a third of the dataset for faster training...
    train_samples = [
        v for i, v in enumerate(train_samples) if not i % 3 == 0
    ]
    shuffle(test_samples)
    shuffle(train_samples)
    processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model,
    )

    train_ds = PPAnTrainDataset(
        datasets=train_samples,
        video_transform=processor,
        epoch_size=1,
        cachefile_name="./train_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_train,
        batch_size=batch_size
    )
    test_ds = PPAnTrainDataset(
        datasets=test_samples,
        video_transform=processor,
        epoch_size=1,
        cachefile_name="./test_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_test,
        batch_size=batch_size
    )
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model,
        problem_type="regression",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).train()

    training_arguments = TrainingArguments(
        num_train_epochs=no_epochs,
        output_dir=str(output_dir),
        evaluation_strategy="steps",
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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
        report_to=["wandb"]
    )
    trainer = PPAnTrainer(
        weight=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=lambda x: x[0]
    )
    # Instantiate the WandbPredictionProgressCallback
    # A copy of the dataset is passed so that the state isn't messed up
    # for the Trainer.
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        val_dataset=copy.copy(test_ds)
    )
    # Add the callback to the trainer
    trainer.add_callback(progress_callback)

    trainer.train(resume_from_checkpoint=checkpoint_dir)
