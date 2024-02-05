import os
from typing import Optional, Union
import copy

from dotenv import load_dotenv
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments
)

from ppan.config import seed, pretrained_model, num_labels
from ppan.dataset import load_rach3, PPAnTrainDataset
from ppan.trainer_callbacks import (WandbPredictionProgressCallback,
                                    SaveCallback)
from ppan.trainer import PPAnTrainer

PathLike = Union[str, bytes, os.PathLike]


def train(dataset_dir: PathLike,
          output_dir: PathLike,
          no_epochs: Optional[int],
          eval_every: Optional[int],
          save_every: Optional[int],
          batch_size: Optional[int],
          max_iters_per_epoch_test: Optional[int],
          max_iters_per_epoch_train: Optional[int],
          class_weights: Optional[float],
          learning_rate: Optional[float],
          weight_decay: Optional[float],
          warmup_ratio: Optional[float],
          scheduler_type: Optional[str],
          checkpoint_dir: Optional[PathLike] = None,
          *_, **__):
    if dataset_dir is None:
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

    load_dotenv()
    no_gpu = int(os.environ.get("PPAN_NO_GPU", 1))
    lr = (learning_rate * batch_size * no_gpu) / 256.

    test_samples, train_samples = load_rach3(dataset_dir)

    processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model,
    )
    train_ds = PPAnTrainDataset(
        datasets=[train_samples],
        video_transform=processor,
        batch_size=batch_size,
        epoch_size=no_epochs,
        cachefile_name="./train_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_train,
        checkpoint_location=checkpoint_dir
    )
    test_ds = PPAnTrainDataset(
        datasets=[test_samples],
        video_transform=processor,
        epoch_size=1,
        batch_size=batch_size,
        cachefile_name="./test_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_test
    )
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model,
        problem_type="regression",
        num_labels=num_labels
    ).train()

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        num_train_epochs=1,
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
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
    save_callback = SaveCallback()
    # Add the callback to the trainer
    trainer.add_callback(progress_callback)
    trainer.add_callback(save_callback)
    trainer.train(resume_from_checkpoint=checkpoint_dir)
