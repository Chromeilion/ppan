import os
from typing import Optional

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments
)

from ppan.config import seed, pretrained_model, num_labels
from ppan.dataset import load_rach3, PPAnTrainDataset
from ppan.stats import WandbPredictionProgressCallback
from ppan.trainer import PPAnTrainer
from ppan.types import PathLike


def main(dataset_dir: list[PathLike],
         output_dir: PathLike,
         checkpoint_dir: Optional[PathLike],
         *_, **__):
    """
    Training function for PPAn.

    Parameters
    ----------
    checkpoint_dir : Optional[PathLike]
        Checkpoint directory from which to resume training.
    dataset_dir : PathLike
        Directory with train and test set in their own folders.
    output_dir : PathLike
        Where to output training results.

    Returns
    -------
    None
    """
    if checkpoint_dir:
        checkpoint_dir = checkpoint_dir[0]
    train_image(dataset_dir=dataset_dir,
                output_dir=output_dir[0],
                checkpoint_dir=checkpoint_dir)


def train_image(dataset_dir: list[PathLike], output_dir: PathLike,
                checkpoint_dir: Optional[PathLike] = None):
    # Wandb logging settings
    os.environ["WANDB_PROJECT"] = "rach3-detector"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    no_epochs = 1
    eval_every = 200
    save_every = 500
    batch_size = 2
    max_iters_per_epoch_test = 50
    max_iters_per_epoch_train = None #4000
    class_weights = None

    """Optimizer Config"""
    lr = 1e-3 * batch_size / 256
    weight_decay = 0.05

    """Scheduler Config"""
    warmup_ratio = 0.1
    scheduler_type = "cosine"

    test, train = load_rach3(dataset_dir[0])

    processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model,
    )
    train = PPAnTrainDataset(
        datasets=[train],
        video_transform=processor,
        batch_size=batch_size,
        epoch_size=no_epochs,
        cachefile_name="./train_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_train
    )
    test = PPAnTrainDataset(
        datasets=[test],
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
        train_dataset=train,
        eval_dataset=test,
        data_collator=lambda x: x[0]
    )
    # Instantiate the WandbPredictionProgressCallback
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        val_dataset=test
    )
    # Add the callback to the trainer
    trainer.add_callback(progress_callback)
    trainer.train(resume_from_checkpoint=checkpoint_dir)
