from torch import multiprocessing
from typing import Optional

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    Trainer,
    TrainingArguments
)

from ppan.config import seed, pretrained_model
from ppan.dataset import load_rach3, ImageVecDataset, load_ytmidi
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
    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    max_steps = 50000
    temporal_res = 2
    clips_per_vid = 8
    eval_every = 500
    save_every = 500

    """Optimizer Config"""
    lr = 1e-5
    weight_decay = 1e-4

    """Scheduler Config"""
    warmup_ratio = 0.1
    scheduler_type = "cosine"

    test, train = load_rach3(dataset_dir[0])
    train_yt, test_yt = load_ytmidi(dataset_dir[1])
    train.extend(train_yt)
    test.extend(test_yt)

    processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model,
        do_center_crop=False,
        size={"height": 224, "width": 224}
    )

    train = ImageVecDataset(
        samples=train,
        clips_per_vid=clips_per_vid,
        video_transform=processor
    )
    test = ImageVecDataset(
        samples=test,
        clips_per_vid=2,
        video_transform=processor,
        epoch_size=1
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model, problem_type="regression",
        num_labels=128*2,
        ignore_mismatched_sizes=True,
        num_frames=7
    ).train()

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        max_steps=max_steps,
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
        optim="adamw_torch",
        weight_decay=weight_decay,
        save_steps=save_every,
        per_device_train_batch_size=8
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train,
        eval_dataset=test,
    )
    multiprocessing.set_start_method("spawn", force=True)
    trainer.train(resume_from_checkpoint=checkpoint_dir)
