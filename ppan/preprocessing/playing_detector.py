import multiprocessing
from pathlib import Path
from typing import Optional

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments
)

from ppan.config import seed, pretrained_playing_det_vit, det_res
from ppan.dataset import load_data, PlayingDataset
from ppan.types import PathLike


def main(dataset_dir: PathLike,
         output_dir: PathLike,
         checkpoint_dir: Optional[PathLike] = None,
         *_, **__):
    """
    Training function for the PPAn playing detector.

    Parameters
    ----------
    checkpoint_dir : Optional[PathLike]
        Checkpoint directory from which to resume training.
    tokenizer_dir : PathLike
    dataset_dir : PathLike
        Directory with train and test set in their own folders.
    output_dir : PathLike
        Where to output training results.
    decoder_checkpoint : PathLike
        Location of pretrained BERT decoder.

    Returns
    -------
    None
    """
    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir[0]

    train_playing_detector(dataset_dir=dataset_dir[0],
                           output_dir=output_dir[0],
                           checkpoint_dir=checkpoint_dir)


def train_playing_detector(dataset_dir: PathLike,
                           output_dir: PathLike,
                           checkpoint_dir: Optional[PathLike] = None):
    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    temporal_res = 3
    clips_per_vid = 6
    eval_every = 250
    save_every = 250
    max_steps = 50000

    """Optimizer Config"""
    lr = 1e-5
    weight_decay = 0.001

    """Scheduler Config"""
    warmup_ratio = 0.1
    scheduler_type = "cosine"

    """Model Config"""
    image_size = det_res

    model = ViTForImageClassification.from_pretrained(
        "/home/chromeilion/Code/Uni/uni2023S/thesis/coding/testing_data"
        "/checkpoint-31250/",
        image_size=image_size,
        ignore_mismatched_sizes=True,
        num_labels=2
    ).train()
    processor = ViTImageProcessor(size=image_size)

    dataset = Path(dataset_dir)
    train_dir = dataset.joinpath("train")
    test_dir = dataset.joinpath("test")

    train = load_data(train_dir)
    test = load_data(test_dir)

    train = PlayingDataset(
        samples=train,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        frame_transform=processor
    )
    test = PlayingDataset(
        samples=test,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        frame_transform=processor,
        epoch_size=1
    )
    multiprocessing.set_start_method("spawn")
    training_arguments = TrainingArguments(
        max_steps=max_steps,
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
        optim="adamw_torch",
        weight_decay=weight_decay,
        auto_find_batch_size=True,
        save_steps=save_every,
        dataloader_drop_last=True
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train,
        eval_dataset=test,
    )
    trainer.train(resume_from_checkpoint=checkpoint_dir)
