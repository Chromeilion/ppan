from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from pathlib import Path
from typing import Optional

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
    get_scheduler
)

from ppan.config import seed, pretrained_playing_det_vit, det_res, device
from ppan.dataset import load_data, PlayingDataset, load_ytmidi
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
    temporal_res = 2
    clips_per_vid = 5
    eval_every = 200
    save_every = 200
    max_steps = 10000

    """Optimizer Config"""
    lr = 0.001
    weight_decay = 0.0001

    """Scheduler Config"""
    warmup_ratio = 0.1
    scheduler_type = "cosine"

    """Model Config"""
    image_size = det_res

    model = ViTForImageClassification.from_pretrained(
        "/home/chromeilion/Code/Uni/uni2023S/thesis/coding/testing_data"
        "/detector_v2/checkpoint-7800/",
        image_size=image_size,
        num_labels=1,
        ignore_mismatched_sizes=True,
        problem_type="regression"
    ).train().to(device)
    processor = ViTImageProcessor(do_resize=True,
                                  size=image_size)
    optimizer = SGD(params=model.parameters(), lr=lr, momentum=0.9)
    lr_sched = CosineAnnealingLR(optimizer, max_steps)
    dataset = Path(dataset_dir)
    train_dir = dataset.joinpath("train")
    test_dir = dataset.joinpath("test")
    train_yt, test_yt = load_ytmidi("/home/chromeilion/Code/Uni/uni2023S"
                                    "/thesis/coding/external_test_data"
                                    "/PianoYT/")
    train = load_data(train_dir)
    test = load_data(test_dir)
    train.extend(train_yt)
    test.extend(test_yt)
    train = PlayingDataset(
        samples=train,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        frame_transform=processor
    )
    test = PlayingDataset(
        samples=test,
        temporal_res=temporal_res,
        clips_per_vid=2,
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
        do_train=True,
        do_eval=True,
        warmup_ratio=warmup_ratio,
        seed=seed,
        save_steps=save_every,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6
    )
    trainer = Trainer(
        optimizers=(optimizer, lr_sched),
        model=model,
        args=training_arguments,
        train_dataset=train,
        eval_dataset=test,
    )
    trainer.train(resume_from_checkpoint=checkpoint_dir)
