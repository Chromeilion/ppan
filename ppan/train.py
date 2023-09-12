from torch import multiprocessing
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)

from ppan.config import seed, pretrained_encoder, main_res, VID_BACKEND
from ppan.dataset import load_data, ImageVecDataset
from ppan.types import PathLike


def main(dataset_dir: PathLike,
         output_dir: PathLike,
         decoder_checkpoint: PathLike,
         tokenizer_dir: PathLike,
         checkpoint_dir: Optional[PathLike],
         *_, **__):
    """
    Training function for PPAn.

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
    if checkpoint_dir:
        checkpoint_dir = checkpoint_dir[0]
    train_image(dataset_dir=dataset_dir[0],
                output_dir=output_dir[0],
                decoder_checkpoint=decoder_checkpoint[0],
                tokenizer_dir=tokenizer_dir[0],
                checkpoint_dir=checkpoint_dir)


def train_image(dataset_dir: PathLike, output_dir: PathLike,
                decoder_checkpoint: PathLike, tokenizer_dir: PathLike,
                checkpoint_dir: Optional[PathLike] = None):
    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    max_steps = 30000
    temporal_res = 3
    clips_per_vid = 8
    eval_every = 300
    save_every = 300

    """Optimizer Config"""
    lr = 1e-5
    weight_decay = 1e-4

    """Scheduler Config"""
    warmup_ratio = 0.15
    scheduler_type = "cosine"

    """What Encoder to Use"""
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_dir)

    """Model Config"""
    image_size = main_res

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder,
        decoder_checkpoint,
        encoder_image_size=image_size,
        encoder_ignore_mismatched_sizes=True
    ).train()
    tokenizer.model_max_length = 30
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    processor = AutoImageProcessor.from_pretrained(pretrained_encoder,
                                                   size=image_size)

    dataset = Path(dataset_dir)
    train_dir = dataset.joinpath("train")
    test_dir = dataset.joinpath("test")

    train = load_data(train_dir)
    test = load_data(test_dir)
    train = ImageVecDataset(
        samples=train,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        tokenizer=tokenizer,
        frame_transform=processor
    )
    test = ImageVecDataset(
        samples=test,
        temporal_res=10,
        clips_per_vid=2,
        tokenizer=tokenizer,
        frame_transform=processor,
        epoch_size=1
    )
    training_arguments = TrainingArguments(
        auto_find_batch_size=True,
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
        optim="adamw_torch_fused",
        weight_decay=weight_decay,
        save_steps=save_every,
        torch_compile=True
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train,
        eval_dataset=test,
    )
    multiprocessing.set_start_method("spawn", force=True)
    trainer.train(resume_from_checkpoint=checkpoint_dir)
