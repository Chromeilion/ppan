from pathlib import Path

from transformers import (
    AutoTokenizer
)

from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)

from ppan.config import device, seed
from ppan.dataset import load_data, ImageVecDataset
from ppan.types import PathLike


def main(dataset_dir: PathLike,
         output_dir: PathLike,
         decoder_checkpoint: PathLike,
         tokenizer_dir: PathLike,
         *_, **__):
    """
    Training function for PPAn.

    Parameters
    ----------
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
    train_image(dataset_dir=dataset_dir[0],
                output_dir=output_dir[0],
                decoder_checkpoint=decoder_checkpoint[0],
                tokenizer_dir=tokenizer_dir[0])


def train_image(dataset_dir: PathLike, output_dir: PathLike,
                decoder_checkpoint: PathLike, tokenizer_dir: PathLike):
    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    epochs = 20
    temporal_res = 1
    clips_per_vid = 5
    eval_every = 200
    save_every = 300

    """Optimizer Config"""
    lr = 1e-5

    """Scheduler Config"""
    warmup_ratio = 0.2
    scheduler_type = "cosine"

    """What Encoder to Use"""
    pretrained_encoder = "facebook/vit-mae-base"

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder,
        decoder_checkpoint
    ).train().to(device)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_dir)
    tokenizer.model_max_length = 20
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    processor = AutoImageProcessor.from_pretrained(pretrained_encoder)

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
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        tokenizer=tokenizer,
        shuffle_every_loop=True,
        frame_transform=processor
    )
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        num_train_epochs=epochs,
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        optim="adamw_torch",
        auto_find_batch_size=True,
        save_steps=save_every,
        do_predict=True
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train,
        eval_dataset=test
    )
    trainer.train()
