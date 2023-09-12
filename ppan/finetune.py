from torch import multiprocessing

from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)

from ppan.config import seed, pretrained_encoder, main_res, \
    pretrained_playing_det_vit, det_res
from ppan.dataset import load_ytmidi, ImageVecDataset
from ppan.types import PathLike


def main(dataset_dir: PathLike,
         output_dir: PathLike,
         tokenizer_dir: PathLike,
         dataset_type,
         pretrained_det_model,
         pretrained_note_model,
         *_, **__):
    train_image(dataset_dir=dataset_dir[0],
                output_dir=output_dir[0],
                tokenizer_dir=tokenizer_dir[0],
                dataset_type=dataset_type[0],
                pretrained_det_model=pretrained_det_model[0],
                pretrained_note_model=pretrained_note_model[0])


def train_image(dataset_dir: PathLike, dataset_type: str,
                output_dir: PathLike, pretrained_note_model: PathLike,
                pretrained_det_model: PathLike, tokenizer_dir: PathLike):
    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    max_steps = 20000
    temporal_res = 3
    clips_per_vid = 8
    eval_every = 500
    save_every = 500
    batch_size = 4

    """Optimizer Config"""
    lr = 5e-6
    weight_decay = 1e-4

    """Scheduler Config"""
    warmup_ratio = 0.1
    scheduler_type = "cosine"

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_dir)

    note_model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_note_model)
    det_model = ViTForImageClassification.from_pretrained(pretrained_det_model)

    tokenizer.model_max_length = 30
    note_model.config.decoder_start_token_id = tokenizer.cls_token_id
    note_model.config.pad_token_id = tokenizer.pad_token_id
    note_processor = AutoImageProcessor.from_pretrained(pretrained_encoder,
                                                        size=main_res)
    det_processor = AutoImageProcessor.from_pretrained(
        pretrained_playing_det_vit,
        size=det_res
    )
    if dataset_type == "pianoyt":
        train, test = load_ytmidi(dataset_dir)
    else:
        raise NotImplementedError("The provided dataset_type is not "
                                  "supported.")

    note_train = ImageVecDataset(
        samples=train,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        tokenizer=tokenizer,
        frame_transform=note_processor,
        rotate=True
    )
    note_test = ImageVecDataset(
        samples=test,
        temporal_res=10,
        clips_per_vid=2,
        tokenizer=tokenizer,
        frame_transform=note_processor,
        epoch_size=1,
        rotate=True
    )
    det_train = ImageVecDataset(
        samples=train,
        temporal_res=temporal_res,
        clips_per_vid=clips_per_vid,
        tokenizer=tokenizer,
        frame_transform=det_processor,
        rotate=True
    )
    det_test = ImageVecDataset(
        samples=test,
        temporal_res=10,
        clips_per_vid=2,
        tokenizer=tokenizer,
        frame_transform=det_processor,
        epoch_size=1,
        rotate=True
    )
    training_arguments = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        optim_args="momentum=0.9",
        optim="sgd",
        weight_decay=weight_decay,
        save_steps=save_every
    )
    note_trainer = Trainer(
        model=note_model,
        args=training_arguments,
        train_dataset=note_train,
        eval_dataset=note_test,
    )
    det_trainer = Trainer(
        model=det_model,
        args=training_arguments,
        train_dataset=det_train,
        eval_dataset=det_test,
    )
    multiprocessing.set_start_method("spawn", force=True)
    note_trainer.train()
    det_trainer.train()
