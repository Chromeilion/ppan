from pathlib import Path

import torch
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
         model_checkpoint: PathLike,
         tokenizer_dir: PathLike,
         *_, **__):
    """
    Evaluation function for PPAn. Takes a trained model and runs it through a
    series of videos. Then it computes the loss between the generated note
    array and target note array.

    Parameters
    ----------
    tokenizer_dir : PathLike
    dataset_dir : PathLike
        Directory with train and test set in their own folders.
    output_dir : PathLike
        Where to output results (on tensorboard)
    model_checkpoint : PathLike
        Location of pretrained PPAn model.

    Returns
    -------
    None
    """
    evaluate(dataset_dir=dataset_dir[0],
             output_dir=output_dir[0],
             model_checkpoint=model_checkpoint[0],
             tokenizer_dir=tokenizer_dir[0])


def evaluate(dataset_dir: PathLike, output_dir: PathLike,
             model_checkpoint: PathLike, tokenizer_dir: PathLike):
    encoder = "facebook/vit-mae-large"
    temporal_res = 1

    model = VisionEncoderDecoderModel.from_pretrained(
        model_checkpoint
    ).to(device)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_dir
    )
    image_processor = AutoImageProcessor.from_pretrained(encoder)

    samples = load_data(dataset_dir)

    dataset = ImageVecDataset(
        samples=samples,
        temporal_res=temporal_res,
        tokenizer=tokenizer,
        frame_transform=image_processor
    )
    for i in dataset:
        pixel_values = torch.unsqueeze(
            torch.tensor(i["pixel_values"]).to(device), dim=0
        )
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True)[0]
        label = tokenizer.batch_decode(
            i["labels"]
        )
        a = "b"
