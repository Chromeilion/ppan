import pickle
from collections import defaultdict

from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor
)

from ppan.config import device, pretrained_model
from ppan.dataset import load_rach3, ImageVecDataset
from ppan.midi import PPAnMidi
from ppan.types import PathLike


def main(dataset_dir: list[PathLike],
         output: PathLike,
         model_checkpoint: PathLike,
         *_, **__):
    """
    Evaluation function for PPAn. Takes a trained model and runs it through a
    series of videos. Then it computes the loss between the generated note
    array and target note array.

    Parameters
    ----------
    dataset_dir : list[PathLike]
        Directory with train and test set in their own folders.
    output : PathLike
        Where to output results (on tensorboard)
    model_checkpoint : PathLike
        Location of pretrained PPAn model.

    Returns
    -------
    None
    """
    evaluate(dataset_dir=dataset_dir,
             output=output[0],
             model_checkpoint=model_checkpoint[0])


def evaluate(dataset_dir: list[PathLike], output: PathLike,
             model_checkpoint: PathLike):
    batch_size = 2
    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint
    ).eval().to(device)
    processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model)

    test, _ = load_rach3(dataset_dir[0])
    dataset = ImageVecDataset(
        datasets=[test],
        video_transform=processor,
        batch_size=batch_size,
        random_=False,
        cachefile_name="./train_cache.txt",
        max_iters_per_epoch=max_iters_per_epoch_train
    )

    preds_rach3 = eval_loop(dataset=dataset, model=model)

    with open(output, "wb") as f:
        pickle.dump(obj=preds_dict, file=f)


def eval_loop(dataset, model):
    preds = defaultdict(dict)
    prev_file = None
    sentences = []
    for i in tqdm(dataset):
        midi_path = i["midi_path"][0]
        if prev_file is None:
            prev_file = midi_path
        if prev_file != midi_path:
            preds[prev_path]["sentences"] = sentences
            preds[prev_path]["time"] = current_time
            sentences = []

        prev_file = midi_path
        prev_path = i["midi_path"][0]
        current_time = i['time']

    return preds


def calculate_pianorolls(preds_dict: dict[str, dict]):
    midi = PPAnMidi()
    for i in preds_dict.items():
        sentences = [j[0] for j in i[1]["sentences"]]
        preds_dict[i[0]]["pred_pianoroll"] = midi.sentences_to_pianoroll(
            sentences=sentences
        )
    return preds_dict
