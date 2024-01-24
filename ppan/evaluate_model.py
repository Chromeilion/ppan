import pickle
from collections import defaultdict

from torch import multiprocessing
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from transformers import (
    VivitForVideoClassification,
    AutoImageProcessor
)

from ppan.config import device, main_res, pretrained_model
from ppan.dataset import EvalDataset, load_ytmidi, load_miditest, load_rach3_split
from ppan.types import PathLike
from ppan.preprocessing.midi import PPAnMidi


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
    temporal_res = 1/30
    model = VivitForVideoClassification.from_pretrained(
        model_checkpoint
    ).eval().to(device)
    image_processor_main = AutoImageProcessor.from_pretrained(pretrained_model)

    preds_yt, preds_midi, preds_rach3 = None, None, None

    multiprocessing.set_start_method("spawn")


    miditest = load_miditest(dataset_dir[2])
    dataset = EvalDataset(
        samples=miditest,
        rotate=True,
        temporal_res=temporal_res
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1,
                            prefetch_factor=2)
    preds_midi = eval_loop(dataset=dataloader, model=model,
                           main_ft=image_processor_main,
                           length=len(miditest))

    samples = load_rach3_split(dataset_dir[0])
    dataset = EvalDataset(
        samples=samples,
        temporal_res=temporal_res
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                            num_workers=1, prefetch_factor=1)
    preds_rach3 = eval_loop(dataset=dataloader, model=model,
                            main_ft=image_processor_main,
                            length=len(samples))

    _, ytmidi_test = load_ytmidi(dataset_dir[1])
    ytmidi_test = ytmidi_test
    dataset = EvalDataset(
        samples=ytmidi_test,
        temporal_res=temporal_res,
        rotate=True
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    preds_yt = eval_loop(dataset=dataloader, model=model,
                         main_ft=image_processor_main,
                         length=len(ytmidi_test))

    preds_dict = {
        "PainoYT": preds_yt,
        "MIDItest": preds_midi,
        "Rach3": preds_rach3
    }
    with open(output, "wb") as f:
        pickle.dump(obj=preds_dict, file=f)


def eval_loop(dataset, model, main_ft, length):
    preds = defaultdict(dict)
    with tqdm(total=length) as pbar:
        prev_file = None
        sentences = []
        for i in dataset:
            midi_path = i["midi_path"][0]
            if prev_file is None:
                prev_file = midi_path
            if prev_file != midi_path:
                pbar.update()
                preds[prev_path]["sentences"] = sentences
                preds[prev_path]["time"] = current_time
                sentences = []
            pixel_values_main = main_ft(i["pixel_values"],
                                        return_tensors='pt',
                                        size=main_res).to(device)

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


def calculate_jaccard(preds_dict):
    scores = []
    midi = PPAnMidi()
    preds_list = []
    for i in preds_dict.values():
        preds_list.extend(i["sentences"])
    for i in preds_list:
        if i[0]:
            scores.append(jaccard_score(
                y_pred=midi.sentence_to_note_vec(i[0]),
                y_true=midi.sentence_to_note_vec(i[1])
            ))
    return sum(scores) / len(scores)
