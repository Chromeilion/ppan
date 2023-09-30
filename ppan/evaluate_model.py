import pickle
from collections import defaultdict

import torch
from torch import multiprocessing
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)
from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    PreTrainedTokenizerFast,
    GenerationConfig,
    ViTForImageClassification
)

from ppan.config import device, pretrained_encoder, \
    pretrained_playing_det_vit, main_res, det_res
from ppan.dataset import load_data, EvalDataset, load_ytmidi, load_miditest
from ppan.types import PathLike
from ppan.preprocessing.midi import PPAnMidi


def main(dataset_dir: PathLike,
         output: PathLike,
         youtubemidi: PathLike,
         miditest: PathLike,
         model_checkpoint: PathLike,
         tokenizer_dir: PathLike,
         detector_checkpoint: PathLike,
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
    output : PathLike
        Where to output results (on tensorboard)
    model_checkpoint : PathLike
        Location of pretrained PPAn model.

    Returns
    -------
    None
    """
    evaluate(dataset_dir=dataset_dir[0],
             youtubemidi=youtubemidi[0],
             miditest=miditest[0],
             output=output[0],
             model_checkpoint=model_checkpoint[0],
             tokenizer_dir=tokenizer_dir[0],
             detector_checkpoint=detector_checkpoint[0])


def evaluate(dataset_dir: PathLike, output: PathLike,
             model_checkpoint: PathLike, tokenizer_dir: PathLike,
             youtubemidi: PathLike, miditest: PathLike,
             detector_checkpoint: PathLike):
    temporal_res = 1/30
    model = VisionEncoderDecoderModel.from_pretrained(
        model_checkpoint
    ).eval().to(device)
    detector = ViTForImageClassification.from_pretrained(detector_checkpoint).eval().to(device)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_dir
    )
    tokenizer.model_max_length = 30
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
        max_new_tokens=30,
        min_new_tokens=1,
        num_beams=10,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.cls_token_id,
        length_penalty=2,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    image_processor_main = AutoImageProcessor.from_pretrained(pretrained_encoder,
                                                              image_size=main_res)
    image_processor_det = AutoImageProcessor.from_pretrained(pretrained_playing_det_vit,
                                                             image_size=det_res)
    preds_yt, preds_midi, preds_rach3 = None, None, None

    multiprocessing.set_start_method("spawn")

    if miditest is not None:
        miditest = load_miditest(miditest)
        dataset = EvalDataset(
            samples=miditest,
            rotate=True,
            temporal_res=temporal_res
        )
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1,
                                prefetch_factor=2)
        preds_midi = eval_loop(dataset=dataloader, model=model,
                               generation_config=generation_config,
                               tokenizer=tokenizer, detector=detector,
                               det_ft=image_processor_det,
                               main_ft=image_processor_main,
                               length=len(miditest))

    if dataset_dir is not None:
        samples = load_data(dataset_dir)
        dataset = EvalDataset(
            samples=samples,
            temporal_res=temporal_res
        )
        dataloader = DataLoader(dataset=dataset, batch_size=1,
                                num_workers=1, prefetch_factor=1)
        preds_rach3 = eval_loop(dataset=dataloader, model=model,
                                generation_config=generation_config,
                                tokenizer=tokenizer, detector=detector,
                                det_ft=image_processor_det,
                                main_ft=image_processor_main,
                                length=len(samples))

    if youtubemidi is not None:
        _, ytmidi_test = load_ytmidi(youtubemidi)
        ytmidi_test = ytmidi_test
        dataset = EvalDataset(
            samples=ytmidi_test,
            temporal_res=temporal_res,
            rotate=True
        )
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        preds_yt = eval_loop(dataset=dataloader, model=model,
                             generation_config=generation_config,
                             tokenizer=tokenizer, detector=detector,
                             det_ft=image_processor_det,
                             main_ft=image_processor_main,
                             length=len(ytmidi_test))

    preds_dict = {
        "PainoYT": preds_yt,
        "MIDItest": preds_midi,
        "Rach3": preds_rach3
    }
    with open(output, "wb") as f:
        pickle.dump(obj=preds_dict, file=f)


def eval_loop(dataset, model, generation_config, tokenizer, detector,
              det_ft, main_ft, length):
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
            pixel_values_det = det_ft(torch.squeeze(i["pixel_values"]),
                                      return_tensors='pt',
                                      size=det_res).to(device)
            det_out = detector(pixel_values_det['pixel_values'])
            det_pred = float(det_out.logits)
            pixel_values_main = main_ft(i["pixel_values"],
                                        return_tensors='pt',
                                        size=main_res).to(device)
            generated_ids = model.generate(pixel_values_main['pixel_values'],
                                           generation_config=generation_config)
            generated_text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True)[0].split(" ")

            sentences.append((det_pred, generated_text, i['labels']))
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
