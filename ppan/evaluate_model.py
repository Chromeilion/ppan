import json
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from accelerate import Accelerator
import mir_eval
import numpy as np
import torch.nn as nn
from partitura import save_performance_midi
from partitura.performance import PerformedPart, Performance
from partitura.utils import pianoroll_to_notearray
from rach3datautils.utils.multimedia import MultimediaTools
from scipy.ndimage import gaussian_filter
from torch import no_grad
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor
)

from ppan.config import pretrained_model, fps, temporal_res
from ppan.dataset import load_all_data, PPAnEvalDataset
from ppan.midi import PPAnMidi

PathLike = Union[str, bytes, os.PathLike]


def evaluate(preds_output: PathLike,
             model_checkpoint: PathLike,
             rach3_dir: Optional[PathLike] = None,
             pianoyt_dir: Optional[PathLike] = None,
             miditest_dir: Optional[PathLike] = None,
             midi_output: Optional[PathLike] = None,
             threshold: Optional[float] = None,
             batch_size: Optional[int] = None,
             *_, **__):
    if not [i for i in [rach3_dir, pianoyt_dir, miditest_dir] if i is not None]:
        raise AttributeError("A dataset directory is required to run "
                             "evaluation.")
    if preds_output is None:
        raise AttributeError("A path to the output file is required.")
    if model_checkpoint is None:
        warnings.warn("No model checkpoint passed. This is not an issue if "
                      "the model predictions have already been calculated and "
                      "the correct path to these predictions is passed in "
                      "preds_output.")
    if threshold is None:
        threshold = 0.5
    if batch_size is None:
        batch_size = 8
    gaussian_sigma = 1

    _, _, miditest, pianoyt_test, rach3_test = load_all_data(
        rach3_dir, pianoyt_dir, miditest_dir
    )
    datasets = [miditest, pianoyt_test, rach3_test]
    dataset_names = ["miditest", "pianoyt", "rach3"]
    [evaluate_on_dataset(
        i,
        dataset_name=j,
        model_checkpoint=model_checkpoint,
        batch_size=batch_size,
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
        midi_output=midi_output
    ) for i, j in zip(datasets, dataset_names)]


def evaluate_on_dataset(samples, dataset_name, model_checkpoint, batch_size,
                        gaussian_sigma, threshold, midi_output):
    preds_output = Path(dataset_name+"_preds.pkl")
    if not preds_output.exists():
        processor = VideoMAEImageProcessor.from_pretrained(
            pretrained_model
        )
        dataset = PPAnEvalDataset(
            datasets=[samples[5]],
            video_transform=processor,
            batch_size=batch_size,
            step=1
        )
        with no_grad():
            preds_rach3 = eval_loop(dataset=dataset,
                                    model_checkpoint=model_checkpoint)

        with open(preds_output, "wb") as f:
            pickle.dump(obj=preds_rach3, file=f)
    else:
        with open(preds_output, "rb") as f:
            preds_rach3 = pickle.load(f)
    for vid_path, preds in preds_rach3.items():
        final_pred = calc_time(preds)
        onset_array = final_pred_to_onset_array(final_pred, threshold,
                                                gaussian_sigma)
        onset_array = onset_array.astype(int) * 100
        session_files = [i for i in samples if vid_path in str(i[2])][0]
        vid_len = MultimediaTools().ff_probe(session_files[2])
        vid_len = float(vid_len["streams"][0]["duration"])
        midi = PPAnMidi(vid_len, temporal_res, 0)
        midi.set_midi(session_files[0])
        mir_stats = calc_stats(
            midi=midi,
            onset_array=onset_array
        )
        with open(f"./{dataset_name}_mir_stats.json", "w") as f:
            json.dump(mir_stats, f)

        if midi_output is not None:
            vid_path = Path(vid_path)
            midi_output = Path(midi_output)
            midi_output.mkdir(exist_ok=True)
            mid_output = midi_output/(vid_path.stem + ".mid")
            save_to_midi(onset_array, str(mid_output))


def final_pred_to_onset_array(final_pred, threshold, sigma) -> np.ndarray:
    """Take model predictions and create an onset array (basically a
    pianoroll but with only onsets). When the model predicts a note over
    multiple frames, the middle predicted frame is used as the onset.
    Also smooths the model output using a gaussian.
    """
    pred_array = np.array([i[1] for i in final_pred])
    pred_array = gaussian_filter(pred_array, axes=[0], sigma=sigma,
                                 radius=16)
    pred_array = pred_array > threshold
    revised_preds = []
    nonzero_preds = pred_array.nonzero()
    nonzero_preds = list(zip(nonzero_preds[0], nonzero_preds[1]))

    nonzero_preds.sort(key=lambda val: (val[1], val[0]))

    p_x, p_y = nonzero_preds[0]
    pp_x = p_x
    for x, y in nonzero_preds[1:]:
        if y != p_y or x - 1 != pp_x:
            mid_idx = p_x + (pp_x - p_x) // 2
            revised_preds.append((mid_idx, p_y))
            p_x = x
            p_y = y
        pp_x = x

    onset_array = np.zeros(shape=pred_array.shape, dtype=np.bool_)
    onset_array[
        [i[0] for i in revised_preds], [i[1] for i in revised_preds]
    ] = 1

    # Sanity check to make sure we haven't messed anything obvious up
    nonzero_onsets = onset_array.nonzero()
    nonzero_onsets = list(zip(nonzero_onsets[0], nonzero_onsets[1]))
    assert all([i in nonzero_preds for i in nonzero_onsets])

    return onset_array.T


def eval_loop(dataset, model_checkpoint):
    accelerator = Accelerator()
    ac_device = accelerator.device

    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint
    ).eval().to(ac_device)

    dataset, model = accelerator.prepare(
        dataset, model
    )

    preds_dict = defaultdict(list)
    sig = nn.Sigmoid()
    for i in tqdm(dataset):
        logits = model(i['pixel_values'].to(ac_device)).logits
        preds = sig(logits)
        all_files = dataset.get_all_video_samples()
        for timestamps, file_idx, pred in zip(i['timestamps'].to(ac_device),
                                              i['file_idx'].to(ac_device),
                                              preds):
            vid_file = all_files[file_idx]
            preds_dict[vid_file].append((pred.cpu().numpy(),
                                         timestamps.cpu().numpy()))

    return preds_dict


def save_to_midi(onset_array, name: str):
    note_array = pianoroll_to_notearray(onset_array,
                                        time_div=fps,
                                        time_unit="sec")
    performance = Performance(
        PerformedPart.from_note_array(
            note_array=note_array
        )
    )
    save_performance_midi(performance_data=performance, out=name)


def perf_to_int_pitch(perf):
    intervals = np.array(
        [[i['note_on'], i['note_off']] for i in perf[0].notes])
    equal = np.where(intervals[:, 1] <= intervals[:, 0])
    if equal[0]:
        intervals[equal, 1] += 0.0001
    pitches = np.array(
        [mir_eval.util.midi_to_hz(i['midi_pitch']) for i in perf[0].notes])
    return intervals, pitches


def calc_perf_eval(pred_perf, true_perf):
    ref_intervals, ref_pitches = perf_to_int_pitch(true_perf)
    est_intervals, est_pitches = perf_to_int_pitch(pred_perf)
    # This line is necessary because the model labels are actually
    # calculated between two frames, therefore, to align the predictions
    # properly, we need to shift everything half a frame.
    est_intervals += temporal_res / 2.

    return mir_eval.transcription.precision_recall_f1_overlap(
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        offset_ratio=None
    )


def calc_stats(midi: PPAnMidi, onset_array: np.ndarray):
    # MIR Eval stats
    note_array_pred = pianoroll_to_notearray(onset_array,
                                             time_div=fps,
                                             time_unit="sec")
    performance_pred = Performance(
        PerformedPart.from_note_array(
            note_array=note_array_pred
        )
    )

    perf_true = midi.performance
    mir_scores = calc_perf_eval(performance_pred, perf_true)

    return mir_scores


def calc_time(preds):
    final_preds = []
    for pred, times in preds:
        middle = times.shape[0] // 2
        time = times[middle]
        if times.shape[0] % 2 == 0:
            time += times[middle+1]
            time = time / 2.
        final_preds.append((time, pred))
    # Because the windows don't start at time zero, we need to insert a few
    # frames at the start of the preds so that they start at zero.
    no_to_insert = int(final_preds[0][0] / (1./30.))
    [final_preds.insert(
        0,
        (temporal_res*i, np.zeros(shape=final_preds[0][1].shape,
                                  dtype=np.bool_)))
        for i in reversed(range(no_to_insert))]
    return final_preds
