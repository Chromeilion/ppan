import numpy as np
import pickle
import json

from rach3datautils.utils.multimedia import MultimediaTools
from partitura.utils import pianoroll_to_notearray
from partitura.performance import PerformedPart, Performance
from partitura import save_performance_midi
from pathlib import Path
from collections import defaultdict
import mir_eval

from torch import no_grad
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor
)

from ppan.config import device, pretrained_model, fps, temporal_res
from ppan.dataset import load_rach3, PPAnEvalDataset
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
    threshold = 0.83

    test, _ = load_rach3(dataset_dir[0])

    if not Path(output).exists():
        batch_size = 20
        model = VideoMAEForVideoClassification.from_pretrained(
            model_checkpoint
        ).eval().to(device)
        processor = VideoMAEImageProcessor.from_pretrained(
            pretrained_model)

        dataset = PPAnEvalDataset(
            datasets=[[test[0]]],
            video_transform=processor,
            batch_size=batch_size,
        )
        with no_grad():
            preds_rach3 = eval_loop(dataset=dataset, model=model)

        with open(output, "wb") as f:
            pickle.dump(obj=preds_rach3, file=f)
    else:
        with open(output, "rb") as f:
            preds_rach3 = pickle.load(f)

    for vid_path, preds in preds_rach3.items():
        final_pred = threshold_and_calc_time(preds, threshold)
        times = np.array([i[0] for i in final_pred])
        onset_array = final_pred_to_onset_array(final_pred)

        session_files = [i for i in test if vid_path in str(i[2])][0]
        vid_len = MultimediaTools().ff_probe(session_files[2])
        vid_len = float(vid_len["streams"][0]["duration"])
        midi = PPAnMidi(vid_len, temporal_res, 0).set_midi(session_files[0])
        mir_stats = calc_stats(
            midi=midi,
            onset_array=onset_array
        )
        with open("./mir_stats", "w") as f:
            json.dump(mir_stats, f)

        save_to_midi(onset_array, "midi_preds.mid")


def final_pred_to_onset_array(final_pred) -> np.ndarray:
    """Take model predictions and create an onset array (basically a
    pianoroll but with only onsets). When the model predicts a note over
    multiple frames, the middle predicted frame is used as the onset.
    """
    pred_array = np.array([i[1] for i in final_pred])

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


def eval_loop(dataset, model):
    preds_dict = defaultdict(list)
    sig = nn.Sigmoid()
    for i in tqdm(dataset):
#        preds = sig(model(i['pixel_values']))
        preds = i['labels']
        all_files = dataset.get_all_video_samples()
        for timestamps, file_idx, pred in zip(i['timestamps'], i['file_idx'],
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
    save_performance_midi(performance_data=performance, out=f"./midis/{name}")


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
    est_intervals, est_pitches = perf_to_int_pitch(pred_perf)
    # This line is necessary because the model labels are actually
    # calculated between two frames, therefore, to align the predictions
    # properly, we need to shift everything half a frame.
    est_intervals += temporal_res / 2
    ref_intervals, ref_pitches = perf_to_int_pitch(true_perf)

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


def threshold_and_calc_time(preds, threshold):
    final_preds = []
    for pred, times in preds:
        vals = pred > threshold
        middle = times.shape[0] // 2
        time = times[middle]
        if times.shape[0] % 2 == 0:
            time += times[middle+1]
            time = time / 2.
        final_preds.append((time, vals))
    # Because the windows dont start at time zero, we need to insert a few
    # frames at the start of the preds so that they start at zero.
    no_to_insert = int(final_preds[0][0] / (1./30.))
    [final_preds.insert(
        0,
        (temporal_res*i, np.zeros(shape=final_preds[0][1].shape, dtype=np.bool_)))
        for i in reversed(range(no_to_insert))]
    return final_preds
