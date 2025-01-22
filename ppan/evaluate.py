import json
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

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

from ppan.config import fps, temporal_res, device, processed_temporal_size, model_no_frames
from ppan.utils import load_all_data
from ppan.midi import PPAnMidi
from ppan.preprocessor import DatasetProcessor
from ppan.model import PPANModel, PPANVideoProcessor


PathLike = Union[str, bytes, os.PathLike]

# This is used when saving predictions to ensure each prediction has a
# unique name. It's not ideal obviously but just a quick way to get it working.
global filecounter
filecounter = 0


def evaluate(preds_output: PathLike,
             model_checkpoint: PathLike,
             rach3_s_dir: Optional[PathLike] = None,
             rach3_x_dir: Optional[PathLike] = None,
             pianoyt_dir: Optional[PathLike] = None,
             miditest_dir: Optional[PathLike] = None,
             midi_output: Optional[PathLike] = None,
             threshold: Optional[float] = None,
             batch_size: Optional[int] = None,
             *_, **__):
    if not [i for i in [rach3_s_dir, rach3_x_dir, pianoyt_dir, miditest_dir] if i is not None]:
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
        batch_size = 2
    else:
        batch_size = int(batch_size)
    gaussian_sigma = 0.2
    gaussian_sigma_frames = 0.5

    # TODO: Get up to date with the rest of the codebase
    _, _, miditest, pianoyt_test, rach3_s_test, rach3_x_test = load_all_data(
        rach3_s_dir, rach3_x_dir, pianoyt_dir, miditest_dir
    )
#    omaps_test, _ = load_omaps(os.environ["PPAN_OMAPS_DIR"])
#    datasets = [omaps_test, rach3_test, miditest, pianoyt_test]
#    dataset_names = ["omaps", "rach3", "miditest", "pianoyt"]
    datasets = [rach3_s_test, rach3_x_test, miditest, pianoyt_test]
    dataset_names = ["r3s", "r3x", "miditest", "pianoyt"]

    [evaluate_on_dataset(
        i,
        dataset_name=j,
        model_checkpoint=model_checkpoint,
        batch_size=batch_size,
        gaussian_sigma=gaussian_sigma,
        gaussian_sigma_frames=gaussian_sigma_frames,
        threshold=threshold,
        midi_output=midi_output
    ) for i, j in zip(datasets, dataset_names) if i is not None]


def evaluate_on_dataset(samples, dataset_name, model_checkpoint, batch_size,
                        gaussian_sigma, gaussian_sigma_frames, threshold, midi_output):
    preds_output = Path(dataset_name+"_preds.pkl")
    if not preds_output.exists():
        model = PPANModel.from_pretrained(
            model_checkpoint
        ).eval().to(device)
        processor = PPANVideoProcessor()
        dataset = DatasetProcessor(
            datasets=samples,
            video_transform=processor,
            batch_size=1,
            temporal_size=model.config.num_frames/30,
            epoch_size=1,
            step=1,
            cachefile_name=f"{dataset_name}_cache.txt"
        )
        with no_grad():
            preds_rach3 = eval_loop(dataset=dataset,
                                    model=model)

        with open(preds_output, "wb") as f:
            pickle.dump(obj=preds_rach3, file=f)
    else:
        with open(preds_output, "rb") as f:
            preds_rach3 = pickle.load(f)
            # The saved preds may be from a different machine, therefore,
            # we fix the paths here.

        fixed_preds = {}
        for key, val in preds_rach3.items():
            p_name = Path(key).name
            all_vid_names = [Path(i.video_path).name for i in samples]
            fixed_preds[samples[all_vid_names.index(p_name)].video_path] = preds_rach3[key]

        preds_rach3 = fixed_preds

    mir_stats = [np.zeros(4), np.zeros(4), np.zeros(3)]
    all_stats = []
    all_vid_paths = []
    for vid_path, preds in preds_rach3.items():
        video_len = float(MultimediaTools().ff_probe(vid_path)["streams"][0]["duration"])
        pred_step = video_len / max([i[1] for i in preds])
        preds = [(i[0], i[1]*pred_step) for i in preds]
        final_pred = calc_time(preds)
        final_pred_onset = [(i[0], i[1][0].squeeze()) for i in final_pred]
        final_pred_frame = [(i[0], i[1][1].squeeze()) for i in final_pred]
        pianoroll = final_pred_to_onset_offset_array(
            final_pred_onset, final_pred_frame, threshold, gaussian_sigma, gaussian_sigma_frames
        )
        pianoroll = pianoroll.astype(int) * 100
        session_files = [i for i in samples if os.path.basename(vid_path) in os.path.basename(str(i.video_path))][0]
        vid_len = MultimediaTools().get_decoded_duration(session_files.video_path)
        try:
            labels = session_files.labels
        except AttributeError:
            labels = PPAnMidi(vid_len, temporal_res, 0)
            labels.set_midi(session_files.midi_path)

        vid_framerate = vid_len/pianoroll.shape[1]
        loc_mir_stats = calc_stats(
            midi=labels,
            pianoroll=pianoroll,
            framerate=vid_framerate
        )
        all_stats.append(loc_mir_stats)
        all_vid_paths.append(vid_path)
        for i, stats in enumerate(loc_mir_stats):
            mir_stats[i] += np.array(stats)
        if midi_output is not None:
            vid_path = Path(vid_path)
            midi_output = Path(midi_output)
            midi_output.mkdir(exist_ok=True)
            mid_output = midi_output/(vid_path.stem + ".mid")
            save_to_midi(pianoroll, str(mid_output))
    with open(f"./{dataset_name}_mir_stats.json", "w") as f:
        json.dump([list(i/len(all_stats)) for i in mir_stats], f)
#    np.save(f"./{dataset_name}_all_mir_stats", np.array(all_stats),
#            allow_pickle=True)
#    np.save(f"./{dataset_name}_all_mir_stats_files", np.array(all_vid_paths),
#            allow_pickle=True)


def final_pred_to_onset_offset_array(final_pred, final_pred_frame, threshold, sigma, sigma_frames) -> tuple[np.ndarray, np.ndarray]:
    """Take model predictions and create an onset array (basically a
    pianoroll but with only onsets) and an offset array. When the model
    predicts a note over multiple frames, the middle predicted frame is used
    as the onset. Also smooths the model output using a gaussian.
    """
    pred_array = np.array([i[1] for i in final_pred]).astype(float)
    pred_array = gaussian_filter(pred_array, axes=[0], sigma=sigma,
                                 radius=8)
    pred_array = pred_array > threshold
    pred_array_frame = np.array([i[1] for i in final_pred_frame]).astype(float)
    pred_array_frame = gaussian_filter(pred_array_frame, axes=[0], sigma=sigma_frames,
                                 radius=4)
    pred_array_frame = pred_array_frame > threshold
    nonzero_preds = pred_array.nonzero()
    nonzero_preds = list(zip(nonzero_preds[0], nonzero_preds[1]))

    nonzero_preds.sort(key=lambda val: (val[1], val[0]))

    p_x, p_y = nonzero_preds[0]
    pp_x = p_x
    pianoroll = np.zeros(shape=pred_array.shape, dtype=np.bool_)
    for x, y in nonzero_preds[1:]:
        if y != p_y or x - 1 != pp_x:
            mid_idx = p_x + (pp_x - p_x) // 2
            pianoroll[mid_idx, p_y] = 1
            current_frame = pred_array_frame[mid_idx, p_y]
            while current_frame == 1 and mid_idx < pred_array_frame.shape[0]:
                current_frame = pred_array_frame[mid_idx, p_y]
                pianoroll[mid_idx, p_y] = 1
                mid_idx += 1
            p_x = x
            p_y = y
        pp_x = x

    return pianoroll.T


def eval_loop(dataset, model):
    preds_dict = defaultdict(list)
    sig = nn.Sigmoid()
    for i in tqdm(dataset):
        pixel_values = i['pixel_values'][:, :model.config.num_frames, ...].to(device)
        model_out = model(pixel_values)
        logits = model_out["logits"]
        preds = sig(logits)
        for timestamps, sample, pred in zip(i['times'].to(device),
                                              i['sample'],
                                              preds):
            preds_dict[str(sample.video_path)].append((pred.cpu().numpy(),
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
    if isinstance(perf, str):
        intervals = []
        notes = []
        with open(perf, "r") as f:
            for line in f:
                line = [i.strip() for i in line.split("\t")]
                intervals.append((float(line[0]), float(line[1])))
                notes.append(mir_eval.util.midi_to_hz(int(line[2])))
        return np.array(intervals), np.array(notes)
    else:
        intervals = np.array(
            [[i['note_on'], i['note_off']] for i in perf[0].notes])
        equal = np.where(intervals[:, 1] <= intervals[:, 0])
        if equal:
            intervals[equal, 1] += 0.0001
        pitches = np.array(
            [mir_eval.util.midi_to_hz(i['midi_pitch']) for i in perf[0].notes])
        return intervals, pitches


def calc_perf_eval(pred_perf, true_perf):
    est_intervals, est_pitches = perf_to_int_pitch(pred_perf)
    ref_intervals, ref_pitches = perf_to_int_pitch(true_perf)
    # Shift our predictions forward 3 frames. I'm not 100% sure why this is
    # necessary.
    est_intervals[:, 0] += 3*temporal_res
    est_intervals[:, 1] += 2.3*temporal_res
    savedir = Path("./trans_res")
    savedir.mkdir(exist_ok=True)
    resdic = {"est_intervals": [list(i) for i in list(est_intervals.astype(float))],
              "ref_intervals": [list(i) for i in list(ref_intervals.astype(float))],
              "est_pitches": list(est_pitches.astype(float)),
              "ref_pitches": list(ref_pitches.astype(float))}

    global filecounter
    try:
        if filecounter > 10:
            pass
    except UnboundLocalError:
        filecounter = 0
    with open(savedir/f"{filecounter}.json", "w") as f:
        json.dump(resdic, f)
    filecounter += 1
    return [mir_eval.transcription.precision_recall_f1_overlap(
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches
    ),
        mir_eval.transcription.precision_recall_f1_overlap(
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        offset_ratio=None
        ),
        mir_eval.transcription.offset_precision_recall_f1(
            est_intervals=est_intervals,
            ref_intervals=ref_intervals,
        ),
    ]


def calc_stats(midi: PPAnMidi, pianoroll: np.ndarray, framerate: float):
    # MIR Eval stats
    note_array_pred = pianoroll_to_notearray(pianoroll,
                                             time_div=30,
                                             time_unit="sec")
    performance_pred = Performance(
        PerformedPart.from_note_array(
            note_array=note_array_pred
        )
    )
    if isinstance(midi, PPAnMidi):
        perf_true = midi.performance
    else:
        perf_true = midi
    mir_scores = calc_perf_eval(performance_pred, perf_true)

    return mir_scores


def calc_time(preds):
    final_preds = []
    for pred, times in preds:
        time = times + model_no_frames/2
        final_preds.append((time, pred))
    return final_preds
