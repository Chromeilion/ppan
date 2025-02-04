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

from ppan.config import fps, temporal_res, device, model_no_frames
from ppan.utils import load_all_data, load_omaps
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
        threshold_onset = 0.5
        threshold_frame = 0.5
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
    omaps_test = load_omaps(os.environ["PPAN_OMAPS_DIR"])
    datasets = [rach3_x_test, miditest, omaps_test, rach3_s_test, pianoyt_test]
    dataset_names = ["r3x", "miditest", "omaps", "r3s", "pianoyt"]

    [evaluate_on_dataset(
        i,
        dataset_name=j,
        model_checkpoint=model_checkpoint,
        batch_size=batch_size,
        gaussian_sigma=gaussian_sigma,
        gaussian_sigma_frames=gaussian_sigma_frames,
        threshold_frame=threshold_frame,
        threshold_onset=threshold_onset,
        midi_output=midi_output
    ) for i, j in zip(datasets, dataset_names) if i is not None]


def evaluate_on_dataset(samples, dataset_name, model_checkpoint, batch_size,
                        gaussian_sigma, gaussian_sigma_frames, threshold_frame, threshold_onset, midi_output):
    preds_output = Path(dataset_name+"_preds.pkl")
    model = PPANModel.from_pretrained(
        model_checkpoint
    ).eval().to(device)
    fr = int(MultimediaTools().ff_probe(samples[0].video_path)["streams"][0][
             "avg_frame_rate"].split("/")[0])
    if not preds_output.exists():
        processor = PPANVideoProcessor(
            resolution=model.config.image_size,
            grayscale=True,
        )
        chunk_size = 4 # in seconds
        dataset = DatasetProcessor(
            datasets=samples,
            video_transform=processor,
            batch_size=1,
            temporal_size=chunk_size, # 10 seconds
            epoch_size=1,
            step=((chunk_size*fr)-model.config.num_frames), # a little overlap so that we don't miss any windows
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
        video_frame_rate = int(MultimediaTools().ff_probe(vid_path)["streams"][0]["avg_frame_rate"].split("/")[0])
        n_frames = round(video_len * video_frame_rate)
        model_half_window = model.config.num_frames // 2 + model.config.num_frames % 2
        preds = [i for i in preds if i[1] + model.config.num_frames < n_frames]
        preds = [(i[0], i[1]+model_half_window) for i in preds]
        final_pred = calc_time(preds, model.config)
        final_pred_onset = [(i[0], i[1][0].squeeze()) for i in final_pred]
        final_pred_frame = [(i[0], i[1][1].squeeze()) for i in final_pred]
        pianoroll = final_pred_to_onset_offset_array(
            final_pred_onset, final_pred_frame, threshold_frame, threshold_onset, gaussian_sigma, gaussian_sigma_frames
        )
        pianoroll = pianoroll.astype(int) * 100
        session_files = [i for i in samples if os.path.basename(vid_path) in os.path.basename(str(i.video_path))][0]
        vid_len = MultimediaTools().get_decoded_duration(session_files.video_path)
        if session_files.note_intervals is not None:
            labels = session_files.note_intervals
        elif session_files.midi_path is not None:
            labels = PPAnMidi(vid_len, temporal_res, 0)
            labels.set_midi(session_files.midi_path)
        else:
            raise AttributeError(f"Missing midi or note intervals for {session_files.video_path}")

        loc_mir_stats = calc_stats(
            midi=labels,
            pianoroll=pianoroll,
            framerate=video_frame_rate
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
        stats = [list(i/len(all_stats)) for i in mir_stats]
        json.dump({"full_note": stats[0], "onsets": stats[1], "offsets_no_pitch": stats[2]},
                  f, indent=4)
#    np.save(f"./{dataset_name}_all_mir_stats", np.array(all_stats),
#            allow_pickle=True)
#    np.save(f"./{dataset_name}_all_mir_stats_files", np.array(all_vid_paths),
#            allow_pickle=True)


def final_pred_to_onset_offset_array(final_pred, final_pred_frame, threshold_frame, threshold_onset, sigma, sigma_frames) -> tuple[np.ndarray, np.ndarray]:
    """Take model predictions and create an onset array (basically a
    pianoroll but with only onsets) and an offset array. When the model
    predicts a note over multiple frames, the peak predicted frame is used
    as the onset. Also smooths the model output using a gaussian.
    """
    pred_array = np.zeros((max([i[0][0] for i in final_pred])+1, 88))
    pred_array_frame = np.zeros_like(pred_array)
    for (frame_onset, pred_onset), (frame_frame, pred_frame) in zip(final_pred, final_pred_frame):
        pred_array[frame_onset[0]] = pred_onset
        pred_array_frame[frame_frame[0]] = pred_frame

    pred_array = gaussian_filter(pred_array, axes=[0], sigma=sigma,
                                 radius=8)
    pred_array_mask = pred_array > threshold_onset
    frame_kernal_radius = 4
    pred_array_frame = gaussian_filter(pred_array_frame, axes=[0], sigma=sigma_frames,
                                 radius=frame_kernal_radius)
    pred_array_frame_mask = pred_array_frame > threshold_frame
    nonzero_preds = pred_array_mask.nonzero()
    nonzero_preds = list(zip(nonzero_preds[0], nonzero_preds[1]))

    nonzero_preds.sort(key=lambda val: (val[1], val[0]))

    pianoroll = np.zeros_like(pred_array, dtype=np.bool_)

    p_x, p_y = nonzero_preds[0]
    pp_x = p_x
    for x, y in nonzero_preds[1:]:
        if y != p_y or x - 1 != pp_x:
            peak_idx = p_x + pred_array[p_x:pp_x+1, p_y].argmax() - 1
            pianoroll[peak_idx, p_y] = 1
            peak_idx += 1
            current_frame = pred_array_frame_mask[peak_idx, p_y]
            while current_frame and peak_idx < pred_array_frame.shape[0]-frame_kernal_radius:
                pianoroll[peak_idx, p_y] = 1
                peak_idx += 1
                current_frame = pred_array_frame_mask[peak_idx, p_y]
                future_frames = pred_array_frame_mask[peak_idx:peak_idx+frame_kernal_radius+1, p_y]
                # We need to compensate for the gaussian smoothing, which
                # extends the offsets by some number of frames.
                if not future_frames.all():
                    break
            p_x = x
            p_y = y
        pp_x = x

    return pianoroll.T


def eval_loop(dataset, model):
    preds_dict = defaultdict(list)
    sig = nn.Sigmoid()
    window_size = model.config.num_frames
    odd = 1
    if window_size % 2 == 0:
        odd = 0

    for i in tqdm(dataset):
        pixel_values = i['pixel_values'].to(device)
        for j in range(window_size // 2, pixel_values.shape[1] - window_size // 2):
            model_out = model(pixel_values[:, j - window_size // 2:j + window_size // 2 + odd, ...])
            logits = model_out["logits"]
            preds = sig(logits)
            for timestamps, sample, pred in zip(i['times'].to(device)+j-window_size//2,
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

    est_intervals += 0.06
    est_intervals[:, 1] -= (est_intervals[:, 1] - est_intervals[:, 0]) * 0.06

    savedir = Path("./trans_res")
    savedir.mkdir(exist_ok=True)
    resdic = {"est_intervals": [list(i) for i in list(est_intervals.astype(float))],
              "ref_intervals": [list(i) for i in list(ref_intervals.astype(float))],
              "est_pitches": list(est_pitches.astype(float)),
              "ref_pitches": list(ref_pitches.astype(float))}

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


def calc_stats(midi: PPAnMidi, pianoroll: np.ndarray, framerate: int):
    # MIR Eval stats
    note_array_pred = pianoroll_to_notearray(pianoroll,
                                             time_div=framerate,
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


def calc_time(preds, conf):
    final_preds = []
    for pred, times in preds:
        final_preds.append((times, pred))
    return final_preds
