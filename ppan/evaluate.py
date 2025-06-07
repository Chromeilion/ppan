import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union
import subprocess

from partitura import save_performance_midi
from partitura.utils import pianoroll_to_notearray
from partitura.performance import Performance, PerformedPart
import numpy as np
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch import no_grad
from tqdm import tqdm

from ppan.config import fps, device
from ppan.preprocessor import DatasetProcessor
from ppan.model import PPANModel, PPANVideoProcessor

PathLike = Union[str, bytes, os.PathLike]


def evaluate_on_dataset(samples, dataset_name, model_checkpoint, greyscale, batch_size,
                        gaussian_sigma, gaussian_sigma_frames, threshold_frame, threshold_onset,
                        frame_only: bool):
    preds_output = Path(dataset_name+"_preds.pkl")
    model = PPANModel.from_pretrained(
        model_checkpoint
    ).eval().to(device)
    processor = PPANVideoProcessor(
        resolution=model.config.image_size,
        grayscale=greyscale,
    )
    fr = 30
    if not preds_output.exists():
        chunk_size = batch_size # in seconds
        dataset = DatasetProcessor(
            datasets=samples,
            video_transform=processor,
            batch_size=1,
            temporal_size=chunk_size,
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

    chunk_size = 12/30
    dataset_sample = next(iter(DatasetProcessor(
        datasets=samples,
        video_transform=processor,
        batch_size=1,
        temporal_size=chunk_size,
        epoch_size=1,
        step=((chunk_size * fr) - model.config.num_frames),
        # a little overlap so that we don't miss any windows
        cachefile_name=f"{dataset_name}_cache.txt"
    )))

    for vid_path, preds in preds_rach3.items():
        video_len = float(subprocess.Popen(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {vid_path}",
            shell=True, stdout=subprocess.PIPE).stdout.read().decode())
        video_frame_rate = 30
        n_frames = round(video_len * video_frame_rate)
        model_half_window = model.config.num_frames // 2 + model.config.num_frames % 2
        preds = [i for i in preds if i[1] + model.config.num_frames < n_frames]
        preds = [(i[0], i[1]+model_half_window) for i in preds]
        final_pred = calc_time(preds, model.config)
        final_pred_onset = [(i[0], i[1][0].squeeze()) for i in final_pred]
        final_pred_frame = [(i[0], i[1][1].squeeze()) for i in final_pred]
        pianoroll = final_pred_to_onset_offset_array(
            final_pred_onset, final_pred_frame, threshold_frame, threshold_onset,
            gaussian_sigma, gaussian_sigma_frames, frame_only
        )
        pianoroll = pianoroll.astype(int) * 100

        vid_path = Path(vid_path)
        midi_output = Path("./midi")
        midi_output.mkdir(exist_ok=True)
        mid_output = midi_output/(vid_path.stem + ".mid")
        pkl_output = midi_output/(vid_path.stem + ".pkl")
        with open(pkl_output, "wb") as f:
            pickle.dump(pianoroll, f)

            save_to_midi(pianoroll, str(mid_output))

    return pianoroll, dataset_sample


def final_pred_to_onset_offset_array(final_pred, final_pred_frame, threshold_frame,
                                     threshold_onset, sigma, sigma_frames,
                                     frames_only: bool = False) -> np.ndarray:
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

    if os.getenv("PPAN_FRAMES_ONLY") == "1" or frames_only:
        return pred_array_frame_mask.T

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


def calc_time(preds, conf):
    final_preds = []
    for pred, times in preds:
        final_preds.append((times, pred))
    return final_preds
