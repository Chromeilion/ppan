import json
import os
from pathlib import Path
from random import shuffle
from typing import Union
import subprocess
import shutil
from collections import defaultdict

import numpy as np
import h5py
import torch
import tqdm
from dotenv import load_dotenv
from torchvision.io import encode_jpeg
from torchvision.transforms.v2.functional import resize
from ultralytics import YOLO
from rach3datautils.utils.multimedia import MultimediaTools

from ppan.config import processed_temporal_size, SAMPLE_TYPE, frame_hd5_file
from ppan.preprocessor import DatasetProcessor
from ppan.utils import load_all_data

PathLike = Union[str, bytes, os.PathLike]


def process(rach3_dir: PathLike,
            pianoyt_dir: PathLike,
            miditest_dir: PathLike,
            output_dir: PathLike,
            *_, **__):
    if output_dir is None:
        raise AttributeError("The output directory is required when "
                             "processing the dataset.")
    output_dir = Path(output_dir)
    load_dotenv()
    test_dir = output_dir/"test"
    train_dir = output_dir/"train"
    output_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)

    test_samples, train_samples, _, _, _ = load_all_data(
        rach3_dir, pianoyt_dir, miditest_dir
    )
    # Remove the automatic crop resize
    for i in range(len(train_samples)):
        sample = list(train_samples[i])
        sample[5] = False
        train_samples[i] = tuple(sample)

    shuffle(test_samples)
    shuffle(train_samples)
    train_ds = DatasetProcessor(
        datasets=train_samples,
        temporal_size=processed_temporal_size,
        epoch_size=1,
        cachefile_name="./render_train_cache.txt",
        batch_size=1
    )
    test_ds = DatasetProcessor(
        datasets=test_samples,
        temporal_size=processed_temporal_size,
        epoch_size=1,
        cachefile_name="./render_test_cache.txt",
        batch_size=1
    )
    save_dataset(test_ds, test_dir)
    save_dataset(train_ds, train_dir)


def save_dataset(ds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    n_frames_tracker = defaultdict(lambda: 0)
    with h5py.File(out_dir/frame_hd5_file, "w") as f:
        for file_idx, i in enumerate(tqdm.tqdm(ds, desc="Processing dataset")):
            # Each individual video gets put into its own directory
            group_name = Path(i['sample'][2]).stem
            if group_name not in f:
                f.create_group(group_name)
            group = f[group_name]

            # Remove batch dimension
            vid = torch.squeeze(i["pixel_values"], dim=0)

            mid_path = out_dir/f"{group_name}.midi"
            if not Path(mid_path).exists():
                shutil.copyfile(i["sample"][0], mid_path)

           # Ensure we have an even value for height and width
            if vid.shape[-1] % 2 != 0 or vid.shape[-2] % 2 != 0:
                w = vid.shape[-1] - (vid.shape[-1] % 2)
                h = vid.shape[-2] - (vid.shape[-2] % 2)
                vid = resize(vid, [h, w])

            frame_dataset = "frames"
            if frame_dataset not in group:
                n_frames = int(subprocess.run(
                    f"ffprobe -v error -select_streams v:0 -count_packets "
                    f"-show_entries stream=nb_read_packets -of csv=p=0 "
                    f"{i['sample'][2]}".split(" "), capture_output=True).stdout.decode())
                dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
                group.create_dataset(frame_dataset, (n_frames,), dtype=dtype)

            for frame_no, frame in enumerate(range(vid.shape[0])):
                if i['times'][0].item()+frame_no >= group[frame_dataset].shape[0]:
                    continue
                n_frames_tracker[group_name] += 1
                frame = vid[frame]
                group[frame_dataset][i['times'][0].item()+frame_no] = encode_jpeg(frame.cpu()).numpy()

            details_path = out_dir/f"{group_name}_video_details.txt"
            if not details_path.exists():
                vid_duration = MultimediaTools().get_decoded_duration(i["sample"][2])
                with open(details_path, "w") as f_d:
                    f_d.write(f"duration: {vid_duration}")

        for vid in f:
            if f[vid]["frames"].shape[0] != n_frames_tracker[vid]:
                raise AttributeError(f"Detected {f[vid]['frames'].shape[0]} frames but only saved {n_frames_tracker[vid]} frames for {vid}.")


def calculate_bounding_boxes(samples, yolo_model_checkpoint) -> list[SAMPLE_TYPE]:
    """
    Get bounding box predictions for a list of samples.
    Utilizes the Ultralytics package.

    Parameters
    ----------
    samples : list[ppan.dataset.SAMPLE_TYPE]
    yolo_model_checkpoint : PathLike

    Returns
    -------
    samples : ppan.dataset.SAMPLE_TYPE
        The same samples as passed in but with bounding boxes added.
    """
    # Load the model
    model = YOLO(yolo_model_checkpoint)

    new_samples = []
    for sample in tqdm.tqdm(samples, desc="Calculating Bounding Boxes"):
        sample_preds = []
        video = sample[2]
        pred = model.predict(source=str(video), stream=True,
                             verbose=False, vid_stride=5)
        # Get predictions over the first 10 seconds.
        [sample_preds.append(next(pred)) for _ in range(5*10)]

        filtered_session_preds = [
            i for i in sample_preds if i.boxes.conf.shape[0] > 0
        ]
        best_pred = max(filtered_session_preds, key=lambda x: x.boxes.conf[0])
        bb_meta = json.loads(best_pred.tojson())[0]['box']
        bb = (round(bb_meta["y1"]), round(bb_meta["y2"]),
              round(bb_meta["x1"]), round(bb_meta["x2"]))
        new_sample = [i for i in sample]
        new_sample[3] = bb
        new_samples.append(new_sample)
    return new_samples
