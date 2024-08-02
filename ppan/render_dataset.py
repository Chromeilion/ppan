import json
import os
from pathlib import Path
from random import shuffle
from typing import Union

import torch
import tqdm
from dotenv import load_dotenv
from torchvision.io import write_jpeg
from torchvision.transforms.v2.functional import resize
from ultralytics import YOLO

from ppan.config import processed_temporal_size, SAMPLE_TYPE
from ppan.preprocessor import PPAnDatasetProcessor
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
    train_samples = [
        (i, j, k, l, m, False, n) for i, j, k, l, m, n in train_samples
    ]
    shuffle(test_samples)
    shuffle(train_samples)
    train_ds = PPAnDatasetProcessor(
        datasets=train_samples,
        temporal_size=processed_temporal_size,
        epoch_size=1,
        cachefile_name="./render_train_cache.txt",
        batch_size=1
    )
    test_ds = PPAnDatasetProcessor(
        datasets=test_samples,
        temporal_size=processed_temporal_size,
        epoch_size=1,
        cachefile_name="./render_test_cache.txt",
        batch_size=1
    )
    save_dataset(train_ds, train_dir)
    save_dataset(test_ds, test_dir)


def save_dataset(ds, out_dir):
    for file_idx, i in enumerate(tqdm.tqdm(ds, desc="Processing dataset")):
        save_dir = out_dir/str(file_idx)
        save_dir.mkdir(exist_ok=True)
        vid = torch.squeeze(i["pixel_values"], dim=0)
        torch.save((i['labels'] > 0.1).cpu(), save_dir/"labels.pt")
        if vid.shape[-1] % 2 != 0 or vid.shape[-2] % 2 != 0:
            w = vid.shape[-1] - (vid.shape[-1] % 2)
            h = vid.shape[-2] - (vid.shape[-2] % 2)
            vid = resize(vid, [h, w])
#        vid = torch.permute(vid, (0, 2, 3, 1)).cpu()
        for frame_no, frame in enumerate(range(vid.shape[0])):
            frame = vid[frame]
            write_jpeg(frame.cpu(), save_dir/f"frame_{frame_no}.jpeg")
#        write_video(str(save_dir/"clip.mp4"), vid, fps=30)
        with open(save_dir/"sample_details.txt", "w") as f:
            f.write("\n".join([str(j) for j in i['sample']]))
            negative = torch.all(torch.squeeze(i['labels']) < 0.1).item()
            f.write(f"\nNegative sample: {negative}")


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
    for sample in tqdm(samples, desc="Calculating Bounding Boxes"):
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