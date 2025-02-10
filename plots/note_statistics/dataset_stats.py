import os
from typing import Union, List, Tuple, Optional
from pathlib import Path
import csv
import json
from rach3datautils.utils.dataset import DatasetUtils
from rach3datautils.utils.multimedia import MultimediaTools
import partitura as pt
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


PathLike = Union[str, bytes, os.PathLike]

# [[midi_path, flac_path, video_path, bounding_box, whether to rotate 180,
#   random_resize]]
SAMPLE_TYPE = List[
    Tuple[PathLike, Optional[PathLike], PathLike,
          Optional[tuple[int, int, int, int]], bool, bool, PathLike]
]
TEST_TRAIN_SPLIT = tuple[list[SAMPLE_TYPE], list[SAMPLE_TYPE]]

plt.style.use('seaborn-v0_8')
plt.rc('text', usetex=True)
plt.rc('text.latex')


@dataclass
class Sample:
    midi_path: PathLike | None
    video_path: PathLike
    dataset: str
    bounding_box: Optional[tuple[int, int, int, int]] = None
    flac_path: Optional[PathLike] = None
    rotation_factor: float = 0
    rotate_180: bool = False
    note_intervals: PathLike = None


def load_rach3(root: PathLike, ds_prefix = "r3s"):
    """
    Load the dataset for use with PPAn.

    Parameters
    ----------
    root : PathLike
    ds_prefix : ds_prefixes

    Returns
    -------
    test : List[Tuple[PathLike, PathLike, PathLike]]
    train : List[Tuple[PathLike, PathLike, PathLike]]
    """
    root = Path(root)
    test, train = root / "test", root / "train"
    bbs_path = root / f"{ds_prefix}_bounding_boxes.json"

    with open(bbs_path, "r") as f:
        meta = json.load(f)
    bbs = {"_".join(key.split("_")[:-1]): value for key, value in meta.items()}
    rot = {"_".join(key.split("_")[:-1]): value["rot_angle"] for key, value in meta.items()}

    return (load_rach3_split(test, bbs, rot, False, ds_prefix),
            load_rach3_split(train, bbs, rot, True, ds_prefix))


def load_rach3_split(root: PathLike,
                     bbs: dict,
                     rot: dict,
                     augment: bool,
                     ds_prefix) -> list[Sample]:
    """
    Load a folder containing Rach3 fi        crops = [bb for _ in range(len(i.midi.splits_list))]
        rots = [r for _ in range(len(i.midi.splits_list))]les (such as test or train folders)

    Parameters
    ----------
    root : PathLike
    bbs : dict
    rot : dict
    augment : bool
    ds_prefix : ds_prefixes

    Returns
    -------
    samples : list[Sample]
    """
    dataset = DatasetUtils(root)
    sessions = dataset.remove_noncomplete(
        subsession_list=dataset.get_sessions(),
        required=["midi.splits_list", "flac.splits_list",
                  "video.splits_list"]
    )
    samples: list[Sample] = []
    for i in sessions:
        bb_meta = bbs[str(i.id)]
        bb = (int(bb_meta["y1"]), int(bb_meta["y2"]),
              int(bb_meta["x1"]), int(bb_meta["x2"]))
        r = float(rot[str(i.id)])
        for mid, flac, video in zip(i.midi.splits_list,
                                    i.flac.splits_list,
                                    i.video.splits_list):
            sample = Sample(
                midi_path=mid,
                flac_path=flac,
                video_path=video,
                bounding_box=bb,
                dataset=ds_prefix,
                rotation_factor=r,
                rotate_180=False
            )
            samples.append(sample)
    return samples


def load_pianoyt(root: PathLike) -> tuple[list[Sample], list[Sample]]:
    data = []
    with open(os.path.join(root, "dataset.csv"), "r") as f:
        reader = csv.reader(f)
        [data.append(i) for i in reader]

    samples_train = []
    samples_test = []
    for i in data:
        video = os.path.join(root, f'processed_videos/{Path(i[0]).name}')
        video = video.replace(" ", "_")
        crop = [int(j) for j in i[4:]]
        crop = (crop[0], crop[1], crop[2], crop[3])
        tup = [os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
               None, video, crop, True, True]
        sample = Sample(
            midi_path=os.path.join(root, f'pianoyt_MIDI/audio_{i[1]}.0.midi'),
            video_path=video,
            bounding_box=crop,
            rotate_180=True,
            dataset="pianoyt"
        )
        if i[3] == "1":
            samples_train.append(sample)
        elif i[3] == "3":
            tup[-1] = False
            samples_test.append(sample)

    return samples_test, samples_train


datasets = {
    "R3s": load_rach3("/home/chromeilion/Code/Uni/uni2023w/rach3_continued/datasets/rach3/rach3_s/", "r3s"),
    "R3x": load_rach3("/home/chromeilion/Code/Uni/uni2023w/rach3_continued/datasets/rach3/rach3_x/", "r3x"),
    "PianoYT": load_pianoyt("/mnt/hdd1/external_test_data/PianoYT/")
}
fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(7, 2.2))
for col, (dataset, (test, train)) in enumerate(datasets.items()):
    splits = {
        "test": test,
        "train": train
    }
    total_dataset_lens = []
    no_samples = 0
    for split, samples in splits.items():

        all_notes_vec = np.zeros(88, dtype=int)
        total_split_video_time = 0
        mmt = MultimediaTools()
        for sample in samples:
            no_samples += 1
            perf = pt.load_performance_midi(sample.midi_path)
            for note in perf.performedparts[0].notes:
                all_notes_vec[note.pnote_dict['midi_pitch']-21] += 1
            total_split_video_time += mmt.get_len(sample.video_path)
        all_notes_norm = all_notes_vec/max(all_notes_vec)
        ax[col].bar(
            range(all_notes_vec.shape[0]),
            all_notes_norm,
            width=1,
            label=split,
            alpha=0.5
        )
        print(f"Total number of notes in {dataset} {split} split: ", sum(all_notes_vec), "\n",
              "Total len of all videos in split: ", total_split_video_time, "\n")
        total_dataset_lens.append(total_split_video_time)
    print(f"Average {dataset} video length: {sum(total_dataset_lens)/no_samples}")

    ax[col].get_yaxis().set_visible(False)
    ax[col].set_title(dataset)
    ax[col].set_xlabel("Piano Keys")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys())
fig.savefig('dataset_distributions.png', dpi=400)
