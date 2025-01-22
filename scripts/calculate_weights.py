import os
import argparse as ap
from pathlib import Path

import tqdm
import torch
from torch.utils.data import DataLoader
from ppan.dataset import PPANDataset, DatasetConfig, DefaultVideoProcessor


def main(dataset_dir: str, *_, **__):
    conf = DatasetConfig(
        DefaultVideoProcessor(),
        output_map={"vid": "vid", "frames": "frames", "onsets": "onsets"},
        stride=1,
        window_size=1,
        lenience=1,
        shared_dict=None,
        max_samples=None
    )
    ds = PPANDataset(root=Path(dataset_dir), config=conf)
    dataloader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    n_batch = len(dataloader)
    mean_p_onset = torch.zeros(88)
    mean_p_frame = torch.zeros(88)
    mean_n_onset = torch.zeros(88)
    mean_n_frame = torch.zeros(88)
    for batch in tqdm.tqdm(dataloader):
        mean_p_onset += batch["onsets"].sum(dim=0) / n_batch
        mean_p_frame += batch["frames"].sum(dim=0) / n_batch
        mean_n_onset += (batch["onsets"] < 0.5).float().sum(dim=0) / n_batch
        mean_n_frame += (batch["frames"] < 0.5).float().sum(dim=0) / n_batch

    print(f"Mean positive onsets: {mean_p_onset}\n"
          f"Mean positive frames: {mean_p_frame}\n"
          f"Mean negative onsets: {mean_n_onset}\n"
          f"Mean negative frames: {mean_n_frame}")

    onset_ratio = mean_n_onset / mean_p_onset
    frame_ratio = mean_n_frame / mean_p_frame
    print(f"onset ratio: {onset_ratio}\n"
          f"frame ratio: {frame_ratio}\n")
    exit()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", required=True)
    args = parser.parse_args()
    main(**vars(args))

