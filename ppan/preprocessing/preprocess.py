from pathlib import Path
from typing import Union, Optional

import pandas as pd
from tqdm import tqdm

from ppan.dataset import load_data
from ppan.preprocessing import midi
from ppan.types import PathLike


def preprocess(samples, output: Union[str, Path]):
    samples = [i[0] for i in samples]
    sentences = midi.extract_sentences(samples=samples)
    dataframe = pd.DataFrame(sentences, dtype="string",
                             columns=["sentence", "file_no"])
    dataframe.to_csv(output, index=False)


def move_samples(samples: list[PathLike], move_dir: Path):
    if not move_dir.exists():
        move_dir.mkdir()
    for i in tqdm(samples, desc=f"Moving samples to: {move_dir}"):
        for j in i:
            j = Path(j)
            new_path = move_dir.joinpath(j.name)
            j.rename(new_path)


def main(dataset_dir: PathLike, output_dir: PathLike,
         dataset_name: Optional[Union[str, list[str]]] = None, *_, **__):
    """
    Do preprocessing for the data. Consists of splitting into train and test,
    as well as preparing the MIDI data for pre-training.

    Parameters
    ----------
    dataset_dir : PathLike
        Directory to dataset where it's already split into train and test
        folders.
    output_dir : PathLike
    dataset_name : Optional[Union[PathLike, List[PathLike]]]
        By default preprocesses train and test. If you have some other name or
        only want to work on a single subset specify it here.
    """
    output_dir = Path(output_dir[0])
    if not output_dir.is_dir():
        raise AttributeError("The output directory must be a directory")
    if not output_dir.exists():
        output_dir.mkdir()
    if dataset_name is None:
        dataset_name = ["train", "test"]
    elif isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset_dir = Path(dataset_dir[0])
    samplesets = {}
    for i in dataset_name:
        path = dataset_dir.joinpath(i)
        samplesets[i] = load_data(path)

    for i in samplesets.items():
        preprocess_dir = output_dir.joinpath("preprocess")
        if not preprocess_dir.exists():
            preprocess_dir.mkdir()
        preprocess_out = preprocess_dir.joinpath(i[0]+".csv")
        preprocess(i[1], preprocess_out)
