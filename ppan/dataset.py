from ppan.types import PathLike

from typing import List, Tuple

from rach3datautils.utils.dataset import DatasetUtils
from rach3datautils.utils.session import Session
from rach3datautils.utils.multimedia import MultimediaTools
from torch.utils.data import Dataset


class Rach3Dataset(Dataset):
    """
    Dataset object for training on the Rach3 dataset. Handles loading and
    preprocessing.
    """
    def __init__(self, path: PathLike):
        self.path: PathLike = path
        self.dataset = DatasetUtils(self.path)
        self.sessions = self.dataset.remove_noncomplete(
            subsession_list=self.dataset.get_sessions(),
            required=["midi.splits_list", "flac.splits_list",
                      "video.splits_list"]
        )
        # [midi_path, flac_path, video_path]
        self.splits: List[Tuple[PathLike, PathLike, PathLike]] = []
        i: Session
        for i in self.sessions:
            [self.splits.append(j) for j in zip(i.midi.splits_list,
                                                i.flac.splits_list,
                                                i.video.splits_list)]

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, item):
        item = self.splits[item]
        # TODO: add file loading according to these specifications:
        # https://pytorch.org/vision/main/models/generated/torchvision.models.video.mvit_v2_s.html#torchvision.models.video.mvit_v2_s
        video = MultimediaTools.load_video(filepath=item[2],
                                           height=1080,
                                           width=1920)

