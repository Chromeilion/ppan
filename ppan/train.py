from typing import Optional

from ppan.dataset import Rach3Dataset, worker_init_fn
from ppan.types import PathLike
from ppan.model import PPAnModel
from ppan.config import device, seq_len

import torch
from torch.utils.data import DataLoader
from torchvision.models.video.mvit import MViT_V2_S_Weights
from tqdm import tqdm


def main(dataset_dir: PathLike, output: Optional[PathLike] = None,
         epochs: Optional[int] = None,
         **kwargs):
    if epochs is None:
        epochs = 5

    torch.manual_seed(42)

    dataset = Rach3Dataset(
        root=dataset_dir,
        video_transform=MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        clip_len=16,
        sample_rate=30
    )
    weights = MViT_V2_S_Weights.KINETICS400_V1

    tgt_mask = torch.triu(torch.ones((seq_len, seq_len),
                                      dtype=torch.bool),
                                      diagonal=1).to(device)

    model = PPAnModel(encoder_weights=weights,
                      n_tokens=dataset.vocab_len,
                      seq_len=seq_len,
                      tgt_mask=tgt_mask,
                      emb_dim=393)
    model.to(device)

    loader = DataLoader(dataset, batch_size=4, num_workers=3, pin_memory=True,
                        prefetch_factor=1, worker_init_fn=worker_init_fn)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for batch in tqdm(loader):
            images = batch["video"].to(device)
            target = batch["target"].to(device)
            pred = model(images, target)
            pred = torch.swapaxes(pred, 1, 2)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())
        print(epoch)