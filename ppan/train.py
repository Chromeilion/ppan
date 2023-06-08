from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video.mvit import MViT_V2_S_Weights
from tqdm import tqdm

from ppan.config import device, seq_len
from ppan.dataset import Rach3Dataset, VID_BACKEND, worker_init_fn
from ppan.model import PPAnModel
from ppan.types import PathLike
from ppan.midi import compute_piano_img


def main(dataset_dir: PathLike,
         output: Optional[PathLike] = None,
         tensorboard: Optional[bool] = None,
         saved_model: Optional[str] = None,
         **kwargs):
    if tensorboard is None:
        tensorboard = False
    if output is None:
        output = "./best_model.tar"
    if VID_BACKEND == "cuda":
        # CUDA does not support the default fork method
        torch.multiprocessing.set_start_method("spawn")

    # TODO: Dont hardcode hyperparamaters
    """Dataset Config"""
    batch_size = 3
    epochs = 20
    prefetch_factor = 1
    num_workers = 1
    clip_len = 16
    sample_rate = 30

    """Optimizer Config"""
    lr = 0.01

    """Model Config"""
    n_layers = 12

    # Tensorboard setup
    rec_rate, writer = None, None
    if tensorboard:
        rec_rate = 5  # Only record a value every x batches
        writer = SummaryWriter()

    dataset = Rach3Dataset(
        root=dataset_dir,
        video_transform=MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        clip_len=clip_len,
        sample_rate=sample_rate
    )
    weights = MViT_V2_S_Weights.KINETICS400_V1

    tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool),
                          diagonal=1).to(device)

    model = PPAnModel(encoder_weights=weights,
                      n_tokens=dataset.vocab_len,
                      seq_len=seq_len,
                      tgt_mask=tgt_mask,
                      emb_dim=393,
                      n_layers=n_layers)
    model.to(device)

    if num_workers > 1:
        worker_init_func = worker_init_fn
    else:
        worker_init_func = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        worker_init_fn=worker_init_func
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    best_loss = torch.inf
    global_step = 0

    if saved_model is not None:
        checkpoint = torch.load(saved_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']
        global_step = checkpoint['step']

    model.train()
    for epoch in tqdm(range(epochs), position=0):
        for batch in tqdm(loader, position=1):
            images = batch["video"].to(device)
            target = batch["target"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            pred = model(img=images, tgt=target, tgt_pad_mask=padding_mask)
            pred = torch.swapaxes(pred, 1, 2)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():  # Taking no chances regarding gradients
                if tensorboard:
                    if global_step % rec_rate == 0:
                        writer.add_scalar(tag="loss/train",
                                          scalar_value=loss.item(),
                                          global_step=global_step)
                    if global_step % (4*rec_rate) == 0:
                        video = batch["video"].cpu().detach()
                        video = torch.swapaxes(video, 1, 2)
                        writer.add_video(
                            tag="batch/train/video",
                            vid_tensor=video,
                            global_step=global_step
                        )
                        target_midi = dataset.tokens_to_midi(target)
                        pred_midi = dataset.tokens_to_midi(
                            torch.argmax(pred, dim=1)
                        )
                        target_img = compute_piano_img(target_midi)
                        pred_img = compute_piano_img(pred_midi)
                        midline = torch.ones(size=(batch_size, 1, 3,
                                                   pred_img.shape[3]))
                        comp_img = torch.cat([target_img, midline,  pred_img],
                                             2)

                        writer.add_images(tag="batch/train/target-pred",
                                          img_tensor=comp_img,
                                          global_step=global_step)

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'step': global_step
                            }, output
                        )
                    global_step += 1

    if tensorboard:
        writer.add_hparams({'lr': lr, 'bsize': batch_size,
                            'clen': clip_len, 'srate': sample_rate},
                           {'hparam/loss': best_loss})
        # Make sure all events have been written to disk
        writer.flush()
