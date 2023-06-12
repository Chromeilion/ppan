from typing import Optional
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video.mvit import MViT_V2_S_Weights
from tqdm import tqdm

from ppan.config import device, seq_len
from ppan.dataset import Rach3Dataset, VID_BACKEND, worker_init_fn, \
    load_data, split_data
from ppan.midi import compute_piano_img
from ppan.model import PPAnModel
from ppan.types import PathLike


def main(dataset_dir: PathLike,
         output: Optional[PathLike] = None,
         tensorboard: Optional[bool] = None,
         saved_model: Optional[str] = None,
         *args, **kwargs):
    """
    Training function for PPAn.

    Parameters
    ----------
    dataset_dir : PathLike
    output : Optional[PathLike]
        Where to output best model file
    tensorboard : bool
        Whether to save statistics to Tensorboard
    saved_model : PathLike
        Location of an already trained model to continue training from

    Returns
    -------
    None
    """
    if tensorboard is None:
        tensorboard = False
    if output is None:
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        output = "./checkpoints/best_model.tar"
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
    max_samples_in_eval = 20

    """Optimizer Config"""
    lr = 0.01

    """Model Config"""
    n_decoder_layers = 8

    """Tensorboard Config"""
    eval_every = 80
    writer = None
    if tensorboard:
        writer = SummaryWriter()

    dataset = load_data(dataset_dir)
    train, test = split_data(dataset, 0.2)
    train = Rach3Dataset(
        samples=train,
        video_transform=MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        clip_len=clip_len,
        sample_rate=sample_rate,
        seq_len=seq_len
    )
    test = Rach3Dataset(
        samples=test,
        video_transform=MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        clip_len=clip_len,
        sample_rate=sample_rate,
        seq_len=seq_len
    )
    weights = MViT_V2_S_Weights.KINETICS400_V1
    vocab_size = train.vocab_len

    model = PPAnModel(encoder_weights=weights,
                      n_tokens=vocab_size,
                      seq_len=seq_len,
                      emb_dim=768,
                      n_decoder_layers=n_decoder_layers)
    model.to(device)

    if num_workers > 1:
        worker_init_func = worker_init_fn
    else:
        worker_init_func = None

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        worker_init_fn=worker_init_func
    )
    test_loader = DataLoader(
        test,
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
        for batch in tqdm(train_loader, position=1):
            # Basic training loop
            images = batch["video"].to(device)
            target = batch["target"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            pred = model(img=images, tgt=target, tgt_pad_mask=padding_mask)
            pred = torch.swapaxes(pred, 1, 2)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % eval_every == 0:
                # Evaluation, saving results, etc.
                model.eval()
                with torch.no_grad():
                    # Short evaluation on test set
                    avg_test_loss = 0
                    for s_num, test_batch in enumerate(test_loader):
                        images_t = test_batch["video"].to(device)
                        target_t = test_batch["target"].to(device)
                        padding_mask_t = batch["padding_mask"].to(device)

                        pred_t = model(img=images_t, tgt=target_t,
                                       tgt_pad_mask=padding_mask_t)
                        pred_t = torch.swapaxes(pred_t, 1, 2)
                        loss_test = loss_fn(pred_t, target_t)
                        avg_test_loss = (avg_test_loss * s_num +
                                         loss_test.item()) / (s_num + 1)

                        if s_num >= max_samples_in_eval:
                            break

                    # Save model if the loss improved on test set
                    if avg_test_loss < best_loss:
                        best_loss = avg_test_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_test,
                            'step': global_step
                            }, output
                        )

                    if tensorboard:
                        write_tensorboard_stats(
                            batch, writer, loss, global_step, avg_test_loss,
                            test, target_t, pred_t, batch_size
                        )

                model.train()
            global_step += 1

    if tensorboard:
        # Add final results for hyperparamater comparison
        writer.add_hparams({'lr': lr, 'bsize': batch_size,
                            'clen': clip_len, 'srate': sample_rate},
                           {'hparam/loss': best_loss})
        # Make sure all events have been written to disk
        writer.flush()


def write_tensorboard_stats(batch, writer, loss, global_step, avg_test_loss,
                            dataset, target_t, pred_t, batch_size):
    writer.add_scalar(tag="loss/train",
                      scalar_value=loss.item(),
                      global_step=global_step)
    writer.add_scalar(tag="loss/test",
                      scalar_value=avg_test_loss,
                      global_step=global_step)

    video = batch["video"].cpu().detach()
    video = torch.swapaxes(video, 1, 2)
    writer.add_video(
        tag="batch/train/video",
        vid_tensor=video,
        global_step=global_step
    )
    target_midi = dataset.tokenizer.tokens_to_midi(
        target_t
    )
    pred_midi = dataset.tokenizer.tokens_to_midi(
        torch.argmax(pred_t, dim=1)
    )
    target_img = compute_piano_img(target_midi)
    pred_img = compute_piano_img(pred_midi)
    midline = torch.ones(size=(batch_size, 1, 3,
                               pred_img.shape[3]))
    comp_img = torch.cat([target_img, midline,
                          pred_img],
                         2)

    writer.add_images(tag="batch/test/target-pred",
                      img_tensor=comp_img,
                      global_step=global_step)
