import os
from pathlib import Path
from typing import Optional, Union
from multiprocessing import Manager
from random import choices, shuffle

import torch
import torchvision.transforms.v2 as v2
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer
)
from transformers.integrations import WandbCallback

from ppan.config import (seed, num_labels,
                         finetune_default, IMAGENET_STD, IMAGENET_MEAN)
from ppan.dataset import (PPANDataset, get_samples)
from ppan.model import PPANModel, PPANConfig, PPANVideoProcessor

PathLike = Union[str, bytes, os.PathLike]


def train(dataset_dir: PathLike,
          pretrained_checkpoint: PathLike,
          output_dir: PathLike,
          no_epochs: Optional[int] = None,
          eval_every: Optional[int] = None,
          save_every: Optional[int] = None,
          encoder_frozen: Optional[bool] = None,
          batch_size: Optional[int] = None,
          learning_rate: Optional[float] = None,
          weight_decay: Optional[float] = None,
          warmup_ratio: Optional[float] = None,
          scheduler_type: Optional[str] = None,
          checkpoint_dir: Optional[PathLike] = None,
          adam_beta1: Optional[float] = None,
          adam_beta2: Optional[float] = None,
          label_smoothing: Optional[float] = None,
          randaug: Optional[bool] = None,
          temporal_jitter: Optional[bool] = None,
          spatial_jitter: Optional[bool] = None,
          *_, **__):
    if dataset_dir is None:
        raise AttributeError("The dataset directory is required for "
                             "model training.")
    if output_dir is None:
        raise AttributeError("The output directory is required for "
                             "model training.")
    if encoder_frozen is None:
        encoder_frozen = True
    if adam_beta1 is None:
        adam_beta1 = finetune_default["adam_beta1"]
    if adam_beta2 is None:
        adam_beta2 = finetune_default["adam_beta2"]
    if no_epochs is None:
        no_epochs = finetune_default["no_epochs"]
    if batch_size is None:
        batch_size = finetune_default["batch_size"]
    if weight_decay is None:
        weight_decay = finetune_default["weight_decay"]
    if warmup_ratio is None:
        warmup_ratio = finetune_default["warmup_ratio"]
    if scheduler_type is None:
        scheduler_type = finetune_default["scheduler_type"]
    if eval_every is None:
        eval_every = finetune_default["eval_every"]
    if save_every is None:
        save_every = finetune_default["save_every"]
    if randaug is None:
        randaug = finetune_default["randaug"]
    if temporal_jitter is None:
        temporal_jitter = finetune_default["temporal_jitter"]
    if spatial_jitter is None:
        spatial_jitter = finetune_default["spatial_jitter"]
    rotate_180 = finetune_default["rotate_180"]
    rand_erase = finetune_default["rand_erase"]
    mask_percentage = finetune_default["mask_percentage"]

    load_dotenv()
    dataset_dir = Path(dataset_dir)
    processor = PPANVideoProcessor(randaug=randaug,
                                   spatial_jitter=spatial_jitter,
                                   rotate_180=rotate_180,
                                   rand_erase=rand_erase)

    # Remap the default dataset output dictionary keys to what our model expects
    output_map = {"vid": "pixel_values", "lab": "label_ids", "mask": None}

    train_samples: list = get_samples(dataset_dir/"train")
    # Going through the entire val set while training is too time-consuming.
    validation_samples = choices(get_samples(dataset_dir/"test"), k=1000)

    # In order to avoid redundant copies of frames being cached, we
    # create a shared dictionary that can be used as a cache by all
    # dataset workers.
    manager = Manager()
    shared_dict_train = manager.dict()
    shared_dict_test = manager.dict()
    model = PPANModel(PPANConfig()).train()
    train_ds = PPANDataset(
        video_processor=processor,
        samples=train_samples,
        output_map=output_map,
        shared_dict=shared_dict_train,
        temporal_jitter=temporal_jitter,
        mask_percentage=mask_percentage,
        tubelet_size=(model.pretrained_model.config.tubelet_size,
                      model.pretrained_model.config.patch_size,
                      model.pretrained_model.config.patch_size)
    )
    val_ds = PPANDataset(
        video_processor=PPANVideoProcessor(),
        samples=validation_samples,
        output_map=output_map,
        shared_dict=shared_dict_test
    )

    training_arguments = TrainingArguments(
        ddp_find_unused_parameters=False,
        num_train_epochs=no_epochs,
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        save_steps=save_every,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=["wandb"],
        dataloader_num_workers=os.cpu_count() // 4 - 1,
        log_on_each_node=False,
        save_total_limit=4,
        max_grad_norm=finetune_default["grad_clip"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_default["lr_adamw"],
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        optimizers=(optimizer, None)
    )
    progress_callback = WandbFinetunePredictionProgressCallback(
        trainer=trainer,
        train_dataset=train_ds,
        val_dataset=val_ds
    )
    trainer.add_callback(progress_callback)
    trainer.train(resume_from_checkpoint=checkpoint_dir)


class WandbFinetunePredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.
    """

    def __init__(self, trainer, val_dataset, train_dataset,
                 num_samples=20, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

    Parameters:
        trainer : Trainer
            The Hugging Face Trainer instance.
        val_dataset : Dataset
            The validation dataset for generating predictions.
        num_samples : int, optional
            Number of samples to select from
            the validation dataset for generating predictions. Defaults to 3.
        freq : int, optional
            Frequency of logging. Defaults to 1.
        """
        super().__init__()
        self.trainer: Trainer = trainer
        self.sample_dataset = next(iter(DataLoader(
            val_dataset,
            batch_size=num_samples,
            shuffle=True
        )))
        sample_train_dataset = next(iter(DataLoader(
            train_dataset,
            batch_size=num_samples,
            shuffle=True
        )))
        # In order to visualize the images we unnormalize them.
        unnormalize = v2.Compose([
            v2.Normalize(
               mean=-torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD),
               std=1/torch.tensor(IMAGENET_STD)
            ),
            v2.ToDtype(torch.uint8, scale=True)
        ])
        self.train_imgs = unnormalize(sample_train_dataset["pixel_values"])
        self.labs = self.sample_dataset["label_ids"]
        self.videos_run = False
        self.freq = freq

    def add_preds_image(self, logits: torch.Tensor, target: torch.Tensor,
                        lab: str, step: int):
        img_t = v2.functional.resize(target[None, :, None],
                                     [num_labels, num_labels // 4])
        img_p = v2.functional.resize(logits[None, :, None],
                                     [num_labels, num_labels // 4])
        img_f = torch.cat((img_t, img_p), dim=2)
        self._wandb.log(
            {lab: wandb.Image(img_f)},
            step=step
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer.state.is_world_process_zero:
            super().on_evaluate(args, state, control, **kwargs)
            if not self.videos_run:
                self._wandb.log(
                    {"Example model training inputs":
                         wandb.Video(self.train_imgs.cpu().numpy(),
                                     "Example model training inputs",
                                     30)},
                    step=state.global_step
                )
                self.videos_run = True

            if state.global_step % state.eval_steps * self.freq == 0:
                model = kwargs["model"]
                with torch.no_grad():
                    sample = {
                        k: v.to(device=model.device) for k, v in
                        self.sample_dataset.items()
                    }
                    preds = model(
                        pixel_values=sample["pixel_values"],
                        labels=sample["label_ids"],
                    )
                logits = preds["logits"].cpu()
                for i in range(logits.shape[0]):
                    self.add_preds_image(
                        torch.squeeze(logits[i]),
                        target=torch.squeeze(self.labs[i]),
                        lab=f"True Labels vs Model Predictions {i}",
                        step=state.global_step
                    )
