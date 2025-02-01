import os
from pathlib import Path
from typing import Optional, Union
from multiprocessing import Manager

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
from timm.data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from ppan.config import seed, num_labels, finetune_default
from ppan.dataset import PPANDataset, DatasetConfig, OutputMap
from ppan.model import (PPANModel, PPANConfig, PPANVideoProcessor, PPANCollate,
                        FRAME_WEIGHTS, ONSET_WEIGHTS)


PathLike = Union[str, bytes, os.PathLike]


def train(dataset_dir: PathLike,
          pretrained_checkpoint: PathLike,
          output_dir: PathLike,
          no_epochs: Optional[int] = None,
          eval_every: Optional[int] = None,
          save_every: Optional[int] = None,
          batch_size: Optional[int] = None,
          learning_rate: Optional[float] = None,
          weight_decay: Optional[float] = None,
          warmup_ratio: Optional[float] = None,
          scheduler_type: Optional[str] = None,
          checkpoint_dir: Optional[PathLike] = None,
          optimizer: Optional[str] = None,
          adam_beta1: Optional[float] = None,
          adam_beta2: Optional[float] = None,
          momentum: Optional[float] = None,
          label_smoothing_conf_onset: Optional[float] = None,
          label_smoothing_conf_frame: Optional[float] = None,
          spatial_jitter: Optional[bool] = None,
          gaussian_noise: Optional[bool] = None,
          color_jitter: Optional[bool] = None,
          rand_erase: Optional[bool] = None,
          grayscale: Optional[bool] = None,
          drop_path: Optional[float] = None,
          dropout: Optional[float] = None,
          rand_rotate: Optional[bool] = None,
          frames_only: Optional[bool] = None,
          onsets_only: Optional[bool] = None,
          class_weights: Optional[tuple[list[float], list[float]]] = None,
          model_architecture: Optional[str] = None,
          window_size: Optional[int] = None,
          image_size: Optional[int] = None,
          *_, **__):
    # The handling of default values is not done well right now, however it
    # works and we can change it later
    if dataset_dir is None:
        raise AttributeError("The dataset directory is required for "
                             "model training.")
    if output_dir is None:
        raise AttributeError("The output directory is required for "
                             "model training.")
    if window_size is None:
        window_size = finetune_default["window_size"]
    if image_size is None:
        image_size = finetune_default["image_size"]
    if model_architecture is None:
        model_architecture = finetune_default["architecture"]
    if class_weights is None:
        class_weights = [ONSET_WEIGHTS, FRAME_WEIGHTS]
    if class_weights[0] == "fancy":
        class_weights = [ONSET_WEIGHTS, FRAME_WEIGHTS]
    if frames_only is None:
        frames_only = finetune_default["frames_only"]
    if onsets_only is None:
        onsets_only = finetune_default["onsets_only"]
    if dropout is None:
        dropout = finetune_default["dropout"]
    if drop_path is None:
        drop_path = finetune_default["drop_path"]
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
    if spatial_jitter is None:
        spatial_jitter = finetune_default["spatial_jitter"]
    if grayscale is None:
        grayscale = finetune_default["grayscale"]
    if rand_erase is None:
        rand_erase = finetune_default["rand_erase"]
    if gaussian_noise is None:
        gaussian_noise = finetune_default["gaussian_noise"]
    if color_jitter is None:
        color_jitter = finetune_default["color_jitter"]
    if rand_rotate is None:
        rand_rotate = finetune_default["rand_rotate"]
    if label_smoothing_conf_frame is None:
        label_smoothing_conf_frame = finetune_default["confidence_frame"]
    if label_smoothing_conf_onset is None:
        label_smoothing_conf_onset = finetune_default["confidence_onset"]
    if optimizer is None:
        optimizer = finetune_default["optimizer"]
    if momentum is None:
        momentum = finetune_default["momentum"]

    load_dotenv()
    print(f"Found {os.cpu_count()} CPUs.")
    # Flag for disabling the dataloader cache on machines with little ram.
    dataloader_cache = os.environ.get("PPAN_DATALOADER_CACHE", None)
    if dataloader_cache is None:
        dataloader_cache = True
    else:
        dataloader_cache = False

    dataset_dir = Path(dataset_dir)
    processor = PPANVideoProcessor(grayscale=grayscale,
                                   rand_rotate=rand_rotate,
                                   spatial_jitter=spatial_jitter,
                                   rand_erase=rand_erase,
                                   gaussian_noise=gaussian_noise,
                                   color_jitter=color_jitter,
                                   resolution=image_size)

    # Remap the default dataset output dictionary keys to what our model expects
    output_map: OutputMap
    if onsets_only and not frames_only:
        output_map = {"vid": "pixel_values",
                      "onsets": "onsets",
                      "frames": None}
    elif frames_only and not onsets_only:
        output_map = {"vid": "pixel_values",
                      "onsets": None,
                      "frames": "frames"}
    else:
        output_map = {"vid": "pixel_values",
                      "onsets": "onsets",
                      "frames": "frames"}

    # In order to avoid redundant copies of frames being cached, we
    # create a shared dictionary that can be used as a cache by all
    # dataset workers.
    shared_dict_train, shared_dict_test = None, None
    if dataloader_cache:
        manager = Manager()
        shared_dict_train = manager.dict()
        shared_dict_test = manager.dict()

    config = PPANConfig(
        pretrained_encoder=model_architecture,
        confidence_frame=label_smoothing_conf_frame,
        confidence_onset=label_smoothing_conf_onset,
        dropout=dropout,
        drop_path=drop_path,
        bce_loss_weight_onset=class_weights[0],
        bce_loss_weight_frame=class_weights[1],
        num_frames=window_size,
        image_size=image_size
    )
    train_ds_config = DatasetConfig(
        video_processor=processor,
        output_map=output_map,
        stride=finetune_default["stride"],
        window_size=window_size,
        lenience=finetune_default["lenience"],
        shared_dict=shared_dict_train
    )
    collate_fn = PPANCollate(config)
    if pretrained_checkpoint is not None:
        model = PPANModel(config).from_pretrained(pretrained_checkpoint).train()
    else:
        model = PPANModel(config).train()

    train_ds = PPANDataset(
        config=train_ds_config,
        root=dataset_dir/"train",
    )
    test_ds_config = DatasetConfig(
        video_processor=processor,
        output_map=output_map,
        stride=finetune_default["stride"],
        window_size=window_size,
        lenience=finetune_default["lenience"],
        shared_dict=shared_dict_test,
        max_samples=1024
    )
    val_ds = PPANDataset(
        root=dataset_dir/"test",
        config=test_ds_config
    )
    if optimizer == "adamw":
        if learning_rate is None:
            learning_rate = finetune_default["lr_adamw"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=weight_decay
        )
    elif optimizer == "sgd":
        if learning_rate is None:
            learning_rate = finetune_default["lr_sgd"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    training_arguments = TrainingArguments(
        ddp_find_unused_parameters=True,
        num_train_epochs=no_epochs,
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=eval_every,
        logging_steps=eval_every,
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        save_safetensors=False, # Required for the CNN
        dataloader_pin_memory=True,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        seed=seed,
        save_steps=save_every,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=["wandb"],
        dataloader_num_workers=int(os.environ.get("PPAN_DATASET_NO_WORKERS", 12)),
        dataloader_prefetch_factor=1,
        log_on_each_node=False,
        save_total_limit=4,
        max_grad_norm=finetune_default["grad_clip"],
        label_names=[i for i in output_map.values() if i is not None]
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        optimizers=(optimizer, None),
        data_collator=collate_fn
    )
    progress_callback = WandbFinetunePredictionProgressCallback(
        trainer=trainer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_collate_fn=collate_fn
    )
    trainer.add_callback(progress_callback)
    trainer.train(resume_from_checkpoint=checkpoint_dir)


class WandbFinetunePredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows us to visualize the
    model predictions as the training progresses.
    """

    def __init__(self, trainer, val_dataset, train_dataset,
                 train_collate_fn=None, test_collate_fn=None,
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
            shuffle=True,
            collate_fn=test_collate_fn
        )))
        sample_train_dataset = next(iter(DataLoader(
            train_dataset,
            batch_size=num_samples,
            shuffle=True,
            collate_fn=train_collate_fn
        )))
        self.train_imgs = sample_train_dataset["pixel_values"].to("cpu")
        # In order to visualize the images we unnormalize them.
        unnormalize = v2.Compose([
            v2.Normalize(
                mean=-torch.tensor(IMAGENET_DEFAULT_MEAN) / torch.tensor(IMAGENET_DEFAULT_STD),
                std=1 / torch.tensor(IMAGENET_DEFAULT_STD)
            ),
            v2.ToDtype(torch.uint8, scale=True)
        ])
        self.train_imgs = unnormalize(sample_train_dataset["pixel_values"])
        self.onsets, self.frames = None, None
        if "onsets" in self.sample_dataset:
            self.onsets = self.sample_dataset["onsets"][:, :, None].to("cpu")
        if "frames" in self.sample_dataset:
            self.frames = self.sample_dataset["frames"][:, :, None].to("cpu")
        self.videos_run = False
        self.freq = freq

    def add_preds_image(self, logits: torch.Tensor, target: torch.Tensor,
                        lab: str, step: int):
        if len(target.shape) == 1:
            target = target[:, None]

        img_t = v2.functional.resize(target[None, ...],
                                     [num_labels, num_labels // 2],
                                     interpolation=v2.InterpolationMode.NEAREST)
        img_p = v2.functional.resize(logits[None, ...],
                                     [num_labels, num_labels // 2],
                                     interpolation=v2.InterpolationMode.NEAREST)
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
                    preds = model(**sample)
                logits = preds["logits"].cpu().permute(0, 2, 1)
                for i in range(logits.shape[0]):
                    to_cat = [j[i] for j in [self.onsets, self.frames] if j is not None]
                    self.add_preds_image(
                        torch.squeeze(logits[i]),
                        target=torch.squeeze(torch.cat(to_cat, dim=1)),
                        lab=f"True vs Pred (Onsets Then Frames) {i}",
                        step=state.global_step
                    )
