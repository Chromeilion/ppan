import os
import random
from pathlib import Path
from typing import Optional, Union
from multiprocessing import Manager

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEConfig,
    TrainingArguments,
    Trainer
)
from transformers.integrations import WandbCallback

from ppan.config import get_model_size
from ppan.config import (seed, num_labels, model_resolution,
                         model_crop_resolution, max_eval_steps,
                         finetune_default, class_weights)
from ppan.dataset import PPANDataset, BaseVideoProcessor, get_samples

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
    if pretrained_checkpoint is None:
        raise AttributeError("A pretrained checkpoint is required for model "
                             "finetuning.")
    if no_epochs is None:
        no_epochs = finetune_default["no_epochs"]
    if batch_size is None:
        batch_size = finetune_default["batch_size"]
    if learning_rate is None:
        learning_rate = finetune_default["learning_rate"]
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

    load_dotenv()
    dataset_dir = Path(dataset_dir)
    # How many frames from the center we want in each clip
    pad = int(get_model_size()[0]["num_frames"] // 2)
    processor = FinetuneProcessor(randaug=randaug,
                                  spatial_jitter=spatial_jitter)
    output_map = {"vid": "pixel_values", "lab": "label_ids"}
    manager = Manager()
    shared_dict_train = manager.dict()
    shared_dict_test = manager.dict()
    train_ds = PPANDataset(
        pad=pad,
        video_processor=processor,
        samples=get_samples(dataset_dir/"train"),
        output_map=output_map,
        shared_dict=shared_dict_train,
        temporal_jitter=temporal_jitter
    )
    test_ds = PPANDataset(
        pad=pad,
        video_processor=FinetuneProcessor(),
        samples=random.sample(get_samples(dataset_dir/"test"), max_eval_steps),
        output_map=output_map,
        shared_dict=shared_dict_test
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_checkpoint,
        problem_type="regression",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        attention_probs_dropout_prob=finetune_default["attention_probs_dropout_prob"],
        hidden_dropout_prob=finetune_default["hidden_dropout_prob"]
    ).train()

    training_arguments = TrainingArguments(
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
        ddp_find_unused_parameters=False,
        save_total_limit=4,
        max_grad_norm=finetune_default["grad_clip"]
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=finetune_default["momentum"],
        weight_decay=weight_decay,
    )
    trainer = FinetuneTrainer(
        weight=torch.log2(torch.tensor(class_weights)),
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        label_smoothing=label_smoothing,
        optimizers=(optimizer, None)
    )
    progress_callback = WandbFinetunePredictionProgressCallback(
        trainer=trainer,
        train_dataset=train_ds,
        val_dataset=test_ds
    )
    trainer.add_callback(progress_callback)

    trainer.train(resume_from_checkpoint=checkpoint_dir)


class FinetuneProcessor(BaseVideoProcessor):
    """
    Video processor for PPAN finetuning
    """
    def __init__(self, randaug: Optional[bool] = None,
                 spatial_jitter: Optional[bool] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if spatial_jitter is None:
            spatial_jitter = False
        if randaug is None:
            randaug = False

        augs = []
        if spatial_jitter:
            augs.append(v2.ScaleJitter(
                target_size=(model_crop_resolution[1], model_crop_resolution[0]),
                scale_range=(0.85, 1.1)
            ))
            augs.append(v2.RandomCrop(size=model_crop_resolution,
                                      pad_if_needed=True))
        else:
            augs.append(v2.CenterCrop(size=model_crop_resolution))

        if randaug:
            augs.append(v2.RandAugment(
                num_ops=2, magnitude=5
            ))
        augs.append(v2.Resize(model_resolution))

        self.augmentations = v2.Compose(augs)

    def process_video(self, vid):
        return self.augmentations(vid)


class FinetuneTrainer(Trainer):
    """Huggingface trainer override adding BCE loss and dynamic class weights
    to the training loop.
    """
    def __init__(self, weight: Optional[torch.tensor] = None,
                 label_smoothing: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if label_smoothing is None:
            label_smoothing = 0

        # Create gaussian kernels
        self.gaussian_kernel = torch.autograd.Variable(
            torch.FloatTensor(
                [[[0.01, 0.15, 1, 0.15, 0.01]]])
        )
        self.smoothing = label_smoothing
        self.confidence = 1 - self.smoothing
        self.do_smoothing = label_smoothing > 0.00001

        self.label_smoothing = label_smoothing

        if weight is not None:
            if isinstance(weight, float):
                weight = [weight]
            self.weight = torch.tensor(weight)
        else:
            self.weight = weight

    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Small override which generates and injects the patch masks into the
        input.
        """
        inputs["pixel_values"] = v2.functional.to_dtype(
            inputs["pixel_values"], torch.float32, scale=True
        )
        return super(FinetuneTrainer, self).training_step(model, inputs)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[
               torch.Tensor]]:
        inputs["pixel_values"] = v2.functional.to_dtype(
            inputs["pixel_values"], torch.float32, scale=True
        )
        return super(FinetuneTrainer, self).prediction_step(
            model, inputs, prediction_loss_only, ignore_keys)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        n_positive = torch.sum(labels) / 2

        # Apply label smoothing
        if self.do_smoothing:
            with torch.no_grad():
                pos_labels = labels > 0.00001
                neg_labels = labels < 0.00001
                labels = torch.squeeze(torch.nn.functional.conv1d(
                    labels[:, None, :], self.gaussian_kernel.to(labels.device),
                    padding=2
                ), dim=1)
                labels[neg_labels] += self.smoothing / labels.shape[1]
                labels[labels > 1] = 1

                labels[pos_labels] *= self.confidence
                n_positive *= self.confidence

        # compute loss using BCE
        if self.weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.weight.to(logits.device))
        else:
            if n_positive < 1:
                weight = torch.tensor(88.).to(logits.device)
            else:
                weight = (torch.numel(labels)-n_positive) / n_positive
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=weight
            )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
        self.train_imgs = v2.functional.grayscale_to_rgb(
            sample_train_dataset["pixel_values"]
        )
        self.sample_dataset["pixel_values"] = v2.functional.to_dtype(
            self.sample_dataset["pixel_values"], torch.float32, scale=True
        )
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
                        return_dict=True
                    )
                logits = preds["logits"].cpu()
                for i in range(logits.shape[0]):
                    self.add_preds_image(
                        torch.squeeze(logits[i]),
                        target=torch.squeeze(self.labs[i]),
                        lab=f"True Labels vs Model Predictions {i}",
                        step=state.global_step
                    )
