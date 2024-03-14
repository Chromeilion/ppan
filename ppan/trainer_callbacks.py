from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
import wandb
from transformers import (Trainer, TrainerCallback, TrainingArguments,
                          TrainerState, TrainerControl)
from transformers.integrations import WandbCallback

from ppan.config import num_labels


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.
    """

    def __init__(self, trainer, val_dataset,
                 num_samples=3, freq=1):
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
        iterator = iter(list(val_dataset))
        iterator_zero = next(iterator)
        self.sample_dataset = [next(iterator) for _ in range(num_samples*2)]
        # move all samples to the same gpu
        self.sample_dataset = [
            {key: val.to(iterator_zero[key].device)
             for key, val in i.items()} for i in self.sample_dataset]
        img_mean = torch.tensor(val_dataset.video_transform.image_mean)
        img_std = torch.tensor(val_dataset.video_transform.image_std)
        self.unnormalize = v2.Compose([
            v2.Normalize(
                mean=-img_mean / img_std,
                std=1 / img_std
            ),
            v2.ToDtype(torch.uint8, scale=True)
        ])
        self.imgs = self.unnormalize(torch.cat(
            [i["pixel_values"] for i in self.sample_dataset],
            dim=0
        ))
        self.labs = torch.cat(
            [i["labels"] for i in self.sample_dataset],
            dim=0
        )
        self.freq = freq
        self.videos_run = False
        self.preds_table = None

    def add_preds_image(self, logits: torch.Tensor, target: torch.Tensor,
                        lab: str, step: int):
        img_t = v2.functional.resize(target[None, :, None],
                                     [num_labels, num_labels // 2])
        img_p = v2.functional.resize(logits[None, :, None],
                                     [num_labels, num_labels // 2])
        img_f = torch.cat((img_t, img_p), dim=2)
        self._wandb.log(
            {lab: wandb.Image(img_f)},
            step=step
        )

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if not self.videos_run:
            self._wandb.log(
                {"Model inputs": wandb.Video(self.imgs.cpu().numpy(),
                                             "Example Model Inputs",
                                             30)},
                step=state.global_step
            )
            self.videos_run = True

        if state.global_step % state.eval_steps * self.freq == 0:
            model = kwargs["model"]
            preds = []
            with torch.no_grad():
                for sample in self.sample_dataset:
                    preds.append(model(**sample, return_dict=True))
            all_logits = torch.cat([i["logits"] for i in preds])
            for i in range(all_logits.shape[0]):
                self.add_preds_image(
                    torch.squeeze(all_logits[i]),
                    target=torch.squeeze(self.labs[i]),
                    lab=f"True Labels vs Model Predictions {i}",
                    step=state.global_step
                )


class SaveCallback(TrainerCallback):
    DATASET_SAVE_NAME = "dataset.json"

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save. For saving the dataloader state.
        """
        save_path = (Path(args.output_dir)/f"checkpoint-{state.global_step}/")
        kwargs["train_dataloader"].dataset.save(save_path)
