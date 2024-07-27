import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import wandb
from transformers import Trainer
from transformers.integrations import WandbCallback
from PIL import Image

from ppan.config import num_labels


class WandbFinetunePredictionProgressCallback(WandbCallback):
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
        self.sample_dataset = next(iter(DataLoader(
            val_dataset,
            batch_size=num_samples,
            shuffle=True
        )))
        self.imgs = v2.functional.grayscale_to_rgb(
            self.sample_dataset["pixel_values"]
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
        if not self.videos_run:
            self._wandb.log(
                {"Model inputs": wandb.Video(self.imgs.cpu().numpy(),
                                             "Example Model Inputs",
                                             30)},
                step=state.global_step
            )
            self.videos_run = True

        if state.global_step % state.eval_steps * self.freq == 0:
            super().on_evaluate(args, state, control, **kwargs)
            model = kwargs["model"]
            preds = []
            with torch.no_grad():
                sample = {
                    k: v.to(device=model.device) for k, v in
                    self.sample_dataset.items()
                }
                preds.append(model(
                    pixel_values=sample["pixel_values"],
                    labels=sample["label_ids"],
                    return_dict=True)
                )
            all_logits = torch.cat([i["logits"].cpu() for i in preds])
            for i in range(all_logits.shape[0]):
                self.add_preds_image(
                    torch.squeeze(all_logits[i]),
                    target=torch.squeeze(self.labs[i]),
                    lab=f"True Labels vs Model Predictions {i}",
                    step=state.global_step
                )
