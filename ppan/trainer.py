from typing import Optional

from torch import nn
from torch import tensor
from transformers import Trainer


class PPAnTrainer(Trainer):
    """Huggingface trainer override so that we can use BCE loss."""
    def __init__(self, weight: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weight is not None:
            self.weight = tensor([weight])
        else:
            self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute loss using BCE
        if self.weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.weight.to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
