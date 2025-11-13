# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional.regression import spearman_corrcoef


def masked_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked MSE loss between input and target."""
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the negative log-likelihood of Bernoulli distribution."""
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.LongTensor,
) -> torch.Tensor:
    """Compute the masked relative error between input and target."""
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


class MaskedMseMetric(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_mse",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mask",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        mask = mask.bool()
        non_masked_preds = preds[mask]
        non_masked_target = target[mask]

        self.sum_mse += torch.nn.functional.mse_loss(
            non_masked_preds,
            non_masked_target,
            reduction="sum",
        )
        self.sum_mask += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum_mse / self.sum_mask


class MaskedSpearmanMetric(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_spearman",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_examples",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        for pred_i, target_i, mask_i in zip(preds, target, mask):
            non_mask_preds = pred_i[mask_i].to("cpu")
            non_mask_targets = target_i[mask_i].to("cpu")
            self.sum_spearman += spearman_corrcoef(non_mask_preds, non_mask_targets)
            self.num_examples += 1

    def compute(self) -> torch.Tensor:
        return self.sum_spearman / self.num_examples
