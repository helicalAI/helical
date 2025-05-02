import torch
import torch.nn.functional as F


def check_nan_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def log_expm1(x):
    """Compute log(exp(x) - 1) in a numerically stable way."""
    # For small x, use log(expm1(x))
    # For large x, use x
    return torch.where(x <= 5, torch.log1p(torch.expm1(x)), x)


def compute_log_norm(x, softplus_approx=True):
    if softplus_approx:
        return torch.clamp(x - F.softplus(-x), min=0.0)
    else:
        return torch.clamp(log_expm1(x), min=0.0)


def logit_softcap(x, softcap, hardcap=1e6):
    """
    Applies a soft cap to logits using tanh scaling.

    Args:
        x (torch.Tensor): Input tensor
        softcap (float): Cap value. If 0 or negative, returns input unchanged

    Returns
    -------
        torch.Tensor: Soft-capped tensor
    """
    if not softcap > 0:
        return x

    # Prevent potential overflow in division
    x = torch.clamp(x, min=-hardcap, max=hardcap)

    # Scale, apply tanh, and rescale
    scaled = x / softcap
    capped = torch.tanh(scaled)
    return softcap * capped


class ZTP_NLL(torch.nn.Module):
    def __init__(
        self,
        eps: float = 1e-5,
        softplus_approx: bool = True,
        max_counts: float = 1e10,
    ):
        self.eps = eps
        self.softplus_approx = softplus_approx
        self.max_counts = max_counts
        super().__init__()

    def sample(self, mu: torch.Tensor, mask: torch.Tensor = None):
        """
        Sample from Zero-Truncated Poisson distribution.

        Args:
        mu (torch.Tensor): The rate parameter of the Poisson distribution.
        mask (torch.Tensor, optional): A boolean mask to indicate which elements to sample.

        Returns
        -------
        torch.Tensor: Samples from the Zero-Truncated Poisson distribution.
        """
        # Ensure mu is positive
        mu = torch.clamp(mu, min=self.eps)

        # Initialize the sample tensor
        sample = torch.zeros_like(mu)

        # Compute the probability of zero for a regular Poisson
        p_zero = torch.exp(-mu)

        # Sample from uniform
        u = torch.distributions.Uniform(p_zero, 1).sample()

        # undo the exponential
        t = -torch.log(u)

        # Sample from Poisson + 1 with adjusted rate parameter
        sample = 1 + torch.poisson(mu - t)

        # Apply mask if provided
        if mask is not None:
            sample = sample.masked_fill(mask, float("nan"))

        return sample

    def forward(
        self,
        mu: torch.Tensor,
        input_counts: torch.Tensor,
        mask: torch.Tensor = None,
        eval_mode: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(mu, torch.Tensor), "mu must be a torch.Tensor"
        assert isinstance(
            input_counts, torch.Tensor
        ), "input_counts must be a torch.Tensor"

        # Clamp input_counts between 1 and max_counts
        counts_clamped = torch.clamp(input_counts, min=self.eps, max=self.max_counts)

        nll = -counts_clamped * torch.log(mu + self.eps)

        log_norm = compute_log_norm(mu + self.eps, self.softplus_approx)
        nll += log_norm
        nll += torch.lgamma(counts_clamped + 1)

        if eval_mode:
            # Only average the likelihoods along genes, not along cells
            if mask is not None:
                nll = nll.masked_fill(mask, 0)
                return torch.Tensor(
                    [nll[i, ~mask[i]].mean() for i in range(nll.size(0))]
                )
            else:
                return nll.mean(dim=1)
        else:
            if mask is not None:
                nll = nll.masked_fill(mask, 0)
                nll = nll.sum() / (~mask).sum()
            else:
                nll = nll.mean()

        return nll


class CrossEntropyLoss(torch.nn.Module):
    def __init__(
        self, shift_right: bool = False, softcap: int = 0, reduction: str = "mean"
    ):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.shift_right = shift_right
        self.softcap = softcap
        self.reduction = reduction

    def forward(self, logits, input_ids, mask=None, **kwargs):
        # Ensure input_ids are of type LongTensor
        input_ids = input_ids.long()

        if self.shift_right:
            input_ids = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            if mask is not None:
                mask = mask[:, 1:]

        if self.softcap > 0:
            logits = logit_softcap(logits, self.softcap)

        if mask is not None:
            # Set masked positions in input_ids to ignore_index (-100)
            input_ids = input_ids.masked_fill(mask, -100)

        # Reshape logits to (batch_size, num_classes, seq_len)
        logits = logits.permute(0, 2, 1)

        # Compute the loss
        loss = self.ce(logits, input_ids)

        # Return the mean loss over non-ignored elements
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def sample(self, logits, mask=None):
        """
        Sample from the predicted distribution.

        Args:
            logits (torch.Tensor): The raw model output of shape (batch_size, seq_len, vocab_size)
            mask (torch.Tensor, optional): A boolean mask indicating which elements to sample.
                                        True indicates masked positions. Defaults to None.

        Returns
        -------
            torch.Tensor: Sampled token indices of shape (batch_size, seq_len)
        """
        # Apply mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample from the probability distribution
        samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)

        # Reshape back to (batch_size, seq_len)
        samples = samples.view(logits.size(0), logits.size(1))

        return samples
