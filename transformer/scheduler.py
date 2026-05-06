"""Noam learning rate schedule from Vaswani et al., 2017.

lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

This linearly increases the lr for the first warmup_steps training steps,
then decreases it proportionally to the inverse square root of the step number.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_schedule(step: int, d_model: int, warmup_steps: int) -> float:
    """Compute Noam lr scale factor for a given step.

    Args:
        step: Current training step (1-indexed).
        d_model: Model dimension.
        warmup_steps: Number of warmup steps.

    Returns:
        Scale factor to multiply the base learning rate by.
    """
    step = max(step, 1)
    scale = (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return scale


def get_noam_scheduler(optimizer: Optimizer, d_model: int, warmup_steps: int) -> LambdaLR:
    """Build a LambdaLR scheduler implementing the Noam schedule.

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = get_noam_scheduler(optimizer, d_model=512, warmup_steps=4000)
        # After each step:
        optimizer.step()
        scheduler.step()

    Args:
        optimizer: Adam optimizer. Set base lr=1.0; the schedule provides the actual lr.
        d_model: Model dimension used in the scale formula.
        warmup_steps: Number of steps over which lr increases linearly.

    Returns:
        LambdaLR scheduler.
    """
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_schedule(step, d_model, warmup_steps),
    )
