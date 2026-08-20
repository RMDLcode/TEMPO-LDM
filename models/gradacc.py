from typing import Optional
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler

class GradientAccumulation:
    """
    Implements gradient accumulation to simulate larger effective batch sizes.
    Supports gradient clipping for stability in mixed-precision training.
    """
    def __init__(
            self,
            actual_batch_size: int,
            expect_batch_size: int,
            loader_len: int,
            optimizer: Optimizer,
            grad_scaler: Optional[GradScaler] = None,
            clip_grad: Optional[float] = None,
            clip_mode: str = "norm",
    ) -> None:
        """
        Args:
            actual_batch_size: Mini-batch size used in training.
            expect_batch_size: Desired effective batch size (must be multiple of actual_batch_size).
            loader_len: Number of mini-batches in one epoch.
            optimizer: Optimizer to update.
            grad_scaler: GradScaler for mixed precision (optional).
            clip_grad: Max gradient norm (if clip_mode='norm') or max absolute value (if 'value').
            clip_mode: 'norm' or 'value'.
        """
        assert expect_batch_size % actual_batch_size == 0, \
            'expect_batch_size must be divisible by actual_batch_size'
        if clip_mode not in ("norm", "value"):
            raise ValueError("clip_mode must be 'norm' or 'value'")

        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.clip_grad = clip_grad
        self.clip_mode = clip_mode
        self.steps_until_update = expect_batch_size // actual_batch_size

    def _clip_gradients(self) -> None:
        """Apply gradient clipping to all parameter groups that have gradients."""
        if self.clip_grad is None:
            return
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        params_with_grad = [p for group in self.optimizer.param_groups
                            for p in group['params'] if p.grad is not None]
        if params_with_grad:
            if self.clip_mode == "norm":
                nn.utils.clip_grad_norm_(params_with_grad, self.clip_grad)
            else:
                nn.utils.clip_grad_value_(params_with_grad, self.clip_grad)

    def step(self, loss: Tensor, step: int) -> None:
        """
        Backward pass with gradient accumulation. Updates optimizer when
        accumulation steps are met or at the end of the loader.
        """
        # Scale loss for accumulation
        loss = loss / self.steps_until_update

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Check if we should update weights
        if (step + 1) % self.steps_until_update == 0 or (step + 1) == self.loader_len:
            self._clip_gradients()
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)