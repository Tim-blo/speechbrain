import math
import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)

@checkpoints.register_checkpoint_hooks
class PARPFineTuningScheduler:

    def __init__(self, lr_initial, lr_peak, n_steps, decay_factor):
        assert lr_peak >= lr_initial
        assert decay_factor > 1
        self.lr_initial = lr_initial
        self.lr_peak = lr_peak
        self.n_warmup_steps = int(0.1*n_steps)
        self.n_constant_steps = int(0.4*n_steps)
        self.n_decay_steps = n_steps - self.n_warmup_steps - self.n_constant_steps
        self.current_lr = lr_initial
        self.decay_factor = decay_factor
        self.losses = []
        self.n_steps = 0

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self._get_lr_value()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return lr

    def _get_lr_value(self):
        n_steps, n_warmup_steps, lr_initial, lr_peak = self.n_steps, self.n_warmup_steps, self.lr_initial, self.lr_peak
        if n_steps < n_warmup_steps:
            return lr_initial + n_steps * (lr_peak - lr_initial) / n_warmup_steps
        n_constant_steps = self.n_constant_steps
        if n_steps < self.n_warmup_steps + n_constant_steps:
            return lr_peak
        n_steps_after_decay = n_steps - n_warmup_steps - n_constant_steps
        return lr_initial + self.decay_factor**(-n_steps_after_decay) * (lr_peak - lr_initial)

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]
