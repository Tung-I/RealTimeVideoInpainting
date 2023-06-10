import math
from pathlib import Path

class Monitor:
    """The class to monitor the training process and save the model checkpoints.
    Args:
        checkpoints_dir: The root directory of the saved model checkpoints.
        mode: The mode of the monitor ('max' or 'min').
        target: The target of the monitor ('Loss', 'MyLoss' or 'MyMetric').
        saved_freq: The saved frequency.
        early_stop: The number of epochs to early stop the training if monitor target is not improved (default: 0, do not early stop the training).
    """
    def __init__(
        self,
        checkpoints_dir: Path,
        mode: str,
        target: str,
        saved_freq: int,
        early_stop: int = 0
    ):
        self.checkpoints_dir = checkpoints_dir
        self.mode = mode
        self.target = target
        self.saved_freq = saved_freq
        self.early_stop = math.inf if early_stop == 0 else early_stop
        self.best = -math.inf if self.mode == 'max' else math.inf
        self.not_improved_count = 0

        if not self.checkpoints_dir.is_dir():
            self.checkpoints_dir.mkdir(parents=True)


    def is_saved(self, epoch: int):
        if epoch % self.saved_freq == 0:
            return self.checkpoints_dir / f'model_{epoch}.pth'
        else:
            return None


    def is_best(self, valid_log: dict):
        score = valid_log[self.target]
        if self.mode == 'max' and score > self.best:
            self.best = score
            self.not_improved_count = 0
            return self.checkpoints_dir / 'model_best.pth'
        elif self.mode == 'min' and score < self.best:
            self.best = score
            self.not_improved_count = 0
            return self.checkpoints_dir / 'model_best.pth'
        else:
            self.not_improved_count += 1
            return None


    def is_early_stopped(self):
        return self.not_improved_count == self.early_stop