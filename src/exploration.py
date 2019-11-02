import numpy as np
from catalyst.rl.core import ExplorationStrategy


class ContinuousActionBinarization(ExplorationStrategy):
    """
    For continuous environments only.
    """

    def __init__(
        self,
        threshold: float = 0,
        upper: float = 1.0,
        lower: float = -1.0
    ):
        super().__init__()

        self.threshold = threshold
        self.upper = upper
        self.lower = lower

    def set_power(self, value):
        super().set_power(value)

    def get_action(self, action):
        action = np.where(
            action > self.threshold,
            self.upper,
            self.lower
        )
        return action
