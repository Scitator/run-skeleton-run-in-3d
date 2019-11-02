from catalyst.rl import registry

from src.env import SkeletonEnvWrapper
from src.exploration import ContinuousActionBinarization

registry.Environment(SkeletonEnvWrapper)
registry.Exploration(ContinuousActionBinarization)
