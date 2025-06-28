import numpy as np
import random
from src.tools.logger_utils import LoggerUtils

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class MCMCSampler:
    def __init__(self, mutators, mode):
        logger.info(f"Using {self.__class__.__name__} as selection strategy!")
        self.p = 1 / len(mutators)
        self.mutators = mutators
        self.mode = mode

    def selected_mutator(self, mu1=None):
        if self.mode == "random" or mu1 is None:
            return self.mutators[np.random.randint(0, len(self.mutators))]
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self.mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self.mutators[k2]
            return mu2

    def sort_mutators(self):
        import random
        random.shuffle(self.mutators)
        self.mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self.mutators):
            if mu.mutator_name == mutator_name:
                return i
        return -1
