import os
import sys
import random
sys.path.append(os.getcwd())

from src.sampler.mcmc_sampler import MCMCSampler

class Mutator:
    def __init__(self, mutator_name, score):
        self.mutator_name = mutator_name
        self.score = score


if __name__ == "__main__":
    from collections import namedtuple
    # Mutator = namedtuple('Mutator', ['mutator_name',"score"])
    mutators = [Mutator("api",0), Mutator("graph",0), Mutator("llm",0), Mutator("variable",0)]
    mcmc_sampler = MCMCSampler(mutators)

    last_mu = None
    for i in range(100):
        mutator = mcmc_sampler.selected_mutator(last_mu)
        # randomly generate the reward in [0,1)
        reward = random.random()
        print(f"Selected mutator: {mutator.mutator_name}. Reward: {reward}")
        mutator.score = reward

    # print the final score
    for mutator in mutators:
        print(f"Mutator: {mutator.mutator_name}. Score: {mutator.score}")
