import numpy as np

playerName = "randomPlayer"
trainingSchedule = None

nPercepts = 75
nActions = 5

class MyCreature:

    def __init__(self):
        # This agent doesn't evolve, and it doesn't have a chromosome.
        # The pass statement is for no-op (replace it with your chromosome initialisation)
        pass

    def AgentFunction(self, percepts):
        # This agent creates a 5-dim vector of random numbers for action.  It
        # ignores percepts and performs random action.  Your agents should not
        # perform random actions - the actions should be deterministic from
        # computation based on self.chromosome and percepts
        actions = np.random.rand((nActions))
        return actions

    def trainingSchedule(self):
        return None


def newGeneration(old_population):

    # This agent doesn't evolve - it just returns the old generation (your agent should
    # measure fitness based on stats provided, do parent selection based on fitness
    # and "breed" new agents with chromosomes that are combination of the parents)

    N = len(old_population)
    fitness = np.zeros((N))

    # This loop iterates over agents in the old population
    for n, agent in enumerate(old_population):
        # This fitness functions just considers length of survival.  It's probably not a great fitness
        # function - you might want to use information from other stats as well
        fitness[n] = agent.turn

    avg_fitness = np.mean(fitness)

    # This function returns old_population - your agent should create new population
    return (old_population, avg_fitness)
