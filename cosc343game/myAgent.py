import numpy as np

playerName = "myAgent"
nPercepts = 75  #This is the number of percepts
nActions = 5    #This is the number of actionss

# Train against random for 5 generations, then against self for 1 generations
trainingSchedule = [("random", 5), ("self", 1)]

# This is the class for your creature/agent

class MyCreature:

    def __init__(self):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values

        # .
        # .
        # .
        pass #This is just a no-operation statement - replace it with your code


    def AgentFunction(self, percepts):

        actions = np.zeros((nActions))

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned and it must be a 5-dim numpy vector or a
        # list with 5 numbers.
        #
        # The index of the largest numbers in the 'actions' vector/list is the action taken
        # with the following interpretation:
        # 0 - move left
        # 1 - move up
        # 2 - move right
        # 3 - move down
        # 4 - eat
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # .
        # .
        # .

        return actions

def newGeneration(old_population):

    # This function should return a list of 'new_agents' that is of the same length as the
    # list of 'old_agents'.  That is, if previous game was played with N agents, the next game
    # should be played with N agents again.

    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boiler plate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, creature in enumerate(old_population):

        # creature is an instance of MyCreature that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the objects has attributes provided by the
        # game enginne:
        #
        # creature.alive - boolean, true if creature is alive at the end of the game
        # creature.turn - turn that the creature lived to (last turn if creature survived the entire game)
        # creature.size - size of the creature
        # creature.strawb_eats - how many strawberries the creature ate
        # creature.enemy_eats - how much energy creature gained from eating enemies
        # creature.squares_visited - how many different squares the creature visited
        # creature.bounces - how many times the creature bounced

        # .
        # .
        # .

        # This fitness functions just considers length of survival.  It's probably not a great fitness
        # function - you might want to use information from other stats as well
        fitness[n] = creature.turn

    # At this point you should sort the agent according to fitness and create new population
    new_population = list()
    for n in range(N):

        # Create new creature
        new_creature = MyCreature()

        # Here you should modify the new_creature's chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_creature.chromosome

        # Consider implementing elitism, mutation and various other
        # strategies for producing new creature.

        # .
        # .
        # .

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you neet to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
