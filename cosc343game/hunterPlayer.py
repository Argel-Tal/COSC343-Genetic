import numpy as np

playerName = "hunterPlayer"
trainingSchedule = None

nPercepts = 75
nActions = 5


class MyCreature:

    def __init__(self):
        # This agent doesn't evolve, and it doesn't have a chromosome.
        self.actionmap = [ [(0,1), 0    , 0, 0    , (0,3)],
                           [1    , (0,1), 0, (0,3), 3    ],
                           [1    , 1    , 0, 3    , 3    ],
                           [1    , (1,2), 2, (2,3), 3    ],
                           [(1,2), 2    , 2, 2    , (2,3)]
                        ]
        self.actioncoordinates = [ [-1,0], [0,1], [0, 1], [0,-1]]

    def AgentFunction(self, percepts):
        # This agent has complicated logic for implementation of various behaviours.  However
        # these behaviours are fixed - there is no chromosome changing the importance of combining
        # the behaviours.  Your agent's behaviours should be controlled by 'self.chromosome' (as well
        # as 'percepts') and evolve over time.  Your agents should not do better than randomPlayer
        # on game 1, but it should play well by game 500.

        # This agent creates a 5-dim vector of zeros for actions
        actions = np.zeros((nActions))

        # Percepts are a 5x5x3 tensor, where 5x5 is the region around the creatures
        # that it sees, and 3 corresponds to different maps.  Here's how to
        # extract different maps
        creature_map = percepts[:,:,0]  # 5x5 map with information about creatures and their size
        food_map = percepts[:,:,1]      # 5x5 map with information about strawberries
        wall_map = percepts[:,:,2]      # 5x5 map with information about wallss

        my_size = creature_map[2,2]

        # This agent's implementation of a FIXED simple run away behaviour
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue

                if np.abs(creature_map[2 + i, 2 + j]) > my_size:
                    amap = self.actionmap[2+i][2+j]

                    if type(amap)==tuple:
                        for a in amap:
                            areverse = (a+2)%4
                            s = self.actioncoordinates[areverse]
                            if creature_map[2 + s[0], 2 + s[1]]==0 and wall_map[2 + s[0], 2 + s[1]]!=1:
                                actions[areverse] = 1
                                return actions
                    else:
                        amapreverse = (amap+2)%4

                        s = self.actioncoordinates[amapreverse]
                        if wall_map[2 + s[0], 2 + s[1]] != 1:
                            actions[amapreverse] = 1
                            return actions

        # This agent's implementation of FIXED eat behaviour - remember, you should not code your agent
        # to always eat - this will make it behave less than random on game 1.
        # This behaviour must be learned in your agent, so it should only be invoked for certain values
        # of the chromosome not for the others.  Think of it this way - if the creatures lived in a world
        # where strawberries were poisonous, they would need to learn not to eat food; coding
        # this behaviour without possibility of the creature learning not to eat would make it
        # impossible for it to adapt to its environment.
        if food_map[2,2]==1:
            actions[4]=1
            return actions

        # This agent's FIXED go towards food behaviour
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue

                if food_map[2 + i, 2 + j] == 1:
                    amap = self.actionmap[2+i][2+j]

                    if type(amap)==tuple:
                        for a in amap:
                            s = self.actioncoordinates[a]
                            if creature_map[2 + s[0], 2 + s[1]]==0:
                                actions[a] = 1
                                return actions
                    else:
                        s = self.actioncoordinates[amap]
                        if creature_map[2 + s[0], 2 + s[1]]==0:
                            actions[amap] = 1
                            return actions

        # This agent's FIXED chase behaviour
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue

                if creature_map[2 + i, 2 + j] < 0:
                    if np.abs(creature_map[2 + i, 2 + j]) < my_size:
                        amap = self.actionmap[2+i][2+j]

                        if type(amap)==tuple:
                            for a in amap:
                               s = self.actioncoordinates[a]
                               if creature_map[2 + s[0], 2 + s[1]]==0 and wall_map[2 + s[0], 2 + s[1]]!=1:
                                  actions[a] = 1
                                  return actions
                        else:
                            s = self.actioncoordinates[amap]
                            if wall_map[2 + s[0], 2 + s[1]] != 1:
                                actions[amap] = 1
                                return actions


        #If no other behaviours were invoked, this agent chooses random behaviour.
        actions = np.random.rand((nActions))

        return actions

def newGeneration(old_population):

    # This agent doesn't evolve - it just returns the old generation (your agent should
    # measure fitness based on stats provided, do parent selection based on fitness
    # and "breed" new agents with chromosomes that are combination of the parents

    N = len(old_population)
    fitness = np.zeros((N))

    # This loop iterates over agents in the old population
    for n, agent in enumerate(old_population):
        # This fitness functions just considers length of survival in turns.  It's probably not a
        # great fitness function - you might want to use information from other stats as well.
        fitness[n] = agent.turn

    avg_fitness = np.mean(fitness)

    # This function returns old_population - your agent should create new population
    return (old_population, avg_fitness)
