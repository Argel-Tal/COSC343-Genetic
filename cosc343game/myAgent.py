import numpy as np
import random as random
import statistics as stats

playerName = "myAgent"
nPercepts = 75              # This is the number of percepts
nActions = 5                # This is the number of actionss
proportionRetained = 0.4    # the proportion of agents that survive into the next generation
fitnessOptionChoice = 6     # Selected fitness function, from list of options
fitnessScores = list()
chromoStdevs = list()

# Train against random for 5 generations, then against self for 1 generations
trainingSchedule = [("random", 40)]
# trainingSchedule = [("random", 50), ("self", 1)]

with open("stats.csv", "w") as myfile:
    myfile.write('')

# distance function
# arrrayIndex is the 2 part position of a detected thing
def manhattanDistance(arrayObject, translatedX, translatedY):
    distance = (abs(arrayObject[0] - (2-translatedX)) + abs(arrayObject[1] - (2-translatedY)))
    return distance

# This is the class for your creature/agent
class MyCreature:

    def __init__(self):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values
        self.weightAgentSize = random.randint(-100, 100)
        self.weightAgentDist = random.randint(-100, 100)
        self.weightAgentAttitude = random.randint(-100, 100)
        self.weightAgentSizeGivenAttitude = random.randint(-100, 100)  # allows for non-linear interaction
        self.weightFood = random.randint(-100, 100)
        self.weightWall = random.randint(-100, 100)
        self.weightConsume = random.randint(-100, 100)
        self.weightSizeDif = random.randint(-100, 100)
        self.weightSizeRelativeFood = random.randint(-100, 100)
        self.chromosome = [self.weightAgentSize, self.weightAgentDist, self.weightAgentAttitude, self.weightAgentSizeGivenAttitude, self.weightFood, self.weightWall, self.weightConsume, self.weightSizeDif, self.weightSizeRelativeFood]
        # .
        # .
        # .
        pass  # This is just a no-operation statement - replace it with your code


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

        nets = np.zeros((nActions))
        translations = [[-1,0],[0,-1],[1,0],[0,1]]
        mySize = percepts[0][2][2]

        for i in range(nActions-1):

            
            netOtherAgentsSizeAttitude = 0
            netOtherAgentsDistAttitude = 0
            netOtherAgentsAttitude = 0

            netRelativeSizesWithAttitude = 0
            netrelativeSizesDistanceAttitude = 0

            netFoods = 0
            netFoodRelativeSize = 0
            netWalls = 0
            currentDirection = translations[i]

            agentMap = percepts[:,:,0]
            foodMap = percepts[:,:,1]
            wallMap = percepts[:,:,2]
            mySize = percepts[2][2][0]

            #  agentsDetected = np.where(percepts[0:] != 0) # assuming the tensor is indexable by map mode first
            for agent, val in np.ndenumerate(agentMap):

                # allows us to use attitude with distance only calculations, as att is otherwise implicit within size
                if val > 0:
                    attVal = 1  # friendly
                else:
                    attVal = -1  # hostile

                # caring about entity size
                netOtherAgentsSizeAttitude += (self.weightAgentSize * val * self.weightAgentAttitude + (self.weightAgentSizeGivenAttitude * val))
                # caring about distance to entity 
                netOtherAgentsDistAttitude += self.weightAgentDist * manhattanDistance(agent, currentDirection[0], currentDirection[1]) * self.weightAgentAttitude * attVal
                # caring about both
                netOtherAgentsAttitude += (self.weightAgentSize * self.weightAgentAttitude * val) * (self.weightAgentDist * manhattanDistance(agent, currentDirection[0], currentDirection[1])) + ((self.weightAgentSizeGivenAttitude * val))
                # caring about both, with relative size considered, instead of just raw size
                netRelativeSizesWithAttitude += (self.weightSizeDif * (mySize - abs(val))) + (self.weightAgentAttitude * attVal) + (self.weightAgentSizeGivenAttitude * (mySize - abs(val)) * attVal)
                netrelativeSizesDistanceAttitude += (self.weightAgentSize * (mySize - abs(val))) * (self.weightAgentDist * manhattanDistance(agent, currentDirection[0], currentDirection[1])) * (self.weightAgentAttitude * attVal) + (self.weightAgentSizeGivenAttitude * (mySize - abs(val)) * attVal)

            for food, val in np.ndenumerate(foodMap):
                netFoods += self.weightFood * (manhattanDistance(food, currentDirection[0], currentDirection[1]))
                netFoodRelativeSize += self.weightSizeRelativeFood * (self.weightFood * (manhattanDistance(food, currentDirection[0], currentDirection[1])))

            for wall, val in np.ndenumerate(wallMap):
                netWalls += (self.weightWall * (manhattanDistance(wall, currentDirection[0], currentDirection[1])))

            nets[i] = netOtherAgentsSizeAttitude + netOtherAgentsDistAttitude + netOtherAgentsAttitude + netRelativeSizesWithAttitude + netrelativeSizesDistanceAttitude + netFoods + netFoodRelativeSize + netWalls

        if percepts[2, 2, 1] == 1:
            nets[4] = ((self.weightConsume**5)/((percepts[0][2][2])+1))  # +1 ensures divide by 0 never happens
        else:
            nets[4] = -1000000000  # prevents the AI from sitting still on spots which are empty


        maxArg = max(nets)
        minArg = min(nets)
        newMax = +1
        newMin = -1

        a = (newMax - newMin) / (maxArg - minArg)
        b = newMax - a * maxArg

        for i in range(len(nets)):
            nets[i] = a * nets[i] + b

        # set the value of 'actions' at the index corresponding to the index of 'nets' with the highest value, to 1
        actions[np.where(nets == max(nets))] = 1
        return actions


def newGeneration(old_population):

    # This function should return a list of 'new_agents' that is of the same length as the
    # list of 'old_agents'.  That is, if previous game was played with N agents, the next game
    # should be played with N agents again.

    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))
    genes = np.zeros(((N, 9)))


    # This loop iterates over your agents in the old population - the purpose of this boiler plate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, creature in enumerate(old_population):
        # creature is an instance of MyCreature that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the objects has attributes provided by the
        # game engine:
        #
        # creature.alive - boolean, true if creature is alive at the end of the game
        # creature.turn - turn that the creature lived to (last turn if creature survived the entire game)
        # creature.size - size of the creature
        # creature.strawb_eats - how many strawberries the creature ate
        # creature.enemy_eats - how much energy creature gained from eating enemies
        # creature.squares_visited - how many different squares the creature visited
        # creature.bounces - how many times the creature bounced
        
        # very basic creature evaluation function:
        fitnessEval0 = creature.turn + creature.size + creature.strawb_eats + creature.enemy_eats # baseline eval function
        fitnessEval1 = creature.alive * (creature.size + creature.strawb_eats + creature.enemy_eats) # score > 0 only when still alive, else score = 0
        fitnessEval2 = creature.turn + creature.strawb_eats + creature.enemy_eats + creature.squares_visited # reward exploration
        fitnessEval3 = creature.alive * (creature.strawb_eats + creature.enemy_eats + creature.squares_visited) # reward exploration, given alive
        fitnessEval4 = creature.turn * (creature.strawb_eats + creature.enemy_eats) # heavily rewards eating other things
        fitnessEval5 = creature.alive * creature.turn * (creature.strawb_eats + creature.enemy_eats) # heavily rewards eating other things
        fitnessEval6 = creature.turn * creature.size + creature.squares_visited# reward getting big fast and exploration
        fitnessEval7 = creature.alive * creature.turn * creature.size + creature.squares_visited# reward getting big fast and exploration, given alive
        fitnessEval8 = creature.turn * (creature.strawb_eats + creature.enemy_eats) # reward getting big fast, based on eats, doesn't account for size of eats
        fitnessEval9 = creature.alive * creature.turn * (creature.strawb_eats + creature.enemy_eats) # reward getting big fast, based on eats, doesn't account for size of eats
        
        fitnessFunctionOptions = [fitnessEval0, fitnessEval1, fitnessEval2, fitnessEval3, fitnessEval4, fitnessEval5, fitnessEval6, fitnessEval7, fitnessEval8, fitnessEval9]

        fitness[n] = fitnessFunctionOptions[fitnessOptionChoice]  # change this to reward diff behaviour

        for counter in range(len(creature.chromosome)):
            genes[n][counter] = creature.chromosome[counter]

    # At this point you should sort the agent according to fitness and create new population
    agentWithScore = list(zip (old_population, fitness))
    sorted(agentWithScore, key = lambda t: t[1])
    old_populationSorted, fitnessSorted = zip(*agentWithScore)

    new_population = list()
    lowerLim = int((N*proportionRetained))
    retainedCreatures = old_populationSorted[0:lowerLim]
    retainedSplit = np.array_split(np.array(retainedCreatures), 2)

    def column(matrix, i):
        return [row[i] for row in matrix]

    # get standard deviation of creature genes
    stdevs = list()
    for i in range(len(genes[0])):
        geneVals = (column(genes, i))
        stdevs.append(stats.pstdev(geneVals))

    # Here you should modify the new_creature's chromosome by selecting two parents (based on their
    # fitness) and crossing their chromosome to overwrite new_creature.chromosome

    # Consider implementing elitism, mutation and various other
    # strategies for producing new creature.
    retainedIndex = 0
    for n in range(N):
        if n < lowerLim:
            new_creature = retainedCreatures[n] # elitism
        else: 
            # Create new creature
            parent1 = retainedSplit[0][retainedIndex]
            parent2 = retainedSplit[1][retainedIndex]

            new_creature = MyCreature()
            chromoRaw = list((np.array(parent1.chromosome) + np.array(parent2.chromosome))/2) # very basic combination
            for i in range(len(stdevs)):
                new_creature.chromosome[i] = int(chromoRaw[i] + int(random.randint(-int(stdevs[i]/2), int(stdevs[i]/2)))) # allows for genetic diversity between children
            retainedIndex += 1

        if retainedIndex >= len(retainedSplit[0])-1:
            retainedIndex = 0

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)


    with open("stats.csv", "a") as myfile:
        myfile.write(str(avg_fitness))
        myfile.write(',')
        myfile.write(str(stdevs)[1:-1])
        myfile.write('\n')
    myfile.close()

    return (new_population, avg_fitness)
