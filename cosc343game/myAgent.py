import numpy as np
import random as random
import statistics as stats

playerName = "myAgent"
nPercepts = 75              # This is the number of percepts
nActions = 5                # This is the number of actions
proportionRetained = 0.15    # the proportion of agents that survive into the next generation
fitnessOptionChoice = 7     # Selected fitness function, from list of options
fitnessScores = list()
chromoStdevs = list()
probabilityOfBreakOut = 0.015  # Probability to break out of very low std values
breakoutThresh = 12
print("using fitness function: "+str(fitnessOptionChoice))

# Train against random for 5 generations, then against self for 1 generations
# trainingSchedule = [("random", 200)]
trainingSchedule = [("hunter", 60)]
# trainingSchedule = [("random", 200), ("hunter", 60)]

with open("stds.csv", "w") as myfile:
    myfile.write('')

with open("means.csv", "w") as myfile2:
    myfile2.write('')

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
        self.mutationRate = random.randint(-100, 100)
        self.chromosome = [self.weightAgentSize, self.weightAgentDist, self.weightAgentAttitude, self.weightAgentSizeGivenAttitude, self.weightFood, self.weightWall, self.weightConsume, self.weightSizeDif, self.weightSizeRelativeFood, self.mutationRate]
        self.missedEats = 0

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
        translations = [[-1,0], [0,-1], [1,0], [0,1]]
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
                netrelativeSizesDistanceAttitude += (self.weightSizeDif * (mySize - abs(val))) * (self.weightAgentDist * manhattanDistance(agent, currentDirection[0], currentDirection[1])) * (self.weightAgentAttitude * attVal) + (self.weightAgentSizeGivenAttitude * (mySize - abs(val)) * attVal)

            for food, val in np.ndenumerate(foodMap):
                netFoods += self.weightFood * (manhattanDistance(food, currentDirection[0], currentDirection[1]))
                netFoodRelativeSize += self.weightSizeRelativeFood * ((self.weightFood / mySize) * (manhattanDistance(food, currentDirection[0], currentDirection[1])))

            for wall, val in np.ndenumerate(wallMap):
                netWalls += (self.weightWall * (manhattanDistance(wall, currentDirection[0], currentDirection[1])))

            nets[i] = netOtherAgentsDistAttitude + netrelativeSizesDistanceAttitude + netFoods + netFoodRelativeSize + netWalls

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

        if(percepts[2, 2, 1] == 1) & (nets[4] != 1):
            self.missedEats = self.missedEats + 1

        return actions


def newGeneration(old_population):

    # This function should return a list of 'new_agents' that is of the same length as the
    # list of 'old_agents'.  That is, if previous game was played with N agents, the next game
    # should be played with N agents again.

    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))
    genes = np.zeros(((N, 10)))


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
        fitnessEval0 = creature.turn + creature.size + creature.strawb_eats + creature.enemy_eats  # baseline eval function
        fitnessEval1 = creature.alive * (creature.size + creature.strawb_eats + creature.enemy_eats)  # score > 0 only when still alive, else score = 0
        # fitness function 2 is my old preferred
        fitnessEval2 = creature.turn + creature.strawb_eats + creature.enemy_eats + creature.squares_visited  # reward exploration
        fitnessEval3 = creature.turn * creature.size**2 + creature.squares_visited - (creature.bounces/creature.squares_visited)

        fitnessEval4 = creature.turn * (creature.strawb_eats + creature.enemy_eats)  # heavily rewards eating other things
        fitnessEval5 = creature.alive * creature.turn * (creature.strawb_eats + creature.enemy_eats)  # heavily rewards eating other things
        # I don't like fitness function 6
        fitnessEval6 = creature.turn * creature.size**2 + creature.squares_visited + creature.enemy_eats # reward getting big fast and exploration
        fitnessEval8 = creature.turn * (creature.strawb_eats + creature.enemy_eats) + creature.squares_visited - (creature.bounces/creature.squares_visited)  # reward getting big fast, based on eats
        fitnessEval9 = creature.alive * creature.turn * (creature.strawb_eats + creature.enemy_eats + (creature.bounces/creature.squares_visited))  # reward getting big fast, based on eats
        # modified versions of fitness functions 2:
        fitnessEval10 = fitnessEval2 + (3.5 * creature.alive * fitnessEval2)  # reward surviving players more than dead ones
        fitnessEval11 = fitnessEval10 - creature.missedEats + creature.squares_visited - creature.bounces**2
        fitnessEval12 = fitnessEval2 - (creature.bounces/creature.squares_visited) - creature.missedEats
        fitnessEval7 = fitnessEval2 + creature.size**2 - (creature.bounces/creature.squares_visited)


        fitnessFunctionOptions = [fitnessEval0, fitnessEval1, fitnessEval2, fitnessEval3, fitnessEval4, fitnessEval5, fitnessEval6, fitnessEval7, fitnessEval8, fitnessEval9, fitnessEval10, fitnessEval11, fitnessEval12]

        fitness[n] = fitnessFunctionOptions[fitnessOptionChoice]  # change this to reward diff behaviour

        for counter in range(len(creature.chromosome)):
            genes[n][counter] = creature.chromosome[counter]

    # At this point you should sort the agent according to fitness and create new population
    agentWithScore = list(zip (old_population, fitness))
    sorted(agentWithScore, key = lambda t: t[1])
    old_populationSorted, fitnessSorted = zip(*agentWithScore)

    new_population = list()
    lowerLim = int((N*proportionRetained))
    retainedCreatures = old_populationSorted[0:lowerLim*2]  # allows for agents who score within the proportionRetained * 2 to have children, despite not surviving themselves
    retainedSplit = np.array_split(np.array(retainedCreatures), 2)  # could make this more dynamic and allow for an apriori variable of how many parents are used per creature

    def column(matrix, i):
        return [row[i] for row in matrix]

    # get standard deviation of creature genes
    stdevs = list()
    means = list()
    for i in range(len(genes[0])):
        geneVals = (column(genes, i))
        means.append(stats.mean(geneVals))
        stdevs.append(stats.pstdev(geneVals))

    # Here you should modify the new_creature's chromosome by selecting two parents (based on their
    # fitness) and crossing their chromosome to overwrite new_creature.chromosome

    # Consider implementing elitism, mutation and various other
    # strategies for producing new creature.
    retainedIndex = 0
    for n in range(N):
        if n < lowerLim:
            new_creature = retainedCreatures[n]  # elitism, limit is still using lowerLim, but retained creatures is now larger than lowerLim
        else: 
            # Create new creature, could make this more dynamic and allow for an apriori variable of how many parents are used per creature
            parent1 = retainedSplit[0][retainedIndex]
            parent2 = retainedSplit[1][retainedIndex]

            new_creature = MyCreature()
            parentPos = random.randrange(0, len(parent1.chromosome)-1)
            parent1Chromo = np.array(parent1.chromosome)[0:parentPos]
            parent2Chromo = np.array(parent2.chromosome)[parentPos:]

            # toggle block comments go here:

             
            # standard deviation based mutation mode
            chromoRaw = list(parent1Chromo) + list(parent2Chromo)  # takes the part of parent1 before the split point, and the part of parent2 that comes after split point

            for i in range(len(stdevs)):
                oddsMutation = random.random()
                mutation = int(random.randint(-int(stdevs[i]/2), int(stdevs[i]/2)))
                #  break out of local minima
                if ((stdevs[i]/2) < breakoutThresh) & (oddsMutation < probabilityOfBreakOut):
                    mutation = random.randint(-int(parent1.mutationRate), int(parent1.mutationRate))
                new_creature.chromosome[i] = int(chromoRaw[i] + mutation)  # allows for genetic diversity between children
            retainedIndex += 1

            '''
            
            # generic random value based mutation mode
            chromoRaw = list(parent1Chromo) + list(parent2Chromo)  # takes the part of parent1 before the split point, and the part of parent2 that comes after split point

            for i in range(len(stdevs)):
                mutation = 0
                oddsMutation = random.random()
                if oddsMutation < probabilityOfBreakOut:
                    mutation = random.randint(-100, 100)
                new_creature.chromosome[i] = int(chromoRaw[i] + mutation)
            retainedIndex += 1

            #  toggle comments go here:
            '''

        if retainedIndex >= len(retainedSplit[0])-1:
            retainedIndex = 0

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)


    with open("stds.csv", "a") as myfile:
        myfile.write(str(avg_fitness))
        myfile.write(',')
        myfile.write(str(stdevs)[1:-1])
        myfile.write('\n')
    myfile.close()

    with open("means.csv", "a") as myfile2:
        myfile2.write(str(avg_fitness))
        myfile2.write(',')
        myfile2.write(str(means)[1:-1])
        myfile2.write('\n')
    myfile2.close()


    return (new_population, avg_fitness)
