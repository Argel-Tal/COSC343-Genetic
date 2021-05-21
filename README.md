# COSC343-Genetic
Genetic AI assignment for Otago Uni's COSC343 paper

Purpose of this assignment was to demonstrate the mechanics of genetic algorithms, through an agent based game. 
- The game is a team-based competitive multiagent environment, in which agents comsume a static resource, or hostile agents smaller than themselves. The side with the most agents left after 100 turns (all agents move once per turn) wins, irregardless of the size of those surviving agents. The game is similar to the game Agar.io (https://agar.io/#ffa), however somewhat simplified to reduce the complexity of the problem. Agents travel along a wrap-around grid space, travelling a fixed distance of 1 square per turn/move action, at right angles (no diagnonal translations).

- The genetic agent is initalised to a random state, and plays games against either a "random" or a "hunter" player. After each game, the genetic agent population is evaluated against a predetermined fitness function, and well preforming agents are allowed to propagate the next generation. This new generation plays the next game, and so forth.

- Ideally, through itterative games, the genetic algorithm should converge on a winning strategy, however the convergence is determined by the fitness function, not by wins||losses, so this function must be well defined such that it promotes winning behaviours.

- Agents are able to perceive up to 2 tiles away from themselves in any direction (including diagonally), creating a 5x5 percept map. The environment's three features, other agents, food items and walls, are seperated onto different percept layers, and do not obscure vision to other tiles. This means the environment is not fully observable for any given agent. Further, as agents do not communicate with each other, they do not form a larger world map from the small pieces each agent perceives, as swarm networks would.

- Chromosome values were written out for analytics, so I could check that the agent was learnin. As agent evolution was typically forced to converge, through itterative resampling with smaller standard deviations each time, non-reducing standard deviation values indicate the agent is still searching for a value of that gene which produces a better fitnesss function score. Standard devaitions of all the chromosomes, using fitness function 7: 
![Picture1](https://user-images.githubusercontent.com/80669114/118360465-c6a37b00-b5db-11eb-8047-9d1d50413236.png)

Creatures' chromosome, each weight is a "gene"
![creatureChromo](https://user-images.githubusercontent.com/80669114/119071655-52833000-ba3e-11eb-9f98-059c0fdabfb7.png)

Reproduction Schema for creating new creatures from exisitng population
![newCreatures](https://user-images.githubusercontent.com/80669114/119071661-53b45d00-ba3e-11eb-9f06-932717cffd75.png)



## Assignment has now been handed in, we're freeeeeeee
