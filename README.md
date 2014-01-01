Reinforcement-Learner
=====================

A python module that uses reinforcement learning to train game agents.

#Setup Information
+ Clone the repository to /some/path
+ cd to /some/path
+ Follow the algorithm instructions below to run various implementations of the A.I.

#Value Iteration Agent

Terminal Command:

`python gridworld.py -a value -i 5`

This assigns values to states where the values are the product of 5 value iterations. Press keys on the pop-up window to cycle through state values and run the Agent with a policy corresponding to the state values through 1 game.

Alternative Terminal Command:

`python gridworld.py -a value -i 100 -k 10`

This assigns values to states where the values are the product of 100 value iterations. Press keys on the pop-up window to cycle through state values and run the Agent with a policy corresponding to the state values through 100 games.

#Q Learning Agent

Implemented code that can be used to create agents that learn how to play value-based games. The qlearning agent can be used to teach a 2 dimensional crawler to walk or to teach pacman how to win games.

Terminal Command for Crawler:

`python crawler.py`

This simulates a 2d crawler that initially moves randomly and then learns how to walk. Lower Step Delay to speed up the learning process. After thousands of steps our agent will have some idea of how to walk, and you must lower Epsilon to 0 to see our agent's progress in learning.

Terminal Command for Pacman:

`python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 55 -l mediumClassic`

This simulates a pacman game where pacman plays 50 training games and learns weights on features of game states such as closeness to ghost, where pacman can eat ghosts, closeness of food and other features. After training on 50 games, pacman simulates his learned policy to play on 5 games.

Alternative Command for Pacman:

`python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l trickyClassic`

This simulates the pacman learning agent on a more difficult maze layout.

