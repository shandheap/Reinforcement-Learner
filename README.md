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