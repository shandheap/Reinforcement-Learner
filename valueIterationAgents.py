# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import sys
import mdp, util
import pdb

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Author - Shandheap Shanmuganathan
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default values as 0
        self.count = 1
        while self.count <= iterations:
          for state in mdp.getStates():
            possibleActions = mdp.getPossibleActions(state)
            if len(possibleActions) == 0:
              continue
            QValues = {}
            for action in possibleActions:
              if action == "exit":
                finalScore = self.mdp.getReward(state, action, 'TERMINAL_STATE')
                self.values[state, self.count] = finalScore
                continue
              else:
                QValues[action] = self.getQValue(state, action)
            maxAction = None
            maxQ = -sys.maxint - 1
            for key, value in QValues.iteritems():
              if value > maxQ:
                maxAction = key
                maxQ = value
            if maxQ != -sys.maxint - 1:
              self.values[state, self.count] = maxQ
          self.count += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state, self.iterations]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        nextTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for transition in nextTransitions:
          reward = self.mdp.getReward(state, action, transition[0])
          previousValue = self.values[transition[0], self.count - 1]
          QValue += transition[1] * (reward + self.discount * previousValue)
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        maxAction = None
        possibleActions = self.mdp.getPossibleActions(state)
        allQValues = []
        if len(possibleActions) == 0:
          return None
        for action in possibleActions:
          QValue = self.getQValue(state, action)
          if len(allQValues) == 0 or QValue > max(allQValues):
            maxAction = action
          allQValues.append(QValue)
        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
