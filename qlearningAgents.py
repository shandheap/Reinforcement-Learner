# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Author - Shandheap Shanmuganathan
    """
    def __init__(self, **args):
        ''' Initialize Q-values here. 
            Author - Shandheap Shanmuganathan '''

        ReinforcementAgent.__init__(self, **args)
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise

          Author - Shandheap Shanmuganathan 
        """

        return self.QValues[state, action]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          Author - Shandheap Shanmuganathan 
        """

        legalActions = self.getLegalActions(state)
        v = - sys.maxint - 1
        if len(legalActions) == 0:
          return 0.0
        for action in legalActions:
          v = max(v, self.getQValue(state, action))
        return v

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          Author - Shandheap Shanmuganathan 
        """

        legalActions = self.getLegalActions(state)
        nextValues = []

        if len(legalActions) == 0:
          return None
        for action in legalActions:
          nextValues.append(self.getQValue(state, action))

        bestScore = max(nextValues)
        bestIndices = [index for index in range(len(nextValues)) if nextValues[index] == bestScore]        
        chosenIndex = random.choice(bestIndices)
        
        return legalActions[chosenIndex]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          Author - Shandheap Shanmuganathan 
        """

        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions) == 0:
          return None
        
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.

          Author - Shandheap Shanmuganathan 
        """

        newQVal = reward + self.discount * self.computeValueFromQValues(nextState)
        oldQVal = self.getQValue(state, action)

        self.QValues[state, action] = (1 - self.alpha) * oldQVal + self.alpha * newQVal

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        # Reset Q Value to 0 before summation
        self.QValues[state, action] = 0
        featureDict = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()

        for feature in featureDict:
          # Update QValues with feature weights
          self.QValues[state, action] += (featureDict[feature] * weights[feature])
        
        return self.QValues[state, action]

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        weights = self.getWeights()

        nextValue = self.computeValueFromQValues(nextState)
        oldQVal = self.QValues[state, action]

        difference = (reward + self.discount * nextValue) - oldQVal
        featureDict = self.featExtractor.getFeatures(state, action)
        
        for feature in featureDict:
          oldWeight = weights[feature]
          weights[feature] = oldWeight + self.alpha * difference * featureDict[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        # if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging