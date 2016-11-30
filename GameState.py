from collections import defaultdict
import random, math
from copy import deepcopy

time = 0

class RLAgent:
    # mass
    # id = index into state
    # history = list of (s, a, o, r, s')
    def __init__(self, id, m):
        self.m = m
        self.id = id
        self.history = []
        self.policy = {}
    def setMass(self, m):
        self.m = m
    def getAction(self, state):
        raise Exception("Override me")
    def learn(self):
        raise Exception("Override me")
    def legalActions(self):
        return ['A','I','W']

class RandomAgent(RLAgent):
    def getAction(self, state):
        return random.choice(self.legalActions())


class NaiveQAgent(RLAgent):
    def initQ(self):
        self.Q = defaultdict(int)

    def getQ(self):
        return self.Q

    def getAction(self, state, explore_prob):
        if random.random() < explore_prob:
            return random.choice(self.legalActions())
        myState = int(state[self.id])
        acts = self.legalActions()
        return sorted(acts, key = lambda a : (self.Q[(myState, a)], random.random()))[-1]


    def learn(self, state, action, reward, newState):
        myState = int(state[self.id])
        myAction = action[self.id]
        myReward = reward[self.id]
        myNewState = int(newState[self.id])
        #if myState > 12.5 : print state, action, reward, newState
        self.Q[(myState,myAction)] += 0.01 * (myReward + \
                                         0.9999 * max([self.Q[(myNewState, a)] for a in self.legalActions()]) - self.Q[(myState,myAction)] )

class GameState:
    # state [m1, m2]
    # end
    def __init__(self, state):
        self.state = state
    def getState(self):
        return self.state
    ### A = Attack, I = Idle, W = Work
    def getProbSuccess(self, m1, m2):
        if m1 <= m2: return 0
        return 1 - math.exp(1 - m1/m2)

    def attack(self, m1, m2, defaultAction):
        a = random.random()
        if m1 > m2 and a < self.getProbSuccess(m1, m2):
            m1 += m2
            m2 = -1
        else:
            m1 *= 0.9
            if defaultAction == 'I':
                m2 *= 0.99
            elif defaultAction == 'W':
                m1 += m2 * 0.06
                m2 *= 0.97
            elif defaultAction == 'A':
                m2 *= 0.9
        return m1, m2

    def applyActions(self, actions):
        oldReward = self.state[0], self.state[1]
        newM0 = self.state[0]
        newM1 = self.state[1]

        if actions == ('A', 'A') :
            if newM0 >= newM1:
                newM0, newM1 = self.attack(newM0, newM1, 'A')
            elif newM0 <= newM1:
                newM1, newM0 = self.attack(newM1, newM0, 'A')
        elif actions == ('A', 'I') :
            newM0, newM1 = self.attack(newM0, newM1, 'I')
        elif actions == ('I', 'A') :
            newM1, newM0 = self.attack(newM1, newM0, 'I')
        elif actions == ('A', 'W') :
            newM0, newM1 = self.attack(newM0, newM1, 'W')
        elif actions == ('W', 'A') :
            newM1, newM0 = self.attack(newM1, newM0, 'W')
        elif actions == ('I', 'I') :
            newM0 *= 0.99
            newM1 *= 0.99
        elif actions == ('I', 'W') :
            newM0 += 0.06 * newM1
            newM1 *= 0.97
        elif actions == ('W', 'I') :
            newM1 += 0.06 * newM0
            newM0 *= 0.97
        elif actions == ('W', 'W') :
            newM0 += 0.06 * self.state[1] - 0.03 * self.state[0]
            newM1 += 0.06 * self.state[0] - 0.03 * self.state[1]

        self.state[0] = newM0
        self.state[1] = newM1
        if self.isEnd():
            if newM0 > 1 and newM1 > 1:
                self.state[0] += 1000
                self.state[1] += 1000

        return [self.state[0]-oldReward[0], self.state[1]-oldReward[1]], [self.state[0], self.state[1]]

    def isEnd(self):
        return min(self.state) <= 1 or max(self.state) >= 13. # min(self.state) >= 12


agents = [NaiveQAgent(0, 10), NaiveQAgent(1, 10)]
agents[0].initQ()
agents[1].initQ()

for i in range(20000): # how to terminate
  time += 1
  game = GameState([10, 10])
  while not game.isEnd():
    state = deepcopy(game.getState())
    actions = (agents[0].getAction(state, 0.05), agents[1].getAction(state, 0.05))
    gains, nextState = game.applyActions(actions)
    agents[0].learn(state, actions, gains, nextState)
    agents[1].learn(state, actions, gains, nextState)
    ### print state, actions
  agents[0].setMass(10)
  agents[1].setMass(10)
  if i % 2000 == 1: print i

print agents[0].getQ()
print agents[1].getQ()

while 1:
  game = GameState([10, 10])
  while not game.isEnd():
    state = game.getState()

    actions = (agents[0].getAction(state, 0.0), agents[1].getAction(state, 0.0))
    gains, nextState = game.applyActions(actions)
    agents[0].learn(state, actions, gains, nextState)
    agents[1].learn(state, actions, gains, nextState)
    print state, actions
    x = raw_input()
