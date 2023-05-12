import numpy as np
import random
import json

class PlayerAgent:
    def __init__(self, learning_rate, eps, gamma, player_name):
        self.player_name = player_name
        self.qvalues_dic = {}
        self.states_list = []
        self.learning_rate = learning_rate
        self.exploratory_eps = eps
        self.gamma = gamma

    def loadPolicy(self,path_policy_json):
        json_file = open(path_policy_json)
        self.qvalues_dic = json.load(json_file)

    def stateAppend(self, gameboard_new):
        self.states_list.append(gameboard_new)

    def qValueDictChecking(self, dictionnary, futur_state):
        if dictionnary.get(futur_state) is None:
            q_value = 0
        else:
            q_value = dictionnary.get(futur_state)
        return q_value

    def nextPosition(self, gameboard, empty_cells_index):
        # Exploratory
        if random.uniform(0, 1) < self.exploratory_eps:
            next_position = np.random.choice(empty_cells_index)
        # Max Q-value
        else:
            qmax = - 100000000000000
            for position in empty_cells_index:
                futur_state = gameboard.copy()
                futur_state[position] = self.player_name
                futur_state_str = str(futur_state)
                # Verify the value of this state and if None create it
                qvalue = self.qValueDictChecking(self.qvalues_dic, futur_state_str)
                if qvalue > qmax:
                    qmax = qvalue
                    next_position = position
        return next_position

    def backpropagationReward(self, reward):
        for state in reversed(self.states_list):
            if self.qvalues_dic.get(state) is None:
                self.qvalues_dic[state] = 0
            self.qvalues_dic[state] = self.qvalues_dic[state] + self.learning_rate * (
                    self.gamma * reward - self.qvalues_dic[state])
            reward = self.qvalues_dic[state]

    def resetStatesList(self):
        self.states_list = []

    def savePolicy(self):
        with open('policy_' + str(self.player_name) + ".json", "w") as outfile:
            json.dump(self.qvalues_dic,outfile, indent = 4)