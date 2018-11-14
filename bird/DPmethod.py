# Used to learn by using Dynamic Programming Algorithm
from bird.Game_make import game_make
from bird.Game import game
import numpy as np
import time

class DynamicProgramming(game):
    def __init__(self):
        game.__init__(self)
        self.V = np.zeros([17, 20])
        self.__rewards_init()
        self.__V_init()
        # used to save the strategy to make decision
        self.strategy = 'random'

        
    def ChangeStrategy(self, strategy):
        self.strategy = strategy

    def __rewards_init(self):
        # initialize rewards, which are set 100 in the goal and -100 in the brides, others are set to -1
        self.rewards = np.zeros([17, 20]) - 1
        for state in self.endstate:
            self.rewards[state[0], state[1]] -= 99
        self.rewards[0, 19] += 101

    def __V_init(self):
        # initialize State Value Function, which are set 1000 in the goal and -1000 in the brides
        for id in self.endstate:
            self.V[id[0], id[1]] = -1000
        self.V[0, 19] = 1000

    def stra_evaluation_random(self, k):
        # evaluate random policy and update the State Value Function by using Bellman Equation
        for iter in range(k):
            self.iteration += 1
            for i in range(17):
                for j in range(20):
                    if i == 0:
                        if j == 0:
                            self.V[i][j] = \
                                0.5 * (self.rewards[i][j] + self.gama * self.V[i + 1][j]) \
                                + 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j + 1])
                        elif j == 19:
                            self.V[i][j] = \
                                0.5 * (self.rewards[i][j] + self.gama * self.V[i + 1][j]) \
                                + 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                        else:
                            self.V[i][j] = \
                                1 / 3 * (self.rewards[i][j] + self.gama * self.V[i + 1][j]) \
                                + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j + 1]) \
                                + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                    elif i == 16:
                        if j == 0:
                            self.V[i][j] = \
                                0.5 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                                + 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j + 1])
                        elif j == 19:
                            self.V[i][j] = \
                                0.5 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                                + 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                        else:
                            self.V[i][j] = \
                                1 / 3 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                                + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j + 1]) \
                                + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                    elif j == 0:
                        self.V[i][j] = \
                            1 / 3 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                            + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j + 1]) \
                            + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i + 1][j])
                    elif j == 19:
                        self.V[i][j] = \
                            1 / 3 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                            + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j - 1]) \
                            + 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i + 1][j])
                    else:
                        self.V[i][j] = \
                            0.25 * (self.rewards[i][j] + self.gama * self.V[i - 1][j]) \
                            + 0.25 * (self.rewards[i][j] + self.gama * self.V[i][j - 1]) \
                            + 0.25 * (self.rewards[i][j] + self.gama * self.V[i + 1][j]) \
                            + 0.25 * (self.rewards[i][j] + self.gama * self.V[i][j + 1])
            self.__V_init()

    def stra_evaluation_greedy(self, k):
        # evaluate greedy policy of V and update the State Value Function by using Bellman Equation
        for iter in range(k):
            self.iteration += 1
            # select the max(V(s'))
            for i in range(17):
                for j in range(20):
                    k = []
                    if i == 0:
                        if j == 0:
                            m = max(self.V[i + 1][j], self.V[i][j + 1])
                            if m == self.V[j + 1][j]:
                                k.append(0)
                            if m == self.V[i + 1][j]:
                                k.append(1)
                        elif j == 19:
                            m = max(self.V[i + 1][j], self.V[i][j - 1])
                            if m == self.V[i + 1][j]:
                                k.append(1)
                            if m == self.V[i][j - 1]:
                                k.append(2)
                        else:
                            m = max(self.V[i + 1][j], self.V[i][j + 1], self.V[i][j - 1])
                            if m == self.V[i][j + 1]:
                                k.append(0)
                            if m == self.V[i + 1][j]:
                                k.append(1)
                            if m == self.V[i][j - 1]:
                                k.append(2)
                    elif i == 16:
                        if j == 0:
                            m = max(self.V[i][j + 1], self.V[i - 1][j])
                            if m == self.V[i][j + 1]:
                                k.append(0)
                            if m == self.V[i - 1][j]:
                                k.append(3)
                        elif j == 19:
                            m = max(self.V[i - 1][j], self.V[i][j - 1])
                            if m == self.V[i][j - 1]:
                                k.append(2)
                            if m == self.V[i - 1][j]:
                                k.append(3)
                        else:
                            m = max(self.V[i - 1][j], self.V[i][j - 1], self.V[i][j + 1])
                            if m == self.V[i][j + 1]:
                                k.append(0)
                            if m == self.V[i][j - 1]:
                                k.append(2)
                            if m == self.V[i - 1][j]:
                                k.append(3)
                    elif j == 0:
                        m = max(self.V[i + 1][j], self.V[i - 1][j], self.V[i][j + 1])
                        if m == self.V[i][j + 1]:
                            k.append(0)
                        if m == self.V[i + 1][j]:
                            k.append(1)
                        if m == self.V[i - 1][j]:
                            k.append(3)
                    elif j == 19:
                        m = max(self.V[i + 1][j], self.V[i - 1][j], self.V[i][j - 1])
                        if m == self.V[i + 1][j]:
                            k.append(1)
                        if m == self.V[i][j - 1]:
                            k.append(2)
                        if m == self.V[i - 1][j]:
                            k.append(3)
                    else:
                        m = max(self.V[i][j + 1], self.V[i + 1][j], self.V[i - 1][j], self.V[i][j - 1])
                        if m == self.V[i][j + 1]:
                            k.append(0)
                        if m == self.V[i + 1][j]:
                            k.append(1)
                        if m == self.V[i][j - 1]:
                            k.append(2)
                        if m == self.V[i - 1][j]:
                            k.append(3)
                    num = np.shape(k)[0]
                    # update the State Value Function V
                    p = np.random.random()
                    if num == 1:
                        n = k[0]
                        if n == 0:
                            self.V[i][j] = (self.rewards[i][j] + self.gama * self.V[i][j + 1])
                        elif n == 1:
                            self.V[i][j] = (self.rewards[i][j] + self.gama * self.V[i + 1][j])
                        elif n == 2:
                            self.V[i][j] = (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                        else:
                            self.V[i][j] = (self.rewards[i][j] + self.gama * self.V[i - 1][j])
                    elif num == 2:
                        self.V[i][j] = 0
                        n1 = k[0]
                        n2 = k[1]
                        if n1 == 0 or n2 == 0:
                            self.V[i][j] += 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j + 1])
                        if n1 == 1 or n2 == 1:
                            self.V[i][j] += 0.5 * (self.rewards[i][j] + self.gama * self.V[i + 1][j])
                        if n1 == 2 or n2 == 2:
                            self.V[i][j] += 0.5 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                        if n1 == 3 or n2 == 3:
                            self.V[i][j] += 0.5 * (self.rewards[i][j] + self.gama * self.V[i - 1][j])
                    elif num == 3:
                        self.V[i][j] = 0
                        n1 = k[0]
                        n2 = k[1]
                        n3 = k[2]
                        if n1 == 0 or n2 == 0 or n3 == 0:
                            self.V[i][j] += 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j + 1])
                        if n1 == 1 or n2 == 1 or n3 == 1:
                            self.V[i][j] += 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i + 1][j])
                        if n1 == 2 or n2 == 2 or n3 == 2:
                            self.V[i][j] += 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i][j - 1])
                        if n1 == 3 or n2 == 3 or n3 == 3:
                            self.V[i][j] += 1 / 3 * (self.rewards[i][j] + self.gama * self.V[i - 1][j])
                    else:
                        self.V[i][j] = 0.25 * ( \
                            (self.rewards[i][j] + self.gama * self.V[i][j + 1]) \
                            + (self.rewards[i][j] + self.gama * self.V[i + 1][j]) \
                            + (self.rewards[i][j] + self.gama * self.V[i][j - 1]) \
                            + (self.rewards[i][j] + self.gama * self.V[i - 1][j]))
    # improve the policy by update self.policy with greedy strategy 
    def stra_improve(self):
        self.state = [0, 0]
        while self.state not in self.endstate:
            old_state = self.state
            self.step_greedy_V()
            if (self.state[1] - old_state[1]) == 1:
                self.policy[old_state[0], old_state[1]] = 0
            elif (self.state[0] - old_state[0]) == 1:
                self.policy[old_state[0], old_state[1]] = 1
            elif (self.state[1] - old_state[1]) == -1:
                self.policy[old_state[0], old_state[1]] = 2
            elif (self.state[0] - old_state[0]) == -1:
                self.policy[old_state[0], old_state[1]] = 3

    def step(self):
        if self.strategy == 'random':
            self.step_random()
        if self.strategy == 'greedy':
            self.step_greedy_Q()

    def show(self):
        # visualize the policy
        game = game_make()
        self.state = [0, 0]
        count = 0
        while self.state not in self.endstate:
            count += 1
            if count > 120:
                break
            time.sleep(0.1)
            game.assign_state(self.state)
            game.flash_background()
            self.step_greedy_V()


dp = DynamicProgramming()
dp.stra_evaluation_random(100)
dp.ChangeStrategy('greedy')
dp.stra_evaluation_greedy(10000)
dp.show()
