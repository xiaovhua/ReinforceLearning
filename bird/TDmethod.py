# Used to learn bu using Time Difference Algorithm
from bird.Game import game
from bird.Game_make import game_make
import numpy as np
import time

class TimeDifference(game):
    def __init__(self):
        game.__init__(self)
        self.__Q_init()
        self.__rewards_init()
        # the learning rate
        self.alpha = 0.5

    # initialize the Action-State Value Function
    def __Q_init(self):
        self.Q = np.zeros([17, 20, 4])

    # initialize the rewards
    def __rewards_init(self):
        self.rewards = np.zeros([17, 20, 4]) - 1
        for i in range(9):
            self.rewards[i][6][0] -= 999
            self.rewards[i][8][2] -= 9999
            self.rewards[16 - i][13][0] -= 999
            self.rewards[16 - i][15][2] -= 999
        for i in range(4):
            self.rewards[i][13][0] -= 999
            self.rewards[i][15][2] -= 999
            self.rewards[16 - i][6][0] -= 999
            self.rewards[16 - i][8][2] -= 999
        self.rewards[9][7][3] = -999
        self.rewards[12][7][1] = -999
        self.rewards[4][14][3] = -999
        self.rewards[7][14][1] = -999
        self.rewards[0][18][0] += 100001
        self.rewards[1][19][3] += 100001

    # Use sarsa(epsilon-greedy) to evaluate policy
    def evaluation_Sarsa(self, k):
        # epsilon in epsilon-greedy
        self.epsilon = 0.1
        for i in range(k):
            self.iteration += 1
            self.state = [0, 0]
            # get the initial (s, a)
            i = self.state[0]
            j = self.state[1]
            self.step_epsilon_greedy_Q(self.epsilon)
            a = self.a
            self.state = [i, j]
            while self.state not in self.endstate:
                # Calculate a S'
                if a == 0:
                    new_i = i
                    new_j = j + 1
                elif a == 1:
                    new_i = i + 1
                    new_j = j
                elif a == 2:
                    new_i = i
                    new_j = j - 1
                else:
                    new_i = i - 1
                    new_j = j
                self.state = [new_i, new_j]
                # Calculate a'
                self.step_epsilon_greedy_Q(self.epsilon)
                new_a = self.a
                self.state = [new_i, new_j]
                # update Q(s,a)
                self.Q[i][j][a] += self.alpha * (self.rewards[i][j][a] + self.gama * self.Q[new_i][new_j][new_a] - self.Q[i][j][a])
                a = new_a
                i = new_i
                j = new_j
            if self.iteration % 500 == 0 :
                print(self.iteration)
                
    # Use Qlearning(greedy) to evaluate policy
    def evaluation_Qlearning(self, k):
        self.epsilon = 0.1
        for i in range(k):
            self.iteration += 1
            self.state = [0, 0]
            # get the initial (s, a)
            i = self.state[0]
            j = self.state[1]
            self.step_epsilon_greedy_Q(self.epsilon)
            a = self.a
            self.state = [i, j]
            while self.state not in self.endstate:
                # Calculate s'
                if a == 0:
                    new_i = i
                    new_j = j + 1
                elif a == 1:
                    new_i = i + 1
                    new_j = j
                elif a == 2:
                    new_i = i
                    new_j = j - 1
                else:
                    new_i = i - 1
                    new_j = j
                self.state = [new_i, new_j]
                # Calculate Qmax(S')
                self.step_greedy_Q()
                new_a = self.a
                self.state = [new_i, new_j]
                Qmax = self.Q[new_i][new_j][new_a]
                # Update Q(s, a)
                self.Q[i][j][a] += self.alpha * (self.rewards[i][j][a] + self.gama * Qmax - self.Q[i][j][a])
                # calculate a'
                self.step_epsilon_greedy_Q(self.epsilon)
                a = self.a
                self.state = [new_i, new_j]
                a = new_a
                i = new_i
                j = new_j
            if self.iteration % 100 == 0:
                print(self.iteration)

    # Use Expected Sarsa to evaluate policy
    def evaluation_expected_Sarsa(self, k):
        self.epsilon = 0.1
        for i in range(k):
            self.iteration += 1
            self.state = [0, 0]
            # Calculate (S, a)
            i = self.state[0]
            j = self.state[1]
            self.step_epsilon_greedy_Q(self.epsilon)
            a = self.a
            self.state = [i, j]
            while self.state not in self.endstate:
                # Calculate s'
                if a == 0:
                    new_i = i
                    new_j = j + 1
                elif a == 1:
                    new_i = i + 1
                    new_j = j
                elif a == 2:
                    new_i = i
                    new_j = j - 1
                else:
                    new_i = i - 1
                    new_j = j
                self.state = [new_i, new_j]
                # Calculate a'
                self.step_epsilon_greedy_Q(self.epsilon)
                new_a = self.a
                self.state = [new_i, new_j]
                # Calculate expected_Q
                expected_Q = (1 - self.epsilon) * self.Q[new_i][new_j][new_a]
                for k in range(4):
                    expected_Q += self.epsilon / 4 * self.Q[new_i][new_j][k]
                # update Q(s, a)
                self.Q[i][j][a] += self.alpha * (self.rewards[i][j][a] + self.gama * expected_Q - self.Q[i][j][a])
                a = new_a
                i = new_i
                j = new_j
            if self.iteration % 100 == 0 :
                print(self.iteration)

    # Use double Qlearning to evaluate policy, which can avoid overfitting by noise
    def evaluation_double_Qlearning(self, k):
        Q1 = self.Q
        Q2 = self.Q.copy()
        self.Q = Q1 + Q2
        for i in range(k):
            self.iteration += 1
            self.state = [0, 0]
            # calculate (S, a)
            i = self.state[0]
            j = self.state[1]
            self.step_epsilon_greedy_Q(self.epsilon)
            a = self.a
            self.state = [i, j]
            while self.state not in self.endstate:
                # Calculate S'
                if a == 0:
                    new_i = i
                    new_j = j + 1
                elif a == 1:
                    new_i = i + 1
                    new_j = j
                elif a == 2:
                    new_i = i
                    new_j = j - 1
                else:
                    new_i = i - 1
                    new_j = j
                self.state = [new_i, new_j]
                # Update Q2
                if np.random.random() > 0.5:
                    if new_i == 0:
                        if new_j == 0:
                            new_a = np.argmax(np.array([Q2[new_i][new_j][0], Q2[new_i][new_j][1]]))
                        elif new_j == 19:
                            new_a = np.argmax(np.array([Q2[new_i][new_j][1], Q2[new_i][new_j][2]]))
                        else:
                            new_a = np.argmax(np.array([Q2[new_i][new_j][0], Q2[new_i][new_j][1], Q2[new_i][new_j][2]]))
                    elif new_i == 16:
                        if new_j == 0:
                            new_a = np.argmax(np.array([Q2[new_i][new_j][0], -10000000000, -10000000000, Q2[new_i][new_j][3]]))
                        elif new_j == 19:
                            new_a = np.argmax(np.array([-10000000000, -10000000000, Q2[new_i][new_j][2], Q2[new_i][new_j][3]]))
                        else:
                            new_a = np.argmax(np.array([Q2[new_i][new_j][0], -10000000000, Q2[new_i][new_j][2], Q2[new_i][new_j][3]]))
                    elif new_j == 0:
                        new_a = np.argmax(np.array([Q2[new_i][new_j][0], Q2[new_i][new_j][1], -10000000000, Q2[new_i][new_j][3]]))
                    elif new_j == 19:
                        new_a = np.argmax(np.array([-10000000000, Q2[new_i][new_j][1], Q2[new_i][new_j][2], Q2[new_i][new_j][3]]))
                    else:
                        new_a = np.argmax(np.array([Q2[new_i][new_j][0], Q2[new_i][new_j][1], Q2[new_i][new_j][2], Q2[new_i][new_j][3]]))
                    Qmax = Q1[new_i][new_j][new_a]
                    Q2[i][j][a] += self.alpha * (self.rewards[i][j][a] + self.gama * Qmax - Q2[i][j][a])
                # Update Q1
                else:
                    if new_i == 0:
                        if new_j == 0:
                            new_a = np.argmax(np.array([Q1[new_i][new_j][0], Q1[new_i][new_j][1]]))
                        elif new_j == 19:
                            new_a = np.argmax(np.array([Q1[new_i][new_j][1], Q1[new_i][new_j][2]]))
                        else:
                            new_a = np.argmax(np.array([Q1[new_i][new_j][0], Q1[new_i][new_j][1], Q1[new_i][new_j][2]]))
                    elif new_i == 16:
                        if new_j == 0:
                            new_a = np.argmax(np.array([Q1[new_i][new_j][0], -10000000000, -10000000000, Q1[new_i][new_j][3]]))
                        elif new_j == 19:
                            new_a = np.argmax(np.array([-10000000000, -10000000000, Q1[new_i][new_j][2], Q1[new_i][new_j][3]]))
                        else:
                            new_a = np.argmax(np.array([Q1[new_i][new_j][0], -10000000000, Q1[new_i][new_j][2], Q1[new_i][new_j][3]]))
                    elif new_j == 0:
                        new_a = np.argmax(np.array([Q1[new_i][new_j][0], Q1[new_i][new_j][1], -10000000000, Q1[new_i][new_j][3]]))
                    elif new_j == 19:
                        new_a = np.argmax(np.array([-10000000000, Q1[new_i][new_j][1], Q1[new_i][new_j][2], Q1[new_i][new_j][3]]))
                    else:
                        new_a = np.argmax(np.array([Q1[new_i][new_j][0], Q1[new_i][new_j][1], Q1[new_i][new_j][2], Q1[new_i][new_j][3]]))
                    Qmax = Q2[new_i][new_j][new_a]
                    Q1[i][j][a] += self.alpha * (self.rewards[i][j][a] + self.gama * Qmax - Q1[i][j][a])
                # Calculate a'
                self.Q = Q1 + Q2
                self.step_epsilon_greedy_Q(self.epsilon)
                new_a = self.a
                self.state = [new_i, new_j]
                a = new_a
                i = new_i
                j = new_j
            if self.iteration % 100 == 0:
                print(self.iteration)

    # visialize the policy
    def show(self):
        game = game_make()
        self.state = [0, 0]
        count = 0
        while self.state not in self.endstate:
            count += 1
            if count > 200:
                break
            time.sleep(0.1)
            game.assign_state(self.state)
            game.flash_background()
            self.step_greedy_Q()



TD = TimeDifference()

#TD.evaluation_Sarsa(5000)

#TD.evaluation_Qlearning(300)

#TD.evaluation_expected_Sarsa(300)

TD.evaluation_double_Qlearning(2000)

TD.show()
