# Used to learn by using Monte Carlo method
from bird.Game_make import game_make
from bird.Game import game
import numpy as np
import time

class MonteCarlo(game):
    def __init__(self):
        game.__init__(self)
        self.__rewards_init()
        self.__Q_init()
        # calculate the occurrences of (s, a), which are used to calculate Gt, the cumulative returns
        self.Num = np.zeros([17, 20, 4])
        # the cumulative returns
        self.Gt = 0
        # used in Exploratory initialization, which is used to decide whether the bird is born in the endstate
        self.end = 0
        self.epsilon = 0.3
        # used in Offpolicy MC
        self.C = np.zeros([17, 20, 4])

    def __rewards_init(self):
        self.rewards = np.zeros([17, 20, 4]) - 1
        # because it's difficult to achieve the goal, I set the rewards arround the goal large
        for i in range(9):
            self.rewards[i][6][0] -= 9999
            self.rewards[i][8][2] -= 9999
            self.rewards[16 - i][13][0] -= 9999
            self.rewards[16 - i][15][2] -= 9999
        for i in range(4):
            self.rewards[i][13][0] -= 9999
            self.rewards[i][15][2] -= 9999
            self.rewards[16 - i][6][0] -= 9999
            self.rewards[16 - i][8][2] -= 9999
        self.rewards[9][7][3] = -9999
        self.rewards[12][7][1] = -9999
        self.rewards[4][14][3] = -9999
        self.rewards[7][14][1] = -9999
        self.rewards[0][18][0] += 1000000000000000001
        self.rewards[1][19][3] += 1000000000000000001

    # initialize the Action State Value Function
    def __Q_init(self):
        self.Q = np.random.random([17, 20, 4]) - 1


    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    """               Exploratory initializat                """
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""


    def __state_init(self):
        # initialize a random state S as initial state
        self.s_a = []
        self.Gt = 0
        self.flag = np.ones([17, 20, 4])
        ran = np.random.randint(0, 340) # Generate a random Integer in [0, 339]
        self.state = [int(ran/20), ran%20]
        p = np.random.random()
        i = self.state[0]
        j = self.state[1]
        if i == 0:
            if j == 0:
                if p < 0.5:
                    self.s_a.append([self.state, 0])
                else:
                    self.s_a.append([self.state, 1])
            elif j == 19:
                if p < 0.5:
                    self.s_a.append([self.state, 1])
                else:
                    self.s_a.append([self.state, 2])
            else:
                if p < 0.333:
                    self.s_a.append([self.state, 0])
                elif p < 0.666:
                    self.s_a.append([self.state, 1])
                else :
                    self.s_a.append([self.state, 2])
        elif i == 16:
            if j == 0:
                if p < 0.5:
                    self.s_a.append([self.state, 0])
                else:
                    self.s_a.append([self.state, 3])
            elif j == 19:
                if p < 0.5:
                    self.s_a.append([self.state, 3])
                else:
                    self.s_a.append([self.state, 2])
            else:
                if p < 0.333:
                    self.s_a.append([self.state, 0])
                elif p < 0.666:
                    self.s_a.append([self.state, 2])
                else :
                    self.s_a.append([self.state, 3])
        elif j == 0:
            if p < 0.333:
                self.s_a.append([self.state, 0])
            elif p < 0.666:
                self.s_a.append([self.state, 1])
            else:
                self.s_a.append([self.state, 3])
        elif j == 19:
            if p < 0.333:
                self.s_a.append([self.state, 1])
            elif p < 0.666:
                self.s_a.append([self.state, 2])
            else:
                self.s_a.append([self.state, 3])
        else :
            if p < 0.25:
                self.s_a.append([self.state, 0])
            elif p < 0.5:
                self.s_a.append([self.state, 1])
            elif p < 0.75:
                self.s_a.append([self.state, 2])
            else:
                self.s_a.append([self.state, 3])

    # take the first step according to the initial random state
    def __first_step(self):
        # jump to the next episode if born in the endstate
        if self.state in self.endstate:
            self.end = 1
            return
        else:
            i = self.state[0]
            j = self.state[1]
            if self.s_a[0][1] == 0:
                self.Gt += self.rewards[i][j][0] * self.gama
                self.state = [i, j + 1]
            elif self.s_a[0][1] == 1:
                self.Gt += self.rewards[i][j][1] * self.gama
                self.state = [i + 1, j]
            elif self.s_a[0][1] == 2:
                self.Gt += self.rewards[i][j][2] * self.gama
                self.state = [i, j - 1]
            elif self.s_a[0][1] == 3:
                self.Gt += self.rewards[i][j][3] * self.gama
                self.state = [i - 1, j]

    """
    ++++++++random++++++++
    """

    def step_MC_random(self, gama):
        # take next step according to a random strategy
        state = self.state.copy()
        i = state[0]
        j = state[1]
        self.step_random()
        # update the episode self.s_a
        self.s_a.append([state, self.a])
        # update the Cumulative Return
        self.Gt += gama * self.rewards[i][j][self.a]

    def evaluate_random(self, k):
        for i in range(k):
            self.iteration += 1
            self.__state_init()
            self.__first_step()
            if self.end:
                self.end = 0
                continue
            else:
                # take an epsimode with penalty self.game
                gama = self.gama ** 2
                while self.state not in self.endstate:
                    self.step_MC_random(gama)
                    gama *= self.gama
                # calculate the Cumulative Return of each (s, a) and update Action-State Value Function Q(s, a)
                gama = self.gama
                for state, action in self.s_a:
                    # first visit MC
                    i = state[0]
                    j = state[1]
                    if self.flag[i][j][action] :
                        # 加锁
                        self.flag[i][j][action] = 0
                        self.Q[i][j][action] = (self.Q[i][j][action] * self.Num[i][j][action] + self.Gt) / (self.Num[i][j][action] + 1)
                        self.Num[i][j][action] += 1
                    self.Gt -= self.rewards[state[0]][state[1]][action] * gama
                    gama *= self.gama

    """
    ++++++++greedy++++++++
    """
    # use greedy policy instead of random policy
    def step_MC_greedy(self, gama):
        state = self.state.copy()
        i = state[0]
        j = state[1]
        self.step_greedy_Q()
        self.s_a.append([state, self.a])
        self.Gt += gama * self.rewards[i][j][self.a]

    def evaluate_greedy(self, k):
        for i in range(k):
            self.iteration += 1
            self.__state_init()
            self.__first_step()
            if self.end:
                self.end = 0
                continue
            else:
                count = 0
                gama = self.gama ** 2
                while self.state not in self.endstate:
                    count += 1
                    if count > 200:
                        self.Gt -= 10
                        break
                    self.step_MC_greedy(gama)
                    gama *= self.gama
                gama = self.gama
                for state, action in self.s_a:
                    i = state[0]
                    j = state[1]
                    # first visit MC
                    if self.flag[i][j][action] :
                        self.flag[i][j][action] = 0
                        self.Q[i][j][action] = (self.Q[i][j][action] * self.Num[i][j][action] + self.Gt) / (self.Num[i][j][action] + 1)
                        self.Num[i][j][action] += 1
                    self.Gt -= self.rewards[state[0]][state[1]][action] * gama
                    gama *= self.gama




    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    """                       On-policy                      """
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    
    # Similar to the previous one, but start in [0, 0] instead of random one, and use a epsilon-greedy policy
    def step_Onpolicy(self, gama, epsilon):
        state = self.state.copy()
        i = state[0]
        j = state[1]
        self.step_epsilon_greedy_Q(epsilon)
        self.s_a.append([state, self.a])
        self.Gt += gama * self.rewards[i][j][self.a]

    def evaluate_Onpolicy(self, k):
        for i in range(k):
            print(self.iteration)
            self.s_a = []
            self.Gt = 0
            self.flag = np.ones([17, 20, 4])
            self.iteration += 1
            self.state = [0, 0]
            gama = self.gama ** 2
            while self.state not in self.endstate:
                self.step_Onpolicy(gama, self.epsilon)
                gama *= self.gama
            gama = self.gama
            for state, action in self.s_a:
                if state == [0, 19]:
                    self.epsilon = 0.05
                i = state[0]
                j = state[1]
                # first visit MC
                if self.flag[i][j][action]:
                    self.flag[i][j][action] = 0
                    self.Q[i][j][action] = (self.Q[i][j][action] * self.Num[i][j][action] + self.Gt) / (self.Num[i][j][action] + 1)
                    self.Num[i][j][action] += 1
                self.Gt -= self.rewards[state[0]][state[1]][action] * gama
                gama *= self.gama





    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    """                      Off-policy                      """
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    def evaluate_Offpolicy(self, k):
        for i in range(k):
            self.s_a = []
            self.Gt = 0
            self.flag = np.ones([17, 20, 4])
            self.iteration += 1
            self.state = [0, 0]
            gama = self.gama ** 2
            while self.state not in self.endstate:
                self.step_Onpolicy(gama, self.epsilon)
                gama *= self.gama
            self.Gt = 0
            W = 1
            sample_size = np.shape(self.s_a)[0]
            for i in range(sample_size):
                state = self.s_a[sample_size - i - 1][0]
                action = self.s_a[sample_size - i - 1][1]
                i = state[0]
                j = state[1]
                self.Gt = self.gama * self.Gt + self.rewards[i][j][action]
                self.C[i][j][action] += W
                self.Q[i][j][action] += W / self.C[i][j][action] * (self.Gt - self.Q[i][j][action])
                k = []
                if i == 0:
                    if j == 0:
                        m = max(self.Q[i][j][0], self.Q[i][j][1])
                        if m == self.Q[i][j][0]:
                            k.append(0)
                        if m == self.Q[i][j][1]:
                            k.append(1)
                    elif j == 19:
                        m = max(self.Q[i][j][2], self.Q[i][j][1])
                        if m == self.Q[i][j][1]:
                            k.append(1)
                        if m == self.Q[i][j][2]:
                            k.append(2)
                    else:
                        m = max(self.Q[i][j][2], self.Q[i][j][1], self.Q[i][j][0])
                        if m == self.Q[i][j][0]:
                            k.append(0)
                        if m == self.Q[i][j][1]:
                            k.append(1)
                        if m == self.Q[i][j][2]:
                            k.append(2)
                elif i == 16:
                    if j == 0:
                        m = max(self.Q[i][j][0], self.Q[i][j][3])
                        if m == self.Q[i][j][0]:
                            k.append(0)
                        if m == self.Q[i][j][3]:
                            k.append(3)
                    elif j == 19:
                        m = max(self.Q[i][j][2], self.Q[i][j][3])
                        if m == self.Q[i][j][2]:
                            k.append(2)
                        if m == self.Q[i][j][3]:
                            k.append(3)
                    else:
                        m = max(self.Q[i][j][2], self.Q[i][j][3], self.Q[i][j][0])
                        if m == self.Q[i][j][0]:
                            k.append(0)
                        if m == self.Q[i][j][2]:
                            k.append(2)
                        if m == self.Q[i][j][3]:
                            k.append(3)
                elif j == 0:
                    m = max(self.Q[i][j][3], self.Q[i][j][1], self.Q[i][j][0])
                    if m == self.Q[i][j][0]:
                        k.append(0)
                    if m == self.Q[i][j][1]:
                        k.append(1)
                    if m == self.Q[i][j][3]:
                        k.append(3)
                elif j == 19:
                    m = max(self.Q[i][j][2], self.Q[i][j][1], self.Q[i][j][3])
                    if m == self.Q[i][j][1]:
                        k.append(1)
                    if m == self.Q[i][j][2]:
                        k.append(2)
                    if m == self.Q[i][j][3]:
                        k.append(3)
                else:
                    m = max(self.Q[i][j][2], self.Q[i][j][1], self.Q[i][j][3], self.Q[i][j][0])
                    if m == self.Q[i][j][0]:
                        k.append(0)
                    if m == self.Q[i][j][1]:
                        k.append(1)
                    if m == self.Q[i][j][2]:
                        k.append(2)
                    if m == self.Q[i][j][3]:
                        k.append(3)
                num = np.shape(k)[0]
                p = np.random.random()
                if num == 1:
                    n = k[0]
                elif num == 2:
                    if p < 0.5:
                        n = k[0]
                    else:
                        n = k[1]
                elif num == 3:
                    if p < 0.333:
                        n = k[0]
                    elif p < 0.666:
                        n = k[1]
                    else:
                        n = k[2]
                else:
                    if p < 0.25:
                        n = k[0]
                    elif p < 0.5:
                        n = k[1]
                    elif p < 0.75:
                        n = k[2]
                    else:
                        n = k[3]
                self.policy[i][j] = n
                if self.policy[i][j] != action:
                    break
                if self.policy[i][j] == action:
                    p = 1 - self.epsilon + self.epsilon / num
                else:
                    p = self.epsilon / num
                W = W / p
            print(self.iteration)



    # visualiz the policy
    def show(self):
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
            self.step_greedy_Q()


MC = MonteCarlo()

# MC.evaluate_random(5000)
# MC.evaluate_greedy(5000)

MC.evaluate_Onpolicy(10000)

# MC.evaluate_Offpolicy(10000)

MC.show()
