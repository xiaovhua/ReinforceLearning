# Used to describe the game into [state, action, rewards, StateTransitionMatrix, penalty] 
import numpy as np
class game():
    def __init__(self):
        # types of action
        self.action = ['e', 's', 'w', 'n']
        # the index of the action which is taken in this step
        self.a = 0
        # the endstate, including the brick and the goal
        self.endstate_init()
        # current state
        self.state = [0, 0]
        # rewards
        self.rewards = np.zeros([17, 20, 4])
        # State Value Function
        self.V = np.zeros([17, 20])
        # Action-State Valve Function
        self.Q = np.zeros([17, 20, 4])
        # policy to make decision
        self.policy = np.zeros([17, 20])
        # penalty
        self.gama = 0.98
        # iteration
        self.iteration = 0
        # the (state, action) of one step
        self.s_a = []
        # the value of epsilon in epsilon-greedy
        self.epsilon = 0.25
        # the State Transition Matrix, the default is one
        self.P = np.ones([17, 20, 4])
        
    # initial the endstate, and describe by a [x, y]
    def endstate_init(self):
        self.endstate = []
        self.endstate.append([0, 19])
        for i in range(9):
            self.endstate.append([i, 7])
            self.endstate.append([16 - i, 14])
        for i in range(4):
            self.endstate.append([i, 14])
            self.endstate.append([16 - i, 7])
    
    # select an action and move to next state with a random policy
    def step_random(self):
        i = self.state[0]
        j = self.state[1]
        p = np.random.random()
        # take an action and then save in self.a
        if i == 0:
            if j == 0:
                if p < 0.5:
                    n = 0
                else:
                    n = 1
            elif j == 19:
                if p < 0.5:
                    n = 1
                else:
                    n = 2
            else:
                if p < 0.333:
                    n = 0
                elif p < 0.666:
                    n = 1
                else:
                    n = 2
        elif i == 16:
            if j == 0:
                if p < 0.5:
                    n = 0
                else:
                    n = 3
            elif j == 19:
                if p < 0.5:
                    n = 3
                else:
                    n = 2
            else:
                if p < 0.333:
                    n = 0
                elif p < 0.666:
                    n = 3
                else:
                    n = 2
        elif j == 0:
            if p < 0.333:
                n = 0
            elif p < 0.666:
                n = 3
            else:
                n = 1
        elif j == 19:
            if p < 0.333:
                n = 1
            elif p < 0.666:
                n = 3
            else:
                n = 2
        else:
            if p < 0.25:
                n = 0
            elif p < 0.5:
                n = 1
            elif p < 0.75:
                n = 2
            else:
                n = 3
        # change self.state according to self.a 
        if n == 0:
            self.a = 0
            self.state = [i, j + 1]
        elif n == 1:
            self.a = 1
            self.state = [i + 1, j]
        elif n == 2:
            self.a = 2
            self.state = [i, j - 1]
        else:
            self.a = 3
            self.state = [i - 1, j]

    # select and move with greedy policy of State Value Function V
    def step_greedy_V(self):
        i = self.state[0]
        j = self.state[1]
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
        # save the index of the max as self.a, and the change the self.state
        if n == 0:
            self.a = 0
            self.state = [i, j + 1]
        elif n == 1:
            self.a = 1
            self.state = [i + 1, j]
        elif n == 2:
            self.a = 2
            self.state = [i, j - 1]
        else:
            self.a = 3
            self.state = [i - 1, j]

    # select and move with greedy policy of Action-State Value Function Q        
    def step_greedy_Q(self):
        i = self.state[0]
        j = self.state[1]
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
        if n == 0:
            self.a = 0
            self.state = [i, j + 1]
        elif n == 1:
            self.a = 1
            self.state = [i + 1, j]
        elif n == 2:
            self.a = 2
            self.state = [i, j - 1]
        else:
            self.a = 3
            self.state = [i - 1, j]

    # select and move with epsilon-greedy policy of State Value Function V
    def step_epsilon_greedy_V(self, epsilon):
        p = np.random.random()
        if p > epsilon:
            self.step_greedy_V()
        else:
            self.step_random()

    # select and move with epsilon-greedy policy of Actiion-State Value Function Q
    def step_epsilon_greedy_Q(self, epsilon):
        p = np.random.random()
        if p > epsilon:
            self.step_greedy_Q()
        else:
            self.step_random()

            
    def stra_evaluation(self, k):
        pass

    def stra_improve(self):
        pass
