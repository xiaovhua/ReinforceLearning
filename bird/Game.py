import numpy as np
class game():
    def __init__(self):
        # A(s),行动种类
        self.action = ['e', 's', 'w', 'n']
        # 当前选择的的行动
        self.a = 0
        # 初始化终止状态:终点,砖块
        self.endstate_init()
        # 状态,初始为[0, 0]
        self.state = [0, 0]
        # 初始化回报
        self.rewards = np.zeros([17, 20, 4])
        # 初始化值函数
        self.V = np.zeros([17, 20])
        # 初始化行为值函数
        self.Q = np.zeros([17, 20, 4])
        # 决策时用到的策略
        self.policy = np.zeros([17, 20])
        # 折扣函数
        self.gama = 0.98
        # 迭代次数
        self.iteration = 0
        # (s, a)对
        self.s_a = []
        # epsilon-greedy 策略的值
        self.epsilon = 0.25
        # P状态转移矩阵
        self.P = np.ones([17, 20, 4])

    def endstate_init(self):
        # 终点和撞墙为结束
        self.endstate = []
        self.endstate.append([0, 19])
        for i in range(9):
            self.endstate.append([i, 7])
            self.endstate.append([16 - i, 14])
        for i in range(4):
            self.endstate.append([i, 14])
            self.endstate.append([16 - i, 7])

    def step_random(self):
        i = self.state[0]
        j = self.state[1]
        p = np.random.random()
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

    def step_greedy_V(self):
        i = self.state[0]
        j = self.state[1]
        k = []
        if i == 0:
            # 左上
            if j == 0:
                m = max(self.V[i + 1][j], self.V[i][j + 1])
                if m == self.V[j + 1][j]:
                    k.append(0)
                if m == self.V[i + 1][j]:
                    k.append(1)
            # 右上
            elif j == 19:
                m = max(self.V[i + 1][j], self.V[i][j - 1])
                if m == self.V[i + 1][j]:
                    k.append(1)
                if m == self.V[i][j - 1]:
                    k.append(2)
            # 上
            else:
                m = max(self.V[i + 1][j], self.V[i][j + 1], self.V[i][j - 1])
                if m == self.V[i][j + 1]:
                    k.append(0)
                if m == self.V[i + 1][j]:
                    k.append(1)
                if m == self.V[i][j - 1]:
                    k.append(2)
        elif i == 16:
            # 左下
            if j == 0:
                m = max(self.V[i][j + 1], self.V[i - 1][j])
                if m == self.V[i][j + 1]:
                    k.append(0)
                if m == self.V[i - 1][j]:
                    k.append(3)
            # 右下
            elif j == 19:
                m = max(self.V[i - 1][j], self.V[i][j - 1])
                if m == self.V[i][j - 1]:
                    k.append(2)
                if m == self.V[i - 1][j]:
                    k.append(3)
            # 下
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
        # 选定 n 为最大值
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

    def step_greedy_Q(self):
        i = self.state[0]
        j = self.state[1]
        k = []
        if i == 0:
            # 左上
            if j == 0:
                m = max(self.Q[i][j][0], self.Q[i][j][1])
                if m == self.Q[i][j][0]:
                    k.append(0)
                if m == self.Q[i][j][1]:
                    k.append(1)
            # 右上
            elif j == 19:
                m = max(self.Q[i][j][2], self.Q[i][j][1])
                if m == self.Q[i][j][1]:
                    k.append(1)
                if m == self.Q[i][j][2]:
                    k.append(2)
            # 上
            else:
                m = max(self.Q[i][j][2], self.Q[i][j][1], self.Q[i][j][0])
                if m == self.Q[i][j][0]:
                    k.append(0)
                if m == self.Q[i][j][1]:
                    k.append(1)
                if m == self.Q[i][j][2]:
                    k.append(2)
        elif i == 16:
            # 左下
            if j == 0:
                m = max(self.Q[i][j][0], self.Q[i][j][3])
                if m == self.Q[i][j][0]:
                    k.append(0)
                if m == self.Q[i][j][3]:
                    k.append(3)
            # 右下
            elif j == 19:
                m = max(self.Q[i][j][2], self.Q[i][j][3])
                if m == self.Q[i][j][2]:
                    k.append(2)
                if m == self.Q[i][j][3]:
                    k.append(3)
            # 下
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
        # 选定 n 为最大值
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

    def step_epsilon_greedy_V(self, epsilon):
        p = np.random.random()
        if p > epsilon:
            self.step_greedy_V()
        else:
            self.step_random()

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
