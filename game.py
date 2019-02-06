import numpy as np

class omok:

    STONE_NONE = 0
    STONE_1 = 1
    STONE_2 = -1 # Input for DQN will be (self.turn * self.board)
                 # which enables to train only one network

    # for dfs
    di = [-1, 0, 1, 1]
    dj = [1, 1, 1, 0]

    def __init__(self, opt):
        self.board = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE], dtype=int)
        self.turn = omok.STONE_1 # default
        self.opt = opt
        self.root = np.array([[[[i, j] for _ in range(4)] for j in range(opt.GAME_SIZE)] for i in range(opt.GAME_SIZE)])
        self.lth = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE, 4], dtype = int)

    def findRoot(self, pos, dir):
        pos = np.array(pos)

        if np.array_equal(self.root[pos[0]][pos[1]][dir], pos):
            return pos
        
        self.root[pos[0]][pos[1]][dir] = self.findRoot(self.root[pos[0]][pos[1]][dir], dir)

        return self.root[pos[0]][pos[1]][dir]

    def getLth(self, pos, dir):
        prt = self.findRoot(pos, dir)

        return self.lth[prt[0]][prt[1]][dir]

    def unionRoot(self, pos1, pos2, dir):
        pos1 = self.findRoot(pos1, dir)
        pos2 = self.findRoot(pos2, dir)

        if dir == 1 or dir == 2 or dir == 3:
            if sum(pos1) > sum(pos2):
                self.root[pos1[0]][pos1[1]][dir] = pos2
                self.lth[pos2[0]][pos2[1]][dir] += self.lth[pos1[0]][pos1[1]][dir]
            else:
                self.root[pos2[0]][pos2[1]][dir] = pos1
                self.lth[pos1[0]][pos1[1]][dir] += self.lth[pos2[0]][pos2[1]][dir]

        else:
            if pos1[0] < pos2[0]:
                self.root[pos2[0]][pos2[1]][dir] = pos1
                self.lth[pos1[0]][pos1[1]][dir] += self.lth[pos2[0]][pos2[1]][dir]
            else:
                self.root[pos1[0]][pos1[1]][dir] = pos2
                self.lth[pos2[0]][pos2[1]][dir] += self.lth[pos1[0]][pos1[1]][dir]

    def safe(self, i, j):
        isSafe = 0 <= i < self.opt.GAME_SIZE and 0 <= j < self.opt.GAME_SIZE
        return isSafe

    def reset(self):
        self.board = np.zeros()
        self.turn = omok.STONE_1

    def updateLth(self, action):
        c = action
        maxLth = 0
        sumLth = 1

        for k in range(4):
            n = [c[0] + omok.di[k], c[1] + omok.dj[k]]
            b = [c[0] - omok.di[k], c[1] - omok.dj[k]]

            if self.safe(n[0], n[1]) and self.board[n[0]][n[1]] == self.turn:
                self.unionRoot(n, c, k)

            if self.safe(b[0], b[1]) and self.board[b[0]][b[1]] == self.turn:
                self.unionRoot(b, c, k)

            lth = self.getLth(action, k)
            sumLth += lth - 1

            if lth > maxLth:
                maxLth = lth

        done = (maxLth >= self.opt.MAX_LENGTH)

        return sumLth, done

    def show(self):
        print('\n' + '=' * 38)

        for i in range(self.opt.GAME_SIZE):
            for j in range(self.opt.GAME_SIZE):
                print(self.board[i][j], end = ' ')
            print()

        print('=' * 38 + '\n')

    def step(self, action): 
        self.board[action[0]][action[1]] = self.turn

        for k in range(4):
            self.lth[action[0]][action[1]][k] = 1
            
        rwd, done = self.updateLth(action)    

        if done: rwd += 10

        self.turn *= -1 # change turn

        return done, rwd