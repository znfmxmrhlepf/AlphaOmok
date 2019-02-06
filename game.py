import numpy as np
import cv2

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
        self.lth = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE, 4], dtype=int)
        self.m = 20 * self.opt.WINDOW_SIZE
        self.img = self.getDefaultImg()
        

    def getDefaultImg(self):
        m = self.m
        imgLth = self.opt.GAME_SIZE * m
        img = np.zeros((imgLth, imgLth, 3), np.uint8)
        img[:] = (153, 204, 255)
        
        for i in range(1, self.opt.GAME_SIZE + 1):
            img = cv2.line(img, (m, m * i), (imgLth - m, m * i), (0, 0, 0), 1)
            img = cv2.line(img, (m * i, m), (m * i, imgLth - m), (0, 0, 0), 1)

        return img

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
        self.img, self.draw = self.getDefaultImg()

    def updateLth(self, act):
        c = act
        maxLth = 0
        sumLth = 1

        for k in range(4):
            self.lth[act[0]][act[1]][k] = 1

        for k in range(4):
            n = [c[0] + omok.di[k], c[1] + omok.dj[k]]
            b = [c[0] - omok.di[k], c[1] - omok.dj[k]]

            if self.safe(n[0], n[1]) and self.board[n[0]][n[1]] == self.turn:
                self.unionRoot(n, c, k)

            if self.safe(b[0], b[1]) and self.board[b[0]][b[1]] == self.turn:
                self.unionRoot(b, c, k)

            lth = self.getLth(act, k)
            sumLth += lth - 1

            if lth > maxLth:
                maxLth = lth

        rwd = sumLth
        done = (maxLth >= self.opt.MAX_LENGTH)
        
        if done:    rwd += 10

        return rwd, done

    def showVal(self):
        print('\n' + '=' * 2 * self.opt.GAME_SIZE)

        for i in range(self.opt.GAME_SIZE):
            for j in range(self.opt.GAME_SIZE):
                print(self.board[i][j], end = ' ')
            print()

        print('=' * 2 * self.opt.GAME_SIZE + '\n')

    def createWindow(self):
        cv2.namedWindow('omok')
        self.showImage()

    def showImage(self):
        cv2.imshow('omok', self.img)
        cv2.waitKey(1)

    def drawStone(self, act):
        m = self.m
        x, y = m * (act[0] + 1), m * (act[1] + 1)
        color = None

        if self.turn == 1: color = (0, 0, 0)
        else: color = (255, 255, 255)

        self.img = cv2.circle(self.img, (x, y), 7 * self.opt.WINDOW_SIZE, color, -1)

    def step(self, act): 
        WrongAct = 0
        
        if self.board[act[0]][act[1]] != 0:
            WrongAct = 1

        self.board[act[0]][act[1]] = self.turn

        if self.opt.SHOW_IMG:
            self.drawStone(act)
            self.showImage()
            
        rwd, done = self.updateLth(act)    
        self.turn *= -1 # change turn
        rwd -= WrongAct * 20

        return done, rwd