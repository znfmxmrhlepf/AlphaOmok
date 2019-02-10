import numpy as np
import cv2

class omok:

    STONE_NONE = 0
    STONE_1 = 1
    STONE_2 = -1 # Input for DQN will be (self.turn * self.board)
                 # which enables to train only one network

    # for dfs
    d = np.array([[1, -1], [0, 1], [1, 1], [1, 0]])

    def __init__(self, opt):
        self.board = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE], dtype=int)
        self.turn = omok.STONE_1 # default
        self.opt = opt
        self.lth = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE, 4], dtype=int)
        self.m = 20 * self.opt.WINDOW_SIZE
        if opt.SHOW_IMG:
            self.img = self.getDefaultImg()
            cv2.namedWindow('omok')
            self.showImage()
        self.stage = 1
        self.past = np.zeros([opt.GAME_SIZE, opt.GAME_SIZE], dtype=float)

    def getDefaultImg(self):
        m = self.m
        imgLth = (self.opt.GAME_SIZE + 1) * m
        img = np.zeros((imgLth, imgLth, 3), np.uint8)
        img[:] = (153, 204, 255)
        
        for i in range(1, self.opt.GAME_SIZE + 2):
            img = cv2.line(img, (m, m * i), (imgLth - m, m * i), (0, 0, 0), 1)
            img = cv2.line(img, (m * i, m), (m * i, imgLth - m), (0, 0, 0), 1)

        return img

    def safe(self, p):
        isSafe = (0 <= p[0] < self.opt.GAME_SIZE and 0 <= p[1] < self.opt.GAME_SIZE)
        return isSafe

    def reset(self):
        self.board = np.zeros([self.opt.GAME_SIZE, self.opt.GAME_SIZE], dtype=int)
        self.turn = omok.STONE_1
        self.lth = np.zeros([self.opt.GAME_SIZE, self.opt.GAME_SIZE, 4], dtype=int)

        if self.opt.SHOW_IMG:
            self.img = self.getDefaultImg()
            self.showImage()

        self.stage = 1
        self.past = np.zeros([self.opt.GAME_SIZE, self.opt.GAME_SIZE], dtype=float)
        
        return self.getState()

    def getStone(self, p):
        return self.board[p[0]][p[1]]

    def updateLth(self, act):
        c = np.array(act)
        maxLth = 0
        sumLth = 1

        rwd = 0
        
        kernel = self.board[max(0, c[0]-2) : min(self.opt.GAME_SIZE, c[0]+3), 
                            max(0, c[1]-2) : min(self.opt.GAME_SIZE, c[1]+3)]
        rwd += (np.sum(np.equal(kernel, self.turn)) - 1) / 2
 
        rms1 = 0
        rms2 = 0

        for k in range(4):
            l = [0, 0]
            
            for i, m in enumerate([-1, 1]):
                n = c + m * omok.d[k]
              
                if self.safe(n) and self.getStone(n) == -1 * self.turn:
                    rms1 += self.lth[n[0]][n[1]][k]**2

                while self.safe(n) and self.getStone(n) == self.turn:
                    l[i] += m
                    n += m * omok.d[k]
      
            L = l[1] - l[0] + 1
            maxLth = max(L, maxLth)
            sumLth += L - 1
            rms2 += (L - 1) ** 2

            for i in range(l[0], l[1] + 1):
                cp = c + i * omok.d[k]
                self.lth[cp[0]][cp[1]][k] = L

        rwd += rms1 ** 0.5 + 2 * rms2 ** 0.5 
                
        done = (maxLth >= self.opt.MAX_LENGTH)
        
        if done:    rwd += 5

        return rwd, done

    def showVal(self):
        print('\n' + '=' * 2 * self.opt.GAME_SIZE)

        for i in range(self.opt.GAME_SIZE):
            for j in range(self.opt.GAME_SIZE):
                print(self.board[i][j], end = ' ')
            print()

        print('=' * 2 * self.opt.GAME_SIZE + '\n')

    def showImage(self):
        cv2.imshow('omok', self.img)
        cv2.waitKey(500)

    def drawStone(self, act):
        m = self.m
        y, x = m * (act[0] + 1), m * (act[1] + 1)
        color = None

        if self.turn == omok.STONE_1: color = (0, 0, 0)
        else: color = (255, 255, 255)

        self.img = cv2.circle(self.img, (x, y), 7 * self.opt.WINDOW_SIZE, color, -1)

    def getState(self):
        curState = np.equal(self.board, self.turn)
        oppState = np.equal(self.board, -1 * self.turn)
        noneState = np.equal(self.board, omok.STONE_NONE)

        state = np.concatenate((np.dstack((self.turn * self.board, curState, oppState, noneState)), 
                        np.multiply(self.lth, np.tile(np.expand_dims(curState, axis=-1), (1, 1, 4))),
                        np.multiply(self.lth, np.tile(np.expand_dims(oppState, axis=-1), (1, 1, 4)))), axis=-1)
                        # feature map
        return state

    def step(self, act): 
        self.past *= self.opt.PAST_DECAY 
        self.past[act[0]][act[1]] = 1

        self.board[act[0]][act[1]] = self.turn

        if self.opt.SHOW_IMG:
            self.drawStone(act)
            self.showImage()
            
        rwd, done = self.updateLth(act)    

        state = self.getState()

        self.turn *= -1
        nstate = self.getState()

        self.stage += 1

        return state, nstate, done, rwd