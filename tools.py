import argparse
import tensorflow as tf
import numpy as np

def getOptions():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--GAME_SIZE", type=int, default=19, help="size of board")
    parser.add_argument("--MAX_LENGTH", type=int, default=5, help="max length of stone")
    parser.add_argument("--SHOW_IMG", type=bool, default=True)
    parser.add_argument("--WINDOW_SIZE", type=int, default=1, help="size of window. ex) 1, 2, 3, ...")
    parser.add_argument("--MAX_MEM_SIZE", type=int, default=2000)
    parser.add_argument("--BATCH_SIZE", type=int, default=128)

    options = parser.parse_args()

    return options

class memory:

    def __init__(self, shape):
        self.shape = shape
        self.mem = np.empty(self.shape)
        self.pointer = 0

    def insert(self, val):
        self.mem[self.pointer] = val
        
        if self.pointer == self.shape[0]:
            self.pointer = 0

class agent:
    def __init__(self, nn, opt):
        self.nn = nn
        self.obsMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.actMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.rwdMem = memory([opt.MAX_MEM_SIZE, ])
        self.nobsMem = self.obsMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.opt = opt

    def valueNet(self):
        obs = tf.placeholder(tf.float32, [None, self.opt.GAME_SIZE. opt.GAME_SIZE, 8])

        W_conv1 = tf.Variable(tf.truncated_normal)

    def smpAct(self, feed):
        pass