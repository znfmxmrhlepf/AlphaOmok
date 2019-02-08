import argparse
import tensorflow as tf
import numpy as np
import random
import time

random.seed(time.time())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getOptions():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--GAME_SIZE", type=int, default=19, help="size of board")
    parser.add_argument("--MAX_LENGTH", type=int, default=5, help="max length of stone")
    parser.add_argument("--SHOW_IMG", type=str2bool, default=False)
    parser.add_argument("--WINDOW_SIZE", type=int, default=1, help="size of window. ex) 1, 2, 3, ...")
    parser.add_argument("--MAX_MEM_SIZE", type=int, default=2000)
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--PAST_DECAY", type=float, default=0.75)

    parser.add_argument("--OBS_DIM", type=int, default=12)
    
    parser.add_argument("--H1_SIZE", type=int, default=48)
    parser.add_argument("--H2_SIZE", type=int, default=96)
    parser.add_argument("--H3_SIZE", type=int, default=96)
    parser.add_argument("--H4_SIZE", type=int, default=96)
    parser.add_argument("--H5_SIZE", type=int, default=48)
    parser.add_argument("--H6_SIZE", type=int, default=1)

    #KERNEL SIZE
    parser.add_argument("--K1_SIZE", type=int, default=7)
    parser.add_argument("--K2_SIZE", type=int, default=5)
    parser.add_argument("--K3_SIZE", type=int, default=5)
    parser.add_argument("--K4_SIZE", type=int, default=3)
    parser.add_argument("--K5_SIZE", type=int, default=3)
    parser.add_argument("--K6_SIZE", type=int, default=1)

    parser.add_argument("--INI_EPS", type=float, default=1)
    parser.add_argument("--FIN_EPS", type=float, default=1e-4)
    parser.add_argument("--EPS_DECAY", type=float, default=0.95)
    parser.add_argument("--EPS_STEPS", type=int, default=10)

    options = parser.parse_args()

    return options

def argMax2D(x):
    am = np.argmax(x.flatten())
    return [am // x.shape[1], am % x.shape[1]]

class memory:

    def __init__(self, shape):
        self.shape = shape
        self.mem = np.empty(self.shape)
        self.pointer = 0

    def push(self, val):
        self.mem[self.pointer] = val
        
        if self.pointer == self.shape[0]:
            self.pointer = 0

class agent:
    def __init__(self, opt):
        self.obsMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.actMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.rwdMem = memory([opt.MAX_MEM_SIZE, ])
        self.nobsMem = self.obsMem = memory([opt.MAX_MEM_SIZE, opt.GAME_SIZE, opt.GAME_SIZE])
        self.opt = opt
        self.eps = opt.INI_EPS

    def valueNet(self):
        opt = self.opt
        obs = tf.placeholder(tf.float32, [None, opt.GAME_SIZE, opt.GAME_SIZE, opt.OBS_DIM])

        s = [1, 1, 1, 1]
        K = [None, opt.K1_SIZE, opt.K2_SIZE, opt.K3_SIZE, opt.K4_SIZE, opt.K5_SIZE, opt.K6_SIZE]
        L = [opt.OBS_DIM, opt.H1_SIZE, opt.H2_SIZE, opt.H3_SIZE, opt.H4_SIZE, opt.H5_SIZE, opt.H6_SIZE]


        W = [None for _ in range(7)]
        b = [None for _ in range(7)]
        h = [None for _ in range(7)]
        h[0] = obs

        for i in range(1, 7):
            W[i] = tf.Variable(tf.truncated_normal(shape=[K[i], K[i], L[i-1], L[i]], stddev=5e-2))
            b[i] = tf.Variable(tf.truncated_normal(shape=[L[i]], stddev=5e-2))
            h[i] = tf.nn.relu(tf.nn.conv2d(h[i-1], W[i], strides=s, padding='SAME') + b[i])

        return obs, tf.squeeze(h[6])

    # Epsilon decay
    def smpAct(self, Q, feed):
        if random.random() <= self.eps:
            return [random.randrange(self.opt.GAME_SIZE), random.randrange(self.opt.GAME_SIZE)]
        
        else:
            return argMax2D(Q.eval(feed_dict=feed))

        # action = np.zeros([self.opt.GAME_SIZE, self.opt.GAME_SIZE], dtype=int)
        # action[i][j] = 1
        # return action