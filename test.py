from tools import getOptions
from tools import qAgent
from tools import argMax2D
from game import omok
import tensorflow as tf
import time
import numpy as np

np.set_printoptions(precision=2)

opt = getOptions()
game = omok(opt)

done = False
rwd = 0

agent = qAgent(opt)
obs, q = agent.valueNet()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

state = game.getState()
while not done:
    act, actBrd = agent.smpAct(q, {obs: [state]})
    print(actBrd)
    state, done, rwd = game.step(act)
    
    print(rwd)