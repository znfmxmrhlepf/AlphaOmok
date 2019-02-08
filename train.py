from game import omok
import tools

import tensorflow as tf
import numpy as np

def trainer(env, opt):
    agent = tools.qAgent(opt)
    sess = tf.InteractiveSession()

    obs, Q1 = agent.valueNet()
    nobs, Q2 = agent.valueNet()

    act = tf.placeholder(tf.float32, [None, opt.GAME_SIZE, opt.GAME_SIZE])
    rwd = tf.placeholder(tf.float32, [None, ])

    val1 = tf.reduce_sum(tf.reduce_sum(tf.multiply(Q1, act), 1), 1)
    val2 = rwd + opt.GAMMA * tf.reduce_max(tf.reduce_max(Q2, 1), 1)

    loss = tf.reduce_mean(tf.square(val1-val2))
    trainStep = tf.train.AdamOptimizer(opt.LR).minimize(loss)

    sess.run(tf.global_variables_initializer())
    saver = tools.restore(sess)

    globalStep = 0

    for i_e in range(1, opt.MAX_EPISODE + 1):
        state = env.reset()
        done = False
        sumLoss = 0
        step = 0
        score = 0

        while not done:
            globalStep += 1
            step += 1

            if not globalStep % opt.EPS_STEP and agent.eps > opt.FIN_EPS:
                agent.eps *= opt.EPS_DECAY

            agent.obsMem.push(state)

            action, actBrd = agent.smpAct(Q1, {obs : [state]})
            state, done, reward = env.step(action)
            score += reward

            agent.actMem.push(actBrd)
            agent.nobsMem.push(state)
            agent.rwdMem.push(reward)

            if globalStep >= opt.MEM_SIZE:
                randIdx = np.random.choice(opt.MEM_SIZE, opt.BATCH_SIZE)

                lossVal, _ = sess.run([loss, trainStep], feed_dict={
                                    obs: agent.obsMem.mem[randIdx],
                                    act: agent.actMem.mem[randIdx],
                                    rwd: agent.rwdMem.mem[randIdx],
                                    nobs: agent.nobsMem.mem[randIdx]})

                sumLoss += lossVal

        if not i_e % 20:
            print("====== Episode %d ended with score = %f, avg_loss = %f ======" %(i_e, score, sumLoss / step))
        
        if i_e > opt.MEM_SIZE and not i_e % 100:
            saver.save(sess, 'checkpoints/omok-dqn', global_step=globalStep)

if __name__=='__main__':
    opt = tools.getOptions()
    env = omok(opt)
    trainer(env, opt)