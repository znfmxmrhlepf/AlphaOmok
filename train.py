from game import omok
import tools

import tensorflow as tf
import numpy as np

def trainer(env, opt):

    p = [tools.qAgent(opt), tools.qAgent(opt)]
    sess = tf.InteractiveSession()

    obs = [None, None]
    nobs = [None, None]
    Q1 = [None, None]
    Q2 = [None, None]
    act = [None, None]
    rwd = [None, None]
    val1 = [None, None]
    val2 = [None, None]
    loss = [None, None]
    trainStep = [None, None]
    
    for i, agent in enumerate(p):
        obs[i], Q1[i] = agent.valueNet()
        nobs[i], Q2[i] = agent.valueNet()

        act[i] = tf.placeholder(tf.float32, [None, opt.GAME_SIZE, opt.GAME_SIZE])
        rwd[i] = tf.placeholder(tf.float32, [None, ])

        val1[i] = tf.reduce_sum(tf.reduce_sum(tf.multiply(Q1[i], act[i]), -1), -1)
        val2[i] = rwd[i] + opt.GAMMA * tf.reduce_max(tf.reduce_max(Q2[i], -1), 1)

        loss[i] = tf.reduce_mean(tf.square(val1[i]-val2[i]))
        trainStep[i] = tf.train.AdamOptimizer(opt.LR).minimize(loss[i])

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

            i = step % 2
            oi = (i+1)//2

            agent = p[i]
            oppAgent = p[oi]

            if not (globalStep // 2 + 1) % opt.EPS_STEP and agent.eps > opt.FIN_EPS:
                agent.eps *= opt.EPS_DECAY

            action, actBrd = agent.smpAct(Q1[i], {obs[i] : [state]}, env.board)
            state, nstate, done, reward = env.step(action)
            score += reward

            if step > 1:
                oppAgent.nobsMem.push(nstate)

                if globalStep >= opt.MEM_SIZE:
                    randIdx = np.random.choice(opt.MEM_SIZE, opt.BATCH_SIZE)
                    lossVal, _ = sess.run([loss[oi], trainStep[oi]], feed_dict={
                                    obs[oi]: oppAgent.obsMem.mem[randIdx],
                                    act[oi]: oppAgent.actMem.mem[randIdx],
                                    rwd[oi]: oppAgent.rwdMem.mem[randIdx],
                                    nobs[oi]: oppAgent.nobsMem.mem[randIdx]})

                    sumLoss += lossVal

                

            if not done:
                agent.obsMem.push(state)
                agent.actMem.push(actBrd)
                agent.rwdMem.push(reward)

            
        if not i_e % 1:
            print("====== Episode %d ended with score = %f, avg_loss = %f ======" %(i_e, score, sumLoss / step))
        
        if i_e > opt.MEM_SIZE and not i_e % 100:
            saver.save(sess, 'checkpoints/omok-dqn', global_step=globalStep)

if __name__=='__main__':
    opt = tools.getOptions()
    env = omok(opt)
    trainer(env, opt)