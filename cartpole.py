import tensorflow as tf
import numpy as np
import gym
from dqn import dqn

def add_dense(inputs, output_size, activation=None, kernel_initializer=None, name="default"):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    with tf.variable_scope(name, reuse=True):
        tf.summary.histogram("kernel", tf.get_variable("kernel"))
        tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def add_dense2(inputs, output_size, activation=None, kernel_initializer=None, name="default"):
    with tf.variable_scope(name):
        kernel = tf.get_variable("kernel", shape=[inputs.shape[1], output_size], initializer=kernel_initializer)
        bias = tf.get_variable("bias", shape=[output_size], initializer=tf.zeros_initializer())
        tf.summary.histogram("kernel", kernel)
        tf.summary.histogram("bias", bias)
        layer = tf.matmul(inputs, kernel) + bias
        if activation == None:
            return layer
        else:
            return activation(layer)

def playgame(model, env):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a, _ = model.getAction(s)
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def train(model, sess, env, num_episodes):
    score_tensor = tf.placeholder(tf.int32, shape=[])
    tf.summary.scalar("loss", model.loss)
    summaries = tf.summary.merge_all()
    score_summary = tf.summary.scalar("score", score_tensor)
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    total_train = 0
    sum = 0
    for i in range(num_episodes):
        step_count = 0
    
        s = env.reset()
        done = False

        while not done:
            s_lst = []
            a_lst = []
            r_lst = []
            done_lst = []

            action = model.get_action(s)
            ns, reward, done, _ = env.step(action)
            s_lst.append(s)
            a_lst.append(action)
            r_lst.append(reward / 100.0)
            done_lst.append(0 if done else 1)
            step_count += 1
            s = ns
            s_lst.append(ns)
            summary_data = model.train_batch(s_lst, a_lst, r_lst, done_lst, summaries)
            writer.add_summary(summary_data, total_train)
            total_train += 1

        score_summary_data = sess.run(score_summary, feed_dict={score_tensor:step_count})
        writer.add_summary(score_summary_data, i)
        sum += step_count
        if i % 20 == 19:
            print("Episode: {} Step: {}".format(i, sum / 20))
            sum = 0
    writer.close()

def getname(depth, hidden_size, learning_rate):
    return f"depth_{depth}_hs_{hidden_size}_lr_{learning_rate}"

def main():
    environment = gym.make('CartPole-v1')
    input_size = environment.observation_space.shape[0]
    output_size = environment.action_space.n
    num_episodes = 1000

    for lr in range(1, 6, 4):
        for dp in range(2, 4):
            for hidden_size in range(32, 65, 16):
                with tf.Session() as sess:
                    name = getname(dp, hidden_size, lr)
                    with tf.variable_scope(name):
                        input = tf.placeholder(tf.float32, [None, input_size], name="input")
                        network = input
                        for i in range(dp):
                            network = add_dense2(network, hidden_size, activation=tf.nn.relu, name=f"dense{i}")
                        network = add_dense(network, output_size, activation=tf.nn.relu, name="q_dense")
                        model = dqn(sess, input, network, epsilon=0.1, learning_rate=lr * 0.0001, name=name)
                        sess.run(tf.global_variables_initializer())
                    train(model, sess, environment, num_episodes)
                tf.reset_default_graph()
    environment.close()

main()