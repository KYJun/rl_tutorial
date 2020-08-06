from __future__ import absolute_import, division, print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os, sys, re, gym, scipy, time
import threading
import multiprocessing

from ac_network import AC_network
from experience_replay import ReplayMemory

class Worker():
    def __init__(self, params, num, global_episodes, tvars, global_network):

        self.params = params
        self.name = "worker_" + str(num)
        self.number = num
        self.model_path = self.params.logdir
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.global_network = global_network

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_network(params, num, tvars, name=self.name)
        self.update_local_ops = self.update_target_graph(tvars, self.local_AC.local_vars)
        
        #The Below code is related to setting up the Doom environment
        self.actions = None

        #load cartpole
        self.env = gym.make('CartPole-v0')
        self.myBuffer = ReplayMemory(max_size=self.params.max_ep_length)


        
    def train(self, sess):
        trainBatch = self.myBuffer.sample(self.total_steps)
        batch_state = np.array(trainBatch[0]).reshape([self.total_steps, self.params.input_dim])
        batch_actions = np.array(trainBatch[1]).reshape([self.total_steps, self.params.num_actions])
        batch_rewards = np.array(trainBatch[2])
        batch_next_state = np.array(trainBatch[3]).reshape([self.total_steps, self.params.input_dim])
        batch_done = np.array(trainBatch[4])

        end_multiplier = -(batch_done - 1)
        
        # Here we take the rewards and values from the buffer, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        #self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        #discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        #self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        #advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        #advantages = discount(advantages,gamma)

        next_Q = np.max(sess.run(self.local_AC.Qout, feed_dict={self.local_AC.input_x:batch_next_state}))
        state_value = np.max(sess.run(self.local_AC.Qout, feed_dict={self.local_AC.input_x:batch_state}))

        batch_target_Q = batch_rewards + (self.params.gamma*next_Q* end_multiplier)
        batch_advantages = batch_target_Q - state_value

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.input_x:batch_state,
            self.local_AC.target_Q:batch_target_Q,
            self.local_AC.actions:batch_actions,
            self.local_AC.advantages:batch_advantages.reshape(self.total_steps, 1)}

        v_l,p_l,e_l, _ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)

        #return v_l/self.total_steps , p_l/self.total_steps , e_l/self.total_steps
        
    def work(self, sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        self.total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = []
                episode_step_count = []
                score = 0
                d = False
                state_input = self.env.reset()
                state_buffer, reward_buffer, action_buffer, next_state_buffer, done_buffer = [], [], [], [], []

                while not d:

                    state_input = state_input.reshape([1, self.params.input_dim])
                    # Run the policy network and get an action to take.
                    curr_policy = sess.run(self.local_AC.probability, feed_dict={self.local_AC.input_x: state_input})

                    # get the action from predicted policy
                    action = np.random.choice(np.arange(len(curr_policy)), p=curr_policy)

                    # step the environment and get new measurements
                    next_state, reward, d, _ = self.env.step(action)

                    next_state = next_state.reshape([1, self.params.input_dim])

                    state_buffer.append(state_input)
                    action_buffer.append([1, 0] if action == 0 else [0, 1])
                    reward_buffer.append(reward if not d or score == 399 else -200)
                    # reward_buffer.append(reward)
                    next_state_buffer.append(next_state)
                    done_buffer.append(d)
                    score += reward
                    self.total_steps += 1

                    state_input = next_state

                self.myBuffer.append(state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer)

                #state_buffer, reward_buffer, action_buffer, next_state_buffer, done_buffer = [], [], [], [], []
                episode_reward.append(score)
                #print(score)

                episode_step_count.append(self.total_steps)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if self.myBuffer != None:
                    #v_l,p_l,e_l = self.train(sess)
                    self.train(sess)
               #     #print(v_l, p_l, e_l)
                    self.update_Target(self.update_local_ops, sess)
                    #print(myBuffer._memory)
                    self.myBuffer.reset()
                    self.total_steps = 0

                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 10 == 0 and episode_count != 0:
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    if self.name == "worker_0":

                        curr_reward = 0

                        for i in range(5):
                            test_done = False
                            state = self.env.reset()
                            while not test_done:
                                state = state.reshape(1, self.params.input_dim)
                                curr_policy = sess.run(self.global_network.probability, feed_dict={self.global_network.input_x: state})

                                # get the action from predicted policy
                                action = np.random.choice(np.arange(len(curr_policy)), p=curr_policy)

                                # step the environment and get new measurements
                                next_state, reward, test_done, _ = self.env.step(action)
                                curr_reward += 1
                                state = next_state

                        print("Episode: {}, Current global reward: {:.1f}".format(episode_count, curr_reward/5))
                        time.sleep(0.5)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

                if episode_count > self.params.total_episodes and self.name == "worker_0":
                    coord.request_stop()


    def update_target_graph(self, from_vars, to_vars):
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


    def update_Target(self, op_holder, sess):
        '''run operation defined in updateTargetGraph function'''
        for op in op_holder:
            sess.run(op)

def run(params):

    tf.reset_default_graph()

    if not os.path.exists(params.logdir):
        os.makedirs(params.logdir)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        #trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        global_network = AC_network(params=params, name='global')  # Generate global network
        global_tvars = tf.trainable_variables()
        #num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
        num_workers = 2
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(params, i, global_episodes, global_tvars, global_network))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if params.load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(params.logdir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            time.sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)


