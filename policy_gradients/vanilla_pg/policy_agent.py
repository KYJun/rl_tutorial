################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## Graph and Session for Policy Gradient method

## v1.0 (REINFORCE, 4/4/18)
################################

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from six.moves import xrange
import os, sys, re, gym, time

from policy_network import Policy_network

class Policy_Agent():

    def __init__(self, params):

        self.env = gym.make('CartPole-v0')

        self.params = params
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.policy_net = Policy_network(params)
            if not os.path.exists(self.params.logdir):
                os.mkdir(self.params.logdir)
            self.saver = tf.train.Saver()

        self.state_buffer, self.reward_buffer, self.action_buffer = [], [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 1
        self.rendering = False

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        # initialize with zeros
        discounted_r = np.zeros_like(r)
        # set start value
        running_add = 0

        # add up gamma*r[t] for each step in episode
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.params.gamma + r[t]
            discounted_r[t] = running_add

        return discounted_r

    def train(self):
        # Launch the graph
        with tf.Session(graph=self.graph) as sess:

            # check for TensorFlow checkpoint
            # if exists, restore Variables
            if self.params.load_model:
                ckpt = tf.train.get_checkpoint_state(self.params.logdir)
                print("Reading model parameters...")
                self.saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                print("Initializing all variables...")
                sess.run(tf.global_variables_initializer())

            # Obtain an initial observation of the environment
            self.state = self.env.reset()
            #loss_all = 0

            while self.episode_number <= self.params.total_episodes:

                # Rendering the environment slows things down,
                # so let's only look at it once our agent is doing a good job.

                #if self.reward_sum / self.params.update_freq >= 180 or rendering == True :
                #    self.env.render()
                #    self.rendering = True

                # Make sure the observation is in a shape the network can handle.
                state_input = np.reshape(self.state, [1, self.params.input_dim])

                # Run the policy network and get an action to take.
                curr_policy = sess.run(self.policy_net.probability, feed_dict={self.policy_net.input_x:state_input})

                # get the action from predicted policy
                action = np.random.choice(np.arange(len(curr_policy)), p=curr_policy)
                #print(curr_policy, action)

                # add state info
                self.state_buffer.append(state_input)

                # add action info
                self.action_buffer.append([1, 0] if action==0 else [0, 1])

                # step the environment and get new measurements
                next_state, reward, done, _ = self.env.step(action)

                # move to next state
                self.state = next_state

                # add up reward
                self.reward_sum += reward

                # record reward
                self.reward_buffer.append(reward)

                # if episode is finished
                if done:
                    # add episode
                    self.episode_number += 1

                    # stack together all inputs, actions and reward for each episode
                    episode_states = np.vstack(self.state_buffer)
                    episode_actions = np.vstack(self.action_buffer)
                    episode_rewards = np.vstack(self.reward_buffer)

                    # reset array memory
                    self.state_buffer, self.reward_buffer, self.action_buffer = [], [], []

                    # compute the discounted reward backwards through time
                    discounted_return = self.discount_rewards(episode_rewards)

                    # size the rewards to be unit normal (helps control the gradient estimator variance)
                    # first half would be Good actions (pos) while latter hald would be Bad actions (neg)
                    discounted_return -= np.mean(discounted_return)
                    discounted_return //= np.std(discounted_return)
                    #print(discounted_return)
                    # Get the gradient for this episode, and save it in the gradBuffer
                    # x: states / y: actions / Advantage: discounted reward

                    _, loss = sess.run([self.policy_net.optimize, self.policy_net.loss],
                                       feed_dict={self.policy_net.input_x: episode_states,
                                                  self.policy_net.input_y: episode_actions,
                                                  self.policy_net.advantages: discounted_return})
                    #print(loss)
                    #loss_all += loss

                    if self.episode_number % self.params.update_freq == 0:

                        #print("{:.2f}".format(loss_all / self.params.update_freq))
                        #loss_all = 0

                        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                        print('Current Episode {} Average reward for episode {:.2f}.  Total average reward {:.2f}.'
                              .format(self.episode_number,
                                      self.reward_sum // self.params.update_freq,
                                      self.running_reward // self.params.update_freq))
                        time.sleep(0.5)

                        if self.reward_sum // self.params.update_freq > 200:
                            print("Task solved in", self.episode_number, 'episodes!')
                            break

                        self.reward_sum = 0

                    if self.episode_number % (5 * self.params.update_freq) == 0:
                        self.saver.save(sess, os.path.join(self.params.logdir, "model.ckpt"), global_step=self.episode_number)
                        print("model saved")

                    self.state = self.env.reset()


        print(self.episode_number, 'Episodes completed.')



