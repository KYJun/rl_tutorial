################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## Agent for basic actor-critic algorithm w/ cartpole problem

## v1.0 (AC appliable, 15/4/18)
################################

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from six.moves import xrange
import os, sys, re, gym, time
from value_network import Value_network
from policy_network import Policy_network
from experience_replay import ReplayMemory

class AC_Agent():

    def __init__(self, params):

        self.env = gym.make('CartPole-v0')
        #self.env = gym.make('Pong-v0')

        self.params = params
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.actor = Policy_network(params)

            self.main_critic = Value_network(params, "primary")
            self.target_critic = Value_network(params, "target")

            self.init = tf.global_variables_initializer()

            if not os.path.exists(self.params.logdir):
                os.mkdir(self.params.logdir)

            self.saver = tf.train.Saver()
            self.tvars = tf.trainable_variables()
            main_start_index = int(len(self.tvars)/3)
            target_start_index = int(2*len(self.tvars)/3)
            self.actor_tvars = self.tvars[:main_start_index]
            self.main_critic_tvars = self.tvars[main_start_index:target_start_index]
            self.target_critic_tvars = self.tvars[target_start_index:]
            #self.actor.backprop(tvars=None)

        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        rendering = False
        self.global_step = 0

        self.critic_targetOps = self.update_critic_TargetGraph(self.main_critic_tvars, self.target_critic_tvars, self.params.tau)

        self.myBuffer = ReplayMemory(max_size=self.params.max_buffer_size)


    def update_critic_TargetGraph(self, main_tfVar, target_tfVar, tau):
        '''Holds operation node for assigning Target values to Target network
        Args:
            tfVars - Variables for training(weights, bias...)
            Tau - rate for updating (low Tau value for slow updates)
        Return:
            op_holder - tf.assign() operation. input for updateTarget Function'''
        assert len(main_tfVar) == len(target_tfVar)
        total_vars = len(main_tfVar)
        op_holder = []

        # for latter-half part of trainable variables (= for Target network variables)
        for idx, var in enumerate(main_tfVar[0:total_vars]):
            # assigning tau*new_value+(1-tau)*old_values
            op_holder.append(target_tfVar[idx].assign(
                (var.value() * tau) + ((1 - tau) * target_tfVar[idx].value())))
        return op_holder

    def update_critic_Target(self, op_holder, sess):
        '''run operation defined in updateTargetGraph function'''
        for op in op_holder:
            sess.run(op)


    def _load_model(self, sess, load_ckpt):
        if load_ckpt:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.params.logdir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # initialize gloabl variables
            print('Initialize variables...')
            sess.run(self.init)

    def rendering(self, rendering):
        if self.reward_sum / self.params.update_freq >= 180 or rendering == True :
            self.env.render()
            rendering = True


    def train(self):

        with tf.Session(graph=self.graph) as sess:

            self._load_model(sess, self.params.load_model)
            self.total_episodes = self.params.total_episodes

            # Obtain an initial observation of the environment
            self.state = self.env.reset()
            #state_input = self.prepro(self.state)
            state_input = self.state.reshape([1, self.params.input_dim])

            for self.episode_number in xrange(self.params.total_episodes):

                done = False
                score = 0

                while not done:

                    if self.global_step > self.params.preTrainStep:

                        #print(self.myBuffer)

                        # Value network update
                        trainBatch = self.myBuffer.sample(self.params.batch_size)

                        #print(trainBatch)
                        batch_state = np.array(trainBatch[0]).reshape([self.params.batch_size, self.params.input_dim])
                        batch_actions = np.array(trainBatch[1]).reshape([self.params.batch_size, self.params.num_actions])
                        batch_rewards = np.array(trainBatch[2])
                        batch_next_state = np.array(trainBatch[3]).reshape([self.params.batch_size, self.params.input_dim])
                        batch_done = np.array(trainBatch[4])

                        end_multiplier = -(batch_done - 1)

                        targetQ_all = sess.run(self.target_critic.Qout, feed_dict={self.target_critic.input_x: batch_next_state})
                        targetQ = batch_rewards + (self.params.gamma * np.max(targetQ_all, axis=-1) * end_multiplier)

                        predictedQ_all = sess.run(self.main_critic.Qout, feed_dict={self.main_critic.input_x: batch_state})

                        # Update the network with our target values.
                        sess.run(self.main_critic.update_value_model,
                                                           feed_dict={self.main_critic.input_x : batch_state,
                                                                      self.main_critic.target_Q : targetQ,
                                                                      self.main_critic.actions : batch_actions})
                        self.update_critic_Target(self.critic_targetOps, sess)

                        batch_advantage = batch_rewards + (self.params.gamma * np.max(targetQ_all, axis=-1) * end_multiplier) - np.max(predictedQ_all)
                        # Policy network update
                        batch_advantage = batch_advantage.reshape([self.params.batch_size, 1])
                        sess.run(self.actor.optimize, feed_dict={self.actor.input_x: batch_state,
                                                                 self.actor.input_y: batch_actions,
                                                                 self.actor.advantages: batch_advantage})


                    # Make sure the observation is in a shape the network can handle.
                    state_buffer, reward_buffer, action_buffer, next_state_buffer, done_buffer = [], [], [], [], []

                    #print(state_input.shape)
                    #prev_state = state_input

                    # Run the policy network and get an action to take.
                    curr_policy = sess.run(self.actor.probability, feed_dict={self.actor.input_x: state_input})

                    # get the action from predicted policy
                    action = np.random.choice(np.arange(len(curr_policy)), p=curr_policy)

                    # step the environment and get new measurements
                    next_state, reward, done, _ = self.env.step(action)

                    next_state = next_state.reshape([1, self.params.input_dim])
                    #next_state = self.prepro(next_state)
                    #next_state = next_state - prev_state

                    state_buffer.append(state_input)
                    action_buffer.append([1, 0] if action == 0 else [0, 1])
                    reward_buffer.append(reward if not done or score == 299 else -100)
                    #reward_buffer.append(reward)
                    next_state_buffer.append(next_state)
                    done_buffer.append(done)

                    state_input = next_state

                    # move to next state

                    # add up reward
                    self.reward_sum += reward
                    score += reward
                    self.global_step += 1
                    self.myBuffer.append(state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer)

                if self.episode_number % self.params.update_freq == 0:
                    self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                    print('Current Episode {} Average reward for episode {:.2f}.  Total average reward {:.2f}.'
                          .format(self.episode_number,
                                  self.reward_sum // self.params.update_freq,
                                  self.running_reward // self.params.update_freq))
                    self.reward_sum = 0
                    time.sleep(0.5)


                self.state = self.env.reset()
                state_input = self.state.reshape([1, self.params.input_dim])
                #state_input = self.prepro(self.state)
                self.global_step += 1