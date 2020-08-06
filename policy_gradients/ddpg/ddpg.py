################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## Agent for deep deterministic policy gradient algorithm 
# w/ cartpole problem

## v1.0 (DDPG appliable 16/4/18~)
################################

from __future__ import absolute_import, division, print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os, gym, time, re, sys
from value_network import Value_network
from policy_network import Policy_network
from experience_replay import ReplayMemory
from noise import OrnsteinUhlenbeckActionNoise

class Ddpg_Agent():

    def __init__(self, params):

        self.env = gym.make('CartPole-v0')
        self.params = params
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.main_actor = Policy_network(params, "primary")
            tvars = tf.trainable_variables()
            tact_start_index = int(len(tvars))

            self.target_actor = Policy_network(params, "target")
            tvars = tf.trainable_variables()
            mcri_start_index = int(len(tvars))

            self.main_critic = Value_network(params, "primary")
            tvars = tf.trainable_variables()
            tcri_start_index = int(len(tvars))

            self.target_critic = Value_network(params, "target")

            self.tvars = tf.trainable_variables()

            self.main_actor_tvars = self.tvars[:tact_start_index]
            self.target_actor_tvars = self.tvars[tact_start_index:mcri_start_index]
            self.main_critic_tvars = self.tvars[mcri_start_index:tcri_start_index]
            self.target_critic_tvars = self.tvars[tcri_start_index:]

            self.main_actor.backprop(self.main_actor_tvars)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        if not os.path.exists(self.params.logdir):
            os.mkdir(self.params.logdir)

        self.myBuffer = ReplayMemory(max_size=self.params.max_buffer_size)
        self.running_reward = None
        self.reward_sum = 0
        self.global_step = 0

        self.actor_targetOps = self.update_TargetGraph(self.main_actor_tvars,
                                                       self.target_actor_tvars,
                                                       self.params.tau)
        self.critic_targetOps = self.update_TargetGraph(self.main_critic_tvars,
                                                        self.target_critic_tvars,
                                                        self.params.tau)

    def update_TargetGraph(self, main_tfVar, target_tfVar, tau):
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

    def update_Target(self, op_holder, sess):
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


    def train(self):
        
        with tf.Session(graph=self.graph) as sess:

            self._load_model(sess, self.params.load_model)
            self.total_episodes = self.params.total_episodes

            # Obtain an initial observation of the environment
            state = self.env.reset()
            state_input = state.reshape([1, self.params.input_dim])

            for episode_number in xrange(self.params.total_episodes):

                done = False
                score = 0

                while not done:

                    if self.global_step > self.params.preTrainStep:

                        # Value network update
                        trainBatch = self.myBuffer.sample(self.params.batch_size)

                        batch_state = np.array(trainBatch[0]).reshape([self.params.batch_size, self.params.input_dim])
                        batch_actions = np.array(trainBatch[1]).reshape([self.params.batch_size, self.params.num_actions])
                        batch_rewards = np.array(trainBatch[2])
                        batch_next_state = np.array(trainBatch[3]).reshape([self.params.batch_size, self.params.input_dim])
                        batch_done = np.array(trainBatch[4])

                        end_multiplier = -(batch_done - 1)

                        target_action = sess.run(self.target_actor.det_prob, feed_dict={self.target_actor.input_x : batch_next_state})
                        target_action = np.array([[1, 0] if i == 0 else [0, 1] for i in target_action])
                        targetQ_all = sess.run(self.target_critic.Qout,
                                               feed_dict={self.target_critic.input_x: batch_next_state,
                                                          self.target_critic.actions: target_action})
                        nextQ = np.sum(np.multiply(targetQ_all, target_action), axis=-1)
                        targetQ = batch_rewards + (self.params.gamma * nextQ * end_multiplier)

                        pred_actions = sess.run(self.main_actor.det_prob,
                                                feed_dict={self.main_actor.input_x: batch_state})
                        pred_actions = np.array([[1, 0] if i == 0 else [0, 1] for i in pred_actions])

                        # Update the network with our target values.
                        sess.run(self.main_critic.update_value_model,
                                                           feed_dict={self.main_critic.input_x : batch_state,
                                                                      self.main_critic.target_Q : targetQ,
                                                                      self.main_critic.actions : batch_actions})
                        self.update_Target(self.critic_targetOps, sess)

                        gradients = sess.run(self.main_critic.action_grads, feed_dict={self.main_critic.input_x : batch_state,
                                                                                       self.main_critic.actions : pred_actions})

                        gradients = np.array(gradients).reshape(self.params.batch_size, self.params.num_actions)
                        sess.run(self.main_actor.optimize, feed_dict={self.main_actor.input_x : batch_state,
                                                                      self.main_actor.action_gradient : gradients})

                        self.update_Target(self.actor_targetOps, sess)



                    # Make sure the observation is in a shape the network can handle.
                    state_buffer, reward_buffer, action_buffer, next_state_buffer, done_buffer = [], [], [], [], []


                    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.params.num_actions))

                    action = sess.run(self.main_actor.logits, feed_dict={self.main_actor.input_x:state_input}) + actor_noise()
                    action = np.argmax(action)

                    # step the environment and get new measurements
                    next_state, reward, done, _ = self.env.step(action)

                    next_state = next_state.reshape([1, self.params.input_dim])

                    state_buffer.append(state_input)
                    action_buffer.append([1, 0] if action == 0 else [0, 1])
                    reward_buffer.append(reward if not done or score == 299 else -100)
                    #reward_buffer.append(reward)
                    next_state_buffer.append(next_state)
                    done_buffer.append(done)

                    # move to next state
                    state_input = next_state

                    # add up reward
                    self.reward_sum += reward
                    score += reward
                    self.global_step += 1
                    self.myBuffer.append(state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer)

                if episode_number % self.params.update_freq == 0:
                    self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                    print('Current Episode {} Average reward for episode {:.2f}.  Total average reward {:.2f}.'
                          .format(episode_number,
                                  self.reward_sum // self.params.update_freq,
                                  self.running_reward // self.params.update_freq))
                    self.reward_sum = 0
                    time.sleep(0.5)

                self.state = self.env.reset()
                state_input = self.state.reshape([1, self.params.input_dim])
                self.global_step += 1




