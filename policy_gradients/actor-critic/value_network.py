################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## Set structure for critic value network

## v1.0 (AC appliable, 15/4/18)
## v1.1 (DDPG appliable 16/4/18~)
################################

from __future__ import absolute_import, division, print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os, sys, re

class Value_network():

    def __init__(self, params, name="None"):

        with tf.name_scope("{}".format(name)):

            self.params = params
            with tf.name_scope("Critic_Forward_propagation"):
                # for forward-propagation #

                # input_x will be current state containing 4 elements each
                self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.params.input_dim], name="states")
                self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.params.num_actions], name="actions")

                xavier_init = tf.contrib.layers.xavier_initializer()

                fc_layer = tf.layers.dense(inputs=self.input_x,
                                           units=self.params.hidden_dim,
                                           activation=tf.nn.relu,
                                           kernel_initializer=xavier_init)

                if self.params.ddpg:
                    fc_layer2 = tf.layers.dense(inputs=fc_layer,
                                                units=self.params.hidden_dim,
                                                activation=tf.nn.relu,
                                                kernel_initializer=xavier_init)

                    fc_layer_a = tf.layers.dense(inputs=self.actions,
                                                 units=self.params.hidden_dim,
                                                 activation=None,
                                                 kernel_initializer=xavier_init,
                                                 use_bias=False)

                    self.Qout = tf.layers.dense(inputs=fc_layer2 + fc_layer_a,
                                                units=self.params.num_actions,
                                                activation=None,
                                                kernel_initializer=xavier_init)
                else:
                    self.Qout = tf.layers.dense(inputs=fc_layer,
                                                units=self.params.num_actions,
                                                activation=None,
                                                kernel_initializer=xavier_init)

            if name=="primary":
                self._optimize()

    def _optimize(self):

        with tf.name_scope("Critic_Back_propagation"):

            self.target_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="target_Q")

            self.predicted_Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions), axis=-1)

            self.td_error = tf.square(tf.reduce_sum(self.target_Q-self.predicted_Q))
            self.loss = tf.reduce_mean(self.td_error)

            self.update_value_model = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate_critic).minimize(self.loss)

            if self.params.ddpg:
                self.action_grads = tf.gradients(self.Qout, self.actions)
