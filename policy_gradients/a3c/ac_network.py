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

class AC_network():

    def __init__(self, params, num=None, tvars=None, name="None"):

        with tf.name_scope("{}".format(name)):

            self.tvars = tvars
            self.num = num
            self.name = name
            self.params = params
            with tf.name_scope("Critic_Forward_propagation"):
                # for forward-propagation #

                # input_x will be current state containing 4 elements each
                self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.params.input_dim], name="states")

                xavier_init = tf.contrib.layers.xavier_initializer()

                fc_layer1 = tf.layers.dense(inputs=self.input_x,
                                           units=self.params.hidden_dim,
                                           activation=tf.nn.relu,
                                           kernel_initializer=xavier_init)


                self.logits = tf.layers.dense(inputs=fc_layer1,
                                            units=self.params.num_actions,
                                            activation=None,
                                            kernel_initializer=xavier_init)

                self.Qout = self.logits
                self.probability = tf.squeeze(tf.nn.softmax(self.logits))

            if self.name != "global":
                start_idx = len(self.tvars) * (self.num + 1)
                self.local_vars = tf.trainable_variables()[start_idx:]
                self._optimize(self.local_vars)

    def _optimize(self, local_vars):

        with tf.name_scope("Critic_Back_propagation"):

            ## for value network update
            self.target_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="target_Q")
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.params.num_actions], name="actions")

            self.predicted_Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions), axis=-1)

            self.td_error = tf.square(tf.reduce_sum(self.target_Q-self.predicted_Q))
            #self.value_loss = 0.5 * tf.reduce_mean(self.td_error)

            self.adam = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)

            #self.value_optimize = self.adam.minimize(self.value_loss)

            ## for policy network update
            self.advantages = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="scoring")

            #self.responsible_outputs = tf.reduce_sum(self.probability * self.actions, [1])
            self.value_loss = tf.reduce_sum(self.td_error)
            self.entropy = - tf.reduce_sum(self.probability * tf.log(self.probability))
            #self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
            #self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            self.nll = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
            self.policy_loss = tf.reduce_mean(tf.multiply(self.nll, self.advantages))
            #self.policy_optimize = self.adam.minimize(self.policy_loss)

            #self.entropy = - tf.reduce_sum(self.probability * tf.log(self.probability))
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            ## for global network update

            #local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            self.gradients = tf.gradients(self.loss, local_vars)
            #self.var_norms = tf.global_norm(local_vars)
            #print(local_vars)
            #grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = self.tvars
            self.apply_grads = self.adam.apply_gradients(zip(self.gradients, global_vars))

