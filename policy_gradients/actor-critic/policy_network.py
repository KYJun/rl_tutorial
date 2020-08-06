################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## Set structure for policy network

## v1.0 (REINFORCE, 4/4/18)
## v1.1 (AC appliable, 15/4/18)
## v1.2 (DDPG appliable 16/4/18~)
################################

from __future__ import absolute_import, division, print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os, sys, re

class Policy_network():

    def __init__(self, params, name="None"):
        with tf.variable_scope("{}".format(name)):

            self.params = params
            with tf.name_scope("Actor_Forward_propagation"):
                # for forward-propagation #

                # input_x will be current state containing 4 elements each
                self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.params.input_dim], name="states")

                xavier_init = tf.contrib.layers.xavier_initializer()

                fc_layer = tf.layers.dense(inputs=self.input_x,
                                           units=self.params.hidden_dim,
                                           activation=tf.nn.relu,
                                           kernel_initializer=xavier_init)


                self.logits = tf.layers.dense(inputs=fc_layer,
                                              units=self.params.num_actions,
                                              activation=None,
                                              kernel_initializer=xavier_init,
                                              name="logits")

                self.probability = tf.squeeze(tf.nn.softmax(self.logits))
                self.det_prob = tf.argmax(self.probability, axis=-1)
                self.adam = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
                if not self.params.ddpg:
                    self.backprop(None)


    def backprop(self, tvars):

        if self.params.ddpg:
          with tf.name_scope("DDPG_Gradient_update"):
            self.action_gradient = tf.placeholder(shape=[None, self.params.num_actions], dtype=tf.float32)
            self.unnormalized_actor_gradients = tf.gradients(self.logits, tvars, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.params.batch_size), self.unnormalized_actor_gradients))

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.params.learning_rate).apply_gradients(zip(self.actor_gradients, tvars))


        else:
          with tf.name_scope("Actor_Back_propagation"):

              self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="actions")
              
              self.advantages = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="scoring")

              self.nll = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
              self.loss = tf.reduce_mean(tf.multiply(self.nll, self.advantages))

              self.optimize = self.adam.minimize(self.loss)


