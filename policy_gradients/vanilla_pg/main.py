################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani

## main code for training CartPole problem

## v1.0 (REINFORCE, 4/4/18)
################################


from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import os, sys, random, re, argparse

from policy_agent import Policy_Agent

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-a", "--num_actions", type=int, default=2, help="possible agent actions")

## for model to save and load
parser.add_argument("-ld", "--logdir", type=str, default=os.path.join(os.getcwd(), "log"), help="log data directory")
parser.add_argument("-lm", "--load_model", type=bool, default=False, help="load from existing checkpoint")

parser.add_argument("-xdim", "--input_dim", type=int, default=4, help="total dimension of X")
parser.add_argument("-hl", "--num_hidden_layer", type=int, default=2, help="hidden layer dim")
parser.add_argument("-hd", "--hidden_dim", type=int, default=10, help="hidden layer dim")
parser.add_argument("-uf", "--update_freq", type=int, default=5, help="update frequency for training")
parser.add_argument("-bs", "--batch_size", type=int, default=5, help="mini batch size for training")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2, help="learning rate for training")

parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount Factor")
parser.add_argument("-ep", "--total_episodes", type=int, default=10000, help="Total episode number")

parser.add_argument("-mc", "--REINFORCE", type=bool, default=True, help="REINFORCE MC algorithm")
parser.add_argument("--trpo", type=bool, default=False, help="TRPO algorithm")
parser.add_argument("--ppo", type=bool, default=False, help="PPO algorithm")

parameters = parser.parse_args()

if __name__ == "__main__":
    agent = Policy_Agent(parameters)
    agent.train()
