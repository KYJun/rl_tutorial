################################
## Policy Gradient Project

## Editted by YoungJun Kim
## Original code by Arthur Julliani
## Referred from Patriak Emmani

## Main script for parameter setting

## v1.0 (AC appliable, 15/4/18)
## v1.1 (DDPG appliable 16/4/18~)
################################
from __future__ import absolute_import, division, print_function
import os, argparse


## cartpole: 4, 32, 1, 5, 5
## pong: 6400, 256, 0.01, 50, 50

from actor_critic_agent import AC_Agent
from ddpg.ddpg import Ddpg_Agent

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-a", "--num_actions", type=int, default=2, help="possible agent actions")

## for model to save and load
parser.add_argument("-ld", "--logdir", type=str, default=os.path.join(os.getcwd(), "ac_log"), help="log data directory")
parser.add_argument("-lm", "--load_model", type=bool, default=False, help="load from existing checkpoint")

parser.add_argument("-xdim", "--input_dim", type=int, default=4, help="total dimension of X")
parser.add_argument("-hl", "--num_hidden_layer", type=int, default=2, help="hidden layer dim")
parser.add_argument("-hd", "--hidden_dim", type=int, default=16, help="hidden layer dim")
parser.add_argument("-uf", "--update_freq", type=int, default=4, help="update frequency for training")
parser.add_argument("-bs", "--batch_size", type=int, default=3, help="mini batch size for training")
parser.add_argument("-lra", "--learning_rate", type=float, default=0.001, help="learning rate for training")
parser.add_argument("-lrc", "--learning_rate_critic", type=float, default=0.001, help="learning rate for training")

parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount Factor")
parser.add_argument("-ep", "--total_episodes", type=int, default=10000, help="Total episode number")

parser.add_argument("-t", "--tau", type=float, default=0.1, help="target critic graph update rate")
parser.add_argument("--max_buffer_size", type=int, default=3, help="max experience buffer size")
parser.add_argument("-pts", "--preTrainStep", type=int, default=10, help="steps for pre training")
parser.add_argument("--ddpg", type=bool, default=False, help="Apply DDPG network")

parameters = parser.parse_args()

if __name__ == "__main__":
	
	if parameters.ddpg:
		agent = Ddpg_Agent(parameters)
		agent.train()
	
	else:
		agent = AC_Agent(parameters)
		agent.train()