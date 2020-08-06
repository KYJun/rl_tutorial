# rl_tutorial
reinforcement learning tutorial scripts and notebooks

## Prerequisites
Tensorflow == 1.15.2
gym 

## Notebooks
: Jupyter Notebooks explaining basic network structure
(Reference : Arthur Juliani)
- Q-Learning_table_and_network : Q-table and Q-network to solve Frozen Lake Problem
- Simple-Policy-Gradient : Policy Gradient network to solve CartPole problem
- DQN : Deep Q Network to solve gridworld problem (requires gridworld.py to define environment)

## DQN
: Deep Q Network + Dueling + Double solving gridword problem (environment defined by gridworld.py)
- Double DQN : enhance target Q Value by using result of two networks
- Dueling DQN : seperate Q Value into State-Value and Advantage

## Policy Gradients
: policy gradient network with applications solving cartpole problem
- Vanilla PG : simple REINFORCE algorithm
- DDPG : deep deterministic policy gradient
- actor-critic : Actor network optimizing policy combined with critic network optimizing value
- A3C : Asynchronous Advantage Actor Critic network

## Reference
- https://github.com/awjuliani/DeepRL-Agents
- https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b
- https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5