from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from tqdm import tqdm
#%matplotlib inline

from gridworld import gameEnv

env = gameEnv(partial=False,size=5)

class Qnetwork():
    def __init__(self,h_size, Target=False):
        
        #h_size: size of final convolution layer connected to fc_layers
        
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        with tf.name_scope("{}".format("Target" if Target else "Primary")):
            with tf.name_scope("Feature_Extraction"):
                xavier_init = tf.contrib.layers.xavier_initializer()
                #get state as one-hot vector
                self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32, name="state")
                #reshape image to 84x84 with 3 channel
                self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])

                ## Value of each k_size, channel, stride is from original code
                ## Can be altered for better performance

                # channel num from input_state to final conv 4 (from 3 to h_size)
                channel_list = [3, 32, 64, 128, h_size]

                # k_size / stride for max pooling
                k_size_list=[8, 4, 3, 7]
                stride_list = [4, 4, 3, 2]

                result = self.imageIn #rename for loop

                ## through 4 layers of convolution (valid padding) 
                # conv1 output: batch_size x 21 x 21 x 32
                # conv2 output: bath_size x 6 x 6 x 64
                # conv3 output: batch_size x 2 x 2 x 128
                # conv4 output: batch_size x 1 x 1 x 512 (final)

                for i in range(4):
                    j = i+1 # for next time_step

                    with tf.name_scope("conv_layer_{}".format(j)):

                        W = tf.Variable(name="conv_w", 
                                        initial_value=xavier_init([k_size_list[i], k_size_list[i], 
                                                                   channel_list[i], channel_list[j]]), 
                                        dtype=tf.float32)
                        b = tf.Variable(name="conv_b", 
                                        initial_value=xavier_init([channel_list[j]]), 
                                        dtype=tf.float32)

                        # convolution stage: SAME padding with W 
                        conv = tf.nn.conv2d(result, 
                                            W, 
                                            strides=[1, 1, 1, 1], 
                                            padding="SAME", 
                                            name="conv")

                        # add bias and activate with Re-LU function
                        h_out = tf.nn.relu(tf.nn.bias_add(conv, b), 
                                           name="h_out")

                        # max_pooling with pre-defined values
                        h_pool = tf.nn.max_pool(value=h_out, 
                                                ksize=[1, 2, 2, 1], 
                                                strides=[1, stride_list[i], stride_list[i], 1], 
                                                padding="SAME",
                                                name="pool")

                        result = h_pool # rename variables for loop

                        tf.summary.histogram("conv_w_{}".format(j), W)
                        tf.summary.histogram("conv_b_{}".format(j), b)
                        tf.summary.histogram("h_pool_{}".format(j), h_pool)

                self.conv4 = result # rename for fc layer computation

            with tf.name_scope("Dueling_fc_layer"):
            # We take the output from the final convolutional layer 
            # and split it into separate advantage and value streams.

                # split final output into 2 by channel_num (batch_sizex1x1x256 each)
                self.streamAC,self.streamVC = tf.split(self.conv4,2,3)

                # A, V flattened to change to 2D tensor (batch_sizex256)
                self.streamA = slim.flatten(self.streamAC)
                self.streamV = slim.flatten(self.streamVC)

                # dimension for weight of A: [256, 4]
                # dimension for weight of V: [256, 1]
                self.AW = tf.Variable(xavier_init([h_size//2,env.actions]), name="weight_A")
                self.VW = tf.Variable(xavier_init([h_size//2,1]), name="weight_V")

                # matrix multiplication to get Advantage and State-Value
                self.Advantage = tf.matmul(self.streamA,self.AW, name="Advantage")
                self.Value = tf.matmul(self.streamV,self.VW, name="Value")

                # Then combine them together to get our final Q-values.
                self.Qout = tf.add(self.Value, 
                                   tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True)), 
                                   name="predicted_Q")
                # predicted action out of Q-value
                self.predict = tf.argmax(self.Qout,1, name="predicted_action")

                # tensorboard summary
                tf.summary.histogram("Advantage_Weight", self.AW)
                tf.summary.histogram("Value_Weight", self.VW)
                tf.summary.histogram("Advantage", self.Advantage)
                tf.summary.histogram("Value", self.Value)
                tf.summary.histogram("Predicted_Q", self.Qout)

            if not Target:
                with tf.name_scope("Target_Network"):
                    #Below we obtain the loss by taking the sum of squares difference 
                    #between the target and prediction Q values.
                    self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32, name="target_Q")
                    self.actions = tf.placeholder(shape=[None],dtype=tf.int32, name="target_action")
                    self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)

                    # Q value from target network
                    self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
                    tf.summary.histogram("Target_Q", self.Q)

                with tf.name_scope("Optimizer"):
                    # MSE for calculating Loss between Target Q and Predicted Q
                    self.td_error = tf.square(self.targetQ - self.Q)
                    self.loss = tf.reduce_mean(self.td_error, name="Loss")

                    # Adam optimizer for updating the Model
                    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001, name="optimizer")
                    self.updateModel = self.trainer.minimize(self.loss, name="update")
                    tf.summary.scalar("Loss", self.loss) 
                    

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        '''set buffers for experience
        Args: 
            buffer_size - how many experience to hold at given timestep'''
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        '''add experience to experience buffer'''
        
        # if buffer is full, delete the old experience
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            
        # append [state, action, reward, next_state, done] tuple into experience buffer
        self.buffer.extend(experience)
            
    def sample(self,size):
        '''sample experience from buffer according to batch size'''
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
        

def processState(states):
    '''reshaping states into 1D tensor'''
    return np.reshape(states,[21168])
    
def updateTargetGraph(tfVars,tau):
    '''Holds operation node for assigning Target values to Target network
    Args:
        tfVars - Variables for training(weights, bias...)
        Tau - rate for updating (low Tau value for slow updates)
    Return:
        op_holder - tf.assign() operation. input for updateTarget Function'''
    total_vars = len(tfVars)
    op_holder = []
    
    # for latter-half part of trainable variables (= for Target network variables)
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        # assigning tau*new_value+(1-tau)*old_values
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    '''run operation defined in updateTargetGraph function'''
    for op in op_holder:
        sess.run(op)
        

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
gamma = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action (epsilon)
endE = 0.1 #Final chance of random action (epsilon)
annealing_steps = 1000. #How many steps of training to reduce startE to endE.
num_episodes = 1000 #How many episodes of game environment to train network with.
pre_train_steps = 1000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn_logss" #The path to save our model checkpoint & train model summary
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

#main_graph = tf.Graph()

#with main_graph.as_default():
tf.reset_default_graph()
#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

# set network for main model
mainQN = Qnetwork(h_size)

# set network for target Q-value computation
targetQN = Qnetwork(h_size, True)

# initialize the varibales
init = tf.global_variables_initializer()

# saver for model
saver = tf.train.Saver()

writer = tf.summary.FileWriter(logdir=path)
summary = tf.summary.merge_all()

# get all the trainable variables
trainables = tf.trainable_variables()

# update operation for target graph, ratio for update is defined as tau
targetOps = updateTargetGraph(trainables,tau)

# set default buffer to hold experiences of episodes
myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
reward_List = []
total_steps = 0

with tf.Session() as sess:
    # initialize
    sess.run(init)
    global_step=0
    writer.add_graph(sess.graph)
    
    # load model if checkpoint exists
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    # training
    for i in tqdm(range(num_episodes), desc='epoch'):
        
        # re-assign current epsiode buffer
        episodeBuffer = experience_buffer()
        
        # Reset environment and get first new observation
        state = env.reset()
        state = processState(state)
        done = False
        reward_All = 0
        j = 0
        
        #The Q-Network
        #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
        while j < max_epLength: 
            j+=1
            global_step+=1
            
            # Epsilon-Greedy Approach
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                action = np.random.randint(0,4)
            else:
                action = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[state]})[0]
            
            # get next state, reward, and done from environment
            next_state, reward, done = env.step(action)
            next_state = processState(next_state)
            total_steps += 1
            
            #Save the experience to our episode buffer.
            episodeBuffer.add(np.reshape(np.array([state, action, reward, next_state, done]),[1,5])) 
            
            # annealing epsilon
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                # every update_freq steps, do training from experience buffer
                if total_steps % (update_freq) == 0:
                    
                    #Get a random batch of experiences.
                    trainBatch = myBuffer.sample(batch_size) 
                    
                    #Below we perform the Double-DQN update to the target Q-values
                    predicted_Q = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    target_Q = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    
                    # check if episode ended
                    # if end, set future discounted reward to 0
                    end_multiplier = -(trainBatch[:,4] - 1)
                    
                    # double Q network - Action from main model, QValue from target model 
                    doubleQ = target_Q[range(batch_size),predicted_Q]
                    targetQ = trainBatch[:,2] + (gamma * doubleQ * end_multiplier)

                    #Update the network with our target values.
                    _, curr_sum = sess.run([mainQN.updateModel, summary],\
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, 
                                   mainQN.actions:trainBatch[:,1], targetQN.scalarInput:np.vstack(trainBatch[:,0])})
                    writer.add_summary(curr_sum, global_step=global_step)
                    
                    # update main QN summary
                    
                    #Update the target network toward the primary network.
                    updateTarget(targetOps,sess)
            
            # save reward
            reward_All += reward
            # move to the next state
            state = next_state

            
            # if episode is done, break the loop
            if done == True:
                break
        
        # update experience buffer of the whole episode to main buffer
        myBuffer.add(episodeBuffer.buffer)
        
        jList.append(j)
        reward_List.append(reward_All)
        
        #Periodically save the model. 
        if i % 100 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(reward_List) % 10 == 0:
            print("Total Steps:{}, Mean Rewards:{}, Epsilon:{}".format(total_steps,np.mean(reward_List[-10:]), e))
    
    # save at the last episode
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
    
print("Percent of succesful episodes: " + str(sum(reward_List)/num_episodes) + "%")
