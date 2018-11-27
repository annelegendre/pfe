# A3C algorithm adapted from the code from Arthur Juliani's Medium post
# testing with a very simple environment
###########################
# started in august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
###########################


# environment


# nao eh mais esse :
# environment:
# action = binary: take risk or not.
#	not taking a risk = 0 reward
#	taking a risk = s% chance to get positive reward, (100-s)% chance to get a negative one
# state = the percentage of chance of getting a positive reward if actor takes risk
# function (s,a) -> s' :
# taking a risk increases the chance of positive future rewards

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
import gym

# py files
from AC_Network import *
from Worker import *
from Cartpole_env import *

from random import choice
from time import sleep
from time import time


#env = gym.make('CartPole-v0') # AI gym cartpole

# -----------------------

max_episode_length = 1000
gamma = .995 # discount rate for advantage estimation and reward discounting
learning_rate = 0.00009
# s_size = 4 for CartPole_v0
# s_size = 2 for Cartpole_env
# s_size = 100 for 
s_size = 4
a_size = 2 
load_model = False
model_path = './model'

# -----------------------
# main 
# -----------------------

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        #workers.append(Worker(Cartpole_env(),i,s_size,a_size,trainer,model_path,global_episodes))
        workers.append(Worker(gym.make('CartPole-v0') ,i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
