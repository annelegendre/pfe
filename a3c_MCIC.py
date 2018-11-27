# A3C algorithm adapted from the code from Arthur Juliani's Medium post
# to solve the MCIC optimisation problem
###########################
# started in august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
###########################


import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
from random import choice
from time import sleep
from time import time

# local source files
from AC_Network import *
from Worker import *
from MCIC_env import *

# -----------------------
# A3C HYPER-PARAMETERS
max_episode_length = 1000
gamma = .9 # discount rate for advantage estimation and reward discounting
learning_rate = 0.00005
s_size = STATE_SIZE # defined in MCIC_env.py
a_size = ACTION_SIZE # defined in MCIC_env.py
load_model = False
model_path = './model'
# -----------------------


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
        workers.append(Worker(MCIC_env(),i,s_size,a_size,trainer,model_path,global_episodes))
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
