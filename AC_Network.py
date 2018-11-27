# this file containts the definition of the computation graphs
# for the Actor and Critic neural networks
# and the definition of the loss functions
###########################
# started in august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
###########################


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os


from random import choice
from time import sleep
from time import time



#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            h_size_1 = 256 # num of neurons in hidden layer 1
            hidden_1 = slim.fully_connected(self.inputs,h_size_1,activation_fn=None)
            h_size_2 = 256 # num of neurons in hidden layer 2
            hidden_2 = slim.fully_connected(hidden_1,h_size_2,activation_fn=tf.nn.relu)
            h_size_3 = 256 # num of neurons in hidden layer 3
            hidden_3 = slim.fully_connected(hidden_2,h_size_3,activation_fn=tf.nn.relu)
            h_size_4 = 128 # num of neurons in hidden layer 4
            hidden_4 = slim.fully_connected(hidden_3,h_size_4,activation_fn=tf.nn.relu)
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden_4,a_size,
                activation_fn=tf.nn.sigmoid, # all policies must be in the range [0,1]
                weights_initializer=normalized_columns_initializer(0.1),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden_4,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None, a_size],dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.action_mean = tf.reduce_mean(self.actions, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.policy_loss = -tf.reduce_sum(self.advantages * tf.log(self.action_mean))
                self.loss = 0.5 * self.value_loss + self.policy_loss #- self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
