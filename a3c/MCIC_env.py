# MCIC environment for the A3C algorithm
# using the Abilene network
######################################################
# august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
######################################################
# imports

import random as rd
import math
import numpy as np
import os
from network_graph import *

######################################################
# utility functions

# returns the size of the state
def get_state_size(K_size, C_size, num_nodes, num_edges):
	state_size = 0
	# e(ij)
	state_size = state_size + num_edges
	# u(c)
	state_size = state_size + C_size
	# w(ij, k)
	state_size = state_size + num_edges*K_size
	# c(ij, k)
	state_size = state_size + num_edges*K_size
	# a(n,c)
	state_size = state_size + num_nodes*C_size
	# q(n, c)
	state_size = state_size + num_nodes*C_size

	#print("DEBUG state_size: "+str(state_size))
	return state_size

# returns the size of the action
def get_action_size(K_size, C_size, num_edges):
	action_size = 0
	# alphas[edge][k]
	action_size = action_size + num_edges*K_size
	# betas[edge][k][c]
	action_size = action_size + num_edges*K_size*C_size
	
	#print("DEBUG action_size: "+str(action_size))
	return action_size
	
#######################################################
#MCIC parameters:
# C = set of commodities
C = [1] # for the moment we consider only 1 commodity
# K = set of processing resource allocation choices
K = [0,1]
# self.e[edge]  
# self.u[c] 
# self.w[edge][k]
# self.B[edge][k] 

# variables (vary with solution):
# self.a[c][node_name]
# self.q[c][node_name]
# self.x[edge][c]
# action/solution:
# self.alphas[edge][k]
# self.betas[edge][k][c]

STATE_SIZE = get_state_size(len(K), len(C), NUM_NODES, NUM_EDGES)
ACTION_SIZE = get_action_size(len(K), len(C), NUM_EDGES)

EPISODE_LENGTH = 1000


class MCIC_env():
	def __init__(self):
		#random.seed()
		self.ep_count = 0
		self.step_count = 0
		self.cost_per_iteration = [None for i in range(EPISODE_LENGTH+1)]
		self.constraint_violation_per_iteration = [None for i in range(EPISODE_LENGTH+1)]
		self.new_episode()

	def new_episode(self):
		# define the parameters and variables for the episode

		# parameters (fixed for each episode)
		self.on_off_resource_allocation()
		# variables (change during an episode)
		# default action/solution : all = 0 ? random? does it matter?
		self.alphas = {}
		self.betas = {}
		for edge in edges:
			self.alphas[edge] = {}
			self.betas[edge] = {}
			for k in K:
				rand0or1 = rd.randint(0,1)
				randfloat = rd.random()
				self.alphas[edge][k] = rand0or1
				self.betas[edge][k] = {}
				for c in C:
					self.betas[edge][k][c] = rand0or1*randfloat

		self.x = {}
		for edge in edges:
			self.x[edge] = {}
			for c in C:
					self.x[edge][c] = 0
		
		# update self.q based on self.a and self.x values
		self.update_Qcvt() 
		# reset the steps count
		self.step_count = 0

	def on_off_resource_allocation(self):
		# set the parameters for an episode
		self.e = {}
		self.u = {}
		self.w = {}
		self.B = {}
		for edge in edges:
			self.e[edge] = 1
			self.w[edge] = {}
			self.B[edge] = {}
			for k in K:
				self.w[edge][k] = k*5
				self.B[edge][k] = k*5
		for c in C:
			self.u[c] = 0
		# origin and destination node for each commodity
		self.destination_node = {}
		self.origin_node = {}
		for c in C:
			# TODO: choose randomly, instead of always
			# having 0 as the origin node and 8 as destination node
			self.origin_node[c] = 0
			self.destination_node[c] = 8
		# param. a and variable q (vary with solution)
		self.a = {}
		self.q = {}
		for c in C:
			self.a[c] = {}
			self.q[c] = {}
			for node, node_name in node_names.items():
				self.q[c][node_name] = 0
				if(node == self.origin_node[c]):
					self.a[c][node_name] = 2
				elif(node == self.destination_node[c]):
					self.a[c][node_name] = -2
				else:
					self.a[c][node_name] = 0

	def reset(self):
		# reset the environment : new episode
		# and returns the state

		# temporary: save the data for generating the graphs
		if(self.ep_count > 0):
			self.save_infos_txt()

		self.new_episode()
		self.ep_count += 1
		return self.get_state()

	def get_state(self):
		# returns the state in a 1D array
		state = []
		# add self.e[edge]
		for edge in edges:
			state.append(self.e[edge])
		# add self.u[c]
		for c in C:
			state.append(self.u[c])
		# add self.w[edge][k]
		for edge in edges:
			for k in K:
				state.append(self.w[edge][k])
		# add self.B[edge][k]
		for edge in edges:
			for k in K:
				state.append(self.B[edge][k])
		# add self.a[c][node_name]
		for c in C:
			for node, node_name in node_names.items():
				state.append(self.a[c][node_name])
		# add self.q[c][node_name]
		for c in C:
			for node, node_name in node_names.items():
				state.append(self.q[c][node_name])
		
		#print("DEBUG len(state): "+str(len(state)))
		return state

	def is_action_valid(self,action):
		# verifies if the action is valid. if not, stops the program
		if(len(action) != ACTION_SIZE):
			print("action size ("+str(len(action))+") should be "+str(ACTION_SIZE)+" !!")
			exit(1)
		for i in range(ACTION_SIZE):
			if(action[i] > 1 or action[i] < 0):
				print("action["+str(i)+"] not in the range [0,1] !!")
				exit(1)
			if(math.isnan(action[i])):
				print("action["+str(i)+"] equals Nan !!")
				exit(1)
		return True

	def update_Qcvt(self):
		gamma = 1 # from the Q update formula
		for c in C:
			for node,node_name in node_names.items():
				outgoing_sum = 0
				for out_edge in get_outgoing_edges(node_name):
					outgoing_sum = outgoing_sum + self.x[out_edge][c]
				incoming_sum = 0
				for in_edge in get_incoming_edges(node_name):
					incoming_sum = incoming_sum + self.x[in_edge][c]
				expr = incoming_sum - outgoing_sum
				self.q[c][node_name] = max(0,(self.q[c][node_name] + gamma * (self.a[c][node_name] + expr)))
		#print("DEBUG updated Q: "+str(self.q))
	
	def update_a(self):
		for c in C:
			for node, node_name in node_names.iteritems():
				if(node == self.origin_node[c]):
					self.a[c][node_name] = 2 # information coming in on this node
				elif(node == self.destination_node[c]):
					self.a[c][node_name] = -2 # information going out on this node
				else:
					self.a[c][node_name] = 0

	def update_Xs(self):
		for edge in edges:
			for c in C:
					self.x[edge][c] = 0
					for k in K:
						self.x[edge][c] = self.x[edge][c] + self.alphas[edge][k]*self.betas[edge][k][c]*self.B[edge][k]
		#print("DEBUG Xs: "+str(self.x))

	def update_alphas_and_betas(self,action):
		i = 0 # iterator for the action array
		# get the alpha values
		# (only the K for each edge with the highest value get a 1, the rest get 0)
		for edge in edges:
			for k in K:
				self.alphas[edge][k] = action[i]
				i = i + 1
		for edge in edges:
			max_alpha_k = K[0]
			for k in K:
				if(self.alphas[edge][k] > self.alphas[edge][max_alpha_k]):
					max_alpha_k = k
			for k in K:
				if(k == max_alpha_k):
					self.alphas[edge][k] = 1
				else:
					self.alphas[edge][k] = 0
		#print("***\nDEBUG self.alphas: " + str(self.alphas))

		# get the beta values
		for edge in edges:
			for k in K:
				for c in C:
					self.betas[edge][k][c] = action[i]
					i = i + 1
		#print("DEBUG self.betas: " + str(self.betas))

	def interaction(self,action):
		# advances into new state based on the action and the previous state
		# action : alphas and betas
		# x : calculated based on the new alphas and betas
		# variable part of the state: q, calculated based on x
		if(self.is_action_valid(action)):
			self.update_alphas_and_betas(action)
			self.update_Xs()
			self.update_Qcvt()
		else:
			print("ERROR: invalid action :\n"+str(action))
			exit(1)
	
	def get_reward(self):
		# returns the reward for current state

		#temporary line for testing:
		#return self.reward_idea() # also not working yet
		constraints_violation = self.get_constraints_violation()
		cost = self.cost_function()
		reward = -self.get_augmented_cost(cost,constraints_violation)
		#print("***\nDEBUG cost: " +str(cost)+"  constraints_violation: "+str(constraints_violation))
		#print("DEBUG reward: "+str(reward))
		return reward

	def get_augmented_cost(self,cost,constraints_violation):
		# returns (cost + P * constraints_violation)
		# where P is ideally the upper bound of cost
		P_constant = 10**4 # temporary. TO DO : use upper bound of cost_function

		# store cost and constraint_violation
		# in order to generate graphs later
		self.cost_per_iteration[self.step_count] = cost
		self.constraint_violation_per_iteration[self.step_count] = P_constant*constraints_violation

		return cost + P_constant*constraints_violation

	def is_episode_finished(self):
		# TO DO: does this function need to exist?
		# ... maybe the episode only ends at a fixed timestep
		return False

	def step(self, action):
		#receives the action, updates the environment
		# and returns the next state, the reward and if the episode is over
		self.interaction(action)
		state = self.get_state()
		reward = self.get_reward()
		done = self.is_episode_finished()
		self.step_count = self.step_count+1	# increase ep count
		return state, reward, done

	def get_constraints_violation(self):
		# returns a value that measures how much the
		# current solution violates the constraints

		# constraint 1
		cons1_violation = 0
		for c in C:
			for node, node_name in node_names.items():
				out_expr = 0
				for out_edge in get_outgoing_edges(node_name):
					out_expr = out_expr + self.x[out_edge][c]
				in_expr = 0
				for in_edge in get_outgoing_edges(node_name):
					in_expr = in_expr + self.x[in_edge][c]
				violation = abs(in_expr - out_expr + self.a[c][node_name])
				cons1_violation = cons1_violation + violation
		#print("DEBUG cons1_violation: "+str(cons1_violation))

		# constraint 2
		cons2_violation = 0
		for edge in edges:
			left_expr = 0
			for c in C:
				left_expr = left_expr + self.x[edge][c]
			right_expr = 0
			for k in K:
				right_expr = right_expr + self.alphas[edge][k]*self.B[edge][k]
			cons2_violation = cons2_violation + max(0, left_expr-right_expr)
		#print("DEBUG cons2_violation: "+str(cons2_violation))

		# constraint 3
		cons3_violation = 0
		for edge in edges:
			for k in K:
				left_expr = 0 # left side of the constraint inequality
				for c in C:
					left_expr = left_expr + self.betas[edge][k][c]
				cons3_violation = cons3_violation + max(0, left_expr-1)
		#print("DEBUG cons3_violation: "+str(cons3_violation))

		# constraint 4
		cons4_violation = 0
		for edge in edges:
			left_expr = 0 # right side of the constraint inequality
			for k in K:
				left_expr = left_expr + self.alphas[edge][k]
			cons4_violation = cons4_violation + max(0, left_expr-1)
		#print("DEBUG cons4_violation: "+str(cons4_violation))

		constraints_violation = cons1_violation**2 + cons2_violation**2+ cons3_violation**2+ cons4_violation**2
		#print("DEBUG constraints_violation: "+str(constraints_violation))
		return constraints_violation

	def cost_function(self):
		# returns the value of the function we are trying to minimize
		#  (using the second formulation, with betas and alphas instead of Xs)
		cost = 0
		for k in K:
			for edge in edges:
				expr = 0
				for c in C:
					expr = expr + (self.e[edge]-self.u[c])*self.betas[edge][k][c]
				expr = expr *  (self.B[edge][k])
				expr = expr + self.w[edge][k]
				expr = expr *  (self.alphas[edge][k])
				cost = cost + expr
		#print("DEBUG cost :" + str(cost))
		return cost 

	def save_infos_txt(self):
		# saves the values of cost and P*sum(constraint_violation)**2 of an iteration
		folder_path = "./costs_and_const_viol/"
		if not os.path.exists(folder_path):
			os.makedirs(folder_path) # create folder if it doesnt exist yet

		cost_filename = folder_path+str(self.ep_count-1)+"_costs.txt"
		constraints_violation_filename = folder_path+str(self.ep_count-1)+"_constraints_violation.txt"
		with open(cost_filename, 'w') as outfile_cost:
			with open(constraints_violation_filename, 'w') as outfile_constraint_violation:
				for i in range(EPISODE_LENGTH):
					outfile_cost.write(str(self.cost_per_iteration[i])+'\n')
					outfile_constraint_violation.write(str(self.constraint_violation_per_iteration[i])+'\n')
				outfile_cost.close()
				print("saved "+cost_filename)
				outfile_constraint_violation.close()
				print("saved "+constraints_violation_filename)
