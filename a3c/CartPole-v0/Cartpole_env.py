###########################
# august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
###########################

# primitive version of cartpole environment
# (without velocities and momentum)

import random as rd



# range of the position of the cart
CART_ANGLE_RANGE = [-100, 100]
# falling rate of the pole
GRAVITY = 1.125

class Cartpole_env():
	def __init__(self):
		#random.seed()
		self.ep_count = 0
		self.mean_step_count = 0.0
		self.new_episode()

	def new_episode(self):
		# define the state randomly
		self.pos = rd.gauss(0, 10) # mean = 0, std_dev = 10
		#self.ang = rd.uniform(CART_ANGLE_RANGE[0], CART_ANGLE_RANGE[1])
		self.ang = rd.gauss(0, 10)
		# reset the ep count
		self.step_count = 0

	def reset(self):
		# reset the environment : new episode
		# and returns the state
		self.new_episode()
		return self.get_state()


	def get_state(self):
		#print("DEBUG pos,ang : " + str(self.pos) + " , " + str(self.ang))
		return self.pos, self.ang

	def interaction(self,action):
		# increase ep count
		self.step_count = self.step_count+1
		# advances into new state based on the action and the previous state
		random_value = rd.random() # between 0 and 1
		debug_old_ang = self.ang
		if action == "left":
			#print("DEBUG left")
			#print("DEBUG pos 1: " + str(self.pos))
			self.pos = self.pos - (random_value/10)-1
			#print("DEBUG pos 2: " + str(self.pos))
			self.ang = self.ang  + (random_value/10)+1
		elif action == "stop": 
			#print("DEBUG stop")
			#print("DEBUG ang 1: " + str(self.ang))
			self.ang = GRAVITY * self.ang # the more equilibrated the less it falls
			#print("DEBUG ang 2: " + str(self.ang))
			# pos doesnt change
		elif action == "right":
			#print("DEBUG right") 
			#print("DEBUG pos 1: " + str(self.pos))
			self.pos = self.pos + (random_value/10)+1
			#print("DEBUG pos 2: " + str(self.pos))
			self.ang = self.ang  - (random_value/10)-1
		else:
			#print("ERROR : nonexisting action : " + str(action))
			exit(1)
		#print("DEBUG action : "+str(action)+" angle : "+str(self.ang)+ " -> "+str(debug_old_ang))

	def get_reward(self):
		# get reward for current state
		#return 10/abs(ang) - (abs(pos)*abs(pos))/250
		reward = 0
		abs_ang = abs(self.ang)
		if abs_ang < 2.5:
			reward = reward + (2.5)/abs_ang
		elif abs_ang > 80:
			reward = reward - 5 * (abs_ang/100)*(abs_ang/100)
		#elif abs(self.ang) < 60:
			#reward = reward + 10
		#else:
			#reward = reward - 10

		return reward

	def is_episode_finished(self):
		step_count_batch = 15.0
		if abs(self.ang) > 95:
			if(self.ep_count % step_count_batch == 0):
				print("DEBUG game over, mean_step_count = " + str(self.mean_step_count))
				self.mean_step_count = 0.0
			else:
				self.mean_step_count = self.mean_step_count + self.step_count/step_count_batch
			self.ep_count = self.ep_count+1
			return True
		else:
			return False

	def step(self, action):
		self.interaction(action)
		state = self.get_state
		return state

