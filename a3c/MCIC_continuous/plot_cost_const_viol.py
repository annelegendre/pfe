######################################################
# august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
######################################################
# script to plot the evolution of the costs and the constraint_violation mesure
# during an execution of the A3C-MCIC
######################################################

#import numpy as np
import matplotlib.pyplot as plt


## parameters: 

#episode to plot:
episode = 0
# plot costs (0) or constraint_violation (1)
plotting = "costs"
#plotting = "constraints_violation"


folder_path = "./costs_and_const_viol/"
cost_path = folder_path + str(episode)+"_costs.txt"
constraints_violation_path = folder_path + str(episode)+"_constraints_violation.txt"
# examples:
# ./costs_and_const_viol/47_constraints_violation.txt
# ./costs_and_const_viol/47_costs.txt


# load file contents
with open(cost_path, 'r') as file:
	    costs_content = file.readlines()
with open(constraints_violation_path, 'r') as file:
	    constraints_violation_content = file.readlines()
# remove `\n` at the end of each line
costs_content = [x.strip() for x in costs_content] 
constraints_violation_content = [x.strip() for x in constraints_violation_content] 

# cast from string to float
costs_data = []
for value in costs_content:
	costs_data.append(float(value))

constraints_violation_data = []
for value in constraints_violation_content:
	constraints_violation_data.append(float(value))

# x axis
x = [i for i in range(len(costs_data))]


# plot graphs
#costs
plt.subplot(211)
plt.plot(x, costs_data)
plt.yscale('linear')
plt.title("costs of episode "+ str(episode))
plt.grid(True)
# constraints_violation
plt.subplot(212)
plt.plot(x, constraints_violation_data)
plt.yscale('linear')
plt.title("constraints_violation of episode "+ str(episode))
plt.grid(True)
plt.show()

#print("DEBUG data: "+ str(data) + " len(data): "+str(len(data)))
