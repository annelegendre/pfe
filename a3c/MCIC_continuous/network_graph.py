######################################################
# august 2018
# Pedro FOLETTO PIMENTA
# laboratory I3S
######################################################
# Abilene network
# to use with A3C-MCIC system
######################################################


# utility functions:

# returns all outgoing edges from a given node
def get_outgoing_edges(node_name):
	outgoing_edges = []
	for edge in edges:
		if edge[0] == node_name:
			outgoing_edges.append(edge)
	return outgoing_edges

# returns all incoming edges from a given node
def get_incoming_edges(node_name):
	incoming_edges = []
	for edge in edges:
		if edge[1] == node_name:
			incoming_edges.append(edge)
	return incoming_edges


######################################################
# Abilene graph definition 

# graph nodes
node_names = {0: "ATLAng",
				1: "CHINng",
				2: "DNVRng",
				3: "HSTNng",
				4: "IPLSng",
				5: "KSCYng",
				6: "LOSAng",
				7: "NYCMng",
				8: "SNVAng",
				9: "STTLng",
				10: "WASHng"}
NUM_NODES = len(node_names)
#print("\nDEBUG node_names : " +str(node_names) + " \nlen(node_names) : " + str(NUM_NODES))
# 'inverted' dict for creating the adj_matrix
names_node = {name:id_node for id_node,name in node_names.iteritems()}

# graph edges
edges = []
edges.append(("HSTNng", "ATLAng"))
edges.append(("ATLAng", "HSTNng"))

edges.append(("WASHng", "ATLAng"))
edges.append(("ATLAng", "WASHng"))

edges.append(("WASHng", "NYCMng"))
edges.append(("NYCMng", "WASHng"))

edges.append(("CHINng", "NYCMng"))
edges.append(("NYCMng", "CHINng"))

edges.append(("CHINng", "IPLSng"))
edges.append(("IPLSng", "CHINng"))

edges.append(("ATLAng", "IPLSng"))
edges.append(("IPLSng", "ATLAng"))

edges.append(("KSCYng", "IPLSng"))
edges.append(("IPLSng", "KSCYng"))

edges.append(("KSCYng", "HSTNng"))
edges.append(("HSTNng", "KSCYng"))

edges.append(("KSCYng", "DNVRng"))
edges.append(("DNVRng", "KSCYng"))

edges.append(("STTLng", "DNVRng"))
edges.append(("DNVRng", "STTLng"))

edges.append(("SNVAng", "DNVRng"))
edges.append(("DNVRng", "SNVAng"))

edges.append(("SNVAng", "STTLng"))
edges.append(("STTLng", "SNVAng"))

edges.append(("SNVAng", "LOSAng"))
edges.append(("LOSAng", "SNVAng"))

edges.append(("HSTNng", "LOSAng"))
edges.append(("LOSAng", "HSTNng"))

NUM_EDGES = len(edges)
#print("\nDEBUG edges : " +str(edges) + "\nlen(edges) : " + str(len(edges)))

# adjacency matrix
adj_matrix = [[0 for i in range(NUM_NODES)] for j in range(NUM_NODES)]
for edge in edges:
	id_node1 = names_node[edge[0]]
	id_node2 = names_node[edge[1]]
	adj_matrix[id_node1][id_node2] = 1
	adj_matrix[id_node2][id_node1] = 1
#print("\nDEBUG adj_matrix :" + str(adj_matrix))

