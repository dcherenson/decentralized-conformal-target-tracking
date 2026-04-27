"""Topology primitives and sampling helpers for observation/communication graphs."""

import numpy as np


class Topology:
	"""Class that governs observation topology and communciation topology"""

	def __init__(self, node_num):
		self.nodes = node_num
		self.edges = []

	def add_edge(self, i, j):
		self.edges.append([i, j])

	def status(self):
		print(self.nodes)

		for edge in self.edges:
			print(edge)


def sample_topologies(node_num, observ_prob, comm_prob, rng=None):
	"""Sample observation and communication topologies in memory."""

	rng = rng or np.random.default_rng()

	# Observation graph: edges may target robots or the landmark index.
	observ_topology = Topology(node_num)
	for i in range(node_num):
		for j in range(node_num + 1):
			if (i != j) and (rng.random() < observ_prob):
				observ_topology.add_edge(i, j)

	# Communication graph: robot-to-robot message passing edges only.
	comm_topology = Topology(node_num)
	for i in range(node_num):
		for j in range(node_num):
			if (i != j) and (rng.random() < comm_prob):
				comm_topology.add_edge(i, j)

	return observ_topology, comm_topology


def generate_topology(node_num, observ_prob, comm_prob):
	"""Persist one sampled topology pair to topology/output.txt."""
	observ_topology, comm_topology = sample_topologies(node_num, observ_prob, comm_prob)

	with open('topology/output.txt', 'w') as output_file:
		# observation topology
		output_file.write(str(len(observ_topology.edges)) + '\n')

		for edge in observ_topology.edges:
			[i, j] = edge
			output_file.write(str(i) + ', ' + str(j) + '\n')

		# communication topology
		output_file.write(str(len(comm_topology.edges)) + '\n')

		for edge in comm_topology.edges:
			[i, j] = edge
			output_file.write(str(i) + ', ' + str(j) + '\n')
