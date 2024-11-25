from abc import ABC, abstractmethod

class Node():
	def __init__(self, state, parent, action, cost):
		self.state = state
		self.parent = parent
		self.action = action
		self.cost = cost

# AFrontier is an abstract base class for Frontier classes for uninformed search

class AFrontier(ABC):
	def __init__(self):
		self.frontier = []
	def push(self, node):
		self.frontier.append(node)
	def empty(self):
		return (len(self.frontier) == 0)
	def contains_state(self, state):
		return any(node.state==state for node in self.frontier)
	@abstractmethod
	def pop(self):
		pass

# StackFrontier Class which will be used for depth-first search	

class StackFrontier(AFrontier):
	# Pop the last pushed node to the frontier
	def pop(self):
		if len(self.frontier) == 0:
			raise Exception("Empty frontier")
		node = self.frontier[-1]
		del self.frontier[-1]
		return node