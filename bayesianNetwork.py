from frontier import StackFrontier, Node
import math

def bn_test():
	print("\n=== Starting Bayesian Network Tests ===\n")
	bn = BayesianNetwork()

	# Test 1: Invalid probability sum
	print("Test 1: Adding node with invalid probabilities (sum ≠ 1)")
	try:
		bn.add_node("A", {"True": 0.8, "False": 0.8})
		print("❌ Failed: Accepted invalid probabilities")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 2: Valid node additions
	print("\nTest 2: Adding valid nodes")
	try:
		bn.add_node("A", {"True": 0.7, "False": 0.3})
		bn.add_node("B", {"True": 0.4, "False": 0.6})
		bn.add_node("C", {"True": 0.5, "False": 0.5})
		print("✅ Passed: Successfully added valid nodes")
	except Exception as e:
		print("❌ Failed:", e)

	# Test 3: Adding edge with non-existent node
	print("\nTest 3: Adding edge with non-existent node")
	try:
		bn.add_edge("A", "NonExistent")
		print("❌ Failed: Accepted edge with non-existent node")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 4: Cycle detection tests
	print("\nTest 4: Cycle detection")
	try:
		# Valid edges
		bn.add_edge("A", "B")
		bn.add_edge("B", "C")
		print("✅ Passed: Added valid edges")
		
		# Try to create cycle
		print("Attempting to create cycle (C -> A)...")
		bn.add_edge("C", "A")
		print("❌ Failed: Accepted a cycle")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 5: Getting parents of non-existent node
	print("\nTest 5: Getting parents of non-existent node")
	try:
		parents = bn.get_parents("NonExistent")
		print("❌ Failed: Retrieved parents of non-existent node")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 6: Getting children of non-existent node
	print("\nTest 6: Getting children of non-existent node")
	try:
		children = bn.get_children("NonExistent")
		print("❌ Failed: Retrieved children of non-existent node")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 7: Updating CPT with invalid probabilities
	print("\nTest 7: Updating CPT with invalid probabilities")
	try:
		bn.update_cpt("A", {"True": 0.6, "False": 0.6})
		print("❌ Failed: Accepted invalid probability update")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 8: Updating CPT of non-existent node
	print("\nTest 8: Updating CPT of non-existent node")
	try:
		bn.update_cpt("NonExistent", {"True": 0.5, "False": 0.5})
		print("❌ Failed: Updated CPT of non-existent node")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 9: Complex cycle detection
	print("\nTest 9: Complex cycle detection")
	try:
		bn.add_node("D", {"True": 0.5, "False": 0.5})
		bn.add_node("E", {"True": 0.5, "False": 0.5})
		bn.add_edge("C", "D")
		bn.add_edge("D", "E")
		print("✅ Passed: Added additional valid edges")
		
		print("Attempting to create complex cycle (E -> B)...")
		bn.add_edge("E", "B")
		print("❌ Failed: Accepted a complex cycle")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 10: Verify final structure
	print("\nTest 10: Verifying final structure")
	try:
		parents_c = bn.get_parents("C")
		children_b = bn.get_children("B")
		print(f"✅ Passed: Final structure verification")
		print(f"   Parents of C: {parents_c}")
		print(f"   Children of B: {children_b}")
	except Exception as e:
		print("❌ Failed: Could not verify final structure:", e)

	   # Test 11: Invalid CPT for unconditional node
	print("\nTest 11: Invalid CPT for an unconditional node")
	try:
		bn.add_node("F", {"True": 0.6, "False": 0.5})  # Probabilities don't sum to 1
		print("❌ Failed: Accepted invalid CPT")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 12: Invalid CPT for conditional node
	print("\nTest 12: Invalid CPT for a conditional node")
	try:
		bn.add_node("G", [
			[["A", "True"], 0.2], # Not formatted correctly
			[["A", "False"], 0.8]
		])
		print("❌ Failed: Accepted invalid CPT")
	except Exception as e:
		print("✅ Passed:", e)

	# Test 13: Valid CPT for conditional node
	print("\nTest 13: Valid CPT for a conditional node")
	try:
		bn.add_node("H", [
			["none", "yes", 0.7],
			["none", "no", 0.3],
			["light", "yes", 0.2],
			["light", "no", 0.8],
		])
		print("✅ Passed: Accepted valid CPT")
	except Exception as e:
		print("❌ Failed:", e)

	# Test 14: Invalid CPT for conditional node
	print("\nTest 13: Invalid CPT for a conditional node")
	try:
		bn.add_node("I", [
			["none", "yes", 0.9], # Sum of complementary proba should eq 1
			["none", "no", 0.3],
			["light", "yes", 0.2],
			["light", "no", 0.8],
		])
		print("❌Failed: Accepted invalid CPT")
	except Exception as e:
		print("✅ Passed:", e)

	print("\nTest 14: Probability computing")
	tp = BayesianNetwork()
	tp.add_node("rain", {"none": 0.7, "light": 0.2, "heavy": 0.1})
	tp.add_node("maintenance", [
		["none", "yes", 0.4],
		["none", "no", 0.6],
		["light", "yes", 0.2],
		["light", "no", 0.8],
		["heavy", "yes", 0.1],
		["heavy", "no", 0.9]
	])
	tp.add_node("train", [
		["none", "yes", "on time", 0.8],
		["none", "yes", "delayed", 0.2],
		["none", "no", "on time", 0.9],
		["none", "no", "delayed", 0.1],
		["light", "yes", "on time", 0.6],
		["light", "yes", "delayed", 0.4],
		["light", "no", "on time", 0.7],
		["light", "no", "delayed", 0.3],
		["heavy", "yes", "on time", 0.4],
		["heavy", "yes", "delayed", 0.6],
		["heavy", "no", "on time", 0.5],
		["heavy", "no", "delayed", 0.5],
	])
	tp.add_node("appointment", [
		["on time", "attend", 0.9],
		["on time", "miss", 0.1],
		["delayed", "attend", 0.6],
		["delayed", "miss", 0.4]
	])
	tp.add_edge("rain", "maintenance")
	tp.add_edge("rain", "train")
	tp.add_edge("maintenance", "train")
	tp.add_edge("train", "appointment")

	# Test case 1: Original test (none, no, on time, attend)
	print("\nTest case 1: P(rain=none, maintenance=no, train=on time, appointment=attend)")
	prob = tp.probability({"rain": "none", "maintenance": "no", "train": "on time", "appointment": "attend"})
	expected = 0.34019999999999995
	if math.isclose(prob, expected):
		print("✅ Passed:", prob, "is close to", expected)
	else:
		print("❌ Failed:", prob, "is not close to", expected)

	# Test case 2: Heavy rain scenario
	print("\nTest case 2: P(rain=heavy, maintenance=yes, train=delayed, appointment=miss)")
	prob = tp.probability({"rain": "heavy", "maintenance": "yes", "train": "delayed", "appointment": "miss"})
	expected = 0.0024
	if math.isclose(prob, expected):
		print("✅ Passed:", prob, "is close to", expected)
	else:
		print("❌ Failed:", prob, "is not close to", expected)

	# Test case 3: Light rain scenario
	print("\nTest case 3: P(rain=light, maintenance=no, train=delayed, appointment=attend)")
	prob = tp.probability({"rain": "light", "maintenance": "no", "train": "delayed", "appointment": "attend"})
	expected = 0.028800000000000006
	if math.isclose(prob, expected):
		print("✅ Passed:", prob, "is close to", expected)
	else:
		print("❌ Failed:", prob, "is not close to", expected)

	# Test case 5: Invalid value
	print("\nTest case 5: Testing invalid value")
	try:
		prob = tp.probability({"rain": "invalid_value"})
		print("❌ Failed: Accepted invalid value")
	except ValueError as e:
		print("✅ Passed: ", e)

	# Test case 6: Invalid node
	print("\nTest case 6: Testing invalid node")
	try:
		prob = tp.probability({"invalid_node": "none"})
		print("❌ Failed: Accepted invalid node")
	except ValueError as e:
		print("✅ Passed: ", e)

	# Test case 7: Invalid dimensions
	print("\nTest case 7: Testing invalid dimensions")
	try:
		prob = tp.probability({"rain": "none", "apointment": "miss"})
		print("❌ Failed: Accepted invalid node")
	except ValueError as e:
		print("✅ Passed: ", e)


	print("\n=== Tests Complete ===")


class BayesianNetwork:
	def __init__(self):
		self.nodes = {}
		self.edges = {}
	
	def check_cpt(self, probabilities):
		"""
		Check if the given cpt is valid.
		:param probabilities: Probabilities in a list of tuples for conditional nodes, and a dictionary for unconditional nodes.
		"""
		if isinstance(probabilities, dict):
			if not math.isclose(sum(probabilities.values()), 1.0):
				return False
			return True
		elif isinstance(probabilities, list):
			conditions_dict = {}
			# Add all complementary probabilities to one list
			for proba in probabilities:
				if len(proba) <= 2:
					raise ValueError("Conditional nodes should be represented like this:\n\t[**kwargs(condtions), probabilitiy]")
				key = tuple(proba[:-2])
				if key not in conditions_dict:
					conditions_dict[key] = []
				conditions_dict[key].append(proba[-1])
			
			# Check if sum of complementary probabilities equals 1
			for conditions, proba in conditions_dict.items():
				if not math.isclose(sum(proba), 1.0):
					return False
			return True
		else:
			raise TypeError("Probabilities must be a dict or a list.")

	def add_node(self, name, probabilities):
		"""
		Add a node to the network.
		:param name: Name of the node
		:param probabilities: Probabilities in a list of tuples for conditional nodes, and a dictionary for unconditional nodes.
		"""
		if not self.check_cpt(probabilities):
			raise Exception("Sum of complementary probabilities should equal 1.")
		self.nodes[name] = probabilities
		self.edges[name] = []

	def check_cycle(self, parent, child):
		"""
		Check if adding an edge could create a cycle
		:param parent: Parent node
		:param child: Child node
		"""
		def neighbors(node):
			"""
			Returns all child nodes of node.
			:param node: parent node
			"""
			if node not in self.nodes:
				raise ValueError("Not a valid node.")
			return self.edges[node]

		init_node = Node(state=child, parent=parent, action=None, cost=0)
		frontier = StackFrontier()
		frontier.push(init_node)
		
		explored = set()

		while True:
			# If frontier empty no cycle detected
			if frontier.empty():
				return False
			
			node = frontier.pop()

			if node.state == parent:
				return True

			explored.add(node.state)

			for neighbor in neighbors(node.state):
				if not frontier.contains_state(neighbor) and neighbor not in explored:
					frontier.push(Node(state=neighbor, parent=node, action=None, cost=0))

	def add_edge(self, parent, child):
		"""
		Add an edge to the network
		:param parent: Parent node
		:param child: Child node
		"""
		if parent not in self.nodes or child not in self.nodes:
			raise ValueError("Parent and child must be names of existing nodes.")
		if self.check_cycle(parent, child):
			raise ValueError("Adding this edge would create a cycle.")
		self.edges[parent].append(child)

	def get_children(self, node):
		"""
		Get all children of a given node.
		"""
		if node not in self.nodes:
			raise ValueError("Node Not valid.")
		return self.edges[node]
	
	def get_parents(self, node):
		"""
		Get all parents of a given node.
		"""
		if node not in self.nodes:
			raise ValueError("Node Not valid.")
		parents = set()
		for parent, children in self.edges.items():
			if node in children:
				parents.add(parent)
		return parents
	
	def get_cpt(self, node):
		"""
		Get Conditional Probability Table for a given node.
		"""
		if node not in self.nodes:
			raise ValueError("Node Not valid.")
		return self.nodes[node]
	
	def get_distribution(self, node):
		"""
			Get distribution for a given node.
		"""
		if node not in self.nodes:
			raise ValueError("Node not valid.")
		return self.nodes[node]

	def update_cpt(self, node, new_probabilities):
		"""
		Update the Conditional Probability Table for a given node.
		"""
		if node not in self.nodes:
			raise ValueError("Not a valid node.")
		if not self.check_cpt(new_probabilities):
			raise ValueError("Sum of all probabilities must be equal to 1.")
		self.nodes[node] = new_probabilities

	def topological_sort(self, nodes):
		visited = set()          # Nodes we've completed
		sorted_nodes = []        # Final sorted list
		temp_visited = set()     # Nodes we're currently processing
		stack = []

		for node in nodes:
			if node not in visited:
				stack.append((node, False))
				
				while stack:
					current, processed = stack.pop()
					
					if processed:
						if current not in visited:  # Only add if not visited
							visited.add(current)
							sorted_nodes.append(current)
							temp_visited.remove(current)  # Remove from processing set
					else:
						if current in temp_visited:  # Skip if already processing
							continue
							
						temp_visited.add(current)  # Mark as being processed
						stack.append((current, True))
						
						# Add unvisited parents to stack
						for parent in self.get_parents(current):
							if parent not in visited:
								stack.append((parent, False))

		return sorted_nodes

	def probability(self, conditions):
		"""
		Computes the joint probability of a set of conditions.

		Args:
			conditions: A dictionary mapping node names to their observed values.

		Returns:
			The joint probability of the given conditions.
		"""
		if len(conditions) != len(self.nodes.keys()):
			raise ValueError("Conditions do not have same dimensions as the model " + str(conditions.values()) + " " + str(len(conditions)))
		relevant_nodes = set()
		# Check for conditions
		for node, proba in conditions.items():
			if node not in self.nodes:
				raise ValueError("Node not in network.")
			if isinstance(self.nodes[node], dict) and proba not in self.nodes[node].keys():
				raise ValueError("Value not in possible values.")
			elif isinstance(self.nodes[node], list):
				if proba not in set(i[-2] for i in self.nodes[node]):
					raise ValueError("Value not in possible values.")
			relevant_nodes.add(node)
		
		# Add parent nodes
		for node in list(relevant_nodes): 
			relevant_nodes.update(self.get_parents(node))

		# Sort from root to child
		relevant_nodes = self.topological_sort(relevant_nodes)
		
		# Compute joint probability
		joint_prob = 1.0
		for i, node in enumerate(relevant_nodes):
			prob = 1.0
			if node in conditions.keys():
				if isinstance(self.nodes[node], dict):
					prob = self.nodes[node][conditions[node]]
				else:
					for entry in self.nodes[node]:
						k = [conditions[j] for j in relevant_nodes if j in self.get_parents(node)] + [conditions[node]]
						if entry[-2] == conditions[node] and entry[:len(k)] == k:
							prob = entry[-1]
							break
				joint_prob *= prob
		return joint_prob
