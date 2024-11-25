# Bayesian Network From Scratch

**Introduction**

This Python implementation provides a framework for creating and manipulating Bayesian Networks. Key features include:

* **Node and Edge Creation:**
  * Add nodes with associated probability distributions (With distribution check).
  * Add edges to define dependencies between nodes.

* **Probability Calculation:**
  * Efficiently compute probabilities of events given evidence.

* **CPT Update:**
  * Modify conditional probability tables to reflect changes in knowledge.

* **Cycle Detection:**
  * Prevent the creation of cyclic networks, ensuring valid Bayesian structures.

**Usage**
1. Clone the Repository:
```bash
git clone https://github.com/yzaazaa/BayesianNetwork
cd BayesianNetwork
```

2. Example:
```bash
from bayesianNetwork import BayesianNetwork

# Create a simple Bayesian Network
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
prob = tp.probability({"rain": "none", "maintenance": "no", "train": "on time", "appointment": "attend"})
print(prob)
```

Output:

```bash
0.34019999999999995
```

**API Reference**

The BayesianNetwork class provides methods for building and manipulating Bayesian networks.

* **add_node(name, probabilities):**
  * Adds a node to the network.

* **add_edge(parent, child):**
  * Adds a directed edge from a parent node to a child node.

* **get_children(node):**
  * Returns a list of child nodes for a given node.

* **get_parents(node):**
  * Returns a list of parent nodes for a given node.

* **get_distribution(node):**
  * Returns the probability distribution of a node.

* **update_cpt(node, new_probabilites):**
  * Updates the conditional probability table (CPT) for a node.

* **probability(conditions):**
  * Calculates the probability of a query given evidence.
  * Conditions: A dictionary mapping random variable names with its value

**Future Work**

This implementation can be further extended by exploring the following areas:

* **Advanced Inference Algorithms**
  * Implementing techniques like variable elimination and belief propagation for efficient inference in large networks.

* **Learning from Data**
  * Developing methods to learn network structure and parameters from data using structure learning and parameter estimation.

* **Visualization**
  * Creating tools to visualize Bayesian networks graphically, aiding in understanding and debugging.

* **Performance Optimization:**
  * Identifying and optimizing performance bottlenecks, especially for large-scale networks.

* **Machine Learning Integration:**
  * Exploring ways to combine Bayesian networks with machine learning techniques for more powerful modeling and prediction.