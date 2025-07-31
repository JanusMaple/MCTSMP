import numpy as np
from map import *
from copy import deepcopy
import time

class TreeNode: 
    def __init__(self, value=None, parent=None, children=[], depth=0) -> None:
        self.value = value
        self.parent = parent
        self.children = children
        self.depth = depth                                      # Depth is zero for the root

    def add_children(self, node): 
        assert isinstance(node, TreeNode)
        self.children.append(node)

class MCTNodeValue: 

    action_list = []                                            # Available actions
    state_d = 4                                                 # State dimension
    c = 4.0                                                     # UCB constant

    def __init__(self, state) -> None:
        self.state = state
        self.N_s = 0                                            # Times of the node being visited
        self.N_a = np.zeros((len(self.action_list),1))          # Times of the action being visited
        self.Q_a = np.zeros((len(self.action_list),1))          # Reward of the action being visited

    def update(self, action, reward): 
        self.N_s = self.N_s + 1
        self.Q_a[action] = (self.Q_a[action] * self.N_a[action] + reward) / (self.N_a[action] + 1)
        self.N_a[action] = self.N_a[action] + 1

    def UCB(self): 
        UCB = self.Q_a + np.sqrt(np.log(self.c * (self.N_s + 1)) / (self.N_a + 1))
        return UCB

class MCTNode(TreeNode): 
    def __init__(self, value=None, parent=None, children=[], depth=0) -> None:
        assert isinstance(value, MCTNodeValue)
        TreeNode.__init__(self, value, parent, children, depth)
        self.expanded = False

    def expand(self): 
        self.expanded = True
        for action in self.value.action_list:                   # TO DO: Expanding too many nodes here
            new_state = deepcopy(self.value.state)
            new_state.apply_action(action)
            new_state.update()
            self.children.append(MCTNode(value=MCTNodeValue(new_state), parent=self, children=[], depth=self.depth+1))

    def update(self, action, reward): 
        self.value.update(action, reward)

    def select(self): 
        UCB = self.value.UCB()
        action = np.argmax(UCB)
        return action

class MCTree: 
    def __init__(self, init_state) -> None:
        self.root = MCTNode(MCTNodeValue(init_state), None, [], 0)
        self.root.expand()
        self.tree_depth = 0

    def select(self): 
        depth = 0
        action_list = np.zeros(self.tree_depth + 2, dtype=int)
        state_list = np.zeros((MCTNodeValue.state_d, self.tree_depth + 3))
        visiting_node = self.root                               # Start from root
        
        while (visiting_node.expanded):                         # Until a leaf node
            state_list[:, depth] = visiting_node.value.state.get_state_as_array()

            action = visiting_node.select()
            action_list[depth] = action

            visiting_node = visiting_node.children[action]
            depth = depth + 1
        state_list[:, depth] = visiting_node.value.state.get_state_as_array()
        
        visiting_node.expand()                                  # Expand the leaf node
        if visiting_node.depth > self.tree_depth: 
            self.tree_depth = visiting_node.depth
        action = visiting_node.select()                         # Select an action for the leaf
        action_list[depth] = action
        state_list[:, depth + 1] = visiting_node.children[action].value.state.get_state_as_array()

        action_list = action_list[0 : depth + 1]
        state_list = state_list[:, 0 : depth + 2]
        expanded_leaf_node = visiting_node
        return action_list, state_list, expanded_leaf_node
    
    def back_propogate(self, action_list, expanded_leaf_node, Q): 
        visiting_node = expanded_leaf_node
        for action in reversed(action_list): 
            visiting_node.update(action, Q)
            visiting_node = visiting_node.parent

class MCTS_Planner: 
    time_budget = 0.09                                           # second
    def __init__(self, map) -> None:
        assert isinstance(map, Map)
        self.map = map
        self.MCTree = MCTree(map.robot)
        MCTNodeValue.state_d = map.robot.state_list_len

    def plan(self): 
        time_start = time.time()
        self.MCTree = MCTree(self.map.robot)

        while time.time() < time_start + self.time_budget: 
            action_list, state_list, expanded_leaf_node = self.MCTree.select()
            
            delta_time = len(action_list)
            future_map = FakeMap(self.map)
            future_map.update_environment_predictively(delta_time, state_list)
            
            Q = future_map.evaluate(state_list)
            self.MCTree.back_propogate(action_list, expanded_leaf_node, Q)
        action = np.argmax(self.MCTree.root.value.Q_a)
        return action
