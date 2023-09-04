# %%
import random
import math
import os
from bandit import Bandit, BanditStepResults, BanditImpl

GAMMA = 1. # decaying rate
BETA = 2. # exploration weight

NODES = {}
class TreeNode(object):
    """Bidirectional tree node implementation"""
    
    def __init__(self, parent: str = None, action: int = None):
        self.parent = parent
        self.action = action
        
        self.count = 0
        self.reward = .0
        self.children = []
        
        NODES.update({str(self): self})
        
    def __repr__(self):
        return "." if self.parent is None else self.parent + "/" + str(self.action)

def ucb_score(child: TreeNode, exploration_weight: float = BETA):
    """Calculate the UCB score from the parent node"""
    
    child_count = child.count + 1e-8
    exploitation_score = child.reward / child_count
    exploration_score = math.sqrt(2 * math.log(NODES[child.parent].count) / child_count)
    return exploitation_score + exploration_weight * exploration_score
    
def select(bandit: Bandit, node: TreeNode):
    """Select the best child node"""
    
    bandit.reset()
    step = BanditStepResults()

    while not bandit.fin() and not step.terminal:
        if len(node.children) < len(bandit.actions()):
            break
            
        scores = [ucb_score(NODES[child]) for child in node.children]
        scores_sorted = sorted(enumerate(scores), key=lambda x: x[1])
        best_child_index, _ = scores_sorted[-1]
        best_child = node.children[best_child_index]
        node = NODES[best_child]
        
        new_step = bandit.step(node.action)
        step.add(new_step)
            
    return node, step

def expand(bandit: Bandit, node: TreeNode):
    """Expand the node by adding a new child node"""
    
    actions_not_available = [NODES[child].action for child in node.children]
    actions_available = [a for a in bandit.actions() if a not in actions_not_available]            
    a = random.choice(actions_available)

    node_child = TreeNode(parent=str(node), action=a)
    NODES[str(node)].children += [str(node_child)]
    return node_child
    
def rollout(bandit: Bandit, node: TreeNode, step: BanditStepResults):
    """Rollout the simulation until the end"""
    
    while not bandit.fin() and not step.terminal:
        a = random.choice(bandit.actions())
        new_step = bandit.step(a)
        step.add(new_step)
    return node, step

def backpropagate(node: TreeNode, step: BanditStepResults):
    """Backpropagate the reward upwards the tree"""
    
    if step.terminal:
        node_str = str(node)
        while node_str in NODES:
            NODES[node_str].reward += GAMMA * step.reward
            NODES[node_str].count += 1
            node_str = NODES[node_str].parent

def mcts_q(bandit: Bandit):
    """Monte Carlo UCT Q-value estimation"""
    
    NODES.clear()
    root = TreeNode()

    while not bandit.fin():  
        node, step = select(bandit, root)
        if not bandit.fin() and not step.terminal:
            leaf = expand(bandit, node)
            leaf, step = rollout(bandit, leaf, step)
        backpropagate(leaf, step)

    all_children = [NODES[child] for child in root.children]
    q_values_of_actions = {child.action: ucb_score(child, 0.0) for child in all_children}
    q_values = [q_values_of_actions[a] for a in bandit.actions()]
    return q_values


import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
)         
        
if __name__ == "__main__":
    BETAS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    if not os.path.exists("betas"):
        os.makedirs("betas")
        
    logger = logging.getLogger(__name__)
    logger.handlers = [logging.StreamHandler()]

    def log_tree(node, depth=0):
        """Log all nodes in hierarchical order"""
        
        node = NODES[node]
        prestr = f"{depth*'--'+'a:'+str(node.action)+' ' if depth else ''}"
        logger.debug(f"{prestr}r:{node.reward} n:{node.count}")
        for child in node.children:
            log_tree(child, depth+1)
            
    for BETA in BETAS:
        filename = f"betas/beta_{BETA}.log"
        logger.handlers[0].stream = open(filename, "w")
        
        bandit = BanditImpl({6: 1., -4: -1.}, 10_000)
        mcts_q(bandit)
        log_tree(".")