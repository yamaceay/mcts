# %%
import random
import math
import numpy as np
import typing
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
)         

logger = logging.getLogger(__name__)
logger.handlers = [logging.StreamHandler()]

GAMMA = 1. # decaying rate
BETA = 2. # exploration weight

Y = typing.TypeVar("Y") # observation
S = typing.TypeVar("S") # state
A = typing.TypeVar("A") # action

class BanditStepResults(object):
    """Bandit step results including the reward and the next observation and whether the simulation has ended"""

    def __init__(self, reward: float = .0, observations: list[Y] = [], terminal: bool = False):
        self.reward = reward
        self.observations = observations
        self.terminal = terminal
        
    def add(self, other):
        self.reward += other.reward
        self.observations += other.observations
        self.terminal |= other.terminal
        
class Bandit:
    """Bandit interface"""

    def reset(self) -> list[Y]:
        """Start with the initial state and reset the environment"""

    def actions(self) -> list[A]:
        """List of possible actions"""

    def step(self, a: A) -> BanditStepResults:
        """Commit the action and return the reward and the next observation"""
    
    def observe(self, s: S) -> list[Y]:
        """Observe the state"""
    
    def fin(self) -> bool:
        """Whether the simulation has ended"""
        
class BanditImpl(Bandit):
    """A bandit implementation"""

    def __init__(self, state_reward: dict[int, float], max_n_steps: int, max_n_steps_in_ep: int = None):
        self.state_reward = state_reward
        
        self.n_steps = 0
        self.max_n_steps = max_n_steps
        self.max_n_steps_in_ep = max_n_steps_in_ep

        self.reset()

    def observe(self, s: int) -> list[int]:
        return [s % 2, s % 3, s % 5] # example of observations
    
    def fin(self) -> bool:
        return self.n_steps >= self.max_n_steps

    def reset(self) -> list[int]:
        self.n_steps_in_ep = 0
        self.state = np.random.randint(low=-2, high=2)
        return self.observe(self.state)
    
    def actions(self) -> list[int]:
        return [-1, 1] # left, right

    def step(self, a: int) -> BanditStepResults:
        self.n_steps += 1
        self.n_steps_in_ep += 1
        self.state += a
        observations = self.observe(self.state)
        step = BanditStepResults(observations = observations)

        if self.state in self.state_reward:
            step.reward = self.state_reward[self.state]
            step.terminal = True
        
        elif self.max_n_steps_in_ep:
            step.terminal = self.max_n_steps_in_ep <= self.n_steps_in_ep
        
        return step

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
    q_values_of_actions = {child.action: ucb_score(child, exploration_weight = 0.0) for child in all_children}
    q_values = [q_values_of_actions[a] for a in bandit.actions()]
    return q_values

def log_tree(node, depth=0):
    """Log all nodes in hierarchical order"""
    
    node = NODES[node]
    prestr = f"{depth*'--'+'a:'+str(node.action)+' ' if depth else ''}"
    logger.debug(f"{prestr}r:{node.reward} n:{node.count}")
    for child in node.children:
        log_tree(child, depth+1)
        
if __name__ == "__main__":
    if not os.path.exists("betas"):
        os.makedirs("betas")
    BETAS = [0.1, 0.5, 1.0, 2.0, 5.0]
    for BETA in BETAS:
        filename = f"betas/beta_{BETA}.log"
        logger.handlers[0].stream = open(filename, "w")
        
        bandit = BanditImpl({5: 1., -5: -1.}, 10_000)
        mcts_q(bandit)
        log_tree(".")