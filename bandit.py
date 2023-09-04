# %%
import numpy as np
import typing

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