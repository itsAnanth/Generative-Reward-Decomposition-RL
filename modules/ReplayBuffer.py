import random
from collections import deque
from .Trajectory import Trajectory
from .Transition import Transition

class ReplayBuffer:
    """A replay buffer for storing transitions and trajectories."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.trajectory_memory = deque([], maxlen=capacity // 10) # Store fewer full trajectories

    def push(self, *args):
        self.memory.append(Transition(*args))

    def push_trajectory(self, states, actions, total_return):
        self.trajectory_memory.append(Trajectory(states, actions, total_return))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample_trajectories(self, batch_size):
        return random.sample(self.trajectory_memory, min(len(self.trajectory_memory), batch_size))

    def __len__(self):
        return len(self.memory)