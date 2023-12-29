# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Actor Network class for policy approximation
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_action, lr):
        """
        Initialize the ActorNetwork with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state
        - fc1_dim: Dimension of the first fully connected layer
        - fc2_dim: Dimension of the second fully connected layer
        - n_action: Number of possible actions
        - lr: Learning rate for the optimizer
        """
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.evaluate = False  # Flag to control exploration vs exploitation
        self.actor = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.Sigmoid(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.Sigmoid(),
            nn.Linear(fc2_dim, n_action),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        Forward pass of the actor network.

        Parameters:
        - state: Input state for which action probabilities are calculated

        Returns:
        - action: Sampled action if in exploration mode, otherwise the action with maximum probability
        - prob: Log probability of the chosen action
        - actions_dist: Action probabilities distribution
        """
        actions_dist = self.actor(state)
        actions = Categorical(actions_dist)
        if not self.evaluate:
            # Exploration: Sample an action from the distribution
            action = actions.sample()
        else:
            # Exploitation: Choose the action with maximum probability
            action = torch.argmax(actions_dist)
        prob = actions.log_prob(action)
        return action, prob, actions_dist

# Critic Network class for estimating state values
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, lr):
        """
        Initialize the CriticNetwork with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state
        - fc1_dim: Dimension of the first fully connected layer
        - fc2_dim: Dimension of the second fully connected layer
        - lr: Learning rate for the optimizer
        """
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.critic = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        Forward pass of the critic network.

        Parameters:
        - state: Input state for which the value is estimated

        Returns:
        - value: Estimated value of the input state
        """
        value = self.critic(state)
        return value

# Example usage of the CriticNetwork
state = (0, 1)
state = torch.tensor(state, dtype=torch.float)
objects = CriticNetwork(2, 128, 215, 0.001)
value = objects.forward(state)
