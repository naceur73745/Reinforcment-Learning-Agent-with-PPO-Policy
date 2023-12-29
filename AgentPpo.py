# Importing necessary modules and classes
from ReplayBuffer import ReplayBuffer
from NetworkPpo import ActorNetwork, CriticNetwork
import torch
import numpy as np

# Class definition for the Agent class
class Agent:
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, lr, batch_size, mem_size, gamma, epsilon_dec, policy_clip, lamda):
        """
        Initialize the Agent class with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state
        - fc1_dim: Dimension of the first fully connected layer
        - fc2_dim: Dimension of the second fully connected layer
        - n_actions: Number of actions in the action space
        - lr: Learning rate for the optimizer
        - batch_size: Size of the mini-batch for learning
        - mem_size: Size of the replay memory
        - gamma: Discount factor for future rewards
        - epsilon_dec: Decay rate for the exploration-exploitation parameter epsilon
        - policy_clip: Clipping parameter for the policy loss
        - lamda: Lambda parameter for the generalized advantage estimate (GAE)
        """
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.epsilon = 1
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = 0.01
        self.gamma = gamma
        # Initialize replay buffer
        self.mem = ReplayBuffer(mem_size, input_dim, n_actions, batch_size)
        # Initialize actor and critic networks
        self.actor = ActorNetwork(input_dim, fc1_dim, fc2_dim, n_actions, lr)
        self.critic = CriticNetwork(input_dim, fc1_dim, fc2_dim, lr)
        self.lamda = lamda
        self.policy_clip = policy_clip
        self.n_actions = n_actions

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        Parameters:
        - state: Current state of the environment

        Returns:
        - action.item(): Chosen action
        - prob: Probability of the chosen action
        - value: Estimated value of the current state
        - action_dist: Probability distribution over actions
        """
        state = torch.tensor(state, dtype=torch.float32)
        action, prob, action_dist = self.actor(state)
        value = self.critic(state)
        return action.item(), prob, value, action_dist

    def learn(self):
        """
        Update the actor and critic networks based on the collected experiences.
        """
        if self.batch_size > self.mem.mem_cntr:
            return

        state, new_state, action, reward, done, old_prob, value, batch = self.mem.sample_mem()
        advantage = np.zeros(len(reward))

        for time_step in range(len(reward) - 1):
            discount = 1
            tmp_adv = 0
            for j in range(time_step, len(reward) - 1):
                tmp_adv += discount * reward[j] + self.gamma * value[j + 1] * (1 - int(done[j])) - value[j]
                discount *= self.gamma * self.lamda
            advantage[time_step] = tmp_adv

        state = torch.tensor(state, dtype=torch.float32)
        old_prob = torch.tensor(old_prob)
        critic_value = self.critic(state)
        action, new_prob, distribution = self.actor(state)
        new_prob = torch.tensor(new_prob)
        prob_ratio = new_prob.exp() / old_prob.exp()
        version = torch.tensor(advantage) * prob_ratio
        clipped_version = torch.clamp(prob_ratio, 1 - self.policy_clip, self.policy_clip) * advantage
        actor_loss = -torch.min(version, clipped_version).mean()
        returns = torch.tensor(advantage) + torch.tensor(value)
        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()
        total_loss = actor_loss + 0.5 * critic_loss

        # Update actor and critic networks
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
