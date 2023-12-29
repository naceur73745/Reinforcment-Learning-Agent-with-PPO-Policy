# Importing necessary library
import numpy as np

# Class to represent a replay buffer for reinforcement learning
class ReplayBuffer:
    
    def __init__(self, mem_size, input_dim, n_actions, batch_size):
        """
        Initialize the ReplayBuffer with specified parameters.

        Parameters:
        - mem_size: Size of the memory buffer
        - input_dim: Dimension of the input state
        - n_actions: Number of possible actions
        - batch_size: Size of the batch for learning
        """
        # Initialize memory size and arrays to store different types of data
        self.mem_size = mem_size
        self.state_mem = np.zeros((mem_size, input_dim))
        self.new_state_mem = np.zeros((mem_size, input_dim))
        self.action_mem = np.zeros((mem_size))
        self.reward_mem = np.zeros((mem_size))
        self.done_mem = np.zeros((mem_size,))
        self.value_mem = np.zeros((mem_size))
        self.prob_mem = np.zeros((mem_size))
        self.mem_cntr = 0
        self.batch_size = batch_size

    def store_action(self, state, new_state, action, reward, done, prob, value):
        """
        Store the experience tuple (state, new_state, action, reward, done, prob, value) in the replay buffer.

        Parameters:
        - state: Current state
        - new_state: New state after the action
        - action: Action taken
        - reward: Reward received
        - done: True if the episode is done, False otherwise
        - prob: Probability of the action (used in some algorithms like PPO)
        - value: Estimated value of the state (used in some algorithms like PPO)
        """
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done
        self.value_mem[index] = value
        self.prob_mem[index] = prob
        self.mem_cntr += 1

    def sample_mem(self):
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
        - state: Array of current states in the batch
        - new_state: Array of new states in the batch
        - action: Array of actions in the batch
        - reward: Array of rewards in the batch
        - done: Array of done flags in the batch
        - prob: Array of probabilities of actions in the batch
        - value: Array of estimated values of states in the batch
        - batch_indices: Indices of the sampled batch in the memory
        """
        # Find the number of available memories
        mem_empty = min(self.mem_size, self.mem_cntr)

        # Sample a batch of indices
        if mem_empty >= self.batch_size:
            batch_indices = np.random.choice(mem_empty, self.batch_size, replace=False)
        else:
            batch_indices = np.random.choice(mem_empty, self.batch_size, replace=True)

        # Retrieve data for the sampled batch
        state = self.state_mem[batch_indices]
        new_state = self.new_state_mem[batch_indices]
        action = self.action_mem[batch_indices]
        reward = self.reward_mem[batch_indices]
        done = self.done_mem[batch_indices]
        prob = self.prob_mem[batch_indices]
        value = self.value_mem[batch_indices]

        return state, new_state, action, reward, done, prob, value, batch_indices
