# Importing the random module
import random 

# Class definition for the Prisoners class
class Prisoners:

    def __init__(self, episode_len, n_round):
        """
        Initialize the Prisoners class with specified parameters.

        Parameters:
        - episode_len: Length of each episode
        - n_round: Number of rounds in each training episode
        """
        self.action_space = 2
        self.n_round = n_round
        self.observation_space = 4
        self.current_round = 0
        self.current_step = 0
        self.episode_len = episode_len
        self.done = False
        # Initial state with random choices for both players
        self.state = (random.choice([0,1]), random.choice([0,1]))
        self.steps = 0
        self.grudge = False
        # Payoff matrix defining rewards for different actions
        self.payoff_matrix = {(0,0): (2,2), (0,1): (0,3), (1,0): (3,0), (1,1): (1,1)}
        # List of strategies for the second player
        self.strategies = ["Always_cooperate", "Always_defect", "Grudge", "Tit_for_Tat"]
        self.index = 0
        self.state_total = []
        self.round_state_list = []
        self.chooosed_startegy_each_round = []
        self.reward_total = []
        self.reward_round_list = []

    def reset(self):
        """
        Reset the environment to a new episode.

        Returns:
        - state: Initial state for the new episode
        """
        # Begin with a random state
        self.state = (random.choice([0,1]), random.choice([0,1]))
        if self.index == len(self.strategies):
            self.index = 0
        self.chooosed_startegy_each_round.append(self.strategies[self.index])
        self.current_step = 0
        self.grudge = False
        self.done = False
        self.round_state_list = []
        self.reward_round_list = []
        return self.state

    def coop(self, action):
        """
        Returns the action for cooperation (always 0).

        Parameters:
        - action: Action taken by the first player

        Returns:
        - action: Action for cooperation
        """
        return 0

    def defect(self, action):
        """
        Returns the action for defection (always 1).

        Parameters:
        - action: Action taken by the first player

        Returns:
        - action: Action for defection
        """
        return 1

    def Grudge(self, action):
        """
        Implements the Grudge strategy.

        Parameters:
        - action: Action taken by the first player

        Returns:
        - action: Action based on the Grudge strategy
        """
        if action == 1:
            self.grudge = True
        if self.grudge:
            return 1
        else:
            return 0

    def Tit_for_Tat(self, action):
        """
        Implements the Tit-for-Tat strategy.

        Parameters:
        - action: Action taken by the first player

        Returns:
        - action: Action based on the Tit-for-Tat strategy
        """
        return self.state[1]

    def evaluate(self, action, value):
        """
        Evaluate the chosen action and update the environment state.

        Parameters:
        - action: Action taken by the first player
        - value: Strategy index chosen for the second player

        Returns:
        - state: Updated state
        - reward[0]: Reward for the first player
        - done: Flag indicating if the episode is done
        - reward: Tuple containing rewards for both players
        """
        if self.current_step == self.episode_len:
            print("Episode is done")
            self.done = True

        # Current state
        self.state = (action, int(random.choice([0,1])))
        # Get the reward
        reward = self.payoff_matrix[self.state]
        # Increment current step
        self.current_step += 1

        return self.state, reward[0], self.done, reward

    def step(self, action):
        """
        Take a step in the environment based on the chosen action.

        Parameters:
        - action: Action taken by the first player

        Returns:
        - state: Updated state
        - reward[0]: Reward for the first player
        - done: Flag indicating if the episode is done
        - reward: Tuple containing rewards for both players
        """
        reward = (0, 0)
        player2_action = 0

        if self.current_round == self.n_round:
            print("Training over")

        elif self.current_step == self.episode_len:
            self.index += 1
            self.reward_total.append(self.reward_round_list)
            self.state_total.append(self.round_state_list)
            self.done = True

        else:
            self.state = (action, player2_action)
            self.current_step += 1
            reward = self.payoff_matrix[self.state]
            self.round_state_list.append(self.state)
            self.reward_round_list.append(reward)

        return self.state, reward[0], self.done, reward
