import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from kaggle_environments import evaluate, make

from .abstract_game import AbstractGame


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 64, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*6+1, 8)
        self.fc2 = nn.Linear(8*7, 1)

        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(64)
#         self.fc = nn.Sequential(
#             nn.Linear(33, 16),
#             nn.BatchNorm1d(16),
#             nn.Linear(16, 1)
#         )
        self.fc = nn.Linear(64*6+1, 1)

    def forward(self, s, filled):
        v = F.relu(self.bn(self.conv(s)))  # value head
#         v = v.max(dim=2)[0].transpose(1, 2)
        v = v.view(v.size(0), -1, v.size(3)).transpose(1, 2)
        v = torch.cat([v, filled], dim=2)
#         v = v.view(-1, 7*65+1)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v)).view(v.size(0), -1)
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
#         p = p.max(dim=2)[0].transpose(1, 2)
        # batch x columns x (channel x rows)
        p = p.view(p.size(0), -1, p.size(3)).transpose(1, 2)
        p = torch.cat([p, filled], dim=2)
        p = self.fc(p).view(-1, 7)
        return p, v


class ConnectNet(nn.Module):
    # Reference: https://github.com/plkmo/AlphaZero_Connect4/tree/master/src
    def __init__(self, temperature: float = 1.0):
        super(ConnectNet, self).__init__()
        self.temperature = temperature
        self.conv = ConvBlock()
        for block in range(12):
            setattr(self, "res_%i" % block, ResBlock(256, 256))
        self.outblock = OutBlock()

    def forward(self, s):
        # batch, columns, channels
        filled = (s[:, 2:, 0, :].transpose(1, 2) == 1).float()
        s = self.conv(s)
        for block in range(12):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s, filled)
        return torch.distributions.Categorical(logits=p / self.temperature), v


class ConnectX(gym.Env):
    def __init__(self, debug=False):
        self.env = make('connectx', debug=debug)
        # Define required gym fields (examples):
        config = self.env.configuration
        assert config.columns == 7
        assert config.rows == 6
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(
            config.columns * config.rows)

    def step(self, action):
        action = int(action)
        if self.env.state[0].status == "ACTIVE":
            state = self.env.step([action, None])[1]
        else:
            state = self.env.step([None, action])[0]
        if self.env.state[0]['status'] == "INVALID" or self.env.state[1]['status'] == "INVALID":
            raise ValueError("Error")
        # if state['reward'] != 0:
        #     print("Reward:", state['reward'])
        #     print(np.asarray(self.env.state[0]['observation']['board']).reshape(6, 7))
        #     print(self.env.state[0]['reward'], self.env.state[1]['reward'])
        return self.env.state[0]['observation']['board'], int(state['reward'] != 0), state['status'] == 'DONE'

    def reset(self):
        return self.env.reset()[0]['observation']['board']

    def render(self, **kwargs):
        print(np.asarray(self.env.state[0]
                         ['observation']['board']).reshape(6, 7))
        # return self.env.render(**kwargs)


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        self.seed = 1  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (3, 6, 7)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(7))
        # List of players. You should only edit the length
        self.players = list(range(2))
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 300  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 8  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64]

        # Fully Connected Network
        self.encoding_size = 32
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        # Training
        self.results_path = (
            Path(__file__).parent / "../cache" / "results" / "connectx" /
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 200000
        self.batch_size = 64  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 10
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 5000

        # Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 500
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 42
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = ConnectX()
        self.raw_board = None

    def get_board(self, obs):
        self.raw_board = np.asarray(
            obs, dtype=np.float32
        ).reshape(6, 7)
        board_player1 = np.where(self.raw_board == 1, 1.0, 0.0)
        board_player2 = np.where(self.raw_board == 2, 1.0, 0.0)
        board_to_play = np.full((6, 7), self.to_play(), dtype="int32")
        return np.array([board_player1, board_player2, board_to_play])

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        board = self.get_board(observation)
        return board, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return int(self.env.env.state[1].status == "ACTIVE")

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """

        legal = []
        for i in range(7):
            if self.raw_board[0][i] == 0:
                legal.append(i)
        return legal

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        board = self.get_board(observation)
        return board

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.expert_action()

    def expert_action(self):
        board = self.raw_board.copy()
        board[board == 2] = -1
        action = np.random.choice(self.legal_actions())
        player = self.to_play() * 2 - 1
        for k in range(3):
            for l in range(4):
                sub_board = board[k: k + 4, l: l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = np.where(sub_board[i, :] == 0)[0][0]
                        if np.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = np.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = np.where(diag == 0)[0][0]
                    if np.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = np.where(anti_diag == 0)[0][0]
                    if np.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if player * sum(anti_diag) > 0:
                            return action
        return action

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"
