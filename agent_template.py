import numpy as np
import torch

from self_play import MCTS
from games.connectx_kaggle import MuZeroConfig
import models


class Player:
    def __init__(self):
        try:
            checkpoint = torch.load("cache/model.checkpoint", map_location="cpu")
            # muzero.load_model(
            #     checkpoint_path="model.checkpoint",
            #     replay_buffer_path=None
            # )
        except FileNotFoundError:
            checkpoint = torch.load("/kaggle_simulations/agent/model.checkpoint", map_location="cpu")
            # muzero.load_model(
            #     checkpoint_path="/kaggle_simulations/agent/model.checkpoint",
            #     replay_buffer_path=None
            # )
        self.config = MuZeroConfig()
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(checkpoint["weights"])
        self.player_num = None

    def get_board(self, obs):
        raw_board = np.asarray(
            obs, dtype=np.int32
        ).reshape(6, 7)
        board_player1 = np.where(raw_board == 1, 1.0, 0.0)
        board_player2 = np.where(raw_board == 2, 1.0, 0.0)
        board_to_play = np.full((6, 7), self.player_num, dtype="int32")
        return np.array([board_player1, board_player2, board_to_play]), raw_board

    def get_next_move(self, obs):
        if self.player_num is None:
            self.player_num = int(obs["mark"]) - 1
        board, raw_board = self.get_board(obs["board"])
        with torch.no_grad():
            root, mcts_info = MCTS(self.config).run(
                self.model,
                board,
                self.legal_actions(raw_board),
                self.player_num,
                True,
            )
        action = self.select_action(
            root,
            0
        )
        return action

    @staticmethod
    def legal_actions(raw_board):
        legal = []
        for i in range(7):
            if raw_board[0][i] == 0:
                legal.append(i)
        return legal

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action


PLAYER = Player()


def my_agent(observation, configuration):
    return PLAYER.get_next_move(observation)
