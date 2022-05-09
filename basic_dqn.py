import gym
from collections import OrderedDict, defaultdict

from stable_baselines3.common.callbacks import CheckpointCallback

from env import RoboTaxiEnv, Action, CellType, Direction, Int2Direction, Direction2Int
import json
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
import torch
from pathlib import Path
import torch.nn.functional as F

saved_game_map = None

def processObs(obs, info):
    """
    Processes observation into state vector for QModel
    :param obs: obs from env
    :return: state vec for qmodel (robot location, robot orientation, bomb locations, squirrel location, nut location, has_acorn)
    """
    # # Extract item locations from map
    # global saved_game_map
    #
    # map_dict = defaultdict(list)
    #
    # if saved_game_map is None:
    #     map = obs[0][1:-1, 1:-1]
    #     saved_game_map = map
    # else:
    #     map = saved_game_map
    # for r in range(map.shape[0]):
    #     for c in range(map.shape[1]):
    #         if map[r, c] != 0:
    #             map_dict[map[r, c]].append((r, c))
    #
    # bomb_locations = map_dict[CellType.BOMB.value]
    # squirrel_location = map_dict[CellType.SQUIRREL.value]
    # nut_location = map_dict[CellType.ACORN.value]

    robot_location = obs[1]
    has_acorn = obs[2]

    # bomb_locations_arr = torch.tensor([item for tup in bomb_locations for item in tup])
    # squirrel_location_arr = torch.tensor(squirrel_location[0])
    # nut_location_arr = torch.tensor(nut_location[0])
    robot_location_arr = torch.tensor(robot_location)
    has_acorn_arr = torch.tensor([has_acorn])
    # robot_orientation_arr = torch.tensor(list(info['player_orientation']))  # F.one_hot(torch.tensor(Direction2Int[Direction(info['player_orientation'])]), num_classes=len(Direction))

    #robot_loc_onehot = F.one_hot(torch.tensor((robot_location_arr[1] - 1) * 10 + (robot_location_arr[0]-1)), num_classes=100)

    # proccessed_feats = torch.cat(
    #     [robot_location_arr, robot_orientation_arr, bomb_locations_arr, squirrel_location_arr, nut_location_arr,
    #      has_acorn_arr])
    proccessed_feats = torch.cat(
        [robot_location_arr, has_acorn_arr])
    return proccessed_feats

class HumanFeedbackSimulator:
    def __init__(self, bomb_locs, acorn_loc, squirrel_loc, strategy='binary_distance'):
        self.strategy = strategy
        self.bomb_locs = bomb_locs
        self.acorn_loc = acorn_loc[0]
        self.squirrel_loc = squirrel_loc[0]

    def feedback(self, state, action, next_state):

        if self.strategy == 'binary_distance':
            next_player_loc = next_state[0:2]
            has_acorn = state[1]

            feedback = 0
            for loc in self.bomb_locs:
                if abs(loc[0] - next_player_loc[0]) + abs(loc[1] - next_player_loc[1]) <= 2:
                    feedback += -0.1
                    break
            if not has_acorn and (abs(next_player_loc[0] - self.acorn_loc[0]) + abs(next_player_loc[1] - self.acorn_loc[1]) <= 2):
                feedback += 0.1
            if has_acorn and (abs(next_player_loc[0] - self.squirrel_loc[0]) + abs(next_player_loc[1] - self.squirrel_loc[1]) <= 2):
                feedback += 0.1
        return feedback

class Wrapper(gym.Env):

    def __init__(self, use_human=False):
        config = {}
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.env = RoboTaxiEnv(config=config['practice']['gameConfig'])

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.ones(3)*0, high=np.concatenate((np.ones(2)*10, np.ones(1))))

        # For human feedback simulation
        self.last_obs = None
        self.use_human = use_human
        self.human_feedback = None

    def reset(self):
        obs, info = self.env.reset()
        feat = processObs(obs, info)
        self.last_obs = feat

        if self.use_human:
            if self.use_human:
                # Extract item locations from map
                map_dict = defaultdict(list)

                map = self.env.map[1:-1, 1:-1]

                for r in range(map.shape[0]):
                    for c in range(map.shape[1]):
                        if map[r, c] != 0:
                            map_dict[map[r, c]].append((r, c))

                bomb_locations = map_dict[CellType.BOMB.value]
                squirrel_location = map_dict[CellType.SQUIRREL.value]
                nut_location = map_dict[CellType.ACORN.value]

                self.human_feedback = HumanFeedbackSimulator(bomb_locs=bomb_locations, squirrel_loc=squirrel_location, acorn_loc=nut_location)
        return feat

    def step(self, action):
        obs, rew, done, info = self.env.step(action + 1)
        feat = processObs(obs, info)

        if self.use_human:
            feedback = self.human_feedback.feedback(self.last_obs, action, feat)
            rew += feedback

        return feat, rew, done, info

    def render(self, mode='human'):
        self.env.render(mode=mode)


if __name__ == '__main__':
    env = Wrapper(use_human=True)
    # env = gym.make('Pendulum-v1')
    save_dir = 'saves'
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix='PPO_es_human')
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./run")
    model.learn(total_timesteps=10000000, log_interval=10, callback=checkpoint_callback, tb_log_name="PPO_es_human")
    model.save(f"{save_dir}/sb3_ts{10000000}")
    # model = PPO.load('saves/PPO_450000_steps.zip')
    #
    # print("Done!")
    #
    # print("No Acorn")
    # policy = [[" " for i in range(10)] for j in range(10)]
    # for r in range(1, 11):
    #     for c in range(1, 11):
    #         feats = processObs((None, (c, r), False), None)
    #         action, _ = model.predict(feats)
    #         policy[r-1][c-1] = ('^', '>', 'v', '<')[action]
    #         print(policy[r-1][c-1], " ", end="")
    #     print()
    #
    # print("With Acorn")
    # policy = [[" " for i in range(10)] for j in range(10)]
    # for r in range(1, 11):
    #     for c in range(1, 11):
    #         feats = processObs((None, (c, r), True), None)
    #         action, _ = model.predict(feats)
    #         policy[r-1][c-1] = ('^', '>', 'v', '<')[action]
    #         print(policy[r-1][c-1], " ", end="")
    #     print()
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()
