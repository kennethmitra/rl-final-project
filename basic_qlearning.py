import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from env import RoboTaxiEnv, Action, CellType, Direction, Int2Direction, Direction2Int
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class QModel(torch.nn.Module):

    def __init__(self, input_len, hidden_size=20, n_actions=5) -> None:
        super().__init__()
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.layers = torch.nn.Sequential(OrderedDict([
            # ('layer1', torch.nn.Linear(self.input_len, self.hidden_size)),
            # ('relu', torch.nn.ReLU()),
            # ('layer2', torch.nn.Linear(self.hidden_size, 1)),
            ('layer1', torch.nn.Linear(self.input_len, 1)),
        ]))

    def forward(self, X):
        return self.layers(X)

    def getQvals(self, state_vec):
        """
        Return array of Q values for each action in current state
        :param state_vec: Feature vector of state
        :return: Array of len # actions
        """
        q_vals = torch.zeros(self.n_actions)
        for act_no in range(self.n_actions):
            action_vec = torch.zeros(self.n_actions)  # One hot vector of action
            action_vec[act_no] = 1
            feats = torch.cat([state_vec, action_vec])
            q_val = self.forward(feats)
            q_vals[act_no] = q_val
        return q_vals

    def get_Q_for_S_A(self, state_vec, action_no):
        """
        Get Q(S, A) for specific S, A
        :param state_vec: states as processed vector
        :param action_no: Action as integer from 0 to len(acts) - 1
        :return: Tensor
        """
        pass

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

    robot_loc_onehot = F.one_hot(torch.tensor((robot_location_arr[1] - 1) * 10 + (robot_location_arr[0]-1)), num_classes=100)

    # proccessed_feats = torch.cat(
    #     [robot_location_arr, robot_orientation_arr, bomb_locations_arr, squirrel_location_arr, nut_location_arr,
    #      has_acorn_arr])
    proccessed_feats = torch.cat(
        [robot_loc_onehot, has_acorn_arr])
    return proccessed_feats


if __name__ == '__main__':
    # http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_7_advanced_q_learning.pdf Double Q-learning?

    # Logging Setup
    writer = SummaryWriter()

    # Initial Setup
    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)
    env = RoboTaxiEnv(config=config['practice']['gameConfig'])
    qModel = QModel(input_len=101+4, hidden_size=15, n_actions=4)
    optim = torch.optim.Adam(params=qModel.parameters(), lr=1e-1)
    loss_fn = torch.nn.MSELoss()

    # Training Loop
    NUM_TS = 200000
    EPSILON = 0.1
    GAMMA = 0.99
    N_ACTIONS = 4

    episode_over = True
    episode_rews = []
    episode_acts = []
    eps_num = 0
    for ts in range(NUM_TS):

        # Reset if needed
        if episode_over:
            eps_sum = np.sum(np.array(episode_rews))
            writer.add_scalar('Reward/Episode_Rew', eps_sum, eps_num)
            eps_num += 1
            print(f"ts: {ts} / {NUM_TS} \t|\t Episode Rew: {eps_sum}")
            episode_rews = []
            episode_acts = []
            obs, info = env.reset()

        # Get Q vals
        state_vec = processObs(obs=obs, info=info)
        q_vals = qModel.getQvals(state_vec=state_vec)

        # e-greedy action selection
        if torch.rand(1).item() < EPSILON:
            action = torch.randint(low=0, high=N_ACTIONS, size=(1,)).item()
        else:
            action = torch.argmax(q_vals)

        episode_acts.append(action)

        # Take action
        new_obs, rew, episode_over, new_info = env.step(action=action+1)
        new_state_vec = processObs(obs=new_obs, info=new_info)

        # env.render()
        # print(new_obs)

        # Learn 1 step
        bootstrap_target = rew + GAMMA * torch.max(qModel.getQvals(state_vec=new_state_vec))
        loss = loss_fn(q_vals[action], bootstrap_target.detach())
        writer.add_scalar('Loss/loss', loss.item(), ts)

        action_arr = np.array(episode_acts)
        writer.add_histogram('Other/action_dist', action_arr, ts)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Log data
        episode_rews.append(rew)

        # Transfer variables
        obs = new_obs
        info = new_info
        state_vec = new_state_vec

    save_dir = 'saves'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    torch.save({'qModel': qModel, 'optim': optim, 'ts': ts}, f'{save_dir}/basic_qlearn_rs_ts{ts}.save')

    qModel = torch.load('saves/basic_qlearn_ts199999.save')['qModel']

    print("No Acorn")
    policy = [[" " for i in range(10)] for j in range(10)]
    for r in range(1, 11):
        for c in range(1, 11):
            feats = processObs((None, (c, r), False), None)
            action_vals = qModel.getQvals(feats)
            action = torch.argmax(action_vals)
            policy[r-1][c-1] = ('^', '>', 'v', '<')[action]
            print(policy[r-1][c-1], " ", end="")
        print()

    print("With Acorn")
    policy = [[" " for i in range(10)] for j in range(10)]
    for r in range(1, 11):
        for c in range(1, 11):
            feats = processObs((None, (c, r), False), None)
            action_vals = qModel.getQvals(feats)
            action = torch.argmax(action_vals)
            policy[r-1][c-1] = ('^', '>', 'v', '<')[action]
            print(policy[r-1][c-1], " ", end="")
        print()