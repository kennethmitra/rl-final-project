from collections import defaultdict

import numpy as np
import gym
import json
import matplotlib.pyplot as plt
from env import RoboTaxiEnv, CellType, loc2tuple
from util import state_action_vec, get_acorn_loc, get_squirrel_loc, get_bomb_loc
from Net import QFunction
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import seaborn as sns


def Q(weights, state, action):
    vec = state_action_vec(state, action)
    return np.dot(weights, vec)


def Q_max_a(weights, state):
    max_Q = -np.inf
    for action in range(1, 5):
        Q_value = Q(weights, state, action)
        if Q_value > max_Q:
            max_Q = Q_value
    return max_Q


def H(weights, state, action):
    vec = state_action_vec(state, action)
    return np.dot(weights, vec)

def visualize_H(h):
    print("H")
    print("No Acorn")
    H_mat_no_acorn = np.zeros((10, 10, 4))
    H_mat_acorn = np.zeros((10, 10, 4))
    for r in range(1, 11):
        for c in range(1, 11):
            for a in range(1, 5):
                H_mat_no_acorn[r-1, c-1, a-1] = H(h, (None, (c, r), False), a)

    print("With Acorn")
    for r in range(1, 11):
        for c in range(1, 11):
            for a in range(1, 5):
                H_mat_acorn[r-1, c-1, a-1] = H(h, (None, (c, r), True), a)

    return H_mat_no_acorn, H_mat_acorn

def visualize_Q(w):
    print("Q")
    print("No Acorn")
    Q_mat_no_acorn = np.zeros((10, 10, 4))
    Q_mat_acorn = np.zeros((10, 10, 4))
    for r in range(1, 11):
        for c in range(1, 11):
            for a in range(1, 5):
                Q_mat_no_acorn[r-1, c-1, a-1] = Q(w, (None, (c, r), False), a)

    print("With Acorn")
    for r in range(1, 11):
        for c in range(1, 11):
            for a in range(1, 5):
                Q_mat_acorn[r-1, c-1, a-1] = Q(w, (None, (c, r), True), a)

    return Q_mat_no_acorn, Q_mat_acorn

def visualize_potential_field(env):
    pf_no_acorn = np.zeros((10, 10))
    pf_acorn = np.zeros((10, 10))
    for r in range(1, 11):
        for c in range(1, 11):
            pf_no_acorn[r-1, c-1] = compute_potential_field(env, (None, (c, r), False))
            pf_acorn[r-1, c-1] = compute_potential_field(env, (None, (c, r), True))
    return pf_no_acorn, pf_acorn

def epsilon_greedy_action(weights, state, epsilon):
    if (np.random.rand() < epsilon):
        return np.random.randint(1, 5)
    else:
        Q_value = np.zeros(4)
        for action in range(1, 5):
            Q_value[action - 1] = Q(weights, state, action)
        best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max())) + 1
        return best_action


def epsilon_greedy_action_with_H(weights, state, human_reward_weight, epsilon, ts, decay_param=1):
    if (np.random.rand() < epsilon):
        return np.random.randint(1, 5)
    else:
        Q_value = np.zeros(4)
        for action in range(1, 5):
            Q_value[action - 1] = Q(weights, state, action) + decay_param**ts * H(human_reward_weight, state, action)
        best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max())) + 1
        return best_action


def control_sharing(weights, state, human_reward_weight, p_human, ts, decay_param=1):
    if (np.random.rand() < decay_param**ts * p_human):
        H_value = np.zeros(4)
        for action in range(1, 5):
            H_value[action - 1] = H(human_reward_weight, state, action)
        best_action = np.random.choice(np.flatnonzero(H_value == H_value.max())) + 1
        return best_action
    else:
        Q_value = np.zeros(4)
        for action in range(1, 5):
            Q_value[action - 1] = Q(weights, state, action)
        best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max())) + 1
        return best_action


def select_action(weights, state, human_reward_weight, epsilon, method, ts, p_human=0.5, decay_param=1):
    if method == "e_greedy":
        return epsilon_greedy_action(weights, state, epsilon)
    elif method == "action_biasing":
        return epsilon_greedy_action_with_H(weights, state, human_reward_weight, epsilon, ts=ts, decay_param=decay_param)
    elif method == "control_sharing":
        return control_sharing(weights, state, human_reward_weight, ts=ts, p_human=p_human, decay_param=decay_param)


sigmoid_fn = lambda x: 1 / (1 + np.exp(-x))


def get_human_reward(env, old_obs, new_obs, simulated=True, soft_rewards=True):
    if not simulated:
        run_till_complete = False
        env.render()
        h_r = input("rate the action:\n")

        if h_r == '':
            h_r = None
        elif h_r == 'q':
            run_till_complete = True
            h_r = None
        else:
            h_r = int(h_r)
    else:
        field_before = compute_potential_field(env, old_obs)
        field_after = compute_potential_field(env, new_obs)

        if soft_rewards:
            # For Soft Rewards
            h_r = 2 * sigmoid_fn(field_before - field_after) - 1  # Scale to (-1, 1)
        else:
            # For Hard Rewards
            h_r = 1 if field_before > field_after else -1

        run_till_complete = False

    return h_r, run_till_complete


def dist(loc1, loc2):
    # return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])  # Manhattan Distance


def compute_potential_field(env, obs):
    BOMB_CONSTANT = 5
    GOAL_CONSTANT = 1
    DIV_EPS = 1e-3

    player_rc = loc2tuple(obs[1])
    has_acorn = obs[2]

    total_potential = 0

    # Bombs repel the player with force proportional to 1/r^2
    for bomb in env.bomb_rcs:
        d = dist(bomb, player_rc)
        total_potential += BOMB_CONSTANT / (d ** 2 + DIV_EPS)

    # Goal attracts the player
    if not has_acorn:
        goal_rc = env.nut_rc
    else:
        goal_rc = env.squirrel_rc

    d = dist(goal_rc, player_rc)
    total_potential += GOAL_CONSTANT * d ** 1.5

    return total_potential


def main(params):
    # Logging Setup
    writer = SummaryWriter()
    writer.add_text("params", str(params))

    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)

    env = RoboTaxiEnv(config=config['practice']['gameConfig'])
    num_episodes = params.get('num_episodes', 2000)
    alpha = params.get('alpha', 0.01)
    alpha_h = params.get('alpha_h', 0.1)
    gamma = params.get('gamma', 0.95) #0.7
    epsilons = params.get('epsilons', [0.1, 0.1, 0.05, 0.05])
    ACTION_SELECTION_METHOD = params.get('ACTION_SELECTION_METHOD', "control_sharing")
    P_HUMAN = params.get('P_HUMAN', 0.3)
    USE_SOFT_REWARDS = params.get('USE_SOFT_REWARDS', True)
    HUMAN_TRAINING_EPISODES = params.get('HUMAN_TRAINING_EPISODES', np.float('inf'))
    DECAY_PARAM = params.get('DECAY_PARAM', 1)
    simulated_human_rewards = True
    run_till_complete = False

    writer.add_text("num_episodes", str(num_episodes))
    writer.add_text("alpha", str(alpha))
    writer.add_text("alpha_h", str(alpha_h))
    writer.add_text("gamma", str(gamma))
    writer.add_text("epsilons", str(epsilons))
    writer.add_text("ACTION_SELECTION_METHOD", ACTION_SELECTION_METHOD)
    writer.add_text("P_HUMAN", str(P_HUMAN))
    writer.add_text("HUMAN_TRAINING_EPISODES", str(HUMAN_TRAINING_EPISODES))
    writer.add_text("DECAY_PARAM", str(DECAY_PARAM))

    w = np.zeros(800)
    h = np.zeros(800)
    R = []
    delivered = 0
    died = 0

    for episode in tqdm(range(num_episodes)):
        if episode < num_episodes / 4:
            epsilon = epsilons[0]
        elif episode < num_episodes / 2:
            epsilon = epsilons[1]
        elif episode < num_episodes / 4 * 3:
            epsilon = epsilons[2]
        else:
            epsilon = epsilons[3]

        cum_reward = 0
        td_error_hist = []
        done = False
        state, info = env.reset()
        if (not run_till_complete and not simulated_human_rewards):
            env.render()
            time.sleep(1)
        while not done:
            action = select_action(w, state, h, epsilon, ts=episode, method=ACTION_SELECTION_METHOD, p_human=P_HUMAN, decay_param=DECAY_PARAM)
            new_state, reward, done, info = env.step(action)
            if not (run_till_complete or episode > HUMAN_TRAINING_EPISODES):
                h_r, run_till_complete = get_human_reward(env, state, new_state, simulated=True, soft_rewards=USE_SOFT_REWARDS)
            else:
                h_r = None

            if reward >= 7:
                delivered += 1
            if reward <= -10:
                died += 1

            if h_r is not None:
                h_pred = H(h, state, action)
                h = h + alpha_h * (h_r - h_pred) * state_action_vec(state, action)

            td_error = H(h, state, action) + gamma * Q_max_a(w, new_state) - Q(w, state, action) #+ reward
            w = w + alpha * td_error * state_action_vec(state, action)

            state = new_state
            cum_reward += reward
            td_error_hist.append(td_error)

        R.append(cum_reward)
        writer.add_scalar('Train/Episode_Reward', cum_reward, episode)
        writer.add_scalar('Train/Episode_Len', len(td_error_hist), episode)
        writer.add_scalar('Params/Epsilon', epsilon, episode)
        writer.add_scalar('Train/td_error', np.array(td_error_hist).mean(), episode)

        # Eval Episode
        done = False
        state, info = env.reset()
        eval_rew_hist = []
        while not done:
            action = select_action(w, state, h, epsilon=0, method='e_greedy', p_human=P_HUMAN, ts=0)  # Deterministic action selection
            new_state, reward, done, info = env.step(action)
            state = new_state
            eval_rew_hist.append(reward)
        writer.add_scalar('Eval/Episode_Reward', np.array(eval_rew_hist).sum(), episode)
        writer.add_scalar('Eval/Episode_Len', len(eval_rew_hist), episode)


        if episode % 100 == 0:
            print("trained episode {}".format(episode))

    plt.plot(R)
    plt.show()
    print('times deliverd:', delivered)
    print('times died:', died)

    action_labels = ["UP", "RIGHT", "DOWN", "LEFT"]
    H_no_acorn, H_acorn = visualize_H(h)
    for action in range(0, 4):
        sns.heatmap(H_no_acorn[:, :, action])
        plt.title(f"H for {action_labels[action]} action NO Acorn")
        writer.add_figure(f"H for {action_labels[action]} action NO Acorn", plt.gcf())
    for action in range(0, 4):
        sns.heatmap(H_acorn[:, :, action])
        plt.title(f"H for {action_labels[action]} action NO Acorn")
        writer.add_figure(f"H for {action_labels[action]} action WITH Acorn", plt.gcf())

    Q_no_acorn, Q_acorn = visualize_Q(w)
    for action in range(0, 4):
        sns.heatmap(Q_no_acorn[:, :, action])
        plt.title(f"Q for {action_labels[action]} action NO Acorn")
        writer.add_figure(f"Q for {action_labels[action]} action NO Acorn", plt.gcf())
    for action in range(0, 4):
        sns.heatmap(Q_acorn[:, :, action])
        plt.title(f"Q for {action_labels[action]} action NO Acorn")
        writer.add_figure(f"Q for {action_labels[action]} action WITH Acorn", plt.gcf())

    # Visualize Potential Field
    pf_no_acorn, pf_acorn = visualize_potential_field(env)
    sns.heatmap(pf_no_acorn, vmax=50)
    plt.title(f"Potential Field NO Acorn")
    writer.add_figure(f"Potential Field No Acorn", plt.gcf())

    sns.heatmap(pf_acorn, vmax=50)
    plt.title(f"Potential Field WITH Acorn")
    writer.add_figure(f"Potential Field WITH Acorn", plt.gcf())


if __name__ == '__main__':

    params_to_try = [
        # {'ACTION_SELECTION_METHOD': 'e_greedy', 'USE_SOFT_REWARDS': True, 'epsilons': [0.1, 0.1, 0.05, 0.05]},
        # {'ACTION_SELECTION_METHOD': 'action_biasing', 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 0.5, 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'e_greedy', 'USE_SOFT_REWARDS': False, 'epsilons': [0.1, 0.1, 0.05, 0.05]},
        # {'ACTION_SELECTION_METHOD': 'action_biasing', 'USE_SOFT_REWARDS': False},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 0.5, 'USE_SOFT_REWARDS': False},

        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.999, 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.999, 'USE_SOFT_REWARDS': False},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.99, 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.99, 'USE_SOFT_REWARDS': False},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.9, 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 0.9, 'USE_SOFT_REWARDS': False},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1, 'USE_SOFT_REWARDS': True},
        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1, 'USE_SOFT_REWARDS': False},

        # {'ACTION_SELECTION_METHOD': 'control_sharing', 'USE_SOFT_REWARDS': True, 'epsilons': [1, 0.75, 0.5, 0.25]},
        # {'ACTION_SELECTION_METHOD': 'e_greedy', 'USE_SOFT_REWARDS': False, 'epsilons': [1, 0.75, 0.5, 0.25]},
        {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1, 'USE_SOFT_REWARDS': True},
        {'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1, 'USE_SOFT_REWARDS': False},
    ]

    for param in params_to_try:
        print(param)
        main(param)
