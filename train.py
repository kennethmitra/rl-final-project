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

def epsilon_greedy_action(weights, state, epsilon):
    if (np.random.rand() < epsilon):
        return np.random.randint(1, 5)
    else:
        Q_value = np.zeros(4)
        for action in range(1, 5):
            Q_value[action - 1] = Q(weights, state, action)
        best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max())) + 1
        return best_action


def epsilon_greedy_action_with_H(weights, state, human_reward_weight, epsilon):
    if (np.random.rand() < epsilon):
        return np.random.randint(1, 5)
    else:
        Q_value = np.zeros(4)
        for action in range(1, 5):
            Q_value[action - 1] = Q(weights, state, action) + H(human_reward_weight, state, action)
        best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max())) + 1
        return best_action


def control_sharing(weights, state, human_reward_weight, p_human):
    if (np.random.rand() < p_human):
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


def select_action(weights, state, human_reward_weight, epsilon, method, p_human=0.5):
    if method == "e_greedy":
        return epsilon_greedy_action(weights, state, epsilon)
    elif method == "action_biasing":
        return epsilon_greedy_action_with_H(weights, state, human_reward_weight, epsilon)
    elif method == "control_sharing":
        return control_sharing(weights, state, human_reward_weight, p_human=p_human)


sigmoid_fn = lambda x: 1 / (1 + np.exp(-x))


def get_human_reward(env, old_obs, new_obs, simulated=True):
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

        # For Soft Rewards
        h_r = 2 * sigmoid_fn(field_before - field_after) - 1  # Scale to (-1, 1)

        # For Hard Rewards
        # h_r = 1 if field_before > field_after else -1

        run_till_complete = False

    return h_r, run_till_complete


def dist(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def compute_potential_field(env, obs):
    BOMB_CONSTANT = 0.5
    GOAL_CONSTANT = 0.25
    DIV_EPS = 1e-6

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
    total_potential += GOAL_CONSTANT * d ** 2

    return total_potential


def main():
    # Logging Setup
    writer = SummaryWriter()

    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)

    env = RoboTaxiEnv(config=config['practice']['gameConfig'])
    num_episodes = 10000
    alpha = 0.01
    alpha_h = 0.1
    gamma = 0.95#0.7
    epsilons = [0.1, 0.1, 0.05, 0.05]
    simulated_human_rewards = True
    run_till_complete = False

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
            action = select_action(w, state, h, epsilon, method="e_greedy", p_human=0.5)
            new_state, reward, done, info = env.step(action)
            if not run_till_complete:
                # env.render()
                h_r, run_till_complete = get_human_reward(env, state, new_state, simulated=True)
                # print("Human Reward:", h_r)
                # time.sleep(2)
            else:
                h_r = None

            if reward >= 7:
                delivered += 1
            if reward <= -10:
                died += 1

            td_error = reward + gamma * Q_max_a(w, new_state) - Q(w, state, action)
            w = w + alpha * td_error * state_action_vec(state, action)

            if h_r is not None:
                h_pred = H(h, state, action)
                h = h + alpha_h * (h_r - h_pred) * state_action_vec(state, action)

            state = new_state
            cum_reward += reward
            td_error_hist.append(td_error)

        R.append(cum_reward)
        writer.add_scalar('Reward/Episode_Reward', cum_reward, episode)
        writer.add_scalar('Params/Epsilon', epsilon, episode)
        writer.add_scalar('Loss/td_error', np.array(td_error_hist).mean(), episode)

        if episode % 100 == 0:
            print("trained episode {}".format(episode))

    plt.plot(R)
    plt.show()
    print('times deliverd:', delivered)
    print('times died:', died)

    H_no_acorn, H_acorn = visualize_H(h)
    sns.heatmap(H_no_acorn[:, :, 0])
    plt.title("Learned H for UP action")
    plt.show()

    Q_no_acorn, Q_acorn = visualize_Q(w)
    sns.heatmap(Q_no_acorn[:, :, 0])
    plt.title("Learned Q for UP action")
    plt.show()


if __name__ == '__main__':
    main()
