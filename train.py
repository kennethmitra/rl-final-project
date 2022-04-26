import numpy as np
import gym
import json
import matplotlib.pyplot as plt
from env import RoboTaxiEnv
from util import  state_action_vec, get_acorn_loc, get_squirrel_loc, get_bomb_loc
from Net import QFunction
import time





def Q(weights, state, action,):
	vec = state_action_vec(state, action)
	return np.dot(weights, vec)


def Q_max_a(weights, state):
	max_Q = -np.inf
	for action in range(1, 5):
		Q_value = Q(weights, state, action)
		if Q_value>max_Q:
			max_Q = Q_value
	return max_Q


def epsilon_greedy_action(weights, state, epsilon):
	if (np.random.rand()<epsilon):
		return np.random.randint(1, 5)
	else:
		Q_value = np.zeros(4)
		for action in range(1, 5):
			Q_value[action-1] = Q(weights, state, action)
		best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max()))+1
		return best_action
	
def epsilon_greedy_action_with_H(weights, state, human_reward_weight, epsilon):
	if (np.random.rand()<epsilon):
		return np.random.randint(1, 5)
	else:
		Q_value = np.zeros(4)
		for action in range(1, 5):
			Q_value[action-1] = Q(weights, state, action) + np.dot(human_reward_weight, state_action_vec(state, action))
		best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max()))+1
		return best_action

def main():
	config = {}
	with open('config.json') as config_file:
		config = json.load(config_file)
	
	env = RoboTaxiEnv(config=config['practice']['gameConfig'])
	num_episodes = 10000
	alpha = 0.01
	alpha_h = 0.1
	gamma = 0.7
	epsilons = [0.1, 0.1, 0.05, 0.05]
	run_till_complete = False

	w = np.zeros(800)
	h = np.zeros(800)
	R = []
	delivered = 0
	died = 0
	
	for episode in range(num_episodes):
		if episode < num_episodes/4:
			epsilon = epsilons[0]
		elif episode < num_episodes/2:
			epsilon = epsilons[1]
		elif episode < num_episodes/4*3:
			epsilon = epsilons[2]
		else:
			epsilon = epsilons[3]

		cum_reward = 0
		done = False
		state, info = env.reset()
		if not run_till_complete:
			env.render()
			time.sleep(1)
		while not done:
			action = epsilon_greedy_action_with_H(w, state, h, epsilon)
			new_state, reward, done, info = env.step(action)
			if not run_till_complete:
				env.render()
				h_r = input("rate the action:\n")
				if h_r == '':
					h_r = 0
				elif h_r == 'q':
					run_till_complete = True
					h_r = 0
				else:
					h_r = int(h_r)
			else:
				h_r = 0

			if reward >= 7:
				delivered +=1
			if reward <= -10:
				died += 1
			td_error = reward+gamma*Q_max_a(w,new_state) - Q(w,state,action)
			w = w + alpha*td_error*state_action_vec(state,action)
			h = h + alpha_h*h_r*state_action_vec(state,action)
			state = new_state
			cum_reward += reward
			
		R.append(cum_reward)
		
		if episode%100 == 0:
			print("trained episode {}".format(episode))
		
	plt.plot(R)
	plt.show()
	print('times deliverd:',delivered)
	print('times died:', died)


if __name__ == '__main__':
	main()