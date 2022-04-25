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

def main():
	config = {}
	with open('config.json') as config_file:
		config = json.load(config_file)
	
	env = RoboTaxiEnv(config=config['practice']['gameConfig'])
	num_episodes = 10000
	alpha = 0.01
	gamma = 0.7
	epsilons = [0.3, 0.2, 0.1, 0.05]
	#epsilon = 0.05
	#q_func = QFunction(28)
	#q_func.load("weights/28features32x16model.pt")

	#w = np.zeros(10)
	#w = np.load("weights/10params_100episodes.npy")
	w = np.zeros(800)
	R = []
	delivered = 0
	
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
		#acorn_loc = get_acorn_loc(state)
		#squirrel_loc = get_squirrel_loc(state)
		#bomb_loc = get_bomb_loc(state)
		#state_mod = [state[1],acorn_loc,squirrel_loc, bomb_loc,state[2]]
		
		while not done:
			#env.render()
			#time.sleep(0.5)
			action = epsilon_greedy_action(w, state, epsilon)
			new_state, reward, done, info = env.step(action)
			#new_state_mod = [new_state[1], acorn_loc, squirrel_loc, bomb_loc, new_state[2]]
			if reward >= 7:
				delivered +=1
			td_error = reward+gamma*Q_max_a(w,new_state) - Q(w,state,action)
			w = w + alpha*td_error*state_action_vec(state,action)
			state = new_state
			cum_reward += reward
		R.append(cum_reward)
		
		if episode%100 == 0:
			print("trained episode {}".format(episode))
		
	plt.plot(R)
	plt.show()
	print(delivered)
	#q_func.save("weights/40features32x16model_map2.pt")
	np.save("weights/800onehot",w)

if __name__ == '__main__':
	main()