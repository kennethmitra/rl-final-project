import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import state_action_vec


class Net(nn.Module):
	def __init__(self, input_size, hidden_layers=(32, 16), output_size=1):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_layers[0])
		self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
		self.fc3 = nn.Linear(hidden_layers[1], output_size)
	
	def forward(self, x):
		x = torch.from_numpy(x).double()
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class QFunction():
	def __init__(self,
				 state_dims):
		"""
		state_dims: the number of dimensions of state space
		"""
		self.model = Net(state_dims).double()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
		self.criterion = torch.nn.MSELoss()
	
	def __call__(self, state, action):
		s = state_action_vec(state, action)
		self.model.eval()
		return self.model(s).detach().numpy()[0]
	
	def value(self, state, action):
		s = state_action_vec(state, action)
		self.model.eval()
		return self.model(s)
	
	def q_max(self,state):
		max_q = -np.inf
		for action in range(1,5):
			q = self.__call__(state,action)
			if(q>max_q):
				max_q = q
		return max_q
	
	def epsilon_greedy_action(self,state, epsilon):
		if (np.random.rand()<epsilon):
			return np.random.randint(1, 5)
		else:
			Q_value = np.zeros(4)
			for action in range(1, 5):
				Q_value[action-1] = self.__call__(state,action)
			best_action = np.random.choice(np.flatnonzero(Q_value == Q_value.max()))+1
			return best_action
	
	def update(self, G, state,action):
		self.optimizer.zero_grad()
		self.model.train()
		output = self.value(state,action)
		loss = self.criterion(output, torch.tensor([G]))
		loss.backward()
		self.optimizer.step()
		return None
	
	def save(self,path):
		torch.save(self.model.state_dict(), path)
	
	def load(self,path):
		self.model.load_state_dict(torch.load(path))
		