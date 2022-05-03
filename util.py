import numpy as np

def get_bomb_loc(state):
	map = state[0]
	bomb_loc = []
	for row in range(map.shape[0]):
		for col in range(map.shape[1]):
			if map[row, col]==2:
				bomb_loc.append((col, row))
	return bomb_loc

def get_acorn_loc(state):
	map = state[0]
	for row in range(map.shape[0]):
		for col in range(map.shape[1]):
			if map[row, col]==3:
				(acorn_x, acorn_y) = (col, row)
				return (acorn_x, acorn_y)
			
def get_squirrel_loc(state):
	map = state[0]
	for row in range(map.shape[0]):
		for col in range(map.shape[1]):
			if map[row, col]==4:
				(squirrel_x, squirrel_y) = (col, row)
				return (squirrel_x, squirrel_y)
'''
def state_action_vec(state,action):
	robot_x,robot_y = state[0]
	acorn_x, acorn_y = state[1]
	squirrel_x, squirrel_y = state[2]
	bomb_loc = state[3]
	bomb_dist = []
	for i in range(3):
		bomb_dist.append(np.linalg.norm([robot_x-bomb_loc[i][0], robot_y-bomb_loc[i][1]]))
	acorn_dist = np.linalg.norm([robot_x-acorn_x, robot_y-acorn_y])
	squirrel_dist = np.linalg.norm([robot_x-squirrel_x, robot_y-squirrel_y])
	has_acorn = int(state[4])
	vec = np.array([robot_x, robot_y, has_acorn,
					acorn_dist * has_acorn, acorn_dist * (1 - has_acorn),
					squirrel_dist * has_acorn, squirrel_dist * (1 - has_acorn)])
	n = len(vec)
	array = np.zeros(4*n)
	array[(action-1)*n:(action-1)*n+n] = vec
	return array
'''
def state_action_vec(state,action):
	robot_x, robot_y = state[1]
	has_acorn = state[2]
	state_action_vec = np.zeros(800)
	state_action_vec[(action-1)*200+has_acorn*100+(robot_y-1)*10+(robot_x-1)] = 1
	return state_action_vec

def print_policy(net, state):
	pass




'''
def state_action_vec(state, action):
	map = state[0]
	(robot_x, robot_y) = state[1]
	(acorn_x, acorn_y) = (0, 0)
	(squirrel_x, squirrel_y) = (0, 0)
	has_acron = state[2]
	acorn_dist = 0.1
	squirrel_dist = 0.1
	bomb_dist = []
	is_bomb = 0
	is_acorn = 0
	is_squirrel = 0
	for row in range(map.shape[0]):
		for col in range(map.shape[1]):
			if map[row, col]==3:
				(acorn_x, acorn_y) = (col, row)
			if map[row, col]==4:
				(squirrel_x, squirrel_y) = (col, row)
	# up
	if (action==1):
		if map[robot_y-1, robot_x]==2:
			is_bomb = 1
		if map[robot_y-1, robot_x]==3:
			is_acorn = 1
		if map[robot_y-1, robot_x]==4:
			is_squirrel = 1
		for row in range(map.shape[0]):
			for col in range(map.shape[1]):
				dist = robot_y-row
				if map[row, col]==2:
					bomb_dist.append(dist)
				if map[row, col]==3:
					acorn_dist = dist
				if map[row, col]==4:
					squirrel_dist = dist
	# right
	elif (action==2):
		if map[robot_y, robot_x+1]==2:
			is_bomb = 1
		if map[robot_y, robot_x+1]==3:
			is_acorn = 1
		if map[robot_y, robot_x+1]==4:
			is_squirrel = 1
		for row in range(map.shape[0]):
			for col in range(map.shape[1]):
				dist = col-robot_x
				if map[row, col]==2:
					bomb_dist.append(dist)
				if map[row, col]==3:
					acorn_dist = dist
				if map[row, col]==4:
					squirrel_dist = dist
	# down
	elif (action==3):
		if map[robot_y+1, robot_x]==2:
			is_bomb = 1
		if map[robot_y+1, robot_x]==3:
			is_acorn = 1
		if map[robot_y+1, robot_x]==4:
			is_squirrel = 1
		for row in range(map.shape[0]):
			for col in range(map.shape[1]):
				dist = row-robot_y
				if map[row, col]==2:
					bomb_dist.append(dist)
				if map[row, col]==3:
					acorn_dist = dist
				if map[row, col]==4:
					squirrel_dist = dist
	# left
	elif (action==4):
		if map[robot_y, robot_x-1]==2:
			is_bomb = 1
		if map[robot_y, robot_x-1]==3:
			is_acorn = 1
		if map[robot_y, robot_x-1]==4:
			is_squirrel = 1
		for row in range(map.shape[0]):
			for col in range(map.shape[1]):
				dist = robot_x-col
				if map[row, col]==2:
					bomb_dist.append(dist)
				if map[row, col]==3:
					acorn_dist = dist
				if map[row, col]==4:
					squirrel_dist = dist
	
	if acorn_dist==0:
		acorn_dist = 100
	if squirrel_dist==0:
		squirrel_dist = 100
	while len(bomb_dist)<3:
		bomb_dist.append(0.1)
	for i in range(3):
		if (bomb_dist[i]==0):
			bomb_dist[i] = 0.1
	d = [1/acorn_dist*(has_acron), 1/acorn_dist*(1-has_acron),
		 1/squirrel_dist*has_acron, 1/squirrel_dist*(1-has_acron),
		 is_bomb,
		 is_acorn,
		 is_squirrel*has_acron, is_acorn*(1-has_acron),
		 robot_x, robot_y]
	
	array = np.array(d)
	return array

'''