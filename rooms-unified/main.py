import time

import numpy as np

start_time = time.time()

from ExperienceReplayMemory import ExperienceReplayMemory
experience_memory = ExperienceReplayMemory(memory_size=10000)

from SubgoalDiscovery import SubgoalDiscovery
subgoal_discovery = SubgoalDiscovery(n_clusters=8,experience_memory=experience_memory)

import gym
from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)

#
# obs = env.reset()
# for i in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     time.sleep(0.01)
# env.close()

from trainer import RandomWalk
random_walk = RandomWalk(env=env,subgoal_discovery=subgoal_discovery,experience_memory=experience_memory)

# lets random walk and find the subgoals such as centroids and outliers
#random_walk.walk()
random_walk.walk()
outliers = subgoal_discovery.outliers
centroids = subgoal_discovery.centroid_subgoals
subgoals = subgoal_discovery.G

randomwalk_USD_time = time.time()
subgoal_discovery.report()
print(subgoals)
print('Elapse time for subgoal discovery: ', randomwalk_USD_time-start_time)

# 把目标先全部映射成这个：东西五维： 坐标x,y 房间号（二维），央视
#
# 通过数据，构建矩阵，svd分解，取前几个特征值（这样可以保证信息损失少）
#神经网络输入就是这个特征值，可以表示子目标，又有表示子目标的关联

# 直接padding，后面慢慢换

# 编码，还不知道怎么做
# A = subgoals
# print(A)
# print('len: ',len(A))
# g = 0
# F = []
# by_down = 5
# by_up = 6
# bx_left = 8
# bx_right = 9
# for a in A:
# 	a[0] = a[0] / 14
# 	if a[0] <= 7:
# 		continue
# 	a[1] = a[1] / 14
# for i in A:
# 	for j in A:
# 		# print('i:',i)
# 		# print('j',j)
# 		if i == j:
# 			continue
# 		num = float(np.dot(i,j))
# 		denom = np.linalg.norm(i) * np.linalg.norm(j)
# 		if denom == 0:
# 			f =0
# 		else:
# 			f = num/denom
# 		F.append(f)
# 	t = np.argmax(F)
# 	q = np.max(F)
# 	print(i,'max similarity:',q,'is:',A[t])
#
# 	F=[]
# print(g)
# from hrl import Controller
# controller = Controller(subgoal_discovery=subgoal_discovery)
#
# env.cross_hallway = True
# from trainer import PretrainController
# pretainer = PretrainController( env=env,
#  								controller=controller,
#  								subgoal_discovery=subgoal_discovery)
# pretainer.train()
# # pretainer.controller.Q.save_model()
#
# # pretainer.controller.Q.load_model()
# from hrl import MetaController
# meta_controller = MetaController(subgoal_discovery=subgoal_discovery)
#
# from trainer import MetaControllerController
# meta_controller_trainer = MetaControllerController( env=env,
# 								controller=pretainer.controller,
# 								meta_controller=meta_controller,
# 								subgoal_discovery=subgoal_discovery)
#
# meta_controller_trainer.train()


#NO
# from trainer import MetaControllerControllerUnified
# meta_controller_controller_trainer = MetaControllerControllerUnified( env=env,
# 									controller=pretainer.controller,
# 									meta_controller=meta_controller,
# 									subgoal_discovery=subgoal_discovery)
#
# meta_controller_controller_trainer.train()
#
#
# from trainer import VanillaRL
# vanilla_rl = VanillaRL(env=env)
# vanilla_rl.train()

