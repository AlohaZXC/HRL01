# import time
# import gym
# from gym_rooms.envs import *
# environment = 'Rooms-v0'
# env = gym.make(environment)
#
# for i in range(10):
#     s = env.reset()
#     # env.render()
#     for j in range(200):
#         a = env.action_space.sample()
#         sp,r,done,info = env.step(a)
#         # env.render()
#         if env.new_pos_before_passing_doorway in env.hallways:
#             doorway = env.new_pos_before_passing_doorway
#             print('-'*30)
#             print('s = ', s)
#             print('doorway = ', doorway)
#             print('sp =', sp)
#         s = sp

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn import svm
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.cluster_centers_)