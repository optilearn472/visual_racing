# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
# base classes
from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.pybulet_class import World
from matplotlib import pyplot as plt
# others
import numpy as np
import os
import time
import random
import cv2
import torch

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl
from scipy import spatial
import math
from pyglet.gl import GL_POINTS

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params :

            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'Br': 3.31, 'Cr': 1.2, 'Dr': 16.0, 'Bf': 3.05, 'Cf': 1.2, 'Df': 13.0, 'lf': 0.18, 'lr': 0.14, 'h': 0.074, 'm': 3.936, 'I': 0.042, 's_min': -0.4, 's_max': 0.4, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':0.0, 'v_max': 8.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4

        try:
            self.wayp_path = kwargs['waypoint']
        except:
            self.wayp_path = './examples/example_waypoints.csv'

        try:
            self.eval = kwargs['eval_flag']
        except:
            self.eval = 0

        try:
            self.depth_render = kwargs['depth_render']
        except:
            self.depth_render = 0

        try:
            self.max_time = kwargs['max_time']
        except:
            self.max_time = 20

        try:
            self.sdf_path = kwargs['sdf_path']
        except:
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/Barcelona/barcelona.sdf'

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.last_steer_cmd = 0
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        # self.my_waypoint = np.loadtxt(self.wayp_path, delimiter=',', skiprows=1)
        # fake_pro = np.zeros(len(self.my_waypoint)).reshape(len(self.my_waypoint), 1)
        # self.my_waypoint = np.append(self.my_waypoint, fake_pro, axis=1)
        wp = np.load(self.wayp_path)
        self.my_waypoint = np.zeros((wp.shape[0], 5))
        # self.my_waypoint = np.load(self.wayp_path)
        self.my_waypoint[:, 0:2] = wp[:, 0:2]

        process = 0
        for i in range(len(self.my_waypoint)):
            pi_1 = self.my_waypoint[i][0:2]
            pi = self.my_waypoint[i-1][0:2]
            if i > 0:
                process += np.linalg.norm(pi_1 - pi)
            self.my_waypoint[i][4] = process
            dp = pi_1 - pi
            self.my_waypoint[i-1][2] = np.arctan2(dp[1], dp[0])
        for i in range(len(self.my_waypoint)):
            p1 = self.my_waypoint[i-1][0:2]
            p2 = self.my_waypoint[i][0:2]
            if i+1 <= len(self.my_waypoint)-1:
                p3 = self.my_waypoint[i+1][0:2]
            else:
                p3 = self.my_waypoint[0][0:2]
            dis1 = np.linalg.norm(p1 - p2)
            dis2 = np.linalg.norm(p1 - p3)
            dis3 = np.linalg.norm(p2 - p3)
            dis = dis1**2 + dis3**2 - dis2**2
            cosA = dis / (2 * dis1 * dis3)
            cosA = min(cosA, 1.0)
            cosA = max(cosA, -1.0)
            sinA = math.sqrt(1 - cosA**2)
            curv = 0.5 * dis2 / sinA
            curv = 1.0 / curv
            self.my_waypoint[i][3] = curv
        self.max_process = self.my_waypoint[-1][4] + np.linalg.norm(self.my_waypoint[-1][0:2] - self.my_waypoint[0][0:2])

        self.tree = spatial.KDTree(self.my_waypoint[:, 0:2])
        self.last_process = 0
        self.last_action = np.zeros(2)
        self.last_nearest_wp = np.zeros(4)
        self.last_pose = np.zeros(2)
        self.last_near_id = 0

        self.traj_point = []
        self.drawn_waypoints = []
        self.obs = {}
        self.world = World(rendering=False, sdf_path=self.sdf_path)
        self.step_num = 0
        self.dep = np.zeros((64, 96))

        self.dagger_observation = dict()
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(110,), dtype=np.float32)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32)


    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        # done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4) or self.current_time > 60
        done = (self.collisions[self.ego_idx]) or self.current_time > self.max_time
        if self.collisions[self.ego_idx]:
            print("collision done, time:", self.current_time)
        # elif np.all(self.toggle_list >= 4):
        #     print("toggle_list.")
        elif self.current_time > self.max_time:
            print("good episode, time:", self.current_time)

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        ang = np.linspace(-3*np.pi/4, 3*np.pi/4, 1080)
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        colll = False
        for i in range(1080):
            x = cos_ang[i] * obs_dict['scans'][0][i]
            y = sin_ang[i] * obs_dict['scans'][0][i]
            if abs(x) < 0.35 and abs(y) < 0.20:
                colll = True
                break
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = [obs_dict['collisions'][0] or colll]

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # call simulation step
        # throttle = float(0.7 + action[0] * 0.3)
        # print(np.linalg.norm(self.my_waypoint[0][0:2]-self.my_waypoint[1][0:2]))

        throttle = float(action[0])
        ctrl_steer = float(action[1])
        ctrl_input = np.array([[ctrl_steer, throttle]])
        # ctrl_input = np.array([[0,0]])
        for i in range(10):
            obs = self.sim.step(ctrl_input)
            if obs['collisions'][0] > 0:
                # print('quick break')
                break
        self.obs = obs
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        ax = obs['acc_x'][0]
        ay = obs['acc_y'][0]

        if self.depth_render == 1:
            pos = (obs['poses_x'][0], obs['poses_y'][0], 0.05)
            orn = (0, 0, obs['poses_theta'][0])
            pp = (pos, orn)
            rgb, dep, p_collision = self.world.set_vehicle_pose(pp, ax, ay)
            # if p_collision == True:
            #     obs['collisions'][0] = 1.0
            self.dep = np.array(dep)
            # np.save('dep', self.dep)
            # self.dep[self.dep>5.0] = 5.0
            self.dep[self.dep>6.0] = 6.0
            # print(np.min(self.dep))
            # self.dep = self.dep / 5.0
            self.dep = self.dep / 6.0
            # print(self.dep)
            # indices = np.random.choice(self.dep.shape[1]*self.dep.shape[0], replace=False, size=int(self.dep.shape[1]*self.dep.shape[0]*0.2))
            # ddp = self.dep
            # ddp[np.unravel_index(indices, self.dep.shape)] = 0 

            # dep = np.clip(dep / 100.0, 0, 1) / 1.5 * 255
            # im_color = cv2.applyColorMap(dep.astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow("", self.dep)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     assert False
        self.step_num += 1
        
        t1 = time.time()
        # print('f1tenth step', t1-t0)

        self.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        cur_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        cur_pos += np.array([random.gauss(0, 0.001), random.gauss(0, 0.001)])
        self.traj_point.append(cur_pos)
        cur_yaw = obs['poses_theta'][0]
        cur_yaw += random.gauss(0, 0.002)
        
        # print(self.tree.data.shape, self.my_waypoint.shape)
        min_d, min_index = self.tree.query(cur_pos, workers=4)
        min_index = np.clip(min_index, 0, len(self.my_waypoint)-1)
        nearest_waypoint = np.array(self.my_waypoint[min_index])
        dp_vec = cur_pos - nearest_waypoint[0:2]
        near_vec = np.array([np.cos(nearest_waypoint[2]), np.sin(nearest_waypoint[2])])
        if np.dot(dp_vec, near_vec) >= 0:
            nearest_pose = nearest_waypoint[0:2] + np.dot(dp_vec, near_vec) * near_vec
            nearest_yaw = nearest_waypoint[2]
            curve_id = min_index
            nearest_waypoint[0:2] = nearest_pose
            near_id = min_index
        else:
            nearest_waypoint = np.array(self.my_waypoint[min_index - 1])
            dp_vec = cur_pos - nearest_waypoint[0:2]
            near_vec = np.array([np.cos(nearest_waypoint[2]), np.sin(nearest_waypoint[2])])
            nearest_pose = nearest_waypoint[0:2] + np.dot(dp_vec, near_vec) * near_vec
            nearest_yaw = nearest_waypoint[2]
            curve_id = min_index - 1
            nearest_waypoint[0:2] = nearest_pose
            near_id = min_index - 1

        # yaw误差（-pi, pi]
        e_yaw = nearest_yaw - cur_yaw
        e_yaw = (e_yaw + np.pi) % (2 * np.pi) - np.pi
        vv_yaw = e_yaw
        # 横向误差
        x0 = nearest_pose[0]
        y0 = nearest_pose[1]
        x_near = cur_pos[0]
        y_near = cur_pos[1]
        dx = x_near - x0
        dy = y_near - y0
        x_vec = np.cos(nearest_yaw)
        y_vec = np.sin(nearest_yaw)
        e_y = dx * y_vec - dy * x_vec
        vv_y = e_y
        # 在车体坐标系下的速度、角速度
        v = obs['linear_vels_x'][0]
        v_body = np.array([obs['linear_vels_x'][0], obs['linear_vels_y'][0]])
        v_body += np.array([random.gauss(0, 0.005), random.gauss(0, 0.005)])
        self.steer = obs['steer_ang'][0]
        steer = self.last_steer_cmd * 0.4
        # steer += random.gauss(0, 0.002)
        w = obs['ang_vels_z'][0]
        w += random.gauss(0, 0.02)
        
        # 参考点
        rm = np.array([[np.cos(cur_yaw), np.sin(cur_yaw)],
                       [-np.sin(cur_yaw), np.cos(cur_yaw)]])
        point_list = []
        pp_list = []
        num_waypoint = len(self.my_waypoint)
        point_id = min_index
        while len(point_list) < 30:
            point_id += 1
            if point_id > num_waypoint - 1:
                point_id = point_id - num_waypoint
            x = self.my_waypoint[point_id][0] - cur_pos[0]
            y = self.my_waypoint[point_id][1] - cur_pos[1]
            dp = np.array([x, y])
            dp = rm @ dp
            point_list.append(dp[0])
            point_list.append(dp[1])
            pp_list.append(dp)
        pp_list = np.array(pp_list)
        point_list = np.array(point_list)
        # plt.plot(pp_list[:, 0], pp_list[:, 1])
        # plt.show()
        # sys.exit()
        point_list = point_list / 5

        # scan
        raw_scan = obs['scans'][0]
        raw_scan = np.clip(raw_scan, 0.0, 15.0)
        

        scann = []
        for i in range(raw_scan.shape[0]):
            if i % 15 == 0:
                if i+15 >= raw_scan.shape[0]:
                    idd = raw_scan.shape[0]
                else:
                    idd = i+15
                scann.append(np.min(raw_scan[i:idd]) / 15.0)
        scann = np.array(scann)

        v_body = v_body / 6.0
        l_a = np.array(self.last_action)
        steer = steer / (np.pi/3)
        e_yaw = e_yaw / np.pi
        e_y = e_y / 2.0
        w = w / 8.0
        cur_obs = np.hstack((e_y, e_yaw, steer, v_body, w, point_list, l_a, scann))
        cur_obs = cur_obs.astype(np.float32)

        # update data member
        self._update_state(obs)
        self.current_time += self.timestep * 10

        # check done
        done, toggle_list = self._check_done()
        info = {'get_collision': self.collisions[self.ego_idx]}

        near_id_wp = self.my_waypoint[near_id]
        process = np.linalg.norm(nearest_waypoint[0:2] - near_id_wp[0:2]) + near_id_wp[4]
        # process = near_id_wp[4]
        delta_process = process - self.last_process
        if delta_process < -(self.max_process - 10):
            delta_process += self.max_process
        if delta_process > (self.max_process - 10):
            delta_process -= self.max_process

        if abs(delta_process) > 3:
            print("####################################")
            print("kd tree result wrong!!!!!!!!!")
            print(delta_process)
            print("cur_process=", process, 'last_process=', self.last_process)
            print("cur_pose = ", cur_pos, "nearest_pose = ", nearest_pose)
            print("last_pose = ", self.last_pose, "last_nearest = ", self.last_nearest_wp)
            print(np.min(raw_scan))
            print(self.map_name)
            print(self.current_time)
            sys.exit()
            

        reward = delta_process - 0.2 * abs(ctrl_steer - self.last_action[1])
        if self.collisions[self.ego_idx]:
            # reward -= 10.0
            # reward -= 0.5
            reward -= 0.5 * np.linalg.norm(np.array([obs['linear_vels_x'][0], obs['linear_vels_y'][0]]))**2
        if min_d > 2.0:
            # reward -= 10.0
            reward -= 0.5 * np.linalg.norm(np.array([obs['linear_vels_x'][0], obs['linear_vels_y'][0]]))**2
            done = True
            print("not get collision.")
        # reward = (obs['linear_vels_x'][0] * np.cos(vv_yaw) - obs['linear_vels_y'][0] * np.sin(vv_yaw)) / 8.0 - abs(vv_y)
        # if self.collisions[self.ego_idx]:
        #     # reward -= 10.0
        #     reward -= 0.5 * np.linalg.norm(np.array([obs['linear_vels_x'][0], obs['linear_vels_y'][0]]))**2

        self.last_action = action
        self.last_pose = cur_pos
        self.last_nearest_wp = nearest_waypoint
        self.last_near_id = near_id
        self.last_process = process
        self.last_steer_cmd = ctrl_steer

        cur_obs = np.clip(cur_obs, -1.0, 1.0)
        # print("v = ", cur_obs[3:5]*50, "w = ", cur_obs[5]*4)

        obs_nan = np.isnan(cur_obs)
        for x in obs_nan:
            if x:
                print(cur_obs)
                sys.exit()
        obs_inf = np.isinf(cur_obs)
        for x in obs_inf:
            if x:
                print(cur_obs)
                sys.exit()

        self.dagger_observation['depth'] = self.dep
        real_dep = self.dep + 0.02  * np.random.standard_normal(size=(64, 96))
        xx = 0.08 + random.gauss(0, 0.05)
        xx = np.clip(xx, 0.005, 0.15)
        indices = np.random.choice(real_dep.shape[1]*real_dep.shape[0], replace=False, size=int(real_dep.shape[1]*real_dep.shape[0]*xx))
        real_dep[np.unravel_index(indices, real_dep.shape)] = 0
        # self.dagger_observation['real_depth'] = real_dep
        self.dagger_observation['real_depth'] = self.dep
        all_img = np.zeros((2, 64, 96))
        all_img[0] = self.dep
        all_img[1] = real_dep
        self.dagger_observation['all_depth'] = all_img
        self.dagger_observation['vel'] = v_body
        self.dagger_observation['omega'] = np.array([w])
        self.dagger_observation['steer'] = np.array([steer])
        self.dagger_observation['last_a'] = l_a
        
        return cur_obs, reward, done, info

    def reset(self):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        random_key = random.random()
        # if random_key < 0.25:
        #     list = ['1', '2', '3', '4', '5', '6']
        # else:
        #     list = ['austria', 'barcelona', 'mco', 'gbr']
        list = ['austria', 'barcelona', 'mco', 'gbr']
        x = random.choice(list)

        direction = ['1', '2']
        dir = random.random()
        if dir <= 0.5:
            d = '1'
        else:
            d = '2'

        if x == 'austria':
            self.map_name = './examples/rl_race/f1tenth_racetracks/austria/aut'
            self.map_path = self.map_name + '.yaml'
            self.map_ext = '.png'
            self.wp_path = './examples/rl_race/f1tenth_racetracks/austria/centerline' + d + '.npy'
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/austria/austria.sdf'
        elif x == 'barcelona':
            self.map_name = './examples/rl_race/f1tenth_racetracks/Barcelona/barcelona_map'
            self.map_path = self.map_name + '.yaml'
            self.map_ext = '.png'
            self.wp_path = './examples/rl_race/f1tenth_racetracks/Barcelona/centerline' + d + '.npy'
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/Barcelona/barcelona.sdf'
        elif x == 'mco':
            self.map_name = './examples/rl_race/f1tenth_racetracks/mco/mco'
            self.map_path = self.map_name + '.yaml'
            self.map_ext = '.png'
            self.wp_path = './examples/rl_race/f1tenth_racetracks/mco/centerline' + d + '.npy'
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/mco/mco.sdf'
        elif x == 'gbr':
            self.map_name = './examples/rl_race/f1tenth_racetracks/gbr/gbr'
            self.map_path = self.map_name + '.yaml'
            self.map_ext = '.png'
            self.wp_path = './examples/rl_race/f1tenth_racetracks/gbr/centerline' + d + '.npy'
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/gbr/gbr.sdf'
        else:
            self.map_name = './examples/rl_race/f1tenth_racetracks/ex' + x + '/ex'
            self.map_path = self.map_name + '.yaml'
            self.map_ext = '.png'
            self.wp_path = './examples/rl_race/f1tenth_racetracks/ex' + x + '/center' + d + '.npy'
            self.sdf_path = './examples/rl_race/f1tenth_racetracks/ex' + x + '/ex.sdf'

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator)
        self.sim.set_map(self.map_path, self.map_ext)
        wp = np.load(self.wp_path)
        self.init_waypoint(wp)
        self.world.close()
        self.world = World(rendering=False, sdf_path=self.sdf_path)
        self.world.reset_cam2body()
        if self.renderer is not None:
            self.renderer.on_close()
            self.renderer = None

        if self.eval == 0:
            random_id = random.randint(0, len(self.my_waypoint)-1)
            # random_id = 0
        elif self.eval == 1:
            # random_id = random.randint(-180, 0) #esp
            # random_id = random.choice(self.random_list) # aut
            random_id = -200 #mco
        else:
            min_d, random_id = self.tree.query(np.array([0,0]))

        wp = self.my_waypoint[random_id]
        poses = np.array([[wp[0], wp[1], wp[2]]])
        self.last_nearest_pose = np.array(wp)
        self.last_pose = np.array([wp[0], wp[1]])
        self.last_near_id = 0
        self.last_process = wp[4]
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.step_num = 0
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        
        self.last_action = np.zeros(2)
        self.last_steer_cmd = 0
        self.traj_point.clear()

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros(2)
        obs, reward, done, info = self.step(action)

        self.render_obs = {
            'ego_idx': self.ego_idx,
            'poses_x': self.poses_x,
            'poses_y': self.poses_y,
            'poses_theta': self.poses_theta,
            'lap_times': self.lap_times,
            'lap_counts': self.lap_counts
            }

        return obs
    
    def get_observation(self):
        depth = torch.from_numpy(self.dagger_observation['real_depth']).float().to(self.device)
        v_body = torch.from_numpy(self.dagger_observation['vel']).float().to(self.device)
        w = torch.from_numpy(self.dagger_observation['omega']).float().to(self.device)
        steer = torch.from_numpy(self.dagger_observation['steer']).float().to(self.device)
        l_a = torch.from_numpy(self.dagger_observation['last_a']).float().to(self.device)
        all_img = torch.from_numpy(self.dagger_observation['all_depth']).float().to(self.device)
        obs = dict()
        body_perception = torch.cat([v_body, w, steer, l_a], dim=-1)
        # body_perception = torch.cat([v_body, w, l_a], dim=-1)
        obs['depth'] = depth.unsqueeze(0)
        obs['body'] = body_perception.unsqueeze(0)
        obs['all_depth'] = all_img.unsqueeze(1)
        return obs


    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']

        if self.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            self.renderer.update_map(self.map_name, self.map_ext)

        self.renderer.update_obs(self.render_obs)

        for render_callback in self.render_callbacks:
            render_callback(self.renderer)

        points = np.array(self.traj_point)
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = self.renderer.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]


        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass

    def init_waypoint(self, wp):
        self.my_waypoint = np.zeros((wp.shape[0], 5))
        self.my_waypoint[:, 0:2] = wp[:, 0:2]
        process = 0
        for i in range(len(self.my_waypoint)):
            pi_1 = self.my_waypoint[i][0:2]
            pi = self.my_waypoint[i-1][0:2]
            if i > 0:
                process += np.linalg.norm(pi_1 - pi)
            self.my_waypoint[i][4] = process
            dp = pi_1 - pi
            self.my_waypoint[i-1][2] = np.arctan2(dp[1], dp[0])
        for i in range(len(self.my_waypoint)):
            p1 = self.my_waypoint[i-1][0:2]
            p2 = self.my_waypoint[i][0:2]
            if i+1 <= len(self.my_waypoint)-1:
                p3 = self.my_waypoint[i+1][0:2]
            else:
                p3 = self.my_waypoint[0][0:2]
            dis1 = np.linalg.norm(p1 - p2)
            dis2 = np.linalg.norm(p1 - p3)
            dis3 = np.linalg.norm(p2 - p3)
            dis = dis1**2 + dis3**2 - dis2**2
            cosA = dis / (2 * dis1 * dis3)
            cosA = min(cosA, 1.0)
            cosA = max(cosA, -1.0)
            sinA = math.sqrt(1 - cosA**2)
            curv = 0.5 * dis2 / sinA
            curv = 1.0 / curv
            self.my_waypoint[i][3] = curv
        self.max_process = self.my_waypoint[-1][4] + np.linalg.norm(self.my_waypoint[-1][0:2] - self.my_waypoint[0][0:2])
        self.tree = spatial.KDTree(self.my_waypoint[:, 0:2])

