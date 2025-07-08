import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gym
import numpy as np
import pybullet as p
from gym import logger
from typing import Tuple
import time
import cv2

Position = Tuple[float, float, float] # x, y, z coordinates
Orientation = Tuple[float, float, float] # euler angles
Pose = Tuple[Position, Orientation]
Quaternion = Tuple[float, float, float, float]
Velocity = Tuple[float, float, float, float, float, float]

class Vehicle:
    def __init__(self):
        self._id = None
        self.urdf = './examples/rl_race/racecar/racecar.urdf'

    @property
    def id(self) -> Any:
        return self._id
    
    def set_vehicle_pose(self, pose: Pose):
        if not self._id:
            self._id = self._load_model(model=self.urdf, initial_pose=pose)
        else:
            pos, orn = pose
            p.resetBasePositionAndOrientation(self._id, pos, p.getQuaternionFromEuler(orn))
    
    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = p.getQuaternionFromEuler(orientation)
        id = p.loadURDF(model, position, orientation)
        # p.changeVisualShape(id, -1, rgbaColor=self._config.color)
        return id


class World:
    def __init__(self, rendering: bool, sdf_path: str):
        self._client = None
        self.rendering = rendering
        # self.sdf = './examples/rl_race/f1tenth_racetracks/Barcelona/barcelona.sdf'
        self.sdf = sdf_path
        self.vehicle = Vehicle()
        self.wall_id = None
        self.init()
        self._up_vector = [0, 0, 1]
        self._camera_vector = [1, 0, 0]
        self._target_distance = 1
        self._fov = 90
        self._near_plane = 0.01
        self._far_plane = 100
        self.width = 96
        self.height = 64
        self.cam2body_matrix = None
        self.cam2body_vec = None
        self.random_flag = True
        self.reset_cam2body()

    def init(self) -> None:
        if self.rendering:
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)

        self._load_scene(self.sdf)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(0.01)
        p.setGravity(0, 0, -9.81)

    def close(self):
        p.disconnect(self._client)

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        self.wall_id = ids
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def reset_cam2body(self):
        if self.random_flag:
            cam2body_roll = random.gauss(0, 0.5) / 180 * np.pi
            cam2body_roll = np.clip(cam2body_roll, -1.0 / 180 * np.pi, 1.0 / 180 * np.pi)
            cam2body_pitch = random.gauss(0, 0.5) / 180 * np.pi
            cam2body_pitch = np.clip(cam2body_pitch, -1.0 / 180 * np.pi, 1.0 / 180 * np.pi)
            cam2body_yaw = random.gauss(0, 1.0) / 180 * np.pi
            cam2body_yaw = np.clip(cam2body_yaw, -2.0 / 180 * np.pi, 2.0 / 180 * np.pi)
        else:
            cam2body_roll = 0.0
            cam2body_pitch = 0.0
            cam2body_yaw = 0.0
        orn = p.getQuaternionFromEuler([cam2body_roll, cam2body_pitch, cam2body_yaw])
        self.cam2body_matrix = p.getMatrixFromQuaternion(orn)
        self.cam2body_matrix = np.array(self.cam2body_matrix).reshape(3, 3)
        if self.random_flag:
            cam2body_x = random.gauss(0, 0.01)
            cam2body_x = np.clip(cam2body_x, -0.02, 0.02)
            cam2body_y = random.gauss(0, 0.01)
            cam2body_y = np.clip(cam2body_y, -0.02, 0.02)
            cam2body_z = random.gauss(0, 0.002)
            cam2body_z = np.clip(cam2body_z, -0.005, 0.005)
        else:
            cam2body_x = 0.0
            cam2body_y = 0.0
            cam2body_z = 0.0
        self.cam2body_vec = np.array([cam2body_x, cam2body_y, cam2body_z])

    def calc_roll_pitch(self, ax, ay):
        ar = 14.647334364296569
        br = 0.09095105207162381
        ap = -2.35362401868478
        bp = 0.5363117108565544
        cp = -0.5000000000000
        roll_noise = random.gauss(0, 0.1)
        roll_noise = np.clip(roll_noise, -0.2, 0.2)
        roll = ar * np.sin(1.0 * np.arctan(br * ay)) + roll_noise
        pitch_noise = random.gauss(0, 0.1)
        pitch_noise = np.clip(pitch_noise, -0.2, 0.2)
        pitch = ap * np.sin(1.0 * np.arctan(bp * ax)) + cp + pitch_noise
        if not self.random_flag:
            roll = 0.0
            pitch = 0.0
        return roll / 180 * np.pi, pitch / 180 * np.pi

    def set_vehicle_pose(self, pose: Pose, ax, ay):
        
        
        ppp, ooo = pose
        yaw = ooo[2]
        roll, pitch = self.calc_roll_pitch(ax, ay)
        self.vehicle.set_vehicle_pose(pose)
        car_pos, orn = pose
        car_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(car_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        cam2car = [0.16 + self.cam2body_vec[0], self.cam2body_vec[1], 0]
        cam_pos = rot_matrix.dot(cam2car)
        cam_pos[0] += car_pos[0]
        cam_pos[1] += car_pos[1]
        cam_pos = (cam_pos[0], cam_pos[1], 0.16)

        width, height = self.width, self.height
        rot_matrix = rot_matrix @ self.cam2body_matrix
        camera_vector = rot_matrix.dot(self._camera_vector)
        up_vector = rot_matrix.dot(self._up_vector)
        target = cam_pos + self._target_distance * camera_vector
        view_matrix = p.computeViewMatrix(cam_pos, target, up_vector)
        aspect_ratio = float(width) / float(height)
        proj_matrix = p.computeProjectionMatrixFOV(self._fov, aspect_ratio, self._near_plane, self._far_plane)
        (_, _, px, depth, _) = p.getCameraImage(width=width,
                                            height=height,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            # renderer=p.ER_TINY_RENDERER,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix)

        rgb_array = np.reshape(px, (height, width, -1))
        rgb_array = rgb_array[:, :, :3]
        depth = self._far_plane * self._near_plane / (self._far_plane - (self._far_plane - self._near_plane) * depth)
        depth = np.reshape(depth, (height, width))
        # depth = cv2.resize(depth, (96, 64), interpolation = cv2.INTER_AREA)
        # depth = depth[6:70, 9:105] 
        
        # p_min, p_max = p.getAABB(self.vehicle.id)
        # id_tuple = p.getOverlappingObjects(p_min, p_max)

        # print(id_tuple)
        # # print(p.getContactPoints(bodyA=self.wall_id, bodyB=self.vehicle.id))
        # collision = bool(p.getContactPoints(bodyA=self.wall_id[1], bodyB=self.vehicle.id))
        # if collision:
        #     print('--------------fucking wall------------------')
        p.stepSimulation()
        collision = False
        points = set([c[2] for c in p.getContactPoints(self.vehicle.id)])
        for point in points:
            if point == 0:
                collision = True
                # print('--------------fucking wall------------------')
        
        return rgb_array, depth, collision

if __name__ == '__main__':
    # sdf = './examples/rl_race/f1tenth_racetracks/austria/austria.sdf'
    sdf = './examples/rl_race/f1tenth_racetracks/ex7/ex.sdf'

    world = World(rendering=True, sdf_path=sdf)
    start = time.time()
    end = time.time()
    while end - start < 5000:
        p.stepSimulation()
        end = time.time()
        pos = (10.0 * np.sin( np.pi * (end - start))+3, 3, 0.0)
        # pos = (0, 0, 0.05)
        orn = (0, 0, 0)
        pp = (pos, orn)
        rgb, dep = world.set_vehicle_pose(pp, 0, 0)
        dep[dep>6.0] = 6.0
        cv2.imshow("", dep / 6.0)
        cv2.imwrite('x1.png', dep / 6.0 * 255)
        key = cv2.waitKey(20)
        if key == 27:
            assert False
        time.sleep(0.01)
    world.close()

    sdf = './examples/rl_race/f1tenth_racetracks/ex6/ex.sdf'

    world = World(rendering=True, sdf_path=sdf)
    start = time.time()
    end = time.time()
    while end - start < 10:
        end = time.time()
        pos = (3.0 * np.sin( np.pi * (end - start))+3, 3, 0.05)
        # pos = (0, 0, 0.05)
        orn = (0, 0, 0)
        pp = (pos, orn)
        rgb, dep = world.set_vehicle_pose(pp)
        dep[dep>6.0] = 6.0
        cv2.imshow("", dep / 6.0)
        cv2.imwrite('x1.png', dep / 6.0 * 255)
        key = cv2.waitKey(20)
        if key == 27:
            assert False
        time.sleep(0.01)
    world.close()

