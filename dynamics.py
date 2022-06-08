import numpy as np
import numpy.random as random

import gym
import fym
from fym.core import BaseEnv, BaseSystem


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_init(num_sample=5):
    initials = []
    random.seed(0)
    while True:
        tmp = np.vstack((3, 3, np.pi)) * (2 * np.random.rand(3,1) -1)
        if np.sqrt(np.sum(tmp[:2, :]**2, axis=0)) < 3:
            initials.append(tmp)
        if len(initials) == num_sample:
            break
    return initials


class ThreeDOF(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(shape=(2,1))
        self.yaw = BaseSystem(shape=(1,1))
        self.V = 1

    def set_dot(self, u, disturbance):
        yaw = self.yaw.state.squeeze()
        self.pos.dot = np.vstack((
            self.V * np.cos(yaw),
            self.V * np.sin(yaw)
        )) + disturbance
        self.yaw.dot = u / self.V


class Env(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.plant = ThreeDOF()

        self.action_space = gym.spaces.Box(
            low=-3,
            high=3,
            shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,1)
        )

    def reset(self, initial="random"):
        if initial == "random":
            self.plant.initial_state = np.vstack((3, 3, np.pi)) * (
                2*random.rand(*self.plant.state.shape) - 1
            )
        else:
            self.plant.initial_state = initial
        super().reset()
        obs = self.observe()
        return obs

    def observe(self):
        posx, posy, yaw = self.plant.state.squeeze()
        r = np.sqrt(posx**2 + posy**2)
        e_r = r - 1.0
        theta = np.arctan2(posy, posx)
        sin_theta = posy / r
        cos_theta = posx / r
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        # yaw = wrap(yaw)
        # x = np.vstack((e_r, theta, yaw))
        x = np.vstack((e_r, cos_theta, sin_theta, cos_yaw, sin_yaw))
        # x = np.vstack((e_r, posx, posy, cos_yaw, sin_yaw))
        # x = np.vstack((posx, posy, cos_yaw, sin_yaw))
        obs = np.float32(x)
        return obs

    def step(self, action):
        pre_obs = self.observe()
        u = np.vstack(action)
        *_, done = self.update(u=u)
        next_obs = self.observe()
        reward = self.get_reward(pre_obs, next_obs, u)
        info = {}
        # done = self.terminate(done, next_obs)
        return next_obs, reward, done, info

    def set_dot(self, t, u):
        disturbance = np.vstack((
            0,
            0
        ))
        # disturbance = random.normal(0, 0.1, (2,1))
        # disturbance = np.vstack((
        #     0.5 * np.sin(t),
        #     0.5 * np.cos(t)
        # ))
        self.plant.set_dot(u, disturbance)
        state = self.observe()
        lyap = state[0][0]**2
        return dict(t=t, **self.observe_dict(), action=u,
                    obs=self.observe(), disturbance=disturbance, lyap=lyap)

    def get_reward(self, pre_obs, next_obs, u):
        # reward = self.lyapunov(pre_obs, next_obs)
        # reward = self.add_lyapunov(pre_obs, next_obs, u)
        reward = self.modified_lyapunov(pre_obs, next_obs, u)
        return reward

    def lyapunov(self, pre_obs, next_obs):
        # r_pre = np.sqrt(pre_obs[0][0]**2 + pre_obs[1][0]**2)
        # r_next = np.sqrt(next_obs[0][0]**2 + next_obs[1][0]**2)
        # del_lyap = (r_next - 1.0)**2 - (r_pre - 1.0)**2
        
        e_r_pre = pre_obs[0][0]
        e_r_next = next_obs[0][0]
        del_lyap = e_r_next**2 - e_r_pre**2
        if del_lyap <= 1e-6:
            reward = 10
        else:
            reward = 1
        return reward

    def add_lyapunov(self, pre_obs, next_obs, u):
        e_r_pre = pre_obs[0][0]
        e_r_next = next_obs[0][0]
        del_lyap = e_r_next**2 - e_r_pre**2
        exp = np.float32(np.exp(
            (
                -e_r_pre.T @ e_r_pre
                -u.T @ np.diag([0.1]) @ u
            ).item()
        ))
        linear = -np.abs(u)/3 + 1
        if del_lyap <= 1e-6:
            reward = 10 + 5*exp
            # reward = -2 + linear.item()
        else:
            # reward = 1 + exp
            reward = 1
        return reward

    def modified_lyapunov(self, pre_obs, next_obs, u):
        # r_pre = np.sqrt(pre_obs[0][0]**2 + pre_obs[1][0]**2)
        # r_next = np.sqrt(next_obs[0][0]**2 + next_obs[1][0]**2)
        # del_lyap = (r_next - 1.0)**2 - (r_pre - 1.0)**2
        
        e_r_pre = pre_obs[0]
        e_r_next = next_obs[0]
        del_lyap = e_r_next**2 - e_r_pre**2
        exp = np.float32(np.exp(
            (
                -e_r_pre.T @ np.diag([1]) @ e_r_pre
                -u.T @ np.diag([0.1]) @ u
            ).item()
        ))
        if del_lyap.item() <= 1e-6:
            reward = -1 + exp
        else:
            reward = -10
        return reward
