from pybullet_envs import gym_pendulum_envs, gym_manipulator_envs, gym_locomotion_envs
from gym.envs import classic_control, box2d, mujoco
from .continuous_acrobot import ContinuousAcrobotEnv
from .continuous_pendubot import ContinuousPendubotEnv
from .gym_monitor_nodone import Monitor

env_list = {
    'HalfCheetahEnv' : mujoco.HalfCheetahEnv,
    'HopperEnv' : mujoco.HopperEnv,
    'AntBulletEnv' : gym_locomotion_envs.AntBulletEnv,
    'PendulumEnv' : classic_control.PendulumEnv,
    'InvertedPendulumBulletEnv' : gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    'AcrobotEnv' : ContinuousAcrobotEnv,
    'PendubotEnv' : ContinuousPendubotEnv,
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str
