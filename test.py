from drl_utils.env_utils.make_env import make_vec_env
from drl_utils.env_utils.evaluate import evaluate_policy
from omegaconf import OmegaConf
import torch as th
import numpy as np 
import pathlib
import hydra
import os
from stable_baselines3.ppo import PPO
from typing import Deque
from crowd_sim.envs.utils.info import *
import argparse

def callback_info(infos, dones, info_deque: Deque):
    if dones:
        for info in infos:
            if isinstance(info['info'], Collision):
                info_deque.append('Collision')
            elif isinstance(info['info'], Timeout):
                info_deque.append('Timeout')
            elif isinstance(info['info'], ReachGoal):
                info_deque.append('ReachGoal')

class TrainWorkSpace():
    def __init__(self, config: OmegaConf, args):

        self.testing(config,os.path.join(os.getcwd(), 'model_weight', f'{args[-1]}'), args[0] , args[1]) 

    def testing(self, config, name, n_eval_episodes ,render):
        th.manual_seed(config.default.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(config.default.seed)
            th.cuda.manual_seed(config.default.seed)
        rng = np.random.default_rng(config.default.seed)

        vecenv = make_vec_env(
                     env_name=f'{config.default.env_name}',
                     n_envs = 1,
                     rng = rng,
                     parallel = True, 
                     max_episode_steps = int(config.system.time_limit/ config.system.time_step),
                     config = config,
                     phase='test',
                     )  
        
        learner = PPO.load(name, env = vecenv)  ## TODO
        test_info = evaluate_policy(learner, vecenv, callback= callback_info , render=render, n_eval_episodes=n_eval_episodes, )
        
        print(f'[Name]:{name.split("/")[-3]}, {test_info}')
        vecenv.close()
        del vecenv

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'crowd_nav', 'configs')),
    config_name='base_config.yaml'
)

def main(cfg):
    # parser = argparse.ArgumentParser(description="test.py args")

    # parser.add_argument("--n_eval", type=int, help="set number of eval", default = 100)
    # parser.add_argument("--render", type=bool, help="set visualize", default = True)
    # parser.add_argument("--weight_path", type=str, help="set weight path")
    n_eval= 100 
    render = True
    weight_path = 'My_Alogrithm'

    # args = parser.parse_args()
    args= [n_eval, render,weight_path]
    TrainWorkSpace(cfg, args)

if __name__ == "__main__":
    main()
