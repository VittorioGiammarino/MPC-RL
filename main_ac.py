# Copyright (c) VG

from pathlib import Path
from collections import deque

import hydra
import gym
import gym_CartPole_BT
import numpy as np

import torch
from utils_folder import utils
from logger_folder.logger import Logger
from buffers.replay_buffer import ReplayBuffer
from record_plot import Recorder

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_dim = obs_spec
    cfg.action_dim = action_spec
    cfg.action_range = [float(-1.0), float(1.0)]
    return hydra.utils.instantiate(cfg)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.full_observation_space_dim,
                                self.train_env.action_space_dim,
                                self.cfg.agent)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        
    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create target envs and agent
        self.train_env = gym.make(self.cfg.env)
        self.eval_env = gym.make(self.cfg.env)

        self.replay_buffer = ReplayBuffer(self.train_env.full_observation_space_dim,
                                          self.train_env.action_space_dim,
                                          int(self.cfg.replay_buffer_size),
                                          self.device)

        self.recorder = Recorder(self.work_dir)
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
        
    def eval_original_controller(self, noise):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.recorder.init(self.eval_env, self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            _ = self.eval_env.reset()

            done = False

            while not done:

                _, reward, done, info = self.eval_env.step_ideal(error = noise)
                self.recorder.record(info, episode)
                total_reward += reward
                step += 1

            episode += 1
            
        self.recorder.save(f'original_controller_noise_{noise}')
        print(f"controller reward w/ param noise {noise}: {total_reward / episode}")

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.recorder.init(self.eval_env, self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            observation = self.eval_env.reset()
            self.agent.reset()

            done = False

            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(observation, self.global_step, eval_mode=True)

                # take env step
                reward = 0
                discount = 1.0
                for i in range(self.cfg.action_repeat):
                    observation, reward_temp, done, info = self.eval_env.step(action)
                    reward += (reward_temp or 0.0)*discount
                    discount *= self.agent.discount
                    self.recorder.record(info, episode)
                    step += 1
                    if done:
                        break

                total_reward += reward

            episode += 1
            
        self.recorder.save(f'{self.global_step}')
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_states, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_states, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_states, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        observation = self.train_env.reset()

        metrics = None
        done = False

        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                observation = self.train_env.reset()

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                self.eval()

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(observation, self.global_step, eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='train')

            # take env step
            reward = 0
            discount = 1.0

            for i in range(self.cfg.action_repeat):
                next_obs, reward_temp, done, _ = self.train_env.step(action)
                reward += (reward_temp or 0.0)*discount
                discount *= self.agent.discount
                if done:
                    break

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.train_env.n_steps else done
            episode_reward += reward
            self.replay_buffer.add(observation, action, reward, next_obs, done,
                                   done_no_max)

            observation = next_obs
            episode_step += 1
            self._global_step += 1
            
    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

@hydra.main(config_path='config_folder', config_name='config_ac')
def main(cfg):
    from main_ac import Workspace as W
    workspace = W(cfg)
    workspace.eval_original_controller(np.array([0, 0, 0, 0]))
    workspace.train()

if __name__ == '__main__':
    main()
