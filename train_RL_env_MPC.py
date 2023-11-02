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
import mpc.mpc as mpc

from test_run_mpc import EnvTrueDynamics, EnvMismatchedDynamics

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, net_action_spec, ctrl_dim, ctrl_horizon_dim, cfg):
    cfg.obs_dim = obs_spec
    cfg.net_action_dim = net_action_spec
    cfg.ctrl_dim = ctrl_dim
    cfg.ctrl_horizon_dim = ctrl_horizon_dim
    cfg.action_range = [float(-1.0), float(1.0)]
    return hydra.utils.instantiate(cfg)

def make_mpc(obs_spec, action_spec, u_lower, u_upper, cfg):
    cfg.n_state = obs_spec[0]
    cfg.n_ctrl = action_spec[0]
    cfg.u_lower =  u_lower
    cfg.u_upper = u_upper
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
                                self.net_action_space, 
                                self.ctrl_dim,
                                self.ctrl_horizon_space, 
                                self.cfg.agent)
        
        # the mpc gets nx and nu, which is the actual state and action dimension of the environment 
        self.mpc_agent = make_mpc(self.train_env.observation_space_dim, 
                                  self.train_env.action_space.shape, 
                                  -self.train_env.max_force,
                                  self.train_env.max_force, 
                                  self.cfg.mpc)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        
    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create target envs and agent
        self.train_env = gym.make(self.cfg.env)
        self.eval_env = gym.make(self.cfg.env)

        # the RL agent will tune the Q and the p matrices
        # the Q matrix is diagonal
        # actor input --> 2*nx + u
        # actor output --> (nx+nu)
        self.ctrl_horizon_space = (self.train_env.action_space.shape[0]*self.cfg.T,)
        self.ctrl_dim = self.train_env.action_space.shape
        self.net_action_space = (self.ctrl_dim[0]+self.train_env.observation_space_dim[0],) # 4 weights parameters for the Q-matrix + 1 for the control penalty

        self.replay_buffer = ReplayBuffer(self.train_env.full_observation_space_dim,
                                          self.train_env.observation_space_dim,
                                          self.net_action_space, int(self.cfg.replay_buffer_size),
                                          self.device)

        self.recorder = Recorder(self.work_dir)

        # initial MPC setup
        self.nu = self.train_env.action_space.shape[0]
        self.N_BATCH = 1
        self.goal_state = torch.tensor(self.eval_env.goal_state) # nx --> this is the goal state or the reference
        self.init_params = np.array([1., 0.1, 1., 0.1, 0.001])
        self.goal_weights = torch.tensor((1., 0.1, 1., 0.1)) # nx --> these are the weights given to each state in the cost-function
        self.ctrl_penalty = 0.001 # controller penalty

        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.nu)))  # nx + nu
        px = -torch.sqrt(self.goal_weights) * self.goal_state # nx
        p = torch.cat((px, torch.zeros(self.nu))) 
        Q = torch.diag(q).repeat(self.cfg.T, self.N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
        p = p.repeat(self.cfg.T, self.N_BATCH, 1)
        self.QP_cost = mpc.QuadCost(Q, p)  # T x B x nx+nu 
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    def unsquash(self, value):
        for i in range(self.net_action_space[0]):
            low = self.cfg.params_range[i][0]
            high = self.cfg.params_range[i][1]
            value[i] = ((value[i]+1.0)/2.0)*(high-low)+low
            value[i] = np.clip(value[i], low, high)

        return value
    
    def calculate_QP_cost(self, params):
        goal_weights = torch.tensor(params[0:4])
        ctrl_penalty = params[-1]

        q = torch.cat((goal_weights, ctrl_penalty * torch.ones(self.nu)))  # nx + nu
        px = -torch.sqrt(goal_weights) * self.goal_state # nx
        p = torch.cat((px, torch.zeros(self.nu)))

        Q = torch.diag(q).repeat(self.cfg.T, self.N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
        p = p.repeat(self.cfg.T, self.N_BATCH, 1)

        return mpc.QuadCost(Q, p)  # T x B x nx+nu 
    
    def eval_MPC(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.recorder.init(self.eval_env, len(self.init_params), self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            _ = self.eval_env.reset()
            done = False
            u_init = None

            while not done:
                state = self.eval_env.state.copy()
                state = torch.tensor(state).view(1, -1)

                # compute action based on current state, dynamics, and cost
                nominal_states, nominal_u, nominal_objs = self.mpc_agent(state, self.QP_cost, EnvMismatchedDynamics(self.eval_env), u_init)
                u = nominal_u[0]  # take first planned action
                u_init = torch.cat((nominal_u[1:], torch.zeros(1, self.N_BATCH, self.nu)), dim=0)
                assert u.ndim == 2 and u.shape[0] == self.eval_env.action_space.shape[0]

                observation, reward, done, info = self.eval_env.step_dynamic(u.detach().cpu().numpy()[0])
                self.recorder.record(info, self.init_params, episode)
                total_reward += reward
                step += 1

            episode += 1
            
        self.recorder.save(f'Initial_MPC_controller')
        print(f"Initial MPC controller reward: {total_reward / episode}")

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.recorder.init(self.eval_env, self.net_action_space[0], self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            observation = self.eval_env.reset()
            self.agent.reset()

            done = False
            u_init = None

            while not done:

                with torch.no_grad(), utils.eval_mode(self.agent):
                    params_squashed = self.agent.act(observation, self.global_step, eval_mode=True)
                    params = self.unsquash(params_squashed) #unsquash params
                    QP_cost_actor = self.calculate_QP_cost(params)

                # take env step
                reward = 0
                discount = 0.99

                for i in range(self.cfg.action_repeat):

                    # MPC start here
                    state = self.eval_env.state.copy()
                    state = torch.tensor(state).view(1, -1)
                    
                    # compute action based on current state, dynamics, and cost
                    nominal_states, nominal_u, nominal_objs = self.mpc_agent(state, 
                                                                            QP_cost_actor, 
                                                                            EnvMismatchedDynamics(self.eval_env), 
                                                                            u_init)
                    u = nominal_u[0]  # take first planned action
                    u_init = torch.cat((nominal_u[1:], torch.zeros(1, self.N_BATCH, self.nu)), dim=0)
                    assert u.ndim == 2 and u.shape[0] == self.eval_env.action_space.shape[0]
                    # MPC end

                    # dynamic step
                    observation, reward_temp, done, info = self.eval_env.step_dynamic(u.detach().cpu().numpy()[0])

                    reward += (reward_temp or 0.0)*discount
                    discount *= self.agent.discount
                    self.recorder.record(info, params, episode)
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
        u_init = None

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
                u_init = None

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                self.eval()

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                params_squashed = self.agent.act(observation, self.global_step, eval_mode=False)
                params = self.unsquash(params_squashed)
                QP_cost_actor = self.calculate_QP_cost(params)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='train')

            # take env step
            reward = 0
            discount = 0.99

            for i in range(self.cfg.action_repeat):
                # MPC start here
                state = self.train_env.state.copy()
                state = torch.tensor(state).view(1, -1)
                
                # compute action based on current state, dynamics, and cost
                nominal_states, nominal_u, nominal_objs = self.mpc_agent(state, 
                                                                        QP_cost_actor, 
                                                                        EnvMismatchedDynamics(self.train_env), 
                                                                        u_init)
                u = nominal_u[0]  # take first planned action
                u_init = torch.cat((nominal_u[1:], torch.zeros(1, self.N_BATCH, self.nu)), dim=0)
                assert u.ndim == 2 and u.shape[0] == self.train_env.action_space.shape[0]
                # MPC end

                # dynamic step
                next_obs, reward_temp, done, _ = self.train_env.step_dynamic(u.detach().cpu().numpy()[0])

                reward += (reward_temp or 0.0)*discount
                discount *= self.agent.discount
                if done:
                    break

            next_state = self.train_env.state.copy()
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.train_env.n_steps else done
            episode_reward += reward
            self.replay_buffer.add(state, observation, params_squashed, reward, next_state, next_obs, done, done_no_max)

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
    from train_RL_env_MPC import Workspace as W
    workspace = W(cfg)
    workspace.eval_MPC()
    workspace.train()

if __name__ == '__main__':
    main()
