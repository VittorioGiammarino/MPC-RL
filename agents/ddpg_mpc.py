import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.nn.functional import normalize

import mpc.mpc as mpc

from utils_folder import utils

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, low=0.001, high=1):
        super().__init__()

        self.low = low
        self.high = high

        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.trunk(obs)
        mu = torch.sigmoid(mu)
        std = torch.ones_like(mu) * std

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.TruncatedNormal(mu, std, self.low, self.high)
        return dist

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

class DDPG_Agent:
    """DDPG algorithm."""
    def __init__(self, obs_dim, net_action_dim, ctrl_dim, ctrl_horizon_dim, action_range, params_range, 
                 device, hidden_dim, hidden_depth, discount, actor_lr, actor_betas, actor_update_frequency, 
                 critic_lr, critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, use_tb, num_expl_steps, stddev_schedule, stddev_clip):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau

        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.ctrl_dim = ctrl_dim 
        self.ctrl_horizon_dim = ctrl_horizon_dim
        self.state_dim = net_action_dim[0] - ctrl_dim[0]
        self.params_range = params_range

        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = DoubleQCritic(obs_dim[0], ctrl_horizon_dim[0], hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim[0], ctrl_horizon_dim[0], hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DDPGActor(obs_dim[0], net_action_dim[0], hidden_dim, hidden_depth, params_range[0], params_range[1]).to(self.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.train()
        self.critic_target.train()
        
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(self.action_range[0], self.action_range[1])
        return action.cpu().numpy()[0]
    
    def calculate_QP_cost(self, params, time_horizon, goal_state):
        goal_weights = params[:, 0:self.state_dim] # --> (batch_size, n_goal_weights)
        ctrl_penalty = params[:, self.state_dim:].reshape(-1, self.ctrl_dim[0]) # --> (batch_size, ctrl_dim)
        assert params[0, self.state_dim:].shape[0] == self.ctrl_dim[0]

        q = torch.cat((goal_weights, ctrl_penalty * torch.ones(self.batch_size, self.ctrl_dim[0]).to(self.device)), dim=1) # --> (batch_size, n_goal_weights + 1)
        px = -torch.sqrt(goal_weights) * goal_state.repeat(self.batch_size,1).to(self.device) # --> (batch_size, n_goal_weights)
        p = torch.cat((px, torch.zeros(self.batch_size, self.ctrl_dim[0]).to(self.device)), dim=1) # --> (batch_size, n_goal_weights + ctrl_dim)

        Q_matrices = []
        p_vectors = []
        for i in range(self.batch_size):
            Q_matrices.append(torch.diag(q[i,:]).repeat(time_horizon, 1, 1, 1))
            p_vectors.append(p[i,:].repeat(time_horizon, 1, 1))

        Q = torch.cat(Q_matrices, dim=1)  # --> (time_horizon, batch_size, n_goal_weights + ctrl_dim, n_goal_weights + ctrl_dim)
        p = torch.cat(p_vectors, dim=1) # --> (time_horizon, batch_size, n_goal_weights + ctrl_dim)
       
        return mpc.QuadCost(Q, p) 

    def update_critic(self, state, obs, u, reward, next_state, next_obs, not_done, 
                      model, mpc_agent, time_horizon, goal_state, step):
        metrics = dict()

        # shapes from buffer
        # state shape --> (batch_size, state_dim)
        # obs shape --> (batch_size, obs_dim)
        # u shape --> (batch_size, u_dim = time_horizon)
        # reward shape --> (batch_size, 1)

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_params = dist.sample(clip=self.stddev_clip) # this ensures that the Q function is positive definite and the values normalized

        QP_cost = self.calculate_QP_cost(next_params, time_horizon, goal_state)
        u_init = u.reshape(-1, self.batch_size, self.ctrl_dim[0]) # u_init shape --> (time_horizon, batch_size, ctrl_dim)
        _, next_nominal_u, _ = mpc_agent(next_state, QP_cost, model, u_init=u_init) # next_nominal_u shape --> (time_horizon, batch_size, ctrl_dim)
        next_nominal_u = next_nominal_u.reshape(self.batch_size, self.ctrl_horizon_dim[0]) # we reshape next_nominal_u to --> (batch_size, u_dim = time_horizon)
        
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_obs, next_nominal_u.detach())
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done *self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, u)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        assert not torch.isnan(critic_loss)

        if self.use_tb:
            metrics["critic_loss"] = critic_loss.item()
            metrics["critic_q1"] = current_Q1.mean().item()
            metrics["critic_q2"] = current_Q2.mean().item()
            metrics["critic_target_q"] = target_Q.mean().item()

        return metrics

    def update_actor(self, state, obs, model, mpc_agent, time_horizon, goal_state, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        params = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(params).sum(-1, keepdim=True)

        QP_cost = self.calculate_QP_cost(params, time_horizon, goal_state)

        _, nominal_u, _ = mpc_agent(state, QP_cost, model, u_init=None) # nominal_u shape --> (time_horizon, batch_size, ctrl_dim)
        nominal_u = nominal_u.reshape(self.batch_size, self.ctrl_horizon_dim[0]) # we reshape nominal_u to --> (batch_size, u_dim = time_horizon)

        Q1, Q2 = self.critic(obs, nominal_u)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        assert not torch.isnan(actor_loss)

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_buffer, model, mpc_agent, time_horizon, goal_state, step):
        metrics = dict()

        state, obs, action, reward, next_state, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(state, obs, action, reward, next_state, next_obs, not_done_no_max, 
                                          model, mpc_agent, time_horizon, goal_state, step))

        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor(state, obs, model, mpc_agent, time_horizon, goal_state, step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return metrics
