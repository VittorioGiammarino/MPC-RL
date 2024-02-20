import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_folder import utils
from utils_folder.utils import SquashedNormal

class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, 
                               hidden_dim, 
                               2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

class SingleValueFunction(nn.Module):
    """Value function network."""
    def __init__(self, obs_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.V = utils.mlp(obs_dim, 
                           hidden_dim, 
                           1, 
                           hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        v = self.V(obs)
        return v

class PPOAgent:
    """PPO algorithm"""
    def __init__(self, 
                 obs_dim, 
                 net_action_dim, 
                 ctrl_dim, 
                 ctrl_horizon_dim, 
                 action_range, 
                 device, 
                 hidden_dim, 
                 hidden_depth, 
                 discount,
                 actor_lr, 
                 actor_betas, 
                 value_lr,
                 value_betas, 
                 batch_size, 
                 log_std_bounds, 
                 use_tb,
                 gae_gamma,
                 gae_lambda,
                 epsilon,
                 TD_enable,
                 c1,
                 c2,
                 entropy_enable
                 ):
        
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.batch_size = batch_size
        self.use_tb = use_tb
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.TD_enable = TD_enable
        self.c1 = c1
        self.c2 = c2
        self.entropy_enable = entropy_enable

        # unused here since PPO assumes a distribution as output of the actor. This clashes with the MPC fomulation.
        # No differentiable MPC can be used with PPO 
        self.ctrl_dim = ctrl_dim  
        self.ctrl_horizon_dim = ctrl_horizon_dim

        self.value = SingleValueFunction(obs_dim[0], hidden_dim, hidden_depth).to(self.device)
        self.actor = DiagGaussianActor(obs_dim[0], net_action_dim[0], hidden_dim, hidden_depth, log_std_bounds).to(self.device)
                           
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.value_optimizer = torch.optim.Adam(self.value.parameters(),
                                                 lr=value_lr,
                                                 betas=value_betas)
        
        self.Total_t = 0
        self.Total_iter = 0

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.value.train(training)
                
    def act(self, obs, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if not eval_mode else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0]), dist
    
    def GAE(self, done, rewards, values):
        
        number_of_steps = len(done)
        last_value = values[-1]
        last_advantage = 0
        last_return = 0
        
        advantages = np.zeros((number_of_steps,), dtype=np.float32) 
        returns = np.zeros((number_of_steps,), dtype=np.float32) 
        
        for t in reversed(range(number_of_steps)):
            
            mask = 1-done[t]
            last_value = mask*last_value
            last_advantage = mask*last_advantage
            last_return = mask*last_return
            
            delta = rewards[t]+self.gae_gamma*last_value-values[t]
            
            last_advantage = delta + self.gae_gamma*self.gae_lambda*last_advantage
            last_return = rewards[t] + self.gae_gamma*last_return
            
            advantages[t] = last_advantage
            returns[t] = last_return
            
            last_value = values[t]
            
        return advantages, returns
    
    def normalize_adv(self, adv):
        return (adv - adv.mean())/(adv.std()+1e-8)
    
    def ppo_loss(self, log_pi, log_pi_old, advantage):
        ratio = torch.exp(log_pi - log_pi_old)
        clipped = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon)
        policy_loss = torch.minimum(ratio*advantage, clipped*advantage)
        
        return policy_loss, ratio, clipped
    
    def value_loss(self, value, values_old, returns):
        if self.TD_enable:
            clipped_value = (value - values_old).clamp(-self.epsilon, self.epsilon)
            value_loss = torch.max((value-returns)**2,(clipped_value-returns)**2)
        else:
            value_loss = (value-returns)**2
            
        return value_loss
    
    def compute_loss(self, minibatch):
        metrics = dict()
        
        value = self.value(minibatch['obs'])

        if self.TD_enable:
            returns = minibatch['values'] - minibatch['advantages'].reshape(minibatch['values'].shape)
            L_vf = self.value_loss(value, minibatch['values'], returns)
        else:
            returns = minibatch['returns'].reshape(value.shape)
            L_vf = self.value_loss(value, minibatch['values'], returns)
        
        normalize_advantage = self.normalize_adv(minibatch['advantages'])
        dist = self.actor(minibatch["obs"])
        log_pi = dist.log_prob(minibatch["actions"])
        
        L_clip, ratio, clipped = self.ppo_loss(log_pi, minibatch['log_pis'], normalize_advantage.reshape(-1,1))
        
        diff_ratio = ratio-clipped
        diff_log_pi = log_pi - minibatch['log_pis']
        
        entropy_bonus = -dist.log_prob(minibatch["actions"]).sum(-1, keepdim=True)
        if self.entropy_enable:
            loss = (-1)*(L_clip - self.c1*L_vf + self.c2*entropy_bonus).mean()
        else:
            loss = (-1)*(L_clip - self.c1*L_vf).mean()
        
        if self.use_tb:
            metrics['returns'] = returns.mean().item()
            metrics['advantage'] = minibatch['advantages'].mean().item()
            metrics['normalized_advantage'] = normalize_advantage.mean().item()
            metrics['actor_log_prob_old'] = minibatch['log_pis'].mean().item()
            metrics['ratio'] = ratio.mean().item() 
            metrics['value'] = value.mean().item()
            metrics['actor_loss'] = L_clip.mean().item()
            metrics['actor_log_prob'] = log_pi.mean().item()
            metrics['actor_ent'] = entropy_bonus.mean().item()
            metrics['value_loss'] = L_vf.mean().item()
            metrics['diff_ratio_clipped_ratio'] = diff_ratio.mean().item()
            metrics['diff_log_pi'] = diff_log_pi.mean().item()
        
        return loss, L_clip, L_vf, entropy_bonus, metrics
    
    def update(self, minibatch):
        self.train()
        
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        loss, L_clip, L_vf, entropy_bonus, metrics = self.compute_loss(minibatch)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        
        self.actor_optimizer.step()
        self.value_optimizer.step()
        
        new_loss, new_L_clip, new_L_vf, new_entropy_bonus, _ = self.compute_loss(minibatch)
        
        diff_loss = loss-new_loss
        diff_L_clip = L_clip-new_L_clip
        diff_L_vf = L_vf-new_L_vf
        diff_entropy = entropy_bonus-new_entropy_bonus
        
        if self.use_tb:
            metrics['diff_loss_after_backprop'] = diff_loss.mean().item() 
            metrics['diff_L_clip_after_backprop'] = diff_L_clip.mean().item() 
            metrics['diff_L_vf_after_backprop'] = diff_L_vf.mean().item() 
            metrics['diff_entropy_after_backprop'] = diff_entropy.mean().item() 
        
        return metrics    
        
        
        
        
