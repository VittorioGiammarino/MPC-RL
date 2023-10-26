import logging
import math
import time

import gym
import gym_CartPole_BT
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
import mpc.mpc as mpc

from gym_CartPole_BT.envs.cartpole_bt_env import angle_normalize

class EnvTrueDynamics(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.g = env.gravity # -10.0
        self.masscart = env.masscart # 5.0
        self.masspole = env.masspole # 1.0
        self.length = env.length # 2.0
        self.friction = env.friction # 1.0

        self.dt = env.tau
        self.max_force = env.max_force

    def forward(self, state, action):
        """Simulates the non-linear dynamics of a simple cart-pendulum system.
        These non-linear ordinary differential equations (ODEs) return the
        time-derivative at time t given the current state of the system.

        Args:
            t (float): Time variable - not used here but included for
                compatibility with solvers like scipy.integrate.solve_ivp.
            x (np.array): State vector. This should be an array of
                shape (4, ) containing the current state of the system.
                y[0] is the x-position of the cart, y[1] is the velocity
                of the cart (dx/dt), y[2] is the angle of the pendulum
                (theta) from the vertical in radians, and y[3] is the
                rate of change of theta (dtheta/dt).
            m (float): Mass of pendulum.
            M (float): Mass of cart.
            L (float): Length of pendulum.
            g (float): Acceleration due to gravity.
            d (float): Damping coefficient for friction between cart and
                ground.
            u (float): Force on cart in x-direction.

        Returns:
            dx (np.array): The time derivate of the state (dx/dt) as an
                array of shape (4,).
        """
        x = state[:,0].view(-1,1)
        xdot = state[:,1].view(-1,1)
        th = state[:,2].view(-1,1)
        thdot = state[:,3].view(-1,1)

        u = action
        u = torch.clamp(u, -self.max_force, self.max_force)
        
        # Temporary variables
        sin_x = torch.sin(th)
        cos_x = torch.cos(th)
        mL = self.masspole * self.length
        D = 1 / (self.length * (self.masscart + self.masspole * (1 - cos_x**2)))
        b = mL * thdot**2 * sin_x - self.friction * xdot + u

        # Non-linear ordinary differential equations describing
        # simple cart-pendulum system dynamics
        new_xdot= xdot + self.dt * D * (-mL * self.g * cos_x * sin_x + self.length * b)
        new_x = x + self.dt * new_xdot
        new_thdot = thdot + self.dt * D * ((self.masspole + self.masscart) * self.g * sin_x - cos_x * b)
        new_th = th + self.dt * new_thdot

        state = torch.cat((new_x, new_xdot, new_th, new_thdot), dim=1)
        return state

class EnvMismatchedDynamics(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.g = -10 # -10.0
        self.masscart = 4.5 # 5.0
        self.masspole = 1.1 # 1.0
        self.length = 2.1 # 2.0
        self.friction = 2.0 # 1.0

        self.dt = env.tau
        self.max_force = env.max_force

    def forward(self, state, action):
        """Simulates the non-linear dynamics of a simple cart-pendulum system.
        These non-linear ordinary differential equations (ODEs) return the
        time-derivative at time t given the current state of the system.

        Args:
            t (float): Time variable - not used here but included for
                compatibility with solvers like scipy.integrate.solve_ivp.
            x (np.array): State vector. This should be an array of
                shape (4, ) containing the current state of the system.
                y[0] is the x-position of the cart, y[1] is the velocity
                of the cart (dx/dt), y[2] is the angle of the pendulum
                (theta) from the vertical in radians, and y[3] is the
                rate of change of theta (dtheta/dt).
            m (float): Mass of pendulum.
            M (float): Mass of cart.
            L (float): Length of pendulum.
            g (float): Acceleration due to gravity.
            d (float): Damping coefficient for friction between cart and
                ground.
            u (float): Force on cart in x-direction.

        Returns:
            dx (np.array): The time derivate of the state (dx/dt) as an
                array of shape (4,).
        """
        x = state[:,0].view(-1,1)
        xdot = state[:,1].view(-1,1)
        th = state[:,2].view(-1,1)
        thdot = state[:,3].view(-1,1)

        u = action
        u = torch.clamp(u, -self.max_force, self.max_force)
        
        # Temporary variables
        sin_x = torch.sin(th)
        cos_x = torch.cos(th)
        mL = self.masspole * self.length
        D = 1 / (self.length * (self.masscart + self.masspole * (1 - cos_x**2)))
        b = mL * thdot**2 * sin_x - self.friction * xdot + u

        # Non-linear ordinary differential equations describing
        # simple cart-pendulum system dynamics
        new_xdot= xdot + self.dt * D * (-mL * self.g * cos_x * sin_x + self.length * b)
        new_x = x + self.dt * new_xdot
        new_thdot = thdot + self.dt * D * ((self.masspole + self.masscart) * self.g * sin_x - cos_x * b)
        new_th = th + self.dt * new_thdot

        state = torch.cat((new_x, new_xdot, new_th, new_thdot), dim=1)
        return state

if __name__ == "__main__":
    ENV_NAME = "CartPole-BT-v0"
    TIMESTEPS = 30  # T
    N_BATCH = 1
    LQR_ITER = 5
        
    env = gym.make(ENV_NAME)
    _ = env.reset()

    nx = env.observation_space_dim[0]
    nu = env.action_space.shape[0]

    u_init = None
    render = False
    retrain_after_iter = 50
    run_iter = env.n_steps

    goal_state = torch.tensor(env.goal_state) # nx --> this is the goal state or the reference
    goal_weights = torch.tensor((0.5, 0.01, 1, 0.1)) # nx --> these are the weights given to each state in the cost-function
    ctrl_penalty = 0.001
    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(nu)))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state # nx
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost) --> check this out more carefully

    # MPC agent
    agent_mpc = mpc.MPC(nx, nu, TIMESTEPS, u_lower=-200.0, u_upper=200.0, lqr_iter=LQR_ITER, exit_unconverged=False, 
                        eps=1e-2, n_batch=N_BATCH, backprop=False, verbose=0, grad_method=mpc.GradMethods.AUTO_DIFF)

    # run MPC
    total_reward = 0
    u_init = None

    begin = time.perf_counter()
    done = False

    while not done:
        state = env.state.copy()
        state = torch.tensor(state).view(1, -1)
        command_start = time.perf_counter()

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = agent_mpc(state, cost, EnvTrueDynamics(env), u_init)
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        s, r, done, _ = env.step_dynamic(action.detach().numpy().squeeze(1))
        total_reward += r
        #print(f"action taken: {action} cost received: {-r} time taken: {elapsed}")

        env.render()

    end_episode = time.perf_counter() - begin
    print(f"Total reward {total_reward}, time per episode {end_episode}")
