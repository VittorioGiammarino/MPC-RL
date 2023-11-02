# MPC-RL

## Instructions

### Use anaconda to create a virtual environment

**Step 1.** Install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** Clone repo and create conda environment

```shell
conda env create -f environment.yml
conda activate cartpole
```

### Train MPC-RL agent

**RL w/ differentiable MPC (NN actor)**
```shell
python train_RL_diff_MPC.py agent=ddpg_mpc
```
**RL w/ differentiable MPC (actor made of paramaters only)**
```shell
python train_RL_diff_MPC.py agent=ddpg_mpc_alternative
```
**RL w/ MPC as part of the environmnent**
```shell
python train_RL_env_MPC.py agent=sac
```

```shell
python train_RL_env_MPC.py agent=ddpg
```

Tensorboard is deactivated by default. It can be activated running 

```shell
python train_RL_diff_MPC.py agent=ddpg_mpc use_tb=true
```

```shell
python train_RL_env_MPC.py agent=sac use_tb=true
```

```shell
python train_RL_env_MPC.py agent=ddpg use_tb=true
```

To access tensorboard run in another shell
```shell
tensorboard --logdir experiments
```