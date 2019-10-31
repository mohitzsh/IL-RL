import argparse
import os
import pickle
from tensorboardX import SummaryWriter

from envs.nine_rooms_env import NineRoomsEnv

from algos.a2c.storage import RolloutStorage
from algos.a2c.models import Policy, MLPBase

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from algos.dqn.dqn import DeepQLearner
from algos.dqn.replay_buffer import ReplayBuffer as DQNReplayBuffer

def get_args():
    arguments = argparse.ArgumentParser()
    
    arguments.add_argument('--run_id', type=str, default='main')
    # Global Training Parameters 
    arguments.add_argument('--num_episodes', type=int, default=1e5)

    # A2C Parameters


    # Global Q Learning Parameters
    arguments.add_argument('--buff_capacity', type=int, default=1e10)
    arguments.add_argument('--batch_size', type=int, default=512)
    arguments.add_argument('--eps_start', type=float, default=1)
    arguments.add_argument('--eps_end', type=float, default=0.01)
    arguments.add_argument('--eps_decay', type=int, default=200)

    # DQN Parameters
    arguments.add_argument('--target_update', type=int, default=10)
    arguments.add_argument('--update_steps', type=int, default=32)
    arguments.add_argument('--lr', type=float, default=1e-3)
    arguments.add_argument('--hidden_size', type=int, default=128)
    arguments.add_argument('--ddqn', type=int, default=0)
    arguments.add_argument('--opt', default='adam')

    # Environment
    arguments.add_argument('--k', type=int, default=1)
    arguments.add_argument('--grid_size', type=int, default=10)
    arguments.add_argument('--pct', type=float, default=0.75)
    arguments.add_argument('--pcf', type=float, default=0.5)
    arguments.add_argument('--max_steps', type=int, default=100)
    arguments.add_argument('--rnd_start', type=int, default=0)
    arguments.add_argument('--gamma', type=float, default=0.9)
    arguments.add_argument('--start_state_exclude_rooms', nargs='*', type=int, default=[])

    # Validation
    arguments.add_argument('--val_episode', type=int, default=10)
    arguments.add_argument('--val_rollouts', type=int, default=10)
    arguments.add_argument('--unseen', type=int, default=0)

    # Checkpointing
    arguments.add_argument('--save_root', type=str, default='save')

    # Others
    arguments.add_argument('--gpu', type=int, default=1)

    arguments.add_argument('--seed', type=int, default=101)

    return arguments.parse_args()

args = get_args()

def main():
    # Complete checkpointing logistics
    save_dir = os.path.join(args.save_root, args.run_id)

    ckpt_dir = os.path.join(save_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get the device to use
    device = torch.device("cuda:0" if args.gpu else "cpu")


    # Write the argument
    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)
    print("Arguments Dumped.")

    writer = SummaryWriter(save_dir)
    
    # Replace with other initial start distribution
    rnd_start = (args.rnd_start == 1)

    env = NineRoomsEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        rnd_start=rnd_start,
    )

    # Setup A2C training
    policy_base = MLPBase(
        input_size=env.observation_size,
        hidden_size=args.hidden_size
    ) 

    actor_critic = Policy(
        action_space=env.action_space,
        base=policy_base
    )

    if args.gpu:
        actor_critic.cuda()

    rollouts = RolloutStorage(args.max_steps, env.observation_shape, env.action_space) 
    _obs, _ = envs.reset()
    obs = torch.from_numy(_obs)
    rollouts.obs[0].copy_(obs)
    if args.gpu:
        rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    
    for ep in range(args.num_episodes):

        for step in range(args.max_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            '''Check if this is even required'''
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)
        

    writer.flush()

if __name__ == '__main__':
    main()
