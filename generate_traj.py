import argparse
import pickle
import os
from itertools import count
import random

import torch

from algos.dqn.mlp import MLP 
from envs.nine_rooms_env import NineRoomsEnv

def get_args():

    arguments = argparse.ArgumentParser()

    arguments.add_argument('--id', type=str, required=True)

    arguments.add_argument('--base_dir', type=str, required=True,
        help="Directory where relevant files will be located. Need two files : 'args.pkl', 'cell_cat_map.pkl' and a foler:  'ckpt'")
    
    arguments.add_argument('--ckpt', type=str, required=True,
        help="File with state dict for the policy to gather demonstrations with.")
    
    arguments.add_argument('--n_demos', type=int, default=100,
        help='Number of demonstrations needed from the model.')
    
    args = arguments.parse_args()

    return args

args = get_args()

def greedy_policy(state, policy_net):

        sample = random.random()
        state_idx = int(state.sum().item())

        # Add Dummy batch dimension
        _state = state.unsqueeze(0)

        with torch.no_grad():
            q_vals = policy_net(_state).detach()

        _idx = q_vals[0].argmax().item()

        k = _idx // env.action_shape[1]
        d = _idx % env.action_shape[1]

        action = (k, d)
            
        return action

def unroll(env, policy_net):


    trajs = []
    for demo in range(args.n_demos):
        # traj is a sequence of (state, action) tuples.
        traj = []

        # NOTE: val argument is redundant
        obs, _ = env.reset(val=True, seen=True)
        state = obs['state_feat']

        for t in count():
            
            _state = torch.from_numpy(state).float().cuda()
            action = greedy_policy(_state, policy_net)
            _action = action[0] * env.action_shape[1] + action[1]
            traj.append((state, _action))

            obs, _, done, _ = env.step(action)
            next_state = obs['state_feat']

            state = next_state

            if done:
                break
        
        trajs.append(traj)

    return trajs

if __name__ == '__main__':

    # Load arguments
    f_args = os.path.join(args.base_dir, 'args.pkl')
    with open(f_args, 'rb') as f:
        loaded_args = pickle.load(f)
    
    # Load cell category map
    f_cell_cat_map = os.path.join(args.base_dir, 'cell_cat_map.pkl')
    with open(f_cell_cat_map, 'rb') as f:
        cell_cat_map = pickle.load(f)['cell_cat_map']
    
    # Load Policy State dict
    f_policy_ckpt = os.path.join(args.base_dir, 'ckpts', args.ckpt)
    with open(f_policy_ckpt, 'rb') as f:
        policy_state_dict = pickle.load(f)

    env = NineRoomsEnv(
        grid_size=loaded_args['grid_size'],
        K=loaded_args['k'],
        pct=loaded_args['pct'],
        pcf=loaded_args['pcf'],
        max_steps=loaded_args['max_steps'],
        rnd_start=loaded_args['rnd_start'],
        start_state_exclude_rooms=loaded_args['start_state_exclude_rooms'],
        cell_cat_map=cell_cat_map
    )
    state_size = env.height * env.width
    action_size = env.K * env.nActions
    policy_net = MLP(input_size=state_size,
                    output_size=action_size,
                    hidden_size=loaded_args['hidden_size'])

    policy_net.load_state_dict(policy_state_dict)

    policy_net = policy_net.cuda()
    
    policy_net.eval()

    trajs = unroll(env, policy_net)

    # Save the trajectory
    dir_trajs = os.path.join(args.base_dir, 'demos', args.ckpt.split('.')[0])
    os.makedirs(dir_trajs, exist_ok=True)
    f_trajs = os.path.join(dir_trajs, 'trajs.pkl')
    with open(f_trajs, 'wb') as f:
        pickle.dump(trajs, f)
    
    print("{} Demonstrations Dumber with model : {}".format(len(trajs), args.ckpt))



