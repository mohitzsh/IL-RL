import argparse
import gym
import numpy as np
from itertools import count
import os
from collections import namedtuple
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import kl 
from torch.utils.data import Dataset, DataLoader

from envs.empty_grid_world import EmptyGridWorld
from envs.nine_rooms_env import NineRoomsEnv
from envs.cluttered import Cluttered

from tensorboardX import SummaryWriter
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
plt.switch_backend('agg')

REINFORCE_MODES = ['reinforce','finetune', 'guided-rule-based', 'hybrid', 'hier']

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--id', type=str, default='main')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')

# Policy Parameterization
parser.add_argument('--hidden-size', type=int, default=128)

# Environment Arguments
parser.add_argument('--env', type=str, choices=['empty', 'cluttered'], default='empty')
parser.add_argument('--grid-size', type=int, default=10)
parser.add_argument('--nb-objects', type=int, default=5)
parser.add_argument('--obj-size', type=int, default=3)
parser.add_argument('--state-encoding', type=str, choices=['thermal', 'one-hot'], default='thermal')

# Training Arguments
parser.add_argument('--max-episodes', type=int, default=1000)
parser.add_argument('--max-steps', type=int, default=100)
parser.add_argument('--val-frequency', type=int, default=50)
# PG Optimization
parser.add_argument('--pg-lr', type=float, default=1e-3)

# Algorithmic Arguments
parser.add_argument('--imitate', type=int, default=1)
parser.add_argument('--reinforce', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--nb-epochs', type=int, default=20)
parser.add_argument('--im-lr', type=float, default=1e-3)
parser.add_argument('--sampling-zone', type=str, default="1-2-3-4")
parser.add_argument('--weigh-class', type=str, default=0)
parser.add_argument('--n-policies', type=int, default=1)
parser.add_argument('--reinforce-mode', type=str, choices=REINFORCE_MODES,
                 default='guided-rule-based')
parser.add_argument('--reinforce-norm', type=str, choices=['sum', 'mean'], default='sum')

parser.add_argument('--beta-im-ep', type=int, default=1000)
parser.add_argument('--beta-im-start', type=float, default=1)
parser.add_argument('--beta-im-end', type=float, default=1)

parser.add_argument('--beta-rl', type=float, default=1)

parser.add_argument('--gpu', type=int, default=1)

# Logging
parser.add_argument('--log-dir', type=str, default='save/logs/PG')

args = parser.parse_args()

eps = np.finfo(np.float32).eps.item()
if args.env == 'empty':
    env = EmptyGridWorld(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        state_encoding=args.state_encoding
    )
elif args.env == 'cluttered':
    env = Cluttered(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        num_objects=args.nb_objects,
        obj_size=args.obj_size,
        state_encoding=args.state_encoding
    )
else:
    raise ValueError
print("Environment Setup Done")


os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(args.log_dir)

# Draw the environment on tensorboardX
env_img = env.render(mode='rgb_array')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(env_img)
writer.add_figure("Environment", fig, 0)


# Create the identifier for thsi run
run_id = "Environment: {} \n Max Steps: {}\n Grid Size: {}".format(args.env, args.max_steps, args.grid_size)
writer.add_text("Params", run_id, 0)
writer.flush()

class DemonsDataset(Dataset):
    def __init__(self, states, actions):

        assert len(states) == len(actions), "Malformed Dataset, States and Actions have to be of the same size."

        self.states = states
        self.actions = actions
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):

        state = self.states[idx]
        action = self.actions[idx]

        state = np.array(state, dtype=np.float32)

        return state, action

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.affine3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)

        x = self.affine2(x)
        x = F.relu(x)

        action_scores = self.affine3(x)

        return action_scores
    
    def select_action(self, state, mode='sample'):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if args.gpu:
            state = state.cuda()

        with torch.no_grad():
            scores = self.forward(state)
        probs = F.softmax(scores, dim=-1)

        if mode == 'sample': 
            m = Categorical(probs=probs)
            action = m.sample()
            prob = torch.exp(m.log_prob(action))
        else:
            prob, action = torch.max(probs, dim=-1)
        
        return action.item(), prob.item()

def finish_episode_finetune(traj, policy, optim_pg):
    R = 0
    policy_loss = []
    returns = []
    states = []
    actions = []
    rewards = []

    for s, a , _ , r in traj:
        states.append(s)
        actions.append(a)
        rewards.append(r)

    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).unsqueeze(1)

    if args.gpu:
        returns = returns.cuda()

    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Get logprobs and probs for actions already sampled
    all_states = torch.from_numpy(np.vstack(states)).float()
    all_actions = torch.from_numpy(np.vstack(actions)).long()

    if args.gpu:
        all_states = all_states.cuda()
        all_actions = all_actions.cuda()
    

    all_scores = policy(all_states) 
    all_probs = F.softmax(all_scores, dim=-1)
    
    sampled_probs = torch.gather(all_probs, 1 , all_actions)

    sampled_lprobs = sampled_probs.log()

    # Get the loss
    obj = sampled_lprobs * returns

    if args.reinforce_norm == "sum":
        loss = -1 * obj.sum()
    else:
        loss = -1 * obj.mean()

    optim_pg.zero_grad()

    loss.backward()

    optim_pg.step()

    return loss.item()

def finetune(policy):

    '''Setup Optimizer for RL finetuning'''
    optim_pg = optim.Adam(policy.parameters(), lr=args.pg_lr)

    for i_episode in range(args.max_episodes):
        ep_reward = 0
        state, _ = env.reset()
        state = state['state_feat']
        traj = []
        for t in count(): 
            action, prob = policy.select_action(state)
            _state, reward, done, _ = env.step(action)
            _state = _state['state_feat']

            traj.append((state, action, prob, reward))
            state = _state
            ep_reward += reward
            if done:
                break

        loss = finish_episode_finetune(traj, policy, optim_pg)
        print('Episode {}\t Reward: {:.2f}'.format(i_episode, ep_reward))
        writer.add_scalar("Train_Returns", ep_reward, i_episode)
        writer.add_scalar("ReinforceLoss", loss, i_episode)
        writer.flush()

        if i_episode % args.val_frequency == 0:
            val_returns = validate(policy)

            writer.add_scalar("Val_Returns", val_returns, i_episode)
            writer.flush()
    
    return policy

def validate(policy):
    val_reward = 0
    state, _ = env.reset()
    state = state['state_feat']
    for t in count(): 
        with torch.no_grad():
            action, _ = policy.select_action(state, mode='greedy')
        state, reward, done, _ = env.step(action)
        state = state['state_feat']

        val_reward += reward
        if done:
            break

    return val_reward

def validate_hier(hierPolicy, policy_il):
    val_reward = 0
    state, _ = env.reset()
    state = state['state_feat']
    for t in count(): 
        with torch.no_grad():
            action, _ = hierPolicy.select_action(state, mode='greedy')
        if action == env.nActions:
            action, _ = policy_il.select_action(state, mode='greedy')
        state, reward, done, _ = env.step(action)
        state = state['state_feat']

        val_reward += reward
        if done:
            break

    return val_reward


def sample_train_data(grid, OP):
    '''
    Depending on args.sampline_zone, sample the states.
    Code for the quadrants
    "1" : +x +y
    "2" : -x +y
    "3" : -x -y
    "4" : +x -y
    center is the grid center
    '''
    zones = args.sampling_zone.split("-")

    dataset_states = []
    dataset_actions = []

    for zone in zones:
        if zone == "1":
            x_range = list(range(env.grid_size // 2, env.grid_size))
            y_range = list(range(0, env.grid_size // 2))

        elif zone == "2":
            x_range = list(range(0, env.grid_size // 2))
            y_range = list(range(0, env.grid_size // 2))

        elif zone == "3":
            x_range = list(range(0, env.grid_size // 2))
            y_range = list(range(env.grid_size // 2, env.grid_size))
        
        else:
            x_range = list(range(env.grid_size // 2, env.grid_size))
            y_range = list(range(env.grid_size // 2, env.grid_size))
        
        all_states = list(product(x_range, y_range))

        # Filter the states which can't be occupied
        filtered_states = [ state for state in all_states if grid[state[0], state[1]] == 0]

        for state in filtered_states:
            i, j = state[0], state[1]
            opt_a = OP[i][j]

            state_feat = env._encode_state((i, j))

            dataset_states.append(np.array(state_feat))
            dataset_actions.append(opt_a)
    
    return dataset_states, dataset_actions
    

def imitate():

    print("Setting up Expert.")
    grid, D, OP = env.run_dp()
    print("Expert Created.")

    '''Get the performance of the oracle'''
    oracle(OP)

    '''Sample Dataset from OP'''
    dataset_states, dataset_actions = sample_train_data(grid, OP)

    '''Setup Policy for Imitation'''
    policy_il = Policy(
        input_size=env.observation_size,
        hidden_size=args.hidden_size,
        output_size=env.nActions
    )

    policy_il.train()

    if args.gpu:
        policy_il.cuda()
    
    '''Setup Optimizer for Imitation'''
    optim_il = optim.Adam(policy_il.parameters(), lr=args.im_lr)

    '''Get action weights for class balanced loss'''
    if args.weigh_class:
        # Compute class weights
        _dataset_actions = np.array(dataset_actions)
        count_array = np.array(list(range(env.nActions)))

        for action in range(env.nActions):
            count_array[action] = (_dataset_actions == action).sum()


        weight = count_array / count_array.sum()

        weight = torch.from_numpy(weight).float()
        if args.gpu:
            weight = weight.cuda()
        
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
        loss_fn = nn.CrossEntropyLoss()

    dataset = DemonsDataset(states=dataset_states, actions=dataset_actions) 
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)

    batch_idx = 0
    for epoch in range(args.nb_epochs):

        for batch in dataloader:
            batch_idx += 1
            
            states, actions = batch[0], batch[1]

            states = states.float()
            gt_actions = actions.long()

            if args.gpu:
                states = states.cuda()
                gt_actions = gt_actions.cuda()

            action_scores = policy_il(states)

            # Compute loss
            loss = loss_fn(action_scores, gt_actions)

            optim_il.zero_grad()
            loss.backward()
            optim_il.step()
            writer.add_scalar('Imitation_Loss', loss.item(), batch_idx)
            writer.flush()

    return policy_il, OP, grid

def plot_entropy(grid):
    '''plot entropy of average policy'''

    assert args.n_policies > 1, "Need multiple policies for this"

    JSD = np.zeros((env.grid_size-2, env.grid_size-2))
    for x in range(1, env.grid_size-1):
        for y in range(1, env.grid_size-1):
            if grid[x][y] == 1:
                continue
            # Get learned Policy
            state = env._encode_state((x, y))

            state = torch.Tensor(state)
            probs_arr = []

            if args.gpu:
                state = state.cuda()
            with torch.no_grad():
                
                for policy in policies:
                    scores = policy(state) 
                    probs = F.softmax(scores, dim=-1).squeeze(0)
                    probs = probs.cpu().numpy()

                    probs_arr.append(probs)
                dist_arr = []
                for i in range(args.n_policies - 1):
                    for j in range(i+1, args.n_policies): 

                        dist = jensenshannon(probs_arr[i], probs_arr[j])

                        dist_arr.append(dist)
                
                jsd = sum(dist_arr) / len(dist_arr)
                JSD[y-1][x-1] = jsd

    import pdb; pdb.set_trace()
    fig, ax = plt.subplots(figsize=(100, 100))
    im = ax.imshow(JSD)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("JSD", rotation=-90, va="bottom")

    writer.add_figure("JSD", fig, 0)
    writer.flush()

            

def visualize_policy(policy, grid, OP):
    '''
    Visualize Optimal Policy and learned policy
    '''

    # Plot a legend separately

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt_w = 2 # Width of a single plot
    plt_h = 2 # Height of a single plot

    fig_w = plt_w * (args.grid_size - 2)
    fig_h = plt_h * (args.grid_size -2)

    fig1, ax1 = plt.subplots(
        nrows=env.grid_size-2,
        ncols=env.grid_size-2,
        sharex=True,
        sharey=True,
        figsize=(fig_w, fig_h))

    fig2, ax2 = plt.subplots(
        nrows=env.grid_size-2,
        ncols=env.grid_size-2,
        sharex=True,
        sharey=True,
        figsize=(fig_w, fig_h))

    actions = list(range(env.nActions))
    color = 'rgbc'
    for x in range(1, env.grid_size-1):
        for y in range(1, env.grid_size-1):
            if grid[x][y] == 1:
                continue
            # Get learned Policy
            state = env._encode_state((x, y))

            state = torch.Tensor(state)
            if args.gpu:
                state = state.cuda()
            with torch.no_grad():
                scores = policy(state) 

            probs = F.softmax(scores, dim=-1).squeeze(0)
            probs = probs.cpu().numpy()

            # Get optimal Policy

            probs_opt = np.zeros(env.nActions)
            probs_opt[OP[x][y]] = 1

            ax1[y-1][x-1].bar(
                x=actions,
                height=probs,
                color=color
            )

            ax2[y-1][x-1].bar(
                x=actions,
                height=probs_opt,
                color=color
            )

            #ax1[y-1][x-1].set_title("({},{})".format(x, y))


    writer.add_figure("Learned_Policy", fig1)
    writer.add_figure("Optimal_Policy", fig2)
    writer.flush()

def oracle(OP):
    '''
    This method checks the correctness of the OP computed using Dynamic Programming.
    '''
    ep_reward = 0
    state, _ = env.reset()
    state = state['agent_pos']
    for t in count(): 
        action = OP[state[0], state[1]]
        state, reward, done, _ = env.step(action)
        state = state['agent_pos']

        ep_reward += reward
        if done:
            break
    
    print("Oracle Returns: {}".format(ep_reward))

def quadrant(agent_pos):
    x_cond = (agent_pos[0] < (env.grid_size // 2))
    y_cond = (agent_pos[1] < (env.grid_size // 2))

    if x_cond:
        if y_cond:
            quadrant = 2 
        else:
            quadrant = 3
    else:
        if y_cond:
            quadrant = 1
        else:
            quadrant = 4
    
    return quadrant

def guided_rule_based_reinforce(policy_il, grid):
    '''
    -------
    Algo:
    -------
    GIVEN: 
    1. \pi_{IL} : Policy pretrained using IL
    2. \pi_{RL} : Randomly initialized policy to be returned by REINFORCE

    DO:
    Select actions from \pi_{IL} is state is a seen one, otherwise sample action from \pi_{RL}
    '''

    # Setup Policy and Optimizer
    policy_rl = Policy(
        input_size=env.observation_size,
        hidden_size=args.hidden_size,
        output_size=env.nActions
    )

    policy_rl.train()

    if args.gpu:
        policy_rl.cuda()
    
    optim_pg = optim.Adam(policy_rl.parameters(), lr=args.pg_lr)
    
    # policy_il should be in eval mode
    policy_il.eval()

    for i_episode in range(args.max_episodes):
        ep_reward = 0
        state, _ = env.reset()

        # Check if state is a seen one
        agent_pos = state['agent_pos']

        quad = quadrant(agent_pos)

        seen = quad in [int(q) for q in args.sampling_zone.split("-")]

        state = state['state_feat']

        traj = []
        for t in count(): 
            if seen:
                policy = policy_il
            else:
                policy = policy_rl 

            with torch.no_grad():
                action, prob = policy.select_action(state)

            _state, reward, done, _ = env.step(action)

            traj.append((state, action, prob, reward))

            state = _state['state_feat']

            # Check if state is seen
            agent_pos = _state['agent_pos']
            quad = quadrant(agent_pos)
            seen = quad in [int(q) for q in args.sampling_zone.split("-")]

            ep_reward += reward
            if done:
                break

        finish_episode(traj, policy_rl, optim_pg)
        print('Episode {}\t Reward: {:.2f}'.format(i_episode, ep_reward))
        writer.add_scalar("Returns", ep_reward, i_episode)
        writer.flush()

        if i_episode % args.val_frequency == 0:
            val_returns = validate(policy_rl)

            writer.add_scalar("Val_Returns", val_returns, i_episode)
            writer.flush()

def regularized_rule_based_reinforce(policy_il, grid):
    '''
    Optimize the following objective:
    Assume:
    \pi_B = (1-\alpha(s)) * \pi_theta  + \alpha(s) \pi_IL

    \pi_B : behavioral policy , \alpha(s) is indicator function telling if expert can be trusted 
    in state s.

    J(\theta) = E_{\tau \sim \pi_B{\tau}}[ r(\tau)] + \alpha(s) KL[\pi_{theta} || \pi_il] 
    '''
    # Setup Policy and Optimizer
    policy_rl = Policy(
        input_size=env.observation_size,
        hidden_size=args.hidden_size,
        output_size=env.nActions
    )

    policy_rl.train()

    if args.gpu:
        policy_rl.cuda()
    
    optim_pg = optim.Adam(policy_rl.parameters(), lr=args.pg_lr)
    
    # policy_il should be in eval mode
    policy_il.eval()

    for i_episode in range(args.max_episodes):
        ep_reward = 0
        state, _ = env.reset()

        # Check if state is a seen one
        agent_pos = state['agent_pos']

        quad = quadrant(agent_pos)

        opt = quad in [int(q) for q in args.sampling_zone.split("-")]


        state = state['state_feat']

        traj = []
        for t in count(): 
            
            if opt:
                policy = policy_il
            else:
                policy = policy_rl 

            with torch.no_grad():
                action, prob = policy.select_action(state)

            _state, reward, done, _ = env.step(action)

            traj.append((state, action, prob, reward, int(opt)))

            state = _state['state_feat']

            # Check if state is seen
            agent_pos = _state['agent_pos']
            quad = quadrant(agent_pos)
            opt = quad in [int(q) for q in args.sampling_zone.split("-")]

            ep_reward += reward
            if done:
                break

        print('Episode {}\t Reward: {:.2f}'.format(i_episode, ep_reward))
        writer.add_scalar("Train_Returns", ep_reward, i_episode)
        writer.flush()
        # ============================================================
        # Update the policy_rl according to the objective function.  
        # ============================================================

        states = []
        actions = []
        rewards = []
        optVals = []

        for state, action, _, reward, optVal in traj:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            optVals.append(optVal)

        # Get the returns
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).unsqueeze(1)

        if args.gpu:
            returns = returns.cuda()

        returns = (returns - returns.mean()) / (returns.std() + eps)

        all_states = torch.from_numpy(np.vstack(states)).float()
        all_actions = torch.from_numpy(np.vstack(actions)).long()
        optVals = torch.from_numpy(np.vstack(optVals)).float()

        if args.gpu:
            all_states = all_states.cuda()
            all_actions = all_actions.cuda()
            optVals = optVals.cuda()

        all_scores_rl = policy_rl(all_states) 
        all_probs_rl = F.softmax(all_scores_rl, dim=-1)

        all_probs_il = F.softmax(policy_il(all_states), dim=-1)
        
        sampled_lprobs_rl = torch.gather(all_probs_rl, 1 , all_actions).log()

        reinforceLoss = (
            -1 * (1 - optVals) * sampled_lprobs_rl * returns
        ).mean()

        # Setup probability distributions for KL Computation
        probRL = Categorical(probs=all_probs_rl)
        probIL = Categorical(probs=all_probs_il)

        KLD = kl.kl_divergence(probIL, probRL)

        # [NOTE] Should this be sum or mean?
        imitationLoss = (optVals * KLD).mean()

        # Get the imitationLoss Coefficient
        #beta_im = np.clip(
        #    args.beta_im_start - i_episode/args.beta_im_ep, 
        #    args.beta_im_end,
        #    args.beta_im_start
        #)

        totalLoss = args.beta_rl * reinforceLoss + args.beta_im_start * imitationLoss

        optim_pg.zero_grad()

        totalLoss.backward()

        optim_pg.step()

        lossDict = {
            "reinforce" : reinforceLoss.item(),
            "imitation" : imitationLoss.item(),
            "total" : totalLoss.item()
        }
        writer.add_scalars("Losses", lossDict, i_episode)
        writer.flush()

        writer.add_scalar("Count_Opt_States", optVals.sum().item(), i_episode)
        writer.flush()

        if i_episode % args.val_frequency == 0:
            val_returns = validate(policy_rl)

            writer.add_scalar("Val_Returns", val_returns, i_episode)
            writer.flush()

    return policy_rl

def hierarchical_reinforce(ilPolicy):
    '''
    Define a policy with action space A' = A+1 
    '''

    class HierPolicy(nn.Module):

        def __init__(
            self,
            input_size,
            hidden_size,
            output_size
        ):
            super(HierPolicy, self).__init__()

            self.affine1 = nn.Linear(input_size, hidden_size)
            self.affine2 = nn.Linear(hidden_size, hidden_size)
            self.affine3 = nn.Linear(hidden_size, output_size)
        
        def forward(self, state):
            x = F.relu(self.affine1(state))
            x = F.relu(self.affine2(x))

            x = self.affine3(x)

            return x

        def select_action(self, state, mode='sample'):
            state = torch.from_numpy(state).float().unsqueeze(0)

            if args.gpu:
                state = state.cuda()
            
            with torch.no_grad():
                action_probs = F.softmax(self.forward(state), dim=-1)
            
            if mode == 'sample':
                m = Categorical(probs=action_probs) 
                action = m.sample()
                prob = torch.exp(m.log_prob(action))
            elif mode == 'greedy':
                prob, action = torch.max(action_probs, dim=-1)
        
            return action.item(), prob.item()
        
    hierPolicy =  HierPolicy(
        input_size=env.observation_size,
        output_size=env.nActions+1,
        hidden_size=args.hidden_size
    )

    hierPolicy.train()
    if args.gpu:
        hierPolicy.cuda()

    optim_pg = optim.Adam(hierPolicy.parameters(), lr=args.pg_lr)

    for i_episode in range(args.max_episodes):
        ep_reward = 0
        state, _ = env.reset()
        state = state['state_feat']
        traj = []
        for t in count(): 
            action, prob = hierPolicy.select_action(state)
            env_action = action

            if action == env.nActions:
                env_action, _ = ilPolicy.select_action(state, mode='greedy') 

            _state, reward, done, _ = env.step(env_action)
            _state = _state['state_feat']

            traj.append((state, action, prob, reward))
            state = _state
            ep_reward += reward
            if done:
                break

        loss = finish_episode_finetune(traj, hierPolicy, optim_pg)
        print('Episode {}\t Reward: {:.2f}'.format(i_episode, ep_reward))
        writer.add_scalar("Train_Returns", ep_reward, i_episode)
        writer.add_scalar("ReinforceLoss", loss, i_episode)
        writer.flush()

        if i_episode % args.val_frequency == 0:
            val_returns = validate_hier(hierPolicy, ilPolicy)

            writer.add_scalar("Val_Returns", val_returns, i_episode)
            writer.flush()
    
    return hierPolicy
    

def main():
    if args.imitate:
        policy_il, OP, grid = imitate()
        visualize_policy(policy_il, grid, OP)
    if args.reinforce:
        
        if args.reinforce_mode == "reinforce":
            policy_rl = Policy(
                input_size=env.observation_size,
                hidden_size=args.hidden_size,
                output_size=env.nActions
            )
            policy_rl.train()
            if args.gpu:
                policy_rl.cuda()

            policy_rl = finetune(policy_rl)
        elif args.reinforce_mode == "finetune":
            policy_rl = finetune(policy_il)
        elif args.reinforce_mode == "guided-rule-based":
            policy_rl = guided_rule_based_reinforce(policy_il, grid)
        elif args.reinforce_mode == "hybrid":
            policy_rl = regularized_rule_based_reinforce(policy_il, grid) 
        elif args.reinforce_mode == "hier":
            policy_rl = hierarchical_reinforce(policy_il)
        else:
            raise ValueError

if __name__ == '__main__':
    main()