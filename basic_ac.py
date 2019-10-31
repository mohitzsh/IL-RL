import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt
plt.switch_backend('agg')


# Logging
from tensorboardX import SummaryWriter

# Environment
from envs.empty_grid_world import EmptyGridWorld

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--rand-start', type=int, default=0)

# Actor Critic
parser.add_argument('--hidden-size', type=int, default=64)

# Grid World
parser.add_argument('--grid-size', type=int, default=10)
parser.add_argument('--max-steps', type=int, default=100)

# Optimizer
parser.add_argument('--lr', type=float, default=1e-3)

# Common
parser.add_argument('--num-episodes', type=int, default=5000)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--log-dir', type=str, default='save/logs')

args = parser.parse_args()

# Setup Environment
env = EmptyGridWorld(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        rnd_start=args.rand_start,
    )

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(args.log_dir)


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, output_size, hidden_size):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())

        # actor's layer
        self.action_head = nn.Linear(hidden_size, output_size)

        # critic's layer
        self.value_head = nn.Linear(hidden_size, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        forward of both actor and critic
        """

        actor_feat = self.actor(state)
        critic_feat = self.critic(state)

        action_prob = F.softmax(self.action_head(actor_feat), dim=-1)

        state_values = self.value_head(critic_feat)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


model = Policy(
    input_size=env.observation_size,
    hidden_size=args.hidden_size,
    output_size=env.nActions
)

if args.gpu:
   model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()

steps = 0
def select_action(state):
    global steps
    steps += 1
    state = torch.from_numpy(state).float()
    if args.gpu:
        state = state.cuda()

    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    writer.add_scalar("Entropy", m.entropy(), steps)
    writer.flush()

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        target = torch.tensor([R])
        if args.gpu:
            target = target.cuda()
        value_losses.append(F.smooth_l1_loss(value, target))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def train():
    running_reward = 10

    # run inifinitely many episodes
    for i_episode in range(args.num_episodes):

        # reset environment and episode reward
        state, _ = env.reset()
        state = state['state_feat']
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in count():

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)
            state = state['state_feat']

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        writer.add_scalar("Returns", ep_reward, i_episode)
        print("Episode: {}, Returns: {:.2f}".format(i_episode, ep_reward))
        writer.flush()

def visualize_policy():

    plt_w = 2 # Width of a single plot
    plt_h = 2 # Height of a single plot

    fig_w = plt_w * (args.grid_size - 2)
    fig_h = plt_h * (args.grid_size -2)

    fig, ax = plt.subplots(
        nrows=env.grid_size-2,
        ncols=env.grid_size-2,
        sharex=True,
        sharey=True,
        figsize=(fig_w, fig_h))

    actions = list(range(env.nActions))
    color = 'rgbc'
    for x in range(1, env.grid_size-1):
        for y in range(1, env.grid_size-1):
            state = env._encode_state((x, y))

            state = torch.Tensor(state)
            if args.gpu:
                state = state.cuda()
            with torch.no_grad():
                probs, _ = model(state) 

            probs = probs.cpu().numpy()

            ax[y-1][x-1].bar(
                x=actions,
                height=probs,
                color=color
            )

            ax[y-1][x-1].set_title("({},{})".format(x, y))

    writer.add_figure("Policy", fig)
    writer.flush()
    


if __name__ == '__main__':
    train()
    visualize_policy()
    writer.close()