import pprint
import math
import random
import copy
import os
import torch
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import functools
import operator
from itertools import count
import numpy as np
import pickle
from algos.dqn.mlp import MLP

class DeepQLearner:

    def __init__(self,
            env,
            memory,
            ddqn=False,
            eps_start=1,
            eps_end=0.01,
            eps_decay=100,
            gamma=0.9,
            hidden_size=128,
            lr=1e-3,
            opt='adam',
            cuda=True,
    ):
        """ initialize policy and target networks, optimizer
        """


        self.env = env

        self.memory = memory
        # Setup epsilon greedy strategy 
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.gamma = gamma 

        self.state_size = self.env.height * self.env.width

        self.action_size = self.env.K * self.env.nActions 

        self.action_shape = self.env.action_shape
        self.state_shape = self.env.state_shape

        self.hidden_size = hidden_size

        self.lr = lr
        self.cuda = cuda

        self.ddqn = ddqn

        self.policy_net = MLP(input_size=self.state_size,
                        output_size=self.action_size,
                        hidden_size=self.hidden_size)
        # initialization
        #for params in self.policy_net.parameters():
        #    init.normal_(params, mean=0, std=1e-2)

        self.target_net = copy.deepcopy(self.policy_net)

        self.target_net.eval()
        self.policy_net.eval()

        if opt == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr) 
        elif opt == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), momentum=0.95, lr=self.lr)
        else:
            raise ValueError

        # move to GPU
        if self.cuda:
            self.cuda = 1
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()

        self.steps_done = 0


    @property 
    def k(self):
        self._k = np.random.randint(self.nheads)

        return self._k

    def unroll(self, update_steps=None, batch_size=None, val=False, seen=True):
        mode = 'greedy' if val else 'eps-greedy'

        obs, _ = self.env.reset(val=val, seen=seen)
        state = obs['state_feat']
        returns = 0
        returns_dis = 0
        dis = 1

        for t in count():
           
            _state = torch.from_numpy(state).float().cuda()
            action = self.act(_state, mode=mode, val=val)

            obs, reward, done, _ = self.env.step(action)
            next_state = obs['state_feat']

            # Returns are computed for plotting

            returns += reward
            returns_dis += dis * reward
            dis *= self.gamma

            # Flatten the action before using it in the experience
            _action = action[0] * self.action_shape[1] + action[1]

            self.memory.push(state, _action, reward, next_state, int(done))

            if (not val) and (self.steps_done % update_steps == 0) and len(self.memory) > batch_size:
                experience = self.memory.sample(batch_size)
                loss = self.update(experience)

                #print("Step: {} Loss: {}".format(self.steps_done, loss))
            state = next_state

            if done:
                return returns, returns_dis 
    
    def _validate(
        self,
        val_rollouts,
        seen=True
    ):
        v_returns = []
        v_dis_returns = []
        for v_e in range(val_rollouts):
            _v_returns, _val_dis_returns = self.unroll(val=True, seen=seen)

            v_returns.append(_v_returns)
            v_dis_returns.append(_val_dis_returns)
        
        v_returns, v_dis_returns = np.array(v_returns), np.array(v_dis_returns)

        return v_returns, v_dis_returns


    def train(
        self,
        num_episodes,
        batch_size,
        target_update,
        update_steps,
        val_rollouts,
        val_episode,
        writer=None,
        ckpt_dir=None,
        unseen=False
    ):

        for e in range(num_episodes):
            
            # Get a rollout
            returns, returns_dis = self.unroll(
                update_steps=update_steps,
                batch_size=batch_size,
                val=False)
            writer.add_scalar("Train-Returns", returns, e)
            writer.add_scalar("Train-Discounted-Returns", returns_dis, e)

            if (e % target_update) == 0:
                self.update_target_net()

            # VALIDATE
            if e % val_episode == 0:

                # Spawn where agent has been during training.
                v_sreturns, v_sdis_returns = self._validate(val_rollouts=val_rollouts, seen=True)

                ret_dict = {
                    'Seen' : np.mean(v_sreturns)
                }

                dis_ret_dict = {
                    'Seen' : np.mean(v_sdis_returns)
                }

                if unseen:
                    # Spawn where agent hasn't been spawned during training.
                    v_usreturns, v_usdis_returns = self._validate(val_rollouts=val_rollouts, seen=False)

                    ret_dict.update({
                        'Unseen' : np.mean(v_usreturns)
                    })

                    dis_ret_dict.update({
                        'Unseen' : np.mean(v_usdis_returns)
                    })

                writer.add_scalars("Val-Returns", ret_dict, e)
                writer.add_scalars("Val-Discounted-Returns", dis_ret_dict, e)

                print("E: {}, SR : {:.3f}, SDR: {:.3f} ".format(e, np.mean(v_sreturns), np.mean(v_sdis_returns)))
                writer.flush()

                # CHECKPOINTING
                if ckpt_dir is not None:
                    fname = os.path.join(ckpt_dir, 'e-{}.pkl'.format(e))
                    policy_net_state_dict, _ = self.get_state_dicts()

                    with open(fname, 'wb') as f:
                        pickle.dump(policy_net_state_dict, f)
                    print("CKPT: {}".format(fname))
            
    
    def update(self, experiences):
        """
        Given the experience sampled from the Replay Buffer, update the Q Value function,
        """
        # Process the experience
        _obs, _actions, _rewards, _next_obs, _dones = experiences

        _obs, _actions, _rewards, _next_obs, _dones = torch.from_numpy(_obs).float(), \
                                                        torch.from_numpy(_actions), \
                                                        torch.Tensor(_rewards), \
                                                        torch.from_numpy(_next_obs).float(), \
                                                        torch.Tensor(_dones).bool()

        experiences = (_obs, _actions, _rewards, _next_obs, _dones)

        # set policy network to train
        self.policy_net.train()
        # unpack s,a,r,s',g
        state, action, reward, next_state, done = experiences

        if self.cuda:
            state, action, next_state = state.cuda(), action.cuda(), next_state.cuda()

        batch_size = state.shape[0]
        state_action_value = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze()
        next_state_value = torch.Tensor(torch.zeros(batch_size)).cuda()

        if self.ddqn:
            _opt_actions = self.policy_net(next_state).detach().max(dim=1)[1] 
            _next_state_value = self.target_net(next_state).detach().gather(1, _opt_actions.unsqueeze(1))
            next_state_value[~done] = _next_state_value[~done].squeeze()
        else:
            next_state_value[~done] = self.target_net(next_state[~done]).detach().max(dim=1)[0]

        expected_value = self.gamma * next_state_value + reward.cuda()
        loss = F.smooth_l1_loss(state_action_value, expected_value)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.clamp_(-1, 1)
        self.optimizer.step()

        # reset policy network to eval
        self.policy_net.eval()

        return loss.item()


    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_state_dicts(self):
        return self.policy_net.state_dict(), self.target_net.state_dict()

    def load(self, policy_state_dict, target_state_dict):

        self.policy_net.load_state_dict(policy_state_dict)
        self.target_net.load_state_dict(target_state_dict)

    def act(self, state, mode='eps-greedy', val=False):

        assert mode in ['eps-greedy', 'greedy'], \
             "Only epsilon greedy and greedy action selection supported"

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        sample = random.random()

        state_idx = int(state.sum().item())
        if mode == 'greedy' or (mode == 'eps-greedy' and sample > eps_threshold):
            # Add Dummy batch dimension
            _state = state.unsqueeze(0)
            # [TODO] Add action pruning
            with torch.no_grad():
                q_vals = self.policy_net(_state).detach()

            _idx = q_vals[0].argmax().item()

            k = _idx // self.action_shape[1]
            d = _idx % self.action_shape[1]

        else:
            action_idx = np.random.choice(self.action_size) 
            k = action_idx // self.action_shape[1]
            d = action_idx % self.action_shape[1]

        action = (k, d)

        if not val:
            self.steps_done += 1
            
        return action