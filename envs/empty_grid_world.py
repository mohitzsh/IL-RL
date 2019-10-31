import numpy as np
from enum import IntEnum
from itertools import product
from gym import spaces

import gym_minigrid
from gym_minigrid.minigrid import Goal, Grid

import sys
# Do a hacky thing for now
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from minigrid_simple import MiniGridSimple

class EmptyGridWorld(MiniGridSimple):

    # Only 4 actions needed, left, right, up and down

    class CardnalActions(IntEnum):
        # Cardinal movement
        right = 0
        down = 1
        left = 2
        up = 3

        def __len__(self):
            return 4

    def __init__(
        self,
        grid_size=20,
        max_steps=100,
        state_encoding="thermal",
        seed=133,
        rnd_start=0,
    ):

        self.state_encoding = state_encoding 
        self.grid_size = grid_size
        
        self._goal_default_pos = (self.grid_size-2, 1)

        # set to 1 if agent is to be randomly spawned
        self.rnd_start = rnd_start

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            seed=seed,
            see_through_walls=False
        )

        self.nActions = len(EmptyGridWorld.CardnalActions)

        # Set the action and observation spaces
        self.actions = EmptyGridWorld.CardnalActions

        self.action_space = spaces.Discrete(self.nActions)

        self.max_cells = (grid_size - 1) * (grid_size -1)

        self.observation_space = spaces.Tuple([
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ])

        self.observation_size = self.grid_size * self.grid_size
        self.observation_shape = (self.observation_size, )

        self.T = max_steps


        # Change the observation space to return the position in the grid

    @property
    def category(self):
        # [TODO] Make sure this doesn't break after self.agent_pos is changed to numpy.ndarray
        return self.cell_cat_map[self.agent_pos] 

    def reward(self):
        # -1 for every action except if the action leads to the goal state
        return 1 if self.success else  -1 / self.T 

    def _gen_grid(self, width, height, val=False, seen=True):

        # Create the grid
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Even during validation, start state distribution
        # should be the same as that during training
        if not self.rnd_start:
            self._agent_default_pos = (1, self.grid_size-2)
        else:
            self._agent_default_pos = None

        # Place the agent at the center
        if self._agent_default_pos is not None:
            self.start_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.start_dir = self._rand_int(0, 4)  # Agent direction doesn't matter

        goal = Goal()
        self.grid.set(*self._goal_default_pos, goal)

        goal.init_pos = goal.curr_pos = self._goal_default_pos

        self.mission = goal.init_pos

    def reset(self, val=False, seen=True):

        obs, info = super().reset(val=val, seen=seen) 

        # add state feature to obs
        state_feat = self._encode_state(obs['agent_pos'])

        obs.update(dict(state_feat=state_feat))

        return obs, info

    def step(self, action):
        
        self.step_count += 1

        '''
         Reward doesn't depend on action, but just state.
         reward = -1 if not (in_goal_state) else 0
        '''

        if not self.done:
            # check if currently at the goal state
            if self.agent_pos == self.mission:
                # No penalty, episode done
                self.done = True
                self.success = True
            else:
                # Cardinal movement
                if action in self.move_actions:
                    move_pos = self.around_pos(action)
                    fwd_cell = self.grid.get(*move_pos)

                    self.agent_dir = (action - 1) % 4

                    if fwd_cell == None or fwd_cell.can_overlap() or self.is_goal(move_pos):
                        self.agent_pos = move_pos
                else:
                    raise ValueError("Invalid Action: {} ".format(action))

        reward = self.reward()
        if self.step_count >= self.max_steps - 1:
            # print("Max Steps Exceeded.")
            self.done = True

        obs = self.gen_obs()

        # Add state features to the observation
        state_feat = self._encode_state(obs['agent_pos'])

        obs.update(dict(state_feat=state_feat))

        info = {
            'done': self.done,
            'agent_pos': np.array(self.agent_pos),
        }

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        if self.done:
            info.update({
                'image': self.encode_grid(),
                'success': self.success,
                'agent_pos': self.agent_pos,
            })

        return obs, reward, self.done , info
    
    def _encode_state(self, state):
        """
        Encode the state to generate observation.
        """

        feat = np.ones(self.width * self.height, dtype=float)
        curr_x, curr_y = state[0], state[1] 

        curr_pos = curr_y * self.width + curr_x

        if self.state_encoding == "thermal":

            feat[curr_pos:] = 0
        elif self.state_encoding == "one-hot":
            feat[:] = 0 
            feat[curr_pos] = 1

        return feat 