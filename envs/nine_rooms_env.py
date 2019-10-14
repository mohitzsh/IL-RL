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

class NineRoomsEnv(MiniGridSimple):

    # Only 4 actions needed, left, right, up and down

    class NineRoomsCardinalActions(IntEnum):
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
        passage_size=1,
        K=10,
        pct=0.75,
        pcf=0.5,
        max_steps=100,
        seed=133,
        rnd_start=0,
        start_state_exclude_rooms=[],
        cell_cat_map=None
    ):
        
        self.grid_size = grid_size
        self.passage_size = passage_size
        
        self._goal_default_pos = (1, 1)

        self.pct = pct
        self.pcf = pcf

        # set to 1 if agent is to be randomly spawned
        self.rnd_start = rnd_start

        # If self.rnd_start =1, don't spawn in these rooms
        self.start_state_exclude_rooms = start_state_exclude_rooms

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            seed=seed,
            see_through_walls=False
        )


        # create cell to [K] mapping
        self.K = K
        if cell_cat_map is None:
            self.cell_cat_map = np.random.randint(low=self.K,size=(self.width, self.height))
        else:
            self.cell_cat_map = cell_cat_map

        self.nActions = len(NineRoomsEnv.NineRoomsCardinalActions)

        # Set the action and observation spaces
        self.actions = NineRoomsEnv.NineRoomsCardinalActions
        self.action_space = spaces.Tuple([
            spaces.Discrete(self.K),
            spaces.Discrete(self.nActions)
        ])
        self.max_cells = (grid_size - 1) * (grid_size -1)

        self.tabular_observation_space = spaces.Tuple([
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ])

        self.state_shape = (
            self.tabular_observation_space.spaces[0].n,
            self.tabular_observation_space.spaces[1].n
        )

        self.action_shape = (
            self.action_space.spaces[0].n,
            self.action_space.spaces[1].n
        )

        self.observation_space.spaces.update({"table" : self.tabular_observation_space})

        self.T = max_steps


        # Change the observation space to return the position in the grid

    @property
    def category(self):
        # [TODO] Make sure this doesn't break after self.agent_pos is changed to numpy.ndarray
        return self.cell_cat_map[self.agent_pos] 

    def reward(self):
        # -1 for every action except if the action leads to the goal state
        return 1 if self.success else 0 

    def _gen_grid(self, width, height, val=False, seen=True):

        # Create the grid
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place horizontal walls through the grid
        self.grid.horz_wall(0, height //3)
        self.grid.horz_wall(0, (2*height) // 3)

        # Place vertical walls through the grid
        self.grid.vert_wall(width//3, 0)
        self.grid.vert_wall((2*width) //3, 0)

        # Create passages
        passage_anchors = [
            ( width // 3, height//3),
            (width // 3, (2*height) // 3),
            ((2*width)// 3, height // 3),
            ((2*width)//3, (2*height)//3)
        ]
        passage_cells = []
        for anchor in passage_anchors:
            for delta in range(-1*self.passage_size, self.passage_size+1):
                passage_cells.append((anchor[0] + delta, anchor[1]))
                passage_cells.append((anchor[0], anchor[1] + delta))
        
        for cell in passage_cells:
            self.grid.set(*cell, None)

        # Even during validation, start state distribution
        # should be the same as that during training
        if not self.rnd_start:
            self._agent_default_pos = ((width - 2) // 2, (height - 2) // 2)
        else:
            self._agent_default_pos = None

        # Place the agent at the center
        if self._agent_default_pos is not None:
            self.start_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.start_dir = self._rand_int(0, 4)  # Agent direction doesn't matter
        else:

            if len(self.start_state_exclude_rooms) == 0:
                self.place_agent()
            else:
                valid_start_pos = []
                if seen:
                    exclude_from = self.start_state_exclude_rooms
                else:
                    exclude_from = [x for x in range(1, 10) if x not in self.start_state_exclude_rooms]
                for room in range(1, 10):
                    if room in exclude_from:
                        continue
                    # Ignore that there are walls for now, can handle that with rejection sampling

                    # Get x coordinates of allowed cells
                    valid_x = []
                    if room % 3 == 1:
                        valid_x = list(range(1, width//3))
                    elif room % 3 == 2:
                        valid_x = list(range(width//3 +1, (2*width) // 3))
                    else:
                        valid_x = list(range((2*width) // 3 +1 , width-1))
                    
                    # Get valid y-coordinates of allowed cells
                    valid_y = []
                    if (room -1) // 3 == 0:
                        valid_y = list(range(1, height//3))
                    elif (room - 1) // 3 == 1:
                        valid_y = list(range(height//3 + 1, (2*height) // 3))
                    else:
                        valid_y = list(range((2*height) // 3 +1 , height-1))
                    
                    room_cells = list(product(valid_x, valid_y))

                    valid_start_pos += room_cells

                # Make sure start position doesn't conflict with other cells
                while True:

                    _start_pos = valid_start_pos[np.random.choice(len(valid_start_pos))]
                    row = _start_pos[1]
                    col = _start_pos[0]
                    cell = self.grid.get(row, col)

                    if cell is None or cell.can_overlap():
                        break
                
                self.start_pos = (col, row)
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
        
        cat = action[0]
        action = action[1]

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
                    if cat == self.category:
                        # With prob pct, move to the correct cell
                        if np.random.uniform() <= self.pct:
                            move_pos = self.around_pos(action)
                        else:
                            move_pos = self.around_pos(np.random.randint(self.nActions))
                    else:
                        # With prob pcf, move to the correct cell
                        if np.random.uniform() <= self.pcf:
                            move_pos = self.around_pos(action)
                        else:
                            move_pos = self.around_pos(np.random.randint(self.nActions))

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
        curr_x, curr_y = state[1], state[0] 

        curr_pos = curr_y * self.width + curr_x

        feat[curr_pos:] = 0

        return feat 
