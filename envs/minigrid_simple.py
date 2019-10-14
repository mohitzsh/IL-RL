import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple, List, NewType

from gym_minigrid.minigrid import MiniGridEnv, Ball, OBJECT_TO_IDX, CELL_PIXELS
from gym_minigrid.register import register
from gym import error, spaces, utils
from collections import namedtuple

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

XYTuple = namedtuple("XYTuple", ["x", "y"])

# MINIMAL_OBJECT_TO_IDX = {
#     'empty'         : 0,
#     'wall'          : 1,
#     'ball'          : 2,
#     'agent'         : 3,
# }
#
# MINIMAL_IDX_TO_OBJECT = dict(
#     zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
#
# MINIMAL_ENCODE = lambda idx: MINIMAL_OBJECT_TO_IDX[IDX_TO_OBJECT[idx]]
# MINIMAL_DECODE = lambda idx: OBJECT_TO_IDX[MINIMAL_IDX_TO_OBJECT[idx]]

Observation = NewType('Observation', Dict[str, np.ndarray])

class NavObject(Ball):
    '''An navigable cell (TODO: Add attributes)'''
    def __init__(self,
                 color: str = 'blue'):
        super().__init__(color=color)

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False


class MiniGridSimple(MiniGridEnv):
    """
    Point navigation wrapper for MiniGridEnv with fully observable
    state space and cardinal actions
    """

    class CardinalActions(IntEnum):
        # Cardinal movement
        right = 0
        down = 1
        left = 2
        up = 3

    def __init__(
        self,
        grid_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        seed: int = 1337,
    ):
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Override MiniGridEnv actions
        self.actions = MiniGridSimple.CardinalActions

        self.move_actions = [
            self.actions.right,
            self.actions.down,
            self.actions.left,
            self.actions.up,
        ]

        self.action_space = spaces.Discrete(len(self.actions))

        # self.encoding_range = len(MINIMAL_OBJECT_TO_IDX.keys())
        self.encoding_range = len(OBJECT_TO_IDX.keys())

        self.agent_pos_observation = spaces.Tuple([
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ])

        self.observation_space = spaces.Dict({
            'agent_pos': self.agent_pos_observation,
        })

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        # Starting position and direction for the agent
        self.start_pos = None
        self.start_dir = None

        self._done = False

        # Initialize the RNG
        self.seed(seed=seed)

        # Rendering
        self.render_rgb = True 
        self.CELL_PIXELS = CELL_PIXELS
        self.render_shape = (self.width * self.CELL_PIXELS,
                             self.height * self.CELL_PIXELS, 3)

        # Initialize the state
        self.reset()

    def seed(self, seed):
        self._seed = seed
        return super().seed(seed)

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, value):
        # print("Setting value: {}".format(value))
        self._done = value

    @property
    def reset_prob(self):
        return self._reset_prob

    @reset_prob.setter
    def reset_prob(self, value):
        assert value <= 1.0 and value >= 0.0
        self._reset_prob = value

    def modify_attr(self, attr, value):
        assert hasattr(self, attr)
        setattr(self, attr, value)

    def around_pos(self, dir: int) -> Tuple[int]:
        """
        Get the absolutie position of one of the 4 cardinal
        cells around agent as specified by dir
        """
        assert dir >= 0 and dir < 4
        pos = self.agent_pos + DIR_TO_VEC[dir]
        pos[0] = pos[0].clip(0, self.width - 1)
        pos[1] = pos[1].clip(0, self.height - 1)

        # Convert to tuple because some indexings break if pos is np.ndarray
        return tuple(pos)

    def reset(self, val=False, seen=True):

        self._gen_grid(self.width, self.height, val=val, seen=seen)
        # These fields should be defined by _gen_grid
        assert self.start_pos is not None
        # assert self.start_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.start_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Place the agent in the starting position and direction
        self.agent_pos = self.start_pos

        self.start_dir = 0
        self.agent_dir = self.start_dir
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        self.done = False
        self.success = False

        info = {
            'done': self.done,
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
        }

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        # Return first observation
        obs = self.gen_obs()
        return obs, info

    def encode_grid(self):
        orig_image = self.grid.encode(vis_mask=None)
        final_image = orig_image
        # final_image = np.copy(orig_image)

        # for i in range(self.grid.width):
        #     for j in range(self.grid.height):
        #         final_image[i, j, 0] = \
        #             MINIMAL_ENCODE(orig_image[i, j, 0])
        #
        # final_image[self.agent_pos[0], self.agent_pos[1], 0] = \
        #     MINIMAL_OBJECT_TO_IDX['agent']
        final_image = final_image[:, :, 0]

        one_hot_img = np.eye(self.encoding_range)[final_image]
        return one_hot_img.astype(np.float32)

    def is_goal(self, pos: np.ndarray) -> int:
        '''pos == mission'''

        return pos == tuple(self.mission)

    def is_not_goal(self, pos: np.ndarray) -> bool:

        raise NotImplementedError
        #'''pos != mission and a NavObject is at pos'''
        #assert type(self.mission) == np.ndarray
        #if not self.is_goal(pos) and isinstance(
        #    self.grid.get(pos[0], pos[1]), NavObject):
        #    return True
        #return False

    def reward(self, action: np.ndarray) -> float:

        return 0.0

    def gen_obs(self) -> Observation:
        """
        Generate the agent's view (fully observable grid)
        """

        agent_pos = np.array(self.agent_pos)

        assert hasattr(self, 'mission'), \
            "environments must define a mission"

        # Observations are dictionaries containing:
        # - an image (fully observable view of the environment)
        # - a textual mission string (instructions for the agent)
        obs = {
            'agent_pos': agent_pos,
            'mission': self.mission,
        }

        return obs

    def step(self, action: np.ndarray) \
        -> Tuple[Observation, np.ndarray, bool, Dict]:

        self.step_count += 1

        reward = 0

        if not self.done:
            # Cardinal movement
            if action in self.move_actions:
                move_pos = self.around_pos(action)
                fwd_cell = self.grid.get(*move_pos)
                self.agent_dir = (action - 1) % 4

                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = move_pos

                if fwd_cell != None and self.is_goal(move_pos):
                    self.agent_pos = move_pos
                    self.success = True
                    self.done = True

            else:
                raise ValueError("{}".format(action))
            
            reward += self.reward(action)


        if self.step_count >= self.max_steps - 1:
            # print("Max Steps Exceeded.")
            self.done = True

        obs = self.gen_obs()

        info = {
            'done': self.done,
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
        }

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        if self.done:
            info.update({
                'image': self.encode_grid(),
                'success': self.success,
                'agent_pos': self.agent_pos,
            })

        return obs, reward, False, info

    def perturb_agent_pos(self):
        # Cardinal movement
        perturb_action = self.np_random.choice(self.move_actions)
        move_pos = self.around_pos(perturb_action)
        fwd_cell = self.grid.get(*move_pos)
        # self.agent_dir = perturb_action - 1
        if fwd_cell == None or fwd_cell.can_overlap():
            self.agent_pos = move_pos

        if fwd_cell != None and fwd_cell.type == 'lava':
            self.done = True

    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None:
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if r.window:
            r.window.setText(self.mission)

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        r.push()
        r.translate(
            CELL_PIXELS * (self.agent_pos[0] + 0.5),
            CELL_PIXELS * (self.agent_pos[1] + 0.5)
        )
        r.rotate(self.agent_dir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 0),
            (0, 12),
            (12,  0),
            (0, -12),
        ])
        r.pop()
        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r

    def enumerate_states(self):
        old_agent_pos = self.agent_pos
        all_obs = {}
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.grid.get(i, j) != None:
                    continue
                _key = XYTuple(x = i, y = j)
                self.agent_pos = np.array([i, j])
                all_obs[_key] = self.gen_obs()
        self.agent_pos = old_agent_pos
        return all_obs

    def place_agent(self, rng=None):
        if not hasattr(self, '_occupancy_grid'):
            # Occupancy grid, has 1s where agent can't be placed
            self._occupancy_grid = np.ones((self.width, self.height))
            for row in range(self.width):
                for col in range(self.height):
                    cell = self.grid.get(row, col)
                    # assert start_cell is None or start_cell.can_overlap()
                    if cell is None or cell.can_overlap():
                        self._occupancy_grid[row, col] = 0
            self._unoccupied_x, self._unoccupied_y = \
                np.where(self._occupancy_grid == 0)

            assert len(self._unoccupied_x) > 0

        if rng == None:
            rng = self.np_random

        start_index = rng.randint(len(self._unoccupied_x))
        self.start_pos = (
            self._unoccupied_x[start_index],
            self._unoccupied_y[start_index],
        )
        self.start_dir = 0