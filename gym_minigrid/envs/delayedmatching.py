from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DelayedMatchingEnv(MiniGridEnv):
    """
    This environment is a memory test similar to the memory environment. At the start of an episode, an item is shown
    for a brief time. After a random number of delay frames, two test items appear.
    The agent has to remember the initial object, and select the location containing the matching object.
    """

    def __init__(
        self,
        seed,
        size=8,
        random_length=False,
    ):
        self.random_length = random_length
        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=False,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Fix the player's start position and orientation
        self.agent_pos = (self._rand_int(1, hallway_end + 1), height // 2)
        self.agent_dir = 0


        # Create the 3 objects
        type_objets = {Circle, Triangle, Square}

        cue_type = self._rand_elem(type_objets)
        cue_obj = cue_type(scale=self._rand_float(0.0, 0.6), color=self._rand_color())
        type_objets.discard(cue_type) # remove cue object type from type left

        second_type = self._rand_elem(type_objets)

        matching_obj = cue_type(scale=self._rand_float(0.0, 0.6), color=self._rand_color())  # test object matching cue
        second_obj = second_type(scale=self._rand_float(0.0, 0.6), color=self._rand_color()) # random second object

        # Place cue object
        cue_pos = self.place_obj(cue_obj)

        #TODO Delay

        # Remove cue obj
        self.grid.set(*cue_pos, None)

        # Place test objects
        matching_pos = self.place_obj(matching_obj)
        second_pos = self.place_obj(second_obj)

        self.success_pos = (matching_pos[0], matching_pos[1])
        self.failure_pos = (second_pos[0], second_pos[1])


        # self.grid.set(1, height // 2 - 1, start_room_obj(scale=self._rand_float(0.0, 0.4), color=self._rand_color()))
        # other_objs = self._rand_elem([[Triangle, Square], [Square, Triangle]])
        # pos0 = (hallway_end + 1, height // 2 - 2)
        # pos1 = (hallway_end + 1, height // 2 + 2)
        # self.grid.set(*pos0, other_objs[0](self._rand_color()))
        # self.grid.set(*pos1, other_objs[1](self._rand_color()))
        #
        # # Choose the target objects
        # if start_room_obj == other_objs[0]:
        #     self.success_pos = (pos0[0], pos0[1])
        #     self.failure_pos = (pos1[0], pos1[1])
        # else:
        #     self.success_pos = (pos1[0], pos1[1])
        #     self.failure_pos = (pos0[0], pos0[1])

        self.mission = 'go to the matching object'

    def step(self, action):
        if action == MiniGridEnv.Actions.pickup:
            action = MiniGridEnv.Actions.toggle
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            done = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            done = True

        return obs, reward, done, info

class DelayedMatchingS17Random(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=17, random_length=True)

register(
    id='MiniGrid-DelayedMatchingS17Random-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS17Random',
)

class DelayedMatchingS13Random(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13, random_length=True)

register(
    id='MiniGrid-DelayedMatchingS13Random-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS13Random',
)

class DelayedMatchingS13(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13)

register(
    id='MiniGrid-DelayedMatchingS13-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS13',
)

class DelayedMatchingS11(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=11)

register(
    id='MiniGrid-DelayedMatchingS11-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS11',
)

class DelayedMatchingS9(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9)

register(
    id='MiniGrid-DelayedMatchingS9-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS9',
)

class DelayedMatchingS7(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7)

register(
    id='MiniGrid-DelayedMatchingS7-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS7',
)
