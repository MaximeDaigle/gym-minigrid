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
        tile_size=32,
        max_delay=30
    ):
        self.tile_size = tile_size
        self.max_frames_delay = max_delay
        super(DelayedMatchingEnv, self).__init__(
            seed=seed,
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=False,
        )
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.width * tile_size, self.height * tile_size, 3),
            dtype='uint8'
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        def cst(low,high):
            return low

        random_fct = self._rand_float
        # random_fct = cst

        # Create the 3 objects
        type_objets = {Circle, Triangle, Square}

        cue_type = self._rand_elem(type_objets)
        cue_obj = cue_type(color=self._rand_color(), random_fct=random_fct)
        type_objets.discard(cue_type) # remove cue object type from type left

        second_type = self._rand_elem(type_objets)

        self.matching_obj = cue_type(color=self._rand_color(), random_fct=random_fct)  # test object matching cue
        self.second_obj = second_type(color=self._rand_color(), random_fct=random_fct) # random second object

        self.cue_pos = self.place_obj(cue_obj)

        self.nb_delay_frames = self._rand_int(1, self.max_frames_delay)
        # self.nb_delay_frames = 1

        self.mission = 'Select the matching object'

    def step(self, action=''):
        label = -1
        done = False

        if self.step_count < self.nb_delay_frames:
            # Remove cue obj
            self.grid.set(*self.cue_pos, None)

        if self.step_count == self.nb_delay_frames:
            # Place test objects
            self.matching_pos = self.place_obj(self.matching_obj)
            second_pos = self.place_obj(self.second_obj)

        if self.step_count >= self.nb_delay_frames + 1:
            label = self.matching_pos[1] + (self.matching_pos[0]*self.height)
            done = True

        obs = self.gen_obs()['image']

        self.step_count += 1
        return obs, label, done, {}

    def reset(self):
        self._gen_grid(self.width, self.height)

        # Step count since episode start
        self.step_count = 0

        # first observation
        obs = self.gen_obs()['image']
        return obs

    def gen_obs(self):
        """
        Generate the observation. Here, it is the complete grid.
        """
        rgb_img = self.render(
            mode='rgb_array',
            tile_size=self.tile_size
        )
        return {
            'mission': self.mission,
            'image': rgb_img
        }

    def render(self, mode='human', close=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

class Circle(WorldObj):
    def __init__(self, random_fct, color='blue'):
        super(Circle, self).__init__('circle', color)
        scale = random_fct(0.0, 0.6)
        self.radius = 0.31*(1-scale)

        # default center is at the middle
        self.cx = 0.5
        self.cy = 0.5

        # Translation space available
        max_right = 0.95 - self.cx - self.radius # right limit - x coordinate of center - radius
        max_left = self.cx - self.radius - 0.05 # x coordinate of center - radius - left limit
        max_up = 0.95 - self.cy - self.radius # up limit - y coordinate of center - radius
        max_down = self.cy - self.radius - 0.05 # y coordinate of center - down limit - radius

        self.cx = random_fct(self.cx - max_left, self.cx + max_right)
        self.cy = random_fct(self.cy - max_down, self.cy + max_up)

    def render(self, img):
        fill_coords(img, point_in_circle(self.cx, self.cy, self.radius), COLORS[self.color])

class Triangle(WorldObj):
    def __init__(self, random_fct, color='blue'):
        super(Triangle, self).__init__('triangle', color)
        self.scale = random_fct(0.0, 0.6)

        # Resize
        a = (0.12*(1+self.scale), 0.12*(1+self.scale))
        b = ((0.12*(1+self.scale) + 0.88*(1-self.scale))/2, 0.78*(1-self.scale))
        c = (0.88*(1-self.scale), 0.12*(1+self.scale))

        # Translation space available
        up = 0.95 - b[1] # cord[1] + up => move at max up
        down = a[1] - 0.05 # cord[1] - down => move at max down
        right = a[0] - 0.05 # cord[0] - right => move at max right
        left = 0.95 - c[0] # cord[0] + left  => move at max left

        # default position is in the bottom right and have almost no empty space to move more in that direction
        move_up = random_fct(0.0, 1.0) < up # ~85% chance of moving up, ~15% to move down
        move_right = random_fct(0.0, 1.0) < right

        if move_up:
            translate_y = random_fct(0, up)
        else:
            translate_y = - random_fct(0, down)

        if move_right:
            translate_x = -random_fct(0, right)
        else:
            translate_x = random_fct(0, left)

        self.a = (a[0] + translate_x, a[1] + translate_y)
        self.b = (b[0] + translate_x, b[1] + translate_y)
        self.c = (c[0] + translate_x, c[1] + translate_y)

    def render(self, img):
        tri_fn = point_in_triangle(self.a, self.b, self.c)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * 2)
        fill_coords(img, tri_fn, COLORS[self.color])

class Square(WorldObj):
    def __init__(self, random_fct, color='blue'):
        super(Square, self).__init__('square', color)
        self.max_x = random_fct(0.55, 0.88)
        self.min_y = random_fct(0.12, 0.45)
        self.min_x = random_fct(0.12, 0.45)
        side_size = self.max_x - self.min_x
        self.max_y = self.min_y + side_size

    def render(self, img):
        fill_coords(img, point_in_rect(self.min_x, self.max_x, self.min_y, self.max_y), COLORS[self.color])

class DelayedMatchingS17Random(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=17)

register(
    id='MiniGrid-DelayedMatchingS17Random-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS17Random',
)

class DelayedMatchingS13Random(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13)

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

class DelayedMatchingS3(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=3)

register(
    id='MiniGrid-DelayedMatchingS3-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS3',
)

class DelayedMatchingS4(DelayedMatchingEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=4)

register(
    id='MiniGrid-DelayedMatchingS4-v0',
    entry_point='gym_minigrid.envs:DelayedMatchingS4',
)