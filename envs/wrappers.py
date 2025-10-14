import datetime
import gym
import numpy as np
import uuid
import pygame
import pygame.freetype
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX  # type: ignore


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        # assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self, seed=None, options=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()


class MiniGridFullObsWrapper(ObservationWrapper):
    """
    Combines MiniGrid's RGB and FullObs wrappers into one, and adds a high-quality
    'human'-style rendered image as 'high_res_image' (generated headlessly).
    """

    def __init__(self, env, tile_size: int = 8):
        super().__init__(env)
        self._tile_size = tile_size

        rgb_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self._tile_size * self.env.width,
                self._tile_size * self.env.height,
                3,
            ),
            dtype="uint8",
        )

        encoded_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),
            dtype="uint8",
        )

        # We'll assume high_res_image uses env.screen_size (same as 'human' render)
        high_res_size = getattr(self.env, "screen_size", 640)
        high_res_space = spaces.Box(
            low=0,
            high=255,
            shape=(high_res_size, high_res_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "rgb_image": rgb_image_space,
                "encoded_image": encoded_image_space,
                "high_res_image": high_res_space,
            }
        )

    def _render_high_res_image(self):
        """Generate a headless high-quality image that looks like MiniGrid's human render."""
        env = self.unwrapped
        img = env.get_frame(env.highlight, env.tile_size, env.agent_pov)
        img = np.transpose(img, axes=(1, 0, 2))

        # Initialize pygame headlessly if needed
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)

        surf = pygame.surfarray.make_surface(img)
        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        if pygame.display.get_surface() is not None:
            bg = bg.convert()  # Safe: only if display surface exists

        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))
        bg = pygame.transform.smoothscale(bg, (env.screen_size, env.screen_size))

        # Optional mission text
        font_size = 22
        text = getattr(env, "mission", "")
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.5
        font.render_to(bg, text_rect, text, size=font_size)

        # Convert to numpy
        rgb_array = pygame.surfarray.array3d(bg)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        return rgb_array

    def observation(self, observation) -> dict:
        env = self.unwrapped
        rgb_image = self.get_frame(highlight=True, tile_size=self._tile_size)

        # Encode the full environment grid (with agent)
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        # Generate high-resolution render
        high_res_image = self._render_high_res_image()

        return {
            **observation,
            "rgb_image": rgb_image,
            "encoded_image": full_grid,
            "high_res_image": high_res_image,
        }
