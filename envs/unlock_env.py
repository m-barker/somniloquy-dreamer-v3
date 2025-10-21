import sys

sys.path.append("/home/matt/dev/somniloquy-dreamer-v3")
from typing import Tuple, Dict
import cv2
import gymnasium as gym
import numpy as np

from minigird_envs.custom_unlock import UnlockEnv
from wrappers import MiniGridRGBObsWrapper


class Unlock:
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        max_length: int = 1024,
        seed: int = 42,
    ):
        assert img_size[0] == img_size[1]

        self._max_length = max_length
        self._random = np.random.RandomState(seed)
        self._done = True
        self._step = 0
        self._img_size = img_size
        self.reward_range = [0, 1]
        self.env = self._create_env()

        # Toggle action is action idx 5 in minigrid, but we don't
        # care about action idx 4 (drop), so re-map
        self.action_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 5,
        }

    def _create_env(self) -> gym.Env:
        env = UnlockEnv(agent_start_cell=(1, 1), room_size=15, render_mode="rgb_array")
        # Forward, turn left, turn right, pickup, toggle
        env.action_space = gym.spaces.Discrete(5)
        # TODO: add RGB partial observability wrapper
        env = MiniGridRGBObsWrapper(env, full_obs=False)
        return env

    @property
    def observation_space(self):
        img_shape = self._img_size + (3,)
        # Assuming full obs for now.
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
                "occupancy_grid": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.env.width, self.env.height, 3),
                    dtype="uint8",
                ),
            }
        )

    @property
    def mission(self):
        return self.env.mission

    @property
    def action_space(self):
        space = self.env.action_space
        space.discrete = True
        return space

    def step(self, action):
        if len(action.shape) >= 1:
            action = int(np.argmax(action))

        action = self.action_mapping[action]

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        is_last = bool(
            terminated
            or (self._max_length and self._step >= self._max_length)
            or truncated
        )

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        resized_img = None
        if rgb_image.shape[:-2] != self._img_size:
            resized_img = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        return (
            {
                "image": resized_img if resized_img is not None else rgb_image,
                "is_terminal": terminated,
                "is_first": False,  # False, as we have just taken a step
                "high_res_image": obs["high_res_image"],
            },
            reward,
            is_last,
            {
                "occupancy_grid": occupancy_grid,
                "agent_direction": direction,
            },
        )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        Resets environment and returns obs, info
        """
        obs, info = self.env.reset(seed=seed)

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        resized_img = None
        if rgb_image.shape[:-2] != self._img_size:
            resized_img = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        return {
            "image": resized_img if resized_img is not None else rgb_image,
            "is_terminal": False,  # we're only just getting started!
            "is_first": True,
            "original_image": rgb_image,
            "high_res_image": obs["high_res_image"],
        }, {"occupancy_grid": occupancy_grid, "agent_direction": direction}

    def close(self):
        return self.env.close()
