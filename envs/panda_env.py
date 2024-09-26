from typing import Tuple, Optional

import cv2
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import PixelObservationWrapper
from .panda_envs.custom_tasks import PushColourTask


class PandaEnv:

    def __init__(
        self,
        task_name: str,
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
        self._env = self._create_env(task_name)

    def _create_env(self, task_name: str) -> gym.Env:

        if task_name == "push_colour":
            env = PushColourTask(render_mode="rgb_array")
        else:
            raise NotImplementedError(f"Panda Task {task_name} not implemented.")
        env = PixelObservationWrapper(env, pixel_keys=["image"], pixels_only=True)
        return env

    @property
    def observation_space(self):
        img_shape = self._img_size + (3,)
        # Assuming full obs for now.
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            }
        )

    @property
    def mission(self):
        return self._env.mission

    @property
    def action_space(self):
        space = self._env.action_space
        return space

    def step(self, action):
        obs, reward, over, truncated, info = self._env.step(action)
        self._step += 1
        self._done = (
            over or (self._max_length and self._step >= self._max_length) or truncated
        )

        return self._obs(
            obs["image"],
            reward,
            is_last=self._done,
            is_terminal=over or truncated,
        )

    def reset(self, seed=None, **kwargs):
        obs, info = self._env.reset()
        self._done = False
        self._step = 0
        result = self._obs(
            obs["image"],
            0.0,
            is_first=True,
        )[0]
        return result

    def _obs(
        self,
        img: np.ndarray,
        reward: float,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> Tuple[dict, float, bool, dict]:
        image = img
        if image.shape[:2] != self._img_size:
            image = cv2.resize(image, self._img_size, interpolation=cv2.INTER_AREA)
        return (
            {
                "image": image,
                "is_terminal": is_terminal,
                "is_first": is_first,
            },
            reward,
            is_last,
            {},
        )

    def close(self):
        return self._env.close()
