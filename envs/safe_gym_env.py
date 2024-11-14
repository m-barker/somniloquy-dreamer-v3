from typing import Tuple, Dict

import gym
import numpy as np
from PIL import Image

from ai_safety_gridworlds.demonstrations import demonstrations
from safe_grid_gym.envs import GridworldEnv


class SafeGymEnv:
    """Implements a wrapper for the AI Safety Gym Environments
    Wraps the safe_gym environment provided by https://github.com/david-lindner/safe-grid-gym
    which in turn is based on:
        - https://github.com/n0p2/ai-safety-gridworlds-viewer
        - https://github.com/n0p2/gym_ai_safety_gridworlds

    """

    def __init__(
        self,
        task_name: str,
        img_size: Tuple[int, int] = (64, 64),
        max_length: int = 1024,
        seed: int = 42,
    ) -> None:
        """SafeGym wrapper

        Args:
            task_name (str): Name of the ai-safety-gridworld task.
            img_size (Tuple[int, int], optional): (H,W) of image. Defaults to (64, 64).
            max_length (int, optional): Maximum number of steps in the episode. Defaults to 1024.
            seed (int, optional): Seed of environment. Defaults to 42.
        """

        self._max_length = max_length
        self._image_size = img_size
        self._step = 0
        self._done = True
        self._env = self._create_env(task_name, seed)

    def _create_env(self, task_name: str, seed: int) -> GridworldEnv:
        """Creates the ai-safety-grid-world environment

        Args:
            task_name (str): Name of the task.
            seed (int): Random seed to use

        Returns:
            GridworldEnv: gym compatable Environment object
        """

        env = GridworldEnv(env_name=task_name)
        env.seed = seed
        return env

    @property
    def observation_space(self):
        img_shape = self._img_size + (3,)
        return gym.spaces.Dict({"image": gym.spaces.Box(0, 255, img_shape, np.uint8)})

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        """Takes a step in the environment

        Args:
            action (_type_): _description_

        Returns:
            Tuple[Dict, float, bool, Dict]: Updated (obs, reward, done, info)
        """
        if len(action.shape) >= 1:
            action = np.argmax(action)

        obs, reward, done, info = self._env.step(action)
        self._step += 1
        self._done = done or (self._max_length and self._step >= self._max_length)  # type: ignore
        # (C, H, W) -> (H, W, C)
        image = self._env.render(mode="rgb_array").transpose(1, 2, 0)
        image = Image.fromarray(image)
        image = image.resize((64, 64), resample=Image.BOX)  # type: ignore

        return (
            {
                "image": image,
                "is_terminal": done,
                "is_first": False,
            },
            reward,
            self._done,
            info.update({"occupancy_grid", obs}),
        )

    def reset(self):
        obs = self._env.reset()
        self._done = False
        self._step = 0
        image = self._env.render(mode="rgb_array").transpose(1, 2, 0)
        image = Image.fromarray(image)
        image = image.resize((64, 64), resample=Image.BOX)  # type: ignore
        return (
            {
                "image": image,
                "is_terminal": False,
                "is_first": True,
            },
            0.0,
            self._done,
            {"occupancy_grid", obs},
        )

    def close(self):
        return self._env.close()
