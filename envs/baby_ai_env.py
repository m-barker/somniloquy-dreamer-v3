import sys

sys.path.append("/home/matt/dev/somniloquy-dreamer-v3")
from typing import Dict, SupportsFloat, Tuple, Optional
import cv2
import gymnasium as gym
import numpy as np

from wrappers import MiniGridFullObsWrapper  # type: ignore


class BabyAI:
    def __init__(
        self,
        task_name: str,
        img_size: Tuple[int, int] = (64, 64),
        actions: str = "needed",
        max_length: int = 1024,
        seed: int = 42,
        full_obs: bool = True,
        human_render: bool = False,
        reward: bool = False,
        terminate: bool = False,
    ):
        """
        Wrapper for the BabyAI environments

        Arguments:
            - task_name (str): name of the BabyAI task. Must be a valid one as
                               given by https://minigrid.farama.org/environments/babyai/

            - img_size (Tuple[int, int]): (H,W) of image to return. Resized if source image is
                                          too big.

            - actions (str): "all" or "needed". If "needed", the action space is filtered to
                             remove actions that are irrelevant for solving the task, e.g.,
                             the 'pickup' action in a navigation task.

            - max_length (int): maximum length of an episode. Returns truncated after this
                                number of environment steps. Defaults to 1024

            - seed (int): seed of the environment. Defaults to 42 - the meaning of life, the
                          universe, and everything.

            - full_obs (bool): whether to use full or partial observability. Defaults to True.

            - human_render (bool): whether to render the environment to the screen. Defaults to
                                   False, as obviously much slower than headless.

            - reward (bool): whether to use the reward function given by the environment. If
                             false, the environment gives no reward signal, used for training
                             a world model with no policy. Defaults to false.

            - terminate (bool): whether the episode should ever terminate. Disabled when the
                                aim is to only learn a dynamics model. Defaults to False.
        """
        assert img_size[0] == img_size[1]
        assert actions in ("all", "needed"), actions

        self._actions = actions
        self._max_length = max_length
        self._random = np.random.RandomState(seed)
        self._seed = seed
        self._full_obs = full_obs
        self._done = True
        self._step = 0
        self._img_size = img_size
        self._human_render = human_render

        self._reward = reward
        self._terminate = terminate

        self._env = self._create_env(task_name)

    def _create_env(self, task_name: str) -> gym.Env:
        print(f"Creating BabyAI environment for task: {task_name}")
        render_mode = None
        if self._human_render:
            render_mode = "human"

        env = gym.make(
            task_name,
            max_episode_steps=self._max_length,
            render_mode=render_mode,
        )
        if self._actions == "needed":
            # Forward, Turn left, Turn right
            env.action_space = gym.spaces.Discrete(3)
        if self._full_obs:
            env = MiniGridFullObsWrapper(env)
        else:
            raise NotImplementedError("Partial observation not implemented yet.")
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
                    shape=(self._env.width, self._env.height, 3),
                    dtype="uint8",
                ),
                "flattened_occupancy_grid": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._env.width * self._env.height * 3,),
                    dtype="uint8",
                ),
                "direction": gym.spaces.Discrete(4),
            }
        )

    @property
    def mission(self):
        return self._env.mission

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action) -> Tuple[Dict, SupportsFloat, bool, Dict]:
        """
        Returns obs, reward, done, info.

        We pass the non-Dreamer required observations (i.e., for the
        narrator) in info.
        """
        if len(action.shape) >= 1:
            action = np.argmax(action)

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        is_last = bool(
            truncated or (self._max_length and self._step >= self._max_length)
        )
        if not self._terminate:
            terminated = False
        if not self._reward:
            reward = 0.0

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        if rgb_image.shape[:-2] != self._img_size:
            rgb_image = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        return (
            {
                "image": rgb_image,
                "is_terminal": terminated,
                "is_first": False,  # False, as we have just taken a step
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
        obs, info = self._env.reset(seed=seed)
        self._step = 0

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        if rgb_image.shape[:-2] != self._img_size:
            rgb_image = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        return {
            "image": rgb_image,
            "is_terminal": False,  # we're only just getting started!
            "is_first": True,
        }, {"occupancy_grid": occupancy_grid, "agent_direction": direction}

    def _obs(
        self,
        img: np.ndarray,
        reward: float,
        occupancy_grid: np.ndarray,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> Tuple[dict, float, bool, dict]:
        image = img
        if image.shape[:2] != self._img_size:
            image = cv2.resize(image, self._img_size, interpolation=cv2.INTER_AREA)
        flattened_occupancy_grid = (occupancy_grid.flatten() / 11).astype(np.float32)
        return (
            {
                "image": image,
                "is_terminal": is_terminal,
                "is_first": is_first,
                "flattened_occupancy_grid": flattened_occupancy_grid,
            },
            reward,
            is_last,
            {
                "occupancy_grid": occupancy_grid,
            },
        )

    def close(self):
        return self._env.close()


if __name__ == "__main__":
    env = BabyAI(
        task_name="BabyAI-GoToLocal-v0",
        full_obs=True,
        human_render=True,
        max_length=16,
        seed=42,
    )
    obs, info = env.reset()
    done = False
    while not done:
        action_arr = np.zeros((env.action_space.n,), dtype=np.int16)
        action = input("Please enter an action: ")
        action_arr[int(action)] = 1
        obs, reward, done, info = env.step(action_arr)
        print(f"Obs: {obs.keys()}")
        print(f"Reward: {reward}")
        print(f"Is terminal: {obs['is_terminal']}")
        print(f"Is done: {done}")
        print(f"Info: {info.keys()}")
