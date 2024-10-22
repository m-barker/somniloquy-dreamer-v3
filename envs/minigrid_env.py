from typing import Tuple, Optional

import cv2  # type: ignore
import gymnasium as gym
import numpy as np

from .minigird_envs.minigrid_four_squares import FourSquares  # type: ignore
from .minigird_envs.teleport import Teleport5by5, TeleportComplex  # type: ignore
from .wrappers import MiniGridFullObsWrapper  # type: ignore


class MiniGrid:

    def __init__(
        self,
        task_name: str,
        img_size: Tuple[int, int] = (64, 64),
        actions: str = "all",
        max_length: int = 1024,
        seed: int = 42,
        full_obs: bool = True,
        human_render: bool = False,
    ):
        assert img_size[0] == img_size[1]
        assert actions in ("all", "needed"), actions

        self._actions = actions
        # Used when actions are needed (i.e., removing pointless actions)
        self._action_mapping: dict[int, int] = {}
        self._max_length = max_length
        self._random = np.random.RandomState(seed)
        self._full_obs = full_obs
        self._done = True
        self._step = 0
        self._img_size = img_size
        self.reward_range = [0, np.inf]
        self._human_render = human_render
        self._env = self._create_env(task_name)

    def _create_env(self, task_name: str) -> gym.Env:
        print(f"Creating MiniGrid environment for task: {task_name}")
        if task_name == "four_squares":
            env = FourSquares()
            if self._full_obs:
                env = MiniGridFullObsWrapper(env)
            else:
                raise NotImplementedError("Partial observation not implemented yet.")
            if self._actions == "needed":
                # Forward, Turn left, Turn right
                env.action_space = gym.spaces.Discrete(3)

        elif task_name == "teleport5x5":
            if self._human_render:
                env = Teleport5by5(render_mode="human")
            else:
                env = Teleport5by5()
            if self._full_obs:
                env = MiniGridFullObsWrapper(env)
            else:
                raise NotImplementedError("Partial observation not implemented yet.")
            env.action_space = gym.spaces.Discrete(3)
            if self._actions == "needed":
                # Forward, Turn left, Turn right
                env.action_space = gym.spaces.Discrete(3)
        elif task_name == "teleport_complex":
            if self._human_render:
                env = TeleportComplex(render_mode="human")
            else:
                env = TeleportComplex()
            if self._full_obs:
                env = MiniGridFullObsWrapper(env)
            else:
                raise NotImplementedError("Partial observation not implemented yet.")
            env.action_space = gym.spaces.Discrete(3)
            if self._actions == "needed":
                # Forward, Turn left, Turn right
                env.action_space = gym.spaces.Discrete(3)
        else:
            env = gym.make(task_name)
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

    def step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)

        obs, reward, over, truncated, info = self._env.step(action)
        self._step += 1
        self._done = (
            over or (self._max_length and self._step >= self._max_length) or truncated
        )

        return self._obs(
            obs["rgb_image"],
            reward,
            obs["encoded_image"],
            is_last=self._done,
            is_terminal=over or truncated,
        )

    def reset(self, seed=None, **kwargs):
        obs, info = self._env.reset()
        self._done = False
        self._step = 0
        obs, _, _, info = self._obs(
            obs["rgb_image"],
            0.0,
            obs["encoded_image"],
            is_first=True,
        )
        return obs, info

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
    from minigrid.manual_control import ManualControl

    env = MiniGrid(task_name="teleport5x5", full_obs=True, human_render=True)._env

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
