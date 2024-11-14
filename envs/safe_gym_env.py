import argparse
from typing import Tuple, Dict
import gym
from PIL import Image
import numpy as np

from ai_safety_gridworlds.demonstrations import demonstrations
from safe_grid_gym.envs import GridworldEnv


class SafeGymEnv:
    """Implements a wrapper for the AI Safety Gym Environments"""

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


def gym_env(args):
    env = mk_env(args)
    env.reset()
    actions = get_actions(args, env)

    rr = []
    episode_return = 0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        episode_return += reward
        rgb_obs = env.render(mode="rgb_array").transpose(1, 2, 0)
        image = Image.fromarray(rgb_obs)
        image = image.resize((64, 64), resample=Image.BOX)
        image.show()

        if done:
            rr.append(episode_return)
            episode_return = 0
            env.reset()


def mk_env(args):
    return GridworldEnv(env_name=args.env_name, render_animation_delay=args.pause)


def get_actions(args, env):
    if args.rand_act:
        return [env.action_space.sample() for _ in range(args.steps)]
    else:
        demo = demonstrations.get_demonstrations(args.env_name)[0]
        return demo.actions


# --------
# main io
# --------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env_name",
        default="island_navigation",
        help="e.g. distributional_shift|side_effects_sokoban",
    )
    parser.add_argument("-r", "--rand_act", action="store_true")
    parser.add_argument("-g", "--gym_make", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gym_env(args)
