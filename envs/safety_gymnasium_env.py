from typing import Dict, Tuple
import safety_gymnasium
import gymnasium as gym
import numpy as np
import cv2


def test_simulation_speed() -> None:
    env = safety_gymnasium.make("SafetyPointGoal2-v0", render_mode="human")
    env.reset()
    env.render()
    action = env.action_space.sample()
    move_forward = action
    move_forward[1] = 0.0  # No Z-axis velocity
    move_forward[0] = 1.0  # Full speed ahead!
    t = 0
    while True:
        if t % 15 == 0:
            input("Taken 15 steps")
        env.step(move_forward)
        env.render()
        t += 1


class SafeNavigationWrapper:
    """Custom wrapper for the Navigation environments of Safety Gymnasium.

    Stores and returns additional information needed in order to narrate
    this task, i.e., go from a sequence of environment states to a natural
    language description.
    """

    def __init__(
        self,
        task_name: str = "SafetyPointGoal2-v0",
        max_steps: int = 5000,
        image_size: Tuple[int, int] = (64, 64),
        seed: int = 42,
    ) -> None:
        """
        Args:
            task_name [str, optional]: name of the safety_gymnasium task to create.
            Defaults to "SafetyPointGoal2-v0"

            max_steps [int, optional]: maximum number of steps before the environment
            returns truncated. Defaults to 5000

            image_size [Tuple[int, int]]: height, width of the RGB image. Resizes the
            one returned by the environment if neccessary. Defaults to (64, 64).

            seed [int, optional]: seed used for environment randomness. Defaults to 42.
        """

        self.env = safety_gymnasium.make(task_name, render_mode="rgb_array")
        self._max_steps = max_steps
        self._img_size = image_size
        self._seed = seed
        self._step = 0

    @property
    def observation_space(self):
        img_shape = self._img_size + (3,)
        return gym.spaces.Dict({"image": gym.spaces.Box(0, 255, img_shape, np.uint8)})

    @property
    def action_space(self):
        return self.env.action_space

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes an image using openCV.
        """

        return cv2.resize(image, self._img_size, interpolation=cv2.INTER_AREA)

    def reset(self) -> Tuple[Dict, Dict]:
        """
        Resets the environment.
        Returns obs, info
        """
        obs, info = self.env.reset()
        rgb_image = self.env.render()
        assert isinstance(rgb_image, np.ndarray)
        rgb_image = self._resize_image(rgb_image)
        self._step = 0

        return (
            {
                "image": rgb_image,
                "is_terminal": False,
                "is_first": True,
            },
            {},
        )

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        """
        Takes a step in the environment using the given action.

        Returns obs, reward, done, info
        """

        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._max_steps:
            truncated = True
        done = bool(terminated or truncated)

        rgb_image = self.env.render()
        assert isinstance(rgb_image, np.ndarray)
        rgb_image = self._resize_image(rgb_image)

        return (
            {
                "image": rgb_image,
                "is_terminal": terminated,
                "is_first": False,
            },
            float(reward),
            done,
            {},
        )


if __name__ == "__main__":
    env = SafeNavigationWrapper()
    obs, info = env.reset()
    # test_simulation_speed()
