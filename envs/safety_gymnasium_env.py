from typing import Dict, List, Tuple
import safety_gymnasium
from safety_gymnasium.builder import BaseTask, Builder
from safety_gymnasium.tasks.safe_navigation.goal.goal_level2 import GoalLevel2
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

        env = safety_gymnasium.make(task_name, render_mode="rgb_array")
        # assert isinstance(env, Builder), type(env)
        self.env: Builder = env
        self.task: BaseTask = self.env.task
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
        agent_pos, goal_pos, hazards_pos, vases_pos = self._get_position_data()

        high_res_image = self.env.render()
        assert isinstance(high_res_image, np.ndarray)
        rgb_image = self._resize_image(high_res_image)

        self._step = 0

        return (
            {
                "image": rgb_image,
                "is_terminal": False,
                "is_first": True,
                "high_res_image": high_res_image,
            },
            {
                "agent_pos": agent_pos,
                "goal_pos": goal_pos,
                "hazards_pos": hazards_pos,
                "vases_pos": vases_pos,
                "cost_hazards": 0.0,
                "cost_vases_contact": 0.0,
                "cost_vases_velocity": 0.0,
                "cost_sum": 0.0,
            },
        )

    def _get_position_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """ """
        obstacle_list = self.task._obstacles
        goal = [x for x in obstacle_list if x.name == "goal"]
        assert len(goal) == 1, f"More than one goal found {goal}"
        goal = goal[0]

        hazards = [x for x in obstacle_list if x.name == "hazards"]
        assert len(hazards) == 1, f"More than one set of hazards found {hazards}"
        hazards = hazards[0]
        vases = [x for x in obstacle_list if x.name == "vases"]
        assert len(vases) == 1, f"More than one set of vases found {vases}"
        vases = vases[0]

        goal_pos = goal.pos
        hazards_pos = hazards.pos
        vases_pos = vases.pos

        assert isinstance(hazards_pos, List)
        assert isinstance(vases_pos, List)

        agent_pos = goal.agent.pos

        return agent_pos, goal_pos, hazards_pos, vases_pos

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        """
        Takes a step in the environment using the given action.

        Returns obs, reward, done, info
        """

        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        agent_pos, goal_pos, hazards_pos, vases_pos = self._get_position_data()

        self._step += 1
        if self._step >= self._max_steps:
            truncated = True
        done = bool(terminated or truncated)

        high_res_image = self.env.render()
        assert isinstance(high_res_image, np.ndarray)
        rgb_image = self._resize_image(high_res_image)

        return (
            {
                "image": rgb_image,
                "is_terminal": terminated,
                "is_first": False,
                "high_res_image": high_res_image,
            },
            float(reward),
            done,
            {
                "agent_pos": agent_pos,
                "goal_pos": goal_pos,
                "hazards_pos": hazards_pos,
                "vases_pos": vases_pos,
                **info,
            },
        )


if __name__ == "__main__":
    env = SafeNavigationWrapper()
    obs, info = env.reset()
    # test_simulation_speed()
