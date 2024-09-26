from typing import Dict, Any, Tuple
import numpy as np

from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym.utils import distance
from gymnasium.wrappers import PixelObservationWrapper


class PushColour(Task):
    def __init__(
        self,
        sim,
        colour_to_push="red",
        reward_type="dense",
        distance_threshold=0.05,
    ):
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self._colour_to_push = colour_to_push
        self._object_size = 0.04
        self.goal_range_low = np.array([-0.4, -0.3, 0])
        self.goal_range_high = np.array([-0.05, 0.3, 0])
        self.obj_range_low = np.array([-0.4, -0.3, 0])
        self.obj_range_high = np.array([-0.05, 0.3, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="red_box",
            half_extents=np.ones(3) * self._object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self._object_size / 2]),
            rgba_color=[1, 0, 0, 1],
        )
        self.sim.create_box(
            body_name="green_box",
            half_extents=np.ones(3) * self._object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self._object_size / 2]),
            rgba_color=[0, 1, 0, 1],
        )
        self.sim.create_box(
            body_name="blue_box",
            half_extents=np.ones(3) * self._object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self._object_size / 2]),
            rgba_color=[0, 0, 1, 1],
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self._object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self._object_size / 2]),
            rgba_color=np.array([0.5, 0.5, 0.0, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("red_box"))
        object_rotation = np.array(self.sim.get_base_rotation("red_box"))
        object_velocity = np.array(self.sim.get_base_velocity("red_box"))
        object_angular_velocity = np.array(
            self.sim.get_base_angular_velocity("red_box")
        )
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def is_failure(self) -> bool:
        """Returns True if any of the boxes fall off the table."""

        red_box_pos = np.array(self.sim.get_base_position("red_box"))
        green_box_pos = np.array(self.sim.get_base_position("green_box"))
        blue_box_pos = np.array(self.sim.get_base_position("blue_box"))
        if red_box_pos[2] < -0.3 or green_box_pos[2] < -0.3 or blue_box_pos[2] < -0.3:
            return True
        return False

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(
            self.sim.get_base_position(f"{self._colour_to_push}_box")
        )
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        red_box_pos = self._sample_object()
        blue_box_pos = self._sample_object()
        green_box_pos = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("red_box", red_box_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose(
            "green_box", green_box_pos, np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.sim.set_base_pose("blue_box", blue_box_pos, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array(
            [0.0, 0.0, self._object_size / 2]
        )  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self._object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        if self.is_failure():
            return -np.array(10.0, dtype=np.float32)
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)


class PushColourTask(RobotTaskEnv):
    def __init__(self, render_mode="human", renderer="Tiny"):
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim)
        task = PushColour(sim)

        super().__init__(
            robot,
            task,
            render_target_position=np.array([-0.5, 0.0, 0.0]),
            render_yaw=275,
            render_pitch=-60,
            render_distance=1.0,
            render_roll=0,
            render_height=64,
            render_width=64,
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated if the agent has reached the target
        # or if a box has fallen off of the table
        terminated = bool(
            self.task.is_success(observation["achieved_goal"], self.task.get_goal())
            or self.task.is_failure()
        )
        truncated = False
        info = {"is_success": terminated}
        reward = float(
            self.task.compute_reward(
                observation["achieved_goal"], self.task.get_goal(), info
            )
        )
        return observation, reward, terminated, truncated, info


def main():
    import cv2

    env = PushColourTask(render_mode="rgb_array", renderer="Tiny")
    env = PixelObservationWrapper(env, pixels_only=True, pixel_keys=["image"])
    env.reset()
    while True:
        action = env.action_space.sample()
        action_space = env.action_space
        print(action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print("reward: ", reward)
        if terminated:
            break


if __name__ == "__main__":
    main()
