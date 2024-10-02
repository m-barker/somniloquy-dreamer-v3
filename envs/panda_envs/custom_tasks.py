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
        self.goal_range_low = np.array([-0.4, 0.1, 0])
        self.goal_range_high = np.array([-0.05, 0.3, 0])
        self.obj_range_low = np.array([-0.4, -0.3, 0])
        self.obj_range_high = np.array([-0.05, 0.0, 0])
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

    def get_obs(self) -> Dict[str, np.ndarray]:
        # position of the boxes.
        red_box_pos = np.array(self.sim.get_base_position("red_box"))
        red_box_rot = np.array(self.sim.get_base_rotation("red_box"))
        blue_box_pos = np.array(self.sim.get_base_position("blue_box"))
        blue_box_rot = np.array(self.sim.get_base_rotation("blue_box"))
        green_box_pos = np.array(self.sim.get_base_position("green_box"))
        green_box_rot = np.array(self.sim.get_base_rotation("green_box"))
        achieved_goal = self.get_achieved_goal()
        desired_goal = self.get_goal()

        observation = {
            "red_box_pos": red_box_pos,
            "red_box_rot": red_box_rot,
            "green_box_pos": green_box_pos,
            "green_box_rot": green_box_rot,
            "blue_box_pos": blue_box_pos,
            "blue_box_rot": blue_box_rot,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return observation

    def is_failure(self) -> bool:
        """Returns True if any of the boxes fall off the table."""

        red_box_pos = np.array(self.sim.get_base_position("red_box"))
        green_box_pos = np.array(self.sim.get_base_position("green_box"))
        blue_box_pos = np.array(self.sim.get_base_position("blue_box"))
        return (
            red_box_pos[2] < -0.3 or green_box_pos[2] < -0.3 or blue_box_pos[2] < -0.3
        )

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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Gets the environment observation, in this case a dictionary containing the
        position and rotation of the coloured boxes on the table.

        Returns:
            Dict[str, np.ndarray]: Obs name: Obs value
        """
        obs: Dict[str, np.ndarray] = self.task.get_obs()
        obs["desired_goal"] = self.task.get_goal().astype(np.float32)
        robot_obs = self.robot.get_obs().astype(np.float32)
        # ee position (x,y,z), velocity (x,y,z), fingers width
        # so shape (7,)
        obs["robot_obs"] = robot_obs
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated if the agent has reached the target
        # or if a box has fallen off of the table
        success = self.task.is_success(
            observation["achieved_goal"], self.task.get_goal()
        )
        terminated = success
        truncated = False
        info = {"is_success": terminated}
        reward = float(
            self.task.compute_reward(
                observation["achieved_goal"], self.task.get_goal(), info
            )
        )
        if success:
            reward += 1000.0
        return observation, reward, terminated, truncated, info


def main():
    import cv2

    env = PushColourTask(render_mode="human")
    # env = PixelObservationWrapper(env, pixels_only=True, pixel_keys=["image"])
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
