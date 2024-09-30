from typing import List, Dict
import numpy as np


class PandaPushColourNarrator:
    """Class for generating narrations of a sequence of observations for the
    custom "panda_push_colour" task.
    """

    def __init__(
        self, colour_to_push: str = "red", movement_delta: float = 0.1
    ) -> None:
        """
        Args:
            colour_to_push (str, optional): Which of the coloured blocks needs.
            to be pushed to the goal. Defaults to "red".

            movement_delta (float, optional): The minimum distance the robot needs to move
            to be considered as moving towards the box. Defaults to 0.1.
        """

        self._colour_to_push = colour_to_push
        self._REQUIRED_OBS_KEYS = [
            "red_box_pos",
            "red_box_rot",
            "green_box_pos",
            "green_box_rot",
            "blue_box_pos",
            "blue_box_rot",
            "desired_goal",
            "robot_obs",
        ]
        self._movement_delta = movement_delta

    def _get_box_movement_masks(
        self, observations: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Returns a mask indicating whether the boxes were moved in the current step.

        Args:
            observations (List[Dict[str, np.ndarray]]): List of environment observations
            that contains the box positions and rotations.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the masks for each box.
        """

        masks = {
            "red_box": np.zeros(len(observations)),
            "green_box": np.zeros(len(observations)),
            "blue_box": np.zeros(len(observations)),
        }

        for t in range(1, len(observations)):
            for box in ["red_box", "green_box", "blue_box"]:
                prev_pos = observations[t - 1][f"{box}_pos"]
                curr_pos = observations[t][f"{box}_pos"]
                if np.linalg.norm(prev_pos - curr_pos) > 0:
                    masks[box][t] = 1
        return masks

    def _no_box_movement(self, box_movement_masks: Dict[str, np.ndarray]) -> bool:
        """Returns True if none of the boxes were moved in the current step.

        Args:
            box_movement_masks (Dict[str, np.ndarray]): A dictionary containing the masks
            for each box.

        Returns:
            bool: True if none of the boxes were moved in the current step.
        """
        return all(np.all(mask == 0) for mask in box_movement_masks.values())

    def _moved_towards_box(self, observations: List[Dict[str, np.ndarray]]) -> bool:
        """Returns True if the robot moved towards the box in the current step.

        Args:
            observations (List[Dict[str, np.ndarray]]): List of environment observations
            that contains the box positions and rotations.

        Returns:
            bool: True if the robot moved towards the box in the current step.
        """
        robot_start_pos = observations[0]["robot_obs"][:3]
        robot_end_pos = observations[-1]["robot_obs"][:3]
        for box in ["red_box", "green_box", "blue_box"]:
            box_start_pos = observations[0][f"{box}_pos"]
            box_end_pos = observations[-1][f"{box}_pos"]
            start_dist = np.linalg.norm(robot_start_pos - box_start_pos)
            end_dist = np.linalg.norm(robot_end_pos - box_end_pos)
            if end_dist - start_dist > self._movement_delta:
                return True
        return False

    def _narrate_ee_trajectory(self, robot_positions: List[np.ndarray]) -> str:
        """Generates a narration for a given sequence of end-effector positions.

        Args:
            robot_positions (List[np.ndarray]): List of end-effector positions.

        Returns:
            str: A string containing the description.
        """

        # obs = [ee_pos (x, y, z), ee_vel (x, y, z), fingers_width]
        start_pos = robot_positions[0][:3]
        end_pos = robot_positions[-1][:3]

        x_start, y_start, z_start = start_pos
        x_end, y_end, z_end = end_pos

        movement_str = "I will move my end effector "

        if x_start < x_end:
            movement_str += "right and "
        elif x_start > x_end:
            movement_str += "left and "
        if y_start < y_end:
            movement_str += "forwards and "
        elif y_start > y_end:
            movement_str += "backwards and "
        if z_start < z_end:
            movement_str += "up and "
        elif z_start > z_end:
            movement_str += "down and "

        if movement_str == "I will move my end effector ":
            movement_str = "I will not move my end effector "

        if movement_str[-4:] == "and ":
            movement_str = movement_str[:-4]

        finger_width_start = robot_positions[0][6]
        finger_width_end = robot_positions[-1][6]

        if finger_width_start < finger_width_end:
            movement_str += "and I will open my fingers "
        elif finger_width_start > finger_width_end:
            movement_str += "and I will close my fingers "

        return movement_str

    def _narrate_ee_box_trajectory(
        self, observations: List[Dict[str, np.ndarray]]
    ) -> str:
        """Narrates the end effector trajectory after it has been determined that it moved towards
        at least one of the boxes.

        Args:
            observations (List[Dict[str, np.ndarray]]): List of environment observations

        Returns:
            str: A string containing the description.
        """

        robot_positions = [obs["robot_obs"] for obs in observations]
        robot_start_pos = robot_positions[0][:3]
        robot_end_pos = robot_positions[-1][:3]

        moved_towards = None
        moved_distance = -float("inf")

        for box in ["red_box", "green_box", "blue_box"]:
            box_start_pos = observations[0][f"{box}_pos"]
            box_end_pos = observations[-1][f"{box}_pos"]
            box_start_dist = np.linalg.norm(robot_start_pos - box_start_pos)
            box_end_dist = np.linalg.norm(robot_end_pos - box_end_pos)
            moved_distance_box = box_start_dist - box_end_dist
            if moved_distance_box > moved_distance:
                moved_distance = moved_distance_box
                moved_towards = box.replace("_", " ")

        if moved_towards is None:
            raise ValueError("No box was moved towards.")

        trajectory_str = f"I will move my end effector towards the {moved_towards} "

        # Check for opening or closing fingers
        finger_width_start = robot_positions[0][6]
        finger_width_end = robot_positions[-1][6]

        if finger_width_start < finger_width_end:
            trajectory_str += "and I will open my fingers "
        elif finger_width_start > finger_width_end:
            trajectory_str += "and I will close my fingers "

        return trajectory_str

    def narrate(self, observations: List[Dict[str, np.ndarray]]) -> str:
        """Generates a narration for a given sequence of observations that describes
        what the robotic arm agent is doing in the environment in this given sequence.

        Args:
            observations (List[Dict[str, np.ndarray]]): List of environment observations
            that contains the box positions and rotations.


        Returns:j
            str:  A string containing the description.
        """
        assert all(
            set(obs.keys()) == set(self._REQUIRED_OBS_KEYS) for obs in observations
        ), f"Observations must contain the required keys {self._REQUIRED_OBS_KEYS}."

        robot_positions = [obs["robot_obs"] for obs in observations]
        box_movement_masks = self._get_box_movement_masks(observations)

        if self._no_box_movement(box_movement_masks):
            if not self._moved_towards_box(observations):
                return self._narrate_ee_trajectory(robot_positions)
            else:
                return self._narrate_ee_box_trajectory(observations)
        narration_str = ""
        for t in range(1, len(observations)):
            for box in ["red_box", "green_box", "blue_box"]:
                box_moved_count = 0
                if box_movement_masks[box][t] == 1:
                    box_moved_count += 1
                    box_start_pos = observations[t - 1][f"{box}_pos"]
                    box_end_pos = observations[-1][f"{box}_pos"]
                    for j in range(t, len(observations)):
                        if box_movement_masks[box][j] == 0:
                            box_end_pos = observations[j - 1][f"{box}_pos"]
                            break
                    box_start_x, box_start_y, box_start_z = box_start_pos
                    box_end_x, box_end_y, box_end_z = box_end_pos
                    box_name = box.replace("_", " ")

                    if box_moved_count > 1:
                        narration_str += "and"
                    if box_end_z < -0.3:
                        narration_str = (
                            f"I will knock the {box_name} off the table clumsy me and "
                        )
                    else:
                        narration_str = f"I will move the {box_name} to the "

                        if box_start_x < box_end_x:
                            narration_str += "right and "
                        elif box_start_x > box_end_x:
                            narration_str += "left and "
                        if box_start_y < box_end_y:
                            narration_str += "forwards and "
                        elif box_start_y > box_end_y:
                            narration_str += "backwards and "

                        if narration_str[-4:] == "and ":
                            narration_str = narration_str[:-4]

        return narration_str
