from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class MiniGridNarrator(ABC):
    def __init__(self) -> None:
        super().__init__()

        # Object ID is channel 0 of the environment observation
        self._OBJECT_IDS = {
            "FLOOR_ID": 3,
            "DOOR_ID": 4,
            "KEY_ID": 5,
            "GOAL_ID": 8,
            "AGENT_ID": 10,
        }
        # Status is channel 2 of the environment observation
        self._STATUS_IDS = {
            "OPEN": 0,
            "CLOSED": 1,
            "LOCKED": 2,
        }

        self._COLOUR_IDS = {
            "red": 0,
            "green": 1,
            "blue": 2,
            "purple": 3,
            "yellow": 4,
            "grey": 5,
        }

        self._ID_TO_COLOUR = dict(
            zip(self._COLOUR_IDS.values(), self._COLOUR_IDS.keys())
        )

    def _get_object_location(
        self, observation: np.ndarray, object_id: int
    ) -> list[tuple]:
        """
        Returns the location of the object in the observation
        """
        observation = observation[:, :, 0]  # Remove colour and status info
        object_locations = np.nonzero(observation == object_id)
        locations: list[tuple] = []
        # third dimension is info
        for (
            col,
            row,
        ) in zip(*object_locations):
            locations.append((col, row))
        return locations

    def _calculate_distance(
        self, location1: tuple, location2: tuple, metric: str = "manhattan"
    ) -> Union[int, float]:
        """
        Returns the distance between two locations
        """
        if metric == "manhattan":
            return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
        else:
            return float(np.linalg.norm(np.array(location1) - np.array(location2)))

    def _agent_moved(self, observations: list[np.ndarray]) -> bool:
        """
        Returns whether the agent moved in the sequence of observations.
        """
        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]
        for i in range(1, len(observations)):
            try:
                agent_current_position = self._get_object_location(
                    observations[i], self._OBJECT_IDS["AGENT_ID"]
                )[0]
                if agent_current_position != agent_start_position:
                    return True
            except IndexError:
                return False
        return False

    def _get_agent_relative_movement_string(
        self,
        observations: list[np.ndarray],
        object_position: tuple[int, int],
        object_name: str,
    ) -> str:
        """_summary_

        Args:
            observations (list[np.ndarray]): _description_
            object_position (tuple[int, int]): _description_
            object_name (str): _description_

        Returns:
            str: _description_
        """

        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        if not self._agent_moved(observations):
            return "the agent did not move "

        agent_end_position = self._get_object_location(
            observations[-1], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        if agent_end_position == agent_start_position:
            return "the agent moved in a circle "

        start_distance = self._calculate_distance(agent_start_position, object_position)
        end_distance = self._calculate_distance(agent_end_position, object_position)

        if start_distance == end_distance:
            return (
                f"the agent stayed the same distance from the {object_name}, but moved "
            )
        elif start_distance > end_distance:
            return f"the agent moved towards the {object_name} "
        else:
            return f"the agent moved away from the {object_name} "

    @abstractmethod
    def narrate(self, observations: list[np.ndarray]) -> str:
        pass


class MiniGridFourSquareNarrator(MiniGridNarrator):
    def narrate(self, observations: list[np.ndarray]) -> str:
        first_obs = observations[0]
        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
            if (
                self._get_object_location(
                    observations[-1], self._OBJECT_IDS["AGENT_ID"]
                )[0]
                == goal_position
            ):
                return "the agent reached the goal "
        except IndexError:
            # Agent is standing on goal
            return "the agent reached the goal "
        if not self._agent_moved(observations):
            return "the agent did not move "
        if (
            self._get_object_location(observations[-1], self._OBJECT_IDS["AGENT_ID"])[0]
            == self._get_object_location(observations[0], self._OBJECT_IDS["AGENT_ID"])[
                0
            ]
        ):
            return "the agent moved in a circle "

        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        agent_end_position = self._get_object_location(
            observations[-1], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        goal_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["GOAL_ID"]
        )[0]

        coloured_square_positions = self._get_object_location(
            observations[0], self._OBJECT_IDS["FLOOR_ID"]
        )

        coloured_square_positions.append(goal_position)

        goal_colour = self._ID_TO_COLOUR[
            first_obs[goal_position[0], goal_position[1], 1]
        ]

        biggest_delta = 0.0
        closest_square = None

        for square_position in coloured_square_positions:
            square_colour = self._ID_TO_COLOUR[
                first_obs[square_position[0], square_position[1], 1]
            ]
            delta = self._calculate_distance(
                square_position, agent_start_position
            ) - self._calculate_distance(square_position, agent_end_position)
            if delta > biggest_delta:
                biggest_delta = delta
                closest_square = square_colour

        if closest_square == goal_colour:
            return f"the agent moved towards the {closest_square} square which is the goal "

        return f"the agent moved towards the {closest_square} square which is not the goal "


class MiniGridEmptyNarrator(MiniGridNarrator):

    def narrate(self, observations: list[np.ndarray]) -> str:
        first_obs = observations[0]
        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
        except IndexError:
            # Agent is standing on goal
            return "the agent reached the goal "

        if (
            self._get_object_location(observations[-1], self._OBJECT_IDS["AGENT_ID"])[0]
            == goal_position
        ):
            return "the agent reached the goal "

        return self._get_agent_relative_movement_string(
            observations,
            self._get_object_location(observations[0], self._OBJECT_IDS["GOAL_ID"])[0],  # type: ignore
            "goal",
        )


class MiniGridDoorKeyNarrator(MiniGridNarrator):

    def _get_key_status(
        self, first_obs: np.ndarray, last_obs: np.ndarray
    ) -> tuple[bool, int]:
        """Determines whether the key has already been picked up, has not been
        picked up, or if the agent picked up the key in a given window of
        observations.

        Args:
            first_obs (np.ndarray): encoded environment observation of first timestep
            of window. Shape (height, width, 3)
            last_obs (np.ndarray): encoded environment observation of last timestep
            of window. Shape (height, width, 3)

        Returns:
            tuple[bool, bool]: (agent_has_key, agent_picked_up_key)
        """

        agent_has_key = False
        agent_picked_up_key = False

        key_start_position = self._get_object_location(
            first_obs, self._OBJECT_IDS["KEY_ID"]
        )
        key_end_position = self._get_object_location(
            last_obs, self._OBJECT_IDS["KEY_ID"]
        )

        if key_start_position and not key_end_position:
            agent_picked_up_key = True
            agent_has_key = True
        elif not key_start_position:
            agent_has_key = True

        return agent_has_key, agent_picked_up_key

    def _get_key_pickup_frame(self, observations: list[np.ndarray]) -> int:
        """
        Returns the frame in which the agent picked up the key.
        """
        for i, obs in enumerate(observations):
            if not self._get_object_location(obs, self._OBJECT_IDS["KEY_ID"]):
                return i
        return -1

    def _get_door_unlock_frame(self, observations: list[np.ndarray]) -> int:
        """
        Returns the frame in which the agent unlocked the door.
        """
        for i, obs in enumerate(observations):
            try:
                door_position = self._get_object_location(
                    obs, self._OBJECT_IDS["DOOR_ID"]
                )[0]
            except IndexError:
                # Agent is standing on door
                door_position = self._get_object_location(
                    obs, self._OBJECT_IDS["AGENT_ID"]
                )[0]
            if obs[door_position[0], door_position[1], 2] != self._STATUS_IDS["LOCKED"]:
                return i
        return -1

    def _get_last_door_change_frame(
        self, observations: list[np.ndarray], door_position: tuple[int, int]
    ) -> int:
        """Returns the frame in which the door was last opened or closed.

        Args:
            observations (list[np.ndarray]): list of observatons to check for door
            door_position (tuple[int, int]): position of the door in the observation

        Returns:
            int: frame number
        """

        for i in range(len(observations) - 1, -1, -1):
            if (
                observations[i][door_position[0], door_position[1], 2]
                != observations[-1][door_position[0], door_position[1], 2]
            ):
                return i
        return -1

    def _get_door_lock_status(
        self,
        first_obs: np.ndarray,
        last_obs: np.ndarray,
        door_position: tuple[int, int],
    ) -> tuple[bool, bool]:
        """Gets whether the door is locked or unlocked, and whether the agent unlocked
        it in the current window of environment steps.

        Args:
            first_obs (np.ndarray): first observation in the window of shape
            (height, width, 3)
            last_obs (np.ndarray): last observation in the window of shape
            (height, width, 3)
            door_position (tuple[int, int]): position of the door in the observation
            (row, col)

        Returns:
            tuple[bool, bool]: door_locked, agent_unlocked_door
        """

        door_locked = False
        agent_unlocked_door = False

        initial_status = first_obs[door_position[0], door_position[1], 2]
        final_status = last_obs[door_position[0], door_position[1], 2]

        if initial_status == self._STATUS_IDS["LOCKED"]:
            door_locked = True
            if final_status != self._STATUS_IDS["LOCKED"]:
                agent_unlocked_door = True
                door_locked = False

        return door_locked, agent_unlocked_door

    def _get_door_open_close_sequence(
        self, observations: list[np.ndarray], door_position: tuple[int, int]
    ) -> str:
        """
        Returns a string describing the sequence of door open and close events.
        """
        door_open_close_sequence = ""
        current_status = observations[0][door_position[0], door_position[1], 2]
        door_changed = False
        for i in range(1, len(observations)):
            next_status = observations[i][door_position[0], door_position[1], 2]
            if next_status != current_status:
                if (
                    next_status == self._STATUS_IDS["OPEN"]
                    and current_status == self._STATUS_IDS["CLOSED"]
                ):
                    if door_changed:
                        door_open_close_sequence += "and then "
                    door_open_close_sequence += "the agent opened the door "
                    door_changed = True
                    current_status = next_status
                elif next_status == self._STATUS_IDS["CLOSED"]:
                    if door_changed:
                        door_open_close_sequence += "and then "
                    door_open_close_sequence += "the agent closed the door "
                    door_changed = True
                    current_status = next_status
        return door_open_close_sequence

    def narrate(self, observations: list[np.ndarray]) -> str:
        """
        Generates a narration from a sequence of observations.
        """
        narration_str = ""

        first_obs = observations[0]
        last_obs = observations[-1]

        agent_has_key, agent_picked_up_key = self._get_key_status(first_obs, last_obs)

        if agent_picked_up_key:
            narration_str += "the agent went and picked up the key, and then "
            pickup_frame = self._get_key_pickup_frame(observations)
            observations = observations[pickup_frame + 1 :]
            if not observations:
                return narration_str[: len("and then ")]
        elif not agent_has_key:
            # Get movement of agent relative to key
            key_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["KEY_ID"]
            )[0]
            narration_str += self._get_agent_relative_movement_string(
                observations, key_position, "key"  # type: ignore
            )
            return narration_str
        try:
            door_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["DOOR_ID"]
            )[0]
        except IndexError:
            # Agent is standing on door
            door_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["AGENT_ID"]
            )[0]
        door_locked, agent_unlocked_door = self._get_door_lock_status(
            first_obs, last_obs, door_position  # type: ignore
        )
        if door_locked:
            narration_str += self._get_agent_relative_movement_string(
                observations, door_position, "door"  # type: ignore
            )
            return narration_str

        elif agent_unlocked_door:
            door_locked = False
            narration_str += "the agent unlocked the door, and then "
            door_unlock_frame = self._get_door_unlock_frame(observations)
            observations = observations[door_unlock_frame + 1 :]
            if not observations:
                return narration_str[: len("and then ")]

        if not door_locked:
            door_open_close_sequence = self._get_door_open_close_sequence(
                observations, door_position  # type: ignore
            )
            narration_str += door_open_close_sequence
            if door_open_close_sequence != "":
                narration_str += "and then "
                door_last_change_frame = self._get_last_door_change_frame(
                    observations, door_position  # type: ignore
                )
                observations = observations[door_last_change_frame + 1 :]
                if not observations:
                    return narration_str[: len("and then ")]

        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
        except IndexError:
            # Agent is standing on goal
            narration_str += "the agent reached the goal "
            return narration_str

        if (
            self._get_object_location(last_obs, self._OBJECT_IDS["AGENT_ID"])[0]
            == goal_position
        ):
            narration_str += "the agent reached the goal "
        else:
            narration_str += self._get_agent_relative_movement_string(
                observations, goal_position, "goal "  # type: ignore
            )

        return narration_str


class MiniGridTeleportNarrator(MiniGridNarrator):
    def narrate(self, observations: list[np.ndarray]) -> str:
        """Generates a narration from a sequence of observations"""
        narration_str = ""
        return narration_str