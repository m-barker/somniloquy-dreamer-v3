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
            "TELEPORTER_ID": 11,
        }
        # Status is channel 2 of the environment observation
        self._DOOR_STATUS_IDS = {
            "OPEN": 0,
            "CLOSED": 1,
            "LOCKED": 2,
        }

        self._TELEPORTER_STATUS_IDS = {
            "ACTIVE": 0,
            "INACTIVE": 1,
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
    ) -> list[tuple[int, int]]:
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
            if (
                obs[door_position[0], door_position[1], 2]
                != self._DOOR_STATUS_IDS["LOCKED"]
            ):
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

        if initial_status == self._DOOR_STATUS_IDS["LOCKED"]:
            door_locked = True
            if final_status != self._DOOR_STATUS_IDS["LOCKED"]:
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
                    next_status == self._DOOR_STATUS_IDS["OPEN"]
                    and current_status == self._DOOR_STATUS_IDS["CLOSED"]
                ):
                    if door_changed:
                        door_open_close_sequence += "and then "
                    door_open_close_sequence += "the agent opened the door "
                    door_changed = True
                    current_status = next_status
                elif next_status == self._DOOR_STATUS_IDS["CLOSED"]:
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
    def __init__(self) -> None:
        super().__init__()

        # Hardcoding to the 5x5 grid size for now.
        # If we use this more, need to make this more general.

        self._TELEPORTER_POSITIONS = {
            "active_teleporter": (3, 4),
            "left_teleporter": (2, 2),
            "right_teleporter": (4, 2),
        }

        self._NON_TELEPORT_POSITIONS = [
            (3, 4),
            (2, 4),
            (1, 4),
            (4, 4),
            (5, 4),
            (3, 5),
            (2, 5),
            (1, 5),
            (4, 5),
            (5, 5),
        ]

        self._LEFT_TELEPORT_POSITIONS = [
            (2, 2),
            (1, 2),
            (1, 1),
            (2, 1),
        ]

        self._RIGHT_TELEPORT_POSITIONS = [
            (4, 2),
            (5, 2),
            (4, 1),
            (5, 1),
        ]

        self._LEFT_GOAL_POSITION = (1, 1)
        self._RIGHT_GOAL_POSITION = (5, 1)

    def narrate(self, observations: list[np.ndarray]) -> str:
        """Generates a narration from a sequence of observations"""
        narration_str = ""
        try:
            agent_start_position = self._get_object_location(
                observations[0], self._OBJECT_IDS["AGENT_ID"]
            )[0]
        except IndexError:
            return "I will reach the goal"

        try:
            agent_end_position = self._get_object_location(
                observations[-1], self._OBJECT_IDS["AGENT_ID"]
            )[0]
        except IndexError:
            if agent_start_position in self._RIGHT_TELEPORT_POSITIONS:
                agent_end_position = self._RIGHT_GOAL_POSITION
            elif agent_start_position in self._LEFT_TELEPORT_POSITIONS:
                agent_end_position = self._LEFT_GOAL_POSITION
            else:
                for obs in observations:
                    agent_pos = self._get_object_location(
                        obs, self._OBJECT_IDS["AGENT_ID"]
                    )[0]
                    if agent_pos in self._RIGHT_TELEPORT_POSITIONS:
                        return "I will go through the teleporter and teleport right"
                    elif agent_pos in self._LEFT_TELEPORT_POSITIONS:
                        return "I will go through the teleporter and teleport left"
                raise ValueError(
                    "Agent did not reach goal or teleporter, but error occurred in getting agent end position"
                )

        if agent_start_position in self._NON_TELEPORT_POSITIONS:
            if agent_end_position in self._NON_TELEPORT_POSITIONS:
                narration_str += "I will not teleport yet"
            elif agent_end_position in self._LEFT_TELEPORT_POSITIONS:
                narration_str += "I will go through the teleporter and teleport left"
            elif agent_end_position in self._RIGHT_TELEPORT_POSITIONS:
                narration_str += "I will go through the teleporter and teleport right"
        elif agent_start_position in self._LEFT_TELEPORT_POSITIONS:
            narration_str += "I have already teleported left"
        elif agent_start_position in self._RIGHT_TELEPORT_POSITIONS:
            narration_str += "I have already teleported right"

        if agent_end_position == self._LEFT_GOAL_POSITION:
            narration_str += " and I will reach the left goal"
        elif agent_end_position == self._RIGHT_GOAL_POSITION:
            narration_str += " and I will reach the right goal"
        return narration_str


class MiniGridComplexTeleportNarrator(MiniGridNarrator):
    def __init__(self) -> None:
        super().__init__()

        self._teleporter_information = {
            "blue teleporter": {
                "position": (2, 7),
                "reachable_from": [(1, 7), (2, 8)],
                "destinations": [(1, 5), (5, 3), (4, 8)],
            },
            "green teleporter": {
                "position": (2, 4),
                "reachable_from": [(1, 4), (2, 5)],
                "destinations": [(2, 2), (4, 8)],
            },
            "left purple teleporter": {
                "position": (4, 6),
                "reachable_from": [(4, 7), (5, 6)],
                "destinations": [(2, 8), (5, 3)],
            },
            "right purple teleporter": {
                "position": (6, 7),
                "reachable_from": [(6, 8), (5, 7), (6, 6)],
                "destinations": [(5, 3), (8, 8)],
            },
        }

        self._goal_position = (8, 1)

        self._room_positions = {
            "blue teleporter room": [(1, 7), (1, 8), (2, 7), (2, 8)],
            "green teleporter room": [(1, 4), (1, 5), (2, 4), (2, 5)],
            "purple teleporter room": [
                (4, 6),
                (4, 7),
                (4, 8),
                (5, 6),
                (5, 7),
                (5, 8),
                (6, 6),
                (6, 7),
                (6, 8),
            ],
            "left goal corridor": [(1, 1), (1, 2), (2, 1), (2, 2)],
            "middle goal corridor": [(4, 3), (4, 4), (5, 4), (5, 3), (6, 4), (6, 3)],
            "bottom goal corridor": [(8, 8), (8, 7), (8, 6), (8, 5)],
        }

    def _get_room_location(self, position: tuple[int, int]) -> str:
        for room, positions in self._room_positions.items():
            if position in positions:
                return room
        raise ValueError(f"Position {position} not in any room")

    def narrate(self, observations: list[np.ndarray]) -> str:
        """Generates a narration from a sequence of observations"""
        narration_str = ""
        goal_corridor = False
        try:
            agent_start_position = self._get_object_location(
                observations[0], self._OBJECT_IDS["AGENT_ID"]
            )[0]
        # This happens in the rare case the agent's start position
        # is on top of the goal -- their position is masked by the goal.
        except IndexError:
            return "I will reach the goal"
        if agent_start_position == self._goal_position:
            return "I will reach the goal"
        teleporter_room = True
        current_room = None
        if agent_start_position in self._room_positions["blue teleporter room"]:
            narration_str += "I start in the blue teleporter room "
            current_room = "blue teleporter room"
        elif agent_start_position in self._room_positions["green teleporter room"]:
            narration_str += "I start in the green teleporter room "
            current_room = "green teleporter room"
        elif agent_start_position in self._room_positions["purple teleporter room"]:
            narration_str += "I start in the purple teleporter room "
            current_room = "purple teleporter room"
        else:
            narration_str += "I start in the goal corridor "
            teleporter_room = False
            goal_corridor = True
            current_room = "goal corridor"
        if teleporter_room:
            teleport = False
            current_obs_index = 1
            first = True
            while current_obs_index < len(observations) and not goal_corridor:
                agent_pos = self._get_object_location(
                    observations[current_obs_index], self._OBJECT_IDS["AGENT_ID"]
                )[0]
                agent_room = self._get_room_location(agent_pos)
                if agent_room == current_room:
                    current_obs_index += 1
                    continue
                teleport = True
                colour = current_room.split(" ")[0]
                if first:
                    first_str = ""
                    first = False
                else:
                    first_str = "then "

                if colour == "purple":
                    # Need to handle the two teleportes in the purple room
                    prev_obs = observations[current_obs_index - 1]
                    prev_agent_pos = self._get_object_location(
                        prev_obs, self._OBJECT_IDS["AGENT_ID"]
                    )[0]
                    if (
                        prev_agent_pos
                        in self._teleporter_information["left purple teleporter"][
                            "reachable_from"
                        ]
                    ):
                        colour = "left purple"
                    elif (
                        prev_agent_pos
                        in self._teleporter_information["right purple teleporter"][
                            "reachable_from"
                        ]
                    ):
                        colour = "right purple "
                    else:
                        raise ValueError(
                            "Agent is in purple room, but teleported from unknown location"
                        )

                narration_str += f"and {first_str}I go through the {colour} teleporter to the {agent_room} "
                current_obs_index += 1
                current_room = agent_room
                if "goal" in agent_room:
                    goal_corridor = True
                    if current_obs_index < len(observations):
                        agent_start_position = self._get_object_location(
                            observations[current_obs_index],
                            self._OBJECT_IDS["AGENT_ID"],
                        )[0]

            if not teleport:
                narration_str += "and I will not teleport yet "
                return narration_str

        if goal_corridor:
            try:
                agent_end_position = self._get_object_location(
                    observations[-1], self._OBJECT_IDS["AGENT_ID"]
                )[0]
            except IndexError:
                narration_str += "and then I will reach the goal"
                return narration_str
            if agent_end_position == self._goal_position:
                narration_str += "and then I will reach the goal"
                return narration_str

            start_dist_to_goal = self._calculate_distance(
                agent_start_position, self._goal_position
            )
            end_dist_to_goal = self._calculate_distance(
                agent_end_position, self._goal_position
            )

            if start_dist_to_goal == end_dist_to_goal:
                narration_str += "and then I will stay the same distance from the goal "
            elif start_dist_to_goal > end_dist_to_goal:
                narration_str += "and then I will move towards the goal "
            else:
                narration_str += "and then I will move away from the goal "

        return narration_str
