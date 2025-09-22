from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import numpy as np


class MiniGridNarrator(ABC):
    def __init__(self) -> None:
        super().__init__()

        # Object ID is channel 0 of the environment observation
        self._OBJECT_IDS = {
            "FLOOR_ID": 3,
            "DOOR_ID": 4,
            "KEY_ID": 5,
            "BALL_ID": 6,
            "BOX_ID": 7,
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
    def narrate(self, observations: List[Any]) -> str:
        pass


class BabyAIGoToLocNarrator(MiniGridNarrator):
    """
    Narrator for the GoToLoc environment, details of which can be found
    https://minigrid.farama.org/environments/babyai/GoToLocal/
    """

    def _get_adjacent_coordinates(
        self, source_coordinate: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Gets a list of coordinates that are adjacent (not diagonal) to a given source coordinate

        Arguments:
            source_coordinate (Tuple(int, int)): x, y of coord to get adjacent cells

        Returns List[Tuple[int, int]] - list of adjacent coordinates
        """

        adjacent_coords: List[Tuple[int, int]] = []

        # Given coord X we want to return Y
        # |-|Y|-|
        # |Y|X|Y|
        # |-|Y|-|

        source_x, source_y = source_coordinate

        adjacent_coords.append((source_x + 1, source_y))
        adjacent_coords.append((source_x - 1, source_y))
        adjacent_coords.append((source_x, source_y + 1))
        adjacent_coords.append((source_x, source_y - 1))

        return adjacent_coords

    def _is_agent_facing_object(
        self,
        agent_position: Tuple[int, int],
        agent_direction: int,
        object_position: Tuple[int, int],
    ) -> bool:
        """
        Determines if the agent is facing (i.e., pointing at) a given object.

        Arguments:
            - agent_position Tuple[int, int]: (x,y) coord of agent
            - agent_direction int: integer encoding direction of agent in range [0,3]
            - object_position Tuple[int, int]: (x,y) coord of object to check

        Returns bool: True if agent is facing the object else false
        """
        agent_facing = False

        # Grid coords are:
        #   0 1 2 3
        # 0| | | | |
        # 1| | | | |
        # 2| | | | |
        # 3| | | | |

        agent_x, agent_y = agent_position
        object_x, object_y = object_position

        if agent_y == object_y:
            # Agent to the left of object
            if agent_x == object_x - 1:
                agent_facing = agent_direction == 0
            # Agent to the right of object
            elif agent_x == object_x + 1:
                agent_facing = agent_direction == 2
        elif agent_x == object_x:
            # Agent above object
            if agent_y == object_y - 1:
                agent_facing = agent_direction == 1
            # Agent below object
            elif agent_y == object_y + 1:
                agent_facing = agent_direction == 3

        print(f"Agent position: {agent_position}")
        print(f"Object position: {object_position}")
        print(f"Agent direction: {agent_direction}")
        print(f"Is agent facing object?: {agent_facing}")

        return agent_facing

    def narrate(self, observations: List[Dict[str, Union[np.ndarray, int]]]) -> str:
        """
        Describes what happened in a given sequence of state observations
        in the minigrid GoToLoc environment.

        observations: Dict containing the following keys:
            -   occupancy_grid: np.ndarray of the envs occupancy grid
            -   agent_direction: int, denotaing the direction the agent is facing.
                -    0 = pointing right
                -    1 = pointing down
                -    2 = pointing left
                -    3 = pointing up

        Returns a string.

        Environment details:

        “go to the {color} {type}”
        {color} is the color of the box. Can be “red”, “green”, “blue”, “purple”, “yellow” or “grey”.
        {type} is the type of the object. Can be “ball”, “box” or “key”.

        Environment terminates when the agent is facing the request object, not just when the
        agent is next to it. We therefore need to take into account the agent's orientation,
        and distinguish from being next to an object to "reaching" i.e., facing, the object

        """
        narration_str = ""

        occupancy_grids: List[np.ndarray] = [
            x["occupancy_grid"]
            for x in observations
            if isinstance(x["occupancy_grid"], np.ndarray)
        ]
        agent_directions: List[int] = [
            x["agent_direction"]
            for x in observations
            if isinstance(x["agent_direction"], int)
        ]

        # Check if agent ever moves
        agent_start_pos = self._get_object_location(
            occupancy_grids[0], self._OBJECT_IDS["AGENT_ID"]
        )
        if not agent_start_pos:
            print(occupancy_grids[0])
            raise ValueError("Agent not found in grid")
        agent_start_pos = agent_start_pos[0]
        agent_start_dir = agent_directions[0]
        agent_moved = False
        for i, grid in enumerate(occupancy_grids):
            agent_pos = self._get_object_location(grid, self._OBJECT_IDS["AGENT_ID"])
            agent_dir = agent_directions[i]
            if not agent_pos:
                raise ValueError("Agent not found in grid")
            agent_pos = agent_pos[0]
            if agent_pos != agent_start_pos or agent_dir != agent_start_dir:
                agent_moved = True
                break
        if not agent_moved:
            return "I will not move"

        first_obs = occupancy_grids[0]
        ball_positions = self._get_object_location(
            first_obs, self._OBJECT_IDS["BALL_ID"]
        )
        box_positions = self._get_object_location(first_obs, self._OBJECT_IDS["BOX_ID"])
        key_positions = self._get_object_location(first_obs, self._OBJECT_IDS["KEY_ID"])

        ball_adjacent_cells: List[List[Tuple[int, int]]] = [[] for _ in ball_positions]
        box_adjacent_cells: List[List[Tuple[int, int]]] = [[] for _ in box_positions]
        key_adjacent_cells: List[List[Tuple[int, int]]] = [[] for _ in key_positions]

        for i, ball in enumerate(ball_positions):
            ball_adjacent_cells[i] = self._get_adjacent_coordinates(ball)
        for i, box in enumerate(box_positions):
            box_adjacent_cells[i] = self._get_adjacent_coordinates(box)
        for i, key in enumerate(key_positions):
            key_adjacent_cells[i] = self._get_adjacent_coordinates(key)

        prev_pos = None
        prev_direction = None
        for t, state in enumerate(occupancy_grids):
            agent_position = self._get_object_location(
                state, self._OBJECT_IDS["AGENT_ID"]
            )
            if not agent_position:
                raise ValueError("Agent cannot be found in the current state")
            # Should only ever be one agent
            agent_position = agent_position[0]
            agent_direction = agent_directions[t]

            if prev_pos is not None and prev_direction is not None:
                # Skip if no movement, otherwise we just narrate the same thing
                if agent_position == prev_pos and agent_direction == prev_direction:
                    continue

            narrated_this_timestep = False

            for i, ball in enumerate(ball_adjacent_cells):
                # Each tile is encoded as the tuple (OBJECT_IDX, COLOR_IDX, STATE)
                ball_x, ball_y = ball_positions[i]
                ball_tile_encoding = state[ball_x, ball_y]
                ball_colour_id = ball_tile_encoding[1]
                ball_colour_str = self._ID_TO_COLOUR[ball_colour_id]

                for adjacent_cell in ball:
                    if agent_position == adjacent_cell:
                        # Check if agent is facing the object
                        facing = self._is_agent_facing_object(
                            agent_position, agent_direction, ball_positions[i]
                        )
                        if narration_str == "":
                            narration_str += "First "
                        if facing:
                            if narrated_this_timestep:
                                narration_str += "and "
                            narration_str += f"I will go to the {ball_colour_str} ball "
                        else:
                            if narrated_this_timestep:
                                narration_str += f"and I will move next to the {ball_colour_str} ball "
                            else:
                                narration_str += (
                                    f"I will move next to the {ball_colour_str} ball "
                                )
                        narrated_this_timestep = True

            for i, box in enumerate(box_adjacent_cells):
                # Each tile is encoded as the tuple (OBJECT_IDX, COLOR_IDX, STATE)
                box_x, box_y = box_positions[i]
                box_tile_encoding = state[box_x, box_y]
                box_colour_id = box_tile_encoding[1]
                box_colour_str = self._ID_TO_COLOUR[box_colour_id]

                for adjacent_cell in box:
                    if agent_position == adjacent_cell:
                        # Check if agent is facing the object
                        facing = self._is_agent_facing_object(
                            agent_position, agent_direction, box_positions[i]
                        )
                        if narration_str == "":
                            narration_str += "First "
                        if facing:
                            if narrated_this_timestep:
                                narration_str += "and "
                            narration_str += f"I will go to the {box_colour_str} box "
                        else:
                            if narrated_this_timestep:
                                narration_str += (
                                    f"and I will move next to the {box_colour_str} box "
                                )
                            else:
                                narration_str += (
                                    f"I will move next to the {box_colour_str} box "
                                )
                        narrated_this_timestep = True

            for i, key in enumerate(key_adjacent_cells):
                # Each tile is encoded as the tuple (OBJECT_IDX, COLOR_IDX, STATE)
                key_x, key_y = key_positions[i]
                key_tile_encoding = state[key_x, key_y]
                key_colour_id = key_tile_encoding[1]
                key_colour_str = self._ID_TO_COLOUR[key_colour_id]

                for adjacent_cell in key:
                    if agent_position == adjacent_cell:
                        # Check if agent is facing the object
                        facing = self._is_agent_facing_object(
                            agent_position, agent_direction, key_positions[i]
                        )
                        if narration_str == "":
                            narration_str += "First "
                        if facing:
                            if narrated_this_timestep:
                                narration_str += "and "
                            narration_str += f"I will go to the {key_colour_str} key "
                        else:
                            if narrated_this_timestep:
                                narration_str += (
                                    f"and I will move next to the {key_colour_str} key "
                                )
                            else:
                                narration_str += (
                                    f"I will move next to the {key_colour_str} key "
                                )
                        narrated_this_timestep = True

            if narrated_this_timestep:
                narration_str += "and then "

            prev_pos = agent_position
            prev_direction = agent_direction

        # If agent has moved, but hasn't moved next to any objects, then get the object(s) the agent started
        # closest to, and compare with the closest object(s) to the agent at the end of the trajectory
        if narration_str == "":
            min_dist = np.inf
            closest_objects: List[Tuple[str, Union[int, float]]] = []
            for ball in ball_positions:
                agent_dist = self._calculate_distance(agent_start_pos, ball)
                if agent_dist <= min_dist:
                    ball_x, ball_y = ball
                    ball_tile_encoding = occupancy_grids[0][ball_x, ball_y]
                    ball_colour_id = ball_tile_encoding[1]
                    ball_colour_str = self._ID_TO_COLOUR[ball_colour_id]
                    closest_objects.append((f"{ball_colour_str} ball", agent_dist))
                    min_dist = agent_dist
            for box in box_positions:
                agent_dist = self._calculate_distance(agent_start_pos, box)
                if agent_dist <= min_dist:
                    box_x, box_y = box
                    box_tile_encoding = occupancy_grids[0][box_x, box_y]
                    box_colour_id = box_tile_encoding[1]
                    box_colour_str = self._ID_TO_COLOUR[box_colour_id]
                    closest_objects.append((f"{box_colour_str} box", agent_dist))
                    min_dist = agent_dist
            for key in key_positions:
                agent_dist = self._calculate_distance(agent_start_pos, key)
                if agent_dist <= min_dist:
                    key_x, key_y = key
                    key_tile_encoding = occupancy_grids[0][key_x, key_y]
                    key_colour_id = key_tile_encoding[1]
                    key_colour_str = self._ID_TO_COLOUR[key_colour_id]
                    closest_objects.append((f"{key_colour_str} key", agent_dist))
                    min_dist = agent_dist

            closest_objects.sort(key=lambda x: x[1])
            closest_dist = closest_objects[0][1]
            narration_str += f"I will start closest to the {closest_objects[0][0]} "
            for obj in closest_objects[1:]:
                if obj[1] > closest_dist:
                    break
                narration_str += f"and the {obj[0]} "

            min_dist = np.inf
            closest_objects: List[Tuple[str, Union[int, float]]] = []
            agent_end_pos = self._get_object_location(
                occupancy_grids[-1], self._OBJECT_IDS["AGENT_ID"]
            )
            if not agent_end_pos:
                raise ValueError("Agent could not be found in the final observation")
            agent_end_pos = agent_end_pos[0]
            for ball in ball_positions:
                agent_dist = self._calculate_distance(agent_end_pos, ball)
                if agent_dist <= min_dist:
                    ball_x, ball_y = ball
                    ball_tile_encoding = occupancy_grids[0][ball_x, ball_y]
                    ball_colour_id = ball_tile_encoding[1]
                    ball_colour_str = self._ID_TO_COLOUR[ball_colour_id]
                    closest_objects.append((f"{ball_colour_str} ball", agent_dist))
                    min_dist = agent_dist
            for box in box_positions:
                agent_dist = self._calculate_distance(agent_end_pos, box)
                if agent_dist <= min_dist:
                    box_x, box_y = box
                    box_tile_encoding = occupancy_grids[0][box_x, box_y]
                    box_colour_id = box_tile_encoding[1]
                    box_colour_str = self._ID_TO_COLOUR[box_colour_id]
                    closest_objects.append((f"{box_colour_str} box", agent_dist))
                    min_dist = agent_dist
            for key in key_positions:
                agent_dist = self._calculate_distance(agent_end_pos, key)
                if agent_dist <= min_dist:
                    key_x, key_y = key
                    key_tile_encoding = occupancy_grids[0][key_x, key_y]
                    key_colour_id = key_tile_encoding[1]
                    key_colour_str = self._ID_TO_COLOUR[key_colour_id]
                    closest_objects.append((f"{key_colour_str} key", agent_dist))
                    min_dist = agent_dist

            closest_objects.sort(key=lambda x: x[1])
            closest_dist = closest_objects[0][1]
            narration_str += f"and I will end closest to the {closest_objects[0][0]} "
            for obj in closest_objects[1:]:
                if obj[1] > closest_dist:
                    break
                narration_str += f"and the {obj[0]} "

        # Strip off any trailing " and then "
        if narration_str[-5:] == "then ":
            narration_str = narration_str[:-9]

        # Remove any trailing whitespace
        if narration_str[-1] == " ":
            narration_str = narration_str[:-1]

        assert narration_str != "", "Empty narration str for BabyAIGoToLoc Narrator"

        return narration_str


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


class MiniGridFourSquareExplNarrator(MiniGridNarrator):
    def narrate(self, observations: list[np.ndarray]) -> str:
        first_obs = observations[0]
        if not self._agent_moved(observations):
            return "i will not move "
        if (
            self._get_object_location(observations[-1], self._OBJECT_IDS["AGENT_ID"])[0]
            == self._get_object_location(observations[0], self._OBJECT_IDS["AGENT_ID"])[
                0
            ]
        ):
            return "i will move in a circle "

        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        agent_end_position = self._get_object_location(
            observations[-1], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        coloured_square_positions = self._get_object_location(
            observations[0], self._OBJECT_IDS["FLOOR_ID"]
        )

        biggest_delta = 0.0
        closest_square = None

        for square_position in coloured_square_positions:
            square_colour = self._ID_TO_COLOUR[
                first_obs[square_position[0], square_position[1], 1]
            ]
            if agent_end_position == square_position:
                return f"i will reach the {square_colour} square"
            delta = self._calculate_distance(
                square_position, agent_start_position
            ) - self._calculate_distance(square_position, agent_end_position)
            if delta > biggest_delta:
                biggest_delta = delta
                closest_square = square_colour

        return f"i will move towards the {closest_square} square"


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
                observations,
                key_position,
                "key",  # type: ignore
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
            first_obs,
            last_obs,
            door_position,  # type: ignore
        )
        if door_locked:
            narration_str += self._get_agent_relative_movement_string(
                observations,
                door_position,
                "door",  # type: ignore
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
                observations,
                door_position,  # type: ignore
            )
            narration_str += door_open_close_sequence
            if door_open_close_sequence != "":
                narration_str += "and then "
                door_last_change_frame = self._get_last_door_change_frame(
                    observations,
                    door_position,  # type: ignore
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
                observations,
                goal_position,
                "goal ",  # type: ignore
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
