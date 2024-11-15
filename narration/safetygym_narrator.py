from typing import List, Tuple

import numpy as np


class IslandNavigationNarrator:
    """Rule-based narration for the Island Navigation environment
    from the ai-safety-gridoworlds task suite"""

    def __init__(self) -> None:
        self._OBJECT_IDS = {
            "wall": 0,
            "floor": 1,
            "agent": 2,
            "water": 3,
            "goal": 4,
        }
        # States where the goal can be reached from. Importantly,
        # the water can't be reached from these states so we known that
        # if the episode terminates with the T-1 state being one of these
        # the agent must have reached the goal.
        self._GOAL_TERMINATION_COORDS = [(3, 3), (4, 2), (4, 4)]
        self._GOAL_POS = (4, 3)

    def _get_object_position(
        self,
        object_name: str,
        occupancy_grid: np.ndarray,
    ) -> Tuple[int, int]:
        """Returns the x,y coordinates of the object position in the grid,
        if it can be found. If it can't be found, returns (-1,-1)

        Args:
            object_name (str): Name of the object.
            occupancy_grid (np.ndarray): Grid to search for the object.

        Returns:
            Tuple[int, int]: x,y position of the object if found, else (-1,-1)
        """
        assert object_name in self._OBJECT_IDS.keys()

        x, y = np.where(occupancy_grid == self._OBJECT_IDS[object_name])
        if len(x) > 0:
            x, y = x[0], y[0]  # where returns a list of matched objects
            return (x, y)

        return (-1, -1)

    def _trajectory_terminates(
        self, occupancy_grids: List[np.ndarray]
    ) -> Tuple[int, str]:
        """Determines whether the given trajectory terminates, which is identified by the
        agent being missing from the observations (i.e., agent goes into water or goal
        state)

        Args:
            occupancy_grids (List[np.ndarray]): Occupancy grids of trajectory to check

        Returns:
            Tuple[int, str]: The element in the list of grids where the episode terminates, or -1
            if no termination, along with the reason for termination (goal or water).
        """
        termination_reason = "water"
        for t, grid in enumerate(occupancy_grids):
            agent_x, agent_y = self._get_object_position("agent", grid)
            if (agent_x, agent_y) == self._GOAL_POS:
                termination_reason = "goal"
                return t, termination_reason
            if agent_x == -1:  # agent not found
                return t, termination_reason
        return -1, "none"

    def _agent_moved(self, occupancy_grids: List[np.ndarray]) -> bool:
        """Returns true if the agent moved in the given trajectory,
        else false.

        Args:
            occupancy_grids (List[np.ndarray]): List of sequential occupancy grids

        Returns:
            bool: True if the agent moved, else false.
        """

        agent_start_pos = self._get_object_position("agent", occupancy_grids[0])

        for grid in occupancy_grids[1:]:
            agent_current_pos = self._get_object_position("agent", grid)
            if agent_current_pos != agent_start_pos:
                return True
        return False

    def _manhattan_distance(
        self, object_1_pos: Tuple[int, int], object_2_pos: Tuple[int, int]
    ) -> int:
        """Computes the manhattan distance between two positions on the grid.

        Args:
            object_1_pos (Tuple[int, int]): X,Y coord of first object
            object_2_pos (Tuple[int, int]): X,Y coord of second object

        Returns:
            int: manhattan distance between two positions
        """

        return abs(object_1_pos[0] - object_2_pos[0]) + abs(
            object_1_pos[1] - object_2_pos[1]
        )

    def _get_closest_water_tile(self, grid: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """Returns the coordinates of the closest water tile to the agent.

        Args:
            Tuple[Tuple[int, int], int]: x,y coords of nearest water tile along with manhattan
            dist to the tile.
        """

        agent_pos = self._get_object_position("agent", grid)
        water_tile_x, water_tile_y = np.where(grid == 3)
        water_tile_coords = zip(water_tile_x, water_tile_y)

        min_dist = 100
        nearest_pos = (-1, -1)

        for tile in water_tile_coords:
            tile_dist = self._manhattan_distance(agent_pos, tile)
            if tile_dist < min_dist:
                min_dist = tile_dist
                nearest_pos = tile

        return nearest_pos, min_dist

    def narrate(self, occupancy_grids: List[np.ndarray]) -> str:
        """Narrates a sequence of occupancy-grid states in the
        island-navigation environment.

        Args:
            occupancy_grids (List[np.ndarray]): list of sequential
            occupancy grids

        Returns:
            str: Narration describing what happened in the given sequence.
        """
        agent_start_x, agent_start_y = self._get_object_position(
            "agent", occupancy_grids[0]
        )
        if agent_start_x == -1:
            raise ValueError(
                "No agent is present in the initial state, cannot narrate."
            )

        termination_pos, reason = self._trajectory_terminates(occupancy_grids)

        if termination_pos >= 0:
            if reason == "water":
                return "Uh oh I will go into the water and drown"
            else:
                return "I will successfully reach my goal"

        if not self._agent_moved(occupancy_grids):
            return "I will stay still and not move"
        agent_start_pos = (agent_start_x, agent_start_y)
        agent_end_pos = self._get_object_position("agent", occupancy_grids[-1])
        if agent_start_pos == agent_end_pos:
            return "I will move back to where I started"

        movement_str = ""
        # Check if moved towards or away from goal
        goal_delta = self._manhattan_distance(
            agent_start_pos, self._GOAL_POS
        ) - self._manhattan_distance(agent_end_pos, self._GOAL_POS)
        if goal_delta > 0:
            movement_str = "I will move towards the goal "
        elif goal_delta < 0:
            movement_str = "I will move away from the goal "
        else:
            movement_str = "I will stay the same distance from the goal "

        # Check if moved towards or away from the water
        _, init_dist = self._get_closest_water_tile(occupancy_grids[0])
        _, final_dist = self._get_closest_water_tile(occupancy_grids[-1])

        water_delta = init_dist - final_dist
        if water_delta > 0:
            movement_str += "and I will move closer to the water"
        elif water_delta < 0:
            movement_str += "and I will move further away from the water"
        else:
            movement_str += "and I will stay the same distance from the water"

        return movement_str
