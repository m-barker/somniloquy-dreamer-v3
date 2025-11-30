from typing import List
import numpy as np


class SafetyGymnasiumNavigationNarrator:
    def __init__(self) -> None:
        pass

    def _2d_eucl_arr_dist(self, point_1: np.ndarray, point_2: np.ndarray) -> float:
        """
        Returns the euclidian distance between the x,y coords of two points.
        """

        point_1_2d = point_1[:2]
        point_2_2d = point_2[:2]

        return np.linalg.norm(point_1_2d - point_2_2d)

    def _get_static_dist_str(
        self,
        start: bool,
        agent_position: np.ndarray,
        goal_position: np.ndarray,
        hazard_position: List[np.ndarray],
        vase_position: List[np.ndarray],
    ) -> str:
        """
        Gets the object the agent is closest to and returns the string of the form
        "I will start nearest the/a goal/vase/hazard"
        """

        closest_dist = np.inf
        closest_obj = "none"

        if closest_obj == "none":
            raise ValueError("No closest object found")

        # All the distance calculations can be vectorised to make this faster
        # if needed. I.e., get all points in an array and use np.linalg.norm
        # on the whole array

        goal_dist = self._2d_eucl_arr_dist(agent_position, goal_position)
        if goal_dist < closest_dist:
            closest_obj = "goal"
            closest_dist = goal_dist

        for hazard in hazard_position:
            dist = self._2d_eucl_arr_dist(agent_position, hazard)
            if dist < closest_dist:
                closest_obj = "hazard"
                closest_dist = dist
        for vase in vase_position:
            dist = self._2d_eucl_arr_dist(agent_position, vase)
            if dist < closest_dist:
                closest_obj = "vase"
                closest_dist = dist

        identifier = "the" if closest_obj == "goal" else "a"
        time = "start" if start else "end"

        return f"I will {time} nearest {identifier} {closest_obj}. "

    def _get_n_goals_reached(self, goal_positions: List[np.ndarray]) -> int:
        """
        Gets the number of goals reached, by checking the number of times the
        goal moves positions
        """

        goal_pos = goal_positions[0]
        n_goals_reached = 0
        for g in goal_positions[1:]:
            if not np.array_equal(g, goal_pos):
                n_goals_reached += 1
                goal_pos = g

        return n_goals_reached

    def _get_goal_reached_str(self, goal_positions: List[np.ndarray]) -> str:
        """
        Gets a string of the form "I will reach x goals" or "I will not reach the goal"
        """

        n_goals_reached = self._get_n_goals_reached(goal_positions)

        if n_goals_reached == 0:
            return "I will not reach any goals. "
        return f"I will reach {n_goals_reached} goals. "

    def _get_goal_distance_str(
        self, agent_positions: List[np.ndarray], goal_positions: List[np.ndarray]
    ) -> str:
        """
        Gets the agent's movement string relative to the last goal that has not been reached.
        Returns a string of the form "My distance to the [new] goal will increase/decrease/stay
        the same"
        """

        # First, get the index of last goal to not be reached
        last_goal_index = 0
        goal_pos = goal_positions[0]
        for index, g in enumerate(goal_positions[1:]):
            if not np.array_equal(goal_pos, g):
                last_goal_index = index
                goal_pos = g

        agent_positions = agent_positions[last_goal_index:]

        agent_start_dist = self._2d_eucl_arr_dist(agent_positions[0], goal_pos)
        agent_end_dist = self._2d_eucl_arr_dist(agent_positions[-1], goal_pos)

        new = goal_pos != 0
        new_str = " new" if new else ""

        if agent_start_dist > agent_end_dist:
            return f"My distance to the{new_str} goal will decrease. "
        elif agent_start_dist < agent_end_dist:
            return f"My distance to the{new_str} goal will increase. "
        else:
            return f"My distance to the{new_str} goal will stay the same. "

    def _get_cost_string(
        self,
        hazard_costs: List[float],
        vase_contact_cost: List[float],
        vase_velocity_cost: List[float],
    ) -> str:
        """
        Gets the cost string that describes the agent entering/exiting hazard zones, colliding/
        moving/stop colliding with vases.
        """

        cost_str = ""

        currently_in_hazard = False
        first = True
        for hc in hazard_costs:
            if not currently_in_hazard:
                if hc != 0:
                    if first:
                        cost_str += (
                            "First I will enter into a hazard and incur some cost. "
                        )
                    else:
                        cost_str += (
                            "Then I will enter into a hazard and incur some cost. "
                        )
                    currently_in_hazard = True
            else:
                if hc == 0:
                    if first:
                        cost_str += "First I will exit the hazard. "
                    else:
                        cost_str += "Then I will exit the hazard. "
                    currently_in_hazard = False
        if currently_in_hazard:
            cost_str += "I will end in a hazard. "

        currently_colliding_with_vase = False
        currently_displacing_vase = False
        first = True

        for vase_collision_cost, vase_displacement_cost in zip(
            vase_contact_cost, vase_velocity_cost
        ):
            if not currently_colliding_with_vase:
                if vase_collision_cost != 0:
                    if first:
                        cost_str += (
                            "First I will collide with a vase and incur some cost. "
                        )
                    else:
                        cost_str += (
                            "Then I will collide with a vase and incur some cost. "
                        )
                    currently_colliding_with_vase = True
            else:
                if vase_collision_cost == 0:
                    if first:
                        cost_str += "First I will stop colliding with the vase. "
                    else:
                        cost_str += "Then I will stop colliding with the vase. "
                    currently_colliding_with_vase = False
        if currently_colliding_with_vase:
            cost_str += "I will end in collision with a vase. "

        return cost_str

    def narrate(
        self,
        agent_positions: List[np.ndarray],
        goal_positions: List[np.ndarray],
        hazard_positions: List[List[np.ndarray]],
        vase_positions: List[List[np.ndarray]],
        hazard_costs: List[float],
        vase_contact_costs: List[float],
        vase_velocity_costs: List[float],
    ) -> str:
        narration_str = ""
        narration_str += self._get_static_dist_str(
            True,
            agent_positions[0],
            goal_positions[0],
            hazard_positions[0],
            vase_positions[0],
        )
        narration_str += self._get_goal_reached_str(goal_positions)
        narration_str += self._get_goal_distance_str(agent_positions, goal_positions)
        narration_str += self._get_cost_string(
            hazard_costs, vase_contact_costs, vase_velocity_costs
        )
        narration_str += self._get_static_dist_str(
            False,
            agent_positions[-1],
            goal_positions[-1],
            hazard_positions[-1],
            vase_positions[-1],
        )

        return narration_str
