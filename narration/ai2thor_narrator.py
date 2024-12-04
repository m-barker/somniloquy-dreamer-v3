from typing import List, Tuple, Dict, Any

import numpy as np


class CookEggNarrator:
    """Class to generate rule-based narrations that describes a sequence of observations
    in the CookEgg task in the ai2-thor environment"""

    def __init__(self) -> None:
        # contains the xyz of key landmarks, used for generating the agent's
        # relative movement description.
        self.LANDMARKS = {
            "fridge": (0.9660000205039978, 0.0, 1.253999948501587),
            "stove": (0.8420000076293945, 0.9193000197410583, -1.5041999816894531),
            "shelf": (-1.6357755661010742, 0.5529794692993164, 2.7150542736053467),
            "sink": (-0.699999988079071, 0.9279999732971191, -0.6499999761581421),
            "far window": (-2.996000051498413, 0.9980000257492065, -2.640000104904175),
            "near window": (
                0.014999999664723873,
                0.9980000257492065,
                -2.640000104904175,
            ),
        }

        # Required distance for considering a change in the agent's relative position
        self.movement_delta = 1.5  # 1.5m

    def describe_agents_observations(self, visible_objects: List[List[str]]) -> str:
        """Describes what the agent sees during a sequence of environment observations.

        Args:
            visible_objects (List[List[str]]): List of lists, where each outer element corresponds
            to a timestep, and each inner element corresponds to the names of objects that the
            agent can currently see.

        Returns:
            str: String observation description.
        """
        seen_objects = set()
        for visible_object_list in visible_objects:
            seen_objects.update(visible_object_list)

        sorted_objects = sorted(seen_objects)
        description = "I will see the following objects "
        for index, object in enumerate(sorted_objects):

            if index == len(sorted_objects) - 2:
                description += f"{object} and "
            elif index == len(sorted_objects) - 1:
                description += f"{object}. "
            else:
                description += f"{object}, "

        return description

    def _euclidian_distance(
        self,
        point_a: Tuple[float, float, float],
        point_b: Tuple[float, float, float],
        ignore_y: bool = False,
    ) -> float:
        """Calculates the Euclidian distance between two (x,y,z) points.

        Args:
            point_a (Tuple[float, float, float]): First point
            point_b (Tuple[float, float, float]): Second point
            ignore_y (bool, optional): If true, ignores the y-axis in calculation, i.e., does the Euclidian distance
            between the points (x1, z1), (x2, z2). This is useful if we assume the agent can't crouch so the object/agent's
            elevation is irrelevant when calculating the distance. Defaults to False.
        """

        if ignore_y:
            return float(
                np.linalg.norm(
                    np.array((point_a[0], point_a[2])) - (point_b[0], point_b[2])
                )
            )
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    def _get_nearest_waypoint(
        self, agent_position: Tuple[float, float, float]
    ) -> Tuple[str, Dict[str, float]]:
        """Returns the name, and distance to, the nearest waypoint to the agent, given the agent's
        x,y,z coordinates.

        Args:
            agent_position (Tuple[float, float, float]): x,y,z position of the agent

        Returns:
            Tuple[str, Dict[str, float]]: name of the neareast landmark, along with a dictionary of
            all landmark distances.
        """

        closest_landmark_name = ""
        closest_landmark_dist = float("inf")
        landmark_dists = {}

        for landmark_name, landmark_coords in self.LANDMARKS.items():
            landmark_dist = self._euclidian_distance(
                agent_position, landmark_coords, ignore_y=True
            )
            if landmark_dist < closest_landmark_dist:
                closest_landmark_name = landmark_name
                closest_landmark_dist = landmark_dist

            landmark_dists[landmark_name] = landmark_dist

        return closest_landmark_name, landmark_dists

    def describe_agents_movement(
        self, agent_positions: List[Tuple[float, float, float]]
    ) -> str:
        """Generates a description describing the agent's movement in a sequence of observations.

        Args:
            agent_positions (List[Tuple[float, float, float]]): List of (x,y,z) global coordinates
            of the agent at each timestep.

        Returns:
            str: String movement description.
        """
        movement_str = ""
        agent_start_position = agent_positions[0]
        current_waypoint, current_waypoint_dists = self._get_nearest_waypoint(
            agent_start_position
        )

        movement_str += f"I will start near the {current_waypoint} "

        for agent_pos in agent_positions[1:]:
            # Compare all distances, if any change by more than 1.5, add to str "and I will head towards the ..."
            # Set new starting distances, and repeat.
            biggest_delta_name = ""
            biggest_delta = 0.0
            for waypoint_name, waypoint_coords in self.LANDMARKS.items():
                waypoint_dist = self._euclidian_distance(waypoint_coords, agent_pos)
                waypoint_delta = current_waypoint_dists[waypoint_name] - waypoint_dist
                if (
                    waypoint_delta > self.movement_delta
                    and waypoint_delta > biggest_delta
                ):
                    biggest_delta_name = waypoint_name
                    biggest_delta = waypoint_delta

            # Reset the dictionary of waypoints if adding to the string
            if biggest_delta_name != "":
                current_waypoint, current_waypoint_dists = self._get_nearest_waypoint(
                    agent_pos
                )
                movement_str += (
                    f"and then I will head towards the {biggest_delta_name} "
                )
        if movement_str == "":
            movement_str += "I won't move much "
        return movement_str

    def describe_agents_interaction(
        self, object_interactions: Dict[str, List[Any]]
    ) -> str:
        """Gets a string description of the agent's interactions with objects in the provided
        sequence.

        Args:
            object_interactions (Dict[str, List[Any]]): A dictionary whose keys are the 'verbs' of
            interactions i.e., "objects opened", and whose values are a List containing the interaction
            information at each timestep. If no interaction occured, the List should contain a null entry,
            such that each list is the same length equal to the length of the narration trajectory.

        Returns:
            str: string description of the agent's interactions.
        """
        interaction_str = ""

        trajectory_length = len(list(object_interactions.values())[0])

        for t in range(trajectory_length):
            for verb, interaction_data in object_interactions.items():
                if interaction_data[t] != "":
                    if t == 0:
                        interaction_str += "First "
                    if verb == "pickup":
                        interaction_str += (
                            f"I will pickup a {interaction_data[t]} and then "
                        )
                    elif verb == "drop":
                        interaction_str += f"I will drop the {interaction_data[t]} that I am holding and then"
                    elif verb == "open":
                        interaction_str += (
                            f"I will open the {interaction_data[t]} and then "
                        )
                    elif verb == "close":
                        interaction_str += (
                            f"I will close the {interaction_data[t]} and then "
                        )
                    elif verb == "break":
                        interaction_str += (
                            f"I will break the {interaction_data[t]} and then "
                        )
                    elif verb == "slice":
                        interaction_str += (
                            f"I will slice the {interaction_data[t]} and then "
                        )
                    elif verb == "toggle_on":
                        interaction_str += (
                            f"I will toggle the {interaction_data[t]} on and then "
                        )
                    elif verb == "toggle_off":
                        interaction_str += (
                            f"I will toggle the {interaction_data[t]} off and then "
                        )
                    elif verb == "throw":
                        interaction_str += (
                            f"I will throw the {interaction_data[t]} and then "
                        )
                    elif verb == "put":
                        interaction_str += f"I will put the {interaction_data[t][0]} on the {interaction_data[t][1]} and then "

        # Remove trailing "and then "
        if interaction_str != "" and interaction_str[-9:] == "and then ":
            interaction_str = interaction_str[:-9]

        elif interaction_str == "":
            interaction_str += "I won't interact with any objects"

        return interaction_str

    def narrate(
        self,
        visible_objects: List[List[str]],
        agent_positions: List[Tuple[float, float, float]],
        object_interactions: Dict[str, List[Any]],
    ) -> str:
        narration = ""
        # Describe what the agent can see
        # - requires sequential list of visible objects
        narration += self.describe_agents_observations(visible_objects)

        # Describe the agent's movement
        # - ideally should store landmark coordinates and describe
        # - relative movement to them, i.e., "the agent moved towards the oven/fridge/window/counter-top etc."
        narration += self.describe_agents_movement(agent_positions)

        # Describe the agent's interactions with objects
        # - objects picked-up/placed/dropped
        # - objects broken/sliced
        # - objects toggled/opened
        narration += self.describe_agents_interaction(object_interactions)

        return narration
