from typing import List, Tuple, Dict, Iterable

import numpy as np


class CookEggNarrator:
    """Class to generate rule-based narrations that describes a sequence of observations
    in the CookEgg task in the ai2-thor environment"""

    def __init__(self) -> None:
        # contains the xyz of key landmarks, used for generating the agent's
        # relative movement description.
        self.LANDMARKS = {"fridge": (1, 2, 3)}

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
        return ""

    def describe_agents_interaction(self) -> str:
        return ""

    def narrate(self) -> str:
        narration = ""
        # Describe what the agent can see
        # - requires sequential list of visible objects
        narration += self.describe_agents_observations()

        # Describe the agent's movement
        # - ideally should store landmark coordinates and describe
        # - relative movement to them, i.e., "the agent moved towards the oven/fridge/window/counter-top etc."
        narration += self.describe_agents_movement()

        # Describe the agent's interactions with objects
        # - objects picked-up/placed/dropped
        # - objects broken/sliced
        # - objects toggled/opened
        narration += self.describe_agents_interaction()

        return narration
