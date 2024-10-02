from typing import List, Dict, Union

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
import crafter


class CrafterNarrator:
    def __init__(self):
        self.OBJECT_IDS = {
            "none": 0,
            "water": 1,
            "grass": 2,
            "stone": 3,
            "path": 4,
            "sand": 5,
            "tree": 6,
            "lava": 7,
            "coal": 8,
            "iron": 9,
            "diamond": 10,
            "table": 11,
            "furnace": 12,
            # "player": 13,
            "cow": 14,
            "zombie": 15,
            "skeleton": 16,
            "arrow": 17,
            "plant": 18,
        }
        self._material_names = [
            "sapling",
            "wood",
            "coal",
            "stone",
            "iron",
            "diamond",
        ]

        self._crafting_names = [
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "wood_sword",
            "stone_sword",
            "iron_sword",
        ]

        self._vitals_names = ["health", "food", "drink", "energy"]

    def _get_seen_objects(self, observations: List[np.ndarray]) -> str:
        """Gets a list of all the objects that the player has seen during a sequence
        of observations.

        Args:
            observations (List[np.ndarray]): List of occupancy-grid observations

        Returns:
            str: The objects seen by the player.
        """
        object_ids_seen = set()
        for obs in observations:
            for row in obs:
                for object_id in row:
                    object_ids_seen.add(object_id)
        objects_seen = [k for k, v in self.OBJECT_IDS.items() if v in object_ids_seen]
        # sort the objects alphabetically
        objects_seen.sort()

        object_str = "I will see "
        for i, obj in enumerate(objects_seen):
            if i == len(objects_seen) - 1:
                object_str += f"and {obj}."
            else:
                object_str += f"{obj}, "
        return object_str

    def _get_harvested_str(self, inventory_history: List[Dict[str, int]]) -> str:
        """Gets a string describing what the player has harvested.

        Args:
            inventory_history: List of the player's inventory at each timestep.

        Returns:
            str: A string describing what the player has harvested
        """
        harvested_str = "I will harvest "
        any_harvested = False
        starting_counts = {k: inventory_history[0].get(k) for k in self._material_names}

        for i, material in enumerate(self._material_names):
            harvest_count = inventory_history[-1][material] - starting_counts[material]  # type: ignore
            if harvest_count > 0:
                harvested_str += f"{harvest_count} {material}, "
                any_harvested = True
        harvested_str = harvested_str[:-2] + "."

        if not any_harvested:
            harvested_str = "I will not harvest anything."
        return harvested_str

    def _get_crafted_str(self, inventory_history: List[Dict[str, int]]) -> str:
        """Gets a string describing what the player has crafted, if anything.

        Args:
            inventory_history (List[Dict[str, int]]): List of the player's inventory at
            each timestep.

        Returns:
            str: A string describing what the player has crafted
        """

        crafting_str = ""
        starting_counts = {k: inventory_history[0].get(k) for k in self._crafting_names}

        for i, crafting_item in enumerate(self._crafting_names):
            crafted_count = (
                inventory_history[-1][crafting_item]  # type: ignore
                - starting_counts[crafting_item]  # type: ignore
            )
            if crafted_count > 0:
                crafting_str += f"I will craft {crafted_count} {crafting_item}."

        if crafting_str == "":
            crafting_str = "I will not craft anything."

        return crafting_str

    def _get_vitals_str(
        self,
        inventory_history: List[Dict[str, int]],
        occupancy_grid_history: List[np.ndarray],
    ) -> str:
        """Gets a string summary describing the players health, thirst, and energy levels.

        Args:
            inventory_history (List[Dict[str, int]]): List of the player's inventory at
            each timestep. The inventory includes the vitals information.

            occupancy_grid_history (List[np.ndarray]): List of occupancy-grid observations from the player's perspective.
            Used to determine if the player has been attacked by a zombie or a skeleton's arrow.

        Returns:
            str: A string describing the player's vitals.
        """

        vitals_str = ""

        health_regen_count = 0
        hunger_regen_count = 0
        thirst_regen_count = 0
        energy_regen_count = 0

        zombie_attack = False
        skeleton_attack = False

        for t in range(1, len(inventory_history)):
            for i, vitals in enumerate(self._vitals_names):
                vitals_change = (
                    inventory_history[t][vitals] - inventory_history[t - 1][vitals]
                )
                if vitals_change > 0:
                    if vitals == "health":
                        health_regen_count += 1
                    elif vitals == "food":
                        hunger_regen_count += 1
                    elif vitals == "drink":
                        thirst_regen_count += 1
                    elif vitals == "energy":
                        energy_regen_count += 1
                elif vitals_change < 0 and vitals == "health":
                    if vitals_change < 1:
                        # Player has been damaged by a zombie or a skeleton's arrow.
                        if vitals_change < -6 and not zombie_attack:
                            vitals_str += "I will be attacked and hit by a zombie whilst I am sleeping. "
                            zombie_attack = True
                        elif not zombie_attack and not skeleton_attack:
                            # Both the zombie and skeleton's arrow deal 2 damage, so need to figure out which one hit the player.
                            player_x, player_y = np.where(
                                occupancy_grid_history[t - 1] == 13
                            )
                            try:
                                zombie_x, zombie_y = np.where(
                                    occupancy_grid_history[t - 1]
                                    == self.OBJECT_IDS["zombie"]
                                )
                            except:
                                zombie_x, zombie_y = float("inf"), float("inf")
                            try:
                                arrow_x, arrow_y = np.where(
                                    occupancy_grid_history[t - 1]
                                    == self.OBJECT_IDS["arrow"]
                                )
                            except:
                                arrow_x, arrow_y = float("inf"), float("inf")

                            check_arrow = True
                            for zx, zy in zip(zombie_x, zombie_y):
                                if (
                                    np.abs(player_x - zx) < 2
                                    and np.abs(player_y - zy) < 2
                                ):
                                    vitals_str += (
                                        "I will be attacked and hit by a zombie. "
                                    )
                                    zombie_attack = True
                                    check_arrow = False
                                    break
                            if check_arrow:
                                for ax, ay in zip(arrow_x, arrow_y):
                                    if (
                                        np.abs(player_x - ax) < 2
                                        and np.abs(player_y - ay) < 2
                                    ):
                                        vitals_str += "I will be attacked and hit by a skeleton's arrow. "
                                        skeleton_attack = True
                                        break
                            elif not zombie_attack and not skeleton_attack:
                                raise ValueError(
                                    "Player has been attacked by an unknown entity."
                                )

        starving = any(d.get("food") == 0 for d in inventory_history)
        dehydrated = any(d.get("drink") == 0 for d in inventory_history)
        exhausted = any(d.get("energy") == 0 for d in inventory_history)

        dead = any(d.get("health") == 0 for d in inventory_history)
        if dead:
            return vitals_str + "My hitpoints will reach zero and I will die."

        # Need to try to figure out what causes the player's health to decrease.

        if health_regen_count > 0:
            vitals_str += f"My health will regenerate {health_regen_count} times. "
        if hunger_regen_count > 0:
            vitals_str += (
                f"I will eat {hunger_regen_count} cows and refill some hunger. "
            )
        else:
            vitals_str += "I will not eat anything and get more hungry. "
        if thirst_regen_count > 0:
            vitals_str += (
                f"I will drink {thirst_regen_count} waters and refill some thirst. "
            )
        else:
            vitals_str += "I will not drink anything and get more thirsty. "
        if energy_regen_count > 0:
            vitals_str += (
                f"I will sleep {energy_regen_count} times and refill some energy. "
            )
        else:
            vitals_str += "I will not sleep and get more tired. "

        if starving:
            vitals_str += (
                "My hunger will reach zero and I will begin to starve and lose health. "
            )
        if dehydrated:
            vitals_str += "My thirst will reach zero and I will begin to dehydrate and lose health. "
        if exhausted:
            vitals_str += "My energy will reach zero and I will become exhausted and lose health. "

        return vitals_str

    def _get_achievement_str(
        self, achievement_obs_history: List[Dict[str, int]]
    ) -> str:
        """Generates a string describing the achievements that the player earned during the
        given sequence of observations.

        Args:
            achievement_obs_history (List[Dict[str, int]]): List of the player's achievement
            counts at each timestep.

        Returns:
            str: A string describing the achievements that the player earned.
        """

        achievement_str = ""

        defeated_skeleton_count = 0
        defeated_zombie_count = 0
        cow_eaten_count = 0
        plant_eaten_count = 0
        placed_table_count = 0
        placed_furnace_count = 0
        placed_plant_count = 0
        placed_stone_count = 0

        for t in range(1, len(achievement_obs_history)):
            for i, achievement in enumerate(achievement_obs_history[t]):
                achievement_change = (
                    achievement_obs_history[t][achievement]
                    - achievement_obs_history[t - 1][achievement]
                )
                if achievement_change > 0:
                    if achievement == "defeat_skeleton":
                        defeated_skeleton_count += 1
                    elif achievement == "defeat_zombie":
                        defeated_zombie_count += 1
                    elif achievement == "eat_cow":
                        cow_eaten_count += 1
                    elif achievement == "eat_plant":
                        plant_eaten_count += 1
                    elif achievement == "place_table":
                        placed_table_count += 1
                    elif achievement == "place_furnace":
                        placed_furnace_count += 1
                    elif achievement == "place_plant":
                        placed_plant_count += 1
                    elif achievement == "place_stone":
                        placed_stone_count += 1

        if defeated_skeleton_count > 0:
            achievement_str += (
                f"I will fight and kill {defeated_skeleton_count} skeletons. "
            )
        if defeated_zombie_count > 0:
            achievement_str += (
                f"I will fight and kill {defeated_zombie_count} zombies. "
            )
        if cow_eaten_count > 0:
            achievement_str += f"I will eat {cow_eaten_count} cows. "
        if plant_eaten_count > 0:
            achievement_str += f"I will eat {plant_eaten_count} plants. "
        if placed_table_count > 0:
            achievement_str += f"I will place {placed_table_count} tables. "
        if placed_furnace_count > 0:
            achievement_str += f"I will place {placed_furnace_count} furnaces. "
        if placed_plant_count > 0:
            achievement_str += f"I will place {placed_plant_count} plants. "
        if placed_stone_count > 0:
            achievement_str += f"I will place {placed_stone_count} stones. "

        return achievement_str

    def narrate(
        self, observations: List[Dict[str, Union[np.ndarray, Dict[str, int]]]]
    ) -> str:
        """Converts a sequence of player observations into a string describing
        what has happened in the sequence.

        Args:
            occupancy_observations (List[np.ndarray]): List of occupancy-grid observations
            from the player's perspective.

            player_inventory_obs (List[Dict[str, int]]): List of the player inventory at
            each timestep.

        Returns:
            str: Textual description of the player's observations
        """

        occupancy_observations = [obs["semantic"] for obs in observations]
        player_inventory_obs = [obs["inventory"] for obs in observations]
        achievement_obs = [obs["achievements"] for obs in observations]

        narration_str = ""
        narration_str += self._get_seen_objects(occupancy_observations) + " "  # type: ignore
        narration_str += self._get_harvested_str(player_inventory_obs) + " "  # type: ignore
        narration_str += self._get_crafted_str(player_inventory_obs) + " "  # type: ignore
        narration_str += self._get_achievement_str(achievement_obs)  # type: ignore
        narration_str += self._get_vitals_str(player_inventory_obs, occupancy_observations) + " "  # type: ignore

        return narration_str


env = gym.make("CrafterReward-v1")  # Or CrafterNoReward-v1
env = HumanRendering(env)
obs, _ = env.reset()
done = False
step_count = 0
narrate_every = 16
obs_hist = []
narrator = CrafterNarrator()
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        done = True
    occupancy_grid = info["semantic"]
    player_x, player_y = np.where(occupancy_grid == 13)
    player_x, player_y = player_x[0], player_y[0]
    local_grid = occupancy_grid[
        player_x - 4 : player_x + 5, player_y - 3 : player_y + 4
    ].T
    obs_hist.append(
        {
            "semantic": local_grid,
            "inventory": info["inventory"],
            "achievements": info["achievements"],
        }
    )
    step_count += 1
    if step_count % narrate_every == 0 or done:
        print(narrator.narrate(obs_hist))
        obs_hist = []
        # input("Press Enter to continue...")  # Pause between narration sequences
