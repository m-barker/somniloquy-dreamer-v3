import sys
from typing import Dict, List

sys.path.append("/home/matt/dev/somniloquy-dreamer-v3")

import minigrid
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym
import matplotlib.pyplot as plt

from narration.minigrd_narrator import BabyAIGoToLocNarrator
from wrappers import MiniGridFullObsWrapper


def main():
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="human")
    env = MiniGridFullObsWrapper(env)
    narrator = BabyAIGoToLocNarrator(simple_narrator=True)
    obs, info = env.reset(seed=100)
    print(obs)

    narration_observations: Dict[str, List] = {
        "occupancy_grid": [obs["encoded_image"]],
        "agent_direction": [int(obs["direction"])],
    }
    print(type(obs["direction"]))
    while True:
        action = input("Please enter an action integer [0-2]")
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        print(obs["high_res_image"].shape)
        narration_observations["occupancy_grid"].append(obs["encoded_image"])
        narration_observations["agent_direction"].append(int(obs["direction"]))
        print(terminated)

        if len(narration_observations["occupancy_grid"]) == 16:
            print(narrator.narrate(narration_observations))
            narration_observations = {
                "occupancy_grid": [],
                "agent_direction": [],
            }


if __name__ == "__main__":
    main()
