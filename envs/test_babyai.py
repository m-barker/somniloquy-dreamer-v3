import sys

sys.path.append("/home/matt/dev/somniloquy-dreamer-v3")

import minigrid
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym

from narration.minigrd_narrator import BabyAIGoToLocNarrator


def main():
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="human")
    env = FullyObsWrapper(env)
    narrator = BabyAIGoToLocNarrator()

    narration_observations = []
    obs, info = env.reset()
    print(obs)

    narration_observations.append(
        {"occupancy_grid": obs["image"], "agent_direction": int(obs["direction"])}
    )
    print(type(obs["direction"]))
    while True:
        action = input("Please enter an action integer [0-2]")
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        narration_observations.append(
            {"occupancy_grid": obs["image"], "agent_direction": int(obs["direction"])}
        )
        print(terminated)

        if len(narration_observations) == 16:
            print(narrator.narrate(narration_observations))
            narration_observations = []


if __name__ == "__main__":
    main()
