import numpy as np

from envs.wrappers import MiniGridFullObsWrapper
from envs.minigird_envs.teleport import TeleportComplex
from narration.minigrd_narrator import MiniGridComplexTeleportNarrator


def main():
    env = TeleportComplex(render_mode="human")
    env = MiniGridFullObsWrapper(env)
    narrator = MiniGridComplexTeleportNarrator()

    obs, info = env.reset()
    print(obs)
    done = False
    obs_history = [obs["encoded_image"]]
    while True:
        action = input("Enter action: ")
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_history.append(obs["encoded_image"])

        if terminated:
            print(narrator.narrate(obs_history))
            obs, info = env.reset()
            obs_history = [obs["encoded_image"]]

        if len(obs_history) == 16:
            print(narrator.narrate(obs_history))
            obs_history = []


if __name__ == "__main__":
    main()
