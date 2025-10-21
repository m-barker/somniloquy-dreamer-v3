import numpy as np
from unlock_env import Unlock
import matplotlib.pyplot as plt


def main():
    env = Unlock()
    obs, info = env.reset()
    while True:
        action = input("Please enter an action: ")
        action = int(action)
        action_arr = np.zeros(5)
        action_arr[action] = 1
        obs, reward, terminated, info = env.step(action_arr)
        plt.imshow(obs["image"])
        plt.axis("off")  # optional: hides the axis ticks
        plt.show()
        if terminated:
            break


if __name__ == "__main__":
    main()
