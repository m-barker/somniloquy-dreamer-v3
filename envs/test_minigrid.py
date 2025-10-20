from minigird_envs.custom_unlock import UnlockEnv


def main():
    env = UnlockEnv(agent_start_cell=(1, 1), room_size=15, render_mode="human")
    obs, info = env.reset()
    while True:
        action = input("Please enter an action: ")
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break


if __name__ == "__main__":
    main()
