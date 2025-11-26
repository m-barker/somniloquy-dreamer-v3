from somniloquy import make_env


class MockConfig:
    def __init__(self):
        self.task = "safetygymnasium_SafetyPointGoal2-v0"
        self.seed = 42
        self.time_limit = 5000
        self.size = (64, 64)


def main():
    config = MockConfig()
    env = make_env(config, "train", 0)
    print(env)
    obs, info = env.reset()
    print(obs)
    print(info)
    done = False
    while not done:
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step({"action": action})
        print(obs)
        print(reward)
        print(done)
        print(info)


if __name__ == "__main__":
    main()
