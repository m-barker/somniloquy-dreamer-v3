from somniloquy import make_env
from narration.safety_gymnasium_narrator import SafetyGymnasiumNavigationNarrator


class MockConfig:
    def __init__(self):
        self.task = "safetygymnasium_SafetyPointGoal2-v0"
        self.seed = 42
        self.time_limit = 5000
        self.size = (64, 64)


def main():
    config = MockConfig()
    narrator = SafetyGymnasiumNavigationNarrator()
    env = make_env(config, "train", 0)
    print(env)
    obs, info = env.reset()
    done = False
    step = 0
    agent_positions = []
    goal_positions = []
    hazard_positions = []
    vase_positions = []
    hazard_costs = []
    vase_contact_costs = []
    vase_velocity_costs = []
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step({"action": action})
        agent_positions.append(info["agent_pos"])
        goal_positions.append(info["goal_pos"])
        hazard_positions.append(info["hazards_pos"])
        vase_positions.append(info["vases_pos"])
        hazard_costs.append(info["cost_hazards"])
        vase_contact_costs.append(info["cost_vases_contact"])
        vase_velocity_costs.append(info["cost_vases_velocity"])
        step += 1
        print(info["cost_sum"])
        if step % 16 == 0:
            narration = narrator.narrate(
                agent_positions,
                goal_positions,
                hazard_positions,
                vase_positions,
                hazard_costs,
                vase_contact_costs,
                vase_contact_costs,
            )
            print(narration)
            agent_positions = []
            goal_positions = []
            hazard_positions = []
            vase_positions = []
            hazard_costs = []
            vase_contact_costs = []
            vase_velocity_costs = []
        # print(info)


if __name__ == "__main__":
    main()
