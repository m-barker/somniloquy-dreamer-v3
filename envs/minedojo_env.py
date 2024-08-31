import minedojo
import numpy as np
import gym
import cv2


class MineDojoEnv(gym.Env):
    def __init__(
        self, task_id="harvest_1_dirt", image_size=(64, 64), world_seed: int = 42
    ):
        self.task_id = task_id
        self.image_size = image_size

        self.env = minedojo.make(
            task_id=task_id, image_size=(160, 256), world_seed=world_seed
        )
        self.action_size = 89  # following frome MineCLIP implementation
        self.sticky_action_length = 30
        self._sticky_attack_counter = 0
        # Maps from action dimension to action no-operation index.
        self._noop_mapping = {0: 0, 1: 0, 2: 0, 3: 12, 4: 12, 5: 0, 6: 0, 7: 0}

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # RGB Observation
                "image": gym.spaces.Box(0, 255, self.image_size + (3,), np.uint8),
                # Pitch and Yaw of the agent
                "compass": gym.spaces.Box(-180, 180, (4,), np.float32),
                # xyz position of the agent
                "position": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.discrete.Discrete(self.action_size)
        space.discrete = True
        return space

    def _action(self, action: np.ndarray) -> np.ndarray:
        """Action: One-hot encoding of the action"""

        action_index = np.argmax(action)
        env_action = np.zeros(8)

        for i in range(8):
            env_action[i] = self._noop_mapping[i]

        # There are 6 movement actions:
        # 0: Forward
        # 1: Forward + Jump
        # 2: Jump
        # 3: Backward
        # 4: Move left
        # 5: Move right
        if action_index < 6:
            if action_index == 0:
                env_action[0] = 1
            elif action_index == 1:
                env_action[0] = 1
                env_action[2] = 1
            elif action_index == 2:
                env_action[2] = 1
            elif action_index == 3:
                env_action[0] = 2
            elif action_index == 4:
                env_action[1] = 1
            elif action_index == 5:
                env_action[1] = 2
            else:
                raise ValueError("Invalid action index")
        # There are 81 camera actions which form the cartesian product
        # of 9 yaw actions and 9 pitch actions, from -60 to 60 degrees.
        # Each with a step of 15 degrees.
        elif action_index < 87:
            pitch = (action_index - 6) // 9
            yaw = (action_index - 6) % 9
            # Convert to -60 to 60 degrees as env action space
            # is from -180 to 180 degrees.
            env_action[3] = pitch + 8
            env_action[4] = yaw + 8
        # Use action
        elif action_index == 87:
            env_action[5] = 1
        # Attack action
        elif action_index == 88:
            env_action[5] = 3
        else:
            raise ValueError("Invalid action index")

        return env_action

    def _obs(self, obs):
        # resize image
        obs["rgb"] = np.transpose(obs["rgb"], (1, 2, 0))
        obs["rgb"] = cv2.resize(obs["rgb"], self.image_size)
        return {"image": obs["rgb"]}

    def reset(self):
        obs = self.env.reset()
        obs = self._obs(obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self._sticky_attack_counter = 0
        return obs

    def step(self, action):
        if np.argmax(action) == 88:
            self._sticky_attack_counter = self.sticky_action_length
        if self._sticky_attack_counter > 0:
            action = np.zeros(self.action_size)
            action[88] = 1
            self._sticky_attack_counter -= 1
        action = self._action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self._obs(obs)
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", done))
        return obs, reward, done, info
