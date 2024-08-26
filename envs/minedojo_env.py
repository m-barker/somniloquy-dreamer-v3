import minedojo
import numpy as np
import gym
import cv2


class MineDojoEnv(gym.Env):
    def __init__(self, task_id="harvest_1_dirt", image_size=(64, 64)):
        self.task_id = task_id
        self.image_size = image_size

        self.env = minedojo.make(task_id=task_id, image_size=(160, 256))
        self.action_size = 61
        self.sticky_action_length = 30
        self._sticky_attack_counter = 0
        self._noop_mapping = {0: 0, 1: 0, 2: 0, 3: 12, 4: 12, 5: 0, 6: 0, 7: 0}

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
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

        if action_index < 3:
            env_action[0] = action_index
        elif action_index < 6:
            env_action[1] = action_index - 3
        elif action_index < 10:
            env_action[2] = action_index - 6
        elif action_index < 35:
            env_action[3] = action_index - 10
        elif action_index < 60:
            env_action[4] = action_index - 35
        else:
            env_action[5] = 3
        return env_action

    def _obs(self, obs):
        # resize image
        obs["rgb"] = np.transpose(obs["rgb"], (1, 2, 0))
        obs["rgb"] = cv2.resize(obs["rgb"], self.image_size)
        # assert obs["rgb"].shape == (64, 64, 3)
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
        if np.argmax(action) == 60:
            self._sticky_attack_counter = self.sticky_action_length
        if self._sticky_attack_counter > 0:
            action = np.zeros(self.action_size)
            action[60] = 1
            self._sticky_attack_counter -= 1
        action = self._action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self._obs(obs)
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", done))
        return obs, reward, done, info
