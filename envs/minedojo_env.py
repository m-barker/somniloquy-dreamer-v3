from __future__ import annotations

import minedojo  # type: ignore
import numpy as np
import gym  # type: ignore
import cv2  # type: ignore
from gym import Wrapper
from collections import deque
from typing import Literal


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
                "compass": gym.spaces.Box(-180, 180, (2,), np.float32),
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

    def _obs(self, obs: dict) -> dict:
        # resize image
        rgb_frame = np.transpose(obs["rgb"], (1, 2, 0))
        rgb_frame = cv2.resize(rgb_frame, self.image_size)
        flattened_compass = np.array(
            [obs["location_stats"]["yaw"], obs["location_stats"]["pitch"]]
        ).squeeze()
        posistion = obs["location_stats"]["pos"]

        # np array (3, 3, 3)
        voxel_meta = obs["voxels"]["block_meta"]
        voxel_meta = voxel_meta.flatten()
        return {
            "image": rgb_frame,
            "compass": flattened_compass,
            "position": posistion,
            "voxel_meta": voxel_meta,
        }

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


class AnimalZooDenseRewardWrapper(Wrapper):
    """From MineCLIP implementation"""

    def __init__(
        self,
        env,
        entity: Literal["cow", "sheep"],
        step_penalty: float | int,
        nav_reward_scale: float | int,
        attack_reward: float | int,
    ):
        assert (
            "rays" in env.observation_space.keys()
        ), "Dense reward function requires lidar observation"
        super().__init__(env)

        self._entity = entity
        assert step_penalty >= 0, f"penalty must be non-negative"
        self._step_penalty = step_penalty
        self._nav_reward_scale = nav_reward_scale
        self._attack_reward = attack_reward

        self._weapon_durability_deque = deque(maxlen=2)  # type: ignore
        self._consecutive_distances = deque(maxlen=2)  # type: ignore
        self._distance_min = np.inf

    def reset(self, **kwargs):
        self._weapon_durability_deque.clear()
        self._consecutive_distances.clear()
        self._distance_min = np.inf

        obs = super().reset(**kwargs)

        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
        else:
            self._consecutive_distances.append(0)
        self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])

        return obs

    def step(self, action):
        obs, _reward, done, info = super().step(action)

        self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])
        valid_attack = (
            self._weapon_durability_deque[0] - self._weapon_durability_deque[1]
        )
        # when dying, the weapon is gone and durability changes to 0
        valid_attack = 1.0 if valid_attack == 1.0 else 0.0

        # attack reward
        attack_reward = valid_attack * self._attack_reward
        # nav reward
        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        nav_reward = 0
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
            nav_reward = self._consecutive_distances[0] - self._consecutive_distances[1]
        nav_reward = max(0, nav_reward)
        nav_reward *= self._nav_reward_scale
        # reset distance min if attacking the entity because entity will run away
        if valid_attack > 0:
            self._distance_min = np.inf
        # total reward
        reward = attack_reward + nav_reward - self._step_penalty + _reward
        return obs, reward, done, info

    def _find_distance_to_entity_if_in_sight(self, obs):
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._entity)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance
