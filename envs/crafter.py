import gym
import numpy as np


class Crafter:
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        assert task in ("reward", "noreward")
        import crafter

        self._env = crafter.Env(size=size, reward=(task == "reward"), seed=seed)
        self._achievements = crafter.constants.achievements.copy()
        self.reward_range = [-np.inf, np.inf]
        self._local_grid_shape = (7, 9)

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(
                0, 255, self._env.observation_space.shape, dtype=np.uint8
            ),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "log_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "flattened_grid": gym.spaces.Box(-np.inf, np.inf, (63,), dtype=np.float32),
            "flattened_inventory": gym.spaces.Box(
                -np.inf, np.inf, (16,), dtype=np.float32
            ),
            "flattened_achievements": gym.spaces.Box(
                -np.inf, np.inf, (22,), dtype=np.float32
            ),
        }
        spaces.update(
            {
                f"log_achievement_{k}": gym.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float32
                )
                for k in self._achievements
            }
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        action_space.discrete = True
        return action_space

    def step(self, action):
        image, reward, terminated, truncated, info = self._env.step(action)
        reward = np.float32(reward)
        log_achievements = {
            f"log_achievement_{k}": info["achievements"][k] if info else 0
            for k in self._achievements
        }

        # Want to focus the occupancy grid to only be what the agent actually
        # sees
        occupancy_grid = info["semantic"]
        player_x, player_y = np.where(occupancy_grid == 13)
        player_x, player_y = player_x[0], player_y[0]
        local_grid = occupancy_grid[
            player_x - 4 : player_x + 5, player_y - 3 : player_y + 4
        ].T

        # Add padding to the local grid if it is smaller than the expected shape
        if local_grid.shape != self._local_grid_shape:
            pad_width = (
                (0, self._local_grid_shape[0] - local_grid.shape[0]),
                (0, self._local_grid_shape[1] - local_grid.shape[1]),
            )
            local_grid = np.pad(
                local_grid, pad_width=pad_width, mode="constant", constant_values=0
            )

        info["semantic"] = local_grid

        # Max entity ID is 18
        flattened_grid = local_grid.flatten() / 18
        # For now, assuming there won't be more than 200 items or achievements
        flattened_inventory = np.array(list(info["inventory"].values())) / 200
        flattened_achievements = np.array(list(info["achievements"].values())) / 200

        assert flattened_achievements.max() <= 1
        assert flattened_inventory.max() <= 1
        assert flattened_grid.max() <= 1

        obs = {
            "image": image,
            "flattened_grid": flattened_grid,
            "flattened_inventory": flattened_inventory,
            "flattened_achievements": flattened_achievements,
            "is_first": False,
            "is_last": terminated or truncated,
            "is_terminal": info["discount"] == 0,
            "log_reward": np.float32(info["reward"] if info else 0.0),
            **log_achievements,
        }
        return obs, reward, terminated or truncated, info

    def render(self):
        return self._env.render()

    def reset(self):
        image, info = self._env.reset()
        # Want to focus the occupancy grid to only be what the agent actually
        # sees
        occupancy_grid = info["semantic"]
        player_x, player_y = np.where(occupancy_grid == 13)
        player_x, player_y = player_x[0], player_y[0]
        local_grid = occupancy_grid[
            player_x - 4 : player_x + 5, player_y - 3 : player_y + 4
        ].T

        # Add padding to the local grid if it is smaller than the expected shape
        if local_grid.shape != self._local_grid_shape:
            pad_width = (
                (0, self._local_grid_shape[0] - local_grid.shape[0]),
                (0, self._local_grid_shape[1] - local_grid.shape[1]),
            )
            local_grid = np.pad(
                local_grid, pad_width=pad_width, mode="constant", constant_values=0
            )

        info["semantic"] = local_grid

        # Max entity ID is 18
        flattened_grid = local_grid.flatten() / 18
        # For now, assuming there won't be more than 500 items or achievements
        flattened_inventory = np.array(list(info["inventory"].values())) / 500
        flattened_achievements = np.array(list(info["achievements"].values())) / 500

        assert flattened_achievements.max() <= 1
        assert flattened_inventory.max() <= 1
        assert flattened_grid.max() <= 1

        obs = {
            "image": image,
            "flattened_grid": flattened_grid,
            "flattened_inventory": flattened_inventory,
            "flattened_achievements": flattened_achievements,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs, info
