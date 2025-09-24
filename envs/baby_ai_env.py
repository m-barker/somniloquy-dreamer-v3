import sys

sys.path.append("/home/matt/dev/somniloquy-dreamer-v3")
from typing import Dict, List, SupportsFloat, Tuple, Optional
import cv2
import gymnasium as gym
import numpy as np

from wrappers import MiniGridFullObsWrapper  # type: ignore


class BabyAI:
    def __init__(
        self,
        task_name: str,
        img_size: Tuple[int, int] = (64, 64),
        actions: str = "needed",
        max_length: int = 1024,
        seed: int = 42,
        full_obs: bool = True,
        human_render: bool = False,
        reward: bool = False,
        terminate: bool = False,
        fixed_env: bool = True,
        fixed_seed: Optional[int] = None,
    ):
        """
        Wrapper for the BabyAI environments

        Arguments:
            - task_name (str): name of the BabyAI task. Must be a valid one as
                               given by https://minigrid.farama.org/environments/babyai/

            - img_size (Tuple[int, int]): (H,W) of image to return. Resized if source image is
                                          too big.

            - actions (str): "all" or "needed". If "needed", the action space is filtered to
                             remove actions that are irrelevant for solving the task, e.g.,
                             the 'pickup' action in a navigation task.

            - max_length (int): maximum length of an episode. Returns truncated after this
                                number of environment steps. Defaults to 1024

            - seed (int): seed of the environment. Defaults to 42 - the meaning of life, the
                          universe, and everything.

            - full_obs (bool): whether to use full or partial observability. Defaults to True.

            - human_render (bool): whether to render the environment to the screen. Defaults to
                                   False, as obviously much slower than headless.

            - reward (bool): whether to use the reward function given by the environment. If
                             false, the environment gives no reward signal, used for training
                             a world model with no policy. Defaults to false.

            - terminate (bool): whether the episode should ever terminate. Disabled when the
                                aim is to only learn a dynamics model. Defaults to False.

            - fixed_env (bool): whether the same exact environment is used throughout training
                                and testing. If false, the objects, start state, etc. are
                                randomly generated after each reset. Defaults to True.

            - fixed_seed (Optional[int]): seed to use to fix the environment generation. Defaults
                                          to None.
        """
        assert img_size[0] == img_size[1]
        assert actions in ("all", "needed"), actions

        self._actions = actions
        self._max_length = max_length
        self._random = np.random.RandomState(seed)
        self._seed = seed
        self._full_obs = full_obs
        self._done = True
        self._step = 0
        self._img_size = img_size
        self._human_render = human_render

        self._reward = reward
        self._terminate = terminate
        self._fixed_env = fixed_env
        self._fixed_seed = fixed_seed

        self._objects = None

        self._env = self._create_env(task_name)

    def _create_env(self, task_name: str) -> gym.Env:
        print(f"Creating BabyAI environment for task: {task_name}")
        render_mode = None
        if self._human_render:
            render_mode = "human"

        env = gym.make(
            task_name,
            max_steps=self._max_length,
            render_mode=render_mode,
        )
        if self._actions == "needed":
            # Forward, Turn left, Turn right
            env.action_space = gym.spaces.Discrete(3)
        if self._full_obs:
            env = MiniGridFullObsWrapper(env)
        else:
            raise NotImplementedError("Partial observation not implemented yet.")
        return env

    @property
    def observation_space(self):
        img_shape = self._img_size + (3,)
        # Assuming full obs for now.
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
                "occupancy_grid": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._env.width, self._env.height, 3),
                    dtype="uint8",
                ),
                "flattened_occupancy_grid": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._env.width * self._env.height * 3,),
                    dtype="uint8",
                ),
                "direction": gym.spaces.Discrete(4),
            }
        )

    @property
    def mission(self):
        return self._env.mission

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def _get_objects_in_scene(
        self, scene: np.ndarray
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Gets the list of object names present in the scene, e.g.,
        "red ball". Along with their posiiton (cell coord) For
        now, assumes that this is static, i.e., no objects can
        be pickedup up, etc.

        Arguments:
            - scene (np.ndarray): occupancy grid representation of the scene

        Returns:
           List[str]: list of object names plus their colour
        """
        objects: List[Tuple[str, Tuple[int, int]]] = []
        objects_of_interest_id_to_name = {5: "key", 6: "ball", 7: "box"}
        colour_id_to_name = {
            0: "red",
            1: "green",
            2: "blue",
            3: "purple",
            4: "yellow",
            5: "grey",
        }
        assert len(scene.shape) == 3, "occupancy grid must have three dimensions"

        for height in range(scene.shape[0]):
            for width in range(scene.shape[1]):
                cell = scene[height][width]
                # Each cell is encoded as the tuple (OBJECT_IDX, COLOR_IDX, STATE)
                object_id = cell[0]
                object_colour_idx = cell[1]

                if object_id in objects_of_interest_id_to_name.keys():
                    object_name = objects_of_interest_id_to_name[object_id]
                    object_colour = colour_id_to_name[object_colour_idx]
                    object_name = f"{object_colour} {object_name}"

                    # Handle multiple objects of the same colour and type
                    unique_id = 1
                    while any(name == object_name for name, _ in objects):
                        object_name = f"{object_name} {unique_id}"
                        unique_id += 1
                    objects.append((object_name, (height, width)))

        return objects

    def step(self, action) -> Tuple[Dict, SupportsFloat, bool, Dict]:
        """
        Returns obs, reward, done, info.

        We pass the non-Dreamer required observations (i.e., for the
        narrator) in info.
        """
        if len(action.shape) >= 1:
            action = np.argmax(action)

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        is_last = bool(
            truncated or (self._max_length and self._step >= self._max_length)
        )
        if not self._terminate:
            terminated = False
        if not self._reward:
            reward = 0.0

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        if rgb_image.shape[:-2] != self._img_size:
            rgb_image = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        reward_info = self._get_reward_dict(occupancy_grid, direction)

        return (
            {
                "image": rgb_image,
                "is_terminal": terminated,
                "is_first": False,  # False, as we have just taken a step
            },
            reward,
            is_last,
            {
                "occupancy_grid": occupancy_grid,
                "agent_direction": direction,
                "reward_info": reward_info,
            },
        )

    def _is_agent_facing_object(
        self,
        agent_position: Tuple[int, int],
        agent_direction: int,
        object_position: Tuple[int, int],
    ) -> bool:
        """
        Determines if the agent is facing (i.e., pointing at) a given object.

        Arguments:
            - agent_position Tuple[int, int]: (x,y) coord of agent
            - agent_direction int: integer encoding direction of agent in range [0,3]
            - object_position Tuple[int, int]: (x,y) coord of object to check

        Returns bool: True if agent is facing the object else false
        """
        agent_facing = False

        # Grid coords are:
        #   0 1 2 3
        # 0| | | | |
        # 1| | | | |
        # 2| | | | |
        # 3| | | | |

        agent_x, agent_y = agent_position
        object_x, object_y = object_position

        if agent_y == object_y:
            # Agent to the left of object
            if agent_x == object_x - 1:
                agent_facing = agent_direction == 0
            # Agent to the right of object
            elif agent_x == object_x + 1:
                agent_facing = agent_direction == 2
        elif agent_x == object_x:
            # Agent above object
            if agent_y == object_y - 1:
                agent_facing = agent_direction == 1
            # Agent below object
            elif agent_y == object_y + 1:
                agent_facing = agent_direction == 3

        return agent_facing

    def _get_reward_dict(self, obs: np.ndarray, agent_direction: int) -> Dict:
        """
        Returns a dictionary of which objects were rewarded for being reached.
        """

        objects = self._objects

        reward_dict = {k[0]: 0.0 for k in objects}

        agent_id = 10
        observation_ids = obs[:, :, 0]  # Remove colour and status info
        agent_location = np.nonzero(observation_ids == agent_id)
        agent_location = (int(agent_location[0]), int(agent_location[1]))

        for object in objects:
            if self._is_agent_facing_object(agent_location, agent_direction, object[1]):
                reward_dict[object[0]] = 1.0

        return reward_dict

    def reset(self, seed=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        Resets environment and returns obs, info
        """
        if self._fixed_env:
            assert self._fixed_seed is not None, (
                "fixed seed can't be none if fixing environment generation"
            )
            seed = self._fixed_seed
        obs, info = self._env.reset(seed=seed)
        self._step = 0

        rgb_image = obs["rgb_image"]
        occupancy_grid = obs["encoded_image"]
        direction = int(obs["direction"])

        if self._objects is None:
            self._objects = self._get_objects_in_scene(occupancy_grid)

        if rgb_image.shape[:-2] != self._img_size:
            rgb_image = cv2.resize(
                rgb_image, self._img_size, interpolation=cv2.INTER_AREA
            )

        return {
            "image": rgb_image,
            "is_terminal": False,  # we're only just getting started!
            "is_first": True,
        }, {"occupancy_grid": occupancy_grid, "agent_direction": direction}

    def close(self):
        return self._env.close()


if __name__ == "__main__":
    env = BabyAI(
        task_name="BabyAI-GoToLocal-v0",
        full_obs=True,
        human_render=True,
        max_length=512,
        seed=42,
        fixed_env=True,
        fixed_seed=100,
    )
    obs, info = env.reset()
    done = False
    step = 0
    while not done:
        action_arr = np.zeros((env.action_space.n,), dtype=np.int16)
        action = input("Please enter an action: ")
        action_arr[int(action)] = 1
        obs, reward, done, info = env.step(action_arr)
        step += 1
        print(f"Step: {step}")
        # print(f"Obs: {obs.keys()}")
        # print(f"Reward: {reward}")
        # print(f"Is terminal: {obs['is_terminal']}")
        # print(f"Is done: {done}")
        # print(f"Info: {info.keys()}")
        # print(f"Reward Information: {info['reward_info']}")
