from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX  # type: ignore
from minigrid.core.constants import COLOR_NAMES  # type: ignore
from minigrid.minigrid_env import MiniGridEnv  # type: ignore
from minigrid.core.mission import MissionSpace  # type: ignore
from minigrid.core.grid import Grid  # type: ignore
from minigrid.core.world_object import Floor, Goal  # type: ignore


class FourSquares(MiniGridEnv):
    def __init__(
        self,
        size: int = 8,
        agent_start_pos: tuple[int, int] = (4, 4),
        agent_start_dir: int = 0,
        max_steps: int = 1000,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_count = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            mission_space=mission_space,
            see_through_walls=True,
            # render_mode="human",
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "navigate to one of the coloured squares"

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.put_obj(Floor(COLOR_NAMES[0]), 1, 1)
        self.put_obj(Floor(COLOR_NAMES[1]), width - 2, 1)
        self.put_obj(Floor(COLOR_NAMES[3]), 1, height - 2)
        self.put_obj(Floor(COLOR_NAMES[2]), width - 2, height - 2)

        goal_position = np.random.choice(4, 1, replace=False)
        if goal_position == 0:
            self.put_obj(Goal(COLOR_NAMES[0]), 1, 1)
            self.put_obj(Floor(COLOR_NAMES[0]), 1, 2)
            self.put_obj(Floor(COLOR_NAMES[0]), 2, 1)
            self.put_obj(Floor(COLOR_NAMES[0]), 2, 2)
            mission_str = "navigate to the blue square"
        elif goal_position == 1:
            self.put_obj(Goal(COLOR_NAMES[1]), width - 2, 1)
            self.put_obj(Floor(COLOR_NAMES[1]), width - 2, 2)
            self.put_obj(Floor(COLOR_NAMES[1]), width - 3, 1)
            self.put_obj(Floor(COLOR_NAMES[1]), width - 3, 2)
            mission_str = "navigate to the green square"
        elif goal_position == 2:
            self.put_obj(Goal(COLOR_NAMES[2]), width - 2, height - 2)
            self.put_obj(Floor(COLOR_NAMES[2]), width - 2, height - 3)
            self.put_obj(Floor(COLOR_NAMES[2]), width - 3, height - 2)
            self.put_obj(Floor(COLOR_NAMES[2]), width - 3, height - 3)
            mission_str = "navigate to the grey square"
        else:
            self.put_obj(Goal(COLOR_NAMES[3]), 1, height - 2)
            self.put_obj(Floor(COLOR_NAMES[3]), 1, height - 3)
            self.put_obj(Floor(COLOR_NAMES[3]), 2, height - 2)
            self.put_obj(Floor(COLOR_NAMES[3]), 2, height - 3)
            mission_str = "navigate to the purple square"

        self.mission = mission_str


class MiniGrid:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(64, 64),
        gray=False,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        resize="opencv",
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2

            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image

            self._image = Image
        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._random = np.random.RandomState(seed)
        with self.LOCK:
            print(f"Creating environment: {name}")
            self._env = FourSquares()
            self._env = MiniGridFullObsWrapper(self._env)
            self._env.action_space = gym.spaces.Discrete(3)
        shape = self._env.observation_space["rgb_image"].shape
        print(f"Observation shape: {shape}")
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
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

    def step(self, action):
        total = 0.0
        dead = False
        if len(action.shape) >= 1:
            action = np.argmax(action)

        obs, reward, over, truncated, info = self._env.step(action)
        self._buffer[0] = obs["rgb_image"]
        self._step += 1
        total += reward
        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]
        # self._screen(self._buffer[0])
        self._done = over or (self._length and self._step >= self._length) or truncated
        return self._obs(
            total,
            is_last=self._done,
            is_terminal=over or truncated,
            encoded_image=obs["encoded_image"],
        )

    def reset(self, goal=None):
        obs, info = self._env.reset()
        if goal is not None:
            while info["mission"] != goal:
                obs, info = self._env
        self._buffer[0] = obs["rgb_image"]
        # self._screen(self._buffer[0])
        self._buffer[1].fill(0)

        self._done = False
        self._step = 0
        obs, reward, is_terminal, encoded_image = self._obs(
            0.0, is_first=True, encoded_image=obs["encoded_image"]
        )

        obs["encoded_image"] = encoded_image["encoded_image"]

        return obs

    def _obs(
        self,
        reward,
        is_first=False,
        is_last=False,
        is_terminal=False,
        encoded_image=None,
    ):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            if self._resize == "pillow":
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]
        assert image.shape == (64, 64, 3)
        return (
            {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {"encoded_image": encoded_image},
        )

    def _screen(self, array):
        self._ale.getScreenRGB2(array)

    def close(self):
        return self._env.close()


class MiniGridFullObsWrapper(ObservationWrapper):
    """Combining the two existing mini-grid RGBImage wrapper and
    FullObs wrapper into one wrapper."""

    def __init__(
        self,
        env,
        tile_size: int = 8,
    ):
        super().__init__(env)
        self._tile_size = tile_size

        rgb_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self._tile_size * self.env.width,
                self._tile_size * self.env.height,
                3,
            ),
            dtype="uint8",
        )

        encoded_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "rgb_image": rgb_image_space,
                "encoded_image": encoded_image_space,
            }
        )

    def observation(self, observation) -> dict:
        rgb_image = self.get_frame(highlight=True, tile_size=self._tile_size)
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
        encoded_image = full_grid
        return {
            **observation,
            "rgb_image": rgb_image,
            "encoded_image": encoded_image,
        }
