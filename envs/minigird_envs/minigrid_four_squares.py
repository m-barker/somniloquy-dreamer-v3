import numpy as np


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
            render_mode="human",
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



