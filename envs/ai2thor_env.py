from typing import Tuple, Dict, List

import numpy as np
import gym

from PIL import Image
from gym import spaces
from gym.utils import seeding
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class AI2ThorBaseEnv(gym.Env):
    def __init__(
        self,
        action_names: List[str],
        scene: str = "FloorPlan10",
        img_size: Tuple[int, int] = (64, 64),
        seed: int = 42,
        max_length: int = 5012,
        headless: bool = True,
    ) -> None:
        super().__init__()
        self._image_size = img_size
        platform = None
        if headless:
            platform = CloudRendering
        self.controller = Controller(scene=scene, platform=platform)
        self.action_names = action_names
        self.seed(seed)
        self._step = 0
        self.done = True
        self.max_length = max_length

        self.move_magnitude = 0.1  # metres to move held object

        # x, y, z position of agent
        self.agent_position: Tuple[float, float, float] = 0.0, 0.0, 0.0

        self.closest_object = None
        self.closest_receptacle = None
        self.closest_graspable_object = None
        self.closest_toggleable_object = None
        self.closest_openable_object = None
        self.closest_sliceable_object = None
        self.closest_breakable_object = None

    def update_agent_position(self, metadata: Dict) -> None:
        """Updates the x, y, z position of the agent after
        each environment step

        Args:
            metadata (Dict): Dictionary of metadata returned by env.
        """
        new_x = metadata["agent"]["position"]["x"]
        new_y = metadata["agent"]["position"]["y"]
        new_z = metadata["agent"]["position"]["z"]

        self.agent_position = (new_x, new_y, new_z)

    def set_closest_objects(self, metadata: Dict) -> None:
        """Sets the set of closest objects to the agent's
        camera.

        Args:
            metadata (Dict): Dictionary of metadata returned by env
        """
        nearest_object_name = None
        nearest_object_dist = float("inf")
        nearest_receptacle_name = None
        nearest_receptacle_dist = float("inf")
        nearest_graspable_name = None
        nearest_graspable_dist = float("inf")
        nearest_toggleable_name = None
        nearest_toggleable_dist = float("inf")
        nearest_openable_name = None
        nearest_openable_dist = float("inf")
        nearest_sliceable_name = None
        nearest_sliceable_dist = float("inf")
        nearest_breakable_name = None
        nearest_breakable_dist = float("inf")

        for object_meta in metadata["objects"]:
            # Agent can't see the object
            if not object_meta["visible"]:
                continue
            if object_meta["distance"] < nearest_object_dist:
                nearest_object_name = object_meta["objectId"]
                nearest_object_dist = object_meta["distance"]
            if object_meta["receptacle"]:
                if object_meta["distance"] < nearest_receptacle_dist:
                    nearest_receptacle_name = object_meta["objectId"]
                    nearest_receptacle_dist = object_meta["distance"]
            if object_meta["pickupable"]:
                if object_meta["distance"] < nearest_graspable_dist:
                    nearest_graspable_name = object_meta["objectId"]
                    nearest_graspable_dist = object_meta["distance"]
            if object_meta["toggleable"]:
                if object_meta["distance"] < nearest_toggleable_dist:
                    nearest_toggleable_name = object_meta["objectId"]
                    nearest_toggleable_dist = object_meta["distance"]
            if object_meta["openable"]:
                if object_meta["distance"] < nearest_openable_dist:
                    nearest_openable_name = object_meta["objectId"]
                    nearest_openable_dist = object_meta["distance"]
            if object_meta["sliceable"]:
                if object_meta["distance"] < nearest_sliceable_dist:
                    nearest_sliceable_name = object_meta["objectId"]
                    nearest_sliceable_dist = object_meta["distance"]
            if object_meta["breakable"]:
                if object_meta["distance"] < nearest_breakable_dist:
                    nearest_breakable_name = object_meta["objectId"]
                    nearest_breakable_dist = object_meta["distance"]

        self.closest_object = nearest_object_name
        self.closest_receptacle = nearest_receptacle_name
        self.closest_graspable_object = nearest_graspable_name
        self.closest_toggleable_object = nearest_toggleable_name
        self.closest_openable_object = nearest_openable_name
        self.closest_sliceable_object = nearest_sliceable_name
        self.closest_breakable_object = nearest_breakable_name

    @property
    def observation_space(self) -> spaces.Dict:
        img_shape = self._image_size + (3,)
        return spaces.Dict({"image": spaces.Box(0, 255, img_shape, np.uint8)})

    @property
    def action_space(self) -> spaces.Discrete:
        space = spaces.Discrete(len(self.action_names))
        space.discrete = True
        return space

    def reset(self) -> Tuple[Dict, Dict]:
        """Resets the environment to its starting state.

        Returns:
            Tuple[Dict, Dict]: obs, info.
        """
        event = self.controller.reset()
        self.update_agent_position(event.metadata)
        self.set_closest_objects(event.metadata)
        obs = self.process_obs(event, is_first=True)
        info = self.filter_metadata(event.metadata)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Takes a step in the environment, and returns the
        standard (obs, reward, done, info) tuple.

        Args:
            action (np.ndarray): One-hot-encoded action.
        """
        assert len(action) == len(self.action_names)
        assert np.count_nonzero(action) == 1

        action_index = np.argmax(action)

        action_name = self.action_names[action_index]
        no_op = False
        if (
            "Object" in action_name
            and "Held" not in action_name
            and "Drop" not in action_name
        ):
            if action_name == "PickupObject":
                no_op = self.closest_graspable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_graspable_object
                    )
            elif action_name == "PutObject":
                no_op = self.closest_receptacle == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_receptacle
                    )
            elif action_name == "OpenObject" or action_name == "CloseObject":
                no_op = self.closest_openable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_openable_object
                    )
            elif action_name == "BreakObject":
                no_op = self.closest_breakable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_breakable_object
                    )
            elif action_name == "SliceObject":
                no_op = self.closest_sliceable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_sliceable_object
                    )
            elif "Toggle" in action_name:
                no_op = self.closest_toggleable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_toggleable_object
                    )
        elif "Held" in action_name:
            event = self.controller.step(
                action=action_name, moveMagnitude=self.move_magnitude
            )
        else:
            event = self.controller.step(action=action_name)

        if no_op:
            # Done does nothing to the state of the environment
            event = self.controller.step(action="Done")

        self.update_agent_position(event.metadata)
        self.set_closest_objects(event.metadata)

        obs = self.process_obs(event)
        reward = self.compute_reward(event)
        done = self.is_terminated(event)
        info = self.filter_metadata(event.metadata)

        self._step += 1
        self._done = done or (self.max_length and self._step >= self.max_length)

        return obs, reward, done, info

    def close(self):
        self.controller.stop()

    def compute_reward(self, event) -> float:
        return 0.0

    def is_terminated(self, event) -> bool:
        return False

    def filter_metadata(self, metadata) -> Dict:
        return metadata

    def process_obs(self, event, is_first: bool = False, done: bool = False) -> Dict:
        return {}

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return seed1


class CookEggEnv(AI2ThorBaseEnv):
    """Gymnasium environment that defines a task to cook an egg
    in an ai2thor kitchen environment.

    The agent gets rewarded for picking up / placing the neccessary
    object. The sequence of steps that need to be completed are:

    1. Open the fridge.
    2. Pickup the egg.
    3. Crack the egg.
    4. Pick up the cracked egg.

    <<<MICROWAVE PATHWAY>>>
    4. Put cracked egg in microwave.
    5. Turn microwave on.
    6. Pickup cooked egg.
    7. Done.

    <<<STOVE PATHWAY>>>
    4. Put cracked egg in pot or pan.
    5. Put pot/pan on stove burner.
    6. Turn-on stove burner.
    7. Pickup cooked egg.
    8. Done.

    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        seed: int = 42,
        max_length: int = 5012,
        headless: bool = True,
    ) -> None:
        ACTION_NAMES = [
            "PickupObject",
            "PutObject",
            "DropHandObject",
            "MoveHeldObjectAhead",
            "MoveHeldObjectBack",
            "MoveHeldObjectLeft",
            "MoveHeldObjectRight",
            "MoveHeldObjectUp",
            "MoveHeldObjectDown",
            "OpenObject",
            "CloseObject",
            "BreakObject",
            "SliceObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "MoveAhead",
            "MoveBack",
            "MoveLeft",
            "MoveRight",
            "RotateRight",
            "RotateLeft",
            "LookUp",
            "LookDown",
        ]
        super().__init__(
            action_names=ACTION_NAMES,
            scene="FloorPlan10",
            img_size=img_size,
            seed=seed,
            max_length=max_length,
            headless=headless,
        )

        self.fridge_opened = False
        self.egg_picked_up = False
        self.egg_cracked = False
        self.microwave_opened = False
        self.microwave_opened_finished = False
        self.microwave_on = False
        self.egg_cracked_picked_up = False
        self.egg_cooked = False
        self.egg_cooked_picked_up = False
        self.egg_in_pan = False
        self.egg_in_pot = False
        self.pan_on_stove = False
        self.pot_on_stove = False
        self.stove_on = False

    def is_terminated(self, event) -> bool:
        # Should return terminated if the agent is holding a cooked egg.
        for object_meta in event.metadata["objects"]:
            if object_meta["objectType"] == "EggCracked":
                if object_meta["isCooked"] and object_meta["isPickedUp"]:
                    return True
        return False

    def compute_reward(self, event) -> float:
        reward = 0.0
        for object_meta in event.metadata["objects"]:
            # Agent must see the object to get the reward
            if not object_meta["visible"]:
                continue
            if object_meta["objectType"] == "Fridge":
                if object_meta["isOpen"] and not self.fridge_opened:
                    reward += 1.0
                    self.fridge_opened = True
            if object_meta["objectType"] == "Microwave":
                if object_meta["isOpen"] and not self.microwave_opened:
                    reward += 1.0
                    self.microwave_opened = True
                elif (
                    not object_meta["isOpen"]
                    and not self.microwave_opened_finished
                    and self.microwave_on
                ):
                    reward += 1.0
                    self.microwave_opened_finished = True
                elif object_meta["isToggled"] and not self.microwave_on:
                    # Check if there is a cracked egg in the microwave
                    for obj in object_meta["receptacleObjectIds"]:
                        if "EggCracked" in obj:
                            for obj_to_check_meta in event.metadata["objects"]:
                                if obj_to_check_meta["objectId"] == obj:
                                    reward += 5.0
                                    self.microwave_on = True
                                    break
            if object_meta["objectType"] == "Pan" or object_meta["objectType"] == "Pot":
                if not self.stove_on:
                    egg_curently_in_pot_pan = False
                    # Check if there is a cracked egg in the pot/pan
                    for obj in object_meta["receptacleObjectIds"]:
                        if "EggCracked" in obj:
                            for obj_to_check_meta in event.metadata["objects"]:
                                if obj_to_check_meta["objectId"] == obj:
                                    if not self.egg_in_pan and not self.egg_in_pot:
                                        reward += 1.0
                                    if object_meta["objectType"] == "Pan":
                                        self.egg_in_pan = True
                                    else:
                                        self.egg_in_pot = True
                                    egg_curently_in_pot_pan = True
                                    break
                    if egg_curently_in_pot_pan:
                        for parent in object_meta["parentReceptacles"]:
                            if "StoveBurner" in parent:
                                if not self.pan_on_stove and not self.pot_on_stove:
                                    reward += 5.0
                                    if object_meta["objectType"] == "Pan":
                                        self.pan_on_stove = True
                                    else:
                                        self.pot_on_stove = True
                                if not self.stove_on:
                                    for obj_to_check_meta in event.metadata["objects"]:
                                        if obj_to_check_meta["objectId"] == parent:
                                            if obj_to_check_meta["isOn"]:
                                                self.stove_on = True
                                                reward += 1.0
                # Finally, give rewards for picking up egg, cracked egg, and cooked cracked egg.
                if object_meta["objectType"] == "Egg":
                    if object_meta["isPickedUp"] and not self.egg_picked_up:
                        self.egg_picked_up = True
                        reward += 1.0
                if object_meta["objectType"] == "EggCracked":
                    if not object_meta["isCooked"] and not self.egg_cracked_picked_up:
                        self.egg_cracked_picked_up = True
                        reward += 1.0
                    elif object_meta["isCooked"] and not self.egg_cooked_picked_up:
                        self.egg_cooked_picked_up
                        reward += 10.0

        return reward

    def process_obs(self, event, is_first: bool = False, done: bool = False) -> Dict:
        rgb_image: np.ndarray = event.frame

        h, w, _ = rgb_image.shape

        if (h, w) != self._image_size:
            image = Image.fromarray(rgb_image)
            image = image.resize(self._image_size, resample=Image.BILINEAR)  # type: ignore
            image = np.asarray(image)  # type: ignore
        else:
            image = rgb_image  # type: ignore

        return {"image": image, "is_terminal": done, "is_first": is_first}


if __name__ == "__main__":
    env = CookEggEnv()
    done = False
    while not done:
        action = np.zeros(env.action_space.n)
        int_action = env.action_space.sample()
        action[int_action] = 1
        obs, reward, done, info = env.step(action)
        if reward > -0.0:
            print(f"Reward: {reward}")
