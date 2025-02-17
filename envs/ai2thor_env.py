from typing import Tuple, Dict, List, Optional, Set

import numpy as np
import gym

from PIL import Image
from gym import spaces
from gym.utils import seeding
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from typing import List, Tuple, Dict, Any

import numpy as np


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

        # Prev steps metadata. Used for computing which object was dropped, etc.
        self.prev_meta = None

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

        # print(f"----------Agent Position----------")
        # print(self.agent_position)
        # print(f"----------------------------------")

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
        nearest_closeable_name = None
        nearest_closeable_dist = float("inf")
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
            if object_meta["openable"] and not object_meta["isOpen"]:
                if object_meta["distance"] < nearest_openable_dist:
                    nearest_openable_name = object_meta["objectId"]
                    nearest_openable_dist = object_meta["distance"]
            if object_meta["openable"] and object_meta["isOpen"]:
                if object_meta["distance"] < nearest_openable_dist:
                    nearest_closeable_name = object_meta["objectId"]
                    nearest_closeable_dist = object_meta["distance"]
            if object_meta["sliceable"] and not object_meta["isSliced"]:
                if object_meta["distance"] < nearest_sliceable_dist:
                    nearest_sliceable_name = object_meta["objectId"]
                    nearest_sliceable_dist = object_meta["distance"]
            if object_meta["breakable"] and not object_meta["isBroken"]:
                if object_meta["distance"] < nearest_breakable_dist:
                    nearest_breakable_name = object_meta["objectId"]
                    nearest_breakable_dist = object_meta["distance"]

        self.closest_object = nearest_object_name
        self.closest_receptacle = nearest_receptacle_name
        self.closest_graspable_object = nearest_graspable_name
        self.closest_toggleable_object = nearest_toggleable_name
        self.closest_openable_object = nearest_openable_name
        self.closest_closeable_object = nearest_closeable_name
        self.closest_sliceable_object = nearest_sliceable_name
        self.closest_breakable_object = nearest_breakable_name

        # print("----------Closest objects----------")
        # print(f"Object: {self.closest_object}")
        # print(f"Receptacle: {self.closest_receptacle}")
        # print(f"Graspable: {self.closest_graspable_object}")
        # print(f"Toggleable: {self.closest_toggleable_object}")
        # print(f"Openable: {self.closest_openable_object}")
        # print(f"Sliceable: {self.closest_sliceable_object}")
        # print(f"Breakable: {self.closest_breakable_object}")
        # print("------------------------------------")

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
        self._step = 0
        self._done = False
        event = self.controller.reset()
        info = self.filter_metadata(event.metadata, is_first=True)
        self.update_agent_position(event.metadata)
        self.set_closest_objects(event.metadata)
        obs = self.process_obs(event, info, is_first=True)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Takes a step in the environment, and returns the
        standard (obs, reward, done, info) tuple.

        Args:
            action (np.ndarray): One-hot-encoded action.
        """
        assert len(action) == len(self.action_names)
        assert np.count_nonzero(action) == 1
        # print(f"TAKING STEP {self._step}")

        action_index = np.argmax(action)

        action_name = self.action_names[action_index]
        object_interacted_with = None
        no_op = False
        if (
            "Object" in action_name
            and "Held" not in action_name
            and "Drop" not in action_name
            and "Throw" not in action_name
        ):
            if action_name == "PickupObject":
                no_op = self.closest_graspable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_graspable_object
                    )
                    object_interacted_with = self.closest_graspable_object
            elif action_name == "PutObject":
                no_op = self.closest_receptacle == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_receptacle
                    )
                    object_interacted_with = self.closest_receptacle
            elif action_name == "OpenObject":
                no_op = self.closest_openable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_openable_object
                    )
                    object_interacted_with = self.closest_openable_object
            elif action_name == "CloseObject":
                no_op = self.closest_closeable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_closeable_object
                    )
                    object_interacted_with = self.closest_closeable_object
            elif action_name == "BreakObject":
                no_op = self.closest_breakable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_breakable_object
                    )
                    object_interacted_with = self.closest_breakable_object
            elif action_name == "SliceObject":
                no_op = self.closest_sliceable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_sliceable_object
                    )
                    object_interacted_with = self.closest_sliceable_object
            elif "Toggle" in action_name:
                no_op = self.closest_toggleable_object == None
                if not no_op:
                    event = self.controller.step(
                        action=action_name, objectId=self.closest_toggleable_object
                    )
                    object_interacted_with = self.closest_toggleable_object
        elif "Held" in action_name:
            event = self.controller.step(
                action=action_name, moveMagnitude=self.move_magnitude
            )
        elif "Throw" in action_name:
            event = self.controller.step(action=action_name, moveMagnitude=150.0)
        else:
            event = self.controller.step(action=action_name)

        if no_op:
            # Done does nothing to the state of the environment
            event = self.controller.step(action="Done")

        self.update_agent_position(event.metadata)
        self.set_closest_objects(event.metadata)

        info = self.filter_metadata(event.metadata, object_interacted_with)
        obs = self.process_obs(event, info)
        reward = self.compute_reward(event)
        done = self.is_terminated(event)

        self._step += 1
        self._done = done or (self.max_length and self._step >= self.max_length)

        # print(event.metadata["errorMessage"])
        self.prev_meta = event.metadata
        return obs, reward, done, {}

    def close(self):
        self.controller.stop()

    def compute_reward(self, event) -> float:
        return 0.0

    def is_terminated(self, event) -> bool:
        return False

    def filter_metadata(
        self,
        metadata,
        object_interacted_with: Optional[str] = None,
        is_first: bool = False,
    ) -> Dict:
        return metadata

    def process_obs(
        self,
        event,
        filtered_meta: Dict[str, str],
        is_first: bool = False,
        done: bool = False,
    ) -> Dict:
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
        headless: bool = False,
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
            "ThrowObject",
        ]
        OBJECT_STR_TO_ID = {"": 0}
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

        self.log_rewards = {
            "log_fridge_opened": 0,
            "log_egg_picked_up": 0,
            "log_egg_cracked": 0,
            "log_microwave_opened": 0,
            "log_microwave_opened_finished": 0,
            "log_microwave_on": 0,
            "log_egg_cracked_picked_up": 0,
            "log_egg_cooked_picked_up": 0,
            "log_egg_in_pan": 0,
            "log_egg_in_pot": 0,
            "log_pan_on_stove": 0,
            "log_pot_on_stove": 0,
            "log_stove_on": 0,
        }

    def is_terminated(self, event) -> bool:
        # Should return terminated if the agent is holding a cooked egg.
        for object_meta in event.metadata["objects"]:
            if object_meta["objectType"] == "EggCracked":
                if object_meta["isCooked"] and object_meta["isPickedUp"]:
                    return True
        return False

    @property
    def observation_space(self) -> spaces.Dict:
        img_shape = self._image_size + (3,)
        my_spaces = {"image": spaces.Box(0, 255, img_shape, np.uint8)}
        my_spaces.update(
            {
                k: gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
                for k in self.log_rewards.keys()
            }
        )
        return spaces.Dict(my_spaces)

    def reset(self) -> Tuple[Dict, Dict]:
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

        for k in self.log_rewards.keys():
            self.log_rewards[k] = 0

        return super().reset()

    def compute_reward(self, event) -> float:
        reward = 0.0
        # Penalty for failing an action.
        if not event.metadata["lastActionSuccess"]:
            reward -= 1
        for object_meta in event.metadata["objects"]:
            # Agent must see the object to get the reward
            if not object_meta["visible"]:
                continue
            if object_meta["objectType"] == "Fridge":
                if object_meta["isOpen"] and not self.fridge_opened:
                    reward += 1.0
                    self.fridge_opened = True
                    self.log_rewards["log_fridge_opened"] += 1
            if object_meta["objectType"] == "Microwave":
                if object_meta["isOpen"] and not self.microwave_opened:
                    reward += 1.0
                    self.microwave_opened = True
                    self.log_rewards["log_microwave_opened"] += 1
                elif (
                    not object_meta["isOpen"]
                    and not self.microwave_opened_finished
                    and self.microwave_on
                ):
                    reward += 1.0
                    self.microwave_opened_finished = True
                    self.log_rewards["log_microwave_opened_finished"] += 1
                elif object_meta["isToggled"] and not self.microwave_on:
                    # Check if there is a cracked egg in the microwave
                    for obj in object_meta["receptacleObjectIds"]:
                        if "EggCracked" in obj:
                            for obj_to_check_meta in event.metadata["objects"]:
                                if obj_to_check_meta["objectId"] == obj:
                                    reward += 5.0
                                    self.microwave_on = True
                                    self.log_rewards["log_microwave_on"] += 1
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
                                        self.log_rewards["log_egg_in_pan"] += 1
                                        self.egg_in_pan = True
                                    else:
                                        self.egg_in_pot = True
                                        self.log_rewards["log_egg_in_pot"] += 1
                                    egg_curently_in_pot_pan = True
                                    break
                    if egg_curently_in_pot_pan:
                        for parent in object_meta["parentReceptacles"]:
                            if "StoveBurner" in parent:
                                if not self.pan_on_stove and not self.pot_on_stove:
                                    reward += 5.0
                                    if object_meta["objectType"] == "Pan":
                                        self.pan_on_stove = True
                                        self.log_rewards["log_pan_on_stove"] += 1
                                    else:
                                        self.pot_on_stove = True
                                        self.log_rewards["log_pot_on_stove"] += 1
                                if not self.stove_on:
                                    for obj_to_check_meta in event.metadata["objects"]:
                                        if obj_to_check_meta["objectId"] == parent:
                                            if obj_to_check_meta["isOn"]:
                                                self.stove_on = True
                                                reward += 1.0
                                                self.log_rewards["log_stove_on"] += 1
            # Finally, give rewards for picking up egg, cracked egg, and cooked cracked egg.
            if object_meta["objectType"] == "Egg":
                if object_meta["isPickedUp"] and not self.egg_picked_up:
                    self.egg_picked_up = True
                    reward += 1.0
                    self.log_rewards["log_egg_picked_up"] += 1
            if object_meta["objectType"] == "EggCracked":
                if (
                    not object_meta["isCooked"]
                    and object_meta["isPickedUp"]
                    and not self.egg_cracked_picked_up
                ):
                    self.egg_cracked_picked_up = True
                    reward += 1.0
                    self.log_rewards["log_egg_cracked_picked_up"] += 1
                elif (
                    object_meta["isCooked"]
                    and object_meta["isPickedUp"]
                    and not self.egg_cooked_picked_up
                ):
                    self.egg_cooked_picked_up = True
                    self.log_rewards["log_egg_cooked_picked_up"]
                    reward += 10.0

        return reward

    def process_obs(
        self,
        event,
        filtered_meta: Dict[str, str],
        is_first: bool = False,
        done: bool = False,
    ) -> Dict:
        rgb_image: np.ndarray = event.frame

        h, w, _ = rgb_image.shape

        if (h, w) != self._image_size:
            image = Image.fromarray(rgb_image)
            image = image.resize(self._image_size, resample=Image.BILINEAR)  # type: ignore
            image = np.asarray(image)  # type: ignore
        else:
            image = rgb_image  # type: ignore

        return {
            "image": image,
            "is_terminal": done,
            "is_first": is_first,
            "agent_position": self.agent_position,
            **self.log_rewards,
            **filtered_meta,
        }

    def filter_metadata(
        self,
        metadata: Dict,
        object_interacted_with: Optional[str] = None,
        is_first: bool = False,
    ) -> Dict:
        """Filters the metadata returned by the environment
        to return a dictionary of data required for downstream tasks (i.e.,
        narration)

        Args:
            metadata (Dict): Metadata returned by the controller object

            object_interactied_with (Optional[str]): Optional name of the object that the
            agent interacted with in this environment step. Defaults to None.

            is_first (bool, optional): Whether this is the first observation of the
            episode. If true, returns the default null string dict, as
            no interactions could have happened. Defaults to False.

        Returns:
            Dict: Dictionary containing object interactions needed for
            generating the narrations.
        """
        object_interaction_dict = {
            "pickup": "",
            "drop": "",
            "open": "",
            "close": "",
            "break": "",
            "slice": "",
            "toggle_on": "",
            "toggle_off": "",
            "throw": "",
            "put": ("", ""),
        }

        # No interactions could have happened yet
        if is_first:
            return object_interaction_dict

        # If action didn't succeed then this is equivalent
        # to no interaction.
        if not metadata["lastActionSuccess"]:
            return object_interaction_dict

        # Go from object ID i.e., Sink|-00.70|+00.93|-00.65|SinkBasin
        # to object name i.e., Sink
        if object_interacted_with is not None:
            object_interacted_with = object_interacted_with.split("|")[0]

        action_name = metadata["lastAction"]

        if action_name == "PickupObject":
            assert object_interacted_with is not None
            object_interaction_dict["pickup"] = object_interacted_with
        elif action_name == "PutObject":
            assert object_interacted_with is not None
            receptacle = object_interacted_with
            # TODO: figure out which object was put in the receptacle
            object_placed = None
            for obj_meta in self.prev_meta["objects"]:
                if obj_meta["isPickedUp"]:
                    object_placed = obj_meta["objectType"]
                    break
            assert object_placed is not None
            object_interaction_dict["put"] = (object_placed, receptacle)

        elif action_name == "DropHandObject" or action_name == "ThrowObject":
            # TODO: figure out which object the agent was holding
            object_dropped = None
            for obj_meta in self.prev_meta["objects"]:
                if obj_meta["isPickedUp"]:
                    object_dropped = obj_meta["objectType"]
                    break
            assert object_dropped is not None
            if action_name == "DropHandObject":
                object_interaction_dict["drop"] = object_dropped
            else:
                object_interaction_dict["throw"] = object_dropped
        elif action_name == "OpenObject":
            assert object_interacted_with is not None
            object_interaction_dict["open"] = object_interacted_with
        elif action_name == "CloseObject":
            assert object_interacted_with is not None
            object_interaction_dict["close"] = object_interacted_with
        elif action_name == "BreakObject":
            assert object_interacted_with is not None
            object_interaction_dict["break"] = object_interacted_with
        elif action_name == "SliceObject":
            assert object_interacted_with is not None
            object_interaction_dict["slice"] = object_interacted_with
        elif action_name == "ToggleObjectOn":
            assert object_interacted_with is not None
            object_interaction_dict["toggle_on"] = object_interacted_with
        elif action_name == "ToggleObjecctOff":
            assert object_interacted_with is not None
            object_interaction_dict["toggle_off"] = object_interacted_with

        return object_interaction_dict


class PickupObjects(AI2ThorBaseEnv):
    """Gymnasium environment that defines a task to pick up as many
    objects as possible in an ai2thor kitchen environment.

    The agent gets rewarded for each UNIQUE object that it picks
    up (such that spamming drop pickup on the same object is not
    an optimal strategy).

    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        seed: int = 42,
        max_length: int = 10000,
        headless: bool = False,
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
            "ThrowObject",
        ]
        OBJECT_STR_TO_ID = {"": 0}
        super().__init__(
            action_names=ACTION_NAMES,
            scene="FloorPlan10",
            img_size=img_size,
            seed=seed,
            max_length=max_length,
            headless=headless,
        )

        # Contains all unique objects in the scene
        self.unique_objects: Set[str] = set()
        # Contains all the unique objects picked up this episode
        self.picked_up_unique_objects: Set[str] = set()

        # This gets populated with all the scene's objects
        # once reset() is called.
        self.log_rewards: Dict[str, int] = {}

        self.sliced_mutables: List[str] = ["Apple", "Bread", "Lettuce", "Potato", "Tomato"]
        self.cracked_mutables: List[str] = ["Egg"]

    def is_terminated(self, event) -> bool:
        # This environment terminates once every object
        # is picked up
        return len(self.unique_objects) > 0 and len(self.unique_objects) == len(
            self.picked_up_unique_objects
        )

    @property
    def observation_space(self) -> spaces.Dict:
        img_shape = self._image_size + (3,)
        my_spaces = {"image": spaces.Box(0, 255, img_shape, np.uint8)}
        my_spaces.update(
            {
                k: gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
                for k in self.log_rewards.keys()
            }
        )
        return spaces.Dict(my_spaces)

    def reset(self) -> Tuple[Dict, Dict]:

        self._step = 0
        self._done = False
        event = self.controller.reset()
        info = self.filter_metadata(event.metadata, is_first=True)
        self.update_agent_position(event.metadata)
        self.set_closest_objects(event.metadata)
        obs = self.process_obs(event, info, is_first=True)

        # Need to add objects that only appear once mutated
        # e.g., AppleSliced

        for object_meta in event.metadata["objects"]:
            self.unique_objects.add(object_meta["objectType"])
            if object_meta["objectType"] in self.sliced_mutables:
                self.unique_objects.add(f"{object_meta['objectType']}Sliced")
            if object_meta["objectType"] in self.cracked_mutables:
                self.unique_objects.add(f"{object_meta['objectType']}Cracked")
        


        for obj_name in self.unique_objects:
            self.log_rewards[obj_name] = 0

        self.picked_up_unique_objects = set()

        return obs, info

    def compute_reward(self, event) -> float:
        reward = 0.0
        for object_meta in event.metadata["objects"]:
            if (
                object_meta["isPickedUp"]
                and object_meta["objectType"] not in self.picked_up_unique_objects
            ):
                reward += 1.0
                self.picked_up_unique_objects.add(object_meta["objectType"])
                self.log_rewards[object_meta["objectType"]] = 1

        return reward

    def process_obs(
        self,
        event,
        filtered_meta: Dict[str, str],
        is_first: bool = False,
        done: bool = False,
    ) -> Dict:
        rgb_image: np.ndarray = event.frame

        h, w, _ = rgb_image.shape

        if (h, w) != self._image_size:
            image = Image.fromarray(rgb_image)
            image = image.resize(self._image_size, resample=Image.BILINEAR)  # type: ignore
            image = np.asarray(image)  # type: ignore
        else:
            image = rgb_image  # type: ignore

        return {
            "image": image,
            "is_terminal": done,
            "is_first": is_first,
            "agent_position": self.agent_position,
            **self.log_rewards,
            **filtered_meta,
        }

    def filter_metadata(
        self,
        metadata: Dict,
        object_interacted_with: Optional[str] = None,
        is_first: bool = False,
    ) -> Dict:
        """Filters the metadata returned by the environment
        to return a dictionary of data required for downstream tasks (i.e.,
        narration)

        Args:
            metadata (Dict): Metadata returned by the controller object

            object_interactied_with (Optional[str]): Optional name of the object that the
            agent interacted with in this environment step. Defaults to None.

            is_first (bool, optional): Whether this is the first observation of the
            episode. If true, returns the default null string dict, as
            no interactions could have happened. Defaults to False.

        Returns:
            Dict: Dictionary containing object interactions needed for
            generating the narrations.
        """
        object_interaction_dict = {
            "pickup": "",
            "drop": "",
            "open": "",
            "close": "",
            "break": "",
            "slice": "",
            "toggle_on": "",
            "toggle_off": "",
            "throw": "",
            "put": ("", ""),
        }

        # No interactions could have happened yet
        if is_first:
            return object_interaction_dict

        # If action didn't succeed then this is equivalent
        # to no interaction.
        if not metadata["lastActionSuccess"]:
            return object_interaction_dict

        # Go from object ID i.e., Sink|-00.70|+00.93|-00.65|SinkBasin
        # to object name i.e., Sink
        if object_interacted_with is not None:
            object_interacted_with = object_interacted_with.split("|")[0]

        action_name = metadata["lastAction"]

        if action_name == "PickupObject":
            assert object_interacted_with is not None
            object_interaction_dict["pickup"] = object_interacted_with
        elif action_name == "PutObject":
            assert object_interacted_with is not None
            receptacle = object_interacted_with
            object_placed = None
            for obj_meta in self.prev_meta["objects"]:
                if obj_meta["isPickedUp"]:
                    object_placed = obj_meta["objectType"]
                    break
            assert object_placed is not None
            object_interaction_dict["put"] = (object_placed, receptacle)

        elif action_name == "DropHandObject" or action_name == "ThrowObject":
            object_dropped = None
            for obj_meta in self.prev_meta["objects"]:
                if obj_meta["isPickedUp"]:
                    object_dropped = obj_meta["objectType"]
                    break
            assert object_dropped is not None
            if action_name == "DropHandObject":
                object_interaction_dict["drop"] = object_dropped
            else:
                object_interaction_dict["throw"] = object_dropped
        elif action_name == "OpenObject":
            assert object_interacted_with is not None
            object_interaction_dict["open"] = object_interacted_with
        elif action_name == "CloseObject":
            assert object_interacted_with is not None
            object_interaction_dict["close"] = object_interacted_with
        elif action_name == "BreakObject":
            assert object_interacted_with is not None
            object_interaction_dict["break"] = object_interacted_with
        elif action_name == "SliceObject":
            assert object_interacted_with is not None
            object_interaction_dict["slice"] = object_interacted_with
        elif action_name == "ToggleObjectOn":
            assert object_interacted_with is not None
            object_interaction_dict["toggle_on"] = object_interacted_with
        elif action_name == "ToggleObjecctOff":
            assert object_interacted_with is not None
            object_interaction_dict["toggle_off"] = object_interacted_with

        return object_interaction_dict


if __name__ == "__main__":
    env = CookEggEnv(headless=False)
    import time
    import json
    from pprint import pprint

    obs, info = env.reset()

    # with open("floor10-initial-conditions.json", "w") as dst:
    #     json.dump(info, dst)

    optimal_action_sequence = [
        20,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        19,
        9,
        17,
        0,
        11,
        22,
        22,
        0,
        21,  # "visible_objects": visible_objects,
        21,
        19,
        15,
        15,
        15,
        15,
        15,
        15,
        20,
        21,
        18,
        9,
        1,
        10,
        13,
        14,
        9,
        0,
    ]

    for episode in range(1):
        observations = []
        step_count = 0
        obs, info = env.reset()
        # visible_objects = []
        agent_positions = []
        object_interactions = []
        # visible_objects.append(obs["visible_objects"])
        agent_positions.append(obs["agent_position"])
        object_interactions.append(
            {
                "pickup": obs["pickup"],
                "drop": obs["drop"],
                "open": obs["open"],
                "close": obs["close"],
                "break": obs["break"],
                "slice": obs["slice"],
                "toggle_on": obs["toggle_on"],
                "toggle_off": obs["toggle_off"],
                "throw": obs["throw"],
                "put": obs["put"],
            }
        )

        observations.append(obs["image"])
        done = False
        # narrator = CookEggNarrator()
        cum_reward = 0.0
        while not done:
            action = np.zeros(env.action_space.n)
            # int_action = env.action_space.sample()
            int_action = int(input(f"Please enter action, 0-{env.action_space.n - 1}"))
            # int_action = optimal_action_sequence[step_count]
            action[int_action] = 1
            obs, reward, done, info = env.step(action)
            # visible_objects.append(obs["visible_objects"])
            # agent_positions.append(obs["agent_position"])
            # object_interactions.append(
            #     {
            #         "pickup": obs["pickup"],
            #         "drop": obs["drop"],
            #         "open": obs["open"],
            #         "close": obs["close"],
            #         "break": obs["break"],
            #         "slice": obs["slice"],
            #         "toggle_on": obs["toggle_on"],
            #         "toggle_off": obs["toggle_off"],
            #         "throw": obs["throw"],
            #         "put": obs["put"],
            #     }
            # )
            # if len(visible_objects) == 16:
            #     combined_dict = {}
            #     for d in object_interactions:
            #         for key, value in d.items():
            #             combined_dict.setdefault(key, []).append(value)
            #     print(narrator.narrate(visible_objects, agent_positions, combined_dict))
            #     visible_objects = []
            #     agent_positions = []
            #     object_interactions = []

            observations.append(obs["image"])
            step_count += 1
            cum_reward += 1
            time.sleep(0.1)
        # print(f"Episode {episode} reward: {cum_reward}")
        # combined_dict = {}
        # for d in object_interactions:        # combined_dict = {}
        # for d in object_interactions:
        #     for key, value in d.items():
        #         combined_dict.setdefault(key, []).append(value)
        # print(narrator.narrate(visible_objects, agent_positions, combined_dict))
        #     for key, value in d.items():
        #         combined_dict.setdefault(key, []).append(value)
        # print(narrator.narrate(visible_objects, agent_positions, combined_dict))

    # observations = np.array(observations)
    # print(observations.shape)
    # imgs = [Image.fromarray(img) for img in observations]
    # frame_durations = [250] * (len(imgs) - 1)
    # frame_durations.append(2500)
    # imgs[0].save(
    #     "optimal_policy.gif",
    #     save_all=True,
    #     append_images=imgs[1:],
    #     duration=frame_durations,
    #     loop=0,
    # )
