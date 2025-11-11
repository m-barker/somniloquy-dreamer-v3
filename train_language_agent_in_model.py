import re
import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

import wandb
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from tools import (
    recursively_collect_optim_state_dict,
    recursively_load_optim_state_dict,
    Logger,
)
from somniloquy import (
    Dreamer,
    setup_args,
    setup,
    make_dataset,
    create_environments,
    load_existing_episodes,
    count_steps,
)
from evaluation import get_posterior_state, bleu_metric_from_strings


class LanguageAgent:
    """Trains a policy to achieve a specified natural language goal
    entirely within the Somniloquy world model.
    """

    _world_model: Dreamer

    def __init__(
        self,
        world_model: Dreamer,
        weights_path: str,
        eval_env,
        task: str,
        wandb_run=None,
        use_learned_reward: bool = False,
        reward_assignment_method: str = "final",
        reward_entailment_method: str = "substring",
        reward_model_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            world_model (Dreamer): Somniloquy model
            weights_path (str): path to the trained model weights
        """

        assert reward_assignment_method in ("all", "once", "final")
        assert reward_entailment_method in ("substring")

        self._world_model = world_model
        self._weights_path = weights_path
        self._eval_env = eval_env
        self._wandb_run = wandb_run

        self.rewards = []

        # If true, uses the learned reward head for the given natural
        # language goal. Used for comparison with language reward
        self._use_learned_reward = use_learned_reward

        # Determines which states are assigned a reward
        self._reward_assignment_method = reward_assignment_method
        # Determines how goal entailment from translation is decided
        self._reward_entailment_method = reward_entailment_method
        self._goal_reached_reward_model_threshold = reward_model_threshold

        self.task = task

        self._manually_calculate_continues = reward_assignment_method == "once"

        self._initialise_wm()

    def _initialise_wm(self) -> None:
        """
        Initialises the world model by loading the weights
        for the wm only (i.e., keep initiliased actor-critic weights)
        """
        weights = torch.load(self._weights_path)
        recursively_load_optim_state_dict(
            self._world_model,
            weights["optims_state_dict"],
            wm_only=True,
        )
        # Filter 'agent_state_dict' to remove actor-critic weights
        network_keys = weights["agent_state_dict"].keys()
        filtered_weights = {}
        keys_to_check = ["Actor", "actor", "Value", "value", "Critic", "critic"]
        for k in network_keys:
            remove_key = False
            for c in keys_to_check:
                if c in k:
                    remove_key = True
                    break
            if not remove_key:
                filtered_weights[k] = weights["agent_state_dict"][k]

        self._world_model.load_state_dict(filtered_weights, strict=False)

    def _generate_batch_translations(
        self, latent_state_batch: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> List[str]:
        """
        Generates a batch of translations given a batch of latent state sequences as
        input.

        Arguments:
            - latent_state_batch: Union[Dict[str, torch.Tensor], torch.Tensor:
              dictionary of latent state components, or stacked latent states
              (i.e., concatenated deterministic and stochastic components).
              Each is of shape (Batch length, Batch size).

        Returns:
            - List[str], list of string plan translations, one per batch.
        """
        if isinstance(latent_state_batch, Dict):
            stacked_states = self._world_model._wm.dynamics.get_feat(latent_state_batch)  # type: ignore
        else:
            assert isinstance(latent_state_batch, torch.Tensor)
            stacked_states = latent_state_batch
        # (T, B, D) -> (B, T, D)
        stacked_states = stacked_states.permute(1, 0, 2)
        plan_translations = self._world_model._wm.heads["language"].generate(
            stacked_states,
            self._world_model._wm.vocab,
            self._world_model._config.dec_max_length,
            self._world_model._config.token_sampling_method,
        )
        string_plan_translations = []
        for plan in plan_translations:
            plan_translation = " ".join(
                [
                    word
                    for word in plan.split()
                    if word not in ["<BOS>", "<EOS>", "<PAD>"]
                ]
            )
            string_plan_translations.append(plan_translation)
        return string_plan_translations

    def _make_prefix_batches(self, x: torch.Tensor, pad_value: int = -1):
        """
        Expand sequences into all prefixes, padded to full length.

        Args:
            x: Tensor of shape (seq_len, batch_size, D)
            pad_value: value used for padding

        Returns:
            padded: Tensor of shape (B, T, T, D) where B is batch size,
            padding_mask: Bool tensor of shape (B, T, T) with
                          True for padding positions, False for real tokens
        """
        seq_len, batch_size, D = x.shape
        num_prefixes = seq_len

        # Allocate padded output: (B, T, T, D)
        padded = torch.full(
            (batch_size, num_prefixes, seq_len, D),
            fill_value=pad_value,
            dtype=x.dtype,
            device=x.device,
        )

        # Fill each prefix
        for i in range(1, seq_len + 1):
            padded[:, i - 1, :i, :] = x[:i].permute(1, 0, 2)

        # Build padding mask (True where pad_value is present across feature dim)
        padding_mask = padded.eq(pad_value).all(
            dim=-1
        )  # (num_prefixes, seq_len, batch_size)

        return padded, padding_mask

    def _bleu_reward(
        self, latent_translation: str, threshold: Optional[float] = None
    ) -> float:
        """
        Computes the BLEU score between the language goal and the
        latent state translation, which is returned as a reward in
        the range of [0-1].

        Args:
            latent_translation (str): latent state translation

            threshold (optional, float): Optional float threshold used to only return
            a non-zero reward if bleu score is above this threshold. Defaults to None.
        """

        goal_str = self.language_goal
        bleu_score: torch.Tensor = bleu_metric_from_strings(
            latent_translation, goal_str
        )
        if threshold:
            if float(bleu_score) < threshold:
                return 0.0
        return float(bleu_score)

    def _sentence_embed_reward(
        self,
        latent_translation: Union[str, List[str]],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        threshold: Optional[float] = None,
    ) -> List[float]:
        model = SentenceTransformer(embedding_model_name)
        goal_embedding = model.encode([self.language_goal])
        if isinstance(latent_translation, str):
            latent_translation = [latent_translation]
        translation_embeddings = model.encode(latent_translation)

        # Shape (1, N) where N is the number of translations
        similarities = model.similarity(goal_embedding, translation_embeddings)
        # Shape (N)
        similarities = similarities.squeeze(0)
        cosine_scores = similarities.tolist()

        if threshold:
            cosine_scores = [s if s > threshold else 0.0 for s in cosine_scores]

        return cosine_scores

    def _entailment_model_reward(
        self,
        latent_translation: Union[str, List[str]],
        entailment_model_name: str = "facebook/bart-large-mnli",
        threshold: Optional[float] = None,
    ) -> List[float]:
        """
        Calculates the degree to which the natural language goal is semantically
        entailed by the latent state translations, using a pre-trained NLI model.

        Args:
            latent_translation (Union[str, List[str]]): latent state translations
            output by Somniloquy

            entailment_model_name (optional, str): Name of the entailment model to
            use. Must be available on HuggingFace and compatable with the API used
            in this function. Defaults to "facebook/bart-large-mnli", a 400M parameter
            model.

            threshold (optional, float): If not None, sets a minimum confidence
            entailment threshold for non-zero reward. Defaults to None.
        """
        nli = pipeline(
            "text-classification",
            model=entailment_model_name,
            top_k=None,
        )

        hypothesis = self.language_goal

        if isinstance(latent_translation, str):
            latent_translation = [latent_translation]

        pairs = [f"{premise} </s><s> {hypothesis}" for premise in latent_translation]

        results = nli(pairs)

        if results:
            reward = [
                float(res["score"]) if res["label"] == "entailment" else 0.0
                for res in results
            ]
        else:
            raise ValueError("No entailment results found")

        if threshold:
            reward = [r if r > threshold else 0.0 for r in reward]

        return reward

    def _language_reward(
        self,
        _,
        imagined_states: Dict[str, torch.Tensor],
        __,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.language_goal is not None
        # Imagined states are of shape (Time, Batch, Dimension)
        T, B, _ = imagined_states["deter"].shape
        stacked_states = self._world_model._wm.dynamics.get_feat(imagined_states)  # type: ignore
        # (T, B, D) -> (B, T, D)
        stacked_states = stacked_states.permute(1, 0, 2)

        # Reward needs to match the (Time, Batch) of the stacked states
        # Dimension is 1 as reward is a scalar
        reward = torch.zeros((T, B, 1))
        continues = (
            torch.ones_like(reward) if self._manually_calculate_continues else None
        )

        if self._reward_assignment_method == "final":
            plan_translations = self._world_model._wm.heads["language"].generate(
                stacked_states,  # shape (B, T, D)
                self._world_model._wm.vocab,
                self._world_model._config.dec_max_length,
                self._world_model._config.token_sampling_method,
            )
            string_plan_translations = [
                " ".join(
                    [
                        word
                        for word in plan.split()
                        if word not in ["<BOS>", "<EOS>", "<PAD>"]
                    ]
                )
                for plan in plan_translations
            ]
            for b, plan_translation in enumerate(string_plan_translations):
                if self._reward_entailment_method == "substring":
                    if self.language_goal in plan_translation:
                        reward[-1, b] = 1.0

        elif self._reward_assignment_method in ("once", "all"):
            # Shape (B, T, T, D), (B, T, T)
            sequence_permutations, sequence_pad_mask = self._make_prefix_batches(
                stacked_states.permute(1, 0, 2)
            )

            # Collapse the batch dimension into the sequence dimension for one pass
            # Original shapes: (batch, T, B, D) and (batch, T, B)
            flat_sequences = sequence_permutations.reshape(
                -1, sequence_permutations.shape[2], sequence_permutations.shape[3]
            )  # shape (B*batch, T, D)
            flat_padding = sequence_pad_mask.reshape(
                -1, sequence_pad_mask.shape[2]
            )  # shape (B*batch, T)

            # Generate translations in a single pass
            plan_translations = self._world_model._wm.heads["language"].generate(
                flat_sequences,  # shape (B*batch, T, D)
                self._world_model._wm.vocab,
                self._world_model._config.dec_max_length,
                self._world_model._config.token_sampling_method,
                src_padding_mask=flat_padding,  # shape (B*batch, T)
            )

            # Post-process translations
            string_plan_translations = [
                " ".join(
                    [
                        word
                        for word in plan.split()
                        if word not in ["<BOS>", "<EOS>", "<PAD>"]
                    ]
                )
                for plan in plan_translations
            ]

            # Reshape back into (batch, B) layout
            string_plan_translations_array = (
                np.array(string_plan_translations).reshape(B, T).T
            )  # shape (T, B)

            # Reward assignment
            for batch in range(B):
                for idx, plan_translation in enumerate(
                    string_plan_translations_array[:, batch]
                ):
                    if self._reward_entailment_method == "substring":
                        if self.language_goal in plan_translation:
                            reward[idx][batch] = 1.0
                            # Reward only the first time the goal is reached
                            if continues is not None:
                                continues[idx:, batch] = 0.0
                            if self._reward_assignment_method == "once":
                                break

        reward = reward.to(self._world_model._config.device)
        if continues is not None:
            continues = continues.to(self._world_model._config.device)
        if self._wandb_run is not None:
            mean_reward_batch = torch.mean(reward, dim=1)
            plan_reward = float(mean_reward_batch.sum())
            log_name = "reward_in_model"
            self._wandb_run.log(
                {
                    log_name: plan_reward,
                }
            )

        return reward, continues

    def _get_env_starting_state(self):
        """
        Gets the starting latent state of the evaluation environment
        """
        obs, info = self._eval_env.reset()()
        no_convert = self._world_model._config.no_convert_list
        obs_to_ignore = self._world_model._config.ignore_list
        starting_latent = get_posterior_state(
            self._world_model,
            obs,
            no_convert,
            obs_to_ignore,
        )
        latent_tensor = self._world_model._wm.dynamics.get_feat(
            starting_latent
        ).unsqueeze(0)
        return latent_tensor, starting_latent, obs, no_convert, obs_to_ignore

    def play_video(self, array, fps=2):
        """
        Play a numpy video array of shape (T, C, H, W) in a window until closed.
        """
        T, C, H, W = array.shape
        array = array.transpose(0, 2, 3, 1)  # (T, H, W, C)
        delay = int(1000 / fps)  # ms per frame

        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        t = 0
        while True:
            frame = array[t]

            # Ensure uint8 and BGR for OpenCV
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if C == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Video", frame)
            key = cv2.waitKey(delay)

            # quit if 'q' pressed or window closed
            if (
                key == ord("q")
                or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1
            ):
                break

            # advance to next frame, loop back if needed
            t = (t + 1) % T

        cv2.destroyAllWindows()

    def _calculate_babyai_true_reward(self, env_info: Dict) -> Tuple[bool, float]:
        """n
        Calculates the ground-truth reward in the babyai environment for the given natural
        language goal

        Args:
            env_info (Dict): info dict returned by the envrionment after each step

        Returns:
            Tuple(bool, float): termination, reward
        """
        done = False
        eval_reward = 0
        if (
            self.language_goal == "go to the red key"
            and env_info["reward_info"]["red key"] == 1.0
        ):
            done = True
        elif (
            self.language_goal == "go to the green ball"
            and env_info["reward_info"]["green ball"] == 1.0
        ):
            done = True
        elif (
            self.language_goal == "go to the blue ball"
            and env_info["reward_info"]["blue ball"] == 1.0
        ):
            done = True
        elif (
            self.language_goal == "go to the purple box"
            and env_info["reward_info"]["purple box"] == 1.0
        ):
            done = True
        if done:
            eval_reward += 1
        return done, eval_reward

    def _calculate_crafter_true_reward(self, env_info: Dict) -> Tuple[bool, float]:
        """n
        Calculates the ground-truth reward in the crafter environment for the given natural
        language goal

        Args:
            env_info (Dict): info dict returned by the envrionment after each step
        Returns:
            Tuple(bool, float): termination, reward
        """

        assert {"semantic", "inventory", "achievements"}.issubset(set(env_info.keys()))

        done = False
        eval_reward = 0

        if "harvest" in self.language_goal:
            params = re.search(r"\bharvest\s+(\d+)\s+(\w+)\b", self.language_goal)
            if params:
                harvest_goal_count = params.group(1)
                harvest_goal_material = params.group(2)
                true_harvest_count = env_info["inventory"][harvest_goal_material]
                done = harvest_goal_count == true_harvest_count
            else:
                raise ValueError(f"Can't figure out language goal {self.language_goal}")
        elif "place" in self.language_goal:
            params = re.search(r"\bplace\s+(\d+)\s+(\w+)\b", self.language_goal)
            if params:
                place_goal_count = int(params.group(1))
                place_goal_object = params.group(2)
                # Strip plural from goal
                if place_goal_object[-1] == "s":
                    place_goal_object = place_goal_object[:-1]
                true_place_count = env_info["achievements"][
                    f"place_{place_goal_object}"
                ]
                done = true_place_count == place_goal_count
            else:
                raise ValueError(
                    f"Can't handle place language goal: {self.language_goal}"
                )
        elif "drink" in self.language_goal:
            params = re.search(r"\bdrink\s+(\d+)\s+waters?\b", self.language_goal)
            if params:
                drink_goal_amount = int(params.group(1))
                true_drink_amount = env_info["achievements"]["collect_drink"]
                done = true_drink_amount == drink_goal_amount
            else:
                raise ValueError(
                    f"Can't handle drink language goal: {self.language_goal}"
                )
        elif "eat" in self.language_goal:
            params = re.search(r"\beat\s+(\d+)\s+(\w+)\b", self.language_goal)
            if params:
                eat_goal_amount = int(params.group(1))
                eat_goal_object = params.group(2)
                if eat_goal_object[-1] == "s":
                    eat_goal_object = eat_goal_object[:-1]
                true_eat_amount = env_info["achievements"][f"eat_{eat_goal_object}"]
                done = true_eat_amount == eat_goal_amount
            else:
                raise ValueError(f"Unhandled eat goal: {self.language_goal}")
        elif "craft" in self.language_goal:
            params = re.search(r"\bcraft\s+(\d+)\s+([a-zA-Z ]+)\b", self.language_goal)
            if params:
                craft_goal_amount = int(params.group(1))
                craft_goal_object = params.group(2)
                true_craft_amount = env_info["achievements"][
                    f"make_{craft_goal_object.replace(' ', '_')}"
                ]
                done = true_craft_amount == craft_goal_amount
            else:
                raise ValueError(f"Unhandled craft goal: {self.language_goal}")

        else:
            raise ValueError(
                f"Unhandled language goal for Crafter: {self.language_goal}"
            )

        if done:
            eval_reward += 1
        return done, eval_reward

    def _eval(
        self, horizon: int = 15, n_eval_episodes: int = 10
    ) -> Tuple[np.ndarray, float]:
        print("Beginning Evaluation....")
        assert n_eval_episodes > 0
        video_array = None
        mean_episode_reward = 0.0
        for episode in range(n_eval_episodes):
            latent_tensor, starting_latent, obs, no_convert, obs_to_ignore = (
                self._get_env_starting_state()
            )
            image_key = "high_res_image" if "high_res_image" in obs.keys() else "image"
            rgb_obs = [obs[image_key]]
            prev_state = starting_latent
            eval_reward = 0.0
            done = False
            for t in range(horizon):
                if done:
                    break
                action = (
                    self._world_model._task_behavior.actor(latent_tensor)
                    .mode()
                    .squeeze(0)
                )
                action_dict = {"action": action.squeeze(0).detach().cpu().numpy()}
                obs, reward, done, info = self._eval_env.step(action_dict)()
                eval_reward += reward
                rgb_obs.append(obs[image_key])
                posterior = get_posterior_state(
                    self._world_model,
                    obs,
                    no_convert,
                    obs_to_ignore,
                    prev_state,
                    action,
                )
                latent_tensor = self._world_model._wm.dynamics.get_feat(
                    posterior
                ).unsqueeze(0)
                prev_state = posterior
                print(
                    f"State Value: {self._world_model._task_behavior.value(latent_tensor).mode()}"
                )
                goal_achieved = False
                if "babyai" in self.task:
                    goal_achieved, step_reward = self._calculate_babyai_true_reward(
                        info
                    )
                    eval_reward += step_reward
                elif "crafter" in self.task:
                    goal_achieved, eval_reward = self._calculate_crafter_true_reward(
                        info
                    )
                done = done or goal_achieved
            # (T, H, W, C)
            video_array = np.stack(rgb_obs, axis=0)
            # (T, C, H, W)
            video_array = video_array.transpose(0, 3, 1, 2)
            mean_episode_reward += eval_reward
            print(f"Evaluation Episode {episode + 1}: {eval_reward}")
        assert video_array is not None
        return video_array, mean_episode_reward / n_eval_episodes

    def _learned_reward_function(
        self, _, imagined_states, __
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the rewards using the learned reward function head
        """

        stacked_latents = self._world_model._wm.dynamics.get_feat(imagined_states)

        reward = self._world_model._wm.heads[self._reward_head_name](
            stacked_latents
        ).mode()

        continues = (
            torch.ones_like(reward) if self._manually_calculate_continues else None
        )

        if continues is not None:
            threshold = self._goal_reached_reward_model_threshold
            T, B, _ = reward.shape
            for b in range(B):
                for t in range(T):
                    # Only one a non-zero reward the first time the goal is reached
                    # In practice, this means if r(s) > epsilon.
                    # reward and continues are of shape (T, B, 1)
                    if reward[t, b] > threshold:
                        reward[t + 1 :, b] = 0.0
                        continues[t:, b] = 0.0
                        break

        if self._wandb_run is not None:
            mean_reward_batch = torch.mean(reward, dim=1)
            plan_reward = float(mean_reward_batch.sum())
            log_name = "reward_in_model"
            self._wandb_run.log(
                {
                    log_name: plan_reward,
                }
            )

        return reward, continues

    def train(
        self,
        language_goal: str,
        n_training_steps: int,
        logdir: pathlib.Path,
        batch_size: int = 64,
        rollout_length: int = 15,
        start_state: Optional[Dict] = None,
        save_every: int = 500,
        eval_every: int = 500,
        n_eval_episodes: int = 10,
        display_video: bool = False,
    ) -> None:
        language_goal = language_goal.lower()
        self.language_goal = language_goal
        for n in range(n_training_steps):
            # If None, make it the starting latent of the environment from the initial observation
            if start_state is None:
                _, start_state, _, _, _ = self._get_env_starting_state()
            # Batchify the start state so we can compute multiple model rollouts for training
            # the policy
            # Start state is a dictionary whose keys are of shape (Batch, *Dimension)
            start_state_batched = {}
            for k, v in start_state.items():
                start_state_batched[k] = (
                    v.repeat(batch_size, 1)
                    if len(v.shape) == 2
                    else v.repeat(batch_size, 1, 1)
                )
            # Add empty time dimension
            for k, v in start_state_batched.items():
                start_state_batched[k] = v.unsqueeze(0)
            if self._use_learned_reward:
                self._reward_head_name = ""
                if self.language_goal == "go to the red key":
                    self._reward_head_name = "red_key_reward"
                elif self.language_goal == "go to the green ball":
                    self._reward_head_name = "green_ball_reward"
                elif self.language_goal == "go to the blue ball":
                    self._reward_head_name = "blue_ball_reward"
                elif self.language_goal == "go to the purple box":
                    self._reward_head_name = "purple_box_reward"
                else:
                    raise ValueError(
                        f"No reward head available for language goal {self.language_goal}"
                    )
                reward_func = self._learned_reward_function
            else:
                reward_func = self._language_reward
            self._world_model._task_behavior._train(
                start_state_batched,
                reward_func,
                reward_returns_continue=self._manually_calculate_continues,
            )  # type: ignore
            if n % save_every == 0:
                items_to_save = {
                    "agent_state_dict": self._world_model.state_dict(),
                    "optims_state_dict": recursively_collect_optim_state_dict(
                        self._world_model
                    ),
                }
                torch.save(items_to_save, os.path.join(logdir, "language_agent.pt"))
            if n % eval_every == 0 and n > 0:
                with torch.no_grad():
                    eval_policy_video, eval_reward = self._eval(
                        horizon=rollout_length, n_eval_episodes=n_eval_episodes
                    )
                if self._wandb_run is not None:
                    log_reward_name = "mean_eval_reward"
                    self._wandb_run.log(
                        {
                            "eval_policy": wandb.Video(
                                eval_policy_video, fps=2, format="mp4"
                            ),
                            log_reward_name: eval_reward,
                        },
                        step=n + 1,  # wandb steps start at 1
                    )
                if display_video:
                    self.play_video(eval_policy_video)

    def _display_image(self, image: np.ndarray) -> None:
        """
        Displays the RGB image to the screen

        Args:
        image (np.ndarray) of shape (H,W,C)
        """

        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Image", bgr)

        # 30–50 ms wait keeps GUI alive but doesn't block the loop
        key = cv2.waitKey(30)
        if key == 27:  # optional: press Esc to close
            cv2.destroyAllWindows()

    def play_game_inside_model(self) -> None:
        """
        Allows a user to control the agent fully inside of the world model.
        Useful for debugging.
        """
        latent_tensor, starting_latent, obs, no_convert, obs_to_ignore = (
            self._get_env_starting_state()
        )
        latent_state = starting_latent
        done = False
        latent_tensors: List[torch.Tensor] = [latent_tensor]
        image = obs["image"]
        while not done:
            self._display_image(image)
            valid_action = False
            print(f"Number of latent tensors so far: {len(latent_tensors)}")
            translation = self._world_model._wm.heads["language"].generate(
                torch.cat(latent_tensors, dim=0).permute(1, 0, 2),
                self._world_model._wm.vocab,
                self._world_model._config.dec_max_length,
                self._world_model._config.token_sampling_method,
            )[0]
            translation = " ".join(
                [
                    word
                    for word in translation.split()
                    if word not in ["<BOS>", "<EOS>", "<PAD>"]
                ]
            )
            print(f"Translator says this has happened: {translation}")
            while not valid_action:
                action_str = input("Please enter an action: ")
                try:
                    action_int = int(action_str)
                    action_arr = np.zeros(self._eval_env.action_space.n)
                    action_arr[action_int] = 1
                    # Shape (1,1 N)
                    action_tensor = (
                        torch.Tensor(action_arr)
                        .unsqueeze(0)
                        .to(self._world_model._config.device)
                    )
                    latent_state = self._world_model._wm.dynamics.img_step(
                        prev_state=latent_state,
                        prev_action=action_tensor,
                    )
                    valid_action = True
                except Exception as e:
                    print(f"Invalid action entered: {e}")
                    valid_action = False

            latent_tensor = self._world_model._wm.dynamics.get_feat(latent_state)
            latent_tensors.append(latent_tensor.unsqueeze(0))
            decoded_image = (
                self._world_model._wm.heads["decoder"](latent_tensor.unsqueeze(0))[
                    "image"
                ]
                .mode()
                .detach()
                .clone()
            )
            decoded_image = torch.clip(255 * decoded_image, 0, 255).to(torch.uint8)
            image = decoded_image[0, 0].detach().cpu().contiguous().numpy().copy()


def load_agent():
    """Loads a pre-trained Dreamer world-model"""
    init_config = setup_args()

    config, logdir = setup(init_config)
    step = count_steps(config.traindir)
    # step in logger is environmental step

    if not config.wandb:
        run = wandb.init(mode="disabled")
    else:
        run = wandb.init(
            project="somniloquy",
            config=vars(config),
        )
    logger = Logger(logdir, config.action_repeat * step, wandb_run=run)
    train_envs, eval_envs = create_environments(config)
    train_eps, _ = load_existing_episodes(config)
    eval_env = eval_envs[0]

    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    train_dataset = make_dataset(train_eps, config)

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)

    return config, agent, eval_env, run, logdir


def main():
    config, agent, eval_env, run, logdir = load_agent()
    reward_assignment_method = config.reward_assignment_method
    reward_entailment_method = config.reward_entailment_method
    if sys.argv[-1] == "instruct":
        while True:
            config, agent, eval_env, run, logdir = load_agent()
            lang_agent = LanguageAgent(
                agent,
                config.checkpoint,
                eval_env,
                config.task,
                run,
                use_learned_reward=config.use_learned_reward,
                reward_assignment_method=reward_assignment_method,
                reward_entailment_method=reward_entailment_method,
            )
            language_goal = input("What would you like me to do?")
            lang_agent.train(
                language_goal,
                150,
                logdir,
                config.imag_horizon,
                eval_every=149,
                display_video=True,
            )

    else:
        print(f"Use Learned Reward: {config.use_learned_reward}")
        lang_agent = LanguageAgent(
            agent,
            config.checkpoint,
            eval_env,
            config.task,
            run,
            use_learned_reward=config.use_learned_reward,
            reward_assignment_method=reward_assignment_method,
            reward_entailment_method=reward_entailment_method,
        )
        # For ease of passing, config language goal can be a single word, now map
        # it to the proper goal
        language_goal = None
        if config.language_goal == "red":
            language_goal = "go to the red key"
        elif config.language_goal == "green":
            language_goal = "go to the green ball"
        elif config.language_goal == "blue":
            language_goal = "go to the blue ball"
        elif config.language_goal == "purple":
            language_goal = "go to the purple box"
        else:
            language_goal = config.language_goal

        if sys.argv[-1] == "play":
            lang_agent.play_game_inside_model()
        else:
            lang_agent.train(
                language_goal,
                int(config.model_steps),
                logdir,
                rollout_length=config.imag_horizon,
            )


if __name__ == "__main__":
    main()
