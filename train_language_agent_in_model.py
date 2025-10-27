import os
import pathlib
import sys
import cv2
from typing import Dict, List, Optional, Tuple, Union
import statistics

import wandb
import numpy as np
import torch
from torcheval.metrics.text import word_error_rate
from tqdm import tqdm

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
from evaluation import get_posterior_state, imagine_trajectory


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
            rgb_obs = [obs["high_res_image"]]
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
                rgb_obs.append(obs["high_res_image"])
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
                # print(f"Reward info: {info['reward_info']}")
                if self._manually_calculate_continues:
                    if (
                        self.language_goal == "go to the red key"
                        and info["reward_info"]["red key"] == 1.0
                    ):
                        done = True
                        eval_reward += 1.0
                    elif (
                        self.language_goal == "go to the green ball"
                        and info["reward_info"]["green ball"] == 1.0
                    ):
                        done = True
                        eval_reward += 1.0
                    elif (
                        self.language_goal == "go to the blue ball"
                        and info["reward_info"]["blue ball"] == 1.0
                    ):
                        done = True
                        eval_reward += 1.0
                    elif (
                        self.language_goal == "go to the purple box"
                        and info["reward_info"]["purple box"] == 1.0
                    ):
                        done = True
                        eval_reward += 1.0
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
        save_every: int = 100,
        eval_every: int = 100,
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

        lang_agent.train(
            language_goal,
            int(config.model_steps),
            logdir,
            rollout_length=config.imag_horizon,
        )


if __name__ == "__main__":
    main()
