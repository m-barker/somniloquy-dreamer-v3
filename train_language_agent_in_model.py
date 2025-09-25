import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union
import statistics

import wandb
import numpy as np
import torch
from torcheval.metrics.text import word_error_rate

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
from evaluation import get_posterior_state


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
        wandb_run=None,
        manually_calculate_continues: bool = False,
    ) -> None:
        """
        Args:
            world_model (Dreamer): Somniloquy model
            weights_path (str): path to the trained model weights
        """
        self._world_model = world_model
        self._weights_path = weights_path
        self._eval_env = eval_env
        self._wandb_run = wandb_run
        self._manually_calculate_continues = manually_calculate_continues

        self.rewards = []

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

    def _language_reward(
        self,
        _,
        imagined_states: Dict[str, torch.Tensor],
        __,
    ) -> torch.Tensor:
        assert self.language_goal is not None
        g = self.language_goal
        # Imagined states are of shape (Time, Batch, Dimension)
        T, B, _ = imagined_states["stoch"].shape

        string_plan_translations = self._generate_batch_translations(imagined_states)

        # Reward needs to match the (Time, Batch) of the stacked states
        # Dimension is 1 as reward is a scalar
        reward = torch.zeros((T, B, 1))

        # At the moment, set the reward for the final state
        # as 1.0 if goal is reached, 0 everywhere else.
        for t, p in enumerate(string_plan_translations):
            if p == g:
                reward[-1, t] = 1.0

        reward = reward.to(self._world_model._config.device)
        if self._wandb_run is not None:
            mean_reward_batch = torch.mean(reward, dim=1)
            plan_reward = float(mean_reward_batch.sum())
            self._wandb_run.log(
                {
                    "reward_in_model": plan_reward,
                }
            )

        return reward

    def _make_prefix_batches(x: torch.Tensor, pad_value: int = -1):
        """
        Expand sequences into all prefixes, padded to full length.

        Args:
            x: Tensor of shape (seq_len, batch_size, D)
            pad_value: value used for padding

        Returns:
            padded: Tensor of shape (num_prefixes, batch_size, seq_len, D)
            padding_mask: Bool tensor of shape (num_prefixes, batch_size, seq_len)
                          True for padding positions, False for real tokens
        """
        seq_len, batch_size, D = x.shape
        num_prefixes = seq_len

        # Allocate padded output: (num_prefixes, seq_len, batch_size, D)
        padded = torch.full(
            (num_prefixes, seq_len, batch_size, D),
            fill_value=pad_value,
            dtype=x.dtype,
            device=x.device,
        )

        # Fill each prefix
        for i in range(1, seq_len + 1):
            padded[i - 1, :i] = x[:i]

        # Build padding mask (True where pad_value is present across feature dim)
        padding_mask = padded.eq(pad_value).all(
            dim=-1
        )  # (num_prefixes, seq_len, batch_size)

        return padded, padding_mask

    def _babyai_language_reward(
        self,
        _,
        imagined_states: Dict[str, torch.Tensor],
        __,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.language_goal is not None
        g = self.language_goal
        # Imagined states are of shape (Time, Batch, Dimension)
        T, B, _ = imagined_states["stoch"].shape
        stacked_states = self._world_model._wm.dynamics.get_feat(imagined_states)  # type: ignore
        # (T, B, D) -> (B, T, D)
        stacked_states = stacked_states.permute(1, 0, 2)

        # Reward needs to match the (Time, Batch) of the stacked states
        # Dimension is 1 as reward is a scalar
        reward = torch.zeros((T, B, 1))
        continues = torch.ones_like(reward)

        # Shape (T, T, B, D), (T, T, B)
        sequence_permutations, sequence_pad_mask = self._make_prefix_batches(
            stacked_states.permute(0, 1, 2)
        )

        for batch in range(B):
            # This can be viewed as now being of shape (batch, T, D)
            sequences_to_translate = sequence_permutations[:, :, batch, :]
            sequences_padding = sequence_pad_mask[:, :, batch]
            plan_translations = self._world_model._wm.heads["language"].generate(
                sequences_to_translate,
                self._world_model._wm.vocab,
                self._world_model._config.dec_max_length,
                self._world_model._config.token_sampling_method,
                src_padding_mask=sequences_padding,
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
            # Now, if the goal is every reached, allocate it to the
            # state that caused this in translation space.
            # We are iterating over the batches, which really are
            # translations of the sequences in order:
            # [0]
            # [0,1]
            # [0,1,2]
            # [0,1,2,3]
            # ...
            # [0,1,2,...,T-1]
            for idx, plan_translation in enumerate(string_plan_translations):
                if g in plan_translation:
                    reward[idx][batch] = 1.0
                    continues[idx:, batch] = 0.0
                    break

        reward = reward.to(self._world_model._config.device)
        if self._wandb_run is not None:
            mean_reward_batch = torch.mean(reward, dim=1)
            plan_reward = float(mean_reward_batch.sum())
            self._wandb_run.log(
                {
                    "reward_in_model": plan_reward,
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
            rgb_obs = [obs["image"]]
            prev_state = starting_latent
            eval_reward = 0.0
            for t in range(horizon):
                action = (
                    self._world_model._task_behavior.actor(latent_tensor)
                    .mode()
                    .squeeze(0)
                )
                action_dict = {"action": action.squeeze(0).detach().cpu().numpy()}
                obs, reward, done, info = self._eval_env.step(action_dict)()
                eval_reward += reward
                rgb_obs.append(obs["image"])
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
            # (T, H, W, C)
            video_array = np.stack(rgb_obs, axis=0)
            # (T, C, H, W)
            video_array = video_array.transpose(0, 3, 1, 2)
            mean_episode_reward += eval_reward
            print(f"Evaluation Episode {episode + 1}: eval_reward")
        assert video_array is not None
        return video_array, mean_episode_reward / n_eval_episodes

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
    ) -> None:
        language_goal = language_goal.lower()
        self.language_goal = language_goal
        for n in range(n_training_steps):
            # If None, make it the starting latent of the environment from the initial observation
            if start_state is None:
                _, start_state, _, _, _ = self._get_env_starting_state()
            # Batchify the start state so we can compute multiple model rollouts for training
            # the policy
            # Start state is a dictionary whose keys are of shape (Batch, Dimension)
            start_state_batched = {
                k: v.repeat(batch_size, 1) for k, v in start_state.items()
            }
            self._world_model._task_behavior._train(
                start_state_batched,
                self._language_reward,
                self._manually_calculate_continues,
            )  # type: ignore
            if n % save_every == 0:
                items_to_save = {
                    "agent_state_dict": self._world_model.state_dict(),
                    "optims_state_dict": recursively_collect_optim_state_dict(
                        self._world_model
                    ),
                }
                torch.save(items_to_save, os.path.join(logdir, "language_agent.pt"))
            if n % eval_every == 0:
                with torch.no_grad():
                    eval_policy_video, eval_reward = self._eval(horizon=rollout_length)
                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {
                            "eval_policy": wandb.Video(
                                eval_policy_video, fps=2, format="mp4"
                            ),
                            "eval_reward": eval_reward,
                        },
                        step=n + 1,  # wandb steps start at 1
                    )


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
    lang_agent = LanguageAgent(agent, config.checkpoint, eval_env, run)
    lang_agent.train(config.language_goal, config.model_steps, logdir)


if __name__ == "__main__":
    main()
