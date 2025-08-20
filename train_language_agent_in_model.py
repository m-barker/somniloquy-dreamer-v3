from typing import Dict, List, Optional
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

    def _language_reward(
        self,
        _,
        imagined_states: Dict[str, torch.Tensor],
        __,
    ) -> torch.Tensor:
        assert self.language_goal is not None
        g = self.language_goal
        stacked_states = self._world_model._wm.dynamics.get_feat(imagined_states)  # type: ignore
        # (T, B, D) -> (B, T, D)
        stacked_states = stacked_states.permute(1, 0, 2)
        plan_translation = self._world_model._wm.heads["language"].generate(
            stacked_states,
            self._world_model._wm.vocab,
            self._world_model._config.dec_max_length,
            self._world_model._config.token_sampling_method,
        )[0]  # Take first element of batch as batch size is 1
        plan_translation = " ".join(
            [
                word
                for word in plan_translation.split()
                if word not in ["<BOS>", "<EOS>", "<PAD>"]
            ]
        )
        # print(f"Plan Translation: {plan_translation}")
        reward = torch.zeros((15, 1, 1))

        if plan_translation == g:
            reward[-1, 0] = 1.0

        # self.rewards.append(float(reward.mean()))
        # print(f"Mean Reward: {statistics.mean(self.rewards)}")
        reward = reward.to(self._world_model._config.device)
        if self._wandb_run is not None:
            plan_reward = float(reward.sum())
            self._wandb_run.log(
                {
                    "plan_reward": plan_reward,
                }
            )
        # print(f"Plan Reward: {reward}")
        return reward

    def _eval(self) -> np.ndarray:
        obs, info = self._eval_env.reset()()
        rgb_obs = [obs["image"]]
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
        prev_state = starting_latent
        for t in range(15):
            action = (
                self._world_model._task_behavior.actor(latent_tensor).mode().squeeze(0)
            )
            action_dict = {"action": action.squeeze(0).detach().cpu().numpy()}
            obs, reward, done, info = self._eval_env.step(action_dict)()
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
        return video_array

    def train(
        self,
        language_goal: str,
        n_training_steps: int,
        rollout_length: int = 15,
        start_state: Optional[torch.Tensor] = None,
        save_every: int = 100,
        eval_every: int = 100,
    ) -> None:
        language_goal = language_goal.lower()
        self.language_goal = language_goal
        for n in range(n_training_steps):
            # TODO: change this to initial latent state of the environment
            start_state = self._world_model._wm.dynamics.initial(1)
            for k, v in start_state.items():
                start_state[k] = v.unsqueeze(0)
            self._world_model._task_behavior._train(start_state, self._language_reward)
            if n % save_every == 0:
                items_to_save = {
                    "agent_state_dict": self._world_model.state_dict(),
                    "optims_state_dict": recursively_collect_optim_state_dict(
                        self._world_model
                    ),
                }
                torch.save(items_to_save, "language_agent.pt")
            if n % eval_every == 0:
                with torch.no_grad():
                    eval_policy_video = self._eval()
                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {
                            "eval_policy": wandb.Video(
                                eval_policy_video, fps=2, format="mp4"
                            )
                        },
                        step=n + 1,  # wandb steps start at 1
                    )


def main():
    import wandb

    init_config = setup_args()

    config, logdir = setup(init_config)
    step = count_steps(config.traindir)
    # step in logger is environmental step

    if not config.wandb:
        run = wandb.init(mode="disabled")
    else:
        run = wandb.init(
            project="somniloquy",
            config=config,
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
    lang_agent = LanguageAgent(agent, config.checkpoint, eval_env, run)
    lang_agent.train("i will reach the blue square", 10000000)


if __name__ == "__main__":
    main()
