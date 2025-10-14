import numpy as np
import torch
import matplotlib.pyplot as plt

from train_language_agent_in_model import load_agent
from evaluation import get_posterior_state
from tools import recursively_load_optim_state_dict


def get_env_starting_state(agent, eval_env):
    obs, info = eval_env.reset()()
    no_convert = agent._config.no_convert_list
    obs_to_ignore = agent._config.ignore_list
    starting_latent = get_posterior_state(
        agent,
        obs,
        no_convert,
        obs_to_ignore,
    )
    latent_tensor = agent._wm.dynamics.get_feat(starting_latent).unsqueeze(0)
    return latent_tensor, starting_latent, obs, no_convert, obs_to_ignore


def get_latent_state_value(agent, latent_state) -> float:
    """
    Assumes that the latent state is for a single batch/sequence
    """
    value_head = agent._task_behavior.value
    state_value = value_head(latent_state).mode()

    return float(torch.squeeze(state_value))


def visualise_values(agent, eval_env) -> None:
    latent_tensor, starting_latent, obs, no_convert, obs_to_ignore = (
        get_env_starting_state(agent, eval_env)
    )
    prev_latent = starting_latent

    done = False
    state_idx = 0
    while not done:
        state_value = get_latent_state_value(agent, latent_tensor)
        rgb_obs = obs["high_res_image"]
        plt.imshow(rgb_obs)
        plt.text(
            20,
            40,  # (x, y) coordinates in pixels
            f"{state_value:.2f}",  # the number to display
            color="red",  # text color
            fontsize=16,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.3"),
        )
        plt.axis("off")
        plt.savefig(f"Value of State : {state_idx}.png")
        plt.close()

        action = int(input("Please enter an action"))
        action_arr = np.zeros(eval_env.action_space.n)
        action_arr[action] = 1
        action_tensor = torch.Tensor(action_arr).to(torch.device("cuda")).unsqueeze(0)
        env_action = {"action": action_arr}
        obs, reward, done, info = eval_env.step(env_action)()
        prev_latent = get_posterior_state(
            agent,
            obs,
            no_convert,
            obs_to_ignore,
            prev_latent,
            action_tensor,
        )
        latent_tensor = agent._wm.dynamics.get_feat(prev_latent).unsqueeze(0)
        state_idx += 1


def load_wm(config, agent) -> None:
    """
    Initialises the world model by loading the weights
    for the wm only (i.e., keep initiliased actor-critic weights)
    """
    weights = torch.load(config.checkpoint)
    recursively_load_optim_state_dict(
        agent,
        weights["optims_state_dict"],
        wm_only=False,
    )

    agent.load_state_dict(weights, strict=False)


def main():
    config, agent, eval_env, wandb_run, logdir = load_agent()
    load_wm(config, agent)
    visualise_values(agent, eval_env)


if __name__ == "__main__":
    main()
