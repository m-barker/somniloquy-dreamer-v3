from typing import List

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dreamer import (
    Dreamer,
    setup_args,
    create_environments,
    setup,
    count_steps,
)
from tools import Logger, recursively_load_optim_state_dict, convert


def add_batch_to_obs(obs):
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs[k] = np.expand_dims(v, axis=0)
        else:
            obs[k] = np.array([v])
    return obs


def sample_rollouts(
    agent,
    initial_state,
    trajectory_length: int,
):
    imagained_states = []
    imagined_actions = []
    prev_state = initial_state
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    for t in range(trajectory_length):
        action = agent._task_behavior.actor(latent_state).sample().squeeze(0)
        imagined_actions.append(action)
        prior = agent._wm.dynamics.img_step(
            prev_state=prev_state,
            prev_action=action,
        )
        prev_state = prior
        latent_state = agent._wm.dynamics.get_feat(prior).unsqueeze(0)
        imagained_states.append(latent_state)
        predicted_continue = agent._wm.heads["cont"](latent_state).mode()
        if predicted_continue[0, 0].detach().cpu().numpy() < 0.5:
            break
    return imagained_states, imagined_actions


def get_posteriors(agent, initial_state, trajectory_length: int, env):

    posteriors = []
    true_obs = []
    prev_state = initial_state
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    for t in range(trajectory_length):
        action = agent._task_behavior.actor(latent_state).sample().squeeze(0)
        action_arr = np.zeros(3)
        action_arr[action.argmax().item()] = 1
        action_dict = {"action": action_arr}
        obs, reward, done, info = env.step(action_dict)()
        true_obs.append(action_arr)
        current_obs = obs.copy()
        current_obs = add_batch_to_obs(current_obs)
        current_obs = {k: convert(v) for k, v in current_obs.items()}
        current_obs = agent._wm.preprocess(current_obs)
        embed = agent._wm.encoder(current_obs)
        post, _ = agent._wm.dynamics.obs_step(
            embed=embed,
            is_first=current_obs["is_first"],
            prev_state=prev_state,
            prev_action=torch.tensor(action_arr)
            .to(agent._config.device)
            .unsqueeze(0)
            .float(),
        )
        latent_state = agent._wm.dynamics.get_feat(post).unsqueeze(0)
        prev_state = post
        posteriors.append(post)
    return posteriors, true_obs


def setup_agent_and_env(args):
    config, logdir = setup(args)
    step = count_steps(logdir)
    logger = Logger(logdir, config.action_repeat * step)
    train_env, _ = create_environments(config)
    train_env = train_env[0]
    action_space = train_env.action_space
    config.num_actions = (
        action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    )
    observation_space = train_env.observation_space
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        dataset=None,
    )
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    return agent, train_env


def main(args):
    agent, env = setup_agent_and_env(args)

    trajectory_length = 16
    n_rollouts = 20

    # Get initial state
    obs = env.reset()()  # Nasty
    transition = obs.copy()
    transition = add_batch_to_obs(transition)
    transition = {k: convert(v) for k, v in transition.items()}
    transition = agent._wm.preprocess(transition)
    embed = agent._wm.encoder(transition)
    init_state, _ = agent._wm.dynamics.obs_step(
        embed=embed,
        is_first=transition["is_first"],
        prev_state=None,
        prev_action=None,
    )
    n_correct = 0
    n_incorrect = 0
    for t in range(n_rollouts):
        posteriors, true_actions = get_posteriors(
            agent,
            initial_state=init_state,
            trajectory_length=trajectory_length,
            env=env,
        )
        true_actions = torch.tensor(true_actions).to(agent._config.device)
        posteriors.insert(0, init_state)
        stochastic_states = [s["stoch"] for s in posteriors]
        prev_states = stochastic_states[:-1]
        next_states = stochastic_states[1:]
        prev_states = torch.cat(prev_states, dim=0).reshape(trajectory_length, -1)
        next_states = torch.cat(next_states, dim=0).reshape(trajectory_length, -1)
        predicted_actions = agent._wm.heads["action_prediction"](
            torch.cat([prev_states, next_states], dim=-1)
        ).mode()

        for i in range(trajectory_length):
            if torch.equal(predicted_actions[i], true_actions[i]):
                n_correct += 1
            else:
                n_incorrect += 1

    print(f"ACCURACY: {n_correct / (n_correct + n_incorrect)}")


if __name__ == "__main__":
    main(setup_args())
