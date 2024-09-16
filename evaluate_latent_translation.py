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


def tokenize_sentence(sentence: str, vocab: dict):
    tokens = sentence.split()
    tokenized_sentence = [vocab[token] for token in tokens]
    return tokenized_sentence


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
    init_stoch = init_state["stoch"]
    init_stoch = init_stoch.argmax(dim=-1)
    init_stoch = init_stoch.flatten() + 29
    test_sentence = "<LATENT> the agent moved towards the blue square which is the goal"
    tokenized_sentence = tokenize_sentence(test_sentence, agent._wm.vocab)
    tokenized_tensor = torch.tensor(tokenized_sentence).to(
        agent._config.device, dtype=torch.long
    )
    tokenized_tensor = tokenized_tensor.unsqueeze(0)
    translation = agent._wm.heads["language"].generate(
        tokenized_tensor,
        vocab=agent._wm.vocab,
        max_sequence_length=255,
        embed_src=True,
        return_tokens=True,
        prompt=init_stoch.unsqueeze(0),
    )
    print(translation)
    print(translation.shape)
    translation = translation.reshape((16, 16))

    stochastic_states = []
    for i in range(16):
        stochastic_state = torch.zeros((16, 16)).to(agent._config.device)
        for j in range(16):
            one_hot_index = translation[i][j] - 29
            stochastic_state[j][one_hot_index] = 1
        stochastic_states.append(stochastic_state)

    stochastic_states = torch.stack(stochastic_states, dim=0)
    predicted_actions = []
    for i in range(15):
        prev_state = stochastic_states[i].flatten()
        next_state = stochastic_states[i + 1].flatten()
        action_input = torch.cat([prev_state, next_state], dim=-1)
        predicted_action = agent._wm.heads["action_prediction"](action_input).mode()
        predicted_actions.append(predicted_action)
        print(predicted_action)

    for action in predicted_actions:
        action = action.detach().cpu().numpy()
        action_dict = {"action": action}
        obs, reward, done, _ = env.step(action_dict)()


if __name__ == "__main__":
    main(setup_args())
