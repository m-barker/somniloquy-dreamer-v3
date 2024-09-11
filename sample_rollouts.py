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


def generate_trajectory_plot(latent_states: List[List[torch.Tensor]]):
    pca = PCA(n_components=2)

    # Concat all latent states
    latent_states_combined = [torch.cat(states, dim=0) for states in latent_states]
    latent_states_combined = torch.cat(latent_states_combined, dim=0)
    latent_states_combined = latent_states_combined.squeeze(1).detach().cpu().numpy()
    pca.fit(latent_states_combined)

    # Make empty plot
    plt.figure()
    plt.title("Imagined Latent Space Trajectories")
    plt.xlabel("Imagined Latent State PCA Component 1")
    plt.ylabel("Imagined Latent State PCA Component 2")
    colours = ["b", "r"]
    labels = ["I will teleport right", "I will teleport left"]
    for index, trajectory in enumerate(latent_states):
        trajectory = torch.cat(trajectory, dim=0).squeeze(1).detach().cpu().numpy()
        trajectory_pca = pca.transform(trajectory)
        plt.scatter(
            trajectory_pca[:, 0],
            trajectory_pca[:, 1],
            label=labels[index],
            c=colours[index],
        )

        # Add arrows
        for i in range(1, len(trajectory_pca)):
            plt.arrow(
                trajectory_pca[i - 1, 0],
                trajectory_pca[i - 1, 1],
                trajectory_pca[i, 0] - trajectory_pca[i - 1, 0],
                trajectory_pca[i, 1] - trajectory_pca[i - 1, 1],
                head_width=0.2,
                head_length=0.3,
                fc=colours[index],
                ec=colours[index],
            )

    # Save plot
    plt.legend()
    plt.savefig("latent_space_trajectories.png")
    plt.close()


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

    all_trajectories = []
    # Sample rollouts
    for t in range(n_rollouts):
        imagained_states, imagined_actions = sample_rollouts(
            agent,
            initial_state=init_state,
            trajectory_length=trajectory_length,
        )
        # Reshape to (T, N, C)
        imagined_tensor = torch.cat(imagained_states, dim=0).permute(1, 0, 2)
        imagined_narration = agent._wm.heads["language"].generate(
            imagined_tensor, agent._wm.vocab, 150, deterministic=False
        )
        true_obs = []
        for action in imagined_actions:
            numpy_action = action.squeeze(0).detach().cpu().numpy()
            obs, reward, done, info = env.step({"action": numpy_action})()
            true_obs.append(info["encoded_image"])
            if done:
                break
        true_narration = agent._wm.narrator.narrate(true_obs)
        print(f"Imagined: {imagined_narration}")
        print(f"True: {true_narration}")
        print(
            "-----------------------------------------------------------------------------------------------------"
        )
        env.reset()()
        all_trajectories.append(imagained_states)

        for index, state in enumerate(imagained_states):
            imagined_img = agent._wm.heads["decoder"](state)["image"].mode()
            imagined_img = imagined_img[0, 0].detach().cpu().numpy()
            imagined_img = np.clip(255 * imagined_img, 0, 255).astype(np.uint8)

            cv2.imwrite(
                f"imagined_img_rollout_{t+1}_step_{index+1}.png",
                cv2.cvtColor(imagined_img, cv2.COLOR_RGB2BGR),
            )

    # generate_trajectory_plot(all_trajectories[:2])


if __name__ == "__main__":
    main(setup_args())
