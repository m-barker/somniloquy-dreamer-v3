from typing import List
import os
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
from narration.minigrd_narrator import MiniGridComplexTeleportNarrator
from minigrid.core.constants import (
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    COLOR_TO_IDX,
    TILE_PIXELS,
)

IDX_TO_OBJECT[11] = "teleporter"


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


def get_posteriors(agent, initial_state, trajectory_length: int, env, actions):

    posteriors = []
    true_obs = []
    prev_state = initial_state
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    for t in range(trajectory_length):
        action_arr = actions[t].squeeze(0).detach().cpu().numpy()
        action_dict = {"action": action_arr}
        obs, reward, done, info = env.step(action_dict)()
        true_obs.append(obs["occupancy_grid"])
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
        posteriors.append(agent._wm.dynamics.get_feat(post).unsqueeze(0))
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
    agent.eval()
    return agent, train_env


def occupancy_grid_to_image(occupancy_grid: np.ndarray) -> np.ndarray:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        for j in range(10):
            object_idx = occupancy_grid[i, j][0]
            object = IDX_TO_OBJECT[object_idx]
            if object == "wall":
                image[i, j] = COLORS["grey"]
            elif object == "empty":
                image[i, j] = np.array([0, 0, 0])
            elif object == "teleporter":
                image[i, j] = COLORS["purple"]
            elif object == "goal":
                image[i, j] = COLORS["green"]
            elif object == "agent":
                image[i, j] = COLORS["red"]
            elif object == "unseen":
                image[i, j] = COLORS["blue"]
            elif object == "floor":
                image[i, j] = np.array([0, 0, 0])
            elif object == "door":
                image[i, j] = COLORS["yellow"]
            elif object == "lava":
                image[i, j] = COLORS["purple"]
            else:
                image[i, j] = COLORS["purple"]

    return image.transpose(1, 0, 2)


def get_initial_state(agent, env):
    obs = env.reset()()
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
    return init_state


def main(args):
    output_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/world_model_evaluation/minigrid-occupancy-grid"
    narrator = MiniGridComplexTeleportNarrator()
    agent, env = setup_agent_and_env(args)
    with torch.no_grad():
        trajectory_length = 16
        n_rollouts = 20

        init_state = get_initial_state(agent, env)

        all_trajectories = []
        # Sample rollouts
        for t in range(n_rollouts):
            imagained_states, imagined_actions = sample_rollouts(
                agent,
                initial_state=init_state,
                trajectory_length=trajectory_length,
            )
            posteriors, true_obs = get_posteriors(
                agent,
                initial_state=init_state,
                trajectory_length=trajectory_length,
                env=env,
                actions=imagined_actions,
            )
            # Reset as getting posteriors steps the environment
            env.reset()()
            true_obs = []
            for action in imagined_actions:
                numpy_action = action.squeeze(0).detach().cpu().numpy()
                obs, reward, done, info = env.step({"action": numpy_action})()
                true_obs.append(obs["occupancy_grid"])
                if done:
                    break
            # true_narration = agent._wm.narrator.narrate(true_obs)
            true_narration = narrator.narrate(true_obs)
            # print(f"Imagined: {imagined_narration}")
            print(f"True: {true_narration}")
            print(
                "-----------------------------------------------------------------------------------------------------"
            )
            env.reset()()
            all_trajectories.append(imagained_states)
            imagined_grids = []
            reconstructed_grids = []
            for index, imagined_state, posterior_state in zip(
                range(trajectory_length), imagained_states, posteriors
            ):
                imagined_grid = agent._wm.heads["decoder"](imagined_state)[
                    "flattened_occupancy_grid"
                ].mode()
                imagined_image = agent._wm.heads["decoder"](imagined_state)[
                    "image"
                ].mode()
                imagined_img = imagined_image[0, 0].detach().cpu().numpy()
                imagined_img = np.clip(255 * imagined_img, 0, 255).astype(np.uint8)
                imagined_grid = imagined_grid[0, 0].detach().cpu().numpy()
                # Round to nearest integer
                imagined_grid = np.round(imagined_grid).astype(np.uint8)
                imagined_grid = imagined_grid.reshape(10, 10, 3)
                imagined_grids.append(imagined_grid)

                reconstructed_grid = agent._wm.heads["decoder"](posterior_state)[
                    "flattened_occupancy_grid"
                ].mode()
                reconstructed_img = agent._wm.heads["decoder"](posterior_state)[
                    "image"
                ].mode()
                reconstructed_img = reconstructed_img[0, 0].detach().cpu().numpy()
                reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype(
                    np.uint8
                )
                reconstructed_grid = reconstructed_grid[0, 0].detach().cpu().numpy()
                # Round to nearest integer
                reconstructed_grid = np.round(reconstructed_grid).astype(np.uint8)
                reconstructed_grid = reconstructed_grid.reshape(10, 10, 3)
                reconstructed_grids.append(reconstructed_grid)

                if np.sum(imagined_grid[:, :, 0] == 10) > 1:
                    print(
                        f"Number of agents in imagined grid: {np.sum(imagined_grid[:, :, 0] == 10)}"
                    )
                if np.sum(reconstructed_grid[:, :, 0] == 10) > 1:
                    print(
                        f"Number of agents in reconstructed grid: {np.sum(reconstructed_grid[:, :, 0] == 10)}"
                    )

                try:
                    true_observation = true_obs[index]
                except IndexError:
                    break
                true_image = occupancy_grid_to_image(true_observation)
                cv2.imwrite(
                    os.path.join(
                        output_path, f"true_img_rollout_{t+1}_step_{index+1}.png"
                    ),
                    cv2.cvtColor(true_image, cv2.COLOR_RGB2BGR),
                )
                # Convert to image
                try:
                    reconstructed_img_from_grid = occupancy_grid_to_image(
                        reconstructed_grid
                    )
                except Exception as e:
                    print(e)
                    print("Error in converting reconstructed grid to image")

                try:
                    imagined_img_from_grid = occupancy_grid_to_image(imagined_grid)
                except Exception as e:
                    print(e)
                    print("Error in converting imagined grid to image")
                cv2.imwrite(
                    os.path.join(
                        output_path, f"imagined_img_rollout_{t+1}_step_{index+1}.png"
                    ),
                    cv2.cvtColor(imagined_img_from_grid, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(
                        output_path,
                        f"imagined_img_rollout_{t+1}_step_{index+1}_from_image.png",
                    ),
                    cv2.cvtColor(imagined_img, cv2.COLOR_RGB2BGR),
                )

                cv2.imwrite(
                    os.path.join(
                        output_path,
                        f"reconstructed_img_rollout_{t+1}_step_{index+1}.png",
                    ),
                    cv2.cvtColor(reconstructed_img_from_grid, cv2.COLOR_RGB2BGR),
                )

                cv2.imwrite(
                    os.path.join(
                        output_path,
                        f"reconstructed_img_rollout_{t+1}_step_{index+1}_from_image.png",
                    ),
                    cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR),
                )

            try:
                print(
                    f"Reconstructed Narration: {narrator.narrate(reconstructed_grids)}"
                )
            except Exception as e:
                print(e)
                print("Error in computing the reconstructed narration")
            try:
                print(f"Imagined Narration: {narrator.narrate(imagined_grids)}")
            except Exception as e:
                print(e)
                print("Error in computing the imagined narration")


if __name__ == "__main__":
    main(setup_args())
