from typing import List, Tuple, Optional, Dict, Any

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import re


from tools import add_batch_to_obs, convert
from generate_plots import generate_image_reconstruction_plot


def imagine_trajectory(
    agent,
    initial_state: Dict[str, torch.Tensor],
    trajectory_length: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Samples a trajectory of imagined rollouts from the world model and
    actor. Returns the imagined states and actions.

    Args:
        agent (Dreamer): Dreamer agent containing the world model and actor.
        initial_state (torch.Tensor): Starting state of the model.
        trajectory_length (int): Length of the imagined trajectory.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Imagined states and actions.
    """
    imagained_states: List[torch.Tensor] = []
    imagined_actions: List[torch.Tensor] = []
    prev_state = initial_state
    done = False
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    for t in range(trajectory_length):
        if done:
            imagained_states.append(torch.zeros_like(latent_state))
            imagined_actions.append(torch.zeros_like(imagined_actions[-1]))
            continue
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
            done = True
    return imagained_states, imagined_actions


def rollout_trajectory(
    agent,
    initial_state: Dict[str, torch.Tensor],
    trajectory_length: int,
    actions: List[torch.Tensor],
    env,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rolls out a trajectory given the planned actions. Returns the posterior
    states and the environment observations.

    Args:
        agent (Dreamer): Dreamer agent used to compute the posterior states.
        initial_state (Dict[str, torch.Tensor]): Starting state of the model.
        trajectory_length (int): Number of timesteps to rollout (must match the length of actions).
        actions (List[torch.Tensor]): List of actions to rollout.
        env (Damy): Environment to rollout the actions in.

    Returns:
        Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]: Posterior states and environment obs and
        posterior info.
    """

    posterior_states: List[torch.Tensor] = []
    posterior_info: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    config = agent._config
    no_convert = config.no_convert_list
    ignore = config.ignore_list

    prev_state = initial_state
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    done = False
    for t in range(trajectory_length):
        if done:
            posterior_states.append(torch.zeros_like(latent_state))
            observations.append(
                {
                    "obs": None,
                    "reward": None,
                    "done": done,
                    "info": None,
                }
            )
            continue
        action = actions[t]
        # if actions are all zeros, trajectory is done
        if torch.all(action == 0):
            done = True
            continue
        obs, reward, done, info = env.step(
            {"action": action.squeeze(0).detach().cpu().numpy()}
        )()
        transition = obs.copy()
        transition = add_batch_to_obs(transition)
        transition = {
            k: (v if k in no_convert else convert(v))
            for k, v in transition.items()
            if k not in ignore
        }
        transition = agent._wm.preprocess(transition)
        embed = agent._wm.encoder(transition)
        posterior, _ = agent._wm.dynamics.obs_step(
            embed=embed,
            is_first=transition["is_first"],
            prev_state=prev_state,
            prev_action=action,
        )
        observations.append(
            {
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        latent_state = agent._wm.dynamics.get_feat(posterior).unsqueeze(0)
        posterior_states.append(latent_state)
        posterior_info.append(posterior)
        prev_state = posterior

    return posterior_states, observations, posterior_info


def sample_rollouts(
    agent,
    env,
    n_samples: int,
    trajectory_length: int = 16,
    n_consecutive_trajectories: int = 1,
) -> Dict[str, Any]:
    """Samples rollouts from the world model and actor. Returns the imagined
    states and actions.

    Args:
        agent (Dreamer): Dreamer agent containing the world model and actor.
        env (Damy): Environment to sample rollouts from.
        n_samples (int): Number of rollouts to sample.
        trajectory_length (int, optional): Length of the imagined trajectory. Defaults to 16.
        n_consecutive_trajectories (int, optional): Number of consecutive trajectories to sample. Defaults to 1.

    Returns:
        Dict[str, Any]: Dictionary containing the imagined states, imagined actions, posterior states, and observations
        for each sample.

    """

    config = agent._config
    no_convert = config.no_convert_list
    ignore = config.ignore_list
    imagined_state_samples: List[List[torch.Tensor]] = []
    imagined_action_samples: List[List[torch.Tensor]] = []
    posterior_state_samples: List[List[torch.Tensor]] = []
    observation_samples: List[List[Dict[str, Any]]] = []
    for sample in range(n_samples):
        obs = env.reset()()
        transition = obs.copy()
        transition = add_batch_to_obs(transition)
        transition = {
            k: (v if k in no_convert else convert(v))
            for k, v in transition.items()
            if k not in ignore
        }
        transition = agent._wm.preprocess(transition)
        embed = agent._wm.encoder(transition)
        init_state, _ = agent._wm.dynamics.obs_step(
            embed=embed,
            is_first=transition["is_first"],
            prev_state=None,
            prev_action=None,
        )
        initial_state = init_state
        imagined_states, imagined_actions = imagine_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=trajectory_length,
        )
        posterior_states, observations, posteriors = rollout_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=trajectory_length,
            actions=imagined_actions,
            env=env,
        )
        imagined_state_samples.append(imagined_states)
        imagined_action_samples.append(imagined_actions)
        posterior_state_samples.append(posterior_states)
        observation_samples.append(observations)
        if n_consecutive_trajectories > 1:
            for traj in range(n_consecutive_trajectories - 1):
                initial_state = posteriors[-1]
                imagined_states, imagined_actions = imagine_trajectory(
                    agent=agent,
                    initial_state=initial_state,
                    trajectory_length=trajectory_length,
                )
                posterior_states, observations, posteriors = rollout_trajectory(
                    agent=agent,
                    initial_state=initial_state,
                    trajectory_length=trajectory_length,
                    actions=imagined_actions,
                    env=env,
                )
                imagined_state_samples[-1].extend(imagined_states)
                imagined_action_samples[-1].extend(imagined_actions)
                posterior_state_samples[-1].extend(posterior_states)
                observation_samples[-1].extend(observations)

    return {
        "imagined_state_samples": imagined_state_samples,
        "imagined_action_samples": imagined_action_samples,
        "posterior_state_samples": posterior_state_samples,
        "observation_samples": observation_samples,
    }


def convert_images_to_numpy(images: List[torch.Tensor]) -> List[np.ndarray]:
    """Converts a list of torch.Tensor images to numpy arrays.

    Args:
        images (List[torch.Tensor]): List of torch.Tensor images.

    Returns:
        List[np.ndarray]: List of numpy arrays.
    """
    images = [img[0, 0].detach().cpu().numpy() for img in images]
    return [np.clip(255 * img, 0, 255).astype(np.uint8) for img in images]


def evaluate_rollouts(
    agent,
    imagined_state_samples: List[List[torch.Tensor]],
    imagined_action_samples: List[List[torch.Tensor]],
    posterior_state_samples: List[List[torch.Tensor]],
    observation_samples: List[List[Dict[str, Any]]],
    trajectory_length: int = 16,
):
    """Evalues a set of sampled rollouts.

    Args:
        agent (Dreamer): _description_
        imagined_state_samples (List[List[torch.Tensor]]): _description_
        imagined_action_samples (List[List[torch.Tensor]]): _description_
        posterior_state_samples (List[List[torch.Tensor]]): _description_
        observation_samples (List[List[Dict[str, Any]]]): _description_
        trajectory_length (int, optional): _description_. Defaults to 16.

    """
    config = agent._config
    for sample in range(len(imagined_state_samples)):
        print(f"SAMPLE LENGTH: {len(imagined_state_samples[sample])}")
        for index, trajectory in enumerate(
            range(0, len(imagined_state_samples[sample]), trajectory_length)
        ):
            # Adjust the end index if the environment terminated early
            end_index = trajectory + trajectory_length
            observations = observation_samples[sample][trajectory:end_index]
            for index, obs in enumerate(observations):
                if obs["obs"] is None:
                    end_index = index
                    break
            observations = observation_samples[sample][trajectory:end_index]
            imagined_states = imagined_state_samples[sample][trajectory:end_index]
            imagined_actions = imagined_action_samples[sample][trajectory:end_index]
            posterior_states = posterior_state_samples[sample][trajectory:end_index]
            # Happens when environment (or imagined trajectory) terminates early.
            if len(imagined_states) == 0:
                continue
            images = [obs["obs"]["image"] for obs in observations]
            rewards = [obs["reward"] for obs in observations]

            if config.enable_language:
                narration_keys = config.narrator["narration_key"]

                if type(narration_keys) is str:
                    try:
                        narration_data = [obs[narration_keys] for obs in observations]
                    except KeyError:
                        narration_data = [
                            obs["info"][narration_keys] for obs in observations
                        ]
                elif type(narration_keys) is list:
                    try:
                        narration_data = [
                            {key: obs[key] for key in narration_keys}
                            for obs in observations
                        ]
                    except KeyError:
                        narration_data = [
                            {key: obs["info"][key] for key in narration_keys}
                            for obs in observations
                        ]
                    # Comine into a single dictionary
                    narration_data = {
                        key: [data[key] for data in narration_data]
                        for key in narration_keys
                    }  # type: ignore
                else:
                    raise ValueError(f"Invalid narration_keys: {narration_keys}")

                # (N, T, C) -> (T, N, C)
                imagined_state_tensor = torch.cat(imagined_states, dim=0).permute(
                    1, 0, 2
                )
                planned_intent: str = agent._wm.heads["language"].generate(
                    imagined_state_tensor,
                    agent._wm.vocab,
                    config.dec_max_length,
                    sampling_method=config.token_sampling_method,
                )

                actual_narration = agent._wm.narrator.narrate(narration_data)

                print(
                    f"Sample {sample} Trajectory {index} Planned Intent: {planned_intent}"
                )
                print(
                    f"Sample {sample} Trajectory {index} Actual Narration: {actual_narration}"
                )

                imagined_images = [
                    agent._wm.heads["decoder"](state)["image"].mode()
                    for state in imagined_states
                ]
                reconstructed_images = [
                    agent._wm.heads["decoder"](state)["image"].mode()
                    for state in posterior_states
                ]
                imagined_images = convert_images_to_numpy(imagined_images)
                reconstructed_images = convert_images_to_numpy(reconstructed_images)

                reconstruction_plot: plt.Figure = generate_image_reconstruction_plot(
                    [imagined_images, reconstructed_images, images],
                    3,
                    len(images),
                    start_time=trajectory,
                )
                reconstruction_plot.suptitle(f"Sample {sample} Trajectory {index}")

                wandb.log(
                    {
                        "reconstruction_plot": reconstruction_plot,
                        "planned_intent": planned_intent,
                        "actual_narration": actual_narration,
                    }
                )
