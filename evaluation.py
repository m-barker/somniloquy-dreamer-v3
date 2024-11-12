from typing import List, Tuple, Union, Dict, Any, Optional
import random

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


from tools import (
    add_batch_to_obs,
    convert,
    word_tokenise_text,
    bleu_metric_from_strings,
    perplexity_metric,
)
from narration.crafter_narrator import CrafterNarrator
from generate_plots import generate_image_reconstruction_plot


@torch.no_grad()
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


@torch.no_grad()
def rollout_trajectory(
    agent,
    initial_state: Dict[str, torch.Tensor],
    trajectory_length: int,
    actions: Union[List[torch.Tensor], torch.Tensor],
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
        posterior = get_posterior_state(
            agent, obs, no_convert, ignore, prev_state, action
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


@torch.no_grad()
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
        obs, info = env.reset()()
        initial_state = get_posterior_state(agent, obs, no_convert, ignore)
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


@torch.no_grad()
def evaluate_rollouts(
    agent,
    imagined_state_samples: List[List[torch.Tensor]],
    imagined_action_samples: List[List[torch.Tensor]],
    posterior_state_samples: List[List[torch.Tensor]],
    observation_samples: List[List[Dict[str, Any]]],
    logger,
    trajectory_length: int = 16,
    wandb_run=None,
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
    bleu_scores = []
    posterior_bleu_scores = []
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
                posterior_state_tensor = torch.cat(posterior_states, dim=0).permute(
                    1, 0, 2
                )
                # our batch size is 1 so take first item in list
                planned_intent = agent._wm.heads["language"].generate(
                    imagined_state_tensor,
                    agent._wm.vocab,
                    config.dec_max_length,
                    sampling_method=config.token_sampling_method,
                )[0]
                planned_intent = " ".join(
                    [
                        word
                        for word in planned_intent.split()
                        if word not in ["<BOS>", "<EOS>", "<PAD>"]
                    ]
                )
                reconstructed_intent = agent._wm.heads["language"].generate(
                    posterior_state_tensor,
                    agent._wm.vocab,
                    config.dec_max_length,
                    sampling_method=config.token_sampling_method,
                )[0]
                reconstructed_intent = " ".join(
                    [
                        word
                        for word in reconstructed_intent.split()
                        if word not in ["<BOS>", "<EOS>", "<PAD>"]
                    ]
                )

                if type(narration_data) is dict:
                    if len(narration_data.keys()) == 1:  # type: ignore
                        narration_data = narration_data[list(narration_data.keys())[0]]  # type: ignore
                actual_narration = agent._wm.narrator.narrate(narration_data)

                try:
                    bleu_score = bleu_metric_from_strings(
                        planned_intent, actual_narration
                    )
                # When too few tokens generated
                except ValueError:
                    bleu_score = torch.tensor(0.0)

                bleu_scores.append(float(bleu_score))  # convert tensor

                try:
                    posterior_bleu_score = bleu_metric_from_strings(
                        reconstructed_intent, actual_narration
                    )
                except ValueError:
                    posterior_bleu_score = torch.tensor(0.0)

                posterior_bleu_scores.append(float(posterior_bleu_score))

                print(
                    f"Sample {sample} Trajectory {index} Planned Intent: {planned_intent}"
                )
                print(
                    f"Sample {sample} Trajectory {index} Actual Narration: {actual_narration}"
                )
                print(
                    f"Sample {sample} Trajectory {index} Imagined BLEU Score: {bleu_score}"
                )
                print(
                    f"Sample {sample} Trajectory {index} Reconstructed BLEU Score: {posterior_bleu_score}"
                )

                # imagined_images = [
                #     agent._wm.heads["decoder"](state)["image"].mode()
                #     for state in imagined_states
                # ]
                # reconstructed_images = [
                #     agent._wm.heads["decoder"](state)["image"].mode()
                #     for state in posterior_states
                # ]
                # imagined_images = convert_images_to_numpy(imagined_images)
                # reconstructed_images = convert_images_to_numpy(reconstructed_images)

                # reconstruction_plot: plt.Figure = generate_image_reconstruction_plot(
                #     [imagined_images, reconstructed_images, images],
                #     3,
                #     len(images),
                #     start_time=trajectory,
                # )
                # reconstruction_plot.suptitle(f"Sample {sample} Trajectory {index}")

                # wandb.log(
                #     {
                #         "reconstruction_plot": reconstruction_plot,
                #         "planned_intent": planned_intent,
                #         "actual_narration": actual_narration,
                #     }
                # )
    bleu_scores = np.array(bleu_scores)
    posterior_bleu_scores = np.array(posterior_bleu_scores)
    mean_score = bleu_scores.mean()
    mean_posterior_score = posterior_bleu_scores.mean()
    wandb_run.log(
        {
            "mean_imagined_bleu_score": mean_score,
            "mean_posterior_bleu_score": mean_posterior_score,
        },
        step=logger.step,
    )


def get_action_translation_dict(n_actions: int):
    action_names = ["<PAD>", "<BOS>", "<EOS>"]
    action_values = [0, 1, 2]
    for i in range(n_actions):
        action_names.append(f"action_{i}")
        action_values.append(i + 3)  # +3 for PAD, BOS, EOS
    return {k: v for k, v in zip(action_names, action_values)}


def get_posterior_state(
    agent,
    obs: Dict,
    no_convert: Optional[List[str]] = None,
    obs_to_ignore: Optional[List[str]] = None,
    prev_state: Optional[Dict] = None,
    prev_action: Optional[torch.Tensor] = None,
) -> Dict:
    """Computes a single-step posterior state given the environment observation.

    Args:
        agent (_type_): Dreamer agent.

        obs (Dict): Current step environment observation in dictionary form.

        no_convert (Optional[List[str]], optional): List of observation keys that should not be converted.
        Defaults to None.

        obs_to_ignore (Optional[List[str]], optional): List of observation keys that can be discarded.
        Defaults to None.

        prev_state (Optional[Dict], optional): Previous posterior state. If None, it is assumed that
        this is the first state in the trajectory. Defaults to None.

        prev_action (Optional[torch.Tensor], optional): Previous action taken. If None, it is assume that
        this is the first timetep of the trajectory. Defaults to None.

    Returns:
        Dict: Posterior state dictionary containing the stochastic and deterministic components.
    """

    no_convert_list = []
    ignore_list = []
    if no_convert is not None:
        no_convert_list = no_convert
    if obs_to_ignore is not None:
        ignore_list = obs_to_ignore

    transition = obs.copy()
    transition = add_batch_to_obs(transition)
    transition = {
        k: (v if k in no_convert_list else convert(v))
        for k, v in transition.items()
        if k not in ignore_list
    }
    transition = agent._wm.preprocess(transition)
    embed = agent._wm.encoder(transition)
    posterior_state, _ = agent._wm.dynamics.obs_step(
        embed=embed,
        is_first=transition["is_first"],
        prev_state=prev_state,
        prev_action=prev_action,
    )

    return posterior_state


@torch.no_grad()
def evaluate_language_to_action(
    agent,
    env,
    source_strings: List[str],
    n_prev_actions: int = 0,
    prev_action_policy: str = "actor",
):
    """Evaluates the language-to-action component of the world model that translates
    a input string into a sequence of actions that should result in a sequence of
    environment steps that are described by this string.

    Args:
        agent: Dreamer Agent containing all model components.

        env : Evaluation environment to rollout in.

        source_strings (List[str]): Possible set of string to test that
        are sampled from.

        n_prev_actions (int, optional): Number of actions to take in the environment
        before landing in the starting state for the evaluation. Used to help robustness test
        the component to ensure diversity of starting state. Defaults to 0.

        prev_action_policy (str, optional): If using previous actions, this sets the type of.
        policy to use. Must be one of "actor" (use the learned policy to take actions) or
        "random" (take random actions). Defaults to "actor".
    """

    config = agent._config

    input_string = random.choice(source_strings)
    tokenised_input = word_tokenise_text([input_string], agent._wm.vocab)
    tokenised_input_tensor = torch.tensor(tokenised_input, dtype=torch.long).to(
        config.device
    )

    no_convert = config.no_convert_list
    ignore = config.ignore_list

    obs, info = env.reset()()
    current_state = get_posterior_state(agent, obs, no_convert, ignore)

    n_actions = config.num_actions
    if n_prev_actions > 0:
        prev_state = current_state
        action = None
        for warmup_step in range(n_prev_actions):
            if prev_action_policy == "random":
                action_arr = np.zeros(n_actions)
                action = np.random.randint(n_actions)
                action_arr[action] = 1  # one-hot representation
                env_action = {"action": action_arr}
                obs, reward, done, info = env.step(env_action)()
            elif prev_action_policy == "actor":
                action_tensor = agent._task_behavior.actor(prev_state).sample()
                obs, reward, done, info = env.step(
                    {"action": action_tensor.squeeze().detach().numpy().cpu()}
                )()
                current_state = get_posterior_state(
                    agent, obs, no_convert, ignore, prev_state, action
                )
                prev_state = current_state

            else:
                raise ValueError(f"Invalid policy type provided: {prev_action_policy}")
            if done:
                env.reset()()
                raise RuntimeError(
                    "Environment terminated during action warmup, cannot evaluate"
                )
        # Need to get the latent state as not computed at each step if using random policy
        if prev_action_policy == "random":
            current_state = get_posterior_state(
                agent, obs, no_convert, ignore, prev_state, action
            )

    current_state_tensor = agent._wm.dynamics.get_feat(current_state)
    initial_state_embed = agent._wm.heads["language"]._initial_embed(
        current_state_tensor
    )
    action_translation_dict = get_action_translation_dict(n_actions)
    translated_action_tokens = (
        agent._wm.heads["language_to_action"].generate(
            input_seq=tokenised_input_tensor,
            vocab=action_translation_dict,
            max_sequence_length=config.enc_max_length,
            return_tokens=True,
            tokens_to_prepend=initial_state_embed,
        )
        - 3
    )  # -3 to recover action one-hot class

    translated_action_tokens = translated_action_tokens[
        translated_action_tokens >= 0
    ]  # remove <BOS>, <EOS>, and <PAD> tokens

    translated_one_hot_actions = torch.zeros(
        (translated_action_tokens.shape[0], n_actions)
    ).to(config.device)
    for t in range(len(translated_action_tokens)):
        translated_one_hot_actions[t, translated_action_tokens[t]] = 1
    print(f"TRANSLATED ACTIONS: {translated_action_tokens}")
    posterior_states, observations, _ = rollout_trajectory(
        agent=agent,
        initial_state=current_state,
        trajectory_length=len(translated_one_hot_actions),
        actions=translated_one_hot_actions.unsqueeze(1),
        env=env,
    )

    end_index = len(observations)
    for index, obs in enumerate(observations):
        if obs["obs"] is None:
            end_index = index
            break

    observations = observations[:end_index]
    posterior_states = posterior_states[:end_index]
    images = [obs["obs"]["image"] for obs in observations]
    reconstructed_images = [
        agent._wm.heads["decoder"](state)["image"].mode() for state in posterior_states
    ]
    reconstructed_images = convert_images_to_numpy(reconstructed_images)
    try:
        reconstruction_plot = generate_image_reconstruction_plot(
            [reconstructed_images, images],
            2,
            len(images),
            row_titles=["Reconstructed", "Actual"],
        )

        reconstruction_plot.savefig("lang-to-action-eval-plot.png")

        # wandb.log(
        #     {
        #         "reconstruction_plot": reconstruction_plot,
        #         "action_source_string": input_string,
        #     }
        # )
    except ValueError:
        return


def display_images_as_video(
    images, delay=100, window_name="Video Loop", target_size=(600, 600)
):
    while True:
        for img in images:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

            # Wait for the specified delay (in ms) between frames
            key = cv2.waitKey(delay)

            # Break if 'q' key is pressed
            if key == ord("q"):
                cv2.destroyAllWindows()
                return


def interactive_language_to_action(agent, env) -> None:
    """Interactively runs the language-to-action translation component, taking language input
    from the command line and rolling out the translated actions in the given environment.

    Args:
        agent: Dreamer Agent.
        env: Environment to rollout the translated actions in.
    """
    obs, info = env.reset()()
    done = False
    config = agent._config
    n_actions = config.num_actions
    action_translation_dict = get_action_translation_dict(n_actions)
    prev_state = None
    prev_action = None
    no_convert = config.no_convert_list
    ignore = config.ignore_list
    while not done:
        source_text = input("Please enter a string to be translated into actions:\n")
        tokenised_source = word_tokenise_text([source_text], agent._wm.vocab)
        tokenised_source_tensor = torch.tensor(tokenised_source, dtype=torch.long).to(
            config.device
        )
        starting_state_posterior = get_posterior_state(
            agent,
            obs,
            no_convert,
            ignore,
            prev_state,
            prev_action,
        )
        starting_state_tensor = agent._wm.dynamics.get_feat(starting_state_posterior)
        starting_state_embed = agent._wm.heads["language"]._initial_embed(
            starting_state_tensor
        )
        action_tokens = (
            agent._wm.heads["language_to_action"].generate(
                input_seq=tokenised_source_tensor,
                vocab=action_translation_dict,
                max_sequence_length=config.enc_max_length,
                return_tokens=True,
                tokens_to_prepend=starting_state_embed,
            )
            - 3
        )  # -3 to recover action one-hot class
        print(action_tokens)
        translated_action_tokens = action_tokens[
            action_tokens >= 0
        ]  # remove <BOS>, <EOS>, and <PAD> tokens
        print(f"TRANSLATED ACTIONS: {translated_action_tokens}")
        translated_one_hot_actions = torch.zeros(
            (translated_action_tokens.shape[0], n_actions)
        ).to(config.device)
        for t in range(len(translated_action_tokens)):
            translated_one_hot_actions[t, translated_action_tokens[t]] = 1

        _, observations, posterior_states = rollout_trajectory(
            agent=agent,
            initial_state=starting_state_posterior,
            trajectory_length=len(translated_one_hot_actions),
            actions=translated_one_hot_actions.unsqueeze(1),
            env=env,
        )
        images = []
        for index, obs in enumerate(observations):
            if obs["obs"] is None:
                done = True
            else:
                images.append(obs["obs"]["image"])
        display_images_as_video(images=images)

        obs = observations[-1]["obs"]
        prev_state = posterior_states[-2]
        prev_action = translated_one_hot_actions[-1].unsqueeze(0)


def minigrid_occupancy_grid_to_image(occupancy_grid: np.ndarray) -> np.ndarray:
    from minigrid.core.constants import (
        IDX_TO_OBJECT,
        COLORS,
    )

    IDX_TO_OBJECT[11] = "teleporter"
    COLORS["white"] = np.array([255, 255, 255])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        for j in range(10):
            object_idx = occupancy_grid[i, j][0]
            try:
                object = IDX_TO_OBJECT[object_idx]
            except KeyError:
                object = "unknown"
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
                image[i, j] = COLORS["white"]

    return image.transpose(1, 0, 2)


def minigrid_narration_using_obs_reconstruction(
    agent,
    imagined_state_samples: List[List[torch.Tensor]],
    imagined_action_samples: List[List[torch.Tensor]],
    posterior_state_samples: List[List[torch.Tensor]],
    observation_samples: List[List[Dict[str, Any]]],
    obs_size: Tuple[int, int, int],
    narrator,
    logger,
    wandb_run,
    trajectory_length: int = 16,
):
    reconstructed_bleu_scores = []
    imagined_bleu_scores = []
    sample_max_reconstructed_bleu_score = 0.0
    sample_max_imagined_bleu_score = 0.0

    for sample in range(len(imagined_state_samples)):
        sample_reconstructed_bleu_scores = []
        sample_imagined_bleu_scores = []
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
            posterior_states = posterior_state_samples[sample][trajectory:end_index]
            # Happens when environment (or imagined trajectory) terminates early.
            if len(imagined_states) == 0:
                continue

            true_occupancy_grid = [
                np.round(obs["obs"]["flattened_occupancy_grid"].reshape(obs_size) * 11)
                for obs in observations
            ]

            imagined_occupancy_grid = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_occupancy_grid"]
                    .mode()
                    .reshape(obs_size)
                    .cpu()
                    .numpy()
                    * 11
                )
                for state in imagined_states
            ]
            reconstructed_occupancy_grid = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_occupancy_grid"]
                    .mode()
                    .reshape(obs_size)
                    .cpu()
                    .numpy()
                    * 11
                )
                for state in posterior_states
            ]

            true_narration = narrator.narrate(true_occupancy_grid)  # type: ignore
            print(f"True Narration: {true_narration}")
            try:
                reconstructed_narration = narrator.narrate(reconstructed_occupancy_grid)
                reconstructed_bleu_score = float(
                    bleu_metric_from_strings(reconstructed_narration, true_narration)
                )
                print(f"Reconstruction Narration: {reconstructed_narration}")

            except Exception as e:
                print(f"Failed to generated reconstructed narration: {e}")
                reconstructed_bleu_score = 0.0
            try:
                imagined_narration = narrator.narrate(imagined_occupancy_grid)
                imagined_bleu_score = float(
                    bleu_metric_from_strings(imagined_narration, true_narration)
                )
                print(f"Imagined Narration: {imagined_narration}")

            except Exception as e:
                print(f"Failed to generated imagined narration: {e}")
                imagined_bleu_score = 0.0

            print(f"Reconstructed_BLEU_score: {reconstructed_bleu_score}")
            print(f"Imagined BLEU score: {imagined_bleu_score}")

            reconstructed_bleu_scores.append(reconstructed_bleu_score)
            imagined_bleu_scores.append(imagined_bleu_score)

            sample_reconstructed_bleu_scores.append(reconstructed_bleu_score)
            sample_imagined_bleu_scores.append(imagined_bleu_score)

        sample_mean_reconstructed_bleu_score = np.array(
            sample_reconstructed_bleu_scores
        ).mean()
        sample_mean_imagined_bleu_score = np.array(sample_imagined_bleu_scores).mean()

        if sample_mean_reconstructed_bleu_score > sample_max_reconstructed_bleu_score:
            sample_max_reconstructed_bleu_score = sample_mean_reconstructed_bleu_score
        if sample_mean_imagined_bleu_score > sample_max_imagined_bleu_score:
            sample_max_imagined_bleu_score = sample_mean_imagined_bleu_score

    imagined_bleu_scores = np.array(imagined_bleu_scores)
    reconstructed_bleu_scores = np.array(reconstructed_bleu_scores)
    mean_imagined_score = imagined_bleu_scores.mean()
    mean_posterior_score = reconstructed_bleu_scores.mean()
    wandb_run.log(
        {
            "obs_decoding_mean_imagined_bleu_score": mean_imagined_score,
            "obs_decoding_mean_posterior_bleu_score": mean_posterior_score,
            "obs_decoding_max_imagined_bleu_score": sample_max_imagined_bleu_score,
            "obs_decoding_max_posterior_bleu_score": sample_max_reconstructed_bleu_score,
        },
        step=logger.step,
    )


def crafter_narration_using_obs_reconstruction(
    agent,
    imagined_state_samples: List[List[torch.Tensor]],
    imagined_action_samples: List[List[torch.Tensor]],
    posterior_state_samples: List[List[torch.Tensor]],
    observation_samples: List[List[Dict[str, Any]]],
    logger,
    wandb_run,
    trajectory_length: int = 16,
) -> None:
    """Decodes the latetnt state to compute reconstructed observations
    that are then passed to the rule-based narrator directly, to evaluate
    the extent to which we need the transformer component.

    Args:
        agent (_type_): Dreamer agent that contains the world model.
        imagined_state_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        imagined_action_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        posterior_state_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        observation_samples (List[List[Dict[str, Any]]]): (n_samples, n_trajectories * traj_length)
        trajectory_length (int, optional): Length of each trajectory. Defaults to 16.
    """
    inventory_keys = [
        "health",
        "food",
        "drink",
        "energy",
        "sapling",
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    ]
    achievement_keys = [
        "collect_coal",
        "collect_diamond",
        "collect_drink",
        "collect_iron",
        "collect_sapling",
        "collect_stone",
        "collect_wood",
        "defeat_skeleton",
        "defeat_zombie",
        "eat_cow",
        "eat_plant",
        "make_iron_pickaxe",
        "make_iron_sword",
        "make_stone_pickaxe",
        "make_stone_sword" "make_wood_pickaxe",
        "make_wood_sword",
        "place_furnace",
        "place_plant",
        "place_stone",
        "place_table",
        "wake_up",
    ]
    reconstructed_bleu_scores = []
    imagined_bleu_scores = []
    for sample in range(len(imagined_state_samples)):
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
            posterior_states = posterior_state_samples[sample][trajectory:end_index]
            # Happens when environment (or imagined trajectory) terminates early.
            if len(imagined_states) == 0:
                continue

            imagined_narration_obs: Dict[str, Any] = {
                "semantic": [],
                "inventory": [],
                "achievements": [],
            }

            reconstructed_narration_obs: Dict[str, Any] = {
                "semantic": [],
                "inventory": [],
                "achievements": [],
            }

            imagined_grid = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_grid"]
                    .mode()
                    .reshape(7, 9)
                    .cpu()
                    .numpy()
                    * 18
                ).astype(np.uint8)
                for state in imagined_states
            ]
            imagined_inventory = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_inventory"]
                    .mode()
                    .cpu()
                    .numpy()
                    .flatten()
                    * 500
                ).astype(np.uint8)
                for state in imagined_states
            ]
            imagined_achievements = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_achievements"]
                    .mode()
                    .cpu()
                    .numpy()
                    .flatten()
                    * 500
                ).astype(np.uint8)
                for state in imagined_states
            ]

            reconstructed_grid = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_grid"]
                    .mode()
                    .reshape(7, 9)
                    .cpu()
                    .numpy()
                    * 18
                ).astype(np.uint8)
                for state in posterior_states
            ]
            reconstructed_inventory = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_inventory"]
                    .mode()
                    .cpu()
                    .numpy()
                    .flatten()
                    * 500
                ).astype(np.uint8)
                for state in posterior_states
            ]
            reconstructed_achievements = [
                np.round(
                    agent._wm.heads["decoder"](state)["flattened_achievements"]
                    .mode()
                    .cpu()
                    .numpy()
                    .flatten()
                    * 500
                ).astype(np.uint8)
                for state in posterior_states
            ]

            # Populate the reconstructed narration data
            for i in range(len(posterior_states)):
                imagined_narration_obs["semantic"].append(imagined_grid[i])
                reconstructed_narration_obs["semantic"].append(reconstructed_grid[i])
                imagined_achievements_dict = {}
                reconstructed_achievements_dict = {}
                for index, achievment_str in enumerate(achievement_keys):
                    imagined_achievements_dict[achievment_str] = imagined_achievements[
                        i
                    ][index]
                    reconstructed_achievements_dict[achievment_str] = (
                        reconstructed_achievements[i][index]
                    )
                imagined_inventory_dict = {}
                reconstructed_inventory_dict = {}
                for index, inventory_str in enumerate(inventory_keys):
                    imagined_inventory_dict[inventory_str] = imagined_inventory[i][
                        index
                    ]
                    reconstructed_inventory_dict[inventory_str] = (
                        reconstructed_inventory[i][index]
                    )
                imagined_narration_obs["achievements"].append(
                    imagined_achievements_dict
                )
                reconstructed_narration_obs["achievements"].append(
                    reconstructed_achievements_dict
                )
                imagined_narration_obs["inventory"].append(imagined_inventory_dict)
                reconstructed_narration_obs["inventory"].append(
                    reconstructed_inventory_dict
                )
                narration_keys = ["semantic", "inventory", "achievements"]
                narrator = CrafterNarrator()
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

            true_narration = narrator.narrate(narration_data)  # type: ignore
            print(f"True Narration: {true_narration}")
            try:
                reconstructed_narration = narrator.narrate(reconstructed_narration_obs)
                reconstructed_bleu_score = float(
                    bleu_metric_from_strings(reconstructed_narration, true_narration)
                )
                print(f"Reconstruction Narration: {reconstructed_narration}")

            except Exception as e:
                print(f"Failed to generated reconstructed narration: {e}")
                reconstructed_bleu_score = 0.0
            try:
                imagined_narration = narrator.narrate(imagined_narration_obs)
                imagined_bleu_score = float(
                    bleu_metric_from_strings(imagined_narration, true_narration)
                )
                print(f"Imagined Narration: {imagined_narration}")

            except Exception as e:
                print(f"Failed to generated imagined narration: {e}")
                imagined_bleu_score = 0.0

            print(f"Reconstructed_BLEU_score: {reconstructed_bleu_score}")
            print(f"Imagined BLEU score: {imagined_bleu_score}")

            reconstructed_bleu_scores.append(reconstructed_bleu_score)
            imagined_bleu_scores.append(imagined_bleu_score)
    imagined_bleu_scores = np.array(imagined_bleu_scores)
    reconstructed_bleu_scores = np.array(reconstructed_bleu_scores)
    mean_imagined_score = imagined_bleu_scores.mean()
    mean_posterior_score = reconstructed_bleu_scores.mean()
    wandb_run.log(
        {
            "obs_decoding_mean_imagined_bleu_score": mean_imagined_score,
            "obs_decoding_mean_posterior_bleu_score": mean_posterior_score,
        },
        step=logger.step,
    )
