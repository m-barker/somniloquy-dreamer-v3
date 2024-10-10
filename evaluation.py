from typing import List, Tuple, Optional, Dict

import torch


from dreamer import Dreamer, Damy
from tools import add_batch_to_obs, convert


def imagine_trajectory(
    agent: Dreamer,
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


def sample_rollouts(
    agent: Dreamer,
    env: Damy,
    n_samples: int,
    trajectory_length: int = 16,
    n_consecutive_trajectories: int = 1,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """Samples rollouts from the world model and actor. Returns the imagined
    states and actions.

    Args:
        agent (Dreamer): Dreamer agent containing the world model and actor.
        env (Damy): Environment to sample rollouts from.
        n_samples (int): Number of rollouts to sample.
        trajectory_length (int, optional): Length of the imagined trajectory. Defaults to 16.
        n_consecutive_trajectories (int, optional): Number of consecutive trajectories to sample. Defaults to 1.

    Returns:
        Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]: Imagined states and actions.
    """
    obs = env.reset()()
    config = agent._config
    no_convert = config.no_convert_list
    ignore = config.ignore_list

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

    imagined_states = []
    imagined_actions = []

    for sample in range(n_samples):
        states, actions = imagine_trajectory(agent, init_state, trajectory_length)
        imagined_states.append(states)
        imagined_actions.append(actions)
        
        if n_consecutive_trajectories > 1 and sample % n_consecutive_trajectories != 0:
            init_state = states[-1]

    return imagined_states, imagined_actions
