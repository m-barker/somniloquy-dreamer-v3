import random
import os
import json
import gc
from typing import List, Tuple, Union, Dict, Any, Optional
from copy import deepcopy

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.functional.text import chrf_score, translation_edit_rate
from torchmetrics.functional.text.bert import bert_score
from torcheval.metrics.text import Perplexity, BLEUScore
from torcheval.metrics.functional import word_error_rate
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm


from tools import (
    add_batch_to_obs,
    convert,
    word_tokenise_text,
    get_task_stopwords,
    remove_punctuation,
)
from narration.crafter_narrator import CrafterNarrator
from narration.ai2thor_narrator import CookEggNarrator
from generate_plots import generate_image_reconstruction_plot


@torch.no_grad()
def imagine_trajectory(
    agent,
    initial_state: Dict[str, torch.Tensor],
    trajectory_length: int,
    actions: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], bool]:
    """Samples a trajectory of imagined rollouts from the world model and
    actor. Returns the imagined states and actions.

    Args:
        agent (Dreamer): Dreamer agent containing the world model and actor.
        initial_state (torch.Tensor): Starting state of the model.
        trajectory_length (int): Length of the imagined trajectory.
        actions (optional, List[torch.Tensor]): Optionally provide the actions for the
        agent to imagine taking. If not provided, the actor component of the world-model
        is used.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Imagined states and actions, and whether
        the world model predicts the trajectory terminates, includes initial state.
    """
    if actions is None:
        imagined_actions: List[torch.Tensor] = []
    prev_state = initial_state
    done = False
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    imagained_states: List[torch.Tensor] = [latent_state.detach().clone()]
    for t in range(trajectory_length):
        # If world model thinks the episode terminates, pad the states/actions
        # with zeros.
        if done:
            imagained_states.append(torch.zeros_like(latent_state))
            if actions is None:
                imagined_actions.append(torch.zeros_like(imagined_actions[-1]))
            else:
                if len(actions) <= t:
                    actions.append(torch.zeros_like(actions[t - 1]))
                else:
                    actions[t] = torch.zeros_like(actions[t - 1])
            continue
        # No actions provided; sample the learned policy.
        if actions is None:
            action = agent._task_behavior.actor(latent_state).sample().squeeze(0)
            imagined_actions.append(action.detach().clone())
        else:
            action = actions[t]
        prior = agent._wm.dynamics.img_step(
            prev_state=prev_state,
            prev_action=action,
        )
        prev_state = prior
        latent_state = agent._wm.dynamics.get_feat(prior).unsqueeze(0)
        imagained_states.append(latent_state.detach().clone())
        predicted_continue = agent._wm.heads["cont"](latent_state).mode()
        # Less than 50% predicted chance that the episode continues according to world model.
        if predicted_continue[0, 0].detach().cpu().numpy() < 0.5:
            done = True
    if actions is None:
        return imagained_states, imagined_actions, done
    return imagained_states, actions, done


@torch.no_grad()
def rollout_trajectory(
    agent,
    initial_state: Dict[str, torch.Tensor],
    trajectory_length: int,
    actions: Union[List[torch.Tensor], torch.Tensor],
    env,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Rolls out a trajectory given the planned actions. Returns the posterior
    states and the environment observations.

    Args:
        agent (Dreamer): Dreamer agent used to compute the posterior states.
        initial_state (Dict[str, torch.Tensor]): Starting state of the model.
        trajectory_length (int): Number of timesteps to rollout (must match the length of actions).
        actions (List[torch.Tensor]): List of actions to rollout.
        env (Damy): Environment to rollout the actions in.

    Returns:
        Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]], bool]: Posterior states and environment obs and
        posterior info, and whether the environment terminates.
    """

    observations: List[Dict[str, Any]] = []
    config = agent._config
    no_convert = config.no_convert_list
    ignore = config.ignore_list

    prev_state = initial_state
    latent_state = agent._wm.dynamics.get_feat(prev_state).unsqueeze(0)
    posterior_states: List[torch.Tensor] = [latent_state]
    posterior_info: List[Dict[str, Any]] = [initial_state]
    done = False
    env_done = False
    for t in range(trajectory_length):
        action = actions[t]
        # if actions are all zeros, imagined trajectory is done
        if done or env_done or torch.all(action == 0):
            posterior_states.append(torch.zeros_like(latent_state))
            observations.append(
                {
                    "obs": None,
                    "reward": None,
                    "done": done,
                    "info": None,
                }
            )
            done = True
            continue
        obs, reward, env_done, info = env.step(
            {"action": action.squeeze(0).detach().cpu().numpy()}
        )()
        posterior = get_posterior_state(
            agent, obs, no_convert, ignore, prev_state, action
        )
        observations.append(
            {
                "obs": obs,
                "reward": reward,
                "done": env_done,
                "info": info,
            }
        )
        latent_state = agent._wm.dynamics.get_feat(posterior).unsqueeze(0)
        posterior_states.append(latent_state)
        posterior_info.append(posterior)
        prev_state = posterior

    return posterior_states, observations, posterior_info, env_done


def get_user_actions(env, trajectory_length: int, starting_obs) -> List[torch.Tensor]:
    """Interactively gets a sequence of actions from the user. Renders
    the environment to the screen to make it easier for the user to select
    the actions.

    Args:
        env: Gym environment with which to take the actions in. Assumed to be
        in the correct starting state.
        trajectory_length (int): number of actions to take.
        starting_obs: initial obs of the environment.

    Returns:
        List[torch.Tensor]: List of one-hot encoded actions in Tensor form,
        ready to be used by the "imagine_trajectory" function.
    """

    actions: List[torch.Tensor] = []
    n_actions = env.action_space.n
    obs = starting_obs
    device = torch.device("cuda")
    for t in range(trajectory_length):
        # Not all environments have a human-render option, so we display the
        # raw image to the screen instead.
        display_image(obs["image"])
        valid_action = False
        while not valid_action:
            try:
                action = int(
                    input(f"Please enter an action between 0-{n_actions-1}:\t")
                )
                valid_action = True
            except ValueError:
                print("Invalid action. Please Try again.")
                continue
        action_arr = np.zeros(n_actions)
        action_arr[action] = 1
        # Adding the batch dimension
        actions.append(torch.Tensor(action_arr).to(device).unsqueeze(0))
        # Dreamer implementation uses the below dictionary representation for actions
        # in the environment.
        env_action = {"action": action_arr}
        # Env.step() returns a function due to being parallisable
        obs, reward, done, info = env.step(env_action)()
        # Actions are padded later with zeros if need be.
        if done:
            break

    return actions


@torch.no_grad()
def sample_rollouts(
    agent,
    env,
    eval_episodes: int,
    trajectory_length: int = 15,
    n_consecutive_plans: Optional[int] = None,
    user_actions: bool = False,
    user_env=None,
) -> Dict[str, Any]:
    """Samples rollouts from the world model and actor. Returns the imagined
    states and actions.

    Args:
        agent (Dreamer): Dreamer agent containing the world model and actor.

        env (Damy): Environment to sample rollouts from.

        eval_episodes (int): Number of episodes to evaluate.

        trajectory_length (int, optional): Length of the imagined trajectory. Defaults to 15.

        n_consecutive_plans (int, optional): Number of consecutive plans to make and translate per episode.
        If None, keeps rolling out until the environment terminates, or terminates according to the world
        model. Defaults to None.

        user_actions (bool, optional): Whether the actions the agent imagines taking is provided by the user or
        the learned policy. Defaults to False.

        user_env (optional): Environment used to allow the user to visualise their choice of actions, wihtout stepping
        in the agent's environment.

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
    for episode_num in range(eval_episodes):
        obs, info = env.reset()()
        initial_state = get_posterior_state(agent, obs, no_convert, ignore)
        initial_obs = {
            "obs": obs,
            "reward": 0.0,
            "done": False,
            "info": info,
        }
        actions = None
        if user_actions:
            assert user_env is not None
            user_obs, user_info = user_env.reset()()
            actions = get_user_actions(user_env, trajectory_length, user_obs)
        imagined_states, imagined_actions, imagined_done = imagine_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=trajectory_length,
            actions=actions,
        )
        posterior_states, observations, posteriors, env_done = rollout_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=trajectory_length,
            actions=imagined_actions,
            env=env,
        )
        imagined_state_samples.append(imagined_states)
        # Insert action of zeros to the last timestep
        imagined_actions.append(torch.zeros_like(imagined_actions[-1]))
        imagined_action_samples.append(imagined_actions)
        posterior_state_samples.append(posterior_states)
        observations.insert(0, initial_obs)
        observation_samples.append(observations)

        stop = False
        plan_count = 0
        while not stop:
            if imagined_done or env_done:
                stop = True
                continue
            initial_state = posteriors[-1]
            imagined_states, imagined_actions, imagined_done = imagine_trajectory(
                agent=agent,
                initial_state=initial_state,
                trajectory_length=trajectory_length,
            )
            posterior_states, observations, posteriors, env_done = rollout_trajectory(
                agent=agent,
                initial_state=initial_state,
                trajectory_length=trajectory_length,
                actions=imagined_actions,
                env=env,
            )
            imagined_state_samples[-1].extend(imagined_states)
            # Insert action of zeros to the last timestep
            imagined_actions.append(torch.zeros_like(imagined_actions[-1]))
            imagined_action_samples[-1].extend(imagined_actions)
            posterior_state_samples[-1].extend(posterior_states)
            # Last trajectory's obs is first obs in current trajectory.
            observations.insert(0, observation_samples[-1][-1])
            observation_samples[-1].extend(observations)

            plan_count += 1
            if n_consecutive_plans is not None:
                if plan_count >= n_consecutive_plans:
                    stop = True

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
def configure_narration_data(
    narration_keys: Union[str, List[str]],
    observations: List[Dict[str, Any]],
    task_name: str,
) -> Union[Dict[str, List[str]], List[Dict[str, str]]]:
    """Configures the data from the observations for narration.

    Args:
        narration_keys (Union[str, List[str]]): Keys to extract from observations for narration.
        observations (List[Dict[str, Any]]): List of observations from which to extract data.
        task_name (str): Name of the task to determine the format of the narration data.

    Raises:
        ValueError: If the narration_keys type is invalid.

    Returns:
        Union[Dict[str, List[str]], List[Dict[str, str]]]: Configured narration data.
    """

    if "ai2thor" in task_name:
        narration_data: Dict[str, List[str]] = {}
        for narration_key in narration_keys:
            for obs in observations:
                if narration_key not in narration_data.keys():
                    narration_data[narration_key] = [obs["obs"][narration_key]]
                else:
                    narration_data[narration_key].append(obs["obs"][narration_key])

    elif type(narration_keys) is str:
        try:
            narration_data = [obs[narration_keys] for obs in observations]  # type: ignore
        except KeyError:
            narration_data = [obs["info"][narration_keys] for obs in observations]  # type: ignore
    elif type(narration_keys) is list:
        try:
            narration_data = [
                {key: obs[key] for key in narration_keys} for obs in observations
            ]  # type: ignore
        except KeyError:
            narration_data = [
                {key: obs["info"][key] for key in narration_keys}
                for obs in observations
            ]  # type: ignore
        # Comine into a single dictionary
        narration_data = {
            key: [data[key] for data in narration_data] for key in narration_keys  # type: ignore
        }  # type: ignore
    else:
        raise ValueError(f"Invalid narration_keys: {narration_keys}")

    return narration_data


@torch.no_grad()
def generate_narration(
    agent,
    task_name: str,
    narration_data: Union[Dict[str, List[str]], List[Dict[str, str]]],
    narrator: Optional[Any] = None,
) -> str:
    """Generates the ground truth narration from a sequence of environment
    observations, pre-formatted to the correct information that the narrator
    requires.

    Args:
        agent: Dreamer agent that contains the narrator component.
        task_name (str): Name of the task to determine the format of the narration.
        narration_data (Union[Dict[str, List[str]], List[Dict[str, str]]]):
        Data to narrate.
        narrator (Optional): Narrator object to use for narration. Defaults to None.

    Returns:
        str: Ground truth narration.
    """

    if narrator is None:
        narrator = agent._wm.narrator

    if "ai2thor" in task_name:
        actual_narration = narrator.narrate(
            narration_data["agent_position"],  # type: ignore
            {
                "pickup": narration_data["pickup"],  # type: ignore
                "drop": narration_data["drop"],  # type: ignore
                "break": narration_data["break"],  # type: ignore
                "open": narration_data["open"],  # type: ignore
                "close": narration_data["close"],  # type: ignore
                "slice": narration_data["slice"],  # type: ignore
                "toggle_on": narration_data["toggle_on"],  # type: ignore
                "toggle_off": narration_data["toggle_off"],  # type: ignore
                "throw": narration_data["throw"],  # type: ignore
                "put": narration_data["put"],  # type: ignore
            },
        )
    elif type(narration_data) is dict:
        if len(narration_data.keys()) == 1:  # type: ignore
            narration_data = narration_data[list(narration_data.keys())[0]]  # type: ignore
        actual_narration = narrator.narrate(narration_data)
    else:
        raise ValueError(f"Unhandled narration data type: {type(narration_data)}")
    return actual_narration


@torch.no_grad()
def generate_translation(
    agent,
    config,
    latent_states: List[torch.Tensor],
    actions: Optional[List[torch.Tensor]] = None,
) -> str:
    """Returns a string containing the translated latent state plan.

    Args:
        agent: Trained World Model.
        config: configuration params of the world Model.
        latent_states (List[torch.Tensor]): List containing each
        latent state plan step.

        actions (List[torch.Tensor], optional): List containing each action
        taken in the plan. Used for the translation baseline. Defaults to None.

    Returns:
        str: Latent state plan translation.
    """

    # (T, N, C) -> (N, T, C)
    latent_state_tensor = torch.cat(latent_states, dim=0).permute(1, 0, 2)

    if config.translation_baseline:
        assert actions is not None
        action_tensor = torch.cat(actions, dim=0).unsqueeze(1).permute(1, 0, 2)
        reconstructed_images = (
            agent._wm.heads["decoder"](latent_state_tensor)["image"]
            .mode()
            .detach()
            .clone()
        )
        reconstructed_images = torch.clip(255 * reconstructed_images, 0, 255).to(
            torch.uint8
        )
        plan_translation = agent._wm.heads["language"].generate(
            reconstructed_images,
            action_tensor,
            agent._wm.vocab,
            config.dec_max_length,
            sampling_method=config.token_sampling_method,
        )
    else:
        # our batch size is 1 so take first item in list
        plan_translation = agent._wm.heads["language"].generate(
            latent_state_tensor,
            agent._wm.vocab,
            config.dec_max_length,
            sampling_method=config.token_sampling_method,
        )[0]
    plan_translation = " ".join(
        [
            word
            for word in plan_translation.split()
            if word not in ["<BOS>", "<EOS>", "<PAD>"]
        ]
    )
    return plan_translation


def get_narration_baseline_str(task_name: str) -> str:
    """Gets a baseline narration string used to capture the constant elements
    due to the rule-based narrator.

    Args:
        task_name (str): Name of the task to determine the baseline narration.

    Returns:
        str: Baseline string.
    """

    if "crafter" in task_name:
        return "I will see. I will not harvest anything. I will not craft anything. I will not eat anything and get more hungry. I will not drink anything and get more thirsty. I will not sleep and get more tired."
    else:
        raise ValueError(f"Unhandled task name: {task_name}")


def perplexity_metric(
    predicted_logits: torch.Tensor, true_tokens: torch.Tensor, padding_index: int = 0
) -> torch.Tensor:
    """Calculates the perplexity of a sequence of logits.

    Args:
        predicted_logits (torch.Tensor): Logits of shape ()
        true_tokens (torch.Tensor): True tokens of shape ()
        padding_index (int, optional): The token ID used for padding. Defaults to 0.

    Returns:
        torch.Tensor: Perplexity metric
    """

    perplexity = Perplexity(ignore_index=padding_index, device=predicted_logits.device)
    perplexity_score = perplexity.update(predicted_logits, true_tokens).compute()
    return perplexity_score


def bleu_metric_from_tokens(
    predicted_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    translation_dict: Dict[str, int],
    n_gram: int = 4,
) -> torch.Tensor:
    """Computes the BLEU score between a batch of predicted tokens and a batch of true tokens

    Args:
        predicted_tokens (torch.Tensor): shape (batch_length, max_seq_length)
        true_tokens (torch.Tensor): shape (batch_length, max_seq_length)
        translation_dict (Dict[str, int]): dictionary that translates from words to token IDs.
        n_gram (int, optional): the number of n-grams to consider. Defaults to 4.

    Returns:
        torch.Tensor: BLEU score that ranges from [0,1]
    """
    candidates: List[str] = []
    references: List[List[str]] = []
    token_to_str = {v: k for k, v in translation_dict.items()}
    tokens_to_remove = [0, 1, 2]  # BOS, EOS, PAD
    for cand_tokens, ref_tokens in zip(predicted_tokens, true_tokens):
        cand_str = "".join(
            [token_to_str[s] for s in cand_tokens if s not in tokens_to_remove]
        )
        ref_str = "".join(
            [token_to_str[s] for s in ref_tokens if s not in tokens_to_remove]
        )
        candidates.append(cand_str)
        references.append(
            [ref_str]
        )  # need to make nested list as in theory there can be multiple reference translations per string

    metric = BLEUScore(n_gram=n_gram)
    metric.update(candidates, references)
    return metric.compute()


def bleu_metric_from_strings(
    predicted_sequence: str,
    true_sequence: str,
    n_gram: int = 4,
    convert_case: bool = True,
    remove_punc: bool = True,
    words_to_remove: Optional[List[str]] = None,
) -> torch.Tensor:
    """Computes the bleu score between a translated and true string.

    Args:
        predicted_sequence (str): Machine translated string

        true_sequence (str): Ground truth string

        n_gram (int, optional): Number of n-grams to consider. Defaults to 4.

        convert_case (bool, optional): Whether to convert the strings to lower
        case (BLEU is not case insensitive). Defaults to True.

        remove_punc (bool, optional): Whether to remove punctuation from
        the strings. Defaults to True.

        words_to_remove (List[str], optional): List of words to remove from both
        strings. Defaults to None.

    Returns:
        torch.Tensor: BLEU score in range [0,1]
    """

    if convert_case:
        predicted_sequence = predicted_sequence.lower()
        true_sequence = true_sequence.lower()
    if remove_punc:
        predicted_sequence = remove_punctuation(predicted_sequence)
        true_sequence = remove_punctuation(true_sequence)

    if words_to_remove is not None:
        # From https://stackoverflow.com/questions/25346058/removing-list-of-words-from-a-string
        tmp_pred = [
            word for word in predicted_sequence.split() if word not in words_to_remove
        ]
        tmp_true = [
            word for word in true_sequence.split() if word not in words_to_remove
        ]
        predicted_sequence = " ".join(tmp_pred)
        true_sequence = " ".join(tmp_true)

    metric = BLEUScore(n_gram=n_gram)
    metric.update(
        predicted_sequence, [true_sequence]
    )  # there can be multiple referenence strings
    return metric.compute()


def calculate_wer(
    predicted_sequence: str,
    true_sequence: str,
) -> torch.Tensor:
    """Computes the word error rate (WER) between a translated and true string.

    Args:
        predicted_sequence (str): Translated string

        true_sequence (str): Ground truth string

    Returns:
        torch.Tensor: WER in range [0, 1+]
        + as can be larger than 1 if the predicted sequence
        is longer (and thus has more errors) than the true sequence.
    """

    return word_error_rate(predicted_sequence, true_sequence)


def compute_rouge_score(
    predicted_sequence: str, true_sequence: str
) -> Dict[str, float]:
    """Computes the ROUGE score between a translated and true string.

    Args:
        predicted_sequence (str): Translated string

        true_sequence (str): Ground truth string

    Returns:
        Dict[str, float]: Dictionary of Rouge scores for different rouge types.
        Each one contains (f1, precision, and recall)
    """

    rouge = ROUGEScore()
    scores = rouge(predicted_sequence, true_sequence)
    # By default, the scores are float tensors
    return {k: float(v) for k, v in scores.items()}


def compute_ter(predicted_sequence: str, true_sequence: str) -> torch.Tensor:
    """Computes the Translation Edit Rate (TER) between a translated and true string.

    Args:
        predicted_sequence (str): translated string by the model.
        true_sequence (str): Ground truth translation string.

    Returns:
        torch.Tensor: Translation Edit Rate
    """

    result = translation_edit_rate(predicted_sequence, [true_sequence])
    assert isinstance(result, torch.Tensor)
    return result


def compute_chrf(
    predicted_sequence: str, true_sequence: str, n_gram: int = 6, beta: int = 2
) -> torch.Tensor:
    """Computes the CHaRacter n-gram F-score++ (chrF++) between a translated and true string.
    by default, computes the 6-gram chr++F2 score, as this was shown to be the best performant
    in the original paper.


    Args:
        predicted_sequence (str): Translated string

        true_sequence (str): Ground truth string

        n_gram (int, optional): Number of charactern-grams to consider.
        Defaults to 6.

        beta (int, optional): Beta parameter for chrF. Defaults to 3.
        A higher beta value gives more weight to recall.


    Returns:
        torch.Tensor: chrF++ score
    """

    score = chrf_score(
        predicted_sequence, [true_sequence], n_char_order=n_gram, beta=beta
    )
    assert isinstance(score, torch.Tensor)
    return score


def compute_meteor_score(
    predicted_sequence: str, true_sequence: str, _gamma: float = 0.5
) -> float:
    """Computes the METEOR score between a translated and true string,
    using nltk's implementation.

    Args:
        predicted_sequence (str): Translated string

        true_sequence (str): Ground truth string

        gamma (float, optional): The gamma parameter for METEOR. Defaults to 0.5.

    Returns:
        float: METEOR score in range [0, 1].
    """
    # This is needed due to the chunking penalty always being non-zero,
    # see https://stackoverflow.com/questions/66944085/why-is-both-identical-sentences-meteor-score-used-in-machine-translation-not-eq
    if predicted_sequence == true_sequence:
        return 1.0

    # NLTK expects tokenized input
    hypothesis = predicted_sequence.split()
    reference = true_sequence.split()
    try:
        meteor_score = single_meteor_score(
            reference=reference, hypothesis=hypothesis, gamma=_gamma
        )
    except LookupError:
        import nltk

        nltk.download("wordnet")
        meteor_score = single_meteor_score(
            reference=reference, hypothesis=hypothesis, gamma=_gamma
        )
    return meteor_score


def compute_bert_score(
    predicted_sequence: str, true_sequence: str, model_name: str = "bert-base-uncased"
) -> Dict[str, float]:
    """Computes the BERT score between a translated and true string.

    Args:
        predicted_sequence (str): Translated string

        true_sequence (str): Ground truth string

        model_name (str, optional): Name of the BERT model to use. Defaults to "bert-base-uncased".

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 scores.
    """
    result = bert_score(predicted_sequence, true_sequence, model_name)
    assert isinstance(result, dict)
    assert isinstance(result["precision"], torch.Tensor)
    assert isinstance(result["recall"], torch.Tensor)
    assert isinstance(result["f1"], torch.Tensor)
    return {
        "bert_precision": float(result["precision"]),
        "bert_recall": float(result["recall"]),
        "bert_f1": float(result["f1"]),
    }


@torch.no_grad()
def compute_translation_metrics(
    generated_translation: str,
    true_translation: str,
    task_name: str,
    remove_punc: bool = True,
    lower_case: bool = True,
) -> Dict[str, float]:
    """Computes a multitude of evaluation metrics between the generated translation
    and the ground-truth translation. The following metrics are computed:
        - BLEU score
        - METEOR score
        - ROUGE score
        - Word Error Rate (WER)
        - Translation Edit Rate (TER)
        - chrF score

    Args:
        generated_translation (str): Latent state translation generated by the model.

        true_translation (str): 'Ground truth' translation computed by the rule-based
        translator.

        task_name (str): Name of the task (used to get the stopwords for the env).

        remove_punc (bool, optional): Whether to remove punctuation from both
        the source and target. Defaults to True.

        lower_case (bool, optional): Whether to convert the source and target to
        lower case. Defaults to True.

    Returns:
        Dict[str, float]: Dictionary of metric_name: metric_value pairs.
    """

    translation_metrics: Dict[str, float] = {}

    # ----- Pre-Processing --------

    if remove_punc:
        generated_translation = remove_punctuation(generated_translation)
        true_translation = remove_punctuation(true_translation)
    if lower_case:
        generated_translation = generated_translation.lower()
        true_translation = true_translation.lower()

    stop_words = get_task_stopwords(task_name)

    # From https://stackoverflow.com/questions/25346058/removing-list-of-words-from-a-string
    generated_translation_no_stopwords = [
        word for word in generated_translation.split() if word not in stop_words
    ]
    true_translation_no_stopwords = [
        word for word in true_translation.split() if word not in stop_words
    ]
    generated_translation_no_stopwords = " ".join(generated_translation_no_stopwords)
    true_translation_no_stopwords = " ".join(true_translation_no_stopwords)

    # ----- BLEU SCORE --------
    try:
        translation_metrics["bleu_score"] = float(
            bleu_metric_from_strings(
                generated_translation,
                true_translation,
            )
        )
    except ValueError:
        translation_metrics["bleu_score"] = 0.0
    try:
        translation_metrics["bleu_score_no_stopwords"] = float(
            bleu_metric_from_strings(
                generated_translation_no_stopwords,
                true_translation_no_stopwords,
            )
        )
    except ValueError:
        translation_metrics["bleu_score_no_stopwords"] = 0.0
    # ----- METEOR SCORE --------
    translation_metrics["meteor"] = compute_meteor_score(
        generated_translation, true_translation
    )
    translation_metrics["meteor_no_stopwords"] = compute_meteor_score(
        generated_translation_no_stopwords, true_translation_no_stopwords
    )
    # ----- ROUGE SCORE --------
    rouge_scores = compute_rouge_score(generated_translation, true_translation)
    rouge_scores_no_stopwords = compute_rouge_score(
        generated_translation_no_stopwords, true_translation_no_stopwords
    )
    rouge_scores_no_stopwords = {
        f"{k}_no_stopwords": v for k, v in rouge_scores_no_stopwords.items()
    }

    translation_metrics.update(rouge_scores)
    translation_metrics.update(rouge_scores_no_stopwords)
    # ----- Word Error Rate --------
    translation_metrics["wer"] = float(
        calculate_wer(generated_translation, true_translation)
    )
    translation_metrics["wer_no_stopwords"] = float(
        calculate_wer(generated_translation_no_stopwords, true_translation_no_stopwords)
    )
    # ----- Translation Edit Rate --------
    translation_metrics["ter"] = float(
        compute_ter(generated_translation, true_translation)
    )
    translation_metrics["ter_no_stopwords"] = float(
        compute_ter(generated_translation_no_stopwords, true_translation_no_stopwords)
    )
    # ----- chrF SCORE --------
    translation_metrics["chrf"] = float(
        compute_chrf(generated_translation, true_translation)
    )
    translation_metrics["chrf_no_stopwords"] = float(
        compute_chrf(generated_translation_no_stopwords, true_translation_no_stopwords)
    )

    return translation_metrics


def update_running_metrics(
    imagined_translation_metrics: Dict[str, float],
    reconstructed_translation_metrics: Dict[str, float],
    running_translation_eval_metrics: Dict[str, np.ndarray],
    episode_number: int,
    total_episodes: int,
    max_plans_per_episode: int,
    current_plan_index: int,
) -> None:
    """Updates the running metrics each time a new plan is translated
    during a given evaluation episode.

    Args:
        imagined_translation_metrics (Dict[str, float]): Dictionary of
        plan translation metrics for the imagined (prior) latent states.

        reconstructed_translation_metrics (Dict[str, float]): Dictionary of
        plan translation metrics for the reconstructed (posterior) latent states.

        running_translation_eval_metrics (Dict[str, np.ndarray]): Dictionary of
        running metrics for the imagined and reconstructed plans.

        episode_number (int): Current evalution episode number

        total_episodes (int): Total number of evaluation episodes.

        max_plans_per_episode (int): Maximum number of plans per episode.

        current_plan_index (int): Current plan index in the episode.
    """

    # Add the metrics the first time a plan is evaluated
    if len(running_translation_eval_metrics) == 0:
        for key in imagined_translation_metrics.keys():
            # We set NaN values to -1
            running_translation_eval_metrics[f"imagined_{key}"] = (
                np.zeros((total_episodes, max_plans_per_episode)) - 1
            )
            running_translation_eval_metrics[f"reconstructed_{key}"] = (
                np.zeros((total_episodes, max_plans_per_episode)) - 1
            )

    # Add the metrics to the running metrics
    for key in imagined_translation_metrics.keys():
        running_translation_eval_metrics[f"imagined_{key}"][
            episode_number, current_plan_index
        ] = imagined_translation_metrics[key]
        running_translation_eval_metrics[f"reconstructed_{key}"][
            episode_number, current_plan_index
        ] = reconstructed_translation_metrics[key]


def compute_evaluation_statistics(
    running_translation_eval_metrics: Dict[str, np.ndarray],
    nan_value: int = -1,
) -> Dict[str, float]:
    """Computes the mean and max of the running metrics.

    Args:
        running_translation_eval_metrics (Dict[str, np.ndarray]): Dictionary of
        running metrics for the imagined and reconstructed plans.

        nan_value (int, optional): Value to use for NaN values. Defaults to -1.

    Returns:
        Dict[str, float]: Dictionary of mean and max metrics.
    """
    mean_metrics = {}
    var_metrics = {}
    std_metrics = {}
    min_metrics = {}
    max_metrics = {}

    for key in running_translation_eval_metrics.keys():
        mean_metrics[f"mean_{key}"] = running_translation_eval_metrics[key][
            running_translation_eval_metrics[key] != nan_value
        ].mean()
        var_metrics[f"var_{key}"] = running_translation_eval_metrics[key][
            running_translation_eval_metrics[key] != nan_value
        ].var()
        std_metrics[f"std_{key}"] = running_translation_eval_metrics[key][
            running_translation_eval_metrics[key] != nan_value
        ].std()
        min_metrics[f"min_{key}"] = running_translation_eval_metrics[key][
            running_translation_eval_metrics[key] != nan_value
        ].min()
        max_metrics[f"max_{key}"] = running_translation_eval_metrics[key][
            running_translation_eval_metrics[key] != nan_value
        ].max()

    return {**mean_metrics, **var_metrics, **std_metrics, **min_metrics, **max_metrics}


def save_translations_json(
    imagined_plan_translation: str,
    reconstructed_plan_translation: str,
    actual_narration: str,
    imagined_metrics: Dict[str, float],
    reconstructed_metrics: Dict[str, float],
    episode_number: int,
    plan_start_t: int,
    plan_end_t: int,
    logdir: str,
    current_training_step: int,
    task_name: str,
) -> None:
    """Saves a JSON file in the provided logdir containing the imagined
    latent plan translation, reconstructed latent translation, and ground
    truth translation, along with the corresponding evaluation metrics.

    Args:
        imagined_plan_translation (str): Imagined latent plan translation

        reconstructed_plan_translation (str): Reconstructed latent plan
        translation

        actual_narration (str): Ground truth latent plan description.

        imagined_metrics (Dict[str, float]): Dictionary containing
        the imagined translation's evaluation metrics

        reconstructed_metrics (Dict[str, float]): Dictionary containing
        the reconstructed translation's evaluation metrics

        episode_number (int): Episode number in the current evaluation
        epoch.

        plan_start_t (int): Start timestep in the evaluation episode of
        the plans

        plan_end_t (int): Last timestep in the evaluation episode of the
        plans

        logdir (str): Directory to store the create JSON file.

        current_training_step (int): Current training step of the agent.

        task_name (str): name of the task, used to get task specific
        stopwords
    """

    output_folder = os.path.join(
        logdir,
        "evaluation",
        "latent_translations",
        f"step_{current_training_step}",
        f"eval_episode_{episode_number}",
    )
    os.makedirs(output_folder, exist_ok=True)

    actual_narration = remove_punctuation(actual_narration)
    actual_narration = actual_narration.lower()

    stop_words = get_task_stopwords(task_name)
    imagined_no_stopwords = [
        word for word in imagined_plan_translation.split() if word not in stop_words
    ]
    imagined_no_stopwords = " ".join(imagined_no_stopwords)
    reconstructed_no_stopwords = [
        word
        for word in reconstructed_plan_translation.split()
        if word not in stop_words
    ]
    reconstructed_no_stopwords = " ".join(reconstructed_no_stopwords)
    actual_no_stopwords = [
        word for word in actual_narration.split() if word not in stop_words
    ]
    actual_no_stopwords = " ".join(actual_no_stopwords)

    output_file_path = os.path.join(
        output_folder,
        f"eval_episode_{episode_number}_plan_{plan_start_t}_{plan_end_t}.json",
    )
    with open(output_file_path, "w") as f:
        json.dump(
            {
                "imagined_plan_translation": imagined_plan_translation,
                "reconstructed_plan_translation": reconstructed_plan_translation,
                "actual_narration": actual_narration,
                "imagined_plan_translation_no_stopwords": imagined_no_stopwords,
                "reconstructed_plan_translation_no_stopwords": reconstructed_no_stopwords,
                "actual_narration_no_stopwords": actual_no_stopwords,
                "imagined_metrics": imagined_metrics,
                "reconstructed_metrics": reconstructed_metrics,
            },
            f,
            indent=4,
        )


@torch.no_grad()
def save_decoded_plan_plot(
    agent,
    imagined_latent_states: List[torch.Tensor],
    reconstructed_latent_states: List[torch.Tensor],
    ground_truth_images: List[np.ndarray],
    episode_number: int,
    plan_start_t: int,
    plan_end_t: int,
    logdir: str,
    current_training_step: int,
) -> None:
    """Saves a plot of the decoded imagined and reconstructed latent
    state images, along with the corresponding ground-truth images
    that come from rolling out the plan.

    Args:
        agent: Trained World Model.

        imagined_latent_states (List[torch.Tensor]): The imagined
        latent state plan.

        reconstructed_latent_states (List[torch.Tensor]): The
        posterior latent state plan.

        ground_truth_images (List[np.ndarray]): The ground truth
        images obtained from rolling out the plan in the
        environment.

        episode_number (int): The current evaluation episode
        in the current evaluation epoch.

        plan_start_t (int): The start timestep in the evaluation
        episode of the plans.

        plan_end_t (int): The last timestep in the evaluation
        episode of the plans.

        logdir (str): Directory to store the created plot.

        current_training_step (int): The current training step
        of the agent.
    """

    output_folder = os.path.join(
        logdir,
        "evaluation",
        "latent_plan_plots",
        f"step_{current_training_step}",
        f"eval_episode_{episode_number}",
    )
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(
        output_folder,
        f"eval_episode_{episode_number}_plan_{plan_start_t}_{plan_end_t}.png",
    )

    imagined_images = [
        agent._wm.heads["decoder"](state)["image"].mode().detach().cpu().clone()
        for state in imagined_latent_states
    ]
    reconstructed_images = [
        agent._wm.heads["decoder"](state)["image"].mode().detach().cpu().clone()
        for state in reconstructed_latent_states
    ]
    imagined_images = convert_images_to_numpy(imagined_images)
    reconstructed_images = convert_images_to_numpy(reconstructed_images)

    reconstruction_plot: plt.Figure = generate_image_reconstruction_plot(
        [imagined_images, reconstructed_images, ground_truth_images],
        3,
        len(ground_truth_images),
        start_time=plan_start_t,
    )
    reconstruction_plot.suptitle(
        f"Eval Episode {episode_number} steps {plan_start_t} : {plan_end_t}"
    )
    reconstruction_plot.savefig(output_file_path)
    plt.close(reconstruction_plot)

    # Clean up
    del imagined_images
    del reconstructed_images
    del reconstruction_plot
    gc.collect()


def get_eval_summary_string(
    imagined_metrics: Dict[str, float],
    reconstructed_metrics: Dict[str, float],
    training_step: int,
) -> str:
    """Generates a string summarising this epoch's evaluation
    results.

    Args:
        imagined_metrics (Dict[str, float]): Evaluation metrics
        for the imagined latent plans

        reconstructed_metrics (Dict[str, float]): Evaluation
        metrics for the reconstructed latent plans.

        training_step (int): Current global training step.

    Returns:
        str: String summarising the evaluation results.
    """

    summary_string = f"Evaluation results at step {training_step}:\n"
    summary_string += "-" * 50 + "\n"

    summary_string += "Imagined Plan Metrics with Stopwords:\n"
    summary_string += "-" * 50 + "\n"
    summary_string += (
        f"\t  Mean Imagined BLEU: {imagined_metrics['mean_imagined_bleu_score']:.2f}\n"
    )
    summary_string += (
        f"\t  Mean Imagined METEOR: {imagined_metrics['mean_imagined_meteor']:.2f}\n"
    )
    summary_string += f"\t  Mean Imagined ROUGE 1 F: {imagined_metrics['mean_imagined_rouge1_fmeasure']:.2f}\n"
    summary_string += (
        f"\t  Mean Imagined WER: {imagined_metrics['mean_imagined_wer']:.2f}\n"
    )
    summary_string += (
        f"\t  Mean Imagined TER: {imagined_metrics['mean_imagined_ter']:.2f}\n"
    )
    summary_string += (
        f"\t  Mean Imagined chrF++: {imagined_metrics['mean_imagined_chrf']:.2f}\n"
    )
    summary_string += "-" * 50 + "\n"

    summary_string += "Reconstructed Plan Metrics with Stopwords:\n"
    summary_string += "-" * 50 + "\n"
    summary_string += f"\t  Mean Reconstructed BLEU: {reconstructed_metrics['mean_reconstructed_bleu_score']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed METEOR: {reconstructed_metrics['mean_reconstructed_meteor']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed ROUGE 1 F: {reconstructed_metrics['mean_reconstructed_rouge1_fmeasure']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed WER: {reconstructed_metrics['mean_reconstructed_wer']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed TER: {reconstructed_metrics['mean_reconstructed_ter']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed chrF++: {reconstructed_metrics['mean_reconstructed_chrf']:.2f}\n"
    summary_string += "-" * 50 + "\n"

    summary_string += "Imagined Plan Metrics without Stopwords:\n"
    summary_string += "-" * 50 + "\n"
    summary_string += f"\t  Mean Imagined BLEU: {imagined_metrics['mean_imagined_bleu_score_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Imagined METEOR: {imagined_metrics['mean_imagined_meteor_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Imagined ROUGE 1 F: {imagined_metrics['mean_imagined_rouge1_fmeasure_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Imagined WER: {imagined_metrics['mean_imagined_wer_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Imagined TER: {imagined_metrics['mean_imagined_ter_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Imagined chrF++: {imagined_metrics['mean_imagined_chrf_no_stopwords']:.2f}\n"
    summary_string += "-" * 50 + "\n"

    summary_string += "Reconstructed Plan Metrics without Stopwords:\n"
    summary_string += "-" * 50 + "\n"
    summary_string += f"\t  Mean Reconstructed BLEU: {reconstructed_metrics['mean_reconstructed_bleu_score_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed METEOR: {reconstructed_metrics['mean_reconstructed_meteor_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed ROUGE 1 F: {reconstructed_metrics['mean_reconstructed_rouge1_fmeasure_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed WER: {reconstructed_metrics['mean_reconstructed_wer_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed TER: {reconstructed_metrics['mean_reconstructed_ter_no_stopwords']:.2f}\n"
    summary_string += f"\t  Mean Reconstructed chrF++: {reconstructed_metrics['mean_reconstructed_chrf_no_stopwords']:.2f}\n"
    summary_string += "-" * 50 + "\n"

    return summary_string


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
    save_plots: bool = False,
    save_translations: bool = False,
) -> None:
    """Evaluates a set of prior (imagined) and posterior latent state
    plans.

    Args:
        agent (_type_): Dreamer agent that includes the world model and
        actor-critic.

        imagined_state_samples (List[List[torch.Tensor]]): List of
        imagined latent state plans. The outer list is for each eval
        episode, and the inner list is for each latent state in the
        episode.64

        imagined_action_samples (List[List[torch.Tensor]]): List of
        actions chosen in the imagined latent state plans.

        posterior_state_samples (List[List[torch.Tensor]]): List of
        posterior latent states, obtained when acting out the agent's
        plan in the environment.

        observation_samples (List[List[Dict[str, Any]]]): List of
        observations from the environment, obtained by rolling out the
        imagined latent plan.

        logger (_type_): Logger object created by the original Dreamer
        implemention. Used  get the current training step number.

        trajectory_length (int, optional): Number of steps in each. Latent
        plan (including the starting state) Defaults to 16.

        wandb_run (_type_, optional): WandB run object used to log the evaluation
        metrics to the WandB cloud. Defaults to None.

        save_plots (bool, optional): Whether to save a plot showing the decoded and
        ground truth images for each latent plan. Defaults to False. If True, the
        images are saved to the logdir of the agent.

        save_translations (bool, optional): Whether to save the ground-truth and
        translated latent plans to a json file. Defaults to False. If True, the
        translations are saved to the logdir of the agent.
    """
    config = agent._config

    episode_rewards: List[float] = []
    num_episode_plans: List[int] = []
    running_translation_eval_metrics: Dict[str, np.ndarray] = {}

    max_plans_per_episode = max(
        [
            len(imagined_state_samples[episode]) // trajectory_length
            for episode in range(len(imagined_state_samples))
        ]
    )
    print(f"Max plans per episode: {max_plans_per_episode}")

    for episode in range(len(imagined_state_samples)):
        episode_reward = 0.0
        num_plans = 0

        # print(f"SAMPLE LENGTH: {len(imagined_state_samples[sample])}")
        for index, trajectory in enumerate(
            range(0, len(imagined_state_samples[episode]), trajectory_length)
        ):
            end_index = trajectory + trajectory_length
            observations = observation_samples[episode][trajectory:end_index]
            # Adjust the end index if the environment terminated early
            for i, obs in enumerate(observations):
                if obs["obs"] is None:
                    end_index = trajectory + i
                    break
            observations = observation_samples[episode][trajectory:end_index]
            imagined_states = imagined_state_samples[episode][trajectory:end_index]
            posterior_states = posterior_state_samples[episode][trajectory:end_index]
            actions = imagined_action_samples[episode][trajectory:end_index]
            assert len(actions) == len(
                imagined_states
            ), "Actions and states must be the same length"
            # Happens when environment (or imagined trajectory) terminates early.
            if len(imagined_states) == 0 or len(observations) == 0:
                continue
            images = [obs["obs"]["image"] for obs in observations]
            rewards = [obs["reward"] for obs in observations]
            episode_reward += sum(rewards)

            if config.enable_language:
                narration_data = configure_narration_data(
                    config.narrator["narration_key"], observations, config.task
                )

                imagined_plan_translation = generate_translation(
                    agent,
                    config,
                    imagined_states,
                    actions,
                )
                reconstructed_plan_translation = generate_translation(
                    agent,
                    config,
                    posterior_states,
                    actions,
                )
                actual_narration = generate_narration(
                    agent, config.task, narration_data
                )

                imagined_translation_metrics = compute_translation_metrics(
                    imagined_plan_translation,
                    actual_narration,
                    config.task,
                )
                reconstructed_translation_metrics = compute_translation_metrics(
                    reconstructed_plan_translation,
                    actual_narration,
                    config.task,
                )

                update_running_metrics(
                    imagined_translation_metrics,
                    reconstructed_translation_metrics,
                    running_translation_eval_metrics,
                    episode,
                    config.n_translation_eval_episodes,
                    max_plans_per_episode,
                    index,
                )

                if save_translations:
                    save_translations_json(
                        imagined_plan_translation,
                        reconstructed_plan_translation,
                        actual_narration,
                        imagined_translation_metrics,
                        reconstructed_translation_metrics,
                        episode + 1,
                        trajectory,
                        end_index,
                        config.logdir,
                        logger.step,
                        config.task,
                    )

                if save_plots:
                    save_decoded_plan_plot(
                        agent,
                        imagined_states,
                        posterior_states,
                        images,
                        episode + 1,
                        trajectory,
                        end_index,
                        config.logdir,
                        logger.step,
                    )
            num_plans += 1
        episode_rewards.append(episode_reward)
        num_episode_plans.append(num_plans)

    if config.enable_language:
        imagined_metrics = {
            k: v for k, v in running_translation_eval_metrics.items() if "imagined" in k
        }
        reconstructed_metrics = {
            k: v
            for k, v in running_translation_eval_metrics.items()
            if "reconstructed" in k
        }
        imagined_metrics = compute_evaluation_statistics(imagined_metrics)
        reconstructed_metrics = compute_evaluation_statistics(reconstructed_metrics)

        if config.log_translation_eval:
            summary_string = get_eval_summary_string(
                imagined_metrics, reconstructed_metrics, logger.step
            )
            summary_string += f"Mean eval episode return: {np.mean(episode_rewards)}"
            print(summary_string)

        wandb_log = {
            **imagined_metrics,
            **reconstructed_metrics,
            "mean_episode_reward": np.mean(episode_rewards),
            "mean_plans_per_episode": np.mean(num_episode_plans),
        }

        if wandb_run is not None:
            wandb_run.log(wandb_log, step=logger.step)


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
    ignore_list = [
        "open",
        "close",
        "pickup",
        "put",
        "slice",
        "throw",
        "toggle_off",
        "toggle_on",
        "break",
        "drop",
        # "agent_position",
    ]
    if no_convert is not None:
        no_convert_list = no_convert
    if obs_to_ignore is not None:
        for ignore in obs_to_ignore:
            ignore_list.append(ignore)

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
    posterior_states, observations, _, _ = rollout_trajectory(
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


def display_image(image: np.ndarray, target_size: Tuple[int, int] = (600, 600)) -> None:
    """Displays an image to the screen using opencv

    Args:
        image (np.ndarray): image of shape (H, W, C)
        target_size (Tuple[int, int], optional): Image size to display. Defaults to (600,600).
    """

    resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img", cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def display_images_as_video(
    images, delay=100, window_name="Video Loop", target_size=(600, 600), loop=False
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

        if not loop:
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

        _, observations, posterior_states, _ = rollout_trajectory(
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
            if len(imagined_states) == 0 or len(observations) == 0:
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
        "make_stone_sword",
        "make_wood_pickaxe",
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
            if len(imagined_states) == 0 or len(observations) == 0:
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
                    * 10
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
                    * 10
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
                    * 10
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
                    * 10
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


def ai2thor_narration_using_obs_reconstruction(
    agent,
    env,
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
        env (_type_): AI2THOR environment.
        imagined_state_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        imagined_action_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        posterior_state_samples (List[List[torch.Tensor]]): (n_samples, n_trajectories * traj_length)
        observation_samples (List[List[Dict[str, Any]]]): (n_samples, n_trajectories * traj_length)
        trajectory_length (int, optional): Length of each trajectory. Defaults to 16.
    """
    reconstructed_bleu_scores = []
    filtererd_reconstructed_bleu_scores = []
    imagined_bleu_scores = []
    filtered_imagined_bleu_scores = []
    sample_rewards = []
    reverse_object_id_dict = env.env.inverse_object_ids
    interaction_vec_keys = [
        "pickup_vec",
        "drop_vec",
        "open_vec",
        "close_vec",
        "break_vec",
        "slice_vec",
        "toggle_on_vec",
        "toggle_off_vec",
        "throw_vec",
        "put_object_vec",
        "put_receptacle_vec",
    ]
    narrations_to_skip = [
        "I will start near the fridge and I wont move much I won't interact with any objects",
        "I will start near the stove and I wont move much I won't interact with any objects",
        "I will start near the sink and I wont move much I won't interact with any objects",
        "I will start near the far window and I wont move much I won't interact with any objects",
        "I will start near the near window and I wont move much I won't interact with any objects",
    ]

    for sample in range(len(imagined_state_samples)):
        sample_reward = 0.0
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
            rewards = [obs["reward"] for obs in observations]
            sample_reward += sum(rewards)
            # Happens when environment (or imagined trajectory) terminates early.
            if len(imagined_states) == 0 or len(observations) == 0:
                continue

            imagined_agent_positions: List[Tuple[float, float, float]] = []
            imagined_agent_interactions: Dict[str, List[Any]] = {
                "pickup": [],
                "drop": [],
                "open": [],
                "close": [],
                "break": [],
                "slice": [],
                "toggle_on": [],
                "toggle_off": [],
                "throw": [],
                "put_object": [],
                "put_receptacle": [],
                "put": [],
            }

            reconstructed_agent_interactions: Dict[str, List[Any]] = {
                "pickup": [],
                "drop": [],
                "open": [],
                "close": [],
                "break": [],
                "slice": [],
                "toggle_on": [],
                "toggle_off": [],
                "throw": [],
                "put_object": [],
                "put_receptacle": [],
                "put": [],
            }

            reconstructed_agent_positions: List[Tuple[float, float, float]] = []

            for state in range(len(imagined_states)):
                imagined_obs = agent._wm.heads["decoder"](imagined_states[state])
                reconstructed_obs = agent._wm.heads["decoder"](posterior_states[state])

                # Remove batch dim, convert to numpy, and then to tuple
                imagined_agent_positions.append(
                    tuple(imagined_obs["agent_position"].mode().cpu().squeeze().numpy())
                )
                reconstructed_agent_positions.append(
                    tuple(
                        reconstructed_obs["agent_position"]
                        .mode()
                        .cpu()
                        .squeeze()
                        .numpy()
                    )
                )

                for interaction_key in interaction_vec_keys:
                    imagined_interaction = (
                        imagined_obs[interaction_key].mode().round().clamp(0, 1)
                    )
                    reconstructed_interaction = (
                        reconstructed_obs[interaction_key].mode().round().clamp(0, 1)
                    )
                    # From one-hot encoding to object name
                    if torch.any(imagined_interaction):
                        object_name = reverse_object_id_dict[
                            imagined_interaction.argmax().item()
                        ]
                        # Remove _vec
                        imagined_agent_interactions[interaction_key[:-4]].append(
                            object_name
                        )
                    else:
                        imagined_agent_interactions[interaction_key[:-4]].append("")

                    if torch.any(reconstructed_interaction):
                        object_name = reverse_object_id_dict[
                            reconstructed_interaction.argmax().item()
                        ]
                        reconstructed_agent_interactions[interaction_key[:-4]].append(
                            object_name
                        )
                    else:
                        reconstructed_agent_interactions[interaction_key[:-4]].append(
                            ""
                        )

                try:
                    # Merge two put encodings into one as expected by the narrator
                    imagined_agent_interactions["put"].append(
                        (
                            imagined_agent_interactions["put_object"][state],
                            imagined_agent_interactions["put_receptacle"][state],
                        )
                    )

                    reconstructed_agent_interactions["put"].append(
                        (
                            reconstructed_agent_interactions["put_object"][state],
                            reconstructed_agent_interactions["put_receptacle"][state],
                        )
                    )
                # No put_object or put_receptacle key
                except IndexError:
                    pass

            # Remove the individual put_object and put_receptacle keys
            imagined_agent_interactions.pop("put_object")
            imagined_agent_interactions.pop("put_receptacle")
            reconstructed_agent_interactions.pop("put_object")
            reconstructed_agent_interactions.pop("put_receptacle")

            narrator = CookEggNarrator()
            narration_data = configure_narration_data(
                narration_keys=list(imagined_agent_interactions.keys())
                + ["agent_position"],
                observations=observations,
                task_name="ai2thor",
            )
            true_narration = generate_narration(
                agent, "ai2thor", narration_data, narrator
            )

            try:
                reconstructed_narration = narrator.narrate(
                    reconstructed_agent_positions, reconstructed_agent_interactions
                )
                reconstructed_bleu_score = float(
                    bleu_metric_from_strings(reconstructed_narration, true_narration)
                )

            except Exception as e:
                print(f"Failed to generated reconstructed narration: {e}")
                reconstructed_bleu_score = 0.0
            try:
                imagined_narration = narrator.narrate(
                    imagined_agent_positions, imagined_agent_interactions
                )
                imagined_bleu_score = float(
                    bleu_metric_from_strings(imagined_narration, true_narration)
                )

            except Exception as e:
                print(f"Failed to generated imagined narration: {e}")
                imagined_bleu_score = 0.0

            if true_narration not in narrations_to_skip:
                filtererd_reconstructed_bleu_scores.append(reconstructed_bleu_score)
                filtered_imagined_bleu_scores.append(imagined_bleu_score)

            reconstructed_bleu_scores.append(reconstructed_bleu_score)
            imagined_bleu_scores.append(imagined_bleu_score)

            print(
                f"Sample {sample} steps {trajectory} : {end_index} Imagined Plan: {imagined_narration}"
            )
            print(
                f"Sample {sample} steps {trajectory} : {end_index} Reconstructed Plan: {reconstructed_narration}"
            )
            print(
                f"Sample {sample} steps {trajectory} : {end_index} Actual Narration: {true_narration}"
            )
            print(
                f"Sample {sample} steps {trajectory} : {end_index} Imagined BLEU Score: {imagined_bleu_score}"
            )
            print(
                f"Sample {sample} steps {trajectory} : {end_index} Reconstructed BLEU Score: {reconstructed_bleu_score}"
            )

        sample_rewards.append(sample_reward)

    imagined_bleu_scores = np.array(imagined_bleu_scores)
    reconstructed_bleu_scores = np.array(reconstructed_bleu_scores)

    if len(imagined_bleu_scores) == 0:
        mean_imagined_score = 0.0
    else:
        mean_imagined_score = imagined_bleu_scores.mean()
    if len(reconstructed_bleu_scores) == 0:
        mean_posterior_score = 0.0
    else:
        mean_posterior_score = reconstructed_bleu_scores.mean()
    if len(filtererd_reconstructed_bleu_scores) == 0:
        filtered_mean_posterior_score = 0.0
    else:
        filtered_mean_posterior_score = np.array(
            filtererd_reconstructed_bleu_scores
        ).mean()
    if len(filtered_imagined_bleu_scores) == 0:
        filtered_mean_imagined_score = 0.0
    else:
        filtered_mean_imagined_score = np.array(filtered_imagined_bleu_scores).mean()

    mean_reward = np.array(sample_rewards).mean()

    wandb_run.log(
        {
            "obs_decoding_mean_imagined_bleu_score": mean_imagined_score,
            "obs_decoding_mean_posterior_bleu_score": mean_posterior_score,
            "obs_decoding_mean_reward": mean_reward,
            "obs_decoding_filtered_mean_posterior_bleu_score": filtered_mean_posterior_score,
            "obs_decoding_filtered_mean_imagined_bleu_score": filtered_mean_imagined_score,
        },
        step=logger.step,
    )


def visual_plan_evaluation(
    agent, env, output_folder: str, plan_length: int = 15
) -> None:

    env_done, imagined_done = None, None
    prev_state, prev_action = None, None
    prev_obs, info = env.reset()()
    config = agent._config
    no_convert = config.no_convert_list
    ignore = config.ignore_list
    plan_number = 1
    while not (env_done or imagined_done):
        initial_state = get_posterior_state(
            agent,
            prev_obs,
            no_convert,
            ignore,
            prev_state,
            prev_action,
        )
        imagined_states, imagined_actions, imagined_done = imagine_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=plan_length,
        )
        translated_plan_str = generate_translation(agent, config, imagined_states)

        with open(os.path.join(output_folder, "translated_plan.txt"), "a+") as f:
            f.write(f"{plan_number}. {translated_plan_str}\n")

        posterior_states, observations, posteriors, env_done = rollout_trajectory(
            agent=agent,
            initial_state=initial_state,
            trajectory_length=plan_length,
            actions=imagined_actions,
            env=env,
        )

        os.makedirs(os.path.join(output_folder, f"plan_step_{plan_number}_images"))

        # + 1 for starting state
        for plan_step in range(plan_length + 1):
            if plan_step == 0:
                img = prev_obs["image"]
            else:
                if observations[plan_step - 1]["obs"] is None:
                    break
                img = observations[plan_step - 1]["obs"]["image"]
            resized_img = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                (600, 600),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    f"plan_step_{plan_number}_images",
                    f"image_{plan_step + 1}.png",
                ),
                resized_img,
            )

        prev_state = posteriors[-1]
        prev_action = imagined_actions[-1]
        prev_obs = observations[-1]["obs"]
        plan_number += 1


def evaluate_consecutive_translations(
    agent,
    env,
    env_no_reset,
    plan_length: int = 15,
    output_path: str = "./consecutive_translation_evaluation.png",
    n_samples: int = 100,
    max_consecutive_plans: Optional[int] = None,
) -> None:
    """Evaluates the performance of the translator across consecutive
    plans, to see how translation performance is impacted by a degrading
    latent-state representation, due to the RNN having to remember more and
    more.

    Args:
        agent: Trained Dreamer Model.
        env: Environment to plan and rollout in.
        env_no_reset: Environment to rollout in without resetting the latent state.
        plan_length (int, optional): Number of steps in each plan. Defaults to 15.
        output_path (str, optional): Path to save the output plot. Defaults to "./consecutive_translation_evaluation.png".
        n_samples (int, optional): Number of samples to evaluate. Defaults to 100.
        max_consecutive_plans (Optional[int], optional): Optional maximum number of
        consecutive plans to evaluate. If None, keeps going until the episode terminates.
        Defaults to None.
    """
    bleu_sample_scores: List[List[float]] = [[] for _ in range(n_samples)]
    blue_sample_reset_scores: List[List[float]] = [[] for _ in range(n_samples)]
    bleu_sample_baseline_scores: List[List[float]] = [[] for _ in range(n_samples)]
    bleu_sample_reset_baseline_scores: List[List[float]] = [
        [] for _ in range(n_samples)
    ]

    for sample in tqdm(range(n_samples)):
        env_done, imagined_done = None, None
        env_reset_done, imagined_done_reset = None, None
        prev_state, prev_action = None, None
        prev_obs, info = env_no_reset.reset()()
        prev_obs_reset, info_reset = env.reset()()
        config = agent._config
        no_convert = config.no_convert_list
        ignore = config.ignore_list
        plan_number = 1
        bleu_scores: List[float] = []
        bleu_scores_reset: List[float] = []
        bleu_baseline_scores: List[float] = []
        bleu_baseline_scores_reset: List[float] = []
        while not (env_done or imagined_done) and not (
            env_reset_done or imagined_done_reset
        ):
            initial_state = get_posterior_state(
                agent,
                prev_obs,
                no_convert,
                ignore,
                prev_state,
                prev_action,
            )

            # To evaluate the performance impact of resetting the latent
            # state after each plan.
            initial_state_reset = get_posterior_state(
                agent,
                prev_obs_reset,
                no_convert,
                ignore,
                None,
                None,
            )

            imagined_states, imagined_actions, imagined_done = imagine_trajectory(
                agent=agent,
                initial_state=initial_state,
                trajectory_length=plan_length,
            )
            translated_plan_str = generate_translation(agent, config, imagined_states)

            imagined_states_reset, imagined_actions_reset, imagined_done_reset = (
                imagine_trajectory(
                    agent=agent,
                    initial_state=initial_state_reset,
                    trajectory_length=plan_length,
                )
            )

            translated_plan_str_reset = generate_translation(
                agent, config, imagined_states_reset
            )

            # Posterior states are in Tensor form, posteriors are in dictionary form.
            posterior_states, observations, posteriors, env_done = rollout_trajectory(
                agent=agent,
                initial_state=initial_state,
                trajectory_length=plan_length,
                actions=imagined_actions,
                env=env_no_reset,
            )

            (
                posterior_states_reset,
                observations_reset,
                posteriors_reset,
                env_reset_done,
            ) = rollout_trajectory(
                agent=agent,
                initial_state=initial_state_reset,
                trajectory_length=plan_length,
                actions=imagined_actions_reset,
                env=env,
            )

            # Filter out the None observations; obs are padded with Nones after the episode terminates
            observations = [obs for obs in observations if obs["obs"] is not None]
            observations_reset = [
                obs for obs in observations_reset if obs["obs"] is not None
            ]

            narration_data = configure_narration_data(
                config.narrator["narration_key"], observations, config.task
            )
            narration_data_reset = configure_narration_data(
                config.narrator["narration_key"], observations_reset, config.task
            )

            true_narration = generate_narration(agent, config.task, narration_data)
            true_narration_reset = generate_narration(
                agent, config.task, narration_data_reset
            )
            baseline_narration = get_narration_baseline_str(config.task)

            bleu_score = float(
                bleu_metric_from_strings(translated_plan_str, true_narration)
            )
            bleu_score_reset = float(
                bleu_metric_from_strings(
                    translated_plan_str_reset, true_narration_reset
                )
            )
            bleu_basline_score = float(
                bleu_metric_from_strings(baseline_narration, true_narration)
            )
            bleu_basline_score_reset = float(
                bleu_metric_from_strings(baseline_narration, true_narration_reset)
            )

            bleu_scores.append(bleu_score)
            bleu_scores_reset.append(bleu_score_reset)
            bleu_baseline_scores.append(bleu_basline_score)
            bleu_baseline_scores_reset.append(bleu_basline_score_reset)

            # print(f"Plan {plan_number} Translated Plan No Reset: {translated_plan_str}")
            # print(f"Plan {plan_number} True Narration No Reset: {true_narration}")
            # print(
            #     "-----------------------------------------------------------------------"
            # )
            # print(
            #     f"Plan {plan_number} Translated Plan Reset: {translated_plan_str_reset}"
            # )
            # print(f"Plan {plan_number} True Narration Reset: {true_narration_reset}")

            prev_state = posteriors[-1]
            prev_action = imagined_actions[-1]
            prev_obs = observations[-1]["obs"]
            prev_obs_reset = observations_reset[-1]["obs"]
            plan_number += 1
            if (
                max_consecutive_plans is not None
                and plan_number > max_consecutive_plans
            ):
                break

            bleu_sample_scores[sample].append(bleu_score)
            blue_sample_reset_scores[sample].append(bleu_score_reset)
            bleu_sample_baseline_scores[sample].append(bleu_basline_score)
            bleu_sample_reset_baseline_scores[sample].append(bleu_basline_score_reset)

    # From https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    # Compute mean and standard deviation of the BLEU scores

    # From https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    bleu_scores = np.zeros(
        [len(bleu_sample_scores), len(max(bleu_sample_scores, key=lambda x: len(x)))]
    )  # type: ignore
    bleu_scores_reset = np.zeros(
        [
            len(blue_sample_reset_scores),
            len(max(blue_sample_reset_scores, key=lambda x: len(x))),
        ]
    )  # type: ignore
    bleu_baseline_scores = np.zeros(
        [
            len(bleu_sample_baseline_scores),
            len(max(bleu_sample_baseline_scores, key=lambda x: len(x))),
        ]
    )  # type: ignore
    bleu_baseline_scores_reset = np.zeros(
        [
            len(bleu_sample_reset_baseline_scores),
            len(max(bleu_sample_reset_baseline_scores, key=lambda x: len(x))),
        ]
    )  # type: ignore

    for i, row in enumerate(bleu_sample_scores):
        bleu_scores[i, : len(row)] = row  # type: ignore
    for i, row in enumerate(blue_sample_reset_scores):
        bleu_scores_reset[i, : len(row)] = row  # type: ignore
    for i, row in enumerate(bleu_sample_baseline_scores):
        bleu_baseline_scores[i, : len(row)] = row  # type: ignore
    for i, row in enumerate(bleu_sample_reset_baseline_scores):
        bleu_baseline_scores_reset[i, : len(row)] = row  # type: ignore

    # Set 0s to NaNs
    bleu_scores[bleu_scores == 0] = np.nan
    bleu_scores_reset[bleu_scores_reset == 0] = np.nan
    bleu_baseline_scores[bleu_baseline_scores == 0] = np.nan
    bleu_baseline_scores_reset[bleu_baseline_scores_reset == 0] = np.nan

    bleu_scores = np.nanmean(bleu_scores, axis=0)
    bleu_scores_reset = np.nanmean(bleu_scores_reset, axis=0)
    bleu_baseline_scores = np.nanmean(bleu_baseline_scores, axis=0)
    bleu_baseline_scores_reset = np.nanmean(bleu_baseline_scores_reset, axis=0)

    bleu_score_std = bleu_scores.std(axis=0)  # type: ignore
    bleu_score_reset_std = bleu_scores_reset.std(axis=0)  # type: ignore
    bleu_baseline_score_std = bleu_baseline_scores.std(axis=0)  # type: ignore
    bleu_baseline_score_reset_std = bleu_baseline_scores_reset.std(axis=0)  # type: ignore

    bleu_data = {"No Reset": bleu_scores, "Reset": bleu_scores_reset}
    plan_labels = [f"Plan {i}" for i in range(1, len(bleu_scores) + 1)]
    x = np.arange(len(plan_labels))
    width = 0.4  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for experiment, score in bleu_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=experiment)
        # Add error bars
        ax.errorbar(
            x + offset,
            score,
            yerr=bleu_score_std if experiment == "No Reset" else bleu_score_reset_std,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
        )
        # Add baseline as red error bar
        ax.errorbar(
            x + offset,
            (
                bleu_baseline_scores
                if experiment == "No Reset"
                else bleu_baseline_scores_reset
            ),
            yerr=(
                bleu_baseline_score_std
                if experiment == "No Reset"
                else bleu_baseline_score_reset_std
            ),
            fmt="none",
            ecolor="red",
            capsize=5,
            capthick=2,
        )
        multiplier += 1

    ax.set_ylabel("BLEU Score [0-1]")
    ax.set_title("Plan BLEU Scores with and without Resetting Latent Starting State")
    ax.set_xticks(x + width, plan_labels)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, 1.2)

    plt.savefig(output_path)


if __name__ == "__main__":
    from dreamer import setup_args, create_environments, Dreamer
    from tools import recursively_load_optim_state_dict

    config = setup_args()
    user_env, env = create_environments(config)
    user_env = user_env[0]
    env = env[0]  # create_environments returns a list due to parallel possibility
    config.num_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=None,
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    rollout_samples = sample_rollouts(
        agent,
        env,
        1,
        trajectory_length=config.eval_trajectory_length,
        user_actions=True,
        user_env=user_env,
    )
    evaluate_rollouts(
        agent,
        rollout_samples["imagined_state_samples"],
        rollout_samples["imagined_action_samples"],
        rollout_samples["posterior_state_samples"],
        rollout_samples["observation_samples"],
        logger=None,
        trajectory_length=config.eval_trajectory_length,
        wandb_run=None,
    )
