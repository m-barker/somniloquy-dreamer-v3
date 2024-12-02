import timeit
import collections
import io
import os
import json
import string
import pathlib
import time
import random
import json
from typing import List, Union, Optional, Tuple, Dict
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.text import Perplexity, BLEUScore

import wandb

from parallel import Parallel, Damy

to_np = lambda x: x.detach().cpu().numpy()
NARRATION_COUNTS: Dict[str, int] = {}


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def add_batch_to_obs(obs):
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs[k] = np.expand_dims(v, axis=0)
        else:
            obs[k] = np.array([v])
    return obs


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name}: {self.interval:.6f} seconds")
        else:
            print(f"Time elapsed: {self.interval:.6f} seconds")


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def simulate(
    agent,
    envs: List[Union[Parallel, Damy]],
    cache: dict,
    directory: pathlib.PurePath,
    logger: Logger,
    is_eval: bool = False,
    limit: Optional[int] = None,
    steps: int = 0,
    episodes: int = 0,
    state: Optional[Tuple] = None,
    no_convert_obs: Optional[List[str]] = None,
    no_save_obs: Optional[List[str]] = None,
    info_keys_to_store: Optional[List[str]] = None,
    wandb_run=None,
    config=None,
    train_env=None,
) -> Tuple:
    """Runs agent interaction with the environment.

    Args:
        agent (_type_): Random Agent or Dreamer Agent.

        envs (List[Union[Parallel, Damy]]): List of environments to step.

        cache (dict): Dictionary of episodes which is added to by this function.

        directory (pathlib.PurePath): Path to save episodes.

        logger (Logger): Logger object to record metrics.

        is_eval (bool, optional): Whether the agent is being trained or evaluated. Defaults to False.

        limit (Optional[int], optional): Max Cache size. Defaults to None.

        steps (int, optional): Total steps to run for each environment. Defaults to 0.

        episodes (int, optional): Total episodes to run for each environment. Defaults to 0.

        state (Optional[Tuple], optional): Previous world model state used for the Actor.
        To decide on which action to take. Defaults to None.

        no_convert_obs (Optional[List[str]], optional): List of observation keys to not convert. Defaults to None.

        no_save_obs (Optional[List[str]], optional): List of observation keys to not save. Defaults to None.

        info_keys_to_store (Optional[List[str]], optional): List of info keys to store. Defaults to None.

    Returns:
        Tuple: Tuple containing (
            Remaining steps,
            Remaining episodes,
            Done status of envs,
            Length of simulation steps,
            Observations seen in simulation,
            Agent State for next simulation,
            Reward from simulation
        )
    """
    ignore = no_save_obs if no_save_obs else []
    no_convert = no_convert_obs if no_convert_obs else []
    info_keys = info_keys_to_store if info_keys_to_store else []
    with Timer("Simulate Function"):
        # initialize or unpack simulation state
        if state is None:
            step, episode = 0, 0
            done = np.ones(len(envs), bool)
            length = np.zeros(len(envs), np.int32)
            obs = [None] * len(envs)
            agent_state = None
            reward = [0] * len(envs)

        else:
            step, episode, done, length, obs, agent_state, reward = state
        while (steps and step < steps) or (episodes and episode < episodes):
            # reset envs if necessary
            if done.any():
                indices = [index for index, d in enumerate(done) if d]
                results = [envs[i].reset() for i in indices]
                results = [r() for r in results]
                for index, result in zip(indices, results):
                    o, info = result
                    o = {
                        k: (v if k in no_convert else convert(v))
                        for k, v in o.items()
                        if k not in ignore
                    }
                    t = o.copy()
                    # action will be added to transition in add_to_cache
                    t["reward"] = 0.0
                    t["discount"] = 1.0
                    # initial state should be added to cache

                    for key in info_keys:
                        if key in info:
                            t[key] = info[key]

                    add_to_cache(cache, envs[index].id, t, no_convert=no_convert)

                    # replace obs with done by initial state
                    obs[index] = result[0]
            # step agents

            if config is not None and config.conditional_actions:
                from evaluation import get_posterior_state

                if agent_state is None:
                    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}  # type: ignore
                    obs = agent._wm.preprocess(obs, keys_to_ignore=["privileged_obs"])
                    embed = agent._wm.encoder(obs)
                    starting_state, _ = agent._wm.dynamics.obs_step(
                        None, None, embed, obs["is_first"]
                    )
                    prev_action = None
                    posterior = None
                else:
                    starting_state = agent_state[0]
                    prev_action = agent_state[1].squeeze()
                    posterior = starting_state
                if config.conditional_policy_attempts == 0:
                    p = np.random.random()
                    if p > config.conditional_epsilon:
                        actions, log_probs = conditional_policy(agent, starting_state, policy_only=True)  # type: ignore
                    else:
                        actions, log_probs = conditional_policy(agent, starting_state, policy_attempts=config.conditional_policy_attempts)  # type: ignore
                else:
                    actions, log_probs = conditional_policy(agent, starting_state)
                for index, action in enumerate(actions):
                    if type(obs) != type({}):
                        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}  # type: ignore
                    _, _ = agent(obs, done, agent_state)
                    action = [
                        {
                            "action": action,
                            "logprob": (
                                log_probs[index].detach().cpu().numpy()
                                if isinstance(log_probs[index], torch.Tensor)
                                else log_probs[index]
                            ),
                        }
                    ]  # type: ignore
                    results = [e.step(a) for e, a in zip(envs, action)]
                    results = [r() for r in results]

                    if (
                        agent._config.task == "safegym_island_navigation"
                        and logger.step % 100 == 0
                    ):
                        wandb_run.log(
                            {
                                "water_incidents": envs[0]._env.num_water_incidents
                                + train_env._env.num_water_incidents
                            },
                            step=logger.step,
                        )

                    if prev_action is not None:
                        prev_action = torch.Tensor(prev_action).to(device=config.device)
                    obs, reward, done = zip(*[p[:3] for p in results])  # type: ignore
                    posterior = get_posterior_state(
                        agent,
                        obs[0],
                        no_convert=no_convert,
                        obs_to_ignore=ignore,
                        prev_state=posterior,
                        prev_action=(
                            prev_action.unsqueeze(0)
                            if prev_action is not None
                            else None
                        ),
                    )
                    prev_action = action[0]["action"]

                    if len(action[0]["action"].shape) > 1:
                        action[0]["action"] = action[0]["action"].squeeze()

                    obs = list(obs)
                    reward = list(reward)
                    done = np.stack(done)  # type: ignore
                    episode += int(done.sum())
                    length += 1
                    step += len(envs)
                    length *= 1 - done
                    # add to cache
                    for a, result, env in zip(action, results, envs):
                        o, r, d, info = result
                        o = {
                            k: (v if k in no_convert else convert(v))
                            for k, v in o.items()
                            if k not in ignore
                        }
                        transition = o.copy()
                        if isinstance(a, dict):
                            transition.update(a)
                        else:
                            transition["action"] = a
                        transition["reward"] = r
                        transition["discount"] = info.get(
                            "discount", np.array(1 - float(d))
                        )

                        for key in info_keys:
                            if key in info:
                                transition[key] = info[key]

                        add_to_cache(cache, env.id, transition, no_convert=no_convert)
                    agent_state = (
                        posterior,
                        torch.Tensor(action[0]["action"])
                        .unsqueeze(0)
                        .to(device=config.device),
                    )
                    if done.any():
                        break
                agent_state = (
                    posterior,
                    torch.Tensor(action[0]["action"])
                    .unsqueeze(0)
                    .to(device=config.device),
                )
            else:
                obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}  # type: ignore
                # agent_state is tuple (latent, action), where latent is the latent dict and
                # action is a dict of action, logprob
                action, agent_state = agent(obs, done, agent_state)
                if isinstance(action, dict):
                    action = [
                        {k: np.array(action[k][i].detach().cpu()) for k in action}
                        for i in range(len(envs))
                    ]
                else:
                    action = np.array(action)
                assert len(action) == len(envs)
                # step envs
                results = [e.step(a) for e, a in zip(envs, action)]
                results = [r() for r in results]

                if (
                    config is not None
                    and config.task == "safegym_island_navigation"
                    and logger.step % 100 == 0
                ):
                    wandb_run.log(
                        {
                            "water_incidents": envs[0]._env.num_water_incidents
                            + train_env._env.num_water_incidents
                        },
                        step=logger.step,
                    )

                # print(f"RESULTS: {results}")
                obs, reward, done = zip(*[p[:3] for p in results])  # type: ignore
                obs = list(obs)
                reward = list(reward)
                done = np.stack(done)  # type: ignore
                episode += int(done.sum())
                length += 1
                step += len(envs)
                length *= 1 - done
                # add to cache
                for a, result, env in zip(action, results, envs):
                    o, r, d, info = result
                    o = {
                        k: (v if k in no_convert else convert(v))
                        for k, v in o.items()
                        if k not in ignore
                    }
                    transition = o.copy()
                    if isinstance(a, dict):
                        transition.update(a)
                    else:
                        transition["action"] = a
                    transition["reward"] = r
                    transition["discount"] = info.get(
                        "discount", np.array(1 - float(d))
                    )

                    for key in info_keys:
                        if key in info:
                            transition[key] = info[key]

                    add_to_cache(cache, env.id, transition, no_convert=no_convert)

            if done.any():
                indices = [index for index, d in enumerate(done) if d]
                # logging for done episode
                for i in indices:
                    # with Timer("Saving episode"):
                    save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                    length = len(cache[envs[i].id]["reward"]) - 1  # type: ignore
                    score = float(np.array(cache[envs[i].id]["reward"]).sum())
                    video = cache[envs[i].id]["image"]
                    # record logs given from environments
                    for key in list(cache[envs[i].id].keys()):
                        if "log_" in key:
                            logger.scalar(
                                key, float(np.array(cache[envs[i].id][key]).max())
                            )
                            # log items won't be used later
                            cache[envs[i].id].pop(key)

                    if not is_eval:
                        step_in_dataset = erase_over_episodes(cache, limit)  # type: ignore
                        logger.scalar(f"dataset_size", step_in_dataset)
                        logger.scalar(f"train_return", score)
                        logger.scalar(f"train_length", length)
                        logger.scalar(f"train_episodes", len(cache))
                        logger.write(step=logger.step)

                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "dataset_size": step_in_dataset,
                                    "train_return": score,
                                    "train_length": length,
                                    "train_episodes": len(cache),
                                },
                                step=logger.step,
                            )

                    else:
                        if not "eval_lengths" in locals():
                            eval_lengths = []
                            eval_scores = []
                            eval_done = False
                        # start counting scores for evaluation
                        eval_scores.append(score)
                        eval_lengths.append(length)

                        score = sum(eval_scores) / len(eval_scores)
                        length = sum(eval_lengths) / len(eval_lengths)  # type: ignore
                        logger.video(f"eval_policy", np.array(video)[None])

                        if len(eval_scores) >= episodes and not eval_done:
                            logger.scalar(f"eval_return", score)
                            logger.scalar(f"eval_length", length)
                            logger.scalar(f"eval_episodes", len(eval_scores))
                            logger.write(step=logger.step)
                            if wandb_run is not None:
                                wandb_run.log(
                                    {
                                        "eval_return": score,
                                        "eval_length": length,
                                        "eval_episodes": len(eval_scores),
                                    },
                                    step=logger.step,
                                )
                            eval_done = True
        if is_eval:
            # keep only last item for saving memory. this cache is used for video_pred later
            while len(cache) > 1:
                # FIFO
                cache.popitem(last=False)  # type: ignore
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(
    cache: dict, id: str, transition: dict, no_convert: Optional[List[str]] = None
) -> None:
    """Adds a transition to the cache, in the episode
    specified by a given ID.

    Args:action
        cache (dict): Dictionary of episodes.
        id (str): The unique ID of the episode to which the
        transition pertains.
        transition (dict): dictionary of transition elements
        containing the keys 'action', 'reward', 'discount', "image", etc.

    """
    no_convert_keys = no_convert if no_convert else []
    # Cache contains current image, current reward, previous action
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            if key in no_convert_keys:
                cache[id][key] = [val]
            else:
                cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                if key in no_convert_keys:
                    if type(val) == Dict:
                        cache[id][key] = [0]
                        cache[id][key].append(val)
                    else:
                        cache[id][key] = [val]
                        cache[id][key].append(val)
                else:
                    cache[id][key] = [convert(0 * val)]
                    cache[id][key].append(convert(val))
            else:
                if key in no_convert_keys:
                    cache[id][key].append(val)
                else:
                    cache[id][key].append(convert(val))


def erase_over_episodes(cache: dict, dataset_size: int) -> int:
    """If required, removes the oldest episodes from the cache

    Args:
        cache (dict): Cache of episodes.
        dataset_size (int): Maximum number of transitions to keep in the cache.

    Returns:
        int: The number of transitions in the dataset.
    """
    step_in_dataset = 0
    # Reversed as we want to keep the most recent episodes
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        print(value)
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory: pathlib.PurePath, episodes: dict) -> bool:
    """Saves the transitions of the episodes to the directory as npz files.

    Args:
        directory (pathlib.PurePath): The directory to save the episodes.
        episodes (dict): The episodes to save -- nested dictionary where
        the outer keys are the episode names and the inner keys are the
        transition item names; transitons are stored as numpy arrays (or lists?).
    Returns
        True.

    """
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])

            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                try:
                    ret = {
                        k: np.append(
                            ret[k],
                            v[index : min(index + possible, total)].copy(),
                            axis=0,
                        )
                        for k, v in episode.items()
                        if "log_" not in k
                    }
                except ValueError:
                    for k, v in episode.items():
                        print(f"KEY {k}")
                        for value in v:
                            print(f"VALUE SHAPE {value.shape}")
                    raise ValueError
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f, allow_pickle=True)
                    episode = {k: episode[k] for k in episode.keys()}

            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f, allow_pickle=True)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def load_json_vocab(file_path: str) -> dict:
    """Load a vocabulary from a JSON file.

    Args:
        file_path (str): path to the JSON file.

    Returns:
        dict: vocabulary as a dictionary.
    """
    with open(file_path, "r", encoding="utf8") as file:
        vocab = json.load(file)
    return vocab


def word_tokenise_text(
    text: list[str],
    vocab: dict,
    required_length: Optional[int] = None,
    padding_token: str = "<PAD>",
) -> np.ndarray:
    """Tokenise a text into words using a vocabulary.

    Args:
        text (list[str]): text to tokenise.
        vocab (dict): vocabulary to use for tokenisation.
        required_length: Optional[int]: optional required
        length for each tokenised string. Adds padding to reach this
        length if needed.
        padding_token (str, optional): Padding token to insert if
        required_length is not None. Defaults to "<PAD>".

    Returns:
        np.ndarray: array of shape (n_sentences, required_length) containing
        token integers.
    """
    tokenised_text = []
    for sentence in text:
        sentence = sentence.replace(",", "").replace(".", "").replace("'", "")
        sentence = sentence.lower()
        tokenised_sentence: list[int] = [vocab[word] for word in sentence.split()]
        if tokenised_sentence[0] != vocab["<BOS>"]:
            tokenised_sentence.insert(0, vocab["<BOS>"])
        if tokenised_sentence[-1] != vocab["<EOS>"]:
            tokenised_sentence.append(vocab["<EOS>"])
        if required_length is not None and len(tokenised_sentence) < required_length:
            tokenised_sentence.extend(
                [vocab[padding_token]] * (required_length - len(tokenised_sentence))
            )
        tokenised_text.append(tokenised_sentence)
    for sequence in tokenised_text:
        if required_length is not None and len(sequence) != required_length:
            raise ValueError(
                f"Sequence length {len(sequence)} does not match required length {required_length}, sequence: {sequence}"
            )
    return np.array(tokenised_text, dtype=np.int32)


def generate_batch_narrations(
    narrator,
    observations: Union[np.ndarray, torch.Tensor, Dict],
    obs_per_narration: int,
    max_narration_length: int,
    vocab: dict,
    device: torch.device,
    is_first: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Generates a batch of narrations given a batch
    of observations.

    Args:
        narrator (_type_): narrator object used to generate narrations
        from a given sequence of observations.
        observations (np.ndarray): observation array of shape
        (batch_size, batch_length, *obs_shape).
        obs_per_narration (int): number of observations to use per narration.
        max_narration_length (int): maximum length of a narration.
        vocab (dict): vocabulary to use for tokenisation.
        device (torch.device): device to put the tensors that are returned on.

    Returns:
        torch.Tensor: narrations tensor of shape (batch_size, batch_length // obs_per_narration, max_narration_length).
    """

    narration_batches: List[np.ndarray] = []
    global NARRATION_COUNTS
    if type(is_first) == torch.Tensor:
        is_first = is_first.detach().cpu().numpy()
    if isinstance(observations, dict):
        for batch_idx in range(list(observations.values())[0].shape[0]):
            batch = {key: value[batch_idx] for key, value in observations.items()}
            batch_length = list(batch.values())[0].shape[0]
            narrations: List[str] = []
            is_first_batch = is_first[batch_idx]
            assert is_first_batch[0] == 1
            is_first_indices = np.where(is_first_batch == 1)[0]
            current_index = 0
            current_is_first_index = 1
            while current_index < batch_length:
                if len(is_first_indices) > current_is_first_index:
                    # Determine the next index to use
                    end_index = min(
                        current_index + obs_per_narration,
                        batch_length,
                        is_first_indices[current_is_first_index],
                    )
                    narration = narrator.narrate(
                        {
                            key: value[current_index:end_index]
                            for key, value in batch.items()
                        }
                    )
                    if narration in NARRATION_COUNTS:
                        NARRATION_COUNTS[narration] += 1
                    else:
                        NARRATION_COUNTS[narration] = 1
                    narrations.append(narration)
                    current_index = end_index
                    if current_index == is_first_indices[current_is_first_index]:
                        current_is_first_index += 1
                else:
                    end_index = min(current_index + obs_per_narration, batch_length)
                    narration = narrator.narrate(
                        {
                            key: value[current_index:end_index]
                            for key, value in batch.items()
                        }
                    )
                    if narration in NARRATION_COUNTS:
                        NARRATION_COUNTS[narration] += 1
                    else:
                        NARRATION_COUNTS[narration] = 1
                    narrations.append(narration)
                    current_index = end_index
            batch_arr = word_tokenise_text(narrations, vocab, max_narration_length)
            narration_batches.append(batch_arr)

    else:
        for idx, batch in enumerate(observations):
            narrations: List[str] = []  # type: ignore
            is_first_batch = is_first[idx]
            assert is_first_batch[0] == 1
            is_first_indices = np.where(is_first_batch == 1)[0]
            current_index = 0
            current_is_first_index = 1
            while current_index < len(batch):
                if len(is_first_indices) > current_is_first_index:
                    # Determine the next index to use
                    end_index = min(
                        current_index + obs_per_narration,
                        len(batch),
                        is_first_indices[current_is_first_index],
                    )
                    narration = narrator.narrate(batch[current_index:end_index])
                    if narration in NARRATION_COUNTS:
                        NARRATION_COUNTS[narration] += 1
                    else:
                        NARRATION_COUNTS[narration] = 1
                    narrations.append(narration)
                    current_index = end_index
                    if current_index == is_first_indices[current_is_first_index]:
                        current_is_first_index += 1
                else:
                    end_index = min(current_index + obs_per_narration, len(batch))
                    narration = narrator.narrate(batch[current_index:end_index])
                    if narration in NARRATION_COUNTS:
                        NARRATION_COUNTS[narration] += 1
                    else:
                        NARRATION_COUNTS[narration] = 1
                    narrations.append(narration)
                    current_index = end_index
            batch_arr = word_tokenise_text(narrations, vocab, max_narration_length)
            narration_batches.append(batch_arr)

    narration_arr = np.concatenate(narration_batches, axis=0)
    narrations_tens = torch.tensor(narration_arr, dtype=torch.long).to(device)
    # print(f"NARRATION COUNTS: {NARRATION_COUNTS}")
    return narrations_tens


def narration_loss(
    predicted_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    pad_idx: int = 0,
    debug: bool = False,
) -> torch.Tensor:
    """Returns the cross entropy loss between the predicted and true tokens.

    Args:
        predicted_tokens (torch.Tensor): The predicted tokens of shape
        (seq_len, batch_size, target_vocab_size)
        true_tokens (torch.Tensor): The true tokens of shaoe
        (batch_size, seq_len)
        pad_idx (int): The padding index.

    Returns:
        torch.Tensor: The cross entropy loss between the predicted and true tokens.
    """
    T, N, V = predicted_tokens.shape
    assert N, T == true_tokens.shape

    # (seq, batch, vocab) -> (batch, seq, vocab)
    predicted_tokens = predicted_tokens.permute(1, 0, 2)
    if debug:
        for batch in range(N):
            batch_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)(
                predicted_tokens[batch], true_tokens[batch]
            )
            print(f"BATCH_{batch} LOSS: {batch_loss:.2f}")
            batch_predicted_tokens = predicted_tokens[batch].argmax(dim=-1)
            print(batch_predicted_tokens)
            print(true_tokens[batch])

    predicted_tokens = predicted_tokens.reshape(-1, V)
    # reshape to (batch_size * seq_len)
    true_tokens = true_tokens.reshape(-1)
    predicted_tokens_argmax = predicted_tokens.argmax(dim=-1)

    return nn.CrossEntropyLoss(ignore_index=pad_idx)(predicted_tokens, true_tokens)


def ctc_loss(
    input_seq_logits: torch.Tensor,
    target_seq: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Computes the CTCLoss between the input and target sequences.

    Args:
        input_seq_logits (torch.Tensor): Shape (T, N, V)
        target_seq (torch.Tensor): Shape (B, T)
        padding_value (int, optional): _description_. Defaults to 0.

    Returns:
        torch.Tensor: _description_
    """
    loss = nn.CTCLoss(blank=200)

    # Need to tell this loss the lengths of each individual sequence
    # (i.e., the index before the first padding token)
    input_seq = input_seq_logits.argmax(dim=-1)
    input_padding_mask = (input_seq == padding_value).long()
    target_padding_mask = (target_seq == padding_value).long()
    # Argmax returns the value of the first maximum if ties
    input_seq_lengths = input_padding_mask.argmax(dim=0)
    target_seq_lengths = target_padding_mask.argmax(dim=-1)

    print(f"Input Seq logits shape: {input_seq_logits.shape}")
    print(f"Input lengths shape: {input_seq_lengths.shape}")
    print(f"Target sequence shape: {target_seq.shape}")
    print(f"Target lengths shape: {target_seq_lengths.shape}")

    return loss(input_seq_logits, target_seq, input_seq_lengths, target_seq_lengths)


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
    remove_punctuation: bool = True,
) -> torch.Tensor:
    """Computes the bleu score between a translated and true string.

    Args:
        predicted_sequence (str): Machine translated string
        true_sequence (str): Ground truth string
        n_gram (int, optional): Number of n-grams to consider. Defaults to 4.
        convert_case (bool, optional): Whether to convert the strings to lower
        case (BLEU is not case insensitive). Defaults to True.
        remove_punctuation (bool, optional): Whether to remove punctuation from
        the strings. Defaults to True.

    Returns:
        torch.Tensor: BLEU score in range [0,1]
    """

    if convert_case:
        predicted_sequence = predicted_sequence.lower()
        true_sequence = true_sequence.lower()
    if remove_punctuation:
        # From https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
        predicted_sequence = predicted_sequence.translate(
            str.maketrans("", "", string.punctuation)
        )
        true_sequence = true_sequence.translate(
            str.maketrans("", "", string.punctuation)
        )

    metric = BLEUScore(n_gram=n_gram)
    metric.update(
        predicted_sequence, [true_sequence]
    )  # there can be multiple referenence strings
    return metric.compute()


@torch.no_grad()
def conditional_policy(
    agent,
    starting_state: Dict[str, torch.Tensor],
    trajectory_length: int = 15,
    condition: str = "uh oh i will go into the water and drown",
    condition_check: bool = False,
    policy_attempts: int = 10,
    random_attempts: int = 20,
    policy_only: bool = False,
) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
    """Plans a sequence of actions conditional on a string description of what should or
    shouldn't occur.

    Args:
        agent (_type_): Dreamer agent.

        starting_state: Dict[str, torch.Tensor]: Encoded starting latent state of the environment.
        trajectory_length (int, optional): Sequence length of actions. Defaults to 15.

        condition (str, optional): String description to check. Defaults to "uh oh i will go into the water and drown".

        condition_check (bool, optional): Whether the condition must be true or false. Defaults to false. If false, this
        asks the question "plan a sequence of actions that won't cause the conditional description".

        policy_attempts (int, optional): Number of times the policy should be sampled from to try and meet the condition.
        Once this is exceeded, result to a random policy that tries to meet the condition. Defaults to 10.

        random_attempts (int, optional): Number fo times the random policy should be sampled from to try and meet the
        condition. If this is exceeded, then a condition violating action sequence is returned, to allow train# ing
        to continue. Defaults to 20.

        policy_only: (bool, optional) Whether to generate a sequence according to the policy and return, ignoring the condition.
        Used as a basline.

    Returns:
        List[torch.Tensor]: list of actions to take.
    """

    condition_satisfied = False
    total_policy_attempts = 0
    total_random_attempts = 0
    # Just want one sequence of actions, chosen according to the policy.
    if policy_only:
        total_policy_attempts = policy_attempts - 1
        total_random_attempts = random_attempts
    while not condition_satisfied and total_random_attempts < random_attempts:
        planned_actions: List[np.ndarray] = []
        log_probs = []
        latent_state = agent._wm.dynamics.get_feat(starting_state).unsqueeze(0)
        imagined_states: List[torch.Tensor] = [latent_state]
        prev_state = starting_state
        for t in range(trajectory_length):
            if total_policy_attempts < policy_attempts:
                policy = agent._task_behavior.actor(latent_state)
                action = policy.sample().squeeze(0)
                log_prob = policy.log_prob(action)
            else:
                int_action = np.random.randint(agent._config.num_actions)
                action_arr = np.zeros(agent._config.num_actions)
                action_arr[int_action] = 1  # one-hot encoding
                action = torch.Tensor(action_arr).to(agent._config.device)
                log_prob = torch.Tensor([1.0]).numpy()
            log_probs.append(log_prob.squeeze())
            planned_actions.append(action.detach().cpu().squeeze().numpy())

            # assert len(log_probs[-1].shape) == 0, print(log_probs[-1])
            assert len(planned_actions[-1].shape) == 1

            if len(action.shape) < 2:
                action = action.unsqueeze(dim=0)
            prior = agent._wm.dynamics.img_step(
                prev_state=prev_state,
                prev_action=action,
            )
            prev_state = prior
            latent_state = agent._wm.dynamics.get_feat(prior).unsqueeze(0)
            imagined_states.append(latent_state)

            predicted_continue = agent._wm.heads["cont"](latent_state).mode()
            # Less than 50% predicted chance that the episode continues according to world model.
            if predicted_continue[0, 0].detach().cpu().numpy() < 0.5:
                break

        if total_policy_attempts < policy_attempts:
            total_policy_attempts += 1
        else:
            total_random_attempts += 1

        # (T, N, C) -> (N, T, C)
        imagined_state_tensor = torch.cat(imagined_states, dim=0).permute(1, 0, 2)

        # our batch size is 1 so take first item in list
        planned_intent = agent._wm.heads["language"].generate(
            imagined_state_tensor,
            agent._wm.vocab,
            agent._config.dec_max_length,
            sampling_method=agent._config.token_sampling_method,
        )[0]
        planned_intent = " ".join(
            [
                word
                for word in planned_intent.split()
                if word not in ["<BOS>", "<EOS>", "<PAD>"]
            ]
        )

        if condition_check:
            condition_satisfied = planned_intent == condition
        else:
            condition_satisfied = planned_intent != condition

    print(f"Planning conditional policy... imagined sequence: {planned_intent}")
    print(f"Number of actions: {len(planned_actions)}")
    return planned_actions, log_probs
