import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random
from abc import ABC, abstractmethod
import json
from typing import List, Union

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


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
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
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
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
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
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            transition["encoded_image"] = info.get("encoded_image", None)
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
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
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
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
                # for k, v in episode.items():
                #     print(f"KEY {k}")
                #     for value in v:
                #         print(f"VALUE SHAPE {value.shape}")
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
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
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
                    print(episode["reward"])
                    print(episode["logprob"])

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
                    episode = np.load(f)
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
    max_length: int,
    padding_token: str = "<PAD>",
) -> np.ndarray:
    """Tokenise a text into words using a vocabulary.

    Args:
        text (list[str]): text to tokenise.
        vocab (dict): vocabulary to use for tokenisation.

    Returns:
        list[int]: tokenised text.
    """
    tokenised_text = []
    for sentence in text:
        sentence = sentence.replace(",", "").replace(".", "")
        tokenised_sentence: list[int] = [vocab[word] for word in sentence.split()]
        if tokenised_sentence[0] != vocab["<BOS>"]:
            tokenised_sentence.insert(0, vocab["<BOS>"])
        if tokenised_sentence[-1] != vocab["<EOS>"]:
            tokenised_sentence.append(vocab["<EOS>"])
        if len(tokenised_sentence) < max_length:
            tokenised_sentence.extend(
                [vocab[padding_token]] * (max_length - len(tokenised_sentence))
            )
        tokenised_text.append(tokenised_sentence)
    return np.array(tokenised_text, dtype=np.int32)


def generate_batch_narrations(
    narrator,
    observations: np.ndarray,
    obs_per_narration: int,
    max_narration_length: int,
    vocab: dict,
    device: torch.device,
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

    batch_length = observations.shape[1]
    narration_batches: List[np.ndarray] = []
    for batch in observations:
        narrations: List[str] = []
        for i in range(0, batch_length, obs_per_narration):
            narration = narrator.narrate(batch[i : i + obs_per_narration])
            narrations.append(narration)
        batch_arr = word_tokenise_text(narrations, vocab, max_narration_length)
        narration_batches.append(batch_arr)

    narration_arr = np.array(narration_batches)
    narrations_tens = torch.tensor(narration_arr, dtype=torch.long).to(device)
    return narrations_tens


class MiniGridNarrator(ABC):
    def __init__(self) -> None:
        super().__init__()

        # Object ID is channel 0 of the environment observation
        self._OBJECT_IDS = {
            "FLOOR_ID": 3,
            "DOOR_ID": 4,
            "KEY_ID": 5,
            "GOAL_ID": 8,
            "AGENT_ID": 10,
        }
        # Status is channel 2 of the environment observation
        self._STATUS_IDS = {
            "OPEN": 0,
            "CLOSED": 1,
            "LOCKED": 2,
        }

        self._COLOUR_IDS = {
            "red": 0,
            "green": 1,
            "blue": 2,
            "purple": 3,
            "yellow": 4,
            "grey": 5,
        }

        self._ID_TO_COLOUR = dict(
            zip(self._COLOUR_IDS.values(), self._COLOUR_IDS.keys())
        )

    def _get_object_location(
        self, observation: np.ndarray, object_id: int
    ) -> list[tuple]:
        """
        Returns the location of the object in the observation
        """
        observation = observation[:, :, 0]  # Remove colour and status info
        object_locations = np.nonzero(observation == object_id)
        locations: list[tuple] = []
        # third dimension is info
        for (
            col,
            row,
        ) in zip(*object_locations):
            locations.append((col, row))
        return locations

    def _calculate_distance(
        self, location1: tuple, location2: tuple, metric: str = "manhattan"
    ) -> Union[int, float]:
        """
        Returns the distance between two locations
        """
        if metric == "manhattan":
            return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
        else:
            return float(np.linalg.norm(np.array(location1) - np.array(location2)))

    def _agent_moved(self, observations: list[np.ndarray]) -> bool:
        """
        Returns whether the agent moved in the sequence of observations.
        """
        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]
        for i in range(1, len(observations)):
            try:
                agent_current_position = self._get_object_location(
                    observations[i], self._OBJECT_IDS["AGENT_ID"]
                )[0]
                if agent_current_position != agent_start_position:
                    return True
            except IndexError:
                return False
        return False

    def _get_agent_relative_movement_string(
        self,
        observations: list[np.ndarray],
        object_position: tuple[int, int],
        object_name: str,
    ) -> str:
        """_summary_

        Args:
            observations (list[np.ndarray]): _description_
            object_position (tuple[int, int]): _description_
            object_name (str): _description_

        Returns:
            str: _description_
        """

        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        if not self._agent_moved(observations):
            return "the agent did not move "

        agent_end_position = self._get_object_location(
            observations[-1], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        if agent_end_position == agent_start_position:
            return "the agent moved in a circle "

        start_distance = self._calculate_distance(agent_start_position, object_position)
        end_distance = self._calculate_distance(agent_end_position, object_position)

        if start_distance == end_distance:
            return (
                f"the agent stayed the same distance from the {object_name}, but moved "
            )
        elif start_distance > end_distance:
            return f"the agent moved towards the {object_name} "
        else:
            return f"the agent moved away from the {object_name} "

    @abstractmethod
    def narrate(self, observations: list[np.ndarray]) -> str:
        pass


class MiniGridFourSquareNarrator(MiniGridNarrator):
    def narrate(self, observations: list[np.ndarray]) -> str:
        first_obs = observations[0]
        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
            if (
                self._get_object_location(
                    observations[-1], self._OBJECT_IDS["AGENT_ID"]
                )[0]
                == goal_position
            ):
                return "the agent reached the goal "
        except IndexError:
            # Agent is standing on goal
            return "the agent reached the goal "
        if not self._agent_moved(observations):
            return "the agent did not move "
        if (
            self._get_object_location(observations[-1], self._OBJECT_IDS["AGENT_ID"])[0]
            == self._get_object_location(observations[0], self._OBJECT_IDS["AGENT_ID"])[
                0
            ]
        ):
            return "the agent moved in a circle "

        agent_start_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        agent_end_position = self._get_object_location(
            observations[-1], self._OBJECT_IDS["AGENT_ID"]
        )[0]

        goal_position = self._get_object_location(
            observations[0], self._OBJECT_IDS["GOAL_ID"]
        )[0]

        coloured_square_positions = self._get_object_location(
            observations[0], self._OBJECT_IDS["FLOOR_ID"]
        )

        coloured_square_positions.append(goal_position)

        goal_colour = self._ID_TO_COLOUR[
            first_obs[goal_position[0], goal_position[1], 1]
        ]

        biggest_delta = 0.0
        closest_square = None

        for square_position in coloured_square_positions:
            square_colour = self._ID_TO_COLOUR[
                first_obs[square_position[0], square_position[1], 1]
            ]
            delta = self._calculate_distance(
                square_position, agent_start_position
            ) - self._calculate_distance(square_position, agent_end_position)
            if delta > biggest_delta:
                biggest_delta = delta
                closest_square = square_colour

        if closest_square == goal_colour:
            return f"the agent moved towards the {closest_square} square which is the goal "

        return f"the agent moved towards the {closest_square} square which is not the goal "


class MiniGridEmptyNarrator(MiniGridNarrator):

    def narrate(self, observations: list[np.ndarray]) -> str:
        first_obs = observations[0]
        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
        except IndexError:
            # Agent is standing on goal
            return "the agent reached the goal "

        if (
            self._get_object_location(observations[-1], self._OBJECT_IDS["AGENT_ID"])[0]
            == goal_position
        ):
            return "the agent reached the goal "

        return self._get_agent_relative_movement_string(
            observations,
            self._get_object_location(observations[0], self._OBJECT_IDS["GOAL_ID"])[0],
            "goal",
        )


class MiniGridDoorKeyNarrator(MiniGridNarrator):

    def _get_key_status(
        self, first_obs: np.ndarray, last_obs: np.ndarray
    ) -> tuple[bool, int]:
        """Determines whether the key has already been picked up, has not been
        picked up, or if the agent picked up the key in a given window of
        observations.

        Args:
            first_obs (np.ndarray): encoded environment observation of first timestep
            of window. Shape (height, width, 3)
            last_obs (np.ndarray): encoded environment observation of last timestep
            of window. Shape (height, width, 3)

        Returns:
            tuple[bool, bool]: (agent_has_key, agent_picked_up_key)
        """

        agent_has_key = False
        agent_picked_up_key = False

        key_start_position = self._get_object_location(
            first_obs, self._OBJECT_IDS["KEY_ID"]
        )
        key_end_position = self._get_object_location(
            last_obs, self._OBJECT_IDS["KEY_ID"]
        )

        if key_start_position and not key_end_position:
            agent_picked_up_key = True
            agent_has_key = True
        elif not key_start_position:
            agent_has_key = True

        return agent_has_key, agent_picked_up_key

    def _get_key_pickup_frame(self, observations: list[np.ndarray]) -> int:
        """
        Returns the frame in which the agent picked up the key.
        """
        for i, obs in enumerate(observations):
            if not self._get_object_location(obs, self._OBJECT_IDS["KEY_ID"]):
                return i
        return -1

    def _get_door_unlock_frame(self, observations: list[np.ndarray]) -> int:
        """
        Returns the frame in which the agent unlocked the door.
        """
        for i, obs in enumerate(observations):
            try:
                door_position = self._get_object_location(
                    obs, self._OBJECT_IDS["DOOR_ID"]
                )[0]
            except IndexError:
                # Agent is standing on door
                door_position = self._get_object_location(
                    obs, self._OBJECT_IDS["AGENT_ID"]
                )[0]
            if obs[door_position[0], door_position[1], 2] != self._STATUS_IDS["LOCKED"]:
                return i
        return -1

    def _get_last_door_change_frame(
        self, observations: list[np.ndarray], door_position: tuple[int, int]
    ) -> int:
        """Returns the frame in which the door was last opened or closed.

        Args:
            observations (list[np.ndarray]): list of observatons to check for door
            door_position (tuple[int, int]): position of the door in the observation

        Returns:
            int: frame number
        """

        for i in range(len(observations) - 1, -1, -1):
            if (
                observations[i][door_position[0], door_position[1], 2]
                != observations[-1][door_position[0], door_position[1], 2]
            ):
                return i

    def _get_door_lock_status(
        self,
        first_obs: np.ndarray,
        last_obs: np.ndarray,
        door_position: tuple[int, int],
    ) -> tuple[bool, bool]:
        """Gets whether the door is locked or unlocked, and whether the agent unlocked
        it in the current window of environment steps.

        Args:
            first_obs (np.ndarray): first observation in the window of shape
            (height, width, 3)
            last_obs (np.ndarray): last observation in the window of shape
            (height, width, 3)
            door_position (tuple[int, int]): position of the door in the observation
            (row, col)

        Returns:
            tuple[bool, bool]: door_locked, agent_unlocked_door
        """

        door_locked = False
        agent_unlocked_door = False

        initial_status = first_obs[door_position[0], door_position[1], 2]
        final_status = last_obs[door_position[0], door_position[1], 2]

        if initial_status == self._STATUS_IDS["LOCKED"]:
            door_locked = True
            if final_status != self._STATUS_IDS["LOCKED"]:
                agent_unlocked_door = True
                door_locked = False

        return door_locked, agent_unlocked_door

    def _get_door_open_close_sequence(
        self, observations: list[np.ndarray], door_position: tuple[int, int]
    ) -> str:
        """
        Returns a string describing the sequence of door open and close events.
        """
        door_open_close_sequence = ""
        current_status = observations[0][door_position[0], door_position[1], 2]
        door_changed = False
        for i in range(1, len(observations)):
            next_status = observations[i][door_position[0], door_position[1], 2]
            if next_status != current_status:
                if (
                    next_status == self._STATUS_IDS["OPEN"]
                    and current_status == self._STATUS_IDS["CLOSED"]
                ):
                    if door_changed:
                        door_open_close_sequence += "and then "
                    door_open_close_sequence += "the agent opened the door "
                    door_changed = True
                    current_status = next_status
                elif next_status == self._STATUS_IDS["CLOSED"]:
                    if door_changed:
                        door_open_close_sequence += "and then "
                    door_open_close_sequence += "the agent closed the door "
                    door_changed = True
                    current_status = next_status
        return door_open_close_sequence

    def narrate(self, observations: list[np.ndarray]) -> str:
        """
        Generates a narration from a sequence of observations.
        """
        narration_str = ""

        first_obs = observations[0]
        last_obs = observations[-1]

        agent_has_key, agent_picked_up_key = self._get_key_status(first_obs, last_obs)

        if agent_picked_up_key:
            narration_str += "the agent went and picked up the key, and then "
            pickup_frame = self._get_key_pickup_frame(observations)
            observations = observations[pickup_frame + 1 :]
            if not observations:
                return narration_str[: len("and then ")]
        elif not agent_has_key:
            # Get movement of agent relative to key
            key_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["KEY_ID"]
            )[0]
            narration_str += self._get_agent_relative_movement_string(
                observations, key_position, "key"
            )
            return narration_str
        try:
            door_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["DOOR_ID"]
            )[0]
        except IndexError:
            # Agent is standing on door
            door_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["AGENT_ID"]
            )[0]
        door_locked, agent_unlocked_door = self._get_door_lock_status(
            first_obs, last_obs, door_position
        )
        if door_locked:
            narration_str += self._get_agent_relative_movement_string(
                observations, door_position, "door"
            )
            return narration_str

        elif agent_unlocked_door:
            door_locked = False
            narration_str += "the agent unlocked the door, and then "
            door_unlock_frame = self._get_door_unlock_frame(observations)
            observations = observations[door_unlock_frame + 1 :]
            if not observations:
                return narration_str[: len("and then ")]

        if not door_locked:
            door_open_close_sequence = self._get_door_open_close_sequence(
                observations, door_position
            )
            narration_str += door_open_close_sequence
            if door_open_close_sequence != "":
                narration_str += "and then "
                door_last_change_frame = self._get_last_door_change_frame(
                    observations, door_position
                )
                observations = observations[door_last_change_frame + 1 :]
                if not observations:
                    return narration_str[: len("and then ")]

        try:
            goal_position = self._get_object_location(
                first_obs, self._OBJECT_IDS["GOAL_ID"]
            )[0]
        except IndexError:
            # Agent is standing on goal
            narration_str += "the agent reached the goal "
            return narration_str

        if (
            self._get_object_location(last_obs, self._OBJECT_IDS["AGENT_ID"])[0]
            == goal_position
        ):
            narration_str += "the agent reached the goal "
        else:
            narration_str += self._get_agent_relative_movement_string(
                observations, goal_position, "goal "
            )

        return narration_str


def narration_loss(
    predicted_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    pad_idx: int = 0,
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
    # vocab = {
    #     "<PAD>": 0,
    #     "<BOS>": 1,
    #     "<EOS>": 2,
    #     "<UNK>": 3,
    #     "the": 4,
    #     "agent": 5,
    #     "reached": 6,
    #     "goal": 7,
    #     "did": 8,
    #     "not": 9,
    #     "move": 10,
    #     "moved": 11,
    #     "in": 12,
    #     "a": 13,
    #     "circle": 14,
    #     "stayed": 15,
    #     "same": 16,
    #     "distance": 17,
    #     "but": 18,
    #     "from": 19,
    #     "away": 20,
    #     "towards": 21,
    #     "square": 22,
    #     "blue": 23,
    #     "grey": 24,
    #     "green": 25,
    #     "purple": 26,
    #     "which": 27,
    #     "is": 28,
    # }

    # Reshape to (batch_size, seq_len, target_vocab_size)
    predicted_tokens = predicted_tokens.permute(1, 0, 2)

    # batch_0_pred = predicted_tokens[0]
    # batch_0_pred = torch.argmax(batch_0_pred, dim=1).cpu().numpy()
    # batch_0_pred = [
    #     list(vocab.keys())[list(vocab.values()).index(i)] for i in batch_0_pred
    # ]
    # batch_0_actual = true_tokens[0].cpu().numpy()
    # batch_0_actual = [
    #     list(vocab.keys())[list(vocab.values()).index(i)] for i in batch_0_actual
    # ]

    # print("Predicted: ", batch_0_pred)
    # print("Actual: ", batch_0_actual)

    # reshape to (batch_size * seq_len, target_vocab_size)
    predicted_tokens = predicted_tokens.reshape(-1, predicted_tokens.shape[-1])
    # reshape to (batch_size * seq_len)
    true_tokens = true_tokens.reshape(-1)

    return nn.CrossEntropyLoss(ignore_index=pad_idx)(predicted_tokens, true_tokens)
