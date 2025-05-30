import math
from typing import Optional, Union, Tuple, Dict, Any
import re
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

        print(
            f"Number of GRU Parameters: {sum(p.numel() for p in self._cell.parameters())}"
        )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_state_dict(self, state):
        """This is the inverse of the get_feat function"""
        if self._discrete:
            stoch = state[:, : self._stoch * self._discrete]
            stoch = stoch.reshape(
                list(stoch.shape[:-1]) + [self._stoch, self._discrete]
            )
            deter = state[:, self._stoch * self._discrete :]
        else:
            stoch = state[:, : self._stoch]
            deter = state[:, self._stoch :]
        return {"stoch": stoch, "deter": deter}

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]  # Creates a new axis.
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class TokenEmbedding(nn.Module):
    """Class for converting token IDs of a fixed vocabulary size into vectors of a fixed length"""

    def __init__(self, vocab_size: int, emb_dim: int, padding_idx: int = 0):
        """Confugures the nn.embedding layer. The embedding layer has a learnable
        weight parameter of shape (vocab_size, emb_dim) whihc is initialised
        as a standard normal distribution, with mean 0 and unit varience.
        see https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            emb_dim (int): Dimensionality of the embedding.
            padding_idx (int, optional): Token ID to ignore when backpropping.
            Defaults to 0.
        """
        # The padding index prevents gradients from being propagated to the
        # embedding parameters for padding tokens
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    """Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Adds positional encoding to a given sequential input tensor.

        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        target_vocab_size: int = 22,
        encoder_bottleneck: bool = True,
        bottleneck_input_size: int = 1024,
        bos_token: int = 1,
        eos_token: int = 2,
        padding_token: int = 0,
        src_token_embedding: bool = False,
        src_vocab_size: Optional[int] = None,
    ) -> None:
        """Transformer encoder-decoder model used for translation.

        Args:
            d_model (int): dimensionality of the embedded tokens, for both the encoder
            and the decoder.

            n_head (int): Number of heads for the multi-head attention. Defaults to 2.

            num_encoder_layers (int): number of stacked encoder layers to use. Defaults to 2.

            num_decoder_layers (int): number of stacked decoder layers to use. Defaults to 2.

            dim_feedforward (int, optional): dimension of the feed-forward network. Defaults to 256.

            dropout (float, optional): frequency of dropout. Defaults to 0.1.

            activation (str, optional): activation function of encoder/decoder intermediate layer.
                                        Defaults to "relu".

            target_vocab_size (int, optional): number of tokens in the target vocabulary. Defaults to 22.

            encoder_bottleneck (bool, optional): whether to use an linear layer to embed the src input
            to a compressed space. Defaults to True.

            bottleneck_input_size (int, optional): size of the input to the encoder bottleneck. Defaults to 1024.

            bos_token (int, optional): beginning of sentence token. Defaults to 1.

            eos_token (int, optional): end of sentence token. Defaults to 2.

            padding_token (int, optional): padding token. Defaults to 0.

            src_token_embedding (bool, optional): whether to use a Token embedding layer for the source tokens. Defaults to False.

            src_vocab_size (Optional[int], optional): size of the source vocabulary. Defaults to None.

        """
        super(TransformerEncoderDecoder, self).__init__()
        self._initial_embed: Optional[nn.Linear] = None

        # Optionally add a fully connected layer to embed the encoder input
        # to a vector of length embed_size.
        if encoder_bottleneck:
            self._initial_embed = nn.Linear(bottleneck_input_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self._target_vocab_size = target_vocab_size

        self._bos_token = bos_token
        self._eos_token = eos_token
        self._padding_token = padding_token
        self.final_layer = nn.Linear(d_model, self._target_vocab_size)
        self.pe = PositionalEncoding(d_model, dropout)
        self.tgt_embedding = TokenEmbedding(self._target_vocab_size, d_model)
        self.src_embedding = None
        if src_token_embedding:
            assert (
                src_vocab_size is not None
            ), "src_vocab_size must be provided if using src token embeddings."
            self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Transformer Initialised with the following parameters:\n")
        print(f"  d_model: {d_model}")
        print(f"  n_head: {n_head}")
        print(f"  num_encoder_layers: {num_encoder_layers}")
        print(f"  num_decoder_layers: {num_decoder_layers}")
        print(f"  dim_feedforward: {dim_feedforward}")
        print(f"  dropout: {dropout}")
        print(f"  activation: {activation}")
        print(f"  target_vocab_size: {target_vocab_size}")
        print(f"  encoder_bottleneck: {encoder_bottleneck}")
        print(f"  bottleneck_input_size: {bottleneck_input_size}")
        print("-" * 80)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        generate_mask: bool = False,
        src_mask: Optional[torch.Tensor] = None,
        tokens_to_append: Optional[torch.Tensor] = None,
        tokens_to_prepend: Optional[torch.Tensor] = None,
        embed_tgt: bool = True,
        generate_src_mask: bool = False,
    ) -> torch.Tensor:
        """_summary_

        Args:
            src (torch.Tensor): shape (batch_size, seq_length, d)

            tgt (torch.Tensor): shape (batch_size, seq_length)

            generate_mask (bool, optional): Whether to generate padding and attention masks. Defaults to False.

            src_mask: (torch.Tensor, optional): Optional mask for src padding where True indicates padding tokens
            to be ignored. Defaults to None.

            tokens_to_append (torch.Tensor, optional): optional tokens to append to the end of the encoder
            sequence. Defaults to None.

            tokens_to_prepend (torch.Tensor, optional): optional tokens to pre-prend to the beginning of the
            encoder sequence. Defaults to None.

            embed_tgt (bool, optional). Whether the provided tgt sequence needs to be embedded, i.e.,
            if integer tokens are given. Defalts to True.

            generate_src_mask (bool, optional). Whether to auto-generate padding masks for the src input.
            Defaults to False.


        Returns:
            torch.Tensor: logits of shape (out_seq_length, batch_size, vocab_size)
        """
        if self._initial_embed is not None:
            src = self._initial_embed(src)

        elif self.src_embedding is not None:
            src = self.src_embedding(src)

        tgt_pad_mask = None
        src_pad_mask = None

        if generate_mask:
            tgt_pad_mask = tgt == (self._padding_token or self._eos_token)

        if embed_tgt:
            tgt = self.tgt_embedding(tgt)

        if src_mask is not None:
            src_pad_mask = src_mask

        if src_pad_mask is None and generate_src_mask:
            src_pad_mask = src == self._padding_token

        # (batch_size, seq_length, d) -> (seq_length, batch_size, d)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        src = self.pe(src)
        tgt = self.pe(tgt)

        tgt_mask = None
        if generate_mask:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
                self.device
            )

        if tokens_to_append is not None:
            src = torch.cat([src, tokens_to_append.unsqueeze(0)], dim=0)

        if tokens_to_prepend is not None:
            src = torch.cat([tokens_to_prepend.unsqueeze(0), src], dim=0)

        # Convert pad_masks to a float masks
        if tgt_pad_mask is not None:
            tgt_pad_mask = tgt_pad_mask.float().masked_fill(
                tgt_pad_mask == 1, float("-inf")
            )
        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask.float().masked_fill(
                src_pad_mask == 1, float("-inf")
            )

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            src_key_padding_mask=src_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        out = self.final_layer(out)

        return out

    def _nuclueus_sampling(self, logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
        """Performs nucleus sampling on the logits. Heavily based on the
        implementation https://nn.labml.ai/sampling/nucleus.html
        and https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b

        and first proposed in Holtzman et al. (2019, https://arxiv.org/abs/1904.09751)

        Args:
            logits (torch.Tensor): logits of shape (batch_size, vocab_size)
            p (float, optional): Nucleus probability mass. Defaults to 0.9.

        Returns:
            torch.Tensor: sampled token
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -float("Inf")
        logits_remapped = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

        return torch.multinomial(F.softmax(logits_remapped, dim=-1), 1)

    @torch.no_grad()
    def generate(
        self,
        input_seq: torch.Tensor,
        vocab: dict,
        max_sequence_length: int,
        sampling_method: str = "nucleus",
        return_tokens: bool = False,
        prompt: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        tokens_to_prepend: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[str, np.ndarray, Tuple[Union[str, np.ndarray], torch.Tensor]]:
        """Generate a sequence of tokens from the input sequence. And convert the token IDs to words.

        Args:
            input_seq (torch.Tensor): shape (batch_length, seq_length, d_model)

            vocab (dict): dictionary mapping token IDs to words.

            max_sequence_length (int): maximum length of the generated sequence.

            sampling_method (str, optional): sampling method to use. Defaults to "nucleus".
            Other options are "greedy".

            return_tokens (bool, optional): whether to return the generated tokens. Defaults to False.

            prompt (Optional[torch.Tensor], optional): optional prompt to prepend to the input sequence.
            Defaults to None.

            return_logits (bool, optional): whether to return the logits. Defaults to False.

            tokens_to_prepend (Optional[torch.Tensor], optional): optional tokens to prepend to the input sequence.

            src_padding_mask (Optional[torch.Tensor], optional): optional padding mask for the input sequence.
            Defaults to None. Assumed to be a Boolean that is True for padding tokens of shape (batch_size, seq_length).

        Returns:
            str: the generated sequence of words.
        """
        self.eval()
        logits = []
        if prompt is None:
            batch_size = input_seq.size(0)
            translated_input = torch.tensor(
                [self._bos_token] * batch_size, device=self.device
            ).unsqueeze(1)
        else:
            translated_input = prompt

        for _ in range(max_sequence_length):
            output_logits = self.forward(
                input_seq,
                translated_input,
                generate_mask=True,
                tokens_to_prepend=tokens_to_prepend,
                src_mask=src_padding_mask,
            )[-1]
            logits.append(output_logits)
            output_probs = F.softmax(output_logits, dim=-1)
            if sampling_method == "greedy":
                predicted_token_ids = torch.argmax(output_probs, dim=-1)
                # Add batch dimension if missing (i.e., if batch size is 1).
                if len(predicted_token_ids.shape) == 1:
                    predicted_token_ids = predicted_token_ids.unsqueeze(0)
            elif sampling_method == "nucleus":
                predicted_token_ids = self._nuclueus_sampling(output_logits)
            else:
                raise NotImplementedError(
                    f"Sampling method {sampling_method} not found."
                )
            translated_input = torch.cat(
                [translated_input, predicted_token_ids[:, -1].unsqueeze(1)],
                dim=1,
            )

        # Convert the output tokens to a string
        translated_input = translated_input.detach().cpu().numpy()
        logits = torch.stack(logits)  # type: ignore

        # Anything after the first EOS token is ignored, so convert to padding
        for batch_num, batch in enumerate(translated_input):
            eos_idx = np.where(batch == self._eos_token)[0]
            if eos_idx.size > 0:
                batch[eos_idx[0] + 1 :] = self._padding_token
                # Logits don't have the <BOS> token hence index is one less
                logits[eos_idx[0] :, batch_num, :] = 0.0
                logits[eos_idx[0] :, batch_num, self._padding_token] = 1.0
        narrations = []
        for batch in translated_input:
            narration = [
                list(vocab.keys())[list(vocab.values()).index(token_id)]
                for token_id in batch
            ]
            narration = " ".join(narration)
            narrations.append(narration)
            translation = narrations  # type: ignore
        self.train()
        if return_logits:
            return translation, logits  # type: ignore
        return translation  # type: ignore


# class Attention(nn.Module):
#     """

#     Taken from the paper by Nguyen et. al. (2022)
#     Taken from their GitHub repository:
#     https://github.com/Fsoft-AIC/Automated-Rationale-Generation/blob/main/strategy_rationale/models/Attention.py

#     Applies an attention mechanism on the output features from the decoder.
#     """

#     def __init__(self, dim):
#         super(Attention, self).__init__()
#         self.dim = dim
#         self.linear1 = nn.Linear(dim * 2, dim)
#         self.linear2 = nn.Linear(dim, 1, bias=False)
#         #self._init_hidden()

#     def _init_hidden(self):
#         nn.init.xavier_normal_(self.linear1.weight)
#         nn.init.xavier_normal_(self.linear2.weight)

#     def forward(self, hidden_state, encoder_outputs):
#         """
#         Arguments:
#             hidden_state {Variable} -- batch_size x dim
#             encoder_outputs {Variable} -- batch_size x seq_len x dim

#         Returns:
#             Variable -- context vector of size batch_size x dim
#         """
#         batch_size, seq_len, _ = encoder_outputs.size()
#         hidden_state = hidden_state.repeat(1, seq_len, 1)
#         inputs = torch.cat((encoder_outputs, hidden_state),
#                            2).view(-1, self.dim * 2)
#         o = self.linear2(torch.tanh(self.linear1(inputs)))
#         e = o.view(batch_size, seq_len)
#         alpha = F.softmax(e, dim=1)
#         context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
#         return context

# class EncoderRNN(nn.Module):
#     """From the paper by Nguyen et. al. (2022)

#     Taken from their GitHub repository:
#     https://github.com/Fsoft-AIC/Automated-Rationale-Generation/blob/main/strategy_rationale/models/EncoderRNN.py

#     """
#     def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
#                  n_layers=1, bidirectional=False, rnn_cell='lstm'):
#         """

#         Args:
#             hidden_dim (int): dim of hidden state of rnn
#             input_dropout_p (int): dropout probability for the input sequence
#             dropout_p (float): dropout probability for the output sequence
#             n_layers (int): number of rnn layers
#             rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
#         """
#         super(EncoderRNN, self).__init__()
#         self.dim_vid = dim_vid
#         self.dim_hidden = dim_hidden
#         self.input_dropout_p = input_dropout_p
#         self.rnn_dropout_p = rnn_dropout_p
#         self.n_layers = n_layers
#         self.bidirectional = bidirectional
#         self.rnn_cell = rnn_cell

#         self.vid2hid = nn.Linear(dim_vid, dim_hidden)
#         self.input_dropout = nn.Dropout(input_dropout_p)

#         if rnn_cell.lower() == 'lstm':
#             self.rnn_cell = nn.LSTM
#         elif rnn_cell.lower() == 'gru':
#             self.rnn_cell = nn.GRU

#         self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
#                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)

#         self._init_hidden()

#     def _init_hidden(self):
#         nn.init.xavier_normal_(self.vid2hid.weight)

#     def forward(self, vid_feats):
#         """
#         Applies a multi-layer RNN to an input sequence.
#         Args:
#             input_var (batch, seq_len): tensor containing the features of the input sequence.
#             input_lengths (list of int, optional): A list that contains the lengths of sequences
#               in the mini-batch
#         Returns: output, hidden
#             - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
#             - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
#         """
#         batch_size, seq_len, dim_vid = vid_feats.size()
#         vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
#         vid_feats = self.input_dropout(vid_feats)
#         vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
#         self.rnn.flatten_parameters()
#         output, hidden = self.rnn(vid_feats)
#         return output, hidden


# class DecoderRNN(nn.Module):
#     """
#     From the paper by Nguyen et. al. (2022)
#     Taken from their GitHub repository:
#     https://github.com/Fsoft-AIC/Automated-Rationale-Generation/blob/main/strategy_rationale/models/DecoderRNN.py


#     Provides functionality for decoding in a seq2seq framework, with an option for attention.
#     Args:
#         vocab_size (int): size of the vocabulary
#         max_len (int): a maximum allowed length for the sequence to be processed
#         dim_hidden (int): the number of features in the hidden state `h`
#         n_layers (int, optional): number of recurrent layers (default: 1)
#         rnn_cell (str, optional): type of RNN cell (default: gru)
#         bidirectional (bool, optional): if the encoder is bidirectional (default False)
#         input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
#         rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

#     """

#     def __init__(self,
#                  vocab_size,
#                  max_len,
#                  dim_hidden,
#                  dim_word,
#                  n_layers=1,
#                  rnn_cell='lstm',
#                  bidirectional=False,
#                  input_dropout_p=0.1,
#                  rnn_dropout_p=0.1):
#         super(DecoderRNN, self).__init__()

#         self.bidirectional_encoder = bidirectional

#         self.dim_output = vocab_size
#         self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
#         self.dim_word = dim_word
#         self.max_length = max_len
#         self.sos_id = 1
#         self.eos_id = 0
#         self.input_dropout = nn.Dropout(input_dropout_p)
#         self.embedding = nn.Embedding(self.dim_output, dim_word)
#         self.attention = Attention(self.dim_hidden)
#         if rnn_cell.lower() == 'lstm':
#             self.rnn_cell = nn.LSTM
#         elif rnn_cell.lower() == 'gru':
#             self.rnn_cell = nn.GRU
#         self.rnn = self.rnn_cell(
#             self.dim_hidden + dim_word,
#             self.dim_hidden,
#             n_layers,
#             batch_first=True,
#             dropout=rnn_dropout_p)

#         self.out = nn.Linear(self.dim_hidden, self.dim_output)

#         self._init_weights()

#     def forward(self,
#                 encoder_outputs,
#                 encoder_hidden,
#                 targets=None,
#                 mode='train',
#                 opt={}):
#         """

#         Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
#         - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
#           hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
#         - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
#         - **targets** (batch, max_length): targets labels of the ground truth sentences

#         Outputs: seq_probs,
#         - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
#         - **seq_preds** (batch_size, max_length): predicted symbols
#         """
#         sample_max = opt.get('sample_max', 1)
#         beam_size = opt.get('beam_size', 1)
#         temperature = opt.get('temperature', 1.0)

#         batch_size, _, _ = encoder_outputs.size()
#         decoder_hidden = self._init_rnn_state(encoder_hidden)

#         seq_logprobs = []
#         seq_preds = []
#         self.rnn.flatten_parameters()
#         if mode == 'train':
#             # use targets as rnn inputs
#             targets_emb = self.embedding(targets)
#             for i in range(self.max_length - 1):
#                 current_words = targets_emb[:, i, :]
#                 context = self.attention(decoder_hidden[0].squeeze(0).unsqueeze(1), encoder_outputs)
#                 decoder_input = torch.cat([current_words, context], dim=1)
#                 decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
#                 decoder_output, decoder_hidden = self.rnn(
#                     decoder_input, decoder_hidden)
#                 logprobs = F.log_softmax(
#                     self.out(decoder_output.squeeze(1)), dim=1)
#                 seq_logprobs.append(logprobs.unsqueeze(1))

#             seq_logprobs = torch.cat(seq_logprobs, 1)

#         elif mode == 'inference':
#             if beam_size > 1:
#                 return self.sample_beam(encoder_outputs, decoder_hidden, opt)
#             for t in range(self.max_length - 1):
#                 context = self.attention(
#                     decoder_hidden[0].squeeze(0).unsqueeze(1), encoder_outputs)

#                 if t == 0:  # input <bos>
#                     it = torch.LongTensor([self.sos_id] * batch_size).cuda()
#                 elif sample_max:
#                     sampleLogprobs, it = torch.max(logprobs, 1)
#                     seq_logprobs.append(sampleLogprobs.view(-1, 1))
#                     it = it.view(-1).long()

#                 else:
#                     # sample according to distribuition
#                     if temperature == 1.0:
#                         prob_prev = torch.exp(logprobs)
#                     else:
#                         # scale logprobs by temperature
#                         prob_prev = torch.exp(torch.div(logprobs, temperature))
#                     it = torch.multinomial(prob_prev, 1).cuda()
#                     sampleLogprobs = logprobs.gather(1, it)
#                     seq_logprobs.append(sampleLogprobs.view(-1, 1))
#                     it = it.view(-1).long()

#                 seq_preds.append(it.view(-1, 1))

#                 xt = self.embedding(it)
#                 decoder_input = torch.cat([xt, context], dim=1)
#                 decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
#                 decoder_output, decoder_hidden = self.rnn(
#                     decoder_input, decoder_hidden)
#                 logprobs = F.log_softmax(
#                     self.out(decoder_output.squeeze(1)), dim=1)

#             seq_logprobs = torch.cat(seq_logprobs, 1)
#             seq_preds = torch.cat(seq_preds[1:], 1)

#         return seq_logprobs, seq_preds

#     def _init_weights(self):
#         """ init the weight of some layers
#         """
#         nn.init.xavier_normal_(self.out.weight)

#     def _init_rnn_state(self, encoder_hidden):
#         """ Initialize the encoder hidden state. """
#         if encoder_hidden is None:
#             return None
#         if isinstance(encoder_hidden, tuple):
#             encoder_hidden = tuple(
#                 [self._cat_directions(h) for h in encoder_hidden])
#         else:
#             encoder_hidden = self._cat_directions(encoder_hidden)
#         return encoder_hidden

#     def _cat_directions(self, h):
#         """ If the encoder is bidirectional, do the following transformation.
#             (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
#         """
#         if self.bidirectional_encoder:
#             h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
#         return h


# class StrategicARG(nn.Module):
#     """
#     Strategic Automated Rationale Generation (ARG) model, used to translate
#     a sequence of actions and image observations into a natural language description
#     that captures what the agent is doing.

#     Based on the work by Nguyen et. al. (2022)
#     https://ieeexplore.ieee.org/document/9918019

#     Used as a baseline for comparing the performance of our latent-state translation
#     """

#     def __init__(
#         self, rgb_encoder_args: Dict[str, Any], action_dim: int, reccurrent_dim: int
#     ):
#         super(StrategicARG, self).__init__()

#         self._encoder = ConvEncoder(
#             rgb_encoder_args["input_shape"],
#             rgb_encoder_args["depth"],
#             rgb_encoder_args["act"],
#             rgb_encoder_args["norm"],
#             rgb_encoder_args["kernel_size"],
#             rgb_encoder_args["minres"],
#         )

#         self._encoded_dim = self._encoder.outdim
#         self._action_dim = action_dim

#         self._lstm_encoder = EncoderRNN(
#             dim_vid=self._encoded_dim,
#             dim_hidden=reccurrent_dim,
#             n_layers=1,
#             rnn_cell="gru",
#             bidirectional=False,
#         )
#         self._lstm_decoder = DecoderRNN(
#             vocab_size=rgb_encoder_args["vocab_size"],
#             max_len=rgb_encoder_args["max_len"],
#             dim_hidden=reccurrent_dim,
#             dim_word=rgb_encoder_args["dim_word"],
#             n_layers=1,
#             rnn_cell="gru",
#             bidirectional=False,
#         )


#     def forward(
#         rgb_images: torch.Tensor,
#         actions: torch.Tensor,
#     )


class BaselineTranslator(nn.Module):
    """Contains a baseline translator that translates a sequence of
    reconstructed RGB observations and actions into a natural language
    description
    """

    def __init__(
        self,
        rgb_encoder_params: Dict[str, Any],
        transformer_params: Dict[str, Any],
        action_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Sets up the RGB CNN encoder, and the encoder-decoder transformer for translation

        Args:
            rgb_encoder_params (Dict[str, Any]): Dictionary of parameters for the CNN ENcoder.
            transformer_params (Dict[str, Any]): Dictionary of parameters for the TransformerEncoderDecoder
            action_dim (int): Dimension of the actions
        """
        super(BaselineTranslator, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._rgb_encoder = ConvEncoder(**rgb_encoder_params).to(self._device)
        self._action_dim = action_dim
        self._encoded_dim = self._rgb_encoder.outdim + self._action_dim

        transformer_params["bottleneck_input_size"] = self._encoded_dim

        self._transformer = TransformerEncoderDecoder(**transformer_params).to(
            self._device
        )

    def forward(
        self,
        rgb_images: torch.Tensor,
        actions: torch.Tensor,
        narrations: torch.Tensor,
        generate_mask: bool = True,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the baseline translator model. The RGB images are passed through the
        CNN encoder, and the actions are concatenated to the encoded RGB images. The resulting
        tensor is passed through the transformer encoder-decoder model to generate the
        translated sequence of tokens, along with the narrations using teacher-forcing.

        Args:
            rgb_images (torch.Tensor): RGB images of shape (batch_size, seq_length, height, width, channels)

            actions (torch.Tensor): action taken at each time step of shape (batch_size, seq_length, action_dim)

            narrations (torch.Tensor): ground truth narrations of shape (batch_size, seq_length)

            generate_mask (bool, optional): Whether to generate padding and attention masks for the
            target (narrations). Defaults to True.

            src_mask (Optional[torch.Tensor], optional): Optional mask for src padding where True indicates
            sequence elements to be ignored. Defaults to None.

        Returns:
            torch.Tensor: predictions of shape (seq_length, batch_size, vocab_size)
        """
        rgb_images = torch.Tensor(rgb_images) / 255.0
        encoded_images = self._rgb_encoder(rgb_images)

        # make final action at each timestep zero
        actions = torch.cat(
            [actions[:, :-1, :], torch.zeros_like(actions[:, -1:, :])], dim=1
        )

        input = torch.cat([encoded_images, actions], dim=-1)

        pred = self._transformer.forward(
            input,
            narrations,
            generate_mask=generate_mask,
            src_mask=src_mask,
        )

        return pred

    @torch.no_grad()
    def generate(
        self,
        rgb_images: torch.Tensor,
        actions: torch.Tensor,
        vocab: dict,
        max_sequence_length: int,
        sampling_method: str = "nucleus",
    ) -> str:
        """Generates a translation of a sequence of reconstructed images and actions.

        Args:
            rgb_images (torch.Tensor): reconstructed RGB images of shape
            (batch_size, seq_length, height, width, channels)

            actions (torch.Tensor): actions taken at each timestep of shape
            (batch_size, seq_length, action_dim)

            vocab (dict): dictionary mapping token IDs to words.

            max_sequence_length (int): maximum length of the generated sequence.

            sampling_method (str, optional): Sampling method to use. Can either be
            "nucleus" or "greedy". Defaults to "nucleus".

        Returns:
            str: the generated sequence of words.
        """
        self.eval()
        rgb_images = torch.Tensor(rgb_images) / 255.0
        encoded_images = self._rgb_encoder(rgb_images)

        # make final action at each timestep zero
        actions = torch.cat(
            [actions[:, :-1, :], torch.zeros_like(actions[:, -1:, :])], dim=1
        )

        input = torch.cat([encoded_images, actions], dim=-1)

        # Take first element as assume batch size is 1 for now.
        result = self._transformer.generate(
            input,
            vocab,
            max_sequence_length,
            sampling_method=sampling_method,
        )[0]
        assert isinstance(result, str), "Result is not a string"
        self.train()
        return result
