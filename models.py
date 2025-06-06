from typing import Tuple, List, Optional, Dict, Union
import copy
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import networks
import tools
from torcheval.metrics.text import Perplexity

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config, narrator=None):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        print(obs_space.spaces.items())
        shapes = {
            k: tuple(v.shape)
            for k, v in obs_space.spaces.items()
            if k != "privileged_obs"
        }
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.device = torch.device(config.device)
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        if config.enable_language:
            assert narrator is not None
            self.narrator = narrator
            self.vocab = tools.load_json_vocab(config.vocab_path)
            transformer_params = {
                "d_model": config.translator_head["token_embed_size"],
                "n_head": config.translator_head["attention_heads"],
                "num_encoder_layers": config.translator_head["encoder_blocks"],
                "num_decoder_layers": config.translator_head["decoder_blocks"],
                "dim_feedforward": config.translator_head["out_head_dim"],
                "encoder_bottleneck": config.translator_head["use_bottleneck"],
                "dropout": config.translator_head["dropout"],
                "activation": config.translator_head["activation"],
                "target_vocab_size": len(self.vocab),
                "bottleneck_input_size": feat_size,
            }
            if config.translation_baseline:
                cnn_encoder_params = {
                    "input_shape": (64, 64, 3),
                    "depth": config.encoder["cnn_depth"],
                    "act": config.encoder["act"],
                    "norm": config.encoder["norm"],
                    "kernel_size": config.encoder["kernel_size"],
                    "minres": config.encoder["minres"],
                }

                self.heads["language"] = networks.BaselineTranslator(
                    cnn_encoder_params,
                    transformer_params,
                    config.num_actions,
                    config.device,
                )

            else:
                self.heads["language"] = networks.TransformerEncoderDecoder(
                    **transformer_params
                )
            self._narration_max_enc_seq = config.enc_max_length
            self._narration_max_dec_seq = config.dec_max_length

        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(f"Latent state size: {feat_size}")
        # print(
        #     f"World model has {sum(param.numel() for param in self.parameters() if param.requires_grad)} variables."
        # )
        # From https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
        print(
            f"World model has {sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())}"
        )
        print(
            f"RSSM has {sum(param.numel() for param in self.dynamics.parameters())} variables."
        )
        if self._config.enable_language:
            print(
                f"Language Component has {sum(param.numel() for param in self.heads['language'].parameters())} variables."
            )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _reshape_to_narration_sequence(
        self, data_to_reshape: torch.Tensor, is_first: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshapes a set of input data of shape (batch_size, batch_length, ...) into a sequence that corresponds
        to the length of textual narrations of shape (batch_size, seq_length, ...). Where seq_length is the maximum
        number of timesteps that a narration sequence covers.

        Args:
            data_to_reshape (torch.Tensor): Shape (batch_size, batch_length, ...)
            is_first (torch.Tensor): Shape (batch_size, batch_length) indicating
            if the transition is the first in the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reshaped data, padding mask
        """

        data_sequences = []
        padding_masks = []
        batch_size, batch_length = data_to_reshape.shape[:2]
        for batch in range(batch_size):
            is_first_batch = is_first[batch].detach().cpu().numpy()
            assert is_first_batch[0] == 1
            is_first_indices = np.where(is_first_batch == 1)[0]
            current_index = 0
            current_is_first_index = 1
            while current_index < batch_length:
                if len(is_first_indices) > current_is_first_index:
                    end_index = min(
                        current_index + self._narration_max_enc_seq,
                        batch_length,
                        is_first_indices[current_is_first_index],
                    )

                    data_sequence = data_to_reshape[batch, current_index:end_index]
                    current_index = end_index
                    if end_index == is_first_indices[current_is_first_index]:
                        current_is_first_index += 1
                else:
                    end_index = min(
                        current_index + self._narration_max_enc_seq,
                        batch_length,
                    )
                    data_sequence = data_to_reshape[batch, current_index:end_index]
                    current_index = end_index

                if data_sequence.shape[0] < self._narration_max_enc_seq:
                    padding = torch.zeros(
                        self._narration_max_enc_seq - data_sequence.shape[0],
                        data_sequence.shape[1],
                    ).to(self.device)
                    padding_mask = torch.ones(self._narration_max_enc_seq).to(
                        self.device
                    )
                    padding_mask[: data_sequence.shape[0]] = 0
                    data_sequence = torch.cat([data_sequence, padding], dim=0)
                else:
                    padding_mask = torch.zeros(self._narration_max_enc_seq).to(
                        self.device
                    )
                data_sequences.append(data_sequence)
                padding_masks.append(padding_mask)

        # Shapes (batch, seq_len, dim)
        reshaped_data = torch.stack(data_sequences)
        padding_masks = torch.stack(padding_masks)

        return reshaped_data, padding_masks

    def _get_language_prediction(
        self,
        data: Dict[str, torch.Tensor],
        post: Dict[str, torch.Tensor],
        narration_data: Union[np.ndarray, Dict],
        language_grads: bool = True,
        baseline: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the language model predictions and ground truth narrations

        Args:
            data (dict[str, torch.Tensor]): replay buffer sample. Each array
            is of shape (batch_size, batch_length, ...)

            post (dict[str, torch.Tensor]): posterior sample from the world model.
            tensors are of shape (batch_size, batch_length, D).

            narration_data (np.ndarray): array containin data to be used
            as input to the narrator function to generate the ground-truth
            narrations. Array is of shape (batch_size, batch_length, ...)

            language_grads (bool, optional): Whether to enable gradients from translation
            loss to flow into the latent state representation learning. Defaults to True.

            baseline (bool, optional): Whether to use the RGB Reconstruction baseline as
            the language model. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: predicted narrations tokens,
            ground-truth narration tokens
        """

        narrations = tools.generate_batch_narrations(
            self.narrator,
            narration_data,
            self._narration_max_enc_seq,
            self._narration_max_dec_seq,
            self.vocab,
            self.device,
            data["is_first"],
            config=self._config,
        ).reshape(-1, self._narration_max_dec_seq)
        # Shape (batch, seq_len, latent_state_dim)
        feat = self.dynamics.get_feat(post)
        feat = feat if language_grads else feat.detach()

        if not baseline:
            latent_sequences, padding_masks = tools.batchify_translator_input(
                feat, data["is_first"], self._narration_max_enc_seq, self.device
            )  # type: ignore

            latent_sequences = (
                latent_sequences if language_grads else latent_sequences.detach()
            )
            pred = self.heads["language"].forward(
                latent_sequences,
                narrations[:, :-1],
                generate_mask=True,
                src_mask=padding_masks,
            )
        else:
            latent_sequences, padding_masks, actions = tools.batchify_translator_input(
                feat,
                data["is_first"],
                self._narration_max_enc_seq,
                self.device,
                data["action"],
            )  # type: ignore

            reconstructed_images = (
                self.heads["decoder"](latent_sequences)["image"].mode().detach().clone()
            )
            reconstructed_images = torch.clip(255 * reconstructed_images, 0, 255).to(
                torch.uint8
            )

            pred = self.heads["language"].forward(
                reconstructed_images,
                actions,
                narrations[:, :-1],
                generate_mask=True,
                src_mask=padding_masks,
            )

        return pred, narrations

    def _language_to_latent_state(
        self,
        posterior_logits: List[torch.Tensor],
        true_narrations: torch.Tensor,
        stoch_starting_token: int = 29,
        bos_token: int = 1,
        eos_token: int = 2,
        padding_token: int = 0,
        translation_token: int = 62,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the reverse translation: from language to a sequence of posterior distributions.

        Args:
            posterior_logits (torch.Tensor): Set of posterior logits of shape (batch, seq_len, n_categories, n_classes)
            true_narrations (torch.Tensor): Set of ground-truth narration tokens of shape (batch, seq_len)
            stoch_starting_token (int, optional): Starting token number for the stochastic state 'words'
            in the vocabulary. Defaults to 29.

        Returns:
            torch.Tensor: KL divergence loss between predicted and actual posterior distributions.
        """
        # batch, seq_len, n_categories, n_classes = posterior_logits.shape
        # posterior_logits = posterior_logits.reshape(
        #     (batch, seq_len * n_categories, n_classes)
        # )
        logit_tokens = []
        for logit in posterior_logits:
            posterior_tokens = torch.argmax(logit, dim=-1)
            posterior_tokens = posterior_tokens + stoch_starting_token
            posterior_tokens = posterior_tokens.flatten()
            bos_token = torch.ones(1).to(self._config.device) * bos_token  # type: ignore
            eos_token = torch.ones(1).to(self._config.device) * eos_token  # type: ignore
            posterior_tokens = torch.cat(
                [bos_token, posterior_tokens, eos_token], dim=0  # type: ignore
            )
            excpected_token_length = (
                self._narration_max_enc_seq * self._config.dyn_discrete
            ) + 2  # Adding two for BOS, EOS tokens

            if posterior_tokens.shape[0] < excpected_token_length:
                padding = (
                    torch.ones(
                        (excpected_token_length - posterior_tokens.shape[0],)
                    ).to(self.device)
                    * padding_token
                )
                posterior_tokens = torch.cat([posterior_tokens, padding], dim=0)

            logit_tokens.append(posterior_tokens)
        logit_tokens = torch.stack(logit_tokens).to(dtype=torch.long)

        if logit_tokens.shape[0] != true_narrations.shape[0]:
            raise ValueError(
                "The number of predicted posterior distributions and ground-truth narrations should be the same."
            )
        translation_token = (
            torch.ones_like(true_narrations[:, 0]).to(self._config.device)
            * translation_token
        ).unsqueeze(1)
        # Remove BOS and EOS tokens from the true narrations
        bos_mask = true_narrations == bos_token
        true_narrations = true_narrations[~bos_mask]
        eos_mask = true_narrations == eos_token
        true_narrations = true_narrations[~eos_mask]
        true_narrations = true_narrations.reshape((logit_tokens.shape[0], -1))
        true_narrations = torch.cat([translation_token, true_narrations], dim=1)
        pred_logits = self.heads["language"].forward(
            true_narrations,
            logit_tokens[:, :-1],
            generate_mask=True,
            embed_src=True,
            generate_src_mask=True,
        )

        return pred_logits, logit_tokens

    def _language_to_action_tokens(
        self,
        actions: torch.Tensor,
        true_narrations: torch.Tensor,
        is_first: torch.Tensor,
        starting_states: torch.Tensor,
        action_starting_token: int = 3,
        bos_token: int = 1,
        eos_token: int = 2,
        padding_token: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Translates from a sequence of English tokens describing an agent's
        behaviour into the sequence of action tokens that the agent took
        when realising this behaviour.

        Args:
            actions (torch.Tensor): batch of actions taken of shape
            (batch_size, batch_length, act_dim)
            true_narrations (torch.Tensor): _description_
            starting_states (torch.Tensor): Initial posterior states of shape
            (batch_size, latent_state_dim)
            action_starting_token (int, optional): _description_. Defaults to 29.
            bos_token (int, optional): _description_. Defaults to 1.
            eos_token (int, optional): _description_. Defaults to 2.
            padding_token (int, optional): _description_. Defaults to 0.
            translation_token (int, optional): _description_. Defaults to 33.

        Returns:
            torch.Tensor: shape (batch_size, seq_len) of predicted actions of shape
            (seq_len, batch_size, act_dim)
        """

        true_actions, action_padding_mask = self._reshape_to_narration_sequence(
            actions, is_first
        )

        # Tokenise actions. Actions are shape (batch, seq_len, act_dim)
        true_actions_tokens = true_actions.argmax(dim=-1).to(dtype=torch.long)
        true_actions_tokens = true_actions_tokens + action_starting_token
        bos_token = (
            torch.ones((true_actions_tokens.shape[0], 1)).to(
                self._config.device, dtype=torch.long
            )
            * bos_token
        )
        eos_token = (
            torch.ones((true_actions_tokens.shape[0], 1)).to(
                self._config.device, dtype=torch.long
            )
            * eos_token
        )
        true_actions_tokens = torch.cat(
            [bos_token, true_actions_tokens, eos_token], dim=1
        )

        # Embed the starting states using the translation components latent state embedder.
        starting_states_embed = self.heads["language"]._initial_embed(starting_states)

        predicted_action_token_logits = self.heads["language_to_action"].forward(
            true_narrations,
            true_actions_tokens[:, :-1],
            generate_mask=True,
            tokens_to_prepend=starting_states_embed,
        )

        # Shape (seq_len, batch)
        predicted_action_tokens = predicted_action_token_logits.argmax(dim=-1)

        N, B = predicted_action_tokens.shape
        # (seq_len, batch) -> (batch, seq_len)
        predicted_action_tokens = predicted_action_tokens.permute(1, 0)

        predicted_one_hot_actions = torch.zeros(
            (B, N, self._config.num_actions), dtype=torch.float32
        ).to(self._config.device)

        for batch in range(B):
            # print(f"Predicted action sequence: {predicted_action_tokens[batch]}")
            # print(f"True action sequence: {true_actions_tokens[batch]}")
            for seq in range(N):
                if predicted_action_tokens[batch, seq] in [0, 1, 2]:
                    continue
                predicted_one_hot_actions[
                    batch,
                    seq,
                    predicted_action_tokens[batch, seq] - action_starting_token,
                ] = 1
        return (
            predicted_one_hot_actions,
            predicted_action_token_logits,
            true_actions_tokens,
        )

    def _process_narration_data(self, data: Dict):
        """Processes the data stored in the replay buffer into the task-appropriate narration
        data format.

        Args:
            data (Dict): Replay buffer sample.
        """

        narration_keys = []
        narration_keys = self._config.narrator["narration_key"]

        if type(narration_keys) is list:
            narration_data = {k: deepcopy(data[k]) for k in narration_keys}
        else:
            narration_data = deepcopy(data[narration_keys])
        if type(narration_data) is dict:
            if len(narration_data.keys()) == 1:  # type: ignore
                narration_data = narration_data[list(narration_data.keys())[0]]  # type: ignore

        return narration_data

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)

        narration_keys = self._config.narrator["narration_key"]
        narration_data = self._process_narration_data(data)

        data = self.preprocess(
            data,
            keys_to_ignore=(
                [narration_keys] if type(narration_keys) is str else narration_keys
            ),
        )
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    if name == "language":
                        pred, narrations = self._get_language_prediction(
                            data,
                            post,
                            narration_data,
                            language_grads=self._config.language_grads,
                            baseline=self._config.translation_baseline,
                        )
                    else:
                        grad_head = name in self._config.grad_heads
                        # Shape (batch, seq_len, latent_state_dim)
                        feat = self.dynamics.get_feat(post)
                        feat = feat if grad_head else feat.detach()
                        pred = head(feat)

                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    if name == "language":
                        loss = tools.narration_loss(pred, narrations[:, 1:])
                        losses[name] = loss
                    else:
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                    if (key != "action_prediction" and key != "language")
                }
                model_loss = sum(scaled.values()) + kl_loss

            if self._config.enable_language:
                # mean of language loss is already taken
                metrics = self._model_opt(
                    torch.mean(model_loss) + losses["language"],
                    self.parameters(),
                )

            else:
                metrics = self._model_opt(
                    torch.mean(model_loss),
                    self.parameters(),
                )

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs, keys_to_ignore: Optional[List[str]] = None):
        ignore = keys_to_ignore or []
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        try:
            # torch.Tensor is an alias of torch.FloatTensor
            obs = {
                k: torch.Tensor(v).to(self._config.device)
                for k, v in obs.items()
                if k not in ignore
            }
        except TypeError:
            for k, v in obs.items():
                print(f"Key: {k}, Value: {v}")
            raise TypeError
        return obs

    def video_pred(self, data, ignore_keys: Optional[List[str]] = None):
        data = self.preprocess(data, ignore_keys)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)

    def intent_prediction(self, data, ignore_keys: Optional[List[str]] = None):
        if type(self._config.narrator["narration_key"]) is list:
            narration_data = {
                k: data[k] for k in self._config.narrator["narration_key"]
            }
        else:
            narration_data = data[self._config.narrator["narration_key"]]
        data = self.preprocess(data, ignore_keys)
        embed = self.encoder(data)
        embed = embed[0, :16].unsqueeze(0)
        if type(self._config.narrator["narration_key"]) is list:
            narration_data_list = {k: v[0, :16] for k, v in narration_data.items()}
            if type(narration_data) is dict:
                if len(narration_data.keys()) == 1:  # type: ignore
                    narration_data_list = narration_data_list[list(narration_data.keys())[0]]  # type: ignore
        else:
            narration_data_list = [narration_data[0][i] for i in range(16)]  # type: ignore
        ground_truth_intent = self.narrator.narrate(narration_data_list)
        states, _ = self.dynamics.observe(
            embed,
            data["action"][0, :16].unsqueeze(0),
            data["is_first"][0, :16].unsqueeze(0),
        )
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(
            data["action"][0, :16].unsqueeze(0), init
        )
        imagined_feat = self.dynamics.get_feat(prior)
        intent = self.heads["language"].generate(
            imagined_feat, self.vocab, self._narration_max_dec_seq
        )[0]
        print(f"Imagined Intent: {intent}")
        print(f"Ground Truth Intent: {ground_truth_intent}")
        return intent, ground_truth_intent


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
