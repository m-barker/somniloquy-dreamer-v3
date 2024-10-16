from typing import Tuple, List, Optional, Dict, Union
import copy
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import networks
import tools

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
            self.heads["language"] = networks.TransformerEncoderDecoder(
                d_model=feat_size,
                target_vocab_size=len(self.vocab),
                max_seq_length=config.dec_max_length,
            )
            self._narration_max_enc_seq = config.enc_max_length
            self._narration_max_dec_seq = config.dec_max_length

        if config.enable_language_to_action:
            self.heads["language_to_action"] = networks.TransformerEncoderDecoder(
                d_model=512,
                target_vocab_size=config.num_actions + 3,  # +3 for BOS, EOS, PAD tokens
                max_seq_length=config.enc_max_length,
                embedding_layer=False,
                src_token_embedding=True,
                src_vocab_size=len(self.vocab),
            )

        if config.action_prediction:
            stochastic_size = config.dyn_stoch * config.dyn_discrete
            self.heads["action_prediction"] = networks.MLP(
                stochastic_size * 2,  # prev and next stochastic state as input
                config.num_actions,
                config.action_prediction_head["layers"],
                config.units,
                config.act,
                config.norm,
                dist=config.action_prediction_head["dist"],
                outscale=config.action_prediction_head["outscale"],
                device=config.device,
                name="action_prediction",
            )

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
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
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
            is_first_batch = is_first[batch].cpu().numpy()
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
                    padding_mask[data_sequence.shape[0] :] = 0
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
        data: Dict[str, np.ndarray],
        post: Dict[str, torch.Tensor],
        narration_data: Union[np.ndarray, Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the language model predictions and ground truth narrations

        Args:
            data (dict[str, np.ndarray]): replay buffer sample. Each array
            is of shape (batch_size, batch_length, ...)

            post (dict[str, torch.Tensor]): posterior sample from the world model.

            narration_data (np.ndarray): array containin data to be used
            as input to the narrator function to generate the ground-truth
            narrations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: predicted narrations tokens,
            ground-truth narration tokens, stochastic logit sequences, padding masks
        """

        narrations = tools.generate_batch_narrations(
            self.narrator,
            narration_data,
            self._narration_max_enc_seq,
            self._narration_max_dec_seq,
            self.vocab,
            self.device,
            data["is_first"],
        ).reshape(-1, self._narration_max_dec_seq)
        # Shape (batch, seq_len, latent_state_dim)
        feat = self.dynamics.get_feat(post)
        # Shape (batch, seq_len, n_categories, n_classes)
        stoch_logits = post["logit"]
        logit_sequences = []
        latent_sequences = []
        padding_masks = []
        starting_states: List[torch.Tensor] = []
        for batch in range(feat.shape[0]):
            is_first_batch = data["is_first"][batch].cpu().numpy()
            assert is_first_batch[0] == 1
            is_first_indices = np.where(is_first_batch == 1)[0]
            current_index = 0
            current_is_first_index = 1
            while current_index < feat.shape[1]:
                if len(is_first_indices) > current_is_first_index:
                    end_index = min(
                        current_index + self._narration_max_enc_seq,
                        feat.shape[1],
                        is_first_indices[current_is_first_index],
                    )

                    latent_sequence = feat[batch, current_index:end_index]
                    starting_states.append(feat[batch, current_index])
                    logit_sequence = stoch_logits[batch, current_index:end_index]
                    current_index = end_index
                    if end_index == is_first_indices[current_is_first_index]:
                        current_is_first_index += 1
                else:
                    end_index = min(
                        current_index + self._narration_max_enc_seq,
                        feat.shape[1],
                    )
                    latent_sequence = feat[batch, current_index:end_index]
                    starting_states.append(feat[batch, current_index])
                    logit_sequence = stoch_logits[batch, current_index:end_index]
                    current_index = end_index

                if latent_sequence.shape[0] < self._narration_max_enc_seq:
                    padding = torch.zeros(
                        self._narration_max_enc_seq - latent_sequence.shape[0],
                        latent_sequence.shape[1],
                    ).to(self.device)
                    padding_mask = torch.ones(self._narration_max_enc_seq).to(
                        self.device
                    )
                    padding_mask[latent_sequence.shape[0] :] = 0
                    latent_sequence = torch.cat([latent_sequence, padding], dim=0)
                else:
                    padding_mask = torch.zeros(self._narration_max_enc_seq).to(
                        self.device
                    )
                latent_sequences.append(latent_sequence)
                logit_sequences.append(logit_sequence)
                padding_masks.append(padding_mask)

        starting_states = torch.stack(starting_states)
        # Shapes (batch, seq_len, dim)
        feat = torch.stack(latent_sequences)
        padding_masks = torch.stack(padding_masks)
        pred = self.heads["language"].forward(
            feat,
            narrations[:, :-1],
            generate_mask=True,
            src_mask=padding_masks,
        )

        return pred, narrations, logit_sequences, padding_masks, starting_states  # type: ignore

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
            bos_token = torch.ones(1).to(self._config.device) * bos_token
            eos_token = torch.ones(1).to(self._config.device) * eos_token
            posterior_tokens = torch.cat(
                [bos_token, posterior_tokens, eos_token], dim=0
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
    ) -> torch.Tensor:
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

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        if self._config.enable_language:
            narration_keys = self._config.narrator["narration_key"]
            if type(narration_keys) is list:
                narration_data = {k: deepcopy(data[k]) for k in narration_keys}
            else:
                narration_data = deepcopy(data[narration_keys])
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
                        (
                            pred,
                            narrations,
                            stoch_logits,
                            padding_masks,
                            starting_states,
                        ) = self._get_language_prediction(data, post, narration_data)

                        if self._config.enable_language_to_latent:
                            pred_tokens, true_tokens = self._language_to_latent_state(
                                stoch_logits, narrations
                            )
                            if type(pred_tokens) is dict:
                                preds.update(pred_tokens)
                            else:
                                preds["language-to-latent"] = pred_tokens

                        if self._config.enable_language_to_action:
                            (
                                predicted_actions,
                                predicted_action_logits,
                                true_action_tokens,
                            ) = self._language_to_action_tokens(
                                data["action"],
                                narrations,
                                data["is_first"],
                                starting_states,
                            )
                            preds["language_to_action"] = predicted_action_logits
                            # # Shape (batch, seq_len, act_dim)
                            # starting_state_dict = self.dynamics.get_state_dict(
                            #     starting_states
                            # )
                            # assert torch.equal(
                            #     self.dynamics.get_feat(starting_state_dict),
                            #     starting_states,
                            # )
                            # imagined_states = self.dynamics.imagine_with_action(
                            #     predicted_actions, starting_state_dict
                            # )
                            # imagined_states = self.dynamics.get_feat(imagined_states)
                            # generated_narrations, generated_logits = self.heads[
                            #     "language"
                            # ].generate(
                            #     imagined_states,
                            #     self.vocab,
                            #     self._narration_max_dec_seq - 1,
                            #     return_logits=True,
                            # )
                            # preds["language_to_action"] = generated_logits

                    elif name == "action_prediction":
                        feat = post["stoch"].reshape(embed.shape[:2] + (-1,))
                        prev_states = feat[:, :-1]
                        next_states = feat[:, 1:]
                        action_pred_input = torch.cat(
                            [prev_states, next_states], dim=-1
                        )
                        pred = head(action_pred_input)

                    elif name == "language_to_action":
                        continue

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
                        # print(f"Language loss: {loss}")
                    elif name == "language_to_action":
                        loss = tools.narration_loss(pred, true_action_tokens[:, 1:])
                        losses[name] = loss
                        print(f"Language to action loss: {loss}")
                    elif name == "language-to-latent":
                        loss = tools.narration_loss(
                            pred, true_tokens[:, 1:], debug=True
                        )
                        losses[name] = loss
                        print(f"Language to latent loss: {loss}")
                        print("----------------------------------------------")
                    elif name == "action_prediction":
                        loss = -pred.log_prob(data["action"][:, 1:])
                        losses[name] = loss
                    elif name == "language-to-action-to-language":
                        loss = tools.narration_loss(pred, narrations[:, 1:])
                        losses[name] = loss
                        print(f"Language to action to language loss: {loss}")
                        print("----------------------------------------------")
                    else:
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                    if key != "action_prediction"
                }
                model_loss = sum(scaled.values()) + kl_loss
            if self._config.action_prediction:
                metrics = self._model_opt(
                    torch.mean(model_loss) + torch.mean(losses["action_prediction"]),
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
        obs = {
            k: torch.Tensor(v).to(self._config.device)
            for k, v in obs.items()
            if k not in ignore
        }
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
        else:
            narration_data_list = [narration_data[0][i] for i in range(16)]
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
