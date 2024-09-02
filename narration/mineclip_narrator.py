from typing import List, Optional

import torch
import numpy as np
import cv2
from mineclip import MineCLIP  # type: ignore


class MineCLIPNarrator:
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        prompts: List[str],
        mineclip_resolution: Optional[List[int]],
    ):
        """Configures the prompts and MineCLIP model.

        Args:
            ckpt_path (str): Path to the MineCLIP pre-trained model to use.
            device (torch.device): Device to run the model on.
            prompts (List[str]): List of string prompts to use for narration.
            mineclip_resolution (List[int]): Resolution of the images given as
            input to the mineclip model. Defaults to [160, 256].
        """
        self.device = device
        if mineclip_resolution is None:
            mineclip_resolution = [160, 256]

        self._mineclip_resolution = mineclip_resolution

        config = {
            "arch": "vit_base_p16_fz.v2.t2",
            "hidden_dim": 512,
            "image_feature_dim": 512,
            "mlp_adapter_spec": "v0-2.t0",
            "pool_type": "attn.d2.nh8.glusw",
            "resolution": self._mineclip_resolution,
        }
        self.model = self._load_model(config, ckpt_path)
        self.prompts = prompts

        # Since prompts are all the same, we want to cache the embeddings
        # to save computation time.
        self.text_embeddings = self.model.encode_text(self.prompts)

    def _load_model(self, config: dict, ckpt_path: str) -> MineCLIP:
        model = MineCLIP(**config).to(self.device)
        model.load_ckpt(ckpt_path, strict=True)
        return model

    def narrate(self, rgb_obs: np.ndarray) -> str:
        """Generates the most likely caption (narration) for the given sequence of
        images (video).

        Args:
            rgb_obs (np.ndarray): Sequence of RGB images to generate narration for.
            shape (T, H, W, C)

        """
        rgb_obs = np.array(
            [cv2.resize(frame, tuple(self._mineclip_resolution)) for frame in rgb_obs]
        )
        # Unsqueeze to add batch dimension.
        video_history = (
            torch.from_numpy(rgb_obs)
            .unsqueeze(0)
            .to(self.device)
            .permute(0, 1, 4, 2, 3)
        )
        image_features = self.model.forward_image_features(video_history)
        video_features = self.model.forward_video_features(image_features)

        _, logits_per_text = self.model.forward_reward_head(
            video_features, self.text_embeddings
        )
        caption_probs = torch.nn.functional.softmax(logits_per_text, dim=0)
        return self.prompts[caption_probs.argmax().item()]  # type: ignore

    def batch_narration(self, rgb_obs: np.ndarray) -> List[str]:
        """Generates the most likely captions (narrations) for a given batch
        of videos..

        Args:
            rgb_obs (np.ndarray): Batch of sequences of RGB images to generate narration for.
            shape (B, T, H, W, C)

        """
        rgb_obs = np.array(
            [
                [cv2.resize(frame, tuple(self._mineclip_resolution)) for frame in video]
                for video in rgb_obs
            ]
        )
        video_history = torch.from_numpy(rgb_obs).to(self.device).permute(0, 1, 4, 2, 3)
        image_features = self.model.forward_image_features(video_history)
        video_features = self.model.forward_video_features(image_features)

        _, logits_per_text = self.model.forward_reward_head(
            video_features, self.text_embeddings
        )
        caption_probs = torch.nn.functional.softmax(logits_per_text, dim=0)
        return [self.prompts[i] for i in caption_probs.argmax(dim=0).tolist()]
