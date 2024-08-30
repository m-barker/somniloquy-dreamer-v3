from typing import List

import torch
import numpy as np
import cv2
from mineclip import MineCLIP  # type: ignore


class MineCLIPNarrator:
    def __init__(self, ckpt_path: str, device: torch.device, prompts: List[str]):
        self.device = device
        config = {
            "arch": "vit_base_p16_fz.v2.t2",
            "hidden_dim": 512,
            "image_feature_dim": 512,
            "mlp_adapter_spec": "v0-2.t0",
            "pool_type": "attn.d2.nh8.glusw",
            "resolution": [160, 256],
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
        rgb_obs = np.array([cv2.resize(frame, (160, 256)) for frame in rgb_obs])
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
        print(self.prompts[caption_probs.argmax().item()])
        return self.prompts[caption_probs.argmax().item()]  # type: ignore
