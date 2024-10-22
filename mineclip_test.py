import torch
from mineclip import MineCLIP
import numpy as np
import minedojo
import cv2
from envs.minedojo_env import MineDojoEnv
import envs.wrappers as wrappers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config: dict):
    model = MineCLIP(**config).to(DEVICE)
    model.load_ckpt("/home/mattbarker/dev/somniloquy-dreamer-v3/attn.pth", strict=True)
    return model


if __name__ == "__main__":
    config = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    model = load_model(config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    prompts = [
        "Mining diamonds",
        "Climbing a hill",
        "I am holding shears",
        "Trying to find a sheep",
        "Breaking a dirt block",
        "Breaking a sand block",
        "Breaking a gravel block",
        "Breaking a tree",
        "Breaking a leaf block",
        "Breaking a stone block",
        "Breaking a cobblestone block",
        "Doing nothing",
        "looking up",
        "looking at a tree",
        "looking down",
        "Looking at a dirt block",
        "looking at a sand block",
        "looking at a gravel block",
        "looking at water",
        "swimming in water",
        "breaking grass",
        "picking up seeds",
        "Moving forwards",
        "Moving backwards",
        "Walking towards a tree",
        "Walking towards a mountain",
        "Walking towards a cave",
        "Walking towards a river",
    ]
    env = MineDojoEnv(
        task_id="harvest_wool",
        image_size=(160, 256),
    )
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    obs, info = env.reset()
    done = False
    while not done:
        rgb_history = []
        for i in range(16):
            action = np.random.randint(0, 61)
            action_arr = np.zeros(61)
            action_arr[action] = 1
            action_dict = {"action": action_arr}
            obs, _, done, _ = env.step(action_dict)
            rgb_history.append(obs["image"])
            # image = cv2.cvtColor(
            #     np.transpose(obs["image"], (1, 2, 0)), cv2.COLOR_RGB2BGR
            # )
            # # save image for visualization
            # cv2.imwrite(
            #     f"frame_{i}.png",
            #     image,
            # )
            if done:
                break
        video_history = torch.tensor(np.array(rgb_history), device=DEVICE).unsqueeze(0)
        video = video_history.permute(0, 1, 4, 2, 3)
        # print(f"VIDEO SHAPE: {video.shape}")
        VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

        image_feats = model.forward_image_features(video)
        video_feats = model.forward_video_features(image_feats)
        assert video_feats.shape == (VIDEO_BATCH, 512)

        text_feats_batch = model.encode_text(prompts)
        assert text_feats_batch.shape == (TEXT_BATCH, 512)

        logits_per_video, logits_per_text = model.forward_reward_head(
            video_feats, text_tokens=text_feats_batch
        )
        probs = torch.nn.functional.softmax(logits_per_text, dim=0)

        print(f"MOST LIKELY CAPTION: {prompts[probs.argmax().item()]}")
