import os
import cv2
import torch
import pathlib
import argparse
import ruamel.yaml as yaml
import numpy as np

import tools
import envs.wrappers as wrappers

from dreamer import make_env


def load_weights(model_path: str, agent):

    checkpoint = torch.load(model_path)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent._should_pretrain._once = False


def initialize_agent(observation_space, action_space, config, logger, train_dataset):
    config.num_actions = (
        action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    )
    from dreamer import Dreamer

    print(f"Number of actions: {config.num_actions}")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    return agent


def load_config(yaml_path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((pathlib.Path(yaml_path).read_text()))

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", "minigrid"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    return parser.parse_args(remaining)


def add_batch_to_obs(obs):
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs[k] = np.expand_dims(v, axis=0)
        else:
            obs[k] = np.array([v])
        # try:
        #     print(f"Key: {k}, Value: {obs[k].shape}")
        # except Exception as e:
        #     print(f"Key: {k}, Value: {obs[k]}")
        #     raise e
    return obs


def evaluate_world_model(
    env, agent, results_folder: str, trajectory_length: int = 16, actions=None
):
    obs = env.reset()
    t = obs.copy()
    t = add_batch_to_obs(t)
    t = {k: tools.convert(v) for k, v in t.items()}
    t = agent._wm.preprocess(t)
    embed = agent._wm.encoder(t)
    post, prior = agent._wm.dynamics.obs_step(
        embed=embed,
        is_first=t["is_first"],
        prev_state=None,
        prev_action=None,
    )
    prior = post
    latent_state = agent._wm.dynamics.get_feat(post).unsqueeze(0)
    reconstructed_obs = agent._wm.heads["decoder"](latent_state)["image"].mode()
    reconstructed_img = reconstructed_obs[0, 0].detach().cpu().numpy()
    reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype(np.uint8)

    # save image
    cv2.imwrite(
        os.path.join(results_folder, "reconstructed_img_t0.png"),
        cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR),
    )
    # save true image
    true_img = obs["image"]
    cv2.imwrite(
        os.path.join(results_folder, "true_img_t0.png"),
        cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR),
    )
    done = False
    while not done:
        with torch.no_grad():
            post_states = []
            prior_states = []
            for t in range(trajectory_length):
                action_arr = np.zeros(3)

                if actions == "keyboard":
                    action = input("Enter action: ")
                    action_arr[int(action)] = 1
                elif actions == "random":
                    action = np.random.randint(env.action_space.n)
                    action_arr[action] = 1
                elif isinstance(actions, list):
                    action = actions[t]
                    action_arr[action] = 1
                else:
                    # Use Actor
                    actor = agent._task_behavior.actor(latent_state)
                    action = actor.mode().argmax().item()
                    action_arr[action] = 1

                action_dict = {"action": action_arr}
                obs, reward, done, info = env.step(action_dict)
                current_obs = obs.copy()
                current_obs = add_batch_to_obs(current_obs)
                current_obs = {k: tools.convert(v) for k, v in current_obs.items()}
                current_obs = agent._wm.preprocess(current_obs)
                embed = agent._wm.encoder(current_obs)
                post, _ = agent._wm.dynamics.obs_step(
                    embed=embed,
                    is_first=current_obs["is_first"],
                    prev_state=post,
                    prev_action=torch.tensor(action_arr)
                    .unsqueeze(0)
                    .to(agent._config.device)
                    .float(),
                )
                prior = agent._wm.dynamics.img_step(
                    prev_state=prior,
                    prev_action=torch.tensor(action_arr)
                    .unsqueeze(0)
                    .to(agent._config.device)
                    .float(),
                )
                latent_state = agent._wm.dynamics.get_feat(post).unsqueeze(0)
                imagined_state = agent._wm.dynamics.get_feat(prior).unsqueeze(0)
                post_states.append(latent_state)
                prior_states.append(imagined_state)
                reconstructed_obs = agent._wm.heads["decoder"](latent_state)[
                    "image"
                ].mode()
                reconstructed_img = reconstructed_obs[0, 0].detach().cpu().numpy()
                reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype(
                    np.uint8
                )
                imagined_img = agent._wm.heads["decoder"](imagined_state)[
                    "image"
                ].mode()
                imagined_img = imagined_img[0, 0].detach().cpu().numpy()
                imagined_img = np.clip(255 * imagined_img, 0, 255).astype(np.uint8)

                # save image
                cv2.imwrite(
                    os.path.join(results_folder, f"reconstructed_img_t{t+1}.png"),
                    cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR),
                )

                # Save the imagined image
                cv2.imwrite(
                    os.path.join(results_folder, f"imagined_img_t{t+1}.png"),
                    cv2.cvtColor(imagined_img, cv2.COLOR_RGB2BGR),
                )

                # save true image
                true_img = obs["image"]
                cv2.imwrite(
                    os.path.join(results_folder, f"true_img_t{t+1}.png"),
                    cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR),
                )
        latent_states = torch.cat(post_states, dim=0).permute(1, 0, 2)
        imagined_states = torch.cat(prior_states, dim=0).permute(1, 0, 2)
        # T, B, D
        narration = agent._wm.heads["language"].generate(
            latent_states, agent._wm.vocab, 50
        )
        intent = agent._wm.heads["language"].generate(
            imagined_states, agent._wm.vocab, 50
        )
        print(f"POSTERIOR NARRATION: {narration}")
        print(f"PRIOR NARRATION: {intent}")
        input("Press Enter to continue...")


if __name__ == "__main__":
    model_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/logdir/minigrid-stochastic/latest.pt"
    config_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/configs.yaml"
    logdir = "/home/mattbarker/dev/somniloquy-dreamer-v3/logdir/minigrid-stochastic"
    results_folder = "/home/mattbarker/dev/somniloquy-dreamer-v3/world_model_evaluation/minigrid-stochastic"
    step = 0
    config = load_config(config_path)
    # config.enable_language = True
    # config.vocab_path = (
    #     "/home/mattbarker/dev/somniloquy-dreamer-v3/vocab/mindojo_harvest_1_dirt.json"
    # )
    # config.dec_max_length = 50
    # config.enc_max_length = 16
    env = make_env(config, "train", 1)
    logger = tools.Logger(logdir, config.action_repeat * step)
    agent = initialize_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config,
        logger=logger,
        train_dataset=None,
    )
    load_weights(model_path, agent)

    evaluate_world_model(
        env, agent, results_folder, trajectory_length=16, actions="keyboard"
    )
