import os
import cv2
import torch
import pathlib
import argparse
import ruamel.yaml as yaml
import numpy as np

import tools
import envs.wrappers as wrappers


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


def load_env(suite: str, task: str, max_steps: int = 100, **kwargs):
    print(f"Creating {suite} environment for {task}...")
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(task, **kwargs)
        env = wrappers.OneHotAction(env)

    elif suite == "minigrid":
        import envs.minigrid as minigrid

        env = minigrid.MiniGrid(task, **kwargs)
        env = wrappers.OneHotAction(env)

    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(task, **kwargs)
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, **kwargs)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, **kwargs)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "minedojo":
        import envs.minedojo_env as minedojo_env

        env = minedojo_env.MineDojoEnv()
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, max_steps)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


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

    with torch.no_grad():
        for t in range(trajectory_length):
            action_arr = np.zeros(env.action_space.n)

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
            reconstructed_obs = agent._wm.heads["decoder"](latent_state)["image"].mode()
            reconstructed_img = reconstructed_obs[0, 0].detach().cpu().numpy()
            reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype(
                np.uint8
            )
            imagined_img = agent._wm.heads["decoder"](imagined_state)["image"].mode()
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


if __name__ == "__main__":
    import sys

    model_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/logdir/minedojo/latest.pt"
    config_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/configs.yaml"
    logdir = "/home/mattbarker/dev/somniloquy-dreamer-v3/logdir/minedojo"
    suite = "minedojo"
    task = "harvest_1_dirt"
    results_folder = (
        "/home/mattbarker/dev/somniloquy-dreamer-v3/world_model_evaluation/minedojo"
    )
    step = 0
    config = load_config(config_path)
    print(config)
    env = load_env(suite=suite, task=task, max_steps=2000)
    logger = tools.Logger(logdir, config.action_repeat * step)
    agent = initialize_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config,
        logger=logger,
        train_dataset=None,
    )
    load_weights(model_path, agent)

    evaluate_world_model(env, agent, results_folder, trajectory_length=100)
