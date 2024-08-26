import os
import cv2
import torch
import pathlib
import argparse
import ruamel.yaml as yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import tools
import envs.wrappers as wrappers
import envs.minigrid as minigrid


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


def load_env(task: str = "four_squares", max_steps: int = 100):
    env = minigrid.MiniGrid(task)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, max_steps)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
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


def evaluate_narration_model(env, agent, actions=None):
    img_folder = "/home/mattbarker/dev/somniloquy-dreamer-v3/evaluation_imgs"
    obs = env.reset()
    while env.mission != "navigate to the blue square":
        obs = env.reset()
    t = obs.copy()
    t = add_batch_to_obs(t)
    # t = {k: tools.convert(v) for k, v in t.items()}
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
        os.path.join(img_folder, "reconstructed_img_t0.png"),
        cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR),
    )
    # save true image
    true_img = obs["image"]
    cv2.imwrite(
        os.path.join(img_folder, "true_img_t0.png"),
        cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR),
    )

    encoded_obs: list = []
    latent_states: list = []
    imagined_states: list = []
    with torch.no_grad():
        for t in range(16):
            action_arr = np.zeros(3)
            # Get action from keyboard
            if actions is not None:
                action = actions[t]
                action_arr[action] = 1
            else:
                action = input("Enter action: ")
                action_arr[int(action)] = 1
            action_dict = {"action": action_arr}
            obs, reward, done, info = env.step(action_dict)
            encoded_obs.append(info["encoded_image"])
            current_obs = obs.copy()
            current_obs = add_batch_to_obs(current_obs)
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
            imagined_states.append(imagined_state)
            latent_states.append(latent_state)
            reconstructed_obs = agent._wm.heads["decoder"](latent_state)["image"].mode()
            reconstructed_img = reconstructed_obs[0, 0].detach().cpu().numpy()
            reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype(
                np.uint8
            )

            # # save image
            # cv2.imwrite(
            #     os.path.join(img_folder, f"reconstructed_img_t{t+1}.png"),
            #     cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR),
            # )
            # # save true image
            # true_img = obs["image"]
            # cv2.imwrite(
            #     os.path.join(img_folder, f"true_img_t{t+1}.png"),
            #     cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR),
            # )

            if done:
                break
    with torch.no_grad():
        latent_states = torch.cat(latent_states, dim=0).permute(1, 0, 2)
        imagined_states = torch.cat(imagined_states, dim=0).permute(1, 0, 2)
        # T, B, D
        narration = agent._wm.heads["language"].generate(
            latent_states, agent._wm.vocab, 50
        )
        intent = agent._wm.heads["language"].generate(
            imagined_states, agent._wm.vocab, 50
        )
        actual_narration = agent._wm.narrator.narrate(encoded_obs)
    print(f"Non-imagined Narration: {narration}")
    print(f"Ground Truth Narration: {actual_narration}")
    print(f"Imagined Narration: {intent}")

    X = latent_states.detach().cpu().numpy().squeeze()
    return X
    # print(X.shape)
    # pca = PCA(n_components=2, svd_solver="full")
    # latent_state_compressed = pca.fit_transform(X)

    # # Plot latent_state_compressed with PCA

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.scatter(latent_state_compressed[:, 0], latent_state_compressed[:, 1])
    # for i in range(latent_state_compressed.shape[0] - 1):
    #     ax.arrow(
    #         latent_state_compressed[i, 0],
    #         latent_state_compressed[i, 1],
    #         latent_state_compressed[i + 1, 0] - latent_state_compressed[i, 0],
    #         latent_state_compressed[i + 1, 1] - latent_state_compressed[i, 1],
    #         head_width=0.2,
    #         head_length=0.3,
    #         fc="red",
    #         ec="red",
    #     )
    # # save to file
    # plt.savefig(os.path.join(img_folder, "latent_state_compressed.png"))


def plot_trajectory_graph(agent, step):
    env = load_env()
    actions = [
        [1] * 16,
        [2] * 16,
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 1, 2, 2, 2],
    ]
    latent_states = None
    latent_state_list = []
    for i in range(6):
        latent_state = evaluate_narration_model(env, agent, actions[i])
        if latent_states is None:
            latent_states = latent_state
        else:
            latent_states = np.concatenate((latent_states, latent_state), axis=0)
        latent_state_list.append(latent_state)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(latent_states)

    # Plot latent_state_compressed with PCA

    fig, ax = plt.subplots()
    matplotlib_colours = {
        "black": "k",
        "yellow": "y",
        "red": "r",
        "green": "g",
        "blue": "b",
        "magenta": "m",
    }
    legends = [
        "Didn't Move",
        "Straight Line",
        "Green Square",
        "Grey Square",
        "Purple Square",
        "Blue square (goal)",
    ]
    for i in range(6):
        colour = list(matplotlib_colours.keys())[i]
        latent_state_compressed = pca.transform(latent_state_list[i])
        ax.scatter(
            latent_state_compressed[:, 0],
            latent_state_compressed[:, 1],
            c=matplotlib_colours[colour],
            label=legends[i],
        )
        for i in range(latent_state_compressed.shape[0] - 1):
            ax.arrow(
                latent_state_compressed[i, 0],
                latent_state_compressed[i, 1],
                latent_state_compressed[i + 1, 0] - latent_state_compressed[i, 0],
                latent_state_compressed[i + 1, 1] - latent_state_compressed[i, 1],
                head_width=0.2,
                head_length=0.3,
                fc=matplotlib_colours[colour],
            )

    # Add Title
    ax.set_title("Latent State Compressed with PCA")
    # Label Axes
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend()

    # save to file
    plt.savefig(
        os.path.join(
            "/home/mattbarker/dev/somniloquy-dreamer-v3/evaluation_imgs_no_grad",
            f"latent_state_compressed_step_{step}.png",
        )
    )
    plt.close()


if __name__ == "__main__":
    model_path = "/home/mattbarker/Desktop/latest.pt"
    config_path = "/home/mattbarker/dev/somniloquy-dreamer-v3/configs.yaml"
    logdir = "/home/mattbarker/dev/somniloquy-dreamer-v3/logdir/minigrid"
    step = 0
    config = load_config(config_path)
    env = load_env()
    logger = tools.Logger(logdir, config.action_repeat * step)
    agent = initialize_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config,
        logger=logger,
        train_dataset=None,
    )
    load_weights(model_path, agent)

    actions = [
        [1] * 16,
        [2] * 16,
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 1, 2, 2, 2],
    ]
    latent_states = None
    latent_state_list = []
    for i in range(6):
        latent_state = evaluate_narration_model(env, agent, actions[i])
        if latent_states is None:
            latent_states = latent_state
        else:
            latent_states = np.concatenate((latent_states, latent_state), axis=0)
        latent_state_list.append(latent_state)
    print(latent_states.shape)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(latent_states)

    # Plot latent_state_compressed with PCA

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    matplotlib_colours = {
        "black": "k",
        "yellow": "y",
        "red": "r",
        "green": "g",
        "blue": "b",
        "magenta": "m",
    }
    legends = [
        "Didn't Move",
        "Straight Line",
        "Green Square",
        "Grey Square",
        "Purple Square",
        "Blue square (goal)",
    ]
    for i in range(6):
        colour = list(matplotlib_colours.keys())[i]
        latent_state_compressed = pca.transform(latent_state_list[i])
        ax.scatter(
            latent_state_compressed[:, 0],
            latent_state_compressed[:, 1],
            c=matplotlib_colours[colour],
            label=legends[i],
        )
        for i in range(latent_state_compressed.shape[0] - 1):
            ax.arrow(
                latent_state_compressed[i, 0],
                latent_state_compressed[i, 1],
                latent_state_compressed[i + 1, 0] - latent_state_compressed[i, 0],
                latent_state_compressed[i + 1, 1] - latent_state_compressed[i, 1],
                head_width=0.2,
                head_length=0.3,
                fc=matplotlib_colours[colour],
            )

    # Add Title
    ax.set_title("Latent State Compressed with PCA")
    # Label Axes
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend()

    # save to file
    plt.savefig(os.path.join(".", "latent_state_compressed.png"))
