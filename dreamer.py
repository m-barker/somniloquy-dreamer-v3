import argparse
import functools
import os
import pathlib
import sys
from typing import List, Union, Tuple, Dict

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
import json
from torch import nn
from torch import distributions as torchd

# from evaluate_narration_model import plot_trajectory_graph
from narration.mineclip_narrator import MineCLIPNarrator
from narration.panda_narrator import PandaPushColourNarrator
from narration.crafter_narrator import CrafterNarrator
from narration.minigrd_narrator import (
    MiniGridFourSquareNarrator,
    MiniGridTeleportNarrator,
    MiniGridComplexTeleportNarrator,
)
from narration.safetygym_narrator import IslandNavigationNarrator
from narration.ai2thor_narrator import CookEggNarrator
from evaluation import (
    sample_rollouts,
    evaluate_rollouts,
    evaluate_language_to_action,
    crafter_narration_using_obs_reconstruction,
    minigrid_narration_using_obs_reconstruction,
    ai2thor_narration_using_obs_reconstruction,
)


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self.training = False
        # this is update step
        if logger is not None:
            self._step = logger.step // config.action_repeat
        else:
            self._step = 0
        self._update_count = 0
        self._dataset = dataset
        narrator = None
        if config.enable_language:
            narrator = configure_narrator(config)
        self._wm = models.WorldModel(
            obs_space, act_space, self._step, config, narrator=narrator
        )
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None):
        step = self._step
        training = self.training
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                # plot_trajectory_graph(self, step=self._logger.step)
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(
                        next(self._dataset),
                        ignore_keys=[
                            "semantic",
                            "inventory",
                            "achievements",
                            "privileged_obs",
                        ]
                        + self._config.narrator["narration_key"],
                    )
                    self._logger.video("train_openl", to_np(openl))
                    pass
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(
            obs,
            keys_to_ignore=["privileged_obs"] + self._config.narrator["narration_key"],
        )
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length, config.seed)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def configure_narrator(config):
    if "minedojo" in config.task:
        if config.mineclip_ckpt_path is not None:
            with open(config.prompt_path, "r") as f:
                prompts = json.load(f)[config.minedojo_task_id]
            narrator = MineCLIPNarrator(
                config.mineclip_ckpt_path,
                torch.device("cuda"),
                prompts,
            )
    elif "minigrid" in config.task:
        if "four_squares" in config.task:
            narrator = MiniGridFourSquareNarrator()
        elif "teleport" in config.task:
            if "complex" in config.task:
                narrator = MiniGridComplexTeleportNarrator()
            else:
                narrator = MiniGridTeleportNarrator()
    elif "panda" in config.task:
        narrator = PandaPushColourNarrator()
    elif "crafter" in config.task:
        narrator = CrafterNarrator()
    elif "safegym" in config.task:
        narrator = IslandNavigationNarrator()
    elif "ai2thor" in config.task:
        narrator = CookEggNarrator()

    return narrator


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    print(f"Creating {suite} environment for {task}...")
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)

    elif suite == "minigrid":
        import envs.minigrid_env as minigrid_env

        env = minigrid_env.MiniGrid(
            task_name=task,
            img_size=config.size,
            actions=config.actions,
            max_length=config.time_limit,
        )
        env = wrappers.OneHotAction(env)

    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "minedojo":
        print(f"Creating MineDojo environment...")
        import envs.minedojo_env as minedojo_env

        env = minedojo_env.MineDojoEnv(task_id=task, world_seed=config.world_seed)
        print(f"ACTION SPACE: {env.action_space}")
    elif suite == "panda":
        import envs.panda_env as panda_env

        env = panda_env.PandaEnv(task, img_size=config.size, seed=config.seed + id)
    elif suite == "safegym":
        import envs.safe_gym_env as safegym_env

        env = safegym_env.SafeGymEnv(
            task,
            img_size=config.size,
            max_length=config.time_limit,
            seed=config.seed + id,
        )
    elif suite == "ai2thor":
        import envs.ai2thor_env as ai2thor_env

        if task == "cook_egg":
            env = ai2thor_env.CookEggEnv(
                img_size=config.size, seed=config.seed, max_length=config.time_limit
            )
        elif task == "pickup":
            env = ai2thor_env.PickupObjects(
                img_size=config.size,
                seed=config.seed,
                max_length=config.time_limit,
                reconstruct_obs=config.evaluate_reconstruction_narration,
            )
        else:
            raise ValueError(f"Invalid ai2thor task: {task}")
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)

    return env


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)


def prefill_dataset(
    train_envs: List[Union[Parallel, Damy]],
    train_eps: dict,
    config: argparse.Namespace,
    logger: tools.Logger,
    action_space,
) -> None:
    """_summary_

    Args:
        train_envs (List[Union[Parallel, Damy]]): _description_
        train_eps (List[dict]): _description_
        config (argparse.Namespace): _description_
    """

    prefill = max(0, config.prefill - count_steps(config.traindir))
    print(f"Prefill dataset ({prefill} steps).")
    if hasattr(action_space, "discrete"):
        random_actor: Union[tools.OneHotDist, torchd.independent.Independent] = (
            tools.OneHotDist(torch.zeros(config.num_actions).repeat(config.envs, 1))
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.Tensor(action_space.low).repeat(config.envs, 1),
                torch.Tensor(action_space.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    _ = tools.simulate(
        random_agent,
        train_envs,
        train_eps,
        config.traindir,
        logger,
        limit=config.dataset_size,
        steps=prefill,
        no_convert_obs=config.narrator["narration_key"],
        no_save_obs=["rays"],  #
        info_keys_to_store=config.narrator["narration_key"],
    )
    logger.step += prefill * config.action_repeat
    print(f"Logger: ({logger.step} steps).")


def create_environments(
    config,
) -> Tuple[List[Union[Parallel, Damy]], List[Union[Parallel, Damy]]]:

    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    return train_envs, eval_envs


def load_existing_episodes(config: argparse.Namespace) -> Tuple[dict, dict]:
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    eval_eps = tools.load_episodes(directory, limit=1)
    return train_eps, eval_eps


def setup(config: argparse.Namespace) -> Tuple[argparse.Namespace, pathlib.Path]:
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    return config, logdir


def main(config):
    import wandb

    # Start a wandb run with `sync_tensorboard=True`
    config, logdir = setup(config)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    run = wandb.init(
        project="somniloquy",
        notes="language-metrics-test",
        config=config,
    )
    logger = tools.Logger(logdir, config.action_repeat * step, wandb_run=run)

    print("Create envs.")
    train_envs, eval_envs = create_environments(config)

    print("Load existing episodes.")
    train_eps, eval_eps = load_existing_episodes(config)

    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"NUMBER OF ACTIONS: {config.num_actions}")
    state = None
    if not config.offline_traindir:
        prefill_dataset(train_envs, train_eps, config, logger, acts)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # interactive_language_to_action(agent, eval_envs[0])
    # raise ValueError("DONE")

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            # partial is used to ensure that training is set to false
            # at every __call__ to the agent.
            # eval_policy = functools.partial(agent, training=False)
            agent.training = False
            with torch.no_grad():
                tools.simulate(
                    agent,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                    no_convert_obs=config.narrator["narration_key"],
                    no_save_obs=["rays"],
                    info_keys_to_store=config.narrator["narration_key"],
                    wandb_run=run,
                )
                rollout_samples = sample_rollouts(
                    agent,
                    eval_envs[0],
                    config.n_eval_samples,
                    trajectory_length=config.eval_trajectory_length,
                    n_consecutive_trajectories=config.eval_n_consecutive_trajectories,
                )
                evaluate_rollouts(
                    agent,
                    rollout_samples["imagined_state_samples"],
                    rollout_samples["imagined_action_samples"],
                    rollout_samples["posterior_state_samples"],
                    rollout_samples["observation_samples"],
                    logger=logger,
                    trajectory_length=config.eval_trajectory_length
                    + 1,  # +1 as we include starting states
                    wandb_run=run,
                    save_plots=config.save_plots,
                    save_translations=config.save_translations,
                )
            if config.evaluate_reconstruction_narration:
                if "crafter" in config.task:
                    crafter_narration_using_obs_reconstruction(
                        agent,
                        rollout_samples["imagined_state_samples"],
                        rollout_samples["imagined_action_samples"],
                        rollout_samples["posterior_state_samples"],
                        rollout_samples["observation_samples"],
                        logger=logger,
                        trajectory_length=config.eval_trajectory_length + 1,
                        wandb_run=run,
                    )
                elif "minigrid" in config.task:
                    minigrid_narration_using_obs_reconstruction(
                        agent,
                        rollout_samples["imagined_state_samples"],
                        rollout_samples["imagined_action_samples"],
                        rollout_samples["posterior_state_samples"],
                        rollout_samples["observation_samples"],
                        logger=logger,
                        trajectory_length=config.eval_trajectory_length + 1,
                        wandb_run=run,
                        narrator=configure_narrator(config),
                        obs_size=train_envs[0]
                        .observation_space["occupancy_grid"]
                        .shape,
                    )
                elif "ai2thor" in config.task:
                    ai2thor_narration_using_obs_reconstruction(
                        agent,
                        eval_envs[0],
                        rollout_samples["imagined_state_samples"],
                        rollout_samples["imagined_action_samples"],
                        rollout_samples["posterior_state_samples"],
                        rollout_samples["observation_samples"],
                        logger=logger,
                        wandb_run=run,
                        trajectory_length=config.eval_trajectory_length + 1,
                    )

            if config.enable_language_to_action:
                evaluate_language_to_action(
                    agent, eval_envs[0], source_strings=config.eval_strings
                )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(
                    next(eval_dataset),
                    ignore_keys=config.narrator["narration_key"],
                )
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        agent.training = True
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            no_convert_obs=config.narrator["narration_key"],
            no_save_obs=["rays"],
            info_keys_to_store=config.narrator["narration_key"],
            wandb_run=run,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / f"latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main(setup_args())
