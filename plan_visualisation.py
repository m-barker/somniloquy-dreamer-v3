import torch
import argparse
from dreamer import setup_args, create_environments, Dreamer
from evaluation import visual_plan_evaluation
from tools import recursively_load_optim_state_dict


def main(output_dir: str):
    config = setup_args()
    env, _ = create_environments(config)
    env = env[0]  # create_environments returns a list due to parallel possibility
    config.num_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=None,
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    visual_plan_evaluation(agent, env, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saves visualisations of plan translation and rollouts"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to save visualisation results",
        required=True,
    )
    args = vars(parser.parse_args())
    main(args["output_dir"])
