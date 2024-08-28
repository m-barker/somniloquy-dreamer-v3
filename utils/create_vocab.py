import argparse
import json
from typing import List


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Converts a list of prompts to a vocabulary"
    )
    parser.add_argument("--prompt_json", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--output_json", type=str)
    return vars(parser.parse_args())


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def main(args: dict) -> None:
    prompts: List[str] = load_json(args["prompt_json"])[args["task_name"]]
    all_prompts = " ".join(prompts)
    vocab = set(all_prompts.split())
    vocab_list = list(vocab)
    vocab_list.insert(0, "<UNK>")
    vocab_list.insert(0, "<EOS>")
    vocab_list.insert(0, "<BOS>")
    vocab_list.insert(0, "<PAD>")

    vocab_dict = {}
    for i, word in enumerate(vocab_list):
        vocab_dict[word] = i
    with open(args["output_json"], "w") as f:
        json.dump(vocab_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
