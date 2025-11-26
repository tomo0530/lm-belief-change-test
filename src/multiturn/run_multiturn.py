import json
import os
import time
from pathlib import Path

import fire
from rich import print
from tqdm import tqdm

from src.core.models import Model
from src.core.utils import GenerationManager, load_data_from_yaml

WORKING_DIR = os.getcwd()
MULTITURN_PROMPT_PATH = f"{WORKING_DIR}/src/prompts/multiturn.yaml"
PROMPTS_TO_MODES = {
    "one-sided": "one_sided_prompts",
    "two-sided": "two_sided_prompts",
}


def _model_family(model_name: str) -> str:
    m = model_name.lower()
    if "gemini" in m:
        return "gpt"
    if "grok" in m:
        return "gpt"
    if "claude" in m or "sonnet" in m:
        return "claude"
    if "deepseek" in m:
        return "deepseek"
    if "gpt-oss" in m or "oss" in m:
        return "oss"
    if "gpt" in m:
        return "gpt"
    raise ValueError(f"Unknown model family for: {model_name}")


def _belief_for(init_belief: dict, family: str, model_type: str) -> str:
    if model_type == "close":
        key_map = {
            "gpt": "belief_text_gpt",
            "claude": "belief_text_claude",
        }
    elif model_type == "open":
        key_map = {
            "oss": "belief_text_oss",
            "deepseek": "belief_text_deepseek",
            "gpt": "belief_text_oss",
            "claude": "belief_text_deepseek",
        }
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    key = key_map.get(family)
    if key is None:
        raise ValueError(f"Model family {family} not supported for type {model_type}")
    return init_belief[key]


def main(mode="two-sided", dataset_name="moral", **kwargs):
    """
    Model A is different from Alpha, Model B is different from Beta
    """
    num_rounds = kwargs.get("num_rounds", 3)
    model_names = kwargs.get("model_names", "gpt-5+claude-sonnet-4-20250514")
    seed = kwargs.get("seed", 0)
    model_type = kwargs.get("model_type", "open")
    persuasion_tech = kwargs.get("persuasion_tech", "discussion")

    # Load data
    dataset_path = (
        f"{WORKING_DIR}/data/multiturn/{dataset_name}/disagreement_{model_type}data.jsonl"
    )
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Load prompts
    PROMPTS = load_data_from_yaml(MULTITURN_PROMPT_PATH)
    if mode == "one-sided":
        system_prompt_a = PROMPTS[PROMPTS_TO_MODES[mode]][persuasion_tech]
        system_prompt_b = PROMPTS[PROMPTS_TO_MODES[mode]]["one_sided_response"]

    elif mode == "two-sided":
        system_prompt_a = PROMPTS[PROMPTS_TO_MODES[mode]][persuasion_tech]
        system_prompt_b = PROMPTS[PROMPTS_TO_MODES[mode]][persuasion_tech]

    else:
        raise ValueError(f"Invalid mode: {mode}")

    model_names = model_names.split("+")
    model_name_a, model_name_b = model_names

    # Initialize models
    model_a = Model(model_name_a, system_prompt=system_prompt_a)
    model_b = Model(model_name_b, system_prompt=system_prompt_b)
    model_name_short_a = model_name_a.split("/")[-1]
    model_name_short_b = model_name_b.split("/")[-1]

    if mode == "one-sided":
        run_dir = kwargs.get(
            "run_dir",
            f"./experiments/multiturn/{dataset_name}/conversations/mode={mode}_seed={seed}_nrounds={num_rounds}_ma={model_name_short_a}_mb={model_name_short_b}_pa={persuasion_tech}",
        )
    elif mode == "two-sided":
        run_dir = kwargs.get(
            "run_dir",
            f"./experiments/multiturn/{dataset_name}/conversations/mode={mode}_seed={seed}_nrounds={num_rounds}_ma={model_name_short_a}_mb={model_name_short_b}_pa=discussion",
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    generation_manager = GenerationManager(
        run_dir=run_dir,
        print_to_stdout=True,
        overwrite=True,
        dry_run=False,
    )

    # Get batched datapoints for parallel processing
    batch_size = kwargs.get("batch_size", 5)
    batch_datapoints = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]

    for i, batch_datapoint in tqdm(
        enumerate(batch_datapoints),
        total=len(batch_datapoints),
        desc="Running multi-turn interaction",
    ):
        print(f"Running batch {i} of {len(batch_datapoints)}")
        conversations_alpha = [[] for _ in range(len(batch_datapoint))]
        conversations_beta = [[] for _ in range(len(batch_datapoint))]

        fam_a = _model_family(model_name_a)
        fam_b = _model_family(model_name_b)

        for idx, dp in enumerate(batch_datapoint):
            datapoint = dp["datapoint"]
            init_belief = dp["init_belief"]

            if dataset_name == "moral":
                query = datapoint["moral_dilemma"]
            else:
                query = datapoint.get("query") or datapoint.get("moral_dilemma", "")

            # Align initial beliefs with A/B models
            init_belief_alpha = _belief_for(init_belief, fam_a, model_type)
            init_belief_beta = _belief_for(init_belief, fam_b, model_type)

            if mode == "two-sided":  # alpha are model A, beta are model B
                conversations_alpha[idx].append({"role": "user", "content": query})
                conversations_alpha[idx].append({"role": "assistant", "content": init_belief_alpha})
                conversations_alpha[idx].append({"role": "user", "content": init_belief_beta})

                conversations_beta[idx].append(
                    {
                        "role": "user",
                        "content": query
                        + f"\n\nAnother agent's initial viewpoint: {init_belief_alpha}",
                    }
                )
                conversations_beta[idx].append({"role": "assistant", "content": init_belief_beta})

            elif mode == "one-sided":  # Model B is persuaded; Model A is persuader
                conversations_alpha[idx].append(
                    {
                        "role": "user",
                        "content": query
                        + f"\n\nAnother agent's initial viewpoint: {init_belief_beta}",
                    }
                )
                conversations_beta[idx].append({"role": "user", "content": query})
                conversations_beta[idx].append({"role": "assistant", "content": init_belief_beta})
            else:
                raise ValueError(f"Invalid mode: {mode}")

        for _ in tqdm(range(num_rounds), total=num_rounds, desc="Running multi-turn persuasion"):
            outputs_alpha = model_a.generate_with_messages(
                conversations_alpha,
                parallelism=len(batch_datapoint),
            )
            for idx, output_alpha in enumerate(outputs_alpha):
                conversations_alpha[idx].append({"role": "assistant", "content": output_alpha})
                conversations_beta[idx].append({"role": "user", "content": output_alpha})

            outputs_beta = model_b.generate_with_messages(
                conversations_beta,
                parallelism=len(batch_datapoint),
            )
            for idx, output_beta in enumerate(outputs_beta):
                conversations_beta[idx].append({"role": "assistant", "content": output_beta})
                conversations_alpha[idx].append({"role": "user", "content": output_beta})

        for idx, datapoint in enumerate(batch_datapoint):
            save_obj_alpha = {
                "system_prompt": system_prompt_a,
                "conversations": conversations_alpha[idx],
                "datapoint": datapoint["datapoint"],
            }
            save_obj_beta = {
                "system_prompt": system_prompt_b,
                "conversations": conversations_beta[idx],
                "datapoint": datapoint["datapoint"],
            }
            # First write for the very first record; then append
            if i == 0 and idx == 0:
                generation_manager.save_jsonl(save_obj_alpha, "multiturn_alpha.jsonl", mode="w")
                generation_manager.save_jsonl(save_obj_beta, "multiturn_beta.jsonl", mode="w")
            else:
                generation_manager.save_jsonl(save_obj_alpha, "multiturn_alpha.jsonl", mode="a")
                generation_manager.save_jsonl(save_obj_beta, "multiturn_beta.jsonl", mode="a")

    exp_config = dict(
        mode=mode,
        dataset_name=dataset_name,
        num_rounds=num_rounds,
        persuasion_tech=persuasion_tech,
    )
    data_config = dict(
        dataset=dataset,
        batch_size=batch_size,
        system_prompt_a=system_prompt_a,
        system_prompt_b=system_prompt_b,
    )
    model_config = dict(
        model_name_a=model_name_a,
        model_name_b=model_name_b,
        model_type=model_type,
    )
    generation_manager.save_generation_config(
        dict(
            exp_config=exp_config,
            data_config=data_config,
            model_config=model_config,
        )
    )

    generation_manager.write_log(f"Run finished: {run_dir}")
    time.sleep(1.0)
    print(f"Run finished: {run_dir}")


if __name__ == "__main__":
    fire.Fire(main)
