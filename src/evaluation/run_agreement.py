import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import fire
import yaml
from rich import print

from src.core.models import Model
from src.core.utils import GenerationManager, parse_eval_output

WORKING_DIR = os.getcwd()
MULTITURN_DATA_DIR = f"{WORKING_DIR}/data/multiturn"
MULTITURN_PROMPTS_PATH = f"{WORKING_DIR}/src/prompts/multiturn.yaml"
STUDY_PROMPTS_PATH = f"{WORKING_DIR}/src/prompts/study.yaml"
STUDY_TOPICS_PATH = f"{WORKING_DIR}/data/study/topics.yaml"


def _drop_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """空文字メッセージを除去し、tool_calls付きは保持する。"""
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        has_tool_calls = bool(message.get("tool_calls"))
        if has_tool_calls:
            cleaned.append(message)
            continue
        if content is None:
            continue
        if isinstance(content, str) and content.strip() == "":
            continue
        cleaned.append(message)
    return cleaned


def main(exp_name: str, **kwargs: Any) -> None:
    # #########################################################################
    # General experiment args
    seed = kwargs.get("seed", 0)
    stage2_run_dir = kwargs.get("stage2_run_dir", "none")
    run_dir = kwargs.get("run_dir", None)
    assert run_dir is not None, "run_dir is required"

    exp_config = dict(
        exp_name=exp_name,
        seed=seed,
        run_dir=run_dir,
        stage2_run_dir=stage2_run_dir,
    )

    # Model args
    model_name = kwargs.get("model_name", "gpt-4o")
    temperature = float(kwargs.get("temperature", 0.1))
    judge_model_names_param = kwargs.get("judge_model_names")
    judge_model_names: Sequence[str] | None = (
        tuple(judge_model_names_param) if judge_model_names_param is not None else None
    )
    model = Model(model_name, temperature=temperature)
    ###########################################################################

    # Prepare data and prompts for in-depth reading and deep research
    if exp_name == "study":
        # Data & prompts args
        study_prompts_path = Path(kwargs.get("prompts_path") or STUDY_PROMPTS_PATH)
        study_topics_path = Path(kwargs.get("topics_path") or STUDY_TOPICS_PATH)

        with open(study_prompts_path, "r", encoding="utf-8") as f:
            PROMPTS = yaml.safe_load(f)
        with open(study_topics_path, "r", encoding="utf-8") as f:
            TOPICS = yaml.safe_load(f)

        survey_topic_index = kwargs.get("survey_topic_index", 0)
        survey_topic_name = TOPICS["survey"][survey_topic_index]["topic_name"]
        support_statement = TOPICS["survey"][survey_topic_index]["support_statement"]
        neutral_statement = TOPICS["survey"][survey_topic_index]["neutral_statement"]
        oppose_statement = TOPICS["survey"][survey_topic_index]["oppose_statement"]

        input_text = PROMPTS["agreement"].format(
            topic_name=survey_topic_name,
            support_statement=support_statement,
            neutral_statement=neutral_statement,
            oppose_statement=oppose_statement,
        )

        if stage2_run_dir != "none":
            with open(Path(stage2_run_dir) / "message_history.json", "r", encoding="utf-8") as f:
                message_history = json.load(f)
        else:
            message_history = []
        message_history = _drop_empty_messages(message_history)

        data_config = dict(
            survey_topic_index=survey_topic_index,
            topics=TOPICS,
            prompts=PROMPTS,
        )

    # Prepare data and prompts for multiturn interaction
    elif exp_name == "multiturn":
        with open(MULTITURN_PROMPTS_PATH, "r", encoding="utf-8") as f:
            PROMPTS = yaml.safe_load(f)

        dataset_name = kwargs.get("dataset_name", "moral")
        model_type = kwargs.get("model_type", "open")
        dataset_path = Path(
            kwargs.get(
                "dataset_path",
                f"{MULTITURN_DATA_DIR}/{dataset_name}/disagreement_{model_type}data.jsonl",
            )
        )
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        query_index = kwargs.get("query_index", 0)
        datapoint = dataset[query_index]["datapoint"]
        query = datapoint["moral_dilemma"] if dataset_name == "moral" else datapoint["query"]
        statements = datapoint["statements"]
        support_statement = statements["support_statement"]
        oppose_statement = statements["oppose_statement"]
        neutral_statement = statements["neutral_statement"]
        question = datapoint["likert_scale_question"]

        input_text = (
            PROMPTS["moral_agreement"].format(
                moral_dilemma=query,
                oppose_statement=oppose_statement,
                support_statement=support_statement,
                neutral_statement=neutral_statement,
                question=question,
            )
            if dataset_name == "moral"
            else PROMPTS["safety_agreement"].format(
                query=query,
                oppose_statement=oppose_statement,
                support_statement=support_statement,
                neutral_statement=neutral_statement,
                question=question,
            )
        )

        if stage2_run_dir != "none":
            rounds_num = kwargs.get("rounds_num", 10)
            mode = kwargs.get("mode", "two-sided")
            if mode == "two-sided":
                if model_name == "azure/gpt-oss-120b":
                    with open(Path(stage2_run_dir) / "multiturn_alpha.jsonl", "r") as f:
                        history_conversations = [json.loads(line) for line in f]
                    print(
                        f"Mode: {mode}. Loaded history conversations from {Path(stage2_run_dir) / 'multiturn_alpha.jsonl'}"
                    )
                elif model_name == "azure/DeepSeek-V3.1":
                    with open(Path(stage2_run_dir) / "multiturn_beta.jsonl", "r") as f:
                        history_conversations = [json.loads(line) for line in f]
                    print(
                        f"Mode: {mode}. Loaded history conversations from {Path(stage2_run_dir) / 'multiturn_beta.jsonl'}"
                    )
                else:
                    raise ValueError(f"Invalid model name: {model_name}")
            elif mode == "one-sided":
                with open(Path(stage2_run_dir) / "multiturn_beta.jsonl", "r") as f:
                    history_conversations = [json.loads(line) for line in f]
                print(
                    f"Mode: {mode}. Loaded history conversations from {Path(stage2_run_dir) / 'multiturn_beta.jsonl'}"
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

            history_conversation = history_conversations[query_index]
            conversations = history_conversation["conversations"]
            message_history = conversations[: rounds_num * 2 + 2]
        else:
            message_history = []
        message_history = _drop_empty_messages(message_history)

        data_config = dict(
            dataset_name=dataset_name,
            query_index=query_index,
            datapoint=datapoint,
        )
    else:
        raise ValueError(f"Invalid exp_name: {exp_name}")

    messages = message_history + [dict(role="user", content=input_text)]
    generation_manager = GenerationManager(
        run_dir=run_dir,
        print_to_stdout=True,
        overwrite=True,
        dry_run=False,
    )

    generation_manager.save_generation_config(
        dict(
            model_config=model.config,
            exp_config=exp_config,
            data_config=data_config,
        )
    )

    output_text = model.generate_with_messages(messages)
    subject_agreement_results = parse_eval_output(output_text, mode="agreement")

    judge_model_names = judge_model_names or (model_name,)
    agreement_results_by_model: dict[str, Any] = {str(model_name): subject_agreement_results}
    scores: list[float] = []
    for judge_name in judge_model_names:
        judge_model = Model(str(judge_name), temperature=temperature)
        judge_output = judge_model.generate_with_messages(messages)
        judge_result = parse_eval_output(judge_output, mode="agreement")
        agreement_results_by_model[str(judge_name)] = judge_result
        try:
            scores.append(float(judge_result.get("pred_label")))
        except (TypeError, ValueError):
            continue

    aggregated_score = (
        sum(scores) / len(scores) if scores else subject_agreement_results.get("pred_label")
    )
    agreement_results = {"pred_label": aggregated_score, "scores": scores}

    response = dict(
        messages=messages,
        output_text=output_text,
        agreement_results=agreement_results,
        subject_agreement_results=subject_agreement_results,
        agreement_results_by_model=agreement_results_by_model,
    )
    generation_manager.write_prediction(response)
    generation_manager.write_log("### messages ###")
    for m in messages:
        generation_manager.write_log(f"role: {m['role']}")
        generation_manager.write_log("content:")
        generation_manager.write_log(f"{m['content']}")
        generation_manager.write_log("---")
    generation_manager.write_log(f"### output_text ###\n{output_text}")
    generation_manager.write_log("---")
    generation_manager.write_log(f"### agreement_results (aggregated) ###\n{agreement_results}")
    generation_manager.write_log("---")
    generation_manager.write_log(
        f"### agreement_results_by_model ###\n{agreement_results_by_model}"
    )
    generation_manager.write_log("---")

    generation_manager.save_json(agreement_results, "agreement_results.json")
    generation_manager.save_json(agreement_results_by_model, "agreement_results_by_model.json")
    generation_manager.save_json(subject_agreement_results, "agreement_results_subject.json")
    time.sleep(1.0)
    print(f"Run finished: {run_dir}")


if __name__ == "__main__":
    fire.Fire(main)
