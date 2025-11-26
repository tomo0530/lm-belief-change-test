import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import fire
import yaml
from rich import print

from src.core.agent_runtime import AgentRuntime
from src.core.models import Model
from src.core.tools import NAMES, TOOL_REGISTRY
from src.core.utils import (
    GenerationManager,
    get_scenario_koizumi_aligned,
    normalize_behavior_result,
    parse_eval_output,
)

WORKING_DIR = os.getcwd()
MULTITURN_DATA_DIR = f"{WORKING_DIR}/data/multiturn"
MULTITURN_PROMPTS_PATH = f"{WORKING_DIR}/src/prompts/multiturn.yaml"
STUDY_PROMPTS_PATH = f"{WORKING_DIR}/src/prompts/study.yaml"
STUDY_TOPICS_PATH = f"{WORKING_DIR}/data/study/topics.yaml"
STUDY_TOOLS_PATH = f"{WORKING_DIR}/data/study/tools.yaml"
BEHAVIOR_SCENARIOS_PATH = f"{WORKING_DIR}/data/behavior/koizumi_behavior.yaml"


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


def run_judge(judge_messages: list[dict[str, str]], judge_model: Model) -> dict[str, Any]:
    judge_outputs = judge_model.generate_with_messages(judge_messages)
    judge_results = parse_eval_output(judge_outputs, mode="label")
    try:
        answer_idx = judge_outputs.find("The answer is:")
        judge_text = judge_outputs[:answer_idx].strip()
    except Exception:
        judge_text = judge_outputs
    return dict(judge_results=judge_results, judge_text=judge_text)


def _load_behavior_scenario(path: Path, scenario_id: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file)
    for scenario in data.get("scenarios", []):
        if int(scenario.get("id", -1)) == scenario_id:
            result: dict[str, Any] = scenario
            return result
    raise ValueError(f"behavior scenario id {scenario_id} not found in {path}")


def _load_behavior_scenarios_data(path: Path) -> dict[str, Any]:
    """行動シナリオ定義全体を読み込む（正規化用）。"""
    with open(path, "r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file)
        return data


def main(exp_name: str, **kwargs: Any) -> None:
    seed = kwargs.get("seed", 0)
    stage2_run_dir = kwargs.get("stage2_run_dir", "none")
    run_dir = kwargs.get("run_dir")
    assert run_dir is not None, "run_dir is required"

    exp_config = dict(exp_name=exp_name, seed=seed, run_dir=run_dir, stage2_run_dir=stage2_run_dir)

    model_name = kwargs.get("model_name", "gpt-5")
    temperature = float(kwargs.get("temperature", 0.1))
    model = Model(model_name, temperature=temperature)
    judge_model_name = kwargs.get("judge_model_name", model_name)
    judge_model_names_param = kwargs.get("judge_model_names")
    judge_model_names: Sequence[str] | None = (
        tuple(judge_model_names_param) if judge_model_names_param is not None else None
    )

    data_config: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    final_text = ""
    judge_prompt = ""
    support_statement = ""
    oppose_statement = ""

    if exp_name == "study":
        behavior_scenario_id = int(kwargs.get("behavior_scenario_id", -1))
        prompts_path = Path(kwargs.get("prompts_path") or STUDY_PROMPTS_PATH)
        topics_path = Path(kwargs.get("topics_path") or STUDY_TOPICS_PATH)

        with open(prompts_path, "r", encoding="utf-8") as f:
            PROMPTS = yaml.safe_load(f)
        with open(topics_path, "r", encoding="utf-8") as f:
            TOPICS = yaml.safe_load(f)

        survey_topic_index = kwargs.get("survey_topic_index", 0)
        survey_topic_name = TOPICS["survey"][survey_topic_index]["topic_name"]
        support_statement = TOPICS["survey"][survey_topic_index]["support_statement"]
        oppose_statement = TOPICS["survey"][survey_topic_index]["oppose_statement"]

        if behavior_scenario_id >= 0:
            scenario_path = Path(kwargs.get("behavior_scenarios_path") or BEHAVIOR_SCENARIOS_PATH)
            scenario = _load_behavior_scenario(scenario_path, behavior_scenario_id)
            # koizumi_aligned_option を取得（正規化用）
            scenarios_data = _load_behavior_scenarios_data(scenario_path)
            koizumi_aligned_option = get_scenario_koizumi_aligned(
                scenarios_data, behavior_scenario_id
            )
            prompt_template = PROMPTS.get("behavior_choice")
            if not prompt_template:
                raise ValueError("behavior_choice prompt is not defined in study prompts.")
            user_prompt = prompt_template.format(
                scenario=scenario["description"],
                option_a=scenario["option_a"],
                option_b=scenario["option_b"],
            )
            if stage2_run_dir != "none":
                with open(
                    Path(stage2_run_dir) / "message_history.json", "r", encoding="utf-8"
                ) as f:
                    message_history = json.load(f)
            else:
                message_history = []
            message_history = _drop_empty_messages(message_history)
            messages = message_history + [dict(role="user", content=user_prompt)]
            final_text = model.generate_with_messages(messages)
            judge_prompt_template = PROMPTS.get("behavior_judgement")
            if not judge_prompt_template:
                raise ValueError("behavior_judgement prompt is not defined in study prompts.")
            judge_prompt = judge_prompt_template.format(
                scenario=scenario["description"],
                response=final_text,
                support_statement=support_statement,
                oppose_statement=oppose_statement,
            )
            judge_results = None
            judge_text = ""
            data_config = dict(
                survey_topic_index=survey_topic_index,
                survey_topic_name=survey_topic_name,
                behavior_scenario_id=behavior_scenario_id,
                scenario=scenario,
                prompt_text=user_prompt,
                koizumi_aligned_option=koizumi_aligned_option,
            )
        else:
            with open(STUDY_TOOLS_PATH, "r", encoding="utf-8") as f:
                TOOLS = yaml.safe_load(f)

            tool_cls = TOOL_REGISTRY.get(survey_topic_index)
            if tool_cls is None:
                raise ValueError(
                    "Tool-based behavior is not available for this topic. "
                    "Provide behavior_scenario_id for simple evaluation."
                )
            tool_impl = tool_cls()

            agentic_prompt = TOOLS["survey"][survey_topic_index]["prompt"]
            agentic_tools = TOOLS["survey"][survey_topic_index]["tools"]

            # ツールベースの場合、koizumi_aligned_option は未定義
            koizumi_aligned_option = None

            data_config = dict(
                survey_topic_index=survey_topic_index,
                survey_topic_name=survey_topic_name,
                prompts=agentic_prompt,
                tools=agentic_tools,
                prompt_text=agentic_prompt,
                koizumi_aligned_option=koizumi_aligned_option,
            )

            if stage2_run_dir != "none":
                with open(
                    Path(stage2_run_dir) / "message_history.json", "r", encoding="utf-8"
                ) as f:
                    message_history = json.load(f)
            else:
                message_history = []
            message_history = _drop_empty_messages(message_history)
            messages = message_history + [dict(role="user", content=agentic_prompt)]

            runtime = AgentRuntime(tool_impl, terminal_names=NAMES)
            try:
                loop_out = runtime.fc_agentic_loop(
                    model=model,
                    base_messages=messages,
                    agentic_tools=agentic_tools,
                    max_steps=kwargs.get("max_steps", 8),
                    allow_repeat=False,
                )
            except Exception as e:
                print(f"[Warning] FC failed: {e}. Falling back to JSON loop.")
                tools_schema_json = json.dumps(agentic_tools, ensure_ascii=False, indent=2)
                loop_out = runtime.json_agentic_loop(
                    model=model,
                    base_messages=messages,
                    tools_schema_json=tools_schema_json,
                    max_steps=kwargs.get("max_steps", 8),
                    allow_repeat=False,
                )

            results = loop_out["results"]
            did_submit = loop_out["did_submit"]
            final_text = loop_out["final_text"]
            messages = loop_out["messages"]

            print("\n[TOOL RESULTS]")
            print(json.dumps(results, ensure_ascii=False, indent=2))
            if final_text:
                print("\n[FINAL MESSAGE]")
                print(final_text.strip())
            if did_submit:
                print("\n[DONE]")

            judge_prompt = PROMPTS["behavior_judgement"].format(
                scenario=agentic_prompt,
                response=final_text,
                support_statement=support_statement,
                oppose_statement=oppose_statement,
            )

    elif exp_name == "multiturn":
        with open(MULTITURN_PROMPTS_PATH, "r", encoding="utf-8") as f:
            PROMPTS = yaml.safe_load(f)

        dataset_name = kwargs.get("dataset_name", "moral")
        model_type = kwargs.get("model_type", "open")
        dataset_path = f"{MULTITURN_DATA_DIR}/{dataset_name}/disagreement_{model_type}data.jsonl"
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        query_index = kwargs.get("query_index", 0)
        datapoint = dataset[query_index]["datapoint"]
        input_text = datapoint["behavior_eval"]

        # multiturn では koizumi_aligned_option は未定義
        koizumi_aligned_option = None

        if stage2_run_dir != "none":
            rounds_num = kwargs.get("rounds_num", 10)
            mode = kwargs.get("mode", "two-sided")
            if mode == "two-sided":
                if model_name == "azure/gpt-oss-120b":
                    with open(
                        Path(stage2_run_dir) / "multiturn_alpha.jsonl", "r", encoding="utf-8"
                    ) as f:
                        history_conversations = [json.loads(line) for line in f]
                    print(
                        f"Mode: {mode}. Loaded history conversations from {Path(stage2_run_dir) / 'multiturn_alpha.jsonl'}"
                    )
                elif model_name == "azure/DeepSeek-V3.1":
                    with open(
                        Path(stage2_run_dir) / "multiturn_beta.jsonl", "r", encoding="utf-8"
                    ) as f:
                        history_conversations = [json.loads(line) for line in f]
                    print(
                        f"Mode: {mode}. Loaded history conversations from {Path(stage2_run_dir) / 'multiturn_beta.jsonl'}"
                    )
                else:
                    raise ValueError(f"Invalid model name: {model_name}")
            elif mode == "one-sided":
                with open(
                    Path(stage2_run_dir) / "multiturn_beta.jsonl", "r", encoding="utf-8"
                ) as f:
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
            koizumi_aligned_option=koizumi_aligned_option,
        )
        messages = message_history + [dict(role="user", content=input_text)]
        final_text = model.generate_with_messages(messages)
        query = input_text
        statements = datapoint["statements"]
        support_statement = statements["support_statement"]
        oppose_statement = statements["oppose_statement"]

        judge_prompt = (
            PROMPTS["moral_behavior"].format(
                moral_dilemma=query,
                response=final_text,
                oppose_statement=oppose_statement,
                support_statement=support_statement,
            )
            if dataset_name == "moral"
            else PROMPTS["safety_behavior"].format(
                query=query,
                response=final_text,
                oppose_statement=oppose_statement,
                support_statement=support_statement,
            )
        )
    else:
        raise ValueError(f"Invalid exp_name: {exp_name}")

    judge_outputs_by_model: dict[str, dict[str, Any]] = {}
    if judge_prompt:
        model_names = judge_model_names or (judge_model_name,)
        judge_messages = [dict(role="user", content=judge_prompt)]
        for model_name_candidate in model_names:
            judge_model = Model(str(model_name_candidate), temperature=temperature)
            judge_outputs = run_judge(judge_messages, judge_model)
            judge_outputs_by_model[str(model_name_candidate)] = judge_outputs

        vote_counts: dict[str, int] = {}
        for outputs in judge_outputs_by_model.values():
            label = outputs.get("judge_results", {}).get("pred_label")
            if label is None:
                continue
            vote_counts[label] = vote_counts.get(label, 0) + 1
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_labels = [lbl for lbl, cnt in vote_counts.items() if cnt == max_votes]
            aggregated_label = top_labels[0] if len(top_labels) == 1 else "Neutral"
        else:
            aggregated_label = parse_eval_output(final_text, mode="label").get("pred_label")
        # 出力テキストは最初のジャッジのものを代表で保存
        primary_judge = str(model_names[0])
        judge_results = {"pred_label": aggregated_label, "votes": vote_counts}
        judge_text = judge_outputs_by_model[primary_judge].get("judge_text", "")
    else:
        judge_results = parse_eval_output(final_text, mode="label")
        judge_text = final_text
        judge_outputs_by_model[str(model_name)] = dict(
            judge_results=judge_results, judge_text=judge_text
        )

    # 正規化された結果を計算
    aggregated_pred_label = judge_results.get("pred_label") if judge_results else None
    normalized_aggregated_label = normalize_behavior_result(
        aggregated_pred_label, koizumi_aligned_option
    )

    normalized_results_by_model: dict[str, str | None] = {}
    for model_key, outputs in judge_outputs_by_model.items():
        raw_label = outputs.get("judge_results", {}).get("pred_label")
        normalized_results_by_model[model_key] = normalize_behavior_result(
            raw_label, koizumi_aligned_option
        )

    normalized_behavior_results = {
        "pred_label": normalized_aggregated_label,
        "koizumi_aligned_option": koizumi_aligned_option,
        "by_model": normalized_results_by_model,
    }

    response = dict(
        messages=messages,
        output_text=final_text,
        behavior_results=judge_results,
        judge_text=judge_text,
        judge_outputs_by_model=judge_outputs_by_model,
        normalized_behavior_results=normalized_behavior_results,
    )
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
    generation_manager.write_prediction(response)
    generation_manager.write_log("### messages ###")
    for m in messages:
        role = m.get("role", "")
        generation_manager.write_log(f"role: {role}")
        generation_manager.write_log("content:")
        if "content" in m:
            generation_manager.write_log(f"{m['content']}")
        elif "tool_calls" in m:
            generation_manager.write_log(json.dumps(m["tool_calls"], ensure_ascii=False))
        else:
            generation_manager.write_log(json.dumps(m, ensure_ascii=False))
        generation_manager.write_log("---")
    generation_manager.write_log(f"### Final text ###\n{final_text}")
    generation_manager.write_log("---")
    generation_manager.write_log(f"### behavior_results ###\n{judge_results}")
    generation_manager.write_log("---")
    generation_manager.write_log(
        f"### normalized_behavior_results ###\n{normalized_behavior_results}"
    )
    generation_manager.write_log("---")

    generation_manager.save_json(judge_results, "behavior_results.json")
    if len(judge_outputs_by_model) > 1:
        generation_manager.save_json(judge_outputs_by_model, "behavior_results_by_model.json")
    generation_manager.save_json(normalized_behavior_results, "normalized_behavior_results.json")
    time.sleep(1.0)
    print(f"Run finished: {run_dir}")


if __name__ == "__main__":
    fire.Fire(main)
