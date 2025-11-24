from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import fire
import yaml

from src.evaluation import run_agreement, run_behavior, run_belief
from src.multiturn import run_multiturn
from src.reading import run_study


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _survey_topic_ids(config: dict[str, Any]) -> Iterable[int]:
    survey_ids = config.get("survey_topic_ids")
    if survey_ids:
        return [int(idx) for idx in survey_ids]
    return range(51, 61)


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _behavior_ids(common: dict[str, Any]) -> Sequence[int]:
    scenarios = common.get("behavior_scenarios")
    if not scenarios:
        return ()
    return tuple(int(item) for item in scenarios)


def _run_stage1(
    pattern: dict[str, Any],
    common: dict[str, Any],
    base_dir: Path,
) -> None:
    subject_llm = pattern["subject_llm"]
    evaluator_llms: Sequence[str] = tuple(pattern.get("evaluator_llms", ()))
    seed = int(common["seed"])
    temperature = float(common["temperature"])
    topics = list(_survey_topic_ids(common))
    behavior_ids = list(_behavior_ids(common))
    model_dir = base_dir / "stage1" / _model_dir_name(subject_llm)
    for topic_id in topics:
        run_dir = model_dir / f"topic_{topic_id}"
        _ensure_dir(run_dir)
        run_belief.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(run_dir),
            survey_topic_index=topic_id,
            temperature=temperature,
            prompts_path=common.get("study_prompts_path"),
            topics_path=common.get("topics_path"),
            judge_model_names=evaluator_llms,
        )
        run_agreement.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(run_dir / "agreement"),
            survey_topic_index=topic_id,
            stage2_run_dir="none",
            prompts_path=common.get("study_prompts_path"),
            topics_path=common.get("topics_path"),
            judge_model_names=evaluator_llms,
        )
    for scenario_id in behavior_ids:
        scenario_dir = model_dir / f"behavior_{scenario_id}"
        _ensure_dir(scenario_dir)
        run_behavior.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(scenario_dir),
            stage2_run_dir="none",
            behavior_scenario_id=scenario_id,
            behavior_scenarios_path=common.get("behavior_scenarios_path"),
            prompts_path=common.get("study_prompts_path"),
            judge_model_names=evaluator_llms,
        )


def _run_stage2(
    pattern: dict[str, Any],
    common: dict[str, Any],
    base_dir: Path,
) -> Path:
    subject_llm = pattern["subject_llm"]
    opponent_llm = pattern["opponent_llm"]
    seed = int(common["seed"])
    temperature = float(common["temperature"])
    dataset_name = common["dataset_name"]
    rounds_num = int(common["multiturn_rounds"])
    stage2_modes: Sequence[str] = tuple(common.get("stage2_modes", ("multiturn", "reading")))

    if "multiturn" in stage2_modes:
        multiturn_dir = base_dir / "stage2_multiturn"
        _ensure_dir(multiturn_dir)
        run_multiturn.main(
            mode="two-sided",
            dataset_name=dataset_name,
            model_names=f"{subject_llm}+{opponent_llm}",
            num_rounds=rounds_num,
            seed=seed,
            persuasion_tech="discussion",
            run_dir=str(multiturn_dir),
        )

    reading_dir = base_dir / "stage2_reading"
    if "reading" in stage2_modes:
        _ensure_dir(reading_dir)
        run_study.main(
            model_name=subject_llm,
            study_topic_type=common["study_topic_type"],
            study_topic_index=int(common["study_topic_index"]),
            seed=seed,
            temperature=temperature,
            run_dir=str(reading_dir),
            content_dir=common["content_dir"],
        )
    return reading_dir


def _run_stage3(
    pattern: dict[str, Any],
    common: dict[str, Any],
    base_dir: Path,
    stage2_reading_dir: Path,
) -> None:
    subject_llm = pattern["subject_llm"]
    evaluator_llms: Sequence[str] = tuple(pattern.get("evaluator_llms", ()))
    seed = int(common["seed"])
    temperature = float(common["temperature"])
    topics = list(_survey_topic_ids(common))
    behavior_ids = list(_behavior_ids(common))

    model_dir = base_dir / "stage3" / _model_dir_name(subject_llm)
    for topic_id in topics:
        run_dir = model_dir / f"topic_{topic_id}"
        _ensure_dir(run_dir)
        run_belief.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(run_dir),
            stage2_run_dir=str(stage2_reading_dir),
            survey_topic_index=topic_id,
            temperature=temperature,
            prompts_path=common.get("study_prompts_path"),
            topics_path=common.get("topics_path"),
            judge_model_names=evaluator_llms,
        )
        run_agreement.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(run_dir / "agreement"),
            stage2_run_dir=str(stage2_reading_dir),
            survey_topic_index=topic_id,
            prompts_path=common.get("study_prompts_path"),
            topics_path=common.get("topics_path"),
            judge_model_names=evaluator_llms,
        )
    for scenario_id in behavior_ids:
        scenario_dir = model_dir / f"behavior_{scenario_id}"
        _ensure_dir(scenario_dir)
        judge_models: Sequence[str] = evaluator_llms or (subject_llm,)
        run_behavior.main(
            exp_name="study",
            model_name=subject_llm,
            seed=seed,
            run_dir=str(scenario_dir),
            stage2_run_dir=str(stage2_reading_dir),
            behavior_scenario_id=scenario_id,
            behavior_scenarios_path=common.get("behavior_scenarios_path"),
            prompts_path=common.get("study_prompts_path"),
            judge_model_names=judge_models,
        )


def main(pattern: int = 1, stage: str = "all") -> None:
    """実験パターンを指定してStage 1-3を実行します。"""
    config_path = Path("config") / "experiment_patterns.yaml"
    config = _load_config(config_path)
    patterns = config["patterns"]
    common = config["common"]
    pattern_map = {p["id"]: p for p in patterns}
    if pattern not in pattern_map:
        raise ValueError(f"pattern {pattern} is not defined in config")
    selected = pattern_map[pattern]
    base_dir = Path(common["output_dir"]) / f"pattern_{pattern}"

    stage_str = str(stage)

    if stage_str in ("1", "all"):
        _run_stage1(selected, common, base_dir)

    stage2_reading_dir = base_dir / "stage2_reading"
    if stage_str in ("2", "all"):
        stage2_reading_dir = _run_stage2(selected, common, base_dir)

    if stage_str in ("3", "all"):
        if not stage2_reading_dir.exists():
            raise FileNotFoundError(f"stage2_reading directory not found: {stage2_reading_dir}")
        _run_stage3(selected, common, base_dir, stage2_reading_dir)


if __name__ == "__main__":
    fire.Fire(main)
