from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fire
import yaml
from rich import print

from src.core.models import Model
from src.core.utils import GenerationManager
from src.reading.scrape_study_content import normalize_title, read_study_content

WORKING_DIR = Path.cwd()
PROMPTS_PATH = WORKING_DIR / "src" / "prompts" / "study.yaml"
TOPICS_PATH = WORKING_DIR / "data" / "study" / "topics.yaml"
DEFAULT_CONTENT_DIR = WORKING_DIR.parent / "content" / "reading"
MAX_CONTENT_TOKENS = 80_000


def main(**kwargs: Any) -> None:
    """指定トピックでStage 2（読書モード）を実行する。

    Parameters
    ----------
    **kwargs : Any
        seed : int
            乱数シード。
        run_dir : str
            出力先ディレクトリ（必須）。
        max_content_tokens : int, optional
            読ませるトークン上限。デフォルトは80,000。
        study_topic_index : int, optional
            読書対象のトピックID。
        study_topic_type : str, optional
            トピック種別（例: conservative）。
        model_name : str, optional
            使用するモデル名。デフォルトは "gpt-5.1"。
        content_dir : str, optional
            読み込み元コンテンツディレクトリ。指定がない場合は
            リポジトリ親の `content/reading` を参照。
    """
    seed = int(kwargs.get("seed", 0))
    run_dir_arg = kwargs.get("run_dir")
    assert run_dir_arg is not None, "run_dir is required"
    run_dir = Path(str(run_dir_arg))
    max_content_tokens = int(kwargs.get("max_content_tokens", MAX_CONTENT_TOKENS))
    study_topic_index = int(kwargs.get("study_topic_index", 0))
    study_topic_type = str(kwargs.get("study_topic_type", "none"))
    model_name = str(kwargs.get("model_name", "gpt-5.1"))
    content_dir_arg = kwargs.get("content_dir")
    content_dir = Path(str(content_dir_arg)) if content_dir_arg is not None else DEFAULT_CONTENT_DIR

    exp_config = dict(
        seed=seed,
        run_dir=str(run_dir),
    )

    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)

    with open(TOPICS_PATH, "r", encoding="utf-8") as file:
        topics = yaml.safe_load(file)

    study_candidates = [
        topic for topic in topics["study"][study_topic_type] if topic["id"] == study_topic_index
    ]
    assert study_candidates, f"study_topic_index {study_topic_index} not found"
    study_topic_name = study_candidates[0]["name"]

    data_config = dict(
        study_topic_index=study_topic_index,
        study_topic_type=study_topic_type,
        topics=topics,
        prompts=prompts,
        max_content_tokens=max_content_tokens,
        content_dir=str(content_dir),
    )

    model = Model(model_name=model_name, temperature=kwargs.get("temperature"))

    generation_manager = GenerationManager(
        run_dir=str(run_dir),
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

    study_topic_name_text = normalize_title(study_topic_name)
    contents = read_study_content(str(content_dir), study_topic_name_text)

    template = prompts["study_content_template"]
    input_text_parts: list[str] = []
    for title_text, text in contents:
        input_text_parts.append(template.format(title_text=title_text, text=text))
    input_text = "\n\n".join(input_text_parts)
    input_text = " ".join(input_text.split(" ")[:max_content_tokens])
    input_text = f"{input_text}\n\n{prompts['init_study_prompt']}"

    response = model.generate(input_text)
    generation_manager.write_prediction(response)
    generation_manager.write_log(f"input_text:\n{input_text}")
    generation_manager.write_log("---")
    generation_manager.write_log(f"output_text:\n{response['output_text']}")
    generation_manager.write_log("---")

    time.sleep(1.0)
    print(f"Run finished: {run_dir}")
    generation_manager.write_log("#" * 100)

    generation_manager.save_json(model.history, "history.json")
    generation_manager.save_json(model.message_history, "message_history.json")


if __name__ == "__main__":
    fire.Fire(main)
