import json
import re
import unicodedata
from pathlib import Path

import yaml
from rich.console import Console

console = Console(markup=False)
FORMAT_LIKERT = r"(?i)[^\w\n]*(the\s+answer\s+is|回答\s*[:：]?)\s*:?\s*([0-9]+)(?:\.)?[^\w\n]*"
FORMAT_LABEL = r"(?i)[^\w\n]*(the\s+answer\s+is|回答\s*[:：]?)\s*:?\s*([A-Z])(?:\.)?[^\w\n]*"


class GenerationManager:
    """
    Example usage:

    ```python
    config = dict(
        config_key1='config_value1',
        config_key2='config_value2',
    )

    generation_manager = GenerationManager(
        run_dir='./experiments/in-depth-reading',
        print_to_stdout=True,
        overwrite=True,
        dry_run=False,
    )
    generation_manager.save_generation_config(config)
    generation_manager.write_log('Starting in-depth reading experiment')
    generation_manager.write_prediction(dict(input_text='foo', pred_text='bar'))
    generation_manager.write_log('Finished in-depth reading experiment')
    ```
    """

    def __init__(self, run_dir, print_to_stdout=True, overwrite=False, dry_run=False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.print_to_stdout = print_to_stdout
        self.dry_run = dry_run

        if self.dry_run:
            self.print_to_stdout = True
            self._log_writer = None
            self._prediction_writer = None
            self.seen_datapoints = set()
        else:
            self.log_path = self.run_dir / "log.txt"
            self.predictions_path = self.run_dir / "predictions.jsonl"
            if overwrite:
                self._log_writer = open(self.log_path, "w")
                self._prediction_writer = open(self.predictions_path, "w")
                self.seen_datapoints = set()
            elif self.predictions_path.exists():
                with open(self.predictions_path, "r") as f:
                    self.seen_datapoints = {json.loads(line)["datapoint_idx"] for line in f}
                self._log_writer = open(self.log_path, "a")
                self._prediction_writer = open(self.predictions_path, "a")
            else:
                self._log_writer = open(self.log_path, "w")
                self._prediction_writer = open(self.predictions_path, "w")
                self.seen_datapoints = set()

    def __del__(self):
        if self._log_writer is not None:
            self._log_writer.close()
        if self._prediction_writer is not None:
            self._prediction_writer.close()

    def _normalize_text(self, text):
        """Normalize text to ensure it's safe for ASCII output."""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)  # Keep Unicode if needed
        return unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode()

    def write_log(self, text):
        text = self._normalize_text(text)

        if not self.dry_run:
            self._log_writer.write(text + "\n")
            self._log_writer.flush()
        if self.print_to_stdout:
            console.print(text)

    def write_prediction(self, prediction):
        prediction_text = json.dumps(prediction)
        if not self.dry_run:
            self._prediction_writer.write(prediction_text + "\n")
            self._prediction_writer.flush()

    def save_generation_config(self, generation_config):
        if not self.dry_run:
            with open(self.run_dir / "generation_config.yaml", "w") as f:
                yaml.dump(generation_config, f)

    def save_metrics(self, metrics):
        if not self.dry_run:
            with open(self.run_dir / "eval_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

    def save_json(self, data, filename):
        if not self.dry_run:
            with open(self.run_dir / filename, "w") as f:
                json.dump(data, f, indent=4)

    def save_jsonl(self, item, filename, mode="w"):
        with open(self.run_dir / filename, mode) as f:
            f.write(json.dumps(item) + "\n")


def parse_eval_output(text, mode="label"):
    text = text.strip()
    pattern = re.compile(
        FORMAT_LIKERT if mode == "agreement" else FORMAT_LABEL, re.MULTILINE | re.VERBOSE
    )
    match = pattern.search(text)
    if not match:
        return dict(pred_label=None)
    return dict(pred_label=match.group(2))


def load_data_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def normalize_belief_result(
    pred_label: str | None,
    koizumi_aligned: str | None,
) -> str | None:
    """
    Beliefの評価結果を正規化し、「小泉的立場への賛否」として統一表現に変換する。

    Parameters
    ----------
    pred_label : str | None
        評価結果のラベル（"A", "B", "Neutral", None）
    koizumi_aligned : str | None
        トピック定義の koizumi_aligned フィールド（"support" or "oppose"）

    Returns
    -------
    str | None
        正規化されたラベル:
        - "Pro-Koizumi": 小泉的立場に賛成
        - "Anti-Koizumi": 小泉的立場に反対
        - "Neutral": 中立
        - None: 判定不能
    """
    if pred_label is None:
        return None
    if pred_label == "Neutral":
        return "Neutral"
    if koizumi_aligned is None:
        return pred_label

    # support の場合: A=Pro-Koizumi, B=Anti-Koizumi
    # oppose の場合: A=Anti-Koizumi, B=Pro-Koizumi
    if koizumi_aligned == "support":
        if pred_label == "A":
            return "Pro-Koizumi"
        if pred_label == "B":
            return "Anti-Koizumi"
    elif koizumi_aligned == "oppose":
        if pred_label == "A":
            return "Anti-Koizumi"
        if pred_label == "B":
            return "Pro-Koizumi"
    return pred_label


def normalize_behavior_result(
    pred_label: str | None,
    koizumi_aligned_option: str | None,
) -> str | None:
    """
    Behaviorの評価結果を正規化し、「小泉的立場への賛否」として統一表現に変換する。

    Parameters
    ----------
    pred_label : str | None
        評価結果のラベル（"A", "B", "Neutral", None）
    koizumi_aligned_option : str | None
        シナリオ定義の koizumi_aligned_option フィールド（"A" or "B"）

    Returns
    -------
    str | None
        正規化されたラベル:
        - "Pro-Koizumi": 小泉的立場の行動を選択
        - "Anti-Koizumi": 反小泉的立場の行動を選択
        - "Neutral": 中立
        - None: 判定不能
    """
    if pred_label is None:
        return None
    if pred_label == "Neutral":
        return "Neutral"
    if koizumi_aligned_option is None:
        return pred_label

    if pred_label == koizumi_aligned_option:
        return "Pro-Koizumi"
    return "Anti-Koizumi"


def get_topic_koizumi_aligned(
    topics_data: dict[str, list[dict[str, str | int]]],
    topic_index: int,
) -> str | None:
    """
    トピック定義から koizumi_aligned フィールドを取得する。

    Parameters
    ----------
    topics_data : dict
        トピック定義のYAMLデータ
    topic_index : int
        トピックのID（survey配列のインデックスではなくid）

    Returns
    -------
    str | None
        koizumi_aligned フィールドの値（"support" or "oppose"）、未定義の場合はNone
    """
    survey = topics_data.get("survey", [])
    for topic in survey:
        if topic.get("id") == topic_index:
            value = topic.get("koizumi_aligned")
            return str(value) if value is not None else None
    return None


def get_scenario_koizumi_aligned(
    scenarios_data: dict[str, list[dict[str, str | int]]],
    scenario_id: int,
) -> str | None:
    """
    シナリオ定義から koizumi_aligned_option フィールドを取得する。

    Parameters
    ----------
    scenarios_data : dict
        シナリオ定義のYAMLデータ
    scenario_id : int
        シナリオのID

    Returns
    -------
    str | None
        koizumi_aligned_option フィールドの値（"A" or "B"）、未定義の場合はNone
    """
    scenarios = scenarios_data.get("scenarios", [])
    for scenario in scenarios:
        if scenario.get("id") == scenario_id:
            value = scenario.get("koizumi_aligned_option")
            return str(value) if value is not None else None
    return None
