import json
import re
import unicodedata
from pathlib import Path

import yaml
from rich.console import Console

console = Console(markup=False)
FORMAT_LIKERT = (
    r"(?i)[^\w\n]*(the\s+answer\s+is|回答\s*[:：]?)\s*:?\s*([0-9]+)(?:\.)?[^\w\n]*"
)
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
