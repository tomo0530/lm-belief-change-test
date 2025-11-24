import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from rich import print

RUN_DIR = "./experiments/reading"


def normalize_title(title_text):
    if not title_text or not isinstance(title_text, str):
        return "untitled"

    # Start with the original title
    normalized = title_text.strip()

    # Remove or replace problematic characters for filenames
    # Characters that are problematic on various filesystems: / \ : * ? " < > |
    char_replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "",
        "'": "",
        "<": "_",
        ">": "_",
        "|": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "&": "and",
        "%": "percent",
        "#": "number",
        "@": "at",
        "+": "plus",
        "=": "equals",
        ",": "_",
        ";": "_",
        "!": "",
        "~": "_",
        "`": "",
        "^": "_",
    }

    # Apply character replacements
    for old_char, new_char in char_replacements.items():
        normalized = normalized.replace(old_char, new_char)

    # Replace multiple spaces/underscores with single underscore
    normalized = re.sub(r"[\s_]+", "_", normalized)

    # Remove leading/trailing underscores and dots
    normalized = normalized.strip("_.")

    # Ensure it's not empty after cleaning
    if not normalized:
        return "untitled"

    # Limit length to avoid filesystem issues (most filesystems support 255 chars)
    max_length = 200  # Leave some room for file extensions
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("_.")

    # Ensure it doesn't start with a dot (hidden file on Unix systems)
    if normalized.startswith("."):
        normalized = "file_" + normalized[1:]

    return normalized


def read_study_content(run_dir, topic_name_text):
    token_threshold = 0
    run_dir = Path(run_dir)

    topic_to_path = defaultdict(list)
    for topic_dir in run_dir.glob("**/"):
        for file_path in topic_dir.glob("*.txt"):
            topic_to_path[topic_dir.name].append((topic_dir.name, file_path.name, str(file_path)))

    # Normalize the topic name text to handle Unicode normalization issues
    lookup_key_nfc = unicodedata.normalize("NFC", topic_name_text)
    key_map = {unicodedata.normalize("NFC", k): k for k in topic_to_path.keys()}
    topic_name_text = key_map.get(lookup_key_nfc, topic_name_text)

    contents = []
    for topic_name, title_text, file_path in topic_to_path[topic_name_text]:
        with open(file_path, "r") as f:
            text = f.read()
        num_tokens = len(text.split())
        if num_tokens < token_threshold:
            continue
        # print(text)
        print(file_path)
        print(f"{num_tokens} tokens")
        print(topic_name)
        print("#" * 100)
        contents.append((title_text, text))
    if not contents:
        raise ValueError(f"No content found for {topic_name_text}")
    return contents
