import json
import yaml
from pathlib import Path
from rich import print
from fire import Fire


def cleanup_raw_events(run_dir):
    events = []

    config_path = run_dir / "generation_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run_path = run_dir / "raw_events.jsonl"
    with open(run_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "event" not in obj or obj["event"][1] is None or len(obj["event"]) < 2:
                continue
            event = obj["event"][1]
            events.append(event)

    dup_content = set()
    message_history = []
    for name, event in events[-1].items():
        if name in ("messages", "supervisor_messages"):
            for message in event:
                if "content" in message:
                    if message["content"] in dup_content or message["content"] == "":
                        continue
                    dup_content.add(message["content"])

                if name == "messages":
                    if "usage_metadata" in message:
                        continue

                    message_history.append(
                        dict(
                            role="user" if message["type"] == "human" else "assistant",
                            content=message["content"],
                        )
                    )

                elif name == "supervisor_messages":
                    message_history.append(
                        dict(
                            role="assistant",
                            content=message["content"],
                        )
                    )
        else:
            if name == "research_brief":
                message_history.append(
                    dict(
                        role="assistant",
                        content=event,
                    )
                )
            elif name == "final_report":
                message_history.append(
                    dict(
                        role="assistant",
                        content=event,
                    )
                )
    return message_history


def merge_assistant_messages(message_history):
    merged_messages = []
    for message in message_history:
        if message["role"] == "assistant" and merged_messages[-1]["role"] == "assistant":
            merged_messages[-1]["content"] += f"\n\n\n\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages


def main(run_dir):
    run_dir = Path(run_dir)
    message_history = cleanup_raw_events(run_dir)
    message_history = merge_assistant_messages(message_history)

    for message in message_history:
        print(message["role"])
        print(message["content"])
        print("---")

    with open(run_dir / "message_history.json", "w") as f:
        json.dump(message_history, f, indent=4)

    with open(run_dir / "log.txt", "w") as f:
        for message in message_history:
            f.write(f"Role={message['role']}\n")
            f.write(f"{message['content']}\n")
            f.write("---" * 100 + "\n")


if __name__ == "__main__":
    Fire(main)
