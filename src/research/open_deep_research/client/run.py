"""
python -m client.run --mode conservative --model_name gpt-5
"""

import os
import sys
import json
import yaml
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from langgraph_sdk import get_client  # async client
from fire import Fire
from rich import print


# -------------------------
# Config (env-overridable)
# -------------------------
WORKDIR = os.getcwd()
BASE_URL = os.getenv("ODR_BASE_URL", "http://127.0.0.1:2024")
AUTH_TOKEN = os.getenv("ODR_AUTH_TOKEN", "dev")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

TOPIC_PATH = f"{WORKDIR}/data/study/topics.yaml"
OUTDIR = Path(f"{WORKDIR}/experiments/research")

conservative_study_topic_indices = [0, 1, 2, 3, 4, 5, 6]
progressive_study_topic_indices = [0, 1, 2, 3, 4, 5, 6]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -------------------------
# Streaming (minimal)
# -------------------------
async def stream_run_events(client, thread_id: str, run_id: str, outfile: Path) -> None:
    """Stream events live for a run and append to JSONL."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("")
    print(f"Streaming events thread={thread_id} run={run_id} → {outfile}")
    with outfile.open("a", encoding="utf-8") as fh:
        try:
            async for ev in client.runs.join_stream(thread_id, run_id):
                row = {"ts": utc_now_iso(), "thread_id": thread_id, "run_id": run_id, "event": ev}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                # tiny heartbeat on stdout
                etype = ev.get("type") if isinstance(ev, dict) else None
                eid = ev.get("id") if isinstance(ev, dict) else None
                print(f"Event: type={etype or '?'} | id={eid or '?'}")
        except asyncio.CancelledError:
            print(f"Stream cancelled for run {run_id}")
            raise
        except Exception as e:
            print(f"Stream error for run {run_id}: {e}", file=sys.stderr)
    print(f"Stream ended for run {run_id}")


# -------------------------
# Topic loading (your logic)
# -------------------------
def load_topics(path: str):
    with open(path, "r") as f:
        topics = yaml.safe_load(f)
    study = topics["study"]
    conservative = {
        t["id"]: t for t in study["conservative"] if t["id"] in conservative_study_topic_indices
    }
    progressive = {
        t["id"]: t for t in study["progressive"] if t["id"] in progressive_study_topic_indices
    }
    return conservative, progressive


# -------------------------
# Create thread + run (async)
# -------------------------
async def create_thread_and_run(client, prompt: str, model_name: str) -> Dict[str, str]:
    # 1) create thread
    thread = await client.threads.create()
    thread_id = thread.get("thread_id") or thread.get("id")
    # 2) start run
    run = await client.runs.create(
        thread_id,
        ASSISTANT_ID,
        input={
            "messages": [{"role": "human", "content": prompt}],
            "config": {"model": model_name},
        },
    )
    run_id = run.get("run_id") or run.get("id")
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "assistant_id": ASSISTANT_ID,
        "base_url": BASE_URL,
        "model_name": model_name,
    }


# -------------------------
# Orchestrate
# -------------------------
async def amain(mode, model_name, seed):
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    headers["x-tavily-api-key"] = TAVILY_API_KEY

    client = get_client(url=BASE_URL, headers=headers)

    if mode == "conservative":
        topics, _ = load_topics(TOPIC_PATH)
    elif mode == "progressive":
        _, topics = load_topics(TOPIC_PATH)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    stream_tasks: List[asyncio.Task] = []

    # For each topic: create run, immediately start streaming it.
    for topic_index, topic in topics.items():
        prompt = (
            f'Research about this book/document: "{topic["title"]}" by {topic["name"]}.\n'
            f"Find sources for relevant information on the web, extract the entire content from each source to inlcude in-context without any truncation, and finally summarize the key points of all the above texts.\n"
        )
        run_obj = await create_thread_and_run(client, prompt, model_name)
        run_obj["study_topic_type"] = mode
        run_obj["study_topic_index"] = topic_index
        run_obj["study_topic"] = topic

        # one file per run
        run_dir = OUTDIR / f"m={model_name}_studytype={mode}_studyidx={topic_index}_seed={seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        outfile = run_dir / "raw_events.jsonl"
        metadata_file = run_dir / "generation_config.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(run_obj, f)

        # start streaming immediately
        task = asyncio.create_task(
            stream_run_events(client, run_obj["thread_id"], run_obj["run_id"], outfile)
        )
        stream_tasks.append(task)

    # Wait for all streams to finish (i.e., runs complete)
    if stream_tasks:
        print(f"Awaiting {len(stream_tasks)} stream(s) to complete…")
        await asyncio.gather(*stream_tasks, return_exceptions=False)
    else:
        print("No topics/runs started; nothing to stream.")


def main(mode, model_name, seed=0):
    asyncio.run(amain(mode, model_name, seed))


if __name__ == "__main__":
    Fire(main)
