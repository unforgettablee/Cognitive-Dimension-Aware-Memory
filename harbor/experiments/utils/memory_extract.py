import os
import json
import pickle
import threading
from filelock import FileLock, Timeout
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.prompts.memory import *

_deepseek_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_deepseek_client() -> OpenAI:
    """Lazily create and cache the DeepSeek client (thread-safe)."""
    global _deepseek_client
    if _deepseek_client is None:
        with _client_lock:
            if _deepseek_client is None:
                _deepseek_client = OpenAI(
                    api_key=os.getenv("API_KEY"),
                    base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
                )
    return _deepseek_client

embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
MEMORY_LOCK_TIMEOUT = 180


def _embed(text: str) -> list[float]:
    return embed_model.encode(text).tolist()


def parse_memory_items(text: str):
    items = []
    blocks = re.split(r'(?=^# Memory Item\s+\d+)', text, flags=re.MULTILINE)

    for block in blocks:
        if not block.strip():
            continue
        m_title = re.search(r'^## Title\s*(.+)', block, flags=re.MULTILINE)
        m_desc = re.search(r'^## Description\s*(.+)', block, flags=re.MULTILINE)
        m_content = re.search(
            r'^## Content\s*((?:.|\n)*?)(?=\n## |\Z)',
            block,
            flags=re.MULTILINE
        )

        if m_title and m_desc:
            title = m_title.group(1).strip()
            description = m_desc.group(1).strip()
            content = m_content.group(1).strip() if m_content else ""

            items.append({
                "title": title.strip(":").strip(),
                "description": description.strip(":").strip(),
                "content": content.strip(":").strip(),
            })

    return items


def _extract_json(text: str) -> str:
    """Robustly extract JSON from LLM output that may wrap it in markdown or add fluff."""
    text = text.strip()
    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try raw { ... } (find outermost braces)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    return text


def _call_llm_with_json(messages: list[dict], max_retries: int = 2) -> dict:
    """Call DeepSeek, extract JSON from response, retry if parse fails."""
    last_error = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages.append({"role": "user", "content": (
                "Your previous output was not valid JSON. "
                "Please output ONLY the JSON object, without markdown wrapping or extra text."
            )})
        response = _get_deepseek_client().chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=messages,
            timeout=300.0,
        )
        raw = response.choices[0].message.content or ""
        try:
            json_str = _extract_json(raw)
            return json.loads(json_str), raw
        except (json.JSONDecodeError, ValueError) as e:
            last_error = (e, raw)
    raise ValueError(f"Failed to parse JSON after {max_retries + 1} attempts. "
                     f"Last error: {last_error[0]}. Raw: {last_error[1][:500]}")


def parse_local_memory_items(text: str):
    return json.loads(_extract_json(text))


def extract_rawtraj_memory(judgement, trajectory, log_dir, task_name, task, commands, benchmark):
    new_memory = {
        "task_name": task_name,
        "task": task,
        "commands": commands,
        "benchmark": benchmark,
        "judgement": judgement,
        "type": "trajectory"
    }
    new_memory["embedding"] = _embed(new_memory["task"])

    try:
        memory_path = f"{log_dir.rsplit('/', 1)[0]}/trajectory_memory.pkl"
        lock_path = memory_path + ".lock"
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(
            f"[{benchmark}] Trajectory Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping trajectory memory update for {log_dir}."
        )
        return
    return


def extract_workflow_memory(judgement, trajectory, log_dir, task_name, task, commands, benchmark):
    prompt = WORKFLOW_CORRECT_EXTRACT_PROMPT if judgement else WORKFLOW_WRONG_EXTRACT_PROMPT
    content = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"### Trajectory:\n{trajectory[1:]}"},
    ]
    try:
        new_memory, memory_text = _call_llm_with_json(content)
    except ValueError as e:
        print(f"[{benchmark}] Failed to parse workflow memory: {e}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Workflow memory not a dict for {log_dir}")
        return
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/workflow_memory.pkl"
    lock_path = memory_path + ".lock"
    new_elem = {
        "benchmark": benchmark,
        "task_name": task_name,
        "llm_judge": judgement,
        "task": task,
        "type": "workflow",
        "workflow": new_memory,
        "key_embedding": _embed(new_memory["goal"]),
    }
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_elem)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(
            f"[{benchmark}] Workflow Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping workflow memory update for {log_dir}."
        )
        return
    return


def extract_traj_memory(judgement, trajectory, log_dir, task_name, task, commands, benchmark):
    system_prompt = CODE_SPECIFIC_CORRECT_PROMPT if judgement else CODE_SPECIFIC_WRONG_PROMPT
    content = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Trajectory:\n{trajectory}"},
    ]
    try:
        new_memory, memory_text = _call_llm_with_json(content)
        new_memory['task_query'] = task
    except ValueError as e:
        print(f"[{benchmark}] Failed to parse local memory: {e}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Local memory not a dict for {log_dir}")
        return
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/local_memory.pkl"
    lock_path = memory_path + ".lock"
    new_memory["generalized_query_embedding"] = _embed(new_memory["generalized_query"])
    new_memory["benchmark"] = benchmark
    new_memory["task_name"] = task_name
    new_memory["commands"] = commands
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            new_text_memory = new_memory.copy()
            new_text_memory.pop("generalized_query_embedding")
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/local_memory.json", "w") as f:
                json.dump({"memory": new_text_memory, "all_memory": [{"when_to_use": x["when_to_use"], "task_query": x["task_query"], "generalized_query": x["generalized_query"], "experience": x["experience"], "tags": x["tags"], "benchmark": x["benchmark"], "task_name": x["task_name"]} for x in all_memory[:-1]]}, f, indent=4)
    except Timeout:
        print(
            f"[{benchmark}] Local Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping local memory update for {log_dir}."
        )
        return


def extract_summary_memory(judgement, trajectory, log_dir, task_name, task, commands, benchmark):
    prompt = SUMMARY_CORRECT_EXTRACT_PROMPT if judgement else SUMMARY_WRONG_EXTRACT_PROMPT
    content = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"### Trajectory:\n{trajectory[1:]}"},
    ]
    try:
        new_memory, memory_text = _call_llm_with_json(content)
    except ValueError as e:
        print(f"[{benchmark}] Failed to parse summary memory: {e}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Summary memory not a dict for {log_dir}")
        return
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/summary_memory_{benchmark}.pkl"
    lock_path = memory_path + ".lock"
    new_memory["embedding"] = _embed(new_memory["task_summary"])
    new_memory["benchmark"] = benchmark
    new_memory["task_name"] = task_name
    new_memory["commands"] = commands
    new_memory["judgement"] = judgement
    new_memory["task"] = task
    new_memory["type"] = "summary"
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(
            f"[{benchmark}] Summary Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping summary memory update for {log_dir}."
        )
        return
    return


def extract_insight_memory(judgement, trajectory, log_dir, task_name, task, commands, benchmark):
    system_prompt = INSIGHT_CORRECT_PROMPT if judgement else INSIGHT_WRONG_PROMPT
    content = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Trajectory:\n{trajectory}"},
    ]
    try:
        new_memory, memory_text = _call_llm_with_json(content)
    except ValueError as e:
        print(f"[{benchmark}] Failed to parse insight memory: {e}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Insight memory not a dict for {log_dir}")
        return
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/insight_memory.pkl"
    lock_path = memory_path + ".lock"

    emb_vec = _embed(new_memory["title"])

    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append({
                "key_embedding": emb_vec,
                "benchmark": benchmark,
                "type": "insight",
                "llm_judge": judgement,
                "task_name": task_name,
                "task": task,
                "insight": new_memory
            })
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
    except Timeout:
        print(
            f"[{benchmark}] Insight Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping insight memory update for {log_dir}."
        )
        return
    return
