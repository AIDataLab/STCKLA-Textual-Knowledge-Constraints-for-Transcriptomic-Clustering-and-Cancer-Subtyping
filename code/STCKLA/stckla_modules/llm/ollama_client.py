"""llm.py definitions moved here without algorithm changes."""

import argparse
import json
import re
import urllib.request
from typing import Any, Dict, Optional


def call_ollama(prompt: str, args: argparse.Namespace, *, force_json: bool = False) -> Dict[str, Any]:
    host = str(args.ollama_host).strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": str(args.qwen_model_name),
        "prompt": prompt,
        "system": "You are a JSON-only engine. Output exactly one valid JSON object. No markdown. No explanation. No <think>.",
        "stream": False,
        "options": {
            "temperature": float(args.temperature),
            "num_predict": int(args.num_predict),
            "num_ctx": int(getattr(args, "num_ctx", 32768)),
        },
    }
    if force_json:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(args.qwen_timeout_sec)) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
        response = str(obj.get("response", "") or "")
        thinking = str(obj.get("thinking", "") or "")
        # Prefer response; fall back to thinking and record used_field.
        raw_text = response.strip() if response.strip() else thinking.strip()
        used = "response" if response.strip() else "thinking"
        return {
            "ok": True,
            "raw_text": raw_text,
            "api_response": obj,
            "used_field": used,
            "response_len": len(response),
            "thinking_len": len(thinking),
            "force_json": bool(force_json),
            "error": "",
        }
    except Exception as e:
        return {"ok": False, "raw_text": "", "api_response": {}, "error": repr(e)}


def strip_thinking_and_fences(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    text = text.strip()
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_json_loads_relaxed(candidate: str) -> Optional[dict]:
    candidate = str(candidate or "").strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            return _try_json_loads_relaxed(obj)
    except Exception:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_first_json_object(text: str) -> Optional[dict]:
    text = strip_thinking_and_fences(text)
    direct = _try_json_loads_relaxed(text)
    if isinstance(direct, dict):
        return direct
    starts = [m.start() for m in re.finditer(r"\{", text)]
    candidates = []
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    candidates.sort(key=lambda c: ("subtype_marker_programs" not in c and "programs" not in c, -len(c)))
    for cand in candidates:
        obj = _try_json_loads_relaxed(cand)
        if isinstance(obj, dict):
            return obj
    return None


def repair_marker_json(raw_text: str, args: argparse.Namespace) -> Dict[str, Any]:
    repair_prompt = f"""
/no_think
Convert the following model output into exactly one valid JSON object.
No markdown. No explanation. No <think>.
The required top-level keys are subtype_marker_programs and notes.
Each subtype_marker_program must contain: class_id, class_name, selected_up_genes, selected_down_genes, rejected_genes, reason.
If the original output uses genes/core_genes/up_genes/markers, map them to selected_up_genes.
If it uses target_class_id, map it to class_id.
Preserve gene symbols exactly.

OUTPUT_TO_REPAIR:
{raw_text[:16000]}
""".strip()
    # Request JSON format when repairing model output.
    return call_ollama(repair_prompt, args, force_json=True)


def extract_marker_json_object(text: str) -> Optional[dict]:
    """Parse model output; accept either a dict or a top-level list of programs."""
    obj = extract_first_json_object(text)
    if isinstance(obj, dict):
        return obj
    stripped = strip_thinking_and_fences(text)
    try:
        arr = json.loads(stripped)
        if isinstance(arr, list):
            return {"subtype_marker_programs": arr, "notes": ["top_level_list_wrapped"]}
    except Exception:
        pass
    # Try extracting first JSON array.
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(stripped[start:end+1])
            if isinstance(arr, list):
                return {"subtype_marker_programs": arr, "notes": ["embedded_list_wrapped"]}
        except Exception:
            pass
    return None
