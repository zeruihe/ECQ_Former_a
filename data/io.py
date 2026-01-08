from __future__ import annotations
import json
from typing import Any, Dict, List

##读json中的数据并放进列表返回
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      1) JSON array: [ {...}, {...} ]
      2) JSON object wrapper: {"data":[...]} or {"annotations":[...]} etc.
      3) JSONL: one JSON per line
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    # JSON array or object
    if content[0] in ["[", "{"]:
        obj = json.loads(content)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # common wrappers
            for key in ["data", "annotations", "items", "samples"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]

    # fallback as JSONL
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
