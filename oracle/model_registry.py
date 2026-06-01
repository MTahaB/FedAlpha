from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def compute_model_hash(model_state: dict[str, Any]) -> str:
    payload = json.dumps(_to_serializable(model_state), sort_keys=True, separators=(",", ":"))
    return "0x" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_model_hash(model_state: dict[str, Any], output_path: str | Path) -> str:
    model_hash = compute_model_hash(model_state)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_hash, encoding="utf-8")
    return model_hash
