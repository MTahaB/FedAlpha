from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def write_ci_status(
    *,
    commit: str,
    status: str,
    reports_dir: Path = Path("reports"),
    details: dict[str, str] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit": commit,
        "status": status,
        "visible_result": f"{status.upper()} for commit {commit}",
        "generated_at": datetime.now(UTC).isoformat(),
        "details": details or {},
    }
    path = reports_dir / "ci_status.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a visible CI status artifact.")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--status", choices=["pass", "fail", "blocked"], required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--detail", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = {}
    for item in args.detail:
        if "=" in item:
            key, value = item.split("=", 1)
            details[key] = value
    path = write_ci_status(
        commit=args.commit,
        status=args.status,
        reports_dir=args.reports_dir,
        details=details,
    )
    print(path)


if __name__ == "__main__":
    main()
