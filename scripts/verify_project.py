from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.checks import collect_status, results_to_json, status_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedAlpha project verification checks.")
    parser.add_argument("--pytest", action="store_true", help="Run the Python pytest suite.")
    parser.add_argument("--docker", action="store_true", help="Run docker compose config if Docker is available.")
    parser.add_argument("--hardhat", action="store_true", help="Run Hardhat tests if npm/node_modules are available.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = collect_status(run_pytest=args.pytest, run_docker=args.docker, run_hardhat=args.hardhat)
    if args.json:
        print(results_to_json(results))
    else:
        for result in results:
            print(f"[{result.status.upper():7}] {result.area:12} {result.name:24} {result.detail}")
        print(status_counts(results))
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
