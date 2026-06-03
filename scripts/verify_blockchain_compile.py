from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile FedAlpha Solidity contracts with py-solc-x.")
    parser.add_argument("--install-solc", action="store_true", help="Install solc 0.8.24 before compiling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from solcx import compile_files, install_solc

    if args.install_solc:
        install_solc("0.8.24")

    compiled = compile_files(
        [
            str(ROOT / "blockchain/contracts/FedAlphaGovernance.sol"),
            str(ROOT / "blockchain/contracts/FedAlphaDAO.sol"),
            str(ROOT / "blockchain/contracts/FedRegistry.sol"),
            str(ROOT / "blockchain/contracts/FedAlphaStaking.sol"),
            str(ROOT / "blockchain/contracts/MultiSigOracle.sol"),
            str(ROOT / "blockchain/contracts/SlashingManager.sol"),
            str(ROOT / "blockchain/contracts/RewardManager.sol"),
        ],
        solc_version="0.8.24",
    )
    for contract_name in sorted(compiled):
        print(contract_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
