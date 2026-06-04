from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationAnchor:
    round_id: int
    model_hash: str
    validation_score: int
    validated: bool
    validation_timestamp: str
    tx_hash: str | None = None
    mode: str = "file"
    audit_label: str = "file-backed audit"


class FileBlockchainClient:
    def __init__(self, events_path: str | Path = "reports/blockchain_events.json"):
        self.events_path = Path(events_path)

    def record_validation(self, anchor: ValidationAnchor) -> dict[str, Any]:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        events = []
        if self.events_path.exists():
            events = json.loads(self.events_path.read_text(encoding="utf-8"))
        events = [_with_audit_label(event) for event in events]
        event = asdict(anchor)
        events.append(event)
        self.events_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
        return event


def _with_audit_label(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("audit_label"):
        return event
    return {
        **event,
        "audit_label": "on-chain transaction" if event.get("tx_hash") else "file-backed audit",
    }


class Web3RegistryClient:
    def __init__(
        self,
        rpc_url: str,
        registry_address: str,
        sender_address: str,
        artifact_path: str | Path = "blockchain/artifacts/contracts/FedRegistry.sol/FedRegistry.json",
    ):
        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError("Install web3 to anchor oracle validations on-chain.") from exc

        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract = self.web3.eth.contract(address=registry_address, abi=artifact["abi"])
        self.sender_address = sender_address

    def verify_round(self, round_id: int, model_hash: str) -> str:
        tx_hash = self.contract.functions.verifyRound(round_id, model_hash).transact({"from": self.sender_address})
        return self.web3.to_hex(tx_hash)


def record_oracle_validation(
    *,
    round_id: int,
    model_hash: str,
    validation_score: int,
    validated: bool,
    validation_timestamp: str,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    report_dir = Path(reports_dir or os.getenv("FEDALPHA_REPORT_DIR", "reports"))
    file_client = FileBlockchainClient(report_dir / "blockchain_events.json")
    anchor = ValidationAnchor(
        round_id=round_id,
        model_hash=model_hash,
        validation_score=int(validation_score),
        validated=bool(validated),
        validation_timestamp=validation_timestamp,
    )
    event = file_client.record_validation(anchor)

    rpc_url = os.getenv("WEB3_PROVIDER_URL")
    registry_address = os.getenv("FED_REGISTRY_ADDRESS")
    sender_address = os.getenv("ORACLE_SENDER_ADDRESS")
    if validated and rpc_url and registry_address and sender_address:
        try:
            tx_hash = Web3RegistryClient(rpc_url, registry_address, sender_address).verify_round(round_id, model_hash)
            event = file_client.record_validation(
                ValidationAnchor(
                    round_id=round_id,
                    model_hash=model_hash,
                    validation_score=int(validation_score),
                    validated=True,
                    validation_timestamp=validation_timestamp,
                    tx_hash=tx_hash,
                    mode="web3",
                    audit_label="on-chain transaction",
                )
            )
        except Exception as exc:
            event["web3_error"] = f"{type(exc).__name__}: {exc}"
    return event
