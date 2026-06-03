from __future__ import annotations

import hashlib

import requests


IPFS_API = "http://ipfs_node:5001/api/v0"
IPFS_GATEWAY = "http://ipfs_node:8080/ipfs"


def pin_model(model_bytes: bytes, ipfs_api: str = IPFS_API) -> str:
    response = requests.post(f"{ipfs_api.rstrip('/')}/add", files={"file": model_bytes}, timeout=30)
    response.raise_for_status()
    return str(response.json()["Hash"])


def model_hash_bytes32(model_bytes: bytes) -> bytes:
    return hashlib.sha256(model_bytes).digest()


def model_hash_hex(model_bytes: bytes) -> str:
    return "0x" + model_hash_bytes32(model_bytes).hex()


def verify_integrity(cid: str, expected_hash: bytes, gateway: str = IPFS_GATEWAY) -> bool:
    response = requests.get(f"{gateway.rstrip('/')}/{cid}", timeout=30)
    response.raise_for_status()
    return model_hash_bytes32(response.content) == expected_hash
