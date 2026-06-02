from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NODE_DIR = ROOT / ".tools" / "node"
LOCAL_DOCKER_CANDIDATES = [
    Path.home() / "AppData" / "Local" / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe",
    Path.home() / "AppData" / "Local" / "Programs" / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
    Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe"),
]


def _local_tool(command: str) -> Path | None:
    candidates = {
        "node": LOCAL_NODE_DIR / "node.exe",
        "npm": LOCAL_NODE_DIR / "npm.cmd",
    }
    if command == "docker":
        for candidate in LOCAL_DOCKER_CANDIDATES:
            if candidate.exists():
                return candidate
    path = candidates.get(command)
    if path and path.exists():
        return path
    return None


@dataclass(frozen=True)
class CheckResult:
    area: str
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _run(command: list[str], cwd: Path = ROOT, timeout: int = 60) -> CheckResult:
    label = " ".join(command)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult("tooling", label, "blocked", f"{command[0]} is not installed or not in PATH.")
    except PermissionError as exc:
        return CheckResult("tooling", label, "blocked", f"Permission denied: {exc}")
    except subprocess.TimeoutExpired:
        return CheckResult("tooling", label, "fail", f"Timed out after {timeout}s.")

    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    detail = output[-1] if output else f"exit={completed.returncode}"
    status = "pass" if completed.returncode == 0 else "fail"
    return CheckResult("tooling", label, status, detail[:300])


def command_check(command: str, area: str) -> CheckResult:
    local = _local_tool(command)
    path = str(local) if local else shutil.which(command)
    if path is None:
        return CheckResult(area, command, "blocked", f"{command} not found in PATH.")

    result = _run([path, "--version"], timeout=20)
    if result.status == "pass":
        return CheckResult(area, command, "pass", result.detail or path)
    if result.status == "blocked":
        return CheckResult(area, command, "blocked", result.detail)
    return CheckResult(area, command, "fail", result.detail)


def python_import_check(module: str, area: str = "python") -> CheckResult:
    try:
        __import__(module)
    except Exception as exc:
        return CheckResult(area, module, "fail", f"{type(exc).__name__}: {exc}")
    return CheckResult(area, module, "pass", "import ok")


def python_test_check() -> CheckResult:
    basetemp = ROOT / f".pytest_tmp_verify_{os.getpid()}"
    result = _run([sys.executable, "-m", "pytest", f"--basetemp={basetemp}"], timeout=120)
    return CheckResult("python", "pytest", result.status, result.detail)


def compileall_check() -> CheckResult:
    modules = ["quant", "federated_learning", "oracle", "data", "dashboard", "tests", "clients", "verification"]
    result = _run([sys.executable, "-m", "compileall", *modules], timeout=120)
    return CheckResult("python", "compileall", result.status, result.detail)


def oracle_health_check() -> CheckResult:
    try:
        from fastapi.testclient import TestClient

        from oracle.validation_api import app

        response = TestClient(app).get("/health")
        if response.status_code == 200 and response.json() == {"status": "ok"}:
            return CheckResult("oracle", "/health", "pass", "status ok")
        return CheckResult("oracle", "/health", "fail", f"{response.status_code}: {response.text[:200]}")
    except Exception as exc:
        return CheckResult("oracle", "/health", "fail", f"{type(exc).__name__}: {exc}")


def blockchain_file_check() -> CheckResult:
    required = [
        "blockchain/contracts/FedAlphaGovernance.sol",
        "blockchain/contracts/FedAlphaDAO.sol",
        "blockchain/contracts/RewardManager.sol",
        "blockchain/hardhat.config.js",
        "blockchain/package.json",
        "blockchain/tests/governance.test.js",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        return CheckResult("blockchain", "source files", "fail", f"missing: {', '.join(missing)}")
    return CheckResult("blockchain", "source files", "pass", f"{len(required)} files present")


def blockchain_static_check() -> CheckResult:
    governance = (ROOT / "blockchain/contracts/FedAlphaGovernance.sol").read_text(encoding="utf-8")
    dao = (ROOT / "blockchain/contracts/FedAlphaDAO.sol").read_text(encoding="utf-8")
    required_tokens = [
        "function stake()",
        "function slash(",
        "function distributeReward(",
        "event ParticipantSlashed",
        "event RewardDistributed",
        "modifier onlyOracle()",
        "function propose(",
        "function vote(",
        "function execute(",
    ]
    haystack = governance + "\n" + dao
    missing = [token for token in required_tokens if token not in haystack]
    if missing:
        return CheckResult("blockchain", "contract surface", "fail", f"missing: {', '.join(missing)}")
    return CheckResult("blockchain", "contract surface", "pass", "staking, slashing, rewards, DAO votes present")


def blockchain_solidity_compile_check() -> CheckResult:
    try:
        from solcx import compile_files, get_installed_solc_versions
    except Exception as exc:
        return CheckResult("blockchain", "solidity compile", "blocked", f"py-solc-x unavailable: {exc}")

    versions = {str(version) for version in get_installed_solc_versions()}
    if "0.8.24" not in versions:
        return CheckResult("blockchain", "solidity compile", "blocked", "solc 0.8.24 is not installed")

    try:
        compiled = compile_files(
            [
                str(ROOT / "blockchain/contracts/FedAlphaGovernance.sol"),
                str(ROOT / "blockchain/contracts/FedAlphaDAO.sol"),
                str(ROOT / "blockchain/contracts/RewardManager.sol"),
            ],
            solc_version="0.8.24",
        )
    except Exception as exc:
        return CheckResult("blockchain", "solidity compile", "fail", f"{type(exc).__name__}: {exc}")

    return CheckResult("blockchain", "solidity compile", "pass", f"{len(compiled)} contracts compiled")


def blockchain_hardhat_check() -> CheckResult:
    npm = _local_tool("npm") or shutil.which("npm")
    if npm is None:
        return CheckResult("blockchain", "hardhat tests", "blocked", "npm not found in PATH")
    node_modules = ROOT / "blockchain/node_modules"
    if not node_modules.exists():
        return CheckResult("blockchain", "hardhat tests", "blocked", "run `npm install` in blockchain/")
    env_path = f"{LOCAL_NODE_DIR};"
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"$env:PATH = '{env_path}' + $env:PATH; & '{npm}' test",
    ]
    result = _run(command, cwd=ROOT / "blockchain", timeout=180)
    return CheckResult("blockchain", "hardhat tests", result.status, result.detail)


def docker_static_check() -> CheckResult:
    compose = ROOT / "docker-compose.yml"
    if not compose.exists():
        return CheckResult("docker", "compose file", "fail", "docker-compose.yml missing")
    text = compose.read_text(encoding="utf-8")
    required = [
        "fl_server:",
        "institution_a:",
        "institution_b:",
        "institution_c:",
        "oracle_api:",
        "dashboard:",
        "blockchain_node:",
        "mlflow:",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        return CheckResult("docker", "compose file", "fail", f"missing services: {', '.join(missing)}")

    dockerfiles = [
        "server/Dockerfile",
        "oracle/Dockerfile",
        "dashboard/Dockerfile",
        "blockchain/Dockerfile.ganache",
        "clients/institution_a/Dockerfile",
        "clients/institution_b/Dockerfile",
        "clients/institution_c/Dockerfile",
        "mlflow_server/Dockerfile",
    ]
    missing_files = [path for path in dockerfiles if not (ROOT / path).exists()]
    if missing_files:
        return CheckResult("docker", "compose file", "fail", f"missing Dockerfiles: {', '.join(missing_files)}")

    return CheckResult("docker", "compose file", "pass", "8 services and 8 Dockerfiles present")


def docker_runtime_check() -> CheckResult:
    docker = _local_tool("docker") or shutil.which("docker")
    if docker is None:
        return CheckResult("docker", "docker runtime", "blocked", "docker not found in PATH")
    result = _run([str(docker), "compose", "config"], timeout=120)
    return CheckResult("docker", "docker compose config", result.status, result.detail)


def data_smoke_check() -> CheckResult:
    path = ROOT / "data/raw/test_smoke/ohlcv.csv"
    if not path.exists():
        return CheckResult("data", "yfinance smoke file", "blocked", "data/raw/test_smoke/ohlcv.csv not present")
    try:
        from quant.data_loader import load_ohlcv_csv

        frame = load_ohlcv_csv(path)
    except Exception as exc:
        return CheckResult("data", "yfinance smoke file", "fail", f"{type(exc).__name__}: {exc}")
    return CheckResult("data", "yfinance smoke file", "pass", f"{len(frame)} rows loaded")


def collect_status(run_pytest: bool = False, run_docker: bool = False, run_hardhat: bool = False) -> list[CheckResult]:
    checks = [
        command_check("python", "tooling"),
        command_check("node", "tooling"),
        command_check("npm", "tooling"),
        command_check("docker", "tooling"),
        python_import_check("numpy"),
        python_import_check("pandas"),
        python_import_check("fastapi", "oracle"),
        python_import_check("streamlit", "dashboard"),
        compileall_check(),
        oracle_health_check(),
        data_smoke_check(),
        blockchain_file_check(),
        blockchain_static_check(),
        blockchain_solidity_compile_check(),
        docker_static_check(),
        docker_runtime_check() if run_docker else CheckResult("docker", "docker compose config", "blocked", "not requested"),
        blockchain_hardhat_check() if run_hardhat else CheckResult("blockchain", "hardhat tests", "blocked", "not requested"),
    ]
    if run_pytest:
        checks.append(python_test_check())
    return checks


def status_counts(results: list[CheckResult]) -> dict[str, int]:
    counts = {"pass": 0, "fail": 0, "blocked": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def results_to_json(results: list[CheckResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)
