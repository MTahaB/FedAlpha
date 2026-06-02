from verification.checks import (
    blockchain_file_check,
    blockchain_static_check,
    collect_status,
    docker_static_check,
)


def test_blockchain_static_checks_pass():
    assert blockchain_file_check().status == "pass"
    assert blockchain_static_check().status == "pass"


def test_docker_compose_static_check_passes():
    assert docker_static_check().status == "pass"


def test_collect_status_returns_results():
    results = collect_status()
    assert results
    assert all(result.status in {"pass", "fail", "blocked"} for result in results)
