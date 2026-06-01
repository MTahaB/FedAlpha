from oracle.model_registry import compute_model_hash


def test_model_hash_is_stable():
    first = compute_model_hash({"b": [2, 3], "a": [1]})
    second = compute_model_hash({"a": [1], "b": [2, 3]})
    assert first == second
    assert first.startswith("0x")
