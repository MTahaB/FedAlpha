import numpy as np

from federated_learning.aggregation.byzantine import simulate_byzantine_updates
from federated_learning.aggregation.fedavg import fedavg
from federated_learning.aggregation.fednova import fednova
from federated_learning.aggregation.krum import krum_layers
from federated_learning.aggregation.median import median_layers
from federated_learning.aggregation.secure_agg import simulate_secure_sum, topk_sparsify
from federated_learning.aggregation.trimmed_mean import trimmed_mean_aggregate, trimmed_mean_layers


def test_fedavg_weighted_average():
    result = fedavg([np.array([1.0, 3.0]), np.array([3.0, 5.0])], weights=[1, 3])[0]
    np.testing.assert_allclose(result, np.array([2.5, 4.5]))


def test_trimmed_mean_reduces_outlier():
    updates = [np.array([1.0]), np.array([1.1]), np.array([0.9]), np.array([100.0]), np.array([1.0])]
    result = trimmed_mean_aggregate(updates, trim_ratio=0.2)
    assert result[0] < 2.0


def test_median_layers_rejects_sign_flip_outlier():
    updates = [
        [np.array([1.0, 1.1]), np.array([0.2])],
        [np.array([1.1, 1.0]), np.array([0.3])],
        [np.array([0.9, 1.2]), np.array([0.25])],
        [np.array([-50.0, -50.0]), np.array([-20.0])],
        [np.array([1.0, 0.95]), np.array([0.22])],
    ]
    result = median_layers(updates)
    np.testing.assert_allclose(result[0], np.array([1.0, 1.0]))
    np.testing.assert_allclose(result[1], np.array([0.22]))


def test_krum_layers_selects_honest_update():
    updates = [
        [np.array([1.0, 1.0]), np.array([0.1])],
        [np.array([1.1, 0.9]), np.array([0.12])],
        [np.array([0.95, 1.05]), np.array([0.11])],
        [np.array([1.05, 1.0]), np.array([0.09])],
        [np.array([50.0, 50.0]), np.array([10.0])],
    ]
    selected = krum_layers(updates, byzantine_clients=1)
    assert selected[0][0] < 2.0


def test_trimmed_mean_layers_handles_multilayer_updates():
    updates = [
        [np.array([1.0]), np.array([2.0])],
        [np.array([1.2]), np.array([2.2])],
        [np.array([0.8]), np.array([1.8])],
        [np.array([100.0]), np.array([100.0])],
        [np.array([1.1]), np.array([2.1])],
    ]
    result = trimmed_mean_layers(updates, trim_ratio=0.2)
    assert result[0][0] < 2.0
    assert result[1][0] < 3.0


def test_fednova_normalizes_by_local_steps():
    global_params = [np.array([0.0])]
    client_params = [[np.array([2.0])], [np.array([4.0])]]
    result = fednova(global_params, client_params, num_examples=[1, 1], local_steps=[1, 2])
    np.testing.assert_allclose(result[0], np.array([3.0]))


def test_fednova_with_median_defense_rejects_malicious_delta():
    global_params = [np.array([0.0])]
    client_params = [[np.array([1.0])], [np.array([1.1])], [np.array([0.9])], [np.array([100.0])]]
    result = fednova(
        global_params,
        client_params,
        num_examples=[1, 1, 1, 1],
        local_steps=[1, 1, 1, 1],
        robust_aggregator="median",
    )
    assert result[0][0] < 2.0


def test_simulate_byzantine_updates_sign_flip():
    updates = [[np.array([1.0])], [np.array([2.0])]]
    poisoned = simulate_byzantine_updates(updates, malicious_clients=1, attack="sign_flip", scale=5.0)
    np.testing.assert_allclose(poisoned[0][0], np.array([-5.0]))
    np.testing.assert_allclose(poisoned[1][0], np.array([2.0]))


def test_secure_sum_masks_cancel():
    updates = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    secure_sum, _ = simulate_secure_sum(updates, seed=1)
    np.testing.assert_allclose(secure_sum, np.sum(updates, axis=0))


def test_topk_sparsify_keeps_largest_values():
    sparse, mask = topk_sparsify(np.array([1.0, -10.0, 2.0, 0.1]), k=0.25)
    assert mask.sum() == 1
    assert sparse[1] == -10.0
