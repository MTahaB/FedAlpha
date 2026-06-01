import numpy as np

from federated_learning.aggregation.fedavg import fedavg
from federated_learning.aggregation.secure_agg import simulate_secure_sum, topk_sparsify
from federated_learning.aggregation.trimmed_mean import trimmed_mean_aggregate


def test_fedavg_weighted_average():
    result = fedavg([np.array([1.0, 3.0]), np.array([3.0, 5.0])], weights=[1, 3])[0]
    np.testing.assert_allclose(result, np.array([2.5, 4.5]))


def test_trimmed_mean_reduces_outlier():
    updates = [np.array([1.0]), np.array([1.1]), np.array([0.9]), np.array([100.0]), np.array([1.0])]
    result = trimmed_mean_aggregate(updates, trim_ratio=0.2)
    assert result[0] < 2.0


def test_secure_sum_masks_cancel():
    updates = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    secure_sum, _ = simulate_secure_sum(updates, seed=1)
    np.testing.assert_allclose(secure_sum, np.sum(updates, axis=0))


def test_topk_sparsify_keeps_largest_values():
    sparse, mask = topk_sparsify(np.array([1.0, -10.0, 2.0, 0.1]), k=0.25)
    assert mask.sum() == 1
    assert sparse[1] == -10.0
