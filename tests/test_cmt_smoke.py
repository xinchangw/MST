import os
import unittest

import numpy as np


os.environ.setdefault("MST_LEAF_MODEL", "leaf_model_mnl")

from mst import MST
from leaf_model_mnl_tensorflow import _mnl_probabilities_np


def _choice_probabilities(A, weights, num_features):
    n_items = weights.shape[1]
    features = A.reshape((-1, num_features, n_items))
    logits = np.sum(features * weights.reshape((1, num_features, n_items)), axis=1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probas = np.exp(logits)
    return probas / np.sum(probas, axis=1, keepdims=True)


def _make_choice_data(n, rng):
    n_items = 3
    num_features = 2

    x_split = rng.integers(0, 2, size=(n, 1))
    product_feature = rng.normal(size=(n, n_items))
    availability = np.ones((n, n_items))
    A = np.concatenate([availability, product_feature], axis=1)

    left_weights = np.array([[0.1, 0.0, -0.1], [1.5, -0.4, 0.2]])
    right_weights = np.array([[-0.1, 0.2, 0.0], [-0.3, 1.2, 0.4]])
    probas = np.zeros((n, n_items))
    left = x_split[:, 0] == 0
    probas[left] = _choice_probabilities(A[left], left_weights, num_features)
    probas[~left] = _choice_probabilities(A[~left], right_weights, num_features)

    Y = np.array([rng.choice(n_items, p=probas[i]) for i in range(n)], dtype=int)
    return x_split, A, Y


class CMTSmokeTests(unittest.TestCase):
    def test_mnl_probabilities_leave_unavailable_items_at_zero(self):
        input_features = np.array(
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.2, 9.0, -0.1],
                ]
            ],
            dtype=np.float32,
        )
        weights = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

        _, probas = _mnl_probabilities_np(input_features, weights, model_type=0, is_bias=True)

        self.assertEqual(probas.shape, (1, 3))
        self.assertEqual(probas[0, 1], 0.0)
        np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(1), rtol=1e-6)

    def test_small_cmt_fit_predict_runs_with_tf2_leaf_model(self):
        rng = np.random.default_rng(0)
        X, A, Y = _make_choice_data(48, rng)
        XT, AT, _ = _make_choice_data(12, rng)

        tree = MST(max_depth=1, min_weights_per_node=5, quant_discret=0.5)
        tree.fit(
            X,
            A,
            Y,
            verbose=False,
            feats_continuous=[False],
            refit_leaves=True,
            num_features=2,
            is_bias=True,
            model_type=0,
            mode="mnl",
            batch_size=8,
            steps=2,
            steps_refit=3,
        )

        predictions = tree.predict(XT, AT)
        self.assertEqual(predictions.shape, (XT.shape[0], 3))
        self.assertTrue(np.all(np.isfinite(predictions)))
        np.testing.assert_allclose(np.sum(predictions, axis=1), np.ones(XT.shape[0]), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
