"""Educational NumPy implementation of a random-forest regressor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class _Node:
    """One node in a scratch CART regression tree."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value: float | None = None) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value = value


class RandomForestScratch:
    """Bootstrap-aggregated CART regressors implemented with NumPy.

    This estimator intentionally favors readable project code over the speed and
    breadth of a production implementation such as scikit-learn.
    """

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: float = 0.5,
        max_samples: float = 0.8,
        max_thresholds: int = 64,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.verbose = verbose
        self.trees_: list[_Node] = []
        self.n_features_in_: int | None = None
        self.feature_importances_: np.ndarray | None = None
        self._split_counts: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._validate_hyperparameters()

    def _validate_hyperparameters(self) -> None:
        integer_parameters = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_thresholds": self.max_thresholds,
        }
        for name, value in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "max_features": self.max_features,
            "max_samples": self.max_samples,
        }.items():
            if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
                raise ValueError(f"{name} must be in the interval (0, 1]")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state,
            int,
        ):
            raise ValueError("random_state must be an integer")

    @staticmethod
    def _training_arrays(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional array")
        if target.ndim != 1 or len(target) != len(features):
            raise ValueError("y must be one-dimensional with one value per row")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("X and y must contain only finite numeric values")
        return features, target

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[int | None, float | None, float]:
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        parent_sse = float(np.square(y - y.mean()).sum())
        row_count = len(y)

        for feature in feature_indices:
            column = X[:, feature]
            order = np.argsort(column, kind="mergesort")
            sorted_column = column[order]
            sorted_target = y[order]

            candidates = np.flatnonzero(np.diff(sorted_column) > 1e-12)
            if candidates.size == 0:
                continue
            left_count = candidates + 1
            valid = (left_count >= self.min_samples_leaf) & (
                row_count - left_count >= self.min_samples_leaf
            )
            candidates = candidates[valid]
            if candidates.size == 0:
                continue
            if candidates.size > self.max_thresholds:
                positions = np.linspace(
                    0,
                    candidates.size - 1,
                    num=self.max_thresholds,
                    dtype=int,
                )
                candidates = candidates[np.unique(positions)]

            cumulative_sum = np.cumsum(sorted_target)
            cumulative_square_sum = np.cumsum(np.square(sorted_target))
            left_count = candidates + 1
            right_count = row_count - left_count
            left_sum = cumulative_sum[candidates]
            left_square_sum = cumulative_square_sum[candidates]
            right_sum = cumulative_sum[-1] - left_sum
            right_square_sum = cumulative_square_sum[-1] - left_square_sum
            left_sse = left_square_sum - np.square(left_sum) / left_count
            right_sse = right_square_sum - np.square(right_sum) / right_count
            gains = parent_sse - left_sse - right_sse
            candidate_index = int(np.argmax(gains))
            gain = float(gains[candidate_index])

            if gain > best_gain:
                split_index = int(candidates[candidate_index])
                best_gain = gain
                best_feature = int(feature)
                best_threshold = float(
                    (sorted_column[split_index] + sorted_column[split_index + 1])
                    / 2.0
                )

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> _Node:
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or np.allclose(y, y[0])
        ):
            return _Node(float(y.mean()))

        if self._rng is None or self.n_features_in_ is None:
            raise RuntimeError("Internal estimator state is not initialized")
        feature_count = max(
            1,
            math.ceil(self.max_features * self.n_features_in_),
        )
        feature_indices = self._rng.choice(
            self.n_features_in_,
            size=feature_count,
            replace=False,
        )
        feature, threshold, gain = self._best_split(X, y, feature_indices)
        if feature is None or threshold is None or gain <= 1e-12:
            return _Node(float(y.mean()))

        left_mask = X[:, feature] <= threshold
        if not left_mask.any() or left_mask.all():
            return _Node(float(y.mean()))

        node = _Node()
        node.feature = feature
        node.threshold = threshold
        if self._split_counts is None:
            raise RuntimeError("Internal feature counts are not initialized")
        self._split_counts[feature] += 1.0
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: Any, y: Any) -> "RandomForestScratch":
        """Fit independent bootstrap CART trees and return this estimator."""
        self._validate_hyperparameters()
        features, target = self._training_arrays(X, y)
        self.n_features_in_ = features.shape[1]
        self.trees_ = []
        self._split_counts = np.zeros(self.n_features_in_, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        sample_count = max(2 * self.min_samples_leaf, math.ceil(self.max_samples * len(features)))
        sample_count = min(sample_count, len(features))
        report_interval = max(1, self.n_estimators // 5)

        for tree_index in range(self.n_estimators):
            bootstrap_indices = self._rng.integers(
                0,
                len(features),
                size=sample_count,
            )
            tree = self._build_tree(
                features[bootstrap_indices],
                target[bootstrap_indices],
                depth=0,
            )
            self.trees_.append(tree)
            completed = tree_index + 1
            if self.verbose and (
                completed % report_interval == 0
                or completed == self.n_estimators
            ):
                print(f"  trained {completed}/{self.n_estimators} trees")

        total_splits = float(self._split_counts.sum())
        if total_splits:
            self.feature_importances_ = self._split_counts / total_splits
        else:
            self.feature_importances_ = self._split_counts.copy()
        self._rng = None
        return self

    @staticmethod
    def _predict_tree(node: _Node, row: np.ndarray) -> float:
        current = node
        while current.value is None:
            if (
                current.feature is None
                or current.threshold is None
                or current.left is None
                or current.right is None
            ):
                raise RuntimeError("Encountered an invalid fitted tree")
            current = (
                current.left
                if row[current.feature] <= current.threshold
                else current.right
            )
        return current.value

    def predict(self, X: Any) -> np.ndarray:
        """Average predictions from all fitted trees."""
        if not self.trees_ or self.n_features_in_ is None:
            raise RuntimeError("RandomForestScratch must be fitted before prediction")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features for prediction"
            )
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite numeric values")
        tree_predictions = np.empty(
            (len(self.trees_), len(features)),
            dtype=float,
        )
        for tree_index, tree in enumerate(self.trees_):
            tree_predictions[tree_index] = [
                self._predict_tree(tree, row) for row in features
            ]
        return tree_predictions.mean(axis=0)
