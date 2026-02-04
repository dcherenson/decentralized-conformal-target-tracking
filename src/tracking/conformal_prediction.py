"""Conformal prediction module for calibration-based uncertainty."""

from __future__ import annotations

import numpy as np


class ConformalPredictionModule:
    """Basic split conformal prediction for regression.

    Stores calibration scores and produces symmetric prediction intervals.
    """

    def __init__(self, module_id: int):
        self.id = module_id
        self.residuals: np.ndarray | None = None

    def calibrate(self, scores: np.ndarray, alpha: float) -> float:
        """Store calibration scores and return conformal quantile."""
        scores = np.array(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array")
        if scores.size == 0:
            raise ValueError("scores must be non-empty")
        self.residuals = scores
        return self.quantile(alpha)

    def compute_mahalanobis_score(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        truth: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance between truth and mean given covariance."""
        mean = np.array(mean, dtype=float)
        truth = np.array(truth, dtype=float)
        covariance = np.array(covariance, dtype=float)

        if mean.shape != truth.shape:
            raise ValueError("mean and truth must have the same shape")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square")
        if covariance.shape[0] != mean.size:
            raise ValueError("covariance dimension must match mean size")

        diff = truth.reshape(-1) - mean.reshape(-1)
        inv_cov = np.linalg.inv(covariance)
        score = float(np.sqrt(diff.T @ inv_cov @ diff))
        return score

    def quantile(self, alpha: float) -> float:
        """Return conformal quantile for miscoverage alpha."""
        if self.residuals is None or self.residuals.size == 0:
            raise ValueError("Module is not calibrated")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        n = self.residuals.size
        q = np.ceil((n + 1) * (1 - alpha)) / n
        return float(np.quantile(self.residuals, q, method="higher"))

    def predict_interval(self, y_pred: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetric prediction interval around y_pred."""
        y_pred = np.array(y_pred, dtype=float)
        radius = self.quantile(alpha)
        lower = y_pred - radius
        upper = y_pred + radius
        return lower, upper

    def __repr__(self) -> str:
        n = 0 if self.residuals is None else self.residuals.size
        return f"ConformalPredictionModule(id={self.id}, n_calib={n})"
