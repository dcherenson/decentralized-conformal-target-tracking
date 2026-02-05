"""Conformal prediction module for calibration-based uncertainty."""

from __future__ import annotations

import numpy as np


class ConformalPredictionModule:
    """Basic split conformal prediction for regression.

    Stores calibration scores and produces symmetric prediction intervals.
    """

    def __init__(self, module_id: int, weights: np.ndarray, alpha: float = 0.1):
        self.id = module_id
        self.residuals: np.ndarray | None = None
        self.alpha: float = alpha  # Default miscoverage level
        # For distributed subgradient method
        self.dist_quantile_estimate: float = 0.0
        self.step_size_base: float = 0.1
        self.step_count: int = 0
        self.weights: np.ndarray = weights

    def calibrate(self, scores: np.ndarray) -> float:
        """Store calibration scores and return conformal quantile."""
        scores = np.array(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array")
        if scores.size == 0:
            raise ValueError("scores must be non-empty")
        self.residuals = scores
        return self.quantile(self.alpha)

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
    
    def initialize_distributed_subgradient(self, initial_quantile: float):
        """ Initialize distributed subgradient  across multiple agents. """
        self.dist_quantile_estimate = initial_quantile


    def run_distributed_subgradient_step(self, step: int, values: np.ndarray, data_per_agent : float):
        """ Run distributed subgradient step across multiple agents. """
        current_step_size = self.step_size_base #/ np.sqrt(step+ 1)
        averaged_estimate = sum(weight * value for weight, value in zip(self.weights, values))
        gradient = self.compute_subgradient_pinball_loss(
            self.dist_quantile_estimate,
            self.residuals,
            (1 - self.alpha) * (1 + 1 / data_per_agent),
        )
        self.dist_quantile_estimate = averaged_estimate - current_step_size * gradient
        

    def compute_subgradient_pinball_loss(self, quantile_estimate: float, residuals: np.ndarray, gamma: float) -> float:
        """ Compute subgradient of pinball loss for quantile estimation. """
        if residuals is None or residuals.size == 0:
            raise ValueError("Module is not calibrated")
        n = residuals.size
        subgradient = 0.0
        for r in residuals:
            if r < quantile_estimate:
                subgradient += (1 - gamma)
            else:
                subgradient += -gamma
        subgradient /= n
        return subgradient

    def __repr__(self) -> str:
        n = 0 if self.residuals is None else self.residuals.size
        return f"ConformalPredictionModule(id={self.id}, n_calib={n})"
