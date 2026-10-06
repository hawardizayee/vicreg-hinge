import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeStdLoss(torch.nn.Module):
    def __init__(self, std_margin: float = 1.0, eps: float = 1e-4):
        """
        Encourages each feature to maintain at least a minimum standard devication.
        Features with std below the margin incur a penalty of (std_margin =std).
        Args:
            std_margin (float,default=1.0):
                Minimum desired standard deviation per feature.
            eps (float, default=1e-4):
                Constant added to the variance inside the square root.
        """
        super().__init__()
        self.std_margin = std_margin
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class HingeVarLoss(torch.nn.Module):
    def __init__(self, var_margin: float = 1.0):
        """
        Encourages each feature to maintain at least a minimum variance.
        Features with var below the margin incur a penalty of (var_margin - var).
        Args:
            var_margin (float, default=1.0):
                Minimum desired variance per feature.
        """
        super().__init__()
        self.var_margin = var_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            var_loss: Scalar tensor loss on variances
        """
        var = x.var(dim=0)
        var_loss = torch.mean(F.relu(self.var_margin - var))
        return var_loss


HINGES = {"std": HingeStdLoss, "var": HingeVarLoss}


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelations.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D,D]
        # calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std or var hinge), and covariance terms."""

    def __init__(self, hinge="std", hinge_coeff=1.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        if hinge not in HINGES:
            raise ValueError(f"loss.hinge must be one of {list(HINGES)}, got {hinge!r}")
        self.hinge_coeff = hinge_coeff
        self.cov_coeff = cov_coeff
        # eps only exists in the std hinge; the var hinge has no square root
        hinge_kwargs = {"eps": eps} if hinge == "std" else {}
        self.hinge_loss_fn = HINGES[hinge](1.0, **hinge_kwargs)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """compute VICReg loss.

        Agrs:
            z1: [B,D] - First projection tensor
            z2: [B,D] - Second projection tensor

        Returns:
            dict with keys: loss,invariance_loss, var_loss, cov_loss
        """

        # Invarinace loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.hinge_loss_fn(z1) + self.hinge_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.hinge_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }
