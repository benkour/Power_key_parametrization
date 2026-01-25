import torch 
import torch.nn as nn
import gpytorch
import numpy as np

class GRUEmbedder(nn.Module):
    """
    Taking the last forward + last backward hidden states and concatenate
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, h_n = self.gru(x)  # h_n: (num_layers*num_directions, B, hidden_size)

        if self.bidirectional:
            forward = h_n[-2]   # (B, hidden_size)
            backward = h_n[-1]  # (B, hidden_size)
            emb = torch.cat([forward, backward], dim=-1)  # (B, 2*hidden_size)
        else:
            emb = h_n[-1]  # (B, hidden_size)

        return emb


class DKLGPModel(gpytorch.models.ExactGP):
    """
    Deep Kernel Learning:
    GP kernel runs on features produced by a neural net (GRUEmbedder).
    It trains by maximizing ExactMarginalLogLikelihood.
    """
    def __init__(self, train_x, train_y, likelihood, feature_extractor: nn.Module, kernel="matern"):
        super().__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()

        # Determine feature dimension by a forward on 1 sample 
        with torch.no_grad():
            z = self.feature_extractor(train_x[:1])
        feat_dim = z.shape[-1]

        # Kernel on learned features
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feat_dim)

        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        z = self.feature_extractor(x)             # (N, D_feat)
        mean_x = self.mean_module(z)              # (N,)
        covar_x = self.covar_module(z)            # (N,N)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)