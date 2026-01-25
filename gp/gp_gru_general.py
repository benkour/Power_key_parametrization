import gpytorch
import torch
import torch.nn as nn
from config import *

class GRUModel(nn.Module):
    """
    GRU feature extractor.
    Input:  (B, T, F)
    Output: (B, D) embedding (D = hidden_size * num_directions)
    """
    def __init__(self, input_size=29, hidden_size=64, num_layers=2, bidirectional=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.embedding_dim = hidden_size * self.num_directions

        self.proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, h_n = self.gru(x)

        if self.bidirectional:
            # last layer forward/backward hidden states
            h_forward = h_n[-2]  # (B, H)
            h_backward = h_n[-1] # (B, H)
            emb = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2H)
        else:
            emb = h_n[-1]  # (B, H)

        return self.proj(emb)  # (B, D)


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard Exact GP model used on top of embeddings.
    """
    def __init__(self, train_x, train_y, likelihood, feature_dim: int):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # Safe, standard kernel choice for regression
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class GRUGP(gpytorch.Module):
    """
    GRU feature extractor + ExactGP
    - Updates GP train_data every forward pass during training
      because GRU features change as GRU weights update.
    """
    def __init__(self, feature_extractor: nn.Module, likelihood, train_x, train_y, device=None):
        super().__init__()

        self.device = device if device is not None else train_x.device
        self.feature_extractor = feature_extractor.to(self.device)
        self.likelihood = likelihood.to(self.device)

        self.train_x = train_x.to(self.device)
        self.train_y = train_y.to(self.device).squeeze(-1)

        # Initial train features
        with torch.no_grad():
            init_features = self.feature_extractor(self.train_x) # (N,D)

        # GP model initialized on extracted features
        self.gp = ExactGPModel(
            train_x=init_features,
            train_y=self.train_y,
            likelihood=self.likelihood,
            feature_dim=init_features.shape[-1]
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        if self.training:
            # Recompute training features because GRU changes during training
            train_features = self.feature_extractor(self.train_x) # (N,D)
            self.gp.set_train_data(inputs=train_features, targets=self.train_y, strict=False)

        x_features = self.feature_extractor(x)
        return self.gp(x_features)
