import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=29, hidden_size=64, output_size=11, num_layers=2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Enabling bidirectionality 
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=self.bidirectional)
        # Mapping the 128 hidden features down to 11 parameters which we want to estimate
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size* self.num_directions),
            nn.Linear(hidden_size* self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        forward = h_n[-2,:,:]
        backward = h_n[-1,:,:]
        embedding = torch.cat([forward, backward], dim=-1) 
        # Prediction should be the shape of batch_size  and 11
        return self.mlp(embedding)
    