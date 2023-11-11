import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.transformer_encoder(x)
        x = self.linear(x)  # (batch_size, n_timesteps, n_classes)

        return x


class CNN1DTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
    ):
        super().__init__()

        self.num_layers = num_layers
        encoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, batch_first=True, dropout=dropout
            )
            for i in range(self.num_layers)
        ]
        conv_layers = [
            nn.Conv1d(
                hidden_size,
                hidden_size,
                (self.num_layers - i) * 2 - 1,
                stride=1,
                padding=0,
            )
            for i in range(self.num_layers)
        ]
        deconv_layers = [
            nn.ConvTranspose1d(
                hidden_size,
                hidden_size,
                (self.num_layers - i) * 2 - 1,
                stride=1,
                padding=0,
            )
            for i in range(self.num_layers)
        ]
        layer_norm_layers = [nn.LayerNorm(hidden_size) for i in range(self.num_layers)]
        layer_norm_layers2 = [nn.LayerNorm(hidden_size) for i in range(self.num_layers)]

        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.permute(0, 2, 1)
        x = self.input_linear(x)
        for i in range(self.num_layers):
            res = x
            x = F.relu(self.conv_layers[i](x.permute(0, 2, 1)).permute(0, 2, 1))
            x = self.layer_norm_layers[i](x)
            x = self.transformer_encoder[i](x)
            x = F.relu(self.deconv_layers[i](x.permute(0, 2, 1)).permute(0, 2, 1))
            x = self.layer_norm_layers2[i](x)
            x = res + x
        logits = self.output_linear(x)
        return logits
