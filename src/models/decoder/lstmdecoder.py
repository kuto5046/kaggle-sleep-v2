import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_classes: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class CNN1DLSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_classes: int,
    ):
        super().__init__()

        self.num_layers = num_layers
        encoder_layers = [
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=1,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
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

        self.lstm_layers = nn.ModuleList(encoder_layers)
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
            x, _ = self.lstm_layers[i](x)
            x = F.relu(self.deconv_layers[i](x.permute(0, 2, 1)).permute(0, 2, 1))
            x = self.layer_norm_layers2[i](x)
            x = res + x
        logits = self.output_linear(x)
        return logits
