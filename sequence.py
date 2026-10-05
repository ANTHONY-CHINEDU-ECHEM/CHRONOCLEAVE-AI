"""Sequence encoders for morphokinetic event streams: a bidirectional LSTM and a TCN."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from chronocleave.data.prepare import STEP_FEATURES
from chronocleave.data.schema import EVENTS, STATIC_FEATURES

EVENT_EMBEDDING = 8


class _Base(nn.Module):
    """Shared input embedding, masked pooling and classification head."""

    def __init__(self, encoded_dim: int, dropout: float):
        super().__init__()
        self.event_embedding = nn.Embedding(len(EVENTS), EVENT_EMBEDDING)
        self.register_buffer("event_index", torch.arange(len(EVENTS)))
        self.head = nn.Sequential(
            nn.Linear(2 * encoded_dim + len(STATIC_FEATURES), 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def embed(self, steps: torch.Tensor) -> torch.Tensor:
        identity = self.event_embedding(self.event_index).unsqueeze(0).expand(steps.shape[0], -1, -1)
        return torch.cat([steps, identity], dim=-1)

    def pool_and_classify(self, encoded: torch.Tensor, steps: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        mask = steps[..., 2:3]                                     # 1 where the event was annotated
        mean = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        peak = (encoded + (mask - 1) * 1e4).amax(dim=1)
        return self.head(torch.cat([mean, peak, static], dim=1)).squeeze(1)


class LSTMClassifier(_Base):
    """Bidirectional LSTM over the ordered developmental events."""

    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__(2 * hidden, dropout)
        self.lstm = nn.LSTM(STEP_FEATURES + EVENT_EMBEDDING, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, steps: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(self.embed(steps))
        return self.pool_and_classify(encoded, steps, static)


class _TemporalBlock(nn.Module):
    """Two dilated convolutions with weight normalisation and a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation)), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation)), nn.ReLU(), nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.shortcut(x))


class TCNClassifier(_Base):
    """Temporal convolutional network with exponentially growing dilation.

    The whole developmental record is available when an embryo is scored, so
    the convolutions are centred rather than causal. With kernel 3 and
    dilations 1, 2 and 4 the receptive field spans all twelve events.
    """

    def __init__(self, channels: int = 64, levels: int = 3, kernel: int = 3, dropout: float = 0.1):
        super().__init__(channels, dropout)
        blocks, width = [], STEP_FEATURES + EVENT_EMBEDDING
        for level in range(levels):
            blocks.append(_TemporalBlock(width, channels, kernel, 2 ** level, dropout))
            width = channels
        self.tcn = nn.Sequential(*blocks)

    def forward(self, steps: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(self.embed(steps).transpose(1, 2)).transpose(1, 2)
        return self.pool_and_classify(encoded, steps, static)


def build_model(params: dict) -> nn.Module:
    """Construct a sequence model from a sweep entry."""
    if params["model"] == "lstm":
        return LSTMClassifier(params.get("hidden", 64), params.get("layers", 1), params.get("dropout", 0.1))
    if params["model"] == "tcn":
        return TCNClassifier(params.get("channels", 64), params.get("levels", 3), params.get("kernel", 3), params.get("dropout", 0.1))
    raise ValueError(f"Unknown model type: {params['model']}")
