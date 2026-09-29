"""U-Net with optional ConvLSTM or temporal bottleneck attention."""
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                    nn.GroupNorm(8, outgoing), nn.SiLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                    nn.GroupNorm(8, outgoing), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class ConvLSTM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)

    def forward(self, x, state):
        h, c = (torch.zeros_like(x), torch.zeros_like(x)) if state is None else state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c)


class TemporalFeatureAttention(nn.Module):
    """Attend independently over historical bottlenecks at each spatial site.

    This is the temporal attention used by ``noisetotrans.py``: the target is
    the query, previous frames provide keys/values, and the softmax weights are
    multiplied by 1.1 before the historical context is fused with the target.
    """
    def __init__(self, channels, attention_channels=64, weight_scale=1.1):
        super().__init__()
        if attention_channels < 1 or weight_scale <= 0:
            raise ValueError("Invalid temporal-attention dimensions or weight scale")
        self.attention_channels = attention_channels
        self.scale = attention_channels ** -.5
        self.weight_scale = weight_scale
        self.query = nn.Conv2d(channels, attention_channels, 1)
        self.key = nn.Conv2d(channels, attention_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.fusion = nn.Sequential(nn.Conv2d(2*channels, channels, 1), nn.ReLU(inplace=True))

    def forward(self, target, history):
        if not history:
            return target
        stacked = torch.stack(history, dim=1)
        batch, length, channels, height, width = stacked.shape
        flattened = stacked.reshape(batch*length, channels, height, width)
        query = self.query(target).unsqueeze(1)
        keys = self.key(flattened).reshape(
            batch, length, self.attention_channels, height, width)
        values = self.value(flattened).reshape(batch, length, channels, height, width)
        weights = torch.softmax((query*keys).sum(dim=2)*self.scale, dim=1)
        context = (weights.mul(self.weight_scale).unsqueeze(2)*values).sum(dim=1)
        return self.fusion(torch.cat((target, context), dim=1))


class Noise2Time(nn.Module):
    """Architecture of the supplied ConvLSTM scripts; 512 -> 16 spatial bottleneck."""
    def __init__(self, base_channels=32, convlstm=True, model=None):
        super().__init__()
        self.architecture = model or ("unet_convlstm" if convlstm else "unet")
        if self.architecture not in ("unet", "unet_convlstm", "unet_transformer"):
            raise ValueError(f"Unknown model architecture: {self.architecture}")
        widths = [base_channels * s for s in (1, 2, 4, 8, 16, 16)]
        self.encoders = nn.ModuleList([ConvBlock(1, widths[0])] +
                                      [ConvBlock(c, c) for c in widths[1:]])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths[:-1], widths[1:])])
        self.bottleneck = ConvBlock(widths[-1], widths[-1])
        self.memory = ConvLSTM(widths[-1]) if self.architecture == "unet_convlstm" else None
        self.temporal_attention = (TemporalFeatureAttention(widths[-1])
                                   if self.architecture == "unet_transformer" else None)
        self.decoders = nn.ModuleList([ConvBlock(2*c, c) for c in widths])
        self.up = nn.ModuleList([nn.ConvTranspose2d(b, a, 2, stride=2)
                                for a, b in zip(widths[:-1], widths[1:])])
        self.final = nn.Conv2d(widths[0], 1, 3, padding=1)

    def forward(self, sequence):
        state = None  # Reset for each independent history window.
        history = []
        # Historical decoder outputs are unused; avoid computing them. Encoder and
        # recurrent gradients still propagate through the entire history.
        temporal = self.memory is not None or self.temporal_attention is not None
        indices = range(sequence.shape[1]) if temporal else [sequence.shape[1]-1]
        for t in indices:
            current = sequence[:, t:t+1]
            x = current
            skips = []
            for level, encoder in enumerate(self.encoders):
                x = encoder(x)
                skips.append(x)
                if level < len(self.down):
                    x = self.down[level](x)
            x = self.bottleneck(x)
            if self.memory is not None:
                x, state = self.memory(x, state)
            elif self.temporal_attention is not None and t < sequence.shape[1]-1:
                history.append(x)
        if self.temporal_attention is not None:
            x = self.temporal_attention(x, history)
        for level in range(5, -1, -1):
            x = self.decoders[level](torch.cat((x, skips[level]), dim=1))
            if level:
                x = self.up[level-1](x)
        return current - self.final(x)



