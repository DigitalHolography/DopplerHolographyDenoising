"""U-Net and optional ConvLSTM used by Noise2Time."""
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


class Noise2Time(nn.Module):
    """Architecture of the supplied ConvLSTM scripts; 512 -> 16 spatial bottleneck."""
    def __init__(self, base_channels=32, convlstm=True):
        super().__init__()
        widths = [base_channels * s for s in (1, 2, 4, 8, 16, 16)]
        self.encoders = nn.ModuleList([ConvBlock(1, widths[0])] +
                                      [ConvBlock(c, c) for c in widths[1:]])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths[:-1], widths[1:])])
        self.bottleneck = ConvBlock(widths[-1], widths[-1])
        self.memory = ConvLSTM(widths[-1]) if convlstm else None
        self.decoders = nn.ModuleList([ConvBlock(2*c, c) for c in widths])
        self.up = nn.ModuleList([nn.ConvTranspose2d(b, a, 2, stride=2)
                                for a, b in zip(widths[:-1], widths[1:])])
        self.final = nn.Conv2d(widths[0], 1, 3, padding=1)

    def forward(self, sequence):
        state = None  # Reset for each independent history window.
        # Historical decoder outputs are unused; avoid computing them. Encoder and
        # recurrent gradients still propagate through the entire history.
        indices = range(sequence.shape[1]) if self.memory is not None else [sequence.shape[1]-1]
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
        for level in range(5, -1, -1):
            x = self.decoders[level](torch.cat((x, skips[level]), dim=1))
            if level:
                x = self.up[level-1](x)
        return current - self.final(x)



