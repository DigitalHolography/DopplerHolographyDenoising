# Published method provenance

## UDVD

- Repository: <https://github.com/sreyas-mohan/udvd>
- Audited commit: `7374e5bafde865c3a7ef11c5543d5c0b19da40af`
- License: MIT; retained in `LICENSE-UDVD.txt`.
- `udvd_arch.py` is the official blind-video architecture with only registry
  removal and packaging changes.

## Neighbor2Neighbor

- Repository: <https://github.com/TaoHuang2018/Neighbor2Neighbor>
- Audited commit: `e66389ebef1e64d306d8fcb95f096cf427243452`
- License: BSD 3-Clause; retained in `LICENSE-NEIGHBOR2NEIGHBOR.txt`.
- `neighbor_arch.py`, the directed pair sampler, and the published objective
  are adapted to one-channel prepared Doppler frames.

## Blind2Unblind

- Repository inspected: <https://github.com/zejinwang/Blind2Unblind>
- Audited commit: `efb6f82077c438f796e9e334beb21a4ac0a175b3`
- No license file was present. Public source without a license cannot be
  redistributed. `blind2unblind.py` is therefore a clean-room implementation
  of the paper's global mask mapper, beta schedule, and re-visible equations.
  It uses the BSD-licensed grayscale U-Net above as its backbone.

