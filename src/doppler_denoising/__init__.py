"""Doppler holography video denoising and evaluation tools."""

from importlib import import_module

__version__ = "0.1.0"

_LEGACY_MODULES = {
    "arterial_peaks": ".preparation.arterial_peaks",
    "collect_videos": ".preparation.collection",
    "noise2time_report": ".evaluation.report",
}


def __getattr__(name):
    """Load renamed modules only when an older caller requests one."""
    if name not in _LEGACY_MODULES:
        raise AttributeError(name)
    module = import_module(_LEGACY_MODULES[name], __name__)
    globals()[name] = module
    return module
