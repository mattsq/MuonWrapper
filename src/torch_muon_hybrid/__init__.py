from .hybrid import MuonHybrid

__all__ = ["MuonHybrid"]

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:          # editable install
    from ._version import __version__
