from importlib.metadata import PackageNotFoundError, version

__all__ = ["data", "fiber_generation", "mesh_generation"]

try:
    __version__ = version("cardiac_benchmark_toolkitx")
except PackageNotFoundError:
    __version__ = "UNDEFINED"
