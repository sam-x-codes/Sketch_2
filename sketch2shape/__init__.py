"""Sketch2Shape AI — from a 2D sketch to a textured 3D asset."""

__all__ = ["run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
