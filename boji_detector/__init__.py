"""boji-detector: explainable, photo-based physique metrics."""

__version__ = "0.2.0"

__all__ = ["__version__", "analyze_image", "analyze_perception"]


def __getattr__(name):
    if name in ("analyze_image", "analyze_perception"):
        from . import analyze

        return getattr(analyze, name)
    raise AttributeError(name)
