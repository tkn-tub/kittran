from __future__ import annotations

__all__ = ["run"]


def run(*args, **kwargs):
    from .simulation import run as _run
    return _run(*args, **kwargs)
