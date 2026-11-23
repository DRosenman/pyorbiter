"""DataFrame backends: pandas (the default) or polars.

The engine computes with pandas internally. This module converts at the
package boundary, so every public function accepts either kind of DataFrame
and hands back the kind it was given. Conversion goes column by column
through numpy, so polars support does not need pyarrow.
"""

from __future__ import annotations

import functools
import weakref

import numpy as np
import pandas as pd

BACKENDS = ("pandas", "polars")

_default_backend = "pandas"

# polars DataFrames have no `.attrs`, so the settings of the run that produced
# a frame are kept here, keyed by the frame's id() and dropped when the frame
# is garbage collected.
_run_settings: dict[int, tuple[weakref.ref, dict]] = {}


def set_backend(backend: str) -> None:
    """Choose the DataFrame library pyorbiter returns: ``"pandas"`` or ``"polars"``.

    This sets the default for :func:`simulate_system` and :func:`get_bodies`;
    both also take a ``backend`` argument for a single call. Functions that
    are *given* a DataFrame always return the same kind they were given.
    """
    global _default_backend
    _default_backend = _check_backend(backend)
    if _default_backend == "polars":
        _polars()


def get_backend() -> str:
    """The DataFrame library pyorbiter currently returns by default."""
    return _default_backend


def _check_backend(backend) -> str:
    if backend not in BACKENDS:
        raise ValueError("`backend` must be 'pandas' or 'polars'.")
    return backend


def resolve_backend(backend: str | None) -> str:
    """An explicit backend, else the default set by :func:`set_backend`."""
    return _default_backend if backend is None else _check_backend(backend)


def _polars():
    try:
        import polars as pl
    except ImportError:
        raise ImportError("The polars backend needs the 'polars' package. "
                          'Install it with: pip install "pyorbiter[polars]"') from None
    return pl


def _polars_kind(obj) -> str | None:
    """``"DataFrame"`` or ``"LazyFrame"`` for a polars frame, else None (without importing polars)."""
    for cls in type(obj).__mro__:
        if cls.__module__.split(".")[0] == "polars" and cls.__name__ in ("DataFrame", "LazyFrame"):
            return cls.__name__
    return None


def is_frame(obj) -> bool:
    """True for a pandas DataFrame or a polars DataFrame / LazyFrame."""
    return isinstance(obj, pd.DataFrame) or _polars_kind(obj) is not None


def to_pandas(frame) -> pd.DataFrame:
    """A pandas copy of a polars frame (pandas frames pass through untouched).

    The run settings recorded for a polars frame become the copy's ``attrs``.
    """
    kind = _polars_kind(frame)
    if kind is None:
        return frame
    settings = run_settings(frame)
    if kind == "LazyFrame":
        frame = frame.collect()
    out = pd.DataFrame({name: frame[name].to_numpy() for name in frame.columns})
    out.attrs = settings
    return out


def to_polars(frame: pd.DataFrame):
    """A polars copy of a pandas DataFrame, with its ``attrs`` recorded as run settings."""
    pl = _polars()
    columns = {}
    for name in frame.columns:
        values = frame[name].to_numpy()
        if values.dtype == object:
            # Text columns (body ids) come out of pandas as object arrays
            try:
                columns[str(name)] = pl.Series(str(name), values.tolist(), dtype=pl.String)
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                columns[str(name)] = pl.Series(str(name), values.tolist())
        else:
            columns[str(name)] = pl.Series(str(name), np.ascontiguousarray(values))
    out = pl.DataFrame(columns)
    _record_run_settings(out, frame.attrs)
    return out


def to_backend(frame: pd.DataFrame, backend: str | None = None):
    """Convert an internal pandas DataFrame to the requested (or default) backend."""
    return to_polars(frame) if resolve_backend(backend) == "polars" else frame


def _record_run_settings(frame, settings: dict) -> None:
    if not settings:
        return
    key = id(frame)
    _run_settings[key] = (weakref.ref(frame, lambda _, key=key: _run_settings.pop(key, None)),
                          dict(settings))


def run_settings(sim_data) -> dict:
    """The settings of the run that produced ``sim_data``.

    A dict with ``G``, ``softening``, ``method`` and ``time_step`` for the
    output of :func:`simulate_system`, :func:`continue_simulation` or
    :func:`shift_reference_frame`. For a pandas DataFrame this is a copy of
    ``sim_data.attrs``. It is empty for a frame that pyorbiter did not
    produce itself, such as one you filtered, joined or read from a file; in
    that case pass ``G`` and ``softening`` to the analysis functions directly.
    """
    if isinstance(sim_data, pd.DataFrame):
        return dict(sim_data.attrs)
    entry = _run_settings.get(id(sim_data))
    if entry is None or entry[0]() is not sim_data:
        return {}
    return dict(entry[1])


def accepts_frames(func):
    """Let a function written for pandas take a polars frame as its first argument.

    A polars DataFrame or LazyFrame is converted on the way in, and a
    DataFrame result is converted back to polars on the way out. pandas input
    is passed straight through.
    """
    @functools.wraps(func)
    def wrapper(sim_data, *args, **kwargs):
        if _polars_kind(sim_data) is None:
            return func(sim_data, *args, **kwargs)
        out = func(to_pandas(sim_data), *args, **kwargs)
        return to_polars(out) if isinstance(out, pd.DataFrame) else out
    return wrapper
