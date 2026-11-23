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
    are *given* a DataFrame always return the same kind they were given (a
    polars LazyFrame is collected, and a polars DataFrame comes back).
    """
    global _default_backend
    backend = _check_backend(backend)
    if backend == "polars":
        _polars()   # fail here, before switching, if polars is missing
    _default_backend = backend


def get_backend() -> str:
    """The DataFrame library pyorbiter currently returns by default."""
    return _default_backend


def _check_backend(backend) -> str:
    if backend not in BACKENDS:
        raise ValueError("`backend` must be 'pandas' or 'polars'.")
    return backend


def resolve_backend(backend: str | None) -> str:
    """An explicit backend, else the default set by :func:`set_backend`."""
    backend = _default_backend if backend is None else _check_backend(backend)
    if backend == "polars":
        _polars()
    return backend


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
            # Text columns (body ids) come out of pandas as object arrays, with
            # NaN for a missing value where polars wants null
            items = [None if missing else value
                     for value, missing in zip(values.tolist(), frame[name].isna().tolist())]
            try:
                columns[str(name)] = pl.Series(str(name), items, dtype=pl.String)
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                columns[str(name)] = pl.Series(str(name), items, strict=False)
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
    ``sim_data.attrs``, which pandas carries through most operations.

    A polars DataFrame has nowhere to keep them, so they are remembered only
    for the exact DataFrame pyorbiter returned. Anything you derive from it
    (``filter``, ``sort``, ``with_columns``, ``clone``, a file read back)
    gives an empty dict here. For a run that did not use the defaults, pass
    ``G`` and ``softening`` to the analysis functions, and ``G``, ``softening``
    and ``method`` to :func:`continue_simulation`, yourself.
    """
    if isinstance(sim_data, pd.DataFrame):
        return dict(sim_data.attrs)
    entry = _run_settings.get(id(sim_data))
    if entry is None or entry[0]() is not sim_data:
        return {}
    return dict(entry[1])


# The columns pyorbiter computes with. Anything else in a polars frame is the
# user's own and never leaves polars.
_CORE_COLUMNS = ("id", "mass", "x", "y", "z", "vx", "vy", "vz", "time")
_ROW = "__pyorbiter_row__"


def _core_to_pandas(frame, settings: dict) -> pd.DataFrame:
    """The columns pyorbiter uses, as pandas, with each row's position in ``frame``."""
    core = [name for name in frame.columns if name in _CORE_COLUMNS]
    out = pd.DataFrame({name: frame[name].to_numpy() for name in core})
    out[_ROW] = np.arange(frame.height, dtype=float)
    out.attrs = settings
    return out


def _rows_to_polars(result: pd.DataFrame, original):
    """Turn a result made of rows of ``original`` back into polars.

    The user's own columns are picked up again from ``original`` by row, so
    they keep their polars types exactly; rows pyorbiter added (as
    :func:`continue_simulation` does) get nulls there. The columns pyorbiter
    did compute are cast back to the types they arrived with.
    """
    pl = _polars()
    out = to_polars(result)
    out = out.with_columns(pl.col(_ROW).fill_nan(None).cast(pl.Int64))
    extras = [name for name in original.columns if name not in result.columns]
    if extras:
        lookup = original.select(extras).with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias(_ROW))
        out = out.join(lookup, on=_ROW, how="left", maintain_order="left")
    ordered = [name for name in original.columns if name in out.columns]
    out = out.select(ordered + [name for name in out.columns
                                if name not in ordered and name != _ROW])
    for name, dtype in original.schema.items():
        if name in out.columns and out.schema[name] != dtype:
            try:
                out = out.with_columns(pl.col(name).cast(dtype))
            except pl.exceptions.PolarsError:
                pass   # keep the computed type rather than fail
    _record_run_settings(out, result.attrs)
    return out


def accepts_frames(func):
    """Let a function written for pandas take a polars frame as its first argument.

    A polars DataFrame (or LazyFrame, which is collected) is converted on the
    way in, and a DataFrame result is converted back to a polars DataFrame on
    the way out. pandas input is passed straight through.
    """
    @functools.wraps(func)
    def wrapper(sim_data, *args, **kwargs):
        kind = _polars_kind(sim_data)
        if kind is None:
            return func(sim_data, *args, **kwargs)
        settings = run_settings(sim_data)
        original = sim_data.collect() if kind == "LazyFrame" else sim_data
        out = func(_core_to_pandas(original, settings), *args, **kwargs)
        if not isinstance(out, pd.DataFrame):
            return out
        if _ROW in out.columns:
            return _rows_to_polars(out, original)
        return to_polars(out)
    return wrapper
