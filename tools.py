"""
This module serves as module of additional unclassified tools
to be used by other package modules. WARNING: in order to prevent circular imports,
it's not allowed to import here internal package modules.
"""
from copy import deepcopy
import pandas as pd


def make_index(index):
    """Transforms multiindex into plain index, leaving only first level"""
    if isinstance(index, pd.core.indexes.multi.MultiIndex) and not index.empty:
        idx = pd.DatetimeIndex(list(map(lambda x: x[0], index)))
    elif isinstance(index, pd.core.indexes.range.RangeIndex):
        idx = index
    else:
        idx = pd.to_datetime(index)
    return idx


def prepare_ts(ts):
    """Pre-process TS before taking to detect_ts method
    Makes a copy of ts (in order to prevent changes in original ts).
    Drops NA's, checks wherher all values are numeric, and removes values
    with the same timestamps.
    """
    ts = deepcopy(ts)
    ts.dropna(inplace=True)
    ts = pd.to_numeric(ts)
    ts = ts.loc[~ts.index.duplicated(keep='last')]
    return ts


def diff_ts(ts, k=1):
    """
    This function implements k times timeseries differencing,
    so that it could later be restored to the original time series
    by cumsum method below.

    Parameters:
    -----------
    ts : pd.Series
        Original time series
    k : int, default=1
        The numer of differentiations

    Returns:
    --------

    """
    # iterativly differencing
    ts_diff = ts
    fill_na_values = []
    for i in range(0, k):
        fill_na_values.append(ts_diff[i])
        ts_diff = ts_diff.diff()
    # filling NA's
    fill_na_values = pd.Series(data=fill_na_values, index=ts[:k].index)
    ts_diff.fillna(fill_na_values, inplace=True)
    return ts_diff


def cumsum_ts(ts, k=1):
    """
    This function implements k times cummulative summ operation,
    that restores time series, differenced by diff_ts mehtod.

    Parameters:
    -----------
    ts : pd.Series
        Time series, differenced by diff_ts method.
    k : int, default=0
        The number of differentiations, used by diff_ts method

    Returns:
    --------

    """
    ts_cumsum = deepcopy(ts)
    for i in range(k-1, -1, -1):
        ts_cumsum[i:] = ts_cumsum[i:].cumsum()

    return ts_cumsum