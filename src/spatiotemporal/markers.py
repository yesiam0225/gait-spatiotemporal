"""Marker CSV loading and Butterworth filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def load_marker_csv(csv_path):
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        if c.endswith(('_x', '_y', '_z')):
            name = c[:-2].strip()
            col_map.setdefault(name, {})[c[-1]] = c
    if 'frame' not in df.columns:
        raise ValueError("CSV must have 'frame' column")
    return df, col_map, df['frame'].to_numpy()


def get_traj(df, col_map, marker):
    if marker not in col_map:
        return None
    cols = col_map[marker]
    if not all(k in cols for k in ('x', 'y', 'z')):
        return None
    return df[[cols['x'], cols['y'], cols['z']]].to_numpy()


def butterworth_filter_traj(traj, fs, cutoff=6.0, order=4):
    if traj is None:
        return None
    filtered = traj.copy()
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    for axis in range(traj.shape[1]):
        col = traj[:, axis]
        valid = ~np.isnan(col)
        if valid.sum() < 4 * order:
            continue
        diff = np.diff(np.concatenate([[False], valid, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            seg = col[s:e]
            if len(seg) >= 4 * order:
                filtered[s:e, axis] = filtfilt(b, a, seg)
    return filtered


def filter_markers(markers, fs, cutoff=6.0):
    return {name: butterworth_filter_traj(t, fs, cutoff)
            for name, t in markers.items()}

