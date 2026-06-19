"""Synthetic foot-marker trajectories for portfolio demo plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _foot_z_cycle(
    n_frames: int,
    fs: float,
    phase_offset: float,
    ground_z: float = 100.0,
    swing_height: float = 220.0,
    gait_hz: float = 1.15,
) -> np.ndarray:
    t = np.arange(n_frames) / fs
    phase = 2 * np.pi * gait_hz * t + phase_offset
    z = ground_z + swing_height * np.clip(np.sin(phase), 0, None)
    # Sharper descent near contact
    contact = np.sin(phase) < -0.2
    z[contact] = ground_z + 15.0 * np.abs(np.sin(phase[contact]))
    return z


def _marker_xyz(
    n_frames: int,
    x_start: float,
    y: float,
    z: np.ndarray,
    walk_speed: float,
    fs: float,
) -> np.ndarray:
    t = np.arange(n_frames) / fs
    x = x_start + walk_speed * t
    out = np.column_stack([x, np.full(n_frames, y), z])
    return out


def write_demo_foot_csv(path: str | Path, n_frames: int = 500, fs: float = 100.0) -> Path:
    """Write synthetic marker CSV for spatiotemporal event detection demo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    walk = 420.0
    z_lheel = _foot_z_cycle(n_frames, fs, 0.0)
    z_ltoe = z_lheel + 8.0 + 12.0 * np.clip(np.sin(2 * np.pi * 1.15 * np.arange(n_frames) / fs), 0, 1)
    z_rheel = _foot_z_cycle(n_frames, fs, np.pi)
    z_rtoe = z_rheel + 8.0 + 12.0 * np.clip(
        np.sin(2 * np.pi * 1.15 * np.arange(n_frames) / fs + np.pi), 0, 1
    )

    frames = np.arange(n_frames)
    lasi = _marker_xyz(n_frames, 200.0, 280.0, np.full(n_frames, 900.0), walk, fs)
    rasi = _marker_xyz(n_frames, 200.0, 120.0, np.full(n_frames, 905.0), walk, fs)
    lhee = _marker_xyz(n_frames, 200.0, 390.0, z_lheel, walk, fs)
    ltoe = _marker_xyz(n_frames, 200.0, 400.0, z_ltoe, walk, fs)
    rhee = _marker_xyz(n_frames, 200.0, 110.0, z_rheel, walk, fs)
    rtoe = _marker_xyz(n_frames, 200.0, 100.0, z_rtoe, walk, fs)
    obst_l = np.tile([1200.0, 80.0, 450.0], (n_frames, 1))
    obst_r = np.tile([1200.0, 320.0, 450.0], (n_frames, 1))

    rows = {
        "frame": frames,
        "time": frames / fs,
        "LASI_x": lasi[:, 0], "LASI_y": lasi[:, 1], "LASI_z": lasi[:, 2],
        "RASI_x": rasi[:, 0], "RASI_y": rasi[:, 1], "RASI_z": rasi[:, 2],
        "LHEE_x": lhee[:, 0], "LHEE_y": lhee[:, 1], "LHEE_z": lhee[:, 2],
        "LTOE_x": ltoe[:, 0], "LTOE_y": ltoe[:, 1], "LTOE_z": ltoe[:, 2],
        "RHEE_x": rhee[:, 0], "RHEE_y": rhee[:, 1], "RHEE_z": rhee[:, 2],
        "RTOE_x": rtoe[:, 0], "RTOE_y": rtoe[:, 1], "RTOE_z": rtoe[:, 2],
        "OBSTACLE_L_x": obst_l[:, 0], "OBSTACLE_L_y": obst_l[:, 1], "OBSTACLE_L_z": obst_l[:, 2],
        "OBSTACLE_R_x": obst_r[:, 0], "OBSTACLE_R_y": obst_r[:, 1], "OBSTACLE_R_z": obst_r[:, 2],
    }
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
