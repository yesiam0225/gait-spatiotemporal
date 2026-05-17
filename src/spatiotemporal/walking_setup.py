"""Walking setup auto-detection from pelvis markers."""

from __future__ import annotations

import logging

import numpy as np

from .models import WalkingSetup

logger = logging.getLogger(__name__)

def _remove_marker_outliers(traj, mad_threshold=10.0):
    """Replace per-axis outlier frames in a marker trajectory with NaN.

    Uses median absolute deviation (MAD) as a robust scale estimate. A frame
    is rejected on a given axis if its value deviates from the per-axis median
    by more than mad_threshold * MAD. If any axis is rejected on a frame, all
    three axes of that frame are set to NaN so the point remains consistent.

    Catches motion-capture artifacts (Vicon mislabels, reconstruction errors)
    without rejecting legitimate walking translation: real walking excursions
    stay within ~5 MAD of the trajectory median, while teleport artifacts
    exceed 100 MAD.

    Parameters
    ----------
    traj : np.ndarray, shape (N, 3)
        Marker trajectory in mm. May contain NaN.
    mad_threshold : float, default 10.0
        MAD multiplier above which a sample is rejected.

    Returns
    -------
    cleaned : np.ndarray, shape (N, 3)
        Copy of traj with outlier frames replaced by NaN.
    n_removed : int
        Number of frames affected.
    """
    cleaned = traj.copy()
    n = len(traj)
    bad_frames = np.zeros(n, dtype=bool)
    for axis in range(3):
        col = traj[:, axis]
        valid = ~np.isnan(col)
        if np.sum(valid) < 10:
            continue
        med = np.median(col[valid])
        mad = np.median(np.abs(col[valid] - med))
        if mad < 1e-6:
            mad = 1.0  # degenerate axis; use a small floor
        bad = np.zeros(n, dtype=bool)
        bad[valid] = np.abs(col[valid] - med) > mad_threshold * mad
        bad_frames |= bad
    cleaned[bad_frames] = np.nan
    return cleaned, int(np.sum(bad_frames))


def determine_setup(markers, obstacle_marker_pair=('OBSTACLE_L', 'OBSTACLE_R'),
                     sampling_rate=100.0):
    lasi = markers.get('LASI')
    if lasi is not None:
        lasi, n_lasi = _remove_marker_outliers(lasi)
        if n_lasi > 0:
            logger.info(
                f"determine_setup: removed {n_lasi} outlier frame(s) from LASI"
            )

    rasi = markers.get('RASI')
    if rasi is not None:
        rasi, n_rasi = _remove_marker_outliers(rasi)
        if n_rasi > 0:
            logger.info(
                f"determine_setup: removed {n_rasi} outlier frame(s) from RASI"
            )

    if lasi is None or rasi is None:
        raise ValueError("LASI and RASI required")

    valid = ~np.isnan(lasi[:, 0]) & ~np.isnan(rasi[:, 0])
    if valid.sum() < 20:
        if (~np.isnan(lasi[:, 0])).sum() > (~np.isnan(rasi[:, 0])).sum():
            pelvis = lasi
            valid = ~np.isnan(lasi[:, 0])
        else:
            pelvis = rasi
            valid = ~np.isnan(rasi[:, 0])
    else:
        pelvis = (lasi + rasi) / 2
    if valid.sum() < 20:
        raise ValueError("Insufficient pelvis data")

    range_x = np.ptp(pelvis[valid, 0])
    range_y = np.ptp(pelvis[valid, 1])
    walking_axis = 0 if range_x > range_y else 1
    ml_axis = 1 - walking_axis

    valid_idx = np.where(valid)[0]
    n = max(10, len(valid_idx) // 4)
    early = pelvis[valid_idx[:n], walking_axis].mean()
    late = pelvis[valid_idx[-n:], walking_axis].mean()
    walking_direction = +1 if late > early else -1

    lasi_ml = np.nanmean(lasi[:, ml_axis])
    rasi_ml = np.nanmean(rasi[:, ml_axis])
    left_ml_sign = +1 if lasi_ml > rasi_ml else -1

    obstacle_pos = 0.0
    if obstacle_marker_pair[0] in markers and obstacle_marker_pair[1] in markers:
        ob_l = markers[obstacle_marker_pair[0]]
        ob_r = markers[obstacle_marker_pair[1]]
        obstacle_pos = float((np.nanmean(ob_l[:, walking_axis]) +
                              np.nanmean(ob_r[:, walking_axis])) / 2)

    return WalkingSetup(walking_axis=walking_axis,
                          walking_direction=walking_direction,
                          ml_axis=ml_axis,
                          left_ml_sign=left_ml_sign,
                          obstacle_pos_on_axis=obstacle_pos,
                          sampling_rate=sampling_rate)

