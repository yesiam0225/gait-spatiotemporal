"""
Spatiotemporal gait analysis for obstacle-crossing walking trials.

Pipeline:
    1. Load marker CSV (gap-filled or corrected)
    2. Auto-detect walking setup (axis, direction, ML, obstacle)
    3. Filter markers (Butterworth 6Hz, optional)
    4. Foot-based gait event detection (pelvis-independent)
    5. Detect obstacle crossing events (lead/trail toe crossing)
    6. Compute per-stride parameters (stride/step/SS/DS%)
    7. Phase classification per leg (approach/crossing_lead/crossing_trail/recovery)
    8. Compute obstacle-crossing parameters
    9. Outlier detection (3-stage: IQR + trial-deviation + physiological)
    10. Aggregation to per-trial and per-subject levels (long format)

Study design: 2 groups (adult/child) x 2 boards (RB/WB) x 2 times (pre/post)
Per subject: 6 trials per condition x 4 conditions = 24 trials

Output CSVs:
    per_stride_data.csv             : finest granularity, all strides
    per_step_data.csv               : one row per foot IC (step timing at landing)
    per_trial_phase_summary.csv     : trial x phase x side, mean/SD across strides
    per_trial_obstacle.csv          : per-trial obstacle parameters
    per_subject_summary.csv         : ANOVA-ready, mean/SD across 6 trials
    per_subject_obstacle_summary.csv: subject-level obstacle parameters

Conventions:
    Length: mm
    Time: seconds
    Velocity: m/s
    Normalized (dimensionless):
        length_norm = length_mm / leg_length_mm
        time_norm   = time_s   / sqrt(leg_length_m / g)
        speed_norm  = speed_m_s / sqrt(g * leg_length_m)

Marker / peak-finding notes:
    - ``_find_peaks_nan_safe`` tolerates short NaN gaps in marker trajectories.
      Peaks adjacent (±1 frame) to NaN frames are dropped to avoid relying on
      unreliable heights.
    - Heel-strike cycle boundaries use either a heel swing peak (height ≥
      ground + 80 mm) or a toe swing peak (height ≥ ground + 50 mm). This
      recovers cycles when the heel barely lifts but the toe swings normally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


logger = logging.getLogger(__name__)
G = 9.81  # m/s^2


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WalkingSetup:
    walking_axis: int           # 0 (X) or 1 (Y)
    walking_direction: int      # +1 or -1
    ml_axis: int
    left_ml_sign: int
    obstacle_pos_on_axis: float
    sampling_rate: float = 100.0


@dataclass
class TrialMetadata:
    subject_id: str
    group: str                  # 'adult' or 'child'
    board: str                  # 'RB' or 'WB'
    time: str                   # 'pre' or 'post'
    trial: int


@dataclass
class Anthropometry:
    leg_length_mm: float        # required for normalization
    mass_kg: Optional[float] = None
    age_years: Optional[float] = None
    height_mm: Optional[float] = None


@dataclass
class GaitEvents:
    left_hs: list[int] = field(default_factory=list)
    right_hs: list[int] = field(default_factory=list)
    left_to: list[int] = field(default_factory=list)
    right_to: list[int] = field(default_factory=list)
    # Strike type for each IC, parallel to left_hs / right_hs
    left_strike_types: list[str] = field(default_factory=list)
    right_strike_types: list[str] = field(default_factory=list)
    lead_toe_crossing: Optional[int] = None
    trail_toe_crossing: Optional[int] = None
    lead_foot_side: Optional[str] = None
    trail_foot_side: Optional[str] = None
    lead_crossing_marker: Optional[str] = None   # 'toe' or 'heel'
    trail_crossing_marker: Optional[str] = None  # 'toe' or 'heel'


@dataclass
class StrideRecord:
    stride_idx_in_trial: int
    side: str                   # 'left' or 'right'
    phase: str                  # approach/crossing_lead/crossing_trail/recovery/unknown
    hs_start_frame: int
    hs_end_frame: int
    to_frame: int
    opp_hs_frame: Optional[int] = None
    opp_to_frame: Optional[int] = None
    # Frame of the previous opposite-foot IC (for step_time); None if first step in trial
    prev_opposite_ic_frame: Optional[int] = None

    # Strike type at the start of this stride (the IC at hs_start_frame)
    ic_start_strike_type: str = 'unknown'  # 'HS', 'TS', 'MS', or 'unknown'
    crossing_marker_used: Optional[str] = None  # 'toe' or 'heel'; crossing phases only

    # Stride parameters
    stride_time_s: float = float('nan')
    stride_length_mm: float = float('nan')
    stance_pct: float = float('nan')
    swing_pct: float = float('nan')
    double_support_1_pct: float = float('nan')
    double_support_2_pct: float = float('nan')
    single_support_pct: float = float('nan')
    gait_speed_m_s: float = float('nan')

    # Step parameters (unified, recorded at this stride's HS_start)
    step_length_mm: Optional[float] = None
    step_time_s: Optional[float] = None
    step_width_mm: Optional[float] = None

    # Normalized
    stride_length_norm: float = float('nan')
    stride_time_norm: float = float('nan')
    gait_speed_norm: float = float('nan')
    step_length_norm: Optional[float] = None
    step_time_norm: Optional[float] = None
    step_width_norm: Optional[float] = None

    outlier_flag: str = ''


@dataclass
class ObstacleParameters:
    lead_foot_side: Optional[str] = None
    trail_foot_side: Optional[str] = None
    lead_toe_clearance_mm: Optional[float] = None
    trail_toe_clearance_mm: Optional[float] = None
    lead_step_before_dist_mm: Optional[float] = None
    lead_step_after_dist_mm: Optional[float] = None
    trail_step_before_dist_mm: Optional[float] = None
    trail_step_after_dist_mm: Optional[float] = None
    crossing_step_length_mm: Optional[float] = None


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PHYSIOLOGICAL_BOUNDS = {
    'adult': {
        'stride_time_s': (0.7, 1.5),
        'stride_length_norm': (0.6, 1.7),
        'stance_pct': (50, 75),
        'swing_pct': (25, 50),
        'double_support_1_pct': (5, 25),
        'double_support_2_pct': (5, 25),
        'single_support_pct': (25, 50),
        'gait_speed_norm': (0.2, 0.6),
    },
    'child': {
        'stride_time_s': (0.5, 1.4),
        'stride_length_norm': (0.5, 1.6),
        'stance_pct': (50, 75),
        'swing_pct': (25, 50),
        'double_support_1_pct': (5, 30),
        'double_support_2_pct': (5, 30),
        'single_support_pct': (25, 50),
        'gait_speed_norm': (0.2, 0.7),
    },
}

OUTLIER_VARS = [
    'stride_time_s', 'stride_length_norm',
    'stance_pct', 'swing_pct',
    'double_support_1_pct', 'double_support_2_pct',
    'single_support_pct', 'gait_speed_norm',
]

VARS_TO_AGGREGATE = [
    'stride_time_s', 'stride_length_mm',
    'step_time_s', 'step_length_mm', 'step_width_mm',
    'stance_pct', 'swing_pct',
    'double_support_1_pct', 'double_support_2_pct', 'single_support_pct',
    'gait_speed_m_s',
    'stride_length_norm', 'step_length_norm', 'step_width_norm',
    'stride_time_norm', 'step_time_norm', 'gait_speed_norm',
]


# =============================================================================
# Marker loading and filtering
# =============================================================================

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


# =============================================================================
# Walking setup auto-detection
# =============================================================================

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


# =============================================================================
# Foot-based gait event detection (pelvis-independent)
# =============================================================================

def _dedupe_nearby_ics(ics_with_types, min_gap_frames):
    """Drop a later IC if it falls within min_gap_frames of the previous (same foot)."""
    if len(ics_with_types) <= 1:
        return ics_with_types
    out = [ics_with_types[0]]
    for frame, strike in ics_with_types[1:]:
        prev_frame, prev_strike = out[-1]
        if frame - prev_frame < min_gap_frames:
            # Prefer TS over HS; otherwise keep the earlier contact.
            if strike == 'TS' and prev_strike == 'HS':
                out[-1] = (frame, strike)
            continue
        out.append((frame, strike))
    return out


def detect_gait_events_foot_based(markers, setup, raw_markers=None,
                                     swing_height_above_ground=80,
                                     swing_distance_frames=50,
                                     hs_ground_tolerance_mm=40,
                                     ts_ground_tolerance_mm=20,
                                     ts_pre_swing_min_height=50,
                                     min_separation_frames=40,
                                     pre_descent_frames=10,
                                     min_hs_pre_descent_speed=30,
                                     ts_pre_descent_speed=200,
                                     ts_accel_peak_min=10000,
                                     to_accel_peak_min=10000,
                                     to_min_velocity_at_peak=100,
                                     to_post_rise_frames=10,
                                     to_post_rise_min=15,
                                     flat_foot_toe_heel_mm=10.0,
                                     flat_foot_ic_check_window_frames=3,
                                     flat_foot_ground_tolerance_mm=60.0,
                                     flat_foot_vz_prominence=80.0,
                                     flat_foot_swing_margin_frames=10,
                                     flat_foot_landing_max_frames=100,
                                     ts_validation_max_gap=50,
                                     preceding_hs_window=5):
    """
    Pelvis-independent gait event detection using Z-minimum priority algorithm.

    Uses RAW (unfiltered) marker data with backward difference for derivatives.
    This is critical because filtering shifts Az peak positions and can mask
    the sharp transitions characteristic of toe-strike impacts.

    HS Detection (per heel swing cycle):
        Default: DEEPEST Z sign-change (Vz neg→pos) with Z ≤ ground + 40 mm
        (biphasic heel-rocker: deepest crossing wins over an earlier shallow one).

        Flat-foot refinement (when toe Z is available): after normal deepest-Z
        HS per cycle, if min |toe−heel| at IC ± ``flat_foot_ic_check_window_frames``
        is ≤ ``flat_foot_toe_heel_mm`` (default 10 mm), re-pick IC from the
        strongest heel/toe downward-Vz peaks in the early post-swing window.

    TO Detection (per toe swing cycle):
        First positive Az peak (≥10000 mm/s²) where:
        - Vz > 100 mm/s (toe rising rapidly)
        - Z near ground (≤+50mm, toe leaving minimum)
        - Z continues rising ≥15mm in next 10 frames (sustained swing)

    TS Detection (with strict criteria to reject stance-phase artifacts):
        Per cycle in toe Z, find deepest Z sign-change with:
        - Z at strike ≤ ground+20mm (must be at actual ground level)
        - Vz[strike-1] < -100 (toe still descending one frame before impact)
        - Pre-10 frames: all Vz < 0, mean |Vz| ≥ 200 mm/s (real swing descent)
        - Pre-swing peak ≥ ground+50mm (real swing, not stance oscillation)
        - Az peak ≥10000 nearby (impact deceleration)
        - Validated only if (a) same-foot HS follows within
          ``ts_validation_max_gap`` frames (default 50 = 500 ms at 100 Hz) AND
          (b) NO same-foot HS occurred within ``preceding_hs_window`` frames
          before (default 5 = 50 ms). Condition (b) rejects the toe-down phase
          of a normal heel-strike landing, which can superficially resemble a
          toe-strike at the acceleration level. The paired following HS is then
          NOT counted as separate IC (heel-rocker after toe strike). The 500 ms
          follow window accommodates slow heel-rocker after toe contact in
          obstacle crossing.

    Args:
        markers: filtered markers dict (used as fallback if raw_markers None)
        raw_markers: RAW unfiltered markers dict (required for accurate
            event detection). If None, falls back to markers (may shift events).
        setup: WalkingSetup with sampling_rate

    Returns: GaitEvents with HS, TO, TS frames + strike_types list parallel to
    hs lists. Strike type is 'HS' for heel-strike ICs or 'TS' for toe-strike ICs.
    """
    fs = setup.sampling_rate
    events = GaitEvents()

    # Use raw markers for event detection if available
    detect_markers = raw_markers if raw_markers is not None else markers

    for side, heel_name, toe_name, hs_list, to_list, strike_list in [
        ('left', 'LHEE', 'LTOE', events.left_hs, events.left_to,
         events.left_strike_types),
        ('right', 'RHEE', 'RTOE', events.right_hs, events.right_to,
         events.right_strike_types),
    ]:
        heel = detect_markers.get(heel_name)
        toe = detect_markers.get(toe_name)
        if heel is None or toe is None:
            logger.warning(f"Missing heel/toe markers for {side}")
            continue

        heel_z = heel[:, 2]
        toe_z = toe[:, 2]

        # Detect HS first (no dependency on TO)
        hs_frames = _detect_heel_strike_zmin(
            heel_z, fs, toe_z=toe_z,
            swing_height=swing_height_above_ground,
            swing_distance=swing_distance_frames,
            ground_tolerance=hs_ground_tolerance_mm,
            flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
            flat_foot_ic_check_window_frames=flat_foot_ic_check_window_frames,
            flat_foot_ground_tolerance_mm=flat_foot_ground_tolerance_mm,
            flat_foot_vz_prominence=flat_foot_vz_prominence,
            flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
            flat_foot_landing_max_frames=flat_foot_landing_max_frames,
            min_separation=min_separation_frames,
            pre_descent_frames=pre_descent_frames,
            min_pre_descent_speed=min_hs_pre_descent_speed,
        )
        # TO detection uses HS-bounded search (more accurate at trial start)
        to_frames = _detect_toe_off(
            toe_z, fs,
            hs_frames=hs_frames,
            accel_peak_min=to_accel_peak_min,
            min_velocity_at_peak=to_min_velocity_at_peak,
            post_rise_frames=to_post_rise_frames,
            post_rise_min=to_post_rise_min,
            min_separation=min_separation_frames,
        )
        ts_candidates = _detect_toe_strike_candidates(
            toe_z, fs,
            ground_tolerance=ts_ground_tolerance_mm,
            pre_swing_min_height=ts_pre_swing_min_height,
            pre_descent_frames=pre_descent_frames,
            min_pre_descent_speed=ts_pre_descent_speed,
            accel_peak_min=ts_accel_peak_min,
            min_separation=min_separation_frames,
        )
        # Validate TS: must precede same-foot HS within window
        ts_valid, hs_paired_set = _validate_toe_strikes(
            ts_candidates, hs_frames,
            max_gap_frames=ts_validation_max_gap,
            preceding_hs_window=preceding_hs_window,
            heel_z=heel_z,
            hs_ground_tolerance_mm=hs_ground_tolerance_mm,
        )

        # Rejected TS with preceding heel contact: promote that heel frame to HS
        # and suppress the following HS that would have been the TS rocker.
        supplemental_hs = []
        suppressed_hs = set()
        for ts in ts_candidates:
            if ts in ts_valid:
                continue
            prec_hs = [h for h in hs_frames
                       if ts - preceding_hs_window <= h < ts]
            if not prec_hs:
                hf = _find_preceding_heel_strike_frame(
                    heel_z, ts, preceding_hs_window, hs_ground_tolerance_mm
                )
                if hf is not None:
                    prec_hs = [hf]
            if not prec_hs:
                continue
            sh = int(prec_hs[0])
            if any(abs(sh - int(h)) < min_separation_frames for h in hs_frames):
                continue
            supplemental_hs.append(sh)
            following = [h for h in hs_frames if ts < h <= ts + ts_validation_max_gap]
            if following:
                suppressed_hs.add(following[0])

        # Build IC list: TS frames + HS frames not paired-with-TS
        # All ICs get strike type; paired HS are excluded
        ics_with_types = []
        ic_frames_seen = set()
        for ts in ts_valid:
            ics_with_types.append((ts, 'TS'))
            ic_frames_seen.add(ts)
        for hs in hs_frames:
            if hs not in hs_paired_set and hs not in suppressed_hs and hs not in ic_frames_seen:
                ics_with_types.append((hs, 'HS'))
                ic_frames_seen.add(hs)
        for hs in supplemental_hs:
            if (hs not in hs_paired_set and hs not in suppressed_hs
                    and hs not in ic_frames_seen):
                ics_with_types.append((hs, 'HS'))
                ic_frames_seen.add(hs)
        ics_with_types.sort(key=lambda x: x[0])
        ics_with_types = _dedupe_nearby_ics(ics_with_types, min_separation_frames)

        for f, t in ics_with_types:
            hs_list.append(int(f))
            strike_list.append(t)
        for to_f in to_frames:
            to_list.append(int(to_f))

    events.left_to.sort()
    events.right_to.sort()
    return events


def _backward_diff(x, dt):
    """Backward difference: x'[i] = (x[i] - x[i-1]) / dt. First frame is NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = (x[i] - x[i-1]) / dt
    return out


def _find_peaks_nan_safe(signal, **kwargs):
    """Wrapper around scipy.signal.find_peaks that tolerates NaN frames.

    NaN frames are replaced with the per-signal 5th-percentile value (a
    conservative low estimate of the marker's resting level) before peak
    finding. This prevents scipy from missing peaks adjacent to short NaN
    gaps. Peaks that fall on a NaN frame or are immediately adjacent (±1
    frame) to a NaN frame are then dropped, because their height and
    prominence cannot be trusted.

    Parameters and return values match scipy.signal.find_peaks.
    """
    nan_mask = np.isnan(signal)
    if not nan_mask.any():
        return find_peaks(signal, **kwargs)
    valid = signal[~nan_mask]
    if len(valid) == 0:
        return np.array([], dtype=int), {}
    floor = float(np.percentile(valid, 5))
    filled = np.where(nan_mask, floor, signal)
    peaks, props = find_peaks(filled, **kwargs)
    if len(peaks) == 0:
        return peaks, props
    # Drop peaks touching NaN frames
    keep = np.ones(len(peaks), dtype=bool)
    n = len(signal)
    for i, p in enumerate(peaks):
        for off in (-1, 0, 1):
            j = p + off
            if 0 <= j < n and nan_mask[j]:
                keep[i] = False
                break
    return peaks[keep], {k: v[keep] for k, v in props.items()
                            if isinstance(v, np.ndarray) and len(v) == len(peaks)}


def _marker_strongest_descent_peak(z, fs, win_start, win_end, *,
                                    prominence=80.0,
                                    other_z=None, max_toe_heel_mm=None):
    """Strongest downward-Vz peak (``find_peaks(-Vz)``) in [win_start, win_end)."""
    if win_start >= win_end - 1:
        return None
    vz = _backward_diff(z, 1 / fs)
    peaks, _ = _find_peaks_nan_safe(
        -vz[win_start:win_end], prominence=prominence)
    if len(peaks) == 0:
        return None
    abs_peaks = peaks.astype(int) + win_start
    if max_toe_heel_mm is not None and other_z is not None:
        flat_peaks = []
        for p in abs_peaks:
            lo = max(0, p - 2)
            hi = min(len(z), p + 3)
            diffs = np.abs(z[lo:hi] - other_z[lo:hi])
            if np.any(np.isfinite(diffs)) and float(np.nanmin(diffs)) <= max_toe_heel_mm:
                flat_peaks.append(p)
        if not flat_peaks:
            return None
        abs_peaks = flat_peaks
    return int(min(abs_peaks, key=lambda p: float(vz[p])))


def _min_toe_heel_diff_at_frame(heel_z, toe_z, frame, *, window_frames=3):
    """Min |toe−heel| Z over ``frame ± window_frames`` (inclusive)."""
    n = len(heel_z)
    lo = max(0, int(frame) - window_frames)
    hi = min(n, int(frame) + window_frames + 1)
    diffs = np.abs(toe_z[lo:hi] - heel_z[lo:hi])
    if not np.any(np.isfinite(diffs)):
        return float('inf')
    return float(np.nanmin(diffs))


def _is_flat_foot_at_ic(heel_z, toe_z, ic_frame, *, max_toe_heel_mm=10.0,
                         ic_check_window_frames=3):
    """True when min |toe−heel| at the IC (±window) is ≤ threshold."""
    if max_toe_heel_mm < 0:
        return False
    return (_min_toe_heel_diff_at_frame(
        heel_z, toe_z, ic_frame, window_frames=ic_check_window_frames)
            <= max_toe_heel_mm)


def _heel_swing_cycle_boundaries(heel_z, fs, toe_z=None, *,
                                  swing_height=80, swing_distance=50,
                                  toe_swing_height=50, toe_swing_distance=50):
    """Return ``[(cycle_start, cycle_end), ...]`` from heel/toe swing peaks."""
    n = len(heel_z)
    valid = heel_z[~np.isnan(heel_z)]
    if len(valid) == 0:
        return []
    ground_z = float(np.percentile(valid, 5))
    heel_swing_peaks, _ = _find_peaks_nan_safe(
        heel_z, height=ground_z + swing_height,
        distance=swing_distance, prominence=30)
    swing_peaks = heel_swing_peaks
    if toe_z is not None:
        valid_t = toe_z[~np.isnan(toe_z)]
        if len(valid_t) > 0:
            ground_toe = float(np.percentile(valid_t, 5))
            toe_swing_peaks, _ = _find_peaks_nan_safe(
                toe_z, height=ground_toe + toe_swing_height,
                distance=toe_swing_distance, prominence=20)
            low_heel_swing = len(heel_swing_peaks) < len(toe_swing_peaks)
            first_heel = (int(heel_swing_peaks[0])
                          if len(heel_swing_peaks) > 0 else None)
            merged = list(heel_swing_peaks)
            for tp in toe_swing_peaks:
                if (first_heel is not None and not low_heel_swing
                        and int(tp) < first_heel):
                    continue
                if not any(abs(int(tp) - int(hp)) < swing_distance
                              for hp in heel_swing_peaks):
                    merged.append(int(tp))
            swing_peaks = np.array(sorted(merged), dtype=int)
    if len(swing_peaks) == 0:
        return []
    bounds = [0] + list(swing_peaks) + [n]
    return [(bounds[i], bounds[i + 1])
            for i in range(len(bounds) - 1)]


def _cycle_for_frame(cycles, frame):
    for cycle_start, cycle_end in cycles:
        if cycle_start <= frame < cycle_end:
            return cycle_start, cycle_end
    return None, None


def _refine_flat_foot_hs_at_ics(heel_z, toe_z, fs, hs_frames, *,
                                 flat_foot_toe_heel_mm=10.0,
                                 flat_foot_ic_check_window_frames=3,
                                 flat_foot_vz_prominence=80.0,
                                 flat_foot_swing_margin_frames=10,
                                 flat_foot_landing_max_frames=100,
                                 swing_height=80, swing_distance=50,
                                 toe_swing_height=50, toe_swing_distance=50):
    """
    Keep normal HS picks; re-pick with descent-Vz peaks only when the IC is
    flat-foot at contact (min |toe−heel| at IC ± window ≤ threshold).
    """
    if toe_z is None or flat_foot_toe_heel_mm < 0 or not hs_frames:
        return list(hs_frames)
    cycles = _heel_swing_cycle_boundaries(
        heel_z, fs, toe_z=toe_z, swing_height=swing_height,
        swing_distance=swing_distance,
        toe_swing_height=toe_swing_height,
        toe_swing_distance=toe_swing_distance)
    refined: list[int] = []
    for hs in hs_frames:
        if not _is_flat_foot_at_ic(
                heel_z, toe_z, hs,
                max_toe_heel_mm=flat_foot_toe_heel_mm,
                ic_check_window_frames=flat_foot_ic_check_window_frames):
            refined.append(int(hs))
            continue
        cycle_start, cycle_end = _cycle_for_frame(cycles, hs)
        if cycle_start is None:
            refined.append(int(hs))
            continue
        descent = _pick_flat_foot_ic_from_descent_peaks(
            heel_z, toe_z, fs, cycle_start, cycle_end,
            vz_prominence=flat_foot_vz_prominence,
            swing_margin_frames=flat_foot_swing_margin_frames,
            landing_max_frames=flat_foot_landing_max_frames,
            max_toe_heel_mm=flat_foot_toe_heel_mm)
        refined.append(int(descent) if descent is not None else int(hs))
    return refined


def _pick_flat_foot_ic_from_descent_peaks(heel_z, toe_z, fs, cycle_start,
                                           cycle_end, *, vz_prominence=80.0,
                                           swing_margin_frames=10,
                                           landing_max_frames=100,
                                           max_toe_heel_mm=10.0):
    """
    Flat-foot IC: per marker, strongest descent-Vz peak in the early post-swing
    window; take the earlier of heel and toe candidates.
    """
    win_start = min(cycle_start + swing_margin_frames,
                    max(cycle_end - 1, cycle_start))
    win_end = min(cycle_start + landing_max_frames, cycle_end)
    if win_start >= win_end - 1:
        return None
    heel_pick = _marker_strongest_descent_peak(
        heel_z, fs, win_start, win_end, prominence=vz_prominence,
        other_z=toe_z, max_toe_heel_mm=max_toe_heel_mm)
    toe_pick = None
    if toe_z is not None:
        toe_pick = _marker_strongest_descent_peak(
            toe_z, fs, win_start, win_end, prominence=vz_prominence,
            other_z=heel_z, max_toe_heel_mm=max_toe_heel_mm)
    candidates = [c for c in (heel_pick, toe_pick) if c is not None]
    if not candidates:
        return None
    return min(candidates)


def _detect_heel_strike_zmin(heel_z, fs, toe_z=None, swing_height=80,
                                swing_distance=50, ground_tolerance=40,
                                flat_foot_ground_tolerance_mm=60.0,
                                min_separation=40, pre_descent_frames=10,
                                min_pre_descent_speed=30,
                                toe_swing_height=50, toe_swing_distance=50,
                                flat_foot_toe_heel_mm=10.0,
                                flat_foot_ic_check_window_frames=3,
                                flat_foot_vz_prominence=80.0,
                                flat_foot_swing_margin_frames=10,
                                flat_foot_landing_max_frames=100):
    """
    HS per swing cycle: deepest near-ground heel Vz+ (normal IC).

    When ``toe_z`` is provided, toe-swing peaks supplement heel-swing peaks for
    cycle boundaries, then each HS is checked for flat-foot contact; flat ICs are
    refined via ``_refine_flat_foot_hs_at_ics`` (descent-Vz peak re-pick).
    """
    velocity = _backward_diff(heel_z, 1/fs)
    n = len(heel_z)
    valid = heel_z[~np.isnan(heel_z)]
    if len(valid) == 0:
        return []
    ground_z = float(np.percentile(valid, 5))

    cycles = _heel_swing_cycle_boundaries(
        heel_z, fs, toe_z=toe_z, swing_height=swing_height,
        swing_distance=swing_distance,
        toe_swing_height=toe_swing_height,
        toe_swing_distance=toe_swing_distance)
    if not cycles:
        return []

    strikes = []
    for i, (cycle_start, cycle_end) in enumerate(cycles):
        is_first_segment = (i == 0)  # before first swing peak

        def _trial_start_ok(j: int) -> bool:
            if not is_first_segment:
                return True
            if j < pre_descent_frames:
                return False
            pre = velocity[j - pre_descent_frames:j]
            if np.any(np.isnan(pre)):
                return False
            if np.mean(np.abs(pre)) < min_pre_descent_speed:
                return False
            return np.sum(pre < 0) >= 7

        vz_crossings: list[tuple[int, float, float]] = []
        for j in range(max(cycle_start, 1), cycle_end):
            if np.isnan(velocity[j]) or np.isnan(velocity[j - 1]):
                continue
            if velocity[j - 1] < 0 and velocity[j] >= 0 and _trial_start_ok(j):
                toe_heel = 999.0
                if toe_z is not None and np.isfinite(toe_z[j]):
                    toe_heel = float(abs(toe_z[j] - heel_z[j]))
                vz_crossings.append((j, float(heel_z[j]), toe_heel))

        strict = [
            c for c in vz_crossings if c[1] <= ground_z + ground_tolerance
        ]
        relaxed = [
            c for c in vz_crossings
            if c[1] <= ground_z + flat_foot_ground_tolerance_mm
        ]
        if not strict and not relaxed:
            continue

        pool = strict or relaxed
        pick = min(pool, key=lambda c: c[1])[0]
        strikes.append(pick)

    # Min separation deduplication
    if len(strikes) > 1:
        cleaned = [strikes[0]]
        for s in strikes[1:]:
            if s - cleaned[-1] >= min_separation:
                cleaned.append(s)
        strikes = cleaned

    if toe_z is not None:
        strikes = _refine_flat_foot_hs_at_ics(
            heel_z, toe_z, fs, strikes,
            flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
            flat_foot_ic_check_window_frames=flat_foot_ic_check_window_frames,
            flat_foot_vz_prominence=flat_foot_vz_prominence,
            flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
            flat_foot_landing_max_frames=flat_foot_landing_max_frames,
            swing_height=swing_height, swing_distance=swing_distance,
            toe_swing_height=toe_swing_height,
            toe_swing_distance=toe_swing_distance)
        if len(strikes) > 1:
            cleaned = [strikes[0]]
            for s in strikes[1:]:
                if s - cleaned[-1] >= min_separation:
                    cleaned.append(s)
            strikes = cleaned
    return strikes


def _detect_toe_off(toe_z, fs, hs_frames=None, accel_peak_min=10000,
                       min_velocity_at_peak=100, ground_tolerance=100,
                       post_rise_frames=10, post_rise_min=15,
                       min_separation=40):
    """
    TO = first positive Az peak per swing cycle, with sustained motion check.

    If hs_frames is provided, search is bounded by ICs (previous IC to next
    swing peak) — more accurate, as it ignores trial-start transient swings.
    Otherwise falls back to swing-peak-bounded search (less robust at
    trial start).

    Az peaks may occur up to ground + 100 mm (post-roll forefoot push-off).
    Rise is accepted if either the short-window rise (post_rise_frames) meets
    post_rise_min, or the toe rises at least 50 mm by the upcoming swing peak.
    """
    velocity = _backward_diff(toe_z, 1/fs)
    accel = _backward_diff(velocity, 1/fs)
    n = len(toe_z)
    valid = toe_z[~np.isnan(toe_z)]
    if len(valid) == 0:
        return []
    ground_z = float(np.percentile(valid, 5))

    swing_peaks, _ = _find_peaks_nan_safe(toe_z, height=ground_z + 30,
                                             distance=80, prominence=20)
    az_peaks, _ = _find_peaks_nan_safe(accel, height=accel_peak_min, distance=8)

    to_frames = []
    for sp_idx, sp in enumerate(swing_peaks):
        # Search window for TO leading to this swing peak:
        # IC-bounded: previous IC to current swing peak (preferred)
        # Special case: first swing peak with no prior IC — search from trial start
        # Otherwise swing-peak-bounded: previous swing peak to current
        if hs_frames is not None:
            prev_hs_arr = [h for h in hs_frames if h < sp]
            if prev_hs_arr:
                look_back = prev_hs_arr[-1] + 5
            elif sp_idx == 0:
                # First swing peak with no prior IC: trial-start TO
                # (subject starts walking from standing posture)
                look_back = 0
            else:
                # Not first swing but no prior IC — unusual, skip
                continue
        else:
            prev_sp_arr = swing_peaks[swing_peaks < sp]
            look_back = prev_sp_arr[-1] + 5 if len(prev_sp_arr) > 0 else 0

        candidates = []
        for ap in az_peaks:
            if look_back <= ap < sp:
                if ap < 10:
                    if (hs_frames is None
                            or not any(h < 30 for h in hs_frames)):
                        continue
                if velocity[ap] <= min_velocity_at_peak:
                    continue
                # Az peak may occur after the toe has rolled off the ground.
                if toe_z[ap] > ground_z + 100:
                    continue
                # Accept slow risers: short-window rise OR rise to swing peak.
                check_idx = min(n - 1, ap + post_rise_frames)
                z_rise_short = toe_z[check_idx] - toe_z[ap]
                z_rise_to_peak = toe_z[sp] - toe_z[ap]
                if z_rise_short < post_rise_min and z_rise_to_peak < 50:
                    continue
                candidates.append(ap)
        if candidates:
            to_frames.append(min(candidates))  # earliest Az peak in swing

    if len(to_frames) > 1:
        cleaned = [to_frames[0]]
        for t in to_frames[1:]:
            if t - cleaned[-1] >= min_separation:
                cleaned.append(t)
        to_frames = cleaned

    # Fallback: one TO per HS–HS interval with no toe-off yet (swing-peak
    # windows often start at the *next* contact, missing push-off before it).
    if hs_frames is not None and len(hs_frames) >= 2:
        for i in range(len(hs_frames) - 1):
            hs_a, hs_b = int(hs_frames[i]), int(hs_frames[i + 1])
            if any(hs_a < t < hs_b for t in to_frames):
                continue
            look_back = hs_a + 5
            search_end = hs_b
            candidates = []
            for ap in az_peaks:
                if look_back <= ap < search_end:
                    if ap < 10:
                        if not any(h < 30 for h in hs_frames):
                            continue
                    if velocity[ap] <= min_velocity_at_peak:
                        continue
                    if toe_z[ap] > ground_z + 100:
                        continue
                    check_idx = min(n - 1, ap + post_rise_frames)
                    z_rise_short = toe_z[check_idx] - toe_z[ap]
                    z_rise_to_peak = (
                        np.nanmax(toe_z[ap:search_end]) - toe_z[ap]
                        if ap < search_end else 0.0)
                    if (np.isnan(z_rise_to_peak)
                            or (z_rise_short < post_rise_min
                                and z_rise_to_peak < 50)):
                        continue
                    candidates.append(ap)
            if candidates:
                to_frames.append(min(candidates))

    if len(to_frames) > 1:
        to_frames = sorted(set(to_frames))
        cleaned = [to_frames[0]]
        for t in to_frames[1:]:
            if t - cleaned[-1] >= min_separation:
                cleaned.append(t)
        to_frames = cleaned
    return to_frames


def _detect_toe_strike_candidates(toe_z, fs, ground_tolerance=20,
                                     pre_swing_min_height=50,
                                     pre_descent_frames=10,
                                     min_pre_descent_speed=200,
                                     accel_peak_min=10000, min_separation=30):
    """
    TS candidates per cycle: deepest Z sign-change with strict criteria.

    Tight criteria to reject stance-phase toe oscillations and accept only
    real swing-to-stance toe-first landings (typically obstacle-crossing
    lead foot recovery).
    """
    velocity = _backward_diff(toe_z, 1/fs)
    accel = _backward_diff(velocity, 1/fs)
    n = len(toe_z)
    valid = toe_z[~np.isnan(toe_z)]
    if len(valid) == 0:
        return []
    ground_z = float(np.percentile(valid, 5))

    swing_peaks, _ = _find_peaks_nan_safe(toe_z, height=ground_z + 30,
                                             distance=50, prominence=20)
    if len(swing_peaks) == 0:
        return []

    boundaries = list(swing_peaks) + [n]
    candidates_out = []
    for i in range(len(boundaries) - 1):
        cycle_start = boundaries[i]
        cycle_end = boundaries[i + 1]

        # Pre-swing height: real swing must reach at least pre_swing_min_height
        cycle_window = toe_z[cycle_start:min(cycle_start + 30, cycle_end)]
        if np.all(np.isnan(cycle_window)):
            continue
        max_swing_height = np.nanmax(cycle_window) - ground_z
        if max_swing_height < pre_swing_min_height:
            continue  # not a real swing, skip cycle

        candidates = []
        for j in range(max(cycle_start, pre_descent_frames), cycle_end):
            if np.isnan(velocity[j]):
                continue
            # Sharp transition: Vz[j-1] strongly negative (impact creates sharp Vz drop)
            if velocity[j-1] < -100:
                if toe_z[j] > ground_z + ground_tolerance:
                    continue
                pre = velocity[j - pre_descent_frames:j]
                if np.any(np.isnan(pre)):
                    continue
                if not (np.all(pre < 0) and
                        np.mean(np.abs(pre)) >= min_pre_descent_speed):
                    continue
                # Az peak nearby (impact)
                check_start = max(0, j - 3)
                check_end = min(n, j + 4)
                nearby_az = accel[check_start:check_end]
                if np.all(np.isnan(nearby_az)) or np.nanmax(nearby_az) < accel_peak_min:
                    continue
                candidates.append((j, toe_z[j]))

        if not candidates:
            continue
        best = min(candidates, key=lambda c: c[1])
        candidates_out.append(best[0])

    if len(candidates_out) > 1:
        cleaned = [candidates_out[0]]
        for s in candidates_out[1:]:
            if s - cleaned[-1] >= min_separation:
                cleaned.append(s)
        candidates_out = cleaned
    return candidates_out


def _find_preceding_heel_strike_frame(heel_z, ts, window, ground_tolerance_mm):
    """Earliest ground-level heel Vz sign-change in [ts - window, ts], or None."""
    if heel_z is None or ts < 1:
        return None
    valid = heel_z[~np.isnan(heel_z)]
    if len(valid) == 0:
        return None
    ground_z = float(np.percentile(valid, 5))
    for j in range(max(1, ts - window), ts):
        if np.isnan(heel_z[j]) or np.isnan(heel_z[j - 1]):
            continue
        if j >= 2 and not np.isnan(heel_z[j - 2]):
            vz_prev = heel_z[j - 1] - heel_z[j - 2]
        else:
            vz_prev = 0.0
        vz = heel_z[j] - heel_z[j - 1]
        if vz_prev < 0 and vz >= 0 and heel_z[j] <= ground_z + ground_tolerance_mm:
            return j
    return None


def _validate_toe_strikes(ts_candidates, hs_frames, max_gap_frames=50,
                            preceding_hs_window=5, heel_z=None,
                            hs_ground_tolerance_mm=40):
    """TS validation.

    A TS candidate is valid only if BOTH:
        1. A same-side HS follows within ``max_gap_frames`` frames (heel rocks
           down after the toe lands — classic toe-strike landing).
        2. No same-side HS occurred within ``preceding_hs_window`` frames
           BEFORE the TS candidate. If such a preceding HS exists, the
           candidate is the toe-down phase of that HS, not a separate TS.
           Preceding HS is checked against detected ``hs_frames`` and, when
           ``heel_z`` is supplied, any ground-level heel Vz sign-change in the
           same window (catches HS contacts missed by cycle-based heel detection).

    Returns
    -------
    valid_ts : list of int
        TS frames that pass both rules.
    hs_paired : set of int
        HS frames matched as the rocker following a valid TS (excluded from
        separate IC counting).
    """
    valid_ts = []
    hs_paired = set()
    for ts in ts_candidates:
        preceding = [h for h in hs_frames
                       if ts - preceding_hs_window <= h < ts]
        if not preceding:
            heel_frame = _find_preceding_heel_strike_frame(
                heel_z, ts, preceding_hs_window, hs_ground_tolerance_mm
            )
            if heel_frame is not None:
                preceding = [heel_frame]
        if preceding:
            continue
        following = [h for h in hs_frames if ts < h <= ts + max_gap_frames]
        if following:
            valid_ts.append(ts)
            hs_paired.add(following[0])
    return valid_ts, hs_paired


def _find_local_minima(signal, min_separation):
    valid = ~np.isnan(signal)
    if valid.sum() < 10:
        return np.array([], dtype=int)
    sig = signal.copy()
    sig[~valid] = np.nanmax(signal) + 1e6
    peaks, _ = find_peaks(-sig, distance=min_separation)
    peaks = peaks[valid[peaks]]
    return peaks


# =============================================================================
# Obstacle event detection
# =============================================================================

def detect_obstacle_events(markers, setup, events):
    walking_axis = setup.walking_axis
    direction = setup.walking_direction
    obstacle_pos = setup.obstacle_pos_on_axis

    if markers.get('LTOE') is None or markers.get('RTOE') is None:
        return

    def first_crossing(traj):
        pos = traj[:, walking_axis]
        for f in range(len(pos) - 1):
            if np.isnan(pos[f]) or np.isnan(pos[f+1]):
                continue
            if direction == +1:
                if pos[f] < obstacle_pos and pos[f+1] >= obstacle_pos:
                    return f + 1
            else:
                if pos[f] > obstacle_pos and pos[f+1] <= obstacle_pos:
                    return f + 1
        return None

    def crossing_for_side(side):
        toe_name = 'LTOE' if side == 'left' else 'RTOE'
        heel_name = 'LHEE' if side == 'left' else 'RHEE'
        toe = markers.get(toe_name)
        toe_frame = first_crossing(toe) if toe is not None else None
        if toe_frame is not None:
            return toe_frame, 'toe'
        heel = markers.get(heel_name)
        heel_frame = first_crossing(heel) if heel is not None else None
        if heel_frame is not None:
            logger.warning(
                f"detect_obstacle_events: {side} toe did not cross obstacle plane; "
                f"falling back to heel crossing at frame {heel_frame + 1}"
            )
            return heel_frame, 'heel'
        return None, None

    cands = []
    for side in ('left', 'right'):
        frame, marker = crossing_for_side(side)
        if frame is not None:
            cands.append((side, frame, marker))
    if not cands:
        logger.warning("No foot crossing detected")
        return
    cands.sort(key=lambda c: c[1])
    events.lead_foot_side = cands[0][0]
    events.lead_toe_crossing = cands[0][1]
    events.lead_crossing_marker = cands[0][2]
    if len(cands) > 1:
        events.trail_foot_side = cands[1][0]
        events.trail_toe_crossing = cands[1][1]
        events.trail_crossing_marker = cands[1][2]


# =============================================================================
# Stride parameter computation
# =============================================================================

def compute_stride_records(markers, events, setup, anthro,
                             min_stride_length_mm=200.0,
                             min_stride_time_s=0.4):
    fs = setup.sampling_rate
    walking_axis = setup.walking_axis
    ml_axis = setup.ml_axis
    leg_length_mm = anthro.leg_length_mm
    leg_length_m = leg_length_mm / 1000

    records = []

    for (side, hs_list, to_list, strike_list, heel_name,
         opp_hs_list, opp_to_list, opp_heel_name) in [
        ('left', events.left_hs, events.left_to, events.left_strike_types,
         'LHEE', events.right_hs, events.right_to, 'RHEE'),
        ('right', events.right_hs, events.right_to, events.right_strike_types,
         'RHEE', events.left_hs, events.left_to, 'LHEE'),
    ]:
        if len(hs_list) < 2:
            continue
        heel = markers.get(heel_name)
        opp_heel = markers.get(opp_heel_name)
        if heel is None or opp_heel is None:
            continue

        # Strike list should align with hs_list. If not, pad with 'unknown'.
        while len(strike_list) < len(hs_list):
            strike_list.append('unknown')

        for i in range(len(hs_list) - 1):
            hs_start = hs_list[i]
            hs_end = hs_list[i + 1]
            strike_at_start = strike_list[i] if i < len(strike_list) else 'unknown'

            tos_in = [t for t in to_list if hs_start < t < hs_end]
            if not tos_in:
                continue
            to_frame = tos_in[0]

            opp_hs_in = [h for h in opp_hs_list if hs_start < h < hs_end]
            opp_to_in = [t for t in opp_to_list if hs_start < t < hs_end]
            opp_hs = opp_hs_in[0] if opp_hs_in else None
            opp_to = opp_to_in[0] if opp_to_in else None

            stride_time = (hs_end - hs_start) / fs
            if stride_time < min_stride_time_s:
                continue

            stance_time = (to_frame - hs_start) / fs
            swing_time = (hs_end - to_frame) / fs
            stance_pct = stance_time / stride_time * 100
            swing_pct = swing_time / stride_time * 100

            ds1_pct = float('nan')
            ds2_pct = float('nan')
            ss_pct = float('nan')
            if opp_to is not None and opp_to > hs_start:
                ds1_pct = (opp_to - hs_start) / (hs_end - hs_start) * 100
            if opp_hs is not None and opp_hs < to_frame:
                ds2_pct = (to_frame - opp_hs) / (hs_end - hs_start) * 100
            if opp_to is not None and opp_hs is not None and opp_hs > opp_to:
                ss_pct = (opp_hs - opp_to) / (hs_end - hs_start) * 100

            if np.isnan(heel[hs_start, 0]) or np.isnan(heel[hs_end, 0]):
                continue
            stride_length = abs(heel[hs_end, walking_axis] -
                                  heel[hs_start, walking_axis])
            if stride_length < min_stride_length_mm:
                continue

            gait_speed_m_s = (stride_length / 1000) / stride_time

            # Step parameters: this stride's HS_start refers back to previous opposite HS
            prev_opp_hs = max([h for h in opp_hs_list if h < hs_start], default=None)
            step_length = None
            step_time = None
            step_width = None
            if prev_opp_hs is not None:
                if (not np.isnan(heel[hs_start, 0]) and
                    not np.isnan(opp_heel[prev_opp_hs, 0])):
                    step_length = float(abs(
                        heel[hs_start, walking_axis] -
                        opp_heel[prev_opp_hs, walking_axis]
                    ))
                    step_time = float((hs_start - prev_opp_hs) / fs)
                if (not np.isnan(heel[hs_start, ml_axis]) and
                    not np.isnan(opp_heel[hs_start, ml_axis])):
                    step_width = float(abs(
                        heel[hs_start, ml_axis] -
                        opp_heel[hs_start, ml_axis]
                    ))

            prev_ic: Optional[int] = None
            if prev_opp_hs is not None:
                prev_ic = int(prev_opp_hs)

            stride_length_norm = stride_length / leg_length_mm
            stride_time_norm = stride_time / np.sqrt(leg_length_m / G)
            gait_speed_norm = gait_speed_m_s / np.sqrt(G * leg_length_m)
            step_length_norm = (step_length / leg_length_mm
                                 if step_length is not None else None)
            step_time_norm = (step_time / np.sqrt(leg_length_m / G)
                                if step_time is not None else None)
            step_width_norm = (step_width / leg_length_mm
                                if step_width is not None else None)

            records.append(StrideRecord(
                stride_idx_in_trial=0,
                side=side,
                phase='unassigned',
                hs_start_frame=int(hs_start),
                hs_end_frame=int(hs_end),
                to_frame=int(to_frame),
                opp_hs_frame=int(opp_hs) if opp_hs is not None else None,
                opp_to_frame=int(opp_to) if opp_to is not None else None,
                prev_opposite_ic_frame=prev_ic,
                ic_start_strike_type=strike_at_start,
                stride_time_s=round(stride_time, 4),
                stride_length_mm=round(stride_length, 2),
                stance_pct=round(stance_pct, 3),
                swing_pct=round(swing_pct, 3),
                double_support_1_pct=(round(ds1_pct, 3)
                                       if not np.isnan(ds1_pct) else float('nan')),
                double_support_2_pct=(round(ds2_pct, 3)
                                       if not np.isnan(ds2_pct) else float('nan')),
                single_support_pct=(round(ss_pct, 3)
                                       if not np.isnan(ss_pct) else float('nan')),
                gait_speed_m_s=round(gait_speed_m_s, 4),
                step_length_mm=(round(step_length, 2)
                                  if step_length is not None else None),
                step_time_s=(round(step_time, 4)
                                if step_time is not None else None),
                step_width_mm=(round(step_width, 2)
                                  if step_width is not None else None),
                stride_length_norm=round(stride_length_norm, 4),
                stride_time_norm=round(stride_time_norm, 4),
                gait_speed_norm=round(gait_speed_norm, 4),
                step_length_norm=(round(step_length_norm, 4)
                                    if step_length_norm is not None else None),
                step_time_norm=(round(step_time_norm, 4)
                                  if step_time_norm is not None else None),
                step_width_norm=(round(step_width_norm, 4)
                                   if step_width_norm is not None else None),
            ))

    records.sort(key=lambda r: r.hs_start_frame)
    for i, r in enumerate(records):
        r.stride_idx_in_trial = i
    return records


# =============================================================================
# Phase classification
# =============================================================================

def assign_phases(records, events):
    """Per-leg phase classification (Option 2: lead/trail crossing strides separate)."""
    if events.lead_foot_side is None:
        for r in records:
            r.phase = 'unknown'
        return

    lead_side = events.lead_foot_side
    trail_side = events.trail_foot_side
    lead_cross = events.lead_toe_crossing
    trail_cross = events.trail_toe_crossing

    for r in records:
        if r.side == lead_side:
            cross_frame = lead_cross
            crossing_label = 'crossing_lead'
            crossing_marker = events.lead_crossing_marker
        elif r.side == trail_side:
            cross_frame = trail_cross
            crossing_label = 'crossing_trail'
            crossing_marker = events.trail_crossing_marker
        else:
            r.phase = 'unknown'
            continue

        if cross_frame is None:
            r.phase = 'unknown'
            continue

        if r.hs_start_frame <= cross_frame <= r.hs_end_frame:
            r.phase = crossing_label
            r.crossing_marker_used = crossing_marker
        elif r.hs_end_frame < cross_frame:
            r.phase = 'approach'
        else:
            r.phase = 'recovery'


# =============================================================================
# Obstacle parameters
# =============================================================================

def compute_obstacle_parameters(markers, events, setup):
    obs = ObstacleParameters(
        lead_foot_side=events.lead_foot_side,
        trail_foot_side=events.trail_foot_side,
    )
    if events.lead_toe_crossing is None:
        return obs

    obstacle_z = _get_obstacle_z(markers)

    # Toe clearance
    for which, cross_frame, attr in [
        ('lead', events.lead_toe_crossing, 'lead_toe_clearance_mm'),
        ('trail', events.trail_toe_crossing, 'trail_toe_clearance_mm'),
    ]:
        side = events.lead_foot_side if which == 'lead' else events.trail_foot_side
        if side is None or cross_frame is None or obstacle_z is None:
            continue
        toe_name = 'LTOE' if side == 'left' else 'RTOE'
        toe = markers.get(toe_name)
        if toe is None:
            continue
        if not np.isnan(toe[cross_frame, 2]):
            setattr(obs, attr, round(float(toe[cross_frame, 2] - obstacle_z), 2))

    # Foot placement before/after
    for which, side, cross_frame, before_attr, after_attr in [
        ('lead', events.lead_foot_side, events.lead_toe_crossing,
         'lead_step_before_dist_mm', 'lead_step_after_dist_mm'),
        ('trail', events.trail_foot_side, events.trail_toe_crossing,
         'trail_step_before_dist_mm', 'trail_step_after_dist_mm'),
    ]:
        if side is None or cross_frame is None:
            continue
        before_d = _last_toe_off_distance(markers, events, setup, side, cross_frame)
        after_d = _first_heel_strike_distance(markers, events, setup, side, cross_frame)
        if before_d is not None:
            setattr(obs, before_attr, round(before_d, 2))
        if after_d is not None:
            setattr(obs, after_attr, round(after_d, 2))

    if (obs.lead_step_before_dist_mm is not None and
        obs.lead_step_after_dist_mm is not None):
        obs.crossing_step_length_mm = round(
            obs.lead_step_before_dist_mm + obs.lead_step_after_dist_mm, 2)

    return obs


def _get_obstacle_z(markers):
    for pair in [('OBSTACLE_L', 'OBSTACLE_R'), ('OBSTACLE_TOP_L', 'OBSTACLE_TOP_R')]:
        if pair[0] in markers and pair[1] in markers:
            z_l = np.nanmean(markers[pair[0]][:, 2])
            z_r = np.nanmean(markers[pair[1]][:, 2])
            return float((z_l + z_r) / 2)
    return None


def _last_toe_off_distance(markers, events, setup, side, before_frame):
    to_list = events.left_to if side == 'left' else events.right_to
    toe_name = 'LTOE' if side == 'left' else 'RTOE'
    toe = markers.get(toe_name)
    if toe is None:
        return None
    last_to = max([t for t in to_list if t < before_frame], default=None)
    if last_to is None or np.isnan(toe[last_to, setup.walking_axis]):
        return None
    return float(abs(toe[last_to, setup.walking_axis] - setup.obstacle_pos_on_axis))


def _first_heel_strike_distance(markers, events, setup, side, after_frame):
    hs_list = events.left_hs if side == 'left' else events.right_hs
    heel_name = 'LHEE' if side == 'left' else 'RHEE'
    heel = markers.get(heel_name)
    if heel is None:
        return None
    next_hs = min([h for h in hs_list if h > after_frame], default=None)
    if next_hs is None or np.isnan(heel[next_hs, setup.walking_axis]):
        return None
    return float(abs(heel[next_hs, setup.walking_axis] - setup.obstacle_pos_on_axis))


# =============================================================================
# Outlier detection
# =============================================================================

def _outliers_iqr_per_subset(records, vars_to_check=None,
                                iqr_threshold=1.5, min_strides=5):
    if vars_to_check is None:
        vars_to_check = OUTLIER_VARS
    if len(records) < min_strides:
        return {}
    flags = {}
    for var in vars_to_check:
        values = np.array([getattr(r, var) for r in records])
        valid = ~np.isnan(values)
        if valid.sum() < min_strides:
            continue
        Q1, Q3 = np.percentile(values[valid], [25, 75])
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        low = Q1 - iqr_threshold * IQR
        high = Q3 + iqr_threshold * IQR
        for r in records:
            v = getattr(r, var)
            if np.isnan(v):
                continue
            if v < low or v > high:
                flags.setdefault(r.stride_idx_in_trial, []).append(f'iqr:{var}')
    return flags


def _outliers_physiological(records, group, bounds=None):
    if bounds is None:
        bounds = DEFAULT_PHYSIOLOGICAL_BOUNDS
    group_bounds = bounds.get(group)
    if group_bounds is None:
        logger.warning(f"No physiological bounds for group '{group}'")
        return {}
    flags = {}
    for r in records:
        for var, (lo, hi) in group_bounds.items():
            v = getattr(r, var, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if v < lo:
                flags.setdefault(r.stride_idx_in_trial, []).append(
                    f'phys:{var}_low')
            elif v > hi:
                flags.setdefault(r.stride_idx_in_trial, []).append(
                    f'phys:{var}_high')
    return flags


def apply_outlier_flags(records, group, iqr_threshold=1.5,
                          min_strides_for_iqr=5, bounds=None):
    """Apply Stage 1 (IQR per phase x side) and Stage 3 (physiological)."""
    phys = _outliers_physiological(records, group, bounds)
    iqr_all = {}
    for phase in ['approach', 'crossing_lead', 'crossing_trail', 'recovery']:
        for side in ['left', 'right']:
            sub = [r for r in records if r.phase == phase and r.side == side]
            sub_flags = _outliers_iqr_per_subset(sub, iqr_threshold=iqr_threshold,
                                                   min_strides=min_strides_for_iqr)
            for k, v in sub_flags.items():
                iqr_all.setdefault(k, []).extend(v)

    for r in records:
        all_flags = []
        all_flags.extend(iqr_all.get(r.stride_idx_in_trial, []))
        all_flags.extend(phys.get(r.stride_idx_in_trial, []))
        r.outlier_flag = ';'.join(all_flags)


def is_severe_outlier(flag_string):
    """Severe = flagged in 2+ stages."""
    if not flag_string or pd.isna(flag_string):
        return False
    flags = str(flag_string).split(';')
    stages = set()
    for f in flags:
        if f.startswith('iqr:'):
            stages.add('iqr')
        elif f.startswith('trial_dev:'):
            stages.add('trial_dev')
        elif f.startswith('phys:'):
            stages.add('phys')
    return len(stages) >= 2


# =============================================================================
# Stage 2: Trial-deviation outlier (across-trial within subject)
# =============================================================================

def detect_trial_deviation_outliers(per_trial_phase_df, sd_threshold=2.0,
                                       vars_to_check=None):
    """
    For each subject x condition x phase x side, find trials whose mean is
    > sd_threshold SD from the across-trial mean.
    Returns DataFrame with added 'trial_outlier_flag' column.
    """
    if vars_to_check is None:
        vars_to_check = [f'{v}_mean' for v in OUTLIER_VARS]

    df = per_trial_phase_df.copy()
    df['trial_outlier_flag'] = ''

    grouping = ['subject_id', 'group', 'board', 'time', 'phase', 'side']
    grouping = [g for g in grouping if g in df.columns]

    for keys, sub in df.groupby(grouping):
        if len(sub) < 3:
            continue
        for var in vars_to_check:
            if var not in sub.columns:
                continue
            values = sub[var].dropna().values
            if len(values) < 3:
                continue
            mean_v = np.mean(values)
            sd_v = np.std(values, ddof=1)
            if sd_v == 0:
                continue
            for idx, row in sub.iterrows():
                v = row[var]
                if pd.isna(v):
                    continue
                if abs(v - mean_v) > sd_threshold * sd_v:
                    short_var = var.replace('_mean', '')
                    existing = df.at[idx, 'trial_outlier_flag'] or ''
                    new_flag = f'trial_dev:{short_var}'
                    df.at[idx, 'trial_outlier_flag'] = (
                        existing + (';' if existing else '') + new_flag)
    return df


# =============================================================================
# Per-trial pipeline
# =============================================================================

def analyze_trial(csv_path, metadata, anthro,
                    obstacle_marker_pair=('OBSTACLE_L', 'OBSTACLE_R'),
                    sampling_rate=100.0, filter_cutoff=6.0, apply_filter=True,
                    outlier_iqr_threshold=1.5, outlier_min_strides=5,
                    physiological_bounds=None):
    df, col_map, _ = load_marker_csv(csv_path)
    markers_raw = {}
    for name in col_map.keys():
        t = get_traj(df, col_map, name)
        if t is not None:
            markers_raw[name] = t

    setup = determine_setup(markers_raw, obstacle_marker_pair, sampling_rate)

    # Keep raw markers for event detection (filter shifts Az peak positions);
    # use filtered markers for spatial parameter computation.
    if apply_filter:
        markers = filter_markers(markers_raw, sampling_rate, filter_cutoff)
    else:
        markers = markers_raw

    # Event detection uses RAW markers (more accurate timing of events,
    # esp. toe-strike impacts which have sharp Vz transitions filter blurs).
    events = detect_gait_events_foot_based(markers, setup,
                                              raw_markers=markers_raw)
    detect_obstacle_events(markers, setup, events)

    records = compute_stride_records(markers, events, setup, anthro)
    assign_phases(records, events)
    obstacle = compute_obstacle_parameters(markers, events, setup)

    apply_outlier_flags(records, metadata.group,
                         iqr_threshold=outlier_iqr_threshold,
                         min_strides_for_iqr=outlier_min_strides,
                         bounds=physiological_bounds)
    return records, obstacle, events, setup


# =============================================================================
# DataFrame builders
# =============================================================================

def stride_records_to_df(records, metadata):
    rows = []
    for r in records:
        d = asdict(r)
        d.update({
            'subject_id': metadata.subject_id,
            'group': metadata.group,
            'board': metadata.board,
            'time': metadata.time,
            'trial': metadata.trial,
        })
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    front = ['subject_id', 'group', 'board', 'time', 'trial',
              'stride_idx_in_trial', 'side', 'phase',
              'ic_start_strike_type', 'crossing_marker_used',
              'hs_start_frame', 'hs_end_frame', 'to_frame',
              'opp_hs_frame', 'opp_to_frame']
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def stride_records_to_per_step_df(records, metadata):
    """
    One row per initial contact (IC) at ``hs_start_frame``: step time/length/width
    from the previous opposite-foot IC (when available).
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "subject_id": metadata.subject_id,
                "group": metadata.group,
                "board": metadata.board,
                "time": metadata.time,
                "trial": metadata.trial,
                "landing_side": r.side,
                "ic_frame": int(r.hs_start_frame),
                "ic_strike_type": r.ic_start_strike_type,
                "prev_opposite_ic_frame": r.prev_opposite_ic_frame,
                "step_time_s": r.step_time_s,
                "step_length_mm": r.step_length_mm,
                "step_width_mm": r.step_width_mm,
                "step_length_norm": r.step_length_norm,
                "step_time_norm": r.step_time_norm,
                "step_width_norm": r.step_width_norm,
                "phase": r.phase,
                "stride_idx_in_trial": int(r.stride_idx_in_trial),
                "outlier_flag": r.outlier_flag,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("ic_frame", kind="mergesort").reset_index(drop=True)
    df.insert(5, "step_idx_in_trial", np.arange(len(df), dtype=np.int64))
    front = [
        "subject_id",
        "group",
        "board",
        "time",
        "trial",
        "step_idx_in_trial",
        "ic_frame",
        "landing_side",
        "ic_strike_type",
        "prev_opposite_ic_frame",
        "step_time_s",
        "step_length_mm",
        "step_width_mm",
        "step_length_norm",
        "step_time_norm",
        "step_width_norm",
        "phase",
        "stride_idx_in_trial",
        "outlier_flag",
    ]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def obstacle_to_row(obstacle, metadata, leg_length_mm):
    row = {
        'subject_id': metadata.subject_id,
        'group': metadata.group,
        'board': metadata.board,
        'time': metadata.time,
        'trial': metadata.trial,
        'lead_foot_side': obstacle.lead_foot_side,
        'trail_foot_side': obstacle.trail_foot_side,
        'lead_toe_clearance_mm': obstacle.lead_toe_clearance_mm,
        'trail_toe_clearance_mm': obstacle.trail_toe_clearance_mm,
        'lead_step_before_dist_mm': obstacle.lead_step_before_dist_mm,
        'lead_step_after_dist_mm': obstacle.lead_step_after_dist_mm,
        'trail_step_before_dist_mm': obstacle.trail_step_before_dist_mm,
        'trail_step_after_dist_mm': obstacle.trail_step_after_dist_mm,
        'crossing_step_length_mm': obstacle.crossing_step_length_mm,
    }
    for k, k_norm in [
        ('lead_toe_clearance_mm', 'lead_toe_clearance_norm'),
        ('trail_toe_clearance_mm', 'trail_toe_clearance_norm'),
        ('lead_step_before_dist_mm', 'lead_step_before_norm'),
        ('lead_step_after_dist_mm', 'lead_step_after_norm'),
        ('trail_step_before_dist_mm', 'trail_step_before_norm'),
        ('trail_step_after_dist_mm', 'trail_step_after_norm'),
        ('crossing_step_length_mm', 'crossing_step_length_norm'),
    ]:
        v = row[k]
        row[k_norm] = round(v / leg_length_mm, 4) if v is not None else None
    return row


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_per_trial_phase(per_stride_df, exclusion_policy='exclude_severe'):
    df = per_stride_df.copy()
    if 'outlier_flag' in df.columns:
        if exclusion_policy == 'exclude_any':
            df_clean = df[df['outlier_flag'].fillna('') == '']
        elif exclusion_policy == 'exclude_severe':
            severe = df['outlier_flag'].fillna('').apply(is_severe_outlier)
            df_clean = df[~severe]
        else:
            df_clean = df
    else:
        df_clean = df

    df_clean = df_clean[df_clean['phase'].isin(['approach', 'recovery'])]

    grouping = ['subject_id', 'group', 'board', 'time', 'trial', 'phase', 'side']
    rows = []
    for keys, sub in df_clean.groupby(grouping):
        # Count flagged in original (before exclusion)
        mask = pd.Series(True, index=df.index)
        for col, val in zip(grouping, keys):
            mask &= (df[col] == val)
        orig = df[mask]
        n_flagged = ((orig['outlier_flag'].fillna('') != '').sum()
                       if 'outlier_flag' in orig.columns else 0)

        row = dict(zip(grouping, keys))
        row['n_strides'] = len(sub)
        row['n_strides_flagged'] = int(n_flagged)
        for var in VARS_TO_AGGREGATE:
            if var in sub.columns:
                values = sub[var].dropna()
                row[f'{var}_mean'] = (round(values.mean(), 4)
                                       if len(values) > 0 else None)
                row[f'{var}_sd'] = (round(values.std(), 4)
                                     if len(values) > 1 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_subject(per_trial_phase_df,
                            apply_trial_deviation_outlier=True,
                            sd_threshold=2.0):
    df = per_trial_phase_df.copy()
    if apply_trial_deviation_outlier and len(df) > 0:
        df = detect_trial_deviation_outliers(df, sd_threshold=sd_threshold)
        # Severe = 2+ trial_dev flags
        df['trial_n_flags'] = df['trial_outlier_flag'].fillna('').apply(
            lambda s: len([f for f in s.split(';') if f]) if s else 0)
        df_clean = df[df['trial_n_flags'] < 2]
    else:
        df_clean = df

    grouping = ['subject_id', 'group', 'board', 'time', 'phase', 'side']
    rows = []
    for keys, sub in df_clean.groupby(grouping):
        row = dict(zip(grouping, keys))
        row['n_trials'] = len(sub)
        row['n_strides_total'] = (int(sub['n_strides'].sum())
                                    if 'n_strides' in sub.columns else 0)
        for var in VARS_TO_AGGREGATE:
            col = f'{var}_mean'
            if col in sub.columns:
                values = sub[col].dropna()
                row[f'{var}_mean'] = (round(values.mean(), 4)
                                       if len(values) > 0 else None)
                row[f'{var}_sd'] = (round(values.std(), 4)
                                     if len(values) > 1 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_obstacle_per_subject(per_trial_obstacle_df):
    grouping = ['subject_id', 'group', 'board', 'time']
    obstacle_vars = [
        'lead_toe_clearance_mm', 'trail_toe_clearance_mm',
        'lead_step_before_dist_mm', 'lead_step_after_dist_mm',
        'trail_step_before_dist_mm', 'trail_step_after_dist_mm',
        'crossing_step_length_mm',
        'lead_toe_clearance_norm', 'trail_toe_clearance_norm',
        'lead_step_before_norm', 'lead_step_after_norm',
        'trail_step_before_norm', 'trail_step_after_norm',
        'crossing_step_length_norm',
    ]
    rows = []
    for keys, sub in per_trial_obstacle_df.groupby(grouping):
        row = dict(zip(grouping, keys))
        row['n_trials'] = len(sub)
        for var in obstacle_vars:
            if var in sub.columns:
                values = sub[var].dropna()
                row[f'{var}_mean'] = (round(values.mean(), 4)
                                       if len(values) > 0 else None)
                row[f'{var}_sd'] = (round(values.std(), 4)
                                     if len(values) > 1 else None)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Multi-trial pipeline
# =============================================================================

def analyze_trials(trial_specs, output_dir,
                     obstacle_marker_pair=('OBSTACLE_L', 'OBSTACLE_R'),
                     sampling_rate=100.0, filter_cutoff=6.0, apply_filter=True,
                     exclusion_policy='exclude_severe',
                     trial_deviation_sd_threshold=2.0,
                     outlier_iqr_threshold=1.5,
                     outlier_min_strides_for_iqr=5,
                     physiological_bounds=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_strides = []
    all_steps = []
    obstacle_rows = []

    for spec in trial_specs:
        try:
            records, obstacle, _, _ = analyze_trial(
                csv_path=spec['csv_path'],
                metadata=spec['metadata'],
                anthro=spec['anthropometry'],
                obstacle_marker_pair=obstacle_marker_pair,
                sampling_rate=sampling_rate,
                filter_cutoff=filter_cutoff,
                apply_filter=apply_filter,
                outlier_iqr_threshold=outlier_iqr_threshold,
                outlier_min_strides=outlier_min_strides_for_iqr,
                physiological_bounds=physiological_bounds,
            )
        except Exception as e:
            logger.error(f"Failed: {spec['csv_path']}: {e}")
            continue
        stride_df = stride_records_to_df(records, spec['metadata'])
        all_strides.append(stride_df)
        all_steps.append(stride_records_to_per_step_df(records, spec['metadata']))
        obstacle_rows.append(obstacle_to_row(obstacle, spec['metadata'],
                                                spec['anthropometry'].leg_length_mm))

    per_stride_df = (pd.concat(all_strides, ignore_index=True)
                     if all_strides else pd.DataFrame())
    per_stride_path = output_dir / 'per_stride_data.csv'
    per_stride_df.to_csv(per_stride_path, index=False)

    per_step_df = (pd.concat(all_steps, ignore_index=True)
                   if all_steps else pd.DataFrame())
    per_step_path = output_dir / 'per_step_data.csv'
    per_step_df.to_csv(per_step_path, index=False)

    per_trial_phase_df = aggregate_per_trial_phase(
        per_stride_df, exclusion_policy=exclusion_policy)
    per_trial_phase_path = output_dir / 'per_trial_phase_summary.csv'
    per_trial_phase_df.to_csv(per_trial_phase_path, index=False)

    per_trial_obstacle_df = pd.DataFrame(obstacle_rows)
    per_trial_obstacle_path = output_dir / 'per_trial_obstacle.csv'
    per_trial_obstacle_df.to_csv(per_trial_obstacle_path, index=False)

    per_subject_df = aggregate_per_subject(
        per_trial_phase_df,
        apply_trial_deviation_outlier=True,
        sd_threshold=trial_deviation_sd_threshold,
    )
    per_subject_path = output_dir / 'per_subject_summary.csv'
    per_subject_df.to_csv(per_subject_path, index=False)

    per_subject_obstacle_df = aggregate_obstacle_per_subject(per_trial_obstacle_df)
    per_subject_obstacle_path = output_dir / 'per_subject_obstacle_summary.csv'
    per_subject_obstacle_df.to_csv(per_subject_obstacle_path, index=False)

    return {
        'per_stride': per_stride_path,
        'per_step': per_step_path,
        'per_trial_phase': per_trial_phase_path,
        'per_trial_obstacle': per_trial_obstacle_path,
        'per_subject': per_subject_path,
        'per_subject_obstacle': per_subject_obstacle_path,
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_manifest_trial_cell(value: Any) -> int:
    """Accept int/float/str with spaces; e.g. 5, 05, ' 06 '."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not np.isnan(value):
        return int(value)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        raise ValueError("empty or invalid trial cell")
    return int(float(s))


def _manifest_first_nonempty(row: "pd.Series", keys: tuple[str, ...], default: str) -> str:
    """First non-empty cell among optional column names (headers already lowercased)."""
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return str(default)


def _manifest_optional_float(row: "pd.Series", keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        try:
            return float(s)
        except ValueError:
            continue
    return None


def _manifest_normalize_group(s: str) -> str:
    low = str(s).strip().lower()
    if low in ("adults", "adult"):
        return "adult"
    if low in ("children", "child"):
        return "child"
    return str(s).strip()


def _run_manifest_batch_cli(args: "argparse.Namespace") -> None:
    """
    CSV manifest: one row per trial. Required columns: ``csv_path`` (or path /
    input / file) and ``trial`` (or trial_number).

    Optional per-row columns (override CLI defaults when present and non-empty):
    ``subject_id`` (or ``subject``), ``group``, ``board`` (or ``board_type``),
    ``time``, ``leg_length_mm`` (or ``leg_length``), ``mass_kg`` (or ``mass``),
    ``age_years`` (or ``age``), ``height_mm`` (or ``height``).

    If every row supplies ``leg_length_mm`` / ``leg_length``, ``--leg-length-mm``
    may be omitted. Otherwise ``--leg-length-mm`` is required as the fallback.

    The manifest must be a **file path** passed to ``--trial-manifest``; do not paste
    CSV lines into the shell (the shell will try to run them as commands).
    """
    man = pd.read_csv(args.trial_manifest)
    man.columns = [str(c).strip().lower() for c in man.columns]
    if "csv_path" not in man.columns and "path" in man.columns:
        man = man.rename(columns={"path": "csv_path"})
    if "input" in man.columns and "csv_path" not in man.columns:
        man = man.rename(columns={"input": "csv_path"})
    if "file" in man.columns and "csv_path" not in man.columns:
        man = man.rename(columns={"file": "csv_path"})
    if "trial" not in man.columns and "trial_number" in man.columns:
        man = man.rename(columns={"trial_number": "trial"})
    if "csv_path" not in man.columns or "trial" not in man.columns:
        raise SystemExit(
            "Trial manifest CSV must include columns 'csv_path' and 'trial' "
            "(aliases: path, input, file for the trajectory path; trial_number for trial)."
        )

    trial_specs: list[dict[str, Any]] = []
    for _, row in man.iterrows():
        p = row["csv_path"]
        if pd.isna(p) or str(p).strip() == "":
            continue
        tr = row["trial"]
        if pd.isna(tr):
            continue
        try:
            trial_num = _parse_manifest_trial_cell(tr)
        except (ValueError, TypeError) as e:
            raise SystemExit(f"Invalid trial value in manifest: {tr!r} ({e})") from e

        leg_mm = _manifest_optional_float(row, ("leg_length_mm", "leg_length"))
        if leg_mm is None:
            if args.leg_length_mm is None:
                raise SystemExit(
                    "Manifest row missing leg_length_mm (or leg_length); "
                    "add a column or pass --leg-length-mm as default."
                )
            leg_mm = float(args.leg_length_mm)

        sub = _manifest_first_nonempty(row, ("subject_id", "subject"), args.subject_id)
        grp_raw = _manifest_first_nonempty(row, ("group",), args.group)
        grp = _manifest_normalize_group(grp_raw)
        if grp not in ("adult", "child"):
            raise SystemExit(
                f"Invalid group {grp_raw!r} (use adult/child or adults/children). "
                f"Row csv_path={str(p).strip()!r}"
            )
        brd = _manifest_first_nonempty(row, ("board", "board_type"), args.board)
        tm = _manifest_first_nonempty(row, ("time",), args.time)

        mass = _manifest_optional_float(row, ("mass_kg", "mass"))
        if mass is None and args.mass_kg is not None:
            mass = float(args.mass_kg)
        age = _manifest_optional_float(row, ("age_years", "age"))
        if age is None and args.age_years is not None:
            age = float(args.age_years)
        height = _manifest_optional_float(row, ("height_mm", "height"))
        if height is None and args.height_mm is not None:
            height = float(args.height_mm)

        trial_specs.append(
            {
                "csv_path": str(p).strip(),
                "metadata": TrialMetadata(
                    subject_id=sub,
                    group=grp,
                    board=brd,
                    time=tm,
                    trial=trial_num,
                ),
                "anthropometry": Anthropometry(
                    leg_length_mm=float(leg_mm),
                    mass_kg=mass,
                    age_years=age,
                    height_mm=height,
                ),
            }
        )
    if not trial_specs:
        raise SystemExit("No valid rows in trial manifest.")

    paths = analyze_trials(
        trial_specs=trial_specs,
        output_dir=str(args.output_dir),
        obstacle_marker_pair=tuple(args.obstacle_pair),
        sampling_rate=float(args.sampling_rate),
        filter_cutoff=float(args.filter_cutoff),
        apply_filter=not bool(args.no_filter),
        exclusion_policy=str(args.exclusion_policy),
        trial_deviation_sd_threshold=float(args.trial_deviation_sd_threshold),
        outlier_iqr_threshold=float(args.outlier_iqr_threshold),
        outlier_min_strides_for_iqr=int(args.outlier_min_strides),
        physiological_bounds=None,
    )
    print("Analysis complete. Files written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def _run_single_trial_csv_cli(args: "argparse.Namespace") -> None:
    """Marker trajectory CSV in → per-stride table CSV out (one trial)."""
    metadata = TrialMetadata(
        subject_id=args.subject_id,
        group=args.group,
        board=args.board,
        time=args.time,
        trial=int(args.trial),
    )
    anthro = Anthropometry(
        leg_length_mm=float(args.leg_length_mm),
        mass_kg=args.mass_kg,
        age_years=args.age_years,
        height_mm=args.height_mm,
    )
    records, obstacle, _events, _setup = analyze_trial(
        csv_path=args.input,
        metadata=metadata,
        anthro=anthro,
        obstacle_marker_pair=tuple(args.obstacle_pair),
        sampling_rate=float(args.sampling_rate),
        filter_cutoff=float(args.filter_cutoff),
        apply_filter=not bool(args.no_filter),
        outlier_iqr_threshold=float(args.outlier_iqr_threshold),
        outlier_min_strides=int(args.outlier_min_strides),
    )
    stride_df = stride_records_to_df(records, metadata)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stride_df.to_csv(out_path, index=False)
    print(f"Wrote per-stride table: {out_path.resolve()}")

    step_path = (
        Path(args.output_step)
        if args.output_step is not None
        else out_path.with_name(f"{out_path.stem}_per_step{out_path.suffix}")
    )
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_df = stride_records_to_per_step_df(records, metadata)
    step_df.to_csv(step_path, index=False)
    print(f"Wrote per-step table: {step_path.resolve()}")
    if args.output_obstacle:
        obs_path = Path(args.output_obstacle)
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        row = obstacle_to_row(obstacle, metadata, anthro.leg_length_mm)
        pd.DataFrame([row]).to_csv(obs_path, index=False)
        print(f"Wrote per-trial obstacle row: {obs_path.resolve()}")


def _main():
    """
    CLI modes:

    1) Batch (JSON spec) — same as before::

        python spatiotemporal.py analysis_spec.json

    2) Single trial (CSV in, CSV out)::

        python spatiotemporal.py --input trial.csv --output strides.csv \\
            --leg-length-mm 850 --subject-id S01 --group adult --board RB \\
            --time pre --trial 1

    3) Many trials (manifest CSV + output dir)::

        python spatiotemporal.py --trial-manifest trials.csv --output-dir ./out \\
            --leg-length-mm 960 --subject-id BBA01 --group adult --board RB --time pre

        trials.csv: required ``csv_path`` + ``trial`` (aliases: path, input, file;
        ``trial_number`` for trial). Optional per-row columns override CLI defaults:
        ``subject_id``, ``group``, ``board``, ``time``, ``leg_length_mm``,
        ``mass_kg``, ``age_years``, ``height_mm``. If every row has ``leg_length_mm``,
        ``--leg-length-mm`` may be omitted.

    JSON spec shape:
    {
        "output_dir": "./output",
        "sampling_rate": 100.0,
        "filter_cutoff": 6.0,
        "apply_filter": true,
        "exclusion_policy": "exclude_severe",
        "trials": [
            {
                "csv_path": "data/subject01/RB_pre/trial_1.csv",
                "subject_id": "subject01",
                "group": "adult",
                "board": "RB",
                "time": "pre",
                "trial": 1,
                "leg_length_mm": 850
            }, ...
        ]
    }
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Spatiotemporal gait parameters: JSON batch, CSV manifest batch, "
            "or single-trial CSV."
        ),
    )
    parser.add_argument(
        "spec_json",
        nargs="?",
        default=None,
        help="Path to batch spec JSON (omit when using --input/--output).",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="CSV",
        help="Labeled marker trajectory CSV (must include frame + *_x/y/z columns).",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="CSV",
        help="Write per-stride results to this CSV path.",
    )
    parser.add_argument(
        "--trial-manifest",
        metavar="CSV",
        default=None,
        help=(
            "Batch: path to a CSV with csv_path and trial per row (aliases: path, input, file; "
            "trial_number for trial). Optional columns subject_id, group, board, time, "
            "leg_length_mm, mass_kg, age_years, height_mm override CLI defaults when set. "
            "Omit --leg-length-mm if every row supplies leg_length_mm. "
            "Use with --output-dir; do not paste CSV into the shell."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Batch (manifest or JSON): directory for aggregate output CSVs.",
    )
    parser.add_argument(
        "--output-step",
        metavar="CSV",
        default=None,
        help=(
            "Single-trial: per-step CSV path (default: next to --output as "
            "<output_stem>_per_step.csv)."
        ),
    )
    parser.add_argument(
        "--leg-length-mm",
        type=float,
        default=None,
        help=(
            "Leg length (mm) for normalization. Required with --input/--output. "
            "With --trial-manifest: required unless every row has leg_length_mm (or leg_length); "
            "otherwise used as fallback for rows missing that column."
        ),
    )
    parser.add_argument("--subject-id", default="unknown")
    parser.add_argument("--group", default="adult", choices=("adult", "child"))
    parser.add_argument("--board", default="RB")
    parser.add_argument("--time", default="pre")
    parser.add_argument(
        "--trial",
        "--Trial",
        type=int,
        default=1,
        dest="trial",
        help="Trial index / number (same as --Trial).",
    )
    parser.add_argument("--mass-kg", type=float, default=None, dest="mass_kg")
    parser.add_argument("--age-years", type=float, default=None, dest="age_years")
    parser.add_argument("--height-mm", type=float, default=None, dest="height_mm")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--filter-cutoff", type=float, default=6.0)
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable Butterworth filtering for spatial trajectories.",
    )
    parser.add_argument(
        "--obstacle-pair",
        nargs=2,
        metavar=("L", "R"),
        default=("OBSTACLE_L", "OBSTACLE_R"),
        dest="obstacle_pair",
        help="Obstacle marker names for setup (default OBSTACLE_L OBSTACLE_R).",
    )
    parser.add_argument("--outlier-iqr-threshold", type=float, default=1.5)
    parser.add_argument("--outlier-min-strides", type=int, default=5)
    parser.add_argument(
        "--exclusion-policy",
        default="exclude_severe",
        choices=("exclude_severe", "exclude_any", "none"),
        help="Stride exclusion when aggregating phases (manifest / JSON batch).",
    )
    parser.add_argument(
        "--trial-deviation-sd-threshold",
        type=float,
        default=2.0,
        help="Across-trial mean deviation threshold for subject-level aggregation.",
    )

    args = parser.parse_args()

    if args.trial_manifest is not None:
        if args.spec_json is not None:
            parser.error("Do not pass a spec JSON path together with --trial-manifest.")
        if args.input is not None or args.output is not None:
            parser.error("Do not combine --trial-manifest with --input/--output.")
        if args.output_dir is None:
            parser.error("--output-dir is required with --trial-manifest.")
        _run_manifest_batch_cli(args)
        return

    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            parser.error("--input and --output must be used together.")
        if args.spec_json is not None:
            parser.error("Do not pass a spec JSON path together with --input/--output.")
        if args.leg_length_mm is None:
            parser.error("--leg-length-mm is required when using --input/--output.")
        _run_single_trial_csv_cli(args)
        return

    if args.spec_json is None:
        parser.error(
            "Provide a batch spec JSON path, --trial-manifest + --output-dir, "
            "or --input + --output for single-trial CSV."
        )

    with open(args.spec_json) as f:
        spec = json.load(f)

    trial_specs = []
    for t in spec["trials"]:
        trial_specs.append(
            {
                "csv_path": t["csv_path"],
                "metadata": TrialMetadata(
                    subject_id=t["subject_id"],
                    group=t["group"],
                    board=t["board"],
                    time=t["time"],
                    trial=t["trial"],
                ),
                "anthropometry": Anthropometry(
                    leg_length_mm=t["leg_length_mm"],
                    mass_kg=t.get("mass_kg"),
                    age_years=t.get("age_years"),
                    height_mm=t.get("height_mm"),
                ),
            }
        )

    paths = analyze_trials(
        trial_specs=trial_specs,
        output_dir=spec.get("output_dir", "."),
        sampling_rate=spec.get("sampling_rate", 100.0),
        filter_cutoff=spec.get("filter_cutoff", 6.0),
        apply_filter=spec.get("apply_filter", True),
        exclusion_policy=spec.get("exclusion_policy", "exclude_severe"),
        trial_deviation_sd_threshold=spec.get("trial_deviation_sd_threshold", 2.0),
        outlier_iqr_threshold=spec.get("outlier_iqr_threshold", 1.5),
        outlier_min_strides_for_iqr=spec.get("outlier_min_strides_for_iqr", 5),
    )
    print("Analysis complete. Files written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def main() -> None:
    """CLI entrypoint for ``python -m spatiotemporal`` and the ``spatiotemporal-gait`` script."""
    _main()
