"""Foot-based gait event detection (pelvis-independent)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks

from .models import GaitEvents, WalkingSetup

logger = logging.getLogger(__name__)

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


def _vicon_frame_at(index: int, frame_labels: np.ndarray | None) -> int:
    """Map row index to Vicon frame label; fall back to index if labels absent."""
    if frame_labels is not None and 0 <= index < len(frame_labels):
        return int(frame_labels[index])
    return int(index)


def _to_skip_early_transient_az(
    ap: int,
    hs_frames: list | None,
    frame_labels: np.ndarray | None,
    *,
    max_early_az_frame: int = 10,
    max_early_hs_frame: int = 30,
) -> bool:
    """True when an Az peak should be skipped as trial-start transient."""
    if _vicon_frame_at(ap, frame_labels) >= max_early_az_frame:
        return False
    if hs_frames is None:
        return True
    return not any(
        _vicon_frame_at(int(h), frame_labels) < max_early_hs_frame
        for h in hs_frames
    )


def detect_gait_events_foot_based(markers, setup, raw_markers=None,
                                     frame_labels: np.ndarray | None = None,
                                     swing_height_above_ground=80,
                                     swing_distance_frames=50,
                                     hs_ground_tolerance_mm=40,
                                     ts_ground_tolerance_mm=20,
                                     ts_pre_swing_min_height=50,
                                     min_separation_frames=40,
                                     pre_descent_frames=10,
                                     min_hs_pre_descent_speed=30,
                                     ts_pre_descent_frames=5,
                                     ts_pre_descent_min_negative_hits=3,
                                     ts_pre_descent_speed=200,
                                     ts_accel_peak_min=10000,
                                     to_accel_peak_min=10000,
                                     to_min_velocity_at_peak=100,
                                     to_az_peak_distance_frames=2,
                                     to_post_rise_frames=10,
                                     to_post_rise_min=15,
                                     flat_foot_toe_heel_mm=10.0,
                                     flat_foot_toe_heel_window_frames=7,
                                     flat_foot_toe_heel_min_hits=4,
                                     flat_foot_ic_pre_check_frames=6,
                                     flat_foot_ground_tolerance_mm=60.0,
                                     flat_foot_vz_prominence=80.0,
                                     flat_foot_swing_margin_frames=10,
                                     flat_foot_landing_max_frames=100,
                                     flat_foot_heel_lag_frames=8,
                                     flat_foot_post_hs_plateau_vz_mm_s=10.0,
                                     flat_foot_post_hs_plateau_frames=10,
                                     ts_validation_max_gap=50,
                                     preceding_hs_window=5,
                                     ts_post_ground_hold_enabled=True,
                                     ts_post_ground_tolerance_mm=20.0,
                                     ts_post_ground_min_frames=8,
                                     ts_post_max_early_rise_mm=8.0,
                                     ts_post_early_rise_frames=8):
    """
    Pelvis-independent gait event detection using Z-minimum priority algorithm.

    Uses RAW (unfiltered) marker data with backward difference for derivatives.
    This is critical because filtering shifts Az peak positions and can mask
    the sharp transitions characteristic of toe-strike impacts.

    HS Detection (per heel swing cycle):
        Default: DEEPEST Z sign-change (Vz neg→pos) with Z ≤ ground + 40 mm
        (biphasic heel-rocker: deepest crossing wins over an earlier shallow one).

        Flat-foot refinement (when toe Z is available): after normal HS per cycle,
        if the IC is flat-foot at contact, re-pick IC from descent-Vz peaks.
        Flat-foot requires BOTH (a) ≥ ``flat_foot_toe_heel_min_hits`` of
        ``flat_foot_toe_heel_window_frames`` ending at IC (default 4-of-7) have
        |toe−heel| ≤ ``flat_foot_toe_heel_mm`` (default 10 mm), AND (b) ≥
        ``flat_foot_pre_ic_min_hits`` of the
        ``flat_foot_pre_ic_check_frames`` frames before IC have
        ``flat_foot_pre_ic_vz_min ≤ heel Vz ≤ flat_foot_pre_ic_vz_max`` (default
        −100–0 mm/s for 3-of-5), capturing slow flat-foot approach before IC.
        When multiple near-ground Vz+ crossings exist and the deepest crossing
        is flat-foot, IC is the later of the toe/heel strongest descent-Vz peak
        (lowest Vz at peak) strictly between the first and deepest crossings.

    TO Detection (primary + optional gap fill):

        Primary — per-foot ``_detect_toe_off`` (IC/swing-bounded Az peaks with
        rise and post-Az Vz push-off checks; ``to_az_peak_distance_frames``=2).
        Multiple validated peaks in one window: **earliest** frame (first
        push-off spike; later spikes in the same burst are often roll/strike).

        Gap fill (only when needed) — for each **consecutive IC pair** in time,
        if no TO on the foot **opposite** the first IC lies strictly between
        those ICs, run the **same strict** TO picker in that window only. Still
        absent after gap fill is valid (e.g. jump).

    TS Detection (with strict criteria to reject stance-phase artifacts):
        Per cycle in toe Z, find deepest Z sign-change with:
        - Z at strike ≤ ground+20mm (must be at actual ground level)
        - Vz[strike-1] < -100 (toe still descending one frame before impact)
        - Pre-5 frames: ≥3 with Vz < 0, mean |Vz| ≥ 200 mm/s (real swing descent)
        - Pre-swing peak ≥ ground+50mm (real swing, not stance oscillation)
        - Az peak ≥10000 nearby (impact deceleration)
        - Validated against **pre-flat-foot-refine** HS frames so
          ``earlier_descent`` does not create a false preceding HS that blocks a
          toe-Az TS (e.g. TS at clearance minimum before the heel rocker).
          Requires (a) same-foot HS (pre-refine) follows within
          ``ts_validation_max_gap`` frames (default 50 = 500 ms at 100 Hz),
          (b) NO same-foot pre-refine HS within ``preceding_hs_window`` frames
          before (default 5 = 50 ms), and (c) when enabled, the toe stays near
          ground after the candidate without an immediate rise (rejects clearance
          dips). Condition (b) rejects the toe-down phase
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
        frame_labels: optional per-row Vicon frame numbers (same length as
            marker arrays). Used for trial-start TO filtering by Vicon frame,
            not array index.

    Returns: GaitEvents with HS, TO, TS frames + strike_types list parallel to
    hs lists. Strike type is 'HS' for heel-strike ICs or 'TS' for toe-strike ICs.
    """
    fs = setup.sampling_rate
    events = GaitEvents()

    # Use raw markers for event detection if available
    detect_markers = raw_markers if raw_markers is not None else markers

    # Per-side IC lists + hs_frames for contralateral TO gap fill.
    side_ic_data: dict[str, tuple[list[tuple[int, str]], np.ndarray, list]] = {}

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

        # HS: raw picks first; flat-foot refine applied after TS validation.
        hs_kw = dict(
            swing_height=swing_height_above_ground,
            swing_distance=swing_distance_frames,
            ground_tolerance=hs_ground_tolerance_mm,
            flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
            flat_foot_toe_heel_window_frames=flat_foot_toe_heel_window_frames,
            flat_foot_toe_heel_min_hits=flat_foot_toe_heel_min_hits,
            flat_foot_ic_pre_check_frames=flat_foot_ic_pre_check_frames,
            flat_foot_ground_tolerance_mm=flat_foot_ground_tolerance_mm,
            flat_foot_vz_prominence=flat_foot_vz_prominence,
            flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
            flat_foot_landing_max_frames=flat_foot_landing_max_frames,
            flat_foot_heel_lag_frames=flat_foot_heel_lag_frames,
            flat_foot_post_hs_plateau_vz_mm_s=flat_foot_post_hs_plateau_vz_mm_s,
            flat_foot_post_hs_plateau_frames=flat_foot_post_hs_plateau_frames,
            min_separation=min_separation_frames,
            pre_descent_frames=pre_descent_frames,
            min_pre_descent_speed=min_hs_pre_descent_speed,
        )
        hs_raw = _detect_heel_strike_raw(heel_z, fs, toe_z=toe_z, **hs_kw)
        hs_frames = (
            _refine_flat_foot_hs_at_ics(
                heel_z, toe_z, fs, hs_raw,
                flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
                flat_foot_toe_heel_window_frames=flat_foot_toe_heel_window_frames,
                flat_foot_toe_heel_min_hits=flat_foot_toe_heel_min_hits,
                flat_foot_ic_pre_check_frames=flat_foot_ic_pre_check_frames,
                flat_foot_vz_prominence=flat_foot_vz_prominence,
                flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
                flat_foot_landing_max_frames=flat_foot_landing_max_frames,
                flat_foot_heel_lag_frames=flat_foot_heel_lag_frames,
                flat_foot_post_hs_plateau_vz_mm_s=flat_foot_post_hs_plateau_vz_mm_s,
                flat_foot_post_hs_plateau_frames=flat_foot_post_hs_plateau_frames,
                swing_height=swing_height_above_ground,
                swing_distance=swing_distance_frames,
                toe_swing_height=50,
                toe_swing_distance=50,
            )
            if toe_z is not None else list(hs_raw)
        )
        if toe_z is not None and len(hs_frames) > 1:
            cleaned = [hs_frames[0]]
            for s in hs_frames[1:]:
                if s - cleaned[-1] >= min_separation_frames:
                    cleaned.append(s)
            hs_frames = cleaned
        # TO from HS/swing windows (strict Az + rise + post-Az Vz checks).
        to_frames = _detect_toe_off(
            toe_z, fs,
            hs_frames=hs_frames,
            frame_labels=frame_labels,
            accel_peak_min=to_accel_peak_min,
            min_velocity_at_peak=to_min_velocity_at_peak,
            post_rise_frames=to_post_rise_frames,
            post_rise_min=to_post_rise_min,
            min_separation=min_separation_frames,
            az_peak_distance=to_az_peak_distance_frames,
        )
        ts_candidates = _detect_toe_strike_candidates(
            toe_z, fs,
            ground_tolerance=ts_ground_tolerance_mm,
            pre_swing_min_height=ts_pre_swing_min_height,
            pre_descent_frames=ts_pre_descent_frames,
            min_pre_descent_negative_hits=ts_pre_descent_min_negative_hits,
            min_pre_descent_speed=ts_pre_descent_speed,
            accel_peak_min=ts_accel_peak_min,
            min_separation=min_separation_frames,
        )
        # Validate TS against pre-refine HS (toe-Az minimum before heel rocker).
        ts_valid, hs_paired_set = _validate_toe_strikes(
            ts_candidates, hs_raw,
            max_gap_frames=ts_validation_max_gap,
            preceding_hs_window=preceding_hs_window,
            heel_z=heel_z,
            toe_z=toe_z,
            fs=fs,
            hs_ground_tolerance_mm=hs_ground_tolerance_mm,
            post_ground_hold_enabled=ts_post_ground_hold_enabled,
            post_ground_tolerance_mm=ts_post_ground_tolerance_mm,
            post_ground_min_frames=ts_post_ground_min_frames,
            post_max_early_rise_mm=ts_post_max_early_rise_mm,
            post_early_rise_frames=ts_post_early_rise_frames,
        )
        hs_blocked_by_ts = _hs_refine_suppressed_by_valid_ts(
            ts_valid, hs_paired_set, hs_raw, hs_frames,
            ts_validation_max_gap=ts_validation_max_gap,
            preceding_hs_window=preceding_hs_window,
        )

        # Rejected TS with preceding heel contact: promote that heel frame to HS
        # and suppress the following HS that would have been the TS rocker.
        supplemental_hs = []
        suppressed_hs = set()
        for ts in ts_candidates:
            if ts in ts_valid:
                continue
            prec_hs = [h for h in hs_raw
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
            if any(abs(sh - int(h)) < min_separation_frames for h in hs_raw):
                continue
            supplemental_hs.append(sh)
            following = [h for h in hs_raw if ts < h <= ts + ts_validation_max_gap]
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
            if (hs not in hs_paired_set and hs not in suppressed_hs
                    and hs not in hs_blocked_by_ts and hs not in ic_frames_seen):
                ics_with_types.append((hs, 'HS'))
                ic_frames_seen.add(hs)
        for hs in supplemental_hs:
            if (hs not in hs_paired_set and hs not in suppressed_hs
                    and hs not in ic_frames_seen):
                ics_with_types.append((hs, 'HS'))
                ic_frames_seen.add(hs)
        ics_with_types.sort(key=lambda x: x[0])
        ics_with_types = _dedupe_nearby_ics(ics_with_types, min_separation_frames)

        side_ic_data[side] = (ics_with_types, toe_z, hs_frames)
        for f, t in ics_with_types:
            hs_list.append(int(f))
            strike_list.append(t)
        for to_f in to_frames:
            to_list.append(int(to_f))

    ic_stream: list[tuple[int, str]] = []
    for side in ('left', 'right'):
        if side in side_ic_data:
            for f, _ in side_ic_data[side][0]:
                ic_stream.append((int(f), side))
    ic_stream.sort(key=lambda x: x[0])

    left_toe = detect_markers.get('LTOE')
    right_toe = detect_markers.get('RTOE')
    if left_toe is not None and right_toe is not None and len(ic_stream) >= 2:
        hs_by_side = {
            s: side_ic_data[s][2] for s in ('left', 'right') if s in side_ic_data}
        _fill_missing_contralateral_to_between_consecutive_ics(
            left_toe[:, 2], right_toe[:, 2], fs, ic_stream,
            events.left_to, events.right_to,
            frame_labels=frame_labels,
            hs_by_side=hs_by_side,
            accel_peak_min=to_accel_peak_min,
            min_velocity_at_peak=to_min_velocity_at_peak,
            post_rise_frames=to_post_rise_frames,
            post_rise_min=to_post_rise_min,
            ic_min_gap=5,
            min_separation=min_separation_frames,
            az_peak_distance=to_az_peak_distance_frames,
        )

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


def _marker_descent_peaks_filtered(z, fs, win_start, win_end, *,
                                    prominence=80.0,
                                    other_z=None, max_toe_heel_mm=None):
    """Downward-Vz peak frames in [win_start, win_end), flat-filtered when requested."""
    if win_start >= win_end - 1:
        return []
    vz = _backward_diff(z, 1 / fs)
    peaks, _ = _find_peaks_nan_safe(
        -vz[win_start:win_end], prominence=prominence)
    if len(peaks) == 0:
        return []
    abs_peaks = [int(p) + win_start for p in peaks.astype(int)]
    if max_toe_heel_mm is not None and other_z is not None:
        flat_peaks = []
        for p in abs_peaks:
            lo = max(0, p - 2)
            hi = min(len(z), p + 3)
            diffs = np.abs(z[lo:hi] - other_z[lo:hi])
            if np.any(np.isfinite(diffs)) and float(np.nanmin(diffs)) <= max_toe_heel_mm:
                flat_peaks.append(p)
        abs_peaks = flat_peaks
    return sorted(abs_peaks)


def _marker_strongest_descent_peak(z, fs, win_start, win_end, *,
                                    prominence=80.0,
                                    other_z=None, max_toe_heel_mm=None):
    """Strongest downward-Vz peak (``find_peaks(-Vz)``) in [win_start, win_end)."""
    peaks = _marker_descent_peaks_filtered(
        z, fs, win_start, win_end, prominence=prominence,
        other_z=other_z, max_toe_heel_mm=max_toe_heel_mm)
    if not peaks:
        return None
    vz = _backward_diff(z, 1 / fs)
    return int(min(peaks, key=lambda p: float(vz[p])))


def _min_toe_heel_diff_at_frame(heel_z, toe_z, frame, *, pre_frames=5,
                                 post_frames=0):
    """Min |toe−heel| Z over [frame − pre_frames, frame + post_frames] (inclusive)."""
    n = len(heel_z)
    lo = max(0, int(frame) - pre_frames)
    hi = min(n, int(frame) + post_frames + 1)
    diffs = np.abs(toe_z[lo:hi] - heel_z[lo:hi])
    if not np.any(np.isfinite(diffs)):
        return float('inf')
    return float(np.nanmin(diffs))


def _toe_heel_within_threshold_hits(heel_z, toe_z, ic_frame, *,
                                      window_frames=7,
                                      max_toe_heel_mm=10.0):
    """Count frames in [IC − (window_frames − 1), IC] with |toe−heel| ≤ threshold."""
    if window_frames <= 0 or max_toe_heel_mm < 0:
        return 0
    ic = int(ic_frame)
    lo = max(0, ic - window_frames + 1)
    hi = ic + 1
    hits = 0
    for j in range(lo, hi):
        if not np.isfinite(heel_z[j]) or not np.isfinite(toe_z[j]):
            continue
        if float(abs(toe_z[j] - heel_z[j])) <= max_toe_heel_mm:
            hits += 1
    return hits


def _has_pre_ic_slow_heel_vz(heel_z, fs, ic_frame, *,
                              pre_check_frames=5,
                              min_hits=3,
                              vz_min_mm_s=-100.0,
                              vz_max_mm_s=0.0):
    """True when ≥``min_hits`` of ``pre_check_frames`` pre-IC heel Vz are in [vz_min, vz_max].

    Only frames strictly before ``ic_frame`` are counted (IC frame excluded).
    """
    if min_hits <= 0 or pre_check_frames <= 0:
        return False
    vz = _backward_diff(heel_z, 1 / fs)
    ic = int(ic_frame)
    lo = max(0, ic - pre_check_frames)
    if lo >= ic:
        return False
    hits = 0
    for j in range(lo, ic):
        v = vz[j]
        if np.isfinite(v) and vz_min_mm_s <= float(v) <= vz_max_mm_s:
            hits += 1
    return hits >= min_hits


def _is_flat_foot_at_ic(heel_z, toe_z, ic_frame, *, fs,
                         max_toe_heel_mm=10.0,
                         toe_heel_window_frames=7,
                         toe_heel_min_hits=4,
                         ic_pre_check_frames=6,
                         flat_foot_pre_ic_check_frames=5,
                         flat_foot_pre_ic_min_hits=3,
                         flat_foot_pre_ic_vz_min=-100.0,
                         flat_foot_pre_ic_vz_max=0.0,
                         post_hs_plateau_vz_mm_s=10.0,
                         post_hs_plateau_frames=10,
                         post_hs_plateau_search_frames=40):
    """True when flat at contact AND slow pre-IC heel approach.

    (1) ≥ ``toe_heel_min_hits`` of ``toe_heel_window_frames`` ending at IC have
        |toe−heel| ≤ ``max_toe_heel_mm``.
    (2) ≥ ``flat_foot_pre_ic_min_hits`` of ``flat_foot_pre_ic_check_frames`` frames
        before IC with ``flat_foot_pre_ic_vz_min ≤ heel Vz ≤ flat_foot_pre_ic_vz_max``.
    """
    del (post_hs_plateau_vz_mm_s, post_hs_plateau_frames,
         post_hs_plateau_search_frames, ic_pre_check_frames)
    if max_toe_heel_mm < 0:
        return False
    geom = (_toe_heel_within_threshold_hits(
        heel_z, toe_z, ic_frame,
        window_frames=toe_heel_window_frames,
        max_toe_heel_mm=max_toe_heel_mm,
    ) >= toe_heel_min_hits)
    if not geom:
        return False
    return _has_pre_ic_slow_heel_vz(
        heel_z, fs, ic_frame,
        pre_check_frames=flat_foot_pre_ic_check_frames,
        min_hits=flat_foot_pre_ic_min_hits,
        vz_min_mm_s=flat_foot_pre_ic_vz_min,
        vz_max_mm_s=flat_foot_pre_ic_vz_max,
    )


def _has_post_hs_small_vz_plateau(heel_z, fs, ic_frame, *,
                                   max_abs_vz_mm_s=10.0,
                                   min_consecutive_frames=10,
                                   search_frames=40):
    """True when heel Vz stays in [-max, 0] mm/s for N consecutive post-IC frames.

  Searches post-IC heel velocity (frames ic_frame+1 …) for at least
  ``min_consecutive_frames`` consecutive samples with
  ``-max_abs_vz_mm_s ≤ Vz ≤ 0`` (excludes small positive rocker rise).
    """
    if min_consecutive_frames <= 0 or max_abs_vz_mm_s < 0:
        return False
    vz = _backward_diff(heel_z, 1 / fs)
    start = int(ic_frame) + 1
    end = min(len(vz), start + search_frames)
    if start >= end:
        return False
    run = 0
    for j in range(start, end):
        v = vz[j]
        if np.isfinite(v) and (-max_abs_vz_mm_s <= float(v) <= 0.0):
            run += 1
            if run >= min_consecutive_frames:
                return True
        else:
            run = 0
    return False


def _toe_holds_ground_after_ts(toe_z, fs, ts_frame, *,
                               ground_tolerance_mm=20.0,
                               min_near_ground_frames=8,
                               max_early_rise_mm=8.0,
                               early_rise_frames=8):
    """True when toe stays near ground after TS without an immediate clearance rise."""
    if early_rise_frames <= 0 or min_near_ground_frames <= 0:
        return True
    valid = toe_z[~np.isnan(toe_z)]
    if len(valid) == 0:
        return False
    ground_z = float(np.percentile(valid, 5))
    n = len(toe_z)
    z0 = toe_z[int(ts_frame)]
    if not np.isfinite(z0):
        return False
    rise_end = min(n, int(ts_frame) + 1 + early_rise_frames)
    for j in range(int(ts_frame) + 1, rise_end):
        if not np.isfinite(toe_z[j]):
            return False
        if float(toe_z[j]) - float(z0) > max_early_rise_mm:
            return False
    start = int(ts_frame) + 1
    search_end = min(n, start + min_near_ground_frames + 20)
    run = 0
    for j in range(start, search_end):
        if np.isfinite(toe_z[j]) and float(toe_z[j]) <= ground_z + ground_tolerance_mm:
            run += 1
            if run >= min_near_ground_frames:
                return True
        else:
            run = 0
    return False


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
                                 flat_foot_toe_heel_window_frames=7,
                                 flat_foot_toe_heel_min_hits=4,
                                 flat_foot_ic_pre_check_frames=6,
                                 flat_foot_vz_prominence=80.0,
                                 flat_foot_swing_margin_frames=10,
                                 flat_foot_landing_max_frames=100,
                                 flat_foot_heel_lag_frames=8,
                                 flat_foot_post_hs_plateau_vz_mm_s=10.0,
                                 flat_foot_post_hs_plateau_frames=10,
                                 swing_height=80, swing_distance=50,
                                 toe_swing_height=50, toe_swing_distance=50):
    """
    Keep normal HS picks; re-pick with descent-Vz peaks when the IC is flat-foot
    at contact (|toe−heel| ≤ threshold and slow pre-IC heel Vz).
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
                heel_z, toe_z, hs, fs=fs,
                max_toe_heel_mm=flat_foot_toe_heel_mm,
                toe_heel_window_frames=flat_foot_toe_heel_window_frames,
                toe_heel_min_hits=flat_foot_toe_heel_min_hits,
                post_hs_plateau_vz_mm_s=flat_foot_post_hs_plateau_vz_mm_s,
                post_hs_plateau_frames=flat_foot_post_hs_plateau_frames):
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
            max_toe_heel_mm=flat_foot_toe_heel_mm,
            heel_lag_frames=flat_foot_heel_lag_frames)
        refined.append(int(descent) if descent is not None else int(hs))
    return refined


def _pick_flat_foot_ic_between_crossings(heel_z, toe_z, fs, crossing_a,
                                           crossing_b, *,
                                           vz_prominence=80.0,
                                           max_toe_heel_mm=10.0):
    """Between two heel Vz+ crossings: later of toe/heel strongest descent peak.

    Window is (min(crossing), max(crossing)) exclusive of the crossing frames.
    Per marker, pick the descent-Vz peak with the most negative heel/toe Vz;
    IC is the later of the two marker picks.
    """
    lo = int(min(crossing_a, crossing_b)) + 1
    hi = int(max(crossing_a, crossing_b))
    if lo >= hi:
        return None
    heel_p = _marker_strongest_descent_peak(
        heel_z, fs, lo, hi, prominence=vz_prominence,
        other_z=toe_z, max_toe_heel_mm=max_toe_heel_mm)
    toe_p = None
    if toe_z is not None:
        toe_p = _marker_strongest_descent_peak(
            toe_z, fs, lo, hi, prominence=vz_prominence,
            other_z=heel_z, max_toe_heel_mm=max_toe_heel_mm)
    picks = [p for p in (heel_p, toe_p) if p is not None]
    if not picks:
        return None
    return int(max(picks))


def _pick_flat_foot_ic_from_descent_peaks(heel_z, toe_z, fs, cycle_start,
                                           cycle_end, *, vz_prominence=80.0,
                                           swing_margin_frames=10,
                                           landing_max_frames=100,
                                           max_toe_heel_mm=10.0,
                                           heel_lag_frames=8):
    """
    Flat-foot IC: descent-Vz peaks on toe and heel in the early post-swing window
    (flat at peak). From the first toe descent through ``heel_lag_frames`` after
    the last toe peak, take the latest peak (later contact between toe and heel).
    """
    win_start = min(cycle_start + swing_margin_frames,
                    max(cycle_end - 1, cycle_start))
    win_end = min(cycle_start + landing_max_frames, cycle_end)
    if win_start >= win_end - 1:
        return None
    heel_peaks = _marker_descent_peaks_filtered(
        heel_z, fs, win_start, win_end, prominence=vz_prominence,
        other_z=toe_z, max_toe_heel_mm=max_toe_heel_mm)
    toe_peaks = []
    if toe_z is not None:
        toe_peaks = _marker_descent_peaks_filtered(
            toe_z, fs, win_start, win_end, prominence=vz_prominence,
            other_z=heel_z, max_toe_heel_mm=max_toe_heel_mm)
    all_peaks = sorted(set(toe_peaks + heel_peaks))
    if not all_peaks:
        return None
    if toe_peaks:
        lower = min(toe_peaks)
        upper = max(toe_peaks) + heel_lag_frames
        between = [p for p in all_peaks if lower <= p <= upper]
    else:
        between = heel_peaks
    if not between:
        return None
    return int(between[-1])


def _hs_refine_suppressed_by_valid_ts(ts_valid, hs_paired_set, hs_raw, hs_refined,
                                       *, ts_validation_max_gap=50,
                                       preceding_hs_window=5):
    """Suppress earlier-descent HS only for the landing paired with a valid TS.

    Only the refined frame that replaced the rocker HS (same index in raw/refine
    lists) is suppressed—not unrelated earlier HS in the trial.
    """
    del preceding_hs_window  # kept for API stability
    suppressed: set[int] = set()
    if not ts_valid or len(hs_raw) != len(hs_refined):
        return suppressed
    hs_raw_set = {int(h) for h in hs_raw}
    for ts in ts_valid:
        following = sorted(
            h for h in hs_raw_set
            if int(ts) < h <= int(ts) + ts_validation_max_gap)
        if not following:
            continue
        paired = following[0]
        for raw_hs, ref_hs in zip(hs_raw, hs_refined):
            raw_i, ref_i = int(raw_hs), int(ref_hs)
            if ref_i == raw_i or raw_i != paired:
                continue
            if ref_i < raw_i:
                suppressed.add(ref_i)
    return suppressed


def _detect_heel_strike_raw(heel_z, fs, toe_z=None, swing_height=80,
                               swing_distance=50, ground_tolerance=40,
                               flat_foot_ground_tolerance_mm=60.0,
                               min_separation=40, pre_descent_frames=10,
                               min_pre_descent_speed=30,
                               toe_swing_height=50, toe_swing_distance=50,
                               flat_foot_toe_heel_mm=10.0,
                               flat_foot_toe_heel_window_frames=7,
                               flat_foot_toe_heel_min_hits=4,
                               flat_foot_ic_pre_check_frames=6,
                               flat_foot_vz_prominence=80.0,
                               flat_foot_swing_margin_frames=10,
                               flat_foot_landing_max_frames=100,
                               flat_foot_heel_lag_frames=8,
                               flat_foot_post_hs_plateau_vz_mm_s=10.0,
                               flat_foot_post_hs_plateau_frames=10):
    """HS per swing cycle before flat-foot refinement (used for TS validation)."""
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
        deepest_pick = min(pool, key=lambda c: c[1])[0]
        if len(pool) >= 2 and toe_z is not None:
            first_pick = min(pool, key=lambda c: c[0])[0]
            ff_kw = dict(
                fs=fs,
                max_toe_heel_mm=flat_foot_toe_heel_mm,
                toe_heel_window_frames=flat_foot_toe_heel_window_frames,
                toe_heel_min_hits=flat_foot_toe_heel_min_hits,
            )
            if (_is_flat_foot_at_ic(heel_z, toe_z, deepest_pick, **ff_kw)
                    and first_pick != deepest_pick):
                between = _pick_flat_foot_ic_between_crossings(
                    heel_z, toe_z, fs, first_pick, deepest_pick,
                    vz_prominence=flat_foot_vz_prominence,
                    max_toe_heel_mm=flat_foot_toe_heel_mm)
                pick = (between if between is not None else first_pick)
            else:
                pick = deepest_pick
        else:
            pick = deepest_pick
        strikes.append(pick)

    # Min separation deduplication
    if len(strikes) > 1:
        cleaned = [strikes[0]]
        for s in strikes[1:]:
            if s - cleaned[-1] >= min_separation:
                cleaned.append(s)
        strikes = cleaned

    return strikes


def _detect_heel_strike_zmin(heel_z, fs, toe_z=None, swing_height=80,
                                swing_distance=50, ground_tolerance=40,
                                flat_foot_ground_tolerance_mm=60.0,
                                min_separation=40, pre_descent_frames=10,
                                min_pre_descent_speed=30,
                                toe_swing_height=50, toe_swing_distance=50,
                                flat_foot_toe_heel_mm=10.0,
                                flat_foot_toe_heel_window_frames=7,
                                flat_foot_toe_heel_min_hits=4,
                                flat_foot_ic_pre_check_frames=6,
                                flat_foot_vz_prominence=80.0,
                                flat_foot_swing_margin_frames=10,
                                flat_foot_landing_max_frames=100,
                                flat_foot_heel_lag_frames=8,
                                flat_foot_post_hs_plateau_vz_mm_s=10.0,
                                flat_foot_post_hs_plateau_frames=10):
    """
    HS per swing cycle: deepest near-ground heel Vz+ (normal IC).

    When ``toe_z`` is provided, applies ``_refine_flat_foot_hs_at_ics`` after
    raw cycle picks (see ``_detect_heel_strike_raw``).
    """
    strikes = _detect_heel_strike_raw(
        heel_z, fs, toe_z=toe_z, swing_height=swing_height,
        swing_distance=swing_distance, ground_tolerance=ground_tolerance,
        flat_foot_ground_tolerance_mm=flat_foot_ground_tolerance_mm,
        min_separation=min_separation, pre_descent_frames=pre_descent_frames,
        min_pre_descent_speed=min_pre_descent_speed,
        toe_swing_height=toe_swing_height, toe_swing_distance=toe_swing_distance,
        flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
        flat_foot_toe_heel_window_frames=flat_foot_toe_heel_window_frames,
        flat_foot_toe_heel_min_hits=flat_foot_toe_heel_min_hits,
        flat_foot_ic_pre_check_frames=flat_foot_ic_pre_check_frames,
        flat_foot_vz_prominence=flat_foot_vz_prominence,
        flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
        flat_foot_landing_max_frames=flat_foot_landing_max_frames,
        flat_foot_heel_lag_frames=flat_foot_heel_lag_frames,
        flat_foot_post_hs_plateau_vz_mm_s=flat_foot_post_hs_plateau_vz_mm_s,
        flat_foot_post_hs_plateau_frames=flat_foot_post_hs_plateau_frames,
    )
    if toe_z is not None:
        strikes = _refine_flat_foot_hs_at_ics(
            heel_z, toe_z, fs, strikes,
            flat_foot_toe_heel_mm=flat_foot_toe_heel_mm,
            flat_foot_toe_heel_window_frames=flat_foot_toe_heel_window_frames,
            flat_foot_toe_heel_min_hits=flat_foot_toe_heel_min_hits,
            flat_foot_ic_pre_check_frames=flat_foot_ic_pre_check_frames,
            flat_foot_vz_prominence=flat_foot_vz_prominence,
            flat_foot_swing_margin_frames=flat_foot_swing_margin_frames,
            flat_foot_landing_max_frames=flat_foot_landing_max_frames,
            flat_foot_heel_lag_frames=flat_foot_heel_lag_frames,
            flat_foot_post_hs_plateau_vz_mm_s=flat_foot_post_hs_plateau_vz_mm_s,
            flat_foot_post_hs_plateau_frames=flat_foot_post_hs_plateau_frames,
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


def _az_peak_followed_by_toe_down(
    velocity: np.ndarray,
    ap: int,
    *,
    post_window: int = 8,
    min_down_run: int = 4,
    down_vz_threshold: float = -50.0,
) -> bool:
    """
    True when ≥``min_down_run`` consecutive frames within ``post_window`` after
    the Az peak have Vz < ``down_vz_threshold`` (toe-down after the spike).
    """
    end = min(len(velocity), ap + post_window + 1)
    max_run = 0
    run = 0
    for i in range(ap + 1, end):
        v = velocity[i]
        if np.isfinite(v) and v < down_vz_threshold:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run >= min_down_run


def _az_peak_followed_by_vz_pushoff(
    velocity: np.ndarray,
    ap: int,
    *,
    early_window: int = 20,
    min_consecutive_positive: int = 5,
    max_negative_immediate: int = 1,
    immediate_frames: int = 3,
    min_post_vz_peak: float = 400.0,
) -> bool:
    """
    True when Az peak at ``ap`` is followed by sustained toe-rise Vz (push-off),
    not a one-frame bump at the bottom of a descent or plateau noise.

    When ``vz0`` at the Az frame is already ≥ ``min_post_vz_peak`` and at least
    ``min_consecutive_positive`` consecutive post-Az frames have Vz > 0, accept
    even if Vz does not grow after the spike (push-off often peaks with Az; Vz
    stays positive but decreases toward swing apex).

    Otherwise require a higher Vz peak strictly after the Az frame.
    """
    vz0 = velocity[ap]
    if not np.isfinite(vz0) or vz0 <= 0:
        return False
    end = min(len(velocity), ap + early_window + 1)
    neg_immediate = 0
    for i in range(ap + 1, min(end, ap + 1 + immediate_frames)):
        v = velocity[i]
        if np.isfinite(v) and v < 0:
            neg_immediate += 1
    if neg_immediate > max_negative_immediate:
        return False
    max_run = 0
    run = 0
    peak_post_vz = vz0
    peak_post_idx = ap
    for i in range(ap + 1, end):
        v = velocity[i]
        if not np.isfinite(v):
            run = 0
            continue
        if v > 0:
            run += 1
            max_run = max(max_run, run)
            if v > peak_post_vz:
                peak_post_vz = v
                peak_post_idx = i
        else:
            run = 0
    if max_run < min_consecutive_positive:
        return False
    if vz0 >= min_post_vz_peak:
        return True
    if peak_post_idx <= ap:
        return False
    if peak_post_vz < min_post_vz_peak:
        return False
    return peak_post_vz > vz0


def _toe_off_strict_candidates_in_window(
    toe_z,
    velocity,
    accel,
    az_peaks,
    ground_z,
    look_back: int,
    search_end: int,
    *,
    frame_labels=None,
    hs_frames=None,
    min_velocity_at_peak=100,
    post_rise_frames=10,
    post_rise_min=15,
    post_az_vz_early_window=20,
    post_az_min_consecutive_positive_vz=5,
    max_early_az_frame=10,
    max_early_hs_frame=30,
) -> list[int]:
    """Az peaks in [look_back, search_end) passing the same checks as ``_detect_toe_off``."""
    n = len(toe_z)
    candidates: list[int] = []
    for ap in az_peaks:
        if look_back <= ap < search_end:
            if _to_skip_early_transient_az(
                    int(ap), hs_frames, frame_labels,
                    max_early_az_frame=max_early_az_frame,
                    max_early_hs_frame=max_early_hs_frame):
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
            toe_down_after = _az_peak_followed_by_toe_down(velocity, ap)
            if z_rise_short < post_rise_min:
                if (np.isnan(z_rise_to_peak) or z_rise_to_peak < 50
                        or toe_down_after):
                    continue
            if toe_down_after:
                continue
            if not _az_peak_followed_by_vz_pushoff(
                    velocity, ap,
                    early_window=post_az_vz_early_window,
                    min_consecutive_positive=post_az_min_consecutive_positive_vz):
                continue
            candidates.append(int(ap))
    return candidates


def _pick_toe_off_strict_in_window(
    toe_z,
    fs,
    win_start: int,
    win_end: int,
    *,
    frame_labels=None,
    hs_frames=None,
    accel_peak_min=10000,
    min_velocity_at_peak=100,
    post_rise_frames=10,
    post_rise_min=15,
    post_az_vz_early_window=20,
    post_az_min_consecutive_positive_vz=5,
    max_early_az_frame=10,
    max_early_hs_frame=30,
    az_peak_distance=2,
) -> int | None:
    """One strict TO frame in [win_start, win_end), or None."""
    if win_start >= win_end:
        return None
    velocity = _backward_diff(toe_z, 1 / fs)
    accel = _backward_diff(velocity, 1 / fs)
    valid = toe_z[~np.isnan(toe_z)]
    if len(valid) == 0:
        return None
    ground_z = float(np.percentile(valid, 5))
    az_peaks, _ = _find_peaks_nan_safe(
        accel, height=accel_peak_min, distance=az_peak_distance)
    candidates = _toe_off_strict_candidates_in_window(
        toe_z, velocity, accel, az_peaks, ground_z,
        win_start, win_end,
        frame_labels=frame_labels,
        hs_frames=hs_frames,
        min_velocity_at_peak=min_velocity_at_peak,
        post_rise_frames=post_rise_frames,
        post_rise_min=post_rise_min,
        post_az_vz_early_window=post_az_vz_early_window,
        post_az_min_consecutive_positive_vz=post_az_min_consecutive_positive_vz,
        max_early_az_frame=max_early_az_frame,
        max_early_hs_frame=max_early_hs_frame,
    )
    if not candidates:
        return None
    return _pick_to_az_candidate(candidates, toe_z)


def _fill_missing_contralateral_to_between_consecutive_ics(
    left_toe_z,
    right_toe_z,
    fs,
    ic_stream,
    left_to_frames,
    right_to_frames,
    *,
    frame_labels=None,
    hs_by_side: dict[str, list] | None = None,
    accel_peak_min=10000,
    min_velocity_at_peak=100,
    post_rise_frames=10,
    post_rise_min=15,
    ic_min_gap=5,
    min_separation=40,
    post_az_vz_early_window=20,
    post_az_min_consecutive_positive_vz=5,
    max_early_az_frame=10,
    max_early_hs_frame=30,
    az_peak_distance=2,
) -> None:
    """
    Gap fill: only for consecutive IC pairs (IC_i, IC_{i+1}).

    If no TO on the foot contralateral to IC_i lies strictly between IC_i and
    IC_{i+1}, try strict ``_detect_toe_off`` criteria in that window. If none
    is found, leave the gap (e.g. jump).
    """
    if len(ic_stream) < 2:
        return

    toe_by_side = {'left': left_toe_z, 'right': right_toe_z}
    to_by_side = {'left': left_to_frames, 'right': right_to_frames}
    hs_by_side = hs_by_side or {}

    def _has_to_strictly_between(side: str, lo: int, hi: int) -> bool:
        return any(lo < t < hi for t in to_by_side[side])

    for i in range(len(ic_stream) - 1):
        ic_a, side_a = ic_stream[i]
        ic_b, _side_b = ic_stream[i + 1]
        opp_side = 'right' if side_a == 'left' else 'left'
        win_start = int(ic_a) + ic_min_gap
        win_end = int(ic_b)
        if win_start >= win_end:
            continue
        if _has_to_strictly_between(opp_side, ic_a, ic_b):
            continue

        pick = _pick_toe_off_strict_in_window(
            toe_by_side[opp_side],
            fs,
            win_start,
            win_end,
            frame_labels=frame_labels,
            hs_frames=hs_by_side.get(opp_side),
            accel_peak_min=accel_peak_min,
            min_velocity_at_peak=min_velocity_at_peak,
            post_rise_frames=post_rise_frames,
            post_rise_min=post_rise_min,
            post_az_vz_early_window=post_az_vz_early_window,
            post_az_min_consecutive_positive_vz=post_az_min_consecutive_positive_vz,
            max_early_az_frame=max_early_az_frame,
            max_early_hs_frame=max_early_hs_frame,
            az_peak_distance=az_peak_distance,
        )
        if pick is None:
            continue
        existing = to_by_side[opp_side]
        if any(abs(pick - t) < min_separation for t in existing):
            continue
        existing.append(int(pick))


def _pick_to_az_candidate(
    candidates: list[int],
    toe_z: np.ndarray,
) -> int:
    """
    Among validated Az peaks in one swing window, pick the earliest frame.

    TO is the first push-off spike in a burst; later peaks (often larger Az)
    tend to be roll-over or pre-strike transients in the same stance.
    """
    return min(candidates)


def _detect_toe_off(toe_z, fs, hs_frames=None, frame_labels=None,
                       accel_peak_min=10000,
                       min_velocity_at_peak=100, ground_tolerance=100,
                       post_rise_frames=10, post_rise_min=15,
                       min_separation=40,
                       post_az_vz_early_window=20,
                       post_az_min_consecutive_positive_vz=5,
                       max_early_az_frame=10,
                       max_early_hs_frame=30,
                       az_peak_distance=2):
    """
    TO = first positive Az peak per swing cycle, with sustained motion check.

    If hs_frames is provided, search is bounded by ICs (previous IC to next
    swing peak) — more accurate, as it ignores trial-start transient swings.
    Otherwise falls back to swing-peak-bounded search (less robust at
    trial start).

    Az peaks may occur up to ground + 100 mm (post-roll forefoot push-off).
    Rise is accepted if either the short-window rise (post_rise_frames) meets
    post_rise_min, or the toe rises at least 50 mm by the upcoming swing peak.
    After the Az peak, at least ``post_az_min_consecutive_positive_vz``
    consecutive frames must have Vz > 0 within ``post_az_vz_early_window`` frames
  (push-off before swing, not a plateau bump).
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
    az_peaks, _ = _find_peaks_nan_safe(
        accel, height=accel_peak_min, distance=az_peak_distance)

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

        candidates = _toe_off_strict_candidates_in_window(
            toe_z, velocity, accel, az_peaks, ground_z,
            look_back, min(n, sp + 1),
            frame_labels=frame_labels,
            hs_frames=hs_frames,
            min_velocity_at_peak=min_velocity_at_peak,
            post_rise_frames=post_rise_frames,
            post_rise_min=post_rise_min,
            post_az_vz_early_window=post_az_vz_early_window,
            post_az_min_consecutive_positive_vz=post_az_min_consecutive_positive_vz,
            max_early_az_frame=max_early_az_frame,
            max_early_hs_frame=max_early_hs_frame,
        )
        if candidates:
            to_frames.append(_pick_to_az_candidate(candidates, toe_z))

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
            candidates = _toe_off_strict_candidates_in_window(
                toe_z, velocity, accel, az_peaks, ground_z,
                look_back, search_end,
                frame_labels=frame_labels,
                hs_frames=hs_frames,
                min_velocity_at_peak=min_velocity_at_peak,
                post_rise_frames=post_rise_frames,
                post_rise_min=post_rise_min,
                post_az_vz_early_window=post_az_vz_early_window,
                post_az_min_consecutive_positive_vz=post_az_min_consecutive_positive_vz,
                max_early_az_frame=max_early_az_frame,
                max_early_hs_frame=max_early_hs_frame,
            )
            if candidates:
                to_frames.append(
                    _pick_to_az_candidate(candidates, toe_z))

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
                                     pre_descent_frames=5,
                                     min_pre_descent_negative_hits=3,
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
                neg_hits = int(np.sum(pre < 0))
                if (neg_hits < min_pre_descent_negative_hits or
                        np.mean(np.abs(pre)) < min_pre_descent_speed):
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
                            preceding_hs_window=5, heel_z=None, toe_z=None,
                            fs=100.0, hs_ground_tolerance_mm=40,
                            post_ground_hold_enabled=False,
                            post_ground_tolerance_mm=20.0,
                            post_ground_min_frames=8,
                            post_max_early_rise_mm=8.0,
                            post_early_rise_frames=8):
    """TS validation.

    A TS candidate is valid only if ALL of:
        1. A same-side HS follows within ``max_gap_frames`` frames (heel rocks
           down after the toe lands — classic toe-strike landing).
        2. No same-side HS occurred within ``preceding_hs_window`` frames
           BEFORE the TS candidate. If such a preceding HS exists, the
           candidate is the toe-down phase of that HS, not a separate TS.
           Preceding HS is checked against detected ``hs_frames`` and, when
           ``heel_z`` is supplied, any ground-level heel Vz sign-change in the
           same window (catches HS contacts missed by cycle-based heel detection).
        3. When ``post_ground_hold_enabled`` and ``toe_z`` are set, the toe
           remains near ground after the candidate without an immediate rise.

    Returns
    -------
    valid_ts : list of int
        TS frames that pass all rules.
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
        if not following:
            continue
        if (post_ground_hold_enabled and toe_z is not None and
                not _toe_holds_ground_after_ts(
                    toe_z, fs, ts,
                    ground_tolerance_mm=post_ground_tolerance_mm,
                    min_near_ground_frames=post_ground_min_frames,
                    max_early_rise_mm=post_max_early_rise_mm,
                    early_rise_frames=post_early_rise_frames)):
            continue
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

