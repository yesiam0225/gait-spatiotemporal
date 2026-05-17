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

from .cli import main
from .constants import (
    DEFAULT_PHYSIOLOGICAL_BOUNDS,
    G,
    OUTLIER_VARS,
    VARS_TO_AGGREGATE,
)
from .export import (
    aggregate_obstacle_per_subject,
    aggregate_per_subject,
    aggregate_per_trial_phase,
    obstacle_to_row,
    stride_records_to_df,
    stride_records_to_per_step_df,
)
from .gait_events import detect_gait_events_foot_based
from .markers import (
    butterworth_filter_traj,
    filter_markers,
    get_traj,
    load_marker_csv,
)
from .models import (
    Anthropometry,
    GaitEvents,
    ObstacleParameters,
    StrideRecord,
    TrialMetadata,
    WalkingSetup,
)
from .obstacle_events import detect_obstacle_events
from .obstacle_params import compute_obstacle_parameters
from .outliers import (
    apply_outlier_flags,
    detect_trial_deviation_outliers,
    is_severe_outlier,
)
from .phases import assign_phases
from .pipeline import analyze_trial, analyze_trials
from .walking_setup import determine_setup
from .strides import compute_stride_records

__all__ = [
    "G",
    "DEFAULT_PHYSIOLOGICAL_BOUNDS",
    "OUTLIER_VARS",
    "VARS_TO_AGGREGATE",
    "WalkingSetup",
    "TrialMetadata",
    "Anthropometry",
    "GaitEvents",
    "StrideRecord",
    "ObstacleParameters",
    "load_marker_csv",
    "get_traj",
    "butterworth_filter_traj",
    "filter_markers",
    "determine_setup",
    "detect_gait_events_foot_based",
    "detect_obstacle_events",
    "compute_stride_records",
    "assign_phases",
    "compute_obstacle_parameters",
    "apply_outlier_flags",
    "is_severe_outlier",
    "detect_trial_deviation_outliers",
    "analyze_trial",
    "stride_records_to_df",
    "stride_records_to_per_step_df",
    "obstacle_to_row",
    "aggregate_per_trial_phase",
    "aggregate_per_subject",
    "aggregate_obstacle_per_subject",
    "analyze_trials",
    "main",
]
