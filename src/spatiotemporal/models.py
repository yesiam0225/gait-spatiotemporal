"""Data structures for gait analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

