"""DataFrame builders and trial/subject aggregation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .constants import OUTLIER_VARS, VARS_TO_AGGREGATE
from .models import ObstacleParameters, StrideRecord, TrialMetadata
from .outliers import detect_trial_deviation_outliers, is_severe_outlier

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
