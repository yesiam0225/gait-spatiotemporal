"""Outlier detection (IQR, physiological, trial-deviation)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import DEFAULT_PHYSIOLOGICAL_BOUNDS, OUTLIER_VARS

logger = logging.getLogger(__name__)

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

