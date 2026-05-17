"""Per-trial and multi-trial analysis pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .export import (
    aggregate_obstacle_per_subject,
    aggregate_per_subject,
    aggregate_per_trial_phase,
    obstacle_to_row,
    stride_records_to_df,
    stride_records_to_per_step_df,
)
from .gait_events import detect_gait_events_foot_based
from .markers import filter_markers, get_traj, load_marker_csv
from .models import Anthropometry, TrialMetadata
from .obstacle_events import detect_obstacle_events
from .obstacle_params import compute_obstacle_parameters
from .outliers import apply_outlier_flags
from .phases import assign_phases
from .walking_setup import determine_setup
from .strides import compute_stride_records

logger = logging.getLogger(__name__)

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
