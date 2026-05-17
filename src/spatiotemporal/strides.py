"""Per-stride spatiotemporal parameter computation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import G
from .models import Anthropometry, GaitEvents, StrideRecord, WalkingSetup

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

