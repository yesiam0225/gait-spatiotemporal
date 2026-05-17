"""Obstacle crossing event detection."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

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

