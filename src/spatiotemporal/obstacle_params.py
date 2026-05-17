"""Obstacle-crossing parameter computation."""

from __future__ import annotations

import numpy as np

from .models import GaitEvents, ObstacleParameters, WalkingSetup

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

