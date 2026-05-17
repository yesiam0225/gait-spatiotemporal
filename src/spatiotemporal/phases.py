"""Per-leg phase classification for obstacle crossing."""

from __future__ import annotations

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

