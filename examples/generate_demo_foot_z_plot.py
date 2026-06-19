#!/usr/bin/env python3
"""Generate foot-Z event detection demo PNG with IC (HS/TS) and TO markers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXAMPLES = Path(__file__).resolve().parent
REPO = EXAMPLES.parent
MARKER_LABEL_SRC = REPO.parent / "marker-label" / "src"
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(REPO / "src"))
if MARKER_LABEL_SRC.is_dir():
    sys.path.insert(0, str(MARKER_LABEL_SRC))

from demo_synthetic_foot import write_demo_foot_csv
from spatiotemporal.gait_events import detect_gait_events_foot_based, _vicon_frame_at
from spatiotemporal.markers import get_traj, load_marker_csv
from spatiotemporal.walking_setup import determine_setup

FOOT_MARKERS = ("LHEE", "LTOE", "RHEE", "RTOE")
ASSETS = REPO / "docs" / "assets"
DEMO_CSV = EXAMPLES / "demo_data" / "demo_foot_markers.csv"

EVENT_IC = "IC"
EVENT_TO = "TO"


def load_trajectory_csv(csv_path: Path) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """
    Load flat labeled CSV or C3D-style export (via marker-label ``parse_labeled_csv``).
    """
    try:
        return load_marker_csv(str(csv_path))
    except ValueError:
        pass

    from marker_label.trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(csv_path)
    labels = meta["all_stems"]
    points = np.asarray(meta["points"], dtype=np.float64)
    frames = np.asarray(meta["frames"], dtype=np.int64)
    times = np.asarray(meta["times"], dtype=np.float64)
    cols: dict[str, np.ndarray] = {"frame": frames, "time": times}
    for i, lab in enumerate(labels):
        stem = str(lab).strip()
        cols[f"{stem}_x"] = points[:, i, 0]
        cols[f"{stem}_y"] = points[:, i, 1]
        cols[f"{stem}_z"] = points[:, i, 2]
    df = pd.DataFrame(cols)
    col_map: dict[str, dict[str, str]] = {}
    for c in df.columns:
        if c.endswith(("_x", "_y", "_z")):
            name = c[:-2].strip()
            col_map.setdefault(name, {})[c[-1]] = c
    return df, col_map, frames


def _plot_events(
    frames: np.ndarray,
    z_by_marker: dict[str, np.ndarray],
    events: list[tuple[str, str, int, str]],
    out_path: Path,
    *,
    dpi: int = 150,
    title: str = "Foot marker height — event detection (demo)",
    xlim: tuple[int, int] | None = None,
) -> None:
    colors = {
        "LHEE": "#1f77b4",
        "LTOE": "#aec7e8",
        "RHEE": "#d62728",
        "RTOE": "#ff9896",
    }
    fig, ax = plt.subplots(figsize=(12, 5), layout="tight")
    for marker in FOOT_MARKERS:
        ax.plot(frames, z_by_marker[marker], label=marker, color=colors[marker], lw=1.2)

    styles = {
        ("left", EVENT_IC, "HS"): ("o", "#0b3d91", "L IC (HS)"),
        ("left", EVENT_IC, "TS"): ("s", "#0b3d91", "L IC (TS)"),
        ("right", EVENT_IC, "HS"): ("o", "#b71c1c", "R IC (HS)"),
        ("right", EVENT_IC, "TS"): ("s", "#b71c1c", "R IC (TS)"),
        ("left", EVENT_TO, ""): ("v", "#0b3d91", "L TO"),
        ("right", EVENT_TO, ""): ("v", "#b71c1c", "R TO"),
    }
    labeled: set[str] = set()
    frame_map = {int(f): i for i, f in enumerate(frames) if np.isfinite(f)}

    for side, event, vicon_frame, strike in events:
        if xlim is not None and not (xlim[0] <= int(vicon_frame) <= xlim[1]):
            continue
        idx = frame_map.get(int(vicon_frame))
        if idx is None:
            continue
        strike_key = strike if event == EVENT_IC else ""
        style = styles.get((side, event, strike_key))
        if style is None and event == EVENT_IC:
            style = styles[(side, EVENT_IC, "HS")]
        if style is None:
            continue
        marker = (
            "LTOE" if side == "left" and (event == EVENT_TO or strike == "TS")
            else "RTOE" if side == "right" and (event == EVENT_TO or strike == "TS")
            else "LHEE" if side == "left"
            else "RHEE"
        )
        z_val = z_by_marker[marker][idx]
        if not np.isfinite(z_val):
            continue
        mk, color, lab = style
        use_lab = lab if lab not in labeled else None
        if use_lab:
            labeled.add(lab)
        ax.scatter(
            [frames[idx]], [z_val],
            marker=mk, color=color, s=55, edgecolors="white", linewidths=0.6,
            zorder=5, label=use_lab,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z (mm)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def collect_events(gait_ev, frame_labels: np.ndarray) -> list[tuple[str, str, int, str]]:
    out: list[tuple[str, str, int, str]] = []
    for side, hs, strikes, tos in (
        ("left", gait_ev.left_hs, gait_ev.left_strike_types, gait_ev.left_to),
        ("right", gait_ev.right_hs, gait_ev.right_strike_types, gait_ev.right_to),
    ):
        for idx, st in zip(hs, strikes):
            out.append((side, EVENT_IC, _vicon_frame_at(int(idx), frame_labels), st or "HS"))
        for idx in tos:
            out.append((side, EVENT_TO, _vicon_frame_at(int(idx), frame_labels), ""))
    return sorted(out, key=lambda x: x[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local marker CSV (flat labeled or C3D-style export)",
    )
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use procedural demo CSV (default when --input omitted)",
    )
    parser.add_argument("--output", type=Path, default=ASSETS / "foot_z_events_demo.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--xlim",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional Vicon frame window",
    )
    args = parser.parse_args()

    if args.input is not None:
        csv_path = args.input
        if not csv_path.is_file():
            raise SystemExit(f"Input not found: {csv_path}")
        title = args.title or "Foot marker height — event detection"
    elif args.synthetic:
        write_demo_foot_csv(DEMO_CSV)
        csv_path = DEMO_CSV
        title = args.title or "Foot marker height — event detection (demo)"
    else:
        raise SystemExit("Pass --input PATH or --synthetic")

    df, col_map, frame_labels = load_trajectory_csv(csv_path)
    markers_raw = {
        n: get_traj(df, col_map, n) for n in col_map if get_traj(df, col_map, n) is not None
    }
    setup = determine_setup(markers_raw, ("OBSTACLE_L", "OBSTACLE_R"), 100.0)
    gait_ev = detect_gait_events_foot_based(
        markers_raw, setup, raw_markers=markers_raw, frame_labels=frame_labels,
    )
    z_by_marker = {m: df[col_map[m]["z"]].to_numpy() for m in FOOT_MARKERS}
    events = collect_events(gait_ev, frame_labels)
    xlim = tuple(args.xlim) if args.xlim is not None else None
    _plot_events(
        frame_labels, z_by_marker, events, args.output,
        dpi=args.dpi, title=title, xlim=xlim,
    )
    print(f"Wrote {args.output} ({len(events)} events)")


if __name__ == "__main__":
    main()
