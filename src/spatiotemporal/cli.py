"""Command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import Anthropometry, TrialMetadata
from .pipeline import analyze_trial, analyze_trials
from .export import obstacle_to_row, stride_records_to_df, stride_records_to_per_step_df

logger = logging.getLogger(__name__)

def _parse_manifest_trial_cell(value: Any) -> int:
    """Accept int/float/str with spaces; e.g. 5, 05, ' 06 '."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not np.isnan(value):
        return int(value)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        raise ValueError("empty or invalid trial cell")
    return int(float(s))


def _manifest_first_nonempty(row: "pd.Series", keys: tuple[str, ...], default: str) -> str:
    """First non-empty cell among optional column names (headers already lowercased)."""
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return str(default)


def _manifest_optional_float(row: "pd.Series", keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        try:
            return float(s)
        except ValueError:
            continue
    return None


def _manifest_normalize_group(s: str) -> str:
    low = str(s).strip().lower()
    if low in ("adults", "adult"):
        return "adult"
    if low in ("children", "child"):
        return "child"
    return str(s).strip()


def _run_manifest_batch_cli(args: "argparse.Namespace") -> None:
    """
    CSV manifest: one row per trial. Required columns: ``csv_path`` (or path /
    input / file) and ``trial`` (or trial_number).

    Optional per-row columns (override CLI defaults when present and non-empty):
    ``subject_id`` (or ``subject``), ``group``, ``board`` (or ``board_type``),
    ``time``, ``leg_length_mm`` (or ``leg_length``), ``mass_kg`` (or ``mass``),
    ``age_years`` (or ``age``), ``height_mm`` (or ``height``).

    If every row supplies ``leg_length_mm`` / ``leg_length``, ``--leg-length-mm``
    may be omitted. Otherwise ``--leg-length-mm`` is required as the fallback.

    The manifest must be a **file path** passed to ``--trial-manifest``; do not paste
    CSV lines into the shell (the shell will try to run them as commands).
    """
    man = pd.read_csv(args.trial_manifest)
    man.columns = [str(c).strip().lower() for c in man.columns]
    if "csv_path" not in man.columns and "path" in man.columns:
        man = man.rename(columns={"path": "csv_path"})
    if "input" in man.columns and "csv_path" not in man.columns:
        man = man.rename(columns={"input": "csv_path"})
    if "file" in man.columns and "csv_path" not in man.columns:
        man = man.rename(columns={"file": "csv_path"})
    if "trial" not in man.columns and "trial_number" in man.columns:
        man = man.rename(columns={"trial_number": "trial"})
    if "csv_path" not in man.columns or "trial" not in man.columns:
        raise SystemExit(
            "Trial manifest CSV must include columns 'csv_path' and 'trial' "
            "(aliases: path, input, file for the trajectory path; trial_number for trial)."
        )

    trial_specs: list[dict[str, Any]] = []
    for _, row in man.iterrows():
        p = row["csv_path"]
        if pd.isna(p) or str(p).strip() == "":
            continue
        tr = row["trial"]
        if pd.isna(tr):
            continue
        try:
            trial_num = _parse_manifest_trial_cell(tr)
        except (ValueError, TypeError) as e:
            raise SystemExit(f"Invalid trial value in manifest: {tr!r} ({e})") from e

        leg_mm = _manifest_optional_float(row, ("leg_length_mm", "leg_length"))
        if leg_mm is None:
            if args.leg_length_mm is None:
                raise SystemExit(
                    "Manifest row missing leg_length_mm (or leg_length); "
                    "add a column or pass --leg-length-mm as default."
                )
            leg_mm = float(args.leg_length_mm)

        sub = _manifest_first_nonempty(row, ("subject_id", "subject"), args.subject_id)
        grp_raw = _manifest_first_nonempty(row, ("group",), args.group)
        grp = _manifest_normalize_group(grp_raw)
        if grp not in ("adult", "child"):
            raise SystemExit(
                f"Invalid group {grp_raw!r} (use adult/child or adults/children). "
                f"Row csv_path={str(p).strip()!r}"
            )
        brd = _manifest_first_nonempty(row, ("board", "board_type"), args.board)
        tm = _manifest_first_nonempty(row, ("time",), args.time)

        mass = _manifest_optional_float(row, ("mass_kg", "mass"))
        if mass is None and args.mass_kg is not None:
            mass = float(args.mass_kg)
        age = _manifest_optional_float(row, ("age_years", "age"))
        if age is None and args.age_years is not None:
            age = float(args.age_years)
        height = _manifest_optional_float(row, ("height_mm", "height"))
        if height is None and args.height_mm is not None:
            height = float(args.height_mm)

        trial_specs.append(
            {
                "csv_path": str(p).strip(),
                "metadata": TrialMetadata(
                    subject_id=sub,
                    group=grp,
                    board=brd,
                    time=tm,
                    trial=trial_num,
                ),
                "anthropometry": Anthropometry(
                    leg_length_mm=float(leg_mm),
                    mass_kg=mass,
                    age_years=age,
                    height_mm=height,
                ),
            }
        )
    if not trial_specs:
        raise SystemExit("No valid rows in trial manifest.")

    paths = analyze_trials(
        trial_specs=trial_specs,
        output_dir=str(args.output_dir),
        obstacle_marker_pair=tuple(args.obstacle_pair),
        sampling_rate=float(args.sampling_rate),
        filter_cutoff=float(args.filter_cutoff),
        apply_filter=not bool(args.no_filter),
        exclusion_policy=str(args.exclusion_policy),
        trial_deviation_sd_threshold=float(args.trial_deviation_sd_threshold),
        outlier_iqr_threshold=float(args.outlier_iqr_threshold),
        outlier_min_strides_for_iqr=int(args.outlier_min_strides),
        physiological_bounds=None,
    )
    print("Analysis complete. Files written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def _run_single_trial_csv_cli(args: "argparse.Namespace") -> None:
    """Marker trajectory CSV in → per-stride table CSV out (one trial)."""
    metadata = TrialMetadata(
        subject_id=args.subject_id,
        group=args.group,
        board=args.board,
        time=args.time,
        trial=int(args.trial),
    )
    anthro = Anthropometry(
        leg_length_mm=float(args.leg_length_mm),
        mass_kg=args.mass_kg,
        age_years=args.age_years,
        height_mm=args.height_mm,
    )
    records, obstacle, _events, _setup = analyze_trial(
        csv_path=args.input,
        metadata=metadata,
        anthro=anthro,
        obstacle_marker_pair=tuple(args.obstacle_pair),
        sampling_rate=float(args.sampling_rate),
        filter_cutoff=float(args.filter_cutoff),
        apply_filter=not bool(args.no_filter),
        outlier_iqr_threshold=float(args.outlier_iqr_threshold),
        outlier_min_strides=int(args.outlier_min_strides),
    )
    stride_df = stride_records_to_df(records, metadata)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stride_df.to_csv(out_path, index=False)
    print(f"Wrote per-stride table: {out_path.resolve()}")

    step_path = (
        Path(args.output_step)
        if args.output_step is not None
        else out_path.with_name(f"{out_path.stem}_per_step{out_path.suffix}")
    )
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_df = stride_records_to_per_step_df(records, metadata)
    step_df.to_csv(step_path, index=False)
    print(f"Wrote per-step table: {step_path.resolve()}")
    if args.output_obstacle:
        obs_path = Path(args.output_obstacle)
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        row = obstacle_to_row(obstacle, metadata, anthro.leg_length_mm)
        pd.DataFrame([row]).to_csv(obs_path, index=False)
        print(f"Wrote per-trial obstacle row: {obs_path.resolve()}")


def _main():
    """
    CLI modes:

    1) Batch (JSON spec) — same as before::

        python spatiotemporal.py analysis_spec.json

    2) Single trial (CSV in, CSV out)::

        python spatiotemporal.py --input trial.csv --output strides.csv \\
            --leg-length-mm 850 --subject-id S01 --group adult --board RB \\
            --time pre --trial 1

    3) Many trials (manifest CSV + output dir)::

        python spatiotemporal.py --trial-manifest trials.csv --output-dir ./out \\
            --leg-length-mm 960 --subject-id SUBJ01 --group adult --board RB --time pre

        trials.csv: required ``csv_path`` + ``trial`` (aliases: path, input, file;
        ``trial_number`` for trial). Optional per-row columns override CLI defaults:
        ``subject_id``, ``group``, ``board``, ``time``, ``leg_length_mm``,
        ``mass_kg``, ``age_years``, ``height_mm``. If every row has ``leg_length_mm``,
        ``--leg-length-mm`` may be omitted.

    JSON spec shape:
    {
        "output_dir": "./output",
        "sampling_rate": 100.0,
        "filter_cutoff": 6.0,
        "apply_filter": true,
        "exclusion_policy": "exclude_severe",
        "trials": [
            {
                "csv_path": "data/subject01/RB_pre/trial_1.csv",
                "subject_id": "subject01",
                "group": "adult",
                "board": "RB",
                "time": "pre",
                "trial": 1,
                "leg_length_mm": 850
            }, ...
        ]
    }
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Spatiotemporal gait parameters: JSON batch, CSV manifest batch, "
            "or single-trial CSV."
        ),
    )
    parser.add_argument(
        "spec_json",
        nargs="?",
        default=None,
        help="Path to batch spec JSON (omit when using --input/--output).",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="CSV",
        help="Labeled marker trajectory CSV (must include frame + *_x/y/z columns).",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="CSV",
        help="Write per-stride results to this CSV path.",
    )
    parser.add_argument(
        "--trial-manifest",
        metavar="CSV",
        default=None,
        help=(
            "Batch: path to a CSV with csv_path and trial per row (aliases: path, input, file; "
            "trial_number for trial). Optional columns subject_id, group, board, time, "
            "leg_length_mm, mass_kg, age_years, height_mm override CLI defaults when set. "
            "Omit --leg-length-mm if every row supplies leg_length_mm. "
            "Use with --output-dir; do not paste CSV into the shell."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Batch (manifest or JSON): directory for aggregate output CSVs.",
    )
    parser.add_argument(
        "--output-step",
        metavar="CSV",
        default=None,
        help=(
            "Single-trial: per-step CSV path (default: next to --output as "
            "<output_stem>_per_step.csv)."
        ),
    )
    parser.add_argument(
        "--leg-length-mm",
        type=float,
        default=None,
        help=(
            "Leg length (mm) for normalization. Required with --input/--output. "
            "With --trial-manifest: required unless every row has leg_length_mm (or leg_length); "
            "otherwise used as fallback for rows missing that column."
        ),
    )
    parser.add_argument("--subject-id", default="unknown")
    parser.add_argument("--group", default="adult", choices=("adult", "child"))
    parser.add_argument("--board", default="RB")
    parser.add_argument("--time", default="pre")
    parser.add_argument(
        "--trial",
        "--Trial",
        type=int,
        default=1,
        dest="trial",
        help="Trial index / number (same as --Trial).",
    )
    parser.add_argument("--mass-kg", type=float, default=None, dest="mass_kg")
    parser.add_argument("--age-years", type=float, default=None, dest="age_years")
    parser.add_argument("--height-mm", type=float, default=None, dest="height_mm")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--filter-cutoff", type=float, default=6.0)
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable Butterworth filtering for spatial trajectories.",
    )
    parser.add_argument(
        "--obstacle-pair",
        nargs=2,
        metavar=("L", "R"),
        default=("OBSTACLE_L", "OBSTACLE_R"),
        dest="obstacle_pair",
        help="Obstacle marker names for setup (default OBSTACLE_L OBSTACLE_R).",
    )
    parser.add_argument("--outlier-iqr-threshold", type=float, default=1.5)
    parser.add_argument("--outlier-min-strides", type=int, default=5)
    parser.add_argument(
        "--exclusion-policy",
        default="exclude_severe",
        choices=("exclude_severe", "exclude_any", "none"),
        help="Stride exclusion when aggregating phases (manifest / JSON batch).",
    )
    parser.add_argument(
        "--trial-deviation-sd-threshold",
        type=float,
        default=2.0,
        help="Across-trial mean deviation threshold for subject-level aggregation.",
    )

    args = parser.parse_args()

    if args.trial_manifest is not None:
        if args.spec_json is not None:
            parser.error("Do not pass a spec JSON path together with --trial-manifest.")
        if args.input is not None or args.output is not None:
            parser.error("Do not combine --trial-manifest with --input/--output.")
        if args.output_dir is None:
            parser.error("--output-dir is required with --trial-manifest.")
        _run_manifest_batch_cli(args)
        return

    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            parser.error("--input and --output must be used together.")
        if args.spec_json is not None:
            parser.error("Do not pass a spec JSON path together with --input/--output.")
        if args.leg_length_mm is None:
            parser.error("--leg-length-mm is required when using --input/--output.")
        _run_single_trial_csv_cli(args)
        return

    if args.spec_json is None:
        parser.error(
            "Provide a batch spec JSON path, --trial-manifest + --output-dir, "
            "or --input + --output for single-trial CSV."
        )

    with open(args.spec_json) as f:
        spec = json.load(f)

    trial_specs = []
    for t in spec["trials"]:
        trial_specs.append(
            {
                "csv_path": t["csv_path"],
                "metadata": TrialMetadata(
                    subject_id=t["subject_id"],
                    group=t["group"],
                    board=t["board"],
                    time=t["time"],
                    trial=t["trial"],
                ),
                "anthropometry": Anthropometry(
                    leg_length_mm=t["leg_length_mm"],
                    mass_kg=t.get("mass_kg"),
                    age_years=t.get("age_years"),
                    height_mm=t.get("height_mm"),
                ),
            }
        )

    paths = analyze_trials(
        trial_specs=trial_specs,
        output_dir=spec.get("output_dir", "."),
        sampling_rate=spec.get("sampling_rate", 100.0),
        filter_cutoff=spec.get("filter_cutoff", 6.0),
        apply_filter=spec.get("apply_filter", True),
        exclusion_policy=spec.get("exclusion_policy", "exclude_severe"),
        trial_deviation_sd_threshold=spec.get("trial_deviation_sd_threshold", 2.0),
        outlier_iqr_threshold=spec.get("outlier_iqr_threshold", 1.5),
        outlier_min_strides_for_iqr=spec.get("outlier_min_strides_for_iqr", 5),
    )
    print("Analysis complete. Files written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def main() -> None:
    """CLI entrypoint for ``python -m spatiotemporal`` and the ``spatiotemporal-gait`` script."""
    _main()
