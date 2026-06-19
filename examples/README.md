# Portfolio demo assets

```bash
pip install -e ".[demo]"
# Local trimmed CSV (do not commit source trials)
python examples/generate_demo_foot_z_plot.py --input path/to/your_trimmed.csv
# Or synthetic foot markers only:
python examples/generate_demo_foot_z_plot.py --synthetic
```

Requires `matplotlib` (`.[demo]` optional dependency).

Output: `docs/assets/foot_z_events_demo.png`

Local CSV under `examples/demo_data/` is gitignored.
