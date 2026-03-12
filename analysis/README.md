# Refactored analysis directory

This refactor splits the code into three layers:

- `tools/` — data loading, signal processing, and analysis
- `plotting/` — reusable plotting helpers
- `viz/` — thin visualization scripts

The analysis layer now treats `track2_permanence.msgpack` as the canonical
upstream input and derives spacing / velocity in memory instead of relying on a
persisted Track3 file.

## Layout

```text
analysis/
├── configs/
│   ├── datasets.json
│   └── trivialdata.json
├── plotting/
│   ├── common.py
│   ├── frequency.py
│   ├── indexed.py
│   └── trajectory.py
├── tools/
│   ├── cli.py
│   ├── derived.py
│   ├── io.py
│   ├── localization.py
│   ├── models.py
│   ├── peaks.py
│   ├── selection.py
│   ├── signal.py
│   └── spectral.py
└── viz/
    ├── avg_fft.py
    ├── avg_fft_sites.py
    ├── localize_peaks.py
    ├── localize_sitepeaks.py
    ├── see_fft.py
    ├── see_positions.py
    └── spacing_timeseries.py
```

## Running scripts

Run them from the `analysis/` directory, for example:

```bash
python3 viz/see_fft.py IMG_0584
python3 viz/avg_fft.py configs/datasets.json --normalize relative
python3 viz/avg_fft_sites.py configs/datasets.json peaks.csv --normalize relative
python3 viz/localize_peaks.py configs/datasets.json peaks.csv --normalize relative
python3 viz/localize_sitepeaks.py configs/datasets.json peaks.csv --normalize relative
python3 viz/see_positions.py IMG_0584 --framestrip
python3 viz/spacing_timeseries.py IMG_0584
```

If the sibling `../track/data/` location is not correct, pass:

```bash
--track-data-root /path/to/track/data
```

## Notes

- The code prioritizes clarity over backward-compatibility.
- The old Track3 analysis file is no longer required for FFT / localization.
- `avg_fft_sites.py` is the dedicated replacement for the old overlay behavior
  that lived in `avg_fft2.py`.
