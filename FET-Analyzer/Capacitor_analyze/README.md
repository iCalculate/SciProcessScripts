# Capacitor Analyze

Entry point:

- `run_capacitor_analysis.m`: asks for a working mode, then runs the corresponding capacitor workflow.

Mode scripts:

- `capacitor_mode_single_file.m`: single-file C-V import, frequency selection, and plotting.
- `capacitor_mode_multi_file.m`: multi-file C-V / I-V preview plotting.
- `capacitor_mode_fitting.m`: area-capacitance fitting workflow and per-frequency Cox extraction.
- `capacitor_mode_breakdown.m`: breakdown I-V grouping, analysis, and plotting.

Shared UI helpers:

- `capacitor_select_single_file.m`: single-file chooser with CLI messages.
- `capacitor_select_multiple_files.m`: multi-file chooser with CLI messages.

Modes:

- `Single-file mode`: select one C-V CSV file and plot one figure with left/right y-axes for `Cp` and `G`. Different frequencies are different curves.
- `Multi-file mode`: select multiple CSV files and generate grouped preview plots.
- `Fitting mode`: select multiple C-V CSV files, then select a device-area `.csv` file and input oxide thickness in nm. The script uses the maximum capacitance at the lowest frequency for each device and fits `C` versus area.
- `Breakdown mode`: select one or more breakdown I-V CSV files. Files are grouped by device index. The first cycle is highlighted and later cycles are faded.

Import function:

- `import_capacitor_measurements.m`: dedicated importer for capacitor test CSV files. It currently supports:
  - `CpG-V Sweep` / `C-V Sweep`
  - `Breakdown I-V`

Area import function:

- `import_capacitor_area_csv.m`: loads device areas from a `.csv` file.
  Expected format:
  - column 1: identifier label
  - column 2: area in `um^2`
  Matching rule:
  - identifiers are kept for display only
  - actual fitting matches CSV rows to devices by order, after the selected C-V files are sorted by device index from the file name

Imported data layout:

- `datasets(k).meta`: file identity and inferred test type.
- `datasets(k).config`: sweep range, frequency list, AC amplitude, impedance model, and compliance.
- `datasets(k).raw`: x-axis, signal matrix, and the raw MATLAB table.

Breakdown-mode outputs:

- Leakage current estimate on the first cycle
- Rising-region semilog slope (`dec/V`)
- Breakdown voltage
- `Vth_0` from the intersection of the low-leakage fit and the rising-region fit

Recommended usage:

```matlab
datasets = run_capacitor_analysis;
```
