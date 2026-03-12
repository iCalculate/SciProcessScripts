# FET Transfer Curve Analysis Scripts

This directory contains MATLAB scripts that automatically extract FET device parameters from **Id–Vg (transfer) curves**. This document describes each script’s role, the expected data format, **how each parameter is extracted**, and the **mapping between variable names and their physical/mathematical meaning**.

---

## 1. Script Roles

| Script | Role |
|--------|------|
| **`import_dataMatrix_and_analyze.m`** | Reads Vg and multiple Id columns from an Excel file (e.g. `dataMatrix.xlsx`), builds the internal data structure, calls `analyze_transfer_curves`, and returns a parameter summary for each device (column). |
| **`analyze_transfer_curves.m`** | Core analysis: preprocesses each transfer curve, selects the subthreshold window, performs linear fits and extrapolations, and computes SS, Vth, Von, gm, etc. Optional plotting. |

**Data flow:** Excel → `import_dataMatrix_and_analyze` → `data.Data.Matrix.{Vg, Id, Vd}` → `analyze_transfer_curves` → `res.PerCurve` / `res.Summary`.

---

## 2. Input Data Format

### 2.1 Excel (for `import_dataMatrix_and_analyze`)

- **Column 1:** Gate voltage **Vg** (V), shared by all devices.
- **Columns 2 onward:** One column per device **Id** (A), row-aligned with column 1 (Vg).
- Rows with non-numeric Vg are dropped. Vg row count must match each Id column’s row count.

### 2.2 Internal struct (for `analyze_transfer_curves`)

- `data.Data.Matrix.Vg`: nPts×1 or nPts×nDevices, gate voltage.
- `data.Data.Matrix.Id`: nPts×nDevices, drain current; each column is one curve.
- `data.Data.Matrix.Vd`: optional, nPts×nDevices; may be NaN when not measured. Per-curve scalar Vd is the column mean.

---

## 3. Parameter Extraction Flow (Overview)

For **each curve** (each Id column), the following steps are applied independently (all logic is in `analyze_transfer_curves.m`):

1. **Preprocessing:** Take `abs(Id)`, apply a compliance mask if available, and replace invalid/zero current with the lower Y-limit for log scaling.
2. **Ioff estimate:** Use the median of the lowest ~10% of currents as the off-state current estimate.
3. **Subthreshold window:** From d(log10|Id|)/dVg, find the first slope peak after “leaving Ioff”, then take the contiguous region where slope ≥ a fraction (PeakFrac) of that peak, enforcing a minimum number of points.
4. **Subthreshold linear fit:** In that window fit log10(|Id|) = a·Vg + b; obtain slope a, intercept b, SS, and R².
5. **Ion/Ioff:** Max/min over all valid currents.
6. **Transconductance gm:** Numerical derivative dId/dVg; mask compliance and neighbors, then take max(gm) and the corresponding Vg (Vth at gm-max).
7. **Vth extrapolation:** Use the subthreshold fit to extrapolate in log space to the Vg at Ioff×10 and at a fixed Iref; optionally, intersection with the above-threshold linear fit.
8. **Von:** Linear fit Id = a_lin·Vg + b_lin in the “high-current” band (e.g. |Id| ≥ 80th percentile). If point count, ΔVg, slope, and R² conditions are met, Von = −b_lin/a_lin (extrapolation to Id=0); the linear fit R² is stored as Von_fit_R2.
9. **Vth(extrap-cross):** Vg where the subthreshold exponential fit and the Von linear fit intersect (same current).

The following section lists each **variable name** with its meaning and how it is computed.

---

## 4. Variable Names, Meanings, and Extraction Methods

The names below correspond to **`res.PerCurve(k)`** fields and **`res.Summary`** column names (column names may be slightly abbreviated; see §5).

| Variable | Meaning | How it is obtained |
|----------|---------|---------------------|
| **Vd** | Drain–source voltage (V) | From input: mean of `data.Data.Matrix.Vd` for that column; NaN when using the Excel path without Vd. |
| **Ioff_est** | Off-state current estimate (A) | **Median** of the **lowest ~10%** of |Id| values (after sorting). Used for “leaving Ioff” and Vth reference. |
| **Vg_win** | Vg range [left, right] of subthreshold fit window (V) | From d(log10|Id|)/dVg, find **first slope peak after leaving Ioff** (leave_idx: |Id| ≥ Ioff_est×IoffFrac). Window = contiguous indices where slope ≥ PeakFrac×peak slope; padded to at least MinWinPts. Vg_win = [Vg(L), Vg(R)]. |
| **idx_win** | Data indices of the subthreshold window | Indices L:R; used for plotting the fit range and computing R². |
| **slope_dec_per_V** | Slope of log10(|Id|) vs Vg in subthreshold (dec/V) | Linear fit **log10(|Id|) = a·Vg + b** in the window; **a** is slope_dec_per_V (robustfit if available, else backslash). |
| **intercept_dec** | Intercept of subthreshold fit (log10(A)) | The **b** in the same fit. |
| **SS_mV_dec** | Subthreshold swing (mV/dec) | **SS = (1/a)×1000** with a = slope_dec_per_V; gate voltage (mV) needed for one decade of current change. |
| **SS_fit_R2** | R² of the subthreshold linear fit | For log10(|Id|) in the window: R² = 1 − SS_res/SS_tot. |
| **Ion_Ioff_ratio** | Ion/Ioff ratio | **Max/min** over all **valid** |Id| (finite and > 0). |
| **gm_max** | Maximum transconductance (A/V) | **gm = dId/dVg**; Id is compliance-masked and lightly smoothed, then differentiated; compliance and adjacent points set to NaN; **max(gm)** is taken. |
| **Vth** | Threshold voltage (V), default definition | Same as **Vth_extrap_Ioff**: Vg where subthreshold fit gives log10(Ioff_est×10), i.e. Vth = (log10(Ioff_est×10) − b)/a. |
| **Vth_extrap_Ioff** | Vth from extrapolation at Ioff×10 (V) | Solve **log10(Ioff_est×10) = a·Vg + b** using the subthreshold fit. |
| **Vth_extrap_Iref** | Vth from extrapolation at fixed Iref (V) | Solve **log10(IrefFixed) = a·Vg + b**; IrefFixed is an option (default 1e-7 A). |
| **Vth_gmmax** | Vg at maximum gm (V) | **Vg** at **argmax(gm)**. |
| **Von** | Turn-on voltage (V) | In the **high-current band** (|Id| ≥ 80th percentile), fit **Id = a_lin·Vg + b_lin**. If point count ≥ MinVonPoints, Vg span ≥ MinVonDeltaVg, |a_lin| ≥ MinVonSlope, and R² ≥ MinVonR2, then **Von = −b_lin/a_lin** (extrapolation to Id=0). Otherwise NaN. |
| **Von_fit_R2** | R² of the Von linear fit | R² of the Id–Vg linear fit in the high-current band; stored whenever that fit is performed, regardless of whether Von is accepted. |
| **Vth_extrap_cross** | Vg at intersection of subthreshold and Von linear fit (V) | Solve **10^(a·V+b) = a_lin·V + b_lin** for V (sign change or min |f(V)|, then fzero or grid). NaN if no valid Von fit. |

---

## 5. Output Structure: PerCurve vs Summary Column Names

- **`res.PerCurve(k)`**: All fields for the k-th curve; names match the variable names above (e.g. `Vd`, `Ioff_est`, `SS_mV_dec`, `Von_fit_R2`).
- **`res.Summary`**: Table with one row per curve. Column names and meaning:

| Summary column | Variable | Meaning |
|----------------|----------|---------|
| Vd | Vd | Drain–source voltage (V) |
| SS_mV_per_dec | SS_mV_dec | Subthreshold swing (mV/dec) |
| SS_fit_R2 | SS_fit_R2 | R² of subthreshold fit |
| Vg_left, Vg_right | Vg_win(1), Vg_win(2) | Left/right Vg of subthreshold window (V) |
| Ioff_est | Ioff_est | Off-state current estimate (A) |
| slope_dec_per_V | slope_dec_per_V | Subthreshold slope (dec/V) |
| Ion_Ioff_ratio | Ion_Ioff_ratio | Ion/Ioff ratio |
| gm_max | gm_max | Maximum transconductance (A/V) |
| Vth | Vth | Threshold voltage (V), default Ioff×10 extrapolation |
| Vth_extrap_Ioff | Vth_extrap_Ioff | Vth at Ioff×10 (V) |
| Vth_extrap_Iref | Vth_extrap_Iref | Vth at fixed Iref (V) |
| Vth_extrap_cross | Vth_extrap_cross | Vth at subthreshold–linear intersection (V) |
| Vth_gmmax | Vth_gmmax | Vg at maximum gm (V) |
| Von | Von | Turn-on voltage (V) |
| Von_fit_R2 | Von_fit_R2 | R² of Von linear fit |

---

## 6. Main Name–Value Options

These options can be passed to `analyze_transfer_curves(..., 'Name', value)` or `import_dataMatrix_and_analyze(..., 'Name', value)` and affect window selection and acceptance criteria.

| Option | Default | Meaning |
|--------|--------|---------|
| YLim | [1e-13, 1e-3] | Log y-axis display range (A). |
| DoPlot | true | Whether to plot overview, per-curve detail, and trend figures. |
| SmoothingPts | 5 | Moving-average length for d(log10|Id|)/dVg. |
| PeakFrac | 0.8 | Subthreshold window: contiguous region where slope ≥ PeakFrac×peak slope. |
| IoffFrac | 3 | “Leaving Ioff” defined as |Id| ≥ Ioff_est×IoffFrac. |
| PadVg | 0.1 | Vg span to extend the fit line on both sides (V). |
| MinWinPts | 6 | Minimum number of points in the subthreshold window. |
| ComplianceFrac | 0.98 | Points with |Id| ≥ this fraction of compliance are masked from fits and gm. |
| DenoiseWindow | 5 | Window size for log-current smoothing and slope detection. |
| MinVonPoints | 5 | Minimum number of points for the Von linear fit. |
| MinVonR2 | 0.95 | Von accepted only if linear fit R² ≥ this value. |
| MinVonSlope | 1e-12 | Von accepted only if |dId/dVg| ≥ this (A/V). |
| MinVonDeltaVg | 0.02 | Minimum Vg span of the Von fit region (V). |
| IrefFixed | 1e-7 | Fixed current (A) used for Vth_extrap_Iref. |

---

## 7. Usage Examples

```matlab
% Default Excel file, with plots
res = import_dataMatrix_and_analyze();

% Specify file, no plots
res = import_dataMatrix_and_analyze('dataMatrix.xlsx', 'DoPlot', false);

% Analyze an existing struct (e.g. from another importer)
res = analyze_transfer_curves(data, 'MinWinPts', 8, 'MinVonR2', 0.98);

% Inspect Von and fit quality for one curve
k = 1;
disp(res.PerCurve(k).Von);
disp(res.PerCurve(k).Von_fit_R2);

% Summary table
disp(res.Summary);
```

---

## 8. Summary

- **Variable names and meanings:** The tables above give the physical or statistical meaning of each output and how it is computed from Vg/Id and the options.
- **Subthreshold chain:** Ioff_est → leave Ioff → slope peak → window → log10(|Id|)=a·Vg+b → SS, SS_fit_R2, Vth extrapolations.
- **Above-threshold:** High-current Id–Vg linear fit → Von (Id=0 extrapolation), Von_fit_R2, and intersection with subthreshold fit → Vth_extrap_cross.
- **Transconductance:** Numerical dId/dVg → gm_max, Vth_gmmax.

If a curve does not meet the relevant conditions (e.g. too few points or low R²), the corresponding output is **NaN** and appears as NaN in Summary.
