## LaplacianMerge Workflow — Merging Marine/Land Bouguer Anomaly Grids (Summary)

Languages: [English](./README.md) | [Chinese (simpled)](./README_CN.md)

This directory contains scripts for merging marine/land gravity grids in the Laplacian domain (scripts are numbered sequentially from 0 to 7). Technical details and theoretical foundations are provided in `Document2.md`.

Core idea: Compute the Laplacian separately on each original grid (to avoid singularities at stitch lines where artificial differences occur), merge the Laplacians using a priority rule (marine priority), fill gaps by interpolation, optionally apply a model constraint in the Laplacian/source domain, and finally invert in the frequency domain to obtain the Bouguer anomaly.

---

### Script Overview (Execution Order)

- **0_read_asc_header.py**
  - Purpose: Utility/example script to read ESRI ASCII header and print basic information (for checking input `.asc` file grid geometry and coverage).
  - Optional execution.

- **1_geometry_and_mask.py** (Step 1)
  - Purpose: Align the marine grid to the land grid (so both share the same global domain Ω), construct the priority mask m(x,y) (marine=1, land-only=0, no-data=NaN), and generate an initial composite field g0(x,y) (visual check only).
  - Outputs:
    - `PriorityMask.asc` — Mask file (used to determine the source of the Laplacian in subsequent steps).
    - `InitialComposite_g0.asc` — Initial composite field (visualization only; not used as input for subsequent computation).
    - `Sea_Aligned_To_LandGrid.asc` — Marine data aligned to the land grid coordinate system (used as input for subsequent steps).

- **2_laplacian_computation.py** (Step 2)
  - Purpose: Compute the discrete Laplacian (5-point finite difference) independently on each original grid, with NaN propagation (if any point in the stencil is NaN, the Laplacian at that point is NaN).
  - Outputs:
    - `Land_Laplacian.asc` — Laplacian computed on the land grid (using land grid header).
    - `Sea_Laplacian_Original.asc` — Original Laplacian on the marine grid (using marine header).
    - `Sea_Laplacian_Aligned.asc` — Marine Laplacian aligned to the land grid (for merging).

- **3_merge_and_interpolate_dual.py** (Steps 3 & 4)
  - Purpose: Merge the two Laplacians using the priority mask (Equation (15)), generating the merged source field L0; identify gaps (Ω_gap) caused by stencil boundaries; interpolate to fill the gaps (local or global interpolation).
  - Intermediate/Output files:
    - `L0_MergedLaplacian_BeforeInterp.asc` — Merged L0 before interpolation (saved for gap inspection).
    - `L_FilledLaplacian.asc` — Filled and interpolated unified Laplacian f(i,j), ready for frequency-domain inversion.

- **4_gap_check.py**
  - Purpose: Unified gap checking script (modes: `all`/`summary`/`quality`/`quick`). Checks mask, L_before, L_filled statistics and quality (neighborhood smoothness, etc.).
  - Usage examples:
    - `python 4_gap_check.py --mode all`
    - `python 4_gap_check.py --mode quality`

- **(4.5) 7_Laplacian_with_model_constraint.py** (Optional, recommended when a reliable large-scale model exists)
  - Purpose: Apply an external model (e.g., ICGEM BouguerModelled.asc) as a large-scale constraint to the merged/interpolated Laplacian/source grid. The model usually has lower resolution but correct large-scale trend; using it as a constraint avoids frequency-domain inversion artifacts that tend to “flatten” extrema (i.e., high values are pulled down and low values pushed up).
  - Why use it here (before inversion): The model constrains low-frequency/large-scale components in the source domain (Laplacian). After applying the constraint to the Laplacian, frequency-domain inversion (Step 5) will reconstruct a Bouguer field that preserves small-scale observational detail in the core regions while matching the model at large scales in external regions.
  - Main behavior:
    - Align merged Laplacian and optional “our” interpolated data to the model grid.
    - Identify core region (erosion of observed coverage), transition band, boundary layer (used to estimate model-vs-lap offset), and external region (model valid, observations absent).
    - Estimate offset from boundary layer (model - laplacian) and apply a smoothly varying weight (Gaussian-smoothed) to blend model and laplacian values across the transition.
  - Inputs/Outputs & Parameters (CLI supported):
    - `--laplacian` L_FilledLaplacian.asc (default) — merged & filled Laplacian input.
    - `--our` L_FilledLaplacian.asc (default) — our merged/interpolated dataset used to define observed coverage/core.
    - `--model` BouguerModelled.asc (default) — external model (ICGEM) used as a constraint.
    - `--output` BouguerLaplacianConstrained.asc (default) — constrained Laplacian written for inversion.
    - `--transition-sigma` (pixels, default 20) — Gaussian smoothing sigma for transition weights.
    - `--safety-padding` (pixels, default 10) — erosion iterations to define the core region.
    - `--boundary-width` (pixels, default 5) — width of boundary layer used to estimate offset.
  - Recommended placement: Run after `3_merge_and_interpolate_dual.py` (which produces `L_FilledLaplacian.asc`) and before `5_frequency_inversion.py`. If you run 5 before 7, you will need to re-run 5 after 7 for the model constraint to affect the final Bouguer field.
  - Example usage:
    - `python 7_Laplacian_with_model_constraint.py --laplacian L_FilledLaplacian.asc --our L_FilledLaplacian.asc --model BouguerModelled.asc --output BouguerLaplacianConstrained.asc --transition-sigma 20 --safety-padding 10 --boundary-width 5`

- **5_frequency_inversion.py** (Step 5)
  - Purpose: Invert the filled (and optionally constrained) Laplacian in the frequency domain to obtain the final Bouguer anomaly field (using the transfer function H(kx,ky) of the discrete Laplacian). Supports direct division or Tikhonov regularization.
  - Default I/O (overridable via command-line arguments):
    - Input: `L_FilledLaplacian.asc` or the constrained output from Step 4.5 (`--input`).
    - Header for dimensions/cellsize: `Land_Laplacian.asc` (`--header`).
    - Output (no regularization): `BouguerFinal.asc`.
    - Output (Tikhonov): `BouguerFinal_Tikhonov.asc`.
  - Usage examples:
    - Direct division (default):
      `python 5_frequency_inversion.py --input L_FilledLaplacian.asc --header Land_Laplacian.asc --output BouguerFinal.asc --no-tikhonov`
    - With Tikhonov:
      `python 5_frequency_inversion.py --tikhonov --alpha 1e-8 --input BouguerLaplacianConstrained.asc --header Land_Laplacian.asc --output BouguerFinal_Tikhonov.asc`

- **6_validate_bouguer_merge.py**
  - Purpose: Compare the original land grid with the merged final result over land-only regions, compute difference statistics, and optionally apply a systematic offset correction to the final grid.
  - Usage example:
    - `python 6_validate_bouguer_merge.py --land BouguerLandGrid.asc --sea BouguerSeaGrid.asc --final BouguerFinal.asc --correct --output BouguerFinal_Corrected.asc --threshold 1.0`

---

### Intermediate File List (in Order of Generation)

| Filename | Generated By | Content & Purpose |
|----------|--------------|-------------------|
| `PriorityMask.asc` | `1_geometry_and_mask.py` | Mask m(x,y): 1=marine priority, 0=land-only, NaN=no data. Used to decide Laplacian source during merging. |
| `InitialComposite_g0.asc` | `1_geometry_and_mask.py` (visualization/check) | Initial composite gravity field g0 stitched by mask (not used for inversion; only for checking coverage and discontinuities). |
| `Sea_Aligned_To_LandGrid.asc` | `1_geometry_and_mask.py` | Marine data aligned to the land grid coordinate system (used for subsequent Laplacian computation or data alignment). |
| `Land_Laplacian.asc` | `2_laplacian_computation.py` | Discrete Laplacian computed on the original land grid; NaN propagated if any stencil point is NaN. |
| `Sea_Laplacian_Original.asc` | `2_laplacian_computation.py` | Laplacian computed on the original marine grid (marine header/dimensions). |
| `Sea_Laplacian_Aligned.asc` | `2_laplacian_computation.py` | Marine Laplacian aligned to the land grid (for merging). |
| `L0_MergedLaplacian_BeforeInterp.asc` | `3_merge_and_interpolate_dual.py` (saved pre-merge state) | Merged Laplacian source L0 (NaN in gap regions). Used to inspect gap distribution and subsequent interpolation. |
| `L_FilledLaplacian.asc` | `3_merge_and_interpolate_dual.py` | Unified Laplacian f(i,j) with gaps filled via interpolation (global or local). This is the preferred input for the optional model-constraint step. |
| `BouguerLaplacianConstrained.asc` | `7_Laplacian_with_model_constraint.py` (optional) | Constrained Laplacian produced by blending the model (e.g., ICGEM) and observed Laplacian in a transition band; used as input to Step 5 for inversion. |
| `BouguerFinal.asc` / `BouguerFinal_Tikhonov.asc` | `5_frequency_inversion.py` | Final Bouguer anomaly grid obtained by frequency-domain inversion from the interpolated (and optionally constrained) Laplacian. |
| `BouguerFinal_Corrected.asc` (optional) | `6_validate_bouguer_merge.py --correct` | Corrected final grid with systematic offset added back if detected in land-only regions and correction requested. |

---

### Dependencies

- Python 3.x
- numpy
- scipy (scipy.signal.convolve2d, scipy.interpolate.griddata, scipy.ndimage, etc.)
- Note: Scripts previously used hardcoded directory paths; most scripts now accept/should accept command-line inputs. Before running, verify the paths or pass files via CLI arguments where available.

---

### Common Parameters

- `INTERP_MODE` (`3_merge_and_interpolate_dual.py`): `'local'` (recommended, faster for large grids) or `'global'` (for small grids or sparse gaps).
- `PADDING` (`3_merge_and_interpolate_dual.py`): For local mode, the expansion neighborhood size (in grid points) around gaps.
- `--tikhonov` / `--alpha` (`5_frequency_inversion.py`): Whether to use Tikhonov regularization and the alpha value (if enabled, typical alpha ranges from 1e-6 to 1e-8; the script defaults to alpha=1e-10).
- `--transition-sigma` (`7_Laplacian_with_model_constraint.py`): Gaussian sigma for transition smoothing (default 20 pixels; recommend 10–50 depending on cellsize and model resolution).
- `--safety-padding` (`7_Laplacian_with_model_constraint.py`): Core erosion in pixels (default 10; recommend 5–20).
- `--boundary-width` (`7_Laplacian_with_model_constraint.py`): Boundary layer width in pixels for offset estimation (default 5; recommend 3–10).

---

### Notes & Known Issues

- File naming and semantics: 7_Laplacian_with_model_constraint.py is an optional step intended to constrain the Laplacian/source domain using an external model (ICGEM). Run it after Step 3 and before Step 5 to affect the final inversion.
- File format consistency: Different scripts may use different nodata conventions (some use -9999, others read from the header `nodata_value`). Ensure your input `.asc` files have consistent headers and nodata settings.
- Interpolation sensitivity: `griddata(method='cubic')` may fail or run slowly on large grids or with few points. The local mode is more robust. If residual NaNs remain, the script attempts nearest-neighbor fill.
- Model caution: If the external model contains systematic bias or different reference frames, aggressive constraint may mask real local signals. Start with conservative transition widths and inspect the boundary_diff statistics.

---

### Recommended Typical Execution Flow (Minimal Manual Intervention)

1. **Preparation:** Place `BouguerLandGrid.asc` and `BouguerSeaGrid.asc` in your designated data directory. Modify paths in the scripts or pass files via CLI where supported.
2. *(Optional)* Check headers: `python 0_read_asc_header.py`.
3. Generate mask and align marine grid: `python 1_geometry_and_mask.py`.
4. Compute Laplacian on each original grid: `python 2_laplacian_computation.py`.
5. Merge Laplacians and interpolate gaps (choose local/global based on data size): modify `INTERP_MODE` and `PADDING`, then run `python 3_merge_and_interpolate_dual.py`.
6. Check interpolation quality: `python 4_gap_check.py --mode all` (review statistics and smoothness).
7. Optional (recommended when a reliable large-scale model exists): apply model constraint to the Laplacian / source field: `python 7_Laplacian_with_model_constraint.py --laplacian L_FilledLaplacian.asc --model BouguerModelled.asc --output BouguerLaplacianConstrained.asc`.
8. Frequency-domain inversion to Bouguer field (Tikhonov optional):
   - No regularization: `python 5_frequency_inversion.py --input BouguerLaplacianConstrained.asc --header Land_Laplacian.asc --output BouguerFinal.asc --no-tikhonov`.
   - With Tikhonov: `python 5_frequency_inversion.py --tikhonov --alpha 1e-8 --input BouguerLaplacianConstrained.asc --header Land_Laplacian.asc --output BouguerFinal_Tikhonov.asc`.
9. Validate and (optionally) apply offset correction based on land-only regions: `python 6_validate_bouguer_merge.py --land BouguerLandGrid.asc --sea BouguerSeaGrid.asc --final BouguerFinal.asc --correct --output BouguerFinal_Corrected.asc --threshold 1.0`.

---

## References

- See `Document2.md` in this directory for the full mathematical derivation and implementation details (Laplacian operator, interpolation, frequency-domain inversion, and Tikhonov regularization).
