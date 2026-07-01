# Parker-Oldenburg-Moho-Inversion

Gravity-constrained Moho depth inversion using the Parker-Oldenburg iterative algorithm with integrated seismic Receiver Function (RF) constraints.

## Overview

This repository implements a complete workflow for inverting the 3D topography of the Moho discontinuity from gravity anomaly observations, enhanced with seismic constraints from receiver function analysis. The method is particularly useful for studying lithospheric structure in complex geodynamic settings.

**Key Innovation:** Integration of sparse seismic receiver function observations as spatial constraints within a frequency-domain gravity inversion framework, enabling joint gravity-seismic interpretation.

---

## Features

✅ **Non-linear Parker-Oldenburg forward modeling** - Accurate computation of gravity anomalies from arbitrary Moho topography

✅ **Iterative depth inversion** - Efficient frequency-domain inversion with convergence control

✅ **Receiver Function constraints** - Incorporate seismic measurements to guide the gravity inversion

✅ **Advanced regularization** - Tikhonov filtering + low-pass frequency filtering for stable inversions

✅ **Flexural isostasy** - Optional lithospheric flexure corrections (customizable elastic thickness)

✅ **Multi-scale processing** - Handles large grids (>3000×3000) using FFT and adaptive padding

✅ **Detailed logging** - Complete iteration history and convergence diagnostics

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Dai411/Parker-Oldenburg-Moho-Inversion.git
cd Parker-Oldenburg-Moho-Inversion

# Create environment
conda create -n moho-inversion python=3.10 numpy scipy matplotlib -y
conda activate moho-inversion

# Install coordinate transformation library (recommended)
conda install -c conda-forge pyproj -y
```

### Basic Workflow (3 Steps)

```bash
cd MohoInversion/

# Step 1: Create uniform initial model
python create_uniform_moho.py

# Step 2: Setup receiver function constraints (automatic)
# → Loaded during Step 3

# Step 3: Run inversion with RF constraints
python p-o_inversion_v4.py
```

For detailed workflow documentation, see **[MohoInversion/README.md](./MohoInversion/README.md)**.

---

## Directory Structure

```
Parker-Oldenburg-Moho-Inversion/
├── README.md                              # This file (project overview)
├── StitchGrids/                           # Multi-source gravity data merging
│   ├── README.md
│   └── *.py (data preparation scripts)
│
├── MohoInversion/                         # Main inversion workflow
│   ├── README.md                          # Detailed workflow documentation
│   ├── create_uniform_moho.py             # Step 1: Initial model creation
│   ├── rf_constraint_mercator.py          # Step 2: RF constraint setup
│   ├── p-o_inversion_v4.py                # Step 3: Main iterative inversion
│   └── p-o_forward_test.py                # Validation/testing script
│
└── Input/
    ├── BouguerFinalWithModel.asc          # Observed gravity grid (input)
    └── MohoFromRF_40N.mrc                 # Receiver function observations (input)
```

---

## Workflow Summary

### Step 1: Create Uniform Initial Model
Generates a uniform Moho depth grid as the starting point for inversion. This helps diagnose whether results are biased by the initial model choice.

**Script:** `MohoInversion/create_uniform_moho.py`

```python
UNIFORM_DEPTH = 20  # km
```

**Output:** `InitialMoho_Uniform.asc`

---

### Step 2: Setup Receiver Function Constraints
Loads sparse seismic receiver function observations, converts coordinates (lat/lon → Mercator projection), and prepares spatial constraint fields for the inversion.

**Script:** `MohoInversion/rf_constraint_mercator.py` (called automatically by Step 3)

**Key Parameters:**
- `RF_SIGMA` = 30 km (Gaussian diffusion radius)
- `RF_LAMBDA_C` = 300 km (frequency filter cutoff)
- `RF_LAMBDA` = 2.0 (relative weight in joint inversion)

---

### Step 3: Iterative Inversion
Performs non-linear gravity inversion with integrated RF constraints. Includes convergence monitoring, early stopping, and detailed logging.

**Script:** `MohoInversion/p-o_inversion_v4.py`

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRHO` | 0.60 g/cm³ | Crustal-mantle density contrast |
| `Z0` | 20.0 km | Average Moho depth |
| `MAX_ITER` | 30 | Maximum iterations |
| `LEARNING_RATE` | 0.5 | Under-relaxation factor |
| `LOW_PASS_WL` | 150 km | Long-wavelength preservation |
| `USE_RF_CONSTRAINT` | True | Enable RF constraints |

**Output:** 
- `FinalMoho_RF_NoTe.asc` (inverted Moho depth in km)
- `inversion_log_v4.txt` (convergence history)

---

## Mathematical Foundation

### Parker-Oldenburg Forward Modeling

The gravity anomaly from a depth interface at $z = Z_0 + h(x,y)$ is:

$$g(x,y) = -2\pi G \Delta\rho \sum_{n=1}^{\infty} \mathcal{F}^{-1}\left\{ e^{-kZ_0} \frac{k^{n-1}}{n!} \mathcal{F}[h^n] \right\}$$

where:
- $G$ = gravitational constant
- $\Delta\rho$ = density contrast (crustal-mantle)
- $k$ = wavenumber
- $h$ = Moho topography (relative to mean depth)

### Inversion Strategy

For gravity residuals $\delta g$:

$$\mathcal{F}[\delta h] \approx -\frac{\mathcal{F}[\delta g]}{2\pi G \Delta\rho \, e^{-kZ_0}} \times \text{filter}(k) \times \text{regularization}(k)$$

**With RF Constraints (Equation 12):**

$$\delta g_{\text{total}} = \delta g_{\text{gravity}} + \lambda_{RF} \cdot \delta g_{RF}$$

---

## Example Results

### Input Parameters
- Study region: Mediterranean (~40°N, 12°E)
- Grid size: 3478 × 3478 cells
- Cell size: 300 m
- Observed gravity: 0–50 mGal (Bouguer anomaly)
- RF stations: ~40 seismic measurements

### Inversion Output
```
Iteration  1: RMS_residual = 45.2 mGal, Depth_change = 0.45 km
Iteration  2: RMS_residual = 38.2 mGal, Depth_change = 0.38 km
...
Iteration 12: [Converged]

Final Statistics:
  RMS residual: 2.35 mGal ✓
  Moho depth range: 15.2 – 28.5 km
  Mean depth: 21.1 ± 3.2 km
  Convergence: 12 iterations (< 1 minute)
```

---

## Advanced Features

### Disable RF Constraints
Edit `MohoInversion/p-o_inversion_v4.py`:
```python
USE_RF_CONSTRAINT = False
```

### Enable Flexural Isostasy
```python
USE_FLEXURE = True
TE = 25.0              # Elastic thickness (km)
```

### Adjust Regularization Strength
```python
ALPHA_TIKHONOV = 50000   # Stronger (smoother)
LOW_PASS_WL = 200        # Longer wavelengths only
```

For more options, see **[MohoInversion/README.md](./MohoInversion/README.md)**.

---

## Testing

Validate forward modeling implementation:
```bash
cd MohoInversion/
python p-o_forward_test.py
```

This creates a synthetic sine-wave Moho, computes gravity, and compares with theoretical predictions.

**Output:** `forward_test_v2.png` (validation plots)

---

## Requirements

### Core Dependencies
- Python ≥ 3.8
- NumPy ≥ 1.19
- SciPy ≥ 1.5

### Optional (Recommended)
- **pyproj** ≥ 3.0 (accurate coordinate conversion)
- **matplotlib** ≥ 3.3 (visualization)

### System Requirements
- RAM: ≥ 8 GB (for 3000×3000 grid processing)
- CPU: Multi-core recommended (FFT operations)

---

## Input Data Format

### Gravity Anomaly Grid (ASC format)
```
ncols        3478
nrows        3478
xllcorner    560000.000000
yllcorner    3239900.000000
cellsize     300.000000
nodata_value -9999.0
45.23 48.15 50.42 ...
38.12 40.56 42.89 ...
...
```

### Receiver Function Constraints (MRC format)
```
# Moho depth from seismic receiver functions
# X(m)  Y(m)  Depth(km)  Lat(deg)  Lon(deg)
NaN     NaN    22.5       40.15     12.76
NaN     NaN    20.3       40.22     12.85
...
```

---

## Output Files

| File | Description |
|------|-------------|
| `InitialMoho_Uniform.asc` | Uniform starting model |
| `FinalMoho_RF_NoTe.asc` | Inverted Moho depth (main result) |
| `inversion_log_v4.txt` | Convergence history & parameters |

All grids are in ASC format (ESRI ASCII grid), compatible with ArcGIS, QGIS, GMT, etc.

---

## References

### Key Publications
1. **Parker, R. L.** (1972). "The rapid calculation of potential anomalies." *Geophysical Journal International*, 31(5), 447–455.
   - Foundation for Fourier-domain potential field calculations

2. **Oldenburg, D. W.** (1974). "The inversion and interpretation of gravity anomalies." *Geophysics*, 39(4), 526–536.
   - Iterative depth inversion framework

3. **Receiver Function Methods** - Applied by Marco Ligi and colleagues
   - Integration of seismic constraints in gravity inversion

---

## Troubleshooting

### Memory Issues on Large Grids
```python
PADDING_RATIO = 0.3    # Reduce from 0.5
```

### Divergent Inversion
```python
LEARNING_RATE = 0.3    # Decrease from 0.5
ALPHA_TIKHONOV = 50000 # Increase regularization
```

### RF Constraints Not Applied
- Verify `USE_RF_CONSTRAINT = True`
- Check RF file path and format
- Ensure RF coordinates within grid bounds

For more troubleshooting, see **[MohoInversion/README.md § Troubleshooting](./MohoInversion/README.md#troubleshooting)**.

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{moho_inversion_2024,
  author = {Dai, Y.},
  title = {Parker-Oldenburg Moho Inversion with Receiver Function Constraints},
  year = {2024},
  url = {https://github.com/Dai411/Parker-Oldenburg-Moho-Inversion}
}
```

---

## License

[Specify your license]

---

## Contact & Support

- **Issues:** Create an issue on GitHub for bug reports and feature requests
- **Questions:** Check existing documentation first
- **Suggestions:** Pull requests are welcome

---

## Related Work

- **StitchGrids/** - Pre-processing multi-source gravity data (see [StitchGrids/README.md](./StitchGrids/README.md))
- **Data sources:** ICGEM gravity models, seismic receiver function databases

---

**Last Updated:** July 2026  
**Version:** v4.0 (RF-constrained iterative inversion)  
**Status:** ✓ Tested and operational
