## Nonlinear Parker-Oldenburg Moho Inversion Workflow

This document provides a comprehensive guide to the three-step workflow for inverting Moho depth from gravity anomalies with integrated seismic receiver function constraints.

**[Return to project README](../README.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Steps](#workflow-steps)
   - [Step 1: Create Uniform Initial Model](#step-1-create-uniform-initial-model)
   - [Step 2: Setup Receiver Function Constraints](#step-2-setup-receiver-function-constraints)
   - [Step 3: Iterative Inversion](#step-3-iterative-inversion)
3. [File Structure](#file-structure)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Advanced Usage](#advanced-usage)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Output Interpretation](#output-interpretation)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Moho inversion workflow consists of three sequential scripts that work together to recover the 3D topography of the Moho discontinuity:

```
Bouguer Anomaly Grid
        ↓
   [Step 1] Create uniform initial Moho model
        ↓
   [Step 2] Load RF seismic constraints
        ↓
   [Step 3] Iterative Parker-Oldenburg inversion
        ↓
   Final Moho Depth Model + Log File
```

**Key Innovation:** Integration of sparse seismic receiver function observations as soft constraints within frequency-domain gravity inversion, enabling joint gravity-seismic interpretation.

---

## Workflow Steps

### Step 1: Create Uniform Initial Model

**Script:** `create_uniform_moho.py`

**Purpose:** Generate a uniform initial Moho depth model for the inversion. This helps diagnose whether results are biased by the initial model choice and provides a stable starting point.

#### Parameters

```python
UNIFORM_DEPTH = 20   # km (Moho depth)
```

#### Workflow

1. Read ASC grid header from reference gravity file
2. Extract grid parameters:
   - Number of columns/rows
   - Georeferencing (xllcorner, yllcorner)
   - Cell size
   - NoData value
3. Create 2D array filled with uniform depth
4. Write to ASC format with proper header

#### Input Files

- `BouguerFinalWithModel.asc` - Reference gravity grid (header information only)

#### Output Files

- `InitialMoho_Uniform.asc` - Uniform Moho initial model (km)

#### Example Usage

```bash
cd MohoInversion/
python create_uniform_moho.py
```

**Output:**
```
Saved: ../MohoInversion/InitialMoho_Uniform.asc
  Uniform depth: 20 km
  Grid size: 3478 x 3478
```

---

### Step 2: Setup Receiver Function Constraints

**Script:** `rf_constraint_mercator.py`

**Purpose:** Load seismic receiver function observations and prepare them as spatial constraint fields for the gravity inversion. This module converts sparse point measurements into smooth correction fields.

#### 2.1 Data Input

**Input Format:** `.mrc` ASCII file

```
# Columns: depth(km)  lat(deg)  lon(deg)
22.5  40.1505  12.7565
20.3  40.2210  12.8530
...
```

#### 2.2 Coordinate Transformation

Converts latitude/longitude to Mercator projection (40°N standard parallel):

```python
from rf_constraint_mercator import latlon_to_mercator

lat, lon = 40.1505, 12.7565
x, y = latlon_to_mercator(lat, lon, lat0=40, lon0=0)
# Output: x=560234.5, y=3245123.8 (meters)
```

**Methods:**
- **Primary:** Uses `pyproj` library for accurate transformation
- **Fallback:** Approximate formula if pyproj unavailable

#### 2.3 RFConstraint Class

**Initialization:**

```python
from rf_constraint_mercator import RFConstraint

rf = RFConstraint(
    rf_file="path/to/MohoFromRF_40N.mrc",
    sigma=30000,           # Gaussian diffusion radius (m)
    lambda_c=300000,       # Low-pass filter cutoff wavelength (m)
    gamma=0.5              # Spatial weight exponent
)
```

**Key Methods:**

##### `precompute(grid_x, grid_y, dx, dy)`

Pre-compute weight fields to avoid repeated calculations:

```python
# Called automatically on first use
rf.precompute(grid_x, grid_y, dx, dy)
```

**What it computes:**
1. Gaussian diffusion kernels (Equation 3-4)
2. Spatial confidence weights (Equations 5-8)
3. Frequency-domain filter (Equations 9-11)

##### `get_correction(gravity_moho, grid_x, grid_y, dx, dy)`

Main function to compute RF correction field:

```python
rf_correction, n_valid = rf.get_correction(
    gravity_moho=moho_model,  # Current Moho estimate (km)
    grid_x=grid_x,            # X-coordinates (m)
    grid_y=grid_y,            # Y-coordinates (m)
    dx=300,                   # Grid spacing (m)
    dy=300
)
```

**Returns:**
- `rf_correction` - Correction field in km (same shape as input)
- `n_valid` - Number of valid RF stations used

**Workflow:**
1. Interpolate gravity model at RF station locations (bilinear)
2. Compute residuals: `residuals = RF_depth - gravity_depth`
3. Gaussian diffusion: Spread residuals using kernels
4. Apply confidence weights based on RF station density
5. Frequency-domain low-pass filtering
6. Return smoothed correction field

#### 2.4 Mathematical Background

**Gaussian Diffusion (Equations 3-4):**

$$\Delta h_{\text{raw}}(x,y) = \frac{\sum_i r_i \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right)}{\sum_i \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right)}$$

**Spatial Confidence Weight (Equations 5-8):**

$$\beta(x,y) = \left[\frac{\text{influence}(x,y)}{\max(\text{influence})}\right]^{\gamma}$$

**Frequency Filtering (Equations 9-11):**

$$\hat{H}_{\text{RF}}(k_x, k_y) = \exp\left(-\frac{k^2}{k_c^2}\right) \hat{H}(k_x, k_y)$$

where $k_c = 2\pi/\lambda_c$ and $\lambda_c$ is the cutoff wavelength.

#### 2.5 File Requirements

**Input:**
- `.mrc` file with valid RF station measurements
- Typical: 40-100 stations covering study region

**Data Format:**
```
depth(km)  lat(deg)  lon(deg)
22.5       40.15     12.76
20.3       40.22     12.85
```

---

### Step 3: Iterative Inversion

**Script:** `p-o_inversion_v4.py`

**Purpose:** Perform iterative gravity inversion with optional RF constraints to obtain the final Moho depth model.

#### 3.1 Physical Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `DRHO` | 0.60 | g/cm³ | Crustal-mantle density contrast |
| `Z0` | 20.0 | km | Average Moho depth |
| `TE` | 2.5 | km | Elastic thickness (if flexure enabled) |
| `N_MAX` | 5 | — | Parker series order (non-linear terms) |

#### 3.2 Inversion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_ITER` | 30 | Maximum number of iterations |
| `TOLERANCE` | 0.001 | Convergence tolerance (RMS relative change) |
| `LEARNING_RATE` | 0.5 | Under-relaxation factor (0 < α ≤ 1) |
| `MAX_DELTA_H` | 3.0 | Maximum depth correction per iteration (km) |
| `USE_EARLY_STOP` | True | Stop if RMS increases |

#### 3.3 Filtering Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `LOW_PASS_WL` | 150 | km | Wavelengths completely passed (long-waves preserved) |
| `HIGH_PASS_WL` | 90 | km | Wavelengths completely blocked (short-waves removed) |
| `ALPHA_TIKHONOV` | 20000 | — | Tikhonov regularization strength |
| `TIKHONOV_ORDER` | 4 | — | Tikhonov filter order (higher = smoother) |
| `MAX_DW_CONT` | 2.0 | — | Maximum downward continuation amplification |
| `PADDING_RATIO` | 0.5 | — | FFT padding ratio (50% extension) |

#### 3.4 Receiver Function Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `USE_RF_CONSTRAINT` | True | — | Enable RF constraints |
| `RF_SIGMA` | 30000 | m | Gaussian diffusion radius |
| `RF_LAMBDA_C` | 300000 | m | Low-pass filter cutoff wavelength |
| `RF_GAMMA` | 0.5 | — | Spatial weight exponent |
| `RF_LAMBDA` | 2.0 | — | RF constraint relative weight (λ_RF) |

#### 3.5 Main Inversion Loop

**Algorithm Overview:**

```
Initialize: h_current = h_initial

For iteration = 1 to MAX_ITER:
  1. Forward model: g_calc = Forward(h_current)
  2. Residuals: Δg = g_obs - g_calc
  3. RF correction (optional): Δg_rf = RFConstraint.get_correction(h_current)
  4. Combined residual: Δg_total = Δg + λ_RF × Δg_rf
  5. Inverse modeling: Δh = Inverse(Δg_total)
  6. Limit step size: Δh = clip(Δh, -MAX_DELTA_H, +MAX_DELTA_H)
  7. Apply learning rate: Δh = LEARNING_RATE × Δh
  8. Update model: h_new = h_current + Δh
  9. Check convergence: if |RMS_change| < TOLERANCE: BREAK
  10. Check divergence: if RMS_new > RMS_old: BREAK (early stop)
  11. h_current = h_new

Return: h_current (final Moho model)
```

#### 3.6 Data Preprocessing

**Step 1: Read Data**
- Load observed gravity grid (ASC format)
- Load initial Moho model
- Extract grid parameters and valid mask

**Step 2: Remove Mean & Interpolate NaNs**
```python
gravity_mean = np.nanmean(gravity_obs)
gravity_zero_mean = gravity_obs - gravity_mean
gravity_filled = fill_nans_fast(gravity_zero_mean)
```

**Step 3: Padding & Tapering**
- Pad grid to optimal FFT size: `2^n`
- Apply Tukey (cosine taper) window to minimize edge artifacts
- Typical padding: 50% extension in each direction

#### 3.7 Forward Modeling (Parker-Oldenburg)

**Frequency Domain Calculation:**

$$\hat{g}(k_x, k_y) = -2\pi G \Delta\rho \, e^{-k Z_0} \sum_{n=1}^{N_{\max}} \frac{k^{n-1}}{n!} \hat{H}^{(n)}(k_x, k_y)$$

where:
- $G = 6.67430 \times 10^{-11}$ m³/(kg·s²)
- $\Delta\rho$ = density contrast (converted to kg/m³)
- $k = \sqrt{k_x^2 + k_y^2}$ = wavenumber
- $\hat{H}^{(n)}$ = FFT of $h^n$

**Algorithm:**
```python
def parker_forward(h_rel, DRHO, Z0, dx, dy, K, upward_cont, flex_factor, N_MAX):
    sum_term = 0
    for n in range(1, N_MAX + 1):
        h_pow = h_rel ** n
        F_h_pow = fft2(h_pow)
        term = (K ** (n-1)) / factorial(n) * F_h_pow
        sum_term += term
    
    F_gravity = -2π G Δρ upward_cont × flex_factor × sum_term
    gravity = real(ifft2(F_gravity))
    return gravity * 1e5  # Convert to mGal
```

#### 3.8 Inverse Modeling

**Frequency Domain Inverse:**

$$\hat{\Delta h}(k_x, k_y) = -\frac{\hat{\Delta g}(k_x, k_y)}{2\pi G \Delta\rho \, e^{-k Z_0} \, f_{\text{flex}}(k)} \times F_{\text{combined}}(k) \times F_{\text{tikhonov}}(k)$$

**Combined Filter:**
$$F_{\text{combined}}(k) = F_{\text{lowpass}}(k) \times F_{\text{tikhonov}}(k)$$

**Cosine Low-Pass Filter:**
$$F_{\text{lp}}(k) = \begin{cases}
1.0 & k \leq k_{\text{low}} \\
0.5[1 + \cos(\pi \frac{k - k_{\text{low}}}{k_{\text{high}} - k_{\text{low}}})] & k_{\text{low}} < k < k_{\text{high}} \\
0.0 & k \geq k_{\text{high}}
\end{cases}$$

**Tikhonov Regularization Filter:**
$$F_{\text{tikhonov}}(k) = \frac{1}{1 + \alpha (k/k_{\max})^n}$$

#### 3.9 Flexural Isostasy (Optional)

If `USE_FLEXURE = True`, applies lithospheric flexure correction:

$$f_{\text{flex}}(k) = \frac{1}{1 + \frac{D k^4}{\Delta\rho g}}$$

where:
- $D = \frac{E t_e^3}{12(1-\nu^2)}$ = flexural rigidity
- $t_e$ = elastic thickness
- $E$ = Young's modulus
- $\nu$ = Poisson's ratio
- $g$ = gravitational acceleration

#### 3.10 Convergence Criteria

**Iteration Termination:**

1. **Normal Convergence:**
   ```
   if |RMS(k) - RMS(k-1)| / RMS(k-1) < TOLERANCE:
       → CONVERGED (success)
   ```

2. **Early Stopping (Divergence Detection):**
   ```
   if RMS(k) > RMS(k-1):
       → Restore previous model and STOP
   ```

3. **Maximum Iterations:**
   ```
   if k == MAX_ITER:
       → STOP (may or may not have converged)
   ```

#### 3.11 Output Files

**Primary Output:**

- `FinalMoho_RF_NoTe.asc` - Final inverted Moho depth (km)
  - Format: ESRI ASCII Grid
  - Projection: Mercator (40°N)
  - Typical range: 15-30 km

**Logging:**

- `inversion_log_v4.txt` - Complete convergence history

**Example Log:**
```
================================================================================
Parker-Oldenburg Moho Inversion v4 (with RF Constraints)
Started: 2024-07-01 12:30:45
================================================================================

Parameters:
  DRHO = 0.60 g/cm3
  Z0 = 20.0 km
  TE = 2.5 km
  USE_FLEXURE = False
  LEARNING_RATE = 0.5
  LOW_PASS_WL = 150 km
  HIGH_PASS_WL = 90 km
  ALPHA_TIKHONOV = 20000
  USE_RF_CONSTRAINT = True
  RF_SIGMA = 30.0 km
  RF_LAMBDA_C = 300.0 km
  RF_GAMMA = 0.5
  RF_LAMBDA = 2.0

1. Reading data...
   Original size: 3478 x 3478
   Valid points: 12087654
   Time: 0.45s

2. Interpolating NaNs & Removing mean...
   Gravity mean: 12.34 mGal
   Time: 0.23s

3. Padding & Tapering for FFT...
   Original: 3478 x 3478
   Padded: 4096 x 4096
   Time: 1.02s

4. Computing wavenumbers and filters...
   Flexure disabled
   Filter: K_low=4.19e-05, K_high=6.98e-05
   Downward cont max: 2.00e+00
   Time: 0.34s

5. Starting iterative inversion...
  Iter  1: RMS_res=45.23 mGal, Dh_rms=0.45 km, Change=2.3%, Time=0.8s
  Iter  2: RMS_res=38.15 mGal, Dh_rms=0.38 km, Change=1.8%, Time=0.7s
  Iter  3: RMS_res=32.67 mGal, Dh_rms=0.31 km, Change=1.4%, Time=0.7s
  ...
  Iter 12: RMS_res=2.35 mGal, Dh_rms=0.02 km, Change=0.08%, Time=0.6s

  [OK] Converged at iteration 12 (RMS change 9.8e-04 < 0.001)

6. Restoring original domain & saving...
   Min: 15.2 km
   Max: 28.5 km
   Mean: 21.1 km
   Std: 3.2 km
   [OK] Saved: .../FinalMoho_RF_NoTe.asc

================================================================================
[OK] Inversion completed!
    Total time: 45.2s (0.75 minutes)
    Final RMS residual: 2.35 mGal
    Output file: .../FinalMoho_RF_NoTe.asc
    Log file: .../inversion_log_v4.txt
================================================================================
Finished: 2024-07-01 12:31:30
```

---

## File Structure

```
MohoInversion/
├── README.md                              # This file
├── create_uniform_moho.py                 # Step 1: Initial model
├── rf_constraint_mercator.py              # Step 2: RF setup
├── p-o_inversion_v4.py                    # Step 3: Main inversion
├── p-o_forward_test.py                    # Validation script
│
├── Input/
│   ├── BouguerFinalWithModel.asc          # Observed gravity
│   └── MohoFromRF_40N.mrc                 # Seismic RF observations
│
└── Output/
    ├── InitialMoho_Uniform.asc            # Step 1 output
    ├── FinalMoho_RF_NoTe.asc              # Step 3 output
    └── inversion_log_v4.txt               # Convergence log
```

---

## Installation & Setup

### Requirements

```
Python >= 3.8
numpy >= 1.19
scipy >= 1.5
pyproj >= 3.0 (recommended, for accurate coordinate conversion)
matplotlib >= 3.3 (optional, for visualization)
```

### Installation

```bash
# Create conda environment
conda create -n moho-inversion python=3.10 numpy scipy matplotlib -y
conda activate moho-inversion

# Install optional coordinate transformation library
conda install -c conda-forge pyproj -y
```

### Configuration

Edit file paths in each script for your system:

**In `create_uniform_moho.py`:**
```python
data_dir = 'YOUR_DATA_PATH/BouguerAnomaly'
```

**In `p-o_inversion_v4.py`:**
```python
data_dir = 'YOUR_DATA_PATH/BouguerAnomaly'
stitch_dir = os.path.join(data_dir, 'StitchGrids')
moho_dir = os.path.join(data_dir, 'MohoInversion')
rf_dir = os.path.join(data_dir, 'ReceiverFunctionConstraints')

gravity_file = os.path.join(stitch_dir, 'BouguerFinalWithModel.asc')
initial_moho_file = os.path.join(moho_dir, 'InitialMoho_Uniform.asc')
rf_file = os.path.join(rf_dir, 'MohoFromRF_40N.mrc')
```

---

## Quick Start

### 1. Create Initial Model

```bash
cd MohoInversion/
python create_uniform_moho.py
```

**Output:**
```
Saved: .../MohoInversion/InitialMoho_Uniform.asc
  Uniform depth: 20 km
  Grid size: 3478 x 3478
```

### 2. Run Inversion

```bash
python p-o_inversion_v4.py
```

**Console Output:**
```
======================================================================
Parker-Oldenburg Iterative Moho Inversion v4 (with RF Constraints)
======================================================================
  Density contrast: 0.60 g/cm3
  Mean depth: 20.0 km
  Learning rate: 0.5
  RF constraint: Enabled
======================================================================

1. Reading data...
   Original size: 3478 x 3478
   Valid points: 12087654
   Time: 0.45s

2. Interpolating NaNs & Removing mean...
   Gravity mean: 12.34 mGal
   Time: 0.23s

[... iterations ...]

======================================================================
[OK] Inversion completed!
    Total time: 45.2s
    Final RMS residual: 2.35 mGal
    Output file: .../FinalMoho_RF_NoTe.asc
======================================================================
```

### 3. (Optional) Test Forward Modeling

```bash
python p-o_forward_test.py
```

Validates forward calculation with synthetic sine-wave relief:
- Creates known Moho model
- Computes gravity anomaly
- Compares with theory

**Output:** `forward_test_v2.png` (validation plots)

---

## Advanced Usage

### Disable RF Constraints

Edit `p-o_inversion_v4.py`:
```python
USE_RF_CONSTRAINT = False
```

Inversion uses gravity data alone.

### Adjust Wavelength Filters

**For longer wavelengths (smoother model):**
```python
LOW_PASS_WL = 200      # km (instead of 150)
HIGH_PASS_WL = 100     # km (instead of 90)
```

**For shorter wavelengths (more detail):**
```python
LOW_PASS_WL = 100
HIGH_PASS_WL = 50
```

### Increase Regularization (Smoother Results)

```python
ALPHA_TIKHONOV = 50000    # Increase from 20000
TIKHONOV_ORDER = 5        # Increase from 4
```

### Decrease Regularization (More Detail)

```python
ALPHA_TIKHONOV = 5000     # Decrease from 20000
```

### Enable Flexural Isostasy

```python
USE_FLEXURE = True
TE = 25.0                 # Elastic thickness (km)
E = 1e11                  # Young's modulus (Pa)
NU = 0.25                 # Poisson's ratio
```

### Adjust Learning Rate

**More conservative (slower, stable):**
```python
LEARNING_RATE = 0.3
```

**More aggressive (faster, risk of oscillation):**
```python
LEARNING_RATE = 0.8
```

### Change RF Constraint Weight

**Stronger RF influence:**
```python
RF_LAMBDA = 5.0
```

**Weaker RF influence:**
```python
RF_LAMBDA = 1.0
```

### Reduce Computation Time

```python
# Reduce padding
PADDING_RATIO = 0.3    # Instead of 0.5

# Stricter convergence
TOLERANCE = 0.01       # Instead of 0.001

# Limit iterations
MAX_ITER = 15          # Instead of 30
```

---

## Mathematical Foundation

### Parker Series Expansion

For depth interface at $z = Z_0 + h(x,y)$, gravity anomaly:

$$g(x,y) = -2\pi G \Delta\rho \sum_{n=1}^{\infty} \mathcal{F}^{-1}\left\{ e^{-kZ_0} \frac{k^{n-1}}{n!} \mathcal{F}[h^n] \right\}$$

### Iterative Inversion

For gravity residual $\delta g$:

$$\mathcal{F}[\delta h] \approx -\frac{\mathcal{F}[\delta g]}{2\pi G \Delta\rho \, e^{-kZ_0} \, f_{\text{flex}}} \times F_{\text{combined}}$$

### RF-Constrained Inversion (Equation 12)

Combined residual with RF correction:

$$\delta g_{\text{total}} = \delta g_{\text{gravity}} + \lambda_{RF} \cdot \delta g_{RF}$$

where $\lambda_{RF}$ is the relative weight of seismic constraints.

### Tikhonov Regularization

Smoothness constraint in frequency domain:

$$F_{\text{tikhonov}}(k) = \frac{1}{1 + \alpha (k/k_{\max})^n}$$

Higher $\alpha$ or $n$ → smoother solution

### Low-Pass Cosine Filter

Preserves long wavelengths, removes short wavelengths:

$$F_{\text{lp}}(k) = \begin{cases}
1.0 & \lambda > \lambda_{\text{low}} \\
0.5[1 + \cos(\pi \frac{k - k_l}{k_h - k_l})] & \lambda_{\text{low}} > \lambda > \lambda_{\text{high}} \\
0.0 & \lambda < \lambda_{\text{high}}
\end{cases}$$

---

## Output Interpretation

### Final Moho Grid

- **Format:** ASC (ESRI ASCII Grid)
- **Projection:** Mercator (40°N standard parallel)
- **Units:** km (positive downward)
- **Typical Range:** 15-30 km for continental crust

### Quality Indicators

**From `inversion_log_v4.txt`:**

1. **Final RMS Residual (mGal)**
   - Typical: 2-5 mGal
   - Indicates misfit to gravity data
   - Lower is better (but don't over-fit)

2. **Convergence Iterations**
   - Typical: 10-20 iterations
   - Fast convergence (<10 iter) = good parameter tuning
   - Slow convergence (>30 iter) = may need regularization adjustment

3. **Model Statistics**
   - Mean depth should ≈ Z0 (typically 18-22 km)
   - Std dev indicates roughness (typically 2-4 km)
   - Too smooth (std < 1 km) = over-regularized
   - Too rough (std > 5 km) = under-regularized

4. **RMS Depth Changes**
   - Should decrease monotonically
   - Final changes < 0.05 km/iteration = converged

---

## Troubleshooting

### Problem: Memory Error on Large Grids

**Solution 1: Reduce Padding**
```python
PADDING_RATIO = 0.3    # Instead of 0.5
```

**Solution 2: Process Sub-grids**
- Split grid into quadrants manually
- Run inversion on each
- Merge results (advanced)

### Problem: Divergent Inversion (RMS Increases)

**Cause:** Learning rate too large or regularization too weak

**Solution:**
```python
LEARNING_RATE = 0.3        # Reduce from 0.5
MAX_DELTA_H = 1.0          # Reduce from 3.0
ALPHA_TIKHONOV = 50000     # Increase from 20000
```

### Problem: Very Slow Convergence (>30 iterations)

**Cause:** Too much regularization or learning rate too small

**Solution:**
```python
LEARNING_RATE = 0.7        # Increase from 0.5
ALPHA_TIKHONOV = 10000     # Decrease from 20000
TOLERANCE = 0.01           # Relax from 0.001
```

### Problem: RF Constraints Not Applied

**Check:**
1. `USE_RF_CONSTRAINT = True` ✓
2. RF file exists and has correct path ✓
3. RF file has valid data (not all NaN) ✓
4. RF coordinates within grid bounds ✓

**Diagnostic:**
```python
# Add debug output in iteration loop:
if rf_constraint is not None:
    print(f"RF stations used: {n_valid}")
    print(f"RF correction: [{np.nanmin(rf_correction):.2f}, {np.nanmax(rf_correction):.2f}] km")
```

### Problem: ImportError for pyproj

**If you see:**
```
Warning: pyproj not installed. Install with: conda install -c conda-forge pyproj
```

**Solution:**
```bash
conda install -c conda-forge pyproj -y
```

**Fallback:** Code will use approximate formula if pyproj unavailable (less accurate).

### Problem: NaN Values in Output

**Cause:** Gravity or initial model has NaN outside valid domain

**Check:**
```python
# In inversion_log_v4.txt:
Valid points: XXXXX
# If this is much less than grid size, domain may be problematic
```

**Solution:**
```python
# In p-o_inversion_v4.py, increase fill tolerance:
valid_mean = np.nanmean(grid)
grid_filled[np.isnan(grid_filled)] = valid_mean
```

### Problem: Output Grid Has Wrong Georeferencing

**Check:** ASC header in `FinalMoho_RF_NoTe.asc`
```
ncols        3478
nrows        3478
xllcorner    560000.000000
yllcorner    3239900.000000
cellsize     300.000000
```

**Must match:** Input gravity grid header (set in Step 1)

---

## References

### Key Publications

1. **Parker, R. L.** (1972). "The rapid calculation of potential anomalies." *Geophysical Journal International*, 31(5), 447–455.
   - Foundation for Fourier-domain potential field calculations
   - Parker expansion formula for gravity from topography

2. **Oldenburg, D. W.** (1974). "The inversion and interpretation of gravity anomalies." *Geophysics*, 39(4), 526–536.
   - Iterative depth inversion framework
   - Non-linear least-squares approach

3. **Receiver Function Methods**
   - Integration of seismic constraints in gravity inversion
   - Applied by Marco Ligi and colleagues

### Related Resources

- [FFT-based potential field calculations](https://wiki.seg.org/)
- [Regularization in geophysical inversion](https://en.wikipedia.org/wiki/Tikhonov_regularization)
- [ICGEM Global Gravity Models](https://icgem.gfz-potsdam.de/)

---

## Citation

If you use this workflow in research, please cite:

```bibtex
@software{moho_inversion_2024,
  author = {Dai, Y.},
  title = {Parker-Oldenburg Moho Inversion with Receiver Function Constraints},
  year = {2024},
  url = {https://github.com/Dai411/Parker-Oldenburg-Moho-Inversion}
}
```

---

**Last Updated:** July 2024  
**Version:** v4.0 (RF-constrained iterative inversion)  
**Status:** ✓ Tested and operational

**[Return to project README](../README.md)**
