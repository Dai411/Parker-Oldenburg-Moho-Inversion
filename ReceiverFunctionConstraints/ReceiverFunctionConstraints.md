## Receiver Function Constraints in Spectral Gravity Inversion  

This file is written by Marco Ligi

### 1.1 Receiver Function Residuals

Receiver Function data provide localized seismic estimates of crustal thickness at sparse continental stations. For each RF station \(S_{i}\), a residual between the observed seismic Moho depth and the current gravity-derived Moho model was computed as:

\[
r_{k} = H_{RF}(s_{k}) - H_{est}(s_{k}) \quad (1)
\]

where:

- \(H_{RF}(k)\) is the Moho depth estimated from Receiver Functions.  
- \(H_{est}(k)\) is the Moho depth predicted by the inversion model at each iteration.

The model prediction at the RF location \((i,j)\) was obtained through bilinear interpolation of the four neighboring grid nodes:

\[
H_{est}^{k}(i,j) = \frac{H(i - 1,j) + H(i + 1,j) + H(i,j + 1) + H(i,j - 1)}{4} \quad (2)
\]

where the four terms correspond to the surrounding grid cells.

These residuals quantify the local mismatch between seismic and gravimetric estimates of crustal thickness.

---

### 1.2 Gaussian Spatial Diffusion of RF Residuals

Because RF observations are sparse and irregularly distributed, direct imposition of pointwise constraints would introduce discontinuities incompatible with Fourier-domain inversion. Such discontinuities would generate high-frequency spectral artifacts and Gibbs oscillations during FFT operations.

To avoid this problem, RF residuals were spatially diffused over the computational grid using isotropic Gaussian kernels.

For each station \(k\), the Gaussian influence function was defined as:

\[
G_{k}(i,j) = e^{\frac{-d_{k}^{2}}{2\sigma^{2}}} \quad (3)
\]

where:

- \(d_{k}\) is the distance between the grid node \((x,y)\) and the RF station,  
- \(\sigma\) controls the spatial influence radius of the station.

The continuous residual field was then obtained through weighted averaging:

\[
R(x,y) = \sum_{k = 1}^{N_{RF}} \frac{r_k G_k(x,y)}{G_k(x,y)} \quad (4)
\]

This formulation produces a smooth residual correction field while preserving the local character of the seismic information.

The Gaussian diffusion additionally acts as a natural low-pass spatial regularization, suppressing unrealistically sharp Moho perturbations and ensuring compatibility with FFT-based spectral inversion.

---

### 1.3 Spatially Adaptive RF Weighting

A critical aspect of the inversion concerns the geographical distribution of RF stations. In the present dataset, RF observations are concentrated almost exclusively within continental regions, whereas oceanic domains contain little or no seismic control. Without additional weighting, Gaussian diffusion alone would artificially propagate continental seismic corrections into adjacent oceanic regions, where the unconstrained gravity inversion already provides reliable Moho estimates.

To prevent this effect, a spatially adaptive confidence function \(\beta (x,y)\) was introduced.

First, a local RF influence field was computed as:

\[
W_{RF}(x,y) = \sum_{k = 1}^{N_{RF}} G_k(x,y) \quad (5)
\]

This quantity measures the cumulative proximity of RF stations to each grid node.

The field was subsequently normalized:

\[
W_{N}(x,y) = \frac{W_{RF}(x,y)}{\max(W_{RF})} \quad (6)
\]

The final confidence weighting function was then defined as:

\[
\beta (x,y) = [W_N(x,y)]^\gamma \quad (7)
\]

where \(\gamma > 1\) controls the spatial localization of the RF influence.

Increasing \(\gamma\) produces a more rapid decay of RF influence away from seismic stations, effectively confining the seismic constraint to continental domains directly supported by RF observations.

---

### 1.4 Fourier-Domain Filtering of RF Constraints

The RF correction field was mirror-extended prior to Fourier transformation to reduce edge discontinuities and minimize spectral leakage. The mirrored field was then transformed into the spectral domain using a two-dimensional FFT:

\[
R_{RF}(k_{x},k_{y}) = \mathcal{F}\{R_{RF}(x,y)\} \quad (9)
\]

To ensure that RF constraints only affected regional and long-wavelength Moho structure, a Gaussian low-pass filter was applied in the wavenumber domain:

\[
F(k_{x},k_{y}) = e^{\left(\frac{k}{k_{c}}\right)^{2}} \quad (10)
\]

where:

- \(k^{2} = k_{x}^{2} + k_{y}^{2}\), and  
- \(k_{c} = 2\pi / \lambda_{c}\) with \(\lambda_{c}\) representing the characteristic cutoff wavelength.

Typical values adopted in this study ranged between 250 and \(400\,\text{km}\), consistent with the expected coherence length of regional Moho structure resolved by RF observations.

The filtered RF contribution was finally expressed as:

\[
R_{RF}^{f}(k_{x},k_{y}) = \lambda_{RF} \, F(k_{x},k_{y}) \, R_{RF}(k_{x},k_{y}) \quad (11)
\]

where \(\lambda_{RF}\) controls the global amplitude of the RF constraint relative to the gravity residual term.

---

### 1.5 Final Constrained Inversion Equation

\[
H_{n + 1}(k_{x},k_{y}) = H_{n}(k_{x},k_{y}) + P(k_{x},k_{y}) \left[ R_{g}(k_{x},k_{y}) + R_{RF}^{f}(k_{x},k_{y}) \right] \quad (12)
\]

This formulation combines gravity and seismic information within a unified spectral inversion framework while preserving numerical stability and FFT efficiency.

The adaptive weighting strategy prevents RF constraints from contaminating offshore regions and simultaneously suppresses the tendency of unconstrained gravity inversion to produce unrealistically deep onshore Moho geometries.

The proposed approach presents several important advantages:

- **i)** sparse RF observations are incorporated without imposing rigid pointwise constraints, thereby avoiding instability associated with exact interpolation of irregular seismic data;  
- **ii)** Gaussian spatial diffusion guarantees smooth transitions between constrained and unconstrained regions, minimizing spectral ringing and Gibbs artifacts in the Fourier domain;  
- **iii)** the adaptive weighting function naturally accounts for heterogeneous RF station coverage, allowing the inversion to balance seismic and gravimetric information according to local observational confidence;  
- **iv)** finally, because the method remains entirely embedded within the FFT-based Parker–Oldenburg framework, the computational efficiency of the spectral inversion is fully preserved.
