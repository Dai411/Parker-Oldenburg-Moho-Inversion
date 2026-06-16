**Theoretical foundation**

The object is to merge two discrete gravity anomaly (DGA) fields,
$g_{sea}(x,y)$ and $g_{land}(x,y)$, sampled on regular grids with size
$\mathrm{\Delta}x$ and $\mathrm{\Delta}y$. To eliminate the vertical
datum shift (DC offset) and incongruences without creating
high-frequency artificial steps (scarp), the merging is performed in the
Laplacian domain.

**Step 1: Geometry setup and priority masking**

Define a global domain $\Omega$ that encompasses both grids. Let
$\Omega_{sea}$ and $\Omega_{land}$ be the domains covered by the sea and
land dataset, respectively. Because $g_{sea}(x,y)$ has priority, we
define a priority mask $m(x,y)$ over $\text{Ω}$:

$$\begin{array}{r}
m(x,y) = \left\{ \begin{aligned}
1,\ \  & if\ (x,y) \in g_{sea}(x,y) \\
0,\ \  & if\ (x,y) \in g_{land}(x,y)\ and\ g_{sea} = NaN \\
NaN,\ \  & else
\end{aligned} \right.\ \#(11)
\end{array}$$

The initial composite gravity field $g_{0}(x,y)$ is established as:

$$\begin{array}{r}
g_{0}(x,y) = \left\{ \begin{aligned}
g_{sea},\ \  & if\ m(x,y) = 1 \\
g_{land},\ \  & if\ m(x,y) = 0 \\
NaN,\ \  & else
\end{aligned} \right.\ \#(12)
\end{array}$$

**Step 2: Independent Laplacian computation on the ordinary space**

To prevent the spatial discontinuity (the scalp) along the boundary from
mapping into the Laplacian as an artificial singularity, the Laplacians
must be calculated independently on each original grid before merging.
Using a standard 5-point second-order finite difference stencil, the
discrete Laplacian operator at node $(i,j)$ is defined as:

$$\begin{array}{r}
\nabla^{2}g_{i,j} = \frac{g_{i + 1,j} - 2g_{i,j} + g_{i - 1,j}}{\mathrm{\Delta}x} + \frac{g_{i,j + 1} - 2g_{i,j} + g_{i,j - 1}}{\mathrm{\Delta}y}\ \#(13)
\end{array}$$

If any node within the stencil is $NaN$, the operator outputs $NaN$:

$$\begin{array}{r}
L_{sea}(i,j) = \left\{ \begin{aligned}
\nabla^{2}g_{sea}(i,j),\ \  & if\ \forall\ m,n\  \in \ \delta_{i,j}:\ g_{sea}(m,n) \neq NaN \\
NaN,\ \  & else
\end{aligned} \right.\ \ \#(14a)
\end{array}$$

$$\begin{array}{r}
L_{land}(i,j) = \left\{ \begin{aligned}
\nabla^{2}g_{land}(i,j),\ \  & if\ \forall\ m,n\  \in \ \delta_{i,j}:\ g_{land}(m,n) \neq NaN \\
NaN,\ \  & else
\end{aligned} \right.\ \ \#(14b)
\end{array}$$

Where
$\delta_{i,j} = \{(i,j);(i + 1,j);(i - 1,j);(i,j + 1);(i,j - 1)\}$.

**Step 3: Source Term Composition and Gap Generation**

Combine the independent Laplacians into a single source grid
$L_{0}(x,y)$ using the priority mask:

$$\begin{array}{r}
L_{0}(i,j) = \left\{ \begin{aligned}
L_{sea},\ \  & if\ m(i,j) = 1 \\
L_{land},\ \  & if\ m(i,j) = 0 \\
NaN,\ \  & else
\end{aligned} \right.\ \#(15)
\end{array}$$

Because the stencils lose valid tracking points near the clipping
boundaries of $\Omega_{sea}$ and $\Omega_{land}$, a natural structural
buffer zone $\Omega_{gap}$ composed entirely of $NaN$ values are formed
along the suture line.

**Step 4: Bi-cubic Spline Interpolation of the Laplacian**

To ensure continuity in the second derivatives across the stich line,
the missing values in $\Omega_{gap}$ are reconstructed. A 2 dimensional
bi-cubic spline surface $S(x,y)$ is fitted to the valid data points of
$L_{0}(x,y)$ surrounding the gap.

For a local cell $(i,j)$ of $\Omega_{gap}$, the interpolated Laplacian
value is computed using a 4x4 neighbourhood of surrounding valid
Laplacian nodes:

$$\begin{array}{r}
L_{int}(x,y) = \sum_{m = 0}^{3}{\sum_{n = 0}^{3}{a_{mn}x^{m}y^{n}}\forall\ x,y \in \Omega_{gap}}\#(16)
\end{array}$$

The coefficients $a_{mn}$ are determined by solving a linear system that
matches the value, first derivatives ($L_{x},L_{y}$) and
cross-derivative ($L_{xy}$) at the boundaries of the valid data
clusters. This yields a continuous unified Laplacian source field
$f(i,j)$ defined over the entire active domain.

**Step 5: Inversion in the Frequency Domain**

Let $f(i,j)$ be the unified Laplacian grid obtained in Step4, and
$G(k_{x},k_{y})$ and $F(k_{x},k_{y})$ be the 2-dimensional Discrete
Fourier Transforms (DFT) of the final Bouguer anomaly $gf(i,j)$ and
$f(i,j)$ respectively.

The continuous isotropic Laplacian operator in the frequency domain
corresponds to a multiplication by
$- 4\pi^{2}({k_{x}}^{2} + {k_{y}}^{2})$. In the discrete domain, using
the 5-point finite difference stencil with spacings $\mathrm{\Delta}x$
and $\mathrm{\Delta}y$, the discrete transfer function (or filter
spectrum) $H(k_{x},k_{y})$ is derived by applying the DFT to the stencil
coefficients:

$$\begin{array}{r}
H\left( k_{x},k_{y} \right) = \frac{2}{\Delta x^{2}}\left\lbrack \cos\left( k_{x}\Delta x \right) - 1 \right\rbrack + \frac{2}{\Delta y^{2}}\left\lbrack \cos\left( k_{y}\Delta y \right) - 1 \right\rbrack\ \#(17)
\end{array}$$

The inverse solution via direct division is:

$$\begin{array}{r}
g_{f}(i,j) = \mathcal{F}^{- 1}\left\{ \frac{F(k_{x},k_{y})}{H(k_{x},k_{y})} \right\}\ \#(18)
\end{array}$$

<span class="mark">Don’t use Tikhonov Regularization:</span>

In practice, direct division $H(k_{x},k_{y})$ is problematic at zero
wavenumber where $H(0,0) = 0$. To stabilizes the inversion, Tikhonov
regularization is applied,

$$\begin{array}{r}
g_{f} = \mathcal{F}^{- 1}\left\{ \frac{F \bullet H}{\left( H^{2} + \alpha^{2} \right)} \right\}\ \#(18a)
\end{array}$$

where $\alpha$ is a small regularization parameter
($\alpha\sim 10^{- 6}\ to\ 10^{- 8}$). This formulation avoids division
by zero while maintaining continuity in the spectral domain. An
alternative approach is to apply a small threshold $\varepsilon$ such
that

$$\begin{array}{r}
H(k) = \max\left( \left| H(k) \right|,\ \varepsilon \right) \bullet sign\left( H(k) \right)\ \#(18b)
\end{array}$$
