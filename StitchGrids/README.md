## Why Do We Need to Stitch?

### Background: Three Sources of Gravity Data

Our study involves three sets of gravity data:

1. **Parker forward modeling results**: "Correct" gravity anomalies obtained by stripping off sedimentary layers
   (strata-corrected, not free-air anomalies).
3. **Onshore gravity anomaly data**: Ground-measured gravity data (free-air anomalies).
4. **Free-air Bougurer anomalies**: Satellite/ shipborne data (free-air anomalies).

**Note**: All `.asc` files must use the same projected coordinate system.

### Why Stitching Is Necessary

#### 1. **Inconsistent Data Reference Frames**
   - Onshore and marine data are acquired using entirely different methods, though both theoretically reference the free-air anomaly datum.
   - Parker forward modeling results have undergone strata correction and therefore need to be unified with the original gravity anomalies to a common reference frame.

#### 2. **Edge Artifacts in Overlapping Regions**
   - Onshore and marine data overlap along coastal zones.
   - Direct merging produces discontinuities or step-like jumps (boundary mismatches).
   - Smooth transitions or weighted fusion are required.

#### 3. **Differences in Spatial Coverage and Resolution**
   - The spatial extent and dimensionality of onshore and marine data may differ.
   - Resolutions are often inconsistent (onshore data may be high-resolution, while marine data may be coarser, or vice versa).

#### 4. **Need for Regularized Inversion Domains**
   - After land–sea merging, the resulting grid is often irregularly shaped.
   - Subsequent Moho inversion requires a regular rectangular grid (to facilitate FFT operations and regularization constraints).
   - **Solution**: Fill boundary gaps and voids using downloaded free-air anomaly data with regular coverage (e.g., ICGEM models).

#### 5. **Optimizing Data Quality and Resolution**
   - Merged data may exhibit locally inconsistent resolutions.
   - These inconsistencies are automatically balanced in later inversion steps using **low-pass filtering** and **Tikhonov regularization**.

### 💡 Core Principle Summary

**In short**: The purpose of stitching is to create a gravity anomaly grid with a **unified datum, regular spatial extent, and appropriate resolution** for inversion. 
              This not only resolves incompatibilities among multi-source data but also provides a stable input for subsequent Moho depth inversion.
