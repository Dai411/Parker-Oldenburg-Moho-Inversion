import numpy as np

def compute_laplacian(data, cellsize, nodata=-9999.0):
    """
    Compute Discrete Laplacian（5-point）
    If stencil contains NaN, the output of this point is NaN
    """
    ny, nx = data.shape
    laplacian = np.full_like(data, np.nan)
    
    # 5 point finite difference
    inv_dx2 = 1.0 / (cellsize * cellsize)
    inv_dy2 = 1.0 / (cellsize * cellsize)
    
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            # Checking the existence of NaN in stencil
            stencil = data[i-1:i+2, j-1:j+2]
            if np.any(np.isnan(stencil)):
                continue
            
            # Formula of 5-point finite difference
            laplacian[i, j] = (
                (data[i+1, j] - 2*data[i, j] + data[i-1, j]) * inv_dy2 +
                (data[i, j+1] - 2*data[i, j] + data[i, j-1]) * inv_dx2
            )
    
    return laplacian
