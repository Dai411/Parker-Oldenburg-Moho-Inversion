"""
7_Laplacian_with_model_constraint.py

Enhanced model-constraint workflow:
- Automatically detect whether the merged Laplacian (L_FilledLaplacian.asc) aligns to the model grid.
- If not aligned, provide a gdalwarp suggestion and (optionally with --fix and rasterio installed) perform resampling to the model grid.
- If the provided model is Bouguer values (default), compute its Laplacian on the model grid and use that as the external constraint.
- Blend observed Laplacian and model Laplacian on the model grid using core/transition/external masks and Gaussian smoothing.

Notes:
- This script is intended to run AFTER 3_merge_and_interpolate_dual.py and BEFORE 5_frequency_inversion.py.
- Default: --model-is-bouguer is True (the model file is interpreted as Bouguer values and will be converted to Laplacian).

"""

import argparse
import os
import sys
import tempfile
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

# Optional: rasterio for automatic resampling
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

import shutil


def read_asc_header(filename):
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            if '.' in line[1] or 'e' in line[1].lower():
                value = float(line[1])
            else:
                value = int(line[1])
            header[key] = value
    return header


def read_asc_grid(filename, header=None):
    if header is None:
        header = read_asc_header(filename)
    data = np.loadtxt(filename, skiprows=6)
    nodata = header.get('nodata_value', -9999)
    data[data == nodata] = np.nan
    return data


def write_asc_grid(filename, data, header):
    output_data = data.copy()
    nodata = header.get('nodata_value', -9999)
    output_data[np.isnan(output_data)] = nodata
    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value {nodata}\n")
        for row in range(output_data.shape[0]):
            f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')


def is_integer_multiple(diff: float, cellsize: float, tol: float = 1e-6):
    try:
        q = diff / cellsize
n    except Exception:
        return False, 0
    nearest = round(q)
    return abs(q - nearest) <= tol, int(nearest)


def align_grid_to_model(src_data, src_header, model_header):
    """Map src grid values to the model grid by integer-index rounding.
    Assumes same cellsize; if not, behaviour is undefined.
    """
    model_nrows = model_header['nrows']
    model_ncols = model_header['ncols']
    model_xmin = model_header['xllcorner']
    model_ymax = model_header['yllcorner'] + model_header['nrows'] * model_header['cellsize']
    cellsize = model_header['cellsize']

    src_xmin = src_header['xllcorner']
    src_ymax = src_header['yllcorner'] + src_header['nrows'] * src_header['cellsize']

    aligned = np.full((model_nrows, model_ncols), np.nan)

    for i in range(src_header['nrows']):
        src_y = src_ymax - i * cellsize
        model_row = int(round((model_ymax - src_y) / cellsize))
        if model_row < 0 or model_row >= model_nrows:
            continue
        for j in range(src_header['ncols']):
            src_val = src_data[i, j]
            if np.isnan(src_val):
                continue
            src_x = src_xmin + j * cellsize
            model_col = int(round((src_x - model_xmin) / cellsize))
            if 0 <= model_col < model_ncols:
                aligned[model_row, model_col] = src_val
    return aligned


def suggest_gdalwarp_command(src: str, dst: str, target_hdr: dict, resample: str = 'bilinear') -> str:
    cell = float(target_hdr['cellsize'])
    xmin = float(target_hdr.get('xllcorner', target_hdr.get('xllcenter', 0)))
    ymin = float(target_hdr.get('yllcorner', target_hdr.get('yllcenter', 0)))
    ncols = int(target_hdr['ncols'])
    nrows = int(target_hdr['nrows'])
    xmax = xmin + ncols * cell
    ymax = ymin + nrows * cell
    cmd = f"gdalwarp -te {xmin} {ymin} {xmax} {ymax} -tr {cell} {cell} -r {resample} \"{src}\" \"{dst}\""
    return cmd


def compute_laplacian(data: np.ndarray, cellsize: float) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    # handle NaNs
    nan_mask = np.isnan(data)
    data_clean = np.nan_to_num(data, nan=0.0)
    from scipy.signal import convolve2d
    lap = convolve2d(data_clean, kernel, mode='same', boundary='symm')
    lap = lap / (cellsize * cellsize)
    # extended nan for stencil
    extended_nan = nan_mask.copy()
    if nan_mask.shape[0] > 1:
        extended_nan[1:, :] |= nan_mask[:-1, :]
        extended_nan[:-1, :] |= nan_mask[1:, :]
    if nan_mask.shape[1] > 1:
        extended_nan[:, 1:] |= nan_mask[:, :-1]
        extended_nan[:, :-1] |= nan_mask[:, 1:]
    lap[extended_nan] = np.nan
    return lap


def resample_to_model_with_rasterio(src_data, src_header, model_header, resample_method='bilinear'):
    if not HAS_RASTERIO:
        raise RuntimeError('rasterio not available')
    src_cell = float(src_header['cellsize'])
    dst_cell = float(model_header['cellsize'])
    if abs(src_cell - dst_cell) > 1e-9:
        # different cellsize: we still allow resample
        pass

    src_xmin = float(src_header['xllcorner'])
    src_ymax = float(src_header['yllcorner']) + src_header['nrows'] * src_header['cellsize']
    dst_xmin = float(model_header['xllcorner'])
    dst_ymax = float(model_header['yllcorner']) + model_header['nrows'] * model_header['cellsize']

    src_transform = from_origin(src_xmin, src_ymax, src_cell, src_cell)
    dst_transform = from_origin(dst_xmin, dst_ymax, dst_cell, dst_cell)

    src_arr = src_data.astype('float32')
    src_nodata = -9999.0
    src_arr_filled = np.where(np.isnan(src_arr), src_nodata, src_arr)

    dst_shape = (model_header['nrows'], model_header['ncols'])
    dst_arr = np.full(dst_shape, src_nodata, dtype='float32')

    method_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }
    resampling = method_map.get(resample_method, Resampling.bilinear)

    reproject(
        source=src_arr_filled,
        destination=dst_arr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=None,
        dst_crs=None,
        src_nodata=src_nodata,
        dst_nodata=src_nodata,
        resampling=resampling
    )

    dst_arr = np.where(dst_arr == src_nodata, np.nan, dst_arr)
    return dst_arr


def parse_args():
    parser = argparse.ArgumentParser(description='Apply model constraint to a merged Laplacian grid (robust to grid alignment).')
    parser.add_argument('--laplacian', default='L_FilledLaplacian.asc', help='Input Laplacian file (merged & filled)')
    parser.add_argument('--our', default='L_FilledLaplacian.asc', help='Our merged/interpolated data (used to define valid/core regions)')
    parser.add_argument('--model', default='BouguerModelled.asc', help='Model file (Bouguer values or Laplacian)')
    parser.add_argument('--output', default='BouguerLaplacianConstrained.asc', help='Output constrained Laplacian file')
    parser.add_argument('--transition-sigma', type=float, default=20.0, help='Gaussian sigma for transition weight (pixels)')
    parser.add_argument('--safety-padding', type=int, default=10, help='Erosion iterations to define core region (pixels)')
    parser.add_argument('--boundary-width', type=int, default=5, help='Boundary layer width for offset estimation (pixels)')
    parser.add_argument('--model-is-bouguer', dest='model_is_bouguer', action='store_true', help='Treat model file as Bouguer values and compute its Laplacian first')
    parser.add_argument('--no-model-is-bouguer', dest='model_is_bouguer', action='store_false', help='Do not compute Laplacian from model; treat model as Laplacian')
    parser.set_defaults(model_is_bouguer=True)
    parser.add_argument('--resample-method', choices=['nearest', 'bilinear', 'cubic'], default='bilinear', help='Resampling method when resampling to model grid')
    parser.add_argument('--fix', action='store_true', help='If set and rasterio available, perform automatic resampling to model grid; otherwise script will print gdalwarp suggestion and exit')
    return parser.parse_args()


def main():
    args = parse_args()

    laplacian_file = args.laplacian
    our_file = args.our
    model_file = args.model
    output_file = args.output

    TRANSITION_SIGMA = args.transition_sigma
    SAFETY_PADDING = args.safety_padding
    BOUNDARY_WIDTH = args.boundary_width

    print('=' * 60)
    print('Laplacian field + model-constraint correction (robust)')
    print('=' * 60)

    # 1. Read headers and data
    print('\n1. Reading input files...')
    model_header = read_asc_header(model_file)
    our_header = read_asc_header(our_file)
    lap_header = read_asc_header(laplacian_file)

    model_data = read_asc_grid(model_file, model_header)
    our_data = read_asc_grid(our_file, our_header)
    lap_data = read_asc_grid(laplacian_file, lap_header)

    print(f'   Model data: {model_data.shape}, valid cells: {np.sum(~np.isnan(model_data))}')
    print(f'   Our data: {our_data.shape}, valid cells: {np.sum(~np.isnan(our_data))}')
    print(f'   Laplacian: {lap_data.shape}, valid cells: {np.sum(~np.isnan(lap_data))}')

    # Quick check: cellsize equality
    if abs(float(model_header['cellsize']) - float(lap_header['cellsize'])) > 1e-9:
        print('Warning: cellsize differs between Laplacian and model. Resampling will be required.')

    # 2. Align or resample laplacian & our data to model grid
    print('\n2. Aligning/resampling Laplacian to model grid...')
    src_cell = float(lap_header['cellsize'])
    model_cell = float(model_header['cellsize'])

    dx = float(our_header['xllcorner']) - float(model_header['xllcorner'])
    dy = float(our_header['yllcorner']) - float(model_header['yllcorner'])

    aligned_direct = False
    if abs(src_cell - model_cell) < 1e-9:
        okx = abs(round(dx / model_cell) - (dx / model_cell)) < 1e-6
        oky = abs(round(dy / model_cell) - (dy / model_cell)) < 1e-6
        if okx and oky:
            aligned_direct = True

    if aligned_direct:
        print('   Detected integer-grid alignment -> using fast integer mapping')
        our_aligned = align_grid_to_model(our_data, our_header, model_header)
        lap_aligned = align_grid_to_model(lap_data, lap_header, model_header)
    else:
        print('   Not integer-aligned')
        if args.fix and HAS_RASTERIO:
            print('   --fix set and rasterio found: performing automatic resampling to model grid...')
            try:
                our_aligned = resample_to_model_with_rasterio(our_data, our_header, model_header, args.resample_method)
                lap_aligned = resample_to_model_with_rasterio(lap_data, lap_header, model_header, args.resample_method)
            except Exception as e:
                print('   Error during rasterio resampling:', e)
                print('   Aborting. You can install rasterio or run the gdalwarp suggestion printed below.')
                sys.exit(1)
        else:
            # suggest gdalwarp
            suggested_dst = os.path.splitext(os.path.basename(laplacian_file))[0] + '_on_model_grid.asc'
            gcmd = suggest_gdalwarp_command(laplacian_file, suggested_dst, model_header, resample=args.resample_method)
            print('   Automatic resampling not performed. Suggested gdalwarp command:')
            print('   ', gcmd)
            if shutil.which('gdalwarp'):
                print('   gdalwarp found on PATH; you can run the command above to create a resampled file and then rerun this script.')
            else:
                print('   Note: gdalwarp not found on PATH. Install GDAL or rerun with --fix and rasterio available to enable automatic resampling.')
            sys.exit(2)

    print(f'   Our aligned valid cells: {np.sum(~np.isnan(our_aligned))}')
    print(f'   Laplacian aligned valid cells: {np.sum(~np.isnan(lap_aligned))}')

    # 3. If model is Bouguer values, compute its Laplacian on model grid
    if args.model_is_bouguer:
        print('\n3. Computing model Laplacian from Bouguer values (model_is_bouguer=True)')
        model_lap = compute_laplacian(model_data, float(model_header['cellsize']))
    else:
        print('\n3. Treating provided model as Laplacian (model_is_bouguer=False)')
        model_lap = model_data.copy()

    # 4. Identify regions on model grid
    print('\n4. Identifying regions (core / transition / boundary / external)')
    our_valid_mask = ~np.isnan(our_aligned)
    core_mask = binary_erosion(our_valid_mask, iterations=SAFETY_PADDING)
    boundary_mask = binary_dilation(our_valid_mask, iterations=BOUNDARY_WIDTH) & ~our_valid_mask
    boundary_mask = boundary_mask & ~np.isnan(lap_aligned)
    external_mask = ~np.isnan(model_lap) & ~our_valid_mask
    transition_mask = our_valid_mask & ~core_mask

    print(f'   Core cells: {np.sum(core_mask)}')
    print(f'   Transition cells: {np.sum(transition_mask)}')
    print(f'   Boundary-layer cells (for offset estimate): {np.sum(boundary_mask)}')
    print(f'   External region cells: {np.sum(external_mask)}')

    # 5. Estimate external offset using boundary layer (model_lap - lap_aligned)
    print('\n5. Estimating offset in the external region...')
    if np.sum(boundary_mask) > 0:
        boundary_diff = model_lap[boundary_mask] - lap_aligned[boundary_mask]
        offset_global = np.nanmean(boundary_diff)
        offset_std = np.nanstd(boundary_diff)
        print(f"   Boundary-layer difference (model - laplacian): mean = {offset_global:.4f}, std = {offset_std:.4f}")
    else:
        print('   Warning: boundary layer is empty; using offset = 0')
        offset_global = 0.0

    # 6. Create transition weights
    print('\n6. Creating transition weight field...')
    weight_initial = np.zeros_like(model_lap, dtype=float)
    weight_initial[core_mask] = 1.0
    weight_smooth = gaussian_filter(weight_initial, sigma=TRANSITION_SIGMA)
    weight_smooth = np.clip(weight_smooth, 0.0, 1.0)
    weight_smooth[core_mask] = 1.0
    print(f"   Weight range: [{np.nanmin(weight_smooth):.4f}, {np.nanmax(weight_smooth):.4f}]")

    # 7. Blend and produce constrained Laplacian on model grid
    print('\n7. Applying model-constraint correction on model grid...')
    offset_field = offset_global * (1.0 - weight_smooth)
    g_corrected = lap_aligned.copy()

    # External region: use model Laplacian values
    g_corrected[external_mask] = model_lap[external_mask]

    # Transition: blend
    transition_rows, transition_cols = np.where(transition_mask)
    for r, c in zip(transition_rows, transition_cols):
        w = weight_smooth[r, c]
        lap_val = lap_aligned[r, c]
        model_val = model_lap[r, c]
        if not np.isnan(lap_val) and not np.isnan(model_val):
            g_corrected[r, c] = (1.0 - w) * model_val + w * lap_val
        elif not np.isnan(lap_val):
            g_corrected[r, c] = lap_val
        elif not np.isnan(model_val):
            g_corrected[r, c] = model_val

    # Core: keep observed Laplacian
    g_corrected[core_mask] = lap_aligned[core_mask]

    # 8. Stats & verification
    print('\n8. Statistics:')
    print(f"\n   Pre-correction Laplacian global mean: {np.nanmean(lap_aligned):.4f}")
    if np.sum(core_mask) > 0:
        print(f"   Pre-correction core mean: {np.nanmean(lap_aligned[core_mask]):.4f}")
    print(f"\n   Post-correction global mean: {np.nanmean(g_corrected):.4f}")
    if np.sum(core_mask) > 0:
        print(f"   Post-correction core mean: {np.nanmean(g_corrected[core_mask]):.4f}")
    if np.sum(external_mask) > 0:
        print(f"   External region mean: {np.nanmean(g_corrected[external_mask]):.4f}")

    if np.sum(core_mask) > 0:
        core_unchanged = np.allclose(g_corrected[core_mask], lap_aligned[core_mask], equal_nan=True)
        print(f"   Core unchanged: {'✓ PASS' if core_unchanged else '✗ FAIL'}")

    if np.sum(external_mask) > 0:
        external_match = np.allclose(g_corrected[external_mask], model_lap[external_mask], equal_nan=True, rtol=1e-5)
        print(f"   External matches model: {'✓ PASS' if external_match else '✗ FAIL'}")

    # 9. Save
    print('\n9. Saving result (model grid header)...')
    write_asc_grid(output_file, g_corrected, model_header)
    print(f'   ✓ Saved: {output_file}')
    print('\n' + '=' * 60)
    print('✅ Laplacian + model-constraint complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
