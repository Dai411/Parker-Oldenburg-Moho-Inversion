"""
validate_bouguer_merge.py

Purpose:
    Validate Bouguer gravity merge results and optionally apply a correction
    based on the offset observed in land-only regions.

Features:
    - Read ESRI ASCII Grid (.asc) files
    - Map sea-grid coverage onto the land grid
    - Compare land-only regions between the land grid and final merged grid
    - Compute statistics (mean, standard deviation, min, max)
    - Optionally apply a systematic offset correction
    - Save corrected output grid

"""

import argparse
import numpy as np
import os


def read_asc_grid(filename):
    """Read an ASC file and return the header information and data array."""
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            value = float(line[1]) if '.' in line[1] else int(line[1])
            header[key] = value
        data = np.loadtxt(f)
        nodata = header['nodata_value']
        data[data == nodata] = np.nan
    return header, data


def write_asc_grid(filename, header, data):
    """Write an ASC file and convert NaN values back to nodata_value."""
    out_data = data.copy()
    nodata = header['nodata_value']
    out_data[np.isnan(out_data)] = nodata

    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value {header['nodata_value']}\n")
        for row in out_data:
            f.write(' '.join(f"{val:.6f}" for val in row) + '\n')


def map_sea_to_land_grid(land_header, sea_header, sea_data):
    """Map sea-grid data onto the land grid and return a valid coverage mask."""
    cellsize = land_header['cellsize']
    land_xmin = land_header['xllcorner']
    land_ymax = land_header['yllcorner'] + land_header['nrows'] * cellsize
    sea_xmin = sea_header['xllcorner']
    sea_ymax = sea_header['yllcorner'] + sea_header['nrows'] * cellsize

    sea_on_land = np.full(
        (land_header['nrows'], land_header['ncols']),
        False,
        dtype=bool
    )

    for i in range(sea_header['nrows']):
        sea_y = sea_ymax - i * cellsize
        land_row = int(round((land_ymax - sea_y) / cellsize))
        if land_row < 0 or land_row >= land_header['nrows']:
            continue

        for j in range(sea_header['ncols']):
            if np.isnan(sea_data[i, j]):
                continue

            sea_x = sea_xmin + j * cellsize
            land_col = int(round((sea_x - land_xmin) / cellsize))

            if 0 <= land_col < land_header['ncols']:
                sea_on_land[land_row, land_col] = True

    return sea_on_land


def compare_land_final(land_data, final_data, land_only_mask):
    """Calculate statistics of differences in land-only regions."""
    diff = land_data[land_only_mask] - final_data[land_only_mask]

    return {
        'count': np.sum(land_only_mask),
        'mean': np.nanmean(diff),
        'std': np.nanstd(diff),
        'min': np.nanmin(diff),
        'max': np.nanmax(diff),
    }


def print_region_stats(name, data):
    """Print summary statistics for a grid."""
    print(
        f"{name} shape: {data.shape}, "
        f"valid cells: {np.sum(~np.isnan(data)):,}"
    )
    print(
        f"  Range: min={np.nanmin(data):.2f}, "
        f"max={np.nanmax(data):.2f}, "
        f"mean={np.nanmean(data):.2f}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate and optionally correct Bouguer final merge results.'
    )

    parser.add_argument(
        '--land',
        default='BouguerLandGrid.asc',
        help='Path to land grid ASC file'
    )

    parser.add_argument(
        '--sea',
        default='BouguerSeaGrid.asc',
        help='Path to sea grid ASC file'
    )

    parser.add_argument(
        '--final',
        default='BouguerFinal.asc',
        help='Path to final merged Bouguer ASC file'
    )

    parser.add_argument(
        '--output',
        default='BouguerFinal_Corrected.asc',
        help='Path to corrected output file (used with --correct)'
    )

    parser.add_argument(
        '--correct',
        action='store_true',
        help='Calculate and save a corrected final grid based on land-only offset'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=1.0,
        help='Acceptable mean difference threshold in land-only region (mGal)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.getcwd()

    land_path = os.path.join(base_dir, args.land)
    sea_path = os.path.join(base_dir, args.sea)
    final_path = os.path.join(base_dir, args.final)

    print('Reading input files...')
    land_header, land_data = read_asc_grid(land_path)
    sea_header, sea_data = read_asc_grid(sea_path)
    final_header, final_data = read_asc_grid(final_path)

    print_region_stats('Land', land_data)
    print_region_stats('Sea', sea_data)
    print_region_stats('Final', final_data)

    print('Mapping sea coverage onto the land grid...')
    sea_on_land = map_sea_to_land_grid(land_header, sea_header, sea_data)
    print(f'Sea-covered cells on land grid: {np.sum(sea_on_land):,}')
    land_only_mask = ~np.isnan(land_data) & ~sea_on_land

    print(f'Land-only valid cells: {np.sum(land_only_mask):,}')

    if np.sum(land_only_mask) == 0:
        print('No land-only region was found. \nValidation or correction cannot be performed.')
        return

    stats = compare_land_final(land_data, final_data, land_only_mask)

    print('\n=== Validation Results for Land-Only Region ===')
    print(f'  Number of cells: {stats["count"]:,}')
    print(f'  Mean difference: {stats["mean"]:.6f} mGal')
    print(f'  Standard deviation: {stats["std"]:.6f} mGal')
    print(f'  Minimum difference: {stats["min"]:.6f} mGal')
    print(f'  Maximum difference: {stats["max"]:.6f} mGal')

    if abs(stats['mean']) < args.threshold:
        print(f'\n✓ Mean difference is below {args.threshold} mGal.
            \nNo significant offset is detected in the land-only region.')
    else:
        print(f'\n⚠ Warning: Mean difference = f'{stats["mean"]:.6f} mGal. 
            \nA systematic offset may exist.')

    if args.correct:
        print('\nComputing correction offset and saving output...')
        offset = stats['mean']
        final_corrected = final_data + offset
        write_asc_grid(args.output, final_header, final_corrected)
        print(f'✓ Corrected grid saved to: {args.output}')
        corrected_diff = (land_data[land_only_mask] - final_corrected[land_only_mask])
        print(f'Corrected mean difference: {np.nanmean(corrected_diff):.6f} mGal')
        print(f'Corrected standard deviation: {np.nanstd(corrected_diff):.6f} mGal')
        print('\nThe corrected grid is recommended for subsequent analysis.')
    else:
        print('\nNo correction was performed. \nUse the --correct option to generate a corrected result.')


if __name__ == '__main__':
    main()
