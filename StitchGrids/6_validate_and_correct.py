import argparse
import numpy as np
import os


def read_asc_grid(filename):
    """读取 ASC 文件，返回头信息和数据数组"""
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
    """写入 ASC 文件，将 NaN 转回 nodata_value"""
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
    """将 Sea 数据映射到 Land 网格，返回 Land 大小的有效掩码"""
    cellsize = land_header['cellsize']
    land_xmin = land_header['xllcorner']
    land_ymax = land_header['yllcorner'] + land_header['nrows'] * cellsize
    sea_xmin = sea_header['xllcorner']
    sea_ymax = sea_header['yllcorner'] + sea_header['nrows'] * cellsize

    sea_on_land = np.full((land_header['nrows'], land_header['ncols']), False, dtype=bool)

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
    diff = land_data[land_only_mask] - final_data[land_only_mask]
    return {
        'count': np.sum(land_only_mask),
        'mean': np.nanmean(diff),
        'std': np.nanstd(diff),
        'min': np.nanmin(diff),
        'max': np.nanmax(diff),
    }


def print_region_stats(name, data):
    print(f"{name} 形状: {data.shape}, 有效点: {np.sum(~np.isnan(data)):,}")
    print(f"  范围: min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate and optionally correct Bouguer final merge results.')
    parser.add_argument('--land', default='BouguerLandGrid.asc',
                        help='Land grid ASC file path')
    parser.add_argument('--sea', default='BouguerSeaGrid.asc',
                        help='Sea grid ASC file path')
    parser.add_argument('--final', default='BouguerFinal.asc',
                        help='Final merged Bouguer ASC file path')
    parser.add_argument('--output', default='BouguerFinal_Corrected.asc',
                        help='Output corrected file path (if --correct supplied)')
    parser.add_argument('--correct', action='store_true',
                        help='Calculate and save corrected final grid based on land-only offset')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Threshold for acceptable mean difference in land-only region (mGal)')
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.getcwd()
    land_path = os.path.join(base_dir, args.land)
    sea_path = os.path.join(base_dir, args.sea)
    final_path = os.path.join(base_dir, args.final)

    print('读取文件...')
    land_header, land_data = read_asc_grid(land_path)
    sea_header, sea_data = read_asc_grid(sea_path)
    final_header, final_data = read_asc_grid(final_path)

    print_region_stats('Land', land_data)
    print_region_stats('Sea', sea_data)
    print_region_stats('Final', final_data)

    print('将 Sea 映射到 Land 网格...')
    sea_on_land = map_sea_to_land_grid(land_header, sea_header, sea_data)
    print(f'Sea 在 Land 网格上的有效点数: {np.sum(sea_on_land):,}')

    land_only_mask = ~np.isnan(land_data) & ~sea_on_land
    print(f'Land 独有区域有效点数: {np.sum(land_only_mask):,}')

    if np.sum(land_only_mask) == 0:
        print('未找到 Land 独有区域，无法进行验证或校正。')
        return

    stats = compare_land_final(land_data, final_data, land_only_mask)
    print('\n=== Land 独有区域（未被 Sea 覆盖）验证结果 ===')
    print(f'  点数: {stats["count"]:,}')
    print(f'  均值: {stats["mean"]:.6f} mGal')
    print(f'  标准差: {stats["std"]:.6f} mGal')
    print(f'  最小值: {stats["min"]:.6f} mGal')
    print(f'  最大值: {stats["max"]:.6f} mGal')

    if abs(stats['mean']) < args.threshold:
        print(f'\n✓ 差异均值小于 {args.threshold} mGal，拼接结果在 Land 独有区域内没有显著偏移。')
    else:
        print(f'\n⚠ 注意：差异均值 = {stats["mean"]:.6f} mGal，可能存在系统性偏移。')

    if args.correct:
        print('\n计算校正偏移并保存结果...')
        offset = stats['mean']
        final_corrected = final_data + offset
        write_asc_grid(args.output, final_header, final_corrected)
        print(f'✓ 已保存校正后的结果: {args.output}')
        corrected_diff = land_data[land_only_mask] - final_corrected[land_only_mask]
        print(f'校正后差异均值: {np.nanmean(corrected_diff):.6f} mGal')
        print(f'校正后差异标准差: {np.nanstd(corrected_diff):.6f} mGal')
        print('\n建议使用校正后的结果进行后续分析。')
    else:
        print('\n未执行校正。若需要生成校正结果，可添加 --correct 选项。')


if __name__ == '__main__':
    main()
