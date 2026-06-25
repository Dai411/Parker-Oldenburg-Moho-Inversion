"""
8_post_inversion_correction.py

反演后修正：在重力域恢复原始 Sea/Land 数据，并对 NoData 区域进行 DC 分量校正

核心逻辑：
1. 读取反演结果（重力值）
2. 读取原始 Land/Sea 重力值（BouguerLandGrid.asc, BouguerSeaGrid.asc）
3. 在观测区域用原始重力值替换反演值
4. NoData 区域保持反演值，并进行 DC 分量校正（用模型的 DC 分量作为目标基准）
5. 边界平滑过渡

输入：
  --final: BouguerFinal.asc (反演结果)
  --land: BouguerLandGrid.asc (原始 Land 重力值)
  --sea: BouguerSeaGrid.asc (原始 Sea 重力值)
  --model: BouguerModelled.asc (模型数据，用于 DC 校正)
  --output: BouguerFinal_Restored.asc
  --transition-width: 过渡带宽度 (像素)
  --no-smooth: 禁用边界平滑
  --no-dc-correction: 禁用 DC 分量校正

DC 校正说明：
  Laplacian 域方法会丢失 DC 分量信息，导致 NoData 区域的反演结果与模型数据
  存在系统性基准偏移。本脚本通过将 NoData 区域的 DC 分量调整为与模型数据一致，
  消除这种偏移，使 NoData 区域与观测区域在重力域自然衔接。
"""

import argparse
import numpy as np
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter


def read_asc_header(filename):
    """
    读取 ESRI ASCII 栅格文件头

    Parameters:
    -----------
    filename : str
        文件路径

    Returns:
    --------
    dict : 包含 ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value
    """
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


def read_asc_grid(filename, return_header=False):
    """
    读取 ESRI ASCII 栅格文件数据，将 nodata 值转为 NaN

    Parameters:
    -----------
    filename : str
        文件路径
    return_header : bool
        是否同时返回 header

    Returns:
    --------
    ndarray : 数据数组 (如果 return_header=False)
    tuple : (header, data) (如果 return_header=True)
    """
    header = read_asc_header(filename)
    with open(filename, 'r') as f:
        for _ in range(6):
            f.readline()
        data = np.loadtxt(f)
    nodata = header.get('nodata_value', -9999)
    data[data == nodata] = np.nan

    if return_header:
        return header, data
    return data


def write_asc_grid(filename, data, header):
    """
    将数据写入 ESRI ASCII 栅格文件，将 NaN 转为 nodata_value

    Parameters:
    -----------
    filename : str
        输出文件路径
    data : ndarray
        数据数组
    header : dict
        文件头信息
    """
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


def align_grid_to_target(src_data, src_header, target_header):
    """
    将源数据对齐到目标网格（最近邻映射）

    假设源网格和目标网格具有相同的 cellsize，仅坐标偏移不同。

    Parameters:
    -----------
    src_data : ndarray
        源数据
    src_header : dict
        源数据头信息
    target_header : dict
        目标网格头信息

    Returns:
    --------
    ndarray : 对齐后的数据 (目标网格尺寸)
    """
    target_nrows = target_header['nrows']
    target_ncols = target_header['ncols']
    target_xmin = target_header['xllcorner']
    target_ymax = target_header['yllcorner'] + target_nrows * target_header['cellsize']
    cellsize = target_header['cellsize']

    src_xmin = src_header['xllcorner']
    src_ymax = src_header['yllcorner'] + src_header['nrows'] * src_header['cellsize']

    aligned = np.full((target_nrows, target_ncols), np.nan)

    for i in range(src_header['nrows']):
        src_y = src_ymax - i * cellsize
        target_row = int(round((target_ymax - src_y) / cellsize))
        if target_row < 0 or target_row >= target_nrows:
            continue
        for j in range(src_header['ncols']):
            src_val = src_data[i, j]
            if np.isnan(src_val):
                continue
            src_x = src_xmin + j * cellsize
            target_col = int(round((src_x - target_xmin) / cellsize))
            if 0 <= target_col < target_ncols:
                aligned[target_row, target_col] = src_val

    return aligned


def apply_dc_correction(result, model_data, obs_mask, nodata_mask, transition_width, verbose=True):
    """
    对 NoData 区域进行 DC 分量校正

    用模型数据的 DC 分量作为目标基准，校正 NoData 区域的反演结果，
    并在过渡带进行平滑过渡，避免新的不连续。

    Parameters:
    -----------
    result : ndarray
        当前结果数组 (将被修改)
    model_data : ndarray
        模型数据 (用于提供 DC 分量基准)
    obs_mask : ndarray (bool)
        观测区域掩码
    nodata_mask : ndarray (bool)
        NoData 区域掩码
    transition_width : int
        过渡带宽度 (像素)
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    ndarray : 校正后的结果数组
    """
    if model_data is None:
        if verbose:
            print('   ⚠️ 未提供模型数据，跳过 DC 校正')
        return result

    # 计算 NoData 区域的均值
    nodata_mean_result = np.nanmean(result[nodata_mask])
    nodata_mean_model = np.nanmean(model_data[nodata_mask])

    # 计算校正量
    correction = nodata_mean_model - nodata_mean_result

    if verbose:
        print(f'\n   🔧 DC 分量校正:')
        print(f'      NoData 区域均值 (反演): {nodata_mean_result:.2f} mGal')
        print(f'      NoData 区域均值 (模型): {nodata_mean_model:.2f} mGal')
        print(f'      校正量: {correction:.2f} mGal')

    # 如果校正量很小，跳过
    if abs(correction) < 0.5:
        if verbose:
            print(f'      ✅ 校正量很小 (< 0.5 mGal)，跳过')
        return result

    # 应用校正到 NoData 区域
    result[nodata_mask] = result[nodata_mask] + correction

    # 在过渡带进行平滑校正（避免新的不连续）
    # 过渡带：NoData 区域中距离观测区域边界 < transition_width 的部分
    dist_to_obs = distance_transform_edt(nodata_mask)
    transition_mask = (dist_to_obs < transition_width) & (dist_to_obs > 0) & nodata_mask

    if np.sum(transition_mask) > 0:
        # 按距离加权：越靠近观测区域，校正量越小
        weight = 1 - dist_to_obs[transition_mask] / transition_width
        # 在过渡带应用部分校正
        result[transition_mask] = result[transition_mask] - correction * (1 - weight)
        if verbose:
            print(f'      过渡带平滑: {np.sum(transition_mask):,} 点')

    # 确保观测区域不受影响
    result[obs_mask] = result[obs_mask]  # 保持原值

    if verbose:
        # 校正后统计
        new_nodata_mean = np.nanmean(result[nodata_mask])
        print(f'      校正后 NoData 均值: {new_nodata_mean:.2f} mGal')

    return result


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='反演后修正：恢复原始 Sea/Land 重力值，并进行 DC 分量校正',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（启用 DC 校正）
  python 8_post_inversion_correction.py --final BouguerFinal.asc --model BouguerModelled.asc

  # 禁用 DC 校正
  python 8_post_inversion_correction.py --no-dc-correction

  # 调整过渡带宽度
  python 8_post_inversion_correction.py --transition-width 30
        """
    )
    parser.add_argument('--final', default='BouguerFinal.asc',
                        help='反演结果文件 (default: BouguerFinal.asc)')
    parser.add_argument('--land', default='BouguerLandGrid.asc',
                        help='原始 Land 重力值文件 (default: BouguerLandGrid.asc)')
    parser.add_argument('--sea', default='BouguerSeaGrid.asc',
                        help='原始 Sea 重力值文件 (default: BouguerSeaGrid.asc)')
    parser.add_argument('--model', default=None,
                        help='模型数据文件，用于 DC 校正 (default: None)')
    parser.add_argument('--output', default='BouguerFinal_Restored.asc',
                        help='输出文件 (default: BouguerFinal_Restored.asc)')
    parser.add_argument('--transition-width', type=int, default=20,
                        help='过渡带宽度 (像素) (default: 20)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='禁用边界平滑 (默认启用)')
    parser.add_argument('--no-dc-correction', action='store_true',
                        help='禁用 DC 分量校正 (默认启用)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print('=' * 60)
    print('Step 8: Post-Inversion Correction (重力域恢复 + DC 校正)')
    print('=' * 60)

    # ============ 1. 读取数据 ============
    print('\n1. 读取数据...')

    final_header, final_data = read_asc_grid(args.final, return_header=True)
    land_header = read_asc_header(args.land)
    land_data = read_asc_grid(args.land)
    sea_header = read_asc_header(args.sea)
    sea_data = read_asc_grid(args.sea)

    print(f'   反演结果: {final_data.shape}, 有效点: {np.sum(~np.isnan(final_data)):,}')
    print(f'   Land 原始数据: {land_data.shape}, 有效点: {np.sum(~np.isnan(land_data)):,}')
    print(f'   Sea 原始数据: {sea_data.shape}, 有效点: {np.sum(~np.isnan(sea_data)):,}')

    # 读取模型数据（用于 DC 校正）
    model_data = None
    if args.model and not args.no_dc_correction:
        if os.path.exists(args.model):
            model_header = read_asc_header(args.model)
            model_data_raw = read_asc_grid(args.model)
            print(f'   模型数据: {model_data_raw.shape}, 有效点: {np.sum(~np.isnan(model_data_raw)):,}')

            # 如果模型网格与目标不一致，进行对齐
            if (model_data_raw.shape != final_data.shape or
                model_header['xllcorner'] != final_header['xllcorner'] or
                model_header['yllcorner'] != final_header['yllcorner']):
                print(f'   ⚠️ 模型网格与目标网格不一致，进行对齐...')
                model_data = align_grid_to_target(model_data_raw, model_header, final_header)
                print(f'   对齐后有效点: {np.sum(~np.isnan(model_data)):,}')
            else:
                model_data = model_data_raw
        else:
            print(f'   ⚠️ 模型文件不存在: {args.model}')

    # ============ 2. 对齐到目标网格 ============
    print('\n2. 对齐 Land/Sea 到目标网格...')

    land_aligned = align_grid_to_target(land_data, land_header, final_header)
    sea_aligned = align_grid_to_target(sea_data, sea_header, final_header)

    print(f'   对齐后 Land 有效点: {np.sum(~np.isnan(land_aligned)):,}')
    print(f'   对齐后 Sea 有效点: {np.sum(~np.isnan(sea_aligned)):,}')

    # ============ 3. 创建观测区域掩码 ============
    print('\n3. 创建观测区域掩码...')

    sea_mask = ~np.isnan(sea_aligned)
    land_mask = ~np.isnan(land_aligned) & ~sea_mask
    obs_mask = sea_mask | land_mask
    nodata_mask = ~obs_mask

    print(f'   Sea 区域: {np.sum(sea_mask):,} 点')
    print(f'   Land 区域: {np.sum(land_mask):,} 点')
    print(f'   总观测区域: {np.sum(obs_mask):,} 点')
    print(f'   NoData 区域: {np.sum(nodata_mask):,} 点')

    # ============ 4. 恢复观测区域 ============
    print('\n4. 恢复观测区域的原始重力值...')

    obs_values = np.full_like(final_data, np.nan)
    obs_values[sea_mask] = sea_aligned[sea_mask]
    obs_values[land_mask] = land_aligned[land_mask]

    print(f'   观测区域原始值范围: [{np.nanmin(obs_values):.2f}, {np.nanmax(obs_values):.2f}]')
    print(f'   观测区域原始值均值: {np.nanmean(obs_values):.2f}')

    # ============ 5. 创建结果数组 ============
    print('\n5. 恢复观测区域，NoData 区域保持反演值...')

    result = final_data.copy()
    result[obs_mask] = obs_values[obs_mask]

    nodata_values = final_data[nodata_mask]
    if np.sum(nodata_mask) > 0:
        print(f'   NoData 区域反演值范围: [{np.nanmin(nodata_values):.2f}, {np.nanmax(nodata_values):.2f}]')
        print(f'   NoData 区域反演值均值: {np.nanmean(nodata_values):.2f}')

    # ============ 6. DC 分量校正 ============
    if not args.no_dc_correction and model_data is not None:
        print('\n6. DC 分量校正...')
        result = apply_dc_correction(
            result=result,
            model_data=model_data,
            obs_mask=obs_mask,
            nodata_mask=nodata_mask,
            transition_width=args.transition_width,
            verbose=True
        )
    else:
        if args.no_dc_correction:
            print('\n6. DC 分量校正: 已禁用 (--no-dc-correction)')
        else:
            print('\n6. DC 分量校正: 无模型数据，跳过')

    # ============ 7. 边界平滑 ============
    if not args.no_smooth and np.sum(nodata_mask) > 0:
        print(f'\n7. 边界平滑 (过渡带宽度={args.transition_width} 像素)...')

        # 计算到 NoData 区域的距离
        dist = distance_transform_edt(nodata_mask)
        weight = np.clip(dist / args.transition_width, 0, 1)
        weight[obs_mask] = 1.0

        # 平滑权重
        weight_smooth = gaussian_filter(weight, sigma=3)
        weight_smooth = np.clip(weight_smooth, 0, 1)

        # 对结果进行平滑
        temp = result.copy()
        temp_smooth = gaussian_filter(temp, sigma=args.transition_width / 3)

        # 只在 NoData 区域应用平滑
        result[nodata_mask] = temp_smooth[nodata_mask]
        # 重新应用观测值
        result[obs_mask] = obs_values[obs_mask]
    elif args.no_smooth:
        print('\n7. 边界平滑: 已禁用 (--no-smooth)')
    else:
        print('\n7. 边界平滑: NoData 区域为空，跳过')

    # ============ 8. 统计结果 ============
    print('\n8. 结果统计:')

    obs_result = result[obs_mask]
    nodata_result = result[nodata_mask]

    print(f'   观测区域: min={np.nanmin(obs_result):.2f}, max={np.nanmax(obs_result):.2f}, mean={np.nanmean(obs_result):.2f}')
    if np.sum(nodata_mask) > 0:
        print(f'   NoData 区域: min={np.nanmin(nodata_result):.2f}, max={np.nanmax(nodata_result):.2f}, mean={np.nanmean(nodata_result):.2f}')

    # ============ 9. 保存 ============
    print('\n9. 保存结果...')
    write_asc_grid(args.output, result, final_header)
    print(f'   ✓ 保存: {args.output}')

    # 保存 NoData 填充区域（用于检查）
    nodata_only = np.full_like(result, np.nan)
    nodata_only[nodata_mask] = result[nodata_mask]
    nodata_output = args.output.replace('.asc', '_NoDataOnly.asc')
    write_asc_grid(nodata_output, nodata_only, final_header)
    print(f'   ✓ NoData 填充区域: {nodata_output}')

    print('\n' + '=' * 60)
    print('✅ Step 8 完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()
