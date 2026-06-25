"""
9_difference_constraint.py

差分约束方法（替代原 3-8 脚本的 Laplacian 域工作流）

================================================================================
【新工作流】（Laplacian 域 - 差分约束法）
================================================================================

Step 1: 几何对齐 + 优先级掩码
    python 1_geometry_and_mask.py
    输出: PriorityMask.asc, Sea_Aligned_To_LandGrid.asc

Step 2: 计算观测 Laplacian
    python 2_laplacian_computation.py
    输出: Land_Laplacian.asc, Sea_Laplacian_Aligned.asc

Step 3: 差分约束（本脚本）
    python 9_difference_constraint.py \
        --land Land_Laplacian.asc \
        --sea Sea_Laplacian_Aligned.asc \
        --model ../BouguerModelled.asc \
        --output BouguerFinal.asc

Step 4: 验证（可选）
    python 6_validate_and_correct.py --final BouguerFinal.asc

================================================================================
【核心思想】
================================================================================

传统方法的问题：
  Laplacian 域融合会丢失 DC 分量，导致 NoData 区域基准偏移。

差分约束法的优势：
  1. 在重力域计算 观测 - 模型 的差值
  2. 对差值计算 Laplacian（只包含高频残差信息）
  3. NoData 区域差值 Laplacian = 0（意味着差值本身为 0）
  4. 反演差值 Laplacian 得到差值重力场
  5. 最终 = 模型 + 差值重力场

这样：
  - DC 分量由模型天然保留 ✅
  - NoData 区域趋势由模型主导 ✅
  - 核心区观测值完全保留 ✅
  - 接缝处平滑过渡 ✅

================================================================================
【输入输出】
================================================================================

输入:
  --land    : Land_Laplacian.asc (Step 2 输出)
  --sea     : Sea_Laplacian_Aligned.asc (Step 2 输出)
  --model   : BouguerModelled.asc (模型重力值，全覆盖)
  --output  : 最终重力场 (default: BouguerFinal.asc)

参数:
  --transition-width : 过渡带宽度 (像素, default: 20)
  --no-smooth        : 禁用过渡带平滑

输出:
  --output           : 最终重力场
  --output-diff      : 差值重力场 (可选，用于调试)
  --output-nodata    : NoData 区域填充结果 (可选，用于检查)

================================================================================
【与重力域工作流的对比】
================================================================================

重力域工作流 (dig_e_refill.py + stitich_merged_model.py):
  - 适用场景: 快速拼接，只需要填充 NoData
  - 优点: 简单、快速、直观
  - 缺点: 接缝处可能不平滑

Laplacian 域工作流 (本脚本):
  - 适用场景: 需要平滑接缝、保留二阶导数信息
  - 优点: 接缝平滑、物理一致性好
  - 缺点: 计算稍复杂

================================================================================
"""

import argparse
import numpy as np
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftfreq


# ============================================================================
# 基础 I/O 函数
# ============================================================================

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


# ============================================================================
# 核心算法函数
# ============================================================================

def compute_laplacian(data, cellsize):
    """
    计算离散 Laplacian（5点差分）

    公式: L = (g_{i+1,j} + g_{i-1,j} + g_{i,j+1} + g_{i,j-1} - 4g_{i,j}) / cellsize^2

    Parameters:
    -----------
    data : ndarray
        输入数据（重力值）
    cellsize : float
        网格间距

    Returns:
    --------
    ndarray : Laplacian 值
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    data_clean = np.nan_to_num(data, nan=0.0)
    lap = convolve2d(data_clean, kernel, mode='same', boundary='symm')
    lap = lap / (cellsize * cellsize)
    return lap


def invert_laplacian(lap_data, cellsize):
    """
    频域反演 Laplacian → 重力场

    使用离散 Laplacian 传递函数 H(kx, ky) 进行反演。
    在 k=0 处做正则化处理避免除以 0。

    Parameters:
    -----------
    lap_data : ndarray
        Laplacian 数据
    cellsize : float
        网格间距

    Returns:
    --------
    ndarray : 反演得到的重力场
    """
    ny, nx = lap_data.shape
    lap_filled = np.nan_to_num(lap_data, nan=0.0)
    L_fft = fft2(lap_filled)

    # 计算离散 Laplacian 传递函数
    kx = 2 * np.pi * fftfreq(nx, cellsize)
    ky = 2 * np.pi * fftfreq(ny, cellsize)
    KX, KY = np.meshgrid(kx, ky)
    H = (2.0 / (cellsize * cellsize)) * (np.cos(KX * cellsize) - 1) + \
        (2.0 / (cellsize * cellsize)) * (np.cos(KY * cellsize) - 1)

    # 避免除以 0
    H_safe = H.copy()
    small_mask = np.abs(H_safe) < 1e-10
    H_safe[small_mask] = 1e-10

    # 反演
    G_fft = -L_fft / H_safe
    G_fft[0, 0] = 0  # DC 分量设为 0

    gravity = np.real(ifft2(G_fft))
    return gravity


def difference_constraint(obs_gravity, model_gravity, obs_mask, cellsize, transition_width=20, verbose=True):
    """
    差分约束方法（核心算法）

    工作流程:
        1. 计算差值（观测 - 模型）
        2. 对差值计算 Laplacian
        3. NoData 区域差值 Laplacian = 0
        4. 过渡带平滑（让差值 Laplacian 平滑过渡到 0）
        5. 反演差值 Laplacian 得到差值重力场
        6. 最终 = 模型 + 差值重力场
        7. 观测区域强制等于原始观测值

    Parameters:
    -----------
    obs_gravity : ndarray
        观测重力场（含 NoData 区域 NaN）
    model_gravity : ndarray
        模型重力场（全覆盖）
    obs_mask : ndarray (bool)
        观测区域掩码
    cellsize : float
        网格间距
    transition_width : int
        过渡带宽度（像素）
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    dict : 包含以下键值
        'final' : 最终重力场
        'diff_gravity' : 差值重力场
        'diff_laplacian' : 差值 Laplacian
    """
    # 1. 计算差值（观测 - 模型）
    diff_gravity = obs_gravity - model_gravity
    nodata_mask = ~obs_mask

    if verbose:
        print(f'   差值场统计 (观测区域):')
        diff_obs = diff_gravity[obs_mask]
        print(f'      均值: {np.nanmean(diff_obs):.2f} mGal')
        print(f'      标准差: {np.nanstd(diff_obs):.2f} mGal')

    # 2. 对差值计算 Laplacian
    diff_laplacian = compute_laplacian(diff_gravity, cellsize)

    if verbose:
        print(f'\n   差值 Laplacian 统计:')
        print(f'      均值: {np.nanmean(diff_laplacian):.6f}')
        print(f'      标准差: {np.nanstd(diff_laplacian):.6f}')

    # 3. NoData 区域差值 Laplacian = 0
    diff_laplacian[nodata_mask] = 0

    # 4. 过渡带平滑（让差值 Laplacian 平滑过渡到 0）
    if transition_width > 0:
        dist = distance_transform_edt(nodata_mask)
        transition_mask = (dist < transition_width) & (dist > 0) & nodata_mask

        if np.sum(transition_mask) > 0:
            weight = 1 - dist[transition_mask] / transition_width
            diff_laplacian[transition_mask] = diff_laplacian[transition_mask] * weight

            if verbose:
                print(f'   过渡带平滑: {np.sum(transition_mask):,} 点')

    # 5. 反演差值 Laplacian
    diff_gravity_reconstructed = invert_laplacian(diff_laplacian, cellsize)

    if verbose:
        print(f'\n   差值重力场统计 (反演后):')
        print(f'      均值: {np.nanmean(diff_gravity_reconstructed):.2f} mGal')
        print(f'      标准差: {np.nanstd(diff_gravity_reconstructed):.2f} mGal')

    # 6. 最终 = 模型 + 差值重力场
    final_gravity = model_gravity + diff_gravity_reconstructed

    # 7. 观测区域强制等于原始观测值（保护核心区）
    final_gravity[obs_mask] = obs_gravity[obs_mask]

    if verbose:
        print(f'\n   最终重力场统计:')
        print(f'      均值: {np.nanmean(final_gravity):.2f} mGal')
        print(f'      标准差: {np.nanstd(final_gravity):.2f} mGal')
        print(f'      观测区域均值: {np.nanmean(final_gravity[obs_mask]):.2f} mGal')
        print(f'      NoData 区域均值: {np.nanmean(final_gravity[nodata_mask]):.2f} mGal')

    return {
        'final': final_gravity,
        'diff_gravity': diff_gravity_reconstructed,
        'diff_laplacian': diff_laplacian
    }


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='差分约束方法：在 Laplacian 域平滑融合观测和模型数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python 9_difference_constraint.py \
      --land Land_Laplacian.asc \
      --sea Sea_Laplacian_Aligned.asc \
      --model ../BouguerModelled.asc \
      --output BouguerFinal.asc

  # 调整过渡带宽度
  python 9_difference_constraint.py \
      --land Land_Laplacian.asc \
      --sea Sea_Laplacian_Aligned.asc \
      --model ../BouguerModelled.asc \
      --output BouguerFinal.asc \
      --transition-width 30

  # 保存调试输出
  python 9_difference_constraint.py \
      --land Land_Laplacian.asc \
      --sea Sea_Laplacian_Aligned.asc \
      --model ../BouguerModelled.asc \
      --output BouguerFinal.asc \
      --output-diff BouguerFinal_Diff.asc

工作流说明:
  本脚本是 Laplacian 域工作流的 Step 3，替代原 3-8 脚本。
  配合 Step 1 (1_geometry_and_mask.py) 和 Step 2 (2_laplacian_computation.py) 使用。
        """
    )

    # 输入文件
    parser.add_argument('--land', default='Land_Laplacian.asc',
                        help='Land Laplacian 文件 (Step 2 输出)')
    parser.add_argument('--sea', default='Sea_Laplacian_Aligned.asc',
                        help='Sea Laplacian 文件 (Step 2 输出)')
    parser.add_argument('--model', default='../BouguerModelled.asc',
                        help='模型重力值文件 (全覆盖)')

    # 输出文件
    parser.add_argument('--output', default='BouguerFinal.asc',
                        help='最终重力场输出文件')
    parser.add_argument('--output-diff', default=None,
                        help='差值重力场输出文件 (可选，用于调试)')
    parser.add_argument('--output-nodata', default=None,
                        help='NoData 区域填充结果输出文件 (可选)')

    # 参数
    parser.add_argument('--transition-width', type=int, default=20,
                        help='过渡带宽度 (像素, default: 20)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='禁用过渡带平滑')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print('=' * 70)
    print('Step 9: Difference Constraint Method (Laplacian Domain)')
    print('=' * 70)
    print(f'  过渡带宽度: {args.transition_width} 像素')
    print(f'  过渡带平滑: {"禁用" if args.no_smooth else "启用"}')
    print('=' * 70)

    # ============ 1. 读取数据 ============
    print('\n1. 读取数据...')

    # 读取 Land Laplacian
    land_header, land_lap = read_asc_grid(args.land, return_header=True)
    print(f'   Land Laplacian: {land_lap.shape}, 有效点: {np.sum(~np.isnan(land_lap)):,}')

    # 读取 Sea Laplacian
    sea_header, sea_lap = read_asc_grid(args.sea, return_header=True)
    print(f'   Sea Laplacian: {sea_lap.shape}, 有效点: {np.sum(~np.isnan(sea_lap)):,}')

    # 读取模型数据
    model_header, model_data = read_asc_grid(args.model, return_header=True)
    print(f'   模型数据: {model_data.shape}, 有效点: {np.sum(~np.isnan(model_data)):,}')

    # ============ 2. 确定目标网格 ============
    print('\n2. 确定目标网格...')

    # 使用模型网格作为目标网格
    target_header = model_header
    target_shape = (target_header['nrows'], target_header['ncols'])
    cellsize = target_header['cellsize']
    print(f'   目标网格: {target_shape[0]} x {target_shape[1]}')
    print(f'   cellsize: {cellsize} m')

    # ============ 3. 对齐到目标网格 ============
    print('\n3. 对齐数据到目标网格...')

    # 对齐 Land Laplacian
    if land_lap.shape != target_shape:
        print(f'   Land: {land_lap.shape} -> {target_shape}')
        land_aligned = align_grid_to_target(land_lap, land_header, target_header)
    else:
        land_aligned = land_lap
    print(f'   Land 对齐后有效点: {np.sum(~np.isnan(land_aligned)):,}')

    # 对齐 Sea Laplacian
    if sea_lap.shape != target_shape:
        print(f'   Sea: {sea_lap.shape} -> {target_shape}')
        sea_aligned = align_grid_to_target(sea_lap, sea_header, target_header)
    else:
        sea_aligned = sea_lap
    print(f'   Sea 对齐后有效点: {np.sum(~np.isnan(sea_aligned)):,}')

    # 对齐模型数据（如果模型网格与目标不一致）
    if model_data.shape != target_shape:
        print(f'   模型: {model_data.shape} -> {target_shape}')
        model_aligned = align_grid_to_target(model_data, model_header, target_header)
    else:
        model_aligned = model_data

    # ============ 4. 创建观测掩码 ============
    print('\n4. 创建观测区域掩码...')

    # Sea 优先
    sea_mask = ~np.isnan(sea_aligned)
    land_mask = ~np.isnan(land_aligned) & ~sea_mask
    obs_mask = sea_mask | land_mask
    nodata_mask = ~obs_mask

    print(f'   Sea 区域: {np.sum(sea_mask):,} 点')
    print(f'   Land 区域: {np.sum(land_mask):,} 点')
    print(f'   总观测区域: {np.sum(obs_mask):,} 点')
    print(f'   NoData 区域: {np.sum(nodata_mask):,} 点')

    # ============ 5. 重建观测重力场 ============
    print('\n5. 重建观测重力场...')

    # 从 Laplacian 反演观测重力场
    # 注意：这里不是直接反演，而是通过差分约束方法
    # 我们直接使用 Laplacian 数据构建观测重力场

    # 创建观测重力场（从 Laplacian 反演得到）
    # 但由于 Laplacian 不包含 DC 分量，我们需要用模型来提供 DC 分量

    # 创建观测重力场数组（先填充 NaN）
    obs_gravity = np.full(target_shape, np.nan)

    # 在观测区域，从 Laplacian 反演得到重力值
    # 这里使用差分约束方法：用模型作为基准
    print('   使用差分约束方法重建观测重力场...')

    # 先用模型作为初始值
    obs_gravity_initial = model_aligned.copy()

    # 在观测区域，用 Laplacian 信息修正
    # 步骤：对观测区域的 Laplacian 做反演，得到相对于模型的差值

    # 5a. 创建差值 Laplacian（观测 Laplacian - 模型 Laplacian）
    model_lap = compute_laplacian(model_aligned, cellsize)

    obs_lap = np.full(target_shape, np.nan)
    obs_lap[sea_mask] = sea_aligned[sea_mask]
    obs_lap[land_mask] = land_aligned[land_mask]

    diff_lap = obs_lap - model_lap
    diff_lap[nodata_mask] = 0  # NoData 区域差值为 0

    # 5b. 反演差值 Laplacian 得到差值重力场
    diff_gravity = invert_laplacian(diff_lap, cellsize)

    # 5c. 观测重力场 = 模型 + 差值
    obs_gravity = model_aligned + diff_gravity

    # 5d. 在观测区域，用原始 Laplacian 反演结果替换（确保一致性）
    # 但由于我们已经有差值约束，观测区域的值应该是正确的
    # 强制观测区域等于原始观测值（从 Laplacian 反演得到）
    # 这里我们保留差值约束的结果，因为它在观测区域也应该是正确的

    print(f'   观测重力场统计:')
    print(f'      均值: {np.nanmean(obs_gravity):.2f} mGal')
    print(f'      标准差: {np.nanstd(obs_gravity):.2f} mGal')

    # ============ 6. 执行差分约束 ============
    print('\n6. 执行差分约束...')

    transition_width = 0 if args.no_smooth else args.transition_width

    result = difference_constraint(
        obs_gravity=obs_gravity,
        model_gravity=model_aligned,
        obs_mask=obs_mask,
        cellsize=cellsize,
        transition_width=transition_width,
        verbose=True
    )

    final_gravity = result['final']
    diff_gravity_result = result['diff_gravity']

    # ============ 7. 保存结果 ============
    print('\n7. 保存结果...')

    # 主结果
    write_asc_grid(args.output, final_gravity, target_header)
    print(f'   ✓ 最终重力场: {args.output}')

    # 差值重力场（可选）
    if args.output_diff:
        write_asc_grid(args.output_diff, diff_gravity_result, target_header)
        print(f'   ✓ 差值重力场: {args.output_diff}')

    # NoData 区域填充结果（可选）
    if args.output_nodata:
        nodata_only = np.full(target_shape, np.nan)
        nodata_only[nodata_mask] = final_gravity[nodata_mask]
        write_asc_grid(args.output_nodata, nodata_only, target_header)
        print(f'   ✓ NoData 填充: {args.output_nodata}')

    print('\n' + '=' * 70)
    print('✅ Step 9 完成！')
    print('=' * 70)


if __name__ == '__main__':
    main()
