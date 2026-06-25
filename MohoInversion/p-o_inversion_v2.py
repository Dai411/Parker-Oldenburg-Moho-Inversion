# parker_oldenburg_inversion_final_fixed.py
"""
Parker-Oldenburg 迭代反演 Moho 深度 (终极修复版)
加入: NaN 插值填补, 最优 FFT 扩边 (Padding), 以及余弦平滑衰减 (Tapering)

参考文献:
    - Parker (1972): The rapid calculation of potential anomalies
    - Oldenburg (1974): The inversion and interpretation of gravity anomalies
"""

import numpy as np
import os
import time
import math
from datetime import datetime
from scipy.fft import fft2, ifft2, fftfreq, next_fast_len
from scipy.interpolate import NearestNDInterpolator

# ============================================================
# 参数配置 (可调)
# ============================================================

# 物理参数
DRHO = 0.40          # 密度差 (g/cm3)
Z0 = 17.5            # 平均界面深度 (km)
TE = 5.0             # 弹性厚度 (km)

# 迭代参数
MAX_ITER = 20        # 最大迭代次数
TOLERANCE = 0.001    # 收敛容差 
N_MAX = 4            # 非线性项最大阶数
LEARNING_RATE = 0.5  # 学习率

# 滤波与延拓参数
LOW_PASS_WL = 150     # km (长波完全保留 - 略微调大以增加稳定性)
HIGH_PASS_WL = 50     # km (短波完全截止 - 提高截止波长，压制高频色块)
MAX_DW_CONT = 50.0    # 向下延拓最大放大倍数 (限制指数爆炸，1e3太大，100更安全)
MAX_DELTA_H = 2.0     # 单次最大修正量 (km)

# 弹性参数 (启用)
USE_FLEXURE = False   # 启用弹性挠曲
E = 1e11              # 杨氏模量 (Pa)
NU = 0.25             # 泊松比
G_GRAV = 9.81         # 重力加速度 (m/s2)

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
# 请确保目录存在，这里保持原样
stitch_dir = os.path.join(data_dir, 'StitchGrids')
moho_dir = os.path.join(data_dir, 'MohoInversion')

gravity_file = os.path.join(stitch_dir, 'BouguerFinalWithModel.asc')
initial_moho_file = os.path.join(moho_dir, 'InitialMoho_Uniform.asc')
output_moho_file = os.path.join(moho_dir, 'FinalMoho.asc')
log_file = os.path.join(moho_dir, 'inversion_log.txt')

# ============================================================
# 读取/写入函数
# ============================================================

def read_asc_header(filename):
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            if '.' in line[1] or 'e' in line[1].lower():
                header[key] = float(line[1])
            else:
                header[key] = int(line[1])
    return header

def read_asc_grid(filename, header=None):
    if header is None:
        header = read_asc_header(filename)
    data = np.loadtxt(filename, skiprows=6)
    nodata = header['nodata_value']
    data[data == nodata] = np.nan
    return data

def write_asc_grid(filename, data, header):
    output_data = data.copy()
    output_data[np.isnan(output_data)] = header['nodata_value']
    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value {header['nodata_value']}\n")
        for row in range(output_data.shape[0]):
            f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')

# ============================================================
# 数据预处理 (插值, 扩边, 衰减)
# ============================================================

def fill_nans(grid):
    """使用最邻近插值填补 NaN，避免边界出现断崖"""
    mask = ~np.isnan(grid)
    if mask.all(): return grid.copy()
    y, x = np.where(mask)
    points = np.column_stack((x, y))
    values = grid[mask]
    interp = NearestNDInterpolator(points, values)
    y_all, x_all = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    return interp(x_all, y_all)

def build_taper_asym(shape, pad_widths):
    """生成非对称余弦衰减窗口 (Tukey 窗的变体)"""
    ny, nx = shape
    taper_y = np.ones(ny)
    taper_x = np.ones(nx)
    top, bottom = pad_widths[0]
    left, right = pad_widths[1]

    # Y 方向边缘衰减
    for i in range(top):
        taper_y[i] = 0.5 * (1 - np.cos(np.pi * i / max(top, 1)))
    for i in range(bottom):
        taper_y[ny - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / max(bottom, 1)))
        
    # X 方向边缘衰减
    for i in range(left):
        taper_x[i] = 0.5 * (1 - np.cos(np.pi * i / max(left, 1)))
    for i in range(right):
        taper_x[nx - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / max(right, 1)))

    T_X, T_Y = np.meshgrid(taper_x, taper_y)
    return T_X * T_Y

# ============================================================
# 核心计算函数
# ============================================================

def compute_wavenumbers(nx, ny, dx, dy):
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K

def lowpass_filter(k, low_wl, high_wl):
    k_low = 2 * np.pi / low_wl
    k_high = 2 * np.pi / high_wl
    filter_k = np.ones_like(k)
    mask_high = k > k_high
    filter_k[mask_high] = 0.0
    mask_trans = (k > k_low) & (k <= k_high)
    filter_k[mask_trans] = 0.5 * (1 + np.cos(np.pi * (k[mask_trans] - k_low) / (k_high - k_low)))
    return filter_k

def flexural_response_factor(k, te, drho_kg, g=9.81, e=1e11, nu=0.25):
    D = (e * te**3) / (12 * (1 - nu**2))
    return 1.0 / (1 + (D * k**4) / (drho_kg * g))

def parker_forward(h_rel, drho, z0, dx, dy, k, upward_cont, flex_factor, n_max=4):
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    sum_term = np.zeros_like(k, dtype=complex)
    
    for n in range(1, n_max + 1):
        h_pow = h_rel ** n
        F_h_pow = fft2(h_pow)
        term = (k ** (n-1)) / math.factorial(n) * F_h_pow
        sum_term += term
    
    F_gravity_si = -2 * np.pi * G_NEW * drho_kg * upward_cont * flex_factor * sum_term
    return np.real(ifft2(F_gravity_si)) * 1e5  # 转 mGal

def parker_inverse_step(gravity_residual, h_current, drho, z0, dx, dy, k, downward_cont, flex_factor, filter_combined, n_max=4):
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    gravity_residual_si = gravity_residual / 1e5
    
    F_residual = fft2(gravity_residual_si)
    inv_flex = 1.0 / flex_factor
    MAX_INV_FLEX = 5.0  # Maximum allowed inverse flexure to prevent instability
    inv_flex = np.clip(inv_flex, 1.0, MAX_INV_FLEX)
    F_delta_h = -F_residual / (2 * np.pi * G_NEW * drho_kg) * downward_cont * inv_flex
    
    for n in range(2, n_max + 1):
        h_pow_prev = h_current ** (n-1)
        F_h_pow_prev = fft2(h_pow_prev)
        term_correction = (k ** (n-1)) / math.factorial(n-1) * F_h_pow_prev
        F_delta_h -= term_correction
    
    F_delta_h = F_delta_h * filter_combined
    return np.real(ifft2(F_delta_h))

# ============================================================
# 主程序
# ============================================================

def main():
    total_start = time.time()
    
    print("=" * 70)
    print("Parker-Oldenburg Iterative Moho Inversion (FFT Optimized)")
    print("=" * 70)
    
    # 1. 读取数据
    print("\n1. Reading data...")
    step_start = time.time()
    
    gravity_header = read_asc_header(gravity_file)
    moho_header = read_asc_header(initial_moho_file)
    gravity_obs = read_asc_grid(gravity_file, gravity_header)
    moho_initial = read_asc_grid(initial_moho_file, moho_header)
    
    nx = gravity_header['ncols']
    ny = gravity_header['nrows']
    dx = gravity_header['cellsize']
    dy = gravity_header['cellsize']
    z0_m = Z0 * 1000
    valid_mask = ~np.isnan(gravity_obs) & ~np.isnan(moho_initial)
    
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 2. 数据插值与去均值
    print("\n2. Interpolating NaNs & Removing mean...")
    step_start = time.time()
    
    grav_filled = fill_nans(gravity_obs)
    moho_filled = fill_nans(moho_initial * 1000)  # km -> m
    
    gravity_mean = np.nanmean(gravity_obs[valid_mask])
    grav_zm = grav_filled - gravity_mean
    h_rel = moho_filled - z0_m
    
    # 3. 计算最优扩边 (Padding) 与衰减 (Tapering)
    print("\n3. Padding & Tapering for FFT...")
    
    # 将网格长宽扩充至少 25%，并寻找最近的 FFT 高效尺寸 (2的幂次或平滑数)
    target_nx = next_fast_len(nx + int(nx * 0.50))
    target_ny = next_fast_len(ny + int(ny * 0.50))
    
    pad_x_left = (target_nx - nx) // 2
    pad_x_right = target_nx - nx - pad_x_left
    pad_y_top = (target_ny - ny) // 2
    pad_y_bottom = target_ny - ny - pad_y_top
    pad_widths = ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right))
    
    # 使用边缘延伸方式进行填充
    grav_pad = np.pad(grav_zm, pad_widths, mode='edge')
    h_rel_pad = np.pad(h_rel, pad_widths, mode='edge')
    
    # 生成衰减窗并应用
    taper = build_taper_asym(grav_pad.shape, pad_widths)
    grav_pad_tapered = grav_pad * taper
    h_rel_pad_tapered = h_rel_pad * taper
    
    print(f"   Original size: {nx} x {ny}")
    print(f"   Padded size: {target_nx} x {target_ny} (FFT optimized)")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 4. 计算波数与滤波器 (在扩边后的网格上运算)
    print("\n4. Computing wavenumbers and filters...")
    step_start = time.time()
    
    KX, KY, K = compute_wavenumbers(target_nx, target_ny, dx, dy)
    
    filter_lowpass = lowpass_filter(K, LOW_PASS_WL * 1000, HIGH_PASS_WL * 1000)
    upward_cont = np.exp(-K * z0_m)
    
    # 限制向下延拓的无限放大，这是消除高频斑块的关键
    downward_cont = np.clip(np.exp(K * z0_m), 0, MAX_DW_CONT) 
    
    drho_kg = DRHO * 1000
    if USE_FLEXURE:
        flex_factor = flexural_response_factor(K, TE * 1000, drho_kg, G_GRAV, E, NU)
    else:
        flex_factor = np.ones_like(K)
        
    filter_combined = filter_lowpass
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 5. 迭代反演 (在 Padded 网格上整体循环)
    print("\n5. Starting iterative inversion...")
    print("-" * 70)
    
    rms_history = []
    h_current_pad = h_rel_pad_tapered.copy()
    
    for iteration in range(1, MAX_ITER + 1):
        iter_start = time.time()
        
        # 正演 (返回扩边后的重力异常)
        grav_calc_pad = parker_forward(h_current_pad, DRHO, Z0, dx, dy, K, upward_cont, flex_factor, N_MAX)
        
        # 计算残差
        gravity_residual_pad = grav_pad_tapered - grav_calc_pad
        
        # !!! 仅在原始真实数据区域评估 RMS，以判断真实收敛性 !!!
        # 切片提取中心区域
        resid_center = gravity_residual_pad[pad_y_top : pad_y_top + ny, pad_x_left : pad_x_left + nx]
        rms_res = np.sqrt(np.mean(resid_center[valid_mask]**2))
        rms_history.append(rms_res)
        
        # 反演修正量
        delta_h_pad = parker_inverse_step(gravity_residual_pad, h_current_pad, DRHO, Z0, dx, dy, K, downward_cont, flex_factor, filter_combined, N_MAX)
        
        # 限制单次修正量并应用学习率
        delta_h_pad = np.clip(delta_h_pad, -MAX_DELTA_H * 1000, MAX_DELTA_H * 1000)
        delta_h_pad = LEARNING_RATE * delta_h_pad
        
        # 更新扩边后的界面模型
        h_new_pad = h_current_pad + delta_h_pad
        h_new_pad = np.clip(h_new_pad, -25000, 25000)
        
        # 计算变化率
        dh_center = delta_h_pad[pad_y_top : pad_y_top + ny, pad_x_left : pad_x_left + nx]
        h_center = h_current_pad[pad_y_top : pad_y_top + ny, pad_x_left : pad_x_left + nx]
        dh_rms = np.sqrt(np.mean(dh_center[valid_mask]**2))
        change_pct = 100 * dh_rms / (np.mean(np.abs(h_center[valid_mask])) + 1)
        
        rms_change = abs(rms_history[-1] - rms_history[-2]) / rms_history[-2] if iteration > 1 else 1.0
        
        print(f"  Iter {iteration:2d}: RMS_res={rms_res:.2f} mGal, Dh_rms={dh_rms/1000:.2f} km, Change={change_pct:.1f}%, Time={time.time()-iter_start:.1f}s")
        
        h_current_pad = h_new_pad
        
        if rms_change < TOLERANCE and iteration > 1:
            print(f"\n  [OK] Converged at iteration {iteration}")
            break
            
        if len(rms_history) > 2 and rms_history[-1] > rms_history[-2] * 1.1:
            print(f"\n  [Warning] RMS slightly increased, stopping early to prevent divergence.")
            break
    
    # 6. 后期处理与保存
    print("\n6. Restoring original domain & saving...")
    
    # 从扩边后的网格中切出中心原始数据部分
    h_final_center = h_current_pad[pad_y_top : pad_y_top + ny, pad_x_left : pad_x_left + nx]
    
    # 恢复为绝对深度并应用 NaN 掩码
    moho_final_km = (z0_m + h_final_center) / 1000
    moho_final_km[~valid_mask] = np.nan
    
    write_asc_grid(output_moho_file, moho_final_km, moho_header)
    print(f"   [OK] Saved: {output_moho_file}")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"[OK] Inversion completed in {total_time:.1f}s!")
    print(f"     Final RMS: {rms_history[-1]:.4f} mGal")
    print("=" * 70)

if __name__ == "__main__":
    main()