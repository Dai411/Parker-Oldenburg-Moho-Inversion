# parker_oldenburg_inversion_final.py
"""
Parker-Oldenburg 迭代反演 Moho 深度 (最终完整版)
集成: NaN 插值填补, FFT 扩边, 余弦平滑衰减, Tikhonov 正则化, 弹性挠曲, Under-relaxation

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
DRHO = 0.6          # 密度差 (g/cm3)
Z0 = 20.0            # 平均界面深度 (km)
TE = 2.0             # 弹性厚度 (km)

# 迭代参数
MAX_ITER = 20          # 最大迭代次数
USE_EARLY_STOP = True  # 启用早停机制
TOLERANCE = 0.001      # 收敛容差
N_MAX = 5              # 非线性项最大阶数
LEARNING_RATE = 0.3    # 松弛因子 (Under-relaxation), h_new = h_old + rate * delta_h

# 滤波与延拓参数
LOW_PASS_WL = 150     # km (长波完全保留)
HIGH_PASS_WL = 90     # km (短波完全截止)
MAX_DW_CONT = 2.0     # 向下延拓最大放大倍数
PADDING_RATIO = 0.4   # 扩边比例 (从 0.25 增加到 0.5)

# 修正量限制
MAX_DELTA_H = 5.0      # 单次最大修正量 (km)

# 正则化参数 (Tikhonov)
ALPHA_TIKHONOV = 5000    # Tikhonov 正则化参数
TIKHONOV_ORDER = 4      # Tikhonov 正则化阶数

# 弹性参数
USE_FLEXURE = True     # 启用弹性挠曲
MAX_INV_FLEX = 10    # 弹性反演最大放大倍数
E = 5e10               # 杨氏模量 (Pa)
NU = 0.25              # 泊松比
G_GRAV = 9.81          # 重力加速度 (m/s2)

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
stitch_dir = os.path.join(data_dir, 'StitchGrids')
moho_dir = os.path.join(data_dir, 'MohoInversion')

gravity_file = os.path.join(stitch_dir, 'BouguerFinalWithModel.asc')
initial_moho_file = os.path.join(moho_dir, 'InitialMoho_Uniform.asc')
output_moho_file = os.path.join(moho_dir, 'FinalMoho_Flex1.asc')
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

def fill_nans_fast(grid):
    """快速填充 NaN (用有效区域均值)"""
    grid_filled = grid.copy()
    valid_mean = np.nanmean(grid)
    if np.isnan(valid_mean):
        return grid_filled
    grid_filled[np.isnan(grid_filled)] = valid_mean
    return grid_filled

    # 一维 Tukey 窗
def tukey_1d(M, padding_ratio=PADDING_RATIO):
    """一维 Tukey 窗，padding_ratio: 边缘衰减比例"""
    if M <= 1:
        return np.ones(M)
    n = np.arange(M)
    window = np.ones(M)
    n1 = int(np.floor(padding_ratio * (M - 1) / 2))
    if n1 > 0:
        window[:n1] = 0.5 * (1 - np.cos(np.pi * n[:n1] / n1))
        window[-n1:] = 0.5 * (1 - np.cos(np.pi * (M - 1 - n[-n1:]) / n1))
    return window

def build_taper(shape, padding_ratio=PADDING_RATIO):
    """
    构建二维 Tukey (余弦锥形) 窗
    shape: (ny, nx)
    alpha: 锥形比例 (0.5 表示 50% 边缘衰减)
    """
    ny, nx = shape
    taper_y = tukey_1d(ny, padding_ratio)
    taper_x = tukey_1d(nx, padding_ratio)
    return np.outer(taper_y, taper_x)

# ============================================================
# 核心计算函数
# ============================================================

def compute_wavenumbers(nx, ny, dx, dy):
    """计算波数网格 (rad/m)"""
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K

def lowpass_filter(k, low_wl, high_wl):
    """
    余弦低通滤波器
    k: 波数 (rad/m)
    low_wl: 长波截止波长 (m) - 完全保留
    high_wl: 短波截止波长 (m) - 完全截止
    """
    k_low = 2 * np.pi / low_wl
    k_high = 2 * np.pi / high_wl
    
    filter_k = np.ones_like(k)
    
    # 短波截止
    mask_high = k > k_high
    filter_k[mask_high] = 0.0
    
    # 过渡带
    mask_trans = (k > k_low) & (k <= k_high)
    filter_k[mask_trans] = 0.5 * (1 + np.cos(np.pi * (k[mask_trans] - k_low) / (k_high - k_low)))
    
    return filter_k

def tikhonov_filter(k, alpha=ALPHA_TIKHONOV, order=TIKHONOV_ORDER):
    """
    Tikhonov 正则化滤波器
    k: 波数 (rad/m)
    alpha: 正则化强度
    order: 阶数 (1=一阶, 2=二阶，二阶更平滑)
    """
    k_norm = k / np.max(k)  # 归一化到 [0,1]
    return 1.0 / (1.0 + alpha * k_norm**order)

def flexural_response_factor(k, te, drho_kg, g=9.81, e=1e11, nu=0.25):
    """
    挠曲响应因子
    k: 波数 (rad/m)
    te: 弹性厚度 (m)
    drho_kg: 密度差 (kg/m3)
    返回: 挠曲因子 (无单位)
    """
    # 挠曲刚度 D
    D = (e * te**3) / (12 * (1 - nu**2))
    
    # 挠曲响应因子
    flex_factor = 1.0 / (1 + (D * k**4) / (drho_kg * g))
    
    return flex_factor

def parker_forward(h_rel, drho, z0, dx, dy, k, upward_cont, flex_factor, n_max=4):
    """Parker-Oldenburg 正演 (含弹性挠曲)"""
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    
    sum_term = np.zeros_like(k, dtype=complex)
    
    for n in range(1, n_max + 1):
        h_pow = h_rel ** n
        F_h_pow = fft2(h_pow)
        term = (k ** (n-1)) / math.factorial(n) * F_h_pow
        sum_term += term
    
    # 正演重力 (SI: m/s2)
    F_gravity_si = -2 * np.pi * G_NEW * drho_kg * upward_cont * flex_factor * sum_term
    gravity_si = np.real(ifft2(F_gravity_si))
    
    # 转换为 mGal
    gravity_mgal = gravity_si * 1e5
    
    return gravity_mgal

def parker_inverse_step(gravity_residual, h_current, drho, z0, dx, dy, k, downward_cont, 
                        flex_factor, filter_combined, tikhonov, n_max=4):
    """Parker-Oldenburg 反演单步 (含弹性挠曲 + Tikhonov 正则化)"""
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    
    # 转换残差到 SI 单位 (m/s2)
    gravity_residual_si = gravity_residual / 1e5
    
    # 傅里叶变换
    F_residual = fft2(gravity_residual_si)
    
    # 弹性反演因子 (限制最大放大倍数)
    inv_flex = 1.0 / flex_factor
    inv_flex = np.clip(inv_flex, 1.0, MAX_INV_FLEX)
    
    # 从残差反演 Moho 修正量
    F_delta_h = -F_residual / (2 * np.pi * G_NEW * drho_kg) * downward_cont * inv_flex
    
    # 减去非线性项的高阶贡献
    for n in range(2, n_max + 1):
        h_pow_prev = h_current ** (n-1)
        F_h_pow_prev = fft2(h_pow_prev)
        term_correction = (k ** (n-1)) / math.factorial(n-1) * F_h_pow_prev
        F_delta_h -= term_correction
    
    # 应用组合滤波器 (低通 + Tikhonov)
    F_delta_h = F_delta_h * filter_combined * tikhonov
    
    # 逆变换
    delta_h = np.real(ifft2(F_delta_h))
    
    return delta_h

# ============================================================
# 主程序
# ============================================================

def main():
    total_start = time.time()
    
    # 创建日志文件
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("=" * 80 + "\n")
        log.write("Parker-Oldenburg Moho Inversion (Final Complete Version)\n")
        log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("=" * 80 + "\n\n")
        log.write("Parameters:\n")
        log.write(f"  DRHO = {DRHO} g/cm3\n")
        log.write(f"  Z0 = {Z0} km\n")
        log.write(f"  TE = {TE} km\n")
        log.write(f"  USE_FLEXURE = {USE_FLEXURE}\n")
        log.write(f"  MAX_INV_FLEX = {MAX_INV_FLEX}\n")
        log.write(f"  LEARNING_RATE = {LEARNING_RATE}\n")
        log.write(f"  LOW_PASS_WL = {LOW_PASS_WL} km\n")
        log.write(f"  HIGH_PASS_WL = {HIGH_PASS_WL} km\n")
        log.write(f"  MAX_DW_CONT = {MAX_DW_CONT}\n")
        log.write(f"  PADDING_RATIO = {PADDING_RATIO}\n")
        log.write(f"  ALPHA_TIKHONOV = {ALPHA_TIKHONOV}\n")
        log.write(f"  MAX_DELTA_H = {MAX_DELTA_H} km\n")
        log.write("\n" + "=" * 80 + "\n\n")
    
    print("=" * 70)
    print("Parker-Oldenburg Iterative Moho Inversion (Final Complete)")
    print("=" * 70)
    print(f"  Density contrast: {DRHO} g/cm3")
    print(f"  Mean depth: {Z0} km")
    print(f"  Elastic thickness: {TE} km (enabled: {USE_FLEXURE})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Filter wavelengths: {LOW_PASS_WL} ~ {HIGH_PASS_WL} km")
    print(f"  Tikhonov alpha: {ALPHA_TIKHONOV}")
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
    
    print(f"   Original size: {nx} x {ny}")
    print(f"   Valid points: {np.sum(valid_mask)}")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 2. 数据插值与去均值
    print("\n2. Interpolating NaNs & Removing mean...")
    step_start = time.time()
    
    grav_filled = fill_nans_fast(gravity_obs)
    moho_filled = fill_nans_fast(moho_initial * 1000)  # km -> m
    
    gravity_mean = np.nanmean(gravity_obs[valid_mask])
    grav_zm = grav_filled - gravity_mean
    h_rel = moho_filled - z0_m
    
    print(f"   Gravity mean: {gravity_mean:.2f} mGal")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 3. 扩边 (Padding) 与衰减窗 (Tapering)
    print("\n3. Padding & Tapering for FFT...")
    step_start = time.time()
    
    # 计算扩边后的尺寸 (使用 FFT 优化尺寸)
    target_nx = next_fast_len(nx + int(nx * PADDING_RATIO))
    target_ny = next_fast_len(ny + int(ny * PADDING_RATIO))
    
    pad_x_left = (target_nx - nx) // 2
    pad_x_right = target_nx - nx - pad_x_left
    pad_y_top = (target_ny - ny) // 2
    pad_y_bottom = target_ny - ny - pad_y_top
    
    # 使用边缘延伸方式填充
    grav_pad = np.pad(grav_zm, ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right)), mode='edge')
    h_rel_pad = np.pad(h_rel, ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right)), mode='edge')
    
    # 应用余弦衰减窗
    taper = build_taper(grav_pad.shape)
    grav_pad_tapered = grav_pad * taper
    h_rel_pad_tapered = h_rel_pad * taper
    
    print(f"   Original: {nx} x {ny}")
    print(f"   Padded: {target_nx} x {target_ny}")
    print(f"   Pad margins: top={pad_y_top}, bottom={pad_y_bottom}, left={pad_x_left}, right={pad_x_right}")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 4. 计算波数与滤波器
    print("\n4. Computing wavenumbers and filters...")
    step_start = time.time()
    
    KX, KY, K = compute_wavenumbers(target_nx, target_ny, dx, dy)
    
    # 低通滤波器
    low_wl_m = LOW_PASS_WL * 1000
    high_wl_m = HIGH_PASS_WL * 1000
    filter_lowpass = lowpass_filter(K, low_wl_m, high_wl_m)
    
    # Tikhonov 正则化
    tikhonov = tikhonov_filter(K, ALPHA_TIKHONOV)
    
    # 向上延拓因子
    upward_cont = np.exp(-K * z0_m)
    
    # 向下延拓因子 (限制放大倍数)
    downward_cont = np.exp(K * z0_m)
    downward_cont = np.clip(downward_cont, 0, MAX_DW_CONT)
    
    # 挠曲响应因子
    drho_kg = DRHO * 1000
    if USE_FLEXURE:
        flex_factor = flexural_response_factor(K, TE * 1000, drho_kg, G_GRAV, E, NU)
        print(f"   Flexure enabled: TE={TE} km")
        print(f"   Flex factor range: [{np.min(flex_factor):.2e}, {np.max(flex_factor):.2e}]")
    else:
        flex_factor = np.ones_like(K)
        print(f"   Flexure disabled")
    
    # 组合滤波器
    filter_combined = filter_lowpass
    
    print(f"   K range: [{np.min(K):.2e}, {np.max(K):.2e}] rad/m")
    print(f"   Filter: K_low={2*np.pi/low_wl_m:.2e}, K_high={2*np.pi/high_wl_m:.2e}")
    print(f"   Downward cont max: {np.max(downward_cont):.2e}")
    print(f"   Tikhonov range: [{np.min(tikhonov):.2e}, {np.max(tikhonov):.2e}]")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 5. 迭代反演
    print("\n5. Starting iterative inversion...")
    print("-" * 70)
    
    rms_history = []
    h_current_pad = h_rel_pad_tapered.copy()
    iter_start = time.time()
    
    for iteration in range(1, MAX_ITER + 1):
        iter_step_start = time.time()
        
        # 正演计算
        grav_calc_pad = parker_forward(h_current_pad, DRHO, Z0, dx, dy, K, upward_cont, flex_factor, N_MAX)
        
        # 计算残差
        gravity_residual_pad = grav_pad_tapered - grav_calc_pad
        
        # 仅在原始数据区域评估 RMS
        resid_center = gravity_residual_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
        rms_res = np.sqrt(np.mean(resid_center[valid_mask]**2))
        rms_history.append(rms_res)
        
        # 反演修正量
        delta_h_pad = parker_inverse_step(gravity_residual_pad, h_current_pad, DRHO, Z0, dx, dy, 
                                        K, downward_cont, flex_factor, filter_combined, tikhonov, N_MAX)
        
        # 限制修正量并应用学习率 (Under-relaxation)
        delta_h_pad = np.clip(delta_h_pad, -MAX_DELTA_H * 1000, MAX_DELTA_H * 1000)
        delta_h_pad = LEARNING_RATE * delta_h_pad
        
        # 计算新模型
        h_new_pad = h_current_pad + delta_h_pad
        h_new_pad = np.clip(h_new_pad, -25000, 25000)
        
        # 备份当前模型（用于早停恢复）
        h_prev_pad = h_current_pad.copy()
        
        # 计算变化率（使用新模型）
        dh_center = delta_h_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
        h_center = h_new_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]  # 注意：用新模型
        dh_rms = np.sqrt(np.mean(dh_center[valid_mask]**2))
        change_pct = 100 * dh_rms / (np.mean(np.abs(h_center[valid_mask])) + 1)
        
        # 收敛检查
        if iteration > 1:
            rms_change = abs(rms_history[-1] - rms_history[-2]) / rms_history[-2]
        else:
            rms_change = 1.0
        
        iter_time = time.time() - iter_step_start
        print(f"  Iter {iteration:2d}: RMS_res={rms_res:.2f} mGal, "
            f"Dh_rms={dh_rms/1000:.2f} km, "
            f"Change={change_pct:.1f}%, "
            f"Time={iter_time:.1f}s")
        
        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"Iter {iteration:2d}: RMS_res={rms_res:.4f} mGal, "
                    f"Dh_rms={dh_rms/1000:.4f} km, "
                    f"Change={change_pct:.2f}%, Time={iter_time:.2f}s\n")
        
        # ========== 早停检查（在更新之前检查，避免更新到更差模型） ==========
        if USE_EARLY_STOP and len(rms_history) > 2 and rms_history[-1] > rms_history[-2]:
            print(f"\n  [Early stop] RMS increased from {rms_history[-2]:.2f} to {rms_history[-1]:.2f} mGal")
            print(f"  Stopping at iteration {iteration-1}, restoring previous model")
            h_current_pad = h_prev_pad  # 恢复到上一轮模型
            rms_history.pop()           # 移除当前的 RMS
            break
        
        # 正式更新模型
        h_current_pad = h_new_pad
        
        # 收敛判断（RMS 变化很小）
        if rms_change < TOLERANCE and iteration > 1:
            print(f"\n  [OK] Converged at iteration {iteration} (RMS change {rms_change:.2e} < {TOLERANCE})")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nConverged at iteration {iteration}\n")
            break
    
    total_iter_time = time.time() - iter_start
    print("-" * 70)
    print(f"  Total iteration time: {total_iter_time:.1f}s")
    
    # 6. 后期处理与保存
    print("\n6. Restoring original domain & saving...")
    step_start = time.time()
    
    # 从扩边网格中切出原始区域
    h_final_center = h_current_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
    
    # 恢复为绝对深度并应用 NaN 掩码
    moho_final_km = (z0_m + h_final_center) / 1000
    moho_final_km[~valid_mask] = np.nan
    
    # 统计
    print(f"   Min: {np.nanmin(moho_final_km):.1f} km")
    print(f"   Max: {np.nanmax(moho_final_km):.1f} km")
    print(f"   Mean: {np.nanmean(moho_final_km):.1f} km")
    print(f"   Std: {np.nanstd(moho_final_km):.1f} km")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 保存
    write_asc_grid(output_moho_file, moho_final_km, moho_header)
    print(f"   [OK] Saved: {output_moho_file}")
    
    # 最终统计
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("[OK] Inversion completed!")
    print(f"    Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"    Final RMS residual: {rms_history[-1]:.4f} mGal")
    print(f"    Output file: {output_moho_file}")
    print(f"    Log file: {log_file}")
    print("=" * 70)
    
    # 写入日志总结
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total time: {total_time:.1f}s\n")
        log.write(f"Final RMS residual: {rms_history[-1]:.6f} mGal\n")
        log.write("=" * 80 + "\n")

if __name__ == "__main__":
    main()