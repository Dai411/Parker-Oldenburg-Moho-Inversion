# p-o_inversion_v1.py
"""
Parker-Oldenburg 迭代反演 Moho 深度 (含弹性挠曲)
基于测试验证的正演公式 (exp(-K*z0) 版本)

参考文献:
    - Parker (1972): The rapid calculation of potential anomalies
    - Oldenburg (1974): The inversion and interpretation of gravity anomalies

输入:
    - BouguerFinalWithModel.asc (最终布格异常)
    - InitialMoho_Uniform.asc (初始 Moho 模型)

输出:
    - FinalMoho.asc (最终 Moho 深度)
    - inversion_log.txt (反演日志)
"""

import numpy as np
import os
import time
import math
from datetime import datetime
from scipy.fft import fft2, ifft2, fftfreq

# ============================================================
# 参数配置 (可调)
# ============================================================

# 物理参数
DRHO = 0.40          # 密度差 (g/cm3) - 传统值起手(0.4);蛇纹岩化0.25
Z0 = 10.0            # 平均界面深度 (km)
TE = 5.0            # 弹性厚度 (km)

# 迭代参数
MAX_ITER = 20        # 最大迭代次数
TOLERANCE = 0.001    # 收敛容差 (RMS 相对变化 < 0.1%)
N_MAX = 4            # 非线性项最大阶数 (n=2 到 N_MAX)
LEARNING_RATE = 0.5  # 学习率 (0.3-0.8，避免震荡)

# 滤波参数 (低通滤波器，压制高频噪声)
#   波长 > LOW_PASS_WL (km) : 完全保留
#   波长 < HIGH_PASS_WL (km): 完全截止
LOW_PASS_WL = 80      # km (长波完全保留)
HIGH_PASS_WL = 30     # km (短波完全截止)

# 修正量限制 (km)
MAX_DELTA_H = 2.0     # 单次最大修正量 (km)

# 弹性参数 (启用)
USE_FLEXURE = False    # 启用弹性挠曲

# 弹性常数
E = 1e11              # 杨氏模量 (Pa)
NU = 0.25             # 泊松比
G_GRAV = 9.81         # 重力加速度 (m/s2)

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
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
                value = float(line[1])
            else:
                value = int(line[1])
            header[key] = value
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
# 核心函数
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

def flexural_response_factor(k, te, drho_kg, g=9.81, e=1e11, nu=0.25):
    """
    挠曲响应因子 (文档1 公式 5-7)
    k: 波数 (rad/m)
    te: 弹性厚度 (m)
    drho_kg: 密度差 (kg/m3)
    返回: 挠曲因子 (无单位)
    
    公式: 1 / (1 + D*k^4/(Δρ*g))
    在反演中，重力异常需要除以这个因子来恢复界面起伏
    """
    # 挠曲刚度 D
    D = (e * te**3) / (12 * (1 - nu**2))
    
    # 挠曲响应因子
    flex_factor = 1.0 / (1 + (D * k**4) / (drho_kg * g))
    
    return flex_factor

def parker_forward(h_rel, drho, z0, dx, dy, k, upward_cont, flex_factor, n_max=4):
    """
    Parker-Oldenburg 正演 (含弹性挠曲)
    h_rel: 界面起伏 (m)，相对于平均深度 z0
    drho: 密度差 (g/cm3)
    z0: 平均深度 (km)
    k: 波数
    upward_cont: 向上延拓因子
    flex_factor: 挠曲响应因子
    返回: gravity (mGal)
    """
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    
    sum_term = np.zeros_like(k, dtype=complex)
    
    for n in range(1, n_max + 1):
        h_pow = h_rel ** n
        F_h_pow = fft2(h_pow)
        term = (k ** (n-1)) / math.factorial(n) * F_h_pow
        sum_term += term
    
    # 正演重力 (SI: m/s2) - 加入挠曲响应因子
    F_gravity_si = -2 * np.pi * G_NEW * drho_kg * upward_cont * flex_factor * sum_term
    gravity_si = np.real(ifft2(F_gravity_si))
    
    # 转换为 mGal
    gravity_mgal = gravity_si * 1e5
    
    return gravity_mgal

def parker_inverse_step(gravity_residual, h_current, drho, z0, dx, dy, k, downward_cont, flex_factor, filter_combined, n_max=4):
    """
    Parker-Oldenburg 反演单步 (含弹性挠曲)
    gravity_residual: 残差重力 (mGal)
    h_current: 当前界面起伏 (m)
    返回: delta_h (m)
    """
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    z0_m = z0 * 1000
    
    # 转换残差到 SI 单位 (m/s2)
    gravity_residual_si = gravity_residual / 1e5
    
    # 傅里叶变换
    F_residual = fft2(gravity_residual_si)
    
    # 从残差反演 Moho 修正量
    # 公式: F[Dh] = -F[dg_res] e^{kz0} / (2πGΔρ) * (1/flex_factor)
    # 注意: 反演时需要除以挠曲因子来补偿弹性支撑
    inv_flex = 1.0 / flex_factor
    F_delta_h = -F_residual / (2 * np.pi * G_NEW * drho_kg) * downward_cont * inv_flex
    
    # 减去非线性项的高阶贡献
    for n in range(2, n_max + 1):
        h_pow_prev = h_current ** (n-1)
        F_h_pow_prev = fft2(h_pow_prev)
        term_correction = (k ** (n-1)) / math.factorial(n-1) * F_h_pow_prev
        F_delta_h -= term_correction
    
    # 应用组合滤波器
    F_delta_h = F_delta_h * filter_combined
    
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
        log.write("Parker-Oldenburg Moho Inversion (Final with Flexure)\n")
        log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("=" * 80 + "\n\n")
        log.write("Parameters:\n")
        log.write(f"  DRHO = {DRHO} g/cm3\n")
        log.write(f"  Z0 = {Z0} km\n")
        log.write(f"  TE = {TE} km\n")
        log.write(f"  USE_FLEXURE = {USE_FLEXURE}\n")
        log.write(f"  MAX_ITER = {MAX_ITER}\n")
        log.write(f"  TOLERANCE = {TOLERANCE}\n")
        log.write(f"  LEARNING_RATE = {LEARNING_RATE}\n")
        log.write(f"  N_MAX = {N_MAX}\n")
        log.write(f"  LOW_PASS_WL = {LOW_PASS_WL} km\n")
        log.write(f"  HIGH_PASS_WL = {HIGH_PASS_WL} km\n")
        log.write(f"  MAX_DELTA_H = {MAX_DELTA_H} km\n")
        log.write("\n" + "=" * 80 + "\n\n")
    
    print("=" * 70)
    print("Parker-Oldenburg Iterative Moho Inversion (with Flexure)")
    print("=" * 70)
    print(f"  Density contrast: {DRHO} g/cm3")
    print(f"  Mean depth: {Z0} km")
    print(f"  Elastic thickness: {TE} km (enabled: {USE_FLEXURE})")
    print(f"  Max iterations: {MAX_ITER}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Filter wavelengths: {LOW_PASS_WL} ~ {HIGH_PASS_WL} km")
    print(f"  Max delta_h per iteration: {MAX_DELTA_H} km")
    print("=" * 70)
    
    # 1. 读取数据
    print("\n1. Reading data...")
    step_start = time.time()
    
    gravity_header = read_asc_header(gravity_file)
    moho_header = read_asc_header(initial_moho_file)
    
    gravity_obs = read_asc_grid(gravity_file, gravity_header)
    moho_initial = read_asc_grid(initial_moho_file, moho_header)
    
    print(f"   Bouguer anomaly: {gravity_obs.shape}, valid: {np.sum(~np.isnan(gravity_obs))}")
    print(f"   Initial Moho: {moho_initial.shape}, valid: {np.sum(~np.isnan(moho_initial))}")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 2. 准备反演数据
    print("\n2. Preparing inversion data...")
    step_start = time.time()
    
    nx = gravity_header['ncols']
    ny = gravity_header['nrows']
    dx = gravity_header['cellsize']
    dy = gravity_header['cellsize']
    
    z0_m = Z0 * 1000  # km -> m
    G_NEW = 6.67430e-11
    
    # 有效区域掩码
    valid_mask = ~np.isnan(gravity_obs) & ~np.isnan(moho_initial)
    
    # 布格异常减均值
    gravity_mean = np.nanmean(gravity_obs[valid_mask])
    gravity_obs_zero_mean = gravity_obs - gravity_mean
    gravity_obs_zero_mean[~valid_mask] = 0
    
    # 初始 Moho 模型 (km -> m)
    moho_abs_m = moho_initial * 1000
    moho_abs_m[~valid_mask] = 0
    
    # 转换为起伏 (相对于 Z0)
    h_rel = moho_abs_m - z0_m
    h_rel[~valid_mask] = 0
    
    print(f"   Bouguer anomaly mean: {gravity_mean:.2f} mGal")
    print(f"   Valid points: {np.sum(valid_mask)}")
    print(f"   Initial Moho mean: {np.mean(moho_abs_m[valid_mask])/1000:.1f} km")
    print(f"   Initial relief range: {np.min(h_rel[valid_mask])/1000:.1f} - {np.max(h_rel[valid_mask])/1000:.1f} km")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 3. 计算波数和滤波器
    print("\n3. Computing wavenumbers and filters...")
    step_start = time.time()
    
    KX, KY, K = compute_wavenumbers(nx, ny, dx, dy)
    
    # 低通滤波器
    low_wl_m = LOW_PASS_WL * 1000
    high_wl_m = HIGH_PASS_WL * 1000
    filter_lowpass = lowpass_filter(K, low_wl_m, high_wl_m)
    
    # 向上延拓因子 exp(-K * z0) (正演用)
    upward_cont = np.exp(-K * z0_m)
    
    # 向下延拓因子 exp(K * z0) (反演用)
    downward_cont = np.exp(K * z0_m)
    downward_cont = np.clip(downward_cont, 0, 1e3)  # 限制避免爆炸
    
    # 挠曲响应因子
    drho_kg = DRHO * 1000
    if USE_FLEXURE:
        flex_factor = flexural_response_factor(K, TE * 1000, drho_kg, G_GRAV, E, NU)
        print(f"   Flexure enabled: TE={TE} km")
    else:
        flex_factor = np.ones_like(K)
        print(f"   Flexure disabled")
    
    # 组合滤波器 (低通 + 挠曲)
    filter_combined = filter_lowpass
    
    print(f"   K range: [{np.min(K):.2e}, {np.max(K):.2e}] rad/m")
    print(f"   Filter: K_low={2*np.pi/low_wl_m:.2e}, K_high={2*np.pi/high_wl_m:.2e}")
    print(f"   Upward cont range: [{np.min(upward_cont):.2e}, {np.max(upward_cont):.2e}]")
    print(f"   Downward cont max: {np.max(downward_cont):.2e}")
    print(f"   Flex factor range: [{np.min(flex_factor):.2e}, {np.max(flex_factor):.2e}]")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 4. 迭代反演
    print("\n4. Starting iterative inversion...")
    print("-" * 70)
    
    rms_history = []
    h_current = h_rel.copy()
    iter_start = time.time()
    
    for iteration in range(1, MAX_ITER + 1):
        iter_step_start = time.time()
        
        # 4.1 正演计算 (含弹性挠曲)
        gravity_calc = parker_forward(h_current, DRHO, Z0, dx, dy, K, upward_cont, flex_factor, N_MAX)
        
        # 4.2 计算残差
        gravity_residual = gravity_obs_zero_mean - gravity_calc
        gravity_residual[~valid_mask] = 0
        
        # 4.3 计算 RMS
        rms_res = np.sqrt(np.mean(gravity_residual[valid_mask]**2))
        rms_history.append(rms_res)
        
        # 4.4 反演修正量 (含弹性挠曲)
        delta_h = parker_inverse_step(gravity_residual, h_current, DRHO, Z0, dx, dy, K, downward_cont, flex_factor, filter_combined, N_MAX)
        
        # 4.5 限制修正量
        delta_h = np.clip(delta_h, -MAX_DELTA_H * 1000, MAX_DELTA_H * 1000)
        
        # 4.6 应用学习率
        delta_h = LEARNING_RATE * delta_h
        
        # 4.7 更新模型
        h_new = h_current + delta_h
        
        # 4.8 限制合理范围 (相对于 Z0，允许 ±25 km)
        h_new = np.clip(h_new, -25000, 25000)
        h_new[~valid_mask] = 0
        
        # 4.9 计算变化
        delta_h_rms = np.sqrt(np.mean((delta_h[valid_mask])**2))
        moho_change_pct = 100 * delta_h_rms / (np.mean(np.abs(h_current[valid_mask])) + 1)
        
        # 4.10 收敛检查
        if iteration > 1:
            rms_change = abs(rms_history[-1] - rms_history[-2]) / rms_history[-2]
        else:
            rms_change = 1.0
        
        # 4.11 输出进度
        iter_time = time.time() - iter_step_start
        progress_msg = (f"  Iter {iteration:2d}: RMS_res={rms_res:.2f} mGal, "
                       f"Dh_rms={delta_h_rms/1000:.2f} km, "
                       f"Change={moho_change_pct:.1f}%, "
                       f"Time={iter_time:.1f}s")
        print(progress_msg)
        
        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"Iter {iteration:2d}: RMS_res={rms_res:.4f} mGal, "
                     f"Dh_rms={delta_h_rms/1000:.4f} km, "
                     f"Change={moho_change_pct:.2f}%, Time={iter_time:.2f}s\n")
        
        # 更新模型
        h_current = h_new
        
        # 收敛检查
        if rms_change < TOLERANCE and iteration > 1:
            print(f"\n  [OK] Converged at iteration {iteration} (RMS change {rms_change:.2e} < {TOLERANCE})")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nConverged at iteration {iteration}\n")
            break
        
        # 早停：RMS 开始增大
        if len(rms_history) > 2 and rms_history[-1] > rms_history[-2] * 1.2:
            print(f"\n  [Warning] RMS increased, stopping at iteration {iteration-1}")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nStopped early at iteration {iteration-1}\n")
            h_current = h_prev
            break
        
        h_prev = h_current.copy()
    
    total_iter_time = time.time() - iter_start
    print("-" * 70)
    print(f"  Total iteration time: {total_iter_time:.1f}s")
    
    # 5. 最终结果处理
    print("\n5. Processing final result...")
    step_start = time.time()
    
    # 转换回绝对深度 (m -> km)
    moho_abs_m = z0_m + h_current
    moho_final_km = moho_abs_m / 1000
    moho_final_km[~valid_mask] = np.nan
    
    # 限制合理范围 (5-45 km)
    #moho_final_km = np.clip(moho_final_km, 5, 45)
    
    # 统计
    print(f"   Min: {np.nanmin(moho_final_km):.1f} km")
    print(f"   Max: {np.nanmax(moho_final_km):.1f} km")
    print(f"   Mean: {np.nanmean(moho_final_km):.1f} km")
    print(f"   Std: {np.nanstd(moho_final_km):.1f} km")
    print(f"   Time: {time.time() - step_start:.2f}s")
    
    # 6. 保存结果
    print("\n6. Saving result...")
    step_start = time.time()
    
    write_asc_grid(output_moho_file, moho_final_km, moho_header)
    print(f"   [OK] Saved: {output_moho_file}")
    
    # 7. 最终统计
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
