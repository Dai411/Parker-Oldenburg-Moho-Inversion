# parker_oldenburg_inversion_v4.py
"""
Parker-Oldenburg 迭代反演 Moho 深度 (v4: 集成 RF 地震约束)
相对于 v3 的改进:
    - 新增 RF (Receiver Function) 地震约束模块
    - 在每次迭代中计算 RF 修正场并合并到反演 (公式 12)
    - 新增 RF 相关参数: RF_SIGMA, RF_LAMBDA_C, RF_GAMMA, RF_LAMBDA
    - 新增自适应空间权重，防止陆地点约束扩散到海洋
    - 新增频域低通滤波 (λc=250-400km)
    - 新增地震约束点的双线性插值残差计算

参考文献:
    - Parker (1972): The rapid calculation of potential anomalies
    - Oldenburg (1974): The inversion and interpretation of gravity anomalies
    - Receiver Function 约束方法 (Marco Ligi)
"""

import numpy as np
import os
import time
import math
import sys
from datetime import datetime
from scipy.fft import fft2, ifft2, fftfreq, next_fast_len
from scipy.interpolate import NearestNDInterpolator

# 添加项目根目录到路径，以便导入 RF 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rf_constraint_mercator import RFConstraint
    RF_AVAILABLE = True
    print("✓ RF constraint module loaded successfully")
except ImportError as e:
    RF_AVAILABLE = False
    print(f"⚠ RF constraint module not available: {e}")
    print("  Running without RF constraints")

# ============================================================
# 参数配置 (可调)
# ============================================================

# 物理参数
DRHO = 0.60          # 密度差 (g/cm3)
Z0 = 20.0            # 平均界面深度 (km)
TE = 2.5             # 弹性厚度 (km)

# 迭代参数
MAX_ITER = 30          # 最大迭代次数
USE_EARLY_STOP = True  # 启用早停机制
TOLERANCE = 0.001      # 收敛容差
N_MAX = 5              # 非线性项最大阶数
LEARNING_RATE = 0.5    # 松弛因子 (Under-relaxation)

# 滤波与延拓参数
LOW_PASS_WL = 150     # km (长波完全保留) best 150
HIGH_PASS_WL = 90     # km (短波完全截止) best 90
MAX_DW_CONT = 2.0     # 向下延拓最大放大倍数
PADDING_RATIO = 0.5   # 扩边比例

# 修正量限制
MAX_DELTA_H = 3.0      # 单次最大修正量 (km)

# 正则化参数 (Tikhonov)
ALPHA_TIKHONOV = 20000    # Tikhonov 正则化参数
TIKHONOV_ORDER = 4      # Tikhonov 正则化阶数

# 弹性参数
USE_FLEXURE = False     # 启用弹性挠曲
MAX_INV_FLEX = 10      # 弹性反演最大放大倍数
E = 5e10               # 杨氏模量 (Pa)
NU = 0.25              # 泊松比
G_GRAV = 9.81          # 重力加速度 (m/s2)

# ========== RF 约束参数 (新增) ==========
USE_RF_CONSTRAINT = True   # 是否启用 RF 地震约束
RF_SIGMA = 30000           # 高斯扩散半径 (m) (30 km)
RF_LAMBDA_C = 300000       # 低通滤波截止波长 (m) (300 km)
RF_GAMMA = 0.5             # 空间权重指数 (控制约束不扩散到海洋) (Default: 2.0)
RF_LAMBDA = 2.0            # RF 约束相对权重 (公式 12 中的 λ_RF) (Default: 0.3)

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
stitch_dir = os.path.join(data_dir, 'StitchGrids')
moho_dir = os.path.join(data_dir, 'MohoInversion')
rf_dir = os.path.join(data_dir, 'ReceiverFunctionConstraints')

gravity_file = os.path.join(stitch_dir, 'BouguerFinalWithModel.asc')
initial_moho_file = os.path.join(moho_dir, 'InitialMoho_Uniform.asc')
output_moho_file = os.path.join(moho_dir, 'FinalMoho_RF_NoTe.asc')
log_file = os.path.join(moho_dir, 'inversion_log_v4.txt')
rf_file = os.path.join(rf_dir, 'MohoFromRF_Full_40N.mrc') #Defaultfilefrom Marco: MohoFromRF_40N.mrc

# ============================================================
# 读取/写入函数
# ============================================================

def log_message(message, log_file=None):
    """既打印到屏幕，也写入日志文件"""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

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

def tukey_1d(M, padding_ratio=0.5):
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

def build_taper(shape, padding_ratio=0.5):
    """构建二维 Tukey (余弦锥形) 窗"""
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
    """余弦低通滤波器"""
    k_low = 2 * np.pi / low_wl
    k_high = 2 * np.pi / high_wl
    
    filter_k = np.ones_like(k)
    mask_high = k > k_high
    filter_k[mask_high] = 0.0
    mask_trans = (k > k_low) & (k <= k_high)
    filter_k[mask_trans] = 0.5 * (1 + np.cos(np.pi * (k[mask_trans] - k_low) / (k_high - k_low)))
    
    return filter_k

def tikhonov_filter(k, alpha=ALPHA_TIKHONOV, order=TIKHONOV_ORDER):
    """Tikhonov 正则化滤波器"""
    k_norm = k / np.max(k)
    return 1.0 / (1.0 + alpha * k_norm**order)

def flexural_response_factor(k, te, drho_kg, g=9.81, e=1e11, nu=0.25):
    """挠曲响应因子"""
    D = (e * te**3) / (12 * (1 - nu**2))
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
    
    F_gravity_si = -2 * np.pi * G_NEW * drho_kg * upward_cont * flex_factor * sum_term
    gravity_si = np.real(ifft2(F_gravity_si))
    gravity_mgal = gravity_si * 1e5
    
    return gravity_mgal

def parker_inverse_step(gravity_residual, h_current, drho, z0, dx, dy, k, downward_cont, 
                        flex_factor, filter_combined, tikhonov, n_max=4):
    """Parker-Oldenburg 反演单步 (含弹性挠曲 + Tikhonov 正则化)"""
    G_NEW = 6.67430e-11
    drho_kg = drho * 1000
    
    gravity_residual_si = gravity_residual / 1e5
    F_residual = fft2(gravity_residual_si)
    
    inv_flex = 1.0 / flex_factor
    inv_flex = np.clip(inv_flex, 1.0, MAX_INV_FLEX)
    
    F_delta_h = -F_residual / (2 * np.pi * G_NEW * drho_kg) * downward_cont * inv_flex
    
    for n in range(2, n_max + 1):
        h_pow_prev = h_current ** (n-1)
        F_h_pow_prev = fft2(h_pow_prev)
        term_correction = (k ** (n-1)) / math.factorial(n-1) * F_h_pow_prev
        F_delta_h -= term_correction
    
    F_delta_h = F_delta_h * filter_combined * tikhonov
    delta_h = np.real(ifft2(F_delta_h))
    
    return delta_h

# ============================================================
# 辅助函数：创建网格坐标
# ============================================================

def create_grid_coords(header):
    """根据 ASC 头信息创建网格坐标"""
    nx = header['ncols']
    ny = header['nrows']
    xll = header['xllcorner']
    yll = header['yllcorner']
    cellsize = header['cellsize']
    
    grid_x = xll + (np.arange(nx) + 0.5) * cellsize
    grid_y = yll + (np.arange(ny) + 0.5) * cellsize
    
    return grid_x, grid_y

# ============================================================
# 主程序
# ============================================================

def main():
    total_start = time.time()
    
    # 初始化 RF 约束
    rf_constraint = None
    if USE_RF_CONSTRAINT and RF_AVAILABLE:
        try:
            rf_constraint = RFConstraint(
                rf_file=rf_file,
                sigma=RF_SIGMA,
                lambda_c=RF_LAMBDA_C,
                gamma=RF_GAMMA
            )
            log_message(f"✓ RF constraint initialized: {RF_SIGMA/1000:.0f} km sigma, "
                  f"{RF_LAMBDA_C/1000:.0f} km cutoff", log_file)
        except Exception as e:
            log_message(f"⚠ Failed to initialize RF constraint: {e}", log_file)
            rf_constraint = None
    elif USE_RF_CONSTRAINT and not RF_AVAILABLE:
        log_message("⚠ RF constraint requested but module not available", log_file)
    
    # 创建日志文件
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("=" * 80 + "\n")
        log.write("Parker-Oldenburg Moho Inversion v4 (with RF Constraints)\n")
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
        log.write(f"  USE_RF_CONSTRAINT = {USE_RF_CONSTRAINT and rf_constraint is not None}\n")
        if rf_constraint:
            log.write(f"  RF_SIGMA = {RF_SIGMA/1000:.0f} km\n")
            log.write(f"  RF_LAMBDA_C = {RF_LAMBDA_C/1000:.0f} km\n")
            log.write(f"  RF_GAMMA = {RF_GAMMA}\n")
            log.write(f"  RF_LAMBDA = {RF_LAMBDA}\n")
        log.write("\n" + "=" * 80 + "\n\n")
    
    log_message("="* 70, log_file)
    log_message("Parker-Oldenburg Iterative Moho Inversion v4 (with RF Constraints)", log_file)
    log_message("=" * 70, log_file)
    log_message(f"  Density contrast: {DRHO} g/cm3", log_file)
    log_message(f"  Mean depth: {Z0} km", log_file)
    log_message(f"  Elastic thickness: {TE} km (enabled: {USE_FLEXURE})", log_file)
    log_message(f"  Learning rate: {LEARNING_RATE}", log_file)
    log_message(f"  Filter wavelengths: {LOW_PASS_WL} ~ {HIGH_PASS_WL} km", log_file)
    log_message(f"  Tikhonov alpha: {ALPHA_TIKHONOV}", log_file)
    log_message(f"  RF constraint: {'Enabled' if rf_constraint else 'Disabled'}", log_file)
    log_message("=" * 70, log_file)
    
    # 1. 读取数据
    log_message("\n1. Reading data...", log_file)
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
    
    log_message(f"   Original size: {nx} x {ny}", log_file)
    log_message(f"   Valid points: {np.sum(valid_mask)}", log_file)
    log_message(f"   Time: {time.time() - step_start:.2f}s", log_file)
    
    # 2. 数据插值与去均值
    log_message("\n2. Interpolating NaNs & Removing mean...", log_file)
    step_start = time.time()
    
    grav_filled = fill_nans_fast(gravity_obs)
    moho_filled = fill_nans_fast(moho_initial * 1000)  # km -> m
    
    gravity_mean = np.nanmean(gravity_obs[valid_mask])
    grav_zm = grav_filled - gravity_mean
    h_rel = moho_filled - z0_m
    
    log_message(f"   Gravity mean: {gravity_mean:.2f} mGal", log_file)
    log_message(f"   Time: {time.time() - step_start:.2f}s", log_file)
    
    # 3. 扩边 (Padding) 与衰减窗 (Tapering)
    log_message("\n3. Padding & Tapering for FFT...", log_file)
    step_start = time.time()
    
    target_nx = next_fast_len(nx + int(nx * PADDING_RATIO))
    target_ny = next_fast_len(ny + int(ny * PADDING_RATIO))
    
    pad_x_left = (target_nx - nx) // 2
    pad_x_right = target_nx - nx - pad_x_left
    pad_y_top = (target_ny - ny) // 2
    pad_y_bottom = target_ny - ny - pad_y_top
    
    grav_pad = np.pad(grav_zm, ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right)), mode='edge')
    h_rel_pad = np.pad(h_rel, ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right)), mode='edge')
    
    taper = build_taper(grav_pad.shape)
    grav_pad_tapered = grav_pad * taper
    h_rel_pad_tapered = h_rel_pad * taper
    
    log_message(f"   Original: {nx} x {ny}", log_file)
    log_message(f"   Padded: {target_nx} x {target_ny}", log_file)
    log_message(f"   Pad margins: top={pad_y_top}, bottom={pad_y_bottom}, left={pad_x_left}, right={pad_x_right}", log_file)
    log_message(f"   Time: {time.time() - step_start:.2f}s", log_file)
    
    # 4. 计算波数与滤波器
    log_message("\n4. Computing wavenumbers and filters...", log_file)
    step_start = time.time()
    
    KX, KY, K = compute_wavenumbers(target_nx, target_ny, dx, dy)
    
    low_wl_m = LOW_PASS_WL * 1000
    high_wl_m = HIGH_PASS_WL * 1000
    filter_lowpass = lowpass_filter(K, low_wl_m, high_wl_m)
    
    tikhonov = tikhonov_filter(K, ALPHA_TIKHONOV)
    upward_cont = np.exp(-K * z0_m)
    downward_cont = np.exp(K * z0_m)
    downward_cont = np.clip(downward_cont, 0, MAX_DW_CONT)
    
    drho_kg = DRHO * 1000
    if USE_FLEXURE:
        flex_factor = flexural_response_factor(K, TE * 1000, drho_kg, G_GRAV, E, NU)
        log_message(f"   Flexure enabled: TE={TE} km", log_file)
        log_message(f"   Flex factor range: [{np.min(flex_factor):.2e}, {np.max(flex_factor):.2e}]", log_file)
    else:
        flex_factor = np.ones_like(K)
        log_message(f"   Flexure disabled", log_file)
    
    filter_combined = filter_lowpass
    
    # 创建原始网格坐标（用于 RF 插值）
    grid_x, grid_y = create_grid_coords(gravity_header)
    
    log_message(f"   K range: [{np.min(K):.2e}, {np.max(K):.2e}] rad/m", log_file)
    log_message(f"   Filter: K_low={2*np.pi/low_wl_m:.2e}, K_high={2*np.pi/high_wl_m:.2e}", log_file)
    log_message(f"   Downward cont max: {np.max(downward_cont):.2e}", log_file)
    log_message(f"   Tikhonov range: [{np.min(tikhonov):.2e}, {np.max(tikhonov):.2e}]", log_file)
    log_message(f"   Time: {time.time() - step_start:.2f}s", log_file)
    
    # 5. 迭代反演
    log_message("\n5. Starting iterative inversion...", log_file)
    log_message("-" * 70, log_file)
    
    rms_history = []
    h_current_pad = h_rel_pad_tapered.copy()
    iter_start = time.time()
    
    for iteration in range(1, MAX_ITER + 1):
        iter_step_start = time.time()
        
        # 正演计算
        grav_calc_pad = parker_forward(h_current_pad, DRHO, Z0, dx, dy, K, upward_cont, flex_factor, N_MAX)
        
        # 计算重力残差
        gravity_residual_pad = grav_pad_tapered - grav_calc_pad
        resid_center = gravity_residual_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
        rms_res = np.sqrt(np.mean(resid_center[valid_mask]**2))
        rms_history.append(rms_res)
        
        # ========== RF 约束修正 (新增) ==========
        rf_correction_pad = np.zeros_like(gravity_residual_pad)
        if rf_constraint is not None:
            # 获取当前模型（原始尺寸）
            h_current_center = h_current_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
            moho_current_km = (z0_m + h_current_center) / 1000
            
            # 计算 RF 修正场
            rf_correction_center, n_valid = rf_constraint.get_correction(
                moho_current_km, grid_x, grid_y, dx, dy
            )
            
            if n_valid > 0:
                # 扩边到 pad 尺寸
                rf_correction_pad = np.pad(rf_correction_center, 
                                           ((pad_y_top, pad_y_bottom), 
                                            (pad_x_left, pad_x_right)), 
                                           mode='edge')
                log_message(f"      RF constraints: {n_valid} stations, "
                      f"correction range=[{np.nanmin(rf_correction_center):.2f}, "
                      f"{np.nanmax(rf_correction_center):.2f}] km", log_file)
        
        # 合并修正 (公式 12)
        # total_correction = gravity_residual + λ_RF * rf_correction
        total_correction_pad = gravity_residual_pad
        if rf_constraint is not None and np.any(rf_correction_pad != 0):
            total_correction_pad = gravity_residual_pad + RF_LAMBDA * rf_correction_pad
        
        # 反演修正量
        delta_h_pad = parker_inverse_step(total_correction_pad, h_current_pad, DRHO, Z0, dx, dy, 
                                        K, downward_cont, flex_factor, filter_combined, tikhonov, N_MAX)
        
        # 限制修正量并应用学习率
        delta_h_pad = np.clip(delta_h_pad, -MAX_DELTA_H * 1000, MAX_DELTA_H * 1000)
        delta_h_pad = LEARNING_RATE * delta_h_pad
        
        # 计算新模型
        h_new_pad = h_current_pad + delta_h_pad
        h_new_pad = np.clip(h_new_pad, -25000, 25000)
        
        # 备份当前模型
        h_prev_pad = h_current_pad.copy()
        
        # 计算变化率
        dh_center = delta_h_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
        h_center = h_new_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
        dh_rms = np.sqrt(np.mean(dh_center[valid_mask]**2))
        change_pct = 100 * dh_rms / (np.mean(np.abs(h_center[valid_mask])) + 1)
        
        # 收敛检查
        if iteration > 1:
            rms_change = abs(rms_history[-1] - rms_history[-2]) / rms_history[-2]
        else:
            rms_change = 1.0
        
        iter_time = time.time() - iter_step_start
        log_message(f"  Iter {iteration:2d}: RMS_res={rms_res:.2f} mGal, "
              f"Dh_rms={dh_rms/1000:.2f} km, "
              f"Change={change_pct:.1f}%, "
              f"Time={iter_time:.1f}s", log_file)
        

        
        # 早停检查
        if USE_EARLY_STOP and len(rms_history) > 2 and rms_history[-1] > rms_history[-2]:
            log_message(f"\n  [Early stop] RMS increased from {rms_history[-2]:.2f} to {rms_history[-1]:.2f} mGal", log_file)
            log_message(f"  Stopping at iteration {iteration-1}, restoring previous model", log_file)
            h_current_pad = h_prev_pad
            rms_history.pop()
            break
        
        # 正式更新模型
        h_current_pad = h_new_pad
        
        # 收敛判断
        if rms_change < TOLERANCE and iteration > 1:
            log_message(f"\n  [OK] Converged at iteration {iteration} (RMS change {rms_change:.2e} < {TOLERANCE})", log_file)
            break
    
    total_iter_time = time.time() - iter_start
    log_message("-" * 70, log_file)
    log_message(f"  Total iteration time: {total_iter_time:.1f}s", log_file)
    
    # 6. 后期处理与保存
    log_message("\n6. Restoring original domain & saving...", log_file)
    step_start = time.time()
    
    h_final_center = h_current_pad[pad_y_top:pad_y_top+ny, pad_x_left:pad_x_left+nx]
    moho_final_km = (z0_m + h_final_center) / 1000
    moho_final_km[~valid_mask] = np.nan
    
    log_message(f"   Min: {np.nanmin(moho_final_km):.1f} km", log_file)
    log_message(f"   Max: {np.nanmax(moho_final_km):.1f} km", log_file)
    log_message(f"   Mean: {np.nanmean(moho_final_km):.1f} km", log_file)
    log_message(f"   Std: {np.nanstd(moho_final_km):.1f} km", log_file)
    log_message(f"   Time: {time.time() - step_start:.2f}s", log_file)
    
    write_asc_grid(output_moho_file, moho_final_km, moho_header)
    log_message(f"   [OK] Saved: {output_moho_file}", log_file)
    
    total_time = time.time() - total_start
    log_message("\n" + "=" * 70, log_file)
    log_message("[OK] Inversion completed!", log_file)
    log_message(f"    Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", log_file)
    log_message(f"    Final RMS residual: {rms_history[-1]:.4f} mGal", log_file)
    log_message(f"    Output file: {output_moho_file}", log_file)
    log_message(f"    Log file: {log_file}", log_file)
    log_message("=" * 70, log_file)
    
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total time: {total_time:.1f}s\n")
        log.write(f"Final RMS residual: {rms_history[-1]:.6f} mGal\n")
        log.write("=" * 80 + "\n")

if __name__ == "__main__":
    main()