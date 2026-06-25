# rf_constraint.py
"""
Receiver Function 约束模块 （读取X Y 坐标）
实现重力-地震联合反演
"""

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator


class RFConstraint:
    """
    Receiver Function 约束类
    将稀疏的地震约束点转化为平滑的频域修正场
    """
    
    def __init__(self, rf_file=None, sigma=30000, lambda_c=300000, gamma=2.0):
        """
        初始化 RF 约束
        
        参数:
            rf_file: .mrc 约束文件路径
            sigma: 高斯扩散半径 (m) (公式3)
            lambda_c: 低通滤波截止波长 (m) (公式10)
            gamma: 空间权重指数 (公式7)
        """
        self.sigma = sigma
        self.lambda_c = lambda_c
        self.gamma = gamma
        
        # 默认路径（根据你的文件位置）
        if rf_file is None:
            base_dir = r'C:\Users\yangln\Desktop\Postdoc\CNR_Italy\Maps\BouguerAnomaly'
            rf_file = os.path.join(base_dir, 'ReceiverFunctionConstraints', 'MohoFromRF_40N.mrc')
        
        # 读取 RF 数据
        self.load_rf_data(rf_file)
    
    # ========== 数据加载 ==========
    
    def load_rf_data(self, filename):
        """读取 .mrc 格式的 RF 约束文件"""
        print(f"Loading RF data from: {filename}")
        
        try:
            data = np.loadtxt(filename)
        except:
            # 如果文件有表头，尝试跳过
            with open(filename, 'r') as f:
                lines = f.readlines()
            # 找出第一个有效数据行（不是注释）
            data_lines = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    try:
                        vals = list(map(float, line.strip().split()[:3]))
                        if len(vals) == 3:
                            data_lines.append(vals)
                    except:
                        continue
            data = np.array(data_lines)
        
        self.rf_x = data[:, 0]      # X 投影坐标 (m)
        self.rf_y = data[:, 1]      # Y 投影坐标 (m)
        self.rf_depth = data[:, 2]  # 地震 Moho 深度 (km)
        
        print(f"Loaded {len(self.rf_depth)} RF stations")
        print(f"  Depth range: {np.min(self.rf_depth):.1f} - {np.max(self.rf_depth):.1f} km")
        print(f"  X range: {np.min(self.rf_x):.0f} - {np.max(self.rf_x):.0f} m")
        print(f"  Y range: {np.min(self.rf_y):.0f} - {np.max(self.rf_y):.0f} m")
    
    # ========== 公式 (1): 残差计算 ==========
    
    def interpolate_at_points(self, gravity_moho, grid_x, grid_y):
        """
        公式 (2): 双线性插值，获取重力模型在 RF 点的深度
        
        参数:
            gravity_moho: 当前重力模型 (km)
            grid_x: 网格 X 坐标 (m)
            grid_y: 网格 Y 坐标 (m)
        
        返回:
            gravity_depth: RF 点的重力模型深度 (km)
        """
        # 确保坐标是升序的
        if grid_x[0] > grid_x[-1]:
            grid_x = grid_x[::-1]
            gravity_moho = gravity_moho[:, ::-1]
        if grid_y[0] > grid_y[-1]:
            grid_y = grid_y[::-1]
            gravity_moho = gravity_moho[::-1, :]
        
        # 创建插值器
        interpolator = RegularGridInterpolator(
            (grid_y, grid_x), gravity_moho, 
            method='linear', bounds_error=False, fill_value=np.nan
        )
        
        # 插值
        points = np.column_stack([self.rf_y, self.rf_x])
        gravity_depth = interpolator(points)
        
        return gravity_depth
    
    def compute_residuals(self, gravity_moho, grid_x, grid_y):
        """
        公式 (1): 计算残差
        r_k = H_RF(s_k) - H_est(s_k)
        
        返回:
            residuals: 各站点的残差 (km)
            valid: 有效站点掩码
        """
        # 获取重力模型在 RF 点的深度
        gravity_depth = self.interpolate_at_points(gravity_moho, grid_x, grid_y)
        
        # 计算残差
        residuals = self.rf_depth - gravity_depth
        
        # 标记有效站点（重力模型有值）
        valid = ~np.isnan(gravity_depth)
        
        return residuals, valid
    
    # ========== 公式 (3-4): 高斯扩散 ==========
    
    def gaussian_kernel(self, x0, y0, grid_x, grid_y):
        """
        公式 (3): 高斯核
        G_k(i,j) = exp(-d^2 / (2σ^2))
        
        参数:
            x0, y0: 站点坐标 (m)
            grid_x, grid_y: 网格坐标
        
        返回:
            kernel: 高斯核矩阵
        """
        # 计算距离平方
        dx = grid_x - x0
        dy = grid_y.reshape(-1, 1) - y0
        dist2 = dx**2 + dy**2
        
        # 高斯核
        kernel = np.exp(-dist2 / (2 * self.sigma**2))
        
        return kernel
    
    def diffuse_residuals(self, residuals, valid, grid_x, grid_y):
        """
        公式 (4): 加权平均扩散残差
        R(x,y) = Σ(r_k * G_k) / Σ(G_k)
        
        参数:
            residuals: 各站点残差 (km)
            valid: 有效站点掩码
            grid_x, grid_y: 网格坐标
        
        返回:
            residual_field: 扩散后的残差场 (km)
        """
        ny, nx = len(grid_y), len(grid_x)
        residual_sum = np.zeros((ny, nx))
        weight_sum = np.zeros((ny, nx))
        
        # 遍历有效站点
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            x0 = self.rf_x[idx]
            y0 = self.rf_y[idx]
            r = residuals[idx]
            kernel = self.gaussian_kernel(x0, y0, grid_x, grid_y)
            residual_sum += r * kernel
            weight_sum += kernel
        
        # 避免除零
        weight_sum[weight_sum == 0] = 1.0
        residual_field = residual_sum / weight_sum
        
        return residual_field
    
    # ========== 公式 (5-7): 空间自适应权重 ==========
    
    def compute_influence_field(self, grid_x, grid_y):
        """
        公式 (5): 计算 RF 影响场
        W_RF(x,y) = Σ G_k(x,y)
        """
        ny, nx = len(grid_y), len(grid_x)
        influence = np.zeros((ny, nx))
        
        for (x0, y0) in zip(self.rf_x, self.rf_y):
            kernel = self.gaussian_kernel(x0, y0, grid_x, grid_y)
            influence += kernel
        
        return influence
    
    def compute_confidence_weights(self, grid_x, grid_y):
        """
        公式 (6-7): 计算置信度权重 β(x,y)
        W_N = W_RF / max(W_RF)
        β = (W_N)^γ
        """
        # 公式 (5): 计算影响场
        influence = self.compute_influence_field(grid_x, grid_y)
        
        # 公式 (6): 归一化
        max_influence = np.max(influence)
        if max_influence > 0:
            influence_norm = influence / max_influence
        else:
            influence_norm = np.zeros_like(influence)
        
        # 公式 (7): 指数加权
        beta = influence_norm ** self.gamma
        
        return beta
    
    # ========== 公式 (8): 应用置信度权重 ==========
    
    def apply_confidence_weights(self, residual_field, grid_x, grid_y):
        """
        公式 (8): R_RF(x,y) = β(x,y) * R(x,y)
        """
        beta = self.compute_confidence_weights(grid_x, grid_y)
        return residual_field * beta
    
    # ========== 公式 (9-11): 频域滤波 ==========
    
    def gaussian_lowpass_filter(self, k, lambda_c):
        """
        公式 (10): 高斯低通滤波器
        F(k) = exp(-(k/k_c)^2)
        
        参数:
            k: 波数 (rad/m)
            lambda_c: 截止波长 (m)
        
        返回:
            filter_k: 滤波器值
        """
        k_c = 2 * np.pi / lambda_c
        return np.exp(-(k / k_c)**2)
    
    def apply_frequency_filter(self, field, dx, dy):
        """
        公式 (9-11): 频域滤波
        
        参数:
            field: 输入场
            dx, dy: 网格间距 (m)
        
        返回:
            filtered: 滤波后的场
        """
        ny, nx = field.shape
        
        # 公式 (9): FFT 变换
        # 先做镜像延拓减少边缘效应
        field_mirror = self._mirror_extension(field)
        
        # FFT
        F = np.fft.fft2(field_mirror)
        
        # 计算波数
        kx = 2 * np.pi * np.fft.fftfreq(field_mirror.shape[1], dx)
        ky = 2 * np.pi * np.fft.fftfreq(field_mirror.shape[0], dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # 公式 (10): 计算滤波器
        filter_k = self.gaussian_lowpass_filter(K, self.lambda_c)
        
        # 公式 (11): 应用滤波
        # (λ_RF 在主循环中控制，这里先不乘)
        F_filtered = F * filter_k
        
        # 逆变换
        filtered_mirror = np.real(np.fft.ifft2(F_filtered))
        
        # 切回原始尺寸
        orig_ny, orig_nx = field.shape
        filtered = filtered_mirror[:orig_ny, :orig_nx]
        
        return filtered
    
    def _mirror_extension(self, field, pad_ratio=0.2):
        """
        镜像延拓，减少边缘效应
        """
        ny, nx = field.shape
        pad_y = int(ny * pad_ratio)
        pad_x = int(nx * pad_ratio)
        
        # 镜像填充
        field_mirror = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
        
        return field_mirror
    
    # ========== 主函数: 获取 RF 约束修正场 ==========
    
    def get_correction(self, gravity_moho, grid_x, grid_y, dx, dy):
        """
        获取 RF 约束修正场
        
        参数:
            gravity_moho: 当前重力模型 (km)
            grid_x, grid_y: 网格坐标 (m)
            dx, dy: 网格间距 (m)
        
        返回:
            rf_correction: RF 约束修正场 (km)
            valid_points: 有效站点数
        """
        # 公式 (1): 计算残差
        residuals, valid = self.compute_residuals(gravity_moho, grid_x, grid_y)
        
        n_valid = np.sum(valid)
        if n_valid == 0:
            print("Warning: No valid RF stations")
            return np.zeros_like(gravity_moho), 0
        
        # 公式 (3-4): 高斯扩散
        residual_field = self.diffuse_residuals(residuals, valid, grid_x, grid_y)
        
        # 公式 (5-7): 计算置信度权重
        beta = self.compute_confidence_weights(grid_x, grid_y)
        
        # 公式 (8): 应用权重
        rf_weighted = residual_field * beta
        
        # 公式 (9-11): 频域滤波
        rf_correction = self.apply_frequency_filter(rf_weighted, dx, dy)
        
        return rf_correction, n_valid
    
    # ========== 辅助: 生成站点位置图 ==========
    
    def get_station_map(self, grid_x, grid_y):
        """
        生成站点位置图（用于可视化）
        """
        ny, nx = len(grid_y), len(grid_x)
        station_map = np.zeros((ny, nx))
        
        for (x0, y0) in zip(self.rf_x, self.rf_y):
            # 找到最近网格点
            ix = np.argmin(np.abs(grid_x - x0))
            iy = np.argmin(np.abs(grid_y - y0))
            station_map[iy, ix] = 1
        
        return station_map


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试路径是否正确
    base_dir = r'C:\Users\yangln\Desktop\Postdoc\CNR_Italy\Maps\BouguerAnomaly'
    rf_file = os.path.join(base_dir, 'ReceiverFunctionConstraints', 'MohoFromRF_40N.mrc')
    
    if os.path.exists(rf_file):
        print(f"✓ RF file found: {rf_file}")
        rf = RFConstraint()
    else:
        print(f"✗ RF file not found: {rf_file}")
        print("Please check the file path")
