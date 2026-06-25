# rf_constraint_mercator.py
"""
Receiver Function 约束模块 (读取经纬度，使用 pyproj 精确投影)
实现重力-地震联合反演
优化版本: 预计算权重场，避免每次迭代重复计算
"""

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

# 尝试导入 pyproj
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not installed. Install with: conda install -c conda-forge pyproj")


def latlon_to_mercator(lat, lon, lat0=40, lon0=0):
    """
    将经纬度转换为墨卡托投影坐标 (标准纬线 40°N)
    
    参数:
        lat: 纬度 (度)
        lon: 经度 (度)
        lat0: 标准纬线 (度) - 墨卡托投影参数
        lon0: 中央经线 (度)
    
    返回:
        x, y: 投影坐标 (米)
    """
    if HAS_PYPROJ:
        # 使用 pyproj 进行精确转换
        source_crs = "EPSG:4326"  # WGS84 经纬度
        target_crs = f"+proj=merc +lat_ts={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        return x, y
    else:
        # 备用公式 (精确度较低)
        R = 6378137
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
        y = R * np.log(np.tan(np.pi/4 + lat_rad/2)) * np.cos(lat0_rad)
        return x, y


class RFConstraint:
    """
    Receiver Function 约束类
    将稀疏的地震约束点转化为平滑的频域修正场
    优化: 预计算权重场，避免每次迭代重复计算高斯核
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
        
        # 预计算的权重场 (避免重复计算)
        self.weight_field = None      # 预计算的高斯权重场
        self.beta_field = None        # 预计算的置信度权重场
        self.precomputed = False
        self.grid_x = None
        self.grid_y = None
        self.dx = None
        self.dy = None
        
        # 默认路径
        if rf_file is None:
            base_dir = r'C:\Users\yangln\Desktop\Postdoc\CNR_Italy\Maps\BouguerAnomaly'
            rf_file = os.path.join(base_dir, 'ReceiverFunctionConstraints', 'MohoFromRF_40N.mrc')
        
        # 读取 RF 数据
        self.load_rf_data(rf_file)
    
    # ========== 数据加载 ==========
    def load_rf_data(self, filename):
        """
        读取 .mrc 格式的 RF 约束文件
        只使用经纬度 (列 3, 4)，忽略 X, Y 列（可能为 NaN）
        """
        print(f"Loading RF data from: {filename}")
        
        # 手动解析文件，只读取需要的列
        data_lines = []
        skipped_count = 0
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                
                # 至少需要 5 列（因为要用深度、纬度、经度）
                if len(parts) < 5:
                    skipped_count += 1
                    continue
                
                try:
                    # 只读取列 2 (深度), 列 3 (纬度), 列 4 (经度)
                    depth_str = parts[2]
                    lat_str = parts[3]
                    lon_str = parts[4]
                    
                    # 处理 NaN 或空值
                    if depth_str.upper() == 'NAN' or depth_str == '':
                        continue  # 跳过深度无效的行
                    if lat_str.upper() == 'NAN' or lat_str == '':
                        continue
                    if lon_str.upper() == 'NAN' or lon_str == '':
                        continue
                    
                    depth = float(depth_str)
                    if depth == -9999 or depth < -9000:  # 假设 -9999 是无效值
                        continue
                    lat = float(lat_str)
                    lon = float(lon_str)
                    
                    data_lines.append([depth, lat, lon])
                    
                except (ValueError, IndexError):
                    skipped_count += 1
                    continue
        
        if len(data_lines) == 0:
            raise ValueError(f"No valid RF data found in {filename}")
        
        data = np.array(data_lines)
        
        print(f"   Loaded {len(data_lines)} stations (skipped {skipped_count} invalid lines)")
        
        # 提取数据
        self.rf_depth = data[:, 0]  # Moho 深度 (km)
        lat = data[:, 1]             # 纬度 (度)
        lon = data[:, 2]             # 经度 (度)
        
        # 转换为墨卡托坐标 (40°N 标准纬线)
        print("   Converting lat/lon to Mercator (40°N)...")
        self.rf_x, self.rf_y = latlon_to_mercator(lat, lon)
        
        # 统计信息
        print(f"Loaded {len(self.rf_depth)} RF stations")
        print(f"  Depth range: {np.min(self.rf_depth):.1f} - {np.max(self.rf_depth):.1f} km")
        print(f"  X range: {np.min(self.rf_x):.0f} - {np.max(self.rf_x):.0f} m")
        print(f"  Y range: {np.min(self.rf_y):.0f} - {np.max(self.rf_y):.0f} m")
        print(f"  Lat range: {np.min(lat):.3f} - {np.max(lat):.3f} deg")
        print(f"  Lon range: {np.min(lon):.3f} - {np.max(lon):.3f} deg")
    
    def precompute(self, grid_x, grid_y, dx, dy):
        """
        预计算高斯扩散权重场和置信度权重场
        只需要在第一次调用 get_correction 时计算一次
        """
        print("   Precomputing RF weight fields...")
        import time
        start = time.time()
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dx = dx
        self.dy = dy
        
        ny, nx = len(grid_y), len(grid_x)
        
        # 预计算权重场 (公式 4 的分母)
        print(f"      Computing weight field ({nx} x {ny})...")
        weight_sum = np.zeros((ny, nx))
        
        total = len(self.rf_x)
        for i, (x0, y0) in enumerate(zip(self.rf_x, self.rf_y)):
            # 每 50 个站点输出一次进度
            if i % 50 == 0:
                print(f"      Station {i+1}/{total}...")
            
            kernel = self._gaussian_kernel(x0, y0, grid_x, grid_y)
            weight_sum += kernel
        
        weight_sum[weight_sum == 0] = 1.0
        self.weight_field = weight_sum
        
        # 预计算置信度权重场 (公式 5-7)
        print(f"      Computing confidence weights...")
        influence = self._compute_influence_field(grid_x, grid_y)
        max_influence = np.max(influence)
        if max_influence > 0:
            influence_norm = influence / max_influence
        else:
            influence_norm = np.zeros_like(influence)
        self.beta_field = influence_norm ** self.gamma
        
        # 预计算低通滤波器
        print(f"      Computing frequency filter...")
        self._precompute_filter(dx, dy)
        
        self.precomputed = True
        elapsed = time.time() - start
        print(f"   Precomputation completed in {elapsed:.1f}s")
    
    def _precompute_filter(self, dx, dy):
        """预计算频域滤波器"""
        ny, nx = self.weight_field.shape
        
        # 镜像延拓尺寸
        pad_ratio = 0.2
        pad_y = int(ny * pad_ratio)
        pad_x = int(nx * pad_ratio)
        self.pad_y = pad_y
        self.pad_x = pad_x
        
        # 波数
        kx = 2 * np.pi * np.fft.fftfreq(nx + 2*pad_x, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny + 2*pad_y, dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # 高斯低通滤波器
        k_c = 2 * np.pi / self.lambda_c
        self.filter_k = np.exp(-(K / k_c)**2)
    
    def _gaussian_kernel(self, x0, y0, grid_x, grid_y):
        """高斯核"""
        dx = grid_x - x0
        dy = grid_y.reshape(-1, 1) - y0
        dist2 = dx**2 + dy**2
        return np.exp(-dist2 / (2 * self.sigma**2))
    
    def _compute_influence_field(self, grid_x, grid_y):
        """计算 RF 影响场"""
        ny, nx = len(grid_y), len(grid_x)
        influence = np.zeros((ny, nx))
        
        for (x0, y0) in zip(self.rf_x, self.rf_y):
            kernel = self._gaussian_kernel(x0, y0, grid_x, grid_y)
            influence += kernel
        
        return influence
    
    # ========== 公式 (1): 残差计算 ==========
    
    def interpolate_at_points(self, gravity_moho, grid_x, grid_y):
        """双线性插值，获取重力模型在 RF 点的深度"""
        # 直接使用原始坐标（假设已经是升序）
        interpolator = RegularGridInterpolator(
            (grid_y, grid_x), gravity_moho, 
            method='linear', bounds_error=False, fill_value=np.nan
        )
        
        points = np.column_stack([self.rf_y, self.rf_x])
        gravity_depth = interpolator(points)
        
        return gravity_depth
    
    def compute_residuals(self, gravity_moho, grid_x, grid_y):
        """公式 (1): 计算残差"""
        gravity_depth = self.interpolate_at_points(gravity_moho, grid_x, grid_y)
        residuals = self.rf_depth - gravity_depth
        valid = ~np.isnan(gravity_depth)
        return residuals, valid
    
    # ========== 快速扩散 (使用预计算权重) ==========
    
    def diffuse_residuals_fast(self, residuals, valid):
        """
        使用预计算的权重场快速扩散残差
        """
        if not self.precomputed:
            raise RuntimeError("Must call precompute() first")
        
        # 构建残差场 (在站点位置)
        ny, nx = self.weight_field.shape
        residual_sum = np.zeros((ny, nx))
        
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            x0 = self.rf_x[idx]
            y0 = self.rf_y[idx]
            r = residuals[idx]
            
            # 找到站点在网格中的索引
            ix = np.argmin(np.abs(self.grid_x - x0))
            iy = np.argmin(np.abs(self.grid_y - y0))
            
            # 高斯核（使用预计算的权重，但需要乘以残差值）
            # 这里仍然需要计算核，但可以优化为只在站点附近计算
            kernel = self._gaussian_kernel(x0, y0, self.grid_x, self.grid_y)
            residual_sum += r * kernel
        
        residual_field = residual_sum / self.weight_field
        
        return residual_field
    
    # ========== 公式 (5-7): 空间自适应权重（使用预计算） ==========
    
    def get_confidence_weights(self):
        """返回预计算的置信度权重场"""
        if not self.precomputed:
            raise RuntimeError("Must call precompute() first")
        return self.beta_field
    
    # ========== 公式 (9-11): 频域滤波 ==========
    
    def apply_frequency_filter(self, field):
        """使用预计算的滤波器进行频域滤波"""
        if not self.precomputed:
            raise RuntimeError("Must call precompute() first")
        
        ny, nx = field.shape
        
        # 镜像延拓
        field_mirror = np.pad(field, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)), mode='reflect')
        
        # FFT
        F = np.fft.fft2(field_mirror)
        
        # 滤波
        F_filtered = F * self.filter_k
        
        # 逆变换
        filtered_mirror = np.real(np.fft.ifft2(F_filtered))
        
        # 切回原始尺寸
        filtered = filtered_mirror[:ny, :nx]
        
        return filtered
    
    # ========== 主函数 ==========
    
    def get_correction(self, gravity_moho, grid_x, grid_y, dx, dy):
        """
        获取 RF 约束修正场
        
        参数:
            gravity_moho: 当前重力模型 (km)
            grid_x, grid_y: 网格坐标 (m)
            dx, dy: 网格间距 (m)
        
        返回:
            rf_correction: RF 约束修正场 (km)
            n_valid: 有效站点数
        """
        # 首次调用时预计算权重场
        if not self.precomputed:
            self.precompute(grid_x, grid_y, dx, dy)
        
        # 公式 (1): 计算残差
        residuals, valid = self.compute_residuals(gravity_moho, grid_x, grid_y)
        
        n_valid = np.sum(valid)
        if n_valid == 0:
            print("Warning: No valid RF stations")
            return np.zeros_like(gravity_moho), 0
        
        # 公式 (3-4): 高斯扩散 (使用快速方法)
        residual_field = self.diffuse_residuals_fast(residuals, valid)
        
        # 公式 (8): 应用置信度权重
        beta = self.get_confidence_weights()
        rf_weighted = residual_field * beta
        
        # 公式 (9-11): 频域滤波
        rf_correction = self.apply_frequency_filter(rf_weighted)
        
        return rf_correction, n_valid
    
    # ========== 辅助函数 ==========
    
    def get_station_map(self, grid_x, grid_y):
        """生成站点位置图"""
        ny, nx = len(grid_y), len(grid_x)
        station_map = np.zeros((ny, nx))
        
        for (x0, y0) in zip(self.rf_x, self.rf_y):
            ix = np.argmin(np.abs(grid_x - x0))
            iy = np.argmin(np.abs(grid_y - y0))
            station_map[iy, ix] = 1
        
        return station_map
    
    def get_rf_stations_in_grid(self, grid_x, grid_y):
        """返回网格内的站点数量"""
        x_min, x_max = grid_x[0], grid_x[-1]
        y_min, y_max = grid_y[0], grid_y[-1]
        
        in_grid = (self.rf_x >= x_min) & (self.rf_x <= x_max) & \
                  (self.rf_y >= y_min) & (self.rf_y <= y_max)
        
        return np.sum(in_grid), len(self.rf_x)


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试转换
    print("Testing coordinate conversion...")
    lat_test = 40.1505
    lon_test = 12.7565
    x, y = latlon_to_mercator(lat_test, lon_test)
    print(f"  IODP 651: lat={lat_test}, lon={lon_test} -> X={x:.0f}, Y={y:.0f}")
    
    # 加载 RF 数据
    print("\n" + "=" * 60)
    rf = RFConstraint()
    
    # 模拟网格坐标
    import numpy as np
    grid_x = np.linspace(560000, 1603100, 3478)
    grid_y = np.linspace(3239900, 4283000, 3478)
    dx = 300
    dy = 300
    
    # 测试预计算
    print("\nTesting precomputation...")
    dummy_moho = np.ones((3478, 3478)) * 20.0
    rf.precompute(grid_x, grid_y, dx, dy)
    
    # 获取修正场
    print("\nGetting correction...")
    correction, n_valid = rf.get_correction(dummy_moho, grid_x, grid_y, dx, dy)
    print(f"  Valid stations: {n_valid}")
    print(f"  Correction range: [{np.nanmin(correction):.2f}, {np.nanmax(correction):.2f}] km")
