import numpy as np
import os

# ===== 设置路径 =====
data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
merged_file = 'BouguerMerged.asc'  # 上一步生成的
# ====================

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

# 读取合并后的数据
header = read_asc_header(merged_file)
data = read_asc_grid(merged_file, header)

print(f"=== 合并后的数据统计 ===")
print(f"形状: {data.shape}")
print(f"有效点数 (非NaN): {np.sum(~np.isnan(data))}")
print(f"NaN 点数: {np.sum(np.isnan(data))}")

# 基本统计
valid_data = data[~np.isnan(data)]
print(f"\n数值统计 (仅有效数据):")
print(f"  最小值: {np.nanmin(data):.2f} mGal")
print(f"  最大值: {np.nanmax(data):.2f} mGal")
print(f"  平均值: {np.nanmean(data):.2f} mGal")
print(f"  标准差: {np.nanstd(data):.2f} mGal")

# 检查海岸线附近的台阶
# 方法：计算相邻行的差值（南北方向梯度），看看海陆交界处是否有异常大的梯度
print(f"\n检查潜在台阶 (南北方向梯度):")
dy_grad = np.gradient(data, axis=0)
dy_grad_abs = np.abs(dy_grad)
print(f"  梯度绝对值中位数: {np.nanmedian(dy_grad_abs):.4f} mGal/格")
print(f"  梯度绝对值 95% 分位数: {np.nanpercentile(dy_grad_abs, 95):.4f} mGal/格")
print(f"  梯度绝对值最大值: {np.nanmax(dy_grad_abs):.4f} mGal/格")

# 简单输出提示
if np.nanpercentile(dy_grad_abs, 99) > 10 * np.nanmedian(dy_grad_abs):
    print("\n⚠️ 警告：梯度分布异常，可能存在台阶或尖峰")
else:
    print("\n✓ 梯度分布正常，没有明显台阶")

# 可选：输出一个简单的剖面图（沿某一行，穿过海陆交界）
print("\n=== 沿 Y=2000 行的数据剖面 (前100列) ===")
sample_row = data[2000, :100]
print(" ".join([f"{x:.1f}" if not np.isnan(x) else " NaN" for x in sample_row]))
