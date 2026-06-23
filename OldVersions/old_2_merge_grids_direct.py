import numpy as np
import os

# ===== 设置数据路径 =====
data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
# ========================

def read_asc_header(filename):
    """读取 ESRI ASCII 栅格文件头"""
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            # 判断是整数还是浮点数
            if '.' in line[1] or 'e' in line[1].lower():
                value = float(line[1])
            else:
                value = int(line[1])
            header[key] = value
    return header

def read_asc_grid(filename, header=None):
    """读取 ESRI ASCII 栅格文件的数据部分"""
    if header is None:
        header = read_asc_header(filename)
    
    # 跳过 6 行头文件
    data = np.loadtxt(filename, skiprows=6)
    
    # 将 nodata 值替换为 NaN
    nodata = header['nodata_value']
    data[data == nodata] = np.nan
    
    return data

# 读取头信息
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

# 读取数据
print("读取 Land 数据...")
land_data = read_asc_grid(land_file, land_header)
print(f"  Land 数组形状: {land_data.shape}")

print("读取 Sea 数据...")
sea_data = read_asc_grid(sea_file, sea_header)
print(f"  Sea 数组形状: {sea_data.shape}")

# 创建初始合并数组（先用 Land 数据）
merged = land_data.copy()

# 计算 Land 网格的坐标范围
land_xmin = land_header['xllcorner']
land_ymax = land_header['yllcorner'] + land_header['nrows'] * land_header['cellsize']  # 左上角 Y
cellsize = land_header['cellsize']

# 遍历 Sea 的每个点，放入对应 Land 位置
print("\n将 Sea 数据覆盖到 Land 网格...")

# Sea 的坐标起始点
sea_xmin = sea_header['xllcorner']
sea_ymax = sea_header['yllcorner'] + sea_header['nrows'] * sea_header['cellsize']  # 左上角 Y

count_valid = 0
count_overlap = 0

for i in range(sea_header['nrows']):  # 行（Y 方向）
    # 计算 Sea 当前点的 Y 坐标（从上到下递减）
    sea_y = sea_ymax - i * cellsize
    
    # 找到对应的 Land 行索引
    land_row = int(round((land_ymax - sea_y) / cellsize))
    
    for j in range(sea_header['ncols']):  # 列（X 方向）
        sea_val = sea_data[i, j]
        
        # 跳过 Sea 的无效值
        if np.isnan(sea_val):
            continue
        count_valid += 1
        
        # 计算 Sea 当前点的 X 坐标
        sea_x = sea_xmin + j * cellsize
        
        # 找到对应的 Land 列索引
        land_col = int(round((sea_x - land_xmin) / cellsize))
        
        # 检查是否在 Land 范围内
        if 0 <= land_row < land_data.shape[0] and 0 <= land_col < land_data.shape[1]:
            merged[land_row, land_col] = sea_val
            count_overlap += 1

print(f"  Sea 有效点数: {count_valid}")
print(f"  成功覆盖到 Land 的点数: {count_overlap}")

# 保存合并后的网格（保存在当前脚本所在目录）
output_filename = 'BouguerMerged.asc'

with open(output_filename, 'w') as f:
    f.write(f"ncols        {land_header['ncols']}\n")
    f.write(f"nrows        {land_header['nrows']}\n")
    f.write(f"xllcorner    {land_header['xllcorner']:.3f}\n")
    f.write(f"yllcorner    {land_header['yllcorner']:.3f}\n")
    f.write(f"cellsize     {land_header['cellsize']}\n")
    f.write(f"nodata_value {land_header['nodata_value']}\n")
    
    # 将 NaN 转换回 nodata 值
    output_data = merged.copy()
    output_data[np.isnan(output_data)] = land_header['nodata_value']
    
    # 写入数据
    for row in range(output_data.shape[0]):
        f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')

print(f"\n✓ 合并完成！已保存到: {output_filename}")
