# create_uniform_moho.py
"""
创建均一初始 Moho 模型
- 所有网格相同深度
- 用于诊断反演是否过度依赖初始模型
"""

import numpy as np
import os
# 参数
UNIFORM_DEPTH = 20   # km

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

# 文件路径
data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
model_file = os.path.join(data_dir, 'BouguerModelled.asc')
output_dir = os.path.join(data_dir, 'MohoInversion')
output_file = os.path.join(output_dir, 'InitialMoho_Uniform.asc')

# 读取网格参数
header = read_asc_header(model_file)
nx = header['ncols']
ny = header['nrows']

# 创建均一深度
moho_uniform = np.full((ny, nx), UNIFORM_DEPTH, dtype=np.float32)

# 保存
write_asc_grid(output_file, moho_uniform, header)
print(f"Saved: {output_file}")
print(f"  Uniform depth: {UNIFORM_DEPTH} km")
print(f"  Grid size: {ny} x {nx}")
