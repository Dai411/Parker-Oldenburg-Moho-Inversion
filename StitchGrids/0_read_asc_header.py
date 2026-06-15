import numpy as np

def read_asc_header(filename):
    """Read ESRI ASCII Grid Header"""
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            value = float(line[1]) if '.' in line[1] or 'e' in line[1].lower() else int(line[1])
            header[key] = value
    return header

# 读取两个文件的头信息
land_header = read_asc_header('C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/BouguerLandGrid.asc')
sea_header = read_asc_header('C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/BouguerSeaGrid.asc')

print("=== BouguerLandGrid.asc ===")
for k, v in land_header.items():
    print(f"  {k}: {v}")

print("\n=== BouguerSeaGrid.asc ===")
for k, v in sea_header.items():
    print(f"  {k}: {v}")

# 计算实际边界
land_xmin = land_header['xllcorner']
land_xmax = land_header['xllcorner'] + land_header['ncols'] * land_header['cellsize']
land_ymin = land_header['yllcorner']
land_ymax = land_header['yllcorner'] + land_header['nrows'] * land_header['cellsize']

sea_xmin = sea_header['xllcorner']
sea_xmax = sea_header['xllcorner'] + sea_header['ncols'] * sea_header['cellsize']
sea_ymin = sea_header['yllcorner']
sea_ymax = sea_header['yllcorner'] + sea_header['nrows'] * sea_header['cellsize']

print("\n=== 空间范围 ===")
print(f"Land: X [{land_xmin:.0f}, {land_xmax:.0f}], Y [{land_ymin:.0f}, {land_ymax:.0f}]")
print(f"Sea:  X [{sea_xmin:.0f}, {sea_xmax:.0f}], Y [{sea_ymin:.0f}, {sea_ymax:.0f}]")

# 检查重叠
overlap_xmin = max(land_xmin, sea_xmin)
overlap_xmax = min(land_xmax, sea_xmax)
overlap_ymin = max(land_ymin, sea_ymin)
overlap_ymax = min(land_ymax, sea_ymax)

print(f"\n重叠区: X [{overlap_xmin:.0f}, {overlap_xmax:.0f}], Y [{overlap_ymin:.0f}, {overlap_ymax:.0f}]")

if overlap_xmax > overlap_xmin and overlap_ymax > overlap_ymin:
    print("✓ 两个网格存在重叠")
else:
    print("✗ 两个网格不存在重叠")
