import argparse
import numpy as np
import os
from scipy.fft import fft2, ifft2, fftfreq

# ===== Set File Path =====
OUTPUT_DIR = 'C:/.../BouguerAnomaly/StitchGrids'
DEFAULT_INPUT_FILE = os.path.join(OUTPUT_DIR, 'L_FilledLaplacian.asc')
DEFAULT_HEADER_FILE = os.path.join(OUTPUT_DIR, 'Land_Laplacian.asc')  # For header, which used for final size
DEFAULT_OUTPUT_TIKHONOV = os.path.join(OUTPUT_DIR, 'BouguerFinal_Tikhonov.asc')
DEFAULT_OUTPUT_DIRECT = os.path.join(OUTPUT_DIR, 'BouguerFinal.asc')
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


def read_asc_grid(filename):
    """Read .asc File, Convert -9999 value to NaN"""
    with open(filename, 'r') as f:
        for _ in range(6):
            f.readline()
        data = np.loadtxt(f)
    data[data == -9999] = np.nan
    return data


def write_asc_grid(filename, data, header):
    """Write .asc File，Convert NaN to -9999"""
    output_data = data.copy()
    output_data[np.isnan(output_data)] = -9999.0

    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value -9999.0\n")

        for row in range(output_data.shape[0]):
            f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')


def compute_laplacian_transfer_function(nx, ny, dx, dy):
    """
    Compute the transfer function of the discrete Laplacian operator H(kx, ky)
    Equation (17) in Document 2 : H = 2/dx^2 * (cos(kx*dx) - 1) + 2/dy^2 * (cos(ky*dy) - 1)
    """
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    H = (2.0 / (dx * dx)) * (np.cos(KX * dx) - 1) + (2.0 / (dy * dy)) * (np.cos(KY * dy) - 1)
    return H


def frequency_domain_inversion(L_data, H, use_tikhonov, alpha):
    valid_mask = ~np.isnan(L_data)
    L_clean = np.nan_to_num(L_data, nan=0.0)
    F = fft2(L_clean)

    if use_tikhonov:
        H_tikhonov = H / (H**2 + alpha**2)
        G = F * H_tikhonov
    else:
        H_safe = H.copy()
        small_mask = np.abs(H_safe) < 1e-10
        H_safe[small_mask] = 1e-10 * np.sign(H_safe[small_mask])
        H_safe[small_mask] = 1e-10
        G = F / H_safe

    g_reconstructed = np.real(ifft2(G))
    g_final = g_reconstructed.copy()
    g_final[~valid_mask] = np.nan
    return g_final


def parse_args():
    parser = argparse.ArgumentParser(
        description='Frequency domain inversion for Bouguer anomaly from a filled Laplacian grid.')
    parser.add_argument('--input', default=DEFAULT_INPUT_FILE,
                        help='Input L_FilledLaplacian.asc File Path')
    parser.add_argument('--header', default=DEFAULT_HEADER_FILE,
                        help='Output file name (default: auto-named based on method)')
    parser.add_argument('--output', default=None,
                        help='Output .asc file')
    parser.add_argument('--tikhonov', dest='tikhonov', action='store_true', 
                        help='Using Tikhonov Regularization')
    parser.add_argument('--no-tikhonov', dest='tikhonov', action='store_false', 
                        help='Direct frequency division (default, no Regularization）')
    parser.set_defaults(tikhonov=False) # Defauat: No regularization
    parser.add_argument('--alpha', type=float, default=1e-10,
                        help='Tikhonov Regularized parameter alpha，only valid with --tikhonov')
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = args.output or (DEFAULT_OUTPUT_TIKHONOV if args.tikhonov else DEFAULT_OUTPUT_DIRECT)

    print('=' * 60)
    print('Step 5: Frequency Domain Inversion (Document2 Step 5)')
    print('=' * 60)

    print('\n1. Read data from .asc file...')
    header = read_asc_header(args.header)
    L_data = read_asc_grid(args.input)

    ny, nx = L_data.shape
    dx = header['cellsize']
    dy = header['cellsize']

    print(f'   Grid Dimension: {ny} x {nx}')
    print(f'   Gird Size: dx={dx}, dy={dy} m')
    print(f'   Valid data in L_data: {np.sum(~np.isnan(L_data)):,}')
    print(f'   Nan in L_data: {np.sum(np.isnan(L_data)):,}')

    print('\n2. Build mask in valid area...')
    valid_mask = ~np.isnan(L_data)
    print(f'   Valid are: {np.sum(valid_mask):,} points ({100 * np.sum(valid_mask) / L_data.size:.1f}%)')

    print('\n3. Prepare FFT input（NaN -> 0）...')
    print(f'   alpha = {args.alpha:.2e}' if args.tikhonov else '   Direct frequency inversion, no alpha parameter')

    print('\n4. Compute discrete Laplacian transfer function H(kx, ky)...')
    H = compute_laplacian_transfer_function(nx, ny, dx, dy)
    print(f'   H Region: {np.min(H):.6e} ~ {np.max(H):.6e}')

    print('\n5. Frequency domain inversion (FFT -> {} -> IFFT)...'.format('Tikhonov Filter' if args.tikhonov else 'Direct Division'))
    g_final = frequency_domain_inversion(L_data, H, args.tikhonov, args.alpha)
    print('   IFFT Finished')

    print('\n6. 结果统计:')
    valid_g = g_final[~np.isnan(g_final)]
    print(f'   最终布格异常有效点: {len(valid_g):,}')
    print(f'   最小值: {np.min(valid_g):.6f} mGal')
    print(f'   最大值: {np.max(valid_g):.6f} mGal')
    print(f'   平均值: {np.mean(valid_g):.6f} mGal')
    print(f'   标准差: {np.std(valid_g):.6f} mGal')

    print('\n7. 保存最终结果...')
    write_asc_grid(output_file, g_final, header)
    print(f'   ✓ 最终布格异常: {output_file}')

    print('\n✅ Step 5 完成！')
    print('\n📌 下一步: 验证最终融合结果')


if __name__ == '__main__':
    main()

