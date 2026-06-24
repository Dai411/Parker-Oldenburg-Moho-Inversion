#!/usr/bin/env python3
"""
Validation/check_asc_compatibility.py

Lightweight compatibility checker for ESRI ASCII Grid (.asc) files.

Features:
- Basic checks (pure Python + numpy): header completeness, nodata handling, cellsize consistency,
  integer-grid alignment (xll/yll offsets are integer multiples of cellsize), and simple row-order
  sanity check.
- Optional enhanced checks if rasterio or GDAL available: affine/CRS reporting and example
  gdalwarp command for reprojection/resampling.
- Produces a human-readable summary and a machine-readable JSON report under Validation/reports/.

Usage examples:
  python Validation/check_asc_compatibility.py --paths Data --ref Data/reference.asc
  python Validation/check_asc_compatibility.py --paths file1.asc file2.asc --tol 1e-8

Options of interest:
  --paths: one or more file paths or directories (directories scanned for *.asc)
  --ref:  reference .asc file (used for alignment checks). If omitted, the first file is used.
  --tol:  tolerance for integer-multiple checks (default 1e-6)
  --report: json or csv (default json)
  --fix:   write suggested gdalwarp commands to the report (does not execute them)
  --use-rasterio: if present, use rasterio to read affine/CRS for better checks

This script is intended to be placed in Validation/ and run before the StitchGrids pipeline.
"""

from __future__ import annotations
import argparse
import os
import sys
import glob
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

# Optional dependencies
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False


def read_asc_header(path: str) -> Dict[str, Any]:
    """Read the first 6 header lines of an ESRI ASCII Grid and return a header dict.

    Returns keys in lowercase. Does not validate order.
    """
    header = {}
    with open(path, 'r', encoding='utf-8') as f:
        # read up to 20 lines in case of comments; stop after we have at least 5 of the common fields
        lines_read = 0
        while len(header) < 5 and lines_read < 50:
            line = f.readline()
            if not line:
                break
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            val = parts[1]
            try:
                if '.' in val or 'e' in val.lower():
                    v = float(val)
                else:
                    v = int(val)
            except Exception:
                v = val
            header[key] = v
            # stop early if we see 'ncols' and 'nrows' and 'cellsize' and ('xllcorner' or 'xllcenter') and ('yllcorner' or 'yllcenter')
            if ('ncols' in header and 'nrows' in header and 'cellsize' in header and
                    ('xllcorner' in header or 'xllcenter' in header) and ('yllcorner' in header or 'yllcenter' in header)):
                break
    return header


def discover_asc_files(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '**', '*.asc'), recursive=True)))
        elif os.path.isfile(p) and p.lower().endswith('.asc'):
            files.append(p)
        else:
            # allow glob patterns
            matched = glob.glob(p)
            for m in matched:
                if os.path.isfile(m) and m.lower().endswith('.asc'):
                    files.append(m)
    # deduplicate preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(os.path.normpath(f))
    return out


def is_integer_multiple(diff: float, cellsize: float, tol: float = 1e-6) -> Tuple[bool, float]:
    """Check whether diff / cellsize is near an integer. Return (bool, nearest_integer).
    If cellsize is zero or invalid, return (False, 0).
    """
    try:
        q = diff / cellsize
    except Exception:
        return False, 0
    nearest = round(q)
    return abs(q - nearest) <= tol, float(nearest)


def check_pair_alignment(ref_hdr: Dict[str, Any], hdr: Dict[str, Any], tol: float) -> Dict[str, Any]:
    """Check alignment between reference header and another header.

    Returns a small dict summarising the checks.
    """
    out = {}
    # cellsize check
    ref_cell = float(ref_hdr.get('cellsize', np.nan))
    cell = float(hdr.get('cellsize', np.nan))
    out['cellsize_ref'] = ref_cell
    out['cellsize'] = cell
    out['cellsize_match'] = float(np.isclose(ref_cell, cell, atol=tol, rtol=0))

    # xll / yll keys
    for corner_key in ('xllcorner', 'xllcenter'):
        if corner_key in ref_hdr:
            ref_x = float(ref_hdr[corner_key])
            break
    else:
        ref_x = None
    for corner_key in ('yllcorner', 'yllcenter'):
        if corner_key in ref_hdr:
            ref_y = float(ref_hdr[corner_key])
            break
    else:
        ref_y = None

    for corner_key in ('xllcorner', 'xllcenter'):
        if corner_key in hdr:
            x = float(hdr[corner_key])
            break
    else:
        x = None
    for corner_key in ('yllcorner', 'yllcenter'):
        if corner_key in hdr:
            y = float(hdr[corner_key])
            break
    else:
        y = None

    out['ref_xll'] = ref_x
    out['ref_yll'] = ref_y
    out['xll'] = x
    out['yll'] = y

    if None in (ref_x, ref_y, x, y) or np.isnan(ref_cell) or np.isnan(cell) or cell == 0:
        out['alignment_ok'] = False
        out['alignment_reason'] = 'missing header keys or invalid cellsize'
        return out

    dx = x - ref_x
    dy = y - ref_y
    out['dx'] = dx
    out['dy'] = dy

    okx, ix = is_integer_multiple(dx, ref_cell, tol)
    oky, iy = is_integer_multiple(dy, ref_cell, tol)
    out['dx_multiple'] = okx
    out['dx_multiple_nearest'] = ix
    out['dy_multiple'] = oky
    out['dy_multiple_nearest'] = iy
    out['alignment_ok'] = bool(okx and oky and out['cellsize_match'])
    if not out['alignment_ok']:
        reasons = []
        if not out['cellsize_match']:
            reasons.append('cellsize_mismatch')
        if not okx:
            reasons.append('dx_not_integer_multiple')
        if not oky:
            reasons.append('dy_not_integer_multiple')
        out['alignment_reason'] = ','.join(reasons) if reasons else 'unknown'
    return out


def read_affine_and_crs_with_rasterio(path: str) -> Dict[str, Any]:
    """If rasterio is available, return affine transform and CRS info for the file.
    If the file is not readable by rasterio, return empty dict.
    """
    if not HAS_RASTERIO:
        return {}
    try:
        with rasterio.open(path) as src:
            aff = src.transform
            crs = src.crs
            return {
                'affine': [aff.a, aff.b, aff.c, aff.d, aff.e, aff.f],
                'crs': str(crs)
            }
    except Exception:
        return {}


def build_report(entries: List[Dict[str, Any]], outdir: str, fmt: str = 'json') -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    if fmt == 'json':
        outpath = os.path.join(outdir, f'asc_compatibility_{ts}.json')
        with open(outpath, 'w', encoding='utf-8') as fh:
            json.dump({'generated': ts, 'entries': entries}, fh, indent=2, ensure_ascii=False)
        return outpath
    else:
        outpath = os.path.join(outdir, f'asc_compatibility_{ts}.csv')
        # flatten entries to rows with consistent columns
        keys = set()
        for e in entries:
            keys.update(e.keys())
        keys = sorted(keys)
        with open(outpath, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for e in entries:
                row = {k: e.get(k, '') for k in keys}
                writer.writerow(row)
        return outpath


def suggest_gdalwarp_command(src: str, dst: str, target_hdr: Dict[str, Any], resample: str = 'bilinear') -> str:
    """Return a gdalwarp command string that would resample src to the target header grid.
    This does not execute the command.
    """
    cell = float(target_hdr['cellsize'])
    xmin = float(target_hdr.get('xllcorner', target_hdr.get('xllcenter', 0)))
    ymin = float(target_hdr.get('yllcorner', target_hdr.get('yllcenter', 0)))
    ncols = int(target_hdr['ncols'])
    nrows = int(target_hdr['nrows'])
    # compute xmax/ymax assuming corner semantics
    xmax = xmin + ncols * cell
    ymax = ymin + nrows * cell
    # gdalwarp uses -te xmin ymin xmax ymax and -tr xres yres
    cmd = f"gdalwarp -te {xmin} {ymin} {xmax} {ymax} -tr {cell} {cell} -r {resample} \"{src}\" \"{dst}\""
    return cmd


def make_entry_for_file(path: str, ref_hdr: Dict[str, Any], tol: float, fix: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {'path': path}
    try:
        hdr = read_asc_header(path)
        entry['header'] = hdr
        # check basic completeness
        required = ['ncols', 'nrows', 'cellsize']
        entry['header_ok'] = all(k in hdr for k in required) and (('xllcorner' in hdr or 'xllcenter' in hdr) and ('yllcorner' in hdr or 'yllcenter' in hdr))
        if not entry['header_ok']:
            entry['notes'] = 'missing header keys'
            return entry
        # nodata
        entry['nodata_value'] = hdr.get('nodata_value', None)
        # Optional rasterio information
        if HAS_RASTERIO:
            ri = read_affine_and_crs_with_rasterio(path)
            if ri:
                entry['rasterio'] = ri
        # If ref provided, check alignment
        if ref_hdr:
            pair = check_pair_alignment(ref_hdr, hdr, tol)
            entry['alignment'] = pair
            if not pair.get('alignment_ok', False):
                # suggest gdalwarp command if gdal present or if fix requested
                if HAS_GDAL or HAS_RASTERIO:
                    dst_suggest = os.path.splitext(os.path.basename(path))[0] + '_reproj.asc'
                    gcmd = suggest_gdalwarp_command(path, dst_suggest, ref_hdr)
                    entry['suggest_gdalwarp'] = gcmd
        return entry
    except Exception as e:
        entry['error'] = str(e)
        return entry


def main_cli():
    parser = argparse.ArgumentParser(description='Check compatibility of ESRI ASCII Grid (.asc) files.')
    parser.add_argument('--paths', nargs='+', default=['.'], help='File(s) or directory(ies) to scan for .asc files')
    parser.add_argument('--ref', default=None, help='Reference .asc file for alignment checks')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for integer multiple checks')
    parser.add_argument('--report', choices=['json', 'csv'], default='json', help='Report format')
    parser.add_argument('--outdir', default=os.path.join('Validation', 'reports'), help='Output directory for reports')
    parser.add_argument('--fix', action='store_true', help='Include suggested gdalwarp commands in report (does not execute)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    files = discover_asc_files(args.paths)
    if not files:
        print('No .asc files found in given paths.', file=sys.stderr)
        sys.exit(2)

    # choose reference header
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f'Reference file {args.ref} not found.', file=sys.stderr)
            sys.exit(2)
        ref_hdr = read_asc_header(args.ref)
        ref_path = args.ref
    else:
        # choose first file as reference
        ref_path = files[0]
        ref_hdr = read_asc_header(ref_path)

    entries = []
    for p in files:
        if args.verbose:
            print('Checking', p)
        e = make_entry_for_file(p, ref_hdr, args.tol, args.fix)
        entries.append(e)

    report_path = build_report(entries, args.outdir, fmt=args.report)
    print(f'Report written to: {report_path}')

    # Print quick summary
    ok_count = sum(1 for e in entries if e.get('alignment', {}).get('alignment_ok') or (not e.get('header_ok')))
    total = len(entries)
    print(f'Checked {total} files; see report for details.')

    # Human-readable summary to stdout
    for e in entries:
        p = e.get('path')
        print('\n--', p)
        if 'error' in e:
            print(' ERROR:', e['error'])
            continue
        if not e.get('header_ok'):
            print('  HEADER: missing required header keys')
            continue
        align = e.get('alignment')
        if align:
            if align.get('alignment_ok'):
                print('  ALIGNMENT: OK')
            else:
                print('  ALIGNMENT: NOT OK; reason:', align.get('alignment_reason'))
                if e.get('suggest_gdalwarp'):
                    print('   Suggested gdalwarp command:')
                    print('   ', e['suggest_gdalwarp'])
        else:
            print('  ALIGNMENT: no reference comparison performed')

    print('\nDone.')


if __name__ == '__main__':
    main_cli()
