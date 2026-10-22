#!/usr/bin/env python3
# Stages a game files folder for the Android bundle (app/build_app.sh): every RomFS file that shrinks to under 70 % goes compressed to <stage>/assets/gamez/<path>
# in the block format compat/apkfs.c reads ("NXZ1", u32 block size, u64 size, u32 block count, u32 offsets [count + 1], blocks: zlib or raw); everything else is an
# APFS clone at <stage>/assets/game/<path> (no copy). The shader pack goes in as a version 2 pack (spv_pak.py --compress). Prints the sizes.
# Usage: nxz_pack.py <game files folder> <stage folder>
import os, shutil, struct, subprocess, sys, zlib
from concurrent.futures import ProcessPoolExecutor
BS = 128 * 1024
def pack(job):
    src, zdst, rdst = job
    data = open(src, 'rb').read(); raw = len(data); offs = [0]; blocks = []
    for i in range(0, raw, BS):
        b = data[i:i + BS]; c = zlib.compress(b, 9)
        if len(c) >= len(b): c = b   # a block that does not shrink stays raw (its length equals the block's size)
        blocks.append(c); offs.append(offs[-1] + len(c))
    total = 20 + 4 * len(offs) + offs[-1]
    if raw and total < raw * 0.7:   # only files that shrink well: the rest stay raw (no decompression on load, mmap-able)
        os.makedirs(os.path.dirname(zdst), exist_ok=True)
        with open(zdst, 'wb') as f:
            f.write(b'NXZ1' + struct.pack('<IQI', BS, raw, len(blocks)) + struct.pack(f'<{len(offs)}I', *offs)); f.writelines(blocks)
        return raw, total
    os.makedirs(os.path.dirname(rdst), exist_ok=True); subprocess.run(['cp', '-c', src, rdst], check=True)
    return raw, raw
if __name__ == '__main__':
    src, stage = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    shutil.rmtree(stage, ignore_errors=True); jobs = []; other = 0
    for d, _, names in os.walk(src):
        for nm in names:
            if nm.startswith('.'): continue
            p = os.path.join(d, nm); rel = os.path.relpath(p, src)
            if rel.startswith('RomFS/'): jobs.append((p, os.path.join(stage, 'assets/gamez', rel), os.path.join(stage, 'assets/game', rel)))
            elif rel == 'shaders.pak':
                os.makedirs(os.path.join(stage, 'assets/game'), exist_ok=True)
                subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spv_pak.py'), '--compress', p, os.path.join(stage, 'assets/game', rel)], check=True)
            else:
                q = os.path.join(stage, 'assets/game', rel); os.makedirs(os.path.dirname(q), exist_ok=True); subprocess.run(['cp', '-c', p, q], check=True); other += os.path.getsize(p)
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))   # big files first, so the pool ends evenly
    with ProcessPoolExecutor(6) as ex: res = list(ex.map(pack, jobs, chunksize=8))
    raw = sum(r for r, _ in res); out = sum(o for _, o in res); nz = sum(1 for r, o in res if o != r)
    print(f'RomFS: {len(res)} files, {raw / 1e9:.2f} GB → {out / 1e9:.2f} GB ({nz} compressed)')
