#!/usr/bin/env python3
# Reflection for the Vulkan replay: each <key>_b.json the shader build wrote (dmsh_build.py) → <key>.vkr, a line format the C replay reads without a JSON parser (nvn_vk.c vkr_load).
# Lines: "chain <version>", "stage <vs|ps|cs|...>", "b <name> <kind> <reg> <off> <bank> <rw> <n>" per binding (reg -1 = none), "c1 <hex>", "drv <c0 read> <c2 read>",
# "tg <x> <y> <z>", "colors <mask> <known>", "vin <location> <u|i|f>", "unfill <what>" when the shader reads driver-segment locations the replay cannot fill.
# Usage: vk_meta.py <json dir> <out dir>   (only <key>.spv files present in <out dir> get a .vkr)
import json, os, re, sys

def conv(j):
    out = [f"chain {j.get('chain', '')}", f"stage {j.get('stage', '')}"]
    for b in j.get('bindings', []):
        name = b.get('msl_name', '?'); kind = b.get('kind', '?'); reg = b.get('reg'); ty = b.get('msl_type', '')
        n = 1
        m = re.match(r'array<.*, (\d+)>$', ty or '')
        if m: n = int(m.group(1))
        out.append(f"b {name} {kind} {reg if isinstance(reg, int) else -1} {b.get('off', 0)} {b.get('bank', 0)} {1 if b.get('rw') else 0} {n}")
    c1 = j.get('c1') or ''
    out.append(f"c1 {c1}")
    dr = j.get('drv_reads') or {}
    out.append(f"drv {1 if '0' in dr else 0} {1 if '2' in dr else 0}")
    tg = j.get('threadgroup_size')
    if isinstance(tg, list) and len(tg) == 3: out.append(f"tg {tg[0]} {tg[1]} {tg[2]}")
    ra = j.get('raster') or {}
    cols = ra.get('colors')
    if isinstance(cols, list): out.append(f"colors {sum(1 << c for c in cols if isinstance(c, int) and 0 <= c < 8)} 1")
    for v in j.get('vertex_inputs', []):
        t = v.get('msl_type', '')
        out.append(f"vin {v.get('location', 0)} {'u' if t.startswith('uint') else 'i' if t.startswith('int') else 'f'}")
    for u in j.get('drv_unfillable') or []: out.append(f"unfill {u}")
    return '\n'.join(out) + '\n'

def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    for f in os.listdir(dst):
        if not f.endswith('.spv'): continue
        k = f[:-4]; jp = os.path.join(src, k + '_b.json')
        if not os.path.exists(jp): continue
        open(os.path.join(dst, k + '.vkr'), 'w').write(conv(json.load(open(jp))))
        n += 1
    print(f'vk_meta: {n} reflection files in {dst}')

if __name__ == '__main__': main()
