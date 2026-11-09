#!/usr/bin/env python3
# Flat normal maps (Pokémon Legends: Arceus "No reflections", profile key flat_normals): the normal maps of the field's ground and plants made flat in the game's own
# terrain packs, so their lighting has no fine detail to react to. On this engine those details lit the shadowed ground purple (docs/HANDOFF.md, problem 3); a
# flat normal map removes it, as the community's no_reflections mod does (its BC5 normal maps: every block one value, R 133 G 124). Made here from the player's own
# files, nothing shipped: each pack of the profile's folder is read, the BC5 textures whose names match the profile's patterns get every block of every level replaced
# by that one block (same size, same levels, same layout), and the packs holding any are written under <out>/romfs/<same path>, for the engine to lay over the RomFS
# (android_main.c flat_normals, hle_fs.c DM_ROMFS_OVERLAY).
# Game Freak's GFLXPACK: header (magic, version, relocated, file count, folder count, file table, hash table); file table entries of 24 bytes: level u16, compression
# u16 (0 none, 1 zlib, 2 LZ4 block, 3 Oodle), unpacked size u32, packed size u32, padding u32, data offset u64. A changed entry is appended at the end of the pack as an
# LZ4 block (as the mod's own packs carry some) and its table entry pointed there; everything else stays byte for byte.
# Oodle entries are decoded by the game's own decoder (dmoodle, dm/host/hle_oodle.c DM_OODLE_SERVE) with the filter "_nrm": only entries holding a normal map's name come
# back over the pipe.
# Usage: flat_normals.py --romfs <RomFS> --out <dir> --folder <RomFS-relative folder> --names <pattern>[,<pattern>...] --oodle <dmoodle> --modules <rewritten modules>
#        (patterns are fnmatch patterns on the texture's name; !<pattern> leaves those out)
#        flat_normals.py --test   (self-test: an LZ4 round trip, a flattened BNTX)
import argparse, fnmatch, os, struct, subprocess, sys, zlib

FLAT_BC5 = bytes.fromhex('85850000000000007c7c000000000000')   # R 133, G 124 in every texel: the no_reflections mod's block
BC5_UNORM = 0x1e01

def lz4_decode(src, n):
    out = bytearray(); i = 0; L = len(src)
    while i < L:
        t = src[i]; i += 1; l = t >> 4
        if l == 15:
            while True:
                b = src[i]; i += 1; l += b
                if b != 255: break
        out += src[i:i + l]; i += l
        if i >= L: break
        off = src[i] | src[i + 1] << 8; i += 2; ml = t & 15
        if ml == 15:
            while True:
                b = src[i]; i += 1; ml += b
                if b != 255: break
        ml += 4; st = len(out) - off
        if off >= ml: out += out[st:st + ml]
        else:
            for k in range(ml): out.append(out[st + k])
    if len(out) != n: raise ValueError(f'LZ4 entry decodes to {len(out)} bytes, not {n}')
    return bytes(out)

def _lsic(n):   # LZ4's length continuation bytes
    out = bytearray()
    while n >= 255: out.append(255); n -= 255
    out.append(n); return bytes(out)

def lz4_encode(data):
    """An LZ4 block: runs of one repeated 16-byte block (the flat texture) become one match each, everything else literals. Valid LZ4 for any decoder"""
    out = bytearray(); lit_start = 0; i = 0; n = len(data)
    def seq(lit, mlen, off):
        ll = len(lit); tok = (min(ll, 15) << 4) | (min(mlen - 4, 15) if mlen else 0)
        out.append(tok)
        if ll >= 15: out.extend(_lsic(ll - 15))
        out.extend(lit)
        if mlen:
            out.extend(struct.pack('<H', off))
            if mlen - 4 >= 15: out.extend(_lsic(mlen - 4 - 15))
    while i + 32 <= n - 12:   # (LZ4: the last 5 bytes are literals, and a match ends at least 12 bytes before the end)
        if data[i:i + 16] == data[i + 16:i + 32]:
            j = i + 32
            while j + 16 <= n - 12 and data[j:j + 16] == data[i:i + 16]: j += 16
            seq(data[lit_start:i + 16], j - (i + 16), 16); lit_start = i = j
        else: i += 16 if (i - lit_start) % 16 == 0 else 1
    seq(data[lit_start:], 0, 0)
    return bytes(out)

def flatten_bntx(b, patterns):
    """(new BNTX bytes, texture name) when b is a BC5 texture whose name matches, else None"""
    if b[:4] != b'BNTX' or b.count(b'BRTI') != 1: return None
    i = b.find(b'BRTI'); fmt, = struct.unpack_from('<I', b, i + 0x1c)
    if fmt != BC5_UNORM: return None
    no, = struct.unpack_from('<Q', b, i + 0x60); nl, = struct.unpack_from('<H', b, no); name = b[no + 2:no + 2 + nl].decode('utf-8', 'replace')
    if not any(fnmatch.fnmatchcase(name, p) for p in patterns if p[:1] != '!') or any(fnmatch.fnmatchcase(name, p[1:]) for p in patterns if p[:1] == '!'): return None   # (!pattern: never)
    size, = struct.unpack_from('<I', b, i + 0x50); ptab, = struct.unpack_from('<Q', b, i + 0x70); mip0, = struct.unpack_from('<Q', b, ptab)
    if mip0 + size > len(b) or size % 16: return None
    return b[:mip0] + FLAT_BC5 * (size // 16) + b[mip0 + size:], name

class Oodle:
    def __init__(self, tool, modules):
        env = dict(os.environ, DM_OODLE_SERVE='_nrm')   # (hle_oodle.c: entries without it answer ~0, nothing sent)
        self.p = subprocess.Popen([tool, modules], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    def decode(self, comp, n):
        self.p.stdin.write(struct.pack('<QQ', len(comp), n)); self.p.stdin.write(comp); self.p.stdin.flush()
        got, = struct.unpack('<Q', self.p.stdout.read(8))
        if got == (1 << 64) - 1: return None
        if got != n: raise RuntimeError(f'the game\'s Oodle decoder failed on a {len(comp)} → {n} byte entry')
        return self.p.stdout.read(n)
    def close(self): self.p.stdin.close(); self.p.wait()

def flatten_pack(d, patterns, oodle):
    """(new pack bytes, names flattened) or (None, [])"""
    if d[:8] != b'GFLXPACK': return None, []
    ver, rel, nf, nfo, ftab, htab = struct.unpack_from('<IIiiQQ', d, 8)
    out = bytearray(d); names = []
    for k in range(nf):
        lvl, typ, usz, csz, pad, off = struct.unpack_from('<HHIIIQ', d, ftab + 24 * k); raw = d[off:off + csz]
        if typ == 3: data = oodle.decode(raw, usz) if oodle else None
        elif typ == 2: data = lz4_decode(raw, usz)
        elif typ == 1: data = zlib.decompress(raw)
        elif typ == 0: data = raw
        else: continue
        if data is None or b'_nrm' not in data: continue
        r = flatten_bntx(data, patterns)
        if not r: continue
        new, name = r; comp = lz4_encode(new)
        while len(out) % 16: out.append(0)
        at = len(out); out += comp
        struct.pack_into('<HHIIIQ', out, ftab + 24 * k, lvl, 2, len(new), len(comp), pad, at); names.append(name)
    return (bytes(out), names) if names else (None, [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--romfs'); ap.add_argument('--out'); ap.add_argument('--folder'); ap.add_argument('--names'); ap.add_argument('--oodle'); ap.add_argument('--modules')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    if a.test: return selftest()
    pats = [p for p in a.names.split(',') if p]
    src = os.path.join(a.romfs, a.folder); dst = os.path.join(a.out, 'romfs', a.folder); tmp = a.out + '.part'
    import shutil; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(os.path.join(tmp, 'romfs', a.folder))
    oodle = Oodle(a.oodle, a.modules) if a.oodle else None
    total = 0; packs = 0
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.gfpak'): continue
        new, names = flatten_pack(open(os.path.join(src, fn), 'rb').read(), pats, oodle)
        if new is None: continue
        open(os.path.join(tmp, 'romfs', a.folder, fn), 'wb').write(new); total += len(names); packs += 1
        print(f'{fn}: {len(names)} normal map(s) flat', flush=True)
    if oodle: oodle.close()
    shutil.rmtree(a.out, ignore_errors=True); os.replace(tmp, a.out)
    print(f'DONE flat normals: {total} in {packs} pack(s) → {dst}', flush=True)

def selftest():
    import random
    rnd = random.Random(1)
    for data in (bytes(rnd.getrandbits(8) for _ in range(5000)), b'x' * 40 + FLAT_BC5 * 500 + b'tail-bytes-1234567', FLAT_BC5 * 3, b'', b'ab' * 7):
        assert lz4_decode(lz4_encode(data), len(data)) == data
    # a minimal one-texture BNTX: header, the BRTI fields read here, the name, the level pointer table, the data
    b = bytearray(0x400); b[:4] = b'BNTX'; i = 0x100; b[i:i + 4] = b'BRTI'
    struct.pack_into('<I', b, i + 0x1c, BC5_UNORM); struct.pack_into('<I', b, i + 0x50, 0x100); struct.pack_into('<Q', b, i + 0x60, 0x200); struct.pack_into('<Q', b, i + 0x70, 0x220)
    struct.pack_into('<H', b, 0x200, 13); b[0x202:0x20f] = b'ha_area00_nrm'; struct.pack_into('<Q', b, 0x220, 0x300); b[0x300:0x400] = bytes(range(256))
    new, name = flatten_bntx(bytes(b), ['ha_area*_nrm'])
    assert name == 'ha_area00_nrm' and new[0x300:0x400] == FLAT_BC5 * 16 and new[:0x300] == bytes(b[:0x300])
    assert flatten_bntx(bytes(b), ['plants_*_nrm']) is None and flatten_bntx(bytes(b), ['ha_*_nrm', '!ha_area00_nrm']) is None
    print('flat_normals self-test: ok')

if __name__ == '__main__': sys.exit(main())
