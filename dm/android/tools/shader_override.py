#!/usr/bin/env python3
# Our own shaders in place of the game's (nvn_vk.c shader_load: shader_override/<hash>.spv in the game's files folder). The work flow:
#   1. find the costly pass: DM_VK_PASS_TIMES (or ☰ → Bug reports → Detailed performance log) logs each frame's 15 costliest passes with their fragment shader's hash
#   2. shader_override.py get <shaders.pak> <hash> <dir>   writes <hash>.spv, <hash>.spvasm (spirv-dis) and <hash>.vkr (the reflection the replacement must keep)
#   3. edit <hash>.spvasm (cheaper maths, fewer samples, an early out; same inputs, outputs, bindings and push constants)
#   4. shader_override.py build <dir>/<hash>.spvasm        assembles it (spirv-as) and validates it (spirv-val, Vulkan 1.1) to <hash>.spv
#   5. adb push <hash>.spv <files folder>/shader_override/  and start the game: the log says "shader <hash>: our own version"
# The NDK's shader tools are used (spirv-dis / spirv-as / spirv-val in ndk/shader-tools); NDK_SHADER_TOOLS points elsewhere. The pack is the phone's shaders.pak
# (adb pull <files folder>/shaders.pak): versions 1-4, plain or zlib blobs (spv_pak.py)
import os, struct, subprocess, sys, zlib

def tools():
    d = os.environ.get('NDK_SHADER_TOOLS')
    if not d:
        for c in (os.path.expanduser('~/android-ndk/android-ndk-r27c/shader-tools/windows-x86_64'), os.path.expanduser('~/android-ndk/android-ndk-r27c/shader-tools/linux-x86_64')):
            if os.path.isdir(c): d = c; break
    if not d: sys.exit('set NDK_SHADER_TOOLS to the NDK shader-tools folder')
    ext = '.exe' if os.name == 'nt' else ''
    return {t: os.path.join(d, t + ext) for t in ('spirv-dis', 'spirv-as', 'spirv-val')}

def pak_get(pak, key_hex):
    key = bytes.fromhex(key_hex); data = open(pak, 'rb').read()
    if data[:4] != b'NXSP': sys.exit('not a shader pack')
    ver, n = struct.unpack_from('<II', data, 4); stride = 44 if ver >= 3 else 32; z = ver in (2, 4)
    def blob(o, l):
        if not l: return None
        b = data[o:o + l]
        return zlib.decompress(b[4:]) if z else b
    lo, hi = 0, n
    while lo < hi:
        m = (lo + hi) // 2; e = data[28 + stride * m:28 + stride * (m + 1)]
        if e[:8] == key:
            so, sl, ro, rl = struct.unpack_from('<QIQI', e, 8)
            return blob(so, sl), blob(ro, rl)
        if e[:8] < key: lo = m + 1
        else: hi = m
    sys.exit(f'{key_hex} is not in {pak}')

def main():
    if len(sys.argv) < 3: sys.exit(__doc__ or 'usage: shader_override.py get <pak> <hash> <dir> | build <file.spvasm>')
    t = tools()
    if sys.argv[1] == 'get':
        pak, h, out = sys.argv[2], sys.argv[3].lower(), sys.argv[4]; os.makedirs(out, exist_ok=True)
        spv, vkr = pak_get(pak, h)
        open(os.path.join(out, h + '.spv'), 'wb').write(spv); open(os.path.join(out, h + '.vkr'), 'wb').write(vkr or b'')
        subprocess.run([t['spirv-dis'], os.path.join(out, h + '.spv'), '-o', os.path.join(out, h + '.spvasm')], check=True)
        print(f'{h}: {len(spv)} bytes; edit {os.path.join(out, h + ".spvasm")} (keep the interface listed in {h}.vkr)')
    elif sys.argv[1] == 'build':
        src = sys.argv[2]; dst = os.path.splitext(src)[0] + '.spv'
        subprocess.run([t['spirv-as'], '--target-env', 'vulkan1.1', src, '-o', dst], check=True)
        r = subprocess.run([t['spirv-val'], '--target-env', 'vulkan1.1', dst], capture_output=True, text=True)
        if r.returncode: sys.exit('spirv-val refused it:\n' + r.stderr)
        print(f'{dst}: valid; push it to <files folder>/shader_override/')
    else: sys.exit('get or build')

if __name__ == '__main__':
    main()
