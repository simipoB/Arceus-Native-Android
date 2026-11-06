# Copies an APK with entries replaced: apk_swap.py <src> <dst> [--extract-native-libs] <entry>=<file> ... (every entry keeps its compression; zipalign then realigns and apksigner signs)
# An entry the APK does not have is added, stored like the libraries (libadrenotools and its hooks). --extract-native-libs sets the manifest's
# android:extractNativeLibs to true, so the installer puts the libraries in a folder of their own (libadrenotools loads its hook libraries from there)
import struct, sys, zipfile
args = sys.argv[1:]; extract = "--extract-native-libs" in args; args = [a for a in args if a != "--extract-native-libs"]
src, dst = args[:2]; rep = dict(a.split("=", 1) for a in args[2:])

def manifest_extract(m):   # binary XML: the attribute whose name maps to android:extractNativeLibs (resource 0x010104ea) becomes boolean true
    m = bytearray(m); ids = []; off = struct.unpack_from("<H", m, 2)[0]; hit = 0
    while off < len(m):
        typ, hsz, size = struct.unpack_from("<HHI", m, off)
        if typ == 0x0180: ids = list(struct.unpack_from("<%dI" % ((size - hsz) // 4), m, off + hsz))   # resource map: string index -> resource id
        elif typ == 0x0102:   # start element: its attributes
            astart, asz, acount = struct.unpack_from("<HHH", m, off + 16 + 8)
            for k in range(acount):
                a = off + 16 + astart + k * asz; name = struct.unpack_from("<I", m, a + 4)[0]
                if name < len(ids) and ids[name] == 0x010104ea:   # raw value none, typed value: boolean (0x12) true (all bits)
                    struct.pack_into("<I", m, a + 8, 0xFFFFFFFF); m[a + 15] = 0x12; struct.pack_into("<I", m, a + 16, 0xFFFFFFFF); hit = 1
        off += size
    assert hit, "no android:extractNativeLibs attribute"
    return bytes(m)

with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", allowZip64=True) as zo:
    lib = None
    for i in zi.infolist():
        data = open(rep.pop(i.filename), "rb").read() if i.filename in rep else zi.read(i)
        if extract and i.filename == "AndroidManifest.xml": data = manifest_extract(data)
        if i.filename.startswith("lib/"): lib = i
        n = zipfile.ZipInfo(i.filename, i.date_time); n.compress_type = i.compress_type; n.external_attr = i.external_attr
        zo.writestr(n, data)
    for name, f in rep.items():   # new entries
        n = zipfile.ZipInfo(name, lib.date_time if lib else (2020, 1, 1, 0, 0, 0)); n.compress_type = lib.compress_type if lib else zipfile.ZIP_STORED
        if lib: n.external_attr = lib.external_attr
        zo.writestr(n, open(f, "rb").read())
print("wrote", dst)
