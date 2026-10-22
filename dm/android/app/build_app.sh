#!/bin/sh
# Builds the game's Android app: the host library (Makefile.host lib), a NativeActivity manifest, the game's own icon.
# Usage: build_app.sh <GAME> <game work folder with extract/<ver>/icon.jpg> <name> <package> [out.apk] [game files folder]
# With a game files folder (RomFS, patched, shaders.pak, shader_map.tsv, font) it builds everything into one installable bundle, like the IPAs: APKs stop at 4 GB, so the
# files go stored and 16 KB-aligned (pages of 4 KB and 16 KB kernels) under assets/game/ of the base APK (everything but RomFS) and data split APKs (RomFS, packed up to SPLIT_MB each), all signed with the
# same key and zipped as <out>.apks (install with SAI or `adb install-multiple`). The host reads them in place through compat/apkfs.c. RomFS files that compress go
# under assets/gamez/ in 128 KB zlib blocks and the shader pack as a compressed version 2 pack (tools/nxz_pack.py; staged next to the game files folder).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); A=$(cd "$HERE/.." && pwd)
G=$1; WORK=$2; NAME=$3; PKG=$4; OUT=${5:-$A/out/$PKG.apk}; DATA=$6
SDK=${ANDROID_SDK:-$HOME/nxnative-work/android-sdk}; BT=$SDK/build-tools/35.0.0; JAR=$SDK/platforms/android-35/android.jar
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}; export PATH=$JAVA_HOME/bin:$PATH
(cd "$A" && make -f Makefile.host GAME=$G -j10 lib >/dev/null)
W=$A/out/app-$PKG; rm -rf "$W"; mkdir -p "$W/res/mipmap-xxxhdpi" "$W/res/mipmap-xxhdpi" "$W/res/mipmap-xhdpi" "$W/lib/arm64-v8a" "$W/compiled"
SPLITREQ=; [ -n "$DATA" ] && SPLITREQ='android:isSplitRequired="true"'
ICON=$(ls "$WORK"/extract/*/icon.jpg 2>/dev/null | tail -1)
for d in xxxhdpi:192 xxhdpi:144 xhdpi:96; do sips -s format png -z ${d#*:} ${d#*:} "$ICON" --out "$W/res/mipmap-${d%%:*}/ic_launcher.png" >/dev/null; done
cat > "$W/AndroidManifest.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="$PKG" android:versionCode="1" android:versionName="1">
    <uses-sdk android:minSdkVersion="33" android:targetSdkVersion="35" />
    <uses-permission android:name="android.permission.VIBRATE" />
    <uses-feature android:name="android.hardware.vulkan.version" android:version="0x403000" android:required="true" />
    <uses-feature android:name="android.hardware.touchscreen" android:required="false" />
    <uses-feature android:name="android.hardware.gamepad" android:required="false" />
    <application $SPLITREQ android:label="$NAME" android:icon="@mipmap/ic_launcher" android:hasCode="true" android:extractNativeLibs="false"
                 android:allowNativeHeapPointerTagging="false" android:isGame="true" android:appCategory="game" android:largeHeap="true"
                 android:theme="@android:style/Theme.Black.NoTitleBar.Fullscreen">
        <activity android:name="android.app.NativeActivity" android:exported="true" android:screenOrientation="sensorLandscape" android:launchMode="singleTask"
                  android:configChanges="orientation|screenSize|screenLayout|keyboard|keyboardHidden|navigation|uiMode|smallestScreenSize|density">
            <meta-data android:name="android.app.lib_name" android:value="nxnative" />
            <intent-filter><action android:name="android.intent.action.MAIN" /><category android:name="android.intent.category.LAUNCHER" /></intent-filter>
        </activity>
    </application>
</manifest>
XML
"$BT/aapt2" compile --dir "$W/res" -o "$W/compiled/res.zip"
"$BT/aapt2" link -o "$W/base.apk" -I "$JAR" --manifest "$W/AndroidManifest.xml" "$W/compiled/res.zip"
cp "$A/out/host-$G-vulkan/lib/libnxnative.so" "$W/lib/arm64-v8a/"
(cd "$W" && zip -q -0 base.apk lib/arm64-v8a/libnxnative.so)
# the app's few Java classes (the name-entry dialog, app/java): compiled for the platform and dexed into the base APK
mkdir -p "$W/classes" "$W/dex"; javac -source 8 -target 8 -nowarn -cp "$JAR" -d "$W/classes" $(find "$HERE/java" -name '*.java') 2>&1 | grep -v "^warning\|^Note\|^1 warning" || true
"$BT/d8" --release --min-api 33 --lib "$JAR" --output "$W/dex" $(find "$W/classes" -name '*.class'); (cd "$W/dex" && zip -q "$W/base.apk" classes.dex)
KS=$A/probe/out/debug.keystore
[ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias debug -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=nxnative debug" >/dev/null 2>&1
sign() { "$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --v1-signing-enabled false --alignment-preserved --out "$2" "$1"; }
mkdir -p "$(dirname "$OUT")"
if [ -z "$DATA" ]; then
    "$BT/zipalign" -f -P 16 4 "$W/base.apk" "$W/aligned.apk"; sign "$W/aligned.apk" "$OUT"
    echo "built $OUT ($(du -h "$OUT" | cut -f1))"; exit 0
fi
# The bundle. The stage holds the compressed files and clones of the rest (no copies)
DATA=$(cd "$DATA" && pwd); S=$(dirname "$DATA")/stage-apk; python3 "$A/tools/nxz_pack.py" "$DATA" "$S"
(cd "$S" && find assets/game/ -type f ! -path 'assets/game/RomFS/*' ! -name '.*' | sort | zip -q -0 "$W/base.apk" -@)
"$BT/zipalign" -f -P 16 16384 "$W/base.apk" "$W/aligned.apk"; B=$W/bundle; mkdir -p "$B"; sign "$W/aligned.apk" "$B/base.apk"; rm -f "$W/aligned.apk"
# RomFS into data splits, by size (each split's ZIP stays under 4 GB and 65535 entries)
(cd "$S" && find assets/game/RomFS assets/gamez/RomFS -type f ! -name '.*' -exec stat -f '%z %N' {} + 2>/dev/null | sort -k2) > "$W/romfs.lst"
awk -v lim=$(( ${SPLIT_MB:-3500} * 1048576 )) -v dir="$W" '{ sz = $1; sub(/^[0-9]+ /, ""); if (n == 0 || tot + sz + 16384 > lim || cnt >= 60000) { n++; tot = 0; cnt = 0 } tot += sz + 16384; cnt++; print > (dir "/split" n ".lst") }' "$W/romfs.lst"
for L in "$W"/split*.lst; do
    N=$(basename "$L" .lst); SP=data${N#split}
    cat > "$W/$SP.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="$PKG" split="$SP" android:versionCode="1" android:versionName="1">
    <application android:hasCode="false" />
</manifest>
XML
    "$BT/aapt2" link -o "$W/$SP.apk" -I "$JAR" --manifest "$W/$SP.xml"
    (cd "$S" && zip -q -0 "$W/$SP.apk" -@ < "$L")
    "$BT/zipalign" -f 16384 "$W/$SP.apk" "$W/$SP.al.apk"; rm -f "$W/$SP.apk"; sign "$W/$SP.al.apk" "$B/split_$SP.apk"; rm -f "$W/$SP.al.apk"
    echo "  split $SP: $(wc -l < "$L" | tr -d ' ') files, $(du -h "$B/split_$SP.apk" | cut -f1)"
done
case "$OUT" in *.apk) OUT=${OUT%.apk}.apks;; esac
rm -f "$OUT"; (cd "$B" && zip -q -0 "$OUT" base.apk split_*.apk)
echo "built $OUT ($(du -h "$OUT" | cut -f1)): base.apk + $(ls "$B"/split_*.apk | wc -l | tr -d ' ') data splits (APKs kept in $B)"
