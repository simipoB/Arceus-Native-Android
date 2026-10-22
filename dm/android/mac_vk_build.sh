#!/bin/sh
# The Mac host with the Vulkan replay (nvn_vk.c, through MoltenVK) instead of the Metal one, to check the Vulkan replay against the Metal replay on the same machine.
# Usage: mac_vk_build.sh <GAME> [out]   (needs: brew install molten-vk vulkan-loader vulkan-headers; run the host with VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); D=$(cd "$HERE/.." && pwd); G=${1:?GAME}; OUT=${2:-$HERE/out/dmhost-vk-$G}
cd "$D/host"; make GAME=$G game_profile >/dev/null; [ -f gen/vpx/libvpx.a ] || make GAME=$G gen/vpx/libvpx.a >/dev/null
GD=$(python3 ../games/game.py dir $G 2>/dev/null || echo ../games/$G)
SRC="main.c loader.c hle.c crash.c probe.c thread.c memstat.m $(ls hle_*.c | tr '\n' ' ') front_mac.m entry.S"
GFX="$(ls ../gfx/*.c ../gfx/nvn/*.c ../gfx/nvn/*.S ../gfx/tex/*.c | tr '\n' ' ') ../android/app/jni/touch_ui.c"
mkdir -p "$(dirname "$OUT")"
clang -O1 -g -Wall -Wno-unused-function -I. -I../gfx/nvn -I/opt/homebrew/include -I../gfx/nvn/include -I../third_party/vulkan-headers/include -Igen/$G -I../games/$G -I../third_party/libvpx -I../third_party/stb_vorbis \
    -fobjc-arc -o "$OUT" $SRC $GFX gen/vpx/libvpx.a -L/opt/homebrew/lib -lvulkan -lz -Wl,-rpath,/opt/homebrew/lib \
    -framework Cocoa -framework QuartzCore -framework Metal -framework AudioToolbox -framework VideoToolbox -framework CoreMedia -framework CoreVideo -framework ImageIO -framework UniformTypeIdentifiers
echo "built $OUT"
