# Pokémon Legends: Arceus on Android (handover)

Title `01001F5010DFA000`, update 1.1.1 (`v262144`, SDK 11.4). Same pipeline as Pokémon Shield ([android.md](android.md)): the game's code runs natively,
graphics are replayed on Vulkan, one `.apks` holds the app and the game files.

## Where things are

| What | Path |
|---|---|
| Game profile (in the repo) | `dm/games/01001F5010DFA000/` |
| Build folder (extract, rewritten code, shaders) | `ARCEUS DUMP/01001F5010DFA000_build/` |
| Rewritten code for Android / for the Mac | `…_build/patched_android/`, `…_build/patched_v262144/` |
| Vulkan shaders (12,851 SPIR-V + `.vkr`) and the pack | `…_build/shader/vk/b-3f2830fcd2cf/`, `…_build/shader/shaders.pak` |
| Android files folder (what goes in the `.apks`) | `ARCEUS DUMP/Arceus-Android/files/` |
| Prewarm list (333 pipelines: your save + a new game) | `files/vk_pipelines_seed.bin` (from `…-work/vk/seed.bin`) |
| Reference save (your save, `main`) | `ARCEUS DUMP/save_ref/main` |
| Work folder: recording runs, prewarm list, test script | `ARCEUS DUMP/01001F5010DFA000-work/` (`tools/vkrun.sh`, `vk/seed.bin`) |

## What it took to boot (all in the repo)

- **Allocator address space:** the global StandardAllocator reserves 32 GB (`main+0x2b4faec`), cut to 3.5 GB (4 GB fit only when the host put the guest block at one address); plus the SDK's 1 GB node area cut
  (`sdk+0x34e048`). Both are in `dm/tools/patch_nso.py` by build ID. The profile has `mem_reserve 0x160000000`, `mem_alias 0x138000000`
  and `mem_map_limit on`.
- **Shared font:** the game hands the font straight to the font engine like SDK 10 games do (profile `shared_font ttf`).
- **Exclusive swap loops:** two loops in `main` spill to the stack between `ldaxr` and `stlxr`. A Cortex-A57 tolerates that; newer cores can
  drop the reservation, so the loop retries forever (a freeze about 25 s into loading a save). `patch_nso.py` now turns any such loop into one
  LSE `swp` in every game (`llsc_swaps`, listed in `<module>.cuts.tsv`).

## Vulkan replay fixes (`dm/gfx/nvn/nvn_vk.c`, `nvn_spv.c`)

- **Sampler ID 0** is a real sampler. It was treated as unbound, so it fell back to an unfiltered dummy and every gradient banded.
- **Transform feedback:** Arceus (like Z-A) skins characters with a vertex-only capture. `spv_tf_variant` builds a capture variant of the
  vertex shader (storage-buffer stores and push constants), and `draw()` runs it under rasterizer discard. The layout comes from the
  `tf` lines in `.vkr`, which `dm/android/tools/vk_meta.py` reads from the Metal source's `main0_tf` (pass the Metal cache dir as its
  third argument).
- **Push descriptor limit:** a program with more descriptors than `maxPushDescriptors` uses descriptor sets (the terrain has 29; Adreno
  allows 32). `DM_VK_MAXPUSH` lowers the limit for testing.
- **Mac only:** a failed host-memory import mirrors the pool instead of dying. MoltenVK allows 16 samplers per stage, so Mac runs use
  `DM_VK_MAXPUSH=16 MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=1` (already in `vkrun.sh`).

## Rebuild commands (from the repo root, macOS)

```sh
B="/Users/m1/Documents/ARCEUS DUMP/01001F5010DFA000_build"
# rewrite the code for Android (after changing patch_nso.py)
for m in main sdk subsdk1; do DM_PATCH_TARGET=android python3 dm/tools/patch_nso.py "$B/extract/v262144/ExeFS/$m" "$B/patched_android" $m; done
# reflection + shader pack (after changing vk_meta.py or the shaders)
V="$B/shader/vk/b-3f2830fcd2cf"; python3 dm/android/tools/vk_meta.py "$V" "$V" "$B/shader/cache/b-3f2830fcd2cf"
python3 dm/android/tools/spv_pak.py "$V" /tmp/v1.pak && python3 dm/android/tools/spv_pak.py --compress /tmp/v1.pak "$B/shader/shaders.pak"
# the .apks
cd dm/android && sh app/build_app.sh 01001F5010DFA000 "$B" "Pokémon Legends: Arceus" com.nxnative.arceus \
  "/Users/m1/Documents/ARCEUS DUMP/Arceus-Android/PokemonLegendsArceus.apks" "/Users/m1/Documents/ARCEUS DUMP/Arceus-Android/files"
```

After changing `patched_android/` or the shaders, copy them into the files folder again before building the `.apks`. The files folder takes the
**uncompressed** pack (`spv_pak.py <dir> files/shaders.pak`); `build_app.sh` compresses it into the APK itself.

Test on the Mac with the Vulkan renderer (the same code as the phone, through MoltenVK): `sh dm/android/mac_vk_build.sh 01001F5010DFA000`, then
`bash "ARCEUS DUMP/01001F5010DFA000-work/tools/vkrun.sh" <run dir> <seconds> done|new [VAR=value …]` (`done` = with your save). Never run it
alongside the Android emulator (memory).

## Known issues

- One blend pipeline (`5356a947…`) uses dual-source blending, but its fragment shaders never write the second output, so it fails on Metal and
  Vulkan alike. One effect is missing.
- Transform feedback captures plain draws only (no indexed, indirect or instanced ones), as on Metal. Arceus makes only plain ones.
- Prewarm on the Mac: 333 pipelines in 2.3 s, then 0 compiled in 5.5 min of play (246 without the list).
- Not yet run on the phone. Check `Download/` logs for `[RED]` lines and for `push descriptors: up to N per program`.

## Continuing in VS Code (Windows)

The Mac holds the game files and builds the full `.apks` (RomFS, shaders, prewarm list). On Windows you change the code, rebuild the
app's library and swap it into the Arceus app already on the phone; the 4 GB of game data stays installed.

### Once

1. **Get the code:** in your Windows checkout of androidswithy, `git pull` (commit `c7cb198` or later). Open the repo folder in VS Code.
2. **Install the game:** copy `PokemonLegendsArceus.apks` (Mac: `ARCEUS DUMP/Arceus-Android/`, 4.2 GB) to the phone and install it with SAI.
3. **Use one signing key on both machines:** the `.apks` is signed with the Mac's `dm/android/probe/out/debug.keystore` (store and key
   password `android`, alias `debug`). Copy that file to Windows and sign with it there. A library signed with your Windows key won't
   install over the Mac-built app (the phone keeps the first app's signature), and switching keys means uninstalling, which deletes the
   save (copy `files/save` off the phone first).
4. **Tools:** NDK r27c at `~/android-ndk/android-ndk-r27c`, Android build-tools 35 (zipalign, apksigner), Python 3, adb, all as for Shield
   (`dm/android/README.md`, "Building on Windows").

### The edit–build–test loop (VS Code terminal, from `dm/android`)

```sh
python tools/win_build_lib.py 01001F5010DFA000                      # → out/host-01001F5010DFA000-vulkan/lib/libnxnative.so
adb shell pm path com.nxnative.arceus                               # the installed base.apk's path
adb pull <that path>/base.apk base_old.apk
python tools/apk_swap.py base_old.apk base_new.apk lib/arm64-v8a/libnxnative.so=out/host-01001F5010DFA000-vulkan/lib/libnxnative.so
zipalign -f -P 16 16384 base_new.apk base_al.apk
apksigner sign --ks debug.keystore --ks-pass pass:android --key-pass pass:android --v1-signing-enabled false --alignment-preserved --out base_signed.apk base_al.apk
adb install-multiple -r -p com.nxnative.arceus base_signed.apk      # keeps the two data splits and the save
```

The game profile is compiled into the library: after editing `dm/games/01001F5010DFA000/game.tsv`, run `win_build_lib.py` again.

### Logs

The app writes `Download/PokemonShield-log.txt` (refreshed while playing) and `Download/PokemonShield-log-previous.txt` (the last run,
the one that crashed if one did). The name is the same for every game for now, so Arceus overwrites Shield's log. Fetch them with
`adb pull /sdcard/Download/PokemonShield-log-previous.txt`. Things to look for:
- `[RED]` lines;
- `push descriptors: up to N per program` (the Adreno limit; programs above it use descriptor sets);
- `prewarm: 365 pipelines listed … built … s` (startup prewarm);
- `pipelines built during play so far` (compiles while playing; the phone records them, so the next launch prewarms them too).

### What needs the Mac

Rewriting the game's code (`patch_nso.py`, after a cut or rule change), the shader build and `.vkr` reflection, recording prewarm runs
(`vkrun.sh`, MoltenVK), and building a fresh `.apks`. All of these read the game files in `ARCEUS DUMP/`.

### Open items (each waits for your OK)

- Touch controls: Arceus needs ZL (aim) and ZR (throw); Shield's layout hides them.
- Sharpening: a contrast-adaptive sharpening pass (CAS / FSR 1 RCAS) at presentation.
- Per-game log file names.
- One blend effect (pipeline `5356a947…`) is missing on Metal and Vulkan alike.
