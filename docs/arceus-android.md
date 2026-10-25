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
