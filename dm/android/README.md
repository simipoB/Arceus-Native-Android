# Android port

First target: Pokémon Shield (`01008DB008C2C000`) on the player's Galaxy S25+ (SM-S936W, Snapdragon 8 Elite SM8750, Adreno 830, 12 GB, Android 16, 4 KB pages, 39-bit user address space).

## Tools on the Mac

- SDK at `~/nxnative-work/android-sdk`: platform 35, build-tools 35.0.0, NDK 27.2.12479018, platform-tools (adb), emulator, `cmdline-tools/latest`.
- JDK 17: `/opt/homebrew/opt/openjdk@17`.
- Emulator: AVD `probe` (Android 15, arm64, Pixel 6) in `~/nxnative-work/android-avd`; start it with `ANDROID_AVD_HOME=~/nxnative-work/android-avd emulator -avd probe -no-window -no-audio -gpu swiftshader_indirect`. It runs ARM64 code natively, so it covers CPU / OS bring-up; its GPU is SwiftShader, so rendering and speed are checked on the phone.
- Command-line binaries run straight from `adb shell /data/local/tmp/...`: the fastest loop for host bring-up before any APK.

## Device probe (`probe/`)

`probe/build.sh` builds `nxnative-probe.apk` (no Gradle). It reports memory, register and Vulkan facts; the player sends it back with Share.
Results on the S25+:

- Code loading: anonymous RW → mprotect RX works, RX → RW → RX works, RWX works, memfd dual mapping works. loader.c's non-iOS path works as is.
- Address space: 39-bit (512 GB). PROT_NONE reservations up to 256 GB succeed. 0x8000000 is taken: modules load anywhere, as on Apple.
- x18: bionic's shadow call stack lives in x18 on every thread, and the kernel keeps x18. The guest must never touch the real x18: the rewriter's x18 → thread-slot rewrite stays.
- `tpidrro_el0` is 0 on Android (Darwin keeps its pthread TSD there). The rewriter's thread-slot route must change, see below.
- Vulkan: Adreno 830, Qualcomm driver, Vulkan 1.3.284. All BC formats sample and filter; D24S8 / X8D24 are depth targets; geometry and tessellation shaders, transform feedback, logicOp, dual-source blend, custom border colour, robustness2, push descriptors, extended dynamic state 1 and 2, vertex input dynamic state, descriptor buffer, index uint8, provoking vertex, depth clip control, conditional rendering are all there. Missing: extended dynamic state 3 and graphics pipeline libraries: pipelines are whole objects, so the recorded prewarm list (`pso_manifest.tsv`) matters as much as on iPad.
- Heaps: 11 GB device-local (unified memory).

## Decisions

1. **Thread slots on Android.** Patched code reaches the host's per-thread slots (K18, KTLS, KTLR, KGATE) with `mrs tpidrro_el0` + key offset on Apple. On Android it uses bionic's application slot `TLS_SLOT_APP` (`tpidr_el0 + 16`, free on every thread and kept by bionic through malloc, stdio and dlopen; tested on the emulator): `mrs T, tpidr_el0; ldr T, [T, #16]` gives a per-thread block, and the key page holds offsets into that block. One extra load per patched site. `patch_nso.py` gets a platform switch; patched images become per platform.
2. **Shaders.** The translator's SPIR-V goes to Vulkan directly: no spirv-cross, no MSL, no metallib. Prebuilt on the Mac, shipped in the APK; the driver's own compile is prewarmed from the profile's `pso_manifest.tsv` on first launch, with a `VkPipelineCache` saved after.
3. **Platform layer.** SDL3 (window, touch, controllers, audio), as the Link's Awakening Android port uses. AVC movies through AMediaCodec instead of VideoToolbox. Guest-thread sampling and crash dumps through signals and `ucontext` instead of Mach thread state.
4. **Graphics.** NVN → Vulkan replay beside `nvn_mtl.m`, sharing the NVN recording layer (`hle_nvn.c`).
5. **GPU drivers.** Qualcomm's own driver first (it has everything above). Loading a driver package (Turnip / Qualcomm builds as emulators use) through libadrenotools is an option in the app, not a requirement.

## Status (2026-10-22)

Phase 1 works on the emulator: `Makefile.host` builds the host as an Android command-line program (`out/host-<id>/dmhost`); Shield's modules rewritten with `DM_PATCH_TARGET=android`
load, relocate, pass the thread-slot check, run the SDK start-up, all module constructors and `nnMain`, mount RomFS, load the shared font, open audio (silent sink), create and commit saves,
and present at a locked 30 fps through the null GPU (`host/null_gfx.c`) for 3-minute scripted-input runs without a crash or red line. The NVN call counts per frame match the Mac host's
on the same input script frame for frame.

Run on the emulator (files under /data/local/tmp/nx: dmhost, patched_android/, RomFS/, font/, shader_map.tsv):

    adb shell 'cd /data/local/tmp/nx && DM_FONT_DIR=$PWD/font DM_ROMFS=$PWD/RomFS DM_SAVE=$PWD/save DM_TEMP=$PWD/temp DM_NVN_SHMAP=$PWD/shader_map.tsv ./dmhost patched_android'

What the Android build changes, all behind `__ANDROID__` / `__APPLE__` or in `compat/` so Apple builds are unchanged (the rewriter's Apple output is byte-identical):
`compat/` (Darwin stand-in headers and `compat.c`: Mach time, safe reads, dispatch semaphores, SHA-256, a pointer dictionary, xattr names, clonefile as a recursive copy,
signal-context registers, heap pointer tagging off), `DM_SECT` / `DM_ASYM` / `DM_UC_*` / `dm_slot_set` in `dm/host/dm.h`, `elf_asm.py` for the Mach-O assembly, the SIGTERM thread
in `hle_lifecycle.c`, the start time from /proc in `main.c`. Not yet on Android: sound output (AAudio), H.264 movies (AMediaCodec), Mach thread sampling in crash dumps.

## Vulkan replay (Phase 2, 2026-10-22)

`dm/gfx/nvn/nvn_vk.c` implements the Metal replay's interface (nvn_mtl.h) on Vulkan; `nvn_spv.c` reads the shaders' SPIR-V. On the Mac it runs through MoltenVK
(`mac_vk_build.sh <GAME>`, run with `VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`) and is checked against the Metal replay: Shield's Motostoke frame from the
100% save matches the Metal frame (mean absolute difference 4/255; the remaining difference is NPC animation timing). Shaders: `tools/vk_shaders.sh` keeps the translator's SPIR-V
(the chain and the Apple builds unchanged), `tools/vk_meta.py` writes each shader's reflection as `<key>.vkr`, `tools/spv_pak.py` packs both into `shaders.pak` (1.3 GB for Shield).
Diagnostics as the Metal replay's: `DM_NVN_SHOT=frame:path`, `DM_NVN_TEXDUMP=frame:path` (same file names, so the two dumps compare file by file).

Memory pools are zero-copy on both: imported host memory on MoltenVK, the driver's cached coherent memory moved onto the pool with `mremap` on Adreno (probe 2 on the S25+).
The emulator's SwiftShader has neither that nor push descriptors: the emulator runs the null GPU build (`GFX=null`) and the app up to graphics start.

## The app (Phase 3, first version)

`app/build_app.sh <GAME> <work folder> <name> <package>` builds the APK: NativeActivity (`app/jni/android_main.c`), the host as `libnxnative.so` (`Makefile.host lib`),
the game's icon. The game files go in `Android/data/<package>/files/`: `RomFS/`, `patched/` (DM_PATCH_TARGET=android modules), `shaders.pak`, `shader_map.tsv`, `font/`.
Present: a FIFO swapchain on the window, the game texture blitted to fit. Input: controllers (letters as labelled), touch zones (left half stick, right half A, two fingers B,
top-right X, top-centre +), back = B. Not yet: sound (silent sink), the designed touch layout (CONTROLS.md), H.264 movies, frame pacing beyond FIFO.

## Phases

1. Host on Android as a command-line binary (graphics null device): load Shield's modules, reach `nnMain`, run game logic. Emulator first, then the phone.
2. NVN → Vulkan replay, brought up on the Mac through MoltenVK, then on the phone.
3. The APK: SDL3 front end, context-aware touch controls with an editable layout, controller support, the game's icon, prebuilt shaders, prewarm.
4. Shield polish: performance, memory, long-session stress runs (the iPad stress tools carry over through adb input).

## All-in-one bundle
`app/build_app.sh <GAME> <work> <name> <pkg> <out.apks> <game files folder>` packs the game into one installable `.apks`: base APK (host library, patched modules,
shaders.pak, shader map, font) and data split APKs (RomFS, ≤3.5 GB each, because an APK can't go past 4 GB), all stored and 4096-aligned under `assets/game/`.
The host reads them in place through `compat/apkfs.c` as `/nxapk/...`, so nothing is extracted or copied on the phone. Saves stay in the app's external files folder.
Install with SAI (Split APKs Installer) or `adb install-multiple base.apk split_data*.apk`. Tested on the emulator: 5 APKs indexed, RomFS mounted from `/nxapk`,
modules mapped from the APK, boot reaches Vulkan init (the emulator's SwiftShader has no push descriptors, so the real GPU test is the phone).

## Stress test (2026-10-22)
- **In-APK files** (`tools/apkfs_test.c`, emulator, 8 threads): all 50,527 files / 14.4 GB read back bit-identical to the source folder through read in random chunk sizes,
  pread at random offsets, mmap of whole files and fopen + fread; 160,000 open / close cycles racing; ENOENT / EROFS / ENOTDIR cases. Found and fixed: `/nxapk` without a
  trailing slash was not recognised; apksigner re-aligned stored entries to 4 bytes (mmap of entries failed), now signed with `--alignment-preserved`.
- **Saves on Android**: the null-GPU host on the emulator with game data from the installed APKs and saves in the app's external folder: save from the menu (X, R, A),
  `save commit #1`, file changed; a second run with `DM_TEST_NO_EXCHANGE=1` (the new fallback swap through `.swp` renames, for file systems without RENAME_EXCHANGE) saved
  too, and the Android-made save loads on the Mac ("Last saved" shows the Android time). An interrupted fallback swap is settled before the save folder is mounted (`dm_swap_recover`).
- **Shaders**: all 38,547 SPIR-V shaders of the pack pass `spirv-val --target-env vulkan1.3`. Vertex outputs now start at (0, 0, 0, 1) like Maxwell's untyped attribute slots
  (`spv_init_outputs`, the SPIR-V side of dmsh_build.py's vs_init: 18,308 vertex shaders), and a fragment shader reading varyings its vertex shader never writes gets a variant
  where they hold (0, 0, 0, 1) (`spv_drop_inputs`, vs_pad's counterpart: 221 of 16,245 pairs, before this a pipeline error: the save screen's party sprites were missing).
  Every rewritten shader validates.
- **Adreno hardening** (character select cycled 60 times under the Khronos validation layer): unused `Float64` / fragment-interlock declarations stripped at load
  (`spv_strip_unused`: every shader declared them, none used them), float outputs to integer colour targets written as their bits in a fragment variant (`spv_int_outputs`),
  no blending on integer targets, unfiltered samplers for integer textures. Left: integer textures sampled through float samplers (wrong values at worst).
- **Compression**: RomFS files compressed only below 70 % of their size (10 GB bundle): frame times match the uncompressed bundle (median 33 ms, boot hitch 87 vs 94 ms).
- **App**: touch controls drawn by the Vulkan present path (`touch_ui.c`, `nvn_ovl.h`, haptics through the Vibrator), AAudio output (`compat/aaudio_unit.c`, 5.1 mixed to
  stereo), controllers merged per device (a Joy-Con pair plays as one), the library linked with `--no-undefined`.
- **Crash after character selection (fixed)**: games free and make memory pools all the time (131 times in two minutes of character creation). On the mremap route,
  freeing a pool unmapped the driver's old range again after `vkFreeMemory`, racing other threads whose new mappings could already sit there; making a pool left a
  gap between moving the driver's pages and the placeholder. Now: `MREMAP_DONTUNMAP` (no gap), the pool's pages replaced in one step, the placeholder left for the driver.
  `tools/remap_race_test.c` (30,000 cycles with 6 threads mapping and unmapping meanwhile): the old sequence segfaults, the new one has 0 failures.
- **Without push descriptors** (`VK_KHR_push_descriptor` missing, `DM_VK_NO_PUSH`): per-draw descriptor sets from per-submission pools. The emulator with `-gpu host`
  (gfxstream on MoltenVK) still cannot run the game: its driver sub-allocates mapped memory from 16 MB blocks, so the mremap route cannot share it (a clear FATAL now).
- **Log within reach**: `Download/PokemonShield-log.txt` (refreshed every 5 s, with memory lines) and `Download/PokemonShield-log-previous.txt` (the previous run).
- **Memory leak (fixed)**: composite programs (stages bound separately, which Shield does all the time) lived in a 64-entry ring; Shield cycles through more than
  64 a frame, so nearly every draw made a new host program with two new Vulkan layouts, never freed (3.56 million in 90 s on the Mac). Now one per shader combination,
  made once: heap 2,085 MB → 119 MB at 90 s, +1.5 MB from 90 to 180 s, process memory flat over 10-minute sessions.
- **Pipeline list and cache**: every pipeline the game makes is recorded (`files/vk_pipelines.bin`); the shipped list (`vk_pipelines_seed.bin`, 1,194 pipelines from a new
  game and a completed save) and the recorded one are built at start on a background thread (one at a time, paused below 1.5 GB free), the driver's compiled pipelines
  kept in `files/vk_pipeline_cache.bin` (saved every 30 s while new ones appear, and when the app goes to the background). Same 4-minute session: 571 pipelines built
  during play without it, 5 with it.
- **Presentation and CPU** (after LAS Recomp's Android port): MAILBOX when the window has it (`DM_VK_PRESENT`), a performance hint session (target = the frame period,
  each frame's work reported), the presenting thread at priority −10, no rebinding of an already bound pipeline; the log line every 30 s carries memory and heat.
- **Text entry**: the game's keyboard requests (names, nicknames, box names; numbers get the number pad) open an Android dialog (`app/java/com/nxnative/TextInput.java`);
  the keyboard's Done key confirms. **DLC**: the Expansion Pass keys (`aoc/1`, `aoc/2`) ship in the base APK (`DM_AOC_DIR`).
- **Without a whole-mapping driver** (the emulator): mirrored pools (`DM_VK_POOL_SHADOW`), so the real app with graphics runs in the emulator (about 3 fps in software).

## Fixes on the phone (2026-10-22, built from Windows)
- **Java from game threads**: game threads run on the game's stacks, outside their pthread stacks, and ART aborts any such thread that attaches
  (`Thread::InitStack: FindStackTop() > GetStackEnd()`). That closed the app at the name entry (`front_text_input`) and on leaving the app (releasing the window
  surface calls into Java). Both now run on short-lived plain threads (`android_main.c`, `nvn_vk.c` `swap_job_run`).
- **Memory pools: the fd route** (`nvn_vk.c` `g_fdmap`). Moving Adreno's mappings onto the pools (mremap) broke on area changes and save loads: the driver hands the
  mappings of freed allocations out again and sub-allocates small ones, so new pools landed on placeholders (crashes in `mtl_pool_add`, a lost device, the driver
  faulting in its own code). Now each pool is a dedicated allocation exported with `VK_KHR_external_memory_fd` and that fd mapped `MAP_SHARED | MAP_FIXED` over
  the pool's pages: zero-copy like mremap, the driver's own mapping never touched. `DM_VK_NO_FDMAP=1` brings back the mremap route (which now validates the driver's
  mapping, moves pages back before freeing and keeps freed pool memory for reuse); `DM_VK_POOL_SHADOW=1` the mirrored route (stable but 70–230 ms submissions).
  Loading the save, leaving and entering houses (both families'), Wedgehurst and Leon's arrival: no crash, 30 fps.
- **Rendering**: misaligned read-only storage buffers (Adreno wants 64) bound as snapshots instead of zeros; packaged uncompressed textures get colour-attachment
  usage (the game draws into one; it was "cannot be attached", its pass lost: Wooloo's fur came out flat).
- **Touch**: system bars hidden (`app/java/com/nxnative/Immersive.java`; the navigation bar sat beside A / B / R and opened recents), outer controls kept ~0.2 of the
  height off the long edges, a right stick (camera) below-left of A B X Y and a D-pad below-right of the left stick, both faint 2 s after their last touch
  (they sit over the dialogue box). Controller device id −1 (adb's virtual keyboard) no longer dropped.
- **Building on Windows** (no Mac): `tools/win_build_lib.py <GAME>` builds `libnxnative.so` with the Windows NDK r27c; `tools/apk_swap.py` replaces it (and
  `classes.dex`, javac + d8 from build-tools 35) in an installed base.apk pulled with adb; then `zipalign -P 16 16384` and `apksigner --alignment-preserved`, and
  `adb install-multiple -r -p <package> base.apk` keeps the data splits. The phone's app is signed with a key made on Windows (not the Mac's `debug.keystore`):
  keep both machines on one key, or a Mac build needs an uninstall (copy `files/save` off first).
- Open: one render target format still "cannot be attached" after an area change; a cube texture with 1 layer reads 0; an edge swipe can open recents and swipe the game away.
