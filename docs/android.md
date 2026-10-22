# Porting a Switch game to Android

This guide turns a Switch game **you own** into an Android app. It does not emulate anything: the game's own ARM64 code runs directly on the phone's CPU, and its
graphics are rebuilt on Vulkan. The result is one `.apks` file (an app split into several APKs, because an APK cannot pass 4 GB) that you install with SAI.

Tested so far: **Pokémon Shield** (`01008DB008C2C000`, update 1.3.2 / v458752) on a **Galaxy S25+** (Snapdragon 8 Elite, Adreno 830, Android 16).
Everything Android-specific is in [`dm/android`](../dm/android); the technical notes are in [`dm/android/README.md`](../dm/android/README.md).

Nothing from any game is in this repository. You bring the game's files and your own `prod.keys`.

## What you need

- A Mac with Apple silicon (the build runs there; the game's shaders are translated there).
- Your game as NSP files (base game and update, plus any DLC) and the `prod.keys` from your own Switch.
- A phone with a 64-bit ARM CPU, Android 13 or newer, and Vulkan 1.3. Built and tuned for Snapdragon (Adreno). It needs push descriptors or falls back to plain
  descriptor sets, BC and ASTC textures (Adreno has both), and a driver that maps each memory allocation on its own (Adreno's does; others fall back to a slower copy).
- About 2.5 times the game's size free on the Mac while building, and the game's size plus a little free on the phone.

### Tools (once)

- **Android SDK** in `~/nxnative-work/android-sdk`: platform 35, build-tools 35.0.0, NDK 27.2.12479018, platform-tools. With Homebrew's `android-commandlinetools`:

  ```sh
  sdkmanager --sdk_root=$HOME/nxnative-work/android-sdk "platforms;android-35" "build-tools;35.0.0" "ndk;27.2.12479018" "platform-tools"
  ```

- **JDK 17**: `brew install openjdk@17`.
- For testing on the Mac (optional): `brew install molten-vk vulkan-loader vulkan-headers vulkan-validationlayers glslang spirv-tools`.
- Python 3 (comes with the Mac's developer tools).

## 1. Extract the game and build its shaders (the iPhone pipeline)

Run the packing script on the folder with your NSPs and `prod.keys`. It extracts the game, rewrites its code, splits and translates its shaders and makes a profile check.
Its outputs (in `<NSP folder>/<title ID>_build/`) are what the Android build starts from:

```sh
python3 scripts/build_ipa_from_nsp.py "<NSP folder>"
```

The game needs a profile in [`dm/games`](../dm/games) (see the main README's list). For a game without one, the script surveys it and guesses a provisional profile:
see [docs/agents](agents/README.md).

## 2. Rewrite the game's code for Android

The game's modules are rewritten once more with the Android target. Android keeps threads' state in a different register than Apple's systems, so the rewriter's thread-slot
code differs:

```sh
B="<NSP folder>/<title ID>_build"; V=<extract version, e.g. v458752>; O="$B/patched_android"; mkdir -p "$O"
for m in main sdk subsdk0 subsdk1; do
  [ -f "$B/extract/$V/ExeFS/$m" ] && DM_PATCH_TARGET=android python3 dm/tools/patch_nso.py "$B/extract/$V/ExeFS/$m" "$O" $m
done
```

## 3. Build the Vulkan shaders

The same shader translation as for the iPhone, but the SPIR-V is kept (the iPhone build turns it into Metal). Then all of it is packed into one file the app maps:

```sh
DM_GAME=<title ID> sh dm/android/tools/vk_shaders.sh "$B/shader/pkg" "$B/shader/cache" "$B/shader/vk"
CHAIN=$(python3 dm/gfx/tools/dmsh_chain.py)
python3 dm/android/tools/spv_pak.py "$B/shader/vk/b-$CHAIN" "$B/shader/shaders.pak"
```

The app fixes shaders at load time for strict mobile drivers (unused 64-bit and interlock declarations removed, outputs started at the Switch's defaults, unwritten
inputs and integer render targets handled). See the "Adreno hardening" notes in `dm/android/README.md`.

## 4. Gather the game files

One folder with everything the app needs. On the same disk, `cp -c` makes copies that take no space:

```sh
F="<NSP folder>/<Game>-Android/files"; mkdir -p "$F"
cp -cR "$B/extract/$V/RomFS" "$F/RomFS"
cp -cR "$B/patched_android" "$F/patched"
cp -c  "$B/shader/shaders.pak" "$F/"
cp -c  dm/games/<title ID>/shader_map.tsv "$F/"
cp -cR "$B/ipa/build/ios/pkg/dd/Build/Products/Release-iphoneos/DmIOS.app/dm/font" "$F/font"
[ -d "$B/aoc" ] && cp -cR "$B/aoc" "$F/aoc"          # DLC you own
```

Optional, and worth it: a **pipeline list** (`vk_pipelines_seed.bin`) so the phone compiles the game's pipelines at start instead of in the middle of play.
Record it by playing the game on the Mac with the same Vulkan renderer (`dm/android/mac_vk_build.sh <title ID>`, then run it with `DM_VK_PSO_RECORD=<file>`; see
`dm/android/README.md`), and copy the file to `$F/vk_pipelines_seed.bin`. Without it the phone records the pipelines while you play and builds them ahead from the
next start on.

## 5. Build the app

One command builds the native library, the app (icon from the game, touch controls, sound, controllers, the keyboard for names), compresses the game files, splits them
into APKs under 4 GB, aligns everything to 16 KB pages, signs it and packs the `.apks`:

```sh
sh dm/android/app/build_app.sh <title ID> "$B" "<Game name>" com.nxnative.<game> "<NSP folder>/<Game>-Android/<Game>.apks" "$F"
```

The signing key is a debug key made on the first build (`dm/android/probe/out/debug.keystore`, not committed). Keep it: an update must be signed with the same key.

## 6. Install on the phone

1. Copy the `.apks` to the phone (USB; it is several GB).
2. Install **SAI (Split APKs Installer)** from the Play Store, open it, *Install APKs*, choose the file, and allow installing from unknown apps when asked.
3. Start the game. The first start compiles the game's pipelines in the background (once; they are kept).

Or with a cable and `adb`: `adb install-multiple base.apk split_data*.apk` from `dm/android/out/app-<package>/bundle/`.

## Playing

- **Touch controls**: a floating stick on the left, A B X Y bottom-right, L / R and − / + at the top. Drag on the right side to turn the camera. Every press gives a haptic
  click. They hide while a controller is used and come back on the next touch.
- **Controllers**: Bluetooth / USB pads by button position; a pair of Joy-Cons plays as one controller.
- **Text**: names and nicknames open a text box with the phone's keyboard (numbers get the number pad).
- **Saves** are in `Android/data/<package>/files/save` and survive updates. Uninstalling deletes them.
- **Back** acts as the B button. It never closes the game.

## When something goes wrong

The app keeps its log where you can reach it: **Download/`<Game>-log-previous.txt`** after a crash (open the app once more first), and the live
**Download/`<Game>-log.txt`**. It records memory and the phone's heat every 30 seconds. Developer notes on what each line means are in `dm/android/README.md`.

## Testing without a phone

- **The game logic, files, saves and sound** run in Android Studio's emulator (arm64 system image) with the GPU stubbed out:
  `make -f dm/android/Makefile.host GAME=<title ID> GFX=null lib`.
- **The real app with graphics** runs in the emulator too (`-gpu guest`), through the copy-based memory route (slow, a few frames a second: for checking, not playing).
- **The Vulkan renderer** runs on the Mac through MoltenVK: `dm/android/mac_vk_build.sh <title ID>`. Use it for screenshots, the validation layer
  (`DM_VK_VALIDATION=1`) and memory checks. Never run it beside the emulator on a 16 GB Mac.
