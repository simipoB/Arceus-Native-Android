# Arceus Native

**Pokémon Legends: Arceus (Switch, update 1.1.1) running natively on Android.** Not emulation: the game's own ARM64 code runs on the phone, and its graphics (NVN) run on Vulkan. Arceus Native is built on the [Galar Native](https://github.com/simipoB/Galar-Native-Android) engine.

> [!WARNING]
> **Early release.** Tested on a Snapdragon 8 Elite (Adreno 830) so far. Reports with logs help most ([CONTRIBUTING.md](CONTRIBUTING.md)).

> [!NOTE]
> **No game files, code, shaders or keys are included.** You need your own copy of the game.

## Install

1. Install `ArceusNative.apk` from [Releases](https://github.com/simipoB/Arceus-Native-Android/releases) (or [build it](docs/BUILDING.md)).
2. Set up the game, one of two ways:
   - **From your own dump:** choose your game files (`.nsp` / `.xci`, base game and update 1.1.1) and your console's `prod.keys`. Keys stay on the phone.
   - **From the older Arceus app:** if the earlier Arceus port (`com.nxnative.arceus`) is installed, tap **Set up from the installed app**. No dump or keys needed, and its save comes along.
3. When setup is done, the app opens straight into the game.

- Official releases are signed with `SHA-256: a5f49b44b6181758d82504f3beb0851c92093f454c792b49688728130673a070`. Any other build says "Unofficial build". An official update installs over the last one and keeps your setup and saves.

## Requirements

- Android 12+, arm64, Vulkan 1.1+.
- Free space for the game's files plus about 1 GB of translated shaders.
- 8 GB of RAM or more is best. On weaker phones, **Lower specs** turns on by itself.

## Compatibility

| GPU | Status |
| --- | --- |
| **Adreno 7xx–8xx** (Snapdragon 8 Gen 1 and later) | Plays at a locked 30 (tested on Adreno 830). |
| **Older Adreno** | Should run; Lower specs helps. Awaiting reports. |
| **Mali, Xclipse, PowerVR** | Untested. The engine's Mali and Xclipse fixes from Galar Native are in, including the fix for the crash a few frames in on Arm's driver (v0.21). Awaiting reports. |

## What's new

- **Mali crash fix (v0.21).** Arm's driver makes more screen buffers than the game asks for (14 for 8 on a Mali-G57). The game had room for 8, so on some Mali phones the driver closed it a few frames in. It now has room for 32 and uses the count the driver reports.
- **Multiply by zero as on the Switch (Mali test, v0.21).** On the Switch's GPU, 0 × infinity is 0. On phones it's NaN, which can show as white, black or missing surfaces. Turning on ☰ → White screen tests (Mali) → *Multiply by zero as on the Switch* rewrites each shader's SPIR-V as it loads so that every multiply follows the Switch's rule. Off by default and a little slower; tell us in an issue if it changes your screen.
- **CPU pinning (v0.22).** The game keeps its two busiest threads on the phone's fastest cores. When a hot phone slows those cores down, the threads are now never squeezed onto fewer cores than there are threads. A hot Galaxy Z Flip5 had put both on one slowed core.
- **Two speed tests, off by default (v0.22).** `DM_VK_PASS_DEFER=1` keeps the current render pass open when the game changes targets in a way that needs no new pass. `DM_VK_ASYNC_PSO=1` builds pipelines the shader lists didn't have in the background instead of pausing the frame. Put either in `env.txt` to try it, and tell us what changed.

## Playing

Arceus is played by touch the way the game invites: only the movement stick, **−**, **+** and **☰** (settings) are on screen, and the rest happens on the game itself.

- **In the world:** tap to interact (A), swipe up to throw. The **Poké Ball button** aims and throws, and the **lock-on button** sits above it.
- **In menus:** tap items directly. The game's own **B Back** prompt is the back button. On the map, drag to pan and pinch to zoom.
- **Controllers:** hide the touch controls automatically. Hold **−** and **+** together for 1 s for ☰, or **−** for 3 s to remap buttons.
- **Touch controls:** frosted grey so they sit on the picture, and they fade when untouched. Move them, or switch on L3 / R3, in ☰ → Controls.
- **Logs:** `Download/PokemonLegendsArceus-log.txt`.

## Settings (☰)

| Setting | Choices |
| --- | --- |
| Picture | 16:9, fill screen, 20:9 |
| Upscaling | Auto, Sharp, SGSR, Smooth |
| Frame generation | Off, On (experimental), Lossless Scaling (after **Import Lossless.dll**) |
| Graphics | Auto, High (1080p), Low (720p), Adaptive |
| No reflections | On (default): smooth ground lighting, no purple shadows. Off: the ground's full lighting detail |
| Lower specs | Auto, Off, On (shorter draw distance, lighter shadows), Lowest |
| Textures | Auto, Full, Half; texture format (compact on Mali / Xclipse / PowerVR) |
| Controls | Button layout, remapping, touch look and layout, rumble, vibrate on press |
| Performance overlay | fps, temperature, GPU, free memory, battery |
| Mods, saves | Import, switch, order and annotate mods; export / import saves |
| Language | English, Español, Português, Français, 日本語 (Auto: the phone's) |

**Auto** rates the phone's CPU, GPU and RAM and picks graphics, textures and filtering from that. The menu shows what it picked. When the phone runs hot, the 3D resolution drops to about 720p until it cools, so the frame rate holds.

## Frame generation

- **On:** for each frame the game draws at 30 fps, the GPU makes the picture halfway to the next one and shows it first, so the screen moves like 60. Slightly more input delay, and some smearing on very fast motion.
- **Lossless Scaling:** if you own Lossless Scaling, tap **Import Lossless.dll** and pick the `Lossless.dll` from your copy. The phone converts its frame-generation shaders once, and Lossless Scaling then appears as a choice. Nothing from Lossless Scaling is in the app.

## Mods

Import a mod `.zip` (`romfs/` file replacements, `exefs/*.pchtxt` code patches for 1.1.1) and switch mods on or off in ☰ → Mods. **Order and notes** sets the load order (higher wins) and warns when two mods replace the same files. If the game crashes with mods on, they're switched off on the next start. The **No reflections** fix is built in, so you don't need a mod for it.

## Known gaps

- Nintendo's online services.
- Faster rendering (experimental) can flicker.

## License

Arceus Native is free software under the **GNU General Public License, version 3 or later** ([LICENSE](LICENSE), [NOTICE](NOTICE)): you may use, share and change it, and anything you distribute that is based on it must be under the same license with its source available. The code that came from d2r-mobile / nxnative, and the engine's code from before the switch, also stay under their original [MIT notice](LICENSE-MIT). Lossless Scaling is not included: its frame generation uses the `Lossless.dll` from your own copy.

## Credits

- **[Galar Native](https://github.com/simipoB/Galar-Native-Android)**: the engine and app this is built on.
- **[d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile)** and **nxnative** (MIT): the engine Galar Native is built on.
- **Ryujinx** (MIT): shader translator. **[LibHac](https://github.com/Thealexbarney/LibHac)**: NSP / XCI reading.
- **Eden** (GPL-3.0): the Lossless Scaling frame-generation chain. **dxbc-spirv** (MIT): converting the imported shaders.
- **[libadrenotools](https://github.com/bylaws/libadrenotools)** (BSD-2-Clause) and **Mesa Turnip / PanVK**: custom drivers.
- **[Snapdragon Game Super Resolution](https://github.com/SnapdragonStudios/snapdragon-gsr)** (Qualcomm, BSD-3-Clause): the SGSR upscaler.
- **[Atmosphère](https://github.com/Atmosphere-NX/Atmosphere)**, **yuzu**: references.
- **SPIRV-Cross**, **CPython**, **capstone**, **LZ4**, **libvpx**, **stb_vorbis**, **Vulkan Headers**, **DejaVu Sans**, **native_app_glue**.

Pokémon and Pokémon Legends: Arceus are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. Not affiliated with or endorsed by them.
