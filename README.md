# Arceus Native

**Pokémon Legends: Arceus (Switch, update 1.1.1) running natively on Android.** Not emulation: the game's own ARM64 code runs on the phone, and its graphics (NVN) run on Vulkan. Arceus Native is built on the [Galar Native](https://github.com/simipoB/Galar-Native-Android) engine.

> [!WARNING]
> **Early release. v0.96 is a pre-release**, so far played only briefly on one test phone (Snapdragon 8 Elite, Adreno 830). Reports with logs help most ([CONTRIBUTING.md](CONTRIBUTING.md)).

> [!NOTE]
> **No game files, code, shaders or keys are included.** You need the ExeFS and RomFS folders extracted from your own copy of the game, with update 1.1.1 applied.

> [!IMPORTANT]
> **AI disclaimer.** Much of Arceus Native's code, including the engine, the graphics fixes and the reverse engineering of the game, was written with the help of AI coding tools. It's tested on real phones before each release, but it's an unofficial fan project and may have bugs that hurt performance, show graphics glitches or crash the game. Back up your saves, and report problems with a log ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Install

1. Install `ArceusNative.apk` from [Releases](https://github.com/simipoB/Arceus-Native-Android/releases) (or [build it](docs/BUILDING.md)).
2. Set up the game, one of two ways:
   - **From your own dump:** choose the **ExeFS** and **RomFS** folders you extracted from your game with **update 1.1.1 applied** (the folder holding both, or each in turn). Setup copies them into the app; with too little room it offers to move them instead. No keys and no `.nsp` / `.xci` any more: v0.98 deletes keys an older version stored.
   - **From the older Arceus app:** if the earlier Arceus port is installed, tap **Set up from the installed app**. No dump or keys needed, and its save comes along.
3. When setup is done, the app opens straight into the game.

- Official releases are signed with `SHA-256: a5f49b44b6181758d82504f3beb0851c92093f454c792b49688728130673a070`. Any other build says "Unofficial build". An official update installs over the last one and keeps your setup and saves.
- **Updates:** ☰ → About → Updates checks this repository's releases for a newer full release (never a pre-release) and installs it after you confirm. It only installs the official build signed with the key above. Turn off **Look for updates when the app starts** to stop the check.

## Requirements

- Android 12+, arm64, Vulkan 1.0 or later.
- Free space for the game's files plus about 1 GB of translated shaders.
- 8 GB of RAM or more is best. On weaker phones, **Lower specs** turns on by itself.

## Compatibility

| GPU | Status |
| --- | --- |
| **Adreno 7xx–8xx** (Snapdragon 8 Gen 1 and later) | Plays at a locked 30 (tested on Adreno 830). |
| **Older Adreno** | Should run; Lower specs helps. Exploding polygons still reported on some phones (#59). |
| **Mali** | Untested by us. With Arm's own driver, characters and Pokémon can still be invisible (#4, #10, #13, #47, #58); the **PanVK** driver (downloaded separately) draws them. |
| **Xclipse** | Untested by us. Shadow flicker: ☰ → System → Barriers between passes → Every pass (#20, #31). |
| **PowerVR** | Untested. Textures are now decoded fully (#32). Awaiting reports. |

## What's new

**v0.96** (pre-release). The version jumps from 0.45 to 0.96 because Arceus Native, Galar Native and Paldea Native now share one engine and one version number. Install over v0.44 or the v0.45 preview; no new setup is needed. Full notes: [releases page](https://github.com/simipoB/Arceus-Native-Android/releases).

- **A steady 30 fps.** When the phone can't keep up, the game steps to lighter settings until it can: a smaller 3D picture first, then the game's own handheld mode, and as a last step a shorter draw distance from the next start. It steps back up once things calm down.
- **Frame rate: Auto, 30, 60 or Dynamic**, changed while you play: 60 where the phone holds it, and the game speed stays right.
- **Shaders built before you play** (the places you have been), and a **heat guard** on every Graphics level.
- **Resolution: 720p, 1080p or 1440p** (1440p works again), a **Draw distance** slider and **Mode** (docked or handheld).
- **Frame generation** leaves the HUD out, follows the camera and depth, shows the real frame at camera cuts and menus, and Lossless Scaling goes 2x to 4x.
- **Updates from inside the app** (☰ → About → Updates), **Quit the game**, and a performance report for slow scenes (☰ → System → Bug reports).
- **More phones and fewer crashes:** Vulkan 1.0 drivers, older Snapdragon drivers, PowerVR textures, older ARMv8.0 CPUs, Chinese text, Mali phones whose driver refuses the screen setup, the camp loading crash on Xclipse 920 (#3). The game's clock now changes days at your local midnight.

**Earlier:** setup fixes and a cooler setup option (v0.44), crash fixes (v0.42), Widescreen with the HUD kept 16:9 (v0.40), the Mali crash fix and the Mali test switches (v0.21), and faster pipelines on Adreno (v0.23). The [releases page](https://github.com/simipoB/Arceus-Native-Android/releases) has each version's notes.

## Playing

Arceus is played by touch the way the game invites: only the movement stick, **−**, **+** and **☰** (settings) are on screen, and the rest happens on the game itself.

- **In the world:** tap to interact (A), swipe up to throw. The **Poké Ball button** aims and throws, and the **lock-on button** sits above it. Push the stick past its ring to dash; swipe **+** left or right to change your ride Pokémon.
- **In menus:** tap items directly. The game's own **B Back** prompt is the back button. On the map, drag to pan and pinch to zoom.
- **Controllers:** hide the touch controls automatically. Hold **−** and **+** together for 1 s for ☰, or **−** for 3 s to remap buttons.
- **Touch controls:** frosted grey so they sit on the picture, and they fade when untouched. Move them, or switch on L3 / R3, in ☰ → Controls.
- **Logs:** `Documents/NXNative/Logs/PokemonLegendsArceus-log.txt` (the run before: `…-log-previous.txt`). If that folder cannot be written, they go to `Download/` as before.

## Settings (☰)

| Setting | Choices |
| --- | --- |
| Picture | 16:9, fill screen, widescreen (the HUD stays 16:9) |
| Resolution | 720p, 1080p, 1440p (from the next start) |
| 3D render scale | Auto, or 50–100 % (changes while you play) |
| Upscaling | Auto, Sharp, SGSR, Smooth |
| Frame rate | Auto, 30, 60, Dynamic (changes while you play) |
| Frame generation | Off, On, Lossless Scaling (after **Import Lossless.dll**) |
| Graphics | Auto, High, Low, Adaptive |
| Mode | Auto, Quality (docked), Performance (handheld) |
| Draw distance | 50–200 % (from the next start) |
| No reflections | On (default): smooth ground lighting, no purple shadows. Off: the ground's full lighting detail |
| Lower specs | Auto, Off, On (shorter draw distance, lighter shadows), Lowest |
| Textures | Auto, Full, Half |
| Shaders at start | Build before play (default), or While playing |
| Controls | Button layout, remapping, touch look and layout, edge-run, ride button, rumble, vibrate on press |
| Performance overlay | fps, temperature, GPU, free memory, battery |
| Mods, saves | Import, switch, order and annotate mods; export / import saves |
| System | Quit the game; Bug reports (record a slow scene's frame for an issue) |
| About | Version, official / unofficial build, Updates |
| Language | English, Español, Português, Français, 日本語 (Auto: the phone's) |

**Auto** rates the phone's CPU, GPU and RAM and picks graphics, textures and filtering from that; while you play it keeps the frame rate steady and the phone cool. The menu shows what it picked.

## Frame generation

Off unless you turn it on (☰ → Display).

- **On:** for each frame the game draws at 30 fps, the GPU makes the picture halfway to the next one and shows it first, so the screen moves like 60. The HUD is left out and put back on top. Slightly more input delay, and some smearing on very fast motion. When the phone struggles, the 3D picture gives way first; frame generation pauses only when the game itself drops under its frame rate.
- **Lossless Scaling:** if you own Lossless Scaling, tap **Import Lossless.dll** and pick the `Lossless.dll` from your copy. The phone converts its frame-generation shaders once, and Lossless Scaling then appears as a choice: 2x to 4x, and Auto picks as many as your screen's refresh rate allows. On GPUs that can't run its shaders (for example Mali-G52 / G57 / G68 / G77 / G78 / G310 and some PowerVR), the app's own frame generation is used instead. Nothing from Lossless Scaling is in the app.

## Mods

Import a mod `.zip` (`romfs/` file replacements, `exefs/*.pchtxt` code patches for 1.1.1) and switch mods on or off in ☰ → Mods. **Order and notes** sets the load order (higher wins) and warns when two mods replace the same files. If the game crashes with mods on, they're switched off on the next start. The **No reflections** fix is built in, so you don't need a mod for it.

## Known gaps

- Nintendo's online services.
- Faster rendering (experimental) can flicker.
- Mali with Arm's own driver: characters and Pokémon can be invisible (use PanVK).
- Widescreen: objects can pop in at the left and right edges.
- Exploding polygons on some Snapdragon phones (#59); shadow flicker on Xclipse (see [Compatibility](#compatibility)).
- Occasional short hitches during play.

## License

Arceus Native is free software under the **GNU General Public License, version 3 or later** ([LICENSE](LICENSE), [NOTICE](NOTICE)): you may use, share and change it, and anything you distribute that is based on it must be under the same license with its source available. The code that came from d2r-mobile / nxnative, and the engine's code from before the switch, also stay under their original [MIT notice](LICENSE-MIT). [Additional terms](ADDITIONAL-TERMS.md) (GPL section 7) apply to the maintainer's code from 2026-11-25 on: a modified build must say it is unofficial and must not claim to be the official build or to be signed with the project's release key. Official builds come only from this repository's [releases](https://github.com/simipoB/Arceus-Native-Android/releases). Lossless Scaling is not included: its frame generation uses the `Lossless.dll` from your own copy.

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
