# Additional terms (GNU GPL version 3, section 7)

These terms add to the GNU General Public License, version 3 or later ([LICENSE](LICENSE)), under its section 7, paragraphs (c) and (e).
They apply to the material in this repository that is copyrighted by the Arceus Native maintainer (simipoB), as published from
2026-11-25 on, in every file of this repository unless that file says otherwise. They do not apply to:

- the code from d2r-mobile / nxnative, or any other code released under the MIT License ([LICENSE-MIT](LICENSE-MIT));
- third-party code (for example under `dm/third_party`, or the GPL-3.0 frame generation code after Eden), which keeps its own license;
- contributions from other people, unless they agree to these terms.

## 1. Marking modified versions (section 7 (c))

If you distribute a modified version of Arceus Native, or an app built from changed Arceus Native source code:

1. Do not present it as the official Arceus Native build. In the app, in its release notes and on its download page, do not say or
   suggest that it is official, that it was signed with the project's release key (the one that also signs Galar Native), or that it was published by the Arceus Native
   project, unless that is true.
2. Mark it as modified in a way people can see. Its About screen, or the closest equivalent, must say that it is an unofficial or
   modified build and name who modified it. The app already shows such a label for any build not signed with the
   project's release key (`BuildSigner`). You may keep that label or replace it with your own clear one, but you may not remove it with
   nothing in its place, or make it say "official".
3. Do not make the app's logs or debug reports say that it is an official build (for example `DM_BUILD_SIGNER=official`) when it
   is not one. Bug reports from modified builds must not be mistaken for reports from official builds.

This does not stop you from changing the code, sharing your changes, or publishing your own builds under the GPL. It only asks that
they are not passed off as the original.

## 2. Names (section 7 (e))

The GPL does not give you any rights to the name "Arceus Native" (or "Galar Native", the engine it shares) or its logo. You may say truthfully that your work is based on
Arceus Native or Galar Native. Do not use either name in a way that makes your build look like the official app.

## Where the official builds are

Official releases are published only at https://github.com/simipoB/Arceus-Native-Android/releases and are signed with the project's
release key.
