// The in-game settings (the gear button of the touch controls, app/jni/touch_ui.c): frame rate, the operation mode (the game's render resolution), texture size and
// filtering (the phone's chip profile by default), anti-aliasing and sharpening. They are lines of env.txt, which the host reads once at start; lines of other switches in the file are kept. Apply writes the file and closes the game,
// so the player saves first; the new settings hold from the next start.
package com.nxnative;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.graphics.Typeface;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

public final class Settings {
    private static volatile boolean open;

    // env.txt as lines: KEY=VALUE, comments and anything else as they are
    private static ArrayList<String> read(String path) {
        ArrayList<String> l = new ArrayList<String>();
        try { BufferedReader r = new BufferedReader(new FileReader(path)); String s; while ((s = r.readLine()) != null) l.add(s); r.close(); } catch (Exception e) { }
        return l;
    }
    private static String get(ArrayList<String> l, String k) {
        for (String s : l) if (s.startsWith(k + "=")) return s.substring(k.length() + 1).trim();
        return null;
    }
    private static void put(ArrayList<String> l, String k, String v) {   // v null: the line goes (the game profile's default)
        for (int i = l.size() - 1; i >= 0; i--) if (l.get(i).startsWith(k + "=")) l.remove(i);
        if (v != null) l.add(k + "=" + v);
    }
    private static TextView heading(Activity a, LinearLayout p, String t) {
        TextView v = new TextView(a); v.setText(t); v.setTypeface(Typeface.DEFAULT_BOLD); v.setPadding(0, 28, 0, 6); p.addView(v); return v;
    }
    private static RadioGroup choice(Activity a, LinearLayout p, String title, String[] opts, int sel) {
        heading(a, p, title);
        RadioGroup g = new RadioGroup(a);
        for (int i = 0; i < opts.length; i++) { RadioButton b = new RadioButton(a); b.setId(1000 + i); b.setText(opts[i]); g.addView(b); }
        g.check(1000 + sel); p.addView(g); return g;
    }
    private static Switch toggle(Activity a, LinearLayout p, String t, boolean on) {
        Switch s = new Switch(a); s.setText(t); s.setChecked(on); s.setPadding(0, 18, 0, 6); p.addView(s); return s;
    }

    public static void show(final Activity a, final String path) {
        if (open) return;
        open = true;
        a.runOnUiThread(new Runnable() {
            public void run() {
                final ArrayList<String> env = read(path);
                LinearLayout l = new LinearLayout(a); l.setOrientation(LinearLayout.VERTICAL); l.setPadding(64, 8, 64, 8);
                final RadioGroup fps = choice(a, l, "Frame rate", new String[] { "30 FPS (as on the Switch: the right game speed)", "60 FPS (test only: Legends: Arceus counts frames, so the game runs too fast)" },
                                              "display".equals(get(env, "DM_PRESENT_INTERVAL")) ? 1 : 0);
                final RadioGroup mode = choice(a, l, "Resolution", new String[] { "Handheld: the game's lower resolution (faster)", "Docked: the game's full resolution (sharper, slower)" },
                                               "1".equals(get(env, "DM_OPMODE")) ? 1 : 0);
                // lower-end phones: the POTATO mod's lighter world (android_main.c lowspec), off by default
                final String[] ls = { null, "1", "2" }; String lcur = get(env, "DM_LOWSPEC");
                final RadioGroup low = choice(a, l, "Lower specs", new String[] { "Off: the world as the game draws it",
                                              "On: shorter draw distance for Pokémon, people and trees, lighter shadows (for slower phones)",
                                              "Lowest: shorter draw distance and no shadows (for the slowest phones)" }, "2".equals(lcur) ? 2 : "1".equals(lcur) ? 1 : 0);
                // textures and filtering: "Recommended" leaves them to the phone's chip profile (android_main.c device_profile); a choice here wins over it
                final String[] tq = { null, "full", "half" };
                String tcur = get(env, "DM_VK_TEX_QUALITY");
                final RadioGroup tex = choice(a, l, "Textures", new String[] { "Recommended for this phone", "Full: the game's textures as they are",
                                              "Half: large textures at half size (less memory and GPU work, a little softer)" }, "half".equals(tcur) ? 2 : "full".equals(tcur) ? 1 : 0);
                final String[] an = { null, "2", "4", "8", "16" };
                String acur = get(env, "DM_VK_MAX_ANISO"); int asel = 0; for (int i = 1; i < an.length; i++) if (an[i].equals(acur)) asel = i;
                final RadioGroup aniso = choice(a, l, "Texture filtering", new String[] { "Recommended for this phone", "2x (fastest)", "4x", "8x", "16x (sharpest at a distance)" }, asel);
                heading(a, l, "Picture");
                final Switch fxaa = toggle(a, l, "Anti-aliasing (FXAA): smoother edges in the 3D scene, menus and text stay sharp", get(env, "DM_VK_FXAA") != null && !"0".equals(get(env, "DM_VK_FXAA")));
                final Switch sharp = toggle(a, l, "Sharpening (AMD CAS) when the picture is scaled to the screen", "1".equals(get(env, "DM_VK_SHARPEN")));
                CompoundButton.OnCheckedChangeListener one = new CompoundButton.OnCheckedChangeListener() {   // one of the two at a time (the same pass draws either)
                    public void onCheckedChanged(CompoundButton b, boolean on) { if (on) (b == fxaa ? sharp : fxaa).setChecked(false); }
                };
                fxaa.setOnCheckedChangeListener(one); sharp.setOnCheckedChangeListener(one);
                TextView note = new TextView(a); note.setPadding(0, 32, 0, 0);
                note.setText("Apply closes the game: save first. The settings hold from the next start.");
                l.addView(note);
                ScrollView sv = new ScrollView(a); sv.addView(l);
                AlertDialog d = new AlertDialog.Builder(a).setTitle("Settings").setView(sv)
                    .setPositiveButton("Apply and close", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface di, int w) {
                            put(env, "DM_PRESENT_INTERVAL", fps.getCheckedRadioButtonId() == 1001 ? "display" : null);
                            put(env, "DM_OPMODE", mode.getCheckedRadioButtonId() == 1001 ? "1" : null);
                            put(env, "DM_LOWSPEC", ls[low.getCheckedRadioButtonId() - 1000]);
                            put(env, "DM_VK_FXAA", fxaa.isChecked() ? "2" : null);   // 2: in the game's frame, before its interface (nvn_vk.c fx_scene); 1 the whole picture at present
                            put(env, "DM_VK_SHARPEN", sharp.isChecked() ? "1" : null);
                            put(env, "DM_VK_TEX_QUALITY", tq[tex.getCheckedRadioButtonId() - 1000]);
                            put(env, "DM_VK_MAX_ANISO", an[aniso.getCheckedRadioButtonId() - 1000]);
                            try { FileWriter f = new FileWriter(path); for (String s : env) f.write(s + "\n"); f.close(); } catch (Exception e) { }
                            a.finishAndRemoveTask();
                            android.os.Process.killProcess(android.os.Process.myPid());
                        }
                    })
                    .setNegativeButton("Cancel", null)
                    .setOnDismissListener(new DialogInterface.OnDismissListener() { public void onDismiss(DialogInterface di) { open = false; } })
                    .create();
                d.show();
            }
        });
    }
}
