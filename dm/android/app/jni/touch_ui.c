// The on-screen controls: what they look like (front_overlay, drawn by the Vulkan replay over the game, nvn_ovl.h), what touching them does (touch_ui_event), and the
// haptic click of each press (Android's Vibrator through JNI, on a thread of its own so input never waits on it). Proportions and look after LAS Recomp's Android port
// (runtime/ui/touch_controls.mm), in units of the screen height, landscape: glossy bubbles (shadow, dark rim, body, glow, a gloss spot, a bright rim), all the same
// size as on the Switch; a floating stick (a compass ring that rests bottom-left and comes to where the thumb lands in the lower-left part); A B X Y in the Switch's diamond bottom-right;
// L / R round in the top corners; − and + at the top centre; a right stick below-left of the diamond, as on a right Joy-Con (the camera, pushed like the real stick).
// Dragging anywhere else on the right turns the camera too, following the finger:
// the stick is the finger's speed, back to rest as soon as the finger stops. A quick tap is held at least 80 ms (the game reads its input once a frame, every 33 ms);
// touch areas are 1.3 times the drawing; a thumb sliding across the face buttons presses the one under it. Hidden while a controller is used, back on the next touch.
// The layout and drawing parts also build into the Mac Vulkan host (mac_vk_build.sh), whose DM_OVL_SHOT composes them over a game frame to check the look.
#include "dm.h"
#include "game_profile.h"   // DM_GAME_TOUCH_ZLZR (the profile key touch_zlzr)
#if defined(DM_GAME_TOUCH_MENU) && DM_GAME_TOUCH_MENU
#define GESTURES_ON 1   // the throw button and the touch gestures (touch_ui_event): the games with touch menus (Legends: Arceus)
#else
#define GESTURES_ON 0
#endif
#include "dm_input.h"
#include "nvn_ovl.h"
#ifdef __ANDROID__
#include <android/input.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <jni.h>
#endif
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { C_STICK, C_CAM, C_RSTICK, C_A, C_B, C_X, C_Y, C_L, C_R, C_ZL, C_ZR, C_PLUS, C_MINUS, C_UP, C_DOWN, C_LEFT, C_RIGHT, C_THROW, C_MENU, C_BACK, C_SEL, C_LOCK, C_L3, C_R3, NCTL };   // (C_MENU: the ☰ button, the in-game settings; C_BACK: the game's own Back; C_L3 / C_R3: pressing the sticks)
typedef struct { float cx, cy, r; int glyph, shown; uint64_t btn; } ctl;
static ctl g_c[NCTL];
static float g_w, g_h;
static struct { int32_t id; int target; float x0, y0, x, y, lx, ly; double lt, t0; int thr, lp, aim; } g_ptr[10];   // (aim: this finger landed while aiming: its lift throws)
static int g_aim;   // aiming after a tap on the throw button: ZR held until the throw   // (thr: this finger's swipe became a throw)   // (t0: when the finger landed)   // fingers down: id, what they hold, where, last move (camera)
static uint64_t g_btn; static int32_t g_lx, g_ly, g_rx, g_ry;
static double g_held_until[NCTL];                   // the 80 ms minimum press
static float g_cam_vx, g_cam_vy; static double g_cam_t;   // the camera finger's speed (pixels / s) and when it last moved
static float g_stick_x, g_stick_y, g_knob_x, g_knob_y; static int g_stick_on;
static float g_rknob_x, g_rknob_y; static int g_rstick_on;
static double g_inner_t;   // the last touch of the D-pad or the right stick   // the right stick: its base stays put, the knob follows the thumb
static int g_visible = 1;
static int g_edit;   // the layout editor is on (touch_ui_event, front_overlay)
static double g_touch_t; static double g_two_t, g_tap_mute;   // two-finger tap: when the second finger landed; single taps ignored until g_tap_mute
int touch_menu_active(void);   // touch_menu.c: one of the game's menus shows
int touch_menu_map(void);   // touch_menu.c: the map is open (a drag pans it, a pinch zooms)
int touch_menu_cutscene(void);   // touch_menu.c: no menu and no field HUD (cutscene, dialogue, loading): the controls go, a tap is A
int touch_menu_back_list(float w, float h, float r[4]);   // touch_menu.c: the open list's rectangle on screen (left, top, right, bottom), 0 = none
#if defined(DM_GAME_TOUCH_MENU) && DM_GAME_TOUCH_MENU
enum { GESTURES = 1 };   // tap = A in the world, swipe up = throw: only where menu taps work too (else a tap in a menu would press A on whatever is selected)
#else
enum { GESTURES = 0 };
#endif   // the last touch anywhere: every control fades 3 s after it (they cover the game's own menus and text)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

// ---------- the player's layout (the in-game editor): position, size and visibility of each control, kept in <files>/touch_layout.txt ----------
// One line per changed control: "<name> <x / width> <y / height> <radius / height> <shown>"; controls without a line keep the default layout
static const char *const CTL_NAME[NCTL] = { "stick", "camera", "rstick", "a", "b", "x", "y", "l", "r", "zl", "zr", "plus", "minus", "up", "down", "left", "right", "throw", "menu", "back", "sel", "lock", "l3", "r3" };
static struct { int set; float fx, fy, fr; int shown; } g_ov[NCTL];
static char g_lay_path[1100]; static int g_lay_loaded;
void touch_ui_set_dir(const char *dir) { snprintf(g_lay_path, sizeof g_lay_path, "%s/touch_layout.txt", dir); g_lay_loaded = 0; }
static void layout_load(void) {
    g_lay_loaded = 1; memset(g_ov, 0, sizeof g_ov); if (!g_lay_path[0]) return;
    FILE *f = fopen(g_lay_path, "r"); char ln[160], nm[32]; float fx, fy, fr; int sh;
    while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "%31s %f %f %f %d", nm, &fx, &fy, &fr, &sh) == 5)
        for (int c = 0; c < NCTL; c++) if (!strcmp(nm, CTL_NAME[c])) { g_ov[c].set = 1; g_ov[c].fx = fx; g_ov[c].fy = fy; g_ov[c].fr = fr; g_ov[c].shown = sh; }
    if (f) fclose(f);
}
static void layout_save(void) {
    if (!g_lay_path[0]) return; FILE *f = fopen(g_lay_path, "w"); if (!f) return;
    for (int c = 0; c < NCTL; c++) if (g_ov[c].set) fprintf(f, "%s %.4f %.4f %.4f %d\n", CTL_NAME[c], g_ov[c].fx, g_ov[c].fy, g_ov[c].fr, g_ov[c].shown);
    fclose(f);
}
static void layout_apply(float w, float h) {
    if (!g_lay_loaded) layout_load();
    for (int c = 0; c < NCTL; c++) if (g_ov[c].set && c != C_CAM) { g_c[c].cx = g_ov[c].fx * w; g_c[c].cy = g_ov[c].fy * h; g_c[c].r = g_ov[c].fr * h; g_c[c].shown = g_ov[c].shown; }
}
static void sel_to_win(float lx, float ly, float *x, float *y);
// L3 / R3 on screen: an option (DM_TOUCH_L3R3=on, Controls → Touch controls), off by default: Sword and Shield hardly use them. -1: not read yet
static int g_l3r3 = -1;
static int l3r3_on(void) { if (g_l3r3 < 0) { const char *e = getenv("DM_TOUCH_L3R3"); g_l3r3 = e && !strcmp(e, "on"); } return g_l3r3; }
static void layout(float w, float h) {
    static int init; if (!init++) for (int i = 0; i < 10; i++) g_ptr[i].id = -1;   // no finger down
    if (w == g_w && h == g_h) return;
    g_w = w; g_h = h;
    // the outer controls keep about 0.2 of the height (~290 px on a 1440 px phone) clear of the screen's long edges, where the system's navigation buttons and edge
    // swipes are: with the A button nearer, a thumb on it opened recents
    float s = h, br = 0.074f * s, gap = 0.135f * s, fx = w - 0.40f * s, fy = h - 0.30f * s, sr = 0.064f * s, mr = 0.040f * s;
    g_c[C_A] = (ctl){ fx + gap, fy, br, OVL_G_A, 1, DM_BTN_A };   // the Switch's diamond, all four the same
    g_c[C_B] = (ctl){ fx, fy + gap, br, OVL_G_B, 1, DM_BTN_B };
    g_c[C_X] = (ctl){ fx, fy - gap, br, OVL_G_X, 1, DM_BTN_X };
    g_c[C_Y] = (ctl){ fx - gap, fy, br, OVL_G_Y, 1, DM_BTN_Y };
    g_c[C_L] = (ctl){ 0.25f * s, 0.30f * s, sr, OVL_G_L, 1, DM_BTN_L };
    g_c[C_R] = (ctl){ w - 0.25f * s, 0.30f * s, sr, OVL_G_R, 1, DM_BTN_R };
    // ZL / ZR above L / R where the game uses them (profile key touch_zlzr: Legends: Arceus aims with ZL and throws with ZR); hidden otherwise (Shield)
#if defined(DM_GAME_TOUCH_ZLZR) && DM_GAME_TOUCH_ZLZR
    g_c[C_ZL] = (ctl){ 0.25f * s, 0.11f * s, sr, OVL_G_ZL, 1, DM_BTN_ZL };
    g_c[C_ZR] = (ctl){ w - 0.25f * s, 0.11f * s, sr, OVL_G_ZR, 1, DM_BTN_ZR };
#else
    g_c[C_ZL] = (ctl){ 0.26f * s, 0.10f * s, sr * 0.8f, OVL_G_ZL, 0, DM_BTN_ZL };
    g_c[C_ZR] = (ctl){ w - 0.26f * s, 0.10f * s, sr * 0.8f, OVL_G_ZR, 0, DM_BTN_ZR };
#endif
    g_c[C_MINUS] = (ctl){ w * 0.5f - 0.13f * s, 0.06f * s, mr, OVL_G_MINUS, 1, DM_BTN_MINUS };
    g_c[C_PLUS] = (ctl){ w * 0.5f + 0.13f * s, 0.06f * s, mr, OVL_G_PLUS, 1, DM_BTN_PLUS };
    g_c[C_STICK] = (ctl){ 0.36f * s, h - 0.36f * s, 0.14f * s, OVL_G_NONE, 1, 0 };   // (raised: its ring sat on the bottom edge)
    {   // the D-pad below-right of the left stick, as on a left Joy-Con (bag pockets, menus): four small buttons with a dot towards their direction
        float dx = 0.80f * s, dy = h - 0.19f * s, dg = 0.085f * s, dr = 0.050f * s;
        g_c[C_UP] = (ctl){ dx, dy - dg, dr, OVL_G_NONE, 1, DM_BTN_UP }; g_c[C_DOWN] = (ctl){ dx, dy + dg, dr, OVL_G_NONE, 1, DM_BTN_DOWN };
        g_c[C_LEFT] = (ctl){ dx - dg, dy, dr, OVL_G_NONE, 1, DM_BTN_LEFT }; g_c[C_RIGHT] = (ctl){ dx + dg, dy, dr, OVL_G_NONE, 1, DM_BTN_RIGHT };
    }
    // THROW (Legends: Arceus): a Poké Ball where the right thumb rests, left of A B X Y: a tap starts aiming (ZR held, g_aim) until the aiming finger lifts or the
    // ball is tapped again (the throw); held = ZR too (the finger's drag turns the aim), released = throw
    // (not shown unless DM_TOUCH_BALL=1: the player throws with ZR, as on the console)
    { static int ball = -1; if (ball < 0) { const char *e = getenv("DM_TOUCH_BALL"); ball = e && atoi(e) > 0; }
    g_c[C_THROW] = (ctl){ w - 0.68f * s, h - 0.42f * s, 0.062f * s, OVL_G_NONE, GESTURES_ON && ball, DM_BTN_ZR };   // the size of Naruto Storm's shuriken button; touched over 1.3 times that (hit)
    g_c[C_LOCK] = (ctl){ w - 0.68f * s, h - 0.58f * s, 0.040f * s, OVL_G_NONE, GESTURES_ON && ball, 0 }; }  // lock-on (g_lockon), above the ball
    g_c[C_RSTICK] = (ctl){ w - 0.80f * s, h - 0.19f * s, 0.11f * s, OVL_G_NONE, 1, 0 };
    g_c[C_MENU] = (ctl){ w * 0.5f - 0.24f * s, 0.06f * s, mr, OVL_G_NONE, 1, 0 };   // ☰ just left of −, at the top centre: the settings (front_open_menu); no Switch button
    // L3 / R3 (pressing a stick in, #108): small buttons just above each stick, outside its touch area (1.3 x its radius) and clear of the D-pad and A B X Y on
    //   16:9 to 20:9 screens; hide or move them in the layout editor like any other
    g_c[C_L3] = (ctl){ g_c[C_STICK].cx, g_c[C_STICK].cy - 0.22f * s, 0.05f * s, OVL_G_L3, 1, DM_BTN_LS };
    g_c[C_R3] = (ctl){ g_c[C_RSTICK].cx, g_c[C_RSTICK].cy - 0.21f * s, 0.05f * s, OVL_G_R3, 1, DM_BTN_RS };
    g_c[C_BACK] = (ctl){ 0, 0, 0.024f * s, OVL_G_B, 0, DM_BTN_B };   // drawn as the game's own B prompt (small); touched over 2.5 times that (hit_slack)
    g_c[C_SEL] = (ctl){ 0, 0, 0, OVL_G_NONE, 0, 0 };
    // touch-first games (Legends: Arceus): only the movement stick, − / + and ☰ by default (touch_ui_event's gestures, touch_menu.c); A B X Y above the selector
    if (GESTURES_ON) { float sx, sy; sel_to_win(1695, 780, &sx, &sy); br = 0.062f * s; gap = 0.118f * s; fx = sx; fy = sy - gap - br - 0.01f * s;
        g_c[C_A] = (ctl){ fx + gap, fy, br, OVL_G_A, 1, DM_BTN_A }; g_c[C_B] = (ctl){ fx, fy + gap, br, OVL_G_B, 1, DM_BTN_B };
        g_c[C_X] = (ctl){ fx, fy - gap, br, OVL_G_X, 1, DM_BTN_X }; g_c[C_Y] = (ctl){ fx - gap, fy, br, OVL_G_Y, 1, DM_BTN_Y }; }
    if (!g_rstick_on) { g_rknob_x = g_c[C_RSTICK].cx; g_rknob_y = g_c[C_RSTICK].cy; }
    layout_apply(w, h);
    if (!l3r3_on()) g_c[C_L3].shown = g_c[C_R3].shown = 0;   // (an option, off by default: the layout editor's own choice only counts while it is on)
    if (!g_stick_on) { g_stick_x = g_knob_x = g_c[C_STICK].cx; g_stick_y = g_knob_y = g_c[C_STICK].cy; }
}
// The back button: just under the open list's bottom-left corner (Arceus's four battle moves, a bag pocket), or left of its bottom row when the list reaches the
// bottom edge; shown only while the game has a list open (touch_menu.c), so it never sits over the field. Called with g_mu held, before hit tests and before drawing
static void back_place_locked(void) {
    float r[4], rad = g_c[C_BACK].r, gap = 0.018f * g_h;
    static int own = -1; if (own < 0) own = getenv("DM_TOUCH_BACK_BUTTON") == NULL;   // by default the game's own Back prompt is the button (menu_back); this one only on request
    if (own || !touch_menu_back_list(g_w, g_h, r)) { g_c[C_BACK].shown = 0; return; }
    float x = r[0] + rad, y = r[3] + gap + rad;
    if (y + rad > g_h - 0.02f * g_h) { x = r[0] - gap - rad; y = r[3] - rad; }   // no room below: beside the last item
    g_c[C_BACK].cx = clampf(x, rad, g_w - rad); g_c[C_BACK].cy = clampf(y, rad, g_h - rad); g_c[C_BACK].shown = 1;
}
static int ctl_off(int c) { return (c == C_L3 || c == C_R3) && !l3r3_on(); }   // left out entirely, the layout editor too
static int hit(int c, float x, float y, float slack) {
    const ctl *k = &g_c[c]; float dx = x - k->cx, dy = y - k->cy, r = k->r * slack;
    return k->shown && dx * dx + dy * dy <= r * r;
}
// The game's item / Pokémon selector (Legends: Arceus field HUD, 1920 x 1080 layout, origin top left, measured from the game's picture): three bubbles in a row,
// the middle one the current item, the X / swap icon above it, the item name below. In window pixels through the same fit the presentation uses (16:9, centred)
enum { SEL_LEFT, SEL_MID, SEL_RIGHT, SEL_SWAP, SEL_ZONE };
// (the picture where the presentation puts it: nvn_ovl.h dm_picture_box, which follows the picture settings)
static void pic_box(float b[4]) { int32_t x[4]; dm_picture_box((int32_t)g_w, (int32_t)g_h, 1920, 1080, x); b[0] = (float)x[0]; b[1] = (float)x[1]; b[2] = x[2] / 1920.0f; b[3] = x[3] / 1080.0f; }
static void sel_to_win(float lx, float ly, float *x, float *y) { float b[4]; pic_box(b); *x = b[0] + lx * b[2]; *y = b[1] + ly * b[3]; }
static int sel_part(float x, float y) {   // which part of the selector a window point is on: SEL_* or -1 (outside its zone)
    float b[4]; pic_box(b); float lx = (x - b[0]) / b[2], ly = (y - b[1]) / b[3];
    if (lx < 1500 || lx > 1890 || ly < 790 || ly > 1040) return -1;
    if (ly < 860 && lx > 1620 && lx < 1730) return SEL_SWAP;
    return lx < 1632 ? SEL_LEFT : lx > 1750 ? SEL_RIGHT : SEL_MID;
}
// The game's own button hints, bottom left of the Legends: Arceus field HUD (same layout space): ▲ the Satchel, ▼ the hint below it, the map with − above them.
// A quick tap on one presses its button (touch_ui_event); they lie in the stick's area, so a drag from them still moves
static uint64_t hud_hint(float x, float y) {
    float b[4]; pic_box(b); float lx = (x - b[0]) / b[2], ly = (y - b[1]) / b[3];
    if (lx < 30 || lx > 180) return 0;
    if (ly >= 760 && ly < 870 && lx < 120) return DM_BTN_MINUS;
    if (ly >= 900 && ly < 975) return DM_BTN_UP;
    if (ly >= 975 && ly < 1055) return DM_BTN_DOWN;
    return 0;
}
// The game's own "Ⓑ Back" prompt, bottom right of its menus (Satchel, Pokédex, Communications; same layout space): a quick tap on it is B while a menu is open
static int menu_back(float x, float y) {
    float b[4]; pic_box(b); float lx = (x - b[0]) / b[2], ly = (y - b[1]) / b[3];
    return lx >= 1800 && ly >= 1010;
}
static int target_at(float x, float y) {
    if (GESTURES_ON && !touch_menu_active() && sel_part(x, y) >= 0) return C_SEL;   // on the game's selector: its own gestures (touch_ui_event); not over a menu, where its flick up (ZR) would switch the menu's tab
    int menu = GESTURES_ON && touch_menu_active();   // a menu open: every control works as in the field (the ball aside); a touch off them taps the menu's items
    if (hit(C_RSTICK, x, y, menu ? 1.0f : 1.3f)) return C_RSTICK;
    for (int c = C_A; c < NCTL; c++) if (!(menu && (c == C_THROW || c == C_LOCK)) && hit(c, x, y, c == C_BACK ? 2.5f : menu ? 1.0f : 1.3f)) return c;   // thumbs miss: touch areas 1.3 times the drawing (the small back button 2.5; in menus the drawing, so items beside them stay tappable)
    if (menu) return hit(C_STICK, x, y, 1.15f) ? C_STICK : C_CAM;   // a menu: the stick on its ring moves the game's cursor as on a Joy-Con; elsewhere items are tapped and lists dragged
    if (x < g_w * 0.42f && y > g_h * 0.30f) return C_STICK;
    return C_CAM;   // everything else (the right side, and the top of the left side, where menus put items: English on the language screen) taps / turns the camera like the right side
}
static int32_t ax(float v) { v = clampf(v, -1, 1); return (int32_t)(v * DM_STICK_MAX); }

#ifdef __ANDROID__
void touch_menu_tap(float x, float y, float w, float h);   // touch_menu.c
void front_open_menu(int what);   // android_main.c: the in-game settings (GameMenu.java), 0 = all, 1 = the controller buttons
void touch_menu_scroll(int right, int down);                // touch_menu.c: D-pad steps on the open menu
void touch_menu_back(void);                                  // touch_menu.c: B for the game's own Back prompt, after the taps waiting
void touch_menu_drag_rate(float r);                          // touch_menu.c: a scroll drag on the menu, -1..1 (up the list positive): rows stepped while it lasts
// ---------- haptics ----------
static JavaVM *g_vm; static jobject g_act; static pthread_cond_t g_hcv = PTHREAD_COND_INITIALIZER; static int g_hpend;
// rumble (hle_hid.c dm_rumble): the game's motor strength, handed to GameActivity.rumble on this thread when it changes, and again every 100 ms while it lasts
// (each hand-over is a short pulse, so a rumble the game stops without a zero value ends by itself)
static float g_ramp; static int g_rpend;
static void *haptic_main(void *p) {
    (void)p; JNIEnv *e; if ((*g_vm)->AttachCurrentThread(g_vm, &e, NULL)) return NULL;
    jclass ctx = (*e)->FindClass(e, "android/content/Context"); jmethodID gss = (*e)->GetMethodID(e, ctx, "getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;");
    jstring name = (*e)->NewStringUTF(e, "vibrator"); jobject vib = (*e)->CallObjectMethod(e, g_act, gss, name);
    jclass vc = (*e)->FindClass(e, "android/os/Vibrator"), ec = (*e)->FindClass(e, "android/os/VibrationEffect");
    jmethodID vibrate = vc ? (*e)->GetMethodID(e, vc, "vibrate", "(Landroid/os/VibrationEffect;)V") : NULL;
    jmethodID pre = ec ? (*e)->GetStaticMethodID(e, ec, "createPredefined", "(I)Landroid/os/VibrationEffect;") : NULL;
    if ((*e)->ExceptionCheck(e)) (*e)->ExceptionClear(e);
    jobject click = pre ? (*e)->CallStaticObjectMethod(e, ec, pre, 0 /* EFFECT_CLICK */) : NULL;
    if (click) click = (*e)->NewGlobalRef(e, click);
    if (!vib || !vibrate || !click) { __android_log_print(ANDROID_LOG_WARN, "nxnative", "haptics unavailable"); (*g_vm)->DetachCurrentThread(g_vm); return NULL; }
    vib = (*e)->NewGlobalRef(e, vib);
    jmethodID rumble = (*e)->GetMethodID(e, (*e)->GetObjectClass(e, g_act), "rumble", "(F)V"); if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionClear(e); rumble = NULL; }
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_hpend && !g_rpend) {
            if (g_ramp <= 0) { pthread_cond_wait(&g_hcv, &g_mu); continue; }
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += 100000000; if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            if (pthread_cond_timedwait(&g_hcv, &g_mu, &ts) == ETIMEDOUT) g_rpend = 1;
        }
        int tap = g_hpend, rum = g_rpend; float amp = g_ramp; g_hpend = g_rpend = 0; pthread_mutex_unlock(&g_mu);
        if (tap) { (*e)->CallVoidMethod(e, vib, vibrate, click); if ((*e)->ExceptionCheck(e)) (*e)->ExceptionClear(e); }
        if (rum && rumble) { (*e)->CallVoidMethod(e, g_act, rumble, (jfloat)amp); if ((*e)->ExceptionCheck(e)) (*e)->ExceptionClear(e); }
    }
    return NULL;
}
// Controllers: Android's input devices polled once a second (InputDevice.getDeviceIds); a gamepad or joystick that is not virtual and is external connecting hides
// the touch controls, its going shows them again
static void *pad_watch(void *p) {
    (void)p; JNIEnv *e; if ((*g_vm)->AttachCurrentThread(g_vm, &e, NULL)) return NULL;
    jclass dc = (*e)->FindClass(e, "android/view/InputDevice"); if (!dc) { (*e)->ExceptionClear(e); return NULL; }
    jmethodID ids = (*e)->GetStaticMethodID(e, dc, "getDeviceIds", "()[I"), get = (*e)->GetStaticMethodID(e, dc, "getDevice", "(I)Landroid/view/InputDevice;");
    jmethodID src = (*e)->GetMethodID(e, dc, "getSources", "()I"), virt = (*e)->GetMethodID(e, dc, "isVirtual", "()Z"), ext = (*e)->GetMethodID(e, dc, "isExternal", "()Z");
    if ((*e)->ExceptionCheck(e)) (*e)->ExceptionClear(e);
    if (!ids || !get || !src || !virt) return NULL;
    dc = (*e)->NewGlobalRef(e, dc);
    for (int was = -1;; sleep(1)) {
        int n = 0; jintArray a = (*e)->CallStaticObjectMethod(e, dc, ids); if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionClear(e); continue; }
        jsize na = a ? (*e)->GetArrayLength(e, a) : 0; jint *v = a ? (*e)->GetIntArrayElements(e, a, NULL) : NULL;
        for (jsize i = 0; v && i < na; i++) { jobject d = (*e)->CallStaticObjectMethod(e, dc, get, v[i]); if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionClear(e); continue; } if (!d) continue;
            jint so = (*e)->CallIntMethod(e, d, src); jboolean vi = (*e)->CallBooleanMethod(e, d, virt), ex = ext ? (*e)->CallBooleanMethod(e, d, ext) : JNI_TRUE;
            if ((*e)->ExceptionCheck(e)) (*e)->ExceptionClear(e);
            if (!vi && ex && ((so & 0x401) == 0x401 || (so & 0x01000010) == 0x01000010)) n++;   // SOURCE_GAMEPAD / SOURCE_JOYSTICK
            (*e)->DeleteLocalRef(e, d); }
        if (v) (*e)->ReleaseIntArrayElements(e, a, v, JNI_ABORT); if (a) (*e)->DeleteLocalRef(e, a);
        int now = n > 0;
        if (now != was) { pthread_mutex_lock(&g_mu); if (now) g_visible = 0; else if (was == 1) g_visible = 1; pthread_mutex_unlock(&g_mu);
            if (was >= 0 || now) fprintf(stderr, "[touch] %s\n", now ? "controller connected: touch controls hidden (a touch shows them)" : "controller gone: touch controls shown"); }
        was = now;
    }
    return NULL;
}
void touch_ui_init(ANativeActivity *a) {
    g_vm = a->vm; g_act = a->clazz;
    static int once; if (once++) return;   // a second GameActivity in the same process (the first handed over to the setup screen): one haptic thread
    pthread_t t; if (!pthread_create(&t, NULL, haptic_main, NULL)) pthread_detach(t);
    if (!pthread_create(&t, NULL, pad_watch, NULL)) pthread_detach(t);
}
// the click on a press: DM_TOUCH_HAPTIC on / off, or by default as DM_RUMBLE (Off silences it too: the player asked for no vibration, #114). -1: not read yet
static int g_haptic = -1, g_rumble_off = -1;
static int haptic_on(void) {
    if (g_haptic < 0) { const char *h = getenv("DM_TOUCH_HAPTIC"); g_haptic = h && !strcmp(h, "off") ? 0 : h && !strcmp(h, "on") ? 1 : 2; }
    if (g_rumble_off < 0) { const char *r = getenv("DM_RUMBLE"); g_rumble_off = r && !strcmp(r, "off"); }
    return g_haptic == 1 || (g_haptic == 2 && !g_rumble_off);
}
void touch_ui_haptic_setting(const char *key, const char *value) {   // (touch_ui_setting, under g_mu)
    if (!strcmp(key, "DM_TOUCH_HAPTIC")) g_haptic = value && !strcmp(value, "off") ? 0 : value && !strcmp(value, "on") ? 1 : 2;
    else if (!strcmp(key, "DM_RUMBLE")) g_rumble_off = value && !strcmp(value, "off");
}
static void click_locked(void) { if (!haptic_on()) return; g_hpend = 1; pthread_cond_signal(&g_hcv); }   // callers hold g_mu
void dm_rumble(float left, float right) {   // hle_hid.c: the game's two motors; one strength for the one motor Android drives
    float a = left > right ? left : right;
    pthread_mutex_lock(&g_mu);
    if ((a > 0) != (g_ramp > 0) || fabsf(a - g_ramp) > 0.04f) { g_ramp = a; g_rpend = 1; pthread_cond_signal(&g_hcv); }
    pthread_mutex_unlock(&g_mu);
}
#endif
#ifdef __ANDROID__

// ---------- touch ----------
// Aiming from the ball locks on too (ZL held with ZR, as both triggers on the Switch: the camera turns to the Pokémon ahead and the throw goes to it); a drag still moves the aim
// (the game moves the lock to the next Pokémon on a flick). DM_TOUCH_AIM_LOCK=0: ZR alone. While aiming the finger turns the camera at AIM_FINE of its usual speed
static int aim_lock(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_TOUCH_AIM_LOCK"); v = !(e && !strcmp(e, "0")); } return v; }
enum { AIM_FINE_PCT = 45 };
static double g_lock_until;   // after a throw from the ball the lock-on stays a quarter second, so the ball leaves while still locked
// The ball as Naruto Storm's shuriken button: a tap throws at once at the Pokémon ahead (ZL locks on, ZR a moment later presses and releases: the throw), no aiming;
// held past QUICK_S it aims (ZL + ZR held, the finger's drag turns the aim finely), and lifting it throws. The lock-on button keeps ZL held (the camera on the Pokémon,
// as Storm's fight camera) until tapped again; a tap on the ball then throws straight at it
#define QUICK_S 0.22
static double g_qt0 = -1;   // a quick throw's start (recompute_locked plays it out)
static int g_lockon;        // the lock-on button is on
static float g_pinch_d0;   // the map: the two fingers' distance at the last zoom step   // after a throw from the ball the lock-on stays a quarter second, so the ball leaves while still locked
static void recompute_locked(void) {
    uint64_t b = 0; float lx = 0, ly = 0, rx = 0, ry = 0; int stick = 0, rstick = 0; double t = now_s();
    int menu = GESTURES_ON && touch_menu_active();   // a menu open: fingers tap and swipe its lists (touch_ui_event), the stick and the camera stay still
    int map = menu && touch_menu_map();   // the map: a finger drags it (the left stick at the finger's speed, the content following the finger), two pinch it (ZL / ZR)
    for (int i = 0; i < 10; i++) {
        if (g_ptr[i].id < 0) continue;
        int c = g_ptr[i].target;
        if (c == C_STICK) { float r = g_c[C_STICK].r, dx = g_ptr[i].x - g_stick_x, dy = g_ptr[i].y - g_stick_y, d = sqrtf(dx * dx + dy * dy);
            if (d > r) { dx *= r / d; dy *= r / d; }
            float nx = dx / r, ny = -dy / r; if (nx * nx + ny * ny < 0.12f * 0.12f) nx = ny = 0;   // dead zone
            lx = nx; ly = ny; stick = 1; g_knob_x = g_stick_x + dx; g_knob_y = g_stick_y + dy; }
        else if (c == C_RSTICK) { const ctl *k = &g_c[C_RSTICK]; float r = k->r, dx = g_ptr[i].x - k->cx, dy = g_ptr[i].y - k->cy, d = sqrtf(dx * dx + dy * dy);
            if (d > r) { dx *= r / d; dy *= r / d; }
            float nx = dx / r, ny = -dy / r; if (nx * nx + ny * ny < 0.12f * 0.12f) nx = ny = 0;
            rx = nx; ry = ny; rstick = 1; g_rknob_x = k->cx + dx; g_rknob_y = k->cy + dy; }
        else if (c == C_THROW) { if (t - g_ptr[i].t0 >= QUICK_S) b |= DM_BTN_ZR | (aim_lock() ? DM_BTN_ZL : 0); }   // held: aiming (a tap is the quick throw, on release)
        else if (c >= C_A) b |= g_c[c].btn;
    }
    for (int c = C_A; c < NCTL; c++) if (c != C_THROW && t < g_held_until[c]) b |= g_c[c].btn;
    if (g_qt0 >= 0) { double q = t - g_qt0;   // the quick throw: ZL 0 – 0.5 s, ZR 0.12 – 0.26 s (pressed and let go: thrown at what ZL locked)
        if (q < 0.5) b |= DM_BTN_ZL; if (q >= 0.12 && q < 0.26) b |= DM_BTN_ZR; if (q >= 0.5) g_qt0 = -1; }
    if (g_lockon && (menu || (GESTURES_ON && touch_menu_cutscene()))) g_lockon = 0;   // a menu or a cutscene ends the lock
    if (g_lockon) b |= DM_BTN_ZL;
    for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && g_ptr[i].thr) b |= DM_BTN_ZR;
    if (g_aim && (menu || (GESTURES_ON && touch_menu_cutscene()))) g_aim = 0;   // a menu opened or a cutscene began: no aiming
    if (g_aim) b |= DM_BTN_ZR | (aim_lock() ? DM_BTN_ZL : 0);   // aiming: the throw held, and the lock-on (aim_lock)
    if (t < g_lock_until && aim_lock()) b |= DM_BTN_ZL;
    for (int i = 0; i < 10; i++) if (GESTURES_ON && g_ptr[i].id >= 0 && g_ptr[i].target == C_CAM && !g_ptr[i].thr && !g_ptr[i].aim && !touch_menu_active() && t - g_ptr[i].t0 > 0.32 &&
                                     fabsf(g_ptr[i].x - g_ptr[i].x0) < g_h * 0.04f && fabsf(g_ptr[i].y - g_ptr[i].y0) < g_h * 0.04f) { if (!g_ptr[i].lp) fprintf(stderr, "[touch] lock on\n"); g_ptr[i].lp = 1; b |= DM_BTN_ZL; }   // press and hold: ZL (lock on)
    for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && g_ptr[i].lp) b |= DM_BTN_ZL;   // a throw swipe holds ZR (aim) until the finger lifts (throw)
    if (rstick || (b & (DM_BTN_UP | DM_BTN_DOWN | DM_BTN_LEFT | DM_BTN_RIGHT))) g_inner_t = t;   // held: stays clear
    g_rstick_on = rstick; if (!rstick) { g_rknob_x = g_c[C_RSTICK].cx; g_rknob_y = g_c[C_RSTICK].cy; }
    if (!stick) { g_stick_on = 0; g_stick_x = g_knob_x = g_c[C_STICK].cx; g_stick_y = g_knob_y = g_c[C_STICK].cy; }
    // the camera: the finger's speed, full turn at about a screen height per 0.6 s, at rest within 60 ms of the finger stopping
    float vmax = g_h / 0.6f; if (t - g_cam_t > 0.06 || (menu && !map)) g_cam_vx = g_cam_vy = 0;
    { int aiming = g_aim; for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && (g_ptr[i].thr || g_ptr[i].lp || g_ptr[i].target == C_THROW)) aiming = 1;
      if (aiming) vmax *= 100.0f / AIM_FINE_PCT; }   // aiming / locked on: finer (the same drag turns less)
    // a drag up / down on an open menu (outside the controls) scrolls its list while the finger stays down, faster the further it is from where it landed (past
    // a dead zone; full speed at a fifth of the screen height); finger up = further down the list (touch_menu.c steps the selection, stopping at the list's ends)
    float mry = 0; int mscroll = 0;
    for (int i = 0; i < 10; i++) if (GESTURES && menu && !map && g_ptr[i].id >= 0 && g_ptr[i].target == C_CAM) {   // (the stick's finger moves the stick)
        float dx = g_ptr[i].x - g_ptr[i].x0, dy = g_ptr[i].y - g_ptr[i].y0, dz = g_h * 0.04f;
        if (fabsf(dy) < dz || fabsf(dy) < fabsf(dx)) continue;
        float v = (fabsf(dy) - dz) / (g_h * 0.2f); if (v > 1) v = 1; if (v < 0.3f) v = 0.3f;   // (the game ignores a slight tilt)
        mry = dy < 0 ? -v : v; mscroll = 1; }
    touch_menu_drag_rate(mscroll ? mry : 0);   // (touch_menu.c steps the list; taps wait for the drag to end)
    if (map) {   // one finger on the map: it follows the finger (full stick at half the screen height a second); two: a pinch, steps of an eighth of their distance
        int nf = 0, f[2] = { -1, -1 }; for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && g_ptr[i].target == C_CAM) { if (nf < 2) f[nf] = i; nf++; }
        if (nf == 1 && !stick) { float vm = g_h / 0.5f; lx = clampf(-g_cam_vx / vm, -1, 1); ly = clampf(g_cam_vy / vm, -1, 1); }
        if (nf == 2) { float d = hypotf(g_ptr[f[0]].x - g_ptr[f[1]].x, g_ptr[f[0]].y - g_ptr[f[1]].y);
            if (g_pinch_d0 <= 0) g_pinch_d0 = d;
            else if (d > g_pinch_d0 * 1.125f || d < g_pinch_d0 / 1.125f) { uint64_t zb = d > g_pinch_d0 ? DM_BTN_ZR : DM_BTN_ZL; g_pinch_d0 = d;
                dm_input_press_frames(zb, 90000000ull, 2); click_locked(); static int nz; if (nz++ < 10) fprintf(stderr, "[touch] map pinch: %s\n", zb == DM_BTN_ZR ? "zoom in" : "zoom out"); }
            g_cam_vx = g_cam_vy = 0; }
        else g_pinch_d0 = 0;
    }
    g_btn = b; g_lx = ax(lx); g_ly = ax(ly);
    if (map) { g_rx = rstick ? ax(rx) : 0; g_ry = rstick ? ax(ry) : 0; }   // (the map's right stick is its own: no camera drag)
    else if (rstick) { g_rx = ax(rx); g_ry = ax(ry); }
    else { static float ex, ey; static double et; double dt = et > 0 ? t - et : 0; et = t;
        float k = dt > 0 ? 1.0f - expf(-(float)dt / 0.07f) : 1.0f, tx = g_cam_vx / vmax, ty = -g_cam_vy / vmax;
        ex += (tx - ex) * k; ey += (ty - ey) * k; if (fabsf(ex) < 0.01f && tx == 0) ex = 0; if (fabsf(ey) < 0.01f && ty == 0) ey = 0;
        g_rx = ax(ex); g_ry = ax(ey); }
}
// ---------- the layout editor: hold − for 2 s to enter; drag a control to move it, pinch on it to resize it, double-tap it to hide / show it;
// hold + for 2 s to go back to the default layout; hold − for 2 s again to save and leave. The game gets no input meanwhile ----------
static double g_edit_hold_t;   // (g_edit_hold_t: when the hold that toggles the editor started)
static struct { int32_t id; int c; float ox, oy, x0, y0; double t0; } g_ep[10];
static struct { int c, active; float d0, r0; } g_pinch;
static int g_last_tap_c = -1; static double g_last_tap_t;
static int edit_hit(float x, float y) {   // the smallest control (shown or not) under the finger
    int best = -1; float br = 1e9f;
    for (int c = 0; c < NCTL; c++) { if (c == C_CAM || ctl_off(c) || (!g_c[c].shown && !g_ov[c].set && c != C_STICK)) continue;
        if (c == C_BACK || c == C_SEL) continue;   // (placed by the game's own list / HUD: not the player's to move)
        if (GESTURES_ON && !g_c[c].shown && !g_ov[c].set && c != C_PLUS && c != C_MINUS && c != C_MENU) continue;
        float dx = x - g_c[c].cx, dy = y - g_c[c].cy, r = g_c[c].r * 1.2f; if (dx * dx + dy * dy <= r * r && g_c[c].r < br) { br = g_c[c].r; best = c; } }
    return best;
}
static void edit_store(int c) { g_ov[c].set = 1; g_ov[c].fx = g_c[c].cx / g_w; g_ov[c].fy = g_c[c].cy / g_h; g_ov[c].fr = g_c[c].r / g_h; g_ov[c].shown = g_c[c].shown; }
static void edit_event(const AInputEvent *e) {
    int32_t act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK;
    size_t idx = (size_t)((AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT);
    if (act == AMOTION_EVENT_ACTION_CANCEL) { for (int i = 0; i < 10; i++) g_ep[i].id = -1; g_pinch.active = 0; return; }
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        size_t k = act == AMOTION_EVENT_ACTION_DOWN ? 0 : idx; float x = AMotionEvent_getX(e, k), y = AMotionEvent_getY(e, k);
        if (act == AMOTION_EVENT_ACTION_DOWN) { for (int i = 0; i < 10; i++) g_ep[i].id = -1; g_pinch.active = 0; }
        int c = edit_hit(x, y);
        for (int i = 0; i < 10; i++) if (g_ep[i].id >= 0 && c >= 0 && g_ep[i].c == c) {   // a second finger on the same control: pinch
            float dx = x - g_ep[i].x0, dy = y - g_ep[i].y0; g_pinch.c = c; g_pinch.active = 1; g_pinch.d0 = sqrtf(dx * dx + dy * dy) + 1; g_pinch.r0 = g_c[c].r; break; }
        for (int i = 0; i < 10; i++) if (g_ep[i].id < 0) { g_ep[i].id = AMotionEvent_getPointerId(e, k); g_ep[i].c = c; g_ep[i].x0 = x; g_ep[i].y0 = y; g_ep[i].t0 = now_s();
            g_ep[i].ox = c >= 0 ? g_c[c].cx - x : 0; g_ep[i].oy = c >= 0 ? g_c[c].cy - y : 0; break; }
    } else if (act == AMOTION_EVENT_ACTION_MOVE) {
        float px[10], py[10]; int n = 0;
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) { int32_t id = AMotionEvent_getPointerId(e, k); float x = AMotionEvent_getX(e, k), y = AMotionEvent_getY(e, k);
            for (int i = 0; i < 10; i++) if (g_ep[i].id == id && g_ep[i].c >= 0) {
                if (g_pinch.active && g_pinch.c == g_ep[i].c) { if (n < 10) { px[n] = x; py[n] = y; n++; } }
                else { int c = g_ep[i].c; g_c[c].cx = clampf(x + g_ep[i].ox, g_c[c].r, g_w - g_c[c].r); g_c[c].cy = clampf(y + g_ep[i].oy, g_c[c].r, g_h - g_c[c].r); edit_store(c); } } }
        if (g_pinch.active && n >= 2) { float dx = px[0] - px[1], dy = py[0] - py[1]; int c = g_pinch.c;
            g_c[c].r = clampf(g_pinch.r0 * (sqrtf(dx * dx + dy * dy) + 1) / g_pinch.d0, g_h * 0.025f, g_h * 0.25f); edit_store(c); }
    } else if (act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) {
        int32_t id = AMotionEvent_getPointerId(e, act == AMOTION_EVENT_ACTION_UP ? 0 : idx); double t = now_s();
        for (int i = 0; i < 10; i++) if (g_ep[i].id == id) { int c = g_ep[i].c;
            float mx = AMotionEvent_getX(e, act == AMOTION_EVENT_ACTION_UP ? 0 : idx) - g_ep[i].x0, my = AMotionEvent_getY(e, act == AMOTION_EVENT_ACTION_UP ? 0 : idx) - g_ep[i].y0;
            if (c >= 0 && t - g_ep[i].t0 < 0.3 && mx * mx + my * my < g_h * g_h * 0.0004f) {   // a tap: a double tap toggles the control
                if (g_last_tap_c == c && t - g_last_tap_t < 0.4 && c != C_PLUS && c != C_MINUS && c != C_MENU) { g_c[c].shown = !g_c[c].shown; edit_store(c); g_last_tap_c = -1; click_locked(); }
                else { g_last_tap_c = c; g_last_tap_t = t; } }
            if (g_pinch.active && g_pinch.c == c) g_pinch.active = 0;
            g_ep[i].id = -1; }
    }
}
// the 2-second holds, checked once a frame (a finger held still sends no events): − toggles the editor (saving on the way out), + in the editor resets the layout
static void edit_holds_locked(void) {
    double t = now_s(); int minus = 0, plus = 0;
    if (g_edit) { for (int i = 0; i < 10; i++) if (g_ep[i].id >= 0 && !g_pinch.active) { if (g_ep[i].c == C_MINUS) minus = 1; if (g_ep[i].c == C_PLUS) plus = 1; } }
    else for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && g_ptr[i].target == C_MINUS) minus = 1;
    if (!minus && !plus) { g_edit_hold_t = 0; return; }
    if (!g_edit_hold_t) { g_edit_hold_t = t; return; }
    if (t - g_edit_hold_t < 2.0) return;
    g_edit_hold_t = 1e18;   // fired: nothing more until the finger lifts
    if (plus && g_edit) { memset(g_ov, 0, sizeof g_ov); remove(g_lay_path); g_w = g_h = 0; click_locked(); fprintf(stderr, "[touch] layout reset to the default\n"); return; }
    if (g_edit) { layout_save(); g_edit = 0; for (int i = 0; i < 10; i++) g_ep[i].id = -1; fprintf(stderr, "[touch] layout saved: %s\n", g_lay_path); }
    else { g_edit = 1; for (int i = 0; i < 10; i++) { if (g_ptr[i].id >= 0) { g_ep[0].id = g_ptr[i].id; g_ep[0].c = C_MINUS; g_ep[0].t0 = t; } g_ptr[i].id = -1; }
           g_btn = 0; g_lx = g_ly = g_rx = g_ry = 0; dm_input_set(0, 0, 0, 0, 0); fprintf(stderr, "[touch] layout editor on\n"); }
    click_locked();
}
// Returns the touch buttons and sticks (for the host's touch source) after this event
void touch_ui_event(const AInputEvent *e, float w, float h, uint64_t *btn, int32_t s[4]) {
    pthread_mutex_lock(&g_mu);
    layout(w, h); back_place_locked(); g_visible = 1; g_touch_t = now_s();
    if (g_edit) { edit_event(e); *btn = 0; s[0] = s[1] = s[2] = s[3] = 0; pthread_mutex_unlock(&g_mu); return; }
    int32_t act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK;
    size_t idx = (size_t)((AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT);
    uint64_t before = g_btn;
    if (act == AMOTION_EVENT_ACTION_CANCEL) for (int i = 0; i < 10; i++) g_ptr[i].id = -1;
    else if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        size_t k = act == AMOTION_EVENT_ACTION_DOWN ? 0 : idx; float x = AMotionEvent_getX(e, k), y = AMotionEvent_getY(e, k);
        if (act == AMOTION_EVENT_ACTION_DOWN) for (int i = 0; i < 10; i++) g_ptr[i].id = -1;
        int t = target_at(x, y);
        if (t >= 0) for (int i = 0; i < 10; i++) if (g_ptr[i].id < 0) {
            if (t == C_STICK || t == C_CAM) for (int j = 0; j < 10; j++) if (g_ptr[j].id >= 0 && (g_ptr[j].target == C_STICK || g_ptr[j].target == C_CAM) && now_s() - g_ptr[j].t0 < 0.25) g_two_t = now_s();
            g_ptr[i].thr = g_ptr[i].lp = 0; g_ptr[i].aim = g_aim && (t == C_RSTICK || t == C_CAM); g_ptr[i].id = AMotionEvent_getPointerId(e, k); g_ptr[i].target = t; g_ptr[i].x0 = g_ptr[i].x = g_ptr[i].lx = x; g_ptr[i].y0 = g_ptr[i].y = g_ptr[i].ly = y; g_ptr[i].lt = g_ptr[i].t0 = now_s();
            if (t >= C_A && t != C_THROW && t != C_LOCK) g_held_until[t] = now_s() + 0.08;
            if (t == C_LOCK) { g_lockon = !g_lockon; fprintf(stderr, "[touch] lock-on %s\n", g_lockon ? "on" : "off"); }
            if (t == C_RSTICK || (t >= C_UP && t <= C_RIGHT)) g_inner_t = now_s();
            if (t == C_STICK) { float r = g_c[C_STICK].r; g_stick_on = 1;   // the base comes to the thumb, kept on screen
                (void)r; g_stick_x = g_c[C_STICK].cx; g_stick_y = g_c[C_STICK].cy; }   // fixed, as on a Joy-Con: the thumb's offset from the base is the tilt
            break; }
    } else if (act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) {
        int32_t id = AMotionEvent_getPointerId(e, act == AMOTION_EVENT_ACTION_UP ? 0 : idx);
        // aiming (a tap on the throw button): the aiming finger's lift (right stick or camera side) throws, as does a second tap on the ball; a first tap starts it
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id && g_ptr[i].aim) {
            int aimed = now_s() - g_ptr[i].t0 > 0.12 || fabsf(g_ptr[i].x - g_ptr[i].x0) > g_h * 0.02f || fabsf(g_ptr[i].y - g_ptr[i].y0) > g_h * 0.02f;
            if (aimed) { g_aim = 0; g_lock_until = now_s() + 0.25; g_cam_vx = g_cam_vy = 0; g_ptr[i].id = -1; fprintf(stderr, "[touch] throw\n"); }
            else g_ptr[i].aim = 0; }   // a brush: still aiming
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id && g_ptr[i].target == C_THROW) {
            if (now_s() - g_ptr[i].t0 < QUICK_S) { g_qt0 = now_s(); fprintf(stderr, "[touch] quick throw\n"); }   // a tap: thrown at once (recompute_locked)
            else { g_lock_until = now_s() + 0.25; fprintf(stderr, "[touch] aimed throw\n"); } }   // held: ZR let go now is the throw; the lock stays a moment
        if (0) {
            g_aim = !g_aim; if (!g_aim) g_lock_until = now_s() + 0.25; fprintf(stderr, "[touch] %s\n", g_aim ? "aiming: right stick or a drag on the right aims, lifting it throws" : "throw"); }
        // gestures outside the controls (stick or camera area): a quick tap (under 0.25 s, barely moved) is a tap on a menu item while a menu shows (touch_menu.c),
        // else, on the camera side, A (talk, pick up, investigate); two fingers tapping together are B (back)
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id && (g_ptr[i].target == C_CAM || (g_ptr[i].target == C_STICK && !touch_menu_active())) && !g_ptr[i].thr && now_s() - g_ptr[i].t0 < 0.3 &&
                                         fabsf(g_ptr[i].x - g_ptr[i].x0) < g_h * 0.02f && fabsf(g_ptr[i].y - g_ptr[i].y0) < g_h * 0.02f) {
            double t = now_s();
            if (t - g_two_t < 0.4) { g_two_t = 0; g_tap_mute = t + 0.3; if (g_aim) { g_aim = 0; for (int j = 0; j < 10; j++) g_ptr[j].aim = 0; fprintf(stderr, "[touch] aim cancelled\n"); }   // two fingers: B, and no throw
                dm_input_press(DM_BTN_B, 80000000ull); click_locked(); }
            else if (t >= g_tap_mute && !g_ptr[i].lp) { static int dg = -1; if (dg < 0) dg = getenv("DM_TOUCH_DIAG") != NULL; if (dg && !touch_menu_active()) touch_menu_tap(g_ptr[i].x0, g_ptr[i].y0, g_w, g_h);   // (diagnosis: what is under the tap)
                uint64_t hb = GESTURES && !touch_menu_active() && !touch_menu_cutscene() ? hud_hint(g_ptr[i].x0, g_ptr[i].y0) : 0;
                if (hb) { dm_input_press_frames(hb, 150000000ull, 4); click_locked(); static int nh; if (nh++ < 20) fprintf(stderr, "[touch] HUD hint %s\n", hb == DM_BTN_UP ? "up" : hb == DM_BTN_DOWN ? "down" : "minus"); }
                else if (touch_menu_active() && menu_back(g_ptr[i].x0, g_ptr[i].y0)) { touch_menu_back(); click_locked(); }
                else if (touch_menu_active()) touch_menu_tap(g_ptr[i].x0, g_ptr[i].y0, g_w, g_h); else if (GESTURES && g_ptr[i].target == C_CAM) { dm_input_press(DM_BTN_A, 80000000ull); click_locked(); } }
        }
        // swipes on an open menu (outside the controls): left / right move across its list, one item per 0.08 of the screen height travelled (finger left =
        // the next one to the right); up / down scroll it with the right stick while the finger is down (recompute_locked)
        for (int i = 0; i < 10; i++) if (GESTURES && g_ptr[i].id == id && g_ptr[i].target == C_CAM && touch_menu_active() && !touch_menu_map()) {
            float sx = g_ptr[i].x - g_ptr[i].x0, sy = g_ptr[i].y - g_ptr[i].y0, d = fmaxf(fabsf(sx), fabsf(sy));
            if (d < g_h * 0.05f) continue;
            int n = (int)(d / (g_h * 0.08f) + 0.5f); if (n < 1) n = 1; if (n > 5) n = 5;
            if (fabsf(sy) >= fabsf(sx)) continue; touch_menu_scroll(sx < 0 ? n : -n, 0);
            click_locked();
        }
        // the selector: a sideways swipe moves through the items (swipe left = the next one, R; right = the previous one, L), as a carousel; a tap on a side bubble
        // selects it (L / R), on the middle bubble throws it (ZR), on the swap icon switches items / Pokémon (X)
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id && g_ptr[i].target == C_SEL) {
            float sx = g_ptr[i].x - g_ptr[i].x0, sy = g_ptr[i].y - g_ptr[i].y0; uint64_t sb = 0;
            if (fabsf(sx) > g_h * 0.035f && fabsf(sx) > fabsf(sy)) sb = sx < 0 ? DM_BTN_R : DM_BTN_L;
            else if (sy < -g_h * 0.06f && fabsf(sy) > fabsf(sx)) sb = DM_BTN_ZR;   // flicked up: throw
            else if (fabsf(sx) < g_h * 0.02f && fabsf(sy) < g_h * 0.02f && now_s() - g_ptr[i].t0 < 0.4) {
                int part = sel_part(g_ptr[i].x0, g_ptr[i].y0);
                sb = part == SEL_LEFT ? DM_BTN_L : part == SEL_RIGHT ? DM_BTN_R : part == SEL_SWAP ? DM_BTN_X : DM_BTN_ZR; }
            if (sb) { dm_input_press_frames(sb, 150000000ull, 4); click_locked(); static int nl; if (nl++ < 20) fprintf(stderr, "[touch] selector %s\n", sb == DM_BTN_L ? "L" : sb == DM_BTN_R ? "R" : sb == DM_BTN_X ? "X" : "ZR"); }
        }
        // quick swipes on the camera side (under 0.35 s, a tenth of the screen height or more): left / right = L / R (the item or Pokémon), down = X (items / Pokémon)
        for (int i = 0; i < 10; i++) if (GESTURES && g_ptr[i].id == id && g_ptr[i].target == C_CAM && !g_ptr[i].thr && !g_ptr[i].lp && now_s() - g_ptr[i].t0 < 0.35 && !touch_menu_active()) {
            float sx = g_ptr[i].x - g_ptr[i].x0, sy = g_ptr[i].y - g_ptr[i].y0; uint64_t sb = 0;
            if (fabsf(sx) > g_h * 0.10f && fabsf(sy) < fabsf(sx) * 0.6f) sb = sx > 0 ? DM_BTN_R : DM_BTN_L;
            else if (sy > g_h * 0.10f && fabsf(sx) < sy * 0.6f) sb = DM_BTN_X;
            if (sb) { dm_input_press(sb, 80000000ull); click_locked(); g_cam_vx = g_cam_vy = 0; }
        }
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id && g_ptr[i].target == C_MENU && hit(C_MENU, g_ptr[i].x, g_ptr[i].y, 1.5f)) { click_locked(); front_open_menu(0); }
        for (int i = 0; i < 10; i++) if (g_ptr[i].id == id) { if (g_ptr[i].target == C_CAM) g_cam_vx = g_cam_vy = 0; g_ptr[i].id = -1; }
    }
    if (act == AMOTION_EVENT_ACTION_MOVE || act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN)
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) {
            int32_t id = AMotionEvent_getPointerId(e, k);
            for (int i = 0; i < 10; i++) if (g_ptr[i].id == id) {
                g_ptr[i].x = AMotionEvent_getX(e, k); g_ptr[i].y = AMotionEvent_getY(e, k);
                if (g_ptr[i].target == C_CAM && !g_ptr[i].thr && act == AMOTION_EVENT_ACTION_MOVE && GESTURES && now_s() - g_ptr[i].t0 < 0.3 && !touch_menu_active()) {   // a quick swipe up: a throw
                    float up = g_ptr[i].y0 - g_ptr[i].y; if (up > g_h * 0.10f && fabsf(g_ptr[i].x - g_ptr[i].x0) < up * 0.5f) { g_ptr[i].thr = 1; g_cam_vx = g_cam_vy = 0; click_locked(); }
                }
                if ((g_ptr[i].target == C_CAM || g_ptr[i].target == C_THROW) && !g_ptr[i].thr && act == AMOTION_EVENT_ACTION_MOVE) {   // the finger's speed, smoothed (the throw button's finger aims)
                    double t = now_s(), dt = t - g_ptr[i].lt;
                    if (dt > 0.001) { float vx = (g_ptr[i].x - g_ptr[i].lx) / (float)dt, vy = (g_ptr[i].y - g_ptr[i].ly) / (float)dt;
                        g_cam_vx = g_cam_vx * 0.5f + vx * 0.5f; g_cam_vy = g_cam_vy * 0.5f + vy * 0.5f; g_cam_t = t;
                        g_ptr[i].lx = g_ptr[i].x; g_ptr[i].ly = g_ptr[i].y; g_ptr[i].lt = t; }
                }
                if (g_ptr[i].target >= C_A && g_ptr[i].target <= C_Y)   // sliding across the face buttons
                    for (int c = C_A; c <= C_Y; c++) if (c != g_ptr[i].target && hit(c, g_ptr[i].x, g_ptr[i].y, 1.0f)) { g_ptr[i].target = c; g_held_until[c] = now_s() + 0.08; break; }
            }
        }
    recompute_locked();
    if (g_btn & ~before) click_locked();   // a new press: one click
    *btn = g_btn; s[0] = g_lx; s[1] = g_ly; s[2] = g_rx; s[3] = g_ry;
    pthread_mutex_unlock(&g_mu);
}
void touch_ui_hide(void) { pthread_mutex_lock(&g_mu); g_visible = 0; pthread_mutex_unlock(&g_mu); }
// GameActivity.java: a controller connected hides the controls at once (no input needed first); the last one disconnected shows them again
void touch_ui_pad(int connected) { pthread_mutex_lock(&g_mu); g_visible = !connected; if (!connected) g_touch_t = now_s(); pthread_mutex_unlock(&g_mu); }
#else
// Mac test hook (DM_OVL_SHOT): show control c as held, the stick pushed by (dx, dy) in units of its radius
void touch_ui_demo(int c, float dx, float dy) {
    pthread_mutex_lock(&g_mu); layout(1, 1); g_ptr[0].id = 1; g_ptr[0].target = c; g_btn = c >= C_A && c < NCTL ? g_c[c].btn : 0; g_w = g_h = 0; pthread_mutex_unlock(&g_mu);
    (void)dx; (void)dy;
}
#endif

// ---------- drawing ----------
// The front end's input push for what changes with time alone (a tap's hold running out, the camera coming to rest): called once a frame by the drawing
#ifdef __ANDROID__
static void frame_input_locked(void) {
    edit_holds_locked(); if (g_edit) return;   // (the editor: the game gets nothing)
    uint64_t b0 = g_btn; int32_t l0 = g_lx, l1 = g_ly, r0 = g_rx, r1 = g_ry;
    recompute_locked();
    if (g_btn != b0 || g_lx != l0 || g_ly != l1 || g_rx != r0 || g_ry != r1) dm_input_set(g_btn, g_lx, g_ly, g_rx, g_ry);   // (the left stick too: a menu drag lets go of it at a list's edge)
}
#else
static void frame_input_locked(void) {}
#endif
static dm_ovl_item *add(dm_ovl_item *it, int *n, int max, float cx, float cy, float r, int shape, int R, int G, int B, float a) {
    if (*n >= max) return NULL;
    dm_ovl_item *o = &it[(*n)++];
    *o = (dm_ovl_item){ cx - r, cy - r, 2 * r, 2 * r, clampf(R, 0, 255) / 255.0f, clampf(G, 0, 255) / 255.0f, clampf(B, 0, 255) / 255.0f, clampf(a, 0, 1), shape, 0.1f, 0.4f, OVL_G_NONE, 0.5f, 0 };
    return o;
}
// a frosted-glass bubble (grey and see-through, so it sits on the game's picture instead of standing out from it): a soft shadow, a darker rim, the body,
//   a faint shine, a thin light rim; the label on top. Pressed: lighter. lit: a control that is on (lock-on), a brighter frost
static void bubble(dm_ovl_item *it, int *n, int max, const ctl *c, int on, int lit, float a) {
    float r = on ? c->r * 0.95f : c->r, x = c->cx, y = c->cy, th = g_h / 1080.0f;
    int g = lit ? 150 : 52; if (on) g += 60;
    add(it, n, max, x, y + r * 0.05f, r * 1.02f, OVL_DISC, 0, 0, 0, 0.18f * a);
    add(it, n, max, x, y, r, OVL_DISC, g / 2, g / 2, g / 2 + 2, 0.40f * a);
    add(it, n, max, x, y, r * 0.90f, OVL_DISC, g, g + 2, g + 6, (on || lit ? 0.70f : 0.45f) * a);
    add(it, n, max, x - r * 0.28f, y - r * 0.38f, r * 0.22f, OVL_DISC, 255, 255, 255, 0.10f * a);
    dm_ovl_item *o = add(it, n, max, x, y, r * 0.90f, OVL_RING, 235, 236, 240, (on || lit ? 0.80f : 0.50f) * a); if (o) o->ring = clampf(2.4f * th / (r * 0.9f), 0.02f, 0.2f);
    o = add(it, n, max, x, y, r, OVL_DISC, 0, 0, 0, 0); if (o) { o->glyph = c->glyph; o->gscale = c->glyph == OVL_G_ZL || c->glyph == OVL_G_ZR ? 0.62f : 0.55f; o->galpha = a; }
}
// the throw button: a Poké Ball as the game's HUD shows it (shadow, dark rim, red top half, white bottom half, a dark band, the white centre button), pressed = slightly smaller and brighter
static void pokeball(dm_ovl_item *it, int *n, int max, const ctl *c, int on, float a) {
    float r = on ? c->r * 0.93f : c->r, x = c->cx, y = c->cy, th = g_h / 1080.0f; int lift = on ? 40 : 0;
    add(it, n, max, x, y + r * 0.07f, r * 1.03f, OVL_DISC, 5, 20, 40, 0.35f * a);
    add(it, n, max, x, y, r, OVL_DISC, 30, 30, 36, 0.92f * a);
    add(it, n, max, x, y, r * 0.90f, OVL_DISC, 238, 238, 242, 0.95f * a);
    add(it, n, max, x, y, r * 0.90f, OVL_HALF, 215 + lift / 3, 35 + lift, 40 + lift, 0.97f * a);
    dm_ovl_item *o = add(it, n, max, x, y, r * 0.90f, OVL_RRECT, 30, 30, 36, 0.95f * a); if (o) { o->y = y - r * 0.08f; o->h = r * 0.16f; o->radius = 0.5f; }
    add(it, n, max, x, y, r * 0.30f, OVL_DISC, 30, 30, 36, 0.95f * a);
    add(it, n, max, x, y, r * 0.21f, OVL_DISC, 250, 250, 250, 0.98f * a);
    add(it, n, max, x - r * 0.32f, y - r * 0.45f, r * 0.18f, OVL_DISC, 255, 255, 255, 0.35f * a);
    o = add(it, n, max, x, y, r, OVL_RING, 210, 225, 240, 0.9f * a); if (o) o->ring = clampf(2.6f * th / r, 0.01f, 0.2f);
}
// the settings button: a bubble with three white bars (☰)
static void menu_icon(dm_ovl_item *it, int *n, int max, const ctl *c, float a) {
    bubble(it, n, max, c, 0, 0, a);
    float bw = c->r * 0.95f, bh = c->r * 0.13f;
    for (int k = -1; k <= 1; k++) { dm_ovl_item *o = add(it, n, max, c->cx, c->cy, c->r, OVL_RRECT, 255, 255, 255, 0.95f * a);
        if (o) { o->x = c->cx - bw / 2; o->y = c->cy + k * c->r * 0.30f - bh / 2; o->w = bw; o->h = bh; o->radius = 0.5f; } }
}
// the back button, as the game draws its button prompts (the Satchel's "B Back"): a near-white disc with a dark B, a faint shadow; pressed = greyer
static void back_button(dm_ovl_item *it, int *n, int max, const ctl *c, int on, float a) {
    float r = c->r, x = c->cx, y = c->cy; int v = on ? 190 : 238;
    add(it, n, max, x, y + r * 0.10f, r * 1.04f, OVL_DISC, 0, 0, 0, 0.30f * a);
    dm_ovl_item *o = add(it, n, max, x, y, r, OVL_DISC, v, v, v, 0.97f * a);
    if (o) { o->glyph = c->glyph; o->gscale = 0.62f; o->galpha = a; o->gdark = 0.85f; }
}
// the stick, in the same frosted grey: a faint base with a light ring and an inner ring, and a grey knob
static void compass(dm_ovl_item *it, int *n, int max, float a, float g_stick_x, float g_stick_y, float g_knob_x, float g_knob_y, float r) {   // (base, knob, radius)
    float th = g_h / 1080.0f, kr = r * 0.42f;
    add(it, n, max, g_stick_x, g_stick_y, r, OVL_DISC, 40, 42, 46, 0.28f * a);
    dm_ovl_item *o = add(it, n, max, g_stick_x, g_stick_y, r, OVL_RING, 235, 236, 240, 0.50f * a); if (o) o->ring = clampf(2.4f * th / r, 0.01f, 0.2f);
    o = add(it, n, max, g_stick_x, g_stick_y, r * 0.80f, OVL_RING, 235, 236, 240, 0.18f * a); if (o) o->ring = clampf(1.6f * th / (r * 0.8f), 0.01f, 0.2f);
    add(it, n, max, g_knob_x, g_knob_y + kr * 0.10f, kr, OVL_DISC, 0, 0, 0, 0.18f * a);
    add(it, n, max, g_knob_x, g_knob_y, kr, OVL_DISC, 120, 122, 128, 0.62f * a);
    add(it, n, max, g_knob_x, g_knob_y - kr * 0.30f, kr * 0.6f, OVL_DISC, 255, 255, 255, 0.10f * a);
    o = add(it, n, max, g_knob_x, g_knob_y, kr, OVL_RING, 240, 240, 245, 0.75f * a); if (o) o->ring = clampf(2.2f * th / kr, 0.02f, 0.3f);
}
// the player's settings (env.txt, set on the app's settings screen): DM_TOUCH_OPACITY 20-100 (%), DM_TOUCH_FADE seconds untouched before the controls go faint (0 = never)
static float g_opacity = -1, g_fade = 3.0f;
// the in-game menu changed one (android_main.c nativeSetting): value NULL = back to the default
void touch_ui_setting(const char *key, const char *value) {
    pthread_mutex_lock(&g_mu);
    if (!strcmp(key, "DM_TOUCH_OPACITY")) g_opacity = value ? clampf((float)atof(value) / 100.0f, 0.2f, 1.0f) : 1.0f;
    else if (!strcmp(key, "DM_TOUCH_FADE")) g_fade = value ? fmaxf(0.0f, (float)atof(value)) : 3.0f;
    else if (!strcmp(key, "DM_TOUCH_L3R3")) { g_l3r3 = value && !strcmp(value, "on"); g_w = g_h = 0; }   // (laid out again on the next frame)
#ifdef __ANDROID__
    else { void touch_ui_haptic_setting(const char *, const char *); touch_ui_haptic_setting(key, value); }   // DM_TOUCH_HAPTIC, DM_RUMBLE
#endif
    g_touch_t = now_s();
    pthread_mutex_unlock(&g_mu);
}
#ifdef __ANDROID__
// the in-game menu's "Move the touch controls": the layout editor (as holding − for 2 s)
void touch_ui_edit(void) {
    pthread_mutex_lock(&g_mu);
    if (!g_edit) { g_edit = 1; for (int i = 0; i < 10; i++) { g_ptr[i].id = -1; g_ep[i].id = -1; } g_btn = 0; g_lx = g_ly = g_rx = g_ry = 0; dm_input_set(0, 0, 0, 0, 0); g_visible = 1; g_touch_t = now_s(); }
    pthread_mutex_unlock(&g_mu);
}
#endif
static void prefs(void) {
    if (g_opacity >= 0) return;
    const char *o = getenv("DM_TOUCH_OPACITY"), *f = getenv("DM_TOUCH_FADE");
    g_opacity = o ? clampf((float)atof(o) / 100.0f, 0.2f, 1.0f) : 1.0f; if (f) g_fade = fmaxf(0.0f, (float)atof(f));
}
int front_overlay(dm_ovl_item *it, int max, int width, int height) {
    pthread_mutex_lock(&g_mu); prefs();
    layout((float)width, (float)height); back_place_locked();
    frame_input_locked();
    int n = 0;
    if (g_visible && g_edit) {   // the editor: every control at full strength with a yellow ring, the hidden ones as ghosts
        for (int c = 0; c < NCTL; c++) { if (c == C_CAM || ctl_off(c)) continue; const ctl *k = &g_c[c];
            if (c == C_BACK || c == C_SEL) continue;   // (placed by the game's own list / HUD: not the player's to move)
            if (GESTURES_ON && !k->shown && !g_ov[c].set && c != C_PLUS && c != C_MINUS && c != C_STICK && c != C_MENU) continue;
            float a = k->shown ? 1.0f : 0.3f;
            if (c == C_STICK || c == C_RSTICK) compass(it, &n, max, a, k->cx, k->cy, k->cx, k->cy, k->r);
            else if (c == C_THROW) pokeball(it, &n, max, k, 0, a); else if (c == C_MENU) menu_icon(it, &n, max, k, a); else bubble(it, &n, max, k, 0, 0, a);
            dm_ovl_item *o = add(it, &n, max, k->cx, k->cy, k->r * 1.12f, OVL_RING, 255, 210, 60, 0.95f * a); if (o) o->ring = clampf(4.0f * (g_h / 1080.0f) / (k->r * 1.12f), 0.02f, 0.3f); }
    } else if (g_visible) {
        double t = now_s(); int idle = (g_fade > 0 && t - g_touch_t > g_fade) && !g_btn && !g_stick_on && !g_rstick_on;   // nothing touched for the fade time: everything faint (in menus and cutscenes too: hiding them there broke the flow of play)
        float ia = idle ? 0.22f : 1.0f;
        int faint = t - g_inner_t > 2.0;   // the D-pad and the right stick fade 2 s after their last touch
        int menu = GESTURES_ON && touch_menu_active();   // its items are tapped: the buttons go (target_at)
        for (int c = C_A; c < NCTL; c++) { if (!g_c[c].shown || (menu && (c == C_THROW || c == C_LOCK))) continue;
            int on = (g_btn & g_c[c].btn) != 0; float al = c >= C_UP && c <= C_RIGHT && faint && !on ? 0.25f : 1.0f; if (idle) al = fminf(al, ia);
            if (c == C_THROW) { pokeball(it, &n, max, &g_c[c], on, al);
                int aiming = 0; for (int i = 0; i < 10; i++) if (g_ptr[i].id >= 0 && g_ptr[i].target == C_THROW && t - g_ptr[i].t0 >= QUICK_S) aiming = 1;
                if (aiming || g_qt0 >= 0) { dm_ovl_item *o = add(it, &n, max, g_c[c].cx, g_c[c].cy, g_c[c].r * 1.22f, OVL_RING, 255, 214, 80, 0.95f); if (o) o->ring = 0.14f; }   // aiming / thrown: a gold ring
                continue; }
            if (c == C_BACK) { back_button(it, &n, max, &g_c[c], on, 1.0f); continue; }   // never faint: it is the menu's own control
            if (c == C_LOCK) {   // lock-on: a reticle, a brighter frost while on
                bubble(it, &n, max, &g_c[c], g_lockon, g_lockon, g_lockon ? 1.0f : al); float r = g_c[c].r;
                dm_ovl_item *o = add(it, &n, max, g_c[c].cx, g_c[c].cy, r * 0.55f, OVL_RING, 255, 255, 255, 0.95f * (g_lockon ? 1.0f : al)); if (o) o->ring = 0.28f;
                add(it, &n, max, g_c[c].cx, g_c[c].cy, r * 0.14f, OVL_DISC, 255, 255, 255, 0.95f * (g_lockon ? 1.0f : al));
                continue; }
            if (c == C_MENU) { menu_icon(it, &n, max, &g_c[c], idle ? fmaxf(al, 0.45f) : al); continue; }   // (never quite gone: it is how the settings are found)
            bubble(it, &n, max, &g_c[c], on, 0, al);
            if (c >= C_UP && c <= C_RIGHT && al == 1.0f) { float r = g_c[c].r, ox = c == C_LEFT ? -0.42f : c == C_RIGHT ? 0.42f : 0, oy = c == C_UP ? -0.42f : c == C_DOWN ? 0.42f : 0;
                add(it, &n, max, g_c[c].cx + ox * r, g_c[c].cy + oy * r, r * 0.22f, OVL_DISC, 255, 255, 255, 0.9f); } }
        compass(it, &n, max, g_stick_on ? 1.0f : idle ? ia : 0.75f, g_stick_x, g_stick_y, g_knob_x, g_knob_y, g_c[C_STICK].r);
        if (g_c[C_RSTICK].shown) compass(it, &n, max, g_rstick_on ? 1.0f : idle ? ia : faint ? 0.25f : 0.75f, g_c[C_RSTICK].cx,   // faint while unused: it and the D-pad sit over the dialogue box
             g_c[C_RSTICK].cy, g_rknob_x, g_rknob_y, g_c[C_RSTICK].r);
        for (int i = 0; i < n; i++) { it[i].a *= g_opacity; it[i].galpha *= g_opacity; }
    }
    pthread_mutex_unlock(&g_mu);
    return n;
}
