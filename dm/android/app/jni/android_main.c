// Android app front end (NativeActivity + android_native_app_glue): game files from the app's external files folder, the host (dm_main) on a big-stack thread, the window handed to
// the Vulkan replay for presentation, controllers and touch into the game's controller state, lifecycle (focus, background) into the host.
// Files, under Android/data/<package>/files/ (copied there from a computer, see dm/android/README.md):
//   RomFS/            the game's RomFS
//   patched/          the code modules rewritten for Android (patch_nso.py, DM_PATCH_TARGET=android)
//   shaders.pak       the Vulkan shaders packed (dm/android/tools/spv_pak.py), or vk/b-<chain>/ with <key>.spv + <key>.vkr; shader_map.tsv beside them
//   font/             the shared fonts (standard.ttf, nintendo_ext.ttf)
//   env.txt           optional KEY=VALUE switches (as the iOS app's)
// Saves and the log (log.txt, the previous run in log.prev.txt) are written there too.
#include <android/log.h>
#include <android/native_activity.h>
#include <android/window.h>
#include <android/input.h>
#include <android_native_app_glue.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <sched.h>
#include <errno.h>
#include <jni.h>
#include <sys/resource.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <dirent.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>
#include <strings.h>
#include <sys/system_properties.h>
#include "dm_input.h"
#include "front.h"
#include "game_profile.h"

int dm_main(int argc, char **argv);   // dm/host/main.c built with -Dmain=dm_main
void dm_focus_set(int st);
void dm_audio_suspend(int on);
#define ALOG(...) __android_log_print(ANDROID_LOG_INFO, "nxnative", __VA_ARGS__)

static struct android_app *g_app;
static _Atomic(ANativeWindow *) g_window;
static char g_files[1024], g_game[1024];
void *front_metal_layer(void) { return atomic_load(&g_window); }   // the window the Vulkan replay presents to (the name is the Apple front ends' interface)
void *front_android_window(void) { return atomic_load(&g_window); }
// The game's keyboard (its name entry): the app's Android dialog (app/java/com/nxnative/TextInput.java) with the phone's keyboard, on the UI thread; this game thread
// waits for it (hle_swkbd.c stops the game's clock meanwhile). 1 OK (out holds the text, UTF-8), 0 cancelled, -1 no dialog (then the game takes it as cancelled)
// The Java calls run on a thread of their own: the calling game thread runs on the game's stack, outside its pthread stack, and ART aborts when such a thread attaches
// (Thread::InitStack: "FindStackTop() > GetStackEnd()"), so the game thread only starts that thread and waits for it
extern int g_swkbd_mode, g_swkbd_numeric_max;   // hle_swkbd.c: what this request wants (numbers: the number pad)
static int text_input_jni(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    JavaVM *vm = g_app->activity->vm; JNIEnv *e; int attached = 0;
    if ((*vm)->GetEnv(vm, (void **)&e, JNI_VERSION_1_6) != JNI_OK) { if ((*vm)->AttachCurrentThread(vm, &e, NULL)) return -1; attached = 1; }
    int r = -1;
    jobject act = g_app->activity->clazz;
    jclass ac = (*e)->GetObjectClass(e, act);
    jobject cl = (*e)->CallObjectMethod(e, act, (*e)->GetMethodID(e, ac, "getClassLoader", "()Ljava/lang/ClassLoader;"));
    jclass clc = (*e)->FindClass(e, "java/lang/ClassLoader");
    jstring cn = (*e)->NewStringUTF(e, "com.nxnative.TextInput");
    jclass ti = (jclass)(*e)->CallObjectMethod(e, cl, (*e)->GetMethodID(e, clc, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;"), cn);
    if ((*e)->ExceptionCheck(e) || !ti) { (*e)->ExceptionClear(e); fprintf(stderr, "[RED] text input: the dialog class is missing\n"); goto done; }
    {
        jmethodID show = (*e)->GetStaticMethodID(e, ti, "show", "(Landroid/app/Activity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IZ)V");
        jmethodID st = (*e)->GetStaticMethodID(e, ti, "state", "()I"), tx = (*e)->GetStaticMethodID(e, ti, "text", "()Ljava/lang/String;");
        (*e)->CallStaticVoidMethod(e, ti, show, act, (*e)->NewStringUTF(e, title ? title : ""), (*e)->NewStringUTF(e, sub ? sub : ""), (*e)->NewStringUTF(e, guide ? guide : ""),
                                   (*e)->NewStringUTF(e, initial ? initial : ""), (jint)max_chars, (jboolean)(g_swkbd_mode == 1 || g_swkbd_numeric_max >= 0));
        if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionDescribe(e); (*e)->ExceptionClear(e); goto done; }
        int s; while (!(s = (*e)->CallStaticIntMethod(e, ti, st))) usleep(50000);
        if (s == 1) { jstring js = (jstring)(*e)->CallStaticObjectMethod(e, ti, tx); const char *c = js ? (*e)->GetStringUTFChars(e, js, NULL) : NULL;
            snprintf(out, (size_t)out_cap, "%s", c ? c : ""); if (c) (*e)->ReleaseStringUTFChars(e, js, c); r = 1; }
        else r = 0;
        fprintf(stderr, "text input '%s': %s\n", title ? title : "", r == 1 ? "entered" : "cancelled");
    }
done:
    if (attached) (*vm)->DetachCurrentThread(vm);
    return r;
}
struct text_input_req { const char *title, *sub, *guide, *initial; int max_chars; char *out; int out_cap, r; };
static void *text_input_main(void *p) {
    struct text_input_req *q = p;
    q->r = text_input_jni(q->title, q->sub, q->guide, q->initial, q->max_chars, q->out, q->out_cap);
    return NULL;
}
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    struct text_input_req q = { title, sub, guide, initial, max_chars, out, out_cap, -1 };
    pthread_t t; if (pthread_create(&t, NULL, text_input_main, &q)) { fprintf(stderr, "[RED] text input: no thread for the dialog\n"); return -1; }
    pthread_join(t, NULL);
    return q.r;
}
void front_save_failed(int err) { ALOG("save not written (errno %d)", err); }

// ---------- Input: controllers (key and motion events of gamepads / joysticks) and touch ----------
// Every controller keeps its own state and the game sees them merged into one: buttons of all of them, each stick axis from whichever controller moves it most. Android lists a
// pair of Joy-Cons as two devices (left: left stick, D-pad buttons, L / ZL / −; right: right stick, A B X Y, R / ZR / +), so held one in each hand they play as one controller
enum { NPAD = 8 };
static struct { int32_t dev; uint64_t btn, hat; int32_t lx, ly, rx, ry; } g_pads[NPAD];
static uint64_t g_pad_btn; static int32_t g_pad_lx, g_pad_ly, g_pad_rx, g_pad_ry;
static int pad_slot(int32_t dev) {
    for (int i = 0; i < NPAD; i++) if (g_pads[i].dev == dev + 2) return i;   // dev + 2: 0 marks a free slot, and the virtual keyboard's id is -1
    for (int i = 0; i < NPAD; i++) if (!g_pads[i].dev) { memset(&g_pads[i], 0, sizeof g_pads[i]); g_pads[i].dev = dev + 2; return i; }
    return NPAD - 1;
}
static int32_t strongest(int32_t a, int32_t b) { return abs(b) > abs(a) ? b : a; }
static void pads_merge(void) {
    g_pad_btn = 0; g_pad_lx = g_pad_ly = g_pad_rx = g_pad_ry = 0;
    for (int i = 0; i < NPAD; i++) if (g_pads[i].dev) { g_pad_btn |= g_pads[i].btn | g_pads[i].hat;
        g_pad_lx = strongest(g_pad_lx, g_pads[i].lx); g_pad_ly = strongest(g_pad_ly, g_pads[i].ly); g_pad_rx = strongest(g_pad_rx, g_pads[i].rx); g_pad_ry = strongest(g_pad_ry, g_pads[i].ry); }
    dm_pad_set(g_pad_btn, g_pad_lx, g_pad_ly, g_pad_rx, g_pad_ry);
}
static uint64_t key_bit(int32_t kc) {
    switch (kc) {   // face buttons by letter, as the controller labels them (A = the bottom button on Xbox-style pads is AKEYCODE_BUTTON_A)
        case AKEYCODE_BUTTON_A: return DM_BTN_B; case AKEYCODE_BUTTON_B: return DM_BTN_A; case AKEYCODE_BUTTON_X: return DM_BTN_Y; case AKEYCODE_BUTTON_Y: return DM_BTN_X;
        case AKEYCODE_BUTTON_L1: return DM_BTN_L; case AKEYCODE_BUTTON_R1: return DM_BTN_R; case AKEYCODE_BUTTON_L2: return DM_BTN_ZL; case AKEYCODE_BUTTON_R2: return DM_BTN_ZR;
        case AKEYCODE_BUTTON_THUMBL: return DM_BTN_LS; case AKEYCODE_BUTTON_THUMBR: return DM_BTN_RS; case AKEYCODE_BUTTON_START: return DM_BTN_PLUS; case AKEYCODE_BUTTON_SELECT: return DM_BTN_MINUS;
        case AKEYCODE_DPAD_UP: return DM_BTN_UP; case AKEYCODE_DPAD_DOWN: return DM_BTN_DOWN; case AKEYCODE_DPAD_LEFT: return DM_BTN_LEFT; case AKEYCODE_DPAD_RIGHT: return DM_BTN_RIGHT;
        // a keyboard (a phone's Bluetooth keyboard, the Android Emulator's PC keyboard): Enter / Space = A, Esc / Backspace = B, X / Y / Q / E / Z / C / Tab / M as named,
        // arrows = D-pad (they arrive as DPAD keys above); WASD = the left stick (key_stick below)
        case AKEYCODE_ENTER: case AKEYCODE_NUMPAD_ENTER: case AKEYCODE_SPACE: case AKEYCODE_DPAD_CENTER: return DM_BTN_A;
        case AKEYCODE_ESCAPE: case AKEYCODE_DEL: return DM_BTN_B;
        case AKEYCODE_X: return DM_BTN_X; case AKEYCODE_Y: return DM_BTN_Y; case AKEYCODE_Q: return DM_BTN_L; case AKEYCODE_E: return DM_BTN_R;
        case AKEYCODE_Z: return DM_BTN_ZL; case AKEYCODE_C: return DM_BTN_ZR; case AKEYCODE_TAB: return DM_BTN_PLUS; case AKEYCODE_M: return DM_BTN_MINUS;
    }
    return 0;
}
static int32_t axis(float v) { if (fabsf(v) < 0.12f) return 0; float s = (fabsf(v) - 0.12f) / 0.88f; if (s > 1) s = 1; return (int32_t)(copysignf(s, v) * DM_STICK_MAX); }
// Touch: the on-screen controls (touch_ui.c)
void touch_ui_init(ANativeActivity *a);
void touch_ui_event(const AInputEvent *e, float w, float h, uint64_t *btn, int32_t s[4]);
void touch_ui_hide(void);
static uint64_t g_touch_btn; static int32_t g_touch_s[4];
static void touch_update(AInputEvent *e) {
    float w = (float)ANativeWindow_getWidth(g_app->window), h = (float)ANativeWindow_getHeight(g_app->window);
    touch_ui_event(e, w, h, &g_touch_btn, g_touch_s);
    dm_input_set(g_touch_btn, g_touch_s[0], g_touch_s[1], g_touch_s[2], g_touch_s[3]);
}
static int32_t on_input(struct android_app *app, AInputEvent *e) {
    (void)app; int32_t src = AInputEvent_getSource(e), type = AInputEvent_getType(e);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        int32_t kc = AKeyEvent_getKeyCode(e); uint64_t b = key_bit(kc);
        if (kc == AKEYCODE_BACK && !b) { uint64_t m = AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN ? DM_BTN_B : 0; dm_input_set(g_touch_btn | m, g_touch_s[0], g_touch_s[1], g_touch_s[2], g_touch_s[3]); return 1; }   // the system back = B (the game never closes on back)
        static uint8_t wasd; int wb = kc == AKEYCODE_W ? 1 : kc == AKEYCODE_S ? 2 : kc == AKEYCODE_A ? 4 : kc == AKEYCODE_D ? 8 : 0;
        if (wb) {   // WASD: the left stick, full tilt
            if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN) wasd |= wb; else if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP) wasd &= ~wb;
            int k = pad_slot(AInputEvent_getDeviceId(e));
            g_pads[k].lx = (wasd & 8 ? DM_STICK_MAX : 0) - (wasd & 4 ? DM_STICK_MAX : 0); g_pads[k].ly = (wasd & 1 ? DM_STICK_MAX : 0) - (wasd & 2 ? DM_STICK_MAX : 0);
            pads_merge(); return 1;
        }
        if (!b) return 0;
        int k = pad_slot(AInputEvent_getDeviceId(e));
        if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN) g_pads[k].btn |= b; else if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP) g_pads[k].btn &= ~b;
        touch_ui_hide(); pads_merge(); return 1;
    }
    if (type == AINPUT_EVENT_TYPE_MOTION && (src & (AINPUT_SOURCE_JOYSTICK | AINPUT_SOURCE_GAMEPAD))) {
        int k = pad_slot(AInputEvent_getDeviceId(e));
        float rx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Z, 0), ry = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RZ, 0);
        if (rx == 0 && ry == 0) { rx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RX, 0); ry = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RY, 0); }   // right stick as RX / RY on some drivers
        g_pads[k].lx = axis(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_X, 0)); g_pads[k].ly = axis(-AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Y, 0));
        g_pads[k].rx = axis(rx); g_pads[k].ry = axis(-ry);
        float hx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0), hy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
        float lt = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_LTRIGGER, 0), rt = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RTRIGGER, 0);
        if (lt == 0) lt = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_BRAKE, 0); if (rt == 0) rt = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_GAS, 0);
        touch_ui_hide();
        g_pads[k].hat = (hx < -0.5f ? DM_BTN_LEFT : 0) | (hx > 0.5f ? DM_BTN_RIGHT : 0) | (hy < -0.5f ? DM_BTN_UP : 0) | (hy > 0.5f ? DM_BTN_DOWN : 0) | (lt > 0.5f ? DM_BTN_ZL : 0) | (rt > 0.5f ? DM_BTN_ZR : 0);
        pads_merge(); return 1;
    }
    // touch, and a mouse as touch (the Android Emulator's window, a mouse on a phone): its button press / release arrive as DOWN / UP; hover moves are ignored
    if (type == AINPUT_EVENT_TYPE_MOTION && ((src & AINPUT_SOURCE_TOUCHSCREEN) == AINPUT_SOURCE_TOUCHSCREEN || (src & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE)) {
        int32_t act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK;
        if (act == AMOTION_EVENT_ACTION_HOVER_MOVE || act == AMOTION_EVENT_ACTION_HOVER_ENTER || act == AMOTION_EVENT_ACTION_HOVER_EXIT || act == AMOTION_EVENT_ACTION_SCROLL ||
            act == AMOTION_EVENT_ACTION_BUTTON_PRESS || act == AMOTION_EVENT_ACTION_BUTTON_RELEASE) return 1;
        static int said; if (!said++) fprintf(stderr, "[dm] input: first pointer event (source %#x, action %d)\n", src, act);
        touch_update(e); return 1; }
    return 0;
}

// ---------- Host start ----------
static int g_started;
// Device profiles: defaults by chip, from its model (ro.soc.manufacturer / ro.soc.model, Android 12+: SM8750, MT6989, s5e9945 ...), as PKShield's. Set without overwriting,
// after env.txt, so the player's own settings always win; DM_PROFILE=<name> picks one by hand, DM_PROFILE=none none. The game already runs in handheld mode (720p) here,
// so a profile sets what is left: anisotropic filtering and texture size (DM_VK_TEX_QUALITY=half: the game's large textures without their largest mip level, nvn_vk.c).
// GPUs without BC textures (Mali, Xclipse, PowerVR) decode the game's textures on upload to 4 to 8 times their size, so their profiles keep textures at half size sooner.
// Starting points from the chips' standing, not yet from measurements on each:
//   sd8     Snapdragon 8 Gen 1 and later, 8s, 8 Elite (SM8450 and up)       nothing changed: the defaults these phones were tuned on (8x filtering, full textures)
//   sd8old  Snapdragon 865 / 870 / 888 (SM8250-SM8350, Adreno 650 / 660)     4x filtering; full textures
//   sd7     Snapdragon 7 series from 2021 (SM73xx and later)                  4x filtering; full textures
//   sd6     Snapdragon 6 and 4 series, 7 series before 2021, 855 and older    2x filtering; half textures
//   high    Dimensity 9000 series, Exynos 2000 series (Xclipse 9xx)           4x filtering; full textures
//   mid     Dimensity 8000 series and 1000-1200, Google Tensor                4x filtering; half textures
//   low     other MediaTek (Dimensity 7000 / 6000, Helio), Exynos 1000 series and older   2x filtering; half textures
// Other chips get no profile
static const struct prof { const char *name, *what, *aniso, *tex; } g_profiles[] = {
    { "sd8", "Snapdragon 8 series", NULL, NULL }, { "sd8old", "Snapdragon 865 / 870 / 888", "4", "full" }, { "sd7", "Snapdragon 7 series", "4", "full" },
    { "sd6", "Snapdragon 6 / 4 series or older 7 / 8 series", "2", "half" }, { "high", "Dimensity 9000 / Exynos 2000 series", "4", "full" },
    { "mid", "Dimensity 8000 series / Google Tensor", "4", "half" }, { "low", "MediaTek Dimensity 7000 and below / Exynos 1000 series", "2", "half" },
};
static const struct prof *profile_of(const char *maker, const char *model) {
    char m[PROP_VALUE_MAX]; size_t k = 0; for (; model[k] && k + 1 < sizeof m; k++) m[k] = (char)tolower((unsigned char)model[k]); m[k] = 0;
    const char *name = NULL; int num;
    if ((!strncmp(m, "sm", 2) && isdigit((unsigned char)m[2])) || (!strncmp(m, "sdm", 3) && isdigit((unsigned char)m[3]))) {   // Qualcomm
        num = atoi(m + (m[1] == 'd' ? 3 : 2));
        name = num >= 8450 && num < 9000 ? "sd8" : num >= 8250 && num < 8450 ? "sd8old" : num >= 7300 && num < 8000 ? "sd7" : num >= 4000 && num < 8250 ? "sd6" : num > 0 && num < 1000 ? "sd6" : NULL;   // (SDM845 and the like: before the SMxxxx names)
    } else if (!strncmp(m, "mt", 2) && isdigit((unsigned char)m[2])) {   // MediaTek: MT6983 / 6985 / 6989 / 6991 / 6993 Dimensity 9000-9500; MT6889-6899 Dimensity 1000-1200 and 8000-8400
        num = atoi(m + 2); name = num == 6983 || num == 6985 || num == 6989 || num == 6991 || num == 6993 ? "high" : num >= 6889 && num <= 6899 ? "mid" : "low";
    } else if (!strncmp(m, "s5e", 3) || strstr(m, "exynos")) {   // Exynos: s5e99xx and "Exynos 2xxx" are the 2000 series; s5e88xx, Exynos 1xxx and 9xx the rest
        const char *d = m; while (*d && !isdigit((unsigned char)*d)) d++; if (!strncmp(m, "s5e", 3)) d = m + 3;
        num = atoi(d); name = (!strncmp(m, "s5e", 3) && num >= 9900 && num < 10000) || (strncmp(m, "s5e", 3) && num >= 2000 && num < 3000) ? "high" : "low";
    } else if (strstr(m, "tensor") || (!strncasecmp(maker, "google", 6) && *m)) name = "mid";
    for (unsigned i = 0; name && i < sizeof g_profiles / sizeof *g_profiles; i++) if (!strcmp(g_profiles[i].name, name)) return &g_profiles[i];
    return NULL;
}
static void device_profile(void) {
    char maker[PROP_VALUE_MAX] = "", model[PROP_VALUE_MAX] = ""; __system_property_get("ro.soc.manufacturer", maker); __system_property_get("ro.soc.model", model);
    const char *force = getenv("DM_PROFILE"); const struct prof *p = NULL;
    if (force) { for (unsigned i = 0; i < sizeof g_profiles / sizeof *g_profiles; i++) if (!strcmp(g_profiles[i].name, force)) p = &g_profiles[i]; }
    else p = profile_of(maker, model);
    if (!p) { fprintf(stderr, "boot: device profile none (chip %s %s)\n", *maker ? maker : "?", *model ? model : "unknown"); return; }
    if (!p->aniso) { fprintf(stderr, "boot: device profile %s, %s (chip %s %s): the defaults, nothing changed\n", p->name, p->what, *maker ? maker : "?", *model ? model : "unknown"); return; }
    setenv("DM_VK_MAX_ANISO", p->aniso, 0); setenv("DM_VK_TEX_QUALITY", p->tex, 0);
    fprintf(stderr, "boot: device profile %s, %s (chip %s %s%s): anisotropic filtering %sx, textures %s\n", p->name, p->what, *maker ? maker : "?", *model ? model : "unknown",
            force ? ", DM_PROFILE" : "", getenv("DM_VK_MAX_ANISO"), getenv("DM_VK_TEX_QUALITY"));
}
static void env_file(const char *path) {   // KEY=VALUE lines, # comments; set only when not already set
    FILE *f = fopen(path, "r"); char ln[512]; if (!f) return;
    while (fgets(ln, sizeof ln, f)) { char *p = ln; while (*p == ' ' || *p == '\t') p++; if (*p == '#' || !*p) continue; char *eq = strchr(p, '='); if (!eq || eq == p) continue;
        *eq = 0; char *v = eq + 1; char *e = strpbrk(v, "\r\n"); if (e) *e = 0; if (!getenv(p)) { setenv(p, v, 0); fprintf(stderr, "boot: env %s=%s\n", p, v); } }
    fclose(f);
}
// "Lower specs" (the settings' DM_LOWSPEC: 1 lower, 2 lowest), for phones that cannot hold the game's frame rate: the POTATO mod's RomFS files, kept in this library
//   (lowspec_files.h) and written to <files>/lowspec on start, are laid over the game's RomFS (hle_fs.c DM_ROMFS_OVERLAY, first root wins): a shorter draw distance for
//   Pokémon, people and trees (bin/misc/app_config/field_common_config.bin) and lighter shadows, at 1 fewer (Low Shadows), at 2 none (No Shadows: field_shadow_config.bin)
#include "lowspec_files.h"
static int write_all(const char *path, const unsigned char *b, size_t n) {
    char d[1200]; snprintf(d, sizeof d, "%s", path);
    for (char *s = d + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(d, 0770); *s = '/'; }
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    int ok = fwrite(b, 1, n, f) == n; return fclose(f) == 0 && ok;
}
static void lowspec(void) {
    const char *e = getenv("DM_LOWSPEC"); int lv = e ? atoi(e) : 0; if (lv < 1 || lv > 2) return;
    char root[1100], f[1300]; snprintf(root, sizeof root, "%s/lowspec/romfs", g_files);
    snprintf(f, sizeof f, "%s/bin/misc/app_config/field_common_config.bin", root);
    int ok = write_all(f, LS_draw_distance, sizeof LS_draw_distance);
    snprintf(f, sizeof f, "%s/bin/misc/app_config/field_shadow_config.bin", root);
    ok &= lv == 2 ? write_all(f, LS_no_shadows, sizeof LS_no_shadows) : write_all(f, LS_low_shadows, sizeof LS_low_shadows);
    if (!ok) { fprintf(stderr, "boot: [RED] lower specs: could not write %s\n", root); return; }
    const char *o = getenv("DM_ROMFS_OVERLAY"); char all[4096];
    snprintf(all, sizeof all, "%s%s%s", root, o && *o ? ":" : "", o && *o ? o : "");   // (ahead of any other overlay)
    setenv("DM_ROMFS_OVERLAY", all, 1);
    fprintf(stderr, "boot: lower specs %d: shorter draw distance, %s (RomFS overlay %s)\n", lv, lv == 2 ? "no shadows" : "lighter shadows", root);
}
static pid_t g_game_tid;   // the game's main thread (performance hints below)
static void *host_thread(void *a) {
    g_game_tid = gettid();
    (void)a; static char alt[65536]; stack_t ss = {.ss_sp = alt, .ss_size = sizeof alt}; sigaltstack(&ss, 0);
    char patched[1100]; snprintf(patched, sizeof patched, "%s/patched", g_game);
    char *argv[3] = { "dmhost", patched, NULL };
    int rc = dm_main(2, argv);
    fprintf(stderr, "boot: dm_main returned %d\n", rc);
    return NULL;
}
// The log where the player can reach it: Android/data is out of reach of the phone's Files app on recent Android versions, Download is not. At each start the previous
// run's log (the one that ended in a crash, if one did) goes to Download/PokemonShield-log-previous.txt; while playing, Download/PokemonShield-log.txt is refreshed every
// 5 seconds, with a memory line (a run the system ends for memory leaves no crash report, only memory that kept rising)
#define DL "/storage/emulated/0/Download"
static void copy_file(const char *a, const char *b) {
    int i = open(a, O_RDONLY), o = i < 0 ? -1 : open(b, O_WRONLY | O_CREAT | O_TRUNC, 0644); char buf[65536]; ssize_t n;
    if (o >= 0) while ((n = read(i, buf, sizeof buf)) > 0) if (write(o, buf, (size_t)n) != n) break;
    if (i >= 0) close(i); if (o >= 0) close(o);
}
static long kb_of(const char *path, const char *key) {
    FILE *f = fopen(path, "r"); char ln[256]; long v = -1; size_t kl = strlen(key);
    while (f && fgets(ln, sizeof ln, f)) if (!strncmp(ln, key, kl)) { v = atol(ln + kl); break; }
    if (f) fclose(f); return v;
}
// The phone's heat (Android 11+ thermal status, 12+ headroom: 1.0 = where throttling starts), looked up at run time, for the log line: a slowdown from heat shows as heat
static const char *thermal_text(char *buf, size_t n) {
    static void *mgr; static int (*st)(void *); static float (*hr)(void *, int); static int tried;
    if (!tried++) { void *lib = dlopen("libandroid.so", RTLD_NOW); void *(*acq)(void) = lib ? (void *(*)(void))dlsym(lib, "AThermal_acquireManager") : NULL;
        st = lib ? (int (*)(void *))dlsym(lib, "AThermal_getCurrentThermalStatus") : NULL; hr = lib ? (float (*)(void *, int))dlsym(lib, "AThermal_getThermalHeadroom") : NULL; mgr = acq ? acq() : NULL; }
    if (!mgr || !st) return "unknown";
    static const char *names[] = { "none", "light", "moderate", "severe", "critical", "emergency", "shutdown" }; int v = st(mgr);
    snprintf(buf, n, "%s, headroom %.2f", v >= 0 && v <= 6 ? names[v] : "?", hr ? hr(mgr, 10) : -1.0f);   // headroom forecast 10 s ahead
    return buf;
}
static void *log_mirror(void *p) {
    (void)p; char src[1200]; snprintf(src, sizeof src, "%s/log.txt", g_files);
    for (int n = 0;; n++) {
        sleep(5);
        char th[64];
        if (n % 6 == 0) fprintf(stderr, "[mem] app %ld MB resident (%ld MB swapped), phone %ld MB available of %ld MB; heat %s\n", kb_of("/proc/self/status", "VmRSS:") / 1024,
                                kb_of("/proc/self/status", "VmSwap:") / 1024, kb_of("/proc/meminfo", "MemAvailable:") / 1024, kb_of("/proc/meminfo", "MemTotal:") / 1024, thermal_text(th, sizeof th));
        copy_file(src, DL "/PokemonShield-log.txt");
    }
    return NULL;
}
// ---------- Memory inspector (development): request files in the files folder, answered every second while the game runs ----------
//   scan.req   lines of 64-bit hex values: every 8-byte-aligned occurrence in the process's private writable memory → scan.out (address, value)
//   peek.req   lines "<hex address> <hex length>": the bytes, as hex lines of 32 → peek.out
//   tree.req   lines "<hex node> <hex bytes per node>": the UI node tree under the node (first child at +0x98, next sibling at +0x90; name hash at +0x68),
//              each node's bytes as in peek.out, depth-first, at most 4000 nodes → tree.out
// Reads go through process_vm_readv on this process, so a bad address is an error, not a crash. Used to map the game's UI node trees (the touch driver)
#include <sys/uio.h>
static ssize_t safe_read(uintptr_t a, void *buf, size_t n) {
    struct iovec l = { buf, n }, r = { (void *)a, n }; return process_vm_readv(getpid(), &l, 1, &r, 1, 0);
}
static void inspector_scan(FILE *req, FILE *out) {
    uint64_t want[64]; int nw = 0; char ln[128];
    while (nw < 64 && fgets(ln, sizeof ln, req)) if (*ln != '#' && *ln > ' ') want[nw++] = strtoull(ln, NULL, 16);
    FILE *m = fopen("/proc/self/maps", "r"); static uint64_t buf[1 << 17]; long hits = 0;
    while (m && fgets(ln, sizeof ln, m) && hits < 20000) { unsigned long s, e; char pr[8] = "";
        if (sscanf(ln, "%lx-%lx %7s", &s, &e, pr) != 3 || pr[0] != 'r' || pr[1] != 'w' || pr[3] != 'p') continue;
        for (unsigned long a = s; a < e; a += sizeof buf) { size_t n = e - a < sizeof buf ? e - a : sizeof buf; if (safe_read(a, buf, n) != (ssize_t)n) continue;
            for (size_t i = 0; i < n / 8; i++) for (int k = 0; k < nw; k++) if (buf[i] == want[k]) { fprintf(out, "%lx %016llx\n", a + i * 8, (unsigned long long)want[k]); hits++; } } }
    if (m) fclose(m); fprintf(out, "# %ld hits\n", hits);
}
static void inspector_peek(FILE *req, FILE *out) {
    char ln[128]; unsigned char b[32];
    while (fgets(ln, sizeof ln, req)) { unsigned long a, n; if (sscanf(ln, "%lx %lx", &a, &n) != 2) continue; if (n > 1 << 20) n = 1 << 20;
        fprintf(out, "@ %lx %lx\n", a, n);
        for (unsigned long o = 0; o < n; o += 32) { size_t k = n - o < 32 ? n - o : 32; if (safe_read(a + o, b, k) != (ssize_t)k) { fprintf(out, "%lx unreadable\n", a + o); continue; }
            fprintf(out, "%lx", a + o); for (size_t i = 0; i < k; i++) fprintf(out, "%s%02x", i % 8 ? "" : " ", b[i]); fputc('\n', out); } }
}
static void tree_walk(FILE *out, uintptr_t n, unsigned long per, int depth, int *count) {
    for (int sib = 0; n && *count < 4000 && sib < 2000; sib++) {
        uint64_t h = 0, child = 0, next = 0; unsigned char b[32];
        if (safe_read(n + 0x68, &h, 8) != 8 || safe_read(n + 0x98, &child, 8) != 8 || safe_read(n + 0x90, &next, 8) != 8) { fprintf(out, "%*s! %lx unreadable\n", depth, "", (unsigned long)n); return; }
        (*count)++; fprintf(out, "%*s@ %lx name %016llx depth %d\n", depth, "", (unsigned long)n, (unsigned long long)h, depth);
        for (unsigned long o = 0; o < per; o += 32) { if (safe_read(n + o, b, 32) != 32) break;
            fprintf(out, "%lx", (unsigned long)(n + o)); for (int i = 0; i < 32; i++) fprintf(out, "%s%02x", i % 8 ? "" : " ", b[i]); fputc('\n', out); }
        if (child && depth < 40) tree_walk(out, (uintptr_t)child, per, depth + 1, count);
        n = (uintptr_t)next;
    }
}
static void inspector_tree(FILE *req, FILE *out) {
    char ln[128]; while (fgets(ln, sizeof ln, req)) { unsigned long a, per; if (sscanf(ln, "%lx %lx", &a, &per) != 2) continue; if (per > 0x1000) per = 0x1000;
        int count = 0; fprintf(out, "# tree %lx\n", a); tree_walk(out, a, per, 0, &count); fprintf(out, "# %d nodes\n", count); }
}
// poke.req: lines "<hex address> <hex bytes>" written into the process (debugging: flip a byte in a live game object); bad addresses fail, not crash → poke.out
static void inspector_poke(FILE *req, FILE *out) {
    char ln[1200]; while (fgets(ln, sizeof ln, req)) { unsigned long a; char hex[1024]; if (sscanf(ln, "%lx %1023s", &a, hex) != 2) continue;
        uint8_t b[512]; size_t n = 0; for (const char *h = hex; h[0] && h[1] && n < sizeof b; h += 2) { unsigned v; if (sscanf(h, "%2x", &v) != 1) break; b[n++] = (uint8_t)v; }
        struct iovec l = { b, n }, r = { (void *)a, n }; ssize_t w = process_vm_writev(getpid(), &l, 1, &r, 1, 0);
        fprintf(out, "%lx %zu bytes: %s\n", a, n, w == (ssize_t)n ? "written" : "failed"); }
}
static void *inspector_main(void *p) {
    (void)p; char q[1200], o[1200], t[1200];
    for (;;) { sleep(1);
        static const char *kind[4] = { "scan", "peek", "tree", "poke" };
        for (int k = 0; k < 4; k++) { snprintf(q, sizeof q, "%s/%s.req", g_files, kind[k]); FILE *r = fopen(q, "r"); if (!r) continue;
            snprintf(t, sizeof t, "%s/%s.tmp", g_files, kind[k]); snprintf(o, sizeof o, "%s/%s.out", g_files, kind[k]); FILE *w = fopen(t, "w");
            if (w) { if (k == 3) inspector_poke(r, w); else if (k == 2) inspector_tree(r, w); else if (k) inspector_peek(r, w); else inspector_scan(r, w); fclose(w); rename(t, o); } fclose(r); unlink(q); } }
    return NULL;
}
// A custom Vulkan driver (a Qualcomm driver package for Adreno 8xx, Mesa Turnip): its files (meta.json and the library, as the package's zip holds them) put in
// <files>/driver/ are copied into the app's private storage, where libadrenotools loads drivers from, and named in DM_VK_DRIVER (nvn_vk.c vk_open; env.txt wins).
// No driver folder: the phone's own driver
static void driver_install(void) {
    char src[1200], dst[1200], a[1500], b[1500], lib[256] = ""; const char *in = g_app->activity->internalDataPath;
    snprintf(src, sizeof src, "%s/driver", g_files); DIR *d = opendir(src); if (!d || !in) { if (d) closedir(d); return; }
    snprintf(dst, sizeof dst, "%s/driver", in); mkdir(dst, 0700);
    struct dirent *e; int n = 0;
    while ((e = readdir(d))) { if (e->d_name[0] == '.') continue; snprintf(a, sizeof a, "%s/%s", src, e->d_name); snprintf(b, sizeof b, "%s/%s", dst, e->d_name);
        struct stat sa, sb; if (stat(a, &sa) || !S_ISREG(sa.st_mode)) continue;
        if (stat(b, &sb) || sb.st_size != sa.st_size || sb.st_mtime < sa.st_mtime) copy_file(a, b);   // changed or new
        n++;
        size_t l = strlen(e->d_name); if (!lib[0] && l > 3 && !strcmp(e->d_name + l - 3, ".so")) snprintf(lib, sizeof lib, "%s", e->d_name); }   // (no meta.json: its one library)
    closedir(d);
    snprintf(a, sizeof a, "%s/meta.json", dst); FILE *m = fopen(a, "r");
    if (m) { char j[4096]; size_t k = fread(j, 1, sizeof j - 1, m); j[k] = 0; fclose(m);
        char *p = strstr(j, "\"libraryName\""); if (p && (p = strchr(p + 13, '"'))) { char *q = strchr(p + 1, '"'); if (q && q - p - 1 < (long)sizeof lib) { memcpy(lib, p + 1, q - p - 1); lib[q - p - 1] = 0; } } }
    snprintf(a, sizeof a, "%s/%s", dst, lib);
    struct stat ls; if (!lib[0] || stat(a, &ls)) { fprintf(stderr, "boot: [RED] %s has no driver library (meta.json's libraryName, or one .so): the phone's own driver\n", src); return; }
    setenv("DM_VK_DRIVER", a, 0); fprintf(stderr, "boot: custom Vulkan driver %s (%d files from %s)\n", getenv("DM_VK_DRIVER"), n, src);
}
static void host_start(void) {
    if (g_started++) return;
    const char *f = g_app->activity->externalDataPath; snprintf(g_files, sizeof g_files, "%s", f ? f : "/sdcard/nxnative");
    mkdir(g_files, 0755);
    char p[1200], q[1200];
    snprintf(p, sizeof p, "%s/log.txt", g_files); snprintf(q, sizeof q, "%s/log.prev.txt", g_files); rename(p, q);
    if (freopen(p, "w", stderr)) setvbuf(stderr, NULL, _IOLBF, 0);
    copy_file(q, DL "/PokemonShield-log-previous.txt");
    { pthread_t lt; if (!pthread_create(&lt, NULL, log_mirror, NULL)) pthread_detach(lt); }
    { pthread_t it; if (!pthread_create(&it, NULL, inspector_main, NULL)) pthread_detach(it); }   // the memory inspector (scan.req / peek.req)
    snprintf(p, sizeof p, "%s/env.txt", g_files); env_file(p);
    device_profile();   // the chip tier's defaults, under env.txt
    lowspec();
    driver_install();
    { char sq[1100]; snprintf(sq, sizeof sq, "%s/shot", g_files); setenv("DM_SHOT_REQ", sq, 0); }   // testing: <files>/shot.req -> <files>/shot.png (nvn_vk.c)
    // Game files: inside the installed APKs (the all-in-one .apks bundle, read through apkfs.c as /nxapk/...), else copied into the files folder
    struct stat st; int inapk = !stat("/nxapk/RomFS", &st);
    snprintf(g_game, sizeof g_game, "%s", inapk ? "/nxapk" : g_files);
    #define SETG(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_game, sub); setenv(k, p, 0); } while (0)
    #define SETP(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_files, sub); setenv(k, p, 0); } while (0)
    setenv("DM_AUDIO", "1", 0);
#if defined(DM_GAME_ANDROID_OPMODE) && DM_GAME_ANDROID_OPMODE >= 0
    { char m[4]; snprintf(m, sizeof m, "%d", DM_GAME_ANDROID_OPMODE); setenv("DM_OPMODE", m, 0); }   // the profile's android_op_mode (env.txt DM_OPMODE wins)
#endif   // real sound output (AAudio, compat/aaudio_unit.c); env.txt can set 0
    // pipelines: every one the game makes is recorded here and built ahead at the next start, the list shipped with the game too, and the driver's compiled code kept
    SETP("DM_VK_PSO_RECORD", "vk_pipelines.bin"); SETP("DM_VK_PCACHE", "vk_pipeline_cache.bin"); SETG("DM_VK_PSO_SEED", "vk_pipelines_seed.bin");
    snprintf(q, sizeof q, "%s/shaders.pak", g_game); if (!stat(q, &st)) setenv("DM_SPV_PAK", q, 0);   // the packed shader set, else vk/b-<chain>/
    { snprintf(p, sizeof p, "%s/aoc", g_game); struct stat as; if (!stat(p, &as)) setenv("DM_AOC_DIR", p, 0); }   // the DLC the player owns (Shield: the Expansion Pass keys; its content is in the update)
    SETG("DM_ROMFS", "RomFS"); SETG("DM_SPV_DIR", "vk"); SETG("DM_NVN_SHMAP", "shader_map.tsv"); SETG("DM_FONT_DIR", "font");
    SETP("DM_SAVE", "save"); SETP("DM_TEMP", "temp");
    SETP("DM_NVN_CAPTURE", "nvn_capture");   // shaders the pack lacks, saved the first time the game hands them over (nvn_vk.c mtl_shader_seen), for translation on the PC
    // a save brought in from elsewhere (a Switch dump): files put in save_import/ (adb can write there, not into the app's own save/) replace the save at start;
    // the old save is kept as save.before-import-<time>
    { char imp[1200], old[1200], src[1300], dst[1300]; snprintf(imp, sizeof imp, "%s/save_import", g_files); struct stat is;
      if (!stat(imp, &is) && S_ISDIR(is.st_mode)) {
          snprintf(p, sizeof p, "%s/save", g_files); snprintf(old, sizeof old, "%s/save.before-import-%ld", g_files, (long)time(NULL)); rename(p, old); mkdir(p, 0755);
          DIR *d = opendir(imp); struct dirent *de; int n = 0;
          while (d && (de = readdir(d))) { if (de->d_name[0] == '.') continue; snprintf(src, sizeof src, "%s/%s", imp, de->d_name); snprintf(dst, sizeof dst, "%s/%s", p, de->d_name);
              copy_file(src, dst); unlink(src); n++; }
          if (d) closedir(d); rmdir(imp);
          fprintf(stderr, "boot: save imported from save_import (%d files); the previous save is in %s\n", n, old); } }
    snprintf(p, sizeof p, "%s/save", g_files); mkdir(p, 0755); snprintf(p, sizeof p, "%s/temp", g_files); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/RomFS", g_game);
    if (stat(p, &st)) { fprintf(stderr, "boot: game files not found: %s (install the bundle, or copy RomFS, patched, shaders.pak, shader_map.tsv and font there)\n", p); ALOG("game files missing in %s", g_files); return; }
    chdir(g_files);
    fprintf(stderr, "boot: game files %s%s\n", g_game, inapk ? " (inside the installed APKs)" : "");
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 64u << 20);
    pthread_t t; pthread_create(&t, &at, host_thread, NULL); pthread_detach(t);
}
// ---------- Performance hints (Android 13+): one session for the game's main thread and the presenting thread, told every frame how long the presenting thread
// worked against the frame's period (nvn_vk.c mtl_present). That is what brings the fastest cores and higher clocks when the game needs them, and lower ones
// when it does not (LAS Recomp's Android port: the prime core otherwise stays parked). The presenting thread also gets the most urgent priority an app may set
static void *g_hint; static int g_hint_failed; static int64_t g_hint_target;
static void *(*p_hint_mgr)(void); static void *(*p_hint_create)(void *, const int32_t *, size_t, int64_t);
static int (*p_hint_target)(void *, int64_t); static int (*p_hint_report)(void *, int64_t); static int (*p_hint_threads)(void *, const int32_t *, size_t);
// The game's busiest threads (the presenting thread is often not the one with the most work: in Legends: Arceus another game thread used a whole core at normal priority
// while the presenting one used 40%): every 2 s the threads' CPU time is read from /proc, and those that used over 45% of a core get the presenting thread's priority and
// join the hint session (APerformanceHint_setThreads, Android 14+), so the scheduler keeps them on the fast cores. At most HOT_MAX besides the presenting thread
enum { HOT_MAX = 3 };
static int32_t g_hot[HOT_MAX]; static int g_nhot;
// The fastest cores (the Snapdragon 8 Elite's two prime Oryon cores at 4.3 GHz, CPUs 6 and 7; on other chips whichever cores have the highest maximum clock): the
// presenting thread and the busiest game thread are pinned there, the rest left to the scheduler. Found from cpufreq, not core numbers. DM_CPU_PIN=0: no pinning
static cpu_set_t g_prime; static int g_nprime = -1;
static void prime_cores(void) {
    if (g_nprime >= 0) return;
    g_nprime = 0; CPU_ZERO(&g_prime);
    const char *e = getenv("DM_CPU_PIN"); if (e && !strcmp(e, "0")) { fprintf(stderr, "[cpu] DM_CPU_PIN=0: threads not pinned\n"); return; }
    long mf[64]; int n = 0; long top = 0, low = -1;
    for (int c = 0; c < 64; c++) { char pth[96]; snprintf(pth, sizeof pth, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c); FILE *f = fopen(pth, "r");
        mf[c] = 0; if (f) { if (fscanf(f, "%ld", &mf[c]) != 1) mf[c] = 0; fclose(f); }
        if (mf[c] > 0) { n = c + 1; if (mf[c] > top) top = mf[c]; if (low < 0 || mf[c] < low) low = mf[c]; } }
    if (!top || top == low) { fprintf(stderr, "[cpu] all cores alike: threads not pinned\n"); return; }   // one cluster: nothing to choose
    for (int c = 0; c < n; c++) if (mf[c] == top) { CPU_SET(c, &g_prime); g_nprime++; }
    fprintf(stderr, "[cpu] fastest cores: %d at %.2f GHz (", g_nprime, top / 1e6); for (int c = 0; c < n; c++) if (mf[c] == top) fprintf(stderr, " %d", c); fprintf(stderr, " )\n");
}
static void pin_prime(int32_t tid, const char *what) {
    prime_cores(); if (g_nprime <= 0) return;
    if (sched_setaffinity(tid, sizeof g_prime, &g_prime)) fprintf(stderr, "[cpu] %s thread %d: pinning failed (%s)\n", what, tid, strerror(errno));
    else fprintf(stderr, "[cpu] %s thread %d pinned to the fastest cores\n", what, tid);
}
static void hot_threads(void) {
    static struct { int32_t tid; uint64_t t; } prev[256]; static int nprev; static uint64_t last_ns;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    if (last_ns && now - last_ns < 2000000000ull) return;
    double span = last_ns ? (now - last_ns) / 1e9 : 0; last_ns = now;
    DIR *d = opendir("/proc/self/task"); if (!d) return;
    static struct { int32_t tid; uint64_t t; } cur[256]; int ncur = 0; struct dirent *de; long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    int32_t best[HOT_MAX]; double bu[HOT_MAX]; int nb = 0;
    while ((de = readdir(d)) && ncur < 256) {
        int32_t tid = atoi(de->d_name); if (tid <= 0) continue;
        char pth[64], buf[512]; snprintf(pth, sizeof pth, "/proc/self/task/%d/stat", tid); FILE *f = fopen(pth, "r"); if (!f) continue;
        size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
        char *q = strrchr(buf, ')'); if (!q) continue;   // fields after the name: state ppid ... utime (14) stime (15)
        unsigned long ut = 0, st = 0; if (sscanf(q + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &ut, &st) != 2) continue;
        uint64_t t = ut + st; cur[ncur].tid = tid; cur[ncur].t = t; ncur++;
        if (!span || tid == gettid()) continue;
        { char cp[64], cm[32] = ""; snprintf(cp, sizeof cp, "/proc/self/task/%d/comm", tid); FILE *cf = fopen(cp, "r"); if (cf) { if (!fgets(cm, sizeof cm, cf)) cm[0] = 0; fclose(cf); }
          if (!strncmp(cm, "dm-", 3)) continue; }   // the app's own helpers (the touch-menu watcher, the sampler): not the game's threads
        for (int i = 0; i < nprev; i++) if (prev[i].tid == tid) { double u = (double)(t - prev[i].t) / hz / span;
            if (u > 0.45) { int k = nb < HOT_MAX ? nb++ : -1; if (k < 0) { k = 0; for (int j = 1; j < HOT_MAX; j++) if (bu[j] < bu[k]) k = j; if (bu[k] >= u) break; }
                best[k] = tid; bu[k] = u; }
            break; }
    }
    closedir(d);
    memcpy(prev, cur, sizeof cur[0] * (size_t)ncur); nprev = ncur;
    if (!nb) return;
    int changed = nb != g_nhot; for (int i = 0; i < nb && !changed; i++) { int seen = 0; for (int j = 0; j < g_nhot; j++) if (g_hot[j] == best[i]) seen = 1; if (!seen) changed = 1; }
    if (!changed) return;
    for (int i = 0; i < nb; i++) { int seen = 0; for (int j = 0; j < g_nhot; j++) if (g_hot[j] == best[i]) seen = 1;
        if (!seen) { setpriority(PRIO_PROCESS, (id_t)best[i], -10); fprintf(stderr, "[cpu] busy game thread %d (%.0f%% of a core): priority -10, in the hint session\n", best[i], bu[i] * 100); } }
    { int bi = 0; for (int i = 1; i < nb; i++) if (bu[i] > bu[bi]) bi = i; static int32_t pinned; if (best[bi] != pinned) { if (pinned && g_nprime > 0) { cpu_set_t all; CPU_ZERO(&all); for (int c = 0; c < CPU_SETSIZE && c < 64; c++) CPU_SET(c, &all); sched_setaffinity(pinned, sizeof all, &all); }   // the one before back to every core
        pin_prime(best[bi], "busiest game"); pinned = best[bi]; } }
    memcpy(g_hot, best, sizeof best[0] * (size_t)nb); g_nhot = nb;
    if (g_hint && p_hint_threads) { int32_t t[HOT_MAX + 1]; size_t n = 0; t[n++] = gettid(); for (int i = 0; i < nb; i++) t[n++] = g_hot[i]; p_hint_threads(g_hint, t, n); }
}
// ---------- Where the game's own code spends its time (DM_PC_SAMPLE=1) ----------
// The app is not profileable, so it samples itself: a thread interrupts the presenting thread and the busiest game threads every 2 ms with SIGPROF, the handler keeps
// the interrupted pc and lr, and every 15 s the most frequent ones go to the log per thread (absolute addresses; the log's "main: base" line turns them into main + offset)
#include <sys/syscall.h>
#include <ucontext.h>
enum { PS_N = 1 << 15 };
typedef struct { uint64_t pc, lr; int32_t tid; } ps_t;
static ps_t g_ps[PS_N]; static _Atomic uint32_t g_psn;
static void on_prof(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si; ucontext_t *uc = uc_; uint32_t i = atomic_fetch_add(&g_psn, 1);
    if (i < PS_N) { g_ps[i].pc = uc->uc_mcontext.pc; g_ps[i].lr = uc->uc_mcontext.regs[30]; g_ps[i].tid = gettid(); }
}
static int ps_cmp(const void *a, const void *b) { const ps_t *x = a, *y = b; if (x->tid != y->tid) return x->tid < y->tid ? -1 : 1; return x->pc < y->pc ? -1 : x->pc > y->pc; }
typedef struct { uint64_t pc, lr; uint32_t n; } ps_top;
static int top_cmp(const void *a, const void *b) { const ps_top *x = a, *y = b; return x->n < y->n ? 1 : x->n > y->n ? -1 : 0; }
static int32_t g_ps_present;
static void *ps_main(void *arg) {
    (void)arg; prctl(PR_SET_NAME, "dm-sampler", 0, 0, 0);
    struct sigaction sa = {.sa_sigaction = on_prof, .sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK}; sigemptyset(&sa.sa_mask); sigaction(SIGPROF, &sa, NULL);
    fprintf(stderr, "[pcs] sampling the presenting thread and the busiest game threads every 2 ms\n");
    static ps_t buf[PS_N]; static ps_top top[4096];
    for (uint64_t tick = 1;; tick++) {
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
        int32_t t[HOT_MAX + 1]; int n = 0; if (g_ps_present) t[n++] = g_ps_present; for (int i = 0; i < g_nhot && n < HOT_MAX + 1; i++) if (g_hot[i] != g_ps_present) t[n++] = g_hot[i];
        for (int i = 0; i < n; i++) syscall(SYS_tgkill, getpid(), t[i], SIGPROF);
        if (tick % 7500) continue;   // 15 s
        uint32_t m = atomic_exchange(&g_psn, 0); if (m > PS_N) m = PS_N; if (!m) continue;
        memcpy(buf, g_ps, sizeof *buf * m); qsort(buf, m, sizeof *buf, ps_cmp);
        for (uint32_t a = 0; a < m;) {   // per thread: its most frequent pcs
            uint32_t b = a; while (b < m && buf[b].tid == buf[a].tid) b++;
            int nt = 0; for (uint32_t i = a; i < b;) { uint32_t j = i; while (j < b && buf[j].pc == buf[i].pc) j++; if (nt < 4096) { top[nt].pc = buf[i].pc; top[nt].lr = buf[i].lr; top[nt].n = j - i; nt++; } i = j; }
            qsort(top, (size_t)nt, sizeof *top, top_cmp);
            char line[4096]; size_t o = (size_t)snprintf(line, sizeof line, "[pcs] thread %d%s: %u samples, top:", buf[a].tid, buf[a].tid == g_ps_present ? " (presenting)" : "", b - a);
            for (int k = 0; k < nt && k < 40 && o < sizeof line - 48; k++) o += (size_t)snprintf(line + o, sizeof line - o, " %llx/%llx*%u", (unsigned long long)top[k].pc, (unsigned long long)top[k].lr, top[k].n);
            fprintf(stderr, "%s\n", line);
            for (int k = 0; k < nt && k < 8; k++) {   // the top ones named: a host library's function (dladdr), or game code (no library: the log's module bases)
                Dl_info di, dl; int h = dladdr((void *)(uintptr_t)top[k].pc, &di) && di.dli_fname, hl = dladdr((void *)(uintptr_t)top[k].lr, &dl) && dl.dli_fname;
                const char *f = h ? strrchr(di.dli_fname, '/') : NULL, *fl = hl ? strrchr(dl.dli_fname, '/') : NULL;
                unsigned long long po = h ? (unsigned long long)((uintptr_t)top[k].pc - (uintptr_t)(di.dli_sname ? di.dli_saddr : di.dli_fbase)) : 0ull;
                unsigned long long lo = hl ? (unsigned long long)((uintptr_t)top[k].lr - (uintptr_t)(dl.dli_sname ? dl.dli_saddr : dl.dli_fbase)) : 0ull;
                fprintf(stderr, "[pcs]   %u x %llx in %s %s+%#llx, called from %s %s+%#llx\n", top[k].n, (unsigned long long)top[k].pc,
                        h ? (f ? f + 1 : di.dli_fname) : "game", h && di.dli_sname ? di.dli_sname : "", po,
                        hl ? (fl ? fl + 1 : dl.dli_fname) : "game", hl && dl.dli_sname ? dl.dli_sname : "", lo); }
            a = b;
        }
    }
    return NULL;
}
void front_frame_work(uint64_t busy_ns, uint64_t target_ns) {
    if (g_hint_failed) return;
    if (!g_hint) {
        static int tried; if (tried++) return;
        setpriority(PRIO_PROCESS, (id_t)gettid(), -10); pin_prime(gettid(), "presenting");
        { const char *e = getenv("DM_PC_SAMPLE"); if (e && atoi(e) > 0) { g_ps_present = gettid(); pthread_t pt; if (!pthread_create(&pt, NULL, ps_main, NULL)) pthread_detach(pt); } }
        void *lib = dlopen("libandroid.so", RTLD_NOW); if (!lib) { g_hint_failed = 1; return; }
        p_hint_mgr = (void *(*)(void))dlsym(lib, "APerformanceHint_getManager");
        p_hint_create = (void *(*)(void *, const int32_t *, size_t, int64_t))dlsym(lib, "APerformanceHint_createSession");
        p_hint_target = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
        p_hint_report = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
        p_hint_threads = (int (*)(void *, const int32_t *, size_t))dlsym(lib, "APerformanceHint_setThreads");
        void *m = p_hint_mgr ? p_hint_mgr() : NULL;
        int32_t t[2]; size_t n = 0; t[n++] = gettid(); if (g_game_tid && g_game_tid != gettid()) t[n++] = g_game_tid;
        g_hint = m && p_hint_create ? p_hint_create(m, t, n, (int64_t)target_ns) : NULL;
        if (!g_hint || !p_hint_report) { g_hint_failed = 1; fprintf(stderr, "[cpu] no performance hint session\n"); return; }
        g_hint_target = (int64_t)target_ns;
        fprintf(stderr, "[cpu] performance hint session: %zu threads, target %.1f ms; presenting thread at priority -10\n", n, target_ns / 1e6);
    }
    if ((int64_t)target_ns != g_hint_target && p_hint_target) p_hint_target(g_hint, g_hint_target = (int64_t)target_ns);
    if (busy_ns) p_hint_report(g_hint, (int64_t)busy_ns);
    hot_threads();
}
// The host's front_run (main.c): the guest already runs on our big-stack host thread
void front_run(void (*guest)(void)) { guest(); }

// The system bars hidden (app/java/com/nxnative/Immersive.java), from the app's event thread (an ordinary thread, attached to Java once and kept attached)
static void immersive(void) {
    static JNIEnv *e; static jclass im; static jmethodID ap; static int failed;
    if (failed) return;
    if (!e && (*g_app->activity->vm)->AttachCurrentThread(g_app->activity->vm, &e, NULL)) { failed = 1; return; }
    if (!im) {
        jobject act = g_app->activity->clazz; jclass ac = (*e)->GetObjectClass(e, act);
        jobject cl = (*e)->CallObjectMethod(e, act, (*e)->GetMethodID(e, ac, "getClassLoader", "()Ljava/lang/ClassLoader;"));
        jclass clc = (*e)->FindClass(e, "java/lang/ClassLoader"); jstring cn = (*e)->NewStringUTF(e, "com.nxnative.Immersive");
        jclass c = (jclass)(*e)->CallObjectMethod(e, cl, (*e)->GetMethodID(e, clc, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;"), cn);
        if ((*e)->ExceptionCheck(e) || !c) { (*e)->ExceptionClear(e); failed = 1; fprintf(stderr, "[RED] immersive: the class is missing\n"); return; }
        im = (jclass)(*e)->NewGlobalRef(e, c); ap = (*e)->GetStaticMethodID(e, im, "apply", "(Landroid/app/Activity;)V");
    }
    (*e)->CallStaticVoidMethod(e, im, ap, g_app->activity->clazz);
    if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionDescribe(e); (*e)->ExceptionClear(e); }
}
// Back from the background: the system may have swapped the game's memory out (Samsung's RAM Plus did, 586 MB), and every swapped page then came back on its own
// fault in the middle of play (battles stuttered: frames of 100–160 ms). One thread asks for all of it back at once, ahead of the game (MADV_WILLNEED on the
// private writable mappings: swapped pages are read back, pages never touched stay untouched)
static void *swapin_main(void *p) {
    (void)p; long before = kb_of("/proc/self/status", "VmSwap:"); FILE *f = fopen("/proc/self/maps", "r"); char ln[512]; double mb = 0;
    while (f && fgets(ln, sizeof ln, f)) { unsigned long s, e; char pr[8] = "";
        if (sscanf(ln, "%lx-%lx %7s", &s, &e, pr) == 3 && pr[0] == 'r' && pr[1] == 'w' && pr[3] == 'p') { madvise((void *)s, e - s, MADV_WILLNEED); mb += (e - s) / 1048576.0; } }
    if (f) fclose(f);
    sleep(3); fprintf(stderr, "[mem] back from the background: swap-in asked over %.0f MB of mappings, swapped %ld → %ld MB\n", mb, before / 1024, kb_of("/proc/self/status", "VmSwap:") / 1024);
    return NULL;
}
static void swapin_start(void) { if (kb_of("/proc/self/status", "VmSwap:") < 32 * 1024) return; pthread_t t; if (!pthread_create(&t, NULL, swapin_main, NULL)) pthread_detach(t); }
__attribute__((weak)) void nvn_vk_cache_save(void);   // the pipeline cache file (nvn_vk.c), written when the app goes to the background
static void on_cmd(struct android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW: atomic_store(&g_window, app->window); immersive(); host_start(); break;
        case APP_CMD_TERM_WINDOW: atomic_store(&g_window, NULL); break;
        case APP_CMD_GAINED_FOCUS: immersive(); if (g_started) { dm_focus_set(1); dm_audio_suspend(0); } break;
        case APP_CMD_LOST_FOCUS: if (g_started) { dm_focus_set(2); } break;
        case APP_CMD_PAUSE: if (g_started) { dm_focus_set(3); dm_audio_suspend(1); if (nvn_vk_cache_save) nvn_vk_cache_save(); } break;
        case APP_CMD_RESUME: if (g_started) { dm_focus_set(1); dm_audio_suspend(0); swapin_start(); } break;
    }
}
void android_main(struct android_app *app) {
    g_app = app; app->onAppCmd = on_cmd; app->onInputEvent = on_input;
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);
    touch_ui_init(app->activity);
    for (;;) {
        int events; struct android_poll_source *src;
        while (ALooper_pollOnce(-1, NULL, &events, (void **)&src) >= 0) { if (src) src->process(app, src); if (app->destroyRequested) { _exit(0); } }
    }
}
