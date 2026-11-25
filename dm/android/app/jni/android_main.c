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
#include <sched.h>
#include <errno.h>
#include <jni.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
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
#include <malloc.h>   // mallopt: the allocator's large-block cache (env.txt read)
#include <sys/system_properties.h>
#include "dm_input.h"
#include "front.h"
#include "game_profile.h"
#include "nvn_ovl.h"   // dm_aspect_mode (the in-game menu's picture setting)

int dm_main(int argc, char **argv);   // dm/host/main.c built with -Dmain=dm_main
void dm_focus_set(int st);
void dm_audio_suspend(int on);
#define ALOG(...) __android_log_print(ANDROID_LOG_INFO, "nxnative", __VA_ARGS__)

static struct android_app *g_app;
static _Atomic(ANativeWindow *) g_window;
static char g_files[1024], g_game[1024];
// The game this start is for: one app and one engine run the games of the profile's family (game.py DM_GAME_FAMILY: Pokémon Shield and Sword, the same engine build).
// The setup screen writes the title ID to <files>/current_game before starting the game; the profile's own game keeps the app's folder (as before: Shield players
// keep their setup and saves), another one has <files>/<title ID>/. Its program ID goes to DM_PROGRAM_ID, its name to the log files
static const struct { const char *id, *stem; } g_family[] = DM_GAME_FAMILY;
static const char *g_stem = DM_GAME_FILE_STEM;
static void choose_game(char *files, size_t n) {
    char p[1100], id[32] = ""; snprintf(p, sizeof p, "%s/current_game", files); FILE *f = fopen(p, "r");
    if (f) { if (fscanf(f, "%31s", id) != 1) id[0] = 0; fclose(f); }
    for (unsigned i = 0; i < sizeof g_family / sizeof *g_family; i++) if (id[0] && !strcasecmp(id, g_family[i].id)) {
        g_stem = g_family[i].stem; setenv("DM_PROGRAM_ID", g_family[i].id, 1);
        if (i) { size_t l = strlen(files); snprintf(files + l, n - l, "/%s", g_family[i].id); mkdir(files, 0755); }
        fprintf(stderr, "boot: game %s (%s)\n", g_family[i].id, g_family[i].stem); return; }
    setenv("DM_PROGRAM_ID", g_family[0].id, 1);
}
// The player (the setup screen's Save data → Player, Profiles.java): <app folder>/current_profile names one of <app folder>/profiles.tsv's "id TAB name" lines (an id
// not in the list: the list's first, as Profiles.current). Player 1 (id 1; also no list, or anything unreadable) keeps the save every player had before,
// <game folder>/save, never moved; another player's is <game folder>/profiles/<id>/save, with its save_place.txt, save_import and backups beside it. One list for
// every game of the app: Sword and Shield (Scarlet and Violet) follow the same player, each game with its own save
static char g_prof[1100], g_prof_id[20] = "1", g_prof_name[128];
static int prof_id_ok(const char *s) { size_t n = strlen(s); if (!n || n > 16) return 0; for (; *s; s++) if (!((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9'))) return 0; return 1; }
static void choose_profile(const char *root) {
    char p[1100], want[20] = "", ln[512], first[20] = "", firstname[128] = ""; snprintf(p, sizeof p, "%s/current_profile", root); FILE *f = fopen(p, "r");
    if (f) { if (fscanf(f, "%19s", want) != 1) want[0] = 0; fclose(f); }
    snprintf(p, sizeof p, "%s/profiles.tsv", root); f = fopen(p, "r"); int found = 0;
    while (f && fgets(ln, sizeof ln, f)) { ln[strcspn(ln, "\r\n")] = 0; char *t = strchr(ln, '\t'); if (!t) continue; *t++ = 0;
        if (!prof_id_ok(ln) || !*t) continue;
        if (!first[0]) { snprintf(first, sizeof first, "%s", ln); snprintf(firstname, sizeof firstname, "%s", t); }
        if (!found && !strcmp(ln, want)) { found = 1; snprintf(g_prof_id, sizeof g_prof_id, "%s", ln); snprintf(g_prof_name, sizeof g_prof_name, "%s", t); } }
    if (f) fclose(f);
    if (!found && first[0]) { snprintf(g_prof_id, sizeof g_prof_id, "%s", first); snprintf(g_prof_name, sizeof g_prof_name, "%s", firstname); }
    if (!strcmp(g_prof_id, "1")) snprintf(g_prof, sizeof g_prof, "%s", g_files);
    else { snprintf(p, sizeof p, "%s/profiles", g_files); mkdir(p, 0755); snprintf(g_prof, sizeof g_prof, "%s/%s", p, g_prof_id); mkdir(g_prof, 0755); }
}
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

// The in-game settings (app/java/com/nxnative/GameMenu.java, an Android dialog over the game): the touch controls' menu button (touch_ui.c), or holding − on a controller
// for 3 s (on_input). what: 0 = the menu, 1 = the controller buttons. Called from the input thread; the Java call runs on a thread of its own and returns at once
static void *open_menu_main(void *p) {
    int what = (int)(intptr_t)p; JavaVM *vm = g_app->activity->vm; JNIEnv *e;
    if ((*vm)->AttachCurrentThread(vm, &e, NULL)) return NULL;
    jobject act = g_app->activity->clazz; jclass ac = (*e)->GetObjectClass(e, act);
    jobject cl = (*e)->CallObjectMethod(e, act, (*e)->GetMethodID(e, ac, "getClassLoader", "()Ljava/lang/ClassLoader;"));
    jclass clc = (*e)->FindClass(e, "java/lang/ClassLoader");
    jclass gm = (jclass)(*e)->CallObjectMethod(e, cl, (*e)->GetMethodID(e, clc, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;"), (*e)->NewStringUTF(e, "com.nxnative.GameMenu"));
    if ((*e)->ExceptionCheck(e) || !gm) { (*e)->ExceptionClear(e); fprintf(stderr, "[RED] menu: GameMenu class missing\n"); }
    else { (*e)->CallStaticVoidMethod(e, gm, (*e)->GetStaticMethodID(e, gm, "show", "(Landroid/app/Activity;I)V"), act, (jint)what);
           if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionDescribe(e); (*e)->ExceptionClear(e); } }
    (*vm)->DetachCurrentThread(vm);
    return NULL;
}
// A one-line note over the game, by its key (nvn_vk.c's Mali white-screen check: front_note): com.nxnative.MaliLadder.note shows it, translated, as a toast. Called
//   from the replay's thread; the Java call runs on a thread of its own (as open_menu_main) and front_note returns at once
static void *note_main(void *p) {
    char *key = p; JavaVM *vm = g_app->activity->vm; JNIEnv *e;
    if ((*vm)->AttachCurrentThread(vm, &e, NULL)) { free(key); return NULL; }
    jobject act = g_app->activity->clazz; jclass ac = (*e)->GetObjectClass(e, act);
    jobject cl = (*e)->CallObjectMethod(e, act, (*e)->GetMethodID(e, ac, "getClassLoader", "()Ljava/lang/ClassLoader;"));
    jclass clc = (*e)->FindClass(e, "java/lang/ClassLoader");
    jclass ml = (jclass)(*e)->CallObjectMethod(e, cl, (*e)->GetMethodID(e, clc, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;"), (*e)->NewStringUTF(e, "com.nxnative.MaliLadder"));
    jmethodID m = NULL;
    if (!(*e)->ExceptionCheck(e) && ml) m = (*e)->GetStaticMethodID(e, ml, "note", "(Landroid/app/Activity;Ljava/lang/String;)V");
    if ((*e)->ExceptionCheck(e) || !m) { (*e)->ExceptionClear(e); fprintf(stderr, "[RED] note: com.nxnative.MaliLadder.note missing\n"); }
    else { (*e)->CallStaticVoidMethod(e, ml, m, act, (*e)->NewStringUTF(e, key)); if ((*e)->ExceptionCheck(e)) { (*e)->ExceptionDescribe(e); (*e)->ExceptionClear(e); } }
    (*vm)->DetachCurrentThread(vm); free(key);
    return NULL;
}
void front_note(const char *key) {
    char *k = strdup(key ? key : ""); pthread_t th;
    if (k && !pthread_create(&th, NULL, note_main, k)) pthread_detach(th); else free(k);
}
void front_open_menu(int what) {
    static _Atomic long long last_ms; struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); long long t = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
    if (t - atomic_load(&last_ms) < 500) return;   // one dialog per press
    atomic_store(&last_ms, t);
    pthread_t th; if (!pthread_create(&th, NULL, open_menu_main, (void *)(intptr_t)what)) pthread_detach(th);
    fprintf(stderr, "menu: opened (%s)\n", what ? "controller buttons" : "settings");
}

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
// Controller buttons: an Android key code → Switch buttons table, made once. DM_PAD_LAYOUT=position (default) maps the face buttons by place, the bottom button of an
// Xbox-style pad (AKEYCODE_BUTTON_A) being the Switch's B as on a Switch pad; labels maps them by letter (A = the button marked A). DM_PAD_MAPPING=<button>:<key code>,...
// (the settings screen's remapping, PadMapping.java) then moves single buttons: A B X Y L R ZL ZR LS RS PLUS MINUS UP DOWN LEFT RIGHT
enum { KC_MAX = 512 };
static uint64_t g_kc_map[KC_MAX];
// the in-game menu's changes to DM_PAD_LAYOUT / DM_PAD_MAPPING (nativeSetting), read here instead of the environment once set; the input thread rebuilds the table
static pthread_mutex_t g_padset_mu = PTHREAD_MUTEX_INITIALIZER; static char g_padset_layout[16], g_padset_map[512]; static int g_padset_live; static _Atomic int g_padset_dirty;
static void pad_map_init(void) {
    memset(g_kc_map, 0, sizeof g_kc_map);
    char lay_live[16], map_live[512]; int live;
    pthread_mutex_lock(&g_padset_mu); live = g_padset_live; strcpy(lay_live, g_padset_layout); strcpy(map_live, g_padset_map); pthread_mutex_unlock(&g_padset_mu);
    static const struct { const char *name; uint64_t bit; int32_t pos, lab; } B[] = {
        { "A", DM_BTN_A, AKEYCODE_BUTTON_B, AKEYCODE_BUTTON_A }, { "B", DM_BTN_B, AKEYCODE_BUTTON_A, AKEYCODE_BUTTON_B },
        { "X", DM_BTN_X, AKEYCODE_BUTTON_Y, AKEYCODE_BUTTON_X }, { "Y", DM_BTN_Y, AKEYCODE_BUTTON_X, AKEYCODE_BUTTON_Y },
        { "L", DM_BTN_L, AKEYCODE_BUTTON_L1, AKEYCODE_BUTTON_L1 }, { "R", DM_BTN_R, AKEYCODE_BUTTON_R1, AKEYCODE_BUTTON_R1 },
        { "ZL", DM_BTN_ZL, AKEYCODE_BUTTON_L2, AKEYCODE_BUTTON_L2 }, { "ZR", DM_BTN_ZR, AKEYCODE_BUTTON_R2, AKEYCODE_BUTTON_R2 },
        { "LS", DM_BTN_LS, AKEYCODE_BUTTON_THUMBL, AKEYCODE_BUTTON_THUMBL }, { "RS", DM_BTN_RS, AKEYCODE_BUTTON_THUMBR, AKEYCODE_BUTTON_THUMBR },
        { "PLUS", DM_BTN_PLUS, AKEYCODE_BUTTON_START, AKEYCODE_BUTTON_START }, { "MINUS", DM_BTN_MINUS, AKEYCODE_BUTTON_SELECT, AKEYCODE_BUTTON_SELECT },
        { "UP", DM_BTN_UP, AKEYCODE_DPAD_UP, AKEYCODE_DPAD_UP }, { "DOWN", DM_BTN_DOWN, AKEYCODE_DPAD_DOWN, AKEYCODE_DPAD_DOWN },
        { "LEFT", DM_BTN_LEFT, AKEYCODE_DPAD_LEFT, AKEYCODE_DPAD_LEFT }, { "RIGHT", DM_BTN_RIGHT, AKEYCODE_DPAD_RIGHT, AKEYCODE_DPAD_RIGHT },
    };
    const char *lay = live ? lay_live : getenv("DM_PAD_LAYOUT"); int labels = lay && !strcmp(lay, "labels");
    for (unsigned i = 0; i < sizeof B / sizeof *B; i++) g_kc_map[labels ? B[i].lab : B[i].pos] |= B[i].bit;
    const char *m = live ? map_live : getenv("DM_PAD_MAPPING"); int moved = 0;
    while (m && *m) {
        char name[8]; int kc; int n = 0;
        if (sscanf(m, " %7[A-Za-z]:%d%n", name, &kc, &n) == 2 && kc > 0 && kc < KC_MAX)
            for (unsigned i = 0; i < sizeof B / sizeof *B; i++) if (!strcasecmp(name, B[i].name)) {
                for (int k = 0; k < KC_MAX; k++) g_kc_map[k] &= ~B[i].bit;   // the button leaves its old key
                g_kc_map[kc] |= B[i].bit; moved++; }
        const char *c = strchr(m, ','); if (!c) break; m = c + 1;
    }
    fprintf(stderr, "boot: controller buttons by %s%s\n", labels ? "label" : "position", moved ? ", some remapped (DM_PAD_MAPPING)" : "");
}
static uint64_t key_bit(int32_t kc) {
    static int init; if (!init || atomic_exchange(&g_padset_dirty, 0)) { init = 1; pad_map_init(); }   // (the input thread only)
    return kc > 0 && kc < KC_MAX ? g_kc_map[kc] : 0;
}
// A controller stick: the dead zone on its tilt, not on each axis. Each axis on its own (as before) shortened every diagonal: a full push at 45 degrees came out at
//   0.94 of full on a perfect stick and about 0.85 on a round-gate one, whose axes stop near 0.65 there (Arceus-Native-Android#16: no running on diagonals with a
//   GameSir X5 Lite). Under 12 % of the tilt nothing; from 92 % full (few sticks reach their rim at every angle); the direction kept
static void stick(float x, float y, int32_t *ox, int32_t *oy) {
    float m = sqrtf(x * x + y * y); if (!(m >= 0.12f)) { *ox = *oy = 0; return; }
    float s = (m - 0.12f) / (0.92f - 0.12f); if (s > 1) s = 1;
    *ox = (int32_t)(x / m * s * DM_STICK_MAX); *oy = (int32_t)(y / m * s * DM_STICK_MAX);
}
// Touch: the on-screen controls (touch_ui.c)
void touch_ui_init(ANativeActivity *a);
void touch_ui_event(const AInputEvent *e, float w, float h, uint64_t *btn, int32_t s[4]);
void touch_ui_hide(void);
void touch_ui_release(void);   // every finger let go (focus lost: a lift Android may never send)
void touch_ui_pad(int connected);
void touch_ui_release(void);   // every finger let go (focus lost: a lift Android may never send)
#include <jni.h>
JNIEXPORT void JNICALL Java_com_nxnative_GameActivity_nativePadConnected(JNIEnv *env, jclass c, jboolean on) { (void)env; (void)c; touch_ui_pad(on); }   // GameActivity.java
void dm_gyro_set(float yaw, float pitch);
JNIEXPORT void JNICALL Java_com_nxnative_GameActivity_nativeGyro(JNIEnv *env, jclass c, jfloat yaw, jfloat pitch) { (void)env; (void)c; dm_gyro_set(yaw, pitch); }   // (hle_hid.c, gyro aiming)
void touch_ui_setting(const char *key, const char *value);
void touch_ui_edit(void);
// GameMenu.java: a setting changed while playing (it also writes env.txt for the next start). The ones the game can take at once go to their owners here, never through
// setenv (other threads read the environment meanwhile); the rest wait for the next start. Returns whether it took effect now
static int g_bg_off = -1;   // DM_BACKGROUND=off: pause in the background (default: keep playing); the settings panel changes it at once
static int bg_play(void) { if (g_bg_off < 0) { const char *e = getenv("DM_BACKGROUND"); g_bg_off = e && !strcmp(e, "off"); } return !g_bg_off; }
extern int dm_fps_live __attribute__((weak));   // loader.c: the live frame-rate target (30 / 60), 0 when the start has none (another game, a fixed 90 / 120, DM_FPS_LIVE=0)
static void window_rate(ANativeWindow *w);
// GameMenu.java: why frame generation is not making pictures right now (nvn_vk.c nvn_vk_framegen_note: "", "paused:<fps>", "slow:<ms>", "off"), for the line under its choice
extern void nvn_vk_framegen_note(char *buf, size_t n) __attribute__((weak));
JNIEXPORT jstring JNICALL Java_com_nxnative_GameMenu_nativeFramegenNote(JNIEnv *env, jclass c) {
    (void)c; char b[48] = ""; if (nvn_vk_framegen_note) nvn_vk_framegen_note(b, sizeof b); return (*env)->NewStringUTF(env, b);
}
// Battery saver (DM_POWER=saver; Performance -> Power, live; #333): frame rate on Auto 30, heat counted from "light", the 3D render scale at most the 720p step,
//   and the performance hint asks for the whole frame (the governor runs the cores slower). 1 on, 0 off, -1 not read yet
static _Atomic int g_power_saver = -1;
static int power_saver(void) { int v = atomic_load(&g_power_saver); if (v < 0) { const char *e = getenv("DM_POWER"); v = e && !strcmp(e, "saver"); atomic_store(&g_power_saver, v); } return v; }
// Graphics → Mode (DM_DISPLAY_MODE, live): Auto (the Graphics level decides), Quality (docked: TV mode, 1080p, the game's full effects), Performance (handheld:
//   720p and the game's lighter settings). The game takes a change as a real dock / undock (hle_oe.c dm_set_opmode → notifications 30 / 31: Sword / Shield and
//   Legends: Arceus redraw at 1280x720 and back at once, Galaxy S25+ 2026-11-20). Pokémon Sword / Shield stay in TV mode for Performance: their handheld mode crashed
//   when a Pokémon Dynamaxes (#211 #115); the same 720p comes from the 3D render scale instead (the game's own scale, capped at the 720p step, never above the
//   player's slider). With Quality or Performance chosen, Graphics Auto switches the mode no more (graphics_auto); heat lowers only the 3D render scale
void dm_set_opmode(int m);   // (host/hle_oe.c)
static int display_galar(void) { return !strcmp(DM_GAME_ID, "01008DB008C2C000") || !strcmp(DM_GAME_ID, "0100ABF008968000"); }
static int display_mode(void) { const char *m = getenv("DM_DISPLAY_MODE"); return m && !strcmp(m, "quality") ? 1 : m && !strcmp(m, "performance") ? 2 : 0; }   // 0 Auto
static float g_user_cap = -1;   // the player's 3D render scale (the slider or the Graphics level's; -1 none), which Performance only ever goes under (Sword / Shield)
// The budget tier (graphics_auto): the cap Graphics Low / Lowest set at start when the app picked that level for the phone and nobody set a cap of their own (0: none).
//   The guard then steps the scale under it, down to 35 %. A cap set in the menu ends it; Auto there brings it back (g_lvl_gen: the guard starts its ladder again)
static float g_lvl_cap, g_lvl_cap0; static _Atomic int g_lvl_gen;
static float display_cap(float c) { return display_mode() == 2 && display_galar() && c > 0.667f ? 0.667f : c; }
static int g_fps_guard;   // (the fps guard's target: defined below with fps_guard_init; changed here when the frame rate is)
JNIEXPORT jboolean JNICALL Java_com_nxnative_GameMenu_nativeSetting(JNIEnv *env, jclass c, jstring jk, jstring jv) {
    (void)c; const char *k = (*env)->GetStringUTFChars(env, jk, NULL), *v0 = (*env)->GetStringUTFChars(env, jv, NULL); const char *v = *v0 ? v0 : NULL; int now = 1;
    if (!strcmp(k, "DM_BACKGROUND")) g_bg_off = v && !strcmp(v, "off");
    else if (!strcmp(k, "DM_POWER")) {   // live: the guard and the hint read it each time; on, a frame rate on Auto at 60 goes to 30 at once
        int on = v && !strcmp(v, "saver"); if (on) setenv("DM_POWER", v, 1); else unsetenv("DM_POWER"); atomic_store(&g_power_saver, on);
        const char *ft = getenv("DM_FPS_TARGET"); int chose60 = ft && atoi(ft) >= 60 && !getenv("DM_AUTO_FPS_SET");
        if (on && !chose60 && &dm_fps_live && __atomic_load_n(&dm_fps_live, __ATOMIC_RELAXED) == 60) { __atomic_store_n(&dm_fps_live, 30, __ATOMIC_RELAXED); window_rate(atomic_load(&g_window)); }
        fprintf(stderr, "power: %s, set while playing\n", on ? "battery saver" : "balanced"); }
    else if (!strcmp(k, "DM_FPS_TARGET")) {   // 30 / 60 at once with the live frame rate (the hold limit, the window's rate, the display pacing); otherwise the next start
        int dyn = v && !strcmp(v, "dyn");
        { extern _Atomic int dm_fps_dyn __attribute__((weak)); if (&dm_fps_dyn) atomic_store(&dm_fps_dyn, dyn); }   // (the game step follows the hold: hle_nvn.c vstep_tick)
        int t = dyn ? 60 : v ? atoi(v) : 0; if (!v) { const char *a = getenv("DM_AUTO_FPS"); t = a && atoi(a) >= 60 ? 60 : 30; }   // (Auto: the phone profile's)
        if (g_fps_guard > 0 && (t == 30 || t == 60)) { g_fps_guard = dyn || t == 30 ? 30 : 60; fprintf(stderr, "fps guard: now defends %d fps%s\n", g_fps_guard, dyn ? " (Dynamic: 60 where the phone can, the picture kept)" : ""); }
        if (&dm_fps_live && __atomic_load_n(&dm_fps_live, __ATOMIC_RELAXED) && (t == 30 || t == 60)) {
            __atomic_store_n(&dm_fps_live, t, __ATOMIC_RELAXED); window_rate(atomic_load(&g_window));
            fprintf(stderr, "frame rate: %d fps, switched while playing\n", t);
        } else now = 0;
    }
    else if (!strcmp(k, "DM_GYRO")) { extern _Atomic int dm_gyro_mode; atomic_store(&dm_gyro_mode, v && !strcmp(v, "on") ? 2 : v && !strcmp(v, "aim") ? 1 : 0); }
    else if (!strcmp(k, "DM_TOUCH_OPACITY") || !strcmp(k, "DM_TOUCH_SIZE") || !strcmp(k, "DM_TOUCH_GESTURES") || !strcmp(k, "DM_TOUCH_CAM_SENS") || !strcmp(k, "DM_TOUCH_CAMERA") || !strcmp(k, "DM_TOUCH_RSTICK") || !strcmp(k, "DM_TOUCH_FADE") || !strcmp(k, "DM_TOUCH_HAPTIC") || !strcmp(k, "DM_TOUCH_L3R3") || !strcmp(k, "DM_TOUCH_STICKS") || !strcmp(k, "DM_TOUCH_EDGE_RUN") || !strcmp(k, "DM_TOUCH_RIDE_SWIPE")) touch_ui_setting(k, v);
    else if (!strcmp(k, "DM_RUMBLE")) { touch_ui_setting(k, v); now = 0; }
    // the 3D render scale (☰ → Graphics → 3D render scale; hle_resscale.c's cap on the game's own scale, Sword / Shield): live. Auto = what the Graphics level gives
    //   (Lowest 0.50, Low 0.55, else none) and the heat guard steps it; a percentage is held, and the guard leaves it alone (graphics_auto: DM_RES_CAP set)
    else if (!strcmp(k, "DM_DISPLAY_MODE")) {
        if (v && *v) setenv("DM_DISPLAY_MODE", v, 1); else unsetenv("DM_DISPLAY_MODE");
        int m = display_mode(); const char *g = getenv("DM_GRAPHICS");
        if (display_galar()) {   // (TV mode always; Performance: the 3D render scale at the 720p step, under the player's own)
            extern void dm_res_cap_set(float) __attribute__((weak)); extern int dm_res_scale_ready(void);
            float u = g_user_cap > 0 ? g_user_cap : 1.0f, c = display_cap(u);
            if (m == 2) { char b[16]; snprintf(b, sizeof b, "%.3f", c); setenv("DM_RES_CAP", b, 1); }
            else if (g_user_cap > 0 && g_user_cap < 1.0f) { char b[16]; snprintf(b, sizeof b, "%.2f", g_user_cap); setenv("DM_RES_CAP", b, 1); } else unsetenv("DM_RES_CAP");
            if (&dm_res_cap_set && dm_res_scale_ready()) dm_res_cap_set(c); else now = 0;
            fprintf(stderr, "mode: %s: 3D render scale %.0f %% (TV mode kept), set while playing\n", m == 2 ? "performance" : m == 1 ? "quality" : "auto", c * 100);
        } else {
            int tv = m == 1 ? 1 : m == 2 ? 0 : !(g && (!strcmp(g, "low") || !strcmp(g, "lowest")));   // (Auto: what the Graphics level gives)
            // Scarlet / Violet: never handheld → TV mode while playing (a player's Scarlet 4.0.0 crashed 5 s after that switch: a null read at main+0xc281d0,
            //   paldea-renderer 83a7fa1); TV mode is then taken from the next start
            if (DM_PORT_PALDEA && tv) { now = 0; fprintf(stderr, "mode: %s: TV mode (docked) from the next start (Scarlet / Violet are not switched to TV mode while playing)\n", m == 1 ? "quality" : "auto"); }
            else { dm_set_opmode(tv); fprintf(stderr, "mode: %s: %s, set while playing\n", m == 2 ? "performance" : m == 1 ? "quality" : "auto", tv ? "TV mode (docked)" : "handheld mode"); }
        } }
#if DM_PORT_ARCEUS
    else if (!strcmp(k, "DM_RES_CAP")) {   // Legends: Arceus's 3D render scale (loader.c arc_scale): live; Auto = the game's own (1.0)
        extern void dm_arc_scale_set(float) __attribute__((weak)); float c = v ? (float)atof(v) : 1.0f; if (!(c >= 0.5f && c <= 1.0f)) c = 1.0f;
        if (v) setenv("DM_RES_CAP", v, 1); else unsetenv("DM_RES_CAP");
        if (dm_arc_scale_set) dm_arc_scale_set(c); else now = 0; }
#else
    else if (!strcmp(k, "DM_RES_CAP")) { extern void dm_res_cap_set(float) __attribute__((weak)); extern int dm_res_scale_ready(void);
        float c = v ? (float)atof(v) : 0; const char *g = getenv("DM_GRAPHICS");
        if (c >= 0.35f && c <= 1.0f) { setenv("DM_RES_CAP", v, 1); g_lvl_cap = 0; }
        else { unsetenv("DM_RES_CAP"); c = g && !strcmp(g, "lowest") ? 0.50f : g && !strcmp(g, "low") ? 0.55f : 1.0f;
               if (g_lvl_cap0 > 0) { char b[16]; snprintf(b, sizeof b, "%.2f", g_lvl_cap0); setenv("DM_RES_CAP", b, 1); g_lvl_cap = c = g_lvl_cap0; } }   // (Auto: the budget tier again)
        atomic_fetch_add(&g_lvl_gen, 1);
        g_user_cap = c; c = display_cap(c);   // (Mode: Performance keeps it at the 720p step at most)
        if (display_cap(1.0f) < 1.0f) { char b[16]; snprintf(b, sizeof b, "%.3f", c); setenv("DM_RES_CAP", b, 1); }   // (held: Auto's guard leaves it alone)
        if (&dm_res_cap_set && dm_res_scale_ready()) { dm_res_cap_set(c); fprintf(stderr, "3D render scale: %.0f %% (%s), set while playing\n", c * 100, v ? "chosen" : "Auto"); } else now = 0; }
#endif
    else if (!strcmp(k, "DM_RSTICK_SENS")) { extern _Atomic int dm_rstick_pct __attribute__((weak)); int p = v ? atoi(v) : 100; if (p < 25 || p > 400) p = 100; if (&dm_rstick_pct) atomic_store(&dm_rstick_pct, p); else now = 0; }   // (the touch click follows it at once; the game's motors are GameActivity's, GameMenu decides)
    else if (!strcmp(k, "DM_LSTICK_SENS")) { extern _Atomic int dm_lstick_pct __attribute__((weak)); int p = v ? atoi(v) : 100; if (p < 50 || p > 300) p = 100; if (&dm_lstick_pct) atomic_store(&dm_lstick_pct, p); else now = 0; }   // (hle_hid.c lsens)
    else if (!strcmp(k, "DM_ASPECT")) __atomic_store_n(&dm_aspect_mode, !v ? DM_ASPECT_FIT : !strcmp(v, "20:9") ? DM_ASPECT_20_9 : !strcmp(v, "fill") || !strcmp(v, "wide") ? DM_ASPECT_FILL : DM_ASPECT_FIT, __ATOMIC_RELAXED);
    else if (!strcmp(k, "DM_COLOR")) { extern volatile int dm_color_mode; dm_color_mode = v && !strcmp(v, "vibrant") ? 1 : v && !strcmp(v, "cinematic") ? 2 : 0; }   // (nvn_vk.c color_mode: live)
    else if (!strcmp(k, "DM_SCALER")) { extern volatile int dm_scaler_mode; if (!v || !*v) v = getenv("DM_AUTO_SCALER"); dm_scaler_mode = v && !strcmp(v, "sgsr") ? 1 : v && !strcmp(v, "fsr") ? 3 : v && !strcmp(v, "smooth") ? 2 : 0; }   // (Auto: the app's pick)
    else if (!strcmp(k, "DM_FRAMEGEN")) { extern volatile int dm_framegen_mode; dm_framegen_mode = v && (!strcmp(v, "on") || !strcmp(v, "1")) ? 1 : v && !strcmp(v, "debug") ? 2 : v && !strcmp(v, "lsfg") ? 3 : 0; window_rate(atomic_load(&g_window));   // nvn_vk.c present_window_locked
        window_rate(atomic_load(&g_window)); }   // (Android is told the doubled rate, or the game's own again)
    else if (!strcmp(k, "DM_LSFG_MULT")) { extern volatile int dm_lsfg_mult; int m = v ? atoi(v) : 0; dm_lsfg_mult = m >= 2 && m <= 4 ? m : 0; }   // nvn_vk.c lsfg_mult: 0 auto
    else if (!strcmp(k, "DM_SCALER_SHARPNESS")) { extern volatile float dm_scaler_sharpness; dm_scaler_sharpness = !v || !strcmp(v, "sharp") ? 0.7f : !strcmp(v, "smooth") ? -1.0f : (float)atof(v); }   // nvn_vk.c scale_draw
    else if (!strcmp(k, "DM_PAD_LAYOUT") || !strcmp(k, "DM_PAD_MAPPING")) {
        pthread_mutex_lock(&g_padset_mu);
        if (!g_padset_live) { const char *l = getenv("DM_PAD_LAYOUT"), *m = getenv("DM_PAD_MAPPING"); snprintf(g_padset_layout, sizeof g_padset_layout, "%s", l ? l : ""); snprintf(g_padset_map, sizeof g_padset_map, "%s", m ? m : ""); g_padset_live = 1; }
        if (!strcmp(k, "DM_PAD_LAYOUT")) snprintf(g_padset_layout, sizeof g_padset_layout, "%s", v ? v : ""); else snprintf(g_padset_map, sizeof g_padset_map, "%s", v ? v : "");
        pthread_mutex_unlock(&g_padset_mu); atomic_store(&g_padset_dirty, 1);
    } else now = 0;
    fprintf(stderr, "menu: %s=%s (%s)\n", k, v ? v : "(default)", now ? "now" : "from the next start");
    (*env)->ReleaseStringUTFChars(env, jk, k); (*env)->ReleaseStringUTFChars(env, jv, v0);
    return now ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT void JNICALL Java_com_nxnative_GameMenu_nativeEditTouch(JNIEnv *env, jclass c) { (void)env; (void)c; touch_ui_edit(); }   // "Move the touch controls"
// Holding − on a controller for 3 s opens the controller buttons (front_open_menu(1)): a press starts a timer thread that checks the same press is still held
uint64_t dm_nvn_frame_count(void) __attribute__((weak));   // (nvn: frames presented)
static _Atomic int g_minus_seq, g_minus_held;
// − and + held together for 1 s: the whole menu (front_open_menu(0)), for controllers and handhelds with built-in controls, where the touch ☰ is hidden
static _Atomic int g_combo_seq, g_combo_held;
static void *combo_hold_main(void *p) {
    int seq = (int)(intptr_t)p; usleep(1000000);
    if (atomic_load(&g_combo_held) && atomic_load(&g_combo_seq) == seq) { atomic_fetch_add(&g_minus_seq, 1); front_open_menu(0); }   // (the − hold's own timer stops)
    return NULL;
}
JNIEXPORT jlong JNICALL Java_com_nxnative_GameActivity_nativeFrames(JNIEnv *env, jclass c) { (void)env; (void)c; return dm_nvn_frame_count ? (jlong)dm_nvn_frame_count() : -1; }   // the performance overlay's fps
static void *minus_hold_main(void *p) {
    int seq = (int)(intptr_t)p; usleep(3000000);
    if (atomic_load(&g_minus_held) && atomic_load(&g_minus_seq) == seq) front_open_menu(1);
    return NULL;
}
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
        { int touch_ui_edit_back(void); if (kc == AKEYCODE_BACK && AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP && touch_ui_edit_back()) return 1; }   // (the layout editor: Back saves and leaves it)
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
        if (b & DM_BTN_MINUS) {   // − held 3 s: the controller buttons (minus_hold_main)
            if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN && AKeyEvent_getRepeatCount(e) == 0) { int seq = atomic_fetch_add(&g_minus_seq, 1) + 1; atomic_store(&g_minus_held, 1);
                pthread_t th; if (!pthread_create(&th, NULL, minus_hold_main, (void *)(intptr_t)seq)) pthread_detach(th); }
            else if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP) atomic_store(&g_minus_held, 0);
        }
        if (b & (DM_BTN_MINUS | DM_BTN_PLUS)) {   // − and + together (combo_hold_main)
            int both = (g_pads[k].btn & (DM_BTN_MINUS | DM_BTN_PLUS)) == (DM_BTN_MINUS | DM_BTN_PLUS);
            if (both && !atomic_load(&g_combo_held)) { int seq = atomic_fetch_add(&g_combo_seq, 1) + 1; atomic_store(&g_combo_held, 1);
                pthread_t th; if (!pthread_create(&th, NULL, combo_hold_main, (void *)(intptr_t)seq)) pthread_detach(th); }
            else if (!both) atomic_store(&g_combo_held, 0);
        }
        touch_ui_hide(); pads_merge(); return 1;
    }
    if (type == AINPUT_EVENT_TYPE_MOTION && (src & (AINPUT_SOURCE_JOYSTICK | AINPUT_SOURCE_GAMEPAD))) {
        int k = pad_slot(AInputEvent_getDeviceId(e));
        float rx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Z, 0), ry = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RZ, 0);
        if (rx == 0 && ry == 0) { rx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RX, 0); ry = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RY, 0); }   // right stick as RX / RY on some drivers
        stick(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_X, 0), -AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Y, 0), &g_pads[k].lx, &g_pads[k].ly);
        stick(rx, -ry, &g_pads[k].rx, &g_pads[k].ry);
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
// ---------- Crash-aware starts: <files>/boot_state says how far a run got (start → device: the graphics device is up → gone after 30 s of play), and "crashed" when
// the crash handler ran (crash.c dm_crash_note). A crashed run's stage decides what the next start does:
//   · crashed before 30 s of play on the phone's own driver, with the device not up or no frames yet: the app offers a driver package for this GPU (notice.txt,
//     SetupActivity reads it and does not start the game straight away); not when the pc was in the game's own code (main / sdk: a driver would not help)
//   · crashed with mods on: the mods are switched off (each folder renamed with a leading "-", loose .pchtxt files to .pchtxt.off), and the player is told
// A run Android ends (swiped away, low memory) writes no "crashed": nothing is changed then
static char g_boot_path[1100]; static int g_boot_fd = -1;
static _Atomic int g_dev_noted;   // "device" written to boot_state (nvn_vk.c right after the device is made, else boot_guard_main within a second)
static int g_safe_advanced;       // this start is one safe_start level up (no driver dialog then: the safer settings are tried first)
void dm_boot_device_up(void) { if (!atomic_exchange(&g_dev_noted, 1) && g_boot_fd >= 0) { static const char m[] = "device\n"; (void)!write(g_boot_fd, m, sizeof m - 1); } }
static char g_crash_src[1200], g_crash_dl[256];   // log.txt and its copy in Download/ (host_start), for dm_crash_log_copy
void dm_crash_log_copy(void) {   // from the crash handler: open / read / write only
    if (!g_crash_src[0] || !g_crash_dl[0]) return;
    int in = open(g_crash_src, O_RDONLY), out = in >= 0 ? open(g_crash_dl, O_WRONLY | O_CREAT | O_TRUNC, 0644) : -1; char b[16384]; ssize_t n;
    if (in >= 0 && out >= 0) while ((n = read(in, b, sizeof b)) > 0) if (write(out, b, (size_t)n) != n) break;
    if (in >= 0) close(in); if (out >= 0) close(out);
}
void dm_crash_note(int in_game) { if (g_boot_fd >= 0) { static const char m[] = "crashed\n", g[] = "crashed in the game's code\n"; if (in_game) (void)!write(g_boot_fd, g, sizeof g - 1); else (void)!write(g_boot_fd, m, sizeof m - 1); } }
const char *nvn_vk_gpu_name(void) __attribute__((weak));
static void notice(const char *kind, const char *text) {
    char p[1200]; snprintf(p, sizeof p, "%s/notice.txt", g_files); FILE *f = fopen(p, "a"); if (f) { fprintf(f, "%s\t%s\n", kind, text); fclose(f); }
    fprintf(stderr, "boot: notice %s: %s\n", kind, text);
}
static int mods_off(const char *mods) {   // every mod switched off; returns how many
    struct dirent **ents = NULL; int n = scandir(mods, &ents, NULL, alphasort), off = 0; if (n < 0) return 0;
    for (int i = 0; i < n; i++) { const char *nm = ents[i]->d_name; char a[1200], b[1200]; struct stat st; size_t l = strlen(nm);
        snprintf(a, sizeof a, "%s/%s", mods, nm);
        if (nm[0] != '.' && nm[0] != '-' && !stat(a, &st) && S_ISDIR(st.st_mode)) { snprintf(b, sizeof b, "%s/-%s", mods, nm); if (!rename(a, b)) off++; }
        else if (l > 7 && !strcasecmp(nm + l - 7, ".pchtxt")) { snprintf(b, sizeof b, "%s.off", a); if (!rename(a, b)) off++; }
        free(ents[i]); }
    free(ents); return off;
}
static int mods_on(const char *mods) {   // how many mods are on (what mods_romfs and the loader will use)
    struct dirent **ents = NULL; int n = scandir(mods, &ents, NULL, alphasort), on = 0; if (n < 0) return 0;
    for (int i = 0; i < n; i++) { const char *nm = ents[i]->d_name; char a[1200]; struct stat st; size_t l = strlen(nm); snprintf(a, sizeof a, "%s/%s", mods, nm);
        if ((nm[0] != '.' && nm[0] != '-' && !stat(a, &st) && S_ISDIR(st.st_mode)) || (l > 7 && !strcasecmp(nm + l - 7, ".pchtxt"))) on++; free(ents[i]); }
    free(ents); return on;
}
static void boot_guard(const char *mods) {
    snprintf(g_boot_path, sizeof g_boot_path, "%s/boot_state", g_files);
    char prev[512] = ""; FILE *f = fopen(g_boot_path, "r"); if (f) { size_t k = fread(prev, 1, sizeof prev - 1, f); prev[k] = 0; fclose(f); }
    if (prev[0] && strstr(prev, "crashed")) {
        int had_mods = strstr(prev, "mods 0") == NULL && strstr(prev, "mods ") != NULL, device = strstr(prev, "device") != NULL;
        const char *drv = getenv("DM_VK_DRIVER"); int stock = !drv || !*drv;
        fprintf(stderr, "boot: the last run crashed (%s)\n", device ? "after the graphics device came up" : "before the graphics device came up");
        if (device) setenv("DM_LAST_CRASH_DEVICE", "1", 1);   // (nvn_vk.c: a start that went without the dma-buf heap route and crashed anyway sends the next one back to it)
        if (device && strstr(prev, "in the game's code")) setenv("DM_LAST_CRASH_GAME", "1", 1);   // (nvn_vk.c's white-screen ladder: a crash in the game's own code is not its route's)
        if (had_mods) { int n = mods_off(mods); if (n) { char t[200]; snprintf(t, sizeof t, "The game closed while %d mod%s were on, so they are switched off. Switch them back on in the menu, Mods.", n, n == 1 ? "" : "s"); notice("mods", t); } }
        else if (stock && !strstr(prev, "playing") && !strstr(prev, "in the game's code") && !g_safe_advanced) {   // (a crash in the game's own code is the engine's to fix, not the driver's: no driver dialog; nor while safe_start has a safer level to try)
            char gpu[256] = ""; char gp[1200]; snprintf(gp, sizeof gp, "%s/gpu.txt", g_files); FILE *g = fopen(gp, "r"); if (g) { if (!fgets(gpu, sizeof gpu, g)) gpu[0] = 0; fclose(g); }
            char *nl = strchr(gpu, '\n'); if (nl) *nl = 0;
            notice("driver", *gpu ? gpu : "unknown");
        }
    }
    { char n[16]; snprintf(n, sizeof n, "%d", mods_on(mods)); setenv("DM_MODS_ACTIVE", n, 1); }   // (hle_ldn.c: local play is off while mods are on)
    g_boot_fd = open(g_boot_path, O_WRONLY | O_CREAT | O_TRUNC, 0660);
    if (g_boot_fd >= 0) { char t[160]; int k = snprintf(t, sizeof t, "start driver %s mods %d\n", getenv("DM_VK_DRIVER") && *getenv("DM_VK_DRIVER") ? "custom" : "stock", mods_on(mods)); (void)!write(g_boot_fd, t, k); }
}
static void *boot_guard_main(void *p) {
    (void)p; int dev = 0; time_t t0 = time(NULL);
    for (;;) { sleep(1);
        const char *gpu = nvn_vk_gpu_name ? nvn_vk_gpu_name() : NULL;
        if (gpu && !dev) { dev = 1; dm_boot_device_up();
            char gp[1200]; snprintf(gp, sizeof gp, "%s/gpu.txt", g_files); FILE *g = fopen(gp, "w"); if (g) { fprintf(g, "%s\n", gpu); fclose(g); } }
        uint64_t fr = dm_nvn_frame_count ? dm_nvn_frame_count() : 0;
        if (dev && fr >= 300 && time(NULL) - t0 >= 30) {   // played: the run is known good
            if (g_boot_fd >= 0) { static const char m[] = "playing\n"; (void)!write(g_boot_fd, m, sizeof m - 1); }
            unlink(g_boot_path); fprintf(stderr, "boot: 30 s of play: this start is good\n"); return NULL; }
    }
}
// The game's frame rate told to Android (30, or DM_FPS_TARGET with the 60 fps patch), as a fixed source: a 90 / 120 Hz phone then runs the display at a rate the
// frames divide evenly (60 fps on a 90 Hz screen would otherwise show frames for 1 and 2 refreshes in turn, an uneven judder). Presentation never tears (FIFO /
// MAILBOX swapchains); this keeps the frames evenly spaced
static void window_rate(ANativeWindow *w) {
    if (!w) return;
    const char *e = getenv("DM_FPS_TARGET"); int fps = e ? atoi(e) : 0; float rate = fps >= 60 && fps <= 120 ? (float)fps : 30.0f;
    { int lf = &dm_fps_live ? __atomic_load_n(&dm_fps_live, __ATOMIC_RELAXED) : 0; if (lf) rate = (float)lf; }   // (the live target, once the game is loaded)
    // frame generation (DM_FRAMEGEN, nvn_vk.c) shows two pictures a game frame: told as 30, Android showed this app only every other one (measured on a Galaxy S25+:
    //   300 of 600 pictures in 10 s, 33.4 ms apart)
    { extern volatile int dm_framegen_mode; int fg = dm_framegen_mode; const char *f = getenv("DM_FRAMEGEN");
      if (fg < 0) fg = f && (!strcmp(f, "on") || !strcmp(f, "1") || !strcmp(f, "debug") || !strcmp(f, "lsfg"));
      if (fg) rate = 120.0f; }   // (not the doubled rate itself: at 60 Hz every picture must make its own refresh with no slack, and a tenth of them missed it; at 120 each has a timed slot)
    // libnativewindow: setFrameRateWithChangeStrategy (Android 12+, ALWAYS: the display switches modes even when the switch is not seamless, so a 144 Hz screen goes
    // to 120 for 30 / 60 fps), else setFrameRate (Android 11); looked up, so a system without them only skips this
    typedef int32_t (*set_rate_fn)(ANativeWindow *, float, int8_t); typedef int32_t (*set_rate2_fn)(ANativeWindow *, float, int8_t, int8_t);
    void *h = dlopen("libnativewindow.so", RTLD_NOW);
    set_rate2_fn r2 = h ? (set_rate2_fn)dlsym(h, "ANativeWindow_setFrameRateWithChangeStrategy") : NULL; set_rate_fn r1 = h ? (set_rate_fn)dlsym(h, "ANativeWindow_setFrameRate") : NULL;
    int r = r2 ? r2(w, rate, ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE, 1 /* ANATIVEWINDOW_CHANGE_FRAME_RATE_ALWAYS */) : r1 ? r1(w, rate, ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE) : -1;
    fprintf(stderr, "boot: window frame rate %.0f fps (fixed source)%s\n", rate, r ? " refused by the system" : "");
}
// RomFS mods: every <mods>/<mod>/romfs folder (and <mods>/romfs) as a RomFS overlay root (hle_fs.c rom_overlay_path: the first root that has a file wins), in the load
// order of <mods>/order.txt (the in-game menu's: one name a line, first wins), then by name; a mod's exefs/*.pchtxt code patches are found by the loader (loader.c
// pchtxt_dir). A mod folder named with a leading "-" or "." is switched off
static char g_roots[8192]; static size_t g_roots_len; static int g_roots_n;
// HD texture packs (DM_HD_TEXTURES: "" Auto = the phone profile's DM_AUTO_HD, "on", "off"; the menu's Display → HD textures): a mod whose RomFS is mostly textures,
//   100 MB or more with at least 3/4 of its bytes under a path naming "texture" (retexture mods such as the HD mod for Sword and Shield), is left out on a phone that
//   should not carry it, as if switched off: more texture memory and slower area loads would cost a weaker phone its 30 fps
static int has_texture_word(const char *s) { for (; *s; s++) if (!strncasecmp(s, "texture", 7)) return 1; return 0; }
static void tex_walk(const char *dir, int intex, uint64_t *all, uint64_t *tex, int depth) {
    DIR *d = depth > 16 ? NULL : opendir(dir); if (!d) return;
    struct dirent *e; char p[1200]; struct stat st;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' || snprintf(p, sizeof p, "%s/%s", dir, e->d_name) >= (int)sizeof p || stat(p, &st)) continue;
        int t = intex || has_texture_word(e->d_name);
        if (S_ISDIR(st.st_mode)) tex_walk(p, t, all, tex, depth + 1); else { *all += (uint64_t)st.st_size; if (t) *tex += (uint64_t)st.st_size; }
    }
    closedir(d);
}
static int texture_pack(const char *romfs) { uint64_t all = 0, tex = 0; tex_walk(romfs, 0, &all, &tex, 0); return all >= (100ull << 20) && tex * 4 >= all * 3; }
static int hd_textures_on(const char **why) {
    const char *e = getenv("DM_HD_TEXTURES"); *why = "HD textures in the menu";
    if (!e || !*e) { e = getenv("DM_AUTO_HD"); *why = "HD textures on Auto: this phone's memory and GPU"; }
    return !e || strcmp(e, "off");   // (no profile: an older app, as before: on)
}
static void mod_root(const char *mods, const char *nm) {
    char d[1200]; struct stat st; static char skipped[2048];
    if (nm[0] == '.' || nm[0] == '-' || !strcmp(nm, "romfs") || strchr(nm, '/')) return;
    snprintf(d, sizeof d, "%s/%s/romfs", mods, nm); size_t dl = strlen(d);
    if (stat(d, &st) || !S_ISDIR(st.st_mode) || strchr(d, ':') || g_roots_len + dl + 2 >= sizeof g_roots) return;
    for (const char *q = g_roots; (q = strstr(q, d)); q += dl) if ((q == g_roots || q[-1] == ':') && (q[dl] == ':' || !q[dl])) return;   // (listed already)
    { char key[300]; snprintf(key, sizeof key, "/%s/", nm); if (strstr(skipped, key)) return;   // (left out already: order.txt, then the folder scan)
      const char *why; if (!hd_textures_on(&why) && texture_pack(d)) {
          if (strlen(skipped) + strlen(key) < sizeof skipped) strcat(skipped, key);
          fprintf(stderr, "boot: mod %s: an HD texture pack, left out (%s)\n", nm, why); return; } }
    g_roots_len += snprintf(g_roots + g_roots_len, sizeof g_roots - g_roots_len, "%s%s", g_roots_len ? ":" : "", d); g_roots_n++;
    fprintf(stderr, "boot: mod %s: RomFS files (load order %d)\n", nm, g_roots_n);
}
static void mods_romfs(const char *mods) {
    if (getenv("DM_ROMFS_OVERLAY")) return;   // env.txt chose
    struct stat st; char d[1200];
    // Trinity games: the pack index with the mods' paths taken out (setup_game.py trinity_index), first, so it is the arc/data.trpfd the game reads
    snprintf(d, sizeof d, "%s/.trinity_index/romfs", mods); if (!stat(d, &st) && S_ISDIR(st.st_mode)) { g_roots_len += snprintf(g_roots, sizeof g_roots, "%s", d); g_roots_n++; }
    snprintf(d, sizeof d, "%s/romfs", mods); if (!stat(d, &st) && S_ISDIR(st.st_mode)) { g_roots_len += snprintf(g_roots + g_roots_len, sizeof g_roots - g_roots_len, "%s%s", g_roots_len ? ":" : "", d); g_roots_n++; }
    snprintf(d, sizeof d, "%s/order.txt", mods); FILE *f = fopen(d, "r");
    if (f) { char ln[512]; while (fgets(ln, sizeof ln, f)) { ln[strcspn(ln, "\r\n")] = 0; if (*ln) mod_root(mods, ln); } fclose(f); }
    struct dirent **ents = NULL; int n = scandir(mods, &ents, NULL, alphasort);
    for (int k = 0; k < n; k++) { mod_root(mods, ents[k]->d_name); free(ents[k]); }
    free(ents);
    if (g_roots_n) { setenv("DM_ROMFS_OVERLAY", g_roots, 1); fprintf(stderr, "boot: %d RomFS mod folder(s)\n", g_roots_n); }
}
#include "env_file.h"   // env_file(): KEY=VALUE files, each key set only when not set (env.txt, driver-check.env; tests/t_safe_start.c)
#include "mali_white.h"   // (mw_env_migrate: tested in dm/gfx/tests/t_mali_white.c)
// env.txt's float-target switches as an older version's Mali tests menu left them ("1", its only choice; no line was its Off), taken out once per env.txt before it is
//   read: the menu is Auto / On / Off now, and with the white-screen check (nvn_vk.c mali_white_vk.inc) such a leftover would hold the check's route for good. The
//   marker line DM_MALI_ENV=2 (written here, and by the menu when it sets either) says it was done: a "1" after it is the player's On. Mali phones only (gpu.txt, the
//   engine's name of this phone's GPU, says Mali): the older menu offered these there alone, and env.txt stays untouched everywhere else
static void mali_env_migrate(const char *path) {
    { char gp[1200], gpu[256] = ""; snprintf(gp, sizeof gp, "%s/gpu.txt", g_files); FILE *g = fopen(gp, "r"); if (g) { if (!fgets(gpu, sizeof gpu, g)) gpu[0] = 0; fclose(g); }
      for (char *c = gpu; *c; c++) *c = (char)tolower((unsigned char)*c);
      if (!strstr(gpu, "mali")) return; }
    static char in[16384], out[16640]; size_t k = 0; FILE *f = fopen(path, "r");
    if (f) { k = fread(in, 1, sizeof in - 1, f); fclose(f); if (k == sizeof in - 1) return; }   // (a file this long is not one the app wrote: left alone)
    in[k] = 0; int rm = 0;
    if (mw_env_migrate(in, out, sizeof out, &rm) != 1) return;
    char t[1200]; snprintf(t, sizeof t, "%s.tmp", path); FILE *w = fopen(t, "w"); if (!w) return;
    int ok = fputs(out, w) >= 0; ok &= fclose(w) == 0;
    if (!ok || rename(t, path)) { unlink(t); fprintf(stderr, "boot: [mali-white] env.txt could not be rewritten (errno %d)\n", errno); return; }
    if (rm) fprintf(stderr, "boot: [mali-white] env.txt: %s%s%s, left by an older version's Mali tests menu, taken out once: Auto (the white-screen check picks the route; the menu's On / Off win from now on)\n",
                    rm & MW_ROUTE_A ? "DM_VK_B10_AS_RGBA16=1" : "", rm == (MW_ROUTE_A | MW_ROUTE_B) ? " and " : "", rm & MW_ROUTE_B ? "DM_VK_B10_NOCOMPRESS=1" : "");
}
// Device profiles: defaults by chip, from its model (ro.soc.manufacturer / ro.soc.model, Android 12+: SM8750, MT6989, s5e9945 ...). Set without overwriting, after env.txt,
// so the player's own settings (the settings screen writes env.txt) always win; DM_PROFILE=<name> below picks one by hand, DM_PROFILE=none none.
// Mali GPUs (MediaTek, Tensor, Exynos before the 2000 series) have no BC texture support: the game's textures are decoded on upload to 4 to 8 times their size, and the
// game's memory is mirrored for the GPU (nvn_vk.c memory pools), so their profiles keep textures at half size sooner than Adreno's
//   sd8   Snapdragon 8 series (SM8xxx)                                    nothing changed: the defaults these phones were tuned on (TV mode, 8x filtering, full textures)
//   sd8old older Snapdragon 8 series (SM8150-SM8475: 855, 865, 870, 888, 8 Gen 1, 8+ Gen 1)  TV mode, handheld mode while hot; 4x filtering; full textures
//   sd7   Snapdragon 7 series from 2021 (SM73xx and later)                TV mode, handheld mode while hot; 4x filtering; full textures
//   sd6   Snapdragon 6 and 4 series, 7 series before 2021 (SM4xxx-SM72xx) handheld mode (720p); 2x filtering; half textures
//   high  Dimensity 9000 series, Exynos 2000 series (Xclipse)             TV mode, handheld mode while hot; 4x filtering; full textures
//   mid   Dimensity 8000 series and 1000-1200, Google Tensor              handheld mode; 4x filtering; half textures
//   low   other MediaTek (Dimensity 7000 / 6000, Helio), Exynos 1000 series and older Exynos   handheld mode; 2x filtering; half textures
// Other chips get no profile (as before)
static const struct prof { const char *name, *what, *graphics, *aniso, *tex; } g_profiles[] = {
    { "sd8", "Snapdragon 8 series", NULL, NULL, NULL }, { "sd8old", "older Snapdragon 8 series (855 to 8+ Gen 1)", "auto", "4", "full" }, { "sd7", "Snapdragon 7 series", "auto", "4", "full" }, { "sd6", "Snapdragon 6 / 4 series or older 7 series", "low", "2", "half" },
    { "high", "Dimensity 9000 / Exynos 2000 series", "auto", "4", "full" }, { "mid", "Dimensity 8000 series / Google Tensor", "low", "4", "half" },
    { "low", "MediaTek Dimensity 7000 and below / Exynos 1000 series", "low", "2", "half" },
};
static const struct prof *profile_of(const char *maker, const char *model) {
    char m[PROP_VALUE_MAX]; size_t k = 0; for (; model[k] && k + 1 < sizeof m; k++) m[k] = (char)tolower((unsigned char)model[k]); m[k] = 0;
    const char *name = NULL; int num;
    if ((!strncmp(m, "sm", 2) && isdigit((unsigned char)m[2])) || (!strncmp(m, "sdm", 3) && isdigit((unsigned char)m[3]))) {   // Qualcomm
        num = atoi(m + (m[1] == 'd' ? 3 : 2)); name = num >= 8000 && num < 8500 ? "sd8old" : num >= 8500 && num < 9000 ? "sd8" : num >= 7300 && num < 8000 ? "sd7" : num >= 4000 && num < 7300 ? "sd6" : num > 0 && num < 1000 ? "sd6" : NULL;   // (SDM845 and the like: before the SMxxxx names)
    } else if (!strncmp(m, "mt", 2) && isdigit((unsigned char)m[2])) {   // MediaTek: MT6983 / 6985 / 6989 / 6991 Dimensity 9000-9400; MT6889-6899 Dimensity 1000-1200 and 8000-8400
        num = atoi(m + 2); name = num == 6983 || num == 6985 || num == 6989 || num == 6991 || num == 6993 ? "high" : num >= 6889 && num <= 6899 ? "mid" : "low";
    } else if (!strncmp(m, "s5e", 3) || strstr(m, "exynos")) {   // Exynos: s5e99xx and "Exynos 2xxx" are the 2000 series (Xclipse); s5e88xx, Exynos 1xxx and 9xx Mali
        const char *d = m; while (*d && !isdigit((unsigned char)*d)) d++; if (!strncmp(m, "s5e", 3)) d = m + 3;
        num = atoi(d); name = (!strncmp(m, "s5e", 3) && num >= 9900 && num < 10000) || (strncmp(m, "s5e", 3) && num >= 2000 && num < 3000) ? "high" : "low";
    } else if (strstr(m, "tensor") || (!strncasecmp(maker, "google", 6) && *m)) name = "mid";
    for (unsigned i = 0; name && i < sizeof g_profiles / sizeof *g_profiles; i++) if (!strcmp(g_profiles[i].name, name)) return &g_profiles[i];
    return NULL;
}
// Mesa PanVK (FristOneRR, chosen as the driver: a libvulkan_panfrost package) on the chips its compatibility table marks as needing PANVK_ATOM_STRIDE=56 to start
// (Exynos 2100 = s5e9840, Exynos 1380 = s5e8835): set for them, unless env.txt (the setup screen's checkbox) already says
static void panvk_defaults(const char *model) {
    const char *drv = getenv("DM_VK_DRIVER"); if (!drv || !strstr(drv, "panfrost")) return;
    char m[PROP_VALUE_MAX]; size_t k = 0; for (; model[k] && k + 1 < sizeof m; k++) m[k] = (char)tolower((unsigned char)model[k]); m[k] = 0;
    if ((strstr(m, "s5e9840") || strstr(m, "s5e8835") || strstr(m, "exynos2100") || strstr(m, "exynos1380")) && !getenv("PANVK_ATOM_STRIDE")) {
        setenv("PANVK_ATOM_STRIDE", "56", 0); fprintf(stderr, "boot: PanVK on %s: PANVK_ATOM_STRIDE=56 (the driver's own table: needed to start on this chip)\n", model); }
}
#include "mem_tier.h"   // mem_tier_half(): half textures by default when little memory is free at start (tests/t_mem_tier.c)
// Before the profiles below, so their log lines show the result; env.txt (read before this) wins over it, as over them
static void mem_tier(void) {
    const char *e = getenv("DM_MEM_TIER"), *tq = getenv("DM_VK_TEX_QUALITY"); long av, tot;
    if (!mem_tier_read("/proc/meminfo", &av, &tot)) return;
    int half = mem_tier_half(av, tot);
    if (e && !atoi(e)) { if (half) fprintf(stderr, "boot: memory tier: %.1f GB available of %.1f GB, half textures would be the default; DM_MEM_TIER=0: left to the profile\n", av / 1048576.0, tot / 1048576.0); return; }
    if (!half) { fprintf(stderr, "boot: memory tier: %.1f GB available of %.1f GB at start: textures left to the profile\n", av / 1048576.0, tot / 1048576.0); return; }
    if (tq) { fprintf(stderr, "boot: memory tier: %.1f GB available of %.1f GB at start, but the textures setting is %s: kept\n", av / 1048576.0, tot / 1048576.0, tq); return; }
    setenv("DM_VK_TEX_QUALITY", "half", 1); setenv("DM_VK_TEX_QUALITY_AUTO", "1", 1);   // (_AUTO: not the player's half, so Arm's drivers before r49 keep full: Q_TEX_HALF_UNSAFE)
    fprintf(stderr, "boot: memory tier: only %.1f GB available of %.1f GB at start (under 2.5 GB, a phone under 9 GB): half textures by default (Textures = Full in the menu, or DM_MEM_TIER=0, keeps full)\n", av / 1048576.0, tot / 1048576.0);
}
static void device_profile(void) {
    char maker[PROP_VALUE_MAX] = "", model[PROP_VALUE_MAX] = ""; __system_property_get("ro.soc.manufacturer", maker); __system_property_get("ro.soc.model", model);
    panvk_defaults(model);
    mem_tier();
    const char *force = getenv("DM_PROFILE"); const struct prof *p = NULL;
    // The app measured the phone (GameSettings.java profile: CPU, GPU and memory tiers, the display's modes) and handed the result over as DM_AUTO_*: those are the
    // defaults, set without overwriting, so the player's own choices (env.txt, read before this) win. The table below stays for a start without them (an older app)
    const char *ag = getenv("DM_AUTO_GRAPHICS");
    if (ag && !force) {
        const char *at = getenv("DM_AUTO_TEX"), *aa = getenv("DM_AUTO_ANISO"), *af = getenv("DM_AUTO_FPS");
        setenv("DM_GRAPHICS", ag, 0); if (aa) setenv("DM_VK_MAX_ANISO", aa, 0);
        if (at && !strcmp(at, "half") && !getenv("DM_VK_TEX_QUALITY")) { setenv("DM_VK_TEX_QUALITY", "half", 1); setenv("DM_VK_TEX_QUALITY_AUTO", "1", 1); }   // (the profile's half: nvn_vk.c may keep full on a driver that crashed with it)
        if (af && atoi(af) >= 60 && !power_saver() && !getenv("DM_FPS_TARGET")) { setenv("DM_FPS_TARGET", af, 1); setenv("DM_AUTO_FPS_SET", "1", 1); }   // (battery saver: Auto stays 30; DM_AUTO_FPS_SET: 60 came from Auto, not the player)   // (60 only when the app found a top phone and a display that 60 divides)
        { const char *as = getenv("DM_AUTO_SCALER"); if (as && *as) setenv("DM_SCALER", as, 0); }   // (SGSR on a mid or top GPU, the lighter Sharp on a low one)
        { const char *al = getenv("DM_AUTO_LOWSPEC"); if (al && *al) setenv("DM_LOWSPEC", al, 0); }   // (Legends: Arceus on a low CPU or GPU: Lower specs, android_main.c lowspec)   // (SGSR on a mid or top GPU, the lighter Sharp on a low one)
        const char *g = getenv("DM_GRAPHICS"), *tq = getenv("DM_VK_TEX_QUALITY"), *an = getenv("DM_VK_MAX_ANISO"), *ft = getenv("DM_FPS_TARGET"), *w = getenv("DM_AUTO_WHAT"), *hz = getenv("DM_DISPLAY_HZ");
        fprintf(stderr, "boot: this phone: %s (chip %s %s; display modes %s Hz): graphics %s, anisotropic filtering %sx, textures %s, frame rate %s\n", w ? w : "?", *maker ? maker : "?",
                *model ? model : "unknown", hz ? hz : "?", g ? g : "default", an ? an : "default", tq ? tq : "full", ft ? ft : "30");
        return;
    }
    if (force) { for (unsigned i = 0; i < sizeof g_profiles / sizeof *g_profiles; i++) if (!strcmp(g_profiles[i].name, force)) p = &g_profiles[i]; }
    else p = profile_of(maker, model);
    // Then the phone's memory, for every phone (GameSettings.java auto(), the same rule): the game's textures are the largest use of memory and a phone that runs out
    // ends the game. Under 8 GB: half textures whatever the chip; under 6 GB also handheld mode. 12 GB or more: full textures unless the chip is a weak one (sd6, low)
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE); double gb = pages > 0 && psz > 0 ? (double)pages * psz / 1e9 : 0;
    const char *graphics = p ? p->graphics : NULL, *tex = p ? p->tex : NULL, *aniso = p ? p->aniso : NULL;   // NULL: the defaults (TV mode, 8x, full)
    int weak = p && (!strcmp(p->name, "sd6") || !strcmp(p->name, "low"));
    if (gb > 0 && gb < 7.5) tex = "half";
    else if (gb >= 11.5 && !weak) tex = NULL;
    if (gb > 0 && gb < 5.5) graphics = "low";
    if (graphics) setenv("DM_GRAPHICS", graphics, 0);
    if (aniso) setenv("DM_VK_MAX_ANISO", aniso, 0);
    if (tex && !getenv("DM_VK_TEX_QUALITY")) { setenv("DM_VK_TEX_QUALITY", tex, 1); if (!strcmp(tex, "half")) setenv("DM_VK_TEX_QUALITY_AUTO", "1", 1); }
    const char *g = getenv("DM_GRAPHICS"), *tq = getenv("DM_VK_TEX_QUALITY"), *an = getenv("DM_VK_MAX_ANISO");
    fprintf(stderr, "boot: device profile %s, %s (chip %s %s%s, %.1f GB memory): graphics %s, anisotropic filtering %sx, textures %s\n", p ? p->name : "none", p ? p->what : "chip not in the table",
            *maker ? maker : "?", *model ? model : "unknown", force ? ", DM_PROFILE" : "", gb, g ? g : "default", an ? an : "default", tq ? tq : "full");
}
#if DM_GAME_PROGRAM_ID == 0x01001F5010DFA000ull   // (Legends: Arceus only: its RomFS files)
// "Draw distance" (the settings' DM_ARC_DRAW: near / far / veryfar; Default: none): the player's OWN config files, read from their RomFS, with the distance values
//   rewritten, laid over the RomFS like Lower specs (nothing of the game's comes with the app). The files are FlatBuffers of named settings, each {name, type, ?, min,
//   max, values: a vector of strings}: a value is changed by appending its new string at the end of the buffer and pointing the vector's slot at it (offsets point
//   forward, so the rest stays valid). Changed: bin/misc/app_config/field_common_config.bin kTreeVisibleDistance [0, 200], kTreeFadeDistanceFar [180, 200],
//   kDistanceOfCameraToTerrainLodPoint 7 (and for Near the people / Pokémon load radii 60 / 40), field_anime_framerate_config.bin distanceBaseValue 60 (where animation
//   slows down); each times the setting's factor, never over the setting's own max. Read at start: a change applies at the next start
static int write_all(const char *path, const unsigned char *b, size_t n) {
    char d[1200]; snprintf(d, sizeof d, "%s", path);
    for (char *s = d + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(d, 0770); *s = '/'; }
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    int ok = fwrite(b, 1, n, f) == n; return fclose(f) == 0 && ok;
}
typedef struct { unsigned char *d; size_t n, cap; } cfgbuf;
static uint32_t cb_u32(const cfgbuf *b, size_t o) { uint32_t v; memcpy(&v, b->d + o, 4); return v; }
static int32_t cb_s32(const cfgbuf *b, size_t o) { int32_t v; memcpy(&v, b->d + o, 4); return v; }
static uint16_t cb_u16(const cfgbuf *b, size_t o) { uint16_t v; memcpy(&v, b->d + o, 2); return v; }
static int cb_str(const cfgbuf *b, size_t o, char *out, size_t n) {   // the FlatBuffer string at o, 0 if none
    if (o + 4 > b->n) return 0; uint32_t L = cb_u32(b, o); if (L == 0 || L >= n || o + 4 + L >= b->n || b->d[o + 4 + L]) return 0;
    memcpy(out, b->d + o + 4, L); out[L] = 0; return 1;
}
// the entry table named `name`: its values vector (*vec) and its max (*mx); 0 if not found
static int cb_entry(const cfgbuf *b, const char *name, size_t *vec, double *mx) {
    size_t L = strlen(name);
    for (size_t so = 0; so + 4 + L < b->n; so += 4) {
        if (cb_u32(b, so) != L || memcmp(b->d + so + 4, name, L) || b->d[so + 4 + L]) continue;
        for (size_t p = 0; p + 4 <= so; p += 4) {
            if (p + cb_u32(b, p) != so) continue;   // the slot pointing at the name
            for (size_t t = p >= 64 ? p - 64 : 0; t <= p; t += 4) {   // the table whose field 0 is that slot
                int64_t vt = (int64_t)t - cb_s32(b, t); if (vt < 0 || (size_t)vt + 16 > b->n) continue;
                uint16_t vsz = cb_u16(b, (size_t)vt); if (vsz < 16 || (size_t)vt + vsz > b->n) continue;
                if (t + cb_u16(b, (size_t)vt + 4) != p) continue;
                uint16_t fmx = cb_u16(b, (size_t)vt + 4 + 8), fv = cb_u16(b, (size_t)vt + 4 + 10); if (!fv) return 0;
                char m[32]; *mx = fmx && cb_str(b, t + fmx + cb_u32(b, t + fmx), m, sizeof m) ? atof(m) : 1e9;
                size_t va = t + fv; *vec = va + cb_u32(b, va); return *vec + 4 <= b->n; } } }
    return 0;
}
static int cb_append(cfgbuf *b, const char *s) {   // a new string at the end (4-aligned); its offset, or -1
    size_t L = strlen(s), need = 4 + L + 1, at = (b->n + 3) & ~(size_t)3, end = (at + need + 3) & ~(size_t)3;
    if (end > b->cap) { size_t c = end + 256; unsigned char *d = realloc(b->d, c); if (!d) return -1; b->d = d; b->cap = c; }
    memset(b->d + b->n, 0, end - b->n); uint32_t l32 = (uint32_t)L; memcpy(b->d + at, &l32, 4); memcpy(b->d + at + 4, s, L); b->n = end; return (int)at;
}
// every value of the entry times k, as an integer if `whole`, never over its max; the count changed
static int cb_scale(cfgbuf *b, const char *name, double k, int whole) {
    size_t vec; double mx; if (!cb_entry(b, name, &vec, &mx)) return 0;
    uint32_t cnt = cb_u32(b, vec); int n = 0;
    for (uint32_t i = 0; i < cnt && i < 8; i++) {
        size_t es = vec + 4 + 4 * i; char v[32]; if (!cb_str(b, es + cb_u32(b, es), v, sizeof v)) continue;
        double x = atof(v) * k; if (x > mx) x = mx;
        char o[32]; if (whole) snprintf(o, sizeof o, "%d", (int)(x + (x >= 0 ? 0.5 : -0.5))); else snprintf(o, sizeof o, "%g", x);
        int at = cb_append(b, o); if (at < 0) return n;
        uint32_t rel = (uint32_t)((size_t)at - es); memcpy(b->d + es, &rel, 4); n++;
    }
    return n;
}
static int cb_load(const char *path, cfgbuf *b) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 16 || n > (1 << 20)) { fclose(f); return 0; }
    b->d = malloc((size_t)n + 512); b->cap = (size_t)n + 512; b->n = (size_t)n;
    int ok = b->d && fread(b->d, 1, (size_t)n, f) == (size_t)n; fclose(f); return ok;
}
static void draw_distance(void) {
    // DM_DRAW_DIST: percent (50-200, the settings' slider; 100 the game's own); the older DM_ARC_DRAW near / far / veryfar are 75 / 150 / 200
    char pct[16] = ""; { const char *d = getenv("DM_DRAW_DIST"), *o = getenv("DM_ARC_DRAW");
        if (d && atoi(d) >= 50 && atoi(d) <= 200 && atoi(d) != 100) snprintf(pct, sizeof pct, "%d", atoi(d));
        else if (!d && o && *o) snprintf(pct, sizeof pct, "%s", !strcmp(o, "near") ? "75" : !strcmp(o, "far") ? "150" : !strcmp(o, "veryfar") ? "200" : ""); }
    const char *e = pct[0] ? pct : NULL;
    if (!e) {   // Default: the game's own, unless the fps guard remembered that this phone needs Near to hold 30 (lock30_draw)
        char p[1100], cur[16] = ""; snprintf(p, sizeof p, "%s/auto_draw.txt", g_files); FILE *f = fopen(p, "r");
        if (f) { if (!fgets(cur, sizeof cur, f)) cur[0] = 0; fclose(f); }
        if (strncmp(cur, "near", 4) && atoi(cur) < 50) return;
        e = "75"; fprintf(stderr, "boot: draw distance near, chosen by the fps guard (this phone's CPU could not hold 30 at the game's own distance; Settings -> Draw distance overrides it)\n"); }
    double k = atoi(e) / 100.0; if (!(k >= 0.5 && k <= 2.0) || k == 1.0) return;
    const char *ls = getenv("DM_LOWSPEC"); if (ls && atoi(ls) > 0) { fprintf(stderr, "boot: draw distance %s left out: Lower specs is on (its own files)\n", e); return; }
    const char *rf = getenv("DM_ROMFS"); char rom[1100]; snprintf(rom, sizeof rom, "%s", rf && *rf ? rf : ""); if (!*rom) snprintf(rom, sizeof rom, "%s/RomFS", g_game);
    char root[1100], in[1300], out[1300]; snprintf(root, sizeof root, "%s/drawdist/romfs", g_files);
    int nc = 0, na = 0; cfgbuf b = { 0 };
    snprintf(in, sizeof in, "%s/bin/misc/app_config/field_common_config.bin", rom);
    if (cb_load(in, &b)) {
        nc += cb_scale(&b, "kTreeVisibleDistance", k, 0); nc += cb_scale(&b, "kTreeFadeDistanceFar", k, 0); nc += cb_scale(&b, "kDistanceOfCameraToTerrainLodPoint", k, 0);
        if (k < 1) { nc += cb_scale(&b, "kArea00NpcLoadRadius", k, 1); nc += cb_scale(&b, "kArea00NpcPokeLoadRadius", k, 1); }   // (Near: fewer people and Pokémon loaded too)
        snprintf(out, sizeof out, "%s/bin/misc/app_config/field_common_config.bin", root); if (nc && !write_all(out, b.d, b.n)) nc = -1; free(b.d); }
    memset(&b, 0, sizeof b);
    snprintf(in, sizeof in, "%s/bin/misc/app_config/field_anime_framerate_config.bin", rom);
    if (cb_load(in, &b)) { na = cb_scale(&b, "distanceBaseValue", k, 0);
        snprintf(out, sizeof out, "%s/bin/misc/app_config/field_anime_framerate_config.bin", root); if (na && !write_all(out, b.d, b.n)) na = -1; free(b.d); }
    if (nc <= 0 && na <= 0) { fprintf(stderr, "boot: [RED] draw distance %s: the player's config files could not be read or changed (%s)\n", e, rom); return; }
    const char *o = getenv("DM_ROMFS_OVERLAY"); char all[4096];
    snprintf(all, sizeof all, "%s%s%s", root, o && *o ? ":" : "", o && *o ? o : "");   // (ahead of the other overlays)
    setenv("DM_ROMFS_OVERLAY", all, 1);
    fprintf(stderr, "boot: draw distance %s (x%.2f): %d value(s) in field_common_config, %d in field_anime_framerate_config, from the player's own files (RomFS overlay %s)\n", e, k, nc, na, root);
}
// "Lower specs" (the settings' DM_LOWSPEC: 1 lower, 2 lowest), for phones that cannot hold the game's frame rate, from the player's OWN config files (as Draw distance;
//   nothing of the game's or of a mod comes with the app): the people / Pokémon load radii and the trees' visible distance halved (field_common_config.bin
//   kArea00NpcLoadRadius 60 → 30, kArea00NpcPokeLoadRadius 40 → 20, kTreeVisibleDistance 200 → 100), and the shadow cascades' splits (field_shadow_config.bin
//   cascadeShadowSplit1..4) set as the POTATO mod set them: lower "-1." / "-1." / "-1." / "-1..", lowest the same with split 1 "1..." (the same strings, so the
//   game reads them exactly as it read the mod's)
static int cb_set(cfgbuf *b, const char *name, const char *val) {   // the entry's first value replaced by val; 1 done
    size_t vec; double mx; if (!cb_entry(b, name, &vec, &mx) || cb_u32(b, vec) < 1) return 0;
    size_t es = vec + 4; int at = cb_append(b, val); if (at < 0) return 0;
    uint32_t rel = (uint32_t)((size_t)at - es); memcpy(b->d + es, &rel, 4); return 1;
}
static const char *own_romfs(char *buf, size_t n) { const char *rf = getenv("DM_ROMFS"); if (rf && *rf) snprintf(buf, n, "%s", rf); else snprintf(buf, n, "%s/RomFS", g_game); return buf; }
static void overlay_first(const char *root) {
    const char *o = getenv("DM_ROMFS_OVERLAY"); char all[4096];
    snprintf(all, sizeof all, "%s%s%s", root, o && *o ? ":" : "", o && *o ? o : "");   // (ahead of any other overlay)
    setenv("DM_ROMFS_OVERLAY", all, 1);
}
static void lowspec(void) {
    const char *e = getenv("DM_LOWSPEC"); int lv = e ? atoi(e) : 0; if (lv < 1 || lv > 2) return;
    char rom[1100], root[1100], in[1300], out[1300]; own_romfs(rom, sizeof rom); snprintf(root, sizeof root, "%s/lowspec/romfs", g_files);
    int nc = 0, ns = 0; cfgbuf b = { 0 };
    snprintf(in, sizeof in, "%s/bin/misc/app_config/field_common_config.bin", rom);
    if (cb_load(in, &b)) {
        nc += cb_scale(&b, "kArea00NpcLoadRadius", 0.5, 1); nc += cb_scale(&b, "kArea00NpcPokeLoadRadius", 0.5, 1); nc += cb_scale(&b, "kTreeVisibleDistance", 0.5, 0);
        snprintf(out, sizeof out, "%s/bin/misc/app_config/field_common_config.bin", root); if (nc && !write_all(out, b.d, b.n)) nc = -1; free(b.d); }
    memset(&b, 0, sizeof b);
    snprintf(in, sizeof in, "%s/bin/misc/app_config/field_shadow_config.bin", rom);
    if (cb_load(in, &b)) {
        ns += cb_set(&b, "cascadeShadowSplit4", "-1."); ns += cb_set(&b, "cascadeShadowSplit3", "-1."); ns += cb_set(&b, "cascadeShadowSplit2", "-1.");
        ns += cb_set(&b, "cascadeShadowSplit1", lv == 2 ? "1..." : "-1..");
        snprintf(out, sizeof out, "%s/bin/misc/app_config/field_shadow_config.bin", root); if (ns && !write_all(out, b.d, b.n)) ns = -1; free(b.d); }
    if (nc <= 0 || ns <= 0) { fprintf(stderr, "boot: [RED] lower specs: the player's config files could not be read or changed (%s): left out\n", rom); return; }
    overlay_first(root);
    fprintf(stderr, "boot: lower specs %d: shorter draw distance (%d values), %s (%d values), from the player's own files (RomFS overlay %s)\n", lv, nc, lv == 2 ? "no shadows" : "lighter shadows", ns, root);
}
// "Style move effects" (the settings' DM_ARC_STYLE_FX: off unless 1): the strong / agile style moves' start sequences (bin/effect/battle/sequence/cmn_battle_waza_
//   {pow,teq}_start.wbin) cut down to what the No More Styles 1.2 mod left, from the PLAYER'S OWN files: of the sequence's 39 steps, its first empty one and the one
//   holding a single command 0x21 (the end of the sequence), that command's two arguments set as the mod set them (0, 32; the game's own are 9, 41). With the game's own effects grey boxes covered the screen on Adreno
//   and Xclipse (Arceus-Native-Android #14 #15 #20); the moves and their power are unchanged, only the full-screen animation as they start goes. A FlatBuffer cut in
//   place: the steps vector's count and its two slots rewritten (the kept steps lie after it, offsets point forward)
static size_t fb_field(const cfgbuf *b, size_t t, int i) {   // a table field's address, 0 if absent
    int64_t vt = (int64_t)t - cb_s32(b, t); if (vt < 0 || (size_t)vt + 4 > b->n) return 0;
    uint16_t vsz = cb_u16(b, (size_t)vt); if ((size_t)(4 + 2 * i + 2) > vsz) return 0;
    uint16_t fo = cb_u16(b, (size_t)vt + 4 + 2 * (size_t)i); return fo ? t + fo : 0;
}
static size_t fb_ref(const cfgbuf *b, size_t a) { return a && a + 4 <= b->n ? a + cb_u32(b, a) : 0; }   // where a uoffset field points
static int seq_strip(cfgbuf *b) {
    size_t root = cb_u32(b, 0), vf = fb_field(b, root, 0), v = fb_ref(b, vf); if (!v || v + 4 > b->n) return 0;
    uint32_t n = cb_u32(b, v); size_t empty = 0, end = 0;
    for (uint32_t k = 0; k < n && v + 8 + 4 * k <= b->n; k++) {
        size_t st = fb_ref(b, v + 4 + 4 * k), cf = st ? fb_field(b, st, 2) : 0, cv = fb_ref(b, cf); if (!st) continue;
        uint32_t cn = cv && cv + 4 <= b->n ? cb_u32(b, cv) : 0;
        if (!empty && cf && cn == 0) empty = st;
        if (!end && cn == 1) { size_t cmd = fb_ref(b, cv + 4), op = cmd ? fb_field(b, cmd, 0) : 0;
            if (op && b->d[op] == 0x21) { end = st; size_t args = fb_ref(b, fb_field(b, cmd, 1));
                for (int i = 0; args && i < 2; i++) { size_t a = fb_field(b, args, i); uint32_t x = i ? 32u : 0u; if (a && a + 4 <= b->n) memcpy(b->d + a, &x, 4); } } }   // (as the mod: 0, 32; the game's own 9, 41)
    }
    if (!empty || !end || empty < v + 12 || end < v + 12) return 0;
    uint32_t two = 2, r0 = (uint32_t)(empty - (v + 4)), r1 = (uint32_t)(end - (v + 8));
    memcpy(b->d + v, &two, 4); memcpy(b->d + v + 4, &r0, 4); memcpy(b->d + v + 8, &r1, 4); return 1;
}
static void style_fx(void) {
    const char *e = getenv("DM_ARC_STYLE_FX"); if (e && !strcmp(e, "1")) { fprintf(stderr, "boot: style move effects: the game's own (DM_ARC_STYLE_FX=1)\n"); return; }
    static const char *const F[2] = { "cmn_battle_waza_pow_start.wbin", "cmn_battle_waza_teq_start.wbin" };
    char rom[1100], root[1100], in[1300], out[1300]; own_romfs(rom, sizeof rom); snprintf(root, sizeof root, "%s/nostyles/romfs", g_files);
    int ok = 1;
    for (int i = 0; i < 2 && ok; i++) { cfgbuf b = { 0 };
        snprintf(in, sizeof in, "%s/bin/effect/battle/sequence/%s", rom, F[i]); snprintf(out, sizeof out, "%s/bin/effect/battle/sequence/%s", root, F[i]);
        ok = cb_load(in, &b) && seq_strip(&b) && write_all(out, b.d, b.n); free(b.d); }
    if (!ok) { fprintf(stderr, "boot: [RED] style move effects: the player's sequence files could not be read or cut (%s): the game's own\n", rom); return; }
    overlay_first(root);
    fprintf(stderr, "boot: style move effects off: the strong / agile style moves start without their full-screen animation, from the player's own files (RomFS overlay %s)\n", root);
}
#else
static void lowspec(void) { }
static void style_fx(void) { }
static void draw_distance(void) { }
#endif
// "No reflections" (profile key flat_normals, the menu's DM_FLAT_NORMALS, on unless 0): the game's terrain packs with their ground and bark normal maps made flat, made by
//   the setup from the player's own files (dm/android/tools/flat_normals.py) into <game>/flat_normals, laid over the RomFS after every mod (hle_fs.c DM_ROMFS_OVERLAY, first
//   root wins, so a mod's own packs come first). On this engine the detail of those normal maps lit shadowed ground purple (docs/HANDOFF.md, problem 3)
static void flat_normals(void) {
    const char *e = getenv("DM_FLAT_NORMALS"); if (e && !strcmp(e, "0")) { fprintf(stderr, "boot: no reflections off (DM_FLAT_NORMALS=0): the ground's normal maps as the game has them\n"); return; }
    char root[1100], done[1200]; struct stat st; snprintf(root, sizeof root, "%s/flat_normals/romfs", g_game); snprintf(done, sizeof done, "%s/flat_normals/done", g_game);
    if (stat(root, &st) || !S_ISDIR(st.st_mode) || stat(done, &st)) return;   // (none made: not this game, or the setup has not made them yet)
    const char *o = getenv("DM_ROMFS_OVERLAY"); char all[4096];
    snprintf(all, sizeof all, "%s%s%s", o && *o ? o : "", o && *o ? ":" : "", root);
    setenv("DM_ROMFS_OVERLAY", all, 1);
    fprintf(stderr, "boot: no reflections: the flat normal maps of %s\n", root);
}
static pid_t g_game_tid;   // the game's main thread (performance hints below)
void dm_boot_mark(const char *what) __attribute__((weak));   // (nvn_vk.c: a "[start]" line with the milliseconds since the process started)
static void *host_thread(void *a) {
    g_game_tid = gettid();
    if (dm_boot_mark) dm_boot_mark("host thread starts");
    (void)a; static char alt[65536]; stack_t ss = {.ss_sp = alt, .ss_size = sizeof alt}; sigaltstack(&ss, 0);
    char patched[1100]; snprintf(patched, sizeof patched, "%s/patched", g_game);
    char *argv[3] = { "dmhost", patched, NULL };
    int rc = dm_main(2, argv);
    fprintf(stderr, "boot: dm_main returned %d\n", rc);
    return NULL;
}
// The log where the player can reach it: Android/data is out of reach of the phone's Files app on recent Android versions, Documents and Download are not. At each
// start the previous run's log (the one that ended in a crash, if one did) goes to <logs>/<game>-log-previous.txt (PokemonShield-…); while playing, <logs>/<game>-log.txt
// is refreshed every 5 seconds, with a memory line (a run the system ends for memory leaves no crash report, only memory that kept rising). <logs> is the app's own
// folder Documents/<docs_dir>/Logs (Galar Native, NXNative for Legends: Arceus, Paldea Native: the profile's docs_dir, as the save folder), kept apart from everything
// else in Download; Download when that folder cannot be made or written
#define DL "/storage/emulated/0/Download"
#define DOCS "/storage/emulated/0/Documents"
static const char *logs_dir(void) {   // (decided once: every log of this run in one place)
    static char d[200]; static int done;
    if (done) return d[0] ? d : DL;
    done = 1;
    char a[160]; snprintf(a, sizeof a, DOCS "/%s", DM_GAME_DOCS_DIR);
    snprintf(d, sizeof d, "%s/Logs", a);
    mkdir(DOCS, 0770); mkdir(a, 0770); mkdir(d, 0770);
    char t[240]; snprintf(t, sizeof t, "%s/.write-test", d);
    int f = open(t, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (f < 0) { d[0] = 0; return DL; }
    close(f); unlink(t); return d;
}
// A name in the logs folder this install may write. Android lets an app overwrite only files it created itself: after the game was uninstalled and installed again (or a
//   file came from another app), opening the old <game>-log.txt for writing fails, and the logs players attach stayed days old (#155). The first of <game>-<what>.txt,
//   <game>-<what>-2.txt … -9.txt that opens for writing: this install's own file, or a new one
static char g_dl_note[320];
static void dl_name(char *out, size_t n, const char *what) {
    const char *dir = logs_dir();
    for (int k = 1; k <= 9; k++) {
        if (k == 1) snprintf(out, n, "%s/%s-%s.txt", dir, g_stem, what); else snprintf(out, n, "%s/%s-%s-%d.txt", dir, g_stem, what, k);
        int f = open(out, O_WRONLY | O_CREAT, 0644);
        if (f >= 0) { close(f); if (k > 1) snprintf(g_dl_note, sizeof g_dl_note, "log: %s/%s-%s.txt belongs to an earlier install: written as %s", dir, g_stem, what, out); return; }
    }
    snprintf(out, n, "%s/%s-%s.txt", dir, g_stem, what);
}
static void copy_file(const char *a, const char *b) {
    int i = open(a, O_RDONLY), o = i < 0 ? -1 : open(b, O_WRONLY | O_CREAT | O_TRUNC, 0644); char buf[65536]; ssize_t n;
    if (o >= 0) while ((n = read(i, buf, sizeof buf)) > 0) if (write(o, buf, (size_t)n) != n) break;
    if (i >= 0) close(i); if (o >= 0) close(o);
}
// ---------- Save folder (the menu's Saves -> Save folder, DM_SAVE_SHARED=1; Arceus-Native-Android#8) ----------
// The save can live in Documents/<docs_dir>/<title id>/save (the profile's docs_dir: Galar Native for Sword / Shield, NXNative for Legends: Arceus) instead of the app's own folder, where a sync tool (Syncthing) or a file manager can reach it while the game is
// closed. Which of the two holds the save now is written in <files>/save_place.txt ("app" / "shared"). A switch takes effect at the next start, before the game mounts
// its save (a game running writes it at any moment): the save is copied to the other place, and whatever was there first is renamed to save.backup-<date>, never
// deleted; the old place keeps its copy as it was. Shared needs "All files access" (the menu asks for it): when the folder cannot be written, the save stays in the
// app's folder and the log says why. Only plain files and folders are copied
static int cp_file(const char *a, const char *b) {
    int i = open(a, O_RDONLY); if (i < 0) return 0;
    int o = open(b, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (o < 0) { close(i); return 0; }
    char buf[65536]; ssize_t n; int ok = 1;
    while ((n = read(i, buf, sizeof buf)) > 0) if (write(o, buf, (size_t)n) != n) { ok = 0; break; }
    if (n < 0) ok = 0;
    close(i); if (close(o)) ok = 0; return ok;
}
static int cp_tree(const char *a, const char *b, int *files) {
    if (mkdir(b, 0770) && errno != EEXIST) return 0;
    DIR *d = opendir(a); if (!d) return 0; struct dirent *e; int ok = 1;
    while (ok && (e = readdir(d))) { if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char pa[1400], pb[1400]; struct stat st; snprintf(pa, sizeof pa, "%s/%s", a, e->d_name); snprintf(pb, sizeof pb, "%s/%s", b, e->d_name);
        if (stat(pa, &st)) { ok = 0; break; }
        if (S_ISDIR(st.st_mode)) ok = cp_tree(pa, pb, files); else if (S_ISREG(st.st_mode)) { ok = cp_file(pa, pb); if (ok) (*files)++; } }
    closedir(d); return ok;
}
static int dir_has_files(const char *p) { DIR *d = opendir(p); if (!d) return 0; struct dirent *e; int n = 0; while ((e = readdir(d))) if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) { n = 1; break; } closedir(d); return n; }
static void move_aside(const char *p) {   // an existing save folder becomes save.backup-<date> next to it
    if (!dir_has_files(p)) { rmdir(p); return; }
    char b[1400]; time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm);
    snprintf(b, sizeof b, "%s.backup-%04d%02d%02d-%02d%02d%02d", p, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    if (rename(p, b)) fprintf(stderr, "boot: [RED] save folder: could not keep %s as a backup (errno %d)\n", p, errno); else fprintf(stderr, "boot: save folder: the earlier save there kept as %s\n", b);
}
static int g_save_imported;   // a save archive was installed into the app's folder at this start: with the save in Documents, it is copied there
static void save_place(void) {
    char app[1100], shared[1100], par[1100], st[1200], cur[16] = "app"; const char *e = getenv("DM_SAVE_SHARED"); int want = e && atoi(e) == 1;
    snprintf(app, sizeof app, "%s/save", g_prof); snprintf(st, sizeof st, "%s/save_place.txt", g_prof);   // (the chosen player's: choose_profile)
    // (Documents/<the profile's docs_dir>: Galar Native, NXNative (Legends: Arceus), Paldea Native: as each app's releases made it; Paldea Native took it from
    //   DM_APP_NAME, its app name, which is the same folder for it. Arceus Native's app name differs from its folder, so the profile decides)
    const char *an = DM_GAME_DOCS_DIR;
    char top[300]; snprintf(top, sizeof top, "/storage/emulated/0/Documents/%s", an);
    { const char *pid = getenv("DM_PROGRAM_ID"); char up[20] = ""; if (pid) for (int k = 0; pid[k] && k < 16; k++) up[k] = (char)toupper((unsigned char)pid[k]);   // (Galar: the running game, Sword or Shield)
      if (up[0]) snprintf(par, sizeof par, "%s/%s", top, up); else snprintf(par, sizeof par, "%s/%016llX", top, (unsigned long long)DM_GAME_PROGRAM_ID); }
    char gpar[1100]; snprintf(gpar, sizeof gpar, "%s", par);   // another player than Player 1: <title id>/profiles/<id>/save (GameSettings.sharedSave)
    if (strcmp(g_prof_id, "1")) snprintf(par, sizeof par, "%s/profiles/%s", gpar, g_prof_id);
    snprintf(shared, sizeof shared, "%s/save", par);
    { FILE *f = fopen(st, "r"); if (f) { if (fscanf(f, "%15s", cur) != 1) strcpy(cur, "app"); fclose(f); } }
    int in_shared = !strcmp(cur, "shared");
    if (want && !in_shared) {   // to Documents
        mkdir("/storage/emulated/0/Documents", 0770); mkdir(top, 0770); mkdir(gpar, 0770);
        if (strcmp(g_prof_id, "1")) { char pp[1200]; snprintf(pp, sizeof pp, "%s/profiles", gpar); mkdir(pp, 0770); } mkdir(par, 0770);
        char probe[1200]; snprintf(probe, sizeof probe, "%s/.write-test", par); int fd = open(probe, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { fprintf(stderr, "boot: [RED] save folder: %s cannot be written (errno %d: All files access not given?): the save stays in the app's folder\n", par, errno); return; }
        close(fd); unlink(probe);
        move_aside(shared); int n = 0;
        if (dir_has_files(app) && !cp_tree(app, shared, &n)) { fprintf(stderr, "boot: [RED] save folder: copying the save to %s failed (errno %d): it stays in the app's folder\n", shared, errno); return; }
        if (!dir_has_files(app)) mkdir(shared, 0770);
        FILE *f = fopen(st, "w"); if (f) { fputs("shared\n", f); fclose(f); } in_shared = 1;
        fprintf(stderr, "boot: save folder: the save copied to %s (%d files); the app's folder keeps its copy as it was\n", shared, n);
    } else if (!want && in_shared) {   // back to the app's folder
        int n = 0;
        if (dir_has_files(shared)) { move_aside(app); if (!cp_tree(shared, app, &n)) { fprintf(stderr, "boot: [RED] save folder: copying the save back from %s failed (errno %d): it stays there\n", shared, errno); setenv("DM_SAVE", shared, 1); return; } }
        FILE *f = fopen(st, "w"); if (f) { fputs("app\n", f); fclose(f); } in_shared = 0;
        fprintf(stderr, "boot: save folder: the save copied back to the app's folder (%d files); %s keeps its copy\n", n, shared);
    }
    if (in_shared && g_save_imported && dir_has_files(app)) {   // an import landed in the app's folder: it becomes the save in Documents too
        int n = 0; move_aside(shared);
        if (cp_tree(app, shared, &n)) fprintf(stderr, "boot: save folder: the imported save copied to %s (%d files)\n", shared, n);
        else fprintf(stderr, "boot: [RED] save folder: copying the imported save to %s failed (errno %d)\n", shared, errno);
    }
    if (in_shared) {
        if (!dir_has_files(shared) && dir_has_files(app)) fprintf(stderr, "boot: [RED] save folder: %s is empty or gone: the app's folder is used\n", shared);
        else { setenv("DM_SAVE", shared, 1); fprintf(stderr, "boot: save folder %s\n", shared); }
    }
}
static long kb_of(const char *path, const char *key) {
    FILE *f = fopen(path, "r"); char ln[256]; long v = -1; size_t kl = strlen(key);
    while (f && fgets(ln, sizeof ln, f)) if (!strncmp(ln, key, kl)) { v = atol(ln + kl); break; }
    if (f) fclose(f); return v;
}
// The phone's heat (Android 11+ thermal status, 12+ headroom: 1.0 = where throttling starts), looked up at run time, for the log line: a slowdown from heat shows as heat
static void *g_th_mgr; static int (*g_th_st)(void *); static float (*g_th_hr)(void *, int);
// Headroom asked at most once a second (Android answers NaN to a second call within one, and the guard's reading and the log line's come a moment apart): within
//   the second, the same forecast gets the answer already had, another NaN. Called from the log thread only (log_mirror: the guard and the [mem] line)
static uint64_t g_hr_t; static float g_hr_v = -1.0f; static int g_hr_fs = -1;
static float th_headroom(int fs) {
    if (!g_th_mgr || !g_th_hr) return -1.0f;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t t = (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
    if (g_hr_fs >= 0 && t - g_hr_t < 1100) return fs == g_hr_fs ? g_hr_v : NAN;
    g_hr_v = g_th_hr(g_th_mgr, fs); g_hr_t = t; g_hr_fs = fs; return g_hr_v;
}
static const char *thermal_text(char *buf, size_t n) {
    static void *mgr; static int (*st)(void *); static float (*hr)(void *, int); static int tried;
    if (!tried++) { void *lib = dlopen("libandroid.so", RTLD_NOW); void *(*acq)(void) = lib ? (void *(*)(void))dlsym(lib, "AThermal_acquireManager") : NULL;
        st = lib ? (int (*)(void *))dlsym(lib, "AThermal_getCurrentThermalStatus") : NULL; hr = lib ? (float (*)(void *, int))dlsym(lib, "AThermal_getThermalHeadroom") : NULL; mgr = acq ? acq() : NULL; }
    g_th_mgr = mgr; g_th_st = st; g_th_hr = hr;
    if (!mgr || !st) return "unknown";
    static const char *names[] = { "none", "light", "moderate", "severe", "critical", "emergency", "shutdown" }; int v = st(mgr);
    float h10 = hr ? th_headroom(10) : -1.0f;   // headroom forecast 10 s ahead; the guard's reading of now when it asked this second
    if (h10 == h10) snprintf(buf, n, "%s, headroom %.2f", v >= 0 && v <= 6 ? names[v] : "?", h10);
    else snprintf(buf, n, "%s, headroom now %.2f", v >= 0 && v <= 6 ? names[v] : "?", g_hr_v);
    return buf;
}
// When the phone counts as hot (heat_trig.h heat_check, one check every 5 s from graphics_auto / heat_guard): since 2026-11-24 three checks in a row of thermal
//   status moderate or worse, or of the headroom now at 0.95 with the status raised, or of Adreno's GPU clock held under 60 %: one reading of the 10 s forecast
//   (A#98 at status "none", A#90 at 0) had cut the scale, and on Legends: Arceus that cut came before the crashes in #90. DM_HEAT_SUSTAIN=0: one reading as
//   before. Every change is logged, and a reading the old rule would have acted on, once a minute at most
#include "heat_trig.h"
static int heat_sustain(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_HEAT_SUSTAIN"); v = !(e && !strcmp(e, "0"));
    fprintf(stderr, "heat: %s\n", v ? "hot after 3 checks in a row (15 s) of thermal status moderate or worse, headroom now 0.95+ with the status raised, or the GPU's clock held (DM_HEAT_SUSTAIN=0: one reading)"
                                     : "hot at one reading of thermal status moderate or worse, or the 10 s forecast at 0.9 (DM_HEAT_SUSTAIN=0)"); } return v; }
static int g_heat_st = -1; static float g_heat_hr = -1.0f;   // the last check's readings, for the guards' own log lines
static int heat_now(int saver, float cs, int *calm) {
    static heat_trig h; static int was; static uint64_t said;
    int sus = heat_sustain(), st = g_th_st(g_th_mgr); float hr = th_headroom(sus ? 0 : 10);
    int why = heat_check(&h, st, hr, cs, saver, sus, calm);
    g_heat_st = st; g_heat_hr = hr == hr ? hr : -1.0f;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t t = (uint64_t)ts.tv_sec;
    if (!!why != !!was)
        fprintf(stderr, "heat: %s (thermal status %d, headroom%s %.2f, GPU limit %.0f %% of its top clock%s)\n", why ? "hot" : "no longer hot", st, sus ? " now" : "", (double)g_heat_hr,
                cs >= 0 ? cs * 100.0 : -1.0, !why ? "" : why == HEAT_STATUS ? "; by the status" : why == HEAT_HEADROOM ? "; by the headroom" : why == HEAT_GPU_CAP ? "; by the GPU's clock limit" : "; by more than one");
    else if (!why && sus && heat_first(&h) && t - said >= 60) { said = t;
        fprintf(stderr, "heat: one hot reading (thermal status %d, headroom now %.2f, GPU limit %.0f %%): nothing done unless %d in a row\n", st, (double)g_heat_hr, cs >= 0 ? cs * 100.0 : -1.0, HEAT_SUSTAIN_N); }
    was = why; return why;
}
// Legends: Arceus: a 3D render scale change while playing is not safe (every crash in #90 and a vivo report came minutes after the heat guard's mid-play drop to
//   67 %, at the same place in the game). Heat there changes the scale only at the next start: <files>/heat_scale.txt holds the percent heat_boot applies under
//   the player's own (Graphics -> Render scale), and a session of 20 minutes with no hot check takes it back for the start after. Every decision is logged
static int heat_file_pct(void) { char p[1100], cur[8] = ""; snprintf(p, sizeof p, "%s/heat_scale.txt", g_files); FILE *f = fopen(p, "r"); if (f) { if (!fgets(cur, sizeof cur, f)) cur[0] = 0; fclose(f); }
    int v = atoi(cur); return v >= 50 && v < 100 ? v : 0; }
static void heat_next_start(int pct, const char *why) {
    if (heat_file_pct() == pct) return;
    char p[1100]; snprintf(p, sizeof p, "%s/heat_scale.txt", g_files); FILE *f = fopen(p, "w"); if (!f) return;
    if (pct) fprintf(f, "%d\n", pct); else fputs("\n", f); fclose(f);
    if (pct) fprintf(stderr, "heat guard: %s: 3D render scale %d %% from the next start (nothing changed while playing: a scale change in play is not safe in this game)\n", why, pct);
    else fprintf(stderr, "heat guard: %s: the 3D render scale of the player's own again from the next start\n", why);
}
static void heat_session(int hot) {   // each check: a whole cool session gives the remembered scale back
    static int checks, hot_seen; checks++; hot_seen |= hot;
    if (checks == 240 && !hot_seen && heat_file_pct()) heat_next_start(0, "20 minutes without the phone getting hot");
}
#if DM_PORT_ARCEUS
static void heat_boot(void) {   // at start, after the phone's profile: the scale heat chose last time, under the player's own
    int pct = heat_file_pct(); if (!pct) return;
    { const char *g = getenv("DM_HEAT_GUARD"); if (g && !strcmp(g, "0")) { fprintf(stderr, "boot: the heat guard remembered 3D render scale %d %%: not applied (DM_HEAT_GUARD=0)\n", pct); return; } }
    const char *e = getenv("DM_RES_CAP"); float cap = e ? (float)atof(e) : 1.0f; if (!(cap >= 0.5f && cap <= 1.0f)) cap = 1.0f;
    if (cap * 100.0f <= (float)pct + 0.5f) { fprintf(stderr, "boot: the heat guard remembered 3D render scale %d %%; the player's own (%d %%) is lower already\n", pct, (int)(cap * 100 + 0.5f)); return; }
    char v[16]; snprintf(v, sizeof v, "%.2f", pct / 100.0); setenv("DM_RES_CAP", v, 1);
    fprintf(stderr, "boot: 3D render scale %d %%, chosen by the heat guard (the phone ran hot last time; 20 cool minutes take it back; DM_HEAT_GUARD=0 leaves it out)\n", pct);
}
#endif
// Graphics "auto": TV mode while the phone keeps cool; handheld mode (720p, lighter effects) once it is hot (thermal status severe or worse, or throttling forecast
// within 10 s), back to TV mode after a minute below "moderate" with headroom to spare. The game switches as when a Switch leaves or enters its dock
void dm_set_opmode(int m);   // host/hle_oe.c: tells the game the console mode changed
// Frame-rate guard (every frame-rate target, 30 included: the project's floor is a locked 30 on every phone; with Graphics on Auto or Adaptive; DM_FPS_GUARD=0 turns
//   it off): the frames presented each 5 s against the target. Two
//   checks in a row short of it (under 93 %) drop to handheld mode (720p, the game's lighter settings), which holds the rate where TV mode cannot; TV mode is tried
//   again after a calm, steady spell, a minute the first time and twice as long after each fall back (up to 16 minutes), so a scene that cannot hold it does not
//   flip between the two. Game speed stays right meanwhile (hle_nvn.c vstep_tick). Graphics on High, Low or Lowest chosen by hand is left alone
// Games whose render size is fixed at start (Pokémon Sword / Shield: a mode change while playing does not change it, hle_resscale.c) step the game's own 3D render
//   scale instead, one step at a time: 100, 85, 67, 60 % of what the game asks, in width and height (72, 44, 36 % of the pixels); Auto starts at 85 % and settles from there. Down after the same two checks short
//   of the target, or while hot at most every 30 s (heat follows slowly); up one step after the same calm, steady spell, with the same doubling wait. Every step down is
//   checked 10 s later, as a lower scale does not always help: on a strong GPU its governor lowers the clock with the load and frames can get longer (Galaxy S25+: 43
//   against 36 ms walking at 60 %, hle_resscale.c). A step for a short frame rate that gained under 1 fps (the slowdown is not the GPU's: the CPU, a clock cap while
//   charging or warm, the replay's own work), or a step for heat that cost over 1 fps, is taken back, and that reason steps nothing for 5 minutes
// A scene that keeps the game's full size (the menus showing Pokémon models: the game's own mode, hle_resscale.c) is drawn at full size whatever the step: no step is
//   taken or judged there. A step on trial when one opens is kept without a verdict, and the short-frame count starts again
// The frame-rate / heat guard and Graphics Auto differ per port (engine unification, docs/ENGINE_UNIFY.md): Legends: Arceus steers its own ladder
//   (handheld / TV mode and loader.c arc_scale, lock30_draw); Sword / Shield and Scarlet / Violet the 3D render scale of host/hle_resscale.c
#if DM_PORT_ARCEUS
int dm_res_scale_ready(void); void dm_res_factor_set(float f);   // host/hle_resscale.c
static const float k_res_steps[] = { 1.0f, 0.85f, 0.7f, 0.6f };
#define NRES ((int)(sizeof k_res_steps / sizeof k_res_steps[0]))
#define RES_HOT 2   // the step heat goes straight to: 70 %, about 720p from 1080p (the upscaler brings it back to the screen's size)
static int g_fps_guard;   // set at start (fps_guard_init): the target, 0 off
static void fps_guard_init(void) {
    const char *t = getenv("DM_FPS_TARGET"), *off = getenv("DM_FPS_GUARD"), *g = getenv("DM_GRAPHICS");
    if (!t || !*t) t = getenv("DM_AUTO_FPS");   // (Frame rate on Auto: what the app chose for this phone)
    int fps = t ? atoi(t) : 0; if (fps <= 0) fps = 30;   // (no target set: the game's own 30)
    if (getenv("DM_FPS_DYN")) fps = 30;   // (Dynamic: the guard defends 30; the rest is the hold's)
    if (fps < 30 || (off && !strcmp(off, "0")) || (g && (!strcmp(g, "high") || !strcmp(g, "low") || !strcmp(g, "lowest")))) return;   // (read before the phone's profile fills DM_GRAPHICS in)
    g_fps_guard = fps; fprintf(stderr, "fps guard: %d fps target: a lighter picture whenever it falls short of 30 or the phone runs hot\n", fps);
}
// The heat guard, for every Graphics level the guard above does not run on (High, Low, Lowest, a scale set by hand): only heat, never the frame rate. Hot (thermal
//   status moderate or worse for 15 s, heat_now): the 720p step under the slider, from the NEXT start (heat_next_start: never while playing in this game); the player's own
//   again after a session of 20 minutes without heat. A Galaxy S25+ on High at 1080p ran its GPU down to 160 MHz (51 °C), the game at 20 fps, 2026-11-20. DM_HEAT_GUARD=0 leaves it out
// Adreno's own GPU limit (kgsl): the thermal engine caps the clock directly, and Android's thermal status can stay "none" or "light" meanwhile (a Galaxy S25+ held at
//   160 MHz of 1200, power level 14 of 15, status not raised, 2026-11-20). Returns the cap as a share of the top clock (1: no cap), -1 where it cannot be read
static float gpu_cap_share(void) {
    static int ok = 1; if (!ok) return -1;
    long cap = 0, top = 0; FILE *f = fopen("/sys/class/kgsl/kgsl-3d0/max_gpuclk", "r");
    if (f) { if (fscanf(f, "%ld", &cap) != 1) cap = 0; fclose(f); }
    f = fopen("/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies", "r");
    if (f) { if (fscanf(f, "%ld", &top) != 1) top = 0; fclose(f); }   // (the first, the highest)
    if (cap <= 0 || top <= 0) { ok = 0; fprintf(stderr, "heat: the GPU's clock limit cannot be read here (not Adreno, or not allowed): Android's thermal status alone\n"); return -1; }
    return (float)cap / (float)top;
}
// One governor with frame generation (nvn_vk.c fg_gov_absorb): when its pictures cost the game frames, this guard's ladder gives way first. dm_guard_fg_absorb: 1 when
//   the guard has a step left that can help (running, not at its lightest step, not after a step that did not help, not CPU-bound): it takes the next step at its next
//   check, judged as its own. dm_guard_heat: 0 cool, 1 hot, 2 hot with the ladder already at its heat step (frame generation then gets lighter too).
//   nvn_vk_fg_gpu_ms: frame generation's GPU time a game frame, counted into the budget the steps are judged by
static _Atomic int g_fg_pressure, g_guard_room, g_guard_heat;
int dm_guard_fg_absorb(void) { if (!atomic_load(&g_guard_room)) return 0; atomic_store(&g_fg_pressure, 1); atomic_store(&g_guard_room, 0); return 1; }
int dm_guard_heat(void) { return atomic_load(&g_guard_heat); }
extern double nvn_vk_fg_gpu_ms(void) __attribute__((weak));
static void heat_guard(void) {   // (Legends: Arceus: the 3D render scale through loader.c arc_scale, under the slider's own value)
    // Off unless DM_HEAT_GUARD=1 (2026-11-24): every crash in two players' reports (Arceus #90, an Infinix on PanVK, and a vivo on Adreno 830) came minutes after
    //   this guard dropped the scale to 67 % while playing, at the same place in the game (thread "s+350180", object main+0x417edf8: main+0x2b94314 / a call to 0
    //   from main+0x2a763c8), and one of them saw striped 3D after it. The game's own scale change in the middle of play is not safe from here
    static int off = -1, on, calm_n; if (off < 0) { const char *e = getenv("DM_HEAT_GUARD"); off = !(e && !strcmp(e, "1")); }
    extern void dm_arc_scale_set(float) __attribute__((weak));
    if (off || !g_th_mgr || !g_th_st || !dm_arc_scale_set) return;
    // (the trigger: heat_now, three checks in a row. The GPU held below 60 % of its top clock counts as hot whatever the status says)
    float cs = gpu_cap_share(); int calm, hot = heat_now(0, cs, &calm) != 0;
    calm_n = calm ? calm_n + 1 : 0;
    heat_session(hot);
    atomic_store(&g_guard_heat, hot ? 2 : 0);   // (no frame-rate steps here: frame generation, where on, gets lighter at once: a live change that is safe)
    // the scale: never while playing (heat_next_start), the 720p step under the slider's own value from the next start
    if (hot && !on) { on = 1; calm_n = 0;
        fprintf(stderr, "heat guard: the phone is hot (thermal status %d, headroom %.2f, GPU limit %.0f %% of its top clock)\n", g_heat_st, (double)g_heat_hr, cs >= 0 ? cs * 100.0 : -1.0);
        if (heat_file_pct()) fprintf(stderr, "heat guard: 3D render scale %d %% is chosen already (in use since this start, or from the next): nothing more changed\n", heat_file_pct());
        else heat_next_start(67, "the phone is hot"); }
    else if (on && calm_n >= 12) { on = 0; fprintf(stderr, "heat guard: cooled down (a calm minute); the scale chosen for the next start stays until a session of 20 minutes without heat\n"); }
}
// Locked 30, the last lever (Legends: Arceus): when even the lightest step of graphics_auto falls short with the CPU the limit, the draw distance is the one setting
//   left that lowers the CPU's work (fewer people and Pokémon loaded, trees and terrain detail closer). It is only read at start, so it is remembered in
//   <files>/auto_draw.txt and applied from the next start, when the player has Draw distance on Default (a distance chosen by hand is never changed);
//   ten calm minutes with the game thread under half the budget take it back for the start after. on: 1 near, 0 back to the game's own
extern double dm_gpu_avg_ms(int frames) __attribute__((weak));
extern _Atomic uint64_t g_nvn_game_work_avg_ns __attribute__((weak));
static void lock30_draw(int on, double cpu, double budget) {
    char p[1100]; snprintf(p, sizeof p, "%s/auto_draw.txt", g_files);
    FILE *f = fopen(p, "r"); char cur[16] = ""; if (f) { if (!fgets(cur, sizeof cur, f)) cur[0] = 0; fclose(f); }
    int was = !strncmp(cur, "near", 4) || atoi(cur) >= 50; if (was == on) return;
    { const char *d = getenv("DM_DRAW_DIST"), *o = getenv("DM_ARC_DRAW"); if ((d && atoi(d) != 100) || (!d && o && *o && strcmp(o, "auto"))) return; }   // (chosen by hand)
    f = fopen(p, "w"); if (!f) return; fputs(on ? "75\n" : "\n", f); fclose(f);
    fprintf(stderr, on ? "fps guard: the CPU cannot hold 30 here even at the lightest settings (game thread %.1f ms of %.1f): draw distance Near from the next start (Settings -> Draw distance)\n"
                       : "fps guard: steady with the CPU to spare (game thread %.1f ms of %.1f): the game's own draw distance again from the next start\n", cpu, budget);
}
static void graphics_auto(void) {
    static int mode = 1, cool, slow, held, wait = 12, primed, lv, since; static uint64_t last_fr;
    const char *g = getenv("DM_GRAPHICS");
    if (!(g && !strcmp(g, "auto") && g_th_mgr && g_th_st) && !g_fps_guard) { heat_guard(); return; }
    // No steps while playing unless DM_ARC_LIVE_STEPS=1 (2026-11-24): the ladder's switch to handheld mode in the middle of play ("graphics auto: the phone is hot",
    //   a heat forecast with the status at none) was followed 15 s later by the same crash as heat_guard's scale drop (thread "s+350180", main+0x2b94314, x20 at a
    //   "VFXB" effect), and the game's smoke fog stayed sized for TV mode, drawn into the top-left two thirds (REDMAGIC 10 Pro, Adreno 830, v0.96). The mode and
    //   scale chosen at start stay; the menu's own slider and Mode still change them on the player's word
    { static int live = -1; if (live < 0) { const char *e = getenv("DM_ARC_LIVE_STEPS"); live = e && !strcmp(e, "1");
          if (!live) fprintf(stderr, "graphics auto: the mode and 3D render scale stay as set at start while playing (DM_ARC_LIVE_STEPS=1 steps them)\n"); }
      if (!live) return; }
    int hot = 0, calm = 1;
    // hot from "moderate" (2): the status a Galaxy S25+ sat at with all its cores capped to 1.4-2.2 GHz; "severe" (3) came too late to save the frame rate. Three
    //   checks in a row (heat_now); Adreno's own GPU limit as in heat_guard
    if (g_th_mgr && g_th_st) { hot = heat_now(0, gpu_cap_share(), &calm) != 0; heat_session(hot); }
    double fps = -1;
    // (the first minute of frames does not count: a save loading in runs slow while the game reads the world, Legends: Arceus at 22.6 fps for its first 20 s; that
    //   dropped TV mode for two minutes on a phone that then held 30 in it, Galaxy S25+ 2026-11-09)
    if (g_fps_guard && dm_nvn_frame_count) { uint64_t fr = dm_nvn_frame_count(); if (primed && fr >= 1800) fps = (fr - last_fr) / 5.0; last_fr = fr; primed = 1; }
    // decisions on an average, not one 5-s reading: on a hot phone the readings swung 25.6 / 15.4 / 21.6, and the 15.4 made the next reading look like a gain, so the
    //   resolution went 85 → 70 → 60 % in ten seconds without a frame gained (a Galaxy S25+ held back by its CPU)
    static double fps_avg; if (fps > 1) fps_avg = fps_avg > 1 ? 0.5 * fps_avg + 0.5 * fps : fps;
    // the floor is 30: frames are paced at 30 or 60 (hle_nvn.c vstep_tick / dyn_tick), so 720p comes in only when even 30 is missed
    int short_ = fps > 1 && fps_avg < 30 * 0.93, full = fps >= 30 * 0.97;   // (no frames at all: a load or a pause, not a slow scene)
    slow = short_ ? slow + 1 : 0;
    cool = calm && (!g_fps_guard || full) ? cool + 1 : 0;
    // Locked 30 for every phone (Legends: Arceus): one ladder of lighter settings, the lightest last, each the game's own: TV mode at 100 % / 85 % of the render scale,
    //   then handheld mode (the game's 720p and its lighter settings, which also ease the CPU) at 100 / 85 / 70 %. The scale multiplies the player's own (Graphics →
    //   Render scale). Down a step after two checks short of 30 (one when the GPU is over 85 % of the frame's budget), or while hot; a scale step is skipped when the CPU
    //   is the limit (GPU under 70 % of the budget, the game thread over 85 %: a smaller picture would only look worse), straight to handheld mode instead. Every step is
    //   judged 15 s later and taken back if it gained under 1.5 fps (6 %), that reason then steps nothing for 5 minutes. Up a step after a calm, steady spell (a minute,
    //   twice as long after each fall back, up to 16), or at once when the GPU's time at the bigger step would still fit in 75 % of the budget. Graphics → Mode chosen
    //   by hand keeps its mode (the scale steps only). Still short at the lightest step with the CPU the limit: draw distance Near from the next start (lock30_draw)
    {
        static const float LS[] = { 1.0f, 0.85f, 1.0f, 0.85f, 0.70f }; static const int LM[] = { 1, 1, 0, 0, 0 };   // scale, mode (1 TV, 0 handheld)
        enum { NL = 5 };
        static int probe, probe_hot, nogain, noheat, cpu_short, said_cpu; static double fps_was;
        extern void dm_arc_scale_set(float) __attribute__((weak));
        int dm = display_mode(), lo = NL - 1;
        if (dm == 1) lo = 1;   // (Quality: TV mode kept, only its scale steps)
        float cap = 1.0f; { const char *c = getenv("DM_RES_CAP"); if (c) { cap = (float)atof(c); if (!(cap >= 0.5f && cap <= 1.0f)) cap = 1.0f; } }
        #define L30_APPLY() do { float sc = cap * LS[lv]; if (sc < 0.5f) sc = 0.5f; if (dm_arc_scale_set) dm_arc_scale_set(sc); \
                                 int m = dm == 2 ? 0 : dm == 1 ? 1 : LM[lv]; if (m != mode) { mode = m; dm_set_opmode(m); } } while (0)
        { static int started; if (!started) { started = 1; if (dm == 2) mode = 0; } }   // (Performance: handheld already, set at start)
        since++;
        if (nogain > 0) nogain--;
        if (noheat > 0) noheat--;
        double budget = 1000.0 / (g_fps_guard > 0 ? g_fps_guard : 30), gpu = dm_gpu_avg_ms ? dm_gpu_avg_ms(60) : -1, cpu = &g_nvn_game_work_avg_ns ? atomic_load(&g_nvn_game_work_avg_ns) / 1e6 : -1;
        if (gpu > 0 && nvn_vk_fg_gpu_ms) gpu += nvn_vk_fg_gpu_ms();   // (frame generation's pictures share the GPU with the game's frame: one budget)
        int gpu_bound = gpu > 0 && gpu > 0.85 * budget, cpu_bound = gpu > 0 && gpu < 0.70 * budget && cpu > 0.85 * budget;
        double gain_need = fps_was * 0.06 > 1.5 ? fps_was * 0.06 : 1.5;
        if (probe && !--probe && fps_avg > 1 && fps_was > 1 && lv > 0 && (probe_hot ? fps_avg < fps_was - 1.0 : fps_avg < fps_was + gain_need)) {   // the step did not help
            lv--; L30_APPLY(); slow = 0; since = 0;
            if (probe_hot) noheat = 60; else nogain = 60;
            fprintf(stderr, "%s: the lighter step %s (%.1f -> %.1f fps on average): back to %s, render scale %d %%; %s for 5 minutes\n", probe_hot ? "graphics auto" : "fps guard",
                    probe_hot ? "cost frames" : "did not help", fps_was, fps_avg, mode ? "TV mode" : "handheld mode", (int)(cap * LS[lv] * 100 + 0.5f),
                    probe_hot ? "heat lowers it no further" : "a short frame rate lowers it no further");
            return;
        }
        if (full) held++;
        if (held >= 24) wait = 12;   // two minutes held at this step: the next step down waits a minute again
        cpu_short = short_ && cpu_bound ? cpu_short + 1 : 0;
        if (cpu_short == 2 && !(said_cpu++ % 6)) fprintf(stderr, "fps guard: %.1f fps, short of %d, and the CPU is the limit (game thread %.1f ms, GPU %.1f ms of %.1f)\n", fps, g_fps_guard, cpu, gpu, budget);
        if (cpu_short >= 6 && (lv >= lo || nogain)) lock30_draw(1, cpu, budget);   // (the lightest step and still short, with the CPU the limit: fewer things to draw from the next start)
        if (full && cpu > 0 && cpu < 0.5 * budget && cool >= 120) lock30_draw(0, cpu, budget);   // (ten calm minutes with the CPU half idle: the normal distance may come back)
        int fgp = atomic_exchange(&g_fg_pressure, 0);   // (frame generation's pictures cost the game frames: a step now, judged like any other)
        // Heat takes no step while playing (2026-11-24): its handheld mode / scale drop is a live scale change, which came before the crashes in #90. Hot: the
        //   720p step from the next start (heat_next_start; DM_HEAT_GUARD=0 leaves it out), and frame generation, where on, gets lighter now (dm_guard_heat 2)
        { static int said; const char *hg = getenv("DM_HEAT_GUARD");
          if (hot && !(hg && !strcmp(hg, "0"))) { if (!heat_file_pct()) heat_next_start(67, "graphics auto: the phone is hot");
              else if (!said++) fprintf(stderr, "graphics auto: the phone is hot; 3D render scale %d %% is chosen already (in use since this start, or from the next)\n", heat_file_pct()); } }
        int for_heat = 0,
            for_rate = (slow >= (gpu_bound ? 1 : 2) || fgp) && !nogain && !probe;
        atomic_store(&g_guard_heat, hot ? 2 : 0);
        atomic_store(&g_guard_room, g_fps_guard && lv < lo && !nogain && !probe && !cpu_bound && !fgp);
        if (fgp && for_rate && lv < lo) fprintf(stderr, "fps guard: frame generation costs the game frames: a lighter step first\n");
        double r_up = lv > 0 && LM[lv - 1] == LM[lv] ? LS[lv - 1] / LS[lv] : 9;   // (up to TV mode: no early guess, the mode's cost is not the pixels')
        int up_early = lv > 0 && !probe && gpu > 0 && cool >= 2 && full && !hot && gpu * r_up * r_up < 0.75 * budget;
        if ((for_heat || for_rate) && lv < lo) {
            if (!for_heat && held < 24 && wait < 192) wait *= 2;   // short again soon after a step up: wait twice as long next time
            probe = 3; probe_hot = for_heat; fps_was = fps_avg > 1 ? fps_avg : fps;
            int to = lv + 1;
            if ((cpu_bound || for_heat) && to < 2 && lo >= 2) to = 2;   // (the CPU the limit, or hot: handheld mode, the game's lighter settings, not a smaller picture first)
            lv = to; L30_APPLY(); cool = 0; held = 0; slow = 0; since = 0;
            fprintf(stderr, "%s: %.1f fps (GPU %.1f ms, game thread %.1f ms of %.1f)%s: %s, render scale %d %%; a step up after %d s steady\n", for_heat ? "graphics auto: the phone is hot" : "fps guard",
                    fps > 0 ? fps : fps_avg, gpu, cpu, budget, cpu_bound ? ", the CPU the limit" : "", mode ? "TV mode" : "handheld mode", (int)(cap * LS[lv] * 100 + 0.5f), wait * 5);
        } else if ((lv > 0 && cool >= wait) || up_early) {
            lv--; L30_APPLY(); cool = 0; held = 0; slow = 0; since = 0; probe = 0;
            fprintf(stderr, "graphics auto: %s: %s, render scale %d %%\n", up_early ? "the GPU has room" : g_fps_guard ? "steady and cool" : "cooled down", mode ? "TV mode" : "handheld mode", (int)(cap * LS[lv] * 100 + 0.5f));
        }
        #undef L30_APPLY
    }
}
#else
int dm_res_scale_ready(void), dm_res_scale_applies(void); void dm_res_factor_set(float f);   // host/hle_resscale.c
static const float k_res_all[] = { 1.0f, 0.85f, 0.667f, 0.6f, 0.5f, 0.42f, 0.35f };   // (0.667: 720p from 1080p; 0.5 on 6 GB phones, 0.42 / 0.35 on the budget tier)
// The guard's ladder (built at start and again when the budget tier starts over): the factors it steps through and the scale each draws (k_res_eff: what the log names).
//   The lowest by the phone's tier: 35 % on a budget phone (the app picked Graphics Low / Lowest: an Adreno 610 / 613, a Mali-G52), 50 % on phones with 6 GB or less,
//   else 60 %; DM_RES_FLOOR=0.35..1 sets it. On the budget tier the level's cap (55 / 50 %) is the top and the ladder goes on under it (the factor alone below the cap
//   changes nothing: those steps are left out). RES_HOT: the step heat goes straight to, the first at about 720p or below
static float k_res_steps[8], k_res_eff[8]; static int NRES = 1, RES_HOT = 1, g_res_lo, g_res_gen = -1;
static void res_ladder(void) {
    const char *e = getenv("DM_RES_FLOOR"); float f = e ? (float)atof(e) : 0, cap = g_lvl_cap;
    if (!(f >= 0.35f && f <= 1.0f)) { long mt = kb_of("/proc/meminfo", "MemTotal:"); f = cap > 0 ? 0.35f : mt > 0 && mt <= 6500000 ? 0.5f : 0.6f; }
    NRES = 0; k_res_steps[NRES] = 1.0f; k_res_eff[NRES++] = cap > 0 ? cap : 1.0f;
    for (int i = 1; i < (int)(sizeof k_res_all / sizeof *k_res_all); i++) { float k = k_res_all[i];
        if (k < f - 0.001f || (cap > 0 && k >= cap - 0.005f)) continue;
        k_res_steps[NRES] = k; k_res_eff[NRES++] = k; }
    RES_HOT = NRES - 1; for (int i = 1; i < NRES; i++) if (k_res_eff[i] <= 0.668f) { RES_HOT = i; break; }
    g_res_lo = NRES - 1;
    fprintf(stderr, "fps guard: 3D render scale %d %% down to %d %% for this phone%s\n", (int)(k_res_eff[0] * 100 + 0.5f), (int)(k_res_eff[g_res_lo] * 100 + 0.5f),
            cap > 0 ? " (budget tier: under the Graphics level's cap)" : "");
}
static int res_lowest(void) { if (g_res_gen < 0) { g_res_gen = 0; res_ladder(); } return g_res_lo; }
// Where a slow frame rate comes from (Intelligent Adaptive Resolution, #321): the GPU's time per frame (hle_nvn.c dm_gpu_avg_ms, timestamps; -1 when the driver gives
//   none) and the game thread's own work (nvn_vk.c g_nvn_game_work_avg_ns), against the frame's budget. A lower resolution helps only a GPU that is behind.
//   That work is the game thread's without its acquire sleep and sync waits, or the replay thread's if longer (perf-1124 G2', hle_nvn.c / nvn_work.h). It was the
//   whole paced frame, about 33 ms at 30 fps, so "the CPU is the limit" held whenever the GPU was under 23 ms (no scale step) and lock-30-lite's calm test
//   (under half the budget) never did; the log lines' "game thread" figure is this one. DM_HINT_WORK=wall: as before
extern double dm_gpu_avg_ms(int frames) __attribute__((weak));
extern _Atomic uint64_t g_nvn_game_work_avg_ns __attribute__((weak));
// One governor with frame generation (nvn_vk.c fg_gov_absorb): when its pictures cost the game frames, the render scale gives way first. dm_guard_fg_absorb: 1 when this
//   guard has a step left that can help (running, the scale applies, not at its lowest, not after a step that did not help, not CPU-bound): it takes the next step at
//   its next check, judged as its own. dm_guard_heat: 0 cool, 1 hot, 2 hot with the scale already at its heat step (frame generation then gets lighter too).
//   nvn_vk_fg_gpu_ms: frame generation's GPU time a game frame, counted into the budget the steps are judged by
static _Atomic int g_fg_pressure, g_guard_room, g_guard_heat;
int dm_guard_fg_absorb(void) { if (!atomic_load(&g_guard_room)) return 0; atomic_store(&g_fg_pressure, 1); atomic_store(&g_guard_room, 0); return 1; }
int dm_guard_heat(void) { return atomic_load(&g_guard_heat); }
extern double nvn_vk_fg_gpu_ms(void) __attribute__((weak));
#define RES_START 1   // where Auto starts: 85 % (between 720p and 1080p), up to the game's full size after a calm minute at a steady 30, down to 720p when short (DM_RES_START=0..3 picks another step)
static int g_fps_guard;   // set at start (fps_guard_init): the target, 0 off
static void fps_guard_init(void) {
    const char *t = getenv("DM_FPS_TARGET"), *off = getenv("DM_FPS_GUARD"), *g = getenv("DM_GRAPHICS");
    if (!t || !*t) t = getenv("DM_AUTO_FPS");   // (Frame rate on Auto: what the app chose for this phone)
    int fps = t ? atoi(t) : 0; if (fps <= 0) fps = 30;   // (no target set: the game's own 30)
    if (getenv("DM_FPS_DYN")) fps = 30;   // (Dynamic: the guard defends 30; the rest is the hold's)
    if (fps < 30 || (off && !strcmp(off, "0")) || (g && (!strcmp(g, "high") || !strcmp(g, "low") || !strcmp(g, "lowest")))) return;   // (read before the phone's profile fills DM_GRAPHICS in)
    g_fps_guard = fps; fprintf(stderr, "fps guard: %d fps target: a lighter picture whenever it falls short of 30 or the phone runs hot\n", fps);
}
// The heat guard, for every Graphics level the guard above does not run on (High, Low, Lowest, a scale set by hand): only heat, never the frame rate. Hot (thermal
//   status moderate or worse for 15 s, heat_now): the 3D render scale goes to the 720p step through the game's own setter; back to the game's size
//   after a calm minute. A Galaxy S25+ on High at 1080p ran its GPU down to 160 MHz (51 °C), the game at 20 fps, 2026-11-20. DM_HEAT_GUARD=0 leaves it out
// Adreno's own GPU limit (kgsl): the thermal engine caps the clock directly, and Android's thermal status can stay "none" or "light" meanwhile (a Galaxy S25+ held at
//   160 MHz of 1200, power level 14 of 15, status not raised, 2026-11-20). Returns the cap as a share of the top clock (1: no cap), -1 where it cannot be read
static float gpu_cap_share(void) {
    static int ok = 1; if (!ok) return -1;
    long cap = 0, top = 0; FILE *f = fopen("/sys/class/kgsl/kgsl-3d0/max_gpuclk", "r");
    if (f) { if (fscanf(f, "%ld", &cap) != 1) cap = 0; fclose(f); }
    f = fopen("/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies", "r");
    if (f) { if (fscanf(f, "%ld", &top) != 1) top = 0; fclose(f); }   // (the first, the highest)
    if (cap <= 0 || top <= 0) { ok = 0; fprintf(stderr, "heat: the GPU's clock limit cannot be read here (not Adreno, or not allowed): Android's thermal status alone\n"); return -1; }
    return (float)cap / (float)top;
}
static void heat_guard(void) {
    static int off = -1, on, calm_n; if (off < 0) { const char *e = getenv("DM_HEAT_GUARD"); off = e && !strcmp(e, "0"); }
    if (off || !g_th_mgr || !g_th_st) return;
    // (the trigger: heat_now, three checks in a row, read every check so a scene at the game's full size does not break the count; battery saver: heat counts
    //   sooner; the GPU held below 60 % of its top clock counts as hot whatever the status says)
    float cs = gpu_cap_share(); int calm, hot = heat_now(power_saver(), cs, &calm) != 0;
    if (!dm_res_scale_ready() || !dm_res_scale_applies()) return;
    calm_n = calm ? calm_n + 1 : 0;
    atomic_store(&g_guard_heat, hot ? 2 : 0);   // (no frame-rate steps here: hot means the scale is at its heat step already)
    // Sword / Shield take the scale while playing (hle_resscale.c: the game's own setter; no crash tied to it), so heat steps it at once
    if (hot && !on) { on = 1; calm_n = 0; dm_res_factor_set(0.667f);
        fprintf(stderr, "heat guard: the phone is hot (thermal status %d, headroom %.2f, GPU limit %.0f %% of its top clock): 3D render scale %d %% (about 720p) until it cools\n", g_heat_st, (double)g_heat_hr, cs >= 0 ? cs * 100.0 : -1.0, (int)(0.667f * 100 + 0.5f)); }
    else if (on && calm_n >= 12) { on = 0; dm_res_factor_set(1.0f); fprintf(stderr, "heat guard: cooled down: the game's own 3D render scale again\n"); }
}
// Locked 30, the last lever (Sword and Shield; Legends: Arceus has the same with its draw distance, lock30_draw there): when the lightest 3D render scale still falls
//   short of 30 for 30 s, lighter shadows (DM_SHADOWS=low: 512 x 512 maps) and a shorter terrain-detail distance (DM_DRAW_DIST=75) are remembered in
//   <files>/auto_lite.txt and used from the next start (both are written into the game's code at load: loader.c galar_shadow_patch / galar_draw_patch), each only
//   where the player left that setting on its default. Ten calm minutes at the game's full size, CPU and GPU under half the budget, take it back for the start after
static int lite_was(void) { char p[1100], cur[8] = ""; snprintf(p, sizeof p, "%s/auto_lite.txt", g_files); FILE *f = fopen(p, "r"); if (f) { if (!fgets(cur, sizeof cur, f)) cur[0] = 0; fclose(f); } return cur[0] == '1'; }
static void lock30_lite(int on, double fps, double gpu, double cpu, double budget) {
    if (lite_was() == on) return;
    char p[1100]; snprintf(p, sizeof p, "%s/auto_lite.txt", g_files); FILE *f = fopen(p, "w"); if (!f) return; fputs(on ? "1\n" : "\n", f); fclose(f);
    fprintf(stderr, on ? "fps guard: %.1f fps even at the lightest 3D render scale (GPU %.1f ms, game thread %.1f ms of %.1f): lighter shadows and draw distance 75 %% from the next start (Settings -> Graphics overrides them)\n"
                       : "fps guard: steady at the full picture with room to spare (%.1f fps; GPU %.1f ms, game thread %.1f ms of %.1f): the game's own shadows and draw distance again from the next start\n", fps, gpu, cpu, budget);
}
static void lite_boot(void) {   // at start, after fps_guard_init: what lock30_lite remembered, under the player's own choices
    if (g_fps_guard != 30 || !lite_was()) return;
    const char *sh = getenv("DM_SHADOWS"), *dd = getenv("DM_DRAW_DIST"); int a = 0, b = 0;
    if (!sh || !*sh) { setenv("DM_SHADOWS", "low", 1); a = 1; }
    if (!dd || !*dd || atoi(dd) == 100) { setenv("DM_DRAW_DIST", "75", 1); b = 1; }
    if (a || b) fprintf(stderr, "boot: %s%s%s, chosen by the fps guard (this phone could not hold 30 at the lightest 3D render scale; Settings -> Graphics overrides it)\n",
                        a ? "shadows low" : "", a && b ? " and " : "", b ? "draw distance 75 %" : "");
}
static void graphics_auto(void) {
    static int mode = 1, cool, slow, held, wait = 12, primed, lv, since; static uint64_t last_fr;
    const char *g = getenv("DM_GRAPHICS");
    if (!(g && !strcmp(g, "auto") && g_th_mgr && g_th_st) && !g_fps_guard) { heat_guard(); return; }
    int hot = 0, calm = 1;
    // hot from "moderate" (2): the status a Galaxy S25+ sat at with all its cores capped to 1.4-2.2 GHz; "severe" (3) came too late to save the frame rate. Three
    //   checks in a row (heat_now); battery saver: heat counts sooner; Adreno's own GPU limit as in heat_guard
    if (g_th_mgr && g_th_st) hot = heat_now(power_saver(), gpu_cap_share(), &calm) != 0;
    double fps = -1;
    // (the first minute of frames does not count: a save loading in runs slow while the game reads the world, Legends: Arceus at 22.6 fps for its first 20 s; that
    //   dropped TV mode for two minutes on a phone that then held 30 in it, Galaxy S25+ 2026-11-09)
    if (g_fps_guard && dm_nvn_frame_count) { uint64_t fr = dm_nvn_frame_count(); if (primed && fr >= 1800) fps = (fr - last_fr) / 5.0; last_fr = fr; primed = 1; }
    // decisions on an average, not one 5-s reading: on a hot phone the readings swung 25.6 / 15.4 / 21.6, and the 15.4 made the next reading look like a gain, so the
    //   resolution went 85 → 70 → 60 % in ten seconds without a frame gained (a Galaxy S25+ held back by its CPU)
    static double fps_avg; if (fps > 1) fps_avg = fps_avg > 1 ? 0.5 * fps_avg + 0.5 * fps : fps;
    // the floor is 30: frames are paced at 30 or 60 (hle_nvn.c vstep_tick / dyn_tick), so 720p comes in only when even 30 is missed
    int short_ = fps > 1 && fps_avg < 30 * 0.93, full = fps >= 30 * 0.97;   // (no frames at all: a load or a pause, not a slow scene)
    slow = short_ ? slow + 1 : 0;
    cool = calm && (!g_fps_guard || full) ? cool + 1 : 0;
    if (dm_res_scale_ready()) {   // the render scale (a cap set by hand, DM_RES_CAP or Lowest, is left alone)
        static int probe, probe_hot, nogain, noheat; static double fps_was;
        if (getenv("DM_RES_CAP") && !(g_lvl_cap > 0)) { atomic_store(&g_guard_room, 0); return; }   // (a cap of the player's own: left alone)
        { int gen = atomic_load(&g_lvl_gen); res_lowest(); if (gen != g_res_gen) { g_res_gen = gen; res_ladder();   // (the menu changed the cap: the ladder from its top)
              lv = 0; dm_res_factor_set(1.0f); probe = 0; nogain = noheat = 0; cool = held = slow = since = 0; if (gen) return; } }
        since++;
        if (nogain > 0) nogain--;
        if (noheat > 0) noheat--;
        if (!dm_res_scale_applies()) { probe = 0; slow = 0; atomic_store(&g_guard_room, 0); return; }
        if (power_saver() && lv < RES_HOT) { lv = RES_HOT; dm_res_factor_set(k_res_steps[lv]); probe = 0; cool = 0; held = 0; slow = 0; since = 0;   // (battery saver: no larger than the 720p step)
            fprintf(stderr, "power: battery saver: 3D render scale %d %%\n", (int)(k_res_eff[lv] * 100 + 0.5f)); return; }   // (a scene at the game's full size: see above)
        // Start at 85 % and settle: a phone that holds 30 there for a calm minute goes up to the game's full size, one short of it goes down to 720p, and back when a step falls
        //   short, so each phone settles at the most it holds at a steady 30. Starting at full size and stepping down only after two short checks let slower phones
        //   (Xclipse 530, Mali, a hot Dimensity) spend minutes under 30, which the game's fixed step turns into slow motion (#289 #299 #300)
        { static int started; if (!started) { started = 1; const char *e = getenv("DM_RES_START"); int st = e && *e ? atoi(e) : g_lvl_cap > 0 ? 0 : RES_START; if (st < 0) st = 0; if (st > res_lowest()) st = res_lowest();
              if (st > 0) { lv = st; dm_res_factor_set(k_res_steps[lv]); cool = 0; held = 0; slow = 0; since = 0;
                            fprintf(stderr, "fps guard: starting at 3D render scale %d %% (%dp); a step up after each calm minute at a steady 30\n", (int)(k_res_eff[lv] * 100 + 0.5f), (int)(1080 * k_res_eff[lv] + 0.5f)); }
              return; } }
        // a step is tried for 15 s, then kept only if the average gained clearly (1.5 fps, or 6 %): a lower resolution only helps a GPU that is behind, not a CPU
        double gain_need = fps_was * 0.06 > 1.5 ? fps_was * 0.06 : 1.5;
        if (probe && !--probe && fps_avg > 1 && fps_was > 1 && lv > 0 && (probe_hot ? fps_avg < fps_was - 1.0 : fps_avg < fps_was + gain_need)) {   // the step did not help
            lv--; dm_res_factor_set(k_res_steps[lv]); slow = 0; since = 0;
            if (probe_hot) noheat = 60; else nogain = 60;
            fprintf(stderr, "%s: the lighter picture %s (%.1f → %.1f fps on average): back to 3D render scale %d %%; %s for 5 minutes\n", probe_hot ? "graphics auto" : "fps guard",
                    probe_hot ? "cost frames" : "did not help", fps_was, fps_avg, (int)(k_res_eff[lv] * 100 + 0.5f),
                    probe_hot ? "heat lowers it no further" : "the slowdown is not the GPU's, so a short frame rate lowers it no further");
            return;
        }
        if (full) held++;
        if (held >= 24) wait = 12;   // two minutes held at this step: the next step down waits a minute again
        // the bottleneck, measured (2 s of frames): with the GPU over 85 % of the frame's budget a short frame rate steps down at the first short check (still judged after 15 s);
        //   with the GPU under 70 % and the game thread over 85 %, the CPU is the limit and the picture stays (a lower one would only look worse). No timestamps: as before
        double budget = 1000.0 / (g_fps_guard > 0 ? g_fps_guard : 30), gpu = dm_gpu_avg_ms ? dm_gpu_avg_ms(60) : -1, cpu = &g_nvn_game_work_avg_ns ? atomic_load(&g_nvn_game_work_avg_ns) / 1e6 : -1;
        if (gpu > 0 && nvn_vk_fg_gpu_ms) gpu += nvn_vk_fg_gpu_ms();   // (frame generation's pictures share the GPU with the game's frame: one budget)
        int gpu_bound = gpu > 0 && gpu > 0.85 * budget, cpu_bound = gpu > 0 && gpu < 0.70 * budget && cpu > 0.85 * budget;
        { static int said_cpu; if (short_ && cpu_bound && !probe && !(said_cpu++ % 12))   // (said once a minute at most)
              fprintf(stderr, "fps guard: %.1f fps, short of %d, but the CPU is the limit (game thread %.1f ms, GPU %.1f ms of %.1f): the 3D render scale stays\n", fps, g_fps_guard, cpu, gpu, budget); }
        { static int lite_short, lite_calm;   // (lock30_lite: 6 short checks at the floor = 30 s; 120 calm ones at full size = 10 minutes)
          lite_short = g_fps_guard == 30 && short_ && (lv >= res_lowest() || nogain) && !probe && !hot ? lite_short + 1 : 0;
          lite_calm = g_fps_guard == 30 && lv == 0 && full && !hot && cpu > 0 && cpu < 0.5 * budget && gpu > 0 && gpu < 0.5 * budget ? lite_calm + 1 : 0;
          if (lite_short >= 6) { lite_short = 0; lock30_lite(1, fps_avg, gpu, cpu, budget); }
          else if (lite_calm >= 120) { lite_calm = 0; lock30_lite(0, fps_avg, gpu, cpu, budget); } }
        int fgp = atomic_exchange(&g_fg_pressure, 0);   // (frame generation's pictures cost the game frames: a step now, judged like any other)
        int for_heat = hot && lv < RES_HOT && since >= 6 && !noheat && !probe,   // (heat alone stops at 720p; only a frame rate short of 30 goes lower)
            for_rate = (slow >= (gpu_bound ? 1 : 3) || fgp) && !nogain && !probe && !cpu_bound;   // (one step at a time: the last one judged first)
        atomic_store(&g_guard_heat, hot ? (lv >= RES_HOT ? 2 : 1) : 0);
        atomic_store(&g_guard_room, g_fps_guard && lv < res_lowest() && !nogain && !probe && !cpu_bound && !fgp);
        if (fgp && for_rate) fprintf(stderr, "fps guard: frame generation costs the game frames: a lighter 3D picture first\n");
        // up early: the GPU's time projected to the next larger size (pixels grow with the square of the scale) fits in 75 % of the budget, after 10 s calm at the full rate
        double r_up = lv > 0 ? k_res_eff[lv - 1] / k_res_eff[lv] : 1;
        int up_early = lv > 0 && !probe && gpu > 0 && cool >= 2 && full && !hot && gpu * r_up * r_up < 0.75 * budget;
        if ((for_heat || for_rate) && lv < res_lowest()) {
            if (!for_heat && held < 24 && wait < 192) wait *= 2;   // short again soon after a step up: wait twice as long next time
            probe = 3; probe_hot = for_heat; fps_was = fps_avg > 1 ? fps_avg : fps;
            lv++; if (for_heat && lv < RES_HOT) lv = RES_HOT;   // hot: straight to 720p (67 % of 1080p), not one step every 30 s while the phone keeps heating up
            dm_res_factor_set(k_res_steps[lv]); cool = 0; held = 0; slow = 0; since = 0;
            if (for_heat) fprintf(stderr, "graphics auto: the phone is hot: 3D render scale %d %% (about %dp)\n", (int)(k_res_eff[lv] * 100 + 0.5f), (int)(1080 * k_res_eff[lv] + 0.5f));
            else fprintf(stderr, "fps guard: %.1f fps, short of %d (GPU %.1f ms, game thread %.1f ms of %.1f): 3D render scale %d %%; a step up after %d s steady\n", fps, g_fps_guard, gpu, cpu, budget, (int)(k_res_eff[lv] * 100 + 0.5f), wait * 5);
        } else if (((lv > 0 && cool >= wait) || up_early) && !(power_saver() && lv <= RES_HOT)) {
            lv--; dm_res_factor_set(k_res_steps[lv]); cool = 0; held = 0; slow = 0; since = 0; probe = 0;
            if (up_early) fprintf(stderr, "graphics auto: the GPU has room (%.1f ms of %.1f): 3D render scale %d %%\n", gpu, budget, (int)(k_res_eff[lv] * 100 + 0.5f));
            else fprintf(stderr, "graphics auto: %s: 3D render scale %d %%\n", g_fps_guard ? "steady and cool" : "cooled down", (int)(k_res_eff[lv] * 100 + 0.5f));
        }
        return;
    }
    if (display_mode()) return;   // (Graphics → Mode chosen: the mode stays as the player set it; heat_guard / the scale do the rest)
    if (mode == 1) {
        if (full) held++;
        if (held >= 24) wait = 12;   // two minutes held in TV mode: the next fall back waits a minute again
        if (hot || slow >= 2) {
            if (!hot && held < 24 && wait < 192) wait *= 2;   // fell back soon after trying TV mode again: wait twice as long next time
            mode = 0; dm_set_opmode(0); cool = 0; held = 0;
            if (hot) fprintf(stderr, "graphics auto: the phone is hot: handheld mode (720p)\n");
            else fprintf(stderr, "fps guard: %.1f fps, short of %d: handheld mode (720p); TV mode again after %d s steady\n", fps, g_fps_guard, wait * 5);
        }
    } else if (cool >= wait && !DM_PORT_PALDEA) {   // (Scarlet / Violet: handheld mode is kept once there: the crash after a handheld → TV switch, see DM_DISPLAY_MODE)
        mode = 1; dm_set_opmode(1); slow = 0; held = 0;
        fprintf(stderr, "graphics auto: %s: TV mode (1080p)\n", g_fps_guard ? "steady and cool" : "cooled down");
    }
}
#endif   // DM_PORT_ARCEUS
// Where the memory goes (4 / 6 GB phones, the user 2026-11-20), once a minute in the log: /proc/self/smaps's resident and swapped sizes by kind: the game's own memory
//   (the guest address space, hle_svc.c), GPU-shared mappings (kgsl / dma-buf: the game's memory pools mapped for the GPU, driver buffers), native libraries, the
//   rest of the app (heap, the driver's own, Python); plus the engine's device memory (dm_vk_mem_stats: images and buffers the GPU alone sees, not in smaps)
int dm_guest_as_contains(uintptr_t a) __attribute__((weak));
void dm_vk_mem_stats(double *blocks_mb, double *used_mb) __attribute__((weak));
static void mem_by(void) {
    FILE *f = fopen("/proc/self/smaps", "r"); if (!f) return;
    enum { GUEST, GPU, LIBS, APP, NB }; static const char *const NM[NB] = { "game", "GPU-shared", "libraries", "app" };
    long rss[NB] = { 0 }, swp[NB] = { 0 }; int cur = APP; char ln[512];
    enum { NN = 512 }; static struct { char name[64]; long kb; } top[NN]; int ntop = 0, slot = -1;   // (the "app" part by mapping name: the malloc heap, the driver, mapped files)
    while (fgets(ln, sizeof ln, f)) {
        unsigned long lo, hi;
        if (sscanf(ln, "%lx-%lx ", &lo, &hi) == 2 && strchr(ln, ' ')) {
            const char *path = strchr(ln, '/'); const char *anon = strstr(ln, "[anon:");
            cur = dm_guest_as_contains && dm_guest_as_contains((uintptr_t)lo) ? GUEST
                : (path && (strstr(path, "kgsl") || strstr(path, "dmabuf") || strstr(path, "/dev/"))) || (anon && strstr(anon, "dmabuf")) ? GPU
                : path && strstr(path, ".so") ? LIBS : APP;
            slot = -1;
            if (cur == APP) { char nm[64]; const char *src = anon ? anon : path ? path : "[anonymous]"; const char *sl = path && !anon ? strrchr(path, '/') : NULL;
                snprintf(nm, sizeof nm, "%s", sl ? sl + 1 : src); nm[strcspn(nm, "\n")] = 0;
                { char *d = nm; while (*d && !(*d >= '0' && *d <= '9' && d > nm && d[-1] == ':')) d++; }   // (names kept whole)
                for (int i = 0; i < ntop; i++) if (!strcmp(top[i].name, nm)) { slot = i; break; }
                if (slot < 0 && ntop < NN) { slot = ntop++; snprintf(top[slot].name, sizeof top[slot].name, "%s", nm); top[slot].kb = 0; } }
            continue; }
        long kb;
        if (sscanf(ln, "Rss: %ld kB", &kb) == 1) { rss[cur] += kb; if (cur == APP && slot >= 0) top[slot].kb += kb; }
        else if (sscanf(ln, "Swap: %ld kB", &kb) == 1) swp[cur] += kb;
    }
    fclose(f);
    char out[512]; int n = snprintf(out, sizeof out, "[mem-by]");
    for (int i = 0; i < NB; i++) n += snprintf(out + n, sizeof out - (size_t)n, " %s %ld MB%s", NM[i], rss[i] / 1024, swp[i] >= 1024 ? "" : ""),
                                  n += swp[i] >= 1024 ? snprintf(out + n, sizeof out - (size_t)n, " (+%ld swapped)", swp[i] / 1024) : 0;
    if (dm_vk_mem_stats) { double b, u; dm_vk_mem_stats(&b, &u); n += snprintf(out + n, sizeof out - (size_t)n, "; engine GPU memory %.0f MB in blocks (%.0f MB in use)", b, u); }
    fprintf(stderr, "%s\n", out);
    n = snprintf(out, sizeof out, "[mem-by] app's largest:");
    for (int k = 0; k < 6; k++) { int b = -1; for (int i = 0; i < ntop; i++) if (top[i].kb > 0 && (b < 0 || top[i].kb > top[b].kb)) b = i;
        if (b < 0 || top[b].kb < 1024) break; n += snprintf(out + n, sizeof out - (size_t)n, " %s %ld MB;", top[b].name, top[b].kb / 1024); top[b].kb = -top[b].kb; }
    for (int i = 0; i < ntop; i++) if (top[i].kb < 0) top[i].kb = 0;
    fprintf(stderr, "%s\n", out);
}
static void pin_update(void);
static void *log_mirror(void *p) {
    (void)p; char src[1200]; snprintf(src, sizeof src, "%s/log.txt", g_files);
    for (int n = 0;; n++) {
        sleep(5);
        char th[64];
        if (!n) { char t0[64]; thermal_text(t0, sizeof t0); }   // looks the thermal functions up
        graphics_auto(); pin_update();
        if (n % 12 == 6) mem_by();
        // DM_TEST_CRASH=<seconds> (a test of the crash path only: crash.c's record, Android's tombstone, the debug report offered at the next start): a native crash
        //   that many seconds after the start. A release app cannot be signalled from adb, so this is the way to make one on purpose. Unset: nothing
        { static int at = -2; if (at == -2) { const char *e = getenv("DM_TEST_CRASH"); at = e && atoi(e) > 0 ? atoi(e) : -1; }
          if (at > 0 && (n + 1) * 5 >= at) { fprintf(stderr, "[RED] DM_TEST_CRASH: crashing on purpose now (a test of the crash path)\n"); fflush(stderr); *(volatile int *)16 = 1; } }
        if (n % 6 == 0) fprintf(stderr, "[mem] app %ld MB resident (%ld MB swapped), phone %ld MB available of %ld MB; heat %s\n", kb_of("/proc/self/status", "VmRSS:") / 1024,
                                kb_of("/proc/self/status", "VmSwap:") / 1024, kb_of("/proc/meminfo", "MemAvailable:") / 1024, kb_of("/proc/meminfo", "MemTotal:") / 1024, thermal_text(th, sizeof th));
        // A freeze report: no frame shown for two of these 5 s looks (5 to 10 s) while the game is in the foreground with its window: what each guest thread waits on is
        //   written to the log (crash_dump_now, the stall dump test runs had), at most 3 times a run, and the copy in Download refreshed below. The game is not stopped.
        //   (Arceus-Native-Android#3: the picture stopped after 3 to 12 minutes on an Xclipse 920, sound going on, and nothing in the log said what was waiting)
        { extern uint64_t dm_nvn_frame_count(void) __attribute__((weak)); extern int dm_focus_now(void) __attribute__((weak)); extern void crash_dump_now(const char *why) __attribute__((weak));
          static uint64_t lastf; static int still, dumps;
          uint64_t f = dm_nvn_frame_count ? dm_nvn_frame_count() : 0; int fg = (!dm_focus_now || dm_focus_now() == 1) && atomic_load(&g_window);
          if (f && f == lastf && fg) { if (++still == 2 && dumps < 3 && crash_dump_now) { dumps++;
              fprintf(stderr, "[RED] freeze: no frame shown for 10 s in the foreground (frame %llu); what each thread waits on follows\n", (unsigned long long)f); crash_dump_now("no frame for 10 s"); } }
          else still = 0;
          lastf = f; }
        // what the resident memory is (cheap: /proc/self/status, no smaps walk that would stall the game's page faults): anonymous (the game's heaps, the driver's
        //   own allocations), file-backed (libraries, mapped game files), shared (buffers shared with the GPU driver), and the device memory this engine holds.
        //   A run Android ends for memory leaves these lines as the record of what grew (DebugBundle.java's report: logs/log.prev.txt)
        if (n % 6 == 0) { extern long nvn_vk_device_mb(void) __attribute__((weak));
            fprintf(stderr, "[mem-by] anonymous %ld MB, files %ld MB, shared %ld MB, engine GPU blocks %ld MB, peak resident %ld MB\n", kb_of("/proc/self/status", "RssAnon:") / 1024,
                    kb_of("/proc/self/status", "RssFile:") / 1024, kb_of("/proc/self/status", "RssShmem:") / 1024, nvn_vk_device_mb ? nvn_vk_device_mb() : -1, kb_of("/proc/self/status", "VmHWM:") / 1024); }
        { static char d[256]; if (!d[0]) dl_name(d, sizeof d, "log"); copy_file(src, d); }
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
#include "safe_start.h"   // safe_start(): one level safer after a start that crashed in the graphics driver (tests/t_safe_start.c)
static void seed_from_assets(void);   // (below: the pipeline list shipped with the app)
static void host_start(void) {
    if (g_started++) return;
    const char *f = g_app->activity->externalDataPath; snprintf(g_files, sizeof g_files, "%s", f ? f : "/sdcard/nxnative");
    mkdir(g_files, 0755); choose_game(g_files, sizeof g_files); choose_profile(f ? f : "/sdcard/nxnative");   // (the app's folder: the player list is the app's, the game folder below it)
    { void touch_ui_set_dir(const char *dir); touch_ui_set_dir(g_files); }   // the player's control layout (touch_ui.c's editor)
    char p[1200], q[1200];
    snprintf(p, sizeof p, "%s/log.txt", g_files); snprintf(q, sizeof q, "%s/log.prev.txt", g_files); rename(p, q);
    if (freopen(p, "w", stderr)) setvbuf(stderr, NULL, _IOLBF, 0);
    // the log's first lines: the credits (the app is GPL-3.0-or-later, its older code also MIT: its copyright notice stays with every copy; the full text is in the menu, About) and, once env.txt is
    //   read, whether this is an official build (BuildSigner's DM_BUILD_SIGNER), so a log attached to an issue says at once whether it comes from a repack
    //   The app's name and repository come from the game profile (GameActivity: DM_APP_NAME / DM_APP_REPO, from assets/app): Galar Native, Paldea Native
    const char *app_name = getenv("DM_APP_NAME"), *app_repo = getenv("DM_APP_REPO");
    if (!app_name || !*app_name) app_name = "Galar Native";
    if (!app_repo || !*app_repo) app_repo = "github.com/simipoB/Galar-Native-Android";
    { const char *v = getenv("DM_APP_VERSION");
      fprintf(stderr, "boot: %s, app version code %s. Copyright (c) 2026 jhaizhou (d2r-mobile, nxnative) and the Galar Native (formerly PKShield Android) contributors. "
                      "GPL-3.0-or-later, older parts also MIT (menu, About). Source and official downloads: %s/releases\n", app_name, v && *v ? v : "?", app_repo); }
    { char d[256]; dl_name(d, sizeof d, "log-previous"); copy_file(q, d); if (g_dl_note[0]) fprintf(stderr, "%s\n", g_dl_note); }
    snprintf(g_crash_src, sizeof g_crash_src, "%s/log.txt", g_files); dl_name(g_crash_dl, sizeof g_crash_dl, "log");
    if (g_dl_note[0]) fprintf(stderr, "%s\n", g_dl_note);
    { pthread_t lt; if (!pthread_create(&lt, NULL, log_mirror, NULL)) pthread_detach(lt); }
    { pthread_t it; if (!pthread_create(&it, NULL, inspector_main, NULL)) pthread_detach(it); }   // the memory inspector (scan.req / peek.req)
    snprintf(p, sizeof p, "%s/env.txt", g_files); mali_env_migrate(p); env_file(p, "env", NULL, 0);
    // the Mali white-screen check's state (nvn_vk.c mali_white_vk.inc), one file per phone: in the app's files folder itself, which Sword and Shield share
    //   (MaliLadder.java reads it there for the PanVK offer); env.txt may name another
    { const char *root = g_app->activity->externalDataPath; snprintf(q, sizeof q, "%s/mali_ladder.txt", root ? root : g_files); setenv("DM_MALI_LADDER_FILE", q, 0); }
    // The allocator's cache of large blocks, made big enough for the GPU driver's: on a Galaxy S25+ (Adreno 830) a third of the game thread's time went to the driver
    //   mapping and unmapping memory, calloc → mmap and free → munmap in every vkCmdBeginRendering (about 250 a frame) and in every command pool reset, because its
    //   blocks were larger than scudo's secondary cache keeps (simpleperf, 2026-11-09). Cached, they are reused; the pages go back to the system on Android's own
    //   timer (M_DECAY_TIME 1) instead of at once. DM_MALLOC_TUNE=0: the system's defaults
    { const char *mt = getenv("DM_MALLOC_TUNE");
      if (!(mt && !strcmp(mt, "0"))) {
          int a = mallopt(M_DECAY_TIME, 1), b = mallopt(M_CACHE_COUNT_MAX, 128), c = mallopt(M_CACHE_SIZE_MAX, 64 << 20);
          fprintf(stderr, "boot: allocator: large blocks cached for reuse (decay %s, cache count %s, cache size 64 MB %s)\n", a ? "on" : "refused", b ? "128" : "refused", c ? "set" : "refused");
      } }
    { const char *s = getenv("DM_BUILD_SIGNER");
      if (s && !strcmp(s, "official")) fprintf(stderr, "boot: build: official (signed with the release key)\n");
      else fprintf(stderr, "boot: build: UNOFFICIAL (%s): not a release from %s, it may have been changed\n", s && *s ? s : "signer unknown", app_repo); }
    g_safe_advanced = safe_start(g_files);   // (after env.txt: the player's settings win; before the phone profile: its defaults do not)
    // The driver check's settings for this driver (DriverCheck.java, the app's self-tests before Play): after env.txt and the safe-start ladder, so the player's own
    //   settings and the ladder's levels win over them (level 5's DM_VK_FLOAT_CONTROLS=0 over the check's 1: a driver that crashed on float controls, #89), and before
    //   the phone profile. The keys taken from it go to DM_DRIVER_CHECK_KEYS: nvn_vk.c takes those values as this driver's defaults, not as the player's (vk_quirks.h
    //   vkq_env_src: the crash markers still apply to them; their [quirk] line says "driver check"). DM_DRIVER_CHECK=0: not read
    if (!(getenv("DM_DRIVER_CHECK") && !atoi(getenv("DM_DRIVER_CHECK")))) { char keys[512] = ""; snprintf(p, sizeof p, "%s/driver-check.env", g_files);
        env_file(p, "driver check", keys, sizeof keys); if (*keys) setenv("DM_DRIVER_CHECK_KEYS", keys, 1); }
    // why Android ended the previous run (GameActivity, ApplicationExitInfo): reason 2 a signal, 3 low memory, 4 / 5 a crash, 6 not responding, 10 / 11 the player /
    //   the app, 13 other (Samsung's game manager, heat), 14 frozen
    { const char *le = getenv("DM_LAST_EXIT"); if (le && *le) fprintf(stderr, "boot: Android says the previous run ended: %s\n", le); }
    // Frame rate Dynamic (DM_FPS_TARGET=dyn): 60 wherever the phone holds it and 30 where it does not, frame by frame (the host's hold, hle_nvn.c vstep_tick), at the
    //   full picture: the fps guard defends 30 only (60 lowers the resolution to hold 60). Everything that reads the target sees 60; DM_FPS_DYN tells the guard
    // Pokémon Sword / Shield: 30 fps for every player (the owner's call, 2026-11-21): 60 and Dynamic slowed the game on phones that could not hold 60 (the game's
    //   own step chooser, main+0x5df360, reads the GPU time too short and picks the 16.67 ms step), and Dynamic stayed at 30. A saved 60 / Dynamic and Auto's 60 are
    //   ignored; DM_UNLOCK_EXPERIMENTS=1 in env.txt lifts the lock (development)
    { const char *u = getenv("DM_UNLOCK_EXPERIMENTS"), *ft = getenv("DM_FPS_TARGET");
      if (display_galar() && !(u && !strcmp(u, "1"))) {
        if (ft && *ft && strcmp(ft, "30")) fprintf(stderr, "boot: frame rate is 30 for Sword / Shield (env.txt had DM_FPS_TARGET=%s)\n", ft);
        unsetenv("DM_FPS_TARGET"); unsetenv("DM_AUTO_FPS"); unsetenv("DM_FPS_DYN"); } }
    { const char *ft = getenv("DM_FPS_TARGET"); if (ft && !strcmp(ft, "dyn")) { setenv("DM_FPS_TARGET", "60", 1); setenv("DM_FPS_DYN", "1", 1); }
      extern _Atomic int dm_fps_dyn __attribute__((weak)); if (&dm_fps_dyn) atomic_store(&dm_fps_dyn, getenv("DM_FPS_DYN") != NULL); }
    fps_guard_init();   // (before the phone profile: Graphics chosen by hand only)
#if !DM_PORT_ARCEUS   // (Sword / Shield's fps guard; Legends: Arceus has its own guard above and no such switches)
    lite_boot();   // (lighter shadows and draw distance, when the fps guard remembered this phone needs them)
#endif
    device_profile();   // the chip tier's defaults, under env.txt
#if DM_PORT_ARCEUS
    heat_boot();   // (the 3D render scale the heat guard chose last time: never changed while playing in this game)
#endif
    // Game files: inside the installed APKs (the all-in-one .apks bundle, read through apkfs.c as /nxapk/...), else copied into the files folder
    struct stat st; int inapk = !stat("/nxapk/RomFS", &st);
    snprintf(g_game, sizeof g_game, "%s", inapk ? "/nxapk" : g_files);
    #define SETG(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_game, sub); setenv(k, p, 0); } while (0)
    #define SETP(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_files, sub); setenv(k, p, 0); } while (0)
    SETP("DM_TEX_CACHE", "texcache");
    { const char *r = g_app->activity->externalDataPath; snprintf(p, sizeof p, "%s/lsfg", r ? r : g_files); setenv("DM_LSFG_DIR", p, 0); }
    // (the player's Lossless Scaling shaders, converted from their Lossless.dll by the app's import, one folder for every game: DM_FRAMEGEN=lsfg, nvn_lsfg.h)
    // shaders translated while the game last ran (nvn_vk.c live_check): the pack has them now (setup_game.py merged them first), so the folder starts empty
    SETP("DM_LIVE_SPV", "live");
    { DIR *ld = opendir(getenv("DM_LIVE_SPV")); struct dirent *e; char f[1200];
      if (ld) { while ((e = readdir(ld))) if (e->d_name[0] != '.') { snprintf(f, sizeof f, "%s/%s", getenv("DM_LIVE_SPV"), e->d_name); unlink(f); } closedir(ld); } }   // textures converted for a GPU without BC, kept for the next start (tex_cache.h)
    setenv("DM_AUDIO", "1", 0);
    // graphics (the app's settings screen): high = the console's TV mode (1080p, full effects), low = handheld mode (720p, the game's own lighter settings),
    // auto = TV mode, made lighter while the phone is hot or slow (graphics_auto below), lowest = handheld mode with the game's 3D render scale capped at 75 % (960x540 in
    // Pokémon Sword / Shield, hle_resscale.c; other games: as low). DM_OPMODE and DM_RES_CAP in env.txt still win
    // Frame generation (DM_FRAMEGEN, Lossless Scaling's too) and Faster rendering (DM_NVN_ASYNC) are locked off for now, whatever env.txt says: first a solid frame
    //   rate without them on every phone, then they come back (the owner's call, 2026-11-12). DM_UNLOCK_EXPERIMENTS=1 in env.txt lifts the lock (development)
    //   Faster rendering is unlocked again (2026-11-13): with the Adreno render passes (#205) and the title screen at the game's own 30 (#159) it held a solid 30 with no
    //   flicker on a Galaxy S25+, title screen and field. Its default (Auto) is the quirk table's Q_NVN_ASYNC (vk_quirks.h: Adreno, Arm's own driver, PowerVR and
    //   Xclipse since perf-1124 fix 3): hle_nvn.c rq_on
#if !DM_PORT_ARCEUS   // (Legends: Arceus keeps frame generation: its releases offer it, at the game's own rate)
    { const char *u = getenv("DM_UNLOCK_EXPERIMENTS"); if (!(u && !strcmp(u, "1"))) {
        const char *fg = getenv("DM_FRAMEGEN");
        if (fg && *fg && strcmp(fg, "0") && strcmp(fg, "off"))
            fprintf(stderr, "boot: frame generation is locked off for now (env.txt had DM_FRAMEGEN=%s)\n", fg);
        setenv("DM_FRAMEGEN", "0", 1); } }
#endif
    // Pokémon Sword / Shield: low and lowest stay in TV mode, the 3D lowered by the engine's cap instead (hle_resscale.c): 0.55 of 1080p ≈ 594p (handheld drew 576p),
    //   0.50 = 960x540 (as lowest drew before). In handheld mode a Dynamax (the Poké Ball growing) crashed: the game's file loader got a null buffer from one of its
    //   own sub-heaps and unpacked an LZ4 entry into it (main+0x5dd7b8 alloc, no null check → main+0x1865410; #211 Adreno 613 on lowest, #115 Xclipse 530 on low,
    //   both "handheld set, TV flag 0"; no crash reported in TV mode). DM_OPMODE=0 in env.txt still gives handheld
    { int m = display_mode();   // (Graphics → Mode: Quality / Performance, before the level's defaults below)
      if (m == 1 || (m == 2 && display_galar())) setenv("DM_OPMODE", "1", 1); else if (m == 2) setenv("DM_OPMODE", "0", 1); }
    { const char *g = getenv("DM_GRAPHICS"); int galar = !strcmp(DM_GAME_ID, "01008DB008C2C000");
      if (g && (!strcmp(g, "low") || !strcmp(g, "lowest"))) setenv("DM_OPMODE", galar ? "1" : "0", 0); else if (g && (!strcmp(g, "high") || !strcmp(g, "auto"))) setenv("DM_OPMODE", "1", 0);
      int own = getenv("DM_RES_CAP") != NULL;   // (the slider or env.txt: the player's own)
      if (g && !strcmp(g, "lowest")) setenv("DM_RES_CAP", galar ? "0.50" : "0.75", 0);
      else if (g && !strcmp(g, "low") && galar) setenv("DM_RES_CAP", "0.55", 0);
      if (!own && galar && g && (!strcmp(g, "low") || !strcmp(g, "lowest"))) g_lvl_cap = g_lvl_cap0 = (float)atof(getenv("DM_RES_CAP"));   // (the budget tier: graphics_auto)
      { const char *rc = getenv("DM_RES_CAP"); float u = rc ? (float)atof(rc) : 1.0f; if (!(u >= 0.35f && u <= 1.0f)) u = 1.0f; g_user_cap = u;   // (Performance on Sword / Shield: the 720p step at most)
        if (display_cap(u) < u) { char b[16]; snprintf(b, sizeof b, "%.3f", display_cap(u)); setenv("DM_RES_CAP", b, 1); fprintf(stderr, "mode: performance: 3D render scale %.0f %% (TV mode kept)\n", display_cap(u) * 100); } } }
#if defined(DM_GAME_ANDROID_OPMODE) && DM_GAME_ANDROID_OPMODE >= 0
    { char m[4]; snprintf(m, sizeof m, "%d", DM_GAME_ANDROID_OPMODE); setenv("DM_OPMODE", m, 0); }
#if DM_GAME_PROGRAM_ID == 0x01001F5010DFA000ull   // Legends: Arceus: 60 fps only when chosen (Auto is 30); at 60 a frame every refresh (its present interval 2
    //   left aside), the game's step following the frame time (hle_nvn.c arc_step)
    setenv("DM_AUTO_FPS", "30", 1);
    // Faster rendering (DM_NVN_ASYNC, the replay on its own thread) is off for this game whatever env.txt says: with it the game crashed at loading screens
    //   (Arceus-Native-Android#3: a Galaxy S23 FE and another phone, a game object found overwritten in the game's own code; gone with it off)
    if (getenv("DM_NVN_ASYNC")) { unsetenv("DM_NVN_ASYNC"); fprintf(stderr, "boot: Faster rendering is off for Legends: Arceus (it crashed the game at loading screens)\n"); }
    { const char *f = getenv("DM_FPS_TARGET"); if (f && atoi(f) >= 60) setenv("DM_PRESENT_INTERVAL", "display", 0); }
#endif   // the profile's android_op_mode (env.txt DM_OPMODE wins)
#endif   // real sound output (AAudio, compat/aaudio_unit.c); env.txt can set 0
    // pipelines: every one the game makes is recorded here and built ahead at the next start, the list shipped with the game too, and the driver's compiled code kept
    SETP("DM_MODS_DIR", "mods");   // IPSwitch .pchtxt code mods, applied when their build ID matches (loader.c); env.txt may point elsewhere
    snprintf(p, sizeof p, "%s/mods", g_files); mkdir(p, 0770);   // made by the app itself, so it can read what is put there (a folder another user made, adb's for one, it cannot)
    boot_guard(p);   // (before the mods are read: a crash with mods switches them off)
    mods_romfs(p);
    lowspec();   // Legends: Arceus's "Lower specs" (DM_LOWSPEC): its RomFS files first in the overlay (after the mods: mods_romfs takes a set DM_ROMFS_OVERLAY for env.txt's choice)
    style_fx();   // Legends: Arceus's "Style move effects" (off unless DM_ARC_STYLE_FX=1): the No More Styles start effects first in the overlay
    draw_distance();   // Legends: Arceus's "Draw distance" (DM_ARC_DRAW): the player's own config files with the distances changed, first in the overlay
    flat_normals();   // Legends: Arceus's "No reflections": its flat normal maps last in the overlay, under any mod
    seed_from_assets();
    SETP("DM_VK_PSO_RECORD", "vk_pipelines.bin"); SETP("DM_VK_PCACHE", "vk_pipeline_cache.bin"); SETG("DM_VK_PSO_SEED", "vk_pipelines_seed.bin");
    snprintf(q, sizeof q, "%s/shaders.pak", g_game); if (!stat(q, &st)) setenv("DM_SPV_PAK", q, 0);   // the packed shader set, else vk/b-<chain>/
    { snprintf(p, sizeof p, "%s/aoc", g_game); struct stat as; if (!stat(p, &as)) setenv("DM_AOC_DIR", p, 0); }   // the DLC the player owns (Shield: the Expansion Pass keys; its content is in the update)
    SETP("DM_NVN_CAPTURE", "nvn_capture");   // shaders the game builds in memory, saved for the setup to translate (nvn_vk.c mtl_shader_seen)
    SETG("DM_ROMFS", "RomFS"); SETG("DM_SPV_DIR", "vk"); SETG("DM_NVN_SHMAP", "shader_map.tsv"); SETG("DM_FONT_DIR", "font");
    // the chosen player's save (choose_profile: Player 1's is <files>/save as always); env.txt's DM_SAVE still wins
    snprintf(p, sizeof p, "%s/save", g_prof); setenv("DM_SAVE", p, 0); SETP("DM_TEMP", "temp");
    fprintf(stderr, "boot: player %s: save in %s\n", g_prof_id, g_prof);
    // the game's account nickname (hle_account.c DM_NICK) is the player's name, cut to its 32 bytes at a whole character; an unchanged Player 1 stays "Player", as before
    if (g_prof_name[0] && (strcmp(g_prof_id, "1") || strcmp(g_prof_name, "Player 1"))) {
        char nick[33]; size_t n = strlen(g_prof_name); if (n > 32) { n = 32; while (n && ((unsigned char)g_prof_name[n] & 0xC0) == 0x80) n--; }
        memcpy(nick, g_prof_name, n); nick[n] = 0; setenv("DM_NICK", nick, 0); }
    // A save archive imported by SetupActivity is staged as save_import/ (the chosen player's). Rename the directories on the same
    // filesystem so nested save layouts survive and the old save remains intact if the import cannot be installed.
    { char imp[1200], old[1300]; struct stat is, ss;
      snprintf(imp, sizeof imp, "%s/save_import", g_prof);
      snprintf(p, sizeof p, "%s/save", g_prof);
      if (!lstat(imp, &is) && S_ISDIR(is.st_mode)) {
          int moved_old = 0, can_install = 1;
          int save_status = stat(p, &ss);
          if (!save_status) {
              snprintf(old, sizeof old, "%s/save.before-import-%ld-%d", g_prof, (long)time(NULL), (int)getpid());
              if (!rename(p, old)) moved_old = 1;
              else { fprintf(stderr, "[RED] save import: could not preserve the current save: %s\n", strerror(errno)); can_install = 0; }
          } else if (errno != ENOENT) {
              fprintf(stderr, "[RED] save import: could not inspect the current save: %s\n", strerror(errno)); can_install = 0;
          }
          if (can_install && !rename(imp, p)) {
              fprintf(stderr, "boot: imported save archive; previous save backup: %s\n", moved_old ? old : "none"); g_save_imported = 1;
          } else {
              if (moved_old) rename(old, p);
              fprintf(stderr, "[RED] save import: could not install imported save; it remains in %s\n", imp);
          }
      } }
    snprintf(p, sizeof p, "%s/save", g_prof); mkdir(p, 0755); snprintf(p, sizeof p, "%s/temp", g_files); mkdir(p, 0755);
    save_place();   // (DM_SAVE_SHARED: the save in Documents, copied over at this start when the choice changed; after an import, which lands in the app's folder)
    snprintf(p, sizeof p, "%s/RomFS", g_game);
    if (stat(p, &st)) { fprintf(stderr, "boot: game files not found: %s (install the bundle, or copy RomFS, patched, shaders.pak, shader_map.tsv and font there)\n", p); ALOG("game files missing in %s", g_files); return; }
    chdir(g_files);
    fprintf(stderr, "boot: game files %s%s\n", g_game, inapk ? " (inside the installed APKs)" : "");
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 64u << 20);
    pthread_t t; pthread_create(&t, &at, host_thread, NULL); pthread_detach(t);
    { pthread_t g; if (!pthread_create(&g, NULL, boot_guard_main, NULL)) pthread_detach(g); }
}
// ---------- The fastest cores for the threads a frame waits on. Left alone, the scheduler kept Violet's main thread moving over the middle cores of a Galaxy S25+ (2.7 GHz) with
// its two fastest cores idle at 1 GHz (pinned, the replay's CPU cost fell from 21 to 14 ms a frame). "Fastest" is each core's clock limit right now, looked at again every 5 s
// (log_mirror): a hot phone can cap its prime cores below the others (1.69 GHz seen against 2.9), and then the threads move. A phone whose cores are all alike is left alone.
// DM_CPU_PIN=off leaves the scheduler to it
static pid_t g_pin_tid[2]; static int g_pin_n; static long g_pin_top;
static void pin_update(void) {
    const char *e = getenv("DM_CPU_PIN"); if ((e && !strcmp(e, "off")) || !g_pin_n) return;
    long top = 0, low = 0, f[32]; cpu_set_t set; CPU_ZERO(&set); int n = 0;
    for (int c = 0; c < 32; c++) { char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", c); FILE *fp = fopen(p, "r"); f[c] = 0;
        if (fp) { if (fscanf(fp, "%ld", &f[c]) != 1) f[c] = 0; fclose(fp); } if (f[c] > top) top = f[c]; if (f[c] && (!low || f[c] < low)) low = f[c]; }
    if (!top) return;
    if (top == low) { if (!g_pin_top) return; for (int c = 0; c < 32; c++) if (f[c]) { CPU_SET(c, &set); n++; } }   // all alike now: every core
    else for (int c = 0; c < 32; c++) if (f[c] == top) { CPU_SET(c, &set); n++; }
    // never fewer cores than pinned threads: a hot Galaxy Z Flip5 capped its prime core to 1.71 GHz with the middle ones at 1.56, and both threads a frame waits on
    //   shared that one core (41-48 ms of CPU work a frame) while three nearly as fast cores sat idle. The next fastest cores join until each thread has one
    long floor_ = top;
    while (n < g_pin_n && top != low) {
        long next = 0; for (int c = 0; c < 32; c++) if (f[c] && f[c] < floor_ && f[c] > next) next = f[c];
        if (!next) break;
        for (int c = 0; c < 32; c++) if (f[c] == next) { CPU_SET(c, &set); n++; }
        floor_ = next; }
    if (top == g_pin_top && !(top == low)) { static cpu_set_t last; if (CPU_EQUAL(&last, &set)) return; last = set; }
    g_pin_top = top;
    for (int i = 0; i < g_pin_n; i++) sched_setaffinity(g_pin_tid[i], sizeof set, &set);
    if (floor_ < top) fprintf(stderr, "[cpu] main threads on %d cores limited to %.2f-%.2f GHz (the fastest alone was fewer cores than threads)\n", n, floor_ / 1e6, top / 1e6);
    else fprintf(stderr, "[cpu] main threads on %d core%s limited to %.2f GHz\n", n, n == 1 ? "" : "s", top / 1e6);
}
// ---------- Performance hints (Android 13+): one session for the game's main thread and the presenting thread, told every frame how long the presenting thread
// worked against the frame's period (nvn_vk.c mtl_present). That is what brings the fastest cores and higher clocks when the game needs them, and lower ones
// when it does not (LAS Recomp's Android port: the prime core otherwise stays parked). The presenting thread also gets the most urgent priority an app may set
static void *g_hint; static int g_hint_failed; static int64_t g_hint_target;
// A/B in one run (DM_AB=<frames>:<key>): the switch named <key> alternates every <frames> presented frames, on (1) in the odd stretches, so both sides run in the
//   same heat: a phone that warms up throttles its cores (a Galaxy S25+ from 4.47 to 1.4 GHz in a long session, 2026-11-09), and two runs one after the other
//   differed by more than most changes (the A/B idea from another Android port of a Switch game). Returns -1 when <key> is not under test, else 0 or 1. The present line (nvn_vk.c)
//   names the side each window was on
int dm_ab(const char *key) {
    static int init; static long every; static char k[32];
    if (!init) { init = 1; const char *e = getenv("DM_AB"), *c = e ? strchr(e, ':') : NULL;
        if (c && (every = atol(e)) > 0) { snprintf(k, sizeof k, "%s", c + 1); fprintf(stderr, "[ab] %s alternates every %ld frames (on in the odd stretches)\n", k, every); } }
    if (every <= 0 || strcmp(key, k)) return -1;
    extern uint64_t dm_nvn_frame_count(void) __attribute__((weak));
    uint64_t f = dm_nvn_frame_count ? dm_nvn_frame_count() : 0; return (int)((f / (uint64_t)every) & 1);
}
int dm_ab_side(void) {   // the side of the frame just presented (-1: no DM_AB); the timing lines name it (hle_nvn.c)
    static int has = -1; if (has < 0) { const char *e = getenv("DM_AB"); has = e && strchr(e, ':') != NULL; }
    if (!has) return -1;
    extern uint64_t dm_nvn_frame_count(void) __attribute__((weak)); const char *e = getenv("DM_AB"); long every = e ? atol(e) : 0; if (every <= 0) return -1;
    uint64_t f = dm_nvn_frame_count ? dm_nvn_frame_count() : 0; return f ? (int)(((f - 1) / (uint64_t)every) & 1) : 0;
}
// The target the hint session is given: three quarters of the frame, not all of it (DM_HINT_TARGET=<percent>, 100: the whole frame as before). With the whole frame
//   the governor lowers the clocks until the work just fits it, and frames that weigh near 33 ms then go over every few frames, each a refresh late: 17-20 fps at
//   settings that held 30 on a Mali-G925 in another Android port of a Switch game, which moved to 25 ms of 33 for its 30 fps. DM_AB=<n>:hint compares the two
static int64_t hint_target(uint64_t period_ns) {
    static int pct = -1; if (pct < 0) { const char *e = getenv("DM_HINT_TARGET"); pct = e ? atoi(e) : 75; if (pct < 30 || pct > 100) pct = 75; }
    int ab = dm_ab("hint"); int p = ab < 0 ? (power_saver() ? 100 : pct) : ab ? 75 : 100;   // (battery saver: the whole frame, so the clocks stay as low as the frame allows)
    return (int64_t)period_ns * p / 100;
}
static void *(*p_hint_mgr)(void); static void *(*p_hint_create)(void *, const int32_t *, size_t, int64_t);
static int (*p_hint_target)(void *, int64_t); static int (*p_hint_report)(void *, int64_t);
void front_frame_work(uint64_t busy_ns, uint64_t period_ns) {
    if (g_hint_failed) return;
    uint64_t target_ns = (uint64_t)hint_target(period_ns);
    if (!g_hint) {
        static int tried; if (tried++) return;
        setpriority(PRIO_PROCESS, (id_t)gettid(), -10);
        // the presenting thread and the game's main thread go to the fastest cores whether or not a hint session comes: Android 12 and older have none, and a refused
        //   session left a budget phone with no pinning at all (the session's own list replaces this one when it comes)
        g_pin_tid[0] = gettid(); g_pin_n = 1; if (g_game_tid && g_game_tid != gettid()) g_pin_tid[g_pin_n++] = g_game_tid;
        pin_update();
        void *lib = dlopen("libandroid.so", RTLD_NOW); if (!lib) { g_hint_failed = 1; fprintf(stderr, "[cpu] no performance hints (no libandroid): main threads pinned all the same\n"); return; }
        p_hint_mgr = (void *(*)(void))dlsym(lib, "APerformanceHint_getManager");
        p_hint_create = (void *(*)(void *, const int32_t *, size_t, int64_t))dlsym(lib, "APerformanceHint_createSession");
        p_hint_target = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
        p_hint_report = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
        void *m = p_hint_mgr ? p_hint_mgr() : NULL;
        int32_t t[2 + 8]; size_t n = 0; t[n++] = gettid(); if (g_game_tid && g_game_tid != gettid()) t[n++] = g_game_tid;
        size_t pinned = n;   // (only these two are pinned to the top cores: the workers below share the frame's work and would crowd them)
        { extern int thr_frame_tids(int *out, int max) __attribute__((weak)); int w[8]; int k = thr_frame_tids ? thr_frame_tids(w, 8) : 0;
          for (int i = 0; i < k; i++) if (w[i] != t[0] && (pinned < 2 || w[i] != t[1])) t[n++] = w[i]; }
        g_hint = m && p_hint_create ? p_hint_create(m, t, n, (int64_t)target_ns) : NULL;
        if (!g_hint && n > pinned && m && p_hint_create) g_hint = p_hint_create(m, t, n = pinned, (int64_t)target_ns);   // (refused with the workers: the two threads as before)
        if (!g_hint || !p_hint_report) { g_hint_failed = 1; fprintf(stderr, "[cpu] no performance hint session: main threads pinned all the same\n"); return; }
        g_hint_target = (int64_t)target_ns; for (size_t i = 0; i < pinned; i++) g_pin_tid[i] = t[i]; g_pin_n = (int)pinned; pin_update();
        fprintf(stderr, "[cpu] performance hint session: %zu threads (%zu of them the game's frame workers), target %.1f ms of the %.1f ms frame; presenting thread at priority -10\n", n, n - pinned, target_ns / 1e6, period_ns / 1e6);
    }
    if ((int64_t)target_ns != g_hint_target && p_hint_target) p_hint_target(g_hint, g_hint_target = (int64_t)target_ns);
    if (busy_ns) p_hint_report(g_hint, (int64_t)busy_ns);
}
// Loading (GameActivity polls this every 500 ms and tells Android 13+ through GameManager.setGameState, so the system can raise the clocks and the storage
//   speed for it): a loading screen as hle_nvn.c tells it by the picture (dm_loading_poll: the game reads 8 MB/s or more while under 60 draws a frame, over at
//   the first full frame; fable-loads.md I1). DM_LOAD_DETECT=reads: the rule before 2026-11-24, the game reading 8 MB/s or more until under 1 MB/s for 2 s,
//   which open-world streaming met too (Arceus #98's "12.03 s load" was play: 430-530 draws a frame). A load also tells the hint session (Android 16:
//   APerformanceHint_notifyWorkloadSpike) that this frame's CPU work jumps, before the governor sees the late frames. DM_LOAD_HINT=0: neither
// A load's report (the white screen before a battle, 2026-11-19): every thread's CPU time over the load (/proc/self/task/<tid>/stat utime + stime), the core it was last
//   on and that core's clock limit, and the time the game's file reads took: says whether a load waits on storage or on a thread's CPU work, and on which kind of core.
//   The percentages are of the window the two snapshots span (the poll that saw the load begin to the one that saw it end); they used to be divided by the
//   load's length while the snapshots spanned 1.5 s more, so a "1.00 s load" was a 2.5 s window and every share ran high (I1). hle_nvn.c prints the rest of
//   the load's cost ([load] over after ...: frames, frame work, acquire sleep, sync waits, uploads, pipelines, touch scans)
#include <dirent.h>
typedef struct { int tid, cpu; uint64_t ticks; char name[20]; } thr_snap;
static int thr_snapshot(thr_snap *o, int max) {
    DIR *d = opendir("/proc/self/task"); if (!d) return 0; struct dirent *e; int n = 0;
    while ((e = readdir(d)) && n < max) { int tid = atoi(e->d_name); if (tid <= 0) continue;
        char p[64], b[512]; snprintf(p, sizeof p, "/proc/self/task/%d/stat", tid); FILE *f = fopen(p, "r"); if (!f) continue;
        size_t k = fread(b, 1, sizeof b - 1, f); fclose(f); b[k] = 0;
        char *lp = strchr(b, '('), *rp = strrchr(b, ')'); if (!lp || !rp) continue;
        thr_snap *t = &o[n]; t->tid = tid; size_t nl = (size_t)(rp - lp - 1); if (nl > sizeof t->name - 1) nl = sizeof t->name - 1; memcpy(t->name, lp + 1, nl); t->name[nl] = 0;
        char *q = rp + 2, *sv = NULL; unsigned long long ut = 0, st = 0; int cpu = -1, field = 3;   // (field 3 is the state, after the name)
        for (char *tok = strtok_r(q, " ", &sv); tok; tok = strtok_r(NULL, " ", &sv), field++) { if (field == 14) ut = strtoull(tok, NULL, 10); else if (field == 15) st = strtoull(tok, NULL, 10); else if (field == 39) { cpu = atoi(tok); break; } }
        t->ticks = ut + st; t->cpu = cpu; n++; }
    closedir(d); return n;
}
static int thr_one(int tid, thr_snap *t) {   // one thread: its name, CPU ticks, and in cpu its state letter (R, S, D ...)
    char p[64], b[512]; snprintf(p, sizeof p, "/proc/self/task/%d/stat", tid); FILE *f = fopen(p, "r"); if (!f) return 0;
    size_t k = fread(b, 1, sizeof b - 1, f); fclose(f); b[k] = 0;
    char *lp = strchr(b, '('), *rp = strrchr(b, ')'); if (!lp || !rp) return 0;
    t->tid = tid; size_t nl = (size_t)(rp - lp - 1); if (nl > sizeof t->name - 1) nl = sizeof t->name - 1; memcpy(t->name, lp + 1, nl); t->name[nl] = 0;
    unsigned long long ut = 0, st = 0; char *sv = NULL; int field = 3;
    for (char *tok = strtok_r(rp + 2, " ", &sv); tok; tok = strtok_r(NULL, " ", &sv), field++) { if (field == 3) t->cpu = tok[0]; else if (field == 14) ut = strtoull(tok, NULL, 10); else if (field == 15) { st = strtoull(tok, NULL, 10); break; } }
    t->ticks = ut + st; return 1;
}
static long core_max_khz(int c) { char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c); FILE *f = fopen(p, "r"); long v = 0; if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); } return v; }
static void load_report(const thr_snap *a, int na, const thr_snap *b, int nb, uint64_t win_ns, uint64_t bytes, uint64_t read_ns) {
    long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    struct { int i; uint64_t d; } top[8]; memset(top, 0, sizeof top);
    for (int i = 0; i < nb; i++) { uint64_t was = 0; for (int j = 0; j < na; j++) if (a[j].tid == b[i].tid) { was = a[j].ticks; break; }
        uint64_t d = b[i].ticks > was ? b[i].ticks - was : 0;
        for (int k = 0; k < 8; k++) if (d > top[k].d) { for (int m = 7; m > k; m--) top[m] = top[m - 1]; top[k].i = i; top[k].d = d; break; } }
    double secs = win_ns / 1e9 > 0.01 ? win_ns / 1e9 : 0.01;
    fprintf(stderr, "[cpu] load report: %.2f s between the polls that saw it begin and end, %.1f MB read in %.2f s of file reads (all threads together); busiest threads (CPU s in that window, last core and its top clock):\n",
            win_ns / 1e9, bytes / 1048576.0, read_ns / 1e9);
    for (int k = 0; k < 8 && top[k].d; k++) { const thr_snap *t = &b[top[k].i];
        fprintf(stderr, "[cpu]   %-16s tid %d: %.2f s CPU (%.0f %% of a core), core %d (%.2f GHz)\n", t->name, t->tid, (double)top[k].d / hz, 100.0 * (double)top[k].d / hz / secs, t->cpu, core_max_khz(t->cpu) / 1e6); }
}
// A load profiled (DM_LOAD_PROF=1; off by default since v0.100: Arceus Native v0.99 closed at start with it on, and only DM_LOAD_PROF=0 let it start;
//   fable-loads.md I3 / I4): while a load runs, up to LP_SLOTS threads are sampled 1000 times a second of their own
//   CPU time (a SIGPROF timer aimed at each thread): the replay thread (hle_nvn.c dm-replay) and the game's main thread (g_game_tid; getpid() was the Java UI
//   thread) from the start, then each thread load_boost's pass finds at a quarter of a core or more. A sample in a game module keeps its pc and lr only (in game
//   code x29 need not be a frame pointer, and a bad read would land in the crash handler); in host code the frame-pointer chain follows (the engine keeps frame
//   pointers). The handler is lock-free: no dladdr, no thread names (module_of only reads the module table). At the end the samples go to
//   load_prof_<run start>_<n>.txt in the data folder (the newest 12 kept; the debug report takes them), with a header that ties them to a build: the app's
//   version code, libnxnative.so's build id, every game module's build id, each slot's name, tid and CPU seconds; frames as main+0x / sdk+0x / libnxnative.so+0x
#include <signal.h>
#include <ucontext.h>
#include <link.h>
enum { LP_N = 32768, LP_SLOTS = 8 };
enum { LP_F = 10 };   // frames a sample keeps: pc, lr, then the frame-pointer chain
static struct { uint64_t f[LP_F]; int nf, who; } g_lp[LP_N]; static _Atomic int g_lp_n; static _Atomic int g_lp_on; static _Atomic int g_lp_tid[LP_SLOTS];
int dm_addr_module(const void *p, char *out, size_t n);   // loader.c
int dm_module_info(int i, const char **name, const char **build);
void *module_of(const void *p);   // loader.c (lock-free: reads the module table)
static void lp_sig(int sig, siginfo_t *si, void *ucv) {
    (void)sig; (void)si; if (!atomic_load(&g_lp_on)) return;
    // On the thread's alternate stack only (SA_ONSTACK; every guest thread has one, crash.c crash_thread_init): patch_nso.py's stubs park two registers in the
    //   16 bytes below sp without moving it (a red zone Apple has, Linux does not), and a signal frame pushed on the game's own stack overwrites them. That was
    //   Arceus Native v0.99 closing at start (18,250 TLS-read + 16,080 x18 stub sites in its main). Rule: any signal aimed at a game thread needs SA_ONSTACK.
    //   A thread without an alternate stack ends the profile here (the frame is already pushed, but no more come)
    { stack_t cur; if (sigaltstack(NULL, &cur) || !(cur.ss_flags & SS_ONSTACK)) { if (module_of((void *)(uintptr_t)((ucontext_t *)ucv)->uc_mcontext.pc)) { atomic_store(&g_lp_on, 0); return; } } }
    ucontext_t *uc = ucv; int t = (int)gettid(), who = -1; for (int k = 0; k < LP_SLOTS; k++) if (atomic_load(&g_lp_tid[k]) == t) { who = k; break; }
    if (who < 0) return;
    int i = atomic_fetch_add(&g_lp_n, 1); if (i >= LP_N) return;
    uint64_t pc = uc->uc_mcontext.pc;
    g_lp[i].who = who; g_lp[i].f[0] = pc; g_lp[i].f[1] = uc->uc_mcontext.regs[30]; int nf = 2;
    if (!module_of((void *)(uintptr_t)pc)) {   // (host code only: the game's x29 need not be a frame pointer)
        uint64_t sp = uc->uc_mcontext.sp, fp = uc->uc_mcontext.regs[29];
        while (nf < LP_F && fp && !(fp & 15) && fp >= sp && fp < sp + (8u << 20)) { uint64_t nx = ((uint64_t *)(uintptr_t)fp)[0], ra = ((uint64_t *)(uintptr_t)fp)[1]; g_lp[i].f[nf++] = ra; if (nx <= fp) break; fp = nx; } }   // (inside its own stack only)
    g_lp[i].nf = nf;
}
static timer_t g_lp_timer[LP_SLOTS]; static int g_lp_ntimer; static uint64_t g_lp_t0[LP_SLOTS];
static int lp_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_LOAD_PROF"); on = e && *e && strcmp(e, "0") != 0; } return on; }
static void lp_add(int tid) {   // one more thread sampled (load_prof's start, load_boost's pass); only while a profile runs, at most LP_SLOTS
    if (!tid || !atomic_load(&g_lp_on) || g_lp_ntimer >= LP_SLOTS) return;
    for (int k = 0; k < g_lp_ntimer; k++) if (atomic_load(&g_lp_tid[k]) == tid) return;
    struct sigevent ev; memset(&ev, 0, sizeof ev); ev.sigev_notify = SIGEV_THREAD_ID; ev.sigev_signo = SIGPROF; ev._sigev_un._tid = tid;
    clockid_t cpu = (clockid_t)((~(unsigned)tid) << 3) | 6;   // (the thread's own CPU clock, CPUCLOCK_SCHED | PERTHREAD: a sample per 1 ms it runs, none while it waits)
    int k = g_lp_ntimer; thr_snap s; g_lp_t0[k] = thr_one(tid, &s) ? s.ticks : 0;
    atomic_store(&g_lp_tid[k], tid);
    if (timer_create(cpu, &ev, &g_lp_timer[k]) && timer_create(CLOCK_MONOTONIC, &ev, &g_lp_timer[k])) { atomic_store(&g_lp_tid[k], 0); return; }
    struct itimerspec it = { { 0, 1000000 }, { 0, 1000000 } }; timer_settime(g_lp_timer[k], 0, &it, NULL); g_lp_ntimer++;
}
// libnxnative.so's GNU build id (NT_GNU_BUILD_ID, the linker's --build-id), hex; "" when absent
static int lp_note_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size; char *out = data; Dl_info di; if (!dladdr((void *)lp_note_cb, &di) || (uintptr_t)info->dlpi_addr != (uintptr_t)di.dli_fbase) return 0;
    for (int i = 0; i < info->dlpi_phnum; i++) { const ElfW(Phdr) *ph = &info->dlpi_phdr[i]; if (ph->p_type != PT_NOTE) continue;
        const uint8_t *p = (const uint8_t *)(info->dlpi_addr + ph->p_vaddr), *e = p + ph->p_memsz;
        while (p + sizeof(ElfW(Nhdr)) <= e) { const ElfW(Nhdr) *nh = (const ElfW(Nhdr) *)p; const uint8_t *nm = p + sizeof *nh, *desc = nm + ((nh->n_namesz + 3) & ~3u);
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 && !memcmp(nm, "GNU", 4)) { size_t o = 0; for (unsigned k = 0; k < nh->n_descsz && o + 3 < 80; k++) o += (size_t)snprintf(out + o, 80 - o, "%02x", desc[k]); return 1; }
            p = desc + ((nh->n_descsz + 3) & ~3u); } }
    return 1;
}
static const char *lib_build_id(void) { static char id[80]; static int done; if (!done) { done = 1; dl_iterate_phdr(lp_note_cb, id); } return id[0] ? id : "-"; }
// this run's start (wall clock, for file names that keep runs apart): the process's start time from /proc/self/stat, else now
static const char *run_stamp(void) {
    static char s[24]; if (s[0]) return s;
    time_t start = time(NULL); FILE *f = fopen("/proc/self/stat", "r"); char b[1024];
    if (f) { size_t k = fread(b, 1, sizeof b - 1, f); fclose(f); b[k] = 0; char *rp = strrchr(b, ')'), *sv = NULL; int field = 3;
        for (char *tok = rp ? strtok_r(rp + 2, " ", &sv) : NULL; tok; tok = strtok_r(NULL, " ", &sv), field++) if (field == 22) {
            long hz = sysconf(_SC_CLK_TCK); struct timespec bt; clock_gettime(CLOCK_BOOTTIME, &bt);
            if (hz > 0) start = time(NULL) - (time_t)(bt.tv_sec - (long long)strtoull(tok, NULL, 10) / hz); break; } }
    struct tm tm; localtime_r(&start, &tm); strftime(s, sizeof s, "%Y%m%d-%H%M%S", &tm); return s;
}
static void lp_frame(FILE *f, uint64_t a) {
    char b[96]; if (dm_addr_module((void *)(uintptr_t)a, b, sizeof b)) { fprintf(f, " %s", b); return; }
    Dl_info di; if (dladdr((void *)(uintptr_t)a, &di) && di.dli_fname) { const char *m = strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname; fprintf(f, " %s+0x%llx", m, (unsigned long long)(a - (uint64_t)(uintptr_t)di.dli_fbase)); return; }
    fprintf(f, " ?+0x%llx", (unsigned long long)a);
}
static int lp_cmp(const void *a, const void *b) { const struct { char n[256]; time_t t; } *x = a, *y = b; return (y->t > x->t) - (y->t < x->t); }
static void lp_keep_newest(int keep) {   // load_prof_*.txt in the data folder beyond the newest `keep` go (older runs' files included)
    DIR *d = opendir(g_files); if (!d) return; struct dirent *e; static struct { char n[256]; time_t t; } l[256]; int n = 0;
    while ((e = readdir(d)) && n < 256) { size_t k = strlen(e->d_name); if (strncmp(e->d_name, "load_prof_", 10) || k < 14 || strcmp(e->d_name + k - 4, ".txt")) continue;
        char p[1400]; struct stat st; snprintf(p, sizeof p, "%s/%s", g_files, e->d_name); if (stat(p, &st)) continue;
        snprintf(l[n].n, sizeof l[n].n, "%s", e->d_name); l[n].t = st.st_mtime; n++; }
    closedir(d); qsort(l, (size_t)n, sizeof *l, lp_cmp);
    for (int i = keep; i < n; i++) { char p[1400]; snprintf(p, sizeof p, "%s/%s", g_files, l[i].n); unlink(p); }
}
void dm_thread_names_refresh(void) __attribute__((weak));   // hle_thread.c (fable-loads.md I5)
static void load_prof(int start) {
    if (!lp_on()) return;
    extern int nvn_rq_tid(void) __attribute__((weak));
    if (start) {
        if (atomic_load(&g_lp_on)) return;
        static int installed; if (!installed) { installed = 1; struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = lp_sig; sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK; sigemptyset(&sa.sa_mask); sigaction(SIGPROF, &sa, NULL); }
        for (int k = 0; k < LP_SLOTS; k++) atomic_store(&g_lp_tid[k], 0);
        atomic_store(&g_lp_n, 0); g_lp_ntimer = 0; atomic_store(&g_lp_on, 1);
        lp_add(nvn_rq_tid ? nvn_rq_tid() : 0); lp_add((int)g_game_tid); return;
    }
    if (!atomic_load(&g_lp_on)) return;
    atomic_store(&g_lp_on, 0); for (int k = 0; k < g_lp_ntimer; k++) timer_delete(g_lp_timer[k]);
    int n = atomic_load(&g_lp_n); if (n > LP_N) n = LP_N;
    if (dm_thread_names_refresh) dm_thread_names_refresh();
    static int seq; char p[1200]; snprintf(p, sizeof p, "%s/load_prof_%s_%d.txt", g_files, run_stamp(), ++seq); FILE *f = fopen(p, "w");
    if (f) {
        const char *ver = getenv("DM_APP_VERSION"); long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
        fprintf(f, "# load profile: app version code %s, game %s, libnxnative.so build id %s\n# game modules:", ver ? ver : "-", DM_GAME_ID, lib_build_id());
        const char *mn, *mb; for (int i = 0; dm_module_info(i, &mn, &mb); i++) fprintf(f, " %s %s;", mn, mb[0] ? mb : "-");
        fprintf(f, "\n# samples %d (one per 1 ms of a thread's CPU); threads:", n);
        for (int k = 0; k < g_lp_ntimer; k++) { int tid = atomic_load(&g_lp_tid[k]); thr_snap s; int ok = tid && thr_one(tid, &s);
            fprintf(f, " %d = %s tid %d %.2f s CPU;", k, ok ? s.name : "(gone)", tid, ok && s.ticks > g_lp_t0[k] ? (double)(s.ticks - g_lp_t0[k]) / hz : 0); }
        fprintf(f, "\n# per line: thread, then pc, lr and (host code only) the frame chain, each as module+offset (main / sdk / ...: the game's modules; libnxnative.so: the engine)\n");
        for (int i = 0; i < n; i++) { fprintf(f, "%d", g_lp[i].who); for (int k = 0; k < g_lp[i].nf; k++) lp_frame(f, g_lp[i].f[k]); fputc('\n', f); }
        fclose(f); fprintf(stderr, "[cpu] load profile: %d samples of %d threads written to %s\n", n, g_lp_ntimer, p);
    }
    g_lp_ntimer = 0; lp_keep_newest(12);
}
// The game's main thread in a load (fable-loads.md I2): a sampler thread looks every 10 ms at its kernel state (R running, S asleep, D in the kernel's I/O),
//   the SVC it is in with the game's return address there (hle_thread.c dm_main_where) and the NVN call it is in (hle_nvn.c dm_nvn_where); no signal, so no
//   EINTR anywhere. At the end of the load the shares print: asleep in nvnWindowAcquireTexture is the load waiting for frames (cause 1), asleep in an SVC on a
//   lock (0x1a ArbitrateLock, 0x1c WaitProcessWideKeyAtomic, 0x34 WaitForAddress ...) is a hand-off between threads (cause 5). DM_LOAD_WAITPROF=0: off
int dm_main_where(int *in_svc, uint32_t *svc, uint64_t *at, uint64_t *lr) __attribute__((weak));
const char *dm_nvn_where(int tid) __attribute__((weak));
typedef struct { char st; int svc; uint64_t lr; const char *nvn; unsigned n; } wp_key;
enum { WP_K = 96 };
static wp_key g_wp[WP_K]; static int g_wp_nk; static unsigned g_wp_total, g_wp_lost; static _Atomic int g_wp_run; static pthread_t g_wp_th; static int g_wp_started;
static void *wp_main(void *a) {
    (void)a; prctl(PR_SET_NAME, "dm-waitprof", 0, 0, 0);
    while (atomic_load(&g_wp_run)) {
        thr_snap s; if (thr_one((int)g_game_tid, &s)) {
            int in = 0; uint32_t svc = 0; uint64_t at = 0, lr = 0; if (dm_main_where && !dm_main_where(&in, &svc, &at, &lr)) in = 0;
            wp_key k = { (char)s.cpu, in ? (int)svc : -1, in ? lr : 0, dm_nvn_where ? dm_nvn_where((int)g_game_tid) : NULL, 1 };
            int i = 0; for (; i < g_wp_nk; i++) if (g_wp[i].st == k.st && g_wp[i].svc == k.svc && g_wp[i].lr == k.lr && g_wp[i].nvn == k.nvn) { g_wp[i].n++; break; }
            if (i == g_wp_nk) { if (g_wp_nk < WP_K) g_wp[g_wp_nk++] = k; else g_wp_lost++; }
            g_wp_total++; }
        usleep(10000); }
    return NULL;
}
static const char *svc_name(int n) {
    switch (n) { case 0x0b: return "SleepThread"; case 0x18: return "WaitSynchronization"; case 0x1a: return "ArbitrateLock"; case 0x1b: return "ArbitrateUnlock";
        case 0x1c: return "WaitProcessWideKeyAtomic"; case 0x1d: return "SignalProcessWideKey"; case 0x34: return "WaitForAddress"; case 0x35: return "SignalToAddress"; default: return "svc"; }
}
static int wp_cmp(const void *a, const void *b) { const wp_key *x = a, *y = b; return (y->n > x->n) - (y->n < x->n); }
static void wait_prof(int start) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_LOAD_WAITPROF"); on = !(e && !strcmp(e, "0")) && g_game_tid; } if (!on) return;
    if (start) { if (g_wp_started) return; g_wp_nk = 0; g_wp_total = g_wp_lost = 0; atomic_store(&g_wp_run, 1); g_wp_started = !pthread_create(&g_wp_th, NULL, wp_main, NULL); return; }
    if (!g_wp_started) return;
    atomic_store(&g_wp_run, 0); pthread_join(g_wp_th, NULL); g_wp_started = 0;
    if (!g_wp_total) return;
    qsort(g_wp, (size_t)g_wp_nk, sizeof *g_wp, wp_cmp);
    fprintf(stderr, "[cpu] load wait profile of the game's main thread (tid %d, %u looks every 10 ms; state R running, S asleep, D in the kernel):\n", (int)g_game_tid, g_wp_total);
    for (int i = 0; i < g_wp_nk && i < 12; i++) { const wp_key *k = &g_wp[i]; char at[96] = "";
        if (k->svc >= 0) { char m[80]; if (!dm_addr_module((void *)(uintptr_t)k->lr, m, sizeof m)) snprintf(m, sizeof m, "0x%llx", (unsigned long long)k->lr); snprintf(at, sizeof at, ", in SVC 0x%x %s returning to %s", (unsigned)k->svc, svc_name(k->svc), m); }
        fprintf(stderr, "[cpu]   %4.1f %% %c%s%s%s\n", 100.0 * k->n / g_wp_total, k->st ? k->st : '?', at, k->nvn ? ", in " : k->svc < 0 ? ", in game or engine code" : "", k->nvn ? k->nvn : ""); }
    if (g_wp_lost) fprintf(stderr, "[cpu]   (%u looks fell outside the %d kinds kept)\n", g_wp_lost, WP_K);
}
// Loads on the fast cores (the white screen before a battle: one thread's CPU work, 2.7 s of a 4 s load on a Galaxy S25+, file reads 0.18 s). While a load runs, every
//   poll (500 ms) the threads that used a quarter of a core or more since the last one and are not boosted already get a little higher priority (nice -4) and,
//   when they fit (boost_plan.h, fable-loads.md L3: busy threads plus the threads pinned already at most the fast cores), the phone's fast cores (a clock limit of
//   70 % of the fastest or more). Each thread's own mask is saved and given back when the load is over. The touch menu's threads (dm-touch, dm-tapdrive,
//   dm-discover) are left out: their memory scans run at nice 10 so they never come before the game (#112: dm-touch was moved to the fast cores at nice -4 in a
//   load). The start-up prewarm's threads (dm-prewarm) are left out too: after a stamped start (nvn_vk.c, perf-1124 S3) they build the rest of the list in the
//   background at nice 10, one pipeline a present, through the first loads, and must not take the fast cores from the game. Each boosted thread is sampled by the load profile too. DM_LOAD_BOOST=0: no boost; pin: always pin (the rule before); nice: never pin
#include "boost_plan.h"
static void load_boost(int loading, uint64_t t) {
    static int mode = -1; if (mode < 0) { const char *e = getenv("DM_LOAD_BOOST"); mode = e && !strcmp(e, "0") ? 9 : e && !strcmp(e, "pin") ? BOOST_PIN : e && !strcmp(e, "nice") ? BOOST_NICE : BOOST_AUTO; }
    if (mode == 9) return;
    enum { NB = 16 };
    static thr_snap prev[512]; static int nprev; static uint64_t prev_t; static int btid[NB], bnice[NB], bpin[NB], nb; static cpu_set_t bmask[NB]; static cpu_set_t fast; static int masks, nfast, ncores;
    if (!masks) { masks = 1; long f[32]; for (int c = 0; c < 32; c++) f[c] = core_max_khz(c);
        nfast = boost_fast_cores(f, 32, &ncores); long top = 0; for (int c = 0; c < 32; c++) if (f[c] > top) top = f[c];
        CPU_ZERO(&fast); for (int c = 0; c < 32; c++) if (f[c] && f[c] * 10 >= top * 7) CPU_SET(c, &fast); }
    if (!loading) {
        if (nb) { for (int i = 0; i < nb; i++) { if (bpin[i]) sched_setaffinity(btid[i], sizeof bmask[i], &bmask[i]); setpriority(PRIO_PROCESS, (id_t)btid[i], bnice[i]); }
            fprintf(stderr, "[cpu] load over: %d thread(s) back to their own priority and cores\n", nb); nb = 0; }
        nprev = 0; return;
    }
    static thr_snap cur[512]; int nc = thr_snapshot(cur, 512); long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    if (nprev && t > prev_t) {
        double win = (double)(t - prev_t) / 1e9; int cand[NB], ncand = 0;
        for (int i = 0; i < nc && nb + ncand < NB; i++) {
            uint64_t was = 0; int seen = 0; for (int j = 0; j < nprev; j++) if (prev[j].tid == cur[i].tid) { was = prev[j].ticks; seen = 1; break; }
            if (!seen || cur[i].ticks <= was || (double)(cur[i].ticks - was) / hz < 0.25 * win) continue;
            if (!strcmp(cur[i].name, "dm-touch") || !strcmp(cur[i].name, "dm-tapdrive") || !strcmp(cur[i].name, "dm-discover") || !strcmp(cur[i].name, "dm-prewarm")) continue;
            int skip = 0; for (int k = 0; k < g_pin_n; k++) if (g_pin_tid[k] == cur[i].tid) skip = 1;
            for (int k = 0; k < nb; k++) if (btid[k] == cur[i].tid) skip = 1;
            if (!skip) cand[ncand++] = i;
        }
        if (ncand) {
            int pin = boost_pin(mode, nfast, ncores, nb + ncand, g_pin_n), unpinned = 0;
            if (!pin) for (int k = 0; k < nb; k++) if (bpin[k]) { sched_setaffinity(btid[k], sizeof bmask[k], &bmask[k]); bpin[k] = 0; unpinned++; }   // (more busy threads than fast cores now: every core back)
            char said[256] = ""; size_t sl = 0;
            for (int c = 0; c < ncand; c++) { const thr_snap *th = &cur[cand[c]];
                errno = 0; int nice_was = getpriority(PRIO_PROCESS, (id_t)th->tid); if (errno) nice_was = 0;
                CPU_ZERO(&bmask[nb]); if (sched_getaffinity(th->tid, sizeof bmask[nb], &bmask[nb])) pin = 0;
                bpin[nb] = pin && !sched_setaffinity(th->tid, sizeof fast, &fast);
                if (nice_was > -4) setpriority(PRIO_PROCESS, (id_t)th->tid, -4);
                btid[nb] = th->tid; bnice[nb] = nice_was; nb++; lp_add(th->tid);
                if (sl < sizeof said - 24) sl += (size_t)snprintf(said + sl, sizeof said - sl, " %s", th->name); }
            if (pin) fprintf(stderr, "[cpu] load: busy thread(s) to the fast cores (%d of %d) at nice -4:%s\n", nfast, ncores, said);
            else fprintf(stderr, "[cpu] load: busy thread(s) at nice -4 on every core (%d boosted + %d pinned, %d fast cores of %d%s):%s\n", nb, g_pin_n, nfast, ncores, unpinned ? ", the ones pinned before let go" : "", said);
        }
    }
    memcpy(prev, cur, sizeof *cur * (size_t)nc); nprev = nc; prev_t = t;
}
// The pipeline list shipped with the app (assets/vk_pipelines_seed.bin, build_apk.py: every pipeline the game used on a phone) into the game folder, where
//   DM_VK_PSO_SEED points: the first play after a setup gets them built before it starts too (nvn_vk.c prewarm), not only what this phone recorded. Copied
//   when missing or of another size (a new app version's list); dm.zip is only unpacked by a setup, so the list is an asset of its own
static void seed_from_assets(void) {
    AAssetManager *am = g_app && g_app->activity ? g_app->activity->assetManager : NULL; if (!am) return;
    AAsset *a = AAssetManager_open(am, "vk_pipelines_seed.bin", AASSET_MODE_STREAMING); if (!a) return;
    char dst[1100], tmp[1110]; snprintf(dst, sizeof dst, "%s/vk_pipelines_seed.bin", g_game); snprintf(tmp, sizeof tmp, "%s.part", dst);
    off_t n = AAsset_getLength(a); struct stat st;
    if (!stat(dst, &st) && st.st_size == n) { AAsset_close(a); return; }
    FILE *f = fopen(tmp, "wb"); char b[65536]; int k, ok = f != NULL; off_t got = 0;
    while (ok && (k = AAsset_read(a, b, sizeof b)) > 0) { ok = fwrite(b, 1, (size_t)k, f) == (size_t)k; got += k; }
    AAsset_close(a); if (f && fclose(f)) ok = 0;
    if (ok && got == n && !rename(tmp, dst)) fprintf(stderr, "pipeline list shipped with the app: %lld bytes into the game folder\n", (long long)n); else unlink(tmp);
}
// the shaders built before the game starts (nvn_vk.c nvn_vk_prewarm_progress, DM_VK_PREWARM): done << 32 | listed while the game waits for them, else -1
JNIEXPORT jlong JNICALL Java_com_nxnative_GameActivity_nativePrewarm(JNIEnv *env, jclass c) {
    (void)env; (void)c; extern int nvn_vk_prewarm_progress(int *done, int *total) __attribute__((weak));
    int d = 0, t = 0; if (!nvn_vk_prewarm_progress || !nvn_vk_prewarm_progress(&d, &t)) return -1;
    return (jlong)((uint64_t)(uint32_t)d << 32 | (uint32_t)t);
}
JNIEXPORT jint JNICALL Java_com_nxnative_GameActivity_nativeLoading(JNIEnv *env, jclass c) {
    (void)env; (void)c;
    static int off = -1; if (off < 0) { const char *e = getenv("DM_LOAD_HINT"); off = e && !strcmp(e, "0"); } if (off) return 0;
    extern _Atomic uint64_t dm_fs_read_bytes __attribute__((weak)); if (!&dm_fs_read_bytes) return 0;
    extern _Atomic uint64_t dm_fs_read_ns __attribute__((weak));
    extern int dm_loading_poll(void) __attribute__((weak));
    static int reads = -1; if (reads < 0) { const char *e = getenv("DM_LOAD_DETECT"); reads = (e && !strcmp(e, "reads")) || !dm_loading_poll;
        fprintf(stderr, "[cpu] loads told by %s\n", reads ? "the game's file reads (DM_LOAD_DETECT=reads)" : "the picture (hle_nvn.c: file reads with under 60 draws a frame)"); }
    static uint64_t last_b, last_t, quiet_since; static int loading;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t t = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec, b = atomic_load(&dm_fs_read_bytes);
    static thr_snap ls[512], le[512]; static int nls, nle; static uint64_t l_t0, l_b0, l_r0, e_t, e_b, e_r;   // the snapshots at its start and (reads rule) at the first quiet poll
    if (last_t && t > last_t) {
        double mbs = (double)(b - last_b) / 1048576.0 / ((double)(t - last_t) / 1e9);
        uint64_t rn = &dm_fs_read_ns ? atomic_load(&dm_fs_read_ns) : 0;
        int now = loading;
        if (!reads) now = dm_loading_poll();
        else if (!loading && mbs >= 8.0) now = 1;
        else if (loading && mbs < 1.0) { if (!quiet_since) { quiet_since = t; nle = thr_snapshot(le, 512); e_t = t; e_b = b; e_r = rn; }   // (the end snapshot: the window ends where the reads went quiet)
                                         else if (t - quiet_since >= 2000000000ull) now = 0; }
        else if (loading) quiet_since = 0;
        if (now && !loading) { loading = 1; quiet_since = 0;
            if (reads) fprintf(stderr, "[cpu] loading (%.0f MB/s of game files): the system is told\n", mbs); else fprintf(stderr, "[cpu] loading screen (%.0f MB/s of game files): the system is told\n", mbs);
            if (dm_thread_names_refresh) dm_thread_names_refresh();   // (the game's own thread names in the report: hle_thread.c)
            nls = thr_snapshot(ls, 512); l_t0 = t; l_b0 = b; l_r0 = rn; load_prof(1); wait_prof(1);
            typedef int (*spike_fn)(void *, int, int, const char *); static spike_fn sp; static int looked;
            if (!looked) { looked = 1; void *lib = dlopen("libandroid.so", RTLD_NOW); sp = lib ? (spike_fn)dlsym(lib, "APerformanceHint_notifyWorkloadSpike") : NULL; }
            if (sp && g_hint) sp(g_hint, 1, 0, "load"); }
        else if (!now && loading) { loading = 0; fprintf(stderr, "[cpu] loading over\n");
            if (!reads) { nle = thr_snapshot(le, 512); e_t = t; e_b = b; e_r = rn; }
            load_report(ls, nls, le, nle, e_t - l_t0, e_b - l_b0, e_r - l_r0); load_prof(0); wait_prof(0); quiet_since = 0; }
        load_boost(loading, t);
    }
    last_b = b; last_t = t; return loading;
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
__attribute__((weak)) void nvn_vk_cache_save(void);
__attribute__((weak)) void nvn_vk_heap_alive(void);   // nvn_vk.c: the dma-buf heap route lived to the background, so its crash marker goes
__attribute__((weak)) void nvn_vk_window_term(void);   // releases the swapchain while the window still exists (nvn_vk.c)   // the pipeline cache file (nvn_vk.c), written when the app goes to the background
static void on_cmd(struct android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW: atomic_store(&g_window, app->window); immersive(); host_start(); window_rate(app->window); break;   // (after host_start: env.txt and the Auto settings decide the rate)
        case APP_CMD_TERM_WINDOW: atomic_store(&g_window, NULL); if (g_started && nvn_vk_window_term) nvn_vk_window_term(); break;
        case APP_CMD_GAINED_FOCUS: immersive(); if (g_started) { dm_focus_set(1); dm_audio_suspend(0); } break;
        // In the background (recents, home, another app) the game keeps running with its sound, as a Switch game does under the HOME menu with a suspend-free
        //   setting: it is not told it lost focus (it would pause itself), and frames are simply not shown while there is no window. DM_BACKGROUND=off pauses it as before
        case APP_CMD_LOST_FOCUS: touch_ui_release(); if (g_started && !bg_play()) { dm_focus_set(2); } break;
        case APP_CMD_PAUSE: touch_ui_release(); if (g_started && nvn_vk_heap_alive) nvn_vk_heap_alive(); if (g_started) { if (!bg_play()) { dm_focus_set(3); dm_audio_suspend(1); } if (nvn_vk_cache_save) nvn_vk_cache_save(); } break;
        case APP_CMD_RESUME: if (g_started) { dm_focus_set(1); dm_audio_suspend(0); swapin_start(); } break;
    }
}
void android_main(struct android_app *app) {
    g_app = app; app->onAppCmd = on_cmd; app->onInputEvent = on_input;
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);
    touch_ui_init(app->activity);
    for (;;) {
        int events; struct android_poll_source *src;
        // the game never started (GameActivity handed over to the setup screen before it had a window): the process goes on for that screen
        while (ALooper_pollOnce(-1, NULL, &events, (void **)&src) >= 0) { if (src) src->process(app, src); if (app->destroyRequested) { if (g_started) _exit(0); return; } }
    }
}
