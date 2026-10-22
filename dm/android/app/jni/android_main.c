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
#include <jni.h>
#include <sys/resource.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "dm_input.h"
#include "front.h"

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
    if (type == AINPUT_EVENT_TYPE_MOTION && (src & AINPUT_SOURCE_TOUCHSCREEN)) { touch_update(e); return 1; }
    return 0;
}

// ---------- Host start ----------
static int g_started;
static void env_file(const char *path) {   // KEY=VALUE lines, # comments; set only when not already set
    FILE *f = fopen(path, "r"); char ln[512]; if (!f) return;
    while (fgets(ln, sizeof ln, f)) { char *p = ln; while (*p == ' ' || *p == '\t') p++; if (*p == '#' || !*p) continue; char *eq = strchr(p, '='); if (!eq || eq == p) continue;
        *eq = 0; char *v = eq + 1; char *e = strpbrk(v, "\r\n"); if (e) *e = 0; if (!getenv(p)) { setenv(p, v, 0); fprintf(stderr, "boot: env %s=%s\n", p, v); } }
    fclose(f);
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
static void host_start(void) {
    if (g_started++) return;
    const char *f = g_app->activity->externalDataPath; snprintf(g_files, sizeof g_files, "%s", f ? f : "/sdcard/nxnative");
    mkdir(g_files, 0755);
    char p[1200], q[1200];
    snprintf(p, sizeof p, "%s/log.txt", g_files); snprintf(q, sizeof q, "%s/log.prev.txt", g_files); rename(p, q);
    if (freopen(p, "w", stderr)) setvbuf(stderr, NULL, _IOLBF, 0);
    copy_file(q, DL "/PokemonShield-log-previous.txt");
    { pthread_t lt; if (!pthread_create(&lt, NULL, log_mirror, NULL)) pthread_detach(lt); }
    snprintf(p, sizeof p, "%s/env.txt", g_files); env_file(p);
    // Game files: inside the installed APKs (the all-in-one .apks bundle, read through apkfs.c as /nxapk/...), else copied into the files folder
    struct stat st; int inapk = !stat("/nxapk/RomFS", &st);
    snprintf(g_game, sizeof g_game, "%s", inapk ? "/nxapk" : g_files);
    #define SETG(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_game, sub); setenv(k, p, 0); } while (0)
    #define SETP(k, sub) do { snprintf(p, sizeof p, "%s/%s", g_files, sub); setenv(k, p, 0); } while (0)
    setenv("DM_AUDIO", "1", 0);   // real sound output (AAudio, compat/aaudio_unit.c); env.txt can set 0
    // pipelines: every one the game makes is recorded here and built ahead at the next start, the list shipped with the game too, and the driver's compiled code kept
    SETP("DM_VK_PSO_RECORD", "vk_pipelines.bin"); SETP("DM_VK_PCACHE", "vk_pipeline_cache.bin"); SETG("DM_VK_PSO_SEED", "vk_pipelines_seed.bin");
    snprintf(q, sizeof q, "%s/shaders.pak", g_game); if (!stat(q, &st)) setenv("DM_SPV_PAK", q, 0);   // the packed shader set, else vk/b-<chain>/
    { snprintf(p, sizeof p, "%s/aoc", g_game); struct stat as; if (!stat(p, &as)) setenv("DM_AOC_DIR", p, 0); }   // the DLC the player owns (Shield: the Expansion Pass keys; its content is in the update)
    SETG("DM_ROMFS", "RomFS"); SETG("DM_SPV_DIR", "vk"); SETG("DM_NVN_SHMAP", "shader_map.tsv"); SETG("DM_FONT_DIR", "font");
    SETP("DM_SAVE", "save"); SETP("DM_TEMP", "temp");
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
static int (*p_hint_target)(void *, int64_t); static int (*p_hint_report)(void *, int64_t);
void front_frame_work(uint64_t busy_ns, uint64_t target_ns) {
    if (g_hint_failed) return;
    if (!g_hint) {
        static int tried; if (tried++) return;
        setpriority(PRIO_PROCESS, (id_t)gettid(), -10);
        void *lib = dlopen("libandroid.so", RTLD_NOW); if (!lib) { g_hint_failed = 1; return; }
        p_hint_mgr = (void *(*)(void))dlsym(lib, "APerformanceHint_getManager");
        p_hint_create = (void *(*)(void *, const int32_t *, size_t, int64_t))dlsym(lib, "APerformanceHint_createSession");
        p_hint_target = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
        p_hint_report = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
        void *m = p_hint_mgr ? p_hint_mgr() : NULL;
        int32_t t[2]; size_t n = 0; t[n++] = gettid(); if (g_game_tid && g_game_tid != gettid()) t[n++] = g_game_tid;
        g_hint = m && p_hint_create ? p_hint_create(m, t, n, (int64_t)target_ns) : NULL;
        if (!g_hint || !p_hint_report) { g_hint_failed = 1; fprintf(stderr, "[cpu] no performance hint session\n"); return; }
        g_hint_target = (int64_t)target_ns;
        fprintf(stderr, "[cpu] performance hint session: %zu threads, target %.1f ms; presenting thread at priority -10\n", n, target_ns / 1e6);
    }
    if ((int64_t)target_ns != g_hint_target && p_hint_target) p_hint_target(g_hint, g_hint_target = (int64_t)target_ns);
    if (busy_ns) p_hint_report(g_hint, (int64_t)busy_ns);
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
__attribute__((weak)) void nvn_vk_cache_save(void);   // the pipeline cache file (nvn_vk.c), written when the app goes to the background
static void on_cmd(struct android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW: atomic_store(&g_window, app->window); immersive(); host_start(); break;
        case APP_CMD_TERM_WINDOW: atomic_store(&g_window, NULL); break;
        case APP_CMD_GAINED_FOCUS: immersive(); if (g_started) { dm_focus_set(1); dm_audio_suspend(0); } break;
        case APP_CMD_LOST_FOCUS: if (g_started) { dm_focus_set(2); } break;
        case APP_CMD_PAUSE: if (g_started) { dm_focus_set(3); dm_audio_suspend(1); if (nvn_vk_cache_save) nvn_vk_cache_save(); } break;
        case APP_CMD_RESUME: if (g_started) { dm_focus_set(1); dm_audio_suspend(0); } break;
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
