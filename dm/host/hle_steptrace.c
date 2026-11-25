// Step trace (DM_STEP_TRACE=1, off by default; fable-game-deep.md S-0): what Sword / Shield 1.3.2 measure of their own frame, the numbers their step chooser takes,
//   and the present interval they ask for, logged once a second. Nothing is written to the game. It settles which host waits the game counts as its work, what its
//   measure would be without them, and whether its 16.7 ms step (index 0 of the table main+0x203eeb0) is reachable untouched.
// What the game does (checked in both mains, 2026-11-24): the frame end main+0x5ea110 (this = the graphics system object G, the pointer in main's bss main+0x2636738,
//   read through the GOT slot main+0x2610910: RELATIVE relocation to 0x2636738 at the same place in both) stores, as signed 64-bit nanoseconds,
//     G+0x1c0  the frame's wall time (now - G+0x1b8, the previous frame end's time)
//     G+0x1d0  that wall time - the wait on the previous frame's fence (main+0x18a55d0, 1 s timeout; Sword's call is 0x90 lower) - *(*(G+0x168)+0xd0), read at
//              the function's entry (a second wait the game takes out; which one is not known yet)
//     G+0x1e0  the GPU time between two counter timestamps
//   and then present interval = the chosen step (ns) / 16,666,666.0 (the double at main+0x1bd53b8), clamped 1..5, or 0 while the byte G+0x85c9 is set, which
//   main+0x18a2cd0 passes to nvnWindowSetPresentInterval only when the window's interval (as nvnWindowGetPresentInterval reports it) differs. The step chooser
//   main+0x5df2e0 (184 words, identical in Sword and Shield) takes max(G+0x1d0, G+0x1e0). The frame end's 256 words differ between the games only in 15 BL targets
//   (calls into shifted code); the G offsets, the GOT slot and the constant are the same, so one row serves both builds.
// Per present (hle_nvn.c nvnQueuePresentTexture, on the game's thread, after the frame end wrote this frame's values) the four values and the host's own waits on that
//   thread in the frame (sync waits and acquire sleeps, hle_nvn.c t_sync_ns / t_acq_ns) are kept; once a second one line with min / median / max in ms, the
//   interval the game set and how often. Every interval change also gets a line of its own with the values that led to it. All reads go through vm_read_overwrite
//   (process_vm_readv on Android): an unmapped address fails the read, it never faults. Arceus, Paldea and other games: no row, nothing done
#include "hle.h"
#include <mach/mach.h>
#include <stdatomic.h>
#include <strings.h>

static const struct { const char *build, *game; uint64_t got, obj; uint32_t wall, cpu, gpu, sub_ptr, sub_off, no_itv; } k_trace[] = {
    { "a16802625e7826bf83b6f9708e475b91", "Shield 1.3.2", 0x2610910, 0x2636738, 0x1c0, 0x1d0, 0x1e0, 0x168, 0xd0, 0x85c9 },
    { "a3b75bcd3311385aeed67fbeeb79cbb7", "Sword 1.3.2",  0x2610910, 0x2636738, 0x1c0, 0x1d0, 0x1e0, 0x168, 0xd0, 0x85c9 },
};
#define NTRACE ((int)(sizeof k_trace / sizeof k_trace[0]))
#define NSAMP 512   // (one second at up to 512 presents)

static int rd(uint64_t a, void *out, size_t n) { vm_size_t got = 0; return a >= 0x1000 && vm_read_overwrite(mach_task_self(), (vm_address_t)a, n, (vm_address_t)out, &got) == KERN_SUCCESS && got == n; }

static int g_on = -1, g_row = -2;   // g_row: the matching row, -1 none (or the GOT slot did not hold what was expected), -2 not looked up yet
static uint8_t *g_base;
static _Atomic int g_itv_last = -1, g_itv_calls;   // the last interval the game set, and its nvnWindowSetPresentInterval calls since the last line
enum { V_WALL, V_CPU, V_GPU, V_SUB, V_HSYNC, V_HACQ, NV };
static const char *const k_vname[NV] = { "wall", "cpu", "gpu", "sub", "host sync", "host acquire" };
static int64_t g_s[NV][NSAMP]; static int g_ns;
static uint64_t g_t0, g_frames, g_prev_sync, g_prev_acq; static int g_have_prev, g_noitv;

static int trace_on(void) {
    if (g_on < 0) { const char *e = getenv("DM_STEP_TRACE"); g_on = e && atoi(e) > 0; }
    return g_on;
}
static int row(void) {
    if (g_row != -2) return g_row;
    int found = 0, r = -1;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main") && g_mod[i].build_id[0]) {
        found = 1;
        for (int k = 0; k < NTRACE; k++) if (!strncasecmp(g_mod[i].build_id, k_trace[k].build, 32)) { r = k; g_base = g_mod[i].base; }
    }
    if (!found) return -1;   // (asked before main is loaded: asked again later)
    if (r < 0) { LOG("DM_STEP_TRACE: no step trace for this game's main (%s): Sword / Shield 1.3.2 only", dm_main_build()); g_row = -1; return -1; }
    // The GOT slot holds the object pointer's address once relocated: anything else means another layout, and nothing is read
    uint64_t slot = 0, want = (uint64_t)(uintptr_t)(g_base + k_trace[r].obj);
    if (!rd((uint64_t)(uintptr_t)(g_base + k_trace[r].got), &slot, sizeof slot) || slot != want) {
        LOG("[RED] DM_STEP_TRACE: %s's GOT slot main+%#llx holds %#llx, not main+%#llx: no step trace", k_trace[r].game, (unsigned long long)k_trace[r].got, (unsigned long long)slot, (unsigned long long)k_trace[r].obj);
        g_row = -1; return -1;
    }
    LOG("DM_STEP_TRACE: the step trace for %s is on (G = **(main+%#llx); G+%#x wall, G+%#x CPU, G+%#x GPU, *(*(G+%#x)+%#x) the wait it takes out; one line a second)", k_trace[r].game,
        (unsigned long long)k_trace[r].got, k_trace[r].wall, k_trace[r].cpu, k_trace[r].gpu, k_trace[r].sub_ptr, k_trace[r].sub_off);
    g_row = r; return r;
}
static int cmp64(const void *a, const void *b) { int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return x < y ? -1 : x > y; }
// The game's own numbers now (ns; 0 where a read fails): 1 when G exists
static int read_now(int k, int64_t v[4], int *noitv) {
    uint64_t g = 0, sub = 0; uint8_t f = 0;
    if (!rd((uint64_t)(uintptr_t)(g_base + k_trace[k].obj), &g, sizeof g) || !g) return 0;   // (not created yet)
    v[0] = v[1] = v[2] = v[3] = 0;
    if (!rd(g + k_trace[k].wall, &v[0], 8)) return 0;   // (a pointer that is not the object: nothing kept)
    rd(g + k_trace[k].cpu, &v[1], 8); rd(g + k_trace[k].gpu, &v[2], 8);
    if (rd(g + k_trace[k].sub_ptr, &sub, sizeof sub) && sub) rd(sub + k_trace[k].sub_off, &v[3], 8);
    *noitv = rd(g + k_trace[k].no_itv, &f, 1) && f;
    return 1;
}
// The game called nvnWindowSetPresentInterval (hle_nvn.c, every call): kept for the next line, and a change gets a line now with the values that led to it
void dm_step_trace_interval(int v) {
    if (!trace_on()) return;
    int was = atomic_exchange(&g_itv_last, v); atomic_fetch_add(&g_itv_calls, 1);
    int k = row(); if (k < 0 || was == v) return;
    int64_t n[4]; int ni = 0;
    if (read_now(k, n, &ni)) LOG("[steptrace] the game sets present interval %d → %d at frame %llu: wall %.2f cpu %.2f gpu %.2f sub %.2f ms%s", was, v, (unsigned long long)g_frames,
                                 n[0] / 1e6, n[1] / 1e6, n[2] / 1e6, n[3] / 1e6, ni ? " (no-interval flag set)" : "");
    else LOG("[steptrace] the game sets present interval %d → %d (its graphics object is not readable)", was, v);
}
// Every present, on the game's thread; sync_ns / acq_ns: that thread's sync waits and acquire sleeps so far (never reset, hle_nvn.c)
void dm_step_trace_present(uint64_t sync_ns, uint64_t acq_ns) {
    if (!trace_on()) return;
    int k = row(); if (k < 0) return;
    int64_t n[4]; int ni = 0;
    if (!read_now(k, n, &ni)) return;
    uint64_t now = now_ns();
    if (!g_t0) g_t0 = now;
    g_frames++;
    if (g_ns < NSAMP) {
        for (int i = 0; i < 4; i++) g_s[i][g_ns] = n[i];
        g_s[V_HSYNC][g_ns] = g_have_prev ? (int64_t)(sync_ns - g_prev_sync) : 0; g_s[V_HACQ][g_ns] = g_have_prev ? (int64_t)(acq_ns - g_prev_acq) : 0;
        g_ns++;
    }
    g_prev_sync = sync_ns; g_prev_acq = acq_ns; g_have_prev = 1; g_noitv += ni;
    if (now - g_t0 < 1000000000ull || !g_ns) return;
    char line[700]; int p = snprintf(line, sizeof line, "[steptrace] %d frames in %.2f s:", g_ns, (now - g_t0) / 1e9);
    for (int i = 0; i < NV && p < (int)sizeof line - 80; i++) {
        qsort(g_s[i], (size_t)g_ns, sizeof g_s[i][0], cmp64);
        p += snprintf(line + p, sizeof line - p, " %s %.2f/%.2f/%.2f", k_vname[i], g_s[i][0] / 1e6, g_s[i][g_ns / 2] / 1e6, g_s[i][g_ns - 1] / 1e6);
    }
    int calls = atomic_exchange(&g_itv_calls, 0);
    snprintf(line + p, sizeof line - p, " ms (min/median/max); interval %d (set %d times in the second)%s", atomic_load(&g_itv_last), calls, g_noitv ? " no-interval flag seen" : "");
    LOG("%s", line);
    g_ns = 0; g_noitv = 0; g_t0 = now;
}
