// Touch on menus that have no touch support (Pokémon Legends: Arceus): a quick tap on a menu item selects it and presses A, by reading the game's UI out of
// its memory. Lists and pop-ups whose selection objects are known (jump, popup_tap) get the tapped item's index written next to it and one D-pad press:
// the selection lands on it at once, drawn by the game; other menus get the game's cursor walked there with D-pad presses. The game's menus are nn::ui2d-style layouts: pane objects with a name, a parent, a local position,
// scale and size and a visible flag. The panes are found by their class (vtable addresses from the game profile, key touch_menu_panes, valid for one build
// of main: touch_menu_build) in the process's writable memory, then kept: a rescan happens only when the kept ones no longer show a cursor.
// Pane fields (ui2d, the same in every pane class seen): +0x18 parent, +0x30 / +0x34 position, +0x48 / +0x4c scale, +0x50 / +0x54 size,
// +0x58 flags (bit 0 visible), +0x59 alpha, +0x5b base position (bits 0-1 horizontal: centre / left / right, bits 2-3 vertical: centre / top / bottom),
// +0x70 global matrix (3 x 4, calculated by the game only for layouts it draws), +0xb0 name.
// Layout space: 1920 x 1080, origin at the centre, y up.
#include <math.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include "dm.h"
#include "dm_input.h"
#include "game_profile.h"

#if defined(DM_GAME_TOUCH_MENU) && DM_GAME_TOUCH_MENU
static const uint64_t PANE_VT[] = DM_GAME_TOUCH_MENU_PANES, LIST_VT[] = DM_GAME_TOUCH_MENU_LISTS, POP_VT[] = DM_GAME_TOUCH_MENU_POPUPS;
enum { NVT = sizeof PANE_VT / sizeof *PANE_VT, NLV = sizeof LIST_VT / sizeof *LIST_VT, NPV = sizeof POP_VT / sizeof *POP_VT, MAXP = 65536, MAXL = 1024 };

static ssize_t rd(uintptr_t a, void *b, size_t n) { struct iovec l = { b, n }, r = { (void *)a, n }; return process_vm_readv(getpid(), &l, 1, &r, 1, 0); }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

typedef struct { uintptr_t a; float x0, y0, x1, y1; int vis; float al; char name[25]; } pane_t;   // a rectangle in layout space; al: its alpha times its parents' (0..1)
static uintptr_t g_vt[NVT], g_lvt[NLV], g_pvt[NPV]; static int g_ok = -1;
static uintptr_t g_lists[MAXL]; static int g_nlists;                  // list selection objects (profile touch_menu_lists) from the last scan
static uintptr_t g_pops[MAXL]; static int g_npops;                    // pop-up choice objects (profile touch_menu_popups)
static uintptr_t *g_found; static int g_nfound;                      // pane objects kept from the last scan
static struct { unsigned long s, e; } g_reg[4096]; static int g_nreg;     // the mappings that held panes (later scans read only these)

static int setup(void) {
    if (g_ok >= 0) return g_ok;
    g_ok = 0;
    if (strcmp(dm_main_build(), DM_GAME_TOUCH_MENU_BUILD)) { fprintf(stderr, "[touch-menu] off: main is build %s, the pane classes were found in %s\n", dm_main_build(), DM_GAME_TOUCH_MENU_BUILD); return 0; }
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { for (int k = 0; k < NVT; k++) g_vt[k] = (uintptr_t)g_mod[i].base + PANE_VT[k];
        for (int k = 0; k < NLV; k++) g_lvt[k] = LIST_VT[k] ? (uintptr_t)g_mod[i].base + LIST_VT[k] : 1;
        for (int k = 0; k < NPV; k++) g_pvt[k] = POP_VT[k] ? (uintptr_t)g_mod[i].base + POP_VT[k] : 1; }
    g_found = malloc(sizeof *g_found * MAXP); g_ok = g_found != NULL;
    return g_ok;
}
// every pane object: 8-byte words equal to a pane class's vtable, in all private writable mappings (first time) or the ones that held panes before
static void scan(int all) {
    static uint64_t buf[1 << 16]; double t0 = now_s(); g_nfound = 0; g_nlists = 0; g_npops = 0;
    static struct { unsigned long s, e; } *reg; static int creg; int nreg = 0;   // (the game has thousands of mappings)
    if (all || !g_nreg) {
        FILE *m = fopen("/proc/self/maps", "r"); char ln[256];
        while (m && fgets(ln, sizeof ln, m)) { unsigned long s, e; char pr[8] = "";
            if (sscanf(ln, "%lx-%lx %7s", &s, &e, pr) == 3 && pr[0] == 'r' && pr[1] == 'w' && pr[3] == 'p') {
                if (nreg == creg) { creg = creg ? creg * 2 : 4096; reg = realloc(reg, sizeof *reg * creg); }
                reg[nreg].s = s; reg[nreg].e = e; nreg++; } }
        if (m) fclose(m);
    } else { if (creg < 4096) { creg = 4096; reg = realloc(reg, sizeof *reg * creg); } for (int i = 0; i < g_nreg; i++) { reg[i].s = g_reg[i].s; reg[i].e = g_reg[i].e; } nreg = g_nreg; }
    int nkeep = 0;
    for (int r = 0; r < nreg; r++) { int hit = 0;
        for (unsigned long a = reg[r].s; a < reg[r].e && g_nfound < MAXP; a += sizeof buf) {
            size_t n = reg[r].e - a < sizeof buf ? reg[r].e - a : sizeof buf; if (rd(a, buf, n) != (ssize_t)n) continue;
            for (size_t i = 0; i < n / 8; i++) { for (int k = 0; k < NVT; k++) if (buf[i] == g_vt[k]) { if (g_nfound < MAXP) g_found[g_nfound++] = a + i * 8; hit = 1; break; }
                for (int k = 0; k < NLV; k++) if (buf[i] == g_lvt[k] && g_nlists < MAXL) { g_lists[g_nlists++] = a + i * 8; hit = 1; }
                for (int k = 0; k < NPV; k++) if (buf[i] == g_pvt[k] && g_npops < MAXL) { g_pops[g_npops++] = a + i * 8; hit = 1; } } }
        if (hit && nkeep < 4096) { g_reg[nkeep].s = reg[r].s; g_reg[nkeep].e = reg[r].e; nkeep++; } }
    g_nreg = nkeep;
    fprintf(stderr, "[touch-menu] scan (%s): %d panes, %d lists, %d pop-ups in %d of %d mappings, %.0f ms (first pane class at %lx)\n", all ? "all memory" : "kept mappings", g_nfound, g_nlists, g_npops, g_nreg, nreg,
            (now_s() - t0) * 1e3, (unsigned long)g_vt[0]);
}
// all memory again, at most every 10 s (a full scan takes 1–2 s of a phone's CPU, 8 s on an emulator): for taps the kept mappings cannot place
static double g_full_t = -100;
static void full_scan(void) { if (now_s() - g_full_t < 10) return; scan(1); g_full_t = now_s(); }
// a pane's rectangle in layout space and whether it shows (itself and every parent visible, alpha above 0), through its parents
static int pane_get(uintptr_t a, pane_t *p) {
    unsigned char b[0xc8]; if (rd(a, b, sizeof b) != (ssize_t)sizeof b) return 0;
    float tx, ty, sx, sy, w, h; memcpy(&tx, b + 0x30, 4); memcpy(&ty, b + 0x34, 4); memcpy(&sx, b + 0x48, 4); memcpy(&sy, b + 0x4c, 4); memcpy(&w, b + 0x50, 4); memcpy(&h, b + 0x54, 4);
    if (!(w == w) || !(h == h) || fabsf(w) > 1e5f || fabsf(h) > 1e5f) return 0;
    p->a = a; memcpy(p->name, b + 0xb0, 24); p->name[24] = 0;
    int vis = (b[0x58] & 1) && b[0x59]; uint8_t base = b[0x5b]; float al = b[0x59] / 255.0f;
    // up the parents: position = parent position + parent scale * own position, scale multiplied (rotation ignored)
    float gx = 0, gy = 0, gsx = 1, gsy = 1; float lx = tx, ly = ty, lsx = sx, lsy = sy; uintptr_t par; memcpy(&par, b + 0x18, 8);
    float chain[64][4]; int n = 0; chain[n][0] = lx; chain[n][1] = ly; chain[n][2] = lsx; chain[n][3] = lsy; n++;
    for (int d = 0; par && d < 63; d++) { unsigned char q[0x60]; if (rd(par, q, sizeof q) != (ssize_t)sizeof q) break;
        float qx, qy, qsx, qsy; memcpy(&qx, q + 0x30, 4); memcpy(&qy, q + 0x34, 4); memcpy(&qsx, q + 0x48, 4); memcpy(&qsy, q + 0x4c, 4);
        if (!((q[0x58] & 1) && q[0x59])) vis = 0; al *= q[0x59] / 255.0f;
        chain[n][0] = qx; chain[n][1] = qy; chain[n][2] = qsx; chain[n][3] = qsy; n++; memcpy(&par, q + 0x18, 8); }
    for (int i = n - 1; i >= 0; i--) { gx += gsx * chain[i][0]; gy += gsy * chain[i][1]; gsx *= chain[i][2]; gsy *= chain[i][3]; }
    float W = fabsf(w * gsx), H = fabsf(h * gsy), x0 = gx - W / 2, y1 = gy + H / 2;
    if ((base & 3) == 1) x0 = gx; else if ((base & 3) == 2) x0 = gx - W;
    if ((base >> 2 & 3) == 1) y1 = gy; else if ((base >> 2 & 3) == 2) y1 = gy + H;
    // Drawn at all: the game keeps menus loaded but not drawn (the Satchel's panes in the field, visible flags set), and never calculates their global matrix (+0x70, 3 x 4,
    // translation at +0x7c / +0x8c), which stays 0 (a drawn pane exactly at the screen centre is taken as not drawn: no menu item seen sits there)
    float mx, my; memcpy(&mx, b + 0x7c, 4); memcpy(&my, b + 0x8c, 4);
    if (mx == 0 && my == 0) vis = 0;
    p->x0 = x0; p->x1 = x0 + W; p->y1 = y1; p->y0 = y1 - H; p->vis = vis; p->al = al;
    if (!(x0 == x0 && y1 == y1 && W == W && H == H) || fabsf(x0) > 1e5f || fabsf(y1) > 1e5f || W > 1e5f || H > 1e5f) return 0;   // not a pane (any more): freed memory reused (NaN passed every on-screen test)
    return 1;
}
static int is_cursor(const char *n) { return strstr(n, "cursor") != NULL; }
// a *cursor* pane that is decoration, not a list's selection: on a button hint (the field HUD's ZR, L_btn_zr_00, drawn in the field), a tab arrow (the main menu's
// N_menu_end_left / right_00, whose layout keeps its last matrix once the menu closes) or the field HUD's gauges and item selector (P_cursor_00 under N_select_00,
// L_key_direct_00, N_gauge_move, L_gauge_nspoke_01, N_guage_nspoke_00 [sic]): taken for a menu, it left the stick working only on its ring in the open field
static int is_chrome(uintptr_t a) {
    char nm[25];
    for (int d = 0; d < 16 && rd(a + 0x18, &a, 8) == 8 && a; d++) { if (rd(a + 0xb0, nm, 24) != 24) break; nm[24] = 0;
        if (!strncmp(nm, "L_btn_", 6) || !strncmp(nm, "N_menu_end_", 11) || !strncmp(nm, "N_gauge", 7) || !strncmp(nm, "N_guage", 7) || !strncmp(nm, "L_gauge", 7) ||
            !strncmp(nm, "L_key_direct", 12) || !strncmp(nm, "N_key_direct", 12)) return 1; }
    return 0;
}
// a showing cursor pane on screen, not decoration: its centre and area
static int cursor_ok(uintptr_t a, float *cx, float *cy, float *area) {
    pane_t p; if (!pane_get(a, &p) || !p.vis || !is_cursor(p.name)) return 0;
    float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar < 16 || ar > 1920 * 1080 * 0.5f) return 0;
    float mx = (p.x0 + p.x1) / 2, my = (p.y0 + p.y1) / 2; if (mx < -960 || mx > 960 || my < -540 || my > 540) return 0;   // on screen only (hidden menus keep theirs off it)
    if (is_chrome(a)) return 0;
    *cx = mx; *cy = my; if (area) *area = ar; return 1;
}
// the game's cursor: `lock` when given (the one seen moving with the D-pad), else the smallest showing pane named *cursor* (its centre)
static int cursor_at(float *cx, float *cy, uintptr_t lock) {
    if (lock) return cursor_ok(lock, cx, cy, NULL);
    float best = 1e30f, x, y, ar; int ok = 0;
    for (int i = 0; i < g_nfound; i++) if (cursor_ok(g_found[i], &x, &y, &ar) && ar < best) { best = ar; *cx = x; *cy = y; ok = 1; }
    return ok;
}
// every showing cursor (a menu may draw several lists, each with its own): to find the one a D-pad press moved
enum { MAXC = 256 };
static int cursors_all(uintptr_t *a, float *x, float *y) {
    int n = 0; for (int i = 0; i < g_nfound && n < MAXC; i++) if (cursor_ok(g_found[i], &x[n], &y[n], NULL)) a[n++] = g_found[i];
    return n;
}
// what was tapped: the smallest showing part (a pane named L_*: list items, buttons) under the point, not a cursor and not most of the screen
static int target_at(float x, float y, pane_t *out) {
    pane_t p; float best = 1e30f; int ok = 0;
    for (int i = 0; i < g_nfound; i++) { if (!pane_get(g_found[i], &p) || !p.vis || is_cursor(p.name) || p.name[0] != 'L' || p.name[1] != '_' || !strncmp(p.name, "L_info", 6)) continue;   // (not the description boxes: no cursor goes there)
        if (x < p.x0 || x > p.x1 || y < p.y0 || y > p.y1) continue;
        float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar < 64 || ar > 1920 * 1080 * 0.4f || ar >= best) continue;
        best = ar; *out = p; ok = 1; }
    return ok;
}
// nothing of the menu under a point: no showing pane there but full-screen ones (backgrounds, the layout's root), the game's world showing through (beside the
// Satchel's lists, the Pokédex's): a tap there goes back (B), as tapping outside a phone's dialog closes it
// and outside the area the menu's items (showing L_* panes on screen) cover: none showing = the menu is opening or closing, not empty (a tap on the Satchel's
// grid just after a pop-up closed found no item for a moment and closed the whole menu)
static int empty_at(float x, float y) {
    pane_t p; float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    for (int i = 0; i < g_nfound; i++) { if (!pane_get(g_found[i], &p) || !p.vis) continue;
        float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar < 64 || ar >= 1920 * 1080 * 0.4f) continue;
        if (x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1) return 0;
        float mx = (p.x0 + p.x1) / 2, my = (p.y0 + p.y1) / 2;
        if (p.name[0] == 'L' && p.name[1] == '_' && mx > -960 && mx < 960 && my > -540 && my < 540) { bx0 = fminf(bx0, p.x0); by0 = fminf(by0, p.y0); bx1 = fmaxf(bx1, p.x1); by1 = fmaxf(by1, p.y1); } }
    if (bx0 > bx1) return 0;
    return !(x >= bx0 - 10 && x <= bx1 + 10 && y >= by0 - 10 && y <= by1 + 10);
}
// for tuning: the showing panes under a point and the showing cursor-like panes, to the log
static void diag(float x, float y) {
    pane_t p; int n = 0, nc = 0, nv = 0;
    for (int i = 0; i < g_nfound; i++) { if (!pane_get(g_found[i], &p) || !p.vis) continue; nv++;
        if (x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1 && n < 24) { fprintf(stderr, "[touch-menu]   under: %-24s %lx [%.0f %.0f %.0f %.0f]\n", p.name, (unsigned long)p.a, p.x0, p.y0, p.x1, p.y1); n++; }
        if ((strstr(p.name, "cursor") || strstr(p.name, "select") || strstr(p.name, "focus")) && nc < 16) { fprintf(stderr, "[touch-menu]   cursor-like: %-24s %lx [%.0f %.0f %.0f %.0f]\n", p.name, (unsigned long)p.a, p.x0, p.y0, p.x1, p.y1); nc++; } }
    fprintf(stderr, "[touch-menu]   %d of %d panes showing\n", nv, g_nfound);
}
static void press(uint64_t b) { dm_input_press_frames(b, 70000000ull, 2); usleep(90000); }   // (at least two of the game's frames: a slow frame rate swallowed 70 ms presses)
static ssize_t wr_raw(uintptr_t a, const void *b, size_t n) { struct iovec l = { (void *)b, n }, r = { (void *)a, n }; return process_vm_writev(getpid(), &l, 1, &r, 1, 0); }
// a 32-bit field of a list or pop-up object, written only while the object is still one (its first word one of the classes' vtables): the addresses are from
// the last scan, and a menu the game has freed since may have its memory reused by anything else
static int is_obj(uintptr_t o) { uint64_t w; if (rd(o, &w, 8) != 8) return 0; for (int k = 0; k < NLV; k++) if (w == g_lvt[k]) return 1; for (int k = 0; k < NPV; k++) if (w == g_pvt[k]) return 1; return 0; }
static int wr_field(uintptr_t o, unsigned off, int32_t v) { return is_obj(o) && wr_raw(o + off, &v, 4) == 4; }
static int fcmp(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static int uniq(float *v, int n) { qsort(v, n, sizeof *v, fcmp); int m = 0; for (int i = 0; i < n; i++) if (!m || v[i] - v[m - 1] > 8) v[m++] = v[i]; return m; }   // distinct centres (8 units apart)
static int rank(const float *u, int n, float x) { for (int i = 0; i < n; i++) if (fabsf(u[i] - x) <= 8) return i; return -1; }
// The tapped item selected by its index (profile touch_menu_lists), with no cursor walk: the list's items = the showing L_* panes of the tapped one's size under
// the same ancestor, their distinct centres giving its columns and rows; the item that holds the list's cursor (one item only: decoration on every row, as the
// Satchel party's N_cursor, is no cursor) gives the current place. A list object with that many columns gets the target's index next to it (left of it, or above
// it in a single column) and one D-pad press moves the game onto it, drawn by the game itself. 1 = selected and A pressed, 2 = tried and missed (left so), 0 = not such a list (walk instead),
// -1 = the list has no cursor: something else has the focus (a pop-up over it), the tap does nothing
// Scrolling and taps kept apart: a jump writes the place in the shown window, so a list still scrolling (a drag on the menu holding the stick, or the game's
// own glide after it) would get it against a window that has moved on (a stress test left the Satchel drawing empty slots and the selection on its party)
static _Atomic int g_drag; static double g_drag_end;
static void touch_menu_drag(int on) { if (on) atomic_store(&g_drag, 1); else if (atomic_exchange(&g_drag, 0)) g_drag_end = now_s(); }
// waits for no drag and every known list's first shown item (+0x10) to hold still for 0.15 s: 0 = still moving after 3 s
static int settle(void) {
    double t0 = now_s();
    while (atomic_load(&g_drag) || now_s() - g_drag_end < 0.3) { if (now_s() - t0 > 3) return 0; usleep(30000); }
    static int32_t a[MAXL], b[MAXL];
    for (;;) { for (int l = 0; l < g_nlists; l++) if (rd(g_lists[l] + 0x10, &a[l], 4) != 4) a[l] = -1;
        usleep(150000); int same = 1; for (int l = 0; l < g_nlists; l++) { if (rd(g_lists[l] + 0x10, &b[l], 4) != 4) b[l] = -1; if (b[l] != a[l]) same = 0; }
        if (same) return 1; if (now_s() - t0 > 3) return 0; }
}
// The list a scroll drag moves (touch_ui.c holds the stick): the last one a tap jumped in or seen moving, else the longest known one (the Satchel's items:
// focused when it opens). At its top row (stick up) or last row (down) the drag lets go of the stick: held on, the game moves on from the list's edge
// (up from the Satchel's top row: to its party)
static _Atomic uintptr_t g_active;
static int touch_menu_at_edge(int up) {
    static uintptr_t pa[MAXL]; static int32_t pv[MAXL][2]; int n = g_nlists < MAXL ? g_nlists : MAXL;
    uintptr_t best = 0; int bc = 0;
    // (a list's fields making sense: items, columns, rows, its place in the window; objects of the class that are no list (fields changing by themselves) left out)
    #define LIST_OK(f) ((f)[0] > 0 && (f)[6] > 0 && (f)[6] <= 16 && (f)[7] > 0 && (f)[7] <= 32 && (f)[2] >= 0 && (f)[2] * (f)[6] < (f)[0] && (f)[3] >= 0 && (f)[3] < (f)[6] * (f)[7])
    for (int l = 0; l < n; l++) { uintptr_t o = g_lists[l]; int32_t f[8]; if (rd(o + 8, f, sizeof f) != (ssize_t)sizeof f || !is_obj(o) || !LIST_OK(f)) continue;
        if (pa[l] == o && (pv[l][0] != f[2] || pv[l][1] != f[3])) atomic_store(&g_active, o);
        pa[l] = o; pv[l][0] = f[2]; pv[l][1] = f[3];
        if (f[0] > bc) { bc = f[0]; best = o; } }
    uintptr_t a = atomic_load(&g_active); int32_t fa[8];
    if (!a || !is_obj(a) || rd(a + 8, fa, sizeof fa) != (ssize_t)sizeof fa || !LIST_OK(fa)) a = best;
    int32_t f[8]; static int nmiss; if (!a || rd(a + 8, f, sizeof f) != (ssize_t)sizeof f || f[6] <= 0) { if (nmiss++ < 20) fprintf(stderr, "[touch-menu] drag: no list (%d known, best %lx, active %lx)\n", n, (unsigned long)best, (unsigned long)atomic_load(&g_active)); return 0; }
    int count = f[0], top = f[2], idx = f[3], cols = f[6];
    int edge = up ? top == 0 && idx < cols : top * cols + idx + cols >= count;
    static uintptr_t la; static int le = -1, lt = -1, li = -1;
    if (a != la || edge != le || top != lt || idx != li) { fprintf(stderr, "[touch-menu] drag on list %lx (%s, %d items, %d columns): first shown %d, place %d%s\n", (unsigned long)a,
        a == atomic_load(&g_active) ? "seen moving" : "the longest", count, cols, top, idx, edge ? ", at its edge: stick let go" : ""); la = a; le = edge; lt = top; li = idx; }
    return edge;
}
static uintptr_t root_of(uintptr_t a) { uintptr_t q; for (int d = 0; d < 32 && rd(a + 0x18, &q, 8) == 8 && q; d++) a = q; return a; }
static int jump(const pane_t *t) {
    if (!g_nlists) return 0;
    if (!settle()) { fprintf(stderr, "[touch-menu] jump: the list is still moving, tap left\n"); return 2; }
    enum { MAXI = 512 }; static float xs[MAXI], ys[MAXI], ux[MAXI], uy[MAXI]; static pane_t it[MAXI]; int n = 0;
    float tw = t->x1 - t->x0, th = t->y1 - t->y0; uintptr_t anc = t->a;
    for (int d = 1; d <= 3 && n < 2; d++) {   // (items may sit one or two null panes deep)
        if (rd(anc + 0x18, &anc, 8) != 8 || !anc) return 0;
        n = 0; pane_t p;
        for (int i = 0; i < g_nfound && n < MAXI; i++) { if (!pane_get(g_found[i], &p) || !p.vis || p.name[0] != 'L' || p.name[1] != '_' || is_cursor(p.name)) continue;
            if (fabsf(p.x1 - p.x0 - tw) > tw * 0.1f + 2 || fabsf(p.y1 - p.y0 - th) > th * 0.1f + 2) continue;
            uintptr_t q = p.a; int k = 0; while (k < d && rd(q + 0x18, &q, 8) == 8 && q) k++; if (k < d || q != anc) continue;
            int dup = 0; for (int j = 0; j < n; j++) if (it[j].a == p.a) dup = 1; if (dup) continue;
            it[n++] = p; } }
    if (n < 2) return 0;
    for (int i = 0; i < n; i++) { ux[i] = xs[i] = (it[i].x0 + it[i].x1) / 2; uy[i] = ys[i] = -(it[i].y0 + it[i].y1) / 2; }   // (rows counted downwards)
    int nc = uniq(ux, n), nr = uniq(uy, n);
    int ct = rank(ux, nc, (t->x0 + t->x1) / 2), rt = rank(uy, nr, -(t->y0 + t->y1) / 2);
    static uintptr_t ca[MAXC]; static float cxs[MAXC], cys[MAXC]; int ncur = cursors_all(ca, cxs, cys), cur = -1, holding = 0, pass = 0;
    static uintptr_t ca0[MAXC]; static float cx0[MAXC], cy0[MAXC]; int ncur0 = ncur; memcpy(ca0, ca, sizeof *ca * ncur); memcpy(cx0, cxs, sizeof *cxs * ncur); memcpy(cy0, cys, sizeof *cys * ncur);
    // cursors of the list's own layout first; when none is on an item, the others too (the Satchel's arrow over its party belongs to the item grid's layout)
    // but those in a pop-up window (a closed pop-up keeps showing its highlight over the list)
    uintptr_t root = root_of(t->a);
    static pane_t win[64]; int nwin = 0; pane_t wp;
    for (int i = 0; i < g_nfound && nwin < 64; i++) if (pane_get(g_found[i], &wp) && wp.vis && !strncmp(wp.name, "W_window", 8) && (wp.x1 - wp.x0) * (wp.y1 - wp.y0) < 1920 * 1080 * 0.3f) win[nwin++] = wp;
  again:;
    int m = 0;
    for (int c = 0; c < ncur0; c++) { int in = 0; for (int k = 0; k < nwin; k++) if (cx0[c] >= win[k].x0 && cx0[c] <= win[k].x1 && cy0[c] >= win[k].y0 && cy0[c] <= win[k].y1) in = 1;
        if (pass ? !in : root_of(ca0[c]) == root) { ca[m] = ca0[c]; cxs[m] = cx0[c]; cys[m] = cy0[c]; m++; } }
    ncur = m; cur = -1; holding = 0;
    // a cursor name found in more than one item is part of every row (the Satchel party's and the Pokédex's N_cursor), not the list's cursor
    static char cn[MAXC][25]; static int cin[MAXC]; for (int c = 0; c < ncur; c++) { if (rd(ca[c] + 0xb0, cn[c], 24) != 24) cn[c][0] = 0; cn[c][24] = 0; cin[c] = -1;
        float bc = 1e30f;   // (the nearest centre of the items it is on: the Satchel's slots overlap their neighbours)
        for (int i = 0; i < n; i++) if (cxs[c] >= it[i].x0 && cxs[c] <= it[i].x1 && cys[c] >= it[i].y0 && cys[c] <= it[i].y1) {
            float d = fabsf(cxs[c] - (it[i].x0 + it[i].x1) / 2) + fabsf(cys[c] - (it[i].y0 + it[i].y1) / 2); if (d < bc) { bc = d; cin[c] = i; } }
        // beside an item, nearer than its size (the Satchel party's arrow, left of the row once an item is being used): that item's
        float bd = 1e30f; for (int i = 0; cin[c] < 0 && i < n; i++) { float dx = fmaxf(fmaxf(it[i].x0 - cxs[c], cxs[c] - it[i].x1), 0), dy = fmaxf(fmaxf(it[i].y0 - cys[c], cys[c] - it[i].y1), 0);
            float d = dx + dy; if (d < bd && dy == 0 && dx < 0.5f * fminf(tw, th)) bd = d; }
        for (int i = 0; cin[c] < 0 && bd < 1e30f && i < n; i++) { float dx = fmaxf(fmaxf(it[i].x0 - cxs[c], cxs[c] - it[i].x1), 0), dy = fmaxf(fmaxf(it[i].y0 - cys[c], cys[c] - it[i].y1), 0);
            if (dy == 0 && dx + dy == bd) cin[c] = i; } }
    static float cw[MAXC]; for (int c = 0; c < ncur; c++) { pane_t cp; cw[c] = pane_get(ca[c], &cp) ? (cp.x1 - cp.x0) * (cp.y1 - cp.y0) * cp.al : 0; }
    for (int i = 0; i < n; i++) { int has = 0;
        for (int c = 0; c < ncur; c++) { if (cin[c] != i) continue; int other = 0, top = 1;
            for (int e = 0; e < ncur; e++) if (cin[e] >= 0 && cin[e] != i && !strcmp(cn[e], cn[c])) { other = 1; if (cw[e] * 1.25f >= cw[c]) top = 0; }
            if (!other || (top && cw[c] > 0)) has = 1; }   // (one row's own copy drawn clearly more than every other row's: the selected row's)
        if (has) { holding++; cur = i; } }
    if (!holding && !pass) { pass = 1; goto again; }
    int known = 0; for (int l = 0; l < g_nlists; l++) { int32_t f[8]; if (rd(g_lists[l] + 8, f, sizeof f) == (ssize_t)sizeof f && is_obj(g_lists[l]) && f[6] == nc && f[7] >= nr - 1 && f[7] <= nr + 1) known = 1; }
    if (holding != 1 && known) holding = 0;   // a list of a known kind is jumped to or left alone, never walked (a pop-up over it took the D-pad: the walk steered that)
    if (holding != 1 || ct < 0 || rt < 0) { fprintf(stderr, "[touch-menu] jump: %d items in %d x %d, %d hold a cursor: not a focused list\n", n, nc, nr, holding);
        for (int c = 0; c < ncur0; c++) { char nm[25] = ""; rd(ca0[c] + 0xb0, nm, 24); nm[24] = 0; pane_t cp; int ok = pane_get(ca0[c], &cp);
            fprintf(stderr, "[touch-menu]   cursor %s (%.0f, %.0f) %.0fx%.0f alpha %.2f%s\n", nm, cx0[c], cy0[c], ok ? cp.x1 - cp.x0 : 0, ok ? cp.y1 - cp.y0 : 0, ok ? cp.al : 0, root_of(ca0[c]) == root ? "" : " (other layout)"); } return holding ? 0 : -1; }
    int cc = rank(ux, nc, xs[cur]), rc = rank(uy, nr, ys[cur]);
    for (int l = 0; l < g_nlists; l++) {
        int32_t f[10]; if (rd(g_lists[l] + 8, f, sizeof f) != (ssize_t)sizeof f) continue;
        // +0x14 is the place in the shown window (cols x rows cells), +0x10 the first row shown (the Pokédex scrolled to No. 098: 97; the Satchel counts rows of 5): a target is
        // a cell of the window holding an item. (A list may show one row more than its window, partly: the Pokédex's 6 for 5)
        int count = f[0], top = f[2], idx = f[3], cols = f[6], rows = f[7], win = cols * rows;   // (+0x8, +0x10, +0x14, +0x20, +0x24)
        if (cols != nc || rows < nr - 1 || rows > nr + 1 || count <= 0 || top < 0 || idx < 0 || idx >= win || top * cols + idx >= count) continue;
        int to = idx + (rt - rc) * cols + (ct - cc); if (to < 0 || to >= win || top * cols + to >= count) continue;
        int32_t pre; uint64_t b;
        if (cols > 1 && to % cols > 0) { pre = to - 1; b = DM_BTN_RIGHT; }
        else if (cols > 1 && to % cols + 1 < cols && top * cols + to + 1 < count) { pre = to + 1; b = DM_BTN_LEFT; }
        else if (to - cols >= 0) { pre = to - cols; b = DM_BTN_DOWN; }
        else if (to + cols < win && top * cols + to + cols < count) { pre = to + cols; b = DM_BTN_UP; }
        else continue;
        int32_t g[4]; if (rd(g_lists[l] + 0x10, g, 8) != 8 || g[0] != top || g[1] != idx) { fprintf(stderr, "[touch-menu] jump: list %lx moved meanwhile, tap left\n", (unsigned long)g_lists[l]); return 2; }
        if (!wr_field(g_lists[l], 0x14, pre)) continue;
        press(b);
        int32_t now = -1; float ax = 0, ay = 0; int on = 0;
        for (int k = 0; k < 8 && !on; k++) { usleep(50000); rd(g_lists[l] + 0x14, &now, 4);
            ncur = cursors_all(ca, cxs, cys); for (int c = 0; c < ncur; c++) if (cxs[c] >= t->x0 && cxs[c] <= t->x1 && cys[c] >= t->y0 && cys[c] <= t->y1) { on = 1; ax = cxs[c]; ay = cys[c]; } }
        fprintf(stderr, "[touch-menu] jump: list %lx (%d items, %d x %d) index %d -> %d (written %d, %s), cursor %s (%.0f, %.0f)\n", (unsigned long)g_lists[l], count, cols, rows,
                idx, to, pre, b == DM_BTN_RIGHT ? "Right" : b == DM_BTN_LEFT ? "Left" : b == DM_BTN_DOWN ? "Down" : "Up", on ? "on it" : "not on it", ax, ay);
        if (on && now == to) { atomic_store(&g_active, g_lists[l]); press(DM_BTN_A); return 1; }
        if (now == pre) wr_field(g_lists[l], 0x14, idx);   // not the list the game moved: its index back
        return 2;   // (the press went somewhere, a swipe or the game in the way: nothing more, no walk on a list of a known kind)
    }
    fprintf(stderr, "[touch-menu] jump: no list object of %d x %d (%d known)\n", nc, nr, g_nlists);
    return 0;
}
// A tap on a pop-up's choice (the Satchel's Use / Favorite / Discard / Cancel: text panes T_text_* in a window W_window_*, over the list): the choices are
// the window's text rows, top to bottom; the pop-up object (profile touch_menu_popups: choice +0x8, choices +0x10) whose choice is the highlighted row gets
// the tapped one's neighbour and one press of Down / Up moves onto it, then A. 1 = the tap was on a pop-up (done, or nothing to do there), 0 = no pop-up there
// the row of a pop-up's choices (o, n of them) its highlight (P_cursor_01) is on, -1 = none showing
static int hl_row(const pane_t *w, const pane_t *o, int n) {
    static uintptr_t ca[MAXC]; static float cxs[MAXC], cys[MAXC]; int nc = cursors_all(ca, cxs, cys);
    for (int c = 0; c < nc; c++) if (cxs[c] >= w->x0 && cxs[c] <= w->x1 && cys[c] >= w->y0 && cys[c] <= w->y1)
        for (int k = 0; k < n; k++) if (fabsf((o[k].y0 + o[k].y1) / 2 - cys[c]) < 30) return k;
    return -1;
}
static int popup_tap(float x, float y) {
    pane_t w, p; int have = 0; float wa = 1e30f;
    for (int i = 0; i < g_nfound; i++) if (pane_get(g_found[i], &p) && p.vis && !strncmp(p.name, "W_window", 8) && x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1) {
        float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar > 1920 * 1080 * 0.3f || ar >= wa) continue; wa = ar; w = p; have = 1; }
    if (!have) return 0;
    uintptr_t root = root_of(w.a); enum { MAXO = 16 }; pane_t o[MAXO]; int n = 0;
    for (int i = 0; i < g_nfound && n < MAXO; i++) { if (!pane_get(g_found[i], &p) || !p.vis || strncmp(p.name, "T_text", 6)) continue;
        float cy = (p.y0 + p.y1) / 2; if (cy < w.y0 || cy > w.y1 || p.x0 < w.x0 - 20 || p.x0 > w.x1 || root_of(p.a) != root) continue;
        int dup = 0; for (int k = 0; k < n; k++) if (fabsf((o[k].y0 + o[k].y1) / 2 - cy) < 8) dup = 1; if (!dup) o[n++] = p; }
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (o[b].y1 > o[a].y1) { pane_t q = o[a]; o[a] = o[b]; o[b] = q; }   // top first
    int to = -1; float bd = 1e30f;
    for (int k = 0; k < n; k++) { float d = fabsf((o[k].y0 + o[k].y1) / 2 - y); if (d < bd) { bd = d; to = k; } }
    if (to < 0 || bd > 40) { fprintf(stderr, "[touch-menu] tap (%.0f, %.0f) in pop-up %s, %d choices: not on one\n", x, y, w.name, n); return 1; }
    int cur = hl_row(&w, o, n);
    if (cur < 0) { fprintf(stderr, "[touch-menu] pop-up %s under the tap shows no highlight: closed, not used\n", w.name); return 0; }   // (a closed one keeps its last matrix)
    if (cur == to) { fprintf(stderr, "[touch-menu] pop-up %s: on choice %d already: A\n", w.name, to); press(DM_BTN_A); return 1; }
    // several pop-up objects match (the game keeps the closed ones, at choice 0): each gets the neighbour, the one the press moves is the live one, the rest get theirs back
    static uintptr_t pa[MAXL]; static int32_t po[MAXL]; int np = 0; int32_t pre = to > 0 ? to - 1 : to + 1; uint64_t b = to > 0 ? DM_BTN_DOWN : DM_BTN_UP;
    for (int again = 0; again < 2 && !np; again++) { if (again) full_scan();
        for (int l = 0; l < g_npops && np < MAXL; l++) { int32_t f[3]; if (rd(g_pops[l] + 8, f, sizeof f) != (ssize_t)sizeof f || f[2] != n || f[0] != cur) continue;
            if (wr_field(g_pops[l], 8, pre)) { pa[np] = g_pops[l]; po[np] = f[0]; np++; } } }
    if (!np) { fprintf(stderr, "[touch-menu] tap (%.0f, %.0f) on choice %d of %d in pop-up %s (highlight on %d): no pop-up object (%d known)\n", x, y, to, n, w.name, cur, g_npops); return 1; }
    press(b);
    int live = -1, on = 0; for (int k = 0; k < 8 && live < 0 && !on; k++) { usleep(50000); for (int q = 0; q < np; q++) { int32_t v; if (rd(pa[q] + 8, &v, 4) == 4 && v == to) live = q; }
        on = hl_row(&w, o, n) == to; }   // (or the highlight is there: the live object may be one the scans have not met yet)
    for (int q = 0; q < np; q++) if (q != live) { int32_t v; if (rd(pa[q] + 8, &v, 4) == 4 && v == pre) wr_field(pa[q], 8, po[q]); }
    fprintf(stderr, "[touch-menu] pop-up %s (%d choices, %d objects): %d -> %d (written %d, %s): %s\n", w.name, n, np, cur, to, pre, b == DM_BTN_DOWN ? "Down" : "Up", live >= 0 ? "on it" : on ? "highlight on it" : "not on it");
    if (live >= 0 || on) press(DM_BTN_A);
    else for (int k = 0; k < n + 2; k++) {   // the live object is a new one the scans have not met: the rest of the way by the highlight, a row a press
        int r = hl_row(&w, o, n); if (r < 0) break;
        if (r == to) { press(DM_BTN_A); fprintf(stderr, "[touch-menu] pop-up %s: on choice %d by the highlight: A\n", w.name, to); break; }
        press(r < to ? DM_BTN_DOWN : DM_BTN_UP); usleep(120000); }
    if (live < 0) full_scan();   // (for the next tap: the live one is in mappings the scans did not keep)
    return 1;
}
// A tap on a tab (L_tab_*: the Satchel's Everyday Items / Key Items, switched with L and R): the tabs are the showing L_tab_* panes beside it, left to right;
// the selected one draws more of itself (its underline: the most showing panes under it, alone). L or R, once per tab between them. 1 = a tab (done)
static int under(uintptr_t a, uintptr_t anc, int depth) { for (int d = 0; d < depth && rd(a + 0x18, &a, 8) == 8 && a; d++) if (a == anc) return 1; return 0; }
static int tab_tap(const pane_t *t) {
    if (strncmp(t->name, "L_tab", 5)) return 0;
    uintptr_t par = 0; if (rd(t->a + 0x18, &par, 8) != 8 || !par) return 1;
    enum { MAXT = 16 }; pane_t tb[MAXT]; int nv[MAXT], n = 0; pane_t p;
    for (int i = 0; i < g_nfound && n < MAXT; i++) { uintptr_t q; if (rd(g_found[i] + 0x18, &q, 8) != 8 || q != par || !pane_get(g_found[i], &p) || !p.vis || strncmp(p.name, "L_tab", 5)) continue;
        int dup = 0; for (int k = 0; k < n; k++) if (tb[k].a == p.a) dup = 1; if (!dup) tb[n++] = p; }
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (tb[b].x0 < tb[a].x0) { pane_t q = tb[a]; tb[a] = tb[b]; tb[b] = q; }
    int k = -1; for (int i = 0; i < n; i++) { nv[i] = 0; if (tb[i].a == t->a) k = i; }
    for (int i = 0; i < g_nfound; i++) for (int j = 0; j < n; j++) if (under(g_found[i], tb[j].a, 4) && pane_get(g_found[i], &p) && p.vis) { nv[j]++; break; }
    int sel = -1, best = -1, ties = 0; for (int i = 0; i < n; i++) { if (nv[i] > best) { best = nv[i]; sel = i; ties = 0; } else if (nv[i] == best) ties++; }
    if (ties) sel = -1;
    fprintf(stderr, "[touch-menu] tab %s (%d of %d), selected %d (showing parts:", t->name, k, n, sel); for (int i = 0; i < n; i++) fprintf(stderr, " %d", nv[i]); fprintf(stderr, ")\n");
    if (k < 0 || n < 2) return 1;
    if (sel < 0) { press(k >= n / 2 ? DM_BTN_R : DM_BTN_L); return 1; }   // (not told apart: toward the tapped side)
    for (int i = 0; i < abs(k - sel); i++) { press(k > sel ? DM_BTN_R : DM_BTN_L); usleep(250000); }
    return 1;
}
static _Atomic int g_map;   // the map is open (watch_main)
// A tap on the map's own controls (map open): the Missions panel's arrow (P_mssn_base_*: R) and its "Y Missions & Requests" row, the zoom bar top right
// (ZL − … + ZR: the left half zooms out, the right half in), the prompts bottom right (A Pin / Stamp, X Choose Destination; B is touch_ui.c's menu_back)
static uint64_t map_button(float x, float y) {
    pane_t p;
    for (int i = 0; i < g_nfound; i++) if (pane_get(g_found[i], &p) && p.vis && x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1 && !strncmp(p.name, "P_mssn_base", 11)) return DM_BTN_R;
    if (x >= 380 && x < 808 && y >= -178 && y <= -108) return DM_BTN_Y;
    if (y >= 400 && y <= 480 && x >= 410 && x <= 745) return x < 576 ? DM_BTN_ZL : DM_BTN_ZR;
    if (y <= -488 && x >= 365 && x < 552) return DM_BTN_A;
    if (y <= -488 && x >= 552 && x < 840) return DM_BTN_X;
    return 0;
}
// steers the cursor into the target with D-pad presses (re-reading where it went after each one), then presses A
static void drive(float tx, float ty) {
    pane_t t; float cx, cy; uintptr_t lock = 0;
    if (!g_nfound || !cursor_at(&cx, &cy, 0)) scan(0);
    if (!cursor_at(&cx, &cy, 0)) full_scan();
    if (!cursor_at(&cx, &cy, 0)) { fprintf(stderr, "[touch-menu] tap (%.0f, %.0f): no cursor showing (not a menu)\n", tx, ty); diag(tx, ty); return; }
    if (popup_tap(tx, ty)) return;
    if (atomic_load(&g_map)) { uint64_t mb = map_button(tx, ty);
        if (mb) { press(mb); fprintf(stderr, "[touch-menu] map tap (%.0f, %.0f): %s\n", tx, ty, mb == DM_BTN_R ? "R" : mb == DM_BTN_Y ? "Y" : mb == DM_BTN_ZL ? "ZL (zoom out)" : mb == DM_BTN_ZR ? "ZR (zoom in)" : mb == DM_BTN_A ? "A" : "X"); return; } }
    if (!target_at(tx, ty, &t) && empty_at(tx, ty) && !atomic_load(&g_map)) { press(DM_BTN_B); fprintf(stderr, "[touch-menu] tap (%.0f, %.0f) outside the menu: B (back)\n", tx, ty); return; }   // (at once: a full scan takes 1–2 s on a phone)
    if (!target_at(tx, ty, &t)) { full_scan(); cursor_at(&cx, &cy, 0); if (popup_tap(tx, ty)) return; }   // a menu opened since the last scan lives in other mappings
    if (!target_at(tx, ty, &t)) {
        if (empty_at(tx, ty) && !atomic_load(&g_map)) { press(DM_BTN_B); fprintf(stderr, "[touch-menu] tap (%.0f, %.0f) outside the menu: B (back)\n", tx, ty); return; }   // (not on the map: its surface is outside every list)
        fprintf(stderr, "[touch-menu] tap (%.0f, %.0f): no menu item there\n", tx, ty); diag(tx, ty); return; }
    fprintf(stderr, "[touch-menu] tap (%.0f, %.0f) on %s [%.0f %.0f %.0f %.0f], cursor at (%.0f, %.0f)\n", tx, ty, t.name, t.x0, t.y0, t.x1, t.y1, cx, cy);
    if (tab_tap(&t)) return;
    int j = jump(&t);
    if (j < 0) {   // a pop-up has the focus (the Satchel's Use / Discard / Cancel): its layout lives in mappings the last scan did not keep
        full_scan(); pane_t u; if (popup_tap(tx, ty)) return;
        if (target_at(tx, ty, &u) && u.a != t.a) { t = u; fprintf(stderr, "[touch-menu] after a full scan: on %s [%.0f %.0f %.0f %.0f]\n", t.name, t.x0, t.y0, t.x1, t.y1); j = jump(&t); }
        if (j < 0) { fprintf(stderr, "[touch-menu] tap on %s: its list is not focused, ignored\n", t.name); diag(tx, ty); return; } }
    if (j > 0) return;
    if (!cursor_at(&cx, &cy, 0)) return;   // (where a failed jump's press left it)
    float mx = (t.x0 + t.x1) / 2, my = (t.y0 + t.y1) / 2; int stuck = 0, tried_other = 0;
    for (int step = 0; step < 40; step++) {
        if (cx >= t.x0 && cx <= t.x1 && cy >= t.y0 && cy <= t.y1) { press(DM_BTN_A); fprintf(stderr, "[touch-menu] on %s after %d moves: A\n", t.name, step); return; }
        float dx = mx - cx, dy = my - cy; int horiz = fabsf(dx) * 1.5f > fabsf(dy); if (tried_other) horiz = !horiz;
        uint64_t b = horiz ? (dx > 0 ? DM_BTN_RIGHT : DM_BTN_LEFT) : (dy > 0 ? DM_BTN_UP : DM_BTN_DOWN);
        static uintptr_t ba[MAXC]; static float bx[MAXC], by[MAXC]; int nb = lock ? 0 : cursors_all(ba, bx, by);   // (before the press, until the moving one is known)
        press(b);
        float nx = cx, ny = cy;   // where the cursor settles (it may glide): read until two reads agree, at most 0.4 s
        for (int k = 0; k < 8; k++) { usleep(50000); float ax, ay; if (!cursor_at(&ax, &ay, lock)) break; int same = fabsf(ax - nx) < 0.5f && fabsf(ay - ny) < 0.5f; nx = ax; ny = ay; if (same && k) break; }
        if (!lock) {   // the cursor this press moved, kept from now on: the smallest one may belong to another list the menu draws (the Satchel's party beside its items)
            float best = 20;   // (more than an animated cursor's wobble, ~6 units; items are ~150 apart)
            for (int i = 0; i < nb; i++) { float ax, ay; if (!cursor_ok(ba[i], &ax, &ay, NULL)) continue;
                float d = fabsf(ax - bx[i]) + fabsf(ay - by[i]); if (d > best) { best = d; lock = ba[i]; nx = ax; ny = ay; } }
            if (lock) { pane_t lp; if (pane_get(lock, &lp)) fprintf(stderr, "[touch-menu] the D-pad moves %s\n", lp.name); }
        }
        if (fabsf(nx - cx) < 1 && fabsf(ny - cy) < 1) { if (++stuck >= 2) { if (tried_other) break; tried_other = 1; stuck = 0; } }
        else { stuck = 0; tried_other = 0; }
        cx = nx; cy = ny;
    }
    fprintf(stderr, "[touch-menu] could not reach %s (cursor ended at (%.0f, %.0f))\n", t.name, cx, cy); diag(tx, ty);
}
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_scan_mu = PTHREAD_MUTEX_INITIALIZER;   // the pane list (scan, drive, the watcher)
// taps and presses of the game's own Back prompt, in the order they came (a Back pressed at once overtook the tap before it: the Satchel closed instead of its pop-up)
static struct { int back; float x, y; } g_q[8]; static int g_nq, g_started;
// Menu open = one of the game's cursor panes showing. The watcher checks the cursor panes (picked out of the pane list) twice a second and refreshes the
// list every 12 s while no menu shows; the touch controls stay faint while a menu shows (touch_ui.c), its items being tapped directly
static _Atomic int g_menu; static uintptr_t g_cur[2048]; static int g_ncur; static double g_scan_t;
static uintptr_t g_items[8192]; static int g_nitems;   // the L_* panes (list items, buttons) of the pane list, picked with the cursors
static uintptr_t g_opts[64]; static int g_nopts;      // the prompt bar's P_option_* panes (Grab / Organize Satchel / Back: Grab goes on an empty slot)
// The map (Legends: Arceus, − in the field): its crosshair P_cursor_cross_00 showing (watch_main sets g_map)
static uintptr_t g_mapc[16]; static int g_nmapc;
static uintptr_t g_hudp[16]; static int g_nhudp;   // the field HUD's ZR, L and R hints (watch_main), found at the scan
int touch_menu_map(void) { return atomic_load(&g_menu) && atomic_load(&g_map); }
static void cursors_pick(void) {
    g_ncur = g_nitems = g_nopts = g_nmapc = g_nhudp = 0; char nm[25];
    for (int i = 0; i < g_nfound; i++) if (rd(g_found[i] + 0xb0, nm, 24) == 24) { nm[24] = 0;
        if (!strcmp(nm, "P_cursor_cross_00") && g_nmapc < 16) g_mapc[g_nmapc++] = g_found[i];
        if ((!strcmp(nm, "L_btn_zr_00") || !strcmp(nm, "L_icon_key_l_00") || !strcmp(nm, "L_icon_key_r_00")) && g_nhudp < 16) g_hudp[g_nhudp++] = g_found[i];
        if (is_cursor(nm)) { if (g_ncur < 2048 && !is_chrome(g_found[i])) g_cur[g_ncur++] = g_found[i]; }
        else if (nm[0] == 'L' && nm[1] == '_' && g_nitems < 8192) g_items[g_nitems++] = g_found[i];
        else if (!strncmp(nm, "P_option_", 9) && g_nopts < 64) g_opts[g_nopts++] = g_found[i]; }
}
static int bar_options(void) {   // the prompt bar's showing actions (bottom right)
    int n = 0; pane_t p; for (int i = 0; i < g_nopts; i++) if (pane_get(g_opts[i], &p) && p.vis && (p.y0 + p.y1) / 2 < -470 && (p.x0 + p.x1) / 2 > 0 && (p.x0 + p.x1) / 2 < 960) n++;
    return n;
}
// The back button (touch_ui.c): drawn just under the list the cursor is in (Arceus's battle moves, the bag's pockets), pressing B. The list = the showing L_* panes
// that share a parent with the item under the cursor; its bounding box is kept here in layout space (y up) and read by the drawing each frame
static pthread_mutex_t g_bmu = PTHREAD_MUTEX_INITIALIZER; static int g_back_ok; static float g_back[4];
static void back_update(int menu) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f, cx = 0, cy = 0; int ok = 0;
    if (menu) {
        pane_t p, best; float ba = 1e30f; int have = 0, cur = 0, ca = 0;
        for (int i = 0; i < g_ncur; i++) if (pane_get(g_cur[i], &p) && p.vis) { float ar = (p.x1 - p.x0) * (p.y1 - p.y0), mx = (p.x0 + p.x1) / 2, my = (p.y0 + p.y1) / 2;
            if (ar < 16 || ar > 1920 * 1080 * 0.5f || mx < -960 || mx > 960 || my < -540 || my > 540) continue;
            if (!cur || ar < ca) { cur = 1; ca = ar; cx = mx; cy = my; } }
        for (int i = 0; cur && i < g_nitems; i++) if (pane_get(g_items[i], &p) && p.vis && cx >= p.x0 && cx <= p.x1 && cy >= p.y0 && cy <= p.y1) {
            float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar < 64 || ar > 1920 * 1080 * 0.4f || ar >= ba) continue; ba = ar; best = p; have = 1; }
        uintptr_t par = 0;
        if (have && rd(best.a + 0x18, &par, 8) == 8 && par)
            for (int i = 0; i < g_nitems; i++) { uintptr_t q; if (rd(g_items[i] + 0x18, &q, 8) != 8 || q != par || !pane_get(g_items[i], &p) || !p.vis) continue;
                float ar = (p.x1 - p.x0) * (p.y1 - p.y0); if (ar < 64 || ar > 1920 * 1080 * 0.4f || p.x1 < -960 || p.x0 > 960 || p.y1 < -540 || p.y0 > 540) continue;
                x0 = fminf(x0, p.x0); y0 = fminf(y0, p.y0); x1 = fmaxf(x1, p.x1); y1 = fmaxf(y1, p.y1); ok = 1; }
    }
    pthread_mutex_lock(&g_bmu);
    if (ok != g_back_ok || (ok && (g_back[0] != x0 || g_back[1] != y0 || g_back[2] != x1 || g_back[3] != y1)))
        { if (ok) fprintf(stderr, "[touch-menu] back button under the list [%.0f %.0f %.0f %.0f]\n", x0, y0, x1, y1); else if (g_back_ok) fprintf(stderr, "[touch-menu] back button hidden\n"); }
    g_back_ok = ok; if (ok) { g_back[0] = x0; g_back[1] = y0; g_back[2] = x1; g_back[3] = y1; }
    pthread_mutex_unlock(&g_bmu);
}
static double g_hud_t; static _Atomic int g_cut;
int touch_menu_cutscene(void) { return atomic_load(&g_cut) && atomic_load(&g_menu) == 0; }
static void *watch_main(void *a) {
    (void)a; prctl(PR_SET_NAME, "dm-touch", 0, 0, 0); if (!setup()) return NULL;
    for (;;) { usleep(500000);
        pthread_mutex_lock(&g_scan_mu);
        // (all memory again while few panes are found: the watcher starts with the overlay, on the title screen, before the field and its menus are loaded)
        if (!g_nfound || (!atomic_load(&g_menu) && now_s() - g_scan_t > 12)) { scan(g_nreg == 0 || g_nfound < 2000); cursors_pick(); g_scan_t = now_s(); }
        int on = 0; pane_t p;
        char why[64] = "";
        static char seen[512]; char now[512] = ""; size_t nl = 0;
        for (int i = 0; i < g_ncur; i++) if (pane_get(g_cur[i], &p) && p.vis) { float ar = (p.x1 - p.x0) * (p.y1 - p.y0), mx = (p.x0 + p.x1) / 2, my = (p.y0 + p.y1) / 2;
            if (ar < 16 || ar >= 1920 * 1080 * 0.5f || mx < -960 || mx > 960 || my < -540 || my > 540) continue;   // on screen only (as cursor_at: hidden menus keep theirs off it)
            if (!on) { snprintf(why, sizeof why, "%s [%.0f %.0f %.0f %.0f]", p.name, p.x0, p.y0, p.x1, p.y1);
                if (!atomic_load(&g_menu)) { char nm[25]; uintptr_t q = g_cur[i]; fprintf(stderr, "[touch-menu] its parents:");   // (for telling menus from HUD decoration)
                    for (int d = 0; d < 12 && rd(q + 0x18, &q, 8) == 8 && q && rd(q + 0xb0, nm, 24) == 24; d++) { nm[24] = 0; fprintf(stderr, " %s", nm); } fprintf(stderr, "\n"); } }
            on = 1; if (nl < sizeof now - 40) nl += (size_t)snprintf(now + nl, sizeof now - nl, " %s(%.0f,%.0f)", p.name, mx, my); }
        if (strcmp(now, seen)) { strcpy(seen, now); fprintf(stderr, "[touch-menu] cursors on screen:%s\n", now[0] ? now : " none");
            static int nd; if (nd++ < 3) { fprintf(stderr, "[touch-menu] panes at the field HUD's item selector:\n"); diag(730, -357); } }
        if (getenv("DM_TOUCH_HUDDIAG")) { static double lt; if (now_s() - lt > 5) { lt = now_s(); int n = 0; fprintf(stderr, "[touch-menu] drawn:");   // (for telling the field from cutscenes)
            for (int i = 0; i < g_nitems && n < 80; i++) if (pane_get(g_items[i], &p) && p.vis) { float mx = (p.x0 + p.x1) / 2, my = (p.y0 + p.y1) / 2;
                if (mx > -960 && mx < 960 && my > -540 && my < 540) { fprintf(stderr, " %s(%.0f,%.0f)", p.name, mx, my); n++; } }
            fprintf(stderr, "\n"); } }
        // the field HUD (the item selector's ZR, L and R hints, bottom right) drawn: gameplay; gone with no menu open for over a second: a cutscene, a dialogue
        // or a loading screen, where the touch controls go (touch_ui.c: a tap anywhere is A then)
        for (int i = 0; i < g_nhudp; i++) if (pane_get(g_hudp[i], &p) && p.vis && p.x1 > -960 && p.x0 < 960 && p.y1 > -540 && p.y0 < 540) { g_hud_t = now_s(); break; }
        {   // ... unless the screen is full of the game's own interface: a battle, or a menu whose cursor a message box hides (the battle Satchel's "You've already sent out …"),
        // where the controls stay. A cutscene draws a few parts at most (its dialogue box): CUT_PARTS layout parts (L_*) on screen or more is no cutscene
        int cut = !on && g_nfound > 2000 && now_s() - g_hud_t > 1.0, parts = 0;
        enum { CUT_PARTS = 14 };
        if (cut) { for (int i = 0; i < g_nitems && parts < CUT_PARTS; i++) { unsigned char b[0x90]; if (rd(g_items[i], b, sizeof b) != (ssize_t)sizeof b || !(b[0x58] & 1) || !b[0x59]) continue;
                       float mx, my; memcpy(&mx, b + 0x7c, 4); memcpy(&my, b + 0x8c, 4); if ((mx == 0 && my == 0) || !(mx > -960 && mx < 960 && my > -540 && my < 540)) continue;
                       parts++; }
                   if (parts >= CUT_PARTS) cut = 0; }
        if (cut != atomic_load(&g_cut)) { atomic_store(&g_cut, cut);
            fprintf(stderr, "[touch-menu] %s\n", cut ? "no field HUD and little else drawn: a cutscene, touch controls hidden, a tap is A" : "field HUD, a battle or a menu: touch controls shown"); } }
        if (getenv("DM_TOUCH_BARDIAG") && on) { static char last[1200]; char now[1200]; int n = 0; now[0] = 0;   // (the prompt bar, bottom right)
            for (int i = 0; i < g_nfound && n < 1150; i++) if (pane_get(g_found[i], &p) && p.vis && (p.y0 + p.y1) / 2 < -470 && (p.x0 + p.x1) / 2 > 300 && (p.x0 + p.x1) / 2 < 960) n += snprintf(now + n, sizeof now - n, " %s", p.name);
            if (strcmp(now, last)) { strcpy(last, now); fprintf(stderr, "[touch-menu] bar:%s\n", now); } }
        back_update(on);
        { int m = 0; for (int i = 0; i < g_nmapc && !m; i++) if (pane_get(g_mapc[i], &p) && p.vis && p.x1 > -960 && p.x0 < 960 && p.y1 > -540 && p.y0 < 540) m = 1;
          if (m != atomic_load(&g_map)) { atomic_store(&g_map, m); fprintf(stderr, "[touch-menu] map %s\n", m ? "open: a drag pans, a pinch zooms, its buttons are tapped" : "closed"); } }
        pthread_mutex_unlock(&g_scan_mu);
        if (on != atomic_load(&g_menu)) { atomic_store(&g_menu, on); fprintf(stderr, "[touch-menu] menu %s%s\n", on ?"open: touch controls kept faint, cursor " : "closed", why); } }
    return NULL;
}
int touch_menu_active(void) {
    static int started; if (!started) { started = 1; pthread_t t; if (!pthread_create(&t, NULL, watch_main, NULL)) pthread_detach(t); }
    return atomic_load(&g_menu);
}
static int g_sx, g_sy;   // swipe steps waiting (touch_menu_scroll): right / down
uint64_t dm_guest_frame(void);   // host/hle.h: frames the game has shown
// one D-pad step of a swipe: pressed, then released for two of the game's frames, so steps in a row are seen as separate presses
static void step(uint64_t b) { uint64_t f = dm_guest_frame(); press(b); for (int k = 0; k < 300 && dm_guest_frame() < f + 4; k++) usleep(10000); }
// The search for a list's selection index (DM_TOUCH_MENU_PROBE=1, for finding one): on the first tap on an open menu, the 32-bit words of the pane mappings that
// follow the cursor through Right, Right, Left, Left, Down, Up (+1, +1, -1, -1, a change, back) and keep still without input, logged with the object around them
// (the nearest word before them pointing into main: a class's vtable)
static void probe(int vert) {
    scan(1);   // (a pop-up's layout and lists live in mappings an earlier scan did not keep)
    size_t tot = 0; for (int r = 0; r < g_nreg; r++) tot += g_reg[r].e - g_reg[r].s;
    uint32_t *s0 = malloc(tot); enum { MAXK = 1 << 22 }; uintptr_t *ka = malloc(sizeof *ka * MAXK); uint32_t *kv = malloc(sizeof *kv * MAXK); int nk = 0;
    float cx, cy; if (!s0 || !ka || !kv) { fprintf(stderr, "[touch-menu] probe: no memory for %zu bytes\n", tot); free(s0); free(ka); free(kv); return; }
    size_t o = 0; for (int r = 0; r < g_nreg; r++) { size_t n = g_reg[r].e - g_reg[r].s; if (rd(g_reg[r].s, (char *)s0 + o, n) != (ssize_t)n) memset((char *)s0 + o, 0xff, n); o += n; }
    if (cursor_at(&cx, &cy, 0)) fprintf(stderr, "[touch-menu] probe: %zu MB in %d mappings, cursor at (%.0f, %.0f)\n", tot >> 20, g_nreg, cx, cy);
    static float px[MAXP], py[MAXP]; static char pv[MAXP]; pane_t pp;   // the showing panes that the press moves (a pop-up's highlight)
    for (int i = 0; i < g_nfound; i++) { pv[i] = pane_get(g_found[i], &pp) && pp.vis; px[i] = (pp.x0 + pp.x1) / 2; py[i] = (pp.y0 + pp.y1) / 2; }
    step(vert ? DM_BTN_DOWN : DM_BTN_RIGHT); usleep(300000);
    for (int i = 0, nm = 0; i < g_nfound && nm < 40; i++) if (pv[i] && pane_get(g_found[i], &pp) && pp.vis && fabsf((pp.x0 + pp.x1) / 2 - px[i]) + fabsf((pp.y0 + pp.y1) / 2 - py[i]) > 4) {
        nm++; fprintf(stderr, "[touch-menu] probe: moved %-24s %lx (%.0f, %.0f) -> (%.0f, %.0f) [%.0f %.0f %.0f %.0f]\n", pp.name, (unsigned long)pp.a, px[i], py[i], (pp.x0 + pp.x1) / 2, (pp.y0 + pp.y1) / 2, pp.x0, pp.y0, pp.x1, pp.y1); }
    static uint32_t buf[1 << 16]; o = 0;
    for (int r = 0; r < g_nreg; r++) for (uintptr_t a = g_reg[r].s; a < g_reg[r].e; a += sizeof buf) {
        size_t n = g_reg[r].e - a < sizeof buf ? g_reg[r].e - a : sizeof buf; uint32_t *p0 = (uint32_t *)((char *)s0 + o + (a - g_reg[r].s));
        if (rd(a, buf, n) == (ssize_t)n) for (size_t i = 0; i < n / 4; i++) if (p0[i] < 4096 && buf[i] == p0[i] + 1 && nk < MAXK) { ka[nk] = a + i * 4; kv[nk] = buf[i]; nk++; }
        if (a + sizeof buf >= g_reg[r].e) o += g_reg[r].e - g_reg[r].s; }
    free(s0);
    if (cursor_at(&cx, &cy, 0)) fprintf(stderr, "[touch-menu] probe: Right: %d words went up by 1, cursor at (%.0f, %.0f)\n", nk, cx, cy);
    static const struct { uint64_t b; int d; const char *nm; } seq[] = { { 0, 0, "still" }, { DM_BTN_RIGHT, 1, "Right" }, { 0, 0, "still" }, { DM_BTN_LEFT, -1, "Left" },
        { DM_BTN_LEFT, -1, "Left" }, { 0, 0, "still" }, { DM_BTN_DOWN, 99, "Down" }, { DM_BTN_UP, -99, "Up" }, { 0, 0, "still" } },
        vseq[] = { { 0, 0, "still" }, { DM_BTN_DOWN, 1, "Down" }, { 0, 0, "still" }, { DM_BTN_UP, -1, "Up" }, { DM_BTN_UP, -1, "Up" }, { 0, 0, "still" } };
    enum { MAXD = 1 << 16 }; static int32_t dn[MAXD]; static uint32_t bd[MAXD];   // per word left after Down: its change, its value before
    const struct { uint64_t b; int d; const char *nm; } *sq = vert ? (const void *)vseq : (const void *)seq; size_t nsq = vert ? sizeof vseq / sizeof *vseq : sizeof seq / sizeof *seq;
    for (size_t s = 0; s < nsq; s++) {
        if (sq[s].b) step(sq[s].b); usleep(sq[s].b ? 300000 : 600000);
        int m = 0;
        for (int i = 0; i < nk; i++) { uint32_t v; if (rd(ka[i], &v, 4) != 4) continue; int ok;
            if (sq[s].d == 99) { ok = v != kv[i] && m < MAXD; if (ok) { bd[m] = kv[i]; dn[m] = (int32_t)(v - kv[i]); } }
            else if (sq[s].d == -99) ok = v == bd[i];
            else ok = v == kv[i] + (uint32_t)sq[s].d;
            if (ok) { if (sq[s].d != 99 && i < MAXD) { dn[m] = dn[i]; bd[m] = bd[i]; } ka[m] = ka[i]; kv[m] = v; m++; } }
        nk = m;
        if (cursor_at(&cx, &cy, 0)) fprintf(stderr, "[touch-menu] probe: %s: %d left, cursor at (%.0f, %.0f)\n", sq[s].nm, nk, cx, cy); }
    uintptr_t mb = 0, me = 0; for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mb = (uintptr_t)g_mod[i].base; me = mb + g_mod[i].size; }
    for (int i = 0; i < nk && i < 48; i++) {
        uintptr_t vt = 0, at = 0; for (uintptr_t b = (ka[i] & ~(uintptr_t)7); b + 0x800 > ka[i] && b > 0x800; b -= 8) { uint64_t w; if (rd(b, &w, 8) == 8 && w >= mb && w < me) { vt = w; at = b; break; } }
        uint32_t h[24]; memset(h, 0, sizeof h); rd(ka[i] - 0x30, h, sizeof h);
        fprintf(stderr, "[touch-menu] probe: %lx = %u (Down %+d), main+%lx at -0x%lx:", (unsigned long)ka[i], kv[i], dn[i], (unsigned long)(vt ? vt - mb : 0), (unsigned long)(ka[i] - at));
        for (int k = 0; k < 24; k++) fprintf(stderr, "%s%08x", k == 12 ? " [" : " ", h[k]); fprintf(stderr, "\n"); }
    free(ka); free(kv);
}
// A drag up / down on an open menu (touch_ui.c: touch_menu_drag_rate, the finger's distance from where it landed, -1..1, positive = up the list) runs the
// list's selection a row a step, faster the further the finger (3 to 12 rows a second), as the D-pad held would: a stick held makes the game skip to a list's end
// in a moment (the Satchel's 420 slots, most of them empty). It stops at the top row, and past the last item (an empty slot after a step down: a step back)
static _Atomic int g_rate;   // thousandths
static void *scroll_main(void *a) { prctl(PR_SET_NAME, "dm-scroll", 0, 0, 0);
    (void)a; int end = 0;
    for (;;) { int r = atomic_load(&g_rate);
        if (!r || !atomic_load(&g_menu)) { end = 0; usleep(30000); continue; }
        int up = r > 0; if (up) end = 0;
        if ((up && touch_menu_at_edge(1)) || (!up && end)) { usleep(30000); continue; }
        int o0 = 0; if (!up && !pthread_mutex_lock(&g_scan_mu)) { o0 = bar_options(); pthread_mutex_unlock(&g_scan_mu); }
        double t0 = now_s(); step(up ? DM_BTN_UP : DM_BTN_DOWN);
        if (!up && o0 && !pthread_mutex_lock(&g_scan_mu)) { usleep(60000); int o1 = bar_options(); pthread_mutex_unlock(&g_scan_mu);   // (an empty slot: the bar loses an action, the Satchel's Grab)
            if (o1 < o0) { step(DM_BTN_UP); end = 1; fprintf(stderr, "[touch-menu] drag: past the last item (%d actions, then %d), back on it\n", o0, o1); } }
        double gap = 1.0 / (3 + 9 * fabsf(r / 1000.0f)) - (now_s() - t0); if (gap > 0) usleep((useconds_t)(gap * 1e6)); }
    return NULL;
}
void touch_menu_drag_rate(float r) {
    int m = (int)(r * 1000); if (m > 1000) m = 1000; if (m < -1000) m = -1000;
    static _Atomic int started; if (m && !atomic_exchange(&started, 1)) { pthread_t t; if (!pthread_create(&t, NULL, scroll_main, NULL)) pthread_detach(t); }
    atomic_store(&g_rate, m); touch_menu_drag(m != 0);
}
static void *driver_main(void *a) { prctl(PR_SET_NAME, "dm-tapdrive", 0, 0, 0);
    (void)a;
    for (;;) { pthread_mutex_lock(&g_mu); while (!g_nq && !g_sx && !g_sy) pthread_cond_wait(&g_cv, &g_mu);
        if (!g_nq && !atomic_load(&g_menu)) { g_sx = g_sy = 0; pthread_mutex_unlock(&g_mu); continue; }   // the menu closed: its waiting steps go
        if (g_sx || g_sy) {   // one step at a time, so a newer swipe (touch_menu_scroll) replaces the steps still waiting
            uint64_t b = g_sy ? (g_sy > 0 ? DM_BTN_DOWN : DM_BTN_UP) : (g_sx > 0 ? DM_BTN_RIGHT : DM_BTN_LEFT);
            if (g_sy) g_sy -= g_sy > 0 ? 1 : -1; else g_sx -= g_sx > 0 ? 1 : -1;
            pthread_mutex_unlock(&g_mu); step(b); continue; }
        int back = g_q[0].back; float x = g_q[0].x, y = g_q[0].y; g_nq--; memmove(g_q, g_q + 1, sizeof *g_q * g_nq); int tap = !back; pthread_mutex_unlock(&g_mu);
        if (back) { dm_input_press_frames(DM_BTN_B, 150000000ull, 4); usleep(200000); fprintf(stderr, "[touch-menu] the menu's own Back prompt: B\n"); continue; }
        const char *pr = getenv("DM_TOUCH_MENU_PROBE");   // (a tap in the top left corner starts it: v = Down and Up)
        if (tap && pr && x < -850 && y > 450 && setup() && atomic_load(&g_menu)) { pthread_mutex_lock(&g_scan_mu); probe(*pr == 'v'); pthread_mutex_unlock(&g_scan_mu); continue; }
        if (tap && setup()) { pthread_mutex_lock(&g_scan_mu); drive(x, y); if (g_nfound) { cursors_pick(); g_scan_t = now_s(); } pthread_mutex_unlock(&g_scan_mu); } }
    return NULL;
}
static void driver_start_locked(void) { if (!g_started) { pthread_t t; if (!pthread_create(&t, NULL, driver_main, NULL)) { pthread_detach(t); g_started = 1; }
} }
// a swipe on the open menu: D-pad steps (right / down; negative = left / up), in place of any still waiting, to the driver thread
void touch_menu_scroll(int right, int down) {
    pthread_mutex_lock(&g_mu); driver_start_locked(); g_sx = right; g_sy = down; pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
    fprintf(stderr, "[touch-menu] swipe: %d right, %d down\n", right, down);
}
// a quick tap at screen point (x, y) of a w x h window showing the 16:9 game picture fitted (as the presentation blits it): to the driver thread
void touch_menu_tap(float x, float y, float w, float h) {
    float s = fminf(w / 1920.0f, h / 1080.0f), ox = (w - 1920 * s) / 2, oy = (h - 1080 * s) / 2;
    float lx = (x - ox) / s - 960, ly = 540 - (y - oy) / s;
    if (lx < -960 || lx > 960 || ly < -540 || ly > 540) return;
    pthread_mutex_lock(&g_mu);
    driver_start_locked();
    if (g_nq == 8) g_nq--;   // (a burst: the newest replaces the last waiting)
    g_q[g_nq].back = 0; g_q[g_nq].x = lx; g_q[g_nq].y = ly; g_nq++; pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
}
// a tap on the game's own Back prompt: B, after the taps still waiting
void touch_menu_back(void) {
    pthread_mutex_lock(&g_mu); driver_start_locked();
    if (g_nq == 8) g_nq--;
    g_q[g_nq].back = 1; g_nq++; pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
}
// The list the back button goes under, in pixels of a w x h window (the 16:9 picture fitted as the presentation blits it): 0 when there is none
int touch_menu_back_list(float w, float h, float r[4]) {
    if (!atomic_load(&g_menu)) return 0;
    pthread_mutex_lock(&g_bmu); int ok = g_back_ok; float b[4]; memcpy(b, g_back, sizeof b); pthread_mutex_unlock(&g_bmu);
    if (!ok) return 0;
    float s = fminf(w / 1920.0f, h / 1080.0f), ox = (w - 1920 * s) / 2, oy = (h - 1080 * s) / 2;
    r[0] = ox + (b[0] + 960) * s; r[2] = ox + (b[2] + 960) * s; r[1] = oy + (540 - b[3]) * s; r[3] = oy + (540 - b[1]) * s;   // (left, top, right, bottom)
    return 1;
}
#else
void touch_menu_tap(float x, float y, float w, float h) { (void)x; (void)y; (void)w; (void)h; }
void touch_menu_scroll(int right, int down) { (void)right; (void)down; }
int touch_menu_active(void) { return 0; }
void touch_menu_back(void) { dm_input_press_frames(DM_BTN_B, 150000000ull, 4); }
void touch_menu_drag_rate(float r) { (void)r; }
int touch_menu_cutscene(void) { return 0; }
int touch_menu_map(void) { return 0; }
int touch_menu_back_list(float w, float h, float r[4]) { (void)w; (void)h; (void)r; return 0; }
#endif
