// Step trace unit test (hle_steptrace.c, DM_STEP_TRACE=1): the row is picked by main's build ID only (Shield / Sword 1.3.2); the GOT slot must hold the object
// pointer's address or nothing is read; a graphics object pointer into unmapped memory, and a sub-object pointer into unmapped memory, are read without a fault;
// the values are kept per present and a line goes out once a second (fake clock), after which the samples start again; interval calls are counted.
// A fake main image and graphics object stand in for the game. PC Linux (make GAME=<id> linux-step) or macOS
#ifndef __APPLE__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // (process_vm_readv)
#endif
#include <sys/uio.h>
#include <unistd.h>
#endif
#include <sys/mman.h>
#include "../hle_steptrace.c"
#ifndef __APPLE__
kern_return_t vm_read_overwrite(task_t t, vm_address_t a, vm_size_t n, vm_address_t to, vm_size_t *got) {
    (void)t; struct iovec l = { (void *)to, n }, r = { (void *)a, n }; ssize_t k = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (k < 0) return 1;
    *got = (vm_size_t)k; return KERN_SUCCESS;
}
#endif
module_t g_mod[MAX_MOD]; int g_nmod;
static uint64_t g_now = 1000000000ull;
uint64_t now_ns(void) { return g_now; }
static int checks, fails;
#define CHECK(c, s) do { checks++; int ok = !!(c); fails += !ok; printf("%s %s\n", ok ? "✓" : "✗", s); } while (0)

int main(void) {
    setenv("DM_STEP_TRACE", "1", 1);
    dm_step_trace_present(0, 0);
    CHECK(g_row == -2, "before main is loaded: the row is looked up again later");
    g_nmod = 1; strcpy(g_mod[0].name, "main"); strcpy(g_mod[0].build_id, "00000000000000000000000000000000");
    dm_step_trace_present(0, 0);
    CHECK(g_row == -1, "another main build: no trace");
    // Shield 1.3.2's main, but the GOT slot does not hold main+0x2636738: nothing is read
    uint8_t *img = calloc(1, 0x2637000); g_mod[0].base = img; strcpy(g_mod[0].build_id, "a16802625e7826bf83b6f9708e475b91");
    g_row = -2; dm_step_trace_present(0, 0);
    CHECK(g_row == -1, "Shield, the GOT slot not relocated as expected: no trace");
    // relocated GOT slot; the object not created yet (null)
    uint64_t want = (uint64_t)(uintptr_t)(img + 0x2636738); memcpy(img + 0x2610910, &want, 8);
    g_row = -2; dm_step_trace_present(0, 0);
    CHECK(g_row == 0 && g_frames == 0, "Shield: the row found, nothing kept while the graphics object is null");
    // the object pointer into unmapped memory: no fault, nothing kept
    uint8_t *hole = mmap(NULL, 0x10000, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0); munmap(hole, 0x10000);
    uint64_t hp = (uint64_t)(uintptr_t)hole; memcpy(img + 0x2636738, &hp, 8);
    dm_step_trace_present(0, 0); dm_step_trace_interval(2);
    CHECK(g_frames == 0, "the graphics object pointer into unmapped memory: read without a fault, nothing kept");
    // a real object whose sub-object pointer (G+0x168) is into unmapped memory: the sub value reads as 0
    uint8_t *obj = calloc(1, 0x8600); uint64_t op = (uint64_t)(uintptr_t)obj; memcpy(img + 0x2636738, &op, 8);
    int64_t wall = 33300000, cpu = 7000000, gpu = 2800000; memcpy(obj + 0x1c0, &wall, 8); memcpy(obj + 0x1d0, &cpu, 8); memcpy(obj + 0x1e0, &gpu, 8);
    memcpy(obj + 0x168, &hp, 8);
    dm_step_trace_present(100, 200);
    CHECK(g_frames == 1 && g_ns == 1 && g_s[V_WALL][0] == wall && g_s[V_CPU][0] == cpu && g_s[V_GPU][0] == gpu && g_s[V_SUB][0] == 0, "the four values kept; an unmapped sub-object reads as 0");
    // the sub-object, and host waits as differences between presents
    uint8_t *sub = calloc(1, 0x100); int64_t sv = 26000000; memcpy(sub + 0xd0, &sv, 8); uint64_t sp = (uint64_t)(uintptr_t)sub; memcpy(obj + 0x168, &sp, 8);
    g_now += 33000000; dm_step_trace_present(100 + 5000000, 200 + 20000000);
    CHECK(g_ns == 2 && g_s[V_SUB][1] == sv && g_s[V_HSYNC][1] == 5000000 && g_s[V_HACQ][1] == 20000000, "the sub value read through G+0x168 → +0xd0; host sync / acquire per frame");
    // interval calls counted, a change remembered
    dm_step_trace_interval(2); dm_step_trace_interval(1);
    CHECK(atomic_load(&g_itv_last) == 1 && atomic_load(&g_itv_calls) == 3, "interval calls counted (the one before the object existed too), the last kept");
    // a second later the line goes out and the samples start again
    for (int i = 0; i < 30; i++) { g_now += 33333333; dm_step_trace_present(100 + 5000000, 200 + 20000000); }
    CHECK(g_ns == 0 && atomic_load(&g_itv_calls) == 0, "after a second: one line, samples and interval calls start again");
    printf(fails ? "FAILED: %d of %d\n" : "all passed: %d checks\n", fails ? fails : checks, checks);
    return fails != 0;
}
