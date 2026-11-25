// Lock and wait SVCs under contention, for both wake paths of hle_thread.c (DM_SVC_WAKE=locked, the v1.0 default, and DM_SVC_WAKE=after: wakes signalled after the
//   SVC lock is let go, waiters listed by address). The mode is read once per process, so the test runs once per mode (make GAME=<id> linux-thread runs both and
//   compares). Guest threads are host pthreads registered with a priority (dm_test_thread_register); the mutex and condition variable are the SDK's user-mode protocol
//   around the SVCs (as t_condvar): CAS on the lock word, ArbitrateLock 0x1a with the waiters bit, ArbitrateUnlock 0x1b; WaitProcessWideKeyAtomic 0x1c, and
//   SignalProcessWideKey 0x1d only when the key is non-zero.
// Lines starting "=" must be the same in both modes (who gets the lock in what order, condition and address wake order, SignalToAddress's value rules, timeouts,
//   CancelSynchronization); the stress part (T_SECS, default 10 s) checks invariants: mutual exclusion, every produced item consumed, every address waiter released at
//   the end, and no stall (a lost wakeup with waits that have no timeout stops a workload for good: the monitor reports it, the watchdog ends a hang)
#include "../dm.h"
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/resource.h>
#include <time.h>
int svc_thread(int num, uint64_t *regs);
uint32_t svc_thread_main_init(void);
uint32_t dm_test_thread_register(int prio);
void (*dm_test_cv_after_release)(void);   // (hle_thread.c's t_condvar hook; unused here)

#define WBIT 0x40000000u
#define R_TIMEOUT 0xEA01u
#define R_CANCELLED 0xEC01u
#define R_BAD_STATE 0xFA01u
static uint64_t svc(int num, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t *r1) {
    uint64_t r[32] = { a0, a1, a2, a3 }; svc_thread(num, r); if (r1) *r1 = r[1]; return r[0];
}
static __thread uint32_t my_tag;
static _Atomic int g_err;
static void err(const char *what) { if (atomic_fetch_add(&g_err, 1) < 20) printf("ERROR: %s\n", what); }
static void mx_lock(_Atomic uint32_t *m) {
    for (;;) {
        uint32_t v = 0;
        if (atomic_compare_exchange_strong(m, &v, my_tag)) return;
        if (!(v & WBIT)) { uint32_t nv = v | WBIT; if (!atomic_compare_exchange_strong(m, &v, nv)) continue; v = nv; }
        svc(0x1A, v & ~WBIT, (uint64_t)(uintptr_t)m, my_tag, 0, NULL);
        if ((atomic_load(m) & ~WBIT) == my_tag) return;
    }
}
static void mx_unlock(_Atomic uint32_t *m) {
    uint32_t v = my_tag; if (atomic_compare_exchange_strong(m, &v, 0)) return;
    if ((v & ~WBIT) != my_tag) err("unlock of a lock this thread does not hold");
    svc(0x1B, (uint64_t)(uintptr_t)m, 0, 0, 0, NULL);
}
static uint64_t cv_wait(_Atomic uint32_t *m, _Atomic uint32_t *key, int64_t ns) {
    uint64_t r = svc(0x1C, (uint64_t)(uintptr_t)m, (uint64_t)(uintptr_t)key, my_tag, (uint64_t)ns, NULL);
    if (r == R_TIMEOUT) mx_lock(m);   // (the kernel does not take the lock back on a timeout: the SDK does)
    else if (r != 0) err("condition wait failed");
    else if ((atomic_load(m) & ~WBIT) != my_tag) err("condition wait returned without the lock");
    return r;
}
static void cv_signal(_Atomic uint32_t *key, int n) { if (atomic_load(key)) svc(0x1D, (uint64_t)(uintptr_t)key, (uint64_t)(int64_t)n, 0, 0, NULL); }
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void ms_sleep(int ms) { usleep((useconds_t)ms * 1000); }
static uint32_t rnd(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

// ---------- Stress ----------
static _Atomic int g_stop;
// A: a counter under one lock, 6 threads of different priority
static _Atomic uint32_t mA; static uint64_t cntA; static int insideA; static _Atomic uint64_t opsA;
static void *thrA(void *p) {
    int prio = (int)(intptr_t)p; my_tag = dm_test_thread_register(prio); uint32_t s = my_tag * 2654435761u | 1;
    while (!atomic_load(&g_stop)) {
        mx_lock(&mA);
        if (insideA++) err("A: two threads inside the lock");
        cntA++; for (volatile uint32_t k = rnd(&s) % 200; k; k--) {}
        insideA--;
        mx_unlock(&mA); atomic_fetch_add(&opsA, 1);
        if ((rnd(&s) & 63) == 0) sched_yield();
    }
    return NULL;
}
// B: a bounded queue (8) with two condition variables, 3 producers and 3 consumers; half the waits have no timeout, half 300 µs
static _Atomic uint32_t mB, kNotEmpty, kNotFull; static uint64_t qB[8]; static int qh, qn, stopB; static uint64_t prodN, prodSum, consN, consSum, toB; static _Atomic uint64_t progB;
static void *prodB(void *p) {
    my_tag = dm_test_thread_register((int)(intptr_t)p); uint64_t seq = (uint64_t)my_tag << 32; uint32_t i = 0;
    for (;;) {
        mx_lock(&mB);
        while (qn == 8 && !stopB) if (cv_wait(&mB, &kNotFull, (i++ & 1) ? 300000 : -1) == R_TIMEOUT) toB++;
        if (stopB) { mx_unlock(&mB); break; }
        uint64_t v = ++seq; qB[(qh + qn++) % 8] = v; prodN++; prodSum += v;
        cv_signal(&kNotEmpty, 1);
        mx_unlock(&mB);
    }
    return NULL;
}
static void *consB(void *p) {
    my_tag = dm_test_thread_register((int)(intptr_t)p); uint32_t i = 0;
    for (;;) {
        mx_lock(&mB);
        while (qn == 0 && !stopB) if (cv_wait(&mB, &kNotEmpty, (i++ & 1) ? 300000 : -1) == R_TIMEOUT) toB++;
        if (qn == 0 && stopB) { mx_unlock(&mB); break; }
        uint64_t v = qB[qh]; qh = (qh + 1) % 8; qn--; consN++; consSum += v;
        cv_signal(&kNotFull, 1);
        mx_unlock(&mB); atomic_fetch_add(&progB, 1);
    }
    return NULL;
}
// C: an address generation word, 4 waiters (wait if equal; a third of the waits 500 µs, the rest none) and 1 signaller (increment + wake all, or SignalToAddress's increment-if-equal)
static _Atomic uint32_t genC; static _Atomic int stopC, leftC;static _Atomic uint64_t wokeC, toC, badC, sigC, progC;
static void *waitC(void *p) {
    my_tag = dm_test_thread_register((int)(intptr_t)p); uint32_t k = 0;
    while (!atomic_load(&stopC)) {
        uint32_t g = atomic_load(&genC);
        uint64_t r = svc(0x34, (uint64_t)(uintptr_t)&genC, 2, g, (k++ % 3 == 0) ? 500000 : (uint64_t)-1, NULL);
        if (r == 0) atomic_fetch_add(&wokeC, 1); else if (r == R_TIMEOUT) atomic_fetch_add(&toC, 1); else if (r == R_BAD_STATE) atomic_fetch_add(&badC, 1); else err("C: WaitForAddress failed");
        atomic_fetch_add(&progC, 1);
    }
    atomic_fetch_add(&leftC, 1);
    return NULL;
}
static void *sigCf(void *p) {
    my_tag = dm_test_thread_register((int)(intptr_t)p); uint32_t i = 0;
    while (!atomic_load(&g_stop)) {
        if (i++ & 1) { atomic_fetch_add(&genC, 1); svc(0x35, (uint64_t)(uintptr_t)&genC, 0, 0, (uint64_t)-1, NULL); }
        else { uint32_t v = atomic_load(&genC); if (svc(0x35, (uint64_t)(uintptr_t)&genC, 1, v, (uint64_t)-1, NULL) == R_BAD_STATE) continue; }
        atomic_fetch_add(&sigC, 1);
        if ((i & 15) == 0) usleep(20);
    }
    // The end: a waiter may have read the generation just before the last change and wait on it after, which only a later signal ends; signals every ms until all left
    atomic_store(&stopC, 1);
    while (atomic_load(&leftC) < 4) { atomic_fetch_add(&genC, 1); svc(0x35, (uint64_t)(uintptr_t)&genC, 0, 0, (uint64_t)-1, NULL); usleep(1000); }
    return NULL;
}
// E: a token passed between two threads through a lock and a condition variable (hand-overs per second, context switches per hand-over)
static _Atomic uint32_t mE, kE; static int turnE; static int nE = 20000;
static void *pingE(void *p) {
    int me = (int)(intptr_t)p; my_tag = dm_test_thread_register(44);
    for (int i = 0; i < nE; i++) { mx_lock(&mE); while (turnE != me) cv_wait(&mE, &kE, -1); turnE = !me; cv_signal(&kE, -1); mx_unlock(&mE); }
    return NULL;
}

// ---------- Deterministic parts ----------
static const int k_prio[6] = { 50, 30, 40, 30, 20, 40 };   // expected order: 4 (20), 1 and 3 (30, first come first), 2 and 5 (40), 0 (50)
static _Atomic uint32_t mD; static _Atomic uint32_t kD; static int ordD[6], nordD;
static void *lockD(void *p) {
    int id = (int)(intptr_t)p; my_tag = dm_test_thread_register(k_prio[id]);
    mx_lock(&mD); ordD[nordD++] = id; mx_unlock(&mD); return NULL;
}
static void *condD(void *p) {
    int id = (int)(intptr_t)p; my_tag = dm_test_thread_register(k_prio[id]);
    mx_lock(&mD); cv_wait(&mD, &kD, -1); ordD[nordD++] = id; mx_unlock(&mD); return NULL;
}
static _Atomic uint32_t wD3; static _Atomic int wokeD3; static uint64_t resD3[3];
static void *addrD(void *p) {
    int id = (int)(intptr_t)p; my_tag = dm_test_thread_register(44);
    resD3[id] = svc(0x34, (uint64_t)(uintptr_t)&wD3, 2, 5, (uint64_t)-1, NULL); atomic_fetch_add(&wokeD3, 1); return NULL;
}
static uint32_t hR, hW; static _Atomic uint32_t hThr; static uint64_t resD5;
static void *syncD(void *p) {
    (void)p; uint32_t h = dm_test_thread_register(44); my_tag = h; atomic_store(&hThr, h);
    uint64_t idx = 9; resD5 = svc(0x18, 0, (uint64_t)(uintptr_t)&hR, 1, (uint64_t)-1, &idx); if (resD5 == 0 && idx != 0) err("D5: wrong index");
    return NULL;
}
static void order_line(const char *what) {
    printf("= %s:", what); for (int i = 0; i < nordD; i++) printf(" %d", ordD[i]); printf("%s\n", nordD == 6 && ordD[0] == 4 && ordD[1] == 1 && ordD[2] == 3 && ordD[3] == 2 && ordD[4] == 5 && ordD[5] == 0 ? " (priority order)" : " NOT priority order");
    if (!(nordD == 6 && ordD[0] == 4 && ordD[1] == 1 && ordD[2] == 3 && ordD[3] == 2 && ordD[4] == 5 && ordD[5] == 0)) err("wake order");
}
static void deterministic(void) {
    pthread_t t[6];
    // D1: six threads queue on a held lock; unlocking hands it on in priority order (equal priority: first come)
    nordD = 0; mx_lock(&mD);
    for (int i = 0; i < 6; i++) { pthread_create(&t[i], NULL, lockD, (void *)(intptr_t)i); ms_sleep(30); }
    mx_unlock(&mD); for (int i = 0; i < 6; i++) pthread_join(t[i], NULL);
    order_line("D1 lock hand-over order"); printf("= D1 lock word at the end %#x\n", atomic_load(&mD));
    // D2: six threads wait on one condition variable; six single signals wake them in priority order, each taking the lock
    nordD = 0;
    for (int i = 0; i < 6; i++) { pthread_create(&t[i], NULL, condD, (void *)(intptr_t)i); ms_sleep(30); }
    for (int i = 0; i < 6; i++) { mx_lock(&mD); cv_signal(&kD, 1); mx_unlock(&mD); ms_sleep(30); }
    for (int i = 0; i < 6; i++) pthread_join(t[i], NULL);
    order_line("D2 condition signal order"); printf("= D2 condition key at the end %u, lock word %#x\n", atomic_load(&kD), atomic_load(&mD));
    // D3: SignalToAddress type 2 (modify by the waiting count): 3 waiters, wake 1 (value kept), wake all (value - 2), wake none waiting (value + 1); type 1 with a wrong value
    atomic_store(&wD3, 5);
    for (int i = 0; i < 3; i++) { pthread_create(&t[i], NULL, addrD, (void *)(intptr_t)i); } ms_sleep(50);
    uint64_t r1 = svc(0x35, (uint64_t)(uintptr_t)&wD3, 2, 5, 1, NULL); ms_sleep(50); int w1 = atomic_load(&wokeD3); uint32_t v1 = atomic_load(&wD3);
    uint64_t r2 = svc(0x35, (uint64_t)(uintptr_t)&wD3, 2, 5, (uint64_t)-1, NULL); for (int i = 0; i < 3; i++) pthread_join(t[i], NULL); uint32_t v2 = atomic_load(&wD3);
    uint64_t r3 = svc(0x35, (uint64_t)(uintptr_t)&wD3, 2, 3, (uint64_t)-1, NULL); uint32_t v3 = atomic_load(&wD3);
    uint64_t r4 = svc(0x35, (uint64_t)(uintptr_t)&wD3, 1, 99, (uint64_t)-1, NULL);
    printf("= D3 signal-modify: wake 1 of 3 → %#llx, %d woken, value %u; wake all of 2 → %#llx, value %u; none waiting → %#llx, value %u; increment-if-equal with a wrong value → %#llx; waiter results %#llx %#llx %#llx\n",
           (unsigned long long)r1, w1, v1, (unsigned long long)r2, v2, (unsigned long long)r3, v3, (unsigned long long)r4, (unsigned long long)resD3[0], (unsigned long long)resD3[1], (unsigned long long)resD3[2]);
    if (!(r1 == 0 && w1 == 1 && v1 == 5 && r2 == 0 && v2 == 3 && r3 == 0 && v3 == 4 && r4 == R_BAD_STATE)) err("D3: SignalToAddress values");
    // D4: timeouts: an address wait and a condition wait of 20 ms, nobody signalling
    static _Atomic uint32_t wD4, mD4, kD4; uint64_t t0 = mono_ns();
    uint64_t ra = svc(0x34, (uint64_t)(uintptr_t)&wD4, 2, 0, 20000000, NULL); uint64_t da = mono_ns() - t0;
    mx_lock(&mD4); t0 = mono_ns(); uint64_t rc = cv_wait(&mD4, &kD4, 20000000); uint64_t dc = mono_ns() - t0; uint32_t lk = atomic_load(&mD4) & ~WBIT; mx_unlock(&mD4);
    printf("= D4 timeouts: address wait → %#llx%s, condition wait → %#llx%s, lock held again after it %s, condition key %u\n", (unsigned long long)ra, da >= 19000000 ? " (after 20 ms)" : " (EARLY)",
           (unsigned long long)rc, dc >= 19000000 ? " (after 20 ms)" : " (EARLY)", lk == my_tag ? "yes" : "no", atomic_load(&kD4));
    if (ra != R_TIMEOUT || rc != R_TIMEOUT || da < 19000000 || dc < 19000000 || lk != my_tag) err("D4: timeouts");
    // D5: CancelSynchronization ends a WaitSynchronization; then SignalEvent ends the next one
    { uint64_t r[32] = {0}; svc_thread(0x45, r); hW = (uint32_t)r[1]; hR = (uint32_t)r[2]; }   // CreateEvent: write end, read end
    atomic_store(&hThr, 0); pthread_create(&t[0], NULL, syncD, NULL); while (!atomic_load(&hThr)) usleep(100); ms_sleep(30);
    uint64_t rcan = svc(0x19, atomic_load(&hThr), 0, 0, 0, NULL); pthread_join(t[0], NULL); uint64_t c1 = resD5;
    atomic_store(&hThr, 0); pthread_create(&t[0], NULL, syncD, NULL); while (!atomic_load(&hThr)) usleep(100); ms_sleep(30);
    uint64_t rsig = svc(0x11, hW, 0, 0, 0, NULL); pthread_join(t[0], NULL); uint64_t c2 = resD5;
    printf("= D5 CancelSynchronization → %#llx, the waiter got %#llx; SignalEvent → %#llx, the waiter got %#llx\n", (unsigned long long)rcan, (unsigned long long)c1, (unsigned long long)rsig, (unsigned long long)c2);
    if (rcan != 0 || c1 != R_CANCELLED || rsig != 0 || c2 != 0) err("D5: cancel / event");
}

static void *watchdog(void *p) {
    int secs = (int)(intptr_t)p; sleep((unsigned)secs);
    printf("FAILED: still running after %d s (a lost wakeup leaves a thread waiting for good)\n", secs); fflush(stdout); _exit(1);
}
int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *mode = getenv("DM_SVC_WAKE"); int secs = getenv("T_SECS") ? atoi(getenv("T_SECS")) : 10; if (secs < 1) secs = 1;
    my_tag = svc_thread_main_init();
    pthread_t wd; pthread_create(&wd, NULL, watchdog, (void *)(intptr_t)(secs + 60)); pthread_detach(wd);
    printf("DM_SVC_WAKE=%s, stress %d s\n", mode ? mode : "(unset: locked)", secs);
    deterministic();
    // Stress: A, B and C at once
    pthread_t ta[6], tp[3], tc[3], tw[4], ts; static const int pa[6] = { 30, 40, 44, 44, 50, 59 };
    for (int i = 0; i < 6; i++) pthread_create(&ta[i], NULL, thrA, (void *)(intptr_t)pa[i]);
    for (int i = 0; i < 3; i++) { pthread_create(&tp[i], NULL, prodB, (void *)(intptr_t)(40 + i * 8)); pthread_create(&tc[i], NULL, consB, (void *)(intptr_t)(36 + i * 8)); }
    for (int i = 0; i < 4; i++) pthread_create(&tw[i], NULL, waitC, (void *)(intptr_t)(30 + i * 10));
    pthread_create(&ts, NULL, sigCf, (void *)(intptr_t)44);
    uint64_t la = 0, lb = 0, lc = 0; int stall = 0;
    for (int s = 0; s < secs; s++) {
        sleep(1);
        uint64_t a = atomic_load(&opsA), b = atomic_load(&progB), c = atomic_load(&progC);
        if (a == la || b == lb || c == lc) { stall++; printf("ERROR: second %d: no progress in%s%s%s\n", s + 1, a == la ? " A" : "", b == lb ? " B" : "", c == lc ? " C" : ""); atomic_fetch_add(&g_err, 1); }
        la = a; lb = b; lc = c;
    }
    atomic_store(&g_stop, 1);
    for (int i = 0; i < 6; i++) pthread_join(ta[i], NULL);
    mx_lock(&mB); stopB = 1; cv_signal(&kNotFull, -1); cv_signal(&kNotEmpty, -1); mx_unlock(&mB);
    for (int i = 0; i < 3; i++) { pthread_join(tp[i], NULL); pthread_join(tc[i], NULL); }
    pthread_join(ts, NULL); for (int i = 0; i < 4; i++) pthread_join(tw[i], NULL);
    printf("  A: %llu lock hand-offs in %d s (%llu / s), counter %llu %s\n", (unsigned long long)opsA, secs, (unsigned long long)(opsA / (uint64_t)secs), (unsigned long long)cntA, cntA == opsA ? "= the operations" : "≠ THE OPERATIONS");
    if (cntA != opsA) err("A: counter");
    printf("  B: %llu items produced, %llu consumed (sums %s), %llu condition waits timed out\n", (unsigned long long)prodN, (unsigned long long)consN, prodSum == consSum ? "equal" : "DIFFER", (unsigned long long)toB);
    if (prodN != consN || prodSum != consSum || qn) err("B: queue");
    printf("  C: %llu signals, waiters woken %llu, timed out %llu, value already moved %llu; all 4 waiters left at the end\n", (unsigned long long)sigC, (unsigned long long)wokeC, (unsigned long long)toC, (unsigned long long)badC);
    printf("= stress invariants: A mutual exclusion and count %s, B every item consumed %s, C every waiter released %s, stalls %d\n", cntA == opsA ? "ok" : "BROKEN", prodN == consN && prodSum == consSum ? "ok" : "BROKEN", "ok", stall);
    // E: ping-pong
    struct rusage u0, u1; getrusage(RUSAGE_SELF, &u0); uint64_t e0 = mono_ns();
    pthread_t pe[2]; for (int i = 0; i < 2; i++) pthread_create(&pe[i], NULL, pingE, (void *)(intptr_t)i); for (int i = 0; i < 2; i++) pthread_join(pe[i], NULL);
    uint64_t de = mono_ns() - e0; getrusage(RUSAGE_SELF, &u1);
    long cs = (u1.ru_nvcsw - u0.ru_nvcsw) + (u1.ru_nivcsw - u0.ru_nivcsw);
    printf("  E: %d token hand-overs in %.1f ms: %.1f µs each, %.2f context switches each\n", 2 * nE, de / 1e6, de / 1e3 / (2.0 * nE), cs / (2.0 * nE));
    int e = atomic_load(&g_err);
    printf(e ? "FAILED: %d errors\n" : "all passed\n", e);
    fflush(stdout); _exit(e ? 1 : 0);
}
