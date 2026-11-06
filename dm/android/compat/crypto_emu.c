// Software ARMv8 Crypto Extension instructions for CPUs (or translators) without them: the Android Emulator's ARM-on-x86 translation (ndk_translation) traps
// AESE / AESD / AESMC / AESIMC, SHA1C / SHA1P / SHA1M / SHA1H / SHA1SU0 / SHA1SU1, SHA256H / SHA256H2 / SHA256SU0 / SHA256SU1 and PMULL{2}.1Q with SIGILL, and the
// Switch SDK's nn::crypto uses them directly (Legends: Arceus's sdk: SHA-1 at boot, AES and GCM later). crash.c's SIGILL handler calls dm_crypto_emu first: it decodes
// the instruction at pc, computes it on the vector registers of the signal context (ARM ARM pseudocode), steps pc, and goes on through the crypto instructions that follow
// so a run of them costs one signal. On a phone with the extension (every arm64 phone since 2016) nothing traps and this never runs.
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ucontext.h>

typedef unsigned __int128 u128;
static uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void w4(u128 v, uint32_t w[4]) { memcpy(w, &v, 16); }
static u128 v4(const uint32_t w[4]) { u128 v; memcpy(&v, w, 16); return v; }

// ---- AES (state bytes in vector order: byte i = row i % 4, column i / 4) ----
static uint8_t SB[256], ISB[256];
static void aes_tables(void) {
    if (SB[0]) return;
    uint8_t p = 1, q = 1;
    do { p = p ^ (uint8_t)(p << 1) ^ (p & 0x80 ? 0x1b : 0);
         q ^= q << 1; q ^= q << 2; q ^= q << 4; if (q & 0x80) q ^= 0x09;
         uint8_t x = q ^ (uint8_t)((q << 1) | (q >> 7)) ^ (uint8_t)((q << 2) | (q >> 6)) ^ (uint8_t)((q << 3) | (q >> 5)) ^ (uint8_t)((q << 4) | (q >> 4));
         SB[p] = x ^ 0x63; } while (p != 1);
    SB[0] = 0x63;
    for (int i = 0; i < 256; i++) ISB[SB[i]] = (uint8_t)i;
}
static uint8_t xt(uint8_t b) { return (uint8_t)(b << 1) ^ (b & 0x80 ? 0x1b : 0); }
static uint8_t gm(uint8_t a, uint8_t b) { uint8_t r = 0; while (b) { if (b & 1) r ^= a; a = xt(a); b >>= 1; } return r; }
static u128 aes_round(u128 v, int dec) {
    uint8_t s[16], o[16]; memcpy(s, &v, 16);
    for (int i = 0; i < 16; i++) { int r = i % 4, c = i / 4; o[i] = dec ? ISB[s[r + 4 * ((c - r + 4) % 4)]] : SB[s[r + 4 * ((c + r) % 4)]]; }
    memcpy(&v, o, 16); return v;
}
static u128 aes_mix(u128 v, int inv) {
    uint8_t s[16], o[16]; memcpy(s, &v, 16);
    for (int c = 0; c < 4; c++) { const uint8_t *a = s + 4 * c;
        for (int r = 0; r < 4; r++) o[4 * c + r] = inv ? gm(a[r], 14) ^ gm(a[(r + 1) % 4], 11) ^ gm(a[(r + 2) % 4], 13) ^ gm(a[(r + 3) % 4], 9)
                                                        : gm(a[r], 2) ^ gm(a[(r + 1) % 4], 3) ^ a[(r + 2) % 4] ^ a[(r + 3) % 4]; }
    memcpy(&v, o, 16); return v;
}

// ---- SHA-1 / SHA-256 ----
static uint32_t sha1f(int op, uint32_t x, uint32_t y, uint32_t z) {
    return op == 0 ? (((y ^ z) & x) ^ z) : op == 1 ? (x ^ y ^ z) : ((x & y) | ((x | y) & z));   // C: choose, P: parity, M: majority
}
static u128 sha1hash(int op, u128 vd, uint32_t y, u128 vm) {
    uint32_t x[4], w[4]; w4(vd, x); w4(vm, w);
    for (int e = 0; e < 4; e++) {
        uint32_t t = sha1f(op, x[1], x[2], x[3]);
        y = y + rol(x[0], 5) + t + w[e];
        x[1] = rol(x[1], 30);
        uint32_t ny = x[3]; x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = y; y = ny;
    }
    return v4(x);
}
static u128 sha256hash(u128 vx, u128 vy, u128 vw, int part1) {
    uint32_t x[4], y[4], w[4]; w4(vx, x); w4(vy, y); w4(vw, w);
    for (int e = 0; e < 4; e++) {
        uint32_t chs = ((y[1] ^ y[2]) & y[0]) ^ y[2], maj = (x[0] & x[1]) | ((x[0] | x[1]) & x[2]);
        uint32_t t = y[3] + (ror(y[0], 6) ^ ror(y[0], 11) ^ ror(y[0], 25)) + chs + w[e];
        x[3] = t + x[3];
        y[3] = t + (ror(x[0], 2) ^ ror(x[0], 13) ^ ror(x[0], 22)) + maj;
        uint32_t y3 = y[3], x3 = x[3];   // <Y,X> = ROL(Y:X, 32)
        y[3] = y[2]; y[2] = y[1]; y[1] = y[0]; y[0] = x3;
        x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = y3;
    }
    return part1 ? v4(x) : v4(y);
}
static u128 pmull64(uint64_t a, uint64_t b) { u128 r = 0; for (int i = 0; i < 64; i++) if (b >> i & 1) r ^= (u128)a << i; return r; }

// One instruction on the registers; 0 = not one of ours
static int emu1(uint32_t in, u128 *v) {
    int d = in & 31, n = in >> 5 & 31, m = in >> 16 & 31;
    switch (in & 0xFFFFFC00) {
        case 0x4E284800: aes_tables(); v[d] = aes_round(v[d] ^ v[n], 0); return 1;   // AESE
        case 0x4E285800: aes_tables(); v[d] = aes_round(v[d] ^ v[n], 1); return 1;   // AESD
        case 0x4E286800: v[d] = aes_mix(v[n], 0); return 1;                          // AESMC
        case 0x4E287800: v[d] = aes_mix(v[n], 1); return 1;                          // AESIMC
        case 0x5E280800: v[d] = rol((uint32_t)v[n], 30); return 1;                   // SHA1H
        case 0x5E281800: { uint32_t a[4], b[4], t[4], r[4]; w4(v[d], a); w4(v[n], b);   // SHA1SU1
            for (int i = 0; i < 4; i++) t[i] = a[i] ^ (i < 3 ? b[i + 1] : 0);
            for (int i = 0; i < 4; i++) r[i] = rol(t[i], 1);
            r[3] ^= rol(t[0], 2); v[d] = v4(r); return 1; }
        case 0x5E282800: { uint32_t a[4], b[4], t[4], r[4]; w4(v[d], a); w4(v[n], b);   // SHA256SU0
            t[0] = a[1]; t[1] = a[2]; t[2] = a[3]; t[3] = b[0];
            for (int i = 0; i < 4; i++) r[i] = (ror(t[i], 7) ^ ror(t[i], 18) ^ (t[i] >> 3)) + a[i];
            v[d] = v4(r); return 1; }
    }
    if ((in & 0xFFE08C00) == 0x5E000000) {   // the three-register SHA group
        switch (in >> 12 & 7) {
            case 0: case 1: case 2: v[d] = sha1hash(in >> 12 & 7, v[d], (uint32_t)v[n], v[m]); return 1;   // SHA1C / SHA1P / SHA1M
            case 3: { u128 r = (v[n] << 64) | (v[d] >> 64); v[d] = r ^ v[d] ^ v[m]; return 1; }               // SHA1SU0
            case 4: v[d] = sha256hash(v[d], v[n], v[m], 1); return 1;                                         // SHA256H
            case 5: v[d] = sha256hash(v[n], v[d], v[m], 0); return 1;                                         // SHA256H2
            case 6: { uint32_t a[4], b[4], c[4], t0[4], r[4]; w4(v[d], a); w4(v[n], b); w4(v[m], c);          // SHA256SU1
                t0[0] = b[1]; t0[1] = b[2]; t0[2] = b[3]; t0[3] = c[0];
                uint32_t t1[2] = { c[2], c[3] };
                for (int e = 0; e < 2; e++) r[e] = (ror(t1[e], 17) ^ ror(t1[e], 19) ^ (t1[e] >> 10)) + a[e] + t0[e];
                for (int e = 2; e < 4; e++) { uint32_t x = r[e - 2]; r[e] = (ror(x, 17) ^ ror(x, 19) ^ (x >> 10)) + a[e] + t0[e]; }
                v[d] = v4(r); return 1; }
        }
    }
    if ((in & 0xBFE0FC00) == 0x0EE0E000) {   // PMULL / PMULL2 Vd.1Q, Vn.1D / 2D, Vm.1D / 2D
        int hi = in >> 30 & 1; v[d] = pmull64((uint64_t)(v[n] >> (hi * 64)), (uint64_t)(v[m] >> (hi * 64))); return 1;
    }
    return 0;
}

int dm_crypto_emu(ucontext_t *uc) {
    struct fpsimd_context *fp = DM_UC_FP_(uc);
    if (!fp) return 0;
    uint64_t pc = DM_UC_SS(uc).__pc; int done = 0;
    static int logged;
    for (; done < 64; done++, pc += 4) {
        uint32_t in; memcpy(&in, (const void *)(uintptr_t)pc, 4);
        if (!emu1(in, (u128 *)fp->vregs)) break;
    }
    if (!done) return 0;
    if (!logged) { logged = 1; fprintf(stderr, "[dm] crypto: this CPU traps ARMv8 crypto instructions (an emulator's ARM translation?): computing them in software (first at %#llx, %08x)\n",
                                        (unsigned long long)DM_UC_SS(uc).__pc, *(const uint32_t *)(uintptr_t)DM_UC_SS(uc).__pc); }
    DM_UC_SS(uc).__pc = pc;
    return 1;
}

// ---- Load-time patching (the path the emulator needs: its ARM translation gives a SIGILL handler no vector registers) ----
// Before a module's code is sealed, every crypto instruction in [base, base + text_len) becomes a B to a stub of its own (one per site, in pages mapped right after the
// module so the B reaches). The stub saves x0-x18, x30, NZCV and q0-q31 on the guest stack, calls dm_crypto_call(q registers, instruction) - plain C over the saved
// registers - restores everything and branches back to the next instruction. Only when the CPU says it lacks the extension (AT_HWCAP) or DM_CRYPTO_PATCH=1.
#include <stdlib.h>
#include <sys/auxv.h>
#include <sys/mman.h>
void dm_crypto_call(u128 *q, uint32_t in) { if (!emu1(in, q)) fprintf(stderr, "[dm] [RED] crypto stub: undecoded %08x\n", in); }
static int is_crypto(uint32_t in) {
    switch (in & 0xFFFFFC00) { case 0x4E284800: case 0x4E285800: case 0x4E286800: case 0x4E287800: case 0x5E280800: case 0x5E281800: case 0x5E282800: return 1; }
    if ((in & 0xFFE08C00) == 0x5E000000 && (in >> 12 & 7) <= 6) return 1;
    return (in & 0xBFE0FC00) == 0x0EE0E000;
}
static int want_patch(void) {
    static int w = -1;
    if (w < 0) { const char *e = getenv("DM_CRYPTO_PATCH");
        unsigned long hw = getauxval(AT_HWCAP), need = (1ul << 3) | (1ul << 4) | (1ul << 5) | (1ul << 6);   // HWCAP_AES, PMULL, SHA1, SHA2
        w = e ? atoi(e) != 0 : (hw & need) != need;
        fprintf(stderr, "[dm] crypto: AT_HWCAP %#lx: %s\n", hw, w ? "the ARMv8 crypto instructions are patched to software stubs" : "the CPU has the crypto extension"); }
    return w;
}
enum { STUB_WORDS = 68, FRAME = 688, QOFF = 176 };
void dm_crypto_patch(const char *name, uint8_t *base, uint64_t text_len, uint64_t mod_size) {
    if (!want_patch()) return;
    uint32_t *code = (uint32_t *)base; uint64_t nw = text_len / 4, n = 0;
    for (uint64_t i = 0; i < nw; i++) if (is_crypto(code[i])) n++;
    if (!n) return;
    size_t len = ((n * STUB_WORDS * 4) + 0x3FFF) & ~(size_t)0x3FFF;
    // Stub pages within branch range of every site: right after the module, else right before it, else further out on either side (MAP_FIXED_NOREPLACE: never over a mapping)
    uint8_t *lo = base, *hi = base + text_len, *end = base + ((mod_size + 0x3FFF) & ~0x3FFFull); uint32_t *st = MAP_FAILED;
    for (int t = 0; t < 64 && st == MAP_FAILED; t++) {
        uint8_t *want = t % 2 == 0 ? end + (size_t)(t / 2) * 0x100000 : lo - len - (size_t)(t / 2) * 0x100000;
        if (want + len - lo > (128ll << 20) - 4 || hi - want > (128ll << 20) - 4) continue;
        void *r = mmap(want, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | 0x100000 /* MAP_FIXED_NOREPLACE */, -1, 0);
        if (r == MAP_FAILED) continue;
        if (r != (void *)want) { munmap(r, len); continue; }   // (a kernel without NOREPLACE treats it as a hint)
        st = r;
    }
    if (st == MAP_FAILED) { fprintf(stderr, "[dm] [RED] crypto: %s: no free pages within branch range for %llu stubs\n", name, (unsigned long long)n); return; }
    uint32_t *p = st; uint64_t done = 0;
    for (uint64_t i = 0; i < nw; i++) {
        uint32_t in = code[i]; if (!is_crypto(in)) continue;
        uint32_t *s = p; int k = 0;
        s[k++] = 0xD10003FF | (FRAME << 10);                                                     // sub sp, sp, #FRAME
        for (int r = 0; r < 18; r += 2) s[k++] = 0xA9000000 | ((r * 8 / 8) & 0x7F) << 15 | (r + 1) << 10 | 31 << 5 | r;   // stp xr, xr+1, [sp, #r*8]
        s[k++] = 0xA9000000 | (18 & 0x7F) << 15 | 30 << 10 | 31 << 5 | 18;                      // stp x18, x30, [sp, #144]
        s[k++] = 0xD53B4200;                                                                    // mrs x0, nzcv
        s[k++] = 0xF9000000 | (160 / 8) << 10 | 31 << 5 | 0;                                     // str x0, [sp, #160]
        for (int r = 0; r < 32; r += 2) s[k++] = 0xAD000000 | (((QOFF + r * 16) / 16) & 0x7F) << 15 | (r + 1) << 10 | 31 << 5 | r;   // stp qr, qr+1
        s[k++] = 0x910003E0 | (QOFF << 10);                                                      // add x0, sp, #QOFF
        s[k++] = 0x52800001 | (in & 0xFFFF) << 5;                                                // movz w1, #lo
        s[k++] = 0x72A00001 | (in >> 16) << 5;                                                   // movk w1, #hi, lsl 16
        int lit = STUB_WORDS - 2; s[k] = 0x58000010 | ((uint32_t)(lit - k) & 0x7FFFF) << 5; k++; // ldr x16, =dm_crypto_call
        s[k++] = 0xD63F0200;                                                                    // blr x16
        for (int r = 0; r < 32; r += 2) s[k++] = 0xAD400000 | (((QOFF + r * 16) / 16) & 0x7F) << 15 | (r + 1) << 10 | 31 << 5 | r;   // ldp qr, qr+1
        s[k++] = 0xF9400000 | (160 / 8) << 10 | 31 << 5 | 0;                                     // ldr x0, [sp, #160]
        s[k++] = 0xD51B4200;                                                                    // msr nzcv, x0
        for (int r = 0; r < 18; r += 2) s[k++] = 0xA9400000 | (r & 0x7F) << 15 | (r + 1) << 10 | 31 << 5 | r;   // ldp xr, xr+1
        s[k++] = 0xA9400000 | (18 & 0x7F) << 15 | 30 << 10 | 31 << 5 | 18;                      // ldp x18, x30
        s[k++] = 0x910003FF | (FRAME << 10);                                                     // add sp, sp, #FRAME
        int64_t back = ((int64_t)(uintptr_t)&code[i + 1] - (int64_t)(uintptr_t)&s[k]) / 4;
        s[k++] = 0x14000000 | ((uint32_t)back & 0x3FFFFFF);                                       // b site + 4
        while (k < lit) s[k++] = 0xD503201F;                                                    // nop
        uint64_t fn = (uint64_t)(uintptr_t)dm_crypto_call; memcpy(&s[lit], &fn, 8);
        int64_t to = ((int64_t)(uintptr_t)s - (int64_t)(uintptr_t)&code[i]) / 4;
        code[i] = 0x14000000 | ((uint32_t)to & 0x3FFFFFF);                                        // the site: b stub
        p += STUB_WORDS; done++;
    }
    __builtin___clear_cache((char *)st, (char *)st + len);
    if (mprotect(st, len, PROT_READ | PROT_EXEC)) fprintf(stderr, "[dm] [RED] crypto: %s: mprotect stubs failed\n", name);
    fprintf(stderr, "[dm] crypto: %s: %llu instructions patched to stubs at %p\n", name, (unsigned long long)done, (void *)st);
}
