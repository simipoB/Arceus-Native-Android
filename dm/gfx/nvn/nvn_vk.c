// Vulkan replay layer of the NVN null device: the interface of the Metal one (nvn_mtl.h, nvn_obj.h) on Vulkan, for Android (Adreno, Mali) and, through MoltenVK, the Mac
// (where it is checked frame by frame against the Metal replay). Built instead of nvn_mtl.m (dm/android/Makefile.host, dm/host/Makefile GFX=vulkan).
// Same model as nvn_mtl.m, written for correctness first:
//  - memory pools are GPU buffers over the game's own pages (zero copy): imported host memory where the driver offers it (VK_EXT_external_memory_host, MoltenVK), else the driver's
//    cached coherent memory moved onto the pool's address with mremap (Android: checked on Adreno 830 by the device probe, dm/android/probe);
//  - one command buffer per submission plus one for its texture uploads, submitted together; a timeline semaphore carries the fence values, a completion thread applies the
//    CPU-visible effects (events, counters) in order once the GPU is done;
//  - textures as in nvn_mtl.m: CPU-filled ones uploaded (deswizzled on the CPU) and re-uploaded when their pages change, render targets on the GPU, shared by storage and description;
//    every image stays in the GENERAL layout, and passes / copies are separated by full memory barriers;
//  - shaders are the translator's SPIR-V (<key>.spv) with the build's reflection (<key>.vkr, dm/android/tools/vk_meta.py); each stage's bindings move to their own range of one
//    push-descriptor set; dynamic rendering, dynamic state for everything the extended dynamic state covers.
#include <zlib.h>
#ifdef __ANDROID__
#include <malloc.h>
#endif
#define VK_NO_PROTOTYPES   // every Vulkan function comes from the driver opened at start (vk_dispatch.h, vk_open)
#include <vulkan/vulkan.h>
#ifdef __ANDROID__
#include <vulkan/vulkan_android.h>
#include <android/native_window.h>
#endif
#include <dlfcn.h>
#include "vk_dispatch.h"
#include "xf_learn.h"   // renderer stage 3: depth / stencil stores the next frame overwrites unread, learnt
#include <pthread.h>
#include <sys/resource.h>
#include <semaphore.h>
#include <sys/resource.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>   // (process_vm_readv / writev: the pool mapping test)
#include "nvn_mtl.h"
#include "nvn_obj.h"
#include "nvn_hmap.h"
#include "nvn_spv.h"
#include "nvn_ovl.h"
#include "nvn_dirty.h"
#include "../tex/nvn_fmt_vk.h"
#include "../tex/nvn_deswizzle.h"
#include "../tex/bc_decode.h"
#include "../tex/etc_encode.h"
#include "../tex/tex_cache.h"
#include "../tex/bc_gpu.h"
#include "../tex/b10_conv.h"   // R11G11B10F <-> RGBA16F texels (DM_VK_B10_AS_RGBA16)
#include "framecap_zip.h"
#include "dmsh_chain.h"
#include "game_profile.h"
#include "../../host/dm.h"
#include "golden_rec.h"   // golden-frame recordings (golden_rec.c; the backend side: golden_vk.inc at the end)

#define VLOG(...) do { fprintf(stderr, "[dm] [nvn-vk] " __VA_ARGS__); fputc('\n', stderr); } while (0)
// A fatal Vulkan error (VDIE) aborts, and SIGABRT is not one of the signals the crash handler takes (crash.c), so boot_state never said "crashed": the next start
//   neither went a safe-start level up nor offered another driver (#280: Mali-G68 r38p1, "vkWaitSemaphores failed: -4 (device lost?)" 12 s in, every start)
void dm_crash_note(int in_game) __attribute__((weak));
#define VDIE(...) do { VLOG("FATAL " __VA_ARGS__); if (dm_crash_note) dm_crash_note(0); abort(); } while (0)
static int g_red_once[176];
#define RED_ONCE(k, ...) do { if (!g_red_once[k]++) VLOG("[RED] " __VA_ARGS__); } while (0)
#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) VDIE("%s failed: %d (%s:%d)", #x, r_, __FILE__, __LINE__); } while (0)
#define NTEX 256

uint64_t guest_ns(void);
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// ---- Start-up marks (perf-1124 I8): one "[start]" line per stage with the milliseconds since the process started, so a log shows where the wait before play goes.
//   The S25+ showed its first frame 10 s after the process started with 1.6 s of it in the pipeline wait, and nothing said where the other 8 s went. Marks from
//   android_main.c (the host thread) and host/main.c (modules mapped, nnMain) come here too. The process start is field 22 of /proc/self/stat (clock ticks since boot,
//   CLOCK_BOOTTIME), as host/main.c's proc_start_mono_ns: CLOCK_MONOTONIC stops in deep sleep, so the age is taken on the boot clock
static uint64_t boot_t0_ns(void) {
    static _Atomic uint64_t t0; uint64_t v = atomic_load(&t0); if (v) return v;
    uint64_t mono = mono_ns(); v = mono;
#ifdef __linux__
    FILE *f = fopen("/proc/self/stat", "r"); char b[1024]; size_t n = f ? fread(b, 1, sizeof b - 1, f) : 0; if (f) fclose(f); b[n] = 0;
    char *p = strrchr(b, ')'); unsigned long long st = 0;
    if (p && sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %*u %*u %*d %*d %*d %*d %*d %*d %llu", &st) == 1) {
        struct timespec bt; clock_gettime(CLOCK_BOOTTIME, &bt); long tck = sysconf(_SC_CLK_TCK);
        int64_t age = (int64_t)bt.tv_sec * 1000000000ll + bt.tv_nsec - (int64_t)(st * (1000000000ull / (uint64_t)(tck > 0 ? tck : 100)));
        if (age > 0 && (uint64_t)age < mono) v = mono - (uint64_t)age;
    }
#endif
    uint64_t z = 0; if (!atomic_compare_exchange_strong(&t0, &z, v)) v = z;
    return v;
}
static double boot_ms(void) { return (mono_ns() - boot_t0_ns()) / 1e6; }
void dm_boot_mark(const char *what) { VLOG("[start] %s: %.0f ms since the process started", what, boot_ms()); }

// ================= Device =================
static VkInstance g_inst; static VkPhysicalDevice g_pd; static VkDevice g_dev; static VkQueue g_q; static uint32_t g_qf;
static VkPhysicalDeviceProperties g_props; static VkPhysicalDeviceMemoryProperties g_memp;
#include "pso_lru.h"     // which pipelines the cap destroys (pso_evict), least recently used first
#include "prewarm_policy.h"   // the prewarm's stamp, hold, list compaction and cap share (perf-1124 S3 / S4)
#include "vk_quirks.h"   // the unified driver layer's table: every per-GPU workaround's default, from one description of the GPU
static gpu_id g_gpuid = { .driver_id = -1 };
static int g_has_swapchain, g_host_import, g_clipctl, g_border, g_divisor, g_eds2, g_moltenvk;
static int g_mclamp;   // MIRROR_CLAMP_TO_EDGE samplers allowed (VK_KHR_sampler_mirror_clamp_to_edge, or Vulkan 1.2's samplerMirrorClampToEdge); else wrapm falls back
static int g_inst_xmemcaps;   // VK_KHR_external_memory_capabilities enabled on the instance
// Passes as dynamic rendering (1) or as Vulkan 1.0 render passes (0: rend_begin); draw state set per draw (1, extended dynamic state) or built into each pipeline (0: sst).
//   0 where the driver lacks them (Arm's r3x drivers for Mali-G57 / G68 / G77 / G78, Vulkan 1.1: A25 5G); DM_VK_NO_DYNREN=1 / DM_VK_NO_EDS=1 force 0 on any driver
static int g_dynren = 1, g_eds = 1;
static int g_bar_clean, g_after_pass, g_game_bar;   // the barrier state of the submission's command buffer (barrier_all)
static int32_t g_game_bits;   // the game's barrier bits since the last draw pass began (NVN_BARRIER_*: pass_begin narrows by them)
static int g_store_none;   // VK_ATTACHMENT_STORE_OP_NONE usable (Vulkan 1.3, VK_KHR / VK_EXT_load_store_op_none): skip stores' read-only depth / stencil
static int g_load_none;    // VK_ATTACHMENT_LOAD_OP_NONE usable (VK_KHR / VK_EXT_load_store_op_none enabled; DM_VK_LOAD_NONE=0: never): depth / stencil a held pass never touches
// Uniform updates deferred inside a pass (ubd_on, DM_VK_UB_DEFER=0 / 1), from the Arceus engine (androidswithy 84fb5df): on by default for Legends: Arceus, where it was
//   measured; off for Sword / Shield and Scarlet / Violet. The only switch that reads this now: that engine's relaxed barriers are vk_quirks.h's Q_BARRIER_RELAXED for
//   every game (DM_VK_BARRIER), and its learnt attachment stores (DM_VK_SKIP_STORES) are off for every game. A per-game choice that is no GPU rule, by the program ID
//   as before (docs/ENGINE_UNIFY.md: candidate for an engine_quirks word)
static int arceus_pass_default(const char *name, const char *off_value, const char *on_value) {
    const char *e = getenv(name);
    if (e && !strcmp(e, off_value)) return 0;
    if (e && !strcmp(e, on_value)) return 1;
#if DM_GAME_PROGRAM_ID == 0x01001F5010DFA000ull
    return 1;
#else
    return 0;
#endif
}
// Frame pacing (VK_GOOGLE_display_timing): each frame asks to be shown one frame interval (1 / the game's frame rate, a whole number of display refreshes) after the
//   last, so a 30 fps game shows every frame for the same 33.3 ms instead of whatever the timer and the display's refresh happen to line up to. It only asks for a time:
//   the present mode stays as chosen (mailbox), so nothing here waits on the compositor. DM_VK_PACE=0 turns it off
static int g_dtiming; static uint64_t g_refresh_ns;
static PFN_vkGetRefreshCycleDurationGOOGLE p_refresh; static PFN_vkGetPastPresentationTimingGOOGLE p_past;
static VkDeviceSize g_host_align = 4096;
static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;
// Linux / Android pools: the driver's memory exported as a file descriptor and mapped a second time over the pool's pages (g_fdmap 1; -1 until the first pool tries it,
// 0 when the driver has no such export or its mmap fails: then the mremap route below)
static int g_fdmap = 0;   // queue submissions come from the submit thread and the present thread
static VkExternalMemoryHandleTypeFlagBits g_fdtype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;   // the fd route's handle type (mtl_init: a dma-buf where the driver exports one)
// The heap route: a dma-buf from Android's own heap (/dev/dma_heap), mapped over the pool's pages and imported into the driver: for a driver that imports dma-bufs but
// exports none (Mali-G615 / G720 / G57: external memory features 0x4, its export failing; its pools mirrored, the game's pictures white: #47, #53, #55). 1 working,
// -1 until the first pool tries it, 0 off. DM_VK_DMAHEAP=1 tries it on any GPU, 0 never
static int g_heapmap = 0, g_heapfd = -1; static const char *g_heapname;
// the heap route's crash marker stays set until 600 presents (20 s at 30 fps) went by with it: a driver that crashed a few frames in (Mali-G615 MC2, #63: on the
//   third frame at every start) sends the next start back to moving the driver's mappings, not only one that crashed importing the first dma-buf
static int g_heap_pending;
static int heap_import_mark(int op);
void nvn_vk_heap_alive(void) { if (g_heap_pending) { g_heap_pending = 0; heap_import_mark(-1); } }   // the app went to the background: not a crash (android_main.c APP_CMD_PAUSE)
#ifdef __linux__
#include <sys/ioctl.h>
struct dm_heap_alloc { uint64_t len; uint32_t fd, fd_flags; uint64_t heap_flags; };
#define DM_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dm_heap_alloc)
static int heap_alloc(size_t n) {
    struct dm_heap_alloc d = { .len = n, .fd_flags = O_RDWR | O_CLOEXEC };
    return ioctl(g_heapfd, DM_HEAP_IOCTL_ALLOC, &d) < 0 ? -1 : (int)d.fd;
}
#endif
static const VkBufferUsageFlags BUF_ALL = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT
    | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
static PFN_vkCmdPushDescriptorSetKHR p_push;
static int g_nopush;   // no VK_KHR_push_descriptor (the emulator's gfxstream, some phones): each draw's bindings go into a descriptor set from the submission's pools
static uint32_t g_maxpush = 32;   // maxPushDescriptors: a program with more descriptors takes the descriptor-set path too (vprog np)
static int g_np;   // the program being bound uses descriptor sets (g_nopush, or over g_maxpush)
static PFN_vkGetMemoryHostPointerPropertiesEXT p_hostprops;
static PFN_vkGetImageSubresourceLayout2EXT p_layout2;   // VK_EXT_image_compression_control: whether an image's framebuffer compression (UBWC, AFBC) stayed on
static int g_compctl;   // VK_EXT_image_compression_control enabled with its feature: an image can be made with its compression off (DM_VK_B10_NOCOMPRESS)
// DM_VK_B10_NOCOMPRESS=1 (a Mali test, off by default; the in-game menu's Mali tests): R11G11B10F render targets keep their format but are made without framebuffer
//   compression (Arm's AFBC), through VK_EXT_image_compression_control where the driver has it, else by the rule the driver itself keys AFBC on (tex_new)
static int g_b10nc;

static int g_fmtlist;   // images may list their view formats (Vulkan 1.2, or VK_KHR_image_format_list): tex_make
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static void completion_start(void);
static void dummies_init(void);
static int host_import_mark(int set);   // (queue_submit)
static int fd_export_mark(int op);   // (mtl_init, mtl_pool_add)
static int mark_file(const char *name, const char *what, int op);
static int heap_import_mark(int op);   // the same for the dma-buf heap route (its own file: a crash of one route does not close the other)
static void pcache_open(void); static void *prewarm_main(void *arg); static void prewarm_start(void);   // pipeline list and cache (below)
static atomic_int g_pw_hold; static int prewarm_all(void); static void prewarm_wait(void);   // the game waiting for the prewarm (below)
// The Vulkan driver: DM_VK_DRIVER=<folder>/<library> (a driver package chosen on the setup screen: Mesa Turnip for Adreno, Mesa PanVK for Mali) is opened through
// libadrenotools (BSD-2-Clause, shipped in the app with its hook libraries); when that is missing or fails, the library itself (PanVK talks to the Mali kernel driver,
// kbase, on its own and needs none of the Adreno hooks); when that fails too, the system's libvulkan
static void *vk_open(void) {
#ifdef __ANDROID__
    const char *drv = getenv("DM_VK_DRIVER");
    if (drv && *drv) {
        char libdir[1024] = "", dir[1024]; Dl_info di;
        if (dladdr((void *)vk_open, &di) && di.dli_fname) { snprintf(libdir, sizeof libdir, "%s", di.dli_fname); char *e = strrchr(libdir, '/'); if (e) e[1] = 0; }   // the app's native library folder
        snprintf(dir, sizeof dir, "%s", drv); char *name = strrchr(dir, '/');
        void *at = dlopen("libadrenotools.so", RTLD_NOW);
        typedef void *(*open_fn)(int, int, const char *, const char *, const char *, const char *, const char *, void **);
        open_fn op = at ? (open_fn)dlsym(at, "adrenotools_open_libvulkan") : NULL;
        if (op && name) {
            char *lib = name + 1; *name = 0;   // folder / library name
            char ddir[1030]; snprintf(ddir, sizeof ddir, "%s/", dir);
            void *h = op(RTLD_NOW, 1 /* ADRENOTOOLS_DRIVER_CUSTOM */, NULL, libdir, ddir, lib, NULL, NULL);
            if (h) { VLOG("Vulkan driver %s (opened through libadrenotools)", drv); return h; }
        }
        void *h = dlopen(drv, RTLD_NOW | RTLD_LOCAL);
        if (h) { VLOG("Vulkan driver %s (opened directly; libadrenotools %s)", drv, op ? "failed" : "missing"); return h; }
        VLOG("[RED] Vulkan driver %s could not be opened (libadrenotools %s, directly: %s): the system driver is used", drv, op ? "failed" : "missing", dlerror());
    }
    return dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#else
    void *h = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL); if (!h) h = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL); if (!h) h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    return h;
#endif
}
#define VK_LOAD_G(f) f = (PFN_##f)vkGetInstanceProcAddr(NULL, #f);
#define VK_LOAD_I(f) f = (PFN_##f)vkGetInstanceProcAddr(g_inst, #f);
static int g_bcgpu;   // the BC layer is on (mtl_init): a mask of the paths that use it, 1 uploads, 2 buffer copies, 4 raw-block copies (7: all)
static int g_fctl_ext, g_drvprops_ext;   // a Vulkan 1.1 driver with VK_KHR_shader_float_controls / VK_KHR_driver_properties (enabled at device creation)
static int g_fctl;    // every shader asks for SignedZeroInfNanPreserve (spv_float_controls; mtl_init decides)
static int g_probe;   // DM_VK_PROBE (spv_probe's mask), 0 off
static void tf_selftest(void);   // tf_selftest.h (included with the capture passes)
static void tfp_decide(void); static int g_tf_pts = -1, g_vstores = 1; static VkShaderModule g_tfp_fs[4];   // captures as points (tf_points.h); the driver has vertex stores
enum { TFP_W = 256, TFP_H = 256 };   // the point targets: up to 65,536 vertices a capture
static int g_tfst_a = -1, g_tfst_b = -1, g_tfst_c = -1;   // the transform feedback self-test's counts (tf_selftest.h), -1 not run
static void vtx_canary(void);   // vtx_canary.h: per-draw vertex sizes checked on this driver at start
static int g_arm_rel, g_arm_p;   // Arm's own Mali driver: its release, r<rel>p<p> (driverVersion is VK_MAKE_VERSION(rel, p, 0): 0xb001000 = r44p1); 0 for any other driver
static int g_arm_beta;           // ... a beta build of it (driverInfo "v1.r49p1-03bet0": the G925's)
static int g_drv_id = -1;        // the driver's ID (VkDriverId) where it reports driver properties, else -1 (mtl_init; the white-screen ladder's key)
// The Mali white-screen ladder and its detector (mali_white_vk.inc, included after the draw counters; its rules: mali_white.h): the route at mtl_init, the lit target
//   and the part drawn of it from pass_begin / draw_ / pass_end, the samples at mtl_present (with the picture shown), the state in the frame capture's notes
struct vtex; static struct vtex *g_mw_px;   // the open pass's lit-target candidate (mw_pass_target), NULL: none
static void mw_init(int *a_on, int *b_on, const char **why_a, const char **why_b); static void mw_pass_target(struct vtex *x, uint32_t lv); static void mw_pass_end(void);
static void mw_draw_rect(float vx, float vy, float vw, float vh, const VkRect2D *sc); static void mw_present(uint64_t n, const tex_t *shown, const int crop[4]); static void mw_capture_note(void);
static int g_clampzero; static _Atomic int g_clampzero_n;   // shaders get spv_clamp_zero (Arm erratum 3779215: mtl_init decides); how many it changed
static int g_fmz; static _Atomic int g_fmz_n;   // shaders get spv_fmz (every multiply by 0 gives 0, as Maxwell's FMZ: DM_VK_FMZ=1, a test); how many it changed
static unsigned g_exact; static _Atomic int g_exact_n;
static int g_qlod; static _Atomic int g_qlod_n;   // shaders get spv_query_lod (textureQueryLod's level clamped: mtl_init decides)
static int g_initloc; static _Atomic int g_initloc_n;   // shaders get spv_init_locals (the translator's registers start at zero, as Mesa's compilers treat undefined values: mtl_init decides)   // shaders get spv_exact_math (Maxwell's log2 / exp2 / inversesqrt / sqrt / F2I at their edge cases: mtl_init decides)
static int g_nanmm; static _Atomic int g_nanmm_n;           // shaders get spv_nan_minmax (min / max / saturate drop NaN as on Maxwell: mtl_init decides; 2: spv_nan_select too); how many it changed
static int g_dualsrc = 1;        // the device blends with a second shader output (dualSrcBlend): Arm's own Mali driver has none (r44p1)
// Optional Vulkan 1.0 features (any of them may be missing on a phone): what the device has (mtl_init); without one, the state that needs it is left out or replaced
//   (pso_build, samp_vk, the draw), never passed to the driver. DM_VK_MIN_FEATURES=1 takes every such route on any GPU (a test of the Vulkan 1.0 baseline)
static int g_drawp = 1;          // draw parameters (VK_KHR_shader_draw_parameters / shaderDrawParameters): without, spv_no_drawparams + dp_push
static int g_nosbsc;             // Vulkan 1.0 without VK_KHR_storage_buffer_storage_class: spv_sb_to_uniform before spv_to_10
static int g_f_clamp = 1, g_f_logic = 1, g_f_nonsolid = 1, g_f_indblend = 1, g_f_biasclamp = 1, g_f_aniso = 1, g_minmax = 1;
static uint32_t g_feat_have = ~0u;   // SPV_NEED_* the device has (spv_feature_needs)
static int g_tlsem = 1;          // the device has timeline semaphores; 0: each submission signals a fence of its own instead (older Adreno stock drivers, Vulkan 1.1, #99)
// Markers of a set-up step that crashed the driver (mark_file below), one set per driver kind: a crash on a driver package does not change the phone's own driver
static const char *drv_mark(const char *base) {
    static char m[4][64]; static int k; char *o = m[k++ & 3]; const char *d = getenv("DM_VK_DRIVER");
    snprintf(o, sizeof m[0], "%s%s", base, d && *d ? "_package" : ""); return o;
}
static void vk_logf(const char *f, ...) { va_list a; va_start(a, f); fputs("[dm] [nvn-vk] ", stderr); vfprintf(stderr, f, a); fputc('\n', stderr); va_end(a); }
static uint32_t g_instance_api = VK_API_VERSION_1_3;   // the instance's apiVersion (mtl_init)
static int g_yflip;   // Vulkan 1.0 without VK_KHR_maintenance1 (or DM_VK_YFLIP=1): NVN's upper-left origin made in the vertex shaders (spv_flip_y), viewports kept positive
static uint32_t g_api;   // the device's Vulkan version (DM_VK_API=11 / 12 tries an older device's path on a newer one)
static PFN_vkVoidFunction vk_dev_fn(const char *name) {   // a device function; before 1.3 the KHR / EXT name of a promoted one (dynamic rendering, timeline semaphores, extended dynamic state)
    char alt[128];
    if (g_api < VK_API_VERSION_1_3) for (int k = 0; k < 2; k++) { snprintf(alt, sizeof alt, "%s%s", name, k ? "EXT" : "KHR"); PFN_vkVoidFunction f = vkGetDeviceProcAddr(g_dev, alt); if (f) return f; }
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(g_dev, name); if (f) return f;
    for (int k = 0; k < 2; k++) { snprintf(alt, sizeof alt, "%s%s", name, k ? "EXT" : "KHR"); if ((f = vkGetDeviceProcAddr(g_dev, alt))) return f; }   // (a core name the driver withholds: its extension name)
    return NULL;
}
#define VK_LOAD_D(f) f = (PFN_##f)vk_dev_fn(#f);
// Renderer phase 3: the held frame's open gap (seg_gap_begin) and how many commands that do work went into it. Each such command goes through a wrapper that counts
//   it there and calls the driver's; state and bindings (pipelines, descriptor sets, push constants, dynamic state) are not work (a pass's secondary sets its own)
static VkCommandBuffer g_gap_cb; static uint32_t g_gap_work;
#define GAPW(name, params, args) static PFN_##name r_##name; static VKAPI_ATTR void VKAPI_CALL w_##name params { if (cb == g_gap_cb) g_gap_work++; r_##name args; }
GAPW(vkCmdPipelineBarrier, (VkCommandBuffer cb, VkPipelineStageFlags a, VkPipelineStageFlags b, VkDependencyFlags c, uint32_t d, const VkMemoryBarrier *e, uint32_t f, const VkBufferMemoryBarrier *g, uint32_t h, const VkImageMemoryBarrier *i), (cb, a, b, c, d, e, f, g, h, i))
GAPW(vkCmdCopyBuffer, (VkCommandBuffer cb, VkBuffer a, VkBuffer b, uint32_t c, const VkBufferCopy *d), (cb, a, b, c, d))
GAPW(vkCmdCopyBufferToImage, (VkCommandBuffer cb, VkBuffer a, VkImage b, VkImageLayout c, uint32_t d, const VkBufferImageCopy *e), (cb, a, b, c, d, e))
GAPW(vkCmdCopyImage, (VkCommandBuffer cb, VkImage a, VkImageLayout b, VkImage c, VkImageLayout d, uint32_t e, const VkImageCopy *f), (cb, a, b, c, d, e, f))
GAPW(vkCmdCopyImageToBuffer, (VkCommandBuffer cb, VkImage a, VkImageLayout b, VkBuffer c, uint32_t d, const VkBufferImageCopy *e), (cb, a, b, c, d, e))
GAPW(vkCmdBlitImage, (VkCommandBuffer cb, VkImage a, VkImageLayout b, VkImage c, VkImageLayout d, uint32_t e, const VkImageBlit *f, VkFilter g), (cb, a, b, c, d, e, f, g))
GAPW(vkCmdResolveImage, (VkCommandBuffer cb, VkImage a, VkImageLayout b, VkImage c, VkImageLayout d, uint32_t e, const VkImageResolve *f), (cb, a, b, c, d, e, f))
GAPW(vkCmdFillBuffer, (VkCommandBuffer cb, VkBuffer a, VkDeviceSize b, VkDeviceSize c, uint32_t d), (cb, a, b, c, d))
GAPW(vkCmdClearColorImage, (VkCommandBuffer cb, VkImage a, VkImageLayout b, const VkClearColorValue *c, uint32_t d, const VkImageSubresourceRange *e), (cb, a, b, c, d, e))
GAPW(vkCmdClearDepthStencilImage, (VkCommandBuffer cb, VkImage a, VkImageLayout b, const VkClearDepthStencilValue *c, uint32_t d, const VkImageSubresourceRange *e), (cb, a, b, c, d, e))
GAPW(vkCmdClearAttachments, (VkCommandBuffer cb, uint32_t a, const VkClearAttachment *b, uint32_t c, const VkClearRect *d), (cb, a, b, c, d))
GAPW(vkCmdDispatch, (VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c), (cb, a, b, c))
GAPW(vkCmdDispatchIndirect, (VkCommandBuffer cb, VkBuffer a, VkDeviceSize b), (cb, a, b))
GAPW(vkCmdDraw, (VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d), (cb, a, b, c, d))
GAPW(vkCmdDrawIndexed, (VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e), (cb, a, b, c, d, e))
GAPW(vkCmdDrawIndirect, (VkCommandBuffer cb, VkBuffer a, VkDeviceSize b, uint32_t c, uint32_t d), (cb, a, b, c, d))
GAPW(vkCmdDrawIndexedIndirect, (VkCommandBuffer cb, VkBuffer a, VkDeviceSize b, uint32_t c, uint32_t d), (cb, a, b, c, d))
GAPW(vkCmdResetQueryPool, (VkCommandBuffer cb, VkQueryPool a, uint32_t b, uint32_t c), (cb, a, b, c))
GAPW(vkCmdWriteTimestamp, (VkCommandBuffer cb, VkPipelineStageFlagBits a, VkQueryPool b, uint32_t c), (cb, a, b, c))
GAPW(vkCmdBeginRendering, (VkCommandBuffer cb, const VkRenderingInfo *a), (cb, a))
GAPW(vkCmdEndRendering, (VkCommandBuffer cb), (cb))
GAPW(vkCmdBeginRenderPass, (VkCommandBuffer cb, const VkRenderPassBeginInfo *a, VkSubpassContents b), (cb, a, b))
GAPW(vkCmdEndRenderPass, (VkCommandBuffer cb), (cb))
GAPW(vkCmdExecuteCommands, (VkCommandBuffer cb, uint32_t a, const VkCommandBuffer *b), (cb, a, b))
#define GAPW_SET(name) if (name) { r_##name = name; name = w_##name; }
static void gap_wrap(void) {
    GAPW_SET(vkCmdPipelineBarrier) GAPW_SET(vkCmdCopyBuffer) GAPW_SET(vkCmdCopyBufferToImage) GAPW_SET(vkCmdCopyImage) GAPW_SET(vkCmdCopyImageToBuffer) GAPW_SET(vkCmdBlitImage)
    GAPW_SET(vkCmdResolveImage) GAPW_SET(vkCmdFillBuffer) GAPW_SET(vkCmdClearColorImage) GAPW_SET(vkCmdClearDepthStencilImage) GAPW_SET(vkCmdClearAttachments) GAPW_SET(vkCmdDispatch)
    GAPW_SET(vkCmdDispatchIndirect) GAPW_SET(vkCmdDraw) GAPW_SET(vkCmdDrawIndexed) GAPW_SET(vkCmdDrawIndirect) GAPW_SET(vkCmdDrawIndexedIndirect) GAPW_SET(vkCmdResetQueryPool)
    GAPW_SET(vkCmdWriteTimestamp) GAPW_SET(vkCmdBeginRendering) GAPW_SET(vkCmdEndRendering) GAPW_SET(vkCmdBeginRenderPass) GAPW_SET(vkCmdEndRenderPass) GAPW_SET(vkCmdExecuteCommands)
}
// Vulkan 1.0 drivers take SPIR-V 1.0 only: every shader module is made from a 1.0 copy (nvn_spv.c spv_to_10) where the shader allows it; one that really needs more
//   is passed as it is (the driver may refuse it) and named once in the log
static PFN_vkCreateShaderModule g_csm_real;
static _Atomic int g_csm10_n, g_csm10_fail;
static VKAPI_ATTR VkResult VKAPI_CALL vk_shader_module_10(VkDevice d, const VkShaderModuleCreateInfo *ci, const VkAllocationCallbacks *a, VkShaderModule *m) {
    size_t nw = 0; uint32_t *sb = NULL;
    if (g_nosbsc && ci->codeSize >= 20) { sb = malloc(ci->codeSize); if (sb) { memcpy(sb, ci->pCode, ci->codeSize); if (spv_sb_to_uniform(sb, ci->codeSize / 4) < 0) RED_ONCE(128, "Vulkan 1.0: a shader's storage buffers could not be declared the 1.0 way"); } }
    uint32_t *c = spv_to_10(sb ? sb : ci->pCode, ci->codeSize / 4, &nw);
    if (!c) { if (ci->codeSize >= 8 && ci->pCode[1] > 0x00010000u && !atomic_fetch_add(&g_csm10_fail, 1)) VLOG("[RED] Vulkan 1.0: a shader needs more than SPIR-V 1.0 (a subgroup instruction): passed as it is");
              VkShaderModuleCreateInfo c1 = *ci; if (sb) c1.pCode = sb; VkResult r1 = g_csm_real(d, &c1, a, m); free(sb); return r1; }
    free(sb);
    VkShaderModuleCreateInfo c2 = *ci; c2.codeSize = nw * 4; c2.pCode = c;
    VkResult r = g_csm_real(d, &c2, a, m); free(c);
    if (atomic_fetch_add(&g_csm10_n, 1) == 0) VLOG("Vulkan 1.0: shaders made SPIR-V 1.0 as they load (first: %s)", r ? "refused" : "ok");
    return r;
}
const char *nvn_vk_gpu_name(void) { return g_dev ? g_props.deviceName : NULL; }   // android_main.c boot_guard
// The quirk table for code outside this file (hle_nvn.c rq_on: Faster rendering's Auto): this GPU's default, -1 before the device exists (as nvn_vk_gpu_name; gpu_id's
//   name, API, vendor and the profile's port / eng are set before the device is made), and -1 for a rule that reads the driver ID or Arm's release (vk_quirks.h
//   quirk_late) until mtl_init has read them, just after (g_gpuid_done); and the start's [quirk] note
static _Atomic int g_gpuid_done;
int nvn_vk_quirk_default(int q) { return !g_dev || (quirk_late(q) && !atomic_load(&g_gpuid_done)) ? -1 : quirk_default(q, &g_gpuid); }
int nvn_vk_quirk_note(int q, int value, int from_env) { return quirk_note(q, value, from_env); }
// mtl_init once for every caller (perf-1124 S5): it used to test g_dev, which is set about 200 lines before the set-up ends, so a second thread arriving meanwhile
//   went on with half a device. With the early start (nvn_vk_early_init, below) a helper thread and the game's first memory pool can both ask: the second waits
//   for the first's set-up to end. The hot callers (every submission, every pool) pay one acquire load as before
static void mtl_init_(void);
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER; static atomic_int g_init_done;
void mtl_init(void) {
    if (atomic_load_explicit(&g_init_done, memory_order_acquire)) return;
    pthread_mutex_lock(&g_init_mu);
    if (!atomic_load_explicit(&g_init_done, memory_order_relaxed)) { mtl_init_(); atomic_store_explicit(&g_init_done, 1, memory_order_release); }
    pthread_mutex_unlock(&g_init_mu);
}
static void mtl_init_(void) {
    if (g_dev) return;
    dm_boot_mark("graphics set-up begins (mtl_init)");
    if (!vkGetInstanceProcAddr) {
        void *lib = vk_open(); if (!lib) VDIE("no Vulkan driver could be opened");
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
        if (!vkGetInstanceProcAddr) vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");   // a Mesa driver opened directly: its loader entry point
        if (!vkGetInstanceProcAddr) VDIE("the Vulkan driver has no vkGetInstanceProcAddr");
        VK_GLOBAL(VK_LOAD_G)
    }
    uint32_t instance_api = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_instance_version = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    if (enumerate_instance_version && enumerate_instance_version(&instance_api) != VK_SUCCESS) instance_api = VK_API_VERSION_1_0;
    // A 1.1+ loader takes any apiVersion: 1.3 is asked for, so the device's own version decides. Android's loader answers 1.1 on older Android releases, and an instance
    //   made at 1.1 made Turnip (strict) give out no 1.2 / 1.3 function: vkWaitSemaphores came back NULL and the first wait jumped to address 0 (#58, Adreno 640,
    //   every version from 0.36); Qualcomm's own driver hands them out anyway
    if (instance_api >= VK_API_VERSION_1_1) instance_api = VK_API_VERSION_1_3;
    g_instance_api = instance_api;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "nxnative", .apiVersion = instance_api };
    uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties *ie = calloc(nie ? nie : 1, sizeof *ie); vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
    const char *iext[8]; uint32_t niext = 0; VkInstanceCreateFlags ifl = 0;
    if (has_ext(ie, nie, "VK_KHR_portability_enumeration")) { iext[niext++] = "VK_KHR_portability_enumeration"; ifl |= 0x00000001; /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */ }
    if (has_ext(ie, nie, "VK_KHR_get_physical_device_properties2")) iext[niext++] = "VK_KHR_get_physical_device_properties2";
    // (a Vulkan 1.0 GPU: vkGetPhysicalDeviceExternalBufferProperties is this extension's; the core name of a 1.3 instance is not allowed on a 1.0 device)
    if ((g_inst_xmemcaps = has_ext(ie, nie, "VK_KHR_external_memory_capabilities"))) iext[niext++] = "VK_KHR_external_memory_capabilities";
    if (has_ext(ie, nie, "VK_KHR_surface")) iext[niext++] = "VK_KHR_surface";
#ifdef __ANDROID__
    if (has_ext(ie, nie, "VK_KHR_android_surface")) iext[niext++] = "VK_KHR_android_surface";
#endif
    free(ie);
    const char *layers[1]; uint32_t nlayers = 0;
    if (getenv("DM_VK_VALIDATION")) layers[nlayers++] = "VK_LAYER_KHRONOS_validation";
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .flags = ifl, .pApplicationInfo = &ai, .enabledExtensionCount = niext, .ppEnabledExtensionNames = iext,
                                 .enabledLayerCount = nlayers, .ppEnabledLayerNames = layers };
    VkResult ir = vkCreateInstance(&ici, NULL, &g_inst); uint32_t npd = 0;
    if (ir == VK_SUCCESS) { VK_INSTANCE(VK_LOAD_I)
        // a Vulkan 1.0 instance has these only under their extension names (VK_KHR_get_physical_device_properties2, enabled above when listed)
        if (!vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceProperties2KHR");
        if (!vkGetPhysicalDeviceFeatures2) vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceFeatures2KHR");
        if (!vkGetPhysicalDeviceExternalBufferProperties) vkGetPhysicalDeviceExternalBufferProperties = (PFN_vkGetPhysicalDeviceExternalBufferProperties)vkGetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
        if (getenv("DM_VK_NO_F2") && atoi(getenv("DM_VK_NO_F2")) > 0) { vkGetPhysicalDeviceProperties2 = NULL; vkGetPhysicalDeviceFeatures2 = NULL; vkGetPhysicalDeviceExternalBufferProperties = NULL; }   // (a test: a driver with only Vulkan 1.0's own calls)
        vkEnumeratePhysicalDevices(g_inst, &npd, NULL); }
    if (!npd) {   // a driver package for another GPU (Turnip, Adreno only, chosen on a Mali-G615: "no Vulkan device" at every start, #71): this start goes on with the phone's own driver
        const char *drv = getenv("DM_VK_DRIVER");
        if (drv && *drv) {
            VLOG("[RED] the chosen Vulkan driver %s found no GPU on this phone (instance %d): the phone's own driver is used (Settings: graphics driver)", drv, (int)ir);
            if (ir == VK_SUCCESS) { PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)vkGetInstanceProcAddr(g_inst, "vkDestroyInstance"); if (di) di(g_inst, NULL); }
            g_inst = VK_NULL_HANDLE; unsetenv("DM_VK_DRIVER"); vkGetInstanceProcAddr = NULL; mtl_init_(); return;
        }
        if (ir != VK_SUCCESS) VDIE("vkCreateInstance failed: %d", (int)ir);
        VDIE("no Vulkan device");
    }
    VkPhysicalDevice pds[8]; if (npd > 8) npd = 8; vkEnumeratePhysicalDevices(g_inst, &npd, pds); g_pd = pds[0];
    {   // DM_VK_DEVICE=<index> or <part of the name>: the device to use where the loader lists several (PC tools: lavapipe beside a real GPU, a profile layer's device)
        const char *dv = getenv("DM_VK_DEVICE");
        if (dv && *dv) { char *x; long k = strtol(dv, &x, 10); int found = 0;
            if (!*x && k >= 0 && k < (long)npd) { g_pd = pds[k]; found = 1; }
            else for (uint32_t i = 0; i < npd && !found; i++) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pds[i], &pp); if (strcasestr(pp.deviceName, dv)) { g_pd = pds[i]; found = 1; } }
            if (!found) VLOG("[RED] DM_VK_DEVICE=%s: no such device among %u: the first one", dv, npd); }
    }
    vkGetPhysicalDeviceProperties(g_pd, &g_props); vkGetPhysicalDeviceMemoryProperties(g_pd, &g_memp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    for (g_qf = 0; g_qf < nq && (qp[g_qf].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT); g_qf++) {}
    if (g_qf == nq) VDIE("no graphics + compute queue");
    uint32_t nde = 0; vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nde, NULL);
    VkExtensionProperties *de = calloc(nde ? nde : 1, sizeof *de); vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nde, de);
    const char *dext[48]; uint32_t ndext = 0;
    g_api = g_props.apiVersion < g_instance_api ? g_props.apiVersion : g_instance_api;   // (the version the driver hands out functions for: the lower of the two)
    if (getenv("DM_VK_API")) { int a = atoi(getenv("DM_VK_API")); uint32_t f = a == 10 ? VK_API_VERSION_1_0 : a == 11 ? VK_API_VERSION_1_1 : a == 12 ? VK_API_VERSION_1_2 : g_api; if (f < g_api) g_api = f; }
    int old = g_api < VK_API_VERSION_1_3, old11 = g_api < VK_API_VERSION_1_2;
    int x_dynren = 0, x_eds = 0;   // (old drivers: the extension is listed and enabled; its feature structure goes into the chain below only then)
    if (old) {   // what Vulkan 1.3 has built in, as extensions (Vulkan 1.1 / 1.2 drivers: older Mali, PowerVR, Adreno 6xx stock drivers)
        // Timeline semaphores are required. Dynamic rendering and extended dynamic state are used where listed; without them passes are render passes (rend_begin)
        //   and the draw state goes into the pipelines (sst): Arm's r32 driver of a Mali-G68 (Galaxy A25 5G) lists neither, and the game stopped here before
        static const char *deps11[] = { "VK_KHR_depth_stencil_resolve", "VK_KHR_create_renderpass2", "VK_KHR_multiview", "VK_KHR_maintenance2" };   // dynamic rendering's prerequisites on 1.1
        // without timeline semaphores (a Vulkan 1.1 Adreno 6xx stock driver: #99, Adreno 613) every submission signals a fence of its own instead (g_tlsem)
        if (has_ext(de, nde, "VK_KHR_timeline_semaphore")) dext[ndext++] = "VK_KHR_timeline_semaphore";
        else { g_tlsem = 0; VLOG("Vulkan %u.%u device without timeline semaphores: a fence for each submission instead", VK_API_VERSION_MAJOR(g_api), VK_API_VERSION_MINOR(g_api)); }
        // before 1.2 its prerequisites must all be there: depth_stencil_resolve and create_renderpass2, and on a 1.0 device multiview and maintenance2 as well
        //   (core in 1.1). One missing: render passes (simdev: Turnip on a Vulkan 1.0 Adreno 610 lists create_renderpass2 without multiview, VUID-...-01387)
        x_dynren = has_ext(de, nde, "VK_KHR_dynamic_rendering");
        if (x_dynren && old11) for (unsigned k = 0; k < sizeof deps11 / sizeof *deps11; k++)
            if ((k < 2 || g_api < VK_API_VERSION_1_1) && !has_ext(de, nde, deps11[k])) { VLOG("dynamic rendering listed without %s: render passes", deps11[k]); x_dynren = 0; break; }
        if (x_dynren) {
            dext[ndext++] = "VK_KHR_dynamic_rendering";
            if (old11) for (unsigned k = 0; k < sizeof deps11 / sizeof *deps11; k++) if (has_ext(de, nde, deps11[k])) dext[ndext++] = deps11[k];
        }
        if ((x_eds = has_ext(de, nde, "VK_EXT_extended_dynamic_state"))) dext[ndext++] = "VK_EXT_extended_dynamic_state";
        if (g_tlsem || x_dynren || x_eds)   // (only what it has: the line said "timeline semaphores" on an Adreno 613 driver without them, #99)
            VLOG("Vulkan %u.%u device: %s%s%s through their extensions", VK_API_VERSION_MAJOR(g_api), VK_API_VERSION_MINOR(g_api), g_tlsem ? "timeline semaphores" : "",
                 x_dynren ? (g_tlsem ? ", dynamic rendering" : "dynamic rendering") : "", x_eds ? (g_tlsem || x_dynren ? ", extended dynamic state" : "extended dynamic state") : "");
        // float controls and the driver's identity (the Mali white fix, spv_float_controls) are extensions on a Vulkan 1.1 driver: older Mali-G52 / G72 / G76 phones
        if (old11) { if (has_ext(de, nde, "VK_KHR_shader_float_controls")) { dext[ndext++] = "VK_KHR_shader_float_controls"; g_fctl_ext = 1; }
                     if (has_ext(de, nde, "VK_KHR_driver_properties")) { dext[ndext++] = "VK_KHR_driver_properties"; g_drvprops_ext = 1; } }
        // an image's list of view formats (tex_make) is Vulkan 1.2, or VK_KHR_image_format_list on 1.1: without it the list is not given at all. Before, a Vulkan 1.1
        //   driver got the list unannounced: the Mali pass test crashed on Arm's r32p1 (Mali-G57 MC2, Vulkan 1.1.177) at the first render pass (#158)
        if (old11 && has_ext(de, nde, "VK_KHR_image_format_list")) { dext[ndext++] = "VK_KHR_image_format_list"; g_fmtlist = 1; }
    }
    if (!old11) g_fmtlist = 1;   // (Vulkan 1.2 and newer: built in)
    // DM_VK_AS_MALI=1: this GPU takes the routes a Mali-G615 / G720 takes (no BC textures: transcoded on upload; no push descriptors, depth clip control, vertex divisor
    //   or host import; pools on dma-bufs from Android's heap), so what goes wrong there shows on a phone at hand (#47, #53, #55). Each still has its own DM_VK_ switch
    static int as_mali = -1; if (as_mali < 0) { const char *e = getenv("DM_VK_AS_MALI"); as_mali = e && atoi(e); }
    if (as_mali) { setenv("DM_VK_NO_PUSH", "1", 0); setenv("DM_VK_NO_BC", "1", 0); setenv("DM_VK_NO_HOST_IMPORT", "1", 0); setenv("DM_VK_DMAHEAP", "1", 0);
        VLOG("DM_VK_AS_MALI: Mali's routes on this GPU (no BC, push descriptors, depth clip control, vertex divisor, host import; dma-buf pools)"); }
    if (has_ext(de, nde, "VK_KHR_push_descriptor") && !getenv("DM_VK_NO_PUSH")) dext[ndext++] = "VK_KHR_push_descriptor";
    else { g_nopush = 1; VLOG("no push descriptors: descriptor sets from per-submission pools"); }
    if (has_ext(de, nde, "VK_KHR_portability_subset")) { dext[ndext++] = "VK_KHR_portability_subset"; g_moltenvk = 1; }
    // (LOAD_OP_NONE is the extensions' alone: Vulkan 1.3 has STORE_OP_NONE but not the load op. So on 1.3 the extension is enabled as well where the driver lists it,
    //   unless DM_VK_LOAD_NONE=0, which makes the device as before; without either extension no pass begins with LOAD_OP_NONE, seg_record_prim)
    { const char *ln = getenv("DM_VK_LOAD_NONE"); int want_ln = !(ln && !strcmp(ln, "0"));
      if (g_api >= VK_API_VERSION_1_3 && !want_ln) g_store_none = 1;
      else if (has_ext(de, nde, "VK_KHR_load_store_op_none")) { dext[ndext++] = "VK_KHR_load_store_op_none"; g_store_none = 1; g_load_none = want_ln; }
      else if (has_ext(de, nde, "VK_EXT_load_store_op_none")) { dext[ndext++] = "VK_EXT_load_store_op_none"; g_store_none = 1; g_load_none = want_ln; }
      else if (g_api >= VK_API_VERSION_1_3) g_store_none = 1; }
    if (getenv("DM_VK_NO_STORE_NONE")) g_store_none = g_load_none = 0;
    VLOG("depth / stencil a pass does not write: %s; not touched at all: %s", g_store_none ? "store op NONE (DM_VK_NO_STORE_NONE=1: stored)" : "stored (no store op NONE here)",
         g_load_none ? "load op NONE too (DM_VK_LOAD_NONE=0: loaded)" : "loaded (no load op NONE here)");
    if (!g_nopush) { VkPhysicalDevicePushDescriptorPropertiesKHR pd = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pd }; if (vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2(g_pd, &pp);
        if (pd.maxPushDescriptors) g_maxpush = pd.maxPushDescriptors; if (getenv("DM_VK_MAXPUSH")) g_maxpush = (uint32_t)atoi(getenv("DM_VK_MAXPUSH"));   // a lower limit tests the set path
        VLOG("push descriptors: up to %u per program", g_maxpush); }
    // Vulkan 1.0 GPUs: every external-memory extension needs VK_KHR_external_memory beside it (core from 1.1); without it listed, no external-memory route
    //   (simdev: Turnip on Adreno 610 with a 1.0 device, VUID-vkCreateDevice-ppEnabledExtensionNames-01387 for VK_EXT_external_memory_host)
    int dev10 = VK_API_VERSION_MAJOR(g_props.apiVersion) == 1 && VK_API_VERSION_MINOR(g_props.apiVersion) == 0;
    int xmem_ok = !dev10 || has_ext(de, nde, "VK_KHR_external_memory");
    if (xmem_ok && (g_host_import = has_ext(de, nde, "VK_EXT_external_memory_host"))) { dext[ndext++] = "VK_EXT_external_memory_host"; if (dev10) dext[ndext++] = "VK_KHR_external_memory"; }
    if ((g_has_swapchain = has_ext(de, nde, "VK_KHR_swapchain"))) dext[ndext++] = "VK_KHR_swapchain";
    if (g_has_swapchain && (g_dtiming = has_ext(de, nde, "VK_GOOGLE_display_timing"))) dext[ndext++] = "VK_GOOGLE_display_timing";   // frame pacing (pace_time)
    if ((g_clipctl = !as_mali && has_ext(de, nde, "VK_EXT_depth_clip_control"))) dext[ndext++] = "VK_EXT_depth_clip_control";
    if ((g_border = has_ext(de, nde, "VK_EXT_custom_border_color"))) dext[ndext++] = "VK_EXT_custom_border_color";
    // NVN's mirror-clamp wrap modes (MIRROR_CLAMP / _TO_EDGE) are Vulkan's MIRROR_CLAMP_TO_EDGE, which is not core 1.0: Mali-G72 r26p0, Mali-G52, PowerVR BXM and
    //   Vulkan 1.0 drivers lack it (simdev). The extension where listed; 1.2's feature below; without either, mirrored repeat (wrapm)
    if ((g_mclamp = has_ext(de, nde, "VK_KHR_sampler_mirror_clamp_to_edge"))) dext[ndext++] = "VK_KHR_sampler_mirror_clamp_to_edge";
    if ((g_divisor = !as_mali && has_ext(de, nde, "VK_EXT_vertex_attribute_divisor"))) dext[ndext++] = "VK_EXT_vertex_attribute_divisor";
    if ((g_eds2 = has_ext(de, nde, "VK_EXT_extended_dynamic_state2"))) dext[ndext++] = "VK_EXT_extended_dynamic_state2";
    int compctl = has_ext(de, nde, "VK_EXT_image_compression_control"); if (compctl) dext[ndext++] = "VK_EXT_image_compression_control";
    // min / max sampler filtering (the game's reduction-mode samplers): Vulkan 1.2's samplerFilterMinmax (read with the features below), or this extension before 1.2
    if (g_api < VK_API_VERSION_1_2 && (g_minmax = has_ext(de, nde, "VK_EXT_sampler_filter_minmax"))) dext[ndext++] = "VK_EXT_sampler_filter_minmax";
    // Vulkan 1.0 drivers: what 1.1 has built in and the engine relies on, as extensions. maintenance1: the flipped viewport (a negative height) that turns NVN's upper-left
    //   origin into Vulkan's; shader_draw_parameters: the base instance 718 of 1,543 Sword / Shield shaders read; storage_buffer_storage_class: a shader that uses it
    //   (the shaders themselves are made SPIR-V 1.0 at module creation: vk_shader_module_10)
    if (g_api < VK_API_VERSION_1_1) {
        static const char *const x10[] = { "VK_KHR_maintenance1", "VK_KHR_shader_draw_parameters", "VK_KHR_storage_buffer_storage_class" };
        // without shader_draw_parameters: spv_no_drawparams (push constants); without storage_buffer_storage_class: spv_sb_to_uniform (Uniform + BufferBlock)
        for (unsigned k = 0; k < sizeof x10 / sizeof *x10; k++) { if (has_ext(de, nde, x10[k])) dext[ndext++] = x10[k]; else if (k == 0) g_yflip = 1; else if (k == 1) g_drawp = 0; else g_nosbsc = 1;
            if (k && !has_ext(de, nde, x10[k])) VLOG("Vulkan 1.0 driver without %s: %s", x10[k], k == 1 ? "draw parameters come from push constants" : "storage buffers declared the Vulkan 1.0 way"); }
        VLOG("Vulkan 1.0 driver: shaders made SPIR-V 1.0 as they load, 1.1 features through their extensions");
    }   // to read whether render targets stay compressed
    if (getenv("DM_VK_YFLIP")) g_yflip = atoi(getenv("DM_VK_YFLIP")) != 0;   // (tries the route on a driver that has maintenance1)
    if (g_yflip) VLOG("no flipped viewports (Vulkan 1.0 without VK_KHR_maintenance1): the upper-left origin is made in the vertex shaders");
    // the GPU described once for the quirk table (vk_quirks.h); the driver ID and Arm's release are added below, where the driver properties are read
    g_gpuid.vendor = g_props.vendorID; g_gpuid.driver_version = g_props.driverVersion; snprintf(g_gpuid.name, sizeof g_gpuid.name, "%s", g_props.deviceName);
    g_gpuid.custom = getenv("DM_VK_DRIVER") && *getenv("DM_VK_DRIVER"); g_gpuid.api = g_props.apiVersion;
    // the game, from its profile (game_profile.h, generated from game.tsv): the app (key port → DM_PORT_*, named in the log) and the game's own per-GPU choices (key
    //   engine_quirks → DM_GAME_ENG_*, one gpu_id.eng bit each, mapped by name here so the order of game.py's list cannot move them; the table's rules read only these bits)
    g_gpuid.port = DM_PORT_ARCEUS ? PORT_ARCEUS : DM_PORT_PALDEA ? PORT_PALDEA : PORT_GALAR;
    g_gpuid.eng = (DM_GAME_ENG_QC11_NO_EDS ? ENG_QC11_NO_EDS : 0u) | (DM_GAME_ENG_ASYNC_PSO_OFF ? ENG_ASYNC_PSO_OFF : 0u) | (DM_GAME_ENG_DRIVER_SELFTESTS ? ENG_DRIVER_SELFTESTS : 0u)
                | (DM_GAME_ENG_NVN_ASYNC_NEVER ? ENG_NVN_ASYNC_NEVER : 0u) | (DM_GAME_ENG_MALI_LIT_STANDIN ? ENG_MALI_LIT_STANDIN : 0u);
    { FILE *f = fopen("/proc/meminfo", "r"); char ln[128]; long kb = 0; while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemTotal: %ld kB", &kb) == 1) break; if (f) fclose(f); g_gpuid.ram_kb = kb; }
    if (getenv("DM_VK_NO_HOST_IMPORT")) g_host_import = 0;
    // Imported host memory pins the game's pools (about 3 GB) for every submission. Samsung's Xclipse driver of the late-2026 One UI update refuses that a few frames into play
    // (vkQueueSubmit: VK_ERROR_OUT_OF_HOST_MEMORY with a gigabyte free, #39), so on Xclipse the driver's own memory backs the pools instead (as on Mali); elsewhere a
    // submission that fails while pools are imported leaves a marker (host_import_failed, next to the pipeline cache) and the next start does the same. DM_VK_HOST_IMPORT=1 forces import.
    // Decided before the fd route below: deciding it after left Xclipse without the fd route, on mirrored pools (every pool twice in memory, swapping, multi-second stalls: #41)
    if (g_host_import && !getenv("DM_VK_HOST_IMPORT")) {
        const char *why = quirk_note(Q_HOST_IMPORT_OFF, quirk_default(Q_HOST_IMPORT_OFF, &g_gpuid), 0) ? "Samsung Xclipse" : NULL;
        // (never on Adreno, vk_quirks.h Q_IMPORT_MARK_OFF: its fallback, moving the driver's mappings, is the route that crashed it on area changes; one failed submission
        //   must not switch it for good)
        if (!why && !quirk_note(Q_IMPORT_MARK_OFF, quirk_default(Q_IMPORT_MARK_OFF, &g_gpuid), 0) && host_import_mark(0)) why = "a submission failed with it last time";
        if (why) { g_host_import = 0; VLOG("memory pools: host import off (%s): the driver's own memory backs the pools", why); }
    }
#ifdef __linux__
    // The fd route's handle type: a dma-buf where the driver exports buffers as one (Android's drivers: Mali refused opaque-fd buffers, #40; a dma-buf is mmap-able by definition,
    // an opaque fd need not be), else an opaque fd
    // A driver that crashed allocating exportable memory (Xclipse 540, driver 0x6000230: a segfault inside vkAllocateMemory on the first pool, #45) left the marker
    // fd_export_mark sets around that first allocation: this start moves the driver's mappings instead (DM_VK_FDMAP=1 tries the fd route again)
    int fd_crashed = fd_export_mark(0) && !(getenv("DM_VK_FDMAP") && atoi(getenv("DM_VK_FDMAP")));
    if (fd_crashed) VLOG("[RED] memory pools: the driver crashed exporting its memory last time: no export this start");
    // the heap route's own marker (heap_import_mark): set by a crash importing the first dma-buf or a failed submission with them; DM_VK_DMAHEAP=1 in env.txt tries it
    //   again (a value from the driver check is this driver's default: the marker still applies, vk_quirks.h vkq_env_player)
    const char *hep = vkq_env_player("DM_VK_DMAHEAP");
    int heap_crashed = heap_import_mark(0) && !(hep && atoi(hep));
    // The marker only proves something while the route without it does better: a start that skipped the heap route and crashed anyway (after the device came up) sends
    //   the next start back to it. Without this a marker left by a run closed as stuck kept every later start on the mirrored route, which crashed an Xclipse 530 at
    //   every start (#76); a phone whose heap route really crashes (#63) stays off it as long as the other route holds
    if (heap_crashed && mark_file("heap_route_skipped", "", 0) && getenv("DM_LAST_CRASH_DEVICE")) {
        heap_import_mark(-1); mark_file("heap_route_skipped", "", -1); heap_crashed = 0;
        VLOG("[RED] memory pools: the last start went without the dma-buf heap route and crashed too: the heap route again");
    }
    if (heap_crashed) { mark_file("heap_route_skipped", "a start that left the dma-buf heap route out (heap_import_crashed); if it crashes too, the next start uses the route again\n", 1);
        VLOG("[RED] memory pools: importing dma-bufs failed last time: not this start"); }
    else mark_file("heap_route_skipped", "", -1);
    // Vulkan 1.0 GPUs (Adreno 5xx's 1.0 drivers, Mali r14p0): external memory is three extensions there, not core. The buffer query only through its instance
    //   extension; the fd routes only with VK_KHR_external_memory beside VK_KHR_external_memory_fd (simdev: adreno-512-vk10, VUID-vkCreateDevice-...-01387)
    //   (dev10 / xmem_ok: worked out before the host-import route, which needs the same)
    if (dev10 && !g_inst_xmemcaps) vkGetPhysicalDeviceExternalBufferProperties = NULL;
    if (!g_host_import && (!fd_crashed || !heap_crashed) && xmem_ok && has_ext(de, nde, "VK_KHR_external_memory_fd") && !(getenv("DM_VK_NO_FDMAP") && atoi(getenv("DM_VK_NO_FDMAP")))) {
        VkExternalMemoryHandleTypeFlagBits cand[2] = { VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
        int dmabuf = has_ext(de, nde, "VK_EXT_external_memory_dma_buf") && !(getenv("DM_VK_NO_DMABUF") && atoi(getenv("DM_VK_NO_DMABUF")));
        for (int i = dmabuf ? 0 : 1; i < 2 && !g_fdmap && !fd_crashed; i++) {
            if (!vkGetPhysicalDeviceExternalBufferProperties) { g_fdmap = -1; break; }   // (a 1.0 instance: the opaque fd, tried as before)
            VkPhysicalDeviceExternalBufferInfo xi = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, .usage = BUF_ALL, .handleType = cand[i] };
            VkExternalBufferProperties xp = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES };
            vkGetPhysicalDeviceExternalBufferProperties(g_pd, &xi, &xp);
            VLOG("memory pools: %s buffers: external memory features %#x", cand[i] == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "dma-buf" : "opaque-fd", (unsigned)xp.externalMemoryProperties.externalMemoryFeatures);
            if (xp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) { g_fdtype = cand[i]; g_fdmap = -1; }
        }
        // Mali (ARM, PanVK too) answers that no buffer is exportable (Mali-G720 and Mali-G615 with PanVK, #53 / #47), and its pools then stay mirrored, where the game's
        //   pictures came out white. The dma-buf route is tried anyway: a buffer it refuses (VK_ERROR_INVALID_EXTERNAL_HANDLE), memory it does not export, or a crash
        //   (fd_export_mark) each fall back to moving the driver's mappings, as today
        // A dma-buf the driver imports, from Android's heap (uncached first: the GPU then sees the game's writes without cache upkeep): Mali, which imports but does not
        //   export (#47: the forced export failed, errno 2), or any GPU with DM_VK_DMAHEAP=1
        // Samsung Xclipse without export (its export crashed the driver: Xclipse 530 / 540, #45 / #57 / #59) took the mremap route, where its driver handed out
        //   mappings the pools had taken and the game crashed on scene changes: it takes the heap route too
        {   const char *he = getenv("DM_VK_DMAHEAP"), *sh = getenv("DM_VK_POOL_SHADOW");   // (he from the driver check, not hep: the default, under the fd route and the marker)
            int want = hep ? atoi(hep) : (!g_fdmap && !heap_crashed && (he ? atoi(he) != 0 : quirk_default(Q_DMAHEAP, &g_gpuid))); quirk_note(Q_DMAHEAP, want, vkq_env_src("DM_VK_DMAHEAP"));
            if (sh && atoi(sh)) want = 0;   // (a forced mirror leaves it out too)
            VkPhysicalDeviceExternalBufferInfo xi = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, .usage = BUF_ALL, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
            VkExternalBufferProperties xp = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES };
            if (want && dmabuf && vkGetPhysicalDeviceExternalBufferProperties) vkGetPhysicalDeviceExternalBufferProperties(g_pd, &xi, &xp);
            if (want && dmabuf && (xp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
                // DM_VK_DMAHEAP_CACHED=1: the cached heap first. The CPU reads and writes uncached memory slowly (the game's own code works in these pools, and the
                //   texture uploads read them); the cached heap is right without cache upkeep only where the GPU sees the CPU's caches, which the self-test's
                //   heap-speed line reports per phone ("right without cache upkeep"). Off by default: wrong on a phone without that, the picture would show stale data
                static const char *heaps_u[] = { "/dev/dma_heap/system-uncached", "/dev/dma_heap/system" }, *heaps_c[] = { "/dev/dma_heap/system", "/dev/dma_heap/system-uncached" };
                const char *hc = getenv("DM_VK_DMAHEAP_CACHED"); const char **heaps = hc && atoi(hc) ? heaps_c : heaps_u;
                for (int i = 0; i < 2 && g_heapfd < 0; i++) if ((g_heapfd = open(heaps[i], O_RDONLY | O_CLOEXEC)) >= 0) g_heapname = heaps[i];
                if (g_heapfd >= 0) { g_heapmap = -1; g_fdmap = 0; g_fdtype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                    VLOG("memory pools: dma-bufs from %s imported into the driver, mapped over the pools (to try)", g_heapname); }
                else VLOG("memory pools: no dma-buf heap could be opened (errno %d)", errno);
            } else if (want) VLOG("memory pools: the driver does not import dma-buf buffers (features %#x)", (unsigned)xp.externalMemoryProperties.externalMemoryFeatures);
        }
        if (!g_fdmap && !g_heapmap && !fd_crashed && dmabuf && quirk_note(Q_DMABUF_TRY, quirk_default(Q_DMABUF_TRY, &g_gpuid), 0)) { g_fdtype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; g_fdmap = -1;
            VLOG("memory pools: Mali lists no exportable buffer: the dma-buf route tried anyway (it falls back by itself)"); }
        if ((g_heapmap || g_fdmap) && dev10) dext[ndext++] = "VK_KHR_external_memory";   // (core from 1.1)
        if (g_heapmap) { dext[ndext++] = "VK_KHR_external_memory_fd"; dext[ndext++] = "VK_EXT_external_memory_dma_buf"; }
        else if (g_fdmap) { dext[ndext++] = "VK_KHR_external_memory_fd"; if (g_fdtype == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) dext[ndext++] = "VK_EXT_external_memory_dma_buf"; }
    }
    VLOG("memory pools: fd export %s (host import %d, VK_KHR_external_memory_fd %d, VK_EXT_external_memory_dma_buf %d)", !g_fdmap ? "off" : g_fdtype == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "to try, as a dma-buf" : "to try, as an opaque fd",
         g_host_import, has_ext(de, nde, "VK_KHR_external_memory_fd"), has_ext(de, nde, "VK_EXT_external_memory_dma_buf"));
#endif
    free(de);
    // Every feature the device has is enabled (the translated shaders declare what they use: 64-bit integers, image formats without a type, …), robust buffer access included:
    // uniform blocks are declared 64 KB and bound with the game's smaller ranges, and Maxwell reads 0 past them
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fbc = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT };
    VkPhysicalDeviceDepthClipControlFeaturesEXT fcc = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, .pNext = g_border ? &fbc : NULL };
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT fdv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, .pNext = g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL };
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT fe2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, .pNext = g_divisor ? (void *)&fdv : g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL };
    void *opt = g_eds2 ? (void *)&fe2 : g_divisor ? (void *)&fdv : g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL;
    VkPhysicalDeviceImageCompressionControlFeaturesEXT fic = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT, .pNext = opt };
    if (compctl) opt = &fic;
    // Vulkan 1.3: its own feature blocks. Older devices: the blocks of their version plus the extensions' own structures (the 1.2 / 1.3 blocks are not allowed there)
    //   (an extension's structure only when it is enabled: dynamic rendering and extended dynamic state may be missing on an old driver)
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT fed = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT };
    VkPhysicalDeviceTimelineSemaphoreFeatures fts = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
    VkPhysicalDeviceDynamicRenderingFeatures fdr = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES };
    void *tail = opt;
    if (old && x_eds) { fed.pNext = tail; tail = &fed; }
    if (old11 && g_tlsem) { fts.pNext = tail; tail = &fts; }   // (only with its extension enabled: a structure of one that is not is invalid)
    // Vulkan 1.1 devices: shaderDrawParameters has its own structure there (the 1.1 block is 1.2+); read for g_drawp, and enabled with the device when present
    VkPhysicalDeviceShaderDrawParametersFeatures fsdp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES };
    if (old11 && g_api >= VK_API_VERSION_1_1) { fsdp.pNext = tail; tail = &fsdp; }
    if (old && x_dynren) { fdr.pNext = tail; tail = &fdr; }
    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = opt };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = old ? tail : (void *)&f13 };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12 };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = old11 ? tail : (void *)&f11 };
    // A Vulkan 1.0 driver without VK_KHR_get_physical_device_properties2 has no Features2 / Properties2 at all: the plain 1.0 query, and the device made with
    //   pEnabledFeatures (below); every extension feature structure then stays zero, so those features count as absent (Galaxy-Android's Vulkan 1.0 renderer works so)
    int have_f2 = vkGetPhysicalDeviceFeatures2 != NULL;
    if (have_f2) vkGetPhysicalDeviceFeatures2(g_pd, &f2);
    else { vkGetPhysicalDeviceFeatures(g_pd, &f2.features); VLOG("Vulkan 1.0 driver without VK_KHR_get_physical_device_properties2: core features only"); }
    int dynren = old ? x_dynren && fdr.dynamicRendering : f13.dynamicRendering, timeline = old11 ? fts.timelineSemaphore : f12.timelineSemaphore;
    int eds = old ? x_eds && fed.extendedDynamicState : 1;
    if (!timeline && g_tlsem) { g_tlsem = 0; VLOG("timeline semaphores listed but not supported: a fence for each submission instead"); }
    // Adreno: render passes by default (Galar Native #205), for every game (vk_quirks.h Q_NO_DYNREN; Legends: Arceus too since its v0.45). Its vkCmdBeginRendering
    //   callocs a large block at every pass begin (freed again when the command pool is reset): on a Galaxy S25+ that was a third of the replay's CPU time in Sword /
    //   Shield, and with render passes and their framebuffers kept (fb_cache.h, FBC_N 1024) the replay went from 27.5 to 17.3 ms a frame. DM_VK_NO_DYNREN=0 / 1 decides
    //   on any GPU; Mali and the rest as before
    { const char *e = getenv("DM_VK_NO_DYNREN"); int no = quirk_note(Q_NO_DYNREN, e ? atoi(e) != 0 : quirk_default(Q_NO_DYNREN, &g_gpuid), e != NULL);
      g_dynren = dynren && !no; }
    // Qualcomm's Vulkan 1.1 drivers (Adreno 6xx / 710 stock, API 1.1.x) misread vertex strides passed per draw (vkCmdBindVertexBuffers2): exploded, stretched
    //   polygons across the view (Arceus #34, Galar #209). Built into each pipeline there unless DM_VK_NO_EDS=0. Per game (vk_quirks.h Q_NO_EDS reads the profile's
    //   engine_quirks qc11_no_eds through gpu_id.eng: Legends: Arceus); Sword / Shield answered #209 with full barriers instead (Q_BARRIER_RELAXED, old Qualcomm
    //   drivers) and keep per-draw state there
    const char *ne = getenv("DM_VK_NO_EDS");
    g_eds = eds && !quirk_note(Q_NO_EDS, ne ? atoi(ne) != 0 : quirk_default(Q_NO_EDS, &g_gpuid), ne != NULL);
    // DM_VK_BASELINE=1 (vk_quirks.h): the Vulkan 1.1 floor on any GPU, each feature's own fallback, as a phone with an old driver runs: render passes, the draw state in
    //   each pipeline, a fence per submission. For reproducing a low-end driver's problem on a phone with a new one
    int baseline = 0;
    { const char *e = getenv("DM_VK_BASELINE"); if (e && atoi(e) > 0) { baseline = 1; g_dynren = 0; g_eds = 0; g_tlsem = 0;
        VLOG("[quirk] Vulkan 1.1 baseline (DM_VK_BASELINE): render passes, draw state in each pipeline, a fence per submission, whatever this driver offers"); } }
    g_gpuid.dynren = g_dynren;   // (Q_VIEW_QUARANTINE reads it)
    quirk_note(Q_PCACHE_GAPS_OFF, quirk_default(Q_PCACHE_GAPS_OFF, &g_gpuid) && !g_dynren, 0);   // (pso_make applies it with render passes only; noted here, once)
    VLOG("passes: %s; draw state: %s", g_dynren ? "dynamic rendering" : !dynren ? "render passes (the driver has no dynamic rendering)" : baseline ? "render passes (DM_VK_BASELINE)"
         : quirk_default(Q_NO_DYNREN, &g_gpuid) && !getenv("DM_VK_NO_DYNREN") ? "render passes (Adreno's default: DM_VK_NO_DYNREN=0 for dynamic rendering)" : "render passes (DM_VK_NO_DYNREN)",
         g_eds ? "set per draw (extended dynamic state)" : !eds ? "built into each pipeline (the driver has no extended dynamic state)" : baseline ? "built into each pipeline (DM_VK_BASELINE)"
               : ne ? "built into each pipeline (DM_VK_NO_EDS)" : g_gpuid.vendor == VEN_ARM ? "built into each pipeline (Arm's driver before r44 ignores per-draw vertex strides; DM_VK_NO_EDS=0 to try)" : "built into each pipeline (Qualcomm Vulkan 1.1 driver: per-draw vertex strides drawn wrong; DM_VK_NO_EDS=0 to try)");
    if (!fcc.depthClipControl) g_clipctl = 0;
    // custom border colours are made without a format (sampler_get), which needs customBorderColorWithoutFormat: a driver without it gets the nearest standard colour
    if (!fbc.customBorderColors || !fbc.customBorderColorWithoutFormat) { if (fbc.customBorderColors) VLOG("custom border colours need a format on this driver: the nearest standard colour instead"); g_border = 0; }
    // Mali-G615 on Arm's own driver before r49 (the G615 MC2's r44p1) closed the game three frames in (addr 0x90 in libGLES_mali.so: #63 #65 #82 #105 #112) unless its
    //   uniform buffers were copied at record time (DM_VK_UBO_SNAPSHOT) and its BC textures decoded, not transcoded to ETC2 (DM_VK_BC_TRANSCODE=off): the two logs that play
    //   on v0.5x (#77, #110) have both, and every crash lacks one of them. Both from the first start there (the safe-start ladder took four crashes to reach them);
    //   env.txt still wins. The snapshots only on the MC1 / MC2 since v0.97: every crash in that table is an MC2, and the G615 MC6 (POCO X6 Pro, r44p1) played the
    //   field without them, its game thread at 25.8 ms a frame against 39 ms with them (997 draws: the copy at every draw is a third of the replay's time)
    if (quirk_note(Q_G615_SAFE, quirk_default(Q_G615_SAFE, &g_gpuid), vkq_env_src("DM_VK_UBO_SNAPSHOT") > vkq_env_src("DM_VK_BC_TRANSCODE") ? vkq_env_src("DM_VK_UBO_SNAPSHOT") : vkq_env_src("DM_VK_BC_TRANSCODE"))) {
        int u = !getenv("DM_VK_UBO_SNAPSHOT"), b = !getenv("DM_VK_BC_TRANSCODE");
        int small = strstr(g_props.deviceName, "MC1") || strstr(g_props.deviceName, "MC2") || !strstr(g_props.deviceName, " MC");   // (no core count in the name: the safe side)
        setenv("DM_VK_UBO_SNAPSHOT", small ? "1" : "0", 0); setenv("DM_VK_BC_TRANSCODE", "off", 0);
        VLOG("Mali-G615 on Arm's driver r%up%u: DM_VK_UBO_SNAPSHOT=%s (%s), DM_VK_BC_TRANSCODE=%s (%s): what the G615 MC2 needs to get past its first frames (#110; snapshots only there)",
             g_props.driverVersion >> 22, g_props.driverVersion >> 12 & 0x3ff, getenv("DM_VK_UBO_SNAPSHOT"), u ? "default here" : "env.txt", getenv("DM_VK_BC_TRANSCODE"), b ? "default here" : "env.txt");
    }
    // BC1–BC7: sampled directly where the GPU has them (Adreno, Xclipse, desktop), else decoded on upload (Mali, PowerVR; DM_VK_NO_BC=1 tries that path anywhere)
    if (getenv("DM_VK_NO_BC") && atoi(getenv("DM_VK_NO_BC"))) f2.features.textureCompressionBC = VK_FALSE;
    nvn_fmt_vk_bc(f2.features.textureCompressionBC);
    // without BC: re-encoded as ETC2 / EAC / ASTC on upload instead (etc_encode.h), the same size as BC in memory, not 4–8 times it as plain texels;
    // DM_VK_BC_TRANSCODE=off keeps the plain decode
    // PowerVR (Imagination): BC textures decoded to plain RGBA, not transcoded to ETC2 / EAC / ASTC. On a Tensor G5 (PowerVR D-Series, Arceus-Native-Android #32)
    //   static world meshes came out flat lavender with the textures transcoded both ways (the BC layer's compute in v0.35-37, the CPU in v0.39); our encoders match
    //   an independent decoder (tests/etc_check.py), so what is left is how this driver samples them. The plain decode is Mali's path. DM_VK_BC_TRANSCODE=on: transcoded
    //   (vk_quirks.h Q_BC_DECODE, every game)
    if (quirk_default(Q_BC_DECODE, &g_gpuid)) { const char *tc = getenv("DM_VK_BC_TRANSCODE");
        quirk_note(Q_BC_DECODE, !(tc && strcmp(tc, "off")), tc != NULL);
        if (!tc) setenv("DM_VK_BC_TRANSCODE", "off", 0);
        VLOG("PowerVR: BC textures %s (DM_VK_BC_TRANSCODE=%s, %s)", tc && strcmp(tc, "off") ? "transcoded to ETC2 / ASTC" : "decoded to RGBA", tc ? tc : "off", tc ? "env.txt" : "default here"); }
    { const char *tc = getenv("DM_VK_BC_TRANSCODE"); nvn_fmt_vk_etc(!f2.features.textureCompressionBC && f2.features.textureCompressionETC2 && !(tc && !strcmp(tc, "off")), f2.features.textureCompressionASTC_LDR); }
    // Descriptors updated after binding: the engine never makes such a layout, and with robustBufferAccess on (f2 passes every feature as queried) a driver whose
    //   robustBufferAccessUpdateAfterBind is false must not have them enabled (Turnip on Adreno 740, VUID-VkDeviceCreateInfo-robustBufferAccess-10247): left off
    if (!old11) { f12.descriptorBindingUniformBufferUpdateAfterBind = f12.descriptorBindingStorageBufferUpdateAfterBind = f12.descriptorBindingUniformTexelBufferUpdateAfterBind =
                  f12.descriptorBindingStorageTexelBufferUpdateAfterBind = f12.descriptorBindingSampledImageUpdateAfterBind = f12.descriptorBindingStorageImageUpdateAfterBind = VK_FALSE; }
    float prio = 1;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = ndext, .ppEnabledExtensionNames = dext };
    if (!have_f2) { dci.pNext = NULL; dci.pEnabledFeatures = &f2.features; }   // (no Features2 on this driver: features the 1.0 way)
    VKCHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    dm_boot_mark("Vulkan device made");
    { extern void dm_boot_device_up(void) __attribute__((weak)); if (dm_boot_device_up) dm_boot_device_up(); }   // (android_main.c boot_state: a crash from here on counts as the driver's)
    VK_DEVICE(VK_LOAD_D)
    gap_wrap();   // (renderer phase 3: a held frame's gaps know whether anything went into them)
    if (g_api < VK_API_VERSION_1_1) { g_csm_real = vkCreateShaderModule; vkCreateShaderModule = vk_shader_module_10; }   // (Vulkan 1.0: every shader through spv_to_10)
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
    p_push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(g_dev, "vkCmdPushDescriptorSetKHR");
    if (compctl && fic.imageCompressionControl) { p_layout2 = (PFN_vkGetImageSubresourceLayout2EXT)vkGetDeviceProcAddr(g_dev, "vkGetImageSubresourceLayout2EXT"); g_compctl = 1; }
    p_hostprops = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (g_dtiming) { p_refresh = (PFN_vkGetRefreshCycleDurationGOOGLE)vkGetDeviceProcAddr(g_dev, "vkGetRefreshCycleDurationGOOGLE");
                     p_past = (PFN_vkGetPastPresentationTimingGOOGLE)vkGetDeviceProcAddr(g_dev, "vkGetPastPresentationTimingGOOGLE"); if (!p_refresh) g_dtiming = 0; }
    // the BC layer (gfx/tex/bc_gpu.c) where the GPU has no BC textures (Mali, PowerVR): uploads and copies of BC blocks converted on the GPU, each kind only once it
    //   came out the same as the CPU's on this GPU; the rest on the CPU as before (DM_VK_BC_GPU=0: all of it)
    { const VkPhysicalDeviceLimits *L = &g_props.limits;   // the limits that differ between vendors and decide what a read past a binding returns (Mali white: #53)
      VLOG("limits: uniform range %u (align %llu), storage range %u (align %llu), push constants %u, vertex attribute offset %u, stride %u, bindings %u, texel buffer %u",
           L->maxUniformBufferRange, (unsigned long long)L->minUniformBufferOffsetAlignment, L->maxStorageBufferRange, (unsigned long long)L->minStorageBufferOffsetAlignment,
           L->maxPushConstantsSize, L->maxVertexInputAttributeOffset, L->maxVertexInputBindingStride, L->maxVertexInputBindings, L->maxTexelBufferElements); }
    // Float controls (spv_float_controls): ARM's own Mali driver gets every shader with SignedZeroInfNanPreserve, where it supports it. Without it the lit scene came out
    //   at the format's maximum on Mali-G615 / G720 / G925 (#47 #53 #64 #68 #69 #72); PanVK on the same GPUs drew it right. DM_VK_FLOAT_CONTROLS=0 / 1 decides on any driver
    int arm_known = 0;   // (Arm's own driver told apart by its driver ID below; without driver properties, by the vendor ID after)
    if (g_api >= VK_API_VERSION_1_2 || g_fctl_ext) {   // (Vulkan 1.1: through the extensions, same structures)
        int drvp = g_api >= VK_API_VERSION_1_2 || g_drvprops_ext;
        VkPhysicalDeviceDriverProperties dp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
        VkPhysicalDeviceFloatControlsProperties fc = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES, .pNext = drvp ? &dp : NULL };
        VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &fc }; if (vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2(g_pd, &pp);
        if (!drvp) snprintf(dp.driverName, sizeof dp.driverName, "%s", g_props.vendorID == 0x13B5 ? "Mali (no driver properties: taken as ARM's)" : "unknown");
        // Samsung's Xclipse driver too: the probe self-test (float-controls) found inf - inf not NaN without them on Xclipse 540 and 940, and every check right with them
        const char *e = getenv("DM_VK_FLOAT_CONTROLS"); g_gpuid.driver_id = drvp ? (int)dp.driverID : -1;
        int arm = gpu_is_arm_driver(&g_gpuid), sec = gpu_is_samsung_driver(&g_gpuid);
        arm_known = 1; if (arm) { g_arm_rel = (int)(g_props.driverVersion >> 22); g_arm_p = (int)(g_props.driverVersion >> 12 & 0x3ff); g_arm_beta = drvp && strstr(dp.driverInfo, "bet") != NULL; }
        if (drvp) g_drv_id = (int)dp.driverID;
        g_fctl = fc.shaderSignedZeroInfNanPreserveFloat32 && (e ? atoi(e) != 0 : quirk_default(Q_FLOAT_CONTROLS, &g_gpuid));
        // The test shader below crashed a driver (#89: Mali-G57, the start ended right after this line): a start that dies there leaves the marker, and later starts on
        //   this driver go without float controls (DM_VK_FLOAT_CONTROLS=1 in env.txt tries again). A DM_VK_FLOAT_CONTROLS=1 from the driver check (driver-check.env:
        //   a driver right only with them) is this driver's default, not the player's: the marker still applies to it (vk_quirks.h vkq_env_player)
        int fc_crashed = g_fctl && mark_file(drv_mark("float_controls_crashed"), "", 0) && !vkq_env_player("DM_VK_FLOAT_CONTROLS");
        if (fc_crashed) g_fctl = 0;
        char rel[24] = ""; if (arm && g_arm_rel) snprintf(rel, sizeof rel, " r%dp%d", g_arm_rel, g_arm_p);
        VLOG("driver %s (id %d)%s%s; shaders keep infinities and NaN (float controls): %s", dp.driverName, (int)dp.driverID, arm ? ", ARM's own" : sec ? ", Samsung's own" : "", rel,
             fc_crashed ? "not asked for: the driver crashed on their test shader at an earlier start (DM_VK_FLOAT_CONTROLS=1 in env.txt tries again)"
             : g_fctl ? "asked for" : !fc.shaderSignedZeroInfNanPreserveFloat32 ? "not supported" : "not asked for");
        if (g_fctl) mark_file(drv_mark("float_controls_crashed"), "building the float-controls test shader; if this file is still here the driver crashed doing so, and later starts go without float controls\n", 1);
        if (g_fctl) {   // a self-test: an empty compute shader patched the same way must make a pipeline, else this start goes without (the game's shaders would fail one by one)
            static const uint32_t tiny[] = { 0x07230203, 0x00010300, 0, 5, 0, 2u << 16 | 17, 1, 3u << 16 | 14, 0, 1, 5u << 16 | 15, 5, 1, 0x6e69616d, 0, 6u << 16 | 16, 1, 17, 1, 1, 1,
                                             2u << 16 | 19, 2, 3u << 16 | 33, 3, 2, 5u << 16 | 54, 2, 1, 0, 3, 2u << 16 | 248, 4, 1u << 16 | 253, 1u << 16 | 56 };
            size_t nw; uint32_t *c = spv_float_controls(tiny, sizeof tiny / 4, &nw); VkShaderModule m = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkPipeline pp = VK_NULL_HANDLE; VkResult tr = VK_ERROR_UNKNOWN;
            VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = c };
            VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            if (c && !vkCreateShaderModule(g_dev, &mci, NULL, &m) && !vkCreatePipelineLayout(g_dev, &pli, NULL, &pl)) {
                VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl,
                    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main" } };
                tr = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pp);
            }
            if (pp) vkDestroyPipeline(g_dev, pp, NULL); if (pl) { PFN_vkDestroyPipelineLayout dpl = (PFN_vkDestroyPipelineLayout)vkGetDeviceProcAddr(g_dev, "vkDestroyPipelineLayout"); if (dpl) dpl(g_dev, pl, NULL); } if (m) vkDestroyShaderModule(g_dev, m, NULL); free(c);
            if (tr != VK_SUCCESS) { g_fctl = 0; VLOG("[RED] float controls: the driver refused a test shader asking for them (%d): not this start", (int)tr); }
            mark_file(drv_mark("float_controls_crashed"), "", -1);   // (the test came back: no crash)
        }
    }
    // A driver without Vulkan 1.2 or the float-controls extension: Arm's own when it is ARM's vendor ID and not a driver package (PanVK reports ARM's ID too)
    if (!arm_known && g_props.vendorID == 0x13B5 && !(getenv("DM_VK_DRIVER") && *getenv("DM_VK_DRIVER"))) { g_arm_rel = (int)(g_props.driverVersion >> 22); g_arm_p = (int)(g_props.driverVersion >> 12 & 0x3ff); }
    g_gpuid.arm_rel = g_arm_rel; g_gpuid.arm_p = g_arm_p; g_gpuid.arm_beta = g_arm_beta;   // (the quirk table's description complete from here)
    atomic_store(&g_gpuid_done, 1);
    gpu_id_log(&g_gpuid); quirk_note(Q_FLOAT_CONTROLS, g_fctl, vkq_env_src("DM_VK_FLOAT_CONTROLS"));
    // Arm erratum 3779215 (r38p1 to r49p0 and r50p0 to r51p0, fixed in r49p1 and r52p0; a beta r49p1, the G925's, is not known to have the fix): a clamp to zero of a value
    //   selected between constants compiled as a plain min or max. Shaders on those drivers get spv_clamp_zero; DM_VK_CLAMP_ZERO=0 / 1 decides on any driver
    { const char *e = getenv("DM_VK_CLAMP_ZERO");
      g_clampzero = quirk_note(Q_CLAMP_ZERO, e ? atoi(e) != 0 : quirk_default(Q_CLAMP_ZERO, &g_gpuid), e != NULL);
      if (g_clampzero) VLOG("shaders: a clamp to zero of a value selected between constants gets a zero the compiler cannot fold (Arm erratum 3779215%s)", e ? ", DM_VK_CLAMP_ZERO" : g_arm_beta ? ", a beta driver" : ""); }
    // NaN in min / max / saturate (spv_nan_minmax): the frame capture of a white scene on Arm's own driver (#110, Mali-G615 MC2 r44p1; #111 the G925 on r49p1 too) has
    //   every lit surface's first draw put the format's maximum in the lit scene, with float controls on; PanVK on the same GPUs draws it right. Maxwell's min / max / saturate
    //   drop a NaN and Mesa's compilers do the same, but SPIR-V's FMin / FMax / FClamp leave it undefined. Arm's own driver gets NMin / NMax / NClamp (Maxwell's answer by
    //   definition); DM_VK_NAN_MINMAX=0 / 1 decides on any driver
    //   v0.59 with this on was still white on Arm's r49p1 (#111 G925, #128 G720): nothing shows that Arm's compiler gives the N forms their NaN rule, so its driver gets
    //   the rule written out as well (spv_nan_select: OpIsNan + OpSelect after each op, 2); DM_VK_NAN_MINMAX=1 is the N forms alone
    // FMZ (spv_fmz): Maxwell's multiplies give 0 when an operand is 0, infinity and NaN included, and SPIR-V's give NaN, which a white or glowing surface can come from
    //   (a light at the eye's position: 0 x inf). Every multiply gets the rule written out. A NaN that reaches the B10G11R11 lit scene comes out as the format's maximum
    //   on Arm's own driver (the white screen: #47 #110 #111 #172) and as black on PanVK (black patches on the brightest areas): the same 0 x inf on both. On Mali GPUs
    //   (Arm's driver and PanVK alike) it is on by default since PanVK on a Mali-G615 MC6 drew everything right with it (Arceus Native v0.21, the same engine);
    //   DM_VK_FMZ=0 / 1 (the in-game menu's Mali tests) decides on any GPU
    //   Since v0.76 on by default for PanVK only: on Arm's own driver it did not help (#172, Mali-G720 r49p1, still white on v0.75) and made it worse on Bifrost (#215,
    //   Mali-G52 r26p0: 86 % white instead of 81 %, the dark outlines gone, about 10 % slower)
    // R11G11B10F render targets as RGBA16F (nvn_fmt_vk.c nvn_fmt_vk_b10_as_rgba16, route A), and the other route (DM_VK_B10_NOCOMPRESS, tex_new, route B: R11G11B10F
    //   kept, its render targets made without framebuffer compression, for the case that Arm's driver mishandles the packed float format in AFBC; the log names how it
    //   was done and what the driver then reports for each target, "b10 target"): Mali's white lit scene. Route A is the default on Arm's own driver with Sword /
    //   Shield, where the white-screen ladder (mali_white_vk.inc) checks the lit scene while playing and moves on to route B, then to offering PanVK, start by start;
    //   both are off on every other driver and game (vk_quirks.h Q_B10_STANDIN / Q_MALI_WHITE_CHECK: Arm's own driver, a game whose profile says mali_lit_standin).
    //   DM_VK_B10_AS_RGBA16 / DM_VK_B10_NOCOMPRESS in env.txt (the menu's Mali tests) always win
    { int a = 0, b = 0; const char *wa = "", *wb = "";
      mw_init(&a, &b, &wa, &wb);
      if (a) { nvn_fmt_vk_b10_as_rgba16(1); VLOG("formats: R11G11B10F drawn and sampled as RGBA16F (%s): texels converted where the game's memory meets the image, its alpha read as 1 ('b10 swap:' lines name each kind once)", wa); }
      // (a result of route A alone does not tell the format from the compression: RGBA16F may get no framebuffer compression where R11G11B10F gets it, on a GPU
      //   without FP16 AFBC. Read it beside DM_VK_B10_NOCOMPRESS's (route B: the ladder's next step), by the picture. Both are Sword / Shield fixes: see b10_on)
      if (nvn_fmt_vk_b10_swapped(0x3F)) VLOG("formats: the float-target test changes the format and may change the compression with it (a GPU without FP16 framebuffer compression): read its result beside DM_VK_B10_NOCOMPRESS's%s",
                                             g_gpuid.eng & ENG_MALI_LIT_STANDIN ? "" : "; a Sword / Shield test: here it only makes every R11G11B10F image twice the size (Legends: Arceus draws its G-buffer and bloom in it)");
      g_b10nc = b;
      if (g_b10nc) VLOG("formats: R11G11B10F render targets made without framebuffer compression (%s): %s%s", wb, g_compctl
            ? "VK_EXT_image_compression_control, compression disabled image by image"
            : "the driver has no VK_EXT_image_compression_control: made mutable with no list of view formats, which keeps Mali's AFBC (and Adreno's UBWC) off",
            g_gpuid.eng & ENG_MALI_LIT_STANDIN ? "" : " (a Sword / Shield test: see b10_on)"); }
    //   Since chain 91294b095153 the translator writes FMZ into the multiplies the game marked (as yuzu / Eden do), on every GPU: PanVK keeps its fix and the multiplies
    //   the game did not mark keep their NaN (the dark outlines on Mali-G52, #215), with a fraction of the cost. The every-multiply copy is off by default, DM_VK_FMZ=1 to compare
    { const char *e = getenv("DM_VK_FMZ"); g_fmz = e && *e && atoi(e) > 0;
      if (g_fmz) VLOG("shaders: a multiply by 0 gives 0 even for infinity and NaN, in every multiply (DM_VK_FMZ); the marked ones have it from the translator"); }
    // DM_VK_PROBE=samples,inputs,div (a diagnostic, #215): those values clamped in every fragment shader (spv_probe), one kind a run, to find which carries the white
    { const char *e = getenv("DM_VK_PROBE"); if (e && *e) { g_probe = (strstr(e, "samples") ? PROBE_SAMPLES : 0) | (strstr(e, "inputs") ? PROBE_INPUTS : 0) | (strstr(e, "div") ? PROBE_DIV : 0) | (strstr(e, "output") ? PROBE_OUTPUT : 0) | (strstr(e, "mul") ? PROBE_MUL : 0) | (strstr(e, "add") ? PROBE_ADD : 0) | (strstr(e, "fma") ? PROBE_FMA : 0)
                                                    | (strstr(e, "nanfix") ? NAN_FIX_OUT : 0) | (strstr(e, "nantex") ? NAN_FIX_TEX : 0);
      if (g_probe & (NAN_FIX_OUT | NAN_FIX_TEX)) VLOG("shaders: NaN fix, fragment shaders turn NaN into 0%s%s (DM_VK_PROBE=%s)", g_probe & NAN_FIX_OUT ? " in what they write" : "",
                        g_probe & NAN_FIX_TEX ? " in every texture sample" : "", e);
      if (strstr(e, "f16target")) { nvn_fmt_vk_r11_f16(1); VLOG("render targets: DIAGNOSTIC, R11G11B10F images stored as RGBA16F (DM_VK_PROBE=%s)", e); }
      if (g_probe) VLOG("shaders: DIAGNOSTIC, fragment shaders clamp%s%s%s%s (DM_VK_PROBE=%s)", g_probe & PROBE_SAMPLES ? " every texture sample" : "",
                        g_probe & PROBE_INPUTS ? " every input from the vertex shader" : "", g_probe & PROBE_DIV ? " every division" : "", g_probe & PROBE_OUTPUT ? " what they write (±64)" : "", e);
      if (g_probe & (PROBE_MUL | PROBE_ADD | PROBE_FMA)) VLOG("shaders: DIAGNOSTIC, fragment shaders clamp every%s%s%s result (±16384)", g_probe & PROBE_MUL ? " multiply" : "",
                        g_probe & PROBE_ADD ? " add / subtract" : "", g_probe & PROBE_FMA ? " fused multiply-add" : ""); } }
    { const char *e = getenv("DM_VK_NAN_MINMAX"); g_nanmm = e ? atoi(e) : quirk_default(Q_NAN_MINMAX, &g_gpuid); if (g_nanmm < 0 || g_nanmm > 2) g_nanmm = 1; quirk_note(Q_NAN_MINMAX, g_nanmm, e != NULL);
      if (g_nanmm) VLOG("shaders: min / max / saturate give the operand that is not NaN, as on the Switch's GPU (NMin / NMax / NClamp%s%s)", g_nanmm == 2 ? " and the rule written out with isnan" : "",
                        e ? ", DM_VK_NAN_MINMAX" : ", Arm's own driver"); }
    // Maxwell's special functions at their edge cases (spv_exact_math): log2 (0) = -inf, exp2 below 2^-126 = 0, inversesqrt (0) = +inf, a negative root NaN, F2I clamped
    //   with NaN 0, where SPIR-V leaves the answer to the driver. The translated lighting takes pow (max (0, x), y) as exp2 (log2 (x) * y), so x = 0 on every surface facing
    //   away from a light, and still white on Arm's own driver in v0.67 with every switch above (#47: Mali-G615 MC6, r44p1); PanVK draws the same shaders right.
    //   On by default with Arm's own driver; DM_VK_EXACT_MATH=<mask> chooses (1 log2, 2 exp2, 4 inversesqrt, 8 sqrt, 16 float to int; 31 all, 0 none) on any driver
    { const char *e = getenv("DM_VK_EXACT_MATH"); g_exact = e ? (unsigned)atoi(e) & EXACT_ALL : quirk_default(Q_EXACT_MATH, &g_gpuid) ? EXACT_ALL : 0; quirk_note(Q_EXACT_MATH, (int)g_exact, e != NULL);
      if (g_exact) VLOG("shaders: Maxwell's answers for log2 / exp2 / inversesqrt / sqrt / float-to-int at their edge cases (mask %u%s)", g_exact, e ? ", DM_VK_EXACT_MATH" : ", Arm's own driver"); }
    // The translator's registers start at zero (spv_init_locals): a register read on a path that did not write it is undefined in SPIR-V; Mesa's compilers make it 0 (PanVK
    //   drew the lit scene right on the phones where Arm's driver drew it at the format's maximum), Arm's driver may keep anything. Still white with spv_exact_math on all
    //   five classes (#47, mali-white-test1). On by default with Arm's own driver; DM_VK_INIT_LOCALS=0 / 1 decides on any driver
    { const char *e = getenv("DM_VK_INIT_LOCALS"); g_initloc = quirk_note(Q_INIT_LOCALS, e ? atoi(e) != 0 : quirk_default(Q_INIT_LOCALS, &g_gpuid), e != NULL);
      if (g_initloc) VLOG("shaders: the translator's registers start at zero (undefined values as Mesa's compilers make them%s)", e ? ", DM_VK_INIT_LOCALS" : ", Arm's own driver"); }
    // textureQueryLod's level (spv_query_lod): Arm's own driver returns floor (LOD) for it, ignoring the texture's level count (dEQP-VK texturequerylod on a Mali-G615 MC6
    //   stock driver: 63 of 190 failed, the LOD component right in all); the game's TMML reads only that component. On by default with Arm's own driver; DM_VK_QUERY_LOD=0 / 1
    { const char *e = getenv("DM_VK_QUERY_LOD"); g_qlod = quirk_note(Q_QUERY_LOD, e ? atoi(e) != 0 : quirk_default(Q_QUERY_LOD, &g_gpuid), e != NULL);
      if (g_qlod) VLOG("shaders: texture LOD queries give the level as clamp (LOD, 0, levels - 1)%s", e ? " (DM_VK_QUERY_LOD)" : " (Arm's own driver)"); }
    // Dual-source blending (the game's blend factors SRC1_*) needs dualSrcBlend; Arm's own driver has none (r44p1), and a pipeline using it there is undefined: those
    //   factors become the first output's (bfac_dev), named once in the log. DM_VK_DUALSRC=keep passes them as they are
    // Transform feedback (Legends: Arceus skins its characters and sways its trees with it) is captured by the vertex shader storing into buffers: that needs
    //   vertexPipelineStoresAndAtomics (Arceus-Native-Android#6: on Arm's driver the player, the Pokemon and swaying trees vanished, PanVK drew them)
    g_vstores = f2.features.vertexPipelineStoresAndAtomics != 0;
    VLOG("vertex shader stores (transform feedback captures): %s", f2.features.vertexPipelineStoresAndAtomics ? "supported" : "[RED] not supported by this driver: captured vertices stay empty, animated models vanish");
    g_dualsrc = f2.features.dualSrcBlend || (getenv("DM_VK_DUALSRC") && !strcmp(getenv("DM_VK_DUALSRC"), "keep"));
    if (getenv("DM_VK_DUALSRC") && !strcmp(getenv("DM_VK_DUALSRC"), "emulate")) g_dualsrc = 0;   // (a test: the two-step blend of draw on a driver that has dualSrcBlend)
    if (!f2.features.dualSrcBlend) VLOG("blending: the driver has no dual-source blending (dualSrcBlend): the game's SRC1 blend factors %s", g_dualsrc ? "passed as they are (DM_VK_DUALSRC=keep)" : "become the first output's");
    // The other optional Vulkan 1.0 features the engine can be asked for: each one missing has its route (Vulkan 1.0 baseline: no phone left out for lacking one)
    { const char *mf = getenv("DM_VK_MIN_FEATURES"); int minf = mf && atoi(mf);
      const VkPhysicalDeviceFeatures *F = &f2.features;
      g_drawp = g_api < VK_API_VERSION_1_1 ? g_drawp : old11 ? fsdp.shaderDrawParameters != 0 : f11.shaderDrawParameters != 0;
      g_minmax = g_api >= VK_API_VERSION_1_2 && !old11 ? f12.samplerFilterMinmax != 0 : g_minmax;
      if (g_api >= VK_API_VERSION_1_2 && !old11 && f12.samplerMirrorClampToEdge) g_mclamp = 1;   // (the feature is enabled with the device: f2 is passed as queried)
      g_f_clamp = F->depthClamp; g_f_logic = F->logicOp; g_f_nonsolid = F->fillModeNonSolid; g_f_indblend = F->independentBlend; g_f_biasclamp = F->depthBiasClamp; g_f_aniso = F->samplerAnisotropy;
      g_feat_have = (F->geometryShader ? SPV_NEED_GEOM : 0) | (F->tessellationShader ? SPV_NEED_TESS : 0) | (F->shaderFloat64 ? SPV_NEED_F64 : 0) | (F->shaderInt64 ? SPV_NEED_I64 : 0)
                  | (F->shaderInt16 ? SPV_NEED_I16 : 0) | (F->shaderTessellationAndGeometryPointSize ? SPV_NEED_TGPSIZE : 0) | (F->shaderImageGatherExtended ? SPV_NEED_GATHEREXT : 0)
                  | (F->shaderStorageImageMultisample ? SPV_NEED_STORMS : 0) | (F->shaderClipDistance ? SPV_NEED_CLIP : 0) | (F->shaderCullDistance ? SPV_NEED_CULL : 0)
                  | (F->imageCubeArray ? SPV_NEED_CUBEARR : 0) | (F->sampleRateShading ? SPV_NEED_SRS : 0) | (F->shaderResourceResidency ? SPV_NEED_SPARSE : 0)
                  | (F->shaderResourceMinLod ? SPV_NEED_MINLOD : 0) | (F->shaderStorageImageExtendedFormats ? SPV_NEED_EXTFMT : 0)
                  | (F->shaderStorageImageReadWithoutFormat ? SPV_NEED_IMGRNF : 0) | (F->shaderStorageImageWriteWithoutFormat ? SPV_NEED_IMGWNF : 0) | (F->multiViewport ? SPV_NEED_MVP : 0)
                  | (F->fragmentStoresAndAtomics ? SPV_NEED_FRAGSTORE : 0);
      if (minf || (getenv("DM_VK_NO_MCLAMP") && atoi(getenv("DM_VK_NO_MCLAMP")))) g_mclamp = 0;   // (test switch: the mirrored-repeat fallback on any GPU)
      if (!g_mclamp) VLOG("samplers: no mirror-clamp-to-edge on this driver: the game's mirror-clamp wrap modes become mirrored repeat (the same inside one mirror)");
      if (minf) { g_drawp = g_minmax = g_f_clamp = g_f_logic = g_f_nonsolid = g_f_indblend = g_f_biasclamp = g_f_aniso = 0; }
      if (getenv("DM_VK_NO_DRAWP")) g_drawp = !atoi(getenv("DM_VK_NO_DRAWP"));   // 1: the push-constant route on any GPU; 0: draw parameters passed as before
      if (!g_drawp) VLOG("draw parameters: the device has none%s: shaders that read them get them from push constants", getenv("DM_VK_NO_DRAWP") ? " (DM_VK_NO_DRAWP)" : "");
      char miss[256] = ""; size_t ml = 0;
      #define MISS(c, s) if (!(c) && ml < sizeof miss - 40) ml += (size_t)snprintf(miss + ml, sizeof miss - ml, "%s%s", ml ? ", " : "", s)
      MISS(g_drawp, "draw parameters (from push constants)"); MISS(g_minmax, "min / max sampler filtering (plain filtering)"); MISS(g_f_clamp, "depth clamp (left off)");
      MISS(g_f_logic, "logic ops (left off)"); MISS(g_f_nonsolid, "line / point fill (filled)"); MISS(g_f_indblend, "per-target blending (the first target's)");
      MISS(g_f_biasclamp, "depth bias clamp (unclamped)"); MISS(g_f_aniso, "anisotropic filtering (off)");
      #undef MISS
      if (ml) VLOG("optional Vulkan features this device lacks%s: %s", minf ? " (DM_VK_MIN_FEATURES)" : "", miss);
      if (!(g_feat_have & SPV_NEED_GEOM) || !(g_feat_have & SPV_NEED_TESS))
          VLOG("shader stages this device lacks: %s%s%s (programs with them are skipped)", g_feat_have & SPV_NEED_GEOM ? "" : "geometry", !(g_feat_have & SPV_NEED_GEOM) && !(g_feat_have & SPV_NEED_TESS) ? ", " : "", g_feat_have & SPV_NEED_TESS ? "" : "tessellation"); }
    // The BC layer's set-up builds and runs its shaders on this GPU: a start that dies there leaves the marker, and later starts on this driver convert on the CPU
    //   (DM_VK_BC_GPU=1 tries again). Not on Vulkan 1.0: its shader is SPIR-V 1.3 and bc_gpu.c makes its module itself, past vk_shader_module_10 (simdev t_vkdev,
    //   Mali-G72 r14p0 profile: VUID-VkShaderModuleCreateInfo-pCode-08737); BC textures are converted on the CPU there
    if (!f2.features.textureCompressionBC && g_api >= VK_API_VERSION_1_1) {
        // (a DM_VK_BC_GPU from the driver check, its bc-textures test having crashed or failed, is this driver's default: the marker still applies, vkq_env_player)
        // PowerVR: its compute results already differ from the CPU's for some kinds at the start-up check, and objects came out purple on a Tensor G5 (#32, with
        //   reflections on and off): every BC texture converted on the CPU there unless DM_VK_BC_GPU=1
        const char *be = getenv("DM_VK_BC_GPU");
        int pvr_off = quirk_note(Q_BC_GPU_OFF, quirk_default(Q_BC_GPU_OFF, &g_gpuid) && !(be && atoi(be) > 0), vkq_env_src("DM_VK_BC_GPU"));
        if (mark_file(drv_mark("bc_gpu_crashed"), "", 0) && !vkq_env_player("DM_VK_BC_GPU")) VLOG("[RED] BC layer: the driver crashed setting it up at an earlier start: BC textures converted on the CPU (DM_VK_BC_GPU=1 in env.txt tries again)");
        else if (pvr_off) VLOG("BC layer: off on PowerVR (BC textures converted on the CPU; DM_VK_BC_GPU=1 turns it on)");
        else {
            mark_file(drv_mark("bc_gpu_crashed"), "setting up the BC layer (its shaders built and checked on this GPU); if this file is still here the driver crashed doing so, and later starts convert on the CPU\n", 1);
            bc_gpu_info bi = { g_dev, g_q, g_qf, vkGetDeviceProcAddr, &g_memp, vk_logf }; g_bcgpu = bc_gpu_init(&bi) ? 7 : 0;
            if (g_bcgpu && getenv("DM_VK_BC_GPU_PATHS")) { g_bcgpu = atoi(getenv("DM_VK_BC_GPU_PATHS")) & 7; VLOG("BC layer: only on paths %d (DM_VK_BC_GPU_PATHS: 1 uploads, 2 buffer copies, 4 raw-block copies)", g_bcgpu); }
            mark_file(drv_mark("bc_gpu_crashed"), "", -1);
        }
    }
    if (g_host_import) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp }; if (vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2(g_pd, &pp);
        g_host_align = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096;
    }
    {   // 24-bit depth where the device renders to it (Adreno), else the 32-bit float formats (Apple GPUs)
        VkFormatProperties a, b; vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_X8_D24_UNORM_PACK32, &a); vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_D24_UNORM_S8_UINT, &b);
        nvn_fmt_vk_depth24(!!(a.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT), !!(b.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
    }
    VLOG("Vulkan device %s (API %u.%u.%u, driver %#x): pools %s, depth clip control %d, custom border colours %d, vertex divisor %d, extended dynamic state 2 %d, BC textures %s",
         g_props.deviceName, VK_API_VERSION_MAJOR(g_props.apiVersion), VK_API_VERSION_MINOR(g_props.apiVersion), VK_API_VERSION_PATCH(g_props.apiVersion), g_props.driverVersion,
         g_host_import ? "imported host memory" : "driver memory moved onto the pool (mremap)", g_clipctl, g_border, g_divisor, g_eds2,
         nvn_fmt_vk_transcode(0x46) ? "transcoded to ETC2 on upload" : nvn_fmt_vk_decode(0x46) ? "decoded on upload" : "sampled by the GPU");
    completion_start();
    dummies_init();
    // the driver self-tests at start, for a game whose profile says driver_selftests (vk_quirks.h Q_DRIVER_SELFTESTS through gpu_id.eng; Legends: Arceus: its releases
    //   run both at start; Sword / Shield / Scarlet / Violet not yet)
    if (quirk_note(Q_DRIVER_SELFTESTS, quirk_default(Q_DRIVER_SELFTESTS, &g_gpuid), 0)) {
        vtx_canary();    // (whether this driver reads vertex sizes given per draw right; if not, they go into each pipeline: vtx_canary.h)
        tf_selftest();   // (Arm's driver, Q_TF_SELFTEST: whether transform feedback captures write; tf_selftest.h)
    }
    tfp_decide();    // (and whether they go as points: tf_points.h)
    dm_boot_mark("device set-up and self-tests done");
    prewarm_start();
}

// ================= Device memory (images, internal buffers) =================
// A first-fit allocator over 64 MB blocks per memory type (larger requests get a block of their own): drivers cap the number of allocations (4096 is common), and a game holds
// thousands of textures
// Blocks by lifetime (DM_VK_MEM_LIFETIME=0: one kind, as before): images and other long-lived objects (VA_LONG), staging and one-off buffers (VA_TEMP) and the
//   submissions' arenas (VA_ARENA) each fill blocks of their own kind. Mixed, an upload's staging sat between textures and a block emptied only when every texture
//   in it went too, so vk_trim gave back almost nothing: PowerVR #117 held 1,648 MB in blocks for 965 in use, Xclipse 530 1,344 / 854, Mali-G615 896 / 362 (Adreno
//   576 / 519; perf-1124). Apart, the staging blocks empty after a load and vk_trim returns them. VA_TEMP / VA_ARENA blocks are 32 MB (a request over 16 MB gets one
//   of its own). An empty block takes any kind (the one vk_trim keeps per memory type serves all). Nothing here frees earlier than before: a block goes back only
//   when everything in it was freed, and everything the GPU may still use is freed only after its fence (defer_add, arena_reset, once_reap)
enum { VA_LONG, VA_TEMP, VA_ARENA, VA_NCLS };
typedef struct { VkDeviceMemory mem; VkDeviceSize size; uint32_t type; void *map; int nfree, cfree; struct { VkDeviceSize off, size; } *fr; int used; uint64_t idle_since; int cls; } vblock;   // idle_since: when used fell to 0 (vk_trim); cls: VA_*
typedef struct { vblock *b; VkDeviceSize off, size; } vmalloc_t;
static vblock **g_blocks; static int g_nblocks; static pthread_mutex_t g_amu = PTHREAD_MUTEX_INITIALIZER;
static int mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < g_memp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_memp.memoryTypes[i].propertyFlags & want) == want && !(g_memp.memoryTypes[i].propertyFlags & avoid)) return (int)i;
    for (uint32_t i = 0; i < g_memp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_memp.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}
static void fr_add(vblock *b, VkDeviceSize off, VkDeviceSize size) {   // insert a free range keeping order, merging with its neighbours
    int i = 0; while (i < b->nfree && b->fr[i].off < off) i++;
    if (i > 0 && b->fr[i - 1].off + b->fr[i - 1].size == off) { b->fr[i - 1].size += size; if (i < b->nfree && off + size == b->fr[i].off) { b->fr[i - 1].size += b->fr[i].size; memmove(b->fr + i, b->fr + i + 1, (b->nfree - i - 1) * sizeof *b->fr); b->nfree--; } return; }
    if (i < b->nfree && off + size == b->fr[i].off) { b->fr[i].off = off; b->fr[i].size += size; return; }
    if (b->nfree == b->cfree) { b->cfree = b->cfree ? b->cfree * 2 : 16; b->fr = realloc(b->fr, b->cfree * sizeof *b->fr); }
    memmove(b->fr + i + 1, b->fr + i, (b->nfree - i) * sizeof *b->fr); b->fr[i].off = off; b->fr[i].size = size; b->nfree++;
}
// The engine's own device memory (android_main.c mem_by): the blocks vk_alloc has made, MB, and how much of them is handed out (images, buffers, staging); the game's
//   memory pools are not in here (they are the game's own pages, mapped for the GPU: /proc/self/smaps sees them)
void dm_vk_mem_stats(double *blocks_mb, double *used_mb) {
    double t = 0, u = 0; pthread_mutex_lock(&g_amu);
    for (int k = 0; k < g_nblocks; k++) { vblock *b = g_blocks[k]; double fr = 0; for (int i = 0; i < b->nfree; i++) fr += (double)b->fr[i].size; t += (double)b->size; u += (double)b->size - fr; }
    pthread_mutex_unlock(&g_amu); *blocks_mb = t / 1048576.0; *used_mb = u / 1048576.0;
}
static int vk_lifetime_on(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_MEM_LIFETIME"); v = !(e && *e == '0');
        VLOG("device memory: %s", v ? "blocks by lifetime (images, staging and arenas apart, so emptied staging blocks go back)" : "one kind of block for everything (DM_VK_MEM_LIFETIME=0)"); }
    return v;
}
static int vk_take(vblock *b, VkDeviceSize al, VkDeviceSize size, VkDeviceSize *off) {   // (g_amu held) first fit inside b: 1 and the offset when it fits
    for (int i = 0; i < b->nfree; i++) {
        VkDeviceSize o = (b->fr[i].off + al - 1) / al * al, end = b->fr[i].off + b->fr[i].size;
        if (o + size > end) continue;
        VkDeviceSize f0 = b->fr[i].off, f1 = end; memmove(b->fr + i, b->fr + i + 1, (b->nfree - i - 1) * sizeof *b->fr); b->nfree--;
        if (o > f0) fr_add(b, f0, o - f0);
        if (o + size < f1) fr_add(b, o + size, f1 - o - size);
        b->used++; *off = o; return 1;
    }
    return 0;
}
static vmalloc_t vk_alloc_c(VkMemoryRequirements rq, VkMemoryPropertyFlags want, int map, int cls) {
    int t = mem_type(rq.memoryTypeBits, want, map ? 0 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (t < 0) VDIE("no memory type for bits %#x flags %#x", rq.memoryTypeBits, want);
    VkDeviceSize gr = g_props.limits.bufferImageGranularity; while (gr & (gr - 1)) gr = (gr | (gr - 1)) + 1;   // (a power of two: one 2016 PowerVR driver reports 9990)
    VkDeviceSize al = rq.alignment > gr ? rq.alignment : gr;
    int life = vk_lifetime_on(); if (!life) cls = VA_LONG;
    pthread_mutex_lock(&g_amu);
    // its own kind first; then (blocks by lifetime) an empty block of another kind, which becomes this kind
    for (int pass = 0; pass < 1 + life; pass++)
        for (int k = 0; k < g_nblocks; k++) {
            vblock *b = g_blocks[k]; VkDeviceSize o; if (b->type != (uint32_t)t || (map && !b->map)) continue;
            if (pass == 0 ? b->cls != cls : b->used != 0) continue;
            if (!vk_take(b, al, rq.size, &o)) continue;
            b->cls = cls; pthread_mutex_unlock(&g_amu);
            return (vmalloc_t){b, o, rq.size};
        }
    VkDeviceSize std = cls == VA_LONG ? 64ull << 20 : 32ull << 20;
    VkDeviceSize bs = rq.size > std / 2 ? (rq.size + 0xffff) & ~0xffffull : std;
    { static long cap = -2; if (cap == -2) { const char *e = getenv("DM_VK_MEM_CAP_MB"); cap = e ? atol(e) : -1; }   // test runs: stop before the machine runs out of memory
      double total = bs; for (int k = 0; k < g_nblocks; k++) total += g_blocks[k]->size;
      if (cap > 0 && total > cap * 1048576.0) { pthread_mutex_unlock(&g_amu); VDIE("device memory would pass DM_VK_MEM_CAP_MB=%ld (%.0f MB in %d blocks)", cap, total / 1048576, g_nblocks); } }
    vblock *b = calloc(1, sizeof *b); b->size = bs; b->type = (uint32_t)t; b->cls = cls;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bs, .memoryTypeIndex = (uint32_t)t };
    VkResult r = vkAllocateMemory(g_dev, &mai, NULL, &b->mem);
    if (r) { pthread_mutex_unlock(&g_amu); VDIE("vkAllocateMemory %llu bytes type %d failed: %d", (unsigned long long)bs, t, r); }
    if (g_memp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vkMapMemory(g_dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
    if (rq.size < bs) fr_add(b, rq.size, bs - rq.size);
    b->used = 1;
    g_blocks = realloc(g_blocks, (g_nblocks + 1) * sizeof *g_blocks); g_blocks[g_nblocks++] = b;
    pthread_mutex_unlock(&g_amu);
    if (getenv("DM_VK_BLOCKLOG")) VLOG("device block %d: %.0f MB type %d kind %d for a %.1f MB request from %p", g_nblocks, bs / 1048576.0, t, cls, rq.size / 1048576.0, __builtin_return_address(0));
    return (vmalloc_t){b, 0, rq.size};
}
static vmalloc_t vk_alloc(VkMemoryRequirements rq, VkMemoryPropertyFlags want, int map) { return vk_alloc_c(rq, want, map, VA_LONG); }   // (long-lived: images, kept buffers)
// Blocks and what is handed out of them, MB, by kind (VA_*): the [mem] line and the tests
static void vk_mem_by_kind(double blocks[VA_NCLS], double used[VA_NCLS]) {
    for (int c = 0; c < VA_NCLS; c++) blocks[c] = used[c] = 0;
    pthread_mutex_lock(&g_amu);
    for (int k = 0; k < g_nblocks; k++) { vblock *b = g_blocks[k]; double fr = 0; for (int i = 0; i < b->nfree; i++) fr += (double)b->fr[i].size;
        int c = b->cls >= 0 && b->cls < VA_NCLS ? b->cls : 0; blocks[c] += (double)b->size / 1048576.0; used[c] += ((double)b->size - fr) / 1048576.0; }
    pthread_mutex_unlock(&g_amu);
}
static void vk_free(vmalloc_t a) {
    if (!a.b) return;
    pthread_mutex_lock(&g_amu); fr_add(a.b, a.off, a.size); if (--a.b->used == 0) a.b->idle_since = mono_ns(); pthread_mutex_unlock(&g_amu);
}
// Blocks given back after a burst: a load fills blocks with upload buffers, copies and new textures, and an allocator that only ever takes blocks keeps that
//   peak for the rest of the session (after loads on a Mali, memory grew in 256 MB steps until Android closed the game in another port of a Switch game).
//   A block with nothing left in it for 10 s goes back to the driver; the first empty block of each memory type stays, so play does not free and take
//   blocks again all the time. Nothing in an empty block is in use: everything the GPU may still read is freed only once it is done (defer_add, arena_reset).
//   Every 300 presents; DM_VK_MEM_TRIM=0 keeps every block as before
static uint64_t g_trim_bytes; static int g_trim_n;
static void vk_trim(void) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_MEM_TRIM"); on = !(e && !strcmp(e, "0")); } if (!on) return;
    uint64_t now = mono_ns(); uint32_t kept = 0; VkDeviceSize freed = 0; int nf = 0;   // kept: memory types that still have one empty block
    pthread_mutex_lock(&g_amu);
    for (int k = 0; k < g_nblocks;) {
        vblock *b = g_blocks[k];
        if (b->used || now - b->idle_since < 10000000000ull) { k++; continue; }
        if (b->type < 32 && !(kept >> b->type & 1)) { kept |= 1u << b->type; k++; continue; }
        vkFreeMemory(g_dev, b->mem, NULL); freed += b->size; nf++;   // (a mapped block is unmapped by its freeing. The count sat inside this comment until 2026-11-24: the line below never printed)
        free(b->fr); free(b); g_blocks[k] = g_blocks[--g_nblocks];
    }
    pthread_mutex_unlock(&g_amu);
    if (nf) { g_trim_bytes += freed; g_trim_n += nf; static int said; if (said++ < 30) VLOG("device memory: %d empty block%s given back (%.0f MB; %.0f MB in all so far)", nf, nf == 1 ? "" : "s", freed / 1048576.0, g_trim_bytes / 1048576.0); }
}
// The device memory this engine holds in its blocks (MB), for the log's memory line (android_main.c log_mirror): a run Android ends for memory then shows whether
//   the growth was the engine's GPU memory or something else
long nvn_vk_device_mb(void) {
    double t = 0; pthread_mutex_lock(&g_amu); for (int k = 0; k < g_nblocks; k++) t += g_blocks[k]->size; pthread_mutex_unlock(&g_amu);
    return (long)(t / 1048576);
}

// ================= Deferred destruction: objects the GPU may still use go when the submissions recorded so far are done =================
enum { DF_IMAGE, DF_VIEW, DF_BUFFER, DF_BUFVIEW, DF_ALLOC, DF_PIPELINE, DF_FRAMEBUFFER };
typedef struct { int kind; uint64_t fence; union { VkImage img; VkImageView view; VkBuffer buf; VkBufferView bview; VkPipeline pipe; VkFramebuffer fb; } h; vmalloc_t a; } dfr;
static dfr *g_dfr; static int g_ndfr, g_cdfr; static pthread_mutex_t g_dfmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_fence;                  // last fence value handed to a submission (submit thread)
static _Atomic uint64_t g_done;           // last fence value the GPU completed (completion thread)
static void defer_add(dfr d) {
    d.fence = g_fence + 1;
    pthread_mutex_lock(&g_dfmu);
    if (g_ndfr == g_cdfr) { g_cdfr = g_cdfr ? g_cdfr * 2 : 256; g_dfr = realloc(g_dfr, g_cdfr * sizeof *g_dfr); }
    g_dfr[g_ndfr++] = d; pthread_mutex_unlock(&g_dfmu);
}
// View quarantine (Arm's own driver with dynamic rendering): vkCmdBeginRendering keeps state per image-view handle value, and a value the driver hands out again after
// vkDestroyImageView hits that stale entry - the pass can leave its attachment unwritten (Arm erratum 3786926, r49p1-r51; MaliBridge MK-06 saw it on a G720 r44p1 too, a
// stale picture once images were being made and destroyed). Destroyed views and images wait in a FIFO, views ahead of their images, so a handle value does not come back
// soon; the memory under them is freed at once (only handles are held). DM_VK_VIEW_QUARANTINE=n sets its size, 0 turns it off
static void fbc_forget(VkImageView v);   // (the framebuffers naming a view, destroyed just before it: render passes below)
typedef struct { uint64_t h; uint8_t img; } qent;
static qent *g_vq; static int g_vqcap = -1; static size_t g_vqhead, g_vqcount;
static int quarantine_on(void) {
    if (g_vqcap < 0) { const char *e = getenv("DM_VK_VIEW_QUARANTINE"); g_vqcap = quirk_note(Q_VIEW_QUARANTINE, e ? atoi(e) : quirk_default(Q_VIEW_QUARANTINE, &g_gpuid), e != NULL);
        if (g_vqcap > 0 && !(g_vq = calloc((size_t)g_vqcap, sizeof *g_vq))) g_vqcap = 0;
        if (g_vqcap > 0) VLOG("image views: the last %d destroyed views and images kept back (Arm's driver: a reused view handle hits stale rendering state)", g_vqcap); }
    return g_vqcap > 0;
}
static void q_put(int img, uint64_t h) {   // caller holds g_dfmu
    if (g_vqcount == (size_t)g_vqcap) { qent e = g_vq[g_vqhead]; g_vqhead = (g_vqhead + 1) % (size_t)g_vqcap; g_vqcount--;
        if (e.img) vkDestroyImage(g_dev, (VkImage)(uintptr_t)e.h, NULL); else { fbc_forget((VkImageView)(uintptr_t)e.h); vkDestroyImageView(g_dev, (VkImageView)(uintptr_t)e.h, NULL); } }
    g_vq[(g_vqhead + g_vqcount) % (size_t)g_vqcap] = (qent){ h, (uint8_t)img }; g_vqcount++;
}
static void defer_drain(void) {
    uint64_t done = atomic_load(&g_done); int qon = quarantine_on(); VkImage imgs[64]; int nimg = 0;
    pthread_mutex_lock(&g_dfmu);
    for (int i = 0; i < g_ndfr;) {
        dfr d = g_dfr[i];
        if (d.fence > done) { i++; continue; }
        if (qon && d.kind == DF_IMAGE && nimg == 64) { i++; continue; }   // (the next drain takes it: its views go into the FIFO ahead of it)
        g_dfr[i] = g_dfr[--g_ndfr];
        if (qon && d.kind == DF_VIEW) { q_put(0, (uint64_t)(uintptr_t)d.h.view); vk_free(d.a); continue; }
        if (qon && d.kind == DF_IMAGE) { imgs[nimg++] = d.h.img; vk_free(d.a); continue; }
        switch (d.kind) {
            case DF_IMAGE: vkDestroyImage(g_dev, d.h.img, NULL); break; case DF_VIEW: fbc_forget(d.h.view); vkDestroyImageView(g_dev, d.h.view, NULL); break;
            case DF_BUFFER: vkDestroyBuffer(g_dev, d.h.buf, NULL); break; case DF_BUFVIEW: vkDestroyBufferView(g_dev, d.h.bview, NULL); break;
            case DF_PIPELINE: vkDestroyPipeline(g_dev, d.h.pipe, NULL); break; case DF_ALLOC: break;
            case DF_FRAMEBUFFER: vkDestroyFramebuffer(g_dev, d.h.fb, NULL); break;
        }
        vk_free(d.a);
    }
    for (int k = 0; k < nimg; k++) q_put(1, (uint64_t)(uintptr_t)imgs[k]);   // (after every view of this drain: a view never outlives its image's handle)
    pthread_mutex_unlock(&g_dfmu);
}

// ================= Memory pools: GPU buffers over the game's pages =================
// Three ways a pool's pages and the GPU share memory: imported host memory (VK_EXT_external_memory_host), the driver's memory moved onto the pool (mremap, needs every
// allocation to be a whole mapping of its own: Adreno), or a mirror (shadow): the GPU buffer in the driver's memory, the game's pages unchanged, the pages the game wrote
// copied in before each submission and what the GPU wrote copied back once it is done (drivers that sub-allocate their mappings: the Android emulator's; DM_VK_POOL_SHADOW=1)
// Only pools the GPU reads as buffers get one (fix 5 of the 2026-11-24 memory analysis): every pool with storage used to, 850-1,100 MB of pinned driver memory on every
//   phone (POCO F6: 926-964 MB in 2,000-2,300 pools), while textures are uploaded from the game's pages by the CPU (tex_upload_) and render targets and command memory
//   are never read by the GPU from the pool at all. A pool is recorded unbacked (buf VK_NULL_HANDLE, plain game memory that can be swapped) and backed on the first
//   pool_buf that finds it (pool_back, late = 1). DM_VK_POOL_GPU_ALL=1: every pool backed when it is made, as before
// (flags: the game's NVN pool flags, -1 unknown; used_buf / used_tex: ever bound as a buffer / read as texture data, for the census line, mem_report)
typedef struct { uintptr_t lo, hi, base; VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; const void *key; void *moved_from; uint8_t *shadow; uint64_t stamp; int tracked_here, fdmapped;
                 int flags; uint8_t backed, used_buf, used_tex, late; } pool_ent;
static _Atomic uint64_t g_gfx_bind_gen;   // graphics work recorded besides draw_'s (and new pools): draw_ sends its whole state again (see g_cb_epoch, mtl_sub_begin)
static pool_ent *g_pools; static int g_npool, g_cpool; static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_int g_nshadow;   // mirrored pools alive
static int g_plast = -1;
// The pools by base address for pool_find's binary search (pool_index.h), kept in step with g_pools under g_mu. DM_VK_POOL_LINEAR=1 goes back to the linear scan
#define PS_BASE(i) (g_pools[i].base)
#define PS_SIZE(i) (g_pools[i].size)
#include "pool_index.h"
#include "pool_pages.h"
// threads named for the load reports and profilers (android_main.c load_report): "Thread-4" for all of them said nothing
#ifdef __linux__
#include <sys/prctl.h>
#define DM_THREAD_NAME(n) prctl(PR_SET_NAME, (n), 0, 0, 0)
#else
#define DM_THREAD_NAME(n) ((void)0)
#endif
// S5 (perf-1124): the device, its self-tests, the pipeline cache's read and the prewarm started on a helper thread as soon as the crash handler is in (host/main.c
//   host_preload, before the game's modules are mapped), so they overlap the loader and the game's boot instead of starting at the game's first NVN memory pool.
//   Nothing in mtl_init needs the window (the surface and swapchain are made at the first present, as before), and every setting it reads (env.txt, the safe-start
//   level, the driver check's settings, boot_state's crash notes) is in the environment before the host thread starts (android_main.c host_start). Off by default
//   until it has run on phones: DM_VK_EARLY_INIT=1 turns it on
static void *early_main(void *a) {
    (void)a; DM_THREAD_NAME("dm-vkinit");
    { extern void crash_thread_init(void) __attribute__((weak)); if (crash_thread_init) crash_thread_init(); }   // (the crash handler's own stack on this thread too)
    dm_boot_mark("early graphics set-up starts (DM_VK_EARLY_INIT=1)");
    mtl_init();
    return NULL;
}
void nvn_vk_early_init(void) {
    const char *e = getenv("DM_VK_EARLY_INIT"); if (!(e && atoi(e) > 0)) return;
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 8u << 20);   // (mtl_init ran on the host thread's 64 MB stack: room for the self-tests)
    pthread_t t; if (!pthread_create(&t, &at, early_main, NULL)) pthread_detach(t); else VLOG("[RED] early graphics set-up: no thread, at the first memory pool as before");
    pthread_attr_destroy(&at);
}
// pages a freed pool could no longer read (pool_pages.h pages_copy, #281): said the first few times, with the running total
static void pages_lost_note(uintptr_t base) {
    static uint64_t said; static int n; uint64_t l = atomic_load(&pages_lost);
    if (l != said && n < 8) { n++; VLOG("[RED] memory pool %p: pages no longer readable when it was freed (the driver had taken them back): given back as zeros (%llu KB so far)", (void *)base, (unsigned long long)(l >> 10)); }
    said = l;
}
static int pp_hook(void *addr, int is_write, void *uc) { (void)is_write; (void)uc; return pp_fault(addr); }
static void pp_hook_once(void) { static atomic_int done; if (!atomic_exchange(&done, 1)) dm_fault_hook_register(pp_hook); }
// MREMAP_DONTUNMAP is tried until the kernel refuses it for the driver's mappings (errno 22, every log so far: Xclipse, Mali): a refused call has already unmapped its
// target, a gap another thread's new mapping could take before the next call replaces it, so it is not tried again
#ifdef __linux__
static atomic_int g_dontunmap = 1;
static void *mremap_keep_old(void *from, size_t n, void *to) {
    if (!atomic_load(&g_dontunmap)) { errno = EINVAL; return MAP_FAILED; }
#ifndef MREMAP_DONTUNMAP
#define MREMAP_DONTUNMAP 4
#endif
    void *q = mremap(from, n, n, MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP, to);
    if (q == MAP_FAILED && errno == EINVAL && atomic_exchange(&g_dontunmap, 0)) VLOG("pool memory: MREMAP_DONTUNMAP refused (errno 22): moving without it from now on");
    return q;
}
#endif
// Pool memory of the mremap route is never given back to the driver: Adreno hands the mappings of freed allocations out again, and one whose pages had been moved
// crashed the next pool (loading a save, leaving the house in Shield) or lost the device. A freed pool's buffer and memory, its pages moved back where the driver
// mapped them, wait here for the next pool of the same size
#ifdef __linux__
typedef struct { VkDeviceSize n; VkBuffer buf; VkDeviceMemory mem; void *p; } pool_mem;
static pool_mem g_pmem[1024]; static int g_npmem; static pthread_mutex_t g_pmmu = PTHREAD_MUTEX_INITIALIZER;
static int pool_mem_take(VkDeviceSize n, VkBuffer *buf, VkDeviceMemory *m, void **p) {
    pthread_mutex_lock(&g_pmmu); int ok = 0;
    for (int i = g_npmem - 1; i >= 0; i--) if (g_pmem[i].n == n) { *buf = g_pmem[i].buf; *m = g_pmem[i].mem; *p = g_pmem[i].p; g_pmem[i] = g_pmem[--g_npmem]; ok = 1; break; }
    pthread_mutex_unlock(&g_pmmu); return ok;
}
static int pool_mem_put(VkDeviceSize n, VkBuffer buf, VkDeviceMemory m, void *p) {
    pthread_mutex_lock(&g_pmmu); int ok = g_npmem < (int)(sizeof g_pmem / sizeof *g_pmem);
    if (ok) g_pmem[g_npmem++] = (pool_mem){ n, buf, m, p };
    pthread_mutex_unlock(&g_pmmu); return ok;
}
#endif
#ifdef __linux__
// The texture tracking's write protection splits a moved pool's range into several mappings, and a driver whose mappings cannot merge again keeps them split (Xclipse 940,
//   #54: every give-back of a texture pool failed with errno 14, then the game crashed next to such a pool). Each piece goes back on its own, to its place in the driver's
//   range (and a cached allocation given back so goes onto the next pool the same way); only when the pieces cover the range exactly and none reaches outside it (a neighbour's mapping is never moved)
static void *move_pieces(uintptr_t base, size_t size, void *to) {
    enum { MAXP = 256 }; uintptr_t s[MAXP], t[MAXP]; int k = 0, ok = 1; FILE *mf = fopen("/proc/self/maps", "r"); char ln[512];
    while (ok && mf && fgets(ln, sizeof ln, mf)) { unsigned long s0, e0;
        if (sscanf(ln, "%lx-%lx", &s0, &e0) != 2 || e0 <= base || s0 >= base + size) continue;
        if (k == MAXP || s0 < base || e0 > base + size || s0 != (k ? t[k - 1] : base)) ok = 0; else { s[k] = s0; t[k] = e0; k++; } }
    if (mf) fclose(mf);
    if (!ok || k < 2 || t[k - 1] != base + size) return MAP_FAILED;
    for (int i = 0; i < k; i++) { char *d = (char *)to + (s[i] - base);
        if (mremap((void *)s[i], t[i] - s[i], t[i] - s[i], MREMAP_MAYMOVE | MREMAP_FIXED, d) != d) return MAP_FAILED; }
    static int said; if (!said++) VLOG("memory pools: a pool's GPU memory moved in %d pieces (its mapping had been split)", k);
    return to;
}
#endif
// Pools backed only once the GPU reads one as a buffer (pool_ent above). Off with DM_VK_POOL_GPU_ALL=1, and where the late move cannot be made safe: without the write
//   tracking's page faults (DM_NO_DIRTY, the Android emulator, whose translation cannot deliver them) the other threads' writes could not be held back during it
static int pool_lazy(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_POOL_GPU_ALL");
#ifdef __linux__
        v = !(e && atoi(e)) && !nvn_dirty_off();
#else
        v = 0;
#endif
        VLOG("memory pools: %s", v ? "GPU memory only for pools the GPU reads as buffers (vertex, index, uniform, storage, indirect, copies); texture-only pools stay the game's own memory (DM_VK_POOL_GPU_ALL=1: every pool)"
                                   : e && atoi(e) ? "every pool backed by GPU memory (DM_VK_POOL_GPU_ALL=1)" : "every pool backed by GPU memory (no write tracking here: pools are not moved later)"); }
    return v;
}
// A pool's GPU buffer and memory (the routes below). late: the pool already exists and the game's other threads may be writing it (pool_buf's first hit); the range is
//   frozen for the time (nvn_dirty_freeze: their writes and protection changes wait), the driver's pages arrive read-only and the thaw sets each page's protection from
//   the write tracking, so no write is lost between the copy and the move and none to a tracked texture page goes unseen. The GPU never had this pool before (no buffer
//   existed), so there is nothing of its to wait for. The import route moves nothing: its protected pages are made writable first (the import pins them with write
//   access; their textures are uploaded again once). A late pool the tracking does not hold (its table full) is not added to it on the mirror routes: a pool being
//   freed holds the table's lock while it waits for the thaw (nvn_dirty_remove_range); untracked, the mirror copies the whole pool at each submission instead
static void pool_back(pool_ent *out) {
    pool_ent e = *out; const void *mem = e.key; uint64_t size = e.hi - e.lo; uintptr_t a = e.base; VkDeviceSize n = e.size; int late = e.late;
    void *p = NULL; int reused = 0;
#ifdef __linux__
    int prot = late ? PROT_READ : PROT_READ | PROT_WRITE;   // (how the driver's pages arrive over the pool's)
    if (late) { nvn_dirty_freeze((void *)a, n, !g_host_import); if (g_host_import) nvn_dirty_release((void *)a, n); }
#endif
#ifdef __linux__
    static int shadow_forced = -1; if (shadow_forced < 0) shadow_forced = getenv("DM_VK_POOL_SHADOW") && atoi(getenv("DM_VK_POOL_SHADOW"));
    // A freed pool's memory waiting for reuse is checked as a new allocation is (below): its mapping must still be readable and writable over the whole pool. Turnip
    //   (Adreno 660, #240) took such ranges back after 80 minutes of play, and the copy into one crashed (SIGSEGV in memcpy at a page-aligned address); one that is not
    //   usable is kept aside, never freed, and a new allocation made
    if (!g_host_import && !shadow_forced && !g_fdmap && !g_heapmap && pool_mem_take(n, &e.buf, &e.mem, &p)) {
        int rw = 0; FILE *mf = fopen("/proc/self/maps", "r"); char ln[512];
        while (mf && fgets(ln, sizeof ln, mf)) { unsigned long s0, e0; char pr[8] = "";
            if (sscanf(ln, "%lx-%lx %7s", &s0, &e0, pr) == 3 && (uintptr_t)p >= s0 && (uintptr_t)p < e0) { rw = pr[0] == 'r' && pr[1] == 'w' && e0 >= (uintptr_t)p + n; break; } }
        if (mf) fclose(mf);
        if (rw) { reused = 1; goto reuse; }
        RED_ONCE(20, "memory pool %p+%#llx: the freed GPU memory waiting for reuse at %p is no longer mapped writable: kept aside, a new allocation instead", mem, (unsigned long long)size, p);
        e.buf = VK_NULL_HANDLE; e.mem = VK_NULL_HANDLE; p = NULL;
    }
#endif
    VkExternalMemoryBufferCreateInfo eb = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT };
    VkExternalMemoryBufferCreateInfo ebfd = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = g_fdtype };
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = g_host_import ? &eb : (g_fdmap || g_heapmap) ? &ebfd : NULL, .size = n, .usage = BUF_ALL };
    VkResult br = vkCreateBuffer(g_dev, &bi, NULL, &e.buf);
#ifdef __linux__
    if (br == VK_ERROR_INVALID_EXTERNAL_HANDLE && g_heapmap < 0) {
        VLOG("[RED] dma-buf buffer rejected (%d): the heap route off", br); g_heapmap = 0; bi.pNext = NULL; br = vkCreateBuffer(g_dev, &bi, NULL, &e.buf);
    }
    if (br == VK_ERROR_INVALID_EXTERNAL_HANDLE && g_fdmap < 0) {
        VLOG("[RED] %s external buffer rejected (%d): moving driver mappings instead (mremap)", g_fdtype == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "dma-buf" : "opaque-fd", br);
        g_fdmap = 0;
        bi.pNext = NULL;
        br = vkCreateBuffer(g_dev, &bi, NULL, &e.buf);
    }
#endif
    if (br != VK_SUCCESS) VDIE("vkCreateBuffer(g_dev, &bi, NULL, &e.buf) failed: %d (%s:%d)", br, __FILE__, __LINE__);
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, e.buf, &rq);
    if (g_host_import) {
        VkMemoryHostPointerPropertiesEXT hp = { .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        p_hostprops(g_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)a, &hp);
        int t = mem_type(rq.memoryTypeBits & hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        VkImportMemoryHostPointerInfoEXT imp = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)a };
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = n, .memoryTypeIndex = (uint32_t)(t < 0 ? 0 : t) };
        VkResult r = t < 0 ? VK_ERROR_OUT_OF_HOST_MEMORY : vkAllocateMemory(g_dev, &mai, NULL, &e.mem);
        if (r) {   // a pool the import refuses (Pokémon Legends: Arceus on MoltenVK, a 344 KB pool, -2 about 25 s into loading a save) is mirrored instead, as on drivers without whole mappings
            RED_ONCE(60, "memory pool %p+%#llx: host memory import failed (%d): mirrored", mem, (unsigned long long)size, r);
            vkDestroyBuffer(g_dev, e.buf, NULL); bi.pNext = NULL; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &e.buf)); vkGetBufferMemoryRequirements(g_dev, e.buf, &rq);
            int ts = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
            VkMemoryAllocateInfo ma = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size > n ? rq.size : n, .memoryTypeIndex = (uint32_t)ts };
            VKCHECK(vkAllocateMemory(g_dev, &ma, NULL, &e.mem));
            void *p = NULL; VKCHECK(vkMapMemory(g_dev, e.mem, 0, VK_WHOLE_SIZE, 0, &p));
            e.shadow = p; atomic_fetch_add(&g_nshadow, 1);
            if (!late && !nvn_dirty_tracked((void *)a, n)) { nvn_dirty_add_range((void *)a, n); e.tracked_here = 1; }
            e.stamp = nvn_dirty_now(); nvn_dirty_protect((void *)a, n);
            memcpy(p, (void *)a, n);
        }
    } else {
        // The driver's own memory, preferably cached and coherent (Adreno type 0xf), filled with what the pool holds, then moved onto the pool's pages: from here on the game's
        // pointers and the GPU address the same memory. The driver's old mapping range gets a PROT_NONE placeholder, so nothing else lands where the driver believes its mapping is
        int t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
        if (t < 0) t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        // a dedicated allocation: a mapping of its own, never a piece of a bigger block the driver sub-allocates from (Adreno did that for small pools, 8 and 16 KB, and
        // handed out addresses inside blocks whose pages an earlier pool had moved away: its write to the placeholder crashed, loading a save and changing areas in Shield)
        VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = e.buf };
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded, .allocationSize = rq.size > n ? rq.size : n, .memoryTypeIndex = (uint32_t)t };
#ifdef __linux__
        // The fd route: the driver's own mapping is never touched (moving it broke Adreno: freed mappings handed out again, crashes and a lost device on area changes);
        // the same pages are mapped once more, shared, over the pool, so the game's pointers and the GPU still address the same memory without copies
        if (g_heapmap) {
            static PFN_vkGetMemoryFdPropertiesKHR fdprops; if (!fdprops) fdprops = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(g_dev, "vkGetMemoryFdPropertiesKHR");
            size_t pgs = (size_t)getpagesize(); VkDeviceSize hn = ((rq.size > n ? rq.size : n) + pgs - 1) & ~(VkDeviceSize)(pgs - 1);
            int first = g_heapmap < 0; if (first) heap_import_mark(1);   // a crash inside the driver importing the first one: the next start leaves this route out
            int fd = heap_alloc(hn), vfd = -1, why = 0; void *tmp = MAP_FAILED;
            VkMemoryFdPropertiesKHR fp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
            if (fd < 0) why = 1;
            else if ((tmp = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) why = 2;
            else if (!fdprops || fdprops(g_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp)) why = 3;
            else {
                uint32_t bits = rq.memoryTypeBits & fp.memoryTypeBits;
                int ti = mem_type(bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0); if (ti < 0) ti = mem_type(bits, 0, 0);
                VkImportMemoryFdInfoKHR im = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &ded, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
                VkMemoryAllocateInfo iai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &im, .allocationSize = hn, .memoryTypeIndex = (uint32_t)ti };
                if (ti < 0) why = 4;
                else if ((im.fd = vfd = dup(fd)) < 0) why = 5;
                else if (vkAllocateMemory(g_dev, &iai, NULL, &e.mem)) { close(vfd); why = 6; }   // (the driver owns the fd only when the import succeeds)
            }
            if (!why) {
                memcpy(tmp, (void *)a, n);   // what the pool holds, into the dma-buf
                void *q = mmap((void *)a, n, prot, MAP_SHARED | MAP_FIXED, fd, 0);   // the pool's pages become the dma-buf, in one step
                munmap(tmp, n); close(fd);
                if (q != (void *)a) VDIE("memory pool %p+%#llx: mapping the dma-buf over it failed (errno %d)", mem, (unsigned long long)size, errno);
                e.fdmapped = 1; if (first) g_heap_pending = 1;   // (the marker stays until 20 s of play went by: mtl_present)
                if (g_heapmap < 0) { g_heapmap = 1; VLOG("memory pools: dma-bufs from %s imported into the driver and mapped over the pools' pages", g_heapname); }
                goto bind;
            }
            int err = errno;
            if (tmp != MAP_FAILED) munmap(tmp, n);
            if (fd >= 0) close(fd);
            // a plain buffer for the routes below (memory bound to a buffer made for dma-bufs must be a dma-buf)
            vkDestroyBuffer(g_dev, e.buf, NULL); bi.pNext = NULL; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &e.buf)); vkGetBufferMemoryRequirements(g_dev, e.buf, &rq);
            ded.buffer = e.buf; mai.allocationSize = rq.size > n ? rq.size : n;
            static const char *whys[] = { "", "allocating it", "mapping it", "reading its memory types", "no memory type both accept", "dup", "importing it" };
            if (first) heap_import_mark(-1);
            if (g_heapmap < 0) { g_heapmap = 0; VLOG("[RED] memory pools: the dma-buf heap route failed (%s, errno %d): moving the driver's mappings instead", whys[why], err); }
            else RED_ONCE(106, "memory pool %p+%#llx: dma-buf failed (%s, errno %d): this pool and any other that fails go the mremap route", mem, (unsigned long long)size, whys[why], err);
        }
        if (g_fdmap) {
            static PFN_vkGetMemoryFdKHR getfd; if (!getfd) getfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g_dev, "vkGetMemoryFdKHR");
            VkExportMemoryAllocateInfo ex = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded, .handleTypes = g_fdtype };
            VkMemoryAllocateInfo xai = mai; xai.pNext = &ex;
            int fd = -1, first = g_fdmap < 0; void *tmp = MAP_FAILED;
            if (first) fd_export_mark(1);   // until the first pool's export resolves: a crash inside the driver leaves it for the next start
            if (getfd && !vkAllocateMemory(g_dev, &xai, NULL, &e.mem)) {
                VkMemoryGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = e.mem, .handleType = g_fdtype };
                if (!getfd(g_dev, &gi, &fd) && fd >= 0) tmp = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (tmp != MAP_FAILED) {
                    memcpy(tmp, (void *)a, n);   // what the pool holds, into the GPU memory
                    void *q = mmap((void *)a, n, prot, MAP_SHARED | MAP_FIXED, fd, 0);   // the pool's pages become the GPU memory, in one step
                    munmap(tmp, n);
                    if (q == (void *)a) {
                        close(fd); e.fdmapped = 1;
                        if (first) fd_export_mark(-1);
                        if (g_fdmap < 0) { g_fdmap = 1; VLOG("memory pools: the driver's memory exported (%s) and mapped over the pool's pages", g_fdtype == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "dma-buf" : "opaque fd"); }
                        goto bind;
                    }
                    if (g_fdmap > 0) VDIE("memory pool %p+%#llx: mapping the GPU memory over it failed (errno %d)", mem, (unsigned long long)size, errno);
                }
                if (fd >= 0) close(fd);
                vkFreeMemory(g_dev, e.mem, NULL); e.mem = VK_NULL_HANDLE;
            }
            // first pool: fall back for good (the buffer was made for export, which the plain allocation below also accepts); a later pool whose export fails (the pool's
            // pages untouched so far) takes the route below on its own
            if (g_fdmap > 0) RED_ONCE(103, "memory pool %p+%#llx: GPU memory export failed: this pool and any other that fails go the mremap route", mem, (unsigned long long)size);
            else { VLOG("memory pools: the driver's memory cannot be exported and mapped (errno %d): moving its mappings instead (mremap)", errno); g_fdmap = 0; }
            if (first) fd_export_mark(-1);
        }
#endif
#ifdef __linux__
        // The mapping the driver returns must be readable and writable over the whole pool. Adreno hands out the CPU mappings of freed allocations again, and one whose
        // pages a pool had moved away (a placeholder, or a range of the driver's that is gone) crashed the first write here (loading a save, changing areas in Shield).
        // Such an allocation is kept aside (never freed, so it is never handed out again) and another one made
        uintptr_t vs = 0, ve = 0;
        for (int tries = 0;; tries++) {
            VKCHECK(vkAllocateMemory(g_dev, &mai, NULL, &e.mem));
            VKCHECK(vkMapMemory(g_dev, e.mem, 0, VK_WHOLE_SIZE, 0, &p));
            int rw = 0; vs = ve = 0; FILE *mf = fopen("/proc/self/maps", "r"); char ln[512], mline[160] = "";
            while (mf && fgets(ln, sizeof ln, mf)) { unsigned long s0, e0; char pr[8] = "";
                if (sscanf(ln, "%lx-%lx %7s", &s0, &e0, pr) == 3 && (uintptr_t)p >= s0 && (uintptr_t)p < e0) { vs = s0; ve = e0; rw = pr[0] == 'r' && pr[1] == 'w';
                    snprintf(mline, sizeof mline, "%s", ln); mline[strcspn(mline, "\n")] = 0; break; } }
            if (mf) fclose(mf);
            if (rw && ve >= (uintptr_t)p + n) break;
            // The Android emulator's GPU (gfxstream, the app under arm64 translation) hands out mappings that /proc/self/maps lists without rw (or not at all);
            //   keeping each aside (313 MB) used up its memory window and the third allocation crashed inside gfxstream. Tried directly instead: the first and last
            //   byte read and written back unchanged (process_vm_*: a fault is an error, not a signal). Usable then, always mirrored (vs = 0: the shadow route below,
            //   never mremap, which would carry the listed protection along)
            {   unsigned char b0, b1; struct iovec l0 = { &b0, 1 }, l1 = { &b1, 1 }, r0 = { p, 1 }, r1 = { (uint8_t *)p + n - 1, 1 };
                int rd_ok = process_vm_readv(getpid(), &l0, 1, &r0, 1, 0) == 1 && process_vm_readv(getpid(), &l1, 1, &r1, 1, 0) == 1;
                int wr_ok = rd_ok && process_vm_writev(getpid(), &l0, 1, &r0, 1, 0) == 1 && process_vm_writev(getpid(), &l1, 1, &r1, 1, 0) == 1;
                static int said; if (said++ < 3) VLOG("memory pools: the driver's mapping %p: %s; read test %s, write test %s (errno %d)", p, vs ? mline : "not in /proc/self/maps",
                                                     rd_ok ? "passed" : "failed", wr_ok ? "passed: used, mirrored" : "failed", errno);
                if (wr_ok) { vs = ve = 0; break; }
                // The emulator's gfxstream maps its host-visible memory write-only ("-w-s /dev/goldfish_address_space"), a device mapping that the process_vm calls
                //   cannot reach (EFAULT either way). The CPU underneath is x86, where a writable page is readable too, so the game's code can use it: taken, mirrored.
                //   (Only that device: phones never have it.)
                char pr[8] = ""; if (vs && sscanf(mline, "%*lx-%*lx %7s", pr) == 1 && pr[1] == 'w' && strstr(mline, "/dev/goldfish_address_space") && ve >= (uintptr_t)p + n) {
                    // moved onto the pool when it starts at p (mremap moves the pool's part and leaves the mapping's last page behind): no copies, no page tracking,
                    //   which the translation cannot do (it cannot deliver the write faults); else mirrored
                    int whole = vs == (uintptr_t)p && !((uintptr_t)p & (getpagesize() - 1));
                    static int said2; if (!said2++) VLOG("memory pools: the emulator's write-only GPU mappings (/dev/goldfish_address_space, %s) used, %s", pr, whole ? "moved onto the pools" : "mirrored");
                    if (whole) ve = (uintptr_t)p + n; else vs = ve = 0;
                    break; } }
            VLOG("[RED] memory pool %p+%#llx: the driver's mapping %p is not usable (%s): kept aside, allocating again", mem, (unsigned long long)size, p, rw ? "too short" : vs ? "not writable" : "not in the memory map, failed the read / write test");
            if (tries >= 16) VDIE("memory pool %p+%#llx: no usable GPU memory mapping after %d tries", mem, (unsigned long long)size, tries + 1);
        }
        {   // mremap moves whole mappings: the driver's must start at p, be page-aligned, and end exactly after the pool
            static int force = -1; if (force < 0) force = getenv("DM_VK_POOL_SHADOW") && atoi(getenv("DM_VK_POOL_SHADOW"));
            if (force || ((uintptr_t)p & (getpagesize() - 1)) || vs != (uintptr_t)p || ve != (uintptr_t)p + n) {
                static int said; if (!said++) VLOG("memory pools: %s: mirrored (pages the game writes copied in before each submission)", force ? "DM_VK_POOL_SHADOW" : "the driver's mappings are not whole mappings of their own");
                e.shadow = p; atomic_fetch_add(&g_nshadow, 1);
                if (!late && !nvn_dirty_tracked((void *)a, n)) { nvn_dirty_add_range((void *)a, n); e.tracked_here = 1; }
                e.stamp = nvn_dirty_now(); nvn_dirty_protect((void *)a, n);   // later writes fault and get a newer generation; then the copy
                memcpy(p, (void *)a, n);
                goto bind;
            }
        }
#else
        VKCHECK(vkAllocateMemory(g_dev, &mai, NULL, &e.mem)); VKCHECK(vkMapMemory(g_dev, e.mem, 0, VK_WHOLE_SIZE, 0, &p));
#endif
#ifdef __linux__
reuse:
#endif
        memcpy(p, (void *)a, n);
#ifdef __linux__
        if (late) mprotect(p, n, PROT_READ);   // (arrives read-only: the thaw decides each page)
        // MREMAP_DONTUNMAP leaves the driver's old range mapped (empty) in the same step, so no other thread's mapping can land in a gap there before the placeholder;
        // where the kernel refuses it for this mapping, the plain move and the placeholder right after
        void *q = mremap_keep_old(p, n, (void *)a);
        if (q == (void *)a) mmap(p, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);   // the empty old range becomes the placeholder (replaced in one step)
        else {
            q = mremap(p, n, n, MREMAP_MAYMOVE | MREMAP_FIXED, (void *)a);
            if (q != (void *)a) q = move_pieces((uintptr_t)p, n, (void *)a);
            if (q != (void *)a) {
                int err = errno;
                VLOG("[RED] memory pool %p+%#llx: moving the GPU memory onto it failed (errno %d): mirrored instead", mem, (unsigned long long)size, err);
                // a refused MREMAP_FIXED unmaps its target before it looks at the source: the pool's pages can be gone here; they come back from the copy made above
                if (late) {   // (built aside and moved in read-only: other threads must not write the pool before the copy is in)
                    void *c = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                    if (c == MAP_FAILED) VDIE("memory pool %p+%#llx: no memory for its pages (errno %d)", mem, (unsigned long long)size, errno);
                    memcpy(c, p, n); mprotect(c, n, PROT_READ);
                    if (mremap(c, n, n, MREMAP_MAYMOVE | MREMAP_FIXED, (void *)a) != (void *)a) { mmap((void *)a, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0); memcpy((void *)a, c, n); munmap(c, n); }
                    mprotect(p, n, PROT_READ | PROT_WRITE);   // (the mirror is written by the copies before each submission)
                } else { mmap((void *)a, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0); memcpy((void *)a, p, n); }
                e.shadow = p; atomic_fetch_add(&g_nshadow, 1);
                if (!late && !nvn_dirty_tracked((void *)a, n)) { nvn_dirty_add_range((void *)a, n); e.tracked_here = 1; }
                e.stamp = nvn_dirty_now(); nvn_dirty_protect((void *)a, n);
                goto bind;
            }
            mmap(p, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        }
        e.moved_from = p;
#else
        VDIE("memory pools need VK_EXT_external_memory_host on this platform");
#endif
    }
#ifdef __linux__
bind:
#endif
    if (!reused) VKCHECK(vkBindBufferMemory(g_dev, e.buf, e.mem, 0));
#ifdef __linux__
    if (late) nvn_dirty_thaw();
#endif
    e.backed = 1; *out = e;
}
static _Atomic uint64_t g_pool_late_n, g_pool_late_bytes, g_pool_late_ns;   // pools backed late, their bytes, the time it took (the census line)
void mtl_pool_add(const void *mem, uint64_t size) {
    mtl_init();
    uintptr_t pg = (uintptr_t)(g_host_import ? g_host_align : (VkDeviceSize)getpagesize()), lo = (uintptr_t)mem, hi = lo + size;
    uintptr_t a = lo & ~(pg - 1), b = (hi + pg - 1) & ~(pg - 1); VkDeviceSize n = b - a;
    pool_ent e = { lo, hi, a, VK_NULL_HANDLE, VK_NULL_HANDLE, n, mem, NULL, NULL, 0, 0 };
    e.flags = -1;
    if (!pool_lazy()) pool_back(&e);
    pthread_mutex_lock(&g_mu);
    if (g_npool == g_cpool) { g_cpool = g_cpool ? g_cpool * 2 : 64; g_pools = realloc(g_pools, g_cpool * sizeof *g_pools); }
    g_pools[g_npool++] = e; ps_add(g_npool - 1); g_plast = -1;
    g_gfx_bind_gen++;   // (a new pool can reuse a deleted one's buffer handle: draw_'s state shadow binds the streams again)
    pthread_mutex_unlock(&g_mu);
}
// The game's NVN flags of a pool just made (hle_nvn.c, after mtl_pool_add): for the census line only
void mtl_pool_flags(const void *mem, int flags) {
    pthread_mutex_lock(&g_mu);
    for (int i = g_npool - 1; i >= 0; i--) if (g_pools[i].key == mem) { g_pools[i].flags = flags; break; }
    pthread_mutex_unlock(&g_mu);
}
void mtl_pool_del(const void *mem) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].key == mem) {
        pool_ent e = g_pools[i]; ps_remove(i); int last = --g_npool; g_pools[i] = g_pools[last]; if (i != last) ps_moved(last, i); g_plast = -1;
        pthread_mutex_unlock(&g_mu);
        // The game gets its pages back as ordinary memory with the same content (the driver's pages there replaced in one step, never leaving a gap another thread's
        // mapping could take); the GPU buffer and its memory go once the GPU is done with them. The placeholder at the driver's old address is the driver's to unmap
        // (vkFreeMemory): unmapping that range here as well, after the free, raced other threads whose new mappings could already sit there (games free and make pools
        // all the time: 131 times in two minutes of Pokémon Shield's character creation)
        if (!e.backed) return;   // (never read by the GPU as a buffer: its pages always were the game's own)
        mtl_finish();
        g_gfx_bind_gen++;   // (its buffer handle goes: draw_'s state shadow forgets it, so a pool backed later with the same handle value is bound again, pool_use_locked)
        if (e.shadow) { atomic_fetch_sub(&g_nshadow, 1); if (e.tracked_here) nvn_dirty_remove_range((void *)e.base); }
        if (e.fdmapped) {   // the game's pages back as ordinary memory with the same content, in one step (pool_pages.h), then a plain free (the driver's own mapping was never touched)
            void *copy = pages_copy(e.base, e.size);
            if (copy == MAP_FAILED) VDIE("memory pool %p+%#llx: no memory to give its pages back (errno %d)", (void *)e.base, (unsigned long long)e.size, errno);
            pages_lost_note(e.base);
            pages_back(e.base, e.size, copy);
            vkDestroyBuffer(g_dev, e.buf, NULL); vkFreeMemory(g_dev, e.mem, NULL);
            return;
        }
        int keep_mem = 0;   // the allocation stays when its pages could not go back (the driver would hand out the placeholder again)
        if (e.moved_from) {
            // The driver's pages go back where the driver mapped them (over the placeholder) before the free: Adreno keeps freed allocations with their mappings and hands
            // them out again, so a later pool's vkMapMemory returned the placeholder's address and the first write there crashed (loading a save in Pokémon Shield)
            void *copy = pages_copy(e.base, e.size);   // (made first: the game's pages become it the moment the driver's leave, pool_pages.h)
            if (copy == MAP_FAILED) VDIE("memory pool %p+%#llx: no memory to give its pages back (errno %d)", (void *)e.base, (unsigned long long)e.size, errno);
            pages_lost_note(e.base);
#ifdef __linux__
            // one protection over the whole range first: the texture tracking's write protection splits it into several mappings, and mremap moves one (EFAULT otherwise)
            mprotect((void *)e.base, e.size, PROT_READ | PROT_WRITE);
            // A refused move has already unmapped its target (the placeholder at the driver's address): the placeholder goes back right away, before anything slower
            //   (move_pieces reads /proc/self/maps), so another thread's new mapping never lands where the next move would replace it
            #define PLACEHOLDER() mmap(e.moved_from, e.size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0)
            pp_hook_once(); int pp = pp_begin(e.base, e.size);   // (a game thread touching the range from here to pages_back waits for it: pool_pages.h)
            int tried = atomic_load(&g_dontunmap); void *q = mremap_keep_old((void *)e.base, e.size, e.moved_from);
            if (q != e.moved_from) { if (tried) PLACEHOLDER(); q = mremap((void *)e.base, e.size, e.size, MREMAP_MAYMOVE | MREMAP_FIXED, e.moved_from); }
            if (q != e.moved_from) { int err = errno; PLACEHOLDER(); q = move_pieces(e.base, e.size, e.moved_from); if (q != e.moved_from) errno = err; }
            if (q != e.moved_from) {
                int err = errno;
                pages_back(e.base, e.size, copy); pp_end(pp);   // the game's pages first (any of the driver's still there go with them)
                VLOG("[RED] memory pool %p: the GPU memory could not go back to the driver (errno %d): kept, not freed", (void *)e.base, err); keep_mem = 1;
                PLACEHOLDER();   // (over any piece that did go back: nothing else lands where the driver believes its mapping is)
            } else {
                pages_back(e.base, e.size, copy); pp_end(pp);
                mprotect(e.moved_from, e.size, PROT_READ | PROT_WRITE);   // pages the texture tracking had write-protected come back writable
            }
            #undef PLACEHOLDER
#else
            pages_back(e.base, e.size, copy);
#endif
        }
#ifdef __linux__
        if (e.moved_from && !keep_mem) { if (!pool_mem_put(e.size, e.buf, e.mem, e.moved_from)) vkDestroyBuffer(g_dev, e.buf, NULL); return; }   // kept for the next pool (full: kept aside)
        if (e.moved_from) { vkDestroyBuffer(g_dev, e.buf, NULL); return; }   // pages not back with the driver: never freed
#endif
        vkDestroyBuffer(g_dev, e.buf, NULL); if (!keep_mem) vkFreeMemory(g_dev, e.mem, NULL);
        return;
    }
    pthread_mutex_unlock(&g_mu);
}
static int pool_find_linear(uint64_t addr, uint64_t len) {
    for (int i = 0; i < g_npool; i++) if (addr >= g_pools[i].base && addr + len <= g_pools[i].base + g_pools[i].size) return i;
    return -1;
}
static int pool_find(uint64_t addr, uint64_t len) {   // caller holds g_mu
    if (g_plast >= 0 && g_plast < g_npool && addr >= g_pools[g_plast].base && addr + len <= g_pools[g_plast].base + g_pools[g_plast].size) return g_plast;
    static int linear = -1, checks; if (linear < 0) { const char *e = getenv("DM_VK_POOL_LINEAR"); linear = e && atoi(e); }
    if (linear || g_psix_n != g_npool) { int i = pool_find_linear(addr, len); return i >= 0 ? (g_plast = i) : -1; }
    int i = ps_find(addr, len);   // (binary search: the last pool whose base is at or below addr)
    if (checks < 4000) {   // the first lookups of a run checked against the scan: a difference turns the binary search off for this run (the scan's answer is used)
        int s = pool_find_linear(addr, len); checks++;
        if (s != i) { linear = 1; VLOG("[RED] memory pools: the sorted index disagreed with the scan for %#llx (%d against %d): the scan from now on", (unsigned long long)addr, i, s); }
        i = s;
    }
    return i >= 0 ? (g_plast = i) : -1;
}
// GPU buffer + offset of [addr, addr + len); 0 when it is in no pool
static _Atomic uint64_t g_snapc_wbgen;   // (the uniform snapshots' drop-everything generation: below, with the snapshots)
// (g_mu held) pool i is about to be read by the GPU as a buffer: backed now if it was not yet, and marked for the census. Every lookup that hands a pool's VkBuffer to a
//   command goes through here (pool_buf, and draw_'s vertex streams looked up under one lock): an unbacked pool's buf is VK_NULL_HANDLE
static void pool_use_locked(int i) {
    if (!g_pools[i].backed) {   // the first time the GPU reads this pool as a buffer: backed now (pool_back; under g_mu, so no other lookup sees it half made)
        uint64_t t0 = mono_ns(); g_pools[i].late = 1; pool_back(&g_pools[i]); uint64_t dt = mono_ns() - t0;
        uint64_t k = atomic_fetch_add(&g_pool_late_n, 1); atomic_fetch_add(&g_pool_late_bytes, g_pools[i].size); atomic_fetch_add(&g_pool_late_ns, dt);
        if (k < 3 || dt > 20000000) VLOG("memory pools: pool %p+%#llx (flags %#x) first read as a buffer: GPU memory now (%.2f ms)", (void *)g_pools[i].lo, (unsigned long long)(g_pools[i].hi - g_pools[i].lo), g_pools[i].flags, dt / 1e6);
        // A new buffer handle, which can be one a deleted pool had: draw_'s state shadow sends its streams again (as mtl_pool_add did when every pool was backed there).
        //   And the pool's pages were just replaced under the freeze, with writes held back until the thaw: the uniform snapshots' copies are dropped (snapc_get)
        g_gfx_bind_gen++; atomic_fetch_add(&g_snapc_wbgen, 1);
    }
    g_pools[i].used_buf = 1;
}
static int pool_buf(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *left) {
    GOLDEN_MEM(addr, len);
    pthread_mutex_lock(&g_mu); int i = pool_find(addr, len ? len : 1);
    if (i >= 0) { pool_use_locked(i); *buf = g_pools[i].buf; *off = addr - g_pools[i].base; if (left) *left = g_pools[i].base + g_pools[i].size - addr; }
    pthread_mutex_unlock(&g_mu);
    return i >= 0;
}
// Texture data read from a pool by the CPU (tex_upload): for the census line only
static void pool_mark_tex(const void *mem) {
    if (!mem) return;
    pthread_mutex_lock(&g_mu); int i = pool_find((uint64_t)(uintptr_t)mem, 1); if (i >= 0) g_pools[i].used_tex = 1; pthread_mutex_unlock(&g_mu);
}
// Mirrored pools: the pages the game wrote since the last submission go into the GPU's copy (called before every submission)
static void shadow_copy_run(uintptr_t a, size_t n, void *u) { pool_ent *e = u; nvn_dirty_protect((void *)a, n); memcpy(e->shadow + (a - e->base), (void *)a, n); }
static void shadow_sync(void) {
    if (!atomic_load(&g_nshadow)) return;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) { pool_ent *e = &g_pools[i]; if (!e->shadow) continue;
        uint64_t s0 = nvn_dirty_now(); nvn_dirty_runs((void *)e->base, e->size, e->stamp, shadow_copy_run, e); e->stamp = s0; }
    pthread_mutex_unlock(&g_mu);
}
// What the GPU writes into a mirrored pool (copies, fills, readbacks, writable storage buffers) is copied back to the game's pages when the submission is done
typedef struct { uint64_t addr, n; } wb_t;
static void wb_note(uint64_t addr, uint64_t n);
// (hle_nvn.c asks only whether shader code sits in a pool: a lookup that never backs one; buf is VK_NULL_HANDLE for a pool not backed yet)
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    GOLDEN_MEM(addr, len);
    pthread_mutex_lock(&g_mu); int i = pool_find(addr, len ? len : 1);
    if (i >= 0) { *buf = (void *)g_pools[i].buf; *off = addr - g_pools[i].base; }
    pthread_mutex_unlock(&g_mu);
    return i >= 0;
}
void mtl_alias_drop_pool(const void *pool);   // textures (section below)

// ================= Submissions =================
// A submission (mtl_sub_begin … mtl_sub_commit) records into its main command buffer; texture uploads and the first-use setup of new images go into its upload command buffer,
// submitted just before it. CPU-visible effects (event writes, counter reports) are collected and applied by the completion thread when the GPU has finished the submission,
// in submission order, before the fence value is published (games wait on either)
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; uint32_t q; } post_op;   // (q: the report's GPU timestamp query + 1, report_ts)
typedef struct { VkBuffer buf; vmalloc_t a; uint8_t *map; VkDeviceSize used, size; } arena_blk;
// A held submission whose stores are decided at its flush (DM_VK_FRAME_HOLD=2): its work in pieces, in order. A gap (no pass open: barriers, copies, dispatches) and each
//   pass's contents are secondary command buffers; the flush records one primary that runs the gaps and, around each pass's secondary, begins and ends the pass with
//   the store ops decided then. Vulkan's render pass compatibility leaves load / store ops out, so a pass's contents recorded against its first ops run under the
//   decided ones (dynamic rendering: formats and samples inherited, the same)
// (work: a gap's commands that do something, counted by the gap_* wrappers; a gap with none, or only the barrier that stands for the Switch's own order between passes
//   on the same targets, lets the passes on either side run as one: renderer phase 3, seg_record_prim)
// (ra / box: the part of the target the pass's contents can write, seg_area: ra 0 unknown (the whole target), 1 known and nothing yet, 2 known: box)
// (dsk / dsw / dst: a game pass that pass_end closed knows whether it wrote its depth / stencil attachment (P.dwrote / swrote, a clear by load op included) and whether
//   a draw in it tested either (P.dtest); dsk 0: not known (our own passes, a pass closed elsewhere), taken as written and tested: seg_record_prim's NONE ops)
typedef struct { int pass, fgp; VkCommandBuffer cb; VkRenderingInfo ri; VkRenderingAttachmentInfo ca[8], da, sa; uint32_t cf[8], dsfmt, samples, work; int ra; VkRect2D box; uint8_t dsk, dsw, dst; } seg_t;
typedef struct {
    VkCommandPool pool; VkCommandBuffer cb, ucb; int ucb_open; uint64_t fence;
    VkCommandBuffer prim; VkCommandBuffer *sec; int nsec, csec; int dec; seg_t *seg; int nseg, cseg; seg_t cur;   // (the held frame's pieces: above; prim the submission's own primary)
    post_op *post; int npost, cpost;
    arena_blk *ar; int nar, car;   // this submission's arena blocks (host-visible): driver segments, converted indices, upload staging
    wb_t *wb; int nwb, cwb;        // mirrored pools: ranges the GPU writes in this submission (copied back to the game's pages at completion)
    VkDescriptorPool *dp; int ndp, cdp;   // g_nopush: this submission's descriptor pools (dp[cdp] in use), reset when it is reused
    uint64_t frame;                 // the frame it belongs to (GPU time per frame, mtl_gpu_ms)
    uint32_t nrq, rq_pend;           // report_ts: timestamp queries used by its counter reports, and how many wait for the open pass to end
    VkFence vf;                     // without timeline semaphores (g_tlsem 0): the fence its submission signals, waited for by the completion thread
} sub_t;
#define NSUB 6
static sub_t g_subs[NSUB]; static sub_t *g_sub; static int g_sub_next;
// Uniform snapshots reused within a submission (DM_VK_UBO_SNAPSHOT on; DM_VK_UBO_SNAPSHOT_REUSE=0: a copy for every uniform binding of every draw, as before).
//   That copy cost the G615 MC6 on r44p1 about 10 ms of replay CPU a frame and 144 MB of arenas, against 24 MB without snapshots (poco-runs b7 / b10, 997 draws):
//   a scene or light block bound by hundreds of draws was copied hundreds of times. A copy is now bound again by later draws of the same submission that bind the
//   same address and length, as long as nothing wrote into the copied range since, so each draw still reads the game's memory as it was when that draw was recorded.
//   Dropped on every write the replay knows of: uniform updates written through (sub_update) and every range the GPU writes in this submission (wb_note: copies,
//   fills, transform feedback, writable storage buffers, texture-to-buffer copies, deferred updates landing; those land when the GPU runs, after every copy was
//   taken, so dropping them is only the safe side), the transform feedback check's marker, and the mirrored pools' write-backs of finished submissions (wb_apply, on
//   the completion thread: the whole table goes at the next lookup). Not dropped for the 16-byte counter reports and semaphore values the completion thread
//   writes (no game binds them as uniforms). Keyed by address and length, two ways a set, the one used least recently replaced (a draw's own object block, bound
//   once, must not push out the scene block every draw binds); cleared at each submission's start (the copies are in its arena)
enum { SNAPC_N = 512 };
static struct { uint64_t addr, len, gen, use; VkBuffer b; VkDeviceSize o; } g_snapc[SNAPC_N]; static uint64_t g_snapc_use;
static uint64_t g_snapc_gen = 1, g_snapc_lo = UINT64_MAX, g_snapc_hi, g_snapc_wbseen;   // (lo / hi: what the live entries span, for a quick miss in snapc_drop)
static _Atomic uint64_t g_snapc_wbgen;   // mirrored pools written back by wb_apply (the completion thread)
static uint64_t g_snap_copies, g_snap_reused, g_snap_bytes;   // copies made / bound again, bytes copied (the present line, per 300 presents)
static int g_snapc_on = -1;
static int snapc_on(void) { if (g_snapc_on < 0) { const char *e = getenv("DM_VK_UBO_SNAPSHOT_REUSE"); g_snapc_on = !(e && !strcmp(e, "0"));
    VLOG("uniform snapshots: %s", g_snapc_on ? "one copy per submission, bound again until its range is written (DM_VK_UBO_SNAPSHOT_REUSE=0: a copy per draw)" : "a copy per draw (DM_VK_UBO_SNAPSHOT_REUSE=0)"); } return g_snapc_on; }
static void snapc_clear(void) { g_snapc_gen++; g_snapc_lo = UINT64_MAX; g_snapc_hi = 0; }
static void snapc_drop(uint64_t lo, uint64_t n) {
    uint64_t hi = lo + n; if (!n || hi <= g_snapc_lo || lo >= g_snapc_hi) return;
    for (int i = 0; i < SNAPC_N; i++) if (g_snapc[i].gen == g_snapc_gen && lo < g_snapc[i].addr + g_snapc[i].len && g_snapc[i].addr < hi) g_snapc[i].gen = 0;
}
static uint32_t snapc_set(uint64_t a, uint64_t n) { uint64_t h = (a >> 4) * 0x9E3779B97F4A7C15ull ^ n * 0xbf58476d1ce4e5b9ull; h ^= h >> 31; return (uint32_t)(h >> 23) & (SNAPC_N - 2); }   // (even: the set's first way)
static int snapc_get(uint64_t a, uint64_t n, VkBuffer *b, VkDeviceSize *o) {
    uint64_t w = atomic_load(&g_snapc_wbgen); if (w != g_snapc_wbseen) { g_snapc_wbseen = w; snapc_clear(); return 0; }
    uint32_t s = snapc_set(a, n);
    for (uint32_t k = s; k < s + 2; k++) if (g_snapc[k].gen == g_snapc_gen && g_snapc[k].addr == a && g_snapc[k].len == n) {
        g_snapc[k].use = ++g_snapc_use; *b = g_snapc[k].b; *o = g_snapc[k].o; return 1; }
    return 0;
}
static void snapc_put(uint64_t a, uint64_t n, VkBuffer b, VkDeviceSize o) {
    uint32_t s = snapc_set(a, n); if (g_snapc[s].gen == g_snapc_gen && (g_snapc[s + 1].gen != g_snapc_gen || g_snapc[s + 1].use < g_snapc[s].use)) s++;
    g_snapc[s].addr = a; g_snapc[s].len = n; g_snapc[s].b = b; g_snapc[s].o = o; g_snapc[s].gen = g_snapc_gen; g_snapc[s].use = ++g_snapc_use;
    if (a < g_snapc_lo) g_snapc_lo = a; if (a + n > g_snapc_hi) g_snapc_hi = a + n;
}
static void wb_note(uint64_t addr, uint64_t n) {
    snapc_drop(addr, n);   // (the uniform snapshots' copies of it: above)
    if (!n || !g_sub || !atomic_load(&g_nshadow)) return;
    pthread_mutex_lock(&g_mu); int k = pool_find(addr, n); int sh = k >= 0 && g_pools[k].shadow; pthread_mutex_unlock(&g_mu);
    if (!sh) return;
    sub_t *s = g_sub; if (s->nwb == s->cwb) { s->cwb = s->cwb ? s->cwb * 2 : 64; s->wb = realloc(s->wb, sizeof *s->wb * (size_t)s->cwb); }
    s->wb[s->nwb++] = (wb_t){ addr, n };
}
static VkSemaphore g_tl;
static pthread_mutex_t g_cmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_ccv = PTHREAD_COND_INITIALIZER;
static sub_t *g_inflight[NSUB * 2]; static int g_nin;   // submitted, not yet completed, in submission order (under g_cmu)
static pthread_t g_sub_thread; static _Atomic int g_sub_thread_set;
static int on_sub_thread(void) { return atomic_load(&g_sub_thread_set) && pthread_equal(pthread_self(), g_sub_thread); }
static uint64_t g_frame_no;   // (defined again below, where it is used most: a tentative definition)
#include "vk_crumbs.h"   // GPU breadcrumbs: what each submission held, printed when the GPU fails

int dm_gpu_ts614(void);   // (hle_nvn.c: DM_GPU_TS=614, the Switch GPU's 614.4 MHz ticks)
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) g = mono_ns(); return dm_gpu_ts614() ? g * 384 / 625 : g * 12 / 625; }
// Counter reports with the GPU's own time (DM_VK_REPORT_TS=0: off). A report's timestamp is when the GPU reached it: Game Freak's engines time their GPU work by the
//   reports around it and lower their own 3D render scale when it runs long (Legends: Arceus main+0xcc35a4: GPU time / 33.3 ms, at 95 % one step lighter, up to 0.8 x 0.7;
//   Sword / Shield main+0xf6acb0, 0.8). Every report used to get the CPU's time when the whole submission was seen complete (post_apply), so two reports in one
//   submission were nearly equal, the game saw no GPU time and never lowered its scale. Now each report writes a Vulkan timestamp query where it is recorded, and one more
//   ends the submission (RQ_END); at completion a report's time is the completion time less the GPU time from it to the end, on the guest's 19.2 MHz clock. Differences
//   within a submission are the GPU's own; between submissions the anchor is the completion thread's wake-up (sub-millisecond). On tilers a timestamp inside a render
//   pass can take the pass out of tile memory (Adreno), so one recorded in an open pass is written as that pass ends (report_ts_flush): never inside a pass
enum { RQ_PER = 64, RQ_END = RQ_PER - 1 };   // queries per submission slot; the last one ends the submission
static VkQueryPool g_rtq; static double g_rtper; static uint64_t g_rt_last;   // (g_rt_last: reports never go back in time)
static void report_ts_init(void) {
    const char *e = getenv("DM_VK_REPORT_TS"); if (e && !strcmp(e, "0")) { VLOG("counter reports: CPU time at completion (DM_VK_REPORT_TS=0)"); return; }
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    if (g_qf >= nq || qp[g_qf].timestampValidBits < 48 || !(g_props.limits.timestampPeriod > 0)) { VLOG("counter reports: CPU time at completion (the queue has no usable timestamps)"); return; }
    VkQueryPoolCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = RQ_PER * NSUB };
    if (vkCreateQueryPool(g_dev, &qi, NULL, &g_rtq)) { g_rtq = VK_NULL_HANDLE; return; }
    g_rtper = g_props.limits.timestampPeriod; VLOG("counter reports: the GPU's own timestamps (%.2f ns a tick), written between passes", g_rtper);
}
static void report_ts_flush(VkCommandBuffer cb, sub_t *s) {   // outside any pass: the timestamps of the reports recorded in the pass that just ended
    for (; s->rq_pend; s->rq_pend--) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_rtq, (uint32_t)(s - g_subs) * RQ_PER + s->nrq - s->rq_pend);
}
static uint64_t report_time(sub_t *s, uint32_t q, uint64_t now_ticks) {   // a report's time (19.2 MHz ticks) from its query and the submission's end
    uint64_t t[2]; uint32_t base = (uint32_t)(s - g_subs) * RQ_PER;
    if (vkGetQueryPoolResults(g_dev, g_rtq, base + q - 1, 1, 8, &t[0], 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS ||
        vkGetQueryPoolResults(g_dev, g_rtq, base + RQ_END, 1, 8, &t[1], 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS || t[1] < t[0]) return now_ticks;
    uint64_t back = (uint64_t)((double)(t[1] - t[0]) * g_rtper * 12.0 / 625.0);   // ns → 19.2 MHz ticks
    uint64_t r = back < now_ticks ? now_ticks - back : now_ticks;
    if (r < g_rt_last) r = g_rt_last; g_rt_last = r;
    return r;
}
static void wb_apply(sub_t *s) {
    int wrote = 0;
    for (int i = 0; i < s->nwb; i++) {
        pthread_mutex_lock(&g_mu); int k = pool_find(s->wb[i].addr, s->wb[i].n); pool_ent e = k >= 0 ? g_pools[k] : (pool_ent){0}; pthread_mutex_unlock(&g_mu);
        if (k < 0 || !e.shadow) continue;
        dm_guest_write_prepare((void *)(uintptr_t)s->wb[i].addr, s->wb[i].n);
        memcpy((void *)(uintptr_t)s->wb[i].addr, e.shadow + (s->wb[i].addr - e.base), s->wb[i].n); wrote = 1;
    }
    if (wrote) atomic_fetch_add(&g_snapc_wbgen, 1);   // (after the writes: the recording thread's next snapshot lookup drops every copy)
    s->nwb = 0;
}
static void post_apply(sub_t *s) {
    wb_apply(s);
    for (int i = 0; i < s->npost; i++) {
        post_op *o = &s->post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else {   // counter report {value, timestamp}: timestamps on the guest's 19.2 MHz clock; occlusion counts are not measured yet (reported as visible)
            uint64_t t = gpu_ticks(); if (o->q && g_rtq) t = report_time(s, o->q, t); else if (t < g_rt_last) t = g_rt_last;   // (report_ts)
            uint64_t rec[2] = { o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 0xffffu : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? t : 0, t };
            memcpy((void *)o->addr, rec, sizeof rec);
        }
    }
    s->npost = 0;
}
// GPU time per frame: a timestamp at the start and the end of every submission's main command buffer, summed per frame once it completed (mtl_gpu_ms; the
// frame report's GPU column and the long-frame lines)
static VkQueryPool g_tsq; static double g_tsper;
enum { GRING = 64 };
static struct { _Atomic uint64_t frame, ns; _Atomic int pending; } g_gpu[GRING];
static void gpu_ts_done(sub_t *s) {
    if (!g_tsq) return;
    uint64_t t[2]; int i = (int)(s - g_subs);
    if (vkGetQueryPoolResults(g_dev, g_tsq, (uint32_t)i * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0]) atomic_fetch_add(&g_gpu[s->frame % GRING].ns, (uint64_t)((t[1] - t[0]) * g_tsper));
    atomic_fetch_sub(&g_gpu[s->frame % GRING].pending, 1);
}
// A GPU wait (a timeline semaphore, else a fence) that does not give up on an error a wait may not return: Arm's driver waits with ppoll() and does not restart it when a
// signal interrupts it, and the wait ends with VK_ERROR_INITIALIZATION_FAILED (-3) (MaliBridge MK-01: every 7-10 s on a Mali-G925 under Android's runtime). Such a wait is
// waited again (naps from 0.5 ms to 20 ms) until its own deadline, or 30 s for one without; success, a timeout at the deadline, a lost device or no memory come back as they
// are. On a driver that follows the spec this costs one comparison
static VkResult vk_wait(const VkSemaphoreWaitInfo *wi, VkFence f, uint64_t timeout) {
    uint64_t t0 = mono_ns(), left = timeout; useconds_t nap = 500;
    for (;;) {
        VkResult r = wi ? vkWaitSemaphores(g_dev, wi, left) : vkWaitForFences(g_dev, 1, &f, VK_TRUE, left);
        if (r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST || r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY) return r;
        uint64_t el = mono_ns() - t0;
        if (r == VK_TIMEOUT && (timeout != UINT64_MAX || el >= 30000000000ull)) return r;   // (an early timeout of a wait with no deadline: not done, waited again)
        if (r != VK_TIMEOUT) RED_ONCE(112, "a GPU wait ended with %d, which a wait may not return (Arm's driver: a signal interrupted it): waited again", (int)r);
        if (timeout != UINT64_MAX && el >= timeout) return VK_TIMEOUT;
        if (timeout == UINT64_MAX && el >= 30000000000ull) return r;
        usleep(nap); nap = nap * 2 > 20000 ? 20000 : nap * 2;
        left = timeout == UINT64_MAX ? UINT64_MAX : timeout - el;
    }
}
static void *completion_main(void *unused) { DM_THREAD_NAME("dm-complete");
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_cmu);
        while (!g_nin) pthread_cond_wait(&g_ccv, &g_cmu);
        sub_t *s = g_inflight[0]; uint64_t want = s->fence; VkFence vf = s->vf;
        pthread_mutex_unlock(&g_cmu);
        VkSemaphoreWaitInfo wi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_tl, .pValues = &want };
        VkResult r = g_tlsem ? vk_wait(&wi, VK_NULL_HANDLE, 2000000000ull) : vk_wait(NULL, vf, 2000000000ull);
        if (r == VK_TIMEOUT) {   // a submission 10 s late: the GPU hangs (no device lost yet); its breadcrumbs printed once
            static uint64_t late, late_t, told; uint64_t now = mono_ns();
            if (late != want) { late = want; late_t = now; } else if (told != want && now - late_t > 10000000000ull) { told = want; nvn_vk_crumbs_dump("GPU hang: a submission more than 10 s late"); }
            continue; }
        if (r) { nvn_vk_crumbs_dump("device lost"); VDIE("%s failed: %d (device lost?)", g_tlsem ? "vkWaitSemaphores" : "vkWaitForFences", r); }
        cr_completed(s); post_apply(s); gpu_ts_done(s);
        pthread_mutex_lock(&g_cmu);
        memmove(g_inflight, g_inflight + 1, (size_t)--g_nin * sizeof *g_inflight);
        atomic_store(&g_done, want);
        pthread_cond_broadcast(&g_ccv);
        pthread_mutex_unlock(&g_cmu);
    }
    return NULL;
}
static void completion_start(void) {
    VkSemaphoreTypeCreateInfo st = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE };
    VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st };
    // A driver that lists timeline semaphores and then fails to make one (Qualcomm's 615.77 package on an Adreno 650 with a Vulkan 1.1 device: error 5, the game
    //   stopped at start, #209) gets a fence for each submission instead, the path drivers without them take
    if (g_tlsem) { VkResult r = vkCreateSemaphore(g_dev, &sci, NULL, &g_tl);
        if (r != VK_SUCCESS) { g_tlsem = 0; g_tl = VK_NULL_HANDLE; VLOG("[RED] timeline semaphore not made (error %d): a fence for each submission instead", r); } }
    // The submission pools without the transient hint: with it the driver gave its memory back at every reset and mapped it again (Adreno 830, a Galaxy S25+:
    //   vkResetCommandPool was 13-16 % of the game thread, most of it munmap / mmap; without the hint it fell under 3 %, simpleperf 2026-11-09). Each pool is reset
    //   once per submission and reused, which is what the memory is kept for. DM_VK_POOL_TRANSIENT=1: the hint as before
    const char *pt = getenv("DM_VK_POOL_TRANSIENT"); int transient = pt && !strcmp(pt, "1");
    if (transient) VLOG("submission command pools with the transient hint (DM_VK_POOL_TRANSIENT=1)");
    for (int i = 0; i < NSUB; i++) {
        VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = transient ? VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 0, .queueFamilyIndex = g_qf };
        VKCHECK(vkCreateCommandPool(g_dev, &pci, NULL, &g_subs[i].pool));
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_subs[i].pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_subs[i].cb)); VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_subs[i].ucb));
    }
    {   uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
        // Two timestamps a submission (the GPU's time per frame): on by default, as the frame-rate guard steers the resolution by it (android_main.c graphics_auto: GPU- or
        //   CPU-bound), except on Mali and PowerVR (vk_quirks.h Q_GPU_TIME_OFF), whose timestamps around work on a tiler are least trustworthy (they keep the old trial
        //   steps). DM_VK_GPU_TIME=0 / 1 decides
        const char *gt = getenv("DM_VK_GPU_TIME"), *gg = getenv("DM_GRAPHICS");
        int want_ts = !quirk_note(Q_GPU_TIME_OFF, gt ? atoi(gt) == 0 : quirk_default(Q_GPU_TIME_OFF, &g_gpuid), gt != NULL); (void)gg;   // (the guard runs on most phones whatever the Graphics level: a top phone's profile says High)
#if DM_GAME_ENG_GPU_TIME_AUTO_ONLY   // (profile engine_quirks gpu_time_auto_only, Legends: Arceus: only where Graphics Auto steers by it, as its releases do)
        if (!gt) want_ts = want_ts && gg && !strcmp(gg, "auto");
#endif
        if (g_qf < nq && qp[g_qf].timestampValidBits && g_props.limits.timestampPeriod > 0 && want_ts) {
            VkQueryPoolCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * NSUB };
            if (vkCreateQueryPool(g_dev, &qi, NULL, &g_tsq)) g_tsq = VK_NULL_HANDLE; g_tsper = g_props.limits.timestampPeriod; } }
    report_ts_init();
    pthread_t t; pthread_create(&t, NULL, completion_main, NULL); pthread_detach(t);
}
uint64_t mtl_last_fence(void) { return g_fence; }
static void hold_flush(int why); static _Atomic uint64_t g_submitted; static _Atomic int g_nheld;
static int g_hp_scene_ok;   // (HUD-free: the frame being presented has its scene copy; mtl_present → present_window_locked)   // (the whole frame held: below)
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (atomic_load(&g_done) >= fence) return 1;
    if (fence > atomic_load(&g_submitted) && atomic_load(&g_nheld)) hold_flush(1);   // waited on (or polled): held work goes to the GPU now
    uint64_t end = timeout_ns == UINT64_MAX ? UINT64_MAX : mono_ns() + timeout_ns;
    pthread_mutex_lock(&g_cmu);
    while (atomic_load(&g_done) < fence) {
        if (end == UINT64_MAX) pthread_cond_wait(&g_ccv, &g_cmu);
        else { uint64_t now = mono_ns(); if (now >= end) break; struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); uint64_t ns = ts.tv_nsec + (end - now);
               ts.tv_sec += ns / 1000000000ull; ts.tv_nsec = ns % 1000000000ull; pthread_cond_timedwait(&g_ccv, &g_cmu, &ts); }
    }
    pthread_mutex_unlock(&g_cmu);
    return atomic_load(&g_done) >= fence;
}
void mtl_finish(void) { mtl_wait(g_fence, UINT64_MAX); defer_drain(); }
// Fence f's work is held for the next present (DM_VK_FRAME_HOLD): a wait on it would send the held frame early (hle_nvn.c / nvn_rq.h rq_sync_fence waits for the present instead)
int mtl_fence_held(uint64_t f) { return f > atomic_load(&g_submitted) && atomic_load(&g_nheld) > 0; }

// Arena: per-submission host-visible memory, 4 MB blocks (bigger requests get their own), 256-byte aligned; reclaimed when the submission's command pool is reset
static uint8_t *arena_alloc(sub_t *s, VkDeviceSize n, VkBuffer *buf, VkDeviceSize *off) {
    n = (n + 255) & ~(VkDeviceSize)255;
    arena_blk *b = s->nar ? &s->ar[s->nar - 1] : NULL;
    if (!b || b->used + n > b->size) {
        VkDeviceSize bs = n > (4u << 20) ? n : 4u << 20;
        for (int i = 0; i < s->nar; i++) if (s->ar[i].used == 0 && s->ar[i].size >= n) { b = &s->ar[i]; goto have; }   // never reached for now: blocks stay in order
        if (s->nar == s->car) { s->car = s->car ? s->car * 2 : 4; s->ar = realloc(s->ar, s->car * sizeof *s->ar); }
        b = &s->ar[s->nar++]; memset(b, 0, sizeof *b); b->size = bs;
        VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bs, .usage = BUF_ALL };
        VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &b->buf));
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, b->buf, &rq);
        b->a = vk_alloc_c(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, VA_ARENA);
        VKCHECK(vkBindBufferMemory(g_dev, b->buf, b->a.b->mem, b->a.off)); b->map = (uint8_t *)b->a.b->map + b->a.off;
    }
have:
    *buf = b->buf; *off = b->used; b->used += n;
    return b->map + *off;
}
static void arena_reset(sub_t *s) {   // the submission's GPU work is done: big blocks go, the first 4 MB block stays for the next use
    for (int i = 0; i < s->nar; i++) {
        if (i == 0 && s->ar[i].size == (4u << 20)) { s->ar[i].used = 0; continue; }
        vkDestroyBuffer(g_dev, s->ar[i].buf, NULL); vk_free(s->ar[i].a);
    }
    s->nar = s->nar && s->ar[0].size == (4u << 20) && s->ar[0].buf ? 1 : 0;
}
static VkCommandBuffer ucb(void);   // the upload command buffer of the current submission (or of a one-off one)
static void pass_end(void);
static int g_rt_stale;   // a target change kept the open pass, to be checked at the next draw or clear (rt_resolve, DM_VK_PASS_DEFER)
static void state_reset(void);
static uint64_t g_frame_no;
uint64_t g_tfchk_at, g_tfchk_frame; uint32_t g_tfchk_n;   // the transform feedback check (draw_, the first capture on Arm's driver)
static uint64_t g_dsc_gen = 1;   // descriptor pools reset (desc_flush's reuse: a set from an older generation is gone)
static uint64_t g_wait_ring_ns;   // DM_VK_PROF: waits for a free submission slot (the GPU 6 submissions behind)
_Atomic uint64_t g_st_slot_ns, g_st_submit_ns;   // the same, and the queue submissions, never reset (hle_nvn.c's long-frame log takes per-frame differences)
static int hold_dec(void); static void seg_gap_begin(sub_t *s);   // (the held frame's pieces: below)
static int g_nlc; static void lc_out_all(void);   // (lazy clears, perf-1124 GPU fix B: kept depth clears, written out by mtl_sub_commit; above pass_begin)
// What draw_ last sent into the command buffer it records into (pipeline, dynamic state, vertex / index buffers, pushed descriptors: its state shadow, below) is
//   known only while nothing else changed it. g_cb_epoch counts every command buffer begun for CB () (each submission's, each secondary of a held frame: Vulkan
//   starts each one with its state undefined, and a handle can come back from the pool); g_gfx_bind_gen counts the other graphics work recorded into one (zcopy_draw,
//   the overlay, scaling, frame generation: other pipelines with fewer dynamic states, other layouts). Either one changed: draw_ binds and sends everything again
static uint64_t g_cb_epoch = 1;
static int state_shadow_mode(void);
void mtl_sub_begin(void) {
    mtl_init();
    prewarm_wait();   // the first one: every listed pipeline is built before the game draws (DM_VK_PREWARM)
    if (g_sub) VDIE("submit re-entered");
    g_sub_thread = pthread_self(); atomic_store(&g_sub_thread_set, 1);
    sub_t *s = &g_subs[g_sub_next]; g_sub_next = (g_sub_next + 1) % NSUB;
    if (s->fence) { uint64_t t0 = mono_ns(); mtl_wait(s->fence, UINT64_MAX); uint64_t d = mono_ns() - t0; g_wait_ring_ns += d; atomic_fetch_add(&g_st_slot_ns, d); }
    defer_drain();
    VKCHECK(vkResetCommandPool(g_dev, s->pool, 0)); arena_reset(s);
    for (int i = 0; i < s->ndp; i++) vkResetDescriptorPool(g_dev, s->dp[i], 0); s->cdp = 0; g_dsc_gen++;   // (desc_flush's reused sets were in these pools)
    snapc_clear();   // (the uniform snapshots' copies were in the last submission's arena)
    s->npost = 0; s->ucb_open = 0;
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (!s->prim) s->prim = s->cb;   // (the submission's own primary: allocated with it)
    s->dec = hold_dec(); s->nsec = 0; s->nseg = 0;
    if (s->dec) seg_gap_begin(s); else { s->cb = s->prim; VKCHECK(vkBeginCommandBuffer(s->cb, &bi)); g_cb_epoch++; }
    s->nrq = 0; s->rq_pend = 0; if (g_rtq) vkCmdResetQueryPool(s->cb, g_rtq, (uint32_t)(s - g_subs) * RQ_PER, RQ_PER);   // (report_ts: in the primary or the first gap, before any of its writes)
    if (g_tsq) { int i = (int)(s - g_subs); s->frame = g_frame_no; unsigned k = s->frame % GRING;
        if (atomic_load(&g_gpu[k].frame) != s->frame) { atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].frame, s->frame); }
        atomic_fetch_add(&g_gpu[k].pending, 1);
        vkCmdResetQueryPool(s->cb, g_tsq, (uint32_t)i * 2, 2); vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, (uint32_t)i * 2); }
    cr_sub_begin(s);
    g_sub = s;
    g_bar_clean = g_after_pass = 0;
    static int inited; if (!inited) { inited = 1; state_reset(); }
}
// The end of a submission: everything the GPU wrote made visible to the CPU (Vulkan asks for this barrier before the fence / semaphore wait the CPU reads after;
//   without it the GPU's writes may still sit in its caches). Adreno and lavapipe showed the data anyway; on Mali the game read back stale data from the pools, the
//   shaders' buffer writes, texture copies to buffers and the mirrored pools' copies back (the scene correct for the first frames, then white: #47)
static void barrier_host(VkCommandBuffer cb) {
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
// Frame generation's camera guidance: the game's view-projection (column-major, its clip space) from a game-specific reader (loader.c), and the matrix from the
//   current frame's clip space to the frame before's, prev x inverse(cur), in double. DM_FG_CAMERA=0: off
int dm_cam_viewproj(float vp[16]) __attribute__((weak));
static int fg_cam_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_FG_CAMERA"); v = !(e && !strcmp(e, "0")); } return v; }
static int fg_cam_m(const float *prev, const float *cur, float *out) {
    double a[16], inv[16]; for (int i = 0; i < 16; i++) a[i] = cur[i];
    inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] - a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] + a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];
    double det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if (!(fabs(det) > 1e-20) || det != det) return 0;
    for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) {   // out = prev x inv (column-major: element (r, c) at c * 4 + r)
        double v = 0; for (int k = 0; k < 4; k++) v += (double)prev[k * 4 + r] * (inv[c * 4 + k] / det); out[c * 4 + r] = (float)v; if (v != v) return 0; }
    return 1;
}
// Barriers in the submission's command buffer (from the Arceus engine; on Adreno each full one waits for the GPU to go idle and flushes its caches):
//   g_bar_clean: the last thing recorded was a full barrier, so a barrier before the next pass orders nothing more (it is left out)
//   g_after_pass: the last thing recorded was the end of a draw pass; g_game_bar: the game called nvnCommandBufferBarrier since the last draw pass began.
// Between two draw passes with no game barrier the barrier waits only for what fragment work needs (barrier_pass): the next pass's vertex work and binning overlap the previous
// pass's rendering. NVN asks the game for a barrier before results of rendering are read as vertices, indices, indirect arguments or by a vertex shader, so only those reads
// rely on the game's own barriers. DM_VK_BARRIER=full: a full barrier before every pass as before (=relaxed: on for a game where it is off by default)
static uint64_t g_bar_full, g_bar_narrow, g_bar_none;   // barriers before draw passes (mtl_barrier_stats)
static uint64_t g_bar_full_game, g_bar_full_work, g_bar_bits_narrow, g_bar_bits_hist[8];   // full ones for a game barrier / after other work; game barriers narrowed by their bits; bits seen in full ones
int dm_ab(const char *key) __attribute__((weak));   // (android_main.c: a switch alternated in one run)
static int barrier_bits_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_BARRIER_BITS"); v = !(e && !strcmp(e, "0")); }
    int ab = dm_ab ? dm_ab("barbits") : -1; return ab >= 0 ? ab : v; }
// Qualcomm's own driver before Vulkan 1.3 (Adreno 6xx / 7xx phones on driver 1.1.128 and the like; not Turnip, which names itself "Turnip Adreno"): skinned and
//   swaying models explode into spikes there with barriers narrowed between draw passes (Arceus-Native-Android #34 #46 #59 on the Adreno 710, Galar #209 on the
//   Adreno 650), as on Xclipse until v0.39 made its barriers full (#31 → #48 #78). The vertex-stride fix and the stride canary did not change it; Turnip on the same
//   phone draws them right. So these drivers get a full barrier before each pass, as Xclipse, PowerVR and Mali (DM_VK_BARRIER=relaxed: narrowed again).
//   The rule is vk_quirks.h's Q_BARRIER_RELAXED (with gpu_is_old_qcom, which was old_qcom_driver here), the same for every game
static int barrier_relaxed(void) { static int v = -1; if (v < 0) { { const char *e = getenv("DM_VK_BARRIER");   // (relaxed by default on Adreno (Qualcomm, vendor 0x5143) for every game; Mali, Xclipse, PowerVR keep full barriers: on those, Legends: Arceus flickered with them relaxed, Arceus-Native-Android #20 #31 #32)
        int set = e && (!strcmp(e, "full") || !strcmp(e, "relaxed"));
        v = quirk_note(Q_BARRIER_RELAXED, set ? !strcmp(e, "relaxed") : quirk_default(Q_BARRIER_RELAXED, &g_gpuid), set); } VLOG("barriers before draw passes: %s", v ? "left out after a full one, narrowed between passes (DM_VK_BARRIER=full: a full one each)" : "a full one each"); } return v; }
static void barrier_all(VkCommandBuffer cb) {
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (g_sub && cb == g_sub->cb) { g_bar_clean = 1; g_after_pass = 0; }
}
static void barrier_before(VkCommandBuffer cb) { if (!(g_sub && cb == g_sub->cb && g_bar_clean && barrier_relaxed())) barrier_all(cb); }   // before work: left out after a full one
static void barrier_pass(VkCommandBuffer cb) {   // draw pass → draw pass: everything earlier done before the next pass's fragment work, its vertex work free to start
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
// ================= Render passes, for drivers without dynamic rendering (g_dynren 0) =================
// Every pass is recorded as dynamic rendering (VkRenderingInfo). Without it the same pass becomes a Vulkan 1.0 render pass: one subpass, the same attachments, layouts
//   and load / store ops, ordered like barrier_all on both sides. Render passes are cached by formats and ops (they name no image view, so they stay valid for good);
//   a framebuffer is made for each pass and destroyed with its submission (defer_add), so none outlives the views it names. Pipelines are made against a compatible
//   render pass (rp_for_pipeline: the same formats and samples; ops and layouts do not matter for compatibility). A colour slot with no view stays VK_ATTACHMENT_UNUSED
//   in both, as in the pipeline's colour format list.
typedef struct { uint32_t cfmt[8], dsfmt; uint8_t n, samples, optimal, cop[8], dop, sop, pad, cst[8], dst, sst, pad2[2]; } rpkey;   // cop / dop / sop: 1 load, 2 clear, 3 none (dop / sop: rp_load_code); optimal: COLOR_ATTACHMENT_OPTIMAL (else GENERAL); cst / dst / sst: 0 store, 1 don't care, 2 none (skip stores)
static nvn_hmap g_rp = {.klen = sizeof(rpkey)};
static pthread_mutex_t g_rpmu = PTHREAD_MUTEX_INITIALIZER;   // the submit thread (passes), the game's and the prewarm thread (pipelines), the present thread
static VkAttachmentStoreOp rp_store(uint8_t v) { return v == 1 ? VK_ATTACHMENT_STORE_OP_DONT_CARE : v == 2 ? VK_ATTACHMENT_STORE_OP_NONE : VK_ATTACHMENT_STORE_OP_STORE; }
static uint8_t rp_store_code(VkAttachmentStoreOp o) { return o == VK_ATTACHMENT_STORE_OP_DONT_CARE ? 1 : o == VK_ATTACHMENT_STORE_OP_NONE ? 2 : 0; }
// (a depth / stencil attachment's load op: an aspect not attached loads, as before; NONE only from seg_record_prim's ds_none, with the extension enabled)
static uint8_t rp_load_code(const VkRenderingAttachmentInfo *a) { return !a ? 1 : a->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 2 : a->loadOp == VK_ATTACHMENT_LOAD_OP_NONE_EXT ? 3 : 1; }
static VkAttachmentLoadOp rp_load(uint8_t v) { return v == 2 ? VK_ATTACHMENT_LOAD_OP_CLEAR : v == 3 ? VK_ATTACHMENT_LOAD_OP_NONE_EXT : VK_ATTACHMENT_LOAD_OP_LOAD; }
static VkRenderPass rp_get(const rpkey *k) {
    pthread_mutex_lock(&g_rpmu); VkRenderPass rp = (VkRenderPass)nvn_hm_get(&g_rp, k); pthread_mutex_unlock(&g_rpmu);
    if (rp) return rp;
    VkImageLayout L = k->optimal ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    VkSampleCountFlagBits sm = (VkSampleCountFlagBits)(k->samples ? k->samples : 1);
    VkAttachmentDescription a[9]; VkAttachmentReference cr[8], dr = { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL }; uint32_t na = 0;
    for (uint32_t i = 0; i < k->n && i < 8; i++) {
        if (!k->cfmt[i]) { cr[i] = (VkAttachmentReference){ VK_ATTACHMENT_UNUSED, L }; continue; }
        a[na] = (VkAttachmentDescription){ 0, (VkFormat)k->cfmt[i], sm, k->cop[i] == 2 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD, rp_store(k->cst[i]),
                                           VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, L, L };
        cr[i] = (VkAttachmentReference){ na++, L };
    }
    if (k->dsfmt) {
        a[na] = (VkAttachmentDescription){ 0, (VkFormat)k->dsfmt, sm, rp_load(k->dop), rp_store(k->dst),
                                           rp_load(k->sop), rp_store(k->sst), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL };
        dr = (VkAttachmentReference){ na++, VK_IMAGE_LAYOUT_GENERAL };
    }
    VkSubpassDescription sp = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = k->n, .pColorAttachments = k->n ? cr : NULL, .pDepthStencilAttachment = k->dsfmt ? &dr : NULL };
    VkSubpassDependency dep[2] = {
        { VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0 },
        { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0 } };
    // The order around a pass is the engine's own barriers before and after it (barrier_all / barrier_pass / barrier_before), as with dynamic rendering: the two full
    //   dependencies made every one of ~140 passes a frame wait for all work before and after it, the relaxed barriers included (Vulkan 1.0 path on a Galaxy S25+:
    //   GPU 44-46 ms a frame, the dynamic rendering path 19 ms there, 2026-11-20). DM_VK_RP_DEPS=full: the two dependencies as before
    static int full = -1; if (full < 0) { const char *e = getenv("DM_VK_RP_DEPS"); full = e && !strcmp(e, "full"); }
    VkRenderPassCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = na, .pAttachments = a, .subpassCount = 1, .pSubpasses = &sp, .dependencyCount = full ? 2u : 0u, .pDependencies = full ? dep : NULL };
    VkResult r = vkCreateRenderPass(g_dev, &ci, NULL, &rp);
    if (r) VDIE("render pass (%u colour, depth / stencil format %u, %u samples) failed: %d", (unsigned)k->n, k->dsfmt, (unsigned)k->samples, (int)r);
    pthread_mutex_lock(&g_rpmu);
    VkRenderPass q = (VkRenderPass)nvn_hm_get(&g_rp, k);
    if (q) { pthread_mutex_unlock(&g_rpmu); vkDestroyRenderPass(g_dev, rp, NULL); return q; }   // another thread made the same one first
    nvn_hm_put(&g_rp, k, (void *)rp);
    static int n; if (++n <= 64) VLOG("render pass %d: %u colour, depth / stencil format %u, %u samples", n, (unsigned)k->n, k->dsfmt, (unsigned)k->samples);
    pthread_mutex_unlock(&g_rpmu);
    return rp;
}
// the render pass a pipeline with these formats is made against (only formats and samples decide compatibility)
static VkRenderPass rp_for_pipeline(uint32_t ncol, const VkFormat *cf, uint32_t dsfmt, uint32_t samples, int optimal) {
    rpkey k; memset(&k, 0, sizeof k); k.n = (uint8_t)ncol; k.samples = (uint8_t)(samples ? samples : 1); k.optimal = (uint8_t)optimal; k.dsfmt = dsfmt; k.dop = k.sop = 1;
    for (uint32_t i = 0; i < ncol && i < 8; i++) { k.cfmt[i] = (uint32_t)cf[i]; k.cop[i] = 1; }
    return rp_get(&k);
}
// Swapchain images (the upscaler and the touch overlay draw into them): their framebuffers are kept until the image's view goes (scfb_forget), not made per frame
static struct { VkImageView v; VkRenderPass rp; uint32_t w, h; VkFramebuffer fb; } g_scfb[32]; static int g_nscfb;
static pthread_mutex_t g_scfbmu = PTHREAD_MUTEX_INITIALIZER;
static void scfb_forget(VkImageView v) {   // the view is being destroyed (the GPU is done with it)
    pthread_mutex_lock(&g_scfbmu);
    for (int i = 0; i < g_nscfb;) if (g_scfb[i].v == v) { vkDestroyFramebuffer(g_dev, g_scfb[i].fb, NULL); g_scfb[i] = g_scfb[--g_nscfb]; } else i++;
    pthread_mutex_unlock(&g_scfbmu);
}
// Framebuffers of the other passes, kept and found again by render pass, views and size (fb_cache.h). DM_VK_FB_CACHE=0: one framebuffer per pass, as before
static VkFramebuffer fbc_create(VkRenderPass rp, const VkImageView *views, uint32_t nv, uint32_t w, uint32_t h, uint32_t layers) {
    VkFramebufferCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp, .attachmentCount = nv, .pAttachments = views, .width = w, .height = h, .layers = layers };
    VkFramebuffer fb; VKCHECK(vkCreateFramebuffer(g_dev, &fi, NULL, &fb)); return fb;
}
#define FBC_CREATE(rp, v, nv, w, h, l) fbc_create(rp, v, nv, w, h, l)
#define FBC_DESTROY(x) vkDestroyFramebuffer(g_dev, (x), NULL)
#define FBC_DEFER(x) defer_add((dfr){ .kind = DF_FRAMEBUFFER, .h.fb = (x) })
#define FBC_N 1024   // (Sword / Shield use about 380 in the field: with 256 the oldest went and most passes made a framebuffer again, 174 a frame)
#include "fb_cache.h"
static int fbc_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_FB_CACHE"); on = !e || atoi(e) != 0; } return on; }
// Begins a pass described as dynamic rendering. cf: each colour slot's format (a slot with no view: ignored), dsfmt: the depth / stencil attachment's format,
//   samples: the attachments' sample count; swapchain: the one colour attachment is a swapchain image (framebuffer kept, see g_scfb)
static void rend_begin_ex(VkCommandBuffer cb, const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples, int swapchain, int secondary, const VkRect2D *area);
static int g_seg_in, g_seg_fgp = -1;   // a pass of a decided submission is open (its contents go to a secondary); the frame recorder's index for the pass about to begin
static VkRenderPass rp_compat(const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples);
static VkCommandBuffer seg_new_cb(sub_t *s) {
    if (s->nsec == s->csec) { int add = 32; s->sec = realloc(s->sec, sizeof *s->sec * (size_t)(s->csec + add));
        VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = s->pool, .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = (uint32_t)add };
        VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, s->sec + s->csec)); s->csec += add; }
    return s->sec[s->nsec++];
}
static void seg_push(sub_t *s, seg_t g) { if (s->nseg == s->cseg) { s->cseg = s->cseg ? s->cseg * 2 : 64; s->seg = realloc(s->seg, sizeof *s->seg * (size_t)s->cseg); } s->seg[s->nseg++] = g; }
static void seg_gap_begin(sub_t *s) {   // a new gap: work outside passes
    VkCommandBuffer cb = seg_new_cb(s); VkCommandBufferInheritanceInfo inh = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, .pInheritanceInfo = &inh };
    VKCHECK(vkBeginCommandBuffer(cb, &bi)); s->cb = cb; g_gap_cb = cb; g_gap_work = 0; g_cb_epoch++;
}
// ---- A held frame's render areas (fix 2 of perf-1124, DM_VK_RENDER_AREA=0 / Q_RENDER_AREA_FULL: off) ----
// Sword / Shield scale the viewport and scissor inside full-size targets (hle_resscale.c: the targets stay full size), and every pass began on the whole target with
//   LOAD / STORE: a tiler loads and stores every tile of it on each of the 67-88 passes a frame (#349, Adreno 613 at 50 %: GPU 69.5-73 ms for 288 draws), so a lower
//   scale saved shading but no tile traffic (#99 #315 #299). In a held frame the pass is begun at the flush (seg_record_prim), when its contents are known: its render
//   area is then the union of every rectangle its contents can write, rounded out to RA_ALIGN pixels. Vulkan restricts load, store, clear and resolve to the render
//   area and leaves the rest of each attachment as it was, which is what the whole-target LOAD / STORE gave it (the same pixels loaded and stored back unchanged), so
//   whatever reads the target later (the game sampling it, copies, mips, a menu at full size, the present) sees the same pixels. Rasterisation never leaves the
//   scissor, and every draw in a pass's secondary sets its own (dynamic state does not carry into a secondary), so the scissors set there and the rectangles cleared
//   (seg_area_note: draw_, clear_in_pass) are all the pass can write. Only the game's own passes (pass_begin) are measured; ours (copies, captures, the clear of a 3D
//   slice) keep the whole target. A pass that clears an attachment by its load op, resolves one or is multiview keeps the whole target too
#define RA_ALIGN 32   // (Adreno's bins and Mali's tiles: whole tiles, a smaller area that cuts one would still load and store it)
static int g_ra_next;   // pass_begin: the pass rend_begin is about to begin is the game's own (its area can be measured)
static uint64_t g_ra_px_full, g_ra_px, g_ra_n;   // passes begun at the flush: pixels of their targets, pixels of their render areas, passes with a smaller area
static int ra_on(void) {
    static int v = -1; if (v >= 0) return v;
    const char *e = getenv("DM_VK_RENDER_AREA"); int full = quirk_note(Q_RENDER_AREA_FULL, e ? !atoi(e) : quirk_default(Q_RENDER_AREA_FULL, &g_gpuid), vkq_env_src("DM_VK_RENDER_AREA"));
    v = !full; if (v && hold_dec()) VLOG("held frames: each pass's render area is the part of its targets its draws reach (DM_VK_RENDER_AREA=0: the whole target)");
    return v;
}
static int seg_area_ok(const VkRenderingInfo *ri) {   // the pass's own operations write nothing outside its draws: no clear by load op, no resolve, no views
    if (ri->viewMask) return 0;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) { const VkRenderingAttachmentInfo *c = &ri->pColorAttachments[i];
        if (c->imageView && (c->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR || c->resolveImageView || c->resolveMode)) return 0; }
    const VkRenderingAttachmentInfo *ds[2] = { ri->pDepthAttachment, ri->pStencilAttachment };
    for (int i = 0; i < 2; i++) if (ds[i] && ds[i]->imageView && (ds[i]->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR || ds[i]->resolveImageView || ds[i]->resolveMode)) return 0;
    return 1;
}
// A rectangle the open pass's contents can write (a scissor set in cb, a cleared rectangle): added to the held pass's area. Elsewhere nothing
static void seg_area_note(VkCommandBuffer cb, VkRect2D r) {
    if (!g_sub || !g_sub->dec || !g_seg_in || cb != g_sub->cb) return;
    seg_t *g = &g_sub->cur; if (!g->ra) return;
    if (!r.extent.width || !r.extent.height) return;
    if (g->ra == 1) { g->box = r; g->ra = 2; return; }
    int32_t x0 = g->box.offset.x < r.offset.x ? g->box.offset.x : r.offset.x, y0 = g->box.offset.y < r.offset.y ? g->box.offset.y : r.offset.y;
    int64_t x1 = (int64_t)g->box.offset.x + g->box.extent.width, y1 = (int64_t)g->box.offset.y + g->box.extent.height;
    if ((int64_t)r.offset.x + r.extent.width > x1) x1 = (int64_t)r.offset.x + r.extent.width;
    if ((int64_t)r.offset.y + r.extent.height > y1) y1 = (int64_t)r.offset.y + r.extent.height;
    g->box = (VkRect2D){ { x0, y0 }, { (uint32_t)(x1 - x0), (uint32_t)(y1 - y0) } };
}
// The render area for passes i..last (step 2: one run as one, seg_record_prim) of s, all on the same targets: their boxes' union rounded out to RA_ALIGN and kept
//   inside the target; the whole target when any of them is unknown, or drew nothing (an empty pass keeps what it did before: nothing to gain)
static VkRect2D seg_area(const sub_t *s, int i, int last) {
    const seg_t *g = &s->seg[i]; VkRect2D full = g->ri.renderArea, r = full;
    if (!ra_on()) return full;
    int have = 0; int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    for (int j = i; j <= last; j += 2) { const seg_t *h = &s->seg[j];
        if (h->ra != 2) return full;
        int64_t a0 = h->box.offset.x, b0 = h->box.offset.y, a1 = a0 + h->box.extent.width, b1 = b0 + h->box.extent.height;
        if (!have || a0 < x0) x0 = a0; if (!have || b0 < y0) y0 = b0; if (!have || a1 > x1) x1 = a1; if (!have || b1 > y1) y1 = b1; have = 1; }
    int64_t fx0 = full.offset.x, fy0 = full.offset.y, fx1 = fx0 + full.extent.width, fy1 = fy0 + full.extent.height;
    x0 = x0 / RA_ALIGN * RA_ALIGN; y0 = y0 / RA_ALIGN * RA_ALIGN; x1 = (x1 + RA_ALIGN - 1) / RA_ALIGN * RA_ALIGN; y1 = (y1 + RA_ALIGN - 1) / RA_ALIGN * RA_ALIGN;
    if (x0 < fx0) x0 = fx0; if (y0 < fy0) y0 = fy0; if (x1 > fx1) x1 = fx1; if (y1 > fy1) y1 = fy1;
    if (x1 <= x0 || y1 <= y0) return full;
    r = (VkRect2D){ { (int32_t)x0, (int32_t)y0 }, { (uint32_t)(x1 - x0), (uint32_t)(y1 - y0) } };
    return r;
}
static void seg_pass_begin(const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples, int game) {   // the gap ends; the pass's contents go to a secondary
    sub_t *s = g_sub; VKCHECK(vkEndCommandBuffer(s->cb)); seg_push(s, (seg_t){ .pass = 0, .fgp = -1, .cb = s->cb, .work = g_gap_work }); g_gap_cb = VK_NULL_HANDLE;
    seg_t *g = &s->cur; memset(g, 0, sizeof *g); g->pass = 1; g->fgp = g_seg_fgp; g->ri = *ri; g->dsfmt = dsfmt; g->samples = samples ? samples : 1;
    g->ra = game && seg_area_ok(ri) && ra_on() ? 1 : 0;   // (a game's pass: its area measured as its contents are recorded, seg_area_note)
    for (uint32_t i = 0; i < ri->colorAttachmentCount && i < 8; i++) { g->ca[i] = ri->pColorAttachments[i]; g->cf[i] = cf && ri->pColorAttachments[i].imageView ? cf[i] : 0; }
    if (ri->pDepthAttachment) g->da = *ri->pDepthAttachment;
    if (ri->pStencilAttachment) g->sa = *ri->pStencilAttachment;
    VkFormat fm[8]; for (int i = 0; i < 8; i++) fm[i] = (VkFormat)g->cf[i];
    VkCommandBufferInheritanceRenderingInfo ir = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, .colorAttachmentCount = ri->colorAttachmentCount, .pColorAttachmentFormats = fm,
        .depthAttachmentFormat = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? (VkFormat)dsfmt : VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = ri->pStencilAttachment && ri->pStencilAttachment->imageView ? (VkFormat)dsfmt : VK_FORMAT_UNDEFINED, .rasterizationSamples = (VkSampleCountFlagBits)g->samples };
    VkCommandBufferInheritanceInfo inh = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, .pNext = g_dynren ? &ir : NULL, .renderPass = g_dynren ? VK_NULL_HANDLE : rp_compat(ri, cf, dsfmt, samples) };
    VkCommandBuffer cb = seg_new_cb(s);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, .pInheritanceInfo = &inh };
    VKCHECK(vkBeginCommandBuffer(cb, &bi)); g->cb = cb; s->cb = cb; g_seg_in = 1; g_cb_epoch++;
}
static void rend_begin(VkCommandBuffer cb, const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples, int swapchain) {
    cr_pass_begin(cb, ri, cf, dsfmt);   // (breadcrumbs: the pass noted for its submission)
    // (a pass begun in the same primary keeps its state: Vulkan's command buffer state lives across render pass instances, as the pipeline binding draw_ already
    //   kept. DM_VK_STATE_SHADOW=2 sends everything again in each pass anyway: a test for a driver that loses state there)
    if (state_shadow_mode() == 2) g_cb_epoch++;
    int game = g_ra_next; g_ra_next = 0;
    if (g_sub && g_sub->dec && cb == g_sub->cb && !g_seg_in && !swapchain) { seg_pass_begin(ri, cf, dsfmt, samples, game); g_seg_fgp = -1; return; }
    g_seg_fgp = -1;
    rend_begin_ex(cb, ri, cf, dsfmt, samples, swapchain, 0, NULL);
}
// (area: the render area when it is smaller than ri's, seg_area; the framebuffer stays the size of ri's, so the cached ones serve both)
static void rend_begin_ex(VkCommandBuffer cb, const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples, int swapchain, int secondary, const VkRect2D *area) {
    if (g_dynren) { if (area) { VkRenderingInfo r2 = *ri; r2.renderArea = *area; vkCmdBeginRendering(cb, &r2); } else vkCmdBeginRendering(cb, ri); return; }
    rpkey k; memset(&k, 0, sizeof k); k.n = (uint8_t)ri->colorAttachmentCount; k.samples = (uint8_t)(samples ? samples : 1);
    VkImageView views[9]; VkClearValue cv[9]; uint32_t nv = 0;
    for (uint32_t i = 0; i < k.n && i < 8; i++) {
        const VkRenderingAttachmentInfo *c = &ri->pColorAttachments[i]; if (!c->imageView) continue;
        k.cfmt[i] = cf[i]; k.cop[i] = c->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 2 : 1; k.cst[i] = rp_store_code(c->storeOp); if (c->imageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) k.optimal = 1;
        cv[nv] = c->clearValue; views[nv++] = c->imageView;
    }
    const VkRenderingAttachmentInfo *d = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment : NULL,
                                    *s = ri->pStencilAttachment && ri->pStencilAttachment->imageView ? ri->pStencilAttachment : NULL;
    if (d || s) {   // (depth and stencil are aspects of one image and view here)
        k.dsfmt = dsfmt; k.dop = rp_load_code(d); k.sop = rp_load_code(s);
        k.dst = d ? rp_store_code(d->storeOp) : 1; k.sst = s ? rp_store_code(s->storeOp) : 1;   // (an aspect not attached: nothing to keep)
        memset(&cv[nv], 0, sizeof cv[nv]); cv[nv].depthStencil.depth = d ? d->clearValue.depthStencil.depth : 0; cv[nv].depthStencil.stencil = s ? s->clearValue.depthStencil.stencil : 0;
        views[nv++] = d ? d->imageView : s->imageView;
    }
    VkRenderPass rp = rp_get(&k);
    uint32_t w = ri->renderArea.offset.x + ri->renderArea.extent.width, h = ri->renderArea.offset.y + ri->renderArea.extent.height;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (swapchain && nv == 1) {
        pthread_mutex_lock(&g_scfbmu);
        for (int i = 0; i < g_nscfb && !fb; i++) if (g_scfb[i].v == views[0] && g_scfb[i].rp == rp && g_scfb[i].w == w && g_scfb[i].h == h) fb = g_scfb[i].fb;
        pthread_mutex_unlock(&g_scfbmu);
    }
    if (!fb && !swapchain && fbc_on()) fb = fbc_get(rp, views, nv, w ? w : 1, h ? h : 1, ri->layerCount ? ri->layerCount : 1);
    if (!fb) {
        VkFramebufferCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp, .attachmentCount = nv, .pAttachments = views, .width = w ? w : 1, .height = h ? h : 1,
                                       .layers = ri->layerCount ? ri->layerCount : 1 };
        VKCHECK(vkCreateFramebuffer(g_dev, &fi, NULL, &fb));
        int kept = 0;
        if (swapchain && nv == 1) { pthread_mutex_lock(&g_scfbmu);
            if (g_nscfb < 32) { g_scfb[g_nscfb].v = views[0]; g_scfb[g_nscfb].rp = rp; g_scfb[g_nscfb].w = w; g_scfb[g_nscfb].h = h; g_scfb[g_nscfb++].fb = fb; kept = 1; }
            pthread_mutex_unlock(&g_scfbmu); }
        if (!kept) defer_add((dfr){ .kind = DF_FRAMEBUFFER, .h.fb = fb });
    }
    VkRenderPassBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb, .renderArea = area ? *area : ri->renderArea, .clearValueCount = nv, .pClearValues = cv };
    vkCmdBeginRenderPass(cb, &bi, secondary ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
}
static void rend_end(VkCommandBuffer cb) {
    // (a held frame: the pass's secondary ends and a new gap begins; the breadcrumbs' pass closes there, its GPU mark with DM_VK_BREADCRUMBS in the gap, after the pass:
    //   a fill is not allowed inside a pass. That mark is work in the gap, so with breadcrumbs on the passes around it are not run as one, seg_record_prim)
    if (g_sub && g_sub->dec && g_seg_in && cb == g_sub->cb) { VKCHECK(vkEndCommandBuffer(cb)); seg_push(g_sub, g_sub->cur); g_seg_in = 0; seg_gap_begin(g_sub); cr_pass_end(g_sub->cb); return; }
    if (g_dynren) vkCmdEndRendering(cb); else vkCmdEndRenderPass(cb);
    cr_pass_end(cb);   // (breadcrumbs: with DM_VK_BREADCRUMBS the GPU marks the pass done)
}
// the render pass a secondary's contents are recorded against: the one rend_begin_ex would begin for ri (compatibility leaves the load / store ops out)
static VkRenderPass rp_compat(const VkRenderingInfo *ri, const uint32_t *cf, uint32_t dsfmt, uint32_t samples) {
    rpkey k; memset(&k, 0, sizeof k); k.n = (uint8_t)ri->colorAttachmentCount; k.samples = (uint8_t)(samples ? samples : 1);
    for (uint32_t i = 0; i < k.n && i < 8; i++) { const VkRenderingAttachmentInfo *c = &ri->pColorAttachments[i]; if (!c->imageView) continue;
        k.cfmt[i] = cf[i]; k.cop[i] = c->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 2 : 1; k.cst[i] = rp_store_code(c->storeOp); if (c->imageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) k.optimal = 1; }
    const VkRenderingAttachmentInfo *d = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment : NULL, *s = ri->pStencilAttachment && ri->pStencilAttachment->imageView ? ri->pStencilAttachment : NULL;
    if (d || s) { k.dsfmt = dsfmt; k.dop = rp_load_code(d); k.sop = rp_load_code(s);
        k.dst = d ? rp_store_code(d->storeOp) : 1; k.sst = s ? rp_store_code(s->storeOp) : 1; }
    return rp_get(&k);
}

// The marker of a submission that failed with imported pools (see mtl_init): set = 1 writes it, 0 tells whether it is there
static int host_import_mark(int set) {
    const char *pc = getenv("DM_VK_PCACHE"); if (!pc) return 0;
    char m[1100]; snprintf(m, sizeof m, "%s", pc); char *sl = strrchr(m, '/'); if (!sl) return 0; snprintf(sl + 1, sizeof m - (size_t)(sl + 1 - m), "host_import_failed");
    if (!set) return access(m, F_OK) == 0;
    FILE *f = fopen(m, "w"); if (f) { fputs("vkQueueSubmit failed with imported host memory: the next start moves driver memory onto the pools instead\n", f); fclose(f); }
    return f != NULL;
}
static int mark_file(const char *name, const char *what, int op) {   // a marker next to the pipeline cache: op 1 writes it, -1 removes it, 0 tells whether it is there
    const char *pc = getenv("DM_VK_PCACHE"); if (!pc) return 0;
    char m[1100]; snprintf(m, sizeof m, "%s", pc); char *sl = strrchr(m, '/'); if (!sl) return 0; snprintf(sl + 1, sizeof m - (size_t)(sl + 1 - m), "%s", name);
    if (op < 0) return unlink(m) == 0;
    if (!op) return access(m, F_OK) == 0;
    FILE *f = fopen(m, "w"); if (f) { fputs(what, f); fclose(f); }
    return f != NULL;
}
static int heap_import_mark(int op) {
    return mark_file("heap_import_crashed", "importing a dma-buf for the first pool, or a submission with dma-buf pools; if this file is still here that failed, and the next start leaves the route out\n", op);
}
// The marker of a driver that crashed exporting its memory (see mtl_init): op 1 writes it, -1 removes it, 0 tells whether it is there
static int fd_export_mark(int op) {
    const char *pc = getenv("DM_VK_PCACHE"); if (!pc) return 0;
    char m[1100]; snprintf(m, sizeof m, "%s", pc); char *sl = strrchr(m, '/'); if (!sl) return 0; snprintf(sl + 1, sizeof m - (size_t)(sl + 1 - m), "fd_export_crashed");
    if (op < 0) return unlink(m) == 0;
    if (!op) return access(m, F_OK) == 0;
    FILE *f = fopen(m, "w"); if (f) { fputs("allocating exportable GPU memory for the first pool; if this file is still here the driver crashed doing so, and the next start moves its mappings instead\n", f); fclose(f); }
    return f != NULL;
}
static int mirror_ordered(void);
static void queue_submit(VkCommandBuffer *cbs, uint32_t n, uint64_t signal, VkFence vf) {
    VkTimelineSemaphoreSubmitInfo ts = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal };
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ts, .commandBufferCount = n, .pCommandBuffers = cbs, .signalSemaphoreCount = 1, .pSignalSemaphores = &g_tl };
    if (!g_tlsem) { si.pNext = NULL; si.signalSemaphoreCount = 0; si.pSignalSemaphores = NULL; }   // (the fence vf says it is done instead)
    if (!mirror_ordered()) shadow_sync();   // (ordered: the submission's own command buffer copies them, shadow_sync_ordered)
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, g_tlsem ? VK_NULL_HANDLE : vf); pthread_mutex_unlock(&g_qmu);
    if (r && g_host_import && host_import_mark(1)) VLOG("[RED] vkQueueSubmit failed with imported host memory: the next start will not import it");
    if (r && g_heapmap > 0 && heap_import_mark(1)) VLOG("[RED] vkQueueSubmit failed with dma-buf pools: the next start moves the driver's mappings instead");
    if (r) { nvn_vk_crumbs_dump("vkQueueSubmit failed"); VDIE("vkQueueSubmit failed: %d", r); }
}
// ================= The whole frame held (renderer phase 2, DM_VK_FRAME_HOLD=1; off by default until measured on phones) =================
// A finished submission is not handed to the GPU at once: the frame's submissions wait until the frame is presented, so the frame is known whole before any of it runs
// (phase 2 decides each pass's stores from that, fg_rec.h). A submission gets its fence value when it is recorded (as before: deferred destruction keys on it), and a
// held one is submitted, in order, the moment anything needs it: a wait on its fence (a game sync, a slot reused, a pool freed), a one-off submission, the present, or
// NSUB - 1 held. Submitting needs no recording (the command buffers are finished), so any thread can do it under g_cmu. Costs up to a frame of latency
typedef struct { sub_t *s; VkCommandBuffer cbs[2]; uint32_t n; uint64_t f; } held_t;
static held_t g_held[NSUB]; static _Atomic int g_nheld; static _Atomic uint64_t g_submitted;   // g_submitted: the last fence value given to the GPU
static uint64_t g_hold_frames, g_hold_split;   // frames submitted whole at their present, early flushes (the present line)
// Default (the user's choice after the S25+ test, 2026-11-19: mode 2 looked better): the quirk table's Q_FRAME_HOLD (mode 2 on Adreno and Arm's own driver, where it has
//   been run) where the replay has a thread of its own (hle_nvn.c DM_NVN_ASYNC), whose ~7 ms more a frame then stays off the game's thread; off where the replay runs on
//   the game's thread. It used to follow the replay thread alone, so a GPU given Faster rendering got the held frame too, untested (perf-1124 fix 3: PowerVR and Xclipse
//   get the thread, not the hold)
int nvn_async_on(void) __attribute__((weak));
// (profile engine_quirks frame_hold_off, Legends: Arceus: off by default, DM_VK_FRAME_HOLD=1 / 2 to try: its transform feedback capture passes and report timestamps
//   have not been run held on a phone)
static int hold_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_FRAME_HOLD");
    v = e ? atoi(e) : !DM_GAME_ENG_FRAME_HOLD_OFF && nvn_async_on && nvn_async_on() ? quirk_default(Q_FRAME_HOLD, &g_gpuid) : 0; if (v < 0 || v > 2) v = 0;
    quirk_note(Q_FRAME_HOLD, v, vkq_env_src("DM_VK_FRAME_HOLD"));
    VLOG("frames: %s", v == 2 ? "held whole until presented, each pass's stores decided from the whole frame (DM_VK_FRAME_HOLD=2): up to a frame later on screen"
                       : v ? "held whole until presented (DM_VK_FRAME_HOLD=1): up to a frame later on screen" : "submitted as recorded"); } return v; }
static uint64_t g_dec_dc, g_dec_fix, g_dec_passes;   // stores left out (overwritten unread), stores put back (read after all: a guess elsewhere was wrong), passes decided
static void seg_record_prim(sub_t *s);   // (below, with the passes)
static void hold_flush_locked(int why) {   // caller holds g_cmu; why 0 the present, 1 something waited on held work
    int n = atomic_load(&g_nheld); if (!n) return;
    for (int i = 0; i < n; i++) { held_t *h = &g_held[i]; if (h->s->dec) seg_record_prim(h->s); queue_submit(h->cbs, h->n, h->f, h->s->vf); g_inflight[g_nin++] = h->s; atomic_store(&g_submitted, h->f); }
    atomic_store(&g_nheld, 0); pthread_cond_broadcast(&g_ccv);
    if (why) g_hold_split++; else g_hold_frames++;
}
static void hold_flush(int why) { if (!atomic_load(&g_nheld)) return; pthread_mutex_lock(&g_cmu); hold_flush_locked(why); pthread_mutex_unlock(&g_cmu); }
// Fence values are handed out under g_cmu so the present thread's submissions (screenshots) and the game's stay in order on the timeline
static uint64_t submit_tracked(sub_t *s, VkCommandBuffer *cbs, uint32_t n) {
    pthread_mutex_lock(&g_cmu);
    uint64_t f = ++g_fence; s->fence = f; cr_submitted(s);
    if (!g_tlsem) {   // a new fence for this submission (the slot is reused only after its last submission completed, so the old fence is no longer waited on)
        static int reuse = -1; if (reuse < 0) { const char *e = getenv("DM_VK_FENCE_REUSE"); reuse = !(e && !strcmp(e, "0")); }   // (DM_VK_FENCE_REUSE=0: destroyed and made again, as before)
        if (s->vf && reuse) VKCHECK(vkResetFences(g_dev, 1, &s->vf));   // (the same fence again: no destroy + create per submission, fable-cpu fix 4)
        else { if (s->vf) vkDestroyFence(g_dev, s->vf, NULL);
               VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; s->vf = VK_NULL_HANDLE; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &s->vf)); }
    }
    if (hold_on() && n <= 2) {   // held until the present (or until something waits on it)
        int k = atomic_load(&g_nheld); held_t *h = &g_held[k]; h->s = s; h->n = n; h->f = f; for (uint32_t i = 0; i < n; i++) h->cbs[i] = cbs[i];
        atomic_store(&g_nheld, k + 1);
        if (k + 1 >= NSUB - 1) hold_flush_locked(1);   // (the ring would need a held slot next)
        pthread_mutex_unlock(&g_cmu);
        return f;
    }
    hold_flush_locked(1);   // (anything held goes first: the timeline in order)
    queue_submit(cbs, n, f, s->vf); atomic_store(&g_submitted, f);
    g_inflight[g_nin++] = s; pthread_cond_broadcast(&g_ccv);
    pthread_mutex_unlock(&g_cmu);
    return f;
}
// Mirrored pools in submission order (Mali, the emulator; #21): the pages the game wrote are copied into the GPU's copy by the GPU itself, at the start of this submission,
// after a full barrier: the CPU used to write them straight into the mirror before each submission, while earlier frames still ran on the GPU and read it (those frames
// drew with the next frame's data), and over what earlier submissions' GPU work had written there (copies, uniform updates, storage writes) before it was copied back
// to the game's pages (the copy-back then brought the game's stale bytes back: lost GPU results, white and black surfaces on a Mali-G615). The ranges earlier submissions
// still in flight write are left out of the copy. DM_VK_MIRROR_ORDERED=0: the old CPU copy
typedef struct { pool_ent *e; sub_t *s; VkCommandBuffer cb; const wb_t *ex; int nex; uint64_t bytes; } mirror_ctx;
static void mirror_piece(mirror_ctx *c, uintptr_t a, size_t n) {
    VkBuffer sb; VkDeviceSize so; uint8_t *m = arena_alloc(c->s, n, &sb, &so); memcpy(m, (const void *)a, n);
    VkBufferCopy cp = { so, a - c->e->base, n }; vkCmdCopyBuffer(c->cb, sb, c->e->buf, 1, &cp); c->bytes += n;
}
static void mirror_run(uintptr_t a, size_t n, void *u) {   // the run's bytes but the ranges in flight (c->ex)
    mirror_ctx *c = u; nvn_dirty_protect((void *)a, n);
    for (uintptr_t at = a, end = a + n; at < end;) {
        uintptr_t skip = at; for (int moved = 1; moved && skip < end;) { moved = 0;
            for (int i = 0; i < c->nex; i++) { uintptr_t lo = (uintptr_t)c->ex[i].addr, hi = lo + (uintptr_t)c->ex[i].n; if (lo <= skip && hi > skip) { skip = hi; moved = 1; } } }
        if (skip != at) { at = skip < end ? skip : end; continue; }
        uintptr_t next = end; for (int i = 0; i < c->nex; i++) { uintptr_t lo = (uintptr_t)c->ex[i].addr; if (lo > at && lo < next) next = lo; }
        mirror_piece(c, at, next - at); at = next;
    }
}
static int mirror_ordered(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_MIRROR_ORDERED"); v = !(e && !strcmp(e, "0")); } return v; }
static void shadow_sync_ordered(sub_t *s) {
    if (!atomic_load(&g_nshadow) || !mirror_ordered()) return;
    static wb_t *ex; static int cex; int nex = 0;
    pthread_mutex_lock(&g_cmu);
    for (int i = 0; i < g_nin; i++) { sub_t *q = g_inflight[i];
        for (int k = 0; k < q->nwb; k++) { if (nex == cex) { cex = cex ? cex * 2 : 256; ex = realloc(ex, sizeof *ex * (size_t)cex); } ex[nex++] = q->wb[k]; } }
    pthread_mutex_unlock(&g_cmu);
    VkCommandBuffer cb = ucb(); if (!cb) return;
    barrier_all(cb);   // what earlier submissions read and write in the mirrors is done first
    mirror_ctx c = { NULL, s, cb, ex, nex, 0 };
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) { pool_ent *e = &g_pools[i]; if (!e->shadow) continue;
        c.e = e; uint64_t s0 = nvn_dirty_now(); nvn_dirty_runs((void *)e->base, e->size, e->stamp, mirror_run, &c); e->stamp = s0; }
    pthread_mutex_unlock(&g_mu);
    { static int said; if (!said++) VLOG("memory pools: mirrors updated by the GPU in submission order (%.1f KB in the first submission, %d ranges in flight left out)", c.bytes / 1024.0, nex); }
}
static uint64_t g_vp_c;
uint64_t mtl_sub_commit(void) {
    sub_t *s = g_sub; if (!s) return g_fence;
    pass_end();
    if (s->dec && g_seg_in) rend_end(s->cb);   // (a pass still open: its secondary closed, a gap after it)
    if (g_nlc) lc_out_all();   // (kept depth clears written out: no kept clear outlives its submission, so the CPU's reads, the present and later submissions see it)
    shadow_sync_ordered(s);   // (into the upload command buffer, which runs before this submission's work)
    if (g_tsq) vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, (uint32_t)(s - g_subs) * 2 + 1);
    if (g_rtq) { report_ts_flush(s->cb, s); vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_rtq, (uint32_t)(s - g_subs) * RQ_PER + RQ_END); }
    barrier_host(s->cb);
    VKCHECK(vkEndCommandBuffer(s->cb));
    if (s->dec) { seg_push(s, (seg_t){ .pass = 0, .fgp = -1, .cb = s->cb, .work = ~0u }); s->cb = s->prim; g_gap_cb = VK_NULL_HANDLE; }   // (the primary is recorded at the flush: seg_record_prim)
    VkCommandBuffer cbs[2]; uint32_t n = 0;
    if (s->ucb_open) { barrier_all(s->ucb); VKCHECK(vkEndCommandBuffer(s->ucb)); cbs[n++] = s->ucb; s->ucb_open = 0; }
    cbs[n++] = s->cb;
    g_sub = NULL;
    uint64_t vt0 = mono_ns(); uint64_t r = submit_tracked(s, cbs, n); uint64_t vd = mono_ns() - vt0; g_vp_c += vd; atomic_fetch_add(&g_st_submit_ns, vd);   // (DM_VK_PROF: the queue submission)
    return r;
}
void mtl_drain(void) { mtl_finish(); }
static void post(post_op o) {
    sub_t *s = g_sub; if (!s) { RED_ONCE(1, "event / counter outside a submission"); return; }
    if (s->npost == s->cpost) { s->cpost = s->cpost ? s->cpost * 2 : 64; s->post = realloc(s->post, s->cpost * sizeof *s->post); }
    s->post[s->npost++] = o;
}
// One-off command buffer for work outside a submission (uploads requested off the submit thread, readbacks): recorded, submitted and waited for at once
static VkCommandPool g_once_pool; static pthread_mutex_t g_once_mu = PTHREAD_MUTEX_INITIALIZER;
// One-off uploads (a texture or a new image's first clear, recorded off the submit thread) are not waited for (DM_VK_ONCE_WAIT=1: waited, as before): a one-off is
//   ordered after everything submitted before it, so waiting for it meant waiting for all the GPU work queued so far. On a slow GPU that was a frame or two each time:
//   a 16x16 texture took 30-70 ms on a Mali-G68 with PanVK (Violet, 2026-11-21), a game thread that made seven new textures stood 3 s. Its full barriers at the end
//   order it before every later submission; its command buffer and staging buffer go once its fence has signalled (once_reap, at the next one-off, at most 64 waiting)
typedef struct { VkCommandBuffer cb; VkFence f; VkBuffer sb; vmalloc_t tmp; } once_pend;
static once_pend g_once_p[64]; static int g_once_np;
// v0.96 (2026-11-23): WAITED by default again. Not waited, Legends: Arceus on a Galaxy S25+ (Adreno 830) showed single frames with terrain and trees pale and
//   untextured while walking and in battle (10-27 such frames in 10 s); with the wait, none. Why the ordering is not enough is not found yet. DM_VK_ONCE_WAIT=0: not waited
static int once_wait(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_ONCE_WAIT"); v = !(e && *e && atoi(e) == 0); } return v; }
static void once_free(once_pend *o) {
    vkDestroyFence(g_dev, o->f, NULL); vkFreeCommandBuffers(g_dev, g_once_pool, 1, &o->cb);
    if (o->sb) { vkDestroyBuffer(g_dev, o->sb, NULL); vk_free(o->tmp); }
}
static void once_reap(int all) {   // (g_once_mu held) the finished ones freed; all: waited for first
    for (int i = 0; i < g_once_np;) {
        if (all) vk_wait(NULL, g_once_p[i].f, UINT64_MAX);
        else if (vkGetFenceStatus(g_dev, g_once_p[i].f) != VK_SUCCESS) { i++; continue; }
        once_free(&g_once_p[i]); g_once_p[i] = g_once_p[--g_once_np]; }
}
static VkCommandBuffer once_begin(void) {
    pthread_mutex_lock(&g_once_mu);
    if (g_once_np) once_reap(0);
    if (!g_once_pool) { VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf }; VKCHECK(vkCreateCommandPool(g_dev, &pci, NULL, &g_once_pool)); }
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_once_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKCHECK(vkBeginCommandBuffer(cb, &bi)); barrier_all(cb);
    return cb;
}
static void once_end(VkCommandBuffer cb) {   // ordered after everything submitted so far (same queue, full barriers), waited for here
    barrier_all(cb); barrier_host(cb); VKCHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &f));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    hold_flush(1);   // (held work first: this one is ordered after everything recorded so far)
    shadow_sync();
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, f); pthread_mutex_unlock(&g_qmu);
    if (r) VDIE("one-off submit failed: %d", r);
    vk_wait(NULL, f, UINT64_MAX); vkDestroyFence(g_dev, f, NULL);
    vkFreeCommandBuffers(g_dev, g_once_pool, 1, &cb);
    pthread_mutex_unlock(&g_once_mu);
}
// once_end without the wait (see once_begin): sb / tmp, a staging buffer of its own, go with it (VK_NULL_HANDLE: none)
static void once_end_async(VkCommandBuffer cb, VkBuffer sb, vmalloc_t tmp) {
    if (once_wait()) { once_end(cb); if (sb) { vkDestroyBuffer(g_dev, sb, NULL); vk_free(tmp); } return; }
    barrier_all(cb); barrier_host(cb); VKCHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &f));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    hold_flush(1);   // (held work first: this one is ordered after everything recorded so far)
    shadow_sync();
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, f); pthread_mutex_unlock(&g_qmu);
    if (r) VDIE("one-off submit failed: %d", r);
    if (g_once_np == (int)(sizeof g_once_p / sizeof *g_once_p)) { once_reap(0); if (g_once_np == (int)(sizeof g_once_p / sizeof *g_once_p)) { vk_wait(NULL, g_once_p[0].f, UINT64_MAX); once_reap(0); } }
    g_once_p[g_once_np++] = (once_pend){ cb, f, sb, tmp };
    static int said; if (!said++) VLOG("one-off uploads: not waited for (DM_VK_ONCE_WAIT=0; the default waits)");
    pthread_mutex_unlock(&g_once_mu);
}
static VkCommandBuffer ucb(void) {
    sub_t *s = g_sub;
    if (!s || !on_sub_thread()) return VK_NULL_HANDLE;
    if (!s->ucb_open) { VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT }; VKCHECK(vkBeginCommandBuffer(s->ucb, &bi)); s->ucb_open = 1; }
    return s->ucb;
}

// ================= Textures =================
typedef struct { uint32_t fmt, vt, lvl, nlvl, lay, nlay, swz, aspect; } vview_key;
typedef struct { vview_key k; VkImageView v; } vview;
typedef struct vtex {
    _Atomic int refs; VkImage img; vmalloc_t a; VkFormat fmt; int nvn_fmt;
    uint32_t w, h, d, levels, layers, samples; VkImageUsageFlags usage; VkImageAspectFlags aspect; int is3d, cube_ok;
    uint8_t fswz[4];                       // the format's own channel mapping (no alpha, r/b swap), NVN_SWZ_*
    vview *views; int nviews; pthread_mutex_t vmu;
    int pkg; uint64_t stamp, seen; const void *span;   // CPU-filled: write-tracking stamp of the uploaded content
    VkBufferView bview;                    // buffer texture (TARGET_BUFFER): a view of its memory pool
    int fixed_fmt; VkFormat twin;          // 1: not mutable (depth / stencil), 2: mutable within fmt / twin only (tex_make: keeps UBWC)
    int drop;                              // mip levels left out (texture quality half: the image is the game's levels from this one on)
    struct vtex *alt; int alt_fmt;         // BC decoded / transcoded on this GPU, viewed as another BC kind (tex_view_for): the same blocks converted for that kind
    _Atomic int pre;                       // uploaded ahead (mtl_tex_preupload) and not drawn yet: counted in g_pre_unused
    struct { uint64_t key, frame; uint16_t lv, ly; uint8_t asp, dc; } sk[8]; int nsk; int presented;   // stores of its attachments not yet read or overwritten (sk_*); shown in the window
    uint8_t dkeep; uint64_t dfirst;        // DM_VK_DEPTH_DISCARD: something read this depth image (keep storing it); the frame it was first a pass's depth (depth_store)
    uint64_t hp_stamp;                     // HUD-free frame generation: g_hp_seq + 1 of the frame it was last a render target in (hp_*)
    xf_t xf;                               // renderer stage 3 (xf_learn.h): its frame-end depth / stencil store, learnt from the next frames (under g_fgrmu)
} vtex;
// the queue family takes timestamps (the frame recorder's pass times, fg_rec.h); looked up once
static int g_tsq_ok(void) {
    static int v = -1; if (v >= 0) return v;
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    v = g_qf < nq && qp[g_qf].timestampValidBits && g_props.limits.timestampPeriod > 0; return v;
}
#include "fg_rec.h"   // renderer phase 1: the frame recorder (records only)
// Renderer stage 3 (xf_learn.h) and phase 3 (seg_record_prim): on with the held frame's decisions (DM_VK_FRAME_HOLD=2); DM_VK_XFRAME=0 / DM_VK_PASS_MERGE=0 leave each out
static int hold_dec(void);
static uint64_t g_xf_dc, g_xf_bad, g_merge_n;   // frame-end stores left out, left-out stores needed after all, passes run as one with the one before
static int xf_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_XFRAME"); v = !(e && !strcmp(e, "0")); } return v && hold_dec(); }
// (DM_AB=<frames>:r3 alternates both, leaving out and merging, in one run; what stage 3 learns is tracked on both sides)
int dm_ab(const char *key) __attribute__((weak));
static int r3_ab(void) { return dm_ab ? dm_ab("r3") : -1; }
static int merge_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PASS_MERGE"); v = !(e && !strcmp(e, "0")); } return v && barrier_relaxed() && r3_ab() != 0; }
// ================= HUD-free frame generation: the scene / HUD split learnt each frame (fg_rec.h's rule, kept small) =================
// The final image is what the pass drawing the shown image reads when it copies one image rendered this frame (Sword / Shield: img2, copied into the shown one); of
// the passes drawing into it, one reading anything rendered this frame is the scene or post-processing, one reading only textures not rendered this frame is the HUD.
// Learnt at each present for the next frame: which of the passes drawing into the final image is the first HUD pass. Then, as that pass begins, the final image is
// copied (the scene, HUD-less) and frame generation is fed that copy, with the frame as shown for fg_gen.frag to put the HUD back on top (nvn_fgen.h hud).
// Our own frame generation only (DM_FRAMEGEN=on); DM_FRAMEGEN_HUDFREE=0 leaves it out. Everything here runs on the thread that builds the GPU's work
static int framegen_mode(void);
enum { HP_N = 512 };
// (z: the pass's depth target; zclr / zcd: it clears it, to what: the far plane, which tells which way round the game keeps depth)
static struct { vtex *c0, *z; uint32_t draws; uint8_t rr, nr, zclr; float zcd; vtex *r[4]; } g_hp[HP_N]; static int g_hp_n;   // this frame's passes so far
static struct { vtex *c0, *z; uint32_t draws; uint8_t rr, nr, zclr; float zcd; vtex *r[4]; int open; } g_hpc;               // the open pass
static uint64_t g_hp_seq;                       // presents so far (the frame number HUD-free uses: hle_nvn.c's g_frame_no is set from the game's thread)
static vtex *g_hp_F; static int g_hp_k;         // learnt: the final image, and which pass drawing into it is the first HUD pass (1-based; 0: none)
static int g_hp_count; static uint64_t g_hp_scene_seq;   // passes into F begun this frame; the frame the scene copy was made in (+1)
static VkImage g_hp_simg; static vmalloc_t g_hp_sa; static uint32_t g_hp_sw, g_hp_sh; static VkFormat g_hp_sfmt; static int g_hp_sfresh;
// Depth guidance (nvn_fgen.h fg_depth_src): the scene's depth, learnt with the split (the depth target of the last pass before the first HUD pass whose shape is the final
//   image's: not a shadow map), copied with the scene; g_hp_zgen counts its copy images (the presenting side names a new one to frame generation), g_hp_zseq the frame copied
static vtex *g_hp_Z; static int g_hp_zfar1;
static VkImage g_hp_zimg; static vmalloc_t g_hp_za; static uint32_t g_hp_zw, g_hp_zh; static VkFormat g_hp_zfmt; static VkImageAspectFlags g_hp_zasp; static int g_hp_zfresh, g_hp_zgen; static uint64_t g_hp_zseq;
static int fg_depth_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_FRAMEGEN_DEPTH"); v = !(e && !strcmp(e, "0")); } return v; }
static int hudfree_on(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("DM_FRAMEGEN_HUDFREE"); v = !(e && !strcmp(e, "0")); }
    int m = framegen_mode(); return v && (m == 1 || m == 3);   // (ours, and Lossless Scaling's: nvn_lsfg.h ls_hud)
}
static void hp_read(vtex *x) {   // (sk_read: sampled, copied, loaded, shown)
    if (!g_hpc.open || !x || x == g_hpc.c0 || x->hp_stamp != g_hp_seq + 1) return;
    g_hpc.rr = 1; for (int i = 0; i < g_hpc.nr; i++) if (g_hpc.r[i] == x) return;
    if (g_hpc.nr < 4) g_hpc.r[g_hpc.nr++] = x;
}
static void hp_draw(void) { if (g_hpc.open) g_hpc.draws++; }
static void hp_pass_end(void) {
    if (!g_hpc.open) return; g_hpc.open = 0;
    if (g_hp_n < HP_N) { g_hp[g_hp_n].c0 = g_hpc.c0; g_hp[g_hp_n].draws = g_hpc.draws; g_hp[g_hp_n].rr = g_hpc.rr; g_hp[g_hp_n].nr = g_hpc.nr; memcpy(g_hp[g_hp_n].r, g_hpc.r, sizeof g_hpc.r);
                         g_hp[g_hp_n].z = g_hpc.z; g_hp[g_hp_n].zclr = g_hpc.zclr; g_hp[g_hp_n].zcd = g_hpc.zcd; g_hp_n++; }
}
static void sk_read(vtex *x); static void depth_read(vtex *x);   // (below: a read of an image, for the stores)
// a game pass begins (after its barrier, before vkCmdBeginRendering): its first colour target noted; the learnt first HUD pass into the final image takes the scene copy first
static void hp_pass_begin(VkCommandBuffer cb, vtex *c0) {
    if (!hudfree_on()) return;
    memset(&g_hpc, 0, sizeof g_hpc); g_hpc.c0 = c0; g_hpc.open = 1;
    if (!c0 || c0 != g_hp_F || !g_hp_k || ++g_hp_count != g_hp_k) return;
    if (c0->is3d || c0->samples > 1 || !(c0->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return;
    if (!g_hp_simg || g_hp_sw != c0->w || g_hp_sh != c0->h || g_hp_sfmt != c0->fmt) {   // (made, or made again at a new size / format; the old one goes when the GPU is done)
        if (g_hp_simg) { defer_add((dfr){ .kind = DF_IMAGE, .h.img = g_hp_simg, .a = g_hp_sa }); g_hp_simg = VK_NULL_HANDLE; }
        VkImageCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = c0->fmt, .extent = { c0->w, c0->h, 1 }, .mipLevels = 1, .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
        if (vkCreateImage(g_dev, &ii, NULL, &g_hp_simg)) { g_hp_simg = VK_NULL_HANDLE; return; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, g_hp_simg, &rq); g_hp_sa = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        VKCHECK(vkBindImageMemory(g_dev, g_hp_simg, g_hp_sa.b->mem, g_hp_sa.off)); g_hp_sw = c0->w; g_hp_sh = c0->h; g_hp_sfmt = c0->fmt; g_hp_sfresh = 1; }
    if (g_hp_sfresh) {   // (first use: UNDEFINED to GENERAL, as every image here)
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_hp_simg, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); g_hp_sfresh = 0; }
    VkImageCopy ic = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { c0->w, c0->h, 1 } };
    vkCmdCopyImage(cb, c0->img, VK_IMAGE_LAYOUT_GENERAL, g_hp_simg, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    vtex *Z = framegen_mode() == 1 && fg_depth_on() ? g_hp_Z : NULL;   // (depth guidance: ours only; Lossless Scaling takes pictures alone)
    if (Z && Z->img && Z->samples == 1 && !Z->is3d && (Z->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (Z->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        if (!g_hp_zimg || g_hp_zw != Z->w || g_hp_zh != Z->h || g_hp_zfmt != Z->fmt) {
            if (g_hp_zimg) { defer_add((dfr){ .kind = DF_IMAGE, .h.img = g_hp_zimg, .a = g_hp_za }); g_hp_zimg = VK_NULL_HANDLE; }
            VkImageCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = Z->fmt, .extent = { Z->w, Z->h, 1 }, .mipLevels = 1, .arrayLayers = 1,
                                     .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
            VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, Z->fmt, &fp);
            if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) && !vkCreateImage(g_dev, &ii, NULL, &g_hp_zimg)) {
                VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, g_hp_zimg, &rq); g_hp_za = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
                VKCHECK(vkBindImageMemory(g_dev, g_hp_zimg, g_hp_za.b->mem, g_hp_za.off)); g_hp_zw = Z->w; g_hp_zh = Z->h; g_hp_zfmt = Z->fmt;
                g_hp_zasp = Z->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT); g_hp_zfresh = 1; g_hp_zgen++; }
            else { g_hp_zimg = VK_NULL_HANDLE; g_hp_zgen++; } }
        if (g_hp_zimg) {
            if (g_hp_zfresh) {
                VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_hp_zimg, .subresourceRange = { g_hp_zasp, 0, 1, 0, 1 } };
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); g_hp_zfresh = 0; }
            VkImageCopy zc = { { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, { 0, 0, 0 }, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, { 0, 0, 0 }, { Z->w, Z->h, 1 } };
            vkCmdCopyImage(cb, Z->img, VK_IMAGE_LAYOUT_GENERAL, g_hp_zimg, VK_IMAGE_LAYOUT_GENERAL, 1, &zc);
            // a read of the game's depth: its store stays (the held frame's verdict, the learnt skips, DM_VK_DEPTH_DISCARD), not one of the HUD pass's own reads
            int op = g_hpc.open; g_hpc.open = 0; sk_read(Z); depth_read(Z); g_hpc.open = op;
            g_hp_zseq = g_hp_seq + 1; } }
    barrier_all(cb);   // (the copy reads the scene before the HUD pass draws on it)
    g_hp_scene_seq = g_hp_seq + 1;
}
// at the present (before the window gets the frame): the split learnt from this frame for the next one, and a fresh frame begins
static void hp_learn(vtex *shown) {
    if (!hudfree_on()) { g_hp_n = 0; return; }
    hp_pass_end();
    vtex *F = shown; int s = -1;
    for (int i = 0; i < g_hp_n; i++) if (g_hp[i].c0 == shown) s = i;
    if (s >= 0 && g_hp[s].nr == 1) F = g_hp[s].r[0];
    int j = 0, last_scene = 0, k = 0, ik = -1;
    // (j counts every pass into F, drawing or not, as hp_pass_begin counts them when it takes the copy: Legends: Arceus begins F with no draws between its
    //   passes, 61 / 65 / 67 of 62..70; counting only the drawing ones took the copy before the scene's own pass, the picture split, 2026-11-20)
    for (int i = 0; i < g_hp_n; i++) { if (g_hp[i].c0 != F) continue; j++; if (!g_hp[i].draws) continue;
        if (g_hp[i].rr) { last_scene = j; k = 0; ik = -1; } else if (!k && j > last_scene) { k = j; ik = i; } }
    {   // the scene's depth: the last depth target before the first HUD pass of the final image's shape and at least a quarter of its size, and which way round its depth is
        //   (what a pass clears it to is the far plane); unknown either way: none
        vtex *Z = NULL; int far1 = -1;
        for (int i = (ik >= 0 ? ik : g_hp_n) - 1; i >= 0 && F && !Z; i--) { vtex *z = g_hp[i].z;
            if (z && g_hp[i].draws && (uint64_t)z->w * 4 * z->h >= (uint64_t)F->w * F->h && fabs((double)z->w * F->h - (double)z->h * F->w) <= 0.02 * (double)z->w * F->h) Z = z; }
        for (int i = 0; i < g_hp_n && Z; i++) if (g_hp[i].z == Z && g_hp[i].zclr) far1 = g_hp[i].zcd > 0.5f;
        if (Z && far1 < 0 && Z == g_hp_Z) far1 = g_hp_zfar1;   // (cleared in an earlier frame only: as learnt then)
        if (far1 < 0) Z = NULL;
        if (Z != g_hp_Z || (Z && far1 != g_hp_zfar1)) { static int said; if (said < 8) { said++;
            if (Z) VLOG("frame generation, depth guidance: the scene's depth %ux%u (format %d), its far plane at %d", Z->w, Z->h, (int)Z->fmt, far1);
            else VLOG("frame generation, depth guidance: no scene depth found (pictures without it)"); } }
        g_hp_Z = Z; if (Z) g_hp_zfar1 = far1; }
    if (F != g_hp_F || k != g_hp_k) { static int said; if (said < 12) { said++; VLOG("frame generation, HUD-free: the final image %ux%u, the HUD from its %d%s pass drawing into it%s", F ? F->w : 0, F ? F->h : 0, k,
                                                                                   k == 1 ? "st" : k == 2 ? "nd" : k == 3 ? "rd" : "th", k ? "" : " (none found: the frame as it is)"); } }
    g_hp_F = F; g_hp_k = k; g_hp_n = 0; g_hp_count = 0; g_hp_seq++;
}
static void depth_read(vtex *x);   // (DM_VK_DEPTH_DISCARD: something reads this image, pass_begin)
// ================= Stores no one reads (tilers: Adreno, Mali; from the Arceus engine) =================
// On a tiler every render pass ends by writing its attachments from tile memory to the image, and a pass whose next user clears the image first or never looks at it again wrote
// for nothing (depth buffers above all). Vulkan needs the store decision when the pass begins, before its commands are known, so it is learnt: each attachment of a pass is keyed
// by the pass's render targets, how many times that set was used before in the frame and the slot, and its store stays pending on the image until the image is next read (sampled,
// copied, loaded by a pass, shown) or overwritten (a pass clearing it). Only an overwrite in the same frame counts: what lasts into the next frame may be the game's history or a
// cache it redraws only sometimes (entering the field from the loading screen, Legends: Arceus starts reading a dozen such images, among them shadow maps and its environment cube,
// which every loading frame had cleared). A key whose store was overwritten unread in its frame SK_LEARN times in a row stores nothing from then on (DONT_CARE);
// if one of its images is then read, the content was undefined for that one use, and the key stores again for good (reported in red). Shown images never skip.
// DM_VK_SKIP_STORES=0 turns it off (=1: on for a game where it is off by default)
enum { SK_LEARN = 90, SK_N = 4096 };
typedef struct { uint64_t key; uint16_t streak, ro; uint8_t off, rooff; } skey_t;   // ro: passes in a row that only read this depth / stencil attachment (sk_ro)
static skey_t g_sk[SK_N]; static int g_nsk_keys;
static pthread_mutex_t g_skmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_sk_skipped, g_sk_wrong;   // stores skipped, skipped stores read after all (the present line)
// Off by default since v0.16, Arceus too: a battle effect that samples a copy of the scene (Darkrai's Dark Void, 2026-11-11 on a Galaxy S25+) showed big flat grey
//   slanted boxes over a washed-out scene, what an attachment whose store was skipped reads back. DM_VK_SKIP_STORES=1 turns it on to compare
static int sk_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SKIP_STORES"); v = e && !strcmp(e, "1"); VLOG("attachment stores: %s", v ? "learnt (a store always overwritten unread is left out; DM_VK_SKIP_STORES=0: every pass stores)" : "every pass stores its attachments"); } return v; }
static skey_t *sk_get(uint64_t key) {
    if (!key) key = 1;
    for (uint32_t i = (uint32_t)(key ^ key >> 29) & (SK_N - 1), n = 0; n < SK_N; i = (i + 1) & (SK_N - 1), n++) {
        if (g_sk[i].key == key) return &g_sk[i];
        if (!g_sk[i].key) { if (g_nsk_keys > SK_N * 3 / 4) return NULL; g_nsk_keys++; g_sk[i].key = key; return &g_sk[i]; }
    }
    return NULL;
}
// The pending stores of x (asp 0: any aspect; lv / ly ~0: any) resolve: read (the store was needed) or overwritten (it was not)
static void sk_resolve_locked(vtex *x, uint32_t lv, uint32_t ly, int asp, int overwritten) {
    for (int i = 0; i < x->nsk;) {
        if ((asp && x->sk[i].asp != asp) || (lv != ~0u && x->sk[i].lv != lv) || (ly != ~0u && x->sk[i].ly != ly)) { i++; continue; }
        skey_t *k = sk_get(x->sk[i].key);
        if (k) {
            if (overwritten && x->sk[i].frame == g_frame_no) { if (k->streak < 60000) k->streak++; }
            else if (overwritten) k->streak = 0;   // overwritten in a later frame: not learnt (see above); a skipped one was harmless
            else { if (x->sk[i].dc && !k->off) { k->off = 1; g_sk_wrong++;
                       VLOG("[RED] skipped store read after all: %ux%u format %d level %u layer %u (%s); that pass stores again", x->w, x->h, x->fmt, x->sk[i].lv, x->sk[i].ly,
                            x->sk[i].asp == 1 ? "colour" : x->sk[i].asp == 2 ? "depth" : "stencil"); }
                   k->streak = 0; }
        }
        x->sk[i] = x->sk[--x->nsk];
    }
}
static vtex *g_lc_open; static void lc_read(vtex *x);   // (lazy clears, perf-1124 GPU fix B: above pass_begin)
static void sk_read(vtex *x) { if (x) { if (g_nlc || x == g_lc_open) lc_read(x); fgr_read_img(x, x->w, x->h, (uint32_t)x->fmt, FGR_READ); hp_read(x); }
    if (x && (x->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) && xf_on()) {   // (stage 3: a depth / stencil image read is never left out)
        pthread_mutex_lock(&g_fgrmu);
        if (xf_read(&x->xf)) { g_xf_bad++; RED_ONCE(62, "a depth store left out at a frame's end was read (%ux%u, format %d): stored again for good", x->w, x->h, (int)x->fmt); }
        pthread_mutex_unlock(&g_fgrmu); } if (!x || !x->nsk) return; pthread_mutex_lock(&g_skmu); sk_resolve_locked(x, ~0u, ~0u, 0, 0); pthread_mutex_unlock(&g_skmu); }
// A pass begins with x's subresource as attachment slot (0-7 colour, 8 depth, 9 stencil), cleared or loaded: what was pending resolves, and whether this pass stores comes back
static uint64_t g_sk_sig, g_sk_last;   // the pass's key so far (pass_begin); the key sk_attach made last
static uint64_t sk_key(int slot) { uint64_t key = (g_sk_sig ^ (uint64_t)(slot + 1) * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull; key ^= key >> 31; return key; }   // (slot's key in the pass)
static int sk_attach(vtex *x, uint32_t lv, uint32_t ly, int asp, int slot, int cleared) {
    pthread_mutex_lock(&g_skmu);
    sk_resolve_locked(x, lv, ly, asp, cleared);
    int dc = 0;
    uint64_t key = sk_key(slot);
    g_sk_last = key;
    skey_t *k = sk_on() ? sk_get(key) : NULL;
    if (k && !k->off && k->streak >= SK_LEARN && !x->presented) { dc = 1; g_sk_skipped++; }
    if (x->nsk == 8) sk_resolve_locked(x, x->sk[0].lv, x->sk[0].ly, x->sk[0].asp, 0);   // no room: the oldest counts as read (it never learns to skip)
    if (x->nsk < 8) { x->sk[x->nsk].key = key; x->sk[x->nsk].frame = g_frame_no; x->sk[x->nsk].lv = (uint16_t)lv; x->sk[x->nsk].ly = (uint16_t)ly; x->sk[x->nsk].asp = (uint8_t)asp; x->sk[x->nsk].dc = (uint8_t)dc; x->nsk++; }
    pthread_mutex_unlock(&g_skmu);
    return dc;
}
// Depth and stencil only read by a pass (testing without writing: the passes after the depth pass, transparent and post passes) need no store at all: VK_ATTACHMENT_STORE_OP_NONE
// keeps the image as it was, without writing tile memory back. Learnt per key as above: 4 read-only passes in a row and the next one begins with NONE. A draw (or clear) that writes
// it after all ends that pass (its content was kept, so ending it is exact) and begins a new one that stores, and the key stops using NONE. DM_VK_NO_STORE_NONE=1 turns it off
// Since perf-1124 GPU fix C (step 2) it runs without the store learning (sk_on) where frames are not held (Legends: Arceus, Xclipse, PowerVR, and any GPU without the
//   held frame): it is exact by itself, as a write restarts the pass storing (pass_writes), and the pass's key is made in pass_begin either way (sk_key). Held frames
//   decide depth / stencil stores from the whole pass instead (ds_none), so there it stays tied to sk_on as before
enum { SK_RO_LEARN = 4 };
static int hold_dec(void);
static int ro_free(void) { return !sk_on() && g_store_none && !hold_dec(); }   // (sk_ro without the store learning)
static int sk_ro(uint64_t key, int cleared) {   // whether the attachment sk_attach just keyed begins with store op NONE
    if (cleared || !(sk_on() || ro_free())) return 0;
    pthread_mutex_lock(&g_skmu); skey_t *k = sk_get(key); int r = k && !k->rooff && k->ro >= SK_RO_LEARN; pthread_mutex_unlock(&g_skmu);
    return r;
}
static void sk_ro_note(uint64_t key, int wrote, int violated) {   // at the end of the pass
    pthread_mutex_lock(&g_skmu); skey_t *k = sk_get(key);
    if (k) { if (violated) k->rooff = 1; if (wrote) k->ro = 0; else if (k->ro < 60000) k->ro++; }
    pthread_mutex_unlock(&g_skmu);
}
static _Atomic int g_cap;   // the frame capture's state (Frame capture, before pass_end)
static void cap_tex(int tid, const tex_t *t, const vtex *x, int vfmt, VkFormat f);
static void cap_printf(const char *f, ...);
static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;   // serial / sharing / CPU-written tables
static nvn_hmap g_by_serial = {.klen = 8}, g_ghost = {.klen = 8}, g_tex_live = {.klen = 8}, g_cpu_tex = {.klen = 8};
enum { NGHOST = 512 }; static uint64_t g_ghost_ring[NGHOST]; static unsigned g_ghost_next;
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } alias_key;
static nvn_hmap g_alias = {.klen = sizeof(alias_key)};
static _Atomic uint64_t g_ser_gen = 1;
static uint64_t g_pkg_uploads, g_pkg_reuploads, g_up_bytes;

static void vtex_ref(vtex *t) { if (t) atomic_fetch_add(&t->refs, 1); }
static atomic_long g_live_tex, g_live_tex_bytes;   // memory accounting (mem_report)
static _Atomic long g_pre_unused;   // bytes of textures uploaded ahead and not drawn yet (the worker's budget, mtl_tex_preupload)
static void vtex_unref(vtex *t) {
    if (!t || atomic_fetch_sub(&t->refs, 1) != 1) return;
    if (atomic_exchange(&t->pre, 0)) atomic_fetch_sub(&g_pre_unused, (long)t->a.size);
    atomic_fetch_sub(&g_live_tex, 1); atomic_fetch_sub(&g_live_tex_bytes, (long)t->a.size);
    for (int i = 0; i < t->nviews; i++) defer_add((dfr){ .kind = DF_VIEW, .h.view = t->views[i].v });
    if (t->bview) defer_add((dfr){ .kind = DF_BUFVIEW, .h.bview = t->bview });
    if (t->img) defer_add((dfr){ .kind = DF_IMAGE, .h.img = t->img, .a = t->a });
    if (t->alt) vtex_unref(t->alt);
    free(t->views); free(t);
}
static int fmt_is_depth(VkFormat f) { return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT; }
static int fmt_has_stencil(VkFormat f) { return f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT; }
static int fmt_compressed(int f) { return (f >= 0x41 && f <= 0x50) || (f >= 0x79 && f <= 0x94); }
static VkImageAspectFlags fmt_aspect(VkFormat f) { return f == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : fmt_is_depth(f) ? VK_IMAGE_ASPECT_DEPTH_BIT | (fmt_has_stencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : VK_IMAGE_ASPECT_COLOR_BIT; }

// Integer colour formats: a float clear colour is converted to integers (the hardware converts it; Vulkan reads the clear value's bits in the format's class)
static int fmt_int_class(VkFormat f) {   // 0 float / normalized, 1 uint, 2 sint
    switch (f) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32: return 1;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_SINT_PACK32: return 2;
        default: return 0;
    }
}
// The same bits in a format of another numeric class (want: 0 float, 1 uint, 2 sint), the channel layout kept: 32-bit channels in all three, the narrower integer ones
//   in the other signedness only (a float format of their size would convert, not reinterpret); 0: none. (tex_view_for takes the 32-bit ones only: a narrower one's
//   other signedness extends its sign the other way)
static VkFormat fmt_as_class(VkFormat f, int want) {
    static const VkFormat T[][3] = {
        { VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT }, { VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT },
        { VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT },
        { 0, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT }, { 0, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT }, { 0, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT },
        { 0, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT }, { 0, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT }, { 0, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT },
        { 0, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32 } };
    if (want < 0 || want > 2) return (VkFormat)0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) for (int c = 0; c < 3; c++) if (T[i][c] && T[i][c] == f) return T[i][want];
    return (VkFormat)0;
}
static VkClearColorValue clear_for(VkFormat f, const float rgba[4]) {
    VkClearColorValue v; int k = fmt_int_class(f);
    for (int i = 0; i < 4; i++) { if (k == 1) v.uint32[i] = rgba[i] <= 0 ? 0 : (uint32_t)rgba[i]; else if (k == 2) v.int32[i] = (int32_t)rgba[i]; else v.float32[i] = rgba[i]; }
    return v;
}
// ================= R11G11B10F drawn as RGBA16F (DM_VK_B10_AS_RGBA16, the Mali float-target test, #282) =================
// The packed float format (NVN 0x3F) is stood in for by RGBA16F (nvn_fmt_vk.c): every 0x3F image is made RGBA16F, its texel 8 bytes where the game's memory holds 4.
// What makes the stand-in read and write as R11G11B10F does:
//  - the game's bytes meet it only through a conversion (b10_conv.h, exact for every packed value): uploads from the game's files and pool content expand them; buffer
//    → texture copies expand and texture → buffer copies pack them on the GPU, in a compute shader with the same rules (b10_conv.comp, b10_copy_buf: in order with
//    what the GPU wrote before); CPU reads pack them; a buffer texture keeps the packed format (it is the game's memory);
//  - the packed format has no alpha: views read alpha 1 (the format's channel mapping), draws never write it and blending reads the target's alpha as 1 (pso_make,
//    pkey.b10); clears store the colour as the packed format would (negative to 0, rounded) with alpha 1, a new image's first clear too;
//  - what a draw writes into it is rounded as the packed format stores it (negative and -inf to 0, its mantissa, at most its largest finite value: spv_b10_outputs
//    in a variant of the fragment shader, pso_make), where the shader's output is what gets stored: without blending, or blending by min / max. A blended result
//    cannot be rounded in a shader: there what the blend adds up to is kept (below 0 or past 65024 too; logged once);
//  - what a shader stores into it as a storage image likewise: through an image the shader declares R11G11B10F at load (spv_b10_image_writes, alpha 1), through one
//    declared without a format by a compute shader after its dispatch (b10_quantize_region); a storage image only where R11G11B10F could be one (tex_new);
//  - a texture → texture copy or blit into the stand-in from another format is then made what R11G11B10F holds there (its range and rounding: b10_quantize_region);
//  - no view across texel sizes (b10_view_fmt): the stand-in viewed in a 4-byte format (on the Switch, its packed bits) is sampled as itself where the view is a
//    float one (it reads the colour, not the bits: said in red) and not made at all for an integer view, a storage image or a render target (said in red); a
//    4-byte image viewed as R11G11B10F is viewed as the packed format it holds;
//  - storage images the shaders declare R11G11B10F are declared RGBA16F (the image they are given); texel buffers keep the packed format;
//  - render targets: the attachment format is RGBA16F everywhere (passes, render passes, pipelines); the pipeline lists keep R11G11B10F (pso_record, prewarm);
//  - the frame capture and the golden recorder read the stand-in as it is (RGBA16F, labelled); the golden player converts when a recording and a replay differ.
// Each kind of meeting is logged once ("b10 swap: ..."), so a player's log shows what the swap met.
// Not a per-game choice: both float-target tests (this one and DM_VK_B10_NOCOMPRESS) key on the format alone. They are tests for Sword / Shield's white lit scene on
//   Arm's own driver; Legends: Arceus draws 21-31 R11G11B10F images a frame (a 1080p G-buffer, its bloom chain), which the stand-in makes twice the size and wider
//   than many Mali GPUs' tile budget, for no white to fix there. Any default-on of either must key on the game (Sword / Shield) as well as on the driver, never on
//   the driver alone: route A's is the quirk table's Q_B10_STANDIN (Arm's own driver, in a game whose profile says mali_lit_standin); the Android menu offers both
//   for Sword / Shield only
static int b10_on(void) { return nvn_fmt_vk_b10_swapped(0x3F); }   // (DM_VK_B10_AS_RGBA16, or DM_VK_PROBE=f16target)
static int b10_swapped_tex(const vtex *x) { return x && x->fmt == VK_FORMAT_R16G16B16A16_SFLOAT && nvn_fmt_vk_b10_swapped(x->nvn_fmt); }
static void b10_note(const char *what, const vtex *x, int other) {
    static const char *seen[64]; static int n; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu); int i = 0; while (i < n && strcmp(seen[i], what)) i++;
    if (i < n || n == 64) { pthread_mutex_unlock(&mu); return; } seen[n++] = what; pthread_mutex_unlock(&mu);
    VLOG("b10 swap: %s (first: image %ux%u, %u levels, %u layers%s; format or factor met: %d)", what, x ? x->w : 0, x ? x->h : 0, x ? x->levels : 0, x ? x->layers : 0, x && x->is3d ? ", 3D" : "", other);
}
static uint32_t vk_fmt_bytes(VkFormat f) {   // a colour format's texel size (0: not one the engine makes)
    int v = (int)f;
    if (v == 1 || (v >= 9 && v <= 15)) return 1; if ((v >= 2 && v <= 8) || (v >= 16 && v <= 22) || (v >= 70 && v <= 76)) return 2; if (v >= 23 && v <= 36) return 3;
    if ((v >= 37 && v <= 69) || (v >= 77 && v <= 83) || (v >= 98 && v <= 100) || v == 122 || v == 123) return 4; if (v >= 84 && v <= 90) return 6;
    if ((v >= 91 && v <= 97) || (v >= 101 && v <= 103) || (v >= 110 && v <= 112)) return 8; if (v >= 104 && v <= 106) return 12; if ((v >= 107 && v <= 109) || (v >= 113 && v <= 115)) return 16;
    return 0;
}
// the format a view of image t asked for in fmt is made in, for use B10V_* (who: the shader binding or target, for the log; may be NULL); VK_FORMAT_UNDEFINED: the
//   view is not made. fmt itself unless the test is on.
//  - the stand-in in a 4-byte format: on the Switch its packed bits read through another format, which the stand-in does not hold. A float view to sample is made in
//    the stand-in's own RGBA16F (it reads the colour, not the bits: wrong, said in red once); an integer view (Vulkan leaves a float image read through one
//    undefined), a storage image or a render target is not made (the slot reads 0, the target is not attached: said in red once, with the first binding's name)
//  - a 4-byte image viewed as R11G11B10F (which the swap made RGBA16F): viewed as the packed format it holds, the bits as they are
enum { B10V_SAMPLED, B10V_STORAGE, B10V_ATTACH };
static VkFormat b10_view_fmt(const vtex *t, VkFormat fmt, int use, const char *who) {
    if (!t || fmt == t->fmt || !b10_on()) return fmt;
    if (b10_swapped_tex(t) && vk_fmt_bytes(fmt) != 8) {
        if (use == B10V_SAMPLED && !fmt_int_class(fmt)) {
            RED_ONCE(154, "b10 swap: the stand-in sampled through a view in 4-byte format %d (%s; on the Switch it reads the packed bits): made in RGBA16F, it reads the colour instead (wrong data: if that scene looks wrong, turn the test off)", fmt, who ? who : "-");
            return t->fmt; }
        if (use == B10V_SAMPLED) RED_ONCE(155, "b10 swap: the stand-in sampled through an integer view in 4-byte format %d (%s; on the Switch its packed bits): there are none here, it reads 0 (wrong data: turn the test off)", fmt, who ? who : "-");
        else if (use == B10V_STORAGE) RED_ONCE(156, "b10 swap: the stand-in bound as a storage image in 4-byte format %d (%s; on the Switch its packed bits): not bound, it reads 0 and takes no writes (wrong data: turn the test off)", fmt, who ? who : "-");
        else RED_ONCE(157, "b10 swap: the stand-in as a render target in 4-byte format %d (%s; on the Switch drawn as its packed bits): not attached, the pass's draws into it are lost (wrong picture: turn the test off)", fmt, who ? who : "-");
        return VK_FORMAT_UNDEFINED;
    }
    if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT && !b10_swapped_tex(t) && vk_fmt_bytes(t->fmt) == 4) { b10_note("a 4-byte image viewed as R11G11B10F: viewed as the packed format it holds", t, t->fmt); return VK_FORMAT_B10G11R11_UFLOAT_PACK32; }
    return fmt;
}
// a clear of the stand-in: the colour as R11G11B10F stores it (negative to 0, rounded to its precision, the largest finite value at most), alpha 1
static VkClearColorValue b10_clear(const float rgba[4]) {
    VkClearColorValue v; v.float32[0] = b10_quantize(rgba[0], B10_MR); v.float32[1] = b10_quantize(rgba[1], B10_MR); v.float32[2] = b10_quantize(rgba[2], B10_MB); v.float32[3] = 1.0f;
    return v;
}
// First use of a new image: to GENERAL (where every image stays), cleared to 0 when asked (new Switch memory reads 0; a new Vulkan image is undefined)
static void img_init(vtex *t, int zero) {
    VkCommandBuffer cb = ucb(); int once = !cb; if (once) cb = once_begin();
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = t->img, .subresourceRange = { t->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    if (zero && t->samples <= 1) {
        VkImageSubresourceRange r = { t->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
        if (t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue c = {{0}}; if (b10_swapped_tex(t)) c.float32[3] = 1.0f;   // (the stand-in's alpha is 1: see b10_on)
            vkCmdClearColorImage(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
        else { VkClearDepthStencilValue c = {0, 0}; vkCmdClearDepthStencilImage(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
    }
    if (once) once_end_async(cb, VK_NULL_HANDLE, (vmalloc_t){0});
}
// Texture quality (DM_VK_TEX_QUALITY=half, the device profile's choice for weaker GPUs): the game's own block-compressed textures of 512x512 and up with mip levels are kept
// without their largest level, a quarter of the memory and bandwidth (and of the decode work on GPUs without BC). Only textures from the game's files: they are never drawn
// into, so render targets and everything the screen is made of keep their size. Views shift their levels down by the same count; copies into or out of such a texture are
// left out (not seen in Shield)
static uint64_t g_drop_n, g_drop_saved;   // textures stored without their largest level, bytes not stored (the memory readout)
// Arm's own driver before r49 closed the game three frames in with half textures (Mali-G615 MC2 on r44p1: #63 #65 #71 #82, the same crash every time); the same phones
// run with full textures (#77, #21) and r49p1 runs half (#53): there half is used only when the player chose it, not when the phone's profile did (DM_VK_TEX_QUALITY_AUTO)
static int tex_quality_half(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_TEX_QUALITY"); v = e && !strcmp(e, "half");
        int unsafe = quirk_default(Q_TEX_HALF_UNSAFE, &g_gpuid), mine = v && !getenv("DM_VK_TEX_QUALITY_AUTO");   // (mine: the player's own half, not the profile's)
        quirk_note(Q_TEX_HALF_UNSAFE, unsafe && !mine, unsafe && mine);
        if (v && unsafe && !mine) { v = 0;
            VLOG("texture quality full, not the phone profile's half: Arm's driver r%dp%d closed the game with half textures (Mali-G615 MC2, #82); DM_VK_TEX_QUALITY=half in env.txt uses it anyway", g_arm_rel, g_arm_p); }
        else if (v) VLOG("texture quality half: the game's large textures without their largest mip level"); }
    return v;
}
static vtex *tex_new(const tex_t *t, int pkg) {
    uint32_t vf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_vk(t->d.format, &vf, swz) || !vf) { RED_ONCE(2, "texture format %#x has no Vulkan equivalent (first case)", t->d.format); return NULL; }
    vtex *x = calloc(1, sizeof *x); atomic_fetch_add(&g_live_tex, 1); x->refs = 1; x->fmt = (VkFormat)vf; x->nvn_fmt = t->d.format; memcpy(x->fswz, swz, 4); pthread_mutex_init(&x->vmu, NULL);
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = x->fmt, .mipLevels = t->d.levels ? t->d.levels : 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT };
    ci.extent = (VkExtent3D){ t->d.width ? t->d.width : 1, t->d.height ? t->d.height : 1, 1 };
    switch (t->d.target) {
        case 0: ci.extent.height = 1; break;                                                   // 1D as one-row 2D (as nvn_mtl.m)
        case 2: ci.imageType = VK_IMAGE_TYPE_3D; ci.extent.depth = t->d.depth ? t->d.depth : 1; ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT; x->is3d = 1; break;   // (2D-array compatible: rendered into slice by slice, att_view)
        case 3: ci.arrayLayers = t->d.height ? t->d.height : 1; ci.extent.height = 1; break;
        case 4: ci.arrayLayers = t->d.depth ? t->d.depth : 1; break;
        case 5: ci.samples = (VkSampleCountFlagBits)(t->d.samples ? t->d.samples : 1); ci.mipLevels = 1; break;
        case 6: ci.samples = (VkSampleCountFlagBits)(t->d.samples ? t->d.samples : 1); ci.arrayLayers = t->d.depth ? t->d.depth : 1; ci.mipLevels = 1; break;
        case 8: ci.arrayLayers = 6; break;
        case 9: ci.arrayLayers = (t->d.depth ? t->d.depth : 1) * 6; break;
        default: break;
    }
    if (pkg && tex_quality_half() && t->has_layout && fmt_compressed(t->d.format) && !(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))
        && (t->d.target == 1 || t->d.target == 4 || t->d.target == 8 || t->d.target == 9) && ci.mipLevels >= 2 && ci.extent.width >= 512 && ci.extent.height >= 512) {
        int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        if (bpb) g_drop_saved += (uint64_t)((ci.extent.width + bw - 1) / bw) * ((ci.extent.height + bh - 1) / bh) * bpb * ci.arrayLayers;
        ci.extent.width >>= 1; ci.extent.height >>= 1; ci.mipLevels--; x->drop = 1; g_drop_n++;
    }
    if (!x->is3d && ci.arrayLayers >= 6 && ci.extent.width == ci.extent.height && ci.samples == VK_SAMPLE_COUNT_1_BIT) { ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; x->cube_ok = 1; }
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, x->fmt, &fp); VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ? VK_IMAGE_USAGE_SAMPLED_BIT : 0);
    if (!fmt_compressed(t->d.format)) {   // packaged textures too: a game may draw into one it loaded (Shield: an RGBA8 one, "cannot be attached", its pass lost)
        if (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        // R11G11B10F's RGBA16F stand-in (DM_VK_B10_AS_RGBA16): a storage image only where R11G11B10F itself could be one on this GPU (storage of it is optional; RGBA16F's
        //   is not). A storage image may lose Mali's framebuffer compression, and the test must change the format alone, not the compression with it
        VkFormatProperties bp = { 0 }; int sw = nvn_fmt_vk_b10_swapped(t->d.format);
        if (sw) vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &bp);
        if (!sw || (bp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        else b10_note("an image the game may bind as a storage image: made without storage usage, as R11G11B10F has none on this GPU", NULL, (int)bp.optimalTilingFeatures);
    }
    if (fmt_compressed(t->d.format) && !nvn_fmt_vk_decode(t->d.format)) ci.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    // Mutable formats switch off Adreno's framebuffer compression (UBWC) unless the formats are listed: depth / stencil images are not mutable at all (their views keep
    // the format), colour images that can be render targets list their format and its sRGB / linear twin (the reinterpretations NVN views make of render targets)
    static int nolist = -1; if (nolist < 0) nolist = getenv("DM_VK_NO_FMT_LIST") != NULL;
    VkFormat twin = x->fmt == VK_FORMAT_R8G8B8A8_UNORM ? VK_FORMAT_R8G8B8A8_SRGB : x->fmt == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM :
                    x->fmt == VK_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_SRGB : x->fmt == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : x->fmt;
    VkFormat fl[2] = { x->fmt, twin }; VkImageFormatListCreateInfo fli = { .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = twin != x->fmt ? 2 : 1, .pViewFormats = fl };
    // Mali (Arm) keeps AFBC only on images that are not mutable (Bifrost) or list their view formats (Valhall): there every colour image that is not block-compressed
    // gets the same treatment as render targets do on Adreno, and one with no sRGB / linear twin is not mutable at all
    static int arm = -1; if (arm < 0) arm = quirk_note(Q_ARM_FORMATS, quirk_default(Q_ARM_FORMATS, &g_gpuid) || getenv("DM_VK_ARM_FORMATS") != NULL, getenv("DM_VK_ARM_FORMATS") != NULL);
    int plain = !fmt_compressed(t->d.format) || nvn_fmt_vk_decode(t->d.format);
    if (!nolist && !x->is3d) {
        if (fmt_is_depth(x->fmt) || x->fmt == VK_FORMAT_S8_UINT) { ci.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; x->fixed_fmt = 1; }
        else if (arm && plain && twin == x->fmt) { ci.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; x->fixed_fmt = 1; }
        else if (g_fmtlist && ((ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) || (arm && plain))) { ci.pNext = &fli; x->fixed_fmt = 2; x->twin = twin; }   // (without a list: mutable, as before lists)
    }
    // DM_VK_B10_NOCOMPRESS (a Mali test, off by default): an R11G11B10F render target made without framebuffer compression, for the case that Arm's driver mishandles
    //   the packed float format in AFBC (the white lit scene). Through VK_EXT_image_compression_control where the driver has it (compression disabled for this image);
    //   else by the driver's own rule, the opposite of the lists above: mutable with no list of view formats (Mali keeps AFBC only on images that are not mutable or
    //   list their view formats; Adreno's UBWC goes the same way). Only these targets; logged with what the driver then reports (b10 nocompress)
    VkImageCompressionControlEXT ncc = { .sType = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, .flags = VK_IMAGE_COMPRESSION_DISABLED_EXT };
    int b10nc = g_b10nc && t->d.format == 0x3F && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (b10nc && g_compctl) { ncc.pNext = ci.pNext; ci.pNext = &ncc; }
    else if (b10nc) { ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; ci.pNext = NULL; x->fixed_fmt = 0; x->twin = 0; }
    // a sample count the GPU has no mode for (Mali has no 2x MSAA: a 2-sample render target could not be created, and the pass drawing into it was lost, a black or
    // white screen) becomes the nearest one it has, the next higher first (2 → 4), else lower; every target of the same request maps the same way, so a pass's
    // attachments still agree, and a shader's sample index stays inside the image
    if (ci.samples > VK_SAMPLE_COUNT_1_BIT) {
        VkImageFormatProperties ifp; VkSampleCountFlags ok = VK_SAMPLE_COUNT_1_BIT;
        if (!vkGetPhysicalDeviceImageFormatProperties(g_pd, ci.format, ci.imageType, ci.tiling, ci.usage, ci.flags, &ifp)) ok = ifp.sampleCounts;
        if (!(ok & ci.samples)) {
            VkSampleCountFlagBits want = ci.samples, got = VK_SAMPLE_COUNT_1_BIT;
            for (uint32_t s = (uint32_t)want << 1; s <= VK_SAMPLE_COUNT_16_BIT; s <<= 1) if (ok & s) { got = (VkSampleCountFlagBits)s; break; }
            if (got == VK_SAMPLE_COUNT_1_BIT) for (uint32_t s = (uint32_t)want >> 1; s > 1; s >>= 1) if (ok & s) { got = (VkSampleCountFlagBits)s; break; }
            RED_ONCE(77, "the GPU has no %u-sample mode for format %d: %u samples instead (Mali: no 2x MSAA)", (unsigned)want, ci.format, (unsigned)got);
            ci.samples = got;
        }
    }
    x->w = ci.extent.width; x->h = ci.extent.height; x->d = ci.extent.depth; x->levels = ci.mipLevels; x->layers = ci.arrayLayers; x->samples = ci.samples; x->usage = ci.usage; x->aspect = fmt_aspect(x->fmt);
    VkResult r = vkCreateImage(g_dev, &ci, NULL, &x->img);
    if (r && b10nc && g_compctl) {   // (a driver that refuses the request: the target as it was made before, said once)
        RED_ONCE(153, "b10 nocompress: the driver refused an R11G11B10F render target with framebuffer compression disabled (%d): made as before", r);
        ci.pNext = ncc.pNext; b10nc = 0; r = vkCreateImage(g_dev, &ci, NULL, &x->img); }
    if (r) { RED_ONCE(3, "vkCreateImage %ux%ux%u format %#x (Vulkan %u) target %d failed: %d", x->w, x->h, x->d, t->d.format, vf, t->d.target, r); free(x); return NULL; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, x->img, &rq);
    x->a = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); atomic_fetch_add(&g_live_tex_bytes, (long)x->a.size);
    VKCHECK(vkBindImageMemory(g_dev, x->img, x->a.b->mem, x->a.off));
    static int logged;
    if (p_layout2 && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && logged < 8) {   // the first render targets: is their framebuffer compression on?
        VkImageCompressionPropertiesEXT cp = { .sType = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT };
        VkSubresourceLayout2EXT sl = { .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT, .pNext = &cp };
        VkImageSubresource2EXT sr = { .sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 } };
        p_layout2(g_dev, x->img, &sr, &sl); logged++;
        VLOG("render target %ux%u format %d: framebuffer compression %s (flags %#x)", x->w, x->h, x->fmt,
             cp.imageCompressionFlags & VK_IMAGE_COMPRESSION_DISABLED_EXT ? "OFF" : "on", (unsigned)cp.imageCompressionFlags);
    }
    // R11G11B10F render targets (NVN 0x3F), whatever the switches (a log line only): the first two (with DM_VK_B10_NOCOMPRESS the first six) named with how they
    //   were made (create flags, usage, the view formats listed: what the drivers key framebuffer compression on) and what the driver then reports, where it can.
    //   A normal log shows whether the lit scene's target is compressed at all, which route B needs; route B's own report proves nothing on Arm's driver, which
    //   answers "compression OFF" for every render target it is asked about (players' logs: #128 #10 #19 #21, Mali-G610 to G925), so a route's result is read off
    //   the picture (the white), never off these lines
    static int b10_said;
    if (t->d.format == 0x3F && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && b10_said < (b10nc ? 6 : 2)) { b10_said++;
        char fl[80] = "none (not mutable)"; const VkBaseInStructure *pn = ci.pNext;
        for (; pn; pn = pn->pNext) if (pn->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) { const VkImageFormatListCreateInfo *l = (const void *)pn; int o = 0;
            for (uint32_t i = 0; i < l->viewFormatCount && o < (int)sizeof fl - 12; i++) o += snprintf(fl + o, sizeof fl - (size_t)o, "%s%d", i ? " " : "", (int)l->pViewFormats[i]); break; }
        if (!pn && (ci.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) snprintf(fl, sizeof fl, "none listed (mutable: any compatible format)");
        const char *how = !b10nc ? "as usual (DM_VK_B10_NOCOMPRESS off)" : g_compctl ? "with compression disabled (DM_VK_B10_NOCOMPRESS, VK_EXT_image_compression_control)"
                        : "mutable with no list of view formats (DM_VK_B10_NOCOMPRESS; no VK_EXT_image_compression_control)";
        char rep[64] = "the driver cannot report its compression";
        if (p_layout2) { VkImageCompressionPropertiesEXT cp = { .sType = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT };
            VkSubresourceLayout2EXT sl = { .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT, .pNext = &cp };
            VkImageSubresource2EXT sr = { .sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 } };
            p_layout2(g_dev, x->img, &sr, &sl);
            snprintf(rep, sizeof rep, "the driver reports framebuffer compression %s (flags %#x)", cp.imageCompressionFlags & VK_IMAGE_COMPRESSION_DISABLED_EXT ? "OFF" : "on", (unsigned)cp.imageCompressionFlags); }
        VLOG("b10 target: R11G11B10F render target %ux%u, %u levels, stored as vk format %d%s: create flags %#x, usage %#x, view formats %s; made %s; %s", x->w, x->h, x->levels,
             (int)x->fmt, b10_swapped_tex(x) ? " (the RGBA16F stand-in, DM_VK_B10_AS_RGBA16)" : "", (unsigned)ci.flags, (unsigned)ci.usage, fl, how, rep);
    }
    return x;
}
static VkImageView view_get(vtex *t, VkImageViewType vt, VkFormat fmt, uint32_t lvl, uint32_t nlvl, uint32_t lay, uint32_t nlay, const uint8_t sw[4], VkImageAspectFlags aspect) {
    fmt = b10_view_fmt(t, fmt, B10V_SAMPLED, NULL); if (!fmt) return VK_NULL_HANDLE;   // (DM_VK_B10_AS_RGBA16: no view across texel sizes; the callers map first, with the use)
    if (t->fixed_fmt && fmt != t->fmt && fmt != t->twin) { RED_ONCE(73, "a view in format %d of an image whose formats are fixed to %d / %d: viewed in its own format (DM_VK_NO_FMT_LIST=1 to compare)", fmt, t->fmt, t->twin); fmt = t->fmt; }
    vview_key k = { (uint32_t)fmt, (uint32_t)vt, lvl, nlvl, lay, nlay, (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24, (uint32_t)aspect };
    pthread_mutex_lock(&t->vmu);
    for (int i = 0; i < t->nviews; i++) if (!memcmp(&t->views[i].k, &k, sizeof k)) { VkImageView v = t->views[i].v; pthread_mutex_unlock(&t->vmu); return v; }
    VkImageViewUsageCreateInfo uci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = t->usage };
    if (t->is3d && (vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY)) uci.usage &= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;   // a 2D(-array) view of a 3D image is for rendering only
    // a view in a format whose features lack some of the image's usages (sRGB views of a storage image): the view only gets what its format supports
    if (fmt != t->fmt) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, fmt, &fp); VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
        if (!(ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) uci.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
        if (!(ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) uci.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (!(ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) uci.usage &= ~VK_IMAGE_USAGE_SAMPLED_BIT; }
    VkImageViewCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &uci, .image = t->img, .viewType = vt, .format = fmt,
        .components = { (VkComponentSwizzle)sw[0], (VkComponentSwizzle)sw[1], (VkComponentSwizzle)sw[2], (VkComponentSwizzle)sw[3] },
        .subresourceRange = { aspect, lvl, nlvl, lay, nlay } };
    VkImageView v = VK_NULL_HANDLE; VkResult r = vkCreateImageView(g_dev, &ci, NULL, &v);
    if (r) { RED_ONCE(4, "vkCreateImageView type %d format %d (image format %d %ux%ux%u layers %u) levels %u+%u layers %u+%u failed: %d", vt, fmt, t->fmt, t->w, t->h, t->d, t->layers, lvl, nlvl, lay, nlay, r); v = VK_NULL_HANDLE; }
    else { t->views = realloc(t->views, (t->nviews + 1) * sizeof *t->views); t->views[t->nviews++] = (vview){k, v}; }
    pthread_mutex_unlock(&t->vmu);
    return v;
}

static uint64_t g_bc_gpu_levels, g_bc_gpu_copies; static char g_bc_line[64];   // levels / copies the BC layer converted (the memory line)
// one level / layer of BC blocks (z slices) into staging: transcoded (etk; read from the texture cache when it was converted before, tex_cache.h) or decoded
static void bc_convert(int kind, int etk, const uint8_t *src, size_t pitch, size_t rows, int w, int h, uint32_t z, uint8_t *out, size_t opitch, size_t orows) {
    size_t in = pitch * rows * z, on = opitch * orows * z; uint64_t key = 0; uint8_t *tmp = NULL;
    if (etk) { key = tex_cache_key(src, in, kind, w, h, (int)z); if (tex_cache_get(key, out, on)) return; tmp = malloc(on); }   // (converted into memory first: staging may be slow to read back)
    uint8_t *o = tmp ? tmp : out;
    for (uint32_t s = 0; s < z; s++) (etk ? bc_transcode_image : bc_decode_image_mt)(kind, src + s * pitch * rows, pitch, w, h, o + s * opitch * orows, opitch);
    if (tmp) { memcpy(out, tmp, on); tex_cache_put(key, tmp, on); free(tmp); }
}
// CPU-filled texture upload: deswizzled level by level / layer by layer into staging (the submission's arena, else a one-off buffer), copied into the image in the upload command buffer
// Time spent here (the long-frame line's upload ms: mtl_upload_ns_take) and in the deswizzle / copy part of it; one upload over 20 ms is logged with its size and split
static _Atomic uint64_t g_upload_ns;
static __thread int t_pre_worker;   // (this thread is the upload-ahead worker, pre_main: its uploads are not waited for by a draw)
static _Atomic uint64_t g_upl_fg_n, g_upl_fg_ns;   // uploads not made by the upload-ahead worker (a draw's first use, a re-upload): a drawing thread waited for them (the load report)
void mtl_upload_fg_stats(uint64_t *n, uint64_t *ns) { *n = atomic_load(&g_upl_fg_n); *ns = atomic_load(&g_upl_fg_ns); }
// A batch of uploads (the pre-upload worker, pre_main): this thread's uploads go into one command buffer with their own staging buffers, submitted and waited for once
//   at the end of the batch instead of once each (each texture was a submission and a wait of its own; an idea from a fork's v0.99, written here)
enum { UPB_MAX = 32 };
typedef struct { VkCommandBuffer cb; int n; VkBuffer sb[UPB_MAX]; vmalloc_t tmp[UPB_MAX]; } upb_t;
static __thread upb_t *t_upb;
static int tex_upload_(vtex *x, const tex_t *t, uint64_t *copy_ns);
static int tex_upload(vtex *x, const tex_t *t) {
    pool_mark_tex(t->mem);
    uint64_t t0 = mono_ns(), cn = 0; int r = tex_upload_(x, t, &cn); uint64_t dt = mono_ns() - t0;
    atomic_fetch_add(&g_upload_ns, dt); if (!t_pre_worker) { atomic_fetch_add(&g_upl_fg_n, 1); atomic_fetch_add(&g_upl_fg_ns, dt); }
    if (dt > 20000000ull) { static int n; if (n++ < 40) VLOG("slow texture upload: %ux%ux%u, %u levels, %u layers, format %#x: %.1f ms (deswizzle / copy %.1f ms)",
                                                            x->w, x->h, x->d, x->levels, x->layers, t->d.format, dt / 1e6, cn / 1e6); }
    return r;
}
// a level / layer's bytes in an upload's staging: its texels as uploaded (o*: decoded or transcoded with BC), or with BC the blocks themselves if those are larger
static VkDeviceSize up_slot(int bck, int bw, int bh, int bpb, int obw, int obh, int obpb, uint32_t w, uint32_t h, uint32_t z) {
    VkDeviceSize out = (VkDeviceSize)((w + obw - 1) / obw) * obpb * ((h + obh - 1) / obh), blocks = (VkDeviceSize)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh);
    return z * (bck && blocks > out ? blocks : out);
}
static int tex_upload_(vtex *x, const tex_t *t, uint64_t *copy_ns) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) return -1;
    // BC on a GPU without it: the blocks are deswizzled into scratch and decoded into staging as plain texels (bc_decode.c)
    int b10 = b10_swapped_tex(x) && bpb == 4 && bw == 1;   // (expanded from scratch, as BC is decoded)
    int bck = nvn_fmt_vk_decode(t->d.format), etk = nvn_fmt_vk_transcode(t->d.format), obpb = etk ? bc_transcode_bytes(etk) : bck ? bc_texel_bytes(bck) : b10 ? 8 : bpb,
        obw = bck && !(etk && bc_transcode_blocks(etk)) ? 1 : bw, obh = bck && !(etk && bc_transcode_blocks(etk)) ? 1 : bh; uint8_t *scratch = NULL;   // (transcoded: ETC2 / EAC blocks, 4x4 like BC)
    unsigned layers = x->is3d ? 1 : x->layers;
    VkDeviceSize total = 0;
    // each level / layer's slot in staging: its converted texels, or with BC the blocks the BC layer reads from it if those are larger (a 1x1 level: 16 bytes of
    //   BC6H block, 8 of RGBA16F texel; with the converted size only, the next layer's blocks overwrote this one's second half before the GPU read it: the Mali white
    //   screen, BC6H cubemaps' 1x1 level, #110)
    #define SLOT(w, h, z) up_slot(bck, bw, bh, bpb, obw, obh, obpb, w, h, z)
    for (unsigned l = 0; l < x->levels; l++) { uint32_t w = x->w >> l ? x->w >> l : 1, h = x->h >> l ? x->h >> l : 1, z = x->is3d ? (x->d >> l ? x->d >> l : 1) : 1;
        total += SLOT(w, h, z) * layers; }
    // DM_VK_BC_GPU_VERIFY=<kind> (a test switch): the first 8 textures of that kind the BC layer converts, uploaded on their own and waited for, read back and compared
    //   with the CPU's decode; the first difference logged
    static int vkind = -100, vleft = 8; if (vkind == -100) { const char *e = getenv("DM_VK_BC_GPU_VERIFY"); vkind = e && *e ? atoi(e) : 0; }
    int vfy = vkind && bck == vkind && !etk && (g_bcgpu & 1) && bc_gpu_can(bck, 0) && !x->is3d && vleft > 0; if (vfy) vleft--;
    VkBuffer vb = VK_NULL_HANDLE; vmalloc_t vm = {0}; uint8_t *vexp = NULL, *vdone = vfy ? calloc(x->levels * layers, 1) : NULL;
    int batch = !vfy && t_upb && t_upb->n < UPB_MAX;   // (a batch: its command buffer, a staging buffer of its own kept until the batch is done)
    VkCommandBuffer cb = vfy ? NULL : batch ? t_upb->cb : ucb(); int once = !cb; if (once) cb = once_begin();
    if (vfy) { VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT }; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &vb));
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, vb, &rq); vm = vk_alloc_c(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, VA_TEMP);
        VKCHECK(vkBindBufferMemory(g_dev, vb, vm.b->mem, vm.off)); vexp = calloc(1, total ? total : 16); memset((uint8_t *)vm.b->map + vm.off, 0xcd, total); }
    VkBuffer sb; VkDeviceSize so = 0; uint8_t *sp; vmalloc_t tmp = {0};
    if (!once && !batch) sp = arena_alloc(g_sub, total ? total : 16, &sb, &so);
    else { VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT }; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &sb));
           VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, sb, &rq); tmp = vk_alloc_c(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, VA_TEMP);
           VKCHECK(vkBindBufferMemory(g_dev, sb, tmp.b->mem, tmp.off)); sp = (uint8_t *)tmp.b->map + tmp.off; }
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .image = x->img, .subresourceRange = { x->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkBufferImageCopy *rg = calloc(x->levels * layers, sizeof *rg); uint32_t nrg = 0; VkDeviceSize off = 0; int linear = !!(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET));
    for (unsigned l = 0; l < x->levels; l++) {
        uint32_t w = x->w >> l ? x->w >> l : 1, h = x->h >> l ? x->h >> l : 1, z = x->is3d ? (x->d >> l ? x->d >> l : 1) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh, opitch = (size_t)((w + obw - 1) / obw) * obpb, orows = (h + obh - 1) / obh;
        if (bck || b10) { free(scratch); scratch = malloc(pitch * rows * z); }
        for (unsigned y = 0; y < layers; y++) {
            uint8_t *dst = bck || b10 ? scratch : sp + off; uint64_t c0 = mono_ns();
            if (linear) {   // LINEAR: rows at the texture's stride (one level, one layer)
                size_t stride = t->d.stride ? t->d.stride : pitch; for (size_t r = 0; r < rows; r++) memcpy(dst + r * pitch, t->mem + r * stride, pitch);
            } else if (nvn_deswizzle(&t->d, l + (unsigned)x->drop, y, t->mem, dst, pitch)) { RED_ONCE(5, "deswizzle failed format %#x target %d", t->d.format, t->d.target); break; }
            *copy_ns += mono_ns() - c0;
            // BC blocks: converted by the BC layer from the raw blocks in staging (they fit: no conversion makes a level smaller), else on the CPU as before
            if (bck && (g_bcgpu & 1) && bc_gpu_can(bck, etk ? 1 : 0)) {
                memcpy(sp + off, scratch, pitch * rows * z);
                VkImageSubresourceLayers sub = { x->aspect, l, x->is3d ? 0 : y, 1 };
                if (bc_gpu_buffer_to_image(cb, bck, etk ? 1 : 0, sb, so + off, (uint32_t)pitch, (uint32_t)(pitch * rows), w, h, z, x->img, sub, (VkOffset3D){0, 0, 0}, x->is3d)) {
                    if (vfy) { vdone[l * layers + y] = 1; barrier_all(cb); bc_decode_image(bck, scratch, pitch, (int)w, (int)h, vexp + off, opitch);
                        VkBufferImageCopy r = { .bufferOffset = off, .bufferRowLength = w, .bufferImageHeight = h, .imageSubresource = sub, .imageExtent = { w, h, 1 } };
                        vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, vb, 1, &r); barrier_all(cb); }
                    g_bc_gpu_levels++; off += SLOT(w, h, z); continue; }
            }
            if (bck) bc_convert(bck, etk, scratch, pitch, rows, (int)w, (int)h, z, sp + off, opitch, orows);
            else if (b10) { b10_rows_expand(scratch, pitch, w, rows * z, sp + off, opitch); if (!l && !y) b10_note("texels from the game's memory expanded on upload", x, 0); }
            rg[nrg++] = (VkBufferImageCopy){ .bufferOffset = so + off, .bufferRowLength = (uint32_t)(opitch / obpb * obw), .bufferImageHeight = (uint32_t)(orows * obh),
                .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, l, x->is3d ? 0 : y, 1 }, .imageExtent = { w, h, z } };
            off += SLOT(w, h, z);
        }
    }
    free(scratch);
    if (nrg) vkCmdCopyBufferToImage(cb, sb, x->img, VK_IMAGE_LAYOUT_GENERAL, nrg, rg);
    free(rg); g_up_bytes += total;
    if (once) once_end_async(cb, sb, tmp);
    else if (batch) { t_upb->sb[t_upb->n] = sb; t_upb->tmp[t_upb->n] = tmp; t_upb->n++; }
    if (vfy) {   // texel by texel (8 bytes: RGBA16F), level by level, layer by layer, in the staging layout
        const uint8_t *got = (const uint8_t *)vm.b->map + vm.off; VkDeviceSize o = 0; int said = 0; unsigned long long bad = 0, all = 0;
        for (unsigned l = 0; l < x->levels; l++) { uint32_t w = x->w >> l ? x->w >> l : 1, h = x->h >> l ? x->h >> l : 1;
            for (unsigned y = 0; y < layers; y++, o += SLOT(w, h, 1))
                for (uint32_t i = 0; vdone[l * layers + y] && i < w * h; i++) { all++; if (!memcmp(got + o + i * 8, vexp + o + i * 8, 8)) continue; bad++;
                    if (!said++) { const uint16_t *g = (const uint16_t *)(got + o + i * 8), *c = (const uint16_t *)(vexp + o + i * 8);
                        VLOG("BC verify: texture %ux%u layers %u levels %u format %#x: level %u (%ux%u) layer %u texel %u,%u: GPU %04x %04x %04x %04x, CPU %04x %04x %04x %04x",
                             x->w, x->h, layers, x->levels, t->d.format, l, w, h, y, i % w, i / w, g[0], g[1], g[2], g[3], c[0], c[1], c[2], c[3]); } } }
        VLOG("BC verify: texture %ux%u layers %u levels %u format %#x: %llu of %llu texels differ (kind %d on the BC layer)", x->w, x->h, layers, x->levels, t->d.format, bad, all, vkind);
        vkDestroyBuffer(g_dev, vb, NULL); vk_free(vm); free(vexp); free(vdone);
    }
    #undef SLOT
    return 0;
}
static vtex *pkg_upload(const tex_t *t) {
    vtex *x = tex_new(t, 1); if (!x) return NULL;
    nvn_dirty_protect(t->mem, t->L.size); x->stamp = nvn_dirty_now();
    if (tex_upload(x, t)) { vtex_unref(x); return NULL; }
    x->pkg = 1; x->span = nvn_dirty_span(t->mem, t->L.size); x->seen = nvn_dirty_now();
    return x;
}
static vtex *serial_get(uint64_t k) {   // caller holds g_smu
    vtex *x = nvn_hm_get(&g_by_serial, &k); if (!x && g_ghost.n) x = nvn_hm_get(&g_ghost, &k); return x;
}
static void serial_put(uint64_t k, vtex *x) {   // takes a reference for the table; caller holds g_smu
    nvn_hmap *m = nvn_hm_get(&g_tex_live, &k) ? &g_by_serial : &g_ghost;
    vtex_ref(x); vtex *old = nvn_hm_put(m, &k, x);
    if (old) vtex_unref(old);
    else if (m == &g_ghost) { uint64_t *slot = &g_ghost_ring[g_ghost_next++ % NGHOST]; if (*slot) { vtex *ev = nvn_hm_del(&g_ghost, slot); if (ev) vtex_unref(ev); } *slot = k; }
    g_ser_gen++;
}
void mtl_tex_init(const tex_t *t) { if (!t->serial) return; pthread_mutex_lock(&g_smu); nvn_hm_put(&g_tex_live, &t->serial, (void *)1); pthread_mutex_unlock(&g_smu); }
// ================= Packaged textures uploaded ahead (DM_VK_PREUPLOAD=0: off) =================
// A texture from the game's packages was uploaded the first time a draw sampled it: deswizzled (and on GPUs without BC converted) on the game's own thread inside that frame.
// Streaming a new area brings hundreds at once (Legends: Arceus flying over the Fieldlands: 350 textures, 86 MB in ten seconds, frames of 100 ms to 1.1 s, a 30 fps that
// dipped to 26). Now each one is handed, as the game initializes it (hle_nvn.c nvnTextureInitialize, after its packaged data is in place), to a worker that uploads it
// through a one-off command buffer (pkg_upload off the submit thread: once_begin / once_end, ordered after everything submitted) and files it under its serial; the first
// draw then finds it uploaded. One that changes afterwards is caught by the dirty pages as before (pkg_upload protects them before it reads) and uploaded again at its first
// use; one used before the worker got to it is uploaded there as before and the worker's copy dropped. The worker runs below the game's threads' priority
enum { PRE_N = 2048 };
static tex_t g_pre[PRE_N]; static unsigned g_pre_head, g_pre_tail; static pthread_mutex_t g_pre_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pre_cv = PTHREAD_COND_INITIALIZER;
static _Atomic uint64_t g_pre_done, g_pre_late, g_pre_full;
// how much the worker may keep uploaded ahead and not drawn yet: 512 MB on phones of 11.5 GB and up, 256 MB from 7.5 GB, 96 MB below (6 GB phones are tight, 4 GB a wall:
//   docs/device-data.txt); DM_VK_PREUPLOAD_MB sets it. Over it, a texture waits for its first use as before
static long pre_budget(void) {
    static long b = -1; if (b >= 0) return b;
    const char *e = getenv("DM_VK_PREUPLOAD_MB"); long mb = e ? atol(e) : 0;
    if (!e) { long kb = 0; FILE *f = fopen("/proc/meminfo", "r"); char ln[128]; while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemTotal: %ld kB", &kb) == 1) break; if (f) fclose(f);
              double gb = kb / 1048576.0; mb = gb >= 11.0 ? 512 : gb >= 7.0 ? 256 : 96; }   // (MemTotal is a little under the phone's rating: 12 GB shows about 11)
    b = mb << 20; VLOG("packaged textures: up to %ld MB uploaded ahead and not drawn yet", mb); return b;
}
// The budget in a loading screen (fable-loads.md L4; hle_nvn.c dm_loading): #112's load left 156 textures to their first use (budget or queue full) and spent
//   325 ms of one 1796 ms frame uploading 1436 of them on the drawing thread. DM_VK_PRE_BUDGET_MB=<n> raises the budget to n MB while a load is on and the phone
//   has 2.5 GB or more available (MemAvailable, read at most once a second); unset (the default until the load report's counters are measured on phones):
//   the budget stays as above. The counters split what was left to its first use by the budget and by the full queue (the load report, hle_nvn.c)
__attribute__((weak)) int dm_loading(void) { return 0; }   // (hle_nvn.c; none in t_vkdev)
static _Atomic uint64_t g_pre_full_budget, g_pre_full_queue;
static long pre_budget_now(void) {
    static long load_b = -1; if (load_b < 0) { const char *e = getenv("DM_VK_PRE_BUDGET_MB"); load_b = e && atol(e) > 0 ? atol(e) << 20 : 0;
        if (load_b) VLOG("packaged textures: up to %ld MB uploaded ahead in a loading screen when 2.5 GB or more is available (DM_VK_PRE_BUDGET_MB)", load_b >> 20); }
    long b = pre_budget(); if (!load_b || load_b <= b || !dm_loading()) return b;
    static uint64_t at; static long avail_kb; uint64_t now = mono_ns();
    if (!at || now - at > 1000000000ull) { at = now; avail_kb = 0; FILE *f = fopen("/proc/meminfo", "r"); char ln[128]; while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld kB", &avail_kb) == 1) break; if (f) fclose(f); }
    return avail_kb >= 2621440 ? load_b : b;   // (2.5 GB)
}
void mtl_pre_stats(uint64_t *budget, uint64_t *queue) { *budget = atomic_load(&g_pre_full_budget); *queue = atomic_load(&g_pre_full_queue); }
static int pre_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PREUPLOAD"); v = !(e && !strcmp(e, "0")); VLOG("packaged textures: %s", v ? "uploaded ahead by a worker as the game creates them" : "uploaded at their first use (DM_VK_PREUPLOAD=0)"); } return v; }
// The worker's batch: its own command pool (no lock shared with the one-off uploads), a full barrier on both sides, submitted after what was submitted so far and waited
//   for here. Held work is not flushed for it (a one-off upload did that each time): what it uploads is new images nothing has drawn yet
static VkCommandPool g_pre_pool;
static void pre_batch_begin(upb_t *b) {
    if (!g_pre_pool) { VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf };
                       VKCHECK(vkCreateCommandPool(g_dev, &pci, NULL, &g_pre_pool)); }
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pre_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &b->cb));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKCHECK(vkBeginCommandBuffer(b->cb, &bi)); barrier_all(b->cb); b->n = 0;
}
static void pre_batch_end(upb_t *b) {
    barrier_all(b->cb); barrier_host(b->cb); VKCHECK(vkEndCommandBuffer(b->cb));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &f));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &b->cb };
    shadow_sync();
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, f); pthread_mutex_unlock(&g_qmu);
    if (r) VDIE("pre-upload batch submit failed: %d", r);
    vk_wait(NULL, f, UINT64_MAX); vkDestroyFence(g_dev, f, NULL);
    vkFreeCommandBuffers(g_dev, g_pre_pool, 1, &b->cb); b->cb = VK_NULL_HANDLE;
    for (int i = 0; i < b->n; i++) { vkDestroyBuffer(g_dev, b->sb[i], NULL); vk_free(b->tmp[i]); }
    b->n = 0;
}
static int pre_batch_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PRE_BATCH"); v = e && atoi(e) == 1 ? 0 : 1; } return v; }   // (DM_VK_PRE_BATCH=1: one at a time, as before)
static _Atomic uint64_t g_pre_batches;
static void *pre_main(void *unused) { DM_THREAD_NAME("dm-texupload"); t_pre_worker = 1;
    (void)unused; setpriority(PRIO_PROCESS, (id_t)gettid(), 4);   // (below the game's threads: it only saves them work)
    static upb_t b; static tex_t cs[UPB_MAX]; static vtex *xs[UPB_MAX];
    for (;;) {
        // what is waiting, up to UPB_MAX textures or 24 MB of their content: uploaded together
        tex_t got[UPB_MAX]; int ng = 0;
        pthread_mutex_lock(&g_pre_mu);
        while (g_pre_tail == g_pre_head) pthread_cond_wait(&g_pre_cv, &g_pre_mu);
        { size_t bytes = 0; int most = pre_batch_on() ? UPB_MAX : 1;
          while (g_pre_tail != g_pre_head && ng < most && (ng == 0 || bytes + g_pre[g_pre_tail % PRE_N].L.size <= (24u << 20))) { got[ng] = g_pre[g_pre_tail++ % PRE_N]; bytes += got[ng].L.size; ng++; } }
        pthread_mutex_unlock(&g_pre_mu);
        int nx = 0; int batched = ng > 1;
        if (batched) { pre_batch_begin(&b); t_upb = &b; }
        for (int i = 0; i < ng; i++) {
            tex_t c = got[i];
            pthread_mutex_lock(&g_smu); int want = nvn_hm_get(&g_tex_live, &c.serial) && !serial_get(c.serial); pthread_mutex_unlock(&g_smu);
            if (!want) { atomic_fetch_add(&g_pre_late, 1); continue; }   // (released, or already uploaded by its first use)
            if (atomic_load(&g_pre_unused) + (long)c.L.size > pre_budget_now()) { atomic_fetch_add(&g_pre_full, 1); atomic_fetch_add(&g_pre_full_budget, 1); continue; }   // (over the budget: its first use uploads it)
            vtex *x = pkg_upload(&c); if (!x) continue;
            cs[nx] = c; xs[nx] = x; nx++;
        }
        if (batched) { t_upb = NULL; pre_batch_end(&b); atomic_fetch_add(&g_pre_batches, 1); }   // (the GPU has them all now: only then are they handed out)
        for (int i = 0; i < nx; i++) {
            vtex *x = xs[i]; tex_t *c = &cs[i];
            pthread_mutex_lock(&g_smu);
            if (nvn_hm_get(&g_tex_live, &c->serial) && !serial_get(c->serial)) { atomic_store(&x->pre, 1); atomic_fetch_add(&g_pre_unused, (long)x->a.size); serial_put(c->serial, x); atomic_fetch_add(&g_pre_done, 1); g_pkg_uploads++; }
            else atomic_fetch_add(&g_pre_late, 1);
            pthread_mutex_unlock(&g_smu); vtex_unref(x);
        }
    }
    return NULL;
}
void mtl_tex_preupload(const tex_t *t) {   // hle_nvn.c nvnTextureInitialize: a texture whose content is in place
    if (!t->serial || !t->mem || !t->has_layout || t->d.target == NVN_TEXTURE_TARGET_BUFFER || !g_dev || !pre_on()) return;
    static int started;
    pthread_mutex_lock(&g_pre_mu);
    if (!started) { started = 1; pthread_t th; if (pthread_create(&th, NULL, pre_main, NULL)) { started = -1; } else pthread_detach(th); }
    if (started < 0 || g_pre_head - g_pre_tail >= PRE_N) { atomic_fetch_add(&g_pre_full, 1); atomic_fetch_add(&g_pre_full_queue, 1); pthread_mutex_unlock(&g_pre_mu); return; }   // (full: uploaded at its first use)
    g_pre[g_pre_head++ % PRE_N] = *t; pthread_cond_signal(&g_pre_cv);
    pthread_mutex_unlock(&g_pre_mu);
}
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    pthread_mutex_lock(&g_smu); nvn_hm_del(&g_tex_live, &t->serial); vtex *x = nvn_hm_del(&g_by_serial, &t->serial); if (x) { vtex_unref(x); g_ser_gen++; } pthread_mutex_unlock(&g_smu);
}
void mtl_tex_cpu_written(const tex_t *t) {
    if (!t->serial) return;
    pthread_mutex_lock(&g_smu); if (!nvn_hm_get(&g_cpu_tex, &t->serial)) { nvn_hm_put(&g_cpu_tex, &t->serial, (void *)1); g_ser_gen++; } pthread_mutex_unlock(&g_smu);
}
int mtl_tex_gpu_owned(uint64_t serial) {
    pthread_mutex_lock(&g_smu); vtex *x = serial_get(serial); int r = x && !x->pkg && (x->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT)); pthread_mutex_unlock(&g_smu);
    return r;
}
void mtl_alias_drop_pool(const void *pool) {
    pthread_mutex_lock(&g_smu);
    for (uint32_t i = 0; i < g_alias.cap; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)pool) continue;
        vtex *x = nvn_hm_del(&g_alias, &k); if (x) vtex_unref(x); i = (uint32_t)-1;   // deletion shifts entries: rescan
    }
    g_ser_gen++;
    pthread_mutex_unlock(&g_smu);
}
// Texture object → its image (nvn_mtl.m tex_mtl_): CPU-filled ones (packaged layout, CPU-written, linear) uploaded and re-uploaded when their pages change; buffer textures as views of
// their pool; the rest are GPU images shared by storage + description. The returned image is valid until the submission completes (tables hold it, replacements are deferred)
static void golden_vk_tex_ref(const tex_t *t);   // (golden_vk.inc)
static vtex *tex_res(const tex_t *t) {
    if (!t || !t->serial) return NULL;
    if (__builtin_expect(g_golden == 2, 0)) golden_vk_tex_ref(t);
    if (t->magic != MAGIC('T', 'E', 'X', 0)) { RED_ONCE(6, "texture object no longer valid (magic %#x, ser=%llu)", t->magic, (unsigned long long)t->serial); return NULL; }
    pthread_mutex_lock(&g_smu);
    int linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && t->d.target != NVN_TEXTURE_TARGET_BUFFER && t->pool && !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    int packaged = (t->has_layout || nvn_hm_get(&g_cpu_tex, &t->serial) || linear) && t->mem && t->d.target != NVN_TEXTURE_TARGET_BUFFER;
    vtex *have = serial_get(t->serial);
    if (packaged) {
        if (have && have->pkg) {
            uint64_t now = nvn_dirty_now();
            int clean = have->span && (now == have->seen || !nvn_dirty_since_span(have->span, t->mem, t->L.size, have->stamp));
            have->seen = now;
            if (atomic_exchange(&have->pre, 0)) atomic_fetch_sub(&g_pre_unused, (long)have->a.size);   // (uploaded ahead: now drawn)
            if (clean) { pthread_mutex_unlock(&g_smu); return have; }
            { static int nl; if (nl < 400 && getenv("DM_VK_REUP")) { nl++; VLOG("re-upload ser=%llu %ux%u format %#x target %d flags %#x layout %d pool flags %#x span %p seen %llu now %llu", (unsigned long long)t->serial, t->d.width, t->d.height,
                t->d.format, t->d.target, t->d.flags, t->has_layout, t->pool ? t->pool->flags : 0, have->span, (unsigned long long)have->seen, (unsigned long long)now); } }
            vtex *nt = pkg_upload(t);   // content changed: a new image (earlier draws keep the old one until they complete)
            if (nt) { serial_put(t->serial, nt); vtex_unref(nt); g_pkg_reuploads++; pthread_mutex_unlock(&g_smu); return nt; }
            pthread_mutex_unlock(&g_smu); return have;
        }
        vtex *nt = pkg_upload(t);
        if (nt) { serial_put(t->serial, nt); vtex_unref(nt); g_pkg_uploads++; }
        pthread_mutex_unlock(&g_smu); return nt;
    }
    if (have) { pthread_mutex_unlock(&g_smu); return have; }
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) {
        uint32_t vf = 0; uint8_t swz[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        VkBuffer b; VkDeviceSize o, left;
        if (!nvn_fmt_to_vk(t->d.format, &vf, swz) || !vf || !bpb || !t->mem || !pool_buf((uint64_t)(uintptr_t)t->mem, (uint64_t)t->d.width * bpb, &b, &o, &left)) {
            RED_ONCE(7, "buffer texture format %#x / storage not usable", t->d.format); pthread_mutex_unlock(&g_smu); return NULL; }
        if (o % g_props.limits.minTexelBufferOffsetAlignment) { RED_ONCE(8, "buffer texture offset %#llx not aligned to %llu", (unsigned long long)o, (unsigned long long)g_props.limits.minTexelBufferOffsetAlignment); pthread_mutex_unlock(&g_smu); return NULL; }
        if (nvn_fmt_vk_b10_swapped(t->d.format)) { vf = VK_FORMAT_B10G11R11_UFLOAT_PACK32; b10_note("a buffer texture: kept in the packed format (its texels are the game's memory)", NULL, (int)vf); }
        vtex *x = calloc(1, sizeof *x); atomic_fetch_add(&g_live_tex, 1); x->refs = 1; x->fmt = (VkFormat)vf; x->nvn_fmt = t->d.format; pthread_mutex_init(&x->vmu, NULL);
        VkDeviceSize n = (VkDeviceSize)t->d.width * bpb; if (n > left) n = left;
        VkBufferViewCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = b, .format = x->fmt, .offset = o, .range = n };
        if (vkCreateBufferView(g_dev, &bci, NULL, &x->bview)) { RED_ONCE(9, "buffer texture view failed (format %u, %llu bytes)", vf, (unsigned long long)n); free(x); pthread_mutex_unlock(&g_smu); return NULL; }
        serial_put(t->serial, x); vtex_unref(x); pthread_mutex_unlock(&g_smu); return x;
    }
    alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d;
    vtex *shared = nvn_hm_get(&g_alias, &ak);
    // DM_VK_LOG_ALIAS (diagnosis): a new texture over the start of memory a render target already uses, but not matched above: both descriptions (the black ID photo)
    if (!shared) { static int on = -1, n; if (on < 0) on = getenv("DM_VK_LOG_ALIAS") != NULL;
        if (on && n < 40) for (uint32_t i = 0; i < g_alias.cap; i++) {
            if (!g_alias.vals[i]) continue;
            alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
            if (k.pool != ak.pool || k.off != ak.off) continue;
            vtex *c = (vtex *)g_alias.vals[i]; n++;
            VLOG("alias miss: new %ux%ux%u target %d format %#x flags %#x levels %u ser %llu over render target %ux%ux%u target %d format %#x flags %#x levels %u (usage %#x)",
                 t->d.width, t->d.height, t->d.depth, t->d.target, t->d.format, t->d.flags, t->d.levels, (unsigned long long)t->serial,
                 k.d.width, k.d.height, k.d.depth, k.d.target, k.d.format, k.d.flags, k.d.levels, (unsigned)c->usage);
        } }
    if (shared) { serial_put(t->serial, shared); pthread_mutex_unlock(&g_smu); return shared; }
    vtex *x = tex_new(t, 0);
    if (!x) { pthread_mutex_unlock(&g_smu); return NULL; }
    img_init(x, !(x->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) || 1);
    nvn_hm_put(&g_alias, &ak, x);                 // the sharing table holds the creation reference
    serial_put(t->serial, x);
    pthread_mutex_unlock(&g_smu);
    return x;
}

// ---------- Texture / sampler registries (descriptor pools) ----------
typedef struct { tex_t t; int ok; view_t v; int has_v; } treg_t;
static treg_t *g_treg; static int g_ntreg; static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static VkSampler *g_svk; static int g_nsreg;
static VkSampler *g_svkn; static int g_nsvkn;   // unfiltered twins for integer textures (samp_nearest_by_id)
static int g_tpool_n;
static uint64_t g_tvm_gen = 1;   // tex_view_for's remembered answers hold while this stays (registrations, the pool size, a second image for another BC kind)
void mtl_set_tex_pool_n(int n) { g_tpool_n = n; g_tvm_gen++; }
static int treg_get(int tid, treg_t *out) { if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) { *out = g_treg[tid]; return 1; } return 0; }   // caller holds g_rmu (push_stage)
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    if (tid < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (tid >= g_ntreg) { int n = tid + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_treg[tid].ok = t != NULL; if (t) g_treg[tid].t = *t; g_treg[tid].has_v = v != NULL; if (v) g_treg[tid].v = *v;
    g_tvm_gen++;
    pthread_mutex_unlock(&g_rmu);
}
void mtl_reg_samp(int sid, const samp_par *sp) {
    if (sid < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (sid >= g_nsreg) { int n = sid + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_svk = realloc(g_svk, n * sizeof *g_svk);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_svk + g_nsreg, 0, (n - g_nsreg) * sizeof *g_svk); g_nsreg = n; }
    if (g_svk[sid]) { defer_add((dfr){ .kind = DF_ALLOC }); /* samplers are tiny: the old one is kept (in-flight descriptors may name it) */ g_svk[sid] = VK_NULL_HANDLE; }
    if (sid < g_nsvkn) g_svkn[sid] = VK_NULL_HANDLE;
    g_sreg[sid] = *sp; g_sreg_ok[sid] = 1;
    pthread_mutex_unlock(&g_rmu);
}
static VkSamplerAddressMode wrapm(int w) {
    switch (w) { case 0: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 2: case 3: return g_mclamp ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;   // (fallback: identical for coordinates in [-1, 1])
                 case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}
static VkSampler samp_vk(const samp_par *p) {
    VkSamplerCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    ci.magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = (p->minf == 1 || p->minf == 3 || p->minf == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = p->minf >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrapm(p->ws); ci.addressModeV = wrapm(p->wt); ci.addressModeW = wrapm(p->wr);
    ci.minLod = p->lod_min < 0 ? 0 : p->lod_min; ci.maxLod = p->minf <= 1 ? 0.25f : p->lod_max; ci.mipLodBias = p->lod_bias;   // no mip filter: the base level only (maxLod 0.25, as GL on Vulkan)
    // anisotropic filtering capped (DM_VK_MAX_ANISO, default 8): above 8x the texture units fetch far more for surfaces at a glancing angle for little visible gain
    { static float cap = -1; if (cap < 0) { const char *e = getenv("DM_VK_MAX_ANISO"); cap = e ? (float)atof(e) : 8.0f; if (cap < 1) cap = 1; }
      float want = p->aniso > cap ? cap : p->aniso;
      if (want > 1 && g_f_aniso) { ci.anisotropyEnable = VK_TRUE; ci.maxAnisotropy = want > g_props.limits.maxSamplerAnisotropy ? g_props.limits.maxSamplerAnisotropy : want; } }
    if (p->cmp_mode) { ci.compareEnable = VK_TRUE; ci.compareOp = (VkCompareOp)(p->cmp_func - 1); }
    VkSamplerReductionModeCreateInfo rd = { .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, .reductionMode = p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX };
    VkSamplerCustomBorderColorCreateInfoEXT bc = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, .format = VK_FORMAT_UNDEFINED };
    const void **tail = &ci.pNext;
    if ((p->reduction == 1 || p->reduction == 2) && g_minmax) { *tail = &rd; tail = &rd.pNext; }
    else if (p->reduction == 1 || p->reduction == 2) RED_ONCE(136, "a min / max filtering sampler on a device without it: plain filtering");
    int std = (p->border[0] == 0 && p->border[1] == 0 && p->border[2] == 0 && (p->border[3] == 0 || p->border[3] == 1)) || (p->border[0] == 1 && p->border[1] == 1 && p->border[2] == 1 && p->border[3] == 1);
    if (g_border && !std) { memcpy(bc.customBorderColor.float32, p->border, 16); ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; *tail = &bc; }
    else ci.borderColor = p->border[3] > 0.5f ? (p->border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK) : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s = VK_NULL_HANDLE; if (vkCreateSampler(g_dev, &ci, NULL, &s)) RED_ONCE(10, "vkCreateSampler failed");
    return s;
}
// The same sampler without filtering, for integer-format textures (Vulkan cannot filter them: undefined on strict drivers). Made on first use, kept per ID
static VkSampler samp_nearest_by_id(int sid) {   // caller holds g_rmu
    if (sid < 0 || sid >= g_nsreg || !g_sreg_ok[sid]) return VK_NULL_HANDLE;
    if (sid >= g_nsvkn) { int n = g_nsreg; g_svkn = realloc(g_svkn, n * sizeof *g_svkn); memset(g_svkn + g_nsvkn, 0, (n - g_nsvkn) * sizeof *g_svkn); g_nsvkn = n; }
    if (!g_svkn[sid]) { samp_par q = g_sreg[sid]; q.magf = 0; q.minf = q.minf <= 1 ? 0 : 2; q.aniso = 0; g_svkn[sid] = samp_vk(&q); }
    return g_svkn[sid];
}
// Sampler ID 0 is a real sampler, as in the Metal replay (Pokémon Legends: Arceus samples through it: treated as unbound it fell back to the unfiltered dummy and every gradient banded)
static VkSampler samp_by_id(int sid) {   // caller holds g_rmu
    if (sid < 0 || sid >= g_nsreg || !g_sreg_ok[sid]) return VK_NULL_HANDLE;
    if (!g_svk[sid]) g_svk[sid] = samp_vk(&g_sreg[sid]);
    return g_svk[sid];
}

// ================= Dummies (unbound slots read 0, as on the Switch) =================
static vtex *g_dummy[8][3];   // [view kind][class]: 0 2D, 1 2D array, 2 cube, 3 cube array, 4 3D, 5 2D multisample; class 0 float 1 uint 2 sint
static vtex *g_dummy_depth[4];
static VkBuffer g_zero_buf; static vmalloc_t g_zero_a; static VkBuffer g_support_buf; static vmalloc_t g_support_a;
static VkSampler g_dummy_samp; static VkBufferView g_dummy_tbuf[3];
static vtex *dummy_make(VkFormat f, VkImageType type, uint32_t layers, int cube, VkSampleCountFlagBits samples, int storage) {
    vtex *x = calloc(1, sizeof *x); atomic_fetch_add(&g_live_tex, 1); x->refs = 1; x->fmt = f; pthread_mutex_init(&x->vmu, NULL);
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = type, .format = f, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = layers, .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0), .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0 };
    if (samples > 1) ci.usage |= fmt_is_depth(f) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VKCHECK(vkCreateImage(g_dev, &ci, NULL, &x->img));
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, x->img, &rq); x->a = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); atomic_fetch_add(&g_live_tex_bytes, (long)x->a.size);
    VKCHECK(vkBindImageMemory(g_dev, x->img, x->a.b->mem, x->a.off));
    x->w = x->h = x->d = 1; x->levels = 1; x->layers = layers; x->samples = samples; x->usage = ci.usage; x->aspect = fmt_aspect(f); x->is3d = type == VK_IMAGE_TYPE_3D; x->cube_ok = cube;
    x->fswz[0] = NVN_SWZ_R; x->fswz[1] = NVN_SWZ_G; x->fswz[2] = NVN_SWZ_B; x->fswz[3] = NVN_SWZ_A;
    VkCommandBuffer cb = once_begin(); {
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = x->img, .subresourceRange = { x->aspect, 0, 1, 0, layers } };   // (WRITE too: the clear below; SYNC-HAZARD-WRITE-AFTER-WRITE)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        VkImageSubresourceRange r = { x->aspect, 0, 1, 0, layers };
        if (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue c = {{0}}; vkCmdClearColorImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
        else { VkClearDepthStencilValue c = {0, 0}; vkCmdClearDepthStencilImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); } }
    once_end(cb);
    return x;
}
static VkBuffer buf_make_c(VkDeviceSize n, vmalloc_t *a, void **map, int cls) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = BUF_ALL }; VkBuffer b; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &b));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, b, &rq); *a = vk_alloc_c(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, cls);
    VKCHECK(vkBindBufferMemory(g_dev, b, a->b->mem, a->off)); if (map) *map = (uint8_t *)a->b->map + a->off; return b;
}
static VkBuffer buf_make(VkDeviceSize n, vmalloc_t *a, void **map) { return buf_make_c(n, a, map, VA_LONG); }
static void dummies_init(void) {
    static const VkFormat F[3] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT };
    for (int c = 0; c < 3; c++) {
        g_dummy[0][c] = dummy_make(F[c], VK_IMAGE_TYPE_2D, 1, 0, VK_SAMPLE_COUNT_1_BIT, 1);
        g_dummy[1][c] = g_dummy[0][c];
        g_dummy[2][c] = dummy_make(F[c], VK_IMAGE_TYPE_2D, 6, 1, VK_SAMPLE_COUNT_1_BIT, 0);
        g_dummy[3][c] = g_dummy[2][c];
        g_dummy[4][c] = dummy_make(F[c], VK_IMAGE_TYPE_3D, 1, 0, VK_SAMPLE_COUNT_1_BIT, 0);
        g_dummy[5][c] = dummy_make(F[c], VK_IMAGE_TYPE_2D, 1, 0, VK_SAMPLE_COUNT_4_BIT, 0);
    }
    g_dummy_depth[0] = dummy_make(VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, 1, 0, VK_SAMPLE_COUNT_1_BIT, 0); g_dummy_depth[1] = g_dummy_depth[0];
    g_dummy_depth[2] = dummy_make(VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, 6, 1, VK_SAMPLE_COUNT_1_BIT, 0); g_dummy_depth[3] = g_dummy_depth[2];
    void *m; g_zero_buf = buf_make(65536, &g_zero_a, &m); memset(m, 0, 65536);
    // Ryujinx's support buffer (SupportBuffer.cs): alpha test 0, BGRA swaps 0, render scale 1.0 (float4 [73] from byte 192, .x read)
    g_support_buf = buf_make(65536, &g_support_a, &m); memset(m, 0, 65536); for (int i = 0; i < 73; i++) *(float *)((uint8_t *)m + 192 + 16 * i) = 1.0f;
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .maxLod = 0.25f }; VKCHECK(vkCreateSampler(g_dev, &sci, NULL, &g_dummy_samp));
    for (int c = 0; c < 3; c++) { VkBufferViewCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = g_zero_buf, .format = F[c] == VK_FORMAT_R8G8B8A8_UNORM ? VK_FORMAT_R32G32B32A32_SFLOAT : F[c], .range = 65536 };
        VKCHECK(vkCreateBufferView(g_dev, &bci, NULL, &g_dummy_tbuf[c])); }
}
static const uint8_t SW_ID[4] = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
// shader image declaration → view kind (dummy table row) and view type
static int vkind(const spv_res *r) {
    if (r->ms) return 5;
    switch (r->dim) { case SPV_DIM_3D: return 4; case SPV_DIM_CUBE: return r->arrayed ? 3 : 2; default: return r->arrayed ? 1 : 0; }
}
static const VkImageViewType VK_OF_KIND[6] = { VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_2D };
static VkImageView dummy_view(const spv_res *r, int storage) {
    int k = vkind(r); vtex *x = r->depth && !storage && k <= 3 ? g_dummy_depth[k] : g_dummy[k][r->sclass < 3 ? r->sclass : 0];
    if (storage && k != 0 && k != 1) x = g_dummy[0][r->sclass < 3 ? r->sclass : 0], k = 0;
    uint32_t nl = k == 2 ? 6 : k == 3 ? 6 : 1;
    return view_get(x, VK_OF_KIND[k], x->fmt, 0, 1, 0, nl, SW_ID, x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect);
}
// Texture ID → view of the kind the shader declares (nvn_mtl.m tex_by_id + tex_fit): the registered view's format / levels / layers, its channel mapping (else the builder's) on top of
// the format's own; a depth-stencil texture reads depth unless the view asks for stencil. VK_NULL_HANDLE: no fitting view (the slot reads 0)
// A format the GPU cannot filter (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT missing): Arm's Mali has no linear filtering of 32-bit floats (no OES_texture_float_linear
//   either), Adreno has. A linear sampler on such a texture is undefined and on Mali reads 0, which would leave e.g. an exposure value at 0 (the white scenes, #215)
static int fmt_no_linear(VkFormat f) {
    static uint8_t known[256];   // 0 unknown, 1 filterable, 2 not
    if ((unsigned)f < 256 && known[f]) return known[f] == 2;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, f, &fp);
    int no = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    if ((unsigned)f < 256) { known[f] = no ? 2 : 1; if (no) VLOG("textures: vk format %d cannot be filtered on this GPU: sampled without filtering", (int)f); }
    return no;
}
static int tex_is_int(int tid) {   // the texture ID's view format is an integer one, or one the GPU cannot filter: sampled through the unfiltered sampler
    treg_t rg; if (!treg_get(tid, &rg)) return 0;
    vtex *x = tex_res(&rg.t); if (!x) return 0;
    VkFormat f = x->fmt; if (rg.has_v && rg.v.set_format) { uint32_t vf = 0; uint8_t s2[4]; if (nvn_fmt_to_vk(rg.v.format, &vf, s2) && vf) f = (VkFormat)vf; }
    return fmt_int_class(f) != 0 || fmt_no_linear(f);
}
// R11G11B10F's stand-in (DM_VK_B10_AS_RGBA16) bound as a storage image of a compute shader that declares it without a format: its stores are made what the packed
//   format holds after the dispatch (dispatch, b10_quantize_region), the level and layers bound here
typedef struct { vtex *x; uint32_t lv, z0, nz; } b10q_ent; static b10q_ent g_b10q[8]; static int g_nb10q;
// the stand-in (or an image viewed as R11G11B10F) bound as a storage image: 0 when it must not be (said in red once)
static int b10_storage_bind(vtex *x, const spv_res *r, VkFormat f, int st, int img_writes, uint32_t l0, uint32_t s0, uint32_t ns, const char *who) {
    // an image the shader declares R11G11B10F was re-declared RGBA16F at load (spv_image_format_swap): a view in another format does not fit that declaration
    if (r->ifmt == SPV_IMGFMT_R11G11B10F && f != VK_FORMAT_R16G16B16A16_SFLOAT) {
        RED_ONCE(158, "b10 swap: %s: an image declared R11G11B10F (re-declared RGBA16F for the stand-in) given an image viewed in vk format %d: not bound, it reads 0 and takes no writes (turn the test off)", who, (int)f);
        return 0; }
    if (!b10_swapped_tex(x)) return 1;
    if (r->ifmt == SPV_IMGFMT_R11G11B10F) b10_note("the stand-in bound as a storage image declared R11G11B10F: its stores made what the packed format holds (the shader, at load), alpha 1", x, st);
    else if (r->ifmt == 0 && img_writes && st == 5) {
        uint32_t nz = x->is3d ? (x->d >> l0 ? x->d >> l0 : 1) : ns, z0 = x->is3d ? 0 : s0; int dup = 0;
        for (int i = 0; i < g_nb10q; i++) if (g_b10q[i].x == x && g_b10q[i].lv == l0 && g_b10q[i].z0 == z0 && g_b10q[i].nz == nz) dup = 1;
        if (!dup && g_nb10q < 8) g_b10q[g_nb10q++] = (b10q_ent){ x, l0, z0, nz };
        b10_note("the stand-in bound as a storage image declared without a format: what the dispatch stores made what R11G11B10F holds after it", x, st); }
    else if (r->ifmt == 0 && img_writes)
        RED_ONCE(159, "b10 swap: %s: a draw's shader writes the stand-in through an image declared without a format: what it stores is not made what R11G11B10F holds (negative values, alpha kept)", who);
    else b10_note("the stand-in bound as a storage image declared in another format, or not written: as it is", x, r->ifmt);
    return 1;
}
// (DM_VK_DRAW_MEMO=0: the draw's lookups done in full every time: tex_view_for's remembered answers, prog_get's live-shader check under its lock)
static int draw_memo_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DRAW_MEMO"); v = !(e && *e == '0'); VLOG("draw lookups: %s", v ? "remembered between draws (DM_VK_DRAW_MEMO=0: not)" : "not remembered (DM_VK_DRAW_MEMO=0)"); } return v; }
static VkImageView tex_view_for_(int tid, const spv_res *r, int storage, const char *skey, int st, int img_writes, vtex **xo);
// tex_view_for's answers remembered (DM_VK_DRAW_MEMO): draws bind the same textures as the draws before them, and every binding went through treg_get (a copy of
//   the registration), tex_res (g_smu, two hash lookups) and view_get (its view list, a scan) each time (fable-cpu.md fix 5). An answer, by texture ID and shader
//   binding, holds while nothing it was made from can have changed: the registrations and the second images for another BC kind (g_tvm_gen), the textures by
//   serial (g_ser_gen: every image put in or taken out, so the image it names is still alive), the game's writes to tracked texture memory (nvn_dirty_now: a
//   packaged texture the CPU wrote is uploaded again by tex_res) and the frame. A hit still marks the image read (sk_read, depth_read: the stores it needs).
//   Not used with a frame capture or a golden recording (they note each binding) or with the R11G11B10F stand-in (b10_on: it decides per binding)
typedef struct { int tid; const spv_res *r; vtex *x; VkImageView v; uint64_t gen, ser, dirty, frame; } tvm_ent;
enum { TVM_N = 1024 };
static tvm_ent g_tvm[TVM_N];   // (direct-mapped; the replay's thread only, under g_rmu like the registrations it caches)
static VkImageView tex_view_for(int tid, const spv_res *r, int storage, const char *skey, int st, int img_writes) {
    if (!draw_memo_on() || atomic_load(&g_cap) == 2 || __builtin_expect(g_golden == 2, 0) || b10_on()) { vtex *x; return tex_view_for_(tid, r, storage, skey, st, img_writes, &x); }
    tvm_ent *e = &g_tvm[((uint32_t)tid * 2654435761u ^ (uint32_t)((uintptr_t)r >> 4)) & (TVM_N - 1)];
    uint64_t ser = atomic_load(&g_ser_gen), dirty = nvn_dirty_now();
    if (e->v && e->tid == tid && e->r == r && e->gen == g_tvm_gen && e->ser == ser && e->dirty == dirty && e->frame == g_frame_no) { sk_read(e->x); depth_read(e->x); return e->v; }
    vtex *x = NULL; VkImageView v = tex_view_for_(tid, r, storage, skey, st, img_writes, &x);
    if (v && x) *e = (tvm_ent){ tid, r, x, v, g_tvm_gen, ser, dirty, g_frame_no }; else e->v = VK_NULL_HANDLE;
    return v;
}
// xo: the image tex_res gave (the one sk_read / depth_read mark), for the memo
static VkImageView tex_view_for_(int tid, const spv_res *r, int storage, const char *skey, int st, int img_writes, vtex **xo) {
    *xo = NULL;
    if (g_tpool_n && tid >= g_tpool_n) return VK_NULL_HANDLE;
    treg_t rg; if (!treg_get(tid, &rg)) { if (tid > 0) RED_ONCE(11, "texture ID %d not registered", tid); return VK_NULL_HANDLE; }
    vtex *x = tex_res(&rg.t); if (!x || !x->img) return VK_NULL_HANDLE;
    *xo = x;
    sk_read(x);   // sampled or bound as an image: its pending stores were needed
    depth_read(x);   // (a shader reads it: DM_VK_DEPTH_DISCARD keeps storing it)
    // A view in another BC kind than its texture (BC5 unsigned read as signed, BC4 likewise, BC6H unsigned as signed): on the Switch (and on a GPU with BC) the same
    //   blocks are read the other way. Where BC is decoded / transcoded on upload the image holds values converted for the texture's own kind, which the other
    //   kind's view would read as garbage (lighting from such normal maps white and black on Mali, #47 / #53 / #56; BC6H's two kinds are not even compatible
    //   formats here). Such a view gets a second image, the game's blocks converted for the view's kind (packaged textures; made once, dropped with the first)
    if (rg.has_v && rg.v.set_format && rg.v.format != rg.t.d.format && x->pkg && rg.t.mem) {
        int ik = nvn_fmt_vk_decode(rg.t.d.format), vk_ = nvn_fmt_vk_decode(rg.v.format);
        if (ik && vk_ && ik != vk_) {
            pthread_mutex_lock(&g_smu);
            if (!x->alt || x->alt_fmt != (int)rg.v.format) {
                tex_t tc = rg.t; tc.d.format = rg.v.format; vtex *a = pkg_upload(&tc);
                if (a) { if (x->alt) vtex_unref(x->alt); x->alt = a; x->alt_fmt = (int)rg.v.format; g_tvm_gen++;
                    static int said; if (said++ < 8) VLOG("texture ID %d (BC format %#x, decoded here) viewed as BC format %#x: converted a second time for that view", tid, rg.t.d.format, rg.v.format); }
            }
            if (x->alt && x->alt_fmt == (int)rg.v.format) x = x->alt;
            pthread_mutex_unlock(&g_smu);
        }
    }
    VkFormat f = x->fmt; uint8_t fsw[4]; memcpy(fsw, x->fswz, 4);
    if (rg.has_v && rg.v.set_format) { uint32_t vf = 0; uint8_t s2[4]; if (nvn_fmt_to_vk(rg.v.format, &vf, s2) && vf) { f = (VkFormat)vf; memcpy(fsw, s2, 4); } }
    if (atomic_load(&g_cap) == 2) { if (storage) cap_printf("  writes image %d: nvn %#x, image vk %d %ux%u, viewed as vk %d\n", tid, rg.t.d.format, x->fmt, x->w, x->h, f);
        else cap_tex(tid, &rg.t, x, rg.has_v && rg.v.set_format && rg.v.format != rg.t.d.format ? (int)rg.v.format : 0, f); }
    uint32_t l0 = rg.has_v && rg.v.set_levels ? rg.v.base_level : 0, nl = rg.has_v && rg.v.set_levels && rg.v.nlevels ? rg.v.nlevels : x->levels + x->drop - (l0 < x->levels + x->drop ? l0 : 0);
    if (x->drop) { uint32_t d = l0 < (uint32_t)x->drop ? l0 : (uint32_t)x->drop; l0 -= d; if (nl > (uint32_t)x->drop - d) nl -= (uint32_t)x->drop - d; }   // the game's level n is the image's n - drop
    uint32_t s0 = rg.has_v && rg.v.set_layers ? rg.v.min_layer : 0, ns = rg.has_v && rg.v.set_layers && rg.v.nlayers ? rg.v.nlayers : x->layers - (s0 < x->layers ? s0 : 0);
    if (l0 >= x->levels || s0 >= x->layers) return VK_NULL_HANDLE;
    if (l0 + nl > x->levels) nl = x->levels - l0; if (s0 + ns > x->layers) ns = x->layers - s0;
    int k = vkind(r); VkImageViewType vt = VK_OF_KIND[k];
    if (k == 4 && !x->is3d) { RED_ONCE(12, "shader declares 3D, texture ID %d is not 3D: reads 0", tid); return VK_NULL_HANDLE; }
    if (k != 4 && x->is3d) { RED_ONCE(13, "shader declares a 2D-family image, texture ID %d is 3D: reads 0", tid); return VK_NULL_HANDLE; }
    if (k == 5 ? x->samples <= 1 : x->samples > 1) { RED_ONCE(14, "multisample mismatch for texture ID %d (shader ms %d, texture %u samples)", tid, k == 5, x->samples); return VK_NULL_HANDLE; }
    if (k == 0 || k == 5) ns = 1;
    // a cube view's layer count can be in cubes, not faces: Shield's environment cube is registered as 1 layer of a 6-face texture (read as 0, its water lost the
    // reflections and turned flat blue)
    if ((k == 2 || k == 3) && x->cube_ok && ns < 6 && x->layers - s0 >= 6) { uint32_t nc = ns ? ns : 1;
        static int said[64]; static int nsaid; int seen = 0; for (int q = 0; q < nsaid; q++) if (said[q] == tid) seen = 1;
        if (!seen && nsaid < 64) { said[nsaid++] = tid; VLOG("cube view of texture ID %d: %ux%u, %u layers from %u, %u levels, Vulkan format %d, %s, usage %#x, serial %llu", tid, x->w, x->h, x->layers, s0, x->levels, x->fmt,
            x->pkg ? "packaged (from the game's files)" : "made at run time", x->usage, (unsigned long long)rg.t.serial); } ns = nc * 6 <= x->layers - s0 ? nc * 6 : (x->layers - s0) / 6 * 6; }
    if (k == 2 || k == 3) { if (!x->cube_ok || ns < 6) { RED_ONCE(15, "shader declares a cube, texture ID %d has %u layers (cube-compatible %d): reads 0", tid, ns, x->cube_ok); return VK_NULL_HANDLE; } ns = k == 2 ? 6 : ns / 6 * 6; }
    uint8_t sw[4] = { NVN_SWZ_R, NVN_SWZ_G, NVN_SWZ_B, NVN_SWZ_A };
    if (rg.has_v && rg.v.set_swizzle) memcpy(sw, rg.v.swz, 4);
    else if (rg.t.set_swz) for (int c = 0; c < 4; c++) sw[c] = (uint8_t)(rg.t.swz >> (3 * c) & 7);
    uint8_t vk[4];
    for (int c = 0; c < 4; c++) {   // the user's mapping applied on top of the format's: a channel picks a format channel (whose own mapping may be zero / one)
        uint8_t s = sw[c] > NVN_SWZ_A ? (uint8_t)(NVN_SWZ_R + c) : sw[c];
        static int nofmt = -1; if (nofmt < 0) nofmt = getenv("DM_VK_NO_FMT_SWZ") != NULL;
        if (s >= NVN_SWZ_R && !nofmt) s = fsw[s - NVN_SWZ_R];
        vk[c] = (uint8_t)(s + 1);   // NVN_SWZ_ZERO 0 → VK_COMPONENT_SWIZZLE_ZERO 1, ONE → 2, R → 3 …
    }
    // A shader that reads an integer texture as another numeric class than its format (Shield: an R16G16B16A16_UINT texture under a float sampler, binding
    //   fp_t_tcb_8; VUID-vkCmdDrawIndexed-format-07753, the Khronos layer 2026-11-20; undefined in Vulkan). The Switch's texture unit returns each channel as 32 bits
    //   by the texture's own format (an integer zero- or sign-extended as its format says) and the shader takes them as its class:
    //   - 32-bit channels (R32 / R32G32 / R32G32B32A32 UINT or SINT) read as float or as the other signedness: exactly the same bits in a view of the shader's class
    //     (fmt_as_class), where the image allows that view: mutable without a format list (fixed_fmt 0: 3D images, DM_VK_NO_FMT_LIST, and outside Mali a format that
    //     cannot be a render target or a driver without format lists, Vulkan 1.1 without VK_KHR_image_format_list), or the view's format is the image's own. Where
    //     the image's formats are fixed (tex_new: not mutable, fixed_fmt 1, every plain colour image on Mali, Q_ARM_FORMATS; a list of its own format and sRGB twin,
    //     fixed_fmt 2, a render-target format wherever format lists exist: Adreno, Xclipse, Vulkan 1.2 drivers) the texture's own format is read as before
    //     (6e0e1dd): a view of another class would be invalid there, and the empty texture is no answer for 32-bit data;
    //   - 8 / 16-bit (and 10-bit) integers read as float keep their own format, as before (6e0e1dd). "Read as float" says little here: the translator declares every
    //     sampled texture float (Ryujinx's SPIR-V backend; QuerySamplerType carries no component type, so r->sclass is 0 for an integer texture too) and takes the
    //     result's bits with a bitcast where the Maxwell code does integer work on it. Such a shader needs the texel's own bits, which most drivers return through
    //     this binding; only a shader that does float work on the value gets 0 on the Switch (denormals, flushed). Which one a game's shaders are is not known yet
    //     (Shield's fp_t_tcb_8 is the one seen): DM_VK_INT_TEX_EMPTY=1 gives these reads the empty texture of their class (0) instead, on every GPU, to compare;
    //   - 8 / 16-bit integers read as the other signedness keep their own format, as before: a view of the other signedness would extend them the other way
    //     (R16_UINT 0xffff as R16_SINT: -1, where the Switch gives 65535);
    //   - a float / UNORM / SNORM / sRGB texture (or a BC one decoded here) read as uint / sint keeps its own format, as before: not seen in Sword / Shield yet.
    //   docs/ENGINE_UNIFY.md lists where each applies. DM_VK_INT_TEX=keep: the texture's own format always, as before (no views of another class)
    if (!storage && !(x->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) && r->sclass < 3) {
        static int mode = -1;   // (bit 0: DM_VK_INT_TEX=keep, bit 1: DM_VK_INT_TEX_EMPTY; one value, so a thread never sees one switch decided and the other not)
        if (mode < 0) { const char *e = getenv("DM_VK_INT_TEX"), *z = getenv("DM_VK_INT_TEX_EMPTY"); mode = (e && !strcmp(e, "keep") ? 1 : 0) | (z && atoi(z) > 0 ? 2 : 0); }
        int keep = mode & 1, empty = mode & 2, have = fmt_int_class(f);
        if (have && have != r->sclass && !keep) {
            int wide = fmt_as_class(f, 0) != 0;   // (32-bit channels: fmt_as_class has a float format for them only)
            VkFormat nf = wide ? fmt_as_class(f, r->sclass) : (VkFormat)0;
            int ok = nf && (!x->fixed_fmt || nf == x->fmt || nf == x->twin);   // (a view in the image's own format or twin is always allowed)
            int zero = !wide && !r->sclass && empty;
            static int said; if (said++ < 8) VLOG("texture ID %d (vk format %d) read as %s by a shader: %s", tid, f, r->sclass == 0 ? "float" : r->sclass == 1 ? "uint" : "sint",
                                                  ok ? "the same bits in a view of that class" : wide ? "its own format, as before (the image's formats are fixed: no view of that class here)"
                                                  : zero ? "reads the empty texture of that class (0, as the Switch's flushed denormals: DM_VK_INT_TEX_EMPTY)"
                                                  : r->sclass ? "its own format, as before (a narrower integer: the Switch extends it by the texture's own format)"
                                                  : "its own format, as before (a narrower integer: its own bits on most drivers, as a shader that takes them as an integer gets them on the Switch)");
            if (zero) return VK_NULL_HANDLE;
            if (ok) f = nf;
        }
    }
    VkImageAspectFlags asp = x->aspect;
    if (asp & VK_IMAGE_ASPECT_DEPTH_BIT) asp = rg.has_v && rg.v.set_dsmode && rg.v.dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && fmt_has_stencil(x->fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (storage) { if (!(x->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { RED_ONCE(16, "texture ID %d bound as an image lacks storage usage", tid); return VK_NULL_HANDLE; } memcpy(vk, SW_ID, 4); }
    if (b10_on()) {   // DM_VK_B10_AS_RGBA16: the format the view is made in, and the stand-in as a storage image (see b10_on)
        char who[96]; snprintf(who, sizeof who, "shader %s binding %s, texture ID %d", skey ? skey : "-", r->name, tid);
        VkFormat bf = b10_view_fmt(x, f, storage ? B10V_STORAGE : B10V_SAMPLED, who); if (!bf) return VK_NULL_HANDLE; f = bf;
        if (storage && !b10_storage_bind(x, r, f, st, img_writes, l0, s0, ns, who)) return VK_NULL_HANDLE;
    }
    return view_get(x, vt, f, l0, nl, s0, ns, vk, asp);
}

// ================= Shaders and programs =================
enum { W_NONE, W_SUPPORT, W_C1, W_DRV0, W_DRV2, W_CBUF, W_SSBO, W_UBOP, W_SBUFU, W_TEX, W_TEXU, W_IMG, W_TBUF };
enum { S_NONE, S_TEX, S_SEP, S_UBO, S_ID };   // where a combined texture's sampler comes from: the texture handle's high bits, a separate sampler slot, a handle in a uniform buffer, a constant ID
typedef struct { uint8_t what, skind, rw; uint16_t reg, off, sreg, soff; } vres_how;
typedef struct vshader {
    char key[24]; int stage; VkShaderStageFlagBits vkstage; VkShaderModule mod, glmod; uint32_t shift;
    spv_res res[64]; vres_how how[64]; int nres;
    spv_input in[32]; int nin;
    spv_input glin[32]; int nglin;              // the GL-depth variant's (glmod) own inputs: the translator may type them differently
    uint8_t drv_read[3]; uint8_t color_mask, color_known;
    VkBuffer c1buf; VkDeviceSize c1off, c1len;
    uint64_t in_mask, out_mask;                 // user varyings by location (spv_io_masks)
    uint32_t *code; size_t nw;                  // fragment shaders: the (rebound) SPIR-V, for variants
    uint8_t out_cls[8];                         // fragment outputs by location: 0 float, 1 uint, 2 sint, 255 none
    uint8_t img_writes;                         // the shader stores into storage images (OpImageWrite)
    struct { uint64_t drop; uint32_t um, sm; int sw; uint32_t qm; VkShaderModule mod; uint32_t nb; } var[8]; int nvar;   // sw: outputs 0 and 1 swapped (g_ds_swap)   // variants: inputs the previous stage never writes (drop), float outputs to integer targets (um / sm)   // qm: outputs into R11G11B10F's stand-in, made what the packed format stores (spv_b10_outputs)
    int tf[4][2]; uint16_t tf_stride[4]; VkShaderModule tfmod, ptsmod;   // transform feedback: per buffer { output location, components } and stride; the capture variant (tf_draw)
    uint8_t dp;                                 // reads draw parameters from push constants (a device without them: spv_no_drawparams; the draw pushes them, dp_push)
    uint32_t lacks;                             // SPV_NEED_* the shader needs and the device lacks (spv_feature_needs)
    uint32_t *glcode; size_t nglw;              // vertex shaders under intbits_on: the GL-depth variant's SPIR-V (code / nw: the shader's own), for the intbits variants
    struct { uint32_t u, s; int gl; VkShaderModule mod; uint32_t nb; } ibv[8]; int nibv;   // the vertex modules with integer inputs bitcast (intbits_mod)
    uint8_t hasgl;                              // a GL-depth variant was made (glmod; it stays 1 while glmod is released: vs_gl)
    _Atomic int rel; uint8_t relgl, nrel;       // modules released (shmod_release): mod (and glmod: relgl) made again from code / glcode by shmod_ensure; nrel: times
    _Atomic uint64_t used_ns;                   // the last pipeline built with it (shmod_release)
    uint64_t spv_hash;                          // FNV-1a of the main module's SPIR-V as handed to the driver (simdev t_vkdev: shader loading on one thread or many)
} vshader;
// The game's shader modules alive now (main, GL-depth, capture, point, fragment and integer-input variants) and the SPIR-V bytes handed to the driver for them; those
//   destroyed by shmod_release and made again by shmod_ensure ([mem] shmod_report)
static _Atomic int64_t g_shmod_n, g_shmod_bytes; static _Atomic uint64_t g_shmod_rel_n, g_shmod_rel_bytes, g_shmod_remade;
static void shmod_count(VkShaderModule m, int64_t bytes, int sign) { if (m) { atomic_fetch_add(&g_shmod_n, sign); atomic_fetch_add(&g_shmod_bytes, sign * bytes); } }
// Integer vertex formats under float inputs (DM_VK_VFMT_INTBITS, default on; 0 turns it off): see spv_vin_raw. The game feeds 8 / 16-bit UINT attributes (packed
//   normals, colours, indices: Sword / Shield's field at locations 7, 8, 11) into inputs the translated shader declares float and converts itself. Adreno hands the
//   bits over unchanged; Arm r44 converts the integer first, and the lit field went white (POCO X6 Pro, G615: 79 % of the lit target at 1024+ -> 0 %)
static int intbits_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_VFMT_INTBITS"); v = !(e && *e == '0');
    VLOG("vertex attributes: 8 / 16-bit integer formats under float inputs %s", v ? "read as integers and bitcast in the vertex shader" : "left as they are (DM_VK_VFMT_INTBITS=0)"); } return v; }
static int intbits_kind(VkFormat f) {   // 1 an 8 / 16-bit UINT format, 2 SINT, 0 other
    switch (f) { case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
                 case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT: return 1;
                 case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
                 case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT: return 2;
                 default: return 0; } }
static pthread_mutex_t g_ibv_mu = PTHREAD_MUTEX_INITIALIZER;
static VkShaderModule intbits_mod(vshader *vs, int gl, uint32_t u, uint32_t sm) {   // the vertex module (gl: the GL-depth variant) with those inputs bitcast; NULL: the shader's own
    const uint32_t *code = gl ? vs->glcode : vs->code; size_t cw = gl ? vs->nglw : vs->nw;
    if (!(u | sm) || !code) return VK_NULL_HANDLE;
    VkShaderModule r = VK_NULL_HANDLE; pthread_mutex_lock(&g_ibv_mu);
    for (int i = 0; i < vs->nibv; i++) if (vs->ibv[i].u == u && vs->ibv[i].s == sm && vs->ibv[i].gl == gl) { r = vs->ibv[i].mod; goto out; }
    if (vs->nibv < 8) { size_t nw = 0; uint32_t *c = spv_vin_raw(code, cw, 0, u, sm, &nw);
        if (c) { VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = c };
                 if (vkCreateShaderModule(g_dev, &mci, NULL, &r)) r = VK_NULL_HANDLE; free(c); shmod_count(r, (int64_t)nw * 4, 1); }
        static _Atomic int said; if ((!vs->nrel && atomic_fetch_add(&said, 1) < 8) || !r) VLOG("%svertex shader %s%s: integer inputs bitcast (uint %#x, int %#x)%s", r ? "" : "[RED] ", vs->key, gl ? " (GL depth)" : "", u, sm, r ? "" : ": NOT made, the shader's own module");
        vs->ibv[vs->nibv].u = u; vs->ibv[vs->nibv].s = sm; vs->ibv[vs->nibv].gl = gl; vs->ibv[vs->nibv].mod = r; vs->ibv[vs->nibv].nb = (uint32_t)(nw * 4); vs->nibv++; }
  out:
    pthread_mutex_unlock(&g_ibv_mu); return r;
}
// A device without draw parameters: the module (any variant of a shader) without them (spv_no_drawparams); c is freed when replaced
static uint32_t *dp_fix(vshader *s, uint32_t *c, size_t *nw, const char *hex) {
    if (g_drawp || !c) return c;
    int mode = 0; size_t n2; uint32_t *d = spv_no_drawparams(c, *nw, &n2, &mode);
    if (!d) { if (mode == 0) { /* no DrawParameters: nothing to do */ } return c; }
    free(c); *nw = n2;
    if (mode == 1) s->dp = 1;
    if (mode == 2) RED_ONCE(129, "shader %s: a draw parameter in a module with push constants already (a capture variant): read as 0", hex);
    return d;
}
static char g_spv_dir[1024];
static nvn_hmap g_shmap = {.klen = 8};      // shader key (8 bytes) → stage + 1 (as a pointer)
static nvn_hmap g_shcache = {.klen = 8};    // shader key → vshader
static _Atomic uint64_t g_spv_mod_bytes, g_spv_kept_bytes;   // ([mem]) SPIR-V handed to the driver as main modules, and kept by us (fragment shaders, for variants)
static int stage_index(const char *s) { return !strcmp(s, "vertex") ? 0 : !strcmp(s, "pixel") ? 1 : !strcmp(s, "compute") ? 5 : !strcmp(s, "hull") ? 3 : !strcmp(s, "domain") ? 4 : !strcmp(s, "geometry") ? 2 : -1; }
static int hexkey(const char *s, uint8_t k[8]) { for (int i = 0; i < 8; i++) { unsigned v; if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1; k[i] = (uint8_t)v; } return 0; }
static void shmap_read(void);
static void shmap_load(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, shmap_read); }   // the game thread (mtl_shader_seen) and the replay both ask
static int shmap_parse(FILE *f, nvn_hmap *m) {   // shader_map.tsv rows (package, index, stage, key, source) → key → stage + 1
    char line[1024]; int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || !strncmp(line, "package", 7)) continue;
        char *c[5] = {0}; int nc = 0; for (char *p = line; nc < 5 && p; nc++) { c[nc] = p; p = strchr(p, '\t'); if (p) *p++ = 0; }
        if (nc < 4) continue; char *nl = strpbrk(c[3], "\r\n"); if (nl) *nl = 0;
        uint8_t k[8]; int st = stage_index(c[2]); if (st < 0 || strlen(c[3]) < 16 || hexkey(c[3], k)) continue;
        nvn_hm_put(m, k, (void *)(intptr_t)(st + 1)); n++;
    }
    return n;
}
static void shmap_read(void) {
    const char *e = getenv("DM_NVN_SHMAP"); FILE *f = e ? fopen(e, "r") : NULL;
    if (!f) VDIE("cannot read the shader map (DM_NVN_SHMAP=%s)", e ? e : "(unset)");
    int n = shmap_parse(f, &g_shmap);
    fclose(f);
    const char *d = getenv("DM_SPV_DIR"); snprintf(g_spv_dir, sizeof g_spv_dir, "%s/b-%s", d ? d : ".", DMSH_CHAIN);
    VLOG("shader map: %d shaders, SPIR-V from %s", n, g_spv_dir);
}
static void *file_read(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); long l = ftell(f); rewind(f);
    void *b = malloc(l > 0 ? (size_t)l : 1); if (l > 0 && fread(b, 1, (size_t)l, f) != (size_t)l) { free(b); fclose(f); return NULL; } fclose(f); *n = (size_t)l; return b;
}
// The shader's own constant segment (c1): written once into a shared host-visible bank, bound at its exact length (reads past it give 0, as on the Switch)
static VkBuffer g_cbank; static uint8_t *g_cbank_map; static VkDeviceSize g_cbank_used, g_cbank_size; static pthread_mutex_t g_cbmu = PTHREAD_MUTEX_INITIALIZER;
static void c1_store(vshader *s, const uint8_t *c1, uint32_t n) {
    pthread_mutex_lock(&g_cbmu);
    VkDeviceSize need = (n + 255) & ~255u; if (!need) need = 256;
    if (!g_cbank || g_cbank_used + need > g_cbank_size) { vmalloc_t a; void *m; g_cbank_size = 4u << 20; g_cbank = buf_make(g_cbank_size, &a, &m); g_cbank_map = m; g_cbank_used = 0; memset(m, 0, g_cbank_size); }
    memcpy(g_cbank_map + g_cbank_used, c1, n); s->c1buf = g_cbank; s->c1off = g_cbank_used; s->c1len = n ? (n + 15) & ~15u : 16; g_cbank_used += need;
    pthread_mutex_unlock(&g_cbmu);
}
enum { TF_BIND = 320 };   // the capture buffers' bindings (past every stage's shifted range, stage_shift)
static uint32_t stage_shift(int st) { return st == 1 ? 64 : st == 2 ? 128 : st == 3 ? 192 : st == 4 ? 256 : 0; }
// The shader set packed into one mapped file (DM_SPV_PAK, dm/android/tools/spv_pak.py): "NXSP", version, count, chain[16], sorted index, data.
// Versions 1/2 index SPIR-V and reflection; versions 3/4 additionally index a GL-depth fallback SPIR-V. Versions 2/4 compress each blob.
static const uint8_t *g_pak; static size_t g_pak_n; static uint32_t g_pak_count, g_pak_ver;
static void pak_open_once(void);
// (once, under pthread_once: the prewarm loads shaders on several threads (shader_get), and a second thread that found "tried" set while the first was still mapping
//   the pack would have read every shader as missing)
static int pak_open(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, pak_open_once); return g_pak != NULL; }
static void pak_open_once(void) {
    const char *path = getenv("DM_SPV_PAK"); if (!path) return;
    int fd = open(path, O_RDONLY); struct stat st; if (fd < 0 || fstat(fd, &st)) { VLOG("[RED] shader pack %s cannot be opened", path); if (fd >= 0) close(fd); return; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED || st.st_size < 28 || memcmp(m, "NXSP", 4)) { VLOG("[RED] shader pack %s is not one", path); return; }
    char chain[17] = {0}; memcpy(chain, (const uint8_t *)m + 12, 16);
    if (strcmp(chain, DMSH_CHAIN)) { VLOG("[RED] shader pack %s has chain %s, this host expects %s", path, chain, DMSH_CHAIN); munmap(m, (size_t)st.st_size); return; }
    uint32_t ver, count; memcpy(&ver, (const uint8_t *)m + 4, 4); memcpy(&count, (const uint8_t *)m + 8, 4);
    if (ver < 1 || ver > 4) { VLOG("[RED] shader pack %s has version %u", path, ver); munmap(m, (size_t)st.st_size); return; }
    size_t entry_size = ver >= 3 ? 44 : 32;
    if (count > ((size_t)st.st_size - 28) / entry_size) { VLOG("[RED] shader pack %s has a truncated index", path); munmap(m, (size_t)st.st_size); return; }
    g_pak_n = (size_t)st.st_size; g_pak_ver = ver; g_pak_count = count; g_pak = m;
    VLOG("shader pack %s: %u shaders%s%s", path, g_pak_count, g_pak_ver == 2 || g_pak_ver == 4 ? " (compressed)" : "", g_pak_ver >= 3 ? " (GL depth fallback)" : "");
}
static void *pak_blob(const uint8_t *p, size_t n, size_t *out) {   // a malloc'd copy of one blob, inflated in a version 2 pack
    if (g_pak_ver != 2 && g_pak_ver != 4) { void *b = malloc(n ? n : 1); memcpy(b, p, n); *out = n; return b; }
    if (n < 4) return NULL;
    uint32_t raw; memcpy(&raw, p, 4); uLongf dl = raw; void *b = malloc(raw ? raw : 1);
    if (uncompress(b, &dl, p + 4, (uLong)(n - 4)) != Z_OK || dl != raw) { free(b); return NULL; }
    *out = raw; return b;
}
static int pak_get(const uint8_t key[8], uint32_t **spv, size_t *ns, char **vkr, size_t *nr, uint32_t **glspv, size_t *ngl) {
    if (!pak_open()) return 0;
    *glspv = NULL; *ngl = 0;
    size_t entry_size = g_pak_ver >= 3 ? 44 : 32;
    uint32_t lo = 0, hi = g_pak_count;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; const uint8_t *e = g_pak + 28 + entry_size * (size_t)m; int c = memcmp(e, key, 8);
        if (!c) { uint64_t so, ro; uint32_t sl, rl; memcpy(&so, e + 8, 8); memcpy(&sl, e + 16, 4); memcpy(&ro, e + 20, 8); memcpy(&rl, e + 28, 4);
                  if (so > g_pak_n || sl > g_pak_n - so || ro > g_pak_n || rl > g_pak_n - ro) return 0;
                  *spv = pak_blob(g_pak + so, sl, ns); *vkr = pak_blob(g_pak + ro, rl, nr);
                  if (!*spv || !*vkr) { free(*spv); free(*vkr); *spv = NULL; *vkr = NULL; return 0; }
                  if (g_pak_ver >= 3) { uint64_t go; uint32_t gl; memcpy(&go, e + 32, 8); memcpy(&gl, e + 40, 4);
                      if (gl) { if (go > g_pak_n || gl > g_pak_n - go || !(*glspv = pak_blob(g_pak + go, gl, ngl))) { free(*spv); free(*vkr); *spv = NULL; *vkr = NULL; return 0; } } }
                  return 1; }
        if (c < 0) lo = m + 1; else hi = m; }
    return 0;
}
static int pak_has(const uint8_t key[8]) {
    if (!pak_open()) return 0;
    size_t entry_size = g_pak_ver >= 3 ? 44 : 32;
    uint32_t lo = 0, hi = g_pak_count;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; int c = memcmp(g_pak + 28 + entry_size * (size_t)m, key, 8); if (!c) return 1; if (c < 0) lo = m + 1; else hi = m; }
    return 0;
}
// Shaders the game builds in memory match no game file, so the player's own setup cannot translate them from the dump (Shield: 45, profile shader_nvn_capture):
// the first time one is handed over and the pack lacks it, its 0x880-byte control section and its code area are saved for the setup's next run to translate.
// Shaders not in the map at all are saved too (the profile's slots only list those its recording reached; the setup takes their stage from the shader program header)
void mtl_shader_seen(const uint8_t key[8], const void *control, const void *code, uint32_t ncode) {
    static const char *dir; static int init, unmapped; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static nvn_hmap seen = {.klen = 8};
    if (!init) { dir = getenv("DM_NVN_CAPTURE"); if (dir) mkdir(dir, 0755); init = 1; }
    if (!dir || !control || !code || !ncode || ncode >= (1u << 24)) return;
    pthread_mutex_lock(&mu);
    if (!nvn_hm_get(&seen, key)) {
        nvn_hm_put(&seen, key, (void *)1);
        shmap_load();
        int mapped = nvn_hm_get(&g_shmap, key) != NULL;
        if ((!mapped && unmapped < 512) || (mapped && !pak_has(key))) {   // 512 a run: a game that builds shaders without end does not fill the storage
            if (!mapped) unmapped++;
            char hex[17], p[1200]; for (int i = 0; i < 8; i++) sprintf(hex + 2 * i, "%02x", key[i]);
            struct stat st; snprintf(p, sizeof p, "%s/%s.code", dir, hex);
            if (stat(p, &st)) {
                FILE *f; snprintf(p, sizeof p, "%s/%s.ctl", dir, hex);
                if ((f = fopen(p, "wb"))) { fwrite(control, 1, 0x880, f); fclose(f); }
                snprintf(p, sizeof p, "%s/%s.code.tmp", dir, hex);
                if ((f = fopen(p, "wb"))) { fwrite(code, 1, ncode, f); fclose(f); char q[1200]; snprintf(q, sizeof q, "%s/%s.code", dir, hex); rename(p, q); }
                VLOG("shader %s captured for translation%s (%s)", hex, mapped ? "" : ", not in the shader map", dir);
            }
        }
    }
    pthread_mutex_unlock(&mu);
}
// Shaders translated while the game runs: the game saves a shader the pack lacks (mtl_shader_seen above, a mod's new effect most often), the app translates it in
// the background (GameActivity, setup_game.py --captures --live) into DM_LIVE_SPV (<key>.spv / .vkr, rows in map.tsv) and rewrites its gen file; every 2 s the replay
// looks at gen, reads the new rows, and the programs that lacked a shader are built again, so the effect draws a few seconds after it first appears, not next start
static nvn_hmap g_livemap = {.klen = 8};   // key → stage + 1, under g_shmu
static volatile int g_live_gen;
static char g_live_dir[1024];
static void live_check(void) {   // under g_shmu
    static double next; static char seen[64];
    if (!*g_live_dir) { const char *d = getenv("DM_LIVE_SPV"); if (!d || !*d) return; snprintf(g_live_dir, sizeof g_live_dir, "%s", d); }
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); double t = ts.tv_sec + ts.tv_nsec / 1e9; if (t < next) return; next = t + 2;
    char p[1200], g[64] = ""; snprintf(p, sizeof p, "%s/gen", g_live_dir); FILE *f = fopen(p, "r"); if (!f) return;
    if (!fgets(g, sizeof g, f)) g[0] = 0; fclose(f);
    if (!g[0] || !strcmp(g, seen)) return;
    snprintf(seen, sizeof seen, "%s", g);
    snprintf(p, sizeof p, "%s/map.tsv", g_live_dir); if (!(f = fopen(p, "r"))) return;
    int n = shmap_parse(f, &g_livemap); fclose(f); g_live_gen++;
    VLOG("live shaders: %d translated while the game runs (%s)", n, g_live_dir);
}
// Gather nudge (Adreno), decided once for every thread that loads shaders (shader_load runs on several prewarm threads at once: shader_get)
static int g_gbias; static pthread_once_t g_gb_once = PTHREAD_ONCE_INIT;
static void gbias_decide(void) {
    const char *e = getenv("DM_VK_GATHER_BIAS"); g_gbias = quirk_note(Q_GATHER_BIAS, e ? atoi(e) != 0 : quirk_default(Q_GATHER_BIAS, &g_gpuid), e != NULL);
    if (g_gbias) VLOG("texture gathers nudged by 1/512 of a texel (Adreno rounding)");
}
// live: g_live_dir as shader_get read it under g_shmu (live_check sets it under that lock; the load itself may run outside it)
// Prewarm's lookups (the shipped pipeline lists name shaders the game builds in memory, not in this player's pack until the game hands them over): a missing one
//   is counted in prewarm's summary line, not a [RED] each (a Shield report: 23 "SPIR-V missing" warnings, all prewarm's, none drawn without)
static __thread int g_sh_quiet;
static vshader *shader_load(const uint8_t key[8], int st, const char *live) {
    char hex[17]; for (int i = 0; i < 8; i++) sprintf(hex + 2 * i, "%02x", key[i]);
    char p[1200]; size_t ns = 0, nr = 0, ngl = 0; uint32_t *code = NULL, *glcode = NULL; char *meta = NULL;
    if (pak_get(key, &code, &ns, &meta, &nr, &glcode, &ngl)) {}
    else {
        snprintf(p, sizeof p, "%s/%s.spv", g_spv_dir, hex); code = file_read(p, &ns);
        snprintf(p, sizeof p, "%s/%s.vkr", g_spv_dir, hex); meta = file_read(p, &nr);
        if (st == 0) { snprintf(p, sizeof p, "%s/%s.gl.spv", g_spv_dir, hex); glcode = file_read(p, &ngl); }
        if (!code && *live) {   // (translated while the game runs: live_check)
            snprintf(p, sizeof p, "%s/%s.spv", live, hex); code = file_read(p, &ns); free(meta); meta = NULL;
            snprintf(p, sizeof p, "%s/%s.vkr", live, hex); meta = file_read(p, &nr);
            if (st == 0) { free(glcode); glcode = NULL; snprintf(p, sizeof p, "%s/%s.gl.spv", live, hex); glcode = file_read(p, &ngl); }
        }
    }
    if (!code || !meta) { if (!g_sh_quiet) VLOG("[RED] shader %s: %s missing", hex, !code ? "SPIR-V" : "reflection (.vkr)"); free(code); free(meta); free(glcode); return NULL; }
    meta = realloc(meta, nr + 1); meta[nr] = 0;
    vshader *s = calloc(1, sizeof *s); strcpy(s->key, hex); s->stage = st; s->shift = stage_shift(st);
    int spst; uint32_t ls[3];
    if (spv_reflect(code, ns / 4, s->res, 64, &s->nres, s->in, 32, &s->nin, &spst, ls)) { VLOG("[RED] shader %s: SPIR-V not readable", hex); free(code); free(meta); free(glcode); free(s); return NULL; }
    static const VkShaderStageFlagBits VS[6] = { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT };
    s->vkstage = spst >= 0 && spst < 6 ? VS[spst] : VK_SHADER_STAGE_VERTEX_BIT;
    // reflection: bindings by name (the combined image takes its texture slot from "<name>" and its sampler from "<name>Smplr"), c1, driver segment reads
    struct { char name[48], kind[16]; int reg, off, bank, rw, n; } b[96]; int nb = 0; uint8_t *c1 = NULL; uint32_t c1n = 0; int unfill = 0;
    char *sv = NULL;
    for (char *ln = strtok_r(meta, "\n", &sv); ln; ln = strtok_r(NULL, "\n", &sv)) {   // (strtok_r: several threads load shaders at once)
        if (!strncmp(ln, "b ", 2) && nb < 96) { if (sscanf(ln + 2, "%47s %15s %d %d %d %d %d", b[nb].name, b[nb].kind, &b[nb].reg, &b[nb].off, &b[nb].bank, &b[nb].rw, &b[nb].n) == 7) nb++; }
        else if (!strncmp(ln, "c1 ", 3)) { const char *h = ln + 3; size_t l = strlen(h) / 2; c1 = malloc(l ? l : 1); for (size_t i = 0; i < l; i++) { unsigned v; sscanf(h + 2 * i, "%2x", &v); c1[i] = (uint8_t)v; } c1n = (uint32_t)l; }
        else if (!strncmp(ln, "drv ", 4)) { int a, c; if (sscanf(ln + 4, "%d %d", &a, &c) == 2) { s->drv_read[0] = (uint8_t)a; s->drv_read[2] = (uint8_t)c; } }
        else if (!strncmp(ln, "colors ", 7)) { int m, k; if (sscanf(ln + 7, "%d %d", &m, &k) == 2) { s->color_mask = (uint8_t)m; s->color_known = (uint8_t)k; } }
        else if (!strncmp(ln, "unfill ", 7)) unfill = 1;
        else if (!strncmp(ln, "tf ", 3)) { int bb, l, nc, sd; if (sscanf(ln + 3, "%d %d %d %d", &bb, &l, &nc, &sd) == 4 && bb >= 0 && bb < 4 && nc >= 1 && nc <= 4) { s->tf[bb][0] = l; s->tf[bb][1] = nc; s->tf_stride[bb] = (uint16_t)sd; } }
    }
    if (unfill) VLOG("[RED] shader %s reads driver-segment locations the replay cannot fill", hex);
    for (int i = 0; i < s->nres; i++) {
        spv_res *r = &s->res[i]; vres_how *h = &s->how[i];
        if (!strcmp(r->name, "support_buffer")) { h->what = W_SUPPORT; continue; }
        int j = -1; for (int k = 0; k < nb; k++) if (!strcmp(b[k].name, r->name)) { j = k; break; }
        if (j < 0) { RED_ONCE(17, "shader %s: binding %s has no reflection (first case)", hex, r->name); continue; }
        const char *k = b[j].kind; h->reg = (uint16_t)(b[j].reg < 0 ? 0 : b[j].reg); h->off = (uint16_t)b[j].off; h->rw = (uint8_t)b[j].rw;
        if (!strcmp(k, "cbuf")) h->what = W_CBUF;
        else if (!strcmp(k, "c1")) h->what = W_C1;
        else if (!strcmp(k, "drvcb")) { h->what = b[j].bank == 2 ? W_DRV2 : W_DRV0; s->drv_read[b[j].bank == 2 ? 2 : 0] = 1; }
        else if (!strcmp(k, "sbuf")) h->what = W_SSBO;
        else if (!strcmp(k, "ubo_ptr")) h->what = W_UBOP;
        else if (!strcmp(k, "sbuf_ubo")) h->what = W_SBUFU;
        else if (!strcmp(k, "img")) h->what = W_IMG;
        else if (!strcmp(k, "tbuf")) h->what = W_TBUF;
        else if (!strcmp(k, "tex") || !strcmp(k, "tex_ubo")) {
            h->what = !strcmp(k, "tex") ? W_TEX : W_TEXU;
            char sn[64]; snprintf(sn, sizeof sn, "%sSmplr", r->name);
            for (int q = 0; q < nb; q++) if (!strcmp(b[q].name, sn)) {
                const char *sk = b[q].kind; h->sreg = (uint16_t)(b[q].reg < 0 ? 0 : b[q].reg); h->soff = (uint16_t)b[q].off;
                h->skind = !strcmp(sk, "samp_tex") ? S_TEX : !strcmp(sk, "samp") ? S_SEP : !strcmp(sk, "samp_ubo") ? S_UBO : !strcmp(sk, "samp_id") ? S_ID : S_NONE;
            }
            if (!h->skind) h->skind = S_TEX;
        }
        else RED_ONCE(18, "shader %s: binding kind %s not implemented (%s)", hex, k, r->name);
    }
    if (c1n || 1) c1_store(s, c1 ? c1 : (const uint8_t *)"", c1n);
    free(c1); free(meta);
    if (s->shift) spv_rebind(code, ns / 4, s->shift);
    ns = spv_strip_unused(code, ns / 4) * 4;   // capabilities declared but unused (Float64, fragment interlock): strict drivers refuse or mis-handle them
    ns = spv_strip_clipcull(code, ns / 4) * 4;   // unused ClipDistance[8] + CullDistance[1]: over the limit of 8 (VUID-RuntimeSpirv-00710), CullDistance optional
    { size_t nw; uint32_t *c2 = spv_image_caps(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }   // and the format-less storage image ones they need
    // DM_VK_B10_AS_RGBA16: a storage image the shader declares R11G11B10F is the RGBA16F stand-in: what the shader stores into it is made what the packed format holds
    //   (spv_b10_image_writes, alpha 1), then the declaration names the view's format
    if (b10_on()) { size_t nw; int left = 0; uint32_t *c2 = spv_b10_image_writes(code, ns / 4, &nw, &left);
        if (c2) { free(code); code = c2; ns = nw * 4; b10_note("a shader's stores into an image declared R11G11B10F: made what the packed format holds, alpha 1", NULL, 0); }
        if (left) RED_ONCE(160, "b10 swap: shader %s stores fewer than 3 components into an image declared R11G11B10F: left as they are", hex);
        if (spv_image_format_swap(code, ns / 4, SPV_IMGFMT_R11G11B10F, SPV_IMGFMT_RGBA16F)) b10_note("a shader's storage image declared R11G11B10F: declared RGBA16F (the image it is given)", NULL, 0); }
    // Adreno picks another 2x2 block than Maxwell for a gather on a texel centre: its gathers are nudged by 1/512 of a texel (spv_gather_bias). DM_VK_GATHER_BIAS=0 / 1 decides
    pthread_once(&g_gb_once, gbias_decide); int gbias = g_gbias;
    if (gbias) { size_t nw; uint32_t *c2 = spv_gather_bias(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }
    if (g_fctl) { size_t nw; uint32_t *c2 = spv_float_controls(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }
    if (g_nanmm) { size_t nw; uint32_t *c2 = spv_nan_minmax(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_nanmm_n, 1); } }   // (before spv_clamp_zero, which knows both)
    if (g_clampzero) { size_t nw; uint32_t *c2 = spv_clamp_zero(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_clampzero_n, 1); } }   // (Arm erratum 3779215)
    if (g_nanmm == 2) { size_t nw; uint32_t *c2 = spv_nan_select(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }   // (after spv_clamp_zero: its zero is the operand selected)
    if (g_fmz) { size_t nw; uint32_t *c2 = spv_fmz(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_fmz_n, 1); } }
    if (g_probe && spst == 4) { size_t nw; uint32_t *c2 = spv_probe(code, ns / 4, &nw, (unsigned)g_probe); if (c2) { free(code); code = c2; ns = nw * 4; } }   // (DM_VK_PROBE, fragment shaders)
    if (g_exact) { size_t nw; uint32_t *c2 = spv_exact_math(code, ns / 4, &nw, g_exact); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_exact_n, 1); } }
    if (g_initloc) { size_t nw; uint32_t *c2 = spv_init_locals(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_initloc_n, 1); } }
    if (g_qlod) { size_t nw; uint32_t *c2 = spv_query_lod(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; atomic_fetch_add(&g_qlod_n, 1); } }
    if (spst == 0 || spst == 2) { size_t nw; uint32_t *c2 = spv_init_outputs(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }   // outputs start at (0, 0, 0, 1) as on the hardware
    if (st == 0 && glcode) {
        size_t nw;
        if (s->shift) spv_rebind(glcode, ngl / 4, s->shift);
        ngl = spv_strip_unused(glcode, ngl / 4) * 4;
        ngl = spv_strip_clipcull(glcode, ngl / 4) * 4;
        uint32_t *c2 = spv_image_caps(glcode, ngl / 4, &nw); if (c2) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (b10_on()) { int left; if ((c2 = spv_b10_image_writes(glcode, ngl / 4, &nw, &left))) { free(glcode); glcode = c2; ngl = nw * 4; }
                        spv_image_format_swap(glcode, ngl / 4, SPV_IMGFMT_R11G11B10F, SPV_IMGFMT_RGBA16F); }
        if (gbias && (c2 = spv_gather_bias(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_fctl && (c2 = spv_float_controls(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_nanmm && (c2 = spv_nan_minmax(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_clampzero && (c2 = spv_clamp_zero(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_nanmm == 2 && (c2 = spv_nan_select(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_fmz && (c2 = spv_fmz(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_exact && (c2 = spv_exact_math(glcode, ngl / 4, &nw, g_exact))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_initloc && (c2 = spv_init_locals(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_qlod && (c2 = spv_query_lod(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        c2 = spv_init_outputs(glcode, ngl / 4, &nw); if (c2) { free(glcode); glcode = c2; ngl = nw * 4; }
        if (g_yflip && (c2 = spv_flip_y(glcode, ngl / 4, &nw))) { free(glcode); glcode = c2; ngl = nw * 4; }
        { spv_res gres[64]; int gnr = 0, gst; uint32_t gls[3]; if (spv_reflect(glcode, ngl / 4, gres, 64, &gnr, s->glin, 32, &s->nglin, &gst, gls)) s->nglin = 0; }
        if (!g_drawp) { nw = ngl / 4; glcode = dp_fix(s, glcode, &nw, hex); ngl = nw * 4; }
        VkShaderModuleCreateInfo gci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = ngl, .pCode = glcode };
        VkResult gr = vkCreateShaderModule(g_dev, &gci, NULL, &s->glmod);
        if (gr) { VLOG("[RED] shader %s: GL depth fallback module failed %d", hex, gr); s->glmod = VK_NULL_HANDLE; }
        s->hasgl = s->glmod != VK_NULL_HANDLE; shmod_count(s->glmod, (int64_t)ngl, 1);
        if (!gr && intbits_on()) { s->glcode = glcode; s->nglw = ngl / 4; atomic_fetch_add(&g_spv_kept_bytes, (uint64_t)ngl); } else free(glcode);
    } else free(glcode);
    if (st == 1) spv_frag_outputs(code, ns / 4, s->out_cls);
    spv_io_masks(code, ns / 4, &s->in_mask, &s->out_mask);
    if (st == 0 && (s->tf_stride[0] | s->tf_stride[1] | s->tf_stride[2] | s->tf_stride[3])) {   // transform feedback: the capture variant (tf_draw)
        size_t tw; uint32_t *tc = spv_tf_variant(code, ns / 4, (const int (*)[2])s->tf, TF_BIND, &tw);
        tc = dp_fix(s, tc, &tw, hex);
        if (tc) { VkShaderModuleCreateInfo tci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = tw * 4, .pCode = tc }; if (vkCreateShaderModule(g_dev, &tci, NULL, &s->tfmod)) s->tfmod = VK_NULL_HANDLE; free(tc); shmod_count(s->tfmod, (int64_t)tw * 4, 1); }
        if (!s->tfmod) VLOG("[RED] shader %s: transform feedback capture variant failed", hex);
        if (g_tf_pts > 0) { size_t pw; uint32_t *pc_ = spv_tf_points_variant(code, ns / 4, (const int (*)[2])s->tf, TFP_W, &pw);   // the point route (tf_points.h)
            pc_ = dp_fix(s, pc_, &pw, hex);
            if (pc_) { VkShaderModuleCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = pw * 4, .pCode = pc_ }; if (vkCreateShaderModule(g_dev, &pci, NULL, &s->ptsmod)) s->ptsmod = VK_NULL_HANDLE; free(pc_); shmod_count(s->ptsmod, (int64_t)pw * 4, 1); }
            if (!s->ptsmod) VLOG("[RED] shader %s: transform feedback point variant failed", hex); }
    }
    if (g_yflip && st == 0) { size_t nw; uint32_t *c2 = spv_flip_y(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } else RED_ONCE(127, "vertex shader %s: no position to flip (Vulkan 1.0 without maintenance1)", hex); }
    if (!g_drawp) { size_t nw = ns / 4; code = dp_fix(s, code, &nw, hex); ns = nw * 4; }
    // what it needs of the device's optional features: a missing geometry / tessellation stage skips its programs (pso_build); the rest is named once and passed on
    if ((s->lacks = spv_feature_needs(code, ns / 4) & ~g_feat_have)) {
        static _Atomic uint32_t said; uint32_t fresh = s->lacks & ~atomic_fetch_or(&said, s->lacks);
        static const char *nm[] = { "geometry shaders", "tessellation", "64-bit floats", "64-bit integers", "16-bit integers", "tessellation / geometry point size", "gather offsets",
            "multisampled storage images", "clip distances", "cull distances", "cube map arrays", "sample-rate shading", "sparse resources", "min LOD", "extended storage formats",
            "storage image reads without a format", "storage image writes without a format", "multiple viewports", "fragment shader stores" };
        for (int b = 0; b < 19; b++) if (fresh >> b & 1) VLOG("[RED] shader %s needs %s, which this device lacks: %s", hex, nm[b], b <= 1 ? "its programs are skipped" : "passed to the driver as it is");
    }
    if (!g_drawp) { size_t nw = ns / 4; code = dp_fix(s, code, &nw, hex); ns = nw * 4; }
    // what it needs of the device's optional features: a missing geometry / tessellation stage skips its programs (pso_build); the rest is named once and passed on
    if ((s->lacks = spv_feature_needs(code, ns / 4) & ~g_feat_have)) {
        static _Atomic uint32_t said; uint32_t fresh = s->lacks & ~atomic_fetch_or(&said, s->lacks);
        static const char *nm[] = { "geometry shaders", "tessellation", "64-bit floats", "64-bit integers", "16-bit integers", "tessellation / geometry point size", "gather offsets",
            "multisampled storage images", "clip distances", "cull distances", "cube map arrays", "sample-rate shading", "sparse resources", "min LOD", "extended storage formats",
            "storage image reads without a format", "storage image writes without a format", "multiple viewports", "fragment shader stores" };
        for (int b = 0; b < 19; b++) if (fresh >> b & 1) VLOG("[RED] shader %s needs %s, which this device lacks: %s", hex, nm[b], b <= 1 ? "its programs are skipped" : "passed to the driver as it is");
    }
    for (size_t i = 5, nw = ns / 4; i < nw;) { uint32_t n = code[i] >> 16; if (!n || i + n > nw) break; if ((code[i] & 0xffff) == 99 /* OpImageWrite */) { s->img_writes = 1; break; } i += n; }
    { uint64_t h = 1469598103934665603ull; const uint8_t *b = (const uint8_t *)code; for (size_t i = 0; i < ns; i++) { h ^= b[i]; h *= 1099511628211ull; } s->spv_hash = h; }   // (simdev: the same with shader loading on one thread or many)
    VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = ns, .pCode = code };
    VkResult r = vkCreateShaderModule(g_dev, &mci, NULL, &s->mod);
    atomic_fetch_add(&g_spv_mod_bytes, (uint64_t)ns); if (st == 1) atomic_fetch_add(&g_spv_kept_bytes, (uint64_t)ns);
    if (!r) shmod_count(s->mod, (int64_t)ns, 1);
    atomic_store(&s->used_ns, mono_ns());
    if (st == 1 || (st == 0 && intbits_on())) { s->code = code; s->nw = ns / 4; if (st == 0) atomic_fetch_add(&g_spv_kept_bytes, (uint64_t)ns); } else free(code);   // (vertex shaders: the intbits variants)
    if (r) { VLOG("[RED] shader %s: vkCreateShaderModule failed %d", hex, r); if (s->glmod) { shmod_count(s->glmod, (int64_t)s->nglw * 4, -1); vkDestroyShaderModule(g_dev, s->glmod, NULL); } free(s); return NULL; }
    return s;
}
static pthread_mutex_t g_shmu = PTHREAD_MUTEX_INITIALIZER;   // the shader cache: the game's thread and the prewarm thread
// Shaders loaded on every thread that asks (perf-1124 S2). shader_get used to hold g_shmu across shader_load: the pack's inflate, the reflection, 10-15 SPIR-V
//   rewrites and 1-4 vkCreateShaderModule, so the prewarm's 6 threads loaded one shader at a time (Legends: Arceus lists 5690 shaders, 231 MB of SPIR-V, for 3901
//   pipelines) and only vkCreateGraphicsPipelines ran in parallel; the game's thread waited behind the prewarm's loads too. Now a key being loaded is marked
//   (g_shloading) and the load runs outside the lock: another thread asking for the same key waits for that load, a different key loads at once.
//   DM_VK_PREWARM_SHADER_MT=0: under the lock as before
static pthread_cond_t g_shcv = PTHREAD_COND_INITIALIZER; static nvn_hmap g_shloading = {.klen = 8};   // keys being loaded (under g_shmu)
static int g_sh_mt = -1;
static int sh_mt(void) {
    if (g_sh_mt < 0) { const char *e = getenv("DM_VK_PREWARM_SHADER_MT"); g_sh_mt = !(e && *e == '0');
        VLOG("shaders: %s", g_sh_mt ? "loaded on every thread that asks, one load per shader (DM_VK_PREWARM_SHADER_MT)" : "loaded one at a time (DM_VK_PREWARM_SHADER_MT=0)"); }
    return g_sh_mt;
}
static vshader *shader_get(const uint8_t key[8], int st) {
    int mt = sh_mt();
    pthread_mutex_lock(&g_shmu);
    vshader *s = nvn_hm_get(&g_shcache, key);
    if (!s && !mt) { s = shader_load(key, st, g_live_dir); if (s) nvn_hm_put(&g_shcache, key, s); }
    else if (!s) {
        int waited = 0;
        while (!(s = nvn_hm_get(&g_shcache, key)) && nvn_hm_get(&g_shloading, key)) { pthread_cond_wait(&g_shcv, &g_shmu); waited = 1; }
        if (!s && !waited) {   // (one that waited for another thread's load that failed gets NULL, as the loading thread does)
            nvn_hm_put(&g_shloading, key, (void *)1);
            char live[sizeof g_live_dir]; memcpy(live, g_live_dir, sizeof live);
            pthread_mutex_unlock(&g_shmu);
            s = shader_load(key, st, live);
            pthread_mutex_lock(&g_shmu);
            if (s) nvn_hm_put(&g_shcache, key, s);
            nvn_hm_del(&g_shloading, key); pthread_cond_broadcast(&g_shcv);
        }
    }
    pthread_mutex_unlock(&g_shmu);
    return s;
}
// ---- Shader modules released when no pipeline is being built from them (DM_VK_SHMOD_RELEASE=0: kept for the whole run, as before)
// Every module the game's shaders got stayed alive for the whole run (2,209-5,700 of them in players' logs), though Vulkan lets a module go as soon as the pipelines
//   that use it are made: a pipeline keeps what it needs (vkDestroyShaderModule, "pipelines created from it remain valid"). Turnip / Xclipse keep roughly the SPIR-V or
//   more per module (perf-1124: up to 0.1-0.3 MB a module, unmeasured on a phone). A vertex / fragment shader whose module has built no pipeline for 60 s has its
//   modules destroyed (main, GL-depth, fragment and integer-input variants); the SPIR-V we keep (code / glcode: fragment shaders always, vertex shaders under
//   DM_VK_VFMT_INTBITS) makes them again, unchanged, at the next build (shmod_ensure), so the pipeline cache's entries still match. Shaders whose SPIR-V is not kept
//   (compute, geometry, tessellation; capture / point variants) keep theirs. Builds hold g_shmod_rw for reading from making the modules until the driver returned the
//   pipeline; the release takes it for writing only when no build is under way (trywrlock: else at the next round). Every 300 presents with pso_evict
static pthread_rwlock_t g_shmod_rw = PTHREAD_RWLOCK_INITIALIZER; static pthread_mutex_t g_shmod_mu = PTHREAD_MUTEX_INITIALIZER;
static int shmod_on(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SHMOD_RELEASE"); v = !(e && *e == '0');
        VLOG("shader modules: %s", v ? "released 60 s after their last pipeline build, made again from the kept SPIR-V when needed" : "kept for the whole run (DM_VK_SHMOD_RELEASE=0)"); }
    return v;
}
static uint64_t g_shmod_idle_ns = 60000000000ull;   // (the tests shorten it)
static void shmod_release(void) {
    if (!shmod_on() || pthread_rwlock_trywrlock(&g_shmod_rw)) return;   // (a pipeline is being built: the next round)
    uint64_t now = mono_ns(); int n = 0; int64_t bytes = 0;
    pthread_mutex_lock(&g_shmu);
    for (uint32_t i = 0; i < g_shcache.cap; i++) {
        vshader *s = g_shcache.vals[i]; if (!s || !s->code || !s->mod || atomic_load(&s->rel)) continue;
        if (now - atomic_load(&s->used_ns) < g_shmod_idle_ns) continue;
        int64_t b = (int64_t)s->nw * 4;
        vkDestroyShaderModule(g_dev, s->mod, NULL); shmod_count(s->mod, b, -1); s->mod = VK_NULL_HANDLE;
        if (s->glmod && s->glcode) { int64_t gb = (int64_t)s->nglw * 4; vkDestroyShaderModule(g_dev, s->glmod, NULL); shmod_count(s->glmod, gb, -1); s->glmod = VK_NULL_HANDLE; s->relgl = 1; b += gb; }
        for (int v = 0; v < s->nvar; v++) if (s->var[v].mod) { vkDestroyShaderModule(g_dev, s->var[v].mod, NULL); shmod_count(s->var[v].mod, s->var[v].nb, -1); }
        s->nvar = 0;   // (variants are made again by frag_module_ as the next builds ask)
        for (int v = 0; v < s->nibv; v++) if (s->ibv[v].mod) { vkDestroyShaderModule(g_dev, s->ibv[v].mod, NULL); shmod_count(s->ibv[v].mod, s->ibv[v].nb, -1); }
        s->nibv = 0;
        if (s->nrel < 255) s->nrel++;
        atomic_store(&s->rel, 1); n++; bytes += b;
    }
    pthread_mutex_unlock(&g_shmu);
    pthread_rwlock_unlock(&g_shmod_rw);
    if (n) { atomic_fetch_add(&g_shmod_rel_n, (uint64_t)n); atomic_fetch_add(&g_shmod_rel_bytes, (uint64_t)bytes);
             static int said; if (said++ < 20) VLOG("shader modules: %d shaders unused for %.0f s released (%.1f MB of SPIR-V; %lld modules left)", n, g_shmod_idle_ns / 1e9, bytes / 1048576.0, (long long)atomic_load(&g_shmod_n)); }
}
// (g_shmod_rw held for reading) s's modules made again if they were released, and its use noted; 0: one could not be made (the pipeline is not built)
static int shmod_ensure(vshader *s) {
    if (!s) return 1;
    atomic_store(&s->used_ns, mono_ns());
    if (!atomic_load(&s->rel)) return 1;
    int ok = 1; pthread_mutex_lock(&g_shmod_mu);
    if (atomic_load(&s->rel)) {
        VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->nw * 4, .pCode = s->code };
        if (vkCreateShaderModule(g_dev, &mci, NULL, &s->mod)) { s->mod = VK_NULL_HANDLE; ok = 0; } else shmod_count(s->mod, (int64_t)s->nw * 4, 1);
        if (ok && s->relgl) { VkShaderModuleCreateInfo gci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->nglw * 4, .pCode = s->glcode };
            if (vkCreateShaderModule(g_dev, &gci, NULL, &s->glmod)) { s->glmod = VK_NULL_HANDLE; ok = 0; } else { shmod_count(s->glmod, (int64_t)s->nglw * 4, 1); s->relgl = 0; } }
        if (ok) { atomic_store(&s->rel, 0); atomic_fetch_add(&g_shmod_remade, 1); }
        else { if (s->mod) { shmod_count(s->mod, (int64_t)s->nw * 4, -1); vkDestroyShaderModule(g_dev, s->mod, NULL); s->mod = VK_NULL_HANDLE; }
               RED_ONCE(163, "shader %s: its released module could not be made again: pipelines with it are not built", s->key); }
    }
    pthread_mutex_unlock(&g_shmod_mu);
    return ok;
}
typedef struct vprog { vshader *sh[6]; VkDescriptorSetLayout dsl[2]; VkPipelineLayout pl[2]; uint8_t np[2], missing; int gen; } vprog;   // missing: a stage had no shader (built again when live shaders come)   // [0] graphics stages, [1] compute; np: descriptor sets instead of pushes
static nvn_hmap g_progs = {.klen = 8};   // program serial → vprog
static vprog *prog_get(const prog_t *p) {
    // (the live-shader check every 64th program looked up, not every draw: it takes g_shmu and, without live shaders, read the environment each time; the
    //   generation it bumps is read as it is in between. DM_VK_DRAW_MEMO=0: every time)
    static unsigned nget; int gen;
    if (draw_memo_on() && (nget++ & 63)) gen = __atomic_load_n(&g_live_gen, __ATOMIC_ACQUIRE);
    else { pthread_mutex_lock(&g_shmu); live_check(); gen = g_live_gen; pthread_mutex_unlock(&g_shmu); }
    vprog *prev = p->serial ? nvn_hm_get(&g_progs, &p->serial) : NULL; if (prev && (!prev->missing || prev->gen == gen)) return prev;
    shmap_load();
    vprog *o = calloc(1, sizeof *o); o->gen = gen;
    for (unsigned i = 0; i < p->nstage; i++) {
        uint8_t k[8]; if (!prog_key(p, i, k)) continue;
        intptr_t st1 = (intptr_t)nvn_hm_get(&g_shmap, k);
        if (!st1) { pthread_mutex_lock(&g_shmu); st1 = (intptr_t)nvn_hm_get(&g_livemap, k); pthread_mutex_unlock(&g_shmu); }
        if (!st1) { o->missing = 1; if (!prev) { char hex[17]; for (int q = 0; q < 8; q++) sprintf(hex + 2 * q, "%02x", k[q]); VLOG("[RED] shader %s is not in the shader map", hex); } continue; }
        vshader *s = shader_get(k, (int)st1 - 1); if (s) o->sh[st1 - 1] = s; else o->missing = 1;
    }
    if (prev && o->missing && !memcmp(prev->sh, o->sh, sizeof o->sh)) { prev->gen = gen; free(o); return prev; }   // (nothing new for it yet)
    if (p->serial) nvn_hm_put(&g_progs, &p->serial, o);   // (the old one stays allocated: pipelines made from it may still be in flight)
    if (prev) VLOG("a program that lacked a shader was built again with the live shaders");
    return o;
}
static VkDescriptorType desc_type(int kind) {
    switch (kind) { case SPV_RES_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; case SPV_RES_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; case SPV_RES_TEX: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case SPV_RES_TBUF: return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; case SPV_RES_IMG: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; case SPV_RES_IBUF: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        case SPV_RES_SAMP: return VK_DESCRIPTOR_TYPE_SAMPLER; case SPV_RES_SIMG: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; }
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}
static VkPipelineLayout prog_layout(vprog *o, int compute) {
    if (o->pl[compute]) return o->pl[compute];
    VkDescriptorSetLayoutBinding b[324]; uint32_t n = 0;
    for (int st = compute ? 5 : 0; st < (compute ? 6 : 5); st++) { vshader *s = o->sh[st]; if (!s) continue;
        for (int i = 0; i < s->nres && n < 320; i++) b[n++] = (VkDescriptorSetLayoutBinding){ s->res[i].binding + s->shift, desc_type(s->res[i].kind), s->res[i].count, s->vkstage, NULL }; }
    int tf = !compute && o->sh[0] && o->sh[0]->tfmod;   // transform feedback: the capture buffers and { first, base[4] } (tf_draw)
    if (tf) for (int k = 0; k < 4; k++) if (o->sh[0]->tf_stride[k]) b[n++] = (VkDescriptorSetLayoutBinding){ TF_BIND + (uint32_t)k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL };
    VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 20 };
    int dp = !compute && o->sh[0] && o->sh[0]->dp;   // draw parameters from push constants (dp_push): one range over both
    if (dp) pcr.size = SPV_DRAWP_PC_OFFSET + 12;
    // more descriptors than one push takes (Pokémon Legends: Arceus' terrain: 29, past the 32 of some Adreno drivers with other programs): this program uses descriptor sets
    uint32_t nd = 0; for (uint32_t i = 0; i < n; i++) nd += b[i].descriptorCount;
    o->np[compute] = (uint8_t)(g_nopush || nd > g_maxpush);
    if (o->np[compute] && !g_nopush) { static int said; if (!said++) VLOG("a program with %u descriptors (push limit %u): descriptor sets for such programs", nd, g_maxpush); }
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = o->np[compute] ? 0 : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = n, .pBindings = b };
    VKCHECK(vkCreateDescriptorSetLayout(g_dev, &dci, NULL, &o->dsl[compute]));
    VkPipelineLayoutCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &o->dsl[compute], .pushConstantRangeCount = tf || dp ? 1u : 0u, .pPushConstantRanges = &pcr };
    VKCHECK(vkCreatePipelineLayout(g_dev, &pci, NULL, &o->pl[compute]));
    return o->pl[compute];
}

// ================= Replay state (as nvn_mtl.m) =================
static struct { p_set_rt rt; int has_rt; uint8_t clear_c[8]; VkClearColorValue cc[8]; uint8_t clear_d, clear_s; float cd; uint32_t cs; p_scissor sc; int has_sc; } R;
static struct {
    const prog_t *prog, *cprog, *bst[5];
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16]; uint64_t tex[6][NTEX], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp; int depth_clamp; float blend_color[4]; int64_t copy_stride, copy_img_stride; int rast_discard; p_tf_bind tf[4]; uint64_t tf_off[4]; int tf_on;
} S;
static _Atomic int g_origin_ll, g_depth_zero;
void mtl_set_origin_lower_left(int on) { if (atomic_exchange(&g_origin_ll, on) != on) VLOG("window origin: %s", on ? "lower left (GL)" : "upper left"); }
void mtl_set_depth_zero(int on) { if (atomic_exchange(&g_depth_zero, on) != on) VLOG("depth mode: %s", on ? "near = 0" : "near = -w (GL)"); }
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1;
}
// NVN binds programs per stage (nvn_mtl.m bind_stages): stages of a program bound for only some of them combine into a host program
static const prog_t *bind_stages(const prog_t *p, int stages) {
    for (int st = 0; st < 5; st++) if (stages >> st & 1) S.bst[st] = p;
    const prog_t *v = S.bst[0]; if (!v) return NULL;
    int same = 1; for (int st = 1; st < 5; st++) if (S.bst[st] && S.bst[st] != v) same = 0;
    if (same && S.bst[1] == v) return v;
    // The combination by the shaders it ends up with, made once and kept: a 64-entry ring of recent combinations was here, and Shield cycles through more than 64
    // a frame, so nearly every draw missed and made a new host program with two new Vulkan layouts that were never freed (3.5 million in 90 s on the Mac: the memory
    // creep, and driver work on every draw). The set of shader combinations is the game's, so this map stops growing once they have all been seen
    static nvn_hmap comb = {.klen = sizeof(vshader *) * 5};
    vshader *key[5] = {0};
    for (int st = 0; st < 5; st++) if (S.bst[st]) { vprog *src = prog_get(S.bst[st]); key[st] = src ? src->sh[st] : NULL; }
    prog_t *cp = nvn_hm_get(&comb, key); if (cp) return cp;
    vprog *o = calloc(1, sizeof *o); memcpy(o->sh, key, sizeof key);
    cp = calloc(1, sizeof *cp); cp->magic = MAGIC('P', 'R', 'O', 'G'); cp->serial = nvn_new_serial();
    nvn_hm_put(&g_progs, &cp->serial, o); nvn_hm_put(&comb, key, cp);
    return cp;
}
void mtl_prog_release(prog_t *p) { (void)p; }   // programs are kept (keyed by serial; a released one is never looked up again)

// ================= Render passes =================
static struct { int open; uint32_t cfmt[8], dfmt, sfmt, samples, tw, th; int hd, hs, ncolor; uint32_t layers; uint64_t dkey, skey; int dnone, snone, dwrote, swrote, dtest; uint32_t b10; } P;   // dkey..: skip stores   // dtest: a draw tested depth or stencil (seg_record_prim's LOAD_OP_NONE)   // layers: a layered pass (a 3D target's slices) when > 1   // b10: colour targets that are R11G11B10F's RGBA16F stand-in (bit i: colour i, DM_VK_B10_AS_RGBA16)
// DM_VK_PROF: what ended each open pass (0 a new render target, 1 a uniform update, 2 a buffer copy, 3 a buffer fill), and the cost of opening one split into its
//   attachment views, the barrier before it and the driver's begin (g_pb_ns); g_brk_other: any work outside a pass that ended one (the causes above included)
static uint64_t g_brk[4], g_pb_ns[3], g_brk_other;
// Every 300 presents (mtl_present), for all players: what ended the passes, and how many passes opened onto exactly the attachments of the one a target change had just
//   ended (g_pass_samev): passes that could go on as one. The data for joining passes on drivers where each one costs a load and a store of its attachments (#99)
static uint64_t g_pass_samev; static int g_pass_last_by_rt; static VkImageView g_pass_lastv[9]; static uint32_t g_pass_lastwh[3];
static VkCommandBuffer CB(void) { return g_sub->cb; }
// ================= Frame capture (a bug report's pictures) =================
// One frame, pass by pass: each pass's colour targets scaled into 320x180 pictures (blits into one array image, copied out as they come), and notes on what the pass
// drew with (target formats, the textures its draws read with their stored / viewed formats and how they were converted here, the first draws' shaders). Asked for
// by the in-game menu (Bug reports: it writes capture.req in the game's folder, holding how many frames to wait first) or DM_VK_CAPTURE_AT=<present number>; written
// as frame-capture.zip in the game's folder and in Download/. Nothing runs or is kept while not asked for. The Mali white scenes (#47, #53, #56) show no error in the
// log: the pictures show which pass goes white, the notes what it read
enum { CAPN = 192, CAPS = 256, CAPP = 12 };   // CAPS targets sampled, CAPP points each (4 x 3), 16 bytes a point
static uint32_t CAPW = 320, CAPH = 180;   // the pictures' size: DM_VK_CAPTURE_SIZE=<w>x<h> (up to 960x540: 192 of them are kept on the GPU and in a buffer), for a detail 320x180 loses
static int g_cap_ns; static VkFormat g_cap_sfmt[CAPS]; static char g_cap_slabel[CAPS][40];
static uint32_t fmt_texel_bytes(VkFormat f) {   // the formats render targets come in here; 0: not sampled
    switch ((int)f) {
    case 37: case 43: case 44: case 50: case 64: case 122: case 100: case 98: case 99: case 82: case 83: return 4;   // RGBA8 / BGRA8 (+sRGB), A2B10G10R10, B10G11R11, R32 F / UI / SI, RG16 F / U
    case 97: case 95: case 96: case 103: case 101: case 102: case 91: return 8;   // RGBA16 F / UI / SI, RG32 F / UI / SI, RGBA16 unorm
    case 109: case 107: case 108: return 16;   // RGBA32 F / UI / SI
    case 76: case 75: case 9: case 16: return f == 9 ? 1 : 2;   // R16 F / UNORM, R8, RG8
    }
    return 0;
}
static _Atomic int g_cap;   // (declared with vtex too) 0 off, 1 armed (g_cap_wait more presents first), 2 recording this frame, 3 recorded: written at the next present
static int g_cap_wait, g_cap_n, g_cap_layout, g_cap_draws, g_cap_ntex; static int g_cap_tex[64];
// Draw by draw (values.txt): in the frame's first full-screen B10G11R11 pass (the lit scene), colour 0's 12 points after each of its first CAPD draws, and each draw's
//   fragment shader: on a GPU whose scene comes out white (Mali, #47 / #53 / #64 / #68 / #69) the first draw that leaves them at the format's maximum names the shader
enum { CAPD = 128 };
static int g_cap_dp, g_cap_nd; static char g_cap_dkey[CAPD][17];   // g_cap_dp: 0 not chosen yet, 1 in that pass, 2 done
static VkFormat g_cap_dfmt;   // that pass's colour 0 as stored: B10G11R11, or the RGBA16F stand-in (DM_VK_B10_AS_RGBA16)
static VkImage g_cap_img; static vmalloc_t g_cap_ia; static VkBuffer g_cap_buf; static vmalloc_t g_cap_ba;
static char *g_cap_meta; static size_t g_cap_len, g_cap_size; static uint64_t g_cap_frame;
static void cap_printf(const char *f, ...) {
    va_list a; va_start(a, f); char tmp[512]; int n = vsnprintf(tmp, sizeof tmp, f, a); va_end(a); if (n <= 0) return; if (n >= (int)sizeof tmp) n = sizeof tmp - 1;
    if (g_cap_len + (size_t)n + 1 > g_cap_size) { g_cap_size = (g_cap_size + (size_t)n + 1) * 2; g_cap_meta = realloc(g_cap_meta, g_cap_size); }
    memcpy(g_cap_meta + g_cap_len, tmp, (size_t)n); g_cap_len += (size_t)n; g_cap_meta[g_cap_len] = 0;
}
static int cap_start(void) {
    { const char *e = getenv("DM_VK_CAPTURE_SIZE"); unsigned w, h; if (e && sscanf(e, "%ux%u", &w, &h) == 2 && w >= 64 && h >= 36 && w <= 960 && h <= 540) { CAPW = w; CAPH = h; } }
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { CAPW, CAPH, 1 }, .mipLevels = 1,
        .arrayLayers = CAPN, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
    if (vkCreateImage(g_dev, &ci, NULL, &g_cap_img)) return 0;
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, g_cap_img, &rq); g_cap_ia = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, g_cap_img, g_cap_ia.b->mem, g_cap_ia.off);
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)CAPN * CAPW * CAPH * 4 + (VkDeviceSize)CAPS * CAPP * 16 + (VkDeviceSize)CAPD * CAPP * 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    if (vkCreateBuffer(g_dev, &bi, NULL, &g_cap_buf)) { vkDestroyImage(g_dev, g_cap_img, NULL); vk_free(g_cap_ia); return 0; }
    vkGetBufferMemoryRequirements(g_dev, g_cap_buf, &rq); g_cap_ba = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1); vkBindBufferMemory(g_dev, g_cap_buf, g_cap_ba.b->mem, g_cap_ba.off);
    g_cap_n = 0; g_cap_layout = 0; g_cap_draws = 0; g_cap_ntex = 0; g_cap_len = 0; g_cap_ns = 0; g_cap_dp = 0; g_cap_nd = 0;
    cap_printf("Frame capture, present %llu. GPU %s (vendor %#x, driver %#x, Vulkan %u.%u). BC textures %s; BC layer %s; float controls %s.\n", (unsigned long long)g_cap_frame, g_props.deviceName, g_props.vendorID,
               g_props.driverVersion, VK_API_VERSION_MAJOR(g_props.apiVersion), VK_API_VERSION_MINOR(g_props.apiVersion),
               nvn_fmt_vk_transcode(0x46) ? "transcoded on upload" : nvn_fmt_vk_decode(0x46) ? "decoded on upload" : "sampled by the GPU", g_bcgpu ? "on" : "off", g_fctl ? "on" : "off");
    cap_printf("Formats are Vulkan VkFormat numbers (vk) and NVN format numbers (nvn). Pictures: NNN.png, %ux%u, values above 1.0 shown as white.\n", CAPW, CAPH);
    mw_capture_note();   // (the Mali white-screen check's state, when its ladder is on)
    cap_printf("\n");
    return 1;
}
static void cap_pass(void) {
    VkCommandBuffer cb = CB(); int idx_first = g_cap_n;
    cap_printf("pass: %ux%u, %d draws, %d colour target(s), depth format vk %u\n", P.tw, P.th, g_cap_draws, P.ncolor, P.dfmt);
    for (int i = 0; i < 8; i++) {
        const tex_t *ct = R.rt.color[i]; if (!ct) continue;
        vtex *x = tex_res(ct); if (!x || !x->img) { cap_printf("  colour %d: no image\n", i); continue; }
        sk_read(x);
        uint32_t lv = R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0, ly = R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0;
        VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, x->fmt, &fp);
        const char *skip = x->samples > 1 ? "multisampled" : fmt_int_class(x->fmt) ? "integer format" : !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? "format cannot be blitted" : g_cap_n >= CAPN ? "no room left" : NULL;
        cap_printf("  colour %d: vk %d (nvn %#x) %ux%u level %u layer %u%s%s", i, x->fmt, ct->d.format, x->w, x->h, lv, ly, x->is3d ? " (3D)" : "", b10_swapped_tex(x) ? " (R11G11B10F stored as RGBA16F)" : "");
        {   uint32_t tb = fmt_texel_bytes(x->fmt);
            if (tb && x->samples <= 1 && g_cap_ns < CAPS) {   // values.txt: the pixels' own numbers (the pictures clamp everything above 1.0 to white)
                uint32_t sw = x->w >> lv ? x->w >> lv : 1, sh = x->h >> lv ? x->h >> lv : 1; VkBufferImageCopy pc[CAPP];
                for (int k = 0; k < CAPP; k++) pc[k] = (VkBufferImageCopy){ .bufferOffset = (VkDeviceSize)CAPN * CAPW * CAPH * 4 + ((VkDeviceSize)g_cap_ns * CAPP + (VkDeviceSize)k) * 16,
                    .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, lv, x->is3d ? 0 : ly, 1 }, .imageOffset = { (int32_t)(sw * (uint32_t)(1 + 2 * (k % 4)) / 8), (int32_t)(sh * (uint32_t)(1 + 2 * (k / 4)) / 6), x->is3d ? (int32_t)ly : 0 },
                    .imageExtent = { 1, 1, 1 } };
                barrier_all(cb); vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_cap_buf, CAPP, pc); barrier_all(cb);
                g_cap_sfmt[g_cap_ns] = x->fmt; snprintf(g_cap_slabel[g_cap_ns], sizeof g_cap_slabel[0], skip ? "colour %d of the pass before %03d" : "%03d.png", skip ? i : g_cap_n, g_cap_n);
                cap_printf(" [values #%d]", g_cap_ns); g_cap_ns++;
            }
        }
        if (skip) { cap_printf(", no picture (%s)\n", skip); continue; }
        if (!g_cap_layout) { g_cap_layout = 1;
            VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .image = g_cap_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, CAPN } };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); }
        int32_t w = (int32_t)(x->w >> lv ? x->w >> lv : 1), h = (int32_t)(x->h >> lv ? x->h >> lv : 1);
        VkImageBlit r = { { VK_IMAGE_ASPECT_COLOR_BIT, lv, x->is3d ? 0 : ly, 1 }, { { 0, 0, x->is3d ? (int32_t)ly : 0 }, { w, h, x->is3d ? (int32_t)ly + 1 : 1 } },
                          { VK_IMAGE_ASPECT_COLOR_BIT, 0, (uint32_t)g_cap_n, 1 }, { { 0, 0, 0 }, { CAPW, CAPH, 1 } } };
        barrier_all(cb);
        vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_cap_img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST);
        barrier_all(cb);
        VkBufferImageCopy bc = { .bufferOffset = (VkDeviceSize)g_cap_n * CAPW * CAPH * 4, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, (uint32_t)g_cap_n, 1 }, .imageExtent = { CAPW, CAPH, 1 } };
        vkCmdCopyImageToBuffer(cb, g_cap_img, VK_IMAGE_LAYOUT_GENERAL, g_cap_buf, 1, &bc);
        barrier_all(cb);
        cap_printf(" -> %03d.png\n", g_cap_n); g_cap_n++;
    }
    (void)idx_first; g_cap_draws = 0; g_cap_ntex = 0; if (g_cap_dp == 1) g_cap_dp = 2;
}
// a texture a draw of the recording frame reads: noted once a pass
static void cap_tex(int tid, const tex_t *t, const vtex *x, int vfmt, VkFormat f) {
    for (int i = 0; i < g_cap_ntex; i++) if (g_cap_tex[i] == tid) return;
    if (g_cap_ntex < 64) g_cap_tex[g_cap_ntex++] = tid;
    cap_printf("  reads texture %d: nvn %#x, image vk %d %ux%u%s, viewed as vk %d; BC here: %s%s\n", tid, t->d.format, x->fmt, x->w, x->h,
               x->pkg ? " (from the game's files)" : " (made at run time)", f, nvn_fmt_vk_transcode(t->d.format) ? "transcoded" : nvn_fmt_vk_decode(t->d.format) ? "decoded" : "no",
               x->alt && x->alt_fmt == vfmt ? "; this view reads its own second conversion" : "");
    if (vfmt) cap_printf("    (view format nvn %#x)\n", vfmt);
}
static void cap_write(void) {
    const char *pc = getenv("DM_VK_PCACHE"); char dir[1024] = "."; if (pc) { snprintf(dir, sizeof dir, "%s", pc); char *sl = strrchr(dir, '/'); if (sl) *sl = 0; }
    char path[1200]; snprintf(path, sizeof path, "%s/frame-capture.zip", dir);
    const uint8_t *px = (const uint8_t *)g_cap_ba.b->map + g_cap_ba.off;
    fczip *z = fczip_open(path); int ok = z != NULL;
    if (z) {
        for (int i = 0; i < g_cap_n; i++) { char nm[32]; snprintf(nm, sizeof nm, "%03d.png", i); ok &= fczip_add_png(z, nm, px + (size_t)i * CAPW * CAPH * 4, CAPW, CAPH); }
        ok &= fczip_add(z, "passes.txt", g_cap_meta ? g_cap_meta : "", g_cap_len);
        {   // values.txt: each sampled target's 12 points (4 columns x 3 rows, left to right then top to bottom), decoded from its format
            size_t cap = 4096 + (size_t)g_cap_ns * CAPP * 96 + (size_t)CAPD * 640, len = 0; char *t = malloc(cap); const uint8_t *sv = px + (size_t)CAPN * CAPW * CAPH * 4;
            if (t) {
                len += (size_t)snprintf(t + len, cap - len, "Pixel values at 12 points of each target (4 columns x 3 rows, left to right, top to bottom). inf / nan: not a number.\n");
                for (int i = 0; i < g_cap_ns && len + 2048 < cap; i++) {
                    len += (size_t)snprintf(t + len, cap - len, "\n#%d %s, vk %d:\n", i, g_cap_slabel[i], g_cap_sfmt[i]);
                    for (int k = 0; k < CAPP; k++) {
                        const uint8_t *q = sv + ((size_t)i * CAPP + (size_t)k) * 16; float v[4] = { 0 }; int nv = 4, isint = 0; uint32_t u[4] = { 0 };
                        switch ((int)g_cap_sfmt[i]) {
                        case 122: { uint32_t w; memcpy(&w, q, 4); uint32_t c[3] = { w & 0x7ff, w >> 11 & 0x7ff, w >> 22 }; int mb[3] = { 6, 6, 5 };
                            for (int j = 0; j < 3; j++) { uint32_t e = c[j] >> mb[j], m = c[j] & ((1u << mb[j]) - 1); v[j] = e == 31 ? (m ? NAN : INFINITY) : e ? ldexpf(1.0f + m / (float)(1u << mb[j]), (int)e - 15) : ldexpf(m / (float)(1u << mb[j]), -14); }
                            nv = 3; break; }
                        case 97: case 76: case 83: { int n = g_cap_sfmt[i] == 97 ? 4 : g_cap_sfmt[i] == 83 ? 2 : 1; for (int j = 0; j < n; j++) { uint16_t h; memcpy(&h, q + 2 * j, 2); uint32_t e = h >> 10 & 31, m = h & 1023;
                            float f = e == 31 ? (m ? NAN : INFINITY) : e ? ldexpf(1.0f + m / 1024.0f, (int)e - 15) : ldexpf(m / 1024.0f, -14); v[j] = h & 0x8000 ? -f : f; } nv = n; break; }
                        case 100: case 103: case 109: { int n = g_cap_sfmt[i] == 100 ? 1 : g_cap_sfmt[i] == 103 ? 2 : 4; memcpy(v, q, 4 * (size_t)n); nv = n; break; }
                        case 95: case 96: case 91: { for (int j = 0; j < 4; j++) { uint16_t h; memcpy(&h, q + 2 * j, 2); u[j] = h; } isint = 1; break; }
                        case 98: case 99: case 107: case 108: case 101: case 102: { int n = g_cap_sfmt[i] >= 107 ? 4 : g_cap_sfmt[i] >= 101 ? 2 : 1; memcpy(u, q, 4 * (size_t)n); nv = n; isint = 1; break; }
                        case 64: { uint32_t w; memcpy(&w, q, 4); v[0] = (w & 1023) / 1023.0f; v[1] = (w >> 10 & 1023) / 1023.0f; v[2] = (w >> 20 & 1023) / 1023.0f; v[3] = (w >> 30) / 3.0f; break; }
                        default: { uint32_t tb = fmt_texel_bytes(g_cap_sfmt[i]); nv = (int)tb; for (int j = 0; j < nv && j < 4; j++) v[j] = q[j] / 255.0f; if (nv > 4) nv = 4; }
                        }
                        len += (size_t)snprintf(t + len, cap - len, "  (");
                        for (int j = 0; j < nv; j++) len += isint ? (size_t)snprintf(t + len, cap - len, "%s%u", j ? ", " : "", u[j]) : (size_t)snprintf(t + len, cap - len, "%s%.4g", j ? ", " : "", v[j]);
                        len += (size_t)snprintf(t + len, cap - len, ")%s", k % 4 == 3 ? "\n" : " ");
                    }
                }
                if (g_cap_nd && len + 512 < cap) {   // draw by draw in the first lit pass (cap_draw_values): colour 0 (B10G11R11) after each draw
                    const uint8_t *dv = sv + (size_t)CAPS * CAPP * 16;
                    len += (size_t)snprintf(t + len, cap - len, "\nDraw by draw, the frame's first full-screen B10G11R11 pass%s: its colour 0 at the same 12 points after each of its first %d draws."
                                                                 "\n'max' counts the points at the format's maximum (65024: overflow); then the fragment shader and the 12 values.\n",
                                            g_cap_dfmt == VK_FORMAT_R16G16B16A16_SFLOAT ? " (stored as RGBA16F: DM_VK_B10_AS_RGBA16; 'max' counts 60000 and up, infinity and NaN)" : "", g_cap_nd);
                    for (int d = 0; d < g_cap_nd && len + 1024 < cap; d++) {
                        float v[CAPP][3]; int nmax = 0;
                        for (int k = 0; k < CAPP; k++) { const uint8_t *q = dv + ((size_t)d * CAPP + (size_t)k) * 16;
                            if (g_cap_dfmt == VK_FORMAT_R16G16B16A16_SFLOAT) { for (int j = 0; j < 3; j++) { uint16_t h; memcpy(&h, q + 2 * j, 2); v[k][j] = b10_half_to_float(h); } }   // (the stand-in's texels)
                            else { uint32_t w; memcpy(&w, q, 4); for (int j = 0; j < 3; j++) v[k][j] = b10_chan_to_float(j == 0 ? w & 0x7ff : j == 1 ? w >> 11 & 0x7ff : w >> 22, j == 2 ? B10_MB : B10_MR); }
                            if (v[k][0] >= 6e4f || v[k][1] >= 6e4f || v[k][2] >= 6e4f || isnan(v[k][0]) || isnan(v[k][1]) || isnan(v[k][2])) nmax++; }
                        len += (size_t)snprintf(t + len, cap - len, "draw %3d, max %2d, fragment %s:", d + 1, nmax, g_cap_dkey[d]);
                        for (int k = 0; k < CAPP; k++) len += (size_t)snprintf(t + len, cap - len, " (%.3g %.3g %.3g)", v[k][0], v[k][1], v[k][2]);
                        len += (size_t)snprintf(t + len, cap - len, "\n");
                    }
                }
                ok &= fczip_add(z, "values.txt", t, len); free(t);
            }
        }
        ok &= fczip_close(z);
    }
    VLOG("frame capture: %d pictures%s %s", g_cap_n, ok ? ", written to" : ", could NOT be written to", path);
    if (ok) {   // a copy in Download/, where a player finds it (when the app may write there)
        const char *id = getenv("DM_PROGRAM_ID"); char dl[256]; snprintf(dl, sizeof dl, "/storage/emulated/0/Download/%s-frame-capture.zip", id ? id : "game");
        FILE *in = fopen(path, "rb"), *out = in ? fopen(dl, "wb") : NULL; char b[65536]; size_t n;
        if (in && out) { while ((n = fread(b, 1, sizeof b, in)) > 0) fwrite(b, 1, n, out); VLOG("frame capture: copied to %s", dl); }
        if (in) fclose(in); if (out) fclose(out);
    }
    vkDestroyImage(g_dev, g_cap_img, NULL); vk_free(g_cap_ia); vkDestroyBuffer(g_dev, g_cap_buf, NULL); vk_free(g_cap_ba);
    g_cap_img = VK_NULL_HANDLE; g_cap_buf = VK_NULL_HANDLE;
}
// at every present: the request file / DM_VK_CAPTURE_AT arm it; a recorded frame is written once the GPU has finished it
static int hold_dec(void);
static void cap_present(uint64_t n) {
    int st = atomic_load(&g_cap);
    if (st == 0 && hold_dec()) return;   // (not with DM_VK_FRAME_HOLD=2: the capture ends a pass and records into its command buffer)
    if (st == 0) {
        static long at = -2; if (at == -2) { const char *e = getenv("DM_VK_CAPTURE_AT"); at = e ? atol(e) : -1; }
        int wait = -1;
        if (at > 0 && (long)n == at) wait = 0;
        else if (n % 30 == 0) {
            const char *pc = getenv("DM_VK_PCACHE"); char req[1100]; if (!pc) return; snprintf(req, sizeof req, "%s", pc); char *sl = strrchr(req, '/'); if (!sl) return;
            snprintf(sl + 1, sizeof req - (size_t)(sl + 1 - req), "capture.req");
            FILE *f = fopen(req, "r"); if (f) { if (fscanf(f, "%d", &wait) != 1) wait = 90; fclose(f); unlink(req); }
        }
        if (wait >= 0) { g_cap_wait = wait; atomic_store(&g_cap, 1); VLOG("frame capture: asked for, in %d frames", wait); }
        return;
    }
    if (st == 1) { if (g_cap_wait-- > 0) return; g_cap_frame = n; if (cap_start()) atomic_store(&g_cap, 2); else { VLOG("[RED] frame capture: no memory for it"); atomic_store(&g_cap, 0); } return; }
    if (st == 2) { atomic_store(&g_cap, 3); return; }
    if (st == 3) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu); cap_write(); atomic_store(&g_cap, 0); }
}
static int P_tf;   // the open pass is transform feedback captures' (tf_pass_begin), not the game's
static struct { uint64_t lo, hi; } g_tfw[64]; static int g_ntfw;   // what the captures in the open capture pass wrote
static void ubd_flush(void);
static int g_nubd;
static int g_lc_hazard;   // (lazy clears: the open pass took a clear by its load op and a draw in it samples that image; draw_ begins it again)
static void pass_end(void) { g_rt_stale = 0; if (P.open) { g_lc_open = NULL; g_lc_hazard = 0;
    if (!P_tf && g_sub && g_sub->dec && g_seg_in) { g_sub->cur.dsk = 1; g_sub->cur.dsw = (uint8_t)(P.dwrote || P.swrote); g_sub->cur.dst = (uint8_t)P.dtest; }   // (a held game pass: what it did to its depth / stencil, seg_record_prim)
    rend_end(CB()); fgr_pass_end(CB()); if (!P_tf) hp_pass_end(); P.open = 0; if (g_mw_px) mw_pass_end();
    if (!P_tf && P.hd) sk_ro_note(P.dkey, P.dwrote, 0);
    if (!P_tf && P.hs) sk_ro_note(P.skey, P.swrote, 0);
    g_after_pass = !P_tf; g_bar_clean = 0;
    if (P_tf) {   // the captures' writes made visible to what reads them (vertex streams, shaders, copies)
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
        vkCmdPipelineBarrier(CB(), VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL); P_tf = 0; g_ntfw = 0; }
    if (g_rtq && g_sub && g_sub->rq_pend) report_ts_flush(CB(), g_sub); if (atomic_load(&g_cap) == 2) cap_pass(); }
    if (g_nubd) ubd_flush(); }
static void tf_end(void) { if (P.open && P_tf) pass_end(); }   // before the game's own work: a capture pass is not the game's pass
static int rt_dup_slot(const p_set_rt *r, int i) {
    if (!r->color[i]) return 0;
    uint8_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0; uint16_t ly = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
    for (int j = 0; j < i; j++) { if (!r->color[j] || r->color[j]->serial != r->color[i]->serial) continue;
        uint8_t lj = r->has_cview[j] && r->cview[j].set_levels ? r->cview[j].base_level : 0; uint16_t yj = r->has_cview[j] && r->cview[j].set_layers ? r->cview[j].min_layer : 0;
        if (lj == lv && yj == ly) return 1; }
    return 0;
}
static int format_supports_attachment(VkFormat f, VkImageUsageFlags usage, VkFormatFeatureFlags *out) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g_pd, f, &fp);
    VkFormatFeatureFlags features = fp.optimalTilingFeatures;
    if (out) *out = features;
    if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    if ((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && !(features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) return 0;
    return 1;
}
// the attachment view of a render target: one level, one layer (3D: one depth slice through a 2D-array view is not available, the slice is taken as layer 0 — first case reported)
static uint32_t g_att_layers = 1;   // att_view: how many layers the view it returned has (a 3D target's slices)
static VkImageView att_view(const tex_t *t, const view_t *v, int has_v, vtex **out, uint32_t *w, uint32_t *h) {
    vtex *x = tex_res(t); *out = x;
    if (!x || !x->img) {
        static int missing;
        if (missing < 64) {
            missing++;
            VLOG("render target ser=%llu NVN format %#x has no Vulkan image%s",
                 (unsigned long long)(t ? t->serial : 0), t ? t->d.format : 0, x ? " handle" : " resource");
        }
        return VK_NULL_HANDLE;
    }
    if (!(x->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
        RED_ONCE(19, "render target ser=%llu NVN format %#x / Vulkan format %d cannot be attached (image usage %#x)",
                 (unsigned long long)t->serial, t->d.format, x->fmt, x->usage);
        return VK_NULL_HANDLE;
    }
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0, ly = has_v && v->set_layers ? v->min_layer : 0;
    g_att_layers = 1;
    if (lv >= x->levels || (!x->is3d && ly >= x->layers)) return VK_NULL_HANDLE;
    if (x->cube_ok) { static uint64_t said[32]; static int nsaid; int seen = 0; uint64_t key = t->serial << 8 | ly << 4 | lv;
        for (int q = 0; q < nsaid; q++) if (said[q] == key) seen = 1;
        if (!seen && nsaid < 32) { said[nsaid++] = key; VLOG("render target: cube texture ser %llu level %u layer %u, view layers %u (set %d)", (unsigned long long)t->serial, lv, ly,
            has_v && v->set_layers ? v->nlayers : 0, has_v && v->set_layers); } }
    VkFormat f = x->fmt;
    if (has_v && v->set_format) { uint32_t vf = 0; uint8_t s[4]; if (nvn_fmt_to_vk(v->format, &vf, s) && vf) f = (VkFormat)vf; }
    if (b10_on()) { char who[48]; snprintf(who, sizeof who, "render target ser=%llu", (unsigned long long)t->serial);   // (DM_VK_B10_AS_RGBA16; pass_begin gives the pass the same format)
        f = b10_view_fmt(x, f, B10V_ATTACH, who); if (!f) return VK_NULL_HANDLE; }
    VkImageUsageFlags attachment_usage = x->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkFormatFeatureFlags features = 0;
    if (!format_supports_attachment(f, attachment_usage, &features)) {
        if (f != x->fmt && format_supports_attachment(x->fmt, attachment_usage, &features)) {
            VLOG("render target ser=%llu view format %d lacks attachment support; using native format %d (features %#x)",
                 (unsigned long long)t->serial, f, x->fmt, features);
            f = x->fmt;
        } else {
            static int rejected;
            if (rejected < 64) {
                rejected++;
                VLOG("render target ser=%llu view format %d rejected: image format %d, image usage %#x, view usage %#x, optimal features %#x",
                     (unsigned long long)t->serial, f, x->fmt, x->usage, attachment_usage, features);
            }
            return VK_NULL_HANDLE;
        }
    }
    *w = x->w >> lv ? x->w >> lv : 1; *h = x->h >> lv ? x->h >> lv : 1;
    // A 3D target: all its slices at that level as a 2D array, the pass layered over them (the shader picks the slice with gl_Layer). Violet fills its 32x32x32 colour-grading
    //   table so, in one instanced draw; with slice 0 alone attached the other 31 stayed zero, tone mapping looked them up, and every 3D scene came out black
    static int flat = -1; if (flat < 0) flat = getenv("DM_VK_NO_LAYERED_3D") != NULL;   // DM_VK_NO_LAYERED_3D: slice 0 only, as before (to compare)
    if (x->is3d && flat) return view_get(x, VK_IMAGE_VIEW_TYPE_2D_ARRAY, f, lv, 1, 0, 1, SW_ID, x->aspect);
    if (x->is3d) { uint32_t dz = x->d >> lv ? x->d >> lv : 1; if (ly >= dz) return VK_NULL_HANDLE; g_att_layers = dz - ly;
        return view_get(x, VK_IMAGE_VIEW_TYPE_2D_ARRAY, f, lv, 1, ly, dz - ly, SW_ID, x->aspect); }
    return view_get(x, VK_IMAGE_VIEW_TYPE_2D, f, lv, 1, ly, 1, SW_ID, x->aspect);
}
// DM_NVN_TEXDUMP=frame:path (as nvn_mtl.m): every colour render target that frame uses, saved once the frame is over as <path with %d = index * 100 + level> (PNG, clamped to [0,1])
static long g_td_frame = -2; static const char *g_td_path; static int g_td_done;
static struct { vtex *x; uint64_t ser; uint32_t lv, sl; } g_td[256]; static int g_ntd;
static void shot_level(vtex *x, uint32_t lv, uint32_t sl, const char *path);
// the dumped frame also logs its passes, draws and sampled textures (which pass reads what), for a picture that comes out wrong
#define TD_TRACE (g_td_frame >= 0 && !g_td_done && (long)g_frame_no == g_td_frame)
static void td_note(vtex *x, uint64_t ser, uint32_t lv, uint32_t sl) {
    if (g_td_frame == -2) { const char *e = getenv("DM_NVN_TEXDUMP"), *c = e ? strchr(e, ':') : NULL; g_td_frame = c ? atol(e) : -1; g_td_path = c ? c + 1 : NULL; }
    // DM_NVN_TEXDUMP_NOW=file:path: when that file appears (looked for every 30 frames), the next frame is dumped to path and the file removed (a scene reached by playing, at no known frame)
    { static const char *nowf, *nowp; static int init; static uint64_t last;
      if (!init) { init = 1; const char *e = getenv("DM_NVN_TEXDUMP_NOW"), *c = e ? strchr(e, ':') : NULL; if (c) { static char f[512]; snprintf(f, sizeof f, "%.*s", (int)(c - e), e); nowf = f; nowp = c + 1; } }
      if (nowf && g_td_frame < 0 && g_frame_no - last >= 30) { last = g_frame_no; if (!access(nowf, F_OK)) { unlink(nowf); g_td_frame = (long)g_frame_no + 1; g_td_path = nowp; g_td_done = 0; g_ntd = 0; } } }
    if (g_td_frame < 0 || g_td_done) return;
    if ((long)g_frame_no > g_td_frame) {
        g_td_done = 1; pass_end(); uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_sub_begin();
        for (int i = 0; i < g_ntd; i++) { char path[1024]; snprintf(path, sizeof path, g_td_path, (int)(g_td[i].ser * 100 + g_td[i].sl * 10 + g_td[i].lv)); shot_level(g_td[i].x, g_td[i].lv, g_td[i].sl, path);
            VLOG("[texdump] %d ser=%llu %ux%u level %u format %d → %s", i, (unsigned long long)g_td[i].ser, g_td[i].x->w, g_td[i].x->h, g_td[i].lv, g_td[i].x->fmt, path); }
        return;
    }
    if ((long)g_frame_no != g_td_frame || g_ntd >= 256) return;
    for (int i = 0; i < g_ntd; i++) if (g_td[i].ser == ser && g_td[i].lv == lv && g_td[i].sl == sl) return;
    vtex_ref(x); g_td[g_ntd].x = x; g_td[g_ntd].ser = ser; g_td[g_ntd].lv = lv; g_td[g_ntd].sl = sl; g_ntd++;
}
// A transform feedback capture runs in a 1x1 pass of its own with no attachments (as nvn_mtl.m tf_draw's encoder): its pipeline has rasterizer discard on, so it has no colour or
// depth state, and inside the game's pass its attachment count would not match the pass (VUID-vkCmdDraw-colorAttachmentCount-06179). Ended by pass_end like any pass, which
// makes the captures' writes visible. Captures in a row share one pass and that one barrier (Legends: Arceus skins about 180 times a frame, and on Adreno each barrier waits
// for the GPU to go idle; from the Arceus engine, androidswithy 84fb5df) unless one reads or rewrites what an earlier one in the pass wrote (keep = 0); anything else
// between them ends the pass (tf_end). DM_VK_TF_BATCH=0: one pass and one barrier each
#include "tf_selftest.h"
#include "tf_points.h"
#include "vtx_canary.h"
static int tf_batch_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_TF_BATCH"); v = !(e && !strcmp(e, "0")); } return v; }
static int tf_hits(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_ntfw; i++) if (lo < g_tfw[i].hi && g_tfw[i].lo < hi) return 1; return 0; }
static void tf_pass_begin(int keep) {
    if (P.open && P_tf && keep && tf_batch_on()) return;
    pass_end(); if (g_nlc) lc_out_all(); memset(&P, 0, sizeof P); P_tf = 1; g_bar_clean = g_after_pass = 0;   // (a kept depth clear is written out first: a capture's vertex shader may read the image)
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {1, 1}}, .layerCount = 1 };
    // a driver that drops the work of a pass with nothing attached (tf_selftest.h decides, g_tf_att): a 1x1 colour attachment it never writes (the pipelines get its
    //   format through P.cfmt like any pass's: pso_get's key)
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_tf_view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE };
    uint32_t cf[1] = { TF_ATT_FMT };
    if (g_tf_att > 0 && g_tf_view) { ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca; P.cfmt[0] = TF_ATT_FMT; P.ncolor = 1; }
    rend_begin(CB(), &ri, cf, 0, 1, 0); P.open = 1; P.tw = P.th = 1; P.samples = 1;
}
static uint64_t g_stat_begins;   // vkCmdBeginRendering calls (the present line's statistics)
static uint64_t g_ro_passes, g_ro_wrong;   // depth / stencil attachments begun with store op NONE, of them written after all (the pass restarted)
// A write to the depth / stencil attachment of the open pass (a draw that writes, a clear): noted, and when the pass began with store op NONE for it the pass restarts storing
static int pass_begin(void);
static void pass_writes(int depth, int stencil) {
    if (!P.open || P_tf) return;
    if ((depth && P.dnone) || (stencil && P.snone)) {
        if (P.dnone) sk_ro_note(P.dkey, 1, depth != 0); if (P.snone) sk_ro_note(P.skey, 1, stencil != 0);
        P.hd = P.hs = 0;   // (noted above)
        g_ro_wrong++; pass_end(); pass_begin();   // the image as it was (NONE kept it); the new pass loads it and stores
    }
    if (depth) P.dwrote = 1; if (stencil) P.swrote = 1;
}
// Target changes that need no new pass (DM_VK_PASS_DEFER, on by default on Adreno; =0 / =1 decides on any GPU): the game sets target A, then B with nothing drawn, then A again, and every change
//   ended the pass, so the driver set up a new one for the same attachments (about 4 of Legends: Arceus' ~55 a frame; on Adreno each costs the driver an allocation
//   in vkCmdBeginRendering). A change while a pass is open now only records the new targets (g_rt_stale); the next draw or clear goes on in the open pass when they
//   are the ones it was begun with (g_pass_resumed), and ends it as before when they are not. Pending clears belong to the old targets: then the change ends the pass at once
static p_set_rt g_pass_rt; static uint64_t g_pass_resumed;
static int pass_defer_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PASS_DEFER"); v = quirk_note(Q_PASS_DEFER, e ? atoi(e) != 0 : quirk_default(Q_PASS_DEFER, &g_gpuid), e != NULL);   // (on by default on Adreno, Qualcomm 0x5143; Mali and the rest as before)
    if (v) VLOG("target changes that need no new pass keep it open (DM_VK_PASS_DEFER)"); } return v; }
static int rt_same(const p_set_rt *a, const p_set_rt *b);
static void sub_split_at_target(void);
static void rt_resolve(void) {   // a deferred target change: the open pass goes on if the targets came back to its own, else it ends as a target change ends it
    if (!g_rt_stale) return;
    g_rt_stale = 0; if (!P.open) return;
    if (rt_same(&g_pass_rt, &R.rt)) { g_pass_resumed++; return; }
    g_brk[0]++; g_pass_last_by_rt = 1; sub_split_at_target(); pass_end();
}
// DM_VK_DEPTH_DISCARD=1 (a test; Galaxy-Android stores off-screen depth with DONT_CARE so a tiled GPU, Mali above all, never writes it back from tile memory): a depth /
// stencil image that nothing reads after its passes is not stored. Read means a later pass loading it, a shader sampling it, or the game copying it; until the image
// has been a pass's depth for 120 frames with no such read it is stored as before, and once anything reads it it is stored for good. A read that first comes later
// than that sees one frame of undefined depth (then the image is stored again): a test, off by default
static int depth_discard_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_DEPTH_DISCARD"); v = e && atoi(e) > 0; if (v) VLOG("passes: depth that nothing reads afterwards is left in tile memory (DM_VK_DEPTH_DISCARD, a test)"); }
    return v;
}
static void depth_read(vtex *x) { if (x && (x->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) x->dkeep = 1; }
static VkAttachmentStoreOp depth_store(vtex *x, int loads) {
    if (!depth_discard_on()) return VK_ATTACHMENT_STORE_OP_STORE;
    if (loads) x->dkeep = 1;   // (this pass reads what the last one stored)
    if (!x->dfirst) x->dfirst = g_frame_no + 1;
    if (x->dkeep || g_frame_no + 1 - x->dfirst < 120) return VK_ATTACHMENT_STORE_OP_STORE;
    static int said; if (!said++) VLOG("passes: a %ux%u depth image (format %d) is no longer stored: nothing read it for 120 frames", x->w, x->h, (int)x->fmt);
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
}
// Lazy clears (perf-1124 GPU fix B). The games set a depth-only target, clear it and set the real targets: the clear was kept (R.clear_*), but the target change began
//   a pass just for it (pass_flush) that cleared and stored the depth image, and the next pass on it loaded what had just been stored (Sword / Shield 8-10 such passes a
//   frame at 1080p, two of them cleared twice with nothing between; Legends: Arceus 12: 59 / 81 MB stored and the same loaded back a frame at 1080p). Such a clear is
//   now kept beside its image (g_lc: the image, its attachment view, size and layers, the targets and the values; lc_defer) and no pass begins for it:
//   - the next game pass that attaches that view as its depth / stencil over the whole of it takes it as its load op CLEAR (lc_settle, pass_begin): what the clear's
//     own pass and the load after it gave;
//   - a second clear of the same view before then replaces it, aspect by aspect (as the second pass overwrote the first);
//   - anything else first writes it out as the pass it was (lc_out: the same targets and values, through pass_begin): any other game pass (lc_settle: it may sample the
//     image, or be on part of it), a capture pass (tf_pass_begin, tfp_pass_begin), a copy, blit or readback into a buffer from or into any image (copy_b2t, copy_t2t,
//     and zcopy_draw / the blits inside it), a texture clear (clear_tex), a dispatch (dispatch), the submission's end (mtl_sub_commit: the CPU's reads, screenshots,
//     the present and every later submission come after it), and as a last net any read of the image on the replay thread with no pass open (sk_read: sampled,
//     copied, shown, read by the CPU: lc_read).
//   A pass that took a clear by its load op and then samples that image (a draw binding it as a texture: sk_read) is ended and begun again, loading, before that draw
//   is recorded (draw_, g_lc_hazard): memory then holds the clear, as the clear-only pass left it. Only depth / stencil clears with no colour target set and an image
//   the CPU never fills are kept; colour clears and everything else are written as before. Two NVN textures over the same memory share one vtex (tex_res: g_alias),
//   so the kept clear is found through either. Only the replay's own thread keeps and writes them (g_lc_thr). DM_VK_LAZY_CLEAR=0: every clear-only pass as before
enum { LC_N = 8 };
typedef struct { vtex *x; VkImageView v; uint32_t w, h, layers; p_set_rt rt; uint8_t d, s; float cd; uint32_t cs, lv, ly; } lc_ent;   // (lv / ly: the view's level and first layer)
static lc_ent g_lc[LC_N], g_lc_cur;   // kept clears (g_nlc of them); the one the pass being begun takes
static int g_lc_busy; static pthread_t g_lc_thr;   // (a kept clear being written out: its pass_begin takes and writes nothing; the thread that keeps them)
static uint64_t g_lc_kept, g_lc_folded, g_lc_replaced, g_lc_written, g_lc_redo, g_lc_net;   // the present line: kept, taken by a load op, replaced, written out as before, passes begun again, by the read net
static int lc_settle(void);
static int pass_begin(void) {
    tf_end();
    if (P.open && g_rt_stale) rt_resolve();
    if (P.open) return 1;
    int lc_take = g_nlc && !g_lc_busy ? lc_settle() : 0;   // (lazy clears: the one kept clear this pass takes by its load op; the others written out before it)
    if (!R.has_rt) { RED_ONCE(21, "draw with no render target set"); return 0; }
    static int prof = -1; if (prof < 0) prof = getenv("DM_VK_PROF") != NULL; uint64_t pt0 = prof ? mono_ns() : 0;   // (DM_VK_PROF: g_pb_ns)
    VkRenderingAttachmentInfo ca[8]; VkRenderingAttachmentInfo da = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO }, sa = da;
    memset(&P, 0, sizeof P); uint32_t tw = 16384, th = 16384, nlay = 0; int any = 0, ncol = 0;   // nlay: the fewest layers of the attachments (a layered pass over a 3D target)
    uint64_t dkey = 0, skey = 0; int dnone = 0, snone = 0;
    fgr_attin fga[10]; int fgn = 0;   // (the frame recorder, while it records)
    vtex *hp_c0 = NULL, *hp_z = NULL; int hp_zclr = 0; float hp_zcd = 0; vtex *xf_x = NULL; int xf_clr = 0;   // (HUD-free: the pass's first colour target; depth guidance: its depth target, cleared to what)
    for (int i = 0; i < 8; i++) { ca[i] = (VkRenderingAttachmentInfo){ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO }; }
    if (sk_on() || ro_free()) {   // this pass's key for the store learning (sk_attach) and sk_ro: its render targets and how many times they were set before in this frame
        static struct { uint64_t sig; uint32_t n; } occ[256]; static uint64_t occ_frame = ~0ull;
        if (occ_frame != g_frame_no) { memset(occ, 0, sizeof occ); occ_frame = g_frame_no; }
        uint64_t hh = 1469598103934665603ull; const p_set_rt *r = &R.rt;
        #define SKH(v) (hh = (hh ^ (uint64_t)(v)) * 1099511628211ull)
        SKH(r->n); for (int i = 0; i < r->n && i < 8; i++) { SKH(r->color[i] ? r->color[i]->serial : 0); SKH(r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0); SKH(r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0); }
        SKH(r->depth ? r->depth->serial : 0); SKH(r->has_dview && r->dview.set_levels ? r->dview.base_level : 0); SKH(r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0);
        #undef SKH
        uint32_t o = 0; for (uint32_t q = (uint32_t)hh & 255, k = 0; k < 256; q = (q + 1) & 255, k++) { if (!occ[q].sig) { occ[q].sig = hh; o = occ[q].n++; break; } if (occ[q].sig == hh) { o = occ[q].n++; break; } }
        g_sk_sig = hh ^ (uint64_t)o * 0xC2B2AE3D27D4EB4Full;
    }
    // A target whose texture has no format (NVN format 0, never built: Sword / Shield bind one with a 1x1 viewport, 60 of 92 players' logs) is a slot switched off,
    //   as on Maxwell: no attachment there, and a pass with nothing else attached runs without attachments (its draws still write buffers and images) instead of
    //   being dropped ("no render target could be attached")
    for (int i = 0; i < R.rt.n && i < 8; i++) {
        if (!R.rt.color[i] || !R.rt.color[i]->d.format || rt_dup_slot(&R.rt, i)) continue;
        vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i], &x, &w, &h);
        if (!v || !(x->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) continue;
        if (!nlay || g_att_layers < nlay) nlay = g_att_layers;
        ca[i].imageView = v; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].storeOp = sk_on() && sk_attach(x, R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0, R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0, 1, i, R.clear_c[i])
                        ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        ca[i].loadOp = R.clear_c[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].clearValue.color = R.cc[i];
        VkFormat f = x->fmt; if (R.rt.has_cview[i] && R.rt.cview[i].set_format) { uint32_t vf = 0; uint8_t s[4]; if (nvn_fmt_to_vk(R.rt.cview[i].format, &vf, s) && vf) f = (VkFormat)vf; }
        f = b10_view_fmt(x, f, B10V_ATTACH, NULL); if (b10_swapped_tex(x) && f == x->fmt) P.b10 |= 1u << i;   // (DM_VK_B10_AS_RGBA16: the format att_view made the view in; the target has no alpha)
        P.cfmt[i] = (uint32_t)f; P.samples = x->samples; if (w < tw) tw = w; if (h < th) th = h; any = 1; ncol = i + 1;
        if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || (P.b10 >> i & 1)) mw_pass_target(x, R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0);   // (the white-screen check's lit target)
        x->hp_stamp = g_hp_seq + 1; if (!hp_c0) hp_c0 = x;
        if (atomic_load(&g_fgr.on) && fgn < 10) fga[fgn++] = (fgr_attin){ x, x->w, x->h, (uint32_t)f, (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0),
            (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0), (uint8_t)i, (uint8_t)(ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR), rp_store_code(ca[i].storeOp) };
        if (TD_TRACE) VLOG("[trace] pass colour %d ser=%llu %ux%u fmt %d %s", i, (unsigned long long)R.rt.color[i]->serial, w, h, (int)x->fmt, R.clear_c[i] ? "clear" : "load");
        td_note(x, R.rt.color[i]->serial, R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0, R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0);
    }
    if (R.rt.depth && R.rt.depth->d.format) {
        vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.depth, &R.rt.dview, R.rt.has_dview, &x, &w, &h);
        if (v && (x->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            if (lc_take && x == g_lc_cur.x && v == g_lc_cur.v) {   // (lazy clears: the kept clear of this view, taken by the load op; a clear of this pass's own is newer and wins)
                if (g_lc_cur.d && !R.clear_d) { R.clear_d = 1; R.cd = g_lc_cur.cd; }
                if (g_lc_cur.s && !R.clear_s) { R.clear_s = 1; R.cs = g_lc_cur.cs; }
                g_lc_open = x; g_lc_folded++; lc_take = 0; vtex_unref(g_lc_cur.x); g_lc_cur.x = NULL; }
            if (!nlay || g_att_layers < nlay) nlay = g_att_layers;
            VkAttachmentStoreOp dso = depth_store(x, ((x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !R.clear_d) || ((x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && !R.clear_s));
            uint32_t dl = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0, dy = R.rt.has_dview && R.rt.dview.set_layers ? R.rt.dview.min_layer : 0;
            if (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da.imageView = v; da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                da.storeOp = sk_on() && sk_attach(x, dl, dy, 2, 8, R.clear_d) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : dso; dkey = sk_on() ? g_sk_last : sk_key(8);
                if (da.storeOp == VK_ATTACHMENT_STORE_OP_STORE && g_store_none && sk_ro(dkey, R.clear_d)) { da.storeOp = VK_ATTACHMENT_STORE_OP_NONE; dnone = 1; }
                da.loadOp = R.clear_d ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; da.clearValue.depthStencil.depth = R.cd; P.hd = 1; P.dfmt = (uint32_t)x->fmt;
                hp_z = x; hp_zclr = R.clear_d; hp_zcd = R.cd; }
            if (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.imageView = v; sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                sa.storeOp = sk_on() && sk_attach(x, dl, dy, 3, 9, R.clear_s) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : dso; skey = sk_on() ? g_sk_last : sk_key(9);
                if (sa.storeOp == VK_ATTACHMENT_STORE_OP_STORE && g_store_none && sk_ro(skey, R.clear_s)) { sa.storeOp = VK_ATTACHMENT_STORE_OP_NONE; snone = 1; }
                sa.loadOp = R.clear_s ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; sa.clearValue.depthStencil.stencil = R.cs; P.hs = 1; P.sfmt = (uint32_t)x->fmt; }
            P.samples = x->samples; if (w < tw) tw = w; if (h < th) th = h; any = 1; x->hp_stamp = g_hp_seq + 1;
            xf_x = x; xf_clr = (!(x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || (P.hd && R.clear_d)) && (!(x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) || (P.hs && R.clear_s));
            if (atomic_load(&g_fgr.on) && fgn < 9) { if (P.hd) fga[fgn++] = (fgr_attin){ x, x->w, x->h, (uint32_t)x->fmt, (uint16_t)dl, (uint16_t)dy, 8, (uint8_t)(da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR), rp_store_code(da.storeOp) };
                if (P.hs) fga[fgn++] = (fgr_attin){ x, x->w, x->h, (uint32_t)x->fmt, (uint16_t)dl, (uint16_t)dy, 9, (uint8_t)(sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR), rp_store_code(sa.storeOp) }; }
        }
    }
    if (lc_take) {   // (lc_settle and the block above see the same targets, so this does not happen; if it did, the clear is kept and written out at the next chance)
        RED_ONCE(164, "a kept depth clear was meant for this pass but its depth view was not attached: kept for the next pass");
        if (g_nlc < LC_N) g_lc[g_nlc++] = g_lc_cur; else vtex_unref(g_lc_cur.x);
        g_lc_cur.x = NULL; lc_take = 0; }
    if (!any) {   // no target attached: an attachment-less pass at the viewport's size (nvn_mtl.m bare_dims), or nothing when targets were set but none could be attached
        int bare = !(R.rt.depth && R.rt.depth->d.format); for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i] && R.rt.color[i]->d.format) bare = 0;   // (format 0: off)
        if (!bare) {
            static int no_target_reports;
            if (no_target_reports < 32) {
                no_target_reports++;
                VLOG("no render target could be attached: colors %d, depth %s, viewport %.0fx%.0f",
                     R.rt.n, R.rt.depth ? "set" : "unset", S.has_vp ? S.vp[0][2] : 0, S.has_vp ? S.vp[0][3] : 0);
            }
            RED_ONCE(22, "no render target could be attached");
            return 0;
        }
        double x1 = S.has_vp ? fmax(S.vp[0][0], S.vp[0][0] + S.vp[0][2]) : R.has_sc ? (double)R.sc.x + R.sc.w : 1, y1 = S.has_vp ? fmax(S.vp[0][1], S.vp[0][1] + S.vp[0][3]) : R.has_sc ? (double)R.sc.y + R.sc.h : 1;
        tw = (uint32_t)fmin(fmax(ceil(x1), 1), 16384); th = (uint32_t)fmin(fmax(ceil(y1), 1), 16384); P.samples = 1;
    }
    if (!P.samples) P.samples = 1;
    P.tw = tw; P.th = th; P.ncolor = ncol; P.layers = nlay ? nlay : 1;
    P.dkey = dkey; P.skey = skey; P.dnone = dnone; P.snone = snone; P.dwrote = R.clear_d; P.swrote = R.clear_s; g_ro_passes += dnone + snone;
    uint64_t pt1 = prof ? mono_ns() : 0;
    // The game's own barrier, by its bits: ORDER_FRAGMENTS (0x2) / _TILED (0x4) and the cache invalidations (texture 0x10, shader 0x20, descriptors 0x40, ZCull 0x80) ask only that
    //   the earlier passes' fragments land before this pass's fragments read them, which the pass-to-pass barrier gives (the vertex work still overlaps); ORDER_PRIMITIVES (0x1)
    //   and ORDER_INDIRECT_DATA (0x8), results read as vertices, indices, indirect arguments or by a vertex shader, keep the full one. Every game barrier was a full one before
    //   (about 58 a frame in Shield's field on a Galaxy S25+). DM_VK_BARRIER_BITS=0: as before; DM_AB=<frames>:barbits alternates the two in one run
    int narrow_ok = g_game_bar && !(g_game_bits & 0x9) && barrier_bits_on();
    if (g_bar_clean && barrier_relaxed()) g_bar_none++;
    else if (g_after_pass && barrier_relaxed() && (!g_game_bar || narrow_ok)) { barrier_pass(CB()); if (g_game_bar) g_bar_bits_narrow++; else g_bar_narrow++;
        if (!g_game_bar && CB() == g_gap_cb && g_gap_work) g_gap_work--; }   // (phase 3: only the order between passes, which one pass keeps by itself; the game asked for none)
    else { barrier_all(CB()); g_bar_full++; if (g_game_bar) { g_bar_full_game++; for (int b = 0; b < 8; b++) if (g_game_bits >> b & 1) g_bar_bits_hist[b]++; } else g_bar_full_work++; }
    g_game_bar = 0; g_game_bits = 0; g_bar_clean = g_after_pass = 0;
    uint64_t pt2 = prof ? mono_ns() : 0;
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {tw, th}}, .layerCount = nlay ? nlay : 1, .colorAttachmentCount = (uint32_t)ncol, .pColorAttachments = ca,
                           .pDepthAttachment = P.hd ? &da : NULL, .pStencilAttachment = P.hs ? &sa : NULL };
    if (xf_x && xf_on()) {   // (stage 3: its frame-end store pending from an earlier frame resolves: cleared whole, or loaded)
        pthread_mutex_lock(&g_fgrmu);
        if (xf_attach(&xf_x->xf, xf_clr && tw >= xf_x->w && th >= xf_x->h && (nlay ? nlay : 1) == 1)) { g_xf_bad++; RED_ONCE(61, "a depth store left out at a frame's end was loaded by a later pass (%ux%u, format %d): stored again for good", xf_x->w, xf_x->h, (int)xf_x->fmt); }
        pthread_mutex_unlock(&g_fgrmu); }
    if (!P_tf) { hp_pass_begin(CB(), hp_c0); g_hpc.z = hp_z; g_hpc.zclr = (uint8_t)hp_zclr; g_hpc.zcd = hp_zcd; }   // (HUD-free frame generation: the scene copy before the first HUD pass)
    if (!P_tf) g_seg_fgp = fgr_pass_begin(CB(), tw, th, nlay ? nlay : 1, P.samples, fga, fgn);   // (the held frame decides this pass's stores by it)   // (the transform feedback capture passes are ours, not the game's)
    g_ra_next = !P_tf;   // (a held frame: the game's pass gets the render area its draws reach, seg_area)
    rend_begin(CB(), &ri, P.cfmt, P.hd ? P.dfmt : P.hs ? P.sfmt : 0, P.samples, 0); g_stat_begins++;
    if (prof) { uint64_t pt3 = mono_ns(); g_pb_ns[0] += pt1 - pt0; g_pb_ns[1] += pt2 - pt1; g_pb_ns[2] += pt3 - pt2; }
    {   // a pass onto exactly the attachments of the one a target change just ended (the same views, size and layers): one that could have gone on (g_pass_samev)
        VkImageView v[9] = { 0 }; for (int i = 0; i < ncol; i++) v[i] = ca[i].imageView; v[8] = P.hd ? da.imageView : P.hs ? sa.imageView : VK_NULL_HANDLE;
        if (g_pass_last_by_rt && !memcmp(v, g_pass_lastv, sizeof v) && tw == g_pass_lastwh[0] && th == g_pass_lastwh[1] && P.layers == g_pass_lastwh[2]) g_pass_samev++;
        memcpy(g_pass_lastv, v, sizeof v); g_pass_lastwh[0] = tw; g_pass_lastwh[1] = th; g_pass_lastwh[2] = P.layers; g_pass_last_by_rt = 0;
    }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    P.open = 1; g_pass_rt = R.rt;
    return 1;
}
static int getenv_nomerge(void) { static int v = -1; if (v < 0) v = getenv("DM_VK_NO_PASS_MERGE") != NULL; return v; }   // A/B switch
// Pending clears of attachments the targets have (perf-1124 GPU fix A). OP_CLEAR_DS notes a depth / stencil clear whether or not the targets have a depth image, and
//   pass_flush began "a pass just for them" all the same: with no depth attached that pass loaded and stored the colour targets and did nothing (Legends: Arceus, 9
//   such passes a frame on a Galaxy S25+, 5 at 1080p: 48 MB loaded and 48 MB stored for nothing). A clear now counts only for an attachment pass_begin attaches: a
//   colour slot holding a texture of a format, the depth / stencil image's aspect it clears (a stencil clear of a depth-only image did nothing there either).
//   DM_VK_EMPTY_PASS=1: every noted clear counts, as before
static int empty_pass_old(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_EMPTY_PASS"); v = e && atoi(e) == 1;
    if (v) VLOG("clears with nothing to clear begin a pass of their own, as before (DM_VK_EMPTY_PASS=1)"); } return v; }
static uint64_t g_empty_dropped;   // passes of clears with nothing to clear left out (the present line)
static int pend_raw(void) { int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i]; return pend; }
static int pend_clears(void) {
    if (empty_pass_old()) return pend_raw();
    for (int i = 0; i < 8; i++) if (R.clear_c[i] && i < R.rt.n && R.rt.color[i] && R.rt.color[i]->d.format) return 1;
    if (!(R.clear_d || R.clear_s) || !R.rt.depth || !R.rt.depth->d.format) return 0;
    vtex *x = tex_res(R.rt.depth); if (!x || !x->img) return 1;   // (no image: as before, pass_begin reports it)
    return (R.clear_d && (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) || (R.clear_s && (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT));
}
// ---- lazy clears (see above pass_begin) ----
static int lc_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_LAZY_CLEAR"); v = !(e && !strcmp(e, "0"));
    VLOG("depth clears with no draw after them: %s", v ? "taken by the next pass on the image as its load op (DM_VK_LAZY_CLEAR=0: a pass each, as before)" : "a pass each (DM_VK_LAZY_CLEAR=0)"); } return v; }
static void lc_out(int i) {   // kept clear i written out as the pass it was, now
    lc_ent e = g_lc[i]; g_lc[i] = g_lc[--g_nlc];
    vtex *x = NULL; uint32_t w = 0, h = 0; VkImageView v = g_sub && !P.open && e.rt.depth ? att_view(e.rt.depth, &e.rt.dview, e.rt.has_dview, &x, &w, &h) : VK_NULL_HANDLE;
    if (v && x == e.x && v == e.v) {   // (the targets still name the image: its pass as it was)
        __typeof__(R) sv = R; int busy = g_lc_busy; g_lc_busy = 1;
        R.rt = e.rt; R.has_rt = 1; memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = e.d; R.cd = e.cd; R.clear_s = e.s; R.cs = e.cs;
        if (pass_begin()) pass_end();
        g_lc_busy = busy; R = sv; g_lc_written++;
    } else if (g_sub && !P.open && (e.x->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {   // (the texture object names another image by now: the kept one cleared directly, the same values)
        g_pass_last_by_rt = 0; barrier_all(CB());   // (as outside_pass: no pass is open here)
        VkClearDepthStencilValue cv = { e.cd, e.cs }; VkImageSubresourceRange rg = { (e.d ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) | (e.s ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u), e.lv, 1, e.ly, e.layers };
        vkCmdClearDepthStencilImage(CB(), e.x->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rg); barrier_all(CB()); g_lc_written++;
    } else RED_ONCE(165, "a kept depth clear could not be written out (%s): lost", !g_sub ? "no submission" : P.open ? "a pass is open" : "its image takes no transfers");
    vtex_unref(e.x);
}
static void lc_out_all(void) { while (g_nlc) lc_out(g_nlc - 1); }
// At a target change with clears pending and no pass open (pass_flush): kept instead of a pass when they are depth / stencil clears of a depth-only target
static int lc_defer(void) {
    if (!lc_on() || g_lc_busy || !g_sub || P.open || !R.has_rt || !(R.clear_d || R.clear_s) || atomic_load(&g_cap) == 2) return 0;   // (a frame capture lists every pass)
    for (int i = 0; i < 8; i++) if (R.clear_c[i]) return 0;
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i] && R.rt.color[i]->d.format) return 0;   // (a colour target: the pass's area could be smaller than the depth view)
    if (!R.rt.depth || !R.rt.depth->d.format) return 0;
    vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.depth, &R.rt.dview, R.rt.has_dview, &x, &w, &h); uint32_t lay = g_att_layers;
    if (!v || !(x->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) || x->pkg) return 0;
    uint8_t d = R.clear_d && (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), s = R.clear_s && (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    if (!d && !s) return 0;
    for (int i = 0; i < g_nlc; i++) if (g_lc[i].x == x) {
        if (g_lc[i].v == v && g_lc[i].w == w && g_lc[i].h == h && g_lc[i].layers == lay) {   // the same view cleared again: it replaces the first, aspect by aspect
            if (d) { g_lc[i].d = 1; g_lc[i].cd = R.cd; } if (s) { g_lc[i].s = 1; g_lc[i].cs = R.cs; } g_lc[i].rt = R.rt; g_lc_replaced++; return 1; }
        lc_out(i); break; }   // (another part of the same image: the first is written out, in order)
    if (g_nlc == LC_N) lc_out(0);
    uint32_t lv = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0, ly = R.rt.has_dview && R.rt.dview.set_layers ? R.rt.dview.min_layer : 0;
    vtex_ref(x); g_lc[g_nlc++] = (lc_ent){ x, v, w, h, lay, R.rt, d, s, R.cd, R.cs, lv, ly }; g_lc_thr = pthread_self(); g_lc_kept++;
    return 1;
}
// pass_begin, before anything else: whether the pass about to begin takes a kept clear by its load op (its depth view the kept one, over the whole of it: no colour
//   target smaller or with fewer layers, as the pass's area is its smallest attachment's; not the scene depth HUD-free frame generation copies before the pass begins,
//   hp_pass_begin); every other kept clear is written out first
static int lc_settle(void) {
    int take = -1;
    if (R.has_rt && R.rt.depth && R.rt.depth->d.format && atomic_load(&g_cap) != 2) {
        vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.depth, &R.rt.dview, R.rt.has_dview, &x, &w, &h); uint32_t lay = g_att_layers;
        if (v && (x->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && x != g_hp_Z)
            for (int i = 0; i < g_nlc; i++) if (g_lc[i].x == x && g_lc[i].v == v && g_lc[i].w == w && g_lc[i].h == h && g_lc[i].layers == lay) take = i;
        for (int i = 0; i < R.rt.n && i < 8 && take >= 0; i++) {
            if (!R.rt.color[i] || !R.rt.color[i]->d.format || rt_dup_slot(&R.rt, i)) continue;
            vtex *cx; uint32_t cw, ch; VkImageView cv = att_view(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i], &cx, &cw, &ch);
            if (!cv || !(cx->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) continue;
            if (cw < w || ch < h || g_att_layers < lay) take = -1; }
    }
    if (take >= 0) { g_lc_cur = g_lc[take]; g_lc[take] = g_lc[--g_nlc]; }
    lc_out_all();
    return take >= 0;
}
// sk_read: a read of x. On the replay thread: a kept clear of x is written out (no pass open), or the draw being recorded begins its pass again (draw_)
static void lc_read(vtex *x) {
    if (!pthread_equal(pthread_self(), g_lc_thr)) return;   // (other threads read after a submission ended: mtl_sub_commit wrote every kept clear out)
    if (x == g_lc_open && P.open && !P_tf) { g_lc_hazard = 1; return; }
    if (g_lc_busy) return;
    for (int i = 0; i < g_nlc; i++) if (g_lc[i].x == x) { if (P.open) g_lc_hazard = 1; else { g_lc_net++; lc_out(i); } return; }
}
static void pass_flush(void) {   // pending clears with no draw after them: a pass just for them (or none: kept for the next pass, lc_defer)
    int pend = pend_clears();
    if (!pend && pend_raw()) {   // (nothing they could clear: no pass, and they do not carry over to the next targets, which a pass begun for them reset)
        if (!P.open && g_sub) g_empty_dropped++;
        memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; }
    if (pend && lc_defer()) { memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; pend = 0; }
    if (pend && !P.open && g_sub) pass_begin();
    pass_end();
}
static void outside_pass(void) { if (P.open) g_brk_other++; g_pass_last_by_rt = 0; pass_end(); barrier_all(CB()); }   // transfer work: no pass open, ordered after everything before

// ================= Pipelines =================
typedef struct {
    void *sh[5]; uint32_t cfmt[8], dfmt, sfmt; uint32_t chmask; blend_st blend[8];
    struct { uint32_t fmt, off; uint8_t bind, used, inst, pad; uint32_t div; } attr[32];
    uint8_t samples, a2c, poly, clamp, topo, logic_en, logic_op, blend_en, depth_zero, discard, ds_swap, tfpts, yflip, b10, pad[2];   // tfpts: a capture drawn as points into that many targets (tf_points.h); yflip: y negated in the vertex shader (g_yflip, upper-left origin)   // ds_swap: first half of an emulated dual-source blend (draw)
    // b10: colour targets that are R11G11B10F's RGBA16F stand-in (DM_VK_B10_AS_RGBA16; bit i: colour i, cfmt RGBA16F): no alpha written, the target's alpha read as 1.
    //   Only in memory: the lists on disk name such a target R11G11B10F with b10 0 (b10_key_disk / b10_key_here), as every phone records it without the test
} pkey;
static nvn_hmap g_pso = {.klen = sizeof(pkey)}, g_pso_fail = {.klen = sizeof(pkey)};
// Without extended dynamic state (g_eds 0) the state the draws otherwise set per draw is part of the pipeline: cull mode, front face, exact topology, depth test / write /
//   compare, stencil test and ops, and the vertex strides. Kept beside pkey (not in it: pkey is the pipeline list's record format, shared with every other phone), with
//   values that change nothing folded together (no depth test: no writes, compare ALWAYS; no stencil test: ops KEEP / ALWAYS; unused streams: stride 0)
typedef struct { uint8_t cull, front, topo, dt, dw, dcmp, st, pad; uint8_t sop[2][4]; uint32_t stride[16]; } sst;   // sop[face]: fail, pass, depth fail, compare
typedef struct { pkey k; sst s; } pkey_s;
// Points and lines are never culled (Vulkan culls polygons only; the dynamic path sends NONE for them, draw_), so their cull mode is folded to NONE in the key: fewer
//   pipelines to build without extended dynamic state (fable-cpu fix 6). The prewarm folds the recorded lists the same way. DM_VK_SST_FOLD=0: kept as the game set it
static int sst_fold_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SST_FOLD"); v = !(e && !strcmp(e, "0")); } return v; }
static int topo_class(VkPrimitiveTopology t);
static void sst_fold(sst *s) { if (sst_fold_on() && topo_class((VkPrimitiveTopology)s->topo) < 2) s->cull = VK_CULL_MODE_NONE; }
static nvn_hmap g_pso_s = {.klen = sizeof(pkey_s)}, g_pso_s_fail = {.klen = sizeof(pkey_s)}, g_pso_s_rec = {.klen = sizeof(pkey)};
static VkPipelineCache g_pcache;
static pthread_mutex_t g_psomu = PTHREAD_MUTEX_INITIALIZER;   // g_pso / g_pso_fail, shader variants, the record file: the game's thread and the prewarm thread
static atomic_int g_pso_new;                                  // pipelines built since the cache file was last written
static atomic_int g_pso_ingame;                               // pipelines the game's draws had to build (a prewarm miss: a possible hitch)
static VkPrimitiveTopology topo_of(int prim) {
    switch (prim) { case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; }
}
static int topo_class(VkPrimitiveTopology t) { return t == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ? 0 : t <= VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ? 1 : 2; }
static VkBlendFactor bfac(int f) {
    switch (f) { case 1: return VK_BLEND_FACTOR_ZERO; case 2: return VK_BLEND_FACTOR_ONE; case 3: return VK_BLEND_FACTOR_SRC_COLOR; case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA; case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 7: return VK_BLEND_FACTOR_DST_ALPHA; case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR; case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA; }
    return VK_BLEND_FACTOR_ONE;
}
// The blend factor on this device: without dual-source blending a second output's factor is the first output's (g_dualsrc: mtl_init)
static VkBlendFactor bfac_dev(int f) {
    VkBlendFactor v = bfac(f);
    if (g_dualsrc || v < VK_BLEND_FACTOR_SRC1_COLOR || v > VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA) return v;
    RED_ONCE(111, "the game blends with its second shader output (blend factor %d) and the driver has no dual-source blending: the first output's instead (DM_VK_DUALSRC=keep to compare)", (int)v);
    return v == VK_BLEND_FACTOR_SRC1_COLOR ? VK_BLEND_FACTOR_SRC_COLOR : v == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ? VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR
         : v == VK_BLEND_FACTOR_SRC1_ALPHA ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
}
static VkBlendOp bop(int e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
// vertex format for a shader input of class ct (0 float, 1 uint, 2 sint): an integer format of the other signedness is read with the same width and the input's sign
static VkFormat vfmt_for(int nvn_fmt, int ct) {
    int fc; VkFormat f = (VkFormat)nvn_vfmt_to_vk(nvn_fmt, &fc);
    if (!f) return VK_FORMAT_UNDEFINED;
    // (Eden's notes: Qualcomm's drivers read USCALED / SSCALED vertex formats wrong; noted once with whether the driver lists them for vertex buffers)
    if (nvn_fmt >= 0x51 && nvn_fmt <= 0x68 && fc == 0) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, f, &fp);
        RED_ONCE(61, "vertex format %#x is integers read as floats (Vulkan format %d, scaled): the driver %s it for vertex buffers", nvn_fmt, (int)f,
                 fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT ? "lists" : "does NOT list"); }
    if (ct == fc || (ct == 0 && fc == 0)) return f;
    static const VkFormat U[] = { VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT };
    static const VkFormat I[] = { VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT };
    static const VkFormat F32[] = { VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT };
    for (int i = 0; i < 12; i++) {
        if (U[i] == f || I[i] == f) { if (ct == 1) return U[i]; if (ct == 2) return I[i]; if (i >= 8) return F32[i - 8]; RED_ONCE(23, "vertex format %#x (8/16-bit integer) feeds a float input", nvn_fmt); return f; }
    }
    // a float-type format into an integer input: the same bytes read as integers of the same layout (Maxwell gives the converted float's bits; noted once)
    RED_ONCE(24, "vertex format %#x feeds an integer input: raw bytes read as integers", nvn_fmt);
    if (f == VK_FORMAT_R32_SFLOAT || f == VK_FORMAT_R32G32_SFLOAT || f == VK_FORMAT_R32G32B32_SFLOAT || f == VK_FORMAT_R32G32B32A32_SFLOAT) { int n = f == VK_FORMAT_R32_SFLOAT ? 0 : f == VK_FORMAT_R32G32_SFLOAT ? 1 : f == VK_FORMAT_R32G32B32_SFLOAT ? 2 : 3; return ct == 1 ? U[8 + n] : I[8 + n]; }
    return f;
}
// The fragment module for a pipeline: on the Switch a varying the previous stage never writes reads undefined values; Vulkan (and Metal under MoltenVK) refuse the pipeline,
// so such inputs become private variables holding the hardware's (0, 0, 0, 1) in a variant of the fragment shader (made once per shader and set of missing locations)
// The fragment module for a pipeline. On the Switch a varying the previous stage never writes reads (0, 0, 0, 1) and a float output to an integer colour target is written
// as its bits; Vulkan (and Metal under MoltenVK) refuse both, so such a pipeline gets a variant of the fragment shader (made once per shader and combination)
static int g_ds_swap;   // the draw being recorded is the first half of an emulated dual-source blend: its fragment outputs 0 and 1 swapped (draw, pso_get)
// qm: outputs into R11G11B10F's RGBA16F stand-in (DM_VK_B10_AS_RGBA16, pso_make), made what the packed format stores (spv_b10_outputs; locations after the swap)
static VkShaderModule frag_module_(vprog *pr, const uint32_t *cfmt, int sw, uint32_t qm);
static VkShaderModule frag_module(vprog *pr, const uint32_t *cfmt) { return frag_module_(pr, cfmt, 0, 0); }
static VkShaderModule frag_module_(vprog *pr, const uint32_t *cfmt, int sw, uint32_t qm) {
    vshader *fs = pr->sh[1], *pv = pr->sh[2] ? pr->sh[2] : pr->sh[4] ? pr->sh[4] : pr->sh[0];
    uint64_t drop = fs->in_mask & ~(pv ? pv->out_mask : 0); uint32_t um = 0, sm = 0;
    for (int i = 0; i < 8; i++) if (cfmt[i] && fs->out_cls[i] == 0) { int c = fmt_int_class((VkFormat)cfmt[i]); if (c == 1) um |= 1u << i; else if (c == 2) sm |= 1u << i; }
    if ((!drop && !um && !sm && !sw && !qm) || !fs->code) return fs->mod;
    pthread_mutex_lock(&g_psomu);
    for (int i = 0; i < fs->nvar; i++) if (fs->var[i].drop == drop && fs->var[i].um == um && fs->var[i].sm == sm && fs->var[i].sw == sw && fs->var[i].qm == qm) { VkShaderModule m = fs->var[i].mod; pthread_mutex_unlock(&g_psomu); return m; }
    pthread_mutex_unlock(&g_psomu);
    size_t nw = fs->nw; uint32_t *c = NULL, *c2;
    if (drop && (c2 = spv_drop_inputs(fs->code, fs->nw, drop, &nw))) c = c2;
    if ((um || sm) && (c2 = spv_int_outputs(c ? c : fs->code, c ? nw : fs->nw, um, sm, &nw))) { free(c); c = c2; }
    if (sw && (c2 = spv_swap_out01(c ? c : fs->code, c ? nw : fs->nw, &nw))) { free(c); c = c2; }
    if (qm) { if ((c2 = spv_b10_outputs(c ? c : fs->code, c ? nw : fs->nw, qm, &nw))) { free(c); c = c2; b10_note("a draw into the stand-in: its colour made what R11G11B10F stores (negative to 0, rounded, at most 65024 / 64512)", NULL, (int)qm); }
              else RED_ONCE(161, "b10 swap: fragment shader %s: its output into the stand-in (targets %#x) could not be made what R11G11B10F stores: written as it is", fs->key, qm); }
    if (!c) return fs->mod;
    VkShaderModule m = VK_NULL_HANDLE; VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = c };
    VkResult r = vkCreateShaderModule(g_dev, &mci, NULL, &m); free(c);
    if (r) { VLOG("[RED] fragment shader %s: variant failed %d", fs->key, r); return fs->mod; }
    shmod_count(m, (int64_t)nw * 4, 1);
    if (drop && !fs->nrel) VLOG("fragment shader %s reads varyings %#llx that %s never writes: those read (0, 0, 0, 1)", fs->key, (unsigned long long)drop, pv ? pv->key : "-");
    if ((um | sm) && !fs->nrel) VLOG("fragment shader %s: float outputs %#x / %#x written as uint / int bits (integer colour targets)", fs->key, um, sm);
    if (sw && !fs->nrel) { static int said; if (said++ < 16) VLOG("fragment shader %s: a copy with outputs 0 and 1 swapped, for its dual-source blend in two steps", fs->key); }
    pthread_mutex_lock(&g_psomu);
    if (fs->nvar < 8) { fs->var[fs->nvar].drop = drop; fs->var[fs->nvar].um = um; fs->var[fs->nvar].sm = sm; fs->var[fs->nvar].sw = sw; fs->var[fs->nvar].qm = qm; fs->var[fs->nvar].mod = m; fs->var[fs->nvar].nb = (uint32_t)(nw * 4); fs->nvar++; }
    pthread_mutex_unlock(&g_psomu);
    return m;
}
// ---------- Pipeline list and cache (Android): every pipeline the game makes is recorded (DM_VK_PSO_RECORD) as its shaders' keys plus the rest of its key; at start the
// recorded list and the list shipped with the game (DM_VK_PSO_SEED) are built on a background thread before the game asks for them, and the driver's compiled pipelines are
// kept in a pipeline cache file (DM_VK_PCACHE), so a later start compiles nothing it already compiled
typedef struct { uint8_t key[5][8]; pkey k; } pso_rec;   // k.sh zeroed
#define PSO_MAGIC "NXPL"
// the lists hold an R11G11B10F target as the game made it; in memory, with DM_VK_B10_AS_RGBA16, it is the RGBA16F stand-in (pkey.b10): written as the one, read as the other
static void b10_key_disk(pkey *k) { for (int i = 0; i < 8; i++) if (k->b10 >> i & 1) k->cfmt[i] = VK_FORMAT_B10G11R11_UFLOAT_PACK32; k->b10 = 0; }
static void b10_key_here(pkey *k) {
    if (!b10_on()) return;
    for (int i = 0; i < 8; i++) if (k->cfmt[i] == VK_FORMAT_B10G11R11_UFLOAT_PACK32) { k->cfmt[i] = VK_FORMAT_R16G16B16A16_SFLOAT; k->b10 |= (uint8_t)(1u << i); }
}
static FILE *g_psorec;
static void pso_record(const vprog *pr, const pkey *k) {   // caller holds g_psomu
    static int tried; if (!tried++) { const char *e = getenv("DM_VK_PSO_RECORD"); if (e) { g_psorec = fopen(e, "ab"); if (g_psorec && ftell(g_psorec) == 0) { uint32_t v[2] = { 1, (uint32_t)sizeof(pso_rec) }; fwrite(PSO_MAGIC, 1, 4, g_psorec); fwrite(v, 4, 2, g_psorec); } } }
    if (!g_psorec) return;
    pso_rec r; memset(&r, 0, sizeof r); r.k = *k; memset(r.k.sh, 0, sizeof r.k.sh); b10_key_disk(&r.k);
    for (int i = 0; i < 5; i++) if (pr->sh[i]) hexkey(pr->sh[i]->key, r.key[i]);
    fwrite(&r, sizeof r, 1, g_psorec); fflush(g_psorec);
}
// Without extended dynamic state (g_eds 0) a pipeline is also its draw state (sst), which the shared list leaves out: each one is recorded beside it as well
//   (DM_VK_PSO_RECORD + ".sst", this phone's own: the state is how the game drew here) and built from there at the next start (prewarm_static)
typedef struct { pso_rec r; sst s; } pso_rec_s;
#define PSO_MAGIC_S "NXPS"
static FILE *g_psorec_s;
static void pso_record_s(const vprog *pr, const pkey *k, const sst *s) {   // caller holds g_psomu
    static int tried; if (!tried++) { const char *e = getenv("DM_VK_PSO_RECORD"); if (e) { char p[1100]; snprintf(p, sizeof p, "%s.sst", e); g_psorec_s = fopen(p, "ab");
        if (g_psorec_s && ftell(g_psorec_s) == 0) { uint32_t v[2] = { 1, (uint32_t)sizeof(pso_rec_s) }; fwrite(PSO_MAGIC_S, 1, 4, g_psorec_s); fwrite(v, 4, 2, g_psorec_s); } } }
    if (!g_psorec_s) return;
    pso_rec_s r; memset(&r, 0, sizeof r); r.r.k = *k; memset(r.r.k.sh, 0, sizeof r.r.k.sh); r.s = *s; b10_key_disk(&r.r.k);
    for (int i = 0; i < 5; i++) if (pr->sh[i]) hexkey(pr->sh[i]->key, r.r.key[i]);
    fwrite(&r, sizeof r, 1, g_psorec_s); fflush(g_psorec_s);
}
static VkPipeline pso_make(vprog *pr, pkey k, const sst *ss);
// ---- Pipelines kept (Arm's driver, Xclipse, phones under 9 GB: vk_quirks.h Q_PIPE_KEEP): at most pso_cap(), the ones unused longest destroyed. Arm's driver keeps ~0.4 MB of the process's memory
//   and a GPU mapping for each pipeline: 5500 of them were 2.3 GB of malloc and 61k mappings when a Mali-G720 closed after ~30 minutes in another Android port of a Switch game;
//   here the Mali players' apps held 1.7-2.6 GB against 1 GB on an Adreno (Shield builds ~1500 at start). Every 300 presents (pso_evict), over the
//   cap, the pipelines unused for 900 presents go, oldest first, after the GPU is done with them (defer_add); one needed again is built again from the pipeline
//   cache. The record lists get each pipeline once (g_pso_rec*), not again when it comes back. DM_VK_PIPE_KEEP=<n> (0: never)
//   DM_VK_PIPE_LRU (default 1; 0: the old rule): a pipeline counts as used when built, so the prewarm's are not all destroyed at the first check (#215: 1747 of 2467
//   right after the prewarm, then built again one by one in play), and one check destroys at most DM_VK_PIPE_EVICT_STEP (half the cap): pso_lru.h
static nvn_hmap g_pso_used = {.klen = sizeof(VkPipeline)}, g_pso_rec = {.klen = sizeof(pkey)}, g_pso_rec_s = {.klen = sizeof(pkey_s)};
static uint64_t g_pso_evicted;
static int pso_cap(void) {
    static int cap = -2; if (cap > -2) return cap;
    const char *e = getenv("DM_VK_PIPE_KEEP");
    cap = quirk_note(Q_PIPE_KEEP, e ? atoi(e) : quirk_default(Q_PIPE_KEEP, &g_gpuid), e != NULL);   // (the phone's memory: g_gpuid.ram_kb, read with the GPU's description)
    if (cap > 0) VLOG("pipelines kept: at most %d (the least recently used ones destroyed when unused for 30 s)", cap); else cap = 0;
    return cap;
}
static int pso_lru_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PIPE_LRU"); v = !(e && !strcmp(e, "0")); } return v; }
static int pso_evict_step(int cap) {
    static int v = -2; if (v > -2) return v;
    const char *e = getenv("DM_VK_PIPE_EVICT_STEP"); v = !pso_lru_on() ? 0 : e && *e ? atoi(e) : pso_lru_step_default(cap);
    if (v < 0) v = 0;
    if (cap > 0) VLOG("pipelines over the cap: %s", v ? "a pipeline counts as used when built; at most this many destroyed per check (DM_VK_PIPE_LRU, DM_VK_PIPE_EVICT_STEP)" : "all destroyed at once (DM_VK_PIPE_LRU=0)");
    if (v) VLOG("pipelines destroyed per check: at most %d", v);
    return v;
}
// a pipeline just built (caller holds g_psomu): its first use is now (pso_lru.h), so the prewarm's are as recent as the start of the game
static void pso_built_locked(VkPipeline p) { if (p && pso_cap() && pso_lru_on()) nvn_hm_put(&g_pso_used, &p, (void *)(uintptr_t)pso_lru_stamp(g_frame_no, 1)); }
static void pso_used(VkPipeline p) {   // a draw binds p (the cap is on): its last use
    pthread_mutex_lock(&g_psomu); nvn_hm_put(&g_pso_used, &p, (void *)(uintptr_t)(g_frame_no + 1)); pthread_mutex_unlock(&g_psomu);
}
typedef struct { VkPipeline p; int s; uint32_t slot; } pso_old;
static void pso_evict(void) {
    int cap = pso_cap(); if (!cap) return;
    int step = pso_evict_step(cap);
    pthread_mutex_lock(&g_psomu);
    uint32_t live = g_pso.n + g_pso_s.n;
    if (live <= (uint32_t)cap) { pthread_mutex_unlock(&g_psomu); return; }
    pso_old *all = malloc(sizeof *all * live), *c = malloc(sizeof *c * live); pso_lru_ent *le = malloc(sizeof *le * live); uint32_t n = 0, nc = 0;
    for (int which = 0; which < 2 && all && c && le; which++) { nvn_hmap *m = which ? &g_pso_s : &g_pso;
        for (uint32_t i = 0; i < m->cap && n < live; i++) { if (!m->vals[i]) continue; VkPipeline p = (VkPipeline)m->vals[i];
            all[n] = (pso_old){ p, which, i }; le[n] = (pso_lru_ent){ (uint64_t)(uintptr_t)nvn_hm_get(&g_pso_used, &p), n }; n++; } }
    if (all && c && le) {   // (pso_lru.h: not used in the last 30 s, least recently used first, at most step)
        nc = pso_lru_pick(le, n, g_frame_no + 1, cap, step);
        for (uint32_t i = 0; i < nc; i++) c[i] = all[le[i].id];
    }
    free(all); free(le);
    if (c && nc) {
        uint32_t drop = nc;
        // the keys of the chosen slots first (deleting from a map moves its other entries), then the deletions by key
        uint8_t *keys = malloc((size_t)drop * (sizeof(pkey_s) > sizeof(pkey) ? sizeof(pkey_s) : sizeof(pkey)));
        size_t ks = sizeof(pkey_s) > sizeof(pkey) ? sizeof(pkey_s) : sizeof(pkey);
        for (uint32_t i = 0; keys && i < drop; i++) { nvn_hmap *m = c[i].s ? &g_pso_s : &g_pso; memcpy(keys + i * ks, m->keys + (size_t)c[i].slot * m->klen, m->klen); }
        for (uint32_t i = 0; keys && i < drop; i++) { nvn_hmap *m = c[i].s ? &g_pso_s : &g_pso;
            if (nvn_hm_del(m, keys + i * ks)) { nvn_hm_del(&g_pso_used, &c[i].p); defer_add((dfr){ .kind = DF_PIPELINE, .h.pipe = c[i].p }); g_pso_evicted++; } }
        free(keys);
    }
    uint32_t now = g_pso.n + g_pso_s.n; pthread_mutex_unlock(&g_psomu); free(c);   // (live - now: destroyed by this check)
    static int said; if (said++ < 20) VLOG("pipelines kept: %u of %u left (%llu destroyed so far, at most %d kept)", now, live, (unsigned long long)g_pso_evicted, cap);
}
static int apso_s_queue(vprog *pr, const pkey_s *ks);   // (below, with the background workers)
// the pipeline for k with the draw state ss built in (g_eds 0); record: 1 a draw's (the shared list gets k once, this phone's list k with ss), 2 a background worker's
//   (built now, recorded as a draw's), 0 the prewarm's
static VkPipeline pso_lookup_s(vprog *pr, const pkey *k, const sst *ss, int record) {
    pkey_s ks; memset(&ks, 0, sizeof ks); ks.k = *k; ks.s = *ss;
    pthread_mutex_lock(&g_psomu);
    VkPipeline p = nvn_hm_get(&g_pso_s, &ks); int failed = p ? 0 : nvn_hm_get(&g_pso_s_fail, &ks) != NULL;
    pthread_mutex_unlock(&g_psomu);
    if (p || failed) return p;
    if (record == 1) {   // a draw's miss: to the background workers as with extended dynamic state (apso_s_queue), not built on the replay thread (#99: p95 216 ms on 1.1)
        int r = apso_s_queue(pr, &ks);
        if (r == 0) return VK_NULL_HANDLE;   // (still being built: the draw is skipped this frame, as pso_lookup does)
        if (r > 0) { pthread_mutex_lock(&g_psomu); p = nvn_hm_get(&g_pso_s, &ks); pthread_mutex_unlock(&g_psomu); return p; }
    }
    p = pso_make(pr, *k, ss);
    pthread_mutex_lock(&g_psomu);
    VkPipeline q = nvn_hm_get(&g_pso_s, &ks);
    if (q) { pthread_mutex_unlock(&g_psomu); if (p) vkDestroyPipeline(g_dev, p, NULL); return q; }
    if (p) { nvn_hm_put(&g_pso_s, &ks, p); atomic_fetch_add(&g_pso_new, 1); pso_built_locked(p);
             if (record && !nvn_hm_get(&g_pso_rec_s, &ks)) { nvn_hm_put(&g_pso_rec_s, &ks, (void *)1); atomic_fetch_add(&g_pso_ingame, 1); pso_record_s(pr, k, ss);
                           if (!nvn_hm_get(&g_pso_s_rec, k)) { nvn_hm_put(&g_pso_s_rec, k, (void *)1); pso_record(pr, k); } } }
    else nvn_hm_put(&g_pso_s_fail, &ks, (void *)1);
    pthread_mutex_unlock(&g_psomu);
    return p;
}
// Pipelines built in the background (DM_VK_ASYNC_PSO, on by default on every GPU since 2026-11-22, Adreno only before; =0 / =1 decides): one the game needs that neither the prewarm list nor the cache had is
//   compiled by a worker, and the draws that need it are skipped until it is done (an object missing for a frame or two instead of a hitch, as emulators' asynchronous
//   shaders do). The queue holds 256; a full queue compiles on the spot as before
// Balance: three workers, and each frame may wait up to 8 ms in all (DM_VK_ASYNC_WAIT_MS) for pipelines being built before it skips their draws, so most come in on
//   their first frame (a short hitch at most) instead of objects missing for seconds while one worker went through a new scene's list (6,336 draws skipped, S25+)
#define APSO_N 256
#define APSO_WORKERS 3
// (a queue entry's has_s: the draw state built in, pso_lookup_s on a driver without extended dynamic state; its key then waits in g_pso_pend_s)
static pthread_cond_t g_apso_dcv = PTHREAD_COND_INITIALIZER; static _Atomic int64_t g_apso_budget_ns = 8000000; static int64_t g_apso_frame_ns = 8000000;
static struct { vprog *pr; pkey k; sst s; uint8_t has_s; } g_apso[APSO_N]; static int g_apso_h, g_apso_n; static pthread_mutex_t g_apso_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_apso_cv = PTHREAD_COND_INITIALIZER; static nvn_hmap g_pso_pend = {.klen = sizeof(pkey)}, g_pso_pend_s = {.klen = sizeof(pkey_s)}; static _Atomic uint64_t g_apso_done, g_apso_skipped;
static VkPipeline pso_lookup(vprog *pr, const pkey *k, int record);
static void *apso_main(void *a) { DM_THREAD_NAME("dm-pipelines");
    (void)a;
    for (;;) {
        pthread_mutex_lock(&g_apso_mu); while (!g_apso_n) pthread_cond_wait(&g_apso_cv, &g_apso_mu);
        vprog *pr = g_apso[g_apso_h].pr; pkey_s ks; memset(&ks, 0, sizeof ks); ks.k = g_apso[g_apso_h].k; int has_s = g_apso[g_apso_h].has_s; if (has_s) ks.s = g_apso[g_apso_h].s;
        g_apso_h = (g_apso_h + 1) % APSO_N; g_apso_n--; pthread_mutex_unlock(&g_apso_mu);
        if (has_s) pso_lookup_s(pr, &ks.k, &ks.s, 2); else pso_lookup(pr, &ks.k, 2);   // (2: compile now, recorded as the game's)
        pthread_mutex_lock(&g_psomu); if (has_s) nvn_hm_del(&g_pso_pend_s, &ks); else nvn_hm_del(&g_pso_pend, &ks.k); pthread_mutex_unlock(&g_psomu); atomic_fetch_add(&g_apso_done, 1);
        pthread_mutex_lock(&g_apso_mu); pthread_cond_broadcast(&g_apso_dcv); pthread_mutex_unlock(&g_apso_mu);   // (a draw waiting for it: see apso_wait)
    }
    return NULL;
}
static int apso_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_ASYNC_PSO"); v = quirk_note(Q_ASYNC_PSO, e ? atoi(e) != 0 : quirk_default(Q_ASYNC_PSO, &g_gpuid), e != NULL);   // (vk_quirks.h: on by default on every GPU: a hitch waiting for a
        //   pipeline was the stutter left on Mali / Xclipse / PowerVR (the owner's call, 2026-11-22); Adreno only before. Profile engine_quirks async_pso_off through
        //   gpu_id.eng (Legends: Arceus): off unless DM_VK_ASYNC_PSO=1, on a slower Adreno (710, #34) its skipped draws were objects missing for frames)

        if (v) { int nw = 0; for (int i = 0; i < APSO_WORKERS; i++) { pthread_t t; if (!pthread_create(&t, NULL, apso_main, NULL)) { pthread_detach(t); nw++; } }
            const char *w = getenv("DM_VK_ASYNC_WAIT_MS"); if (w && *w) { int ms = atoi(w); if (ms < 0) ms = 0; if (ms > 33) ms = 33; g_apso_frame_ns = ms * 1000000LL; g_apso_budget_ns = g_apso_frame_ns; }
            if (!nw) v = 0; else VLOG("pipelines the lists did not have: built in the background by %d workers; a frame waits up to %lld ms for them, then their draws are skipped until done (DM_VK_ASYNC_PSO)", nw, (long long)(g_apso_frame_ns / 1000000)); } }
    return v;
}
static int64_t apso_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
// waits (within what is left of this frame's budget) for a queued pipeline (pend: g_pso_pend, or g_pso_pend_s for a key with draw state); 1 when it is built
static int apso_wait_in(nvn_hmap *pend, const void *k) {
    int64_t left = atomic_load(&g_apso_budget_ns); if (left <= 0) return 0;
    int64_t t0 = apso_now(); int done = 0;
    for (;;) {
        pthread_mutex_lock(&g_psomu); done = !nvn_hm_get(pend, k); pthread_mutex_unlock(&g_psomu);
        if (done) break;
        int64_t used = apso_now() - t0; if (used >= left) break;
        struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); int64_t ns = dl.tv_nsec + (left - used < 2000000 ? left - used : 2000000);   // (re-checked every 2 ms at most)
        dl.tv_sec += ns / 1000000000LL; dl.tv_nsec = ns % 1000000000LL;
        pthread_mutex_lock(&g_apso_mu); pthread_cond_timedwait(&g_apso_dcv, &g_apso_mu, &dl); pthread_mutex_unlock(&g_apso_mu);
    }
    atomic_fetch_sub(&g_apso_budget_ns, apso_now() - t0);
    return done;
}
static int apso_wait(const pkey *k) { return apso_wait_in(&g_pso_pend, k); }
// Without extended dynamic state (Vulkan 1.1 drivers: Adreno 610 / 613 / 710 on Qualcomm's 1.1, Arm before r44) a draw's pipeline holds its draw state, so many more are
//   needed, and they used to be built on the replay thread whatever DM_VK_ASYNC_PSO said: "pipelines built in play", p95 216 ms on an Adreno 613 (#99). They go to the
//   same workers now, under the same rule and the same 8 ms budget a frame. DM_VK_ASYNC_PSO_S=0: built on the spot as before. Legends: Arceus keeps async_pso_off
//   (apso_on 0 there): built on the spot. Returns 1 built (look again), 0 still being built (skip the draw this frame), -1 not queued (build it now)
static int apso_s_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_ASYNC_PSO_S"); v = (!e || atoi(e) != 0) && apso_on();
        if (!g_eds) VLOG("pipelines with the draw state built in: %s", v ? "the ones the lists lacked built in the background too (DM_VK_ASYNC_PSO_S)" : "built when a draw needs them"); }
    return v;
}
static int apso_s_queue(vprog *pr, const pkey_s *ks) {
    if (!apso_s_on()) return -1;
    pthread_mutex_lock(&g_psomu);
    int queued = nvn_hm_get(&g_pso_pend_s, ks) != NULL;
    if (!queued) { pthread_mutex_lock(&g_apso_mu);
        if (g_apso_n < APSO_N) { int t = (g_apso_h + g_apso_n) % APSO_N; g_apso[t].pr = pr; g_apso[t].k = ks->k; g_apso[t].s = ks->s; g_apso[t].has_s = 1; g_apso_n++; queued = 1;
                                 nvn_hm_put(&g_pso_pend_s, ks, (void *)1); pthread_cond_signal(&g_apso_cv); }
        pthread_mutex_unlock(&g_apso_mu); }
    pthread_mutex_unlock(&g_psomu);
    if (!queued) return -1;   // (the queue full: built on the spot, as pso_lookup does)
    if (apso_wait_in(&g_pso_pend_s, ks)) return 1;
    atomic_fetch_add(&g_apso_skipped, 1); return 0;
}
static VkPipeline pso_lookup(vprog *pr, const pkey *k, int record) {
    pthread_mutex_lock(&g_psomu);
    VkPipeline p = nvn_hm_get(&g_pso, k); int failed = p ? 0 : nvn_hm_get(&g_pso_fail, k) != NULL;
    if (!p && !failed && record == 1 && apso_on()) {   // the game's draw: to the worker, skipped this time
        int queued = nvn_hm_get(&g_pso_pend, k) != NULL;
        if (!queued) { pthread_mutex_lock(&g_apso_mu);
            if (g_apso_n < APSO_N) { int t = (g_apso_h + g_apso_n) % APSO_N; g_apso[t].pr = pr; g_apso[t].k = *k; g_apso[t].has_s = 0; g_apso_n++; queued = 1; nvn_hm_put(&g_pso_pend, k, (void *)1); pthread_cond_signal(&g_apso_cv); }
            pthread_mutex_unlock(&g_apso_mu); }
        if (queued) { pthread_mutex_unlock(&g_psomu);
            if (apso_wait(k)) { pthread_mutex_lock(&g_psomu); p = nvn_hm_get(&g_pso, k); failed = p ? 0 : nvn_hm_get(&g_pso_fail, k) != NULL; pthread_mutex_unlock(&g_psomu); return p; }
            atomic_fetch_add(&g_apso_skipped, 1); return VK_NULL_HANDLE; } }
    pthread_mutex_unlock(&g_psomu);
    if (p || failed) return p;
    if (record == 2) record = 1;
    p = pso_make(pr, *k, NULL);   // compiled outside the lock: the prewarm thread and the game's thread compile in parallel
    pthread_mutex_lock(&g_psomu);
    VkPipeline q = nvn_hm_get(&g_pso, k);
    if (q) { pthread_mutex_unlock(&g_psomu); if (p) vkDestroyPipeline(g_dev, p, NULL); return q; }   // the other thread finished the same one first
    if (p) { nvn_hm_put(&g_pso, k, p); pso_built_locked(p); if (record && !nvn_hm_get(&g_pso_rec, k)) { nvn_hm_put(&g_pso_rec, k, (void *)1); pso_record(pr, k); atomic_fetch_add(&g_pso_ingame, 1); } atomic_fetch_add(&g_pso_new, 1); }
    else nvn_hm_put(&g_pso_fail, k, (void *)1);
    pthread_mutex_unlock(&g_psomu);
    return p;
}
static size_t g_pcache_disk;   // the bytes of pipeline cache the file holds as far as this run knows: what the driver took at pcache_open, then each save's
static _Atomic uint64_t g_pcache_saved_at, g_pw_released_at;   // (mono_ns) the last save's start, and when prewarm_finished let the game go (simdev t_vkdev: S1's order)
static atomic_int g_pw_complete; static void pw_stamp_after_save(int before, size_t n);   // (S3, below)
static void pcache_save(void) {
    const char *path = getenv("DM_VK_PCACHE"); if (!path || !g_pcache) return;
    uint64_t t0 = mono_ns(); atomic_store(&g_pcache_saved_at, t0);
    pw_stamp_after_save(1, 0);   // (the lists' sizes before the cache is read: a pipeline recorded after that is not claimed to be in it)
    size_t n = 0; if (vkGetPipelineCacheData(g_dev, g_pcache, &n, NULL) || !n) return;
    void *d = malloc(n); if (vkGetPipelineCacheData(g_dev, g_pcache, &n, d)) { free(d); return; }
    char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb");
    if (f) { int ok = fwrite(d, 1, n, f) == n; ok &= !fclose(f); if (ok && !rename(tmp, path)) { g_pcache_disk = n; pw_stamp_after_save(0, n); } }
    free(d); atomic_store(&g_pso_new, 0);
    VLOG("pipeline cache saved: %.1f MB in %.0f ms", n / 1048576.0, (mono_ns() - t0) / 1e6);
}
// The prewarm's save (perf-1124 S1): after the game is let go (prewarm_main), and only when the driver's cache grew since it was read or last written. It used to
//   come before the release, on every start: vkGetPipelineCacheData plus a 60-106 MB write inside the wait, also when every pipeline came from the cache.
//   DM_VK_PCACHE_SAVE=always: written every time, as before (still after the release)
static void pcache_save_new(void) {
    const char *path = getenv("DM_VK_PCACHE"), *e = getenv("DM_VK_PCACHE_SAVE"); if (!path || !g_pcache) return;
    size_t n = 0;
    pw_stamp_after_save(1, 0);
    if (!(e && !strcmp(e, "always")) && !vkGetPipelineCacheData(g_dev, g_pcache, &n, NULL) && n == g_pcache_disk) {
        VLOG("pipeline cache: nothing new since it was read (%.1f MB): not written again (DM_VK_PCACHE_SAVE=always writes it)", n / 1048576.0);
        pw_stamp_after_save(0, g_pcache_disk); return; }
    pcache_save();
}
void nvn_vk_cache_save(void) { if (g_dev) pcache_save(); }
void *saver(void *a) { pcache_save(); atomic_store((atomic_int *)a, 0); return NULL; }   // the app calls it when going to the background
static void pcache_open(void) {
    const char *path = getenv("DM_VK_PCACHE"); size_t n = 0; void *d = NULL; uint64_t t0 = mono_ns();
    if (path) { FILE *f = fopen(path, "rb"); if (f) { fseek(f, 0, SEEK_END); long l = ftell(f); rewind(f); if (l > 0) { d = malloc((size_t)l); n = fread(d, 1, (size_t)l, f); } fclose(f); } }
    VkPipelineCacheCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = n, .pInitialData = d };
    if (vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache)) { ci.initialDataSize = 0; ci.pInitialData = NULL; vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache); n = 0; }   // a cache of another driver version: start empty
    free(d); g_pcache_disk = n;
    if (path) VLOG("pipeline cache: %.1f MB from %s (read and handed to the driver in %.0f ms)", n / 1048576.0, path, (mono_ns() - t0) / 1e6);
}
static int pso_rec_load(const char *path, pso_rec **out, int *nout) {
    FILE *f = path ? fopen(path, "rb") : NULL; if (!f) return 0;
    char m[4]; uint32_t v[2];
    if (fread(m, 1, 4, f) != 4 || memcmp(m, PSO_MAGIC, 4) || fread(v, 4, 2, f) != 2 || v[0] != 1 || v[1] != sizeof(pso_rec)) { fclose(f); VLOG("[RED] pipeline list %s: not this version's format", path); return 0; }
    pso_rec r; int k = 0;
    while (fread(&r, sizeof r, 1, f) == 1) { if (*nout % 1024 == 0) *out = realloc(*out, sizeof **out * (size_t)(*nout + 1024)); (*out)[(*nout)++] = r; k++; }
    fclose(f); return k;
}
// The prewarm stays gentle on memory: one pipeline at a time with a short pause after each (the driver's compiler gets its memory back between them), and on Linux / Android
// it waits while the system has less than DM_VK_PREWARM_MIN_MB (1536) available, so it never pushes the phone into the low-memory killer while the game is loading
static long mem_avail_mb(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r"); char ln[128]; long v = -1;
    while (f && fgets(ln, sizeof ln, f)) if (!strncmp(ln, "MemAvailable:", 13)) { v = atol(ln + 13) / 1024; break; }
    if (f) fclose(f); return v;
#else
    return -1;
#endif
}
// DM_VK_PREWARM: "all" (the default) builds every listed pipeline before the game draws its first frame, on several threads (the game waits at its first submission
//   and the app shows the count, nvn_vk_prewarm_progress), so nothing is built while playing: on Adreno a pipeline still being built skips its draw (objects missing
//   or popping in for the first minutes), elsewhere it is a hitch. An S25+ built its 1,400-1,600 recorded pipelines in 7-21 s one at a time. "background": as before,
//   one at a time with a pause after each while the game plays
static int prewarm_all(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PREWARM"); v = !(e && !strcmp(e, "background")); } return v; }
static atomic_int g_pw_done, g_pw_total, g_pw_hold;   // pipelines handled / listed, and 1 while the game waits for them
static pthread_mutex_t g_pw_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pw_cv = PTHREAD_COND_INITIALIZER;
// for the app's progress line (android_main.c nativePrewarm): 1 while the game waits for the pipelines
int nvn_vk_prewarm_progress(int *done, int *total) { *done = atomic_load(&g_pw_done); *total = atomic_load(&g_pw_total); return atomic_load(&g_pw_hold); }
static void prewarm_finished(void) { pthread_mutex_lock(&g_pw_mu); atomic_store(&g_pw_released_at, mono_ns()); atomic_store(&g_pw_hold, 0); pthread_cond_broadcast(&g_pw_cv); pthread_mutex_unlock(&g_pw_mu); }
// the game's first submission: waits until the prewarm is done (at most DM_VK_PREWARM_WAIT seconds, 300 by default; then the rest is built while it plays)
// the game is playing (its first submission past prewarm_wait): from then on the prewarm builds on one thread, one pipeline after each present (prewarm_pace)
static atomic_int g_pw_playing; static _Atomic uint64_t g_pw_play_n; static atomic_int g_pw_warm; static void pw_release(const char *why);
static void prewarm_wait(void) {
    { static int first; if (!first++) dm_boot_mark("the game's first submission"); }
    if (!atomic_load(&g_pw_hold)) { if (!atomic_load(&g_pw_warm)) atomic_store(&g_pw_playing, 1); return; }   // (a matched stamp: from the first present, mtl_present)
    const char *e = getenv("DM_VK_PREWARM_WAIT"); double lim = e ? atof(e) : 300; uint64_t t0 = mono_ns();
    pthread_mutex_lock(&g_pw_mu);
    while (atomic_load(&g_pw_hold) && (mono_ns() - t0) / 1e9 < lim) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += 200000000; if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&g_pw_cv, &g_pw_mu, &ts);
    }
    int late = atomic_load(&g_pw_hold); atomic_store(&g_pw_hold, 0); if (!atomic_load(&g_pw_warm)) atomic_store(&g_pw_playing, 1); pthread_mutex_unlock(&g_pw_mu);
    VLOG("prewarm: the game waited %.1f s for its pipelines (%d of %d)%s", (mono_ns() - t0) / 1e9, atomic_load(&g_pw_done), atomic_load(&g_pw_total),
         late ? ": the wait ran out, the rest are built while it plays" : "");
    dm_boot_mark("the game goes on past its first submission");
}
// During play (S25+ Arceus, Xclipse 530 Violet: hitches of 60-580 ms with the render thread stuck in its submit while background pipelines were being built, the driver
//   and the cores shared with it): one pipeline right after each present, so a build overlaps the start of the next frame's work, not its submission; at most
//   one per frame, at a lower priority. DM_VK_PREWARM_PLAY=free: as before (3 ms apart)
__attribute__((weak)) uint64_t dm_guest_frame(void) { return 0; }   // (presents so far: host hle_os.c's; this one only for the graphics tests, which run no prewarm)
static int prewarm_play_free(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PREWARM_PLAY"); v = e && !strcmp(e, "free"); } return v; }
static void prewarm_pace_play(void) {
    if (!atomic_load(&g_pw_playing) || prewarm_play_free()) return;
    static __thread int lowered; if (!lowered) { lowered = 1; setpriority(PRIO_PROCESS, (id_t)gettid(), 10); }
    static _Atomic uint64_t last; uint64_t f0 = atomic_load(&last);
    for (int k = 0; k < 2000 && dm_guest_frame() <= f0; k++) usleep(2000);   // (the next present, at most 4 s: a paused game still gets its pipelines)
    atomic_store(&last, dm_guest_frame());
}
static void prewarm_pace(int i, int *waits) {
    if (!prewarm_all() && prewarm_play_free()) usleep(3000);
    if (i % 8) return;
    static long floor_mb = -1; if (floor_mb < 0) { const char *e = getenv("DM_VK_PREWARM_MIN_MB"); floor_mb = e ? atol(e) : 1536; }
    // S4 (d): under the floor while the game waits, the game is let go instead (the pauses counted against the wait: most of a Mali-G610's 21-26 s, #112);
    //   the rest is built while it plays, with the same pauses. DM_VK_PREWARM_LOWMEM=wait: the game waits through them, as before
    static int lm = -1; if (lm < 0) { const char *e = getenv("DM_VK_PREWARM_LOWMEM"); lm = !(e && !strcmp(e, "wait")); }
    for (int k = 0; k < 600; k++) { long m = mem_avail_mb(); if (m < 0 || m >= floor_mb) break; if (!k) { (*waits)++; if (lm) pw_release("low memory"); } usleep(100000); }   // at most a minute per wait
}
// ---- A later start without the hold, the list kept from growing, the cap (perf-1124 fable-loads.md S3 / S4; the rules without a GPU: prewarm_policy.h)
// S3: a complete prewarm leaves a stamp beside the pipeline cache (<DM_VK_PCACHE>.stamp: the lists' sizes, the cache's, the driver's pipelineCacheUUID and
//   version, the app's version code, the shader pack and the shader rewrites in force). A later start that finds the same stamp knows every listed pipeline is
//   in the driver's cache: DM_VK_PREWARM_HOLD decides how much of the list the game waits for then. "none" where a pipeline missing in play is built in the
//   background with its draws skipped (DM_VK_ASYNC_PSO, every GPU's default), so the game goes on at once and the prewarm's threads keep building from the
//   cache until the first present; the first 600 list entries (boot, title screen, the first area: the list is in first-use order) where a missing pipeline is
//   built on the spot, a hitch (Legends: Arceus, profile async_pso_off). "all": every pipeline, as before. A start whose stamp differs (the first, a grown list,
//   a new driver or app) holds for every pipeline as before. Measured waits: Arceus on an Adreno 830 + Turnip 5-11 s at every start with a 105.8 MB cache,
//   Shield on the S25+ 1.6 s, a Mali-G610 21-26 s (fable-loads.md section 1)
static atomic_int g_pw_warm;       // this start matched the stamp: the prewarm's threads go on until the first present (prewarm_wait, mtl_present)
static int g_pw_hold_n;            // >0: the game is let go once the first g_pw_hold_n list entries are handled
static atomic_int g_pw_hold_cnt;   // (of those, handled so far)
static atomic_int g_pw_complete;   // every list entry was handled this run: the cache holds the lists (the stamp may be written)
static uint32_t pw_flags(void) {   // the shader rewrites in force (another set: other SPIR-V, the driver's cache entries do not match)
    pthread_once(&g_gb_once, gbias_decide);
    uint32_t v[] = { (uint32_t)g_fctl, (uint32_t)g_nanmm, (uint32_t)g_clampzero, (uint32_t)g_fmz, g_exact, (uint32_t)g_initloc, (uint32_t)g_qlod, (uint32_t)g_yflip, (uint32_t)g_drawp,
                     (uint32_t)b10_on(), (uint32_t)g_eds, (uint32_t)g_dynren, (uint32_t)g_probe, (uint32_t)intbits_on(), (uint32_t)g_gbias, (uint32_t)g_tf_pts, (uint32_t)g_clipctl, g_api };
    uint32_t h = 2166136261u; for (size_t i = 0; i < sizeof v / sizeof *v; i++) { h ^= v[i]; h *= 16777619u; }
    return h;
}
static uint64_t pw_fsize(const char *p, const char *suffix, uint64_t *mtime) {
    char q[1200]; struct stat st; if (mtime) *mtime = 0;
    if (!p || !*p || snprintf(q, sizeof q, "%s%s", p, suffix) >= (int)sizeof q || stat(q, &st)) return 0;
    if (mtime) *mtime = (uint64_t)st.st_mtime;
    return (uint64_t)st.st_size;
}
static void pw_stamp_now(pw_stamp *s) {
    memset(s, 0, sizeof *s); const char *seed = getenv("DM_VK_PSO_SEED"), *rec = getenv("DM_VK_PSO_RECORD"), *app = getenv("DM_APP_VERSION");
    s->seed = pw_fsize(seed, "", NULL); s->rec = pw_fsize(rec, "", NULL); s->rec_sst = pw_fsize(rec, ".sst", NULL); s->cs_seed = pw_fsize(seed, ".cs", NULL); s->cs_rec = pw_fsize(rec, ".cs", NULL);
    s->pak = pw_fsize(getenv("DM_SPV_PAK"), "", &s->pak_mtime);
    s->vendor = g_props.vendorID; s->device = g_props.deviceID; s->drv_ver = g_props.driverVersion; memcpy(s->uuid, g_props.pipelineCacheUUID, 16);
    s->flags = pw_flags(); snprintf(s->app, sizeof s->app, "%s", app ? app : ""); snprintf(s->chain, sizeof s->chain, "%s", DMSH_CHAIN);
}
static int pw_stamp_path(char *p, size_t n) { const char *c = getenv("DM_VK_PCACHE"); return c && *c && snprintf(p, n, "%s.stamp", c) < (int)n; }
// the stamp after a save (s: the lists as they were before the cache was read for it; s->cache: what the file holds now)
static void pw_stamp_write(const pw_stamp *s) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; char p[1200], t[1300], line[512];
    if (!atomic_load(&g_pw_complete) || !pw_stamp_path(p, sizeof p)) return;
    pw_stamp_fmt(s, line, sizeof line); snprintf(t, sizeof t, "%s.tmp", p);
    pthread_mutex_lock(&mu);
    FILE *f = fopen(t, "w"); if (f) { int ok = fputs(line, f) >= 0; ok &= !fclose(f); if (ok) rename(t, p); }
    pthread_mutex_unlock(&mu);
}
static void pw_stamp_clear(void) { char p[1200]; if (pw_stamp_path(p, sizeof p)) unlink(p); }
// pcache_save's two halves: before (1) the lists as they are before the cache is read; after (0) the stamp with the bytes written (n). Only once this run's
//   prewarm was complete: every listed pipeline went through the cache, and every one a draw built since was recorded and built through it too
static pw_stamp g_pw_pre; static pthread_mutex_t g_pw_pre_mu = PTHREAD_MUTEX_INITIALIZER;
static void pw_stamp_after_save(int before, size_t n) {
    if (!atomic_load(&g_pw_complete)) return;
    if (before) { pw_stamp s; pw_stamp_now(&s); pthread_mutex_lock(&g_pw_pre_mu); g_pw_pre = s; pthread_mutex_unlock(&g_pw_pre_mu); return; }
    pthread_mutex_lock(&g_pw_pre_mu); pw_stamp s = g_pw_pre; pthread_mutex_unlock(&g_pw_pre_mu);
    s.cache = n; pw_stamp_write(&s);
}
// 1: a pipeline missing in play is built in the background and its draws skipped (apso_on's rule, read here without starting its workers); 0: built on the spot
static int pw_misses_skip(void) {
    const char *e = getenv("DM_VK_ASYNC_PSO"); int a = e ? atoi(e) != 0 : quirk_default(Q_ASYNC_PSO, &g_gpuid);
    if (!g_eds) { const char *s = getenv("DM_VK_ASYNC_PSO_S"); a = a && (!s || atoi(s) != 0); }
    return a;
}
// the game let go before every pipeline is built (a matched stamp's count, low memory: S4 d); once
static void pw_release(const char *why) {
    if (!atomic_load(&g_pw_hold)) return;
    prewarm_finished();
    VLOG("prewarm: the game goes on after %d of %d pipelines (%s); the rest are built while it plays", atomic_load(&g_pw_done), atomic_load(&g_pw_total), why);
}
static void pw_count_handled(int i) {   // list entry i handled (built, already there, skipped): the count hold's release
    if (i >= g_pw_hold_n) return;
    int lim = g_pw_hold_n < atomic_load(&g_pw_total) ? g_pw_hold_n : atomic_load(&g_pw_total);
    if (atomic_fetch_add(&g_pw_hold_cnt, 1) + 1 == lim) { char w[96]; snprintf(w, sizeof w, "the stamp matched: the first %d list entries (DM_VK_PREWARM_HOLD)", g_pw_hold_n); pw_release(w); }
}
// S4 (a): a pipeline from the lists counts as recorded, so one the cap destroys and a draw builds again is not appended again (the record list only grew on
//   phones under the cap: 31336 listed for 4120 distinct on a Mali-G610, #112). (b) the record lists without their repeats when they are over a tenth of them.
//   DM_VK_PSO_COMPACT=0: neither
static int pw_compact_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PSO_COMPACT"); v = !(e && *e == '0'); } return v; }
// (read, compacted and written beside it without the lock; under g_psomu, where pso_record / pso_record_s append, only the check that nothing was appended
//   meanwhile, the rename and the reopening: the game's own pipeline lookups wait for that much, not for a 20 MB read and write)
static void pso_list_compact(const char *path, const char *magic, size_t esz, const nvn_hmap *skip, FILE **open_f) {
    if (!path) return;
    FILE *f = fopen(path, "rb"); uint8_t *d = NULL; long l = 0;
    if (f) { fseek(f, 0, SEEK_END); l = ftell(f); rewind(f); if (l > 12) { d = malloc((size_t)l); if (d && fread(d, 1, (size_t)l, f) != (size_t)l) { free(d); d = NULL; } } fclose(f); }
    uint32_t v[2] = { 0, 0 }; if (d) memcpy(v, d + 4, 8);
    if (!d || memcmp(d, magic, 4) || v[0] != 1 || v[1] != esz || ((size_t)l - 12) % esz) { free(d); return; }
    int n = (int)(((size_t)l - 12) / esz), m = pw_dedupe(d + 12, n, esz, skip);
    if (pw_compact_worth(n, m)) {
        char t[1200]; snprintf(t, sizeof t, "%s.tmp", path); FILE *o = fopen(t, "wb"); int ok = 0;
        if (o) { ok = fwrite(d, 1, 12 + (size_t)m * esz, o) == 12 + (size_t)m * esz; ok &= !fclose(o); }
        pthread_mutex_lock(&g_psomu);
        struct stat st; int same = !stat(path, &st) && st.st_size == l;   // (an entry appended meanwhile: left as it is, compacted at a later start)
        if (ok && same && !rename(t, path)) {
            if (*open_f) { fclose(*open_f); *open_f = fopen(path, "ab"); }   // (this run's appends go on into the new file)
            pthread_mutex_unlock(&g_psomu);
            VLOG("pipeline list %s: %d entries, %d of them repeats or in the shipped list: written again with %d", path, n, n - m, m);
        } else { pthread_mutex_unlock(&g_psomu); unlink(t); }
    }
    free(d);
}
static void pw_compact_lists(void) {
    if (!pw_compact_on()) return;
    const char *rec = getenv("DM_VK_PSO_RECORD"), *seed = getenv("DM_VK_PSO_SEED"); if (!rec || !*rec) return;
    nvn_hmap sk = {.klen = sizeof(pso_rec)}; pso_rec *rs = NULL; int n = 0;
    if (pso_rec_load(seed, &rs, &n)) for (int i = 0; i < n; i++) nvn_hm_put(&sk, &rs[i], (void *)1);
    free(rs);
    pso_list_compact(rec, PSO_MAGIC, sizeof(pso_rec), sk.n ? &sk : NULL, &g_psorec);
    free(sk.keys); free(sk.vals);
    char p[1100]; snprintf(p, sizeof p, "%s.sst", rec); pso_list_compact(p, PSO_MAGIC_S, sizeof(pso_rec_s), NULL, &g_psorec_s);
}
// S4 (c): under a cap of live pipelines (vk_quirks.h Q_PIPE_KEEP: Arm's driver, Xclipse, phones under 9 GB) the prewarm built every listed one and the cap destroyed
//   most of them 10 s into play (about 3400 of 4120 on a Mali-G610). Now the first cap x 9/10 are kept; the rest only go into the driver's cache (built and let go,
//   as DM_VK_PREWARM=cache) at a start whose stamp differs, and are skipped at one whose stamp matched (the cache has them). DM_VK_PREWARM_CAP=0: every one kept
static int pw_cap_keep(void) { const char *e = getenv("DM_VK_PREWARM_CAP"); if (e && *e == '0') return 0; return pw_keep_of_cap(pso_cap()); }
// The prewarm without extended dynamic state: this phone's own list of pipelines with their draw state (pso_record_s), the same pacing as below
static nvn_hmap g_pw_sprogs = {.klen = 40};   // (its programs, by their shaders' keys)
static void *prewarm_static(void) {
    const char *e = getenv("DM_VK_PSO_RECORD"); char path[1100]; if (!e) return NULL; snprintf(path, sizeof path, "%s.sst", e);
    FILE *f = fopen(path, "rb");
    if (!f) { VLOG("prewarm: nothing yet (no extended dynamic state: pipelines are recorded with their draw state as the game draws, and built from the next start)"); return NULL; }
    char m[4]; uint32_t v[2];
    if (fread(m, 1, 4, f) != 4 || memcmp(m, PSO_MAGIC_S, 4) || fread(v, 4, 2, f) != 2 || v[0] != 1 || v[1] != sizeof(pso_rec_s)) { fclose(f); VLOG("[RED] pipeline list %s: not this version's format", path); return NULL; }
    uint64_t t0 = mono_ns(); int waits = 0, n = 0, built = 0, had = 0, failed = 0, noshader = 0, cached = 0, skipped = 0, kept = 0;
    int keep = pw_cap_keep(), compact = pw_compact_on();
    pso_rec_s r;
    { long here = ftell(f); fseek(f, 0, SEEK_END); long end = ftell(f); fseek(f, here, SEEK_SET); atomic_store(&g_pw_total, (int)((end - here) / (long)sizeof r)); }
    while (fread(&r, sizeof r, 1, f) == 1) {
        if (n) pw_count_handled(n - 1);   // (the entry before this one is handled)
        n++; atomic_fetch_add(&g_pw_done, 1);
        vprog *pr = nvn_hm_get(&g_pw_sprogs, r.r.key);
        if (!pr) {
            pr = calloc(1, sizeof *pr); int ok = 1;
            for (int st = 0; st < 5; st++) { static const uint8_t z[8]; if (!memcmp(r.r.key[st], z, 8)) continue; g_sh_quiet = 1; pr->sh[st] = shader_get(r.r.key[st], st); g_sh_quiet = 0; if (!pr->sh[st]) ok = 0; }
            if (!ok || !pr->sh[0]) { free(pr); noshader++; continue; }
            nvn_hm_put(&g_pw_sprogs, r.r.key, pr);
        }
        pkey k = r.r.k; for (int st = 0; st < 5; st++) k.sh[st] = pr->sh[st];
        b10_key_here(&k);   // (DM_VK_B10_AS_RGBA16: an R11G11B10F target is the stand-in here)
        prog_layout(pr, 0);
        sst_fold(&r.s);   // (a list recorded before the fold: the key the draws look up now)
        pkey_s ks; memset(&ks, 0, sizeof ks); ks.k = k; ks.s = r.s;
        pthread_mutex_lock(&g_psomu); int have = nvn_hm_get(&g_pso_s, &ks) != NULL;
        if (compact) { nvn_hm_put(&g_pso_rec_s, &ks, (void *)1); nvn_hm_put(&g_pso_s_rec, &k, (void *)1); }   // (S4 a: listed already)
        pthread_mutex_unlock(&g_psomu);
        if (have) { had++; kept++; continue; }
        if (keep && kept >= keep) {   // (S4 c: past the cap)
            if (atomic_load(&g_pw_warm)) { skipped++; continue; }
            prewarm_pace_play(); VkPipeline p = pso_make(pr, k, &r.s); if (p) { vkDestroyPipeline(g_dev, p, NULL); cached++; } else failed++;
            prewarm_pace(built + cached + failed, &waits); continue; }
        prewarm_pace_play();
        if (pso_lookup_s(pr, &k, &r.s, 0)) { built++; kept++; } else failed++;
        prewarm_pace(built + cached + failed, &waits);
    }
    if (n) pw_count_handled(n - 1);
    fclose(f);
    VLOG("prewarm (draw state built in): %d pipelines recorded on this phone: %d built, %d already there, %d failed, %d with a shader missing, %.1f s (paused %d times for low memory)",
         n, built, had, failed, noshader, (mono_ns() - t0) / 1e9, waits);
    if (keep && (cached || skipped)) VLOG("prewarm: %d kept (the cap: at most %d live pipelines), %d past it only put into the pipeline cache, %d skipped (already in it)", kept, pso_cap(), cached, skipped);
    atomic_store(&g_pw_complete, 1);
    return NULL;   // (the cache is saved by prewarm_main, after the game is let go)
}
// the main prewarm's shared state: the list, the next entry to take, the programs made so far (their layouts made once, under pw.mu: prog_layout is not for two threads)
// t_*: the prewarm's stage timers (perf-1124 I7), nanoseconds summed over its threads: waiting for pw.mu, loading shaders (shader_get), making layouts
//   (prog_layout), building pipelines (pso_lookup / pso_make). Printed with the prewarm's summary, so a log says which part of the wait is ours and which the driver's
// keep / kept: the cap's share (S4 c) and how many are kept so far; cached / skipped: past it, put into the driver's cache only / left out (a matched stamp)
static struct { pso_rec *rs; int n, ns, cache_only, keep, compact; atomic_int next, unfit, built, had, failed, noshader, waits, kept, cached, skipped, dups; pthread_mutex_t mu; nvn_hmap progs, seen;
                _Atomic uint64_t t_lock, t_shader, t_layout, t_pipe; } g_pw = {
    .mu = PTHREAD_MUTEX_INITIALIZER, .progs = {.klen = 40}, .seen = {.klen = sizeof(pso_rec)} };
// a pipeline from the shipped list (recorded on another phone): its colour / depth / stencil targets and vertex formats must be ones this GPU has, or it is left
//   out (a driver handed an unsupported format may fail or worse; this phone's own list only holds what it built)
static int fmt_has(uint32_t f, VkFormatFeatureFlags need, int buffer) {
    if (!f) return 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, (VkFormat)f, &fp);
    return ((buffer ? fp.bufferFeatures : fp.optimalTilingFeatures) & need) == need;
}
static int pso_formats_ok(const pkey *k) {
    for (int i = 0; i < 8; i++) if (!fmt_has(k->cfmt[i], VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT, 0)) return 0;
    if (!fmt_has(k->dfmt, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, 0) || !fmt_has(k->sfmt, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, 0)) return 0;
    for (int i = 0; i < 32; i++) if (k->attr[i].used && !fmt_has(k->attr[i].fmt, VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT, 1)) return 0;
    return 1;
}
// one list entry; 1: this extra thread stops (the game plays: one thread builds the rest)
static int prewarm_one(int i, void *arg, int *waits) {
    atomic_fetch_add(&g_pw_done, 1);
    if (i < g_pw.ns && !pso_formats_ok(&g_pw.rs[i].k)) { atomic_fetch_add(&g_pw.unfit, 1); return 0; }
    uint64_t tl = mono_ns(); pthread_mutex_lock(&g_pw.mu); atomic_fetch_add(&g_pw.t_lock, mono_ns() - tl);
    if (nvn_hm_get(&g_pw.seen, &g_pw.rs[i])) { pthread_mutex_unlock(&g_pw.mu); if (i >= g_pw.ns) atomic_fetch_add(&g_pw.dups, 1); return 0; }
    nvn_hm_put(&g_pw.seen, &g_pw.rs[i], (void *)1);
    vprog *pr = nvn_hm_get(&g_pw.progs, g_pw.rs[i].key);
    if (!pr && sh_mt()) {   // (perf-1124 S2: the shaders loaded outside pw.mu, on every prewarm thread; only the layout under it)
        pthread_mutex_unlock(&g_pw.mu);
        vprog *np = calloc(1, sizeof *np); int ok = 1; uint64_t ts = mono_ns();
        for (int st = 0; st < 5; st++) { static const uint8_t z[8]; if (!memcmp(g_pw.rs[i].key[st], z, 8)) continue; g_sh_quiet = 1; np->sh[st] = shader_get(g_pw.rs[i].key[st], st); g_sh_quiet = 0; if (!np->sh[st]) ok = 0; }
        atomic_fetch_add(&g_pw.t_shader, mono_ns() - ts);
        if (!ok || !np->sh[0]) { free(np); atomic_fetch_add(&g_pw.noshader, 1); return 0; }
        tl = mono_ns(); pthread_mutex_lock(&g_pw.mu); atomic_fetch_add(&g_pw.t_lock, mono_ns() - tl);
        if ((pr = nvn_hm_get(&g_pw.progs, g_pw.rs[i].key))) free(np);   // (another thread made the same program meanwhile)
        else { uint64_t tp = mono_ns(); prog_layout(np, 0); atomic_fetch_add(&g_pw.t_layout, mono_ns() - tp); nvn_hm_put(&g_pw.progs, g_pw.rs[i].key, np); pr = np; }
    }
    if (!pr) {
        pr = calloc(1, sizeof *pr); int ok = 1; uint64_t ts = mono_ns();
        for (int st = 0; st < 5; st++) { static const uint8_t z[8]; if (!memcmp(g_pw.rs[i].key[st], z, 8)) continue; g_sh_quiet = 1; pr->sh[st] = shader_get(g_pw.rs[i].key[st], st); g_sh_quiet = 0; if (!pr->sh[st]) ok = 0; }
        uint64_t tp = mono_ns(); atomic_fetch_add(&g_pw.t_shader, tp - ts);
        if (!ok || !pr->sh[0]) { free(pr); pthread_mutex_unlock(&g_pw.mu); atomic_fetch_add(&g_pw.noshader, 1); return 0; }
        prog_layout(pr, 0); atomic_fetch_add(&g_pw.t_layout, mono_ns() - tp);
        nvn_hm_put(&g_pw.progs, g_pw.rs[i].key, pr);
    }
    pthread_mutex_unlock(&g_pw.mu);
    pkey k = g_pw.rs[i].k; for (int st = 0; st < 5; st++) k.sh[st] = pr->sh[st];
    b10_key_here(&k);   // (DM_VK_B10_AS_RGBA16: an R11G11B10F target is the stand-in here)
    pthread_mutex_lock(&g_psomu); int have = nvn_hm_get(&g_pso, &k) != NULL;
    if (g_pw.compact) nvn_hm_put(&g_pso_rec, &k, (void *)1);   // (S4 a: listed already, not appended again when the cap destroys it and a draw builds it again)
    pthread_mutex_unlock(&g_psomu);
    if (have) { atomic_fetch_add(&g_pw.had, 1); atomic_fetch_add(&g_pw.kept, 1); return 0; }
    int cache_only = g_pw.cache_only;
    if (g_pw.keep && !cache_only && atomic_load(&g_pw.kept) >= g_pw.keep) {   // (S4 c: past the cap)
        if (atomic_load(&g_pw_warm)) { atomic_fetch_add(&g_pw.skipped, 1); return 0; }
        cache_only = 2;
    }
    int last_one = arg && atomic_load(&g_pw_playing) && !prewarm_play_free();   // (the game started: this extra thread builds the one it took, then one thread builds the rest)
    prewarm_pace_play();
    uint64_t tb = mono_ns();
    if (cache_only) { VkPipeline p = pso_make(pr, k, NULL); if (p) { vkDestroyPipeline(g_dev, p, NULL); atomic_fetch_add(cache_only == 2 ? &g_pw.cached : &g_pw.built, 1); } else atomic_fetch_add(&g_pw.failed, 1); }   // (into the driver's cache, not kept)
    else if (pso_lookup(pr, &k, 0)) { atomic_fetch_add(&g_pw.built, 1); atomic_fetch_add(&g_pw.kept, 1); } else atomic_fetch_add(&g_pw.failed, 1);
    atomic_fetch_add(&g_pw.t_pipe, mono_ns() - tb);
    if (atomic_load(&g_pw_playing)) atomic_fetch_add(&g_pw_play_n, 1);   // (handled while the game runs: a stamped start's prewarm goes on through the first loads, hle_nvn.c's [load] line)
    prewarm_pace(atomic_load(&g_pw.built) + atomic_load(&g_pw.cached) + atomic_load(&g_pw.failed), waits);
    return last_one;
}
static void *prewarm_worker(void *arg) {
    if (arg) DM_THREAD_NAME("dm-prewarm");   // (the threads started for it; the caller keeps its own name)
    int waits = 0;
    for (int i; (i = atomic_fetch_add(&g_pw.next, 1)) < g_pw.n; ) {
        int stop = prewarm_one(i, arg, &waits);
        pw_count_handled(i);
        if (stop) break;
    }
    atomic_fetch_add(&g_pw.waits, waits);
    return NULL;
}
// how many build at once: one in the background; before the game, the cores less one (at most 6), 2 on phones with 6 GB or less (each build takes the driver's memory)
static int prewarm_threads(void) {
    if (!prewarm_all()) return 1;
    const char *e = getenv("DM_VK_PREWARM_THREADS"); if (e && atoi(e) > 0) return atoi(e) > 16 ? 16 : atoi(e);
    long cpus = sysconf(_SC_NPROCESSORS_ONLN), pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    int t = cpus > 1 ? (int)cpus - 1 : 1; if (t > 6) t = 6;
    if (pages > 0 && psz > 0 && (double)pages * psz < 6.5e9 && t > 2) t = 2;
    return t;
}
// Compute pipelines in the prewarm. A compute pipeline was only ever built at its first dispatch, on the replay thread (dispatch below): the driver's compile sat
//   in the middle of a frame the first time the game ran that shader (Scarlet / Violet's GPU culling and instancing as the open world loads; Sword / Shield have
//   72 compute shaders, 41 distinct). The list of graphics pipelines (pso_rec) has no room for them (five stage keys), so the compute shaders the game has
//   dispatched on this phone are kept beside it, <DM_VK_PSO_RECORD>.cs ("NXPC" + 8-byte shader keys; a shipped <DM_VK_PSO_SEED>.cs is read too when there is
//   one). At the prewarm each is built once into the driver's pipeline cache and let go: the dispatch's own build then comes from the cache, with the game's own
//   program and layout as before (nothing else changes there). DM_VK_PREWARM_COMPUTE=0: off
#define PSO_MAGIC_C "NXPC"
static void cs_record(const vshader *cs) {   // the replay thread only (dispatch): a compute shader's first pipeline on this run
    static nvn_hmap known = {.klen = 8}; static FILE *f; static int tried;
    uint8_t k[8]; if (hexkey(cs->key, k)) return;
    if (!tried++) { const char *e = getenv("DM_VK_PSO_RECORD"); char p[1100];
        if (e && *e && snprintf(p, sizeof p, "%s.cs", e) < (int)sizeof p) {
            FILE *r = fopen(p, "rb"); char mg[4]; uint8_t o[8]; int have = 0;   // (what earlier runs wrote: not written twice)
            if (r) { if (fread(mg, 1, 4, r) == 4 && !memcmp(mg, PSO_MAGIC_C, 4)) { have = 1; while (fread(o, 8, 1, r) == 1) nvn_hm_put(&known, o, (void *)1); } fclose(r); }
            f = fopen(p, have ? "ab" : "wb"); if (f && !have) fwrite(PSO_MAGIC_C, 1, 4, f); } }
    if (nvn_hm_get(&known, k)) return;
    nvn_hm_put(&known, k, (void *)1);
    if (f) { fwrite(k, 8, 1, f); fflush(f); }
}
static void prewarm_compute(void) {
    { const char *e = getenv("DM_VK_PREWARM_COMPUTE"); if (e && !strcmp(e, "0")) return; }
    const char *lists[2] = { getenv("DM_VK_PSO_SEED"), getenv("DM_VK_PSO_RECORD") }; static nvn_hmap seen = {.klen = 8};
    int n = 0, built = 0, missing = 0, failed = 0; uint64_t t0 = mono_ns();
    for (int l = 0; l < 2; l++) { char p[1100]; if (!lists[l] || !*lists[l] || snprintf(p, sizeof p, "%s.cs", lists[l]) >= (int)sizeof p) continue;
        FILE *f = fopen(p, "rb"); if (!f) continue;
        char mg[4]; uint8_t k[8];
        if (fread(mg, 1, 4, f) == 4 && !memcmp(mg, PSO_MAGIC_C, 4)) while (fread(k, 8, 1, f) == 1) {
            if (nvn_hm_get(&seen, k)) continue;
            nvn_hm_put(&seen, k, (void *)1); n++;
            prewarm_pace_play();   // (once the game plays: one build after each present, like the graphics pipelines)
            pthread_mutex_lock(&g_pw.mu);   // (prog_layout is not for two threads: the graphics prewarm's lock)
            vshader *cs = shader_get(k, 5); vprog *pr = cs ? calloc(1, sizeof *pr) : NULL; VkPipelineLayout pl = VK_NULL_HANDLE;
            if (pr) { pr->sh[5] = cs; pl = prog_layout(pr, 1); }   // (a program of its own, kept: its layout lives as long as the cache entry is useful)
            pthread_mutex_unlock(&g_pw.mu);
            if (!cs || !pl) { missing++; continue; }
            VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs->mod, .pName = "main" }, .layout = pl };
            VkPipeline pp = VK_NULL_HANDLE;
            if (vkCreateComputePipelines(g_dev, g_pcache, 1, &ci, NULL, &pp)) failed++; else { vkDestroyPipeline(g_dev, pp, NULL); built++; } }
        fclose(f); }
    if (n) VLOG("prewarm: %d compute pipelines listed: %d built into the pipeline cache, %d failed, %d with a shader missing, %.1f s", n, built, failed, missing, (mono_ns() - t0) / 1e9);
}
static void *prewarm_run(void) {
    uint64_t t0 = mono_ns();
    prewarm_compute();   // (first: few, and the open world's first frames need them before most graphics pipelines)
    if (!g_eds) return prewarm_static();   // the shared lists leave out the draw state this driver needs built in: this phone's own list instead
    pso_rec *rs = NULL; int n = 0;
    int ns = pso_rec_load(getenv("DM_VK_PSO_SEED"), &rs, &n), nr = pso_rec_load(getenv("DM_VK_PSO_RECORD"), &rs, &n);
    if (!n) { free(rs); return NULL; }
    dm_boot_mark("prewarm: pipeline lists read");
    g_pw.rs = rs; g_pw.n = n; g_pw.ns = ns; atomic_store(&g_pw_total, n);   // (the shipped list first: pso_rec_load appends)
    int nt = prewarm_threads(); pthread_t th[16]; int started = 0;
    // DM_VK_PREWARM=cache: every pipeline built once into the driver's pipeline cache (saved below) and destroyed at once, instead of kept (opt-in: on the S25+ it
    //   saved nothing, the heap grew the same with 365 pipelines as with 1157 kept, 2026-11-20)
    { const char *e = getenv("DM_VK_PREWARM"); g_pw.cache_only = e && !strcmp(e, "cache"); if (g_pw.cache_only) VLOG("prewarm: into the pipeline cache only, not kept (DM_VK_PREWARM=cache)"); }
    g_pw.keep = pw_cap_keep(); g_pw.compact = pw_compact_on();   // (read once here: pso_cap's first call logs, not for several threads)
    for (int i = 1; i < nt; i++) if (!pthread_create(&th[started], NULL, prewarm_worker, (void *)1)) started++;
    prewarm_worker(NULL);
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
    VLOG("prewarm: %d pipelines listed (%d shipped, %d recorded here): %d built%s, %d already there, %d failed, %d with a shader missing, %d shipped ones with formats this GPU lacks, %.1f s on %d thread(s)%s (paused %d times for low memory)",
         n, ns, nr, atomic_load(&g_pw.built), g_pw.cache_only ? " (cache only)" : "", atomic_load(&g_pw.had), atomic_load(&g_pw.failed), atomic_load(&g_pw.noshader), atomic_load(&g_pw.unfit), (mono_ns() - t0) / 1e9, started + 1,
         prewarm_all() ? ", before the game" : "", atomic_load(&g_pw.waits));
    if (g_pw.keep && (atomic_load(&g_pw.cached) || atomic_load(&g_pw.skipped)))
        VLOG("prewarm: %d kept (the cap: at most %d live pipelines), %d past it only put into the pipeline cache, %d skipped (already in it; DM_VK_PREWARM_CAP=0 keeps every one)",
             atomic_load(&g_pw.kept), pso_cap(), atomic_load(&g_pw.cached), atomic_load(&g_pw.skipped));
    if (atomic_load(&g_pw.dups)) VLOG("prewarm: %d entries of the list recorded here repeat earlier ones", atomic_load(&g_pw.dups));
    VLOG("prewarm timers (summed over its %d thread(s)): %.0f ms waiting for the program lock, %.0f ms loading shaders, %.0f ms making layouts, %.0f ms building pipelines",
         started + 1, atomic_load(&g_pw.t_lock) / 1e6, atomic_load(&g_pw.t_shader) / 1e6, atomic_load(&g_pw.t_layout) / 1e6, atomic_load(&g_pw.t_pipe) / 1e6);
    dm_boot_mark("prewarm: every listed pipeline handled");
    free(rs); g_pw.rs = NULL;
    atomic_store(&g_pw_complete, 1);
    return NULL;   // (the cache is saved by prewarm_main, after the game is let go)
}
static atomic_int g_pw_ended;   // the prewarm thread has returned (simdev t_vkdev's start-up timing)
static void *prewarm_main(void *arg) {
    DM_THREAD_NAME("dm-prewarm"); (void)arg;
    prewarm_run();
    prewarm_finished();   // the game goes on first (perf-1124 S1), then the lists and the cache are written, if anything changed
    pw_compact_lists();
    pcache_save_new(); dm_boot_mark("prewarm: pipeline cache handled");
    atomic_store(&g_pw_ended, 1); return NULL;
}
// the end of mtl_init: the pipeline cache read, and the prewarm started when there is a list (the game's first submission waits for it: prewarm_wait)
static void prewarm_start(void) {
    pcache_open();
    atomic_store(&g_pw_warm, 0); atomic_store(&g_pw_complete, 0); atomic_store(&g_pw_hold_cnt, 0); g_pw_hold_n = 0;
    if (getenv("DM_VK_PSO_SEED") || getenv("DM_VK_PSO_RECORD")) {
        if (prewarm_all()) {
            // S3: the stamp of the last complete prewarm against this start's lists, cache and driver
            pw_stamp now; pw_stamp_now(&now); now.cache = g_pcache_disk;
            char p[1200], text[512] = ""; FILE *f = pw_stamp_path(p, sizeof p) ? fopen(p, "r") : NULL; if (f) { if (!fgets(text, sizeof text, f)) text[0] = 0; fclose(f); }
            int warm = pw_stamp_match(text, &now), skip = pw_misses_skip();
            int mode = pw_hold_parse(getenv("DM_VK_PREWARM_HOLD"), skip ? PW_HOLD_NONE : 600);
            pw_stamp_clear();   // (written again when this start's prewarm is complete: a start that ends before it leaves none)
            if (!warm || mode == PW_HOLD_ALL) atomic_store(&g_pw_hold, 1);
            else { atomic_store(&g_pw_warm, 1); if (mode > 0) { g_pw_hold_n = mode; atomic_store(&g_pw_hold, 1); } }
            VLOG("prewarm: %s: %s", !text[0] ? "no stamp of a complete prewarm" : warm ? "the stamp matches (every listed pipeline is in the pipeline cache)" : "the stamp differs (lists, cache, driver, app or shaders changed)",
                 !warm || mode == PW_HOLD_ALL ? "the game waits for every listed pipeline" : mode > 0 ? "the game waits for the first list entries only (a pipeline missing in play is a hitch here)"
                 : "the game goes on at once, the pipelines are built from the cache meanwhile (a missing one's draws are skipped until it is built)");
            if (warm && mode > 0) VLOG("prewarm: the first %d list entries before the game (DM_VK_PREWARM_HOLD=all | none | <count>)", mode);
        }
        pthread_t t; if (!pthread_create(&t, NULL, prewarm_main, NULL)) pthread_detach(t); else { atomic_store(&g_pw_hold, 0); atomic_store(&g_pw_ended, 1); }
    }
}
// Builds the pipeline for key k (no lookup: pso_get and the prewarm do that); ss: the draw state built in, on a driver without extended dynamic state (else NULL)
// A vertex attribute's format as the shader reads it. Vulkan wants the format's numeric type to match the shader input's (VUID-VkGraphicsPipelineCreateInfo-Input-08733;
//   the driver is free to do anything otherwise); NVIDIA's hardware, which the game was made for, hands a 32-bit attribute's bits to the shader unchanged whatever its
//   type. A 32-bit integer format under a float input therefore becomes the float format of the same size, and a 32-bit float format under an integer input the
//   integer one: the same bits reach the shader, by the spec. Smaller integer formats have no bit-for-bit match (left as they are, logged once)
// The vertex module a pipeline gets (pso_make) and its inputs: the GL-depth variant on a driver without depth clip control while the game's depth is GL's
static int vs_gl(const vshader *vs, int depth_zero) { return vs && !g_clipctl && !depth_zero && vs->hasgl && vs->nglin; }   // (hasgl: glmod may be released, shmod_release)
static const spv_input *vs_inputs(const vshader *vs, int depth_zero, int *n) { if (vs_gl(vs, depth_zero)) { *n = vs->nglin; return vs->glin; } *n = vs ? vs->nin : 0; return vs ? vs->in : NULL; }
static VkFormat attr_fmt_for(const vshader *vs, int depth_zero, uint32_t loc, VkFormat f) {
    int nin; const spv_input *in = vs_inputs(vs, depth_zero, &nin);
    int cls = -1; for (int i = 0; i < nin; i++) if (in[i].location == loc) { cls = in[i].sclass; break; }
    if (cls < 0) return f;
    int fi = f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32G32B32A32_SFLOAT ? (int)(f - VK_FORMAT_R32_UINT) : -1;   // R32 .. R32G32B32A32, each UINT / SINT / SFLOAT
    if (fi >= 0) { int base = fi / 3 * 3, kind = fi % 3;   // kind 0 uint, 1 sint, 2 float
        int want = cls == 0 ? 2 : cls == 1 ? 0 : 1;
        if (kind != want) { static int said; if (!said++) VLOG("vertex attributes: a 32-bit %s format under a %s shader input is read as %s (the same bits, as the Switch's GPU hands them over)",
                                                                 kind == 2 ? "float" : "integer", cls == 0 ? "float" : "integer", want == 2 ? "float" : "integer");
                            return (VkFormat)(VK_FORMAT_R32_UINT + base + want); } }
    else if (cls == 0) { static int said; int ints = (f >= VK_FORMAT_R8_UINT && f <= VK_FORMAT_R8_SINT) || (f >= VK_FORMAT_R8G8_UINT && f <= VK_FORMAT_R8G8_SINT) ||
            (f >= VK_FORMAT_R8G8B8A8_UINT && f <= VK_FORMAT_R8G8B8A8_SINT) || (f >= VK_FORMAT_R16_UINT && f <= VK_FORMAT_R16_SINT) || (f >= VK_FORMAT_R16G16_UINT && f <= VK_FORMAT_R16G16_SINT) ||
            (f >= VK_FORMAT_R16G16B16A16_UINT && f <= VK_FORMAT_R16G16B16A16_SINT);
        if (ints && !said++) VLOG("%svertex attributes: an 8 / 16-bit integer format %d under a float shader input (location %u): %s", intbits_on() ? "" : "[RED] ", (int)f, loc,
                                  intbits_on() ? "read as integers and bitcast in the vertex shader (intbits_mod)" : "no format gives the same bits; left as it is (DM_VK_VFMT_INTBITS=0)"); }
    return f;
}
static VkPipeline pso_make_(vprog *pr, pkey k, const sst *ss);
static VkPipeline pso_make(vprog *pr, pkey k, const sst *ss) {   // (the shaders' modules alive from here until the driver returned the pipeline: shmod_release)
    pthread_rwlock_rdlock(&g_shmod_rw);
    int ok = 1; for (int i = 0; i < 5; i++) ok &= shmod_ensure(pr->sh[i]);
    VkPipeline p = ok ? pso_make_(pr, k, ss) : VK_NULL_HANDLE;
    pthread_rwlock_unlock(&g_shmod_rw);
    return p;
}
static VkPipeline pso_make_(vprog *pr, pkey k, const sst *ss) {
    vshader *vs = pr->sh[0], *fs = pr->sh[1]; VkPipeline p = VK_NULL_HANDLE;
    // a capture drawn as points (a key recorded while that route was on, or now): only with the route on and its shaders made. A recorded one built ahead at a start
    //   with the route off had no point modules: Adreno's driver crashed in vkCreateGraphicsPipelines on the null module
    if (k.tfpts && (g_tf_pts <= 0 || k.tfpts > 4 || !vs || !vs->ptsmod || !tfp_init() || !g_tfp_fs[k.tfpts - 1])) return VK_NULL_HANDLE;
    if (!g_eds && !ss) { RED_ONCE(110, "a pipeline without its draw state on a driver without extended dynamic state (vertex %s): not made", vs ? vs->key : "-"); return VK_NULL_HANDLE; }
    if (!g_clipctl && !k.depth_zero && vs && !vs->glmod) RED_ONCE(27, "GL depth mode without VK_EXT_depth_clip_control and no translated depth fallback for vertex shader %s", vs->key);
    VkVertexInputBindingDescription vb[17]; VkVertexInputAttributeDescription va[32]; VkVertexInputBindingDivisorDescriptionEXT vd[16]; uint32_t nvb = 0, nva = 0, nvd = 0; uint32_t have = 0;
    for (uint32_t loc = 0; loc < 32; loc++) {
        if (!k.attr[loc].used) continue;
        va[nva++] = (VkVertexInputAttributeDescription){ loc, k.attr[loc].bind, attr_fmt_for(vs, k.depth_zero, loc, (VkFormat)k.attr[loc].fmt), k.attr[loc].bind == 16 ? 0 : k.attr[loc].off };
        uint32_t b = k.attr[loc].bind; if (have >> b & 1) continue; have |= 1u << b;
        vb[nvb++] = (VkVertexInputBindingDescription){ b, 0, b != 16 && k.attr[loc].inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX };
        if (b != 16 && k.attr[loc].inst && k.attr[loc].div > 1) { if (g_divisor) vd[nvd++] = (VkVertexInputBindingDivisorDescriptionEXT){ b, k.attr[loc].div }; else RED_ONCE(26, "instance divisor %u needs VK_EXT_vertex_attribute_divisor", k.attr[loc].div); }
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT vds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, .vertexBindingDivisorCount = nvd, .pVertexBindingDivisors = vd };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nvd ? &vds : NULL, .vertexBindingDescriptionCount = nvb, .pVertexBindingDescriptions = vb, .vertexAttributeDescriptionCount = nva, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = k.topo == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : k.topo == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, .negativeOneToOne = !k.depth_zero };   // NVN's GL depth mode: z in [-w, w]
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = g_clipctl ? &dcc : NULL, .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .depthClampEnable = k.clamp, .rasterizerDiscardEnable = k.discard,
        .polygonMode = k.poly == 1 ? VK_POLYGON_MODE_LINE : k.poly == 2 ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL, .depthBiasEnable = VK_TRUE, .lineWidth = 1 };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = (VkSampleCountFlagBits)(k.samples ? k.samples : 1), .alphaToCoverageEnable = k.a2c };
    VkPipelineDepthStencilStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState cb[8]; uint32_t ncb = 0; VkFormat cf[8];
    for (int i = 0; i < 8; i++) { if (k.cfmt[i]) ncb = (uint32_t)i + 1; }
    for (uint32_t i = 0; i < ncb; i++) {
        cf[i] = (VkFormat)k.cfmt[i]; memset(&cb[i], 0, sizeof cb[i]);
        uint32_t m = (k.chmask >> (4 * i)) & 0xf;
        if (fs && fs->color_known && !(fs->color_mask >> i & 1)) m = 0;   // outputs the shader does not write keep their content (nvn_mtl.m pso_desc)
        if (k.tfpts) m = 0xf;   // (a capture's targets: every channel)
        if (k.b10 >> i & 1) m &= 7;   // R11G11B10F's RGBA16F stand-in (DM_VK_B10_AS_RGBA16): the packed format has no alpha to write (the stand-in's stays 1)
        cb[i].colorWriteMask = m;
        if (k.blend_en >> i & 1 && m && !fmt_int_class(cf[i])) { cb[i].blendEnable = VK_TRUE;   // integer targets never blend (Vulkan: undefined)
            cb[i].srcColorBlendFactor = bfac_dev(k.blend[i].src_rgb); cb[i].dstColorBlendFactor = bfac_dev(k.blend[i].dst_rgb); cb[i].srcAlphaBlendFactor = bfac_dev(k.blend[i].src_a); cb[i].dstAlphaBlendFactor = bfac_dev(k.blend[i].dst_a);
            cb[i].colorBlendOp = bop(k.blend[i].eq_rgb); cb[i].alphaBlendOp = bop(k.blend[i].eq_a);
            // ... and blending reads a target without alpha as alpha 1 (Vulkan's rule, the Switch's too): the stand-in's own alpha is never read
            if (k.b10 >> i & 1) { VkBlendFactor *f4[4] = { &cb[i].srcColorBlendFactor, &cb[i].dstColorBlendFactor, &cb[i].srcAlphaBlendFactor, &cb[i].dstAlphaBlendFactor };
                for (int q = 0; q < 4; q++) { VkBlendFactor f = *f4[q], g = f == VK_BLEND_FACTOR_DST_ALPHA ? VK_BLEND_FACTOR_ONE : f == VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA ? VK_BLEND_FACTOR_ZERO
                                                  : f == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE && q < 2 ? VK_BLEND_FACTOR_ZERO : f;   // (saturate: min(source alpha, 1 - 1) for colour)
                    if (g != f) { *f4[q] = g; b10_note("blending that reads the target's alpha: read as 1 (the packed format has none)", NULL, (int)f); } } } }
    }
    // Optional Vulkan 1.0 features the device may lack (mtl_init): the state that needs one is left out, never passed on
    if (!g_f_indblend && ncb > 1) {   // without independentBlend every target takes the same state: the first written target's (no blending if any target is an integer one)
        uint32_t t = 0; while (t < ncb && !cb[t].colorWriteMask) t++;
        if (t < ncb) { int differ = 0, anyint = 0; for (uint32_t i = 0; i < ncb; i++) { if (memcmp(&cb[i], &cb[t], sizeof cb[t])) differ = 1; if (cf[i] && fmt_int_class(cf[i])) anyint = 1; }
            if (differ) { VkPipelineColorBlendAttachmentState one = cb[t]; if (anyint) one.blendEnable = VK_FALSE; for (uint32_t i = 0; i < ncb; i++) cb[i] = one;
                RED_ONCE(131, "targets blended differently on a device without independentBlend (vertex %s): every target as the first written one", vs ? vs->key : "-"); } }
    }
    // R11G11B10F's RGBA16F stand-in (DM_VK_B10_AS_RGBA16, see b10_on): what the draw writes into it made what the packed format stores (frag_module_ qm), where the
    //   shader's output is what gets stored: no blending, or blending by min / max (rounding first or after gives the same there). A blended sum cannot be rounded
    //   in a shader: it is kept as the blend makes it (logged once)
    uint32_t qm = 0;
    if (k.b10 && fs && !k.tfpts && !k.discard) for (uint32_t i = 0; i < ncb; i++) {
        uint8_t cls = fs->out_cls[k.ds_swap && i <= 1 ? i ^ 1 : i];   // (the variant with outputs 0 and 1 swapped writes target i from output i ^ 1)
        if (!(k.b10 >> i & 1) || !(cb[i].colorWriteMask & 7) || cls != 0) continue;
        if (!cb[i].blendEnable || cb[i].colorBlendOp == VK_BLEND_OP_MIN || cb[i].colorBlendOp == VK_BLEND_OP_MAX) qm |= 1u << i;
        else b10_note("a draw blended into the stand-in: the blend's result is kept as it is (R11G11B10F would store a sum below 0 as 0, one past 65024 as 65024)", NULL, (int)cb[i].colorBlendOp);
    }
    VkPipelineShaderStageCreateInfo st[5]; uint32_t nst = 0;
    VkShaderModule ibm = VK_NULL_HANDLE;   // the vertex module with integer-format inputs bitcast (intbits_mod), of the variant this pipeline draws with
    if (intbits_on() && vs && !k.tfpts && !k.discard) { uint32_t um = 0, sm = 0; int vgl = vs_gl(vs, k.depth_zero), nvi; const spv_input *vi = vs_inputs(vs, k.depth_zero, &nvi);
        for (int i = 0; i < nvi; i++) { uint32_t loc = vi[i].location; if (loc >= 32 || vi[i].sclass != 0 || !k.attr[loc].used || k.attr[loc].bind == 16) continue;
            int ik = intbits_kind((VkFormat)k.attr[loc].fmt); if (ik == 1) um |= 1u << loc; else if (ik == 2) sm |= 1u << loc; }
        if (um | sm) ibm = intbits_mod(vs, vgl, um, sm); }
    for (int i = 0; i < 5; i++) if (pr->sh[i] && !(i == 1 && (k.discard || k.tfpts))) st[nst++] = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = pr->sh[i]->vkstage, .module = i == 1 ? frag_module_(pr, k.cfmt, k.ds_swap, qm) : i == 0 && k.tfpts && pr->sh[0]->ptsmod ? pr->sh[0]->ptsmod : i == 0 && k.discard && pr->sh[0]->tfmod ? pr->sh[0]->tfmod : i == 0 && ibm ? ibm : i == 0 && !g_clipctl && !k.depth_zero && pr->sh[0]->glmod ? pr->sh[0]->glmod : pr->sh[i]->mod, .pName = "main" };
    if (k.tfpts && k.tfpts <= 4) st[nst++] = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = g_tfp_fs[k.tfpts - 1], .pName = "main" };   // (the points' outputs into the targets)
    // the y flip (spv_flip_y): -1 for the upper-left origin, 1 otherwise; a module without the constant (capture variants) ignores the entry
    static const VkSpecializationMapEntry yfe = { SPV_FLIP_Y_SPEC_ID, 0, 4 }; const float yfv = k.yflip ? -1.0f : 1.0f; const VkSpecializationInfo yfi = { 1, &yfe, 4, &yfv };
    if (g_yflip) for (uint32_t i = 0; i < nst; i++) if (st[i].stage == VK_SHADER_STAGE_VERTEX_BIT) st[i].pSpecializationInfo = &yfi;
    if (!g_f_clamp && rs.depthClampEnable) { rs.depthClampEnable = VK_FALSE; RED_ONCE(132, "depth clamp on a device without it: left off (geometry past the far plane is clipped)"); }
    if (!g_f_nonsolid && rs.polygonMode != VK_POLYGON_MODE_FILL) { rs.polygonMode = VK_POLYGON_MODE_FILL; RED_ONCE(133, "line / point fill on a device without fillModeNonSolid: filled"); }
    int logic_en = k.tfpts ? 0 : k.logic_en;
    if (!g_f_logic && logic_en) { logic_en = 0; RED_ONCE(134, "a logic op (%u) on a device without logicOp: plain writes", (unsigned)k.logic_op); }
    // a geometry / tessellation stage the device does not have: the program is skipped
    for (int i = 0; i < 5; i++) if (pr->sh[i] && (pr->sh[i]->lacks & (SPV_NEED_GEOM | SPV_NEED_TESS))) { RED_ONCE(135, "a program with a geometry / tessellation stage this device lacks (shader %s): not drawn", pr->sh[i]->key); return VK_NULL_HANDLE; }
    VkPipelineColorBlendStateCreateInfo bs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .logicOpEnable = logic_en, .logicOp = (VkLogicOp)k.logic_op,
                                           .attachmentCount = ncb, .pAttachments = cb };
    VkDynamicState dyn[24]; uint32_t nd = 0;
    VkDynamicState base[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP };
    unsigned nbase = g_eds ? sizeof base / sizeof *base : 7;   // (without extended dynamic state: Vulkan 1.0's dynamic states only, the first 7)
    for (unsigned i = 0; i < nbase; i++) dyn[nd++] = base[i];
    if (!g_eds) {   // the draw's state built in (sst, filled by pso_get as the draw would set it)
        rs.cullMode = ss->cull; rs.frontFace = (VkFrontFace)ss->front; ia.topology = (VkPrimitiveTopology)ss->topo;
        ds.depthTestEnable = ss->dt; ds.depthWriteEnable = ss->dw; ds.depthCompareOp = (VkCompareOp)ss->dcmp; ds.stencilTestEnable = ss->st;
        ds.front = (VkStencilOpState){ (VkStencilOp)ss->sop[0][0], (VkStencilOp)ss->sop[0][1], (VkStencilOp)ss->sop[0][2], (VkCompareOp)ss->sop[0][3], 0, 0, 0 };
        ds.back = (VkStencilOpState){ (VkStencilOp)ss->sop[1][0], (VkStencilOp)ss->sop[1][1], (VkStencilOp)ss->sop[1][2], (VkCompareOp)ss->sop[1][3], 0, 0, 0 };
        for (uint32_t i = 0; i < nvb; i++) vb[i].stride = vb[i].binding < 16 ? ss->stride[vb[i].binding] : 0;
    }
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = nd, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = ncb, .pColorAttachmentFormats = cf,
        .depthAttachmentFormat = (VkFormat)k.dfmt, .stencilAttachmentFormat = (VkFormat)k.sfmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = nst, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &bs, .pDynamicState = &dy, .layout = prog_layout(pr, 0) };
    if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(ncb, cf, k.dfmt ? k.dfmt : k.sfmt, k.samples, 0); gi.subpass = 0; }
#ifdef __ANDROID__
    // the last line before a driver crash names the shaders it was compiling. One line per pipeline was ~4000 line-buffered writes in every start's wait (perf-1124
    //   S6): only with DM_VK_PIPE_LOG=1, or at the start after one that crashed with the device up (boot_state: DM_LAST_CRASH_DEVICE), so a crash that comes
    //   back still names its shaders. DM_VK_PIPE_LOG=0: never
    static int plog = -1; if (plog < 0) { const char *e = getenv("DM_VK_PIPE_LOG"); plog = e && *e ? atoi(e) != 0 : getenv("DM_LAST_CRASH_DEVICE") != NULL;
        VLOG("pipelines: %s", plog ? "one line each as they are built (DM_VK_PIPE_LOG=1, or the last start crashed)" : "not logged one by one (DM_VK_PIPE_LOG=1 logs each)"); }
    if (plog) { static _Atomic int npipe; VLOG("pipeline %d: vertex %s fragment %s", atomic_fetch_add(&npipe, 1) + 1, vs->key, fs ? fs->key : "-"); }
#endif
    // PowerVR (BXM-8-256, driver 0x5d9576) crashes in vkCreateGraphicsPipelines when a pipeline cache is passed and the render pass has an unused colour slot
    //   (VK_ATTACHMENT_UNUSED: a target with a gap): found and isolated by the Persona 5 Royal Android port (Brandon04Gonzalez/p5r-android, MIT). Such a pipeline
    //   is made without the cache on PowerVR; every other pipeline, and every other GPU, keeps it
    VkPipelineCache pc_ = g_pcache;
    if (quirk_default(Q_PCACHE_GAPS_OFF, &g_gpuid) && !g_dynren) for (uint32_t i = 0; i < ncb; i++) if (!cf[i]) { pc_ = VK_NULL_HANDLE; break; }
    VkResult r = vkCreateGraphicsPipelines(g_dev, pc_, 1, &gi, NULL, &p);
    if (r == VK_INCOMPLETE && p) { RED_ONCE(107, "graphics pipeline creation answered VK_INCOMPLETE (out of spec; Mali, Adreno 5xx): the pipeline it made is used"); r = VK_SUCCESS; }
    if (r) { VLOG("[RED] graphics pipeline (vertex %s fragment %s) failed: %d", vs->key, fs ? fs->key : "-", r); return VK_NULL_HANDLE; }
    return p;
}
// vbm: the vertex bindings the pipeline reads (bit 16: the zeros of unbound attributes), so the draw binds and looks up only those
static VkPipeline pso_get(vprog *pr, VkPrimitiveTopology topo, uint32_t *vbm) {
    vshader *vs = pr->sh[0], *fs = pr->sh[1];
    pkey k; memset(&k, 0, sizeof k);
    for (int i = 0; i < 5; i++) k.sh[i] = pr->sh[i];
    memcpy(k.cfmt, P.cfmt, sizeof k.cfmt); k.dfmt = P.hd ? P.dfmt : 0; k.sfmt = P.hs ? P.sfmt : 0; k.samples = (uint8_t)P.samples; k.b10 = (uint8_t)P.b10;
    k.chmask = S.chmask.mask; k.blend_en = (uint8_t)S.color.blend_en; for (int i = 0; i < 8; i++) if (k.cfmt[i]) k.blend[i] = S.blend[i];
    k.a2c = S.ms.a2c; k.poly = S.poly.mode == NVN_POLYGON_MODE_LINE ? 1 : S.poly.mode == NVN_POLYGON_MODE_POINT ? 2 : 0; k.clamp = (uint8_t)!!S.depth_clamp; k.topo = (uint8_t)topo_class(topo);
    k.logic_en = S.color.logic_op != NVN_LOGIC_OP_COPY; k.logic_op = (uint8_t)S.color.logic_op; k.depth_zero = (uint8_t)atomic_load(&g_depth_zero); k.yflip = (uint8_t)(g_yflip && !atomic_load(&g_origin_ll)); k.discard = (uint8_t)!!S.rast_discard;
    if (P_tf && g_tf_pts > 0) { k.discard = 0; k.tfpts = (uint8_t)P.ncolor; }   // (a capture drawn as points: tf_points.h)
    k.ds_swap = (uint8_t)g_ds_swap;
    int nvin; const spv_input *vin = vs_inputs(vs, k.depth_zero, &nvin);   // (the inputs of the vertex module this pipeline gets)
    for (int i = 0; i < nvin; i++) {
        uint32_t loc = vin[i].location; if (loc >= 32) continue;
        k.attr[loc].used = 1;
        if ((int)loc < S.nattr && S.attr[loc].fmt) { int st = S.attr[loc].stream; VkFormat f = vfmt_for(S.attr[loc].fmt, vin[i].sclass);
            if (f && st < 16) { k.attr[loc].fmt = f; k.attr[loc].off = S.attr[loc].offset; k.attr[loc].bind = (uint8_t)st; k.attr[loc].inst = S.stream[st].divisor != 0; k.attr[loc].div = S.stream[st].divisor; continue; }
            RED_ONCE(25, "vertex attribute format %#x not mapped", S.attr[loc].fmt); }
        k.attr[loc].fmt = vin[i].sclass == 1 ? VK_FORMAT_R32G32B32A32_UINT : vin[i].sclass == 2 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT; k.attr[loc].bind = 16;   // unbound: reads 0
    }
    { uint32_t m = 0; for (int loc = 0; loc < 32; loc++) if (k.attr[loc].used) m |= 1u << k.attr[loc].bind; *vbm = m; }
    if (!g_eds) {   // the state draw_ sets per draw where the driver lets it, as the same values (and the same rules: depth / stencil only with such an attachment)
        sst s; memset(&s, 0, sizeof s);
        s.cull = (uint8_t)(S.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : S.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);
        int ccw = S.poly.front == NVN_FRONT_FACE_CCW; if (atomic_load(&g_origin_ll)) ccw = !ccw;
        s.front = (uint8_t)(ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE); s.topo = (uint8_t)topo;
        s.dt = (uint8_t)(P.hd && S.ds.dt); s.dw = (uint8_t)(s.dt && S.ds.dw); s.dcmp = s.dt && S.ds.df && S.ds.df - 1 <= 7 ? (uint8_t)(S.ds.df - 1) : 7;
        s.st = (uint8_t)(P.hs && S.ds.st);
        if (s.st) {   // (a value outside the enums: KEEP / ALWAYS, as a driver would never be handed)
            int f[2][4] = { { S.ds.fail_f, S.ds.zpass_f, S.ds.zfail_f, S.ds.sf_f }, { S.ds.fail_b, S.ds.zpass_b, S.ds.zfail_b, S.ds.sf_b } };
            for (int fc = 0; fc < 2; fc++) for (int j = 0; j < 4; j++) { int v = f[fc][j] - 1; s.sop[fc][j] = (uint8_t)(v >= 0 && v <= 7 ? v : j == 3 ? 7 : 0); }
        } else for (int fc = 0; fc < 2; fc++) s.sop[fc][3] = 7;
        for (int loc = 0; loc < 32; loc++) if (k.attr[loc].used && k.attr[loc].bind < 16) s.stride[k.attr[loc].bind] = S.stream[k.attr[loc].bind].stride;
        sst_fold(&s);
        return pso_lookup_s(pr, &k, &s, 1);
    }
    return pso_lookup(pr, &k, 1);
}

// ================= Descriptors =================
static uint32_t g_draw_binst, g_dispatch_dim[3];
static VkBuffer g_c0_buf; static VkDeviceSize g_c0_off;
// ================= Uniform updates deferred inside a pass (from the Arceus engine; nvn_mtl.m mtl_sub_update) =================
// Vulkan cannot copy into a buffer inside a render pass, so each nvnCommandBufferUpdateUniformBuffer used to end the pass: on a tiler every end stores all attachments to
// memory and the next begin loads them back (Legends: Arceus: about 59 passes a frame). An update inside an open pass is now kept in the arena instead; a draw after it whose
// uniform binding overlaps it reads a shadow made of the deferred updates only (latest first per byte), and only when they cover the whole bound range, so the shadow is exactly
// what the GPU would read. The real copies are recorded in order when the pass ends, so later passes and the CPU see them as before. The pass still ends before a draw that would
// need more (ubd_conflict): a binding covered only in part, or the deferred data read as vertices, indices, indirect arguments or storage. DM_VK_UB_DEFER=0 turns it off
typedef struct { uint64_t lo, hi; VkBuffer buf; VkDeviceSize off; const uint8_t *map; } ubd_ent;
enum { UBD_MAX = 512 };
static ubd_ent g_ubd[UBD_MAX];
static uint64_t g_ubd_deferred, g_ubd_conflicts;   // updates deferred, passes ended before a draw that needed more (mtl_ubd_stats)
static int ubd_on(void) { static int v = -1; if (v < 0) { v = arceus_pass_default("DM_VK_UB_DEFER", "0", "1"); VLOG("uniform updates inside a pass: %s", v ? "deferred to its end (DM_VK_UB_DEFER=0: they end the pass)" : "end the pass"); } return v; }
static int ubd_hits(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_nubd; i++) if (lo < g_ubd[i].hi && g_ubd[i].lo < hi) return 1; return 0; }
static int ubd_covers(uint64_t lo, uint64_t hi) {   // the union of the deferred ranges holds all of [lo, hi), swept from lo
    for (uint64_t at = lo; at < hi;) { uint64_t next = at;
        for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= at && g_ubd[i].hi > next) next = g_ubd[i].hi;
        if (next == at) return 0; at = next; }
    return 1;
}
static void ubd_read(uint64_t lo, uint64_t n, void *out) {   // [lo, lo+n) as the deferred updates leave it, in order (ubd_covers holds)
    for (int i = 0; i < g_nubd; i++) { const ubd_ent *e = &g_ubd[i]; uint64_t a = e->lo > lo ? e->lo : lo, b = e->hi < lo + n ? e->hi : lo + n;
        if (a < b) memcpy((uint8_t *)out + (a - lo), e->map + (a - e->lo), (size_t)(b - a)); }
}
// The shadow of uniform range [lo, hi) (ubd_covers holds): one deferred update holding all of it with nothing later over it is read in place; otherwise the updates are laid into
// the arena in order. Bound at the shadow's own length
static void ubd_shadow(uint64_t lo, uint64_t hi, VkDeviceSize al, VkBuffer *b, VkDeviceSize *o, VkDeviceSize *n) {
    for (int i = g_nubd - 1; i >= 0; i--) { const ubd_ent *e = &g_ubd[i];
        if (e->lo <= lo && hi <= e->hi) { VkDeviceSize at = e->off + (lo - e->lo);
            if (al && at % al == 0) { *b = e->buf; *o = at; *n = e->hi - lo; return; }
            break; }
        if (lo < e->hi && e->lo < hi) break;   // a later update over part of it: composed below
    }
    uint8_t *d = arena_alloc(g_sub, hi - lo, b, o); ubd_read(lo, hi - lo, d); *n = hi - lo;
}
// The deferred copies, in order, once the pass has ended: what the pass's draws read is done before they land, and overlapping ones are ordered
static void ubd_flush(void) {
    int n = g_nubd; g_nubd = 0; if (!n) return;
    VkCommandBuffer cb = g_sub->cb; barrier_before(cb);
    for (int i = 0; i < n; i++) { const ubd_ent *e = &g_ubd[i]; VkBuffer db; VkDeviceSize dof;
        for (int j = 0; j < i; j++) if (e->lo < g_ubd[j].hi && g_ubd[j].lo < e->hi) { barrier_all(cb); break; }
        if (!pool_buf(e->lo, e->hi - e->lo, &db, &dof, NULL)) continue;
        VkBufferCopy c = { e->off, dof, e->hi - e->lo }; vkCmdCopyBuffer(cb, e->buf, db, 1, &c); wb_note(e->lo, e->hi - e->lo); }
    barrier_all(cb);
}
static uint32_t ubo_word(int stage, int reg, int off) {   // a bindless handle the game keeps in a uniform buffer: read from the pool now (nvn_mtl.m ubo_read)
    if (stage < 0 || stage >= 6 || reg < 0 || reg >= 16) return 0;
    const p_bind_buf *u = &S.ubo[stage][reg]; uint32_t h = 0;
    if (u->addr && (uint64_t)off + 4 <= u->size) {
        if (g_nubd && ubd_hits(u->addr + off, u->addr + off + 4) && ubd_covers(u->addr + off, u->addr + off + 4)) ubd_read(u->addr + off, 4, &h);   // a deferred update's data
        else memcpy(&h, (const void *)(uintptr_t)(u->addr + off), 4); }
    return h;
}
// g_nopush: the writes of all stages of a draw, gathered here and written into one set (desc_flush)
static struct { VkWriteDescriptorSet w[320]; VkDescriptorBufferInfo bi[320]; VkDescriptorImageInfo ii[320]; VkBufferView bv[320]; uint32_t n; } D;
static VkDescriptorSet desc_set_alloc(VkDescriptorSetLayout dsl) {
    sub_t *s = g_sub;
    for (;;) {
        if (s->cdp == s->ndp) {
            VkDescriptorPoolSize ps[] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2048 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2048 }, { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 2048 },
                { VK_DESCRIPTOR_TYPE_SAMPLER, 2048 }, { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2048 } };
            VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2048, .poolSizeCount = 8, .pPoolSizes = ps };
            VkDescriptorPool p; VKCHECK(vkCreateDescriptorPool(g_dev, &pci, NULL, &p));
            s->dp = realloc(s->dp, sizeof *s->dp * (size_t)(s->ndp + 1)); s->dp[s->ndp++] = p;
        }
        VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = s->dp[s->cdp], .descriptorSetCount = 1, .pSetLayouts = &dsl };
        VkDescriptorSet ds; VkResult r = vkAllocateDescriptorSets(g_dev, &ai, &ds);
        if (r == VK_SUCCESS) return ds;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) VDIE("vkAllocateDescriptorSets failed: %d", r);
        s->cdp++;   // full: the next pool (made above when there is none)
    }
}
// Descriptor sets reused within a submission (drivers without push descriptors: Mali, Xclipse, the emulator; programs over maxPushDescriptors). A set whose writes are
//   the same as one already written in this submission (layout, bindings, buffers, offsets, ranges, views, samplers) is bound again instead of allocated and written
//   anew: vkAllocateDescriptorSets and vkUpdateDescriptorSets were ~18 % of the render thread on Qualcomm's own driver in another Android port of a Switch game, and consecutive
//   draws of one material often bind exactly the same. Keyed by two 64-bit hashes of every write; the sets stay valid until the submission's pools reset (g_dsc_gen).
//   DM_VK_DESC_REUSE=0: a new set for every draw as before
// DM_VK_UBO_SNAPSHOT=1 (mtl_init sets it for a Mali-G615 on Arm's driver before r49, which needed it to start; the menu's Mali tests: Auto / Off = 0 / On = 1):
//   uniform buffers copied when their draw is recorded (push_stage), uniform updates written into the game's memory as they are recorded (sub_update)
static uint64_t g_snap_wt;   // uniform updates written through for the snapshots (the present line)
static int g_ubo_snap = -1;
static int ubo_snap_on(void) { if (g_ubo_snap < 0) { const char *e = getenv("DM_VK_UBO_SNAPSHOT"); g_ubo_snap = e && atoi(e) > 0; VLOG("uniform buffers: %s", g_ubo_snap ? "copied when their draw is recorded, updates written through (DM_VK_UBO_SNAPSHOT)" : "read by the GPU from the pools"); if (g_ubo_snap) snapc_on(); } return g_ubo_snap; }
enum { DSC_N = 4096 };
static struct { uint64_t h1, h2, gen; VkDescriptorSet ds; } g_dsc[DSC_N];
static uint64_t g_dsc_hits, g_dsc_new;   // sets reused / written (the present line)
static int dsc_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DESC_REUSE"); v = !(e && !strcmp(e, "0")); if (!v) VLOG("DM_VK_DESC_REUSE=0: a new descriptor set for every draw"); } return v; }
static void dsc_mix(uint64_t *h1, uint64_t *h2, uint64_t v) { *h1 = (*h1 ^ v) * 0x100000001b3ull; *h2 = (*h2 + v) * 0x9E3779B97F4A7C15ull; *h2 ^= *h2 >> 29; }
static void desc_flush(VkPipelineBindPoint bp, VkPipelineLayout pl, VkDescriptorSetLayout dsl) {
    if (!g_np || !D.n) { D.n = 0; return; }
    uint64_t h1 = 0xcbf29ce484222325ull, h2 = 0x84222325cbf29ce4ull; uint32_t slot = 0; int reuse = dsc_on() && (D.n < 320);
    if (reuse) {
        dsc_mix(&h1, &h2, (uint64_t)(uintptr_t)dsl); dsc_mix(&h1, &h2, D.n);
        for (uint32_t i = 0; i < D.n; i++) { const VkWriteDescriptorSet *w = &D.w[i];
            dsc_mix(&h1, &h2, (uint64_t)w->dstBinding << 32 | (uint32_t)w->descriptorType); dsc_mix(&h1, &h2, w->dstArrayElement); dsc_mix(&h1, &h2, w->descriptorCount);
            for (uint32_t k = 0; k < w->descriptorCount; k++) {
                if (w->pBufferInfo) { dsc_mix(&h1, &h2, (uint64_t)(uintptr_t)w->pBufferInfo[k].buffer); dsc_mix(&h1, &h2, w->pBufferInfo[k].offset); dsc_mix(&h1, &h2, w->pBufferInfo[k].range); }
                else if (w->pImageInfo) { dsc_mix(&h1, &h2, (uint64_t)(uintptr_t)w->pImageInfo[k].imageView); dsc_mix(&h1, &h2, (uint64_t)(uintptr_t)w->pImageInfo[k].sampler); dsc_mix(&h1, &h2, w->pImageInfo[k].imageLayout); }
                else if (w->pTexelBufferView) dsc_mix(&h1, &h2, (uint64_t)(uintptr_t)w->pTexelBufferView[k]);
                else dsc_mix(&h1, &h2, 0x5eedull); } }
        slot = (uint32_t)(h1 ^ h2 >> 17) & (DSC_N - 1);
        if (g_dsc[slot].gen == g_dsc_gen && g_dsc[slot].h1 == h1 && g_dsc[slot].h2 == h2 && g_dsc[slot].ds) {
            vkCmdBindDescriptorSets(CB(), bp, pl, 0, 1, &g_dsc[slot].ds, 0, NULL); g_dsc_hits++; D.n = 0; return; }
    }
    VkDescriptorSet ds = desc_set_alloc(dsl);
    for (uint32_t i = 0; i < D.n; i++) D.w[i].dstSet = ds;
    vkUpdateDescriptorSets(g_dev, D.n, D.w, 0, NULL); g_dsc_new++;
    if (reuse) { g_dsc[slot].h1 = h1; g_dsc[slot].h2 = h2; g_dsc[slot].gen = g_dsc_gen; g_dsc[slot].ds = ds; }
    vkCmdBindDescriptorSets(CB(), bp, pl, 0, 1, &ds, 0, NULL);
    D.n = 0;
}
// ================= The draw-state shadow (DM_VK_STATE_SHADOW) =================
// draw_ sent the driver every draw's whole state, changed or not: 19 dynamic-state calls, all 17 vertex bindings, every descriptor of every stage, the index buffer;
//   about 25 calls a draw. Mesa's drivers (Turnip) compare most of it and drop repeats; the closed drivers seem not to (26-36 us a draw on Arm r44p1 / Xclipse /
//   PanVK, 55 on PowerVR, against 12-14 on Turnip at the same draw counts: #299 #324 #270 #215, Arceus #92). Now the last value sent for each is kept here and a call
//   whose value the driver already has is left out. What Vulkan keeps, and so what this may rely on (the spec's "Pipeline Dynamic State", "Pipeline Layout
//   Compatibility", vkCmdPushDescriptorSetKHR): command buffer state lives across render pass instances, barriers and pipeline binds; a pipeline bind leaves
//   alone the states it has as dynamic, and every pipeline pso_make builds (capture variants too) has the same dynamic set; pushed descriptors stay until a push
//   with another set layout or a set bound with an incompatible one. What resets it: a new command buffer (g_cb_epoch: each submission's, each secondary of a
//   held frame), any other graphics work recorded into it (g_gfx_bind_gen: zcopy_draw, the overlay, scaling, frame generation bind pipelines with fewer dynamic
//   states and their own layouts), a transform feedback capture (its viewport, its descriptors: draw_ forgets the shadow there), a program drawn with descriptor
//   sets (its set at 0 disturbs the pushed one). Pushed descriptors are also tied to the pipeline layout they were pushed with: another one, all of them again.
//   DM_VK_STATE_SHADOW=0 (or DM_VK_BIND_ALWAYS=1): everything sent for every draw, as before; =2: also everything again in each pass (render pass begin)
typedef struct { uint64_t stamp; VkDescriptorType type; union { VkDescriptorBufferInfo b; VkDescriptorImageInfo i; VkBufferView v; } u; } dsh_ent;
enum { DSH_N = 320,   // (every stage's shifted bindings are below 320: TF_BIND)
       SH_VP = 1, SH_SC = 2, SH_CULL = 4, SH_FRONT = 8, SH_TOPO = 16, SH_DT = 32, SH_DW = 64, SH_DCMP = 128, SH_ST = 256, SH_SOPF = 512, SH_SOPB = 1024,
       SH_CMF = 2048, SH_CMB = 4096, SH_WMF = 8192, SH_WMB = 16384, SH_REFF = 32768, SH_REFB = 65536, SH_BIAS = 131072, SH_BLEND = 262144 };
static struct {
    VkCommandBuffer cb; uint64_t epoch, gen;   // where and when the values below were sent (sh_sync)
    uint32_t have;                             // SH_*: the dynamic states that hold a known value
    VkViewport vp; VkRect2D sc; uint32_t cull, front, topo, dt, dw, dcmp, st, sop[2][4], cmask[2], wmask[2], ref[2]; float bias[3], blend[4];
    VkBuffer vb[17]; VkDeviceSize vo[17], vs[17]; uint32_t vb_have;   // the vertex bindings, by binding
    VkBuffer ib; VkDeviceSize io; VkIndexType it; int ib_have;
    VkPipelineLayout dpl; uint64_t dstamp;     // the layout descriptors were last pushed with; an entry of d holds what the driver has while its stamp is dstamp
    dsh_ent d[DSH_N];
} SH;
static int g_dsh, g_dsh_cnt;   // push_stage_h: leave out the pushed descriptors the driver has; count them in the state line (set by draw_ around its stages)
static uint64_t g_sh_draws, g_sh_sent, g_sh_skip, g_sh_dw_all, g_sh_dw_sent, g_sh_vb_all, g_sh_vb_sent;   // (the state line at present: DM_VK_STATE_LOG)
static int state_shadow_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("DM_VK_STATE_SHADOW"); int v = e ? atoi(e) : 1; if (getenv("DM_VK_BIND_ALWAYS")) v = 0; m = v < 0 ? 0 : v > 2 ? 2 : v;
        VLOG("draw state: %s", m == 0 ? "all of it sent for every draw (DM_VK_STATE_SHADOW=0)" : m == 1 ? "only what changed sent to the driver (DM_VK_STATE_SHADOW=0: all of it)"
                                                    : "only what changed, and all of it again in each pass (DM_VK_STATE_SHADOW=2)"); }
    return m;
}
static void sh_forget(void) { SH.cb = VK_NULL_HANDLE; }   // (the next draw sends everything)
static void sh_desc_forget(void) { SH.dpl = VK_NULL_HANDLE; SH.dstamp++; }
static void sh_sync(VkCommandBuffer cb) {
    uint64_t gen = atomic_load(&g_gfx_bind_gen);
    if (SH.cb == cb && SH.epoch == g_cb_epoch && SH.gen == gen) return;
    SH.cb = cb; SH.epoch = g_cb_epoch; SH.gen = gen; SH.have = 0; SH.vb_have = 0; SH.ib_have = 0; sh_desc_forget();
}
static int dsh_same(const VkWriteDescriptorSet *w) {
    if (w->dstBinding >= DSH_N || w->descriptorCount != 1) return 0;
    const dsh_ent *e = &SH.d[w->dstBinding]; if (e->stamp != SH.dstamp || e->type != w->descriptorType) return 0;
    if (w->pBufferInfo) return e->u.b.buffer == w->pBufferInfo->buffer && e->u.b.offset == w->pBufferInfo->offset && e->u.b.range == w->pBufferInfo->range;
    if (w->pImageInfo) return e->u.i.sampler == w->pImageInfo->sampler && e->u.i.imageView == w->pImageInfo->imageView && e->u.i.imageLayout == w->pImageInfo->imageLayout;
    if (w->pTexelBufferView) return e->u.v == *w->pTexelBufferView;
    return 0;
}
static void dsh_note(const VkWriteDescriptorSet *w) {
    if (w->dstBinding >= DSH_N || w->descriptorCount != 1) return;
    dsh_ent *e = &SH.d[w->dstBinding]; e->stamp = SH.dstamp; e->type = w->descriptorType; memset(&e->u, 0, sizeof e->u);
    if (w->pBufferInfo) e->u.b = *w->pBufferInfo; else if (w->pImageInfo) e->u.i = *w->pImageInfo; else if (w->pTexelBufferView) e->u.v = *w->pTexelBufferView; else e->stamp = 0;
}
// A stage's descriptor writes pushed: those the driver has under the same layout left out (g_dsh), the rest in one push
static void push_writes(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout pl, uint32_t nw, VkWriteDescriptorSet *w) {
    if (!g_dsh) { p_push(cb, bp, pl, 0, nw, w); if (g_dsh_cnt) { g_sh_dw_all += nw; g_sh_dw_sent += nw; g_sh_sent++; } return; }
    if (SH.dpl != pl) { sh_desc_forget(); SH.dpl = pl; }   // (another layout: every pushed descriptor is undefined now)
    uint32_t k = 0;
    for (uint32_t i = 0; i < nw; i++) if (!dsh_same(&w[i])) w[k++] = w[i];   // (the info each write points at stays where it is)
    g_sh_dw_all += nw; g_sh_dw_sent += k;
    if (!k) { g_sh_skip++; return; }
    p_push(cb, bp, pl, 0, k, w); g_sh_sent++;
    for (uint32_t i = 0; i < k; i++) dsh_note(&w[i]);
}
// The draw's index buffer: left out when the driver has the same one bound
static void sh_index(VkCommandBuffer cb, int sh, int cnt, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    if (sh && SH.ib_have && SH.ib == b && SH.io == o && SH.it == t) { g_sh_skip++; return; }
    vkCmdBindIndexBuffer(cb, b, o, t); g_sh_sent += cnt;
    if (sh) { SH.ib = b; SH.io = o; SH.it = t; SH.ib_have = 1; }
}
// (held: the caller holds g_rmu already: a draw takes it once for its five stages instead of once a stage, an idea from a fork's v0.98)
static void push_stage_h(vshader *s, int st, VkPipelineBindPoint bp, VkPipelineLayout pl, int held) {
    if (!s || !s->nres) return;
    VkWriteDescriptorSet w_[64]; VkDescriptorBufferInfo bi_[64]; VkDescriptorImageInfo ii_[64]; VkBufferView bv_[64]; uint32_t nw = 0;
    VkWriteDescriptorSet *w = w_; VkDescriptorBufferInfo *bi = bi_; VkDescriptorImageInfo *ii = ii_; VkBufferView *bv = bv_;
    if (g_np) { if (D.n + 64 > 320) D.n = 320 - 64; w = D.w + D.n; bi = D.bi + D.n; ii = D.ii + D.n; bv = D.bv + D.n; }
    if (!held) pthread_mutex_lock(&g_rmu);
    for (int i = 0; i < s->nres; i++) {
        const spv_res *r = &s->res[i]; const vres_how *h = &s->how[i];
        w[nw] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = r->binding + s->shift, .descriptorCount = 1, .descriptorType = desc_type(r->kind) };
        if (r->count > 1) RED_ONCE(28, "shader %s: array binding %s (%u) gets its first element only", s->key, r->name, r->count);
        switch (r->kind) {
            case SPV_RES_UBO: case SPV_RES_SSBO: {
                VkBuffer b = g_zero_buf; VkDeviceSize o = 0, n = 65536;
                if (h->what == W_SUPPORT) b = g_support_buf;
                else if (h->what == W_C1) { b = s->c1buf; o = s->c1off; n = s->c1len; }
                else if (h->what == W_DRV0 || h->what == W_DRV2) {
                    int bank = h->what == W_DRV2 ? 2 : 0;
                    if (s->drv_read[bank]) {   // driver segments, filled from the current bindings (nvn_mtl.m encode_args_)
                        uint8_t *d = arena_alloc(g_sub, 0xA00, &b, &o); memset(d, 0, 0xA00); n = 0xA00;
                        if (!bank) {
                            int base = DM_GAME_NVN_SSBO_BASE(st);
                            if (base >= 0) for (int q = 0; q < 16; q++) { uint64_t a = S.ssbo[st][q].addr; uint32_t sz = (uint32_t)S.ssbo[st][q].size; memcpy(d + base + 16 * q, &a, 8); memcpy(d + base + 16 * q + 8, &sz, 4); }
                            if (st != 5) { const float mp = 2047.0f; memcpy(d + 0x610, &mp, 4); memcpy(d + 0x644, &g_draw_binst, 4); }
                            else { memcpy(d + 0x200, g_dispatch_dim, 12); g_c0_buf = b; g_c0_off = o + 0x200;
                                   for (int q = 0; q < 16; q++) { uint64_t a = S.ubo[5][q].addr; uint32_t sz = (uint32_t)S.ubo[5][q].size; memcpy(d + 0x210 + 16 * q, &a, 8); memcpy(d + 0x210 + 16 * q + 8, &sz, 4); } }
                        } else {
                            for (int q = 0; q < DM_GAME_NVN_TEX_SLOTS; q++) { uint32_t th = (uint32_t)S.tex[st][q]; memcpy(d + DM_GAME_NVN_TEX_BASE + 8 * q, &th, 4); }
                            for (int q = 0; q < DM_GAME_NVN_IMG_SLOTS; q++) { uint32_t ih = (uint32_t)S.img[st][q]; memcpy(d + DM_GAME_NVN_IMG_BASE + 8 * q, &ih, 4); }
                            for (int q = 0; q < 128; q++) { uint32_t sh = (uint32_t)S.samp[st][q]; memcpy(d + DM_GAME_NVN_SAMP_BASE + 8 * q, &sh, 4); }
                        }
                    }
                } else {
                    const p_bind_buf *u = NULL; uint64_t addr = 0, size = 0;
                    if (h->what == W_CBUF || h->what == W_UBOP) u = h->reg < 16 ? &S.ubo[st][h->reg] : NULL;
                    else if (h->what == W_SSBO) u = h->reg < 16 ? &S.ssbo[st][h->reg] : NULL;
                    if (u) { addr = u->addr; size = u->size; }
                    if (h->what == W_SBUFU) { uint32_t lo = ubo_word(st, h->reg, h->off), hi = ubo_word(st, h->reg, h->off + 4); addr = (uint64_t)hi << 32 | lo; size = 0; }
                    if (addr) GOLDEN_MEM(addr, r->kind == SPV_RES_UBO ? 65536 : size ? size + 4096 : (1u << 20));   // (what the shader can reach: bound below as the pool's rest)
                    if (TD_TRACE) { float f[8] = {0}; if (addr) memcpy(f, (const void *)(uintptr_t)addr, sizeof f);
                        VLOG("[trace]   %s %s stage %d reg %d at %#llx size %llu: %g %g %g %g %g %g %g %g", r->kind == SPV_RES_UBO ? "uniform" : "storage", r->name, st, h->reg, (unsigned long long)addr, (unsigned long long)size, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]); }
                    int shadowed = 0;   // written by uniform updates deferred in this pass (ubd_conflict ended the pass if they cover it only in part)
                    if (g_nubd && h->what == W_CBUF && r->kind == SPV_RES_UBO && addr && size && ubd_hits(addr, addr + size) && ubd_covers(addr, addr + size)) {
                        ubd_shadow(addr, addr + size, g_props.limits.minUniformBufferOffsetAlignment, &b, &o, &n); shadowed = 1; }
                    VkBuffer pb; VkDeviceSize po, left;
                    if (shadowed) ;
                    else if (addr && pool_buf(addr, 1, &pb, &po, &left)) {
                        VkDeviceSize al = r->kind == SPV_RES_UBO ? g_props.limits.minUniformBufferOffsetAlignment : g_props.limits.minStorageBufferOffsetAlignment;
                        // the rest of the pool up to the block the shader declares (64 KB uniform blocks), not the game's bound size: Maxwell (and the Metal replay, which binds a raw
                        // address) reads past the bound size into the memory that follows, and Shield's lighting reads there (bound at the exact size, the scene came out at half brightness)
                        VkDeviceSize want = left, cap = r->kind == SPV_RES_UBO ? g_props.limits.maxUniformBufferRange : g_props.limits.maxStorageBufferRange; (void)size;
                        if (want > left) want = left; if (want > cap) want = cap;
                        // DM_VK_UBO_SNAPSHOT=1 (diagnosis, Mali white #47 / #53 / #64 / #68): every uniform buffer copied when its draw is recorded, as Adreno's 64-byte alignment makes
                        //   most of them, instead of the GPU reading the pool live; the game's bound size + 4 KB (what the lighting reads past it), at most 64 KB.
                        //   One copy per submission while its range is not written (snapc_get, above wb_note), unless DM_VK_UBO_SNAPSHOT_REUSE=0
                        if (ubo_snap_on() && r->kind == SPV_RES_UBO) {
                            VkDeviceSize m = size ? size + 4096 : 65536; if (m > want) m = want; if (m > 65536) m = 65536;
                            if (snapc_on() && snapc_get(addr, m, &b, &o)) g_snap_reused++;
                            else { uint8_t *d = arena_alloc(g_sub, m, &b, &o); memcpy(d, (const void *)(uintptr_t)addr, m); g_snap_copies++; g_snap_bytes += m; if (snapc_on()) snapc_put(addr, m, b, o); }
                            n = m;
                        }
                        else if (po % al == 0) { b = pb; o = po; n = want; }
                        else if (r->kind == SPV_RES_UBO) {   // misaligned uniform data: a snapshot in the arena (the GPU-side updates of this submission are in the pool already)
                            if (want > 65536) want = 65536;   // the translated shaders declare 64 KB uniform blocks: never more to copy (MoltenVK reports a range of gigabytes)
                            uint8_t *d = arena_alloc(g_sub, want, &b, &o); memcpy(d, (const void *)(uintptr_t)addr, want); n = want;
                            RED_ONCE(29, "uniform buffer at pool offset %#llx is not %llu-aligned: bound as a snapshot", (unsigned long long)po, (unsigned long long)al);
                        } else if (!h->rw) {   // misaligned read-only storage data (Adreno wants 64, the Switch less): copied on the GPU, in order, to an aligned arena slot
                            // A snapshot of guest memory taken here missed what the GPU wrote earlier in the submission: Violet's tone mapping reads its exposure that way
                            //   (written by a pass in the same frame), read zeros, and every 3D scene came out black (the academy video, the ID photo). The copy runs between
                            //   render passes: the open one is ended and opened again (its clears were taken when it first began, so it reopens loading)
                            VkDeviceSize m = size && size < want ? size : want; if (m > (size ? 1u << 20 : 65536u)) m = size ? 1u << 20 : 65536u;
                            static int snap = -1; if (snap < 0) snap = getenv("DM_VK_SSBO_GPU_COPY") == NULL;   // the CPU snapshot unless DM_VK_SSBO_GPU_COPY (that copy lost the device on an Adreno 830 at Violet's enrollment screen: experimental)
                            if (snap) { uint8_t *d = arena_alloc(g_sub, m, &b, &o); memcpy(d, (const void *)(uintptr_t)addr, m); n = m; } else {
                            (void)arena_alloc(g_sub, m, &b, &o); n = m;
                            int reopen = P.open && !P_tf && bp == VK_PIPELINE_BIND_POINT_GRAPHICS; VkCommandBuffer ccb = CB(); pass_end();
                            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
                            vkCmdPipelineBarrier(ccb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                            VkBufferCopy bc = { po, o, m }; vkCmdCopyBuffer(ccb, pb, b, 1, &bc);
                            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                            vkCmdPipelineBarrier(ccb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                            if (reopen) pass_begin(); }
                            RED_ONCE(30, "storage buffer at pool offset %#llx is not %llu-aligned: %s", (unsigned long long)po, (unsigned long long)al, snap ? "bound as a snapshot" : "copied on the GPU to an aligned slot");
                        } else RED_ONCE(32, "writable storage buffer at pool offset %#llx is not %llu-aligned: bound to zeros", (unsigned long long)po, (unsigned long long)al);
                        if (r->kind == SPV_RES_SSBO && h->rw) wb_note(addr, size ? size : (want < 65536 ? want : 65536));   // a writable storage buffer: its bound range comes back (mirrored pools)
                    } else if (h->what != W_NONE && addr) RED_ONCE(31, "stage %d %s slot %d at %#llx is in no memory pool", st, r->kind == SPV_RES_UBO ? "uniform" : "storage", h->reg, (unsigned long long)addr);
                }
                if (r->kind == SPV_RES_UBO && n > g_props.limits.maxUniformBufferRange) n = g_props.limits.maxUniformBufferRange;
                bi[nw] = (VkDescriptorBufferInfo){ b, o, n ? n : 16 }; w[nw].pBufferInfo = &bi[nw]; nw++; break;
            }
            case SPV_RES_TEX: case SPV_RES_SIMG: case SPV_RES_IMG: {
                int storage = r->kind == SPV_RES_IMG; int tid = 0, sid = 0;
                if (h->what == W_TEX) { uint64_t th = h->reg < NTEX ? S.tex[st][h->reg] : 0; tid = (int)(th & 0xfffff); sid = (int)(th >> 20 & 0xfff); }
                else if (h->what == W_TEXU) { uint32_t th = ubo_word(st, h->reg, h->off); tid = (int)(th & 0xfffff); sid = (int)(th >> 20); }
                else if (h->what == W_IMG) tid = h->reg < 32 ? (int)(S.img[st][h->reg] & 0xfffff) : 0;
                if (h->skind == S_SEP) sid = h->sreg < 128 ? (int)(S.samp[st][h->sreg] >> 20) : 0;
                else if (h->skind == S_UBO) sid = (int)(ubo_word(st, h->sreg, h->soff) >> 20);
                else if (h->skind == S_ID) sid = h->sreg;
                VkImageView v = tid ? tex_view_for(tid, r, storage, s->key, st, s->img_writes) : VK_NULL_HANDLE;
                if (TD_TRACE) VLOG("[trace]   %s %s tid %d ser=%llu%s", storage ? "image" : "texture", r->name, tid, 0ull, v ? "" : " (no view: dummy)");
                // DM_VK_TEXOVR=<fs key>:<binding>=0|1,...: that binding reads 0 or 1. Without it, the game profile's tex_unwritten_one pairs read 1 (textures the game
                // never draws on this host: Pokémon Legends: Arceus' ambient occlusion result, whose pass never runs); DM_VK_TEXOVR=none turns that off
                if (!storage) { static const char *ov = (const char *)1; if (ov == (const char *)1) { ov = getenv("DM_VK_TEXOVR");
                        if (ov && !strcmp(ov, "none")) ov = NULL;
                        else if (!ov && DM_GAME_TEX_UNWRITTEN_ONE[0]) { static char def[4096]; size_t o = 0; const char *q = DM_GAME_TEX_UNWRITTEN_ONE;
                            while (*q && o + 64 < sizeof def) { const char *e = strchr(q, ' '); size_t n = e ? (size_t)(e - q) : strlen(q); o += (size_t)snprintf(def + o, sizeof def - o, "%.*s=1,", (int)n, q); q += n; while (*q == ' ') q++; }
                            ov = def; VLOG("textures the game never draws here, read as 1: %s", DM_GAME_TEX_UNWRITTEN_ONE); } }
                    if (ov) { char pat[96]; snprintf(pat, sizeof pat, "%s:%s=", s->key, r->name); const char *m = strstr(ov, pat);
                        if (m) { int k = vkind(r); vtex *dx = r->depth && k <= 3 ? g_dummy_depth[k] : g_dummy[k][r->sclass < 3 ? r->sclass : 0]; uint8_t c = m[strlen(pat)] == '1' ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_ZERO;
                                 uint8_t sw[4] = { c, c, c, c }; uint32_t nl = k == 2 || k == 3 ? 6 : 1;
                                 v = view_get(dx, VK_OF_KIND[k], dx->fmt, 0, 1, 0, nl, sw, dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : dx->aspect); } } }
                if (!v) v = dummy_view(r, storage);
                VkSampler smp = storage ? VK_NULL_HANDLE : (tid && tex_is_int(tid) ? samp_nearest_by_id(sid) : samp_by_id(sid)); if (!storage && !smp) smp = g_dummy_samp;
                ii[nw] = (VkDescriptorImageInfo){ smp, v, VK_IMAGE_LAYOUT_GENERAL }; w[nw].pImageInfo = &ii[nw]; nw++; break;
            }
            case SPV_RES_SAMP: ii[nw] = (VkDescriptorImageInfo){ g_dummy_samp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL }; w[nw].pImageInfo = &ii[nw]; nw++; RED_ONCE(32, "separate sampler binding %s not resolved", r->name); break;
            case SPV_RES_TBUF: case SPV_RES_IBUF: {
                int tid = h->reg < NTEX ? (int)(S.tex[st][h->reg] & 0xfffff) : 0; VkBufferView view = VK_NULL_HANDLE;
                if (r->kind == SPV_RES_IBUF) tid = h->reg < 32 ? (int)(S.img[st][h->reg] & 0xfffff) : 0;
                { treg_t rg; if (treg_get(tid, &rg)) { vtex *x = tex_res(&rg.t); sk_read(x); if (x && x->bview) view = x->bview; } }
                bv[nw] = view ? view : g_dummy_tbuf[r->sclass < 3 ? r->sclass : 0]; w[nw].pTexelBufferView = &bv[nw]; nw++; break;
            }
        }
    }
    if (!held) pthread_mutex_unlock(&g_rmu);
    if (g_np) D.n += nw;
    else if (nw) { if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) push_writes(CB(), bp, pl, nw, w); else p_push(CB(), bp, pl, 0, nw, w); }   // (a draw's: push_writes, the state shadow)
}
static void push_stage(vshader *s, int st, VkPipelineBindPoint bp, VkPipelineLayout pl) { push_stage_h(s, st, bp, pl, 0); }

// ================= Draws =================
// draw_ binds a pipeline only when it is not the one it bound last in this command buffer (Vulkan keeps the binding across passes). Anything else that binds a
//   graphics pipeline in the submission's command buffer (zcopy_draw) counts g_gfx_bind_gen up, so the game's next draw binds its own again and sends all of its
//   state again (the state shadow below)
// DM_VK_PROF: where the replay's CPU time goes per draw (pass + pipeline, dynamic state + vertex streams, descriptors, flush + draw), logged every 300 presents
static int g_vprof = -1; static uint64_t g_vp_t[4], g_vp_n, g_vp_o[5], g_vp_pass, g_vp_pso;
#define VPROF_ON (g_vprof < 0 ? (g_vprof = getenv("DM_VK_PROF") != NULL) : g_vprof)
static void draw_(const p_draw *dw, const p_draw_idx *di, uint64_t *mark);
// the frame capture's draw by draw values (g_cap_dp): after a draw of the chosen pass, the pass is ended (without logging it), colour 0's 12 points copied, and the pass
//   opened again (loading: its clears were taken when it first began). Only in the one captured frame
static void cap_draw_values(void) {
    if (!P.open || P_tf || g_cap_nd >= CAPD) return;
    const tex_t *ct = R.rt.color[0]; vtex *x = ct ? tex_res(ct) : NULL; if (!x || !x->img || x->samples > 1) return;
    sk_read(x);
    if (!g_cap_dp) { if ((x->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && !b10_swapped_tex(x)) || P.tw < 640) return; g_cap_dp = 1; g_cap_dfmt = x->fmt; }   // (the stand-in: the same pass, DM_VK_B10_AS_RGBA16)
    uint32_t lv = R.rt.has_cview[0] && R.rt.cview[0].set_levels ? R.rt.cview[0].base_level : 0, ly = R.rt.has_cview[0] && R.rt.cview[0].set_layers ? R.rt.cview[0].min_layer : 0;
    uint32_t sw = x->w >> lv ? x->w >> lv : 1, sh = x->h >> lv ? x->h >> lv : 1; VkBufferImageCopy pc[CAPP];
    for (int k = 0; k < CAPP; k++) pc[k] = (VkBufferImageCopy){ .bufferOffset = (VkDeviceSize)CAPN * CAPW * CAPH * 4 + (VkDeviceSize)CAPS * CAPP * 16 + ((VkDeviceSize)g_cap_nd * CAPP + (VkDeviceSize)k) * 16,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, lv, x->is3d ? 0 : ly, 1 }, .imageOffset = { (int32_t)(sw * (uint32_t)(1 + 2 * (k % 4)) / 8), (int32_t)(sh * (uint32_t)(1 + 2 * (k / 4)) / 6), x->is3d ? (int32_t)ly : 0 },
        .imageExtent = { 1, 1, 1 } };
    vprog *cp = S.prog ? prog_get(S.prog) : NULL; snprintf(g_cap_dkey[g_cap_nd], sizeof g_cap_dkey[0], "%s", cp && cp->sh[1] ? cp->sh[1]->key : "-");
    VkCommandBuffer cb = CB(); rend_end(cb); P.open = 0;
    barrier_all(cb); vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_cap_buf, CAPP, pc); barrier_all(cb);
    g_cap_nd++; pass_begin();
}
// Dual-source blending, on every driver: Legends: Arceus' bloom adds each blurred level as output 0 + target x output 1's alpha. Arm's own Mali driver (r44p1) has
//   no dualSrcBlend at all, and where a driver has it the translated shaders still put output 1 at Location 1, not at Location 0 Index 1 where Vulkan's SRC1
//   factors read it: undefined, and the glow went missing on an Adreno 830 too.
//   Such a draw is recorded twice, the same on screen as one: first with outputs 0 and 1 swapped (frag_module_ sw) and blend ZERO, target x <its SRC1 factor as SRC>,
//   so the target is scaled by output 1; then as it is with blend <its source factor>, ONE, adding output 0. Done when the blend is ADD, the target factors are
//   SRC1_* / ZERO / ONE and the source factors do not read the target (which the first step changed); otherwise the old fallback (bfac_dev). DM_VK_DUALSRC=off: never
//   (the first output's factors where the driver has no dualSrcBlend); DM_VK_DUALSRC=keep: the game's factors passed to the driver as they are
static int ds_src_ok(int f) { return f == 1 || f == 2 || (f >= 3 && f <= 6) || (f >= 0x61 && f <= 0x64); }   // ZERO, ONE, SRC_*, CONSTANT_*: not the target, not output 1
static int ds_dst_map(int f) { return f == 1 ? 1 : f == 2 ? 2 : f == 0x10 ? 3 : f == 0x11 ? 4 : f == 0x12 ? 5 : f == 0x13 ? 6 : -1; }   // SRC1_x as SRC_x; ZERO / ONE kept
static void draw_one(const p_draw *dw, const p_draw_idx *di);
// DM_SHADOWS=off (Scarlet / Violet 4.0.0 only, loader.c dm_shadow_off): the sun's shadow map is the one square DEPTH32F target drawn with no colour attachment
//   (about 19 passes and 160 draws a frame docked, a quarter of the frame's draws). Its draws are dropped, so the map keeps the far depth it is cleared to and
//   nothing comes out shadowed: the whole pass is replay and GPU work that never happens. Clears still run. The first target taken for it is logged with its size
__attribute__((weak)) int dm_shadow_off;
static int shadow_target(const p_set_rt *rt) {
    static const tex_t *last; static int last_is, logged;
    const tex_t *t = rt->depth;
    if (!t) return 0;
    for (int i = 0; i < rt->n && i < 8; i++) if (rt->color[i]) return 0;
    if (t == last) return last_is;
    last = t;
    last_is = t->d.format == NVN_FORMAT_DEPTH32F && t->d.width == t->d.height && t->d.width >= 512;
    if (last_is && !logged) { logged = 1; LOG("[nvn] shadows off: draws to the %ux%u depth-only target (target %d, %u layers) are dropped (DM_SHADOWS=off)", t->d.width, t->d.height, t->d.target, t->d.depth); }
    return last_is;
}
static void draw(const p_draw *dw, const p_draw_idx *di) {
    if (&dm_shadow_off && dm_shadow_off && R.has_rt && shadow_target(&R.rt)) return;
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_DUALSRC"); on = !(e && (!strcmp(e, "off") || !strcmp(e, "keep"))); }
    const blend_st b = S.blend[0];
    int ds = on && !S.rast_discard && (S.color.blend_en & 1) && ((b.dst_rgb >= 0x10 && b.dst_rgb <= 0x13) || (b.dst_a >= 0x10 && b.dst_a <= 0x13));
    if (ds && (b.eq_rgb > 1 || b.eq_a > 1 || !ds_src_ok(b.src_rgb) || !ds_src_ok(b.src_a) || ds_dst_map(b.dst_rgb) < 0 || ds_dst_map(b.dst_a) < 0)) {
        RED_ONCE(112, "dual-source blend that two steps cannot replace (rgb %u,%u,%u alpha %u,%u,%u): the first output's instead", (unsigned)b.src_rgb, (unsigned)b.dst_rgb, (unsigned)b.eq_rgb, (unsigned)b.src_a, (unsigned)b.dst_a, (unsigned)b.eq_a);
        ds = 0; }
    if (!ds) { draw_one(dw, di); return; }
    { static int said; if (!said++) VLOG("blending: the game's dual-source blends are drawn in two steps (DM_VK_DUALSRC=keep: as one, by the driver)"); }
    blend_st a = b; a.src_rgb = 1; a.dst_rgb = (uint64_t)ds_dst_map(b.dst_rgb); a.src_a = 1; a.dst_a = (uint64_t)ds_dst_map(b.dst_a); a.eq_rgb = 1; a.eq_a = 1;
    blend_st c = b; c.dst_rgb = 2; c.dst_a = 2; c.eq_rgb = 1; c.eq_a = 1;
    S.blend[0] = a; g_ds_swap = 1; draw_one(dw, di); g_ds_swap = 0;
    S.blend[0] = c; draw_one(dw, di);
    S.blend[0] = b;
}
static void draw_one(const p_draw *dw, const p_draw_idx *di) {
    if (atomic_load(&g_cap) == 2) { g_cap_draws++;
        if (g_cap_draws <= 3 && S.prog) { vprog *cp = prog_get(S.prog); if (cp && cp->sh[0]) cap_printf("  draw %d: vertex %s fragment %s\n", g_cap_draws, cp->sh[0]->key, cp->sh[1] ? cp->sh[1]->key : "-"); } }
    if (atomic_load(&g_cap) == 2 && g_cap_dp < 2) { draw_(dw, di, NULL); cap_draw_values(); return; }
    if (!VPROF_ON) { draw_(dw, di, NULL); return; }
    uint64_t m[5] = { mono_ns(), 0, 0, 0, 0 }; draw_(dw, di, m); uint64_t e = mono_ns();
    for (int i = 0; i < 4; i++) { uint64_t a = m[i], b = m[i + 1] ? m[i + 1] : e; if (b > a) g_vp_t[i] += b - a; if (!m[i + 1]) break; }
    g_vp_n++;
}
// Whether a draw needs the uniform updates deferred in the open pass landed first (nvn_mtl.m ubd_conflict): they hold its vertices, indices, indirect arguments or a storage
// buffer, or cover a uniform binding or a bindless handle only in part
static int ubd_conflict(const vprog *pr, const p_draw *dw, const p_draw_idx *di) {
    for (int s_ = 0; s_ < 16; s_++) if (S.vbuf[s_].address && ubd_hits(S.vbuf[s_].address, S.vbuf[s_].address + (S.vbuf[s_].size ? S.vbuf[s_].size : 1))) return 1;
    if (di && di->ibuf && ubd_hits(di->ibuf, di->ibuf + (uint64_t)(di->count > 0 ? di->count : 1) * 4)) return 1;
    uint64_t ind = dw ? dw->ind : di ? di->ind : 0; if (ind && ubd_hits(ind, ind + 32)) return 1;
    uint64_t mdi = dw ? dw->mdi_cnt : di ? di->mdi_cnt : 0; if (mdi && ubd_hits(mdi, mdi + 4)) return 1;
    for (int st = 0; st < 5; st++) { const vshader *s = pr->sh[st]; if (!s) continue;
        for (int i = 0; i < s->nres; i++) { const vres_how *h = &s->how[i]; if (h->reg >= 16) continue;
            if (h->what == W_CBUF || h->what == W_UBOP || h->what == W_SSBO) {
                const p_bind_buf *u = h->what == W_SSBO ? &S.ssbo[st][h->reg] : &S.ubo[st][h->reg]; if (!u->addr) continue;
                uint64_t hi = u->addr + (u->size ? u->size : 1);
                if (h->what == W_CBUF && s->res[i].kind == SPV_RES_UBO && u->size) { if (ubd_hits(u->addr, hi) && !ubd_covers(u->addr, hi)) return 1; }
                else if (ubd_hits(u->addr, hi)) return 1;
            } else if (h->what == W_TEXU || h->what == W_SBUFU) {
                const p_bind_buf *u = &S.ubo[st][h->reg]; if (!u->addr) continue;
                uint64_t a = u->addr + h->off, n = h->what == W_SBUFU ? 8 : 4; if (ubd_hits(a, a + n) && !ubd_covers(a, a + n)) return 1;
                if (h->what == W_SBUFU) { uint32_t lo = ubo_word(st, h->reg, h->off), hw = ubo_word(st, h->reg, h->off + 4); uint64_t sa = (uint64_t)hw << 32 | lo; if (sa && ubd_hits(sa, sa + 1)) return 1; }
            }
            if (h->skind == S_UBO && h->sreg < 16 && S.ubo[st][h->sreg].addr) { uint64_t a = S.ubo[st][h->sreg].addr + h->soff; if (ubd_hits(a, a + 4) && !ubd_covers(a, a + 4)) return 1; }
        } }
    return 0;
}
static void draw_(const p_draw *dw, const p_draw_idx *di, uint64_t *mark) {
    g_draw_binst = (uint32_t)(dw ? dw->base_instance : di ? di->base_instance : 0);
    if ((dw && dw->mdi_cnt) || (di && di->mdi_cnt)) { uint64_t a = dw ? dw->mdi_cnt : di->mdi_cnt; int i = dw ? dw->mdi_i : di->mdi_i; uint32_t n = 0; memcpy(&n, (const void *)(uintptr_t)a, 4); if ((uint32_t)i >= n) return; }
    if (!S.prog) { RED_ONCE(33, "draw with no program bound"); return; }
    vprog *pr = prog_get(S.prog); if (!pr->sh[0]) { RED_ONCE(34, "program has no vertex shader (or it is not mapped)"); return; }
    // Rasterizer discard: the vertex stage alone. With transform feedback active the draw captures its outputs (the shader's capture variant stores them into the bound
    // feedback buffers, as nvn_mtl.m tf_draw: Pokémon Legends: Arceus / Z-A skin characters so); otherwise it changes nothing and is skipped
    VkBuffer tfb[4] = {0}; VkDeviceSize tfo[4] = {0}, tfr[4] = {0}; uint32_t tfpc[5] = {0}; int tfcap = 0;
    uint32_t tfp_base[4] = {0}, tfp_rows = 0; int tfp_n = 0;   // a capture as points (tf_points.h): the buffers' first words, the target rows, how many targets
    if (S.rast_discard) {
        vshader *vs = pr->sh[0];
        if (!S.tf_on) return;
        if (g_tf_pts > 0 ? !vs->ptsmod || !tfp_init() : !vs->tfmod) { RED_ONCE(100, "transform feedback with vertex shader %s, which has no capture variant: nothing captured", vs->key); return; }
        if (!dw || dw->ind || dw->mdi_cnt || dw->instances > 1 || dw->prim == 6 || dw->prim == 7) { RED_ONCE(101, "transform feedback draw that is indexed, indirect, instanced or a fan / quad list (vertex %s): not captured", vs->key); return; }
        if (dw->count <= 0) return;
        VkDeviceSize al = g_props.limits.minStorageBufferOffsetAlignment ? g_props.limits.minStorageBufferOffsetAlignment : 256;
        tfpc[0] = (uint32_t)dw->first;
        for (int b = 0; b < 4; b++) { if (!vs->tf_stride[b]) continue;
            uint64_t n = (uint64_t)dw->count * vs->tf_stride[b], at = S.tf[b].addr + S.tf_off[b]; VkDeviceSize o;
            if (!S.tf[b].addr || S.tf_off[b] + n > S.tf[b].size || !pool_buf(at, n, &tfb[b], &o, NULL)) {
                RED_ONCE(102, "transform feedback buffer %d of vertex %s is unbound, too small or in no memory pool (%#llx + %llu of %llu): not captured", b, vs->key, (unsigned long long)S.tf[b].addr, (unsigned long long)(S.tf_off[b] + n), (unsigned long long)S.tf[b].size);
                return; }
            tfo[b] = o & ~(al - 1); tfr[b] = o - tfo[b] + n; tfpc[1 + b] = (uint32_t)((o - tfo[b]) / 4); }   // bound at an aligned offset; the shader starts base[b] words in
        tfcap = 1;
    }
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK && !tfcap) return;
    if (g_nubd && ubd_conflict(pr, dw, di)) { g_ubd_conflicts++; pass_end(); }   // the deferred uniform updates land before this draw (a new pass loads what the old one stored)
    uint64_t pp0 = mark ? mono_ns() : 0;
    if (tfcap) {   // into the open capture pass unless this capture reads or rewrites what one there wrote (its outputs, vertex streams, uniform / storage buffers)
        int keep = g_ntfw < 60;
        for (int b = 0; keep && b < 4; b++) if (tfb[b]) { uint64_t at = S.tf[b].addr + S.tf_off[b]; if (tf_hits(at, at + (uint64_t)dw->count * pr->sh[0]->tf_stride[b])) keep = 0; }
        for (int s_ = 0; keep && s_ < 16; s_++) if (S.vbuf[s_].address && tf_hits(S.vbuf[s_].address, S.vbuf[s_].address + (S.vbuf[s_].size ? S.vbuf[s_].size : 65536))) keep = 0;
        for (int st = 0; keep && st < 5; st++) for (int q = 0; keep && q < 16; q++) {
            if (S.ubo[st][q].addr && tf_hits(S.ubo[st][q].addr, S.ubo[st][q].addr + (S.ubo[st][q].size ? S.ubo[st][q].size : 65536))) keep = 0;
            if (S.ssbo[st][q].addr && tf_hits(S.ssbo[st][q].addr, S.ssbo[st][q].addr + (S.ssbo[st][q].size ? S.ssbo[st][q].size : 65536))) keep = 0; }
        if (g_tf_pts > 0) {   // as points: one target per capture buffer, the buffers numbered from 0 with stride = components x 4, TFP_W pixels a row
            vshader *vs_ = pr->sh[0]; int nb_ = 0;
            for (int b = 0; b < 4; b++) if (vs_->tf_stride[b]) { if (b != nb_ || vs_->tf_stride[b] != 4u * (uint32_t)vs_->tf[b][1]) nb_ = -1; else nb_++; if (nb_ < 0) break; }
            if (nb_ <= 0 || (uint64_t)dw->count > (uint64_t)TFP_W * TFP_H) { RED_ONCE(126, "transform feedback as points: vertex %s's capture (%d vertices) does not fit the targets: not captured", vs_->key, dw->count); return; }
            for (int b = 0; b < 4; b++) tfp_base[b] = tfpc[1 + b];
            tfp_rows = ((uint32_t)dw->count + TFP_W - 1) / TFP_W; tfp_n = nb_;
            tfpc[1] = tfp_rows; tfpc[2] = tfpc[3] = tfpc[4] = 0;   // (the point variant's push constants: { first, rows })
            tfp_pass_begin(nb_, tfp_rows);
        } else tf_pass_begin(keep);
    } else if (!pass_begin()) return;
    if (!tfcap && (P.hd || P.hs)) {   // a draw writing depth / stencil: noted; in a pass begun with store op NONE for it, the pass restarts storing (pass_writes)
        int st_w = P.hs && S.ds.st && (S.st_wmask[0] || S.st_wmask[1]) && (S.ds.fail_f != NVN_STENCIL_OP_KEEP || S.ds.zpass_f != NVN_STENCIL_OP_KEEP || S.ds.zfail_f != NVN_STENCIL_OP_KEEP ||
                   S.ds.fail_b != NVN_STENCIL_OP_KEEP || S.ds.zpass_b != NVN_STENCIL_OP_KEEP || S.ds.zfail_b != NVN_STENCIL_OP_KEEP);
        pass_writes(P.hd && S.ds.dt && S.ds.dw, st_w);
        if (!P.open) return;
        if ((P.hd && S.ds.dt) || (P.hs && S.ds.st)) P.dtest = 1;   // (reads the attachment: seg_record_prim's LOAD_OP_NONE needs a pass that never does)
    }
    int prim = dw ? dw->prim : di->prim; VkPrimitiveTopology topo = topo_of(prim);
    if (tfcap && g_tf_pts > 0) topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;   // (a capture as points: one per vertex, tf_points.h)
    uint64_t pp1 = mark ? mono_ns() : 0;
    uint32_t vbm = 0; VkPipeline p = pso_get(pr, topo, &vbm); if (!p) return;
    if (mark) { g_vp_pass += pp1 - pp0; g_vp_pso += mono_ns() - pp1; }   // (DM_VK_PROF: the "pass + pipeline" stage split)
    if (TD_TRACE) VLOG("[trace] draw vs %s fs %s blend_en %#x", pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-", (unsigned)S.color.blend_en);
    VkCommandBuffer cb = CB(); if (mark) mark[1] = mono_ns();
    // The state shadow (above push_stage_h): a call whose value the driver has in this command buffer is left out. A capture sends all of its state (its own
    //   viewport, its buffers) and leaves the shadow unknown for the next draw; cnt: the draw counts in the state line (captures do not)
    int sh = !tfcap && state_shadow_mode() != 0, cnt = !tfcap;
    if (tfcap) sh_forget(); else { if (sh) sh_sync(cb); g_sh_draws++; }
    #define SH_SEND(bit, same, store, call) do { if (sh && (SH.have & (bit)) && (same)) g_sh_skip++; else { call; if (sh) { store; SH.have |= (bit); } g_sh_sent += cnt; } } while (0)
    { static VkCommandBuffer bcb; static VkPipeline bp; static uint64_t bfence, bgen, bepoch;   // the same pipeline already bound in this command buffer: no rebind (Vulkan keeps it across passes)
      static int off = -1; if (off < 0) off = getenv("DM_VK_BIND_ALWAYS") != NULL;   // (bgen: g_gfx_bind_gen, another graphics pipeline bound there since; bepoch: a command buffer begun since)
      if (off || cb != bcb || bp != p || bfence != g_sub->fence || bgen != g_gfx_bind_gen || bepoch != g_cb_epoch) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); bcb = cb; bp = p; bfence = g_sub->fence; bgen = g_gfx_bind_gen; bepoch = g_cb_epoch; if (pso_cap()) pso_used(p);
          g_sh_sent += cnt; cr_pipe(pr->sh[0] ? pr->sh[0]->key : NULL, pr->sh[1] ? pr->sh[1]->key : NULL); } else g_sh_skip += cnt; }   // (breadcrumbs: the shaders bound, in this pass)
    // dynamic state
    float vx = 0, vy = 0, vw = (float)P.tw, vh = (float)P.th;
    if (S.has_vp) { vx = S.vp[0][0]; vy = S.vp[0][1]; vw = S.vp[0][2]; vh = S.vp[0][3]; }
    VkViewport v = { vx, vy, vw, vh, S.dr[0], S.dr[1] };
    { static int on = -1, n; static float seen[48][6]; if (on < 0) on = getenv("DM_VK_LOG_VP") != NULL;   // DM_VK_LOG_VP: the first distinct viewports (with the target size) the game draws with
      if (on && n < 48) { float k[6] = { vx, vy, vw, vh, (float)P.tw, (float)P.th }; int i; for (i = 0; i < n && memcmp(seen[i], k, sizeof k); i++) {}
        if (i == n) { memcpy(seen[n++], k, sizeof k); VLOG("viewport %g,%g %gx%g depth %g..%g on a %ux%u target (scissor %s %d,%d %ux%u)", vx, vy, vw, vh, S.dr[0], S.dr[1], P.tw, P.th, R.has_sc ? "set" : "none", R.sc.x, R.sc.y, R.sc.w, R.sc.h); } } }
    if (!atomic_load(&g_origin_ll) && !g_yflip) { v.y = vy + vh; v.height = -vh; }   // NVN upper-left: NDC +1 on the first row (Vulkan puts -1 there); g_yflip: done in the vertex shader
    if (v.minDepth < 0) v.minDepth = 0; if (v.minDepth > 1) v.minDepth = 1; if (v.maxDepth < 0) v.maxDepth = 0; if (v.maxDepth > 1) v.maxDepth = 1;
    SH_SEND(SH_VP, !memcmp(&SH.vp, &v, sizeof v), SH.vp = v, vkCmdSetViewport(cb, 0, 1, &v));   // (compared bit for bit: -0 and NaN are sent again)
    if (tfcap && tfp_n) { VkViewport pv = { 0, 0, (float)TFP_W, (float)tfp_rows, 0, 1 }; vkCmdSetViewport(cb, 0, 1, &pv); }   // (the targets, one pixel a vertex)
    VkRect2D sc = {{0, 0}, {P.tw, P.th}};
    if (R.has_sc && !tfcap) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h; if (x1 > (long)P.tw) x1 = P.tw; if (y1 > (long)P.th) y1 = P.th;
        if (x1 <= x0 || y1 <= y0) return; sc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}}; }
    SH_SEND(SH_SC, SH.sc.offset.x == sc.offset.x && SH.sc.offset.y == sc.offset.y && SH.sc.extent.width == sc.extent.width && SH.sc.extent.height == sc.extent.height, SH.sc = sc,
            vkCmdSetScissor(cb, 0, 1, &sc));
    seg_area_note(cb, sc);   // (a held pass's area: the scissors its draws use, the game's scaled one in Sword / Shield's scene passes; noted whether or not the shadow
                             //   left the call out, since a call left out sets the same scissor the pass already has)
    if (g_mw_px) mw_draw_rect(vx, vy, vw, vh, &sc);   // (the white-screen check: the part of the lit target this draw covers)
    if (g_eds) {   // (without extended dynamic state these are built into the pipeline: pso_get's sst, the same values)
    // (points and lines never cull, so none is sent for them: Arm's driver culled them with a dynamic cull mode and those draws vanished, erratum 3786517, r41-r51)
    uint32_t cm = topo_class(topo) < 2 ? VK_CULL_MODE_NONE : S.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : S.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    SH_SEND(SH_CULL, SH.cull == cm, SH.cull = cm, vkCmdSetCullMode(cb, cm));
    int ccw = S.poly.front == NVN_FRONT_FACE_CCW; if (atomic_load(&g_origin_ll)) ccw = !ccw;   // lower-left: the y flip relative to GL window coordinates reverses the winding
    uint32_t ff = ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    SH_SEND(SH_FRONT, SH.front == ff, SH.front = ff, vkCmdSetFrontFace(cb, (VkFrontFace)ff));
    SH_SEND(SH_TOPO, SH.topo == (uint32_t)topo, SH.topo = (uint32_t)topo, vkCmdSetPrimitiveTopology(cb, topo));
    }
    if (!tfcap) {   // a capture pipeline (rasterizer discard) has no depth / stencil state to set (VUID-vkCmdDraw-None-08608)
    if (g_eds) {
    uint32_t dt = P.hd && S.ds.dt, dwr = P.hd && S.ds.dw, dc = (uint32_t)(S.ds.df ? S.ds.df - 1 : 7), stt = P.hs && S.ds.st;
    SH_SEND(SH_DT, SH.dt == dt, SH.dt = dt, vkCmdSetDepthTestEnable(cb, dt)); SH_SEND(SH_DW, SH.dw == dwr, SH.dw = dwr, vkCmdSetDepthWriteEnable(cb, dwr));
    SH_SEND(SH_DCMP, SH.dcmp == dc, SH.dcmp = dc, vkCmdSetDepthCompareOp(cb, (VkCompareOp)dc));
    SH_SEND(SH_ST, SH.st == stt, SH.st = stt, vkCmdSetStencilTestEnable(cb, stt));
    uint32_t sf[4] = { (uint32_t)(S.ds.fail_f - 1), (uint32_t)(S.ds.zpass_f - 1), (uint32_t)(S.ds.zfail_f - 1), (uint32_t)(S.ds.sf_f - 1) },
             sb[4] = { (uint32_t)(S.ds.fail_b - 1), (uint32_t)(S.ds.zpass_b - 1), (uint32_t)(S.ds.zfail_b - 1), (uint32_t)(S.ds.sf_b - 1) };
    SH_SEND(SH_SOPF, !memcmp(SH.sop[0], sf, sizeof sf), memcpy(SH.sop[0], sf, sizeof sf), vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)sf[0], (VkStencilOp)sf[1], (VkStencilOp)sf[2], (VkCompareOp)sf[3]));
    SH_SEND(SH_SOPB, !memcmp(SH.sop[1], sb, sizeof sb), memcpy(SH.sop[1], sb, sizeof sb), vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)sb[0], (VkStencilOp)sb[1], (VkStencilOp)sb[2], (VkCompareOp)sb[3]));
    }
    SH_SEND(SH_CMF, SH.cmask[0] == S.st_vmask[0], SH.cmask[0] = S.st_vmask[0], vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_vmask[0]));
    SH_SEND(SH_CMB, SH.cmask[1] == S.st_vmask[1], SH.cmask[1] = S.st_vmask[1], vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_vmask[1]));
    SH_SEND(SH_WMF, SH.wmask[0] == S.st_wmask[0], SH.wmask[0] = S.st_wmask[0], vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_wmask[0]));
    SH_SEND(SH_WMB, SH.wmask[1] == S.st_wmask[1], SH.wmask[1] = S.st_wmask[1], vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_wmask[1]));
    SH_SEND(SH_REFF, SH.ref[0] == S.st_ref[0], SH.ref[0] = S.st_ref[0], vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_ref[0]));
    SH_SEND(SH_REFB, SH.ref[1] == S.st_ref[1], SH.ref[1] = S.st_ref[1], vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, S.st_ref[1]));
    float bz[3] = { 0, 0, 0 }; if (S.poly.offset_en & 4) { bz[0] = S.po.units; bz[1] = g_f_biasclamp ? S.po.clamp : 0.0f; bz[2] = S.po.factor; }   // (no depthBiasClamp: unclamped)
    SH_SEND(SH_BIAS, !memcmp(SH.bias, bz, sizeof bz), memcpy(SH.bias, bz, sizeof bz), vkCmdSetDepthBias(cb, bz[0], bz[1], bz[2]));
    SH_SEND(SH_BLEND, !memcmp(SH.blend, S.blend_color, sizeof SH.blend), memcpy(SH.blend, S.blend_color, sizeof SH.blend), vkCmdSetBlendConstants(cb, S.blend_color));
    }
    #undef SH_SEND
    // vertex streams: the pool buffers at their addresses, strides from the stream state; binding 16 = zeros for unbound attributes
    if (!sh) {   // (all 17, looked up and bound for every draw: DM_VK_STATE_SHADOW=0, and transform feedback captures)
    VkBuffer vb[17]; VkDeviceSize vo[17], vs_[17];
    for (int s_ = 0; s_ < 16; s_++) { VkBuffer b; VkDeviceSize o;
        if (S.vbuf[s_].address && pool_buf(S.vbuf[s_].address, 1, &b, &o, NULL)) { vb[s_] = b; vo[s_] = o; } else { vb[s_] = g_zero_buf; vo[s_] = 0; }
        vs_[s_] = S.stream[s_].stride; }
    vb[16] = g_zero_buf; vo[16] = 0; vs_[16] = 0;
    if (g_eds) vkCmdBindVertexBuffers2(cb, 0, 17, vb, vo, NULL, vs_);
    else vkCmdBindVertexBuffers(cb, 0, 17, vb, vo);   // (the strides are in the pipeline: sst)
    g_sh_sent += cnt; if (cnt) { g_sh_vb_all += 17; g_sh_vb_sent += 17; }
    } else {   // only the bindings the pipeline reads (vbm, from pso_get), looked up under one lock, and bound only where they changed: one call over the changed span
    VkBuffer vb[17]; VkDeviceSize vo[17], vs_[17]; uint32_t chg = 0;
    for (int b = 0; b < 16; b++) if ((vbm >> b & 1) && S.vbuf[b].address) GOLDEN_MEM(S.vbuf[b].address, 1);   // (as pool_buf notes them for a golden recording)
    pthread_mutex_lock(&g_mu);
    for (int b = 0; b < 17; b++) { if (!(vbm >> b & 1)) continue;
        int i = b < 16 && S.vbuf[b].address ? pool_find(S.vbuf[b].address, 1) : -1;
        if (i >= 0) pool_use_locked(i);   // (backed now if the GPU never read it as a buffer: as pool_buf does)
        if (i >= 0) { vb[b] = g_pools[i].buf; vo[b] = S.vbuf[b].address - g_pools[i].base; } else { vb[b] = g_zero_buf; vo[b] = 0; }
        vs_[b] = b < 16 ? S.stream[b].stride : 0; }
    pthread_mutex_unlock(&g_mu);
    for (int b = 0; b < 17; b++) if (vbm >> b & 1) { g_sh_vb_all++;
        if (!(SH.vb_have >> b & 1) || SH.vb[b] != vb[b] || SH.vo[b] != vo[b] || (g_eds && SH.vs[b] != vs_[b])) chg |= 1u << b; }
    if (!chg) g_sh_skip++;
    else {
        int lo = __builtin_ctz(chg), hi = 31 - __builtin_clz(chg);
        for (int b = lo; b <= hi; b++) if (!(chg >> b & 1)) {   // (between two that changed: what the driver has already, or zeros where it has nothing the pipeline reads)
            if (SH.vb_have >> b & 1) { vb[b] = SH.vb[b]; vo[b] = SH.vo[b]; vs_[b] = SH.vs[b]; } else { vb[b] = g_zero_buf; vo[b] = 0; vs_[b] = 0; } }
        if (g_eds) vkCmdBindVertexBuffers2(cb, (uint32_t)lo, (uint32_t)(hi - lo + 1), vb + lo, vo + lo, NULL, vs_ + lo);
        else vkCmdBindVertexBuffers(cb, (uint32_t)lo, (uint32_t)(hi - lo + 1), vb + lo, vo + lo);
        for (int b = lo; b <= hi; b++) { SH.vb[b] = vb[b]; SH.vo[b] = vo[b]; SH.vs[b] = vs_[b]; SH.vb_have |= 1u << b; }
        g_sh_vb_sent += (uint64_t)__builtin_popcount(chg); g_sh_sent++;
    }
    }
    if (mark) mark[2] = mono_ns();
    VkPipelineLayout pl = prog_layout(pr, 0); g_np = pr->np[0];
    { int any = 0; for (int st = 0; st < 5; st++) any |= pr->sh[st] && pr->sh[st]->nres;
      if (any) { pthread_mutex_lock(&g_rmu); g_dsh = sh && !g_np; g_dsh_cnt = cnt; for (int st = 0; st < 5; st++) push_stage_h(pr->sh[st], st, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 1); g_dsh = 0; pthread_mutex_unlock(&g_rmu); } }
    if (sh && g_np) sh_desc_forget();   // (drawn with a descriptor set, bound at 0 below: the pushed descriptors are disturbed)
    if (g_lc_hazard && !tfcap) {   // (lazy clears: this draw samples the depth image whose clear its pass took by the load op, so memory does not hold the clear yet. The pass is
        //   ended (storing the clear and the draws so far) and begun again, loading, and the draw recorded there from the start: memory holds what the clear-only pass
        //   and the passes after it left, as before. Nothing of this draw was drawn: its state and bindings in the old pass are left unused)
        g_lc_hazard = 0; g_lc_redo++; D.n = 0;
        pass_end(); if (g_nlc) lc_out_all();
        if (pass_begin()) draw_(dw, di, NULL);
        return;
    }
    if (mark) mark[3] = mono_ns();
    if (tfcap) {   // the capture buffers, then the draw, then its writes made visible to what reads them (vertex streams, shaders, copies)
        VkWriteDescriptorSet w_[4]; VkDescriptorBufferInfo bi_[4]; uint32_t nw = 0;
        VkWriteDescriptorSet *w = w_; VkDescriptorBufferInfo *bi = bi_;
        if (g_np) { if (D.n + 4 > 320) D.n = 316; w = D.w + D.n; bi = D.bi + D.n; }
        for (int b = 0; b < 4; b++) if (tfb[b]) { bi[nw] = (VkDescriptorBufferInfo){ tfb[b], tfo[b], tfr[b] };
            w[nw] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = TF_BIND + (uint32_t)b, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[nw] }; nw++; }
        if (g_np) D.n += nw; else if (nw) p_push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, nw, w);
        desc_flush(VK_PIPELINE_BIND_POINT_GRAPHICS, pl, pr->dsl[0]);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof tfpc, tfpc);
        // On Arm's own driver (or DM_VK_TF_CHECK=1): the first capture's output starts as a marker, read back 60 frames later (tf_check): still the marker = the captures
        //   wrote nothing on this driver (#6). Written through the guest's memory before the submission runs, as a host write (mirrored pools take it in at submit).
        //   vk_quirks.h Q_TF_CHECK
        { static int chk = -1; if (chk < 0) { const char *e = getenv("DM_VK_TF_CHECK"); chk = quirk_note(Q_TF_CHECK, e ? atoi(e) != 0 : quirk_default(Q_TF_CHECK, &g_gpuid), e != NULL); }
          extern uint64_t g_tfchk_at, g_tfchk_frame; extern uint32_t g_tfchk_n;
          if (chk == 1 && tfb[0] && pr->sh[0]->tf_stride[0]) { chk = 2; uint64_t at = S.tf[0].addr + S.tf_off[0], n = (uint64_t)dw->count * pr->sh[0]->tf_stride[0]; if (n > 64) n = 64;
              n &= ~3ull; if (n) { dm_guest_write_prepare((void *)(uintptr_t)at, n); snapc_drop(at, n); for (uint64_t i = 0; i < n; i += 4) *(uint32_t *)(uintptr_t)(at + i) = 0x7fc0deadu;
                  g_tfchk_at = at; g_tfchk_n = (uint32_t)n; g_tfchk_frame = g_frame_no; } } }
        if (pr->sh[0]->dp) { uint32_t dv[3] = { (uint32_t)dw->first, (uint32_t)dw->base_instance, 0 }; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, SPV_DRAWP_PC_OFFSET, sizeof dv, dv); }
        vkCmdDraw(cb, (uint32_t)dw->count, 1, (uint32_t)dw->first, (uint32_t)dw->base_instance);
        vshader *vs = pr->sh[0];
        if (tfp_n) {   // as points: the targets into the capture buffers
            int nc_[4]; for (int b = 0; b < 4; b++) nc_[b] = vs->tf[b][1];
            pass_end(); tfp_repack(CB(), tfp_n, (uint32_t)dw->count, tfb, tfo, tfr, tfp_base, nc_); }
        for (int b = 0; b < 4; b++) if (tfb[b]) { uint64_t n = (uint64_t)dw->count * vs->tf_stride[b], at = S.tf[b].addr + S.tf_off[b]; wb_note(at, n); S.tf_off[b] += n;
            if (g_ntfw < 64) { g_tfw[g_ntfw].lo = at; g_tfw[g_ntfw].hi = at + n; g_ntfw++; } }
        if (!tf_batch_on()) pass_end();   // (batched: the pass stays open for the next capture; its end makes the writes visible)
        { static int once; if (!once++) VLOG("transform feedback: first capture (vertex %s, %d vertices)", vs->key, dw->count); }
        return;
    }
    desc_flush(VK_PIPELINE_BIND_POINT_GRAPHICS, pl, pr->dsl[0]);
    uint32_t inst = (uint32_t)((dw ? dw->instances : di->instances) > 0 ? (dw ? dw->instances : di->instances) : 1);
    if (pr->sh[0]->dp) {   // a device without draw parameters: the draw's base vertex / base instance, as the shader reads them (spv_no_drawparams); 0 for an indirect draw
        int ind = (dw && dw->ind) || (di && di->ind);
        uint32_t dv[3] = { ind ? 0u : dw ? (uint32_t)dw->first : (uint32_t)di->base_vertex, ind ? 0u : (uint32_t)(dw ? dw->base_instance : di->base_instance), 0 };
        if (ind) RED_ONCE(130, "an indirect draw with vertex shader %s, which reads a draw parameter, on a device without them: read as 0", pr->sh[0]->key);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, SPV_DRAWP_PC_OFFSET, sizeof dv, dv);
    }
    if ((dw && dw->ind) || (di && di->ind)) {
        VkBuffer ab; VkDeviceSize ao; uint64_t ind = dw ? dw->ind : di->ind;
        if (prim == 6 || prim == 7) { RED_ONCE(35, "indirect fan / quad draw not supported yet"); return; }
        if (!pool_buf(ind, dw ? 16 : 20, &ab, &ao, NULL)) { RED_ONCE(36, "indirect arguments in no pool"); return; }
        if (dw) vkCmdDrawIndirect(cb, ab, ao, 1, 16);
        else { VkBuffer ib; VkDeviceSize io; if (di->itype == 0) { RED_ONCE(37, "indirect draw with 8-bit indices not supported yet"); return; }
               if (!pool_buf(di->ibuf, 1, &ib, &io, NULL)) { RED_ONCE(38, "index buffer in no pool"); return; }
               sh_index(cb, sh, cnt, ib, io, di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32); vkCmdDrawIndexedIndirect(cb, ab, ao, 1, 20); }
        return;
    }
    if (dw && (prim == 6 || prim == 7)) {   // fans / quads: expanded into a 32-bit triangle index list in the arena
        uint32_t nt = prim == 7 ? (uint32_t)(dw->count / 4) * 2 : dw->count >= 3 ? (uint32_t)dw->count - 2 : 0; if (!nt) return;
        VkBuffer ib; VkDeviceSize io; uint32_t *d = (uint32_t *)arena_alloc(g_sub, (VkDeviceSize)nt * 12, &ib, &io);
        for (uint32_t t = 0; t < nt; t++) { if (prim == 6) { d[3 * t] = 0; d[3 * t + 1] = t + 1; d[3 * t + 2] = t + 2; } else { uint32_t q = (t >> 1) * 4, hh = t & 1; d[3 * t] = q; d[3 * t + 1] = q + 1 + hh; d[3 * t + 2] = q + 2 + hh; } }
        sh_index(cb, sh, cnt, ib, io, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, nt * 3, inst, 0, dw->first, (uint32_t)dw->base_instance); return;
    }
    if (dw) { if (dw->count) vkCmdDraw(cb, (uint32_t)dw->count, inst, (uint32_t)dw->first, (uint32_t)dw->base_instance); return; }
    if (!di->count) return;
    int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
    if (di->itype == 0 || prim == 6 || prim == 7) {   // 8-bit indices / indexed fans and quads: rewritten into 16- / 32-bit indices in the arena from guest memory
        const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf; if (!src) return;
        #define IX(i) (isz == 1 ? (uint32_t)src[i] : isz == 2 ? (uint32_t)((const uint16_t *)src)[i] : ((const uint32_t *)src)[i])
        VkBuffer ib; VkDeviceSize io; uint32_t n;
        if (prim == 6 || prim == 7) {
            uint32_t nt = prim == 7 ? (uint32_t)(di->count / 4) * 2 : di->count >= 3 ? (uint32_t)di->count - 2 : 0; if (!nt) return;
            uint32_t *d = (uint32_t *)arena_alloc(g_sub, (VkDeviceSize)nt * 12, &ib, &io);
            for (uint32_t t = 0; t < nt; t++) { if (prim == 6) { d[3 * t] = IX(0); d[3 * t + 1] = IX(t + 1); d[3 * t + 2] = IX(t + 2); } else { uint32_t q = (t >> 1) * 4, hh = t & 1; d[3 * t] = IX(q); d[3 * t + 1] = IX(q + 1 + hh); d[3 * t + 2] = IX(q + 2 + hh); } }
            sh_index(cb, sh, cnt, ib, io, VK_INDEX_TYPE_UINT32); n = nt * 3;
        } else {
            uint16_t *d = (uint16_t *)arena_alloc(g_sub, (VkDeviceSize)di->count * 2, &ib, &io); for (int32_t i = 0; i < di->count; i++) d[i] = src[i];
            sh_index(cb, sh, cnt, ib, io, VK_INDEX_TYPE_UINT16); n = (uint32_t)di->count;
        }
        #undef IX
        vkCmdDrawIndexed(cb, n, inst, 0, di->base_vertex, (uint32_t)di->base_instance); return;
    }
    VkBuffer ib; VkDeviceSize io;
    if (!pool_buf(di->ibuf, (uint64_t)di->count * isz, &ib, &io, NULL)) { RED_ONCE(38, "index buffer in no pool"); return; }
    sh_index(cb, sh, cnt, ib, io, di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
}
static nvn_hmap g_cps = {.klen = sizeof(void *)};
static int b10_quantize_region(VkCommandBuffer cb, vtex *x, uint32_t level, uint32_t z0, uint32_t nz, int32_t x0, int32_t y0, uint32_t w, uint32_t h);
static void dispatch(const p_dispatch *dp) {
    if (g_nlc) lc_out_all();   // (kept depth clears written out before the dispatch records anything, and before its barrier: it may read or write any image it binds)
    if (!S.cprog) { RED_ONCE(39, "dispatch with no program bound"); return; }
    vprog *pr = prog_get(S.cprog); vshader *cs = pr->sh[5]; if (!cs) { RED_ONCE(40, "dispatched program has no compute shader (or it is not mapped)"); return; }
    VkPipelineLayout pl = prog_layout(pr, 1); g_np = pr->np[1];
    void *key = pr; VkPipeline p = nvn_hm_get(&g_cps, &key);
    if (!p) { VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs->mod, .pName = "main" }, .layout = pl };
        if (vkCreateComputePipelines(g_dev, g_pcache, 1, &ci, NULL, &p)) { VLOG("[RED] compute pipeline %s failed", cs->key); return; }
        nvn_hm_put(&g_cps, &key, p); cs_record(cs); }   // (listed for the next start's prewarm: prewarm_compute)
    outside_pass();
    VkCommandBuffer cb = CB();
    if (atomic_load(&g_cap) == 2) { cap_printf("compute dispatch %ux%ux%u%s: shader %s\n", (unsigned)dp->x, (unsigned)dp->y, (unsigned)dp->z, dp->ind ? " (indirect)" : "", cs->key); g_cap_ntex = 0; }
    g_dispatch_dim[0] = (uint32_t)dp->x; g_dispatch_dim[1] = (uint32_t)dp->y; g_dispatch_dim[2] = (uint32_t)dp->z; g_c0_buf = VK_NULL_HANDLE;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    g_nb10q = 0;   // (the stand-in bound below as a storage image declared without a format: b10_storage_bind)
    push_stage(cs, 5, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
    desc_flush(VK_PIPELINE_BIND_POINT_COMPUTE, pl, pr->dsl[1]);
    if (dp->ind) { VkBuffer ab; VkDeviceSize ao; if (!pool_buf(dp->ind, 12, &ab, &ao, NULL)) { RED_ONCE(41, "indirect dispatch arguments in no pool"); return; }
        if (g_c0_buf) { VkBufferCopy c = { ao, g_c0_off, 12 }; vkCmdCopyBuffer(cb, ab, g_c0_buf, 1, &c); barrier_all(cb); }   // the shader reads the counts from c0 + 0x200
        vkCmdDispatchIndirect(cb, ab, ao); }
    else vkCmdDispatch(cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    barrier_all(cb);
    // DM_VK_B10_AS_RGBA16: what the dispatch stored into the stand-in through an image declared without a format, made what R11G11B10F holds (range, rounding, alpha 1)
    for (int i = 0; i < g_nb10q; i++) { vtex *x = g_b10q[i].x; uint32_t lv = g_b10q[i].lv;
        if (x->samples > 1 || !b10_quantize_region(cb, x, lv, g_b10q[i].z0, g_b10q[i].nz, 0, 0, x->w >> lv ? x->w >> lv : 1, x->h >> lv ? x->h >> lv : 1))
            RED_ONCE(162, "b10 swap: what a dispatch stored into the stand-in (declared without a format) could not be made what R11G11B10F holds (no conversion shader): kept as it is"); }
    g_nb10q = 0;
}

// ================= Buffer operations (in submission order) =================
static void sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!pool_buf(src, n, &sb, &so, NULL) || !pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(42, "buffer copy %#llx → %#llx outside the pools", (unsigned long long)src, (unsigned long long)dst); return; }
    if (P.open) g_brk[2]++;
    outside_pass();
    if (sb == db && so < dof + n && dof < so + n) {   // overlapping ranges of one buffer: through the arena
        VkBuffer tb; VkDeviceSize to; arena_alloc(g_sub, n, &tb, &to);
        VkBufferCopy c1 = { so, to, n }; vkCmdCopyBuffer(CB(), sb, tb, 1, &c1); barrier_all(CB()); VkBufferCopy c2 = { to, dof, n }; vkCmdCopyBuffer(CB(), tb, db, 1, &c2);
    } else { VkBufferCopy c = { so, dof, n }; vkCmdCopyBuffer(CB(), sb, db, 1, &c); }
    barrier_all(CB()); wb_note(dst, n);
}
static void sub_update(const void *data, uint64_t dst, uint64_t n) {   // the data is copied now (the game may reuse its command memory before the GPU runs)
    if (!n) return;
    VkBuffer db; VkDeviceSize dof; if (!pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(43, "uniform update outside the pools"); return; }
    // With the uniform snapshots (ubo_snap_on) every draw reads its uniform data as the game's memory holds it when the draw is recorded, so an update recorded
    //   before it must be there by then: written into that memory now, in recording order, as well as by the copy below (which lands only when the GPU runs it,
    //   after every snapshot of this submission was taken: those draws read the data from before the update)
    if (ubo_snap_on()) { dm_guest_write_prepare((void *)(uintptr_t)dst, n); memcpy((void *)(uintptr_t)dst, data, n); g_snap_wt++; snapc_drop(dst, n); }   // (copies of the range taken before it: gone)
    if (P.open && !P_tf && g_nubd < UBD_MAX && ubd_on()) {   // kept until the pass ends (see ubd_flush); draws after it read it through ubd_shadow
        VkBuffer tb; VkDeviceSize to; uint8_t *m = arena_alloc(g_sub, n, &tb, &to); memcpy(m, data, n);
        g_ubd[g_nubd++] = (ubd_ent){ dst, dst + n, tb, to, m }; g_ubd_deferred++; return; }
    if (P.open) g_brk[1]++;
    outside_pass();
    VkBuffer tb; VkDeviceSize to; void *m = arena_alloc(g_sub, n, &tb, &to); memcpy(m, data, n);
    VkBufferCopy c = { to, dof, n }; vkCmdCopyBuffer(CB(), tb, db, 1, &c); barrier_all(CB()); wb_note(dst, n);
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n) return;
    VkBuffer db; VkDeviceSize dof; if (!pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(44, "buffer fill outside the pools"); return; }
    if (P.open) g_brk[3]++;
    outside_pass();
    if (dof % 4 || n % 4) { RED_ONCE(45, "buffer fill not 4-byte aligned (%#llx, %llu)", (unsigned long long)dof, (unsigned long long)n); return; }
    vkCmdFillBuffer(CB(), db, dof, n, value); barrier_all(CB()); wb_note(dst, n);
}

// ================= Texture copies and clears =================
// A scaled / mirrored / format-changing copy into a depth / stencil image whose format is no blit destination (D24S8 on the Adreno 830: vkCmdBlitImage into it is
//   invalid, VUID-vkCmdBlitImage-dstImage-02000, found by the Khronos layer 2026-11-20): drawn instead. One pass with the destination as its depth / stencil attachment:
//   the depth from the source's depth (zcopy_d.frag, gl_FragDepth, nearest like a depth blit), then the stencil one bit a draw (zcopy_s.frag discards where the source's
//   bit is clear; write mask = that bit; Vulkan 1.0 has no stencil export), after the depth draw set it to 0. Plain descriptor sets (no push descriptors), Vulkan 1.0.
//   Anything it cannot take (the same image, multisample, 3D, an image not made for this) keeps the blit. DM_VK_ZCOPY=0: the blit always, as before
#include "overlay/zcopy_spv.h"
typedef struct { float m[4]; int32_t lim[4]; } zcopy_pc;   // source texel = pixel * m.xy + m.zw; lim.xy: the source level's last texel, lim.z: the stencil bit
static struct { VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkSampler samp; int failed, n; VkFormat fmt[6]; VkPipeline pd[6], ps[6]; } ZC;
static int fmt_feat(VkFormat f, VkFormatFeatureFlags want) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, f, &fp); return (fp.optimalTilingFeatures & want) == want; }
static VkPipeline zc_pipe(VkFormat fmt, int sbits) {
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE; VkPipeline p = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo mv = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof zcopy_vert_spv, .pCode = zcopy_vert_spv },
                             mf = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sbits ? sizeof zcopy_s_frag_spv : sizeof zcopy_d_frag_spv, .pCode = sbits ? zcopy_s_frag_spv : zcopy_d_frag_spv };
    if (vkCreateShaderModule(g_dev, &mv, NULL, &vm) || vkCreateShaderModule(g_dev, &mf, NULL, &fm)) goto out;
    int st = fmt_has_stencil(fmt);
    VkPipelineShaderStageCreateInfo sg[2] = { { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main" },
                                              { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main" } };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkStencilOpState so = { VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xff, 0, 0 };   // (write mask and reference: dynamic)
    VkPipelineDepthStencilStateCreateInfo dss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = !sbits, .depthWriteEnable = !sbits,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS, .stencilTestEnable = st, .front = so, .back = so, .maxDepthBounds = 1 };
    VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkDynamicState dyn[4] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 4, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .depthAttachmentFormat = fmt, .stencilAttachmentFormat = st ? fmt : VK_FORMAT_UNDEFINED };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 2, .pStages = sg, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &dss, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = ZC.pl };
    if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(0, NULL, (uint32_t)fmt, 1, 0); }
    if (vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &p)) p = VK_NULL_HANDLE;
out:
    if (vm) vkDestroyShaderModule(g_dev, vm, NULL); if (fm) vkDestroyShaderModule(g_dev, fm, NULL);
    return p;
}
static int zc_setup(VkFormat fmt, VkPipeline *pd, VkPipeline *ps) {   // the two pipelines for a destination format (ps: none without stencil)
    if (ZC.failed) return 0;
    if (!ZC.pl) {
        VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
        VkDescriptorSetLayoutBinding b[2] = { { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL }, { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL } };
        VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b };
        VkPushConstantRange pr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(zcopy_pc) };
        VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &ZC.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
        if (vkCreateSampler(g_dev, &sci, NULL, &ZC.samp) || vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &ZC.dsl) || vkCreatePipelineLayout(g_dev, &pli, NULL, &ZC.pl)) {
            ZC.failed = 1; ZC.pl = VK_NULL_HANDLE; RED_ONCE(137, "depth / stencil texture copy: cannot make its sampler / layouts: blitted as before"); return 0; }
    }
    for (int i = 0; i < ZC.n; i++) if (ZC.fmt[i] == fmt) { *pd = ZC.pd[i]; *ps = ZC.ps[i]; return !!*pd; }
    if (ZC.n == 6) return 0;
    VkPipeline a = zc_pipe(fmt, 0), b = fmt_has_stencil(fmt) ? zc_pipe(fmt, 1) : VK_NULL_HANDLE;
    if (!a || (fmt_has_stencil(fmt) && !b)) { if (a) vkDestroyPipeline(g_dev, a, NULL); if (b) vkDestroyPipeline(g_dev, b, NULL); a = b = VK_NULL_HANDLE;
        RED_ONCE(138, "depth / stencil texture copy: its pipelines failed for format %d: blitted as before", fmt); }
    else VLOG("depth / stencil texture copy ready (format %d%s)", fmt, b ? ", with stencil" : "");
    ZC.fmt[ZC.n] = fmt; ZC.pd[ZC.n] = a; ZC.ps[ZC.n++] = b; *pd = a; *ps = b;
    return !!a;
}
// Draws the copy (outside a pass; the caller orders it after); 0: not taken (blit it). s0 / s1: the source box's corners as a blit's (s1 < s0: mirrored)
static int zcopy_draw(vtex *s, vtex *d, const VkImageSubresourceLayers *ss, const VkImageSubresourceLayers *ds, VkOffset2D s0, VkOffset2D s1, VkOffset2D d0, VkExtent2D de) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_ZCOPY"); on = !e || atoi(e) != 0; }
    if (!on || s == d || s->img == d->img || s->samples > 1 || d->samples > 1 || s->is3d || d->is3d || !de.width || !de.height) return 0;
    if (!(s->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || !(d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || !(s->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !(d->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) return 0;
    if (!fmt_feat(s->fmt, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) || ss->layerCount != ds->layerCount) return 0;
    VkPipeline pd, ps; if (!zc_setup(d->fmt, &pd, &ps)) return 0;
    int stencil = (s->aspect & d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && ps;
    uint32_t sw = s->w >> ss->mipLevel, sh = s->h >> ss->mipLevel, dw = d->w >> ds->mipLevel, dh = d->h >> ds->mipLevel; if (!sw) sw = 1; if (!sh) sh = 1; if (!dw) dw = 1; if (!dh) dh = 1;
    int32_t x0 = d0.x > 0 ? d0.x : 0, y0 = d0.y > 0 ? d0.y : 0, x1 = d0.x + (int32_t)de.width, y1 = d0.y + (int32_t)de.height;
    if (x1 > (int32_t)dw) x1 = (int32_t)dw; if (y1 > (int32_t)dh) y1 = (int32_t)dh;
    if (x1 <= x0 || y1 <= y0) return 1;   // (nothing of it inside the image)
    zcopy_pc pc; pc.m[0] = (float)(s1.x - s0.x) / (float)de.width; pc.m[1] = (float)(s1.y - s0.y) / (float)de.height;
    pc.m[2] = (float)s0.x - (float)d0.x * pc.m[0]; pc.m[3] = (float)s0.y - (float)d0.y * pc.m[1]; pc.lim[0] = (int32_t)sw - 1; pc.lim[1] = (int32_t)sh - 1; pc.lim[2] = pc.lim[3] = 0;
    VkViewport vpt = { (float)d0.x, (float)d0.y, (float)de.width, (float)de.height, 0, 1 }; VkRect2D sc = { { x0, y0 }, { (uint32_t)(x1 - x0), (uint32_t)(y1 - y0) } };
    g_gfx_bind_gen++;   // (draw_ binds the game's pipeline again at its next draw: this command buffer's is the copy's now)
    for (uint32_t z = 0; z < ss->layerCount; z++) {
        VkImageView sv = view_get(s, VK_IMAGE_VIEW_TYPE_2D, s->fmt, ss->mipLevel, 1, ss->baseArrayLayer + z, 1, SW_ID, VK_IMAGE_ASPECT_DEPTH_BIT),
                    stv = stencil ? view_get(s, VK_IMAGE_VIEW_TYPE_2D, s->fmt, ss->mipLevel, 1, ss->baseArrayLayer + z, 1, SW_ID, VK_IMAGE_ASPECT_STENCIL_BIT) : VK_NULL_HANDLE,
                    dv = view_get(d, VK_IMAGE_VIEW_TYPE_2D, d->fmt, ds->mipLevel, 1, ds->baseArrayLayer + z, 1, SW_ID, d->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
        if (!sv || !dv || (stencil && !stv)) return z ? 1 : 0;
        VkDescriptorSet set = desc_set_alloc(ZC.dsl);
        VkDescriptorImageInfo ii[2] = { { ZC.samp, sv, VK_IMAGE_LAYOUT_GENERAL }, { ZC.samp, stv, VK_IMAGE_LAYOUT_GENERAL } };
        VkWriteDescriptorSet wr[2] = { { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0] },
                                       { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[1] } };
        vkUpdateDescriptorSets(g_dev, stencil ? 2 : 1, wr, 0, NULL);
        VkRenderingAttachmentInfo za = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
        VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1, .pDepthAttachment = &za, .pStencilAttachment = fmt_has_stencil(d->fmt) ? &za : NULL };
        // Begun through rend_begin, as the game's own passes: in a held frame (DM_VK_FRAME_HOLD=2: Faster rendering's Auto on Adreno) the submission's command
        //   buffer outside a pass is a secondary (a gap, seg_gap_begin), where a render pass cannot begin (VUID-vkCmdBeginRenderPass-bufferlevel: Adreno's render
        //   passes, Q_NO_DYNREN). rend_begin makes the copy a pass of its own there: recorded into a secondary that inherits it, the pass begun around it in the primary
        //   at the flush (seg_record_prim). CB () is that secondary until rend_end; elsewhere it is the submission's command buffer, as before
        rend_begin(CB(), &ri, NULL, (uint32_t)d->fmt, 1, 0);
        VkCommandBuffer cb = CB();
        vkCmdSetViewport(cb, 0, 1, &vpt); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, ZC.pl, 0, 1, &set, 0, NULL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pd);
        vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil ? 0xff : 0); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);   // (the stencil to 0, its bits drawn next)
        vkCmdPushConstants(cb, ZC.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, 3, 1, 0, 0);
        if (stencil) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, ps); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
            for (int bit = 0; bit < 8; bit++) { pc.lim[2] = bit; vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 1u << bit);
                vkCmdPushConstants(cb, ZC.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc); vkCmdDraw(cb, 3, 1, 0, 0); }
            pc.lim[2] = 0;
        }
        rend_end(cb);
        if (z + 1 < ss->layerCount) barrier_all(CB());   // (a held frame: in the gap after the copy's pass)
    }
    static int said; if (!said++) VLOG("a texture copy into a depth / stencil image (format %d, no blit destination on this GPU) drawn instead (%s)", d->fmt, stencil ? "depth and stencil" : "depth");
    return 1;
}
// DM_VK_LOG_COPY (diagnosis): each distinct copy (source / destination serials) once, up to 400
static int log_copy_pair(uint64_t a, uint64_t b) { static int on = -1, n; static uint64_t seen[400][2]; if (on < 0) on = getenv("DM_VK_LOG_COPY") != NULL;
    if (!on) return 0; for (int i = 0; i < n; i++) if (seen[i][0] == a && seen[i][1] == b) return 0;
    if (n >= 400) return 0; seen[n][0] = a; seen[n][1] = b; n++; return 1; }
// ---- R11G11B10F's stand-in in copies (DM_VK_B10_AS_RGBA16, see b10_on): the game's 4-byte packed texels and the 8-byte stand-in meet in a compute shader on buffers
//   (b10_conv.comp: the rules of b10_conv.h bit for bit), between the game's memory and a copy of the region in the submission's arena, which the image is copied to or
//   from. On the GPU, in the submission's order: the copy sees what the GPU wrote before it, and what it writes is there for what follows. Any Vulkan 1.0 GPU (no
//   format features asked of R11G11B10F: lavapipe and Adreno 5xx cannot blit into it); the stand-in's usage is not changed (a storage image would lose Mali's AFBC)
#include "b10_conv_spv.h"
static VkShaderModule g_b10c_sm; static VkDescriptorSetLayout g_b10c_dsl; static VkPipelineLayout g_b10c_pl; static VkPipeline g_b10c_cs; static int g_b10c_ready;
static int b10c_init(void) {
    if (g_b10c_ready) return g_b10c_ready > 0;
    g_b10c_ready = -1;
    VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof b10_conv_spv, .pCode = b10_conv_spv };
    VkDescriptorSetLayoutBinding lb[2] = { { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = g_nopush ? 0 : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = lb };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 36 };
    if (vkCreateShaderModule(g_dev, &mi, NULL, &g_b10c_sm) || vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_b10c_dsl)) goto fail;
    { VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_b10c_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
      if (vkCreatePipelineLayout(g_dev, &pi, NULL, &g_b10c_pl)) goto fail;
      VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_b10c_sm, .pName = "main" }, .layout = g_b10c_pl };
      if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_b10c_cs)) goto fail; }
    g_b10c_ready = 1; VLOG("b10 swap: R11G11B10F <-> RGBA16F conversion shader ready (copies between the stand-in and the game's memory)");
    return 1;
fail:
    VLOG("[RED] b10 swap: the conversion shader could not be made: copies in from the game's memory are expanded on the CPU, copies out are left out");
    return 0;
}
// mode 0 pack / 1 expand / 2 quantize (b10_conv.comp) of w x h x n texels: the RGBA16F ones at hoff in hbuf (tight), the packed ones at koff in kbuf (krow / kimg words
//   apart; mode 2: none). 0 when it cannot run here (the shader, a binding past the device's storage-buffer range): nothing recorded then
static int b10c_run(VkCommandBuffer cb, int mode, VkBuffer hbuf, VkDeviceSize hoff, VkBuffer kbuf, VkDeviceSize koff, VkDeviceSize kbytes, uint32_t w, uint32_t h, uint32_t n, uint32_t krow, uint32_t kimg) {
    if (!b10c_init()) return 0;
    VkDeviceSize al = g_props.limits.minStorageBufferOffsetAlignment ? g_props.limits.minStorageBufferOffsetAlignment : 256, hb0 = hoff / al * al, kb0 = koff / al * al;
    VkDeviceSize nt = (VkDeviceSize)w * h * n, hr = hoff - hb0 + nt * 8, kr = koff - kb0 + kbytes;
    if (mode == 2) { kbuf = hbuf; kb0 = hb0; koff = hoff; kr = hr; }
    if (!nt || hr > g_props.limits.maxStorageBufferRange || kr > g_props.limits.maxStorageBufferRange || ((hoff - hb0) | (koff - kb0)) & 3) return 0;
    VkDescriptorBufferInfo bi[2] = { { hbuf, hb0, hr }, { kbuf, kb0, kr } };
    VkWriteDescriptorSet wd[2] = { { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
                                   { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1] } };
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (g_nopush && !(ds = desc_set_alloc(g_b10c_dsl))) return 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_b10c_cs);
    if (!g_nopush) p_push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_b10c_pl, 0, 2, wd);
    else { wd[0].dstSet = wd[1].dstSet = ds; vkUpdateDescriptorSets(g_dev, 2, wd, 0, NULL); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_b10c_pl, 0, 1, &ds, 0, NULL); }
    uint32_t pc[9] = { w, h, n, (uint32_t)mode, (uint32_t)((hoff - hb0) / 4), (uint32_t)((koff - kb0) / 4), krow, kimg, 0 };
    uint32_t most = (g_props.limits.maxComputeWorkGroupCount[0] < 65535u ? g_props.limits.maxComputeWorkGroupCount[0] : 65535u) * 64u;   // (texels a dispatch takes)
    for (VkDeviceSize i0 = 0; i0 < nt; i0 += most) { pc[8] = (uint32_t)i0; VkDeviceSize left = nt - i0 < most ? nt - i0 : most;
        vkCmdPushConstants(cb, g_b10c_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc); vkCmdDispatch(cb, (uint32_t)((left + 63) / 64), 1, 1); }
    return 1;
}
// the game's packed texels (buffer kb at ko, row / img bytes apart) to or from the stand-in x (level, first layer / slice z0, nz of them, at x0, y0, w x h).
//   1: done; 0: not here (nothing recorded)
static int b10_copy_buf(VkCommandBuffer cb, vtex *x, int to_buffer, VkBuffer kb, VkDeviceSize ko, size_t row, size_t img, uint32_t level, uint32_t z0, uint32_t nz, int32_t x0, int32_t y0, uint32_t w, uint32_t h) {
    // DM_VK_B10_CPU_COPY=1 (a test switch, golden_check.py b10): copies from the game's memory into the stand-in take the CPU route, as where the shader cannot run
    static int cpu = -1; if (cpu < 0) { const char *e = getenv("DM_VK_B10_CPU_COPY"); cpu = e && atoi(e) > 0; if (cpu && b10_on()) VLOG("b10 swap: copies from the game's memory into the stand-in expanded on the CPU (DM_VK_B10_CPU_COPY, a test)"); }
    if (cpu && !to_buffer) return 0;
    if (!w || !h || !nz || (ko | row | img) & 3 || !g_sub || !b10c_init()) return 0;
    VkDeviceSize kbytes = (VkDeviceSize)img * (nz - 1) + (VkDeviceSize)row * (h - 1) + (VkDeviceSize)w * 4;
    VkBuffer hb; VkDeviceSize ho; arena_alloc(g_sub, (VkDeviceSize)w * h * nz * 8, &hb, &ho);
    VkBufferImageCopy ic = { .bufferOffset = ho, .bufferRowLength = w, .bufferImageHeight = h, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, x->is3d ? 0 : z0, x->is3d ? 1 : nz },
        .imageOffset = { x0, y0, x->is3d ? (int32_t)z0 : 0 }, .imageExtent = { w, h, x->is3d ? nz : 1 } };
    if (to_buffer) {
        vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &ic); barrier_all(cb);
        if (!b10c_run(cb, 0, hb, ho, kb, ko, kbytes, w, h, nz, (uint32_t)(row / 4), (uint32_t)(img / 4))) { RED_ONCE(152, "b10 swap: a copy into the game's memory too big for one binding: left out"); return 1; }
    } else {
        if (!b10c_run(cb, 1, hb, ho, kb, ko, kbytes, w, h, nz, (uint32_t)(row / 4), (uint32_t)(img / 4))) return 0;
        barrier_all(cb); vkCmdCopyBufferToImage(cb, hb, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    }
    barrier_all(cb);
    return 1;
}
// a region of the stand-in as R11G11B10F stores it (its range and rounding, alpha 1): what a copy or blit from another format wrote into it
static int b10_quantize_region(VkCommandBuffer cb, vtex *x, uint32_t level, uint32_t z0, uint32_t nz, int32_t x0, int32_t y0, uint32_t w, uint32_t h) {
    if (!w || !h || !nz || !g_sub || !b10c_init()) return 0;
    VkBuffer hb; VkDeviceSize ho; arena_alloc(g_sub, (VkDeviceSize)w * h * nz * 8, &hb, &ho);
    VkBufferImageCopy ic = { .bufferOffset = ho, .bufferRowLength = w, .bufferImageHeight = h, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, x->is3d ? 0 : z0, x->is3d ? 1 : nz },
        .imageOffset = { x0, y0, x->is3d ? (int32_t)z0 : 0 }, .imageExtent = { w, h, x->is3d ? nz : 1 } };
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &ic); barrier_all(cb);
    if (!b10c_run(cb, 2, hb, ho, VK_NULL_HANDLE, 0, 0, w, h, nz, 0, 0)) return 0;
    barrier_all(cb); vkCmdCopyBufferToImage(cb, hb, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic); barrier_all(cb);
    return 1;
}
static void copy_b2t(const p_copy_b2t *c, int to_buffer) {
    vtex *x = tex_res(c->tex);
    sk_read(x);   // copied from, or written in part
    if (log_copy_pair(c->tex->serial, c->src | (uint64_t)to_buffer << 63)) VLOG("frame %llu copy %s: texture ser %llu %ux%u format %#x flags %#x region %d,%d %dx%d, buffer %#llx%s", (unsigned long long)g_frame_no, to_buffer ? "texture→buffer" : "buffer→texture", (unsigned long long)c->tex->serial,
                         c->tex->d.width, c->tex->d.height, c->tex->d.format, c->tex->d.flags, c->r.xoffset, c->r.yoffset, c->r.width, c->r.height, (unsigned long long)c->src, x && x->img ? "" : " (no image)");
    if (!x || !x->img) return;
    if (x->drop) { RED_ONCE(104, "a texture kept at half quality copied %s a buffer: left out (DM_VK_TEX_QUALITY=full to compare)", to_buffer ? "to" : "from"); return; }
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;
    int nz = c->r.depth ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize o; if (!pool_buf(c->src, img * nz, &b, &o, NULL)) { RED_ONCE(46, "texture copy buffer side outside the pools"); return; }
    if (row % bpb || (img % row)) { RED_ONCE(47, "texture copy pitch %zu not a whole number of %d-byte blocks", row, bpb); return; }
    if (b10_swapped_tex(x) && bpb == 4) {   // R11G11B10F's stand-in (DM_VK_B10_AS_RGBA16): the game's packed texels converted on the GPU (b10_copy_buf), in order
        if (x->samples > 1 || c->r.width <= 0 || c->r.height <= 0) { b10_note("a multisampled or empty copy between the stand-in and a buffer: left out", x, 0); return; }
        outside_pass();
        if (b10_copy_buf(CB(), x, to_buffer, b, o, row, img, level, x->is3d ? (uint32_t)c->r.zoffset : layer0 + (uint32_t)c->r.zoffset, (uint32_t)nz, c->r.xoffset, c->r.yoffset, (uint32_t)c->r.width, (uint32_t)c->r.height)) {
            if (to_buffer) wb_note(c->src, img * nz);
            b10_note(to_buffer ? "a copy into the game's memory: packed on the GPU" : "a copy from the game's memory: expanded on the GPU", x, 0); return; }
        // no blit between the formats on this GPU: in, from the CPU's view of the game's memory (what the GPU wrote there earlier in the frame is not seen); out, left out
        if (to_buffer) { b10_note("a copy into the game's memory: left out (no blit between the formats on this GPU)", x, 0); return; }
        const uint8_t *src = (const uint8_t *)(uintptr_t)c->src; size_t orow = (size_t)c->r.width * 8, oimg = orow * rows;
        VkBuffer sb; VkDeviceSize so; uint8_t *sp = arena_alloc(g_sub, oimg * nz > 0 ? oimg * nz : 16, &sb, &so);
        for (int s = 0; s < nz; s++) b10_rows_expand(src + (size_t)s * img, row, (uint32_t)c->r.width, rows, sp + (size_t)s * oimg, orow);
        b10_note("a copy from the game's memory: expanded on the CPU (no blit between the formats on this GPU)", x, 0);
        b = sb; o = so; row = orow; img = oimg; bpb = 8;
    } else {
    int bck = nvn_fmt_vk_decode(c->tex->d.format);
    if (bck && to_buffer) { RED_ONCE(98, "a BC texture decoded on upload copied back to a buffer: skipped (format %#x)", c->tex->d.format); return; }
    if (bck && (g_bcgpu & 2) && bc_gpu_can(bck, nvn_fmt_vk_transcode(c->tex->d.format) ? 1 : 0)) {   // the BC layer: from the pool's GPU-side blocks, so blocks the GPU wrote come out right too
        outside_pass();
        VkImageSubresourceLayers sub = { x->aspect, level, x->is3d ? 0 : layer0 + (uint32_t)c->r.zoffset, 1 };
        if (bc_gpu_buffer_to_image(CB(), bck, nvn_fmt_vk_transcode(c->tex->d.format) ? 1 : 0, b, o, (uint32_t)row, (uint32_t)img, (uint32_t)c->r.width, (uint32_t)c->r.height, (uint32_t)nz, x->img, sub,
                                   (VkOffset3D){ c->r.xoffset, c->r.yoffset, x->is3d ? c->r.zoffset : 0 }, x->is3d)) { g_bc_gpu_copies++; barrier_all(CB()); return; }
    }
    if (bck) {   // BC blocks the game copies in: decoded now (the CPU has written them) into the submission's arena, then copied as plain texels
        const uint8_t *src = (const uint8_t *)(uintptr_t)c->src; int etk = nvn_fmt_vk_transcode(c->tex->d.format), tb = etk ? bc_transcode_bytes(bck) : bc_texel_bytes(bck), blk = etk && bc_transcode_blocks(bck);
        size_t orow = blk ? (size_t)((c->r.width + 3) / 4) * tb : (size_t)c->r.width * tb, oimg = orow * (blk ? (size_t)((c->r.height + 3) / 4) : (size_t)c->r.height);
        VkBuffer sb; VkDeviceSize so; uint8_t *sp = arena_alloc(g_sub, oimg * nz ? oimg * nz : 16, &sb, &so);
        for (int s = 0; s < nz; s++) (etk ? bc_transcode_image : bc_decode_image_mt)(bck, src + s * img, row, c->r.width, c->r.height, sp + s * oimg, orow);
        b = sb; o = so; row = orow; img = oimg; bpb = tb; if (!blk) bw = bh = 1;
    }
    }
    outside_pass();
    VkBufferImageCopy r = { .bufferOffset = o, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(img / row * bh),
        .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, level, x->is3d ? 0 : layer0 + (uint32_t)c->r.zoffset, x->is3d ? 1 : (uint32_t)nz },
        .imageOffset = { c->r.xoffset, c->r.yoffset, x->is3d ? c->r.zoffset : 0 }, .imageExtent = { (uint32_t)c->r.width, (uint32_t)c->r.height, x->is3d ? (uint32_t)nz : 1 } };
    if (to_buffer) depth_read(x);   // (DM_VK_DEPTH_DISCARD)
    if (to_buffer) vkCmdCopyImageToBuffer(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r); else vkCmdCopyBufferToImage(CB(), b, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier_all(CB()); if (to_buffer) wb_note(c->src, img * nz);
}
static void copy_t2t(const p_copy_t2t *c) {
    pass_flush();
    vtex *s = tex_res(c->src), *d = tex_res(c->dst);
    sk_read(s); sk_read(d);
    if (log_copy_pair(c->src->serial, c->dst->serial)) VLOG("frame %llu copy texture→texture: ser %llu %ux%u format %#x flags %#x region %d,%d %dx%d → ser %llu %ux%u format %#x flags %#x region %d,%d %dx%d%s", (unsigned long long)g_frame_no, (unsigned long long)c->src->serial,
                         c->src->d.width, c->src->d.height, c->src->d.format, c->src->d.flags, c->sr.xoffset, c->sr.yoffset, c->sr.width, c->sr.height, (unsigned long long)c->dst->serial,
                         c->dst->d.width, c->dst->d.height, c->dst->d.format, c->dst->d.flags, c->dr.xoffset, c->dr.yoffset, c->dr.width, c->dr.height, s && d && s->img && d->img ? "" : " (an image missing)");
    if (!s || !d || !s->img || !d->img) return;
    depth_read(s);   // (copied from: DM_VK_DEPTH_DISCARD keeps storing it)
    if (s->drop || d->drop) { RED_ONCE(105, "a texture kept at half quality in a texture copy: left out (DM_VK_TEX_QUALITY=full to compare)"); return; }
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    int nz = c->sr.depth ? c->sr.depth : 1;
    outside_pass();
    VkImageSubresourceLayers ss = { s->aspect, sl, s->is3d ? 0 : sz0 + (uint32_t)c->sr.zoffset, s->is3d ? 1 : (uint32_t)nz }, ds = { d->aspect, dl, d->is3d ? 0 : dz0 + (uint32_t)c->dr.zoffset, d->is3d ? 1 : (uint32_t)nz };
    if (c->flags & DM_COPY_RESOLVE) {
        if (s->samples > 1 && d->samples == 1 && s->fmt == d->fmt) { VkImageResolve r = { ss, {0, 0, 0}, ds, {0, 0, 0}, { s->w, s->h, 1 } }; vkCmdResolveImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
        else if (s->samples == d->samples && s->fmt == d->fmt) { VkImageCopy r = { ss, {0, 0, 0}, ds, {0, 0, 0}, { s->w, s->h, 1 } }; vkCmdCopyImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
        else RED_ONCE(48, "downsample case not implemented");
        barrier_all(CB()); return;
    }
    // Raw BC blocks: a texture of 8- or 16-byte texels (a uint render target the game's GPU wrote blocks into, each texel a block) copied into a BC texture, or a
    //   BC texture into one: Vulkan copies such size-compatible formats block for block (extent in the source's texels); where the GPU has no BC the destination
    //   is decoded / transcoded, and the BC layer converts the blocks. The blit below cannot take a compressed format (the copy was lost before)
    { int sbw, sbh, dbw, dbh, sbpb = nvn_fmt_bpb(c->src->d.format, &sbw, &sbh), dbpb = nvn_fmt_bpb(c->dst->d.format, &dbw, &dbh), dk = nvn_fmt_vk_decode(c->dst->d.format);
      int raw_in = sbw == 1 && dbw == 4 && sbpb == dbpb && (sbpb == 8 || sbpb == 16), raw_out = sbw == 4 && dbw == 1 && sbpb == dbpb && (dbpb == 8 || dbpb == 16);
      if ((raw_in || raw_out) && s->samples == 1 && d->samples == 1) {
          if (raw_in && dk) {
              int mode = nvn_fmt_vk_transcode(c->dst->d.format) ? 1 : 0;
              uint32_t w = (uint32_t)c->dr.width, h = (uint32_t)c->dr.height;
              if (w > (uint32_t)c->sr.width * 4) w = (uint32_t)c->sr.width * 4; if (h > (uint32_t)c->sr.height * 4) h = (uint32_t)c->sr.height * 4;
              if ((g_bcgpu & 4) && bc_gpu_can(dk, mode) && bc_gpu_image_to_image(CB(), dk, mode, s->img, ss, (VkOffset3D){ c->sr.xoffset, c->sr.yoffset, s->is3d ? c->sr.zoffset : 0 }, (uint32_t)c->sr.width,
                      (uint32_t)c->sr.height, (uint32_t)nz, w, h, d->img, ds, (VkOffset3D){ c->dr.xoffset, c->dr.yoffset, d->is3d ? c->dr.zoffset : 0 }, d->is3d)) { g_bc_gpu_copies++; barrier_all(CB()); return; }
              RED_ONCE(108, "raw BC blocks copied into a texture decoded on this GPU (format %#x) and the BC layer cannot take them: left out", c->dst->d.format); barrier_all(CB()); return;
          }
          if (raw_out && nvn_fmt_vk_decode(c->src->d.format)) { RED_ONCE(109, "a BC texture decoded on this GPU copied out as raw blocks (format %#x): left out", c->src->d.format); return; }
          VkImageCopy r = { ss, { c->sr.xoffset, c->sr.yoffset, s->is3d ? c->sr.zoffset : 0 }, ds, { c->dr.xoffset, c->dr.yoffset, d->is3d ? c->dr.zoffset : 0 },
                            { (uint32_t)c->sr.width, (uint32_t)c->sr.height, s->is3d ? (uint32_t)nz : 1 } };
          vkCmdCopyImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); barrier_all(CB());
          { static int said; if (!said++) VLOG("raw BC blocks copied between a %s texture (format %#x) and a %s one (format %#x), block for block", raw_in ? "uint" : "BC", c->src->d.format, raw_in ? "BC" : "uint", c->dst->d.format); }
          return;
      } }
    int same = c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt && !(c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT)) && s->samples == d->samples;
    if (same) { VkImageCopy r = { ss, { c->sr.xoffset, c->sr.yoffset, s->is3d ? c->sr.zoffset : 0 }, ds, { c->dr.xoffset, c->dr.yoffset, d->is3d ? c->dr.zoffset : 0 }, { (uint32_t)c->sr.width, (uint32_t)c->sr.height, s->is3d ? (uint32_t)nz : 1 } };
        vkCmdCopyImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
    else {   // format change / scaling / mirroring: a blit (filtered when asked; depth and integer formats nearest)
        int32_t sx0 = c->sr.xoffset, sy0 = c->sr.yoffset, sx1 = sx0 + c->sr.width, sy1 = sy0 + c->sr.height;
        if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { int32_t t = sx0; sx0 = sx1; sx1 = t; }
        if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { int32_t t = sy0; sy0 = sy1; sy1 = t; }
        VkImageBlit r = { ss, {{sx0, sy0, 0}, {sx1, sy1, 1}}, ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}} };
        int lin = (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && !(s->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT);
        if (s->samples > 1 || d->samples > 1) RED_ONCE(49, "multisample texture blit not supported");
        else if ((d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !fmt_feat(d->fmt, VK_FORMAT_FEATURE_BLIT_DST_BIT)
                 && zcopy_draw(s, d, &ss, &ds, (VkOffset2D){ sx0, sy0 }, (VkOffset2D){ sx1, sy1 }, (VkOffset2D){ c->dr.xoffset, c->dr.yoffset }, (VkExtent2D){ (uint32_t)c->dr.width, (uint32_t)c->dr.height })) {}
        else vkCmdBlitImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    barrier_all(CB());
    // R11G11B10F's stand-in (DM_VK_B10_AS_RGBA16, see b10_on) on one side only. Into it from another format: what the copy or blit wrote is then made what R11G11B10F
    //   would hold (its range and rounding, alpha 1: b10_quantize_region). Out of it: read as it is (its alpha is 1)
    if (b10_swapped_tex(s) != b10_swapped_tex(d)) {
        int q = b10_swapped_tex(d) && d->samples == 1 && c->dr.width > 0 && c->dr.height > 0 &&
                b10_quantize_region(CB(), d, dl, d->is3d ? (uint32_t)c->dr.zoffset : ds.baseArrayLayer, d->is3d ? (uint32_t)nz : ds.layerCount, c->dr.xoffset, c->dr.yoffset, (uint32_t)c->dr.width, (uint32_t)c->dr.height);
        b10_note(b10_swapped_tex(s) ? "a texture copy out of the stand-in into another format: read as it is (alpha 1)"
                 : q ? "a texture copy into the stand-in from another format: made what R11G11B10F holds (its range and rounding)" : "a texture copy into the stand-in from another format: copied as it is",
                 b10_swapped_tex(s) ? s : d, b10_swapped_tex(s) ? d->fmt : s->fmt);
    }
    // A linear texture is plain rows in the game's memory, which its CPU reads: the copied area goes there too. Violet takes the player's ID photo so: the screen
    //   copied into a linear texture, read and compressed on the CPU, and uploaded again for the enrollment form; with the copy left in the GPU image the CPU
    //   read zeros and the photo came out black. DM_VK_NO_LINEAR_WB=1: the image only, as before
    { static int off = -1; if (off < 0) off = getenv("DM_VK_NO_LINEAR_WB") != NULL;
      if (!off && (c->dst->d.flags & NVN_TEXTURE_FLAGS_LINEAR) && c->dst->mem && !nvn_fmt_vk_decode(c->dst->d.format) && !d->is3d && d->samples <= 1 && !dl) {
          int bw, bh, bpb = nvn_fmt_bpb(c->dst->d.format, &bw, &bh);
          if (bpb && bw == 1 && bh == 1) {
              size_t stride = c->dst->d.stride ? c->dst->d.stride : (size_t)c->dst->d.width * bpb;
              p_copy_b2t wb; memset(&wb, 0, sizeof wb);
              wb.src = (uint64_t)(uintptr_t)(c->dst->mem + (size_t)c->dr.yoffset * stride + (size_t)c->dr.xoffset * bpb); wb.tex = c->dst; wb.view = c->dv; wb.has_view = c->has_dv; wb.r = c->dr;
              wb.r.depth = 1;
              int64_t ks = S.copy_stride, ki = S.copy_img_stride; S.copy_stride = (int64_t)stride; S.copy_img_stride = 0;
              copy_b2t(&wb, 1);
              S.copy_stride = ks; S.copy_img_stride = ki;
              static int said; if (!said++) VLOG("linear texture %ux%u (format %#x, stride %zu) written by a texture copy: also written to its memory, which the CPU reads", c->dst->d.width, c->dst->d.height, c->dst->d.format, stride);
          } } }
}
static void clear_tex(const p_clear_tex *c) {
    vtex *x = tex_res(c->tex); if (!x || !x->img) return;
    sk_read(x);   // (counted as a read: the pending store keeps storing)
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    int nz = c->r.depth ? c->r.depth : 1;
    uint32_t w = x->w >> lv ? x->w >> lv : 1, h = x->h >> lv ? x->h >> lv : 1;
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width != w || (uint32_t)c->r.height != h || (c->mask & 0xf) != 0xf) RED_ONCE(50, "partial texture clear (region / mask): cleared whole");
    outside_pass();
    VkImageSubresourceRange rg = { x->aspect, lv, 1, x->is3d ? 0 : l0 + (uint32_t)c->r.zoffset, x->is3d ? 1 : (uint32_t)nz };
    if (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue v; memcpy(&v, c->c.u, 16);
        if (b10_swapped_tex(x)) { v = b10_clear(c->c.f); b10_note("a texture clear: the colour as the packed format stores it, alpha 1", x, 0); }   // (DM_VK_B10_AS_RGBA16)
        vkCmdClearColorImage(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    else { VkClearDepthStencilValue v = { c->c.f[0], x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? c->c.u[1] : c->c.u[0] }; vkCmdClearDepthStencilImage(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    barrier_all(CB());
}
static int scissor_full(const tex_t *t, const view_t *v, int has_v) {
    unsigned lv = has_v && v->set_levels ? v->base_level : 0; uint32_t w = t->d.width >> lv ? t->d.width >> lv : 1, h = (t->d.target == 0 || t->d.target == 3 ? 1 : t->d.height) >> lv; if (!h) h = 1;
    return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)w && (int64_t)R.sc.y + R.sc.h >= (int64_t)h);
}
// Samsung's Xclipse driver crashes recording vkCmdClearAttachments in a layered pass over a 3D image (probe self-test 3d-layered: Xclipse 940 driver 24.0.534 and
//   Xclipse 540 24.0.560, SIGSEGV "while recording the per-layer vkCmdClearAttachments"; a pass per slice drew right on both: 3d-slices). There the same clear (the
//   pass's layer 0, that rectangle) is a pass of its own on that slice, cleared by its load op; the game's pass then opens again on its next draw (loading what is there).
//   DM_VK_CLEAR_BY_PASS=1 / 0 decides on any driver (the driver check's 3d-layered test records the clear the same way here, and asks for =1 on a driver that crashes
//   recording the vkCmdClearAttachments: probe selftest.c t_3d_mode)
static int clear_by_pass(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_CLEAR_BY_PASS"); v = quirk_note(Q_CLEAR_BY_PASS, e ? atoi(e) != 0 : quirk_default(Q_CLEAR_BY_PASS, &g_gpuid), vkq_env_src("DM_VK_CLEAR_BY_PASS"));
                                    if (v) VLOG("clears inside a layered pass over a 3D target: a pass of their own on the slice (Samsung Xclipse driver, or DM_VK_CLEAR_BY_PASS)"); }
    return v;
}
static int clear_slice_by_pass(int ci, const VkClearValue *v, VkRect2D rc) {
    const tex_t *t = ci >= 0 && ci < 8 ? R.rt.color[ci] : NULL; vtex *x = t ? tex_res(t) : NULL; uint32_t cf = ci >= 0 && ci < 8 ? P.cfmt[ci] : 0;
    if (!x || !x->img || !x->is3d || !cf) return 0;
    sk_read(x);
    uint32_t lv = R.rt.has_cview[ci] && R.rt.cview[ci].set_levels ? R.rt.cview[ci].base_level : 0, ly = R.rt.has_cview[ci] && R.rt.cview[ci].set_layers ? R.rt.cview[ci].min_layer : 0;
    VkImageView sv = view_get(x, VK_IMAGE_VIEW_TYPE_2D_ARRAY, (VkFormat)cf, lv, 1, ly, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT);   // the pass's layer 0: the attachment's first slice
    if (!sv) return 0;
    uint32_t samples = P.samples;
    pass_end(); barrier_all(CB());
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = sv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = *v };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = rc, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    rend_begin(CB(), &ri, &cf, 0, samples, 0); rend_end(CB()); barrier_all(CB());
    static int n; if (++n <= 4) VLOG("clear in a layered pass (3D target, slice %u, %ux%u): a pass of its own", ly, rc.extent.width, rc.extent.height);
    return 1;
}
static void clear_in_pass(int color_index, const VkClearValue *v, VkImageAspectFlags asp) {   // a clear inside an open pass (or a partial one): vkCmdClearAttachments over the scissor
    if (!pass_begin()) return;
    // nothing attached there (a slot switched off: format 0, or a target that could not be attached): nothing to clear, and Vulkan forbids naming it
    if ((asp & VK_IMAGE_ASPECT_COLOR_BIT) && (color_index < 0 || color_index >= P.ncolor || !P.cfmt[color_index])) return;
    if (!(asp & VK_IMAGE_ASPECT_COLOR_BIT)) { if (!P.hd) asp &= ~VK_IMAGE_ASPECT_DEPTH_BIT; if (!P.hs) asp &= ~VK_IMAGE_ASPECT_STENCIL_BIT; if (!asp) return; }
    VkClearAttachment a = { asp, (uint32_t)(color_index < 0 ? 0 : color_index), *v };
    VkRect2D rc = {{0, 0}, {P.tw, P.th}};
    if (R.has_sc) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h; if (x1 > (long)P.tw) x1 = P.tw; if (y1 > (long)P.th) y1 = P.th;
        if (x1 <= x0 || y1 <= y0) return; rc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}}; }
    if (P.layers > 1 && (asp & VK_IMAGE_ASPECT_COLOR_BIT) && clear_by_pass() && clear_slice_by_pass(color_index, v, rc)) return;
    VkClearRect cr = { rc, 0, 1 };
    seg_area_note(CB(), rc);   // (inside the held pass's area: VUID-vkCmdClearAttachments-pRects-00016)
    vkCmdClearAttachments(CB(), 1, &a, 1, &cr);
}

// ================= Command replay =================
static void bh_fresh(const p_bind_handles *b) { if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); }
static int state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: { const p_bind_program *b = a; if (b->stages & 0x20) S.cprog = b->prog; if (b->stages & 0x1f) S.prog = bind_stages(b->prog, b->stages & 0x1f); break; }
        case OP_BIND_STATE: { const p_bind_state *b = a; blend_st bs;
            switch (b->kind) { case 0: memcpy(&bs, b->bytes, sizeof bs); S.blend[bs.target] = bs; break; case 1: memcpy(&S.color, b->bytes, sizeof S.color); break; case 2: memcpy(&S.chmask, b->bytes, sizeof S.chmask); break;
                case 3: memcpy(&S.ms, b->bytes, sizeof S.ms); break; case 4: memcpy(&S.poly, b->bytes, sizeof S.poly); break; case 5: memcpy(&S.ds, b->bytes, sizeof S.ds); break; }
            break; }
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; memcpy(S.attr, b->st, 4 * b->n); S.nattr = b->n; break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; memcpy(S.stream, b->st, 8 * b->n); S.nstream = b->n; break; }
        case OP_BIND_VBUF: { const p_bind_vbuf *b = a; memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
        case OP_BIND_UBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16) S.ubo[b->stage][b->index] = *b; break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16) S.ssbo[b->stage][b->index] = *b; break; }
        case OP_BIND_TEX: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first + b->n <= NTEX) memcpy(&S.tex[b->stage][b->first], b->h, 8 * b->n); break; }
        case OP_BIND_SAMP: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first + b->n <= 128) memcpy(&S.samp[b->stage][b->first], b->h, 8 * b->n); break; }
        case OP_BIND_IMG: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) memcpy(&S.img[b->stage][b->first], b->h, 8 * b->n); break; }
        case OP_STENCIL: { const p_stencil *b = a; for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; } break; }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)a)->on; break;
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)a)->on; break;
        case OP_TF_BIND: { const p_tf_bind *b = a; if (b->index >= 0 && b->index < 4) S.tf[b->index] = *b; break; }
        case OP_TF_STATE: { const p_tf_state *b = a;   // Begin writes from the start of each buffer; Pause / Resume keep the offsets (as nvn_mtl.m: the host keeps them, the game's control buffer is not used)
            if (b->what == 0) { S.tf_on = 1; memset(S.tf_off, 0, sizeof S.tf_off); } else S.tf_on = b->what == 3; break; }
        default: return 0;
    }
    return 1;
}
static uint64_t g_stat_draws, g_stat_passes, g_stat_rt_same;
#include "mali_white_vk.inc"   // the Mali white-screen ladder and its detector (counts draws by g_stat_draws)
// The same render targets set again while their pass is open: the pass goes on (on a tiler every pass end stores the attachments and every begin loads them back;
// Legends: Arceus sets render targets about 120 times a frame). Views compared field by field where set
static int rt_same(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != b->n || a->depth != b->depth || a->has_dview != b->has_dview || (a->has_dview && memcmp(&a->dview, &b->dview, sizeof a->dview))) return 0;
    for (int i = 0; i < a->n && i < 8; i++) if (a->color[i] != b->color[i] || a->has_cview[i] != b->has_cview[i] || (a->has_cview[i] && memcmp(&a->cview[i], &b->cview[i], sizeof a->cview[i]))) return 0;
    return 1;
}
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
// DM_VK_SPLIT=<draws> (default 150): a submission goes to the GPU in parts of that many draws, while the rest is still being replayed. A game that submits a frame and waits for it
//   (Scarlet: 43 ms a frame waiting on the submission it had just made) otherwise waits for the whole replay and then the whole GPU frame, one after the other; on the
//   console the GPU starts at once. The cut waits for the next change of render target, so no pass is split (cutting inside a pass, then opening it again, made
//   Sword flash white)
// A frame held whole (DM_VK_FRAME_HOLD) is not split (2026-11-24, fable-cpu fix 4): its parts would not reach the GPU before the present anyway, and from 5 parts on
//   (NSUB - 1, above ~750 draws at 150) the hold was flushed early, so in the heaviest scenes the frame was never whole (store decisions, passes run as one) and each part
//   cost a command pool reset, a command buffer and a queue submission. Profile vk_split_held whole (Sword / Shield; Scarlet / Violet split them as before, they wait on
//   the submission they just made); DM_VK_SPLIT_HELD=1 / 0 overrides. Not held: every DM_GAME_VK_SPLIT draws (profile vk_split: 150, Sword / Shield 600; DM_VK_SPLIT=<n>)
#ifndef DM_GAME_VK_SPLIT
#define DM_GAME_VK_SPLIT 150   // (a game_profile.h from before the profile keys)
#endif
#ifndef DM_GAME_VK_SPLIT_HELD_WHOLE
#define DM_GAME_VK_SPLIT_HELD_WHOLE 0
#endif
static int g_split_n, g_split_due;
static int split_every(void) {   // draws per part; 0: never split
    static int every = -1;
    if (every < 0) { const char *e = getenv("DM_VK_SPLIT"), *h = getenv("DM_VK_SPLIT_HELD"); every = e ? atoi(e) : DM_GAME_VK_SPLIT; if (every < 0) every = 0;   // (Scarlet outdoors 70 to 33 ms a frame; DM_VK_SPLIT=0 off)
        int held = hold_on() && (h && *h ? atoi(h) == 0 : DM_GAME_VK_SPLIT_HELD_WHOLE);
        if (held && every) VLOG("submissions: a frame held whole is not split (DM_VK_SPLIT_HELD=1 splits it every %d draws)", every);
        else if (every) VLOG("submissions go to the GPU every %d draws (DM_VK_SPLIT)", every);
        if (held) every = 0; }
    return every;
}
static void sub_split(void) {
    int every = split_every();
    if (!every || ++g_split_n < every) return;
    g_split_due = 1;   // cut at the next change of render target, between passes: a pass cut and opened again flashed white in Sword
}
static void sub_split_at_target(void) { if (g_split_due) { g_split_due = 0; g_split_n = 0; mtl_sub_commit(); mtl_sub_begin(); } }
void mtl_exec(const uint8_t *p) {
    if (!g_sub) { RED_ONCE(52, "commands replayed outside a submission"); return; }
    for (int guard = 0; guard < (1 << 26); guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        uint64_t vt0 = g_vprof > 0 && h.op != OP_CALL && h.op != OP_DRAW && h.op != OP_DRAW_IDX ? mono_ns() : 0;
        switch (h.op) {
            case OP_END: return;
            case OP_BEGIN: break;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); mtl_exec(t); break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); post((post_op){1, val, m, v, 0, 0}); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); uint32_t q = 0;
                if (g_rtq && g_sub && g_sub->nrq < RQ_END) {   // its GPU timestamp (report_ts): now outside a pass, else when the open pass ends
                    q = ++g_sub->nrq;
                    if (P.open) g_sub->rq_pend++; else vkCmdWriteTimestamp(CB(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_rtq, (uint32_t)(g_sub - g_subs) * RQ_PER + q - 1);
                }
                post((post_op){2, NULL, 0, 0, ad, (uint32_t)t, q}); break; }
            case OP_RESET_COUNTER: case OP_RENDER_COND: break;
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = g_fence + 1; s->fenced = 2; break; }   // (2: fenced inside a command list, DM_VK_PROF tells them apart)
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
            case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); sub_copy(s, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); sub_update(a + 16, d, n); break; }
            case OP_SET_RT: g_stat_passes++; tf_end(); if (P.open && R.has_rt && !getenv_nomerge() && rt_same(&R.rt, (const p_set_rt *)a)) { g_stat_rt_same++; break; }
                if (P.open && pass_defer_on() && !getenv_nomerge()) {   // (deferred: rt_resolve at the next draw or clear)
                    int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                    if (!pend) { R.rt = *(const p_set_rt *)a; g_rt_stale = 1; break; } }
                g_rt_stale = 0;
                if (P.open) { g_brk[0]++; g_pass_last_by_rt = 1; }
                sub_split_at_target();
                pass_flush(); R.rt = *(const p_set_rt *)a; R.has_rt = 1; break;
            case OP_CLEAR_COLOR: { tf_end();
                rt_resolve();
                const p_clear_color *c = (const p_clear_color *)a;
                if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(53, "colour clear target %d invalid", c->index); break; }
                VkClearColorValue cv; memcpy(cv.float32, c->rgba, 16);
                { vtex *cx = tex_res(R.rt.color[c->index]); VkFormat cf = cx ? cx->fmt : VK_FORMAT_UNDEFINED;
                  if (R.rt.has_cview[c->index] && R.rt.cview[c->index].set_format) { uint32_t vf = 0; uint8_t sw[4]; if (nvn_fmt_to_vk(R.rt.cview[c->index].format, &vf, sw) && vf) cf = (VkFormat)vf; }
                  cv = clear_for(cf, c->rgba);
                  if (b10_swapped_tex(cx) && b10_view_fmt(cx, cf, B10V_ATTACH, NULL) == cx->fmt) { cv = b10_clear(c->rgba); b10_note("a render target clear: the colour as the packed format stores it, alpha 1", cx, 0); } }
                if (P.open || c->mask != 0xf || !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) {
                    if (c->mask != 0xf) RED_ONCE(54, "colour clear with a partial channel mask: all channels cleared");
                    VkClearValue v = { .color = cv }; clear_in_pass(c->index, &v, VK_IMAGE_ASPECT_COLOR_BIT); break; }
                R.clear_c[c->index] = 1; R.cc[c->index] = cv; break;
            }
            case OP_CLEAR_DS: { tf_end();
                rt_resolve();
                const p_clear_ds *c = (const p_clear_ds *)a;
                if (P.open || (R.rt.depth && !scissor_full(R.rt.depth, &R.rt.dview, R.rt.has_dview))) {
                    VkClearValue v = { .depthStencil = { c->depth, (uint32_t)c->stencil } };
                    VkImageAspectFlags asp = (c->depth_mask && P.hd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask && P.hs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
                    if (!P.open) pass_begin();
                    pass_writes(c->depth_mask && P.hd, c->stencil_mask && P.hs);
                    asp = (c->depth_mask && P.hd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask && P.hs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
                    if (asp) clear_in_pass(-1, &v, asp); break; }
                if (c->depth_mask) { R.clear_d = 1; R.cd = c->depth; }
                if (c->stencil_mask) { R.clear_s = 1; R.cs = (uint32_t)c->stencil; }
                break;
            }
            case OP_DRAW: draw((const p_draw *)a, NULL); g_stat_draws++; fgr_draw(); hp_draw(); sub_split(); break;
            case OP_DRAW_IDX: draw(NULL, (const p_draw_idx *)a); g_stat_draws++; fgr_draw(); hp_draw(); sub_split(); break;
            case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a, 0); break;
            case OP_COPY_T2B: copy_b2t((const p_copy_b2t *)a, 1); break;
            case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
            case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
            case OP_DISPATCH: dispatch((const p_dispatch *)a); fgr_dispatch(); break;
            case OP_BARRIER: { int32_t bits; memcpy(&bits, a, 4); if (bits) { g_game_bar = 1; g_game_bits |= bits; } break; }   // ordered at the next draw pass (pass_begin)
            case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: state_op(h.op, a); break;
        }
        if (vt0) { uint64_t d = mono_ns() - vt0; int c = h.op == OP_SET_RT ? 0 : h.op == OP_CLEAR_COLOR || h.op == OP_CLEAR_DS || h.op == OP_CLEAR_TEX ? 1 : h.op == OP_COPY_B2T || h.op == OP_COPY_T2B || h.op == OP_COPY_T2T || h.op == OP_COPY_B2B || h.op == OP_UPDATE_UB || h.op == OP_CLEAR_BUF ? 2 : h.op == OP_DISPATCH ? 3 : 4; g_vp_o[c] += d; }
        p += h.size;
    }
    RED_ONCE(55, "command list did not end");
}

// ================= CPU texture reads =================
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) return;
    if (nvn_fmt_vk_decode(t->d.format)) { RED_ONCE(99, "CPU read of a BC texture decoded on upload (format %#x): not available", t->d.format); return; }
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh; int nz = r->depth ? r->depth : 1;
    vtex *x = tex_res(t); if (!x || !x->img) return;
    sk_read(x);
    int b10 = b10_swapped_tex(x) && bpb == 4; size_t xb = b10 ? wb * 2 : wb;   // (R11G11B10F's stand-in, DM_VK_B10_AS_RGBA16: read as RGBA16F, packed below)
    VkDeviceSize n = xb * hb * nz; vmalloc_t a; void *m; VkBuffer b = buf_make_c(n ? n : 16, &a, &m, VA_TEMP);
    VkCommandBuffer cb = once_begin();
    VkBufferImageCopy rg = { .bufferOffset = 0, .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, level, x->is3d ? 0 : layer0 + (uint32_t)r->zoffset, x->is3d ? 1 : (uint32_t)nz },
        .imageOffset = { r->xoffset, r->yoffset, x->is3d ? r->zoffset : 0 }, .imageExtent = { (uint32_t)r->width, (uint32_t)r->height, x->is3d ? (uint32_t)nz : 1 } };
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &rg);
    once_end(cb);
    if (b10) { for (int z = 0; z < nz; z++) b10_rows_pack((uint8_t *)m + (size_t)z * hb * xb, xb, (uint32_t)r->width, hb, dst + z * img, row);
               b10_note("the CPU read the texture: packed", x, 0); }
    else for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, (uint8_t *)m + (z * hb + y) * wb, wb);
    vkDestroyBuffer(g_dev, b, NULL); vk_free(a);
}

// ================= Present: no window yet (the Android surface comes with the app); paced to the game's period, screenshots on request =================
static _Atomic uint64_t g_present_n; static uint64_t g_last_present;
static uint32_t crc_tab[256];
static uint32_t crc32_(uint32_t c, const uint8_t *p, size_t n) { if (!crc_tab[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t r = i; for (int k = 0; k < 8; k++) r = r & 1 ? 0xedb88320u ^ (r >> 1) : r >> 1; crc_tab[i] = r; }
    c = ~c; while (n--) c = crc_tab[(c ^ *p++) & 0xff] ^ (c >> 8); return ~c; }
static void png_chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    uint8_t h[8] = { (uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n, (uint8_t)type[0], (uint8_t)type[1], (uint8_t)type[2], (uint8_t)type[3] };
    fwrite(h, 1, 8, f); if (n) fwrite(d, 1, n, f); uint32_t c = crc32_(crc32_(0, h + 4, 4), d, n); uint8_t cc[4] = { (uint8_t)(c >> 24), (uint8_t)(c >> 16), (uint8_t)(c >> 8), (uint8_t)c }; fwrite(cc, 1, 4, f);
}
static void png_write(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h, int flip) {   // uncompressed (stored deflate blocks): no zlib needed
    size_t raw = (size_t)h * (1 + 4 * w); uint8_t *r = malloc(raw);
    for (uint32_t y = 0; y < h; y++) { r[y * (1 + 4 * w)] = 0; memcpy(r + y * (1 + 4 * w) + 1, rgba + (size_t)(flip ? h - 1 - y : y) * 4 * w, 4 * w); }
    size_t nb = (raw + 65534) / 65535, zn = 2 + raw + 5 * nb + 4; uint8_t *z = malloc(zn), *q = z; *q++ = 0x78; *q++ = 0x01;
    uint32_t a1 = 1, a2 = 0; for (size_t i = 0; i < raw; i++) { a1 = (a1 + r[i]) % 65521; a2 = (a2 + a1) % 65521; }
    for (size_t o = 0; o < raw; o += 65535) { size_t l = raw - o < 65535 ? raw - o : 65535; *q++ = o + l == raw; *q++ = (uint8_t)l; *q++ = (uint8_t)(l >> 8); *q++ = (uint8_t)~l; *q++ = (uint8_t)(~l >> 8); memcpy(q, r + o, l); q += l; }
    uint32_t ad = a2 << 16 | a1; *q++ = (uint8_t)(ad >> 24); *q++ = (uint8_t)(ad >> 16); *q++ = (uint8_t)(ad >> 8); *q++ = (uint8_t)ad;
    FILE *f = fopen(path, "wb"); if (!f) { VLOG("[RED] cannot write %s", path); free(r); free(z); return; }
    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 13, 10, 26, 10 }; fwrite(sig, 1, 8, f);
    uint8_t ih[13] = { (uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 6, 0, 0, 0 };
    png_chunk(f, "IHDR", ih, 13); png_chunk(f, "IDAT", z, (uint32_t)(q - z)); png_chunk(f, "IEND", NULL, 0); fclose(f); free(r); free(z);
}
static void shot_ex(vtex *x, const int crop[4], const char *path, uint32_t lv, uint32_t sl);
static void shot(vtex *x, const int crop[4], const char *path) { shot_ex(x, crop, path, 0, 0); }
static void shot_level(vtex *x, uint32_t lv, uint32_t sl, const char *path) { shot_ex(x, NULL, path, lv, sl); }
static void shot_ex(vtex *x, const int crop[4], const char *path, uint32_t lv, uint32_t sl) {
    sk_read(x);
    uint32_t lw = x->w >> lv ? x->w >> lv : 1, lh = x->h >> lv ? x->h >> lv : 1;
    uint32_t cx = crop ? (uint32_t)crop[0] : 0, cy = crop ? (uint32_t)crop[1] : 0, w = crop && crop[2] ? (uint32_t)crop[2] : lw, h = crop && crop[3] ? (uint32_t)crop[3] : lh;
    mtl_finish();
    if (!(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || x->samples > 1 || fmt_int_class(x->fmt)) { VLOG("[texdump] %s: depth / multisample image not saved", path); return; }
    // a blit into an RGBA8 image converts any colour format, then a copy into host memory
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
    VkImage im; VKCHECK(vkCreateImage(g_dev, &ci, NULL, &im)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, im, &rq); vmalloc_t ia = vk_alloc_c(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VA_TEMP); vkBindImageMemory(g_dev, im, ia.b->mem, ia.off);
    vmalloc_t ba; void *m; VkBuffer b = buf_make_c((VkDeviceSize)w * h * 4, &ba, &m, VA_TEMP);
    VkCommandBuffer cb = once_begin();
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkImageBlit bl = { { VK_IMAGE_ASPECT_COLOR_BIT, lv, sl, 1 }, {{(int32_t)cx, (int32_t)cy, 0}, {(int32_t)(cx + w), (int32_t)(cy + h), 1}}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}} };
    // 8-bit RGBA / BGRA (sRGB or not): the stored bytes as they are, which is what a display shows (a blit would decode sRGB into linear values and darken the picture);
    // other formats are converted by a blit
    int raw8 = x->fmt == VK_FORMAT_R8G8B8A8_UNORM || x->fmt == VK_FORMAT_R8G8B8A8_SRGB, bgra8 = x->fmt == VK_FORMAT_B8G8R8A8_UNORM || x->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    if (raw8 || bgra8) { VkImageCopy ic = { { VK_IMAGE_ASPECT_COLOR_BIT, lv, sl, 1 }, {(int32_t)cx, (int32_t)cy, 0}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {0, 0, 0}, { w, h, 1 } };
                         vkCmdCopyImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, im, VK_IMAGE_LAYOUT_GENERAL, 1, &ic); }
    else vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, im, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
    barrier_all(cb);
    VkBufferImageCopy rg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } }; vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, b, 1, &rg);
    once_end(cb);
    static int keep_a = -1; if (keep_a < 0) keep_a = getenv("DM_NVN_TEXDUMP_ALPHA") != NULL;   // DM_NVN_TEXDUMP_ALPHA: the real alpha channel (default opaque, to look at)
    for (uint32_t i = 0; i < w * h; i++) { uint8_t *q = (uint8_t *)m + 4 * i; if (bgra8) { uint8_t t = q[0]; q[0] = q[2]; q[2] = t; } if (!keep_a) q[3] = 255; }
    png_write(path, m, w, h, atomic_load(&g_origin_ll));
    vkDestroyImage(g_dev, im, NULL); vk_free(ia); vkDestroyBuffer(g_dev, b, NULL); vk_free(ba);
    VLOG("screenshot %s (%ux%u)", path, w, h);
}
// ---------- Window presentation (Android: the app's ANativeWindow, front_metal_layer()) ----------
// A FIFO swapchain the size of the window; each present blits the game's window texture into it, scaled to fit with black bars, flipped in GL (lower-left) mode,
// through a swapchain format of the same sRGB-ness as the game's texture so its bytes reach the screen unchanged
void *front_metal_layer(void);
// SC_MAX: room for the swapchain's images. Arm's driver makes more than asked for (a Mali-G57 on r32p1 made 14 for 8): with room for 8 only, an index from
// vkAcquireNextImageKHR past 8 read W.view[] as an image, and Arm's barrier handling closed the game a few frames in (SIGSEGV at 0x90 in libGLES_mali.so, #158 #163)
#define SC_MAX 32
static struct { void *win; VkSurfaceKHR surf; VkSwapchainKHR sc; VkImage img[SC_MAX]; VkImageView view[SC_MAX]; uint32_t n; VkFormat fmt; VkExtent2D ext; int srgb;
                VkExtent2D lext; int rot;   // lext: the window as the app sees it (landscape); rot: quarter turns the swapchain is pre-rotated by (ext is then the panel's own size)
                VkSemaphore acq[3], done[3], idone[SC_MAX]; VkFence fence[3]; VkCommandPool pool; VkCommandBuffer cb[3]; uint32_t slot; } W;
// Making and dropping the window surface runs on a thread of its own: present_window runs on a game thread, whose stack is the game's (outside its pthread stack), and
// releasing an Android surface calls into Java (BLASTBufferQueue's corner-radii callback), which attaches the calling thread; ART aborts the process when that thread's
// stack is not its pthread stack (Thread::InitStack: "FindStackTop() > GetStackEnd()"). That was the crash on leaving the app (recents, home) and coming back
static void swap_destroy_now(void);
static int swap_ensure_now(void *win, int srgb);
struct swap_job { int make; void *win; int srgb, r; };
static void *swap_job_main(void *p) {
    struct swap_job *j = p;
    if (j->make) j->r = swap_ensure_now(j->win, j->srgb); else swap_destroy_now();
    return NULL;
}
static int swap_job_run(int make, void *win, int srgb) {
    struct swap_job j = { make, win, srgb, 0 }; pthread_t t;
    if (pthread_create(&t, NULL, swap_job_main, &j)) { RED_ONCE(59, "window: no thread for the swapchain"); return 0; }
    pthread_join(t, NULL);
    return j.r;
}
static void swap_destroy(void) { if (W.sc || W.surf) swap_job_run(0, NULL, 0); W.win = NULL; }
static int swap_ensure(void *win, int srgb) {
#ifdef __ANDROID__
    // the window changed size (a foldable opened or closed, a rotation): drivers need not report it as out of date, so a new swapchain of the new size
    if (W.sc && W.win == win) { int32_t nw = ANativeWindow_getWidth((ANativeWindow *)win), nh = ANativeWindow_getHeight((ANativeWindow *)win);
        if (nw > 0 && nh > 0 && ((uint32_t)nw != W.ext.width || (uint32_t)nh != W.ext.height)) swap_destroy(); }
#endif
    if (W.sc && W.win == win && W.srgb == srgb) return 1;
    return swap_job_run(1, win, srgb);
}
// The semaphore a present of swapchain image i waits on, one per image (made the first time): one per frame slot (3) was signalled again for another image while the
//   driver could still hold it for an image presented earlier and not acquired again (validation: VUID-vkQueueSubmit-pSignalSemaphores-00067; the swapchain has more
//   images than slots). The image is acquired again before its semaphore is signalled again, which is what makes it safe
static VkSemaphore done_img(uint32_t i) {
    if (i >= SC_MAX) i = SC_MAX - 1;
    if (!W.idone[i]) { VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; vkCreateSemaphore(g_dev, &sci, NULL, &W.idone[i]); }
    return W.idone[i];
}
static void swap_destroy_now(void) {
    if (W.sc) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu);   // (the queue is externally synchronized: other threads submit to it)
                for (uint32_t i = 0; i < W.n; i++) if (W.view[i]) { scfb_forget(W.view[i]); vkDestroyImageView(g_dev, W.view[i], NULL); W.view[i] = VK_NULL_HANDLE; }
                vkDestroySwapchainKHR(g_dev, W.sc, NULL); W.sc = VK_NULL_HANDLE; }
    if (W.surf) { vkDestroySurfaceKHR(g_inst, W.surf, NULL); W.surf = VK_NULL_HANDLE; }
    W.win = NULL;
}
static int scale_setup(VkFormat fmt, int which);
static int framegen_mode(void);
// ---------- Software present: when the window's swapchain cannot be made ----------
// With a custom driver loaded through libadrenotools, the swapchain is made by Android's Vulkan loader from the window's own buffers (gralloc), which the driver must
// import. Mali kernels that refuse that import (a Mali-G52 on kernel 4.14 and a Mali-G615 with Mesa PanVK, #215 #227) failed every swapchain with
// VK_ERROR_INVALID_EXTERNAL_HANDLE: the game ran, the screen stayed black. Then the swapchain is ours: images in the GPU's memory, each presented frame copied to a
// buffer the CPU reads and written into the window with ANativeWindow_lock, which imports nothing. It costs a copy of the frame a present (8 MB at 1080p).
// DM_VK_SOFT_PRESENT=0: never (the black screen as before), =1: from the start, on any GPU (for testing)
#ifdef __ANDROID__
enum { SW_N = 3 };
static struct { int on; ANativeWindow *win; VkImage img[SW_N]; vmalloc_t mem[SW_N]; VkBuffer buf[SW_N]; vmalloc_t bmem[SW_N]; uint8_t *map[SW_N]; int coherent[SW_N];
                VkCommandPool pool; VkCommandBuffer cb[SW_N]; VkFence fence[SW_N]; uint32_t w, h, next; int bgra, pending; uint64_t presents; } SW = { .pending = -1 };
static int sw_mode(void) { static int v = -2; if (v == -2) { const char *e = getenv("DM_VK_SOFT_PRESENT"); v = e && *e ? atoi(e) : -1; } return v; }   // -1 auto, 0 never, 1 always
static void sw_free(void) {
    SW.pending = -1;
    for (int i = 0; i < SW_N; i++) {
        if (SW.img[i]) { vkDestroyImage(g_dev, SW.img[i], NULL); vk_free(SW.mem[i]); SW.img[i] = VK_NULL_HANDLE; }
        if (SW.buf[i]) { vkDestroyBuffer(g_dev, SW.buf[i], NULL); vk_free(SW.bmem[i]); SW.buf[i] = VK_NULL_HANDLE; SW.map[i] = NULL; }
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL sw_create(VkDevice d, const VkSwapchainCreateInfoKHR *ci, const VkAllocationCallbacks *al, VkSwapchainKHR *out) {
    (void)d; (void)al; sw_free();
    SW.w = ci->imageExtent.width; SW.h = ci->imageExtent.height; SW.next = 0;
    SW.bgra = ci->imageFormat == VK_FORMAT_B8G8R8A8_UNORM || ci->imageFormat == VK_FORMAT_B8G8R8A8_SRGB;
    if (ANativeWindow_setBuffersGeometry(SW.win, (int32_t)SW.w, (int32_t)SW.h, WINDOW_FORMAT_RGBA_8888)) { VLOG("[RED] window: software present: the window refused %ux%u RGBA", SW.w, SW.h); return VK_ERROR_INITIALIZATION_FAILED; }
    if (!SW.pool) {
        VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf };
        if (vkCreateCommandPool(g_dev, &pci, NULL, &SW.pool)) return VK_ERROR_INITIALIZATION_FAILED;
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = SW.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = SW_N };
        vkAllocateCommandBuffers(g_dev, &cai, SW.cb);
        for (int i = 0; i < SW_N; i++) { VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT }; vkCreateFence(g_dev, &fci, NULL, &SW.fence[i]); }
    }
    for (int i = 0; i < SW_N; i++) {
        VkImageCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = ci->imageFormat, .extent = { SW.w, SW.h, 1 }, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = ci->imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
        if (vkCreateImage(g_dev, &ii, NULL, &SW.img[i])) { sw_free(); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, SW.img[i], &rq); SW.mem[i] = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (!SW.mem[i].b || vkBindImageMemory(g_dev, SW.img[i], SW.mem[i].b->mem, SW.mem[i].off)) { vkDestroyImage(g_dev, SW.img[i], NULL); SW.img[i] = VK_NULL_HANDLE; sw_free(); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
        VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)SW.w * SW.h * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
        if (vkCreateBuffer(g_dev, &bi, NULL, &SW.buf[i])) { sw_free(); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
        vkGetBufferMemoryRequirements(g_dev, SW.buf[i], &rq);
        SW.bmem[i] = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1); SW.coherent[i] = 0;   // (cached: the CPU reads it all)
        if (!SW.bmem[i].b) { SW.bmem[i] = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1); SW.coherent[i] = 1; }
        if (!SW.bmem[i].b || !SW.bmem[i].b->map || vkBindBufferMemory(g_dev, SW.buf[i], SW.bmem[i].b->mem, SW.bmem[i].off)) { vkDestroyBuffer(g_dev, SW.buf[i], NULL); SW.buf[i] = VK_NULL_HANDLE; sw_free(); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
        SW.map[i] = (uint8_t *)SW.bmem[i].b->map + SW.bmem[i].off;
    }
    *out = (VkSwapchainKHR)(uintptr_t)0x50f75c;   // (ours: no driver sees it)
    VLOG("window: software present: %u images %ux%u of ours, each frame copied into the window by the CPU (%s readback memory)", SW_N, SW.w, SW.h, SW.coherent[0] ? "uncached" : "cached");
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sw_destroy(VkDevice d, VkSwapchainKHR sc, const VkAllocationCallbacks *al) { (void)d; (void)sc; (void)al; sw_free(); }
static VKAPI_ATTR VkResult VKAPI_CALL sw_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *imgs) {
    (void)d; (void)sc; if (!imgs) { *n = SW_N; return VK_SUCCESS; }
    uint32_t k = *n < SW_N ? *n : SW_N; for (uint32_t i = 0; i < k; i++) imgs[i] = SW.img[i]; *n = k; return k < SW_N ? VK_INCOMPLETE : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sw_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, VkFence fence, uint32_t *idx) {
    (void)d; (void)sc; (void)timeout;
    uint32_t i = SW.next; SW.next = (SW.next + 1) % SW_N; *idx = i;
    vkWaitForFences(g_dev, 1, &SW.fence[i], VK_TRUE, 2000000000ull);   // (its last copy is read already: sw_present waited for it)
    if (sem || fence) {   // the image is ready at once: signal what the caller waits on
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .signalSemaphoreCount = sem ? 1u : 0u, .pSignalSemaphores = &sem };
        pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, fence); pthread_mutex_unlock(&g_qmu);
        if (r) return r;
    }
    return VK_SUCCESS;
}
// Image i's copy (fence signalled) into the window
static VkResult sw_post(uint32_t i) {
    if (vkWaitForFences(g_dev, 1, &SW.fence[i], VK_TRUE, 1000000000ull)) return VK_TIMEOUT;
    if (!SW.coherent[i]) { VkMappedMemoryRange mr = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = SW.bmem[i].b->mem, .offset = 0, .size = VK_WHOLE_SIZE };
                           if (vkInvalidateMappedMemoryRanges) vkInvalidateMappedMemoryRanges(g_dev, 1, &mr); }
    ANativeWindow_Buffer wb;
    if (ANativeWindow_lock(SW.win, &wb, NULL)) { RED_ONCE(124, "window: software present: the window cannot be locked for the CPU"); return VK_ERROR_SURFACE_LOST_KHR; }
    uint32_t h = (uint32_t)wb.height < SW.h ? (uint32_t)wb.height : SW.h, w = (uint32_t)wb.width < SW.w ? (uint32_t)wb.width : SW.w;
    if (wb.format == WINDOW_FORMAT_RGBA_8888 || wb.format == WINDOW_FORMAT_RGBX_8888) {
        for (uint32_t y = 0; y < h; y++) {
            uint8_t *dst = (uint8_t *)wb.bits + (size_t)y * (size_t)wb.stride * 4; const uint8_t *src = SW.map[i] + (size_t)y * SW.w * 4;
            if (!SW.bgra) memcpy(dst, src, (size_t)w * 4);
            else for (uint32_t x = 0; x < w; x++) { dst[4 * x] = src[4 * x + 2]; dst[4 * x + 1] = src[4 * x + 1]; dst[4 * x + 2] = src[4 * x]; dst[4 * x + 3] = src[4 * x + 3]; }
        }
    } else RED_ONCE(124, "window: software present: the window's format is %d, not RGBA: frames not copied", wb.format);
    ANativeWindow_unlockAndPost(SW.win);
    if (!SW.presents++) VLOG("window: software present: first frame on screen");
    return VK_SUCCESS;
}
// (called with g_qmu held, as every vkQueuePresentKHR here is). The copy of this frame is started and the previous frame, whose copy is done, is posted:
//   the GPU and the CPU work side by side, one frame later on screen
static VKAPI_ATTR VkResult VKAPI_CALL sw_present(VkQueue q, const VkPresentInfoKHR *pi) {
    VkResult res = VK_SUCCESS;
    for (uint32_t s = 0; s < pi->swapchainCount; s++) {
        uint32_t i = pi->pImageIndices[s]; if (i >= SW_N || !SW.img[i]) { res = VK_ERROR_OUT_OF_DATE_KHR; continue; }
        VkCommandBuffer cb = SW.cb[i]; vkResetCommandBuffer(cb, 0);
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(cb, &bi);
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = SW.img[i], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        VkBufferImageCopy bc = { 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { SW.w, SW.h, 1 } };
        vkCmdCopyImageToBuffer(cb, SW.img[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, SW.buf[i], 1, &bc);
        ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; ib.dstAccessMask = 0; ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkBufferMemoryBarrier bb = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = SW.buf[i], .size = VK_WHOLE_SIZE };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 1, &bb, 1, &ib);
        vkEndCommandBuffer(cb);
        VkPipelineStageFlags ws[8]; for (int k = 0; k < 8; k++) ws[k] = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = s == 0 ? (pi->waitSemaphoreCount < 8 ? pi->waitSemaphoreCount : 8) : 0, .pWaitSemaphores = pi->pWaitSemaphores,
            .pWaitDstStageMask = ws, .commandBufferCount = 1, .pCommandBuffers = &cb };
        vkResetFences(g_dev, 1, &SW.fence[i]);
        VkResult r = vkQueueSubmit(q, 1, &si, SW.fence[i]);
        if (r == VK_SUCCESS) { if (SW.pending >= 0 && (uint32_t)SW.pending != i) { VkResult rp = sw_post((uint32_t)SW.pending); if (rp) res = rp; } SW.pending = (int)i; }
        else res = r;
        if (pi->pResults) pi->pResults[s] = res;
    }
    return res;
}
// The software swapchain's size: at most DM_VK_SOFT_PRESENT_H lines (default 720), the width in proportion; the compositor scales the window's buffer to the screen,
//   so the frame copied each present stays small (a 3120x1440 screen: 18 MB a frame, 14 fps on a Galaxy S25+)
static void sw_fit(VkSwapchainCreateInfoKHR *ci) {
    const char *e = getenv("DM_VK_SOFT_PRESENT_H"); uint32_t mh = e && atoi(e) >= 240 ? (uint32_t)atoi(e) : 720;
    uint32_t w = ci->imageExtent.width, h = ci->imageExtent.height;
    if (h > mh) { w = (uint32_t)((uint64_t)w * mh / h) & ~1u; h = mh; }
    ci->imageExtent = (VkExtent2D){ w, h }; W.ext = W.lext = ci->imageExtent; W.rot = 0; ci->preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
}
// The swapchain functions become ours (the surface goes first: a surface holds the window for the GPU, and the CPU cannot lock it then)
static void sw_install(void *win) {
    if (W.surf) { vkDestroySurfaceKHR(g_inst, W.surf, NULL); W.surf = VK_NULL_HANDLE; }
    SW.win = (ANativeWindow *)win; SW.on = 1;
    vkCreateSwapchainKHR = sw_create; vkDestroySwapchainKHR = sw_destroy; vkGetSwapchainImagesKHR = sw_images; vkAcquireNextImageKHR = sw_acquire; vkQueuePresentKHR = sw_present;
    p_refresh = NULL; p_past = NULL; g_dtiming = 0;   // (display timing would name our swapchain to the driver)
}
#endif
static int swap_ensure_now(void *win, int srgb) {
    swap_destroy_now();
#ifdef __ANDROID__
    VkAndroidSurfaceCreateInfoKHR si = { .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .window = (ANativeWindow *)win };
    if (vkCreateAndroidSurfaceKHR(g_inst, &si, NULL, &W.surf)) { RED_ONCE(56, "cannot create the window surface"); return 0; }
#else
    (void)win; return 0;
#endif
    VkBool32 ok = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(g_pd, g_qf, W.surf, &ok); if (!ok) { RED_ONCE(57, "the queue cannot present to the window"); return 0; }
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, W.surf, &caps);
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, W.surf, &nf, NULL); VkSurfaceFormatKHR fs[32]; if (nf > 32) nf = 32; vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, W.surf, &nf, fs);
    VkFormat want = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM; W.fmt = fs[0].format;
    for (uint32_t i = 0; i < nf; i++) if (fs[i].format == want) W.fmt = want;
    W.ext = caps.currentExtent;
#ifdef __ANDROID__
    if (W.ext.width == 0xffffffffu || 1) { W.ext.width = (uint32_t)ANativeWindow_getWidth((ANativeWindow *)win); W.ext.height = (uint32_t)ANativeWindow_getHeight((ANativeWindow *)win); }
#endif
    // MAILBOX when the window has it: the game paces its own frames (mtl_present), and with FIFO a slow compositor (a screen recording, the notification shade) held
    // presentation, which shares the queue lock with the game's submissions, so the whole game waited on the compositor (LAS Recomp's Android port measured 4–21 fps with
    // the GPU mostly idle under FIFO, a steady rate with MAILBOX). DM_VK_PRESENT=fifo|mailbox|immediate to choose
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    { const char *e = getenv("DM_VK_PRESENT"); VkPresentModeKHR want = !e || !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
      uint32_t npm = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, NULL); VkPresentModeKHR pms[16]; if (npm > 16) npm = 16; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, pms);
      for (uint32_t i = 0; i < npm; i++) if (pms[i] == want) pm = want; }
    uint32_t n = caps.minImageCount + (framegen_mode() == 3 ? 4 : pm == VK_PRESENT_MODE_MAILBOX_KHR || framegen_mode() ? 2 : 1);   // (frame generation holds two images at once: the in-between picture and the real frame; Lossless Scaling's up to four, 4x) if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;
    // Pre-rotation (DM_VK_PRETRANSFORM=1, opt-in until checked on phones): the panel's own orientation is portrait and the game landscape, so the compositor turns
    // every frame; with the swapchain made in the panel's orientation (preTransform = currentTransform) the upscaler and the overlay draw the picture already turned
    // and the compositor has nothing to do. Only when the upscaler's pipeline exists: the blit cannot turn the picture
    if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;   // (a driver with a low maximum refused the swapchain outright: every frame failed, a black screen)
    W.rot = 0; W.lext = W.ext; VkSurfaceTransformFlagBitsKHR pre = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    { const char *e = getenv("DM_VK_PRETRANSFORM"); VkSurfaceTransformFlagBitsKHR t = caps.currentTransform;
      int q = t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? 1 : t == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? 2 : t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ? 3 : 0;
      if (e && atoi(e) && q && scale_setup(W.fmt, 0)) { W.rot = q; pre = t; if (q & 1) W.ext = (VkExtent2D){ W.lext.height, W.lext.width }; } }
    VkSwapchainCreateInfoKHR ci = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = W.surf, .minImageCount = n, .imageFormat = W.fmt, .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = W.ext, .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = pre,
        .compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, .presentMode = pm, .clipped = VK_TRUE };
    // what the surface allows, before asking: no more images than its maximum, a composite alpha mode it lists (one listing neither opaque nor inherit got an invalid one)
    if (caps.maxImageCount && ci.minImageCount > caps.maxImageCount) ci.minImageCount = caps.maxImageCount;
    if (!(caps.supportedCompositeAlpha & ci.compositeAlpha))
        for (uint32_t b = 1; b <= VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR; b <<= 1) if (caps.supportedCompositeAlpha & b) { ci.compositeAlpha = (VkCompositeAlphaFlagBitsKHR)b; break; }
#ifdef __ANDROID__
    if (SW.on || sw_mode() == 1) { sw_install(win); sw_fit(&ci); }   // (again for a new window: its surface was only for the size queries above)
#endif
    VkResult r = vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc);
    // A swapchain the driver refuses (older Mali drivers among them) left the game with no picture at all. Again with the plainest one every driver must make: the fewest
    //   images, FIFO, no pre-rotation; a window another surface still holds, or a lost surface, gets a new surface first. Each step logged
    if (r == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {
#ifdef __ANDROID__
        VLOG("[RED] window: swapchain creation failed (%d): with a new surface", r);
        vkDestroySurfaceKHR(g_inst, W.surf, NULL); W.surf = VK_NULL_HANDLE;
        VkAndroidSurfaceCreateInfoKHR si2 = { .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .window = (ANativeWindow *)win };
        if (vkCreateAndroidSurfaceKHR(g_inst, &si2, NULL, &W.surf)) { RED_ONCE(56, "cannot create the window surface"); return 0; }
        ci.surface = W.surf; r = vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc);
#endif
    }
    if (r) {
        VLOG("[RED] window: swapchain creation failed (%d, %u images, present mode %d): again with %u images, FIFO, no pre-rotation", r, ci.minImageCount, (int)ci.presentMode, caps.minImageCount);
        ci.minImageCount = caps.minImageCount ? caps.minImageCount : 2; ci.presentMode = pm = VK_PRESENT_MODE_FIFO_KHR;
        if (ci.preTransform != caps.currentTransform && (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)) ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        if (W.rot) { W.rot = 0; W.ext = W.lext; ci.imageExtent = W.ext; }
        r = vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc);
    }
#ifdef __ANDROID__
    if (r && !SW.on && sw_mode() != 0) {   // the window cannot get a swapchain from the driver: ours instead (software present)
        VLOG("[RED] window: swapchain creation failed (%d): the frames go to the window through the CPU instead (DM_VK_SOFT_PRESENT=0 turns that off)", r);
        sw_install(win); sw_fit(&ci); ci.surface = VK_NULL_HANDLE; r = vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc);
    }
#endif
    if (r) { RED_ONCE(58, "swapchain creation failed: %d", r); return 0; }
    { uint32_t made = 0; vkGetSwapchainImagesKHR(g_dev, W.sc, &made, NULL);   // (every image the driver made: it may make more than minImageCount asked for)
      W.n = made < SC_MAX ? made : SC_MAX; vkGetSwapchainImagesKHR(g_dev, W.sc, &W.n, W.img);
      if (made > SC_MAX) RED_ONCE(114, "the swapchain has %u images, room for %d: the others are skipped when handed out", made, SC_MAX);
      if (made != n) VLOG("window: the driver made %u swapchain images for %u asked for", made, n); }
    for (uint32_t i = 0; i < W.n; i++) { VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = W.img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = W.fmt,
                                           .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } }; vkCreateImageView(g_dev, &vci, NULL, &W.view[i]); }
    if (!W.pool) {
        VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf }; vkCreateCommandPool(g_dev, &pci, NULL, &W.pool);
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = W.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 3 }; vkAllocateCommandBuffers(g_dev, &cai, W.cb);
        for (int i = 0; i < 3; i++) { VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; vkCreateSemaphore(g_dev, &sci, NULL, &W.acq[i]); vkCreateSemaphore(g_dev, &sci, NULL, &W.done[i]);
            VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT }; vkCreateFence(g_dev, &fci, NULL, &W.fence[i]); }
    }
    W.win = win; W.srgb = srgb;
    VLOG("window: swapchain %ux%u, %u images, format %d, %s%s", W.ext.width, W.ext.height, W.n, W.fmt, pm == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : pm == VK_PRESENT_MODE_FIFO_KHR ? "fifo" : "immediate",
         W.rot ? ", pre-rotated" : "");
    g_refresh_ns = 0;
    if (g_dtiming && p_refresh) { VkRefreshCycleDurationGOOGLE rc = { 0 }; if (!p_refresh(g_dev, W.sc, &rc) && rc.refreshDuration) g_refresh_ns = rc.refreshDuration; }
    VLOG("frame pacing: %s", !g_dtiming ? "off (no VK_GOOGLE_display_timing)" : !g_refresh_ns ? "off (the display's refresh is unknown)"
                             : getenv("DM_VK_PACE") && !atoi(getenv("DM_VK_PACE")) ? "off (DM_VK_PACE=0; frame timing still measured)" : "on");
    if (g_refresh_ns) VLOG("display refresh %.3f ms (%.2f Hz)", g_refresh_ns / 1e6, 1e9 / (double)g_refresh_ns);
    return 1;
}
// ---------- Frame pacing ----------
static int pace_off(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PACE"); v = e && !atoi(e); } return v; }
// The display's refresh period for the game's vsync timer (hle_sdk16.c vsync_thread): its 60 Hz then follows the display's own clock; 0 when not pacing
uint64_t dm_display_refresh_ns(void) { return pace_off() ? 0 : g_refresh_ns; }
static struct { uint64_t last_log, n, uneven, pauses, sum, nsum, prev_actual; uint32_t id; } g_pace;
static uint64_t g_pace_asked;   // the time the last frame presented asked to be shown (0: none: no display timing, pacing off, a rate the display's own); mtl_present
// The asked times (vk_pace.h): the grid (g_vkp.next, shared with frame generation's pictures), the GPU's time of the last frames, the display's record of the
//   frames paced by it (g_vkp_plain: by presentID, so frame generation's pictures, paced their own way, stay out of it). Off by default since the S25+ runs of
//   2026-11-24: the display's record raised the lead to an interval and two refreshes (66.7 ms at 30 fps), and from there each late hand-over moved the grid and
//   lost a frame: 269-299 frames shown in 10 s (mean interval 33.5-37.3 ms) against 299-302 with the times as before (perf-nolead, v0.98), uneven frames no fewer.
//   DM_PACE_LEAD=1 turns it on; DM_VK_PACE_LEAD=<ms> (a fixed lead) keeps it off
#include "vk_pace.h"
static vk_pacer g_vkp; static uint8_t g_vkp_plain[64]; static uint64_t g_vkp_gf;
// One grid (perf-1124 G3'): the game hands frames over on 2 / 60 s while the display shows them 2 of its refreshes apart (33.37 ms on a 59.93 Hz panel). The
//   asked times now keep their grid, snapped to the display's vsyncs, within legacy's lead (vk_pace.h vkp_want_grid). DM_PACE_GRID (dm_pace_grid_mode):
//   unset 0 that; game 1: vkp_want_legacy, as before; display 2: that and the game's acquire grid on whole refreshes of the display too (hle_nvn.c
//   present_hold). The analysis asked for 2 by default; t_vk_pace showed it locks the hand-over's phase for good (a bad phase stays bad: 4.8 -> 8.9 % uneven
//   at GPU 22 ms), so it waits for a phone A/B
int dm_pace_grid_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_PACE_GRID"); v = e && !strcmp(e, "game") ? 1 : e && !strcmp(e, "display") ? 2 : 0;
    VLOG("frame pacing: %s (DM_PACE_GRID %s)", v == 1 ? "asked times clamped to a lead from now, as before" : v == 2 ? "asked times on the display's vsyncs, the game's frames on whole refreshes of the display"
         : "asked times on the display's vsyncs, the game's own grid", e ? e : "unset; game: as before, display: the game on the display's grid too"); } return v; }
static int pace_adapt(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_PACE_LEAD"), *l = getenv("DM_VK_PACE_LEAD"); v = e && strcmp(e, "0") && !(l && atol(l) > 0);
    VLOG("frame pacing: %s", v ? "the lead follows the GPU's time and the display's record (DM_PACE_LEAD=1)" : "one frame interval, as before 2026-11-24 (DM_PACE_LEAD=1: the lead follows the GPU's time)");
    // the hold (vk_pace.h): no grid move for a hitch or a load (a 1 s grace period), none chasing a GPU over the interval. DM_PACE_HOLD=0: every late frame moves it
    const char *h = getenv("DM_PACE_HOLD"); g_vkp.hold_off = h && !strcmp(h, "0");
    if (v) VLOG("frame pacing: %s", g_vkp.hold_off ? "every late frame moves the grid (DM_PACE_HOLD=0)" : "hitches and loads keep the grid for 1 s, a GPU over the interval gets a fixed lead (DM_PACE_HOLD=0: off)"); } return v; }
// The time this frame should reach the screen: one frame interval (1 / the frame rate, rounded to whole refreshes) after the last frame's. A frame that comes late shows at
//   the next refresh it can make; the target never runs more than one interval ahead of now (with mailbox a newer frame would replace one still waiting). With pacing off
//   (DM_VK_PACE=0, or a frame rate that is the display's own) the frame still gets an id and no time (0: as soon as it can), so its timing is measured the same way.
// Every 10 s the display's record of the frames goes to the log: intervals between shown frames, uneven = more than half a refresh off the target (a stutter you can
//   see), pauses = longer than 4 intervals (loading, a pipeline built in play: left out of the mean). The compositor shows every frame a fixed refresh or two after its
//   time, so lateness against the asked time says nothing; the intervals do
static int g_fg_pace;   // frame generation is presenting, g_fg_mult presents a game frame an interval / g_fg_mult apart: 1 the first in-between picture (present_window_locked),
                        // 2 the ones after it, the real frame last (fg_flush_one_locked)
static int g_fg_mult = 2;
static uint64_t g_fg_period;   // the game's frame time as frame generation sees it (fg_period: measured, whole refreshes); 0: the target's
static int g_fg_lead = 300, g_fg_early = 150;   // in hundredths of a refresh: how far ahead the in-between picture's time is asked, how long before that time the real frame is handed over (DM_FRAMEGEN_TUNE=lead,early)
extern int dm_fps_live __attribute__((weak));   // (loader.c: the live target the in-game menu switches, 30 or 60; 0 = the start's own)
static int game_fps(void) { int lf = &dm_fps_live ? __atomic_load_n(&dm_fps_live, __ATOMIC_RELAXED) : 0; const char *e = getenv("DM_FPS_TARGET"); return lf ? lf : e && atoi(e) >= 30 ? atoi(e) : 30; }
static uint64_t game_period_ns(void) { return 1000000000ull / (uint64_t)game_fps(); }
static int pace_time(VkPresentTimeGOOGLE *pt) {
    // the display's refresh asked again once a second: Android changes the mode after the swapchain is made (the app's frame rate vote, android_main.c window_rate:
    //   a Galaxy S25+ was at 60 Hz when the swapchain was made and at 120 a moment later, and pictures paced for 60 landed 8 and 25 ms apart)
    { static uint64_t last; uint64_t t = mono_ns(); if (g_dtiming && p_refresh && W.sc && t - last >= 1000000000ull) { last = t; VkRefreshCycleDurationGOOGLE rc = { 0 };
        if (!p_refresh(g_dev, W.sc, &rc) && rc.refreshDuration && (rc.refreshDuration > g_refresh_ns + g_refresh_ns / 16 || rc.refreshDuration + rc.refreshDuration / 16 < g_refresh_ns)) {
            VLOG("display refresh now %.3f ms (%.2f Hz), was %.3f ms", rc.refreshDuration / 1e6, 1e9 / (double)rc.refreshDuration, g_refresh_ns / 1e6); g_refresh_ns = rc.refreshDuration; g_vkp.next = 0; g_vkp.vs = 0; } } }
    if (!g_refresh_ns) return 0;
    uint64_t r = g_refresh_ns, now = mono_ns(); int fps = game_fps();
    if (g_fg_pace) fps *= g_fg_mult;   // (g_fg_period, when known, replaces it below)
    uint64_t frame = 1000000000ull / (uint64_t)fps; if (g_fg_pace && g_fg_period) frame = g_fg_period / (uint64_t)g_fg_mult;
    uint64_t k = (frame + r / 2) / r; if (!k) k = 1;
    // (frame generation asks times even one refresh apart: handed over unasked, two pictures that reach the compositor within one refresh show as one, the older
    //   dropped; on the S25+ at 4x 908 of the 1200 pictures in 10 s were seen)
    uint64_t iv = k * r; int pace = !pace_off() && (k >= 2 || g_fg_pace);
    if (p_past) {
        uint32_t n = 0;
        if (!p_past(g_dev, W.sc, &n, NULL) && n) {
            VkPastPresentationTimingGOOGLE t[16]; if (n > 16) n = 16;
            if (p_past(g_dev, W.sc, &n, t) >= 0) for (uint32_t i = 0; i < n; i++) {
                uint64_t a = t[i].actualPresentTime; g_pace.n++;
                if (g_vkp_plain[t[i].presentID % 64] == 1) { g_vkp_plain[t[i].presentID % 64] = 0;   // (a frame paced by vkp_want: the display's record of it)
                    int ch = vkp_seen(&g_vkp, t[i].desiredPresentTime, a, t[i].presentMargin, now, r);
                    if (ch) VLOG("frame pacing: %u of the last %u frames showed a refresh late%s: the lead %s a refresh, now GPU %.1f ms + %d refreshes (%.1f ms)",
                                 g_vkp.last_late, g_vkp.last_n, ch < 0 ? " (none for a while)" : "", ch > 0 ? "grows by" : "gives back",
                                 vkp_gpu_p95(&g_vkp) / 1e6, 1 + g_vkp.extra, vkp_need(&g_vkp, r) / 1e6); }
                if (a && !pace_adapt()) g_vkp.vs = a;   // (vkp_want_grid's phase: any frame shown is shown at a vsync)
                if (g_pace.prev_actual && a > g_pace.prev_actual) { uint64_t d = a - g_pace.prev_actual;
                    if (d > 4 * iv) g_pace.pauses++;
                    else { g_pace.sum += d; g_pace.nsum++; if (d + r / 2 < iv || d > iv + r / 2) g_pace.uneven++; } }
                g_pace.prev_actual = a; }
        }
    }
    uint64_t want = 0;
    // How far ahead of being handed over a frame is asked to show (the lead): one whole frame interval (33.4 ms at 30 fps). It was one refresh (16.7 ms), but the GPU
    //   draws a Legends: Arceus frame in 20-25 ms on a Galaxy S25+: a frame not finished by its time showed a refresh later, 33 then 50 ms apart, a shake every few
    //   seconds (5-19 uneven frames in 10 s with the game's own work at 7-12 ms). A frame now has its interval to be drawn: a steady 33 ms for some 16 ms more input delay.
    //   DM_VK_PACE_LEAD=<ms> sets it by hand (16 is the former one refresh)
    uint64_t lead = iv; { static long ms = -2; if (ms == -2) { const char *e = getenv("DM_VK_PACE_LEAD"); ms = e && atol(e) > 0 ? atol(e) : -1; }
        if (ms > 0) lead = (uint64_t)ms * 1000000ull; if (lead < r) lead = r; }
    // With DM_PACE_LEAD=1 (vk_pace.h vkp_want; off by default, see pace_adapt): a frame keeps its place on the grid only when its GPU time and the compositor's lead fit
    //   before it, else the grid moves once; times sit half a refresh before a reported vsync. At a held 30 the frames had come in 50 / 16.7 ms pairs at an unchanged
    //   33.3 ms mean (S25+ 4-20 %, POCO F6 19-33 % uneven); t_vk_pace's simulated compositor: 9-40 % before, under 1 % now, for 6-17 ms more input delay.
    //   Frame generation's pictures keep their own times (below), the real frame's from the grid
    int adapt = pace && !g_fg_pace && pace_adapt();
    if (adapt) {   // the GPU's time of the frames done since the last look (a frame three back is done: three swapchain slots, each waited on)
        uint64_t fno = g_frame_no; if (fno > 3 && g_vkp_gf + 64 < fno) g_vkp_gf = fno - 64;
        for (; fno > 3 && g_vkp_gf + 3 < fno; g_vkp_gf++) { double ms = mtl_gpu_ms(g_vkp_gf + 1); if (ms > 0 && ms < 1000) vkp_gpu(&g_vkp, (uint64_t)(ms * 1e6)); }
        want = vkp_want(&g_vkp, now, r, iv);
        g_vkp_plain[(g_pace.id + 1) % 64] = 1;
    } else if (pace) {
        want = dm_pace_grid_mode() == 1 ? vkp_want_legacy(&g_vkp, now, r, iv, lead) : vkp_want_grid(&g_vkp, now, r, iv, lead);   // (late: back to a full lead; ran ahead of the display: back in step)
        // frame generation: both pictures must reach the display early enough to make the refresh asked for, or one shows a refresh late and they land 8 and 25 ms
        //   apart (measured on a Galaxy S25+ at 120 Hz). The in-between picture asks for a time g_fg_lead ahead; the real frame exactly half an interval after it
        if (g_fg_pace == 1) { uint64_t lead = r * (uint64_t)g_fg_lead / 100; if (want < now + lead || want > now + lead + r) want = now + lead; }
        if (g_fg_pace == 2 && g_vkp.next) want = g_vkp.next + iv;
        g_vkp.next = want;
    }
    if (!adapt) g_vkp_plain[(g_pace.id + 1) % 64] = 0;
    g_pace_asked = want;
    pt->presentID = ++g_pace.id; pt->desiredPresentTime = want;
    if (now - g_pace.last_log >= 10000000000ull) {
        if (g_pace.last_log && g_pace.n)
            VLOG("frame timing (pacing %s): %llu frames shown in the last 10 s, %llu uneven, %llu pauses, mean interval %.2f ms (target %.2f ms = %llu refreshes of %.2f ms); "
                 "lead %.1f ms (GPU p95 %.1f ms + %d refreshes), grid moved %u times; held: %u late frames after a hitch, %u at the fixed lead (GPU over the interval)",
                 pace ? "on" : "off", (unsigned long long)g_pace.n, (unsigned long long)g_pace.uneven, (unsigned long long)g_pace.pauses,
                 g_pace.nsum ? g_pace.sum / 1e6 / (double)g_pace.nsum : 0.0, iv / 1e6, (unsigned long long)k, r / 1e6,
                 (pace_adapt() ? (g_vkp.over ? iv + 2 * r : vkp_lead(&g_vkp, r, iv)) : lead) / 1e6, vkp_gpu_p95(&g_vkp) / 1e6, 1 + g_vkp.extra, g_vkp.rephase, g_vkp.kept, g_vkp.fixed);
        g_vkp.rephase = g_vkp.kept = g_vkp.fixed = 0;
        g_pace.n = g_pace.uneven = g_pace.pauses = g_pace.sum = g_pace.nsum = 0; g_pace.last_log = now;
    }
    return 1;
}
// ---------- Touch overlay (nvn_ovl.h): drawn into the swapchain image after the game picture, one quad per control ----------
#include "overlay/ovl_spv.h"
#include "overlay/ovl_font.h"
__attribute__((weak)) int front_overlay(dm_ovl_item *it, int max, int width, int height) { (void)it; (void)max; (void)width; (void)height; return 0; }
static struct { VkFormat fmt; VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds; VkSampler samp;
                VkImage font; VkImageView fview; vmalloc_t fa, sa; VkBuffer stage; int font_ready, failed; } O;
typedef struct { float rect[4], color[4], shape[4], glyph[4], target[2], rot; } ovl_pc;
static uint64_t g_cmp_n, g_cmp_passes, g_cmp_clears, g_cmp_one;   // present_compose: pictures composed, passes on the swapchain image, whole-image clears, composed in one pass
static int ovl_setup(VkFormat fmt) {
    if (O.failed) return 0;
    if (O.pipe && O.fmt == fmt) return 1;
    if (O.pipe) { vkDestroyPipeline(g_dev, O.pipe, NULL); O.pipe = VK_NULL_HANDLE; }
    if (!O.pl) {
        VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f };
        VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8_UNORM, .extent = { ovl_font_w, ovl_font_h, 1 }, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
        if (vkCreateSampler(g_dev, &sci, NULL, &O.samp) || vkCreateImage(g_dev, &ci, NULL, &O.font)) { O.failed = 1; RED_ONCE(70, "overlay: cannot create its font"); return 0; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, O.font, &rq); O.fa = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, O.font, O.fa.b->mem, O.fa.off);
        VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = O.font, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8_UNORM, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCreateImageView(g_dev, &vci, NULL, &O.fview);
        void *m; O.stage = buf_make(sizeof ovl_font, &O.sa, &m); memcpy(m, ovl_font, sizeof ovl_font);
        VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
        VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b }; vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &O.dsl);
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps }; vkCreateDescriptorPool(g_dev, &dpi, NULL, &O.dp);
        VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = O.dp, .descriptorSetCount = 1, .pSetLayouts = &O.dsl }; vkAllocateDescriptorSets(g_dev, &dai, &O.ds);
        VkDescriptorImageInfo ii = { O.samp, O.fview, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = O.ds, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii };
        vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL);
        VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ovl_pc) };
        VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &O.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
        vkCreatePipelineLayout(g_dev, &pli, NULL, &O.pl);
    }
    VkShaderModule vm, fm;
    VkShaderModuleCreateInfo mv = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof ovl_vert_spv, .pCode = ovl_vert_spv }, mf = mv; mf.codeSize = sizeof ovl_frag_spv; mf.pCode = ovl_frag_spv;
    vkCreateShaderModule(g_dev, &mv, NULL, &vm); vkCreateShaderModule(g_dev, &mf, NULL, &fm);
    VkPipelineShaderStageCreateInfo st[2] = { { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main" },
                                              { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main" } };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState cba = { .blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf };
    VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = O.pl };
    if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(1, &fmt, 0, 1, 1); }
    VkResult r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &O.pipe);
    vkDestroyShaderModule(g_dev, vm, NULL); vkDestroyShaderModule(g_dev, fm, NULL);
    if (r) { O.failed = 1; RED_ONCE(71, "overlay: pipeline failed %d", r); return 0; }
    O.fmt = fmt; VLOG("touch overlay ready (format %d)", fmt);
    return 1;
}
// Records the overlay into cb over image img (view, size ext, format fmt), which is in TRANSFER_DST layout; leaves it in layout fin. Returns 0 (nothing recorded) when there is nothing to draw
static int ovl_draw_turned(VkCommandBuffer cb, VkImage img, VkImageView view, VkExtent2D ext, VkExtent2D lext, int rot, VkFormat fmt, VkImageLayout fin);
static int ovl_draw(VkCommandBuffer cb, VkImage img, VkImageView view, VkExtent2D ext, VkFormat fmt, VkImageLayout fin) { return ovl_draw_turned(cb, img, view, ext, ext, 0, fmt, fin); }
// (ext: the image; lext: the window the controls are laid out in; rot: the quarter turns between them)
// The controls to draw (it, at most max) for a window lext, with the pipeline for fmt and the label atlas ready: recorded into cb outside any pass. 0: nothing to draw
static int ovl_prep(VkCommandBuffer cb, VkFormat fmt, VkExtent2D lext, dm_ovl_item *it, int max) {
    int n = front_overlay(it, max, (int)lext.width, (int)lext.height);
    if (n <= 0 || !ovl_setup(fmt)) return 0;
    if (!O.font_ready) {   // the label atlas, once, in this same command buffer
        VkImageMemoryBarrier fb = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = O.font, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &fb);
        VkBufferImageCopy bc = { 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { ovl_font_w, ovl_font_h, 1 } };
        vkCmdCopyBufferToImage(cb, O.stage, O.font, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
        fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; fb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &fb);
        O.font_ready = 1;
    }
    return n;
}
// The controls (it, n from ovl_prep) drawn in the open pass over an image of size ext (lext / rot as below)
static void ovl_rec(VkCommandBuffer cb, VkExtent2D ext, VkExtent2D lext, int rot, const dm_ovl_item *it, int n) {
    VkViewport v = { 0, 0, (float)ext.width, (float)ext.height, 0, 1 }; VkRect2D sc = { {0, 0}, ext };
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, O.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, O.pl, 0, 1, &O.ds, 0, NULL);
    for (int i = 0; i < n; i++) {
        const dm_ovl_item *q = &it[i];
        ovl_pc pc = { { q->x, q->y, q->w, q->h }, { q->r, q->g, q->b, q->a }, { (float)q->shape, q->ring, q->radius, q->h > 0 ? q->w / q->h : 1 },
                      { (float)q->glyph, q->gscale > 0 ? q->gscale : 0.5f, q->galpha, q->gdark }, { (float)lext.width, (float)lext.height }, (float)rot };
        vkCmdPushConstants(cb, O.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, 6, 1, 0, 0);
    }
}
static int ovl_draw_turned(VkCommandBuffer cb, VkImage img, VkImageView view, VkExtent2D ext, VkExtent2D lext, int rot, VkFormat fmt, VkImageLayout fin) {
    dm_ovl_item it[128]; int n = view ? ovl_prep(cb, fmt, lext, it, 128) : 0;
    if (n <= 0) return 0;
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, ext }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    { uint32_t cf = (uint32_t)fmt; int sc = 0; for (uint32_t i = 0; i < W.n; i++) if (W.view[i] == view) sc = 1;   // (a screenshot's own image: a framebuffer per pass)
      rend_begin(cb, &ri, &cf, 0, 1, sc); g_cmp_passes += sc; }
    ovl_rec(cb, ext, lext, rot, it, n);
    rend_end(cb);
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = fin;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    return 1;
}
#ifndef __ANDROID__
// DM_OVL_SHOT=<frame>:<path.png> (Mac Vulkan host): the game frame fitted into a phone-sized picture (2340x1080) with the touch overlay over it, as present_window composes it
void touch_ui_demo(int c, float dx, float dy);
static void ovl_shot(vtex *x, const int crop[4], const char *path) {
    VkExtent2D ext = { 2340, 1080 }; VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = { ext.width, ext.height, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    VkImage im; if (vkCreateImage(g_dev, &ci, NULL, &im)) return;
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, im, &rq); vmalloc_t ia = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, im, ia.b->mem, ia.off);
    VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    VkImageView vw; vkCreateImageView(g_dev, &vci, NULL, &vw);
    vmalloc_t ba; void *map; VkBuffer rb = buf_make((VkDeviceSize)ext.width * ext.height * 4, &ba, &map);
    { const char *d = getenv("DM_OVL_PRESS"); touch_ui_demo(d ? atoi(d) : 2, 0, 0); }
    VkCommandBuffer cb = once_begin();
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    int32_t cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int32_t)x->w, ch = crop && crop[3] ? crop[3] : (int32_t)x->h;
    double sc = fmin((double)ext.width / cw, (double)ext.height / ch); int32_t dw = (int32_t)(cw * sc), dh = (int32_t)(ch * sc), dx = ((int32_t)ext.width - dw) / 2, dy = ((int32_t)ext.height - dh) / 2;
    int flip = atomic_load(&g_origin_ll);
    VkImageBlit bl = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{cx, flip ? cy + ch : cy, 0}, {cx + cw, flip ? cy : cy + ch, 1}}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{dx, dy, 0}, {dx + dw, dy + dh, 1}} };
    vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    if (!ovl_draw(cb, im, vw, ext, fmt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)) {
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
    VkBufferImageCopy bc = { 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { ext.width, ext.height, 1 } };
    vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &bc);
    once_end(cb);
    png_write(path, map, ext.width, ext.height, 0);
    VLOG("overlay shot %s", path);
}
#endif
// The window lock: present_window (a game thread) and nvn_vk_window_term (Android's main thread, when the window goes away) never touch the swapchain at once
static pthread_mutex_t g_wmu = PTHREAD_MUTEX_INITIALIZER;
// ---------- The game picture's upscaler (overlay/scale.frag): bilinear plus contrast-adaptive sharpening at the frame's own pixel size ----------
// Used when the picture is scaled up (Low graphics: a 720p frame on a 1080p-1440p screen; Legends: Arceus at its lower internal resolutions), so the lighter settings
// keep their detail instead of the soft look of a plain stretch. DM_SCALER=sharp (default) | smooth (the plain linear stretch, as before) | a sharpness 0.0-2.0.
// The blit stays the path whenever the upscaler cannot run (no push descriptors, a frame that cannot be sampled, a pipeline that failed): nothing is lost on a driver that lacks one
#include "overlay/scale_spv.h"
#include "overlay/sgsr_spv.h"   // Snapdragon Game Super Resolution 1 (Qualcomm, BSD-3-Clause: dm/third_party/sgsr)
static struct { VkFormat fmt[2]; VkPipeline pipe[2]; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkSampler samp; int failed, bad[2]; } SC;   // [0] sharp, [1] SGSR
typedef struct { float p0[4], p1[4], box[4], rot[4]; } scale_pc;   // sharp: crop uv, (1/w, 1/h, sharpness); SGSR: ViewportInfo (1/w, 1/h, w, h), crop uv
__attribute__((weak)) volatile int dm_scaler_mode = -1;   // 0 sharp, 1 SGSR, 2 smooth (the plain stretch), 3 FSR 1; -1: not read yet (DM_SCALER); the in-game menu sets it
static int scaler_mode(void) {
    int m = dm_scaler_mode; if (m >= 0) return m;
    const char *e = getenv("DM_SCALER"); m = e && !strcmp(e, "sgsr") ? 1 : e && !strcmp(e, "fsr") ? 3 : e && (!strcmp(e, "smooth") || !strcmp(e, "off")) ? 2 : 0;
    dm_scaler_mode = m; return m;
}
// The picture's colour (DM_COLOR: "vibrant" 1 / "cinematic" 2, else the game's 0; ☰ -> Colour, set live by android_main.c): graded in the upscaler's last shader
//   (scale.frag, fsr_rcas.frag: grade), so with a colour chosen that shader runs even when the picture is not scaled up. SGSR and the plain stretch keep the game's colour
__attribute__((weak)) volatile int dm_color_mode = -1;
static int color_mode(void) { int m = dm_color_mode; if (m >= 0) return m; const char *e = getenv("DM_COLOR"); m = e && !strcmp(e, "vibrant") ? 1 : e && !strcmp(e, "cinematic") ? 2 : 0; dm_color_mode = m; return m; }
__attribute__((weak)) volatile float dm_scaler_sharpness = -2;   // (an aligned float load / store is single-copy atomic on arm64)   // -2: not read yet; -1: off (smooth); the in-game menu sets it (android_main.c)
static float scale_sharpness(void) {
    float s = dm_scaler_sharpness;
    if (s > -2) return s;
    const char *e = getenv("DM_SCALER");
    s = !e || !*e || !strcmp(e, "sharp") || !strcmp(e, "fsr") || !strcmp(e, "sgsr") ? 0.7f : !strcmp(e, "smooth") || !strcmp(e, "off") ? -1.0f : fminf(2.0f, fmaxf(0.0f, (float)atof(e)));
    dm_scaler_sharpness = s;
    return s;
}
static int scale_setup(VkFormat fmt, int which) {
    if (SC.failed || SC.bad[which] || !p_push) return 0;
    if (SC.pipe[which] && SC.fmt[which] == fmt) return 1;
    if (SC.pipe[which]) { vkDestroyPipeline(g_dev, SC.pipe[which], NULL); SC.pipe[which] = VK_NULL_HANDLE; }
    if (!SC.pl) {
        VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f };
        VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
        VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 1, .pBindings = &b };
        VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(scale_pc) };
        if (vkCreateSampler(g_dev, &sci, NULL, &SC.samp) || vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &SC.dsl)) { SC.failed = 1; RED_ONCE(74, "upscaler: cannot create its sampler / layout: plain stretch"); return 0; }
        VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &SC.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
        if (vkCreatePipelineLayout(g_dev, &pli, NULL, &SC.pl)) { SC.failed = 1; RED_ONCE(75, "upscaler: no pipeline layout: plain stretch"); return 0; }
    }
    VkShaderModule vm, fm;
    VkShaderModuleCreateInfo mv = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = which ? sizeof sgsr_vert_spv : sizeof scale_vert_spv, .pCode = which ? sgsr_vert_spv : scale_vert_spv }, mf = mv;
    mf.codeSize = which ? sizeof sgsr_frag_spv : sizeof scale_frag_spv; mf.pCode = which ? sgsr_frag_spv : scale_frag_spv;
    vkCreateShaderModule(g_dev, &mv, NULL, &vm); vkCreateShaderModule(g_dev, &mf, NULL, &fm);
    VkPipelineShaderStageCreateInfo st[2] = { { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main" },
                                              { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main" } };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
    VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = SC.pl };
    if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(1, &fmt, 0, 1, 1); }
    VkResult r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &SC.pipe[which]);
    vkDestroyShaderModule(g_dev, vm, NULL); vkDestroyShaderModule(g_dev, fm, NULL);
    if (r) { SC.bad[which] = 1; SC.pipe[which] = VK_NULL_HANDLE; RED_ONCE(76, "upscaler %s: pipeline failed %d (%s)", which ? "SGSR" : "sharp", r, which ? "the sharp one instead" : "plain stretch"); return 0; }
    SC.fmt[which] = fmt; VLOG("upscaler %s ready (format %d)", which ? "SGSR" : "sharp", fmt);
    return 1;
}
// ---------- AMD FidelityFX Super Resolution 1 (DM_SCALER=fsr; overlay/fsr_easu.frag + fsr_rcas.frag, MIT: dm/third_party/fsr1) ----------
// Two passes as AMD made it: EASU upscales the game frame into an image the size of the picture on screen (FS.img, the swapchain's format, made again when that
//   size changes), RCAS sharpens it onto the screen (the sharpness setting: 0.7 → 0.2 stops, AMD's default; higher is sharper). Both use scale.vert with the crop
//   0..1, so their uv is the position inside the box. Push constants: scale.vert's 64 bytes, then FSR's four constant words (128 in all, every GPU's minimum).
//   Whatever fails (a pipeline, the image) leaves the sharp upscaler to do it, as before
#include "overlay/fsr_spv.h"
typedef struct { float src[4], outsz[4], box[4], rot[4]; uint32_t con[4][4]; } fsr_pc;
static struct { VkPipelineLayout pl; VkPipeline pipe[2]; VkFormat fmt; int failed; VkImage img; VkImageView view; vmalloc_t mem; uint32_t w, h; VkFormat ifmt; int laid; } FS;
static VkPipeline fsr_pipe(VkFormat fmt, const uint32_t *fs, size_t fsz) {
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE; VkPipeline p = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo mv = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof scale_vert_spv, .pCode = scale_vert_spv }, mf = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = fsz, .pCode = fs };
    if (vkCreateShaderModule(g_dev, &mv, NULL, &vm) || vkCreateShaderModule(g_dev, &mf, NULL, &fm)) goto out;
    VkPipelineShaderStageCreateInfo st[2] = { { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main" },
                                              { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main" } };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
    VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
    VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = FS.pl };
    if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(1, &fmt, 0, 1, 1); }
    if (vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &p)) p = VK_NULL_HANDLE;
out:
    if (vm) vkDestroyShaderModule(g_dev, vm, NULL); if (fm) vkDestroyShaderModule(g_dev, fm, NULL);
    return p;
}
static int fsr_setup(VkFormat fmt) {
    if (FS.failed || !p_push || !scale_setup(fmt, 0)) return 0;   // (the sampler and descriptor layout are the sharp upscaler's)
    if (FS.pipe[0] && FS.fmt == fmt) return 1;
    for (int i = 0; i < 2; i++) if (FS.pipe[i]) { vkDestroyPipeline(g_dev, FS.pipe[i], NULL); FS.pipe[i] = VK_NULL_HANDLE; }
    if (!FS.pl) {
        VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(fsr_pc) };
        VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &SC.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
        if (g_props.limits.maxPushConstantsSize < sizeof(fsr_pc) || vkCreatePipelineLayout(g_dev, &pli, NULL, &FS.pl)) { FS.failed = 1; RED_ONCE(52, "FSR: no pipeline layout: the sharp upscaler instead"); return 0; }
    }
    FS.pipe[0] = fsr_pipe(fmt, fsr_easu_frag_spv, sizeof fsr_easu_frag_spv); FS.pipe[1] = fsr_pipe(fmt, fsr_rcas_frag_spv, sizeof fsr_rcas_frag_spv);
    if (!FS.pipe[0] || !FS.pipe[1]) { FS.failed = 1; RED_ONCE(52, "FSR: its pipelines failed on this driver: the sharp upscaler instead"); return 0; }
    FS.fmt = fmt; VLOG("upscaler FSR 1 (EASU + RCAS) ready (format %d)", fmt);
    return 1;
}
static int fsr_target(uint32_t w, uint32_t h, VkFormat fmt) {   // the EASU image, w x h in the swapchain's format
    if (FS.img && FS.w == w && FS.h == h && FS.ifmt == fmt) return 1;
    if (FS.img) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu); vkDestroyImageView(g_dev, FS.view, NULL); vkDestroyImage(g_dev, FS.img, NULL); vk_free(FS.mem); FS.img = VK_NULL_HANDLE; FS.view = VK_NULL_HANDLE; }
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT };
    if (vkCreateImage(g_dev, &ci, NULL, &FS.img)) { FS.img = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, FS.img, &rq); FS.mem = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, FS.img, FS.mem.b->mem, FS.mem.off);
    VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = FS.img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    if (vkCreateImageView(g_dev, &vi, NULL, &FS.view)) { vkDestroyImage(g_dev, FS.img, NULL); vk_free(FS.mem); FS.img = VK_NULL_HANDLE; return 0; }
    FS.w = w; FS.h = h; FS.ifmt = fmt; FS.laid = 0;
    return 1;
}
static void fsr_easu_con(uint32_t con[4][4], float vw, float vh, float iw, float ih, float ow, float oh, float ox, float oy) {   // FsrEasuConOffset (ffx_fsr1.h) on the CPU
    float c[4][4] = { { vw / ow, vh / oh, 0.5f * vw / ow - 0.5f + ox, 0.5f * vh / oh - 0.5f + oy }, { 1.0f / iw, 1.0f / ih, 1.0f / iw, -1.0f / ih },
                      { -1.0f / iw, 2.0f / ih, 1.0f / iw, 2.0f / ih }, { 0.0f, 4.0f / ih, 0, 0 } };
    memcpy(con, c, sizeof c); con[3][2] = con[3][3] = 0;
}
// The picture's box (dm_picture_box, in the window's unturned layout) as a scissor in the swapchain image: turned with it and kept inside it
static VkRect2D box_scissor(const int32_t box[4]) {
    int32_t sx = box[0], sy = box[1], sw = box[2], shh = box[3];
    if (W.rot & 1) { int32_t t = sx; sx = sy; sy = t; t = sw; sw = shh; shh = t; }   // a turned swapchain: the box in the turned image (it is centred, so either turn lands the same)
    if (W.rot == 2) { sx = (int32_t)W.ext.width - sx - sw; sy = (int32_t)W.ext.height - sy - shh; }
    if (sx < 0) { sw += sx; sx = 0; } if (sy < 0) { shh += sy; sy = 0; }
    if (sx + sw > (int32_t)W.ext.width) sw = (int32_t)W.ext.width - sx; if (sy + shh > (int32_t)W.ext.height) shh = (int32_t)W.ext.height - sy;
    return (VkRect2D){ { sx, sy }, { (uint32_t)(sw > 0 ? sw : 0), (uint32_t)(shh > 0 ? shh : 0) } };
}
// RCAS from FS.img (EASU's result, in SHADER_READ_ONLY) into the box of the swapchain image whose pass is open in cb
static void fsr_rcas_rec(VkCommandBuffer cb, const int32_t box[4], float sharp) {
    uint32_t ow = (uint32_t)box[2], oh = (uint32_t)box[3];
    VkViewport sv2 = { 0, 0, (float)W.ext.width, (float)W.ext.height, 0, 1 };
    VkRect2D ssc = box_scissor(box);
    vkCmdSetViewport(cb, 0, 1, &sv2); vkCmdSetScissor(cb, 0, 1, &ssc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FS.pipe[1]);
    VkDescriptorImageInfo ri2 = { SC.samp, FS.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ri2 };
    p_push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FS.pl, 0, 1, &w);
    float lw = (float)W.lext.width, lh = (float)W.lext.height;
    fsr_pc rc = { { 0, 0, 1, 1 }, { (float)ow, (float)oh, 0, 0 }, { box[0] / lw * 2 - 1, box[1] / lh * 2 - 1, (box[0] + box[2]) / lw * 2 - 1, (box[1] + box[3]) / lh * 2 - 1 }, { (float)W.rot, 0, 0, 0 } };
    rc.con[1][0] = (uint32_t)color_mode();   // (fsr_rcas.frag: grade)
    float stops = sharp < 0 ? 2.0f : fminf(2.0f, fmaxf(0.0f, 1.4f - 1.714f * sharp)); float lin = exp2f(-stops); memcpy(&rc.con[0][0], &lin, 4);   // (FsrRcasCon: stops → linear; con[1] (half) unused here)
    vkCmdPushConstants(cb, FS.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof rc, &rc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}
// The game frame (view sv, tw x th, crop cx cy cw ch) through EASU into FS.img, left for RCAS to sample (recorded outside any pass). 0: nothing recorded
static int fsr_easu(VkCommandBuffer cb, VkImageView sv, uint32_t tw, uint32_t th, int32_t cx, int32_t cy, int32_t cw, int32_t ch, int flip, const int32_t box[4]) {
    uint32_t ow = (uint32_t)box[2], oh = (uint32_t)box[3];
    if (!ow || !oh || !fsr_setup(W.fmt) || !fsr_target(ow, oh, W.fmt)) return 0;
    // EASU: the whole EASU image, unrotated
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_READ_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = FS.img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = FS.view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, { ow, oh } }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    { uint32_t cf = (uint32_t)W.fmt; rend_begin(cb, &ri, &cf, 0, 1, 0); }
    VkViewport v = { 0, 0, (float)ow, (float)oh, 0, 1 }; VkRect2D sc = { { 0, 0 }, { ow, oh } };
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FS.pipe[0]);
    VkDescriptorImageInfo di = { SC.samp, sv, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &di };
    p_push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FS.pl, 0, 1, &w);
    fsr_pc pc = { { 0, 0, 1, 1 }, { (float)ow, (float)oh, 0, 0 }, { -1, -1, 1, 1 }, { 0, 0, 0, 0 } };
    fsr_easu_con(pc.con, (float)cw, (float)ch, (float)tw, (float)th, (float)ow, (float)oh, (float)cx, (float)cy);
    pc.con[3][2] = flip ? 1u : 0u;
    vkCmdPushConstants(cb, FS.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    rend_end(cb);
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    return 1;
}
// The game frame through EASU, then RCAS into swapchain image ii's box; the swapchain image is left in COLOR_ATTACHMENT with its pass begun no more: the caller records
//   the same end as for the other upscalers. 0: nothing recorded
static int fsr_draw(VkCommandBuffer cb, VkImageView sv, uint32_t tw, uint32_t th, int32_t cx, int32_t cy, int32_t cw, int32_t ch, int flip, uint32_t ii, const int32_t box[4], float sharp) {
    if (!fsr_easu(cb, sv, tw, th, cx, cy, cw, ch, flip, box)) return 0;
    // RCAS: into the swapchain image's box (the caller has it in COLOR_ATTACHMENT and records nothing else in between)
    VkImageMemoryBarrier sb = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,   /* (READ: the pass after loads it; SYNC-HAZARD-READ-AFTER-WRITE) */
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &sb);
    VkRenderingAttachmentInfo sa = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = W.view[ii], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };   // (LOAD: the black around the box stays)
    VkRenderingInfo si = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, W.ext }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &sa };
    { uint32_t cf = (uint32_t)W.fmt; rend_begin(cb, &si, &cf, 0, 1, 1); g_cmp_passes++; }
    fsr_rcas_rec(cb, box, sharp);
    rend_end(cb);
    return 1;
}
// The picture (view sv, tw x th, crop cx cy cw ch, flipped when its origin is lower left) drawn by upscaler which (0 sharp at sharpness sh, 1 SGSR) into the box of the
//   swapchain image whose pass is open in cb. The whole image is the viewport: the box is in the vertices. The vertices are one triangle twice the box's size (scale.vert /
//   sgsr.vert), so the scissor must be the box: with the whole image the picture spilled past the box's right edge and the triangle's long edge cut the corner below it
//   diagonally (a 2560x1440 picture on a 3120x1440 Galaxy S25+)
static void scale_rec(VkCommandBuffer cb, int which, VkImageView sv, uint32_t tw, uint32_t th, int32_t cx, int32_t cy, int32_t cw, int32_t ch, int flip, const int32_t box[4], float sh) {
    VkViewport v = { 0, 0, (float)W.ext.width, (float)W.ext.height, 0, 1 };
    VkRect2D sc = box_scissor(box);
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, SC.pipe[which]);
    VkDescriptorImageInfo di = { SC.samp, sv, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &di };
    p_push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, SC.pl, 0, 1, &w);
    float u0 = (float)cx / tw, u1 = (float)(cx + cw) / tw, v0 = (float)cy / th, v1 = (float)(cy + ch) / th;
    float lw = (float)W.lext.width, lh = (float)W.lext.height;
    scale_pc pc = { { 0 }, { 0 }, { box[0] / lw * 2 - 1, box[1] / lh * 2 - 1, (box[0] + box[2]) / lw * 2 - 1, (box[1] + box[3]) / lh * 2 - 1 }, { (float)W.rot, 0, 0, 0 } };
    float uv[4] = { u0, flip ? v1 : v0, u1, flip ? v0 : v1 };
    if (which) { float vi[4] = { 1.0f / tw, 1.0f / th, (float)tw, (float)th }; memcpy(pc.p0, vi, sizeof vi); memcpy(pc.p1, uv, sizeof uv); }
    else { float t[4] = { 1.0f / tw, 1.0f / th, sh, (float)color_mode() }; memcpy(pc.p0, uv, sizeof uv); memcpy(pc.p1, t, sizeof t); }
    vkCmdPushConstants(cb, SC.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}
// The game frame x (crop cx cy cw ch, flipped when its origin is lower left) drawn into swapchain image ii's box, the rest black; the image is left in TRANSFER_DST layout,
// as after the blit. Returns 0 (nothing recorded: the caller blits) when the upscaler does not apply
static int scale_draw_view(VkCommandBuffer cb, VkImageView sv, uint32_t tw, uint32_t th, int32_t cx, int32_t cy, int32_t cw, int32_t ch, int flip, uint32_t ii, const int32_t box[4]) {
    int mode = scaler_mode(); float sh = scale_sharpness();
    if (!W.rot && !color_mode() && (mode == 2 || (sh < 0 && mode != 3) || box[2] <= cw || box[3] <= ch)) return 0;   // not scaling up, or the plain stretch chosen: the blit (a turned swapchain, or a colour chosen, always draws here)
    if (!sv || tw == 0 || th == 0 || !W.view[ii]) return 0;
    if (mode == 3 && box[2] > cw && box[3] > ch && fsr_draw(cb, sv, tw, th, cx, cy, cw, ch, flip, ii, box, sh)) {   // FSR 1 (fsr_draw); else the sharp upscaler below
        VkImageMemoryBarrier eb = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &eb);
        static int said; if (!said++) VLOG("upscaler FSR 1: %dx%d → %dx%d%s", cw, ch, box[2], box[3], W.rot ? " (pre-rotated)" : "");
        return 1;
    }
    int which = mode == 1 && !color_mode() ? 1 : 0; if (which && !scale_setup(W.fmt, 1)) which = 0;   // (SGSR has no colour: with one chosen, the sharp upscaler)
    if (!scale_setup(W.fmt, which)) return 0;
    if (mode == 2 || sh < 0) sh = 0;   // (turned but smooth: plain bilinear)
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,   /* (READ: the pass after loads it; SYNC-HAZARD-READ-AFTER-WRITE) */
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = W.view[ii], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };   // (LOAD: the black the caller cleared stays around the box)
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, W.ext }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    { uint32_t cf = (uint32_t)W.fmt; rend_begin(cb, &ri, &cf, 0, 1, 1); g_cmp_passes++; }
    scale_rec(cb, which, sv, tw, th, cx, cy, cw, ch, flip, box, sh);
    rend_end(cb);
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    static int said; if (!said++) VLOG("upscaler %s: %dx%d → %dx%d%s", which ? "SGSR" : "sharp", cw, ch, box[2], box[3], W.rot ? " (pre-rotated)" : "");
    return 1;
}
// (the game's own frame x: sampled through a view of it; sv / tw / th above: any picture in the GENERAL layout, frame generation's in-between one)
static int scale_draw(VkCommandBuffer cb, vtex *x, int32_t cx, int32_t cy, int32_t cw, int32_t ch, int flip, uint32_t ii, const int32_t box[4]) {
    if (!(x->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || x->is3d) return 0;
    if (!W.rot && !color_mode() && (scaler_mode() == 2 || (scale_sharpness() < 0 && scaler_mode() != 3) || box[2] <= cw || box[3] <= ch)) return 0;   // (the blit: no view needed)
    return scale_draw_view(cb, view_get(x, VK_IMAGE_VIEW_TYPE_2D, x->fmt, 0, 1, 0, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT), x->w, x->h, cx, cy, cw, ch, flip, ii, box);
}
// ---------- Frame generation (nvn_fgen.h): DM_FRAMEGEN=on, off by default, experimental ----------
// A 30 fps game shown at 60: for each game frame the picture halfway between it and the frame before is made on the GPU (motion found on a brightness pyramid, both
// frames moved half way and averaged) and shown first; the real frame follows half a frame interval later, handed to the display by a small thread so the game's
// thread never waits for it. So the real frame is seen half an interval late (16.7 ms at 30 fps, plus the GPU's time for the picture): that is the input delay it
// costs, and the log states the measured figure every 10 s. The whole final picture is interpolated, HUD included: a pixel that is the same in both frames is left
// alone, so still HUD and text stay sharp; a HUD element that moves or sits half-transparent over a moving scene is moved with what is under it (the scene before
// the UI pass would have to be captured to do better: not done yet).
// Normal presentation stays the path whenever it cannot run: a format the driver cannot render to, no second swapchain image free in time, a frame that came after
// a pause, a frame size that keeps changing, a GPU that needs longer than DM_FRAMEGEN_BUDGET_MS (default 6) for the picture
#define FG_MEM
typedef vmalloc_t fg_mem;
static int fg_mem_get(VkDevice dev, VkPhysicalDevice pd, VkImage im, fg_mem *m) {
    (void)pd; VkMemoryRequirements rq; vkGetImageMemoryRequirements(dev, im, &rq); *m = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    return m->b && !vkBindImageMemory(dev, im, m->b->mem, m->off);
}
static void fg_mem_put(VkDevice dev, fg_mem m) { (void)dev; vk_free(m); }
#include "nvn_fgen.h"
// DM_FRAMEGEN=lsfg: the same presentation, the picture made by the player's own Lossless Scaling shaders (nvn_lsfg.h, GPL-3.0-or-later) instead of nvn_fgen.h.
// DM_LSFG_DIR: the folder of the shaders converted from the player's Lossless.dll (<id>.spv; the app's import puts them there), DM_LSFG_FLOW: the motion search's
// size in percent of the frame (25..100; default 50 for frames 1000 lines or taller, else 75)
#include "nvn_lsfg.h"
#include "nvn_fgcut.h"
__attribute__((weak)) volatile int dm_framegen_mode = -1;   // 0 off, 1 on, 2 on with the motion drawn over the picture (DM_FRAMEGEN=debug), 3 Lossless Scaling's (DM_FRAMEGEN=lsfg);
                                                            // -1: not read yet; the in-game menu sets it
static int framegen_mode(void) {
    int m = dm_framegen_mode; if (m >= 0) return m;
    const char *e = getenv("DM_FRAMEGEN"); m = e && (!strcmp(e, "on") || !strcmp(e, "1")) ? 1 : e && !strcmp(e, "debug") ? 2 : e && !strcmp(e, "lsfg") ? 3 : 0;
    dm_framegen_mode = m; return m;
}
static struct { fgen g; lsfg ls; int ls_tried, mult_cap; uint64_t ivl, ivl_last; fg_plan plan; int off, up, p, have_prev, resizes, hf_last, zgen; uint64_t resize_t;
                VkSemaphore acqx[3][LS_MAXGEN], donex[3][LS_MAXGEN]; VkQueryPool qp; double tsper; int q_live[3], q_pics[3]; uint64_t budget;
                // the pictures after the first one waiting for their time, the real frame last (under g_wmu): pending of them from phead on, presented g_fg_mult = pmult a game frame
                int pending, phead, pmult; uint32_t pii[LS_MAXGEN]; VkSemaphore pdone[LS_MAXGEN]; uint64_t pdue[LS_MAXGEN], seq, arrived; sem_t wake;
                uint64_t n_gen, n_plain, gpu_sum, gpu_n, gpu_max, cpu_sum, lag_sum, lag_n, last_log;
                // slow: off by the GPU-time cut-off (tried again when the menu switches frame generation on or to the other kind); mode_seen: the mode the last frame had;
                // held: paused while the game itself runs under DM_FRAMEGEN_MIN_FPS (rate_ivl: its frame interval, a slow running mean of rate_n intervals); note: for the menu
                int slow, mode_seen, held, min_fps; uint64_t rate_ivl, rate_last, rate_n; char note[48];
                // fg_adapt: 3 s windows of the game's own frames (aw_*), its rate without pictures (base_fps), the lighter settings taken (flow_cut: the motion search a
                // quarter smaller each; mult_cap above), windows in a row that kept the game's rate (ok_win), windows since the last step up (since_up), the wait before one
                uint64_t aw_t, aw_max, aw_prev, last_gen_t; int aw_n, aw_gen, aw_skip, flow_cut, ok_win, since_up, up_wait; double base_fps;
                // the floor's time rule (a pause after 2 s under it, back after 5 s over it; grace_t: the guard took the cost, no pause before then); hot_win: windows hot
                uint64_t flip_t, grace_t; int flip_to, hot_win;
                // camera cuts and still frames (nvn_fgcut.h): its state, whether it was tried; still_n: read-backs in a row that barely moved; hf_ever: HUD-free has worked
                //   (a frame without the scene then is a full-screen menu or a loading screen); counts for the 10-s line
                fgc cut; int cut_tried, still_n, hf_ever; uint64_t n_cut, n_still, n_menu;
                // work size (Lossless Scaling HUD-free): the scale the chain runs at (1: the frame's own size), the one asked and since when
                float ws, ws_want; uint64_t ws_t; int ls_fell; } FGE;
// The real frame that waits behind its in-between picture goes to the display: at its time (fg_thread), or at once when the next frame, a new swapchain or the
// window's end comes first. Called with the window lock held
static void fg_flush_one_locked(void) {   // the oldest waiting picture
    if (!FGE.pending) return;
    int h = FGE.phead++; FGE.pending--;
    VkPresentInfoKHR pi = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &FGE.pdone[h], .swapchainCount = 1, .pSwapchains = &W.sc, .pImageIndices = &FGE.pii[h] };
    VkPresentTimeGOOGLE pt; VkPresentTimesInfoGOOGLE pti = { .sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, .swapchainCount = 1, .pTimes = &pt };
    g_fg_pace = 2; g_fg_mult = FGE.pmult; if (pace_time(&pt)) pi.pNext = &pti; g_fg_pace = 0;
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueuePresentKHR(g_q, &pi); pthread_mutex_unlock(&g_qmu);
    if (!FGE.pending) { FGE.lag_sum += mono_ns() - FGE.arrived; FGE.lag_n++; }   // (the real frame, last of them)
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { FGE.pending = 0; swap_destroy(); }
}
static void fg_flush_locked(void) { while (FGE.pending) fg_flush_one_locked(); }
static void *fg_thread(void *a) { DM_THREAD_NAME("dm-framegen");
    (void)a;
    for (;;) {
        sem_wait(&FGE.wake);
        for (;;) {   // each waiting picture at its time, while they are still this frame's (a new frame or a new swapchain flushes them itself)
            pthread_mutex_lock(&g_wmu); uint64_t seq = FGE.seq; int pend = FGE.pending, h = FGE.phead; uint64_t due = pend ? FGE.pdue[h] : 0; pthread_mutex_unlock(&g_wmu);
            if (!pend) break;
            struct timespec ts = { (time_t)(due / 1000000000ull), (long)(due % 1000000000ull) }; clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
            pthread_mutex_lock(&g_wmu); int same = FGE.pending && FGE.seq == seq && FGE.phead == h; if (same) fg_flush_one_locked(); pthread_mutex_unlock(&g_wmu);
            if (!same) break; }
    }
    return NULL;
}
// Lossless Scaling: pictures a game frame, the real one included (2x .. 4x). DM_LSFG_MULT=2 / 3 / 4, or auto: the most of 4 and 3 whose rate the display's refresh
// divides evenly (the game at 30 fps on 120 Hz: 4x; at 90 Hz: 3x; measured, fg_period), else 2x (3x at 120 Hz would show the pictures 1 and 2 refreshes apart in turn). Never above mult_cap, which a
// GPU too slow for it lowers
__attribute__((weak)) volatile int dm_lsfg_mult = -1;   // 0 auto, 2..4; -1 not read yet; the in-game menu sets it
// (Auto, n refreshes a game frame: 4 or 8: 4x; 3 or 6: 3x; 2: 2x; 5, 7: 4x, the steps one refresh uneven)
// The game's frame time for frame generation: what it measures (a running mean of the frame intervals, pauses left out), never below the target's, in whole
// refreshes when the display's are known. A game meant for 60 that runs at 30 is treated as 30: pictures go between its frames as they come, 4x at 120 Hz
static uint64_t fg_period(uint64_t t_arr) {
    uint64_t target = game_period_ns(), d = FGE.ivl_last ? t_arr - FGE.ivl_last : 0; FGE.ivl_last = t_arr;
    if (d && d < 4 * target && d < 250000000ull) FGE.ivl = FGE.ivl ? (FGE.ivl * 7 + d) / 8 : d;
    uint64_t p = FGE.ivl > target ? FGE.ivl : target, r = g_refresh_ns;
    if (r) { uint64_t n = (p + r / 2) / r; p = (n ? n : 1) * r; }
    return p;
}
static int lsfg_mult(void) {
    int m = dm_lsfg_mult;
    if (m < 0) { const char *e = getenv("DM_LSFG_MULT"); m = e ? atoi(e) : 0; if (m < 2 || m > 4) m = 0; dm_lsfg_mult = m; }
    if (!m) {   // auto: as many pictures as the display's refreshes a game frame (fg_period: measured, whole refreshes): a 30 fps game at 120 Hz 4x, at 90 Hz 3x, at 60 Hz 2x;
                //   a 60 fps game at 120 Hz 2x. 6 refreshes: 3x (two each), 8: 4x; 5 or 7: 4x (the steps one refresh uneven); 1 (the game at the display's rate): 2x
        uint64_t r = g_refresh_ns, p = g_fg_period; int n = r && p ? (int)((p + r / 2) / r) : 0;
        m = n >= 4 && n % 4 == 0 ? 4 : n >= 3 && n % 3 == 0 ? 3 : n == 2 ? 2 : n >= 5 ? 4 : 2; }
    if (m > FGE.mult_cap) m = FGE.mult_cap;
    return m < 2 ? 2 : m;
}
static void fg_off(const char *why) {
    FGE.off = 1; FGE.have_prev = 0; snprintf(FGE.note, sizeof FGE.note, "off");
    FGE.n_gen = FGE.n_plain = FGE.gpu_sum = FGE.gpu_n = FGE.gpu_max = FGE.cpu_sum = FGE.lag_sum = FGE.lag_n = 0; FGE.last_log = 0;
    VLOG("[RED] frame generation off until the next start (%s): normal presentation", why);
}
// One governor (android_main.c graphics_auto): the 3D render scale, the pictures a frame and the motion search's size share the GPU. When frame generation costs the
//   game its frames, the render scale gives way first (the guard steps it down and judges the step as its own); only when it has no room left (at its lowest, a CPU
//   limit, or a step that did not help) does frame generation get lighter. The guard counts frame generation's GPU time into its budget (nvn_vk_fg_gpu_ms), and heat:
//   hot with the scale already at its heat step, frame generation steps down too (fg_adapt). Returns 1 when the guard took it
int dm_guard_fg_absorb(void) __attribute__((weak)); int dm_guard_heat(void) __attribute__((weak));
static int fg_gov_absorb(const char *why, uint64_t now) {
    if (!dm_guard_fg_absorb || !dm_guard_fg_absorb()) return 0;
    VLOG("frame generation costs the game frames (%s): the 3D render scale gives way first", why);
    FGE.grace_t = now + 15000000000ull; FGE.aw_skip = 4; FGE.ok_win = 0; memset(&FGE.plan, 0, sizeof FGE.plan); return 1;
}
static _Atomic uint64_t g_fg_gpu_ema;   // frame generation's GPU time a game frame (ns, slow mean), for the guard's budget
double nvn_vk_fg_gpu_ms(void) { return framegen_mode() && !FGE.off && FGE.last_gen_t && mono_ns() - FGE.last_gen_t < 2000000000ull ? atomic_load(&g_fg_gpu_ema) / 1e6 : 0.0; }
// One step lighter: Lossless Scaling's pictures a frame one fewer (4x, 3x, 2x), then its motion search a quarter smaller (75 %, 50 %, 25 % of the frame), then off
// (tried again when the menu switches it on). The window after a step is left to settle; a step down soon after a step up doubles the wait before the next one up
static void fg_step_down(const char *why, const char *note) {
    int m = framegen_mode(), cur = m == 3 ? lsfg_mult() : 2;
    if (m == 3 && cur > 2) { FGE.mult_cap = cur - 1; VLOG("[RED] frame generation lighter (%s): %dx from now on", why, cur - 1); }
    else if (m == 3 && FGE.flow_cut < 2) { FGE.flow_cut++; VLOG("[RED] frame generation lighter (%s): the motion search a quarter smaller (%d %% less)", why, 25 * FGE.flow_cut); }
    else if (!strcmp(note, "heat")) { memset(&FGE.plan, 0, sizeof FGE.plan); return; }   // (heat alone never switches it off: the guard keeps the scale down meanwhile)
    else { char w[192]; snprintf(w, sizeof w, "%s, at its lightest; switching it on again in the menu tries again", why); fg_off(w); FGE.slow = 1; snprintf(FGE.note, sizeof FGE.note, "%s", note); }
    memset(&FGE.plan, 0, sizeof FGE.plan); FGE.aw_skip = 1; FGE.ok_win = 0;
    if (FGE.since_up < 10 && FGE.up_wait < 160) FGE.up_wait *= 2;   // (a step up that did not hold: wait longer before the next, 1 min .. 8 min)
}
// Frame generation that knows what it costs this phone: every 3 s of the game's own frames, the rate they came at. Windows without pictures (paused, or the first
// 3 s) give the game's own rate; windows with pictures throughout must keep it (92 %, or of the game's target when it runs at its cap), else one step lighter. After
// up_wait windows (20 = 1 min) that kept it, a step back up (the motion search first, then a picture more). Windows with a pause over 0.5 s (a load) are left out
static void fg_adapt(uint64_t now, int gen) {
    if (!FGE.up_wait) FGE.up_wait = 20;
    if (gen) FGE.last_gen_t = now;
    uint64_t d = FGE.aw_prev ? now - FGE.aw_prev : 0; FGE.aw_prev = now;
    if (!FGE.aw_t) { FGE.aw_t = now; FGE.aw_n = FGE.aw_gen = 0; FGE.aw_max = 0; return; }
    if (d > FGE.aw_max) FGE.aw_max = d;
    FGE.aw_n++; FGE.aw_gen += gen != 0;
    if (now - FGE.aw_t < 3000000000ull) return;
    double fps = FGE.aw_n * 1e9 / (double)(now - FGE.aw_t); int all = FGE.aw_gen * 10 >= FGE.aw_n * 9, none = !FGE.aw_gen; uint64_t mx = FGE.aw_max;
    FGE.aw_t = now; FGE.aw_n = FGE.aw_gen = 0; FGE.aw_max = 0;
    if (mx > 500000000ull || FGE.off) return;
    if (FGE.aw_skip) { FGE.aw_skip--; return; }
    if (none) { FGE.base_fps = fps; return; }
    if (!all || FGE.base_fps <= 0) return;
    double tgt = game_fps(), ref = FGE.base_fps < tgt * 0.97 ? FGE.base_fps : tgt; FGE.since_up++;
    if (fps < ref * 0.92) { char w[96], n[24]; snprintf(w, sizeof w, "the game's own frames %.1f fps with it, %.1f without", fps, FGE.base_fps); snprintf(n, sizeof n, "cost:%.0f", fps);
        if (!fg_gov_absorb(w, now)) fg_step_down(w, n);
        return; }
    // heat: hot for 20 s with the render scale already at its heat step (the guard's 2): a step lighter, the same way (the scale gave way first)
    int heat = dm_guard_heat ? dm_guard_heat() : 0;
    FGE.hot_win = heat == 2 ? FGE.hot_win + 1 : 0;
    if (FGE.hot_win >= 7 && (FGE.mult_cap > 2 || FGE.flow_cut < 2)) { FGE.hot_win = 0; fg_step_down("the phone is hot and the 3D render scale is already down for it", "heat"); return; }
    if (heat) { FGE.ok_win = 0; return; }   // (no step back up while warm)
    if (++FGE.ok_win >= FGE.up_wait && (FGE.flow_cut || FGE.mult_cap < 4)) {
        FGE.ok_win = 0; FGE.since_up = 0; FGE.aw_skip = 1; memset(&FGE.plan, 0, sizeof FGE.plan);
        if (FGE.flow_cut) { FGE.flow_cut--; VLOG("frame generation heavier again: the motion search back to %d %% less (the game kept %.1f fps for %d s)", 25 * FGE.flow_cut, fps, FGE.up_wait * 3); }
        else { FGE.mult_cap++; VLOG("frame generation heavier again: up to %dx (the game kept %.1f fps for %d s)", FGE.mult_cap, fps, FGE.up_wait * 3); } }
}
// The menu's line under Frame generation (GameMenu.nativeFramegenNote): "" running or off, "paused:<fps>" under the minimum frame rate, "cost:<fps>" lightest setting still cost the game its frames, "slow:<ms>" cut off for the GPU's
// time (until switched on again), "off" off for another reason until the next start
void nvn_vk_framegen_note(char *buf, size_t n) { if (!n) return; pthread_mutex_lock(&g_wmu); snprintf(buf, n, "%s", framegen_mode() ? FGE.note : ""); pthread_mutex_unlock(&g_wmu); }
// Frame generation can take this frame (the crop cx cy cw ch of x): its semaphores, thread and GPU timer once, its images for this size and format
// Lossless Scaling cannot run on this driver or with these shaders: our own frame generation instead, not off (what it lacks said once; the menu's note says "ours")
static int fg_ls_fallback(const char *why) {
    if (!FGE.ls_fell) VLOG("[RED] Lossless Scaling frame generation cannot run here (%s): our own frame generation instead", why ? why : "?");
    FGE.ls_fell = 1; dm_framegen_mode = 1; snprintf(FGE.note, sizeof FGE.note, "ours");
    return 0;
}
// The work size (Lossless Scaling, HUD-free): the scale of the game's 3D render size (dm_scene_scale: what the game and the guard draw the 3D at; 1 where unknown) in
//   eighths, rounded up; a new scale is taken after 5 s at it, so the images are not made again on every short swing. DM_FG_WORK=0: the frame's own size;
//   DM_FG_WORK=50..100: that percentage
float dm_scene_scale(void) __attribute__((weak));
static float fg_work_scale(uint64_t now) {
    static int fixed = -2; if (fixed == -2) { const char *e = getenv("DM_FG_WORK"); fixed = !e ? -1 : !strcmp(e, "0") ? 100 : atoi(e) >= 50 && atoi(e) <= 100 ? atoi(e) : -1; }
    if (fixed > 0) return fixed / 100.0f;
    float s = dm_scene_scale ? dm_scene_scale() : 1.0f; if (!(s >= 0.3f && s <= 1.0f)) s = 1.0f;
    float q = ceilf(s * 8.0f - 0.01f) / 8.0f; if (q < 0.5f) q = 0.5f; if (q > 1.0f) q = 1.0f;
    if (!(FGE.ws > 0)) FGE.ws = 1.0f;
    if (q != FGE.ws_want) { FGE.ws_want = q; FGE.ws_t = now; }
    if (q != FGE.ws && now - FGE.ws_t >= 5000000000ull) { VLOG("frame generation: works at %.0f %% of the frame (the game's 3D at %.0f %%), its pictures brought back to the frame's size after", q * 100, s * 100); FGE.ws = q; }
    return FGE.ws;
}
static int fg_ready(vtex *x, int32_t cx, int32_t cy, int32_t cw, int32_t ch, float ws) {
    if (x->is3d || !(x->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || cx < 0 || cy < 0 || cw <= 0 || ch <= 0 || (uint32_t)(cx + cw) > x->w || (uint32_t)(cy + ch) > x->h) {
        RED_ONCE(120, "frame generation: the game's frame cannot be copied (usage %#x, %ux%u, crop %d,%d %dx%d): normal presentation", (unsigned)x->usage, x->w, x->h, cx, cy, cw, ch); return 0; }
    if (!FGE.up) {
        VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; int bad = 0;
        for (int i = 0; i < 3; i++) for (int k = 0; k < LS_MAXGEN; k++) bad |= vkCreateSemaphore(g_dev, &sci, NULL, &FGE.acqx[i][k]) || vkCreateSemaphore(g_dev, &sci, NULL, &FGE.donex[i][k]);
        FGE.mult_cap = 4;
        pthread_t t; if (bad || sem_init(&FGE.wake, 0, 0) || pthread_create(&t, NULL, fg_thread, NULL)) { fg_off("no semaphores / thread for it"); return 0; }
        pthread_detach(t);
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
        if (g_qf < nq && qp[g_qf].timestampValidBits && g_props.limits.timestampPeriod > 0) {   // (two timestamps a frame, around its own passes)
            VkQueryPoolCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 6 };
            if (vkCreateQueryPool(g_dev, &qi, NULL, &FGE.qp)) FGE.qp = VK_NULL_HANDLE; FGE.tsper = g_props.limits.timestampPeriod; }
        { const char *e = getenv("DM_FRAMEGEN_BUDGET_MS"); double ms = e && atof(e) > 0 ? atof(e) : 6.0; FGE.budget = (uint64_t)(ms * 1e6); }
        { const char *e = getenv("DM_FRAMEGEN_TUNE"); int a, b; if (e && sscanf(e, "%d,%d", &a, &b) == 2 && a >= 50 && a <= 400 && b >= -100 && b <= 300) { g_fg_lead = a; g_fg_early = b;
            VLOG("frame generation: DM_FRAMEGEN_TUNE: the in-between picture asked %.2f refreshes ahead, the real frame handed over %.2f refreshes before that time", a / 100.0, b / 100.0); } }
        FGE.up = 1;
    }
    if (!FGE.cut_tried) { FGE.cut_tried = 1;
        if (fgc_init(&FGE.cut, g_dev, g_pd)) VLOG("frame generation: camera cuts %s (the newer frame shown instead of a blend), still frames and full-screen menus left without pictures%s",
                                                    !FGE.cut.on ? "not handled (DM_FG_CUT=0)" : FGE.cut.can_select ? "handled" : "handled for our own frame generation only (no storage writes without a format)", "");
        else VLOG("frame generation: no cut / still check (%s)", FGE.cut.why ? FGE.cut.why : "?"); }
    fgen *g = &FGE.g; lsfg *ls = framegen_mode() == 3 ? &FGE.ls : NULL; float flow = 1; uint32_t cw_work = (uint32_t)cw, ch_work = (uint32_t)ch;
    if (ls) {
        if (!FGE.ls_tried) { FGE.ls_tried = 1; const char *d = getenv("DM_LSFG_DIR");
            if (g_api < VK_API_VERSION_1_3) return fg_ls_fallback("it needs a Vulkan 1.3 driver");
            if (!d || !*d || !ls_load(ls, d)) return fg_ls_fallback(d && *d ? ls->why : "the Lossless Scaling shaders are not imported"); }
        { const char *e = getenv("DM_LSFG_FLOW"); int pc = e && atoi(e) > 0 ? atoi(e) : 50;   // (light: half the frame's size, any height)
          pc -= 25 * FGE.flow_cut; flow = (pc < 25 ? 25 : pc > 100 ? 100 : pc) / 100.0f; }   // (S25+, 1080p: 100% took 6 ms a picture, 50% 2.9-4.4 ms)
        { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, x->fmt, &fp); const VkFormatFeatureFlags bl = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
          if ((fp.optimalTilingFeatures & bl) != bl) ws = 1.0f; }   // (the frame is scaled down with a blit)
        uint32_t ww = ws < 1.0f ? ((uint32_t)((float)cw * ws + 1.0f) & ~1u) : (uint32_t)cw, wh = ws < 1.0f ? ((uint32_t)((float)ch * ws + 1.0f) & ~1u) : (uint32_t)ch;
        if (ww > (uint32_t)cw) ww = (uint32_t)cw;
        if (wh > (uint32_t)ch) wh = (uint32_t)ch;
        if (ls->w == ww && ls->h == wh && ls->ow == (uint32_t)cw && ls->oh == (uint32_t)ch && ls->fmt == x->fmt && ls->flow == flow) return 1;
        cw_work = ww; ch_work = wh; }
    else if (g->w == (uint32_t)cw && g->h == (uint32_t)ch && g->fmt == x->fmt) return 1;
    uint64_t now = mono_ns();   // a game that changes its frame size all the time (dynamic resolution) would rebuild the images each time: not for it
    if (now - FGE.resize_t > 20000000000ull) { FGE.resize_t = now; FGE.resizes = 0; }
    int ws_only = ls && ls->w && ls->ow == (uint32_t)cw && ls->oh == (uint32_t)ch && ls->fmt == x->fmt;   // (only the work size changed: at most every 5 s, fg_work_scale)
    if (!ws_only && ++FGE.resizes > 8) { fg_off("the game's frame size changed 8 times in 20 s"); return 0; }
    if (ls ? ls->w : g->w) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu); }   // (its images are destroyed: nothing in flight may use them)
    FGE.have_prev = 0; memset(FGE.q_live, 0, sizeof FGE.q_live);
    if (ls) {
        if (!ls_setup_work(ls, g_dev, g_pd, x->fmt, cw_work, ch_work, flow, (uint32_t)cw, (uint32_t)ch)) {
            if (ls->failed) return fg_ls_fallback(ls->why ? ls->why : "setup failed");
            RED_ONCE(122, "Lossless Scaling frame generation: %s (%dx%d): normal presentation", ls->why ? ls->why : "not for this frame", cw, ch);
            return 0; }
        VLOG("Lossless Scaling frame generation ready: %dx%d frames (format %d), working at %ux%u, motion on %ux%u (%.0f%%), %d images, %dx now (display %.1f Hz, game %d fps%s)%s; the GPU's time for it %s; "
             "off above %.1f ms a picture, half that more for each picture after the first", cw, ch, (int)x->fmt, ls->w, ls->h, ls->fw, ls->fh, flow * 100, ls->nall, lsfg_mult(),
             g_refresh_ns ? 1e9 / (double)g_refresh_ns : 0.0, game_fps(), dm_lsfg_mult > 0 ? ", DM_LSFG_MULT" : ", auto", ls->stripped ? "; without the shaders' float controls (the driver has none)" : "",
             FGE.qp ? "measured" : "cannot be measured on this driver (no timestamps)", FGE.budget / 1e6);
        return 1; }
    if (!fg_setup(g, g_dev, g_pd, x->fmt, (uint32_t)cw, (uint32_t)ch)) {
        if (g->failed) fg_off(g->why ? g->why : "setup failed"); else RED_ONCE(121, "frame generation: %s (%dx%d): normal presentation", g->why ? g->why : "not for this frame", cw, ch);
        return 0; }
    VLOG("frame generation ready: %dx%d frames (format %d), motion on %ux%u down to %ux%u; the GPU's time for it %s; off above %.1f ms a picture", cw, ch, (int)x->fmt,
         g->lw[0], g->lh[0], g->lw[FG_LEVELS - 1], g->lh[FG_LEVELS - 1], FGE.qp ? "measured" : "cannot be measured on this driver (no timestamps)", FGE.budget / 1e6);
    return 1;
}
// ---------- Frame generation's PC score: real frames recorded for tests/t_fg_score.c (DM_FG_DUMP) ----------
// DM_FG_DUMP=<start present>:<count>[:<scale %>]: from that present on (g_present_n; 0: once fgdump.req appears in the output folder, looked for every 30 presents),
//   count consecutive presented frames (the frame x with its crop, as frame generation gets it) are read back into one host buffer as they are presented: copied
//   when kept at their size in an 8-bit RGBA / BGRA format, else blitted (scaled to scale %, or converted to RGBA8). After the last one the GPU is waited for once
//   and a thread writes them to <DM_CAPTURE_DIR, else the files folder>/fgdump/NNNN.ppm with info.txt, so the writing does not slow the frames being recorded.
//   The score needs consecutive frames: a present that did not reach the recording (returned early) ends it there. Off: one getenv, then a test of one int a present
static struct { int st, want, n, scale, flip, bgra, conv; long start; uint64_t first, last; uint32_t sw, sh, cw, ch; VkFormat fmt; VkImage img; VkDeviceMemory imem, bmem; VkBuffer buf;
                uint8_t *map; char dir[1024]; } FGD;   // st: 0 not read yet, -1 off, 1 waiting for its start, 2 recording, 3 recorded (written at the next present), 5 writing, 4 written
static void fgd_free(void) {
    if (FGD.buf) vkDestroyBuffer(g_dev, FGD.buf, NULL);
    if (FGD.bmem) vkFreeMemory(g_dev, FGD.bmem, NULL);
    if (FGD.img) vkDestroyImage(g_dev, FGD.img, NULL);
    if (FGD.imem) vkFreeMemory(g_dev, FGD.imem, NULL);
    FGD.buf = VK_NULL_HANDLE; FGD.bmem = FGD.imem = VK_NULL_HANDLE; FGD.img = VK_NULL_HANDLE; FGD.map = NULL;
}
static void *fgd_write(void *arg) {   // (its own thread: the frames are the GPU's no more, and nothing else touches FGD until st 4)
    (void)arg; char p[1200]; snprintf(p, sizeof p, "%s/fgdump", FGD.dir); mkdir(p, 0755);
    for (int i = 0; i < 100000; i++) { snprintf(p, sizeof p, "%s/fgdump/%04d.ppm", FGD.dir, i); if (unlink(p)) break; }   // (a longer recording's frames from before)
    size_t fs = (size_t)FGD.sw * FGD.sh * 4; uint8_t *row = malloc((size_t)FGD.sw * 3); int ok = row != NULL, i = 0;
    for (; i < FGD.n && ok; i++) {
        snprintf(p, sizeof p, "%s/fgdump/%04d.ppm", FGD.dir, i); FILE *f = fopen(p, "wb"); if (!f) { ok = 0; break; }
        fprintf(f, "P6\n%u %u\n255\n", FGD.sw, FGD.sh);
        for (uint32_t y = 0; y < FGD.sh; y++) {   // (rows as a display shows them: a GL-origin frame is stored upside down)
            const uint8_t *s = FGD.map + (size_t)i * fs + (size_t)(FGD.flip ? FGD.sh - 1 - y : y) * FGD.sw * 4;
            for (uint32_t x = 0; x < FGD.sw; x++) { row[x * 3] = s[x * 4 + (FGD.bgra ? 2 : 0)]; row[x * 3 + 1] = s[x * 4 + 1]; row[x * 3 + 2] = s[x * 4 + (FGD.bgra ? 0 : 2)]; }
            if (fwrite(row, 1, (size_t)FGD.sw * 3, f) != (size_t)FGD.sw * 3) ok = 0; }
        if (fclose(f)) ok = 0; }
    free(row);
    snprintf(p, sizeof p, "%s/fgdump/info.txt", FGD.dir); FILE *f = fopen(p, "w");
    if (f) { fprintf(f, "frames %d\nsize %u %u\nsource %u %u\nformat %d\nscale %d\nconverted %d\nfirst present %llu\n", i, FGD.sw, FGD.sh, FGD.cw, FGD.ch, (int)FGD.fmt, FGD.scale, FGD.conv,
                     (unsigned long long)FGD.first); fclose(f); }
    VLOG("%sfg dump: %d frames written to %s/fgdump (%ux%u of the frame's %ux%u, format %d)%s", ok ? "" : "[RED] ", i, FGD.dir, FGD.sw, FGD.sh, FGD.cw, FGD.ch, (int)FGD.fmt,
         ok ? "" : ": writing failed");
    fgd_free(); FGD.st = 4;
    return NULL;
}
// In the present's cb (after its first barrier): x's crop read back when this present is one of the recorded ones; also arms the recording and starts the writing
static void fgd_frame(VkCommandBuffer cb, vtex *x, int32_t cx, int32_t cy, int32_t cw, int32_t ch) {
    if (FGD.st < 0 || FGD.st >= 4) return;
    uint64_t n = atomic_load(&g_present_n);
    if (!FGD.st) {
        const char *e = getenv("DM_FG_DUMP"); long s = 0; int c = 0, sc = 100;
        if (!e || sscanf(e, "%ld:%d:%d", &s, &c, &sc) < 2 || s < 0 || c < 3) { FGD.st = -1; return; }
        FGD.start = s; FGD.want = c > 2000 ? 2000 : c; FGD.scale = sc < 10 ? 10 : sc > 100 ? 100 : sc;
        const char *d = getenv("DM_CAPTURE_DIR"), *t = getenv("DM_TEX_CACHE");   // (the files folder: DM_TEX_CACHE is its texcache/)
        if (d && *d) snprintf(FGD.dir, sizeof FGD.dir, "%s", d);
        else { snprintf(FGD.dir, sizeof FGD.dir, "%s", t ? t : "."); char *sl = t ? strrchr(FGD.dir, '/') : NULL; if (sl) *sl = 0; }
        FGD.st = 1; VLOG("fg dump: %d frames at %d %% from %s, then written to %s/fgdump", FGD.want, FGD.scale, s ? "the present given (DM_FG_DUMP)" : "when fgdump.req appears there", FGD.dir); }
    if (FGD.st == 3) {   // the last recorded frame went to the GPU with the present before: waited for once, then written on a thread of its own
        pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu);
        pthread_t th; FGD.st = 5; if (pthread_create(&th, NULL, fgd_write, NULL)) fgd_write(NULL); else pthread_detach(th);
        return; }
    if (FGD.st == 1) {
        if (FGD.start) { if ((long)n < FGD.start) return; }
        else { if (n % 30) return; char r[1100]; snprintf(r, sizeof r, "%s/fgdump.req", FGD.dir); if (access(r, F_OK)) return; unlink(r); }
        // the first frame: its size and format decide how the frames are read back (the buffer for all of them made now, not while recording)
        VkFormat f = x->fmt; int raw = f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB;
        const char *bad = x->is3d || x->samples > 1 || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || !(x->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? "the frame cannot be copied" : NULL;
        uint32_t sw = (uint32_t)cw, sh = (uint32_t)ch;
        if (FGD.scale < 100) { sw = ((uint32_t)cw * (uint32_t)FGD.scale / 100) & ~1u; sh = ((uint32_t)ch * (uint32_t)FGD.scale / 100) & ~1u; }
        VkFormat imf = raw ? f : VK_FORMAT_R8G8B8A8_UNORM; int blit = !raw || sw != (uint32_t)cw || sh != (uint32_t)ch;
        if (!bad && blit) { VkFormatProperties a, b; vkGetPhysicalDeviceFormatProperties(g_pd, f, &a); vkGetPhysicalDeviceFormatProperties(g_pd, imf, &b);
            if (!(a.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(b.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
                if (raw) { sw = (uint32_t)cw; sh = (uint32_t)ch; blit = 0; VLOG("fg dump: the frame's format cannot be blitted here: recorded at its own size"); }
                else bad = "its format can be neither copied as it is nor blitted"; } }
        if (sw < 16 || sh < 16) bad = "the frame is too small";
        uint64_t fs = (uint64_t)sw * sh * 4; int want = FGD.want;
        if (fs * (uint64_t)want > (1ull << 30)) { want = (int)((1ull << 30) / fs); VLOG("fg dump: %d frames of %ux%u would need over 1 GB: %d frames", FGD.want, sw, sh, want); }
        if (!bad && want < 3) bad = "not even 3 frames fit in 1 GB";
        if (!bad && blit) {
            VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = imf, .extent = { sw, sh, 1 }, .mipLevels = 1, .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
            VkMemoryRequirements rq = { 0 }; int t = -1;
            if (!vkCreateImage(g_dev, &ci, NULL, &FGD.img)) { vkGetImageMemoryRequirements(g_dev, FGD.img, &rq); t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); }
            else FGD.img = VK_NULL_HANDLE;
            VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = (uint32_t)t };
            if (t < 0 || vkAllocateMemory(g_dev, &ai, NULL, &FGD.imem) || vkBindImageMemory(g_dev, FGD.img, FGD.imem, 0)) bad = "no memory for its scaling image"; }
        if (!bad) {
            VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = fs * (uint64_t)want, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
            VkMemoryRequirements rq = { 0 }; int t = -1; void *m = NULL;
            if (!vkCreateBuffer(g_dev, &bci, NULL, &FGD.buf)) { vkGetBufferMemoryRequirements(g_dev, FGD.buf, &rq);   // (cached first: the CPU reads all of it back)
                t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
                if (t < 0) t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0); }
            else FGD.buf = VK_NULL_HANDLE;
            VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = (uint32_t)t };
            if (t < 0 || vkAllocateMemory(g_dev, &ai, NULL, &FGD.bmem) || vkBindBufferMemory(g_dev, FGD.buf, FGD.bmem, 0) || vkMapMemory(g_dev, FGD.bmem, 0, VK_WHOLE_SIZE, 0, &m))
                bad = "no host memory for the frames";
            FGD.map = m; }
        if (bad) { VLOG("[RED] fg dump: not recorded (%s; %ux%u, format %d)", bad, (unsigned)cw, (unsigned)ch, (int)f); fgd_free(); FGD.st = -1; return; }
        FGD.want = want; FGD.n = 0; FGD.sw = sw; FGD.sh = sh; FGD.cw = (uint32_t)cw; FGD.ch = (uint32_t)ch; FGD.fmt = f; FGD.first = n; FGD.last = n - 1;
        FGD.flip = atomic_load(&g_origin_ll); FGD.bgra = f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB; FGD.conv = !raw; FGD.st = 2;
        VLOG("fg dump: recording %d frames from present %llu (%ux%u, format %d, %s)", want, (unsigned long long)n, sw, sh, (int)f, blit ? "blitted" : "copied");
        if (FGD.img) { VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = FGD.img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); } }
    // recording: this present's frame, unless the frame changed (size, format) or a present was missed, which ends the recording with the frames so far
    if (n != FGD.last + 1 || x->fmt != FGD.fmt || (uint32_t)cw != FGD.cw || (uint32_t)ch != FGD.ch) {
        VLOG("fg dump: ended after %d frames (%s)", FGD.n, n != FGD.last + 1 ? "a present was missed" : "the frame's size or format changed");
        if (FGD.n >= 3) FGD.st = 3; else { fgd_free(); FGD.st = -1; }
        return; }
    VkDeviceSize off = (VkDeviceSize)FGD.n * FGD.sw * FGD.sh * 4;
    if (!FGD.img) { VkBufferImageCopy bc = { off, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { cx, cy, 0 }, { (uint32_t)cw, (uint32_t)ch, 1 } };
        vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, FGD.buf, 1, &bc); }
    else {   // (the full barrier before this in cb orders the blit after the last frame's copy out of the image)
        VkImageBlit b = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { cx, cy, 0 }, { cx + cw, cy + ch, 1 } }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { 0, 0, 0 }, { (int32_t)FGD.sw, (int32_t)FGD.sh, 1 } } };
        vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, FGD.img, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        VkBufferImageCopy bc = { off, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { FGD.sw, FGD.sh, 1 } };
        vkCmdCopyImageToBuffer(cb, FGD.img, VK_IMAGE_LAYOUT_GENERAL, FGD.buf, 1, &bc); }
    VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };   // (read by the CPU later)
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    FGD.last = n; if (++FGD.n >= FGD.want) FGD.st = 3;
}
static void present_window_locked(vtex *x, const int crop[4]);
static void present_window(vtex *x, const int crop[4]) { pthread_mutex_lock(&g_wmu); present_window_locked(x, crop); pthread_mutex_unlock(&g_wmu); }
// The window is going away (android_main.c APP_CMD_TERM_WINDOW): its swapchain and surface are released before that returns, as Android expects
void nvn_vk_window_term(void) { if (!g_dev) return; pthread_mutex_lock(&g_wmu); fg_flush_locked(); swap_destroy(); pthread_mutex_unlock(&g_wmu); }
// One swapchain image composed in cb: black, the picture scaled into its box (the upscaler, else the plain linear stretch), the touch overlay; left ready to present
typedef struct { vtex *x; VkImage img; VkImageView view; uint32_t w, h; int32_t cx, cy, cw, ch; } pres_src;   // x: the game's frame; NULL: a picture of ours (img / view / w / h, GENERAL layout)
// In one pass (fix 8 of perf-1124; DM_VK_COMPOSE_ONE_PASS=0: the three steps below): the image cleared black by the pass's load op, the picture drawn into its box by the
//   upscaler (the plain stretch: the sharp upscaler at sharpness 0, plain bilinear as the blit's linear filter), the controls over it, one store. Before, the whole
//   image was cleared (vkCmdClearColorImage), then loaded and stored by the upscaler's pass and again by the overlay's: on a tiler one full-screen clear and two
//   full-screen load / store cycles a picture, at the panel's size (2400x1080 on #349's Adreno 613, whose game draws 640x360). 0: not possible here (no push
//   descriptors for the upscaler, a picture that cannot be sampled): the caller composes as before
static int g_compose_one = -1;   // (t_vkdev sets it to compare the two)
static int compose_one_on(void) { int v = g_compose_one; if (v < 0) { const char *e = getenv("DM_VK_COMPOSE_ONE_PASS"); g_compose_one = v = !(e && !strcmp(e, "0")); if (v) VLOG("present: each picture composed in one pass (DM_VK_COMPOSE_ONE_PASS=0: clear, upscale, overlay as three)"); } return v; }
static int present_compose_one(VkCommandBuffer cb, uint32_t ii, const pres_src *s, int flip) {
    if (!compose_one_on() || !W.view[ii]) return 0;
    VkImageView sv = s->view; uint32_t tw = s->w, th = s->h;
    if (s->x) { if (!(s->x->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || s->x->is3d) return 0;
        sv = view_get(s->x, VK_IMAGE_VIEW_TYPE_2D, s->x->fmt, 0, 1, 0, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT); tw = s->x->w; th = s->x->h; }
    if (!sv || !tw || !th || !scale_setup(W.fmt, 0)) return 0;   // (the sampler and layout every upscaler uses, and the sharp one: the plain stretch's here)
    int32_t cx = s->cx, cy = s->cy, cw = s->cw, ch = s->ch;
    int32_t box[4]; dm_picture_box((int32_t)W.lext.width, (int32_t)W.lext.height, cw, ch, box);
    int mode = scaler_mode(); float sh = scale_sharpness(); int up = box[2] > cw && box[3] > ch;
    int plain = !W.rot && !color_mode() && (mode == 2 || (sh < 0 && mode != 3) || !up);   // (where the three-step compose blits)
    int which = !plain && mode == 1 && !color_mode() ? 1 : 0; if (which && !scale_setup(W.fmt, 1)) which = 0;   // (SGSR has no colour: with one chosen, the sharp upscaler)
    // the picture made visible to the fragment shader that samples it (the game's frame: its passes, copies; frame generation's: its render or compute passes)
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    int fsr = !plain && mode == 3 && up && fsr_easu(cb, sv, tw, th, cx, cy, cw, ch, flip, box);   // (FSR 1's first half into its own image, before the pass; else the sharp upscaler)
    if (plain || (!fsr && (mode == 2 || sh < 0))) sh = 0;   // (the plain stretch, or turned but smooth: plain bilinear; FSR keeps the setting, as fsr_draw)
    dm_ovl_item it[128]; int n = ovl_prep(cb, W.fmt, W.lext, it, 128);   // (the label atlas uploaded outside the pass)
    // (from COLOR_ATTACHMENT_OUTPUT: the stage the image's acquire semaphore is waited at, so the transition follows it)
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = W.view[ii], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue.color = {{ 0, 0, 0, 1 }} };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, W.ext }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    { uint32_t cf = (uint32_t)W.fmt; rend_begin(cb, &ri, &cf, 0, 1, 1); g_cmp_passes++; }
    if (fsr) fsr_rcas_rec(cb, box, sh); else scale_rec(cb, which, sv, tw, th, cx, cy, cw, ch, flip, box, sh);
    if (n > 0) ovl_rec(cb, W.ext, W.lext, W.rot, it, n);
    rend_end(cb);
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = 0; ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    static int said; if (!said++) VLOG("present: composed in one pass by the %s (%dx%d → %dx%d%s)", fsr ? "FSR 1 upscaler" : plain ? "plain stretch" : which ? "SGSR upscaler" : "sharp upscaler",
                                       cw, ch, box[2], box[3], W.rot ? ", pre-rotated" : "");
    g_cmp_one++;
    return 1;
}
static void present_compose(VkCommandBuffer cb, uint32_t ii, const pres_src *s, int flip) {
    if (ii >= W.n) { RED_ONCE(115, "swapchain image %u handed out, %u known: not drawn", ii, W.n); return; }
    g_cmp_n++;
    if (present_compose_one(cb, ii, s, flip)) return;
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cb, W.img[ii], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr); g_cmp_clears++;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    int32_t cx = s->cx, cy = s->cy, cw = s->cw, ch = s->ch;
    int32_t box[4]; dm_picture_box((int32_t)W.lext.width, (int32_t)W.lext.height, cw, ch, box); int32_t dx = box[0], dy = box[1], dw = box[2], dh = box[3];
    if (!(s->x ? scale_draw(cb, s->x, cx, cy, cw, ch, flip, ii, box) : scale_draw_view(cb, s->view, s->w, s->h, cx, cy, cw, ch, flip, ii, box))) {   // the upscaler, else the plain linear stretch
        VkImageBlit bl = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{cx, flip ? cy + ch : cy, 0}, {cx + cw, flip ? cy : cy + ch, 1}}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{dx, dy, 0}, {dx + dw, dy + dh, 1}} };
        vkCmdBlitImage(cb, s->x ? s->x->img : s->img, VK_IMAGE_LAYOUT_GENERAL, W.img[ii], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    }
    if (!ovl_draw_turned(cb, W.img[ii], W.view[ii], W.ext, W.lext, W.rot, W.fmt, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)) {
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = 0; ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
}
static void present_window_locked(vtex *x, const int crop[4]) {
    g_gfx_bind_gen++;   // (its own command buffers, but other pipelines and layouts: draw_'s state shadow trusts nothing it shares with them)
    fg_flush_locked();   // (a real frame still waiting behind its in-between picture: out now, before this frame's)
    void *win = front_metal_layer();
    if (!win) { if (W.sc) swap_destroy(); return; }
    if (!g_has_swapchain || !x || !x->img || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || x->samples > 1) return;
    int srgb = x->fmt == VK_FORMAT_R8G8B8A8_SRGB || x->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    if (!swap_ensure(win, srgb)) return;
    uint64_t t_arr = mono_ns(), period = game_period_ns();
    if (framegen_mode()) g_fg_period = period = fg_period(t_arr);   // (frame generation: the game's frame time as measured, so a game short of its target still gets pictures)
    int32_t cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int32_t)x->w, ch = crop && crop[3] ? crop[3] : (int32_t)x->h;
    uint32_t sl = W.slot++ % 3;
    vk_wait(NULL, W.fence[sl], UINT64_MAX);
    if (FGE.q_live[sl]) {   // the GPU's time for the in-between picture this slot recorded last (its fence has signalled)
        uint64_t t[2]; FGE.q_live[sl] = 0;
        if (vkGetQueryPoolResults(g_dev, FGE.qp, sl * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] >= t[0]) {
            uint64_t ns = (uint64_t)((double)(t[1] - t[0]) * FGE.tsper); FGE.gpu_sum += ns; FGE.gpu_n++; if (ns > FGE.gpu_max) FGE.gpu_max = ns;
            { uint64_t e = atomic_load(&g_fg_gpu_ema); atomic_store(&g_fg_gpu_ema, e ? (e * 15 + ns) / 16 : ns); }
            int np = FGE.q_pics[sl] > 0 ? FGE.q_pics[sl] : 1; uint64_t lim = FGE.budget + (uint64_t)(np - 1) * FGE.budget / 2;   // (each picture after the first: half the budget more)
            if (!FGE.off && fg_plan_cost(&FGE.plan, ns, t_arr, lim)) {
                char why[96], n[24]; snprintf(why, sizeof why, "the GPU needs over %.1f ms for %d picture%s, %.1f ms the last frame", lim / 1e6, np, np > 1 ? "s" : "", ns / 1e6);
                snprintf(n, sizeof n, "slow:%.1f", ns / 1e6);
                // (working at the 3D render size, a lower scale makes the pictures cheaper too: the scale gives way first)
                if (!(FGE.ls.w && FGE.ls.w < FGE.ls.ow && fg_gov_absorb(why, t_arr))) fg_step_down(why, n); } } }
    {   // this slot's cut / still numbers, now its fence has signalled (three presents ago): two still pairs in a row leave the next pictures out
        fgc_result cr; float mean = 0; static float still = -1;
        if (still < 0) { const char *e = getenv("DM_FG_STILL"); still = e && atof(e) >= 0 ? (float)atof(e) : 0.0015f; }
        if (fgc_read(&FGE.cut, (int)sl, &cr, &mean)) { FGE.still_n = mean < still ? FGE.still_n + 1 : 0; if (cr.flag) FGE.n_cut++; } }
    int fgm = framegen_mode();
#if DM_GAME_PROGRAM_ID == 0x0100A3D008C5C000ull || DM_GAME_PROGRAM_ID == 0x01008F6008C5E000ull
    // Scarlet / Violet: off until the game holds a steady 30 (pictures between uneven frames look worse than none); DM_FG_PALDEA=1 tries it anyway
    { static int ok = -1; if (ok < 0) { const char *e = getenv("DM_FG_PALDEA"); ok = e && !strcmp(e, "1"); if (fgm && !ok) VLOG("frame generation: off for Scarlet / Violet until the game holds a steady 30 (DM_FG_PALDEA=1 tries it)"); }
      if (!ok) fgm = 0; }
#endif
    if (fgm != FGE.mode_seen) {   // switched in the menu (off and on again, or the other kind): a GPU-time cut-off gets a fresh try, measured and cut off again the same way
        if (fgm && FGE.off && FGE.slow) { FGE.off = FGE.slow = 0; FGE.note[0] = 0; memset(FGE.q_live, 0, sizeof FGE.q_live);
            VLOG("frame generation switched on again in the menu: tried again from its full setting, made lighter again step by step if it costs too much"); }
        if (fgm && FGE.up) FGE.mult_cap = 4;
        FGE.flow_cut = 0; FGE.ok_win = 0; FGE.since_up = 99; FGE.up_wait = 20; FGE.base_fps = 0; FGE.aw_t = 0; FGE.aw_skip = 0; memset(&FGE.plan, 0, sizeof FGE.plan);
        FGE.mode_seen = fgm; FGE.held = 1; FGE.rate_n = 0; FGE.rate_ivl = 0; FGE.rate_last = 0; }
    if (fgm) {   // Frame generation runs while the game holds its own frame rate: DM_FRAMEGEN_MIN_FPS (default the target less 2: 28 at 30, 58 at 60) or more, paused
        //   under it, back from 2 over it. Under its target the in-between pictures cost the game frames it could not spare (#189, Adreno 613: the game at 14-22 fps,
        //   4x of it moved worse). Time rule: paused after 2 s under, back after 5 s over (no flicking on and off at the edge); when its own cost took the game under,
        //   the render scale gives way first (fg_gov_absorb: no pause for 15 s while the guard steps). Decided after 16 frames; until then held
        if (!FGE.min_fps) { const char *e = getenv("DM_FRAMEGEN_MIN_FPS"); int f = e ? atoi(e) : 0; FGE.min_fps = f >= 1 && f <= 120 ? f : game_fps() - 2 >= 10 ? game_fps() - 2 : 28; }
        uint64_t d = FGE.rate_last ? t_arr - FGE.rate_last : 0; FGE.rate_last = t_arr;
        if (d && d < 1000000000ull) { FGE.rate_ivl = FGE.rate_ivl ? (FGE.rate_ivl * 15 + d) / 16 : d; FGE.rate_n++; }
        if (FGE.rate_n >= 16) {
            uint64_t lo = 1000000000ull / (uint64_t)(FGE.min_fps > 1 ? FGE.min_fps - 1 : 1), hi = 1000000000ull / (uint64_t)(FGE.min_fps + 1);   // (under min-1 fps: paused; min+1 or more: on)
            int was = FGE.held, want = FGE.rate_ivl > lo ? 1 : FGE.rate_ivl < hi ? 0 : was; double fps = 1e9 / (double)FGE.rate_ivl;
            if (want != was) {
                if (FGE.flip_to != want + 1) { FGE.flip_to = want + 1; FGE.flip_t = t_arr; }
                if (t_arr - FGE.flip_t < (want ? 2000000000ull : 5000000000ull) || (want && t_arr < FGE.grace_t)) want = was; }
            else FGE.flip_to = 0;
            int now = want;
            if (now && !was && !FGE.off && FGE.last_gen_t && t_arr - FGE.last_gen_t < 3000000000ull) {   // (it was making pictures: its cost took the game under the floor)
                char w[96], n[24]; snprintf(w, sizeof w, "the game fell to %.1f fps with it", fps); snprintf(n, sizeof n, "cost:%.0f", fps);
                if (fg_gov_absorb(w, t_arr)) now = was; else fg_step_down(w, n); }
            if (now != was) {
                FGE.held = now; FGE.flip_to = 0;
                if (now) VLOG("[RED] frame generation paused: the game itself runs at %.1f fps, under the %d fps it needs (DM_FRAMEGEN_MIN_FPS): normal presentation until it holds %d for 5 s", fps, FGE.min_fps - 1, FGE.min_fps + 1);
                else if (FGE.n_gen + FGE.n_plain || FGE.last_log) VLOG("frame generation back on: the game runs at %.1f fps", fps); }
            if (!FGE.off && strcmp(FGE.note, "ours")) { if (FGE.held) snprintf(FGE.note, sizeof FGE.note, "paused:%.0f", fps); else FGE.note[0] = 0; } } }
    // the work size: Lossless Scaling HUD-free only (its pictures go back to the frame's size in its HUD pass; without one, the HUD would be drawn at the smaller size)
    int hf_can = g_hp_scene_ok && g_hp_simg && cx == 0 && cy == 0 && g_hp_sw == (uint32_t)cw && g_hp_sh == (uint32_t)ch && g_hp_sfmt == x->fmt;
    float wsc = fgm == 3 && hf_can && FGE.ls.hbase ? fg_work_scale(t_arr) : FGE.ls.w && FGE.ls.w < FGE.ls.ow && fgm == 3 ? FGE.ws : 1.0f;
    int fg = fgm && !FGE.off && !FGE.held && FGE.base_fps > 0 && fg_ready(x, cx, cy, cw, ch, wsc), gen = 0;   // (the first 3 s: the game's own rate measured, fg_adapt)
    fgm = framegen_mode() && fgm ? framegen_mode() : fgm;   // (fg_ready may have fallen back from Lossless Scaling to ours)
    if (!fg) { FGE.have_prev = 0; FGE.ls.n = 0; }   // (the copy of the frame before goes stale the moment a frame is not copied; Lossless Scaling's history starts again)
    // HUD-free: the frames it works from are the scene copies, the HUD put back from the frame as shown (nvn_fgen.h hud, nvn_lsfg.h ls_hud). A switch between the two
    //   kinds: the frames before are of the other kind, no picture from them (decided before the swapchain images are asked for)
    int hf = fg && hf_can && (fgm != 3 || ls_hud_ok(&FGE.ls));
    if (hf) FGE.hf_ever = 1;
    if (fg && hf != FGE.hf_last) { FGE.have_prev = 0; FGE.ls.n = 0; }
    FGE.hf_last = hf;
    if (fg) { FGE.g.debug = fgm == 2; gen = fg_plan_frame(&FGE.plan, t_arr, period) && FGE.have_prev && (fgm != 3 || FGE.ls.n >= 2); }   // (lsfg: three frames of history first)
    // left out: a still pair (two read-backs in a row, nvn_fgcut.h) and a frame without the scene once HUD-free has worked (a full-screen menu, a loading screen:
    //   DM_FG_MENU_SKIP=0 keeps them)
    if (gen && FGE.still_n >= 2) { gen = 0; FGE.n_still++; }
    { static int ms = -1; if (ms < 0) { const char *e = getenv("DM_FG_MENU_SKIP"); ms = !(e && !strcmp(e, "0")); }
      if (gen && ms && FGE.hf_ever && !hf && hudfree_on()) { gen = 0; FGE.n_menu++; } }
    uint32_t ii, imgs[LS_MAXGEN + 1]; int npic = 0; VkResult r = vkAcquireNextImageKHR(g_dev, W.sc, UINT64_MAX, W.acq[sl], VK_NULL_HANDLE, &ii);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { swap_destroy(); return; }
    if (r && r != VK_SUBOPTIMAL_KHR) return;
    imgs[0] = ii;
    if (gen) {   // an image more for each picture after the first and for the real frame; none free within 4 ms: fewer pictures (evenly spaced still), or the frame alone
        int want = fgm == 3 ? lsfg_mult() - 1 : 1;
        for (int k = 0; k < want; k++) { VkResult r2 = vkAcquireNextImageKHR(g_dev, W.sc, 4000000ull, FGE.acqx[sl][k], VK_NULL_HANDLE, &imgs[1 + k]);
            if (r2 != VK_SUCCESS && r2 != VK_SUBOPTIMAL_KHR) break;
            npic++; }
        gen = npic > 0; }
    vkResetFences(g_dev, 1, &W.fence[sl]);
    VkCommandBuffer cb = W.cb[sl]; vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT }; vkBeginCommandBuffer(cb, &bi);
    barrier_all(cb);
    fgd_frame(cb, x, cx, cy, cw, ch);   // (DM_FG_DUMP: real frames for the PC score; nothing while off)
    int flip = atomic_load(&g_origin_ll);
    pres_src real = { x, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0, cx, cy, cw, ch };
    if (fg) {   // the frame copied (the game draws over its own next), its brightness pyramid, and with a frame before it the picture between the two
        fgen *g = &FGE.g; lsfg *ls = fgm == 3 ? &FGE.ls : NULL; int p = FGE.p ^= 1; uint64_t c0 = mono_ns();
        if (FGE.qp && gen) { vkCmdResetQueryPool(cb, FGE.qp, sl * 2, 2); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, FGE.qp, sl * 2); }
        if (ls) ls_begin(ls, cb); else fg_begin(g, cb);
        VkImageCopy ic = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { cx, cy, 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { (uint32_t)cw, (uint32_t)ch, 1 } };
        if (!ls) g->hud_on = hf;
        if (ls && ls->w != (uint32_t)cw) {   // work size: the frame (or its scene) scaled down into c[p]; HUD-free: the scene also kept at the frame's size for the HUD rule
            VkImageBlit b = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { hf ? 0 : cx, hf ? 0 : cy, 0 }, { (hf ? 0 : cx) + cw, (hf ? 0 : cy) + ch, 1 } },
                              { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { 0, 0, 0 }, { (int32_t)ls->w, (int32_t)ls->h, 1 } } };
            vkCmdBlitImage(cb, hf ? g_hp_simg : x->img, VK_IMAGE_LAYOUT_GENERAL, ls->c[p].img, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
            if (hf) { VkImageCopy sc = ic; sc.srcOffset = (VkOffset3D){ 0, 0, 0 };
                vkCmdCopyImage(cb, g_hp_simg, VK_IMAGE_LAYOUT_GENERAL, ls->sfull.img, VK_IMAGE_LAYOUT_GENERAL, 1, &sc);
                vkCmdCopyImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, ls->hud.img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic); } }
        else if (hf) { VkImageCopy sc = ic; sc.srcOffset = (VkOffset3D){ 0, 0, 0 };
            vkCmdCopyImage(cb, g_hp_simg, VK_IMAGE_LAYOUT_GENERAL, ls ? ls->c[p].img : g->c[p].img, VK_IMAGE_LAYOUT_GENERAL, 1, &sc);
            vkCmdCopyImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, ls ? ls->hud.img : g->hud.img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic); }
        else vkCmdCopyImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, ls ? ls->c[p].img : g->c[p].img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        if (ls) {   // (its own barriers: the copy is made visible to its compute passes)
            ls_ingest(ls, cb, p);
            if (FGE.have_prev) fgc_stat(&FGE.cut, cb, (int)sl, ls->c[p ^ 1].view, ls->c[p].view);   // (the two frames the pictures go between; read back for stills)
            if (gen) { for (int k = 0; k < npic; k++) ls_generate(ls, cb, npic, k);
                for (int k = 0; k < npic; k++) fgc_select(&FGE.cut, cb, (int)sl, k, ls->c[p].view, ls->m[k].view, ls->w, ls->h);   // (a cut: the newer frame instead)
                if (FGE.qp) { vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, FGE.qp, sl * 2 + 1); FGE.q_live[sl] = 1; FGE.q_pics[sl] = npic; }
                for (int k = 0; k < npic; k++) { ls_img *e = &ls->m[k];
                    if (hf) { ls_hud(ls, cb, k); e = &ls->o; }   // (its HUD back on top; o is drawn here before the next picture's pass writes it)
                    uint32_t ew = e == &ls->o ? ls->ow : ls->w, eh = e == &ls->o ? ls->oh : ls->h;   // (the work size: m is smaller than the frame, o is the frame's)
                    pres_src mid = { NULL, e->img, e->pview, ew, eh, 0, 0, (int32_t)ew, (int32_t)eh }; present_compose(cb, imgs[k], &mid, flip); } }
        } else {
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        fg_ingest(g, cb, p);
        if (FGE.have_prev) fgc_stat(&FGE.cut, cb, (int)sl, g->c[p ^ 1].view, g->c[p].view);   // (ours: its picture is a render target: fgc_select_rt below)
        {   // depth guidance: this frame's scene depth (copied with the scene) → its nearness. A new copy image: named to frame generation once nothing in flight uses the old one
            int zgen = g_hp_zimg ? g_hp_zgen : 0;
            if (FGE.zgen != zgen) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu);
                int ok = fg_depth_src(g, g_hp_zimg, g_hp_zfmt, g_hp_zfar1); FGE.zgen = zgen;
                if (g_hp_zimg) VLOG("frame generation, depth guidance: %s (%ux%u, format %d)", ok ? "on" : "the driver cannot sample that depth format: off", g_hp_zw, g_hp_zh, (int)g_hp_zfmt); }
            g->z_on = fg_depth_on(); g->z_far1 = g_hp_zfar1;
            fg_depth_ingest(g, cb, p, hf && g_hp_zimg && g_hp_zseq == g_hp_scene_seq && g_hp_Z); }
        {   // camera guidance: the game's own view-projection for this frame and the one before (dm_cam_viewproj, where the game's camera is known)
            static float vp[2][16]; static int vok[2], said;
            vok[p] = fg_cam_on() && dm_cam_viewproj && dm_cam_viewproj(vp[p]);
            float m[16]; int ok = gen && vok[p] && vok[p ^ 1] && fg_cam_m(vp[p ^ 1], vp[p], m);
            fg_camera(g, p, ok ? m : NULL, atomic_load(&g_depth_zero));
            if (ok && g->camb[0] > 0.5f && !said) { said = 1; VLOG("frame generation, camera guidance: on (the game's camera and depth start the motion search; DM_FG_CAMERA=0 off)"); }
            static unsigned n, nread, nz; n++; nread += vok[p] != 0; nz += g->z_on && g->z_ok[p];   // (why not, once a while: the camera read or the depth)
            if (n == 300 && !said) VLOG("frame generation, camera guidance: not on yet: camera read %u of %u frames%s, depth %u of %u", nread, n,
                                        !dm_cam_viewproj ? " (no reader for this game)" : "", nz, n);
            if (n >= 300) n = nread = nz = 0; }
        if (gen) { fg_generate(g, cb, p);
            fgc_select_rt(&FGE.cut, cb, (int)sl, hf ? g->hud.view : g->c[p].view, g->m.view, g->fmt, g->w, g->h);   // (a cut: the newer frame as shown instead)
            if (FGE.qp) { vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, FGE.qp, sl * 2 + 1); FGE.q_live[sl] = 1; FGE.q_pics[sl] = 1; }
            pres_src mid = { NULL, g->m.img, g->m.view, g->w, g->h, 0, 0, cw, ch };
            present_compose(cb, ii, &mid, flip); }
        }
        FGE.have_prev = 1; FGE.cpu_sum += mono_ns() - c0;
    }
    if (!gen) npic = 0;
    present_compose(cb, imgs[npic], &real, flip);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws[LS_MAXGEN + 1]; VkSemaphore wsem[LS_MAXGEN + 1] = { W.acq[sl] }, ssem[LS_MAXGEN + 1] = { done_img(imgs[0]) };   // (ssem[j]: image imgs[j]'s present)
    for (int k = 0; k <= LS_MAXGEN; k++) ws[k] = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    for (int k = 0; k < npic; k++) { wsem[1 + k] = FGE.acqx[sl][k]; ssem[1 + k] = done_img(imgs[1 + k]); }
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = (uint32_t)(1 + npic), .pWaitSemaphores = wsem, .pWaitDstStageMask = ws, .commandBufferCount = 1, .pCommandBuffers = &cb,
                        .signalSemaphoreCount = (uint32_t)(1 + npic), .pSignalSemaphores = ssem };
    VkPresentInfoKHR pi = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &ssem[0], .swapchainCount = 1, .pSwapchains = &W.sc, .pImageIndices = &ii };
    VkPresentTimeGOOGLE pt = { 0 }; VkPresentTimesInfoGOOGLE pti = { .sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, .swapchainCount = 1, .pTimes = &pt };
    g_fg_pace = gen; g_fg_mult = npic + 1; if (pace_time(&pt)) pi.pNext = &pti; g_fg_pace = 0;   // (a time asked for, nothing waited on: the present mode stays as it is)
    pthread_mutex_lock(&g_qmu);
    VkResult rs = r = vkQueueSubmit(g_q, 1, &si, W.fence[sl]);
    if (!r) r = vkQueuePresentKHR(g_q, &pi);
    pthread_mutex_unlock(&g_qmu);
    if (gen && !rs) {   // the pictures after the first and the real frame: handed over by fg_thread, each an interval / (npic + 1) after the one before
        uint64_t now = mono_ns(), sp = period / (uint64_t)(npic + 1), rf = pace_off() ? 0 : g_refresh_ns, want = pt.desiredPresentTime;
        FGE.pending = npic; FGE.phead = 0; FGE.pmult = npic + 1; FGE.arrived = t_arr; FGE.seq++;
        for (int k = 1; k <= npic; k++) {
            FGE.pii[k - 1] = imgs[k]; FGE.pdone[k - 1] = done_img(imgs[k]);
            // with a display time asked: each handed over early hundredths of a refresh before the time of the picture ahead of it (it asks its own an interval / n later);
            // without: at its own time
            uint64_t at = want + (uint64_t)(k - 1) * sp, ea = rf * (uint64_t)g_fg_early / 100;
            FGE.pdue[k - 1] = want && rf ? (at > now + ea ? at - ea : now) : now + (uint64_t)k * sp; }
        FGE.n_gen += (uint64_t)npic; sem_post(&FGE.wake); }
    else if (fg) FGE.n_plain++;
    if (fgm) fg_adapt(t_arr, gen && !rs);
    if (fgm && t_arr - FGE.last_log >= 10000000000ull) {
        if (FGE.last_log && FGE.n_gen + FGE.n_plain)
            VLOG("frame generation: %llu in-between pictures in the last 10 s, %llu frames without one; GPU %.2f ms a picture (longest %.2f), recorded in %.2f ms; "
                 "the real frame reaches the display %.1f ms after the game finished it (the input delay it adds); camera cuts %llu, left out still %llu / menu %llu", (unsigned long long)FGE.n_gen, (unsigned long long)FGE.n_plain,
                 FGE.gpu_n ? FGE.gpu_sum / 1e6 / (double)FGE.gpu_n : 0.0, FGE.gpu_max / 1e6, FGE.n_gen + FGE.n_plain ? FGE.cpu_sum / 1e6 / (double)(FGE.n_gen + FGE.n_plain) : 0.0,
                 FGE.lag_n ? FGE.lag_sum / 1e6 / (double)FGE.lag_n : 0.0, (unsigned long long)FGE.n_cut, (unsigned long long)FGE.n_still, (unsigned long long)FGE.n_menu);
        FGE.n_gen = FGE.n_plain = FGE.gpu_sum = FGE.gpu_n = FGE.gpu_max = FGE.cpu_sum = FGE.lag_sum = FGE.lag_n = 0; FGE.n_cut = FGE.n_still = FGE.n_menu = 0; FGE.last_log = t_arr; }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { fg_flush_locked(); swap_destroy(); }
}
// What the driver holds for our shaders and pipelines (perf-1124 fix 7: counted before anything was released): the modules alive, their SPIR-V, those released and
//   made again (shmod_release), the pipelines kept by kind; the host heap beside it (Android) shows what releasing gave back on this driver
static void shmod_report(long heap) {
    pthread_mutex_lock(&g_psomu); uint32_t pg = g_pso.n, ps = g_pso_s.n; pthread_mutex_unlock(&g_psomu);
    VLOG("[mem] shader modules %lld alive (%.1f MB of SPIR-V handed to the driver), %llu released so far (%.1f MB), %llu made again; pipelines kept %u (with draw state %u), compute %u, "
         "%llu destroyed over the cap; host heap %.0f MB", (long long)atomic_load(&g_shmod_n), atomic_load(&g_shmod_bytes) / 1048576.0, (unsigned long long)atomic_load(&g_shmod_rel_n),
         atomic_load(&g_shmod_rel_bytes) / 1048576.0, (unsigned long long)atomic_load(&g_shmod_remade), pg, ps, g_cps.n, (unsigned long long)g_pso_evicted, heap / 1048576.0);
}
// The pool census (fix 5 of the 2026-11-24 memory analysis asked for it before any change: which pools the GPU needs as buffers): the live pools by what was done with
//   them, ever bound as a buffer (vertex, index, uniform, storage, indirect, copies: pool_buf), only read as texture data (tex_upload) or neither, and by the game's
//   NVN pool flags; how many are on GPU memory. One line, at the second memory readout (about 2 minutes in) and every 10th after it
static void pool_census(void) {
    enum { NG = 12 }; struct { int flags, n; double b, buf, tex, none; } g[NG]; int ng = 0, other = 0;
    int n = 0, nbuf = 0, ntex = 0, nnone = 0, nback = 0; double all = 0, bbuf = 0, btex = 0, bnone = 0, bback = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) { const pool_ent *e = &g_pools[i]; double sz = (double)e->size; n++; all += sz;
        int k = 0; while (k < ng && g[k].flags != e->flags) k++;
        if (k == ng && ng < NG) { g[ng].flags = e->flags; g[ng].n = 0; g[ng].b = g[ng].buf = g[ng].tex = g[ng].none = 0; ng++; }
        if (k == ng) other++;
        if (k < ng) { g[k].n++; g[k].b += sz; }
        if (e->used_buf) { nbuf++; bbuf += sz; if (k < ng) g[k].buf += sz; }
        else if (e->used_tex) { ntex++; btex += sz; if (k < ng) g[k].tex += sz; }
        else { nnone++; bnone += sz; if (k < ng) g[k].none += sz; }
        if (e->backed) { nback++; bback += sz; } }
    pthread_mutex_unlock(&g_mu);
    for (int a = 1; a < ng; a++) for (int b = a; b > 0 && g[b].b > g[b - 1].b; b--) { __typeof__(g[0]) t = g[b]; g[b] = g[b - 1]; g[b - 1] = t; }   // largest first
    char fl[1024]; int o = 0; const double M = 1048576.0;
    for (int k = 0; k < ng && o < (int)sizeof fl - 120; k++)
        o += snprintf(fl + o, sizeof fl - (size_t)o, "%s%#x: %d, %.0f MB (buffers %.0f, textures only %.0f, neither %.0f)", k ? "; " : "", g[k].flags, g[k].n, g[k].b / M, g[k].buf / M, g[k].tex / M, g[k].none / M);
    if (other) snprintf(fl + o, sizeof fl - (size_t)o, "; %d more with other flags", other);
    VLOG("[mem] pool census: %d pools, %.0f MB: read as buffers %d (%.0f MB), texture data only %d (%.0f MB), neither %d (%.0f MB); on GPU memory %d (%.0f MB; %llu backed late, %.0f MB, %.0f ms in all; %s); by flags %s",
         n, all / M, nbuf, bbuf / M, ntex, btex / M, nnone, bnone / M, nback, bback / M, (unsigned long long)atomic_load(&g_pool_late_n), atomic_load(&g_pool_late_bytes) / M,
         atomic_load(&g_pool_late_ns) / 1e6, pool_lazy() ? "only pools read as buffers" : "every pool", fl);
}
// Once a minute: where the renderer's memory is, so growth can be told apart (caches that level off: shaders, pipelines; anything else that keeps rising is a leak)
static void mem_report(void) {
    { static int nrep; nrep++; if (nrep == 2 || (nrep > 2 && (nrep - 2) % 10 == 0)) pool_census(); }
    pthread_mutex_lock(&g_amu); double dev = 0, devused = 0; int nb = g_nblocks;
    for (int i = 0; i < g_nblocks; i++) { dev += g_blocks[i]->size; double fr = 0; for (int k = 0; k < g_blocks[i]->nfree; k++) fr += g_blocks[i]->fr[k].size; devused += g_blocks[i]->size - fr; }
    pthread_mutex_unlock(&g_amu);
    double arena = 0; int ndp = 0; for (int i = 0; i < NSUB; i++) { for (int k = 0; k < g_subs[i].nar; k++) arena += g_subs[i].ar[k].size; ndp += g_subs[i].ndp; }
    pthread_mutex_lock(&g_dfmu); int ndf = g_ndfr; pthread_mutex_unlock(&g_dfmu);
    pthread_mutex_lock(&g_mu); int npool = g_npool; double pool = 0; for (int i = 0; i < g_npool; i++) pool += g_pools[i].size; pthread_mutex_unlock(&g_mu);
    long heap = -1;
#ifdef __ANDROID__
    { struct mallinfo mi = mallinfo(); heap = (long)mi.uordblks; }
#endif
    if (g_drop_n) VLOG("[mem] texture quality half: %llu textures uploaded without their largest level so far, about %.0f MB of game texture data left out",
                       (unsigned long long)g_drop_n, g_drop_saved / 1048576.0);
    VLOG("[mem] device blocks %d: %.0f MB (%.0f MB in use); textures %ld live, %.0f MB; pools %d, %.0f MB; arenas %.0f MB; descriptor pools %d; deferred %d; "
         "shaders %u, pipelines %u, textures by serial %u, aliases %u, ghosts %u; host heap %.0f MB; SPIR-V %.0f MB in the driver's modules, %.0f MB kept by us",
         nb, dev / 1048576, devused / 1048576, atomic_load(&g_live_tex), atomic_load(&g_live_tex_bytes) / 1048576.0, npool, pool / 1048576, arena / 1048576, ndp, ndf,
         g_shcache.n, g_pso.n, g_by_serial.n, g_alias.n, g_ghost.n, heap / 1048576.0, atomic_load(&g_spv_mod_bytes) / 1048576.0, atomic_load(&g_spv_kept_bytes) / 1048576.0);
    { double kb[VA_NCLS], ku[VA_NCLS]; vk_mem_by_kind(kb, ku);
      VLOG("[mem] device blocks by kind, MB in blocks / in use: long-lived %.0f / %.0f, staging %.0f / %.0f, arenas %.0f / %.0f; %.0f MB in %d empty blocks given back so far",
           kb[VA_LONG], ku[VA_LONG], kb[VA_TEMP], ku[VA_TEMP], kb[VA_ARENA], ku[VA_ARENA], g_trim_bytes / 1048576.0, g_trim_n); }
    shmod_report(heap);
}
// The front end's frame report (Android: performance hints, android_main.c): the frame's work against the frame's period (perf-1124 G2': the game thread's own
//   work or the replay thread's, whichever is longer, hle_nvn.c / nvn_work.h; DM_HINT_WORK=wall: the presenting thread's time since its last present, as before)
__attribute__((weak)) void front_frame_work(uint64_t busy_ns, uint64_t target_ns) { (void)busy_ns; (void)target_ns; }   // (no front end report: nothing)
_Atomic uint64_t g_nvn_game_work_ns;   // hle_nvn.c: the frame's work (nvn_work.h: the game thread's own, without its acquire sleep and sync waits, or the replay thread's)
_Atomic uint64_t g_nvn_game_work_avg_ns;   // its average over about 16 frames (android_main.c: the fps guard tells a CPU-bound slowdown from a GPU-bound one)
_Atomic uint64_t g_nvn_replay_work_ns;   // hle_nvn.c rq_run: the replay thread's work for its last frame (DM_NVN_ASYNC; 0 without it)
// ---- the held frame's stores (DM_VK_FRAME_HOLD=2) ----
static int hold_dec(void) {
    static int v = -1; if (v >= 0) return v;
    v = hold_on() >= 2; if (v) { fgr_cont_start(); VLOG("frames held: each pass's stores decided from the whole frame (overwritten unread: left out; read: kept)"); }
    return v;
}
// One pass's attachments: a store whose content is overwritten unread before anything reads it is left out (DON'T CARE); a store left out elsewhere (the learnt skips,
//   DM_VK_SKIP_STORES; depth discards) whose content is read is put back. Unknown (nothing touches it again in the held work): as it was. Caller holds g_fgrmu
static void seg_decide(seg_t *g) {
    if (g->fgp < 0 || g->fgp >= g_fgr.np || g_fgr.overflow) return;
    const fgr_pass *p = &g_fgr.p[g->fgp]; g_dec_passes++;
    for (uint32_t t = 0; t < p->natt; t++) { const fgr_att *a = &p->a[t];
        VkRenderingAttachmentInfo *ai = a->slot < 8 ? &g->ca[a->slot] : a->slot == 8 ? &g->da : &g->sa;
        if (!ai->imageView || ai->storeOp == VK_ATTACHMENT_STORE_OP_NONE) continue;
        int fate = fgr_fate_exact(g->fgp, a);
        if (fate == 4 && a->slot >= 8 && a->img >= 0 && xf_on()) {   // stage 3: a frame-end depth / stencil store, learnt from the next frames (a plain 2D image, never shown)
            vtex *x = (vtex *)(uintptr_t)g_fgr.im[a->img].x;
            if (x && x->levels <= 1 && x->layers <= 1 && !x->is3d && !x->presented) {
                int dc = ai->storeOp == VK_ATTACHMENT_STORE_OP_STORE && xf_skip(&x->xf) && r3_ab() != 0;
                if (dc) { ai->storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; g_xf_dc++; }
                if (a->slot == 8 || !(x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) xf_pending(&x->xf, dc); }
            continue; }
        if (fate == 1 && ai->storeOp == VK_ATTACHMENT_STORE_OP_STORE) { ai->storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; g_dec_dc++; }
        else if (fate == 0 && ai->storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE) { ai->storeOp = VK_ATTACHMENT_STORE_OP_STORE; g_dec_fix++; }
    }
}
// Phase 3: pass h can run inside pass g: the same attachments (views, formats, samples, area, layers), h loads all of them (no clear, which would need one inside),
//   and g resolves nothing at its end (a resolve there would move to h's end)
static int seg_same(const seg_t *g, const seg_t *h) {
    const VkRenderingInfo *a = &g->ri, *b = &h->ri;
    if (a->colorAttachmentCount != b->colorAttachmentCount || a->layerCount != b->layerCount || a->viewMask != b->viewMask || g->samples != h->samples || g->dsfmt != h->dsfmt ||
        a->renderArea.offset.x != b->renderArea.offset.x || a->renderArea.offset.y != b->renderArea.offset.y || a->renderArea.extent.width != b->renderArea.extent.width ||
        a->renderArea.extent.height != b->renderArea.extent.height || !a->pDepthAttachment != !b->pDepthAttachment || !a->pStencilAttachment != !b->pStencilAttachment) return 0;
    for (uint32_t t = 0; t < a->colorAttachmentCount && t < 8; t++) {
        if (g->ca[t].imageView != h->ca[t].imageView || g->cf[t] != h->cf[t]) return 0;
        if (h->ca[t].imageView && h->ca[t].loadOp != VK_ATTACHMENT_LOAD_OP_LOAD) return 0;
        if (g->ca[t].resolveImageView || g->ca[t].resolveMode) return 0; }
    if (a->pDepthAttachment && (g->da.imageView != h->da.imageView || (h->da.imageView && h->da.loadOp != VK_ATTACHMENT_LOAD_OP_LOAD) || g->da.resolveImageView)) return 0;
    if (a->pStencilAttachment && (g->sa.imageView != h->sa.imageView || (h->sa.imageView && h->sa.loadOp != VK_ATTACHMENT_LOAD_OP_LOAD) || g->sa.resolveImageView)) return 0;
    return 1;
}
// Depth / stencil a held pass did not write (perf-1124 GPU fix C, step 1): the post and HUD passes attach the scene's D24S8 for one full-screen draw (Sword / Shield
//   passes 120-122, 152, 154, 156 of a field frame; Legends: Arceus 16-34) and loaded and stored it each time, though nothing in them writes it. The pass and every pass
//   run inside it (seg_record_prim) are known whole at the flush: when none wrote depth or stencil (pass_end's dsk / dsw: draws that write, clears in the pass, a clear by
//   load op) the store is VK_ATTACHMENT_STORE_OP_NONE, which keeps the image as it is in memory: exact, nothing learnt, so a depth buffer the game samples, copies or
//   keeps into the next frame afterwards reads what it held. Both aspects or none (D24S8 is one packed image): left as it is when either is DON'T CARE (a store the
//   frame overwrites unread, seg_decide). When none of them tested either (P.dtest), nothing in the pass touches the attachment and it begins with
//   VK_ATTACHMENT_LOAD_OP_NONE too (only with VK_KHR / VK_EXT_load_store_op_none enabled, g_load_none; else it loads as before). A store NONE that a later pass's
//   begin chose (sk_ro) is put back to STORE when an earlier pass of the run wrote. DM_VK_NO_STORE_NONE=1: no NONE at all; DM_VK_LOAD_NONE=0: no load NONE
static uint64_t g_dsn_store, g_dsn_load, g_dsn_fix;   // held passes stored with NONE here, of them loaded with NONE too; NONE stores put back for a run that wrote
static void ds_none(sub_t *s, int i, int last) {
    seg_t *g = &s->seg[i];
    VkRenderingAttachmentInfo *a[2] = { g->ri.pDepthAttachment && g->da.imageView ? &g->da : NULL, g->ri.pStencilAttachment && g->sa.imageView ? &g->sa : NULL };
    if (!a[0] && !a[1]) return;
    int known = 1, wrote = 0, tested = 0;
    for (int j = i; j <= last; j += 2) { const seg_t *h = &s->seg[j]; if (!h->dsk) known = 0; wrote |= h->dsw; tested |= h->dst; }
    if (!known || wrote) { for (int k = 0; k < 2; k++) if (a[k] && a[k]->storeOp == VK_ATTACHMENT_STORE_OP_NONE) { a[k]->storeOp = VK_ATTACHMENT_STORE_OP_STORE; g_dsn_fix++; } return; }
    if (!g_store_none) return;
    for (int k = 0; k < 2; k++) if (a[k] && a[k]->storeOp != VK_ATTACHMENT_STORE_OP_STORE && a[k]->storeOp != VK_ATTACHMENT_STORE_OP_NONE) return;
    int any = 0; for (int k = 0; k < 2; k++) if (a[k] && a[k]->storeOp == VK_ATTACHMENT_STORE_OP_STORE) { a[k]->storeOp = VK_ATTACHMENT_STORE_OP_NONE; any = 1; }
    g_dsn_store += any;
    if (!g_load_none || tested) return;
    for (int k = 0; k < 2; k++) if (a[k] && a[k]->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD) return;
    for (int k = 0; k < 2; k++) if (a[k]) a[k]->loadOp = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
    g_dsn_load++;
}
// At the flush (caller holds g_cmu): the submission's primary runs its pieces in order, each pass begun and ended around its secondary with the stores decided now
static void seg_record_prim(sub_t *s) {
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKCHECK(vkBeginCommandBuffer(s->prim, &bi));
    pthread_mutex_lock(&g_fgrmu); fgr_rstart_build();
    for (int i = 0; i < s->nseg; i++) { seg_t *g = &s->seg[i];
        if (!g->pass) { vkCmdExecuteCommands(s->prim, 1, &g->cb); continue; }
        seg_decide(g);
        int last = i;   // phase 3: the passes after it on the same targets, nothing but the order between passes in the gaps, run in it (their stores: the last one's)
        while (merge_on() && last + 2 < s->nseg && !s->seg[last + 1].pass && !s->seg[last + 1].work && s->seg[last + 2].pass && seg_same(g, &s->seg[last + 2])) {
            last += 2; seg_decide(&s->seg[last]); g_merge_n++; }
        seg_t *e = &s->seg[last];
        for (int t = 0; t < 8; t++) g->ca[t].storeOp = e->ca[t].storeOp;
        g->da.storeOp = e->da.storeOp; g->sa.storeOp = e->sa.storeOp;
        ds_none(s, i, last);   // (fix C: depth / stencil none of these passes wrote)
        g->ri.pColorAttachments = g->ca; g->ri.pDepthAttachment = g->ri.pDepthAttachment ? &g->da : NULL; g->ri.pStencilAttachment = g->ri.pStencilAttachment ? &g->sa : NULL;
        g->ri.flags |= VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
        VkRect2D ra = seg_area(s, i, last); const VkRect2D *fa = &g->ri.renderArea;   // (the part of the targets these passes' draws reach)
        int small = ra.offset.x != fa->offset.x || ra.offset.y != fa->offset.y || ra.extent.width != fa->extent.width || ra.extent.height != fa->extent.height;
        g_ra_px_full += (uint64_t)fa->extent.width * fa->extent.height; g_ra_px += (uint64_t)ra.extent.width * ra.extent.height; g_ra_n += small;
        rend_begin_ex(s->prim, &g->ri, g->cf, g->dsfmt, g->samples, 0, 1, small ? &ra : NULL);
        for (int j = i; j <= last; j += 2) vkCmdExecuteCommands(s->prim, 1, &s->seg[j].cb);
        if (g_dynren) vkCmdEndRendering(s->prim); else vkCmdEndRenderPass(s->prim);
        i = last;
    }
    pthread_mutex_unlock(&g_fgrmu);
    VKCHECK(vkEndCommandBuffer(s->prim)); s->nseg = 0;
}
void nvn_fgr_arm_series(void) { fgr_arm_series(atomic_load(&g_present_n)); }   // (hle_resscale.c: a battle begins)
int nvn_on_replay_thread(void) __attribute__((weak));   // (hle_nvn.c: 1 on the replay thread, DM_NVN_ASYNC)
int nvn_hint_wall(void) __attribute__((weak));          // (hle_nvn.c: DM_HINT_WORK=wall)
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    { static _Atomic int first; if (!atomic_exchange(&first, 1)) { dm_boot_mark("first present"); atomic_store(&g_pw_playing, 1); } }   // (the prewarm builds one pipeline a present from here)
    uint64_t hp_scene = g_hp_scene_seq == g_hp_seq + 1;   // (HUD-free: this frame's scene copy was made; read before hp_learn starts the next frame)
    // (the copy was taken at the pass learnt from the frame before; this frame's own split can differ (a title screen's passes change from frame to frame):
    //   then the copy is not this frame's scene and this frame goes without HUD-free; S25+, Legends: Arceus's title screen split, 2026-11-20)
    { vtex *oF = g_hp_F; int ok = g_hp_k; hp_learn(tex_res(t)); g_hp_scene_ok = hp_scene && g_hp_F == oF && g_hp_k == ok; }
    if (hold_on()) {
        if (hold_dec()) { vtex *sx = tex_res(t); if (sx) fgr_read_img(sx, sx->w, sx->h, (uint32_t)sx->fmt, FGR_SHOWN); }   // (shown: its last stores are needed)
        hold_flush(0);   // the whole frame to the GPU, before it is shown
        if (hold_dec() && !g_sub && !atomic_load(&g_nheld)) fgr_cont_reset();
    }
    if (sk_on()) { vtex *px = tex_res(t); if (px) { pthread_mutex_lock(&g_skmu); px->presented = 1; pthread_mutex_unlock(&g_skmu); sk_read(px); } }   // shown: never skips its stores
    uint64_t n = atomic_fetch_add(&g_present_n, 1) + 1;
    cap_present(n);
    mw_present(n, t, crop);   // the Mali white-screen check: finished samples read, this frame's lit target and the picture shown sampled when due (nothing unless its ladder is on)
    { vtex *px = tex_res(t); fgr_present(n, px, px ? px->w : 0, px ? px->h : 0, px ? (uint32_t)px->fmt : 0); }   // (the frame recorder: frame boundary, the shown image)
    // the heap route's crash marker goes after 20 s of play on it: 600 presents, or 20 s of wall time and 60 presents (a hot Xclipse 530 building 2,000 pipelines on the
    //   way into Motostoke drew about 300 frames in a minute; the player closed it as stuck, the marker stayed, and the next start took the mirrored route and crashed, #74)
    { static uint64_t t0; if (g_heap_pending && !t0) t0 = mono_ns();
      if (g_heap_pending && (n >= 600 || (n >= 60 && mono_ns() - t0 >= 20000000000ull))) { g_heap_pending = 0; heap_import_mark(-1);
          VLOG("memory pools: 20 s of play on dma-bufs from the heap (%llu presents): the route is kept", (unsigned long long)n); } }
    if (VPROF_ON && n % 300 == 0 && g_vp_n) { double f = 300.0 * 1e6;
        VLOG("[prof] replay per frame: %.0f draws; pass + pipeline %.2f ms, dynamic state + vertex %.2f ms, descriptors %.2f ms, flush + draw %.2f ms (%.1f us a draw)", g_vp_n / 300.0,
             g_vp_t[0] / f, g_vp_t[1] / f, g_vp_t[2] / f, g_vp_t[3] / f, (g_vp_t[0] + g_vp_t[1] + g_vp_t[2] + g_vp_t[3]) / 1e3 / (double)g_vp_n);
        VLOG("[prof] other ops per frame: set target %.2f ms, clears %.2f ms, copies %.2f ms, compute %.2f ms, state %.2f ms; commit %.2f ms", g_vp_o[0] / f, g_vp_o[1] / f, g_vp_o[2] / f, g_vp_o[3] / f, g_vp_o[4] / f, g_vp_c / f);
        VLOG("[prof] of pass + pipeline: opening passes %.2f ms, pipeline lookup %.2f ms (%.2f us a draw)", g_vp_pass / f, g_vp_pso / f, g_vp_pso / 1e3 / (double)g_vp_n);
        VLOG("[prof] opening a pass, per frame: attachment views %.2f ms, barrier %.2f ms, driver begin %.2f ms", g_pb_ns[0] / f, g_pb_ns[1] / f, g_pb_ns[2] / f);
        VLOG("[prof] open passes ended per frame: by a new render target %.1f, a uniform update %.1f, a buffer copy %.1f, a buffer fill %.1f; by any work outside a pass %.1f",
             g_brk[0] / 300.0, g_brk[1] / 300.0, g_brk[2] / 300.0, g_brk[3] / 300.0, g_brk_other / 300.0);
        g_vp_pass = g_vp_pso = 0; memset(g_pb_ns, 0, sizeof g_pb_ns); memset(g_brk, 0, sizeof g_brk); g_brk_other = 0;
        { extern _Atomic uint64_t g_wait_sync_ns, g_wait_sync_n; uint64_t ws = atomic_exchange(&g_wait_sync_ns, 0), wn = atomic_exchange(&g_wait_sync_n, 0);
          VLOG("[prof] waits per frame: sync waits %.2f ms (%.1f a frame), submission slot %.2f ms", ws / f, wn / 300.0, g_wait_ring_ns / f); g_wait_ring_ns = 0; }
        memset(g_vp_t, 0, sizeof g_vp_t); memset(g_vp_o, 0, sizeof g_vp_o); g_vp_c = 0; g_vp_n = 0; }
    static uint64_t g_present_exit;   // when the last present returned (after its pacing sleep): from there to this call is the presenting thread's time (DM_HINT_WORK=wall)
    { int wall = nvn_hint_wall && nvn_hint_wall();
      if (g_present_exit && period_s > 0) { uint64_t busy = mono_ns() - g_present_exit, gw = atomic_load(&g_nvn_game_work_ns); front_frame_work(!wall ? gw : busy > gw ? busy : gw, (uint64_t)(period_s * 1e9)); } }
    static long shots[256]; static int nshot = -1; static const char *shot_path;
    if (nshot < 0) { nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 256; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); } }
    for (int i = 0; i < nshot; i++) if ((long)n == shots[i]) { vtex *x = tex_res(t); char path[1024]; snprintf(path, sizeof path, shot_path, (int)n); if (x && x->img) shot(x, crop, path); }
    g_pace_asked = 0; if (front_metal_layer()) present_window(tex_res(t), crop);
#ifndef __ANDROID__
    { static long of = -1; static const char *op; if (of < 0) { const char *e = getenv("DM_OVL_SHOT"), *c = e ? strchr(e, ':') : NULL; of = c ? atol(e) : 0; op = c ? c + 1 : NULL; }
      if (of && (long)n == of && op) { vtex *x = tex_res(t); if (x && x->img) ovl_shot(x, crop, op); } }
#endif
    // The present waits out the rest of the period since the last one (period_s: hle_nvn.c's present_hold, a load's shorter hold included, L1'). Not on the replay
    //   thread (Faster rendering) for a frame that asked the display for its time (pace_time): the game is paced at its image acquire, rq_push holds a second
    //   queued present back, and the display shows the frame at its time; the sleep here only held the replay back, and with it every game sync wait that drained
    //   the queue (perf-1124 G1': S25+ "GPU sync" 4-10 ms a frame in the field, 23-32 ms in loads at 2.5-5 ms of GPU). A frame with no time asked (no display
    //   timing, pacing off) keeps the sleep: nothing else would space it. DM_VK_PRESENT_SLEEP=1: the sleep everywhere, as before
    static int sleep_all = -1; if (sleep_all < 0) { const char *e = getenv("DM_VK_PRESENT_SLEEP"); sleep_all = e && *e && strcmp(e, "0") && strcmp(e, "off");
        VLOG("present: %s (DM_VK_PRESENT_SLEEP%s)", sleep_all ? "waits out the frame period on every thread" : "the replay thread does not wait out the frame period for a frame with a display time",
             sleep_all ? "=1" : " unset; =1: always waits"); }
    int nosleep = !sleep_all && g_pace_asked && nvn_on_replay_thread && nvn_on_replay_thread();
    uint64_t now = mono_ns(), due = g_last_present + (uint64_t)(period_s * 1e9);
    if (period_s > 0 && due > now && !nosleep) { struct timespec ts = { (time_t)((due - now) / 1000000000ull), (long)((due - now) % 1000000000ull) }; nanosleep(&ts, NULL); now = due; }
    g_last_present = now; g_present_exit = mono_ns();
    if (n % 900 == 0 && atomic_load(&g_pso_new) >= 10) { static atomic_int busy; if (!atomic_exchange(&busy, 1)) {   // new pipelines: the cache file is rewritten in the background
        pthread_t t; void *saver(void *a); if (pthread_create(&t, NULL, saver, &busy)) atomic_store(&busy, 0); else pthread_detach(t); } }
    if (n % 1800 == 0) { VLOG("pipelines built during play so far: %d", atomic_load(&g_pso_ingame)); mem_report(); }
    if (n % 1800 == 0 && (g_nanmm || g_clampzero || g_fmz)) VLOG("shaders changed so far: %d for NaN in min / max / saturate, %d for Arm erratum 3779215, %d for FMZ", atomic_load(&g_nanmm_n), atomic_load(&g_clampzero_n), atomic_load(&g_fmz_n));
    if ((n == 1 || n % 300 == 0) && atomic_load(&g_pre_done) + atomic_load(&g_pre_late) + atomic_load(&g_pre_full))
        VLOG("textures uploaded ahead %llu (%ld MB of them not drawn yet), used before the worker got to them %llu, left to their first use (budget or queue full) %llu; in %llu batches", (unsigned long long)atomic_load(&g_pre_done), atomic_load(&g_pre_unused) >> 20, (unsigned long long)atomic_load(&g_pre_late), (unsigned long long)atomic_load(&g_pre_full), (unsigned long long)atomic_load(&g_pre_batches));
    if (n % 300 == 0 && ubo_snap_on()) {   // (copies / reused / MB: the last 300 presents; the arenas' size is in the [mem] line)
        VLOG("present %llu: uniform snapshots: %llu updates written through so far; last 300 presents: %.0f copies a frame (%.1f MB), %.0f bound again (DM_VK_UBO_SNAPSHOT_REUSE=%d)", (unsigned long long)n,
             (unsigned long long)g_snap_wt, g_snap_copies / 300.0, g_snap_bytes / 300.0 / 1048576.0, g_snap_reused / 300.0, snapc_on());
        g_snap_copies = g_snap_reused = g_snap_bytes = 0; }
    if (n % 300 == 0 && (sk_on() || ubd_on() || barrier_relaxed()))
        VLOG("present %llu: barriers before passes full %llu / narrowed %llu / left out %llu; uniform updates deferred in a pass %llu (passes ended to land them %llu); "
             "stores skipped %llu (read after all %llu), depth / stencil begun store-none %llu (written after all %llu)", (unsigned long long)n,
             (unsigned long long)g_bar_full, (unsigned long long)g_bar_narrow, (unsigned long long)g_bar_none, (unsigned long long)g_ubd_deferred, (unsigned long long)g_ubd_conflicts,
             (unsigned long long)g_sk_skipped, (unsigned long long)g_sk_wrong, (unsigned long long)g_ro_passes, (unsigned long long)g_ro_wrong);
    if (n % 300 == 0 && barrier_relaxed()) VLOG("present %llu: full barriers before passes %llu: for a game barrier %llu (bits primitives %llu, fragments %llu, tiled %llu, indirect %llu, texture %llu, shader %llu), after other work %llu; game barriers narrowed by their bits %llu",
        (unsigned long long)n, (unsigned long long)g_bar_full, (unsigned long long)g_bar_full_game, (unsigned long long)g_bar_bits_hist[0], (unsigned long long)g_bar_bits_hist[1], (unsigned long long)g_bar_bits_hist[2],
        (unsigned long long)g_bar_bits_hist[3], (unsigned long long)g_bar_bits_hist[4], (unsigned long long)g_bar_bits_hist[5], (unsigned long long)g_bar_full_work, (unsigned long long)g_bar_bits_narrow);
    if (n % 300 == 0 && hold_on()) VLOG("present %llu: frames held whole %llu, flushed early by a wait %llu times; passes decided %llu: stores left out %llu, put back %llu; frame-end depth stores left out %llu (needed after all %llu); passes run as one with the one before %llu", (unsigned long long)n,
                                       (unsigned long long)g_hold_frames, (unsigned long long)g_hold_split, (unsigned long long)g_dec_passes, (unsigned long long)g_dec_dc, (unsigned long long)g_dec_fix,
                                       (unsigned long long)g_xf_dc, (unsigned long long)g_xf_bad, (unsigned long long)g_merge_n);
    if (n % 300 == 0 && hold_dec() && g_ra_px_full) VLOG("present %llu: render areas: %llu passes begun on part of their targets; %.1f %% of the targets' pixels loaded / stored (DM_VK_RENDER_AREA)",
                                       (unsigned long long)n, (unsigned long long)g_ra_n, 100.0 * (double)g_ra_px / (double)g_ra_px_full);
    if (n % 300 == 0) { pso_evict(); vk_trim(); shmod_release(); }   // (pipelines kept: Arm's driver, Xclipse, phones under 9 GB; empty device blocks given back; idle shader modules released)
    if (n % 300 == 0 && g_dsc_hits + g_dsc_new) VLOG("present %llu: descriptor sets written %llu, reused %llu (%.0f %%)", (unsigned long long)n, (unsigned long long)g_dsc_new, (unsigned long long)g_dsc_hits, 100.0 * (double)g_dsc_hits / (double)(g_dsc_hits + g_dsc_new));
    // the state shadow's count (DM_VK_STATE_LOG=<presents>, 300 by default; 0: never): driver calls a draw for its state (pipeline, dynamic state, vertex and index
    //   buffers, descriptor pushes; the draw itself not counted), sent and left out as unchanged
    { static int every = -1; if (every < 0) { const char *e = getenv("DM_VK_STATE_LOG"); every = e ? atoi(e) : 300; }
      if (every > 0 && n % (uint64_t)every == 0 && g_sh_draws) { double d = (double)g_sh_draws;
          VLOG("present %llu: draw state to the driver: %.2f calls a draw sent, %.2f left out as unchanged (%llu draws); descriptors pushed %.0f %% of %.1f a draw, vertex bindings bound %.0f %% of %.1f a draw",
               (unsigned long long)n, g_sh_sent / d, g_sh_skip / d, (unsigned long long)g_sh_draws, g_sh_dw_all ? 100.0 * (double)g_sh_dw_sent / (double)g_sh_dw_all : 0.0, g_sh_dw_all / d,
               g_sh_vb_all ? 100.0 * (double)g_sh_vb_sent / (double)g_sh_vb_all : 0.0, g_sh_vb_all / d);
          g_sh_draws = g_sh_sent = g_sh_skip = g_sh_dw_all = g_sh_dw_sent = g_sh_vb_all = g_sh_vb_sent = 0; } }
    if (n % 1800 == 0 && (g_nanmm || g_clampzero || g_exact || g_initloc)) VLOG("shaders changed so far: %d for NaN in min / max / saturate, %d for Arm erratum 3779215, %d for Maxwell's special-function edge cases, %d with registers starting at zero",
                                                                atomic_load(&g_nanmm_n), atomic_load(&g_clampzero_n), atomic_load(&g_exact_n), atomic_load(&g_initloc_n));
    { if (g_tfchk_at && g_frame_no >= g_tfchk_frame + 60) { uint32_t left = 0; for (uint32_t i = 0; i < g_tfchk_n; i += 4) left += *(const uint32_t *)(uintptr_t)(g_tfchk_at + i) == 0x7fc0deadu;
          if (left == g_tfchk_n / 4) VLOG("[RED] transform feedback check: the captures wrote nothing on this driver (the marker is still there 60 frames later): skinned and swaying models get no vertices");
          else VLOG("transform feedback check: the captures write their vertices (%u of %u marker words replaced)", g_tfchk_n / 4 - left, g_tfchk_n / 4);
          g_tfchk_at = 0; } }
    if (n == 1 || n % 300 == 0) VLOG("present %llu: %llu render target sets (%llu kept the open pass, %llu passes begun), %llu draws so far; textures uploaded %llu (re-uploaded %llu, %.1f MB)%s", (unsigned long long)n, (unsigned long long)g_stat_passes, (unsigned long long)g_stat_rt_same, (unsigned long long)g_stat_begins,
                                     (unsigned long long)g_stat_draws, (unsigned long long)g_pkg_uploads, (unsigned long long)g_pkg_reuploads, g_up_bytes / 1048576.0,
                                     g_bcgpu ? (snprintf(g_bc_line, sizeof g_bc_line, "; BC layer: %llu levels, %llu copies", (unsigned long long)g_bc_gpu_levels, (unsigned long long)g_bc_gpu_copies), g_bc_line) : "");
    atomic_store(&g_apso_budget_ns, g_apso_frame_ns);   // (each frame's wait for pipelines being built: apso_wait)
    if (n % 300 == 0 && atomic_load(&g_apso_done) + atomic_load(&g_apso_skipped)) VLOG("present %llu: pipelines built in the background %llu, draws skipped waiting for one %llu (DM_VK_ASYNC_PSO)", (unsigned long long)n, (unsigned long long)atomic_load(&g_apso_done), (unsigned long long)atomic_load(&g_apso_skipped));
    if (n % 300 == 0 && g_pass_resumed) VLOG("present %llu: %llu target changes kept their open pass (DM_VK_PASS_DEFER)", (unsigned long long)n, (unsigned long long)g_pass_resumed);
    if (n % 300 == 0) VLOG("present %llu: passes left out so far: clear-only passes folded %llu (depth clears kept %llu: taken by the next pass's load op %llu, replaced by a second clear %llu, "
                           "written out as before %llu, of them by a read %llu; passes begun again for a draw sampling a taken clear %llu); empty depth passes dropped %llu; "
                           "depth stores skipped %llu (begun store-none %llu, written after all %llu; held passes stored with none %llu, loaded with none too %llu; none put back %llu)",
                           (unsigned long long)n, (unsigned long long)(g_lc_folded + g_lc_replaced - g_lc_redo), (unsigned long long)g_lc_kept, (unsigned long long)g_lc_folded, (unsigned long long)g_lc_replaced,
                           (unsigned long long)g_lc_written, (unsigned long long)g_lc_net, (unsigned long long)g_lc_redo, (unsigned long long)g_empty_dropped,
                           (unsigned long long)(g_ro_passes - g_ro_wrong + g_dsn_store), (unsigned long long)g_ro_passes, (unsigned long long)g_ro_wrong,
                           (unsigned long long)g_dsn_store, (unsigned long long)g_dsn_load, (unsigned long long)g_dsn_fix);
    if (n % 300 == 0 && !VPROF_ON)   // (running totals: DM_VK_PROF prints these per 300 frames instead, and resets them)
        VLOG("passes so far: %llu begun; ended by a new target %llu, a uniform update %llu, a buffer copy %llu, a buffer fill %llu, any work outside a pass %llu (those three included); "
             "%llu opened onto exactly the attachments a target change had just ended", (unsigned long long)g_stat_begins, (unsigned long long)g_brk[0], (unsigned long long)g_brk[1],
             (unsigned long long)g_brk[2], (unsigned long long)g_brk[3], (unsigned long long)g_brk_other, (unsigned long long)g_pass_samev);
    if (!g_dynren && n % 300 == 0) { pthread_mutex_lock(&g_fbcmu);
        VLOG("render pass framebuffers so far: %llu found again, %llu made, %d kept%s", (unsigned long long)g_fbc_hit, (unsigned long long)g_fbc_made, g_nfbc, fbc_on() ? "" : " (DM_VK_FB_CACHE=0: one per pass)");
        pthread_mutex_unlock(&g_fbcmu); }
    return n;
}
void mtl_present_wait(uint64_t ticket) { (void)ticket; }
double mtl_presented_time(uint64_t n) { (void)n; return 0; }
uint32_t mtl_present_qos(void) { return 0; }
int mtl_has_window(void) { return 0; }
int mtl_display_hz(void) { return 60; }
__attribute__((weak)) int nvn_game_fps(void) { return 60; }
int mtl_fps_target(void) { const char *e = getenv("DM_FPS"); int f = e ? atoi(e) : 0; return f == 30 || f == 60 ? f : nvn_game_fps() == 30 ? 30 : 60; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_set_frame(uint64_t f) { g_frame_no = f; }
double mtl_gpu_ms(uint64_t f) { unsigned k = f % GRING; if (!g_tsq || atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending) > 0) return -1; return atomic_load(&g_gpu[k].ns) / 1e6; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
// statistics read by the host's report lines
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_passes; *draws = g_stat_draws; }
void mtl_alloc_stats(uint64_t *a, uint64_t *b, int *n) { *a = *b = 0; *n = 0; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
void mtl_create_ns(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
void mtl_pso_stats(uint64_t *draw, uint64_t *bg) { *draw = (uint64_t)atomic_load(&g_pso_ingame); *bg = atomic_load(&g_apso_done); }   // (pipelines built by draws / in the background: the load report, hle_nvn.c)
uint64_t mtl_prewarm_play_n(void) { return atomic_load(&g_pw_play_n); }   // (list pipelines the start-up prewarm handled after the game went on: the load report, hle_nvn.c)
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = g_pkg_uploads; *r = g_pkg_reuploads; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return g_up_bytes; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_upload_ns, 0); }   // (texture uploads' time since the last call: the long-frame line)
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
void mtl_res_stats(char *out, size_t n) { if (n) snprintf(out, n, "Vulkan: no residency set"); }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts) { *deferred = g_ubd_deferred; *conflicts = g_ubd_conflicts; }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = g_bar_none + g_bar_narrow; *full = g_bar_full; }
uint64_t mtl_blit_splits(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
#include "golden_vk.inc"   // golden-frame recordings: the backend side (golden_rec.h golden_be_*)
