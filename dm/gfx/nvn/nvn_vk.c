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
#include <dirent.h>
#include <sys/uio.h>   // process_vm_readv / writev (the pool mapping check)
static int g_vk_wo_driver;   // the driver maps GPU memory write-only (the Android Emulator): its mappings are never moved, every pool is mirrored
#include <zlib.h>
#ifdef __ANDROID__
#include <malloc.h>
#define VK_NO_PROTOTYPES   // every Vulkan function comes from the driver opened at start (vk_dispatch.h, vk_open)
#endif
#include <vulkan/vulkan.h>
#ifdef __ANDROID__
#include <vulkan/vulkan_android.h>
#include <android/native_window.h>
#include <dlfcn.h>
#include "vk_dispatch.h"
#endif
#include <pthread.h>
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
#include <sys/prctl.h>
#include "nvn_mtl.h"
#include "nvn_obj.h"
#include "nvn_hmap.h"
#include "nvn_spv.h"
#include "nvn_ovl.h"
#include "nvn_dirty.h"
#include "../tex/nvn_fmt_vk.h"
#include "../tex/bc_decode.h"
#include "../tex/nvn_deswizzle.h"
#include "dmsh_chain.h"
#include "game_profile.h"
#include "../../host/dm.h"

#define VLOG(...) do { fprintf(stderr, "[dm] [nvn-vk] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#define VDIE(...) do { VLOG("FATAL " __VA_ARGS__); abort(); } while (0)
static int g_red_once[128];
#define RED_ONCE(k, ...) do { if (!g_red_once[k]++) VLOG("[RED] " __VA_ARGS__); } while (0)
#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) VDIE("%s failed: %d (%s:%d)", #x, r_, __FILE__, __LINE__); } while (0)
#define NTEX 256

uint64_t guest_ns(void);
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

// ================= Device =================
static VkInstance g_inst; static VkPhysicalDevice g_pd; static VkDevice g_dev; static VkQueue g_q; static uint32_t g_qf;
static VkPhysicalDeviceProperties g_props; static VkPhysicalDeviceMemoryProperties g_memp;
static int g_has_swapchain, g_host_import, g_clipctl, g_border, g_divisor, g_eds2, g_moltenvk;
static VkDeviceSize g_host_align = 4096;
static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;
// Linux / Android pools: the driver's memory exported as a file descriptor and mapped a second time over the pool's pages (g_fdmap 1; -1 until the first pool tries it,
// 0 when the driver has no such export or its mmap fails: then the mremap route below)
static int g_fdmap = 0;   // queue submissions come from the submit thread and the present thread
static PFN_vkCmdPushDescriptorSetKHR p_push;
static int g_nopush;   // no VK_KHR_push_descriptor (the emulator's gfxstream, some phones): each draw's bindings go into a descriptor set from the submission's pools
static uint32_t g_maxpush = 32;   // maxPushDescriptors: a program with more descriptors takes the descriptor-set path too (vprog np)
static int g_np;   // the program being bound uses descriptor sets (g_nopush, or over g_maxpush)
static PFN_vkGetMemoryHostPointerPropertiesEXT p_hostprops;

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static void completion_start(void);
static void dummies_init(void);
static int host_import_mark(int set);   // (queue_submit)
static void pcache_open(void); static void *prewarm_main(void *arg);   // pipeline list and cache (below)
#ifdef __ANDROID__
// The Vulkan driver: DM_VK_DRIVER=<folder>/<library> (an Adreno driver package such as Qualcomm's newer builds or Mesa Turnip, put in place by
// android_main.c's driver_install) is opened through libadrenotools (BSD-2-Clause, shipped in the app with its hook libraries); without it, or when that fails, the system's libvulkan
static void *vk_open(void) {
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
        VLOG("[RED] Vulkan driver %s could not be opened (libadrenotools %s, %s): the system driver is used", drv, op ? "failed" : "missing", dlerror() ? dlerror() : "");
    }
    return dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
}
#define VK_LOAD_G(f) f = (PFN_##f)vkGetInstanceProcAddr(NULL, #f);
#define VK_LOAD_I(f) f = (PFN_##f)vkGetInstanceProcAddr(g_inst, #f);
#define VK_LOAD_D(f) f = (PFN_##f)vkGetDeviceProcAddr(g_dev, #f);
#endif
static int g_gpu_vendor, g_gpu_gen, g_api, g_store_none;   // 1 Adreno (gen 6 / 7 / 8), 2 Mali (gen 3 Bifrost or older, 4 Valhall, 5 fifth generation); device API as 10 * major + minor
void mtl_init(void) {
    if (g_dev) return;
#ifdef __ANDROID__
    if (!vkGetInstanceProcAddr) {
        void *lib = vk_open(); if (!lib) VDIE("no Vulkan driver could be opened");
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr"); if (!vkGetInstanceProcAddr) VDIE("the Vulkan driver has no vkGetInstanceProcAddr");
        VK_GLOBAL(VK_LOAD_G)
    }
#endif
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "nxnative", .apiVersion = VK_API_VERSION_1_3 };
    uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties *ie = calloc(nie ? nie : 1, sizeof *ie); vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
    const char *iext[6]; uint32_t niext = 0; VkInstanceCreateFlags ifl = 0;
    if (has_ext(ie, nie, "VK_KHR_portability_enumeration")) { iext[niext++] = "VK_KHR_portability_enumeration"; ifl |= 0x00000001; /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */ }
    if (has_ext(ie, nie, "VK_KHR_get_physical_device_properties2")) iext[niext++] = "VK_KHR_get_physical_device_properties2";
    if (has_ext(ie, nie, "VK_KHR_surface")) iext[niext++] = "VK_KHR_surface";
#ifdef __ANDROID__
    if (has_ext(ie, nie, "VK_KHR_android_surface")) iext[niext++] = "VK_KHR_android_surface";
#endif
    free(ie);
    const char *layers[1]; uint32_t nlayers = 0;
    if (getenv("DM_VK_VALIDATION")) layers[nlayers++] = "VK_LAYER_KHRONOS_validation";
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .flags = ifl, .pApplicationInfo = &ai, .enabledExtensionCount = niext, .ppEnabledExtensionNames = iext,
                                 .enabledLayerCount = nlayers, .ppEnabledLayerNames = layers };
    VKCHECK(vkCreateInstance(&ici, NULL, &g_inst));
#ifdef __ANDROID__
    VK_INSTANCE(VK_LOAD_I)
#endif
    uint32_t npd = 0; vkEnumeratePhysicalDevices(g_inst, &npd, NULL); if (!npd) VDIE("no Vulkan device");
    VkPhysicalDevice pds[8]; if (npd > 8) npd = 8; vkEnumeratePhysicalDevices(g_inst, &npd, pds); g_pd = pds[0];
    vkGetPhysicalDeviceProperties(g_pd, &g_props); vkGetPhysicalDeviceMemoryProperties(g_pd, &g_memp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    for (g_qf = 0; g_qf < nq && (qp[g_qf].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT); g_qf++) {}
    if (g_qf == nq) VDIE("no graphics + compute queue");
    uint32_t nde = 0; vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nde, NULL);
    VkExtensionProperties *de = calloc(nde ? nde : 1, sizeof *de); vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nde, de);
    const char *dext[32]; uint32_t ndext = 0;
    // GPU family (the log, and gpu_family() for the few settings that differ): Adreno 6xx / 7xx / 8xx, Mali by architecture
    {   const char *nm = g_props.deviceName; const char *q;
        if (g_props.vendorID == 0x5143 && (q = strstr(nm, "Adreno")) && (q = strpbrk(q, "0123456789"))) { int m = atoi(q); g_gpu_vendor = 1; g_gpu_gen = m >= 100 ? m / 100 : 0; }
        else if (g_props.vendorID == 0x13B5) { g_gpu_vendor = 2; q = strpbrk(nm, "0123456789"); int m = q ? atoi(q) : 0;
            // Mali architecture: 5th generation (G715 / G720 / G925 and the Immortalis parts), Valhall (G57 / G68 / G77 / G78 / G310 / G510 / G610 / G710), Bifrost (G31 – G76) and older
            g_gpu_gen = (m >= 715 || m == 625 || m == 725) ? 5 : (m == 57 || m == 68 || m == 77 || m == 78 || m >= 310) ? 4 : 3; }
        VLOG("GPU: %s (%s%s%d)", nm, g_gpu_vendor == 1 ? "Adreno " : g_gpu_vendor == 2 ? "Mali " : "other", g_gpu_vendor == 1 ? "" : g_gpu_vendor == 2 ? "generation " : "", g_gpu_vendor == 1 ? g_gpu_gen * 100 : g_gpu_gen);
    }
    // Vulkan 1.3 is what the replay is written against (dynamic rendering, extended dynamic state, timeline semaphores). Older drivers (Adreno 6xx and Mali on Android 12 and earlier
    // report 1.1 or 1.2) have the same as extensions when they are recent enough: those are enabled and their functions loaded under their extension names (vk_suffixed)
    g_api = VK_API_VERSION_MINOR(g_props.apiVersion) + 10 * VK_API_VERSION_MAJOR(g_props.apiVersion);
    if (g_api < 13) {
        const char *need[] = { "VK_KHR_dynamic_rendering", "VK_EXT_extended_dynamic_state", "VK_KHR_timeline_semaphore", "VK_KHR_create_renderpass2", "VK_KHR_depth_stencil_resolve", "VK_KHR_maintenance2", "VK_KHR_multiview" };
        for (unsigned i = 0; i < sizeof need / sizeof *need; i++) {
            if (g_api >= 12 && i >= 2) break;   // core in 1.2
            if (g_api >= 11 && i >= 5) break;   // core in 1.1
            if (!has_ext(de, nde, need[i])) VDIE("this GPU's Vulkan driver is version %u.%u without %s (Vulkan 1.3 is needed, or 1.1 / 1.2 with its extensions)%s", VK_API_VERSION_MAJOR(g_props.apiVersion),
                                                 VK_API_VERSION_MINOR(g_props.apiVersion), need[i], g_gpu_vendor == 1 ? "; on Adreno 6xx a Mesa Turnip driver package (custom driver) has it" : "");
            dext[ndext++] = need[i]; }
        if (has_ext(de, nde, "VK_KHR_shader_float16_int8")) dext[ndext++] = "VK_KHR_shader_float16_int8";
        if (g_api < 12 && has_ext(de, nde, "VK_KHR_8bit_storage")) dext[ndext++] = "VK_KHR_8bit_storage";
        VLOG("Vulkan %u.%u driver: dynamic rendering, extended dynamic state%s through extensions", VK_API_VERSION_MAJOR(g_props.apiVersion), VK_API_VERSION_MINOR(g_props.apiVersion), g_api < 12 ? ", timeline semaphores" : "");
    }
    // Store op NONE (an attachment the pass only reads is neither written back nor dropped: Adreno 7xx / 8xx and recent Mali drivers; sk_ro)
    if (!getenv("DM_VK_NO_STORE_NONE")) {
        if (g_api >= 14) g_store_none = 1;
        else if (has_ext(de, nde, "VK_KHR_load_store_op_none")) { dext[ndext++] = "VK_KHR_load_store_op_none"; g_store_none = 1; }
        else if (has_ext(de, nde, "VK_EXT_load_store_op_none")) { dext[ndext++] = "VK_EXT_load_store_op_none"; g_store_none = 1; }
        else if (has_ext(de, nde, "VK_QCOM_render_pass_store_ops")) { dext[ndext++] = "VK_QCOM_render_pass_store_ops"; g_store_none = 1; }
    }
    if (has_ext(de, nde, "VK_KHR_push_descriptor") && !getenv("DM_VK_NO_PUSH")) dext[ndext++] = "VK_KHR_push_descriptor";
    else { g_nopush = 1; VLOG("no push descriptors: descriptor sets from per-submission pools"); }
    if (has_ext(de, nde, "VK_KHR_portability_subset")) { dext[ndext++] = "VK_KHR_portability_subset"; g_moltenvk = 1; }
    if (!g_nopush) { VkPhysicalDevicePushDescriptorPropertiesKHR pd = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pd }; vkGetPhysicalDeviceProperties2(g_pd, &pp);
        if (pd.maxPushDescriptors) g_maxpush = pd.maxPushDescriptors; if (getenv("DM_VK_MAXPUSH")) g_maxpush = (uint32_t)atoi(getenv("DM_VK_MAXPUSH"));   // a lower limit tests the set path
        VLOG("push descriptors: up to %u per program", g_maxpush); }
    if ((g_host_import = has_ext(de, nde, "VK_EXT_external_memory_host"))) dext[ndext++] = "VK_EXT_external_memory_host";
    if ((g_has_swapchain = has_ext(de, nde, "VK_KHR_swapchain"))) dext[ndext++] = "VK_KHR_swapchain";
    if ((g_clipctl = has_ext(de, nde, "VK_EXT_depth_clip_control"))) dext[ndext++] = "VK_EXT_depth_clip_control";
    if ((g_border = has_ext(de, nde, "VK_EXT_custom_border_color"))) dext[ndext++] = "VK_EXT_custom_border_color";
    if ((g_divisor = has_ext(de, nde, "VK_EXT_vertex_attribute_divisor"))) dext[ndext++] = "VK_EXT_vertex_attribute_divisor";
    if ((g_eds2 = has_ext(de, nde, "VK_EXT_extended_dynamic_state2"))) dext[ndext++] = "VK_EXT_extended_dynamic_state2";
#ifdef __linux__
    if (!g_host_import && has_ext(de, nde, "VK_KHR_external_memory_fd") && !(getenv("DM_VK_NO_FDMAP") && atoi(getenv("DM_VK_NO_FDMAP")))) { dext[ndext++] = "VK_KHR_external_memory_fd"; g_fdmap = -1; }
    VLOG("memory pools: fd export %s (host import %d, VK_KHR_external_memory_fd %d)", g_fdmap ? "to try" : "off", g_host_import, has_ext(de, nde, "VK_KHR_external_memory_fd"));
#endif
    free(de);
    if (getenv("DM_VK_NO_HOST_IMPORT")) g_host_import = 0;
    // Imported host memory pins the game's pools (about 3 GB) for every submission. Samsung's Xclipse driver of the late-2026 One UI update refuses that a few frames into play
    // (vkQueueSubmit: VK_ERROR_OUT_OF_HOST_MEMORY with a gigabyte free, #39), so on Xclipse the driver's own memory is moved onto the pools instead (as on Mali); elsewhere a
    // submission that fails while pools are imported leaves a marker (host_import_failed, next to the pipeline cache) and the next start does the same. DM_VK_HOST_IMPORT=1 forces import
    if (g_host_import && !getenv("DM_VK_HOST_IMPORT")) {
        const char *why = g_props.vendorID == 0x144D ? "Samsung Xclipse" : NULL;
        if (!why && host_import_mark(0)) why = "a submission failed with it last time";
        if (why) { g_host_import = 0; VLOG("memory pools: host import off (%s): driver memory moved onto the pools", why); }
    }
    // Every feature the device has is enabled (the translated shaders declare what they use: 64-bit integers, image formats without a type, …), robust buffer access included:
    // uniform blocks are declared 64 KB and bound with the game's smaller ranges, and Maxwell reads 0 past them
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fbc = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT };
    VkPhysicalDeviceDepthClipControlFeaturesEXT fcc = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, .pNext = g_border ? &fbc : NULL };
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT fdv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, .pNext = g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL };
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT fe2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, .pNext = g_divisor ? (void *)&fdv : g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL };
    void *ext_chain = g_eds2 ? (void *)&fe2 : g_divisor ? (void *)&fdv : g_clipctl ? (void *)&fcc : g_border ? (void *)&fbc : NULL;
    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = ext_chain };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13 };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12 };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11 };
    // older drivers: the 1.1 / 1.2 structures they know, and the extensions' own
    VkPhysicalDeviceDynamicRenderingFeaturesKHR xdr = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, .pNext = ext_chain };
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT xds = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, .pNext = &xdr };
    VkPhysicalDeviceTimelineSemaphoreFeatures xts = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, .pNext = &xds };
    VkPhysicalDevice16BitStorageFeatures x16 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, .pNext = &xts };
    VkPhysicalDeviceShaderDrawParametersFeatures xdp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, .pNext = &x16 };
    if (g_api == 12) { f12.pNext = &xds; f11.pNext = &f12; }   // 1.2: its own structures (timeline semaphores in them) and the 1.3 extensions'
    else if (g_api < 12) f2.pNext = &xdp;
    vkGetPhysicalDeviceFeatures2(g_pd, &f2);
    if (g_api < 13) { f13.dynamicRendering = xdr.dynamicRendering; if (g_api < 12) f12.timelineSemaphore = xts.timelineSemaphore;
        if (!xds.extendedDynamicState) VDIE("extended dynamic state reported but not supported"); }
    if (!f13.dynamicRendering || !f12.timelineSemaphore) VDIE("dynamic rendering %d / timeline semaphores %d missing", f13.dynamicRendering, f12.timelineSemaphore);
    if (!fcc.depthClipControl) g_clipctl = 0;
    if (!fbc.customBorderColors) g_border = 0;
    float prio = 1;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = ndext, .ppEnabledExtensionNames = dext };
    VKCHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
#ifdef __ANDROID__
    VK_DEVICE(VK_LOAD_D)
    if (g_api < 13) {   // the 1.3 functions under their extension names
        #define VK_SUFFIXED(f, sfx) { PFN_##f p_ = (PFN_##f)vkGetDeviceProcAddr(g_dev, #f sfx); if (p_) f = p_; if (!f) VDIE("the driver has no %s", #f sfx); }
        VK_SUFFIXED(vkCmdBeginRendering, "KHR") VK_SUFFIXED(vkCmdEndRendering, "KHR") VK_SUFFIXED(vkCmdBindVertexBuffers2, "EXT")
        VK_SUFFIXED(vkCmdSetCullMode, "EXT") VK_SUFFIXED(vkCmdSetFrontFace, "EXT") VK_SUFFIXED(vkCmdSetPrimitiveTopology, "EXT") VK_SUFFIXED(vkCmdSetDepthTestEnable, "EXT")
        VK_SUFFIXED(vkCmdSetDepthWriteEnable, "EXT") VK_SUFFIXED(vkCmdSetDepthCompareOp, "EXT") VK_SUFFIXED(vkCmdSetStencilTestEnable, "EXT") VK_SUFFIXED(vkCmdSetStencilOp, "EXT")
        if (g_api < 12) VK_SUFFIXED(vkWaitSemaphores, "KHR")
        #undef VK_SUFFIXED
    }
#endif
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
    p_push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(g_dev, "vkCmdPushDescriptorSetKHR");
    p_hostprops = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (g_host_import) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp }; vkGetPhysicalDeviceProperties2(g_pd, &pp);
        g_host_align = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096;
    }
    {   // 24-bit depth where the device renders to it (Adreno), else the 32-bit float formats (Apple GPUs)
        VkFormatProperties a, b; vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_X8_D24_UNORM_PACK32, &a); vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_D24_UNORM_S8_UINT, &b);
        nvn_fmt_vk_depth24(!!(a.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT), !!(b.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
    }
    // BC1-BC7: sampled directly where the GPU has them (Adreno, desktop), else decoded on upload (Mali, Xclipse, PowerVR; DM_VK_NO_BC=1 tries that path anywhere)
    if (getenv("DM_VK_NO_BC") && atoi(getenv("DM_VK_NO_BC"))) f2.features.textureCompressionBC = VK_FALSE;
    nvn_fmt_vk_bc(f2.features.textureCompressionBC);
    VLOG("Vulkan device %s (API %u.%u.%u, driver %#x): pools %s, depth clip control %d, custom border colours %d, vertex divisor %d, extended dynamic state 2 %d, store op none %d, BC textures %s",
         g_props.deviceName, VK_API_VERSION_MAJOR(g_props.apiVersion), VK_API_VERSION_MINOR(g_props.apiVersion), VK_API_VERSION_PATCH(g_props.apiVersion), g_props.driverVersion,
         g_host_import ? "imported host memory" : "driver memory moved onto the pool (mremap)", g_clipctl, g_border, g_divisor, g_eds2, g_store_none,
         nvn_fmt_vk_decode(0x46) ? "decoded on upload" : "sampled by the GPU");
    completion_start();
    dummies_init();
    pcache_open();
    if (getenv("DM_VK_PSO_SEED") || getenv("DM_VK_PSO_RECORD")) { pthread_t t; if (!pthread_create(&t, NULL, prewarm_main, NULL)) pthread_detach(t); }
}

// ================= Device memory (images, internal buffers) =================
// A first-fit allocator over 64 MB blocks per memory type (larger requests get a block of their own): drivers cap the number of allocations (4096 is common), and a game holds
// thousands of textures
typedef struct { VkDeviceMemory mem; VkDeviceSize size; uint32_t type; void *map; int nfree, cfree; struct { VkDeviceSize off, size; } *fr; int used; } vblock;
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
static vmalloc_t vk_alloc(VkMemoryRequirements rq, VkMemoryPropertyFlags want, int map) {
    int t = mem_type(rq.memoryTypeBits, want, map ? 0 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (t < 0) VDIE("no memory type for bits %#x flags %#x", rq.memoryTypeBits, want);
    VkDeviceSize al = rq.alignment > g_props.limits.bufferImageGranularity ? rq.alignment : g_props.limits.bufferImageGranularity;
    pthread_mutex_lock(&g_amu);
    for (int k = 0; k < g_nblocks; k++) {
        vblock *b = g_blocks[k]; if (b->type != (uint32_t)t || (map && !b->map)) continue;
        for (int i = 0; i < b->nfree; i++) {
            VkDeviceSize o = (b->fr[i].off + al - 1) / al * al, end = b->fr[i].off + b->fr[i].size;
            if (o + rq.size > end) continue;
            VkDeviceSize f0 = b->fr[i].off, f1 = end; memmove(b->fr + i, b->fr + i + 1, (b->nfree - i - 1) * sizeof *b->fr); b->nfree--;
            if (o > f0) fr_add(b, f0, o - f0);
            if (o + rq.size < f1) fr_add(b, o + rq.size, f1 - o - rq.size);
            b->used++; pthread_mutex_unlock(&g_amu);
            return (vmalloc_t){b, o, rq.size};
        }
    }
    VkDeviceSize bs = rq.size > (32ull << 20) ? (rq.size + 0xffff) & ~0xffffull : 64ull << 20;
    { static long cap = -2; if (cap == -2) { const char *e = getenv("DM_VK_MEM_CAP_MB"); cap = e ? atol(e) : -1; }   // test runs: stop before the machine runs out of memory
      double total = bs; for (int k = 0; k < g_nblocks; k++) total += g_blocks[k]->size;
      if (cap > 0 && total > cap * 1048576.0) { pthread_mutex_unlock(&g_amu); VDIE("device memory would pass DM_VK_MEM_CAP_MB=%ld (%.0f MB in %d blocks)", cap, total / 1048576, g_nblocks); } }
    vblock *b = calloc(1, sizeof *b); b->size = bs; b->type = (uint32_t)t;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bs, .memoryTypeIndex = (uint32_t)t };
    VkResult r = vkAllocateMemory(g_dev, &mai, NULL, &b->mem);
    if (r) { pthread_mutex_unlock(&g_amu); VDIE("vkAllocateMemory %llu bytes type %d failed: %d", (unsigned long long)bs, t, r); }
    if (g_memp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vkMapMemory(g_dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
    if (rq.size < bs) fr_add(b, rq.size, bs - rq.size);
    b->used = 1;
    g_blocks = realloc(g_blocks, (g_nblocks + 1) * sizeof *g_blocks); g_blocks[g_nblocks++] = b;
    pthread_mutex_unlock(&g_amu);
    if (getenv("DM_VK_BLOCKLOG")) VLOG("device block %d: %.0f MB type %d for a %.1f MB request from %p", g_nblocks, bs / 1048576.0, t, rq.size / 1048576.0, __builtin_return_address(0));
    return (vmalloc_t){b, 0, rq.size};
}
static void vk_free(vmalloc_t a) {
    if (!a.b) return;
    pthread_mutex_lock(&g_amu); fr_add(a.b, a.off, a.size); a.b->used--; pthread_mutex_unlock(&g_amu);
}

// ================= Deferred destruction: objects the GPU may still use go when the submissions recorded so far are done =================
enum { DF_IMAGE, DF_VIEW, DF_BUFFER, DF_BUFVIEW, DF_ALLOC, DF_PIPELINE };
typedef struct { int kind; uint64_t fence; union { VkImage img; VkImageView view; VkBuffer buf; VkBufferView bview; VkPipeline pipe; } h; vmalloc_t a; } dfr;
static dfr *g_dfr; static int g_ndfr, g_cdfr; static pthread_mutex_t g_dfmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_fence;                  // last fence value handed to a submission (submit thread)
static _Atomic uint64_t g_done;           // last fence value the GPU completed (completion thread)
static void defer_add(dfr d) {
    d.fence = g_fence + 1;
    pthread_mutex_lock(&g_dfmu);
    if (g_ndfr == g_cdfr) { g_cdfr = g_cdfr ? g_cdfr * 2 : 256; g_dfr = realloc(g_dfr, g_cdfr * sizeof *g_dfr); }
    g_dfr[g_ndfr++] = d; pthread_mutex_unlock(&g_dfmu);
}
static void defer_drain(void) {
    uint64_t done = atomic_load(&g_done);
    pthread_mutex_lock(&g_dfmu);
    for (int i = 0; i < g_ndfr;) {
        dfr d = g_dfr[i];
        if (d.fence > done) { i++; continue; }
        g_dfr[i] = g_dfr[--g_ndfr];
        switch (d.kind) {
            case DF_IMAGE: vkDestroyImage(g_dev, d.h.img, NULL); break; case DF_VIEW: vkDestroyImageView(g_dev, d.h.view, NULL); break;
            case DF_BUFFER: vkDestroyBuffer(g_dev, d.h.buf, NULL); break; case DF_BUFVIEW: vkDestroyBufferView(g_dev, d.h.bview, NULL); break;
            case DF_PIPELINE: vkDestroyPipeline(g_dev, d.h.pipe, NULL); break; case DF_ALLOC: break;
        }
        vk_free(d.a);
    }
    pthread_mutex_unlock(&g_dfmu);
}

// ================= Memory pools: GPU buffers over the game's pages =================
// Three ways a pool's pages and the GPU share memory: imported host memory (VK_EXT_external_memory_host), the driver's memory moved onto the pool (mremap, needs every
// allocation to be a whole mapping of its own: Adreno), or a mirror (shadow): the GPU buffer in the driver's memory, the game's pages unchanged, the pages the game wrote
// copied in before each submission and what the GPU wrote copied back once it is done (drivers that sub-allocate their mappings: the Android emulator's; DM_VK_POOL_SHADOW=1)
typedef struct { uintptr_t lo, hi, base; VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; const void *key; void *moved_from; uint8_t *shadow; uint64_t stamp; int tracked_here, fdmapped; } pool_ent;
static pool_ent *g_pools; static int g_npool, g_cpool; static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_int g_nshadow;   // mirrored pools alive
static int g_plast = -1;
static const VkBufferUsageFlags BUF_ALL = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT
    | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
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
void mtl_pool_add(const void *mem, uint64_t size) {
    mtl_init();
    uintptr_t pg = (uintptr_t)(g_host_import ? g_host_align : (VkDeviceSize)getpagesize()), lo = (uintptr_t)mem, hi = lo + size;
    uintptr_t a = lo & ~(pg - 1), b = (hi + pg - 1) & ~(pg - 1); VkDeviceSize n = b - a;
    pool_ent e = { lo, hi, a, VK_NULL_HANDLE, VK_NULL_HANDLE, n, mem, NULL, NULL, 0, 0 };
    void *p = NULL; int reused = 0;
#ifdef __linux__
    static int shadow_forced = -1; if (shadow_forced < 0) shadow_forced = getenv("DM_VK_POOL_SHADOW") && atoi(getenv("DM_VK_POOL_SHADOW"));
    if (!g_host_import && !shadow_forced && !g_fdmap && pool_mem_take(n, &e.buf, &e.mem, &p)) { reused = 1; goto reuse; }
#endif
    VkExternalMemoryBufferCreateInfo eb = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT };
    VkExternalMemoryBufferCreateInfo ebfd = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = g_host_import ? &eb : g_fdmap ? &ebfd : NULL, .size = n, .usage = BUF_ALL };
    VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &e.buf));
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
            if (!nvn_dirty_tracked((void *)a, n)) { nvn_dirty_add_range((void *)a, n); e.tracked_here = 1; }
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
        if (g_fdmap) {
            static PFN_vkGetMemoryFdKHR getfd; if (!getfd) getfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g_dev, "vkGetMemoryFdKHR");
            VkExportMemoryAllocateInfo ex = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
            VkMemoryAllocateInfo xai = mai; xai.pNext = &ex;
            int fd = -1; void *tmp = MAP_FAILED;
            if (getfd && !vkAllocateMemory(g_dev, &xai, NULL, &e.mem)) {
                VkMemoryGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = e.mem, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
                if (!getfd(g_dev, &gi, &fd) && fd >= 0) tmp = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (tmp != MAP_FAILED) {
                    memcpy(tmp, (void *)a, n);   // what the pool holds, into the GPU memory
                    void *q = mmap((void *)a, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);   // the pool's pages become the GPU memory, in one step
                    munmap(tmp, n);
                    if (q == (void *)a) {
                        close(fd); e.fdmapped = 1;
                        if (g_fdmap < 0) { g_fdmap = 1; VLOG("memory pools: the driver's memory exported (fd) and mapped over the pool's pages"); }
                        goto bind;
                    }
                    if (g_fdmap > 0) VDIE("memory pool %p+%#llx: mapping the GPU memory over it failed (errno %d)", mem, (unsigned long long)size, errno);
                }
                if (fd >= 0) close(fd);
                vkFreeMemory(g_dev, e.mem, NULL); e.mem = VK_NULL_HANDLE;
            }
            if (g_fdmap > 0) VDIE("memory pool %p+%#llx: GPU memory export failed", mem, (unsigned long long)size);
            VLOG("memory pools: the driver's memory cannot be exported and mapped (errno %d): moving its mappings instead (mremap)", errno);
            g_fdmap = 0;   // first pool: fall back for good (the buffer was made for export, which the plain allocation below also accepts)
        }
#endif
#ifdef __linux__
        // The mapping the driver returns must be readable and writable over the whole pool. Adreno hands out the CPU mappings of freed allocations again, and one whose
        // pages a pool had moved away (a placeholder, or a range of the driver's that is gone) crashed the first write here (loading a save, changing areas in Shield).
        // Such an allocation is kept aside (never freed, so it is never handed out again) and another one made
        uintptr_t vs = 0, ve = 0; unsigned long long voff = 0; char vpath[256] = "";
        for (int tries = 0;; tries++) {
            VKCHECK(vkAllocateMemory(g_dev, &mai, NULL, &e.mem));
            VKCHECK(vkMapMemory(g_dev, e.mem, 0, VK_WHOLE_SIZE, 0, &p));
            int rw = 0; vs = ve = 0; voff = 0; vpath[0] = 0; FILE *mf = fopen("/proc/self/maps", "r"); char ln[512];
            while (mf && fgets(ln, sizeof ln, mf)) { unsigned long s0, e0; char pr[8] = "";
                if (sscanf(ln, "%lx-%lx %7s", &s0, &e0, pr) == 3 && (uintptr_t)p >= s0 && (uintptr_t)p < e0) { vs = s0; ve = e0; rw = pr[0] == 'r' && pr[1] == 'w';
                    if (sscanf(ln, "%*lx-%*lx %*7s %llx %*s %*s %255s", &voff, vpath) != 2) vpath[0] = 0;
                    if (!rw && pr[1] == 'w' && !mprotect((void *)s0, e0 - s0, PROT_READ | PROT_WRITE)) {   // write-only (the Android Emulator's goldfish address space): readable too
                        static int said; if (!said++) VLOG("memory pools: the driver maps GPU memory write-only (%s): made readable, and every pool mirrored (moving these mappings loses their pages)", pr);
                        rw = 1; g_vk_wo_driver = 1; }
                    break; } }
            if (mf) fclose(mf);
            if (!rw) {   // /proc/self/maps does not vouch for it (the Android Emulator's driver mappings under ARM translation): ask the kernel whether its first and last pages
                         // read and write (process_vm_readv / writev of a byte onto itself never faults); if so, use it mirrored (vs = 0 sends it to the mirrored path below)
                uint8_t bt; struct iovec l = { &bt, 1 }, r0 = { p, 1 }, r1 = { (uint8_t *)p + n - 1, 1 }; pid_t me = getpid();
                if (process_vm_readv(me, &l, 1, &r0, 1, 0) == 1 && process_vm_writev(me, &l, 1, &r0, 1, 0) == 1 &&
                    process_vm_readv(me, &l, 1, &r1, 1, 0) == 1 && process_vm_writev(me, &l, 1, &r1, 1, 0) == 1) {
                    static int said; if (!said++) VLOG("memory pools: the driver's mappings are not listed as writable but take writes: mirrored");
                    vs = ve = 0; break; }
            }
            if (rw && ve >= (uintptr_t)p + n) break;
            {   static int nd; if (nd++ < 2) {   // what the process sees there (diagnosis: the Android Emulator's driver)
                    FILE *m2 = fopen("/proc/self/maps", "r"); char l2[512]; int k = 0;
                    while (m2 && fgets(l2, sizeof l2, m2)) { unsigned long s0, e0; if (sscanf(l2, "%lx-%lx", &s0, &e0) == 2 && e0 > (uintptr_t)p - 0x200000 && s0 < (uintptr_t)p + n + 0x200000 && k++ < 12) VLOG("  maps near %p: %s", p, l2); }
                    if (m2) fclose(m2);
                    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
                    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) VLOG("  memory type %u: flags %#x heap %u (%.0f MB)%s", i, mp.memoryTypes[i].propertyFlags, mp.memoryTypes[i].heapIndex, mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size / 1048576.0, (int)i == t ? " (used)" : ""); } }
            VLOG("[RED] memory pool %p+%#llx: the driver's mapping %p is not usable (%s): kept aside, allocating again", mem, (unsigned long long)size, p, rw ? "too short" : "not writable");
            if (tries >= 16) VDIE("memory pool %p+%#llx: no usable GPU memory mapping after %d tries", mem, (unsigned long long)size, tries + 1);
        }
        {   // mremap moves whole mappings: the driver's must start at p, page-aligned, and cover the pool (a driver that sub-allocates from bigger mappings breaks this)
            static int force = -1; if (force < 0) force = getenv("DM_VK_POOL_SHADOW") && atoi(getenv("DM_VK_POOL_SHADOW"));
            if (n >= (16u << 20)) VLOG("memory pool %p+%#llx: driver mapping %p [%#lx-%#lx] offset %#llx %s, write-only driver %d", mem, (unsigned long long)size, p, (unsigned long)vs, (unsigned long)ve, voff, vpath, g_vk_wo_driver);
            if (!force && g_vk_wo_driver && vpath[0] == '/' && !((uintptr_t)p & (getpagesize() - 1))) {
                // The Android Emulator (write-only mappings of /dev/goldfish_address_space): the same device pages mapped once more, through the driver's own open file,
                // over the pool's pages, so the game writes straight into GPU memory. Mirroring needs write faults, which the emulator's ARM translation cannot deliver
                // for every store (an FP store into a protected pool page aborted the process: "Cannot process signal 11")
                uint64_t off = voff + ((uintptr_t)p - vs); int done = 0;
                DIR *dd = opendir("/proc/self/fd"); struct dirent *de;
                while (dd && !done && (de = readdir(dd))) {
                    char lp[64], tgt[256]; snprintf(lp, sizeof lp, "/proc/self/fd/%s", de->d_name); ssize_t k = readlink(lp, tgt, sizeof tgt - 1);
                    if (k <= 0) continue; tgt[k] = 0; if (strcmp(tgt, vpath)) continue;
                    int fd = atoi(de->d_name);
                    volatile uint32_t *t = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)off);
                    if (t == MAP_FAILED) continue;
                    uint32_t was = ((volatile uint32_t *)p)[0], mark = 0x6e786e61u ^ (uint32_t)(uintptr_t)t;   // the same pages? a word written through one shows through the other
                    ((volatile uint32_t *)p)[0] = mark; int same = t[0] == mark; ((volatile uint32_t *)p)[0] = was;
                    munmap((void *)t, n);
                    if (n >= (16u << 20)) VLOG("  fd %d: test mapping %s", fd, same ? "shows the same pages" : "shows other pages");
                    if (!same) continue;
                    memcpy(p, (void *)a, n);   // what the pool holds, into the GPU memory
                    void *q = mmap((void *)a, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, (off_t)off);
                    if (q != (void *)a) VDIE("memory pool %p+%#llx: mapping the GPU memory over it failed (errno %d)", mem, (unsigned long long)size, errno);
                    done = 1;
                }
                if (dd) closedir(dd);
                if (done) { e.fdmapped = 1; static int said; if (!said++) VLOG("memory pools: the driver's pages mapped over the pool's through its own file (%s)", vpath); goto bind; }
                static int nsaid; if (!nsaid++) VLOG("[RED] memory pools: the driver's pages could not be mapped again through its files: mirrored");
            }
            if (force || g_vk_wo_driver || ((uintptr_t)p & (getpagesize() - 1)) || vs != (uintptr_t)p || ve < (uintptr_t)p + n) {
                static int said; if (!said++) VLOG("memory pools: %s: mirrored (pages the game writes copied in before each submission)", force ? "DM_VK_POOL_SHADOW" : "the driver's mappings are not whole mappings of their own");
                e.shadow = p; atomic_fetch_add(&g_nshadow, 1);
                if (!nvn_dirty_tracked((void *)a, n)) { nvn_dirty_add_range((void *)a, n); e.tracked_here = 1; }
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
        // MREMAP_DONTUNMAP leaves the driver's old range mapped (empty) in the same step, so no other thread's mapping can land in a gap there before the placeholder;
        // where the kernel refuses it for this mapping, the plain move and the placeholder right after
#ifndef MREMAP_DONTUNMAP
#define MREMAP_DONTUNMAP 4
#endif
        void *q = mremap(p, n, n, MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP, (void *)a);
        if (q == (void *)a) mmap(p, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);   // the empty old range becomes the placeholder (replaced in one step)
        else {
            static int said; if (!said++) VLOG("pool memory: MREMAP_DONTUNMAP refused (errno %d), moving without it", errno);
            q = mremap(p, n, n, MREMAP_MAYMOVE | MREMAP_FIXED, (void *)a);
            if (q != (void *)a) VDIE("memory pool %p+%#llx: moving the GPU memory onto it failed (errno %d)", mem, (unsigned long long)size, errno);
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
    pthread_mutex_lock(&g_mu);
    if (g_npool == g_cpool) { g_cpool = g_cpool ? g_cpool * 2 : 64; g_pools = realloc(g_pools, g_cpool * sizeof *g_pools); }
    g_pools[g_npool++] = e; g_plast = -1;
    pthread_mutex_unlock(&g_mu);
}
static void vk_finish(void);
static void vk_pool_del(const void *mem) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].key == mem) {
        pool_ent e = g_pools[i]; g_pools[i] = g_pools[--g_npool]; g_plast = -1;
        pthread_mutex_unlock(&g_mu);
        // The game gets its pages back as ordinary memory with the same content (the driver's pages there replaced in one step, never leaving a gap another thread's
        // mapping could take); the GPU buffer and its memory go once the GPU is done with them. The placeholder at the driver's old address is the driver's to unmap
        // (vkFreeMemory): unmapping that range here as well, after the free, raced other threads whose new mappings could already sit there (games free and make pools
        // all the time: 131 times in two minutes of Pokémon Shield's character creation)
        vk_finish();
        if (e.shadow) { atomic_fetch_sub(&g_nshadow, 1); if (e.tracked_here) nvn_dirty_remove_range((void *)e.base); }
        if (e.fdmapped) {   // the game's pages back as ordinary memory with the same content, then a plain free (the driver's own mapping was never touched)
            void *keep = malloc(e.size); memcpy(keep, (void *)e.base, e.size);
            mmap((void *)e.base, e.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            memcpy((void *)e.base, keep, e.size); free(keep);
            vkDestroyBuffer(g_dev, e.buf, NULL); vkFreeMemory(g_dev, e.mem, NULL);
            return;
        }
        int keep_mem = 0;   // the allocation stays when its pages could not go back (the driver would hand out the placeholder again)
        if (e.moved_from) {
            // The driver's pages go back where the driver mapped them (over the placeholder) before the free: Adreno keeps freed allocations with their mappings and hands
            // them out again, so a later pool's vkMapMemory returned the placeholder's address and the first write there crashed (loading a save in Pokémon Shield)
            void *keep = malloc(e.size); memcpy(keep, (void *)e.base, e.size);
#ifdef __linux__
            // one protection over the whole range first: the texture tracking's write protection splits it into several mappings, and mremap moves one (EFAULT otherwise)
            mprotect((void *)e.base, e.size, PROT_READ | PROT_WRITE);
            void *q = mremap((void *)e.base, e.size, e.size, MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP, e.moved_from);
            if (q != e.moved_from) q = mremap((void *)e.base, e.size, e.size, MREMAP_MAYMOVE | MREMAP_FIXED, e.moved_from);   // (then the gap below is filled at once)
            if (q != e.moved_from) { VLOG("[RED] memory pool %p: the GPU memory could not go back to the driver (errno %d): kept, not freed", (void *)e.base, errno); keep_mem = 1; }
            else mprotect(e.moved_from, e.size, PROT_READ | PROT_WRITE);   // pages the texture tracking had write-protected come back writable
#endif
            mmap((void *)e.base, e.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            memcpy((void *)e.base, keep, e.size); free(keep);
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
static int pool_find(uint64_t addr, uint64_t len) {   // caller holds g_mu
    if (g_plast >= 0 && g_plast < g_npool && addr >= g_pools[g_plast].base && addr + len <= g_pools[g_plast].base + g_pools[g_plast].size) return g_plast;
    for (int i = 0; i < g_npool; i++) if (addr >= g_pools[i].base && addr + len <= g_pools[i].base + g_pools[i].size) return g_plast = i;
    return -1;
}
// GPU buffer + offset of [addr, addr + len); 0 when it is in no pool
static int pool_buf(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *left) {
    pthread_mutex_lock(&g_mu); int i = pool_find(addr, len ? len : 1);
    if (i >= 0) { *buf = g_pools[i].buf; *off = addr - g_pools[i].base; if (left) *left = g_pools[i].base + g_pools[i].size - addr; }
    pthread_mutex_unlock(&g_mu);
    return i >= 0;
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
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    VkBuffer b; VkDeviceSize o; if (!pool_buf(addr, len, &b, &o, NULL)) return 0;
    *buf = (void *)b; *off = o; return 1;
}
static void vk_alias_drop_pool(const void *pool);   // textures (section below)

// ================= Submissions =================
// A submission (mtl_sub_begin … mtl_sub_commit) records into its main command buffer; texture uploads and the first-use setup of new images go into its upload command buffer,
// submitted just before it. CPU-visible effects (event writes, counter reports) are collected and applied by the completion thread when the GPU has finished the submission,
// in submission order, before the fence value is published (games wait on either)
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;
typedef struct { VkBuffer buf; vmalloc_t a; uint8_t *map; VkDeviceSize used, size; } arena_blk;
typedef struct {
    VkCommandPool pool; VkCommandBuffer cb, ucb; int ucb_open; uint64_t fence;
    post_op *post; int npost, cpost;
    arena_blk *ar; int nar, car;   // this submission's arena blocks (host-visible): driver segments, converted indices, upload staging
    wb_t *wb; int nwb, cwb;        // mirrored pools: ranges the GPU writes in this submission (copied back to the game's pages at completion)
    VkDescriptorPool *dp; int ndp, cdp;   // g_nopush: this submission's descriptor pools (dp[cdp] in use), reset when it is reused
    uint64_t frame;                 // the frame it belongs to (GPU time per frame, mtl_gpu_ms)
} sub_t;
#define NSUB 16   // (the replay hands a frame to the GPU in pieces, sub_split)
static sub_t g_subs[NSUB]; static sub_t *g_sub; static int g_sub_next;
static void wb_note(uint64_t addr, uint64_t n) {
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

static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) g = mono_ns(); return g * 12 / 625; }
static void wb_apply(sub_t *s) {
    for (int i = 0; i < s->nwb; i++) {
        pthread_mutex_lock(&g_mu); int k = pool_find(s->wb[i].addr, s->wb[i].n); pool_ent e = k >= 0 ? g_pools[k] : (pool_ent){0}; pthread_mutex_unlock(&g_mu);
        if (k < 0 || !e.shadow) continue;
        dm_guest_write_prepare((void *)(uintptr_t)s->wb[i].addr, s->wb[i].n);
        memcpy((void *)(uintptr_t)s->wb[i].addr, e.shadow + (s->wb[i].addr - e.base), s->wb[i].n);
    }
    s->nwb = 0;
}
static void post_apply(sub_t *s) {
    wb_apply(s);
    for (int i = 0; i < s->npost; i++) {
        post_op *o = &s->post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else {   // counter report {value, timestamp}: timestamps on the guest's 19.2 MHz clock; occlusion counts are not measured yet (reported as visible)
            uint64_t t = gpu_ticks();
            { static long div = -1; static uint64_t t0; if (div < 0) { const char *e = getenv("DM_VK_TS_DIV"); div = e ? atol(e) : 0; t0 = t; }   // debugging: counter time runs DM_VK_TS_DIV times slower
              if (div > 1) t = t0 + (t - t0) / (uint64_t)div; }
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
static void *completion_main(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_cmu);
        while (!g_nin) pthread_cond_wait(&g_ccv, &g_cmu);
        sub_t *s = g_inflight[0]; uint64_t want = s->fence;
        pthread_mutex_unlock(&g_cmu);
        VkSemaphoreWaitInfo wi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_tl, .pValues = &want };
        VkResult r = vkWaitSemaphores(g_dev, &wi, 2000000000ull);
        if (r == VK_TIMEOUT) continue;
        if (r) VDIE("vkWaitSemaphores failed: %d (device lost?)", r);
        post_apply(s); gpu_ts_done(s);
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
    VKCHECK(vkCreateSemaphore(g_dev, &sci, NULL, &g_tl));
    for (int i = 0; i < NSUB; i++) {
        VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = g_qf };
        VKCHECK(vkCreateCommandPool(g_dev, &pci, NULL, &g_subs[i].pool));
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_subs[i].pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_subs[i].cb)); VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_subs[i].ucb));
    }
    {   uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
        if (g_qf < nq && qp[g_qf].timestampValidBits && g_props.limits.timestampPeriod > 0 && getenv("DM_VK_GPU_TIME") && atoi(getenv("DM_VK_GPU_TIME"))) {   // on request only (DM_VK_GPU_TIME=1): timestamps may cost on tilers
            VkQueryPoolCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * NSUB };
            if (vkCreateQueryPool(g_dev, &qi, NULL, &g_tsq)) g_tsq = VK_NULL_HANDLE; g_tsper = g_props.limits.timestampPeriod; } }
    pthread_t t; pthread_create(&t, NULL, completion_main, NULL); pthread_detach(t);
}
static int vk_wait(uint64_t fence, uint64_t timeout_ns) {   // a real fence (the timeline); mtl_wait takes the game's (render thread section)
    if (atomic_load(&g_done) >= fence) return 1;
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
static void vk_finish(void) { vk_wait(g_fence, UINT64_MAX); defer_drain(); }

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
        b->a = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
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
static int g_pe_why;   // what ends the open pass (see pass_end)
static int g_bar_clean, g_after_pass, g_game_bar;   // see barrier_all
static int P_tf;       // the open pass is a transform feedback capture's
static struct { uint64_t lo, hi; } g_tfw[64]; static int g_ntfw;   // what the captures in the open capture pass wrote (tf_pass_begin)
static void state_reset(void);
static uint64_t g_frame_no;
// ---------- The replay's own profile (DM_VK_PROF=1): where its time goes, as milliseconds per frame every 300 presents ----------
enum { PF_BUILD, PF_WAIT, PF_DRAW, PF_PSO, PF_PUSH, PF_FLUSH, PF_BEGIN, PF_UPLOAD, PF_DISPATCH, PF_SUBMIT, PF_PRESENT, PF_N };
static const char *const k_pf[PF_N] = { "submission built", "waiting for an old submission", "draws (all)", "  pipeline lookup", "  descriptors", "  descriptor sets",
                                        "  pass begins", "texture uploads", "dispatches", "vkQueueSubmit", "present" };
static uint64_t g_pf[PF_N], g_pfn[PF_N]; static int g_prof = -1;
static inline uint64_t prof_t(void) { if (g_prof < 0) { const char *e = getenv("DM_VK_PROF"); g_prof = e && atoi(e) > 0; } return g_prof ? mono_ns() : 0; }
static inline void prof_add(int k, uint64_t t0) { if (t0) { g_pf[k] += mono_ns() - t0; g_pfn[k]++; } }
static void prof_report(uint64_t n) {
    if (g_prof <= 0 || n % 300) return;
    char line[1400]; size_t o = 0; o += (size_t)snprintf(line + o, sizeof line - o, "[prof] per frame:");
    for (int k = 0; k < PF_N; k++) { o += (size_t)snprintf(line + o, sizeof line - o, " %s %.2f ms (%llu)%s", k_pf[k], g_pf[k] / 300.0 / 1e6, (unsigned long long)(g_pfn[k] / 300), k + 1 < PF_N ? ";" : ""); g_pf[k] = g_pfn[k] = 0; }
    VLOG("%s", line);
}
static uint64_t g_pf_build0, g_sub_t0;
void mtl_sub_begin(void) {
    mtl_init();
    if (g_sub) VDIE("submit re-entered");
    g_sub_thread = pthread_self(); atomic_store(&g_sub_thread_set, 1);
    sub_t *s = &g_subs[g_sub_next]; g_sub_next = (g_sub_next + 1) % NSUB;
    { uint64_t t = prof_t(); if (s->fence) vk_wait(s->fence, UINT64_MAX); prof_add(PF_WAIT, t); }
    defer_drain();
    VKCHECK(vkResetCommandPool(g_dev, s->pool, 0)); arena_reset(s);
    for (int i = 0; i < s->ndp; i++) vkResetDescriptorPool(g_dev, s->dp[i], 0); s->cdp = 0;
    s->npost = 0; s->ucb_open = 0;
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKCHECK(vkBeginCommandBuffer(s->cb, &bi));
    if (g_tsq) { int i = (int)(s - g_subs); s->frame = g_frame_no; unsigned k = s->frame % GRING;
        if (atomic_load(&g_gpu[k].frame) != s->frame) { atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].frame, s->frame); }
        atomic_fetch_add(&g_gpu[k].pending, 1);
        vkCmdResetQueryPool(s->cb, g_tsq, (uint32_t)i * 2, 2); vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, (uint32_t)i * 2); }
    g_sub = s; g_bar_clean = g_after_pass = 0; g_pf_build0 = prof_t(); g_sub_t0 = mono_ns();
    static int inited; if (!inited) { inited = 1; state_reset(); }
}
// Barriers in the submission's command buffer (on Adreno each one waits for the GPU to go idle and flushes its caches):
//   g_bar_clean: the last thing recorded was a full barrier, so a barrier before the next work orders nothing more (it is left out)
//   g_after_pass: the last thing recorded was the end of a draw pass; g_game_bar: the game called nvnCommandBufferBarrier since the last draw pass began.
// Between two draw passes with no game barrier the barrier waits only for what fragment work needs (barrier_pass): the next pass's vertex work and binning overlap the previous
// pass's rendering. NVN asks the game for a barrier before results of rendering are read as vertices, indices, indirect arguments or by a vertex shader, so only those reads
// rely on the game's own barriers (nvn_mtl.m DM_NVN_BARRIER=game). DM_VK_BARRIER=full: a full barrier before every pass as before
static uint64_t g_bar_full, g_bar_narrow, g_bar_none;   // barriers before draw passes (the present line)
static int barrier_relaxed(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_BARRIER"); v = !(e && !strcmp(e, "full")); if (!v) VLOG("DM_VK_BARRIER=full: a full barrier before every pass"); } return v; }
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
// The marker of a submission that failed with imported pools (see mtl_init): set = 1 writes it, 0 tells whether it is there
static int host_import_mark(int set) {
    const char *pc = getenv("DM_VK_PCACHE"); if (!pc) return 0;
    char m[1100]; snprintf(m, sizeof m, "%s", pc); char *sl = strrchr(m, '/'); if (!sl) return 0; snprintf(sl + 1, sizeof m - (size_t)(sl + 1 - m), "host_import_failed");
    if (!set) return access(m, F_OK) == 0;
    FILE *f = fopen(m, "w"); if (f) { fputs("vkQueueSubmit failed with imported host memory: the next start moves driver memory onto the pools instead\n", f); fclose(f); }
    return f != NULL;
}
static int mirror_ordered(void);
static void queue_submit_impl(VkCommandBuffer *cbs, uint32_t n, uint64_t signal);
static void queue_submit(VkCommandBuffer *cbs, uint32_t n, uint64_t signal) { uint64_t pf_t0 = prof_t(); queue_submit_impl(cbs, n, signal); prof_add(PF_SUBMIT, pf_t0); }
static void queue_submit_impl(VkCommandBuffer *cbs, uint32_t n, uint64_t signal) {
    VkTimelineSemaphoreSubmitInfo ts = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal };
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ts, .commandBufferCount = n, .pCommandBuffers = cbs, .signalSemaphoreCount = 1, .pSignalSemaphores = &g_tl };
    if (!mirror_ordered()) shadow_sync();   // (ordered: the submission's own command buffer copies them, shadow_sync_ordered)
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_qmu);
    if (r && g_host_import && host_import_mark(1)) VLOG("[RED] vkQueueSubmit failed with imported host memory: the next start will not import it");
    if (r) VDIE("vkQueueSubmit failed: %d", r);
}
// Fence values are handed out under g_cmu so the present thread's submissions (screenshots) and the game's stay in order on the timeline
static uint64_t submit_tracked(sub_t *s, VkCommandBuffer *cbs, uint32_t n) {
    pthread_mutex_lock(&g_cmu);
    uint64_t f = ++g_fence; s->fence = f;
    queue_submit(cbs, n, f);
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
uint64_t mtl_sub_commit(void) {
    sub_t *s = g_sub; if (!s) return g_fence;
    prof_add(PF_BUILD, g_pf_build0); g_pf_build0 = 0;
    g_pe_why = 9; pass_end();
    shadow_sync_ordered(s);   // (into the upload command buffer, which runs before this submission's work)
    if (g_tsq) vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, (uint32_t)(s - g_subs) * 2 + 1);
    VKCHECK(vkEndCommandBuffer(s->cb));
    VkCommandBuffer cbs[2]; uint32_t n = 0;
    if (s->ucb_open) { barrier_all(s->ucb); VKCHECK(vkEndCommandBuffer(s->ucb)); cbs[n++] = s->ucb; s->ucb_open = 0; }
    cbs[n++] = s->cb;
    g_sub = NULL;
    return submit_tracked(s, cbs, n);
}
static void post(post_op o) {
    sub_t *s = g_sub; if (!s) { RED_ONCE(1, "event / counter outside a submission"); return; }
    if (s->npost == s->cpost) { s->cpost = s->cpost ? s->cpost * 2 : 64; s->post = realloc(s->post, s->cpost * sizeof *s->post); }
    s->post[s->npost++] = o;
}
// One-off command buffer for work outside a submission (uploads requested off the submit thread, readbacks): recorded, submitted and waited for at once
static VkCommandPool g_once_pool; static pthread_mutex_t g_once_mu = PTHREAD_MUTEX_INITIALIZER;
static VkCommandBuffer once_begin(void) {
    pthread_mutex_lock(&g_once_mu);
    if (!g_once_pool) { VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf }; VKCHECK(vkCreateCommandPool(g_dev, &pci, NULL, &g_once_pool)); }
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_once_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; VKCHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKCHECK(vkBeginCommandBuffer(cb, &bi)); barrier_all(cb);
    return cb;
}
static void once_end(VkCommandBuffer cb) {   // ordered after everything submitted so far (same queue, full barriers), waited for here
    barrier_all(cb); VKCHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &f));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    shadow_sync();
    pthread_mutex_lock(&g_qmu); VkResult r = vkQueueSubmit(g_q, 1, &si, f); pthread_mutex_unlock(&g_qmu);
    if (r) VDIE("one-off submit failed: %d", r);
    vkWaitForFences(g_dev, 1, &f, VK_TRUE, UINT64_MAX); vkDestroyFence(g_dev, f, NULL);
    vkFreeCommandBuffers(g_dev, g_once_pool, 1, &cb);
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
    int drop;                              // mip levels left out (texture quality half: the image is the game's levels from this one on)
    int fixed_fmt; VkFormat twin;          // 1: not mutable (depth / stencil), 2: mutable within fmt / twin only (tex_make: keeps UBWC)
    struct { uint64_t key, frame; uint16_t lv, ly; uint8_t asp, dc; } sk[8]; int nsk; int presented;   // stores of its attachments not yet read or overwritten (sk_*); shown in the window
} vtex;
// ================= Stores no one reads (tilers: Adreno, Mali) =================
// On a tiler every render pass ends by writing its attachments from tile memory to the image, and a pass whose next user clears the image first or never looks at it again wrote
// for nothing (depth buffers above all). Vulkan needs the store decision when the pass begins, before its commands are known, so it is learnt: each attachment of a pass is keyed
// by the pass's render targets, how many times that set was used before in the frame and the slot, and its store stays pending on the image until the image is next read (sampled,
// copied, loaded by a pass, shown) or overwritten (a pass clearing it). Only an overwrite in the same frame counts: what lasts into the next frame may be the game's history or a
// cache it redraws only sometimes (entering the field from the loading screen, Legends: Arceus starts reading a dozen such images, among them shadow maps and its environment cube,
// which every loading frame had cleared). A key whose store was overwritten unread in its frame SK_LEARN times in a row stores nothing from then on (DONT_CARE);
// if one of its images is then read, the content was undefined for that one use, and the key stores again for good (reported once). Shown images never skip.
// DM_VK_SKIP_STORES=0 turns it off
enum { SK_LEARN = 90, SK_N = 4096 };
typedef struct { uint64_t key; uint16_t streak, ro; uint8_t off, rooff; } skey_t;   // ro: passes in a row that only read this depth / stencil attachment (sk_ro)
static skey_t g_sk[SK_N]; static int g_nsk_keys;
static pthread_mutex_t g_skmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_sk_skipped, g_sk_wrong;   // stores skipped, skipped stores read after all (the present line)
static int sk_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SKIP_STORES"); v = !(e && !strcmp(e, "0")); if (!v) VLOG("DM_VK_SKIP_STORES=0: every pass stores its attachments"); } return v; }
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
static void sk_read(vtex *x) { if (!x || !x->nsk) return; pthread_mutex_lock(&g_skmu); sk_resolve_locked(x, ~0u, ~0u, 0, 0); pthread_mutex_unlock(&g_skmu); }
// A pass begins with x's subresource as attachment slot (0-7 colour, 8 depth, 9 stencil), cleared or loaded: what was pending resolves, and whether this pass stores comes back
static uint64_t g_sk_sig, g_sk_last;   // the pass's key so far (pass_begin); the key sk_attach made last
static int sk_attach(vtex *x, uint32_t lv, uint32_t ly, int asp, int slot, int cleared) {
    pthread_mutex_lock(&g_skmu);
    sk_resolve_locked(x, lv, ly, asp, cleared);
    int dc = 0;
    uint64_t key = (g_sk_sig ^ (uint64_t)(slot + 1) * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull; key ^= key >> 31;
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
enum { SK_RO_LEARN = 4 };
static int sk_ro(uint64_t key, int cleared) {   // whether the attachment sk_attach just keyed begins with store op NONE
    if (cleared || !sk_on()) return 0;
    pthread_mutex_lock(&g_skmu); skey_t *k = sk_get(key); int r = k && !k->rooff && k->ro >= SK_RO_LEARN; pthread_mutex_unlock(&g_skmu);
    return r;
}
static void sk_ro_note(uint64_t key, int wrote, int violated) {   // at the end of the pass
    pthread_mutex_lock(&g_skmu); skey_t *k = sk_get(key);
    if (k) { if (violated) k->rooff = 1; if (wrote) k->ro = 0; else if (k->ro < 60000) k->ro++; }
    pthread_mutex_unlock(&g_skmu);
}

static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;   // serial / sharing / CPU-written tables
static nvn_hmap g_by_serial = {.klen = 8}, g_ghost = {.klen = 8}, g_tex_live = {.klen = 8}, g_cpu_tex = {.klen = 8};
enum { NGHOST = 512 }; static uint64_t g_ghost_ring[NGHOST]; static unsigned g_ghost_next;
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } alias_key;
static nvn_hmap g_alias = {.klen = sizeof(alias_key)};
static _Atomic uint64_t g_ser_gen = 1;
static uint64_t g_pkg_uploads, g_pkg_reuploads, g_up_bytes;

static void vtex_ref(vtex *t) { if (t) atomic_fetch_add(&t->refs, 1); }
static atomic_long g_live_tex, g_live_tex_bytes;   // memory accounting (mem_report)
static void vtex_unref(vtex *t) {
    if (!t || atomic_fetch_sub(&t->refs, 1) != 1) return;
    atomic_fetch_sub(&g_live_tex, 1); atomic_fetch_sub(&g_live_tex_bytes, (long)t->a.size);
    for (int i = 0; i < t->nviews; i++) defer_add((dfr){ .kind = DF_VIEW, .h.view = t->views[i].v });
    if (t->bview) defer_add((dfr){ .kind = DF_BUFVIEW, .h.bview = t->bview });
    if (t->img) defer_add((dfr){ .kind = DF_IMAGE, .h.img = t->img, .a = t->a });
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
static VkClearColorValue clear_for(VkFormat f, const float rgba[4]) {
    VkClearColorValue v; int k = fmt_int_class(f);
    for (int i = 0; i < 4; i++) { if (k == 1) v.uint32[i] = rgba[i] <= 0 ? 0 : (uint32_t)rgba[i]; else if (k == 2) v.int32[i] = (int32_t)rgba[i]; else v.float32[i] = rgba[i]; }
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
        if (t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue c = {{0}}; vkCmdClearColorImage(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
        else { VkClearDepthStencilValue c = {0, 0}; vkCmdClearDepthStencilImage(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
    }
    if (once) once_end(cb);
}
// Texture quality (DM_VK_TEX_QUALITY=half, the device profile's choice for weaker GPUs): the game's own block-compressed textures of 512x512 and up with mip levels are kept
// without their largest level, a quarter of the memory and bandwidth (and of the decode work on GPUs without BC). Only textures from the game's files: they are never drawn
// into, so render targets and everything the screen is made of keep their size. Views shift their levels down by the same count; copies into or out of such a texture are
// left out
static uint64_t g_drop_n, g_drop_saved;   // textures stored without their largest level, bytes not stored (the memory readout)
static int tex_quality_half(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_TEX_QUALITY"); v = e && !strcmp(e, "half"); if (v) VLOG("texture quality half: the game's large textures without their largest mip level"); }
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
        case 2: ci.imageType = VK_IMAGE_TYPE_3D; ci.extent.depth = t->d.depth ? t->d.depth : 1; ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT; x->is3d = 1; break;   // (2D-array compatible: rendered into slice by slice, att_view; from the Galar / Paldea engine)
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
    if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (fmt_compressed(t->d.format) && !nvn_fmt_vk_decode(t->d.format)) ci.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    // Mutable formats switch off Adreno's framebuffer compression (UBWC) unless the formats are listed: depth / stencil images are not mutable at all (their views keep
    // the format), colour images that can be render targets list their format and its sRGB / linear twin (the reinterpretations NVN views make of render targets)
    static int nolist = -1; if (nolist < 0) nolist = getenv("DM_VK_NO_FMT_LIST") != NULL;
    VkFormat twin = x->fmt == VK_FORMAT_R8G8B8A8_UNORM ? VK_FORMAT_R8G8B8A8_SRGB : x->fmt == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM :
                    x->fmt == VK_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_SRGB : x->fmt == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : x->fmt;
    VkFormat fl[2] = { x->fmt, twin }; VkImageFormatListCreateInfo fli = { .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = twin != x->fmt ? 2 : 1, .pViewFormats = fl };
    if (!nolist && !x->is3d) {
        if (fmt_is_depth(x->fmt) || x->fmt == VK_FORMAT_S8_UINT) { ci.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; x->fixed_fmt = 1; }
        else if (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) { ci.pNext = &fli; x->fixed_fmt = 2; x->twin = twin; }
    }
    x->w = ci.extent.width; x->h = ci.extent.height; x->d = ci.extent.depth; x->levels = ci.mipLevels; x->layers = ci.arrayLayers; x->samples = ci.samples; x->usage = ci.usage; x->aspect = fmt_aspect(x->fmt);
    VkResult r = vkCreateImage(g_dev, &ci, NULL, &x->img);
    if (r) { RED_ONCE(3, "vkCreateImage %ux%ux%u format %#x (Vulkan %u) target %d failed: %d", x->w, x->h, x->d, t->d.format, vf, t->d.target, r); free(x); return NULL; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, x->img, &rq);
    x->a = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); atomic_fetch_add(&g_live_tex_bytes, (long)x->a.size);
    VKCHECK(vkBindImageMemory(g_dev, x->img, x->a.b->mem, x->a.off));
    return x;
}
static VkImageView view_get(vtex *t, VkImageViewType vt, VkFormat fmt, uint32_t lvl, uint32_t nlvl, uint32_t lay, uint32_t nlay, const uint8_t sw[4], VkImageAspectFlags aspect) {
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

// CPU-filled texture upload: deswizzled level by level / layer by layer into staging (the submission's arena, else a one-off buffer), copied into the image in the upload command buffer
static int tex_upload_impl(vtex *x, const tex_t *t);
static int tex_upload(vtex *x, const tex_t *t) { uint64_t pf_t0 = prof_t(); int r = tex_upload_impl(x, t); prof_add(PF_UPLOAD, pf_t0); return r; }
static int tex_upload_impl(vtex *x, const tex_t *t) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) return -1;
    // BC on a GPU without it: the blocks are deswizzled into scratch and decoded into staging as plain texels (bc_decode.c)
    int bck = nvn_fmt_vk_decode(t->d.format), obpb = bck ? bc_texel_bytes(bck) : bpb, obw = bck ? 1 : bw, obh = bck ? 1 : bh; uint8_t *scratch = NULL;
    unsigned layers = x->is3d ? 1 : x->layers;
    VkDeviceSize total = 0;
    for (unsigned l = 0; l < x->levels; l++) { uint32_t w = x->w >> l ? x->w >> l : 1, h = x->h >> l ? x->h >> l : 1, z = x->is3d ? (x->d >> l ? x->d >> l : 1) : 1;
        total += (VkDeviceSize)((w + obw - 1) / obw) * obpb * ((h + obh - 1) / obh) * z * layers; }
    VkCommandBuffer cb = ucb(); int once = !cb; if (once) cb = once_begin();
    VkBuffer sb; VkDeviceSize so = 0; uint8_t *sp; vmalloc_t tmp = {0};
    if (!once) sp = arena_alloc(g_sub, total ? total : 16, &sb, &so);
    else { VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT }; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &sb));
           VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, sb, &rq); tmp = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
           VKCHECK(vkBindBufferMemory(g_dev, sb, tmp.b->mem, tmp.off)); sp = (uint8_t *)tmp.b->map + tmp.off; }
    VkBufferImageCopy *rg = calloc(x->levels * layers, sizeof *rg); uint32_t nrg = 0; VkDeviceSize off = 0; int linear = !!(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET));
    for (unsigned l = 0; l < x->levels; l++) {
        uint32_t w = x->w >> l ? x->w >> l : 1, h = x->h >> l ? x->h >> l : 1, z = x->is3d ? (x->d >> l ? x->d >> l : 1) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh, opitch = (size_t)((w + obw - 1) / obw) * obpb, orows = (h + obh - 1) / obh;
        if (bck) { free(scratch); scratch = malloc(pitch * rows * z); }
        for (unsigned y = 0; y < layers; y++) {
            uint8_t *dst = bck ? scratch : sp + off;
            if (linear) {   // LINEAR: rows at the texture's stride (one level, one layer)
                size_t stride = t->d.stride ? t->d.stride : pitch; for (size_t r = 0; r < rows; r++) memcpy(dst + r * pitch, t->mem + r * stride, pitch);
            } else if (nvn_deswizzle(&t->d, l + (unsigned)x->drop, y, t->mem, dst, pitch)) { RED_ONCE(5, "deswizzle failed format %#x target %d", t->d.format, t->d.target); break; }
            if (bck) for (uint32_t s = 0; s < z; s++) bc_decode_image(bck, scratch + s * pitch * rows, pitch, (int)w, (int)h, sp + off + s * opitch * orows, opitch);
            rg[nrg++] = (VkBufferImageCopy){ .bufferOffset = so + off, .bufferRowLength = (uint32_t)(opitch / obpb * obw), .bufferImageHeight = (uint32_t)(orows * obh),
                .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, l, x->is3d ? 0 : y, 1 }, .imageExtent = { w, h, z } };
            off += opitch * orows * z;
        }
    }
    free(scratch);
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .image = x->img, .subresourceRange = { x->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    if (nrg) vkCmdCopyBufferToImage(cb, sb, x->img, VK_IMAGE_LAYOUT_GENERAL, nrg, rg);
    free(rg); g_up_bytes += total;
    { static int z = -1; if (z < 0) z = getenv("DM_VK_BC6_ZERO") != NULL; if (z && (t->d.format == 0x4F || t->d.format == 0x50)) memset(sp, 0, (size_t)total); }   // debugging: BC6H textures black
    { static const char *bd = (const char *)1; if (bd == (const char *)1) bd = getenv("DM_VK_BC6DUMP");   // debugging: level 0 of each BC6H texture, as uploaded, to <dir>
      if (bd && (t->d.format == 0x4F || t->d.format == 0x50) && !x->is3d) { char p[1024]; snprintf(p, sizeof p, "%s/bc6_%llu_%ux%u_l%u_f%x.bin", bd, (unsigned long long)t->serial, x->w, x->h, layers, t->d.format);
        FILE *f = fopen(p, "wb"); if (f) { fwrite(sp, 1, (size_t)(((x->w + 3) / 4) * 16) * ((x->h + 3) / 4) * layers, f); fclose(f); } } }
    if (once) { once_end(cb); vkDestroyBuffer(g_dev, sb, NULL); vk_free(tmp); }
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
static void vk_tex_release(const tex_t *t) {
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
static void vk_alias_drop_pool(const void *pool) {
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
static vtex *tex_res(const tex_t *t) {
    if (!t || !t->serial) return NULL;
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
        vtex *x = calloc(1, sizeof *x); atomic_fetch_add(&g_live_tex, 1); x->refs = 1; x->fmt = (VkFormat)vf; x->nvn_fmt = t->d.format; pthread_mutex_init(&x->vmu, NULL);
        VkDeviceSize n = (VkDeviceSize)t->d.width * bpb; if (n > left) n = left;
        VkBufferViewCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = b, .format = x->fmt, .offset = o, .range = n };
        if (vkCreateBufferView(g_dev, &bci, NULL, &x->bview)) { RED_ONCE(9, "buffer texture view failed (format %u, %llu bytes)", vf, (unsigned long long)n); free(x); pthread_mutex_unlock(&g_smu); return NULL; }
        serial_put(t->serial, x); vtex_unref(x); pthread_mutex_unlock(&g_smu); return x;
    }
    alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d;
    vtex *shared = nvn_hm_get(&g_alias, &ak);
    // No exact match: a texture object over the same memory with the same size and format but other flags / levels (a render target sampled through a second texture object
    //   made without the render-target flags: as a new image it read black; Violet's 3D scene and ID photo, the Galar / Paldea engine)
    if (!shared) for (uint32_t i = 0; i < g_alias.cap && !shared; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != ak.pool || k.off != ak.off || k.d.width != t->d.width || k.d.height != t->d.height || k.d.depth != t->d.depth || k.d.format != t->d.format || k.d.target != t->d.target) continue;
        vtex *c = (vtex *)g_alias.vals[i]; if (!(c->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) continue;
        shared = c; static int said; if (!said++) VLOG("texture over a render target's memory with other flags (%#x vs %#x, %ux%u format %#x): sharing its image", t->d.flags, k.d.flags, t->d.width, t->d.height, t->d.format);
    }
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
void mtl_set_tex_pool_n(int n) { g_tpool_n = n; }
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    if (tid < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (tid >= g_ntreg) { int n = tid + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_treg[tid].ok = t != NULL; if (t) g_treg[tid].t = *t; g_treg[tid].has_v = v != NULL; if (v) g_treg[tid].v = *v;
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
    switch (w) { case 0: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
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
    // anisotropic filtering capped (DM_VK_MAX_ANISO, default 8): above 8x the texture units fetch far more for surfaces at a glancing angle (ground, roads) for little visible gain
    { static float cap = -1; if (cap < 0) { const char *e = getenv("DM_VK_MAX_ANISO"); cap = e ? (float)atof(e) : 8.0f; if (cap < 1) cap = 1; }
      float want = p->aniso > cap ? cap : p->aniso;
      if (want > 1) { ci.anisotropyEnable = VK_TRUE; ci.maxAnisotropy = want > g_props.limits.maxSamplerAnisotropy ? g_props.limits.maxSamplerAnisotropy : want; } }
    if (p->cmp_mode) { ci.compareEnable = VK_TRUE; ci.compareOp = (VkCompareOp)(p->cmp_func - 1); }
    VkSamplerReductionModeCreateInfo rd = { .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, .reductionMode = p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX };
    VkSamplerCustomBorderColorCreateInfoEXT bc = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, .format = VK_FORMAT_UNDEFINED };
    const void **tail = &ci.pNext;
    if (p->reduction == 1 || p->reduction == 2) { *tail = &rd; tail = &rd.pNext; }
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
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = x->img, .subresourceRange = { x->aspect, 0, 1, 0, layers } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        VkImageSubresourceRange r = { x->aspect, 0, 1, 0, layers };
        if (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue c = {{0}}; vkCmdClearColorImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); }
        else { VkClearDepthStencilValue c = {0, 0}; vkCmdClearDepthStencilImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); } }
    once_end(cb);
    return x;
}
static VkBuffer buf_make(VkDeviceSize n, vmalloc_t *a, void **map) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = BUF_ALL }; VkBuffer b; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &b));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, b, &rq); *a = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1);
    VKCHECK(vkBindBufferMemory(g_dev, b, a->b->mem, a->off)); if (map) *map = (uint8_t *)a->b->map + a->off; return b;
}
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
static int tex_is_int(int tid) {   // the texture ID's view format is an integer one
    if (tid <= 0 || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    treg_t rg = g_treg[tid]; vtex *x = tex_res(&rg.t); if (!x) return 0;
    VkFormat f = x->fmt; if (rg.has_v && rg.v.set_format) { uint32_t vf = 0; uint8_t s2[4]; if (nvn_fmt_to_vk(rg.v.format, &vf, s2) && vf) f = (VkFormat)vf; }
    return fmt_int_class(f) != 0;
}
static VkImageView tex_view_for(int tid, const spv_res *r, int storage) {
    if (g_tpool_n && tid >= g_tpool_n) return VK_NULL_HANDLE;
    if (tid <= 0 || tid >= g_ntreg || !g_treg[tid].ok) { if (tid > 0) RED_ONCE(11, "texture ID %d not registered", tid); return VK_NULL_HANDLE; }
    treg_t rg = g_treg[tid];
    vtex *x = tex_res(&rg.t); if (!x || !x->img) return VK_NULL_HANDLE;
    sk_read(x);   // sampled or bound as an image: its pending stores were needed
    VkFormat f = x->fmt; uint8_t fsw[4]; memcpy(fsw, x->fswz, 4);
    if (rg.has_v && rg.v.set_format) { uint32_t vf = 0; uint8_t s2[4]; if (nvn_fmt_to_vk(rg.v.format, &vf, s2) && vf) { f = (VkFormat)vf; memcpy(fsw, s2, 4); } }
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
    VkImageAspectFlags asp = x->aspect;
    if (asp & VK_IMAGE_ASPECT_DEPTH_BIT) asp = rg.has_v && rg.v.set_dsmode && rg.v.dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && fmt_has_stencil(x->fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (storage) { if (!(x->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { RED_ONCE(16, "texture ID %d bound as an image lacks storage usage", tid); return VK_NULL_HANDLE; } memcpy(vk, SW_ID, 4); }
    return view_get(x, vt, f, l0, nl, s0, ns, vk, asp);
}

// ================= Shaders and programs =================
enum { W_NONE, W_SUPPORT, W_C1, W_DRV0, W_DRV2, W_CBUF, W_SSBO, W_UBOP, W_SBUFU, W_TEX, W_TEXU, W_IMG, W_TBUF };
enum { S_NONE, S_TEX, S_SEP, S_UBO, S_ID };   // where a combined texture's sampler comes from: the texture handle's high bits, a separate sampler slot, a handle in a uniform buffer, a constant ID
typedef struct { uint8_t what, skind, rw; uint16_t reg, off, sreg, soff; } vres_how;
typedef struct vshader {
    char key[24]; int stage; VkShaderStageFlagBits vkstage; VkShaderModule mod; uint32_t shift;
    spv_res res[64]; vres_how how[64]; int nres;
    spv_input in[32]; int nin;
    uint8_t drv_read[3]; uint8_t color_mask, color_known;
    VkBuffer c1buf; VkDeviceSize c1off, c1len;
    uint64_t in_mask, out_mask;                 // user varyings by location (spv_io_masks)
    uint32_t *code; size_t nw;                  // fragment shaders: the (rebound) SPIR-V, for variants
    uint8_t out_cls[8];                         // fragment outputs by location: 0 float, 1 uint, 2 sint, 255 none
    struct { uint64_t drop; uint32_t um, sm; VkShaderModule mod; } var[8]; int nvar;   // variants: inputs the previous stage never writes (drop), float outputs to integer targets (um / sm)
    int tf[4][2]; uint16_t tf_stride[4]; VkShaderModule tfmod;   // transform feedback: per buffer { output location, components } and stride; the capture variant (tf_draw)
} vshader;
static char g_spv_dir[1024];
static nvn_hmap g_shmap = {.klen = 8};      // shader key (8 bytes) → stage + 1 (as a pointer)
static nvn_hmap g_shcache = {.klen = 8};    // shader key → vshader
static int stage_index(const char *s) { return !strcmp(s, "vertex") ? 0 : !strcmp(s, "pixel") ? 1 : !strcmp(s, "compute") ? 5 : !strcmp(s, "hull") ? 3 : !strcmp(s, "domain") ? 4 : !strcmp(s, "geometry") ? 2 : -1; }
static int hexkey(const char *s, uint8_t k[8]) { for (int i = 0; i < 8; i++) { unsigned v; if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1; k[i] = (uint8_t)v; } return 0; }
static void shmap_read(void);
static void shmap_load(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, shmap_read); }   // the game thread (mtl_shader_seen) and the replay both ask
static void shmap_read(void) {
    const char *e = getenv("DM_NVN_SHMAP"); FILE *f = e ? fopen(e, "r") : NULL;
    if (!f) VDIE("cannot read the shader map (DM_NVN_SHMAP=%s)", e ? e : "(unset)");
    char line[1024]; int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || !strncmp(line, "package", 7)) continue;
        char *c[5] = {0}; int nc = 0; for (char *p = line; nc < 5 && p; nc++) { c[nc] = p; p = strchr(p, '\t'); if (p) *p++ = 0; }
        if (nc < 4) continue; char *nl = strpbrk(c[3], "\r\n"); if (nl) *nl = 0;
        uint8_t k[8]; int st = stage_index(c[2]); if (st < 0 || strlen(c[3]) < 16 || hexkey(c[3], k)) continue;
        nvn_hm_put(&g_shmap, k, (void *)(intptr_t)(st + 1)); n++;
    }
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
// The shader set packed into one mapped file (DM_SPV_PAK, dm/android/tools/spv_pak.py): "NXSP", version, count, chain[16], sorted {key[8], spv off/len, vkr off/len}, data.
// Version 2 stores each blob as u32 size + a zlib stream (a quarter of the size)
static const uint8_t *g_pak; static size_t g_pak_n; static uint32_t g_pak_count, g_pak_ver;
static int pak_read(void);
static void pak_read_once(void) { pak_read(); }
static int pak_open(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, pak_read_once); return g_pak != NULL; }   // the game thread (pak_has) and the replay both ask
static int pak_read(void) {
    const char *path = getenv("DM_SPV_PAK"); if (!path) return 0;
    int fd = open(path, O_RDONLY); struct stat st; if (fd < 0 || fstat(fd, &st)) { VLOG("[RED] shader pack %s cannot be opened", path); if (fd >= 0) close(fd); return 0; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED || st.st_size < 28 || memcmp(m, "NXSP", 4)) { VLOG("[RED] shader pack %s is not one", path); return 0; }
    char chain[17] = {0}; memcpy(chain, (const uint8_t *)m + 12, 16);
    if (strcmp(chain, DMSH_CHAIN)) { VLOG("[RED] shader pack %s has chain %s, this host expects %s", path, chain, DMSH_CHAIN); munmap(m, (size_t)st.st_size); return 0; }
    g_pak = m; g_pak_n = (size_t)st.st_size; memcpy(&g_pak_ver, g_pak + 4, 4); memcpy(&g_pak_count, g_pak + 8, 4);
    if (g_pak_ver != 1 && g_pak_ver != 2) { VLOG("[RED] shader pack %s has version %u", path, g_pak_ver); g_pak = NULL; munmap(m, (size_t)st.st_size); return 0; }
    VLOG("shader pack %s: %u shaders%s", path, g_pak_count, g_pak_ver == 2 ? " (compressed)" : "");
    return 1;
}
static void *pak_blob(const uint8_t *p, size_t n, size_t *out) {   // a malloc'd copy of one blob, inflated in a version 2 pack
    if (g_pak_ver != 2) { void *b = malloc(n ? n : 1); memcpy(b, p, n); *out = n; return b; }
    if (n < 4) return NULL;
    uint32_t raw; memcpy(&raw, p, 4); uLongf dl = raw; void *b = malloc(raw ? raw : 1);
    if (uncompress(b, &dl, p + 4, (uLong)(n - 4)) != Z_OK || dl != raw) { free(b); return NULL; }
    *out = raw; return b;
}
static int pak_get(const uint8_t key[8], uint32_t **spv, size_t *ns, char **vkr, size_t *nr) {
    if (!pak_open()) return 0;
    uint32_t lo = 0, hi = g_pak_count;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; const uint8_t *e = g_pak + 28 + 32 * (size_t)m; int c = memcmp(e, key, 8);
        if (!c) { uint64_t so, ro; uint32_t sl, rl; memcpy(&so, e + 8, 8); memcpy(&sl, e + 16, 4); memcpy(&ro, e + 20, 8); memcpy(&rl, e + 28, 4);
                  if (so + sl > g_pak_n || ro + rl > g_pak_n) return 0;
                  *spv = pak_blob(g_pak + so, sl, ns); *vkr = pak_blob(g_pak + ro, rl, nr);
                  if (!*spv || !*vkr) { free(*spv); free(*vkr); *spv = NULL; *vkr = NULL; return 0; }
                  return 1; }
        if (c < 0) lo = m + 1; else hi = m; }
    return 0;
}
// Whether the pack holds a key (no copy: mtl_shader_seen asks for every shader the game hands over). Never opens the pack: the game thread hands shaders over before the
// APK's files can be read (/nxapk), and a failed open is not retried (every shader of the run was lost, a black screen); until the replay has opened it, the pack is taken to hold it
static int pak_has(const uint8_t key[8]) {
    if (!__atomic_load_n(&g_pak, __ATOMIC_ACQUIRE)) return 1;
    uint32_t lo = 0, hi = g_pak_count;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; int c = memcmp(g_pak + 28 + 32 * (size_t)m, key, 8); if (!c) return 1; if (c < 0) lo = m + 1; else hi = m; }
    return 0;
}
// Shaders the game builds in memory, or that the shipped shader map does not list, match nothing in the bundled pack: their draws were skipped (characters and move effects
// glitching or freezing, as in Pokémon Shield's dev build). The first time one is handed over (nvnProgramSetShaders) and the pack lacks it, its 0x880-byte control section and its
// code area are saved under DM_NVN_CAPTURE, to be translated on the PC and added to the pack. At most 512 shaders the map does not know a run
void mtl_shader_seen(const uint8_t key[8], const void *control, const void *code, uint32_t ncode) {
    static const char *dir; static int init, unmapped; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static nvn_hmap seen = {.klen = 8};
    if (!init) { dir = getenv("DM_NVN_CAPTURE"); if (dir) mkdir(dir, 0755); init = 1; }
    if (!dir || !control || !code || !ncode || ncode >= (1u << 24)) return;
    pthread_mutex_lock(&mu);
    if (!nvn_hm_get(&seen, key)) {
        nvn_hm_put(&seen, key, (void *)1);
        shmap_load();
        int mapped = nvn_hm_get(&g_shmap, key) != NULL;
        if ((!mapped && unmapped < 512) || (mapped && !pak_has(key))) {
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
static vshader *shader_load(const uint8_t key[8], int st) {
    char hex[17]; for (int i = 0; i < 8; i++) sprintf(hex + 2 * i, "%02x", key[i]);
    char p[1200]; size_t ns = 0, nr = 0; uint32_t *code = NULL; char *meta = NULL;
    if (pak_get(key, &code, &ns, &meta, &nr)) {}
    else {
        snprintf(p, sizeof p, "%s/%s.spv", g_spv_dir, hex); code = file_read(p, &ns);
        snprintf(p, sizeof p, "%s/%s.vkr", g_spv_dir, hex); meta = file_read(p, &nr);
    }
    if (!code || !meta) { VLOG("[RED] shader %s: %s missing", hex, !code ? "SPIR-V" : "reflection (.vkr)"); free(code); free(meta); return NULL; }
    meta = realloc(meta, nr + 1); meta[nr] = 0;
    vshader *s = calloc(1, sizeof *s); strcpy(s->key, hex); s->stage = st; s->shift = stage_shift(st);
    int spst; uint32_t ls[3];
    if (spv_reflect(code, ns / 4, s->res, 64, &s->nres, s->in, 32, &s->nin, &spst, ls)) { VLOG("[RED] shader %s: SPIR-V not readable", hex); free(code); free(meta); free(s); return NULL; }
    static const VkShaderStageFlagBits VS[6] = { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT };
    s->vkstage = spst >= 0 && spst < 6 ? VS[spst] : VK_SHADER_STAGE_VERTEX_BIT;
    // reflection: bindings by name (the combined image takes its texture slot from "<name>" and its sampler from "<name>Smplr"), c1, driver segment reads
    struct { char name[48], kind[16]; int reg, off, bank, rw, n; } b[96]; int nb = 0; uint8_t *c1 = NULL; uint32_t c1n = 0; int unfill = 0;
    for (char *ln = strtok(meta, "\n"); ln; ln = strtok(NULL, "\n")) {
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
    { size_t nw; uint32_t *c2 = spv_image_caps(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }   // and the format-less storage image ones they need
    if (spst == 0 || spst == 2) { size_t nw; uint32_t *c2 = spv_init_outputs(code, ns / 4, &nw); if (c2) { free(code); code = c2; ns = nw * 4; } }   // outputs start at (0, 0, 0, 1) as on the hardware
    if (st == 1) spv_frag_outputs(code, ns / 4, s->out_cls);
    spv_io_masks(code, ns / 4, &s->in_mask, &s->out_mask);
    if (st == 0 && (s->tf_stride[0] | s->tf_stride[1] | s->tf_stride[2] | s->tf_stride[3])) {   // transform feedback: the capture variant (tf_draw)
        size_t tw; uint32_t *tc = spv_tf_variant(code, ns / 4, (const int (*)[2])s->tf, TF_BIND, &tw);
        if (tc) { VkShaderModuleCreateInfo tci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = tw * 4, .pCode = tc }; if (vkCreateShaderModule(g_dev, &tci, NULL, &s->tfmod)) s->tfmod = VK_NULL_HANDLE; free(tc); }
        if (!s->tfmod) VLOG("[RED] shader %s: transform feedback capture variant failed", hex);
    }
    VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = ns, .pCode = code };
    VkResult r = vkCreateShaderModule(g_dev, &mci, NULL, &s->mod);
    if (st == 1) { s->code = code; s->nw = ns / 4; } else free(code);
    if (r) { VLOG("[RED] shader %s: vkCreateShaderModule failed %d", hex, r); free(s); return NULL; }
    return s;
}
static pthread_mutex_t g_shmu = PTHREAD_MUTEX_INITIALIZER;   // the shader cache: the game's thread and the prewarm thread
static vshader *shader_get(const uint8_t key[8], int st) {
    pthread_mutex_lock(&g_shmu);
    vshader *s = nvn_hm_get(&g_shcache, key);
    if (!s) { s = shader_load(key, st); if (s) nvn_hm_put(&g_shcache, key, s); }
    pthread_mutex_unlock(&g_shmu);
    return s;
}
typedef struct vprog { vshader *sh[6]; VkDescriptorSetLayout dsl[2]; VkPipelineLayout pl[2]; uint8_t np[2]; } vprog;   // [0] graphics stages, [1] compute; np: descriptor sets instead of pushes
static nvn_hmap g_progs = {.klen = 8};   // program serial → vprog
static vprog *prog_get(const prog_t *p) {
    vprog *o = p->serial ? nvn_hm_get(&g_progs, &p->serial) : NULL; if (o) return o;
    shmap_load();
    o = calloc(1, sizeof *o);
    for (unsigned i = 0; i < p->nstage; i++) {
        uint8_t k[8]; if (!prog_key(p, i, k)) continue;
        intptr_t st1 = (intptr_t)nvn_hm_get(&g_shmap, k);
        if (!st1) { char hex[17]; for (int q = 0; q < 8; q++) sprintf(hex + 2 * q, "%02x", k[q]); VLOG("[RED] shader %s is not in the shader map", hex); continue; }
        vshader *s = shader_get(k, (int)st1 - 1); if (s) o->sh[st1 - 1] = s;
    }
    if (p->serial) nvn_hm_put(&g_progs, &p->serial, o);
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
    // more descriptors than one push takes (Pokémon Legends: Arceus' terrain: 29, past the 32 of some Adreno drivers with other programs): this program uses descriptor sets
    uint32_t nd = 0; for (uint32_t i = 0; i < n; i++) nd += b[i].descriptorCount;
    o->np[compute] = (uint8_t)(g_nopush || nd > g_maxpush);
    if (o->np[compute] && !g_nopush) { static int said; if (!said++) VLOG("a program with %u descriptors (push limit %u): descriptor sets for such programs", nd, g_maxpush); }
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = o->np[compute] ? 0 : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = n, .pBindings = b };
    VKCHECK(vkCreateDescriptorSetLayout(g_dev, &dci, NULL, &o->dsl[compute]));
    VkPipelineLayoutCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &o->dsl[compute], .pushConstantRangeCount = tf ? 1u : 0u, .pPushConstantRanges = &pcr };
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

// ================= Inline uniform updates deferred inside a pass (nvn_mtl.m mtl_sub_update) =================
// Vulkan cannot copy into a buffer inside a render pass, so each nvnCommandBufferUpdateUniformBuffer used to end the pass: on a tiler (Adreno) every end stores all attachments
// to memory and the next begin loads them back (Legends: Arceus: about 59 passes a frame, none kept open across a render target set). An update inside an open pass is now kept
// in the arena instead; a draw after it whose uniform binding overlaps it reads a shadow made of the deferred updates only (latest first per byte), and only when they cover the
// whole bound range, so the shadow is exactly what the GPU would read. The real copies are recorded in order when the pass ends, so later passes and the CPU see them as before.
// The pass still ends before a draw that would need more (ubd_conflict): a binding covered only in part, or the deferred data read as vertices, indices, indirect arguments or
// storage. DM_VK_UB_DEFER=0 turns it off
typedef struct { uint64_t lo, hi; VkBuffer buf; VkDeviceSize off; const uint8_t *map; } ubd_ent;
enum { UBD_MAX = 512 };
static ubd_ent g_ubd[UBD_MAX]; static int g_nubd;
static uint64_t g_ubd_deferred, g_ubd_conflicts;   // updates deferred, passes ended before a draw that needed more (the present line's statistics)
static int ubd_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_UB_DEFER"); v = !(e && !strcmp(e, "0")); if (!v) VLOG("DM_VK_UB_DEFER=0: uniform updates inside a pass end the pass as before"); } return v; }
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
// the arena in order. Bound at the shadow's own length (as nvn_mtl.m, which binds the shadow's address with nothing of the pool after it)
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

// ================= Render passes =================
static struct { int open; uint32_t cfmt[8], dfmt, sfmt, samples, tw, th; int hd, hs, ncolor; uint64_t dkey, skey; int dnone, snone, dwrote, swrote; } P;
static VkCommandBuffer CB(void) { return g_sub->cb; }
// What ended each pass (the present line): 0 other, 1 dispatch, 2 buffer copy, 3 uniform update, 4 fill, 5 buffer/texture copy, 6 texture copy, 7 texture clear, 8 new render targets, 9 submit
static uint64_t g_pe_n[16];
static void pass_end(void) {
    if (P.open) { vkCmdEndRendering(CB()); P.open = 0; g_pe_n[g_pe_why < 16 ? g_pe_why : 0]++;
        if (P_tf) {   // the captures' writes made visible to what reads them (vertex streams, shaders, copies)
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
            vkCmdPipelineBarrier(CB(), VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL); }
        if (!P_tf && P.hd) sk_ro_note(P.dkey, P.dwrote, 0);
        if (!P_tf && P.hs) sk_ro_note(P.skey, P.swrote, 0);
        g_after_pass = !P_tf; g_bar_clean = 0; P_tf = 0; g_ntfw = 0; }
    g_pe_why = 0; if (g_nubd) ubd_flush();
}
static int rt_dup_slot(const p_set_rt *r, int i) {
    if (!r->color[i]) return 0;
    uint8_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0; uint16_t ly = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
    for (int j = 0; j < i; j++) { if (!r->color[j] || r->color[j]->serial != r->color[i]->serial) continue;
        uint8_t lj = r->has_cview[j] && r->cview[j].set_levels ? r->cview[j].base_level : 0; uint16_t yj = r->has_cview[j] && r->cview[j].set_layers ? r->cview[j].min_layer : 0;
        if (lj == lv && yj == ly) return 1; }
    return 0;
}
// the attachment view of a render target: one level, one layer (3D: one depth slice through a 2D-array view is not available, the slice is taken as layer 0 — first case reported)
static uint32_t g_att_layers = 1;   // att_view: how many layers the view it returned has (a 3D target's slices)
static VkImageView att_view(const tex_t *t, const view_t *v, int has_v, vtex **out, uint32_t *w, uint32_t *h) {
    vtex *x = tex_res(t); *out = x; if (!x || !x->img) return VK_NULL_HANDLE;
    if (!(x->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) { RED_ONCE(19, "render target ser=%llu format %#x cannot be attached (Vulkan format %d)", (unsigned long long)t->serial, t->d.format, x->fmt); return VK_NULL_HANDLE; }
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0, ly = has_v && v->set_layers ? v->min_layer : 0;
    g_att_layers = 1;
    if (lv >= x->levels || (!x->is3d && ly >= x->layers)) return VK_NULL_HANDLE;
    if (x->cube_ok) { static uint64_t said[32]; static int nsaid; int seen = 0; uint64_t key = t->serial << 8 | ly << 4 | lv;
        for (int q = 0; q < nsaid; q++) if (said[q] == key) seen = 1;
        if (!seen && nsaid < 32) { said[nsaid++] = key; VLOG("render target: cube texture ser %llu level %u layer %u, view layers %u (set %d)", (unsigned long long)t->serial, lv, ly,
            has_v && v->set_layers ? v->nlayers : 0, has_v && v->set_layers); } }
    VkFormat f = x->fmt;
    if (has_v && v->set_format) { uint32_t vf = 0; uint8_t s[4]; if (nvn_fmt_to_vk(v->format, &vf, s) && vf) f = (VkFormat)vf; }
    *w = x->w >> lv ? x->w >> lv : 1; *h = x->h >> lv ? x->h >> lv : 1;
    // A 3D target: all its slices at that level as a 2D array, the pass layered over them (the shader picks the slice with gl_Layer). The colour-grading table is filled so,
    //   in one instanced draw; a 3D view is no attachment, and with slice 0 alone the other slices stay zero (Violet: every 3D scene black; the Galar / Paldea engine)
    static int flat = -1; if (flat < 0) flat = getenv("DM_VK_NO_LAYERED_3D") != NULL;   // DM_VK_NO_LAYERED_3D: the old view (to compare)
    if (x->is3d && flat) return view_get(x, VK_IMAGE_VIEW_TYPE_3D, f, lv, 1, ly, 1, SW_ID, x->aspect);
    if (x->is3d) { uint32_t dz = x->d >> lv ? x->d >> lv : 1; if (ly >= dz) return VK_NULL_HANDLE; g_att_layers = dz - ly;
        return view_get(x, VK_IMAGE_VIEW_TYPE_2D_ARRAY, f, lv, 1, ly, dz - ly, SW_ID, x->aspect); }
    return view_get(x, VK_IMAGE_VIEW_TYPE_2D, f, lv, 1, ly, 1, SW_ID, x->aspect);
}
// DM_NVN_TEXDUMP=frame:path (as nvn_mtl.m): every colour render target that frame uses, saved once the frame is over as <path with %d = index * 100 + level> (PNG, clamped to [0,1])
static long g_td_frame = -2; static const char *g_td_path; static int g_td_done;
static struct { vtex *x; uint64_t ser; uint32_t lv, sl; } g_td[256]; static int g_ntd;
static void shot_level(vtex *x, uint32_t lv, uint32_t sl, const char *path);
static void td_note(vtex *x, uint64_t ser, uint32_t lv, uint32_t sl) {
    if (g_td_frame == -2) { const char *e = getenv("DM_NVN_TEXDUMP"), *c = e ? strchr(e, ':') : NULL; g_td_frame = c ? atol(e) : -1; g_td_path = c ? c + 1 : NULL; }
    if (g_td_frame < 0 || g_td_done) return;
    if ((long)g_frame_no > g_td_frame) {
        g_td_done = 1; pass_end(); uint64_t f = mtl_sub_commit(); vk_wait(f, UINT64_MAX); mtl_sub_begin();
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
// for the GPU to go idle) unless one reads or rewrites what an earlier one in the pass wrote (keep = 0); anything else between them ends the pass. DM_VK_TF_BATCH=0: one pass each
static int tf_batch_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_TF_BATCH"); v = !(e && !strcmp(e, "0")); } return v; }
static int tf_hits(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_ntfw; i++) if (lo < g_tfw[i].hi && g_tfw[i].lo < hi) return 1; return 0; }
static void tf_pass_begin(int keep) {
    if (P.open && P_tf && keep && tf_batch_on()) return;
    g_pe_why = P_tf ? 11 : 10; pass_end(); memset(&P, 0, sizeof P); P_tf = 1; g_bar_clean = g_after_pass = 0;
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {1, 1}}, .layerCount = 1 };
    vkCmdBeginRendering(CB(), &ri); P.open = 1; P.tw = P.th = 1; P.samples = 1;
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
static int pass_begin_impl(void);
static int pass_begin(void) { uint64_t pf_t0 = prof_t(); int r = pass_begin_impl(); prof_add(PF_BEGIN, pf_t0); return r; }
static int pass_begin_impl(void) {
    if (P.open && P_tf) { g_pe_why = 11; pass_end(); }   // a capture pass is not the game's
    if (P.open) return 1;
    if (!R.has_rt) { RED_ONCE(21, "draw with no render target set"); return 0; }
    VkRenderingAttachmentInfo ca[8]; VkRenderingAttachmentInfo da = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO }, sa = da;
    memset(&P, 0, sizeof P); uint32_t tw = 16384, th = 16384, nlay = 0; int any = 0, ncol = 0; uint64_t dkey = 0, skey = 0; int dnone = 0, snone = 0;   // nlay: the fewest layers of the attachments (a layered pass over a 3D target)
    for (int i = 0; i < 8; i++) { ca[i] = (VkRenderingAttachmentInfo){ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO }; }
    {   // this pass's key for the store learning (sk_attach): its render targets and how many times they were set before in this frame
        static struct { uint64_t sig; uint32_t n; } occ[256]; static uint64_t occ_frame = ~0ull;
        if (occ_frame != g_frame_no) { memset(occ, 0, sizeof occ); occ_frame = g_frame_no; }
        uint64_t h = 1469598103934665603ull; const p_set_rt *r = &R.rt;
        #define SKH(v) (h = (h ^ (uint64_t)(v)) * 1099511628211ull)
        SKH(r->n); for (int i = 0; i < r->n && i < 8; i++) { SKH(r->color[i] ? r->color[i]->serial : 0); SKH(r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0); SKH(r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0); }
        SKH(r->depth ? r->depth->serial : 0); SKH(r->has_dview && r->dview.set_levels ? r->dview.base_level : 0); SKH(r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0);
        #undef SKH
        uint32_t o = 0; for (uint32_t q = (uint32_t)h & 255, k = 0; k < 256; q = (q + 1) & 255, k++) { if (!occ[q].sig) { occ[q].sig = h; o = occ[q].n++; break; } if (occ[q].sig == h) { o = occ[q].n++; break; } }
        g_sk_sig = h ^ (uint64_t)o * 0xC2B2AE3D27D4EB4Full;
    }
    for (int i = 0; i < R.rt.n && i < 8; i++) {
        if (!R.rt.color[i] || rt_dup_slot(&R.rt, i)) continue;
        vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i], &x, &w, &h);
        if (!v || !(x->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) continue;
        if (!nlay || g_att_layers < nlay) nlay = g_att_layers;
        ca[i].imageView = v; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].storeOp = sk_attach(x, R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0, R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0, 1, i, R.clear_c[i])
                        ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        ca[i].loadOp = R.clear_c[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].clearValue.color = R.cc[i];
        VkFormat f = x->fmt; if (R.rt.has_cview[i] && R.rt.cview[i].set_format) { uint32_t vf = 0; uint8_t s[4]; if (nvn_fmt_to_vk(R.rt.cview[i].format, &vf, s) && vf) f = (VkFormat)vf; }
        P.cfmt[i] = (uint32_t)f; P.samples = x->samples; if (w < tw) tw = w; if (h < th) th = h; any = 1; ncol = i + 1;
        td_note(x, R.rt.color[i]->serial, R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0, R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0);
    }
    if (R.rt.depth) {
        vtex *x; uint32_t w, h; VkImageView v = att_view(R.rt.depth, &R.rt.dview, R.rt.has_dview, &x, &w, &h);
        if (v && (x->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            if (!nlay || g_att_layers < nlay) nlay = g_att_layers;
            uint32_t dl = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0, dy = R.rt.has_dview && R.rt.dview.set_layers ? R.rt.dview.min_layer : 0;
            if (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da.imageView = v; da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                da.storeOp = sk_attach(x, dl, dy, 2, 8, R.clear_d) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE; dkey = g_sk_last;
                if (da.storeOp == VK_ATTACHMENT_STORE_OP_STORE && g_store_none && sk_ro(dkey, R.clear_d)) { da.storeOp = VK_ATTACHMENT_STORE_OP_NONE; dnone = 1; }
                da.loadOp = R.clear_d ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; da.clearValue.depthStencil.depth = R.cd; P.hd = 1; P.dfmt = (uint32_t)x->fmt; }
            if (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.imageView = v; sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                sa.storeOp = sk_attach(x, dl, dy, 3, 9, R.clear_s) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE; skey = g_sk_last;
                if (sa.storeOp == VK_ATTACHMENT_STORE_OP_STORE && g_store_none && sk_ro(skey, R.clear_s)) { sa.storeOp = VK_ATTACHMENT_STORE_OP_NONE; snone = 1; }
                sa.loadOp = R.clear_s ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; sa.clearValue.depthStencil.stencil = R.cs; P.hs = 1; P.sfmt = (uint32_t)x->fmt; }
            P.samples = x->samples; if (w < tw) tw = w; if (h < th) th = h; any = 1;
        }
    }
    if (!any) {   // no target attached: an attachment-less pass at the viewport's size (nvn_mtl.m bare_dims), or nothing when targets were set but none could be attached
        int bare = !R.rt.depth; for (int i = 0; i < R.rt.n; i++) if (R.rt.color[i]) bare = 0;
        if (!bare) { RED_ONCE(22, "no render target could be attached"); return 0; }
        double x1 = S.has_vp ? fmax(S.vp[0][0], S.vp[0][0] + S.vp[0][2]) : R.has_sc ? (double)R.sc.x + R.sc.w : 1, y1 = S.has_vp ? fmax(S.vp[0][1], S.vp[0][1] + S.vp[0][3]) : R.has_sc ? (double)R.sc.y + R.sc.h : 1;
        tw = (uint32_t)fmin(fmax(ceil(x1), 1), 16384); th = (uint32_t)fmin(fmax(ceil(y1), 1), 16384); P.samples = 1;
    }
    if (!P.samples) P.samples = 1;
    P.tw = tw; P.th = th; P.ncolor = ncol; P.dkey = dkey; P.skey = skey; P.dnone = dnone; P.snone = snone; P.dwrote = R.clear_d; P.swrote = R.clear_s;
    g_ro_passes += dnone + snone;
    if (g_bar_clean && barrier_relaxed()) g_bar_none++;
    else if (g_after_pass && !g_game_bar && barrier_relaxed()) { barrier_pass(CB()); g_bar_narrow++; }
    else { barrier_all(CB()); g_bar_full++; }
    g_game_bar = 0; g_bar_clean = g_after_pass = 0;
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {tw, th}}, .layerCount = nlay ? nlay : 1, .colorAttachmentCount = (uint32_t)ncol, .pColorAttachments = ca,
                           .pDepthAttachment = P.hd ? &da : NULL, .pStencilAttachment = P.hs ? &sa : NULL };
    vkCmdBeginRendering(CB(), &ri); g_stat_begins++;
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    P.open = 1;
    return 1;
}
static int getenv_nomerge(void) { static int v = -1; if (v < 0) v = getenv("DM_VK_NO_PASS_MERGE") != NULL; return v; }   // A/B switch
static void pass_flush(void) {   // pending clears with no draw after them: a pass just for them
    if (P_tf) { g_pe_why = 11; pass_end(); }
    int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    if (pend && !P.open && g_sub) pass_begin();
    pass_end();
}
static void outside_pass(void) { pass_end(); barrier_before(CB()); }   // transfer work: no pass open, ordered after everything before

// ================= Pipelines =================
typedef struct {
    void *sh[5]; uint32_t cfmt[8], dfmt, sfmt; uint32_t chmask; blend_st blend[8];
    struct { uint32_t fmt, off; uint8_t bind, used, inst, pad; uint32_t div; } attr[32];
    uint8_t samples, a2c, poly, clamp, topo, logic_en, logic_op, blend_en, depth_zero, discard, pad[6];
} pkey;
static nvn_hmap g_pso = {.klen = sizeof(pkey)}, g_pso_fail = {.klen = sizeof(pkey)};
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
static VkBlendOp bop(int e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
// vertex format for a shader input of class ct (0 float, 1 uint, 2 sint): an integer format of the other signedness is read with the same width and the input's sign
static VkFormat vfmt_for(int nvn_fmt, int ct) {
    int fc; VkFormat f = (VkFormat)nvn_vfmt_to_vk(nvn_fmt, &fc);
    if (!f) return VK_FORMAT_UNDEFINED;
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
static VkShaderModule frag_module(vprog *pr, const uint32_t *cfmt) {
    vshader *fs = pr->sh[1], *pv = pr->sh[2] ? pr->sh[2] : pr->sh[4] ? pr->sh[4] : pr->sh[0];
    uint64_t drop = fs->in_mask & ~(pv ? pv->out_mask : 0); uint32_t um = 0, sm = 0;
    for (int i = 0; i < 8; i++) if (cfmt[i] && fs->out_cls[i] == 0) { int c = fmt_int_class((VkFormat)cfmt[i]); if (c == 1) um |= 1u << i; else if (c == 2) sm |= 1u << i; }
    if ((!drop && !um && !sm) || !fs->code) return fs->mod;
    pthread_mutex_lock(&g_psomu);
    for (int i = 0; i < fs->nvar; i++) if (fs->var[i].drop == drop && fs->var[i].um == um && fs->var[i].sm == sm) { VkShaderModule m = fs->var[i].mod; pthread_mutex_unlock(&g_psomu); return m; }
    pthread_mutex_unlock(&g_psomu);
    size_t nw = fs->nw; uint32_t *c = NULL, *c2;
    if (drop && (c2 = spv_drop_inputs(fs->code, fs->nw, drop, &nw))) c = c2;
    if ((um || sm) && (c2 = spv_int_outputs(c ? c : fs->code, c ? nw : fs->nw, um, sm, &nw))) { free(c); c = c2; }
    if (!c) return fs->mod;
    VkShaderModule m = VK_NULL_HANDLE; VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = c };
    VkResult r = vkCreateShaderModule(g_dev, &mci, NULL, &m); free(c);
    if (r) { VLOG("[RED] fragment shader %s: variant failed %d", fs->key, r); return fs->mod; }
    if (drop) VLOG("fragment shader %s reads varyings %#llx that %s never writes: those read (0, 0, 0, 1)", fs->key, (unsigned long long)drop, pv ? pv->key : "-");
    if (um | sm) VLOG("fragment shader %s: float outputs %#x / %#x written as uint / int bits (integer colour targets)", fs->key, um, sm);
    pthread_mutex_lock(&g_psomu);
    if (fs->nvar < 8) { fs->var[fs->nvar].drop = drop; fs->var[fs->nvar].um = um; fs->var[fs->nvar].sm = sm; fs->var[fs->nvar].mod = m; fs->nvar++; }
    pthread_mutex_unlock(&g_psomu);
    return m;
}
// ---------- Pipeline list and cache (Android): every pipeline the game makes is recorded (DM_VK_PSO_RECORD) as its shaders' keys plus the rest of its key; at start the
// recorded list and the list shipped with the game (DM_VK_PSO_SEED) are built on a background thread before the game asks for them, and the driver's compiled pipelines are
// kept in a pipeline cache file (DM_VK_PCACHE), so a later start compiles nothing it already compiled
typedef struct { uint8_t key[5][8]; pkey k; } pso_rec;   // k.sh zeroed
#define PSO_MAGIC "NXPL"
static FILE *g_psorec;
static void pso_record(const vprog *pr, const pkey *k) {   // caller holds g_psomu
    static int tried; if (!tried++) { const char *e = getenv("DM_VK_PSO_RECORD"); if (e) { g_psorec = fopen(e, "ab"); if (g_psorec && ftell(g_psorec) == 0) { uint32_t v[2] = { 1, (uint32_t)sizeof(pso_rec) }; fwrite(PSO_MAGIC, 1, 4, g_psorec); fwrite(v, 4, 2, g_psorec); } } }
    if (!g_psorec) return;
    pso_rec r; memset(&r, 0, sizeof r); r.k = *k; memset(r.k.sh, 0, sizeof r.k.sh);
    for (int i = 0; i < 5; i++) if (pr->sh[i]) hexkey(pr->sh[i]->key, r.key[i]);
    fwrite(&r, sizeof r, 1, g_psorec); fflush(g_psorec);
}
static VkPipeline pso_make(vprog *pr, pkey k);
static VkPipeline pso_lookup(vprog *pr, const pkey *k, int record) {
    pthread_mutex_lock(&g_psomu);
    VkPipeline p = nvn_hm_get(&g_pso, k); int failed = p ? 0 : nvn_hm_get(&g_pso_fail, k) != NULL;
    pthread_mutex_unlock(&g_psomu);
    if (p || failed) return p;
    p = pso_make(pr, *k);   // compiled outside the lock: the prewarm thread and the game's thread compile in parallel
    pthread_mutex_lock(&g_psomu);
    VkPipeline q = nvn_hm_get(&g_pso, k);
    if (q) { pthread_mutex_unlock(&g_psomu); if (p) vkDestroyPipeline(g_dev, p, NULL); return q; }   // the other thread finished the same one first
    if (p) { nvn_hm_put(&g_pso, k, p); if (record) { pso_record(pr, k); atomic_fetch_add(&g_pso_ingame, 1); } atomic_fetch_add(&g_pso_new, 1); }
    else nvn_hm_put(&g_pso_fail, k, (void *)1);
    pthread_mutex_unlock(&g_psomu);
    return p;
}
static void pcache_save(void) {
    const char *path = getenv("DM_VK_PCACHE"); if (!path || !g_pcache) return;
    size_t n = 0; if (vkGetPipelineCacheData(g_dev, g_pcache, &n, NULL) || !n) return;
    void *d = malloc(n); if (vkGetPipelineCacheData(g_dev, g_pcache, &n, d)) { free(d); return; }
    char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb");
    if (f) { int ok = fwrite(d, 1, n, f) == n; ok &= !fclose(f); if (ok) rename(tmp, path); }
    free(d); atomic_store(&g_pso_new, 0);
    VLOG("pipeline cache saved: %.1f MB", n / 1048576.0);
}
void nvn_vk_cache_save(void) { if (g_dev) pcache_save(); }
void *saver(void *a) { pcache_save(); atomic_store((atomic_int *)a, 0); return NULL; }   // the app calls it when going to the background
static void pcache_open(void) {
    const char *path = getenv("DM_VK_PCACHE"); size_t n = 0; void *d = NULL;
    if (path) { FILE *f = fopen(path, "rb"); if (f) { fseek(f, 0, SEEK_END); long l = ftell(f); rewind(f); if (l > 0) { d = malloc((size_t)l); n = fread(d, 1, (size_t)l, f); } fclose(f); } }
    VkPipelineCacheCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = n, .pInitialData = d };
    if (vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache)) { ci.initialDataSize = 0; ci.pInitialData = NULL; vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache); n = 0; }   // a cache of another driver version: start empty
    free(d);
    if (path) VLOG("pipeline cache: %.1f MB from %s", n / 1048576.0, path);
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
static void prewarm_pace(int i, int *waits) {
    usleep(3000);
    if (i % 8) return;
    static long floor_mb = -1; if (floor_mb < 0) { const char *e = getenv("DM_VK_PREWARM_MIN_MB"); floor_mb = e ? atol(e) : 1536; }
    for (int k = 0; k < 600; k++) { long m = mem_avail_mb(); if (m < 0 || m >= floor_mb) break; if (!k) (*waits)++; usleep(100000); }   // at most a minute per wait
}
static void *prewarm_main(void *arg) {
    (void)arg; uint64_t t0 = mono_ns(); int waits = 0;
    pso_rec *rs = NULL; int n = 0;
    int ns = pso_rec_load(getenv("DM_VK_PSO_SEED"), &rs, &n), nr = pso_rec_load(getenv("DM_VK_PSO_RECORD"), &rs, &n);
    if (!n) return NULL;
    static nvn_hmap progs = {.klen = 40}; static nvn_hmap seen = {.klen = sizeof(pso_rec)};
    int built = 0, had = 0, failed = 0, noshader = 0;
    for (int i = 0; i < n; i++) {
        if (nvn_hm_get(&seen, &rs[i])) continue; nvn_hm_put(&seen, &rs[i], (void *)1);
        vprog *pr = nvn_hm_get(&progs, rs[i].key);
        if (!pr) {
            pr = calloc(1, sizeof *pr); int ok = 1;
            for (int st = 0; st < 5; st++) { static const uint8_t z[8]; if (!memcmp(rs[i].key[st], z, 8)) continue; pr->sh[st] = shader_get(rs[i].key[st], st); if (!pr->sh[st]) ok = 0; }
            if (!ok || !pr->sh[0]) { free(pr); noshader++; continue; }
            nvn_hm_put(&progs, rs[i].key, pr);
        }
        pkey k = rs[i].k; for (int st = 0; st < 5; st++) k.sh[st] = pr->sh[st];
        prog_layout(pr, 0);
        pthread_mutex_lock(&g_psomu); int have = nvn_hm_get(&g_pso, &k) != NULL; pthread_mutex_unlock(&g_psomu);
        if (have) { had++; continue; }
        if (pso_lookup(pr, &k, 0)) built++; else failed++;
        prewarm_pace(built + failed, &waits);
    }
    VLOG("prewarm: %d pipelines listed (%d shipped, %d recorded here): %d built, %d already there, %d failed, %d with a shader missing, %.1f s (paused %d times for low memory)", n, ns, nr, built, had, failed, noshader, (mono_ns() - t0) / 1e9, waits);
    free(rs);
    pcache_save();
    return NULL;
}
// Builds the pipeline for key k (no lookup: pso_get and the prewarm do that)
static VkPipeline pso_make(vprog *pr, pkey k) {
    vshader *vs = pr->sh[0], *fs = pr->sh[1]; VkPipeline p = VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo st[5]; uint32_t nst = 0;
    for (int i = 0; i < 5; i++) if (pr->sh[i] && !(i == 1 && k.discard)) st[nst++] = (VkPipelineShaderStageCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = pr->sh[i]->vkstage, .module = i == 1 ? frag_module(pr, k.cfmt) : i == 0 && k.discard && pr->sh[0]->tfmod ? pr->sh[0]->tfmod : pr->sh[i]->mod, .pName = "main" };
    VkVertexInputBindingDescription vb[17]; VkVertexInputAttributeDescription va[32]; VkVertexInputBindingDivisorDescriptionEXT vd[16]; uint32_t nvb = 0, nva = 0, nvd = 0; uint32_t have = 0;
    for (uint32_t loc = 0; loc < 32; loc++) {
        if (!k.attr[loc].used) continue;
        va[nva++] = (VkVertexInputAttributeDescription){ loc, k.attr[loc].bind, (VkFormat)k.attr[loc].fmt, k.attr[loc].bind == 16 ? 0 : k.attr[loc].off };
        uint32_t b = k.attr[loc].bind; if (have >> b & 1) continue; have |= 1u << b;
        vb[nvb++] = (VkVertexInputBindingDescription){ b, 0, b != 16 && k.attr[loc].inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX };
        if (b != 16 && k.attr[loc].inst && k.attr[loc].div > 1) { if (g_divisor) vd[nvd++] = (VkVertexInputBindingDivisorDescriptionEXT){ b, k.attr[loc].div }; else RED_ONCE(26, "instance divisor %u needs VK_EXT_vertex_attribute_divisor", k.attr[loc].div); }
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT vds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, .vertexBindingDivisorCount = nvd, .pVertexBindingDivisors = vd };
    VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nvd ? &vds : NULL, .vertexBindingDescriptionCount = nvb, .pVertexBindingDescriptions = vb, .vertexAttributeDescriptionCount = nva, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = k.topo == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : k.topo == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, .negativeOneToOne = !k.depth_zero };   // NVN's GL depth mode: z in [-w, w]
    VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = g_clipctl ? &dcc : NULL, .viewportCount = 1, .scissorCount = 1 };
    if (!g_clipctl && !k.depth_zero) RED_ONCE(27, "GL depth mode without VK_EXT_depth_clip_control: depth is off by the [-w, w] → [0, w] mapping");
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
        cb[i].colorWriteMask = m;
        if (k.blend_en >> i & 1 && m && !fmt_int_class(cf[i])) { cb[i].blendEnable = VK_TRUE;   // integer targets never blend (Vulkan: undefined)
            cb[i].srcColorBlendFactor = bfac(k.blend[i].src_rgb); cb[i].dstColorBlendFactor = bfac(k.blend[i].dst_rgb); cb[i].srcAlphaBlendFactor = bfac(k.blend[i].src_a); cb[i].dstAlphaBlendFactor = bfac(k.blend[i].dst_a);
            cb[i].colorBlendOp = bop(k.blend[i].eq_rgb); cb[i].alphaBlendOp = bop(k.blend[i].eq_a); }
    }
    VkPipelineColorBlendStateCreateInfo bs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .logicOpEnable = k.logic_en, .logicOp = (VkLogicOp)k.logic_op,
                                           .attachmentCount = ncb, .pAttachments = cb };
    VkDynamicState dyn[24]; uint32_t nd = 0;
    VkDynamicState base[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP };
    for (unsigned i = 0; i < sizeof base / sizeof *base; i++) dyn[nd++] = base[i];
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = nd, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = ncb, .pColorAttachmentFormats = cf,
        .depthAttachmentFormat = (VkFormat)k.dfmt, .stencilAttachmentFormat = (VkFormat)k.sfmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = nst, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &bs, .pDynamicState = &dy, .layout = prog_layout(pr, 0) };
#ifdef __ANDROID__
    { static int npipe; VLOG("pipeline %d: vertex %s fragment %s", ++npipe, vs->key, fs ? fs->key : "-"); }   // the last line before a driver crash names the shaders it was compiling
#endif
    VkResult r = vkCreateGraphicsPipelines(g_dev, g_pcache, 1, &gi, NULL, &p);
    if (r) { VLOG("[RED] graphics pipeline (vertex %s fragment %s) failed: %d", vs->key, fs ? fs->key : "-", r); return VK_NULL_HANDLE; }
    return p;
}
static VkPipeline pso_get_impl(vprog *pr, VkPrimitiveTopology topo);
static VkPipeline pso_get(vprog *pr, VkPrimitiveTopology topo) { uint64_t pf_t0 = prof_t(); VkPipeline r = pso_get_impl(pr, topo); prof_add(PF_PSO, pf_t0); return r; }
static VkPipeline pso_get_impl(vprog *pr, VkPrimitiveTopology topo) {
    vshader *vs = pr->sh[0], *fs = pr->sh[1];
    pkey k; memset(&k, 0, sizeof k);
    for (int i = 0; i < 5; i++) k.sh[i] = pr->sh[i];
    memcpy(k.cfmt, P.cfmt, sizeof k.cfmt); k.dfmt = P.hd ? P.dfmt : 0; k.sfmt = P.hs ? P.sfmt : 0; k.samples = (uint8_t)P.samples;
    k.chmask = S.chmask.mask; k.blend_en = (uint8_t)S.color.blend_en; for (int i = 0; i < 8; i++) if (k.cfmt[i]) k.blend[i] = S.blend[i];
    k.a2c = S.ms.a2c; k.poly = S.poly.mode == NVN_POLYGON_MODE_LINE ? 1 : S.poly.mode == NVN_POLYGON_MODE_POINT ? 2 : 0; k.clamp = (uint8_t)!!S.depth_clamp; k.topo = (uint8_t)topo_class(topo);
    k.logic_en = S.color.logic_op != NVN_LOGIC_OP_COPY; k.logic_op = (uint8_t)S.color.logic_op; k.depth_zero = (uint8_t)atomic_load(&g_depth_zero); k.discard = (uint8_t)!!S.rast_discard;
    for (int i = 0; i < vs->nin; i++) {
        uint32_t loc = vs->in[i].location; if (loc >= 32) continue;
        k.attr[loc].used = 1;
        if ((int)loc < S.nattr && S.attr[loc].fmt) { int st = S.attr[loc].stream; VkFormat f = vfmt_for(S.attr[loc].fmt, vs->in[i].sclass);
            if (f && st < 16) { k.attr[loc].fmt = f; k.attr[loc].off = S.attr[loc].offset; k.attr[loc].bind = (uint8_t)st; k.attr[loc].inst = S.stream[st].divisor != 0; k.attr[loc].div = S.stream[st].divisor; continue; }
            RED_ONCE(25, "vertex attribute format %#x not mapped", S.attr[loc].fmt); }
        k.attr[loc].fmt = vs->in[i].sclass == 1 ? VK_FORMAT_R32G32B32A32_UINT : vs->in[i].sclass == 2 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT; k.attr[loc].bind = 16;   // unbound: reads 0
    }
    return pso_lookup(pr, &k, 1);
}

// ================= Descriptors =================
static uint32_t g_draw_binst, g_dispatch_dim[3];
static VkBuffer g_c0_buf; static VkDeviceSize g_c0_off;
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
static void desc_flush_impl(VkPipelineBindPoint bp, VkPipelineLayout pl, VkDescriptorSetLayout dsl);
static void desc_flush(VkPipelineBindPoint bp, VkPipelineLayout pl, VkDescriptorSetLayout dsl) { uint64_t pf_t0 = prof_t(); desc_flush_impl(bp, pl, dsl); prof_add(PF_FLUSH, pf_t0); }
static void desc_flush_impl(VkPipelineBindPoint bp, VkPipelineLayout pl, VkDescriptorSetLayout dsl) {
    if (!g_np || !D.n) { D.n = 0; return; }
    VkDescriptorSet ds = desc_set_alloc(dsl);
    for (uint32_t i = 0; i < D.n; i++) D.w[i].dstSet = ds;
    vkUpdateDescriptorSets(g_dev, D.n, D.w, 0, NULL);
    vkCmdBindDescriptorSets(CB(), bp, pl, 0, 1, &ds, 0, NULL);
    D.n = 0;
}
static void push_stage_impl(vshader *s, int st, VkPipelineBindPoint bp, VkPipelineLayout pl);
static uint64_t g_stat_draws;
static _Atomic uint64_t g_fx_shown[4], g_fx_frames;
static int dl_on(void) { static long at = -2; if (at == -2) { const char *e = getenv("DM_VK_DRAWLOG"); at = e ? atol(e) : -1; } return at >= 0 && (long)atomic_load(&g_fx_frames) == at; }   // DM_VK_DRAWLOG=<present>: each draw of that frame and what it samples to the log
static void push_stage(vshader *s, int st, VkPipelineBindPoint bp, VkPipelineLayout pl) { uint64_t pf_t0 = prof_t(); push_stage_impl(s, st, bp, pl); prof_add(PF_PUSH, pf_t0); }
static void push_stage_impl(vshader *s, int st, VkPipelineBindPoint bp, VkPipelineLayout pl) {
    if (!s || !s->nres) return;
    VkWriteDescriptorSet w_[64]; VkDescriptorBufferInfo bi_[64]; VkDescriptorImageInfo ii_[64]; VkBufferView bv_[64]; uint32_t nw = 0;
    VkWriteDescriptorSet *w = w_; VkDescriptorBufferInfo *bi = bi_; VkDescriptorImageInfo *ii = ii_; VkBufferView *bv = bv_;
    if (g_np) { if (D.n + 64 > 320) D.n = 320 - 64; w = D.w + D.n; bi = D.bi + D.n; ii = D.ii + D.n; bv = D.bv + D.n; }
    pthread_mutex_lock(&g_rmu);
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
                        if (po % al == 0) { b = pb; o = po; n = want; }
                        else if (r->kind == SPV_RES_UBO) {   // misaligned uniform data: a snapshot in the arena (the GPU-side updates of this submission are in the pool already)
                            if (want > 65536) want = 65536;   // the translated shaders declare 64 KB uniform blocks: never more to copy (MoltenVK reports a range of gigabytes)
                            uint8_t *d = arena_alloc(g_sub, want, &b, &o); memcpy(d, (const void *)(uintptr_t)addr, want); n = want;
                            RED_ONCE(29, "uniform buffer at pool offset %#llx is not %llu-aligned: bound as a snapshot", (unsigned long long)po, (unsigned long long)al);
                        } else if (!h->rw) {   // misaligned read-only storage data (Adreno wants 64, the Switch less): a snapshot as for uniforms, not zeros
                            VkDeviceSize m = size && size < want ? size : want; if (m > (size ? 1u << 20 : 65536u)) m = size ? 1u << 20 : 65536u;
                            uint8_t *d = arena_alloc(g_sub, m, &b, &o); memcpy(d, (const void *)(uintptr_t)addr, m); n = m;
                            RED_ONCE(30, "storage buffer at pool offset %#llx is not %llu-aligned: bound as a snapshot", (unsigned long long)po, (unsigned long long)al);
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
                VkImageView v = tid ? tex_view_for(tid, r, storage) : VK_NULL_HANDLE;
                if (dl_on() && !(tid > 0 && tid < g_ntreg && g_treg[tid].ok))
                    VLOG("[texdump] draw %llu st %d %s %s: tid %d UNRESOLVED (how %d reg %d off %d) rt0 %llu", (unsigned long long)g_stat_draws, st, s->key, r->name, tid, (int)h->what, (int)h->reg, (int)h->off,
                         R.has_rt && R.rt.color[0] ? (unsigned long long)R.rt.color[0]->serial : 0ULL);
                if (dl_on() && tid > 0 && tid < g_ntreg && g_treg[tid].ok) {   // texdump frame: what each draw samples
                    vtex *x = tex_res(&g_treg[tid].t);
                    if (x) { VLOG("[texdump] draw %llu st %d %s %s: tid %d ser=%llu %ux%u fmt %d nvn %#x levels %u layers %u%s mem %p [%016llx %016llx] rt0 %llu", (unsigned long long)g_stat_draws, st, s->key, r->name, tid,
                                  (unsigned long long)g_treg[tid].t.serial, x->w, x->h, (int)x->fmt, x->nvn_fmt, x->levels, x->layers, x->pkg ? " pkg" : "", (void *)g_treg[tid].t.mem,
                                  g_treg[tid].t.mem ? (unsigned long long)((const uint64_t *)g_treg[tid].t.mem)[0] : 0ULL, g_treg[tid].t.mem && x->w * x->h >= 16384 ? (unsigned long long)((const uint64_t *)g_treg[tid].t.mem)[4096] : 0ULL, R.has_rt && R.rt.color[0] ? (unsigned long long)R.rt.color[0]->serial : 0ULL);
                             if (!x->pkg) for (uint32_t sl = 0; sl < (x->layers < 6 ? x->layers : 6); sl++) td_note(x, g_treg[tid].t.serial, 0, sl); }
                }
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
                if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) { vtex *x = tex_res(&g_treg[tid].t); if (x && x->bview) view = x->bview; }
                bv[nw] = view ? view : g_dummy_tbuf[r->sclass < 3 ? r->sclass : 0]; w[nw].pTexelBufferView = &bv[nw]; nw++; break;
            }
        }
    }
    pthread_mutex_unlock(&g_rmu);
    if (g_np) D.n += nw;
    else if (nw) p_push(CB(), bp, pl, 0, nw, w);
}

// ================= Draws =================
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
static uint64_t g_stat_draws, g_stat_passes, g_stat_rt_same, g_splits;
static int rt_same(const p_set_rt *a, const p_set_rt *b);
// ---------- FXAA inside the game's frame (DM_VK_FXAA=2) ----------
// DM_VK_FXAA=1 smooths the finished picture at present: the game's interface (text, menus, button prompts) is softened with it. Here the 3D scene alone is smoothed: the
// first draw of a frame into the texture the game shows that samples a screen-shaped render target of its own making (the scene being composited, Legends: Arceus upscales its
// scene so) runs FXAA on that target first, through a copy (a pass cannot read what it writes); the game's interface drawn after it stays sharp. Nothing else changes: the same
// texture, its handles and its descriptors
#include "overlay/scene_fxaa_spv.h"
static struct { VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds; VkSampler samp;
                struct { VkFormat f; VkPipeline p; } pipe[4]; int npipe;
                VkImage img; VkImageView view; vmalloc_t a; uint32_t w, h; VkFormat fmt; int failed; } FX;
static _Atomic uint64_t g_fx_shown[4], g_fx_frames;   // (declared above too) the textures shown lately (mtl_present), presents so far
static uint64_t g_fx_done, g_fx_tries;
static _Atomic long g_passlog;                         // the present whose passes go to the log (pass_log): DM_VK_PASSLOG=<present>, or a file <DM_SHOT_REQ>.passlog (the next one)                 // the frame the scene was smoothed in; draws into the shown texture looked at this frame
static uint64_t g_bind_gen;                            // bumped when something else binds a pipeline in the replay's command buffer (draw_impl's bind cache)
static int fx_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_FXAA"); on = e && atoi(e) == 2; if (on) VLOG("FXAA on the game's 3D scene, before its interface"); } return on; }
static _Atomic uint32_t g_fx_sw, g_fx_sh;   // the shown texture's size
static void fx_shown(uint64_t serial, uint32_t w, uint32_t h) {   // mtl_present
    atomic_fetch_add(&g_fx_frames, 1); atomic_store(&g_fx_sw, w); atomic_store(&g_fx_sh, h);
    for (int i = 0; i < 4; i++) if (atomic_load(&g_fx_shown[i]) == serial) return;
    for (int i = 3; i > 0; i--) atomic_store(&g_fx_shown[i], atomic_load(&g_fx_shown[i - 1]));
    atomic_store(&g_fx_shown[0], serial);
}
static int fx_is_shown(uint64_t serial) { for (int i = 0; i < 4; i++) if (serial && atomic_load(&g_fx_shown[i]) == serial) return 1; return 0; }
static VkPipeline fx_pipe(VkFormat fmt) {
    for (int i = 0; i < FX.npipe; i++) if (FX.pipe[i].f == fmt) return FX.pipe[i].p;
    if (FX.npipe == 4) return VK_NULL_HANDLE;
    if (!FX.pl) {
        VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f };
        VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
        VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b };
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
        VkPushConstantRange pr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };
        if (vkCreateSampler(g_dev, &sci, NULL, &FX.samp) || vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &FX.dsl) || vkCreateDescriptorPool(g_dev, &dpi, NULL, &FX.dp)) { FX.failed = 1; return VK_NULL_HANDLE; }
        VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = FX.dp, .descriptorSetCount = 1, .pSetLayouts = &FX.dsl };
        VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &FX.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
        if (vkAllocateDescriptorSets(g_dev, &dai, &FX.ds) || vkCreatePipelineLayout(g_dev, &pli, NULL, &FX.pl)) { FX.failed = 1; return VK_NULL_HANDLE; }
    }
    VkShaderModule vm, fm;
    VkShaderModuleCreateInfo mv = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof fx_vert_spv, .pCode = fx_vert_spv }, mf = mv; mf.codeSize = sizeof fx_frag_spv; mf.pCode = fx_frag_spv;
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
    VkPipelineDepthStencilStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt };
    VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = FX.pl };
    VkPipeline pp = VK_NULL_HANDLE; VkResult r = vkCreateGraphicsPipelines(g_dev, g_pcache, 1, &gi, NULL, &pp);
    vkDestroyShaderModule(g_dev, vm, NULL); vkDestroyShaderModule(g_dev, fm, NULL);
    if (r) { RED_ONCE(110, "scene FXAA: pipeline for format %d failed %d", fmt, r); return VK_NULL_HANDLE; }
    FX.pipe[FX.npipe].f = fmt; FX.pipe[FX.npipe].p = pp; FX.npipe++;
    return pp;
}
static int fx_scratch(vtex *x) {   // the copy the pass reads: as the scene, remade when the scene's size or format changes (rare: the device waits)
    if (FX.img && FX.w == x->w && FX.h == x->h && FX.fmt == x->fmt) return 1;
    if (FX.img) { pthread_mutex_lock(&g_qmu); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu); vkDestroyImageView(g_dev, FX.view, NULL); vkDestroyImage(g_dev, FX.img, NULL); vk_free(FX.a); FX.img = VK_NULL_HANDLE; }
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = x->fmt, .extent = { x->w, x->h, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
    if (vkCreateImage(g_dev, &ci, NULL, &FX.img)) { FX.img = VK_NULL_HANDLE; FX.failed = 1; RED_ONCE(111, "scene FXAA: cannot make its %ux%u copy", x->w, x->h); return 0; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, FX.img, &rq); FX.a = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, FX.img, FX.a.b->mem, FX.a.off);
    VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = FX.img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = x->fmt, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCreateImageView(g_dev, &vci, NULL, &FX.view);
    VkDescriptorImageInfo di = { FX.samp, FX.view, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = FX.ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &di };
    vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL);
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = FX.img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(CB(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    FX.w = x->w; FX.h = x->h; FX.fmt = x->fmt;
    VLOG("scene FXAA: the scene is %ux%u, format %d", x->w, x->h, x->fmt);
    return 1;
}
static int fx_float(VkFormat f) { return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R32G32B32A32_SFLOAT || f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32; }
// The tone-mapped scene: the first pass of a frame that drew into one 8-bit colour target the size of the shown texture, with no depth and no other target, ending (the
// game sets other targets). Legends: Arceus tone-maps its HDR scene so into such a texture with one draw, then draws its interface (text boxes, menus) into that same texture and
// copies it to the shown one: smoothed here, between the two, the interface stays sharp
static int fx_ldr(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32; }
static uint64_t g_fx_d0;   // draws when the targets were last set
static void fx_scene(vtex *x);
static void fx_set_rt(const p_set_rt *nr) {   // OP_SET_RT, before the targets change
    uint64_t drew = g_stat_draws - g_fx_d0; g_fx_d0 = g_stat_draws;
    if (!fx_on() || FX.failed || !drew || !R.has_rt || P_tf) return;
    uint64_t fr = atomic_load(&g_fx_frames); if (g_fx_done == fr + 1) return;
    const p_set_rt *r = &R.rt; if (r->n < 1 || !r->color[0] || r->depth || fx_is_shown(r->color[0]->serial)) return;
    for (int i = 1; i < r->n && i < 8; i++) if (r->color[i]) return;
    if (r->has_cview[0] && ((r->cview[0].set_levels && r->cview[0].base_level) || (r->cview[0].set_layers && r->cview[0].min_layer))) return;
    if (rt_same(r, nr)) return;   // the pass goes on
    vtex *x = tex_res(r->color[0]);
    if (!x || !x->img || x->layers != 1 || x->samples > 1 || !fx_ldr(x->fmt) || !(x->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return;
    if (x->w != atomic_load(&g_fx_sw) || x->h != atomic_load(&g_fx_sh)) return;
    g_fx_done = fr + 1;
    fx_scene(x);
}
static void fx_scene(vtex *x) {
    VkPipeline pp = fx_pipe(x->fmt); if (!pp) return;
    VkImageView v = view_get(x, VK_IMAGE_VIEW_TYPE_2D, x->fmt, 0, 1, 0, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT); if (!v) return;
    g_pe_why = 13; pass_end();
    sk_read(x);   // read: its pending store was needed
    if (!fx_scratch(x)) return;
    VkCommandBuffer cb = CB();
    { static int said; if (!said++) VLOG("scene FXAA: the tone-mapped scene (%ux%u format %d) is smoothed before the interface is drawn into it", x->w, x->h, x->fmt); }
    // what drew the scene (and the last frame's FXAA read of the copy) before the copy
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkImageCopy ic = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { x->w, x->h, 1 } };
    vkCmdCopyImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, FX.img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };   // every pixel is written: nothing to load
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, { x->w, x->h } }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp); g_bind_gen++;
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FX.pl, 0, 1, &FX.ds, 0, NULL);
    float pc[4] = { 1.0f / x->w, 1.0f / x->h, (float)fx_float(x->fmt), 0 };
    vkCmdPushConstants(cb, FX.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, pc);
    VkViewport vw = { 0, 0, (float)x->w, (float)x->h, 0, 1 }; VkRect2D sc = { {0, 0}, { x->w, x->h } };
    vkCmdSetViewport(cb, 0, 1, &vw); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static void draw_impl(const p_draw *dw, const p_draw_idx *di);
static void draw(const p_draw *dw, const p_draw_idx *di) { uint64_t pf_t0 = prof_t(); draw_impl(dw, di); prof_add(PF_DRAW, pf_t0); }
// debugging: DM_VK_TRACE_FS=<fragment key>[,<key>...] logs what becomes of the draws using them (the first 20 per reason): where each one leaves draw_impl
// tracing runs from frame DM_VK_TRACE_FROM on, or (DM_VK_TRACE_FROM=file) once <files>/trace.on exists (checked about once a second)
static int trace_now(void) {
    static long from = -2; static int on; static uint64_t next; static char path[1100];
    if (from == -2) { const char *e = getenv("DM_VK_TRACE_FROM"); from = !e ? 0 : !strcmp(e, "file") ? -1 : atol(e);
        const char *pc = getenv("DM_VK_PCACHE"); snprintf(path, sizeof path, "%s", pc ? pc : ""); char *sl = strrchr(path, '/'); if (sl) snprintf(sl + 1, sizeof path - (size_t)(sl + 1 - path), "trace.on"); }
    if (from >= 0) return (long)g_frame_no >= from;
    if (!on) { uint64_t t = mono_ns(); if (t >= next) { next = t + 1000000000ull; on = access(path, F_OK) == 0; } }
    return on;
}
static const char *trace_key(void) {
    static const char *tr = (const char *)1; if (tr == (const char *)1) tr = getenv("DM_VK_TRACE_FS");
    if (!tr || !S.prog || !trace_now()) return NULL;
    vprog *pr = prog_get(S.prog);
    for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->vkstage == VK_SHADER_STAGE_FRAGMENT_BIT && strstr(tr, pr->sh[st]->key)) return pr->sh[st]->key;
    return NULL;
}
#define TRACE_DRAW(k, why) do { if (k) { static int n_; if (n_++ < 20) { VLOG("[trace] draw fs %s: %s (draw %llu, rt0 %llu, cull %d, discard %d)", k, why, (unsigned long long)g_stat_draws, \
    R.has_rt && R.rt.color[0] ? (unsigned long long)R.rt.color[0]->serial : 0ULL, (int)S.poly.cull, (int)S.rast_discard); \
    VLOG("[trace]   viewport %.0f %.0f %.0f %.0f", S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]); \
    for (int q_ = 0; q_ < 3; q_++) { const p_bind_buf *u_ = &S.ubo[4][q_]; float f_[4] = {0}; \
        if (u_->addr && u_->size >= 0x970) memcpy(f_, (const void *)(uintptr_t)(u_->addr + 0x960), 16); \
        VLOG("[trace]   fs cbuf %d: %#llx size %llu, +0x960 = %g %g %g %g", q_, (unsigned long long)u_->addr, (unsigned long long)u_->size, f_[0], f_[1], f_[2], f_[3]); } } } } while (0)
static void draw_impl(const p_draw *dw, const p_draw_idx *di) {
    g_draw_binst = (uint32_t)(dw ? dw->base_instance : di ? di->base_instance : 0);
    const char *tk = trace_key();
    TRACE_DRAW(tk, "entered");
    { static long trt = -2; if (trt == -2) { const char *e = getenv("DM_VK_TRACE_RT"); trt = e ? atol(e) : -1; }   // debugging: DM_VK_TRACE_RT=<serial> logs the draws into that colour target (the first 40)
      if (trt >= 0 && R.has_rt && trace_now()) for (int c = 0; c < 8; c++) if (R.rt.color[c] && (long)R.rt.color[c]->serial == trt) { static int n_; if (n_++ < 40) {
          const char *fk = "-", *vk = "-"; if (S.prog) { vprog *pp = prog_get(S.prog); for (int st = 0; st < 5; st++) if (pp->sh[st]) { if (pp->sh[st]->vkstage == VK_SHADER_STAGE_FRAGMENT_BIT) fk = pp->sh[st]->key; else if (!st) vk = pp->sh[st]->key; } }
          VLOG("[trace] draw %llu into %ld (colour %d): vertex %s fragment %s, prog %#llx, mdi %d, viewport %.0f %.0f %.0f %.0f", (unsigned long long)g_stat_draws, trt, c, vk, fk, (unsigned long long)S.prog, (dw && dw->mdi_cnt) || (di && di->mdi_cnt), S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]); } break; } }
    if ((dw && dw->mdi_cnt) || (di && di->mdi_cnt)) { uint64_t a = dw ? dw->mdi_cnt : di->mdi_cnt; int i = dw ? dw->mdi_i : di->mdi_i; uint32_t n = 0; memcpy(&n, (const void *)(uintptr_t)a, 4); if ((uint32_t)i >= n) { TRACE_DRAW(tk, "indirect count says skip"); return; } }
    if (!S.prog) { RED_ONCE(33, "draw with no program bound"); return; }
    vprog *pr = prog_get(S.prog); if (!pr->sh[0]) { RED_ONCE(34, "program has no vertex shader (or it is not mapped)"); return; }
    { static const char *skip = (const char *)1; if (skip == (const char *)1) skip = getenv("DM_VK_SKIP_FS");   // debugging: DM_VK_SKIP_FS=<fragment key>[,<key>...] skips the draws using them
      if (skip) for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->vkstage == VK_SHADER_STAGE_FRAGMENT_BIT && strstr(skip, pr->sh[st]->key)) return; }
    if (dl_on()) { const char *fk = "-"; for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->vkstage == VK_SHADER_STAGE_FRAGMENT_BIT) fk = pr->sh[st]->key;
        VLOG("[texdump] draw %llu fs %s rt0 %llu mem %p count %d", (unsigned long long)g_stat_draws, fk, R.has_rt && R.rt.color[0] ? (unsigned long long)R.rt.color[0]->serial : 0ULL, R.has_rt && R.rt.color[0] ? (void *)R.rt.color[0]->mem : NULL, dw ? (int)dw->count : di ? (int)di->count : -1); }
    // Rasterizer discard: the vertex stage alone. With transform feedback active the draw captures its outputs (the shader's capture variant stores them into the bound
    // feedback buffers, as nvn_mtl.m tf_draw: Pokémon Legends: Arceus / Z-A skin characters so); otherwise it changes nothing and is skipped
    VkBuffer tfb[4] = {0}; VkDeviceSize tfo[4] = {0}, tfr[4] = {0}; uint32_t tfpc[5] = {0}; int tfcap = 0;
    if (S.rast_discard) {
        vshader *vs = pr->sh[0];
        if (!S.tf_on) { TRACE_DRAW(tk, "rasterizer discard without transform feedback"); return; }
        if (!vs->tfmod) { RED_ONCE(100, "transform feedback with vertex shader %s, which has no capture variant: nothing captured", vs->key); return; }
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
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK && !tfcap) { TRACE_DRAW(tk, "culls front and back"); return; }
    if (g_nubd && ubd_conflict(pr, dw, di)) { g_ubd_conflicts++; g_pe_why = 12; pass_end(); }   // the deferred uniform updates land before this draw (a new pass loads what the old one stored)
    if (tfcap) {   // whether this capture can join the open capture pass: it reads nothing and rewrites nothing an earlier capture there wrote
        int keep = g_ntfw < 60;
        for (int b = 0; keep && b < 4; b++) if (tfb[b]) { uint64_t at = S.tf[b].addr + S.tf_off[b]; if (tf_hits(at, at + (uint64_t)dw->count * pr->sh[0]->tf_stride[b])) keep = 0; }
        for (int s_ = 0; keep && s_ < 16; s_++) if (S.vbuf[s_].address && tf_hits(S.vbuf[s_].address, S.vbuf[s_].address + (S.vbuf[s_].size ? S.vbuf[s_].size : 65536))) keep = 0;
        for (int st = 0; keep && st < 5; st++) for (int q = 0; keep && q < 16; q++) {
            if (S.ubo[st][q].addr && tf_hits(S.ubo[st][q].addr, S.ubo[st][q].addr + (S.ubo[st][q].size ? S.ubo[st][q].size : 65536))) keep = 0;
            if (S.ssbo[st][q].addr && tf_hits(S.ssbo[st][q].addr, S.ssbo[st][q].addr + (S.ssbo[st][q].size ? S.ssbo[st][q].size : 65536))) keep = 0; }
        tf_pass_begin(keep);
    } else if (!pass_begin()) { TRACE_DRAW(tk, "pass_begin failed"); return; }
    TRACE_DRAW(tk, "pass open, continuing");
    if (!tfcap && (P.hd || P.hs)) {
        int st_w = P.hs && S.ds.st && (S.st_wmask[0] || S.st_wmask[1]) && (S.ds.fail_f != NVN_STENCIL_OP_KEEP || S.ds.zpass_f != NVN_STENCIL_OP_KEEP || S.ds.zfail_f != NVN_STENCIL_OP_KEEP ||
                   S.ds.fail_b != NVN_STENCIL_OP_KEEP || S.ds.zpass_b != NVN_STENCIL_OP_KEEP || S.ds.zfail_b != NVN_STENCIL_OP_KEEP);
        pass_writes(P.hd && S.ds.dt && S.ds.dw, st_w);
        if (!P.open) return;
    }
    int prim = dw ? dw->prim : di->prim; VkPrimitiveTopology topo = topo_of(prim);
    VkPipeline p = pso_get(pr, topo); if (!p) return;
    VkCommandBuffer cb = CB();
    { static VkCommandBuffer bcb; static VkPipeline bp; static uint64_t bfence, bgen;   // the same pipeline already bound in this command buffer: no rebind (Vulkan keeps it across passes)
      static int off = -1; if (off < 0) off = getenv("DM_VK_BIND_ALWAYS") != NULL;
      if (off || cb != bcb || bp != p || bfence != g_sub->fence || bgen != g_bind_gen) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); bcb = cb; bp = p; bfence = g_sub->fence; bgen = g_bind_gen; } }
    // dynamic state
    float vx = 0, vy = 0, vw = (float)P.tw, vh = (float)P.th;
    if (S.has_vp) { vx = S.vp[0][0]; vy = S.vp[0][1]; vw = S.vp[0][2]; vh = S.vp[0][3]; }
    VkViewport v = { vx, vy, vw, vh, S.dr[0], S.dr[1] };
    if (!atomic_load(&g_origin_ll)) { v.y = vy + vh; v.height = -vh; }   // NVN upper-left: NDC +1 on the first row (Vulkan puts -1 there)
    if (v.minDepth < 0) v.minDepth = 0; if (v.minDepth > 1) v.minDepth = 1; if (v.maxDepth < 0) v.maxDepth = 0; if (v.maxDepth > 1) v.maxDepth = 1;
    vkCmdSetViewport(cb, 0, 1, &v);
    if (atomic_load(&g_passlog) == (long)atomic_load(&g_fx_frames)) { static uint64_t last; if (last != g_stat_passes) { last = g_stat_passes;   // the pass log: the area the set's first draw covers
        VLOG("[passlog]   first draw: viewport %.0f,%.0f %.0fx%.0f, scissor %s%d,%d %dx%d (targets %ux%u)", vx, vy, vw, vh, R.has_sc ? "" : "(none) ", R.has_sc ? R.sc.x : 0, R.has_sc ? R.sc.y : 0, R.has_sc ? R.sc.w : 0, R.has_sc ? R.sc.h : 0, P.tw, P.th); } }
    VkRect2D sc = {{0, 0}, {P.tw, P.th}};
    if (R.has_sc && !tfcap) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h; if (x1 > (long)P.tw) x1 = P.tw; if (y1 > (long)P.th) y1 = P.th;
        if (x1 <= x0 || y1 <= y0) return; sc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}}; }
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdSetCullMode(cb, S.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : S.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);
    int ccw = S.poly.front == NVN_FRONT_FACE_CCW; if (atomic_load(&g_origin_ll)) ccw = !ccw;   // lower-left: the y flip relative to GL window coordinates reverses the winding
    vkCmdSetFrontFace(cb, ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, topo);
    if (!tfcap) {   // a capture pipeline (rasterizer discard) has no depth / stencil state to set (VUID-vkCmdDraw-None-08608)
    vkCmdSetDepthTestEnable(cb, P.hd && S.ds.dt); vkCmdSetDepthWriteEnable(cb, P.hd && S.ds.dw); vkCmdSetDepthCompareOp(cb, (VkCompareOp)(S.ds.df ? S.ds.df - 1 : 7));
    vkCmdSetStencilTestEnable(cb, P.hs && S.ds.st);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)(S.ds.fail_f - 1), (VkStencilOp)(S.ds.zpass_f - 1), (VkStencilOp)(S.ds.zfail_f - 1), (VkCompareOp)(S.ds.sf_f - 1));
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)(S.ds.fail_b - 1), (VkStencilOp)(S.ds.zpass_b - 1), (VkStencilOp)(S.ds.zfail_b - 1), (VkCompareOp)(S.ds.sf_b - 1));
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_vmask[0]); vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_vmask[1]);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_wmask[0]); vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_wmask[1]);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_ref[0]); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, S.st_ref[1]);
    if (S.poly.offset_en & 4) vkCmdSetDepthBias(cb, S.po.units, S.po.clamp, S.po.factor); else vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetBlendConstants(cb, S.blend_color);
    }
    // vertex streams: the pool buffers at their addresses, strides from the stream state; binding 16 = zeros for unbound attributes
    VkBuffer vb[17]; VkDeviceSize vo[17], vs_[17];
    for (int s_ = 0; s_ < 16; s_++) { VkBuffer b; VkDeviceSize o;
        if (S.vbuf[s_].address && pool_buf(S.vbuf[s_].address, 1, &b, &o, NULL)) { vb[s_] = b; vo[s_] = o; } else { vb[s_] = g_zero_buf; vo[s_] = 0; }
        vs_[s_] = S.stream[s_].stride; }
    vb[16] = g_zero_buf; vo[16] = 0; vs_[16] = 0;
    vkCmdBindVertexBuffers2(cb, 0, 17, vb, vo, NULL, vs_);
    VkPipelineLayout pl = prog_layout(pr, 0); g_np = pr->np[0];
    for (int st = 0; st < 5; st++) push_stage(pr->sh[st], st, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    if (tfcap) {   // the capture buffers, then the draw, then its writes made visible to what reads them (vertex streams, shaders, copies)
        VkWriteDescriptorSet w_[4]; VkDescriptorBufferInfo bi_[4]; uint32_t nw = 0;
        VkWriteDescriptorSet *w = w_; VkDescriptorBufferInfo *bi = bi_;
        if (g_np) { if (D.n + 4 > 320) D.n = 316; w = D.w + D.n; bi = D.bi + D.n; }
        for (int b = 0; b < 4; b++) if (tfb[b]) { bi[nw] = (VkDescriptorBufferInfo){ tfb[b], tfo[b], tfr[b] };
            w[nw] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = TF_BIND + (uint32_t)b, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[nw] }; nw++; }
        if (g_np) D.n += nw; else if (nw) p_push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, nw, w);
        desc_flush(VK_PIPELINE_BIND_POINT_GRAPHICS, pl, pr->dsl[0]);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof tfpc, tfpc);
        vkCmdDraw(cb, (uint32_t)dw->count, 1, (uint32_t)dw->first, (uint32_t)dw->base_instance);
        vshader *vs = pr->sh[0];
        for (int b = 0; b < 4; b++) if (tfb[b]) { uint64_t n = (uint64_t)dw->count * vs->tf_stride[b], at = S.tf[b].addr + S.tf_off[b]; wb_note(at, n); S.tf_off[b] += n;
            if (g_ntfw < 64) { g_tfw[g_ntfw].lo = at; g_tfw[g_ntfw].hi = at + n; g_ntfw++; } }
        if (!tf_batch_on()) { g_pe_why = 11; pass_end(); }   // (the pass stays open for the next capture; pass_end makes the writes visible)
        { static int once; if (!once++) VLOG("transform feedback: first capture (vertex %s, %d vertices)", vs->key, dw->count); }
        return;
    }
    desc_flush(VK_PIPELINE_BIND_POINT_GRAPHICS, pl, pr->dsl[0]);
    uint32_t inst = (uint32_t)((dw ? dw->instances : di->instances) > 0 ? (dw ? dw->instances : di->instances) : 1);
    if ((dw && dw->ind) || (di && di->ind)) {
        VkBuffer ab; VkDeviceSize ao; uint64_t ind = dw ? dw->ind : di->ind;
        if (prim == 6 || prim == 7) { RED_ONCE(35, "indirect fan / quad draw not supported yet"); return; }
        if (!pool_buf(ind, dw ? 16 : 20, &ab, &ao, NULL)) { RED_ONCE(36, "indirect arguments in no pool"); return; }
        if (dw) vkCmdDrawIndirect(cb, ab, ao, 1, 16);
        else { VkBuffer ib; VkDeviceSize io; if (di->itype == 0) { RED_ONCE(37, "indirect draw with 8-bit indices not supported yet"); return; }
               if (!pool_buf(di->ibuf, 1, &ib, &io, NULL)) { RED_ONCE(38, "index buffer in no pool"); return; }
               vkCmdBindIndexBuffer(cb, ib, io, di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32); vkCmdDrawIndexedIndirect(cb, ab, ao, 1, 20); }
        return;
    }
    if (dw && (prim == 6 || prim == 7)) {   // fans / quads: expanded into a 32-bit triangle index list in the arena
        uint32_t nt = prim == 7 ? (uint32_t)(dw->count / 4) * 2 : dw->count >= 3 ? (uint32_t)dw->count - 2 : 0; if (!nt) return;
        VkBuffer ib; VkDeviceSize io; uint32_t *d = (uint32_t *)arena_alloc(g_sub, (VkDeviceSize)nt * 12, &ib, &io);
        for (uint32_t t = 0; t < nt; t++) { if (prim == 6) { d[3 * t] = 0; d[3 * t + 1] = t + 1; d[3 * t + 2] = t + 2; } else { uint32_t q = (t >> 1) * 4, hh = t & 1; d[3 * t] = q; d[3 * t + 1] = q + 1 + hh; d[3 * t + 2] = q + 2 + hh; } }
        vkCmdBindIndexBuffer(cb, ib, io, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, nt * 3, inst, 0, dw->first, (uint32_t)dw->base_instance); return;
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
            vkCmdBindIndexBuffer(cb, ib, io, VK_INDEX_TYPE_UINT32); n = nt * 3;
        } else {
            uint16_t *d = (uint16_t *)arena_alloc(g_sub, (VkDeviceSize)di->count * 2, &ib, &io); for (int32_t i = 0; i < di->count; i++) d[i] = src[i];
            vkCmdBindIndexBuffer(cb, ib, io, VK_INDEX_TYPE_UINT16); n = (uint32_t)di->count;
        }
        #undef IX
        vkCmdDrawIndexed(cb, n, inst, 0, di->base_vertex, (uint32_t)di->base_instance); return;
    }
    VkBuffer ib; VkDeviceSize io;
    if (!pool_buf(di->ibuf, (uint64_t)di->count * isz, &ib, &io, NULL)) { RED_ONCE(38, "index buffer in no pool"); return; }
    vkCmdBindIndexBuffer(cb, ib, io, di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
}
static nvn_hmap g_cps = {.klen = sizeof(void *)};
static void dispatch_impl(const p_dispatch *dp);
static void dispatch(const p_dispatch *dp) { uint64_t pf_t0 = prof_t(); dispatch_impl(dp); prof_add(PF_DISPATCH, pf_t0); }
static void dispatch_impl(const p_dispatch *dp) {
    if (!S.cprog) { RED_ONCE(39, "dispatch with no program bound"); return; }
    vprog *pr = prog_get(S.cprog); vshader *cs = pr->sh[5]; if (!cs) { RED_ONCE(40, "dispatched program has no compute shader (or it is not mapped)"); return; }
    VkPipelineLayout pl = prog_layout(pr, 1); g_np = pr->np[1];
    void *key = pr; VkPipeline p = nvn_hm_get(&g_cps, &key);
    if (!p) { VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs->mod, .pName = "main" }, .layout = pl };
        if (vkCreateComputePipelines(g_dev, g_pcache, 1, &ci, NULL, &p)) { VLOG("[RED] compute pipeline %s failed", cs->key); return; }
        nvn_hm_put(&g_cps, &key, p); }
    g_pe_why = 1; outside_pass();
    VkCommandBuffer cb = CB();
    g_dispatch_dim[0] = (uint32_t)dp->x; g_dispatch_dim[1] = (uint32_t)dp->y; g_dispatch_dim[2] = (uint32_t)dp->z; g_c0_buf = VK_NULL_HANDLE;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    push_stage(cs, 5, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
    desc_flush(VK_PIPELINE_BIND_POINT_COMPUTE, pl, pr->dsl[1]);
    if (dp->ind) { VkBuffer ab; VkDeviceSize ao; if (!pool_buf(dp->ind, 12, &ab, &ao, NULL)) { RED_ONCE(41, "indirect dispatch arguments in no pool"); return; }
        if (g_c0_buf) { VkBufferCopy c = { ao, g_c0_off, 12 }; vkCmdCopyBuffer(cb, ab, g_c0_buf, 1, &c); barrier_all(cb); }   // the shader reads the counts from c0 + 0x200
        vkCmdDispatchIndirect(cb, ab, ao); }
    else vkCmdDispatch(cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    barrier_all(cb);
}

// ================= Buffer operations (in submission order) =================
static void sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!pool_buf(src, n, &sb, &so, NULL) || !pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(42, "buffer copy %#llx → %#llx outside the pools", (unsigned long long)src, (unsigned long long)dst); return; }
    g_pe_why = 2; outside_pass();
    if (sb == db && so < dof + n && dof < so + n) {   // overlapping ranges of one buffer: through the arena
        VkBuffer tb; VkDeviceSize to; arena_alloc(g_sub, n, &tb, &to);
        VkBufferCopy c1 = { so, to, n }; vkCmdCopyBuffer(CB(), sb, tb, 1, &c1); barrier_all(CB()); VkBufferCopy c2 = { to, dof, n }; vkCmdCopyBuffer(CB(), tb, db, 1, &c2);
    } else { VkBufferCopy c = { so, dof, n }; vkCmdCopyBuffer(CB(), sb, db, 1, &c); }
    barrier_all(CB()); wb_note(dst, n);
}
static void sub_update(const void *data, uint64_t dst, uint64_t n) {   // the data is copied now (the game may reuse its command memory before the GPU runs)
    if (!n) return;
    VkBuffer db; VkDeviceSize dof; if (!pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(43, "uniform update outside the pools"); return; }
    if (P.open && !P_tf && g_nubd < UBD_MAX && ubd_on()) {   // kept until the pass ends (see ubd_flush); draws after it read it through ubd_shadow
        VkBuffer tb; VkDeviceSize to; uint8_t *m = arena_alloc(g_sub, n, &tb, &to); memcpy(m, data, n);
        g_ubd[g_nubd++] = (ubd_ent){ dst, dst + n, tb, to, m }; g_ubd_deferred++; return; }
    g_pe_why = 3; outside_pass();
    VkBuffer tb; VkDeviceSize to; void *m = arena_alloc(g_sub, n, &tb, &to); memcpy(m, data, n);
    VkBufferCopy c = { to, dof, n }; vkCmdCopyBuffer(CB(), tb, db, 1, &c); barrier_all(CB()); wb_note(dst, n);
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n) return;
    VkBuffer db; VkDeviceSize dof; if (!pool_buf(dst, n, &db, &dof, NULL)) { RED_ONCE(44, "buffer fill outside the pools"); return; }
    g_pe_why = 4; outside_pass();
    if (dof % 4 || n % 4) { RED_ONCE(45, "buffer fill not 4-byte aligned (%#llx, %llu)", (unsigned long long)dof, (unsigned long long)n); return; }
    vkCmdFillBuffer(CB(), db, dof, n, value); barrier_all(CB()); wb_note(dst, n);
}

// ================= Texture copies and clears =================
static void copy_b2t(const p_copy_b2t *c, int to_buffer) {
    vtex *x = tex_res(c->tex); if (!x || !x->img) return;
    sk_read(x);   // copied from, or written in part
    if (x->drop) { RED_ONCE(104, "a texture kept at half quality copied %s a buffer: left out (DM_VK_TEX_QUALITY=full to compare)", to_buffer ? "to" : "from"); return; }
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;
    int nz = c->r.depth ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize o; if (!pool_buf(c->src, img * nz, &b, &o, NULL)) { RED_ONCE(46, "texture copy buffer side outside the pools"); return; }
    if (row % bpb || (img % row)) { RED_ONCE(47, "texture copy pitch %zu not a whole number of %d-byte blocks", row, bpb); return; }
    int bck = nvn_fmt_vk_decode(c->tex->d.format);
    if (bck && to_buffer) { RED_ONCE(98, "a BC texture decoded on upload copied back to a buffer: skipped (format %#x)", c->tex->d.format); return; }
    if (bck) {   // BC blocks the game copies in: decoded now (the CPU has written them) into the submission's arena, then copied as plain texels
        const uint8_t *src = (const uint8_t *)(uintptr_t)c->src; int tb = bc_texel_bytes(bck); size_t orow = (size_t)c->r.width * tb, oimg = orow * (size_t)c->r.height;
        VkBuffer sb; VkDeviceSize so; uint8_t *sp = arena_alloc(g_sub, oimg * nz ? oimg * nz : 16, &sb, &so);
        for (int s = 0; s < nz; s++) bc_decode_image(bck, src + s * img, row, c->r.width, c->r.height, sp + s * oimg, orow);
        b = sb; o = so; row = orow; img = oimg; bpb = tb; bw = bh = 1;
    }
    g_pe_why = 5; outside_pass();
    VkBufferImageCopy r = { .bufferOffset = o, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(img / row * bh),
        .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, level, x->is3d ? 0 : layer0 + (uint32_t)c->r.zoffset, x->is3d ? 1 : (uint32_t)nz },
        .imageOffset = { c->r.xoffset, c->r.yoffset, x->is3d ? c->r.zoffset : 0 }, .imageExtent = { (uint32_t)c->r.width, (uint32_t)c->r.height, x->is3d ? (uint32_t)nz : 1 } };
    if (to_buffer) vkCmdCopyImageToBuffer(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r); else vkCmdCopyBufferToImage(CB(), b, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier_all(CB()); if (to_buffer) wb_note(c->src, img * nz);
}
static void copy_t2t(const p_copy_t2t *c) {
    g_pe_why = 6; pass_flush();
    vtex *s = tex_res(c->src), *d = tex_res(c->dst); if (!s || !d || !s->img || !d->img) return;
    if (s->drop || d->drop) { RED_ONCE(105, "a texture kept at half quality in a texture copy: left out (DM_VK_TEX_QUALITY=full to compare)"); return; }
    sk_read(s); sk_read(d);
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    int nz = c->sr.depth ? c->sr.depth : 1;
    g_pe_why = 6; outside_pass();
    VkImageSubresourceLayers ss = { s->aspect, sl, s->is3d ? 0 : sz0 + (uint32_t)c->sr.zoffset, s->is3d ? 1 : (uint32_t)nz }, ds = { d->aspect, dl, d->is3d ? 0 : dz0 + (uint32_t)c->dr.zoffset, d->is3d ? 1 : (uint32_t)nz };
    if (c->flags & DM_COPY_RESOLVE) {
        if (s->samples > 1 && d->samples == 1 && s->fmt == d->fmt) { VkImageResolve r = { ss, {0, 0, 0}, ds, {0, 0, 0}, { s->w, s->h, 1 } }; vkCmdResolveImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
        else if (s->samples == d->samples && s->fmt == d->fmt) { VkImageCopy r = { ss, {0, 0, 0}, ds, {0, 0, 0}, { s->w, s->h, 1 } }; vkCmdCopyImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
        else RED_ONCE(48, "downsample case not implemented");
        barrier_all(CB()); return;
    }
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
        else vkCmdBlitImage(CB(), s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    barrier_all(CB());
}
static void clear_tex(const p_clear_tex *c) {
    vtex *x = tex_res(c->tex); if (!x || !x->img) return;
    sk_read(x);   // (counted as a read: the pending store keeps storing)
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    int nz = c->r.depth ? c->r.depth : 1;
    uint32_t w = x->w >> lv ? x->w >> lv : 1, h = x->h >> lv ? x->h >> lv : 1;
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width != w || (uint32_t)c->r.height != h || (c->mask & 0xf) != 0xf) RED_ONCE(50, "partial texture clear (region / mask): cleared whole");
    g_pe_why = 7; outside_pass();
    VkImageSubresourceRange rg = { x->aspect, lv, 1, x->is3d ? 0 : l0 + (uint32_t)c->r.zoffset, x->is3d ? 1 : (uint32_t)nz };
    if (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue v; memcpy(&v, c->c.u, 16); vkCmdClearColorImage(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    else { VkClearDepthStencilValue v = { c->c.f[0], x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? c->c.u[1] : c->c.u[0] }; vkCmdClearDepthStencilImage(CB(), x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    barrier_all(CB());
}
static int scissor_full(const tex_t *t, const view_t *v, int has_v) {
    unsigned lv = has_v && v->set_levels ? v->base_level : 0; uint32_t w = t->d.width >> lv ? t->d.width >> lv : 1, h = (t->d.target == 0 || t->d.target == 3 ? 1 : t->d.height) >> lv; if (!h) h = 1;
    return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)w && (int64_t)R.sc.y + R.sc.h >= (int64_t)h);
}
static void clear_in_pass(int color_index, const VkClearValue *v, VkImageAspectFlags asp) {   // a clear inside an open pass (or a partial one): vkCmdClearAttachments over the scissor
    if (!pass_begin()) return;
    VkClearAttachment a = { asp, (uint32_t)(color_index < 0 ? 0 : color_index), *v };
    VkRect2D rc = {{0, 0}, {P.tw, P.th}};
    if (R.has_sc) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h; if (x1 > (long)P.tw) x1 = P.tw; if (y1 > (long)P.th) y1 = P.th;
        if (x1 <= x0 || y1 <= y0) return; rc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}}; }
    VkClearRect cr = { rc, 0, 1 };
    vkCmdClearAttachments(CB(), 1, &a, 1, &cr);
}

// ================= Command replay =================
static void bh_fresh(const p_bind_handles *b) { if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); }
static int state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: { const p_bind_program *b = a; if (b->stages & 0x20) S.cprog = b->prog; if (b->stages & 0x1f) S.prog = bind_stages(b->prog, b->stages & 0x1f);
            { const char *k = trace_key(); if (k) { static int n_; if (n_++ < 20) VLOG("[trace] program with fs %s bound (stages %#x, prog %#llx)", k, (unsigned)b->stages, (unsigned long long)b->prog); } }
            break; }
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
// DM_VK_PASSLOG=<present>: every render-target set of that frame to the log (the targets: serial, size, format, view level / layer; the draws made on the set before it), to see
// why passes end ("new targets" ends nearly all of them in Legends: Arceus battles)
static int rt_same(const p_set_rt *a, const p_set_rt *b);
static void pass_log(const p_set_rt *r) {
    long at = atomic_load(&g_passlog);
    if (!at || (long)atomic_load(&g_fx_frames) != at) return;
    static uint64_t d0; char ln[1024]; size_t o = (size_t)snprintf(ln, sizeof ln, "[passlog] +%llu draws | set:", (unsigned long long)(g_stat_draws - d0)); d0 = g_stat_draws;
    for (int i = 0; i < r->n && i < 8 && o < sizeof ln - 96; i++) { const tex_t *t = r->color[i]; if (!t) { o += (size_t)snprintf(ln + o, sizeof ln - o, " c%d -", i); continue; }
        vtex *x = tex_res(t); o += (size_t)snprintf(ln + o, sizeof ln - o, " c%d %llu @%p %ux%u f%d l%u/%u", i, (unsigned long long)t->serial, (void *)t->mem, x ? x->w : 0, x ? x->h : 0, x ? (int)x->fmt : 0,
                                                     r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0, r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0); }
    if (r->depth) { vtex *x = tex_res(r->depth); o += (size_t)snprintf(ln + o, sizeof ln - o, " d %llu @%p %ux%u l%u", (unsigned long long)r->depth->serial, (void *)r->depth->mem, x ? x->w : 0, x ? x->h : 0,
                                                                        r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0); }
    o += (size_t)snprintf(ln + o, sizeof ln - o, "%s", P.open && R.has_rt && rt_same(&R.rt, r) ? " (same: kept)" : "");
    VLOG("%s", ln);
}
// The same render targets set again while their pass is open: the pass goes on (on a tiler every pass end stores the attachments and every begin loads them back;
// Legends: Arceus sets render targets about 120 times a frame). Views compared field by field where set
// Textures by serial, not by pointer: each SetRenderTargets records snapshots of the texture objects (hle_nvn.c tex_snap), so the same target set again never had the same pointers
// and no pass was ever kept (Legends: Arceus: 968 passes for 1105 draws a frame in the field). Views by the fields that are set (the game's 16-byte view struct is copied as it is)
static uint64_t rt_ser(const tex_t *t) { return t ? t->serial : 0; }
static int view_eq(const view_t *a, const view_t *b) {
    if (a->set_levels != b->set_levels || a->set_layers != b->set_layers || a->set_format != b->set_format || a->set_target != b->set_target) return 0;
    if (a->set_levels && (a->base_level != b->base_level || a->nlevels != b->nlevels)) return 0;
    if (a->set_layers && (a->min_layer != b->min_layer || a->nlayers != b->nlayers)) return 0;
    if (a->set_format && a->format != b->format) return 0;
    if (a->set_target && a->target != b->target) return 0;
    return 1;
}
static int rt_same(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != b->n || rt_ser(a->depth) != rt_ser(b->depth) || a->has_dview != b->has_dview || (a->has_dview && !view_eq(&a->dview, &b->dview))) return 0;
    for (int i = 0; i < a->n && i < 8; i++) if (rt_ser(a->color[i]) != rt_ser(b->color[i]) || a->has_cview[i] != b->has_cview[i] || (a->has_cview[i] && !view_eq(&a->cview[i], &b->cview[i]))) return 0;
    return 1;
}
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
static int rt_on(void);
// The replay of a submission took 8–9 ms in Legends: Arceus and the GPU started only at its end (one vkQueueSubmit), then ran 22 ms: the two one after the other, and the game
// waits for both every frame. At a change of render targets, once the piece recorded so far took DM_VK_SPLIT_MS (2 ms; 0: never), it is submitted and recording goes on in
// the next command buffer: the GPU works on the frame while the rest is replayed. The game's fence is the last piece's (mtl_submit / the render thread)
static void sub_split(void) {
    static double ms = -1; if (ms < 0) { const char *e = getenv("DM_VK_SPLIT_MS"); ms = e ? atof(e) : 2.0; if (ms > 0) VLOG("submissions handed to the GPU in pieces of %.1f ms of replay", ms); }
    if (ms <= 0 || !g_sub || P_tf || mono_ns() - g_sub_t0 < (uint64_t)(ms * 1e6)) return;
    g_pe_why = 9; pass_end(); mtl_sub_commit(); mtl_sub_begin(); g_splits++;
}
void mtl_exec(const uint8_t *p) {
    if (!g_sub) { RED_ONCE(52, "commands replayed outside a submission"); return; }
    for (int guard = 0; guard < (1 << 26); guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END: return;
            case OP_BEGIN: break;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); mtl_exec(t); break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); post((post_op){1, val, m, v, 0, 0}); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); post((post_op){2, NULL, 0, 0, ad, (uint32_t)t}); break; }
            case OP_RESET_COUNTER: case OP_RENDER_COND: break;
            case OP_FENCE_SYNC: if (!rt_on()) { sync_t *s; memcpy(&s, a, 8); s->fence = g_fence + 1; s->fenced = 1; } break;   // (render thread: fenced when queued, rt_scan)
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
            case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); sub_copy(s, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); sub_update(a + 16, d, n); break; }
            case OP_SET_RT: g_stat_passes++; pass_log((const p_set_rt *)a); fx_set_rt((const p_set_rt *)a); if (P.open && !P_tf && R.has_rt && !getenv_nomerge() && rt_same(&R.rt, (const p_set_rt *)a)) { g_stat_rt_same++; break; }
                sub_split();   // a piece of the frame long enough: to the GPU now, the rest follows
                g_pe_why = 8; pass_flush(); R.rt = *(const p_set_rt *)a; R.has_rt = 1; break;
            case OP_CLEAR_COLOR: {
                if (P_tf) { g_pe_why = 11; pass_end(); }   // not into a capture pass
                const p_clear_color *c = (const p_clear_color *)a;
                if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(53, "colour clear target %d invalid", c->index); break; }
                VkClearColorValue cv; memcpy(cv.float32, c->rgba, 16);
                if (dl_on()) VLOG("[texdump] clear c%d ser=%llu to %g %g %g %g mask %#x", c->index, (unsigned long long)R.rt.color[c->index]->serial, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3], c->mask);
                { vtex *cx = tex_res(R.rt.color[c->index]); VkFormat cf = cx ? cx->fmt : VK_FORMAT_UNDEFINED;
                  if (R.rt.has_cview[c->index] && R.rt.cview[c->index].set_format) { uint32_t vf = 0; uint8_t sw[4]; if (nvn_fmt_to_vk(R.rt.cview[c->index].format, &vf, sw) && vf) cf = (VkFormat)vf; }
                  cv = clear_for(cf, c->rgba); }
                if (P.open || c->mask != 0xf || !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) {
                    if (c->mask != 0xf) RED_ONCE(54, "colour clear with a partial channel mask: all channels cleared");
                    VkClearValue v = { .color = cv }; clear_in_pass(c->index, &v, VK_IMAGE_ASPECT_COLOR_BIT); break; }
                R.clear_c[c->index] = 1; R.cc[c->index] = cv; break;
            }
            case OP_CLEAR_DS: {
                if (P_tf) { g_pe_why = 11; pass_end(); }
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
            case OP_DRAW: draw((const p_draw *)a, NULL); g_stat_draws++; break;
            case OP_DRAW_IDX: draw(NULL, (const p_draw_idx *)a); g_stat_draws++; break;
            case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a, 0); break;
            case OP_COPY_T2B: copy_b2t((const p_copy_b2t *)a, 1); break;
            case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
            case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
            case OP_DISPATCH: dispatch((const p_dispatch *)a); break;
            case OP_BARRIER: { int32_t bits; memcpy(&bits, a, 4); if (bits) g_game_bar = 1; break; }   // ordered at the next draw pass (pass_begin)
            case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: state_op(h.op, a); break;
        }
        p += h.size;
    }
    RED_ONCE(55, "command list did not end");
}

// ================= CPU texture reads =================
static void vk_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) return;
    if (nvn_fmt_vk_decode(t->d.format)) { RED_ONCE(99, "CPU read of a BC texture decoded on upload (format %#x): not available", t->d.format); return; }
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh; int nz = r->depth ? r->depth : 1;
    vtex *x = tex_res(t); if (!x || !x->img) return;
    sk_read(x);
    VkDeviceSize n = wb * hb * nz; vmalloc_t a; void *m; VkBuffer b = buf_make(n ? n : 16, &a, &m);
    VkCommandBuffer cb = once_begin();
    VkBufferImageCopy rg = { .bufferOffset = 0, .imageSubresource = { x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, level, x->is3d ? 0 : layer0 + (uint32_t)r->zoffset, x->is3d ? 1 : (uint32_t)nz },
        .imageOffset = { r->xoffset, r->yoffset, x->is3d ? r->zoffset : 0 }, .imageExtent = { (uint32_t)r->width, (uint32_t)r->height, x->is3d ? (uint32_t)nz : 1 } };
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &rg);
    once_end(cb);
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, (uint8_t *)m + (z * hb + y) * wb, wb);
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
static void shot(vtex *x, const int crop[4], const char *path) { sk_read(x); shot_ex(x, crop, path, 0, 0); }
static void shot_level(vtex *x, uint32_t lv, uint32_t sl, const char *path) { shot_ex(x, NULL, path, lv, sl); }
static void shot_ex(vtex *x, const int crop[4], const char *path, uint32_t lv, uint32_t sl) {
    uint32_t lw = x->w >> lv ? x->w >> lv : 1, lh = x->h >> lv ? x->h >> lv : 1;
    uint32_t cx = crop ? (uint32_t)crop[0] : 0, cy = crop ? (uint32_t)crop[1] : 0, w = crop && crop[2] ? (uint32_t)crop[2] : lw, h = crop && crop[3] ? (uint32_t)crop[3] : lh;
    vk_finish();
    if (!(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || x->samples > 1 || fmt_int_class(x->fmt)) { VLOG("[texdump] %s: depth / multisample image not saved", path); return; }
    // a blit into an RGBA8 image converts any colour format, then a copy into host memory
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
    VkImage im; VKCHECK(vkCreateImage(g_dev, &ci, NULL, &im)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, im, &rq); vmalloc_t ia = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); vkBindImageMemory(g_dev, im, ia.b->mem, ia.off);
    vmalloc_t ba; void *m; VkBuffer b = buf_make((VkDeviceSize)w * h * 4, &ba, &m);
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
    for (uint32_t i = 0; i < w * h; i++) { uint8_t *q = (uint8_t *)m + 4 * i; if (bgra8) { uint8_t t = q[0]; q[0] = q[2]; q[2] = t; } q[3] = 255; }
    png_write(path, m, w, h, atomic_load(&g_origin_ll));
    vkDestroyImage(g_dev, im, NULL); vk_free(ia); vkDestroyBuffer(g_dev, b, NULL); vk_free(ba);
    VLOG("screenshot %s (%ux%u)", path, w, h);
}
// ---------- Window presentation (Android: the app's ANativeWindow, front_metal_layer()) ----------
// A FIFO swapchain the size of the window; each present blits the game's window texture into it, scaled to fit with black bars, flipped in GL (lower-left) mode,
// through a swapchain format of the same sRGB-ness as the game's texture so its bytes reach the screen unchanged
void *front_metal_layer(void);
// ext: the window as the player sees it; phys: the swapchain images (ext turned a quarter when the swapchain is pre-rotated); rot: cos, sin of the pre-rotation
static struct { void *win; VkSurfaceKHR surf; VkSwapchainKHR sc; VkImage img[8]; VkImageView view[8]; uint32_t n; VkFormat fmt; VkExtent2D ext, phys; float rot[2]; int srgb;
                VkSemaphore acq[3], done[3]; VkFence fence[3]; VkCommandPool pool; VkCommandBuffer cb[3]; uint32_t slot; } W;
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
    if (W.sc && W.win == win && W.srgb == srgb) return 1;
    return swap_job_run(1, win, srgb);
}
static void swap_destroy_now(void) {
    if (W.sc) { vkDeviceWaitIdle(g_dev); for (uint32_t i = 0; i < W.n; i++) if (W.view[i]) { vkDestroyImageView(g_dev, W.view[i], NULL); W.view[i] = VK_NULL_HANDLE; }
                vkDestroySwapchainKHR(g_dev, W.sc, NULL); W.sc = VK_NULL_HANDLE; }
    if (W.surf) { vkDestroySurfaceKHR(g_inst, W.surf, NULL); W.surf = VK_NULL_HANDLE; }
    W.win = NULL;
}
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
    // The window's images at most DM_VK_SURFACE_H lines (1080: the game's own picture, drawn 1:1; 0: the screen's), the screen's shape kept: Android scales them to the window in the
    // display hardware for nothing. The present pass (picture, sharpening, touch controls) used to fill the S25+'s 3120x1440, 4.5 million pixels a frame, for a 1920x1080 picture
    { const char *e = getenv("DM_VK_SURFACE_H"); uint32_t want = e ? (uint32_t)atoi(e) : 1080, sh = W.ext.width < W.ext.height ? W.ext.width : W.ext.height;
      if (want && sh > want) { double k = (double)want / sh; VkExtent2D e2 = { (uint32_t)(W.ext.width * k + 0.5), (uint32_t)(W.ext.height * k + 0.5) };
          if (e2.width >= caps.minImageExtent.width && e2.height >= caps.minImageExtent.height && (!caps.maxImageExtent.width || (e2.width <= caps.maxImageExtent.width && e2.height <= caps.maxImageExtent.height)) &&
              e2.width <= (caps.maxImageExtent.width ? caps.maxImageExtent.width : e2.width) ) {
              VLOG("window images %ux%u for a %ux%u window (scaled by the display hardware; DM_VK_SURFACE_H=0 for the full size)", e2.width, e2.height, W.ext.width, W.ext.height); W.ext = e2; } } }
    // MAILBOX when the window has it: the game paces its own frames (mtl_present), and with FIFO a slow compositor (a screen recording, the notification shade) held
    // presentation, which shares the queue lock with the game's submissions, so the whole game waited on the compositor (LAS Recomp's Android port measured 4–21 fps with
    // the GPU mostly idle under FIFO, a steady rate with MAILBOX). DM_VK_PRESENT=fifo|mailbox|immediate to choose
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    { const char *e = getenv("DM_VK_PRESENT"); VkPresentModeKHR want = !e || !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
      uint32_t npm = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, NULL); VkPresentModeKHR pms[16]; if (npm > 16) npm = 16; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, pms);
      for (uint32_t i = 0; i < npm; i++) if (pms[i] == want) pm = want; }
    uint32_t n = caps.minImageCount + (pm == VK_PRESENT_MODE_MAILBOX_KHR ? 2 : 1); if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;
    // Pre-rotation: a phone whose panel is portrait shows a landscape game turned a quarter, and an identity swapchain leaves that turn to the compositor (a GPU composition pass
    // or the display's rotator, every frame). With the panel's own transform the picture is drawn turned (the overlay vertex shader's rot) into images of the panel's shape, and the
    // compositor shows them as they are. DM_VK_PREROTATE=0 off; 90 / 180 / 270 forces that turn (testing: the compositor turns back what differs from the panel's)
    VkSurfaceTransformFlagBitsKHR tr = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform; int deg = 0;
    { const char *e = getenv("DM_VK_PREROTATE"); int want = !e ? -1 : atoi(e);
      VkSurfaceTransformFlagBitsKHR t = want < 0 ? caps.currentTransform : want == 90 ? VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR : want == 180 ? VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR :
                                        want == 270 ? VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR : VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
      int d = t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? 90 : t == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? 180 : t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ? 270 : 0;
      if (d && want != 0 && (caps.supportedTransforms & t)) { tr = t; deg = d; } }
    W.rot[0] = deg == 180 ? -1.0f : deg ? 0.0f : 1.0f; W.rot[1] = deg == 90 ? 1.0f : deg == 270 ? -1.0f : 0.0f;
    W.phys = deg == 90 || deg == 270 ? (VkExtent2D){ W.ext.height, W.ext.width } : W.ext;
    VLOG("window: %ux%u, surface transform %#x (supported %#x), current extent %ux%u; pre-rotation %d", W.ext.width, W.ext.height, caps.currentTransform, caps.supportedTransforms,
         caps.currentExtent.width, caps.currentExtent.height, deg);
    VkSwapchainCreateInfoKHR ci = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = W.surf, .minImageCount = n, .imageFormat = W.fmt, .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = W.phys, .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = tr,
        .compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, .presentMode = pm, .clipped = VK_TRUE };
    VkResult r = vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc);
    if (r) { RED_ONCE(58, "swapchain creation failed: %d", r); return 0; }
    W.n = 8; vkGetSwapchainImagesKHR(g_dev, W.sc, &W.n, W.img);
    for (uint32_t i = 0; i < W.n; i++) { VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = W.img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = W.fmt,
                                           .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } }; vkCreateImageView(g_dev, &vci, NULL, &W.view[i]); }
    if (!W.pool) {
        VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf }; vkCreateCommandPool(g_dev, &pci, NULL, &W.pool);
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = W.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 3 }; vkAllocateCommandBuffers(g_dev, &cai, W.cb);
        for (int i = 0; i < 3; i++) { VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; vkCreateSemaphore(g_dev, &sci, NULL, &W.acq[i]); vkCreateSemaphore(g_dev, &sci, NULL, &W.done[i]);
            VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT }; vkCreateFence(g_dev, &fci, NULL, &W.fence[i]); }
    }
    W.win = win; W.srgb = srgb;
    VLOG("window: swapchain %ux%u, %u images, format %d, %s", W.phys.width, W.phys.height, W.n, W.fmt, pm == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : pm == VK_PRESENT_MODE_FIFO_KHR ? "fifo" : "immediate");
    return 1;
}
// ---------- Touch overlay (nvn_ovl.h): drawn into the swapchain image after the game picture, one quad per control ----------
#include "overlay/ovl_spv.h"
#include "overlay/ovl_font.h"
__attribute__((weak)) int front_overlay(dm_ovl_item *it, int max, int width, int height) { (void)it; (void)max; (void)width; (void)height; return 0; }
static struct { VkFormat fmt; VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds[3]; VkSampler samp;
                VkImage font; VkImageView fview; vmalloc_t fa, sa; VkBuffer stage; int font_ready, failed; } O;
typedef struct { float rect[4], color[4], shape[4], glyph[4], target[2], pad[2], rot[4]; } ovl_pc;   // ovl.vert / ovl.frag's push constants (rot at 80: std430 vec4 alignment)
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
        // binding 0 the label atlas, binding 1 the game's picture (present_window sets it per present slot; the atlas stands in until then)
        VkDescriptorSetLayoutBinding b[2] = { { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL }, { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL } };
        VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b }; vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &O.dsl);
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 };
        VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 3, .poolSizeCount = 1, .pPoolSizes = &ps }; vkCreateDescriptorPool(g_dev, &dpi, NULL, &O.dp);
        VkDescriptorSetLayout ls[3] = { O.dsl, O.dsl, O.dsl };
        VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = O.dp, .descriptorSetCount = 3, .pSetLayouts = ls };
        if (vkAllocateDescriptorSets(g_dev, &dai, O.ds)) { O.failed = 1; RED_ONCE(72, "overlay: cannot allocate its descriptor sets"); return 0; }
        VkDescriptorImageInfo ii = { O.samp, O.fview, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[6];
        for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = O.ds[i / 2], .dstBinding = (uint32_t)(i % 2), .descriptorCount = 1,
                                                                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii };
        vkUpdateDescriptorSets(g_dev, 6, w, 0, NULL);
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
    VkResult r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &O.pipe);
    vkDestroyShaderModule(g_dev, vm, NULL); vkDestroyShaderModule(g_dev, fm, NULL);
    if (r) { O.failed = 1; RED_ONCE(71, "overlay: pipeline failed %d", r); return 0; }
    O.fmt = fmt; VLOG("touch overlay ready (format %d)", fmt);
    return 1;
}
static void ovl_font_upload(VkCommandBuffer cb) {   // outside a pass
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
}
// The pipeline and set bound, then the quad first (the game's picture) if any and one quad per control, in the pass the caller opened over an image of size phys; ext is the
// picture as the player sees it, rot the pre-rotation (cos, sin)
static void ovl_items(VkCommandBuffer cb, VkDescriptorSet ds, VkExtent2D ext, VkExtent2D phys, const float rot[2], const dm_ovl_item *it, int n, const ovl_pc *first) {
    VkViewport v = { 0, 0, (float)phys.width, (float)phys.height, 0, 1 }; VkRect2D sc = { {0, 0}, phys };
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, O.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, O.pl, 0, 1, &ds, 0, NULL);
    if (first) { vkCmdPushConstants(cb, O.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof *first, first); vkCmdDraw(cb, 6, 1, 0, 0); }
    for (int i = 0; i < n; i++) {
        const dm_ovl_item *q = &it[i];
        ovl_pc pc = { { q->x, q->y, q->w, q->h }, { q->r, q->g, q->b, q->a }, { (float)q->shape, q->ring, q->radius, q->h > 0 ? q->w / q->h : 1 },
                      { (float)q->glyph, q->gscale > 0 ? q->gscale : 0.5f, q->galpha, q->gdark }, { (float)ext.width, (float)ext.height }, { 0, 0 }, { rot[0], rot[1], 0, 0 } };
        vkCmdPushConstants(cb, O.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, 6, 1, 0, 0);
    }
}
// Records the overlay into cb over image img (view, size ext, format fmt), which is in TRANSFER_DST layout; leaves it in layout fin. Returns 0 (nothing recorded) when there is nothing to draw
static int ovl_draw(VkCommandBuffer cb, VkDescriptorSet ds, VkImage img, VkImageView view, VkExtent2D ext, VkFormat fmt, VkImageLayout fin) {
    dm_ovl_item it[128]; int n = front_overlay(it, 128, (int)ext.width, (int)ext.height);
    if (n <= 0 || !view || !ovl_setup(fmt)) return 0;
    ovl_font_upload(cb);
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, ext }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    vkCmdBeginRendering(cb, &ri);
    static const float ident[2] = { 1, 0 };
    ovl_items(cb, ds, ext, ext, ident, it, n, NULL);
    vkCmdEndRendering(cb);
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
    if (!ovl_draw(cb, O.ds[0], im, vw, ext, fmt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)) {
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
// The frame into swapchain image ii in one pass (on a tiler the image is written once, never read): cleared black, the game's picture drawn as a quad fitted to the window (the
// letterbox), the touch overlay over it, all turned by the swapchain's pre-rotation. It used to be a transfer clear, a blit and a pass loading the image back for the overlay: three
// full-screen writes and a full-screen read a frame. 0 (nothing recorded) when the picture cannot be sampled or the overlay pipeline is not there; DM_VK_PRESENT_PASS=0 also
static int present_pass(VkCommandBuffer cb, vtex *x, const int crop[4], uint32_t ii, uint32_t sl) {
    static int off = -1; if (off < 0) { const char *e = getenv("DM_VK_PRESENT_PASS"); off = e && !strcmp(e, "0"); }
    if (off || !(x->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !W.view[ii] || !ovl_setup(W.fmt)) return 0;
    VkImageView fv = view_get(x, VK_IMAGE_VIEW_TYPE_2D, x->fmt, 0, 1, 0, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT); if (!fv) return 0;
    { static int said; if (!said++) VLOG("present: one pass (picture %ux%u format %d drawn into the window image, pre-rotation %.0f, %.0f)", x->w, x->h, x->fmt, W.rot[0], W.rot[1]); }
    VkDescriptorImageInfo di = { O.samp, fv, VK_IMAGE_LAYOUT_GENERAL };   // this slot's set: its last use finished (W.fence[sl])
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = O.ds[sl], .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &di };
    vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL);
    dm_ovl_item it[128]; int n = front_overlay(it, 128, (int)W.ext.width, (int)W.ext.height); if (n < 0) n = 0;
    ovl_font_upload(cb);
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = W.view[ii], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = { .color = {{ 0, 0, 0, 1 }} } };
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { {0, 0}, W.phys }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
    vkCmdBeginRendering(cb, &ri);
    int32_t cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int32_t)x->w, ch = crop && crop[3] ? crop[3] : (int32_t)x->h;
    double sc = fmin((double)W.ext.width / cw, (double)W.ext.height / ch); float dw = (float)(int32_t)(cw * sc), dh = (float)(int32_t)(ch * sc), dx = (float)(((int32_t)W.ext.width - (int32_t)dw) / 2), dy = (float)(((int32_t)W.ext.height - (int32_t)dh) / 2);
    float u0 = (float)cx / x->w, u1 = (float)(cx + cw) / x->w, v0 = (float)cy / x->h, v1 = (float)(cy + ch) / x->h;
    if (atomic_load(&g_origin_ll)) { float t = v0; v0 = v1; v1 = t; }   // a GL-origin picture: its first row is the bottom
    static int fxaa = -1, sharp = -1; if (fxaa < 0) { const char *e = getenv("DM_VK_FXAA"); fxaa = e && atoi(e) == 1; if (fxaa) VLOG("present: FXAA on the game's picture"); }
    if (sharp < 0) { const char *e = getenv("DM_VK_SHARPEN"); sharp = e && atoi(e) > 0 && !fxaa; if (sharp) VLOG("present: the game's picture sharpened (CAS)"); }
    ovl_pc pic = { { dx, dy, dw, dh }, { 0, 0, 0, 1 }, { 4, (float)fxaa, (float)sharp, 1 }, { u0, v0, u1, v1 }, { (float)W.ext.width, (float)W.ext.height }, { 0, 0 }, { W.rot[0], W.rot[1], 0, 0 } };
    ovl_items(cb, O.ds[sl], W.ext, W.phys, W.rot, it, n, &pic);
    vkCmdEndRendering(cb);
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = 0; ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    return 1;
}
static void present_window_impl(vtex *x, const int crop[4]);
static void present_window(vtex *x, const int crop[4]) { uint64_t pf_t0 = prof_t(); present_window_impl(x, crop); prof_add(PF_PRESENT, pf_t0); }
static void present_window_impl(vtex *x, const int crop[4]) {
    void *win = front_metal_layer();
    if (!win) { if (W.sc) swap_destroy(); return; }
    if (!g_has_swapchain || !x || !x->img || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || x->samples > 1) return;
    int srgb = x->fmt == VK_FORMAT_R8G8B8A8_SRGB || x->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    if (!swap_ensure(win, srgb)) return;
    uint32_t sl = W.slot++ % 3;
    vkWaitForFences(g_dev, 1, &W.fence[sl], VK_TRUE, UINT64_MAX);
    uint32_t ii; VkResult r = vkAcquireNextImageKHR(g_dev, W.sc, UINT64_MAX, W.acq[sl], VK_NULL_HANDLE, &ii);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { swap_destroy(); return; }
    if (r && r != VK_SUBOPTIMAL_KHR) return;
    vkResetFences(g_dev, 1, &W.fence[sl]);
    VkCommandBuffer cb = W.cb[sl]; vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT }; vkBeginCommandBuffer(cb, &bi);
    barrier_all(cb);
    if (!present_pass(cb, x, crop, ii, sl)) {   // (the picture cannot be sampled: cleared, blitted and overlaid by transfer as before; not turned)
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = W.img[ii], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cb, W.img[ii], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    int32_t cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int32_t)x->w, ch = crop && crop[3] ? crop[3] : (int32_t)x->h;
    double sc = fmin((double)W.phys.width / cw, (double)W.phys.height / ch); int32_t dw = (int32_t)(cw * sc), dh = (int32_t)(ch * sc), dx = ((int32_t)W.phys.width - dw) / 2, dy = ((int32_t)W.phys.height - dh) / 2;
    int flip = atomic_load(&g_origin_ll);
    VkImageBlit bl = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{cx, flip ? cy + ch : cy, 0}, {cx + cw, flip ? cy : cy + ch, 1}}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, {{dx, dy, 0}, {dx + dw, dy + dh, 1}} };
    vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, W.img[ii], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    if (!ovl_draw(cb, O.ds[sl], W.img[ii], W.view[ii], W.phys, W.fmt, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)) {
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = 0; ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
    }
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &W.acq[sl], .pWaitDstStageMask = &ws, .commandBufferCount = 1, .pCommandBuffers = &cb,
                        .signalSemaphoreCount = 1, .pSignalSemaphores = &W.done[sl] };
    VkPresentInfoKHR pi = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &W.done[sl], .swapchainCount = 1, .pSwapchains = &W.sc, .pImageIndices = &ii };
    pthread_mutex_lock(&g_qmu);
    r = vkQueueSubmit(g_q, 1, &si, W.fence[sl]);
    if (!r) r = vkQueuePresentKHR(g_q, &pi);
    pthread_mutex_unlock(&g_qmu);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) swap_destroy();
}
// Once a minute: where the renderer's memory is, so growth can be told apart (caches that level off: shaders, pipelines; anything else that keeps rising is a leak)
static void mem_report(void) {
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
    VLOG("[mem] device blocks %d: %.0f MB (%.0f MB in use); textures %ld live, %.0f MB; pools %d, %.0f MB; arenas %.0f MB; descriptor pools %d; deferred %d; "
         "shaders %u, pipelines %u, textures by serial %u, aliases %u, ghosts %u; host heap %.0f MB",
         nb, dev / 1048576, devused / 1048576, atomic_load(&g_live_tex), atomic_load(&g_live_tex_bytes) / 1048576.0, npool, pool / 1048576, arena / 1048576, ndp, ndf,
         g_shcache.n, g_pso.n, g_by_serial.n, g_alias.n, g_ghost.n, heap / 1048576.0);
}
// The front end's frame report (Android: performance hints, android_main.c): the presenting thread's work since its last present against the frame's period
__attribute__((weak)) void front_frame_work(uint64_t busy_ns, uint64_t target_ns) { (void)busy_ns; (void)target_ns; }   // (no front end report: nothing)
static uint64_t vk_present(const tex_t *t, double period_s, const int crop[4]) {
    uint64_t n = atomic_fetch_add(&g_present_n, 1) + 1;
    static uint64_t g_present_exit;   // when the last present returned (after its pacing sleep): from there to this call is the frame's work
    if (g_present_exit && period_s > 0) front_frame_work(mono_ns() - g_present_exit, (uint64_t)(period_s * 1e9));
    static long shots[256]; static int nshot = -1; static const char *shot_path;
    if (nshot < 0) { nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 256; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); } }
    {   // on request (testing, the Android front end sets DM_SHOT_REQ = <files>/shot): a file <that>.req makes the next shown frame <that>.png (checked every 15 presents)
        static const char *rq; static int init; if (!init++) rq = getenv("DM_SHOT_REQ");
        if (rq && n % 15 == 0) { char rp[1024], pp[1024]; snprintf(rp, sizeof rp, "%s.req", rq); snprintf(pp, sizeof pp, "%s.png", rq);
            if (!access(rp, F_OK)) { unlink(rp); vtex *x = tex_res(t); if (x && x->img) shot(x, crop, pp); } } }
    for (int i = 0; i < nshot; i++) if ((long)n == shots[i]) { vtex *x = tex_res(t); char path[1024]; snprintf(path, sizeof path, shot_path, (int)n); if (x && x->img) shot(x, crop, path); }
    { vtex *px = tex_res(t); if (px) { pthread_mutex_lock(&g_skmu); px->presented = 1; pthread_mutex_unlock(&g_skmu); sk_read(px); } }   // shown: never skips its stores
    if (t) { vtex *sx = tex_res(t); fx_shown(t->serial, sx ? sx->w : 0, sx ? sx->h : 0); }   // the scene FXAA looks for the composite into it
    {   // the pass log: DM_VK_PASSLOG=<present>, or the next present after a file <DM_SHOT_REQ>.passlog appears (checked every 15 presents)
        static int init; static const char *rq; if (!init++) { rq = getenv("DM_SHOT_REQ"); const char *e = getenv("DM_VK_PASSLOG"); if (e) atomic_store(&g_passlog, atol(e)); }
        if (rq && n % 15 == 0) { char rp[1024]; snprintf(rp, sizeof rp, "%s.passlog", rq); if (!access(rp, F_OK)) { unlink(rp); atomic_store(&g_passlog, (long)atomic_load(&g_fx_frames) + 1); VLOG("[passlog] next frame"); } } }
    if (front_metal_layer()) present_window(tex_res(t), crop);
#ifndef __ANDROID__
    { static long of = -1; static const char *op; if (of < 0) { const char *e = getenv("DM_OVL_SHOT"), *c = e ? strchr(e, ':') : NULL; of = c ? atol(e) : 0; op = c ? c + 1 : NULL; }
      if (of && (long)n == of && op) { vtex *x = tex_res(t); if (x && x->img) ovl_shot(x, crop, op); } }
#endif
    uint64_t now = mono_ns(), due = g_last_present + (uint64_t)(period_s * 1e9);
    if (period_s > 0 && due > now) { struct timespec ts = { (time_t)((due - now) / 1000000000ull), (long)((due - now) % 1000000000ull) }; nanosleep(&ts, NULL); now = due; }
    g_last_present = now; g_present_exit = mono_ns();
    if (n % 900 == 0 && atomic_load(&g_pso_new) >= 10) { static atomic_int busy; if (!atomic_exchange(&busy, 1)) {   // new pipelines: the cache file is rewritten in the background
        pthread_t t; void *saver(void *a); if (pthread_create(&t, NULL, saver, &busy)) atomic_store(&busy, 0); else pthread_detach(t); } }
    if (n % 1800 == 0) { VLOG("pipelines built during play so far: %d", atomic_load(&g_pso_ingame)); mem_report(); }
    prof_report(n);
    if (n % 300 == 0) VLOG("barriers before draw passes: %llu full, %llu fragment work only, %llu left out; submissions split for the GPU %llu times", (unsigned long long)g_bar_full, (unsigned long long)g_bar_narrow, (unsigned long long)g_bar_none, (unsigned long long)g_splits);
    if (n % 300 == 0) VLOG("passes ended by: dispatch %llu, buffer copy %llu, uniform update %llu, fill %llu, buffer-texture copy %llu, texture copy %llu, texture clear %llu, new targets %llu, submit %llu, a capture breaking a pass %llu, captures %llu, other %llu",
        (unsigned long long)g_pe_n[1], (unsigned long long)g_pe_n[2], (unsigned long long)g_pe_n[3], (unsigned long long)g_pe_n[4], (unsigned long long)g_pe_n[5], (unsigned long long)g_pe_n[6], (unsigned long long)g_pe_n[7], (unsigned long long)g_pe_n[8], (unsigned long long)g_pe_n[9], (unsigned long long)g_pe_n[10], (unsigned long long)g_pe_n[11], (unsigned long long)g_pe_n[0]);
    if (n == 1 || n % 300 == 0) VLOG("present %llu: %llu render target sets (%llu kept the open pass, %llu passes begun), %llu draws so far; uniform updates kept in the pass %llu (%llu passes ended for them); stores skipped %llu (%llu read after all), depth / stencil not stored %llu (%llu written after all); textures uploaded %llu (re-uploaded %llu, %.1f MB)", (unsigned long long)n, (unsigned long long)g_stat_passes, (unsigned long long)g_stat_rt_same, (unsigned long long)g_stat_begins,
                                     (unsigned long long)g_stat_draws, (unsigned long long)g_ubd_deferred, (unsigned long long)g_ubd_conflicts, (unsigned long long)g_sk_skipped, (unsigned long long)g_sk_wrong, (unsigned long long)g_ro_passes, (unsigned long long)g_ro_wrong, (unsigned long long)g_pkg_uploads, (unsigned long long)g_pkg_reuploads, g_up_bytes / 1048576.0);
    return n;
}
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
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = g_pkg_uploads; *r = g_pkg_reuploads; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return g_up_bytes; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
void mtl_res_stats(char *out, size_t n) { if (n) snprintf(out, n, "Vulkan: no residency set"); }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts) { *deferred = g_ubd_deferred; *conflicts = g_ubd_conflicts; }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = g_bar_narrow + g_bar_none; *full = g_bar_full; }
uint64_t mtl_blit_splits(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }

// ================= The render thread (DM_VK_RENDER_THREAD=1: on; off by default) =================
// The replay of a submission (Vulkan commands for the game's NVN commands: 8–15 ms a frame in Legends: Arceus battles) used to run on the game's thread inside
// nvnQueueSubmitCommands, half of its frame. Submissions, presents and texture releases are now queued in order to a thread of their own, as a GPU takes command buffers: the game
// goes on with the next frame meanwhile. What the game can see stays as on the console: the fences it gets are numbers of queued submissions (a sync waits until the thread has
// replayed that submission, then until the GPU has run it), syncs fenced inside command buffers are fenced when queued, calls that read or free what queued work uses wait for the
// queue first (pool removal, texel reads, QueueFinish), and a present waits for the one before it, so the game runs at most one frame ahead (the pacing stays the thread's)
enum { RQ = 64, VRING = 4096 };
typedef struct { int kind; int n; uint64_t *h; uint64_t v; tex_t t; double period; int crop[4]; int has_crop; } rjob;   // kind 0 submit (v), 1 present (v: its ticket), 2 texture release
static rjob g_rq[RQ]; static uint64_t g_rq_in, g_rq_out, g_rq_done;   // queued, taken, finished
static pthread_mutex_t g_rqmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_rq_cv = PTHREAD_COND_INITIALIZER, g_rq_dcv = PTHREAD_COND_INITIALIZER;
static uint64_t g_v_in, g_v_done, g_p_in, g_p_done, g_vreal[VRING];   // submissions queued / replayed (their real fences), presents queued / done
static uint64_t g_rt_drains, g_rt_drain_ns, g_pfence[64];   // g_pfence: the submissions queued when each present was (by ticket)
static void *rt_main(void *a);
static int g_rt_on;
static void rt_start(void) {   // once (the game's threads can ask at the same time)
    const char *e = getenv("DM_VK_RENDER_THREAD"); g_rt_on = e && atoi(e) > 0;   // off unless asked: the game waits for its own newest submission every frame (no gain), and a crash in the game's code on entering the Fieldlands followed it
    if (g_rt_on) { pthread_t t; if (pthread_create(&t, NULL, rt_main, NULL)) g_rt_on = 0; else pthread_detach(t); }
    VLOG("render thread: %s", g_rt_on ? "the game's submissions and presents are replayed on a thread of their own" : "off (replayed on the game's thread)");
}
static int rt_on(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, rt_start); return g_rt_on; }
static void rt_push(rjob j) {   // caller holds g_rqmu
    while (g_rq_in - g_rq_done >= RQ) pthread_cond_wait(&g_rq_dcv, &g_rqmu);
    g_rq[g_rq_in % RQ] = j; g_rq_in++; pthread_cond_broadcast(&g_rq_cv);
}
static void rt_drain(const char *why) {   // everything queued so far done
    if (!rt_on()) return;
    uint64_t t0 = mono_ns(); pthread_mutex_lock(&g_rqmu); uint64_t want = g_rq_in; int waited = 0;
    while (g_rq_done < want) { waited = 1; pthread_cond_wait(&g_rq_dcv, &g_rqmu); }
    if (waited) { g_rt_drains++; g_rt_drain_ns += mono_ns() - t0; }
    uint64_t n = g_rt_drains; pthread_mutex_unlock(&g_rqmu);
    if (waited && (n <= 5 || n % 500 == 0)) VLOG("render thread: the game waited for the queue (%s), %llu times so far, %.1f ms in all", why, (unsigned long long)n, g_rt_drain_ns / 1e6);
}
// Syncs fenced in the queued command buffers (nvnCommandBufferFenceSync): fenced now with the submission's number
static void rt_scan(const uint8_t *p, uint64_t v, int depth) {
    for (int guard = 0; guard < (1 << 26) && p; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        if (h.op == OP_END) return;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_CALL) { const uint8_t *t; memcpy(&t, a, 8); if (depth < 16) rt_scan(t, v, depth + 1); }
        else if (h.op == OP_FENCE_SYNC) { sync_t *sy; memcpy(&sy, a, 8); sy->fence = v; sy->fenced = 1; }
        if (!h.size) return;
        p += h.size;
    }
}
static void rt_replay(const uint64_t *h, int n) {
    mtl_sub_begin(); mtl_sub_index(h, n);
    for (int i = 0; i < n; i++) { mtl_exec_rest(h + i + 1, n - i - 1); mtl_exec((const uint8_t *)(uintptr_t)h[i]); }
    mtl_exec_rest(NULL, 0);
}
static void *rt_main(void *a) {
    (void)a; prctl(PR_SET_NAME, "dm-render", 0, 0, 0);
    for (;;) {
        pthread_mutex_lock(&g_rqmu);
        while (g_rq_out == g_rq_in) pthread_cond_wait(&g_rq_cv, &g_rqmu);
        rjob j = g_rq[g_rq_out % RQ]; g_rq_out++;
        pthread_mutex_unlock(&g_rqmu);
        uint64_t real = 0;
        if (j.kind == 0) { rt_replay(j.h, j.n); real = mtl_sub_commit(); free(j.h); }
        else if (j.kind == 1) vk_present(&j.t, j.period, j.has_crop ? j.crop : NULL);
        else if (j.kind == 2) vk_tex_release(&j.t);
        pthread_mutex_lock(&g_rqmu);
        if (j.kind == 0) { g_vreal[j.v % VRING] = real; g_v_done = j.v; }
        if (j.kind == 1) g_p_done = j.v;
        g_rq_done++; pthread_cond_broadcast(&g_rq_dcv);
        pthread_mutex_unlock(&g_rqmu);
    }
    return NULL;
}
// nvnQueueSubmitCommands (hle_nvn.c, under its submission lock)
void mtl_submit(const uint64_t *h, int n) {
    if (!rt_on()) { mtl_sub_begin(); mtl_sub_index(h, n); for (int i = 0; i < n; i++) { mtl_exec_rest(h + i + 1, n - i - 1); mtl_exec((const uint8_t *)(uintptr_t)h[i]); } mtl_exec_rest(NULL, 0); mtl_sub_commit(); return; }
    rjob j = { .kind = 0, .n = n }; j.h = malloc(sizeof *j.h * (size_t)(n ? n : 1)); memcpy(j.h, h, sizeof *j.h * (size_t)n);
    pthread_mutex_lock(&g_rqmu);
    j.v = ++g_v_in;
    for (int i = 0; i < n; i++) rt_scan((const uint8_t *)(uintptr_t)h[i], j.v, 0);
    rt_push(j);
    pthread_mutex_unlock(&g_rqmu);
}
uint64_t mtl_last_fence(void) { if (!rt_on()) return g_fence; pthread_mutex_lock(&g_rqmu); uint64_t v = g_v_in; pthread_mutex_unlock(&g_rqmu); return v; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (!rt_on()) return vk_wait(fence, timeout_ns);
    if (!fence) return 1;
    uint64_t end = timeout_ns == UINT64_MAX ? UINT64_MAX : mono_ns() + timeout_ns, w0 = mono_ns();
    pthread_mutex_lock(&g_rqmu);
    static uint64_t nw, nwait, wns, lag[4]; int must = g_v_done < fence; uint64_t behind = g_v_in - fence; nw++;
    while (g_v_done < fence) {
        if (end == UINT64_MAX) pthread_cond_wait(&g_rq_dcv, &g_rqmu);
        else { uint64_t now = mono_ns(); if (now >= end) break; struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); uint64_t ns = ts.tv_nsec + (end - now);
               ts.tv_sec += ns / 1000000000ull; ts.tv_nsec = ns % 1000000000ull; pthread_cond_timedwait(&g_rq_dcv, &g_rqmu, &ts); }
    }
    if (must) { nwait++; wns += mono_ns() - w0; lag[behind < 3 ? behind : 3]++; }
    if (nw % 600 == 0) VLOG("render thread: the game waited on %llu of %llu fences for the replay, %.1f ms in all (on the newest submission %llu, one before %llu, two %llu, older %llu)",
        (unsigned long long)nwait, (unsigned long long)nw, wns / 1e6, (unsigned long long)lag[0], (unsigned long long)lag[1], (unsigned long long)lag[2], (unsigned long long)lag[3]);
    int replayed = g_v_done >= fence; uint64_t real = replayed ? (g_v_done - fence < VRING ? g_vreal[fence % VRING] : g_vreal[g_v_done % VRING]) : 0;
    pthread_mutex_unlock(&g_rqmu);
    if (!replayed) return 0;
    if (end == UINT64_MAX) return vk_wait(real, UINT64_MAX);
    uint64_t now = mono_ns(); return vk_wait(real, now >= end ? 0 : end - now);
}
void mtl_finish(void) { rt_drain("finish"); vk_finish(); }
void mtl_drain(void) { mtl_finish(); }
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    if (!rt_on()) return vk_present(t, period_s, crop);
    rjob j = { .kind = 1, .period = period_s }; if (t) j.t = *t; if (crop) { memcpy(j.crop, crop, sizeof j.crop); j.has_crop = 1; }
    pthread_mutex_lock(&g_rqmu);
    j.v = ++g_p_in; g_pfence[j.v % 64] = g_v_in;
    while (g_p_done + 1 < j.v) pthread_cond_wait(&g_rq_dcv, &g_rqmu);   // the one before shown: at most one frame ahead
    rt_push(j);
    pthread_mutex_unlock(&g_rqmu);
    return j.v;
}
// The fence of a window texture's acquire: its last present's (what was queued before it), not everything queued since; with the render thread the game would otherwise wait
// for the whole frame it just queued at every acquire
uint64_t mtl_present_fence(uint64_t ticket) {
    if (!rt_on() || !ticket) return mtl_last_fence();
    pthread_mutex_lock(&g_rqmu); uint64_t f = g_p_in - ticket < 64 ? g_pfence[ticket % 64] : g_v_in; pthread_mutex_unlock(&g_rqmu); return f;
}
void mtl_present_wait(uint64_t ticket) {
    if (!rt_on() || !ticket) return;
    pthread_mutex_lock(&g_rqmu); while (g_p_done < ticket) pthread_cond_wait(&g_rq_dcv, &g_rqmu); pthread_mutex_unlock(&g_rqmu);
}
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    if (!rt_on()) { vk_tex_release(t); return; }
    rjob j = { .kind = 2 }; j.t = *t;
    pthread_mutex_lock(&g_rqmu); rt_push(j); pthread_mutex_unlock(&g_rqmu);
}
void mtl_pool_del(const void *mem) { rt_drain("memory pool removed"); vk_pool_del(mem); }
void mtl_alias_drop_pool(const void *pool) { rt_drain("memory pool aliases dropped"); vk_alias_drop_pool(pool); }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    rt_drain("texels read"); vk_read_texels(t, level, layer0, r, dst, row, img);
}
