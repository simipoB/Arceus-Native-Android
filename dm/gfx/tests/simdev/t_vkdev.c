// t_vkdev: the engine's own Vulkan device set-up (nvn_vk.c mtl_init, compiled here as it is, in this translation unit so its decisions can be read) on whatever
//   device the loader gives it: on a PC, Mesa's lavapipe, alone or behind the Khronos profiles layer pretending to be a phone's GPU (simdev/run_simdev.sh). It checks
//   what the engine decided against what that device reports (read again independently, through the same instance), and against what is known for the GPU named
//   by SIMDEV_PROFILE (Arm's release from driverVersion, no dual-source blending on r44p1, the Vulkan 1.0 routes, ...). Then a little real work through the engine:
//   samplers with every optional feature asked for, a memory pool mapped for the GPU, a fill submitted and waited for, a depth / stencil copy drawn (zcopy_draw),
//   integer textures bound under a float sampler (tex_view_for), the draw-state shadow's reset rules and descriptor pushes, the draw memo (shadow_work), a load through
//   the device-memory allocator by lifetime (MEMLIFE), shader modules released and made again (SHMOD).
//   Build: simdev/build_vkdev.sh. Run: SIMDEV_PROFILE=<name> ./t_vkdev (exit 1 on a failed check; "FAIL" lines name it).
//   DM_VK_FRAME_HOLD=2 (run_simdev.sh's second t_vkdev run): every submission held and decided, as with Faster rendering on Adreno: its passes in secondaries.
//   SIMDEV_DC_FC=1 / 2: a float_controls_crashed marker from an earlier start, and DM_VK_FLOAT_CONTROLS=1 from the driver check (1: named in DM_DRIVER_CHECK_KEYS, as
//   android_main.c names driver-check.env's keys) or from env.txt (2): the check's value is this driver's default and the marker applies; the player's skips it.
// Nothing here changes the engine: the few symbols it takes from the rest of the host (page tracking, the game's clock, the overlay) are stubbed below, unused by the set-up.
#include "../../nvn/nvn_vk.c"
#include <execinfo.h>
#include <signal.h>

static int fails, checks;
static int g_dcfc;   // SIMDEV_DC_FC (above)
// The table while the device is being made: nvn_vk.c calls this right after vkCreateDevice (android_main.c's boot_state hook). The rules that read the GPU's name,
//   version, vendor and the game's profile answer already; those that read the driver ID or Arm's release (vk_quirks.h quirk_late) answer -1 until mtl_init has them
static int g_up_early = -1, g_up_late = -1;
void dm_boot_device_up(void) {
    g_up_early = g_up_late = 1;
    for (int q = 0; q < Q_N; q++) { int v = nvn_vk_quirk_default(q); if (quirk_late(q)) g_up_late &= v == -1; else g_up_early &= v == quirk_default(q, &g_gpuid); }
}
static const char *volatile g_step = "start";   // (the step a crash happened in)
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define DEC(k, fmt, ...) printf("DECISION %-22s " fmt "\n", k, __VA_ARGS__)

// ---- the rest of the host, stubbed (none of it is reached by device set-up, samplers, pools or a fill) ----
void nvn_dirty_add_range(const void *mem, size_t size) { (void)mem; (void)size; }
void nvn_dirty_remove_range(const void *mem) { (void)mem; }
int nvn_dirty_tracked(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
void nvn_dirty_protect(const void *addr, size_t len) { (void)addr; (void)len; }
int nvn_dirty_off(void) { return 0; }   // (pools backed late, at their first use as a buffer: the route the phones take)
static uintptr_t g_fz_a; static size_t g_fz_n;   // (no tracked pages here: the thaw leaves the frozen range writable, as nvn_dirty.c does for untracked pages)
void nvn_dirty_freeze(const void *addr, size_t len, int wp) { (void)wp; size_t pg = (size_t)getpagesize(); g_fz_a = (uintptr_t)addr & ~(pg - 1); g_fz_n = (((uintptr_t)addr + len + pg - 1) & ~(pg - 1)) - g_fz_a; }
void nvn_dirty_thaw(void) { if (g_fz_n) mprotect((void *)g_fz_a, g_fz_n, PROT_READ | PROT_WRITE); g_fz_n = 0; }
void nvn_dirty_release(const void *addr, size_t len) { (void)addr; (void)len; }
uint64_t nvn_dirty_now(void) { return 1; }
const void *nvn_dirty_span(const void *addr, size_t len) { (void)addr; (void)len; return NULL; }
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp) { (void)span; (void)addr; (void)len; (void)stamp; return 1; }
void nvn_dirty_runs(const void *addr, size_t len, uint64_t stamp, void (*cb)(uintptr_t a, size_t n, void *u), void *u) { (void)stamp; cb((uintptr_t)addr, len, u); }
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) { (void)fn; }
void dm_guest_write_prepare(void *addr, size_t len) { (void)addr; (void)len; }
void *front_metal_layer(void) { return NULL; }
_Atomic uint64_t g_wait_sync_ns, g_wait_sync_n;
uint64_t guest_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
uint64_t nvn_new_serial(void) { static _Atomic uint64_t s = 1; return atomic_fetch_add(&s, 1); }
int prog_key(const prog_t *p, unsigned stage, uint8_t key[8]) { (void)p; (void)stage; memset(key, 0, 8); return 0; }
void touch_ui_demo(int c, float dx, float dy) { (void)c; (void)dx; (void)dy; }
int dm_gpu_ts614(void) __attribute__((weak)); int dm_gpu_ts614(void) { return 0; }   // (newer engines: hle_nvn.c)

// ---- what the engine asks of the loader, watched: vkGetInstanceProcAddr wrapped, so the instance and device it makes are seen as they are made ----
static PFN_vkGetInstanceProcAddr real_gipa;
static PFN_vkCreateInstance real_ci; static PFN_vkCreateDevice real_cd;
static PFN_vkGetPhysicalDeviceFeatures2 real_f2; static PFN_vkGetPhysicalDeviceProperties2 real_p2; static PFN_vkGetPhysicalDeviceExternalBufferProperties real_xb;
static uint32_t w_inst_api; static int w_inst_p2ext;    // the instance as made: its apiVersion, VK_KHR_get_physical_device_properties2 enabled
static int w_f2_calls, w_f2_bad;                         // Features2 / Properties2 / external buffer queries, and those made where the instance has no right to them
static char w_dext[64][VK_MAX_EXTENSION_NAME_SIZE]; static uint32_t w_ndext; static int w_dev_made; static _Atomic int w_dev_count;
static VkResult w_dev_result = VK_RESULT_MAX_ENUM;
static int w_struct_bad;
// The GPU's identity. The profiles layer simulates limits, features, formats and extensions but on purpose not who the device is (vendorID, deviceID, driverVersion,
//   deviceName, deviceType and the driver properties are "not modifiable" there), and the engine's per-GPU choices key on exactly that. run_simdev.sh passes the
//   profile's identity (SIMDEV_VENDOR, SIMDEV_DEVICE_ID, SIMDEV_DRIVER_VERSION, SIMDEV_DEVICE_NAME, SIMDEV_DEVICE_TYPE, SIMDEV_DRIVER_ID / NAME / INFO) and the
//   answers the engine gets are given it here, above every layer
static PFN_vkGetPhysicalDeviceProperties real_p1;
static void ident_apply(VkPhysicalDeviceProperties *p) {
    const char *e;
    if ((e = getenv("SIMDEV_VENDOR")) && *e) p->vendorID = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("SIMDEV_DEVICE_ID")) && *e) p->deviceID = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("SIMDEV_DRIVER_VERSION")) && *e) p->driverVersion = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("SIMDEV_DEVICE_TYPE")) && *e) p->deviceType = (VkPhysicalDeviceType)atoi(e);
    if ((e = getenv("SIMDEV_DEVICE_NAME")) && *e) snprintf(p->deviceName, sizeof p->deviceName, "%s", e);
}
static void ident_apply2(VkPhysicalDeviceProperties2 *p) {
    ident_apply(&p->properties);
    for (VkBaseOutStructure *s = p->pNext; s; s = s->pNext) {
        VkDriverId *id = NULL; char *name = NULL, *info = NULL;
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) { VkPhysicalDeviceDriverProperties *d = (void *)s; id = &d->driverID; name = d->driverName; info = d->driverInfo; }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES) { VkPhysicalDeviceVulkan12Properties *d = (void *)s; id = &d->driverID; name = d->driverName; info = d->driverInfo; }
        const char *e;
        if (id && (e = getenv("SIMDEV_DRIVER_ID")) && *e) *id = (VkDriverId)atoi(e);
        if (name && (e = getenv("SIMDEV_DRIVER_NAME"))) snprintf(name, VK_MAX_DRIVER_NAME_SIZE, "%s", e);
        if (info && (e = getenv("SIMDEV_DRIVER_INFO"))) snprintf(info, VK_MAX_DRIVER_INFO_SIZE, "%s", e);
    }
}
static VKAPI_ATTR void VKAPI_CALL w_p1(VkPhysicalDevice p, VkPhysicalDeviceProperties *o) { real_p1(p, o); ident_apply(o); }
static void f2_note(const char *what) { w_f2_calls++; if (w_inst_api < VK_API_VERSION_1_1 && !w_inst_p2ext) { w_f2_bad++; printf("FAIL %s:%d: %s called on a Vulkan 1.0 instance without VK_KHR_get_physical_device_properties2\n", __FILE__, __LINE__, what); fails++; } }
static VKAPI_ATTR void VKAPI_CALL w_f2(VkPhysicalDevice p, VkPhysicalDeviceFeatures2 *f) { f2_note("vkGetPhysicalDeviceFeatures2"); real_f2(p, f); }
static VKAPI_ATTR void VKAPI_CALL w_p2(VkPhysicalDevice p, VkPhysicalDeviceProperties2 *f) { f2_note("vkGetPhysicalDeviceProperties2"); real_p2(p, f); ident_apply2(f); }
static VKAPI_ATTR void VKAPI_CALL w_xb(VkPhysicalDevice p, const VkPhysicalDeviceExternalBufferInfo *i, VkExternalBufferProperties *o) {
    if (w_inst_api < VK_API_VERSION_1_1 && !w_inst_p2ext) { printf("FAIL %s:%d: vkGetPhysicalDeviceExternalBufferProperties on a Vulkan 1.0 instance\n", __FILE__, __LINE__); fails++; }
    real_xb(p, i, o); }
static VKAPI_ATTR VkResult VKAPI_CALL w_ci(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a, VkInstance *o) {
    w_inst_api = ci->pApplicationInfo ? ci->pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) if (!strcmp(ci->ppEnabledExtensionNames[i], "VK_KHR_get_physical_device_properties2")) w_inst_p2ext = 1;
    return real_ci(ci, a, o);
}
// Every structure in the device's create chain is allowed by the device's version or by an extension the engine enabled (a 1.2 / 1.3 block on an older device,
//   or an extension's structure without the extension, is invalid usage: drivers may refuse the device or ignore it)
static int w_has(const char *e) { for (uint32_t i = 0; i < w_ndext; i++) if (!strcmp(w_dext[i], e)) return 1; return 0; }
// lavapipe itself, opened directly (its ICD library, no loader, no layers): what it really has, to tell the simulation's gaps from the engine's choices
static struct { int tried; VkPhysicalDevice pd; PFN_vkGetPhysicalDeviceFeatures2 f2; VkExtensionProperties *ext; uint32_t next; } LVP;
static size_t feat_size(VkStructureType t) {
    switch ((int)t) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: return sizeof(VkPhysicalDeviceFeatures2);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: return sizeof(VkPhysicalDeviceVulkan11Features);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: return sizeof(VkPhysicalDeviceVulkan12Features);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: return sizeof(VkPhysicalDeviceVulkan13Features);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: return sizeof(VkPhysicalDeviceTimelineSemaphoreFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES: return sizeof(VkPhysicalDeviceDynamicRenderingFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES: return sizeof(VkPhysicalDeviceShaderDrawParametersFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT: return sizeof(VkPhysicalDeviceExtendedDynamicStateFeaturesEXT);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT: return sizeof(VkPhysicalDeviceExtendedDynamicState2FeaturesEXT);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT: return sizeof(VkPhysicalDeviceCustomBorderColorFeaturesEXT);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT: return sizeof(VkPhysicalDeviceDepthClipControlFeaturesEXT);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT: return sizeof(VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT: return sizeof(VkPhysicalDeviceImageCompressionControlFeaturesEXT);
    }
    return 0;
}
static const char *feat_name(VkStructureType t) { static char b[32]; snprintf(b, sizeof b, "structure %d", (int)t); return b; }
static void lvp_open(void) {
    if (LVP.tried++) return;
    const char *icd = getenv("VK_ICD_FILENAMES"); char lib[512] = "libvulkan_lvp.so";
    if (icd) { FILE *f = fopen(icd, "r"); char j[4096] = ""; if (f) { size_t n = fread(j, 1, sizeof j - 1, f); j[n] = 0; fclose(f); }
               char *p = strstr(j, "\"library_path\""); if (p && (p = strchr(p + 14, '"'))) { char *e = strchr(p + 1, '"'); if (e && e - p - 1 < (long)sizeof lib) { memcpy(lib, p + 1, (size_t)(e - p - 1)); lib[e - p - 1] = 0; } } }
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL); if (!h) { printf("NOTE lavapipe not opened directly (%s): no simulation-gap filtering\n", lib); return; }
    PFN_vkGetInstanceProcAddr gi = (PFN_vkGetInstanceProcAddr)dlsym(h, "vk_icdGetInstanceProcAddr"); if (!gi) return;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai }; VkInstance in;
    if (((PFN_vkCreateInstance)gi(NULL, "vkCreateInstance"))(&ici, NULL, &in)) return;
    uint32_t n = 1; ((PFN_vkEnumeratePhysicalDevices)gi(in, "vkEnumeratePhysicalDevices"))(in, &n, &LVP.pd); if (!n) { LVP.pd = VK_NULL_HANDLE; return; }
    LVP.f2 = (PFN_vkGetPhysicalDeviceFeatures2)gi(in, "vkGetPhysicalDeviceFeatures2");
    PFN_vkEnumerateDeviceExtensionProperties ee = (PFN_vkEnumerateDeviceExtensionProperties)gi(in, "vkEnumerateDeviceExtensionProperties");
    ee(LVP.pd, NULL, &LVP.next, NULL); LVP.ext = calloc(LVP.next + 1, sizeof *LVP.ext); ee(LVP.pd, NULL, &LVP.next, LVP.ext);
}
static int lvp_has_ext(const char *e) { lvp_open(); if (!LVP.ext) return 1; for (uint32_t i = 0; i < LVP.next; i++) if (!strcmp(LVP.ext[i].extensionName, e)) return 1; return 0; }
// member names of the feature structures the engine enables (vulkan_core.h order), for the SIMGAP lines
static const char *const FN10[] = { "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray", "independentBlend", "geometryShader", "tessellationShader", "sampleRateShading", "dualSrcBlend", "logicOp", "multiDrawIndirect", "drawIndirectFirstInstance", "depthClamp", "depthBiasClamp", "fillModeNonSolid", "depthBounds", "wideLines", "largePoints", "alphaToOne", "multiViewport", "samplerAnisotropy", "textureCompressionETC2", "textureCompressionASTC_LDR", "textureCompressionBC", "occlusionQueryPrecise", "pipelineStatisticsQuery", "vertexPipelineStoresAndAtomics", "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize", "shaderImageGatherExtended", "shaderStorageImageExtendedFormats", "shaderStorageImageMultisample", "shaderStorageImageReadWithoutFormat", "shaderStorageImageWriteWithoutFormat", "shaderUniformBufferArrayDynamicIndexing", "shaderSampledImageArrayDynamicIndexing", "shaderStorageBufferArrayDynamicIndexing", "shaderStorageImageArrayDynamicIndexing", "shaderClipDistance", "shaderCullDistance", "shaderFloat64", "shaderInt64", "shaderInt16", "shaderResourceResidency", "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer", "sparseResidencyImage2D", "sparseResidencyImage3D", "sparseResidency2Samples", "sparseResidency4Samples", "sparseResidency8Samples", "sparseResidency16Samples", "sparseResidencyAliased", "variableMultisampleRate", "inheritedQueries", };
static const char *const FN11[] = { "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess", "storagePushConstant16", "storageInputOutput16", "multiview", "multiviewGeometryShader", "multiviewTessellationShader", "variablePointersStorageBuffer", "variablePointers", "protectedMemory", "samplerYcbcrConversion", "shaderDrawParameters", };
static const char *const FN12[] = { "samplerMirrorClampToEdge", "drawIndirectCount", "storageBuffer8BitAccess", "uniformAndStorageBuffer8BitAccess", "storagePushConstant8", "shaderBufferInt64Atomics", "shaderSharedInt64Atomics", "shaderFloat16", "shaderInt8", "descriptorIndexing", "shaderInputAttachmentArrayDynamicIndexing", "shaderUniformTexelBufferArrayDynamicIndexing", "shaderStorageTexelBufferArrayDynamicIndexing", "shaderUniformBufferArrayNonUniformIndexing", "shaderSampledImageArrayNonUniformIndexing", "shaderStorageBufferArrayNonUniformIndexing", "shaderStorageImageArrayNonUniformIndexing", "shaderInputAttachmentArrayNonUniformIndexing", "shaderUniformTexelBufferArrayNonUniformIndexing", "shaderStorageTexelBufferArrayNonUniformIndexing", "descriptorBindingUniformBufferUpdateAfterBind", "descriptorBindingSampledImageUpdateAfterBind", "descriptorBindingStorageImageUpdateAfterBind", "descriptorBindingStorageBufferUpdateAfterBind", "descriptorBindingUniformTexelBufferUpdateAfterBind", "descriptorBindingStorageTexelBufferUpdateAfterBind", "descriptorBindingUpdateUnusedWhilePending", "descriptorBindingPartiallyBound", "descriptorBindingVariableDescriptorCount", "runtimeDescriptorArray", "samplerFilterMinmax", "scalarBlockLayout", "imagelessFramebuffer", "uniformBufferStandardLayout", "shaderSubgroupExtendedTypes", "separateDepthStencilLayouts", "hostQueryReset", "timelineSemaphore", "bufferDeviceAddress", "bufferDeviceAddressCaptureReplay", "bufferDeviceAddressMultiDevice", "vulkanMemoryModel", "vulkanMemoryModelDeviceScope", "vulkanMemoryModelAvailabilityVisibilityChains", "shaderOutputViewportIndex", "shaderOutputLayer", "subgroupBroadcastDynamicId", };
static const char *const FN13[] = { "robustImageAccess", "inlineUniformBlock", "descriptorBindingInlineUniformBlockUpdateAfterBind", "pipelineCreationCacheControl", "privateData", "shaderDemoteToHelperInvocation", "shaderTerminateInvocation", "subgroupSizeControl", "computeFullSubgroups", "synchronization2", "textureCompressionASTC_HDR", "shaderZeroInitializeWorkgroupMemory", "dynamicRendering", "shaderIntegerDotProduct", "maintenance4", };
static void lvp_mask(VkStructureType t, const char *what, VkBool32 *b, size_t n) {
    lvp_open(); if (!LVP.f2) return;
    static unsigned char q[1024]; memset(q, 0, sizeof q); VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkBool32 *have;
    if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) { LVP.f2(LVP.pd, &f2); have = (VkBool32 *)&f2.features; }
    else { ((VkBaseOutStructure *)q)->sType = t; f2.pNext = q; LVP.f2(LVP.pd, &f2); have = (VkBool32 *)((VkBaseOutStructure *)q + 1); }
    const char *const *nm = NULL; size_t nn = 0;
#define NM(a) (nm = a, nn = sizeof a / sizeof *a)
    if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) NM(FN10); else if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) NM(FN11);
    else if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) NM(FN12); else if (t == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) NM(FN13);
#undef NM
    for (size_t i = 0; i < n; i++) if (b[i] && !have[i]) { b[i] = VK_FALSE;
        if (i < nn) printf("SIMGAP feature %s: the profile has it, lavapipe has not (not enabled)\n", nm[i]);
        else printf("SIMGAP feature %s member %zu: the profile has it, lavapipe has not (not enabled)\n", what, i); }
}
static VKAPI_ATTR VkResult VKAPI_CALL w_cd(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a, VkDevice *o) {
    w_dev_made = 1; atomic_fetch_add(&w_dev_count, 1); w_ndext = ci->enabledExtensionCount > 64 ? 64 : ci->enabledExtensionCount;
    for (uint32_t i = 0; i < w_ndext; i++) snprintf(w_dext[i], sizeof w_dext[i], "%s", ci->ppEnabledExtensionNames[i]);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
    uint32_t api = pp.apiVersion < w_inst_api ? pp.apiVersion : w_inst_api;
    for (const VkBaseInStructure *s = ci->pNext; s; s = s->pNext) {
        const char *need = NULL; uint32_t need_api = 0; const char *name = "";
        switch ((int)s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: name = "Features2"; need_api = VK_API_VERSION_1_1; need = "VK_KHR_get_physical_device_properties2"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: name = "Vulkan11Features"; need_api = VK_API_VERSION_1_2; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: name = "Vulkan12Features"; need_api = VK_API_VERSION_1_2; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: name = "Vulkan13Features"; need_api = VK_API_VERSION_1_3; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: name = "TimelineSemaphoreFeatures"; need_api = VK_API_VERSION_1_2; need = "VK_KHR_timeline_semaphore"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES: name = "DynamicRenderingFeatures"; need_api = VK_API_VERSION_1_3; need = "VK_KHR_dynamic_rendering"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES: name = "ShaderDrawParametersFeatures"; need_api = VK_API_VERSION_1_1; need = "VK_KHR_shader_draw_parameters"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT: name = "ExtendedDynamicStateFeatures"; need_api = VK_API_VERSION_1_3; need = "VK_EXT_extended_dynamic_state"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT: name = "ExtendedDynamicState2Features"; need_api = VK_API_VERSION_1_3; need = "VK_EXT_extended_dynamic_state2"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT: name = "CustomBorderColorFeatures"; need_api = ~0u; need = "VK_EXT_custom_border_color"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT: name = "DepthClipControlFeatures"; need_api = ~0u; need = "VK_EXT_depth_clip_control"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT: name = "VertexAttributeDivisorFeatures"; need_api = ~0u; need = "VK_EXT_vertex_attribute_divisor"; break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT: name = "ImageCompressionControlFeatures"; need_api = ~0u; need = "VK_EXT_image_compression_control"; break;
        default: printf("NOTE device create chain: structure type %d not checked here\n", (int)s->sType); continue;
        }
        int ok = (need_api != ~0u && api >= need_api) || (need && (w_has(need) || (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 && w_inst_p2ext)));
        if (!ok) { w_struct_bad++; printf("FAIL %s:%d: vkCreateDevice chain holds %s on a Vulkan %u.%u device without %s\n", __FILE__, __LINE__, name,
                                           VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), need ? need : "(core in a later version only)"); fails++; }
    }
    // The simulation's gap: the profile may promise features / extensions lavapipe has not got, and the engine (rightly) enables every feature the device reports.
    //   Those are taken out of the request before it goes down (the validation layer still sees a request within the simulated device), each one named as SIMGAP:
    //   a limit of the simulation, not an engine finding
    VkDeviceCreateInfo c = *ci; const char *keep[64]; uint32_t nk = 0;
    for (uint32_t i = 0; i < ci->enabledExtensionCount && nk < 64; i++) { if (lvp_has_ext(ci->ppEnabledExtensionNames[i])) keep[nk++] = ci->ppEnabledExtensionNames[i]; else printf("SIMGAP extension %s: the profile lists it, lavapipe has not got it (not enabled)\n", ci->ppEnabledExtensionNames[i]); }
    c.enabledExtensionCount = nk; c.ppEnabledExtensionNames = keep;
    VkPhysicalDeviceFeatures ef; if (ci->pEnabledFeatures) { ef = *ci->pEnabledFeatures; lvp_mask(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, "VkPhysicalDeviceFeatures", (VkBool32 *)&ef, sizeof ef / 4); c.pEnabledFeatures = &ef; }
    // the chain copied, each structure's VkBool32 members ANDed with lavapipe's
    static unsigned char copy[16][1024]; VkBaseOutStructure *prev = NULL; int ncopy = 0; c.pNext = NULL;
    for (const VkBaseInStructure *s = ci->pNext; s && ncopy < 16; s = s->pNext) {
        size_t sz = feat_size(s->sType); VkBaseOutStructure *d = (VkBaseOutStructure *)copy[ncopy++];
        memcpy(d, s, sz ? sz : sizeof(VkBaseInStructure)); d->pNext = NULL;
        if (sz) { if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) lvp_mask(s->sType, "VkPhysicalDeviceFeatures", (VkBool32 *)&((VkPhysicalDeviceFeatures2 *)d)->features, sizeof(VkPhysicalDeviceFeatures) / 4);
                  else lvp_mask(s->sType, feat_name(s->sType), (VkBool32 *)(d + 1), (sz - sizeof(VkBaseOutStructure)) / 4); }
        if (prev) prev->pNext = d; else c.pNext = d; prev = d;
    }
    return w_dev_result = real_cd(pd, &c, a, o);
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL w_gipa(VkInstance inst, const char *n) {
    PFN_vkVoidFunction f = real_gipa(inst, n); if (!f) return NULL;
    if (!strcmp(n, "vkCreateInstance")) { real_ci = (PFN_vkCreateInstance)f; return (PFN_vkVoidFunction)w_ci; }
    if (!strcmp(n, "vkCreateDevice")) { real_cd = (PFN_vkCreateDevice)f; return (PFN_vkVoidFunction)w_cd; }
    if (!strcmp(n, "vkGetPhysicalDeviceProperties")) { real_p1 = (PFN_vkGetPhysicalDeviceProperties)f; return (PFN_vkVoidFunction)w_p1; }
    if (!strcmp(n, "vkGetPhysicalDeviceFeatures2") || !strcmp(n, "vkGetPhysicalDeviceFeatures2KHR")) { real_f2 = (PFN_vkGetPhysicalDeviceFeatures2)f; return (PFN_vkVoidFunction)w_f2; }
    if (!strcmp(n, "vkGetPhysicalDeviceProperties2") || !strcmp(n, "vkGetPhysicalDeviceProperties2KHR")) { real_p2 = (PFN_vkGetPhysicalDeviceProperties2)f; return (PFN_vkVoidFunction)w_p2; }
    if (!strcmp(n, "vkGetPhysicalDeviceExternalBufferProperties") || !strcmp(n, "vkGetPhysicalDeviceExternalBufferPropertiesKHR")) { real_xb = (PFN_vkGetPhysicalDeviceExternalBufferProperties)f; return (PFN_vkVoidFunction)w_xb; }
    return f;
}

// ---- the device as it reports itself (after the engine made its instance: the same loader, layers and profile) ----
static struct { VkPhysicalDeviceProperties p; VkPhysicalDeviceFeatures f; uint32_t next; VkExtensionProperties *ext;
                int has_f2, has_p2; VkPhysicalDeviceVulkan11Features f11; VkPhysicalDeviceVulkan12Features f12; VkPhysicalDeviceVulkan13Features f13;
                VkPhysicalDeviceTimelineSemaphoreFeatures fts; VkPhysicalDeviceDynamicRenderingFeatures fdr; VkPhysicalDeviceShaderDrawParametersFeatures fsdp;
                VkPhysicalDeviceCustomBorderColorFeaturesEXT fbc; VkPhysicalDeviceDepthClipControlFeaturesEXT fcc;
                VkPhysicalDeviceDriverProperties drv; int drv_ok; VkPhysicalDeviceFloatControlsProperties fc; int fc_ok; } DV;
static int dev_ext(const char *e) { for (uint32_t i = 0; i < DV.next; i++) if (!strcmp(DV.ext[i].extensionName, e)) return 1; return 0; }
static void device_read(void) {
    vkGetPhysicalDeviceProperties(g_pd, &DV.p); vkGetPhysicalDeviceFeatures(g_pd, &DV.f);
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &DV.next, NULL); DV.ext = calloc(DV.next + 1, sizeof *DV.ext); vkEnumerateDeviceExtensionProperties(g_pd, NULL, &DV.next, DV.ext);
    uint32_t api = DV.p.apiVersion < g_instance_api ? DV.p.apiVersion : g_instance_api;
    PFN_vkGetPhysicalDeviceFeatures2 f2 = real_f2; PFN_vkGetPhysicalDeviceProperties2 p2 = real_p2;
    DV.has_f2 = f2 && (api >= VK_API_VERSION_1_1 || w_inst_p2ext); DV.has_p2 = p2 && DV.has_f2;
    if (DV.has_f2) {   // each structure on its own, only where the device's version or extension allows it
        VkPhysicalDeviceFeatures2 q = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
#define F2(s, st, cond) do { if (cond) { memset(&s, 0, sizeof s); s.sType = st; q.pNext = &s; real_f2(g_pd, &q); q.pNext = NULL; s.pNext = NULL; } } while (0)
        F2(DV.f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, api >= VK_API_VERSION_1_2);
        F2(DV.f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, api >= VK_API_VERSION_1_2);
        F2(DV.f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, api >= VK_API_VERSION_1_3);
        F2(DV.fts, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, api >= VK_API_VERSION_1_2 || dev_ext("VK_KHR_timeline_semaphore"));
        F2(DV.fdr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, api >= VK_API_VERSION_1_3 || dev_ext("VK_KHR_dynamic_rendering"));
        F2(DV.fsdp, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, api >= VK_API_VERSION_1_1);
        F2(DV.fbc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, dev_ext("VK_EXT_custom_border_color"));
        F2(DV.fcc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, dev_ext("VK_EXT_depth_clip_control"));
#undef F2
        VkPhysicalDeviceProperties2 pq = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        if (api >= VK_API_VERSION_1_2 || dev_ext("VK_KHR_driver_properties")) { DV.drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES; pq.pNext = &DV.drv; real_p2(g_pd, &pq); ident_apply2(&pq); DV.drv_ok = 1; DV.drv.pNext = NULL; }
        if (api >= VK_API_VERSION_1_2 || dev_ext("VK_KHR_shader_float_controls")) { DV.fc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES; pq.pNext = &DV.fc; real_p2(g_pd, &pq); ident_apply2(&pq); DV.fc_ok = 1; DV.fc.pNext = NULL; }
    }
}

// ---- what is known for each simulated GPU (profiles/<name>.json; the facts players' logs and Arm's notes give) ----
typedef struct { const char *name; uint32_t vendor; int api_minor; int arm_rel, arm_p, arm_beta; int dualsrc; int clampzero; int dynren; int tlsem; int bc_native; } expect_t;
static const expect_t EXPECT[] = {
    //  name                   vendor  api  rel p beta dual clamp0 dynren tlsem bc
    { "mali-g720-r44p1",      0x13B5, 3,  44, 1, 0,   0,   1,     1,     1,    0 },   // r44p1: no dualSrcBlend, no robustness2; erratum 3779215 (r38p1-r49p0)
    { "mali-g720-r49p1",      0x13B5, 3,  49, 1, 1,   1,   1,     1,     1,    0 },   // the report's driver is a beta r49p1 ("v1.r49p1-03bet0"): the erratum kept
    { "mali-g72-r26p0",       0x13B5, 1,  26, 0, 0,   0,   0,     0,     1,    0 },   // Bifrost, Vulkan 1.1: render passes, timeline semaphores by extension
    { "mali-g72-r14p0",       0x13B5, 0,  14, 0, 0,   0,   0,     0,     0,    0 },   // Vulkan 1.0: a fence per submission, SPIR-V 1.0, Arm's release from the vendor ID
    { "powervr-bxm-8-256",    0x1010, 1,  0,  0, 0,   0,   0,     0,     1,    0 },
    { "powervr-dxt-48-1536",  0x1010, 3,  0,  0, 0,   1,   0,     1,     1,    0 },   // (a 1.4 driver: the engine's instance asks for 1.3)
    { "xclipse-920",          0x144D, 3,  0,  0, 0,   1,   0,     1,     1,    0 },   // (no BC in these reports: Samsung's driver lists none)
    { "xclipse-940",          0x144D, 3,  0,  0, 0,   1,   0,     1,     1,    0 },
    // added 2026-11-21 (generated from each profile: vendor, Vulkan version up to 1.3, Arm release from driverVersion, dualSrcBlend, the erratum range, dynamic rendering
    //   where the device has it except on Adreno (render passes by default), timeline semaphores, textureCompressionBC)
    { "adreno-830",          0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-750",          0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-740",          0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-735",          0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-710-vk13",     0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-710-vk11",     0x5143, 1,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-660",          0x5143, 1,  0 , 0, 0,   1,   0,     0,     1,    0 },
    { "adreno-650",          0x5143, 1,  0 , 0, 0,   1,   0,     0,     0,    0 },
    { "adreno-619",          0x5143, 1,  0 , 0, 0,   1,   0,     0,     0,    0 },
    { "adreno-613-vk11",     0x5143, 1,  0 , 0, 0,   1,   0,     0,     0,    1 },
    { "adreno-613-vk13",     0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "adreno-610",          0x5143, 1,  0 , 0, 0,   1,   0,     0,     0,    1 },
    { "adreno-506",          0x5143, 1,  0 , 0, 0,   1,   0,     0,     0,    0 },
    { "adreno-512-vk10",     0x5143, 0,  0 , 0, 0,   0,   0,     0,     0,    0 },
    { "turnip-adreno-650",   0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "turnip-adreno-740",   0x5143, 3,  0 , 0, 0,   1,   0,     0,     1,    1 },
    { "mali-g615-mc6-r44p1", 0x13B5, 3,  44, 1, 0,   0,   1,     1,     1,    0 },
    { "mali-g615-mc2-r44p1", 0x13B5, 3,  44, 1, 0,   0,   1,     1,     1,    0 },
    { "mali-g57-r32p1",      0x13B5, 1,  32, 1, 0,   0,   0,     0,     1,    0 },
    { "mali-g57-r38p1",      0x13B5, 3,  38, 1, 0,   0,   1,     1,     1,    0 },
    { "mali-g57-r54p1",      0x13B5, 3,  54, 1, 0,   1,   0,     1,     1,    0 },
    { "mali-g52-r32p1",      0x13B5, 1,  32, 1, 0,   0,   0,     0,     1,    0 },
    { "mali-g52-r49p1",      0x13B5, 3,  49, 1, 1,   1,   1,     1,     1,    0 },
    { "mali-g610-r32p1",     0x13B5, 1,  32, 1, 0,   0,   0,     0,     1,    0 },
    { "mali-g610-r38p1",     0x13B5, 1,  38, 1, 0,   0,   1,     0,     1,    0 },
    { "mali-g68-r32p1",      0x13B5, 1,  32, 1, 1,   0,   0,     0,     1,    0 },
    { "mali-g710-r32p1",     0x13B5, 1,  32, 1, 0,   0,   0,     0,     1,    0 },
    { "mali-g310-r44p1",     0x13B5, 3,  44, 1, 0,   0,   1,     1,     1,    0 },
    { "mali-g925-r49p1",     0x13B5, 3,  49, 1, 1,   1,   1,     1,     1,    0 },
    { "powervr-ge8320",      0x1010, 1,  0 , 0, 0,   0,   0,     0,     0,    0 },
    { "powervr-ge8322-vk13", 0x1010, 3,  0 , 0, 0,   1,   0,     1,     1,    0 },
    { "xclipse-530",         0x144D, 3,  0 , 0, 0,   1,   0,     1,     1,    0 },
    { "xclipse-540",         0x144D, 3,  0 , 0, 0,   1,   0,     1,     1,    0 },
    { "xclipse-950",         0x144D, 3,  0 , 0, 0,   1,   0,     1,     1,    0 },
};

static void decisions_and_checks(const char *profile) {
    uint32_t api = DV.p.apiVersion < g_instance_api ? DV.p.apiVersion : g_instance_api;
    int arm_drv = DV.drv_ok ? DV.drv.driverID == VK_DRIVER_ID_ARM_PROPRIETARY : DV.p.vendorID == 0x13B5;
    int sec_drv = DV.drv_ok ? DV.drv.driverID == VK_DRIVER_ID_SAMSUNG_PROPRIETARY : DV.p.vendorID == 0x144D;
    printf("DEVICE %s, vendor %#x, Vulkan %u.%u.%u, driverVersion %#x, driver %s (id %d) %s\n", DV.p.deviceName, DV.p.vendorID, VK_API_VERSION_MAJOR(DV.p.apiVersion),
           VK_API_VERSION_MINOR(DV.p.apiVersion), VK_API_VERSION_PATCH(DV.p.apiVersion), DV.p.driverVersion, DV.drv_ok ? DV.drv.driverName : "-", DV.drv_ok ? (int)DV.drv.driverID : -1, DV.drv_ok ? DV.drv.driverInfo : "");
    printf("DEVICE extensions enabled by the engine (%u):", w_ndext); for (uint32_t i = 0; i < w_ndext; i++) printf(" %s", w_dext[i]); printf("\n");
    DEC("api", "%u.%u", VK_API_VERSION_MAJOR(g_api), VK_API_VERSION_MINOR(g_api));
    DEC("arm_release", "r%dp%d", g_arm_rel, g_arm_p); DEC("arm_beta", "%d", g_arm_beta);
    DEC("dual_source_blend", "%d", g_dualsrc); DEC("dynamic_rendering", "%d", g_dynren); DEC("extended_dyn_state", "%d", g_eds); DEC("eds2", "%d", g_eds2);
    DEC("timeline_semaphores", "%d", g_tlsem); DEC("push_descriptors", "%d", !g_nopush); DEC("max_push", "%u", g_maxpush); DEC("format_lists", "%d", g_fmtlist);
    DEC("store_op_none", "%d", g_store_none); DEC("host_import", "%d", g_host_import); DEC("fd_route", "%d", g_fdmap); DEC("heap_route", "%d", g_heapmap);
    DEC("depth_clip_control", "%d", g_clipctl); DEC("custom_border", "%d", g_border); DEC("vertex_divisor", "%d", g_divisor);
    DEC("float_controls", "%d", g_fctl); DEC("clamp_zero", "%d", g_clampzero); DEC("nan_minmax", "%d", g_nanmm); DEC("exact_math", "%u", g_exact); DEC("init_locals", "%d", g_initloc);
    DEC("draw_params", "%d", g_drawp); DEC("sb_as_uniform", "%d", g_nosbsc); DEC("minmax_filter", "%d", g_minmax); DEC("aniso", "%d", g_f_aniso);
    DEC("depth_clamp", "%d", g_f_clamp); DEC("logic_op", "%d", g_f_logic); DEC("independent_blend", "%d", g_f_indblend);
    DEC("spirv10_modules", "%d", g_csm_real != NULL);
    DEC("bc_textures", "%s", nvn_fmt_vk_transcode(0x46) ? "transcoded" : nvn_fmt_vk_decode(0x46) ? "decoded" : "sampled");
    DEC("bc_gpu_layer", "%d", g_bcgpu);
    int brel = barrier_relaxed(), pdef = pass_defer_on(), apso = apso_on(), cbp = clear_by_pass(), pkeep = pso_cap();
    DEC("barrier_relaxed", "%d", brel); DEC("pass_defer", "%d", pdef); DEC("async_pso", "%d", apso); DEC("clear_by_pass", "%d", cbp); DEC("pipelines_kept", "%d", pkeep);

    // what the engine enabled exists
    CHECK(w_dev_made && w_dev_result == VK_SUCCESS, "the device was made (vkCreateDevice %d)", (int)w_dev_result);
    for (uint32_t i = 0; i < w_ndext; i++) CHECK(dev_ext(w_dext[i]), "extension %s enabled but the device does not list it", w_dext[i]);
    for (uint32_t i = 0; i < w_ndext; i++) for (uint32_t j = i + 1; j < w_ndext; j++) CHECK(strcmp(w_dext[i], w_dext[j]), "extension %s enabled twice", w_dext[i]);
    // the version and what follows from it
    CHECK(g_api == api, "g_api %#x, the device's and instance's lower version %#x", g_api, api);
    CHECK((g_csm_real != NULL) == (api < VK_API_VERSION_1_1), "SPIR-V 1.0 modules (%d) exactly on a Vulkan 1.0 device", g_csm_real != NULL);
    if (api < VK_API_VERSION_1_2) CHECK(!g_tlsem || dev_ext("VK_KHR_timeline_semaphore"), "timeline semaphores used on Vulkan 1.%u without the extension", VK_API_VERSION_MINOR(api));
    if (g_tlsem) CHECK(api >= VK_API_VERSION_1_2 ? DV.f12.timelineSemaphore : DV.fts.timelineSemaphore, "timeline semaphores used, the feature is off");
    if (api < VK_API_VERSION_1_2) CHECK(!g_fmtlist || w_has("VK_KHR_image_format_list"), "view format lists on Vulkan 1.%u without VK_KHR_image_format_list (#158)", VK_API_VERSION_MINOR(api));
    if (api < VK_API_VERSION_1_3) CHECK(!g_store_none || w_has("VK_KHR_load_store_op_none") || w_has("VK_EXT_load_store_op_none"), "store op NONE without its extension");
    if (g_dynren) CHECK(api >= VK_API_VERSION_1_3 ? DV.f13.dynamicRendering : (w_has("VK_KHR_dynamic_rendering") && DV.fdr.dynamicRendering), "dynamic rendering used where the device has none");
    if (g_eds && api < VK_API_VERSION_1_3) CHECK(w_has("VK_EXT_extended_dynamic_state"), "extended dynamic state used without the extension");
    CHECK(!g_eds2 || w_has("VK_EXT_extended_dynamic_state2"), "eds2 without the extension");
    CHECK(g_nopush == !w_has("VK_KHR_push_descriptor"), "push descriptors %d, extension enabled %d", !g_nopush, w_has("VK_KHR_push_descriptor"));
    CHECK(!g_clipctl || (w_has("VK_EXT_depth_clip_control") && DV.fcc.depthClipControl), "depth clip control used without extension / feature");
    CHECK(!g_border || (w_has("VK_EXT_custom_border_color") && DV.fbc.customBorderColors && DV.fbc.customBorderColorWithoutFormat), "custom border colours without extension / formatless feature");
    CHECK(!g_divisor || w_has("VK_EXT_vertex_attribute_divisor"), "vertex divisor without the extension");
    CHECK(!g_host_import || w_has("VK_EXT_external_memory_host"), "host import without the extension");
    CHECK(!g_minmax || (api >= VK_API_VERSION_1_2 ? DV.f12.samplerFilterMinmax : w_has("VK_EXT_sampler_filter_minmax")), "min / max filtering without the feature / extension");
    if (api >= VK_API_VERSION_1_1) CHECK(g_drawp == (api >= VK_API_VERSION_1_2 ? DV.f11.shaderDrawParameters : DV.fsdp.shaderDrawParameters) != 0, "draw parameters %d", g_drawp);
    else CHECK(g_drawp == w_has("VK_KHR_shader_draw_parameters"), "Vulkan 1.0: draw parameters %d, extension %d", g_drawp, w_has("VK_KHR_shader_draw_parameters"));
    if (api < VK_API_VERSION_1_1) CHECK(g_nosbsc == !w_has("VK_KHR_storage_buffer_storage_class"), "Vulkan 1.0: storage buffers as uniform %d", g_nosbsc);
    // optional Vulkan 1.0 features: each one the engine uses is one the device has (the fallback taken otherwise)
    CHECK(g_dualsrc == (DV.f.dualSrcBlend != 0), "dual-source blending %d, the device's dualSrcBlend %d", g_dualsrc, DV.f.dualSrcBlend);
    CHECK(g_f_aniso == (DV.f.samplerAnisotropy != 0) && g_f_clamp == (DV.f.depthClamp != 0) && g_f_logic == (DV.f.logicOp != 0) && g_f_indblend == (DV.f.independentBlend != 0)
          && g_f_nonsolid == (DV.f.fillModeNonSolid != 0) && g_f_biasclamp == (DV.f.depthBiasClamp != 0), "optional 1.0 features follow the device");
    // BC textures: sampled only where the device has them; PowerVR decoded, not transcoded
    if (!DV.f.textureCompressionBC) CHECK(nvn_fmt_vk_transcode(0x46) || nvn_fmt_vk_decode(0x46), "BC1 sampled on a device without BC");
    if (DV.p.vendorID == 0x1010) CHECK(!nvn_fmt_vk_transcode(0x46), "PowerVR: BC transcoded (#32: should be decoded)");
    // the quirk table's defaults (vendor / driver keyed), as vk_quirks.h states them
    { int v = g_arm_rel * 100 + g_arm_p;
      CHECK(g_arm_rel == (arm_drv ? (int)(DV.p.driverVersion >> 22) : 0) && g_arm_p == (arm_drv ? (int)(DV.p.driverVersion >> 12 & 0x3ff) : 0), "Arm release r%dp%d from driverVersion %#x", g_arm_rel, g_arm_p, DV.p.driverVersion);
      CHECK(g_clampzero == (g_arm_rel && ((v >= 3801 && v < 4901) || (g_arm_beta && v >= 4901 && v < 4902) || (v >= 5000 && v < 5101))), "clamp-zero erratum %d for r%dp%d%s", g_clampzero, g_arm_rel, g_arm_p, g_arm_beta ? " beta" : "");
      CHECK(g_nanmm == (g_arm_rel ? 2 : 0) && (g_exact == (g_arm_rel ? EXACT_ALL : 0u)) && g_initloc == (g_arm_rel != 0), "Arm shader rules (nan %d exact %u init %d)", g_nanmm, g_exact, g_initloc);
      const char *fce = getenv("DM_VK_FLOAT_CONTROLS");   // (SIMDEV_DC_FC: set, with the marker; from the driver check the marker applies)
      int want_fc = DV.fc_ok && DV.fc.shaderSignedZeroInfNanPreserveFloat32 && (fce ? atoi(fce) != 0 : (arm_drv || sec_drv)) && g_dcfc != 1;
      if (DV.fc_ok) CHECK(g_fctl == want_fc, "float controls %d (Arm %d, Samsung %d, supported %d, DM_VK_FLOAT_CONTROLS %s%s)", g_fctl, arm_drv, sec_drv, DV.fc.shaderSignedZeroInfNanPreserveFloat32,
                          fce ? fce : "unset", g_dcfc == 1 ? " from the driver check, marker" : g_dcfc == 2 ? " from env.txt, marker" : "");
      else CHECK(!g_fctl, "float controls asked for where the driver cannot say it has them"); }
    CHECK(g_dynren == 0 || DV.p.vendorID != 0x5143, "Adreno: render passes by default");
    // (background pipelines: every GPU since beta-exit f09e58f, unless the game's profile says async_pso_off, Legends: Arceus #34)
    CHECK(pdef == (DV.p.vendorID == 0x5143) && apso == !DM_GAME_ENG_ASYNC_PSO_OFF && cbp == (DV.p.vendorID == 0x144D), "per-vendor pass / pipeline / clear defaults");
    // barriers before passes, every game: relaxed on Adreno but Qualcomm's own drivers before Vulkan 1.3 (a name starting "Adreno": Turnip is not one), full elsewhere
    { int oldqc = DV.p.vendorID == 0x5143 && VK_API_VERSION_MINOR(DV.p.apiVersion) < 3 && !strncmp(DV.p.deviceName, "Adreno", 6);
      CHECK(brel == (DV.p.vendorID == 0x5143 && !oldqc), "barriers relaxed %d (Adreno %d, Qualcomm's driver before 1.3 %d)", brel, DV.p.vendorID == 0x5143, oldqc); }
    // draw state in each pipeline on Qualcomm's Vulkan 1.0 / 1.1 drivers for a game whose profile says qc11_no_eds (Legends: Arceus #34)
    if (DM_GAME_ENG_QC11_NO_EDS && DV.p.vendorID == 0x5143 && VK_API_VERSION_MAJOR(DV.p.apiVersion) == 1 && VK_API_VERSION_MINOR(DV.p.apiVersion) < 2) CHECK(!g_eds, "Qualcomm 1.1: draw state per draw");
    if (DV.p.vendorID == 0x13B5) CHECK(pkeep == 800, "Mali: at most 800 pipelines kept (%d)", pkeep);
    if (DV.p.vendorID == 0x144D) CHECK(!g_host_import, "Xclipse: pools not imported as host memory (#39)");
    // the profile's own facts
    if (profile && *profile) {
        const expect_t *e = NULL; for (unsigned i = 0; i < sizeof EXPECT / sizeof *EXPECT; i++) if (!strcmp(EXPECT[i].name, profile)) e = &EXPECT[i];
        // the auto-discovered profiles (discover.py) have no row: profile_env.py's SIMDEV_EXPECT, computed from the profile by the rows' own rules
        static expect_t ax; const char *xs = getenv("SIMDEV_EXPECT"); unsigned v[10];
        if (xs && *xs && sscanf(xs, "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) == 10) {
            ax = (expect_t){ profile, v[0], (int)v[1], (int)v[2], (int)v[3], (int)v[4], (int)v[5], (int)v[6], (int)v[7], (int)v[8], (int)v[9] };
            if (e) CHECK(e->vendor == ax.vendor && e->api_minor == ax.api_minor && e->arm_rel == ax.arm_rel && e->arm_p == ax.arm_p && e->arm_beta == ax.arm_beta && e->dualsrc == ax.dualsrc
                         && e->clampzero == ax.clampzero && e->dynren == ax.dynren && e->tlsem == ax.tlsem && e->bc_native == ax.bc_native,
                         "%s: the hand-written row and profile_env.py's SIMDEV_EXPECT (%s) differ", profile, xs);
            else e = &ax; }
        CHECK(e != NULL, "no expectations for profile %s", profile);
        if (e) {
            CHECK(DV.p.vendorID == e->vendor, "%s: vendor %#x, simulated %#x (is the profile layer on?)", profile, e->vendor, DV.p.vendorID);
            CHECK((int)VK_API_VERSION_MINOR(g_api) == e->api_minor, "%s: Vulkan 1.%d expected, the engine runs 1.%u", profile, e->api_minor, VK_API_VERSION_MINOR(g_api));
            CHECK(g_arm_rel == e->arm_rel && g_arm_p == e->arm_p && g_arm_beta == e->arm_beta, "%s: Arm r%dp%d%s expected, decided r%dp%d%s", profile, e->arm_rel, e->arm_p, e->arm_beta ? " beta" : "", g_arm_rel, g_arm_p, g_arm_beta ? " beta" : "");
            CHECK(g_dualsrc == e->dualsrc, "%s: dual-source blending %d expected, %d", profile, e->dualsrc, g_dualsrc);
            CHECK(g_clampzero == e->clampzero, "%s: clamp-zero erratum %d expected, %d", profile, e->clampzero, g_clampzero);
            CHECK(g_dynren == e->dynren, "%s: dynamic rendering %d expected, %d", profile, e->dynren, g_dynren);
            CHECK(g_tlsem == e->tlsem, "%s: timeline semaphores %d expected, %d", profile, e->tlsem, g_tlsem);
            CHECK((!nvn_fmt_vk_transcode(0x46) && !nvn_fmt_vk_decode(0x46)) == e->bc_native, "%s: BC sampled natively %d expected", profile, e->bc_native);
            if (!strcmp(profile, "mali-g720-r44p1")) CHECK(!dev_ext("VK_EXT_robustness2"), "r44p1 lists VK_EXT_robustness2: the profile is wrong");
        }
    }
}

// ---- the unified driver layer (vk_quirks.h), where the engine has it: the GPU description it decides from, and every quirk's default for this GPU ----
#ifdef VK_QUIRKS_H
static void quirk_checks(void) {
    const gpu_id *g = &g_gpuid;
    printf("QUIRKS gpu_id: vendor %#x, driver version %#x, driver id %d, Arm r%dp%d%s, api %#x, dynren %d, custom %d, port %d, name %s\n", g->vendor, g->driver_version,
           g->driver_id, g->arm_rel, g->arm_p, g->arm_beta ? " beta" : "", g->api, g->dynren, g->custom, g->port, g->name);
    for (int q = 0; q < Q_N; q++) printf("QUIRK %-50s default %d, decided %d%s\n", Q_INFO[q].what, quirk_default(q, g), g_qval[q], g_qset[q] ? "" : " (not decided yet)");
    CHECK(g->vendor == DV.p.vendorID && g->driver_version == DV.p.driverVersion && !strcmp(g->name, DV.p.deviceName), "gpu_id: the device's identity");
    // (the driver ID as mtl_init reads it: with the driver properties, where it reads them, Vulkan 1.2 or 1.1 with the float-controls extension; elsewhere -1, and
    //   Arm's own driver is told by its vendor ID there, as before the table)
    { int rd = (g_api >= VK_API_VERSION_1_2 || g_fctl_ext) && (g_api >= VK_API_VERSION_1_2 || g_drvprops_ext) && DV.drv_ok;
      CHECK(g->driver_id == (rd ? (int)DV.drv.driverID : -1), "gpu_id: driver id %d, the device's %d (read by mtl_init: %d)", g->driver_id, DV.drv_ok ? (int)DV.drv.driverID : -1, rd); }
    CHECK(g->arm_rel == g_arm_rel && g->arm_p == g_arm_p && g->arm_beta == g_arm_beta && g->dynren == g_dynren && !g->custom, "gpu_id: complete after set-up");
    int arm = g->vendor == VEN_ARM, sec = g->vendor == VEN_SAMSUNG, img = g->vendor == VEN_IMG, qc = g->vendor == VEN_QCOM;
    CHECK(quirk_default(Q_DMAHEAP, g) == (arm || sec) && quirk_default(Q_HOST_IMPORT_OFF, g) == sec && quirk_default(Q_CLEAR_BY_PASS, g) == sec, "Mali / Xclipse memory and clear rules");
    int oldqc = qc && VK_API_VERSION_MINOR(g->api) < 3 && !strncmp(g->name, "Adreno", 6), qc11 = qc && VK_API_VERSION_MAJOR(g->api) == 1 && VK_API_VERSION_MINOR(g->api) < 2;
    CHECK(quirk_default(Q_PCACHE_GAPS_OFF, g) == img && quirk_default(Q_ARM_FORMATS, g) == arm && quirk_default(Q_BARRIER_RELAXED, g) == (qc && !oldqc), "PowerVR / Arm / Adreno rules");
    CHECK(quirk_default(Q_TEX_HALF_UNSAFE, g) == (g->arm_rel && g->arm_rel < 49), "half textures unsafe before r49 (#82): %d for r%d", quirk_default(Q_TEX_HALF_UNSAFE, g), g->arm_rel);
    CHECK(quirk_default(Q_VIEW_QUARANTINE, g) == (g->arm_rel && g->dynren ? 2048 : 0), "view quarantine on Arm's driver with dynamic rendering");
    CHECK(!quirk_default(Q_G615_SAFE, g) || strstr(g->name, "G615"), "the G615 start-up set only on a G615");
    CHECK(quirk_default(Q_BC_GPU_OFF, g) == img && quirk_default(Q_BC_DECODE, g) == img && quirk_default(Q_GPU_TIME_OFF, g) == (arm || img),
          "PowerVR: BC converted on the CPU and decoded; no GPU timestamps on Mali and PowerVR (every game)");
    // the game, from its profile (game.tsv port / engine_quirks → DM_PORT_* / DM_GAME_ENG_*), and the rules that read it, as 6e0e1dd's sites had them for this game
    //   (background pipelines as beta-exit's since f09e58f: every GPU unless async_pso_off)
    CHECK(g->port == (DM_PORT_ARCEUS ? PORT_ARCEUS : DM_PORT_PALDEA ? PORT_PALDEA : PORT_GALAR), "gpu_id: port %d, the profile's %s", g->port, DM_PORT_ARCEUS ? "arceus" : DM_PORT_PALDEA ? "paldea" : "galar");
    CHECK(!!(g->eng & ENG_QC11_NO_EDS) == !!DM_GAME_ENG_QC11_NO_EDS && !!(g->eng & ENG_ASYNC_PSO_OFF) == !!DM_GAME_ENG_ASYNC_PSO_OFF
          && !!(g->eng & ENG_DRIVER_SELFTESTS) == !!DM_GAME_ENG_DRIVER_SELFTESTS && !!(g->eng & ENG_NVN_ASYNC_NEVER) == !!DM_GAME_ENG_NVN_ASYNC_NEVER
          && !!(g->eng & ENG_MALI_LIT_STANDIN) == !!DM_GAME_ENG_MALI_LIT_STANDIN, "gpu_id: engine_quirks bits %#x, not the profile's", g->eng);
    CHECK(quirk_default(Q_NO_DYNREN, g) == qc && quirk_default(Q_ASYNC_PSO, g) == !DM_GAME_ENG_ASYNC_PSO_OFF && quirk_default(Q_NO_EDS, g) == ((qc11 && DM_GAME_ENG_QC11_NO_EDS) || (g->vendor == 0x13B5 && !g->custom && (g->driver_version >> 22) > 0 && (g->driver_version >> 22) < 44)),
          "rules for this game: render passes and draw state on Adreno, background pipelines on every GPU (beta-exit f09e58f)");
    // Faster rendering's Auto: Adreno by name and Arm's own driver (v0.97), PowerVR and Xclipse (perf-1124 fix 3); the held frame where both were run together
    { int arm_own = g->driver_id >= 0 ? g->driver_id == 9 : g->vendor == 0x13B5, async = (strstr(g->name, "Adreno") || arm_own || img || g->vendor == 0x144D) && !DM_GAME_ENG_NVN_ASYNC_NEVER;
      CHECK(quirk_default(Q_NVN_ASYNC, g) == async && quirk_default(Q_FRAME_HOLD, g) == (strstr(g->name, "Adreno") || arm_own ? 2 : 0) && quirk_default(Q_DRIVER_SELFTESTS, g) == !!DM_GAME_ENG_DRIVER_SELFTESTS,
            "the replay thread's Auto (%d), the held frame (%d) and the start-up self-tests for this game", quirk_default(Q_NVN_ASYNC, g), quirk_default(Q_FRAME_HOLD, g)); }
    // Mali's white-screen fix as mali-all e6f6d0c decided it (mw_default_route / mw_enabled with arm_own = Arm's own driver, galar = DM_PORT_GALAR): route A and the
    //   check on Arm's own driver with Sword / Shield, nowhere else (PanVK's driver ID included)
    CHECK(quirk_default(Q_B10_STANDIN, g) == (MALI_LIT_DEFAULT && g->arm_rel != 0 && DM_PORT_GALAR) && quirk_default(Q_MALI_WHITE_CHECK, g) == (MALI_LIT_DEFAULT && g->arm_rel != 0 && DM_PORT_GALAR),
          "the white-screen fix's defaults: stand-in %d, check %d (Arm's own driver %d, Sword / Shield %d)", quirk_default(Q_B10_STANDIN, g), quirk_default(Q_MALI_WHITE_CHECK, g),
          g->arm_rel != 0, DM_PORT_GALAR);
    if (!getenv("DM_VK_BARRIER")) CHECK(g_qset[Q_BARRIER_RELAXED] && g_qval[Q_BARRIER_RELAXED] == quirk_default(Q_BARRIER_RELAXED, g), "barriers decided as the table says");
    // the table for code outside nvn_vk.c (hle_nvn.c rq_on): while the device was being made, after; every rule's decision at its site so far, where no setting decided
    CHECK(g_up_early == 1 && g_up_late == 1, "while the device was being made: the early rules answered as the table (%d), the late ones -1 (%d)", g_up_early, g_up_late);
    { int same = 1; for (int q = 0; q < Q_N; q++) same &= nvn_vk_quirk_default(q) == quirk_default(q, g); CHECK(same, "nvn_vk_quirk_default answers every rule once the device is made"); }
    for (int q = 0; q < Q_N; q++) {
        if (!g_qset[q] || g_qenv[q]) continue;
        int d = quirk_default(q, g), want = d;
        if (q == Q_DMAHEAP || q == Q_FLOAT_CONTROLS) continue;   // (they note what the route or the driver allowed: checked above)
        if (q == Q_PCACHE_GAPS_OFF) want = d && !g_dynren;      // (applied with render passes only)
        if (q == Q_EXACT_MATH) want = d ? (int)EXACT_ALL : 0;   // (the mask of every edge case)
        if (q == Q_FRAME_HOLD) want = !DM_GAME_ENG_FRAME_HOLD_OFF && nvn_async_on && nvn_async_on() ? d : 0;   // (only where the replay has its own thread: none here)
        CHECK(g_qval[q] == want, "%s: %d decided at its site, the table's default %d", Q_INFO[q].what, g_qval[q], want);
    }
    if (!getenv("DM_VK_PIPE_KEEP")) CHECK(pso_cap() == quirk_default(Q_PIPE_KEEP, g), "pipelines kept %d, the table %d", pso_cap(), quirk_default(Q_PIPE_KEEP, g));
    // GPU timestamps (completion_start): Q_GPU_TIME_OFF inverted at its site; Legends: Arceus (gpu_time_auto_only) only with Graphics Auto
    { const char *gt = getenv("DM_VK_GPU_TIME"), *gg = getenv("DM_GRAPHICS");
      int want = g_tsq_ok() && !quirk_default(Q_GPU_TIME_OFF, g) && (!DM_GAME_ENG_GPU_TIME_AUTO_ONLY || (gg && !strcmp(gg, "auto")));
      if (!gt) CHECK((g_tsq != VK_NULL_HANDLE) == want, "GPU timestamps %d, the table and the profile say %d", g_tsq != VK_NULL_HANDLE, want);
      if (!gt) CHECK(g_qset[Q_GPU_TIME_OFF] && g_qval[Q_GPU_TIME_OFF] == quirk_default(Q_GPU_TIME_OFF, g), "GPU timestamps off noted as the table says"); }
    // a driver-check value with a crash marker (SIMDEV_DC_FC): the check's DM_VK_FLOAT_CONTROLS=1 is this driver's default (the marker applies, its [quirk] line says
    //   "driver check"); the player's env.txt skips the marker
    if (g_dcfc) {
        int sup = DV.fc_ok && DV.fc.shaderSignedZeroInfNanPreserveFloat32 && (g_api >= VK_API_VERSION_1_2 || g_fctl_ext);
        CHECK(g_qset[Q_FLOAT_CONTROLS] && g_qenv[Q_FLOAT_CONTROLS] == (g_dcfc == 1 ? 2 : 1), "float controls noted from %d, want %s", g_qenv[Q_FLOAT_CONTROLS], g_dcfc == 1 ? "the driver check (2)" : "env.txt (1)");
        if (g_dcfc == 1) CHECK(!g_fctl, "DM_VK_FLOAT_CONTROLS=1 from the driver check with a float_controls_crashed marker: no float controls (the marker applies to a default)");
        else CHECK(g_fctl == sup, "DM_VK_FLOAT_CONTROLS=1 in env.txt with the marker: float controls %d (the player's setting tries again), supported %d", g_fctl, sup);
    }
}
#else
static void quirk_checks(void) { printf("QUIRKS: this engine has no vk_quirks.h (the per-GPU choices are checked one by one above)\n"); }
#endif

// ---- a little real work through the engine ----
static void work(void) {
    // samplers: anisotropy, min / max reduction, a custom border colour, depth compare; each through samp_vk's fallbacks (the validation layer judges the result)
    samp_par sp[] = {
        { .minf = 5, .magf = 1, .ws = 0, .wt = 0, .wr = 0, .lod_max = 16, .aniso = 16 },
        { .minf = 5, .magf = 1, .reduction = 1, .lod_max = 16, .aniso = 1 },
        { .minf = 5, .magf = 1, .reduction = 2, .lod_max = 16, .aniso = 1 },
        { .minf = 1, .magf = 1, .ws = 5, .wt = 5, .wr = 5, .lod_max = 1, .aniso = 1, .border = { 0.25f, 0.5f, 0.75f, 1 } },   // (NVN wrap 5: clamp to border)
        { .minf = 1, .magf = 1, .ws = 3, .wt = 2, .wr = 3, .lod_max = 1, .aniso = 1 },   // (NVN wrap 2 / 3: the mirror-clamp modes)
        { .minf = 1, .magf = 1, .cmp_mode = 1, .cmp_func = 3, .lod_max = 1, .aniso = 1 },
    };
    g_step = "samplers"; for (unsigned i = 0; i < sizeof sp / sizeof *sp; i++) { VkSampler s = samp_vk(&sp[i]); CHECK(s != VK_NULL_HANDLE, "sampler %u not made", i); }
    // a memory pool the GPU writes and the CPU reads back (the route the engine chose: host import, fd export, dma-buf heap or moved mappings)
    size_t n = 8u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mem != MAP_FAILED, "mmap"); if (mem == MAP_FAILED) return;
    memset(mem, 0x11, n);
    g_step = "mtl_pool_add"; mtl_pool_add(mem, n);
    VkBuffer b; VkDeviceSize o; CHECK(pool_buf((uint64_t)(uintptr_t)mem + 4096, 4096, &b, &o, NULL), "the pool has no GPU buffer");
    int shifted = 0;
    // The last route, the driver's own mapping moved onto the pool (mremap), is how a phone's driver without import or export is used: its GPU reaches the memory by
    //   the GPU's own addresses. lavapipe's "GPU" is the CPU reading through the mapping the engine moved away, and it crashes in its worker thread (it does so on
    //   plain lavapipe too, DM_VK_NO_HOST_IMPORT=1 DM_VK_NO_FDMAP=1). There the submissions go without the fill: the fences / timeline are still exercised
    int moved = !g_host_import && !g_fdmap && !g_heapmap;
    if (moved) { printf("SIMGAP pool: the moved-mapping route (mremap) cannot run on lavapipe (its GPU is the CPU reading the moved mapping): empty submissions instead\n");
        for (int round = 0; round < 3; round++) { g_step = "submission (empty)"; mtl_sub_begin(); uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish(); } }
    for (int round = 0; round < 3 && !moved; round++) {   // (three: the fence / timeline ring reused). Pages 1, 3, 5 filled; 2, 4, 6 must stay as they were
        uint8_t *pg = mem + 4096 * (2 * round + 1); uint32_t v = 0xC0DE0000u + (uint32_t)round;
        g_step = "submission"; mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)pg, 4096, v);
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX);
        mtl_finish();
        const uint32_t *w = (const uint32_t *)pg, *nx = (const uint32_t *)(pg + 4096); int bad = 0, next = 0;
        for (int k = 0; k < 1024; k++) { bad += w[k] != v; next += nx[k] == v; }
        // lavapipe's exported opaque-fd memory starts with a page of its own (llvmpipe's fd allocations keep a header there): mapped at offset 0, as the engine does,
        //   everything lands one page off. An opaque fd's layout is the driver's business (Vulkan says nothing about mmap of one), so this is lavapipe's answer, noted
        if (bad == 1024 && next == 1024 && g_fdmap && g_fdtype == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) { shifted++; continue; }
        CHECK(!bad, "round %d: the GPU's fill not seen in the pool by the CPU (%d of 1024 words; first %#x)", round, bad, w[0]);
        CHECK(!next && nx[1] == 0x11111111u, "round %d: the fill went past its range", round);
    }
    if (shifted) printf("SIMGAP pool: the opaque-fd route (lavapipe's fd mapped at offset 0) put the GPU's writes one page after the pool's (%d of 3 rounds): lavapipe's fd layout, see README\n", shifted);
    printf("WORK pool route: %s\n", g_host_import ? "host import" : g_heapmap ? "dma-buf heap" : g_fdmap ? "fd export" : "driver memory moved onto the pool");
}

// A depth / stencil copy drawn (zcopy_draw: the copy_t2t route for a depth format that is no blit destination, D24S8 on Qualcomm's driver), 16x16 scaled to 8x8, in
//   a submission of the game's kind: with DM_VK_FRAME_HOLD=2 a held one, whose command buffer outside passes is a secondary (the copy's pass must become a segment of
//   its own, rend_begin; on Adreno's render passes a vkCmdBeginRenderPass there is invalid, which the validation layer reports). Then read back: each destination
//   texel the source's at (2x + 1, 2y + 1), depth and stencil; and draw_'s pipeline-bind cache told (g_gfx_bind_gen). Only on Vulkan 1.1 and newer (view_get chains
//   VkImageViewUsageCreateInfo, Vulkan 1.1's)
static vtex *zt_image(VkFormat f, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    vtex *x = calloc(1, sizeof *x); x->refs = 1; x->fmt = f; pthread_mutex_init(&x->vmu, NULL);
    VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage };
    if (vkCreateImage(g_dev, &ci, NULL, &x->img)) { free(x); return NULL; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, x->img, &rq); x->a = vk_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VKCHECK(vkBindImageMemory(g_dev, x->img, x->a.b->mem, x->a.off));
    x->w = w; x->h = h; x->d = 1; x->levels = 1; x->layers = 1; x->samples = 1; x->usage = usage; x->aspect = fmt_aspect(f); x->fixed_fmt = 1;   // (depth: not mutable, as tex_new)
    return x;
}
// The submission ring reused twice over (NSUB slots): without timeline semaphores each slot's fence is reset and used again (DM_VK_FENCE_REUSE, fable-cpu fix 4),
//   which the validation layer judges (a fence reset while pending is an error); and sub_split's interval: none for a frame held whole where the profile says vk_split_held whole, else the profile's vk_split
static void ring_work(void) {
    g_step = "submission ring"; uint64_t f = 0;
    for (int i = 0; i < 2 * NSUB + 1; i++) { mtl_sub_begin(); f = mtl_sub_commit(); }
    CHECK(mtl_wait(f, 2000000000ull), "the ring's last submission (%llu) done", (unsigned long long)f); mtl_finish();
    for (int i = 0; i < NSUB; i++) if (!g_tlsem) CHECK(g_subs[i].vf != VK_NULL_HANDLE, "slot %d has its fence", i);
    if (!getenv("DM_VK_SPLIT") && !getenv("DM_VK_SPLIT_HELD")) CHECK(split_every() == (hold_on() && DM_GAME_VK_SPLIT_HELD_WHOLE ? 0 : DM_GAME_VK_SPLIT), "split every %d draws (held %d, profile %d, held frames whole %d)", split_every(), hold_on(), DM_GAME_VK_SPLIT, DM_GAME_VK_SPLIT_HELD_WHOLE);
    printf("submissions: %s, split every %d draws (0: held frames whole)\n", g_tlsem ? "timeline semaphore" : "a fence per slot, reset and reused", split_every());
}
static void zcopy_work(void) {
    enum { S = 16, D = 8 };
    if (g_api < VK_API_VERSION_1_1) { printf("SIMGAP zcopy: not on Vulkan 1.0 (view_get's view usage structure is Vulkan 1.1's)\n"); return; }
    VkFormat f = VK_FORMAT_UNDEFINED; static const VkFormat cand[4] = { VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM };
    for (int i = 0; i < 4 && !f; i++) if (fmt_feat(cand[i], VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) f = cand[i];
    if (!f) { printf("SIMGAP zcopy: no depth format to sample, draw into and copy\n"); return; }
    vtex *s = zt_image(f, S, S, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
         *d = zt_image(f, D, D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    if (!s || !d) { printf("SIMGAP zcopy: the images could not be made (format %d: the simulated GPU's formats are not all lavapipe's)\n", f); return; }
    int st = fmt_has_stencil(f), d24 = f == VK_FORMAT_D24_UNORM_S8_UINT, d16 = f == VK_FORMAT_D16_UNORM, dsz = d16 ? 2 : 4;
    // the source: depth (x + y * 16 + 1) / 300, stencil (x * 13 + y * 7 + 5) & 255, through a host buffer
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = S * S * 5, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    VkBuffer hb; VKCHECK(vkCreateBuffer(g_dev, &bci, NULL, &hb)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, hb, &rq);
    vmalloc_t hm = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1); VKCHECK(vkBindBufferMemory(g_dev, hb, hm.b->mem, hm.off));
    uint8_t *hp = (uint8_t *)hm.b->map + hm.off; uint32_t want_d[S * S];
    for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) { double v = (x + y * S + 1) / 300.0; uint32_t w;
        if (d24) w = (uint32_t)(v * 16777215.0 + 0.5); else if (d16) w = (uint32_t)(v * 65535.0 + 0.5); else { float fv = (float)v; memcpy(&w, &fv, 4); }
        want_d[y * S + x] = w; memcpy(hp + (y * S + x) * dsz, &w, (size_t)dsz); hp[S * S * dsz + y * S + x] = (uint8_t)((x * 13 + y * 7 + 5) & 255); }
    g_step = "zcopy: the source";
    img_init(s, 0); img_init(d, 0);   // (to GENERAL, where every image stays)
    { VkCommandBuffer cb = once_begin();
      VkBufferImageCopy bc[2] = { { .bufferOffset = 0, .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, .imageExtent = { S, S, 1 } },
                                  { .bufferOffset = (VkDeviceSize)S * S * dsz, .imageSubresource = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1 }, .imageExtent = { S, S, 1 } } };
      vkCmdCopyBufferToImage(cb, hb, s->img, VK_IMAGE_LAYOUT_GENERAL, st ? 2 : 1, bc); once_end(cb); }
    // the copy, in a submission as copy_t2t makes it (outside any pass, ordered after: the barrier)
    g_step = "zcopy: the copy";
    uint64_t gen0 = g_gfx_bind_gen;
    mtl_sub_begin(); int held = g_sub->dec;
    VkImageSubresourceLayers ss = { s->aspect, 0, 0, 1 }, ds = { d->aspect, 0, 0, 1 };
    int took = zcopy_draw(s, d, &ss, &ds, (VkOffset2D){ 0, 0 }, (VkOffset2D){ S, S }, (VkOffset2D){ 0, 0 }, (VkExtent2D){ D, D });
    barrier_all(CB());
    uint64_t fence = mtl_sub_commit(); mtl_wait(fence, UINT64_MAX); mtl_finish();
    CHECK(took, "zcopy_draw took the copy (format %d)", f);
    CHECK(g_gfx_bind_gen == gen0 + 1, "zcopy_draw told draw_'s pipeline-bind cache (generation %llu, was %llu)", (unsigned long long)g_gfx_bind_gen, (unsigned long long)gen0);
    // read back
    g_step = "zcopy: read back";
    memset(hp, 0xcd, S * S * 5);
    { VkCommandBuffer cb = once_begin();
      VkBufferImageCopy rc[2] = { { .bufferOffset = 0, .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, .imageExtent = { D, D, 1 } },
                                  { .bufferOffset = (VkDeviceSize)D * D * dsz, .imageSubresource = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1 }, .imageExtent = { D, D, 1 } } };
      vkCmdCopyImageToBuffer(cb, d->img, VK_IMAGE_LAYOUT_GENERAL, hb, st ? 2 : 1, rc); once_end(cb); }
    int bad_d = 0, bad_s = 0; uint32_t got_d = 0, exp_d = 0; unsigned got_s = 0, exp_s = 0;
    for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) {
        int sx = 2 * x + 1, sy = 2 * y + 1; uint32_t w = want_d[sy * S + sx], g = 0; memcpy(&g, hp + (y * D + x) * dsz, (size_t)dsz);
        int ok = d24 ? (int)((g & 0xffffff) - w) >= -2 && (int)((g & 0xffffff) - w) <= 2 : d16 ? (int)(g & 0xffff) - (int)w >= -2 && (int)(g & 0xffff) - (int)w <= 2 : g == w;
        if (!ok && !bad_d++) { got_d = g; exp_d = w; }
        if (st) { unsigned ws = (unsigned)((sx * 13 + sy * 7 + 5) & 255), gs = hp[D * D * dsz + y * D + x]; if (gs != ws && !bad_s++) { got_s = gs; exp_s = ws; } } }
    CHECK(!bad_d && !bad_s, "the copy drawn into format %d (%s, %s): depth %d of %d texels wrong (first %#x, want %#x), stencil %d (first %u, want %u)", f, held ? "a held submission" : "a submission as recorded",
          g_dynren ? "dynamic rendering" : "render passes", bad_d, D * D, got_d, exp_d, bad_s, got_s, exp_s);
    printf("WORK zcopy: format %d, %s, %s: %s\n", f, held ? "held (its pass a segment of its own)" : "as recorded", g_dynren ? "dynamic rendering" : "render passes", bad_d || bad_s ? "WRONG" : "right");
}
// An integer texture under a float sampler, as the translator declares every sampled texture (tex_view_for): an 8 / 16-bit one keeps its own format (6e0e1dd's
//   binding; DM_VK_INT_TEX_EMPTY=1: the empty texture, no view), a 32-bit one is viewed as float (the same bits) where its image allows another format, else in its
//   own (DM_VK_INT_TEX=keep: its own always). R32_UINT can be a render target, so its 2D images have fixed formats wherever format lists exist (and on Mali): a 3D
//   one (mutable without a list everywhere) shows the float view. The textures made and registered as the game's are (tex_res → tex_new, mtl_reg_tex); the view's
//   format read back from the image's view cache. Only on Vulkan 1.1 and newer (view_get, as zcopy_work)
static void inttex_work(void) {
    if (g_api < VK_API_VERSION_1_1) { printf("SIMGAP inttex: not on Vulkan 1.0 (view_get's view usage structure is Vulkan 1.1's)\n"); return; }
    const char *ie = getenv("DM_VK_INT_TEX"), *ze = getenv("DM_VK_INT_TEX_EMPTY"); int keep = ie && !strcmp(ie, "keep"), empty = ze && atoi(ze) > 0;
    static const struct { int nvn, target; VkFormat own, as_float; const char *name; } F[3] = {   // (NVN formats 0x2C RGBA16UI, 0x0B R32UI: tex/nvn_fmt_vk.c; target 1 2D, 2 3D)
        { 0x2C, 1, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_UNDEFINED, "R16G16B16A16_UINT" }, { 0x0B, 1, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT, "R32_UINT" },
        { 0x0B, 2, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT, "R32_UINT 3D" } };
    for (int i = 0; i < 3; i++) {
        if (!fmt_feat(F[i].own, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) { printf("SIMGAP inttex: %s cannot be sampled on this device\n", F[i].name); continue; }
        g_step = "inttex";
        spv_res r; memset(&r, 0, sizeof r); r.kind = SPV_RES_TEX; r.dim = F[i].target == 2 ? SPV_DIM_3D : SPV_DIM_2D; r.sclass = 0;   // a combined sampler, its class float
        tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.serial = nvn_new_serial(); t.d.format = F[i].nvn; t.d.target = F[i].target;
        t.d.width = t.d.height = 4; t.d.depth = F[i].target == 2 ? 4 : 1; t.d.levels = 1;
        int tid = 3900 + i; mtl_reg_tex(tid, &t, NULL);
        pthread_mutex_lock(&g_rmu); VkImageView v = tex_view_for(tid, &r, 0, NULL, 1, 0); pthread_mutex_unlock(&g_rmu);   // (its caller holds g_rmu: push_stage; a fragment shader that writes no image)
        vtex *x = tex_res(&t); VkFormat got = VK_FORMAT_UNDEFINED;
        if (x && v) { pthread_mutex_lock(&x->vmu); for (int k = 0; k < x->nviews; k++) if (x->views[k].v == v) got = (VkFormat)x->views[k].k.fmt; pthread_mutex_unlock(&x->vmu); }
        VkFormat want = !F[i].as_float ? (empty && !keep ? VK_FORMAT_UNDEFINED : F[i].own)
                      : !keep && x && (!x->fixed_fmt || F[i].as_float == x->fmt || F[i].as_float == x->twin) ? F[i].as_float : F[i].own;
        CHECK(x && got == want, "%s under a float sampler: view format %d, want %d (its image: %s; DM_VK_INT_TEX %s, DM_VK_INT_TEX_EMPTY %d)", F[i].name, got, want,
              !x ? "not made" : x->fixed_fmt == 1 ? "formats fixed, not mutable" : x->fixed_fmt == 2 ? "formats fixed, a format list" : "mutable", keep ? "keep" : "unset", empty);
        printf("WORK inttex: %s under a float sampler: %s\n", F[i].name, !v ? "the empty texture" : got == F[i].own ? "its own format" : got == F[i].as_float ? "a float view (the same bits)" : "another view");
        mtl_reg_tex(tid, NULL, NULL);
    }
}

// ---- The draw-state shadow (nvn_vk.c, DM_VK_STATE_SHADOW) and the draw memo (DM_VK_DRAW_MEMO): what draw_ leaves out must be what the driver already has.
//   The reset rules on real submissions (a new command buffer, other graphics work); the pushed descriptors and the index buffer through stand-ins for the
//   driver's calls (the count and the bindings each call carries); tex_view_for's remembered answer against a texture ID registered again
static uint32_t sw_calls, sw_n, sw_bind[8];
static VKAPI_ATTR void VKAPI_CALL sw_push(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout pl, uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    (void)cb; (void)bp; (void)pl; (void)set; sw_calls++; sw_n = n; for (uint32_t i = 0; i < n && i < 8; i++) sw_bind[i] = w[i].dstBinding; }
static uint32_t si_calls; static VkDeviceSize si_off;
static VKAPI_ATTR void VKAPI_CALL sw_index(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) { (void)cb; (void)b; (void)t; si_calls++; si_off = o; }
static void shadow_work(void) {
    g_step = "state shadow";
    // reset rules, on the engine's own submissions
    mtl_sub_begin(); int held = g_sub->dec; VkCommandBuffer cb = CB();
    sh_sync(cb); SH.have = SH_VP | SH_SC; SH.vb_have = 1; SH.ib_have = 1; uint64_t ds0 = SH.dstamp;
    sh_sync(cb); CHECK(SH.have == (SH_VP | SH_SC) && SH.vb_have == 1 && SH.ib_have == 1 && SH.dstamp == ds0, "state shadow: kept in the same command buffer");
    g_gfx_bind_gen++; sh_sync(cb); CHECK(!SH.have && !SH.vb_have && !SH.ib_have && SH.dstamp != ds0, "state shadow: forgotten after other graphics work (g_gfx_bind_gen)");
    SH.have = SH_VP; sh_sync(cb); CHECK(SH.have == SH_VP, "state shadow: kept again");
    uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
    int same = 0;
    for (int i = 0; i <= NSUB; i++) { mtl_sub_begin(); if (CB() == cb) { same = 1; break; } f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish(); }   // (until the ring hands the same handle out again)
    if (!same) mtl_sub_begin();
    sh_sync(CB()); CHECK(!SH.have, "state shadow: forgotten in a new submission%s", same ? " (the same command buffer handle, begun again)" : "");
    f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
    printf("WORK state shadow: reset rules checked (%s frames%s)\n", held ? "held" : "plain", same ? ", a command buffer handle reused" : "");
    // pushed descriptors: the writes the driver has under the same layout left out
    PFN_vkCmdPushDescriptorSetKHR keep = p_push; p_push = sw_push; g_dsh = 1; g_dsh_cnt = 1; sh_forget(); sh_desc_forget();
    VkDescriptorBufferInfo bi = { (VkBuffer)(uintptr_t)0x1000, 256, 64 }; VkDescriptorImageInfo ii = { (VkSampler)(uintptr_t)0x2000, (VkImageView)(uintptr_t)0x3000, VK_IMAGE_LAYOUT_GENERAL };
    VkBufferView bv = (VkBufferView)(uintptr_t)0x4000; VkPipelineLayout pa = (VkPipelineLayout)(uintptr_t)0x5000, pb = (VkPipelineLayout)(uintptr_t)0x6000;
    VkWriteDescriptorSet w[3];
    #define PUSH(pl) do { VkWriteDescriptorSet w0[3] = { \
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &bi }, \
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 33, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii }, \
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 70, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .pTexelBufferView = &bv } }; \
        memcpy(w, w0, sizeof w); sw_calls = sw_n = 0; push_writes(VK_NULL_HANDLE, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 3, w); } while (0)
    PUSH(pa); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all 3 the first time (%u calls, %u writes)", sw_calls, sw_n);
    PUSH(pa); CHECK(sw_calls == 0, "pushed descriptors: none again when nothing changed (%u calls)", sw_calls);
    ii.imageView = (VkImageView)(uintptr_t)0x3100; PUSH(pa); CHECK(sw_calls == 1 && sw_n == 1 && sw_bind[0] == 33, "pushed descriptors: only the texture that changed (%u writes, binding %u)", sw_n, sw_bind[0]);
    bi.offset = 512; ii.sampler = (VkSampler)(uintptr_t)0x2100; PUSH(pa); CHECK(sw_calls == 1 && sw_n == 2 && sw_bind[0] == 0 && sw_bind[1] == 33, "pushed descriptors: the buffer offset and the sampler that changed (%u writes)", sw_n);
    bi.range = 128; PUSH(pa); CHECK(sw_calls == 1 && sw_n == 1 && sw_bind[0] == 0, "pushed descriptors: a buffer range that changed (%u writes)", sw_n);
    bv = (VkBufferView)(uintptr_t)0x4100; PUSH(pa); CHECK(sw_calls == 1 && sw_n == 1 && sw_bind[0] == 70, "pushed descriptors: a texel buffer view that changed (%u writes)", sw_n);
    PUSH(pb); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all again under another layout (%u writes)", sw_n);
    PUSH(pa); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all again back under the first layout (%u writes)", sw_n);
    sh_desc_forget(); PUSH(pa); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all again after a descriptor set was bound (sh_desc_forget) (%u writes)", sw_n);
    g_gfx_bind_gen++; sh_sync(VK_NULL_HANDLE); PUSH(pa); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all again after other graphics work (%u writes)", sw_n);
    g_dsh = 0; PUSH(pa); PUSH(pa); CHECK(sw_calls == 1 && sw_n == 3, "pushed descriptors: all of them every time with the shadow off (%u writes)", sw_n);
    #undef PUSH
    p_push = keep; g_dsh_cnt = 0;
    // the index buffer
    PFN_vkCmdBindIndexBuffer ki = vkCmdBindIndexBuffer; vkCmdBindIndexBuffer = sw_index; sh_forget(); sh_sync(VK_NULL_HANDLE); si_calls = 0;
    VkBuffer ib = (VkBuffer)(uintptr_t)0x7000;
    sh_index(VK_NULL_HANDLE, 1, 1, ib, 0, VK_INDEX_TYPE_UINT16); sh_index(VK_NULL_HANDLE, 1, 1, ib, 0, VK_INDEX_TYPE_UINT16); CHECK(si_calls == 1, "index buffer: bound once for two draws with the same one (%u)", si_calls);
    sh_index(VK_NULL_HANDLE, 1, 1, ib, 0, VK_INDEX_TYPE_UINT32); sh_index(VK_NULL_HANDLE, 1, 1, ib, 64, VK_INDEX_TYPE_UINT32); CHECK(si_calls == 3 && si_off == 64, "index buffer: bound again for another type and offset (%u)", si_calls);
    sh_index(VK_NULL_HANDLE, 0, 1, ib, 64, VK_INDEX_TYPE_UINT32); CHECK(si_calls == 4, "index buffer: always bound with the shadow off (%u)", si_calls);
    vkCmdBindIndexBuffer = ki; sh_forget();
    g_sh_draws = g_sh_sent = g_sh_skip = g_sh_dw_all = g_sh_dw_sent = g_sh_vb_all = g_sh_vb_sent = 0;
    printf("WORK state shadow: pushed descriptors and the index buffer checked\n");
    // tex_view_for's memo: a texture ID registered again must not answer with the view it had
    if (g_api < VK_API_VERSION_1_1 || !fmt_feat(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) { printf("SIMGAP draw memo: not on Vulkan 1.0 (view_get's view usage structure is Vulkan 1.1's)\n"); return; }
    if (!draw_memo_on()) { printf("SIMGAP draw memo: DM_VK_DRAW_MEMO=0\n"); return; }
    g_step = "draw memo";
    spv_res r; memset(&r, 0, sizeof r); r.kind = SPV_RES_TEX; r.dim = SPV_DIM_2D;
    tex_t t1, t2; memset(&t1, 0, sizeof t1); t1.magic = MAGIC('T', 'E', 'X', 0); t1.d.format = NVN_FORMAT_RGBA8; t1.d.target = 1; t1.d.width = t1.d.height = 4; t1.d.depth = 1; t1.d.levels = 1;
    t2 = t1; t1.serial = nvn_new_serial(); t2.serial = nvn_new_serial(); t2.d.width = t2.d.height = 8;
    int tid = 3950; VkImageView v[4];
    tex_t *seq[4] = { &t1, &t1, &t2, &t1 };   // (the last: t1's image exists already, only the registration changes)
    for (int i = 0; i < 4; i++) { if (i == 0 || seq[i] != seq[i - 1]) mtl_reg_tex(tid, seq[i], NULL);
        pthread_mutex_lock(&g_rmu); v[i] = tex_view_for(tid, &r, 0, NULL, 1, 0); pthread_mutex_unlock(&g_rmu); }
    vtex *x1 = tex_res(&t1), *x2 = tex_res(&t2);
    int in1 = 0, in2 = 0; if (x1) for (int k = 0; k < x1->nviews; k++) in1 |= x1->views[k].v == v[0]; if (x2) for (int k = 0; k < x2->nviews; k++) in2 |= x2->views[k].v == v[2];
    CHECK(v[0] && v[0] == v[1] && in1, "draw memo: the same view for the same registration");
    CHECK(v[2] && v[2] != v[0] && in2, "draw memo: the new texture's view after the ID was registered again");
    CHECK(v[3] == v[0], "draw memo: the first texture's view again after it was registered back");
    mtl_reg_tex(tid, NULL, NULL);
    printf("WORK draw memo: a texture ID registered again checked\n");
}

// ---- R11G11B10F, the format the Mali float-target tests change (DM_VK_B10_AS_RGBA16: drawn as RGBA16F; DM_VK_B10_NOCOMPRESS: its targets without framebuffer
//   compression), through the engine's own paths, judged by the validation layer against the simulated GPU, and checked bit for bit: a render target made and
//   cleared in a pass (the load op, then vkCmdClearAttachments), copied into a pool and back, a pool's every channel code copied in and out, a texture clear, a
//   CPU-filled linear texture uploaded, CPU reads, views in 4-byte formats (the stand-in's refused or made in RGBA16F by use), an image-flagged texture's storage
//   usage against the GPU's R11G11B10F storage support; with the stand-in, an RGBA8 copy into it. Run with and without the switches
static tex_t *b10_tex(pool_t *pl, uint8_t *mem, int fmt, uint32_t w, uint32_t h, int linear) {
    tex_t *t = calloc(1, sizeof *t); t->magic = MAGIC('T', 'E', 'X', 0); t->d.target = NVN_TEXTURE_TARGET_2D; t->d.format = fmt; t->d.width = w; t->d.height = h; t->d.depth = 1; t->d.levels = 1;
    t->d.flags = linear ? NVN_TEXTURE_FLAGS_LINEAR : NVN_TEXTURE_FLAGS_COMPRESSIBLE; if (linear) t->d.stride = w * 4;
    nvn_tex_layout(&t->d, &t->L); t->pool = pl; t->off = mem - pl->mem; t->mem = mem; t->serial = nvn_new_serial(); mtl_tex_init(t);
    return t;
}
static int b10_same(uint32_t a, uint32_t b) {   // packed texels equal, a NaN channel matching any NaN
    for (int c = 0; c < 3; c++) { int mb = c == 2 ? 5 : 6, sh = c == 0 ? 0 : c == 1 ? 11 : 22; uint32_t m = c == 2 ? 0x3ff : 0x7ff, x = a >> sh & m, y = b >> sh & m;
        int nx = (x >> mb) == 31 && (x & ((1u << mb) - 1)), ny = (y >> mb) == 31 && (y & ((1u << mb) - 1)); if (nx || ny ? !(nx && ny) : x != y) return 0; }
    return 1;
}
static void b10_work(void) {
    // the moved-mapping pool route cannot run on lavapipe (work's SIMGAP): there the memory is not made a pool and the copies through one are left out
    int moved = !g_host_import && !g_fdmap && !g_heapmap;
    if (moved) printf("SIMGAP b10: the pool route (moved mappings) cannot run on lavapipe: the copies through a pool left out, the rest runs\n");
    // VK_EXT_image_compression_control listed by the profile but not lavapipe's: not enabled on the device (w_cd), so the engine must not use it here
    if ((g_compctl || p_layout2) && !lvp_has_ext("VK_EXT_image_compression_control")) { g_compctl = 0; p_layout2 = NULL;
        printf("SIMGAP b10: VK_EXT_image_compression_control is the profile's, not lavapipe's: DM_VK_B10_NOCOMPRESS takes its other route here\n"); }
    size_t n = 1u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); CHECK(mem != MAP_FAILED, "b10 mmap"); if (mem == MAP_FAILED) return;
    memset(mem, 0, n); g_step = "b10 pool"; if (!moved) mtl_pool_add(mem, n);
    pool_t *pl = calloc(1, sizeof *pl); pl->magic = MAGIC('P', 'O', 'O', 'L'); pl->flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED; pl->mem = mem; pl->size = n;
    // lavapipe's opaque-fd memory is the pool one page off (work's SIMGAP): there the checks of what the GPU read from or wrote into the pool are left out; the commands
    //   still run (the validation layer judges them)
    int pool_ok = 0; if (!moved) { mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)(mem + 0x8000), 4096, 0xB10B10B1u); uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
        uint32_t w0; memcpy(&w0, mem + 0x8000, 4); pool_ok = w0 == 0xB10B10B1u;
        if (!pool_ok) printf("SIMGAP b10: the GPU does not see this pool where the CPU does (lavapipe's opaque-fd layout): the pool's checks are left out, its commands still run\n"); }
    enum { W = 64 }; uint8_t *bufA = mem + 0x10000, *bufB = mem + 0x20000, *bufC = mem + 0x30000, *lin = mem + 0x40000;
    uint32_t codes[W * W]; for (uint32_t i = 0; i < W * W; i++) codes[i] = (i & 0x7ff) | ((i * 7 + 3) & 0x7ff) << 11 | ((i * 13) & 0x3ff) << 22;   // (every R, G and B code)
    memcpy(bufB, codes, sizeof codes); memcpy(lin, codes, sizeof codes);
    tex_t *T1 = b10_tex(pl, mem + 0x80000, NVN_FORMAT_R11G11B10F, W, W, 0), *T5 = b10_tex(pl, mem + 0x90000, NVN_FORMAT_R11G11B10F, 16, 16, 0);
    tex_t *T6 = b10_tex(pl, lin, NVN_FORMAT_R11G11B10F, W, W, 1), *T3 = b10_tex(pl, mem + 0xA0000, NVN_FORMAT_RGBA8, W, W, 0), *T4 = b10_tex(pl, mem + 0xB0000, NVN_FORMAT_R11G11B10F, W, W, 0);
    // the commands, as hle_nvn.c records them: the target set, a clear (taken by the pass's load op), then a second clear inside the open pass
    static uint64_t cmdbuf[512]; uint8_t *q = (uint8_t *)cmdbuf;
    #define B10OP(op_, T, v) do { op_hdr h_ = { (op_), (uint32_t)((8 + sizeof(T) + 7) & ~7u) }; memcpy(q, &h_, 8); memset(q + 8, 0, h_.size - 8); memcpy(q + 8, &(v), sizeof(T)); q += h_.size; } while (0)
    p_set_rt rt; memset(&rt, 0, sizeof rt); rt.n = 1; rt.color[0] = T1; B10OP(OP_SET_RT, p_set_rt, rt);
    p_clear_color c1 = { 0, 0xf, { 0.25f, -1.0f, 8.0f, 0.0f } }; B10OP(OP_CLEAR_COLOR, p_clear_color, c1);
    op_hdr end = { OP_END, 8 }; memcpy(q, &end, 8);
    uint8_t *second = q + 8; q = second; p_clear_color c2 = { 0, 0xf, { 0.5f, 0.125f, 3.0f, 0.0f } }; B10OP(OP_CLEAR_COLOR, p_clear_color, c2); memcpy(q, &end, 8);
    #undef B10OP
    g_step = "b10 submission"; mtl_sub_begin();
    mtl_exec((const uint8_t *)cmdbuf); CHECK(pass_begin(), "b10: the R11G11B10F target's pass did not begin");
    if (P.open) CHECK((P.b10 & 1) == (b10_on() ? 1u : 0u) && P.cfmt[0] == (uint32_t)(b10_on() ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_B10G11R11_UFLOAT_PACK32),
                      "b10: the pass's target format %u (stand-in mask %#x)", P.cfmt[0], P.b10);
    mtl_exec(second); pass_end();
    p_copy_b2t cp; memset(&cp, 0, sizeof cp); cp.r.width = W; cp.r.height = W; cp.r.depth = 1;
    if (!moved) { cp.tex = T1; cp.src = (uint64_t)(uintptr_t)bufA; copy_b2t(&cp, 1);   // the cleared target into the pool (packed)
                  cp.src = (uint64_t)(uintptr_t)bufB; copy_b2t(&cp, 0);                // every code from the pool into it (expanded)
                  cp.src = (uint64_t)(uintptr_t)bufC; copy_b2t(&cp, 1); }              // and out again
    p_clear_tex ct; memset(&ct, 0, sizeof ct); ct.tex = T5; ct.r.width = 16; ct.r.height = 16; ct.r.depth = 1; ct.mask = 0xf; ct.c.f[0] = -1; ct.c.f[1] = 0.5f; ct.c.f[2] = 3; ct.c.f[3] = 1; clear_tex(&ct);
    if (b10_on()) {   // an RGBA8 texture (cleared) copied into the stand-in: then made what R11G11B10F holds
        p_clear_tex c3; memset(&c3, 0, sizeof c3); c3.tex = T3; c3.r.width = W; c3.r.height = W; c3.r.depth = 1; c3.mask = 0xf; c3.c.f[0] = 1; c3.c.f[1] = 0; c3.c.f[2] = 1; c3.c.f[3] = 1; clear_tex(&c3);
        p_copy_t2t tt; memset(&tt, 0, sizeof tt); tt.src = T3; tt.dst = T4; tt.sr.width = tt.dr.width = W; tt.sr.height = tt.dr.height = W; tt.sr.depth = tt.dr.depth = 1; copy_t2t(&tt); }
    uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
    // a view of the target in a 4-byte format (on the Switch its packed bits): R11G11B10F takes it (viewed in its own format where its formats are fixed); the stand-in has
    //   no packed bits: a float view to sample is made in RGBA16F (it reads the colour), an integer view, a storage view or a render target is not made (b10_view_fmt)
    { vtex *x = tex_res(T1); uint8_t sw[4] = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
      VkImageView vu = x ? view_get(x, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_UINT, 0, 1, 0, 1, sw, VK_IMAGE_ASPECT_COLOR_BIT) : VK_NULL_HANDLE;
      VkImageView vf = x ? view_get(x, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_SFLOAT, 0, 1, 0, 1, sw, VK_IMAGE_ASPECT_COLOR_BIT) : VK_NULL_HANDLE;
      if (b10_on()) {
          CHECK(x && !vu && vf, "b10: the stand-in's 4-byte views: an integer one not made (%p), a float one made in RGBA16F (%p)", (void *)vu, (void *)vf);
          CHECK(x && b10_view_fmt(x, VK_FORMAT_R32_UINT, B10V_SAMPLED, "t_vkdev") == VK_FORMAT_UNDEFINED && b10_view_fmt(x, VK_FORMAT_R32_SFLOAT, B10V_SAMPLED, "t_vkdev") == VK_FORMAT_R16G16B16A16_SFLOAT
                && b10_view_fmt(x, VK_FORMAT_R32_SFLOAT, B10V_STORAGE, "t_vkdev") == VK_FORMAT_UNDEFINED && b10_view_fmt(x, VK_FORMAT_R8G8B8A8_UNORM, B10V_ATTACH, "t_vkdev") == VK_FORMAT_UNDEFINED
                && b10_view_fmt(x, VK_FORMAT_R16G16B16A16_SFLOAT, B10V_ATTACH, "t_vkdev") == VK_FORMAT_R16G16B16A16_SFLOAT, "b10: the stand-in's views by use (b10_view_fmt)");
      } else CHECK(x && vu && vf && b10_view_fmt(x, VK_FORMAT_R32_UINT, B10V_STORAGE, "t_vkdev") == VK_FORMAT_R32_UINT, "b10: views of the target in 4-byte formats (%p, %p)", (void *)vu, (void *)vf); }
    // a texture the game may bind as a storage image: storage usage exactly where R11G11B10F itself has storage on this GPU, the stand-in too (it must not change the
    //   compression a storage image may cost)
    { tex_t *T7 = b10_tex(pl, mem + 0xC0000, NVN_FORMAT_R11G11B10F, 16, 16, 0); T7->d.flags |= NVN_TEXTURE_FLAGS_IMAGE; vtex *x = tex_res(T7);
      VkFormatProperties bp; vkGetPhysicalDeviceFormatProperties(g_pd, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &bp); int want = !!(bp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
      CHECK(x && !!(x->usage & VK_IMAGE_USAGE_STORAGE_BIT) == want, "b10: an image-flagged R11G11B10F texture%s: storage usage %d, R11G11B10F storage on this GPU %d",
            b10_on() ? " (the stand-in)" : "", x ? !!(x->usage & VK_IMAGE_USAGE_STORAGE_BIT) : -1, want);
      printf("WORK b10: an image-flagged R11G11B10F texture: storage usage %s (R11G11B10F storage %s on this GPU)\n", x && (x->usage & VK_IMAGE_USAGE_STORAGE_BIT) ? "yes" : "no", want ? "supported" : "not supported"); }
    // the checks: the clear (as R11G11B10F stores it), every code through the pool, the texture clear, the upload, the RGBA8 copy
    uint32_t qc = b10_float_to_chan(0.5f, B10_MR) | b10_float_to_chan(0.125f, B10_MR) << 11 | b10_float_to_chan(3.0f, B10_MB) << 22, q5 = b10_float_to_chan(-1, B10_MR) | b10_float_to_chan(0.5f, B10_MR) << 11 | b10_float_to_chan(3, B10_MB) << 22;
    int bad[6] = { 0 }; uint32_t got[W * W]; NVNcopyRegion r = { 0, 0, 0, W, W, 1 }, r16 = { 0, 0, 0, 16, 16, 1 };
    for (int i = 0; i < W * W; i++) { uint32_t a, cc; memcpy(&a, bufA + 4 * i, 4); memcpy(&cc, bufC + 4 * i, 4); bad[0] += !b10_same(a, qc); bad[1] += !b10_same(cc, codes[i]); }
    mtl_read_texels(T1, 0, 0, &r, (uint8_t *)got, W * 4, W * W * 4); for (int i = 0; i < W * W; i++) bad[2] += !b10_same(got[i], codes[i]);
    mtl_read_texels(T5, 0, 0, &r16, (uint8_t *)got, 16 * 4, 16 * 16 * 4); for (int i = 0; i < 16 * 16; i++) bad[3] += !b10_same(got[i], q5);
    mtl_read_texels(T6, 0, 0, &r, (uint8_t *)got, W * 4, W * W * 4); for (int i = 0; i < W * W; i++) bad[4] += !b10_same(got[i], codes[i]);
    if (b10_on()) { uint32_t m = b10_float_to_chan(1, B10_MR) | b10_float_to_chan(1, B10_MB) << 22; mtl_read_texels(T4, 0, 0, &r, (uint8_t *)got, W * 4, W * W * 4); for (int i = 0; i < W * W; i++) bad[5] += !b10_same(got[i], m); }
    if (!pool_ok) bad[0] = bad[1] = bad[2] = 0;   // (the SIMGAP above: what the GPU read from the pool and wrote into it is somewhere else)
    CHECK(!bad[0], "b10: the cleared target copied into the pool: %d of %d texels not (0.5, 0.125, 3)", bad[0], W * W);
    CHECK(!bad[1], "b10: every channel code through the target and back into the pool: %d of %d texels differ", bad[1], W * W);
    CHECK(!bad[2], "b10: the CPU's read of the target holding every code: %d of %d texels differ", bad[2], W * W);
    CHECK(!bad[3], "b10: the texture clear (-1, 0.5, 3) read by the CPU: %d of 256 texels differ", bad[3]);
    CHECK(!bad[4], "b10: the CPU-filled linear texture uploaded and read back: %d of %d texels differ", bad[4], W * W);
    CHECK(!bad[5], "b10: the RGBA8 texture copied into the stand-in: %d of %d texels not (1, 0, 1)", bad[5], W * W);
    printf("WORK b10: R11G11B10F %s%s: clears, copies through a pool, upload, CPU reads%s: %s%s\n", b10_on() ? "as its RGBA16F stand-in" : "as itself", g_b10nc ? (g_compctl ? ", targets with compression disabled" : ", targets mutable with no format list") : "",
           b10_on() ? ", an RGBA8 copy into it" : "", bad[0] + bad[1] + bad[2] + bad[3] + bad[4] + bad[5] ? "WRONG" : "exact", pool_ok ? "" : " (the pool's checks left out: SIMGAP)");
}

// ---- the Mali white-screen check (mali_white_vk.inc): whom its ladder runs for and the route it gives (route A by default on Arm's own driver with Sword / Shield
//   only; env.txt's own settings win), whether its verdicts are kept (white ones; fine ones only with no other white-screen test) and the start marker is up, then
//   its compute pass, forced on here on every GPU, on a lit-size R11G11B10F target (or its RGBA16F stand-in) cleared to 65504, +infinity, NaN (counted apart from
//   saturated), a finite white under the old threshold (2000), the lower bucket (100), a normal colour, black and near black, and with an RGBA8 picture shown
//   cleared to white, almost white and grey, each sample read back after its fence and counted; the validation layer judges the pass against the simulated GPU
static void mw_work(void) {
    const char *ea = getenv("DM_VK_B10_AS_RGBA16"), *eb = getenv("DM_VK_B10_NOCOMPRESS"), *le = getenv("DM_MALI_LADDER"), *se = getenv("DM_MALI_WHITE_SIM");
    int arm = g_arm_rel != 0, armd = arm && MALI_LIT_DEFAULT, en = mw_enabled(armd, DM_PORT_GALAR, le, se), a = 0, b = 0;   // (armd: v0.96 has the fix off by default, vk_quirks.h)
    int route = mw_route_apply(en ? g_mw.want : mw_default_route(armd, DM_PORT_GALAR), ea, eb, &a, &b);
    char ot[200]; int others = mw_other_tests(ot, sizeof ot), record = route == MW_ROUTE_A || route == MW_ROUTE_B || route == MW_ROUTE_BOTH, record_fine = record && !others;
    const char *pr = getenv("DM_VK_PROBE"); int f16 = pr && strstr(pr, "f16target") != NULL;   // (DM_VK_PROBE=f16target makes the stand-in too: b10_on)
    DEC("mali_ladder", "%d", g_mw.on); DEC("mali_route", "%s", mw_route_name(g_mw.route)); DEC("mali_check", "%d", g_mw.check); DEC("mali_record", "%d", g_mw.record);
    DEC("mali_record_fine", "%d", g_mw.record_fine); DEC("mali_marker", "%d", g_mw.marker); DEC("mali_stage", "%s", mw_stage_name(g_mw.st.stage)); DEC("mali_galar", "%d", DM_PORT_GALAR);
    CHECK(g_mw.on == (en != 0), "white-screen ladder on %d, expected %d (Arm's own driver %d, Sword / Shield %d)", g_mw.on, en != 0, arm, DM_PORT_GALAR);
    CHECK(b10_on() == (a || f16) && g_b10nc == b, "white-screen route: stand-in %d (expected %d), no compression %d (expected %d)", b10_on(), a || f16, g_b10nc, b);
    if (!(le && *le) && !(se && *se) && !(ea && *ea) && !(eb && *eb)) CHECK(b10_on() == ((armd && DM_PORT_GALAR) || f16) && !g_b10nc && g_mw.on == (armd && DM_PORT_GALAR),
        "route A and the ladder by default exactly on Arm's own driver with Sword / Shield (Arm %d): stand-in %d, ladder %d", arm, b10_on(), g_mw.on);
    if (en && !getenv("DM_MALI_LADDER_FILE")) {   // (no state file here: the ladder's first start)
        CHECK(g_mw.check == MW_CHECK_FULL, "the ladder's first start checks the lit scene while playing (check %d)", g_mw.check);
        CHECK(g_mw.record == record && g_mw.record_fine == record_fine, "verdicts kept: white %d, fine %d, expected %d, %d (route %s, other white-screen tests: %s)", g_mw.record, g_mw.record_fine,
              record, record_fine, mw_route_name(route), others ? ot : "none");
        CHECK(g_mw.marker == (record_fine && route == MW_ROUTE_A), "the start marker %d, expected %d (it marks the ladder's own route A, as the ladder asked, with no other test)", g_mw.marker,
              record_fine && route == MW_ROUTE_A);
        CHECK(!g_mw.marker || g_mw.st.stage == MW_STAGE_LOADING, "the start marker's stage after mtl_init: %s (loading: the check's compute pass made, no picture yet)", mw_stage_name(g_mw.st.stage));
    }
    g_step = "mali check: compute pass"; CHECK(mw_gpu_init(), "the white-screen check's compute pass made"); if (g_mw.gpu <= 0) return;
    enum { LW = 1280, LH = 720 };
    size_t n = 8u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); CHECK(mem != MAP_FAILED, "mali check mmap"); if (mem == MAP_FAILED) return;
    pool_t *pl = calloc(1, sizeof *pl); pl->magic = MAGIC('P', 'O', 'O', 'L'); pl->flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED; pl->mem = mem; pl->size = n;
    tex_t *T = b10_tex(pl, mem, NVN_FORMAT_R11G11B10F, LW, LH, 0);
    // expected: saturated (a channel at 1024+, or +inf), at 64+ (or saturated), near black (every channel under 1/1024), infinite, NaN (counted apart)
    static const struct { const char *what; float v[3]; uint32_t sat, low, dark, inf, nan; int32_t r[4]; } cs[] = {
        { "65504 (white)", { 65504.0f, 65504.0f, 65504.0f }, MW_TEX, MW_TEX, 0, 0, 0, { 0, 0, LW, LH } },
        { "65504, a part at an offset", { 65504.0f, 65504.0f, 65504.0f }, MW_TEX, MW_TEX, 0, 0, 0, { 100, 50, 640, 360 } },
        { "+infinity", { INFINITY, INFINITY, INFINITY }, MW_TEX, MW_TEX, 0, MW_TEX, 0, { 0, 0, LW, LH } },
        { "NaN", { NAN, NAN, NAN }, 0, 0, 0, 0, MW_TEX, { 0, 0, LW, LH } },
        { "2000 (a finite white the old 60000 threshold missed; Arm left 21250 in #110)", { 2000.0f, 2000.0f, 2000.0f }, MW_TEX, MW_TEX, 0, 0, 0, { 0, 0, LW, LH } },
        { "100 (the lower bucket only)", { 100.0f, 1.0f, 0.5f }, 0, MW_TEX, 0, 0, 0, { 0, 0, LW, LH } },
        { "a normal colour (0.5, 0.25, 0.125)", { 0.5f, 0.25f, 0.125f }, 0, 0, 0, 0, 0, { 0, 0, LW, LH } },
        { "black (0, 0, 0: a cleared target)", { 0.0f, 0.0f, 0.0f }, 0, 0, MW_TEX, 0, 0, { 0, 0, LW, LH } },
        { "near black (0.0005, 0.0002, 0)", { 0.0005f, 0.0002f, 0.0f }, 0, 0, MW_TEX, 0, 0, { 0, 0, LW, LH } },
        { "dark but not black (0.002, 0, 0)", { 0.002f, 0.0f, 0.0f }, 0, 0, 0, 0, 0, { 0, 0, LW, LH } } };
    int bad = 0;
    for (unsigned i = 0; i < sizeof cs / sizeof *cs; i++) {
        g_step = "mali check: clear"; mtl_sub_begin();
        p_clear_tex ct; memset(&ct, 0, sizeof ct); ct.tex = T; ct.r.width = LW; ct.r.height = LH; ct.r.depth = 1; ct.mask = 0xf; for (int k = 0; k < 3; k++) ct.c.f[k] = cs[i].v[k]; ct.c.f[3] = 1; clear_tex(&ct);
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
        vtex *x = tex_res(T); CHECK(x && x->img && (x->usage & VK_IMAGE_USAGE_SAMPLED_BIT), "mali check: the target made, sampled"); if (!x || !x->img) return;
        g_step = "mali check: sample"; int ok = mw_sample(x, cs[i].r, 0, 0, 0, 100, NULL, NULL); CHECK(ok, "mali check: the sample of %s submitted", cs[i].what); if (!ok) { bad++; continue; }
        int k = 0; for (int s = 1; s < MW_SLOTS; s++) if (g_mw.slot[s].busy && g_mw.slot[s].seq > g_mw.slot[k].seq) k = s;
        CHECK(vkWaitForFences(g_dev, 1, &g_mw.slot[k].f, VK_TRUE, 10000000000ull) == VK_SUCCESS, "mali check: the sample's fence");
        mw_count c = mw_read_slot(k); vtex *sx = g_mw.slot[k].x; g_mw.slot[k].busy = 0; g_mw.slot[k].x = NULL; vtex_unref(sx);
        int right = c.sat == cs[i].sat && c.low == cs[i].low && c.dark == cs[i].dark && c.inf == cs[i].inf && c.nan == cs[i].nan && !c.shown; bad += !right;
        CHECK(right, "mali check, target %s cleared to %s, part %d,%d %dx%d: %u of %d saturated, %u at 64+, %u near black (%u infinite, %u NaN; largest finite %g; picture shown %u), expected %u, %u, %u (%u, %u; none)",
              x->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? "RGBA16F (stand-in)" : "R11G11B10F", cs[i].what, cs[i].r[0], cs[i].r[1], cs[i].r[2], cs[i].r[3], c.sat, MW_TEX, c.low, c.dark, c.inf, c.nan,
              (double)c.mx, c.shown, cs[i].sat, cs[i].low, cs[i].dark, cs[i].inf, cs[i].nan);
    }
    // the picture shown (an RGBA8 window texture, 320x240, its crop the whole of it): white where every colour channel is at 0.99 or more (252.45 of 255)
    tex_t *S = b10_tex(pl, mem + (4u << 20), NVN_FORMAT_RGBA8, 320, 240, 0);
    static const struct { const char *what; float v[3]; uint32_t white; } ss[] = { { "white (1, 1, 1)", { 1, 1, 1 }, MW_TEX }, { "almost white (1, 1, 0.98: 250 of 255)", { 1, 1, 0.98f }, 0 },
                                                                                    { "grey (0.5)", { 0.5f, 0.5f, 0.5f }, 0 } };
    int sbad = 0;
    for (unsigned i = 0; i < sizeof ss / sizeof *ss; i++) {
        g_step = "mali check: shown clear"; mtl_sub_begin();
        p_clear_tex ct; memset(&ct, 0, sizeof ct); ct.tex = S; ct.r.width = 320; ct.r.height = 240; ct.r.depth = 1; ct.mask = 0xf; for (int k = 0; k < 3; k++) ct.c.f[k] = ss[i].v[k]; ct.c.f[3] = 1; clear_tex(&ct);
        ct.tex = T; ct.r.width = LW; ct.r.height = LH; for (int k = 0; k < 3; k++) ct.c.f[k] = 0.5f; clear_tex(&ct);
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
        vtex *x = tex_res(T), *sx = tex_res(S); int sok = mw_shown_ok(sx);
        CHECK(sok, "mali check: the RGBA8 picture shown is one the check reads (format %d)", sx ? (int)sx->fmt : -1); if (!sok) { sbad++; continue; }
        int32_t r[4] = { 0, 0, LW, LH }, sr[4] = { 0, 0, 320, 240 };
        g_step = "mali check: shown sample"; int ok = mw_sample(x, r, 0, 0, 0, 100, sx, sr); CHECK(ok, "mali check: the sample with the picture shown %s submitted", ss[i].what); if (!ok) { sbad++; continue; }
        int k = 0; for (int s = 1; s < MW_SLOTS; s++) if (g_mw.slot[s].busy && g_mw.slot[s].seq > g_mw.slot[k].seq) k = s;
        CHECK(vkWaitForFences(g_dev, 1, &g_mw.slot[k].f, VK_TRUE, 10000000000ull) == VK_SUCCESS, "mali check: the shown sample's fence");
        mw_count c = mw_read_slot(k); vtex *lx = g_mw.slot[k].x, *px = g_mw.slot[k].sx; g_mw.slot[k].busy = 0; g_mw.slot[k].x = g_mw.slot[k].sx = NULL; vtex_unref(lx); if (px) vtex_unref(px);
        int right = c.shown == MW_TEX && c.swhite == ss[i].white && c.sat == 0 && c.nan == 0; sbad += !right;
        CHECK(right, "mali check, the picture shown cleared to %s: %u of %u white (lit target %u saturated), expected %u of %d", ss[i].what, c.swhite, c.shown, c.sat, ss[i].white, MW_TEX);
    }
    printf("WORK mali check: the lit target as %s, cleared and sampled %d times (65504, a part of it, +inf, NaN, 2000, 100, a normal colour, black, near black, dark): %s; "
           "the picture shown (RGBA8) %d times (white, almost white, grey): %s\n", b10_on() ? "its RGBA16F stand-in" : "R11G11B10F", (int)(sizeof cs / sizeof *cs), bad ? "WRONG" : "counted right",
           (int)(sizeof ss / sizeof *ss), sbad ? "WRONG" : "counted right");
}

// ---- uniform snapshots reused within a submission (DM_VK_UBO_SNAPSHOT with DM_VK_UBO_SNAPSHOT_REUSE, nvn_vk.c snapc_*): a frame of 1,000 draws (the field scene of
//   poco-runs b10 had 997) in submissions of 150 (DM_VK_SPLIT), each draw binding a scene and a light block every draw shares, one of 40 material blocks and its own
//   object block (written by the game before the submission, as it does before nvnQueueSubmitCommands), with a uniform update into the scene block every 50 draws and a
//   GPU fill into a material block every 75, recorded through push_stage_h with the snapshots forced on, reuse off (the copy per draw) and on. Every copy a draw binds
//   must hold what the game's memory held when that draw was recorded, and every update must make the next draw's scene block a new copy. A model of the field's
//   bindings, not the game's own stream: the copies, bytes and arena blocks per frame of both are printed
static const uint8_t *snap_map(VkBuffer b, VkDeviceSize o) { for (int i = 0; i < g_sub->nar; i++) if (g_sub->ar[i].buf == b) return g_sub->ar[i].map + o; return NULL; }
typedef struct { uint64_t copies, reused, bytes, arena, used, upd, upd_new; double ms; int bad; } snap_res;
static snap_res snap_frame(uint8_t *mem, vshader *vs, int reuse) {
    enum { DRAWS = 1000, PER_SUB = 150, NMAT = 40 };
    snap_res r; memset(&r, 0, sizeof r);
    int keep_snap = g_ubo_snap, keep_reuse = g_snapc_on, keep_np = g_np; g_ubo_snap = 1; g_snapc_on = reuse;
    uint64_t c0 = g_snap_copies, r0 = g_snap_reused, b0 = g_snap_bytes, base = (uint64_t)(uintptr_t)mem;
    uint64_t scene = base + 0x10000, light = base + 0x20000, mat = base + 0x40000, obj = base + 0x100000;   // (materials 4 KB apart, objects 256 bytes apart)
    VkBuffer last_b = VK_NULL_HANDLE; VkDeviceSize last_o = 0; int updated = 0;
    for (int d = 0; d < DRAWS; d++) {
        if (d % PER_SUB == 0) {
            if (d) { for (int i = 0; i < g_sub->nar; i++) { r.arena += g_sub->ar[i].size; r.used += g_sub->ar[i].used; } uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); }
            for (int k = d; k < d + PER_SUB && k < DRAWS; k++) for (int q = 0; q < 64; q++) ((uint32_t *)(uintptr_t)(obj + (uint64_t)k * 0x100))[q] = 0x0B000000u + (uint32_t)k * 64 + (uint32_t)q;
            mtl_sub_begin(); last_b = VK_NULL_HANDLE; }
        if (d % 50 == 25) { uint32_t v[64]; for (int q = 0; q < 64; q++) v[q] = 0x5CE00000u + (uint32_t)d * 64 + (uint32_t)q; sub_update(v, scene + 0x40, sizeof v); updated = 1; r.upd++; }
        if (d % 75 == 40) mtl_sub_fill(mat + (uint64_t)(d % NMAT) * 0x1000, 256, 0xF1110000u + (uint32_t)d);
        S.ubo[0][0] = (p_bind_buf){ 0, 0, scene, 0x400 }; S.ubo[0][1] = (p_bind_buf){ 0, 1, light, 0x800 };
        S.ubo[0][2] = (p_bind_buf){ 0, 2, mat + (uint64_t)(d * 7 % NMAT) * 0x1000, 0x100 }; S.ubo[0][3] = (p_bind_buf){ 0, 3, obj + (uint64_t)d * 0x100, 0x100 };
        D.n = 0; g_np = 1; uint64_t t0 = mono_ns();
        push_stage_h(vs, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, VK_NULL_HANDLE, 0);
        r.ms += (mono_ns() - t0) / 1e6; g_np = keep_np;
        for (int i = 0; i < 4; i++) { const VkDescriptorBufferInfo *bi = &D.bi[i]; const uint8_t *m = snap_map(bi->buffer, bi->offset);
            if (!m || memcmp(m, (const void *)(uintptr_t)S.ubo[0][i].addr, (size_t)bi->range)) { if (!r.bad) printf("snapshots: draw %d binding %d: %s\n", d, i, m ? "the copy differs from the game's memory" : "not in the arena"); r.bad++; } }
        if (updated && last_b) { r.upd_new += D.bi[0].buffer != last_b || D.bi[0].offset != last_o; } updated = 0;
        last_b = D.bi[0].buffer; last_o = D.bi[0].offset; D.n = 0;
    }
    for (int i = 0; i < g_sub->nar; i++) { r.arena += g_sub->ar[i].size; r.used += g_sub->ar[i].used; }
    uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
    r.copies = g_snap_copies - c0; r.reused = g_snap_reused - r0; r.bytes = g_snap_bytes - b0;
    g_ubo_snap = keep_snap; g_snapc_on = keep_reuse;
    return r;
}
static void snap_work(void) {
    if (!g_host_import && !g_fdmap && !g_heapmap) { printf("SIMGAP snapshots: the moved-mapping pool route cannot run on lavapipe (see the pool work above)\n"); return; }
    size_t n = 4u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); CHECK(mem != MAP_FAILED, "snapshots mmap"); if (mem == MAP_FAILED) return;
    for (size_t i = 0; i < n / 4; i++) ((uint32_t *)mem)[i] = (uint32_t)i * 2654435761u;
    mtl_pool_add(mem, n);
    vshader *vs = calloc(1, sizeof *vs); snprintf(vs->key, sizeof vs->key, "snaptest"); vs->nres = 4;
    for (int i = 0; i < 4; i++) { vs->res[i].kind = SPV_RES_UBO; vs->res[i].binding = (uint32_t)i; vs->res[i].count = 1; snprintf(vs->res[i].name, sizeof vs->res[i].name, "fp_c%d", i); vs->how[i].what = W_CBUF; vs->how[i].reg = (uint16_t)i; }
    snap_res a = snap_frame(mem, vs, 0), b = snap_frame(mem, vs, 1);   // (twice each, the second pair reported: the first touches the arenas' new pages)
    { snap_res a2 = snap_frame(mem, vs, 0), b2 = snap_frame(mem, vs, 1); a2.bad += a.bad; b2.bad += b.bad; a = a2; b = b2; }
    {   // a mirrored pool written back by the completion thread (wb_apply) between two draws: the next draw copies again
        int keep_snap = g_ubo_snap, keep_reuse = g_snapc_on, keep_np = g_np; g_ubo_snap = 1; g_snapc_on = 1; uint64_t c0 = g_snap_copies; mtl_sub_begin();
        for (int k = 0; k < 3; k++) { if (k == 2) atomic_fetch_add(&g_snapc_wbgen, 1); D.n = 0; g_np = 1; push_stage_h(vs, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, VK_NULL_HANDLE, 0); g_np = keep_np; D.n = 0; }
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); g_ubo_snap = keep_snap; g_snapc_on = keep_reuse;
        CHECK(g_snap_copies - c0 == 8, "snapshots reused: a write-back by the completion thread drops every copy (%llu copies for 3 draws of 4 blocks, 8 expected)", (unsigned long long)(g_snap_copies - c0)); }
    CHECK(!a.bad && !b.bad, "snapshots: every bound copy holds the game's memory as at its draw (a copy per draw: %d wrong, reused: %d wrong)", a.bad, b.bad);
    CHECK(a.copies == 4000 && !a.reused, "snapshots, a copy per draw: %llu copies, %llu reused (4,000 and 0 expected)", (unsigned long long)a.copies, (unsigned long long)a.reused);
    CHECK(b.copies + b.reused == 4000 && b.copies < a.copies / 2, "snapshots reused: %llu copies + %llu reused (of 4,000)", (unsigned long long)b.copies, (unsigned long long)b.reused);
    CHECK(b.upd && b.upd_new == b.upd, "snapshots reused: each uniform update gave the next draw a new copy of the scene block (%llu of %llu)", (unsigned long long)b.upd_new, (unsigned long long)b.upd);
    printf("WORK uniform snapshots, 1,000 draws x 4 blocks a frame: a copy per draw %llu copies, %.1f MB copied, arenas %.1f MB (%.1f MB used), %.2f ms; reused %llu copies + %llu bound again, "
           "%.1f MB copied, arenas %.1f MB (%.1f MB used), %.2f ms\n", (unsigned long long)a.copies, a.bytes / 1048576.0, a.arena / 1048576.0, a.used / 1048576.0, a.ms,
           (unsigned long long)b.copies, (unsigned long long)b.reused, b.bytes / 1048576.0, b.arena / 1048576.0, b.used / 1048576.0, b.ms);
}
// ---- perf-1124 fixes 2 and 8 ----
static VkImageView tv_view(vtex *x) {
    VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = x->img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = x->fmt, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    VkImageView v = VK_NULL_HANDLE; return vkCreateImageView(g_dev, &vi, NULL, &v) ? VK_NULL_HANDLE : v;
}
static uint8_t *tv_readback(vtex *x, VkImageLayout lay, VkBuffer *hb, vmalloc_t *hm, VkCommandBuffer cb) {   // (cb open: the copy recorded there; the caller ends it)
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)x->w * x->h * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    VKCHECK(vkCreateBuffer(g_dev, &bci, NULL, hb)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, *hb, &rq);
    *hm = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1); VKCHECK(vkBindBufferMemory(g_dev, *hb, hm->b->mem, hm->off));
    VkBufferImageCopy bc = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { x->w, x->h, 1 } };
    vkCmdCopyImageToBuffer(cb, x->img, lay, *hb, 1, &bc);
    return (uint8_t *)hm->b->map + hm->off;
}
// Fix 2, a held frame's render areas (seg_area): a 1280x720 target and a frame of 8 game passes, each writing only part of what render scale 0.5 leaves (640x360 at the
//   top left, as Sword / Shield scale the viewport and scissor inside full-size targets), separated by the pass-to-pass barrier pass_begin records (so on Adreno's
//   profiles the passes run as one, seg_record_prim). Each pass's writes are a vkCmdClearAttachments over its scissor, noted as clear_in_pass notes it. Read back: the
//   part written has the passes' colour, everything else the colour from before (Vulkan's load / store outside the render area keep it); the validation layer checks every
//   clear lies inside its pass's area. MEASURE: target pixels and render-area pixels for the frame (held frames only: DM_VK_FRAME_HOLD=2; DM_VK_RENDER_AREA=0 compares)
static void ra_work(void) {
    enum { TW = 1280, TH = 720, NP = 8 };
    vtex *t = zt_image(VK_FORMAT_R8G8B8A8_UNORM, TW, TH, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageView v = t ? tv_view(t) : VK_NULL_HANDLE;
    if (!v) { printf("SIMGAP render area: no RGBA8 target\n"); return; }
    img_init(t, 0);
    { VkCommandBuffer cb = once_begin(); VkClearColorValue bg = {{ 0.2f, 0.4f, 0.6f, 1 }}; VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdClearColorImage(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, &bg, 1, &rr); once_end(cb); }
    VkRect2D rc[NP]; for (int p = 0; p < NP; p++) rc[p] = (VkRect2D){ { (p % 4) * 60, (p / 4) * 60 }, { 460, 280 } };   // (their union: 640x340)
    uint64_t f0 = g_ra_px_full, a0 = g_ra_px, n0 = g_ra_n, m0 = g_merge_n;
    g_step = "render area: the frame";
    mtl_sub_begin(); int held = g_sub->dec;
    for (int p = 0; p < NP; p++) {
        if (p) { barrier_pass(CB()); if (CB() == g_gap_cb && g_gap_work) g_gap_work--; }   // (as pass_begin: only the order between passes)
        VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
        VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { { 0, 0 }, { TW, TH } }, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
        uint32_t cf = VK_FORMAT_R8G8B8A8_UNORM; g_ra_next = 1; rend_begin(CB(), &ri, &cf, 0, 1, 0);
        VkClearAttachment a = { VK_IMAGE_ASPECT_COLOR_BIT, 0, { .color = {{ 1.0f, 0.0f, 0.0f, 1.0f }} } }; VkClearRect cr = { rc[p], 0, 1 };
        seg_area_note(CB(), rc[p]); vkCmdClearAttachments(CB(), 1, &a, 1, &cr);
        rend_end(CB());
    }
    barrier_all(CB());
    uint64_t fence = mtl_sub_commit(); mtl_wait(fence, UINT64_MAX); mtl_finish();
    g_step = "render area: read back";
    VkBuffer hb; vmalloc_t hm; VkCommandBuffer cb = once_begin(); const uint8_t *px = tv_readback(t, VK_IMAGE_LAYOUT_GENERAL, &hb, &hm, cb); once_end(cb);
    int bad_in = 0, bad_out = 0, fx = -1, fy = -1;
    for (int y = 0; y < TH; y++) for (int x = 0; x < TW; x++) { int in = 0;
        for (int p = 0; p < NP && !in; p++) in = x >= rc[p].offset.x && y >= rc[p].offset.y && x < rc[p].offset.x + (int)rc[p].extent.width && y < rc[p].offset.y + (int)rc[p].extent.height;
        const uint8_t *q = px + ((size_t)y * TW + x) * 4;
        int ok = in ? q[0] == 255 && q[1] == 0 && q[2] == 0 : abs(q[0] - 51) <= 1 && abs(q[1] - 102) <= 1 && abs(q[2] - 153) <= 1;
        if (!ok) { if (in) bad_in++; else bad_out++; if (fx < 0) { fx = x; fy = y; } } }
    CHECK(!bad_in && !bad_out, "render area (%s, %s): %d pixels written wrong, %d outside the writes changed (first at %d,%d)", held ? "held" : "as recorded", g_dynren ? "dynamic rendering" : "render passes", bad_in, bad_out, fx, fy);
    uint64_t full = g_ra_px_full - f0, area = g_ra_px - a0, nsm = g_ra_n - n0, merged = g_merge_n - m0;
    if (held && ra_on()) CHECK(nsm > 0 && area < full, "render area: the held frame's passes began on part of the target (%llu of %llu pixels)", (unsigned long long)area, (unsigned long long)full);
    if (held && !ra_on()) CHECK(area == full, "render area off: the whole target (%llu of %llu pixels)", (unsigned long long)area, (unsigned long long)full);
    printf("MEASURE render area at scale 0.5 (%s, %s, DM_VK_RENDER_AREA %s): %d game passes on a %dx%d target, %llu begun (%llu run inside the one before); target pixels %llu, render-area pixels %llu (%.1f %%)\n",
           held ? "held" : "as recorded", g_dynren ? "dynamic rendering" : "render passes", ra_on() ? "on" : "off", NP, TW, TH, (unsigned long long)(held ? NP - merged : NP), (unsigned long long)merged,
           (unsigned long long)(held ? full : (uint64_t)NP * TW * TH), (unsigned long long)(held ? area : (uint64_t)NP * TW * TH), held && full ? 100.0 * (double)area / (double)full : 100.0);
}
// Fix 8, the present composed in one pass: a 160x90 picture into a 400x240 stand-in for a swapchain image (the picture's box 400x225, black above and below), by the
//   sharp upscaler, the plain stretch, FSR 1 and SGSR, composed both ways (g_compose_one 0: clear, upscaler pass or blit, as before; 1: one pass). Read back: the upscalers'
//   pictures the same both ways, the plain stretch within a step or two (the blit's filter against the shader's bilinear), black outside the box. MEASURE: passes
//   and whole-image clears on the swapchain image per picture
static void compose_work(void) {
    enum { SW = 400, SH = 240, PW = 160, PH = 90 };
    if (!g_has_swapchain) { printf("SIMGAP compose: no VK_KHR_swapchain (the present layout)\n"); return; }
    vtex *d = zt_image(VK_FORMAT_R8G8B8A8_UNORM, SW, SH, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
         *s = zt_image(VK_FORMAT_R8G8B8A8_UNORM, PW, PH, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageView dv = d ? tv_view(d) : VK_NULL_HANDLE, sv = s ? tv_view(s) : VK_NULL_HANDLE;
    if (!dv || !sv) { printf("SIMGAP compose: images not made\n"); return; }
    img_init(s, 0);
    {   // the picture: a gradient with a checker, through a host buffer
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PW * PH * 4, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
        VkBuffer b; VKCHECK(vkCreateBuffer(g_dev, &bci, NULL, &b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, b, &rq);
        vmalloc_t m = vk_alloc(rq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1); VKCHECK(vkBindBufferMemory(g_dev, b, m.b->mem, m.off));
        uint8_t *p = (uint8_t *)m.b->map + m.off;
        for (int y = 0; y < PH; y++) for (int x = 0; x < PW; x++) { uint8_t *q = p + (y * PW + x) * 4; q[0] = (uint8_t)(x * 255 / (PW - 1)); q[1] = (uint8_t)(y * 255 / (PH - 1)); q[2] = ((x / 8 + y / 8) & 1) ? 200 : 40; q[3] = 255; }
        VkCommandBuffer cb = once_begin(); VkBufferImageCopy bc = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { PW, PH, 1 } };
        vkCmdCopyBufferToImage(cb, b, s->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bc); once_end(cb); }
    W.n = 1; W.img[0] = d->img; W.view[0] = dv; W.fmt = VK_FORMAT_R8G8B8A8_UNORM; W.ext = W.lext = (VkExtent2D){ SW, SH }; W.rot = 0;
    static uint8_t out[4][2][SW * SH * 4]; int done[4][2] = { { 0 } };
    const int modes[4] = { 0, 2, 3, 1 }; static const char *const mname[4] = { "sharp upscaler", "plain stretch", "FSR 1", "SGSR" };
    int32_t box[4]; dm_picture_box(SW, SH, PW, PH, box);
    for (int mi = 0; mi < 4; mi++) for (int one = 0; one < 2; one++) {
        dm_scaler_mode = modes[mi]; g_compose_one = one;
        uint64_t p0 = g_cmp_passes, c0 = g_cmp_clears, o0 = g_cmp_one;
        g_step = "compose";
        VkCommandBuffer cb = once_begin();
        pres_src ps = { NULL, s->img, sv, PW, PH, 0, 0, PW, PH };
        present_compose(cb, 0, &ps, 0);
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = d->img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        VkBuffer hb; vmalloc_t hm; const uint8_t *px = tv_readback(d, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, &hb, &hm, cb);
        once_end(cb);
        memcpy(out[mi][one], px, sizeof out[mi][one]); done[mi][one] = 1;
        int was_one = g_cmp_one != o0;
        if (one && !was_one) printf("SIMGAP compose: one pass not possible on this device (no push descriptors for the upscaler): composed as before\n");
        printf("MEASURE compose (%s, %s): passes on the swapchain image %llu, whole-image clears %llu%s\n", mname[mi], one ? "one pass" : "as before",
               (unsigned long long)(g_cmp_passes - p0), (unsigned long long)(g_cmp_clears - c0), was_one ? " (one pass: the clear is the pass's load op)" : "");
    }
    dm_scaler_mode = -1; g_compose_one = -1;
    for (int mi = 0; mi < 4; mi++) {
        if (!done[mi][0] || !done[mi][1]) continue;
        int maxd = 0, nblack = 0, fx = -1, fy = -1;
        for (int y = 0; y < SH; y++) for (int x = 0; x < SW; x++) { const uint8_t *a = out[mi][0] + (y * SW + x) * 4, *b = out[mi][1] + (y * SW + x) * 4;
            int in = x >= box[0] && y >= box[1] && x < box[0] + box[2] && y < box[1] + box[3];
            if (!in && (b[0] || b[1] || b[2] || b[3] != 255)) { if (!nblack++ && fx < 0) { fx = x; fy = y; } }
            for (int c = 0; c < 4; c++) { int dd = abs(a[c] - b[c]); if (dd > maxd) { maxd = dd; if (fx < 0) { fx = x; fy = y; } } } }
        int tol = mi == 1 ? 6 : 0;
        CHECK(!nblack && maxd <= tol, "compose (%s): one pass against as before: largest difference %d (allowed %d), %d pixels outside the box not black (first at %d,%d)", mname[mi], maxd, tol, nblack, fx, fy);
        printf("WORK compose (%s): one pass against as before: largest channel difference %d\n", mname[mi], maxd);
    }
    memset(&W.img, 0, sizeof W.img); memset(&W.view, 0, sizeof W.view); W.n = 0;
}
// Device memory by lifetime (vk_alloc_c, DM_VK_MEM_LIFETIME): a load as the engine does one, through the real allocator on this device's memory types. Batches of
//   16 uploads, each a 4 MB texture (VA_LONG, kept) and its 4 MB staging buffer (VA_TEMP, freed when its batch is done), and a 6 MB arena block per batch (VA_ARENA,
//   freed); then what vk_trim could give back (empty blocks, one kept per memory type). MEMLIFE prints the numbers for both settings (run_simdev.sh runs it as it is;
//   DM_VK_MEM_LIFETIME=0 ./t_vkdev for the old allocator). With blocks by lifetime every staging / arena block must be empty after the load
static double ml_returnable(void) {   // MB vk_trim would give back now (blocks with nothing in them, less the first one of each memory type)
    double r = 0; uint32_t kept = 0; pthread_mutex_lock(&g_amu);
    for (int k = 0; k < g_nblocks; k++) { vblock *b = g_blocks[k]; if (b->used) continue; if (b->type < 32 && !(kept >> b->type & 1)) { kept |= 1u << b->type; continue; } r += b->size / 1048576.0; }
    pthread_mutex_unlock(&g_amu); return r;
}
static void memlife_work(void) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4u << 20, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
    VkBuffer probe; if (vkCreateBuffer(g_dev, &bi, NULL, &probe)) { printf("SIMGAP memlife: no buffer\n"); return; }
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, probe, &rq); vkDestroyBuffer(g_dev, probe, NULL);
    enum { NB = 4, PER = 16 }; vmalloc_t keep[NB * PER]; int nk = 0;
    double b0[VA_NCLS], u0[VA_NCLS]; vk_mem_by_kind(b0, u0);
    double peak = 0;
    for (int batch = 0; batch < NB; batch++) {
        vmalloc_t tmp[PER], ar; VkMemoryRequirements r4 = rq; r4.size = 4u << 20; VkMemoryRequirements r6 = rq; r6.size = 6u << 20;
        ar = vk_alloc_c(r6, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, VA_ARENA);
        for (int i = 0; i < PER; i++) { keep[nk++] = vk_alloc_c(r4, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VA_LONG);
                                        tmp[i] = vk_alloc_c(r4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1, VA_TEMP); }
        double bb, uu; dm_vk_mem_stats(&bb, &uu); if (bb > peak) peak = bb;
        for (int i = 0; i < PER; i++) vk_free(tmp[i]);   // (the batch done: its fence signalled, once_reap / pre_batch_end)
        vk_free(ar);   // (arena_reset)
    }
    double b1[VA_NCLS], u1[VA_NCLS], tb = 0, tu = 0; vk_mem_by_kind(b1, u1); for (int c = 0; c < VA_NCLS; c++) { tb += b1[c] - b0[c]; tu += u1[c] - u0[c]; }
    double ret = ml_returnable();
    printf("MEMLIFE %s: after the load %.0f MB in blocks for %.0f MB in use (peak %.0f MB in all); vk_trim could give back %.0f MB -> %.0f MB in blocks; "
           "long-lived %.0f / %.0f, staging %.0f / %.0f, arenas %.0f / %.0f\n", vk_lifetime_on() ? "blocks by lifetime" : "one kind (DM_VK_MEM_LIFETIME=0)", tb, tu, peak, ret, tb - ret,
           b1[VA_LONG] - b0[VA_LONG], u1[VA_LONG] - u0[VA_LONG], b1[VA_TEMP] - b0[VA_TEMP], u1[VA_TEMP] - u0[VA_TEMP], b1[VA_ARENA] - b0[VA_ARENA], u1[VA_ARENA] - u0[VA_ARENA]);
    if (vk_lifetime_on()) {
        CHECK(u1[VA_TEMP] - u0[VA_TEMP] < 0.01 && u1[VA_ARENA] - u0[VA_ARENA] < 0.01, "memlife: staging / arena blocks still hold %.1f / %.1f MB after the load", u1[VA_TEMP] - u0[VA_TEMP], u1[VA_ARENA] - u0[VA_ARENA]);
        CHECK(u1[VA_LONG] - u0[VA_LONG] > NB * PER * 4 - 0.5, "memlife: the textures are not all in long-lived blocks (%.1f MB)", u1[VA_LONG] - u0[VA_LONG]);
        // with one memory type for everything (lavapipe, Mali, PowerVR) the staging blocks left over are the slack vk_trim returns
        CHECK(ret + 32.5 >= (b1[VA_TEMP] - b0[VA_TEMP]) + (b1[VA_ARENA] - b0[VA_ARENA]) || g_memp.memoryTypeCount > 1, "memlife: only %.0f MB of %.0f MB staging / arena blocks could go back", ret,
              (b1[VA_TEMP] - b0[VA_TEMP]) + (b1[VA_ARENA] - b0[VA_ARENA]));
    }
    for (int i = 0; i < nk; i++) vk_free(keep[i]);
}
// Shader modules released (shmod_release / shmod_ensure, DM_VK_SHMOD_RELEASE): 400 shaders as shader_load leaves them (SPIR-V kept, module made), released after
//   their idle time, made again for a build; the module count and the host heap before / after (what this driver frees per module: lavapipe keeps a copy of the SPIR-V
//   and its NIR). The test shaders go from the cache afterwards
#include <malloc.h>
static void shmod_work(void) {
    if (!shmod_on()) { printf("SHMOD released: off (DM_VK_SHMOD_RELEASE=0)\n"); return; }
    enum { N = 400 }; vshader *sh[N]; uint8_t key[N][8];
    int64_t n0 = atomic_load(&g_shmod_n);
    for (int i = 0; i < N; i++) {
        const uint32_t *src = i & 1 ? scale_frag_spv : scale_vert_spv; size_t sz = i & 1 ? sizeof scale_frag_spv : sizeof scale_vert_spv;
        vshader *s = calloc(1, sizeof *s); snprintf(s->key, sizeof s->key, "test%04d", i); s->stage = i & 1; s->code = malloc(sz); memcpy(s->code, src, sz); s->nw = sz / 4;
        VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = s->code };
        CHECK(!vkCreateShaderModule(g_dev, &mci, NULL, &s->mod), "shmod: module %d not made", i); shmod_count(s->mod, (int64_t)sz, 1);
        atomic_store(&s->used_ns, i < N / 2 ? 1 : mono_ns());   // (the first half long unused, the second just used)
        memset(key[i], 0, 8); key[i][0] = 0xfe; key[i][1] = (uint8_t)(i >> 8); key[i][2] = (uint8_t)i; pthread_mutex_lock(&g_shmu); nvn_hm_put(&g_shcache, key[i], s); pthread_mutex_unlock(&g_shmu); sh[i] = s;
    }
    struct mallinfo2 m0 = mallinfo2(); int64_t na = atomic_load(&g_shmod_n);
    shmod_release();
    struct mallinfo2 m1 = mallinfo2(); int64_t nb = atomic_load(&g_shmod_n);
    int rel = 0, kept = 0; for (int i = 0; i < N; i++) { if (atomic_load(&sh[i]->rel)) rel += !sh[i]->mod; else kept += sh[i]->mod != VK_NULL_HANDLE; }
    CHECK(rel == N / 2 && kept == N / 2, "shmod: %d released, %d kept (want %d / %d: those unused for %.0f s / those just used)", rel, kept, N / 2, N / 2, g_shmod_idle_ns / 1e9);
    CHECK(na - nb == N / 2, "shmod: the count of alive modules fell by %lld, not %d", (long long)(na - nb), N / 2);
    // a build: rdlock, made again (the same SPIR-V), noted as used
    pthread_rwlock_rdlock(&g_shmod_rw); int ok = 1; for (int i = 0; i < N; i++) ok &= shmod_ensure(sh[i]); pthread_rwlock_unlock(&g_shmod_rw);
    int back = 0; for (int i = 0; i < N; i++) back += sh[i]->mod && !atomic_load(&sh[i]->rel);
    CHECK(ok && back == N && atomic_load(&g_shmod_n) == na, "shmod: %d of %d modules made again (alive %lld, want %lld)", back, N, (long long)atomic_load(&g_shmod_n), (long long)na);
    // a build under way holds the release off
    pthread_rwlock_rdlock(&g_shmod_rw); for (int i = 0; i < N; i++) atomic_store(&sh[i]->used_ns, 1); shmod_release(); int none = 1; for (int i = 0; i < N; i++) none &= sh[i]->mod != VK_NULL_HANDLE; pthread_rwlock_unlock(&g_shmod_rw);
    CHECK(none, "shmod: modules released while a pipeline build held them");
    printf("SHMOD released %d of %d test modules (%lld -> %lld alive, %lld of them the engine's own before): host heap %.2f -> %.2f MB (%.1f KB per module on this driver); made again: %d\n",
           rel, N, (long long)na, (long long)nb, (long long)n0, m0.uordblks / 1048576.0, m1.uordblks / 1048576.0, ((double)m0.uordblks - (double)m1.uordblks) / 1024.0 / (N / 2), back);
    for (int i = 0; i < N; i++) { pthread_mutex_lock(&g_shmu); nvn_hm_del(&g_shcache, key[i]); pthread_mutex_unlock(&g_shmu);
        if (sh[i]->mod) { shmod_count(sh[i]->mod, (int64_t)sh[i]->nw * 4, -1); vkDestroyShaderModule(g_dev, sh[i]->mod, NULL); } free(sh[i]->code); free(sh[i]); }
}

// PREWARM (perf-1124 S1-S5): the wait before play, timed on this device. A synthetic shader set and pipeline list as a phone has them (vk/b-<chain>/<key>.spv + .vkr,
//   vk_pipelines.bin, plus its .sst list for a driver without extended dynamic state), built by the engine's own prewarm (prewarm_start, the end of mtl_init) twice:
//   a first start (no pipeline cache file) and a later one (the cache file the first one saved; every pipeline, shader and program made again, as in a new process).
//   SIMDEV_PW_N pipelines (400 by default) over as many fragment shaders of SIMDEV_PW_OPS arithmetic instructions each (800: about 16 KB of SPIR-V; the game's
//   are 40 KB on average) and SIMDEV_PW_N / 4 vertex shaders. Each SPIR-V differs in its generator word, so the driver compiles every pipeline.
//   Printed: "PREWARM <start>: hold <ms> (the game's first submission held), done <ms>", the engine's own [start] / prewarm timer lines around them.
//   SIMDEV_NO_PREWARM=1: skipped
static uint32_t *pw_frag_spv(int ops, uint32_t gen, size_t *nw) {   // an OpFAdd / OpFMul chain on constants into the one colour output
    size_t cap = 64 + (size_t)ops * 5 + 16; uint32_t *w = calloc(cap, 4), n = 0, id = 11;
    w[n++] = 0x07230203; w[n++] = 0x00010000; w[n++] = gen; w[n++] = 0; w[n++] = 0;   // (the bound below)
    w[n++] = 2u << 16 | 17; w[n++] = 1;                                   // OpCapability Shader
    w[n++] = 3u << 16 | 14; w[n++] = 0; w[n++] = 1;                       // OpMemoryModel Logical GLSL450
    w[n++] = 6u << 16 | 15; w[n++] = 4; w[n++] = 1; w[n++] = 0x6e69616d; w[n++] = 0; w[n++] = 2;   // OpEntryPoint Fragment %1 "main" %2
    w[n++] = 3u << 16 | 16; w[n++] = 1; w[n++] = 7;                       // OpExecutionMode %1 OriginUpperLeft
    w[n++] = 4u << 16 | 71; w[n++] = 2; w[n++] = 30; w[n++] = 0;          // OpDecorate %2 Location 0
    w[n++] = 2u << 16 | 19; w[n++] = 3;                                   // %3 void
    w[n++] = 3u << 16 | 33; w[n++] = 4; w[n++] = 3;                       // %4 function type
    w[n++] = 3u << 16 | 22; w[n++] = 5; w[n++] = 32;                      // %5 float
    w[n++] = 4u << 16 | 23; w[n++] = 6; w[n++] = 5; w[n++] = 4;           // %6 vec4
    w[n++] = 4u << 16 | 32; w[n++] = 7; w[n++] = 3; w[n++] = 6;           // %7 pointer Output vec4
    w[n++] = 4u << 16 | 59; w[n++] = 7; w[n++] = 2; w[n++] = 3;           // %2 the output
    w[n++] = 4u << 16 | 43; w[n++] = 5; w[n++] = 8; w[n++] = 0x3f800000;  // %8 1.0
    w[n++] = 4u << 16 | 43; w[n++] = 5; w[n++] = 9; w[n++] = 0x3f000000;  // %9 0.5
    w[n++] = 5u << 16 | 54; w[n++] = 3; w[n++] = 1; w[n++] = 0; w[n++] = 4;   // %1 main
    w[n++] = 2u << 16 | 248; w[n++] = 10;                                 // %10 its label
    uint32_t prev = 8;
    for (int i = 0; i < ops; i++, id++) { w[n++] = 5u << 16 | (i & 1 ? 133u : 129u); w[n++] = 5; w[n++] = id; w[n++] = prev; w[n++] = i % 3 ? 9 : 8; prev = id; }
    uint32_t col = id++;
    w[n++] = 7u << 16 | 80; w[n++] = 6; w[n++] = col; w[n++] = prev; w[n++] = prev; w[n++] = prev; w[n++] = 8;   // OpCompositeConstruct
    w[n++] = 3u << 16 | 62; w[n++] = 2; w[n++] = col;                     // OpStore
    w[n++] = 1u << 16 | 253; w[n++] = 1u << 16 | 56;                      // OpReturn, OpFunctionEnd
    w[3] = id; *nw = n; return w;
}
static int pw_write(const char *path, const void *d, size_t n) { FILE *f = fopen(path, "wb"); if (!f) return 0; int ok = fwrite(d, 1, n, f) == n; return !fclose(f) && ok; }
static void pw_key(uint8_t k[8], int kind, int i) { memset(k, 0, 8); k[0] = 0x70; k[1] = (uint8_t)kind; k[2] = (uint8_t)(i >> 8); k[3] = (uint8_t)i; k[7] = 1; }
static void pw_hex(const uint8_t k[8], char hex[17]) { for (int i = 0; i < 8; i++) sprintf(hex + 2 * i, "%02x", k[i]); }
// what a new process would not have: every pipeline, program and shader of the first start (the driver's pipeline cache is read again from its file)
static void pw_forget(void) {
    pthread_mutex_lock(&g_psomu);
    for (uint32_t i = 0; i < g_pso.cap; i++) if (g_pso.vals[i]) vkDestroyPipeline(g_dev, (VkPipeline)g_pso.vals[i], NULL);
    for (uint32_t i = 0; i < g_pso_s.cap; i++) if (g_pso_s.vals[i]) vkDestroyPipeline(g_dev, (VkPipeline)g_pso_s.vals[i], NULL);
    nvn_hm_clear(&g_pso, NULL); nvn_hm_clear(&g_pso_s, NULL); nvn_hm_clear(&g_pso_fail, NULL); nvn_hm_clear(&g_pso_s_fail, NULL); nvn_hm_clear(&g_pso_used, NULL);
    nvn_hm_clear(&g_pso_rec, NULL); nvn_hm_clear(&g_pso_rec_s, NULL); nvn_hm_clear(&g_pso_s_rec, NULL);
    pthread_mutex_unlock(&g_psomu);
    pthread_mutex_lock(&g_shmu);
    nvn_hmap fresh = {.klen = 8};
    for (uint32_t i = 0; i < g_shcache.cap; i++) { vshader *s = g_shcache.vals[i]; if (!s) continue;
        if (strncmp(s->key, "70", 2)) { nvn_hm_put(&fresh, g_shcache.keys + (size_t)i * 8, s); continue; }   // (the test's own shaders: keys 70...)
        if (s->mod) { shmod_count(s->mod, (int64_t)s->nw * 4, -1); vkDestroyShaderModule(g_dev, s->mod, NULL); }
        for (int v = 0; v < s->nvar; v++) if (s->var[v].mod) vkDestroyShaderModule(g_dev, s->var[v].mod, NULL);
        for (int v = 0; v < s->nibv; v++) if (s->ibv[v].mod) vkDestroyShaderModule(g_dev, s->ibv[v].mod, NULL);
        if (s->glmod) vkDestroyShaderModule(g_dev, s->glmod, NULL); }   // (the vshader itself is left: a layout made from it may still name it)
    free(g_shcache.keys); free(g_shcache.vals); g_shcache = fresh;
    pthread_mutex_unlock(&g_shmu);
    nvn_hm_clear(&g_pw.progs, NULL); nvn_hm_clear(&g_pw.seen, NULL);
    atomic_store(&g_pw.next, 0); atomic_store(&g_pw.unfit, 0); atomic_store(&g_pw.built, 0); atomic_store(&g_pw.had, 0); atomic_store(&g_pw.failed, 0); atomic_store(&g_pw.noshader, 0); atomic_store(&g_pw.waits, 0);
    atomic_store(&g_pw.t_lock, 0); atomic_store(&g_pw.t_shader, 0); atomic_store(&g_pw.t_layout, 0); atomic_store(&g_pw.t_pipe, 0);
    atomic_store(&g_pw.kept, 0); atomic_store(&g_pw.cached, 0); atomic_store(&g_pw.skipped, 0); atomic_store(&g_pw.dups, 0); nvn_hm_clear(&g_pw_sprogs, NULL);
    atomic_store(&g_pw_done, 0); atomic_store(&g_pw_total, 0); atomic_store(&g_pw_playing, 0); atomic_store(&g_pw_ended, 0);
    if (g_pcache) { PFN_vkDestroyPipelineCache dpc = (PFN_vkDestroyPipelineCache)vkGetDeviceProcAddr(g_dev, "vkDestroyPipelineCache"); if (dpc) dpc(g_dev, g_pcache, NULL); g_pcache = VK_NULL_HANDLE; }
}
typedef struct { double hold_ms, done_ms; int built, had, at_release, warm, kept, cached, skipped; int saved_before_release, saved; } pw_run_t;
static long long pw_mtime(const char *p) { struct stat st; if (stat(p, &st)) return -1; return (long long)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec; }
// one start: the prewarm as mtl_init starts it, the game's first submission at once (prewarm_wait's wait, without its 200 ms polling)
static pw_run_t pw_start(const char *label, const char *cache) {
    pw_run_t r; memset(&r, 0, sizeof r);
    long long m0 = pw_mtime(cache);
    uint64_t t0 = mono_ns(); atomic_store(&g_pcache_saved_at, 0); atomic_store(&g_pw_released_at, 0);
    prewarm_start();
    while (atomic_load(&g_pw_hold)) usleep(200);
    r.hold_ms = (mono_ns() - t0) / 1e6; r.at_release = atomic_load(&g_pw_done); r.warm = atomic_load(&g_pw_warm);
    // (the game's first present is not made here: with a matched stamp the prewarm's threads go on as they would until then)
    while (!atomic_load(&g_pw_ended)) usleep(500);
    r.done_ms = (mono_ns() - t0) / 1e6; r.built = atomic_load(&g_pw.built); r.had = atomic_load(&g_pw.had); r.saved = pw_mtime(cache) != m0;
    r.kept = atomic_load(&g_pw.kept); r.cached = atomic_load(&g_pw.cached); r.skipped = atomic_load(&g_pw.skipped);
    { uint64_t s = atomic_load(&g_pcache_saved_at), l = atomic_load(&g_pw_released_at); r.saved_before_release = s && (!l || s < l); }
    printf("PREWARM %s: hold %.0f ms (the game's first submission held, %d entries handled by then), done %.0f ms; %d built, %d already there%s; stamp %s; the cache file was %s before the game went on, %s\n",
           label, r.hold_ms, r.at_release, r.done_ms, r.built, r.had, r.cached || r.skipped ? " (past the cap: some cache-only or skipped)" : "", r.warm ? "matched" : "not matched",
           r.saved_before_release ? "WRITTEN" : "not written", r.saved ? "written" : "not written at all");
    // S1: the game goes on before the cache is written (nvn_vk_prewarm_progress says 0 while the file is still the old one)
    CHECK(!r.saved_before_release, "prewarm (%s): the pipeline cache file was written before the game was let go", label);
    return r;
}
// the test shaders' SPIR-V hashes as loaded (S2: the same whether shaders load on one thread or many)
static int pw_hashes(uint64_t *h, int max) {
    int n = 0; pthread_mutex_lock(&g_shmu);
    for (uint32_t i = 0; i < g_shcache.cap && n < max; i++) { vshader *s = g_shcache.vals[i]; if (!s || strncmp(s->key, "70", 2)) continue;
        uint64_t kk = strtoull(s->key, NULL, 16); h[2 * n] = kk; h[2 * n + 1] = s->spv_hash; n++; }
    pthread_mutex_unlock(&g_shmu);
    for (int a = 1; a < n; a++) for (int b = a; b > 0 && h[2 * (b - 1)] > h[2 * b]; b--) { uint64_t t0 = h[2 * b], t1 = h[2 * b + 1]; h[2 * b] = h[2 * (b - 1)]; h[2 * b + 1] = h[2 * (b - 1) + 1]; h[2 * (b - 1)] = t0; h[2 * (b - 1) + 1] = t1; }
    return n;
}
static void prewarm_work(void) {
    if (getenv("SIMDEV_NO_PREWARM")) { printf("PREWARM skipped (SIMDEV_NO_PREWARM)\n"); return; }
    int N = getenv("SIMDEV_PW_N") ? atoi(getenv("SIMDEV_PW_N")) : 400, OPS = getenv("SIMDEV_PW_OPS") ? atoi(getenv("SIMDEV_PW_OPS")) : 800, NV = N / 4 > 0 ? N / 4 : 1;
    if (N > 65535) N = 65535;
    char dir[96], sub[160], p[300], hex[17]; snprintf(dir, sizeof dir, "/tmp/t_vkdev_pw.%d", (int)getpid()); snprintf(sub, sizeof sub, "%s/b-%s", dir, DMSH_CHAIN);
    mkdir(dir, 0700); mkdir(sub, 0700);
    snprintf(p, sizeof p, "%s/shader_map.tsv", dir); FILE *map = fopen(p, "w"); if (!map) { CHECK(0, "prewarm: no scratch folder %s", dir); return; }
    fprintf(map, "package\tindex\tstage\tkey\tsource\n");
    uint8_t k[8];
    for (int i = 0; i < NV; i++) { pw_key(k, 0, i); pw_hex(k, hex); size_t sz = sizeof zcopy_vert_spv; uint32_t *w = malloc(sz); memcpy(w, zcopy_vert_spv, sz); w[2] = 0x10000u + (uint32_t)i;
        snprintf(p, sizeof p, "%s/%s.spv", sub, hex); pw_write(p, w, sz); free(w); snprintf(p, sizeof p, "%s/%s.vkr", sub, hex); pw_write(p, "# t_vkdev prewarm\n", 18);
        fprintf(map, "pw\t%d\tvertex\t%s\tt_vkdev\n", i, hex); }
    for (int i = 0; i < N; i++) { pw_key(k, 1, i); pw_hex(k, hex); size_t nw; uint32_t *w = pw_frag_spv(OPS, 0x20000u + (uint32_t)i, &nw);
        snprintf(p, sizeof p, "%s/%s.spv", sub, hex); pw_write(p, w, nw * 4); free(w); snprintf(p, sizeof p, "%s/%s.vkr", sub, hex); pw_write(p, "# t_vkdev prewarm\n", 18);
        fprintf(map, "pw\t%d\tpixel\t%s\tt_vkdev\n", i, hex); }
    fclose(map);
    snprintf(p, sizeof p, "%s/shader_map.tsv", dir); setenv("DM_NVN_SHMAP", p, 1); setenv("DM_SPV_DIR", dir, 1); unsetenv("DM_SPV_PAK"); unsetenv("DM_LIVE_SPV");
    shmap_load();
    char list[200], cache[200], lst[220]; snprintf(list, sizeof list, "%s/vk_pipelines.bin", dir); snprintf(cache, sizeof cache, "%s/vk_pipeline_cache.bin", dir); snprintf(lst, sizeof lst, "%s.sst", list);
    FILE *f = fopen(list, "wb"), *fs = fopen(lst, "wb");
    if (!f || !fs) { CHECK(0, "prewarm: the lists cannot be written in %s", dir); if (f) fclose(f); if (fs) fclose(fs); return; }
    { uint32_t v[2] = { 1, (uint32_t)sizeof(pso_rec) }, vs[2] = { 1, (uint32_t)sizeof(pso_rec_s) }; fwrite(PSO_MAGIC, 1, 4, f); fwrite(v, 4, 2, f); fwrite(PSO_MAGIC_S, 1, 4, fs); fwrite(vs, 4, 2, fs); }
    for (int i = 0; i < N; i++) {
        pso_rec_s r; memset(&r, 0, sizeof r);
        pw_key(r.r.key[0], 0, i % NV); pw_key(r.r.key[1], 1, i);
        r.r.k.cfmt[0] = VK_FORMAT_R8G8B8A8_UNORM; r.r.k.chmask = 0xf; r.r.k.samples = 1; r.r.k.topo = 2; r.r.k.depth_zero = 1;
        r.s.topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; r.s.dcmp = VK_COMPARE_OP_ALWAYS; for (int q = 0; q < 2; q++) r.s.sop[q][3] = VK_COMPARE_OP_ALWAYS;
        fwrite(&r.r, sizeof r.r, 1, f); fwrite(&r, sizeof r, 1, fs);
    }
    fclose(f); fclose(fs);
    setenv("DM_VK_PSO_RECORD", list, 1); unsetenv("DM_VK_PSO_SEED"); setenv("DM_VK_PCACHE", cache, 1); unlink(cache);
    printf("PREWARM list: %d pipelines, %d vertex + %d fragment shaders (%d instructions each), %s\n", N, NV, N, OPS,
           g_eds ? "the shared list" : "this phone's list with the draw state (no extended dynamic state)");
    pw_forget(); { char sp[220]; snprintf(sp, sizeof sp, "%s.stamp", cache); unlink(sp); }
    int cap = pso_cap(), keep = pw_cap_keep(), skip = pw_misses_skip();
    if (cap) printf("PREWARM the cap: at most %d live pipelines, %d kept by the prewarm\n", cap, keep);
    pw_run_t c = pw_start("first start", cache);
    // S3: no stamp yet, the game waits for every pipeline; S4 c: at most the cap's share kept, the rest only into the cache
    CHECK(!c.warm && c.at_release == N, "prewarm: the first start let the game go after %d of %d", c.at_release, N);
    if (keep && g_eds) CHECK(c.kept <= keep + 16 && c.kept + c.cached == N, "prewarm: under the cap %d kept (at most %d + threads) and %d cache-only of %d", c.kept, keep, c.cached, N);
    int nh = N + NV; uint64_t *h1 = calloc((size_t)nh * 2, 8), *h0 = calloc((size_t)nh * 2, 8); int n1 = pw_hashes(h1, nh);
    CHECK(!g_eds || c.built == N, "prewarm: %d of %d built at the first start", c.built, N);
    CHECK(pw_mtime(cache) >= 0 && c.saved, "prewarm: no pipeline cache file after the first start");
    pw_forget();
    pw_run_t w = pw_start("later start", cache);
    CHECK(!g_eds || keep || w.built == N, "prewarm: %d of %d built at the later start", w.built, N);
    // S3: the stamp matched; where a missing pipeline's draws are skipped (this GPU's DM_VK_ASYNC_PSO) the game goes on at once, else after the first 600 entries
    CHECK(w.warm, "prewarm: the later start did not match the first start's stamp");
    if (skip) CHECK(w.at_release < N, "prewarm: the later start still held the game for all %d pipelines", N);
    if (keep && g_eds) CHECK(w.skipped == N - w.kept, "prewarm: past the cap at a matched start %d skipped, want %d", w.skipped, N - w.kept);
    // S2: the same SPIR-V with shaders loaded one at a time (DM_VK_PREWARM_SHADER_MT=0)
    pw_forget(); g_sh_mt = 0;
    pw_start("later start, shaders loaded one at a time", cache);
    g_sh_mt = 1; int n0 = pw_hashes(h0, nh);
    CHECK(n1 == nh && n0 == nh && !memcmp(h0, h1, (size_t)nh * 16), "prewarm: the shaders' SPIR-V differs between loading on many threads and on one (%d / %d of %d loaded)", n1, n0, nh);
    free(h0); free(h1);
    // S3: DM_VK_PREWARM_HOLD=<count> (Legends: Arceus's default, 600): the game goes on once the first <count> entries are handled
    { pw_forget(); char hc[16]; int cnt = N / 4 > 0 ? N / 4 : 1; snprintf(hc, sizeof hc, "%d", cnt); setenv("DM_VK_PREWARM_HOLD", hc, 1);
      pw_run_t h = pw_start("later start, DM_VK_PREWARM_HOLD=N/4", cache); unsetenv("DM_VK_PREWARM_HOLD");
      CHECK(h.warm && h.at_release >= cnt && h.at_release < N, "prewarm: a count hold of %d let the game go after %d of %d", cnt, h.at_release, N); }
    // S4 a / b: the record list with a fifth of it repeated (the cap's destroyed-and-rebuilt pipelines, appended again before): the stamp differs, the game waits
    //   for every pipeline, and after it the list is written again without the repeats (same format, same order)
    if (g_eds) { FILE *a = fopen(list, "ab"); int extra = N / 5;
      for (int i = 0; i < extra; i++) { pso_rec r; memset(&r, 0, sizeof r); pw_key(r.key[0], 0, i % NV); pw_key(r.key[1], 1, i);
          r.k.cfmt[0] = VK_FORMAT_R8G8B8A8_UNORM; r.k.chmask = 0xf; r.k.samples = 1; r.k.topo = 2; r.k.depth_zero = 1; fwrite(&r, sizeof r, 1, a); }
      fclose(a);
      pw_forget(); pw_run_t g = pw_start("later start, the list grown by repeats", cache);
      CHECK(!g.warm && g.at_release == N + extra, "prewarm: a grown list let the game go after %d of %d (want all: the stamp differs)", g.at_release, N + extra);
      struct stat st; CHECK(!stat(list, &st) && st.st_size == 12 + (off_t)N * (off_t)sizeof(pso_rec), "prewarm: the record list is %lld bytes after the compaction, want %lld (%d entries)",
                            (long long)st.st_size, 12 + (long long)N * (long long)sizeof(pso_rec), N);
      pw_forget(); pw_run_t g2 = pw_start("later start, after the compaction", cache);
      CHECK(g2.warm, "prewarm: the start after the compaction did not match its stamp"); }
    unsetenv("DM_VK_PSO_RECORD"); unsetenv("DM_VK_PCACHE");
}

// S5 (perf-1124): mtl_init from two threads at once, as the early set-up thread (nvn_vk_early_init, DM_VK_EARLY_INIT=1) and the game's first memory pool would:
//   one device, and each caller returns only when the set-up is complete (it used to return as soon as g_dev was set, 200 lines before the end)
static _Atomic int g_race_ok = -1;
static void *init_racer(void *a) { (void)a; mtl_init(); atomic_store(&g_race_ok, atomic_load(&g_gpuid_done) && g_pcache != VK_NULL_HANDLE); return NULL; }
static void init_race(void) {
    setenv("DM_VK_EARLY_INIT", "1", 1); nvn_vk_early_init(); unsetenv("DM_VK_EARLY_INIT");
    pthread_t t; int th = !pthread_create(&t, NULL, init_racer, NULL);
    mtl_init(); int main_ok = atomic_load(&g_gpuid_done) && g_pcache != VK_NULL_HANDLE;
    if (th) pthread_join(t, NULL);
    for (int i = 0; i < 2000 && atomic_load(&w_dev_count) < 1; i++) usleep(1000);
    CHECK(main_ok && atomic_load(&g_race_ok) == 1 && atomic_load(&w_dev_count) == 1, "mtl_init from three threads: set-up complete when it returned %d / %d, %d devices made (want 1)",
          main_ok, atomic_load(&g_race_ok), atomic_load(&w_dev_count));
    printf("INIT three callers at once (the early set-up thread, a racer, the main thread): %d device made, each returned with the set-up complete: %s\n",
           atomic_load(&w_dev_count), main_ok && atomic_load(&g_race_ok) == 1 ? "yes" : "NO");
}
// ---- perf-1124 GPU fixes A and B (nvn_vk.c pend_clears, lc_*): clear-only depth passes ----
// A 64x64 DEPTH32F target D (and D2), an RGBA8 colour target C and a smaller one Cs, the commands as hle_nvn.c records them. Each case starts from D and D2 cleared to
//   0 (clear_tex), runs its commands in one submission, waits and reads D (or D2) back through the CPU's read (mtl_read_texels): every texel must hold what the
//   clear-only pass gave before. The cases are the ways a kept clear ends: taken by the next pass's load op, replaced by a second clear, written out by the
//   submission's end, by another pass, by a pass on part of the image, by a texture copy, a texture clear, a readback into a buffer, a dispatch, a capture pass, and
//   a draw sampling the image in the pass that took it (the restart, g_lc_hazard). Fix A: a depth clear with no depth target, a stencil clear of a depth-only one.
//   Passes begun (vkCmdBeginRendering / vkCmdBeginRenderPass of the game's passes) are counted per case against what the switches say; MEASURE prints the total
//   (DM_VK_LAZY_CLEAR=0 DM_VK_EMPTY_PASS=1: as before)
static uint8_t *lc_q;
#define LCOP(op_, T, v) do { op_hdr h_ = { (op_), (uint32_t)((8 + sizeof(T) + 7) & ~7u) }; memcpy(lc_q, &h_, 8); memset(lc_q + 8, 0, h_.size - 8); memcpy(lc_q + 8, &(v), sizeof(T)); lc_q += h_.size; } while (0)
static void lc_end(void) { op_hdr e = { OP_END, 8 }; memcpy(lc_q, &e, 8); lc_q += 8; }
static void lc_zero(tex_t *T, uint32_t W, float v) { p_clear_tex ct; memset(&ct, 0, sizeof ct); ct.tex = T; ct.r.width = W; ct.r.height = W; ct.r.depth = 1; ct.mask = 0xf; ct.c.f[0] = v; clear_tex(&ct); }
static int lc_bad(tex_t *T, uint32_t W, float want) {
    static float got[64 * 64]; memset(got, 0xff, sizeof got); NVNcopyRegion r = { 0, 0, 0, (int)W, (int)W, 1 };
    mtl_read_texels(T, 0, 0, &r, (uint8_t *)got, W * 4, W * W * 4); int bad = 0; for (uint32_t i = 0; i < W * W; i++) bad += got[i] != want; return bad;
}
static void lc_work(void) {
    enum { W = 64 };
    size_t n = 1u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); CHECK(mem != MAP_FAILED, "clears mmap"); if (mem == MAP_FAILED) return;
    memset(mem, 0, n); if (g_host_import || g_fdmap || g_heapmap) mtl_pool_add(mem, n);   // (the readback case's buffer: work's SIMGAP for the moved-mapping route)
    pool_t *pl = calloc(1, sizeof *pl); pl->magic = MAGIC('P', 'O', 'O', 'L'); pl->flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED; pl->mem = mem; pl->size = n;
    tex_t *D = b10_tex(pl, mem + 0x10000, NVN_FORMAT_DEPTH32F, W, W, 0), *D2 = b10_tex(pl, mem + 0x30000, NVN_FORMAT_DEPTH32F, W, W, 0);
    tex_t *C = b10_tex(pl, mem + 0x50000, NVN_FORMAT_RGBA8, W, W, 0), *Cs = b10_tex(pl, mem + 0x70000, NVN_FORMAT_RGBA8, W / 2, W / 2, 0);
    vtex *dx = tex_res(D), *d2 = tex_res(D2), *cx = tex_res(C);
    if (!dx || !dx->img || !(dx->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) || !d2 || !d2->img || !cx || !cx->img) { printf("SIMGAP clears: no DEPTH32F / RGBA8 target here\n"); return; }
    int lazy = lc_on(), olda = empty_pass_old(); uint64_t total = 0; int nbad = 0;
    p_set_rt rd, rcd, rc, rsd; memset(&rd, 0, sizeof rd); rd.depth = D; rcd = rd; rcd.n = 1; rcd.color[0] = C; memset(&rc, 0, sizeof rc); rc.n = 1; rc.color[0] = C; rsd = rcd; rsd.color[0] = Cs;
    static uint64_t cmd[512];
    #define LC_CASE(name, want_new, want_old, BODY, T, val) do { g_step = "clears: " name; \
        mtl_sub_begin(); lc_zero(D, W, 0); lc_zero(D2, W, 0); { uint64_t f_ = mtl_sub_commit(); mtl_wait(f_, UINT64_MAX); mtl_finish(); } \
        uint64_t b0_ = g_stat_begins, w0_ = g_lc_written, k0_ = g_lc_kept; mtl_sub_begin(); lc_q = (uint8_t *)cmd; BODY; \
        uint64_t f_ = mtl_sub_commit(); mtl_wait(f_, UINT64_MAX); mtl_finish(); uint64_t nb_ = g_stat_begins - b0_; total += nb_; \
        int bad_ = lc_bad(T, W, val); nbad += bad_ != 0; \
        CHECK(!bad_, "clears, %s: %d of %d texels not %g", name, bad_, W * W, (double)(val)); \
        CHECK(nb_ == (uint64_t)(lazy ? want_new : want_old), "clears, %s: %llu passes begun, %d expected (DM_VK_LAZY_CLEAR %d)", name, (unsigned long long)nb_, lazy ? want_new : want_old, lazy); \
        CHECK(g_nlc == 0, "clears, %s: %d kept clears left after the submission", name, g_nlc); \
        printf("WORK clears, %s: %llu passes begun, %llu kept, %llu written out\n", name, (unsigned long long)nb_, (unsigned long long)(g_lc_kept - k0_), (unsigned long long)(g_lc_written - w0_)); } while (0)
    #define RUN() do { lc_end(); mtl_exec((const uint8_t *)cmd); } while (0)
    p_clear_ds z25 = { 0.25f, 1, 0, 0 }, z10 = { 0.1f, 1, 0, 0 }, z75 = { 0.75f, 1, 0, 0 }, z50 = { 0.5f, 1, 0, 0 }, z30 = { 0.3f, 1, 0, 0 }, z60 = { 0.6f, 1, 0, 0 }, z20 = { 0.2f, 1, 0, 0 }, z40 = { 0.4f, 1, 0, 0 };
    // taken by the next pass's load op (pass_begin stands for its first draw)
    LC_CASE("taken by the next pass", 1, 2, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z25); LCOP(OP_SET_RT, p_set_rt, rcd); RUN();
        CHECK(!lazy || g_nlc == 1, "clears: the clear kept (%d)", g_nlc); CHECK(pass_begin(), "clears: the pass began"); CHECK(!lazy || g_lc_open == dx, "clears: the pass took the kept clear"); pass_end(); }, D, 0.25f);
    // cleared twice with nothing between (Sword / Shield's first two passes of a frame), then taken
    LC_CASE("cleared twice, then taken", 1, 3, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z10); LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z75); LCOP(OP_SET_RT, p_set_rt, rcd); RUN();
        pass_begin(); pass_end(); }, D, 0.75f);
    LC_CASE("written out by the submission's end", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z50); LCOP(OP_SET_RT, p_set_rt, rc); RUN(); }, D, 0.5f);
    LC_CASE("written out by a pass on other targets", 2, 2, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z30); LCOP(OP_SET_RT, p_set_rt, rc); RUN(); pass_begin();
        CHECK(g_nlc == 0, "clears: written out before the other pass"); pass_end(); }, D, 0.3f);
    LC_CASE("written out by a pass on part of it", 2, 2, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z40); LCOP(OP_SET_RT, p_set_rt, rsd); RUN(); pass_begin();
        CHECK(g_lc_open == NULL, "clears: a pass smaller than the image did not take it"); pass_end(); }, D, 0.4f);
    { p_copy_t2t tt; memset(&tt, 0, sizeof tt); tt.src = D; tt.dst = D2; tt.sr.width = tt.dr.width = W; tt.sr.height = tt.dr.height = W; tt.sr.depth = tt.dr.depth = 1;
      LC_CASE("written out by a texture copy from it", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z60); LCOP(OP_SET_RT, p_set_rt, rc); LCOP(OP_COPY_T2T, p_copy_t2t, tt); RUN(); }, D2, 0.6f);
      tt.src = D2; tt.dst = D;   // (into it: the clear first, the copy over it)
      LC_CASE("written out by a texture copy into it", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z60); LCOP(OP_SET_RT, p_set_rt, rc); LCOP(OP_COPY_T2T, p_copy_t2t, tt); RUN(); }, D, 0.0f); }
    { p_clear_tex ct; memset(&ct, 0, sizeof ct); ct.tex = D; ct.r.width = W; ct.r.height = W; ct.r.depth = 1; ct.mask = 0xf; ct.c.f[0] = 0.9f;
      LC_CASE("written out by a texture clear of it", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z20); LCOP(OP_SET_RT, p_set_rt, rc); LCOP(OP_CLEAR_TEX, p_clear_tex, ct); RUN(); }, D, 0.9f); }
    { p_copy_b2t cb; memset(&cb, 0, sizeof cb); cb.tex = D; cb.src = (uint64_t)(uintptr_t)(mem + 0x90000); cb.r.width = W; cb.r.height = W; cb.r.depth = 1;
      int pooled = 0; { VkBuffer b; VkDeviceSize o; pooled = pool_buf(cb.src, W * W * 4, &b, &o, NULL); }
      if (pooled) LC_CASE("written out by a readback into a buffer", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z50); LCOP(OP_SET_RT, p_set_rt, rc); RUN();
          uint64_t w0 = g_lc_written; copy_b2t(&cb, 1); CHECK(!lazy || g_lc_written == w0 + 1, "clears: written out before the readback's copy"); }, D, 0.5f);
      else printf("SIMGAP clears: this pool is not the GPU's here: the readback case left out\n"); }
    LC_CASE("written out by a dispatch", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z30); LCOP(OP_SET_RT, p_set_rt, rc); RUN();
        uint64_t w0 = g_lc_written; const prog_t *kp = S.cprog; S.cprog = NULL; p_dispatch dp; memset(&dp, 0, sizeof dp); dispatch(&dp); S.cprog = kp;
        CHECK(!lazy || g_lc_written == w0 + 1, "clears: written out by the dispatch"); }, D, 0.3f);
    LC_CASE("written out by a capture pass", 1, 1, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z40); LCOP(OP_SET_RT, p_set_rt, rc); RUN();
        uint64_t w0 = g_lc_written; tf_pass_begin(0); CHECK(!lazy || g_lc_written == w0 + 1, "clears: written out before the capture pass"); CHECK(P.open && P_tf, "clears: the capture pass is open as a capture pass"); pass_end(); }, D, 0.4f);
    // the pass that took it samples it: a draw binding it as a texture reads it (sk_read), the pass is begun again before that draw (draw_ does what is done here)
    LC_CASE("taken, then sampled in that pass", 2, 2, { LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, z75); LCOP(OP_SET_RT, p_set_rt, rcd); RUN(); pass_begin();
        sk_read(dx); CHECK(g_lc_hazard == lazy, "clears: a read of the image in the pass that took its clear: the draw begins the pass again (%d)", g_lc_hazard);
        if (g_lc_hazard) { g_lc_hazard = 0; pass_end(); pass_begin(); CHECK(g_lc_open == NULL && !R.clear_d, "clears: the pass begun again loads"); }
        pass_end(); }, D, 0.75f);   // (as before: the clear-only pass and the pass; now the pass that took it and the one begun again)
    // fix A: a depth clear with the colour target alone, a stencil clear of a depth-only image (DEPTH32F has no stencil)
    p_clear_ds s1 = { 0.0f, 0, 7, 0xff };
    { mtl_sub_begin(); lc_zero(D, W, 0); { uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish(); }
      uint64_t b0 = g_stat_begins, e0 = g_empty_dropped; mtl_sub_begin(); lc_q = (uint8_t *)cmd; LCOP(OP_SET_RT, p_set_rt, rc); LCOP(OP_CLEAR_DS, p_clear_ds, z50); LCOP(OP_SET_RT, p_set_rt, rc);
      LCOP(OP_SET_RT, p_set_rt, rd); LCOP(OP_CLEAR_DS, p_clear_ds, s1); LCOP(OP_SET_RT, p_set_rt, rc); RUN();
      uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish(); uint64_t nb = g_stat_begins - b0; total += nb;
      CHECK(nb == (olda ? 2u : 0u), "clears, A: %llu passes begun for a depth clear with no depth target and a stencil clear of a depth-only image, %d expected (DM_VK_EMPTY_PASS %d)", (unsigned long long)nb, olda ? 2 : 0, olda);
      CHECK(olda || g_empty_dropped - e0 == 2, "clears, A: %llu empty passes dropped", (unsigned long long)(g_empty_dropped - e0));
      int bad = lc_bad(D, W, 0.0f); CHECK(!bad, "clears, A: the depth image kept as it was (%d texels changed)", bad); nbad += bad != 0;
      printf("WORK clears, A: %llu passes begun for clears with nothing to clear\n", (unsigned long long)nb); }
    #undef LC_CASE
    #undef RUN
    printf("MEASURE clear-only passes (DM_VK_LAZY_CLEAR %d, DM_VK_EMPTY_PASS old %d, %s, %s): %llu passes begun over the cases; kept %llu, taken by a load op %llu, replaced %llu, written out %llu (by a read %llu); %s\n",
           lazy, olda, g_dynren ? "dynamic rendering" : "render passes", hold_dec() ? "held" : "as recorded", (unsigned long long)total, (unsigned long long)g_lc_kept, (unsigned long long)g_lc_folded,
           (unsigned long long)g_lc_replaced, (unsigned long long)g_lc_written, (unsigned long long)g_lc_net, nbad ? "WRONG" : "every texel as before");
}
// ---- perf-1124 GPU fix C (nvn_vk.c ds_none, sk_ro without the store learning): depth / stencil a pass does not write ----
// The load / store NONE extension a profile lists but lavapipe has not got (w_cd left it out): the ops it brings are not used here (a gap of the simulation)
static void ds_simgap(void) {
    int en = (w_has("VK_KHR_load_store_op_none") && lvp_has_ext("VK_KHR_load_store_op_none")) || (w_has("VK_EXT_load_store_op_none") && lvp_has_ext("VK_EXT_load_store_op_none"));
    if (g_load_none && !en) { g_load_none = 0; printf("SIMGAP load / store NONE: the profile's extension is not lavapipe's: LOAD_OP_NONE not used here\n"); }
    if (g_store_none && g_api < VK_API_VERSION_1_3 && !en) { g_store_none = 0; printf("SIMGAP load / store NONE: the profile's extension is not lavapipe's: STORE_OP_NONE not used here\n"); }
    DEC("store_none", "%d", g_store_none); DEC("load_none", "%d", g_load_none);
}
// A 64x64 DEPTH32F target D at 0.5 with an RGBA8 colour target C, a game pass on both per frame (pass_begin / pass_end stand for its draws), read back after each:
//   - held (DM_VK_FRAME_HOLD=2): a pass that writes nothing is stored with NONE at the flush (and loaded with NONE where the extension is), D still 0.5; a pass
//     that clears D inside it (0.8) and a pass after it on the same targets that writes nothing (run as one on Adreno's profiles): stored, D 0.8;
//   - as recorded: four passes on the same targets that only read, the fifth begins with store NONE (sk_ro, now without DM_VK_SKIP_STORES), D still 0.5; then a
//     pass that clears D inside it: it is begun again storing (pass_writes), D 0.8.
//   MEASURE: depth stores skipped
static void ds_work(void) {
    enum { W = 64 };
    size_t n = 1u << 20; uint8_t *mem = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); CHECK(mem != MAP_FAILED, "depth stores mmap"); if (mem == MAP_FAILED) return;
    memset(mem, 0, n);
    pool_t *pl = calloc(1, sizeof *pl); pl->magic = MAGIC('P', 'O', 'O', 'L'); pl->flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED; pl->mem = mem; pl->size = n;
    tex_t *D = b10_tex(pl, mem + 0x10000, NVN_FORMAT_DEPTH32F, W, W, 0), *C = b10_tex(pl, mem + 0x50000, NVN_FORMAT_RGBA8, W, W, 0);
    vtex *dx = tex_res(D), *cx = tex_res(C);
    if (!dx || !dx->img || !(dx->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) || !cx || !cx->img) { printf("SIMGAP depth stores: no DEPTH32F / RGBA8 target here\n"); return; }
    p_set_rt rcd; memset(&rcd, 0, sizeof rcd); rcd.n = 1; rcd.color[0] = C; rcd.depth = D;
    static uint64_t cmd[64]; p_clear_ds z80 = { 0.8f, 1, 0, 0 };
    int held = hold_dec(); uint64_t s0 = g_dsn_store, l0 = g_dsn_load, r0 = g_ro_passes, w0 = g_ro_wrong;
    g_step = "depth stores"; mtl_sub_begin(); lc_zero(D, W, 0.5f); { uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish(); }
    int nro = held ? 1 : 5, none_at = -1, bad = 0;
    for (int i = 0; i < nro; i++) {   // passes that only read (or do nothing with) the depth
        g_frame_no++; mtl_sub_begin(); lc_q = (uint8_t *)cmd; LCOP(OP_SET_RT, p_set_rt, rcd); lc_end(); mtl_exec((const uint8_t *)cmd);
        CHECK(pass_begin(), "depth stores: the pass began"); if (P.dnone && none_at < 0) none_at = i; pass_end();
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
        int b = lc_bad(D, W, 0.5f); bad += b; CHECK(!b, "depth stores: after read-only pass %d the depth image %d texels not 0.5", i, b); }
    if (held) {
        CHECK(!g_store_none || g_dsn_store == s0 + 1, "depth stores, held: a pass that wrote nothing stored with NONE (%llu)", (unsigned long long)(g_dsn_store - s0));
        CHECK(g_dsn_load - l0 == (uint64_t)(g_store_none && g_load_none), "depth stores, held: loaded with NONE where the extension is (%llu, extension %d)", (unsigned long long)(g_dsn_load - l0), g_load_none);
    } else CHECK(none_at == (ro_free() ? SK_RO_LEARN : -1), "depth stores, as recorded: the pass begun with store NONE: %d, expected %d (store NONE %d)", none_at, ro_free() ? SK_RO_LEARN : -1, g_store_none);
    {   // a pass that writes the depth (a clear inside it), and one after it on the same targets that writes nothing
        uint64_t sw = g_dsn_store; g_frame_no++; mtl_sub_begin(); lc_q = (uint8_t *)cmd; LCOP(OP_SET_RT, p_set_rt, rcd); lc_end(); mtl_exec((const uint8_t *)cmd);
        CHECK(pass_begin(), "depth stores: the writing pass began"); int was_none = P.dnone;
        lc_q = (uint8_t *)cmd; LCOP(OP_CLEAR_DS, p_clear_ds, z80); lc_end(); mtl_exec((const uint8_t *)cmd);   // (inside the open pass: clear_in_pass, pass_writes)
        CHECK(!was_none || (g_ro_wrong == w0 + 1 && !P.dnone), "depth stores: the pass begun with NONE was begun again storing (%llu)", (unsigned long long)(g_ro_wrong - w0));
        pass_end(); CHECK(pass_begin(), "depth stores: the pass after it began"); pass_end();
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX); mtl_finish();
        int b = lc_bad(D, W, 0.8f); bad += b; CHECK(!b, "depth stores: after the pass that cleared it to 0.8, %d texels not 0.8", b);
        if (held) CHECK(g_dsn_store - sw <= 1, "depth stores, held: the writing pass kept its store (%llu passes stored with NONE)", (unsigned long long)(g_dsn_store - sw)); }
    printf("MEASURE depth stores (%s, %s, store NONE %d, load NONE %d): depth stores skipped %llu (held passes stored with NONE %llu, loaded with NONE %llu; begun store-none %llu, written after all %llu); %s\n",
           held ? "held" : "as recorded", g_dynren ? "dynamic rendering" : "render passes", g_store_none, g_load_none, (unsigned long long)(g_dsn_store - s0 + (g_ro_passes - r0) - (g_ro_wrong - w0)),
           (unsigned long long)(g_dsn_store - s0), (unsigned long long)(g_dsn_load - l0), (unsigned long long)(g_ro_passes - r0), (unsigned long long)(g_ro_wrong - w0), bad ? "WRONG" : "every texel as before");
}
// A crash prints where it was (addresses for addr2line -f -e t_vkdev; the run's log keeps them) and the step the test was in
static void on_crash(int sig) {
    void *bt[48]; int n = backtrace(bt, 48);
    char m[160]; int k = snprintf(m, sizeof m, "CRASH signal %d during %s; backtrace (addr2line -f -e t_vkdev <address - load base>):\n", sig, g_step); write(1, m, (size_t)k);
    backtrace_symbols_fd(bt, n, 1);
    signal(sig, SIG_DFL); raise(sig);
}
int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGSEGV, on_crash); signal(SIGBUS, on_crash); signal(SIGABRT, on_crash);
    unsetenv("DM_VK_PCACHE");   // (no marker files from this test)
    { const char *e = getenv("SIMDEV_DC_FC"); g_dcfc = e ? atoi(e) : 0;   // (a marker file of its own then, in a scratch folder: header)
      if (g_dcfc) { char dir[64], p[128]; snprintf(dir, sizeof dir, "/tmp/t_vkdev_dc.%d", (int)getpid()); mkdir(dir, 0700);
          snprintf(p, sizeof p, "%s/pipeline_cache.bin", dir); setenv("DM_VK_PCACHE", p, 1);
          snprintf(p, sizeof p, "%s/float_controls_crashed", dir); FILE *f = fopen(p, "w"); if (f) { fputs("t_vkdev: a crash on the float-controls test shader at an earlier start\n", f); fclose(f); }
          setenv("DM_VK_FLOAT_CONTROLS", "1", 1); if (g_dcfc == 1) setenv("DM_DRIVER_CHECK_KEYS", "DM_VK_FLOAT_CONTROLS", 1); else unsetenv("DM_DRIVER_CHECK_KEYS");
          printf("SIMDEV_DC_FC=%d: float_controls_crashed in %s, DM_VK_FLOAT_CONTROLS=1 from %s\n", g_dcfc, dir, g_dcfc == 1 ? "the driver check" : "env.txt"); } }
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL); if (!lib) { printf("FAIL no libvulkan.so.1\n"); return 1; }
    real_gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    vkGetInstanceProcAddr = w_gipa; VK_GLOBAL(VK_LOAD_G)   // (what mtl_init does when it opens the loader itself, through the watching wrapper)
    CHECK(nvn_vk_quirk_default(Q_NVN_ASYNC) == -1 && nvn_vk_quirk_default(Q_FLOAT_CONTROLS) == -1, "the table answers -1 before the device exists (hle_nvn.c rq_on: Auto off then)");
    g_step = "mtl_init"; if (getenv("SIMDEV_ONE_INIT")) mtl_init(); else init_race();
    device_read();
    const char *profile = getenv("SIMDEV_PROFILE");
    decisions_and_checks(profile);
    quirk_checks();
    ds_simgap();
    g_step = "work"; if (!getenv("SIMDEV_NO_WORK")) { work(); ring_work(); zcopy_work(); inttex_work(); shadow_work(); g_step = "b10"; b10_work(); g_step = "mali check"; mw_work(); g_step = "uniform snapshots"; snap_work(); g_step = "render area"; ra_work(); g_step = "clears"; lc_work(); g_step = "depth stores"; ds_work(); g_step = "compose"; compose_work();
                                                   g_step = "memlife"; memlife_work(); g_step = "shmod"; g_shmod_idle_ns = 1000000000ull; shmod_work(); g_shmod_idle_ns = 60000000000ull;
                                                   g_step = "prewarm"; prewarm_work(); }
    printf("%s t_vkdev %s: %d of %d checks failed\n", fails ? "FAIL" : "PASS", profile && *profile ? profile : "(no profile)", fails, checks);
    fflush(stdout);
    _exit(fails ? 1 : 0);   // (the engine's threads keep running; the device is not torn down)
}
