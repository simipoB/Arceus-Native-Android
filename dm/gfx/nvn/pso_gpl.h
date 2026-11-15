// Graphics pipeline libraries (VK_EXT_graphics_pipeline_library; included by nvn_vk.c, used by pso_make).
//
// A game draws each shader pair with many states (blend, targets, vertex layout, depth), and every combination is a whole pipeline the driver compiles: the first time a
// new one is met in play (a zone, an effect) the frame waits 20-150 ms for it (more on Mali). With libraries a pipeline is four parts made once each and reused:
//   vertex input (layouts, topology) | pre-rasterisation (the vertex shader, viewport, rasterisation) | fragment shader (with depth / stencil, multisampling)
//   | fragment output (blend, target formats)
// and a new combination only links parts already made, which a driver with fast linking does in well under a millisecond. Opt-in (DM_VK_GPL=1) until measured on phones;
// only where the driver has the feature and reports fast linking (gpl_init). Anything that fails falls back to the whole pipeline as before (pso_make).
// Not used for captures (rasterizer discard, the point route) or for pipelines with more than a vertex and a fragment stage.

// (g_gpl, declared at the top of nvn_vk.c: set at device creation when DM_VK_GPL=1, the driver has the feature and fast linking)
static pthread_mutex_t g_gpl_mu = PTHREAD_MUTEX_INITIALIZER;
enum { GPL_N = 8192 };                               // parts kept per kind (a full table: parts made for one link and destroyed after it)
static struct { uint64_t key; VkPipeline p; } g_gpl_tab[4][GPL_N];
static uint64_t g_gpl_made[4], g_gpl_hit[4], g_gpl_links, g_gpl_fail;

static uint64_t gpl_h(uint64_t h, const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; } return h; }
#define GPL_HV(h, v) ((h) = gpl_h((h), &(v), sizeof(v)))

static VkPipeline gpl_find(int kind, uint64_t key) {
    for (uint32_t i = 0, s = (uint32_t)(key % GPL_N); i < GPL_N; i++, s = (s + 1) % GPL_N) {
        if (!g_gpl_tab[kind][s].p) return VK_NULL_HANDLE;
        if (g_gpl_tab[kind][s].key == key) return g_gpl_tab[kind][s].p;
    }
    return VK_NULL_HANDLE;
}
static int gpl_keep(int kind, uint64_t key, VkPipeline p) {   // 1 kept, 0 table full (the caller destroys it after the link)
    for (uint32_t i = 0, s = (uint32_t)(key % GPL_N); i < GPL_N / 2; i++, s = (s + 1) % GPL_N)
        if (!g_gpl_tab[kind][s].p) { g_gpl_tab[kind][s].key = key; g_gpl_tab[kind][s].p = p; return 1; }
    return 0;
}

// one part: gi's own sub-structures, only those the part reads
static VkPipeline gpl_part(int kind, const VkGraphicsPipelineCreateInfo *gi, VkPipelineCache pc, const VkPipelineShaderStageCreateInfo *st, uint32_t nst) {
    static const VkGraphicsPipelineLibraryFlagsEXT fl[4] = { VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
                                                             VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT };
    VkGraphicsPipelineLibraryCreateInfoEXT lci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, .pNext = (void *)gi->pNext, .flags = fl[kind] };   // (gi->pNext: the rendering info, or NULL with render passes)
    VkGraphicsPipelineCreateInfo c = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &lci, .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR, .pDynamicState = gi->pDynamicState };
    if (kind == 0) { c.pVertexInputState = gi->pVertexInputState; c.pInputAssemblyState = gi->pInputAssemblyState; }
    if (kind == 1) { c.stageCount = nst; c.pStages = st; c.pViewportState = gi->pViewportState; c.pRasterizationState = gi->pRasterizationState; c.layout = gi->layout; }
    if (kind == 2) { c.stageCount = nst; c.pStages = st; c.pDepthStencilState = gi->pDepthStencilState; c.pMultisampleState = gi->pMultisampleState; c.layout = gi->layout; }
    if (kind == 3) { c.pColorBlendState = gi->pColorBlendState; c.pMultisampleState = gi->pMultisampleState; }
    if (kind) { c.renderPass = gi->renderPass; c.subpass = gi->subpass; }
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_dev, pc, 1, &c, NULL, &p)) return VK_NULL_HANDLE;
    return p;
}

// the whole pipeline gi describes, linked from parts; VK_NULL_HANDLE: not done (the caller makes it whole)
static VkPipeline gpl_link(const VkGraphicsPipelineCreateInfo *gi, VkPipelineCache pc) {
    if (!g_gpl || gi->stageCount != 2 || gi->pRasterizationState->rasterizerDiscardEnable) return VK_NULL_HANDLE;
    const VkPipelineShaderStageCreateInfo *vs = NULL, *fs = NULL;
    for (uint32_t i = 0; i < gi->stageCount; i++) { if (gi->pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT) vs = &gi->pStages[i]; else if (gi->pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) fs = &gi->pStages[i]; }
    if (!vs || !fs || vs->pSpecializationInfo || fs->pSpecializationInfo) return VK_NULL_HANDLE;
    // the keys: everything each part reads (the dynamic state list and the rendering target description go into all four, as both reach every part)
    const VkPipelineDynamicStateCreateInfo *dy = gi->pDynamicState; const VkPipelineRenderingCreateInfo *rci = gi->pNext;
    uint64_t common = 0xcbf29ce484222325ull;
    common = gpl_h(common, dy->pDynamicStates, dy->dynamicStateCount * sizeof *dy->pDynamicStates); GPL_HV(common, gi->renderPass); GPL_HV(common, gi->subpass);
    if (rci) { GPL_HV(common, rci->viewMask); GPL_HV(common, rci->colorAttachmentCount); common = gpl_h(common, rci->pColorAttachmentFormats, rci->colorAttachmentCount * sizeof(VkFormat));
               GPL_HV(common, rci->depthAttachmentFormat); GPL_HV(common, rci->stencilAttachmentFormat); }
    uint64_t key[4] = { common ^ 0x11, common ^ 0x22, common ^ 0x33, common ^ 0x44 };
    { const VkPipelineVertexInputStateCreateInfo *vi = gi->pVertexInputState; const VkPipelineInputAssemblyStateCreateInfo *ia = gi->pInputAssemblyState;
      key[0] = gpl_h(key[0], vi->pVertexBindingDescriptions, vi->vertexBindingDescriptionCount * sizeof *vi->pVertexBindingDescriptions);
      key[0] = gpl_h(key[0], vi->pVertexAttributeDescriptions, vi->vertexAttributeDescriptionCount * sizeof *vi->pVertexAttributeDescriptions);
      if (vi->pNext) { const VkPipelineVertexInputDivisorStateCreateInfoEXT *d = vi->pNext; key[0] = gpl_h(key[0], d->pVertexBindingDivisors, d->vertexBindingDivisorCount * sizeof *d->pVertexBindingDivisors); }
      GPL_HV(key[0], ia->topology); GPL_HV(key[0], ia->primitiveRestartEnable); }
    { const VkPipelineRasterizationStateCreateInfo *rs = gi->pRasterizationState; const VkPipelineViewportStateCreateInfo *vp = gi->pViewportState;
      GPL_HV(key[1], vs->module); GPL_HV(key[1], gi->layout); GPL_HV(key[1], rs->depthClampEnable); GPL_HV(key[1], rs->polygonMode); GPL_HV(key[1], rs->cullMode);
      GPL_HV(key[1], rs->frontFace); GPL_HV(key[1], rs->depthBiasEnable); GPL_HV(key[1], rs->lineWidth);
      if (vp->pNext) { const VkPipelineViewportDepthClipControlCreateInfoEXT *dc = vp->pNext; GPL_HV(key[1], dc->negativeOneToOne); } }
    { const VkPipelineDepthStencilStateCreateInfo *ds = gi->pDepthStencilState; const VkPipelineMultisampleStateCreateInfo *ms = gi->pMultisampleState;
      GPL_HV(key[2], fs->module); GPL_HV(key[2], gi->layout); GPL_HV(key[2], ds->depthTestEnable); GPL_HV(key[2], ds->depthWriteEnable); GPL_HV(key[2], ds->depthCompareOp);
      GPL_HV(key[2], ds->stencilTestEnable); GPL_HV(key[2], ds->front); GPL_HV(key[2], ds->back); GPL_HV(key[2], ms->rasterizationSamples); GPL_HV(key[2], ms->alphaToCoverageEnable); }
    { const VkPipelineColorBlendStateCreateInfo *bs = gi->pColorBlendState; const VkPipelineMultisampleStateCreateInfo *ms = gi->pMultisampleState;
      GPL_HV(key[3], bs->logicOpEnable); GPL_HV(key[3], bs->logicOp); GPL_HV(key[3], bs->attachmentCount);
      key[3] = gpl_h(key[3], bs->pAttachments, bs->attachmentCount * sizeof *bs->pAttachments); GPL_HV(key[3], ms->rasterizationSamples); GPL_HV(key[3], ms->alphaToCoverageEnable); }
    VkPipeline part[4] = { 0 }; int temp[4] = { 0 };
    pthread_mutex_lock(&g_gpl_mu);
    for (int k = 0; k < 4; k++) { part[k] = gpl_find(k, key[k]); if (part[k]) g_gpl_hit[k]++; }
    pthread_mutex_unlock(&g_gpl_mu);
    // the missing parts, made outside the lock (another thread making the same one at once only wastes one: the second is kept or destroyed below)
    for (int k = 0; k < 4; k++) if (!part[k]) {
        part[k] = gpl_part(k, gi, pc, k == 1 ? vs : k == 2 ? fs : NULL, k == 1 || k == 2);
        if (!part[k]) goto fail;
        pthread_mutex_lock(&g_gpl_mu);
        VkPipeline had = gpl_find(k, key[k]);
        if (had) { pthread_mutex_unlock(&g_gpl_mu); vkDestroyPipeline(g_dev, part[k], NULL); part[k] = had; }
        else { temp[k] = !gpl_keep(k, key[k], part[k]); g_gpl_made[k]++; pthread_mutex_unlock(&g_gpl_mu); }
    }
    { VkPipelineLibraryCreateInfoKHR li = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4, .pLibraries = part };
      VkGraphicsPipelineCreateInfo c = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li, .layout = gi->layout };   // (fast linking: no link-time optimisation asked for)
      VkPipeline p = VK_NULL_HANDLE;
      VkResult r = vkCreateGraphicsPipelines(g_dev, pc, 1, &c, NULL, &p);
      for (int k = 0; k < 4; k++) if (temp[k] && part[k]) vkDestroyPipeline(g_dev, part[k], NULL);
      if (r || !p) { pthread_mutex_lock(&g_gpl_mu); g_gpl_fail++; pthread_mutex_unlock(&g_gpl_mu); RED_ONCE(62, "pipeline library: the link failed (%d): whole pipelines where it does", r); return VK_NULL_HANDLE; }
      pthread_mutex_lock(&g_gpl_mu); uint64_t n = ++g_gpl_links;
      if (n == 1 || n % 500 == 0) VLOG("pipeline libraries: %llu pipelines linked; parts made / reused: vertex input %llu / %llu, vertex %llu / %llu, fragment %llu / %llu, output %llu / %llu; %llu failed",
          (unsigned long long)n, (unsigned long long)g_gpl_made[0], (unsigned long long)g_gpl_hit[0], (unsigned long long)g_gpl_made[1], (unsigned long long)g_gpl_hit[1],
          (unsigned long long)g_gpl_made[2], (unsigned long long)g_gpl_hit[2], (unsigned long long)g_gpl_made[3], (unsigned long long)g_gpl_hit[3], (unsigned long long)g_gpl_fail);
      pthread_mutex_unlock(&g_gpl_mu);
      return p; }
fail:
    for (int k = 0; k < 4; k++) if (temp[k] && part[k]) vkDestroyPipeline(g_dev, part[k], NULL);
    pthread_mutex_lock(&g_gpl_mu); g_gpl_fail++; pthread_mutex_unlock(&g_gpl_mu);
    RED_ONCE(62, "pipeline library: a part could not be made: whole pipelines where that happens");
    return VK_NULL_HANDLE;
}
