// Transform feedback captures as points (included by nvn_vk.c after tf_selftest.h): for drivers whose vertex shaders cannot store into buffers.
//
// Legends: Arceus skins its characters and sways its trees with transform feedback. The engine captures it with the vertex shader storing its outputs into buffers
// (spv_tf_variant), which needs vertexPipelineStoresAndAtomics: Arm's own Mali driver lacks it (Mali-G610, r42) or writes nothing (the self-test's legs A and B), and the
// player, Pokemon and NPCs vanish (Arceus-Native-Android #47 #51 #54 #57 #58), while its compute shaders do write (leg C). Here each captured vertex is drawn instead as a
// point on a pixel of its own in TFP_W-wide RGBA32UI colour targets, one per capture buffer, the outputs as their bits (spv_tf_points_variant, tf_points.frag), and a
// compute shader copies each target into its capture buffer at the right stride and offset (tf_repack.comp). Decided at the start (tfp_decide): on when the driver has no
// vertex stores, or the self-test's A and B wrote nothing while C wrote all; DM_VK_TF_POINTS=0 / 1 decides on any GPU (1 on Adreno to compare with the store route).
#include "tf_points_spv.h"
static VkDescriptorSet desc_set_alloc(VkDescriptorSetLayout dsl);   // (nvn_vk.c: a set from the submission's pools, without push descriptors)

static VkImage g_tfp_img[4]; static VkImageView g_tfp_view[4]; static VkDeviceMemory g_tfp_mem[4];
static VkShaderModule g_tfp_csm; static VkDescriptorSetLayout g_tfp_dsl; static VkPipelineLayout g_tfp_pl; static VkPipeline g_tfp_cs; static int g_tfp_ready;

static void tfp_decide(void) {
    const char *e = getenv("DM_VK_TF_POINTS");
    if (e && *e) g_tf_pts = atoi(e) > 0;
    else g_tf_pts = !g_vstores || (g_tfst_a >= 0 && g_tfst_a != 96 && g_tfst_b != 96 && g_tfst_c == 96);
    if (g_tf_pts) VLOG("transform feedback captures: as points into colour targets, copied by a compute shader (%s)", e && *e ? "DM_VK_TF_POINTS" :
                       !g_vstores ? "this driver has no vertex shader stores" : "the self-test: vertex shader stores wrote nothing, compute did");
}
static int tfp_init(void) {
    if (g_tfp_ready) return g_tfp_ready > 0;
    g_tfp_ready = -1;
    for (int i = 0; i < 4; i++) {
        VkImageCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_UINT, .extent = { TFP_W, TFP_H, 1 },
            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
        if (vkCreateImage(g_dev, &ic, NULL, &g_tfp_img[i])) goto fail;
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, g_tfp_img[i], &rq); int t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = (uint32_t)(t < 0 ? 0 : t) };
        if (t < 0 || vkAllocateMemory(g_dev, &ai, NULL, &g_tfp_mem[i]) || vkBindImageMemory(g_dev, g_tfp_img[i], g_tfp_mem[i], 0)) goto fail;
        VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_tfp_img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_UINT,
                                     .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        if (vkCreateImageView(g_dev, &vi, NULL, &g_tfp_view[i])) goto fail;
    }
    { const uint32_t *fs[4] = { tf_pts1_spv, tf_pts2_spv, tf_pts3_spv, tf_pts4_spv }; size_t fl[4] = { sizeof tf_pts1_spv, sizeof tf_pts2_spv, sizeof tf_pts3_spv, sizeof tf_pts4_spv };
      for (int i = 0; i < 4; i++) { VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = fl[i], .pCode = fs[i] };
          if (vkCreateShaderModule(g_dev, &mi, NULL, &g_tfp_fs[i])) goto fail; } }
    { VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof tf_repack_spv, .pCode = tf_repack_spv };
      if (vkCreateShaderModule(g_dev, &mi, NULL, &g_tfp_csm)) goto fail;
      VkDescriptorSetLayoutBinding lb[2] = { { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
      VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = g_nopush ? 0 : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                            .bindingCount = 2, .pBindings = lb };
      if (vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_tfp_dsl)) goto fail;
      VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
      VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_tfp_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
      if (vkCreatePipelineLayout(g_dev, &pi, NULL, &g_tfp_pl)) goto fail;
      VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_tfp_csm, .pName = "main" }, .layout = g_tfp_pl };
      if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_tfp_cs)) goto fail; }
    g_tfp_ready = 1; VLOG("transform feedback captures as points: %d x %d targets ready", TFP_W, TFP_H);
    return 1;
fail:
    VLOG("[RED] transform feedback captures as points: the targets or shaders could not be made: nothing captured");
    return 0;
}
// The capture's pass: its n targets, rows x TFP_W of them; what was in them is overwritten, after the last repack has read it
static void tfp_pass_begin(int n, uint32_t rows) {
    pass_end(); if (g_nlc) lc_out_all(); memset(&P, 0, sizeof P); P_tf = 1; g_bar_clean = g_after_pass = 0;   // (a kept depth clear written out first: nvn_vk.c lc_out_all)
    VkCommandBuffer cb = CB(); VkImageMemoryBarrier ib[4]; VkRenderingAttachmentInfo ca[4]; uint32_t cf[4];
    for (int b = 0; b < n; b++) {
        ib[b] = (VkImageMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_READ_BIT, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = g_tfp_img[b], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        ca[b] = (VkRenderingAttachmentInfo){ .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_tfp_view[b], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
        cf[b] = VK_FORMAT_R32G32B32A32_UINT; P.cfmt[b] = VK_FORMAT_R32G32B32A32_UINT; }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, (uint32_t)n, ib);
    VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {TFP_W, rows}}, .layerCount = 1, .colorAttachmentCount = (uint32_t)n, .pColorAttachments = ca };
    rend_begin(cb, &ri, cf, 0, 1, 0); P.open = 1; P.ncolor = n; P.tw = TFP_W; P.th = rows; P.samples = 1;
}
// After the draw (its pass ended): each target into its capture buffer (count vertices, nc[b] words each, from word base[b] of the bound range), then visible to what reads them
static void tfp_repack(VkCommandBuffer cb, int n, uint32_t count, const VkBuffer *tfb, const VkDeviceSize *tfo, const VkDeviceSize *tfr, const uint32_t *base, const int *nc) {
    VkImageMemoryBarrier ib[4];
    for (int b = 0; b < n; b++) ib[b] = (VkImageMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT, .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_tfp_img[b], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, (uint32_t)n, ib);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_tfp_cs);
    for (int b = 0; b < n; b++) { if (!tfb[b]) continue;
        VkDescriptorImageInfo ii = { VK_NULL_HANDLE, g_tfp_view[b], VK_IMAGE_LAYOUT_GENERAL }; VkDescriptorBufferInfo bi = { tfb[b], tfo[b], tfr[b] };
        VkWriteDescriptorSet w[2] = { { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii },
                                      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi } };
        if (!g_nopush) p_push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_tfp_pl, 0, 2, w);
        else { VkDescriptorSet ds = desc_set_alloc(g_tfp_dsl); if (!ds) continue; w[0].dstSet = w[1].dstSet = ds; vkUpdateDescriptorSets(g_dev, 2, w, 0, NULL);
               vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_tfp_pl, 0, 1, &ds, 0, NULL); }
        uint32_t pc[4] = { count, TFP_W, (uint32_t)nc[b], base[b] };
        vkCmdPushConstants(cb, g_tfp_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, (count + 63) / 64, 1, 1); }
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
