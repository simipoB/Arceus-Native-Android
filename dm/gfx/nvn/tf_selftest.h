// Transform feedback self-test (included by nvn_vk.c; run once by mtl_init after the device is made).
//
// Legends: Arceus skins its characters and sways its trees with transform feedback, which the engine captures with the vertex shader storing its outputs into a
// buffer, drawn with rasterizer discard in a 1x1 pass of its own with no attachments (tf_pass_begin). On Arm's own Mali driver (r44p1, Mali-G615) the player,
// the Pokémon, NPCs and swaying trees are not drawn, while PanVK on the same phones draws them (Arceus-Native-Android#1 #4 #6 #10 #11 #13), and the engine
// logs nothing wrong. This runs that kind of capture on the device with a tiny shader of its own (tf_selftest_spv.h: o[i] = 3i + 1 for 96 vertices) and reads
// the buffer back, three ways:
//   A  as the engine does it: no attachments, rasterizer discard
//   B  the same with a 1x1 colour attachment (a tiler driver may drop the work of a pass with nothing to render into)
//   C  a compute shader (the control: storage writes as such)
// and logs how many of the 96 each wrote. When A writes nothing and B writes them all, capture passes get that 1x1 colour attachment from now on (g_tf_att;
// DM_VK_TF_ATT=0 / 1 decides on any driver). On Arm's own driver, or anywhere with DM_VK_TF_SELFTEST=1. A start that dies inside the test leaves a marker and
// later starts on that driver skip it (DM_VK_TF_SELFTEST=1 runs it again).
#include "tf_selftest_spv.h"

static int g_tf_att = -1;                   // capture passes with a 1x1 colour attachment (tf_pass_begin): -1 not decided yet = no
static VkImage g_tf_img; static VkImageView g_tf_view; static VkDeviceMemory g_tf_imem;   // that attachment (made by the test, kept for the passes)
#define TF_ATT_FMT VK_FORMAT_R8G8B8A8_UNORM

static int tfst_img(void) {   // the 1x1 colour attachment, in COLOR_ATTACHMENT_OPTIMAL once tfst_run's command buffer has run
    if (g_tf_view) return 1;
    VkImageCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = TF_ATT_FMT, .extent = { 1, 1, 1 }, .mipLevels = 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    if (vkCreateImage(g_dev, &ic, NULL, &g_tf_img)) return 0;
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(g_dev, g_tf_img, &rq); int t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = (uint32_t)(t < 0 ? 0 : t) };
    if (t < 0 || vkAllocateMemory(g_dev, &ai, NULL, &g_tf_imem) || vkBindImageMemory(g_dev, g_tf_img, g_tf_imem, 0)) { vkDestroyImage(g_dev, g_tf_img, NULL); g_tf_img = VK_NULL_HANDLE; return 0; }
    VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_tf_img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = TF_ATT_FMT,
                                 .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    if (vkCreateImageView(g_dev, &vi, NULL, &g_tf_view)) { g_tf_view = VK_NULL_HANDLE; return 0; }
    return 1;
}

// One way (0 = A, 1 = B, 2 = C): how many of the 96 words hold 3i + 1 afterwards; -1 when it could not be set up
static int tfst_run(int way, VkBuffer buf, const uint32_t *map, VkDescriptorSetLayout dsl, VkPipelineLayout pl, VkDescriptorSet ds, VkCommandPool cp) {
    enum { N = 96 };
    VkPipeline p = VK_NULL_HANDLE; int r = -1;
    if (way == 2) {
        VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof tfst_cs, .pCode = tfst_cs }; VkShaderModule m;
        if (vkCreateShaderModule(g_dev, &mi, NULL, &m)) return -1;
        VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main" }, .layout = pl };
        VkResult e = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &p); vkDestroyShaderModule(g_dev, m, NULL); if (e) return -1;
    } else {
        if (way == 1 && !tfst_img()) return -1;
        VkShaderModuleCreateInfo mi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof tfst_vs, .pCode = tfst_vs }; VkShaderModule m;
        if (vkCreateShaderModule(g_dev, &mi, NULL, &m)) return -1;
        VkPipelineShaderStageCreateInfo st = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = m, .pName = "main" };
        VkPipelineVertexInputStateCreateInfo vin = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
        VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .rasterizerDiscardEnable = VK_TRUE, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
        VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
        VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = way == 1, .pAttachments = &cba };
        // the dynamic state the engine's own capture pipelines have (pso_make): viewport and scissor; with extended dynamic state cull mode, front face and topology too
        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY };
        VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = g_eds ? 5 : 2, .pDynamicStates = dyn };
        VkFormat cf = TF_ATT_FMT;
        VkPipelineRenderingCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = way == 1, .pColorAttachmentFormats = &cf };
        VkGraphicsPipelineCreateInfo gi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 1, .pStages = &st, .pVertexInputState = &vin,
            .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pl };
        if (!g_dynren) { gi.pNext = NULL; gi.renderPass = rp_for_pipeline(way == 1, &cf, 0, 1, 0); gi.subpass = 0; }
        VkResult e = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &p); vkDestroyShaderModule(g_dev, m, NULL); if (e) return -1;
    }
    VkCommandBuffer cmd; VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    if (vkAllocateCommandBuffers(g_dev, &cai, &cmd)) { vkDestroyPipeline(g_dev, p, NULL); return -1; }
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdFillBuffer(cmd, buf, 0, N * 4, 0x7fc0deadu);   // the marker: still there = not written
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (way == 1) {   // the attachment into COLOR_ATTACHMENT_OPTIMAL (kept so: the capture passes use it as it is, contents never read)
        VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = g_tf_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
    if (way == 2) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, N / 64 + (N % 64 != 0), 1, 1);
    } else {
        VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_tf_view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE };
        VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {1, 1}}, .layerCount = 1, .colorAttachmentCount = way == 1, .pColorAttachments = &ca };
        uint32_t cf = TF_ATT_FMT; rend_begin(cmd, &ri, &cf, 0, 1, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
        VkViewport v = { 0, 0, 1, 1, 0, 1 }; VkRect2D sc = {{0, 0}, {1, 1}}; vkCmdSetViewport(cmd, 0, 1, &v); vkCmdSetScissor(cmd, 0, 1, &sc);
        if (g_eds) { vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE); vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE); vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST); }
        vkCmdDraw(cmd, N, 1, 0, 0);
        rend_end(cmd);
    }
    VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cmd);
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f;
    if (!vkCreateFence(g_dev, &fi, NULL, &f)) {
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
        pthread_mutex_lock(&g_qmu); VkResult e = vkQueueSubmit(g_q, 1, &si, f); pthread_mutex_unlock(&g_qmu);
        if (!e && vkWaitForFences(g_dev, 1, &f, VK_TRUE, 2000000000ull) == VK_SUCCESS) { r = 0; for (int i = 0; i < N; i++) r += map[i] == 3u * (uint32_t)i + 1u; }
        vkDestroyFence(g_dev, f, NULL);
    }
    vkFreeCommandBuffers(g_dev, cp, 1, &cmd); vkDestroyPipeline(g_dev, p, NULL);
    return r;
}

static void tf_selftest(void) {
    const char *e = getenv("DM_VK_TF_ATT"); if (e && *e) { g_tf_att = atoi(e) != 0; VLOG("transform feedback captures: %s (DM_VK_TF_ATT)", g_tf_att ? "with a 1x1 colour attachment" : "in passes with no attachments"); }
    const char *on = getenv("DM_VK_TF_SELFTEST"); int run = on ? atoi(on) != 0 : g_arm_rel > 0;
    if (!run) return;
    if (!on && mark_file(drv_mark("tf_selftest_crashed"), "", 0)) { VLOG("[RED] transform feedback self-test: the driver crashed in it at an earlier start: skipped (DM_VK_TF_SELFTEST=1 runs it again)"); return; }
    mark_file(drv_mark("tf_selftest_crashed"), "running the transform feedback self-test; if this file is still here the driver crashed in it, and later starts skip it\n", 1);
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void *map = NULL; int a = -1, b = -1, c = -1;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkDescriptorPool dp = VK_NULL_HANDLE; VkDescriptorSet ds = VK_NULL_HANDLE; VkCommandPool cp = VK_NULL_HANDLE;
    if (vkCreateBuffer(g_dev, &bci, NULL, &buf)) goto out;
    { VkMemoryRequirements rq; vkGetBufferMemoryRequirements(g_dev, buf, &rq);
      int t = mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0); if (t < 0) goto out;
      VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = (uint32_t)t };
      if (vkAllocateMemory(g_dev, &ai, NULL, &mem) || vkBindBufferMemory(g_dev, buf, mem, 0) || vkMapMemory(g_dev, mem, 0, VK_WHOLE_SIZE, 0, &map)) goto out; }
    { VkDescriptorSetLayoutBinding lb = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, NULL };
      VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &lb };
      if (vkCreateDescriptorSetLayout(g_dev, &li, NULL, &dsl)) goto out;
      VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl };
      if (vkCreatePipelineLayout(g_dev, &pi, NULL, &pl)) goto out;
      VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
      VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
      if (vkCreateDescriptorPool(g_dev, &dpi, NULL, &dp)) goto out;
      VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
      if (vkAllocateDescriptorSets(g_dev, &dai, &ds)) goto out;
      VkDescriptorBufferInfo dbi = { buf, 0, 4096 };
      VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
      vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL);
      VkCommandPoolCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = g_qf };
      if (vkCreateCommandPool(g_dev, &ci, NULL, &cp)) goto out; }
    a = tfst_run(0, buf, map, dsl, pl, ds, cp); b = tfst_run(1, buf, map, dsl, pl, ds, cp); c = tfst_run(2, buf, map, dsl, pl, ds, cp);
out:
    if (cp) vkDestroyCommandPool(g_dev, cp, NULL); if (dp) vkDestroyDescriptorPool(g_dev, dp, NULL); if (pl) vkDestroyPipelineLayout(g_dev, pl, NULL);
    if (dsl) vkDestroyDescriptorSetLayout(g_dev, dsl, NULL); if (buf) vkDestroyBuffer(g_dev, buf, NULL); if (mem) vkFreeMemory(g_dev, mem, NULL);
    mark_file(drv_mark("tf_selftest_crashed"), "", -1);
#define TFST_S(x) ((x) < 0 ? "could not be set up" : (x) == 96 ? "all 96 written" : (x) == 0 ? "[RED] NOTHING written" : "[RED] only some written")
    VLOG("transform feedback self-test: vertex-shader stores in a pass with no attachments (as the captures run): %s (%d); with a 1x1 colour attachment: %s (%d); compute: %s (%d)",
         TFST_S(a), a, TFST_S(b), b, TFST_S(c), c);
    if (g_tf_att < 0) { g_tf_att = a != 96 && b == 96 && g_tf_view;
        if (g_tf_att) VLOG("transform feedback captures: with a 1x1 colour attachment from now on (this driver drops them in a pass with nothing attached; DM_VK_TF_ATT=0 puts them back)"); }
}
