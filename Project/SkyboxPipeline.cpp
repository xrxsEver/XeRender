#include "SkyboxPipeline.h"
#include <fstream>
#include <vector>

VkShaderModule SkyboxPipeline::loadShader(VkDevice device, const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader file: " + path);

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = buffer.size();
    info.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shader;
    if (vkCreateShaderModule(device, &info, nullptr, &shader) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module.");

    return shader;
}

void SkyboxPipeline::create(VkDevice device,
    VkExtent2D extent,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalDescriptorSetLayout,  // set 0
    VkDescriptorSetLayout skyboxDescriptorSetLayout,  // set 1
    VkSampleCountFlagBits msaaSamples)
{
    // ---- LOAD SHADERS ----
    vertShader = loadShader(device, "shaders/skybox.vert.spv");
    fragShader = loadShader(device, "shaders/skybox.frag.spv");

    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = vertShader;
    vs.pName = "main";

    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = fragShader;
    fs.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vs, fs };

    // ---- Vertex input (positions only: vec3 at location 0) ----
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(float) * 3; // vec3 position
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDesc{};
    attrDesc.binding = 0;
    attrDesc.location = 0;
    attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDesc.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attrDesc;

    // ---- Input Assembly ----
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly.primitiveRestartEnable = VK_FALSE;

    // ---- Viewport & Scissor ----
    VkViewport viewport{};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = (float)extent.width;
    viewport.height = (float)extent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // ---- Rasterizer ----
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.depthClampEnable = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE; // view cube from inside
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = VK_FALSE;
    raster.lineWidth = 1.0f;

    // ---- Multisampling ----
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = msaaSamples;
    ms.sampleShadingEnable = VK_FALSE;
    ms.minSampleShading = 1.0f;
    ms.pSampleMask = nullptr;
    ms.alphaToCoverageEnable = VK_FALSE;
    ms.alphaToOneEnable = VK_FALSE;

    // ---- Depth Stencil ----
    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE; // skybox shouldn't write depth
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth.depthBoundsTestEnable = VK_FALSE;
    depth.stencilTestEnable = VK_FALSE;

    // ---- Color Blend ----
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.logicOpEnable = VK_FALSE;
    blend.logicOp = VK_LOGIC_OP_COPY;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    blend.blendConstants[0] = 0.0f;
    blend.blendConstants[1] = 0.0f;
    blend.blendConstants[2] = 0.0f;
    blend.blendConstants[3] = 0.0f;

    // ---- Pipeline Layout (both set 0 and set 1) ----
  // Use a plain C-array to avoid any <array> compile issues and add runtime checks.
    if (globalDescriptorSetLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("SkyboxPipeline::create - globalDescriptorSetLayout is VK_NULL_HANDLE");
    }
    if (skyboxDescriptorSetLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("SkyboxPipeline::create - skyboxDescriptorSetLayout is VK_NULL_HANDLE");
    }

    VkDescriptorSetLayout setLayoutsArr[2];
    setLayoutsArr[0] = globalDescriptorSetLayout;
    setLayoutsArr[1] = skyboxDescriptorSetLayout;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(float); // skyboxScale

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayoutsArr;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create skybox pipeline layout!");


    // ---- Graphics Pipeline ----
    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = shaderStages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &ms;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.layout = layout;
    info.renderPass = renderPass;
    info.subpass = 0;
    info.basePipelineHandle = VK_NULL_HANDLE;
    info.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create skybox pipeline!");
}



void SkyboxPipeline::bind(VkCommandBuffer cmd) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
}

void SkyboxPipeline::destroy(VkDevice device)
{
    if (pipeline)       vkDestroyPipeline(device, pipeline, nullptr);
    if (layout)         vkDestroyPipelineLayout(device, layout, nullptr);
    if (vertShader)     vkDestroyShaderModule(device, vertShader, nullptr);
    if (fragShader)     vkDestroyShaderModule(device, fragShader, nullptr);

    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    vertShader = fragShader = VK_NULL_HANDLE;
}
