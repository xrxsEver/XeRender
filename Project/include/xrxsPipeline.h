#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include "Shader2D.h"

class xrxsPipeline {
public:
    xrxsPipeline();
    ~xrxsPipeline();

    void initialize(VkPhysicalDevice physicalDevice, VkDevice device, VkRenderPass renderPass, Shader2D& shader2D);

    VkPipeline getPipeline() const { return graphicsPipeline; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }

private:
    VkPipeline graphicsPipeline;
    VkPipelineLayout pipelineLayout;

    VkPipelineShaderStageCreateInfo createShaderStageInfo(VkShaderModule shaderModule, VkShaderStageFlagBits stage);
    VkPipelineVertexInputStateCreateInfo createVertexInputStateInfo();
    VkPipelineInputAssemblyStateCreateInfo createInputAssemblyStateInfo();
    VkPipelineViewportStateCreateInfo createViewportStateInfo();
    VkPipelineRasterizationStateCreateInfo createRasterizationStateInfo();
    VkPipelineMultisampleStateCreateInfo createMultisampleStateInfo();
    VkPipelineColorBlendStateCreateInfo createColorBlendStateInfo();
    VkPipelineDynamicStateCreateInfo createDynamicStateInfo();
    VkPushConstantRange createPushConstantRange();
};
