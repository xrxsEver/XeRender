#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include "Vertex.h"
#include "VulkanUtil.h"

class UnderwaterWaterPipeline {
public:
    UnderwaterWaterPipeline() = default;
    ~UnderwaterWaterPipeline() = default;

    // create: device, swap extent, renderPass, global descriptor set layout, water descriptor set layout, msaa samples
    void create(
        VkDevice device,
        VkExtent2D extent,
        VkRenderPass renderPass,
        VkDescriptorSetLayout globalDescriptorSetLayout,
        VkDescriptorSetLayout waterDescriptorSetLayout,
        VkSampleCountFlagBits msaaSamples,
        bool isSunraysPipeline 
    );

    void destroy(VkDevice device);

    void bind(VkCommandBuffer cmd);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

private:
    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
};


