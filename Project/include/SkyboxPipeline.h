#pragma once
#include <vulkan/vulkan.h>
#include <string>

class SkyboxPipeline
{
public:
    SkyboxPipeline() = default;
    ~SkyboxPipeline() = default;

    void create(VkDevice device,
        VkExtent2D swapchainExtent,
        VkRenderPass renderPass,
        VkDescriptorSetLayout globalDescriptorSetLayout,  // set 0
        VkDescriptorSetLayout skyboxDescriptorSetLayout,  // set 1
        VkSampleCountFlagBits msaaSamples); // new param

    void destroy(VkDevice device);

    void bind(VkCommandBuffer cmd) const;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

private:
    VkShaderModule vertShader = VK_NULL_HANDLE;
    VkShaderModule fragShader = VK_NULL_HANDLE;

private:
    VkShaderModule loadShader(VkDevice device, const std::string& path);
};
