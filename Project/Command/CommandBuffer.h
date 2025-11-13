#pragma once
#include <vulkan/vulkan.h>

class CommandBuffer {
public:
    void initialize(VkDevice device, VkCommandPool commandPool);
    void reset();
    void begin(const VkCommandBufferBeginInfo* beginInfo);
    void end();
    void beginRenderPass(const VkRenderPassBeginInfo& renderPassInfo, VkSubpassContents contents);
    void endRenderPass();
    VkCommandBuffer getVkCommandBuffer() const;

private:
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
};
