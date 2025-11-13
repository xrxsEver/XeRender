#include "CommandBuffer.h"

void CommandBuffer::initialize(VkDevice device, VkCommandPool commandPool) {
    this->device = device;
    this->commandPool = commandPool;

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
}

void CommandBuffer::reset() {
    vkResetCommandBuffer(commandBuffer, 0);
}

void CommandBuffer::begin(const VkCommandBufferBeginInfo* beginInfo) {
    vkBeginCommandBuffer(commandBuffer, beginInfo);
}

void CommandBuffer::end() {
    vkEndCommandBuffer(commandBuffer);
}

void CommandBuffer::beginRenderPass(const VkRenderPassBeginInfo& renderPassInfo, VkSubpassContents contents) {
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, contents);
}

void CommandBuffer::endRenderPass() {
    vkCmdEndRenderPass(commandBuffer);
}

VkCommandBuffer CommandBuffer::getVkCommandBuffer() const {
    return commandBuffer;
}
