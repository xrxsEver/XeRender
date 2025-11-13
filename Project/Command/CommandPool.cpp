#include "CommandPool.h"
#include "CommandPool.h"

void CommandPool::initialize(const VkDevice& device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    VkUtils::QueueFamilyIndices indices = VkUtils::FindQueueFamilies(physicalDevice, surface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = indices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_CommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    m_VkDevice = device;
}

CommandBuffer CommandPool::createCommandBuffer() const {
    CommandBuffer cmdBuffer;
    cmdBuffer.initialize(m_VkDevice, m_CommandPool);
    return cmdBuffer;
}

void CommandPool::create(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_CommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void CommandPool::destroy() {
    if (m_VkDevice != VK_NULL_HANDLE && m_CommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_VkDevice, m_CommandPool, nullptr);
    }
}
