#pragma once
#include "vulkan/vulkan_core.h"
#include "CommandBuffer.h"
#include "VulkanUtil.h" 

class CommandPool {
public:
    // Constructors
    CommandPool() : m_CommandPool(VK_NULL_HANDLE), m_VkDevice(VK_NULL_HANDLE) {}
    CommandPool(const CommandPool&) = default; // Copy constructor

    void create(VkDevice device, uint32_t queueFamilyIndex);

    // Methods
    void initialize(const VkDevice& device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
    void destroy();
    CommandBuffer createCommandBuffer() const;
    VkCommandPool getVkCommandPool() const { return m_CommandPool; }
private:
    VkCommandPool m_CommandPool;
    VkDevice m_VkDevice;
};
