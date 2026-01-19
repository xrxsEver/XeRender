#pragma once

#include <vector>
#include <atomic>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "Vertex.h"
#include "VulkanUtil.h"

class WaterMesh
{
public:
    WaterMesh() = default;
    ~WaterMesh() = default;

    void create(VkDevice device,
                VkPhysicalDevice gpu,
                VkCommandPool commandPool,
                VkQueue graphicsQueue,
                int resolution = 256,
                float worldSize = 200.0f);

    void destroy(VkDevice device);

    void draw(VkCommandBuffer cmd);

    void setValid(bool valid) { isValid.store(valid, std::memory_order_release); }
    bool getValid() const { return isValid.load(std::memory_order_acquire); }

    uint32_t getIndexCount() const { return indexCount; }

private:
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    uint32_t indexCount = 0;
    std::atomic_bool isValid = ATOMIC_VAR_INIT(false);
};
