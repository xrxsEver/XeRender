#pragma once

#include <vector>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "Vertex.h"      
#include "VulkanUtil.h"  

class OceanBottomMesh
{
public:
    OceanBottomMesh() = default;
    ~OceanBottomMesh() = default;

    void create(VkDevice device,
        VkPhysicalDevice gpu,
        VkCommandPool commandPool,
        VkQueue graphicsQueue,
        int resolution = 256,
        float worldSize = 200.0f,
        float depth = -50.0f);

    void destroy(VkDevice device);

    void draw(VkCommandBuffer cmd);

    uint32_t getIndexCount() const { return indexCount; }

private:
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    uint32_t indexCount = 0;
};


