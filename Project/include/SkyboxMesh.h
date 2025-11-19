#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>

class SkyboxMesh
{
public:
    SkyboxMesh() = default;
    ~SkyboxMesh() = default;

    void create(VkDevice device,
        VkPhysicalDevice physicalDevice,
        VkCommandPool commandPool,
        VkQueue graphicsQueue);

    void destroy(VkDevice device);

    void draw(VkCommandBuffer cmd) const;

private:
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    uint32_t indexCount = 0;

private:
    void createCubeData(std::vector<float>& v,
        std::vector<uint32_t>& i);

    void createBuffer(VkDevice device,
        VkPhysicalDevice physicalDevice,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& memory);
};
