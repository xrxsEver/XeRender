#include "SkyboxMesh.h"
#include "VulkanUtil.h"   // for VkUtils::CreateBuffer and CopyBuffer

void SkyboxMesh::create(VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkCommandPool commandPool,
    VkQueue graphicsQueue)
{
    std::vector<float> vertices;
    std::vector<uint32_t> indices;

    createCubeData(vertices, indices);
    indexCount = static_cast<uint32_t>(indices.size());
    std::cout << "[SkyboxMesh] vertices = " << vertices.size()
        << ", indices = " << indices.size() << std::endl;

    // ---- Vertex Buffer ----
    VkDeviceSize vSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer stagingVB;
    VkDeviceMemory stagingVBMem;

    std::tie(stagingVB, stagingVBMem) = VkUtils::CreateBuffer(  //crash here 
        device, physicalDevice, vSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(device, stagingVBMem, 0, vSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)vSize);
    vkUnmapMemory(device, stagingVBMem);

    std::tie(vertexBuffer, vertexMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, vSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkUtils::CopyBuffer(stagingVB, vertexBuffer, vSize, device, commandPool, graphicsQueue);

    vkDestroyBuffer(device, stagingVB, nullptr);
    vkFreeMemory(device, stagingVBMem, nullptr);

    // ---- Index Buffer ----
    VkDeviceSize iSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingIB;
    VkDeviceMemory stagingIBMem;

    std::tie(stagingIB, stagingIBMem) = VkUtils::CreateBuffer(
        device, physicalDevice, iSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkMapMemory(device, stagingIBMem, 0, iSize, 0, &data);
    memcpy(data, indices.data(), (size_t)iSize);
    vkUnmapMemory(device, stagingIBMem);

    std::tie(indexBuffer, indexMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, iSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkUtils::CopyBuffer(stagingIB, indexBuffer, iSize, device, commandPool, graphicsQueue);

    vkDestroyBuffer(device, stagingIB, nullptr);
    vkFreeMemory(device, stagingIBMem, nullptr);
}

void SkyboxMesh::destroy(VkDevice device)
{
    if (vertexBuffer) vkDestroyBuffer(device, vertexBuffer, nullptr);
    if (vertexMemory) vkFreeMemory(device, vertexMemory, nullptr);

    if (indexBuffer) vkDestroyBuffer(device, indexBuffer, nullptr);
    if (indexMemory) vkFreeMemory(device, indexMemory, nullptr);

    vertexBuffer = indexBuffer = VK_NULL_HANDLE;
    vertexMemory = indexMemory = VK_NULL_HANDLE;
}

void SkyboxMesh::draw(VkCommandBuffer cmd) const
{
    VkBuffer vb[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };

    vkCmdBindVertexBuffers(cmd, 0, 1, vb, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}

void SkyboxMesh::createCubeData(std::vector<float>& v, std::vector<uint32_t>& i)
{
    v = {
        -1, -1,  1,  // 0
         1, -1,  1,  // 1
         1,  1,  1,  // 2
        -1,  1,  1,  // 3
        -1, -1, -1,  // 4
         1, -1, -1,  // 5
         1,  1, -1,  // 6
        -1,  1, -1   // 7
    };

    i = {
        0,1,2, 2,3,0,   // front
        1,5,6, 6,2,1,   // right
        5,4,7, 7,6,5,   // back
        4,0,3, 3,7,4,   // left
        3,2,6, 6,7,3,   // top
        4,5,1, 1,0,4    // bottom
    };
}


void SkyboxMesh::createBuffer(VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& memory)
{
    std::tie(buffer, memory) =
        VkUtils::CreateBuffer(device, physicalDevice, size, usage, properties);
}
