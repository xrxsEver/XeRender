#include "WaterMesh.h"
#include <stdexcept>

void WaterMesh::create(VkDevice device,
    VkPhysicalDevice gpu,
    VkCommandPool commandPool,
    VkQueue graphicsQueue,
    int resolution,
    float worldSize)
{
    const int N = resolution;
    const float half = worldSize * 0.5f;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve((N + 1) * (N + 1));

    // -------------------------
    // Generate vertices
    // -------------------------
    for (int y = 0; y <= N; y++)
    {
        for (int x = 0; x <= N; x++)
        {
            float px = ((float)x / N) * worldSize - half;
            float pz = ((float)y / N) * worldSize - half;

            Vertex v{};
            v.pos = { px, 0.0f, pz };
            v.normal = { 0.0f, 1.0f, 0.0f };
            v.texCoord = { x / float(N), y / float(N) };
            v.color = { 1.0f, 1.0f, 1.0f };
            v.tangent = { 1.0f, 0.0f, 0.0f };
            v.bitangent = { 0.0f, 0.0f, 1.0f };

            vertices.push_back(v);
        }
    }

    // -------------------------
    // Generate indices
    // -------------------------
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            uint32_t i0 = y * (N + 1) + x;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + (N + 1);
            uint32_t i3 = i2 + 1;

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }

    indexCount = indices.size();

    // =============================================================
    // CREATE VERTEX BUFFER (WITH STAGING)
    // =============================================================

    VkDeviceSize vertexSize = vertices.size() * sizeof(Vertex);

    // Create staging buffer
    auto [stagingVB, stagingVBMem] = VkUtils::CreateBuffer(
        device, gpu, vertexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Upload data
    void* data = nullptr;
    vkMapMemory(device, stagingVBMem, 0, vertexSize, 0, &data);
    memcpy(data, vertices.data(), vertexSize);
    vkUnmapMemory(device, stagingVBMem);

    // Create actual GPU buffer
    auto [vb, vbMem] = VkUtils::CreateBuffer(
        device, gpu, vertexSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    vertexBuffer = vb;
    vertexMemory = vbMem;

    // Copy Staging → GPU
    VkUtils::CopyBuffer(stagingVB, vertexBuffer, vertexSize, device, commandPool, graphicsQueue);

    // Clean up staging
    vkDestroyBuffer(device, stagingVB, nullptr);
    vkFreeMemory(device, stagingVBMem, nullptr);

    // =============================================================
    // CREATE INDEX BUFFER (WITH STAGING)
    // =============================================================

    VkDeviceSize indexSize = indices.size() * sizeof(uint32_t);

    auto [stagingIB, stagingIBMem] = VkUtils::CreateBuffer(
        device, gpu, indexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    vkMapMemory(device, stagingIBMem, 0, indexSize, 0, &data);
    memcpy(data, indices.data(), indexSize);
    vkUnmapMemory(device, stagingIBMem);

    auto [ib, ibMem] = VkUtils::CreateBuffer(
        device, gpu, indexSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    indexBuffer = ib;
    indexMemory = ibMem;

    VkUtils::CopyBuffer(stagingIB, indexBuffer, indexSize, device, commandPool, graphicsQueue);

    vkDestroyBuffer(device, stagingIB, nullptr);
    vkFreeMemory(device, stagingIBMem, nullptr);
}

void WaterMesh::destroy(VkDevice device)
{
    if (vertexBuffer) vkDestroyBuffer(device, vertexBuffer, nullptr);
    if (vertexMemory) vkFreeMemory(device, vertexMemory, nullptr);

    if (indexBuffer) vkDestroyBuffer(device, indexBuffer, nullptr);
    if (indexMemory) vkFreeMemory(device, indexMemory, nullptr);
}

void WaterMesh::draw(VkCommandBuffer cmd)
{
    VkBuffer buffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };

    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}
