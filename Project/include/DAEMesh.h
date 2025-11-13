#pragma once

#include "include/VulkanBase.h"
#include "DAEDataBuffer.h"
#include "Vertex.h"
#include "glm/glm.hpp"
#include <vector>
#include <memory>

class DAEMesh {
public:
    DAEMesh();
    void initialize(VkPhysicalDevice physicalDevice, VkDevice device);
    void destroyMesh(VkDevice device);
    void addVertex(glm::vec2 pos, glm::vec3 color);
    void addTriangle(uint16_t i1, uint16_t i2, uint16_t i3, uint16_t offset = 0);
    void draw(VkPipelineLayout pipelineLayout, VkCommandBuffer commandBuffer);

private:
    std::vector<Vertex> m_Vertices;
    std::vector<uint16_t> m_Indices;
    std::unique_ptr<DAEDataBuffer> m_VertexBuffer;
    std::unique_ptr<DAEDataBuffer> m_IndexBuffer;
};
