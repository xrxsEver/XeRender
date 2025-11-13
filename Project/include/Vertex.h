#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <vulkan/vulkan.h>
#include <array>
#include <vector>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;  
    glm::vec3 normal;
    glm::vec3 tangent;  
    glm::vec3 bitangent;


    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct UBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 viewPos;
};

struct ToggleInfo {
    VkBool32 applyNormalMap;
    VkBool32 applyMetalnessMap;
    VkBool32 applySpecularMap;
    VkBool32 viewNormalOnly;
    VkBool32 viewMetalnessOnly;
    VkBool32 viewSpecularOnly;
    VkBool32 RimLight;
};



struct MeshData {
    glm::mat4 model;
};

struct Light {
    glm::vec3 position;       // 12 bytes
    alignas(16) glm::vec3 color; // 16 bytes due to alignment
    float intensity;          // 4 bytes
};

struct LightInfo {
    Light lights[2];            // Two lights, each properly aligned
    alignas(16) glm::vec3 viewPos;  // Camera/viewer position
    alignas(16) glm::vec3 ambientColor;  // Ambient light color
    float ambientIntensity;         // Ambient light intensity
    alignas(16) float padding[3];  // Padding for alignment (optional, for future-proofing)
};


struct SceneObject {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

#endif // VERTEX_H
