#define TINYOBJLOADER_IMPLEMENTATION
#include "Lib/tiny_obj_loader.h"
#include "ModelLoader.h"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include "Lib/json.hpp"

using json = nlohmann::json;

bool ModelLoader::loadOBJ(const std::string& filename, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
        std::cerr << "Failed to load/parse .obj file: " << warn << err << std::endl;
        return false;
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }

    return true;
}

std::vector<SceneObject> ModelLoader::loadSceneFromJson(const std::string& filePath) {
    std::vector<SceneObject> sceneObjects;

    std::ifstream sceneFile(filePath);
    if (!sceneFile.is_open()) {
        std::cerr << "Failed to open scene file: " << filePath << std::endl;
        return sceneObjects;
    }

    json sceneJson;
    sceneFile >> sceneJson;

    std::string sceneName = sceneJson["scene"]["name"];
    //std::cout << "Loading scene: " << sceneName << std::endl;

    for (const auto& object : sceneJson["scene"]["objects"]) {
        std::string type = object["type"];
        std::string modelPath = object.contains("model") ? object["model"].get<std::string>() : "";

        SceneObject sceneObject;  // Create a new SceneObject for each item

        if (type == "sphere" || type == "cube") {
            //std::cout << "Creating primitive: " << type << std::endl;
            if (type == "sphere") {
                float radius = object.contains("radius") ? object["radius"].get<float>() : 1.0f; // Default radius
                glm::vec3 position = object.contains("position")
                    ? glm::vec3(object["position"][0], object["position"][1], object["position"][2])
                    : glm::vec3(0.0f, 0.0f, 0.0f); // Default position
                generateSphere(sceneObject.vertices, sceneObject.indices, position, radius);
            }
            else if (type == "cube") {
                glm::vec3 scale = object.contains("scale")
                    ? glm::vec3(object["scale"][0], object["scale"][1], object["scale"][2])
                    : glm::vec3(1.0f, 1.0f, 1.0f); // Default scale
                glm::vec3 position = object.contains("position")
                    ? glm::vec3(object["position"][0], object["position"][1], object["position"][2])
                    : glm::vec3(0.0f, 0.0f, 0.0f); // Default position
                generateCube(sceneObject.vertices, sceneObject.indices, position, scale);
            }
        }
        else if (!modelPath.empty()) {
            if (!loadOBJ(modelPath, sceneObject.vertices, sceneObject.indices)) {
                std::cerr << "Failed to load model: " << modelPath << std::endl;
                continue;
            }
        }

        sceneObjects.push_back(sceneObject);  // Add the SceneObject to the vector
    }

    return sceneObjects;
}


void ModelLoader::generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& position, const glm::vec3& scale) {
    glm::vec3 cubeVertices[8] = {
        {-0.5f, -0.5f, -0.5f},
        { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f},
        {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f},
        { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f}
    };

    uint32_t cubeIndices[36] = {
        0, 1, 2, 2, 3, 0,  // Back face
        4, 5, 6, 6, 7, 4,  // Front face
        0, 1, 5, 5, 4, 0,  // Bottom face
        2, 3, 7, 7, 6, 2,  // Top face
        0, 3, 7, 7, 4, 0,  // Left face
        1, 2, 6, 6, 5, 1   // Right face
    };

    // Scale and translate cube vertices
    for (int i = 0; i < 8; i++) {
        glm::vec3 transformedVertex = (cubeVertices[i] * scale) + position;
        Vertex vertex{};
        vertex.pos = transformedVertex;
        vertex.color = glm::vec3(1.0f); // Default color
        vertex.texCoord = glm::vec2(0.0f, 0.0f); // Temporary texture coordinates
        vertex.normal = glm::normalize(cubeVertices[i]); // Calculate normal
        vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f); // Default tangent
        vertex.bitangent = glm::vec3(0.0f, 1.0f, 0.0f); // Default bitangent
        vertices.push_back(vertex);
    }

    indices.insert(indices.end(), std::begin(cubeIndices), std::end(cubeIndices));
}

void ModelLoader::generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& position, float radius, int sectorCount, int stackCount) {
    float x, y, z, xy; // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius; // normal
    float s, t; // texCoord

    const float PI = 3.14159265359f;
    float sectorStep = 2 * PI / sectorCount;
    float stackStep = PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i) {
        stackAngle = PI / 2 - i * stackStep; // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle); // r * cos(u)
        z = radius * sinf(stackAngle); // r * sin(u)

        for (int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep; // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle); // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle); // r * cos(u) * sin(v)
            Vertex vertex{};
            vertex.pos = glm::vec3(x, y, z) + position;
            vertex.normal = glm::vec3(x * lengthInv, y * lengthInv, z * lengthInv);
            vertex.texCoord = glm::vec2((float)j / sectorCount, (float)i / stackCount);
            vertex.color = glm::vec3(1.0f); // Default color
            vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f); // Default tangent
            vertex.bitangent = glm::vec3(0.0f, 1.0f, 0.0f); // Default bitangent
            vertices.push_back(vertex);
        }
    }

    // Generating sphere indices
    for (int i = 0; i < stackCount; ++i) {
        int k1 = i * (sectorCount + 1); // beginning of current stack
        int k2 = k1 + sectorCount + 1;  // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            if (i != (stackCount - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
}

