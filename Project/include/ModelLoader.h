#pragma once

#include "Vertex.h"
#include <vector>
#include <string>
#include "Lib/json.hpp"

class ModelLoader {
public:
    static bool loadOBJ(const std::string& filename, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

    static std::vector<SceneObject> loadSceneFromJson(const std::string& filePath);

    static  void generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& position, const glm::vec3& scale);
    static  void generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& position, float radius, int sectorCount = 36, int stackCount = 18);
};
