#define TINYOBJLOADER_IMPLEMENTATION
#include "Lib/tiny_obj_loader.h"
#include "ModelLoader.h"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include "Lib/json.hpp"
#include <stb_image.h>
#include "VulkanUtil.h"
#include "stb_image_write.h" // make sure stb_image_write.h is available

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

        if (type == "skybox") continue; // Skip the skybox, we'll handle it separately

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


CubemapTexture ModelLoader::CreateCubemapFromHorizontalCross(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkCommandPool commandPool,
    VkQueue graphicsQueue,
    const std::string& path)
{
    CubemapTexture cubemap{};

    int width, height, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error(std::string("Failed to load cubemap image: ") + path);

    // Debug print
    std::cout << "CreateCubemapFromHorizontalCross: loaded image " << path << " size = " << width << "x" << height << std::endl;

    // Detect layout and compute faceSize and offsets
    enum class Layout { H_STRIP, V_STRIP, H_CROSS, V_CROSS, UNKNOWN };
    Layout layout = Layout::UNKNOWN;
    int faceSize = 0;
    // horizontal 6x1 strip
    if (width % 6 == 0 && (width / 6) == height) {
        layout = Layout::H_STRIP;
        faceSize = width / 6;
    }
    // vertical 1x6 strip
    else if (height % 6 == 0 && (height / 6) == width) {
        layout = Layout::V_STRIP;
        faceSize = height / 6;
    }
    // horizontal cross (4x3)
    else if (width % 4 == 0 && height % 3 == 0 && (width / 4) == (height / 3)) {
        layout = Layout::H_CROSS;
        faceSize = width / 4;
    }
    // vertical cross (3x4)
    else if (width % 3 == 0 && height % 4 == 0 && (width / 3) == (height / 4)) {
        layout = Layout::V_CROSS;
        faceSize = width / 3;
    }

    if (layout == Layout::UNKNOWN || faceSize <= 0) {
        stbi_image_free(pixels);
        throw std::runtime_error("Unsupported cubemap layout or non-square faces. Image size: " + std::to_string(width) + "x" + std::to_string(height));
    }

    // Prepare 6 face buffers (RGBA)
    std::array<std::vector<uint8_t>, 6> faces;
    for (auto& f : faces) f.resize(faceSize * faceSize * 4);

    auto copyRect = [&](int srcX, int srcY, std::vector<uint8_t>& dst) {
        // srcX, srcY specify top-left pixel of face in source image
        for (int y = 0; y < faceSize; ++y) {
            uint8_t* srcRow = pixels + ((srcY + y) * width + srcX) * 4;
            uint8_t* dstRow = dst.data() + y * faceSize * 4;
            memcpy(dstRow, srcRow, faceSize * 4);
        }
        };

    // Map faces in the order [+X, -X, +Y, -Y, +Z, -Z]
    if (layout == Layout::H_STRIP) {
        // faces horizontally left->right
        for (int f = 0; f < 6; ++f)
            copyRect(f * faceSize, 0, faces[f]);
    }
    else if (layout == Layout::V_STRIP) {
        for (int f = 0; f < 6; ++f)
            copyRect(0, f * faceSize, faces[f]);
    }
    else if (layout == Layout::H_CROSS) {
        // Horizontal cross (4x3) layout assumed in this mapping:
        //   [    ][ +Y ][    ][    ]
        //   [ -X ][ +Z ][ +X ][ -Z ]
        //   [    ][ -Y ][    ][    ]
        // We'll extract faces accordingly. Coordinates are (col, row) * faceSize
        // +X = (2,1), -X = (0,1), +Y = (1,0), -Y = (1,2), +Z = (1,1), -Z = (3,1)
        copyRect(2 * faceSize, 1 * faceSize, faces[0]); // +X
        copyRect(0 * faceSize, 1 * faceSize, faces[1]); // -X
        copyRect(1 * faceSize, 0 * faceSize, faces[2]); // +Y
        copyRect(1 * faceSize, 2 * faceSize, faces[3]); // -Y
        copyRect(1 * faceSize, 1 * faceSize, faces[4]); // +Z
        copyRect(3 * faceSize, 1 * faceSize, faces[5]); // -Z
    }
    else { // V_CROSS
        // Vertical cross (3x4) layout mapping (common arrangement):
        //   [    ][ +Y ][    ]
        //   [ -X ][ +Z ][ +X ]
        //   [    ][ -Y ][    ]
        //   [    ][ -Z ][    ]
        // Coordinates: +X=(2,1) ; -X=(0,1) ; +Y=(1,0) ; -Y=(1,2) ; +Z=(1,1) ; -Z=(1,3)
        copyRect(2 * faceSize, 1 * faceSize, faces[0]); // +X
        copyRect(0 * faceSize, 1 * faceSize, faces[1]); // -X
        copyRect(1 * faceSize, 0 * faceSize, faces[2]); // +Y
        copyRect(1 * faceSize, 2 * faceSize, faces[3]); // -Y
        copyRect(1 * faceSize, 1 * faceSize, faces[4]); // +Z
        copyRect(1 * faceSize, 3 * faceSize, faces[5]); // -Z
    }

    // Now create staging buffer and upload combined data (face0..face5 contiguous)
    size_t layerSize = faceSize * faceSize * 4;
    size_t totalSize = layerSize * 6;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    {
        auto bufferPair = VkUtils::CreateBuffer(
            device,
            physicalDevice,
            totalSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        stagingBuffer = std::get<0>(bufferPair);
        stagingMemory = std::get<1>(bufferPair);
    }

    void* dataDst;
    vkMapMemory(device, stagingMemory, 0, totalSize, 0, &dataDst);
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dataDst);
    for (int f = 0; f < 6; ++f) {
        memcpy(dstPtr + f * layerSize, faces[f].data(), layerSize);
    }
    vkUnmapMemory(device, stagingMemory);

    // Create cube image with square face size
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imgInfo.extent = { (uint32_t)faceSize, (uint32_t)faceSize, 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 6;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imgInfo, nullptr, &cubemap.image) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cubemap image!");

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device, cubemap.image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtils::findMemoryType(
        physicalDevice,
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    if (vkAllocateMemory(device, &allocInfo, nullptr, &cubemap.memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate cubemap memory!");

    vkBindImageMemory(device, cubemap.image, cubemap.memory, 0);

    // Command buffer, transition, buffer->image copy (similar to original, but with correct extents)
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmdBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.image = cubemap.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 6;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier
    );

    std::vector<VkBufferImageCopy> regions(6);
    for (uint32_t face = 0; face < 6; ++face) {
        VkBufferImageCopy region{};
        region.bufferOffset = (VkDeviceSize)layerSize * face;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = face;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { (uint32_t)faceSize, (uint32_t)faceSize, 1 };
        regions[face] = region;
    }

    vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer, cubemap.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(regions.size()), regions.data());

    // Transition to SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier
    );

    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmdBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    stbi_image_free(pixels);

    // Create image view (cube)
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.image = cubemap.image;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;

    if (vkCreateImageView(device, &viewInfo, nullptr, &cubemap.view) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cubemap view!");


    VkPhysicalDeviceProperties physProps{};
    vkGetPhysicalDeviceProperties(physicalDevice, &physProps);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f; // you only have 1 mip; clamp to 0 for safety
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &cubemap.sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create cubemap sampler!");
    }


    for (int f = 0; f < 6; ++f) {
        std::string fname = "debug_sky_face_" + std::to_string(f) + ".png";
        int writeOk = stbi_write_png(fname.c_str(), faceSize, faceSize, 4, faces[f].data(), faceSize * 4);
        if (writeOk) {
            std::cout << "Wrote cubemap face " << f << " -> " << fname << std::endl;
        }
        else {
            std::cerr << "Failed to write cubemap face " << f << std::endl;
        }
        // print first pixel RGBA
        uint8_t r = faces[f][0];
        uint8_t g = faces[f][1];
        uint8_t b = faces[f][2];
        uint8_t a = faces[f][3];
    }


    return cubemap;
}




