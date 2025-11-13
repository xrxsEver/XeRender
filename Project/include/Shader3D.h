#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

class Shader3D {
public:
    Shader3D(VkDevice device, const std::string& vertPath, const std::string& fragPath);
    ~Shader3D();

    VkShaderModule getVertShaderModule() const { return vertShaderModule; }
    VkShaderModule getFragShaderModule() const { return fragShaderModule; }

    std::vector<VkPipelineShaderStageCreateInfo> getShaderStages() const;

    uint32_t getStageCount() const { return 2; }

private:
    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> readFile(const std::string& filename);

    VkDevice device;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
};
