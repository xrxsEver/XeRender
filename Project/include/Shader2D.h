#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

class Shader2D {
public:
    Shader2D(VkDevice device, const std::string& vertPath, const std::string& fragPath);
    ~Shader2D();

    uint32_t getStageCount() const;
    const VkPipelineShaderStageCreateInfo* getShaderStages() const;
    VkShaderModule getVertShaderModule();
    VkShaderModule getFragShaderModule();

private:
    VkDevice device;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkShaderModule createShaderModule(const std::vector<char>& code);
};
