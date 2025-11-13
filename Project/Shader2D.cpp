#include "Shader2D.h"
#include "VulkanUtil.h"

Shader2D::Shader2D(VkDevice device, const std::string& vertPath, const std::string& fragPath)
    : device(device) {
    auto vertShaderCode = VkUtils::readFile(vertPath);
    auto fragShaderCode = VkUtils::readFile(fragPath);

    vertShaderModule = createShaderModule(vertShaderCode);
    fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    shaderStages = { vertShaderStageInfo, fragShaderStageInfo };
}

Shader2D::~Shader2D() {
    if (vertShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vertShaderModule = VK_NULL_HANDLE;
    }
    if (fragShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        fragShaderModule = VK_NULL_HANDLE;
    }
}


uint32_t Shader2D::getStageCount() const {
    return static_cast<uint32_t>(shaderStages.size());
}

const VkPipelineShaderStageCreateInfo* Shader2D::getShaderStages() const {
    return shaderStages.data();
}

VkShaderModule Shader2D::getVertShaderModule()
{
    return vertShaderModule;
}

VkShaderModule Shader2D::getFragShaderModule()
{
    return fragShaderModule;
}

VkShaderModule Shader2D::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}
