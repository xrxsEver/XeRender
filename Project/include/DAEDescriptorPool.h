// DAEDescriptorPool.h
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include "DAEUniformBufferObject.h"

template <class UBO>
class DAEDescriptorPool {
public:
    DAEDescriptorPool(VkDevice device, size_t count);
    ~DAEDescriptorPool();

    void initialize(const VkUtils::VulkanContext& context);
    void setUBO(UBO data, size_t index);
   // void createDescriptorSets(VkDescriptorSetLayout descriptorSetLayout, const std::vector<VkBuffer>& buffers);
    void createDescriptorSets(VkDescriptorSetLayout descriptorSetLayout, const std::vector<VkBuffer>& buffers, VkImageView textureImageView, VkSampler textureSampler);
    void bindDescriptorSet(VkCommandBuffer buffer, VkPipelineLayout layout, size_t index);
    std::vector<VkDescriptorSet> getDescriptorSets() const { return m_DescriptorSets; }
    VkDescriptorPool getDescriptorPool() const { return m_DescriptorPool; } 

private:
    VkDevice m_Device;
    size_t m_Count;
    VkDescriptorPool m_DescriptorPool;
    VkDescriptorSetLayout m_DescriptorSetLayout;
    std::vector<VkDescriptorSet> m_DescriptorSets;
    std::vector<std::unique_ptr<DAEUniformBufferObject<UBO>>> m_UBOs;

    void createDescriptorPool(const VkUtils::VulkanContext& context);
    void createDescriptorSetLayout(const VkUtils::VulkanContext& context);
    void createUBOs(const VkUtils::VulkanContext& context);
};

// Include the implementation file
#include "DAEDescriptorPool.inl"
