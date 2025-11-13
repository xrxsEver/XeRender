#include "DAEDescriptorPool.h"
#include "DAEUniformBufferObject.h"
#include <stdexcept>
#include <string>

template <class UBO>
DAEDescriptorPool<UBO>::DAEDescriptorPool(VkDevice device, size_t count)
    : m_Device(device), m_Count(count) {
}

template <class UBO>
DAEDescriptorPool<UBO>::~DAEDescriptorPool() {
    vkDestroyDescriptorSetLayout(m_Device, m_DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
}

template <class UBO>
void DAEDescriptorPool<UBO>::initialize(const VkUtils::VulkanContext& context) {
    createDescriptorSetLayout(context);
    createUBOs(context);
    createDescriptorPool(context);
}

template <class UBO>
void DAEDescriptorPool<UBO>::createDescriptorPool(const VkUtils::VulkanContext& context) {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(m_Count);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(m_Count)*2;  

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(m_Count)*2;

    VkResult result = vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool: " + std::to_string(result));
    }
}

template <class UBO>
void DAEDescriptorPool<UBO>::setUBO(UBO data, size_t index) {
    if (index >= m_UBOs.size()) {
        throw std::out_of_range("Index out of range");
    }
    m_UBOs[index]->update(data);
}


template <class UBO>
void DAEDescriptorPool<UBO>::createDescriptorSets(VkDescriptorSetLayout descriptorSetLayout, const std::vector<VkBuffer>& buffers, VkImageView textureImageView, VkSampler textureSampler) {
    std::vector<VkDescriptorSetLayout> layouts(m_Count, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_DescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(m_Count);
    allocInfo.pSetLayouts = layouts.data();

    m_DescriptorSets.resize(m_Count);
    if (vkAllocateDescriptorSets(m_Device, &allocInfo, m_DescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    for (size_t i = 0; i < m_Count; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UBO);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = m_DescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = m_DescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}


template <class UBO>
void DAEDescriptorPool<UBO>::bindDescriptorSet(VkCommandBuffer buffer, VkPipelineLayout layout, size_t index) {
    if (index >= m_DescriptorSets.size()) {
        throw std::out_of_range("Index out of range");
    }
    vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &m_DescriptorSets[index], 0, nullptr);
}

template <class UBO>
void DAEDescriptorPool<UBO>::createDescriptorSetLayout(const VkUtils::VulkanContext& context) {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_DescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

template <class UBO>
void DAEDescriptorPool<UBO>::createUBOs(const VkUtils::VulkanContext& context) {
    m_UBOs.resize(m_Count);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkDeviceSize bufferSize = sizeof(UBO);

    for (size_t i = 0; i < m_Count; i++) {
        m_UBOs[i] = std::make_unique<DAEUniformBufferObject<UBO>>(
            context.physicalDevice,
            context.device,
            usage,
            properties,
            bufferSize
        );
    }
}

template class DAEDescriptorPool<UBO>;
