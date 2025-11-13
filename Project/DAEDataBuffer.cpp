#include "DAEDataBuffer.h"

DAEDataBuffer::DAEDataBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size)
    : m_VkDevice(device), m_Size(size) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &m_VkBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, m_VkBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &m_VkBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, m_VkBuffer, m_VkBufferMemory, 0);
}

void DAEDataBuffer::upload(VkDeviceSize size, void* data) {
    void* destData;
    vkMapMemory(m_VkDevice, m_VkBufferMemory, 0, size, 0, &destData);
    memcpy(destData, data, (size_t)size);
    vkUnmapMemory(m_VkDevice, m_VkBufferMemory);
}


void DAEDataBuffer::update()
{
}

void DAEDataBuffer::map(VkDeviceSize size, void* data) {
    vkMapMemory(m_VkDevice, m_VkBufferMemory, 0, size, 0, &data);
}

void DAEDataBuffer::destroy() {
    if (m_VkBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_VkDevice, m_VkBuffer, nullptr);
        m_VkBuffer = VK_NULL_HANDLE;
    }
    if (m_VkBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_VkDevice, m_VkBufferMemory, nullptr);
        m_VkBufferMemory = VK_NULL_HANDLE;
    }
}

void DAEDataBuffer::bindAsVertexBuffer(VkCommandBuffer commandBuffer) {
    VkBuffer vertexBuffers[] = { m_VkBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
}

void DAEDataBuffer::bindAsIndexBuffer(VkCommandBuffer commandBuffer) {
    vkCmdBindIndexBuffer(commandBuffer, m_VkBuffer, 0, VK_INDEX_TYPE_UINT16);
}

VkBuffer DAEDataBuffer::getVkBuffer() {
    return m_VkBuffer;
}

VkDeviceSize DAEDataBuffer::getSizeInBytes() {
    return m_Size;
}

uint32_t DAEDataBuffer::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}



