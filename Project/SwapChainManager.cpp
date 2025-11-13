#include "SwapChainManager.h"
#include "VulkanBase.h"
#include "VulkanUtil.h"
#include <algorithm> 
#include <iostream> // For debugging

SwapChainManager::SwapChainManager(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, GLFWwindow* window)
	: m_device(device), m_physicalDevice(physicalDevice), m_surface(surface), m_window(window),
	m_swapChain(VK_NULL_HANDLE), m_swapChainImageFormat(VkFormat::VK_FORMAT_UNDEFINED),
	m_swapChainExtent({}) {
	createSwapChain();
	createImageViews();
}


SwapChainManager::~SwapChainManager() {
	cleanupSwapChain();
}


void SwapChainManager::createSwapChain() {
    // Clean up old swapchain if it exists
    if (m_swapChain != VK_NULL_HANDLE) {
        cleanupSwapChain();
    }

    // Query swap chain support details
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

    // Choose surface format, present mode, and extent
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Determine the number of images in the swap chain
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    // Fill out swap chain creation info structure
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkUtils::QueueFamilyIndices indices = VkUtils::FindQueueFamilies(m_physicalDevice, m_surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE; // If there was an old swapchain, set it here

    // Create the swap chain
    VkResult result = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapChain);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateSwapchainKHR failed with error: " << result << std::endl;
        throw std::runtime_error("failed to create swap chain!");
    }

    // Ensure the swapchain handle is valid
    if (m_swapChain == VK_NULL_HANDLE) {
        throw std::runtime_error("Swapchain handle is VK_NULL_HANDLE after creation!");
    }

    // Retrieve the swap chain images
    result = vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "First vkGetSwapchainImagesKHR failed with error: " << result << std::endl;
        throw std::runtime_error("failed to get swap chain images count!");
    }

    m_swapChainImages.resize(imageCount);
    result = vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, m_swapChainImages.data());
    if (result != VK_SUCCESS) {
        std::cerr << "Second vkGetSwapchainImagesKHR failed with error: " << result << std::endl;
        throw std::runtime_error("failed to get swap chain images!");
    }

    //std::cout << "Successfully retrieved " << m_swapChainImages.size() << " swapchain images." << std::endl;

    m_swapChainImageFormat = surfaceFormat.format;
    m_swapChainExtent = extent;
}



void SwapChainManager::createImageViews() {
    m_swapChainImageViews.resize(m_swapChainImages.size());

    for (size_t i = 0; i < m_swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = m_swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = m_swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &createInfo, nullptr, &m_swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }

    }
}


void SwapChainManager::cleanupSwapChain() {

    for (auto imageView : m_swapChainImageViews) {
        vkDestroyImageView(m_device, imageView, nullptr);
    }
    m_swapChainImageViews.clear();

    if (m_swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
        m_swapChain = VK_NULL_HANDLE;
    }
}


SwapChainSupportDetails SwapChainManager::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);
  //std::cout << "Capabilities: \n";
  //std::cout << "  minImageCount: " << details.capabilities.minImageCount << "\n";
  //std::cout << "  maxImageCount: " << details.capabilities.maxImageCount << "\n";
  //std::cout << "  currentExtent: (" << details.capabilities.currentExtent.width << ", " << details.capabilities.currentExtent.height << ")\n";

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
  // std::cout << "Formats: \n";
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
      // for (const auto& format : details.formats) {
      //     std::cout << "  format: " << format.format << ", colorSpace: " << format.colorSpace << "\n";
      // }
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);
  //  std::cout << "Present Modes: \n";
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, details.presentModes.data());
      // for (const auto& presentMode : details.presentModes) {
      //     std::cout << "  presentMode: " << presentMode << "\n";
      // }
    }

    return details;
}




VkSurfaceFormatKHR SwapChainManager::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}

	return availableFormats[0];
}

VkPresentModeKHR SwapChainManager::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return availablePresentMode;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChainManager::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	else {
		int width, height;
		glfwGetFramebufferSize(m_window, &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

VkSwapchainKHR SwapChainManager::getSwapChain() const {
	return m_swapChain;
}

std::vector<VkImage> SwapChainManager::getSwapChainImages() const {
	return m_swapChainImages;
}

std::vector<VkImageView> SwapChainManager::getSwapChainImageViews() const {
	return m_swapChainImageViews;
}

VkFormat SwapChainManager::getSwapChainImageFormat() const {
	return m_swapChainImageFormat;
}

VkExtent2D SwapChainManager::getSwapChainExtent() const {
	return m_swapChainExtent;
}
