#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "DAEDescriptorPool.h"
#include "DAEUniformBufferObject.h"
#include "Vertex.h"
#include "VulkanUtil.h"
#include "command/CommandBuffer.h"
#include "command/CommandPool.h"
#include <algorithm>
#include "Camera.h"
#include "imgui.h"

// Forward declarations
class SwapChainManager;
class DAEMesh;
class Shader2D;
class Shader3D;
class xrxsPipeline;

const int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanBase {
public:
    VulkanBase();
    ~VulkanBase();
    void run();

private:
    void initWindow();
    void initVulkan();
    void initImGui();
    void mainLoop();
    void cleanup();
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFrameBuffers();
    void createCommandPool();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    void recreateSwapChain();
    void updateUniformBuffer(uint32_t currentImage);
    void processInput(float deltaTime);

    void keyEvent(int key, int scancode, int action, int mods);
    void mouseMove(GLFWwindow* window, double xpos, double ypos);
    void mouseEvent(GLFWwindow* window, int button, int action, int mods);
    void beginRenderPass(const CommandBuffer& buffer, VkFramebuffer currentBuffer, VkExtent2D extent);
    void endRenderPass(const CommandBuffer& buffer);
    void recordCommandBuffer(CommandBuffer& commandBuffer, uint32_t imageIndex);

    void printMatrix(const glm::mat4& mat, const std::string& name);
    void loadModel();
    void createTextureSampler();
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);


    GLFWwindow* window;
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;

    Camera camera;

    std::unique_ptr<SwapChainManager> swapChainManager;

    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    CommandPool commandPool;
    std::vector<CommandBuffer> commandBuffers;

    size_t currentFrame = 0;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    std::unique_ptr<DAEDescriptorPool<UBO>> descriptorPool;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

  


    //Mouse var 
    bool lmbPressed = false;
    void mouseScroll(GLFWwindow* window, double xoffset, double yoffset);

    glm::vec2 m_DragStart;
    float m_Radius = 10.0f;
    float m_Rotation = 0.0f;

    bool framebufferResized = false;
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkValidationLayerSupport();

    VkDebugUtilsMessengerEXT debugMessenger;
    std::unique_ptr<Shader2D> shader2D;
    std::unique_ptr<Shader3D> shader3D;
    std::unique_ptr<xrxsPipeline> m_Pipeline;
    std::unique_ptr<DAEMesh> m_Mesh;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    std::vector<const char*> getRequiredExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    static const std::vector<const char*> deviceExtensions;

    //texture part 

    void createTextureImage();
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createTextureImageView();
    void loadTexture(const std::string& filePath, VkImage& textureImage, VkDeviceMemory& textureImageMemory);
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkSamplerCreateInfo samplerInfo;

    //Depth part

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    void createDepthResources();
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    bool hasStencilComponent(VkFormat format);


    VkImage metalnessImage;
    VkDeviceMemory metalnessImageMemory;
    VkImageView metalnessImageView;

    // Normal texture
    VkImage normalImage;
    VkDeviceMemory normalImageMemory;
    VkImageView normalImageView;

    // Specular texture
    VkImage specularImage;
    VkDeviceMemory specularImageMemory;
    VkImageView specularImageView;

    void createAdditionalTextures();


    uint32_t mipLevels;
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkSampleCountFlagBits getMaxUsableSampleCount();
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    void createColorResources();

    //light 

    std::vector<VkBuffer> lightInfoBuffers;
    std::vector<VkDeviceMemory> lightInfoBuffersMemory;
    void createLightInfoBuffers();

    void updateLightInfoBuffer(uint32_t currentImage);
    bool rotationEnabled = false;
    bool rKeyPressed = false;
    ImVec4 backgroundColor = ImVec4(0.04f, 0.1f, 0.09f, 0.1f);  // Initial background color for ImGui
    VkClearValue clearColor;  // Clear value used by Vulkan
    bool wireframeEnabled = false;
    bool currentWireframeState = false;  // Store the last known wireframe state
    void updatePipelineIfNeeded();


    //toggleInfo
    std::vector<VkBuffer> toggleInfoBuffers;
    std::vector<VkDeviceMemory> toggleInfoBuffersMemory;
    void createToggleInfoBuffers();
    void updateToggleInfo(uint32_t currentImage, const ToggleInfo& toggleInfo);

    ToggleInfo currentToggleInfo;

    //light setup

    glm::vec3 light0Color = glm::vec3(1.0f, 0.6f, 0.2f); // Yellow light default color
    float light0Intensity = 2.5f;

    glm::vec3 light1Color = glm::vec3(1.0f, 1.0f, 1.0f); // White light default color
    float light1Intensity = 3.0f;

    glm::vec3 light0Position = glm::vec3(0.0f, 0.0f, 30.0f); // Yellow light position
    glm::vec3 light1Position = glm::vec3(10.0f, 40.0f, 0.0f); // White light position

    glm::vec3 ambientColor = glm::vec3(1.0f, 1.0f, 1.0f); // White light default color
    float ambientIntensity = 3.0f;


    void loadSceneFromJson(const std::string& sceneFilePath);
    std::vector<SceneObject> sceneObjects;
    
    // screenshot image 
    VkImage screenshotImage;
    VkDeviceMemory screenshotImageMemory;


    void createScreenshotImage(VkExtent2D extent, VkFormat format);
    void blitImage(VkImage srcImage, VkImage dstImage, VkExtent2D extent);
    void saveScreenshot(VkImage image, VkExtent2D extent, const std::string& filename);
    void takeScreenshot();

    bool screenshotRequested = false;  
    bool captureScreenshot = false;    
    std::string lastScreenshotFilename = "";
    ImTextureID screenshotTextureID = nullptr;

};
