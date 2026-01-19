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
#include "SkyboxMesh.h"
#include "SkyboxPipeline.h"
#include "WaterMesh.h"
#include "WaterPipeline.h"
#include "UnderwaterWaterPipeline.h"
#include "OceanBottomMesh.h"
#include "WaterTestingSystem.h"

// Forward declarations
class SwapChainManager;
class DAEMesh;
class Shader2D;
class Shader3D;
class xrxsPipeline;

const int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanBase
{
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
    void mouseMove(GLFWwindow *window, double xpos, double ypos);
    void mouseEvent(GLFWwindow *window, int button, int action, int mods);
    void beginRenderPass(const CommandBuffer &buffer, VkFramebuffer currentBuffer, VkExtent2D extent);
    void endRenderPass(const CommandBuffer &buffer);
    void recordCommandBuffer(CommandBuffer &commandBuffer, uint32_t imageIndex);

    void DrawSceneObjects(CommandBuffer &commandBuffer, uint32_t imageIndex);

    void printMatrix(const glm::mat4 &mat, const std::string &name);
    void loadModel();
    void createTextureSampler();
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, bool isCubemap);

    GLFWwindow *window;
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

    // Mouse var
    bool lmbPressed = false;
    void mouseScroll(GLFWwindow *window, double xoffset, double yoffset);

    glm::vec2 m_DragStart;
    float m_Radius = 10.0f;
    float m_Rotation = 0.0f;

    bool framebufferResized = false;
    bool isRecreatingSwapChain = false;
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

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
    std::vector<const char *> getRequiredExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    static const std::vector<const char *> deviceExtensions;

    // texture part

    void createTextureImage();
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    void createTextureImageView();
    VkImageView createCubemapImageView(VkImage image, VkFormat format);
    void loadTexture(const std::string &filePath, VkImage &textureImage, VkDeviceMemory &textureImageMemory);
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkSamplerCreateInfo samplerInfo;

    // Depth part

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    void createDepthResources();
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
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

    // light

    std::vector<VkBuffer> lightInfoBuffers;
    std::vector<VkDeviceMemory> lightInfoBuffersMemory;
    void createLightInfoBuffers();

    void updateLightInfoBuffer(uint32_t currentImage);
    bool rotationEnabled = false;
    bool rKeyPressed = false;
    ImVec4 backgroundColor = ImVec4(0.04f, 0.1f, 0.09f, 0.1f); // Initial background color for ImGui
    VkClearValue clearColor;                                   // Clear value used by Vulkan
    bool wireframeEnabled = false;
    bool currentWireframeState = false; // Store the last known wireframe state
    void updatePipelineIfNeeded();

    // toggleInfo
    std::vector<VkBuffer> toggleInfoBuffers;
    std::vector<VkDeviceMemory> toggleInfoBuffersMemory;
    void createToggleInfoBuffers();
    void updateToggleInfo(uint32_t currentImage, const ToggleInfo &toggleInfo);

    ToggleInfo currentToggleInfo;

    // light setup

    glm::vec3 light0Color = glm::vec3(1.0f, 0.6f, 0.2f); // Yellow light default color
    float light0Intensity = 2.5f;

    glm::vec3 light1Color = glm::vec3(1.0f, 1.0f, 1.0f); // White light default color
    float light1Intensity = 3.0f;

    glm::vec3 light0Position = glm::vec3(50.0f, 500.0f, -100.0f); // Sun position - high in sky
    glm::vec3 light1Position = glm::vec3(10.0f, 40.0f, 0.0f);     // White light position

    glm::vec3 ambientColor = glm::vec3(1.0f, 1.0f, 1.0f); // White light default color
    float ambientIntensity = 3.0f;

    void loadSceneFromJson(const std::string &sceneFilePath);
    std::vector<SceneObject> sceneObjects;

    // screenshot image
    VkImage screenshotImage;
    VkDeviceMemory screenshotImageMemory;

    void createScreenshotImage(VkExtent2D extent, VkFormat format);
    void blitImage(VkImage srcImage, VkImage dstImage, VkExtent2D extent);
    void saveScreenshot(VkImage image, VkExtent2D extent, const std::string &filename);
    void takeScreenshot();

    bool screenshotRequested = false;
    bool captureScreenshot = false;
    std::string lastScreenshotFilename = "";
    ImTextureID screenshotTextureID = nullptr;

    // ---- SKYBOX RESOURCES ----

    // SKYBOX
    std::unique_ptr<SkyboxMesh> skyboxMesh;
    std::unique_ptr<SkyboxPipeline> skyboxPipeline;

    VkDescriptorPool skyboxDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout skyboxDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet skyboxDescriptorSet = VK_NULL_HANDLE;

    VkImage skyboxImage = VK_NULL_HANDLE;
    VkDeviceMemory skyboxImageMemory = VK_NULL_HANDLE;
    VkImageView skyboxImageView = VK_NULL_HANDLE;
    VkSampler skyboxSampler = VK_NULL_HANDLE;

    // ---- SKYBOX DESCRIPTORS ----
    void createSkyboxDescriptorSetLayout();
    void createSkyboxDescriptorSet();

    void createSkyboxDescriptorPool();

    bool useSolidBackground = false; // default ON

    // Water rendering members
    std::unique_ptr<WaterMesh> waterMesh;
    std::unique_ptr<WaterPipeline> waterPipeline;

    // Underwater rendering members
    std::unique_ptr<UnderwaterWaterPipeline> underwaterWaterPipeline;
    std::unique_ptr<OceanBottomMesh> oceanBottomMesh;
    // Full-screen volumetric sunrays pipeline
    std::unique_ptr<WaterPipeline> sunraysPipeline;

    void createWaterResources();
    void createWaterDescriptorSetLayout();
    void createWaterDescriptorSet();
    void createWaterDescriptorPool();
    void createWaterDescriptors();
    void updateWaterDescriptors();
    VkImageView loadWaterTexture(const std::string &fileName);
    void createSceneColorTexture();
    void createWaterSampler();
    void createSceneReflectionTexture();
    void createSceneReflectionRenderPassAndFramebuffer();
    void createSceneRefractionRenderPassAndFramebuffer();
    void createSceneRenderPassAndFramebuffer();

    // ============================================================================
    // WATER RESOURCES — IMAGES / VIEWS / SAMPLERS
    // ============================================================================

    // Normal map
    VkImage waterNormalImage = VK_NULL_HANDLE;
    VkDeviceMemory waterNormalImageMemory = VK_NULL_HANDLE;
    VkImageView waterNormalImageView = VK_NULL_HANDLE;

    // DUDV map (for refraction distortion)
    VkImage waterDudvImage = VK_NULL_HANDLE;
    VkDeviceMemory waterDudvImageMemory = VK_NULL_HANDLE;
    VkImageView waterDudvImageView = VK_NULL_HANDLE;

    // Caustics animation texture
    VkImage waterCausticImage = VK_NULL_HANDLE;
    VkDeviceMemory waterCausticImageMemory = VK_NULL_HANDLE;
    VkImageView waterCausticImageView = VK_NULL_HANDLE;

    VkSampler waterSampler = VK_NULL_HANDLE;

    VkDescriptorSet waterDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout waterDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool waterDescriptorPool = VK_NULL_HANDLE;

    VkImage sceneColorImage = VK_NULL_HANDLE;
    VkDeviceMemory sceneColorImageMemory = VK_NULL_HANDLE;
    VkImageView sceneColorImageView = VK_NULL_HANDLE;
    VkSampler sceneColorSampler = VK_NULL_HANDLE;

    VkRenderPass sceneRenderPass = VK_NULL_HANDLE;
    VkFramebuffer sceneFramebuffer = VK_NULL_HANDLE;

    bool sceneOffscreenReady = false;

    float waterSpeed = 1.0f;
    bool showDebugRays = false;
    // === WATER COLOR AND SETTINGS ===
    glm::vec3 waterBaseColor = glm::vec3(0.0f, 0.3f, 0.5f);  // Deep blue/cyan base
    glm::vec3 waterLightColor = glm::vec3(1.0f, 1.0f, 1.0f); // White directional light
    float waterAmbient = 0.2f;
    float waterShininess = 512.0f;
    float waterCausticIntensity = 2.0f;
    float waterDistortionStrength = 0.04f;
    float waterFresnelR0 = 0.02f;
    float waterSurfaceOpacity = 0.55f;

    // Underwater specific settings
    float oceanBottomCausticIntensity = 1.0f;
    float underwaterGodRayIntensity = 1.0f;
    float underwaterScatteringIntensity = 0.5f;
    float underwaterOpacity = 0.9f;
    float underwaterFogDensity = 0.05f;

    // Marine Snow - suspended particles for scale reference
    float marineSnowIntensity = 0.5f; // 0 = off, 1 = heavy particulates
    float marineSnowSize = 1.0f;      // Particle size multiplier
    float marineSnowSpeed = 1.0f;     // Drift speed

    // Chromatic Aberration - underwater lens effect
    float chromaticAberrationStrength = 0.15f; // Color separation intensity
    bool showMarineSnowDebug = false;          // Debug: highlight particles
    bool showChromaticDebug = false;           // Debug: exaggerate CA

    glm::vec3 underwaterShallowColor = {0.0f, 0.6f, 0.8f};
    glm::vec3 underwaterDeepColor = {0.0f, 0.1f, 0.25f};

    VkDeviceMemory sceneReflectionImageMemory;
    VkImageView sceneReflectionImageView;
    VkSampler sceneReflectionSampler;

    VkDeviceMemory sceneRefractionImageMemory;
    VkImageView sceneRefractionImageView;
    VkSampler sceneRefractionSampler;

    VkRenderPass sceneReflectionRenderPass;
    VkRenderPass sceneRefractionRenderPass;

    VkFramebuffer sceneReflectionFramebuffer;
    VkFramebuffer sceneRefractionFramebuffer;

    VkImage sceneReflectionImage;
    VkImage sceneRefractionImage;

    // Extents (Resolution) for the off-screen textures
    VkExtent2D reflectionExtent = {800, 600};
    VkExtent2D refractionExtent = {800, 600};
    glm::mat4 reflectionViewMatrix;

    glm::mat4 calculateProjectionMatrix();

    void recordReflectionPass(CommandBuffer &commandBuffer, uint32_t imageIndex);
    void recordRefractionPass(CommandBuffer &commandBuffer, uint32_t imageIndex);
    void insertWaterTextureBarriers(CommandBuffer &commandBuffer);

    void createSceneRefractionTexture();
    void createImGuiRenderPass();
    void createImGuiFramebuffers();

    VkRenderPass imguiRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> imguiFramebuffers;

    // ============================================================================
    // WATER TESTING SYSTEM
    // ============================================================================
    std::unique_ptr<WaterTestingSystem> waterTestingSystem;

    // Test state tracking
    bool isTestModeActive = false;
    int currentTestConfigIndex = 0;
    int currentTestRunIndex = 0;
    std::vector<WaterTestConfig> pendingTestConfigs;
    std::vector<TestRunResult> completedTestResults;
    std::string testOutputFilePath = "test_results/water_test_results.csv";

    // Test timing - for accurate frame time measurement
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    std::chrono::high_resolution_clock::time_point frameStartTimePoint; // Set BEFORE drawFrame
    double lastCpuTimeMs = 0.0;

    // Benchmark timing - GPU-synchronized frame timing
    bool isBenchmarkActive = false;
    double gpuSyncedFrameTimeMs = 16.67; // GPU-synchronized frame time in ms (default ~60fps)

    // Test UI state
    int selectedTestType = 0; // 0=Performance, 1=ImageQuality, 2=TradeOff, 3=Custom
    bool autoExportResults = true;
    bool captureTestScreenshots = false;

    // Methods for testing
    void initializeWaterTestingSystem();
    void cleanupWaterTestingSystem();
    void startWaterTest(const WaterTestConfig &config);
    void updateWaterTest();
    void preFrameWaterTestUpdate();  // Camera setup before rendering
    void postFrameWaterTestUpdate(); // Timing measurement after GPU sync
    void endWaterTest();
    void applyTestConfiguration(const WaterTestConfig &config);
    void renderTestingUI();
    void exportTestResults();
};
