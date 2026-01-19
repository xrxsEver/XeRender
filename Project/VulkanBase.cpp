#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32

// Prevent Windows min/max macros from conflicting with std::min/std::max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <Lib/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include "VulkanBase.h"
#include "SwapChainManager.h"
#include "DAEMesh.h"
#include "Shader2D.h"
#include "Shader3D.h"
#include "xrxsPipeline.h"
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include "ModelLoader.h"
#include <glm/glm.hpp>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <filesystem>

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> VulkanBase::deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VulkanBase::VulkanBase()
    : camera(glm::vec3(0.0f, 1.5f, 55.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f),
      currentToggleInfo({VK_TRUE, VK_TRUE, VK_TRUE, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE})
{ // Initialize ToggleInfo to default values
    initWindow();
    initVulkan();
    initImGui();
}

VulkanBase::~VulkanBase()
{
    cleanup();
}

void VulkanBase::run()
{
    mainLoop();
}

void VulkanBase::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(VkUtils::WIDTH, VkUtils::HEIGHT, "XeRender", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods)
                       {
        auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            app->processInput(key);
        } });
    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos)
                             {
        auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
        app->mouseMove(window, xpos, ypos); });
    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods)
                               {
        auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
        app->mouseEvent(window, button, action, mods); });
    glfwSetScrollCallback(window, [](GLFWwindow *window, double xoffset, double yoffset)
                          {
        auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
        app->mouseScroll(window, xoffset, yoffset); });

    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VulkanBase::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    swapChainManager = std::make_unique<SwapChainManager>(device, physicalDevice, surface, window);
    createRenderPass();
    createImGuiRenderPass();

    createDescriptorSetLayout();
    createCommandPool(); // Need commandPool for createWaterResources()
    // Create water resources and descriptor set layout early so graphics pipeline can include it
    createWaterResources(); // Needs commandPool, so must be after createCommandPool()
    createWaterSampler();   // Create the sampler for water textures
    createWaterDescriptorSetLayout();
    createGraphicsPipeline();

    createColorResources();
    createDepthResources();
    createFrameBuffers();
    createImGuiFramebuffers();

    createTextureImage();
    createAdditionalTextures();

    createTextureImageView();

    createTextureSampler();

    // ---- SKYBOX INIT (replace previous block) ----
    int w = 0, h = 0, ch = 0;
    if (stbi_info("textures/skybox.jpg", &w, &h, &ch))
    {
        std::cout << "initVulkan: skybox.jpg exists, size = " << w << " x " << h << " (channels = " << ch << ")\n";
    }
    else
    {
        std::cout << "initVulkan: WARNING - skybox.jpg not found or cannot read image info\n";
    }

    // 1) Load cubemap (returns struct with image/view/sampler)
    CubemapTexture cb;
    try
    {
        cb = ModelLoader::CreateCubemapFromHorizontalCross(
            device,
            physicalDevice,
            commandPool.getVkCommandPool(),
            graphicsQueue,
            "textures/skybox.jpg");
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("Failed to create cubemap: ") + e.what());
    }

    // 2) Assign to VulkanBase members
    skyboxImage = cb.image;
    skyboxImageMemory = cb.memory;
    skyboxImageView = cb.view;
    skyboxSampler = cb.sampler;

    // ---- DEBUG CUBEMAP LOADING -------------------------------------------------
    //     std::cout << "\n=== DEBUG: Cubemap Loaded ===\n";
    //     std::cout << "Image:   " << cb.image << "\n";
    //     std::cout << "Memory:  " << cb.memory << "\n";
    //     std::cout << "View:    " << cb.view << "\n";
    //     std::cout << "Sampler: " << cb.sampler << "\n";
    //
    //     // I can also check if each face was written correctly (ModelLoader printed them)
    //     std::cout << "Check ModelLoader output above for 'face[0..5] first pixel RGBA'.\n";
    //
    //     std::cout << "\n=== DEBUG: Assigned Cubemap To VulkanBase ===\n";
    //     std::cout << "skyboxImage       = " << skyboxImage << "\n";
    //     std::cout << "skyboxImageMemory = " << skyboxImageMemory << "\n";
    //     std::cout << "skyboxImageView   = " << skyboxImageView << "\n";
    //     std::cout << "skyboxSampler     = " << skyboxSampler << "\n";

    // Quick sanity: ensure loader returned valid handles
    if (skyboxImage == VK_NULL_HANDLE || skyboxImageView == VK_NULL_HANDLE || skyboxSampler == VK_NULL_HANDLE)
    {
        throw std::runtime_error("initVulkan: Cubemap loader returned null handle(s). Check image layout and loader implementation.");
    }

    // 3) Create skybox mesh
    skyboxMesh = std::make_unique<SkyboxMesh>();
    skyboxMesh->create(device, physicalDevice, commandPool.getVkCommandPool(), graphicsQueue);

    // 4) Create descriptor pool/layout and allocate descriptor set
    createSkyboxDescriptorPool();      // must create skyboxDescriptorPool (one combined sampler entry)
    createSkyboxDescriptorSetLayout(); // descriptor set layout for cubemap sampler
    createSkyboxDescriptorSet();       // will now succeed because imageView & sampler are valid

    // 5) Create skybox pipeline (needs descriptor set layout and renderPass)
    //  std::cout << "[DEBUG] initVulkan: About to create skybox pipeline\n";
    //  std::cout << "[DEBUG] initVulkan: renderPass = " << renderPass << "\n";
    //  std::cout << "[DEBUG] initVulkan: imguiRenderPass = " << imguiRenderPass << "\n";
    //  std::cout << "[DEBUG] initVulkan: msaaSamples = " << msaaSamples << "\n";

    if (renderPass == imguiRenderPass)
    {
        throw std::runtime_error("initVulkan: CRITICAL BUG - renderPass equals imguiRenderPass!");
    }
    if (renderPass == VK_NULL_HANDLE)
    {
        throw std::runtime_error("initVulkan: CRITICAL BUG - renderPass is NULL!");
    }

    skyboxPipeline = std::make_unique<SkyboxPipeline>();
    skyboxPipeline->create(
        device,
        swapChainManager->getSwapChainExtent(),
        renderPass,
        descriptorSetLayout,
        skyboxDescriptorSetLayout,
        msaaSamples);

    // ---- SKYBOX END ----

    sceneObjects = ModelLoader::loadSceneFromJson("res/scene.json");

    // Aggregate the vertex and index data from all SceneObjects
    uint32_t indexOffset = 0;
    for (const auto &obj : sceneObjects)
    {
        vertices.insert(vertices.end(), obj.vertices.begin(), obj.vertices.end());

        for (const auto &index : obj.indices)
        {
            indices.push_back(index + indexOffset);
        }

        indexOffset += static_cast<uint32_t>(obj.vertices.size());
    }

    loadModel();

    createSceneColorTexture();

    createSceneRenderPassAndFramebuffer();

    createSceneReflectionTexture(); // Creates sceneReflectionImage/View/Sampler
    createSceneReflectionRenderPassAndFramebuffer();

    createSceneRefractionRenderPassAndFramebuffer(); // Creates sceneRefractionImage/View/RenderPass/Framebuffer

    createUniformBuffers();
    createLightInfoBuffers();
    createToggleInfoBuffers();

    createDescriptorPool();

    createDescriptorSets();

    // --------- WATER INIT ---------
    waterMesh = std::make_unique<WaterMesh>();
    // Make the water plane much larger so it appears effectively unlimited from the camera.
    // Note: increasing resolution increases memory and GPU cost. Adjust resolution if needed.
    waterMesh->create(device, physicalDevice, commandPool.getVkCommandPool(), graphicsQueue, 512, 20000.0f); // 512 resolution, 20k units world size
    // Note: createWaterResources() and createWaterDescriptorSetLayout() are now called earlier in initVulkan()

    createWaterDescriptorSet();

    // DEBUG: Verify MSAA samples
    //  std::cout << "\n=== WATER PIPELINE CREATION DEBUG ===\n";
    //  std::cout << "msaaSamples: " << msaaSamples << " (should be 8)\n";
    //  std::cout << "renderPass: " << renderPass << "\n";
    //  std::cout << "imguiRenderPass: " << imguiRenderPass << "\n";

    if (renderPass == imguiRenderPass)
    {
        throw std::runtime_error("initVulkan: CRITICAL BUG - renderPass equals imguiRenderPass during water creation!");
    }
    if (renderPass == VK_NULL_HANDLE)
    {
        throw std::runtime_error("initVulkan: CRITICAL BUG - renderPass is NULL during water creation!");
    }

    std::cout << "Creating water pipeline with MSAA samples: " << msaaSamples << "\n";

    waterPipeline = std::make_unique<WaterPipeline>();
    waterPipeline->create(
        device,
        swapChainManager->getSwapChainExtent(),
        renderPass,
        descriptorSetLayout,
        waterDescriptorSetLayout,
        msaaSamples,
        false);

    std::cout << "Water pipeline created successfully\n";
    std::cout << "Water pipeline layout: " << waterPipeline->layout << "\n";
    std::cout << "Water pipeline: " << waterPipeline->pipeline << "\n";
    std::cout << "===================================\n\n";
    // ---------------------------------------------------

    // --------- UNDERWATER WATER PIPELINE INIT ---------
    underwaterWaterPipeline = std::make_unique<UnderwaterWaterPipeline>();
    underwaterWaterPipeline->create(
        device,
        swapChainManager->getSwapChainExtent(),
        renderPass,
        descriptorSetLayout,
        waterDescriptorSetLayout,
        msaaSamples,
        true);
    std::cout << "Underwater water pipeline created successfully\n";

    // --------- SUNRAYS FULL-SCREEN PIPELINE INIT ---------
    // Uses sunrays.vert/sunrays.frag via WaterPipeline with isSunraysPipeline=true
    sunraysPipeline = std::make_unique<WaterPipeline>();
    sunraysPipeline->create(
        device,
        swapChainManager->getSwapChainExtent(),
        renderPass,
        descriptorSetLayout,
        waterDescriptorSetLayout,
        msaaSamples,
        /*isSunraysPipeline=*/true);
    std::cout << "Sunrays pipeline created successfully\n";

    // --------- OCEAN BOTTOM MESH INIT ---------
    oceanBottomMesh = std::make_unique<OceanBottomMesh>();
    oceanBottomMesh->create(device, physicalDevice, commandPool.getVkCommandPool(), graphicsQueue, 256, 20000.0f, -50.0f);
    std::cout << "Ocean bottom mesh created successfully\n";
    // ---------------------------------------------------

    createCommandBuffers();
    createSyncObjects();

    // --------- WATER TESTING SYSTEM INIT ---------
    initializeWaterTestingSystem();
    // ---------------------------------------------
}

void VulkanBase::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Initialize Vulkan for ImGui
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = VkUtils::FindQueueFamilies(physicalDevice, surface).graphicsFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = descriptorPool->getDescriptorPool();
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = swapChainManager->getSwapChainImages().size();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = [](VkResult err)
    {
        if (err != VK_SUCCESS)
        {
            throw std::runtime_error("ImGui Vulkan initialization failed with error code: " + std::to_string(err));
        }
    };

    // Validate all required Vulkan objects and values
    IM_ASSERT(init_info.Instance != VK_NULL_HANDLE);
    IM_ASSERT(init_info.PhysicalDevice != VK_NULL_HANDLE);
    IM_ASSERT(init_info.Device != VK_NULL_HANDLE);
    IM_ASSERT(init_info.Queue != VK_NULL_HANDLE);
    IM_ASSERT(init_info.DescriptorPool != VK_NULL_HANDLE);
    IM_ASSERT(init_info.MinImageCount >= 2);
    IM_ASSERT(init_info.ImageCount >= init_info.MinImageCount);
    IM_ASSERT(renderPass != VK_NULL_HANDLE);

    // Initialize ImGui Vulkan binding
    // ImGui_ImplVulkan_Init(&init_info, renderPass);
    // std::cout << "[DEBUG] initImGui: Initializing ImGui with imguiRenderPass (1-sample)\n";
    // std::cout << "[DEBUG] initImGui: ImGui MSAASamples = " << init_info.MSAASamples << " (should be VK_SAMPLE_COUNT_1_BIT = 1)\n";
    ImGui_ImplVulkan_Init(&init_info, imguiRenderPass);

    // Upload Fonts
    VkCommandBuffer command_buffer = beginSingleTimeCommands();
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    endSingleTimeCommands(command_buffer);
    vkDeviceWaitIdle(device);

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void VulkanBase::mainLoop()
{
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Pre-frame update for water testing (camera positioning only)
        if (isTestModeActive)
        {
            preFrameWaterTestUpdate();

            // START timing BEFORE drawFrame - this is when the frame begins
            frameStartTimePoint = std::chrono::high_resolution_clock::now();
        }
        else if (isBenchmarkActive)
        {
            // For Quick Benchmark, also measure GPU-synced time
            frameStartTimePoint = std::chrono::high_resolution_clock::now();
            processInput(deltaTime);
        }
        else
        {
            processInput(deltaTime);
        }

        drawFrame();

        // During testing, force GPU synchronization to get accurate frame timing
        // Without this, the GPU has multiple frames in flight and we only measure CPU submission time
        if (isTestModeActive)
        {
            // Wait for GPU to finish ALL work - this is critical for accurate timing
            vkQueueWaitIdle(graphicsQueue);

            // Fetch GPU timing results AFTER GPU has finished (timestamps are now valid)
            if (waterTestingSystem)
            {
                waterTestingSystem->readGpuTimestamps();
            }

            // NOW measure time - from before drawFrame to after GPU is done
            auto frameEndTime = std::chrono::high_resolution_clock::now();
            lastCpuTimeMs = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTimePoint).count();

            postFrameWaterTestUpdate();
        }
        else if (isBenchmarkActive)
        {
            // Wait for GPU to finish ALL work - critical for accurate benchmark timing
            vkQueueWaitIdle(graphicsQueue);

            // Measure GPU-synced frame time
            auto frameEndTime = std::chrono::high_resolution_clock::now();
            gpuSyncedFrameTimeMs = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTimePoint).count();
        }

        takeScreenshot();
    }

    vkDeviceWaitIdle(device);
}

void VulkanBase::cleanup()
{

    // Wait for the device to be idle before starting the cleanup process
    vkDeviceWaitIdle(device);

    // Cleanup water testing system first
    cleanupWaterTestingSystem();

    ImGui::DestroyContext();

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool.getVkCommandPool(), nullptr);

    for (auto framebuffer : swapChainFramebuffers)
    {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    for (size_t i = 0; i < lightInfoBuffers.size(); i++)
    {
        vkDestroyBuffer(device, lightInfoBuffers[i], nullptr);
        vkFreeMemory(device, lightInfoBuffersMemory[i], nullptr);
    }

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChainManager->getSwapChain(), nullptr);

    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);

    vkDestroyDevice(device, nullptr);

    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    if (sceneFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, sceneFramebuffer, nullptr);
        sceneFramebuffer = VK_NULL_HANDLE;
    }
    if (sceneRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, sceneRenderPass, nullptr);
        sceneRenderPass = VK_NULL_HANDLE;
    }

    // Refraction cleanup
    if (sceneRefractionFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, sceneRefractionFramebuffer, nullptr);
        sceneRefractionFramebuffer = VK_NULL_HANDLE;
    }
    if (sceneRefractionRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, sceneRefractionRenderPass, nullptr);
        sceneRefractionRenderPass = VK_NULL_HANDLE;
    }
    if (sceneRefractionSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, sceneRefractionSampler, nullptr);
        sceneRefractionSampler = VK_NULL_HANDLE;
    }
    if (sceneRefractionImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, sceneRefractionImageView, nullptr);
        sceneRefractionImageView = VK_NULL_HANDLE;
    }
    if (sceneRefractionImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, sceneRefractionImage, nullptr);
        vkFreeMemory(device, sceneRefractionImageMemory, nullptr);
        sceneRefractionImage = VK_NULL_HANDLE;
    }

    // Reflection cleanup
    if (sceneReflectionFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, sceneReflectionFramebuffer, nullptr);
        sceneReflectionFramebuffer = VK_NULL_HANDLE;
    }
    if (sceneReflectionRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, sceneReflectionRenderPass, nullptr);
        sceneReflectionRenderPass = VK_NULL_HANDLE;
    }
    if (sceneReflectionSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, sceneReflectionSampler, nullptr);
        sceneReflectionSampler = VK_NULL_HANDLE;
    }
    if (sceneReflectionImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, sceneReflectionImageView, nullptr);
        sceneReflectionImageView = VK_NULL_HANDLE;
    }
    if (sceneReflectionImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, sceneReflectionImage, nullptr);
        vkFreeMemory(device, sceneReflectionImageMemory, nullptr);
        sceneReflectionImage = VK_NULL_HANDLE;
    }

    // Water cleanup (if created)
    if (waterSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, waterSampler, nullptr);
        waterSampler = VK_NULL_HANDLE;
    }
    if (waterDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, waterDescriptorSetLayout, nullptr);
        waterDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (waterNormalImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, waterNormalImageView, nullptr);
        waterNormalImageView = VK_NULL_HANDLE;
    }
    if (waterNormalImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, waterNormalImage, nullptr);
        vkFreeMemory(device, waterNormalImageMemory, nullptr);
    }
    if (waterPipeline)
    {
        waterPipeline->destroy(device);
        waterPipeline.reset();
    }
    if (waterMesh)
    {
        waterMesh->destroy(device);
        waterMesh.reset();
    }
    if (underwaterWaterPipeline)
    {
        underwaterWaterPipeline->destroy(device);
        underwaterWaterPipeline.reset();
    }
    if (oceanBottomMesh)
    {
        oceanBottomMesh->destroy(device);
        oceanBottomMesh.reset();
    }
    for (auto fb : imguiFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);

    vkDestroyRenderPass(device, imguiRenderPass, nullptr);

    if (VkUtils::enableValidationLayers)
    {
        VkUtils::DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
}

void VulkanBase::createInstance()
{
    if (VkUtils::enableValidationLayers && !checkValidationLayerSupport())
    {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (VkUtils::enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create instance!");
    }
}

void VulkanBase::keyEvent(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void VulkanBase::mouseMove(GLFWwindow *window, double xpos, double ypos)
{
    if (ImGui::GetIO().WantCaptureMouse)
    {
        return; // Skip camera movement if ImGui is capturing the mouse
    }

    // Existing camera movement code
    static bool firstMouse = true;
    static float lastX = 800.0f / 2.0;
    static float lastY = 600.0f / 2.0;

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    if (lmbPressed)
    {
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // Reversed since y-coordinates range from bottom to top
        lastX = xpos;
        lastY = ypos;

        camera.processMouseMovement(xoffset, yoffset);
    }
    else
    {
        lastX = xpos;
        lastY = ypos;
    }
}

void VulkanBase::mouseEvent(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        lmbPressed = true;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        lmbPressed = false;
    }
}

void VulkanBase::mouseScroll(GLFWwindow *window, double xoffset, double yoffset)
{
    camera.processMouseScroll(static_cast<float>(yoffset));
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

void VulkanBase::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
}

void VulkanBase::setupDebugMessenger()
{
    if (!VkUtils::enableValidationLayers)
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    if (VkUtils::CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void VulkanBase::createRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainManager->getSwapChainImageFormat();
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = swapChainManager->getSwapChainImageFormat();
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create render pass!");
    }

    // DEBUG: Verify render pass was created successfully
    // std::cout << "[DEBUG] createRenderPass: Main render pass created with handle: " << renderPass << "\n";
    // std::cout << "[DEBUG] createRenderPass: Color attachment samples: " << msaaSamples << "\n";
    // std::cout << "[DEBUG] createRenderPass: Depth attachment samples: " << msaaSamples << "\n";
}

void VulkanBase::createImGuiRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainManager->getSwapChainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // <-- KEEP previous scene
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &imguiRenderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create ImGui render pass!");

    // DEBUG: Verify ImGui render pass was created successfully and is different from main
    //   std::cout << "[DEBUG] createImGuiRenderPass: ImGui render pass created with handle: " << imguiRenderPass << "\n";
    //   std::cout << "[DEBUG] createImGuiRenderPass: Samples: VK_SAMPLE_COUNT_1_BIT\n";
    //   std::cout << "[DEBUG] Main renderPass handle: " << renderPass << "\n";
}

void VulkanBase::createImGuiFramebuffers()
{
    imguiFramebuffers.resize(swapChainManager->getSwapChainImageViews().size());

    for (size_t i = 0; i < imguiFramebuffers.size(); i++)
    {
        VkImageView attachment = swapChainManager->getSwapChainImageViews()[i];

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = imguiRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &attachment;
        fbInfo.width = swapChainManager->getSwapChainExtent().width;
        fbInfo.height = swapChainManager->getSwapChainExtent().height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &imguiFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create ImGui framebuffer!");
    }
}

void VulkanBase::createSurface()
{
    VkWin32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = glfwGetWin32Window(window);
    createInfo.hinstance = GetModuleHandle(nullptr);

    if (vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface!");
    }
}

void VulkanBase::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
    int highestScore = 0;

    for (const auto &device : devices)
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        std::cout << "Found GPU: " << deviceProperties.deviceName << std::endl;

        // Check if the device is an NVIDIA RTX GPU
        bool isNvidiaRTX = (std::string(deviceProperties.deviceName).find("NVIDIA") != std::string::npos) &&
                           (std::string(deviceProperties.deviceName).find("RTX") != std::string::npos);

        if (isNvidiaRTX && isDeviceSuitable(device))
        {
            selectedDevice = device;
            std::cout << "Selected NVIDIA RTX GPU: " << deviceProperties.deviceName << std::endl;
            break; // Since we found an RTX, we can stop searching
        }
    }

    // Fallback: If no NVIDIA RTX device was found, pick the first suitable device
    if (selectedDevice == VK_NULL_HANDLE)
    {
        for (const auto &device : devices)
        {
            if (isDeviceSuitable(device))
            {
                selectedDevice = device;
                break;
            }
        }
    }

    if (selectedDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    physicalDevice = selectedDevice;
    msaaSamples = getMaxUsableSampleCount();
}

void VulkanBase::createLogicalDevice()
{
    VkUtils::QueueFamilyIndices indices = VkUtils::FindQueueFamilies(physicalDevice, surface);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.fillModeNonSolid = VK_TRUE; // Enable non-solid fill modes

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

void VulkanBase::framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
    auto app = reinterpret_cast<VulkanBase *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
    // Immediately flag that we're in a recreation state to stop all rendering
    app->isRecreatingSwapChain = true;
    // Mark water mesh as invalid immediately
    if (app->waterMesh)
    {
        app->waterMesh->setValid(false);
    }
}

void VulkanBase::createFrameBuffers()
{
    swapChainFramebuffers.resize(swapChainManager->getSwapChainImageViews().size());

    for (size_t i = 0; i < swapChainManager->getSwapChainImageViews().size(); i++)
    {
        std::array<VkImageView, 3> attachments = {
            colorImageView,
            depthImageView,
            swapChainManager->getSwapChainImageViews()[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainManager->getSwapChainExtent().width;
        framebufferInfo.height = swapChainManager->getSwapChainExtent().height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanBase::createCommandPool()
{
    VkUtils::QueueFamilyIndices queueFamilyIndices = VkUtils::FindQueueFamilies(physicalDevice, surface);

    commandPool.create(device, queueFamilyIndices.graphicsFamily.value());
}

void printCurrentWorkingDirectory()
{
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
}

void VulkanBase::loadModel()
{
    // printCurrentWorkingDirectory();

    std::string modelPath = "Res/Model.obj";
    std::filesystem::path fullPath = std::filesystem::absolute(modelPath);

    // std::cout << "Attempting to load model from path: " << fullPath << std::endl;  //uncomment to see the path its trying to load

    if (!std::filesystem::exists(fullPath))
    {
        throw std::runtime_error("Model file does not exist: " + fullPath.string());
    }

    SceneObject modelObject;

    if (!ModelLoader::loadOBJ(modelPath, modelObject.vertices, modelObject.indices))
    {
        throw std::runtime_error("Failed to load model!");
    }

    sceneObjects.push_back(modelObject); // Store the model as a SceneObject

    // Aggregate the vertex and index data
    uint32_t indexOffset = static_cast<uint32_t>(vertices.size());

    vertices.insert(vertices.end(), modelObject.vertices.begin(), modelObject.vertices.end());

    for (const auto &index : modelObject.indices)
    {
        indices.push_back(index + indexOffset);
    }

    createVertexBuffer();
    createIndexBuffer();
}

void VulkanBase::createVertexBuffer()
{
    // std::cout << "Creating vertex buffer..." << std::endl;
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    std::tie(stagingBuffer, stagingBufferMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    std::tie(vertexBuffer, vertexBufferMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkUtils::CopyBuffer(stagingBuffer, vertexBuffer, bufferSize, device, commandPool.getVkCommandPool(), graphicsQueue);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    // std::cout << "Vertex buffer created: " << vertexBuffer << std::endl;
}

void VulkanBase::createIndexBuffer()
{
    // std::cout << "Creating index buffer..." << std::endl;
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    std::tie(stagingBuffer, stagingBufferMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    std::tie(indexBuffer, indexBufferMemory) = VkUtils::CreateBuffer(
        device, physicalDevice, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkUtils::CopyBuffer(stagingBuffer, indexBuffer, bufferSize, device, commandPool.getVkCommandPool(), graphicsQueue);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    // std::cout << "Index buffer created: " << indexBuffer << std::endl;
}

void VulkanBase::recordCommandBuffer(CommandBuffer &commandBuffer, uint32_t imageIndex)
{
    // SAFETY: Skip command buffer recording entirely during swap chain recreation
    // to prevent accessing destroyed resources (images, descriptor sets, etc.)
    if (isRecreatingSwapChain)
    {
        return;
    }

    if (imageIndex >= commandBuffers.size() || imageIndex >= swapChainFramebuffers.size() || imageIndex >= descriptorSets.size())
    {
        throw std::out_of_range("imageIndex is out of range.");
    }

    // ==============================================================================
    struct alignas(16) WaterPushConstant
    {
        float time;
        float scale;
        float debugRays;
        float renderingMode;  // 0=BL, 1=PB, 2=OPT
        glm::vec4 baseColor;  // Used for SHALLOW Color
        glm::vec4 lightColor; // Used for DEEP Color (repurposed)
        float ambient;
        float shininess;
        float causticIntensity;
        float distortionStrength;
        float godRayIntensity;
        float scatteringIntensity;
        float opacity;
        float fogDensity;
        // God-ray tuning parameters (added to match shader push-constant expectations)
        float godExposure;
        float godDecay;
        float godDensity;
        float godSampleScale;
    };

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBuffer.begin(&beginInfo);

    // === START GPU TIMER (for accurate GPU timing during tests) ===
    if (waterTestingSystem && isTestModeActive)
    {
        waterTestingSystem->resetTimestampQueries(commandBuffer.getVkCommandBuffer());
        waterTestingSystem->writeTimestampStart(commandBuffer.getVkCommandBuffer());
    }

    static glm::vec3 underwaterShallowColor = {0.0f, 0.6f, 0.8f}; // Bright Teal
    static glm::vec3 underwaterDeepColor = {0.0f, 0.1f, 0.25f};   // Dark Blue

    // Determine if camera is underwater
    float waterHeight = 0.0f;
    bool isUnderwater = camera.position.y < waterHeight - 0.1f;
    static bool showDebugRays = false;
    // Persistent tuning controls (editable in ImGui; values persist between frames)
    static float godExposure = 0.6f;
    static float godDecay = 0.96f;
    static float godDensity = 0.5f;
    static float godSampleScale = 1.0f;

    // Underwater rendering mode selection
    static int currentRenderingMode = 0; // 0=BL, 1=PB, 2=OPT
    static const char *renderingModes[] = {"BL (Baseline)", "PB (Physically-Based)", "OPT (Optimized)"};
    // --- 1. SET CLEAR COLOR TO DEEP COLOR IF UNDERWATER ---
    // This is CRITICAL. The background must match the deep fog to hide seams.
    if (isUnderwater)
    {
        clearColor.color = {{underwaterDeepColor.r, underwaterDeepColor.g, underwaterDeepColor.b, 1.0f}};
    }
    else if (useSolidBackground)
    {
        clearColor.color = {{backgroundColor.x, backgroundColor.y, backgroundColor.z, backgroundColor.w}};
    }
    else
    {
        clearColor.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainManager->getSwapChainExtent();

    std::array<VkClearValue, 3> clearValues{};
    clearValues[0] = clearColor;
    clearValues[1].depthStencil = {1.0f, 0};
    clearValues[2] = clearColor;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    commandBuffer.beginRenderPass(renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // DEBUG: Verify we're in the correct render pass
    //  std::cout << "[DEBUG] Main render pass started. renderPass handle: " << renderPass
    //            << " (should be non-null)\n";

    // ------------------------------------------------------------------

    if (isUnderwater)
    {
        // Performance optimization based on rendering mode
        float qualityMultiplier = 1.0f;
        bool enableAdvancedEffects = true;
        bool skipOceanBottom = false;

        switch (currentRenderingMode)
        {
        case 0: // BL (Baseline) - Simple performance mode
            qualityMultiplier = 0.4f;
            enableAdvancedEffects = false;
            skipOceanBottom = true; // Skip ocean bottom for performance
            break;
        case 1: // PB (Physically-Based) - Full quality
            qualityMultiplier = 1.2f;
            enableAdvancedEffects = true;
            break;
        case 2: // OPT (Optimized) - High quality with optimizations
            qualityMultiplier = 1.0f;
            enableAdvancedEffects = true;
            break;
        }

        // Push constant struct (Aligned to 16 bytes for GPU)
        WaterPushConstant underwaterWaterPushData{};
        underwaterWaterPushData.time = static_cast<float>(glfwGetTime()) * waterSpeed;
        underwaterWaterPushData.scale = 1.0f;
        underwaterWaterPushData.renderingMode = static_cast<float>(currentRenderingMode);
        underwaterWaterPushData.baseColor = glm::vec4(underwaterShallowColor, 1.0f);
        underwaterWaterPushData.lightColor = glm::vec4(underwaterDeepColor, 1.0f);

        // Repurpose ambient for chromatic aberration strength
        underwaterWaterPushData.ambient = chromaticAberrationStrength;
        // Repurpose shininess for marine snow size (multiply by 100 to get reasonable range in shader)
        underwaterWaterPushData.shininess = marineSnowSize * 100.0f;

        underwaterWaterPushData.causticIntensity = enableAdvancedEffects ? oceanBottomCausticIntensity * qualityMultiplier : 0.0f;
        underwaterWaterPushData.distortionStrength = waterDistortionStrength * (enableAdvancedEffects ? 1.0f : 0.5f);
        // Respect user setting; allow zero intensity to truly disable rays
        underwaterWaterPushData.godRayIntensity = underwaterGodRayIntensity * qualityMultiplier;

        // Repurpose scatteringIntensity for marine snow intensity
        underwaterWaterPushData.scatteringIntensity = marineSnowIntensity * qualityMultiplier;

        // Underwater volumetric strength (do not clamp; user may want subtle fog)
        underwaterWaterPushData.opacity = underwaterOpacity;
        underwaterWaterPushData.fogDensity = underwaterFogDensity * (enableAdvancedEffects ? 1.0f : 0.7f);
        // God-ray tuning with performance adjustments
        underwaterWaterPushData.godExposure = godExposure * qualityMultiplier;
        underwaterWaterPushData.godDecay = currentRenderingMode == 0 ? 0.98f : godDecay; // Faster decay for baseline
        underwaterWaterPushData.godDensity = godDensity * qualityMultiplier;
        underwaterWaterPushData.godSampleScale = godSampleScale * (currentRenderingMode == 0 ? 0.5f : 1.0f);

        // Debug flags encoded in debugRays: 0=off, 1=rays, 2=snow, 3=both, 4+=chromatic
        float debugValue = 0.0f;
        if (showDebugRays)
            debugValue = 1.0f;
        if (showMarineSnowDebug)
            debugValue = 2.0f;
        if (showDebugRays && showMarineSnowDebug)
            debugValue = 3.0f;
        if (showChromaticDebug)
            debugValue = 4.0f;
        underwaterWaterPushData.debugRays = debugValue;

        // 1. Draw ocean bottom first (skip for baseline mode for performance)
        if (!skipOceanBottom)
        {
            vkCmdBindPipeline(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            std::array<VkDescriptorSet, 2> oceanBottomSets = {descriptorSets[imageIndex], waterDescriptorSet};
            vkCmdBindDescriptorSets(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 2, oceanBottomSets.data(), 0, nullptr);

            vkCmdPushConstants(commandBuffer.getVkCommandBuffer(), pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPushConstant), &underwaterWaterPushData);

            oceanBottomMesh->draw(commandBuffer.getVkCommandBuffer());
        }

        // 2. Draw scene objects
        this->DrawSceneObjects(commandBuffer, imageIndex);

        // 3. Draw Water Surface (always use the water surface shader)
        if (waterPipeline && waterMesh && waterMesh->getValid())
        {
            waterPipeline->bind(commandBuffer.getVkCommandBuffer());

            std::array<VkDescriptorSet, 2> waterSets = {descriptorSets[imageIndex], waterDescriptorSet};
            vkCmdBindDescriptorSets(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    waterPipeline->layout, 0, static_cast<uint32_t>(waterSets.size()),
                                    waterSets.data(), 0, nullptr);

            WaterPushConstant waterData{};
            waterData.time = static_cast<float>(glfwGetTime()) * waterSpeed;
            waterData.scale = 1.0f;
            waterData.renderingMode = static_cast<float>(currentRenderingMode);
            waterData.baseColor = glm::vec4(waterBaseColor, 1.0f);
            waterData.lightColor = glm::vec4(waterLightColor, 1.0f);
            waterData.ambient = waterAmbient;
            waterData.shininess = waterShininess;
            waterData.causticIntensity = enableAdvancedEffects ? waterCausticIntensity * qualityMultiplier : 0.0f;
            waterData.distortionStrength = waterDistortionStrength * (enableAdvancedEffects ? 1.0f : 0.6f);
            waterData.godRayIntensity = 0.0f;
            waterData.scatteringIntensity = 0.0f;
            waterData.opacity = waterSurfaceOpacity;
            waterData.fogDensity = underwaterFogDensity; // used for underside absorption
            waterData.debugRays = 0.0f;
            waterData.godExposure = godExposure;
            waterData.godDecay = godDecay;
            waterData.godDensity = godDensity;
            waterData.godSampleScale = godSampleScale;

            vkCmdPushConstants(commandBuffer.getVkCommandBuffer(), waterPipeline->layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPushConstant), &waterData);

            waterMesh->draw(commandBuffer.getVkCommandBuffer());
        }

        // 4. Underwater volumetric fog/scattering pass (fullscreen, alpha blended)
        if (underwaterWaterPipeline && (enableAdvancedEffects || currentRenderingMode == 0))
        {
            underwaterWaterPipeline->bind(commandBuffer.getVkCommandBuffer());
            std::array<VkDescriptorSet, 2> uwSets = {descriptorSets[imageIndex], waterDescriptorSet};
            vkCmdBindDescriptorSets(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    underwaterWaterPipeline->layout, 0, static_cast<uint32_t>(uwSets.size()),
                                    uwSets.data(), 0, nullptr);
            vkCmdPushConstants(commandBuffer.getVkCommandBuffer(), underwaterWaterPipeline->layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPushConstant), &underwaterWaterPushData);
            vkCmdDraw(commandBuffer.getVkCommandBuffer(), 3, 1, 0, 0);
        }

        // 5. God rays pass (fullscreen additive) - Quality based on mode
        if (sunraysPipeline && underwaterGodRayIntensity > 0.01f)
        {
            sunraysPipeline->bind(commandBuffer.getVkCommandBuffer());
            std::array<VkDescriptorSet, 2> sunraySets = {descriptorSets[imageIndex], waterDescriptorSet};
            vkCmdBindDescriptorSets(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    sunraysPipeline->layout, 0, static_cast<uint32_t>(sunraySets.size()),
                                    sunraySets.data(), 0, nullptr);
            vkCmdPushConstants(commandBuffer.getVkCommandBuffer(), sunraysPipeline->layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPushConstant), &underwaterWaterPushData);
            vkCmdDraw(commandBuffer.getVkCommandBuffer(), 3, 1, 0, 0);
        }
    }
    else
    {
        // === ABOVE WATER RENDERING ===

        // 1. Draw Scene
        this->DrawSceneObjects(commandBuffer, imageIndex);

        // 2. Draw Water Surface (skip if mesh is invalid during resize)
        if (waterPipeline && waterMesh && waterMesh->getValid())
        {
            waterPipeline->bind(commandBuffer.getVkCommandBuffer());

            std::array<VkDescriptorSet, 2> waterSets = {descriptorSets[imageIndex], waterDescriptorSet};
            vkCmdBindDescriptorSets(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    waterPipeline->layout, 0, static_cast<uint32_t>(waterSets.size()),
                                    waterSets.data(), 0, nullptr);

            WaterPushConstant waterData{};
            waterData.time = static_cast<float>(glfwGetTime()) * waterSpeed;
            waterData.scale = 1.0f;
            waterData.baseColor = glm::vec4(waterBaseColor, 1.0f);
            waterData.lightColor = glm::vec4(waterLightColor, 1.0f);
            waterData.ambient = waterAmbient;
            waterData.shininess = waterShininess;
            waterData.causticIntensity = waterCausticIntensity;
            waterData.distortionStrength = waterDistortionStrength;
            waterData.godRayIntensity = 0.0f;
            waterData.scatteringIntensity = 0.0f;
            waterData.opacity = waterSurfaceOpacity;
            waterData.fogDensity = 0.0f;
            waterData.debugRays = showDebugRays ? 1.0f : 0.0f;
            // Above-water god-ray tuning (kept in push constants too)
            waterData.godExposure = godExposure;
            waterData.godDecay = godDecay;
            waterData.godDensity = godDensity;
            waterData.godSampleScale = godSampleScale;

            vkCmdPushConstants(commandBuffer.getVkCommandBuffer(), waterPipeline->layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPushConstant), &waterData);

            waterMesh->draw(commandBuffer.getVkCommandBuffer());
        }
    }

    // End render pass before ImGui setup (shared for both underwater and above water)
    commandBuffer.endRenderPass();

    // Now setup ImGui frame (outside of render pass)
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // =========================================================================
    // COLLAPSIBLE PANEL STATE & ANIMATION
    // =========================================================================
    static bool panelOpen = true;
    static float panelAnim = 1.0f;
    static bool keyPressed = false;
    const float PANEL_WIDTH = 260.0f;
    const float COLLAPSED_WIDTH = 42.0f;

    // Toggle with ` key (grave accent / tilde)
    if (glfwGetKey(window, GLFW_KEY_GRAVE_ACCENT) == GLFW_PRESS && !keyPressed)
    {
        panelOpen = !panelOpen;
        keyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_GRAVE_ACCENT) == GLFW_RELEASE)
        keyPressed = false;

    // Smooth animation
    float target = panelOpen ? 1.0f : 0.0f;
    panelAnim += (target - panelAnim) * ImGui::GetIO().DeltaTime * 12.0f;
    float ease = panelAnim * panelAnim * (3.0f - 2.0f * panelAnim);
    float currentWidth = COLLAPSED_WIDTH + (PANEL_WIDTH - COLLAPSED_WIDTH) * ease;

    // =========================================================================
    // MODERN DARK THEME
    // =========================================================================
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.TabRounding = 6.0f;
    style.WindowPadding = ImVec2(12, 10);
    style.FramePadding = ImVec2(10, 5);
    style.ItemSpacing = ImVec2(8, 5);
    style.ScrollbarSize = 10.0f;
    style.GrabMinSize = 10.0f;

    // Colors
    ImVec4 bg = ImVec4(0.07f, 0.07f, 0.09f, 0.97f);
    ImVec4 bgLight = ImVec4(0.12f, 0.12f, 0.15f, 1.0f);
    ImVec4 accent = ImVec4(0.40f, 0.70f, 1.0f, 1.0f);
    ImVec4 accentDim = ImVec4(0.25f, 0.50f, 0.80f, 0.7f);
    ImVec4 text = ImVec4(0.92f, 0.92f, 0.94f, 1.0f);
    ImVec4 textDim = ImVec4(0.50f, 0.50f, 0.55f, 1.0f);
    ImVec4 green = ImVec4(0.35f, 0.90f, 0.50f, 1.0f);
    ImVec4 yellow = ImVec4(1.0f, 0.85f, 0.35f, 1.0f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.2f, 0.25f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    ImGui::PushStyleColor(ImGuiCol_TextDisabled, textDim);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, bgLight);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.18f, 0.22f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, accentDim);
    ImGui::PushStyleColor(ImGuiCol_Header, bgLight);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, accentDim);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, accent);
    ImGui::PushStyleColor(ImGuiCol_Button, bgLight);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, accentDim);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, accent);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, accent);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.55f, 0.80f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, accent);
    ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_TabHovered, accentDim);
    ImGui::PushStyleColor(ImGuiCol_TabActive, accent);
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.25f, 0.25f, 0.30f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, bgLight);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, accent);
    const int COLOR_COUNT = 23;

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(currentWidth, ImGui::GetIO().DisplaySize.y));

    ImGui::Begin("##Panel", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    // =========================================================================
    // HEADER WITH TOGGLE
    // =========================================================================
    {
        // Toggle button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        if (ImGui::Button(panelOpen ? "<<" : ">>", ImVec2(28, 28)))
        {
            panelOpen = !panelOpen;
        }
        ImGui::PopStyleColor();

        if (panelOpen && ease > 0.5f)
        {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, accent);
            ImGui::Text("XeRender");
            ImGui::PopStyleColor();

            ImGui::SameLine(currentWidth - 70);
            float fps = (isBenchmarkActive || isTestModeActive) && gpuSyncedFrameTimeMs > 0
                            ? static_cast<float>(1000.0 / gpuSyncedFrameTimeMs)
                            : ImGui::GetIO().Framerate;
            ImGui::PushStyleColor(ImGuiCol_Text, (isBenchmarkActive || isTestModeActive) ? green : textDim);
            ImGui::Text("%.0f", fps);
            ImGui::PopStyleColor();
        }
    }

    // Only render content when panel is mostly open
    if (ease > 0.3f)
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // =====================================================================
        // SCENE SECTION
        // =====================================================================
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Rotate", &rotationEnabled);
            ImGui::SameLine(120);
            ImGui::Checkbox("Wireframe", &wireframeEnabled);

            ImGui::Spacing();

            if (ImGui::TreeNode("Materials"))
            {
                ImGui::Checkbox("Normal", (bool *)&currentToggleInfo.applyNormalMap);
                ImGui::SameLine(100);
                ImGui::Checkbox("Metal", (bool *)&currentToggleInfo.applyMetalnessMap);
                ImGui::Checkbox("Specular", (bool *)&currentToggleInfo.applySpecularMap);

                ImGui::Spacing();
                ImGui::TextDisabled("Debug Views");
                ImGui::Checkbox("Normal##V", (bool *)&currentToggleInfo.viewNormalOnly);
                ImGui::SameLine(100);
                ImGui::Checkbox("Metal##V", (bool *)&currentToggleInfo.viewMetalnessOnly);
                ImGui::Checkbox("Spec##V", (bool *)&currentToggleInfo.viewSpecularOnly);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Background"))
            {
                ImGui::Checkbox("Solid Color", &useSolidBackground);
                if (useSolidBackground)
                {
                    ImGui::ColorEdit3("##BgCol", (float *)&backgroundColor, ImGuiColorEditFlags_NoInputs);
                }
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Camera"))
            {
                glm::vec3 camPos = camera.getPosition();
                ImGui::TextDisabled("%.1f, %.1f, %.1f", camPos.x, camPos.y, camPos.z);
                ImGui::SliderFloat("Sens", &camera.mouseSensitivity, 0.025f, 1.5f, "%.2f");
                ImGui::SliderFloat("Speed", &camera.movementSpeed, 0.001f, 0.050f, "%.3f");
                if (ImGui::Button("Screenshot", ImVec2(-1, 0)))
                    captureScreenshot = true;
                ImGui::TreePop();
            }
        }

        // =====================================================================
        // LIGHTING SECTION
        // =====================================================================
        if (ImGui::CollapsingHeader("Lighting"))
        {
            if (ImGui::TreeNode("Sun"))
            {
                ImGui::ColorEdit3("##SunCol", (float *)&light0Color, ImGuiColorEditFlags_NoInputs);
                ImGui::SameLine();
                ImGui::SliderFloat("##SunInt", &light0Intensity, 0.0f, 20.0f, "%.1f");
                ImGui::SliderFloat("X##Sun", &light0Position.x, -500.0f, 500.0f);
                ImGui::SliderFloat("Y##Sun", &light0Position.y, 0.0f, 1000.0f);
                ImGui::SliderFloat("Z##Sun", &light0Position.z, -500.0f, 500.0f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Secondary"))
            {
                ImGui::ColorEdit3("##L2Col", (float *)&light1Color, ImGuiColorEditFlags_NoInputs);
                ImGui::SameLine();
                ImGui::SliderFloat("##L2Int", &light1Intensity, 0.0f, 20.0f, "%.1f");
                ImGui::SliderFloat3("Pos##L2", (float *)&light1Position[0], -100.0f, 100.0f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Ambient"))
            {
                ImGui::ColorEdit3("##AmbCol", (float *)&ambientColor, ImGuiColorEditFlags_NoInputs);
                ImGui::SameLine();
                ImGui::SliderFloat("##AmbInt", &ambientIntensity, 0.0f, 20.0f, "%.1f");
                ImGui::TreePop();
            }

            ImGui::Checkbox("Rim Light", (bool *)&currentToggleInfo.RimLight);
        }

        // =====================================================================
        // WATER SECTION
        // =====================================================================
        if (ImGui::CollapsingHeader("Water", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Rendering mode at top
            ImGui::Combo("Mode", &currentRenderingMode, renderingModes, IM_ARRAYSIZE(renderingModes));

            ImGui::Spacing();

            if (ImGui::TreeNode("Surface"))
            {
                ImGui::ColorEdit3("Color##Surf", (float *)&waterBaseColor, ImGuiColorEditFlags_NoInputs);
                ImGui::SliderFloat("Opacity", &waterSurfaceOpacity, 0.0f, 1.0f);
                ImGui::SliderFloat("Speed", &waterSpeed, 0.0f, 5.0f);
                ImGui::SliderFloat("Distort", &waterDistortionStrength, 0.0f, 0.1f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Underwater"))
            {
                ImGui::ColorEdit3("Shallow", (float *)&underwaterShallowColor, ImGuiColorEditFlags_NoInputs);
                ImGui::SameLine();
                ImGui::ColorEdit3("Deep", (float *)&underwaterDeepColor, ImGuiColorEditFlags_NoInputs);
                ImGui::SliderFloat("God Rays", &underwaterGodRayIntensity, 0.0f, 3.0f);
                ImGui::SliderFloat("Caustics", &oceanBottomCausticIntensity, 0.0f, 5.0f);
                ImGui::SliderFloat("Fog", &underwaterFogDensity, 0.0f, 0.2f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Particles"))
            {
                ImGui::SliderFloat("Amount", &marineSnowIntensity, 0.0f, 2.0f);
                ImGui::SliderFloat("Size", &marineSnowSize, 0.2f, 3.0f);
                ImGui::SliderFloat("Drift", &marineSnowSpeed, 0.0f, 3.0f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Effects"))
            {
                ImGui::SliderFloat("Chromatic", &chromaticAberrationStrength, 0.0f, 0.5f);
                ImGui::TextDisabled("God Ray Tuning");
                ImGui::SliderFloat("Exposure", &godExposure, 0.0f, 3.0f);
                ImGui::SliderFloat("Decay", &godDecay, 0.7f, 1.0f);
                ImGui::SliderFloat("Density", &godDensity, 0.1f, 2.0f);
                ImGui::SliderFloat("Scale", &godSampleScale, 0.25f, 2.0f);
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Debug"))
            {
                ImGui::Checkbox("Rays", &showDebugRays);
                ImGui::SameLine();
                ImGui::Checkbox("Snow", &showMarineSnowDebug);
                ImGui::SameLine();
                ImGui::Checkbox("CA", &showChromaticDebug);
                if (showDebugRays || showMarineSnowDebug || showChromaticDebug)
                {
                    ImGui::TextColored(yellow, "Debug ON");
                }
                ImGui::TreePop();
            }
        }

        // =====================================================================
        // BENCHMARK SECTION
        // =====================================================================
        if (ImGui::CollapsingHeader("Benchmark"))
        {
            static bool runningBenchmark = false;
            static float benchmarkTime = 0.0f;
            static float benchmarkFps[3] = {0.0f, 0.0f, 0.0f};
            static float benchmarkFpsSum[3] = {0.0f, 0.0f, 0.0f};
            static int benchmarkFrameCount[3] = {0, 0, 0};
            static int savedRenderingMode = 0;
            static bool firstBenchmarkFrame = false;
            static glm::vec3 savedCameraPos;
            static float savedCameraYaw, savedCameraPitch;

            if (!runningBenchmark)
            {
                if (ImGui::Button("Run Benchmark", ImVec2(-1, 28)))
                {
                    runningBenchmark = true;
                    isBenchmarkActive = true;
                    firstBenchmarkFrame = true;
                    benchmarkTime = 0.0f;
                    gpuSyncedFrameTimeMs = 0.0;
                    savedRenderingMode = currentRenderingMode;
                    savedCameraPos = camera.position;
                    savedCameraYaw = camera.yaw;
                    savedCameraPitch = camera.pitch;
                    camera.position = glm::vec3(0.0f, -25.0f, 30.0f);
                    camera.setYaw(-90.0f);
                    camera.setPitch(15.0f);
                    for (int i = 0; i < 3; ++i)
                    {
                        benchmarkFps[i] = 0.0f;
                        benchmarkFpsSum[i] = 0.0f;
                        benchmarkFrameCount[i] = 0;
                    }
                }
            }
            else
            {
                if (firstBenchmarkFrame)
                {
                    firstBenchmarkFrame = false;
                    ImGui::TextColored(yellow, "Warming up...");
                }
                else if (gpuSyncedFrameTimeMs > 0.1)
                {
                    float gpuSyncedDeltaTime = static_cast<float>(gpuSyncedFrameTimeMs / 1000.0);
                    benchmarkTime += gpuSyncedDeltaTime;
                    float gpuSyncedFps = static_cast<float>(1000.0 / gpuSyncedFrameTimeMs);

                    ImGui::ProgressBar(benchmarkTime / 6.0f, ImVec2(-1, 0));

                    float warmupTime = 0.5f;
                    float testDuration = 2.0f;

                    if (benchmarkTime < testDuration)
                    {
                        if (currentRenderingMode != 0)
                            currentRenderingMode = 0;
                        if (benchmarkTime > warmupTime)
                        {
                            benchmarkFpsSum[0] += gpuSyncedFps;
                            benchmarkFrameCount[0]++;
                        }
                    }
                    else if (benchmarkTime < testDuration * 2)
                    {
                        if (currentRenderingMode != 1)
                            currentRenderingMode = 1;
                        if (benchmarkTime > testDuration + warmupTime)
                        {
                            benchmarkFpsSum[1] += gpuSyncedFps;
                            benchmarkFrameCount[1]++;
                        }
                    }
                    else if (benchmarkTime < testDuration * 3)
                    {
                        if (currentRenderingMode != 2)
                            currentRenderingMode = 2;
                        if (benchmarkTime > testDuration * 2 + warmupTime)
                        {
                            benchmarkFpsSum[2] += gpuSyncedFps;
                            benchmarkFrameCount[2]++;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 3; ++i)
                        {
                            if (benchmarkFrameCount[i] > 0)
                                benchmarkFps[i] = benchmarkFpsSum[i] / benchmarkFrameCount[i];
                        }
                        runningBenchmark = false;
                        isBenchmarkActive = false;
                        currentRenderingMode = savedRenderingMode;
                        camera.position = savedCameraPos;
                        camera.setYaw(savedCameraYaw);
                        camera.setPitch(savedCameraPitch);
                    }
                }
            }

            // Results
            if (!runningBenchmark && (benchmarkFps[0] > 0 || benchmarkFps[1] > 0 || benchmarkFps[2] > 0))
            {
                ImGui::TextColored(green, "BL: %.0f", benchmarkFps[0]);
                ImGui::SameLine(90);
                ImGui::TextColored(yellow, "PB: %.0f", benchmarkFps[1]);
                ImGui::SameLine(170);
                ImGui::TextColored(accent, "OPT: %.0f", benchmarkFps[2]);

                if (ImGui::Button("Clear##Bench", ImVec2(-1, 0)))
                {
                    for (int i = 0; i < 3; ++i)
                        benchmarkFps[i] = 0.0f;
                }
            }
        }

        // =====================================================================
        // TESTING SECTION
        // =====================================================================
        if (ImGui::CollapsingHeader("Testing"))
        {
            renderTestingUI();
        }

        // =====================================================================
        // HELP (Collapsed by default)
        // =====================================================================
        if (ImGui::CollapsingHeader("Controls"))
        {
            ImGui::TextDisabled("LMB + WASD = Move");
            ImGui::TextDisabled("Q/E = Up/Down");
            ImGui::TextDisabled("Scroll = Speed");
            ImGui::TextDisabled("P = Screenshot");
            ImGui::TextDisabled("R = Rotate");
            ImGui::TextDisabled("` = Toggle Panel");
        }
    } // end if (ease > 0.3f)

    ImGui::PopStyleColor(COLOR_COUNT);
    ImGui::End();
    ImGui::Render();

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = imguiRenderPass;
    rpInfo.framebuffer = imguiFramebuffers[imageIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = swapChainManager->getSwapChainExtent();
    rpInfo.clearValueCount = 0; // <--- NO CLEARING
    rpInfo.pClearValues = nullptr;

    commandBuffer.beginRenderPass(rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer.getVkCommandBuffer(), 0);
    commandBuffer.endRenderPass();

    // === END GPU TIMER (before ending command buffer) ===
    if (waterTestingSystem && isTestModeActive)
    {
        waterTestingSystem->writeTimestampEnd(commandBuffer.getVkCommandBuffer());
    }

    updateToggleInfo(currentFrame, currentToggleInfo);

    commandBuffer.end();
}

void VulkanBase::beginRenderPass(const CommandBuffer &buffer, VkFramebuffer currentBuffer, VkExtent2D extent)
{
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = currentBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = extent;

    VkClearValue clearColor = {{0.5f, 0.2f, 0.2f, 1.0f}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(buffer.getVkCommandBuffer(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanBase::endRenderPass(const CommandBuffer &buffer)
{
    vkCmdEndRenderPass(buffer.getVkCommandBuffer());
}

void VulkanBase::recreateSwapChain()
{
    // Set flag immediately to prevent any command buffer recording
    isRecreatingSwapChain = true;

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    // ===== MARK WATER MESH AS INVALID DURING RECREATION =====
    if (waterMesh)
    {
        waterMesh->setValid(false);
    }

    // ===== CLEANUP SCENE/OFFSCREEN RESOURCES BEFORE SWAP CHAIN RECREATION =====

    // Cleanup scene color texture resources
    if (sceneColorSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, sceneColorSampler, nullptr);
        sceneColorSampler = VK_NULL_HANDLE;
    }
    if (sceneColorImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, sceneColorImageView, nullptr);
        sceneColorImageView = VK_NULL_HANDLE;
    }
    if (sceneColorImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, sceneColorImage, nullptr);
        vkFreeMemory(device, sceneColorImageMemory, nullptr);
        sceneColorImage = VK_NULL_HANDLE;
        sceneColorImageMemory = VK_NULL_HANDLE;
    }

    if (sceneReflectionFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, sceneReflectionFramebuffer, nullptr);
        sceneReflectionFramebuffer = VK_NULL_HANDLE;
    }
    if (sceneReflectionRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, sceneReflectionRenderPass, nullptr);
        sceneReflectionRenderPass = VK_NULL_HANDLE;
    }
    if (sceneReflectionImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, sceneReflectionImageView, nullptr);
        sceneReflectionImageView = VK_NULL_HANDLE;
    }
    if (sceneReflectionImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, sceneReflectionImage, nullptr);
        sceneReflectionImage = VK_NULL_HANDLE;
    }
    if (sceneReflectionImageMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, sceneReflectionImageMemory, nullptr);
        sceneReflectionImageMemory = VK_NULL_HANDLE;
    }

    if (sceneRefractionFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, sceneRefractionFramebuffer, nullptr);
        sceneRefractionFramebuffer = VK_NULL_HANDLE;
    }
    if (sceneRefractionRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, sceneRefractionRenderPass, nullptr);
        sceneRefractionRenderPass = VK_NULL_HANDLE;
    }
    if (sceneRefractionImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, sceneRefractionImageView, nullptr);
        sceneRefractionImageView = VK_NULL_HANDLE;
    }
    if (sceneRefractionImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, sceneRefractionImage, nullptr);
        sceneRefractionImage = VK_NULL_HANDLE;
    }
    if (sceneRefractionImageMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, sceneRefractionImageMemory, nullptr);
        sceneRefractionImageMemory = VK_NULL_HANDLE;
    }

    swapChainManager->cleanupSwapChain();

    swapChainManager->createSwapChain();
    swapChainManager->createImageViews();

    // IMPORTANT: Recreate render passes before recreating resources that depend on them
    vkDestroyRenderPass(device, renderPass, nullptr);
    createRenderPass();

    createColorResources();
    createDepthResources();
    createFrameBuffers();

    // ===== RECREATE SCENE/OFFSCREEN RESOURCES AFTER SWAP CHAIN =====
    createSceneColorTexture();
    createSceneRenderPassAndFramebuffer();
    createSceneReflectionTexture();
    createSceneReflectionRenderPassAndFramebuffer();
    createSceneRefractionRenderPassAndFramebuffer();

    // Update water descriptors to point at the newly-created image views/samplers
    // so descriptor sets don't reference destroyed handles.
    updateWaterDescriptors();

    createGraphicsPipeline();

    // IMPORTANT: Recreate skybox and water pipelines with the new render pass
    if (skyboxPipeline)
    {
        //    std::cout << "[DEBUG] recreateSwapChain: Destroying and recreating skybox pipeline\n";
        //    std::cout << "[DEBUG] recreateSwapChain: Using renderPass: " << renderPass << "\n";
        skyboxPipeline->destroy(device);
        skyboxPipeline->create(
            device,
            swapChainManager->getSwapChainExtent(),
            renderPass,
            descriptorSetLayout,
            skyboxDescriptorSetLayout,
            msaaSamples);
        //      std::cout << "[DEBUG] recreateSwapChain: Skybox pipeline recreated\n";
    }

    if (waterPipeline)
    {
        //    std::cout << "[DEBUG] recreateSwapChain: Destroying and recreating water pipeline\n";
        //    std::cout << "[DEBUG] recreateSwapChain: Using renderPass: " << renderPass << "\n";
        waterPipeline->destroy(device);
        waterPipeline->create(
            device,
            swapChainManager->getSwapChainExtent(),
            renderPass,
            descriptorSetLayout,
            waterDescriptorSetLayout,
            msaaSamples,
            false);
    }
    if (underwaterWaterPipeline)
    {
        underwaterWaterPipeline->destroy(device);
        underwaterWaterPipeline->create(
            device,
            swapChainManager->getSwapChainExtent(),
            renderPass,
            descriptorSetLayout,
            waterDescriptorSetLayout,
            msaaSamples,
            true);
        //    std::cout << "[DEBUG] recreateSwapChain: Water pipeline recreated\n";
    }

    // Recreate sunrays pipeline with new swap chain extent
    if (sunraysPipeline)
    {
        sunraysPipeline->destroy(device);
        sunraysPipeline->create(
            device,
            swapChainManager->getSwapChainExtent(),
            renderPass,
            descriptorSetLayout,
            waterDescriptorSetLayout,
            msaaSamples,
            true); // isSunraysPipeline = true
    }

    // Recreate ImGui framebuffers with new swapchain images
    for (auto fb : imguiFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    imguiFramebuffers.clear();
    createImGuiFramebuffers();

    createCommandBuffers();

    // ===== MARK WATER MESH AS VALID AFTER RECREATION =====
    if (waterMesh)
    {
        waterMesh->setValid(true);
    }

    // Clear the flag - swap chain recreation is complete
    isRecreatingSwapChain = false;
}

bool VulkanBase::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto &layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

bool VulkanBase::isDeviceSuitable(VkPhysicalDevice device)
{
    // Retrieve the properties and features of the device
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    // Retrieve the queue family indices for graphics and presentation
    VkUtils::QueueFamilyIndices indices = VkUtils::FindQueueFamilies(device, surface);

    // Check if the required device extensions are supported
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    // Verify if the swap chain is adequate (non-empty formats and present modes)
    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        VkUtils::SwapChainSupportDetails swapChainSupport = VkUtils::QuerySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    // Log details for diagnostics
    // std::cout << "Checking device: " << deviceProperties.deviceName << std::endl;
    // std::cout << "  Device Type: " << deviceProperties.deviceType << std::endl;
    // std::cout << "  API Version: " << deviceProperties.apiVersion << std::endl;
    // std::cout << "  Extensions Supported: " << (extensionsSupported ? "Yes" : "No") << std::endl;
    // std::cout << "  Swap Chain Adequate: " << (swapChainAdequate ? "Yes" : "No") << std::endl;
    // std::cout << "  Queue Families Complete: " << (indices.isComplete() ? "Yes" : "No") << std::endl;
    // std::cout << "  Anisotropy Support: " << (deviceFeatures.samplerAnisotropy ? "Yes" : "No") << std::endl;

    // Return true if the device is suitable for use
    return indices.isComplete() && extensionsSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
}

std::vector<const char *> VulkanBase::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (VkUtils::enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool VulkanBase::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto &extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

void VulkanBase::createAdditionalTextures()
{
    // Load and create Metalness Texture
    loadTexture("textures/vehicle_metalness.png", metalnessImage, metalnessImageMemory);
    metalnessImageView = createImageView(metalnessImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);

    // Load and create Normal Texture
    loadTexture("textures/vehicle_normal.png", normalImage, normalImageMemory);
    normalImageView = createImageView(normalImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);

    // Load and create Specular Texture
    loadTexture("textures/vehicle_specular.png", specularImage, specularImageMemory);
    specularImageView = createImageView(specularImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);

    // If using a common sampler for these textures:
    createTextureSampler();
}

void VulkanBase::loadTexture(const std::string &filePath, VkImage &textureImage, VkDeviceMemory &textureImageMemory)
{
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(filePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image: " + filePath);
    }

    VkDeviceSize imageSize = texWidth * texHeight * 4;
    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
}

void VulkanBase::createTextureImage()
{
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load("textures/texture.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
}

void VulkanBase::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = VkUtils::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanBase::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        width,
        height,
        1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

void VulkanBase::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = VkUtils::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanBase::createTextureImageView()
{
    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);
}

VkImageView VulkanBase::createCubemapImageView(VkImage image, VkFormat format)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = format;

    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create cubemap image view!");
    }
    return imageView;
}

void VulkanBase::createDepthResources()
{
    VkFormat depthFormat = findDepthFormat();

    createImage(swapChainManager->getSwapChainExtent().width, swapChainManager->getSwapChainExtent().height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1, false);
}

VkFormat VulkanBase::findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

bool VulkanBase::hasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkSampleCountFlagBits VulkanBase::getMaxUsableSampleCount()
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT)
    {
        return VK_SAMPLE_COUNT_64_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_32_BIT)
    {
        return VK_SAMPLE_COUNT_32_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_16_BIT)
    {
        return VK_SAMPLE_COUNT_16_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_8_BIT)
    {
        return VK_SAMPLE_COUNT_8_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_4_BIT)
    {
        return VK_SAMPLE_COUNT_4_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_2_BIT)
    {
        return VK_SAMPLE_COUNT_2_BIT;
    }

    return VK_SAMPLE_COUNT_1_BIT;
}

void VulkanBase::createColorResources()
{
    VkFormat colorFormat = swapChainManager->getSwapChainImageFormat();

    createImage(swapChainManager->getSwapChainExtent().width, swapChainManager->getSwapChainExtent().height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
    colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1, false);
}

void VulkanBase::createLightInfoBuffers()
{
    VkDeviceSize bufferSize = sizeof(LightInfo);

    lightInfoBuffers.resize(swapChainManager->getSwapChainImages().size());
    lightInfoBuffersMemory.resize(swapChainManager->getSwapChainImages().size());

    for (size_t i = 0; i < swapChainManager->getSwapChainImages().size(); i++)
    {
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightInfoBuffers[i],
                     lightInfoBuffersMemory[i]);
    }
}

void VulkanBase::updateLightInfoBuffer(uint32_t currentImage)
{
    LightInfo lightInfo;

    // Update light 1
    lightInfo.lights[0].position = light0Position;
    lightInfo.lights[0].color = light0Color;
    lightInfo.lights[0].intensity = light0Intensity;

    // Update light 2
    lightInfo.lights[1].position = light1Position;
    lightInfo.lights[1].color = light1Color;
    lightInfo.lights[1].intensity = light1Intensity;

    // Set ambient light properties
    lightInfo.ambientColor = ambientColor;
    lightInfo.ambientIntensity = ambientIntensity;

    // Update the camera/view position
    lightInfo.viewPos = camera.getPosition();

    // Update the uniform buffer with this data
    void *data;
    vkMapMemory(device, lightInfoBuffersMemory[currentImage], 0, sizeof(LightInfo), 0, &data);
    memcpy(data, &lightInfo, sizeof(LightInfo));
    vkUnmapMemory(device, lightInfoBuffersMemory[currentImage]);
}

void VulkanBase::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
{
    // Check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(commandBuffer,
                       image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit,
                       VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

VkFormat VulkanBase::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void VulkanBase::createSyncObjects()
{
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void VulkanBase::createCommandBuffers()
{
    commandBuffers.resize(swapChainFramebuffers.size());
    // std::cout << "Resized commandBuffers to " << commandBuffers.size() << std::endl; //uncommnet to see the command buffer size

    for (size_t i = 0; i < commandBuffers.size(); i++)
    {
        commandBuffers[i].initialize(device, commandPool.getVkCommandPool());
    }
}

void VulkanBase::createUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(UBO);

    uniformBuffers.resize(swapChainManager->getSwapChainImages().size());
    uniformBuffersMemory.resize(swapChainManager->getSwapChainImages().size());

    for (size_t i = 0; i < swapChainManager->getSwapChainImages().size(); i++)
    {
        std::tie(uniformBuffers[i], uniformBuffersMemory[i]) = VkUtils::CreateBuffer(
            device, physicalDevice, bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

void VulkanBase::createDescriptorSetLayout()
{
    // UBO for the transformation matrices
    VkDescriptorSetLayoutBinding uboLayoutBinding1{};
    uboLayoutBinding1.binding = 0;
    uboLayoutBinding1.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding1.descriptorCount = 1;
    uboLayoutBinding1.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding1.pImmutableSamplers = nullptr;

    // Sampler for the base texture (albedo/diffuse)
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    // UBO for the light information
    VkDescriptorSetLayoutBinding uboLayoutBinding2{};
    uboLayoutBinding2.binding = 2;
    uboLayoutBinding2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding2.descriptorCount = 1;
    uboLayoutBinding2.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding2.pImmutableSamplers = nullptr;

    // Sampler for the metalness texture
    VkDescriptorSetLayoutBinding metalnessLayoutBinding{};
    metalnessLayoutBinding.binding = 3;
    metalnessLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    metalnessLayoutBinding.descriptorCount = 1;
    metalnessLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    metalnessLayoutBinding.pImmutableSamplers = nullptr;

    // Sampler for the normal texture
    VkDescriptorSetLayoutBinding normalLayoutBinding{};
    normalLayoutBinding.binding = 4;
    normalLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalLayoutBinding.descriptorCount = 1;
    normalLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    normalLayoutBinding.pImmutableSamplers = nullptr;

    // Sampler for the specular texture
    VkDescriptorSetLayoutBinding specularLayoutBinding{};
    specularLayoutBinding.binding = 5;
    specularLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    specularLayoutBinding.descriptorCount = 1;
    specularLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    specularLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding toggleInfoLayoutBinding{};
    toggleInfoLayoutBinding.binding = 6;
    toggleInfoLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    toggleInfoLayoutBinding.descriptorCount = 1;
    toggleInfoLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    toggleInfoLayoutBinding.pImmutableSamplers = nullptr;

    // Combine all bindings into a single array
    std::array<VkDescriptorSetLayoutBinding, 7> bindings = {
        uboLayoutBinding1,
        samplerLayoutBinding,
        uboLayoutBinding2,
        metalnessLayoutBinding,
        normalLayoutBinding,
        specularLayoutBinding,
        toggleInfoLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void VulkanBase::createGraphicsPipeline()
{
    // CRITICAL SAFETY CHECK
    if (renderPass == VK_NULL_HANDLE)
    {
        throw std::runtime_error("createGraphicsPipeline: renderPass is NULL!");
    }
    if (renderPass == imguiRenderPass)
    {
        throw std::runtime_error("createGraphicsPipeline: renderPass is imguiRenderPass! This is a critical bug.");
    }

    shader3D = std::make_unique<Shader3D>(device, "shaders/3d_shader.vert.spv", "shaders/3d_shader.frag.spv");

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptionsArray = Vertex::getAttributeDescriptions();
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(
        attributeDescriptionsArray.begin(), attributeDescriptionsArray.end());

    if (wireframeEnabled)
    {
        attributeDescriptions.erase(
            std::remove_if(attributeDescriptions.begin(), attributeDescriptions.end(),
                           [](const VkVertexInputAttributeDescription &desc)
                           {
                               return desc.location >= 3;
                           }),
            attributeDescriptions.end());
    }

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainManager->getSwapChainExtent().width);
    viewport.height = static_cast<float>(swapChainManager->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainManager->getSwapChainExtent();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = wireframeEnabled ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.minSampleShading = .25f;
    multisampling.rasterizationSamples = msaaSamples;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDescriptorSetLayout, 2> setLayouts = {descriptorSetLayout, waterDescriptorSetLayout};

    // --- CRITICAL FIX START ---
    // Define the Push Constant Range for the MAIN pipeline (Ocean Floor)
    // This must match the WaterPipeline range (96 bytes, Vertex + Fragment)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = 96;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();

    // Enable the push constant range
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    // --- CRITICAL FIX END ---

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    auto shaderStages = shader3D->getShaderStages();

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    std::cout << "[DEBUG] createGraphicsPipeline: Created graphics pipeline with Push Constants (Size 80)\n";
}

void VulkanBase::updatePipelineIfNeeded()
{
    if (currentWireframeState != wireframeEnabled)
    {
        vkDeviceWaitIdle(device);
        // std::cout << "[DEBUG] updatePipelineIfNeeded: Destroying graphics pipeline: " << graphicsPipeline << "\n";
        // std::cout << "[DEBUG] updatePipelineIfNeeded: Current renderPass: " << renderPass << "\n";
        // std::cout << "[DEBUG] updatePipelineIfNeeded: ImGui renderPass: " << imguiRenderPass << "\n";
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        createGraphicsPipeline();
        // std::cout << "[DEBUG] updatePipelineIfNeeded: Recreated graphics pipeline: " << graphicsPipeline << "\n";
        currentWireframeState = wireframeEnabled;
    }
}

void VulkanBase::createToggleInfoBuffers()
{
    VkDeviceSize bufferSize = sizeof(ToggleInfo);

    toggleInfoBuffers.resize(swapChainManager->getSwapChainImages().size());
    toggleInfoBuffersMemory.resize(swapChainManager->getSwapChainImages().size());

    for (size_t i = 0; i < toggleInfoBuffers.size(); i++)
    {
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     toggleInfoBuffers[i], toggleInfoBuffersMemory[i]);
    }
}

void VulkanBase::updateToggleInfo(uint32_t currentImage, const ToggleInfo &toggleInfo)
{
    void *data;
    vkMapMemory(device, toggleInfoBuffersMemory[currentImage], 0, sizeof(ToggleInfo), 0, &data);
    memcpy(data, &toggleInfo, sizeof(ToggleInfo));
    vkUnmapMemory(device, toggleInfoBuffersMemory[currentImage]);
}

void VulkanBase::loadSceneFromJson(const std::string &sceneFilePath)
{
    // Load the scene objects from the JSON file
    std::vector<SceneObject> sceneObjects = ModelLoader::loadSceneFromJson(sceneFilePath);

    uint32_t indexOffset = static_cast<uint32_t>(vertices.size()); // Offset for indices

    for (const auto &obj : sceneObjects)
    {
        // Append the object's vertices to the main vertices vector
        vertices.insert(vertices.end(), obj.vertices.begin(), obj.vertices.end());

        // Append the object's indices to the main indices vector with the offset applied
        for (const auto &index : obj.indices)
        {
            indices.push_back(index + indexOffset);
        }

        // Update the index offset for the next object
        indexOffset += static_cast<uint32_t>(obj.vertices.size());
    }

    // After aggregating all vertices and indices, you can create buffers
    createVertexBuffer();
    createIndexBuffer();
}

void VulkanBase::createScreenshotImage(VkExtent2D extent, VkFormat format)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR; // Important for CPU access
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateImage(device, &imageInfo, nullptr, &screenshotImage);

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, screenshotImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = VkUtils::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(device, &allocInfo, nullptr, &screenshotImageMemory);
    vkBindImageMemory(device, screenshotImage, screenshotImageMemory, 0);
}

void VulkanBase::blitImage(VkImage srcImage, VkImage dstImage, VkExtent2D extent)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dstImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blit{};
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;

    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = 0;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    vkCmdBlitImage(
        commandBuffer,
        srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit,
        VK_FILTER_NEAREST);

    endSingleTimeCommands(commandBuffer);
}

void VulkanBase::saveScreenshot(VkImage image, VkExtent2D extent, const std::string &filename)
{
    VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(device, image, &subResource, &subResourceLayout);

    const char *data;
    vkMapMemory(device, screenshotImageMemory, 0, VK_WHOLE_SIZE, 0, (void **)&data);
    data += subResourceLayout.offset;

    // Calculate the exact number of bytes per row
    int rowSize = extent.width * 4; // Assuming 4 bytes per pixel (RGBA)
    std::vector<unsigned char> pixels(rowSize * extent.height);

    for (int y = 0; y < extent.height; y++)
    {
        for (int x = 0; x < extent.width; x++)
        {
            unsigned char r = data[x * 4 + 0];
            unsigned char g = data[x * 4 + 1];
            unsigned char b = data[x * 4 + 2];
            unsigned char a = data[x * 4 + 3];

            // Swap R and B channels (convert RGBA to BGRA)
            pixels[y * rowSize + x * 4 + 0] = b;
            pixels[y * rowSize + x * 4 + 1] = g;
            pixels[y * rowSize + x * 4 + 2] = r;
            pixels[y * rowSize + x * 4 + 3] = a;
        }
        data += subResourceLayout.rowPitch; // Move to the next row (considering padding)
    }

    vkUnmapMemory(device, screenshotImageMemory);

    // Write to a JPEG file using stb_image_write
    stbi_write_jpg(filename.c_str(), extent.width, extent.height, 4, pixels.data(), 100);
}

void VulkanBase::takeScreenshot()
{
    if (!captureScreenshot)
    {
        return; // Only take a screenshot if requested
    }

    VkExtent2D extent = swapChainManager->getSwapChainExtent();
    VkFormat format = swapChainManager->getSwapChainImageFormat();
    const auto &swapChainImages = swapChainManager->getSwapChainImages();

    std::cout << "currentFrame: " << currentFrame << ", swapChainImages size: " << swapChainImages.size() << std::endl;

    if (currentFrame >= swapChainImages.size())
    {
        std::cerr << "Error: currentFrame index out of range!" << std::endl;
        captureScreenshot = false; // Reset the flag
        return;
    }

    createScreenshotImage(extent, format);

    // Transition the swapchain image to TRANSFER_SRC layout before copying
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapChainImages[currentFrame];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0; // Or appropriate access mask
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);

    // Blit the swapchain image to the screenshot image
    blitImage(swapChainImages[currentFrame], screenshotImage, extent);

    // Transition the swapchain image back to PRESENT_SRC layout
    commandBuffer = beginSingleTimeCommands();
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; // Or appropriate access mask
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);

    static int screenshotCount = 0;
    std::string filename = "ScreenShots/screenshot_" + std::to_string(++screenshotCount) + ".jpg";
    saveScreenshot(screenshotImage, extent, filename);

    // Reset the flag
    captureScreenshot = false;

    // Cleanup
    vkDestroyImage(device, screenshotImage, nullptr);
    vkFreeMemory(device, screenshotImageMemory, nullptr);
}

void VulkanBase::createDescriptorPool()
{
    descriptorPool = std::make_unique<DAEDescriptorPool<UBO>>(device, swapChainManager->getSwapChainImages().size());
    VkUtils::VulkanContext context{device, physicalDevice};
    descriptorPool->initialize(context);
}

void VulkanBase::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    // === ADD THIS NEW BLOCK ===
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = 0;                         // No need to wait on previous writes
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // Must be ready for shader reads

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;          // Start as soon as possible
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // Wait until fragment shader
    }
    // ==========================
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

VkCommandBuffer VulkanBase::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool.getVkCommandPool();
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanBase::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool.getVkCommandPool(), 1, &commandBuffer);
}

void VulkanBase::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(swapChainManager->getSwapChainImages().size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool->getDescriptorPool();
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(swapChainManager->getSwapChainImages().size());
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < descriptorSets.size(); i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UBO);

        if (i >= lightInfoBuffers.size())
        {
            throw std::runtime_error("lightInfoBuffers is out of range!");
        }

        VkDescriptorBufferInfo lightInfoBufferInfo{};
        lightInfoBufferInfo.buffer = lightInfoBuffers[i];
        lightInfoBufferInfo.offset = 0;
        lightInfoBufferInfo.range = sizeof(LightInfo);

        VkDescriptorImageInfo baseImageInfo{};
        baseImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        baseImageInfo.imageView = textureImageView;
        baseImageInfo.sampler = textureSampler;

        // ImageInfo for Metalness Texture
        VkDescriptorImageInfo metalnessImageInfo{};
        metalnessImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        metalnessImageInfo.imageView = metalnessImageView;
        metalnessImageInfo.sampler = textureSampler;

        // ImageInfo for Normal Texture
        VkDescriptorImageInfo normalImageInfo{};
        normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalImageInfo.imageView = normalImageView;
        normalImageInfo.sampler = textureSampler;

        // ImageInfo for Specular Texture
        VkDescriptorImageInfo specularImageInfo{};
        specularImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        specularImageInfo.imageView = specularImageView;
        specularImageInfo.sampler = textureSampler;

        if (i >= toggleInfoBuffers.size())
        {
            throw std::runtime_error("toggleInfoBuffers is out of range!");
        }

        if (!swapChainManager)
        {
            throw std::runtime_error("swapChainManager is not initialized!");
        }

        auto swapChainImagesSize = swapChainManager->getSwapChainImages().size();
        if (swapChainImagesSize == 0)
        {
            throw std::runtime_error("swapChainImages has no elements!");
        }

        toggleInfoBuffers.resize(swapChainImagesSize);

        VkDescriptorBufferInfo toggleInfoBufferInfo{};
        toggleInfoBufferInfo.buffer = toggleInfoBuffers[i];
        toggleInfoBufferInfo.offset = 0;
        toggleInfoBufferInfo.range = sizeof(ToggleInfo);

        std::array<VkWriteDescriptorSet, 7> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &baseImageInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2; // Binding index for LightInfo
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &lightInfoBufferInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSets[i];
        descriptorWrites[3].dstBinding = 3; // Binding index for Metalness texture
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pImageInfo = &metalnessImageInfo;

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = descriptorSets[i];
        descriptorWrites[4].dstBinding = 4; // Binding index for Normal texture
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &normalImageInfo;

        descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[5].dstSet = descriptorSets[i];
        descriptorWrites[5].dstBinding = 5; // Binding index for Specular texture
        descriptorWrites[5].dstArrayElement = 0;
        descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[5].descriptorCount = 1;
        descriptorWrites[5].pImageInfo = &specularImageInfo;

        descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[6].dstSet = descriptorSets[i];
        descriptorWrites[6].dstBinding = 6; // Binding index for ToggleInfo
        descriptorWrites[6].dstArrayElement = 0;
        descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pBufferInfo = &toggleInfoBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

VkImageView VulkanBase::createImageView(
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    uint32_t mipLevels,
    bool isCubemap)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;

    if (isCubemap)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        viewInfo.subresourceRange.layerCount = 6;
    }
    else
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.subresourceRange.layerCount = 1;
    }

    viewInfo.format = format;

    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image view!");
    }

    return imageView;
}

void VulkanBase::createTextureSampler()
{

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void VulkanBase::drawFrame()
{
    // SAFETY: Skip entire frame if swap chain recreation is in progress or framebuffer was resized
    if (isRecreatingSwapChain || framebufferResized)
    {
        // Trigger recreation now and return
        if (framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        return;
    }

    // Wait for the frame's fence to ensure the GPU has finished
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Double-check after fence wait - resize callback could have fired during wait
    if (isRecreatingSwapChain || framebufferResized)
    {
        if (framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        return;
    }

    updatePipelineIfNeeded();

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device,
        swapChainManager->getSwapChain(),
        UINT64_MAX,
        imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE,
        &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Triple-check before recording - one more safety check
    if (isRecreatingSwapChain)
    {
        return;
    }

    // Reset the fence so we can use it for the next submission
    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    //  UPDATE UNIFORMS FIRST (before recording command buffer)
    updateUniformBuffer(imageIndex);
    updateLightInfoBuffer(imageIndex);
    updateToggleInfo(imageIndex, currentToggleInfo);

    //  THEN reset and record the command buffer for this frame (use currentFrame, not imageIndex)
    vkResetCommandBuffer(commandBuffers[currentFrame].getVkCommandBuffer(), 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    // Submit the command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    VkCommandBuffer commandBuffer = commandBuffers[currentFrame].getVkCommandBuffer();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapChainManager->getSwapChain()};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        // Set recreation flag FIRST before anything else
        isRecreatingSwapChain = true;

        // Disable water mesh immediately when resize is detected
        if (waterMesh)
        {
            waterMesh->setValid(false);
        }

        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    // Move to next frame
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanBase::updateUniformBuffer(uint32_t currentImage)
{
    // Standard UBO struct (used for Refraction and Main Pass)
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // 1. NORMAL CAMERA (Refraction Pass / Main Pass)
    UBO ubo{};
    // Use standard camera view/projection
    ubo.view = camera.getViewMatrix();
    ubo.proj = glm::perspective(glm::radians(camera.zoom), swapChainManager->getSwapChainExtent().width / (float)swapChainManager->getSwapChainExtent().height, 0.1f, 1000.0f);
    ubo.proj[1][1] *= -1; // Flip Y for Vulkan
    ubo.model = glm::mat4(1.0f);
    ubo.lightPos = glm::vec4(light0Position, 1.0f);
    // ==========================================

    ubo.viewPos = glm::vec4(camera.getPosition(), 1.0f);
    // 2. REFLECTION CAMERA (Reflection Pass)
    UBO uboRefl{};
    uboRefl.proj = ubo.proj;

    // Get current camera position and direction
    glm::vec3 camPos = camera.getPosition();

    glm::vec3 camFront = camera.front;

    float waterHeight = 0.0f; // Assuming water is at Y=0

    // a) Mirror the camera position across the water plane (Y=0)
    glm::vec3 reflCamPos = camPos;
    reflCamPos.y = 2.0f * waterHeight - camPos.y;

    // b) Mirror the camera direction (pitch must be flipped)
    glm::vec3 reflCamFront = camFront;
    reflCamFront.y *= -1.0f;

    // c) Create the new reflection view matrix
    // CRITICAL: Flip the UP vector to glm::vec3(0.0f, -1.0f, 0.0f)
    uboRefl.view = glm::lookAt(
        reflCamPos,                  // Eye position (mirrored)
        reflCamPos + reflCamFront,   // Target position (mirrored direction)
        glm::vec3(0.0f, -1.0f, 0.0f) // Mirrored Up Vector
    );
    uboRefl.model = glm::mat4(1.0f);

    // Update the NORMAL UBO for the Main Pass
    void *data;
    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);

    // Store the reflection view matrix for use in recordCommandBuffer
    // (Ensure you have 'glm::mat4 reflectionViewMatrix;' as a member variable in VulkanBase.h)
    reflectionViewMatrix = uboRefl.view;
}

void VulkanBase::createSkyboxDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding = 0;
    samplerBinding.descriptorCount = 1;
    samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.pImmutableSamplers = nullptr;
    samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT; // include both if shader samples in vertex

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &samplerBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &skyboxDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create skybox descriptor set layout!");
}

void VulkanBase::createSkyboxDescriptorSet()
{
    // Sanity checks: must have valid image view and sampler before allocating descriptor
    if (skyboxImageView == VK_NULL_HANDLE)
    {
        throw std::runtime_error("createSkyboxDescriptorSet: skyboxImageView is VK_NULL_HANDLE. Cubemap creation failed or not called yet.");
    }
    if (skyboxSampler == VK_NULL_HANDLE)
    {
        throw std::runtime_error("createSkyboxDescriptorSet: skyboxSampler is VK_NULL_HANDLE. Cubemap sampler creation failed.");
    }

    // ---- Allocate descriptor set ----
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = skyboxDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &skyboxDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &skyboxDescriptorSet) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate skybox descriptor set!");

    // ---- Descriptor for cubemap ----
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = skyboxImageView;
    imageInfo.sampler = skyboxSampler;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = skyboxDescriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    std::cout << "createSkyboxDescriptorSet: imageView=" << skyboxImageView << " sampler=" << skyboxSampler << "\n";

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    // std::cout << "=== DEBUG: Skybox Descriptor Updated ===\n";
    // std::cout << "DescriptorSet = " << skyboxDescriptorSet << "\n";
    // std::cout << "View in descriptor = " << skyboxImageView
    //           << ", Sampler = " << skyboxSampler << "\n";
}

void VulkanBase::createSkyboxDescriptorPool()
{
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &skyboxDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create skybox descriptor pool!");
}

void VulkanBase::createWaterResources()
{
    // -----------------------------------------------------------------
    // REPLACE PLACEHOLDERS WITH REAL TEXTURE LOADING
    // -----------------------------------------------------------------

    // 1. Load Normal Map
    // This gives the water its wave shape.
    // Ensure "textures/water_normal.png" exists.
    try
    {
        loadTexture("textures/water_normal.jpg", waterNormalImage, waterNormalImageMemory);
        // loadTexture updates 'mipLevels', use it to create the view
        waterNormalImageView = createImageView(waterNormalImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);
        std::cout << "[Water] Loaded water_normal.jpg\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Water] Error loading water_normal.jpg: " << e.what() << "\n";
        // Fallback logic could go here, but for now we crash or handle via existing exception
        throw;
    }

    // 2. Load DUDV Map (Distortion)
    // This creates the wobbly distortion effect and refraction.
    // Ensure "textures/water_dudv.png" exists.
    try
    {
        loadTexture("textures/water_dudv.jpg", waterDudvImage, waterDudvImageMemory);
        waterDudvImageView = createImageView(waterDudvImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);
        std::cout << "[Water] Loaded water_dudv.jpg\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Water] Error loading water_dudv.jpg: " << e.what() << "\n";
        throw;
    }

    // 3. Load Caustics Map
    // This creates the light patterns on the ocean floor.
    // Ensure "textures/water_caustics.png" exists.
    try
    {
        loadTexture("textures/water_caustic.jpg", waterCausticImage, waterCausticImageMemory);
        waterCausticImageView = createImageView(waterCausticImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);
        std::cout << "[Water] Loaded water_caustic.jpg\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Water] Error loading water_caustic.jpg: " << e.what() << "\n";
        throw;
    }
}

void VulkanBase::createWaterDescriptorSetLayout()
{
    // We have 5 bindings (0-4)
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};

    // binding 0 ? scene color texture (RENAMED to Refraction)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 1 ? water normal map
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 2 ? DUDV map
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 3 ? caustic texture
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 4 ? scene reflection texture
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = (uint32_t)bindings.size();
    info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &waterDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create water descriptor set layout!");
}

void VulkanBase::createWaterDescriptorPool()
{
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 4; // we use 4 bindings

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 1;
    info.poolSizeCount = 1;
    info.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &info, nullptr, &waterDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create water descriptor pool!");
}

void VulkanBase::createWaterDescriptors()
{
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = waterDescriptorPool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &waterDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &alloc, &waterDescriptorSet) != VK_SUCCESS)
        throw std::runtime_error("Failed allocating water descriptor set!");
}

void VulkanBase::updateWaterDescriptors()
{
    // Validate that all required handles are valid before updating descriptors
    if (waterDescriptorSet == VK_NULL_HANDLE ||
        sceneColorImageView == VK_NULL_HANDLE ||
        sceneColorSampler == VK_NULL_HANDLE ||
        waterNormalImageView == VK_NULL_HANDLE ||
        waterSampler == VK_NULL_HANDLE ||
        waterDudvImageView == VK_NULL_HANDLE ||
        waterCausticImageView == VK_NULL_HANDLE ||
        sceneReflectionImageView == VK_NULL_HANDLE ||
        sceneReflectionSampler == VK_NULL_HANDLE)
    {
        // Validate that all required handles are valid before updating descriptors
        bool missingResources = false;

        if (waterDescriptorSet == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'waterDescriptorSet' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (sceneColorImageView == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'sceneColorImageView' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (sceneColorSampler == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'sceneColorSampler' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (waterNormalImageView == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'waterNormalImageView' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (waterSampler == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'waterSampler' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (waterDudvImageView == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'waterDudvImageView' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (waterCausticImageView == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'waterCausticImageView' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (sceneReflectionImageView == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'sceneReflectionImageView' is VK_NULL_HANDLE\n";
            missingResources = true;
        }
        if (sceneReflectionSampler == VK_NULL_HANDLE)
        {
            std::cout << "[ERROR] updateWaterDescriptors: 'sceneReflectionSampler' is VK_NULL_HANDLE\n";
            missingResources = true;
        }

        if (missingResources)
        {
            std::cout << "[WARNING] updateWaterDescriptors: Skipping update due to above invalid resources.\n";
            return;
        }
    }

    // Image info for the scene reflection/refraction texture
    VkDescriptorImageInfo sceneColorInfo{};
    sceneColorInfo.imageView = sceneColorImageView;
    sceneColorInfo.sampler = sceneColorSampler;
    sceneColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Image info for normal map
    VkDescriptorImageInfo normalInfo{};
    normalInfo.imageView = waterNormalImageView;
    normalInfo.sampler = waterSampler;
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Image info for DUDV distortion map
    VkDescriptorImageInfo dudvInfo{};
    dudvInfo.imageView = waterDudvImageView;
    dudvInfo.sampler = waterSampler;
    dudvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Image info for caustic texture
    VkDescriptorImageInfo causticInfo{};
    causticInfo.imageView = waterCausticImageView;
    causticInfo.sampler = waterSampler;
    causticInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Image info for scene reflection texture
    VkDescriptorImageInfo reflectionInfo{};
    reflectionInfo.imageView = sceneReflectionImageView;
    reflectionInfo.sampler = sceneReflectionSampler;
    reflectionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // ------------------------------
    // Write all descriptor bindings
    // ------------------------------

    std::array<VkWriteDescriptorSet, 5> writes{};

    // Binding 0 ? scene color (refraction)
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = waterDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &sceneColorInfo;

    // Binding 1 ? normal map
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = waterDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &normalInfo;

    // Binding 2 ? DUDV map
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = waterDescriptorSet;
    writes[2].dstBinding = 2;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &dudvInfo;

    // Binding 3 ? caustics
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = waterDescriptorSet;
    writes[3].dstBinding = 3;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &causticInfo;

    // Binding 4 ? reflection texture
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = waterDescriptorSet;
    writes[4].dstBinding = 4;
    writes[4].dstArrayElement = 0;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].descriptorCount = 1;
    writes[4].pImageInfo = &reflectionInfo;

    // Update all descriptors
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    std::cout << "[Water] Descriptor updated successfully.\n";
}

void VulkanBase::createWaterSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;

    // IMPORTANT: This allows the water texture to repeat infinitely,
    // which enables the scrolling/animation effect.
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.anisotropyEnable = VK_TRUE; // Turn on Anisotropy for sharper water at angles
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1000.0f; // Allow high mip levels

    if (vkCreateSampler(device, &samplerInfo, nullptr, &waterSampler) != VK_SUCCESS)
        throw std::runtime_error("Failed to create water sampler!");
}

void VulkanBase::createSceneColorTexture()
{
    // Extent
    VkExtent2D extent = swapChainManager->getSwapChainExtent();
    VkFormat format = swapChainManager->getSwapChainImageFormat();

    // Create a single-sampled image (sampleable) to render the scene into
    // usage: color attachment + sampled
    createImage(
        extent.width,
        extent.height,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        sceneColorImage,
        sceneColorImageMemory);

    // Create image view (2D)
    sceneColorImageView = createImageView(sceneColorImage, format, VK_IMAGE_ASPECT_COLOR_BIT, 1, false);

    // Create sampler for sampling the offscreen image in the water shader
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sceneColorSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneColorTexture: failed to create sampler!");
    }

    transitionImageLayout(sceneColorImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    // std::cout << "[DEBUG] createSceneColorTexture: created scene color image/view/sampler\n";
    sceneOffscreenReady = true;
}

void VulkanBase::createSceneReflectionTexture()
{
    VkExtent2D extent = swapChainManager->getSwapChainExtent();
    uint32_t mipLevels = 1; // no mips needed for render target

    createImage(
        extent.width,
        extent.height,
        mipLevels,
        VK_SAMPLE_COUNT_1_BIT,
        swapChainManager->getSwapChainImageFormat(),
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        sceneReflectionImage,
        sceneReflectionImageMemory);

    // create image view
    sceneReflectionImageView = createImageView(sceneReflectionImage, swapChainManager->getSwapChainImageFormat(), VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);

    // create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sceneReflectionSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create scene reflection sampler!");
    }

    // Ensure the reflection image is transitioned to a shader-readable layout so
    // it can be sampled even if the offscreen reflection pass is not executed.
    VkFormat format = swapChainManager->getSwapChainImageFormat();
    transitionImageLayout(sceneReflectionImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);
}

void VulkanBase::createSceneRefractionTexture()
{
    VkExtent2D extent = swapChainManager->getSwapChainExtent();
    uint32_t mipLevels = 1; // no mips needed for render target

    createImage(
        extent.width,
        extent.height,
        mipLevels,
        VK_SAMPLE_COUNT_1_BIT,
        swapChainManager->getSwapChainImageFormat(),
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        sceneRefractionImage,
        sceneRefractionImageMemory);

    sceneRefractionImageView = createImageView(sceneRefractionImage, swapChainManager->getSwapChainImageFormat(), VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, false);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sceneRefractionSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create scene refraction sampler!");
    }
}

void VulkanBase::createSceneReflectionRenderPassAndFramebuffer()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainManager->getSwapChainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // we'll transition to SHADER_READ after rendering

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = nullptr; // No depth attachment for offscreen reflection pass

    std::array<VkAttachmentDescription, 1> attachments = {colorAttachment};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &sceneReflectionRenderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create reflection render pass!");
    }

    std::array<VkImageView, 1> attachmentsViews = {sceneReflectionImageView};

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = sceneReflectionRenderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachmentsViews.size());
    fbInfo.pAttachments = attachmentsViews.data();
    fbInfo.width = swapChainManager->getSwapChainExtent().width;
    fbInfo.height = swapChainManager->getSwapChainExtent().height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &sceneReflectionFramebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create reflection framebuffer!");
    }
}

void VulkanBase::createSceneRefractionRenderPassAndFramebuffer()
{
    if (refractionExtent.width == 0 || refractionExtent.height == 0)
    {
        refractionExtent = swapChainManager->getSwapChainExtent();
    }

    VkExtent2D extent = refractionExtent;
    VkFormat format = swapChainManager->getSwapChainImageFormat();

    createImage(
        extent.width,
        extent.height,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        sceneRefractionImage,
        sceneRefractionImageMemory);

    sceneRefractionImageView = createImageView(sceneRefractionImage, format, VK_IMAGE_ASPECT_COLOR_BIT, 1, false);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sceneRefractionSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneRefractionRenderPassAndFramebuffer: failed to create sampler!");
    }

    // If the refraction pass is not executed before sampling, make sure the image
    // is at least transitioned to SHADER_READ_ONLY_OPTIMAL so shaders can sample it.
    transitionImageLayout(sceneRefractionImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = nullptr; // No depth attachment for offscreen refraction pass

    std::array<VkAttachmentDescription, 1> attachments = {colorAttachment};

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &sceneRefractionRenderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneRefractionRenderPassAndFramebuffer: failed to create render pass");
    }

    // Ensure the image views exist
    if (sceneRefractionImageView == VK_NULL_HANDLE)
    {
        throw std::runtime_error("createSceneRefractionRenderPassAndFramebuffer: sceneRefractionImageView is VK_NULL_HANDLE");
    }

    std::array<VkImageView, 1> attachmentsViews = {sceneRefractionImageView};

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = sceneRefractionRenderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachmentsViews.size());
    fbInfo.pAttachments = attachmentsViews.data();
    fbInfo.width = extent.width;
    fbInfo.height = extent.height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &sceneRefractionFramebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneRefractionRenderPassAndFramebuffer: failed to create framebuffer");
    }

    // std::cout << "[DEBUG] createSceneRefractionRenderPassAndFramebuffer: created\n";
}

void VulkanBase::createSceneRenderPassAndFramebuffer()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainManager->getSwapChainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // we want to keep data so we can sample it
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // final layout must be SHADER_READ_ONLY_OPTIMAL so we can sample it in subsequent pass
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dependencyFlags = 0;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &sceneRenderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneRenderPassAndFramebuffer: failed to create render pass");
    }

    VkImageView attachments[1] = {sceneColorImageView};

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = sceneRenderPass;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = attachments;
    fbInfo.width = swapChainManager->getSwapChainExtent().width;
    fbInfo.height = swapChainManager->getSwapChainExtent().height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &sceneFramebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("createSceneRenderPassAndFramebuffer: failed to create framebuffer");
    }

    // std::cout << "[DEBUG] createSceneRenderPassAndFramebuffer: created\n";
}

void VulkanBase::createWaterDescriptorSet()
{
    // DEBUG: Validate all image views are valid
    // std::cout << "\n=== WATER DESCRIPTOR SET CREATION DEBUG ===\n";
    // std::cout << "sceneColorImageView: " << sceneColorImageView << " (should not be null)\n";
    // std::cout << "waterNormalImageView: " << waterNormalImageView << "\n";
    // std::cout << "waterDudvImageView: " << waterDudvImageView << "\n";
    // std::cout << "waterCausticImageView: " << waterCausticImageView << "\n";
    // std::cout << "sceneReflectionImageView: " << sceneReflectionImageView << "\n";

    // Validate samplers
    // std::cout << "sceneColorSampler: " << sceneColorSampler << "\n";
    // std::cout << "textureSampler: " << textureSampler << "\n";
    // std::cout << "sceneReflectionSampler: " << sceneReflectionSampler << "\n";
    // std::cout << "=========================================\n\n";

    VkDescriptorPool pool = descriptorPool->getDescriptorPool();

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &waterDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &waterDescriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate water descriptor set!");
    }

    // Binding 0: refractionTex (sceneColorImageView)
    VkDescriptorImageInfo sceneColorInfo{};
    sceneColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sceneColorInfo.imageView = sceneColorImageView;
    sceneColorInfo.sampler = sceneColorSampler;

    // Binding 1: waterNormalMap
    VkDescriptorImageInfo normalInfo{};
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalInfo.imageView = waterNormalImageView;
    normalInfo.sampler = textureSampler;

    // Binding 2: waterDudvMap
    VkDescriptorImageInfo dudvInfo{};
    dudvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dudvInfo.imageView = waterDudvImageView;
    dudvInfo.sampler = textureSampler;

    // Binding 3: causticTex
    VkDescriptorImageInfo causticInfo{};
    causticInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // FIX: Corrected from ca-usticInfo to causticInfo
    causticInfo.imageView = waterCausticImageView;
    causticInfo.sampler = textureSampler;

    // Binding 4: reflectionTex (sceneReflectionImageView)
    VkDescriptorImageInfo reflectionInfo{};
    reflectionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    reflectionInfo.imageView = sceneReflectionImageView;
    reflectionInfo.sampler = sceneReflectionSampler;

    std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

    //  Binding 0 (Refraction)
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = waterDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &sceneColorInfo;

    //  Binding 1 (Normal Map)
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = waterDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &normalInfo;

    //  Binding 2 (DUDV Map)
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = waterDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pImageInfo = &dudvInfo;

    //  Binding 3 (Caustic)
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = waterDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pImageInfo = &causticInfo;

    //  Binding 4 (Reflection)
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = waterDescriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pImageInfo = &reflectionInfo;

    vkUpdateDescriptorSets(device,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(),
                           0, nullptr);
}

glm::mat4 VulkanBase::calculateProjectionMatrix()
{
    VkExtent2D extent = swapChainManager->getSwapChainExtent();
    float aspectRatio = (float)extent.width / (float)extent.height;

    glm::mat4 proj = glm::perspective(
        glm::radians(camera.zoom),
        aspectRatio,
        0.1f,
        1000.0f);

    proj[1][1] *= -1;

    return proj;
}
// ==============================================================================
// 1. REFLECTION PASS IMPLEMENTATION
// ==============================================================================
void VulkanBase::recordReflectionPass(CommandBuffer &commandBuffer, uint32_t imageIndex)
{
    // --- TEMPORARILY MANIPULATE CAMERA STATE ---
    // Store original state to restore later
    glm::vec3 originalPos = this->camera.position;
    float originalPitch = this->camera.pitch;

    glm::mat4 currentProj = this->calculateProjectionMatrix();

    glm::vec3 reflectedPos = originalPos;
    reflectedPos.y = -originalPos.y + 0.5f;

    this->camera.position = reflectedPos;
    this->camera.pitch = -originalPitch; // Flip pitch angle for reflection

    glm::mat4 reflectedView = this->camera.getViewMatrix();

    this->camera.position = originalPos;
    this->camera.pitch = originalPitch;
    // --------------------------------------------------

    // CRITICAL: Update UBO with the reflected view matrix for this draw call
    // You MUST call your UBO update function here!
    // Example: this->updateGlobalUBO(imageIndex, reflectedView, originalProj);

    // --------------------------------------------------

    // 2. BEGIN REFLECTION RENDER PASS
    VkClearValue reflectionClearValue{};
    reflectionClearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo reflectionPassInfo{};
    reflectionPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    reflectionPassInfo.renderPass = this->sceneReflectionRenderPass;
    reflectionPassInfo.framebuffer = this->sceneReflectionFramebuffer;
    reflectionPassInfo.renderArea.offset = {0, 0};
    reflectionPassInfo.renderArea.extent = this->reflectionExtent;
    reflectionPassInfo.clearValueCount = 1;
    reflectionPassInfo.pClearValues = &reflectionClearValue;

    vkCmdBeginRenderPass(commandBuffer.getVkCommandBuffer(), &reflectionPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // 3. DRAW SCENE
    this->DrawSceneObjects(commandBuffer, imageIndex);
    // --- 4. END PASS ---
    vkCmdEndRenderPass(commandBuffer.getVkCommandBuffer());

    // Restore UBO to the original view matrix after the pass is done
    // this->updateUniformBuffer(imageIndex, camera.getViewMatrix(), camera.getProjectionMatrix());
}

// ==============================================================================
// 2. REFRACTION PASS IMPLEMENTATION
// ==============================================================================
void VulkanBase::recordRefractionPass(CommandBuffer &commandBuffer, uint32_t imageIndex)
{
    updateUniformBuffer(imageIndex);

    VkClearValue refractionClearValue{};
    refractionClearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo refractionPassInfo{};
    refractionPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    refractionPassInfo.renderPass = this->sceneRefractionRenderPass;
    refractionPassInfo.framebuffer = this->sceneRefractionFramebuffer;
    refractionPassInfo.renderArea.offset = {0, 0};
    refractionPassInfo.renderArea.extent = this->refractionExtent;
    refractionPassInfo.clearValueCount = 1;
    refractionPassInfo.pClearValues = &refractionClearValue;

    vkCmdBeginRenderPass(commandBuffer.getVkCommandBuffer(), &refractionPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    this->DrawSceneObjects(commandBuffer, imageIndex);

    vkCmdEndRenderPass(commandBuffer.getVkCommandBuffer());
}

// ==============================================================================
// 3. IMAGE BARRIERS IMPLEMENTATION
// ==============================================================================
void VulkanBase::insertWaterTextureBarriers(CommandBuffer &commandBuffer)
{
    VkImageMemoryBarrier barrierReflection{};
    barrierReflection.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierReflection.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrierReflection.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrierReflection.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierReflection.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierReflection.image = sceneReflectionImage;
    barrierReflection.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierReflection.subresourceRange.baseMipLevel = 0;
    barrierReflection.subresourceRange.levelCount = 1;
    barrierReflection.subresourceRange.baseArrayLayer = 0;
    barrierReflection.subresourceRange.layerCount = 1;
    barrierReflection.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrierReflection.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkImageMemoryBarrier barrierRefraction = barrierReflection;
    barrierRefraction.image = sceneRefractionImage;
    std::array<VkImageMemoryBarrier, 2> barriers = {barrierReflection, barrierRefraction};
    vkCmdPipelineBarrier(
        commandBuffer.getVkCommandBuffer(),
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());
}

void VulkanBase::DrawSceneObjects(CommandBuffer &commandBuffer, uint32_t imageIndex)
{
    if (!useSolidBackground)
    {
        //  std::cout << "[DEBUG] About to bind skybox pipeline: " << skyboxPipeline->pipeline << "\n";
        skyboxPipeline->bind(commandBuffer.getVkCommandBuffer());

        std::array<VkDescriptorSet, 2> sets = {descriptorSets[imageIndex], skyboxDescriptorSet};
        vkCmdBindDescriptorSets(
            commandBuffer.getVkCommandBuffer(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            skyboxPipeline->layout,
            0,
            static_cast<uint32_t>(sets.size()),
            sets.data(),
            0,
            nullptr);

        // Push constant: skybox scale
        float skyboxScale = 500.0f;
        vkCmdPushConstants(commandBuffer.getVkCommandBuffer(),
                           skyboxPipeline->layout,
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           sizeof(float),
                           &skyboxScale);

        skyboxMesh->draw(commandBuffer.getVkCommandBuffer());
    }

    // std::cout << "[DEBUG] DrawSceneObjects: About to bind graphics pipeline: " << graphicsPipeline << "\n";
    // std::cout << "[DEBUG] DrawSceneObjects: Main renderPass: " << renderPass << "\n";
    // std::cout << "[DEBUG] DrawSceneObjects: ImGui renderPass: " << imguiRenderPass << "\n";

    vkCmdBindPipeline(commandBuffer.getVkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer.getVkCommandBuffer(), 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer.getVkCommandBuffer(), indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // Bind both descriptor sets: set 0 (scene) and set 1 (water - needed for caustic texture in shader)
    // Even when not underwater, we need to bind set 1 because the shader declares it
    std::array<VkDescriptorSet, 2> sceneSets = {descriptorSets[imageIndex], waterDescriptorSet};
    vkCmdBindDescriptorSets(
        commandBuffer.getVkCommandBuffer(),
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0, 2,
        sceneSets.data(),
        0, nullptr);

    // std::cout << "[DEBUG] DrawSceneObjects: About to call vkCmdDrawIndexed with " << indices.size() << " indices\n";
    vkCmdDrawIndexed(commandBuffer.getVkCommandBuffer(), static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
}
void VulkanBase::printMatrix(const glm::mat4 &mat, const std::string &name)
{
    std::cout << name << ":\n";
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << mat[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void VulkanBase::processInput(float deltaTime)
{
    if (lmbPressed)
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_W, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_S, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_A, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_D, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_Q, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            camera.processKeyboard(GLFW_KEY_E, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rKeyPressed)
    {
        rotationEnabled = !rotationEnabled;
        rKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
    {
        rKeyPressed = false;
    }

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !screenshotRequested)
    {
        screenshotRequested = true;
        captureScreenshot = true;
    }

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE && screenshotRequested)
    {
        screenshotRequested = false; // Reset the request flag
    }
}

// ============================================================================
// WATER TESTING SYSTEM IMPLEMENTATION
// ============================================================================

void VulkanBase::initializeWaterTestingSystem()
{
    waterTestingSystem = std::make_unique<WaterTestingSystem>();

    VkUtils::QueueFamilyIndices indices = VkUtils::FindQueueFamilies(physicalDevice, surface);
    waterTestingSystem->initialize(device, physicalDevice, graphicsQueue, indices.graphicsFamily.value());

    // Set default camera path
    waterTestingSystem->setCameraPath(DeterministicCameraPath::createUnderwaterPath());

    // Create output directory
    std::filesystem::create_directories("test_results");

    lastFrameTime = std::chrono::high_resolution_clock::now();

    std::cout << "[VulkanBase] Water testing system initialized\n";
}

void VulkanBase::cleanupWaterTestingSystem()
{
    if (waterTestingSystem)
    {
        // Export any remaining results
        if (!completedTestResults.empty() && autoExportResults)
        {
            exportTestResults();
        }
        waterTestingSystem->cleanup();
        waterTestingSystem.reset();
    }
}

void VulkanBase::startWaterTest(const WaterTestConfig &config)
{
    if (!waterTestingSystem || isTestModeActive)
        return;

    isTestModeActive = true;
    currentTestRunIndex = 0;

    // Reset frame timing for accurate measurement from the start
    lastFrameTime = std::chrono::high_resolution_clock::now();
    lastCpuTimeMs = 0.0;

    // Apply test configuration to rendering
    applyTestConfiguration(config);

    // Start the test run
    waterTestingSystem->startTestRun(config, currentTestRunIndex);

    std::cout << "[VulkanBase] Started water test: " << config.name << "\n";
}

void VulkanBase::preFrameWaterTestUpdate()
{
    if (!waterTestingSystem || !isTestModeActive)
        return;

    // Get camera state for deterministic path
    const auto &config = waterTestingSystem->getCurrentConfig();
    uint32_t currentFrame = waterTestingSystem->getCurrentFrameIndex();

    // Apply deterministic camera if test is running (before rendering)
    if (waterTestingSystem->isTestRunning())
    {
        CameraKeyframe keyframe = waterTestingSystem->getCameraStateForFrame(currentFrame);

        // Apply camera state (override user input during test)
        camera.position = keyframe.position;
        camera.setYaw(keyframe.yaw);
        camera.setPitch(keyframe.pitch);
    }
}

void VulkanBase::postFrameWaterTestUpdate()
{
    if (!waterTestingSystem || !isTestModeActive)
        return;

    // Frame timing (lastCpuTimeMs) is now calculated in mainLoop:
    // - Timer starts BEFORE drawFrame()
    // - Timer ends AFTER vkQueueWaitIdle()
    // This gives us the TRUE frame time including all GPU work

    // Get current state
    const auto &config = waterTestingSystem->getCurrentConfig();
    uint32_t currentFrame = waterTestingSystem->getCurrentFrameIndex();

    if (waterTestingSystem->isTestRunning())
    {
        // Record frame metrics with accurate timing
        waterTestingSystem->recordFrame(
            currentFrame,
            lastCpuTimeMs,
            camera.position,
            camera.getYaw(),
            camera.getPitch());

        // Check if test run is complete
        if (currentFrame >= config.totalFrames - 1)
        {
            // End current run
            TestRunResult result = waterTestingSystem->endTestRun();

            // Append to CSV file
            if (autoExportResults)
            {
                waterTestingSystem->appendRunToCSV(result, testOutputFilePath);
            }

            // Store result
            completedTestResults.push_back(result);

            // Check if more runs needed
            currentTestRunIndex++;
            if (currentTestRunIndex < config.repeatCount)
            {
                // Start next run
                waterTestingSystem->startTestRun(config, currentTestRunIndex);
            }
            else
            {
                // Check if more configs in queue
                currentTestConfigIndex++;
                if (currentTestConfigIndex < static_cast<int>(pendingTestConfigs.size()))
                {
                    // Start next config
                    currentTestRunIndex = 0;
                    applyTestConfiguration(pendingTestConfigs[currentTestConfigIndex]);
                    waterTestingSystem->startTestRun(pendingTestConfigs[currentTestConfigIndex], 0);
                }
                else
                {
                    // All tests complete
                    endWaterTest();
                }
            }
        }
    }
}

// Legacy function - kept for compatibility, now split into pre/post frame updates
void VulkanBase::updateWaterTest()
{
    preFrameWaterTestUpdate();
    // Note: postFrameWaterTestUpdate() is now called after drawFrame() in mainLoop()
}

void VulkanBase::endWaterTest()
{
    if (!waterTestingSystem)
        return;

    isTestModeActive = false;
    currentTestConfigIndex = 0;
    currentTestRunIndex = 0;
    pendingTestConfigs.clear();

    std::cout << "[VulkanBase] Water testing completed. Total runs: " << completedTestResults.size() << "\n";

    // Generate summary report
    if (!completedTestResults.empty())
    {
        std::vector<AggregatedRunMetrics> summaryMetrics;
        for (const auto &r : completedTestResults)
        {
            summaryMetrics.push_back(r.aggregated);
        }
        waterTestingSystem->exportSummaryToCSV(summaryMetrics, "test_results/summary.csv");
    }
}

void VulkanBase::applyTestConfiguration(const WaterTestConfig &config)
{
    // Apply turbidity
    switch (config.turbidity)
    {
    case TurbidityLevel::Low:
        underwaterFogDensity = 0.02f;
        underwaterScatteringIntensity = 0.3f;
        break;
    case TurbidityLevel::Medium:
        underwaterFogDensity = 0.01f;
        underwaterScatteringIntensity = 0.5f;
        break;
    case TurbidityLevel::High:
        underwaterFogDensity = 0.1f;
        underwaterScatteringIntensity = 0.8f;
        break;
    }

    // Apply depth configuration
    switch (config.depth)
    {
    case DepthLevel::Shallow:
        // Camera path will handle positioning
        underwaterDeepColor = glm::vec3(0.0f, 0.2f, 0.4f);
        break;
    case DepthLevel::Deep:
        underwaterDeepColor = glm::vec3(0.0f, 0.05f, 0.15f);
        break;
    }

    // Apply light motion
    switch (config.lightMotion)
    {
    case LightMotion::Static:
        rotationEnabled = false;
        break;
    case LightMotion::Moving:
        rotationEnabled = true;
        break;
    }

    // Set camera path based on config
    switch (config.depth)
    {
    case DepthLevel::Shallow:
        waterTestingSystem->setCameraPath(DeterministicCameraPath::createSurfacePath());
        break;
    case DepthLevel::Deep:
        waterTestingSystem->setCameraPath(DeterministicCameraPath::createDepthTransitionPath());
        break;
    }

    std::cout << "[VulkanBase] Applied test configuration: " << config.toString() << "\n";
}

void VulkanBase::renderTestingUI()
{
    if (!waterTestingSystem)
        return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("=== Water Performance Testing ===");
    ImGui::Spacing();

    // Test type selection
    const char *testTypes[] = {"Performance", "Image Quality", "Trade-Off Sweep", "Custom"};
    ImGui::Combo("Test Type", &selectedTestType, testTypes, IM_ARRAYSIZE(testTypes));

    ImGui::Checkbox("Auto-Export to CSV", &autoExportResults);
    ImGui::Checkbox("Capture Screenshots", &captureTestScreenshots);

    // Output file path
    static char outputPath[256] = "test_results/water_test_results.csv";
    ImGui::InputText("Output File", outputPath, sizeof(outputPath));
    testOutputFilePath = outputPath;

    ImGui::Spacing();

    // Test status
    if (isTestModeActive)
    {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "TEST RUNNING");

        float progress = waterTestingSystem->getProgress();
        ImGui::ProgressBar(progress / 100.0f, ImVec2(-1, 0),
                           (std::to_string((int)progress) + "%%").c_str());

        ImGui::Text("Config: %d/%d", currentTestConfigIndex + 1, (int)pendingTestConfigs.size());
        ImGui::Text("Run: %d/%d", currentTestRunIndex + 1, waterTestingSystem->getCurrentConfig().repeatCount);
        ImGui::Text("Frame: %u/%u", waterTestingSystem->getCurrentFrameIndex(),
                    waterTestingSystem->getTotalFrames());

        if (ImGui::Button("Stop Test"))
        {
            endWaterTest();
        }
    }
    else
    {
        // Generate and run test configurations
        if (ImGui::Button("Run Selected Test Suite"))
        {
            completedTestResults.clear();
            currentTestConfigIndex = 0;

            switch (selectedTestType)
            {
            case 0: // Performance
                pendingTestConfigs = WaterTestingSystem::generatePerformanceTestConfigs();
                break;
            case 1: // Image Quality
                pendingTestConfigs = WaterTestingSystem::generateImageQualityTestConfigs();
                break;
            case 2: // Trade-Off
                pendingTestConfigs = WaterTestingSystem::generateTradeOffSweepConfigs();
                break;
            case 3: // Custom - single config with current settings
            {
                WaterTestConfig custom;
                custom.name = "Custom_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                custom.totalFrames = 300;
                custom.warmupFrames = 10;
                custom.repeatCount = 1;
                pendingTestConfigs = {custom};
            }
            break;
            }

            if (!pendingTestConfigs.empty())
            {
                startWaterTest(pendingTestConfigs[0]);
            }
        }

        ImGui::SameLine();

        // Quick single-run button
        if (ImGui::Button("Quick Run (1 Config)"))
        {
            completedTestResults.clear();

            WaterTestConfig quickConfig;
            quickConfig.name = "QuickTest_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count() % 10000);
            quickConfig.totalFrames = 300;
            quickConfig.warmupFrames = 10;
            quickConfig.repeatCount = 1;
            quickConfig.turbidity = TurbidityLevel::Medium;
            quickConfig.depth = DepthLevel::Shallow;
            quickConfig.lightMotion = LightMotion::Static;

            pendingTestConfigs = {quickConfig};
            currentTestConfigIndex = 0;
            startWaterTest(quickConfig);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();

    // Show last run results
    if (!completedTestResults.empty())
    {
        ImGui::Text("Last Run Results:");
        const auto &last = completedTestResults.back().aggregated;
        ImGui::Text("  Mean FPS: %.2f", last.meanFPS);
        ImGui::Text("  Median FPS: %.2f", last.medianFPS);
        ImGui::Text("  1%% Low FPS: %.2f", last.fps1Low);
        ImGui::Text("  Mean Frame Time: %.3f ms", last.meanFrameTime);
        ImGui::Text("  Std Dev: %.3f ms", last.stddevFrameTime);
        ImGui::Text("  Valid Frames: %d", last.validFrameCount);
        ImGui::Text("  Outliers: %d", last.outlierCount);

        if (ImGui::Button("Export All Results"))
        {
            exportTestResults();
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear Results"))
        {
            completedTestResults.clear();
        }
    }

    ImGui::Spacing();

    // Test configuration preview
    if (ImGui::CollapsingHeader("Test Configuration Preview"))
    {
        ImGui::Text("Pending configs: %d", (int)pendingTestConfigs.size());

        int previewCount = std::min(5, (int)pendingTestConfigs.size());
        for (int i = 0; i < previewCount; i++)
        {
            ImGui::BulletText("%s", pendingTestConfigs[i].name.c_str());
        }
        if (pendingTestConfigs.size() > 5)
        {
            ImGui::Text("... and %d more", (int)pendingTestConfigs.size() - 5);
        }
    }
}

void VulkanBase::exportTestResults()
{
    if (completedTestResults.empty() || !waterTestingSystem)
        return;

    // Create timestamp for unique filename
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time);

    std::stringstream ss;
    ss << "test_results/full_report_"
       << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";

    TestSuiteResult suite;
    suite.suiteName = "WaterTestSuite";
    suite.timestamp = now;
    suite.runs = completedTestResults;
    suite.outputDirectory = "test_results";

    waterTestingSystem->exportToCSV(suite, ss.str());

    // Also generate chart data
    std::vector<AggregatedRunMetrics> metrics;
    for (const auto &r : completedTestResults)
    {
        metrics.push_back(r.aggregated);
    }

    TestReportGenerator::generatePerformanceChartData(metrics,
                                                      "test_results/chart_data_" + std::to_string(time) + ".csv");

    std::cout << "[VulkanBase] Exported test results to: " << ss.str() << "\n";
}