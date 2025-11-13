cmake_minimum_required(VERSION 3.10)

# Project Name
project(VulkanProject)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include FetchContent module
include(FetchContent)

# Fetch GLFW
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.8 # Specify the version you need
)

FetchContent_MakeAvailable(glfw)

# Fetch ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.89.2 # Specify the version you need or the latest stable tag
)

FetchContent_MakeAvailable(imgui)

# Find the required packages
find_package(Vulkan REQUIRED)

# Include Directories
include_directories(${Vulkan_INCLUDE_DIRS} include ${glfw_INCLUDE_DIRS} Lib ${imgui_SOURCE_DIR})

# Add subdirectories
add_subdirectory(shaders)
add_subdirectory(Command)

# Set source files
set(SOURCES
    main.cpp
    VulkanBase.cpp
    include/VulkanUtil.cpp
    xrxsPipeline.cpp 
    DAEMesh.cpp
    DAEDataBuffer.cpp
    SwapChainManager.cpp
    Shader2D.cpp
    ModelLoader.cpp
    Lib/stb_image.h
    Lib/stb_image_write.h
    Lib/json.hpp  
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)

# Set header files
set(HEADERS
    include/VulkanBase.h
    include/VulkanUtil.h
    include/xrxsPipeline.h
    include/Vertex.h 
    include/DAEMesh.h
    include/DAEDescriptorPool.h
    include/DAEUniformBufferObject.h 
    include/DAEDataBuffer.h
    include/DAEDescriptorPool.inl 
    include/SwapChainManager.h
    include/Shader2D.h
    include/ModelLoader.h
)

# Create ImGui as a static library
add_library(imgui STATIC 
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)

# Ensure that ImGui can find GLFW headers
target_include_directories(imgui PRIVATE ${glfw_SOURCE_DIR}/include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} "include/Camera.h" "include/Camera.cpp" "include/Shader3D.h" "Shader3D.cpp")

# Link libraries
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} include ${glfw_INCLUDE_DIRS} Lib ${imgui_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Vulkan_LIBRARIES} glfw CommandLib imgui)

# Ensure shaders are built before the main project
add_dependencies(${PROJECT_NAME} shaders)
