# Set source files
set(COMMAND_SOURCES
    CommandBuffer.cpp
    CommandPool.cpp
)

# Set header files
set(COMMAND_HEADERS
    CommandBuffer.h
    CommandPool.h
)

# Add library target
add_library(CommandLib ${COMMAND_SOURCES} ${COMMAND_HEADERS})

# Include directories
target_include_directories(CommandLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../include ${glfw_INCLUDE_DIRS})
target_link_libraries(CommandLib PRIVATE glfw)
