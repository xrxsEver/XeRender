#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

class Camera
{
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);

    glm::mat4 getViewMatrix() const;
    void processKeyboard(int direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    void processMouseScroll(float yoffset);

    glm::vec3 getPosition() const { return position; }

    // Getters for yaw/pitch (for testing system)
    float getYaw() const { return yaw; }
    float getPitch() const { return pitch; }

    // Setters for yaw/pitch (for deterministic camera paths in testing)
    void setYaw(float newYaw)
    {
        yaw = newYaw;
        updateCameraVectors();
    }
    void setPitch(float newPitch)
    {
        pitch = newPitch;
        updateCameraVectors();
    }

    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float yaw;
    float pitch;

    float movementSpeed;
    float mouseSensitivity;
    float zoom;

private:
    void updateCameraVectors();
};
