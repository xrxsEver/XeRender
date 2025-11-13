#include "Camera.h"

const float YAW = -90.0f;  // Look along the negative Z axis
const float PITCH = 0.0f;  // No initial pitch
const float SPEED = 0.0005f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : front(glm::vec3(0.0f, 0.0f, -1.0f)), movementSpeed(SPEED), mouseSensitivity(SENSITIVITY * 0.5f), zoom(ZOOM)
{
    this->position = position;
    this->worldUp = up;
    this->yaw = yaw;
    this->pitch = pitch;
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

void Camera::processKeyboard(int direction, float deltaTime) {
    float velocity = movementSpeed * deltaTime;
    if (direction == GLFW_KEY_W)
        position += front * velocity;
    if (direction == GLFW_KEY_S)
        position -= front * velocity;
    if (direction == GLFW_KEY_A)
        position -= right * velocity;
    if (direction == GLFW_KEY_D)
        position += right * velocity;
    if (direction == GLFW_KEY_Q)  // Move up
        position += up * velocity;
    if (direction == GLFW_KEY_E)  // Move down
        position -= up * velocity;
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (constrainPitch) {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }

    updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset) {
    float scrollSensitivity = 0.05f;
    float maxSpeed = 0.4f;  
    float minSpeed = 0.004f;

    if (yoffset > 0) {
        movementSpeed *= (1.0f + scrollSensitivity);
    }
    else if (yoffset < 0) {
        movementSpeed *= (1.0f - scrollSensitivity);
    }

    if (movementSpeed > maxSpeed) {
        movementSpeed = maxSpeed;
    }
    else if (movementSpeed < minSpeed) {
        movementSpeed = minSpeed;
    }
}

void Camera::updateCameraVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    this->front = glm::normalize(front);
    this->right = glm::normalize(glm::cross(this->front, this->worldUp));
    this->up = glm::normalize(glm::cross(this->right, this->front));
}
