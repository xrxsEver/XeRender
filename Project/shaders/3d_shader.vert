#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// ADDED: Push Constants to match the Pipeline Layout (80 bytes)
// This must be present because we set VK_SHADER_STAGE_VERTEX_BIT in C++
layout(push_constant) uniform WaterPush {
    float time;
    float scale;
    vec2 _pad;
    vec4 baseColor;
    vec4 lightColor;
    float ambient;
    float shininess;
    float causticIntensity;
    float distortionStrength;
    float godRayIntensity;
    float scatteringIntensity;
    float opacity;
    float fogDensity;
} pc;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPosition;

void main() {
    fragNormal = mat3(ubo.model) * inNormal; // Transform the normal to world space
    fragTexCoord = inTexCoord;
    fragPosition = vec3(ubo.model * vec4(inPosition, 1.0)); // World space position

    gl_Position = ubo.proj * ubo.view * vec4(fragPosition, 1.0);
}