#version 450

layout(std140, set = 0, binding = 0) uniform CameraUBO {
    mat4 model; // keep this to match C++ layout if present
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform Push {
    float skyboxScale;
} pushConsts;

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 vTexDir;

void main() {
    // direction for sampling the cubemap
    
    mat3 fixMat = mat3(
        1,  0,  0,
        0,  0,  1,
        0, -1,  0
    );

    vTexDir = fixMat * inPos;

    // Remove translation from view matrix (keep rotation only)
    mat4 rotView = mat4(mat3(ubo.view));

    // scale the unit cube with push constant (so you can tweak in CPU)
    vec4 worldPos = vec4(inPos * pushConsts.skyboxScale, 1.0);

    // apply rotation-only view and projection
    vec4 clip = ubo.proj * rotView * worldPos;

    // Force depth to far plane by setting z = w
    gl_Position = clip.xyww;
}
