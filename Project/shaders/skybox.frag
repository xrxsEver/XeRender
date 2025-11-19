#version 450

layout(location = 0) in vec3 vTexDir;
layout(location = 0) out vec4 outColor;

// cubemap sampler in set 1 binding 0
layout(set = 1, binding = 0) uniform samplerCube skyboxTex;

void main() {
    outColor = texture(skyboxTex, normalize(vTexDir));
}
