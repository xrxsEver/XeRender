#version 450

// Underwater volumetric pass is a fullscreen triangle.
layout(location = 0) out vec2 vScreenUV;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vScreenUV = uv;
    gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
}