#version 450

// We output UV coordinates to the fragment shader
layout(location = 0) out vec2 vScreenUV;

void main() {
    // Generate a full-screen triangle using the vertex index (0, 1, 2)
    // This covers the screen range [-1, 1] in X and Y
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    
    vScreenUV = uv;
    
    // Output position in Clip Space
    gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
}