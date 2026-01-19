#version 450

// Vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

// Global UBO (set 0, binding 0) - match your C++ UBO layout
layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// push constant: time, scale, colors, and lighting parameters (64 bytes total with std140 layout)
layout(push_constant) uniform WaterPush {
    float time;
    float scale;
    float debugRays;
    float _pad1;
    vec4 baseColor;     // RGB color + alpha padding
    vec4 lightColor;    // RGB color + alpha padding
    float ambient;
    float shininess;
    float causticIntensity;
    float distortionStrength;
    float godRayIntensity;
    float scatteringIntensity;
    float opacity;
    float fogDensity;
    float godExposure;
    float godDecay;
    float godDensity;
    float godSampleScale;
} pc;

// Outputs to fragment
layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec2 vScreenUV; // for sampling scene texture

// Gerstner wave function for realistic water displacement
vec3 GerstnerWave(vec4 wave, vec3 p) {
    float steepness = wave.z;
    float wavelength = wave.w;
    float k = 2.0 * 3.14159 / wavelength;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(wave.xy);
    float f = k * (dot(d, p.xz) - c * pc.time);
    float a = steepness / k;

    return vec3(
        d.x * (a * cos(f)),
        a * sin(f),
        d.y * (a * cos(f))
    );
}

void main() {
    // === REALISTIC GERSTNER WAVES ===
    // Multiple waves with different frequencies for complex water surface
    vec3 gridPoint = inPosition;
    vec3 displacedPos = gridPoint;
    vec3 normal = inNormal;

    // Wave 1: Long wavelength (swell-like)
    vec4 wave1 = vec4(1.0, 0.0, 0.25, 60.0);
    displacedPos += GerstnerWave(wave1, gridPoint);
    
    // Wave 2: Medium wavelength
    vec4 wave2 = vec4(0.2, 0.4, 0.15, 31.0);
    displacedPos += GerstnerWave(wave2, gridPoint);
    
    // Wave 3: Short wavelength (ripples)
    vec4 wave3 = vec4(-0.3, 0.25, 0.1, 18.0);
    displacedPos += GerstnerWave(wave3, gridPoint);

    // Scale the overall displacement
    float heightScale = pc.scale * 0.5; // Moderate wave height
    displacedPos = vec3(
        gridPoint.x + (displacedPos.x - gridPoint.x) * heightScale,
        gridPoint.y + (displacedPos.y - gridPoint.y) * heightScale,
        gridPoint.z + (displacedPos.z - gridPoint.z) * heightScale
    );
    
    // === NORMAL CALCULATION ===
    // Calculate approximate normal from wave displacement
    float epsilon = 0.1;
    vec3 pos1 = displacedPos;
    
    vec3 pos2 = gridPoint + vec3(epsilon, 0.0, 0.0);
    pos2 += GerstnerWave(wave1, pos2) * heightScale;
    pos2 += GerstnerWave(wave2, pos2) * heightScale;
    pos2 += GerstnerWave(wave3, pos2) * heightScale;
    
    vec3 pos3 = gridPoint + vec3(0.0, 0.0, epsilon);
    pos3 += GerstnerWave(wave1, pos3) * heightScale;
    pos3 += GerstnerWave(wave2, pos3) * heightScale;
    pos3 += GerstnerWave(wave3, pos3) * heightScale;
    
    vec3 tangent = normalize(pos2 - pos1);
    vec3 bitangent = normalize(pos3 - pos1);
    normal = normalize(cross(bitangent, tangent));
    
    vWorldPos = (ubo.model * vec4(displacedPos, 1.0)).xyz;
    vNormal = mat3(transpose(inverse(ubo.model))) * normal;
    vUV = inTexCoord;

    // Calculate clip space position
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(displacedPos, 1.0);

    // Calculate screen space UV (for sampling scene color/depth)
    // Convert NDC [-1, 1] to UV [0, 1]
    vScreenUV = gl_Position.xy / gl_Position.w * 0.5 + 0.5;
}