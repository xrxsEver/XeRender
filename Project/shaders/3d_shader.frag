#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;

layout(location = 0) out vec4 FragColor;

layout(binding = 1) uniform sampler2D baseTexture;
layout(binding = 3) uniform sampler2D metalnessTexture;
layout(binding = 4) uniform sampler2D normalTexture;
layout(binding = 5) uniform sampler2D specularTexture;
layout(set = 1, binding = 3) uniform sampler2D causticTex;

#define MAX_LIGHTS 2
struct Light { vec3 position; vec3 color; float intensity; };
layout(binding = 2) uniform LightInfo { Light lights[MAX_LIGHTS]; vec3 viewPos; vec3 ambientColor; float ambientIntensity; } lightInfo;
layout(binding = 6) uniform ToggleInfo { bool applyNormalMap; bool applyMetalnessMap; bool applySpecularMap; bool viewNormalOnly; bool viewMetalnessOnly; bool viewSpecularOnly; bool applyRimLight; } toggleInfo;

layout(push_constant) uniform WaterPush {
    float time; float scale; vec2 _pad;
    vec4 baseColor;  // ImGui: Shallow Color
    vec4 lightColor; // ImGui: Deep Color
    float ambient; float shininess; float causticIntensity; 
    float distortionStrength; float godRayIntensity; float scatteringIntensity; 
    float opacity; float fogDensity;
} pc;

void main() {
    vec3 baseColor = texture(baseTexture, fragTexCoord).rgb;
    vec3 norm = normalize(fragNormal);
    if (toggleInfo.applyNormalMap) {
        vec3 normalMapColor = texture(normalTexture, fragTexCoord).rgb;
        norm = normalize(normalMapColor * 2.0 - 1.0);
    }

    // Colors
    vec3 deepColor = pc.lightColor.rgb; // Dark Blue
    
    // Lighting
    // If light position is 0, default to overhead
    vec3 lightPos = lightInfo.lights[0].position;
    if (length(lightPos) < 0.1) lightPos = vec3(0.0, 100.0, 0.0);
    
    vec3 sunDir = normalize(lightPos);
    float diff = max(dot(norm, sunDir), 0.0);
    
    // Ambient - Make it brighter and blue-ish
    vec3 ambient = vec3(0.2, 0.4, 0.5) * 0.5; 
    
    // Caustics
    vec2 causticUV = fragPosition.xz * 0.05 + vec2(pc.time * 0.05);
    float caustic = texture(causticTex, causticUV).r;
    caustic += texture(causticTex, causticUV * 0.7 - vec2(pc.time * 0.02)).r;
    caustic = pow(caustic, 3.0) * 3.0; // Sharpen
    
    vec3 causticColor = vec3(0.8, 0.9, 1.0) * caustic * pc.causticIntensity;

    // Combine
    vec3 finalColor = baseColor * (ambient + diff) + (causticColor * baseColor);

    // === SEAM FIX: DISTANCE FOG ===
    // This must match the surface shader's Deep Color blend
    float dist = length(fragPosition - lightInfo.viewPos);
    float fogFactor = 1.0 - exp(-dist * pc.fogDensity);
    fogFactor = clamp(fogFactor, 0.0, 1.0);

    // Fade floor into the deep water color
    finalColor = mix(finalColor, deepColor, fogFactor);

    FragColor = vec4(finalColor, 1.0);
}