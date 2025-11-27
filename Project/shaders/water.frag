#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec2 vScreenUV;

layout(location = 0) out vec4 outColor;

// GLOBAL UBO (set=0 binding=0) - same as vert
layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// WATER-SPECIFIC TEXTURES in set 1
layout(set = 1, binding = 0) uniform sampler2D refractionTex;   // Refraction (what's under)
layout(set = 1, binding = 1) uniform sampler2D waterNormalMap; // normal map (tiled)
layout(set = 1, binding = 2) uniform sampler2D waterDudvMap;   // dudv/distortion map (animated)
layout(set = 1, binding = 3) uniform sampler2D causticTex;     // optional (caustic map)
layout(set = 1, binding = 4) uniform sampler2D reflectionTex;   // Reflection (what's above)

// push constant layout must match pipeline (64 bytes total)
layout(push_constant) uniform WaterPush {
    float time;
    float scale;
    vec4 baseColor;     // RGB color + alpha padding
    vec4 lightColor;    // RGB color + alpha padding
    float ambient;
    float shininess;
    float causticIntensity;
    float distortionStrength;
} pc;

// Water properties for photorealistic rendering
const float R_0 = 0.02; // Schlick's approximation base reflectivity for water
const vec3 DEEP_WATER_COLOR = vec3(0.05, 0.2, 0.4); // Deep, darker blue
const vec3 SHALLOW_WATER_COLOR = vec3(0.2, 0.5, 0.8); // Lighter blue-green

void main() {
    // Get water color from push constant or use realistic defaults
    vec3 waterBaseColor = pc.baseColor.rgb;
    if (length(waterBaseColor - vec3(1.0)) < 0.1) {
        waterBaseColor = SHALLOW_WATER_COLOR; // Realistic shallow water
    }
    
    vec3 waterLightColor = pc.lightColor.rgb;
    if (length(waterLightColor - vec3(1.0)) < 0.1) {
        waterLightColor = vec3(0.9, 0.95, 1.0); // Nearly white light
    }
    
    float waterAmbient = pc.ambient;
    float waterShininess = pc.shininess;
    float causticIntensity = pc.causticIntensity;
    float distortionStrength = pc.distortionStrength;

    // View direction
    vec3 N = normalize(vNormal); // Base normal (y-up)
    vec3 V = normalize(ubo.view[3].xyz - vWorldPos); // Vector from fragment to camera
    float VdotN = max(0.01, dot(V, N)); // Clamp to avoid division by zero

    // === DYNAMIC WAVE NORMAL PERTURBATION ===
    // Create animated normal map sampling for realistic wave motion
    vec2 mapUV = vUV * 3.5; // Scale of normal map tiling
    
    // Two moving waves for complex wave pattern
    vec2 wave1_uv = mapUV + vec2(pc.time * 0.08, pc.time * 0.06);
    vec2 wave2_uv = mapUV + vec2(-pc.time * 0.05, pc.time * 0.09) * 0.7;
    
    // Sample and blend two normal maps for more complex waves
    vec3 normal1 = normalize(texture(waterNormalMap, wave1_uv).rgb * 2.0 - 1.0);
    vec3 normal2 = normalize(texture(waterNormalMap, wave2_uv * 0.8).rgb * 2.0 - 1.0);
    vec3 perturbedNormal = normalize(normal1 + normal2) * 0.5;
    
    // Apply distortion strength and blend with base normal
    N = normalize(mix(N, perturbedNormal.xzy, distortionStrength * 0.8));

    // === DIRECTIONAL LIGHTING ===
    vec3 L = normalize(vec3(0.7, 1.0, -0.3)); // Realistic sun direction
    float LdotN = max(0.0, dot(L, N));
    float LdotNsharp = pow(LdotN, 1.2); // Sharper transition for water
    
    // === FRESNEL REFLECTIVITY (Schlick's approximation) ===
    float fresnel = R_0 + (1.0 - R_0) * pow(clamp(1.0 - VdotN, 0.0, 1.0), 5.0);
    
    // === SPECULAR HIGHLIGHTS ===
    // Create two specular highlights for more realistic look
    vec3 specReflect = reflect(-L, N);
    float spec1 = pow(max(0.0, dot(specReflect, V)), waterShininess) * 0.9;
    
    // Secondary, softer highlight for subsurface scattering effect
    vec3 specReflect2 = reflect(-L, normalize(mix(N, perturbedNormal.xzy, 0.3)));
    float spec2 = pow(max(0.0, dot(specReflect2, V)), waterShininess * 0.4) * 0.4;
    
    vec3 specular = waterLightColor * (spec1 + spec2);

    // === DEPTH-BASED COLOR (water absorption effect) ===
    // Simulate water getting darker with depth
    float depthFactor = smoothstep(0.0, 50.0, length(vWorldPos)); // Higher = deeper
    vec3 depthAdjustedColor = mix(SHALLOW_WATER_COLOR, DEEP_WATER_COLOR, depthFactor * 0.6);
    
    // === BASE LIGHTING ===
    vec3 ambient = waterBaseColor * waterAmbient * 0.3; // Subtle ambient
    vec3 diffuse = depthAdjustedColor * LdotNsharp * 0.6; // Use depth-adjusted color for diffuse
    
    // === COMBINE LIGHTING ===
    vec3 color = ambient + diffuse + specular;
    
    // === DYNAMIC WAVE ANIMATION ===
    // Add subtle color variation based on distance to simulate caustics-like effect
    float causticWave = sin(pc.time * 0.5 + vWorldPos.x * 0.05) * 0.1 + 
                        cos(pc.time * 0.3 + vWorldPos.z * 0.07) * 0.08;
    color += vec3(0.05, 0.08, 0.1) * causticWave;
    
    // === FINAL COLOR COMPOSITION ===
    // Start with computed lighting, then blend in base water color for richness
    color = mix(color, waterBaseColor, 0.5);
    
    // Apply Fresnel-based transparency: more transparent at grazing angles
    float finalAlpha = mix(0.7, 0.95, fresnel);
    
    outColor = vec4(color, finalAlpha);
}