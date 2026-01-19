#version 450

layout(location = 0) in vec2 vScreenUV;

layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 model; mat4 view; mat4 proj; vec4 lightPos; vec4 viewPos;
} ubo;

// Underwater volumetric pass: fog/scattering only (fullscreen)
layout(set = 1, binding = 3) uniform sampler2D causticTex;

layout(push_constant) uniform WaterPush {
    float time; 
    float scale; 
    float debugRays; 
    float renderingMode; // 0=BL, 1=PB, 2=OPT
    vec4 baseColor; vec4 lightColor; 
    float ambient; float shininess; float causticIntensity; 
    float distortionStrength; float godRayIntensity; float scatteringIntensity; 
    float opacity; float fogDensity; 
    float godExposure; float godDecay; float godDensity; float godSampleScale; 
} pc;

// Extract packed values from push constants (same encoding as sunrays.frag)
float getMarineSnowIntensity() { return pc.scatteringIntensity; }
float getMarineSnowSize() { return max(0.5, pc.shininess * 0.01); }
bool isSnowDebugOn() { return pc.debugRays >= 2.0 && pc.debugRays < 4.0; }

// ============================================================================
// MARINE SNOW - Multi-layer particles for scale reference
// ============================================================================
float quickHash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float marineSnowLayered(vec2 uv, float time, float size) {
    float particles = 0.0;
    
    // Fine dust (tiny specks)
    vec2 p1 = uv * (600.0 / size) + vec2(time * 0.3, -time * 0.8);
    particles += pow(quickHash(floor(p1)), 8.0) * 0.3;
    
    // Small particles
    vec2 p2 = uv * (300.0 / size) + vec2(time * 0.15, -time * 0.5);
    particles += pow(quickHash(floor(p2)), 7.0) * 0.35;
    
    // Medium debris (sparse)
    vec2 p3 = uv * (120.0 / size) + vec2(time * 0.08, -time * 0.25);
    particles += pow(quickHash(floor(p3)), 6.0) * 0.25;
    
    return particles;
}

// Optimized interference for scattering (2 samples only)
float scatteringInterferenceFast(vec2 uv, float phase, float frequency) {
    vec2 uv1 = uv * frequency;
    vec2 offset1 = vec2(sin(phase * 0.5), cos(phase * 0.7)) * 0.12;
    float wave1 = texture(causticTex, uv1 + offset1).r;
    
    vec2 uv2 = uv * frequency * 1.2;
    vec2 offset2 = vec2(-sin(phase * 0.6), -cos(phase * 0.4)) * 0.1;
    float wave2 = texture(causticTex, uv2 + offset2).g;
    
    return wave1 * wave2 * 1.4 + (wave1 + wave2) * 0.2;
}

void main() {
    const float waterHeight = 0.0; // TODO: wire from CPU
    bool cameraUnderwater = (ubo.viewPos.y < waterHeight);
    if (!cameraUnderwater) {
        outColor = vec4(0.0);
        return;
    }

    // Performance optimizations based on rendering mode
    int renderMode = int(pc.renderingMode);
    float qualityScale = 1.0;
    bool useAdvancedScattering = true;
    bool useWavelengthDependent = true;
    
    if (renderMode == 0) {
        // BL (Baseline): Simple RGB attenuation + fog
        qualityScale = 0.5;
        useAdvancedScattering = false;
        useWavelengthDependent = false;
    } else if (renderMode == 1) {
        // PB (Physically-Based): Full wavelength-dependent effects
        qualityScale = 1.2;
        useAdvancedScattering = true;
        useWavelengthDependent = true;
    } else {
        // OPT (Optimized): Balanced quality with optimizations
        qualityScale = 1.0;
        useAdvancedScattering = true;
        useWavelengthDependent = true;
    }

    // Compute world ray direction for this pixel to create a "cut" at the horizon.
    vec2 ndc = vScreenUV * 2.0 - 1.0;
    vec4 viewRay = inverse(ubo.proj) * vec4(ndc, 1.0, 1.0);
    vec3 viewDirVS = normalize(viewRay.xyz / max(viewRay.w, 1e-6));
    vec3 rayDirWS = normalize((inverse(ubo.view) * vec4(viewDirVS, 0.0)).xyz);

    // Hard cut above the water surface: this pass must NOT cover the surface/sky.
    // (The water surface shader handles the "looking up" case.)
    float horizonMask = smoothstep(0.00, -0.08, rayDirWS.y);

    // Depth below water surface at camera; cheap approximation for fog thickness.
    float depthBelow = max(0.0, waterHeight - ubo.viewPos.y);
    float fogDist = depthBelow / max(0.18, -rayDirWS.y);

    vec3 shallowColor = pc.baseColor.rgb;
    vec3 deepColor = pc.lightColor.rgb;
    vec3 fogColor = mix(deepColor, shallowColor, 0.35);

    // Different absorption models based on rendering mode
    vec3 transmittance;
    if (useWavelengthDependent) {
        // Beer–Lambert wavelength absorption: R absorbed fastest, G medium, B slowest
        vec3 sigma = vec3(0.18, 0.07, 0.03) * max(0.0, pc.fogDensity) * qualityScale;
        transmittance = exp(-sigma * fogDist);
    } else {
        // Simple uniform absorption for baseline
        float sigma = 0.08 * max(0.0, pc.fogDensity);
        transmittance = vec3(exp(-sigma * fogDist));
    }

    // Fog amount used for alpha blending; keep subtle and let surface remain visible.
    float fogAmount = (1.0 - max(transmittance.r, max(transmittance.g, transmittance.b))) * horizonMask;

    // Vertical light falloff (darker with depth)
    float depthFalloff = exp(-0.06 * depthBelow * qualityScale);

    // Scattering calculation based on rendering mode
    vec3 scatterCol = vec3(0.0);
    
    // FIX #4: Separate phase (time) from frequency (scale)
    float phase = pc.time * 1.2;       // Movement speed - independent of pattern size
    float frequency = max(0.5, pc.scale) * 1.5;  // Pattern density - independent of speed
    
    if (useAdvancedScattering) {
        // Advanced forward scattering term modulated by caustic noise
        vec3 SUN_POS = ubo.lightPos.xyz;
        if (length(SUN_POS) < 0.1) SUN_POS = vec3(0.0, 100.0, -10.0);
        vec3 SUN_DIR = normalize(SUN_POS);

        // FIX #1: Use FAST interference pattern
        float noise = scatteringInterferenceFast(vScreenUV, phase, frequency);
        
        // Multi-scattering approximation
        float scatter = pc.scatteringIntensity * max(0.0, dot(-rayDirWS, SUN_DIR)) * (0.6 + 0.4 * noise);
        if (renderMode == 1) {
            // Rayleigh scattering for PB mode
            float rayleighPhase = 0.75 * (1.0 + dot(-rayDirWS, SUN_DIR) * dot(-rayDirWS, SUN_DIR));
            scatter += rayleighPhase * 0.3 * pc.scatteringIntensity;
        }
        
        scatterCol = fogColor * scatter * fogAmount * qualityScale;
    } else {
        // Simple scattering for baseline mode
        float scatter = pc.scatteringIntensity * 0.3;
        scatterCol = fogColor * scatter * fogAmount;
    }

    vec3 fog = fogColor * fogAmount;
    vec3 finalColor = (fog + scatterCol) * depthFalloff;
    
    // =========================================================================
    // MARINE SNOW - Suspended particles for scale reference
    // =========================================================================
    float snowIntensity = getMarineSnowIntensity();
    float snowSize = getMarineSnowSize();
    
    if (snowIntensity > 0.01 && renderMode >= 1) {
        float snow = marineSnowLayered(vScreenUV, pc.time, snowSize);
        
        // Visible in foggy areas
        float snowVisibility = fogAmount * 0.4 + 0.15;
        float snowAmount = snow * snowVisibility * qualityScale * snowIntensity * 0.2;
        
        // Add with blue-green tint
        vec3 snowColor = vec3(0.65, 0.82, 0.95) * snowAmount * depthFalloff;
        
        // Debug: bright green overlay
        if (isSnowDebugOn()) {
            snowColor = vec3(0.0, 1.0, 0.3) * snowAmount * 4.0;
        }
        
        finalColor += snowColor;
    }

    // Use user-controlled opacity to avoid full-screen solid fill
    float alpha = clamp(fogAmount, 0.0, 1.0) * clamp(pc.opacity, 0.0, 1.0);
    outColor = vec4(finalColor, alpha);
}