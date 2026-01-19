#version 450

// Input from vertex shader
layout(location = 0) in vec2 vScreenUV;

layout(location = 0) out vec4 outColor;

// Scene Matrices (VP) & Light Pos
layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 model; mat4 view; mat4 proj;
    vec4 lightPos; vec4 viewPos;
} ubo;

// Textures
layout(set = 1, binding = 1) uniform sampler2D waterNormalMap;
layout(set = 1, binding = 2) uniform sampler2D waterDudvMap;
layout(set = 1, binding = 3) uniform sampler2D causticTex; // Used for ray noise
layout(set = 1, binding = 0) uniform sampler2D refractionTex; // The Scene Background

// Push Constants - using existing fields efficiently
// debugRays: 0=off, 1=rays, 2=snow, 3=both, 4+=chromatic debug
// scatteringIntensity: repurposed lower bits for marine snow intensity
// ambient: repurposed for chromatic aberration strength
// shininess: repurposed for marine snow size
layout(push_constant) uniform WaterPush {
    float time; float scale; float debugRays; float renderingMode; // 0=BL, 1=PB, 2=OPT
    vec4 baseColor; vec4 lightColor; 
    float ambient; float shininess; float causticIntensity; 
    float distortionStrength; float godRayIntensity; float scatteringIntensity; 
    float opacity; float fogDensity;
    float godExposure; float godDecay; float godDensity; float godSampleScale;
} pc;

// Extract packed values from push constants
float getMarineSnowIntensity() { return pc.scatteringIntensity; }
float getMarineSnowSize() { return max(0.5, pc.shininess * 0.01); } // shininess/100 for reasonable range
float getChromaticStrength() { return pc.ambient; }  // ambient repurposed for CA
bool isSnowDebugOn() { return pc.debugRays >= 2.0 && pc.debugRays < 4.0; }
bool isChromaticDebugOn() { return pc.debugRays >= 4.0; }

// ============================================================================
// MARINE SNOW - Suspended particulates for scale reference
// Without particles, viewer can't tell if scene is bathtub or ocean
// ============================================================================
float quickHash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Multi-layer marine snow for depth and variety
float marineSnowLayered(vec2 uv, float time, float size, float speed) {
    float particles = 0.0;
    
    // Layer 1: Fine dust (tiny specks - most numerous)
    vec2 p1 = uv * (800.0 / size) + vec2(time * 0.4 * speed, -time * 1.0 * speed);
    particles += pow(quickHash(floor(p1)), 8.0) * 0.3;  // Higher power = sparser
    
    // Layer 2: Small particles
    vec2 p2 = uv * (400.0 / size) + vec2(time * 0.2 * speed, -time * 0.6 * speed);
    particles += pow(quickHash(floor(p2)), 7.0) * 0.35;
    
    // Layer 3: Medium debris (sparse)
    vec2 p3 = uv * (150.0 / size) + vec2(time * 0.1 * speed, -time * 0.3 * speed);
    particles += pow(quickHash(floor(p3)), 6.0) * 0.25;
    
    // Occasional bright sparkles (rare light-catching motes)
    vec2 sparkleUV = uv * (600.0 / size) + vec2(time * 0.15 * speed, -time * 0.5 * speed);
    float sparkle = pow(quickHash(floor(sparkleUV) + floor(time * 2.0)), 12.0);
    particles += sparkle * 0.5;
    
    return particles;
}

// ============================================================================
// OPTIMIZED INTERFERENCE - Only 2 texture samples instead of 3
// ============================================================================
float interferencePatternFast(vec2 uv, float phase, float frequency) {
    // Wave 1: Moving outward
    vec2 uv1 = uv * frequency;
    vec2 offset1 = vec2(sin(phase * 0.7), cos(phase * 0.5)) * 0.1;
    float wave1 = texture(causticTex, uv1 + offset1).g;
    
    // Wave 2: Moving inward (opposing direction creates interference)
    vec2 uv2 = uv * frequency * 1.3;
    vec2 offset2 = vec2(-sin(phase * 0.6), -cos(phase * 0.8)) * 0.12;
    float wave2 = texture(causticTex, uv2 + offset2).r;
    
    // Interference: multiply for standing wave, add for brightness
    return wave1 * wave2 * 1.5 + (wave1 + wave2) * 0.25;
}

void main() {
    // Only add rays when camera is underwater
    const float u_waterHeight_const = 0.0;
    float underMask = (ubo.viewPos.y < u_waterHeight_const) ? 1.0 : 0.0;
    
    // Early out if above water
    if (underMask < 0.5) {
        outColor = vec4(0.0);
        return;
    }

    // Performance optimizations based on rendering mode
    int renderMode = int(pc.renderingMode);
    int samples;
    float qualityScale;
    
    // FIXED: Reasonable sample counts for good FPS
    if (renderMode == 0) {
        samples = 24;        // Baseline: fast
        qualityScale = 0.6;
    } else if (renderMode == 1) {
        samples = 48;        // PB: quality (was 96!)
        qualityScale = 1.0;
    } else {
        samples = 32;        // Optimized: balanced (was 64)
        qualityScale = 0.85;
    }

    // God Ray Calculation
    vec3 SUN_POS = ubo.lightPos.xyz;
    if (length(SUN_POS) < 0.1) SUN_POS = vec3(0.0, 100.0, -10.0);

    // Project sun to screen space
    vec4 sunClip = ubo.proj * ubo.view * vec4(SUN_POS, 1.0);
    vec2 sunScreen = sunClip.xy / sunClip.w * 0.5 + 0.5;
    
    // Check if sun is roughly in front of camera (FIXED: allow negative W for underwater sun above)
    float sunFacing = 1.0; // Always try to render rays underwater
    
    // Additional check: is sun above water surface? (it should be)
    float sunAboveWater = (SUN_POS.y > u_waterHeight_const) ? 1.0 : 0.5;

    vec2 toSun = sunScreen - vScreenUV;
    float distToSun = length(toSun);
    
    // Ray Marching Settings
    float density = max(0.5, pc.godDensity * qualityScale);
    float decay = clamp(pc.godDecay, 0.85, 0.99); 
    float exposure = max(0.2, pc.godExposure * qualityScale);

    vec2 rayStep = toSun / float(samples) * density;
    
    // Separate phase (time) from frequency (scale) - FIX #4
    float phase = pc.time * 1.5;
    float frequency = max(0.5, pc.scale) * 2.0;
    
    vec3 accum = vec3(0.0);
    float illum = 1.0;
    vec2 sampleUV = vScreenUV;

    float stepScale = pc.godSampleScale;

    // OPTIMIZED LOOP - single interference call per sample
    for (int i = 0; i < samples; ++i) {
        sampleUV += rayStep * stepScale;
        vec2 clampedUV = clamp(sampleUV, vec2(0.001), vec2(0.999));
        
        // Single interference pattern call (2 texture samples)
        float noise = interferencePatternFast(clampedUV, phase, frequency);
        
        // Accumulate ray intensity
        accum += vec3(noise * illum);
        illum *= decay; 
    }
    
    vec3 godRays = (accum / float(samples)) * exposure * pc.godRayIntensity * sunFacing * sunAboveWater;
    
    // Attenuate rays by fog (but not too aggressively)
    float raysFog = exp(-pc.fogDensity * distToSun);
    godRays *= raysFog;
    
    // =========================================================================
    // CHROMATIC ABERRATION - Water acts as a giant lens
    // Different wavelengths bend by different amounts, creating color fringing
    // Makes it look like a physical camera, not a math equation
    // =========================================================================
    float caStrength = getChromaticStrength();
    float chromaticOffset = distToSun * caStrength;
    
    // Debug mode: exaggerate chromatic aberration
    if (isChromaticDebugOn()) {
        chromaticOffset *= 5.0;
    }
    
    // Red bends least (longer wavelength), blue bends most (shorter wavelength)
    godRays.r *= (1.0 + chromaticOffset * 0.4);   // Warm/orange outer fringe
    godRays.g *= (1.0 + chromaticOffset * 0.15);  // Slight green shift
    godRays.b *= (1.0 + chromaticOffset * 0.6);   // Cyan/blue inner fringe
    
    // Color tinting - underwater blue
    godRays *= vec3(0.6, 0.85, 1.0);

    // =========================================================================
    // MARINE SNOW - Suspended particles for scale reference
    // Without dust specks, viewer can't gauge if it's a bathtub or ocean
    // Particles give the eye a reference point for scale and movement speed
    // =========================================================================
    float snowIntensity = getMarineSnowIntensity();
    float snowSize = getMarineSnowSize();
    
    vec3 snowColor = vec3(0.0);
    if (snowIntensity > 0.01 && renderMode >= 1) {
        // Layered marine snow with controllable size
        float snow = marineSnowLayered(vScreenUV, pc.time, snowSize, 1.0);
        
        // Particles are illuminated by god rays (catch the light)
        float rayIllumination = length(godRays) * 2.0 + 0.1;
        snow *= rayIllumination * snowIntensity * qualityScale;
        
        // Depth-based density (more particles visible deeper)
        float depthFactor = clamp(-ubo.viewPos.y * 0.05, 0.0, 1.0);
        snow *= (0.4 + depthFactor * 0.6);
        
        // Particles have slight blue-white tint (scatter blue light)
        snowColor = vec3(0.75, 0.88, 1.0) * snow;
        
        // Debug: show particles in bright green
        if (isSnowDebugOn()) {
            snowColor = vec3(0.0, 1.0, 0.3) * snow * 3.0;
        }
    }

    // Combine
    vec3 finalRays = godRays + snowColor;
    
    // DEBUG MODE: Show purple outline where rays exist
    if (pc.debugRays > 0.5 && pc.debugRays < 2.0) {
        float rayPresence = length(godRays);
        // Purple outline for ray presence
        vec3 debugColor = vec3(0.8, 0.2, 1.0) * rayPresence * 2.0;
        // Yellow for high intensity areas
        debugColor += vec3(1.0, 1.0, 0.0) * max(0.0, rayPresence - 0.3);
        finalRays = mix(finalRays, debugColor, 0.7);
        
        // Show sun position marker
        if (distToSun < 0.05) {
            finalRays = vec3(1.0, 0.0, 1.0); // Magenta marker at sun
        }
    }
    
    outColor = vec4(finalRays, 1.0);
}