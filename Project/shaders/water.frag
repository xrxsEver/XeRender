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
    // match C++ UBO (lightPos, viewPos) — use vec4 in std140 for proper alignment
    vec4 lightPos;
    vec4 viewPos;
} ubo;

// Explicit uniforms per requirements (aliased to existing UBO data where possible)
// NOTE: `cameraPosition` is sourced from `ubo.viewPos.xyz`.
//       `waterHeight` is currently a constant; wire from CPU when available.
const float u_waterHeight_const = 0.0; // TODO: replace with uniform when descriptor is wired
vec3 cameraPosition() { return ubo.viewPos.xyz; }
float waterHeight() { return u_waterHeight_const; }

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
    float debugRays;
    float renderingMode; // 0=BL, 1=PB, 2=OPT
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

// Water properties for photorealistic rendering
const float R_0 = 0.02; // Schlick's approximation base reflectivity for water
const vec3 DEEP_WATER_COLOR = vec3(0.05, 0.2, 0.4); // Deep, darker blue
const vec3 SHALLOW_WATER_COLOR = vec3(0.2, 0.5, 0.8); // Lighter blue-green


vec3 getCaustics(vec3 worldPos) {
    // Performance optimization based on rendering mode
    int renderMode = int(pc.renderingMode);
    
    // Skip caustics for baseline mode for performance
    if (renderMode == 0) {
        return vec3(0.0);
    }
    
    // 1. Project downwards (XZ plane)
    vec2 causticUV = worldPos.xz * 0.5; // Scale texture
    
    // 2. Animate with time
    float speed = pc.time * 0.2;
    vec3 c1 = texture(causticTex, causticUV + vec2(speed, speed)).rgb;
    vec3 c2 = texture(causticTex, causticUV + vec2(-speed, 0.5 * speed)).rgb;
    
    // 3. Enhanced caustics for physically-based mode
    vec3 caustics;
    if (renderMode == 1) {
        // Physically-based: Multi-layered caustics
        vec3 c3 = texture(causticTex, causticUV * 1.7 + vec2(speed * 0.7, -speed * 0.3)).rgb;
        caustics = min(c1, c2) * 1.8 + c3 * 0.5;
    } else {
        // Optimized: Balanced caustics
        caustics = min(c1, c2) * 2.0;
    }
    
    // 4. Mask by height (fade out as we go deeper or above water)
    float heightMask = clamp((waterHeight() - worldPos.y) * 0.2, 0.0, 1.0); 
    
    return caustics * pc.causticIntensity * heightMask;
}

void main() {
    // Performance optimizations based on rendering mode
    int renderMode = int(pc.renderingMode);
    float qualityScale = 1.0;
    bool useAdvancedWaves = true;
    bool usePhysicallyBasedReflection = true;
    
    if (renderMode == 0) {
        // BL (Baseline): Simplified water rendering
        qualityScale = 0.6;
        useAdvancedWaves = false;
        usePhysicallyBasedReflection = false;
    } else if (renderMode == 1) {
        // PB (Physically-Based): Enhanced quality
        qualityScale = 1.2;
        useAdvancedWaves = true;
        usePhysicallyBasedReflection = true;
    } else {
        // OPT (Optimized): Balanced quality
        qualityScale = 1.0;
        useAdvancedWaves = true;
        usePhysicallyBasedReflection = true;
    }

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
    float waterShininess = pc.shininess * qualityScale;
    float causticIntensity = pc.causticIntensity * qualityScale;
    float distortionStrength = pc.distortionStrength * (useAdvancedWaves ? 1.0 : 0.5);

    // View direction
    vec3 N = normalize(vNormal); // Base normal (y-up)
    // Use actual camera/world position from UBO for correct Fresnel
    vec3 V = normalize(ubo.viewPos.xyz - vWorldPos); // Vector from fragment to camera
    float VdotN = max(0.01, dot(V, N)); // Clamp to avoid division by zero

    // === DYNAMIC WAVE NORMAL PERTURBATION ===
    if (useAdvancedWaves) {
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
    } else {
        // Simple wave animation for baseline mode
        vec2 simpleUV = vUV * 2.0 + vec2(pc.time * 0.05, pc.time * 0.03);
        vec3 simpleNormal = normalize(texture(waterNormalMap, simpleUV).rgb * 2.0 - 1.0);
        N = normalize(mix(N, simpleNormal.xzy, distortionStrength * 0.4));
    }

    // === DIRECTIONAL LIGHTING ===
    vec3 L = normalize(vec3(0.7, 1.0, -0.3)); // Realistic sun direction
    float LdotN = max(0.0, dot(L, N));
    float LdotNsharp = pow(LdotN, useAdvancedWaves ? 1.2 : 1.0); // Sharper transition for advanced mode
    
    // === FRESNEL REFLECTIVITY (Schlick's approximation) ===
    float fresnel;
    if (usePhysicallyBasedReflection) {
        // Accurate Fresnel calculation
        fresnel = R_0 + (1.0 - R_0) * pow(clamp(1.0 - VdotN, 0.0, 1.0), 5.0);
    } else {
        // Simplified Fresnel for baseline
        fresnel = R_0 + (1.0 - R_0) * pow(clamp(1.0 - VdotN, 0.0, 1.0), 2.0);
    }
    
    // === SPECULAR HIGHLIGHTS ===
    vec3 specular = vec3(0.0);
    if (useAdvancedWaves) {
        // Create two specular highlights for more realistic look
        vec3 specReflect = reflect(-L, N);
        float spec1 = pow(max(0.0, dot(specReflect, V)), waterShininess) * 0.9;
        
        // Secondary, softer highlight for subsurface scattering effect
        vec3 specReflect2 = reflect(-L, normalize(mix(N, vec3(0,1,0), 0.3)));
        float spec2 = pow(max(0.0, dot(specReflect2, V)), waterShininess * 0.4) * 0.4;
        
        specular = waterLightColor * (spec1 + spec2);
    } else {
        // Simple specular for baseline
        vec3 specReflect = reflect(-L, N);
        float spec = pow(max(0.0, dot(specReflect, V)), waterShininess * 0.7) * 0.6;
        specular = waterLightColor * spec;
    }

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
    if (useAdvancedWaves) {
        // Add subtle color variation based on distance to simulate caustics-like effect
        float causticWave = sin(pc.time * 0.5 + vWorldPos.x * 0.05) * 0.1 + 
                            cos(pc.time * 0.3 + vWorldPos.z * 0.07) * 0.08;
        color += vec3(0.05, 0.08, 0.1) * causticWave;
    }
    
    // === REFLECTION / REFRACTION SAMPLING ===
    vec2 distortion = vec2(0.0);
    if (useAdvancedWaves) {
        // Distort screen UVs with dudv map for wavy sampling
        vec2 dudv = texture(waterDudvMap, vUV + vec2(pc.time * 0.05, pc.time * 0.03)).rg * 2.0 - 1.0;
        distortion = dudv * distortionStrength * 0.03;
    } else {
        // Simple distortion for baseline
        distortion = vec2(sin(pc.time + vUV.x * 10.0), cos(pc.time + vUV.y * 10.0)) * distortionStrength * 0.01;
    }
    
    vec2 reflUV = clamp(vScreenUV + distortion, vec2(0.0), vec2(1.0));
    vec2 refrUV = clamp(vScreenUV - distortion * 0.5, vec2(0.0), vec2(1.0));

    vec3 reflectionCol = texture(reflectionTex, reflUV).rgb;
    vec3 refractionCol = texture(refractionTex, refrUV).rgb;

    // Combine reflection/refraction using Fresnel term
    vec3 reflRefr = mix(refractionCol, reflectionCol, fresnel);

    // Combine computed lighting with environment contribution
    color = mix(color, reflRefr, usePhysicallyBasedReflection ? 0.5 * fresnel : 0.3 * fresnel);

    // === UNDERWATER VIEW ===
    // If the camera is underwater, render the underside appearance of the same surface mesh.
    float waterHeightVal = waterHeight();
    bool cameraUnder = (cameraPosition().y < waterHeightVal);
    if (cameraUnder) {
        // When looking up from underwater, we see the underside of the water surface
        vec3 V = normalize(ubo.viewPos.xyz - vWorldPos);
        
        // Calculate viewing angle - use surface normal dot view for proper underwater Fresnel
        // For underwater: looking "up" at the surface from below
        vec3 surfaceNormalDown = vec3(0.0, -1.0, 0.0); // Surface normal pointing down (into water)
        float cosTheta = max(0.0, dot(V, -surfaceNormalDown)); // How much we're looking "up"
        
        // === DISTANCE-BASED DETAIL FADE (minimal fading to preserve sky visibility) ===
        // FIX #3: Relax distance fading to prevent sky reflection from being suppressed
        float viewDistance = length(vWorldPos - ubo.viewPos.xyz);
        // Only apply very subtle fading at extreme distances
        float patternFade = 1.0 - smoothstep(200.0, 500.0, viewDistance); // Much larger range
        float detailFade = max(0.8, patternFade) * qualityScale; // Minimum 80% detail always
        
        // === PROJECTIVE TEXTURE COORDINATES (Screen Space) ===
        vec2 projTexCoord = vScreenUV;
        
        // === ANIMATED DUDV DISTORTION (always active) ===
        // FIX #1: Use world-space coordinates (vWorldPos.xz) instead of vUV to anchor waves to world
        // This prevents the 'static lens dirt' effect where ripples stick to the camera
        float timeScale = pc.time * 0.03;
        vec2 worldUV = vWorldPos.xz * 0.15; // Scale world coordinates for texture tiling
        vec2 dudvUV1 = worldUV * 4.0 + vec2(timeScale, timeScale * 0.8);
        vec2 dudvUV2 = worldUV * 6.0 + vec2(-timeScale * 1.2, timeScale * 0.6);
        vec2 dudvUV3 = worldUV * 2.5 + vec2(timeScale * 0.7, -timeScale * 0.5);
        
        vec2 dudv1 = texture(waterDudvMap, dudvUV1).rg * 2.0 - 1.0;
        vec2 dudv2 = texture(waterDudvMap, dudvUV2).rg * 2.0 - 1.0;
        vec2 dudv3 = texture(waterDudvMap, dudvUV3).rg * 2.0 - 1.0;
        
        // Combine distortion layers - make them more pronounced
        float distortionScale = pc.distortionStrength * 0.8; // Doubled from 0.04
        vec2 totalDistortion = (dudv1 + dudv2 * 0.6 + dudv3 * 0.3) * distortionScale;
        
        // Apply distortion to projective texture coordinates
        vec2 distortedReflUV = clamp(projTexCoord + totalDistortion, vec2(0.001), vec2(0.999));
        vec2 distortedRefrUV = clamp(projTexCoord - totalDistortion * 0.5, vec2(0.001), vec2(0.999));
        
        // === SAMPLE REFLECTION AND REFRACTION WITH DISTORTED COORDINATES ===
        vec3 reflectionSample = texture(reflectionTex, distortedReflUV).rgb;
        vec3 refractionSample = texture(refractionTex, distortedRefrUV).rgb;
        
        // === WATER NORMAL MAP (always sampled, complexity based on mode) ===
        // FIX #1 (continued): Use world-space coordinates for normal maps too
        vec2 baseUV = worldUV * 5.0; // Use worldUV defined above
        vec3 blendedNormal;
        
        if (useAdvancedWaves) {
            // Advanced multi-layer normal blending
            vec2 uv1 = baseUV + vec2(pc.time * 0.03, pc.time * 0.02);
            vec3 norm1 = texture(waterNormalMap, uv1).rgb * 2.0 - 1.0;
            
            vec2 uv2 = baseUV * 0.7 + vec2(-pc.time * 0.025, pc.time * 0.03);
            vec3 norm2 = texture(waterNormalMap, uv2).rgb * 2.0 - 1.0;
            
            vec2 uv3 = baseUV * 1.3 + vec2(pc.time * 0.02, -pc.time * 0.025);
            vec3 norm3 = texture(waterNormalMap, uv3).rgb * 2.0 - 1.0;
            
            blendedNormal = normalize(norm1 + norm2 * 0.7 + norm3 * 0.5);
        } else {
            vec2 uv1 = baseUV + vec2(pc.time * 0.025, pc.time * 0.02);
            blendedNormal = normalize(texture(waterNormalMap, uv1).rgb * 2.0 - 1.0);
        }
        
        // Apply normal strength - underwater surface normal points DOWN
        // Increase normal strength for more visible wave patterns
        float normalStrength = pc.distortionStrength * 40.0 * qualityScale;
        vec3 N_under = normalize(vec3(
            blendedNormal.x * normalStrength + totalDistortion.x * 12.0,
            -1.0, // Surface normal pointing down
            blendedNormal.y * normalStrength + totalDistortion.y * 12.0
        ));
        
        // === FRESNEL - SNELL'S WINDOW EFFECT ===
        // Critical angle for total internal reflection (water to air) ~48.6 degrees
        float criticalAngleCos = 0.66;
        
        // Inside Snell's window: can see above water (reflection texture shows sky/above)
        // Outside Snell's window: total internal reflection
        float insideWindow = smoothstep(criticalAngleCos - 0.15, criticalAngleCos + 0.15, cosTheta);
        
        // Fresnel for underwater surface
        float fresnelUnder = R_0 + (1.0 - R_0) * pow(1.0 - cosTheta, 4.0);
        
        // === SURFACE COLOR FROM NORMAL AND DUDV (more prominent) ===
        vec3 surfaceBaseColor = mix(SHALLOW_WATER_COLOR, DEEP_WATER_COLOR, 0.3);
        
        // Normal map creates visible wave patterns on surface - increase influence
        vec3 normalColor = blendedNormal * 0.5 + 0.5;
        surfaceBaseColor = mix(surfaceBaseColor, normalColor * vec3(0.4, 0.65, 0.85), 0.7);
        
        // DuDv adds flowing ripple color variation - increase influence
        vec3 dudvColor = vec3(
            0.3 + totalDistortion.x * 18.0,
            0.5 + totalDistortion.y * 15.0,
            0.65 + (totalDistortion.x + totalDistortion.y) * 10.0
        );
        surfaceBaseColor = mix(surfaceBaseColor, dudvColor, 0.6);
        
        // Add caustics
        vec3 caustics = getCaustics(vWorldPos) * 1.5;
        surfaceBaseColor += caustics;
        
        // === LIGHTING WITH PERTURBED NORMALS ===
        vec3 L = normalize(vec3(0.5, 1.0, -0.3)); // Light from above
        
        // FIX #2: Use wrap/translucent lighting instead of standard Lambertian to prevent dark ceiling
        // Wrap lighting simulates light scattering through the thin water surface
        float NdotL_raw = dot(-N_under, L);
        float NdotL_wrap = NdotL_raw * 0.5 + 0.5; // Wrap lighting: maps [-1,1] to [0,1]
        float NdotL_translucent = max(0.0, NdotL_raw) * 0.6 + NdotL_wrap * 0.4; // Blend standard + wrap
        vec3 diffuse = surfaceBaseColor * NdotL_translucent * 1.2; // Increased brightness
        
        // FIX #2: Add strong specular sun highlight for bright sparkle on surface
        vec3 H = normalize(L + V);
        float spec = pow(max(0.0, dot(-N_under, H)), pc.shininess * 0.5 * qualityScale);
        vec3 specular = vec3(1.0, 0.98, 0.95) * spec * 3.5; // Strong sun highlight
        
        // Secondary softer specular for additional brightness
        float spec2 = pow(max(0.0, dot(-N_under, H)), pc.shininess * 0.2 * qualityScale);
        specular += vec3(0.6, 0.75, 0.9) * spec2 * 2.0; // Increased for more brightness
        
        // Ambient - increased to prevent darkness
        vec3 ambient = surfaceBaseColor * 0.65;
        
        // === COMBINE BASE LIGHTING ===
        // This contains all surface details: normals, dudv patterns, caustics, speculars
        vec3 litColor = ambient + diffuse + specular;
        
        // === REFLECTION/REFRACTION MIXING ===
        // FIX #3: Simplified Snell's window - clear separation between sky view and underwater reflection
        vec3 finalColor;
        
        if (insideWindow > 0.5) {
            // INSIDE Snell's window: Looking up at sky - use reflection texture (sky/above water)
            // Strong sky contribution with surface lighting overlay
            finalColor = mix(reflectionSample, litColor, 0.25); // 75% sky, 25% surface detail
        } else {
            // OUTSIDE Snell's window: Total internal reflection - looking at grazing angle
            // Show underwater scene reflection (refraction texture) with surface detail
            vec3 underwaterReflection = mix(refractionSample * 0.8, litColor, 0.4);
            finalColor = underwaterReflection;
        }
        
        // Smooth transition zone around the Snell's window edge
        float transitionZone = smoothstep(0.4, 0.6, insideWindow);
        vec3 skyView = mix(reflectionSample, litColor, 0.25);
        vec3 grazingView = mix(refractionSample * 0.8, litColor, 0.4);
        finalColor = mix(grazingView, skyView, transitionZone);
        
        // === SNELL'S WINDOW BRIGHT EDGE ===
        float windowEdge = smoothstep(0.5, 0.66, cosTheta) * smoothstep(0.82, 0.66, cosTheta);
        finalColor += vec3(0.35, 0.55, 0.7) * windowEdge * 0.8;
        
        // === DISTANCE FOG (minimal, preserve sky visibility) ===
        // FIX #3: Reduce fog intensity to prevent obscuring sky reflection
        float fogStrength = max(0.2, pc.fogDensity * 0.5); // Reduced fog strength
        float distFog = 1.0 - exp(-fogStrength * viewDistance * 0.004); // Reduced fog rate
        float horizonFog = smoothstep(0.15, 0.0, cosTheta) * 0.2; // Reduced horizon fog
        float totalFog = clamp(distFog + horizonFog, 0.0, 0.4); // Lower cap to preserve sky
        
        vec3 fogColor = mix(vec3(0.04, 0.12, 0.22), pc.baseColor.rgb * 0.4, 0.25);
        finalColor = mix(finalColor, fogColor, totalFog);
        
        // === FINAL ALPHA ===
        float underwaterAlpha = clamp(pc.opacity, 0.0, 1.0) * (1.0 - totalFog * 0.3);
        
        outColor = vec4(finalColor, underwaterAlpha);
        return;
    }

    // === FINAL COLOR COMPOSITION ===
    // Above water branch: stronger Fresnel reflection, mild refraction
    // Increase reflection contribution and keep water tint subtle
    color = mix(color, waterBaseColor, 0.2);

    // Apply Fresnel-based transparency: ensure visible transparency
    // Lower base opacity and cap maximum to avoid a "solid" look
    // Reduce default surface opacity range (user controls via pc.opacity / ImGui)
    float finalAlpha = mix(0.04, 0.65, fresnel) * clamp(pc.opacity, 0.0, 1.0);

    outColor = vec4(color, finalAlpha);
}