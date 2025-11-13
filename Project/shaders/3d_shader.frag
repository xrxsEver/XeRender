#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;

layout(location = 0) out vec4 FragColor;

layout(binding = 1) uniform sampler2D baseTexture;
layout(binding = 3) uniform sampler2D metalnessTexture;
layout(binding = 4) uniform sampler2D normalTexture;
layout(binding = 5) uniform sampler2D specularTexture;

#define MAX_LIGHTS 2

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

layout(binding = 2) uniform LightInfo {
    Light lights[MAX_LIGHTS];
    vec3 viewPos;
    vec3 ambientColor;  // Ambient light color
    float ambientIntensity;  // Ambient light intensity
} lightInfo;

layout(binding = 6) uniform ToggleInfo {
    bool applyNormalMap;
    bool applyMetalnessMap;
    bool applySpecularMap;
    bool viewNormalOnly;
    bool viewMetalnessOnly;
    bool viewSpecularOnly;
    bool applyRimLight;  // Toggle for rim lighting
} toggleInfo;

void main() {
    vec3 baseColor = texture(baseTexture, fragTexCoord).rgb;
    vec3 normalMapColor = texture(normalTexture, fragTexCoord).rgb;
    vec3 metalnessMapColor = texture(metalnessTexture, fragTexCoord).rgb;
    vec3 specularMapColor = texture(specularTexture, fragTexCoord).rgb;

    vec3 finalColor = vec3(0.0);

    vec3 norm = normalize(fragNormal);
    if (toggleInfo.applyNormalMap) {
        norm = normalize(normalMapColor * 2.0 - 1.0);
    }

    vec3 viewDir = normalize(lightInfo.viewPos - fragPosition);

    // Ambient lighting
    vec3 ambient = lightInfo.ambientColor * lightInfo.ambientIntensity * baseColor;
    finalColor += ambient;

    for (int i = 0; i < MAX_LIGHTS; i++) {
        vec3 lightDir = normalize(lightInfo.lights[i].position - fragPosition);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * baseColor * lightInfo.lights[i].color * lightInfo.lights[i].intensity;

        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4.0);
        vec3 specular = spec * vec3(1.0) * lightInfo.lights[i].color * lightInfo.lights[i].intensity;

        if (toggleInfo.applyMetalnessMap) {
            diffuse *= metalnessMapColor;
        }
        if (toggleInfo.applySpecularMap) {
            specular *= specularMapColor;
        }

        finalColor += diffuse + specular;
    }

    // Rim Lighting
    if (toggleInfo.applyRimLight) {
        vec3 rimColor = vec3(1.0);  // White rim light by default
        float rimIntensity = 0.75;  // Control the strength of the rim light
        float rim = 1.0 - max(dot(norm, viewDir), 0.0);
        rim = smoothstep(0.0, 1.0, rim);
        finalColor += rim * rimColor * rimIntensity;
    }

    if (toggleInfo.viewNormalOnly) {
        finalColor = norm * 0.5 + 0.5;
    } else if (toggleInfo.viewMetalnessOnly) {
        finalColor = metalnessMapColor;
    } else if (toggleInfo.viewSpecularOnly) {
        finalColor = specularMapColor;
    }

    FragColor = vec4(finalColor, 1.0);
}
