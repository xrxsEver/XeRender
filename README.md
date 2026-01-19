# XeRender 2.0

**A Real-Time Underwater Rendering Engine Using Vulkan**

---

## Abstract

XeRender is a high-performance real-time rendering engine built on the Vulkan graphics API, specifically designed to simulate realistic underwater environments. The engine implements multiple rendering pipelines with physically-based lighting models, volumetric effects, and advanced post-processing techniques. All rendering parameters are exposed through an interactive ImGui interface, enabling real-time parameter adjustment for research, visualization, and artistic exploration of underwater optical phenomena.

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Rendering Pipelines](#rendering-pipelines)
4. [Optical Phenomena Implementation](#optical-phenomena-implementation)
5. [User Interface & Real-Time Controls](#user-interface--real-time-controls)
6. [Technical Specifications](#technical-specifications)
7. [Building & Installation](#building--installation)
8. [Dependencies](#dependencies)
9. [License](#license)

---

## Introduction

Underwater environments present unique challenges for real-time rendering due to the complex interaction of light with water. XeRender addresses these challenges by implementing:

- **Wavelength-dependent light absorption** using Beer-Lambert law
- **Volumetric god rays** (crepuscular rays) with radial blur techniques
- **Dynamic caustic patterns** simulating refracted sunlight on surfaces
- **Marine snow particles** for depth perception and scale reference
- **Fresnel-based reflections** using Schlick's approximation
- **Multi-layer wave normal perturbation** for realistic water surface animation

The engine supports three rendering modes for comparative analysis:
- **Baseline (BL)**: Simplified rendering for performance benchmarking
- **Physically-Based (PB)**: Full-quality physically accurate rendering
- **Optimized (OPT)**: Balanced quality-performance trade-off

---

## System Architecture

### Core Components

```
XeRender/
├── VulkanBase           # Core Vulkan initialization & main loop
├── SwapChainManager     # Swap chain creation & management
├── CommandPool/Buffer   # Command recording infrastructure
├── DAEDescriptorPool    # Descriptor set management
├── DAEUniformBufferObject # GPU uniform buffer handling
└── Camera               # Interactive camera system
```

### Rendering Infrastructure

| Component | Description |
|-----------|-------------|
| `WaterPipeline` | Water surface rendering with reflections/refractions |
| `UnderwaterWaterPipeline` | Volumetric underwater effects (fog, scattering) |
| `SkyboxPipeline` | Environment cubemap rendering |
| `xrxsPipeline` | Sun rays post-processing pipeline |

### Mesh Components

| Mesh | Purpose |
|------|---------|
| `WaterMesh` | Tessellated water surface geometry |
| `OceanBottomMesh` | Procedural ocean floor |
| `SkyboxMesh` | Cubemap geometry for environment |
| `DAEMesh` | General-purpose mesh loading |

---

## Rendering Pipelines

### 1. Water Surface Pipeline (`WaterPipeline`)

Handles above-water and water surface rendering with:

- **Dynamic wave animation** via dual-layer normal map sampling
- **Fresnel reflectance** using Schlick's approximation ($R_0 = 0.02$)
- **Refraction distortion** through DUDV mapping
- **Depth-based color blending** between shallow and deep water colors

**Shader Files:** `water.vert`, `water.frag`

### 2. Underwater Volumetric Pipeline (`UnderwaterWaterPipeline`)

Full-screen post-process for underwater atmosphere:

- **Wavelength-dependent absorption**: $T = e^{-\sigma \cdot d}$ where $\sigma_{RGB} = (0.18, 0.07, 0.03)$
- **Volumetric fog** with exponential falloff
- **Forward scattering** simulation
- **Marine snow particles** using multi-layer procedural generation

**Shader Files:** `underwater_water.vert`, `underwater_water.frag`

### 3. God Rays Pipeline (Sun Rays)

Radial blur-based volumetric light shafts:

- **Screen-space radial sampling** from projected sun position
- **Exponential decay** along ray direction
- **Caustic-modulated ray intensity**
- **Chromatic aberration** option for artistic effect

**Shader Files:** `sunrays.vert`, `sunrays.frag`

### 4. Skybox Pipeline (`SkyboxPipeline`)

Environment rendering using cubemap textures:

- **Seamless horizon blending** with water surface
- **HDR support** for realistic sky luminance

**Shader Files:** `skybox.vert`, `skybox.frag`

---

## Optical Phenomena Implementation

### Caustics

Animated light patterns caused by water surface refraction:

```glsl
vec3 getCaustics(vec3 worldPos) {
    vec2 causticUV = worldPos.xz * 0.5;
    float speed = time * 0.2;
    vec3 c1 = texture(causticTex, causticUV + vec2(speed, speed)).rgb;
    vec3 c2 = texture(causticTex, causticUV + vec2(-speed, 0.5 * speed)).rgb;
    return min(c1, c2) * causticIntensity * heightMask;
}
```

### God Rays (Crepuscular Rays)

Volumetric light shafts using screen-space radial blur with configurable:
- **Exposure**: Overall ray brightness
- **Decay**: Light falloff along ray
- **Density**: Ray sampling density
- **Scale**: Pattern frequency

### Marine Snow

Multi-layer particle system for scale reference:
- Fine dust layer (800 particles/unit)
- Small particles layer (400 particles/unit)
- Medium debris layer (150 particles/unit)
- Rare sparkle highlights

---

## User Interface & Real-Time Controls

XeRender provides comprehensive real-time parameter adjustment through ImGui:

### Surface Properties
| Parameter | Description | Range |
|-----------|-------------|-------|
| Color | Base water surface color | RGB |
| Texture | Normal/DUDV map intensity | 0.0 - 1.0 |
| Opacity | Water surface transparency | 0.0 - 1.0 |
| Distortion | Refraction distortion strength | 0.0 - 1.0 |

### Underwater Environment
| Parameter | Description | Range |
|-----------|-------------|-------|
| Shallow Color | Near-surface water tint | RGB |
| Deep Color | Deep water color | RGB |
| God Ray Intensity | Volumetric ray brightness | 0.0 - 2.0 |
| Caustic Intensity | Caustic pattern strength | 0.0 - 2.0 |
| Fog Density | Underwater visibility | 0.0 - 1.0 |

### Particle System
| Parameter | Description | Range |
|-----------|-------------|-------|
| Amount | Marine snow density | 0.0 - 1.0 |
| Size | Particle scale | 0.5 - 5.0 |
| Drift | Horizontal movement | 0.0 - 1.0 |
| Chromatic Aberration | Color fringing effect | 0.0 - 1.0 |

### God Ray Tuning
| Parameter | Description | Range |
|-----------|-------------|-------|
| Exposure | Ray brightness multiplier | 0.0 - 2.0 |
| Decay | Light falloff rate | 0.9 - 1.0 |
| Density | Sample density | 0.0 - 1.0 |
| Scale | Pattern scale factor | 0.1 - 2.0 |

### Debug Tools
- **Particles Debug**: Visualize particle distribution
- **Rays Debug**: Display ray sampling pattern
- **Chromatic Debug**: Show aberration channels
- **FPS Counter**: Real-time performance monitoring

---

## Technical Specifications

### Graphics API
- **Vulkan 1.x** with validation layers support
- **SPIR-V** compiled shaders
- **MSAA** multi-sample anti-aliasing
- **RTX auto-detection** for optimal GPU selection

### Rendering Features
- Push constants for per-frame parameters (64 bytes)
- Descriptor sets for texture binding
- Double-buffered uniform buffers
- Dynamic viewport/scissor state

### Uniform Buffer Layout
```cpp
struct UBO {
    mat4 model;      // Model transformation
    mat4 view;       // View matrix
    mat4 proj;       // Projection matrix
    vec4 lightPos;   // Sun/light position
    vec4 viewPos;    // Camera position
};
```

### Push Constant Structure
```cpp
struct WaterPush {
    float time;              // Animation time
    float scale;             // Pattern scale
    float debugRays;         // Debug visualization mode
    float renderingMode;     // BL=0, PB=1, OPT=2
    vec4 baseColor;          // Water base color
    vec4 lightColor;         // Light color
    float ambient;           // Ambient intensity
    float shininess;         // Specular power
    float causticIntensity;  // Caustic strength
    float distortionStrength;// Refraction distortion
    float godRayIntensity;   // Ray brightness
    float scatteringIntensity;// Scatter/snow amount
    float opacity;           // Surface opacity
    float fogDensity;        // Fog thickness
    float godExposure;       // Ray exposure
    float godDecay;          // Ray decay
    float godDensity;        // Ray density
    float godSampleScale;    // Ray scale
};
```

---

## Building & Installation

### Prerequisites

- **CMake** 3.10 or higher
- **Vulkan SDK** (with validation layers)
- **GLFW3** for windowing
- **GLM** for mathematics
- **C++17** compatible compiler

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/XeRender.git
cd XeRender

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Run
./VulkanProject
```

### Shader Compilation

Shaders must be compiled to SPIR-V before running:

```bash
cd Project/shaders
glslangValidator -V water.vert -o shaders/water.vert.spv
glslangValidator -V water.frag -o shaders/water.frag.spv
# Repeat for all shader files...
```

---

## Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Graphics API | Apache 2.0 |
| [GLFW](https://www.glfw.org/) | Window management | zlib |
| [GLM](https://github.com/g-truc/glm) | Mathematics | MIT |
| [Dear ImGui](https://github.com/ocornut/imgui) | User interface | MIT |
| [stb_image](https://github.com/nothings/stb) | Image loading | Public Domain |
| [nlohmann/json](https://github.com/nlohmann/json) | Configuration | MIT |
| [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) | OBJ loading | MIT |

---

## References

1. Premoze, S., & Ashikhmin, M. (2001). *Rendering Natural Waters*. Computer Graphics Forum.
2. Mitchell, J. L. (2007). *Light Shafts: Rendering Rays of Light Using Post-Processing*. GPU Gems 3.
3. Schlick, C. (1994). *An Inexpensive BRDF Model for Physically-based Rendering*. Eurographics.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

Developed as part of the Digital Arts & Entertainment program.

**XeRender** © 2025-2026

