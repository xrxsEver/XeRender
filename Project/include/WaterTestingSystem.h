#pragma once

// Prevent Windows min/max macros from conflicting with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <memory>

// ============================================================================
// TEST MODE CONFIGURATION
// ============================================================================

// Set to 1 for reduced test matrix (faster iteration), 0 for full exhaustive sweep
#ifndef FAST_TEST_MODE
#define FAST_TEST_MODE 1
#endif

// ============================================================================
// TEST PARAMETER CONSTANTS
// ============================================================================
// Centralized test parameters - modify these to scale tests up/down

namespace TestParams
{
    // Frame counts
#if FAST_TEST_MODE
    constexpr int PERF_TOTAL_FRAMES = 120;  // Reduced from 300
    constexpr int PERF_WARMUP_FRAMES = 5;   // Reduced from 10
    constexpr int PERF_REPEAT_COUNT = 3;    // Reduced from 10
    constexpr int IQ_TOTAL_FRAMES = 30;     // Reduced from 60
    constexpr int IQ_WARMUP_FRAMES = 3;     // Reduced from 5
    constexpr int SWEEP_TOTAL_FRAMES = 100; // Reduced from 200
    constexpr int SWEEP_WARMUP_FRAMES = 5;  // Reduced from 10
    constexpr int SWEEP_REPEAT_COUNT = 2;   // Reduced from 5
#else
    constexpr int PERF_TOTAL_FRAMES = 300; // Original full test
    constexpr int PERF_WARMUP_FRAMES = 10;
    constexpr int PERF_REPEAT_COUNT = 10;
    constexpr int IQ_TOTAL_FRAMES = 60;
    constexpr int IQ_WARMUP_FRAMES = 5;
    constexpr int SWEEP_TOTAL_FRAMES = 200;
    constexpr int SWEEP_WARMUP_FRAMES = 10;
    constexpr int SWEEP_REPEAT_COUNT = 5;
#endif

    // Sample count levels for trade-off sweep
    // FAST: MIN (performance-bound) and MID (balanced)
    // FULL: Full sweep from 1 to 16
#if FAST_TEST_MODE
    constexpr int SAMPLE_COUNT_MIN = 1; // Performance-bound
    constexpr int SAMPLE_COUNT_MID = 8; // Balanced
#else
    // Full sweep values defined in implementation
#endif

    // Caustic ray count levels
    // FAST: ZERO (off) and MID (representative on)
    // FULL: Full sweep from 16 to 256
#if FAST_TEST_MODE
    constexpr int CAUSTIC_RAYS_OFF = 0;  // Caustics disabled
    constexpr int CAUSTIC_RAYS_MID = 64; // Representative on
#else
    // Full sweep values defined in implementation
#endif
}

// ============================================================================
// CONFIGURATION STRUCTURES
// ============================================================================

// Turbidity levels for testing
enum class TurbidityLevel
{
    Low = 0,
    Medium = 1,
    High = 2
};

// Depth configuration
enum class DepthLevel
{
    Shallow = 0,
    Deep = 1
};

// Light motion configuration
enum class LightMotion
{
    Static = 0,
    Moving = 1
};

// Rendering mode (matches your existing modes)
enum class RenderingMode
{
    BL = 0,
    PB = 1,
    OPT = 2
};

// Test configuration structure
struct WaterTestConfig
{
    std::string name;
    TurbidityLevel turbidity = TurbidityLevel::Medium;
    DepthLevel depth = DepthLevel::Shallow;
    LightMotion lightMotion = LightMotion::Static;
    RenderingMode renderingMode = RenderingMode::PB;

    // Trade-off sweep parameters
    int sampleCount = 8;
    int causticRayCount = 64;
    bool asyncEnabled = false;
    bool tilingEnabled = false;

    // Test parameters - use centralized constants
    int totalFrames = TestParams::PERF_TOTAL_FRAMES;
    int warmupFrames = TestParams::PERF_WARMUP_FRAMES;
    int repeatCount = TestParams::PERF_REPEAT_COUNT;

    std::string toString() const
    {
        std::stringstream ss;
        ss << "Config[" << name << "]:"
           << " Turb=" << static_cast<int>(turbidity)
           << " Depth=" << static_cast<int>(depth)
           << " Light=" << static_cast<int>(lightMotion)
           << " Mode=" << static_cast<int>(renderingMode)
           << " Samples=" << sampleCount
           << " Caustics=" << causticRayCount;
        return ss.str();
    }
};

// ============================================================================
// CAMERA PATH DEFINITIONS
// ============================================================================

struct CameraKeyframe
{
    glm::vec3 position;
    float yaw;
    float pitch;
    float timestamp; // normalized 0-1
};

struct DeterministicCameraPath
{
    std::string name;
    std::vector<CameraKeyframe> keyframes;
    float totalDuration = 10.0f; // seconds

    // Interpolate camera state at time t (0-1)
    CameraKeyframe interpolate(float t) const
    {
        if (keyframes.empty())
            return {};
        if (keyframes.size() == 1)
            return keyframes[0];

        t = glm::clamp(t, 0.0f, 1.0f);

        // Find surrounding keyframes
        size_t i = 0;
        for (; i < keyframes.size() - 1; ++i)
        {
            if (keyframes[i + 1].timestamp >= t)
                break;
        }

        const auto &k0 = keyframes[i];
        size_t nextIdx = (i + 1 < keyframes.size()) ? (i + 1) : (keyframes.size() - 1);
        const auto &k1 = keyframes[nextIdx];

        float segmentT = (k1.timestamp > k0.timestamp)
                             ? (t - k0.timestamp) / (k1.timestamp - k0.timestamp)
                             : 0.0f;

        // Smooth interpolation
        float smoothT = segmentT * segmentT * (3.0f - 2.0f * segmentT);

        CameraKeyframe result;
        result.position = glm::mix(k0.position, k1.position, smoothT);
        result.yaw = k0.yaw + (k1.yaw - k0.yaw) * smoothT;
        result.pitch = k0.pitch + (k1.pitch - k0.pitch) * smoothT;
        result.timestamp = t;

        return result;
    }

    static DeterministicCameraPath createUnderwaterPath()
    {
        DeterministicCameraPath path;
        path.name = "UnderwaterSweep";
        path.totalDuration = 10.0f;

        // Full 360° smooth circular orbit around object at center (0,0,0)
        // Radius = 55, constant depth Y = -25 (underwater), 8 keyframes for smooth interpolation
        constexpr float R = 55.0f;
        constexpr float Y = -25.0f;
        constexpr float R45 = 38.89f; // R * sin(45°) ≈ R * 0.7071

        path.keyframes = {
            {glm::vec3(0.0f, Y, R), 0.0f, 5.0f, 0.0f},         // Back (positive Z), looking at center
            {glm::vec3(R45, Y, R45), 45.0f, 5.0f, 0.125f},     // Back-right
            {glm::vec3(R, Y, 0.0f), 90.0f, 5.0f, 0.25f},       // Right side
            {glm::vec3(R45, Y, -R45), 135.0f, 5.0f, 0.375f},   // Front-right
            {glm::vec3(0.0f, Y, -R), 180.0f, 5.0f, 0.5f},      // Front (negative Z)
            {glm::vec3(-R45, Y, -R45), -135.0f, 5.0f, 0.625f}, // Front-left
            {glm::vec3(-R, Y, 0.0f), -90.0f, 5.0f, 0.75f},     // Left side
            {glm::vec3(-R45, Y, R45), -45.0f, 5.0f, 0.875f},   // Back-left
            {glm::vec3(0.0f, Y, R), 0.0f, 5.0f, 1.0f}};        // Back (complete loop)

        return path;
    }

    static DeterministicCameraPath createSurfacePath()
    {
        DeterministicCameraPath path;
        path.name = "SurfaceSweep";
        path.totalDuration = 10.0f;

        // Full 360° smooth circular orbit around object at center (0,0,0)
        // Radius = 55, constant depth Y = -5, 8 keyframes for smooth interpolation
        constexpr float R = 55.0f;
        constexpr float Y = -5.0f;
        constexpr float R45 = 38.89f; // R * sin(45°) ≈ R * 0.7071

        path.keyframes = {
            {glm::vec3(0.0f, Y, R), 0.0f, 5.0f, 0.0f},         // Back (positive Z), looking at center
            {glm::vec3(R45, Y, R45), 45.0f, 5.0f, 0.125f},     // Back-right
            {glm::vec3(R, Y, 0.0f), 90.0f, 5.0f, 0.25f},       // Right side
            {glm::vec3(R45, Y, -R45), 135.0f, 5.0f, 0.375f},   // Front-right
            {glm::vec3(0.0f, Y, -R), 180.0f, 5.0f, 0.5f},      // Front (negative Z)
            {glm::vec3(-R45, Y, -R45), -135.0f, 5.0f, 0.625f}, // Front-left
            {glm::vec3(-R, Y, 0.0f), -90.0f, 5.0f, 0.75f},     // Left side
            {glm::vec3(-R45, Y, R45), -45.0f, 5.0f, 0.875f},   // Back-left
            {glm::vec3(0.0f, Y, R), -90.0f, 5.0f, 1.0f}};      // Back (complete loop)

        return path;
    }

    static DeterministicCameraPath createDepthTransitionPath()
    {
        DeterministicCameraPath path;
        path.name = "DepthTransition";
        path.totalDuration = 10.0f;

        // Full 360° smooth circular orbit around object at center (0,0,0)
        // Radius = 55, depth transitions from Y=-5 to Y=-35 while orbiting
        constexpr float R = 55.0f;
        constexpr float R45 = 38.89f; // R * sin(45°) ≈ R * 0.7071

        path.keyframes = {
            {glm::vec3(0.0f, -5.0f, R), 0.0f, 5.0f, 0.0f},           // Back, shallow
            {glm::vec3(R45, -9.0f, R45), 45.0f, 3.0f, 0.125f},       // Back-right
            {glm::vec3(R, -13.0f, 0.0f), 90.0f, 1.0f, 0.25f},        // Right side
            {glm::vec3(R45, -17.0f, -R45), 135.0f, -1.0f, 0.375f},   // Front-right
            {glm::vec3(0.0f, -21.0f, -R), 180.0f, -3.0f, 0.5f},      // Front
            {glm::vec3(-R45, -25.0f, -R45), -135.0f, -3.0f, 0.625f}, // Front-left
            {glm::vec3(-R, -29.0f, 0.0f), -90.0f, -1.0f, 0.75f},     // Left side
            {glm::vec3(-R45, -53.0f, R45), -45.0f, 1.0f, 0.875f},    // Back-left
            {glm::vec3(0.0f, -55.0f, R), 0.0f, 3.0f, 1.0f}};         // Back, deep

        return path;
    }
};

// ============================================================================
// METRICS STRUCTURES
// ============================================================================

struct FrameMetrics
{
    uint32_t frameIndex;
    double frameTimeMs;
    double gpuTimeMs;
    double cpuTimeMs;
    uint64_t timestampNs;

    // GPU memory usage (if available)
    uint64_t gpuMemoryUsedBytes = 0;

    // Additional timing breakdown
    double waterPassTimeMs = 0.0;
    double scenePassTimeMs = 0.0;
    double postProcessTimeMs = 0.0;

    // Camera state at this frame
    glm::vec3 cameraPosition;
    float cameraYaw;
    float cameraPitch;

    bool isWarmupFrame = false;
    bool isOutlier = false;
};

struct ImageQualityMetrics
{
    uint32_t frameIndex;
    double psnr = 0.0;   // Peak Signal-to-Noise Ratio
    double ssim = 0.0;   // Structural Similarity Index
    double mse = 0.0;    // Mean Squared Error
    double deltaE = 0.0; // Color difference (CIE Delta E)
    std::string screenshotPath;
};

struct TemporalMetrics
{
    uint32_t startFrame;
    uint32_t endFrame;
    double avgFrameToFrameSSIM = 0.0;
    double minFrameToFrameSSIM = 0.0;
    double temporalFlickerScore = 0.0; // Lower is better
    double opticalFlowCoherence = 0.0; // Higher is better
};

struct AggregatedRunMetrics
{
    std::string configName;
    int runIndex;
    int validFrameCount;
    int outlierCount;

    // Frame time statistics (in ms)
    double meanFrameTime;
    double medianFrameTime;
    double stddevFrameTime;
    double minFrameTime;
    double maxFrameTime;
    double percentile1Low; // 1% low (worst 1% average)
    double percentile99;   // 99th percentile

    // FPS statistics
    double meanFPS;
    double medianFPS;
    double fps1Low;

    // GPU time statistics
    double meanGpuTime;
    double medianGpuTime;
    double stddevGpuTime;

    // Image quality (if applicable)
    double avgSSIM;
    double avgPSNR;

    // Temporal stability
    double temporalStability;
};

// ============================================================================
// TEST RESULT STRUCTURES
// ============================================================================

struct TestRunResult
{
    WaterTestConfig config;
    int runIndex;
    std::vector<FrameMetrics> frameMetrics;
    std::vector<ImageQualityMetrics> imageQualityMetrics;
    TemporalMetrics temporalMetrics;
    AggregatedRunMetrics aggregated;

    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point endTime;
};

struct TestSuiteResult
{
    std::string suiteName;
    std::chrono::system_clock::time_point timestamp;
    std::vector<TestRunResult> runs;
    std::string outputDirectory;
};

// ============================================================================
// WATER TESTING SYSTEM CLASS
// ============================================================================

class WaterTestingSystem
{
public:
    WaterTestingSystem() = default;
    ~WaterTestingSystem() = default;

    // Initialize with Vulkan handles for GPU timing
    void initialize(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue graphicsQueue, uint32_t queueFamilyIndex);
    void cleanup();

    // ========== TEST EXECUTION ==========

    // Start a new test run with given configuration
    void startTestRun(const WaterTestConfig &config, int runIndex = 0);

    // Called each frame during test - frameTimeMs should be the actual frame time
    // measured AFTER drawFrame() completes (including GPU sync)
    void recordFrame(uint32_t frameIndex, double frameTimeMs, const glm::vec3 &camPos, float yaw, float pitch);

    // End current test run and aggregate results
    TestRunResult endTestRun();

    // Check if test is currently running
    bool isTestRunning() const { return m_isRunning; }

    // Get current frame index during test
    uint32_t getCurrentFrameIndex() const { return m_currentFrameIndex; }

    // Get total frames for current test
    uint32_t getTotalFrames() const { return m_currentConfig.totalFrames; }

    // Get progress percentage
    float getProgress() const
    {
        return m_isRunning ? (float)m_currentFrameIndex / m_currentConfig.totalFrames * 100.0f : 0.0f;
    }

    // ========== CAMERA PATH ==========

    // Set camera path for deterministic testing
    void setCameraPath(const DeterministicCameraPath &path) { m_cameraPath = path; }

    // Get interpolated camera state for current test frame
    CameraKeyframe getCameraStateForFrame(uint32_t frameIndex) const;

    // ========== CONFIGURATION PRESETS ==========

    // Generate all test configurations for comprehensive testing
    static std::vector<WaterTestConfig> generatePerformanceTestConfigs();
    static std::vector<WaterTestConfig> generateImageQualityTestConfigs();
    static std::vector<WaterTestConfig> generateTradeOffSweepConfigs();

    // ========== IMAGE QUALITY ==========

    // Save screenshot for current frame
    void captureScreenshot(uint32_t frameIndex, const std::vector<uint8_t> &pixels,
                           uint32_t width, uint32_t height);

    // Compute image quality metrics against reference
    ImageQualityMetrics computeImageQuality(const std::vector<uint8_t> &testImage,
                                            const std::vector<uint8_t> &referenceImage,
                                            uint32_t width, uint32_t height);

    // ========== TEMPORAL ANALYSIS ==========

    // Compute frame-to-frame SSIM for temporal stability
    double computeFrameToFrameSSIM(const std::vector<uint8_t> &frame1,
                                   const std::vector<uint8_t> &frame2,
                                   uint32_t width, uint32_t height);

    // Analyze temporal stability of a sequence
    TemporalMetrics analyzeTemporalStability(const std::vector<std::vector<uint8_t>> &frames,
                                             uint32_t width, uint32_t height);

    // ========== DATA EXPORT ==========

    // Export results to CSV (Excel-compatible)
    void exportToCSV(const TestSuiteResult &results, const std::string &filepath);

    // Export single run to CSV
    void exportRunToCSV(const TestRunResult &run, const std::string &filepath);

    // Append run to existing CSV
    void appendRunToCSV(const TestRunResult &run, const std::string &filepath);

    // Export aggregated summary
    void exportSummaryToCSV(const std::vector<AggregatedRunMetrics> &metrics, const std::string &filepath);

    // ========== STATISTICS ==========

    // Remove warmup frames and outliers, compute aggregated metrics
    AggregatedRunMetrics aggregateMetrics(const std::vector<FrameMetrics> &rawMetrics,
                                          const WaterTestConfig &config);

    // Get current test results
    const TestRunResult &getCurrentResult() const { return m_currentResult; }

    // Set output directory for screenshots and exports
    void setOutputDirectory(const std::string &dir) { m_outputDirectory = dir; }

    // Get configuration parameters to apply to rendering
    const WaterTestConfig &getCurrentConfig() const { return m_currentConfig; }

    // ========== GPU TIMING ==========

    // Reset timestamp queries - call at the START of command buffer recording
    void resetTimestampQueries(VkCommandBuffer cmd);

    // Write start timestamp - call AFTER vkBeginCommandBuffer, BEFORE any rendering
    void writeTimestampStart(VkCommandBuffer cmd);

    // Write end timestamp - call AFTER all rendering, BEFORE vkEndCommandBuffer
    void writeTimestampEnd(VkCommandBuffer cmd);

    // Read GPU timestamps and compute gpuTimeMs - call AFTER frame is complete (after vkWaitForFences)
    void readGpuTimestamps();

    // Get the last computed GPU time in milliseconds
    double getLastGpuTimeMs() const { return m_lastGpuTimeMs; }

private:
    // Vulkan handles for GPU timing
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueryPool m_timestampQueryPool = VK_NULL_HANDLE;
    uint32_t m_queueFamilyIndex = 0;
    float m_timestampPeriod = 1.0f; // nanoseconds per timestamp tick

    // GPU timing state
    bool m_timestampsWritten = false;     // True if timestamps were written this frame
    bool m_queryPoolNeedsReset = true;    // True if query pool needs reset before first use
    bool m_startTimestampWritten = false; // True if start timestamp was written this frame
    double m_lastGpuTimeMs = 0.0;         // Last computed GPU time
    static constexpr uint32_t QUERY_START = 0;
    static constexpr uint32_t QUERY_END = 1;

    // Test state
    bool m_isRunning = false;
    uint32_t m_currentFrameIndex = 0;
    WaterTestConfig m_currentConfig;
    TestRunResult m_currentResult;
    DeterministicCameraPath m_cameraPath;

    // Timing
    std::chrono::high_resolution_clock::time_point m_frameStartTime;
    std::chrono::high_resolution_clock::time_point m_testStartTime;

    // Output
    std::string m_outputDirectory = "test_results";

    // Frame buffer for temporal analysis
    std::vector<std::vector<uint8_t>> m_frameBuffer;

    // Helper functions
    void createTimestampQueryPool();
    double getTimestampMs(uint64_t timestamp) const;
    bool isOutlier(double value, double mean, double stddev, double threshold = 5.0) const;
    double computeSSIM(const uint8_t *img1, const uint8_t *img2, uint32_t width, uint32_t height);
    double computePSNR(const uint8_t *img1, const uint8_t *img2, uint32_t width, uint32_t height);
    double computeMSE(const uint8_t *img1, const uint8_t *img2, uint32_t width, uint32_t height);

    // Statistics helpers
    double calculateMean(const std::vector<double> &values) const;
    double calculateMedian(std::vector<double> values) const;
    double calculateStdDev(const std::vector<double> &values, double mean) const;
    double calculatePercentile(std::vector<double> values, double percentile) const;
};

// ============================================================================
// TEST REPORT GENERATOR
// ============================================================================

class TestReportGenerator
{
public:
    // Generate comprehensive Excel-compatible CSV report
    static void generateReport(const TestSuiteResult &suite, const std::string &basePath);

    // Generate performance comparison chart data
    static void generatePerformanceChartData(const std::vector<AggregatedRunMetrics> &metrics,
                                             const std::string &filepath);

    // Generate trade-off curve data (quality vs performance)
    static void generateTradeOffCurveData(const std::vector<TestRunResult> &results,
                                          const std::string &filepath);

    // Utility functions (public for use by WaterTestingSystem)
    static std::string escapeCSV(const std::string &str);
    static std::string formatTimestamp(const std::chrono::system_clock::time_point &tp);
};
