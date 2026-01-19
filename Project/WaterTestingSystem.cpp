// Prevent Windows min/max macros from conflicting with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "WaterTestingSystem.h"
#include <filesystem>
#include <iostream>
#include <cstring>

// ============================================================================
// INITIALIZATION AND CLEANUP
// ============================================================================

void WaterTestingSystem::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                                    VkQueue graphicsQueue, uint32_t queueFamilyIndex)
{
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_graphicsQueue = graphicsQueue;
    m_queueFamilyIndex = queueFamilyIndex;

    // Get timestamp period for converting GPU timestamps to nanoseconds
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    m_timestampPeriod = props.limits.timestampPeriod;

    // Create timestamp query pool for GPU timing
    createTimestampQueryPool();

    // Create output directory
    std::filesystem::create_directories(m_outputDirectory);

    std::cout << "[WaterTestingSystem] Initialized. Timestamp period: " << m_timestampPeriod << " ns\n";
}

void WaterTestingSystem::cleanup()
{
    if (m_timestampQueryPool != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(m_device, m_timestampQueryPool, nullptr);
        m_timestampQueryPool = VK_NULL_HANDLE;
    }
}

void WaterTestingSystem::createTimestampQueryPool()
{
    if (m_device == VK_NULL_HANDLE)
        return;

    VkQueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = 2; // QUERY_START and QUERY_END

    if (vkCreateQueryPool(m_device, &queryPoolInfo, nullptr, &m_timestampQueryPool) != VK_SUCCESS)
    {
        std::cerr << "[WaterTestingSystem] Warning: Failed to create timestamp query pool\n";
        m_timestampQueryPool = VK_NULL_HANDLE;
    }
    else
    {
        std::cout << "[WaterTestingSystem] Created timestamp query pool for GPU timing\n";
        m_queryPoolNeedsReset = true; // Mark that we need to reset before first use
    }
}

// ============================================================================
// GPU TIMING IMPLEMENTATION
// ============================================================================

void WaterTestingSystem::resetTimestampQueries(VkCommandBuffer cmd)
{
    if (m_timestampQueryPool == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
        return;

    // Reset the timestamp queries before writing new ones
    vkCmdResetQueryPool(cmd, m_timestampQueryPool, 0, 2);
    m_timestampsWritten = false;
    m_startTimestampWritten = false;
    m_queryPoolNeedsReset = false; // Reset has been issued in this command buffer
}

void WaterTestingSystem::writeTimestampStart(VkCommandBuffer cmd)
{
    if (m_timestampQueryPool == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
        return;

    // Only write if reset has been issued (reset must come first in the same command buffer)
    if (m_queryPoolNeedsReset)
    {
        // Reset wasn't called yet - skip this frame's timing
        m_startTimestampWritten = false;
        return;
    }

    // Write timestamp at the TOP of the pipeline (before any GPU work)
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_timestampQueryPool, QUERY_START);
    m_startTimestampWritten = true;
}

void WaterTestingSystem::writeTimestampEnd(VkCommandBuffer cmd)
{
    if (m_timestampQueryPool == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
        return;

    // Only write if start timestamp was written (prevents writing end without start)
    // This handles the case where test mode is activated mid-frame
    if (!m_startTimestampWritten)
    {
        // Start wasn't written - skip end timestamp to avoid validation error
        return;
    }

    // Write timestamp at the BOTTOM of the pipeline (after all GPU work)
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timestampQueryPool, QUERY_END);
    m_timestampsWritten = true;
}

void WaterTestingSystem::readGpuTimestamps()
{
    if (m_timestampQueryPool == VK_NULL_HANDLE)
    {
        m_lastGpuTimeMs = 0.0;
        return;
    }

    if (!m_timestampsWritten)
    {
        // Timestamps weren't written this frame - keep previous value or set to 0
        // Don't overwrite m_lastGpuTimeMs if we have a valid previous value
        return;
    }

    uint64_t timestamps[2] = {0, 0};

    // Read the timestamp results
    // VK_QUERY_RESULT_64_BIT: results are 64-bit values
    // VK_QUERY_RESULT_WAIT_BIT: wait for results to be available (GPU must have finished)
    VkResult result = vkGetQueryPoolResults(
        m_device,
        m_timestampQueryPool,
        0,                  // firstQuery
        2,                  // queryCount
        sizeof(timestamps), // dataSize
        timestamps,         // pData
        sizeof(uint64_t),   // stride
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result == VK_SUCCESS)
    {
        // Calculate GPU time in milliseconds
        // timestampPeriod is in nanoseconds per tick
        if (timestamps[QUERY_END] > timestamps[QUERY_START])
        {
            uint64_t timestampDiff = timestamps[QUERY_END] - timestamps[QUERY_START];
            double nanoseconds = static_cast<double>(timestampDiff) * m_timestampPeriod;
            m_lastGpuTimeMs = nanoseconds / 1000000.0; // Convert ns to ms
        }
        else
        {
            // Invalid timestamps (end <= start), keep previous value
            std::cerr << "[WaterTestingSystem] Warning: Invalid timestamps (end <= start)\n";
        }
    }
    else
    {
        // Query failed - could be VK_NOT_READY if GPU hasn't finished
        if (result != VK_NOT_READY)
        {
            std::cerr << "[WaterTestingSystem] Warning: vkGetQueryPoolResults failed with code " << result << "\n";
        }
    }

    m_timestampsWritten = false;
}

// ============================================================================
// TEST EXECUTION
// ============================================================================

void WaterTestingSystem::startTestRun(const WaterTestConfig &config, int runIndex)
{
    m_isRunning = true;
    m_currentFrameIndex = 0;
    m_currentConfig = config;

    // Reset GPU timing state for new test run
    m_queryPoolNeedsReset = true;
    m_timestampsWritten = false;
    m_startTimestampWritten = false;
    m_lastGpuTimeMs = 0.0;

    m_currentResult = TestRunResult{};
    m_currentResult.config = config;
    m_currentResult.runIndex = runIndex;
    m_currentResult.startTime = std::chrono::system_clock::now();
    m_currentResult.frameMetrics.clear();
    m_currentResult.frameMetrics.reserve(config.totalFrames);

    m_frameBuffer.clear();

    m_testStartTime = std::chrono::high_resolution_clock::now();
    m_frameStartTime = m_testStartTime;

    // Set default camera path if not set
    if (m_cameraPath.keyframes.empty())
    {
        m_cameraPath = DeterministicCameraPath::createUnderwaterPath();
    }

    std::cout << "[WaterTestingSystem] Started test run " << runIndex
              << " for config: " << config.toString() << "\n";
}

void WaterTestingSystem::recordFrame(uint32_t frameIndex, double frameTimeMs,
                                     const glm::vec3 &camPos, float yaw, float pitch)
{
    if (!m_isRunning)
        return;

    auto now = std::chrono::high_resolution_clock::now();

    // NOTE: readGpuTimestamps() should be called BEFORE this function in the mainLoop
    // (after vkQueueWaitIdle). Do NOT call it here as it would reset m_lastGpuTimeMs to 0.

    FrameMetrics metrics{};
    metrics.frameIndex = m_currentFrameIndex;
    metrics.frameTimeMs = frameTimeMs;                 // Total frame time (CPU + GPU + present)
    metrics.gpuTimeMs = m_lastGpuTimeMs;               // Actual GPU time from timestamp queries
    metrics.cpuTimeMs = frameTimeMs - m_lastGpuTimeMs; // Approximate CPU time (may be negative if GPU > total)
    if (metrics.cpuTimeMs < 0.0)
        metrics.cpuTimeMs = 0.0; // Clamp to 0
    metrics.timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now.time_since_epoch())
                              .count();
    metrics.cameraPosition = camPos;
    metrics.cameraYaw = yaw;
    metrics.cameraPitch = pitch;
    metrics.isWarmupFrame = (m_currentFrameIndex < static_cast<uint32_t>(m_currentConfig.warmupFrames));

    m_currentResult.frameMetrics.push_back(metrics);
    m_currentFrameIndex++;

    // Progress logging every 50 frames
    if (m_currentFrameIndex % 50 == 0)
    {
        double currentFPS = frameTimeMs > 0.0 ? 1000.0 / frameTimeMs : 0.0;
        std::cout << "[WaterTestingSystem] Progress: " << m_currentFrameIndex
                  << "/" << m_currentConfig.totalFrames
                  << " (" << std::fixed << std::setprecision(1) << getProgress() << "%)"
                  << " - FPS: " << std::setprecision(1) << currentFPS
                  << " | GPU: " << std::setprecision(2) << m_lastGpuTimeMs << "ms\n";
    }
}

TestRunResult WaterTestingSystem::endTestRun()
{
    m_isRunning = false;
    m_currentResult.endTime = std::chrono::system_clock::now();

    // Aggregate metrics
    m_currentResult.aggregated = aggregateMetrics(m_currentResult.frameMetrics, m_currentConfig);

    std::cout << "[WaterTestingSystem] Test run completed.\n";
    std::cout << "  Mean FPS: " << std::fixed << std::setprecision(2)
              << m_currentResult.aggregated.meanFPS << "\n";
    std::cout << "  Mean Frame Time: " << m_currentResult.aggregated.meanFrameTime << " ms\n";
    std::cout << "  1% Low FPS: " << m_currentResult.aggregated.fps1Low << "\n";
    std::cout << "  Outliers removed: " << m_currentResult.aggregated.outlierCount << "\n";

    return m_currentResult;
}

CameraKeyframe WaterTestingSystem::getCameraStateForFrame(uint32_t frameIndex) const
{
    if (m_currentConfig.totalFrames == 0)
        return {};
    float t = static_cast<float>(frameIndex) / static_cast<float>(m_currentConfig.totalFrames);
    return m_cameraPath.interpolate(t);
}

// ============================================================================
// CONFIGURATION PRESETS
// ============================================================================

std::vector<WaterTestConfig> WaterTestingSystem::generatePerformanceTestConfigs()
{
    std::vector<WaterTestConfig> configs;

#if FAST_TEST_MODE
    // =========================================================================
    // FAST MODE: Reduced representative subset
    // =========================================================================
    // Rationale for reductions:
    // - Turbidity: LOW and HIGH bracket the visual range (MEDIUM is interpolated)
    // - Depth: SHALLOW only (depth effects tested separately if needed)
    // - Light: MOVING only (most stressful, exposes worst-case performance)
    // - This reduces from 3×2×2×3 = 36 configs to 2×1×1×3 = 6 configs per mode
    // =========================================================================

    std::vector<TurbidityLevel> turbidities = {
        TurbidityLevel::Low, // Visual clarity baseline
        TurbidityLevel::High // Maximum scattering stress test
        // TurbidityLevel::Medium dropped - interpolates between extremes
    };

    std::vector<DepthLevel> depths = {
        DepthLevel::Shallow // Primary use case
        // DepthLevel::Deep dropped - test separately if depth effects matter
    };

    std::vector<LightMotion> lightMotions = {
        LightMotion::Moving // Most stressful case, exposes temporal artifacts
        // LightMotion::Static dropped - Moving is superset of Static challenges
    };

    std::vector<RenderingMode> modes = {
        RenderingMode::BL, // Baseline
        RenderingMode::PB, // Physically-based
        RenderingMode::OPT // Optimized
    };

    for (auto mode : modes)
    {
        for (auto turb : turbidities)
        {
            for (auto depth : depths)
            {
                for (auto light : lightMotions)
                {
                    WaterTestConfig config;
                    config.name = "Perf_Mode" + std::to_string(static_cast<int>(mode)) +
                                  "_T" + std::to_string(static_cast<int>(turb)) +
                                  "_D" + std::to_string(static_cast<int>(depth)) +
                                  "_L" + std::to_string(static_cast<int>(light));
                    config.renderingMode = mode;
                    config.turbidity = turb;
                    config.depth = depth;
                    config.lightMotion = light;
                    config.totalFrames = TestParams::PERF_TOTAL_FRAMES;
                    config.warmupFrames = TestParams::PERF_WARMUP_FRAMES;
                    config.repeatCount = TestParams::PERF_REPEAT_COUNT;
                    configs.push_back(config);
                }
            }
        }
    }

    std::cout << "[WaterTestingSystem] FAST_TEST_MODE: Generated " << configs.size()
              << " performance test configs (reduced from 36)\n";

#else
    // =========================================================================
    // FULL MODE: Original exhaustive sweep
    // =========================================================================
    // All combinations of turbidity, depth, light motion for each rendering mode

    std::vector<TurbidityLevel> turbidities = {TurbidityLevel::Low, TurbidityLevel::Medium, TurbidityLevel::High};
    std::vector<DepthLevel> depths = {DepthLevel::Shallow, DepthLevel::Deep};
    std::vector<LightMotion> lightMotions = {LightMotion::Static, LightMotion::Moving};
    std::vector<RenderingMode> modes = {RenderingMode::BL, RenderingMode::PB, RenderingMode::OPT};

    for (auto mode : modes)
    {
        for (auto turb : turbidities)
        {
            for (auto depth : depths)
            {
                for (auto light : lightMotions)
                {
                    WaterTestConfig config;
                    config.name = "Perf_Mode" + std::to_string(static_cast<int>(mode)) +
                                  "_T" + std::to_string(static_cast<int>(turb)) +
                                  "_D" + std::to_string(static_cast<int>(depth)) +
                                  "_L" + std::to_string(static_cast<int>(light));
                    config.renderingMode = mode;
                    config.turbidity = turb;
                    config.depth = depth;
                    config.lightMotion = light;
                    config.totalFrames = TestParams::PERF_TOTAL_FRAMES;
                    config.warmupFrames = TestParams::PERF_WARMUP_FRAMES;
                    config.repeatCount = TestParams::PERF_REPEAT_COUNT;
                    configs.push_back(config);
                }
            }
        }
    }

    std::cout << "[WaterTestingSystem] FULL_TEST_MODE: Generated " << configs.size()
              << " performance test configs (exhaustive sweep)\n";
#endif

    return configs;
}

std::vector<WaterTestConfig> WaterTestingSystem::generateImageQualityTestConfigs()
{
    std::vector<WaterTestConfig> configs;

    // Representative configs for image quality comparison
    // Note: Image quality tests always use the same reduced set since they're
    // primarily for visual comparison between rendering modes, not exhaustive coverage.

    std::vector<RenderingMode> modes = {RenderingMode::BL, RenderingMode::PB, RenderingMode::OPT};

    for (auto mode : modes)
    {
        WaterTestConfig config;
        config.name = "IQ_Mode" + std::to_string(static_cast<int>(mode));
        config.renderingMode = mode;
        config.turbidity = TurbidityLevel::Medium;
        config.depth = DepthLevel::Shallow;
        config.lightMotion = LightMotion::Static;
        config.totalFrames = TestParams::IQ_TOTAL_FRAMES;
        config.warmupFrames = TestParams::IQ_WARMUP_FRAMES;
        config.repeatCount = 1; // Single run for image quality
        configs.push_back(config);
    }

    return configs;
}

std::vector<WaterTestConfig> WaterTestingSystem::generateTradeOffSweepConfigs()
{
    std::vector<WaterTestConfig> configs;

#if FAST_TEST_MODE
    // =========================================================================
    // FAST MODE: Reduced trade-off sweep
    // =========================================================================
    // Rationale for reductions:
    // - Sample count: MIN (1) shows performance ceiling, MID (8) shows balanced quality
    // - Caustic rays: OFF (0) vs MID (64) shows impact of caustics feature
    // - Async/tiling: Only BASELINE OFF vs FULL OPTIMIZED ON (skip partial combos)
    // =========================================================================

    // Sample count sweep: MIN and MID only
    // Dropped: 2, 4, 16 - MIN and MID bracket the practical range
    std::vector<int> sampleCounts = {
        TestParams::SAMPLE_COUNT_MIN, // 1 - Performance-bound, lowest quality
        TestParams::SAMPLE_COUNT_MID  // 8 - Balanced quality/performance
    };

    for (int samples : sampleCounts)
    {
        WaterTestConfig config;
        config.name = "Sweep_Samples" + std::to_string(samples);
        config.sampleCount = samples;
        config.causticRayCount = TestParams::CAUSTIC_RAYS_MID; // Fixed mid-level caustics
        config.renderingMode = RenderingMode::PB;
        config.turbidity = TurbidityLevel::Low; // Clear water for sample visibility
        config.depth = DepthLevel::Shallow;
        config.lightMotion = LightMotion::Moving;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    // Caustic ray count sweep: OFF and MID only
    // Dropped: 16, 32, 128, 256 - OFF vs MID shows feature impact
    std::vector<int> causticRayCounts = {
        TestParams::CAUSTIC_RAYS_OFF, // 0 - Caustics disabled
        TestParams::CAUSTIC_RAYS_MID  // 64 - Representative quality
    };

    for (int rays : causticRayCounts)
    {
        WaterTestConfig config;
        config.name = "Sweep_Caustics" + std::to_string(rays);
        config.sampleCount = TestParams::SAMPLE_COUNT_MID; // Fixed mid-level samples
        config.causticRayCount = rays;
        config.renderingMode = RenderingMode::PB;
        config.turbidity = TurbidityLevel::Low; // Clear water for caustic visibility
        config.depth = DepthLevel::Shallow;
        config.lightMotion = LightMotion::Moving;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    // Async/tiling: BASELINE OFF vs FULL OPTIMIZED ON only
    // Dropped: {true,false}, {false,true} - only compare baseline to fully optimized
    std::vector<std::pair<bool, bool>> asyncTilingCombos = {
        {false, false}, // BASELINE: All optimizations OFF
        {true, true}    // FULL_OPT: All optimizations ON
    };

    for (auto [async, tiling] : asyncTilingCombos)
    {
        WaterTestConfig config;
        config.name = "Sweep_Async" + std::to_string(async) + "_Tiling" + std::to_string(tiling);
        config.asyncEnabled = async;
        config.tilingEnabled = tiling;
        config.sampleCount = TestParams::SAMPLE_COUNT_MID;
        config.causticRayCount = TestParams::CAUSTIC_RAYS_MID;
        config.renderingMode = RenderingMode::PB;
        config.turbidity = TurbidityLevel::Low;
        config.depth = DepthLevel::Shallow;
        config.lightMotion = LightMotion::Moving;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    std::cout << "[WaterTestingSystem] FAST_TEST_MODE: Generated " << configs.size()
              << " trade-off sweep configs (reduced from 14)\n";

#else
    // =========================================================================
    // FULL MODE: Original exhaustive sweep
    // =========================================================================

    // Sample count sweep - full range
    std::vector<int> sampleCounts = {1, 2, 4, 8, 16};

    for (int samples : sampleCounts)
    {
        WaterTestConfig config;
        config.name = "Sweep_Samples" + std::to_string(samples);
        config.sampleCount = samples;
        config.causticRayCount = 64;
        config.renderingMode = RenderingMode::PB;
        config.turbidity = TurbidityLevel::Medium;
        config.depth = DepthLevel::Shallow;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    // Caustic ray count sweep - full range
    std::vector<int> causticRayCounts = {16, 32, 64, 128, 256};

    for (int rays : causticRayCounts)
    {
        WaterTestConfig config;
        config.name = "Sweep_Caustics" + std::to_string(rays);
        config.sampleCount = 8;
        config.causticRayCount = rays;
        config.renderingMode = RenderingMode::PB;
        config.turbidity = TurbidityLevel::Medium;
        config.depth = DepthLevel::Shallow;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    // Async/tiling combinations - all permutations
    std::vector<std::pair<bool, bool>> asyncTilingCombos = {
        {false, false}, {true, false}, {false, true}, {true, true}};

    for (auto [async, tiling] : asyncTilingCombos)
    {
        WaterTestConfig config;
        config.name = "Sweep_Async" + std::to_string(async) + "_Tiling" + std::to_string(tiling);
        config.asyncEnabled = async;
        config.tilingEnabled = tiling;
        config.sampleCount = 8;
        config.causticRayCount = 64;
        config.renderingMode = RenderingMode::PB;
        config.totalFrames = TestParams::SWEEP_TOTAL_FRAMES;
        config.warmupFrames = TestParams::SWEEP_WARMUP_FRAMES;
        config.repeatCount = TestParams::SWEEP_REPEAT_COUNT;
        configs.push_back(config);
    }

    std::cout << "[WaterTestingSystem] FULL_TEST_MODE: Generated " << configs.size()
              << " trade-off sweep configs (exhaustive sweep)\n";
#endif

    return configs;
}

// ============================================================================
// IMAGE QUALITY METRICS
// ============================================================================

void WaterTestingSystem::captureScreenshot(uint32_t frameIndex, const std::vector<uint8_t> &pixels,
                                           uint32_t width, uint32_t height)
{
    if (pixels.empty())
        return;

    // Store for temporal analysis
    if (m_frameBuffer.size() < 30)
    { // Keep last 30 frames for temporal analysis
        m_frameBuffer.push_back(pixels);
    }
    else
    {
        m_frameBuffer.erase(m_frameBuffer.begin());
        m_frameBuffer.push_back(pixels);
    }

    // Save to disk for representative frames
    if (frameIndex % 30 == 0 || frameIndex == m_currentConfig.totalFrames - 1)
    {
        std::string filename = m_outputDirectory + "/" + m_currentConfig.name +
                               "_run" + std::to_string(m_currentResult.runIndex) +
                               "_frame" + std::to_string(frameIndex) + ".raw";

        std::ofstream file(filename, std::ios::binary);
        if (file.is_open())
        {
            file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
            file.close();
        }
    }
}

ImageQualityMetrics WaterTestingSystem::computeImageQuality(
    const std::vector<uint8_t> &testImage,
    const std::vector<uint8_t> &referenceImage,
    uint32_t width, uint32_t height)
{

    ImageQualityMetrics metrics{};

    if (testImage.size() != referenceImage.size() || testImage.empty())
    {
        return metrics;
    }

    metrics.mse = computeMSE(testImage.data(), referenceImage.data(), width, height);
    metrics.psnr = computePSNR(testImage.data(), referenceImage.data(), width, height);
    metrics.ssim = computeSSIM(testImage.data(), referenceImage.data(), width, height);

    return metrics;
}

double WaterTestingSystem::computeFrameToFrameSSIM(const std::vector<uint8_t> &frame1,
                                                   const std::vector<uint8_t> &frame2,
                                                   uint32_t width, uint32_t height)
{
    if (frame1.size() != frame2.size() || frame1.empty())
        return 0.0;
    return computeSSIM(frame1.data(), frame2.data(), width, height);
}

TemporalMetrics WaterTestingSystem::analyzeTemporalStability(
    const std::vector<std::vector<uint8_t>> &frames,
    uint32_t width, uint32_t height)
{

    TemporalMetrics metrics{};
    if (frames.size() < 2)
        return metrics;

    metrics.startFrame = 0;
    metrics.endFrame = static_cast<uint32_t>(frames.size() - 1);

    std::vector<double> ssimValues;
    std::vector<double> diffValues;

    for (size_t i = 1; i < frames.size(); ++i)
    {
        double ssim = computeSSIM(frames[i - 1].data(), frames[i].data(), width, height);
        ssimValues.push_back(ssim);

        // Compute frame difference for flicker detection
        double totalDiff = 0.0;
        for (size_t j = 0; j < frames[i].size(); ++j)
        {
            totalDiff += std::abs(static_cast<int>(frames[i][j]) - static_cast<int>(frames[i - 1][j]));
        }
        diffValues.push_back(totalDiff / frames[i].size());
    }

    metrics.avgFrameToFrameSSIM = calculateMean(ssimValues);
    metrics.minFrameToFrameSSIM = *std::min_element(ssimValues.begin(), ssimValues.end());

    // Flicker score based on variance of frame differences
    double meanDiff = calculateMean(diffValues);
    metrics.temporalFlickerScore = calculateStdDev(diffValues, meanDiff);

    // Coherence - inverse of flicker (higher is better)
    metrics.opticalFlowCoherence = 1.0 / (1.0 + metrics.temporalFlickerScore);

    return metrics;
}

// ============================================================================
// STATISTICS AND AGGREGATION
// ============================================================================

AggregatedRunMetrics WaterTestingSystem::aggregateMetrics(
    const std::vector<FrameMetrics> &rawMetrics,
    const WaterTestConfig &config)
{

    AggregatedRunMetrics agg{};
    agg.configName = config.name;

    // Step 1: Remove warmup frames
    std::vector<double> frameTimes;
    std::vector<double> gpuTimes;

    for (const auto &m : rawMetrics)
    {
        if (!m.isWarmupFrame)
        {
            frameTimes.push_back(m.frameTimeMs);
            gpuTimes.push_back(m.gpuTimeMs);
        }
    }

    if (frameTimes.empty())
        return agg;

    // Step 2: Calculate initial statistics
    double mean = calculateMean(frameTimes);
    double stddev = calculateStdDev(frameTimes, mean);

    // Step 3: Remove outliers (> mean + 5σ)
    std::vector<double> cleanFrameTimes;
    std::vector<double> cleanGpuTimes;
    int outlierCount = 0;

    for (size_t i = 0; i < frameTimes.size(); ++i)
    {
        if (!isOutlier(frameTimes[i], mean, stddev, 5.0))
        {
            cleanFrameTimes.push_back(frameTimes[i]);
            cleanGpuTimes.push_back(gpuTimes[i]);
        }
        else
        {
            outlierCount++;
        }
    }

    agg.validFrameCount = static_cast<int>(cleanFrameTimes.size());
    agg.outlierCount = outlierCount;

    if (cleanFrameTimes.empty())
        return agg;

    // Step 4: Calculate final statistics
    agg.meanFrameTime = calculateMean(cleanFrameTimes);
    agg.medianFrameTime = calculateMedian(cleanFrameTimes);
    agg.stddevFrameTime = calculateStdDev(cleanFrameTimes, agg.meanFrameTime);
    agg.minFrameTime = *std::min_element(cleanFrameTimes.begin(), cleanFrameTimes.end());
    agg.maxFrameTime = *std::max_element(cleanFrameTimes.begin(), cleanFrameTimes.end());
    agg.percentile1Low = calculatePercentile(cleanFrameTimes, 99.0); // Worst 1%
    agg.percentile99 = calculatePercentile(cleanFrameTimes, 99.0);

    // FPS calculations
    std::vector<double> fpsValues;
    for (double ft : cleanFrameTimes)
    {
        if (ft > 0)
            fpsValues.push_back(1000.0 / ft);
    }

    agg.meanFPS = calculateMean(fpsValues);
    agg.medianFPS = calculateMedian(fpsValues);
    agg.fps1Low = calculatePercentile(fpsValues, 1.0); // Worst 1%

    // GPU time statistics
    agg.meanGpuTime = calculateMean(cleanGpuTimes);
    agg.medianGpuTime = calculateMedian(cleanGpuTimes);
    agg.stddevGpuTime = calculateStdDev(cleanGpuTimes, agg.meanGpuTime);

    return agg;
}

// ============================================================================
// DATA EXPORT (CSV / Excel Compatible)
// ============================================================================

void WaterTestingSystem::exportToCSV(const TestSuiteResult &results, const std::string &filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[WaterTestingSystem] Failed to open file for export: " << filepath << "\n";
        return;
    }

    // Header
    file << "Suite," << results.suiteName << "\n";
    file << "Timestamp," << TestReportGenerator::formatTimestamp(results.timestamp) << "\n\n";

    // Aggregated results header
    file << "Config,Run,ValidFrames,Outliers,MeanFPS,MedianFPS,1%LowFPS,"
         << "MeanFrameTime_ms,MedianFrameTime_ms,StdDevFrameTime_ms,"
         << "MinFrameTime_ms,MaxFrameTime_ms,99thPercentile_ms,"
         << "MeanGpuTime_ms,MedianGpuTime_ms,StdDevGpuTime_ms\n";

    for (const auto &run : results.runs)
    {
        const auto &a = run.aggregated;
        file << a.configName << ","
             << run.runIndex << ","
             << a.validFrameCount << ","
             << a.outlierCount << ","
             << std::fixed << std::setprecision(2)
             << a.meanFPS << ","
             << a.medianFPS << ","
             << a.fps1Low << ","
             << a.meanFrameTime << ","
             << a.medianFrameTime << ","
             << a.stddevFrameTime << ","
             << a.minFrameTime << ","
             << a.maxFrameTime << ","
             << a.percentile99 << ","
             << a.meanGpuTime << ","
             << a.medianGpuTime << ","
             << a.stddevGpuTime << "\n";
    }

    file.close();
    std::cout << "[WaterTestingSystem] Exported results to: " << filepath << "\n";
}

void WaterTestingSystem::exportRunToCSV(const TestRunResult &run, const std::string &filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    // Per-frame data header
    file << "FrameIndex,FrameTime_ms,GpuTime_ms,CpuTime_ms,Timestamp_ns,"
         << "CameraX,CameraY,CameraZ,CameraYaw,CameraPitch,"
         << "IsWarmup,IsOutlier\n";

    for (const auto &m : run.frameMetrics)
    {
        file << m.frameIndex << ","
             << std::fixed << std::setprecision(4)
             << m.frameTimeMs << ","
             << m.gpuTimeMs << ","
             << m.cpuTimeMs << ","
             << m.timestampNs << ","
             << m.cameraPosition.x << ","
             << m.cameraPosition.y << ","
             << m.cameraPosition.z << ","
             << m.cameraYaw << ","
             << m.cameraPitch << ","
             << (m.isWarmupFrame ? 1 : 0) << ","
             << (m.isOutlier ? 1 : 0) << "\n";
    }

    file.close();
}

void WaterTestingSystem::appendRunToCSV(const TestRunResult &run, const std::string &filepath)
{
    bool fileExists = std::filesystem::exists(filepath);

    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "[WaterTestingSystem] Failed to open file for append: " << filepath << "\n";
        return;
    }

    // Write header if new file
    if (!fileExists)
    {
        file << "Timestamp,Config,Run,ValidFrames,Outliers,MeanFPS,MedianFPS,1%LowFPS,"
             << "MeanFrameTime_ms,MedianFrameTime_ms,StdDevFrameTime_ms,"
             << "MinFrameTime_ms,MaxFrameTime_ms,99thPercentile_ms,"
             << "MeanGpuTime_ms,MedianGpuTime_ms,StdDevGpuTime_ms,"
             << "Turbidity,Depth,LightMotion,RenderMode,SampleCount,CausticRays\n";
    }

    const auto &a = run.aggregated;
    const auto &c = run.config;

    file << TestReportGenerator::formatTimestamp(run.endTime) << ","
         << c.name << ","
         << run.runIndex << ","
         << a.validFrameCount << ","
         << a.outlierCount << ","
         << std::fixed << std::setprecision(2)
         << a.meanFPS << ","
         << a.medianFPS << ","
         << a.fps1Low << ","
         << a.meanFrameTime << ","
         << a.medianFrameTime << ","
         << a.stddevFrameTime << ","
         << a.minFrameTime << ","
         << a.maxFrameTime << ","
         << a.percentile99 << ","
         << a.meanGpuTime << ","
         << a.medianGpuTime << ","
         << a.stddevGpuTime << ","
         << static_cast<int>(c.turbidity) << ","
         << static_cast<int>(c.depth) << ","
         << static_cast<int>(c.lightMotion) << ","
         << static_cast<int>(c.renderingMode) << ","
         << c.sampleCount << ","
         << c.causticRayCount << "\n";

    file.close();
    std::cout << "[WaterTestingSystem] Appended run to: " << filepath << "\n";
}

void WaterTestingSystem::exportSummaryToCSV(const std::vector<AggregatedRunMetrics> &metrics,
                                            const std::string &filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    file << "Config,ValidFrames,MeanFPS,MedianFPS,1%LowFPS,MeanFrameTime_ms,StdDevFrameTime_ms\n";

    for (const auto &m : metrics)
    {
        file << m.configName << ","
             << m.validFrameCount << ","
             << std::fixed << std::setprecision(2)
             << m.meanFPS << ","
             << m.medianFPS << ","
             << m.fps1Low << ","
             << m.meanFrameTime << ","
             << m.stddevFrameTime << "\n";
    }

    file.close();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

double WaterTestingSystem::getTimestampMs(uint64_t timestamp) const
{
    return static_cast<double>(timestamp) * m_timestampPeriod / 1000000.0;
}

bool WaterTestingSystem::isOutlier(double value, double mean, double stddev, double threshold) const
{
    return std::abs(value - mean) > threshold * stddev;
}

double WaterTestingSystem::computeSSIM(const uint8_t *img1, const uint8_t *img2,
                                       uint32_t width, uint32_t height)
{
    // Simplified SSIM implementation (luminance channel only for speed)
    const double C1 = 6.5025;  // (0.01 * 255)^2
    const double C2 = 58.5225; // (0.03 * 255)^2

    size_t pixelCount = width * height;
    if (pixelCount == 0)
        return 0.0;

    // Assume RGBA format, use only R channel for speed
    double mean1 = 0.0, mean2 = 0.0;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        mean1 += img1[i * 4];
        mean2 += img2[i * 4];
    }
    mean1 /= pixelCount;
    mean2 /= pixelCount;

    double var1 = 0.0, var2 = 0.0, covar = 0.0;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        double d1 = img1[i * 4] - mean1;
        double d2 = img2[i * 4] - mean2;
        var1 += d1 * d1;
        var2 += d2 * d2;
        covar += d1 * d2;
    }
    var1 /= pixelCount;
    var2 /= pixelCount;
    covar /= pixelCount;

    double ssim = ((2 * mean1 * mean2 + C1) * (2 * covar + C2)) /
                  ((mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2));

    return ssim;
}

double WaterTestingSystem::computePSNR(const uint8_t *img1, const uint8_t *img2,
                                       uint32_t width, uint32_t height)
{
    double mse = computeMSE(img1, img2, width, height);
    if (mse < 1e-10)
        return 100.0; // Perfect match
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

double WaterTestingSystem::computeMSE(const uint8_t *img1, const uint8_t *img2,
                                      uint32_t width, uint32_t height)
{
    size_t pixelCount = width * height * 4; // RGBA
    if (pixelCount == 0)
        return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        double diff = static_cast<double>(img1[i]) - static_cast<double>(img2[i]);
        sum += diff * diff;
    }

    return sum / pixelCount;
}

double WaterTestingSystem::calculateMean(const std::vector<double> &values) const
{
    if (values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double WaterTestingSystem::calculateMedian(std::vector<double> values) const
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0)
    {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

double WaterTestingSystem::calculateStdDev(const std::vector<double> &values, double mean) const
{
    if (values.size() < 2)
        return 0.0;
    double sum = 0.0;
    for (double v : values)
    {
        double diff = v - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / (values.size() - 1));
}

double WaterTestingSystem::calculatePercentile(std::vector<double> values, double percentile) const
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>((percentile / 100.0) * (values.size() - 1));
    return values[index];
}

// ============================================================================
// TEST REPORT GENERATOR
// ============================================================================

void TestReportGenerator::generateReport(const TestSuiteResult &suite, const std::string &basePath)
{
    std::filesystem::create_directories(basePath);

    // Main summary file
    std::string summaryPath = basePath + "/test_summary.csv";
    std::ofstream summary(summaryPath);

    summary << "Water Rendering Test Report\n";
    summary << "Generated," << formatTimestamp(suite.timestamp) << "\n";
    summary << "Total Runs," << suite.runs.size() << "\n\n";

    // Aggregated results
    summary << "AGGREGATED RESULTS\n";
    summary << "Config,Run,MeanFPS,MedianFPS,1%LowFPS,MeanFrameTime_ms,StdDev_ms,Outliers\n";

    for (const auto &run : suite.runs)
    {
        const auto &a = run.aggregated;
        summary << a.configName << ","
                << run.runIndex << ","
                << std::fixed << std::setprecision(2)
                << a.meanFPS << ","
                << a.medianFPS << ","
                << a.fps1Low << ","
                << a.meanFrameTime << ","
                << a.stddevFrameTime << ","
                << a.outlierCount << "\n";
    }

    summary.close();
    std::cout << "[TestReportGenerator] Generated report: " << summaryPath << "\n";
}

void TestReportGenerator::generatePerformanceChartData(
    const std::vector<AggregatedRunMetrics> &metrics,
    const std::string &filepath)
{

    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    file << "Config,MeanFPS,MedianFPS,1%LowFPS,MeanFrameTime_ms\n";

    for (const auto &m : metrics)
    {
        file << m.configName << ","
             << std::fixed << std::setprecision(2)
             << m.meanFPS << ","
             << m.medianFPS << ","
             << m.fps1Low << ","
             << m.meanFrameTime << "\n";
    }

    file.close();
}

void TestReportGenerator::generateTradeOffCurveData(
    const std::vector<TestRunResult> &results,
    const std::string &filepath)
{

    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    // For trade-off curves: quality metric vs performance
    file << "Config,SampleCount,CausticRays,MeanFPS,FrameTime_ms,SSIM,PSNR\n";

    for (const auto &r : results)
    {
        file << r.config.name << ","
             << r.config.sampleCount << ","
             << r.config.causticRayCount << ","
             << std::fixed << std::setprecision(2)
             << r.aggregated.meanFPS << ","
             << r.aggregated.meanFrameTime << ","
             << r.aggregated.avgSSIM << ","
             << r.aggregated.avgPSNR << "\n";
    }

    file.close();
}

std::string TestReportGenerator::escapeCSV(const std::string &str)
{
    if (str.find(',') != std::string::npos || str.find('"') != std::string::npos)
    {
        std::string escaped = "\"";
        for (char c : str)
        {
            if (c == '"')
                escaped += "\"\"";
            else
                escaped += c;
        }
        escaped += "\"";
        return escaped;
    }
    return str;
}

std::string TestReportGenerator::formatTimestamp(const std::chrono::system_clock::time_point &tp)
{
    auto time = std::chrono::system_clock::to_time_t(tp);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}
