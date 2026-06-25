#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

using std::max;
using std::min;
using std::isfinite;

struct CpuDim3 { int x = 1; int y = 1; int z = 1; };
thread_local CpuDim3 threadIdx;
thread_local CpuDim3 blockIdx;
thread_local CpuDim3 blockDim;
thread_local CpuDim3 gridDim;

using cudaStream_t = void*;

#define __global__
#define __shared__
#define __syncthreads() ((void)0)
#define __threadfence() ((void)0)

namespace {
std::mutex g_atomic_mutex;

inline int cpu_resolve_workers(int requested, int64_t logical_threads) {
    if (logical_threads <= 1) return 1;
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 1;
    int w = requested > 0 ? requested : hw;
    if (w < 1) w = 1;
    if (w > 1024) w = 1024;
    if (static_cast<int64_t>(w) > logical_threads) w = static_cast<int>(logical_threads);
    return std::max(1, w);
}
}  // namespace

inline int atomicAdd(int* addr, int val) {
#if defined(_MSC_VER)
    return static_cast<int>(_InterlockedExchangeAdd(reinterpret_cast<volatile long*>(addr), static_cast<long>(val)));
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_fetch_add(addr, val, __ATOMIC_RELAXED);
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    *addr = old + val;
    return old;
#endif
}

inline unsigned int atomicAdd(unsigned int* addr, unsigned int val) {
#if defined(_MSC_VER)
    return static_cast<unsigned int>(_InterlockedExchangeAdd(reinterpret_cast<volatile long*>(addr), static_cast<long>(val)));
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_fetch_add(addr, val, __ATOMIC_RELAXED);
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    *addr = old + val;
    return old;
#endif
}

inline float atomicAdd(float* addr, float val) {
#if defined(_MSC_VER)
    volatile long* bits = reinterpret_cast<volatile long*>(addr);
    long old_bits = *bits;
    for (;;) {
        long assumed = old_bits;
        float old_value;
        std::memcpy(&old_value, &assumed, sizeof(float));
        const float new_value = old_value + val;
        long new_bits;
        std::memcpy(&new_bits, &new_value, sizeof(float));
        old_bits = _InterlockedCompareExchange(bits, new_bits, assumed);
        if (old_bits == assumed) return old_value;
    }
#elif defined(__GNUC__) || defined(__clang__)
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    uint32_t* bits = reinterpret_cast<uint32_t*>(addr);
    uint32_t old_bits = __atomic_load_n(bits, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t assumed = old_bits;
        float old_value;
        std::memcpy(&old_value, &assumed, sizeof(float));
        const float new_value = old_value + val;
        uint32_t new_bits;
        std::memcpy(&new_bits, &new_value, sizeof(float));
        if (__atomic_compare_exchange_n(bits, &old_bits, new_bits, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return old_value;
        }
    }
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    float old = *addr;
    *addr = old + val;
    return old;
#endif
}

template <typename Fn>
void cpu_parallel_ranges(int64_t n, int requested_workers, Fn&& fn) {
    if (n <= 0) return;
    const int T = cpu_resolve_workers(requested_workers, n);
    if (T <= 1 || n < 4096) {
        fn(0, n, 0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(T - 1));
    const int64_t chunk = (n + T - 1) / T;
    for (int tid = 1; tid < T; ++tid) {
        const int64_t begin = static_cast<int64_t>(tid) * chunk;
        const int64_t end = std::min<int64_t>(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&, begin, end, tid]() { fn(begin, end, tid); });
    }
    fn(0, std::min<int64_t>(n, chunk), 0);
    for (auto& th : pool) th.join();
}

template <typename T>
void parallel_fill_cpu(T* data, int n, T value, int workers) {
    if (data == nullptr || n <= 0) return;
    cpu_parallel_ranges(static_cast<int64_t>(n), workers, [&](int64_t begin, int64_t end, int) {
        std::fill(data + begin, data + end, value);
    });
}

inline int atomicMin(int* addr, int val) {
#if defined(_MSC_VER)
    volatile long* p = reinterpret_cast<volatile long*>(addr);
    long old = *p;
    while (val < old) {
        long prev = _InterlockedCompareExchange(p, static_cast<long>(val), old);
        if (prev == old) break;
        old = prev;
    }
    return static_cast<int>(old);
#elif defined(__GNUC__) || defined(__clang__)
    int old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val < old && !__atomic_compare_exchange_n(addr, &old, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return old;
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (val < old) *addr = val;
    return old;
#endif
}

inline int atomicMax(int* addr, int val) {
#if defined(_MSC_VER)
    volatile long* p = reinterpret_cast<volatile long*>(addr);
    long old = *p;
    while (val > old) {
        long prev = _InterlockedCompareExchange(p, static_cast<long>(val), old);
        if (prev == old) break;
        old = prev;
    }
    return static_cast<int>(old);
#elif defined(__GNUC__) || defined(__clang__)
    int old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val > old && !__atomic_compare_exchange_n(addr, &old, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return old;
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (val > old) *addr = val;
    return old;
#endif
}

inline int atomicExch(int* addr, int val) {
#if defined(_MSC_VER)
    return static_cast<int>(_InterlockedExchange(reinterpret_cast<volatile long*>(addr), static_cast<long>(val)));
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_exchange_n(addr, val, __ATOMIC_RELAXED);
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    *addr = val;
    return old;
#endif
}

inline unsigned int atomicExch(unsigned int* addr, unsigned int val) {
#if defined(_MSC_VER)
    return static_cast<unsigned int>(_InterlockedExchange(reinterpret_cast<volatile long*>(addr), static_cast<long>(val)));
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_exchange_n(addr, val, __ATOMIC_RELAXED);
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    *addr = val;
    return old;
#endif
}

inline int atomicCAS(int* addr, int compare, int val) {
#if defined(_MSC_VER)
    return static_cast<int>(_InterlockedCompareExchange(reinterpret_cast<volatile long*>(addr), static_cast<long>(val), static_cast<long>(compare)));
#elif defined(__GNUC__) || defined(__clang__)
    int expected = compare;
    __atomic_compare_exchange_n(addr, &expected, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (old == compare) *addr = val;
    return old;
#endif
}

inline unsigned int atomicCAS(unsigned int* addr, unsigned int compare, unsigned int val) {
#if defined(_MSC_VER)
    return static_cast<unsigned int>(_InterlockedCompareExchange(reinterpret_cast<volatile long*>(addr), static_cast<long>(val), static_cast<long>(compare)));
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int expected = compare;
    __atomic_compare_exchange_n(addr, &expected, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
#else
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    if (old == compare) *addr = val;
    return old;
#endif
}

#define AVABM_PART_SKIP_COMMON 1
#include "main_common_cpu.hpp"
#include "../cuda/main_core_grid_spawn.cu"
#include "../cuda/main_core_decision.cu"
#include "../cuda/main_core_motion.cu"

namespace {

template <typename Kernel, typename... Args>
void launch_cpu_kernel(int blocks, int threads, int workers, Kernel kernel, Args... args) {
    if (blocks <= 0 || threads <= 0) return;
    const int64_t total = static_cast<int64_t>(blocks) * static_cast<int64_t>(threads);
    if (total <= 0) return;
    const int W = cpu_resolve_workers(workers, total);
    auto run_range = [&](int64_t begin, int64_t end) {
        blockDim = CpuDim3{threads, 1, 1};
        gridDim = CpuDim3{blocks, 1, 1};
        for (int64_t linear = begin; linear < end; ++linear) {
            blockIdx = CpuDim3{static_cast<int>(linear / threads), 0, 0};
            threadIdx = CpuDim3{static_cast<int>(linear % threads), 0, 0};
            kernel(args...);
        }
    };
    if (W <= 1 || total < 1024) {
        run_range(0, total);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(W - 1));
    const int64_t chunk = (total + W - 1) / W;
    for (int wid = 1; wid < W; ++wid) {
        const int64_t begin = static_cast<int64_t>(wid) * chunk;
        const int64_t end = std::min<int64_t>(total, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([=, &run_range]() { run_range(begin, end); });
    }
    run_range(0, std::min<int64_t>(total, chunk));
    for (auto& th : pool) th.join();
}

inline void clear_int_cpu(int* data, int n, int value, int workers) {
    parallel_fill_cpu(data, n, value, workers);
}

inline void clear_float_cpu(float* data, int n, float value, int workers) {
    parallel_fill_cpu(data, n, value, workers);
}

inline void begin_world_grid_rebuild_cpu(SpatialGrid& grid, int world_cells, int workers) {
    if (grid.cell_head == nullptr || world_cells <= 0) return;
    parallel_fill_cpu(grid.cell_head, world_cells, WORLD_CELL_EMPTY, workers);
    if (grid.cell_epoch != nullptr) {
        parallel_fill_cpu(grid.cell_epoch, world_cells, 0, workers);
    }
}

inline void begin_lane_grid_rebuild_cpu(SpatialGrid& grid, int num_lanes, int workers) {
    if (grid.lane_cell_head == nullptr || num_lanes <= 0 || grid.lane_cells_per_lane <= 0) return;
    parallel_fill_cpu(grid.lane_cell_head, num_lanes * grid.lane_cells_per_lane, WORLD_CELL_EMPTY, workers);
}

inline void rebuild_active_list_cpu(ECSArrays ecs, int* active_ids, int* active_count, int max_entities, int threads, int entity_blocks, int workers) {
#if AVABM_ACTIVE_LIST_ENABLED
    if (active_ids == nullptr || active_count == nullptr || max_entities <= 0) return;
    *active_count = 0;
    launch_cpu_kernel(entity_blocks, threads, workers, compact_active_entities_kernel, ecs, active_ids, active_count, max_entities);
#else
    (void)ecs; (void)active_ids; (void)active_count; (void)max_entities; (void)threads; (void)entity_blocks; (void)workers;
#endif
}

inline void rebuild_active_archetypes_cpu(ECSArrays ecs, const int* active_ids, const int* active_count, int* lane_active_ids, int* lane_active_count, int* connector_active_ids, int* connector_active_count, int max_entities, int threads, int active_blocks, int workers) {
#if AVABM_ACTIVE_LIST_ENABLED
    if (lane_active_ids == nullptr || lane_active_count == nullptr || connector_active_ids == nullptr || connector_active_count == nullptr || max_entities <= 0) return;
    *lane_active_count = 0;
    *connector_active_count = 0;
    launch_cpu_kernel(active_blocks, threads, workers, compact_active_archetypes_kernel, ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids, connector_active_count, max_entities);
#else
    (void)ecs; (void)active_ids; (void)active_count; (void)lane_active_ids; (void)lane_active_count; (void)connector_active_ids; (void)connector_active_count; (void)max_entities; (void)threads; (void)active_blocks; (void)workers;
#endif
}

}  // namespace


#ifndef AVABM_CPU_PARALLEL_STATS_ENABLED
#define AVABM_CPU_PARALLEL_STATS_ENABLED 1
#endif

#if AVABM_CPU_PARALLEL_STATS_ENABLED
void stats_system_cpu_parallel(ECSArrays ecs,
                               RoadNetwork road,
                               PerceptionSoA perception,
                               DecisionSoA decision,
                               float* metrics,
                               float dt,
                               int max_entities,
                               const int* active_ids,
                               const int* active_count,
                               int workers) {
    if (metrics == nullptr || max_entities <= 0) return;
    constexpr int ST_ACTIVE = 0;
    constexpr int ST_SPEED_SUM = 1;
    constexpr int ST_SPEED_COUNT = 2;
    constexpr int ST_SLOW_COUNT = 3;
    constexpr int ST_STOP_COUNT = 4;
    constexpr int ST_STANDSTILL_TIME = 5;
    constexpr int ST_ACCEL_COUNT = 6;
    constexpr int ST_ACCEL_SUM = 7;
    constexpr int ST_ACCEL_SQ_SUM = 8;
    constexpr int ST_DECEL_COUNT = 9;
    constexpr int ST_DECEL_SUM = 10;
    constexpr int ST_DECEL_SQ_SUM = 11;
    constexpr int ST_HARD_BRAKE = 12;
    constexpr int ST_MIN_GAP_SUM = 13;
    constexpr int ST_MIN_GAP_COUNT = 14;
    constexpr int ST_HEADWAY_SUM = 15;
    constexpr int ST_HEADWAY_COUNT = 16;
    constexpr int ST_TIME_LOSS_SUM = 17;
    constexpr int ST_TIME_LOSS_COUNT = 18;
    constexpr int ST_QUEUE_DELAY_SUM = 19;
    constexpr int ST_QUEUE_DELAY_COUNT = 20;
    constexpr int STATS_LOCAL_SLOTS = 21;

    const int active_raw = (active_ids != nullptr && active_count != nullptr) ? *active_count : max_entities;
    const int n = std::max(0, std::min(active_raw, max_entities));
    if (n <= 0) return;
    const int T = cpu_resolve_workers(workers, n);
    std::vector<double> sums(static_cast<size_t>(T) * STATS_LOCAL_SLOTS, 0.0);
    cpu_parallel_ranges(static_cast<int64_t>(n), T, [&](int64_t begin, int64_t end, int tid) {
        double* local = sums.data() + static_cast<size_t>(tid) * STATS_LOCAL_SLOTS;
        for (int64_t pos = begin; pos < end; ++pos) {
            const int i = (active_ids != nullptr) ? active_ids[pos] : static_cast<int>(pos);
            if (i < 0 || i >= max_entities) continue;
            if (ecs.alive[i] != ENTITY_ALIVE) continue;
            float v = ecs.speed[i];
            float a = ecs.accel[i];
            if (!isfinite(v)) v = 0.0f;
            if (!isfinite(a)) a = 0.0f;
            local[ST_ACTIVE] += 1.0;
            local[ST_SPEED_SUM] += static_cast<double>(v);
            local[ST_SPEED_COUNT] += 1.0;
            if (v < 2.0f) local[ST_SLOW_COUNT] += 1.0;
            if (v < 0.2f) {
                local[ST_STOP_COUNT] += 1.0;
                local[ST_STANDSTILL_TIME] += static_cast<double>(dt);
            }
            if (a > 0.05f) {
                local[ST_ACCEL_COUNT] += 1.0;
                local[ST_ACCEL_SUM] += static_cast<double>(a);
                local[ST_ACCEL_SQ_SUM] += static_cast<double>(a) * static_cast<double>(a);
            } else if (a < -0.05f) {
                const float d = -a;
                local[ST_DECEL_COUNT] += 1.0;
                local[ST_DECEL_SUM] += static_cast<double>(d);
                local[ST_DECEL_SQ_SUM] += static_cast<double>(d) * static_cast<double>(d);
            }
            if (a < -3.5f) local[ST_HARD_BRAKE] += 1.0;
            const float fg = perception.front_gap != nullptr ? perception.front_gap[i] : 1.0e9f;
            if (fg < 1.0e8f) {
                local[ST_MIN_GAP_SUM] += static_cast<double>(fg);
                local[ST_MIN_GAP_COUNT] += 1.0;
                if (v > 0.5f) {
                    local[ST_HEADWAY_SUM] += static_cast<double>(fg / fmaxf(v, 0.5f));
                    local[ST_HEADWAY_COUNT] += 1.0;
                }
            }
            float desired = decision.desired_speed != nullptr ? decision.desired_speed[i] : 0.0f;
            if (!isfinite(desired) || desired < 0.1f) {
                const int lane = ecs.lane_id[i];
                if (lane >= 0 && lane < road.num_lanes) {
                    desired = desired_speed_ecs(i, lane, ecs, road);
                } else {
                    desired = MAX_SPEED_FALLBACK;
                }
            }
            if (desired > 0.5f) {
                const float loss_ratio = clampf_cuda((desired - v) / desired, 0.0f, 1.0f);
                local[ST_TIME_LOSS_SUM] += static_cast<double>(loss_ratio) * static_cast<double>(dt);
                local[ST_TIME_LOSS_COUNT] += 1.0;
                if (loss_ratio > 0.65f && v < 1.0f) {
                    local[ST_QUEUE_DELAY_SUM] += static_cast<double>(dt);
                    local[ST_QUEUE_DELAY_COUNT] += 1.0;
                }
            }
        }
    });
    double total[STATS_LOCAL_SLOTS]{};
    for (int tid = 0; tid < T; ++tid) {
        const double* local = sums.data() + static_cast<size_t>(tid) * STATS_LOCAL_SLOTS;
        for (int k = 0; k < STATS_LOCAL_SLOTS; ++k) total[k] += local[k];
    }
    auto add_metric = [&](int metric_idx, double value) {
        if (value != 0.0 && avabm_metric_enabled_ecs(metric_idx)) {
            metrics[metric_idx] += static_cast<float>(value);
        }
    };
    add_metric(METRIC_ACTIVE, total[ST_ACTIVE]);
    add_metric(METRIC_SPEED_SUM, total[ST_SPEED_SUM]);
    add_metric(METRIC_SPEED_COUNT, total[ST_SPEED_COUNT]);
    add_metric(METRIC_SLOW_COUNT, total[ST_SLOW_COUNT]);
    add_metric(METRIC_STOP_COUNT, total[ST_STOP_COUNT]);
    add_metric(METRIC_STANDSTILL_TIME, total[ST_STANDSTILL_TIME]);
    add_metric(METRIC_ACCEL_COUNT, total[ST_ACCEL_COUNT]);
    add_metric(METRIC_ACCEL_SUM, total[ST_ACCEL_SUM]);
    add_metric(METRIC_ACCEL_SQ_SUM, total[ST_ACCEL_SQ_SUM]);
    add_metric(METRIC_DECEL_COUNT, total[ST_DECEL_COUNT]);
    add_metric(METRIC_DECEL_SUM, total[ST_DECEL_SUM]);
    add_metric(METRIC_DECEL_SQ_SUM, total[ST_DECEL_SQ_SUM]);
    add_metric(METRIC_HARD_BRAKE, total[ST_HARD_BRAKE]);
    add_metric(METRIC_MIN_GAP_SUM, total[ST_MIN_GAP_SUM]);
    add_metric(METRIC_MIN_GAP_COUNT, total[ST_MIN_GAP_COUNT]);
    add_metric(METRIC_HEADWAY_SUM, total[ST_HEADWAY_SUM]);
    add_metric(METRIC_HEADWAY_COUNT, total[ST_HEADWAY_COUNT]);
    add_metric(METRIC_TIME_LOSS_SUM, total[ST_TIME_LOSS_SUM]);
    add_metric(METRIC_TIME_LOSS_COUNT, total[ST_TIME_LOSS_COUNT]);
    add_metric(METRIC_QUEUE_DELAY_SUM, total[ST_QUEUE_DELAY_SUM]);
    add_metric(METRIC_QUEUE_DELAY_COUNT, total[ST_QUEUE_DELAY_COUNT]);
}
#endif

extern "C" void launch_step_cpu_ecs(
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    SpatialGrid grid,
    SpawnConfig spawn,
    PerceptionSoA perception,
    DecisionSoA decision,
    int* reservation_table,
    uint32_t* rng_state,
    float* metrics,
    int* active_ids,
    int* active_count,
    int* lane_active_ids,
    int* lane_active_count,
    int* connector_active_ids,
    int* connector_active_count,
    int initial_grid_valid,
    float current_time,
    float dt,
    int max_entities,
    int step_index,
    int cpu_workers) {
    if (max_entities <= 0 || road.num_lanes <= 0) return;
    if (grid.cell_size <= 0.1f || grid.width <= 0 || grid.height <= 0) return;
    if (dt <= 0.0f || dt > 0.5f) return;
    const int threads = 256;
    const int entity_blocks_raw = (max_entities + threads - 1) / threads;
    const int entity_blocks = entity_blocks_raw > 0 ? entity_blocks_raw : 1;
    int active_blocks = entity_blocks < AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS ? entity_blocks : AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS;
    if (active_blocks < 1) active_blocks = 1;
    const int64_t world_cells64 = static_cast<int64_t>(grid.width) * static_cast<int64_t>(grid.height);
    if (world_cells64 <= 0 || world_cells64 > static_cast<int64_t>(std::numeric_limits<int>::max())) return;
    const int world_cells = static_cast<int>(world_cells64);
    int spawn_blocks = (spawn.num_spawn_points + threads - 1) / threads;
    if (spawn_blocks < 1) spawn_blocks = 1;

    const bool fast_physics = (AVABM_FAST_PHYSICS_MODE != 0);
    const bool do_route_repair = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_ROUTE_REPAIR_INTERVAL);
    const bool do_spawn_overlap = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_SPAWN_OVERLAP_INTERVAL);
    const bool do_priority = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_PRIORITY_INTERVAL);
    const bool do_local_avoidance = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_LOCAL_AVOIDANCE_INTERVAL);
    const bool do_front_clear = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_FRONT_CLEAR_INTERVAL);
    const bool do_contact = (!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_CONTACT_INTERVAL);
    const bool expensive_metrics_enabled = (AVABM_EXPENSIVE_SAFETY_METRICS_ENABLED != 0);
#if AVABM_METRICS_MODE == 0
    const bool any_metrics_enabled = false;
#else
    const bool any_metrics_enabled = true;
#endif
    const bool do_collision = expensive_metrics_enabled && any_metrics_enabled && ((!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_COLLISION_INTERVAL));
    const bool do_safety_metrics = expensive_metrics_enabled && any_metrics_enabled && ((!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_SAFETY_METRICS_INTERVAL));
    const bool do_stats = any_metrics_enabled && ((!fast_physics) || avabm_due_interval_ecs(step_index, AVABM_STATS_INTERVAL));

    // CPU temp buffers are allocated by the Python binding per call, so rebuild the
    // active list every substep. This preserves model semantics while skipping only
    // CUDA's persistent-list performance optimization.
    (void)initial_grid_valid;
    const bool do_full_active_compact = true;

    if (any_metrics_enabled && metrics != nullptr) {
        clear_float_cpu(metrics + 6, METRICS_SIZE - 6, 0.0f, cpu_workers);
    }
    if (do_full_active_compact) {
        rebuild_active_list_cpu(ecs, active_ids, active_count, max_entities, threads, entity_blocks, cpu_workers);
    }

    const int total_slots = (reservation_table != nullptr && road.num_nodes > 0) ? road.num_nodes * RES_HORIZON_SLOTS : 0;
    if (total_slots > 0) {
        clear_int_cpu(reservation_table, total_slots, RESERVATION_FREE, cpu_workers);
    }

    // Always rebuild the CPU world grid at the start of the tick. This is equivalent
    // to CUDA's non-persistent path and avoids stale host-side temp-buffer state.
    begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (spawn.num_spawn_points > 0) {
        launch_cpu_kernel(spawn_blocks, threads, cpu_workers, spawn_system_kernel, ecs, road, grid, spawn, rng_state, metrics, reservation_table, reservation_table != nullptr ? total_slots : 0, current_time, dt, max_entities, step_index, active_ids, active_count);
    }
    if (total_slots > 0) {
        clear_int_cpu(reservation_table, total_slots, RESERVATION_FREE, cpu_workers);
    }

    rebuild_active_archetypes_cpu(ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids, connector_active_count, max_entities, threads, active_blocks, cpu_workers);

    if (do_spawn_overlap) {
        begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, resolve_spawn_overlap_system_kernel, ecs, road, grid, spawn, metrics, current_time, dt, max_entities, active_ids, active_count);
    }

    begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (do_route_repair) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, route_lane_repair_system_kernel, ecs, road, metrics, dt, max_entities, active_ids, active_count);
        begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);
    }

#if AVABM_LANE_HASH_GRID_ENABLED
    begin_lane_grid_rebuild_cpu(grid, road.num_lanes, cpu_workers);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, lane_hash_build_system, ecs, road, grid, max_entities, lane_active_ids, lane_active_count);
#endif

    launch_cpu_kernel(active_blocks, threads, cpu_workers, turn_signal_system_kernel, ecs, decision, road, metrics, dt, max_entities, active_ids, active_count);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, perception_system_kernel, ecs, road, grid, perception, metrics, max_entities, active_ids, active_count);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, decision_system_kernel, ecs, road, signals, grid, perception, decision, reservation_table, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);

    if (do_priority && reservation_table != nullptr && road.num_nodes > 0) {
        const int node_blocks = (road.num_nodes + threads - 1) / threads;
        launch_cpu_kernel(node_blocks, threads, cpu_workers, clear_intersection_priority_gate_kernel, reservation_table, road.num_nodes);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, mark_intersection_occupancy_kernel, ecs, road, reservation_table, max_entities, connector_active_ids, connector_active_count);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, select_intersection_priority_candidates_kernel, ecs, road, signals, decision, perception, reservation_table, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, apply_intersection_priority_gate_kernel, ecs, road, signals, decision, grid, perception, reservation_table, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);
    }

    if (do_local_avoidance) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, local_obstacle_avoidance_system_kernel, ecs, road, grid, perception, decision, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);
    }
    if (do_front_clear) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, front_clear_must_go_system_kernel, ecs, road, signals, grid, perception, decision, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);
    }

    launch_cpu_kernel(active_blocks, threads, cpu_workers, lane_change_system_kernel, ecs, decision, road, dt, max_entities, lane_active_ids, lane_active_count);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, motion_system_kernel, ecs, decision, road, perception, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, connector_enter_system_kernel, ecs, decision, road, grid, metrics, current_time, dt, max_entities, lane_active_ids, lane_active_count);

    rebuild_active_archetypes_cpu(ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids, connector_active_count, max_entities, threads, active_blocks, cpu_workers);
    begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    launch_cpu_kernel(active_blocks, threads, cpu_workers, connector_motion_system_kernel, ecs, road, grid, metrics, current_time, dt, max_entities, connector_active_ids, connector_active_count);
    begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (do_contact) {
        for (int contact_pass = 0; contact_pass < CONTACT_RESOLVE_PASSES; ++contact_pass) {
            launch_cpu_kernel(active_blocks, threads, cpu_workers, contact_resolve_system_kernel, ecs, road, grid, metrics, current_time, dt, max_entities, active_ids, active_count);
            begin_world_grid_rebuild_cpu(grid, world_cells, cpu_workers);
            launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);
        }
    }
    if (do_collision) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, collision_system_kernel, ecs, grid, metrics, dt, max_entities, active_ids, active_count);
    }
    if (do_safety_metrics) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, safety_metrics_system_kernel, ecs, grid, metrics, max_entities, active_ids, active_count);
    }
    if (do_stats) {
#if AVABM_CPU_PARALLEL_STATS_ENABLED
        stats_system_cpu_parallel(ecs, road, perception, decision, metrics, dt, max_entities, active_ids, active_count, cpu_workers);
#else
        // Fallback parity path: the CUDA stats kernel uses __shared__ block reduction,
        // so one logical CPU thread keeps the same source body correct.
        launch_cpu_kernel(1, 1, 1, stats_system_kernel, ecs, road, perception, decision, metrics, dt, max_entities, active_ids, active_count);
#endif
    }
}
