#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

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
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    *addr = old + val;
    return old;
}

inline unsigned int atomicAdd(unsigned int* addr, unsigned int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    *addr = old + val;
    return old;
}

inline float atomicAdd(float* addr, float val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    float old = *addr;
    *addr = old + val;
    return old;
}

inline int atomicMin(int* addr, int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (val < old) *addr = val;
    return old;
}

inline int atomicMax(int* addr, int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (val > old) *addr = val;
    return old;
}

inline int atomicExch(int* addr, int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    *addr = val;
    return old;
}

inline unsigned int atomicExch(unsigned int* addr, unsigned int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    *addr = val;
    return old;
}

inline int atomicCAS(int* addr, int compare, int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (old == compare) *addr = val;
    return old;
}

inline unsigned int atomicCAS(unsigned int* addr, unsigned int compare, unsigned int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    unsigned int old = *addr;
    if (old == compare) *addr = val;
    return old;
}

#define AVABM_PART_SKIP_COMMON 1
#include "main_common_cpu.hpp"
#include "../avabm_cuda/main_core_grid_spawn.cu"
#include "../avabm_cuda/main_core_decision.cu"
#include "../avabm_cuda/main_core_motion.cu"

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

inline void clear_int_cpu(int* data, int n, int value) {
    if (data == nullptr || n <= 0) return;
    std::fill(data, data + n, value);
}

inline void clear_float_cpu(float* data, int n, float value) {
    if (data == nullptr || n <= 0) return;
    std::fill(data, data + n, value);
}

inline void begin_world_grid_rebuild_cpu(SpatialGrid& grid, int world_cells) {
    if (grid.cell_head == nullptr || world_cells <= 0) return;
    std::fill(grid.cell_head, grid.cell_head + world_cells, WORLD_CELL_EMPTY);
    if (grid.cell_epoch != nullptr) {
        std::fill(grid.cell_epoch, grid.cell_epoch + world_cells, 0);
    }
}

inline void begin_lane_grid_rebuild_cpu(SpatialGrid& grid, int num_lanes) {
    if (grid.lane_cell_head == nullptr || num_lanes <= 0 || grid.lane_cells_per_lane <= 0) return;
    std::fill(grid.lane_cell_head, grid.lane_cell_head + num_lanes * grid.lane_cells_per_lane, WORLD_CELL_EMPTY);
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
        clear_float_cpu(metrics + 6, METRICS_SIZE - 6, 0.0f);
    }
    if (do_full_active_compact) {
        rebuild_active_list_cpu(ecs, active_ids, active_count, max_entities, threads, entity_blocks, cpu_workers);
    }

    const int total_slots = (reservation_table != nullptr && road.num_nodes > 0) ? road.num_nodes * RES_HORIZON_SLOTS : 0;
    if (total_slots > 0) {
        clear_int_cpu(reservation_table, total_slots, RESERVATION_FREE);
    }

    // Always rebuild the CPU world grid at the start of the tick. This is equivalent
    // to CUDA's non-persistent path and avoids stale host-side temp-buffer state.
    begin_world_grid_rebuild_cpu(grid, world_cells);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (spawn.num_spawn_points > 0) {
        launch_cpu_kernel(spawn_blocks, threads, cpu_workers, spawn_system_kernel, ecs, road, grid, spawn, rng_state, metrics, reservation_table, reservation_table != nullptr ? total_slots : 0, current_time, dt, max_entities, step_index, active_ids, active_count);
    }
    if (total_slots > 0) {
        clear_int_cpu(reservation_table, total_slots, RESERVATION_FREE);
    }

    rebuild_active_archetypes_cpu(ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids, connector_active_count, max_entities, threads, active_blocks, cpu_workers);

    if (do_spawn_overlap) {
        begin_world_grid_rebuild_cpu(grid, world_cells);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, resolve_spawn_overlap_system_kernel, ecs, road, grid, spawn, metrics, current_time, dt, max_entities, active_ids, active_count);
    }

    begin_world_grid_rebuild_cpu(grid, world_cells);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (do_route_repair) {
        launch_cpu_kernel(active_blocks, threads, cpu_workers, route_lane_repair_system_kernel, ecs, road, metrics, dt, max_entities, active_ids, active_count);
        begin_world_grid_rebuild_cpu(grid, world_cells);
        launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);
    }

#if AVABM_LANE_HASH_GRID_ENABLED
    begin_lane_grid_rebuild_cpu(grid, road.num_lanes);
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
    begin_world_grid_rebuild_cpu(grid, world_cells);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    launch_cpu_kernel(active_blocks, threads, cpu_workers, connector_motion_system_kernel, ecs, road, grid, metrics, current_time, dt, max_entities, connector_active_ids, connector_active_count);
    begin_world_grid_rebuild_cpu(grid, world_cells);
    launch_cpu_kernel(active_blocks, threads, cpu_workers, spatial_hash_build_system, ecs, grid, max_entities, active_ids, active_count);

    if (do_contact) {
        for (int contact_pass = 0; contact_pass < CONTACT_RESOLVE_PASSES; ++contact_pass) {
            launch_cpu_kernel(active_blocks, threads, cpu_workers, contact_resolve_system_kernel, ecs, road, grid, metrics, current_time, dt, max_entities, active_ids, active_count);
            begin_world_grid_rebuild_cpu(grid, world_cells);
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
        // The CUDA stats kernel uses __shared__ block reduction. Running one logical
        // thread keeps the same source body correct on CPU; all other systems above
        // remain parallelized through launch_cpu_kernel.
        launch_cpu_kernel(1, 1, 1, stats_system_kernel, ecs, road, perception, decision, metrics, dt, max_entities, active_ids, active_count);
    }
}
