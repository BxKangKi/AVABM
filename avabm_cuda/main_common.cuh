#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#pragma warning(disable: 4819)
#endif
#include <windows.h>
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#ifdef _WIN32
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif
#include <stdint.h>
#include <math.h>
#define HUMAN 0
#define AV    1
#define LIGHT_RED    0
#define LIGHT_YELLOW 1
#define LIGHT_GREEN  2
#define TURN_LEFT     -1
#define TURN_STRAIGHT  0
#define TURN_RIGHT     1
#define TURN_ANY       99
#define INDICATOR_NONE    0
#define INDICATOR_LEFT   -1
#define INDICATOR_RIGHT   1
#define INDICATOR_HAZARD  2
#define VEH_ON_LANE       0
#define VEH_IN_CONNECTOR  1
#define ENTITY_FREE      0
#define ENTITY_ALIVE     1
#define ENTITY_SPAWNING  2
#define SPAWN_ACCUMULATOR_MAX 10000.0f
#define SPAWN_MAX_PER_POINT_PER_STEP 8
#define WORLD_CELL_EMPTY -1
#define DEFAULT_LANE_WIDTH 3.5f
#define LANE_SURFACE_MARGIN 0.6f
#define MAX_SPEED_FALLBACK 13.9f
#ifndef AVABM_MIN_CRUISE_SPEED_ENABLED
#define AVABM_MIN_CRUISE_SPEED_ENABLED 1
#endif
#ifndef AVABM_DEVICE_FORCEINLINE
#define AVABM_DEVICE_FORCEINLINE 0
#endif
#ifndef AVABM_DEVICE_NOINLINE
#define AVABM_DEVICE_NOINLINE 0
#endif
#if AVABM_DEVICE_FORCEINLINE
#define AVABM_DINLINE static __device__ __forceinline__
#elif AVABM_DEVICE_NOINLINE
#define AVABM_DINLINE static __device__ __noinline__
#else
#define AVABM_DINLINE static __device__ inline
#endif
#ifndef AVABM_HDINLINE
#define AVABM_HDINLINE static __host__ __device__ inline
#endif
#ifndef AVABM_MIN_CRUISE_SPEED_KMH
#define AVABM_MIN_CRUISE_SPEED_KMH 40.0f
#endif
#ifndef AVABM_FAST_EQUIV_MATH
#define AVABM_FAST_EQUIV_MATH 1
#endif
#ifndef AVABM_USE_ASYNC_MEMSET_CLEAR
#define AVABM_USE_ASYNC_MEMSET_CLEAR 1
#endif
#ifndef AVABM_SPAWN_GRID_INSERT_FASTPATH
#define AVABM_SPAWN_GRID_INSERT_FASTPATH 1
#endif
#ifndef AVABM_MIN_CRUISE_HARD_FREEFLOW
#define AVABM_MIN_CRUISE_HARD_FREEFLOW 1
#endif
#define AVABM_KMH_TO_MPS 0.2777777778f
#ifndef AVABM_SPAWN_ALLOC_SCAN_LIMIT
#define AVABM_SPAWN_ALLOC_SCAN_LIMIT 4096
#endif
#ifndef AVABM_ECS_CURSOR_ALLOCATOR
#define AVABM_ECS_CURSOR_ALLOCATOR 1
#endif
#ifndef AVABM_INCREMENTAL_ACTIVE_LIST
#define AVABM_INCREMENTAL_ACTIVE_LIST 1
#endif
#ifndef AVABM_ACTIVE_FULL_COMPACT_INTERVAL
#define AVABM_ACTIVE_FULL_COMPACT_INTERVAL 8
#endif
AVABM_DINLINE int avabm_warp_append_slot(bool predicate, int* counter) {
    if (counter == nullptr) return -1;
    unsigned int full_mask = __activemask();
    unsigned int mask = __ballot_sync(full_mask, predicate);
    if (mask == 0u || !predicate) return -1;
    unsigned int lane = (unsigned int)(threadIdx.x & 31);
    unsigned int leader = (unsigned int)(__ffs((int)mask) - 1);
    int base = 0;
    if (lane == leader) {
        base = atomicAdd(counter, __popc(mask));
    }
    base = __shfl_sync(mask, base, (int)leader);
    unsigned int prior_mask = (lane == 0u) ? 0u : ((1u << lane) - 1u);
    int rank = __popc(mask & prior_mask);
    return base + rank;
}
#ifndef AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS
#define AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS 128
#endif
#ifndef AVABM_ACTIVE_LIST_ENABLED
#define AVABM_ACTIVE_LIST_ENABLED 1
#endif
#ifndef AVABM_LAZY_GRID_ENABLED
#define AVABM_LAZY_GRID_ENABLED 1
#endif
#ifndef AVABM_LANE_HASH_GRID_ENABLED
#define AVABM_LANE_HASH_GRID_ENABLED 0
#endif
#ifndef AVABM_PERSISTENT_START_GRID
#define AVABM_PERSISTENT_START_GRID 1
#endif
#ifndef AVABM_ECS_PERSISTENT_PERCEPTION_GRID_REUSE
#define AVABM_ECS_PERSISTENT_PERCEPTION_GRID_REUSE 1
#endif
#ifndef AVABM_ROUTE_REPAIR_INTERVAL
#define AVABM_ROUTE_REPAIR_INTERVAL 1
#endif
#ifndef AVABM_EXPENSIVE_SAFETY_METRICS_ENABLED
#define AVABM_EXPENSIVE_SAFETY_METRICS_ENABLED 1
#endif
#ifndef AVABM_EXPENSIVE_SAFETY_METRICS_INTERVAL
#define AVABM_EXPENSIVE_SAFETY_METRICS_INTERVAL 1
#endif
#ifndef AVABM_FAST_PHYSICS_MODE
#define AVABM_FAST_PHYSICS_MODE 0
#endif
#ifndef AVABM_SKIP_PRESPAWN_ACTIVE_REBUILD
#define AVABM_SKIP_PRESPAWN_ACTIVE_REBUILD 0
#endif
#ifndef AVABM_SPAWN_OVERLAP_INTERVAL
#define AVABM_SPAWN_OVERLAP_INTERVAL 1
#endif
#ifndef AVABM_LOCAL_AVOIDANCE_INTERVAL
#define AVABM_LOCAL_AVOIDANCE_INTERVAL 1
#endif
#ifndef AVABM_FRONT_CLEAR_INTERVAL
#define AVABM_FRONT_CLEAR_INTERVAL 1
#endif
#ifndef AVABM_CONTACT_INTERVAL
#define AVABM_CONTACT_INTERVAL 1
#endif
#ifndef AVABM_COLLISION_INTERVAL
#define AVABM_COLLISION_INTERVAL 1
#endif
#ifndef AVABM_SAFETY_METRICS_INTERVAL
#define AVABM_SAFETY_METRICS_INTERVAL AVABM_EXPENSIVE_SAFETY_METRICS_INTERVAL
#endif
#ifndef AVABM_STATS_INTERVAL
#define AVABM_STATS_INTERVAL 1
#endif
#ifndef AVABM_PRIORITY_INTERVAL
#define AVABM_PRIORITY_INTERVAL 1
#endif
#ifndef AVABM_LANE_BUCKET_SCAN_LIMIT
#define AVABM_LANE_BUCKET_SCAN_LIMIT 0
#endif
#ifndef AVABM_WORLD_BUCKET_SCAN_LIMIT
#define AVABM_WORLD_BUCKET_SCAN_LIMIT 0
#endif
AVABM_DINLINE int avabm_lane_scan_limit_ecs(int max_entities) {
#if AVABM_LANE_BUCKET_SCAN_LIMIT > 0
    return max_entities < AVABM_LANE_BUCKET_SCAN_LIMIT ? max_entities : AVABM_LANE_BUCKET_SCAN_LIMIT;
#else
    return max_entities;
#endif
}
AVABM_DINLINE int avabm_world_scan_limit_ecs(int max_entities) {
#if AVABM_WORLD_BUCKET_SCAN_LIMIT > 0
    return max_entities < AVABM_WORLD_BUCKET_SCAN_LIMIT ? max_entities : AVABM_WORLD_BUCKET_SCAN_LIMIT;
#else
    return max_entities;
#endif
}
AVABM_HDINLINE bool avabm_due_interval_ecs(int step_index, int interval) {
    return interval <= 1 || step_index <= 0 || (step_index % interval) == 0;
}
#if AVABM_ACTIVE_LIST_ENABLED
#define AVABM_ACTIVE_LOOP_BEGIN(MAX_N, ACTIVE_IDS, ACTIVE_COUNT_PTR) \
int avabm_tid = blockIdx.x * blockDim.x + threadIdx.x; \
int avabm_stride = blockDim.x * gridDim.x; \
int avabm_active_n_raw = ((ACTIVE_IDS) != nullptr && (ACTIVE_COUNT_PTR) != nullptr) ? *(ACTIVE_COUNT_PTR) : (MAX_N); \
int avabm_active_n = avabm_active_n_raw < (MAX_N) ? avabm_active_n_raw : (MAX_N); \
if (avabm_active_n < 0) avabm_active_n = 0; \
for (int avabm_pos = avabm_tid; avabm_pos < avabm_active_n; avabm_pos += avabm_stride) { \
int i = ((ACTIVE_IDS) != nullptr) ? (ACTIVE_IDS)[avabm_pos] : avabm_pos; \
if (i < 0 || i >= (MAX_N)) continue;
#else
#define AVABM_ACTIVE_LOOP_BEGIN(MAX_N, ACTIVE_IDS, ACTIVE_COUNT_PTR) \
(void)(ACTIVE_IDS); \
(void)(ACTIVE_COUNT_PTR); \
int avabm_tid = blockIdx.x * blockDim.x + threadIdx.x; \
int avabm_stride = blockDim.x * gridDim.x; \
for (int i = avabm_tid; i < (MAX_N); i += avabm_stride) {
#endif
#define AVABM_ACTIVE_LOOP_END() }
#define MAX_ACCEL_AV     2.8f
#define MAX_ACCEL_HUMAN  2.0f
#define MAX_DECEL_AV     4.0f
#define MAX_DECEL_HUMAN  3.4f
#define EMERGENCY_DECEL  7.0f
#define SAFE_TIME_HEADWAY_AV     0.9f
#define SAFE_TIME_HEADWAY_HUMAN  1.75f
#define SAFE_GAP_AV              3.0f
#define SAFE_GAP_HUMAN           6.8f
#define LANE_CHANGE_FRONT_GAP_AV     12.0f
#define LANE_CHANGE_REAR_GAP_AV      10.0f
#define LANE_CHANGE_FRONT_GAP_HUMAN  24.0f
#define LANE_CHANGE_REAR_GAP_HUMAN   27.0f
#define LANE_CHANGE_DURATION_AV    2.5f
#define LANE_CHANGE_DURATION_HUMAN 4.8f
#define LANE_CHANGE_MIN_DURATION_AV    0.65f
#define LANE_CHANGE_MIN_DURATION_HUMAN 0.85f
#define LC_COOLDOWN_AV    2.0f
#define LC_COOLDOWN_HUMAN 6.5f
#define CONNECTOR_DEFAULT_LEN 18.0f
#define CONNECTOR_MIN_LEN      6.0f
#define CONNECTOR_MAX_LEN     60.0f
#define CONNECTOR_EXIT_EPS     0.15f
#define CONNECTOR_ENTER_EPS    0.15f
#define CONNECTOR_SAME_NODE_EPS 0.35f
#define CONNECTOR_EXIT_OFFSET_MIN 3.0f
#define CONNECTOR_EXIT_OFFSET_BASE 6.0f
#define CONNECTOR_EXIT_OFFSET_MAX 18.0f
#define CONNECTOR_TRIGGER_MARGIN 0.75f
#define SURFACE_TURN_MIN_RADIUS      6.8f
#define SURFACE_TURN_MAX_RADIUS     24.0f
#define SURFACE_TURN_ARC_FIT_TOL     2.4f
#define SURFACE_TURN_MIN_DELTA_RAD   0.18f
#define SURFACE_TURN_MAX_SWEEP_RAD   2.05f
#define SURFACE_TURN_CONFLICT_RADIUS 4.8f
#define SURFACE_TURN_SAMPLE_COUNT    5
#define SURFACE_TURN_HANDLE_MIN      1.4f
#define SURFACE_TURN_HANDLE_MAX      8.0f
#define SURFACE_TURN_TANGENT_BLEND   0.82f
#define CONNECTOR_SPEED_AV     7.2f
#define CONNECTOR_SPEED_HUMAN  4.9f
#define DEFAULT_STOP_OFFSET 3.0f
#define MIN_BUMPER_GAP 2.5f
#define NO_BACKWARD_EPS 0.02f
#define TURN_CONTROL_DIST_AV     75.0f
#define TURN_CONTROL_DIST_HUMAN 115.0f
#define TURN_SPEED_STRAIGHT_AV   22.0f
#define TURN_SPEED_STRAIGHT_HUMAN 14.5f
#define TURN_SPEED_HARD_AV        5.5f
#define TURN_SPEED_HARD_HUMAN     3.7f
#define TURN_SPEED_UTURN_AV       3.6f
#define TURN_SPEED_UTURN_HUMAN    2.4f
#ifndef WORLD_MAX_CELL_RADIUS
#define WORLD_MAX_CELL_RADIUS 5
#endif
#ifndef CONTACT_CELL_RADIUS
#define CONTACT_CELL_RADIUS 3
#endif
#ifndef COLLISION_CELL_RADIUS
#define COLLISION_CELL_RADIUS 3
#endif
#ifndef AVABM_TURBO_LOCAL_AVOID_CULL
#define AVABM_TURBO_LOCAL_AVOID_CULL 0
#endif
#ifndef AVABM_TURBO_FAST_NEIGHBOR_SENSORS
#define AVABM_TURBO_FAST_NEIGHBOR_SENSORS 1
#endif
#ifndef LOCAL_AVOID_CULL_APPROACH_DIST
#define LOCAL_AVOID_CULL_APPROACH_DIST 65.0f
#endif
#ifndef LOCAL_AVOID_CULL_FRONT_GAP
#define LOCAL_AVOID_CULL_FRONT_GAP 90.0f
#endif
#ifndef AVABM_FAST_PIPELINE_MODE
#define AVABM_FAST_PIPELINE_MODE 0
#endif
#ifndef AVABM_SPAWN_OVERLAP_REPAIR_ENABLED
#define AVABM_SPAWN_OVERLAP_REPAIR_ENABLED 1
#endif
#ifndef AVABM_ROUTE_REPAIR_GRID_REBUILD_ENABLED
#define AVABM_ROUTE_REPAIR_GRID_REBUILD_ENABLED 1
#endif
#ifndef AVABM_SAFETY_METRICS_EVERY_N
#define AVABM_SAFETY_METRICS_EVERY_N 1
#endif
#ifndef AVABM_STATS_EVERY_N
#define AVABM_STATS_EVERY_N 1
#endif
#define RES_SLOT_DT        0.5f
#define RES_HORIZON_SLOTS 16
#define RESERVATION_FREE  -1
#define TTC_CRITICAL 1.5f
#define TTC_WARNING  3.0f
#define MOBIL_THRESHOLD_AV       0.08f
#define MOBIL_THRESHOLD_HUMAN    0.28f
#define MOBIL_MIN_ADVANTAGE      0.05f
#define SENSOR_FRONT_RANGE_AV       155.0f
#define SENSOR_FRONT_RANGE_HUMAN    135.0f
#define SENSOR_FRONT_HALF_FOV_AV      0.74f
#define SENSOR_FRONT_HALF_FOV_HUMAN   0.62f
#define SENSOR_SIDE_RANGE_AV         68.0f
#define SENSOR_SIDE_RANGE_HUMAN      58.0f
#define SENSOR_SIDE_HALF_FOV          0.82f
#define SENSOR_CONE_EDGE_MARGIN       1.35f
#define SENSOR_RAY_WIDTH              1.15f
#define KIN_WHEELBASE_FACTOR          0.58f
#define KIN_MIN_WHEELBASE             2.20f
#define KIN_MAX_WHEELBASE             3.35f
#define KIN_MIN_YAW_SPEED             0.18f
#define MAX_STEER_AV                  0.56f
#define MAX_STEER_HUMAN               0.48f
#define MAX_STEER_RATE_AV             0.95f
#define MAX_STEER_RATE_HUMAN          0.58f
#define MAX_YAW_RATE_AV               1.05f
#define MAX_YAW_RATE_HUMAN            0.78f
#define PATH_HEADING_LOCK_AV          0.30f
#define PATH_HEADING_LOCK_HUMAN       0.38f
#define SIGNAL_CREEP_SPEED_AV         0.85f
#define SIGNAL_CREEP_SPEED_HUMAN      0.65f
#define SIGNAL_CREEP_HOLD_DIST        2.15f
#define YIELD_CREEP_SPEED_AV          1.65f
#define YIELD_CREEP_SPEED_HUMAN       1.15f
#define STALL_RECOVERY_FRONT_GAP     15.0f
#define STALL_RECOVERY_MIN_END_DIST   4.0f
#define INTERACTION_RANGE_AV         46.0f
#define INTERACTION_RANGE_HUMAN      54.0f
#define INTERACTION_TTC_SOFT          2.35f
#define INTERACTION_TTC_HARD          1.05f
#define INTERSECTION_APPROACH_RANGE  38.0f
#define INTERSECTION_TIME_WINDOW      1.85f
#define INTERSECTION_PRIORITY_EPS     0.40f
#define INTERSECTION_STOP_BUFFER      1.25f
#define DIRECTIONAL_SAME_APPROACH_DOT  0.72f
#define DIRECTIONAL_ONCOMING_DOT      -0.55f
#define DIRECTIONAL_SIDE_DOT_ABS       0.42f
#define DIRECTIONAL_OTHER_STOP_EPS     0.18f
#define DIRECTIONAL_SIDE_RANGE_AV     42.0f
#define DIRECTIONAL_SIDE_RANGE_HUMAN  48.0f
#define DIRECTIONAL_ONCOMING_RANGE_AV 30.0f
#define DIRECTIONAL_ONCOMING_RANGE_HUMAN 34.0f
#define CONNECTOR_HEADING_LOCK_AV      0.18f
#define CONNECTOR_HEADING_LOCK_HUMAN   0.24f
#define TURN_LANE_PREP_BASE_DIST       180.0f
#define TURN_LANE_PREP_PER_LANE_DIST     95.0f
#define TURN_LANE_MIN_LC_DIST            70.0f
#define LANE_CHANGE_NO_START_DIST_TO_NODE 14.0f
#define LANE_CHANGE_FINISH_BEFORE_NODE     8.5f
#define TURN_LANE_HARD_HOLD_DIST       14.0f
#define TURN_LANE_STOP_BUFFER           8.0f
#define TURN_LANE_WRONG_LANE_SPEED_AV   6.5f
#define TURN_LANE_WRONG_LANE_SPEED_HUMAN 4.8f
#define UNSIGNAL_PRIORITY_APPROACH_RANGE  38.0f
#define UNSIGNAL_PRIORITY_NEAR_LINE_DIST   8.5f
#define UNSIGNAL_RIGHT_PRIORITY_CROSS      0.22f
#define UNSIGNAL_RIGHT_PRIORITY_WINDOW     1.55f
#define UNSIGNAL_ARRIVAL_EPS              0.35f
#define UNSIGNAL_STOPPED_EPS              0.22f
#define UNSIGNAL_STOPPED_FAR_IGNORE_DIST  10.0f
#define UNSIGNAL_RELEASE_FRONT_GAP         7.0f
#define DEADLOCK_PATIENCE_AV               0.55f
#define DEADLOCK_PATIENCE_HUMAN            0.80f
#define DEADLOCK_RELEASE_PERIOD            0.65f
#define DEADLOCK_RELEASE_CREEP_AV          4.60f
#define DEADLOCK_RELEASE_CREEP_HUMAN       3.55f
#define CONNECTOR_ENTRY_CLEAR_RADIUS      10.5f
#define CONNECTOR_ENTRY_PARALLEL_RADIUS    6.0f
#define INDICATOR_TURN_LOOKAHEAD_AV      115.0f
#define INDICATOR_TURN_LOOKAHEAD_HUMAN   135.0f
#define INDICATOR_LC_LOOKAHEAD_AV         95.0f
#define INDICATOR_LC_LOOKAHEAD_HUMAN     120.0f
#define INDICATOR_MIN_ON_TIME              0.35f
#define INDICATOR_TRUST_WINDOW             7.5f
#define INDICATOR_CONFLICT_EXTRA_TIME      1.20f
#define DEADLOCK_ESCAPE_PATIENCE_SCALE     0.24f
#define DEADLOCK_INDICATOR_PATIENCE_SCALE  0.20f
#define DEADLOCK_ESCAPE_FRONT_GAP          8.0f
#define CONNECTOR_CROSS_CONFLICT_RADIUS    5.4f
#define CONNECTOR_CROSS_TIME_WINDOW        1.15f
#define CONNECTOR_CROSS_STOP_BUFFER        2.8f
#define CONNECTOR_CROSS_SAMPLE_COUNT       5
#define ANTI_PASS_THROUGH_GAP              2.75f
#define CONTACT_RESOLVE_INFLATE            0.28f
#define CONTACT_RESOLVE_BACKOFF            1.65f
#define CONTACT_RESOLVE_MAX_PUSH          11.50f
#ifndef CONTACT_RESOLVE_PASSES
#define CONTACT_RESOLVE_PASSES             9
#endif
#define PRIORITY_GATE_APPROACH_RANGE       44.0f
#define PRIORITY_GATE_NEAR_LINE_DIST        8.0f
#define PRIORITY_GATE_STOP_BUFFER           2.3f
#define PRIORITY_GATE_RELEASE_SPEED_AV      5.8f
#define PRIORITY_GATE_RELEASE_SPEED_HUMAN   4.4f
#define PRIORITY_GATE_MAX_RELEASE_ACCEL     1.15f
#define PRIORITY_GATE_ID_BITS              20
#define PRIORITY_GATE_ID_MASK              ((1 << PRIORITY_GATE_ID_BITS) - 1)
#define PRIORITY_GATE_EMPTY                0x7fffffff
#define PRIORITY_GATE_SLOT_BEST             0
#define PRIORITY_GATE_SLOT_OCCUPIED         1
#define PRIORITY_GATE_SLOT_COUNT            2
#define PRIORITY_GATE_SLOT_GRANTED          3
#define PRIORITY_GATE_SLOT_STRIDE           RES_HORIZON_SLOTS
#define PRIORITY_GATE_PATH_SCAN_RANGE      48.0f
#define PRIORITY_GATE_EXIT_SPACE            8.5f
#define PRIORITY_GATE_ACTIVE_CLEAR_FRACTION 0.22f
#define PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST 12.0f
#define PRIORITY_GATE_BEHAVIOR_BIAS_SCALE   5.0f
#define HUMAN_AI_ASSERTIVE_BOOST_AV          0.85f
#define HUMAN_AI_ASSERTIVE_BOOST_HUMAN       1.65f
#define HUMAN_AI_COURTESY_HOLD_SCALE         0.30f
#define RIGHT_TURN_CORNER_MIN_HANDLE         1.0f
#define RIGHT_TURN_CORNER_MAX_HANDLE        12.0f
#define RIGHT_TURN_CORNER_MAX_PROJ_FRAC      0.82f
#define RIGHT_TURN_CORNER_TANGENT_BLEND      0.70f
#define RIGHT_TURN_MAX_HEADING_ERR           0.48f
#define RIGHT_TURN_CURB_BIAS_MIN             0.45f
#define RIGHT_TURN_CURB_BIAS_MAX             1.15f
#define RIGHT_TURN_CHORD_LATERAL_LIMIT       0.38f
#define RIGHT_TURN_FILLET_MIN_PARAM          0.55f
#define RIGHT_TURN_FILLET_MAX_PARAM         13.5f
#define RIGHT_TURN_FILLET_TANGENT_BLEND      0.88f
#define RIGHT_TURN_FILLET_MAX_HEADING_ERR    0.36f
#define FRONT_CLEAR_PRIORITY_MIN_GAP         16.0f
#define FRONT_CLEAR_PRIORITY_TIME             0.95f
#define FRONT_CLEAR_RELEASE_GAP_MULT          1.45f
#define FRONT_CLEAR_ASSERTIVE_WAIT_SCALE      0.55f
#define PRIORITY_GATE_FRONT_CLEAR_BONUS       80
#define PRIORITY_GATE_FRONT_BLOCK_PENALTY    260
#define PRIORITY_GATE_BLOCKED_OTHER_IGNORE_WAIT 0.35f
#define LANE_CHANGE_SIGNAL_LEAD_TIME          0.40f
#define LANE_CHANGE_PREP_SLOW_DIST           95.0f
#define LANE_CHANGE_PREP_HARD_DIST           34.0f
#define LANE_CHANGE_PREP_MIN_CAP_AV           9.4f
#define LANE_CHANGE_PREP_MIN_CAP_HUMAN        7.8f
#define LC_PREP_COAST_ACCEL_LIMIT             0.82f
#define LC_PREP_MAX_BRAKE_AV                  0.32f
#define LC_PREP_MAX_BRAKE_HUMAN               0.25f
#define LANE_CHANGE_COOP_ASSERTIVE_PROB       0.14f
#define LANE_CHANGE_COOP_RELAX_SCALE          0.68f
#define LANE_CHANGE_COOP_ZONE_MULT            1.85f
#define LANE_CHANGE_ACTIVE_SPEED_CAP_SCALE    0.98f
#define LC_INDICATOR_SIDE_RANGE              38.0f
#define LC_INDICATOR_SIDE_FRONT               9.0f
#define LC_INDICATOR_SIDE_REAR               30.0f
#define LC_COURTESY_DECEL_AV                  0.42f
#define LC_COURTESY_DECEL_HUMAN               0.32f
#define LC_ASSERTIVE_ACCEL_AV                 1.10f
#define LC_ASSERTIVE_ACCEL_HUMAN              1.45f
#define LC_ASSERTIVE_BLOCK_GAP               22.0f
#define LC_ASSERTIVE_BLOCK_TIME               1.20f
#define LOCAL_AVOID_RANGE                    24.0f
#define LOCAL_AVOID_HORIZON                   1.20f
#define LOCAL_AVOID_COLLISION_MARGIN          1.10f
#define LOCAL_AVOID_FRONT_CLEAR_BONUS        120
#define LOCAL_AVOID_CONNECTOR_BONUS          170
#define LOCAL_AVOID_INSIDE_BOX_BONUS         130
#define LOCAL_AVOID_STOP_BUFFER               2.0f
#define LOCAL_AVOID_IMMEDIATE_OVERLAP_INFLATE 0.08f
#define FRONT_CLEAR_MUST_GO_ENABLED             1
#define FRONT_CLEAR_MUST_GO_GAP                18.0f
#define FRONT_CLEAR_MUST_GO_TIME_GAP            0.85f
#define FRONT_CLEAR_MUST_GO_SPEED               1.35f
#define FRONT_CLEAR_MUST_GO_ACCEL_SCALE         0.72f
#define FRONT_CLEAR_MUST_GO_NODE_EXTRA          2.5f
#define FRONT_CLEAR_MUST_GO_RED_HOLD_DIST       2.2f
#define FRONT_CLEAR_MUST_GO_LC_ACCEL_SCALE      0.52f
#define FRONT_CLEAR_MUST_GO_LC_UNSAFE_ACCEL_SCALE 0.24f
#define LC_ACTIVE_UNSAFE_BRAKE                  0.22f
#define SMART_AI_VERSION                         30
#define SPAWN_RACE_REQUEUE_ENABLED              1
#define SPAWN_RACE_RECENT_WINDOW                0.30f
#define SPAWN_RACE_REQUEUE_FULLSCAN_MAX       4096
#define CONTACT_LONGITUDINAL_REPAIR_GAP         5.70f
#define CONTACT_ROUTE_REPAIR_EXTRA              2.85f
#define CONTACT_CROSS_BACKOFF_MIN               2.70f
#define CONTACT_CROSS_BACKOFF_SPEED_TIME        0.35f
#define CONNECTOR_PROTECTED_PROGRESS_FRAC       0.16f
#define CONNECTOR_PROTECTED_EXIT_FRAC           0.30f
#define CONNECTOR_INBOX_MIN_CLEAR_SPEED_AV      3.40f
#define CONNECTOR_INBOX_MIN_CLEAR_SPEED_HUMAN   2.65f
#define SMART_STALL_SPEED_EPS                   0.42f
#define SMART_STALL_FRONT_GAP                   18.0f
#define SMART_STALL_CLEAR_ACCEL_SCALE           0.62f
#define SMART_STALL_RELEASE_WAIT                0.08f
#define MISSED_TURN_ESCAPE_WAIT                 0.45f
#define MISSED_TURN_ESCAPE_DIST                 18.0f
#define MISSED_TURN_ESCAPE_SPEED_AV             2.80f
#define MISSED_TURN_ESCAPE_SPEED_HUMAN          2.10f
#define LOCAL_AVOID_GRANTED_CONNECTOR_GRACE     1
#define COMPLETE_OVERLAP_RELEASE_ENABLED        1
#define COMPLETE_OVERLAP_RELEASE_DIST           2.35f
#define COMPLETE_OVERLAP_RELEASE_PERIOD         0.55f
#define COMPLETE_OVERLAP_RELEASE_MAX_SPEED      2.40f
#define COMPLETE_OVERLAP_RELEASE_SPEED_AV       5.20f
#define COMPLETE_OVERLAP_RELEASE_SPEED_HUMAN    4.05f
#define COMPLETE_OVERLAP_RELEASE_ACCEL_SCALE    0.78f
#define COMPLETE_OVERLAP_CONTACT_FORWARD_NUDGE  2.85f
#define COMPLETE_OVERLAP_CONTACT_BACKOFF_EXTRA  3.10f
#define MISSION_ABANDON_ENABLED                   1
#define MISSION_ABANDON_WAIT                      0.24f
#define MISSION_ABANDON_ACTIVE_LC_WAIT            0.16f
#define MISSION_ABANDON_SPEED_EPS                 1.35f
#define MISSION_ABANDON_FRONT_GAP                22.0f
#define MISSION_ABANDON_PREP_RATIO                0.98f
#define MISSION_ABANDON_NO_START_EXTRA           34.0f
#define MISSION_ABANDON_EARLY_EXTRA              16.0f
#define MISSION_ABANDON_RELEASE_SPEED_AV          6.20f
#define MISSION_ABANDON_RELEASE_SPEED_HUMAN       4.90f
#define MISSION_ABANDON_ACCEL_SCALE               0.78f
#define CRUISE_RANDOM_LANE_CHANGE_ENABLED          1
#define CRUISE_RANDOM_LANE_MIN_LINK_LENGTH      210.0f
#define CRUISE_RANDOM_LANE_MIN_DIST_TO_NODE     145.0f
#define CRUISE_RANDOM_LANE_COOLDOWN_READY         0.05f
#define CRUISE_RANDOM_LANE_MAX_GROUP              16
#define ROUTE_LANE_RUNTIME_REPAIR_ENABLED          1
#define ROUTE_LANE_REPAIR_LOOKAHEAD                8
#define ROUTE_LANE_REPAIR_LOOKBACK                 4
#define ROUTE_NEXT_LANE_EQUIV_SCAN                14
#define CRUISE_RANDOM_LANE_CHANGE_CHECK_PERIOD     4.0f
#define CRUISE_RANDOM_LANE_CHANGE_PROB_AV          0.105f
#define CRUISE_RANDOM_LANE_CHANGE_PROB_HUMAN       0.060f
#define LANE_SPREAD_EMPTY_FRONT_GAP               55.0f
#define LANE_SPREAD_EMPTY_REAR_GAP                34.0f
#define ROUTE_LANE_REPAIR_SPEED_CAP                4.0f
#define CRUISE_RANDOM_LANE_DECISION_PERIOD        4.0f
#define CRUISE_RANDOM_LANE_CHANGE_PROB            0.34f
#define CRUISE_RANDOM_LANE_UTILITY_TOL            0.32f
#define OPEN_LANE_LC_UTILITY_TOL                  0.22f
#define ROUTE_MISMATCH_REPAIR_ENABLED                1
#define ROUTE_POS_REPAIR_SCAN_MAX                  24
#define UPCOMING_EXIT_LANE_PREP_ENABLED             1
#define UPCOMING_EXIT_LOOKAHEAD_LANES               3
#define UPCOMING_EXIT_PREP_EXTRA_DIST            12.0f
#define UPCOMING_EXIT_PREP_MAX_DIST             130.0f
#define WRONG_LANE_STALL_FORCE_WAIT               0.55f
#define STALE_BRAKE_CLEAR_FRONT_GAP             20.0f
#define STALE_BRAKE_CLEAR_ACCEL_AV               0.72f
#define STALE_BRAKE_CLEAR_ACCEL_HUMAN            0.50f
#define CONTACT_REPAIR_HOLD_ACCEL                0.00f
#define CONTACT_REPAIR_LOSER_SPEED_CAP           0.35f
#define LANE_CHANGE_OVERLAP_SNAP_ENABLED          1
#define CONTACT_HARD_VERIFY_INFLATE              0.04f
#define NODE_OVERLAP_HARD_BACKOFF                7.20f
#define NODE_OVERLAP_HARD_FORWARD                4.80f
#define NODE_OVERLAP_HARD_MIN_SPEED_AV           3.10f
#define NODE_OVERLAP_HARD_MIN_SPEED_HUMAN        2.35f
#define LC_OVERLAP_COMMIT_T                      0.56f
#define LC_OVERLAP_MIN_SPEED                     1.25f
#define LC_PREP_OPEN_FRONT_RELAX_GAP            34.0f
#define LC_PREP_TARGET_FRONT_TIME                0.70f
#define LC_PREP_TARGET_REAR_TIME                 0.85f
#define CONGESTION_ESCAPE_LC_ENABLED                1
#define CONGESTION_ESCAPE_CURRENT_GAP           18.0f
#define CONGESTION_ESCAPE_FRONT_GAIN             8.0f
#define CONGESTION_ESCAPE_SEARCH_RADIUS        135.0f
#define CONGESTION_ESCAPE_MIN_DIST_TO_NODE      42.0f
#define CONGESTION_ESCAPE_MIN_LINK_LENGTH       42.0f
#define CONGESTION_ESCAPE_COOLDOWN_READY         0.15f
#define LANE_SPREAD_CHANGE_ENABLED                 1
#define LANE_SPREAD_FRONT_GAIN                  12.0f
#define LANE_SPREAD_REAR_GAIN                    5.0f
#define LANE_SPREAD_SEARCH_RADIUS              125.0f
#define LANE_SPREAD_MIN_DIST_TO_NODE            55.0f
#define LANE_SPREAD_MIN_LINK_LENGTH             70.0f
#define LANE_SPREAD_MIN_CURRENT_GAP              0.0f
#define LANE_SPREAD_COOLDOWN_READY               0.20f
#define OPEN_LANE_REAR_GAP_AV                   12.0f
#define OPEN_LANE_REAR_GAP_HUMAN                20.0f
#define OPEN_LANE_FRONT_GAP_AV                  12.0f
#define OPEN_LANE_FRONT_GAP_HUMAN               20.0f
#define OPEN_LANE_TARGET_REAR_SPEED_TIME         0.85f
#define OPEN_LANE_TARGET_FRONT_SPEED_TIME        0.45f
#define ZIPPER_MERGE_ENABLED                       1
#define ZIPPER_MERGE_RANGE                      40.0f
#define ZIPPER_MERGE_CLOSER_EPS                  2.25f
#define ZIPPER_MERGE_ALTERNATE_PERIOD            0.85f
#define ZIPPER_MERGE_EXIT_GAP                   10.0f
#define LANE_COUNT_CHANGE_CONTINUATION_ENABLED     1
#define LANE_COUNT_CHANGE_MAX_TURN_DEG           62.0f
#define LANE_COUNT_CHANGE_PREP_ENABLED             1
#define LANE_COUNT_CHANGE_PREP_MIN_DIST          34.0f
#define LANE_COUNT_CHANGE_PREP_MAX_DIST         420.0f
#define LANE_COUNT_CHANGE_PREP_PER_DROPPED      110.0f
#define LANE_COUNT_CHANGE_EDGE_ALIGN_EPS          1.25f
#define DESTINATION_SPREAD_LC_ENABLED              1
#define DESTINATION_SPREAD_MIN_DIST_TO_END       95.0f
#define DESTINATION_SPREAD_MIN_LINK_LENGTH      130.0f
#define DESTINATION_SPREAD_RANDOM_PROB            0.28f
#define MIDROAD_NEG_ACCEL_WATCHDOG_ENABLED          1
#define MIDROAD_NEG_ACCEL_CLEAR_SPEED             0.55f
#define MIDROAD_NEG_ACCEL_CLEAR_FRONT_GAP         28.0f
#define MIDROAD_NEG_ACCEL_CLEAR_MIN_END_DIST      36.0f
#define MIDROAD_NEG_ACCEL_RELEASE_SCALE            0.66f
#define STOPPED_NEG_ACCEL_ZERO_SPEED              0.08f
#define OPEN_LANE_EMPTIEST_GROUP_SCAN_ENABLED       1
#define OPEN_LANE_EMPTIEST_SCAN_PERIOD             3.0f
#define OPEN_LANE_EMPTIEST_FRONT_GAIN              6.0f
#define OPEN_LANE_EMPTIEST_SCORE_GAIN              2.0f
#define OPEN_LANE_EMPTIEST_MIN_EXIT_DIST          85.0f
#define MIDROAD_ZERO_ACCEL_WATCHDOG_ENABLED          1
#define MIDROAD_ZERO_ACCEL_CLEAR_SPEED             0.48f
#define MIDROAD_ZERO_ACCEL_CLEAR_FRONT_GAP         22.0f
#define MIDROAD_ZERO_ACCEL_RELEASE_SCALE            0.68f
#define MIDROAD_ZERO_ACCEL_MIN_END_DIST            12.0f
#define RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED          1
#define RIGHT_EDGE_BOTTLENECK_MIN_GROUP              3
#define RIGHT_EDGE_BOTTLENECK_IDX_LIMIT              1
#define RIGHT_EDGE_BOTTLENECK_PENALTY            260.0f
#define RIGHT_EDGE_INNER_BONUS                     16.0f
#define RIGHT_EDGE_SAFE_FRONT_LOSS                 24.0f
#define RIGHT_EDGE_SCAN_SCORE_GAIN                -24.0f
#define LANE_DROP_AMBIGUOUS_RIGHT_EDGE_FALLBACK      1
#define LANE_DROP_ACTIVE_LC_ABORT_T                0.38f
#define LANE_DROP_ACTIVE_LC_COMMIT_T               0.62f
#define LANE_DROP_ACTIVE_LC_BARRIER_EXTRA           1.25f
#define LANE_DROP_ACTIVE_LC_RELEASE_SPEED          1.60f
#define LANE_DROP_MERGE_GATE_ENABLED                1
#define LANE_DROP_CONNECTOR_GROUP_CLEAR_ENABLED     1
#define LANE_DROP_RECEIVING_BALANCE_ENABLED         1
#define LANE_DROP_ACTIVE_LC_ABORT_ON_UNSAFE         1
#define RIGHT_EDGE_FORCE_INNER_LC_ENABLED           1
#define RIGHT_EDGE_FORCE_INNER_MIN_EXIT_DIST       35.0f
#define RIGHT_EDGE_FORCE_INNER_MAX_CUR_IDX          1
#define RIGHT_EDGE_FORCE_INNER_MIN_TARGET_GAIN    -28.0f
#define LANE_DROP_TAPER_SAME_TARGET_ONLY             1
#define CONNECTOR_GROUP_CLEAR_SAME_TARGET_ONLY       1
#define NODE_CONTINUATION_RELEASE_ENABLED            1
#define LANE_GAIN_RECEIVING_BALANCE_ENABLED          1
#define LANE_GAIN_AMBIGUOUS_RIGHT_EDGE_FALLBACK      1
#define THROUGH_LANE_BALANCE_TARGET_ENABLED          1
#define THROUGH_LANE_BALANCE_TARGET_BONUS           58.0f
#define THROUGH_LANE_BALANCE_DISTANCE_PENALTY        9.0f
#define THROUGH_LANE_BALANCE_RIGHT_EDGE_PENALTY    210.0f
#define OVERLAP_CONNECTOR_RELEASE_ENABLED            1
#define OVERLAP_CONNECTOR_RELEASE_WAIT               0.70f
#define OVERLAP_CONTACT_PRIORITY_NUDGE_ENABLED       1
#define OVERLAP_CONTACT_WINNER_NUDGE                 2.10f
#define OVERLAP_CONTACT_WINNER_MIN_SPEED             2.35f
#define MISSED_EXIT_STRAIGHT_FALLBACK_ENABLED      1
#define MISSED_EXIT_STRAIGHT_MAX_DEG             58.0f
#define MISSED_EXIT_STRAIGHT_RAMP_PENALTY        42.0f
#define MISSED_EXIT_STRAIGHT_RIGHT_EDGE_PENALTY   8.5f
#define MISSED_EXIT_OFFROUTE_TAIL_ENABLED          1
#define THROUGH_LANE_BALANCE_LC_ENABLED             1
#define THROUGH_LANE_BALANCE_LC_MIN_LINK_LENGTH  12.0f
#define THROUGH_LANE_BALANCE_LC_MIN_DIST_TO_NODE 12.0f
#define THROUGH_LANE_BALANCE_FORCE_RIGHT_IDX        0
#define FLOW_BALANCE_MANDATORY_FRONT_SCALE       0.72f
#define FLOW_BALANCE_MANDATORY_REAR_SCALE        0.76f
#define CONNECTOR_ENTRY_WAIT_RELEASE_ENABLED        1
#define CONNECTOR_ENTRY_WAIT_RELEASE_MAX          45.0f
#define GRIDLOCK_NODE_DRAIN_ENABLED              1
#define GRIDLOCK_NODE_DRAIN_WAIT                 1.65f
#define GRIDLOCK_NODE_DRAIN_FRONT_GAP           13.0f
#define GRIDLOCK_NODE_DRAIN_NEAR_DIST           16.0f
#define GRIDLOCK_CONNECTOR_EXIT_FORCE_GAP        2.20f
#define GRIDLOCK_CONNECTOR_EXIT_FORCE_SPEED_AV   2.60f
#define GRIDLOCK_CONNECTOR_EXIT_FORCE_SPEED_HUMAN 1.90f
#define GRIDLOCK_STALLED_CONNECTOR_SPEED         0.20f
#define GRIDLOCK_PAIR_RELEASE_WAIT_ADVANTAGE     0.60f
#define INTERCHANGE_EDGE_ONLY_ENABLED             1
#define INTERCHANGE_RAMP_MAX_GROUP_LANES          1
#define INTERCHANGE_MAIN_MIN_GROUP_LANES          2
#define RIGHT_TURN_ENTRY_BACKOFF_MIN          6.00f
#define RIGHT_TURN_ENTRY_BACKOFF_MAX         16.00f
#define RIGHT_TURN_ENTRY_BACKOFF_BASE         7.20f
#define RIGHT_TURN_ENTRY_BACKOFF_PER_DEG      0.060f
#define TURN_ARC_STRICT_MAX_SWEEP_RAD         2.05f
#define INTERSECTION_BOX_LANE_WIDTH_MULT       1.55f
#define INTERSECTION_BOX_MIN_DEPTH             5.25f
#define INTERSECTION_BOX_MAX_DEPTH            19.50f
#define INTERSECTION_BOX_ENTRY_MARGIN          0.65f
#define INTERSECTION_BOX_PRIORITY_BONUS       96
#define LANE_CHANGE_BOX_CLEAR_MULT             3.80f
#define STRAIGHT_NO_TURN_SPEED_CAP_DEG        12.0f
#define RIGHT_TURN_CURB_BIAS_FRAC              0.18f
#define FRONT_CLEAR_PRIORITY_GAP              16.0f
#define FRONT_CLEAR_PRIORITY_BONUS              18
#define FRONT_BLOCKED_PRIORITY_PENALTY           8
#define FRONT_EMPTY_RELEASE_WAIT_SCALE        0.42f
#define LANE_CHANGE_SIGNAL_PREP_TIME          0.42f
#define LANE_CHANGE_DEADLINE_TIME_MARGIN      0.55f
#define LANE_CHANGE_PREP_BRAKE_AV             1.45f
#define LANE_CHANGE_PREP_BRAKE_HUMAN          1.10f
#define LANE_CHANGE_REAR_ASSERT_TIME          2.10f
#define LANE_CHANGE_REAR_ASSERT_GAP_MULT      1.70f
#define INDICATOR_MERGE_COURTESY_RANGE        30.0f
#define INDICATOR_MERGE_SIDE_RANGE             8.5f
#define INDICATOR_MERGE_ASSERT_RATE           0.12f
#define INDICATOR_MERGE_YIELD_DECEL_AV        0.85f
#define INDICATOR_MERGE_YIELD_DECEL_HUMAN     0.65f
#define INDICATOR_MERGE_ASSERT_ACCEL_AV       0.75f
#define INDICATOR_MERGE_ASSERT_ACCEL_HUMAN    0.52f
#define CONNECTOR_NAV_AVOIDANCE_ENABLED          0
#define SMART_STALL_SPEED                  SMART_STALL_SPEED_EPS
#define SMART_STALL_RELEASE_SPEED_AV       4.9f
#define SMART_STALL_RELEASE_SPEED_HUMAN    3.8f
#define SMART_STALL_RELEASE_ACCEL_SCALE    SMART_STALL_CLEAR_ACCEL_SCALE
#define PRIORITY_GATE_FRONT_CLEAR_OVERRIDE_WAIT 0.85f
#define CONNECTOR_EXIT_SPACE_TIME          0.90f
#define CONNECTOR_EXIT_SPACE_MIN          12.0f
#define LC_ACTIVE_FREEZE_FRONT_MIN          5.2f
#define LC_ACTIVE_FREEZE_REAR_MIN           6.8f
#define LC_ACTIVE_FREEZE_T_MAX              0.38f
#define LC_ACTIVE_MIDLINE_ABORT_T_MIN       0.38f
#define LC_ACTIVE_MIDLINE_ABORT_T_MAX       0.62f
#define LC_ACTIVE_MIDLINE_COMMIT_T          0.58f
#define LC_ACTIVE_MIDLINE_STUCK_SPEED       0.90f
#define LC_ACTIVE_MIDLINE_ABORT_COOLDOWN    1.35f
#define RENDER_BODY_VERTS_PER_VEHICLE 6
#define RENDER_FULL_VERTS_PER_VEHICLE 30
#define METRIC_SPAWNED       0
#define METRIC_EXITED        1
#define METRIC_TRAVEL_TIME   6
#define METRIC_ACTIVE        7
#define METRIC_ACCEL_COUNT   8
#define METRIC_ACCEL_SUM     9
#define METRIC_ACCEL_SQ_SUM  10
#define METRIC_DECEL_COUNT   11
#define METRIC_DECEL_SUM     12
#define METRIC_DECEL_SQ_SUM  13
#define METRIC_SPEED_SUM     14
#define METRIC_SPEED_COUNT   15
#define METRIC_SLOW_COUNT    16
#define METRIC_STOP_COUNT    19
#define METRIC_COLLISION     20
#define METRIC_SPAWN_FAIL    22
#define METRIC_CONNECTOR_IN  25
#define METRIC_CONNECTOR_RUN 26
#define METRIC_RES_ACCEPT    34
#define METRIC_RES_REJECT    35
#define METRIC_LC_ACCEPT     36
#define METRIC_LC_REJECT     37
#define METRIC_TTC_CRITICAL  38
#define METRIC_TTC_WARNING   39
#define METRIC_HARD_BRAKE    40
#define METRIC_NEAR_MISS     41
#define METRIC_COOP_YIELD    42
#define METRIC_MOBIL_EVAL    43
#define METRIC_DELAY_SUM             44
#define METRIC_DELAY_COUNT           45
#define METRIC_REACTION_SUM          46
#define METRIC_REACTION_COUNT        47
#define METRIC_RESPONSE_LAG_SUM      48
#define METRIC_RESPONSE_LAG_COUNT    49
#define METRIC_STEER_ABS_SUM         50
#define METRIC_STEER_COUNT           51
#define METRIC_YAW_RATE_ABS_SUM      52
#define METRIC_YAW_RATE_COUNT        53
#define METRIC_HEADWAY_SUM           54
#define METRIC_HEADWAY_COUNT         55
#define METRIC_MIN_GAP_SUM           56
#define METRIC_MIN_GAP_COUNT         57
#define METRIC_INTERSECTION_WAIT     58
#define METRIC_RED_LIGHT_STOP        59
#define METRIC_YELLOW_STOP           60
#define METRIC_YELLOW_GO             61
#define METRIC_RED_LIGHT_VIOLATION   62
#define METRIC_SENSOR_DETECTION      63
#define METRIC_SENSOR_FRONT_HIT      64
#define METRIC_CONFLICT_YIELD        65
#define METRIC_INTERACTION_BRAKE     66
#define METRIC_QUEUE_DELAY_SUM       67
#define METRIC_QUEUE_DELAY_COUNT     68
#define METRIC_LANE_CHANGE_TIME_SUM  69
#define METRIC_LANE_CHANGE_TIME_COUNT 70
#define METRIC_CONNECTOR_DELAY_SUM   71
#define METRIC_CONNECTOR_DELAY_COUNT 72
#define METRIC_COMFORT_BRAKE         73
#define METRIC_STANDSTILL_TIME       74
#define METRIC_TIME_LOSS_SUM         75
#define METRIC_TIME_LOSS_COUNT       76
#define METRIC_TURN_LANE_PREP        77
#define METRIC_TURN_LANE_BLOCK       78
#define METRIC_TURN_LANE_ILLEGAL     79
#define METRIC_UNSIGNAL_RIGHT_YIELD  80
#define METRIC_UNSIGNAL_PRIORITY_GO  81
#define METRIC_UNSIGNAL_CONFLICT     82
#define METRIC_DEADLOCK_WAIT         83
#define METRIC_DEADLOCK_RELEASE      84
#define METRIC_DEADLOCK_CREEP        85
#define METRIC_CONNECTOR_SAFE_YIELD  86
#define METRIC_PRIORITY_ENTRY_BLOCK  87
#define METRIC_INDICATOR_LEFT_ON     88
#define METRIC_INDICATOR_RIGHT_ON    89
#define METRIC_INDICATOR_CONFLICT_YIELD 90
#define METRIC_INDICATOR_PRIORITY_GO 91
#define METRIC_ANTI_COLLISION_BRAKE  92
#define METRIC_PENETRATION_PREVENTED 93
#define METRIC_CONNECTOR_CROSS_YIELD 94
#define METRIC_DEADLOCK_ESCAPE_GO    95
#define METRIC_PRIORITY_GATE_CANDIDATE 96
#define METRIC_PRIORITY_GATE_GRANTED   97
#define METRIC_PRIORITY_GATE_BLOCKED   98
#define METRIC_INTERSECTION_OCCUPIED_HOLD 99
#define METRIC_FORCE_PASS_THROUGH      100
#define METRIC_UNIQUE_PRIORITY_TIE     101
#define METRIC_DEADLOCK_PRIORITY_RELEASE 102
#define METRIC_ENTRY_QUEUE_HOLD        103
#define METRIC_PRIORITY_CONFLICT_FREE_GO 104
#define METRIC_PRIORITY_PATH_BLOCK       105
#define METRIC_PRIORITY_ACTIVE_PATH_HOLD 106
#define METRIC_HUMAN_AI_ASSERTIVE_GO     107
#define METRIC_HUMAN_AI_COURTESY_YIELD   108
#define METRIC_RIGHT_TURN_SYMMETRIC_PATH    109
#define METRIC_RIGHT_TURN_EXIT_GAP_HOLD  110
#define METRIC_FRONT_SPACE_RELEASE       111
#define METRICS_SIZE                 112
#ifndef AVABM_METRICS_MODE
#define AVABM_METRICS_MODE 1
#endif
AVABM_DINLINE bool avabm_metric_enabled_ecs(int metric_idx) {
#if AVABM_METRICS_MODE == 0
    (void)metric_idx;
    return false;
#elif AVABM_METRICS_MODE == 2
    switch (metric_idx) {
        case METRIC_SPAWNED:
        case METRIC_EXITED:
        case METRIC_TRAVEL_TIME:
        case METRIC_ACTIVE:
        case METRIC_SPEED_SUM:
        case METRIC_SPEED_COUNT:
        case METRIC_SLOW_COUNT:
        case METRIC_STOP_COUNT:
        case METRIC_COLLISION:
        case METRIC_SPAWN_FAIL:
        case METRIC_CONNECTOR_IN:
        case METRIC_CONNECTOR_RUN:
        case METRIC_LC_ACCEPT:
        case METRIC_LC_REJECT:
        case METRIC_HARD_BRAKE:
        case METRIC_QUEUE_DELAY_SUM:
        case METRIC_QUEUE_DELAY_COUNT:
        case METRIC_STANDSTILL_TIME:
        case METRIC_TIME_LOSS_SUM:
        case METRIC_TIME_LOSS_COUNT:
        case METRIC_RED_LIGHT_VIOLATION:
        case METRIC_PENETRATION_PREVENTED:
            return true;
        default:
            return false;
    }
#else
    (void)metric_idx;
    return true;
#endif
}
#define AVABM_METRIC_ADD(METRICS_PTR, METRIC_IDX, VALUE_EXPR) \
do { \
if ((METRICS_PTR) != nullptr && avabm_metric_enabled_ecs((int)(METRIC_IDX))) { \
atomicAdd(&((METRICS_PTR)[(METRIC_IDX)]), (VALUE_EXPR)); \
} \
} while (0)
struct RenderVertex {
    float x, y;
    float r, g, b, a;
    float size;
};
struct ECSArrays {
    int* alive;
    float* x;
    float* y;
    float* s;
    float* speed;
    float* accel;
    float* heading;
    float* steer_angle;
    float* length;
    float* width;
    int* driver_type;
    float* reaction_time;
    float* min_gap;
    float* aggressiveness;
    float* politeness;
    float* risk_tolerance;
    float* comfort_decel;
    float* desired_speed_factor;
    int* lane_id;
    int* route_id;
    int* route_pos;
    float* entry_time;
    int* vehicle_state;
    int* connector_from_lane;
    int* connector_to_lane;
    float* connector_s;
    float* connector_length;
    int* lane_change_active;
    int* lane_change_from_lane;
    int* lane_change_to_lane;
    float* lane_change_t;
    float* lane_change_duration;
    float* lc_cooldown;
    int* turn_signal;
    float* turn_signal_time;
};
struct RoadNetwork {
    const float* lane_length;
    const float* lane_start_x;
    const float* lane_start_y;
    const float* lane_end_x;
    const float* lane_end_y;
    const float* lane_speed_limit;
    const int* lane_start_node;
    const int* lane_end_node;
    const int* left_lane;
    const int* right_lane;
    const int* route_offsets;
    const int* route_lanes;
    const int* route_turns;
    int num_lanes;
    int num_nodes;
    int num_routes;
};
struct Signals {
    const int* signal_node;
    const int* signal_turn;
    const float* signal_cycle;
    const float* signal_green_start;
    const float* signal_green_end;
    const float* signal_yellow_start;
    const float* signal_yellow_end;
    int num_signals;
};
struct SpatialGrid {
    int* cell_head;
    int* cell_next;
    int* cell_epoch;
    int epoch;
    int* lane_cell_head;
    int* lane_cell_next;
    int lane_cells_per_lane;
    float min_x;
    float min_y;
    float cell_size;
    int width;
    int height;
};
struct SpawnConfig {
    float* spawn_accumulator;
    const float* demand_vps;
    const float* demand_profile_vps;
    const int* demand_profile_has;
    const int* spawn_lane;
    const int* spawn_route;
    int* spawn_alloc_cursor;
    int num_spawn_points;
    int demand_profile_slots;
    float demand_profile_slot_seconds;
    float av_penetration;
};
struct PerceptionSoA {
    float* front_gap;
    float* front_speed;
    float* front_s;
    float* front_length;
    int* front_lane;
    float* target_front_gap;
    float* target_front_speed;
    float* target_rear_gap;
    float* target_rear_speed;
};
struct DecisionSoA {
    float* desired_speed;
    float* target_accel;
    int* wants_lane_change;
    int* lane_change_target;
    int* wants_connector;
    int* connector_target_lane;
    int* should_exit;
};
__global__ void clear_intersection_priority_gate_kernel(int* priority_table, int num_nodes);
__global__ void mark_intersection_occupancy_kernel(ECSArrays ecs, RoadNetwork road, int* priority_table, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void select_intersection_priority_candidates_kernel(ECSArrays ecs, RoadNetwork road, Signals signals,
        DecisionSoA decision, PerceptionSoA perception, int* priority_table, float* metrics, float current_time, float dt,
        int max_entities, const int* active_ids, const int* active_count);
__global__ void apply_intersection_priority_gate_kernel(ECSArrays ecs, RoadNetwork road, Signals signals, DecisionSoA decision,
        SpatialGrid grid, PerceptionSoA perception, int* priority_table, float* metrics, float current_time, float dt,
        int max_entities, const int* active_ids, const int* active_count);
__global__ void clear_int_kernel(int* data, int n, int value);
__global__ void clear_float_kernel(float* data, int n, float value);
__global__ void clear_decision_kernel(DecisionSoA decision, int n, const int* active_ids, const int* active_count);
__global__ void compact_active_entities_kernel(ECSArrays ecs, int* active_ids, int* active_count, int max_entities);
__global__ void compact_active_archetypes_kernel(ECSArrays ecs, const int* active_ids, const int* active_count,
        int* lane_active_ids, int* lane_active_count, int* connector_active_ids, int* connector_active_count, int max_entities);
__global__ void compact_active_all_archetypes_kernel(ECSArrays ecs, int* active_ids, int* active_count, int* lane_active_ids,
        int* lane_active_count, int* connector_active_ids, int* connector_active_count, int max_entities);
__global__ void spatial_hash_build_system(ECSArrays ecs, SpatialGrid grid, int max_entities, const int* active_ids,
        const int* active_count);
__global__ void lane_hash_build_system(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, int max_entities, const int* active_ids,
        const int* active_count);
__global__ void resolve_spawn_overlap_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, SpawnConfig spawn,
        float* metrics, float current_time, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void spawn_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, SpawnConfig spawn, uint32_t* rng_state,
        float* metrics, int* spawn_lane_locks, int spawn_lock_count, float current_time, float dt, int max_entities,
        int step_index, int* active_ids, int* active_count);
__global__ void turn_signal_system_kernel(ECSArrays ecs, DecisionSoA decision, RoadNetwork road, float* metrics, float dt,
        int max_entities, const int* active_ids, const int* active_count);
__global__ void perception_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, PerceptionSoA perception,
        float* metrics, int max_entities, const int* active_ids, const int* active_count);
__global__ void clear_reservation_system(int* reservation_table, int total_slots);
__global__ void decision_system_kernel(ECSArrays ecs, RoadNetwork road, Signals signals, SpatialGrid grid,
        PerceptionSoA perception, DecisionSoA decision, int* reservation_table, float* metrics, float current_time, float dt,
        int max_entities, const int* active_ids, const int* active_count);
__global__ void lane_change_system_kernel(ECSArrays ecs, DecisionSoA decision, RoadNetwork road, float dt, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void motion_system_kernel(ECSArrays ecs, DecisionSoA decision, RoadNetwork road, PerceptionSoA perception,
        float* metrics, float current_time, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void connector_enter_system_kernel(ECSArrays ecs, DecisionSoA decision, RoadNetwork road, SpatialGrid grid,
        float* metrics, float current_time, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void connector_motion_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, float* metrics,
        float current_time, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void route_lane_repair_system_kernel(ECSArrays ecs, RoadNetwork road, float* metrics, float dt, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void contact_resolve_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, float* metrics,
        float current_time, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void collision_system_kernel(ECSArrays ecs, SpatialGrid grid, float* metrics, float dt, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void local_obstacle_avoidance_system_kernel(ECSArrays ecs, RoadNetwork road, SpatialGrid grid, PerceptionSoA perception,
        DecisionSoA decision, float* metrics, float current_time, float dt, int max_entities, const int* active_ids,
        const int* active_count);
__global__ void front_clear_must_go_system_kernel(ECSArrays ecs, RoadNetwork road, Signals signals, SpatialGrid grid,
        PerceptionSoA perception, DecisionSoA decision, float* metrics, float current_time, float dt, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void safety_metrics_system_kernel(ECSArrays ecs, SpatialGrid grid, float* metrics, int max_entities,
        const int* active_ids, const int* active_count);
__global__ void stats_system_kernel(ECSArrays ecs, RoadNetwork road, PerceptionSoA perception, DecisionSoA decision,
        float* metrics, float dt, int max_entities, const int* active_ids, const int* active_count);
__global__ void render_textured_body_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities);
__global__ void render_body_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities);
__global__ void render_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities);
__global__ void render_textured_body_dense_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities, const int* active_ids,
        const int* active_count, int* render_count);
__global__ void render_body_dense_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities, const int* active_ids,
        const int* active_count, int* render_count);
__global__ void render_dense_system_kernel(RenderVertex* out, ECSArrays ecs, int max_entities, const int* active_ids,
        const int* active_count, int* render_count);
__global__ void clear_render_tail_system_kernel(RenderVertex* out, const int* render_count, const int* previous_render_count,
        int max_entities, int verts_per_vehicle, int force_full_tail_clear);
__global__ void remember_render_count_system_kernel(const int* render_count, int* previous_render_count, int max_entities);
AVABM_DINLINE float clampf_cuda(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}
AVABM_DINLINE int clampi_cuda(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}
AVABM_DINLINE float avabm_square_ecs(float x) {
    return x * x;
}
AVABM_DINLINE float avabm_fourth_power_ecs(float x) {
#if AVABM_FAST_EQUIV_MATH
    float x2 = x * x;
    return x2 * x2;
#else
    return powf(x, 4.0f);
#endif
}
AVABM_DINLINE float avabm_second_power_ecs(float x) {
#if AVABM_FAST_EQUIV_MATH
    return x * x;
#else
    return powf(x, 2.0f);
#endif
}
AVABM_DINLINE float avabm_min_cruise_speed_mps_ecs() {
#if AVABM_MIN_CRUISE_SPEED_ENABLED
    return fmaxf(0.0f, ((float)AVABM_MIN_CRUISE_SPEED_KMH) * AVABM_KMH_TO_MPS);
#else
    return 0.0f;
#endif
}
AVABM_DINLINE uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    if (x == 0u) x = 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}
AVABM_DINLINE float rand_uniform(uint32_t& state) {
    return (float)(xorshift32(state) & 0x00FFFFFF) / 16777216.0f;
}
AVABM_DINLINE uint32_t hash_u32_ecs(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
AVABM_DINLINE float hash01_ecs(uint32_t x) {
    return (float)(hash_u32_ecs(x) & 0x00FFFFFFu) / 16777216.0f;
}
AVABM_DINLINE bool timed_pair_random_self_wins_ecs(int self, int other, float current_time, float period) {
    int a = self < other ? self : other;
    int b = self < other ? other : self;
    float safe_period = fmaxf(period, 0.01f);
    uint32_t slot = (uint32_t)floorf(fmaxf(current_time, 0.0f) / safe_period);
    uint32_t h = hash_u32_ecs(((uint32_t)(a + 1) * 747796405u) ^ ((uint32_t)(b + 3) * 2891336453u) ^ (slot * 277803737u));
    bool lower_id_wins = (h & 1u) == 0u;
    return (self == a) ? lower_id_wins : !lower_id_wins;
}
AVABM_DINLINE float wrap_pi(float a) {
    while (a > 3.14159265359f) a -= 6.28318530718f;
    while (a < -3.14159265359f) a += 6.28318530718f;
    return a;
}
AVABM_DINLINE float smoothstep01(float t) {
    t = clampf_cuda(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
AVABM_DINLINE float unwrap_angle_near(float target, float reference) {
    return reference + wrap_pi(target - reference);
}
AVABM_DINLINE float advance_heading_limited(float current, float target, float max_yaw_rate, float dt) {
    float delta = wrap_pi(target - current);
    float max_delta = fmaxf(0.01f, max_yaw_rate * fmaxf(dt, 0.001f));
    delta = clampf_cuda(delta, -max_delta, max_delta);
    return wrap_pi(current + delta);
}
AVABM_DINLINE float vehicle_wheelbase_from_length(float len) {
    return clampf_cuda(len * KIN_WHEELBASE_FACTOR, KIN_MIN_WHEELBASE, KIN_MAX_WHEELBASE);
}
AVABM_DINLINE float vehicle_yaw_rate_from_steer(float speed, float steer, float wheelbase) {
    if (speed < KIN_MIN_YAW_SPEED) return 0.0f;
    return speed * tanf(steer) / fmaxf(wheelbase, 0.1f);
}
AVABM_DINLINE float advance_heading_bicycle_ecs(int id, ECSArrays ecs, float target_heading, float speed, float dt,
        float& steer_out, float& yaw_rate_out) {
    bool human = ecs.driver_type[id] == HUMAN;
    float old_h = ecs.heading[id];
    float wheelbase = vehicle_wheelbase_from_length(ecs.length[id]);
    float max_steer = human ? MAX_STEER_HUMAN : MAX_STEER_AV;
    float max_steer_rate = human ? MAX_STEER_RATE_HUMAN : MAX_STEER_RATE_AV;
    float max_yaw_rate = human ? MAX_YAW_RATE_HUMAN : MAX_YAW_RATE_AV;
    float prev_steer = clampf_cuda(ecs.steer_angle[id], -max_steer, max_steer);
    if (speed < KIN_MIN_YAW_SPEED || dt <= 0.0f) {
        float relax = clampf_cuda(max_steer_rate * fmaxf(dt, 0.001f), 0.0f, 1.0f);
        steer_out = prev_steer + clampf_cuda(-prev_steer, -relax, relax);
        yaw_rate_out = 0.0f;
        return old_h;
    }
    float target = unwrap_angle_near(target_heading, old_h);
    float err = wrap_pi(target - old_h);
    float lookahead = speed * (human ? 1.70f : 1.35f) + wheelbase * 1.85f;
    lookahead = clampf_cuda(lookahead, wheelbase * 1.35f, human ? 26.0f : 22.0f);
    float curvature = 2.0f * sinf(err) / fmaxf(lookahead, 0.1f);
    float steer_cmd = atanf(wheelbase * curvature);
    steer_cmd = clampf_cuda(steer_cmd, -max_steer, max_steer);
    float max_delta_steer = max_steer_rate * fmaxf(dt, 0.001f);
    float steer = prev_steer + clampf_cuda(steer_cmd - prev_steer, -max_delta_steer, max_delta_steer);
    steer = clampf_cuda(steer, -max_steer, max_steer);
    float yaw_rate = vehicle_yaw_rate_from_steer(speed, steer, wheelbase);
    yaw_rate = clampf_cuda(yaw_rate, -max_yaw_rate, max_yaw_rate);
    float new_h = wrap_pi(old_h + yaw_rate * dt);
    float path_err = wrap_pi(target_heading - new_h);
    float lock_err = human ? PATH_HEADING_LOCK_HUMAN : PATH_HEADING_LOCK_AV;
    if (fabsf(path_err) > lock_err) {
        float corrected = wrap_pi(target_heading - copysignf(lock_err, path_err));
        float corrected_yaw = wrap_pi(corrected - old_h) / fmaxf(dt, 0.001f);
        corrected_yaw = clampf_cuda(corrected_yaw, -max_yaw_rate * 1.35f, max_yaw_rate * 1.35f);
        new_h = wrap_pi(old_h + corrected_yaw * dt);
        yaw_rate = corrected_yaw;
        float steer_from_yaw = atanf(yaw_rate * wheelbase / fmaxf(speed, KIN_MIN_YAW_SPEED));
        steer = clampf_cuda(steer_from_yaw, -max_steer, max_steer);
    }
    steer_out = steer;
    yaw_rate_out = yaw_rate;
    return new_h;
}
AVABM_DINLINE float enforce_path_heading_error_limit_ecs(int id, ECSArrays ecs, float candidate_heading, float path_heading,
        float speed, float dt, float max_error, float& steer_out, float& yaw_rate_out) {
    if (speed < KIN_MIN_YAW_SPEED || dt <= 0.0f) {
        return candidate_heading;
    }
    float err = wrap_pi(path_heading - candidate_heading);
    if (fabsf(err) <= max_error) {
        return candidate_heading;
    }
    bool human = ecs.driver_type[id] == HUMAN;
    float max_steer = human ? MAX_STEER_HUMAN : MAX_STEER_AV;
    float max_yaw_rate = human ? MAX_YAW_RATE_HUMAN : MAX_YAW_RATE_AV;
    float old_h = ecs.heading[id];
    float locked = wrap_pi(path_heading - copysignf(max_error, err));
    float yaw_rate = wrap_pi(locked - old_h) / fmaxf(dt, 0.001f);
    yaw_rate = clampf_cuda(yaw_rate, -max_yaw_rate * 1.55f, max_yaw_rate * 1.55f);
    float new_h = wrap_pi(old_h + yaw_rate * dt);
    float wheelbase = vehicle_wheelbase_from_length(ecs.length[id]);
    float steer = atanf(yaw_rate * wheelbase / fmaxf(speed, KIN_MIN_YAW_SPEED));
    steer_out = clampf_cuda(steer, -max_steer, max_steer);
    yaw_rate_out = yaw_rate;
    return new_h;
}
AVABM_DINLINE float sensor_half_fov_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_FRONT_HALF_FOV_HUMAN : SENSOR_FRONT_HALF_FOV_AV;
}
AVABM_DINLINE float sensor_front_range_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_FRONT_RANGE_HUMAN : SENSOR_FRONT_RANGE_AV;
}
AVABM_DINLINE float sensor_side_range_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_SIDE_RANGE_HUMAN : SENSOR_SIDE_RANGE_AV;
}
AVABM_DINLINE bool point_in_oriented_cone(float ox, float oy, float heading, float px, float py, float range, float half_fov,
        float lateral_margin, float& forward, float& lateral, float& dist) {
    float rx = px - ox;
    float ry = py - oy;
    float d2 = rx * rx + ry * ry;
    if (d2 <= 1.0e-6f) {
        forward = 0.0f;
        lateral = 0.0f;
        dist = 0.0f;
        return true;
    }
    dist = sqrtf(d2);
    if (dist > range) return false;
    float fx = cosf(heading);
    float fy = sinf(heading);
    float sx = -fy;
    float sy = fx;
    forward = rx * fx + ry * fy;
    lateral = rx * sx + ry * sy;
    if (forward < -lateral_margin) return false;
    float cone_half_width = fmaxf(forward, 0.0f) * tanf(half_fov) + lateral_margin;
    return fabsf(lateral) <= cone_half_width;
}
AVABM_DINLINE bool sensor_front_cone_detects_ecs(int self, int other, ECSArrays ecs, float range, float half_fov, float& forward,
        float& lateral, float& dist) {
    float margin = SENSOR_CONE_EDGE_MARGIN + 0.5f * fmaxf(ecs.width[other], 1.0f) + SENSOR_RAY_WIDTH;
    return point_in_oriented_cone(ecs.x[self], ecs.y[self], ecs.heading[self], ecs.x[other], ecs.y[other], range, half_fov, margin,
            forward, lateral, dist);
}
AVABM_DINLINE bool sensor_rear_mirror_detects_ecs(int self, int other, ECSArrays ecs, float range, float& forward, float& lateral,
        float& dist) {
    float margin = SENSOR_CONE_EDGE_MARGIN + 0.5f * fmaxf(ecs.width[other], 1.0f) + SENSOR_RAY_WIDTH;
    return point_in_oriented_cone(ecs.x[self], ecs.y[self], wrap_pi(ecs.heading[self] + 3.14159265359f), ecs.x[other],
            ecs.y[other], range, SENSOR_SIDE_HALF_FOV, margin, forward, lateral, dist);
}
AVABM_DINLINE float apply_reaction_delay_accel(float prev_accel, float target_accel, float reaction_time, int dtype, float dt,
        float* metrics) {
    float rt = fmaxf(reaction_time, dtype == HUMAN ? 0.45f : 0.08f);
    float alpha = dt / fmaxf(rt + dt, 0.001f);
    if (dtype == AV) {
        alpha = fmaxf(alpha, 0.42f);
    } else {
        alpha *= 0.90f;
    }
    if (target_accel < -3.0f) {
        alpha = fmaxf(alpha, dtype == HUMAN ? 0.48f : 0.72f);
    }
    alpha = clampf_cuda(alpha, 0.05f, 1.0f);
    float effective = prev_accel + (target_accel - prev_accel) * alpha;
    if (metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_DELAY_SUM, rt);
        AVABM_METRIC_ADD(metrics, METRIC_DELAY_COUNT, 1.0f);
        AVABM_METRIC_ADD(metrics, METRIC_REACTION_SUM, reaction_time);
        AVABM_METRIC_ADD(metrics, METRIC_REACTION_COUNT, 1.0f);
        AVABM_METRIC_ADD(metrics, METRIC_RESPONSE_LAG_SUM, fabsf(target_accel - effective));
        AVABM_METRIC_ADD(metrics, METRIC_RESPONSE_LAG_COUNT, 1.0f);
    }
    return effective;
}
AVABM_DINLINE int world_cell_index(float px, float py, float world_min_x, float world_min_y, float world_cell_size,
        int world_grid_w, int world_grid_h) {
    if (!isfinite(px) || !isfinite(py) || world_cell_size <= 0.0f) return -1;
    int cx = (int)floorf((px - world_min_x) / world_cell_size);
    int cy = (int)floorf((py - world_min_y) / world_cell_size);
    if (cx < 0 || cx >= world_grid_w) return -1;
    if (cy < 0 || cy >= world_grid_h) return -1;
    return cy * world_grid_w + cx;
}
AVABM_DINLINE bool grid_lazy_enabled_ecs(const SpatialGrid grid) {
#if AVABM_LAZY_GRID_ENABLED
    return grid.cell_epoch != nullptr && grid.epoch > 0;
#else
    return false;
#endif
}
AVABM_DINLINE void grid_prepare_cell_for_write_ecs(SpatialGrid grid, int cell) {
    if (!grid_lazy_enabled_ecs(grid)) return;
    int epoch = grid.epoch;
    int seen = grid.cell_epoch[cell];
    if (seen == epoch) return;
    if (seen != -epoch && atomicCAS(&grid.cell_epoch[cell], seen, -epoch) == seen) {
        grid.cell_head[cell] = WORLD_CELL_EMPTY;
        __threadfence();
        atomicExch(&grid.cell_epoch[cell], epoch);
    } else {
        int guard = 0;
        while (grid.cell_epoch[cell] == -epoch && guard < 4096) {
            ++guard;
        }
    }
}
AVABM_DINLINE int grid_head_ecs(SpatialGrid grid, int cell) {
    if (cell < 0) return WORLD_CELL_EMPTY;
    if (!grid_lazy_enabled_ecs(grid)) return grid.cell_head[cell];
    int epoch = grid.epoch;
    int seen = grid.cell_epoch[cell];
    if (seen == epoch) return grid.cell_head[cell];
    if (seen == -epoch) {
        int guard = 0;
        while (grid.cell_epoch[cell] == -epoch && guard < 4096) {
            ++guard;
        }
        if (grid.cell_epoch[cell] == epoch) return grid.cell_head[cell];
    }
    return WORLD_CELL_EMPTY;
}
AVABM_DINLINE void lane_dir(int lane, const RoadNetwork road, float& dx, float& dy) {
    dx = road.lane_end_x[lane] - road.lane_start_x[lane];
    dy = road.lane_end_y[lane] - road.lane_start_y[lane];
    float n = sqrtf(dx * dx + dy * dy);
    if (n < 1.0e-5f) {
        dx = 1.0f;
        dy = 0.0f;
    } else {
        dx /= n;
        dy /= n;
    }
}
AVABM_DINLINE float lane_heading(int lane, const RoadNetwork road) {
    float dx, dy;
    lane_dir(lane, road, dx, dy);
    return atan2f(dy, dx);
}
AVABM_DINLINE void lane_xy_from_s(int lane, float ss, const RoadNetwork road, float& ox, float& oy) {
    float L = fmaxf(road.lane_length[lane], 0.1f);
    float t = clampf_cuda(ss / L, 0.0f, 1.0f);
    ox = road.lane_start_x[lane] + t * (road.lane_end_x[lane] - road.lane_start_x[lane]);
    oy = road.lane_start_y[lane] + t * (road.lane_end_y[lane] - road.lane_start_y[lane]);
}
AVABM_DINLINE void lane_xy_heading_from_s(int lane, float ss, const RoadNetwork road, float& ox, float& oy, float& oh) {
    lane_xy_from_s(lane, ss, road, ox, oy);
    oh = lane_heading(lane, road);
}
AVABM_DINLINE bool lane_connected(int a, int b, const RoadNetwork road) {
    return road.lane_end_node[a] == road.lane_start_node[b];
}
AVABM_DINLINE int route_pos_for_lane_ecs(int route_id, int lane, const RoadNetwork road) {
    if (route_id < 0 || route_id >= road.num_routes) return -1;
    if (lane < 0 || lane >= road.num_lanes) return -1;
    int ro0 = road.route_offsets[route_id];
    int ro1 = road.route_offsets[route_id + 1];
    if (ro1 <= ro0 || ro1 - ro0 > 2048) return -1;
    for (int k = ro0; k < ro1; ++k) {
        if (road.route_lanes[k] == lane) {
            return k - ro0;
        }
    }
    return -1;
}
AVABM_DINLINE float turn_angle_deg(int a, int b, const RoadNetwork road) {
    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);
    float dot = clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
    float cross = ax * by - ay * bx;
    return fabsf(atan2f(cross, dot)) * 57.2957795f;
}
AVABM_DINLINE float lane_signed_turn_deg(int a, int b, const RoadNetwork road) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return 0.0f;
    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);
    float dot = clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
    float cross = ax * by - ay * bx;
    return atan2f(cross, dot) * 57.2957795f;
}
AVABM_DINLINE int turn_code_from_lanes_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (from_lane < 0 || to_lane < 0) return TURN_STRAIGHT;
    float signed_deg = lane_signed_turn_deg(from_lane, to_lane, road);
    if (signed_deg > 25.0f) return TURN_LEFT;
    if (signed_deg < -25.0f) return TURN_RIGHT;
    return TURN_STRAIGHT;
}
AVABM_DINLINE void lane_midpoint_ecs(int lane, const RoadNetwork road, float& mx, float& my) {
    mx = 0.5f * (road.lane_start_x[lane] + road.lane_end_x[lane]);
    my = 0.5f * (road.lane_start_y[lane] + road.lane_end_y[lane]);
}
AVABM_DINLINE bool valid_lane_ecs(int lane, const RoadNetwork road) {
    return lane >= 0 && lane < road.num_lanes;
}
AVABM_DINLINE bool lanes_share_link_geometry_ecs(int a, int b, const RoadNetwork road) {
    if (!valid_lane_ecs(a, road) || !valid_lane_ecs(b, road)) return false;
    bool same_nodes = road.lane_start_node[a] == road.lane_start_node[b] && road.lane_end_node[a] == road.lane_end_node[b];
    if (!same_nodes) return false;
    float ah = lane_heading(a, road);
    float bh = lane_heading(b, road);
    return fabsf(wrap_pi(bh - ah)) < 0.35f;
}
AVABM_DINLINE float neighbor_lateral_cross_ecs(int lane, int neighbor, const RoadNetwork road) {
    if (!lanes_share_link_geometry_ecs(lane, neighbor, road)) return 0.0f;
    float dx, dy;
    lane_dir(lane, road, dx, dy);
    float mx, my, nx, ny;
    lane_midpoint_ecs(lane, road, mx, my);
    lane_midpoint_ecs(neighbor, road, nx, ny);
    return dx * (ny - my) - dy * (nx - mx);
}
AVABM_DINLINE int geometric_left_neighbor_ecs(int lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int candidates[2] = {
        road.left_lane[lane], road.right_lane[lane]
    };
    int best = -1;
    float best_score = 0.05f;
    for (int k = 0; k < 2; ++k) {
        int nb = candidates[k];
        if (!valid_lane_ecs(nb, road)) continue;
        float side = neighbor_lateral_cross_ecs(lane, nb, road);
        if (side > best_score) {
            best_score = side;
            best = nb;
        }
    }
    return best;
}
AVABM_DINLINE int geometric_right_neighbor_ecs(int lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int candidates[2] = {
        road.left_lane[lane], road.right_lane[lane]
    };
    int best = -1;
    float best_score = -0.05f;
    for (int k = 0; k < 2; ++k) {
        int nb = candidates[k];
        if (!valid_lane_ecs(nb, road)) continue;
        float side = neighbor_lateral_cross_ecs(lane, nb, road);
        if (side < best_score) {
            best_score = side;
            best = nb;
        }
    }
    return best;
}
AVABM_DINLINE bool is_leftmost_lane_ecs(int lane, const RoadNetwork road) {
    return valid_lane_ecs(lane, road) && geometric_left_neighbor_ecs(lane, road) < 0;
}
AVABM_DINLINE bool is_rightmost_lane_ecs(int lane, const RoadNetwork road) {
    return valid_lane_ecs(lane, road) && geometric_right_neighbor_ecs(lane, road) < 0;
}
AVABM_DINLINE int rightmost_lane_in_group_ecs(int lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int cur = lane;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP; ++k) {
        int nb = geometric_right_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return cur;
}
AVABM_DINLINE int lane_group_count_and_index_ecs(int lane, const RoadNetwork road, int& out_index) {
    out_index = -1;
    int cur = rightmost_lane_in_group_ecs(lane, road);
    if (!valid_lane_ecs(cur, road)) return 0;
    int count = 0;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP && valid_lane_ecs(cur, road); ++k) {
        if (cur == lane) out_index = count;
        count++;
        int nb = geometric_left_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return count;
}
AVABM_DINLINE int lane_at_right_to_left_index_ecs(int lane, int target_index, const RoadNetwork road) {
    int cur = rightmost_lane_in_group_ecs(lane, road);
    if (!valid_lane_ecs(cur, road)) return -1;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP && valid_lane_ecs(cur, road); ++k) {
        if (k == target_index) return cur;
        int nb = geometric_left_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return -1;
}
AVABM_DINLINE int leftmost_lane_in_group_ecs(int lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int cur = lane;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP; ++k) {
        int nb = geometric_left_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return cur;
}
AVABM_DINLINE float lane_endpoint_dist2_ecs(int lane, float px, float py, bool at_end, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return 1.0e30f;
    float x = at_end ? road.lane_end_x[lane] : road.lane_start_x[lane];
    float y = at_end ? road.lane_end_y[lane] : road.lane_start_y[lane];
    float dx = x - px;
    float dy = y - py;
    return dx * dx + dy * dy;
}
AVABM_DINLINE int nearest_outer_lane_to_point_ecs(int group_lane, float px, float py, bool at_end, const RoadNetwork road) {
    if (!valid_lane_ecs(group_lane, road)) return -1;
    int right = rightmost_lane_in_group_ecs(group_lane, road);
    int left = leftmost_lane_in_group_ecs(group_lane, road);
    if (!valid_lane_ecs(right, road)) right = group_lane;
    if (!valid_lane_ecs(left, road)) left = group_lane;
    float dr = lane_endpoint_dist2_ecs(right, px, py, at_end, road);
    float dl = lane_endpoint_dist2_ecs(left, px, py, at_end, road);
    return dr <= dl ? right : left;
}
AVABM_DINLINE bool lane_groups_same_ecs(int a, int b, const RoadNetwork road) {
    if (!valid_lane_ecs(a, road) || !valid_lane_ecs(b, road)) return false;
    if (a == b) return true;
    if (road.lane_start_node[a] != road.lane_start_node[b]) return false;
    if (road.lane_end_node[a] != road.lane_end_node[b]) return false;
    float ah = lane_heading(a, road);
    float bh = lane_heading(b, road);
    return fabsf(wrap_pi(bh - ah)) < 0.35f;
}
AVABM_DINLINE int lane_group_count_ecs(int lane, const RoadNetwork road) {
    int idx = -1;
    return lane_group_count_and_index_ecs(lane, road, idx);
}
AVABM_DINLINE bool route_lane_current_compatible_ecs(int route_lane, int current_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(route_lane, road) || !valid_lane_ecs(current_lane, road)) return false;
    if (route_lane == current_lane) return true;
    return lane_groups_same_ecs(route_lane, current_lane, road);
}
AVABM_DINLINE int repair_route_pos_for_current_lane_ecs(int current_lane, int route_id, int route_pos, const RoadNetwork road) {
#if ROUTE_MISMATCH_REPAIR_ENABLED
    if (!valid_lane_ecs(current_lane, road)) return -1;
    if (route_id < 0 || route_id >= road.num_routes) return -1;
    int ro0 = road.route_offsets[route_id];
    int ro1 = road.route_offsets[route_id + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || route_len > 4096) return -1;
    if (route_pos >= 0 && route_pos < route_len) {
        int route_lane = road.route_lanes[ro0 + route_pos];
        if (route_lane_current_compatible_ecs(route_lane, current_lane, road)) return route_pos;
        int next_pos = route_pos + 1;
        if (next_pos < route_len) {
            int next_lane = road.route_lanes[ro0 + next_pos];
            if (valid_lane_ecs(next_lane, road) && lane_connected(current_lane, next_lane, road)) {
                return route_pos;
            }
        }
    }
    int best = -1;
    int best_score = 0x3fffffff;
    int scan = min(route_len, ROUTE_POS_REPAIR_SCAN_MAX);
    int lo = max(0, route_pos - scan / 3);
    int hi = min(route_len - 1, route_pos + scan);
    if (route_pos < 0 || route_pos >= route_len) {
        lo = 0;
        hi = min(route_len - 1, scan - 1);
    }
    for (int k = lo; k <= hi; ++k) {
        int route_lane = road.route_lanes[ro0 + k];
        if (!route_lane_current_compatible_ecs(route_lane, current_lane, road)) continue;
        int d = k - route_pos;
        if (d < 0) d = -d + 3;
        if (d < best_score) {
            best_score = d;
            best = k;
        }
    }
    if (best >= 0) return best;
    for (int k = lo; k < hi; ++k) {
        int next_lane = road.route_lanes[ro0 + k + 1];
        if (valid_lane_ecs(next_lane, road) && lane_connected(current_lane, next_lane, road)) {
            return k;
        }
    }
#endif
    return -1;
}
AVABM_DINLINE bool missed_exit_tail_sentinel_active_ecs(int current_lane, int route_id, int route_pos, const RoadNetwork road) {
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
    if (!valid_lane_ecs(current_lane, road)) return false;
    if (route_id < 0 || route_id >= road.num_routes) return false;
    int ro0 = road.route_offsets[route_id];
    int ro1 = road.route_offsets[route_id + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || route_pos < 0 || route_pos >= route_len) return false;
    if (route_pos < route_len - 1) return false;
    int tail_lane = road.route_lanes[ro0 + route_pos];
    return !route_lane_current_compatible_ecs(tail_lane, current_lane, road);
#else
    (void)current_lane;
    (void)route_id;
    (void)route_pos;
    (void)road;
    return false;
#endif
}
AVABM_DINLINE int repair_route_pos_unless_missed_exit_tail_ecs(int current_lane, int route_id, int route_pos,
        const RoadNetwork road) {
    if (missed_exit_tail_sentinel_active_ecs(current_lane, route_id, route_pos, road)) return -1;
    return repair_route_pos_for_current_lane_ecs(current_lane, route_id, route_pos, road);
}
AVABM_DINLINE int connected_equivalent_lane_from_group_ecs(int from_lane, int group_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(group_lane, road)) return -1;
    int right = rightmost_lane_in_group_ecs(group_lane, road);
    if (!valid_lane_ecs(right, road)) right = group_lane;
    int cur = right;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP && valid_lane_ecs(cur, road); ++k) {
        if (lane_connected(from_lane, cur, road)) return cur;
        int nb = geometric_left_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return -1;
}
AVABM_DINLINE int interchange_receiving_outer_lane_ecs(int from_lane, int candidate_lane, const RoadNetwork road) {
#if INTERCHANGE_EDGE_ONLY_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(candidate_lane, road)) return candidate_lane;
    if (!lane_connected(from_lane, candidate_lane, road)) return candidate_lane;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(candidate_lane, road);
    if (from_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES && to_count >= INTERCHANGE_MAIN_MIN_GROUP_LANES) {
        float px = road.lane_end_x[from_lane];
        float py = road.lane_end_y[from_lane];
        int edge = nearest_outer_lane_to_point_ecs(candidate_lane, px, py, false, road);
        if (valid_lane_ecs(edge, road) && lane_connected(from_lane, edge, road)) return edge;
    }
#endif
    return candidate_lane;
}
AVABM_DINLINE int interchange_source_outer_lane_ecs(int from_lane, int to_lane, const RoadNetwork road) {
#if INTERCHANGE_EDGE_ONLY_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return -1;
    if (!lane_connected(from_lane, to_lane, road)) return -1;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    if (from_count >= INTERCHANGE_MAIN_MIN_GROUP_LANES && to_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES) {
        float px = road.lane_start_x[to_lane];
        float py = road.lane_start_y[to_lane];
        int edge = nearest_outer_lane_to_point_ecs(from_lane, px, py, true, road);
        if (valid_lane_ecs(edge, road)) return edge;
    }
#endif
    return -1;
}
AVABM_DINLINE bool wide_lane_count_change_continuation_ecs(int from_lane, int to_lane, const RoadNetwork road) {
#if LANE_COUNT_CHANGE_CONTINUATION_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;
    if (!lane_connected(from_lane, to_lane, road)) return false;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    if (from_count < 2 || to_count < 2 || from_count == to_count) return false;
    if (from_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES || to_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES) return false;
    float deg = fabsf(lane_signed_turn_deg(from_lane, to_lane, road));
    return deg <= LANE_COUNT_CHANGE_MAX_TURN_DEG;
#else
    return false;
#endif
}
AVABM_DINLINE bool wide_group_continuation_candidate_ecs(int from_lane, int candidate_group_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(candidate_group_lane, road)) return false;
    if (road.lane_end_node[from_lane] != road.lane_start_node[candidate_group_lane]) return false;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(candidate_group_lane, road);
    if (from_count < 2 || to_count < 2) return false;
    if (from_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES || to_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES) return false;
    float deg = fabsf(lane_signed_turn_deg(from_lane, candidate_group_lane, road));
    return deg <= LANE_COUNT_CHANGE_MAX_TURN_DEG;
}
AVABM_DINLINE int effective_turn_code_ecs(int from_lane, int to_lane, int route_turn, const RoadNetwork road) {
    int geom_turn = turn_code_from_lanes_ecs(from_lane, to_lane, road);
    if (wide_lane_count_change_continuation_ecs(from_lane, to_lane, road) || (route_turn == TURN_STRAIGHT &&
            wide_group_continuation_candidate_ecs(from_lane, to_lane, road))) {
        return TURN_STRAIGHT;
    }
    if (geom_turn != TURN_STRAIGHT || route_turn == TURN_STRAIGHT) return geom_turn;
    return route_turn;
}
AVABM_DINLINE int lane_count_reduction_drop_side_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (!wide_lane_count_change_continuation_ecs(from_lane, to_lane, road)) return 0;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    if (from_count <= to_count) return 0;
    int from_right = rightmost_lane_in_group_ecs(from_lane, road);
    int from_left = leftmost_lane_in_group_ecs(from_lane, road);
    int to_right = rightmost_lane_in_group_ecs(to_lane, road);
    int to_left = leftmost_lane_in_group_ecs(to_lane, road);
    if (!valid_lane_ecs(from_right, road) || !valid_lane_ecs(from_left, road) || !valid_lane_ecs(to_right, road) ||
            !valid_lane_ecs(to_left, road)) {
        return 0;
    }
    float dr = lane_endpoint_dist2_ecs(from_right, road.lane_start_x[to_right], road.lane_start_y[to_right], true, road);
    float dl = lane_endpoint_dist2_ecs(from_left, road.lane_start_x[to_left], road.lane_start_y[to_left], true, road);
    float eps2 = LANE_COUNT_CHANGE_EDGE_ALIGN_EPS * LANE_COUNT_CHANGE_EDGE_ALIGN_EPS;
    if (dr > dl + eps2) return -1;
    if (dl > dr + eps2) return 1;
    return 0;
}
AVABM_DINLINE int lane_count_gain_side_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (!wide_lane_count_change_continuation_ecs(from_lane, to_lane, road)) return 0;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    if (from_count >= to_count) return 0;
    int from_right = rightmost_lane_in_group_ecs(from_lane, road);
    int from_left = leftmost_lane_in_group_ecs(from_lane, road);
    int to_right = rightmost_lane_in_group_ecs(to_lane, road);
    int to_left = leftmost_lane_in_group_ecs(to_lane, road);
    if (!valid_lane_ecs(from_right, road) || !valid_lane_ecs(from_left, road) || !valid_lane_ecs(to_right, road) ||
            !valid_lane_ecs(to_left, road)) {
        return 0;
    }
    float dr = lane_endpoint_dist2_ecs(from_right, road.lane_start_x[to_right], road.lane_start_y[to_right], true, road);
    float dl = lane_endpoint_dist2_ecs(from_left, road.lane_start_x[to_left], road.lane_start_y[to_left], true, road);
    float eps2 = LANE_COUNT_CHANGE_EDGE_ALIGN_EPS * LANE_COUNT_CHANGE_EDGE_ALIGN_EPS;
    if (dr > dl + eps2) return -1;
    if (dl > dr + eps2) return 1;
#if LANE_GAIN_AMBIGUOUS_RIGHT_EDGE_FALLBACK
    return -1;
#else
    return 0;
#endif
}
AVABM_DINLINE int lane_count_reduction_step_target_ecs(int lane, int next_lane, const RoadNetwork road) {
#if LANE_COUNT_CHANGE_PREP_ENABLED
    if (!valid_lane_ecs(lane, road) || !valid_lane_ecs(next_lane, road)) return -1;
    if (!wide_lane_count_change_continuation_ecs(lane, next_lane, road)) return -1;
    int cur_idx = -1;
    int from_count = lane_group_count_and_index_ecs(lane, road, cur_idx);
    int to_count = lane_group_count_ecs(next_lane, road);
    if (from_count <= to_count || from_count <= 1 || to_count <= 0 || cur_idx < 0) return -1;
    int drop = min(from_count - to_count, from_count - 1);
    int side = lane_count_reduction_drop_side_ecs(lane, next_lane, road);
#if LANE_DROP_AMBIGUOUS_RIGHT_EDGE_FALLBACK
    if (side == 0) side = -1;
#endif
    int target = -1;
    if (side < 0 && cur_idx < drop) {
        target = geometric_left_neighbor_ecs(lane, road);
    } else if (side > 0 && cur_idx >= from_count - drop) {
        target = geometric_right_neighbor_ecs(lane, road);
    }
    if (valid_lane_ecs(target, road) && lane_groups_same_ecs(lane, target, road)) {
        return target;
    }
#endif
    return -1;
}
AVABM_DINLINE bool lane_count_merge_transition_ecs(int from_lane, int to_lane, const RoadNetwork road) {
#if LANE_DROP_MERGE_GATE_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;
    if (!lane_connected(from_lane, to_lane, road)) return false;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    return from_count >= 2 && to_count >= 2 && from_count > to_count &&
            wide_lane_count_change_continuation_ecs(from_lane, to_lane, road);
#else
    return false;
#endif
}
AVABM_DINLINE bool lane_count_merge_pair_conflict_ecs(int a_from, int a_to, int b_from, int b_to, const RoadNetwork road) {
#if LANE_DROP_MERGE_GATE_ENABLED
    if (!valid_lane_ecs(a_from, road) || !valid_lane_ecs(a_to, road) || !valid_lane_ecs(b_from, road) ||
            !valid_lane_ecs(b_to, road)) return false;
    if (road.lane_end_node[a_from] != road.lane_end_node[b_from]) return false;
    bool a_merge = lane_count_merge_transition_ecs(a_from, a_to, road);
    bool b_merge = lane_count_merge_transition_ecs(b_from, b_to, road);
    if (!a_merge && !b_merge) return false;
    if (!lane_groups_same_ecs(a_from, b_from, road) && a_from != b_from) return false;
    if (a_to == b_to) return true;
#if LANE_DROP_TAPER_SAME_TARGET_ONLY
    return false;
#else
    return lane_groups_same_ecs(a_to, b_to, road);
#endif
#else
    return false;
#endif
}
AVABM_DINLINE int balanced_receiving_index_for_lane_count_change_ecs(int from_lane, int group_lane, const RoadNetwork road) {
    int from_idx = -1;
    int from_count = lane_group_count_and_index_ecs(from_lane, road, from_idx);
    int target_idx_dummy = -1;
    int to_count = lane_group_count_and_index_ecs(group_lane, road, target_idx_dummy);
    if (from_count <= 0 || to_count <= 0 || from_idx < 0) return -1;
    if (to_count == 1) return 0;
    if (from_count == 1) return clampi_cuda(target_idx_dummy >= 0 ? target_idx_dummy : 0, 0, to_count - 1);
#if LANE_DROP_RECEIVING_BALANCE_ENABLED
    if (from_count > to_count && wide_lane_count_change_continuation_ecs(from_lane, group_lane, road)) {
        int drop = min(from_count - to_count, from_count - 1);
        int side = lane_count_reduction_drop_side_ecs(from_lane, group_lane, road);
#if LANE_DROP_AMBIGUOUS_RIGHT_EDGE_FALLBACK
        if (side == 0) side = -1;
#endif
        if (side < 0) {
            return clampi_cuda(from_idx - drop, 0, to_count - 1);
        }
        if (side > 0) {
            return clampi_cuda(from_idx, 0, to_count - 1);
        }
    }
#endif
#if LANE_GAIN_RECEIVING_BALANCE_ENABLED
    if (from_count < to_count && wide_lane_count_change_continuation_ecs(from_lane, group_lane, road)) {
        int add = min(to_count - from_count, to_count - 1);
        int side = lane_count_gain_side_ecs(from_lane, group_lane, road);
#if LANE_GAIN_AMBIGUOUS_RIGHT_EDGE_FALLBACK
        if (side == 0) side = -1;
#endif
        if (side < 0) {
            return clampi_cuda(from_idx + add, 0, to_count - 1);
        }
        if (side > 0) {
            return clampi_cuda(from_idx, 0, to_count - 1);
        }
    }
#endif
    float denom = fmaxf(1.0f, (float)(from_count - 1));
    int mapped = (int)floorf(((float)from_idx * (float)(to_count - 1) / denom) + 0.5f);
    return clampi_cuda(mapped, 0, to_count - 1);
}
AVABM_DINLINE int connected_balanced_equivalent_lane_from_group_ecs(int from_lane, int group_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(group_lane, road)) return -1;
    int target_count = lane_group_count_ecs(group_lane, road);
    if (target_count <= 0) return -1;
    int mapped_idx = balanced_receiving_index_for_lane_count_change_ecs(from_lane, group_lane, road);
    if (mapped_idx < 0) {
        int from_idx = -1;
        int from_count = lane_group_count_and_index_ecs(from_lane, road, from_idx);
        if (from_count > 1 && from_idx >= 0 && target_count > 1) {
            mapped_idx = (int)floorf(((float)from_idx * (float)(target_count - 1) / fmaxf(1.0f, (float)(from_count - 1))) + 0.5f);
        } else {
            mapped_idx = 0;
        }
        mapped_idx = clampi_cuda(mapped_idx, 0, target_count - 1);
    }
    for (int radius = 0; radius < CRUISE_RANDOM_LANE_MAX_GROUP; ++radius) {
        int idxs[2] = {
            mapped_idx - radius, mapped_idx + radius
        };
        for (int q = 0; q < 2; ++q) {
            if (radius == 0 && q == 1) continue;
            int idx = idxs[q];
            if (idx < 0 || idx >= target_count) continue;
            int cand = lane_at_right_to_left_index_ecs(group_lane, idx, road);
            if (valid_lane_ecs(cand, road) && lane_connected(from_lane, cand, road)) return cand;
        }
    }
    return -1;
}
AVABM_DINLINE int lane_steps_to_specific_lane_ecs(int lane, int target_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road) || !valid_lane_ecs(target_lane, road)) return 99;
    if (lane == target_lane) return 0;
    if (!lane_groups_same_ecs(lane, target_lane, road)) return 99;
    int cur_idx = -1;
    int target_idx = -1;
    int cur_count = lane_group_count_and_index_ecs(lane, road, cur_idx);
    int target_count = lane_group_count_and_index_ecs(target_lane, road, target_idx);
    if (cur_count <= 0 || target_count <= 0 || cur_idx < 0 || target_idx < 0) return 99;
    int d = target_idx - cur_idx;
    return d < 0 ? -d : d;
}
AVABM_DINLINE int adjacent_lane_toward_specific_lane_ecs(int lane, int target_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road) || !valid_lane_ecs(target_lane, road)) return -1;
    if (lane == target_lane) return lane;
    if (!lane_groups_same_ecs(lane, target_lane, road)) return -1;
    int cur_idx = -1;
    int target_idx = -1;
    int cur_count = lane_group_count_and_index_ecs(lane, road, cur_idx);
    int target_count = lane_group_count_and_index_ecs(target_lane, road, target_idx);
    if (cur_count <= 0 || target_count <= 0 || cur_idx < 0 || target_idx < 0) return -1;
    if (target_idx > cur_idx) return geometric_left_neighbor_ecs(lane, road);
    if (target_idx < cur_idx) return geometric_right_neighbor_ecs(lane, road);
    return lane;
}
AVABM_DINLINE int random_cruise_lane_step_target_ecs(int id, int lane, ECSArrays ecs, const RoadNetwork road) {
    int cur_index = -1;
    int count = lane_group_count_and_index_ecs(lane, road, cur_index);
    if (count <= 1 || cur_index < 0) return -1;
    uint32_t key = ((uint32_t)(id + 1) * 747796405u) ^ ((uint32_t)(ecs.route_id[id] + 101) * 2891336453u) ^
            ((uint32_t)(ecs.route_pos[id] + 17) * 277803737u) ^ ((uint32_t)(road.lane_start_node[lane] + 3) * 1442695041u) ^
            ((uint32_t)(road.lane_end_node[lane] + 5) * 1597334677u);
    uint32_t h = hash_u32_ecs(key);
    int target_index = 0;
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
    if (count >= 3) {
        target_index = 1 + (int)(h % (uint32_t)(count - 1));
    } else {
        target_index = (int)(h % (uint32_t)count);
    }
#else
    target_index = (int)(h % (uint32_t)count);
#endif
    if (target_index == cur_index) return -1;
    if (target_index > cur_index) return geometric_left_neighbor_ecs(lane, road);
    return geometric_right_neighbor_ecs(lane, road);
}
AVABM_DINLINE int missed_exit_straight_fallback_lane_ecs(int id, int from_lane, int requested_lane, const RoadNetwork road) {
#if MISSED_EXIT_STRAIGHT_FALLBACK_ENABLED
    if (!valid_lane_ecs(from_lane, road)) return -1;
    int from_idx = -1;
    int from_count = lane_group_count_and_index_ecs(from_lane, road, from_idx);
    int best = -1;
    float best_score = 1.0e30f;
    uint32_t h = hash_u32_ecs(((uint32_t)(id + 1) * 747796405u) ^ ((uint32_t)(from_lane + 17) * 2891336453u) ^
            ((uint32_t)(road.lane_end_node[from_lane] + 31) * 277803737u));
    for (int cand0 = 0; cand0 < road.num_lanes; ++cand0) {
        if (!valid_lane_ecs(cand0, road) || !lane_connected(from_lane, cand0, road)) continue;
        if (cand0 == requested_lane) continue;
        float deg0 = fabsf(lane_signed_turn_deg(from_lane, cand0, road));
        bool continuation = wide_group_continuation_candidate_ecs(from_lane, cand0, road);
        if (!continuation && deg0 > MISSED_EXIT_STRAIGHT_MAX_DEG) continue;
        int cand = cand0;
        int balanced = connected_balanced_equivalent_lane_from_group_ecs(from_lane, cand0, road);
        if (valid_lane_ecs(balanced, road) && lane_connected(from_lane, balanced, road)) {
            cand = balanced;
        }
        if (cand == requested_lane) continue;
        float deg = fabsf(lane_signed_turn_deg(from_lane, cand, road));
        if (!continuation && deg > MISSED_EXIT_STRAIGHT_MAX_DEG) continue;
        int to_idx = -1;
        int to_count = lane_group_count_and_index_ecs(cand, road, to_idx);
        if (to_count <= 0 || to_idx < 0) continue;
        float score = deg;
        if (from_count >= INTERCHANGE_MAIN_MIN_GROUP_LANES && to_count <= INTERCHANGE_RAMP_MAX_GROUP_LANES) {
            score += MISSED_EXIT_STRAIGHT_RAMP_PENALTY;
        }
        score -= fminf((float)to_count, 6.0f) * 2.0f;
        if (to_count >= 2) {
            int target_idx = (int)(h % (uint32_t)to_count);
            score += fabsf((float)(to_idx - target_idx)) * 3.25f;
            if (to_count >= 3 && to_idx == 0) score += MISSED_EXIT_STRAIGHT_RIGHT_EDGE_PENALTY;
        }
        if (continuation) score -= 9.0f;
        if (from_idx >= 0 && to_count > 1 && from_count > 1) {
            int rel_idx = (int)floorf(((float)from_idx * (float)(to_count - 1) / fmaxf(1.0f, (float)(from_count - 1))) + 0.5f);
            score += fabsf((float)(to_idx - clampi_cuda(rel_idx, 0, to_count - 1))) * 1.10f;
        }
        if (score < best_score) {
            best_score = score;
            best = cand;
        }
    }
    return best;
#else
    (void)id;
    (void)from_lane;
    (void)requested_lane;
    (void)road;
    return -1;
#endif
}
AVABM_DINLINE bool turn_requires_dedicated_lane_ecs(int turn) {
    return turn == TURN_LEFT || turn == TURN_RIGHT;
}
AVABM_DINLINE bool lane_legal_for_turn_ecs(int lane, int turn, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return false;
    if (turn == TURN_LEFT) return is_leftmost_lane_ecs(lane, road);
    if (turn == TURN_RIGHT) return is_rightmost_lane_ecs(lane, road);
    return true;
}
AVABM_DINLINE int adjacent_lane_toward_turn_lane_ecs(int lane, int turn, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    if (turn == TURN_LEFT) return geometric_left_neighbor_ecs(lane, road);
    if (turn == TURN_RIGHT) return geometric_right_neighbor_ecs(lane, road);
    return -1;
}
AVABM_DINLINE int receiving_lane_for_turn_ecs(int candidate_lane, int turn, const RoadNetwork road) {
    if (!valid_lane_ecs(candidate_lane, road)) return candidate_lane;
    int cur = candidate_lane;
    if (turn == TURN_RIGHT) {
        for (int k = 0; k < 12; ++k) {
            int nb = geometric_right_neighbor_ecs(cur, road);
            if (!valid_lane_ecs(nb, road)) break;
            cur = nb;
        }
        return cur;
    }
    if (turn == TURN_LEFT) {
        for (int k = 0; k < 12; ++k) {
            int nb = geometric_left_neighbor_ecs(cur, road);
            if (!valid_lane_ecs(nb, road)) break;
            cur = nb;
        }
        return cur;
    }
    return candidate_lane;
}
AVABM_DINLINE bool missed_exit_straight_target_ecs(int id, int from_lane, int target_lane, ECSArrays ecs, const RoadNetwork road) {
#if MISSED_EXIT_STRAIGHT_FALLBACK_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(target_lane, road)) return false;
    if (!lane_connected(from_lane, target_lane, road)) return false;
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    if (rid < 0 || rid >= road.num_routes) return false;
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || rpos < 0 || rpos >= route_len) return false;
    int next_pos = rpos + 1;
    if (next_pos < 0 || next_pos >= route_len) {
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
        int route_lane = road.route_lanes[ro0 + rpos];
        if (!route_lane_current_compatible_ecs(route_lane, from_lane, road)) {
            int fallback_tail = missed_exit_straight_fallback_lane_ecs(id, from_lane, -1, road);
            return valid_lane_ecs(fallback_tail, road) && fallback_tail == target_lane;
        }
#endif
        return false;
    }
    int requested = road.route_lanes[ro0 + next_pos];
    int fallback = missed_exit_straight_fallback_lane_ecs(id, from_lane, requested, road);
    if (valid_lane_ecs(fallback, road) && fallback == target_lane) return true;
    if (valid_lane_ecs(requested, road) && lane_connected(from_lane, requested, road)) {
        int route_turn = road.route_turns[ro0 + rpos];
        int turn = effective_turn_code_ecs(from_lane, requested, route_turn, road);
        int adjusted = interchange_receiving_outer_lane_ecs(from_lane, requested, road);
        adjusted = receiving_lane_for_turn_ecs(adjusted, turn, road);
        adjusted = interchange_receiving_outer_lane_ecs(from_lane, adjusted, road);
        if (valid_lane_ecs(adjusted, road)) {
            int fallback2 = missed_exit_straight_fallback_lane_ecs(id, from_lane, adjusted, road);
            if (valid_lane_ecs(fallback2, road) && fallback2 == target_lane) return true;
        }
    }
    return false;
#else
    (void)id;
    (void)from_lane;
    (void)target_lane;
    (void)ecs;
    (void)road;
    return false;
#endif
}
AVABM_DINLINE void prepare_missed_exit_route_tail_ecs(int id, int from_lane, int fallback_lane, ECSArrays ecs,
        const RoadNetwork road) {
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(fallback_lane, road)) return;
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    if (rid < 0 || rid >= road.num_routes) return;
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 1 || rpos < 0 || rpos >= route_len) return;
    int scan_hi = min(route_len, rpos + ROUTE_NEXT_LANE_EQUIV_SCAN + 3);
    for (int pos = rpos + 1; pos < scan_hi; ++pos) {
        int route_lane = road.route_lanes[ro0 + pos];
        if (route_lane_current_compatible_ecs(route_lane, fallback_lane, road)) {
            ecs.route_pos[id] = max(0, pos - 1);
            return;
        }
    }
    ecs.route_pos[id] = max(0, route_len - 2);
#else
    (void)id;
    (void)from_lane;
    (void)fallback_lane;
    (void)ecs;
    (void)road;
#endif
}
AVABM_DINLINE bool abandon_destination_to_straight_or_tail_ecs(int id, int lane, int requested_lane, ECSArrays ecs,
        const RoadNetwork road, int& out_next_lane, bool& out_has_next, bool& out_missed_exit_straight) {
#if MISSION_ABANDON_ENABLED
    out_next_lane = -1;
    out_has_next = false;
    out_missed_exit_straight = false;
    if (!valid_lane_ecs(lane, road)) return false;
    int straight_escape = missed_exit_straight_fallback_lane_ecs(id, lane, requested_lane, road);
    if (valid_lane_ecs(straight_escape, road) && lane_connected(lane, straight_escape, road)) {
        out_next_lane = straight_escape;
        out_has_next = true;
        out_missed_exit_straight = true;
        return true;
    }
    int rid = ecs.route_id[id];
    if (rid < 0 || rid >= road.num_routes) return false;
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || route_len > 4096) return false;
    ecs.route_pos[id] = route_len - 1;
    ecs.lane_change_active[id] = 0;
    ecs.lane_change_from_lane[id] = lane;
    ecs.lane_change_to_lane[id] = lane;
    ecs.lane_change_t[id] = 0.0f;
    ecs.lc_cooldown[id] = fmaxf(ecs.lc_cooldown[id], 0.25f);
    ecs.connector_length[id] = 0.0f;
    if (ecs.turn_signal != nullptr) {
        ecs.turn_signal[id] = INDICATOR_NONE;
        if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[id] = 0.0f;
    }
    return true;
#else
    (void)id;
    (void)lane;
    (void)requested_lane;
    (void)ecs;
    (void)road;
    out_next_lane = -1;
    out_has_next = false;
    out_missed_exit_straight = false;
    return false;
#endif
}
AVABM_DINLINE int lane_steps_to_turn_lane_ecs(int lane, int turn, const RoadNetwork road) {
    if (!turn_requires_dedicated_lane_ecs(turn)) return 0;
    if (lane_legal_for_turn_ecs(lane, turn, road)) return 0;
    int cur = lane;
    for (int step = 1; step <= 16; ++step) {
        cur = adjacent_lane_toward_turn_lane_ecs(cur, turn, road);
        if (cur < 0 || cur >= road.num_lanes) return 99;
        if (lane_legal_for_turn_ecs(cur, turn, road)) return step;
    }
    return 99;
}
AVABM_DINLINE float turn_lane_prep_distance_ecs(int lane_steps, float speed, int dtype) {
    if (lane_steps <= 0) return 0.0f;
    float human_factor = dtype == HUMAN ? 1.28f : 1.0f;
    float dynamic = fmaxf(speed, 4.0f) * (dtype == HUMAN ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV);
    float d = TURN_LANE_PREP_BASE_DIST + TURN_LANE_PREP_PER_LANE_DIST * lane_steps + dynamic;
    return clampf_cuda(d * human_factor, 35.0f, 190.0f);
}
AVABM_DINLINE float turn_lane_hold_accel_ecs(float dist_to_end, float speed, int dtype) {
    float hold_dist = fmaxf(0.65f, dist_to_end - (DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER));
    float req = -(speed * speed) / fmaxf(2.0f * hold_dist, 0.5f);
    float max_b = dtype == HUMAN ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    return clampf_cuda(req, -EMERGENCY_DECEL, 0.0f) - 0.05f * max_b;
}
AVABM_DINLINE int upcoming_exit_lane_step_target_ecs(int id, int lane, ECSArrays ecs, const RoadNetwork road, float dist_to_end,
        float speed, int dtype, int* out_event_turn, float* out_event_distance) {
#if UPCOMING_EXIT_LANE_PREP_ENABLED
    if (out_event_turn != nullptr) *out_event_turn = TURN_STRAIGHT;
    if (out_event_distance != nullptr) *out_event_distance = 1.0e9f;
    if (!valid_lane_ecs(lane, road) || ecs.vehicle_state[id] != VEH_ON_LANE) return -1;
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    if (rid < 0 || rid >= road.num_routes) return -1;
    int repaired_pos = repair_route_pos_unless_missed_exit_tail_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[id] = repaired_pos;
    }
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 1 || rpos < 0 || rpos >= route_len - 1) return -1;
    float event_dist = fmaxf(0.0f, dist_to_end);
    int max_k = min(route_len - 2, rpos + UPCOMING_EXIT_LOOKAHEAD_LANES);
    for (int k = rpos; k <= max_k; ++k) {
        int src_lane = road.route_lanes[ro0 + k];
        int dst_lane = road.route_lanes[ro0 + k + 1];
        if (!valid_lane_ecs(src_lane, road) || !valid_lane_ecs(dst_lane, road)) break;
        int route_turn = road.route_turns[ro0 + k];
        int event_turn = effective_turn_code_ecs(src_lane, dst_lane, route_turn, road);
        int edge_source_lane = interchange_source_outer_lane_ecs(src_lane, dst_lane, road);
        bool mandatory_event = valid_lane_ecs(edge_source_lane, road) || event_turn == TURN_LEFT || event_turn == TURN_RIGHT;
        if (mandatory_event) {
            bool wants_right = event_turn == TURN_RIGHT;
            if (valid_lane_ecs(edge_source_lane, road)) {
                int edge_idx = -1;
                int edge_count = lane_group_count_and_index_ecs(edge_source_lane, road, edge_idx);
                if (edge_count > 1 && edge_idx >= 0) {
                    wants_right = edge_idx <= (edge_count - 1) / 2;
                } else {
                    float px = road.lane_start_x[dst_lane];
                    float py = road.lane_start_y[dst_lane];
                    int right_edge = rightmost_lane_in_group_ecs(src_lane, road);
                    int left_edge = leftmost_lane_in_group_ecs(src_lane, road);
                    float dr = lane_endpoint_dist2_ecs(right_edge, px, py, true, road);
                    float dl = lane_endpoint_dist2_ecs(left_edge, px, py, true, road);
                    wants_right = dr <= dl;
                }
            }
            int target_edge_on_current = wants_right ? rightmost_lane_in_group_ecs(lane, road) : leftmost_lane_in_group_ecs(lane,
                    road);
            if (!valid_lane_ecs(target_edge_on_current, road) || target_edge_on_current == lane) return -1;
            if (!lane_groups_same_ecs(lane, target_edge_on_current, road)) return -1;
            int steps = lane_steps_to_specific_lane_ecs(lane, target_edge_on_current, road);
            if (steps <= 0 || steps >= 99) return -1;
            float prep = turn_lane_prep_distance_ecs(steps, speed, dtype) + UPCOMING_EXIT_PREP_EXTRA_DIST;
            prep = clampf_cuda(prep, 70.0f, UPCOMING_EXIT_PREP_MAX_DIST);
            if (event_dist <= prep) {
                if (out_event_turn != nullptr) *out_event_turn = event_turn;
                if (out_event_distance != nullptr) *out_event_distance = event_dist;
                return adjacent_lane_toward_specific_lane_ecs(lane, target_edge_on_current, road);
            }
            return -1;
        }
        event_dist += fmaxf(road.lane_length[dst_lane], 0.1f);
    }
#endif
    return -1;
}
AVABM_DINLINE int route_next_lane_for_vehicle_ecs(int id, ECSArrays ecs, const RoadNetwork road) {
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        int to_ln = ecs.connector_to_lane[id];
        return valid_lane_ecs(to_ln, road) ? to_ln : -1;
    }
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    int lane = ecs.lane_id[id];
    if (rid < 0 || rid >= road.num_routes) return -1;
    if (!valid_lane_ecs(lane, road)) return -1;
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || route_len > 4096) return -1;
    bool skip_tail_repair = false;
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
    if (rpos >= 0 && rpos < route_len && rpos >= route_len - 1) {
        int tail_lane = road.route_lanes[ro0 + rpos];
        skip_tail_repair = !route_lane_current_compatible_ecs(tail_lane, lane, road);
    }
#endif
    if (!skip_tail_repair) {
        int repaired_pos = repair_route_pos_unless_missed_exit_tail_ecs(lane, rid, rpos, road);
        if (repaired_pos >= 0) {
            rpos = repaired_pos;
            ecs.route_pos[id] = repaired_pos;
        }
    }
    if (rpos < 0 || rpos >= route_len) return -1;
    int next_pos = rpos + 1;
    if (next_pos < 0 || next_pos >= route_len) {
        return -1;
    }
    int scan_hi = min(route_len, next_pos + ROUTE_NEXT_LANE_EQUIV_SCAN);
    for (int pos = next_pos; pos < scan_hi; ++pos) {
        int candidate = road.route_lanes[ro0 + pos];
        if (!valid_lane_ecs(candidate, road)) continue;
        int route_turn = road.route_turns[ro0 + max(0, pos - 1)];
        int turn = effective_turn_code_ecs(lane, candidate, route_turn, road);
#if LANE_DROP_RECEIVING_BALANCE_ENABLED
        if (turn == TURN_STRAIGHT || wide_group_continuation_candidate_ecs(lane, candidate, road)) {
            int balanced = connected_balanced_equivalent_lane_from_group_ecs(lane, candidate, road);
            if (valid_lane_ecs(balanced, road) && lane_connected(lane, balanced, road)) {
                ecs.route_pos[id] = max(0, pos - 1);
                return balanced;
            }
        }
#endif
        int adjusted = interchange_receiving_outer_lane_ecs(lane, candidate, road);
        adjusted = receiving_lane_for_turn_ecs(adjusted, turn, road);
        adjusted = interchange_receiving_outer_lane_ecs(lane, adjusted, road);
        if (valid_lane_ecs(adjusted, road) && lane_connected(lane, adjusted, road)) {
            ecs.route_pos[id] = max(0, pos - 1);
            return adjusted;
        }
        if (lane_connected(lane, candidate, road)) {
            ecs.route_pos[id] = max(0, pos - 1);
            return candidate;
        }
        int equiv = connected_equivalent_lane_from_group_ecs(lane, candidate, road);
        if (valid_lane_ecs(equiv, road)) {
            int adj2 = interchange_receiving_outer_lane_ecs(lane, equiv, road);
            adj2 = receiving_lane_for_turn_ecs(adj2, turn, road);
            adj2 = interchange_receiving_outer_lane_ecs(lane, adj2, road);
            if (valid_lane_ecs(adj2, road) && lane_connected(lane, adj2, road)) equiv = adj2;
            if (lane_connected(lane, equiv, road)) {
                ecs.route_pos[id] = max(0, pos - 1);
                return equiv;
            }
        }
    }
#if MISSED_EXIT_STRAIGHT_FALLBACK_ENABLED
    {
        int requested = (next_pos >= 0 && next_pos < route_len) ? road.route_lanes[ro0 + next_pos] : -1;
        int fallback = missed_exit_straight_fallback_lane_ecs(id, lane, requested, road);
        if (valid_lane_ecs(fallback, road) && lane_connected(lane, fallback, road)) {
            ecs.route_pos[id] = max(0, next_pos - 1);
            return fallback;
        }
    }
#endif
    return -1;
}
AVABM_DINLINE int route_turn_for_vehicle_ecs(int id, ECSArrays ecs, const RoadNetwork road) {
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        return effective_turn_code_ecs(ecs.connector_from_lane[id], ecs.connector_to_lane[id], TURN_STRAIGHT, road);
    }
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    int lane = ecs.vehicle_state[id] == VEH_IN_CONNECTOR ? ecs.connector_from_lane[id] : ecs.lane_id[id];
    if (rid < 0 || rid >= road.num_routes) return TURN_STRAIGHT;
    int repaired_pos = repair_route_pos_unless_missed_exit_tail_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[id] = repaired_pos;
    }
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (rpos < 0 || rpos >= route_len) return TURN_STRAIGHT;
    int route_turn = road.route_turns[ro0 + rpos];
    int next_lane = route_next_lane_for_vehicle_ecs(id, ecs, road);
    rpos = ecs.route_pos[id];
    if (rpos >= 0 && rpos < route_len) route_turn = road.route_turns[ro0 + rpos];
    return effective_turn_code_ecs(lane, next_lane, route_turn, road);
}
AVABM_DINLINE int indicator_from_lateral_move_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return INDICATOR_NONE;
    float side = neighbor_lateral_cross_ecs(from_lane, to_lane, road);
    if (side > 0.05f) return INDICATOR_LEFT;
    if (side < -0.05f) return INDICATOR_RIGHT;
    return INDICATOR_NONE;
}
AVABM_DINLINE int indicator_state_ecs(int id, ECSArrays ecs) {
    if (ecs.turn_signal == nullptr) return INDICATOR_NONE;
    int v = ecs.turn_signal[id];
    if (v == INDICATOR_LEFT || v == INDICATOR_RIGHT || v == INDICATOR_HAZARD) return v;
    return INDICATOR_NONE;
}
AVABM_DINLINE bool indicator_active_ecs(int id, ECSArrays ecs) {
    return indicator_state_ecs(id, ecs) == INDICATOR_LEFT || indicator_state_ecs(id, ecs) == INDICATOR_RIGHT;
}
AVABM_DINLINE int indicator_target_lane_ecs(int lane, int signal, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return -1;
    if (signal == INDICATOR_LEFT) return geometric_left_neighbor_ecs(lane, road);
    if (signal == INDICATOR_RIGHT) return geometric_right_neighbor_ecs(lane, road);
    return -1;
}
AVABM_DINLINE bool indicator_matches_lateral_move_ecs(int from_lane, int to_lane, ECSArrays ecs, int id, const RoadNetwork road) {
    int needed = indicator_from_lateral_move_ecs(from_lane, to_lane, road);
    if (needed == INDICATOR_NONE) return true;
    return indicator_state_ecs(id, ecs) == needed;
}
AVABM_DINLINE int intended_turn_with_indicator_ecs(int id, int from_lane, int to_lane, ECSArrays ecs, const RoadNetwork road) {
    int route_turn = route_turn_for_vehicle_ecs(id, ecs, road);
    int turn = effective_turn_code_ecs(from_lane, to_lane, route_turn, road);
    int sig = indicator_state_ecs(id, ecs);
    if (ecs.vehicle_state[id] != VEH_IN_CONNECTOR) {
        if (sig == INDICATOR_LEFT) return TURN_LEFT;
        if (sig == INDICATOR_RIGHT) return TURN_RIGHT;
    }
    return turn;
}
AVABM_DINLINE float lane_heading_dot_ecs(int a, int b, const RoadNetwork road) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return 1.0f;
    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);
    return clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
}
AVABM_DINLINE bool same_approach_same_direction_lanes_ecs(int a, int b, const RoadNetwork road) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return false;
    if (a == b) return true;
    bool same_nodes = road.lane_start_node[a] == road.lane_start_node[b] && road.lane_end_node[a] == road.lane_end_node[b];
    return same_nodes && lane_heading_dot_ecs(a, b, road) > 0.82f;
}
AVABM_DINLINE bool opposite_corridor_lanes_ecs(int a, int b, const RoadNetwork road) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return false;
    bool reverse_nodes = road.lane_start_node[a] == road.lane_end_node[b] && road.lane_end_node[a] == road.lane_start_node[b];
    return reverse_nodes && lane_heading_dot_ecs(a, b, road) < -0.82f;
}
AVABM_DINLINE bool right_turn_corner_line_params_ecs(int from_lane, int to_lane, const RoadNetwork road, float& corner_x,
        float& corner_y, float& from_param, float& to_param) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;
    if (turn_code_from_lanes_ecs(from_lane, to_lane, road) != TURN_RIGHT) return false;
    float h0 = lane_heading(from_lane, road);
    float h3 = lane_heading(to_lane, road);
    float delta = wrap_pi(h3 - h0);
    if (delta >= -0.10f || fabsf(delta) > 2.45f) return false;
    float e0x = road.lane_end_x[from_lane];
    float e0y = road.lane_end_y[from_lane];
    float s1x = road.lane_start_x[to_lane];
    float s1y = road.lane_start_y[to_lane];
    float t0x = cosf(h0);
    float t0y = sinf(h0);
    float t1x = cosf(h3);
    float t1y = sinf(h3);
    float rx = s1x - e0x;
    float ry = s1y - e0y;
    float det = t0x * t1y - t0y * t1x;
    if (fabsf(det) < 1.0e-4f) return false;
    from_param = (rx * t1y - ry * t1x) / det;
    to_param = (rx * t0y - ry * t0x) / det;
    if (!isfinite(from_param) || !isfinite(to_param)) return false;
    if (fabsf(from_param) > 32.0f || fabsf(to_param) > 32.0f) return false;
    corner_x = e0x + t0x * from_param;
    corner_y = e0y + t0y * from_param;
    return true;
}
AVABM_DINLINE float connector_local_handoff_estimate_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (from_lane < 0 || from_lane >= road.num_lanes || to_lane < 0 || to_lane >= road.num_lanes) {
        return 0.0f;
    }
    float raw_dx = road.lane_start_x[to_lane] - road.lane_end_x[from_lane];
    float raw_dy = road.lane_start_y[to_lane] - road.lane_end_y[from_lane];
    float raw_chord = sqrtf(raw_dx * raw_dx + raw_dy * raw_dy);
    float corner_x, corner_y, from_param, to_param;
    if (right_turn_corner_line_params_ecs(from_lane, to_lane, road, corner_x, corner_y, from_param, to_param)) {
        float L_to = fmaxf(road.lane_length[to_lane], 0.1f);
        float lane_w = DEFAULT_LANE_WIDTH;
        float desired_radius = lane_w * 1.65f;
        float desired = to_param + desired_radius;
        float max_off = fminf(CONNECTOR_EXIT_OFFSET_MAX, L_to * 0.55f);
        max_off = fminf(max_off, fmaxf(0.0f, L_to - 0.75f));
        if (max_off > 0.05f) {
            return clampf_cuda(desired, fminf(CONNECTOR_EXIT_OFFSET_MIN, max_off), max_off);
        }
    }
    if (!isfinite(raw_chord) || raw_chord >= CONNECTOR_SAME_NODE_EPS) {
        return 0.0f;
    }
    float L_to = fmaxf(road.lane_length[to_lane], 0.1f);
    float max_off = fminf(CONNECTOR_EXIT_OFFSET_MAX, L_to * 0.45f);
    max_off = fminf(max_off, fmaxf(0.0f, L_to - 0.75f));
    if (max_off <= 0.05f) return 0.0f;
    float h0 = lane_heading(from_lane, road);
    float h1 = lane_heading(to_lane, road);
    float angle = fabsf(wrap_pi(h1 - h0)) * 57.2957795f;
    float desired = CONNECTOR_EXIT_OFFSET_BASE + 0.025f * angle;
    if (angle < 8.0f) desired = CONNECTOR_EXIT_OFFSET_BASE - 0.5f;
    float lo = fminf(CONNECTOR_EXIT_OFFSET_MIN, max_off);
    return clampf_cuda(desired, lo, max_off);
}
AVABM_DINLINE float connector_entry_backoff_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return 0.0f;
    int turn = turn_code_from_lanes_ecs(from_lane, to_lane, road);
    if (turn != TURN_RIGHT) return 0.0f;
    float deg = fabsf(lane_signed_turn_deg(from_lane, to_lane, road));
    float backoff = RIGHT_TURN_ENTRY_BACKOFF_BASE + RIGHT_TURN_ENTRY_BACKOFF_PER_DEG * deg;
    float corner_x, corner_y, from_param, to_param;
    if (right_turn_corner_line_params_ecs(from_lane, to_lane, road, corner_x, corner_y, from_param, to_param)) {
        backoff = fmaxf(backoff, -from_param + DEFAULT_LANE_WIDTH * 1.55f);
    }
    float lane_L = fmaxf(road.lane_length[from_lane], 0.1f);
    backoff = clampf_cuda(backoff, RIGHT_TURN_ENTRY_BACKOFF_MIN, RIGHT_TURN_ENTRY_BACKOFF_MAX);
    return fminf(backoff, fmaxf(0.0f, lane_L * 0.48f));
}
AVABM_DINLINE void connector_surface_endpoints_ecs(int from_lane, int to_lane, const RoadNetwork road, float& p0x, float& p0y,
        float& h0, float& p3x, float& p3y, float& h3, float& handoff_s) {
    h0 = lane_heading(from_lane, road);
    h3 = lane_heading(to_lane, road);
    float entry_backoff = connector_entry_backoff_ecs(from_lane, to_lane, road);
    p0x = road.lane_end_x[from_lane] - cosf(h0) * entry_backoff;
    p0y = road.lane_end_y[from_lane] - sinf(h0) * entry_backoff;
    handoff_s = connector_local_handoff_estimate_ecs(from_lane, to_lane, road);
    p3x = road.lane_start_x[to_lane] + cosf(h3) * handoff_s;
    p3y = road.lane_start_y[to_lane] + sinf(h3) * handoff_s;
}
AVABM_DINLINE float lane_width_estimate_ecs(int lane, const RoadNetwork road) {
    if (!valid_lane_ecs(lane, road)) return DEFAULT_LANE_WIDTH;
    float sx = 0.5f * (road.lane_start_x[lane] + road.lane_end_x[lane]);
    float sy = 0.5f * (road.lane_start_y[lane] + road.lane_end_y[lane]);
    float best = 1.0e9f;
    int nl = road.left_lane[lane];
    int nr = road.right_lane[lane];
    if (valid_lane_ecs(nl, road)) {
        float tx = 0.5f * (road.lane_start_x[nl] + road.lane_end_x[nl]);
        float ty = 0.5f * (road.lane_start_y[nl] + road.lane_end_y[nl]);
        float dx = tx - sx;
        float dy = ty - sy;
        best = fminf(best, sqrtf(dx * dx + dy * dy));
    }
    if (valid_lane_ecs(nr, road)) {
        float tx = 0.5f * (road.lane_start_x[nr] + road.lane_end_x[nr]);
        float ty = 0.5f * (road.lane_start_y[nr] + road.lane_end_y[nr]);
        float dx = tx - sx;
        float dy = ty - sy;
        best = fminf(best, sqrtf(dx * dx + dy * dy));
    }
    if (!isfinite(best) || best < 2.2f || best > 5.2f) return DEFAULT_LANE_WIDTH;
    return best;
}
AVABM_DINLINE float intersection_box_depth_ecs(int from_lane, int to_lane, const RoadNetwork road) {
    float lw = lane_width_estimate_ecs(from_lane, road);
    if (valid_lane_ecs(to_lane, road)) {
        lw = fminf(lw, lane_width_estimate_ecs(to_lane, road));
    }
    float entry = connector_entry_backoff_ecs(from_lane, to_lane, road);
    float d = fmaxf(lw * INTERSECTION_BOX_LANE_WIDTH_MULT, entry + lw * 0.35f);
    return clampf_cuda(d, INTERSECTION_BOX_MIN_DEPTH, INTERSECTION_BOX_MAX_DEPTH);
}
AVABM_DINLINE bool inside_intersection_box_ecs(float dist_to_end, int from_lane, int to_lane, const RoadNetwork road) {
    return dist_to_end <= intersection_box_depth_ecs(from_lane, to_lane, road) + INTERSECTION_BOX_ENTRY_MARGIN;
}
AVABM_DINLINE float lane_change_no_start_distance_ecs(int lane, const RoadNetwork road) {
    float lw = lane_width_estimate_ecs(lane, road);
    return fmaxf(LANE_CHANGE_NO_START_DIST_TO_NODE, lw * LANE_CHANGE_BOX_CLEAR_MULT);
}
AVABM_DINLINE float lane_change_finish_distance_ecs(int lane, const RoadNetwork road) {
    float lw = lane_width_estimate_ecs(lane, road);
    return fmaxf(LANE_CHANGE_FINISH_BEFORE_NODE, lw * (LANE_CHANGE_BOX_CLEAR_MULT * 0.62f));
}
AVABM_DINLINE bool connector_surface_arc_params_raw_ecs(float p0x, float p0y, float h0, float p3x, float p3y, float h3, float& cx,
        float& cy, float& radius, float& a0, float& sweep, float& arc_len) {
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);
    if (abs_delta < SURFACE_TURN_MIN_DELTA_RAD) return false;
    float sign = delta >= 0.0f ? 1.0f : -1.0f;
    float n0x = -sinf(h0);
    float n0y = cosf(h0);
    float n1x = -sinf(h3);
    float n1y = cosf(h3);
    float vx = sign * (n0x - n1x);
    float vy = sign * (n0y - n1y);
    float den = vx * vx + vy * vy;
    float dx = p3x - p0x;
    float dy = p3y - p0y;
    float chord = sqrtf(dx * dx + dy * dy);
    if (!isfinite(chord) || chord < 0.25f || den < 1.0e-5f) {
        return false;
    }
    float r_fit = (dx * vx + dy * vy) / den;
    float r_chord = chord / fmaxf(2.0f * sinf(abs_delta * 0.5f), 0.12f);
    if (!isfinite(r_fit) || r_fit < SURFACE_TURN_MIN_RADIUS * 0.55f) {
        r_fit = r_chord;
    }
    radius = clampf_cuda(r_fit, SURFACE_TURN_MIN_RADIUS, SURFACE_TURN_MAX_RADIUS);
    float c0x = p0x + sign * radius * n0x;
    float c0y = p0y + sign * radius * n0y;
    float c1x = p3x + sign * radius * n1x;
    float c1y = p3y + sign * radius * n1y;
    float fit_err = sqrtf((c1x - c0x) * (c1x - c0x) + (c1y - c0y) * (c1y - c0y));
    if (fit_err > fmaxf(SURFACE_TURN_ARC_FIT_TOL, radius * 0.42f)) {
        return false;
    }
    cx = 0.5f * (c0x + c1x);
    cy = 0.5f * (c0y + c1y);
    a0 = atan2f(p0y - cy, p0x - cx);
    float a1 = atan2f(p3y - cy, p3x - cx);
    sweep = a1 - a0;
    if (sign > 0.0f && sweep < 0.0f) sweep += 6.28318530718f;
    if (sign < 0.0f && sweep > 0.0f) sweep -= 6.28318530718f;
    if (fabsf(sweep) < 0.03f || fabsf(sweep) > SURFACE_TURN_MAX_SWEEP_RAD) {
        return false;
    }
    arc_len = fabsf(radius * sweep);
    return isfinite(arc_len) && arc_len >= CONNECTOR_MIN_LEN * 0.35f;
}
AVABM_DINLINE void connector_surface_fallback_bezier_ecs(int from_lane, int to_lane, float u, const RoadNetwork road, float& ox,
        float& oy, float& oh) {
    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);
    float span_dx = p3x - p0x;
    float span_dy = p3y - p0y;
    float span = sqrtf(span_dx * span_dx + span_dy * span_dy);
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);
    float scale = fminf(fmaxf(span * 0.32f, 1.2f), 7.0f);
    if (handoff_s > 0.0f) scale = fminf(scale, fmaxf(1.0f, handoff_s * 0.38f));
    float p1x = p0x + cosf(h0) * scale;
    float p1y = p0y + sinf(h0) * scale;
    float p2x = p3x - cosf(h3) * scale;
    float p2y = p3y - sinf(h3) * scale;
    if (abs_delta > 0.45f) {
        float sign = delta >= 0.0f ? 1.0f : -1.0f;
        float side_h = h0 + sign * 1.57079632679f;
        float bulge = fminf(fmaxf(span * 0.16f, 0.8f), 3.8f);
        p1x += cosf(side_h) * bulge * 0.14f;
        p1y += sinf(side_h) * bulge * 0.14f;
        p2x += cosf(side_h) * bulge * 0.38f;
        p2y += sinf(side_h) * bulge * 0.38f;
    }
    u = clampf_cuda(u, 0.0f, 1.0f);
    float w = 1.0f - u;
    float uu = u * u;
    float ww = w * w;
    ox = ww * w * p0x + 3.0f * ww * u * p1x + 3.0f * w * uu * p2x + uu * u * p3x;
    oy = ww * w * p0y + 3.0f * ww * u * p1y + 3.0f * w * uu * p2y + uu * u * p3y;
    float dxdt = 3.0f * ww * (p1x - p0x) + 6.0f * w * u * (p2x - p1x) + 3.0f * uu * (p3x - p2x);
    float dydt = 3.0f * ww * (p1y - p0y) + 6.0f * w * u * (p2y - p1y) + 3.0f * uu * (p3y - p2y);
    float interp_h = wrap_pi(h0 + wrap_pi(h3 - h0) * smoothstep01(u));
    float d2 = dxdt * dxdt + dydt * dydt;
    if (isfinite(d2) && d2 > 1.0e-5f) {
        float tan_h = atan2f(dydt, dxdt);
        if (fabsf(wrap_pi(tan_h - interp_h)) < 1.1f) {
            oh = wrap_pi(interp_h + 0.72f * wrap_pi(tan_h - interp_h));
        } else {
            oh = interp_h;
        }
    } else {
        oh = interp_h;
    }
}
AVABM_DINLINE bool connector_right_turn_corner_xy_heading_ecs(int from_lane, int to_lane, float u, const RoadNetwork road,
        float& ox, float& oy, float& oh) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;
    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);
    if (delta >= -0.18f || abs_delta > 2.30f) return false;
    float t0x = cosf(h0);
    float t0y = sinf(h0);
    float t3x = cosf(h3);
    float t3y = sinf(h3);
    float dx = p3x - p0x;
    float dy = p3y - p0y;
    float chord = sqrtf(dx * dx + dy * dy);
    if (!isfinite(chord) || chord < 0.45f) return false;
    float det = t0x * t3y - t0y * t3x;
    if (fabsf(det) < 1.0e-4f) return false;
    float a = (dx * t3y - dy * t3x) / det;
    float b = (t0x * dy - t0y * dx) / det;
    float max_param = fminf(RIGHT_TURN_FILLET_MAX_PARAM, fmaxf(2.5f, chord * 1.35f));
    float ac = clampf_cuda(a, RIGHT_TURN_FILLET_MIN_PARAM, max_param);
    float bc = clampf_cuda(b, RIGHT_TURN_FILLET_MIN_PARAM, max_param);
    float q1x = p0x + t0x * ac;
    float q1y = p0y + t0y * ac;
    float q2x = p3x - t3x * bc;
    float q2y = p3y - t3y * bc;
    float qx = 0.5f * (q1x + q2x);
    float qy = 0.5f * (q1y + q2y);
    float ux = dx / chord;
    float uy = dy / chord;
    float px = -uy;
    float py = ux;
    float qrx = qx - p0x;
    float qry = qy - p0y;
    float qproj = clampf_cuda(qrx * ux + qry * uy, 0.02f * chord, 0.98f * chord);
    float lane_w = fminf(lane_width_estimate_ecs(from_lane, road), lane_width_estimate_ecs(to_lane, road));
    float qlat_limit = clampf_cuda(fmaxf(lane_w * 0.62f, chord * RIGHT_TURN_CHORD_LATERAL_LIMIT), 1.10f, 4.40f);
    float qlat = clampf_cuda(qrx * px + qry * py, -qlat_limit, qlat_limit);
    qx = p0x + qproj * ux + qlat * px;
    qy = p0y + qproj * uy + qlat * py;
    u = clampf_cuda(u, 0.0f, 1.0f);
    float w = 1.0f - u;
    ox = w * w * p0x + 2.0f * w * u * qx + u * u * p3x;
    oy = w * w * p0y + 2.0f * w * u * qy + u * u * p3y;
    float dxdt = 2.0f * w * (qx - p0x) + 2.0f * u * (p3x - qx);
    float dydt = 2.0f * w * (qy - p0y) + 2.0f * u * (p3y - qy);
    float interp_h = wrap_pi(h0 + delta * smoothstep01(u));
    float d2 = dxdt * dxdt + dydt * dydt;
    if (isfinite(d2) && d2 > 1.0e-6f) {
        float tan_h = atan2f(dydt, dxdt);
        float err = clampf_cuda(wrap_pi(tan_h - interp_h), -RIGHT_TURN_FILLET_MAX_HEADING_ERR, RIGHT_TURN_FILLET_MAX_HEADING_ERR);
        oh = wrap_pi(interp_h + RIGHT_TURN_FILLET_TANGENT_BLEND * err);
    } else {
        oh = interp_h;
    }
    return true;
}
AVABM_DINLINE bool connector_right_turn_lane_following_xy_heading_ecs(int from_lane, int to_lane, float u, const RoadNetwork road,
        float& ox, float& oy, float& oh) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;
    if (turn_code_from_lanes_ecs(from_lane, to_lane, road) != TURN_RIGHT) return false;
    float h0 = lane_heading(from_lane, road);
    float h3 = lane_heading(to_lane, road);
    float delta = wrap_pi(h3 - h0);
    if (delta >= -0.10f || fabsf(delta) > 2.45f) return false;
    float t0x = cosf(h0);
    float t0y = sinf(h0);
    float t3x = cosf(h3);
    float t3y = sinf(h3);
    float corner_x, corner_y, from_param, to_param;
    if (!right_turn_corner_line_params_ecs(from_lane, to_lane, road, corner_x, corner_y, from_param, to_param)) {
        return false;
    }
    float lane_w = fminf(lane_width_estimate_ecs(from_lane, road), lane_width_estimate_ecs(to_lane, road));
    lane_w = clampf_cuda(lane_w, 2.7f, 4.4f);
    float entry_backoff = connector_entry_backoff_ecs(from_lane, to_lane, road);
    float exit_handoff = connector_local_handoff_estimate_ecs(from_lane, to_lane, road);
    float end_x = road.lane_end_x[from_lane];
    float end_y = road.lane_end_y[from_lane];
    float start_x = road.lane_start_x[to_lane];
    float start_y = road.lane_start_y[to_lane];
    float p0x = end_x - t0x * entry_backoff;
    float p0y = end_y - t0y * entry_backoff;
    float p3x = start_x + t3x * exit_handoff;
    float p3y = start_y + t3y * exit_handoff;
    float r_start = entry_backoff + from_param;
    float r_exit = exit_handoff - to_param;
    float radius = fminf(r_start, r_exit);
    radius = clampf_cuda(radius, lane_w * 1.10f, lane_w * 2.75f);
    float ax = corner_x - t0x * radius;
    float ay = corner_y - t0y * radius;
    float bx = corner_x + t3x * radius;
    float by = corner_y + t3y * radius;
    float lead_len = fmaxf(0.0f, sqrtf((ax - p0x) * (ax - p0x) + (ay - p0y) * (ay - p0y)));
    float exit_len = fmaxf(0.0f, sqrtf((p3x - bx) * (p3x - bx) + (p3y - by) * (p3y - by)));
    float handle = clampf_cuda(radius * 0.62f, lane_w * 0.65f, lane_w * 2.10f);
    float c1x = ax + t0x * handle;
    float c1y = ay + t0y * handle;
    float c2x = bx - t3x * handle;
    float c2y = by - t3y * handle;
    float curve_len = 0.0f;
    float last_x = ax;
    float last_y = ay;
    for (int k = 1; k <= 5; ++k) {
        float q = (float)k / 5.0f;
        float w = 1.0f - q;
        float qx = w*w*w*ax + 3.0f*w*w*q*c1x + 3.0f*w*q*q*c2x + q*q*q*bx;
        float qy = w*w*w*ay + 3.0f*w*w*q*c1y + 3.0f*w*q*q*c2y + q*q*q*by;
        float dx = qx - last_x;
        float dy = qy - last_y;
        curve_len += sqrtf(dx*dx + dy*dy);
        last_x = qx;
        last_y = qy;
    }
    curve_len = fmaxf(curve_len, lane_w * 1.20f);
    float total_len = fmaxf(lead_len + curve_len + exit_len, lane_w * 2.0f);
    float s = clampf_cuda(u, 0.0f, 1.0f) * total_len;
    if (lead_len > 0.05f && s < lead_len) {
        float q = s / lead_len;
        ox = p0x + (ax - p0x) * q;
        oy = p0y + (ay - p0y) * q;
        oh = h0;
        return true;
    }
    if (s > lead_len + curve_len && exit_len > 0.05f) {
        float q = clampf_cuda((s - lead_len - curve_len) / exit_len, 0.0f, 1.0f);
        ox = bx + (p3x - bx) * q;
        oy = by + (p3y - by) * q;
        oh = h3;
        return true;
    }
    float q = clampf_cuda((s - lead_len) / curve_len, 0.0f, 1.0f);
    float w = 1.0f - q;
    ox = w*w*w*ax + 3.0f*w*w*q*c1x + 3.0f*w*q*q*c2x + q*q*q*bx;
    oy = w*w*w*ay + 3.0f*w*w*q*c1y + 3.0f*w*q*q*c2y + q*q*q*by;
    float dxdt = 3.0f*w*w*(c1x - ax) + 6.0f*w*q*(c2x - c1x) + 3.0f*q*q*(bx - c2x);
    float dydt = 3.0f*w*w*(c1y - ay) + 6.0f*w*q*(c2y - c1y) + 3.0f*q*q*(by - c2y);
    float tan_h = atan2f(dydt, dxdt);
    float interp = wrap_pi(h0 + delta * smoothstep01(q));
    float err = clampf_cuda(wrap_pi(tan_h - interp), -0.42f, 0.42f);
    oh = wrap_pi(interp + 0.90f * err);
    return isfinite(ox) && isfinite(oy) && isfinite(oh);
}
AVABM_DINLINE bool connector_surface_arc_xy_heading_ecs(int from_lane, int to_lane, float u, const RoadNetwork road, float& ox,
        float& oy, float& oh) {
    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);
    float cx, cy, radius, a0, sweep, arc_len;
    if (!connector_surface_arc_params_raw_ecs(p0x, p0y, h0, p3x, p3y, h3, cx, cy, radius, a0, sweep, arc_len)) {
        return false;
    }
    if (fabsf(sweep) > TURN_ARC_STRICT_MAX_SWEEP_RAD) return false;
    u = clampf_cuda(u, 0.0f, 1.0f);
    float theta = a0 + sweep * u;
    ox = cx + cosf(theta) * radius;
    oy = cy + sinf(theta) * radius;
    float tangent = theta + (sweep >= 0.0f ? 1.57079632679f : -1.57079632679f);
    float interp = wrap_pi(h0 + wrap_pi(h3 - h0) * smoothstep01(u));
    float err = clampf_cuda(wrap_pi(tangent - interp), -0.65f, 0.65f);
    oh = wrap_pi(interp + 0.82f * err);
    return true;
}
AVABM_DINLINE void connector_surface_path_xy_heading_ecs(int from_lane, int to_lane, float u, const RoadNetwork road, float& ox,
        float& oy, float& oh) {
    if (from_lane < 0 || from_lane >= road.num_lanes || to_lane < 0 || to_lane >= road.num_lanes) {
        ox = 0.0f;
        oy = 0.0f;
        oh = 0.0f;
        return;
    }
    if (connector_right_turn_lane_following_xy_heading_ecs(from_lane, to_lane, u, road, ox, oy, oh)) {
        return;
    }
    if (connector_surface_arc_xy_heading_ecs(from_lane, to_lane, u, road, ox, oy, oh)) {
        return;
    }
    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);
    u = clampf_cuda(u, 0.0f, 1.0f);
    float dx = p3x - p0x;
    float dy = p3y - p0y;
    float chord = sqrtf(dx * dx + dy * dy);
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);
    float handle = fminf(fmaxf(chord * 0.36f, SURFACE_TURN_HANDLE_MIN), SURFACE_TURN_HANDLE_MAX);
    if (handoff_s > 0.0f) {
        handle = fminf(handle, fmaxf(SURFACE_TURN_HANDLE_MIN, handoff_s * 0.42f));
    }
    if (abs_delta < 0.25f) {
        handle = fminf(handle, fmaxf(SURFACE_TURN_HANDLE_MIN, chord * 0.28f));
    }
    float t0x = cosf(h0);
    float t0y = sinf(h0);
    float t3x = cosf(h3);
    float t3y = sinf(h3);
    float p1x = p0x + t0x * handle;
    float p1y = p0y + t0y * handle;
    float p2x = p3x - t3x * handle;
    float p2y = p3y - t3y * handle;
    if (chord > 0.25f) {
        float ux = dx / chord;
        float uy = dy / chord;
        float p1proj = (p1x - p0x) * ux + (p1y - p0y) * uy;
        float p2proj = (p2x - p0x) * ux + (p2y - p0y) * uy;
        float lo = -0.10f * chord;
        float hi = 1.10f * chord;
        if (p1proj < lo || p1proj > hi) {
            float cl = clampf_cuda(p1proj, lo, hi);
            p1x += (cl - p1proj) * ux;
            p1y += (cl - p1proj) * uy;
        }
        if (p2proj < lo || p2proj > hi) {
            float cl = clampf_cuda(p2proj, lo, hi);
            p2x += (cl - p2proj) * ux;
            p2y += (cl - p2proj) * uy;
        }
    }
    float w = 1.0f - u;
    float uu = u * u;
    float ww = w * w;
    ox = ww * w * p0x + 3.0f * ww * u * p1x + 3.0f * w * uu * p2x + uu * u * p3x;
    oy = ww * w * p0y + 3.0f * ww * u * p1y + 3.0f * w * uu * p2y + uu * u * p3y;
    float dxdt = 3.0f * ww * (p1x - p0x) + 6.0f * w * u * (p2x - p1x) + 3.0f * uu * (p3x - p2x);
    float dydt = 3.0f * ww * (p1y - p0y) + 6.0f * w * u * (p2y - p1y) + 3.0f * uu * (p3y - p2y);
    float interp_h = wrap_pi(h0 + delta * smoothstep01(u));
    float d2 = dxdt * dxdt + dydt * dydt;
    if (isfinite(d2) && d2 > 1.0e-6f) {
        float tan_h = atan2f(dydt, dxdt);
        float err = wrap_pi(tan_h - interp_h);
        if (fabsf(err) > 0.85f) {
            err = clampf_cuda(err, -0.85f, 0.85f);
        }
        oh = wrap_pi(interp_h + SURFACE_TURN_TANGENT_BLEND * err);
    } else {
        oh = interp_h;
    }
}
AVABM_DINLINE bool connector_swept_paths_overlap_ecs(int a_from, int a_to, int b_from, int b_to, const RoadNetwork road) {
    if (a_to < 0 || a_to >= road.num_lanes || b_to < 0 || b_to >= road.num_lanes) {
        return false;
    }
    if (a_to == b_to) return true;
    const float threshold = SURFACE_TURN_CONFLICT_RADIUS;
    const float threshold2 = threshold * threshold;
    for (int ia = 0; ia < SURFACE_TURN_SAMPLE_COUNT; ++ia) {
        float ua = ((float)ia + 0.5f) / (float)SURFACE_TURN_SAMPLE_COUNT;
        float ax, ay, ah;
        connector_surface_path_xy_heading_ecs(a_from, a_to, ua, road, ax, ay, ah);
        for (int ib = 0; ib < SURFACE_TURN_SAMPLE_COUNT; ++ib) {
            float ub = ((float)ib + 0.5f) / (float)SURFACE_TURN_SAMPLE_COUNT;
            float bx, by, bh;
            connector_surface_path_xy_heading_ecs(b_from, b_to, ub, road, bx, by, bh);
            float dx = ax - bx;
            float dy = ay - by;
            if (dx * dx + dy * dy <= threshold2) return true;
        }
    }
    return false;
}
AVABM_DINLINE bool intersection_conflict_relevant_lanes_ecs(int self_lane, int self_next_lane, int other_lane, int other_next_lane,
        bool other_in_connector, const RoadNetwork road) {
    if (self_lane < 0 || self_lane >= road.num_lanes || other_lane < 0 || other_lane >= road.num_lanes) {
        return false;
    }
    if (same_approach_same_direction_lanes_ecs(self_lane, other_lane, road)) {
        return false;
    }
    float dot = lane_heading_dot_ecs(self_lane, other_lane, road);
    int self_turn = turn_code_from_lanes_ecs(self_lane, self_next_lane, road);
    int other_turn = turn_code_from_lanes_ecs(other_lane, other_next_lane, road);
    bool self_left = self_turn == TURN_LEFT;
    bool other_left = other_turn == TURN_LEFT;
    bool swept_overlap = connector_swept_paths_overlap_ecs(self_lane, self_next_lane, other_lane, other_next_lane, road);
    if (!other_in_connector && !swept_overlap) {
        return false;
    }
    if (dot > DIRECTIONAL_SAME_APPROACH_DOT) return false;
    if (dot < DIRECTIONAL_ONCOMING_DOT) {
        if (other_in_connector) return true;
        return self_left || other_left;
    }
    if (fabsf(dot) <= DIRECTIONAL_SIDE_DOT_ABS) {
        if (!other_in_connector && self_turn == TURN_RIGHT && other_turn == TURN_RIGHT) return false;
        return true;
    }
    if (other_in_connector) return true;
    if (self_left || other_left) return true;
    if (self_next_lane >= 0 && self_next_lane == other_next_lane) return true;
    return false;
}
AVABM_DINLINE bool intersection_conflict_relevant_vehicles_ecs(int self, int self_lane, int self_next_lane, int other,
        int other_lane, int other_next_lane, bool other_in_connector, ECSArrays ecs, const RoadNetwork road) {
    if (self_lane < 0 || self_lane >= road.num_lanes || other_lane < 0 || other_lane >= road.num_lanes) return false;
    if (same_approach_same_direction_lanes_ecs(self_lane, other_lane, road)) return false;
    int self_turn = intended_turn_with_indicator_ecs(self, self_lane, self_next_lane, ecs, road);
    int other_turn = intended_turn_with_indicator_ecs(other, other_lane, other_next_lane, ecs, road);
    bool swept_overlap = connector_swept_paths_overlap_ecs(self_lane, self_next_lane, other_lane, other_next_lane, road);
    if (!other_in_connector && !swept_overlap) {
        return false;
    }
    float dot = lane_heading_dot_ecs(self_lane, other_lane, road);
    if (dot > DIRECTIONAL_SAME_APPROACH_DOT) return false;
    if (dot < DIRECTIONAL_ONCOMING_DOT) {
        if (other_in_connector) return true;
        return self_turn == TURN_LEFT || other_turn == TURN_LEFT;
    }
    if (fabsf(dot) <= DIRECTIONAL_SIDE_DOT_ABS) {
        if (!other_in_connector && self_turn == TURN_RIGHT && other_turn == TURN_RIGHT) return false;
        return true;
    }
    if (other_in_connector) return true;
    if (self_turn == TURN_LEFT || other_turn == TURN_LEFT) return true;
    if (self_next_lane >= 0 && self_next_lane == other_next_lane) return true;
    return false;
}
AVABM_DINLINE float directional_attention_range_ecs(int self_lane, int other_lane, int dtype, const RoadNetwork road) {
    bool human = dtype == HUMAN;
    float dot = lane_heading_dot_ecs(self_lane, other_lane, road);
    if (dot < DIRECTIONAL_ONCOMING_DOT) {
        return human ? DIRECTIONAL_ONCOMING_RANGE_HUMAN : DIRECTIONAL_ONCOMING_RANGE_AV;
    }
    if (fabsf(dot) <= DIRECTIONAL_SIDE_DOT_ABS) {
        return human ? DIRECTIONAL_SIDE_RANGE_HUMAN : DIRECTIONAL_SIDE_RANGE_AV;
    }
    return human ? INTERACTION_RANGE_HUMAN : INTERACTION_RANGE_AV;
}
AVABM_DINLINE bool directional_vehicle_conflict_relevant_ecs(int self, int self_lane, int self_next_lane, int other, ECSArrays ecs,
        const RoadNetwork road) {
    if (other == self || ecs.alive[other] != ENTITY_ALIVE) return false;
    if (self_lane < 0 || self_lane >= road.num_lanes) return false;
    bool other_in_connector = ecs.vehicle_state[other] == VEH_IN_CONNECTOR;
    int other_lane = other_in_connector ? ecs.connector_from_lane[other] : ecs.lane_id[other];
    int other_next_lane = other_in_connector ? ecs.connector_to_lane[other] : route_next_lane_for_vehicle_ecs(other, ecs, road);
    if (other_lane < 0 || other_lane >= road.num_lanes) return false;
    int self_node = road.lane_end_node[self_lane];
    int other_node = road.lane_end_node[other_lane];
    if (self_node != other_node) {
        return false;
    }
    return intersection_conflict_relevant_vehicles_ecs(self, self_lane, self_next_lane, other, other_lane, other_next_lane,
            other_in_connector, ecs, road);
}
AVABM_DINLINE float turn_speed_cap(float angle_deg, int dtype) {
    bool human = dtype == HUMAN;
    float straight = human ? TURN_SPEED_STRAIGHT_HUMAN : TURN_SPEED_STRAIGHT_AV;
    float hard = human ? TURN_SPEED_HARD_HUMAN : TURN_SPEED_HARD_AV;
    float uturn = human ? TURN_SPEED_UTURN_HUMAN : TURN_SPEED_UTURN_AV;
    if (angle_deg < 8.0f) return straight;
    if (angle_deg > 115.0f) {
        float t = clampf_cuda((angle_deg - 115.0f) / 65.0f, 0.0f, 1.0f);
        return hard + (uturn - hard) * smoothstep01(t);
    }
    float t = clampf_cuda((angle_deg - 8.0f) / 107.0f, 0.0f, 1.0f);
    return straight + (hard - straight) * smoothstep01(t);
}
AVABM_DINLINE float connector_length_between_lanes(int from_lane, int to_lane, const RoadNetwork road) {
    if (from_lane < 0 || to_lane < 0) return CONNECTOR_DEFAULT_LEN;
    if (from_lane >= road.num_lanes || to_lane >= road.num_lanes) return CONNECTOR_DEFAULT_LEN;
    const int samples = 8;
    float px, py, ph;
    connector_surface_path_xy_heading_ecs(from_lane, to_lane, 0.0f, road, px, py, ph);
    float total = 0.0f;
    float last_x = px;
    float last_y = py;
    for (int k = 1; k <= samples; ++k) {
        float u = (float)k / (float)samples;
        float x, y, h;
        connector_surface_path_xy_heading_ecs(from_lane, to_lane, u, road, x, y, h);
        float dx = x - last_x;
        float dy = y - last_y;
        total += sqrtf(dx * dx + dy * dy);
        last_x = x;
        last_y = y;
    }
    if (!isfinite(total) || total < 0.25f) total = CONNECTOR_DEFAULT_LEN;
    float angle = turn_angle_deg(from_lane, to_lane, road);
    float min_len = angle < 8.0f ? CONNECTOR_MIN_LEN * 0.55f : CONNECTOR_MIN_LEN;
    return clampf_cuda(total, min_len, CONNECTOR_MAX_LEN);
}
AVABM_DINLINE float connector_geometry_chord(int from_lane, int to_lane, const RoadNetwork road) {
    if (from_lane < 0 || to_lane < 0) return 1.0e9f;
    float dx = road.lane_start_x[to_lane] - road.lane_end_x[from_lane];
    float dy = road.lane_start_y[to_lane] - road.lane_end_y[from_lane];
    float chord = sqrtf(dx * dx + dy * dy);
    if (!isfinite(chord)) return 1.0e9f;
    return chord;
}
AVABM_DINLINE bool connector_uses_handoff(int from_lane, int to_lane, const RoadNetwork road) {
    return connector_geometry_chord(from_lane, to_lane, road) < CONNECTOR_SAME_NODE_EPS;
}
AVABM_DINLINE float connector_exit_handoff_s(int from_lane, int to_lane, const RoadNetwork road) {
    return connector_local_handoff_estimate_ecs(from_lane, to_lane, road);
}
AVABM_DINLINE float connector_route_distance_to_next_lane_s(int from_lane, int to_lane, float remain_from_lane, float next_lane_s,
        const RoadNetwork road) {
    float clen = connector_length_between_lanes(from_lane, to_lane, road);
    float entry_backoff = connector_entry_backoff_ecs(from_lane, to_lane, road);
    float handoff = connector_exit_handoff_s(from_lane, to_lane, road);
    float after_handoff = fmaxf(0.0f, next_lane_s - handoff);
    return fmaxf(0.0f, remain_from_lane - entry_backoff) + clen + after_handoff;
}
AVABM_DINLINE void connector_xy_heading_from_s(int from_lane, int to_lane, float conn_s, float conn_len, const RoadNetwork road,
        float& ox, float& oy, float& oh) {
    conn_len = fmaxf(conn_len, CONNECTOR_MIN_LEN);
    float u = clampf_cuda(conn_s / conn_len, 0.0f, 1.0f);
    connector_surface_path_xy_heading_ecs(from_lane, to_lane, u, road, ox, oy, oh);
}
AVABM_DINLINE bool in_phase(float p, float a, float b) {
    if (a <= b) return p >= a && p < b;
    return p >= a || p < b;
}
AVABM_DINLINE int signal_state(float t, float cycle, float green_start, float green_end, float yellow_start, float yellow_end) {
    if (cycle <= 1.0f) return LIGHT_GREEN;
    float p = fmodf(t, cycle);
    if (p < 0.0f) p += cycle;
    if (in_phase(p, green_start, green_end)) return LIGHT_GREEN;
    if (in_phase(p, yellow_start, yellow_end)) return LIGHT_YELLOW;
    return LIGHT_RED;
}
AVABM_DINLINE bool signal_turn_match(int signal_turn, int turn) {
    if (signal_turn == TURN_ANY) return true;
    if (turn == TURN_LEFT) return signal_turn == TURN_LEFT;
    if (turn == TURN_RIGHT) return signal_turn == TURN_RIGHT || signal_turn == TURN_STRAIGHT;
    return signal_turn == TURN_STRAIGHT;
}
AVABM_DINLINE int get_signal_for_lane_turn(int lane, int turn, float current_time, const RoadNetwork road, const Signals signals) {
    int node = road.lane_end_node[lane];
    int found = LIGHT_GREEN;
    for (int k = 0; k < signals.num_signals; ++k) {
        if (signals.signal_node[k] != node) continue;
        if (!signal_turn_match(signals.signal_turn[k], turn)) continue;
        int st = signal_state(current_time, signals.signal_cycle[k], signals.signal_green_start[k], signals.signal_green_end[k],
                signals.signal_yellow_start[k], signals.signal_yellow_end[k]);
        if (st == LIGHT_RED) return LIGHT_RED;
        if (st == LIGHT_YELLOW) found = LIGHT_YELLOW;
    }
    return found;
}
AVABM_DINLINE bool node_has_signal_ecs(int node, const Signals signals) {
    if (node < 0) return false;
    for (int k = 0; k < signals.num_signals; ++k) {
        if (signals.signal_node[k] == node) return true;
    }
    return false;
}
AVABM_DINLINE float desired_speed_ecs(int id, int lane, const ECSArrays ecs, const RoadNetwork road) {
    float limit = road.lane_speed_limit[lane];
    if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    int dtype = ecs.driver_type[id];
    float factor = ecs.desired_speed_factor[id];
    float aggr = ecs.aggressiveness[id];
    if (!isfinite(factor) || factor <= 0.1f) {
        factor = dtype == AV ? 0.97f : 0.88f;
    }
    if (!isfinite(aggr)) {
        aggr = dtype == AV ? 0.55f : 0.50f;
    }
    float min_cruise = avabm_min_cruise_speed_mps_ecs();
    float effective_limit = fmaxf(limit, min_cruise);
    if (dtype == AV) {
        return clampf_cuda(effective_limit * factor, fmaxf(3.0f, min_cruise), 36.0f);
    }
    float human_factor = factor + 0.10f * (aggr - 0.5f);
    return clampf_cuda(effective_limit * human_factor, fmaxf(3.0f, min_cruise), 38.0f);
}
AVABM_DINLINE float estimate_follow_accel_ecs(float v, float desired_v, float front_gap, float front_v, int dtype, float min_gap_i,
        float reaction_i, float comfort_decel_i, float aggressiveness_i, float risk_i) {
    bool human = dtype == HUMAN;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    if (!isfinite(aggressiveness_i)) aggressiveness_i = human ? 0.5f : 0.55f;
    if (!isfinite(risk_i)) risk_i = human ? 0.5f : 0.35f;
    if (!isfinite(comfort_decel_i) || comfort_decel_i <= 0.1f) {
        comfort_decel_i = human ? 2.8f : 3.5f;
    }
    max_accel *= 0.75f + 0.65f * aggressiveness_i;
    float T = fmaxf(reaction_i, 0.2f);
    T *= human ? 1.0f : 0.85f;
    T *= 1.20f - 0.35f * risk_i;
    float s0 = fmaxf(min_gap_i, 1.0f) * (1.15f - 0.45f * risk_i);
    float b = clampf_cuda(comfort_decel_i, 1.0f, max_decel);
    float dv = v - front_v;
    float sqrt_ab = sqrtf(fmaxf(max_accel * b, 0.1f));
    float s_star = s0 + fmaxf(0.0f, v * T + (v * dv) / fmaxf(2.0f * sqrt_ab, 0.1f));
    float free_term = avabm_fourth_power_ecs(v / fmaxf(desired_v, 0.1f));
    float interact = 0.0f;
    if (front_gap < 1.0e8f) {
        interact = avabm_second_power_ecs(s_star / fmaxf(front_gap, 0.5f));
    }
    float a = max_accel * (1.0f - free_term - interact);
    return clampf_cuda(a, -EMERGENCY_DECEL, max_accel);
}
AVABM_DINLINE float relative_closing_ttc_accel_limit_ecs(int self, int other, ECSArrays ecs, float horizon, float* metrics) {
    float rx = ecs.x[other] - ecs.x[self];
    float ry = ecs.y[other] - ecs.y[self];
    float dist = sqrtf(fmaxf(rx * rx + ry * ry, 0.001f));
    float vix = cosf(ecs.heading[self]) * ecs.speed[self];
    float viy = sinf(ecs.heading[self]) * ecs.speed[self];
    float vjx = cosf(ecs.heading[other]) * ecs.speed[other];
    float vjy = sinf(ecs.heading[other]) * ecs.speed[other];
    float rvx = vjx - vix;
    float rvy = vjy - viy;
    float closing = -((rx * rvx + ry * rvy) / fmaxf(dist, 0.1f));
    float combined = 0.5f * ecs.length[self] + 0.5f * ecs.length[other] + MIN_BUMPER_GAP;
    float gap = dist - combined;
    if (gap <= 0.0f) return -EMERGENCY_DECEL;
    float vv = rvx * rvx + rvy * rvy;
    float t_near = 1.0e9f;
    if (vv > 0.01f) {
        t_near = clampf_cuda(-((rx * rvx + ry * rvy) / vv), 0.0f, horizon);
    }
    float near_x = rx + rvx * t_near;
    float near_y = ry + rvy * t_near;
    float near_sep = sqrtf(fmaxf(near_x * near_x + near_y * near_y, 0.001f));
    bool projected_overlap = near_sep < combined + 1.75f;
    bool closing_overlap = false;
    float ttc = 1.0e9f;
    if (closing > 0.05f) {
        ttc = gap / closing;
        closing_overlap = ttc < INTERACTION_TTC_SOFT;
    }
    if (!projected_overlap && !closing_overlap) return 1000.0f;
    bool human = ecs.driver_type[self] == HUMAN;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    float risk = clampf_cuda(ecs.risk_tolerance[self], 0.0f, 1.0f);
    float severity = 0.0f;
    if (closing_overlap) {
        severity = fmaxf(severity, (INTERACTION_TTC_SOFT - ttc) / INTERACTION_TTC_SOFT);
    }
    if (projected_overlap) {
        severity = fmaxf(severity, (combined + 1.75f - near_sep) / fmaxf(combined + 1.75f, 0.1f));
    }
    severity = clampf_cuda(severity * (1.20f - 0.45f * risk), 0.0f, 1.0f);
    float limit = -max_decel * (0.25f + 0.95f * severity);
    if (ttc < INTERACTION_TTC_HARD || gap < MIN_BUMPER_GAP) {
        limit = fminf(limit, -EMERGENCY_DECEL);
    }
    if (metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_INTERACTION_BRAKE, 1.0f);
    }
    return clampf_cuda(limit, -EMERGENCY_DECEL, 0.0f);
}
AVABM_DINLINE float interaction_accel_limit_ecs(int self, int lane, int next_lane, ECSArrays ecs, RoadNetwork road,
        SpatialGrid grid, int max_entities, float* metrics) {
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    bool human = ecs.driver_type[self] == HUMAN;
    float max_range = human ? DIRECTIONAL_SIDE_RANGE_HUMAN : DIRECTIONAL_SIDE_RANGE_AV;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(max_range / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float best_limit = 1000.0f;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_connector = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_in_connector ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    if (other_lane >= 0 && other_lane < road.num_lanes &&
                            directional_vehicle_conflict_relevant_ecs(self, lane, next_lane, j, ecs, road)) {
                        float rx = ecs.x[j] - ecs.x[self];
                        float ry = ecs.y[j] - ecs.y[self];
                        float dist = sqrtf(fmaxf(rx * rx + ry * ry, 0.001f));
                        float attention_range = directional_attention_range_ecs(lane, other_lane, ecs.driver_type[self], road);
                        if (other_in_connector) {
                            attention_range = fmaxf(attention_range, 26.0f);
                        }
                        if (dist <= attention_range) {
                            float limit = relative_closing_ttc_accel_limit_ecs(self, j, ecs, 2.6f, metrics);
                            best_limit = fminf(best_limit, limit);
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return best_limit;
}
AVABM_DINLINE bool local_same_path_following_ecs(int a, int b, ECSArrays ecs) {
    if (ecs.vehicle_state[a] == VEH_ON_LANE && ecs.vehicle_state[b] == VEH_ON_LANE) {
        if (ecs.lane_id[a] == ecs.lane_id[b]) return true;
        if (ecs.lane_change_active[a] != 0 && (ecs.lane_change_from_lane[a] == ecs.lane_id[b] ||
                ecs.lane_change_to_lane[a] == ecs.lane_id[b])) return true;
        if (ecs.lane_change_active[b] != 0 && (ecs.lane_change_from_lane[b] == ecs.lane_id[a] ||
                ecs.lane_change_to_lane[b] == ecs.lane_id[a])) return true;
    }
    if (ecs.vehicle_state[a] == VEH_IN_CONNECTOR && ecs.vehicle_state[b] == VEH_IN_CONNECTOR) {
        return ecs.connector_from_lane[a] == ecs.connector_from_lane[b] && ecs.connector_to_lane[a] == ecs.connector_to_lane[b];
    }
    return false;
}
AVABM_DINLINE bool local_front_clear_for_id_ecs(int id, PerceptionSoA perception, ECSArrays ecs) {
    if (id < 0) return true;
    float fg = perception.front_gap != nullptr ? perception.front_gap[id] : 1.0e9f;
    if (!isfinite(fg)) fg = 0.0f;
    float needed = fmaxf(FRONT_CLEAR_PRIORITY_MIN_GAP, fmaxf(ecs.length[id], 4.0f) + MIN_BUMPER_GAP + fmaxf(0.0f,
            ecs.speed[id]) * FRONT_CLEAR_PRIORITY_TIME);
    return fg > needed;
}
AVABM_DINLINE int local_avoidance_priority_key_ecs(int id, PerceptionSoA perception, ECSArrays ecs, RoadNetwork road) {
    int key = id & 1023;
    bool front_clear = local_front_clear_for_id_ecs(id, perception, ecs);
    if (front_clear) key -= LOCAL_AVOID_FRONT_CLEAR_BONUS;
    else key += LOCAL_AVOID_FRONT_CLEAR_BONUS;
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        key -= LOCAL_AVOID_CONNECTOR_BONUS;
    } else {
        int ln = ecs.lane_id[id];
        int nx = route_next_lane_for_vehicle_ecs(id, ecs, road);
        if (valid_lane_ecs(ln, road) && valid_lane_ecs(nx, road)) {
            float dist = fmaxf(0.0f, road.lane_length[ln] - ecs.s[id]);
            if (inside_intersection_box_ecs(dist, ln, nx, road)) key -= LOCAL_AVOID_INSIDE_BOX_BONUS;
        }
    }
    float wait = clampf_cuda(ecs.connector_length[id], 0.0f, 60.0f);
    key -= clampi_cuda((int)floorf(wait * 12.0f), 0, 180);
    if (ecs.driver_type[id] == HUMAN) {
        float behavior = clampf_cuda(0.55f * ecs.aggressiveness[id] + 0.35f * ecs.risk_tolerance[id] +
                0.10f * (1.0f - ecs.politeness[id]), 0.0f, 1.0f);
        key -= (int)(behavior * 18.0f);
    }
    return key;
}
AVABM_DINLINE float local_obstacle_avoidance_accel_limit_ecs(int self, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        PerceptionSoA perception, int max_entities, float dt, float* metrics) {
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(LOCAL_AVOID_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float sx = ecs.x[self];
    float sy = ecs.y[self];
    float svx = cosf(ecs.heading[self]) * fmaxf(0.0f, ecs.speed[self]);
    float svy = sinf(ecs.heading[self]) * fmaxf(0.0f, ecs.speed[self]);
    float best_limit = 1000.0f;
    int self_key = local_avoidance_priority_key_ecs(self, perception, ecs, road);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    if (!local_same_path_following_ecs(self, j, ecs)) {
                        float rx = ecs.x[j] - sx;
                        float ry = ecs.y[j] - sy;
                        float ovx = cosf(ecs.heading[j]) * fmaxf(0.0f, ecs.speed[j]);
                        float ovy = sinf(ecs.heading[j]) * fmaxf(0.0f, ecs.speed[j]);
                        float rvx = ovx - svx;
                        float rvy = ovy - svy;
                        float vv = rvx * rvx + rvy * rvy;
                        float t = 0.0f;
                        if (vv > 0.01f) {
                            t = clampf_cuda(-((rx * rvx + ry * rvy) / vv), 0.0f, LOCAL_AVOID_HORIZON);
                        }
                        float nx = rx + rvx * t;
                        float ny = ry + rvy * t;
                        float sep2 = nx * nx + ny * ny;
                        float dist_now = sqrtf(fmaxf(rx * rx + ry * ry, 0.001f));
                        float combined = 0.36f * (fmaxf(ecs.length[self], 3.0f) + fmaxf(ecs.length[j],
                                3.0f)) + 0.50f * fmaxf(fmaxf(ecs.width[self], 1.4f), fmaxf(ecs.width[j],
                                1.4f)) + LOCAL_AVOID_COLLISION_MARGIN;
                        bool projected_conflict = sep2 < combined * combined;
                        bool close_now = dist_now < combined + 0.75f;
                        float closing = 0.0f;
                        if (dist_now > 0.1f) closing = -((rx * (rvx) + ry * (rvy)) / dist_now);
                        if (projected_conflict && (closing > 0.05f || close_now)) {
                            int other_key = local_avoidance_priority_key_ecs(j, perception, ecs, road);
                            bool self_yields = self_key > other_key || (self_key == other_key && self > j);
                            if (self_yields) {
                                float stop_dist = fmaxf(dist_now - combined - LOCAL_AVOID_STOP_BUFFER, 0.55f);
                                float req = -(ecs.speed[self] * ecs.speed[self]) / fmaxf(2.0f * stop_dist, 0.5f);
                                req = clampf_cuda(req, -EMERGENCY_DECEL, -0.05f);
                                best_limit = fminf(best_limit, req);
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    if (best_limit < 999.0f && metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_ANTI_COLLISION_BRAKE, 1.0f);
        AVABM_METRIC_ADD(metrics, METRIC_INTERACTION_BRAKE, 1.0f);
    }
    return best_limit;
}
AVABM_DINLINE float intersection_conflict_accel_limit_ecs(int self, int lane, int next_lane, float dist_to_end, ECSArrays ecs,
        RoadNetwork road, SpatialGrid grid, int max_entities, float current_time, float* metrics) {
    if (next_lane < 0 || next_lane >= road.num_lanes) return 1000.0f;
    if (dist_to_end > INTERSECTION_APPROACH_RANGE) return 1000.0f;
    int node = road.lane_end_node[lane];
    if (node < 0) return 1000.0f;
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    float self_arrival = current_time + dist_to_end / fmaxf(ecs.speed[self], 0.8f);
    bool human = ecs.driver_type[self] == HUMAN;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(DIRECTIONAL_SIDE_RANGE_HUMAN / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float best_limit = 1000.0f;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_intersection = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_from = other_in_intersection ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_in_intersection ? ecs.connector_to_lane[j] : route_next_lane_for_vehicle_ecs(j, ecs,
                            road);
                    if (other_from >= 0 && other_from < road.num_lanes) {
                        int other_node = road.lane_end_node[other_from];
                        if (other_node == node &&
                                intersection_conflict_relevant_vehicles_ecs(self, lane, next_lane, j, other_from, other_next,
                                other_in_intersection, ecs, road)) {
                            float other_dist = 0.0f;
                            float other_arrival = current_time;
                            if (other_in_intersection) {
                                other_arrival = current_time - 0.25f;
                            } else {
                                other_dist = fmaxf(0.0f, road.lane_length[other_from] - ecs.s[j]);
                                float attention_range = directional_attention_range_ecs(lane, other_from, ecs.driver_type[self],
                                        road);
                                attention_range = fminf(attention_range, INTERSECTION_APPROACH_RANGE);
                                if (other_dist <= attention_range) {
                                    other_arrival = current_time + other_dist / fmaxf(ecs.speed[j], 0.8f);
                                } else {
                                    other_arrival = 1.0e9f;
                                }
                            }
                            if (!other_in_intersection && ecs.speed[j] < DIRECTIONAL_OTHER_STOP_EPS &&
                                    other_dist > fmaxf(12.0f, dist_to_end + 4.0f)) {
                                other_arrival = 1.0e9f;
                            }
                            if (other_arrival < 1.0e8f) {
                                float dt_arrival = other_arrival - self_arrival;
                                bool other_priority = other_in_intersection || dt_arrival < -INTERSECTION_PRIORITY_EPS ||
                                        (fabsf(dt_arrival) < INTERSECTION_TIME_WINDOW && j < self);
                                if (other_priority) {
                                    float stop_dist = fmaxf(dist_to_end - INTERSECTION_STOP_BUFFER, 0.75f);
                                    float req = -(ecs.speed[self] * ecs.speed[self]) / fmaxf(2.0f * stop_dist, 0.5f);
                                    req = clampf_cuda(req, -EMERGENCY_DECEL, -0.03f);
                                    req = fminf(req, -0.35f * max_decel);
                                    best_limit = fminf(best_limit, req);
                                    if (metrics != nullptr) {
                                        AVABM_METRIC_ADD(metrics, METRIC_CONFLICT_YIELD, 1.0f);
                                        AVABM_METRIC_ADD(metrics, METRIC_COOP_YIELD, 1.0f);
                                    }
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return best_limit;
}
AVABM_DINLINE bool approach_from_self_right_ecs(int self_lane, int other_lane, const RoadNetwork road) {
    if (self_lane < 0 || other_lane < 0) return false;
    float sx, sy, ox, oy;
    lane_dir(self_lane, road, sx, sy);
    lane_dir(other_lane, road, ox, oy);
    float cross = sx * oy - sy * ox;
    return cross > UNSIGNAL_RIGHT_PRIORITY_CROSS;
}
AVABM_DINLINE int turn_priority_rank_unsignal_ecs(int turn) {
    if (turn == TURN_STRAIGHT) return 0;
    if (turn == TURN_RIGHT) return 1;
    if (turn == TURN_LEFT) return 2;
    return 1;
}
AVABM_DINLINE unsigned int deadlock_release_score_ecs(int id, int node, float current_time) {
    int slot = (int)floorf(current_time / fmaxf(DEADLOCK_RELEASE_PERIOD, 0.25f));
    unsigned int x = (unsigned int)id * 747796405u;
    x ^= (unsigned int)node * 2891336453u;
    x ^= (unsigned int)slot * 277803737u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return x;
}
AVABM_DINLINE bool unsignal_other_has_priority_ecs(int self, int other, int self_lane, int self_next_lane, int other_lane,
        int other_next_lane, bool other_in_connector, float self_arrival, float other_arrival, const RoadNetwork road) {
    if (other_in_connector) return true;
    int self_turn = turn_code_from_lanes_ecs(self_lane, self_next_lane, road);
    int other_turn = turn_code_from_lanes_ecs(other_lane, other_next_lane, road);
    bool other_from_right = approach_from_self_right_ecs(self_lane, other_lane, road);
    bool self_from_right = approach_from_self_right_ecs(other_lane, self_lane, road);
    float dt_arrival = other_arrival - self_arrival;
    float dot = lane_heading_dot_ecs(self_lane, other_lane, road);
    bool crossing_straight_pair = self_turn == TURN_STRAIGHT && other_turn == TURN_STRAIGHT &&
            fabsf(dot) <= DIRECTIONAL_SIDE_DOT_ABS;
    if (self_turn == TURN_STRAIGHT && other_turn != TURN_STRAIGHT) return false;
    if (other_turn == TURN_STRAIGHT && self_turn != TURN_STRAIGHT) {
        return dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW + 0.85f;
    }
    if (crossing_straight_pair) {
        if (other_from_right && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
        if (self_from_right && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;
    }
    if (other_from_right && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
    if (self_from_right && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;
    if (dot < DIRECTIONAL_ONCOMING_DOT) {
        int sr = turn_priority_rank_unsignal_ecs(self_turn);
        int orr = turn_priority_rank_unsignal_ecs(other_turn);
        if (sr > orr && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
        if (sr < orr && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;
    }
    if (dt_arrival < -UNSIGNAL_ARRIVAL_EPS) return true;
    if (dt_arrival > UNSIGNAL_ARRIVAL_EPS) return false;
    return other < self;
}
AVABM_DINLINE float unsignal_priority_accel_limit_ecs(int self, int lane, int next_lane, float dist_to_end, float front_gap,
        ECSArrays ecs, RoadNetwork road, SpatialGrid grid, int max_entities, float current_time, float dt, float* metrics,
        bool* out_blocked, bool* out_release, bool* out_conflict_seen) {
    if (out_blocked) *out_blocked = false;
    if (out_release) *out_release = false;
    if (out_conflict_seen) *out_conflict_seen = false;
    if (next_lane < 0 || next_lane >= road.num_lanes) return 1000.0f;
    if (dist_to_end > UNSIGNAL_PRIORITY_APPROACH_RANGE) {
        if (ecs.vehicle_state[self] == VEH_ON_LANE) ecs.connector_length[self] = 0.0f;
        return 1000.0f;
    }
    int node = road.lane_end_node[lane];
    if (node < 0) return 1000.0f;
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    float self_arrival = current_time + fmaxf(0.0f, dist_to_end - DEFAULT_STOP_OFFSET) / fmaxf(ecs.speed[self], 0.8f);
    bool human = ecs.driver_type[self] == HUMAN;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    float patience = human ? DEADLOCK_PATIENCE_HUMAN : DEADLOCK_PATIENCE_AV;
    if (indicator_active_ecs(self, ecs)) {
        patience *= DEADLOCK_INDICATOR_PATIENCE_SCALE;
    } else {
        patience *= DEADLOCK_ESCAPE_PATIENCE_SCALE;
    }
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(UNSIGNAL_PRIORITY_APPROACH_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    bool must_yield = false;
    bool moving_priority = false;
    bool connector_priority = false;
    int conflict_count = 0;
    unsigned int best_release_score = deadlock_release_score_ecs(self, node, current_time);
    int best_release_id = self;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_connector = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_in_connector ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_in_connector ? ecs.connector_to_lane[j] : route_next_lane_for_vehicle_ecs(j, ecs, road);
                    if (other_lane >= 0 && other_lane < road.num_lanes && road.lane_end_node[other_lane] == node &&
                            intersection_conflict_relevant_vehicles_ecs(self, lane, next_lane, j, other_lane, other_next,
                            other_in_connector, ecs, road)) {
                        float other_dist = 0.0f;
                        float other_arrival = current_time - 0.15f;
                        if (!other_in_connector) {
                            other_dist = fmaxf(0.0f, road.lane_length[other_lane] - ecs.s[j]);
                            float attention = directional_attention_range_ecs(lane, other_lane, ecs.driver_type[self], road);
                            attention = fmaxf(attention, UNSIGNAL_PRIORITY_NEAR_LINE_DIST);
                            attention = fminf(attention, UNSIGNAL_PRIORITY_APPROACH_RANGE);
                            if (other_dist > attention) {
                                j = grid.cell_next[j];
                                guard++;
                                continue;
                            }
                            other_arrival = current_time + fmaxf(0.0f, other_dist - DEFAULT_STOP_OFFSET) / fmaxf(ecs.speed[j],
                                    0.8f);
                            if (ecs.speed[j] < UNSIGNAL_STOPPED_EPS && other_dist > UNSIGNAL_STOPPED_FAR_IGNORE_DIST &&
                                    other_dist > dist_to_end + 5.0f) {
                                j = grid.cell_next[j];
                                guard++;
                                continue;
                            }
                        }
                        bool other_priority = unsignal_other_has_priority_ecs(self, j, lane, next_lane, other_lane, other_next,
                                other_in_connector, self_arrival, other_arrival, road);
                        if (other_priority) {
                            must_yield = true;
                            conflict_count++;
                            if (metrics != nullptr) {
                                AVABM_METRIC_ADD(metrics, METRIC_UNSIGNAL_CONFLICT, 1.0f);
                                if (indicator_active_ecs(j, ecs) || indicator_active_ecs(self, ecs)) {
                                    AVABM_METRIC_ADD(metrics, METRIC_INDICATOR_CONFLICT_YIELD, 1.0f);
                                }
                            }
                            if (other_in_connector) connector_priority = true;
                            if (!other_in_connector && ecs.speed[j] > 1.15f &&
                                    other_arrival <= self_arrival + 0.65f) moving_priority = true;
                        }
                        bool other_stopped_near = !other_in_connector && ecs.speed[j] < UNSIGNAL_STOPPED_EPS &&
                                other_dist <= UNSIGNAL_PRIORITY_NEAR_LINE_DIST + 5.0f && ecs.connector_length[j] > patience * 0.35f;
                        if (other_stopped_near || other_in_connector) {
                            unsigned int score = deadlock_release_score_ecs(j, node, current_time);
                            if (score < best_release_score || (score == best_release_score && j < best_release_id)) {
                                best_release_score = score;
                                best_release_id = j;
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    float wait_time = ecs.connector_length[self];
    if (!isfinite(wait_time) || wait_time < 0.0f || ecs.vehicle_state[self] != VEH_ON_LANE) wait_time = 0.0f;
    bool near_stop_line = dist_to_end <= UNSIGNAL_PRIORITY_NEAR_LINE_DIST + DEFAULT_STOP_OFFSET;
    if (must_yield && near_stop_line && ecs.speed[self] < 0.75f) {
        wait_time = fminf(wait_time + dt, 30.0f);
    } else if (!must_yield || !near_stop_line) {
        wait_time = fmaxf(0.0f, wait_time - 2.0f * dt);
    }
    ecs.connector_length[self] = wait_time;
    if (!must_yield) {
        if (wait_time <= 0.01f && metrics != nullptr) AVABM_METRIC_ADD(metrics, METRIC_UNSIGNAL_PRIORITY_GO, 1.0f);
        return 1000.0f;
    }
    if (out_blocked) *out_blocked = true;
    if (out_conflict_seen) *out_conflict_seen = conflict_count > 0;
    float clear_front = fmaxf(UNSIGNAL_RELEASE_FRONT_GAP, ecs.length[self] + MIN_BUMPER_GAP + 5.0f);
    bool front_clear = front_gap > clear_front;
    bool front_super_clear = front_gap > clear_front * FRONT_CLEAR_RELEASE_GAP_MULT;
    bool front_clear_assertive_release = wait_time >= patience * FRONT_CLEAR_ASSERTIVE_WAIT_SCALE && near_stop_line &&
            front_super_clear && !connector_priority && !moving_priority;
    bool front_empty_release = wait_time >= fmaxf(0.06f, patience * FRONT_EMPTY_RELEASE_WAIT_SCALE) && near_stop_line &&
            front_clear && !connector_priority && (best_release_id == self || front_super_clear);
    bool extended_wait_release = wait_time >= patience * 1.25f && near_stop_line && front_clear && !connector_priority &&
            (best_release_id == self || front_super_clear);
    bool deadlock_candidate = (wait_time >= patience && near_stop_line && front_clear && !moving_priority && !connector_priority &&
            (best_release_id == self || front_clear_assertive_release)) || extended_wait_release || front_empty_release ||
            front_clear_assertive_release;
    if (deadlock_candidate) {
        if (out_release) *out_release = true;
        ecs.connector_length[self] = fminf(wait_time, patience + 0.5f);
        if (metrics != nullptr) {
            AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_RELEASE, 1.0f);
            AVABM_METRIC_ADD(metrics, METRIC_UNSIGNAL_PRIORITY_GO, 1.0f);
            AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_ESCAPE_GO, 1.0f);
            if (front_empty_release) AVABM_METRIC_ADD(metrics, METRIC_FRONT_SPACE_RELEASE, 1.0f);
            if (indicator_active_ecs(self, ecs)) AVABM_METRIC_ADD(metrics, METRIC_INDICATOR_PRIORITY_GO, 1.0f);
        }
        return 1000.0f;
    }
    if (metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_UNSIGNAL_RIGHT_YIELD, 1.0f);
        AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_WAIT, wait_time > 0.0f ? dt : 0.0f);
    }
    float stop_dist = fmaxf(dist_to_end - DEFAULT_STOP_OFFSET - INTERSECTION_STOP_BUFFER, 0.75f);
    float req = -(ecs.speed[self] * ecs.speed[self]) / fmaxf(2.0f * stop_dist, 0.5f);
    req = clampf_cuda(req, -EMERGENCY_DECEL, -0.05f);
    req = fminf(req, -0.30f * max_decel);
    return req;
}
AVABM_DINLINE int connector_entry_unique_priority_key_ecs(int id, int lane, int next_lane, ECSArrays ecs, RoadNetwork road) {
    float L = valid_lane_ecs(lane, road) ? fmaxf(road.lane_length[lane], 0.1f) : 1.0f;
    float dist = valid_lane_ecs(lane, road) ? fmaxf(0.0f, L - ecs.s[id]) : 999.0f;
    float wait = clampf_cuda(ecs.connector_length[id], 0.0f, 60.0f);
    int turn = TURN_STRAIGHT;
    if (valid_lane_ecs(next_lane, road)) {
        int route_turn = route_turn_for_vehicle_ecs(id, ecs, road);
        turn = effective_turn_code_ecs(lane, next_lane, route_turn, road);
    }
    int turn_rank = 20;
    if (turn == TURN_STRAIGHT) turn_rank = 0;
    else if (turn == TURN_RIGHT) turn_rank = 8;
    else if (turn == TURN_LEFT) turn_rank = 16;
    int dist_part = clampi_cuda((int)(dist * 12.0f), 0, 12000);
    int wait_credit = clampi_cuda((int)(wait * 42.0f), 0, 2400);
    int behavior_credit = 0;
    if (ecs.driver_type[id] == HUMAN) {
        float behavior = clampf_cuda(0.55f * ecs.aggressiveness[id] + 0.35f * ecs.risk_tolerance[id] +
                0.10f * (1.0f - ecs.politeness[id]), 0.0f, 1.0f);
        behavior_credit = (int)(behavior * 16.0f);
    }
    int raw = 20000 + dist_part + turn_rank - wait_credit - behavior_credit;
    raw = clampi_cuda(raw, 0, 30000);
    return (raw << PRIORITY_GATE_ID_BITS) | (id & PRIORITY_GATE_ID_MASK);
}
AVABM_DINLINE bool connector_exit_space_clear_ecs(int self, int from_lane, int to_lane, ECSArrays ecs, RoadNetwork road,
        SpatialGrid grid, int max_entities) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return true;
    float hx, hy;
    float handoff_s = connector_exit_handoff_s(from_lane, to_lane, road);
    handoff_s = clampf_cuda(handoff_s, 0.0f, fmaxf(road.lane_length[to_lane], 0.1f));
    lane_xy_from_s(to_lane, handoff_s, road, hx, hy);
    int base = world_cell_index(hx, hy, grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return true;
    float v = fmaxf(0.0f, ecs.speed[self]);
    float required = fmaxf(CONNECTOR_EXIT_SPACE_MIN, fmaxf(PRIORITY_GATE_EXIT_SPACE,
            ecs.length[self] + MIN_BUMPER_GAP + v * CONNECTOR_EXIT_SPACE_TIME));
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf((required + 8.0f) / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool merge_bundle = lane_count_merge_transition_ecs(from_lane, to_lane, road);
                    bool on_surface = ecs.lane_id[j] == to_lane;
#if LANE_DROP_CONNECTOR_GROUP_CLEAR_ENABLED
                    if (merge_bundle && valid_lane_ecs(ecs.lane_id[j], road)) {
#if CONNECTOR_GROUP_CLEAR_SAME_TARGET_ONLY
                        on_surface = on_surface || (ecs.lane_id[j] == to_lane);
#else
                        on_surface = on_surface || lane_groups_same_ecs(ecs.lane_id[j], to_lane, road);
#endif
                    }
#endif
                    if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                        on_surface = on_surface || ecs.connector_from_lane[j] == to_lane || ecs.connector_to_lane[j] == to_lane;
#if LANE_DROP_CONNECTOR_GROUP_CLEAR_ENABLED
                        if (merge_bundle) {
#if CONNECTOR_GROUP_CLEAR_SAME_TARGET_ONLY
                            on_surface = on_surface || ecs.connector_from_lane[j] == to_lane || ecs.connector_to_lane[j] == to_lane;
#else
                            if (valid_lane_ecs(ecs.connector_from_lane[j], road)) {
                                on_surface = on_surface || lane_groups_same_ecs(ecs.connector_from_lane[j], to_lane, road);
                            }
                            if (valid_lane_ecs(ecs.connector_to_lane[j], road)) {
                                on_surface = on_surface || lane_groups_same_ecs(ecs.connector_to_lane[j], to_lane, road);
                            }
#endif
                        }
#endif
                    }
                    if (ecs.lane_change_active[j] != 0) {
                        on_surface = on_surface || ecs.lane_change_from_lane[j] == to_lane || ecs.lane_change_to_lane[j] == to_lane;
#if LANE_DROP_CONNECTOR_GROUP_CLEAR_ENABLED
                        if (merge_bundle) {
#if CONNECTOR_GROUP_CLEAR_SAME_TARGET_ONLY
                            on_surface = on_surface || ecs.lane_change_from_lane[j] == to_lane ||
                                    ecs.lane_change_to_lane[j] == to_lane;
#else
                            if (valid_lane_ecs(ecs.lane_change_from_lane[j], road)) {
                                on_surface = on_surface || lane_groups_same_ecs(ecs.lane_change_from_lane[j], to_lane, road);
                            }
                            if (valid_lane_ecs(ecs.lane_change_to_lane[j], road)) {
                                on_surface = on_surface || lane_groups_same_ecs(ecs.lane_change_to_lane[j], to_lane, road);
                            }
#endif
                        }
#endif
                    }
                    if (!on_surface) {
                        j = grid.cell_next[j];
                        guard++;
                        continue;
                    }
                    float eff_s = ecs.s[j];
                    if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                        if (ecs.connector_to_lane[j] == to_lane) {
                            int cf = ecs.connector_from_lane[j];
                            float clen = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                            float other_handoff = connector_exit_handoff_s(cf, to_lane, road);
                            eff_s = other_handoff - fmaxf(0.0f, clen - ecs.connector_s[j]);
                        } else if (ecs.connector_from_lane[j] == to_lane) {
                            eff_s = fmaxf(road.lane_length[to_lane], 0.1f) + ecs.connector_s[j];
#if LANE_DROP_CONNECTOR_GROUP_CLEAR_ENABLED
#if !CONNECTOR_GROUP_CLEAR_SAME_TARGET_ONLY
                        } else if (merge_bundle && valid_lane_ecs(ecs.connector_to_lane[j], road) &&
                                lane_groups_same_ecs(ecs.connector_to_lane[j], to_lane, road)) {
                            int cf = ecs.connector_from_lane[j];
                            int ct = ecs.connector_to_lane[j];
                            float clen = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                            float other_handoff = connector_exit_handoff_s(cf, ct, road);
                            eff_s = other_handoff - fmaxf(0.0f, clen - ecs.connector_s[j]);
                        } else if (merge_bundle && valid_lane_ecs(ecs.connector_from_lane[j], road) &&
                                lane_groups_same_ecs(ecs.connector_from_lane[j], to_lane, road)) {
                            eff_s = fmaxf(road.lane_length[to_lane], 0.1f) + ecs.connector_s[j];
#endif
#endif
                        }
                    }
                    float ds = eff_s - handoff_s;
                    if (ds >= -1.0f) {
                        float gap = ds - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                        if (gap < required) return false;
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return true;
}
AVABM_DINLINE bool zipper_merge_same_receiving_lanes_ecs(int a_next, int b_next, RoadNetwork road) {
#if ZIPPER_MERGE_ENABLED
    if (!valid_lane_ecs(a_next, road) || !valid_lane_ecs(b_next, road)) return false;
    if (a_next == b_next) return true;
#if LANE_DROP_TAPER_SAME_TARGET_ONLY
    return false;
#else
    return same_approach_same_direction_lanes_ecs(a_next, b_next, road);
#endif
#else
    return false;
#endif
}
AVABM_DINLINE bool zipper_merge_self_yields_ecs(int self, int lane, int next_lane, int other, int other_lane, int other_next_lane,
        ECSArrays ecs, RoadNetwork road, float current_time) {
#if ZIPPER_MERGE_ENABLED
    if (!valid_lane_ecs(lane, road) || !valid_lane_ecs(other_lane, road)) return false;
    if (!zipper_merge_same_receiving_lanes_ecs(next_lane, other_next_lane, road)) return false;
    if (!same_approach_same_direction_lanes_ecs(lane, other_lane, road) &&
            road.lane_end_node[lane] != road.lane_end_node[other_lane]) return false;
    float dist_self = fmaxf(0.0f, road.lane_length[lane] - ecs.s[self]);
    float dist_other = fmaxf(0.0f, road.lane_length[other_lane] - ecs.s[other]);
    bool one_near_merge = dist_self <= ZIPPER_MERGE_RANGE || dist_other <= ZIPPER_MERGE_RANGE;
    if (!one_near_merge) return false;
    if (dist_other + ZIPPER_MERGE_CLOSER_EPS < dist_self) return true;
    if (dist_self + ZIPPER_MERGE_CLOSER_EPS < dist_other) return false;
    int lo_lane = lane < other_lane ? lane : other_lane;
    int hi_lane = lane < other_lane ? other_lane : lane;
    float period = fmaxf(ZIPPER_MERGE_ALTERNATE_PERIOD, 0.10f);
    uint32_t slot = (uint32_t)floorf(fmaxf(0.0f, current_time) / period);
    int preferred_lane = ((slot & 1u) == 0u) ? lo_lane : hi_lane;
    if (lane != other_lane) {
        if (lane != preferred_lane) return true;
        return false;
    }
    return self > other;
#else
    return false;
#endif
}
AVABM_DINLINE bool connector_entry_clear_ecs(int self, int lane, int next_lane, ECSArrays ecs, DecisionSoA decision,
        RoadNetwork road, SpatialGrid grid, float current_time, int max_entities) {
    int node = road.lane_end_node[lane];
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return true;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(CONNECTOR_ENTRY_CLEAR_RADIUS / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    if (!connector_exit_space_clear_ecs(self, lane, next_lane, ecs, road, grid, max_entities)) {
        return false;
    }
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_conn = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_from = other_conn ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_conn ? ecs.connector_to_lane[j] : route_next_lane_for_vehicle_ecs(j, ecs, road);
                    if (!other_conn && decision.wants_connector[j] != 0) {
                        int decided_next = decision.connector_target_lane[j];
                        if (valid_lane_ecs(decided_next, road) && lane_connected(other_from, decided_next, road)) {
                            other_next = decided_next;
                        }
                    }
                    if (other_from >= 0 && other_from < road.num_lanes && road.lane_end_node[other_from] == node) {
                        bool relevant = intersection_conflict_relevant_vehicles_ecs(self, lane, next_lane, j, other_from,
                                other_next, other_conn, ecs, road);
                        float dxp = ecs.x[j] - ecs.x[self];
                        float dyp = ecs.y[j] - ecs.y[self];
                        float d = sqrtf(fmaxf(dxp * dxp + dyp * dyp, 0.001f));
                        bool same_receiving_merge = zipper_merge_same_receiving_lanes_ecs(next_lane, other_next, road);
                        bool lane_drop_gate = lane_count_merge_pair_conflict_ecs(lane, next_lane, other_from, other_next, road);
                        if (other_conn && (relevant || same_receiving_merge || lane_drop_gate)) {
                            float other_len = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                            float other_s = clampf_cuda(ecs.connector_s[j], 0.0f, other_len);
                            bool active_path_already_clear = other_s >= other_len * PRIORITY_GATE_ACTIVE_CLEAR_FRACTION ||
                                    (other_len - other_s) <= PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST;
                            bool waited_overlap_release = false;
#if OVERLAP_CONNECTOR_RELEASE_ENABLED
                            if ((same_receiving_merge || lane_drop_gate) && !active_path_already_clear) {
                                bool self_yields = zipper_merge_self_yields_ecs(self, lane, next_lane, j, other_from, other_next,
                                        ecs, road, current_time);
                                bool other_yields = zipper_merge_self_yields_ecs(j, other_from, other_next, self, lane, next_lane,
                                        ecs, road, current_time);
                                float self_wait = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                bool exit_has_space = connector_exit_space_clear_ecs(self, lane, next_lane, ecs, road, grid,
                                        max_entities);
                                waited_overlap_release = self_wait >= OVERLAP_CONNECTOR_RELEASE_WAIT && exit_has_space &&
                                        !self_yields && other_yields;
                            }
#endif
                            if (!active_path_already_clear && !waited_overlap_release) {
                                return false;
                            }
                        }
                        if (!other_conn && decision.wants_connector[j] != 0 && d < CONNECTOR_ENTRY_CLEAR_RADIUS) {
                            bool merge_relevant = relevant || same_receiving_merge || lane_drop_gate;
                            if (merge_relevant) {
                                if (same_receiving_merge || lane_drop_gate) {
                                    bool self_yields = zipper_merge_self_yields_ecs(self, lane, next_lane, j, other_from,
                                            other_next, ecs, road, current_time);
                                    bool other_yields = zipper_merge_self_yields_ecs(j, other_from, other_next, self, lane,
                                            next_lane, ecs, road, current_time);
                                    if (self_yields && !other_yields) {
                                        return false;
                                    }
                                    if (!self_yields && other_yields) {
                                        j = grid.cell_next[j];
                                        guard++;
                                        continue;
                                    }
                                }
                                int self_key = connector_entry_unique_priority_key_ecs(self, lane, next_lane, ecs, road);
                                int other_key = connector_entry_unique_priority_key_ecs(j, other_from, other_next, ecs, road);
                                if (other_key < self_key) {
                                    return false;
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return true;
}
AVABM_DINLINE bool intersection_priority_context_ecs(int id, ECSArrays ecs, RoadNetwork road, Signals signals, float current_time,
        int& lane, int& next_lane, int& turn, int& node, float& dist_to_end, bool& signal_permits) {
    lane = -1;
    next_lane = -1;
    turn = TURN_STRAIGHT;
    node = -1;
    dist_to_end = 1.0e9f;
    signal_permits = false;
    if (ecs.alive[id] != ENTITY_ALIVE || ecs.vehicle_state[id] != VEH_ON_LANE) return false;
    lane = ecs.lane_id[id];
    if (!valid_lane_ecs(lane, road)) return false;
    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    if (rid < 0 || rid >= road.num_routes || rpos < 0) return false;
    int repaired_pos = repair_route_pos_unless_missed_exit_tail_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[id] = repaired_pos;
    }
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || rpos < 0 || rpos >= route_len) return false;
    next_lane = route_next_lane_for_vehicle_ecs(id, ecs, road);
    rpos = ecs.route_pos[id];
    if (!valid_lane_ecs(next_lane, road) || !lane_connected(lane, next_lane, road)) return false;
    turn = route_turn_for_vehicle_ecs(id, ecs, road);
    int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
    adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
    adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
    if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
        next_lane = adjusted_next;
    }
    int interchange_source_lane = interchange_source_outer_lane_ecs(lane, next_lane, road);
    if (valid_lane_ecs(interchange_source_lane, road) && lane != interchange_source_lane) return false;
    if (!lane_legal_for_turn_ecs(lane, turn, road)) return false;
    float L = fmaxf(road.lane_length[lane], 0.1f);
    dist_to_end = fmaxf(0.0f, L - ecs.s[id]);
    if (dist_to_end > PRIORITY_GATE_APPROACH_RANGE) return false;
    node = road.lane_end_node[lane];
    if (node < 0 || node >= road.num_nodes) return false;
    bool inside_box = inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
    int st = get_signal_for_lane_turn(lane, turn, current_time, road, signals);
    signal_permits = (st == LIGHT_GREEN);
    if (st == LIGHT_YELLOW) {
        float v = fmaxf(ecs.speed[id], 0.0f);
        float stop_need = v * fmaxf(ecs.reaction_time[id], 0.25f) + (v * v) / (2.0f * 3.0f) + 2.0f;
        signal_permits = dist_to_end <= stop_need;
    }
    if (st == LIGHT_RED) signal_permits = false;
    if (inside_box) signal_permits = true;
    return signal_permits;
}
AVABM_DINLINE float priority_front_clear_gap_ecs(int id, ECSArrays ecs) {
    if (id < 0) return FRONT_CLEAR_PRIORITY_MIN_GAP;
    float v = fmaxf(0.0f, ecs.speed[id]);
    float len = fmaxf(ecs.length[id], 4.0f);
    return fmaxf(FRONT_CLEAR_PRIORITY_MIN_GAP, len + MIN_BUMPER_GAP + v * FRONT_CLEAR_PRIORITY_TIME);
}
AVABM_DINLINE bool priority_front_clear_ecs(int id, PerceptionSoA perception, ECSArrays ecs) {
    if (id < 0) return true;
    float fg = perception.front_gap != nullptr ? perception.front_gap[id] : 1.0e9f;
    if (!isfinite(fg)) fg = 0.0f;
    return fg > priority_front_clear_gap_ecs(id, ecs);
}
AVABM_DINLINE int priority_gate_key_ecs(int id, int lane, int next_lane, int turn, float dist_to_end, PerceptionSoA perception,
        ECSArrays ecs, RoadNetwork road) {
    float wait_time = ecs.connector_length[id];
    if (!isfinite(wait_time) || wait_time < 0.0f || ecs.vehicle_state[id] != VEH_ON_LANE) wait_time = 0.0f;
    int wait_bucket = clampi_cuda((int)floorf(wait_time * 6.0f), 0, 31);
    int dist_bucket = clampi_cuda((int)floorf(dist_to_end * 0.75f), 0, 63);
    bool inside_box = inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
    bool front_clear = priority_front_clear_ecs(id, perception, ecs);
    int turn_bias = 2;
    if (turn == TURN_STRAIGHT) turn_bias = 0;
    else if (turn == TURN_RIGHT) turn_bias = 5;
    else if (turn == TURN_LEFT) turn_bias = 9;
    int key = (31 - wait_bucket) * 64 + dist_bucket;
    if (front_clear) key -= PRIORITY_GATE_FRONT_CLEAR_BONUS;
    else key += PRIORITY_GATE_FRONT_BLOCK_PENALTY;
    if (inside_box) {
        key -= INTERSECTION_BOX_PRIORITY_BONUS;
    }
    if (ecs.driver_type[id] == HUMAN) {
        float behavior = 0.58f * clampf_cuda(ecs.aggressiveness[id], 0.0f, 1.0f) + 0.42f * clampf_cuda(ecs.risk_tolerance[id],
                0.0f, 1.0f) - 0.36f * clampf_cuda(ecs.politeness[id], 0.0f, 1.0f);
        int behavior_bias = clampi_cuda((int)floorf((0.48f - behavior) * PRIORITY_GATE_BEHAVIOR_BIAS_SCALE), -4, 4);
        key += behavior_bias;
    }
    if (indicator_active_ecs(id, ecs) && turn != TURN_STRAIGHT) {
        key -= 1;
    }
    key = clampi_cuda(key + turn_bias, 0, 2047);
    return key;
}
AVABM_DINLINE bool priority_gate_path_blocked_ecs(int self, int lane, int next_lane, int turn, int node, float dist_to_end,
        int self_packed, ECSArrays ecs, RoadNetwork road, Signals signals, SpatialGrid grid, PerceptionSoA perception,
        int max_entities, float current_time, float dt, float* metrics, bool* out_active_hold, bool* out_candidate_hold,
        bool* out_conflict_free) {
    if (out_active_hold) *out_active_hold = false;
    if (out_candidate_hold) *out_candidate_hold = false;
    if (out_conflict_free) *out_conflict_free = true;
    int base_cell = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base_cell < 0) return false;
    int bc_x = base_cell % grid.width;
    int bc_y = base_cell / grid.width;
    int cr = clampi_cuda((int)ceilf(PRIORITY_GATE_PATH_SCAN_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    bool blocked = false;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_conn = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_conn ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_conn ? ecs.connector_to_lane[j] : -1;
                    if (valid_lane_ecs(other_lane, road) && road.lane_end_node[other_lane] == node) {
                        if (other_conn) {
                            other_next = ecs.connector_to_lane[j];
                            if (valid_lane_ecs(other_next, road)) {
                                bool relevant = intersection_conflict_relevant_vehicles_ecs(self, lane, next_lane, j, other_lane,
                                        other_next, true, ecs, road);
                                if (relevant) {
                                    if (out_conflict_free) *out_conflict_free = false;
                                    float other_len = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                                    float other_s = clampf_cuda(ecs.connector_s[j], 0.0f, other_len);
                                    bool active_path_already_clear = other_s >= other_len * PRIORITY_GATE_ACTIVE_CLEAR_FRACTION ||
                                            (other_len - other_s) <= PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST;
                                    bool waited_overlap_release = false;
#if OVERLAP_CONNECTOR_RELEASE_ENABLED
                                    if (!active_path_already_clear) {
                                        bool same_receiving_merge = zipper_merge_same_receiving_lanes_ecs(next_lane, other_next,
                                                road);
                                        bool lane_drop_gate = lane_count_merge_pair_conflict_ecs(lane, next_lane, other_lane,
                                                other_next, road);
                                        bool self_front_clear = priority_front_clear_ecs(self, perception, ecs);
                                        float self_wait = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                        waited_overlap_release = (same_receiving_merge || lane_drop_gate) &&
                                                self_wait >= OVERLAP_CONNECTOR_RELEASE_WAIT && self_front_clear &&
                                                connector_exit_space_clear_ecs(self, lane, next_lane, ecs, road, grid,
                                                max_entities);
                                    }
#endif
                                    if (!active_path_already_clear && !waited_overlap_release) {
                                        if (out_active_hold) *out_active_hold = true;
                                        blocked = true;
                                    }
                                }
                            }
                        } else {
                            int olane, onext, oturn, onode;
                            float odist;
                            bool osignal;
                            if (intersection_priority_context_ecs(j, ecs, road, signals, current_time, olane, onext, oturn, onode,
                                    odist, osignal) && onode == node) {
                                bool relevant = intersection_conflict_relevant_vehicles_ecs(self, lane, next_lane, j, olane, onext,
                                        false, ecs, road);
                                if (relevant) {
                                    if (out_conflict_free) *out_conflict_free = false;
                                    bool pairwise_other_priority = false;
                                    bool unsignal_pair = !node_has_signal_ecs(node, signals);
                                    if (unsignal_pair) {
                                        float self_arrival = dist_to_end / fmaxf(ecs.speed[self], 1.0f);
                                        float other_arrival = odist / fmaxf(ecs.speed[j], 1.0f);
                                        pairwise_other_priority = unsignal_other_has_priority_ecs(self, j, lane, next_lane, olane,
                                                onext, false, self_arrival, other_arrival, road);
                                    }
                                    int other_key = priority_gate_key_ecs(j, olane, onext, oturn, odist, perception, ecs, road);
                                    int other_packed = (other_key << PRIORITY_GATE_ID_BITS) | (j & PRIORITY_GATE_ID_MASK);
                                    bool self_inside_box = inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
                                    bool other_inside_box = inside_intersection_box_ecs(odist, olane, onext, road);
                                    bool other_goes_first;
                                    if (self_inside_box && other_inside_box) {
                                        other_goes_first = other_packed < self_packed;
                                    } else if (self_inside_box != other_inside_box) {
                                        other_goes_first = other_inside_box;
                                    } else {
                                        float self_wait = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                        float other_wait = clampf_cuda(ecs.connector_length[j], 0.0f, 60.0f);
                                        bool timed_release_order = self_wait > 0.75f || other_wait > 0.75f ||
                                                dist_to_end <= PRIORITY_GATE_NEAR_LINE_DIST ||
                                                odist <= PRIORITY_GATE_NEAR_LINE_DIST;
                                        other_goes_first = (unsignal_pair &&
                                                !timed_release_order) ? pairwise_other_priority : (other_packed < self_packed);
                                    }
                                    bool self_front_clear = priority_front_clear_ecs(self, perception, ecs);
                                    bool other_front_clear = priority_front_clear_ecs(j, perception, ecs);
                                    float self_wait_now = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                    float other_wait_now = clampf_cuda(ecs.connector_length[j], 0.0f, 60.0f);
                                    if (other_goes_first && self_front_clear && !other_front_clear && !other_inside_box &&
                                            self_wait_now >= PRIORITY_GATE_BLOCKED_OTHER_IGNORE_WAIT &&
                                            other_wait_now < self_wait_now + 2.0f) {
                                        other_goes_first = false;
                                    }
#if GRIDLOCK_NODE_DRAIN_ENABLED
                                    if (other_goes_first && self_front_clear && !other_inside_box &&
                                            self_wait_now >= GRIDLOCK_NODE_DRAIN_WAIT &&
                                            dist_to_end <= GRIDLOCK_NODE_DRAIN_NEAR_DIST &&
                                            other_wait_now + GRIDLOCK_PAIR_RELEASE_WAIT_ADVANTAGE <= self_wait_now &&
                                            connector_exit_space_clear_ecs(self, lane, next_lane, ecs, road, grid, max_entities)) {
                                        other_goes_first = false;
                                        if (metrics != nullptr) {
                                            AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_RELEASE, 1.0f);
                                            AVABM_METRIC_ADD(metrics, METRIC_FRONT_SPACE_RELEASE, 1.0f);
                                        }
                                    }
#endif
                                    if (other_goes_first) {
                                        if (out_candidate_hold) *out_candidate_hold = true;
                                        blocked = true;
                                    }
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    if (blocked && metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_PRIORITY_PATH_BLOCK, 1.0f);
        if (out_active_hold && *out_active_hold) AVABM_METRIC_ADD(metrics, METRIC_PRIORITY_ACTIVE_PATH_HOLD, 1.0f);
        if (ecs.driver_type[self] == HUMAN) AVABM_METRIC_ADD(metrics, METRIC_HUMAN_AI_COURTESY_YIELD, 1.0f);
    }
    return blocked;
}
static inline void avabm_clear_int_async(int* data, int n, int value, cudaStream_t stream) {
    if (data == nullptr || n <= 0) return;
#if AVABM_USE_ASYNC_MEMSET_CLEAR
    if (value == 0 || value == -1) {
        cudaMemsetAsync(data, value == 0 ? 0 : 0xff, (size_t)n * sizeof(int), stream);
        return;
    }
#endif
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clear_int_kernel<<<blocks, threads, 0, stream>>>(data, n, value);
}
static inline void avabm_clear_float_zero_async(float* data, int n, cudaStream_t stream) {
    if (data == nullptr || n <= 0) return;
#if AVABM_USE_ASYNC_MEMSET_CLEAR
    cudaMemsetAsync(data, 0, (size_t)n * sizeof(float), stream);
#else
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clear_float_kernel<<<blocks, threads, 0, stream>>>(data, n, 0.0f);
#endif
}
static inline void avabm_begin_grid_rebuild_async(SpatialGrid& grid, int world_cells, cudaStream_t stream) {
    if (grid.cell_head == nullptr || world_cells <= 0) return;
#if AVABM_LAZY_GRID_ENABLED
    if (grid.cell_epoch != nullptr && grid.epoch > 0) {
        grid.epoch += 1;
        if (grid.epoch <= 0) {
            cudaMemsetAsync(grid.cell_epoch, 0, (size_t)world_cells * sizeof(int), stream);
            grid.epoch = 1;
        }
        return;
    }
#endif
    avabm_clear_int_async(grid.cell_head, world_cells, WORLD_CELL_EMPTY, stream);
}
static inline void avabm_begin_lane_grid_rebuild_async(SpatialGrid& grid, int num_lanes, cudaStream_t stream) {
    if (grid.lane_cell_head == nullptr || num_lanes <= 0 || grid.lane_cells_per_lane <= 0) return;
    int total = num_lanes * grid.lane_cells_per_lane;
    avabm_clear_int_async(grid.lane_cell_head, total, WORLD_CELL_EMPTY, stream);
}
AVABM_DINLINE bool avabm_lane_grid_available_ecs(SpatialGrid grid) {
    return grid.lane_cell_head != nullptr && grid.lane_cell_next != nullptr && grid.lane_cells_per_lane > 0;
}
AVABM_DINLINE int avabm_lane_grid_cell_from_s_ecs(int lane, float s, RoadNetwork road, SpatialGrid grid) {
    if (!avabm_lane_grid_available_ecs(grid)) return -1;
    if (!valid_lane_ecs(lane, road)) return -1;
    float L = fmaxf(road.lane_length[lane], 0.1f);
    float u = clampf_cuda(s / L, 0.0f, 0.999999f);
    int c = (int)floorf(u * (float)grid.lane_cells_per_lane);
    c = clampi_cuda(c, 0, grid.lane_cells_per_lane - 1);
    return lane * grid.lane_cells_per_lane + c;
}
AVABM_DINLINE int avabm_lane_grid_head_ecs(int lane, int cell, RoadNetwork road, SpatialGrid grid) {
    if (!avabm_lane_grid_available_ecs(grid)) return WORLD_CELL_EMPTY;
    if (!valid_lane_ecs(lane, road)) return WORLD_CELL_EMPTY;
    if (cell < 0 || cell >= grid.lane_cells_per_lane) return WORLD_CELL_EMPTY;
    return grid.lane_cell_head[lane * grid.lane_cells_per_lane + cell];
}
static inline void avabm_rebuild_active_list_async(ECSArrays ecs, int* active_ids, int* active_count, int max_entities,
        int threads, int entity_blocks, cudaStream_t stream) {
#if AVABM_ACTIVE_LIST_ENABLED
    if (active_ids == nullptr || active_count == nullptr || max_entities <= 0) return;
    cudaMemsetAsync(active_count, 0, sizeof(int), stream);
    compact_active_entities_kernel<<<entity_blocks, threads, 0, stream>>>(ecs, active_ids, active_count, max_entities);
#endif
}
static inline void avabm_rebuild_active_archetypes_async(ECSArrays ecs, const int* active_ids, const int* active_count,
        int* lane_active_ids, int* lane_active_count, int* connector_active_ids, int* connector_active_count, int max_entities,
        int threads, int active_blocks, cudaStream_t stream) {
#if AVABM_ACTIVE_LIST_ENABLED
    if (lane_active_ids == nullptr || lane_active_count == nullptr || connector_active_ids == nullptr ||
            connector_active_count == nullptr || max_entities <= 0) return;
    cudaMemsetAsync(lane_active_count, 0, sizeof(int), stream);
    cudaMemsetAsync(connector_active_count, 0, sizeof(int), stream);
    compact_active_archetypes_kernel<<<active_blocks, threads, 0,
            stream>>>(ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids,
            connector_active_count, max_entities);
#endif
}
static inline void avabm_rebuild_active_all_archetypes_async(ECSArrays ecs, int* active_ids, int* active_count,
        int* lane_active_ids, int* lane_active_count, int* connector_active_ids, int* connector_active_count, int max_entities,
        int threads, int entity_blocks, cudaStream_t stream) {
#if AVABM_ACTIVE_LIST_ENABLED
    if (active_ids == nullptr || active_count == nullptr || lane_active_ids == nullptr || lane_active_count == nullptr ||
            connector_active_ids == nullptr || connector_active_count == nullptr || max_entities <= 0) return;
    cudaMemsetAsync(active_count, 0, sizeof(int), stream);
    cudaMemsetAsync(lane_active_count, 0, sizeof(int), stream);
    cudaMemsetAsync(connector_active_count, 0, sizeof(int), stream);
    compact_active_all_archetypes_kernel<<<entity_blocks, threads, 0,
            stream>>>(ecs, active_ids, active_count, lane_active_ids, lane_active_count, connector_active_ids,
            connector_active_count, max_entities);
#endif
}
AVABM_DINLINE void insert_new_spawn_into_grid_ecs(int id, ECSArrays ecs, SpatialGrid grid) {
    if (id < 0) return;
    int cell = world_cell_index(ecs.x[id], ecs.y[id], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (cell < 0) return;
    grid_prepare_cell_for_write_ecs(grid, cell);
    int old_head;
    do {
        old_head = grid.cell_head[cell];
        grid.cell_next[id] = old_head;
    } while (atomicCAS(&grid.cell_head[cell], old_head, id) != old_head);
}
AVABM_DINLINE void init_driver_personality_ecs(int id, int dtype, uint32_t& rs, ECSArrays ecs) {
    float u1 = rand_uniform(rs);
    float u2 = rand_uniform(rs);
    float u3 = rand_uniform(rs);
    float u4 = rand_uniform(rs);
    float u5 = rand_uniform(rs);
    if (dtype == AV) {
        ecs.aggressiveness[id] = clampf_cuda(0.45f + 0.22f * u1, 0.30f, 0.75f);
        ecs.politeness[id] = clampf_cuda(0.65f + 0.25f * u2, 0.50f, 0.95f);
        ecs.risk_tolerance[id] = clampf_cuda(0.30f + 0.25f * u3, 0.20f, 0.65f);
        ecs.comfort_decel[id] = 3.2f + 0.8f * u4;
        ecs.desired_speed_factor[id] = 0.94f + 0.06f * u5;
        ecs.lc_cooldown[id] = 0.0f;
    } else {
        ecs.aggressiveness[id] = clampf_cuda(0.18f + 0.48f * u1, 0.10f, 0.78f);
        ecs.politeness[id] = clampf_cuda(0.24f + 0.48f * u2, 0.12f, 0.82f);
        ecs.risk_tolerance[id] = clampf_cuda(0.12f + 0.52f * u3, 0.05f, 0.72f);
        ecs.comfort_decel[id] = 2.2f + 1.4f * u4;
        ecs.desired_speed_factor[id] = 0.74f + 0.20f * u5;
        ecs.lc_cooldown[id] = 0.0f;
    }
}
AVABM_DINLINE bool spawn_area_clear(int lane, float spawn_s, float px, float py, float new_len, ECSArrays ecs, RoadNetwork road,
        SpatialGrid grid, int max_entities, int exclude) {
    int base = world_cell_index(px, py, grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return true;
    float limit = MAX_SPEED_FALLBACK;
    if (lane >= 0 && lane < road.num_lanes) {
        limit = road.lane_speed_limit[lane];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    }
    limit = fmaxf(limit, avabm_min_cruise_speed_mps_ecs());
    float required_gap = fmaxf(12.0f, new_len + 0.80f * limit);
    float radial_gap = fmaxf(7.0f, 0.65f * required_gap);
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                int alive_state_j = ecs.alive[j];
                bool blocks_spawn_j = alive_state_j == ENTITY_ALIVE;
#if AVABM_SPAWN_GRID_INSERT_FASTPATH
                blocks_spawn_j = blocks_spawn_j || alive_state_j == ENTITY_SPAWNING;
#endif
                if (j != exclude && blocks_spawn_j) {
                    float ddx = ecs.x[j] - px;
                    float ddy = ecs.y[j] - py;
                    float d2 = ddx * ddx + ddy * ddy;
                    if (d2 < radial_gap * radial_gap) return false;
                    bool same_surface = ecs.lane_id[j] == lane;
                    same_surface = same_surface || ecs.connector_from_lane[j] == lane;
                    same_surface = same_surface || ecs.connector_to_lane[j] == lane;
                    same_surface = same_surface || ecs.lane_change_from_lane[j] == lane;
                    same_surface = same_surface || ecs.lane_change_to_lane[j] == lane;
                    if (same_surface) {
                        float gap = fabsf(ecs.s[j] - spawn_s) - 0.5f * ecs.length[j] - 0.5f * new_len;
                        if (gap < required_gap) return false;
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return true;
}
AVABM_DINLINE bool spawn_area_clear_fullscan(int lane, float spawn_s, float px, float py, float new_len, ECSArrays ecs,
        RoadNetwork road, int max_entities, int exclude) {
    float limit = MAX_SPEED_FALLBACK;
    if (lane >= 0 && lane < road.num_lanes) {
        limit = road.lane_speed_limit[lane];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    }
    limit = fmaxf(limit, avabm_min_cruise_speed_mps_ecs());
    float required_gap = fmaxf(12.0f, new_len + 0.80f * limit);
    float radial_gap = fmaxf(7.0f, 0.65f * required_gap);
    for (int j = 0; j < max_entities; ++j) {
        if (j == exclude || ecs.alive[j] != ENTITY_ALIVE) continue;
        float ddx = ecs.x[j] - px;
        float ddy = ecs.y[j] - py;
        float d2 = ddx * ddx + ddy * ddy;
        if (d2 < radial_gap * radial_gap) return false;
        bool same_surface = ecs.lane_id[j] == lane;
        same_surface = same_surface || ecs.connector_from_lane[j] == lane;
        same_surface = same_surface || ecs.connector_to_lane[j] == lane;
        same_surface = same_surface || ecs.lane_change_from_lane[j] == lane;
        same_surface = same_surface || ecs.lane_change_to_lane[j] == lane;
        if (same_surface) {
            float gap = fabsf(ecs.s[j] - spawn_s) - 0.5f * ecs.length[j] - 0.5f * new_len;
            if (gap < required_gap) return false;
        }
    }
    return true;
}
AVABM_DINLINE void requeue_spawn_demand_for_vehicle_ecs(int lane, int route, SpawnConfig spawn) {
#if SPAWN_RACE_REQUEUE_ENABLED
    if (spawn.spawn_accumulator == nullptr || spawn.num_spawn_points <= 0) return;
    int fallback = -1;
    for (int p = 0; p < spawn.num_spawn_points; ++p) {
        if (spawn.spawn_lane[p] != lane) continue;
        if (fallback < 0) fallback = p;
        if (spawn.spawn_route[p] == route) {
            atomicAdd(&spawn.spawn_accumulator[p], 1.0f);
            return;
        }
    }
    if (fallback >= 0) {
        atomicAdd(&spawn.spawn_accumulator[fallback], 1.0f);
    }
#endif
}
AVABM_DINLINE float spawn_rate_vps_at(const SpawnConfig spawn, int p, float current_time) {
    float base = 0.0f;
    if (spawn.demand_vps != nullptr && p >= 0 && p < spawn.num_spawn_points) {
        base = fmaxf(spawn.demand_vps[p], 0.0f);
        if (!isfinite(base)) base = 0.0f;
    }
    if (spawn.demand_profile_vps == nullptr || spawn.demand_profile_has == nullptr || spawn.demand_profile_slots <= 0 || p < 0 ||
            p >= spawn.num_spawn_points || spawn.demand_profile_has[p] == 0) {
        return base;
    }
    int n = spawn.demand_profile_slots;
    float period = fmaxf(spawn.demand_profile_slot_seconds, 1.0e-3f);
    float cycle = period * (float)n;
    float local = current_time;
    if (isfinite(cycle) && cycle > 0.0f) {
        local = fmodf(current_time, cycle);
        if (local < 0.0f) local += cycle;
    } else {
        local = 0.0f;
    }
    float idxf = local / period;
    int i0 = clampi_cuda((int)floorf(idxf), 0, n - 1);
    int i1 = (i0 + 1) % n;
    float frac = idxf - floorf(idxf);
    frac = clampf_cuda(frac, 0.0f, 1.0f);
    int base_idx = p * n;
    float r0 = spawn.demand_profile_vps[base_idx + i0];
    float r1 = spawn.demand_profile_vps[base_idx + i1];
    if (!isfinite(r0) || r0 < 0.0f) r0 = base;
    if (!isfinite(r1) || r1 < 0.0f) r1 = base;
    float rate = r0 * (1.0f - frac) + r1 * frac;
    if (!isfinite(rate) || rate < 0.0f) rate = base;
    return fmaxf(rate, 0.0f);
}
AVABM_DINLINE bool vehicle_on_lane_surface_ecs(int id, int target_lane, ECSArrays ecs, RoadNetwork road) {
    if (target_lane < 0) return false;
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        if (ecs.connector_from_lane[id] == target_lane) return true;
        if (ecs.connector_to_lane[id] == target_lane) return true;
    }
    if (ecs.lane_id[id] == target_lane) return true;
    if (ecs.lane_change_active[id] != 0) {
        if (ecs.lane_change_from_lane[id] == target_lane) return true;
        if (ecs.lane_change_to_lane[id] == target_lane) return true;
    }
    return false;
}
AVABM_DINLINE void avabm_lane_grid_scan_front_ecs(int self, int query_lane, float query_self_s, float route_offset,
        float search_radius, ECSArrays ecs, RoadNetwork road, SpatialGrid grid, int max_entities, float& front_gap,
        float& front_speed, float& front_s, float& front_len, int& front_lane) {
#if AVABM_LANE_HASH_GRID_ENABLED
    if (!avabm_lane_grid_available_ecs(grid)) return;
    if (!valid_lane_ecs(query_lane, road)) return;
    float L = fmaxf(road.lane_length[query_lane], 0.1f);
    float s0 = clampf_cuda(query_self_s - 2.0f, 0.0f, L);
    float s1 = clampf_cuda(query_self_s + fmaxf(search_radius - route_offset, 0.0f) + 2.0f, 0.0f, L);
    if (s1 < s0) return;
    int c0 = clampi_cuda((int)floorf((s0 / L) * (float)grid.lane_cells_per_lane), 0, grid.lane_cells_per_lane - 1);
    int c1 = clampi_cuda((int)floorf((fminf(s1, L * 0.999999f) / L) * (float)grid.lane_cells_per_lane), 0,
            grid.lane_cells_per_lane - 1);
    for (int c = c0; c <= c1; ++c) {
        int j = avabm_lane_grid_head_ecs(query_lane, c, road, grid);
        int guard = 0;
        while (j >= 0 && guard < avabm_lane_scan_limit_ecs(max_entities)) {
            if (j != self && ecs.alive[j] == ENTITY_ALIVE && ecs.vehicle_state[j] == VEH_ON_LANE) {
                if (vehicle_on_lane_surface_ecs(j, query_lane, ecs, road)) {
                    float ds = route_offset + ecs.s[j] - query_self_s;
                    if (ds > 0.0f && ds <= search_radius + 1.0f) {
                        float gap = ds - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                        if (gap < front_gap) {
                            front_gap = gap;
                            front_speed = ecs.speed[j];
                            front_s = ecs.s[j];
                            front_len = ecs.length[j];
                            front_lane = query_lane;
                        }
                    }
                }
            }
            j = grid.lane_cell_next[j];
            guard++;
        }
    }
#else
    (void)self;
    (void)query_lane;
    (void)query_self_s;
    (void)route_offset;
    (void)search_radius;
    (void)ecs;
    (void)road;
    (void)grid;
    (void)max_entities;
    (void)front_gap;
    (void)front_speed;
    (void)front_s;
    (void)front_len;
    (void)front_lane;
#endif
}
AVABM_DINLINE bool avabm_lane_grid_scan_neighbors_ecs(int self, int target_lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        int max_entities, float search_radius, float& front_gap, float& front_speed, float& rear_gap, float& rear_speed) {
#if AVABM_LANE_HASH_GRID_ENABLED
    if (!avabm_lane_grid_available_ecs(grid)) return false;
    if (!valid_lane_ecs(target_lane, road)) return false;
    float L = fmaxf(road.lane_length[target_lane], 0.1f);
    float self_s = clampf_cuda(ecs.s[self], 0.0f, L);
    float s0 = clampf_cuda(self_s - search_radius - 2.0f, 0.0f, L);
    float s1 = clampf_cuda(self_s + search_radius + 2.0f, 0.0f, L);
    int c0 = clampi_cuda((int)floorf((s0 / L) * (float)grid.lane_cells_per_lane), 0, grid.lane_cells_per_lane - 1);
    int c1 = clampi_cuda((int)floorf((fminf(s1, L * 0.999999f) / L) * (float)grid.lane_cells_per_lane), 0,
            grid.lane_cells_per_lane - 1);
    for (int c = c0; c <= c1; ++c) {
        int j = avabm_lane_grid_head_ecs(target_lane, c, road, grid);
        int guard = 0;
        while (j >= 0 && guard < avabm_lane_scan_limit_ecs(max_entities)) {
            if (j != self && ecs.alive[j] == ENTITY_ALIVE && ecs.vehicle_state[j] == VEH_ON_LANE) {
                if (vehicle_on_lane_surface_ecs(j, target_lane, ecs, road)) {
                    float ds = ecs.s[j] - self_s;
                    if (fabsf(ds) <= search_radius + 1.0f) {
                        float gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                        if (ds > 0.0f) {
                            if (gap < front_gap) {
                                front_gap = gap;
                                front_speed = ecs.speed[j];
                            }
                        } else {
                            if (gap < rear_gap) {
                                rear_gap = gap;
                                rear_speed = ecs.speed[j];
                            }
                        }
                    }
                }
            }
            j = grid.lane_cell_next[j];
            guard++;
        }
    }
    return true;
#else
    (void)self;
    (void)target_lane;
    (void)ecs;
    (void)road;
    (void)grid;
    (void)max_entities;
    (void)search_radius;
    (void)front_gap;
    (void)front_speed;
    (void)rear_gap;
    (void)rear_speed;
    return false;
#endif
}
AVABM_DINLINE void find_front_on_route_ecs(int self, int lane, int next_lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        int max_entities, float search_radius, float* metrics, float& front_gap, float& front_speed, float& front_s,
        float& front_len, int& front_lane) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;
    front_s = 1.0e9f;
    front_len = 4.5f;
    front_lane = -1;
    float curr_L = valid_lane_ecs(lane, road) ? fmaxf(road.lane_length[lane], 0.1f) : 0.1f;
    float self_s = ecs.s[self];
    float remain_curr = curr_L - self_s;
    int dtype = ecs.driver_type[self];
    float sense_range = fminf(search_radius, sensor_front_range_for_driver(dtype));
    float sense_fov = sensor_half_fov_for_driver(dtype);
#if AVABM_LANE_HASH_GRID_ENABLED
    if (avabm_lane_grid_available_ecs(grid) && valid_lane_ecs(lane, road)) {
        avabm_lane_grid_scan_front_ecs(self, lane, self_s, 0.0f, search_radius, ecs, road, grid, max_entities, front_gap,
                front_speed, front_s, front_len, front_lane);
        if (valid_lane_ecs(next_lane, road) && remain_curr <= search_radius + 8.0f) {
            float next_handoff = connector_exit_handoff_s(lane, next_lane, road);
            float next_offset = fmaxf(0.0f, remain_curr - connector_entry_backoff_ecs(lane, next_lane,
                    road)) + connector_length_between_lanes(lane, next_lane, road);
            avabm_lane_grid_scan_front_ecs(self, next_lane, next_handoff, next_offset, search_radius, ecs, road, grid,
                    max_entities, front_gap, front_speed, front_s, front_len, front_lane);
        }
        bool near_connector = valid_lane_ecs(next_lane, road) && remain_curr <= fminf(search_radius + 8.0f, 90.0f);
        bool close_enough_front = front_lane >= 0 && front_gap < fmaxf(35.0f, ecs.speed[self] * 2.5f + 8.0f);
        if (!near_connector || close_enough_front) {
            return;
        }
    }
#endif
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
#if AVABM_TURBO_FAST_NEIGHBOR_SENSORS
                    if (metrics != nullptr && avabm_metric_enabled_ecs(METRIC_SENSOR_DETECTION)) {
                        float sensor_fwd, sensor_lat, sensor_dist;
                        bool detected = sensor_front_cone_detects_ecs(self, j, ecs, sense_range, sense_fov, sensor_fwd, sensor_lat,
                                sensor_dist);
                        if (detected) {
                            AVABM_METRIC_ADD(metrics, METRIC_SENSOR_DETECTION, 1.0f);
                        }
                    }
#else
                    float sensor_fwd, sensor_lat, sensor_dist;
                    bool detected = sensor_front_cone_detects_ecs(self, j, ecs, sense_range, sense_fov, sensor_fwd, sensor_lat,
                            sensor_dist);
                    if (detected) {
                        AVABM_METRIC_ADD(metrics, METRIC_SENSOR_DETECTION, 1.0f);
                    }
#endif
                    float route_ds = 1.0e9f;
                    int candidate_lane = -1;
                    if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                        int cf = ecs.connector_from_lane[j];
                        int ct = ecs.connector_to_lane[j];
                        if (cf == lane) {
                            float entry_backoff = connector_entry_backoff_ecs(cf, ct, road);
                            float ds = fmaxf(0.0f, remain_curr - entry_backoff) + ecs.connector_s[j];
                            if (ds > 0.0f && ds < route_ds) {
                                route_ds = ds;
                                candidate_lane = lane;
                            }
                        }
                        if (next_lane >= 0 && ct == next_lane) {
                            float ds = remain_curr + ecs.connector_s[j];
                            if (ds > 0.0f && ds < route_ds) {
                                route_ds = ds;
                                candidate_lane = next_lane;
                            }
                        }
                    } else {
                        bool on_curr = vehicle_on_lane_surface_ecs(j, lane, ecs, road);
                        bool on_next = next_lane >= 0 && next_lane < road.num_lanes &&
                                vehicle_on_lane_surface_ecs(j, next_lane, ecs, road);
                        if (on_curr) {
                            float ds = ecs.s[j] - self_s;
                            if (ds > 0.0f && ds < route_ds) {
                                route_ds = ds;
                                candidate_lane = lane;
                            }
                        }
                        if (on_next) {
                            float ds_next = connector_route_distance_to_next_lane_s(lane, next_lane, remain_curr, ecs.s[j], road);
                            if (ds_next > 0.0f && ds_next < route_ds) {
                                route_ds = ds_next;
                                candidate_lane = next_lane;
                            }
                        }
                    }
                    if (candidate_lane >= 0 && route_ds <= search_radius + 1.0f) {
                        float gap = route_ds - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                        if (gap < front_gap) {
                            front_gap = gap;
                            front_speed = ecs.speed[j];
                            if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                                front_s = curr_L + ecs.connector_s[j];
                            } else {
                                front_s = ecs.s[j];
                            }
                            front_len = ecs.length[j];
                            front_lane = candidate_lane;
                            AVABM_METRIC_ADD(metrics, METRIC_SENSOR_FRONT_HIT, 1.0f);
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}
AVABM_DINLINE void find_lane_neighbors_ecs(int self, int target_lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        int max_entities, float search_radius, float* metrics, float& front_gap, float& front_speed, float& rear_gap,
        float& rear_speed) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;
    rear_gap = 1.0e9f;
    rear_speed = 0.0f;
#if AVABM_LANE_HASH_GRID_ENABLED
    if (valid_lane_ecs(target_lane, road)) {
        float target_L = fmaxf(road.lane_length[target_lane], 0.1f);
        float self_s_on_target = clampf_cuda(ecs.s[self], 0.0f, target_L);
        bool midblock_query = self_s_on_target > search_radius + 8.0f && (target_L - self_s_on_target) > search_radius + 8.0f;
        if (midblock_query &&
                avabm_lane_grid_scan_neighbors_ecs(self, target_lane, ecs, road, grid, max_entities, search_radius, front_gap,
                front_speed, rear_gap, rear_speed)) {
            return;
        }
    }
#endif
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    int dtype = ecs.driver_type[self];
    float front_range = fminf(search_radius, sensor_front_range_for_driver(dtype));
    float mirror_range = fminf(search_radius, sensor_side_range_for_driver(dtype));
    float front_fov = sensor_half_fov_for_driver(dtype);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    if (vehicle_on_lane_surface_ecs(j, target_lane, ecs, road)) {
                        float eff_s = ecs.s[j];
                        if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                            if (ecs.connector_to_lane[j] == target_lane) {
                                int cf = ecs.connector_from_lane[j];
                                float clen = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                                float handoff_s = connector_exit_handoff_s(cf, target_lane, road);
                                eff_s = handoff_s - fmaxf(0.0f, clen - ecs.connector_s[j]);
                            } else if (ecs.connector_from_lane[j] == target_lane) {
                                eff_s = fmaxf(road.lane_length[target_lane], 0.1f) + ecs.connector_s[j];
                            }
                        }
                        float ds = eff_s - ecs.s[self];
                        float sf, sl, sd;
                        bool seen = false;
                        bool lane_geometry_seen = fabsf(ds) <= search_radius;
#if AVABM_TURBO_FAST_NEIGHBOR_SENSORS
                        if (metrics != nullptr && avabm_metric_enabled_ecs(METRIC_SENSOR_DETECTION)) {
                            if (ds >= 0.0f) {
                                seen = sensor_front_cone_detects_ecs(self, j, ecs, front_range, front_fov, sf, sl, sd);
                            } else {
                                seen = sensor_rear_mirror_detects_ecs(self, j, ecs, mirror_range, sf, sl, sd);
                            }
                        }
#else
                        if (ds >= 0.0f) {
                            seen = sensor_front_cone_detects_ecs(self, j, ecs, front_range, front_fov, sf, sl, sd);
                        } else {
                            seen = sensor_rear_mirror_detects_ecs(self, j, ecs, mirror_range, sf, sl, sd);
                        }
#endif
                        if (seen || lane_geometry_seen) {
                            if (seen) {
                                AVABM_METRIC_ADD(metrics, METRIC_SENSOR_DETECTION, 1.0f);
                            }
                            float gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                            if (ds > 0.0f) {
                                if (gap < front_gap) {
                                    front_gap = gap;
                                    front_speed = ecs.speed[j];
                                }
                            } else {
                                if (gap < rear_gap) {
                                    rear_gap = gap;
                                    rear_speed = ecs.speed[j];
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}
AVABM_DINLINE void indicator_merge_response_accel_ecs(int self, int lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        int max_entities, float current_time, float dt, float front_gap, float* metrics, float& yield_limit, float& assert_boost) {
    yield_limit = 1000.0f;
    assert_boost = 0.0f;
    if (!valid_lane_ecs(lane, road)) return;
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(INDICATOR_MERGE_COURTESY_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    bool human = ecs.driver_type[self] == HUMAN;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float yield_decel = human ? INDICATOR_MERGE_YIELD_DECEL_HUMAN : INDICATOR_MERGE_YIELD_DECEL_AV;
    float assert_accel = human ? INDICATOR_MERGE_ASSERT_ACCEL_HUMAN : INDICATOR_MERGE_ASSERT_ACCEL_AV;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && ecs.vehicle_state[j] == VEH_ON_LANE) {
                    int other_lane = ecs.lane_id[j];
                    int sig = indicator_state_ecs(j, ecs);
                    int target_lane = indicator_target_lane_ecs(other_lane, sig, road);
                    if (target_lane == lane && lanes_share_link_geometry_ecs(other_lane, lane, road) &&
                            ecs.turn_signal_time != nullptr && ecs.turn_signal_time[j] >= INDICATOR_MIN_ON_TIME) {
                        float ds = ecs.s[j] - ecs.s[self];
                        if (ds >= -INDICATOR_MERGE_SIDE_RANGE && ds <= INDICATOR_MERGE_COURTESY_RANGE) {
                            float raw_gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                            float close_gap = fmaxf(INDICATOR_MERGE_SIDE_RANGE, ecs.speed[self] * 0.85f + 5.5f);
                            if (raw_gap < close_gap) {
                                uint32_t slot = (uint32_t)floorf(current_time / 3.0f);
                                float h = hash01_ecs(((uint32_t)(self + 1) * 747796405u) ^ ((uint32_t)(j + 3) * 2891336453u) ^
                                        (slot * 277803737u));
                                float assert_chance = INDICATOR_MERGE_ASSERT_RATE;
                                assert_chance *= 0.50f + 0.75f * clampf_cuda(ecs.aggressiveness[self], 0.0f,
                                        1.0f) + 0.45f * clampf_cuda(ecs.risk_tolerance[self], 0.0f, 1.0f);
                                assert_chance *= 1.15f - 0.55f * clampf_cuda(ecs.politeness[self], 0.0f, 1.0f);
                                assert_chance = clampf_cuda(assert_chance, 0.02f, 0.28f);
                                float safe_front = fmaxf(ecs.length[self] + MIN_BUMPER_GAP + 4.0f, ecs.speed[self] * 0.75f + 6.0f);
                                bool can_assert = h < assert_chance && front_gap > safe_front;
                                if (can_assert) {
                                    float desired = desired_speed_ecs(self, lane, ecs, road);
                                    float a = (desired - ecs.speed[self]) / fmaxf(dt, 0.01f);
                                    a = clampf_cuda(a, 0.0f, max_accel * assert_accel);
                                    assert_boost = fmaxf(assert_boost, a);
                                    if (metrics != nullptr) AVABM_METRIC_ADD(metrics, METRIC_HUMAN_AI_ASSERTIVE_GO, 1.0f);
                                } else {
                                    float intensity = clampf_cuda((close_gap - raw_gap) / fmaxf(close_gap, 0.5f), 0.0f, 1.0f);
                                    float y = -yield_decel * (0.65f + 0.75f * intensity);
                                    yield_limit = fminf(yield_limit, y);
                                    if (metrics != nullptr) {
                                        AVABM_METRIC_ADD(metrics, METRIC_HUMAN_AI_COURTESY_YIELD, 1.0f);
                                        AVABM_METRIC_ADD(metrics, METRIC_INDICATOR_CONFLICT_YIELD, 1.0f);
                                    }
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}
AVABM_DINLINE bool try_reserve_slot_ecs(int self, int node, float arrival_time, float crossing_time, int* reservation_table,
        int num_nodes, float current_time, float* metrics) {
    if (reservation_table == nullptr) return true;
    if (node < 0 || node >= num_nodes) return true;
    float rel = arrival_time - current_time;
    int start_slot = (int)floorf(rel / RES_SLOT_DT);
    start_slot = clampi_cuda(start_slot, 0, RES_HORIZON_SLOTS - 1);
    int need_slots = (int)ceilf(crossing_time / RES_SLOT_DT);
    need_slots = clampi_cuda(need_slots, 1, 4);
    if (start_slot + need_slots > RES_HORIZON_SLOTS) {
        start_slot = RES_HORIZON_SLOTS - need_slots;
    }
    int base = node * RES_HORIZON_SLOTS;
    for (int k = 0; k < need_slots; ++k) {
        int idx = base + start_slot + k;
        int owner = reservation_table[idx];
        if (owner != RESERVATION_FREE && owner != self) {
            AVABM_METRIC_ADD(metrics, METRIC_RES_REJECT, 1.0f);
            return false;
        }
    }
    for (int k = 0; k < need_slots; ++k) {
        int idx = base + start_slot + k;
        int old = atomicCAS(&reservation_table[idx], RESERVATION_FREE, self);
        if (old != RESERVATION_FREE && old != self) {
            AVABM_METRIC_ADD(metrics, METRIC_RES_REJECT, 1.0f);
            return false;
        }
    }
    AVABM_METRIC_ADD(metrics, METRIC_RES_ACCEPT, 1.0f);
    return true;
}
AVABM_DINLINE float signal_accel_limit_ecs(int lane, int turn, float ss, float v, float reaction_time, int dtype,
        float current_time, const RoadNetwork road, const Signals signals, int* out_state, float* out_stop_s) {
    if (out_state) *out_state = LIGHT_GREEN;
    if (out_stop_s) *out_stop_s = 1.0e9f;
    float L = fmaxf(road.lane_length[lane], 0.1f);
    float stop_s = fmaxf(0.0f, L - DEFAULT_STOP_OFFSET);
    float d = stop_s - ss;
    int st = get_signal_for_lane_turn(lane, turn, current_time, road, signals);
    if (out_state) *out_state = st;
    if (d > 190.0f) return 1000.0f;
    if (st == LIGHT_GREEN) return 1000.0f;
    if (d < -0.25f) return 1000.0f;
    bool human = dtype == HUMAN;
    float rt = fmaxf(reaction_time, human ? 0.75f : 0.12f);
    float comfortable_b = human ? (0.78f * MAX_DECEL_HUMAN) : (0.86f * MAX_DECEL_AV);
    comfortable_b = fmaxf(comfortable_b, 1.2f);
    float comfortable_stop = v * rt + (v * v) / fmaxf(2.0f * comfortable_b, 0.1f) + INTERSECTION_STOP_BUFFER;
    bool stop = false;
    if (st == LIGHT_RED) {
        stop = true;
    } else {
        stop = d > comfortable_stop * 0.72f;
    }
    if (!stop) return 1000.0f;
    if (out_stop_s) *out_stop_s = stop_s;
    if (v < 0.28f && d > SIGNAL_CREEP_HOLD_DIST + 1.25f) {
        return 1000.0f;
    }
    float effective_d = fmaxf(d - fminf(v * rt * 0.35f, d * 0.45f), 0.55f);
    float req = -(v * v) / fmaxf(2.0f * effective_d, 0.5f);
    if (st == LIGHT_RED && d < comfortable_stop * 0.85f) {
        req = fminf(req, -comfortable_b);
    }
    return clampf_cuda(req, -EMERGENCY_DECEL, 0.0f);
}
AVABM_DINLINE float effective_s_on_lane_surface_ecs(int id, int target_lane, ECSArrays ecs, RoadNetwork road) {
    if (!valid_lane_ecs(target_lane, road)) return 1.0e9f;
    float eff_s = ecs.s[id];
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        if (ecs.connector_to_lane[id] == target_lane) {
            int cf = ecs.connector_from_lane[id];
            float clen = fmaxf(ecs.connector_length[id], CONNECTOR_MIN_LEN);
            float handoff_s = connector_exit_handoff_s(cf, target_lane, road);
            eff_s = handoff_s - fmaxf(0.0f, clen - ecs.connector_s[id]);
        } else if (ecs.connector_from_lane[id] == target_lane) {
            eff_s = fmaxf(road.lane_length[target_lane], 0.1f) + ecs.connector_s[id];
        }
    }
    return eff_s;
}
AVABM_DINLINE bool indicator_targets_lane_ecs(int id, int target_lane, ECSArrays ecs, RoadNetwork road) {
    if (!valid_lane_ecs(target_lane, road)) return false;
    if (ecs.lane_change_active[id] != 0) {
        return ecs.lane_change_to_lane[id] == target_lane;
    }
    int lane = ecs.lane_id[id];
    if (!valid_lane_ecs(lane, road)) return false;
    int sig = indicator_state_ecs(id, ecs);
    if (sig == INDICATOR_LEFT) {
        return geometric_left_neighbor_ecs(lane, road) == target_lane;
    }
    if (sig == INDICATOR_RIGHT) {
        return geometric_right_neighbor_ecs(lane, road) == target_lane;
    }
    return false;
}
AVABM_DINLINE bool lane_change_assertive_response_ecs(int responder, int merger, float current_time, ECSArrays ecs) {
    float behavior = clampf_cuda(0.62f * ecs.aggressiveness[responder] +
            0.42f * ecs.risk_tolerance[responder] - 0.34f * ecs.politeness[responder], 0.0f, 1.0f);
    float p = clampf_cuda(LANE_CHANGE_COOP_ASSERTIVE_PROB * (0.30f + 1.15f * behavior), 0.015f, 0.24f);
    uint32_t slot = (uint32_t)floorf(current_time * 1.7f);
    uint32_t h = (uint32_t)responder * 1103515245u ^ (uint32_t)merger * 2654435761u ^ slot * 2246822519u;
    return hash01_ecs(h) < p;
}
AVABM_DINLINE bool lane_change_assertive_blocker_ecs(int self, int target_lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        float current_time, int max_entities, float* metrics) {
    if (!valid_lane_ecs(target_lane, road)) return true;
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return false;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(LC_INDICATOR_SIDE_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float self_s = ecs.s[self];
    float self_v = fmaxf(0.0f, ecs.speed[self]);
    float block_gap = fmaxf(LC_ASSERTIVE_BLOCK_GAP, self_v * LC_ASSERTIVE_BLOCK_TIME + ecs.length[self] + MIN_BUMPER_GAP);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && vehicle_on_lane_surface_ecs(j, target_lane, ecs, road)) {
                    float eff_s = effective_s_on_lane_surface_ecs(j, target_lane, ecs, road);
                    float ds = eff_s - self_s;
                    if (ds > LC_INDICATOR_SIDE_FRONT || ds < -LC_INDICATOR_SIDE_REAR) {
                        j = grid.cell_next[j];
                        guard++;
                        continue;
                    }
                    float gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                    bool close = gap < block_gap;
                    bool closing_fast = ds < 0.0f && ecs.speed[j] > self_v + 0.45f && gap < block_gap * 1.35f;
                    bool assertive = lane_change_assertive_response_ecs(j, self, current_time, ecs);
                    if (close && (assertive || closing_fast || ecs.accel[j] > 0.35f)) {
                        if (metrics != nullptr) AVABM_METRIC_ADD(metrics, METRIC_LC_REJECT, 1.0f);
                        return true;
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    return false;
}
AVABM_DINLINE float lane_change_courtesy_accel_limit_ecs(int self, int lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        float current_time, int max_entities, float* out_boost, bool* out_assertive) {
    if (out_boost) *out_boost = 0.0f;
    if (out_assertive) *out_assertive = false;
    if (!valid_lane_ecs(lane, road)) return 1000.0f;
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(LC_INDICATOR_SIDE_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    bool human = ecs.driver_type[self] == HUMAN;
    float decel_base = human ? LC_COURTESY_DECEL_HUMAN : LC_COURTESY_DECEL_AV;
    float accel_base = human ? LC_ASSERTIVE_ACCEL_HUMAN : LC_ASSERTIVE_ACCEL_AV;
    float limit = 1000.0f;
    float boost = 0.0f;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && indicator_targets_lane_ecs(j, lane, ecs, road)) {
                    int jl = ecs.lane_id[j];
                    if (valid_lane_ecs(jl, road) && same_approach_same_direction_lanes_ecs(lane, jl, road)) {
                        float ds = ecs.s[j] - ecs.s[self];
                        if (ds <= LC_INDICATOR_SIDE_FRONT && ds >= -LC_INDICATOR_SIDE_REAR) {
                            float gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                            float zone = fmaxf(LC_ASSERTIVE_BLOCK_GAP * LANE_CHANGE_COOP_ZONE_MULT,
                                    ecs.length[self] + ecs.length[j] + MIN_BUMPER_GAP + ecs.speed[self] * 0.85f);
                            if (gap < zone) {
                                bool assertive = lane_change_assertive_response_ecs(self, j, current_time, ecs);
                                float severity = clampf_cuda((zone - gap) / fmaxf(zone, 0.5f), 0.0f, 1.0f);
                                if (assertive) {
                                    boost = fmaxf(boost, accel_base * (0.35f + 0.80f * severity));
                                    if (out_assertive) *out_assertive = true;
                                } else {
                                    limit = fminf(limit, -decel_base * (0.35f + 0.80f * severity));
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    if (out_boost) *out_boost = boost;
    return limit;
}
AVABM_DINLINE bool open_lane_candidate_safe_ecs(int self, float target_front_gap, float target_front_speed, float target_rear_gap,
        float target_rear_speed, ECSArrays ecs, bool urgent) {
    bool human = ecs.driver_type[self] == HUMAN;
    float v = fmaxf(0.0f, ecs.speed[self]);
    if (!isfinite(target_front_gap)) target_front_gap = 0.0f;
    if (!isfinite(target_rear_gap)) target_rear_gap = 0.0f;
    if (!isfinite(target_front_speed)) target_front_speed = 0.0f;
    if (!isfinite(target_rear_speed)) target_rear_speed = 0.0f;
    float rear_req = human ? OPEN_LANE_REAR_GAP_HUMAN : OPEN_LANE_REAR_GAP_AV;
    rear_req += fmaxf(0.0f, target_rear_speed - v) * OPEN_LANE_TARGET_REAR_SPEED_TIME;
    rear_req += 0.5f * fmaxf(ecs.length[self], 4.0f) + MIN_BUMPER_GAP;
    float front_req = human ? OPEN_LANE_FRONT_GAP_HUMAN : OPEN_LANE_FRONT_GAP_AV;
    front_req += fmaxf(0.0f, v - target_front_speed) * OPEN_LANE_TARGET_FRONT_SPEED_TIME;
    front_req += 0.5f * fmaxf(ecs.length[self], 4.0f) + MIN_BUMPER_GAP;
    if (urgent) {
        rear_req *= 0.82f;
        front_req *= 0.84f;
    }
    return target_front_gap > front_req && target_rear_gap > rear_req && !(target_rear_speed > v + 1.2f &&
            target_rear_gap < rear_req * 1.45f);
}
AVABM_DINLINE int pick_open_lane_target_ecs(int self, int lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        PerceptionSoA perception, float current_time, int max_entities, bool allow_congestion_escape, bool allow_spread) {
    if (!valid_lane_ecs(lane, road) || ecs.vehicle_state[self] != VEH_ON_LANE) return -1;
    if (ecs.lane_change_active[self] != 0) return -1;
    float L = fmaxf(road.lane_length[lane], 0.1f);
    float dist_to_end = fmaxf(0.0f, L - ecs.s[self]);
    float v = fmaxf(0.0f, ecs.speed[self]);
    bool human = ecs.driver_type[self] == HUMAN;
    float current_gap = perception.front_gap != nullptr ? perception.front_gap[self] : 1.0e9f;
    float current_front_speed = perception.front_speed != nullptr ? perception.front_speed[self] : 0.0f;
    if (!isfinite(current_gap)) current_gap = 0.0f;
    float blocked_gap = fmaxf(CONGESTION_ESCAPE_CURRENT_GAP, fmaxf(ecs.length[self], 4.0f) + MIN_BUMPER_GAP + v * 1.05f);
    bool front_lane_blocked = current_gap < blocked_gap || (current_gap < blocked_gap * 1.7f && current_front_speed + 2.0f < v);
    bool congestion_mode =
#if CONGESTION_ESCAPE_LC_ENABLED
    allow_congestion_escape && front_lane_blocked && ecs.lc_cooldown[self] <= CONGESTION_ESCAPE_COOLDOWN_READY &&
            L >= CONGESTION_ESCAPE_MIN_LINK_LENGTH && dist_to_end >= CONGESTION_ESCAPE_MIN_DIST_TO_NODE;
#else
    false;
#endif
    bool spread_mode =
#if LANE_SPREAD_CHANGE_ENABLED
    allow_spread && ecs.lc_cooldown[self] <= LANE_SPREAD_COOLDOWN_READY && L >= LANE_SPREAD_MIN_LINK_LENGTH &&
            dist_to_end >= LANE_SPREAD_MIN_DIST_TO_NODE && current_gap >= LANE_SPREAD_MIN_CURRENT_GAP;
#else
    false;
#endif
    if (!congestion_mode && !spread_mode) return -1;
    int cur_group_idx = -1;
    int cur_group_count = lane_group_count_and_index_ecs(lane, road, cur_group_idx);
    bool right_edge_pressure = false;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
    right_edge_pressure = allow_spread && cur_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP && cur_group_idx >= 0 &&
            cur_group_idx <= RIGHT_EDGE_BOTTLENECK_IDX_LIMIT;
#endif
    int balance_target_idx = -1;
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
    if (allow_spread && cur_group_count > 1 && cur_group_idx >= 0) {
        uint32_t lane_balance_hash = hash_u32_ecs(((uint32_t)(self + 701) * 1103515245u) ^ ((uint32_t)(ecs.route_id[self] +
                53) * 2654435761u) ^ ((uint32_t)(road.lane_start_node[lane] + 17) * 747796405u) ^
                ((uint32_t)(road.lane_end_node[lane] + 29) * 2891336453u));
        if (cur_group_count >= 3) {
            balance_target_idx = 1 + (int)(lane_balance_hash % (uint32_t)(cur_group_count - 1));
        } else {
            balance_target_idx = (int)(lane_balance_hash % (uint32_t)cur_group_count);
        }
    }
#endif
    int candidates[2] = {
        geometric_left_neighbor_ecs(lane, road), geometric_right_neighbor_ecs(lane, road)
    };
    int best_lane = -1;
    float best_score = -1.0e20f;
    float best_fg = 1.0e9f;
    float best_fv = 0.0f;
    float best_rg = 1.0e9f;
    float best_rv = 0.0f;
    for (int k = 0; k < 2; ++k) {
        int target = candidates[k];
        if (!valid_lane_ecs(target, road)) continue;
        if (!same_approach_same_direction_lanes_ecs(lane, target, road)) continue;
        float fg, fv, rg, rv;
        float search_r = congestion_mode ? CONGESTION_ESCAPE_SEARCH_RADIUS : LANE_SPREAD_SEARCH_RADIUS;
        find_lane_neighbors_ecs(self, target, ecs, road, grid, max_entities, search_r, nullptr, fg, fv, rg, rv);
        bool safe = open_lane_candidate_safe_ecs(self, fg, fv, rg, rv, ecs, congestion_mode);
        if (!safe) continue;
        float front_gain = fg - current_gap;
        float rear_gain = rg - (human ? OPEN_LANE_REAR_GAP_HUMAN : OPEN_LANE_REAR_GAP_AV);
        int target_group_idx = -1;
        int target_group_count = lane_group_count_and_index_ecs(target, road, target_group_idx);
        bool target_is_inner = cur_group_count == target_group_count && target_group_idx >= 0 && cur_group_idx >= 0 &&
                target_group_idx > cur_group_idx;
        bool target_moves_toward_balance = false;
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
        if (balance_target_idx >= 0 && target_group_count == cur_group_count && target_group_idx >= 0 && cur_group_idx >= 0) {
            int cur_err = cur_group_idx > balance_target_idx ? cur_group_idx - balance_target_idx :
                    balance_target_idx - cur_group_idx;
            int tgt_err = target_group_idx > balance_target_idx ? target_group_idx - balance_target_idx :
                    balance_target_idx - target_group_idx;
            target_moves_toward_balance = tgt_err < cur_err;
        }
#endif
        bool empty_interval = fg >= LANE_SPREAD_EMPTY_FRONT_GAP && rg >= LANE_SPREAD_EMPTY_REAR_GAP;
        bool good_for_congestion = congestion_mode && (front_gain >= CONGESTION_ESCAPE_FRONT_GAIN || fg >= blocked_gap * 1.25f ||
                empty_interval);
        bool stable_random_preference = false;
        if (spread_mode) {
            uint32_t slot = (uint32_t)floorf(current_time / 5.0f);
            uint32_t h = hash_u32_ecs(((uint32_t)(self + 1) * 747796405u) ^ ((uint32_t)(ecs.route_id[self] + 13) * 2891336453u) ^
                    ((uint32_t)(target + 31) * 277803737u) ^ slot);
            stable_random_preference = (h & 7u) == 0u;
        }
        bool good_for_spread = spread_mode && (front_gain >= LANE_SPREAD_FRONT_GAIN || (empty_interval &&
                stable_random_preference) || (front_gain >= 6.0f && rear_gain >= LANE_SPREAD_REAR_GAIN && stable_random_preference)
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
        || (target_moves_toward_balance && front_gain >= -10.0f && rear_gain >= -4.0f)
#endif
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
        || (right_edge_pressure && target_is_inner && front_gain >= -RIGHT_EDGE_SAFE_FRONT_LOSS)
#endif
        );
        if (!good_for_congestion && !good_for_spread) continue;
        float side_hash = hash01_ecs(((uint32_t)(self + 5) * 1103515245u) ^ ((uint32_t)(target + 7) * 2654435761u));
        float score = fg + 0.22f * rg - 0.45f * fmaxf(0.0f, rv - v) + side_hash;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
        if (target_group_count == cur_group_count && target_group_idx >= 0 && cur_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP) {
            score += (float)target_group_idx * RIGHT_EDGE_INNER_BONUS;
            if (target_group_idx == 0) score -= RIGHT_EDGE_BOTTLENECK_PENALTY;
            if (right_edge_pressure && target_group_idx > cur_group_idx) score += RIGHT_EDGE_BOTTLENECK_PENALTY * 0.75f;
        }
#endif
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
        if (balance_target_idx >= 0 && target_group_count == cur_group_count && target_group_idx >= 0) {
            int err = target_group_idx > balance_target_idx ? target_group_idx - balance_target_idx :
                    balance_target_idx - target_group_idx;
            score += THROUGH_LANE_BALANCE_TARGET_BONUS - (float)err * THROUGH_LANE_BALANCE_DISTANCE_PENALTY;
            if (cur_group_count >= 3 && target_group_idx == 0) score -= THROUGH_LANE_BALANCE_RIGHT_EDGE_PENALTY;
        }
#endif
        if (good_for_congestion) score += 35.0f;
        if (front_lane_blocked && fg > blocked_gap * 2.0f) score += 12.0f;
        if (score > best_score) {
            best_score = score;
            best_lane = target;
            best_fg = fg;
            best_fv = fv;
            best_rg = rg;
            best_rv = rv;
        }
    }
#if OPEN_LANE_EMPTIEST_GROUP_SCAN_ENABLED
    if ((best_lane < 0 || right_edge_pressure) && (spread_mode || congestion_mode)) {
        int cur_idx = cur_group_idx;
        int group_count = cur_group_count;
        if (group_count > 1 && cur_idx >= 0) {
            int desired_group_lane = -1;
            float desired_score = current_gap;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
            if (group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP) {
                desired_score += (float)cur_idx * RIGHT_EDGE_INNER_BONUS;
                if (cur_idx == 0) desired_score -= RIGHT_EDGE_BOTTLENECK_PENALTY;
            }
#endif
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
            if (balance_target_idx >= 0) {
                int err = cur_idx > balance_target_idx ? cur_idx - balance_target_idx : balance_target_idx - cur_idx;
                desired_score += THROUGH_LANE_BALANCE_TARGET_BONUS - (float)err * THROUGH_LANE_BALANCE_DISTANCE_PENALTY;
                if (group_count >= 3 && cur_idx == 0) desired_score -= THROUGH_LANE_BALANCE_RIGHT_EDGE_PENALTY;
            }
#endif
            uint32_t slot = (uint32_t)floorf(current_time / OPEN_LANE_EMPTIEST_SCAN_PERIOD);
            int right = rightmost_lane_in_group_ecs(lane, road);
            int cur = right;
            for (int kk = 0; kk < CRUISE_RANDOM_LANE_MAX_GROUP && valid_lane_ecs(cur, road); ++kk) {
                if (cur != lane && same_approach_same_direction_lanes_ecs(lane, cur, road)) {
                    float fg, fv, rg, rv;
                    find_lane_neighbors_ecs(self, cur, ecs, road, grid, max_entities, LANE_SPREAD_SEARCH_RADIUS, nullptr, fg, fv,
                            rg, rv);
                    bool empty_interval = fg >= LANE_SPREAD_EMPTY_FRONT_GAP && rg >= LANE_SPREAD_EMPTY_REAR_GAP;
                    float hash_bias = hash01_ecs(((uint32_t)(self + 113) * 1103515245u) ^ ((uint32_t)(cur + 19) * 2654435761u) ^
                            (slot * 747796405u));
                    float closing_penalty = fmaxf(0.0f, rv - v) * 0.55f;
                    float score = fg + 0.30f * rg - closing_penalty + hash_bias;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                    if (group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP) {
                        score += (float)kk * RIGHT_EDGE_INNER_BONUS;
                        if (kk == 0) score -= RIGHT_EDGE_BOTTLENECK_PENALTY;
                        if (right_edge_pressure && kk > cur_idx) score += RIGHT_EDGE_BOTTLENECK_PENALTY * 0.60f;
                    }
#endif
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
                    if (balance_target_idx >= 0) {
                        int err = kk > balance_target_idx ? kk - balance_target_idx : balance_target_idx - kk;
                        score += THROUGH_LANE_BALANCE_TARGET_BONUS - (float)err * THROUGH_LANE_BALANCE_DISTANCE_PENALTY;
                        if (group_count >= 3 && kk == 0) score -= THROUGH_LANE_BALANCE_RIGHT_EDGE_PENALTY;
                    }
#endif
                    if (empty_interval) score += 18.0f;
                    if (fg > current_gap + OPEN_LANE_EMPTIEST_FRONT_GAIN) score += 7.0f;
                    float gain_need = OPEN_LANE_EMPTIEST_SCORE_GAIN;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                    if (right_edge_pressure && kk > cur_idx) gain_need = RIGHT_EDGE_SCAN_SCORE_GAIN;
#endif
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
                    if (balance_target_idx >= 0) {
                        int cur_err = cur_idx > balance_target_idx ? cur_idx - balance_target_idx : balance_target_idx - cur_idx;
                        int tgt_err = kk > balance_target_idx ? kk - balance_target_idx : balance_target_idx - kk;
                        if (tgt_err < cur_err) gain_need = fminf(gain_need, -12.0f);
                    }
#endif
                    if (score > desired_score + gain_need) {
                        desired_score = score;
                        desired_group_lane = cur;
                    }
                }
                int nb = geometric_left_neighbor_ecs(cur, road);
                if (!valid_lane_ecs(nb, road)) break;
                cur = nb;
            }
            int step = valid_lane_ecs(desired_group_lane, road) ? adjacent_lane_toward_specific_lane_ecs(lane, desired_group_lane,
                    road) : -1;
            if (valid_lane_ecs(step, road) && same_approach_same_direction_lanes_ecs(lane, step, road)) {
                float fg, fv, rg, rv;
                find_lane_neighbors_ecs(self, step, ecs, road, grid, max_entities, LANE_SPREAD_SEARCH_RADIUS, nullptr, fg, fv, rg,
                        rv);
                bool safe_step = open_lane_candidate_safe_ecs(self, fg, fv, rg, rv, ecs, congestion_mode);
                bool enough_step_gain = fg > current_gap + 3.0f || fg >= LANE_SPREAD_EMPTY_FRONT_GAP;
                int step_idx = -1;
                lane_group_count_and_index_ecs(step, road, step_idx);
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                if (right_edge_pressure && step_idx > cur_idx && fg >= current_gap - RIGHT_EDGE_SAFE_FRONT_LOSS) {
                    enough_step_gain = true;
                }
#endif
#if THROUGH_LANE_BALANCE_TARGET_ENABLED
                if (balance_target_idx >= 0 && step_idx >= 0) {
                    int cur_err = cur_idx > balance_target_idx ? cur_idx - balance_target_idx : balance_target_idx - cur_idx;
                    int step_err = step_idx > balance_target_idx ? step_idx - balance_target_idx : balance_target_idx - step_idx;
                    if (step_err < cur_err && fg >= current_gap - 10.0f) enough_step_gain = true;
                }
#endif
                if (safe_step && enough_step_gain) {
                    best_lane = step;
                    best_fg = fg;
                    best_fv = fv;
                    best_rg = rg;
                    best_rv = rv;
                }
            }
        }
    }
#endif
    if (best_lane >= 0) {
        perception.target_front_gap[self] = best_fg;
        perception.target_front_speed[self] = best_fv;
        perception.target_rear_gap[self] = best_rg;
        perception.target_rear_speed[self] = best_rv;
    }
    return best_lane;
}
AVABM_DINLINE bool mobil_decision_ecs(int i, int target_lane, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        PerceptionSoA perception, float* metrics, float current_time, int max_entities, bool mandatory, bool opportunistic) {
    if (target_lane < 0 || target_lane >= road.num_lanes) return false;
    if (ecs.lc_cooldown[i] > 0.0f && !mandatory && !opportunistic) return false;
    if (lane_change_assertive_blocker_ecs(i, target_lane, ecs, road, grid, current_time, max_entities, metrics)) {
        return false;
    }
    bool human = ecs.driver_type[i] == HUMAN;
    bool flow_balance_mandatory = mandatory && opportunistic;
    AVABM_METRIC_ADD(metrics, METRIC_MOBIL_EVAL, 1.0f);
    float tf_gap = perception.target_front_gap[i];
    float tf_v = perception.target_front_speed[i];
    float tr_gap = perception.target_rear_gap[i];
    float tr_v = perception.target_rear_speed[i];
    float rear_required = (human ? LANE_CHANGE_REAR_GAP_HUMAN : LANE_CHANGE_REAR_GAP_AV) * (1.20f - 0.55f * ecs.risk_tolerance[i]);
    rear_required += tr_v * (human ? 1.0f : 0.65f);
    if (flow_balance_mandatory) {
        rear_required *= FLOW_BALANCE_MANDATORY_REAR_SCALE;
    }
    if (tr_gap < rear_required || (tr_v > ecs.speed[i] + 0.65f && tr_gap < rear_required * 1.55f)) {
        AVABM_METRIC_ADD(metrics, METRIC_LC_REJECT, 1.0f);
        return false;
    }
    float front_required = (human ? LANE_CHANGE_FRONT_GAP_HUMAN :
            LANE_CHANGE_FRONT_GAP_AV) * (1.15f - 0.45f * ecs.risk_tolerance[i]);
    front_required += ecs.speed[i] * (human ? 0.85f : 0.55f);
    if (flow_balance_mandatory) {
        front_required *= FLOW_BALANCE_MANDATORY_FRONT_SCALE;
    }
    if (tf_gap < front_required) {
        AVABM_METRIC_ADD(metrics, METRIC_LC_REJECT, 1.0f);
        return false;
    }
    if (mandatory) {
        AVABM_METRIC_ADD(metrics, METRIC_LC_ACCEPT, 1.0f);
        return true;
    }
    int lane = ecs.lane_id[i];
    float desired_curr = desired_speed_ecs(i, lane, ecs, road);
    float desired_next = desired_speed_ecs(i, target_lane, ecs, road);
    float a_old = estimate_follow_accel_ecs(ecs.speed[i], desired_curr, perception.front_gap[i], perception.front_speed[i],
            ecs.driver_type[i], ecs.min_gap[i], ecs.reaction_time[i], ecs.comfort_decel[i], ecs.aggressiveness[i],
            ecs.risk_tolerance[i]);
    float a_new = estimate_follow_accel_ecs(ecs.speed[i], desired_next, tf_gap, tf_v, ecs.driver_type[i], ecs.min_gap[i],
            ecs.reaction_time[i], ecs.comfort_decel[i], ecs.aggressiveness[i], ecs.risk_tolerance[i]);
    float ego_gain = a_new - a_old;
    float threshold = human ? MOBIL_THRESHOLD_HUMAN : MOBIL_THRESHOLD_AV;
    threshold *= 1.30f - 0.55f * ecs.aggressiveness[i];
    float utility = ego_gain - threshold;
    float hysteresis = human ? 0.15f : 0.10f;
    bool accepted = utility > MOBIL_MIN_ADVANTAGE + hysteresis;
    if (!accepted && opportunistic) {
        uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
        uint32_t h = hash_u32_ecs(((uint32_t)(i + 1) * 747796405u) ^ ((uint32_t)(target_lane + 13) * 2891336453u) ^
                ((uint32_t)(ecs.route_pos[i] + 17) * 277803737u) ^ (slot * 1103515245u));
        float p = CRUISE_RANDOM_LANE_CHANGE_PROB * (0.62f + 0.48f * clampf_cuda(ecs.aggressiveness[i], 0.0f,
                1.0f) + 0.35f * clampf_cuda(ecs.risk_tolerance[i], 0.0f, 1.0f));
        p = clampf_cuda(p, 0.045f, 0.36f);
        float tol = CRUISE_RANDOM_LANE_UTILITY_TOL;
        accepted = utility > -tol && hash01_ecs(h) < p;
    }
    if (accepted) AVABM_METRIC_ADD(metrics, METRIC_LC_ACCEPT, 1.0f);
    else AVABM_METRIC_ADD(metrics, METRIC_LC_REJECT, 1.0f);
    return accepted;
}
AVABM_DINLINE void find_front_in_connector_ecs(int self, ECSArrays ecs, RoadNetwork road, SpatialGrid grid, int max_entities,
        float search_radius, float& front_gap, float& front_speed) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;
    int from_ln = ecs.connector_from_lane[self];
    int to_ln = ecs.connector_to_lane[self];
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float self_progress = ecs.connector_s[self];
    float clen = fmaxf(ecs.connector_length[self], CONNECTOR_MIN_LEN);
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    float ds = 1.0e9f;
                    if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR && ecs.connector_from_lane[j] == from_ln &&
                            ecs.connector_to_lane[j] == to_ln) {
                        ds = ecs.connector_s[j] - self_progress;
                    } else if (ecs.vehicle_state[j] == VEH_ON_LANE && ecs.lane_id[j] == to_ln) {
                        float handoff_s = connector_exit_handoff_s(from_ln, to_ln, road);
                        float lane_ahead_s = ecs.s[j] - handoff_s;
                        if (lane_ahead_s >= 0.0f) {
                            ds = (clen - self_progress) + lane_ahead_s;
                        }
                    }
                    if (ds > 0.0f && ds < 1.0e8f) {
                        float gap = ds - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                        if (gap < front_gap) {
                            front_gap = gap;
                            front_speed = ecs.speed[j];
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}
AVABM_DINLINE float connector_cross_conflict_accel_limit_ecs(int self, ECSArrays ecs, RoadNetwork road, SpatialGrid grid,
        int max_entities, float dt, float* metrics) {
    int from_ln = ecs.connector_from_lane[self];
    int to_ln = ecs.connector_to_lane[self];
    if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road)) return 1000.0f;
    int node = road.lane_end_node[from_ln];
    float self_len = fmaxf(ecs.connector_length[self], CONNECTOR_MIN_LEN);
    float self_s = ecs.connector_s[self];
    float self_v = fmaxf(ecs.speed[self], 0.15f);
    if (self_s > fmaxf(1.5f, self_len * CONNECTOR_PROTECTED_PROGRESS_FRAC) ||
            (self_len - self_s) < self_len * CONNECTOR_PROTECTED_EXIT_FRAC) {
        return 1000.0f;
    }
    int base = world_cell_index(ecs.x[self], ecs.y[self], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return 1000.0f;
    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(CONNECTOR_ENTRY_CLEAR_RADIUS / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);
    float best_stop_s = 1.0e9f;
    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;
            int j = grid_head_ecs(grid, cy * grid.width + cx);
            int guard = 0;
            while (j >= 0 && guard < avabm_world_scan_limit_ecs(max_entities)) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                    int of = ecs.connector_from_lane[j];
                    int ot = ecs.connector_to_lane[j];
                    if (valid_lane_ecs(of, road) && valid_lane_ecs(ot, road) && road.lane_end_node[of] == node) {
                        if (connector_swept_paths_overlap_ecs(from_ln, to_ln, of, ot, road)) {
                            float other_len = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                            float other_s = ecs.connector_s[j];
                            float other_v = fmaxf(ecs.speed[j], 0.15f);
                            for (int a = 1; a <= CONNECTOR_CROSS_SAMPLE_COUNT; ++a) {
                                float ua = ((float)a) / ((float)CONNECTOR_CROSS_SAMPLE_COUNT + 1.0f);
                                float sa = ua * self_len;
                                if (sa < self_s - 0.25f) continue;
                                float ax, ay, ah;
                                connector_surface_path_xy_heading_ecs(from_ln, to_ln, ua, road, ax, ay, ah);
                                for (int b = 1; b <= CONNECTOR_CROSS_SAMPLE_COUNT; ++b) {
                                    float ub = ((float)b) / ((float)CONNECTOR_CROSS_SAMPLE_COUNT + 1.0f);
                                    float sb = ub * other_len;
                                    if (sb < other_s - 0.25f) continue;
                                    float bx, by, bh;
                                    connector_surface_path_xy_heading_ecs(of, ot, ub, road, bx, by, bh);
                                    float ddx = ax - bx;
                                    float ddy = ay - by;
                                    if (ddx * ddx +
                                            ddy * ddy > CONNECTOR_CROSS_CONFLICT_RADIUS * CONNECTOR_CROSS_CONFLICT_RADIUS) continue;
                                    float ta = (sa - self_s) / self_v;
                                    float tb = (sb - other_s) / other_v;
                                    if (ta < -0.10f || tb < -0.10f) continue;
                                    bool other_first = tb < ta - 0.18f || (fabsf(tb - ta) <= 0.18f && j < self);
                                    if (other_first && fabsf(ta - tb) < CONNECTOR_CROSS_TIME_WINDOW) {
                                        best_stop_s = fminf(best_stop_s, sa);
                                    }
                                }
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }
    if (best_stop_s >= 1.0e8f) return 1000.0f;
    float stop_dist = fmaxf(best_stop_s - self_s - CONNECTOR_CROSS_STOP_BUFFER, 0.55f);
    float req = -(ecs.speed[self] * ecs.speed[self]) / fmaxf(2.0f * stop_dist, 0.5f);
    req = clampf_cuda(req, -EMERGENCY_DECEL, -0.05f);
    if (metrics != nullptr) {
        AVABM_METRIC_ADD(metrics, METRIC_CONNECTOR_CROSS_YIELD, 1.0f);
        AVABM_METRIC_ADD(metrics, METRIC_ANTI_COLLISION_BRAKE, 1.0f);
    }
    return req;
}
AVABM_DINLINE void vehicle_obb_axes(float h, float& fx, float& fy, float& sx, float& sy) {
    fx = cosf(h);
    fy = sinf(h);
    sx = -fy;
    sy = fx;
}
AVABM_DINLINE bool obb_overlap_sat(float ax, float ay, float ah, float al, float aw, float bx, float by, float bh, float bl,
        float bw, float inflate) {
    al = fmaxf(al + inflate, 0.1f);
    aw = fmaxf(aw + inflate, 0.1f);
    bl = fmaxf(bl + inflate, 0.1f);
    bw = fmaxf(bw + inflate, 0.1f);
    float afx, afy, asx, asy;
    float bfx, bfy, bsx, bsy;
    vehicle_obb_axes(ah, afx, afy, asx, asy);
    vehicle_obb_axes(bh, bfx, bfy, bsx, bsy);
    float dx = bx - ax;
    float dy = by - ay;
    float axes_x[4] = {
        afx, asx, bfx, bsx
    };
    float axes_y[4] = {
        afy, asy, bfy, bsy
    };
    for (int k = 0; k < 4; ++k) {
        float ux = axes_x[k];
        float uy = axes_y[k];
        float dist = fabsf(dx * ux + dy * uy);
        float ra = 0.5f * al * fabsf(ux * afx + uy * afy) + 0.5f * aw * fabsf(ux * asx + uy * asy);
        float rb = 0.5f * bl * fabsf(ux * bfx + uy * bfy) + 0.5f * bw * fabsf(ux * bsx + uy * bsy);
        if (dist > ra + rb) return false;
    }
    return true;
}
AVABM_DINLINE bool swept_overlap_ecs(float ax, float ay, float ah, float av, float al, float aw, float bx, float by, float bh,
        float bv, float bl, float bw, float dt, float horizon, float inflate) {
    float total_t = fmaxf(dt, horizon);
    int slices = 5;
    float afx = cosf(ah);
    float afy = sinf(ah);
    float bfx = cosf(bh);
    float bfy = sinf(bh);
    for (int k = 0; k <= slices; ++k) {
        float u = (float)k / (float)slices;
        float t = total_t * u;
        float apx = ax + afx * av * t;
        float apy = ay + afy * av * t;
        float bpx = bx + bfx * bv * t;
        float bpy = by + bfy * bv * t;
        if (obb_overlap_sat(apx, apy, ah, al, aw, bpx, bpy, bh, bl, bw, inflate)) {
            return true;
        }
    }
    return false;
}
AVABM_DINLINE void sync_vehicle_to_path_ecs(int id, ECSArrays ecs, RoadNetwork road) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        int from_ln = ecs.connector_from_lane[id];
        int to_ln = ecs.connector_to_lane[id];
        if (valid_lane_ecs(from_ln, road) && valid_lane_ecs(to_ln, road)) {
            float clen = fmaxf(ecs.connector_length[id], CONNECTOR_MIN_LEN);
            ecs.connector_s[id] = clampf_cuda(ecs.connector_s[id], 0.0f, clen);
            float px, py, ph;
            connector_xy_heading_from_s(from_ln, to_ln, ecs.connector_s[id], clen, road, px, py, ph);
            ecs.x[id] = px;
            ecs.y[id] = py;
            ecs.heading[id] = ph;
        }
    } else {
        int ln = ecs.lane_id[id];
        if (valid_lane_ecs(ln, road)) {
            ecs.s[id] = clampf_cuda(ecs.s[id], 0.0f, fmaxf(0.0f, road.lane_length[ln] - 0.05f));
            if (ecs.lane_change_active[id] != 0) {
                int from_ln = ecs.lane_change_from_lane[id];
                int to_ln = ecs.lane_change_to_lane[id];
                if (valid_lane_ecs(from_ln, road) && valid_lane_ecs(to_ln, road) &&
                        same_approach_same_direction_lanes_ecs(from_ln, to_ln, road)) {
                    float dur = fmaxf(ecs.lane_change_duration[id], 0.1f);
                    float t = clampf_cuda(ecs.lane_change_t[id] / dur, 0.0f, 1.0f);
                    float u = smoothstep01(t);
                    float s_from = clampf_cuda(ecs.s[id], 0.0f, fmaxf(0.0f, road.lane_length[from_ln] - 0.05f));
                    float s_to = clampf_cuda(ecs.s[id], 0.0f, fmaxf(0.0f, road.lane_length[to_ln] - 0.05f));
                    float ax, ay, ah;
                    float bx, by, bh;
                    lane_xy_heading_from_s(from_ln, s_from, road, ax, ay, ah);
                    lane_xy_heading_from_s(to_ln, s_to, road, bx, by, bh);
                    ecs.x[id] = ax + (bx - ax) * u;
                    ecs.y[id] = ay + (by - ay) * u;
                    ecs.heading[id] = wrap_pi(ah + wrap_pi(bh - ah) * u);
                    return;
                }
            }
            float px, py, ph;
            lane_xy_heading_from_s(ln, ecs.s[id], road, px, py, ph);
            ecs.x[id] = px;
            ecs.y[id] = py;
            ecs.heading[id] = ph;
        }
    }
}
AVABM_DINLINE void move_vehicle_back_on_path_ecs(int id, float back, ECSArrays ecs, RoadNetwork road) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return;
    back = clampf_cuda(back, 0.0f, CONTACT_RESOLVE_MAX_PUSH);
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        ecs.connector_s[id] = fmaxf(0.0f, ecs.connector_s[id] - back);
    } else {
        ecs.s[id] = fmaxf(0.0f, ecs.s[id] - back);
    }
    sync_vehicle_to_path_ecs(id, ecs, road);
    ecs.speed[id] = fminf(fmaxf(ecs.speed[id], 0.0f), CONTACT_REPAIR_LOSER_SPEED_CAP);
    ecs.accel[id] = CONTACT_REPAIR_HOLD_ACCEL;
}
AVABM_DINLINE bool repair_pair_same_lane_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
    if (ecs.vehicle_state[a] != VEH_ON_LANE || ecs.vehicle_state[b] != VEH_ON_LANE) return false;
    int la = ecs.lane_id[a];
    int lb = ecs.lane_id[b];
    if (!valid_lane_ecs(la, road) || la != lb) return false;
    int front = ecs.s[a] >= ecs.s[b] ? a : b;
    int rear = front == a ? b : a;
    float desired_center_ds = 0.5f * ecs.length[front] + 0.5f * ecs.length[rear] + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_rear_s = ecs.s[front] - desired_center_ds;
    if (ecs.s[rear] > target_rear_s) {
        ecs.s[rear] = fmaxf(0.0f, target_rear_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(rear, ecs, road);
        ecs.speed[rear] = fminf(ecs.speed[rear], fmaxf(0.0f, ecs.speed[front] - 0.3f));
        ecs.accel[rear] = CONTACT_REPAIR_HOLD_ACCEL;
        float actual_center_ds = ecs.s[front] - ecs.s[rear];
        float residual = desired_center_ds - actual_center_ds;
        if (residual > 0.05f) {
            int flane = ecs.lane_id[front];
            if (valid_lane_ecs(flane, road)) {
                float before_front_s = ecs.s[front];
                ecs.s[front] = fminf(fmaxf(0.0f, road.lane_length[flane] - 0.05f), ecs.s[front] + fminf(CONTACT_RESOLVE_MAX_PUSH,
                        residual + CONTACT_ROUTE_REPAIR_EXTRA));
                if (ecs.s[front] > before_front_s + 0.001f) {
                    sync_vehicle_to_path_ecs(front, ecs, road);
                    ecs.speed[front] = fmaxf(ecs.speed[front], OVERLAP_CONTACT_WINNER_MIN_SPEED);
                    ecs.accel[front] = fmaxf(ecs.accel[front], 0.0f);
                }
            }
        }
    }
    return true;
}
AVABM_DINLINE bool repair_pair_same_connector_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
    if (ecs.vehicle_state[a] != VEH_IN_CONNECTOR || ecs.vehicle_state[b] != VEH_IN_CONNECTOR) return false;
    if (ecs.connector_from_lane[a] != ecs.connector_from_lane[b] ||
            ecs.connector_to_lane[a] != ecs.connector_to_lane[b]) return false;
    int front = ecs.connector_s[a] >= ecs.connector_s[b] ? a : b;
    int rear = front == a ? b : a;
    float desired_center_ds = 0.5f * ecs.length[front] + 0.5f * ecs.length[rear] + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_rear_s = ecs.connector_s[front] - desired_center_ds;
    if (ecs.connector_s[rear] > target_rear_s) {
        ecs.connector_s[rear] = fmaxf(0.0f, target_rear_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(rear, ecs, road);
        ecs.speed[rear] = fminf(ecs.speed[rear], fmaxf(0.0f, ecs.speed[front] - 0.3f));
        ecs.accel[rear] = CONTACT_REPAIR_HOLD_ACCEL;
        float actual_center_ds = ecs.connector_s[front] - ecs.connector_s[rear];
        float residual = desired_center_ds - actual_center_ds;
        if (residual > 0.05f) {
            float clen = fmaxf(ecs.connector_length[front], CONNECTOR_MIN_LEN);
            float before_front_s = ecs.connector_s[front];
            ecs.connector_s[front] = fminf(clen, ecs.connector_s[front] + fminf(CONTACT_RESOLVE_MAX_PUSH,
                    residual + CONTACT_ROUTE_REPAIR_EXTRA));
            if (ecs.connector_s[front] > before_front_s + 0.001f) {
                sync_vehicle_to_path_ecs(front, ecs, road);
                ecs.speed[front] = fmaxf(ecs.speed[front], OVERLAP_CONTACT_WINNER_MIN_SPEED);
                ecs.accel[front] = fmaxf(ecs.accel[front], 0.0f);
            }
        }
    }
    return true;
}
AVABM_DINLINE bool repair_connector_to_lane_pair_ecs(int conn, int lane_vehicle, ECSArrays ecs, RoadNetwork road) {
    if (ecs.vehicle_state[conn] != VEH_IN_CONNECTOR || ecs.vehicle_state[lane_vehicle] != VEH_ON_LANE) return false;
    int from_ln = ecs.connector_from_lane[conn];
    int to_ln = ecs.connector_to_lane[conn];
    int lane = ecs.lane_id[lane_vehicle];
    if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road) || lane != to_ln) return false;
    float handoff = connector_exit_handoff_s(from_ln, to_ln, road);
    if (ecs.s[lane_vehicle] < handoff) return false;
    float clen = fmaxf(ecs.connector_length[conn], CONNECTOR_MIN_LEN);
    float desired_center_ds = 0.5f * ecs.length[conn] + 0.5f * ecs.length[lane_vehicle] + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_conn_s = clen + (ecs.s[lane_vehicle] - handoff) - desired_center_ds;
    if (ecs.connector_s[conn] > target_conn_s) {
        ecs.connector_s[conn] = clampf_cuda(target_conn_s - CONTACT_ROUTE_REPAIR_EXTRA, 0.0f, clen);
        sync_vehicle_to_path_ecs(conn, ecs, road);
        ecs.speed[conn] = fminf(ecs.speed[conn], fmaxf(0.0f, ecs.speed[lane_vehicle] - 0.3f));
        ecs.accel[conn] = CONTACT_REPAIR_HOLD_ACCEL;
        float conn_progress = ecs.connector_s[conn];
        float lane_progress = clen + (ecs.s[lane_vehicle] - handoff);
        float actual_center_ds = lane_progress - conn_progress;
        float residual = desired_center_ds - actual_center_ds;
        if (residual > 0.05f && valid_lane_ecs(lane, road)) {
            float before_lane_s = ecs.s[lane_vehicle];
            ecs.s[lane_vehicle] = fminf(fmaxf(0.0f, road.lane_length[lane] - 0.05f),
                    ecs.s[lane_vehicle] + fminf(CONTACT_RESOLVE_MAX_PUSH, residual + CONTACT_ROUTE_REPAIR_EXTRA));
            if (ecs.s[lane_vehicle] > before_lane_s + 0.001f) {
                sync_vehicle_to_path_ecs(lane_vehicle, ecs, road);
                ecs.speed[lane_vehicle] = fmaxf(ecs.speed[lane_vehicle], OVERLAP_CONTACT_WINNER_MIN_SPEED);
                ecs.accel[lane_vehicle] = fmaxf(ecs.accel[lane_vehicle], 0.0f);
            }
        }
    }
    return true;
}
AVABM_DINLINE bool repair_lane_to_connector_pair_ecs(int lane_vehicle, int conn, ECSArrays ecs, RoadNetwork road) {
    if (ecs.vehicle_state[lane_vehicle] != VEH_ON_LANE || ecs.vehicle_state[conn] != VEH_IN_CONNECTOR) return false;
    int lane = ecs.lane_id[lane_vehicle];
    int from_ln = ecs.connector_from_lane[conn];
    int to_ln = ecs.connector_to_lane[conn];
    if (!valid_lane_ecs(lane, road) || lane != from_ln || !valid_lane_ecs(to_ln, road)) return false;
    float start_s = fmaxf(0.0f, road.lane_length[from_ln] - connector_entry_backoff_ecs(from_ln, to_ln, road));
    float desired_center_ds = 0.5f * ecs.length[lane_vehicle] + 0.5f * ecs.length[conn] + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_lane_s = start_s + ecs.connector_s[conn] - desired_center_ds;
    if (ecs.s[lane_vehicle] > target_lane_s) {
        ecs.s[lane_vehicle] = fmaxf(0.0f, target_lane_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(lane_vehicle, ecs, road);
        ecs.speed[lane_vehicle] = fminf(ecs.speed[lane_vehicle], fmaxf(0.0f, ecs.speed[conn] - 0.3f));
        ecs.accel[lane_vehicle] = CONTACT_REPAIR_HOLD_ACCEL;
        float lane_progress = ecs.s[lane_vehicle];
        float conn_progress = start_s + ecs.connector_s[conn];
        float actual_center_ds = conn_progress - lane_progress;
        float residual = desired_center_ds - actual_center_ds;
        if (residual > 0.05f) {
            float clen = fmaxf(ecs.connector_length[conn], CONNECTOR_MIN_LEN);
            float before_conn_s = ecs.connector_s[conn];
            ecs.connector_s[conn] = fminf(clen, ecs.connector_s[conn] + fminf(CONTACT_RESOLVE_MAX_PUSH,
                    residual + CONTACT_ROUTE_REPAIR_EXTRA));
            if (ecs.connector_s[conn] > before_conn_s + 0.001f) {
                sync_vehicle_to_path_ecs(conn, ecs, road);
                ecs.speed[conn] = fmaxf(ecs.speed[conn], OVERLAP_CONTACT_WINNER_MIN_SPEED);
                ecs.accel[conn] = fmaxf(ecs.accel[conn], 0.0f);
            }
        }
    }
    return true;
}
AVABM_DINLINE bool repair_route_overlap_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
    if (repair_pair_same_lane_ecs(a, b, ecs, road)) return true;
    if (repair_pair_same_connector_ecs(a, b, ecs, road)) return true;
    if (repair_connector_to_lane_pair_ecs(a, b, ecs, road)) return true;
    if (repair_connector_to_lane_pair_ecs(b, a, ecs, road)) return true;
    if (repair_lane_to_connector_pair_ecs(a, b, ecs, road)) return true;
    if (repair_lane_to_connector_pair_ecs(b, a, ecs, road)) return true;
    return false;
}
AVABM_DINLINE int contact_priority_score_ecs(int id, ECSArrays ecs, RoadNetwork road) {
    int score = 0;
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        score += 240;
        float clen = fmaxf(ecs.connector_length[id], CONNECTOR_MIN_LEN);
        score += clampi_cuda((int)floorf(160.0f * ecs.connector_s[id] / fmaxf(clen, 0.1f)), 0, 160);
    } else {
        int lane = ecs.lane_id[id];
        int next_lane = route_next_lane_for_vehicle_ecs(id, ecs, road);
        if (valid_lane_ecs(lane, road) && valid_lane_ecs(next_lane, road)) {
            float dist_to_end = fmaxf(0.0f, road.lane_length[lane] - ecs.s[id]);
            if (inside_intersection_box_ecs(dist_to_end, lane, next_lane, road)) score += 190;
            if (dist_to_end < PRIORITY_GATE_NEAR_LINE_DIST + 3.0f) score += 40;
        }
    }
    float wait = ecs.connector_length[id];
    if (!isfinite(wait) || wait < 0.0f || ecs.vehicle_state[id] != VEH_ON_LANE) wait = 0.0f;
    score += clampi_cuda((int)floorf(wait * 20.0f), 0, 240);
    score += clampi_cuda((int)floorf(fmaxf(0.0f, ecs.speed[id]) * 4.0f), 0, 90);
    score += clampi_cuda((int)floorf(ecs.aggressiveness[id] * 22.0f), 0, 22);
    if (indicator_active_ecs(id, ecs)) score += 16;
    return score;
}
AVABM_DINLINE void move_vehicle_forward_on_path_ecs(int id, float forward, ECSArrays ecs, RoadNetwork road) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return;
    forward = clampf_cuda(forward, 0.0f, CONTACT_RESOLVE_MAX_PUSH);
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        float clen = fmaxf(ecs.connector_length[id], CONNECTOR_MIN_LEN);
        ecs.connector_s[id] = fminf(clen, ecs.connector_s[id] + forward);
    } else {
        int lane = ecs.lane_id[id];
        if (valid_lane_ecs(lane, road)) {
            ecs.s[id] = fminf(fmaxf(0.0f, road.lane_length[lane] - 0.05f), ecs.s[id] + forward);
        }
    }
    sync_vehicle_to_path_ecs(id, ecs, road);
}
AVABM_DINLINE void overlap_release_boost_vehicle_ecs(int id, ECSArrays ecs, RoadNetwork road, float dt) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return;
    bool human = ecs.driver_type[id] == HUMAN;
    float release_v = human ? COMPLETE_OVERLAP_RELEASE_SPEED_HUMAN : COMPLETE_OVERLAP_RELEASE_SPEED_AV;
    if (ecs.vehicle_state[id] == VEH_ON_LANE && valid_lane_ecs(ecs.lane_id[id], road)) {
        release_v = fminf(release_v, fmaxf(1.0f, desired_speed_ecs(id, ecs.lane_id[id], ecs, road)));
    }
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float release_a = (release_v - ecs.speed[id]) / fmaxf(dt, 0.01f);
    release_a = clampf_cuda(release_a, 0.0f, max_accel * COMPLETE_OVERLAP_RELEASE_ACCEL_SCALE);
    move_vehicle_forward_on_path_ecs(id, COMPLETE_OVERLAP_CONTACT_FORWARD_NUDGE, ecs, road);
    ecs.speed[id] = fmaxf(ecs.speed[id], fminf(release_v, ecs.speed[id] + release_a * fmaxf(dt, 0.01f)));
    ecs.accel[id] = fmaxf(ecs.accel[id], release_a);
}
AVABM_DINLINE float lane_change_progress01_ecs(int id, ECSArrays ecs) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE || ecs.lane_change_active[id] == 0) return 0.0f;
    float dur = fmaxf(ecs.lane_change_duration[id], 0.1f);
    return clampf_cuda(ecs.lane_change_t[id] / dur, 0.0f, 1.0f);
}
AVABM_DINLINE bool force_lane_change_endpoint_for_overlap_ecs(int id, ECSArrays ecs, RoadNetwork road) {
#if LANE_CHANGE_OVERLAP_SNAP_ENABLED
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return false;
    if (ecs.vehicle_state[id] != VEH_ON_LANE || ecs.lane_change_active[id] == 0) return false;
    int from_ln = ecs.lane_change_from_lane[id];
    int to_ln = ecs.lane_change_to_lane[id];
    if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road)) {
        ecs.lane_change_active[id] = 0;
        return false;
    }
    float t = lane_change_progress01_ecs(id, ecs);
    int final_lane = (t >= LC_OVERLAP_COMMIT_T) ? to_ln : from_ln;
    float L = fmaxf(road.lane_length[final_lane], 0.1f);
    ecs.s[id] = clampf_cuda(ecs.s[id], 0.0f, fmaxf(0.0f, L - 0.05f));
    ecs.lane_id[id] = final_lane;
    ecs.lane_change_active[id] = 0;
    ecs.lane_change_from_lane[id] = final_lane;
    ecs.lane_change_to_lane[id] = final_lane;
    ecs.lane_change_t[id] = 0.0f;
    ecs.lc_cooldown[id] = fmaxf(ecs.lc_cooldown[id], LC_ACTIVE_MIDLINE_ABORT_COOLDOWN * 0.65f);
    int repaired = repair_route_pos_unless_missed_exit_tail_ecs(final_lane, ecs.route_id[id], ecs.route_pos[id], road);
    if (repaired >= 0) ecs.route_pos[id] = repaired;
    if (ecs.turn_signal != nullptr) {
        ecs.turn_signal[id] = INDICATOR_NONE;
        if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[id] = 0.0f;
    }
    sync_vehicle_to_path_ecs(id, ecs, road);
    ecs.speed[id] = fmaxf(ecs.speed[id], LC_OVERLAP_MIN_SPEED);
    ecs.accel[id] = fmaxf(ecs.accel[id], 0.0f);
    return true;
#else
    (void)id;
    (void)ecs;
    (void)road;
    return false;
#endif
}
AVABM_DINLINE bool repair_lane_change_overlap_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
#if LANE_CHANGE_OVERLAP_SNAP_ENABLED
    bool changed = false;
    changed = force_lane_change_endpoint_for_overlap_ecs(a, ecs, road) || changed;
    changed = force_lane_change_endpoint_for_overlap_ecs(b, ecs, road) || changed;
    if (changed) {
        repair_route_overlap_ecs(a, b, ecs, road);
    }
    return changed;
#else
    (void)a;
    (void)b;
    (void)ecs;
    (void)road;
    return false;
#endif
}
AVABM_DINLINE int hard_overlap_priority_score_ecs(int id, ECSArrays ecs, RoadNetwork road) {
    int score = contact_priority_score_ecs(id, ecs, road);
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        float clen = fmaxf(ecs.connector_length[id], CONNECTOR_MIN_LEN);
        score += 520 + clampi_cuda((int)floorf(260.0f * ecs.connector_s[id] / fmaxf(clen, 0.1f)), 0, 260);
    } else if (ecs.vehicle_state[id] == VEH_ON_LANE && valid_lane_ecs(ecs.lane_id[id], road)) {
        int lane = ecs.lane_id[id];
        float dist_to_end = fmaxf(0.0f, road.lane_length[lane] - ecs.s[id]);
        if (dist_to_end <= GRIDLOCK_NODE_DRAIN_NEAR_DIST + 5.0f) {
            score += 260 + clampi_cuda((int)floorf((GRIDLOCK_NODE_DRAIN_NEAR_DIST + 5.0f - dist_to_end) * 10.0f), 0, 260);
        }
    }
    if (ecs.lane_change_active[id] != 0) {
        float t = lane_change_progress01_ecs(id, ecs);
        score += (t >= LC_OVERLAP_COMMIT_T) ? 55 : -95;
    }
    return score;
}
AVABM_DINLINE bool hard_separate_overlap_pair_ecs(int a, int b, ECSArrays ecs, RoadNetwork road, float current_time) {
    if (a < 0 || b < 0 || ecs.alive[a] != ENTITY_ALIVE || ecs.alive[b] != ENTITY_ALIVE) return false;
    bool changed = repair_lane_change_overlap_ecs(a, b, ecs, road);
    if (changed) {
        bool cleared = !obb_overlap_sat(ecs.x[a], ecs.y[a], ecs.heading[a], ecs.length[a], ecs.width[a], ecs.x[b], ecs.y[b],
                ecs.heading[b], ecs.length[b], ecs.width[b], CONTACT_HARD_VERIFY_INFLATE);
        if (cleared) return true;
    }
    int sa = hard_overlap_priority_score_ecs(a, ecs, road);
    int sb = hard_overlap_priority_score_ecs(b, ecs, road);
    bool a_wins = true;
    if (sa > sb) a_wins = true;
    else if (sb > sa) a_wins = false;
    else a_wins = timed_pair_random_self_wins_ecs(a, b, current_time, COMPLETE_OVERLAP_RELEASE_PERIOD);
    int winner = a_wins ? a : b;
    int loser = a_wins ? b : a;
    float loser_back = NODE_OVERLAP_HARD_BACKOFF + fmaxf(ecs.speed[loser], 0.0f) * CONTACT_CROSS_BACKOFF_SPEED_TIME;
    loser_back = fminf(CONTACT_RESOLVE_MAX_PUSH, fmaxf(loser_back, CONTACT_RESOLVE_BACKOFF));
    move_vehicle_back_on_path_ecs(loser, loser_back, ecs, road);
    float forward = fminf(CONTACT_RESOLVE_MAX_PUSH, NODE_OVERLAP_HARD_FORWARD);
    move_vehicle_forward_on_path_ecs(winner, forward, ecs, road);
    bool human = ecs.driver_type[winner] == HUMAN;
    float min_v = human ? NODE_OVERLAP_HARD_MIN_SPEED_HUMAN : NODE_OVERLAP_HARD_MIN_SPEED_AV;
    ecs.speed[winner] = fmaxf(ecs.speed[winner], min_v);
    ecs.accel[winner] = fmaxf(ecs.accel[winner], 0.0f);
    ecs.speed[loser] = fminf(fmaxf(ecs.speed[loser], 0.0f), CONTACT_REPAIR_LOSER_SPEED_CAP);
    ecs.accel[loser] = CONTACT_REPAIR_HOLD_ACCEL;
    return true;
}
AVABM_DINLINE bool local_avoid_same_stream_skip_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
    if (ecs.vehicle_state[a] != VEH_ON_LANE || ecs.vehicle_state[b] != VEH_ON_LANE) return false;
    if (ecs.lane_change_active[a] != 0 || ecs.lane_change_active[b] != 0) return false;
    int la = ecs.lane_id[a];
    int lb = ecs.lane_id[b];
    if (!valid_lane_ecs(la, road) || !valid_lane_ecs(lb, road)) return false;
    if (la != lb) return false;
    float dh = fabsf(wrap_pi(ecs.heading[a] - ecs.heading[b]));
    return dh < 0.36f;
}
AVABM_DINLINE int local_avoid_priority_score_ecs(int id, ECSArrays ecs, RoadNetwork road, PerceptionSoA perception) {
    int score = 0;
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        score += LOCAL_AVOID_CONNECTOR_BONUS;
    }
    if (ecs.vehicle_state[id] == VEH_ON_LANE) {
        int lane = ecs.lane_id[id];
        int next_lane = route_next_lane_for_vehicle_ecs(id, ecs, road);
        if (valid_lane_ecs(lane, road)) {
            float dist_to_end = fmaxf(0.0f, road.lane_length[lane] - ecs.s[id]);
            if (valid_lane_ecs(next_lane, road) && inside_intersection_box_ecs(dist_to_end, lane, next_lane, road)) {
                score += LOCAL_AVOID_INSIDE_BOX_BONUS;
            }
            if (dist_to_end < PRIORITY_GATE_NEAR_LINE_DIST + 3.0f) score += 24;
        }
    }
    if (priority_front_clear_ecs(id, perception, ecs)) score += LOCAL_AVOID_FRONT_CLEAR_BONUS;
    else score -= 90;
    float wait_time = ecs.connector_length[id];
    if (!isfinite(wait_time) || wait_time < 0.0f) wait_time = 0.0f;
    score += clampi_cuda((int)floorf(wait_time * 22.0f), 0, 220);
    score += clampi_cuda((int)floorf(fmaxf(0.0f, ecs.speed[id]) * 3.0f), 0, 80);
    score += clampi_cuda((int)floorf(ecs.aggressiveness[id] * 20.0f), 0, 20);
    if (indicator_active_ecs(id, ecs)) score += 12;
    return score;
}
AVABM_DINLINE bool local_avoid_pair_relevant_ecs(int a, int b, ECSArrays ecs, RoadNetwork road) {
    if (obb_overlap_sat(ecs.x[a], ecs.y[a], ecs.heading[a], ecs.length[a], ecs.width[a], ecs.x[b], ecs.y[b], ecs.heading[b],
            ecs.length[b], ecs.width[b], LOCAL_AVOID_IMMEDIATE_OVERLAP_INFLATE)) {
        return true;
    }
    bool ac = ecs.vehicle_state[a] == VEH_IN_CONNECTOR;
    bool bc = ecs.vehicle_state[b] == VEH_IN_CONNECTOR;
    if (ac || bc) {
        if (ac && bc) {
            int af = ecs.connector_from_lane[a];
            int at = ecs.connector_to_lane[a];
            int bf = ecs.connector_from_lane[b];
            int bt = ecs.connector_to_lane[b];
            if (!valid_lane_ecs(af, road) || !valid_lane_ecs(at, road) || !valid_lane_ecs(bf, road) ||
                    !valid_lane_ecs(bt, road)) return false;
            if (af == bf && at == bt) return true;
            return road.lane_end_node[af] == road.lane_end_node[bf] && connector_swept_paths_overlap_ecs(af, at, bf, bt, road);
        }
        int c = ac ? a : b;
        int o = ac ? b : a;
        int cf = ecs.connector_from_lane[c];
        int ct = ecs.connector_to_lane[c];
        int ol = ecs.lane_id[o];
        int on = route_next_lane_for_vehicle_ecs(o, ecs, road);
        if (!valid_lane_ecs(cf, road) || !valid_lane_ecs(ct, road) || !valid_lane_ecs(ol, road)) return false;
        if (ol == cf || ol == ct || on == cf || on == ct) return true;
        if (valid_lane_ecs(on, road) && road.lane_end_node[ol] == road.lane_end_node[cf]) {
            if (ac) {
                return intersection_conflict_relevant_vehicles_ecs(c, cf, ct, o, ol, on, false, ecs, road);
            }
            return intersection_conflict_relevant_vehicles_ecs(o, ol, on, c, cf, ct, true, ecs, road);
        }
        return false;
    }
    int la = ecs.lane_id[a];
    int lb = ecs.lane_id[b];
    if (!valid_lane_ecs(la, road) || !valid_lane_ecs(lb, road)) return false;
    if (la == lb) return true;
    if (ecs.lane_change_active[a] != 0) {
        int af = ecs.lane_change_from_lane[a];
        int at = ecs.lane_change_to_lane[a];
        if (lb == af || lb == at || lanes_share_link_geometry_ecs(lb, at, road) ||
                lanes_share_link_geometry_ecs(lb, af, road)) return true;
    }
    if (ecs.lane_change_active[b] != 0) {
        int bf = ecs.lane_change_from_lane[b];
        int bt = ecs.lane_change_to_lane[b];
        if (la == bf || la == bt || lanes_share_link_geometry_ecs(la, bt, road) ||
                lanes_share_link_geometry_ecs(la, bf, road)) return true;
    }
    if (lanes_share_link_geometry_ecs(la, lb, road)) {
        return indicator_targets_lane_ecs(a, lb, ecs, road) || indicator_targets_lane_ecs(b, la, ecs, road);
    }
    int na = route_next_lane_for_vehicle_ecs(a, ecs, road);
    int nb = route_next_lane_for_vehicle_ecs(b, ecs, road);
    if (valid_lane_ecs(na, road) && valid_lane_ecs(nb, road) && lane_count_merge_pair_conflict_ecs(la, na, lb, nb, road)) {
        float da = fmaxf(0.0f, road.lane_length[la] - ecs.s[a]);
        float db = fmaxf(0.0f, road.lane_length[lb] - ecs.s[b]);
        if (da <= LANE_COUNT_CHANGE_PREP_MAX_DIST || db <= LANE_COUNT_CHANGE_PREP_MAX_DIST) return true;
    }
    if (valid_lane_ecs(na, road) && valid_lane_ecs(nb, road) && road.lane_end_node[la] == road.lane_end_node[lb]) {
        float da = fmaxf(0.0f, road.lane_length[la] - ecs.s[a]);
        float db = fmaxf(0.0f, road.lane_length[lb] - ecs.s[b]);
        if (da <= PRIORITY_GATE_PATH_SCAN_RANGE && db <= PRIORITY_GATE_PATH_SCAN_RANGE) {
            return intersection_conflict_relevant_vehicles_ecs(a, la, na, b, lb, nb, false, ecs, road);
        }
    }
    return false;
}
AVABM_DINLINE void write_render_quad(RenderVertex* out, int base, float cx, float cy, float h, float L, float W, float r, float g,
        float b, float a) {
    float c = cosf(h);
    float ss = sinf(h);
    float dx = c * L * 0.5f;
    float dy = ss * L * 0.5f;
    float nx = -ss * W * 0.5f;
    float ny = c * W * 0.5f;
    float x0 = cx - dx - nx;
    float y0 = cy - dy - ny;
    float x1 = cx + dx - nx;
    float y1 = cy + dy - ny;
    float x2 = cx + dx + nx;
    float y2 = cy + dy + ny;
    float x3 = cx - dx + nx;
    float y3 = cy - dy + ny;
    RenderVertex v0 = {
        x0, y0, r, g, b, a, 1
    };
    RenderVertex v1 = {
        x1, y1, r, g, b, a, 1
    };
    RenderVertex v2 = {
        x2, y2, r, g, b, a, 1
    };
    RenderVertex v3 = {
        x3, y3, r, g, b, a, 1
    };
    out[base + 0] = v0;
    out[base + 1] = v1;
    out[base + 2] = v2;
    out[base + 3] = v0;
    out[base + 4] = v2;
    out[base + 5] = v3;
}
AVABM_DINLINE void write_render_textured_quad(RenderVertex* out, int base, float cx, float cy, float h, float L, float W,
        int driver_type, int turn_signal, float turn_signal_time) {
    float c = cosf(h);
    float ss = sinf(h);
    float dx = c * L * 0.5f;
    float dy = ss * L * 0.5f;
    float nx = -ss * W * 0.5f;
    float ny = c * W * 0.5f;
    float x0 = cx - dx - nx;
    float y0 = cy - dy - ny;
    float x1 = cx + dx - nx;
    float y1 = cy + dy - ny;
    float x2 = cx + dx + nx;
    float y2 = cy + dy + ny;
    float x3 = cx - dx + nx;
    float y3 = cy - dy + ny;
    float driver_flag = driver_type == AV ? 1.0f : 0.0f;
    float blink_flag = 0.0f;
    if (turn_signal == INDICATOR_LEFT || turn_signal == INDICATOR_RIGHT || turn_signal == INDICATOR_HAZARD) {
        float phase = fmodf(fmaxf(turn_signal_time, 0.0f) * 2.25f, 1.0f);
        blink_flag = phase < 0.55f ? 1.0f : 0.0f;
    }
    RenderVertex v0 = {
        x0, y0, 0.0f, 0.0f, driver_flag, blink_flag, 1.0f
    };
    RenderVertex v1 = {
        x1, y1, 1.0f, 0.0f, driver_flag, blink_flag, 1.0f
    };
    RenderVertex v2 = {
        x2, y2, 1.0f, 1.0f, driver_flag, blink_flag, 1.0f
    };
    RenderVertex v3 = {
        x3, y3, 0.0f, 1.0f, driver_flag, blink_flag, 1.0f
    };
    out[base + 0] = v0;
    out[base + 1] = v1;
    out[base + 2] = v2;
    out[base + 3] = v0;
    out[base + 4] = v2;
    out[base + 5] = v3;
}
