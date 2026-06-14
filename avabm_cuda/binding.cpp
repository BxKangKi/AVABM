/*
EN: PyTorch/CUDA binding for the AVABM surface-aware traffic simulator.
KO: 도로 면 기반 AVABM 교통 시뮬레이터용 PyTorch/CUDA 바인딩입니다.
    이번 버전은 ABI를 priority/deadlock/lane-side 버전과 유지하며, 렌더링 차선 표시는 Python 정적 VBO에서 처리합니다.
*/
// binding.cpp
#ifdef _MSC_VER
// EN/KO: Suppress deprecation/codepage warnings from external CUDA/PyTorch headers
// only, so the build log highlights real AVABM compile errors.
#pragma warning(push)
#pragma warning(disable: 4996)
#pragma warning(disable: 4819)
#endif
// KO 문법 이유: torch/extension.h는 torch/all.h까지 끌어와 binding.cpp 컴파일이 매우 무거워집니다.
// KO 논리 이유: 이 파일은 Tensor 타입, torch::empty, TORCH_CHECK, pybind11 바인딩만 필요하므로
//    기본값은 가벼운 header 조합을 사용합니다. 호환 문제가 생기면 config.txt에서
//    CUDA_USE_FULL_TORCH_EXTENSION_HEADER=1로 바꿔 기존 전체 header로 되돌릴 수 있습니다.
#ifdef AVABM_USE_FULL_TORCH_EXTENSION_HEADER
#include <torch/extension.h>
#elif defined(__has_include)
#if __has_include(<torch/types.h>) && __has_include(<torch/csrc/utils/pybind.h>)
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#else
#include <torch/extension.h>
#endif
#else
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#endif
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cmath>
#include <cstdint>
#include <limits>

// ============================================================
// Binding constants
// ============================================================

constexpr int MAX_CONFLICT_LANES_BINDING = 8;
constexpr int RES_HORIZON_SLOTS_BINDING = 16;
constexpr int64_t METRICS_SIZE_MIN = 112;

// ============================================================
// ECS POD structs - must match main.cu layout
// ============================================================

/*
EN: Syntax reason: these structs contain only raw pointers and primitive scalar
    fields.  That layout can be copied by value into CUDA launch arguments and
    matches the SoA structs declared in main.cu.
KO: 문법 이유: 이 구조체들은 raw pointer와 기본 scalar 필드만 갖습니다. 그래서
    CUDA kernel launch 인자로 값 복사할 수 있고 main.cu의 SoA 구조체와 ABI가 맞습니다.

EN: Logic reason: Python tensors own the memory; binding.cpp only passes their
    addresses after dtype/shape/device checks.  Keeping this layer simple reduces
    the chance that a deadlock/spawn fix is invalidated by a host-side ABI mismatch.
KO: 논리 이유: 메모리 소유권은 Python tensor에 있고, binding.cpp는 dtype/shape/device
    검사를 통과한 주소만 넘깁니다. 이 계층을 단순하게 유지해야 deadlock/spawn 수정이
    host-side ABI 불일치 때문에 무효화되는 위험이 줄어듭니다.
*/
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

// ============================================================
// CUDA ECS launcher declarations
// ============================================================

extern "C" void launch_step_cuda_ecs(
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
    float current_time,
    float dt,
    int max_entities,
    int step_index,
    cudaStream_t stream
);

extern "C" void register_render_vbo_cuda(unsigned int vbo);
extern "C" void set_vehicle_texture_render_cuda(int enabled);

extern "C" void launch_render_vbo_cuda_ecs(
    ECSArrays ecs,
    int max_entities,
    cudaStream_t stream
);

extern "C" void unregister_render_vbo_cuda();

// ============================================================
// Tensor check helpers
// ============================================================

/*
KO 문법 이유: TORCH_CHECK는 조건이 false일 때 Python 예외를 던져 C++/CUDA 호출을
   즉시 중단합니다.
KO 논리 이유: CUDA kernel은 잘못된 dtype/shape를 복구할 수 없으므로 launch 전에
   명확히 막아야 오래된/잘못된 tensor가 교차로 정체나 차선 고정처럼 보이는 오류를
   만들지 않습니다.
*/
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static inline void check_same_cuda_device(
    const torch::Tensor& t,
    const char* name,
    const torch::Tensor& ref,
    const char* ref_name
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(ref.is_cuda(), ref_name, " must be CUDA tensor");

    TORCH_CHECK(
        t.device() == ref.device(),
        name,
        " must be on the same CUDA device as ",
        ref_name,
        "; got ",
        t.device(),
        " vs ",
        ref.device()
    );
}

static inline void check_1d_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t expected_numel,
    c10::ScalarType dtype,
    const torch::Tensor& ref_device_tensor
) {
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.numel() == expected_numel, name, ".numel() must equal ", expected_numel);
    TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    check_same_cuda_device(t, name, ref_device_tensor, "s");
}

static inline void check_agent_float_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t max_agents,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, max_agents, torch::kFloat32, ref_device_tensor);
}

static inline void check_agent_int_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t max_agents,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, max_agents, torch::kInt32, ref_device_tensor);
}

static inline void check_lane_float_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_lanes,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_lanes, torch::kFloat32, ref_device_tensor);
}

static inline void check_lane_int_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_lanes,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_lanes, torch::kInt32, ref_device_tensor);
}

static inline void check_spawn_float_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_spawn_points,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_spawn_points, torch::kFloat32, ref_device_tensor);
}

static inline void check_spawn_int_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_spawn_points,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_spawn_points, torch::kInt32, ref_device_tensor);
}

static inline void check_signal_int_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_signals,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_signals, torch::kInt32, ref_device_tensor);
}

static inline void check_signal_float_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_signals,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_signals, torch::kFloat32, ref_device_tensor);
}

static inline void check_node_int_tensor(
    const torch::Tensor& t,
    const char* name,
    int64_t num_nodes,
    const torch::Tensor& ref_device_tensor
) {
    check_1d_tensor(t, name, num_nodes, torch::kInt32, ref_device_tensor);
}

static inline void check_int64_to_int_range(int64_t v, const char* name) {
    TORCH_CHECK(
        v >= static_cast<int64_t>(std::numeric_limits<int>::min()) &&
            v <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        name,
        " exceeds int range"
    );
}

static inline void check_nonnegative_int64_to_int_range(int64_t v, const char* name) {
    TORCH_CHECK(v >= 0, name, " must be non-negative");
    check_int64_to_int_range(v, name);
}

// ============================================================
// ECS struct builders
// ============================================================

static inline ECSArrays make_ecs_arrays(
    torch::Tensor s,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor speed,
    torch::Tensor accel,
    torch::Tensor heading,
    torch::Tensor steer_angle,

    torch::Tensor vehicle_length,
    torch::Tensor vehicle_width,
    torch::Tensor reaction_time,
    torch::Tensor min_gap,

    torch::Tensor lane_id,
    torch::Tensor active,
    torch::Tensor driver_type,
    torch::Tensor route_id,
    torch::Tensor route_pos,

    torch::Tensor vehicle_state,
    torch::Tensor connector_from_lane,
    torch::Tensor connector_to_lane,
    torch::Tensor connector_s,
    torch::Tensor connector_length,

    torch::Tensor lane_change_active,
    torch::Tensor lane_change_from_lane,
    torch::Tensor lane_change_to_lane,
    torch::Tensor lane_change_t,
    torch::Tensor lane_change_duration,

    torch::Tensor aggressiveness,
    torch::Tensor politeness,
    torch::Tensor risk_tolerance,
    torch::Tensor comfort_decel,
    torch::Tensor desired_speed_factor,
    torch::Tensor lc_cooldown,

    torch::Tensor turn_signal,
    torch::Tensor turn_signal_time,

    torch::Tensor entry_time
) {
    ECSArrays ecs{};

    ecs.alive = active.data_ptr<int>();

    ecs.x = x.data_ptr<float>();
    ecs.y = y.data_ptr<float>();
    ecs.s = s.data_ptr<float>();
    ecs.speed = speed.data_ptr<float>();
    ecs.accel = accel.data_ptr<float>();
    ecs.heading = heading.data_ptr<float>();
    ecs.steer_angle = steer_angle.data_ptr<float>();

    ecs.length = vehicle_length.data_ptr<float>();
    ecs.width = vehicle_width.data_ptr<float>();

    ecs.driver_type = driver_type.data_ptr<int>();
    ecs.reaction_time = reaction_time.data_ptr<float>();
    ecs.min_gap = min_gap.data_ptr<float>();
    ecs.aggressiveness = aggressiveness.data_ptr<float>();
    ecs.politeness = politeness.data_ptr<float>();
    ecs.risk_tolerance = risk_tolerance.data_ptr<float>();
    ecs.comfort_decel = comfort_decel.data_ptr<float>();
    ecs.desired_speed_factor = desired_speed_factor.data_ptr<float>();

    ecs.lane_id = lane_id.data_ptr<int>();

    ecs.route_id = route_id.data_ptr<int>();
    ecs.route_pos = route_pos.data_ptr<int>();
    ecs.entry_time = entry_time.data_ptr<float>();

    ecs.vehicle_state = vehicle_state.data_ptr<int>();

    ecs.connector_from_lane = connector_from_lane.data_ptr<int>();
    ecs.connector_to_lane = connector_to_lane.data_ptr<int>();
    ecs.connector_s = connector_s.data_ptr<float>();
    ecs.connector_length = connector_length.data_ptr<float>();

    ecs.lane_change_active = lane_change_active.data_ptr<int>();
    ecs.lane_change_from_lane = lane_change_from_lane.data_ptr<int>();
    ecs.lane_change_to_lane = lane_change_to_lane.data_ptr<int>();
    ecs.lane_change_t = lane_change_t.data_ptr<float>();
    ecs.lane_change_duration = lane_change_duration.data_ptr<float>();
    ecs.lc_cooldown = lc_cooldown.data_ptr<float>();

    ecs.turn_signal = turn_signal.data_ptr<int>();
    ecs.turn_signal_time = turn_signal_time.data_ptr<float>();

    return ecs;
}

// ============================================================
// Per-step temporary buffer cache
// ============================================================

struct StepTempBufferCache {
    int64_t max_agents = -1;
    int device_index = -9999;

    torch::Tensor perception_front_gap;
    torch::Tensor perception_front_speed;
    torch::Tensor perception_front_s;
    torch::Tensor perception_front_length;
    torch::Tensor perception_front_lane;

    torch::Tensor perception_target_front_gap;
    torch::Tensor perception_target_front_speed;
    torch::Tensor perception_target_rear_gap;
    torch::Tensor perception_target_rear_speed;

    torch::Tensor decision_desired_speed;
    torch::Tensor decision_target_accel;
    torch::Tensor decision_wants_lane_change;
    torch::Tensor decision_lane_change_target;
    torch::Tensor decision_wants_connector;
    torch::Tensor decision_connector_target_lane;
    torch::Tensor decision_should_exit;

    bool valid_for(int64_t n, const torch::Tensor& ref) const {
        return max_agents == n
            && device_index == ref.device().index()
            && perception_front_gap.defined()
            && perception_front_gap.is_cuda()
            && perception_front_gap.device() == ref.device();
    }

    void ensure(int64_t n, const torch::Tensor& ref) {
        if (valid_for(n, ref)) return;

        // KO 문법 이유: torch::Tensor를 struct 멤버로 보관하면 reference count가 유지되어
        //    함수가 끝나도 CUDA 임시 버퍼가 해제되지 않습니다.
        // KO 논리 이유: 기존 코드는 sim.step()마다 16개의 max_agents 크기 tensor를 새로
        //    만들었습니다. 각 tick의 알고리즘은 그대로 두고, 같은 크기의 scratch buffer만
        //    재사용하면 Python/C++ 경계와 GPU allocator overhead가 크게 줄어듭니다.
        auto float_opts = ref.options().dtype(torch::kFloat32);
        auto int_opts = ref.options().dtype(torch::kInt32);

        perception_front_gap = torch::empty({n}, float_opts);
        perception_front_speed = torch::empty({n}, float_opts);
        perception_front_s = torch::empty({n}, float_opts);
        perception_front_length = torch::empty({n}, float_opts);
        perception_front_lane = torch::empty({n}, int_opts);

        perception_target_front_gap = torch::empty({n}, float_opts);
        perception_target_front_speed = torch::empty({n}, float_opts);
        perception_target_rear_gap = torch::empty({n}, float_opts);
        perception_target_rear_speed = torch::empty({n}, float_opts);

        decision_desired_speed = torch::empty({n}, float_opts);
        decision_target_accel = torch::empty({n}, float_opts);
        decision_wants_lane_change = torch::empty({n}, int_opts);
        decision_lane_change_target = torch::empty({n}, int_opts);
        decision_wants_connector = torch::empty({n}, int_opts);
        decision_connector_target_lane = torch::empty({n}, int_opts);
        decision_should_exit = torch::empty({n}, int_opts);

        max_agents = n;
        device_index = ref.device().index();
    }
};

static thread_local StepTempBufferCache g_step_temp_buffers;

// ============================================================
// Main simulation step binding
// ============================================================

void step_cuda(
    torch::Tensor s,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor speed,
    torch::Tensor accel,
    torch::Tensor heading,
    torch::Tensor steer_angle,

    torch::Tensor vehicle_length,
    torch::Tensor vehicle_width,
    torch::Tensor reaction_time,
    torch::Tensor min_gap,

    torch::Tensor lane_id,
    torch::Tensor active,
    torch::Tensor driver_type,
    torch::Tensor route_id,
    torch::Tensor route_pos,

    torch::Tensor vehicle_state,
    torch::Tensor connector_from_lane,
    torch::Tensor connector_to_lane,
    torch::Tensor connector_s,
    torch::Tensor connector_length,

    torch::Tensor lane_change_active,
    torch::Tensor lane_change_from_lane,
    torch::Tensor lane_change_to_lane,
    torch::Tensor lane_change_t,
    torch::Tensor lane_change_duration,

    torch::Tensor aggressiveness,
    torch::Tensor politeness,
    torch::Tensor risk_tolerance,
    torch::Tensor comfort_decel,
    torch::Tensor desired_speed_factor,
    torch::Tensor lc_cooldown,

    torch::Tensor turn_signal,
    torch::Tensor turn_signal_time,

    torch::Tensor lane_length,
    torch::Tensor lane_start_x,
    torch::Tensor lane_start_y,
    torch::Tensor lane_end_x,
    torch::Tensor lane_end_y,
    torch::Tensor lane_speed_limit,
    torch::Tensor lane_start_node,
    torch::Tensor lane_end_node,
    torch::Tensor left_lane,
    torch::Tensor right_lane,

    torch::Tensor conflict_lanes,

    torch::Tensor route_offsets,
    torch::Tensor route_lanes,
    torch::Tensor route_turns,

    torch::Tensor spawn_accumulator,
    torch::Tensor demand_vps,
    torch::Tensor demand_profile_vps,
    torch::Tensor demand_profile_has,
    int64_t demand_profile_slots,
    double demand_profile_slot_seconds,
    torch::Tensor spawn_lane,
    torch::Tensor spawn_route,

    torch::Tensor entry_time,

    torch::Tensor lane_cell_head,
    torch::Tensor lane_cell_next,

    torch::Tensor world_cell_head,
    torch::Tensor world_cell_next,

    double world_min_x,
    double world_min_y,
    double world_cell_size,
    int64_t world_grid_w,
    int64_t world_grid_h,

    torch::Tensor signal_node,
    torch::Tensor signal_turn,
    torch::Tensor signal_cycle,
    torch::Tensor signal_green_start,
    torch::Tensor signal_green_end,
    torch::Tensor signal_yellow_start,
    torch::Tensor signal_yellow_end,

    torch::Tensor rng_state,
    torch::Tensor metrics,

    double current_time,
    double dt,
    double av_penetration,
    int64_t max_agents,
    int64_t num_spawn_points,
    int64_t num_lanes,
    int64_t num_signals,
    int64_t step_index,

    torch::Tensor intersection_lock,
    torch::Tensor reservation_table,
    int64_t num_nodes
) {
    // ========================================================
    // Scalar validation
    // ========================================================

    TORCH_CHECK(max_agents > 0, "max_agents must be positive");
    TORCH_CHECK(num_lanes > 0, "num_lanes must be positive");
    TORCH_CHECK(num_spawn_points >= 0, "num_spawn_points must be non-negative");
    TORCH_CHECK(demand_profile_slots > 0, "demand_profile_slots must be positive");
    TORCH_CHECK(num_signals >= 0, "num_signals must be non-negative");
    TORCH_CHECK(num_nodes > 0, "num_nodes must be positive");

    TORCH_CHECK(world_grid_w > 0, "world_grid_w must be positive");
    TORCH_CHECK(world_grid_h > 0, "world_grid_h must be positive");
    TORCH_CHECK(world_cell_size > 0.0, "world_cell_size must be positive");

    TORCH_CHECK(std::isfinite(world_min_x), "world_min_x must be finite");
    TORCH_CHECK(std::isfinite(world_min_y), "world_min_y must be finite");
    TORCH_CHECK(std::isfinite(world_cell_size), "world_cell_size must be finite");

    TORCH_CHECK(dt > 0.0, "dt must be positive");
    TORCH_CHECK(dt <= 0.25, "dt must be <= 0.25");
    TORCH_CHECK(std::isfinite(current_time), "current_time must be finite");
    TORCH_CHECK(std::isfinite(dt), "dt must be finite");
    TORCH_CHECK(std::isfinite(av_penetration), "av_penetration must be finite");
    TORCH_CHECK(std::isfinite(demand_profile_slot_seconds), "demand_profile_slot_seconds must be finite");
    TORCH_CHECK(demand_profile_slot_seconds > 0.0, "demand_profile_slot_seconds must be positive");
    TORCH_CHECK(av_penetration >= 0.0 && av_penetration <= 1.0, "av_penetration must be in [0, 1]");

    check_nonnegative_int64_to_int_range(max_agents, "max_agents");
    check_nonnegative_int64_to_int_range(num_lanes, "num_lanes");
    check_nonnegative_int64_to_int_range(num_spawn_points, "num_spawn_points");
    check_nonnegative_int64_to_int_range(demand_profile_slots, "demand_profile_slots");
    check_nonnegative_int64_to_int_range(num_signals, "num_signals");
    check_nonnegative_int64_to_int_range(num_nodes, "num_nodes");
    check_nonnegative_int64_to_int_range(world_grid_w, "world_grid_w");
    check_nonnegative_int64_to_int_range(world_grid_h, "world_grid_h");
    check_nonnegative_int64_to_int_range(step_index, "step_index");

    TORCH_CHECK(
        world_grid_w <= std::numeric_limits<int64_t>::max() / world_grid_h,
        "world_grid_w * world_grid_h overflows int64"
    );

    const int64_t world_cells = world_grid_w * world_grid_h;

    TORCH_CHECK(
        world_cells <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "world_grid_w * world_grid_h exceeds int range"
    );

    TORCH_CHECK(
        num_nodes <= std::numeric_limits<int64_t>::max() / RES_HORIZON_SLOTS_BINDING,
        "num_nodes * RES_HORIZON_SLOTS overflows int64"
    );

    const int64_t reservation_slots =
        num_nodes * static_cast<int64_t>(RES_HORIZON_SLOTS_BINDING);

    CHECK_CUDA(s);
    const c10::cuda::CUDAGuard device_guard(s.device());

    // ========================================================
    // Agent tensors
    // ========================================================

    check_agent_float_tensor(s, "s", max_agents, s);
    check_agent_float_tensor(x, "x", max_agents, s);
    check_agent_float_tensor(y, "y", max_agents, s);
    check_agent_float_tensor(speed, "speed", max_agents, s);
    check_agent_float_tensor(accel, "accel", max_agents, s);
    check_agent_float_tensor(heading, "heading", max_agents, s);
    check_agent_float_tensor(steer_angle, "steer_angle", max_agents, s);

    check_agent_float_tensor(vehicle_length, "vehicle_length", max_agents, s);
    check_agent_float_tensor(vehicle_width, "vehicle_width", max_agents, s);
    check_agent_float_tensor(reaction_time, "reaction_time", max_agents, s);
    check_agent_float_tensor(min_gap, "min_gap", max_agents, s);

    check_agent_int_tensor(lane_id, "lane_id", max_agents, s);
    check_agent_int_tensor(active, "active", max_agents, s);
    check_agent_int_tensor(driver_type, "driver_type", max_agents, s);
    check_agent_int_tensor(route_id, "route_id", max_agents, s);
    check_agent_int_tensor(route_pos, "route_pos", max_agents, s);

    check_agent_int_tensor(vehicle_state, "vehicle_state", max_agents, s);
    check_agent_int_tensor(connector_from_lane, "connector_from_lane", max_agents, s);
    check_agent_int_tensor(connector_to_lane, "connector_to_lane", max_agents, s);
    check_agent_float_tensor(connector_s, "connector_s", max_agents, s);
    check_agent_float_tensor(connector_length, "connector_length", max_agents, s);

    check_agent_int_tensor(lane_change_active, "lane_change_active", max_agents, s);
    check_agent_int_tensor(lane_change_from_lane, "lane_change_from_lane", max_agents, s);
    check_agent_int_tensor(lane_change_to_lane, "lane_change_to_lane", max_agents, s);
    check_agent_float_tensor(lane_change_t, "lane_change_t", max_agents, s);
    check_agent_float_tensor(lane_change_duration, "lane_change_duration", max_agents, s);

    check_agent_float_tensor(aggressiveness, "aggressiveness", max_agents, s);
    check_agent_float_tensor(politeness, "politeness", max_agents, s);
    check_agent_float_tensor(risk_tolerance, "risk_tolerance", max_agents, s);
    check_agent_float_tensor(comfort_decel, "comfort_decel", max_agents, s);
    check_agent_float_tensor(desired_speed_factor, "desired_speed_factor", max_agents, s);
    check_agent_float_tensor(lc_cooldown, "lc_cooldown", max_agents, s);
    check_agent_int_tensor(turn_signal, "turn_signal", max_agents, s);
    check_agent_float_tensor(turn_signal_time, "turn_signal_time", max_agents, s);

    check_agent_float_tensor(entry_time, "entry_time", max_agents, s);

    // ========================================================
    // Lane tensors
    // ========================================================

    check_lane_float_tensor(lane_length, "lane_length", num_lanes, s);
    check_lane_float_tensor(lane_start_x, "lane_start_x", num_lanes, s);
    check_lane_float_tensor(lane_start_y, "lane_start_y", num_lanes, s);
    check_lane_float_tensor(lane_end_x, "lane_end_x", num_lanes, s);
    check_lane_float_tensor(lane_end_y, "lane_end_y", num_lanes, s);
    check_lane_float_tensor(lane_speed_limit, "lane_speed_limit", num_lanes, s);

    check_lane_int_tensor(lane_start_node, "lane_start_node", num_lanes, s);
    check_lane_int_tensor(lane_end_node, "lane_end_node", num_lanes, s);
    check_lane_int_tensor(left_lane, "left_lane", num_lanes, s);
    check_lane_int_tensor(right_lane, "right_lane", num_lanes, s);

    // conflict_lanes retained for Python ABI compatibility.
    CHECK_INPUT(conflict_lanes);
    CHECK_INT32(conflict_lanes);
    check_same_cuda_device(conflict_lanes, "conflict_lanes", s, "s");
    TORCH_CHECK(conflict_lanes.dim() == 2, "conflict_lanes must be 2D");
    TORCH_CHECK(conflict_lanes.size(0) == num_lanes, "conflict_lanes.size(0) must equal num_lanes");
    TORCH_CHECK(
        conflict_lanes.size(1) == MAX_CONFLICT_LANES_BINDING,
        "conflict_lanes.size(1) must equal ",
        MAX_CONFLICT_LANES_BINDING
    );

    // ========================================================
    // Route tensors
    // ========================================================

    CHECK_INPUT(route_offsets);
    CHECK_INT32(route_offsets);
    check_same_cuda_device(route_offsets, "route_offsets", s, "s");
    TORCH_CHECK(route_offsets.dim() == 1, "route_offsets must be 1D");
    TORCH_CHECK(route_offsets.numel() >= 2, "route_offsets must have at least 2 elements");
    TORCH_CHECK(
        route_offsets.numel() <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "route_offsets.numel() exceeds int range"
    );
    int64_t num_routes = route_offsets.numel() - 1;
    check_nonnegative_int64_to_int_range(num_routes, "num_routes");

    CHECK_INPUT(route_lanes);
    CHECK_INT32(route_lanes);
    check_same_cuda_device(route_lanes, "route_lanes", s, "s");
    TORCH_CHECK(route_lanes.dim() == 1, "route_lanes must be 1D");
    TORCH_CHECK(
        route_lanes.numel() <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "route_lanes.numel() exceeds int range"
    );

    CHECK_INPUT(route_turns);
    CHECK_INT32(route_turns);
    check_same_cuda_device(route_turns, "route_turns", s, "s");
    TORCH_CHECK(route_turns.dim() == 1, "route_turns must be 1D");
    TORCH_CHECK(route_turns.numel() == route_lanes.numel(), "route_turns.numel() must equal route_lanes.numel()");

    // ========================================================
    // Spawn tensors
    // ========================================================

    check_spawn_float_tensor(spawn_accumulator, "spawn_accumulator", num_spawn_points, s);
    check_spawn_float_tensor(demand_vps, "demand_vps", num_spawn_points, s);

    CHECK_INPUT(demand_profile_vps);
    CHECK_FLOAT32(demand_profile_vps);
    check_same_cuda_device(demand_profile_vps, "demand_profile_vps", s, "s");
    TORCH_CHECK(demand_profile_vps.dim() == 2, "demand_profile_vps must be 2D");
    TORCH_CHECK(demand_profile_vps.size(0) == num_spawn_points, "demand_profile_vps.size(0) must equal num_spawn_points");
    TORCH_CHECK(demand_profile_vps.size(1) == demand_profile_slots, "demand_profile_vps.size(1) must equal demand_profile_slots");

    check_spawn_int_tensor(demand_profile_has, "demand_profile_has", num_spawn_points, s);
    check_spawn_int_tensor(spawn_lane, "spawn_lane", num_spawn_points, s);
    check_spawn_int_tensor(spawn_route, "spawn_route", num_spawn_points, s);

    // ========================================================
    // Legacy lane-cell tensors
    // ========================================================

    CHECK_INPUT(lane_cell_head);
    CHECK_INT32(lane_cell_head);
    check_same_cuda_device(lane_cell_head, "lane_cell_head", s, "s");
    TORCH_CHECK(lane_cell_head.dim() == 1, "lane_cell_head must be 1D");

    CHECK_INPUT(lane_cell_next);
    CHECK_INT32(lane_cell_next);
    check_same_cuda_device(lane_cell_next, "lane_cell_next", s, "s");
    TORCH_CHECK(lane_cell_next.dim() == 1, "lane_cell_next must be 1D");
    TORCH_CHECK(
        lane_cell_next.numel() == max_agents || lane_cell_next.numel() == 0,
        "lane_cell_next.numel() must equal max_agents or be empty"
    );

    // ========================================================
    // World grid tensors
    // ========================================================

    CHECK_INPUT(world_cell_head);
    CHECK_INT32(world_cell_head);
    check_same_cuda_device(world_cell_head, "world_cell_head", s, "s");
    TORCH_CHECK(world_cell_head.dim() == 1, "world_cell_head must be 1D");
    TORCH_CHECK(world_cell_head.numel() == world_cells, "world_cell_head.numel() must equal world_grid_w * world_grid_h");

    CHECK_INPUT(world_cell_next);
    CHECK_INT32(world_cell_next);
    check_same_cuda_device(world_cell_next, "world_cell_next", s, "s");
    TORCH_CHECK(world_cell_next.dim() == 1, "world_cell_next must be 1D");
    TORCH_CHECK(world_cell_next.numel() == max_agents, "world_cell_next.numel() must equal max_agents");

    // ========================================================
    // Signal tensors
    // ========================================================

    check_signal_int_tensor(signal_node, "signal_node", num_signals, s);
    check_signal_int_tensor(signal_turn, "signal_turn", num_signals, s);
    check_signal_float_tensor(signal_cycle, "signal_cycle", num_signals, s);
    check_signal_float_tensor(signal_green_start, "signal_green_start", num_signals, s);
    check_signal_float_tensor(signal_green_end, "signal_green_end", num_signals, s);
    check_signal_float_tensor(signal_yellow_start, "signal_yellow_start", num_signals, s);
    check_signal_float_tensor(signal_yellow_end, "signal_yellow_end", num_signals, s);

    // ========================================================
    // RNG / metrics
    // ========================================================

    CHECK_INPUT(rng_state);
    CHECK_INT32(rng_state);
    check_same_cuda_device(rng_state, "rng_state", s, "s");
    TORCH_CHECK(rng_state.dim() == 1, "rng_state must be 1D");
    TORCH_CHECK(
        rng_state.numel() >= max_agents + num_spawn_points,
        "rng_state.numel() must be at least max_agents + num_spawn_points"
    );

    CHECK_INPUT(metrics);
    CHECK_FLOAT32(metrics);
    check_same_cuda_device(metrics, "metrics", s, "s");
    TORCH_CHECK(metrics.dim() == 1, "metrics must be 1D");
    TORCH_CHECK(metrics.numel() >= METRICS_SIZE_MIN, "metrics must have at least ", METRICS_SIZE_MIN, " elements");

    // ========================================================
    // Intersection / reservation tensors
    // ========================================================

    check_node_int_tensor(intersection_lock, "intersection_lock", num_nodes, s);

    check_1d_tensor(
        reservation_table,
        "reservation_table",
        reservation_slots,
        torch::kInt32,
        s
    );

    // ========================================================
    // Temporary ECS perception / decision buffers
    // ========================================================

    g_step_temp_buffers.ensure(max_agents, s);

    PerceptionSoA perception{};
    perception.front_gap = g_step_temp_buffers.perception_front_gap.data_ptr<float>();
    perception.front_speed = g_step_temp_buffers.perception_front_speed.data_ptr<float>();
    perception.front_s = g_step_temp_buffers.perception_front_s.data_ptr<float>();
    perception.front_length = g_step_temp_buffers.perception_front_length.data_ptr<float>();
    perception.front_lane = g_step_temp_buffers.perception_front_lane.data_ptr<int>();

    perception.target_front_gap = g_step_temp_buffers.perception_target_front_gap.data_ptr<float>();
    perception.target_front_speed = g_step_temp_buffers.perception_target_front_speed.data_ptr<float>();
    perception.target_rear_gap = g_step_temp_buffers.perception_target_rear_gap.data_ptr<float>();
    perception.target_rear_speed = g_step_temp_buffers.perception_target_rear_speed.data_ptr<float>();

    DecisionSoA decision{};
    decision.desired_speed = g_step_temp_buffers.decision_desired_speed.data_ptr<float>();
    decision.target_accel = g_step_temp_buffers.decision_target_accel.data_ptr<float>();
    decision.wants_lane_change = g_step_temp_buffers.decision_wants_lane_change.data_ptr<int>();
    decision.lane_change_target = g_step_temp_buffers.decision_lane_change_target.data_ptr<int>();
    decision.wants_connector = g_step_temp_buffers.decision_wants_connector.data_ptr<int>();
    decision.connector_target_lane = g_step_temp_buffers.decision_connector_target_lane.data_ptr<int>();
    decision.should_exit = g_step_temp_buffers.decision_should_exit.data_ptr<int>();

    // ========================================================
    // Build ECS structs
    // ========================================================

    ECSArrays ecs = make_ecs_arrays(
        s,
        x,
        y,
        speed,
        accel,
        heading,
        steer_angle,

        vehicle_length,
        vehicle_width,
        reaction_time,
        min_gap,

        lane_id,
        active,
        driver_type,
        route_id,
        route_pos,

        vehicle_state,
        connector_from_lane,
        connector_to_lane,
        connector_s,
        connector_length,

        lane_change_active,
        lane_change_from_lane,
        lane_change_to_lane,
        lane_change_t,
        lane_change_duration,

        aggressiveness,
        politeness,
        risk_tolerance,
        comfort_decel,
        desired_speed_factor,
        lc_cooldown,

        turn_signal,
        turn_signal_time,

        entry_time
    );

    RoadNetwork road{};
    road.lane_length = lane_length.data_ptr<float>();
    road.lane_start_x = lane_start_x.data_ptr<float>();
    road.lane_start_y = lane_start_y.data_ptr<float>();
    road.lane_end_x = lane_end_x.data_ptr<float>();
    road.lane_end_y = lane_end_y.data_ptr<float>();
    road.lane_speed_limit = lane_speed_limit.data_ptr<float>();

    road.lane_start_node = lane_start_node.data_ptr<int>();
    road.lane_end_node = lane_end_node.data_ptr<int>();
    road.left_lane = left_lane.data_ptr<int>();
    road.right_lane = right_lane.data_ptr<int>();

    road.route_offsets = route_offsets.data_ptr<int>();
    road.route_lanes = route_lanes.data_ptr<int>();
    road.route_turns = route_turns.data_ptr<int>();

    road.num_lanes = static_cast<int>(num_lanes);
    road.num_nodes = static_cast<int>(num_nodes);
    road.num_routes = static_cast<int>(num_routes);

    Signals signals{};
    signals.signal_node = signal_node.data_ptr<int>();
    signals.signal_turn = signal_turn.data_ptr<int>();
    signals.signal_cycle = signal_cycle.data_ptr<float>();
    signals.signal_green_start = signal_green_start.data_ptr<float>();
    signals.signal_green_end = signal_green_end.data_ptr<float>();
    signals.signal_yellow_start = signal_yellow_start.data_ptr<float>();
    signals.signal_yellow_end = signal_yellow_end.data_ptr<float>();
    signals.num_signals = static_cast<int>(num_signals);

    SpatialGrid grid{};
    grid.cell_head = world_cell_head.data_ptr<int>();
    grid.cell_next = world_cell_next.data_ptr<int>();
    grid.min_x = static_cast<float>(world_min_x);
    grid.min_y = static_cast<float>(world_min_y);
    grid.cell_size = static_cast<float>(world_cell_size);
    grid.width = static_cast<int>(world_grid_w);
    grid.height = static_cast<int>(world_grid_h);

    SpawnConfig spawn{};
    spawn.spawn_accumulator = spawn_accumulator.data_ptr<float>();
    spawn.demand_vps = demand_vps.data_ptr<float>();
    spawn.demand_profile_vps = demand_profile_vps.data_ptr<float>();
    spawn.demand_profile_has = demand_profile_has.data_ptr<int>();
    spawn.spawn_lane = spawn_lane.data_ptr<int>();
    spawn.spawn_route = spawn_route.data_ptr<int>();
    spawn.num_spawn_points = static_cast<int>(num_spawn_points);
    spawn.demand_profile_slots = static_cast<int>(demand_profile_slots);
    spawn.demand_profile_slot_seconds = static_cast<float>(demand_profile_slot_seconds);
    spawn.av_penetration = static_cast<float>(av_penetration);

    // ========================================================
    // Launch ECS CUDA pipeline
    // ========================================================

    auto stream = at::cuda::getCurrentCUDAStream(s.device().index());

    launch_step_cuda_ecs(
        ecs,
        road,
        signals,
        grid,
        spawn,
        perception,
        decision,
        reservation_table.data_ptr<int>(),
        reinterpret_cast<uint32_t*>(rng_state.data_ptr<int>()),
        metrics.data_ptr<float>(),
        static_cast<float>(current_time),
        static_cast<float>(dt),
        static_cast<int>(max_agents),
        static_cast<int>(step_index),
        stream.stream()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================
// Render VBO binding
// ============================================================

void register_render_vbo(int64_t vbo) {
    TORCH_CHECK(vbo >= 0, "vbo must be non-negative");
    TORCH_CHECK(
        vbo <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "vbo exceeds unsigned int range"
    );

    register_render_vbo_cuda(static_cast<unsigned int>(vbo));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void set_vehicle_texture_render(bool enabled) {
    set_vehicle_texture_render_cuda(enabled ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_render_vbo(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor heading,
    torch::Tensor steer_angle,
    torch::Tensor active,
    torch::Tensor driver_type,
    torch::Tensor vehicle_length,
    torch::Tensor vehicle_width,
    int64_t max_agents
) {
    TORCH_CHECK(max_agents > 0, "max_agents must be positive");
    check_nonnegative_int64_to_int_range(max_agents, "max_agents");

    CHECK_CUDA(x);
    const c10::cuda::CUDAGuard device_guard(x.device());

    check_agent_float_tensor(x, "x", max_agents, x);
    check_agent_float_tensor(y, "y", max_agents, x);
    check_agent_float_tensor(heading, "heading", max_agents, x);
    check_agent_float_tensor(steer_angle, "steer_angle", max_agents, x);
    check_agent_int_tensor(active, "active", max_agents, x);
    check_agent_int_tensor(driver_type, "driver_type", max_agents, x);
    check_agent_float_tensor(vehicle_length, "vehicle_length", max_agents, x);
    check_agent_float_tensor(vehicle_width, "vehicle_width", max_agents, x);

    // Render only needs a subset of ECS arrays.
    ECSArrays ecs{};
    ecs.alive = active.data_ptr<int>();
    ecs.x = x.data_ptr<float>();
    ecs.y = y.data_ptr<float>();
    ecs.heading = heading.data_ptr<float>();
    ecs.steer_angle = steer_angle.data_ptr<float>();
    ecs.driver_type = driver_type.data_ptr<int>();
    ecs.length = vehicle_length.data_ptr<float>();
    ecs.width = vehicle_width.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

    launch_render_vbo_cuda_ecs(
        ecs,
        static_cast<int>(max_agents),
        stream.stream()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unregister_render_vbo() {
    unregister_render_vbo_cuda();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================
// PyBind module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "step",
        &step_cuda,
        "CUDA ECS traffic ABM step with perception, decision, lane-change, connector, collision, safety, and metrics systems"
    );

    m.def(
        "register_render_vbo",
        &register_render_vbo,
        "Register OpenGL VBO for CUDA rendering"
    );

    m.def(
        "set_vehicle_texture_render",
        &set_vehicle_texture_render,
        "Enable or disable textured vehicle rendering in the CUDA VBO writer"
    );

    m.def(
        "update_render_vbo",
        &update_render_vbo,
        "Update OpenGL VBO from ECS CUDA tensors"
    );

    m.def(
        "unregister_render_vbo",
        &unregister_render_vbo,
        "Unregister OpenGL VBO"
    );
}