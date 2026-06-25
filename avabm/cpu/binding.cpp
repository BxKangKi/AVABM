#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "cpu_port_api.hpp"

namespace {

constexpr int HUMAN = 0;
constexpr int AV = 1;
constexpr int VEH_ON_LANE = 0;
constexpr int ENTITY_FREE = 0;
constexpr int ENTITY_ALIVE = 1;
constexpr int ENTITY_SPAWNING = 2;
constexpr int INDICATOR_NONE = 0;
constexpr int RES_HORIZON_SLOTS = 16;
constexpr float MAX_SPEED_FALLBACK = 13.9f;
constexpr float MIN_BUMPER_GAP = 2.5f;
constexpr float SAFE_TIME_HEADWAY_AV = 0.9f;
constexpr float SAFE_TIME_HEADWAY_HUMAN = 1.75f;
constexpr float SAFE_GAP_AV = 3.0f;
constexpr float SAFE_GAP_HUMAN = 6.8f;
constexpr float MAX_ACCEL_AV = 2.8f;
constexpr float MAX_ACCEL_HUMAN = 2.0f;
constexpr float MAX_DECEL_AV = 4.0f;
constexpr float MAX_DECEL_HUMAN = 3.4f;
constexpr float EMERGENCY_DECEL = 7.0f;
constexpr float LANE_CHANGE_DURATION_AV = 2.5f;
constexpr float LANE_CHANGE_DURATION_HUMAN = 4.8f;
constexpr float SPAWN_ACCUMULATOR_MAX = 10000.0f;
constexpr int SPAWN_MAX_PER_POINT_PER_STEP = 8;
constexpr float HUGE_GAP = 1.0e20f;
constexpr int METRICS_SIZE_MIN = 112;

enum MetricIndex {
    METRIC_SPAWNED = 0,
    METRIC_EXITED = 1,
    METRIC_TRAVEL_TIME = 6,
    METRIC_ACTIVE = 7,
    METRIC_ACCEL_COUNT = 8,
    METRIC_ACCEL_SUM = 9,
    METRIC_ACCEL_SQ_SUM = 10,
    METRIC_DECEL_COUNT = 11,
    METRIC_DECEL_SUM = 12,
    METRIC_DECEL_SQ_SUM = 13,
    METRIC_SPEED_SUM = 14,
    METRIC_SPEED_COUNT = 15,
    METRIC_SLOW_COUNT = 16,
    METRIC_STOP_COUNT = 19,
    METRIC_SPAWN_FAIL = 22,
    METRIC_SENSOR_DETECTION = 63,
    METRIC_SENSOR_FRONT_HIT = 64,
    METRIC_QUEUE_DELAY_SUM = 67,
    METRIC_QUEUE_DELAY_COUNT = 68,
    METRIC_STANDSTILL_TIME = 74,
    METRIC_TIME_LOSS_SUM = 75,
    METRIC_TIME_LOSS_COUNT = 76,
    METRIC_HARD_BRAKE = 40,
    METRIC_HEADWAY_SUM = 54,
    METRIC_HEADWAY_COUNT = 55,
    METRIC_MIN_GAP_SUM = 56,
    METRIC_MIN_GAP_COUNT = 57,
};

std::atomic<int> g_cpu_threads{0};
std::atomic<int> g_alloc_cursor{0};

inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline bool finitef(float v) {
    return std::isfinite(static_cast<double>(v));
}

inline int default_thread_count() {
    const unsigned int hc = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hc == 0 ? 1 : hc));
}

inline int resolved_thread_count() {
    const int requested = g_cpu_threads.load(std::memory_order_relaxed);
    if (requested > 0) return std::max(1, requested);
    return default_thread_count();
}

uint32_t lcg_next(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float rand_uniform(uint32_t& state) {
    return static_cast<float>((lcg_next(state) >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

struct TensorF {
    PyObject* obj = nullptr;
    float* data = nullptr;
    int64_t numel = 0;
};

struct TensorI {
    PyObject* obj = nullptr;
    int* data = nullptr;
    int64_t numel = 0;
};

bool set_error(const std::string& msg) {
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return false;
}

PyObject* tuple_item(PyObject* args, Py_ssize_t idx, const char* name) {
    if (!PyTuple_Check(args) || idx < 0 || idx >= PyTuple_Size(args)) {
        PyErr_Format(PyExc_TypeError, "missing argument %s at index %zd", name, idx);
        return nullptr;
    }
    return PyTuple_GET_ITEM(args, idx);  // borrowed
}

bool py_bool_attr(PyObject* obj, const char* attr, bool* out) {
    PyObject* v = PyObject_GetAttrString(obj, attr);
    if (!v) return false;
    const int b = PyObject_IsTrue(v);
    Py_DECREF(v);
    if (b < 0) return false;
    *out = (b != 0);
    return true;
}

bool py_bool_method0(PyObject* obj, const char* method, bool* out) {
    PyObject* v = PyObject_CallMethod(obj, method, nullptr);
    if (!v) return false;
    const int b = PyObject_IsTrue(v);
    Py_DECREF(v);
    if (b < 0) return false;
    *out = (b != 0);
    return true;
}

bool py_int_method0(PyObject* obj, const char* method, int64_t* out) {
    PyObject* v = PyObject_CallMethod(obj, method, nullptr);
    if (!v) return false;
    const long long r = PyLong_AsLongLong(v);
    Py_DECREF(v);
    if (PyErr_Occurred()) return false;
    *out = static_cast<int64_t>(r);
    return true;
}

bool py_uintptr_method0(PyObject* obj, const char* method, uintptr_t* out) {
    PyObject* v = PyObject_CallMethod(obj, method, nullptr);
    if (!v) return false;
    unsigned long long r = PyLong_AsUnsignedLongLong(v);
    Py_DECREF(v);
    if (PyErr_Occurred()) return false;
    *out = static_cast<uintptr_t>(r);
    return true;
}

bool py_dtype_matches(PyObject* obj, const char* wanted) {
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
    if (!dtype) return false;
    PyObject* dtype_str = PyObject_Str(dtype);
    Py_DECREF(dtype);
    if (!dtype_str) return false;
    const char* s = PyUnicode_AsUTF8(dtype_str);
    bool ok = (s != nullptr && std::string(s) == wanted);
    Py_DECREF(dtype_str);
    if (!ok && !PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "tensor dtype must be %s", wanted);
    }
    return ok;
}

bool validate_tensor_common(PyObject* obj, const char* name, const char* dtype, int64_t* numel, uintptr_t* ptr) {
    bool is_cuda = false;
    if (!py_bool_attr(obj, "is_cuda", &is_cuda)) {
        PyErr_Format(PyExc_TypeError, "%s must look like a torch.Tensor with is_cuda", name);
        return false;
    }
    if (is_cuda) {
        PyErr_Format(PyExc_ValueError, "%s must be a CPU tensor for avabm_cpu", name);
        return false;
    }
    bool contiguous = false;
    if (!py_bool_method0(obj, "is_contiguous", &contiguous)) {
        PyErr_Format(PyExc_TypeError, "%s must support is_contiguous()", name);
        return false;
    }
    if (!contiguous) {
        PyErr_Format(PyExc_ValueError, "%s must be contiguous", name);
        return false;
    }
    if (!py_dtype_matches(obj, dtype)) {
        PyErr_Format(PyExc_TypeError, "%s has wrong dtype; expected %s", name, dtype);
        return false;
    }
    if (!py_int_method0(obj, "numel", numel)) {
        PyErr_Format(PyExc_TypeError, "%s must support numel()", name);
        return false;
    }
    if (*numel < 0) {
        PyErr_Format(PyExc_ValueError, "%s numel() is negative", name);
        return false;
    }
    if (!py_uintptr_method0(obj, "data_ptr", ptr)) {
        PyErr_Format(PyExc_TypeError, "%s must support data_ptr()", name);
        return false;
    }
    return true;
}

bool check_dim_1(PyObject* obj, const char* name) {
    int64_t dim = -1;
    if (!py_int_method0(obj, "dim", &dim)) {
        PyErr_Clear();
        PyObject* ndim = PyObject_GetAttrString(obj, "ndim");
        if (!ndim) {
            PyErr_Format(PyExc_TypeError, "%s must expose dim() or ndim", name);
            return false;
        }
        dim = PyLong_AsLongLong(ndim);
        Py_DECREF(ndim);
        if (PyErr_Occurred()) return false;
    }
    if (dim != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1D", name);
        return false;
    }
    return true;
}

bool get_tensor_f(PyObject* args, Py_ssize_t idx, const char* name, int64_t expected, TensorF* out) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    uintptr_t ptr = 0;
    int64_t n = 0;
    if (!validate_tensor_common(obj, name, "torch.float32", &n, &ptr)) return false;
    if (!check_dim_1(obj, name)) return false;
    if (n != expected) {
        PyErr_Format(PyExc_ValueError, "%s.numel() must equal %lld, got %lld", name, static_cast<long long>(expected), static_cast<long long>(n));
        return false;
    }
    out->obj = obj;
    out->data = reinterpret_cast<float*>(ptr);
    out->numel = n;
    return true;
}

bool get_tensor_i(PyObject* args, Py_ssize_t idx, const char* name, int64_t expected, TensorI* out) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    uintptr_t ptr = 0;
    int64_t n = 0;
    if (!validate_tensor_common(obj, name, "torch.int32", &n, &ptr)) return false;
    if (!check_dim_1(obj, name)) return false;
    if (n != expected) {
        PyErr_Format(PyExc_ValueError, "%s.numel() must equal %lld, got %lld", name, static_cast<long long>(expected), static_cast<long long>(n));
        return false;
    }
    out->obj = obj;
    out->data = reinterpret_cast<int*>(ptr);
    out->numel = n;
    return true;
}

bool get_tensor_f_at_least(PyObject* args, Py_ssize_t idx, const char* name, int64_t minimum, TensorF* out, bool require_1d = false) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    uintptr_t ptr = 0;
    int64_t n = 0;
    if (!validate_tensor_common(obj, name, "torch.float32", &n, &ptr)) return false;
    if (require_1d && !check_dim_1(obj, name)) return false;
    if (n < minimum) {
        PyErr_Format(PyExc_ValueError, "%s.numel() must be at least %lld, got %lld", name, static_cast<long long>(minimum), static_cast<long long>(n));
        return false;
    }
    out->obj = obj;
    out->data = reinterpret_cast<float*>(ptr);
    out->numel = n;
    return true;
}

bool get_tensor_i_at_least(PyObject* args, Py_ssize_t idx, const char* name, int64_t minimum, TensorI* out, bool require_1d = false) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    uintptr_t ptr = 0;
    int64_t n = 0;
    if (!validate_tensor_common(obj, name, "torch.int32", &n, &ptr)) return false;
    if (require_1d && !check_dim_1(obj, name)) return false;
    if (n < minimum) {
        PyErr_Format(PyExc_ValueError, "%s.numel() must be at least %lld, got %lld", name, static_cast<long long>(minimum), static_cast<long long>(n));
        return false;
    }
    out->obj = obj;
    out->data = reinterpret_cast<int*>(ptr);
    out->numel = n;
    return true;
}

bool get_i64(PyObject* args, Py_ssize_t idx, const char* name, int64_t* out) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    long long v = PyLong_AsLongLong(obj);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "%s must be int-like", name);
        return false;
    }
    *out = static_cast<int64_t>(v);
    return true;
}

bool get_double(PyObject* args, Py_ssize_t idx, const char* name, double* out) {
    PyObject* obj = tuple_item(args, idx, name);
    if (!obj) return false;
    double v = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "%s must be float-like", name);
        return false;
    }
    *out = v;
    return true;
}

void check_nonnegative_i64_to_int(int64_t v, const char* name) {
    if (v < 0) {
        PyErr_Format(PyExc_ValueError, "%s must be non-negative", name);
        return;
    }
    if (v > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        PyErr_Format(PyExc_OverflowError, "%s exceeds int range", name);
        return;
    }
}

float lane_len_or_geom(int ln, const float* lane_length, const float* sx, const float* sy, const float* ex, const float* ey) {
    float L = lane_length[ln];
    if (finitef(L) && L > 0.1f) return L;
    const float dx = ex[ln] - sx[ln];
    const float dy = ey[ln] - sy[ln];
    L = std::sqrt(dx * dx + dy * dy);
    return (finitef(L) && L > 0.1f) ? L : 0.1f;
}

float lane_heading(int ln, const float* sx, const float* sy, const float* ex, const float* ey) {
    return std::atan2(ey[ln] - sy[ln], ex[ln] - sx[ln]);
}

void lane_xy_from_s(int ln, float s, const float* lane_length, const float* sx, const float* sy, const float* ex, const float* ey, float& ox, float& oy) {
    const float L = lane_len_or_geom(ln, lane_length, sx, sy, ex, ey);
    const float q = clampf(s / std::max(L, 1.0e-6f), 0.0f, 1.0f);
    ox = sx[ln] + (ex[ln] - sx[ln]) * q;
    oy = sy[ln] + (ey[ln] - sy[ln]) * q;
}

int route_pos_for_lane(int rid, int ln, const int* route_offsets, const int* route_lanes, int num_routes) {
    if (rid < 0 || rid >= num_routes) return -1;
    const int ro0 = route_offsets[rid];
    const int ro1 = route_offsets[rid + 1];
    if (ro1 <= ro0) return -1;
    for (int k = ro0; k < ro1; ++k) {
        if (route_lanes[k] == ln) return k - ro0;
    }
    return -1;
}

float spawn_rate_at(const float* demand_vps, const float* profile, const int* has_profile, int p, int slots, float slot_seconds, float current_time) {
    float rate = demand_vps[p];
    if (has_profile != nullptr && profile != nullptr && slots > 0 && has_profile[p] != 0 && slot_seconds > 1.0e-6f) {
        const float cycle = std::max(slot_seconds * static_cast<float>(slots), slot_seconds);
        float local = std::fmod(std::max(0.0f, current_time), cycle);
        if (local < 0.0f) local += cycle;
        int slot = clampi(static_cast<int>(std::floor(local / slot_seconds)), 0, slots - 1);
        rate = profile[static_cast<int64_t>(p) * slots + slot];
    }
    if (!finitef(rate) || rate < 0.0f) rate = 0.0f;
    return rate;
}

bool spawn_area_clear(int ln, float spawn_s, float required_gap, const int* active, const int* lane_id, const float* s, const float* length, int max_agents) {
    for (int i = 0; i < max_agents; ++i) {
        if (active[i] != ENTITY_ALIVE) continue;
        if (lane_id[i] != ln) continue;
        const float li = (finitef(length[i]) && length[i] > 0.1f) ? length[i] : 4.5f;
        const float gap = std::fabs(s[i] - spawn_s) - 0.5f * li;
        if (gap < required_gap) return false;
    }
    return true;
}

int find_free_entity(int max_agents, int* active) {
    if (max_agents <= 0) return -1;
    const int start = g_alloc_cursor.fetch_add(97, std::memory_order_relaxed);
    for (int k = 0; k < max_agents; ++k) {
        const int idx = (start + k) % max_agents;
        if (active[idx] == ENTITY_FREE) {
            active[idx] = ENTITY_SPAWNING;
            return idx;
        }
    }
    return -1;
}

template <typename Fn>
void parallel_ranges(int64_t n, Fn&& fn) {
    if (n <= 0) return;
    const int T = std::min<int64_t>(static_cast<int64_t>(resolved_thread_count()), n);
    if (T <= 1) {
        fn(0, n, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(T - 1));
    const int64_t chunk = (n + T - 1) / T;
    for (int tid = 1; tid < T; ++tid) {
        const int64_t begin = static_cast<int64_t>(tid) * chunk;
        const int64_t end = std::min<int64_t>(n, begin + chunk);
        if (begin >= end) break;
        threads.emplace_back([&, begin, end, tid]() { fn(begin, end, tid); });
    }
    const int64_t end0 = std::min<int64_t>(n, chunk);
    fn(0, end0, 0);
    for (auto& th : threads) th.join();
}

struct MetricBlock {
    double v[METRICS_SIZE_MIN];
    MetricBlock() { std::fill(std::begin(v), std::end(v), 0.0); }
    void add(int idx, double value) {
        if (idx >= 0 && idx < METRICS_SIZE_MIN) v[idx] += value;
    }
};

struct Inputs {
    TensorF s, x, y, speed, accel, heading, steer_angle, vehicle_length, vehicle_width;
    TensorF reaction_time, min_gap;
    TensorI lane_id, active, driver_type, route_id, route_pos, vehicle_state;
    TensorI connector_from_lane, connector_to_lane;
    TensorF connector_s, connector_length;
    TensorI lane_change_active, lane_change_from_lane, lane_change_to_lane;
    TensorF lane_change_t, lane_change_duration;
    TensorF aggressiveness, politeness, risk_tolerance, comfort_decel, desired_speed_factor, lc_cooldown;
    TensorI turn_signal;
    TensorF turn_signal_time;
    TensorF lane_length, lane_start_x, lane_start_y, lane_end_x, lane_end_y, lane_speed_limit;
    TensorI lane_start_node, lane_end_node, left_lane, right_lane, conflict_lanes;
    TensorI route_offsets, route_lanes, route_turns;
    TensorF spawn_accumulator, demand_vps, demand_profile_vps;
    TensorI demand_profile_has, spawn_lane, spawn_route;
    TensorF entry_time;
    TensorI lane_cell_head, lane_cell_next, world_cell_head, world_cell_next;
    TensorI signal_node, signal_turn;
    TensorF signal_cycle, signal_green_start, signal_green_end, signal_yellow_start, signal_yellow_end;
    TensorI rng_state;
    TensorF metrics;
    TensorI intersection_lock, reservation_table;
    int64_t demand_profile_slots = 0;
    double demand_profile_slot_seconds = 0.0;
    double world_min_x = 0.0;
    double world_min_y = 0.0;
    double world_cell_size = 0.0;
    int64_t world_grid_w = 0;
    int64_t world_grid_h = 0;
    double current_time = 0.0;
    double dt = 0.0;
    double av_penetration = 0.0;
    int64_t max_agents = 0;
    int64_t num_spawn_points = 0;
    int64_t num_lanes = 0;
    int64_t num_signals = 0;
    int64_t step_index = 0;
    int64_t num_nodes = 0;
    int64_t batch_steps = 1;
};

bool parse_inputs(PyObject* args, bool batched, Inputs* in) {
    const Py_ssize_t expected = batched ? 87 : 86;
    if (!PyTuple_Check(args) || PyTuple_Size(args) != expected) {
        PyErr_Format(PyExc_TypeError, "%s expects %zd arguments, got %zd", batched ? "step_batch" : "step", expected, PyTuple_Check(args) ? PyTuple_Size(args) : -1);
        return false;
    }
    if (!get_i64(args, 78, "max_agents", &in->max_agents)) return false;
    if (!get_i64(args, 79, "num_spawn_points", &in->num_spawn_points)) return false;
    if (!get_i64(args, 80, "num_lanes", &in->num_lanes)) return false;
    if (!get_i64(args, 81, "num_signals", &in->num_signals)) return false;
    if (!get_i64(args, 82, "step_index", &in->step_index)) return false;
    if (!get_i64(args, 85, "num_nodes", &in->num_nodes)) return false;
    if (batched && !get_i64(args, 86, "batch_steps", &in->batch_steps)) return false;
    if (!batched) in->batch_steps = 1;

    check_nonnegative_i64_to_int(in->max_agents, "max_agents"); if (PyErr_Occurred()) return false;
    check_nonnegative_i64_to_int(in->num_spawn_points, "num_spawn_points"); if (PyErr_Occurred()) return false;
    check_nonnegative_i64_to_int(in->num_lanes, "num_lanes"); if (PyErr_Occurred()) return false;
    check_nonnegative_i64_to_int(in->num_signals, "num_signals"); if (PyErr_Occurred()) return false;
    check_nonnegative_i64_to_int(in->num_nodes, "num_nodes"); if (PyErr_Occurred()) return false;
    if (in->max_agents <= 0) return set_error("max_agents must be positive");
    if (in->num_lanes <= 0) return set_error("num_lanes must be positive");
    if (in->batch_steps <= 0) return set_error("batch_steps must be positive");

    if (!get_i64(args, 52, "demand_profile_slots", &in->demand_profile_slots)) return false;
    if (!get_double(args, 53, "demand_profile_slot_seconds", &in->demand_profile_slot_seconds)) return false;
    if (!get_double(args, 61, "world_min_x", &in->world_min_x)) return false;
    if (!get_double(args, 62, "world_min_y", &in->world_min_y)) return false;
    if (!get_double(args, 63, "world_cell_size", &in->world_cell_size)) return false;
    if (!get_i64(args, 64, "world_grid_w", &in->world_grid_w)) return false;
    if (!get_i64(args, 65, "world_grid_h", &in->world_grid_h)) return false;
    if (!get_double(args, 75, "current_time", &in->current_time)) return false;
    if (!get_double(args, 76, "dt", &in->dt)) return false;
    if (!get_double(args, 77, "av_penetration", &in->av_penetration)) return false;
    if (!std::isfinite(in->current_time)) return set_error("current_time must be finite");
    if (!std::isfinite(in->dt) || in->dt <= 0.0 || in->dt > 0.5) return set_error("dt must be in (0, 0.5]");
    if (in->demand_profile_slots < 0) return set_error("demand_profile_slots must be non-negative");

    const int64_t ma = in->max_agents;
    const int64_t nl = in->num_lanes;
    const int64_t ns = in->num_spawn_points;
    const int64_t nsg = in->num_signals;
    const int64_t nn = in->num_nodes;

    if (!get_tensor_f(args, 0, "s", ma, &in->s)) return false;
    if (!get_tensor_f(args, 1, "x", ma, &in->x)) return false;
    if (!get_tensor_f(args, 2, "y", ma, &in->y)) return false;
    if (!get_tensor_f(args, 3, "speed", ma, &in->speed)) return false;
    if (!get_tensor_f(args, 4, "accel", ma, &in->accel)) return false;
    if (!get_tensor_f(args, 5, "heading", ma, &in->heading)) return false;
    if (!get_tensor_f(args, 6, "steer_angle", ma, &in->steer_angle)) return false;
    if (!get_tensor_f(args, 7, "vehicle_length", ma, &in->vehicle_length)) return false;
    if (!get_tensor_f(args, 8, "vehicle_width", ma, &in->vehicle_width)) return false;
    if (!get_tensor_f(args, 9, "reaction_time", ma, &in->reaction_time)) return false;
    if (!get_tensor_f(args, 10, "min_gap", ma, &in->min_gap)) return false;
    if (!get_tensor_i(args, 11, "lane_id", ma, &in->lane_id)) return false;
    if (!get_tensor_i(args, 12, "active", ma, &in->active)) return false;
    if (!get_tensor_i(args, 13, "driver_type", ma, &in->driver_type)) return false;
    if (!get_tensor_i(args, 14, "route_id", ma, &in->route_id)) return false;
    if (!get_tensor_i(args, 15, "route_pos", ma, &in->route_pos)) return false;
    if (!get_tensor_i(args, 16, "vehicle_state", ma, &in->vehicle_state)) return false;
    if (!get_tensor_i(args, 17, "connector_from_lane", ma, &in->connector_from_lane)) return false;
    if (!get_tensor_i(args, 18, "connector_to_lane", ma, &in->connector_to_lane)) return false;
    if (!get_tensor_f(args, 19, "connector_s", ma, &in->connector_s)) return false;
    if (!get_tensor_f(args, 20, "connector_length", ma, &in->connector_length)) return false;
    if (!get_tensor_i(args, 21, "lane_change_active", ma, &in->lane_change_active)) return false;
    if (!get_tensor_i(args, 22, "lane_change_from_lane", ma, &in->lane_change_from_lane)) return false;
    if (!get_tensor_i(args, 23, "lane_change_to_lane", ma, &in->lane_change_to_lane)) return false;
    if (!get_tensor_f(args, 24, "lane_change_t", ma, &in->lane_change_t)) return false;
    if (!get_tensor_f(args, 25, "lane_change_duration", ma, &in->lane_change_duration)) return false;
    if (!get_tensor_f(args, 26, "aggressiveness", ma, &in->aggressiveness)) return false;
    if (!get_tensor_f(args, 27, "politeness", ma, &in->politeness)) return false;
    if (!get_tensor_f(args, 28, "risk_tolerance", ma, &in->risk_tolerance)) return false;
    if (!get_tensor_f(args, 29, "comfort_decel", ma, &in->comfort_decel)) return false;
    if (!get_tensor_f(args, 30, "desired_speed_factor", ma, &in->desired_speed_factor)) return false;
    if (!get_tensor_f(args, 31, "lc_cooldown", ma, &in->lc_cooldown)) return false;
    if (!get_tensor_i(args, 32, "turn_signal", ma, &in->turn_signal)) return false;
    if (!get_tensor_f(args, 33, "turn_signal_time", ma, &in->turn_signal_time)) return false;

    if (!get_tensor_f(args, 34, "lane_length", nl, &in->lane_length)) return false;
    if (!get_tensor_f(args, 35, "lane_start_x", nl, &in->lane_start_x)) return false;
    if (!get_tensor_f(args, 36, "lane_start_y", nl, &in->lane_start_y)) return false;
    if (!get_tensor_f(args, 37, "lane_end_x", nl, &in->lane_end_x)) return false;
    if (!get_tensor_f(args, 38, "lane_end_y", nl, &in->lane_end_y)) return false;
    if (!get_tensor_f(args, 39, "lane_speed_limit", nl, &in->lane_speed_limit)) return false;
    if (!get_tensor_i(args, 40, "lane_start_node", nl, &in->lane_start_node)) return false;
    if (!get_tensor_i(args, 41, "lane_end_node", nl, &in->lane_end_node)) return false;
    if (!get_tensor_i(args, 42, "left_lane", nl, &in->left_lane)) return false;
    if (!get_tensor_i(args, 43, "right_lane", nl, &in->right_lane)) return false;
    if (!get_tensor_i_at_least(args, 44, "conflict_lanes", nl * 8, &in->conflict_lanes, false)) return false;

    if (!get_tensor_i_at_least(args, 45, "route_offsets", 2, &in->route_offsets, true)) return false;
    if (!get_tensor_i_at_least(args, 46, "route_lanes", 0, &in->route_lanes, true)) return false;
    if (!get_tensor_i_at_least(args, 47, "route_turns", 0, &in->route_turns, true)) return false;
    if (in->route_lanes.numel != in->route_turns.numel) return set_error("route_lanes and route_turns length mismatch");

    if (!get_tensor_f(args, 48, "spawn_accumulator", ns, &in->spawn_accumulator)) return false;
    if (!get_tensor_f(args, 49, "demand_vps", ns, &in->demand_vps)) return false;
    if (!get_tensor_f_at_least(args, 50, "demand_profile_vps", 0, &in->demand_profile_vps, false)) return false;
    if (!get_tensor_i(args, 51, "demand_profile_has", ns, &in->demand_profile_has)) return false;
    if (in->demand_profile_slots > 0 && in->demand_profile_vps.numel < ns * in->demand_profile_slots) return set_error("demand_profile_vps is too small");
    if (!get_tensor_i(args, 54, "spawn_lane", ns, &in->spawn_lane)) return false;
    if (!get_tensor_i(args, 55, "spawn_route", ns, &in->spawn_route)) return false;
    if (!get_tensor_f(args, 56, "entry_time", ma, &in->entry_time)) return false;

    if (!get_tensor_i_at_least(args, 57, "lane_cell_head", nl, &in->lane_cell_head, false)) return false;
    if (!get_tensor_i(args, 58, "lane_cell_next", ma, &in->lane_cell_next)) return false;
    if (!get_tensor_i_at_least(args, 59, "world_cell_head", 0, &in->world_cell_head, false)) return false;
    if (!get_tensor_i(args, 60, "world_cell_next", ma, &in->world_cell_next)) return false;

    if (!get_tensor_i(args, 66, "signal_node", nsg, &in->signal_node)) return false;
    if (!get_tensor_i(args, 67, "signal_turn", nsg, &in->signal_turn)) return false;
    if (!get_tensor_f(args, 68, "signal_cycle", nsg, &in->signal_cycle)) return false;
    if (!get_tensor_f(args, 69, "signal_green_start", nsg, &in->signal_green_start)) return false;
    if (!get_tensor_f(args, 70, "signal_green_end", nsg, &in->signal_green_end)) return false;
    if (!get_tensor_f(args, 71, "signal_yellow_start", nsg, &in->signal_yellow_start)) return false;
    if (!get_tensor_f(args, 72, "signal_yellow_end", nsg, &in->signal_yellow_end)) return false;

    if (!get_tensor_i_at_least(args, 73, "rng_state", ma + ns, &in->rng_state, true)) return false;
    if (!get_tensor_f_at_least(args, 74, "metrics", METRICS_SIZE_MIN, &in->metrics, true)) return false;
    if (!get_tensor_i(args, 83, "intersection_lock", nn, &in->intersection_lock)) return false;
    if (!get_tensor_i(args, 84, "reservation_table", nn * RES_HORIZON_SLOTS, &in->reservation_table)) return false;
    return true;
}


struct CpuCudaParityTempBuffers {
    std::vector<float> perception_front_gap;
    std::vector<float> perception_front_speed;
    std::vector<float> perception_front_s;
    std::vector<float> perception_front_length;
    std::vector<int> perception_front_lane;
    std::vector<float> perception_target_front_gap;
    std::vector<float> perception_target_front_speed;
    std::vector<float> perception_target_rear_gap;
    std::vector<float> perception_target_rear_speed;
    std::vector<float> decision_desired_speed;
    std::vector<float> decision_target_accel;
    std::vector<int> decision_wants_lane_change;
    std::vector<int> decision_lane_change_target;
    std::vector<int> decision_wants_connector;
    std::vector<int> decision_connector_target_lane;
    std::vector<int> decision_should_exit;
    std::vector<int> active_ids;
    std::vector<int> active_count;
    std::vector<int> lane_active_ids;
    std::vector<int> lane_active_count;
    std::vector<int> connector_active_ids;
    std::vector<int> connector_active_count;
    std::vector<int> spawn_alloc_cursor;
    int64_t capacity = -1;

    void ensure(int64_t max_entities) {
        if (max_entities <= 0) max_entities = 1;
        if (capacity == max_entities && !active_ids.empty()) return;
        capacity = max_entities;
        const size_t n = static_cast<size_t>(max_entities);
        perception_front_gap.assign(n, 1.0e9f);
        perception_front_speed.assign(n, 0.0f);
        perception_front_s.assign(n, 1.0e9f);
        perception_front_length.assign(n, 0.0f);
        perception_front_lane.assign(n, -1);
        perception_target_front_gap.assign(n, 1.0e9f);
        perception_target_front_speed.assign(n, 0.0f);
        perception_target_rear_gap.assign(n, 1.0e9f);
        perception_target_rear_speed.assign(n, 0.0f);
        decision_desired_speed.assign(n, 0.0f);
        decision_target_accel.assign(n, 0.0f);
        decision_wants_lane_change.assign(n, 0);
        decision_lane_change_target.assign(n, -1);
        decision_wants_connector.assign(n, 0);
        decision_connector_target_lane.assign(n, -1);
        decision_should_exit.assign(n, 0);
        active_ids.assign(n, 0);
        active_count.assign(1, 0);
        lane_active_ids.assign(n, 0);
        lane_active_count.assign(1, 0);
        connector_active_ids.assign(n, 0);
        connector_active_count.assign(1, 0);
        spawn_alloc_cursor.assign(1, 0);
    }
};

static CpuCudaParityTempBuffers g_cpu_cuda_parity_temp;
static std::mutex g_cpu_cuda_parity_step_mutex;

static ECSArrays make_ecs_arrays_cuda_parity(const Inputs& in) {
    ECSArrays ecs{};
    ecs.alive = in.active.data;
    ecs.x = in.x.data;
    ecs.y = in.y.data;
    ecs.s = in.s.data;
    ecs.speed = in.speed.data;
    ecs.accel = in.accel.data;
    ecs.heading = in.heading.data;
    ecs.steer_angle = in.steer_angle.data;
    ecs.length = in.vehicle_length.data;
    ecs.width = in.vehicle_width.data;
    ecs.driver_type = in.driver_type.data;
    ecs.reaction_time = in.reaction_time.data;
    ecs.min_gap = in.min_gap.data;
    ecs.aggressiveness = in.aggressiveness.data;
    ecs.politeness = in.politeness.data;
    ecs.risk_tolerance = in.risk_tolerance.data;
    ecs.comfort_decel = in.comfort_decel.data;
    ecs.desired_speed_factor = in.desired_speed_factor.data;
    ecs.lane_id = in.lane_id.data;
    ecs.route_id = in.route_id.data;
    ecs.route_pos = in.route_pos.data;
    ecs.entry_time = in.entry_time.data;
    ecs.vehicle_state = in.vehicle_state.data;
    ecs.connector_from_lane = in.connector_from_lane.data;
    ecs.connector_to_lane = in.connector_to_lane.data;
    ecs.connector_s = in.connector_s.data;
    ecs.connector_length = in.connector_length.data;
    ecs.lane_change_active = in.lane_change_active.data;
    ecs.lane_change_from_lane = in.lane_change_from_lane.data;
    ecs.lane_change_to_lane = in.lane_change_to_lane.data;
    ecs.lane_change_t = in.lane_change_t.data;
    ecs.lane_change_duration = in.lane_change_duration.data;
    ecs.lc_cooldown = in.lc_cooldown.data;
    ecs.turn_signal = in.turn_signal.data;
    ecs.turn_signal_time = in.turn_signal_time.data;
    return ecs;
}

static RoadNetwork make_road_cuda_parity(const Inputs& in, int num_routes) {
    RoadNetwork road{};
    road.lane_length = in.lane_length.data;
    road.lane_start_x = in.lane_start_x.data;
    road.lane_start_y = in.lane_start_y.data;
    road.lane_end_x = in.lane_end_x.data;
    road.lane_end_y = in.lane_end_y.data;
    road.lane_speed_limit = in.lane_speed_limit.data;
    road.lane_start_node = in.lane_start_node.data;
    road.lane_end_node = in.lane_end_node.data;
    road.left_lane = in.left_lane.data;
    road.right_lane = in.right_lane.data;
    road.route_offsets = in.route_offsets.data;
    road.route_lanes = in.route_lanes.data;
    road.route_turns = in.route_turns.data;
    road.num_lanes = static_cast<int>(in.num_lanes);
    road.num_nodes = static_cast<int>(in.num_nodes);
    road.num_routes = num_routes;
    return road;
}

static Signals make_signals_cuda_parity(const Inputs& in) {
    Signals signals{};
    signals.signal_node = in.signal_node.data;
    signals.signal_turn = in.signal_turn.data;
    signals.signal_cycle = in.signal_cycle.data;
    signals.signal_green_start = in.signal_green_start.data;
    signals.signal_green_end = in.signal_green_end.data;
    signals.signal_yellow_start = in.signal_yellow_start.data;
    signals.signal_yellow_end = in.signal_yellow_end.data;
    signals.num_signals = static_cast<int>(in.num_signals);
    return signals;
}

static PerceptionSoA make_perception_cuda_parity(CpuCudaParityTempBuffers& tmp) {
    PerceptionSoA p{};
    p.front_gap = tmp.perception_front_gap.data();
    p.front_speed = tmp.perception_front_speed.data();
    p.front_s = tmp.perception_front_s.data();
    p.front_length = tmp.perception_front_length.data();
    p.front_lane = tmp.perception_front_lane.data();
    p.target_front_gap = tmp.perception_target_front_gap.data();
    p.target_front_speed = tmp.perception_target_front_speed.data();
    p.target_rear_gap = tmp.perception_target_rear_gap.data();
    p.target_rear_speed = tmp.perception_target_rear_speed.data();
    return p;
}

static DecisionSoA make_decision_cuda_parity(CpuCudaParityTempBuffers& tmp) {
    DecisionSoA d{};
    d.desired_speed = tmp.decision_desired_speed.data();
    d.target_accel = tmp.decision_target_accel.data();
    d.wants_lane_change = tmp.decision_wants_lane_change.data();
    d.lane_change_target = tmp.decision_lane_change_target.data();
    d.wants_connector = tmp.decision_wants_connector.data();
    d.connector_target_lane = tmp.decision_connector_target_lane.data();
    d.should_exit = tmp.decision_should_exit.data();
    return d;
}

static SpawnConfig make_spawn_cuda_parity(const Inputs& in, CpuCudaParityTempBuffers& tmp) {
    SpawnConfig spawn{};
    spawn.spawn_accumulator = in.spawn_accumulator.data;
    spawn.demand_vps = in.demand_vps.data;
    spawn.demand_profile_vps = in.demand_profile_vps.data;
    spawn.demand_profile_has = in.demand_profile_has.data;
    spawn.spawn_lane = in.spawn_lane.data;
    spawn.spawn_route = in.spawn_route.data;
    spawn.spawn_alloc_cursor = tmp.spawn_alloc_cursor.data();
    spawn.num_spawn_points = static_cast<int>(in.num_spawn_points);
    spawn.demand_profile_slots = static_cast<int>(in.demand_profile_slots);
    spawn.demand_profile_slot_seconds = static_cast<float>(in.demand_profile_slot_seconds);
    spawn.av_penetration = static_cast<float>(in.av_penetration);
    return spawn;
}

static SpatialGrid make_grid_cuda_parity(const Inputs& in) {
    SpatialGrid grid{};
    grid.cell_head = in.world_cell_head.data;
    grid.cell_next = in.world_cell_next.data;
    grid.cell_epoch = nullptr;
    grid.epoch = 0;
    grid.lane_cell_head = in.lane_cell_head.numel > 0 ? in.lane_cell_head.data : nullptr;
    grid.lane_cell_next = in.lane_cell_next.numel > 0 ? in.lane_cell_next.data : nullptr;
    grid.lane_cells_per_lane = (in.num_lanes > 0 && in.lane_cell_head.numel > 0)
        ? static_cast<int>(in.lane_cell_head.numel / in.num_lanes)
        : 0;
    grid.min_x = static_cast<float>(in.world_min_x);
    grid.min_y = static_cast<float>(in.world_min_y);
    grid.cell_size = static_cast<float>(in.world_cell_size);
    grid.width = static_cast<int>(in.world_grid_w);
    grid.height = static_cast<int>(in.world_grid_h);
    return grid;
}

void step_cpu_impl(const Inputs& in) {
    std::lock_guard<std::mutex> guard(g_cpu_cuda_parity_step_mutex);

    const int ma = static_cast<int>(in.max_agents);
    const int nl = static_cast<int>(in.num_lanes);
    const int num_routes = static_cast<int>(in.route_offsets.numel) - 1;
    if (ma <= 0 || nl <= 0 || num_routes < 0) return;
    if (in.world_grid_w <= 0 || in.world_grid_h <= 0 || in.world_cell_size <= 0.0) return;

    CpuCudaParityTempBuffers& tmp = g_cpu_cuda_parity_temp;
    tmp.ensure(ma);

    ECSArrays ecs = make_ecs_arrays_cuda_parity(in);
    RoadNetwork road = make_road_cuda_parity(in, num_routes);
    Signals signals = make_signals_cuda_parity(in);
    PerceptionSoA perception = make_perception_cuda_parity(tmp);
    DecisionSoA decision = make_decision_cuda_parity(tmp);
    SpawnConfig spawn = make_spawn_cuda_parity(in, tmp);
    const int workers = resolved_thread_count();

    for (int64_t b = 0; b < in.batch_steps; ++b) {
        SpatialGrid grid = make_grid_cuda_parity(in);
        launch_step_cpu_ecs(ecs,
                            road,
                            signals,
                            grid,
                            spawn,
                            perception,
                            decision,
                            in.reservation_table.data,
                            reinterpret_cast<uint32_t*>(in.rng_state.data),
                            in.metrics.data,
                            tmp.active_ids.data(),
                            tmp.active_count.data(),
                            tmp.lane_active_ids.data(),
                            tmp.lane_active_count.data(),
                            tmp.connector_active_ids.data(),
                            tmp.connector_active_count.data(),
                            0,
                            static_cast<float>(in.current_time + static_cast<double>(b) * in.dt),
                            static_cast<float>(in.dt),
                            ma,
                            static_cast<int>(in.step_index + b),
                            workers);
    }
}

PyObject* py_step(PyObject*, PyObject* args) {
    Inputs in;
    if (!parse_inputs(args, false, &in)) return nullptr;
    Py_BEGIN_ALLOW_THREADS
    step_cpu_impl(in);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* py_step_batch(PyObject*, PyObject* args) {
    Inputs in;
    if (!parse_inputs(args, true, &in)) return nullptr;
    Py_BEGIN_ALLOW_THREADS
    step_cpu_impl(in);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* py_set_num_threads(PyObject*, PyObject* args) {
    long long n = 0;
    if (!PyArg_ParseTuple(args, "L", &n)) return nullptr;
    if (n < 0) n = 0;
    if (n > 1024) n = 1024;
    g_cpu_threads.store(static_cast<int>(n), std::memory_order_relaxed);
    Py_RETURN_NONE;
}

PyObject* py_get_num_threads(PyObject*, PyObject*) {
    return PyLong_FromLongLong(static_cast<long long>(resolved_thread_count()));
}

PyObject* py_backend_name(PyObject*, PyObject*) {
    return PyUnicode_FromString("cpu-cuda-source-parity");
}

PyObject* py_noop_none(PyObject*, PyObject*) {
    Py_RETURN_NONE;
}

PyObject* py_noop_bool(PyObject*, PyObject* args) {
    int enabled = 0;
    PyArg_ParseTuple(args, "p", &enabled);
    Py_RETURN_NONE;
}

PyObject* py_update_render_vbo(PyObject*, PyObject* args) {
    if (!PyTuple_Check(args) || PyTuple_Size(args) != 9) {
        PyErr_Format(PyExc_TypeError, "update_render_vbo expects 9 arguments, got %zd", PyTuple_Check(args) ? PyTuple_Size(args) : -1);
        return nullptr;
    }
    long long max_agents_ll = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 8));
    if (PyErr_Occurred()) return nullptr;
    TensorI active;
    if (!get_tensor_i(args, 4, "active", static_cast<int64_t>(max_agents_ll), &active)) return nullptr;
    int64_t n = 0;
    for (int64_t i = 0; i < active.numel; ++i) {
        if (active.data[i] == ENTITY_ALIVE) ++n;
    }
    return PyLong_FromLongLong(static_cast<long long>(n));
}

PyObject* py_update_render_vbo_interpolated(PyObject*, PyObject* args) {
    if (!PyTuple_Check(args) || PyTuple_Size(args) != 17) {
        PyErr_Format(PyExc_TypeError, "update_render_vbo_interpolated_full_draw expects 17 arguments, got %zd", PyTuple_Check(args) ? PyTuple_Size(args) : -1);
        return nullptr;
    }
    PyObject* reduced = PyTuple_New(9);
    if (!reduced) return nullptr;
    const int map[9] = {5, 6, 7, 8, 9, 10, 11, 12, 15};
    for (int i = 0; i < 9; ++i) {
        PyObject* item = PyTuple_GET_ITEM(args, map[i]);
        Py_INCREF(item);
        PyTuple_SET_ITEM(reduced, i, item);
    }
    PyObject* ret = py_update_render_vbo(nullptr, reduced);
    Py_DECREF(reduced);
    return ret;
}

static PyMethodDef Methods[] = {
    {"step", py_step, METH_VARARGS, "CPU-parallel ECS traffic ABM step"},
    {"step_batch", py_step_batch, METH_VARARGS, "CPU-parallel batched ECS traffic ABM steps"},
    {"set_num_threads", py_set_num_threads, METH_VARARGS, "Set CPU backend worker count; <=0 uses hardware concurrency"},
    {"get_num_threads", py_get_num_threads, METH_NOARGS, "Return resolved CPU backend worker count"},
    {"backend_name", py_backend_name, METH_NOARGS, "Return backend name"},
    {"register_render_vbo", py_noop_none, METH_VARARGS, "No-op placeholder for CUDA API compatibility"},
    {"set_vehicle_texture_render", py_noop_bool, METH_VARARGS, "No-op placeholder for CUDA API compatibility"},
    {"update_render_vbo", py_update_render_vbo, METH_VARARGS, "Return active draw count for CPU rendering fallback"},
    {"update_render_vbo_full_draw", py_update_render_vbo, METH_VARARGS, "Return active draw count for CPU rendering fallback"},
    {"update_render_vbo_interpolated_full_draw", py_update_render_vbo_interpolated, METH_VARARGS, "Return active draw count for CPU rendering fallback"},
    {"unregister_render_vbo", py_noop_none, METH_NOARGS, "No-op placeholder for CUDA API compatibility"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "avabm_cpu_ext",
    "Lightweight C++ CPU-parallel AVABM backend.",
    -1,
    Methods
};

}  // namespace

PyMODINIT_FUNC PyInit_avabm_cpu_ext(void) {
    return PyModule_Create(&ModuleDef);
}
