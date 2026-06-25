#pragma once
#include <cstdint>

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
    int cpu_workers);
