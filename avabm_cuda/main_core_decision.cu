#ifndef AVABM_PART_SKIP_COMMON
#include "main_common.cuh"
#endif
__global__ void decision_system_kernel(ECSArrays ecs, RoadNetwork road, Signals signals, SpatialGrid grid,
        PerceptionSoA perception, DecisionSoA decision, int* reservation_table, float* metrics, float current_time, float dt,
        int max_entities, const int* active_ids, const int* active_count) {
    AVABM_ACTIVE_LOOP_BEGIN(max_entities, active_ids, active_count) if (ecs.alive[i] != ENTITY_ALIVE) continue;
    int lane = ecs.lane_id[i];
    if (lane < 0 || lane >= road.num_lanes) {
        decision.should_exit[i] = 1;
        continue;
    }
    if (ecs.vehicle_state[i] == VEH_IN_CONNECTOR) {
        decision.target_accel[i] = 0.0f;
        continue;
    }
    int rid = ecs.route_id[i];
    int rpos = ecs.route_pos[i];
    if (rid < 0 || rid >= road.num_routes) {
        decision.should_exit[i] = 1;
        continue;
    }
    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    bool skip_initial_tail_repair = false;
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
    if (route_len > 0 && rpos >= 0 && rpos < route_len && rpos >= route_len - 1) {
        int tail_lane = road.route_lanes[ro0 + rpos];
        skip_initial_tail_repair = !route_lane_current_compatible_ecs(tail_lane, lane, road);
    }
#endif
    if (!skip_initial_tail_repair) {
        int repaired_pos = repair_route_pos_unless_missed_exit_tail_ecs(lane, rid, rpos, road);
        if (repaired_pos >= 0) {
            rpos = repaired_pos;
            ecs.route_pos[i] = repaired_pos;
        }
    }
    if (route_len <= 0 || rpos < 0 || rpos >= route_len) {
        decision.should_exit[i] = 1;
        continue;
    }
    int route_lane = road.route_lanes[ro0 + rpos];
    bool route_lane_incompatible_now = !route_lane_current_compatible_ecs(route_lane, lane, road);
    if (route_lane_incompatible_now) {
        int downstream_next = (rpos + 1 < route_len) ? road.route_lanes[ro0 + rpos + 1] : -1;
        int straight_recovery = missed_exit_straight_fallback_lane_ecs(i, lane, downstream_next, road);
        bool missed_exit_tail =
#if MISSED_EXIT_OFFROUTE_TAIL_ENABLED
        (rpos >= route_len - 1 && !valid_lane_ecs(downstream_next, road));
#else
        false;
#endif
        if ((!valid_lane_ecs(downstream_next, road) || !lane_connected(lane, downstream_next, road)) &&
                !valid_lane_ecs(straight_recovery, road) && !missed_exit_tail) {
            decision.should_exit[i] = 1;
            continue;
        }
    }
    int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
    rpos = ecs.route_pos[i];
    int turn = route_turn_for_vehicle_ecs(i, ecs, road);
    bool has_next = next_lane >= 0 && next_lane < road.num_lanes && lane_connected(lane, next_lane, road);
    bool missed_exit_straight_target = false;
    if (has_next) {
        missed_exit_straight_target = missed_exit_straight_target_ecs(i, lane, next_lane, ecs, road);
        if (missed_exit_straight_target) {
            turn = TURN_STRAIGHT;
        } else {
            turn = effective_turn_code_ecs(lane, next_lane, turn, road);
            int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
            adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
            adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
            if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
                next_lane = adjusted_next;
            }
        }
    }
    int interchange_source_lane = (has_next &&
            !missed_exit_straight_target) ? interchange_source_outer_lane_ecs(lane, next_lane, road) : -1;
    bool human = ecs.driver_type[i] == HUMAN;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    float v = ecs.speed[i];
    float s_curr = ecs.s[i];
    float desired_v = desired_speed_ecs(i, lane, ecs, road);
    float curr_L = fmaxf(road.lane_length[lane], 0.1f);
    float dist_to_end = curr_L - s_curr;
    if (!has_next && dist_to_end <= DEFAULT_STOP_OFFSET + 0.75f) {
        decision.should_exit[i] = 1;
        continue;
    }
    bool interchange_needs_outer_lane = has_next && valid_lane_ecs(interchange_source_lane, road) &&
            lane != interchange_source_lane;
    bool ordinary_turn_needs_dedicated_lane = has_next && turn_requires_dedicated_lane_ecs(turn);
    bool turn_needs_dedicated_lane = ordinary_turn_needs_dedicated_lane || interchange_needs_outer_lane;
    bool turn_lane_ok = true;
    if (interchange_needs_outer_lane) {
        turn_lane_ok = false;
    } else if (ordinary_turn_needs_dedicated_lane) {
        turn_lane_ok = lane_legal_for_turn_ecs(lane, turn, road);
    }
    int turn_lc_target = -1;
    if (!turn_lane_ok) {
        if (interchange_needs_outer_lane) {
            turn_lc_target = adjacent_lane_toward_specific_lane_ecs(lane, interchange_source_lane, road);
        } else {
            turn_lc_target = adjacent_lane_toward_turn_lane_ecs(lane, turn, road);
        }
    }
    int turn_lane_steps = 0;
    if (interchange_needs_outer_lane) {
        turn_lane_steps = lane_steps_to_specific_lane_ecs(lane, interchange_source_lane, road);
    } else if (ordinary_turn_needs_dedicated_lane) {
        turn_lane_steps = lane_steps_to_turn_lane_ecs(lane, turn, road);
    }
    float turn_prep_dist = turn_lane_prep_distance_ecs(turn_lane_steps >= 99 ? 4 : turn_lane_steps, v, ecs.driver_type[i]);
    float lc_no_start_dist = lane_change_no_start_distance_ecs(lane, road);
    float lc_deadline_dist = dist_to_end - lc_no_start_dist;
    float lc_duration_nominal = human ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV;
    bool mandatory_lc_pending = turn_needs_dedicated_lane && !turn_lane_ok && turn_lc_target >= 0 &&
            turn_lc_target < road.num_lanes;
    if (mandatory_lc_pending) {
        bool too_late_to_start_lc = dist_to_end <= fmaxf(lc_no_start_dist, MISSED_TURN_ESCAPE_DIST);
        float front_open = fmaxf(SMART_STALL_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 8.0f);
        float w = ecs.connector_length[i];
        if (!isfinite(w) || w < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) w = 0.0f;
        bool waiting_for_mandatory_gap = dist_to_end <= fmaxf(turn_prep_dist, lc_no_start_dist + 24.0f) &&
                ecs.lane_change_active[i] == 0 && (ecs.speed[i] < 1.2f || perception.front_gap[i] > front_open);
        if (waiting_for_mandatory_gap) {
            w = fminf(w + dt, 30.0f);
        } else if (dist_to_end > turn_prep_dist + 12.0f) {
            w = 0.0f;
        }
        ecs.connector_length[i] = w;
        bool clear_front_for_abandon = perception.front_gap[i] > fmaxf(front_open, MISSION_ABANDON_FRONT_GAP);
        bool target_lane_gap_looks_closed = perception.target_front_gap[i] < fmaxf(9.5f,
                ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.55f) ||
                perception.target_rear_gap[i] < fmaxf(10.5f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f,
                perception.target_rear_speed[i]) * 0.65f);
        bool waited_out_missed_exit = w >= MISSION_ABANDON_WAIT && clear_front_for_abandon && (too_late_to_start_lc ||
                dist_to_end <= fmaxf(lc_no_start_dist + MISSION_ABANDON_NO_START_EXTRA,
                turn_prep_dist * MISSION_ABANDON_PREP_RATIO) || (ecs.speed[i] <= MISSION_ABANDON_SPEED_EPS &&
                target_lane_gap_looks_closed));
        bool early_gridlock_abandon = w >= MISSION_ABANDON_WAIT && clear_front_for_abandon &&
                ecs.speed[i] <= MISSION_ABANDON_SPEED_EPS && dist_to_end <= turn_prep_dist + MISSION_ABANDON_EARLY_EXTRA;
        if (too_late_to_start_lc || waited_out_missed_exit || early_gridlock_abandon) {
            int escaped_next = -1;
            bool escaped_has_next = false;
            bool escaped_missed_exit = false;
            bool abandoned = abandon_destination_to_straight_or_tail_ecs(i, lane, next_lane, ecs, road, escaped_next,
                    escaped_has_next, escaped_missed_exit);
            if (abandoned) {
                next_lane = escaped_next;
                has_next = escaped_has_next;
                missed_exit_straight_target = escaped_missed_exit;
                turn = TURN_STRAIGHT;
                interchange_source_lane = -1;
                interchange_needs_outer_lane = false;
                ordinary_turn_needs_dedicated_lane = false;
                turn_needs_dedicated_lane = false;
                turn_lane_ok = true;
                turn_lc_target = -1;
                turn_lane_steps = 0;
                turn_prep_dist = 0.0f;
                mandatory_lc_pending = false;
                desired_v = fmaxf(desired_v, human ? MISSION_ABANDON_RELEASE_SPEED_HUMAN : MISSION_ABANDON_RELEASE_SPEED_AV);
                if (escaped_has_next) {
                    ecs.connector_length[i] = fmaxf(ecs.connector_length[i], MISSION_ABANDON_WAIT + dt);
                } else {
                    ecs.connector_length[i] = 0.0f;
                }
                if (ecs.turn_signal != nullptr) {
                    ecs.turn_signal[i] = INDICATOR_NONE;
                    if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = 0.0f;
                }
                AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_ESCAPE_GO, 1.0f);
            } else if (w >= MISSED_TURN_ESCAPE_WAIT && clear_front_for_abandon) {
                turn_lane_ok = true;
                mandatory_lc_pending = false;
                desired_v = fmaxf(desired_v, human ? MISSION_ABANDON_RELEASE_SPEED_HUMAN : MISSION_ABANDON_RELEASE_SPEED_AV);
                AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_BLOCK, 1.0f);
            }
        }
    }
    if (ecs.lane_change_active[i] != 0) {
        float active_front_req_pre = fmaxf(18.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.90f);
        float active_target_front_req_pre = fmaxf(8.5f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.40f);
        float active_target_rear_req_pre = fmaxf(9.5f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f,
                perception.target_rear_speed[i]) * 0.55f);
        bool active_lc_open_pre = perception.front_gap[i] > active_front_req_pre &&
                perception.target_front_gap[i] > active_target_front_req_pre &&
                perception.target_rear_gap[i] > active_target_rear_req_pre;
        if (!active_lc_open_pre) {
            desired_v = fminf(desired_v, fmaxf(human ? LANE_CHANGE_PREP_MIN_CAP_HUMAN : LANE_CHANGE_PREP_MIN_CAP_AV,
                    desired_v * LANE_CHANGE_ACTIVE_SPEED_CAP_SCALE));
        }
    }
    if (next_lane >= 0 && next_lane < road.num_lanes) {
        float ctrl_dist = human ? TURN_CONTROL_DIST_HUMAN : TURN_CONTROL_DIST_AV;
        if (dist_to_end < ctrl_dist) {
            float angle = turn_angle_deg(lane, next_lane, road);
            bool route_straight_bend_open = turn == TURN_STRAIGHT && angle <= 38.0f &&
                    perception.front_gap[i] > fmaxf(24.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.05f);
            if ((angle > STRAIGHT_NO_TURN_SPEED_CAP_DEG || turn != TURN_STRAIGHT) && !route_straight_bend_open) {
                desired_v = fminf(desired_v, turn_speed_cap(angle, ecs.driver_type[i]));
            }
        }
    }
    if (turn_needs_dedicated_lane && !turn_lane_ok && dist_to_end < turn_prep_dist) {
        float hard_prep_zone = fmaxf(lc_no_start_dist + LANE_CHANGE_PREP_HARD_DIST,
                TURN_LANE_HARD_HOLD_DIST + DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER);
        bool prep_front_blocked = perception.front_gap[i] < fmaxf(18.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.90f);
        if (dist_to_end < hard_prep_zone || prep_front_blocked) {
            float wrong_lane_cap = human ? TURN_LANE_WRONG_LANE_SPEED_HUMAN : TURN_LANE_WRONG_LANE_SPEED_AV;
            if (dist_to_end < TURN_LANE_HARD_HOLD_DIST + DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER) {
                wrong_lane_cap = fminf(wrong_lane_cap, human ? 1.8f : 2.4f);
            } else if (!prep_front_blocked) {
                wrong_lane_cap = fmaxf(wrong_lane_cap, human ? 8.6f : 10.2f);
            }
            desired_v = fminf(desired_v, wrong_lane_cap);
        }
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_PREP, 1.0f);
    }
    float acc_cmd = estimate_follow_accel_ecs(v, desired_v, perception.front_gap[i], perception.front_speed[i], ecs.driver_type[i],
            ecs.min_gap[i], ecs.reaction_time[i], ecs.comfort_decel[i], ecs.aggressiveness[i], ecs.risk_tolerance[i]);
    if (perception.front_gap[i] < 1.0e8f) {
        float safe = (human ? SAFE_GAP_HUMAN : SAFE_GAP_AV) + v * (human ? SAFE_TIME_HEADWAY_HUMAN : SAFE_TIME_HEADWAY_AV);
        if (perception.front_gap[i] < safe) {
            float ratio = safe / fmaxf(perception.front_gap[i], 0.5f);
            acc_cmd = fminf(acc_cmd, -max_decel * ratio * ratio);
        }
    }
    if (mandatory_lc_pending && dist_to_end < turn_prep_dist) {
        float pressure = clampf_cuda((turn_prep_dist - dist_to_end) / fmaxf(turn_prep_dist, 1.0f), 0.0f, 1.0f);
        float deadline = fmaxf(0.0f, lc_deadline_dist);
        bool hard_deadline = deadline < LANE_CHANGE_PREP_HARD_DIST;
        float prep_front_gap = perception.target_front_gap[i];
        float prep_front_speed = perception.target_front_speed[i];
        float prep_rear_gap = perception.target_rear_gap[i];
        float prep_rear_speed = perception.target_rear_speed[i];
        if (valid_lane_ecs(turn_lc_target, road) && same_approach_same_direction_lanes_ecs(lane, turn_lc_target, road)) {
            find_lane_neighbors_ecs(i, turn_lc_target, ecs, road, grid, max_entities, human ? 155.0f : 135.0f, nullptr,
                    prep_front_gap, prep_front_speed, prep_rear_gap, prep_rear_speed);
        }
        float target_front_req = fmaxf(9.5f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * LC_PREP_TARGET_FRONT_TIME);
        float target_rear_req = fmaxf(10.5f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f,
                prep_rear_speed) * LC_PREP_TARGET_REAR_TIME);
        bool target_gap_unsafe = prep_front_gap < target_front_req || prep_rear_gap < target_rear_req;
        bool current_front_open = perception.front_gap[i] > fmaxf(LC_PREP_OPEN_FRONT_RELAX_GAP,
                ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.15f);
        float min_cap = human ? LANE_CHANGE_PREP_MIN_CAP_HUMAN : LANE_CHANGE_PREP_MIN_CAP_AV;
        float time_budget = lc_duration_nominal * (1.08f + 0.32f * pressure);
        float prep_cap = deadline / fmaxf(time_budget, 0.35f);
        prep_cap = clampf_cuda(prep_cap, min_cap, desired_v);
        if (hard_deadline) {
            prep_cap = fminf(prep_cap, min_cap + 0.65f * deadline / fmaxf(LANE_CHANGE_PREP_HARD_DIST, 1.0f));
        } else if (current_front_open) {
            prep_cap = fmaxf(prep_cap, fmaxf(min_cap, v - 0.45f));
        }
        bool needs_prep_limit = hard_deadline || target_gap_unsafe || !current_front_open;
        if (needs_prep_limit) {
            desired_v = fminf(desired_v, prep_cap);
            float prep_a = (prep_cap - v) / fmaxf(dt, 0.01f);
            float accel_cap = LC_PREP_COAST_ACCEL_LIMIT;
            if (current_front_open && !hard_deadline) {
                accel_cap = fminf(fmaxf(LC_PREP_COAST_ACCEL_LIMIT, max_accel * 0.42f), max_accel * 0.70f);
            }
            if (prep_a > accel_cap) {
                prep_a = accel_cap;
            }
            float max_prep_brake = human ? LC_PREP_MAX_BRAKE_HUMAN : LC_PREP_MAX_BRAKE_AV;
            if (hard_deadline) {
                max_prep_brake = fminf(max_decel, max_prep_brake * 1.55f);
            } else if (current_front_open) {
                max_prep_brake = fminf(max_prep_brake, 0.16f);
            }
            prep_a = clampf_cuda(prep_a, -max_prep_brake, accel_cap);
            acc_cmd = fminf(acc_cmd, prep_a);
        } else {
            float open_accel_cap = fminf(max_accel * 0.72f, fmaxf(LC_PREP_COAST_ACCEL_LIMIT, 1.10f));
            acc_cmd = fminf(acc_cmd, open_accel_cap);
        }
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_PREP, 1.0f);
    }
    if (ecs.lane_change_active[i] != 0) {
        float active_front_req = fmaxf(18.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.95f);
        float active_target_front_req = fmaxf(9.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.45f);
        float active_target_rear_req = fmaxf(10.0f, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f,
                perception.target_rear_speed[i]) * 0.58f);
        bool active_lc_open = perception.front_gap[i] > active_front_req &&
                perception.target_front_gap[i] > active_target_front_req && perception.target_rear_gap[i] > active_target_rear_req;
        float active_accel_cap = active_lc_open ? fminf(max_accel * 0.72f, fmaxf(LC_PREP_COAST_ACCEL_LIMIT,
                1.25f)) : LC_PREP_COAST_ACCEL_LIMIT;
        acc_cmd = fminf(acc_cmd, active_accel_cap);
    }
    float courtesy_boost = 0.0f;
    bool courtesy_assertive = false;
    float courtesy_limit = lane_change_courtesy_accel_limit_ecs(i, lane, ecs, road, grid, current_time, max_entities,
            &courtesy_boost, &courtesy_assertive);
    if (courtesy_limit < 999.0f) {
        acc_cmd = fminf(acc_cmd, courtesy_limit);
        AVABM_METRIC_ADD(metrics, METRIC_HUMAN_AI_COURTESY_YIELD, 1.0f);
    }
    if (courtesy_boost > 0.0f && perception.front_gap[i] > fmaxf(ecs.length[i] + MIN_BUMPER_GAP + 6.0f, 12.0f)) {
        acc_cmd = fmaxf(acc_cmd, fminf(courtesy_boost, max_accel * 0.65f));
        if (courtesy_assertive) AVABM_METRIC_ADD(metrics, METRIC_HUMAN_AI_ASSERTIVE_GO, 1.0f);
    }
    int sig_state = LIGHT_GREEN;
    float stop_s = 1.0e9f;
    bool self_inside_intersection_box = has_next && turn_lane_ok && inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
    float sig_acc = signal_accel_limit_ecs(lane, turn, s_curr, v, ecs.reaction_time[i], ecs.driver_type[i], current_time, road,
            signals, &sig_state, &stop_s);
    if (self_inside_intersection_box) {
        sig_acc = 1000.0f;
    }
    acc_cmd = fminf(acc_cmd, sig_acc);
    if (self_inside_intersection_box && perception.front_gap[i] > fmaxf(ecs.length[i] + MIN_BUMPER_GAP + 3.0f, 7.0f)) {
        float clear_v = human ? 3.1f : 4.2f;
        float clear_a = (clear_v - v) / fmaxf(dt, 0.01f);
        clear_a = clampf_cuda(clear_a, 0.0f, max_accel * 0.55f);
        acc_cmd = fmaxf(acc_cmd, clear_a);
    }
    int conflict_node = has_next ? road.lane_end_node[lane] : -1;
    bool signalized_node = has_next && node_has_signal_ecs(conflict_node, signals);
    bool unsignal_node = has_next && !signalized_node;
    bool unsignal_blocked = false;
    bool unsignal_deadlock_release = false;
    bool unsignal_conflict_seen = false;
    if (!has_next && dist_to_end < 1.0f) {
        decision.should_exit[i] = 1;
    }
    bool signal_stop_command = sig_acc < -0.01f && stop_s < 1.0e8f;
    if (sig_state == LIGHT_RED) {
        if (signal_stop_command) AVABM_METRIC_ADD(metrics, METRIC_RED_LIGHT_STOP, 1.0f);
        if (stop_s < 1.0e8f && s_curr > stop_s + 0.25f && v > 0.2f) {
            AVABM_METRIC_ADD(metrics, METRIC_RED_LIGHT_VIOLATION, 1.0f);
        }
    } else if (sig_state == LIGHT_YELLOW) {
        if (signal_stop_command) AVABM_METRIC_ADD(metrics, METRIC_YELLOW_STOP, 1.0f);
        else AVABM_METRIC_ADD(metrics, METRIC_YELLOW_GO, 1.0f);
    }
    if (signal_stop_command && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        AVABM_METRIC_ADD(metrics, METRIC_INTERSECTION_WAIT, dt);
    }
    float interaction_limit = 1000.0f;
    if (has_next && turn_lane_ok && !self_inside_intersection_box && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        interaction_limit = interaction_accel_limit_ecs(i, lane, next_lane, ecs, road, grid, max_entities, nullptr);
        if (interaction_limit < 999.0f) {
            AVABM_METRIC_ADD(metrics, METRIC_INTERACTION_BRAKE, 1.0f);
        }
    }
    acc_cmd = fminf(acc_cmd, interaction_limit);
    bool reservation_granted = true;
    bool directional_conflict_present = false;
    if (has_next && turn_lane_ok && !self_inside_intersection_box && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        if (unsignal_node) {
            float unsignal_limit = unsignal_priority_accel_limit_ecs(i, lane, next_lane, dist_to_end, perception.front_gap[i], ecs,
                    road, grid, max_entities, current_time, dt, metrics, &unsignal_blocked, &unsignal_deadlock_release,
                    &unsignal_conflict_seen);
            directional_conflict_present = unsignal_conflict_seen;
            if (unsignal_blocked && !unsignal_deadlock_release) {
                acc_cmd = fminf(acc_cmd, unsignal_limit);
                AVABM_METRIC_ADD(metrics, METRIC_INTERSECTION_WAIT, dt);
                AVABM_METRIC_ADD(metrics, METRIC_CONFLICT_YIELD, 1.0f);
                AVABM_METRIC_ADD(metrics, METRIC_COOP_YIELD, 1.0f);
            } else if (unsignal_deadlock_release) {
                float release_v = human ? DEADLOCK_RELEASE_CREEP_HUMAN : DEADLOCK_RELEASE_CREEP_AV;
                float release_a = (release_v - v) / fmaxf(dt, 0.01f);
                release_a = clampf_cuda(release_a, 0.0f, max_accel * 0.38f);
                acc_cmd = fmaxf(acc_cmd, release_a);
                acc_cmd = fminf(acc_cmd, max_accel * 0.38f);
                AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_CREEP, 1.0f);
            }
        } else {
            float conflict_limit = intersection_conflict_accel_limit_ecs(i, lane, next_lane, dist_to_end, ecs, road, grid,
                    max_entities, current_time, nullptr);
            if (conflict_limit < 999.0f) {
                directional_conflict_present = true;
            }
            if (conflict_limit < acc_cmd) {
                acc_cmd = conflict_limit;
                AVABM_METRIC_ADD(metrics, METRIC_INTERSECTION_WAIT, dt);
                AVABM_METRIC_ADD(metrics, METRIC_CONFLICT_YIELD, 1.0f);
                AVABM_METRIC_ADD(metrics, METRIC_COOP_YIELD, 1.0f);
            }
        }
    } else if (ecs.vehicle_state[i] == VEH_ON_LANE) {
        ecs.connector_length[i] = fmaxf(0.0f, ecs.connector_length[i] - dt);
    }
    if (has_next && turn_lane_ok && !unsignal_node && !self_inside_intersection_box && directional_conflict_present &&
            dist_to_end < 30.0f) {
        int node = road.lane_end_node[lane];
        float arrival_time = current_time + dist_to_end / fmaxf(v, 1.0f);
        float angle = turn_angle_deg(lane, next_lane, road);
        float crossing_time = 1.6f;
        if (angle > 45.0f) crossing_time += 0.8f;
        if (angle > 100.0f) crossing_time += 1.0f;
        crossing_time += human ? 0.45f : 0.20f;
        bool reserved = try_reserve_slot_ecs(i, node, arrival_time, crossing_time, reservation_table, road.num_nodes, current_time,
                metrics);
        reservation_granted = reserved;
        if (!reserved) {
            float crawl_v = human ? YIELD_CREEP_SPEED_HUMAN : YIELD_CREEP_SPEED_AV;
            float a = (crawl_v - v) / fmaxf(dt, 0.01f);
            float no_front_gap = fmaxf(STALL_RECOVERY_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 5.0f);
            bool can_creep_to_line = dist_to_end > DEFAULT_STOP_OFFSET + SIGNAL_CREEP_HOLD_DIST &&
                    perception.front_gap[i] > no_front_gap;
            if (can_creep_to_line && v < crawl_v) {
                float creep_a = clampf_cuda(a, 0.0f, max_accel * 0.45f);
                acc_cmd = fmaxf(acc_cmd, creep_a);
                acc_cmd = fminf(acc_cmd, max_accel * 0.45f);
            } else {
                acc_cmd = fminf(acc_cmd, clampf_cuda(a, -max_decel, 0.0f));
            }
            AVABM_METRIC_ADD(metrics, METRIC_COOP_YIELD, 1.0f);
            AVABM_METRIC_ADD(metrics, METRIC_INTERSECTION_WAIT, dt);
        }
    }
    float stop_line_dist = stop_s < 1.0e8f ? stop_s - s_curr : dist_to_end - DEFAULT_STOP_OFFSET;
    float rt = fmaxf(ecs.reaction_time[i], human ? 0.75f : 0.12f);
    float comfortable_b = human ? (0.78f * MAX_DECEL_HUMAN) : (0.86f * MAX_DECEL_AV);
    comfortable_b = fmaxf(comfortable_b, 1.2f);
    float comfortable_stop = v * rt + (v * v) / fmaxf(2.0f * comfortable_b, 0.1f) + INTERSECTION_STOP_BUFFER;
    bool yellow_committed = sig_state == LIGHT_YELLOW && stop_line_dist <= comfortable_stop * 0.72f &&
            dist_to_end <= fmaxf(6.0f, v * dt + 2.0f);
    bool signal_permits_connector = sig_state == LIGHT_GREEN || yellow_committed;
    if (sig_state == LIGHT_RED) signal_permits_connector = false;
    if (self_inside_intersection_box) signal_permits_connector = true;
    if (sig_state != LIGHT_GREEN && !self_inside_intersection_box && stop_s < 1.0e8f && dist_to_end > 0.0f) {
        float predicted_s = s_curr + fmaxf(0.0f, v + acc_cmd * dt) * dt;
        if (predicted_s > stop_s - 0.15f && !yellow_committed) {
            acc_cmd = -EMERGENCY_DECEL;
        }
    }
    if (has_next && !self_inside_intersection_box && !signal_permits_connector && dist_to_end < 8.0f) {
        float stop_dist = fmaxf(dist_to_end - DEFAULT_STOP_OFFSET, 0.55f);
        float req = -(v * v) / fmaxf(2.0f * stop_dist, 0.5f);
        acc_cmd = fminf(acc_cmd, clampf_cuda(req, -EMERGENCY_DECEL, 0.0f));
    }
    if (has_next && turn_needs_dedicated_lane && !turn_lane_ok) {
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_ILLEGAL, 1.0f);
        bool no_adjacent_turn_lane = turn_lc_target < 0 || turn_lc_target >= road.num_lanes;
        bool too_close_for_lane_change = dist_to_end <= fmaxf(TURN_LANE_MIN_LC_DIST, fmaxf(1.0f,
                v) * (human ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV) * 0.45f + DEFAULT_STOP_OFFSET +
                        TURN_LANE_STOP_BUFFER);
        if (no_adjacent_turn_lane || too_close_for_lane_change) {
            int escaped_next = -1;
            bool escaped_has_next = false;
            bool escaped_missed_exit = false;
            bool abandoned = abandon_destination_to_straight_or_tail_ecs(i, lane, next_lane, ecs, road, escaped_next,
                    escaped_has_next, escaped_missed_exit);
            if (abandoned) {
                next_lane = escaped_next;
                has_next = escaped_has_next;
                missed_exit_straight_target = escaped_missed_exit;
                turn = TURN_STRAIGHT;
                turn_needs_dedicated_lane = false;
                turn_lane_ok = true;
                mandatory_lc_pending = false;
                acc_cmd = fmaxf(acc_cmd, max_accel * MISSION_ABANDON_ACCEL_SCALE);
                AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_ESCAPE_GO, 1.0f);
            } else {
                acc_cmd = fminf(acc_cmd, turn_lane_hold_accel_ecs(dist_to_end, v, ecs.driver_type[i]));
                AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_BLOCK, 1.0f);
            }
        }
    }
    bool signal_requires_stop = sig_state == LIGHT_RED || (sig_state == LIGHT_YELLOW && !yellow_committed);
    float stop_line_gap = stop_s < 1.0e8f ? stop_s - s_curr : dist_to_end - DEFAULT_STOP_OFFSET;
    bool hard_signal_hold = signal_requires_stop && stop_line_gap <= SIGNAL_CREEP_HOLD_DIST;
    bool front_clear_for_creep = perception.front_gap[i] > fmaxf(STALL_RECOVERY_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 5.0f);
    bool hard_unsignal_hold = unsignal_blocked && !unsignal_deadlock_release && !front_clear_for_creep &&
            dist_to_end <= DEFAULT_STOP_OFFSET + UNSIGNAL_PRIORITY_NEAR_LINE_DIST &&
            ecs.connector_length[i] < (human ? 0.65f : 0.45f);
    if (has_next && v < 0.32f && acc_cmd <= 0.02f && front_clear_for_creep && !hard_signal_hold && !hard_unsignal_hold &&
            !(turn_needs_dedicated_lane && !turn_lane_ok) && dist_to_end > STALL_RECOVERY_MIN_END_DIST) {
        float creep_v = unsignal_deadlock_release ? (human ? DEADLOCK_RELEASE_CREEP_HUMAN : DEADLOCK_RELEASE_CREEP_AV) :
                (signal_requires_stop ? (human ? SIGNAL_CREEP_SPEED_HUMAN : SIGNAL_CREEP_SPEED_AV) : (human ?
                YIELD_CREEP_SPEED_HUMAN : YIELD_CREEP_SPEED_AV));
        float creep_a = (creep_v - v) / fmaxf(dt, 0.01f);
        if (creep_a > 0.0f) {
            creep_a = clampf_cuda(creep_a, 0.0f, max_accel * 0.40f);
            acc_cmd = fmaxf(acc_cmd, creep_a);
            acc_cmd = fminf(acc_cmd, max_accel * 0.40f);
            AVABM_METRIC_ADD(metrics, METRIC_QUEUE_DELAY_SUM, dt);
            AVABM_METRIC_ADD(metrics, METRIC_QUEUE_DELAY_COUNT, 1.0f);
        }
    }
    bool smart_front_clear = perception.front_gap[i] > fmaxf(SMART_STALL_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f,
            v) * 0.85f);
    bool smart_red_hold = signal_requires_stop && !self_inside_intersection_box && stop_line_gap <= SIGNAL_CREEP_HOLD_DIST + 1.0f;
    bool smart_wrong_lane_hold = turn_needs_dedicated_lane && !turn_lane_ok;
    float smart_wait_time = ecs.connector_length[i];
    if (!isfinite(smart_wait_time) || smart_wait_time < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) {
        smart_wait_time = 0.0f;
    }
    if (has_next && v < SMART_STALL_SPEED && acc_cmd <= 0.04f && smart_front_clear && !smart_red_hold && !smart_wrong_lane_hold &&
            (self_inside_intersection_box || dist_to_end <= DEFAULT_STOP_OFFSET + PRIORITY_GATE_NEAR_LINE_DIST + 2.0f ||
            smart_wait_time >= SMART_STALL_RELEASE_WAIT)) {
        if (unsignal_blocked && smart_wait_time >= SMART_STALL_RELEASE_WAIT && smart_front_clear) {
            unsignal_deadlock_release = true;
        }
        float smart_v = self_inside_intersection_box ? (human ? CONNECTOR_INBOX_MIN_CLEAR_SPEED_HUMAN :
                CONNECTOR_INBOX_MIN_CLEAR_SPEED_AV) : (human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV);
        float smart_a = (smart_v - v) / fmaxf(dt, 0.01f);
        smart_a = clampf_cuda(smart_a, 0.0f, max_accel * SMART_STALL_RELEASE_ACCEL_SCALE);
        if (smart_a > 0.0f) {
            acc_cmd = fmaxf(acc_cmd, smart_a);
            acc_cmd = fminf(acc_cmd, max_accel * SMART_STALL_RELEASE_ACCEL_SCALE);
            AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_CREEP, 1.0f);
        }
    }
#if MIDROAD_NEG_ACCEL_WATCHDOG_ENABLED
    {
        float stale_clear_gap = fmaxf(MIDROAD_NEG_ACCEL_CLEAR_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.25f);
        bool midroad_clear_brake = v <= MIDROAD_NEG_ACCEL_CLEAR_SPEED && acc_cmd < -0.05f &&
                perception.front_gap[i] > stale_clear_gap && dist_to_end > MIDROAD_NEG_ACCEL_CLEAR_MIN_END_DIST &&
                !hard_signal_hold && !hard_unsignal_hold && !smart_wrong_lane_hold && !(turn_needs_dedicated_lane && !turn_lane_ok);
        if (midroad_clear_brake) {
            float release_v = human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV;
            float release_a = (release_v - v) / fmaxf(dt, 0.01f);
            release_a = clampf_cuda(release_a, 0.0f, max_accel * MIDROAD_NEG_ACCEL_RELEASE_SCALE);
            acc_cmd = fmaxf(acc_cmd, release_a);
            if (ecs.accel[i] < 0.0f) ecs.accel[i] = 0.0f;
            if (metrics != nullptr) AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_CREEP, 1.0f);
        }
    }
#endif
#if MIDROAD_ZERO_ACCEL_WATCHDOG_ENABLED
    {
        float zero_clear_gap = fmaxf(MIDROAD_ZERO_ACCEL_CLEAR_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.10f);
        bool midroad_zero_hold = has_next && v <= MIDROAD_ZERO_ACCEL_CLEAR_SPEED && acc_cmd >= -0.04f && acc_cmd <= 0.035f &&
                perception.front_gap[i] > zero_clear_gap && dist_to_end > MIDROAD_ZERO_ACCEL_MIN_END_DIST && !hard_signal_hold &&
                !hard_unsignal_hold && !smart_wrong_lane_hold && !(turn_needs_dedicated_lane && !turn_lane_ok);
        if (midroad_zero_hold) {
            float release_v = human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV;
            float release_a = (release_v - v) / fmaxf(dt, 0.01f);
            release_a = clampf_cuda(release_a, 0.0f, max_accel * MIDROAD_ZERO_ACCEL_RELEASE_SCALE);
            if (release_a > 0.0f) {
                acc_cmd = fmaxf(acc_cmd, release_a);
                if (ecs.accel[i] < 0.0f) ecs.accel[i] = 0.0f;
                if (metrics != nullptr) AVABM_METRIC_ADD(metrics, METRIC_DEADLOCK_CREEP, 1.0f);
            }
        }
    }
#endif
    int ll = geometric_left_neighbor_ecs(lane, road);
    int rr = geometric_right_neighbor_ecs(lane, road);
    int upcoming_event_turn = TURN_STRAIGHT;
    float upcoming_event_dist = 1.0e9f;
    int lookahead_lc_target = -1;
    if (ecs.lane_change_active[i] == 0 && has_next && !turn_needs_dedicated_lane) {
        lookahead_lc_target = upcoming_exit_lane_step_target_ecs(i, lane, ecs, road, dist_to_end, v, ecs.driver_type[i],
                &upcoming_event_turn, &upcoming_event_dist);
    }
    int lane_drop_lc_target = -1;
#if LANE_COUNT_CHANGE_PREP_ENABLED
    if (ecs.lane_change_active[i] == 0 && has_next && turn == TURN_STRAIGHT && !turn_needs_dedicated_lane) {
        int prep_target = lane_count_reduction_step_target_ecs(lane, next_lane, road);
        if (valid_lane_ecs(prep_target, road)) {
            int cur_idx = -1;
            int from_count = lane_group_count_and_index_ecs(lane, road, cur_idx);
            int to_count = lane_group_count_ecs(next_lane, road);
            int dropped = max(1, from_count - to_count);
            float prep_dist = LANE_COUNT_CHANGE_PREP_MIN_DIST + LANE_COUNT_CHANGE_PREP_PER_DROPPED * (float)dropped + fmaxf(0.0f,
                    v) * 2.2f;
            prep_dist = clampf_cuda(prep_dist, LANE_COUNT_CHANGE_PREP_MIN_DIST, LANE_COUNT_CHANGE_PREP_MAX_DIST);
            if (from_count > to_count && dist_to_end <= prep_dist &&
                    dist_to_end > fmaxf(lc_no_start_dist, LANE_COUNT_CHANGE_PREP_MIN_DIST)) {
                lane_drop_lc_target = prep_target;
            }
        }
    }
#endif
    int balance_cur_idx = -1;
    int balance_group_count = lane_group_count_and_index_ecs(lane, road, balance_cur_idx);
    bool right_edge_balance_override = balance_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP && balance_cur_idx >= 0 &&
            balance_cur_idx <= THROUGH_LANE_BALANCE_FORCE_RIGHT_IDX && upcoming_event_dist > RIGHT_EDGE_FORCE_INNER_MIN_EXIT_DIST;
    int lc_target = -1;
    bool mandatory_lc = false;
    bool opportunistic_lc = false;
    if (ecs.lane_change_active[i] == 0 && turn_needs_dedicated_lane && !turn_lane_ok && turn_lc_target >= 0 &&
            turn_lc_target < road.num_lanes && dist_to_end > fmaxf(TURN_LANE_MIN_LC_DIST, lc_no_start_dist)) {
        lc_target = turn_lc_target;
        mandatory_lc = true;
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_PREP, 1.0f);
    } else if (valid_lane_ecs(lookahead_lc_target, road) && dist_to_end > lc_no_start_dist) {
        lc_target = lookahead_lc_target;
        mandatory_lc = true;
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_PREP, 1.0f);
    } else if (valid_lane_ecs(lane_drop_lc_target, road) && dist_to_end > lc_no_start_dist) {
        lc_target = lane_drop_lc_target;
        mandatory_lc = true;
        AVABM_METRIC_ADD(metrics, METRIC_TURN_LANE_PREP, 1.0f);
    } else if (
#if THROUGH_LANE_BALANCE_LC_ENABLED
    ecs.lane_change_active[i] == 0 && !turn_needs_dedicated_lane && (turn == TURN_STRAIGHT ||
            upcoming_event_dist > OPEN_LANE_EMPTIEST_MIN_EXIT_DIST) && balance_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP &&
            road.lane_length[lane] >= THROUGH_LANE_BALANCE_LC_MIN_LINK_LENGTH &&
            dist_to_end > fmaxf(THROUGH_LANE_BALANCE_LC_MIN_DIST_TO_NODE, lc_no_start_dist + 2.0f) &&
            (ecs.lc_cooldown[i] <= CRUISE_RANDOM_LANE_COOLDOWN_READY || right_edge_balance_override)
#else
    false
#endif
    ) {
        int cruise_target = random_cruise_lane_step_target_ecs(i, lane, ecs, road);
        if (valid_lane_ecs(cruise_target, road) && same_approach_same_direction_lanes_ecs(lane, cruise_target, road)) {
            lc_target = cruise_target;
            opportunistic_lc = true;
            int cur_idx_force = -1;
            int tgt_idx_force = -1;
            int cur_count_force = lane_group_count_and_index_ecs(lane, road, cur_idx_force);
            int tgt_count_force = lane_group_count_and_index_ecs(cruise_target, road, tgt_idx_force);
            if (cur_count_force >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP && tgt_count_force == cur_count_force &&
                    cur_idx_force <= THROUGH_LANE_BALANCE_FORCE_RIGHT_IDX && tgt_idx_force > cur_idx_force) {
                mandatory_lc = true;
                opportunistic_lc = true;
            }
        }
    } else if (ecs.lane_change_active[i] == 0 && next_lane >= 0 && next_lane != lane && dist_to_end > lc_no_start_dist &&
            upcoming_event_dist <= OPEN_LANE_EMPTIEST_MIN_EXIT_DIST) {
        if (ll == next_lane) lc_target = ll;
        else if (rr == next_lane) lc_target = rr;
    } else if (ecs.lane_change_active[i] == 0 && (has_next || lane_group_count_ecs(lane, road) > 1) && (turn == TURN_STRAIGHT ||
            upcoming_event_dist > OPEN_LANE_EMPTIEST_MIN_EXIT_DIST) && !turn_needs_dedicated_lane) {
        int open_target = pick_open_lane_target_ecs(i, lane, ecs, road, grid, perception, current_time, max_entities, true, true);
        if (valid_lane_ecs(open_target, road)) {
            lc_target = open_target;
            mandatory_lc = false;
            opportunistic_lc = true;
#if RIGHT_EDGE_FORCE_INNER_LC_ENABLED
            int cur_idx_force = -1;
            int tgt_idx_force = -1;
            int cur_count_force = lane_group_count_and_index_ecs(lane, road, cur_idx_force);
            int tgt_count_force = lane_group_count_and_index_ecs(open_target, road, tgt_idx_force);
            float force_blocked_gap = fmaxf(CONGESTION_ESCAPE_CURRENT_GAP, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.95f);
            bool force_front_blocked = perception.front_gap[i] < force_blocked_gap ||
                    (perception.front_gap[i] < force_blocked_gap * 1.85f && perception.front_speed[i] + 2.0f < v);
            bool force_right_to_inner = cur_count_force >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP && tgt_count_force == cur_count_force &&
                    cur_idx_force >= 0 && tgt_idx_force > cur_idx_force && cur_idx_force <= RIGHT_EDGE_FORCE_INNER_MAX_CUR_IDX &&
                    upcoming_event_dist > RIGHT_EDGE_FORCE_INNER_MIN_EXIT_DIST;
            if (force_front_blocked || force_right_to_inner) {
                mandatory_lc = true;
                opportunistic_lc = true;
            }
#endif
        }
#if CRUISE_RANDOM_LANE_CHANGE_ENABLED
        else if (ecs.lc_cooldown[i] <= CRUISE_RANDOM_LANE_COOLDOWN_READY &&
                road.lane_length[lane] >= CRUISE_RANDOM_LANE_MIN_LINK_LENGTH &&
                dist_to_end > fmaxf(CRUISE_RANDOM_LANE_MIN_DIST_TO_NODE, lc_no_start_dist + 20.0f)) {
            uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
            uint32_t h = hash_u32_ecs(((uint32_t)(i + 1) * 1103515245u) ^ ((uint32_t)(ecs.route_id[i] + 23) * 2654435761u) ^
                    ((uint32_t)(road.lane_start_node[lane] + 31) * 747796405u) ^ (slot * 2891336453u));
            float p = CRUISE_RANDOM_LANE_CHANGE_PROB * (0.70f + 0.38f * clampf_cuda(ecs.aggressiveness[i], 0.0f, 1.0f));
            p = clampf_cuda(p, 0.035f, 0.32f);
            if (hash01_ecs(h) < p) {
                int cruise_target = random_cruise_lane_step_target_ecs(i, lane, ecs, road);
                if (valid_lane_ecs(cruise_target, road) && same_approach_same_direction_lanes_ecs(lane, cruise_target, road)) {
                    lc_target = cruise_target;
                    mandatory_lc = false;
                    opportunistic_lc = true;
                }
            }
        }
#endif
    } else if (
#if DESTINATION_SPREAD_LC_ENABLED
    ecs.lane_change_active[i] == 0 && !has_next && lane_group_count_ecs(lane, road) > 1 &&
            road.lane_length[lane] >= DESTINATION_SPREAD_MIN_LINK_LENGTH &&
            dist_to_end > fmaxf(DESTINATION_SPREAD_MIN_DIST_TO_END, lc_no_start_dist + 20.0f)
#else
    false
#endif
    ) {
        int open_target = pick_open_lane_target_ecs(i, lane, ecs, road, grid, perception, current_time, max_entities, true, true);
        if (valid_lane_ecs(open_target, road)) {
            lc_target = open_target;
            opportunistic_lc = true;
#if RIGHT_EDGE_FORCE_INNER_LC_ENABLED
            int cur_idx_force = -1;
            int tgt_idx_force = -1;
            int cur_count_force = lane_group_count_and_index_ecs(lane, road, cur_idx_force);
            int tgt_count_force = lane_group_count_and_index_ecs(open_target, road, tgt_idx_force);
            if (cur_count_force >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP && tgt_count_force == cur_count_force && cur_idx_force >= 0 &&
                    tgt_idx_force > cur_idx_force && cur_idx_force <= RIGHT_EDGE_FORCE_INNER_MAX_CUR_IDX) {
                mandatory_lc = true;
                opportunistic_lc = true;
            }
#endif
        }
#if CRUISE_RANDOM_LANE_CHANGE_ENABLED
        else if (ecs.lc_cooldown[i] <= CRUISE_RANDOM_LANE_COOLDOWN_READY) {
            uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
            uint32_t h = hash_u32_ecs(((uint32_t)(i + 19) * 1103515245u) ^ ((uint32_t)(ecs.route_id[i] + 41) * 2654435761u) ^
                    ((uint32_t)(lane + 97) * 747796405u) ^ (slot * 2891336453u));
            float p = clampf_cuda(DESTINATION_SPREAD_RANDOM_PROB * (0.65f + 0.45f * clampf_cuda(ecs.aggressiveness[i], 0.0f,
                    1.0f)), 0.035f, 0.42f);
            if (hash01_ecs(h) < p) {
                int cruise_target = random_cruise_lane_step_target_ecs(i, lane, ecs, road);
                if (valid_lane_ecs(cruise_target, road) && same_approach_same_direction_lanes_ecs(lane, cruise_target, road)) {
                    lc_target = cruise_target;
                    opportunistic_lc = true;
                }
            }
        }
#endif
    }
    if (lc_target >= 0 && !same_approach_same_direction_lanes_ecs(lane, lc_target, road)) {
        lc_target = -1;
    }
    if (lc_target >= 0) {
        find_lane_neighbors_ecs(i, lc_target, ecs, road, grid, max_entities, human ? 160.0f : 140.0f, nullptr,
                perception.target_front_gap[i], perception.target_front_speed[i], perception.target_rear_gap[i],
                perception.target_rear_speed[i]);
        int lc_sig = indicator_from_lateral_move_ecs(lane, lc_target, road);
        if (lc_sig != INDICATOR_NONE && ecs.turn_signal != nullptr) {
            if (ecs.turn_signal[i] == lc_sig) {
                if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = fminf(ecs.turn_signal_time[i] + dt, 60.0f);
            } else {
                ecs.turn_signal[i] = lc_sig;
                if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = dt;
            }
        }
        bool ok = mobil_decision_ecs(i, lc_target, ecs, road, grid, perception, metrics, current_time, max_entities, mandatory_lc,
                opportunistic_lc);
        if (ok) {
            decision.wants_lane_change[i] = 1;
            decision.lane_change_target[i] = lc_target;
        }
    }
    float connector_trigger_dist = CONNECTOR_EXIT_EPS;
    if (has_next) {
        float v_after_cmd = fmaxf(0.0f, v + acc_cmd * dt);
        float box_depth = intersection_box_depth_ecs(lane, next_lane, road);
        connector_trigger_dist = fmaxf(fmaxf(CONNECTOR_EXIT_EPS, box_depth), v_after_cmd * dt + CONNECTOR_TRIGGER_MARGIN);
    }
    bool node_continuation_release = false;
#if NODE_CONTINUATION_RELEASE_ENABLED
    if (has_next && turn == TURN_STRAIGHT) {
        int from_count_release = lane_group_count_ecs(lane, road);
        int to_count_release = lane_group_count_ecs(next_lane, road);
        bool taper_continuation = wide_lane_count_change_continuation_ecs(lane, next_lane, road);
        bool ramp_edge_merge = from_count_release <= INTERCHANGE_RAMP_MAX_GROUP_LANES &&
                to_count_release >= INTERCHANGE_MAIN_MIN_GROUP_LANES;
        bool front_space_release = perception.front_gap[i] > fmaxf(CONNECTOR_EXIT_SPACE_MIN,
                ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.75f);
        node_continuation_release = (taper_continuation || ramp_edge_merge) && front_space_release &&
                dist_to_end <= connector_trigger_dist + 4.0f;
    }
#endif
    bool priority_permits_connector = self_inside_intersection_box || !unsignal_node || !unsignal_blocked ||
            unsignal_deadlock_release || node_continuation_release;
    if (node_continuation_release) {
        reservation_granted = true;
    }
    if (has_next && reservation_granted && signal_permits_connector && priority_permits_connector && turn_lane_ok &&
            dist_to_end <= connector_trigger_dist) {
        decision.wants_connector[i] = 1;
        decision.connector_target_lane[i] = next_lane;
    }
    decision.desired_speed[i] = desired_v;
    decision.target_accel[i] = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);
    AVABM_ACTIVE_LOOP_END()
}
