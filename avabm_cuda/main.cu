// main.cu
// ECS-based CUDA traffic simulation core - right-turn entry-arc and symmetric lane marking build
// EN: Surface-aware, lane-disciplined traffic simulation core.
// KO: 도로를 단순 중심선이 아닌 "폭을 가진 면"으로 보고, 실제 차선/우선권/충돌 가능성을 반영하는 CUDA 코어입니다.
// EN: Most new comments are bilingual so the model can be studied and tuned later.
// KO: 나중에 코드를 해석하고 튜닝할 수 있도록 주요 알고리즘 주석은 영어/한글을 함께 적었습니다.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdint.h>
#include <math.h>

// ============================================================
// Constants
// ============================================================

#define HUMAN 0
#define AV    1

#define LIGHT_RED    0
#define LIGHT_YELLOW 1
#define LIGHT_GREEN  2

#define TURN_LEFT     -1
#define TURN_STRAIGHT  0
#define TURN_RIGHT     1
#define TURN_ANY       99

// EN: Turn indicator state shared through ECS. Rendering is optional;
//     drivers use this state to predict other vehicles' intended path.
// KO: ECS에 저장되는 방향지시등 상태입니다. 렌더링은 선택 사항이고,
//     다른 차량은 이 값을 보고 상대 차량의 진행 의도를 예측합니다.
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

// EN: Surface-aware connector geometry.
// KO: 도로 면 기반 회전 기하. 차가 교차로 중앙점까지 들어가서 꺾지 않고,
//     현재 차선 중심과 진입 차선 중심을 잇는 원호/완화 곡선을 따라 회전하게 합니다.
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

#define WORLD_MAX_CELL_RADIUS 5

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
#define LANE_CHANGE_NO_START_DIST_TO_NODE 92.0f
#define LANE_CHANGE_FINISH_BEFORE_NODE    28.0f
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

// EN: Indicator / deadlock / anti-penetration tuning constants.
// KO: 방향지시등, 교착 해소, 차량 투과 방지용 튜닝 상수입니다.
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
#define CONTACT_RESOLVE_INFLATE            0.18f
#define CONTACT_RESOLVE_BACKOFF            0.75f
#define CONTACT_RESOLVE_MAX_PUSH           4.20f
#define CONTACT_RESOLVE_PASSES             3

// EN: Intersection priority-gate constants.  The gate is applied before a car
//     enters the intersection.  After entry, the connector is treated as a
//     protected movement so the car keeps clearing the box.
// KO: 교차로 우선순위 게이트 상수입니다. 차량이 교차로에 들어가기 전에만
//     적용하고, 진입 후에는 보호된 진행으로 보아 교차로 박스를 계속 비우게 합니다.
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

// EN: Conflict-aware priority gate and right-turn corner tuning.
// KO: 실제로 경로가 겹치는 차량끼리만 우선순위를 비교하고, 우회전은 짧은 코너 경로를 사용합니다.
#define PRIORITY_GATE_PATH_SCAN_RANGE      48.0f
#define PRIORITY_GATE_EXIT_SPACE            8.5f
#define PRIORITY_GATE_ACTIVE_CLEAR_FRACTION 0.34f
#define PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST 9.0f
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

// EN: More assertive deadlock release, front-clear priority and lane-change etiquette.
// KO: 앞이 빈 차량 우선 통과, 교착 해소, 방향지시등 기반 차선변경 양보/거부 모델입니다.
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
#define LANE_CHANGE_PREP_MIN_CAP_AV           5.2f
#define LANE_CHANGE_PREP_MIN_CAP_HUMAN        4.0f
#define LC_PREP_COAST_ACCEL_LIMIT            -0.02f
#define LC_PREP_MAX_BRAKE_AV                  2.8f
#define LC_PREP_MAX_BRAKE_HUMAN               2.2f
#define LANE_CHANGE_COOP_ASSERTIVE_PROB       0.14f
#define LANE_CHANGE_COOP_RELAX_SCALE          0.68f
#define LANE_CHANGE_COOP_ZONE_MULT            1.85f
#define LANE_CHANGE_ACTIVE_SPEED_CAP_SCALE    0.86f
#define LC_INDICATOR_SIDE_RANGE              38.0f
#define LC_INDICATOR_SIDE_FRONT               9.0f
#define LC_INDICATOR_SIDE_REAR               30.0f
#define LC_COURTESY_DECEL_AV                  1.35f
#define LC_COURTESY_DECEL_HUMAN               1.05f
#define LC_ASSERTIVE_ACCEL_AV                 1.10f
#define LC_ASSERTIVE_ACCEL_HUMAN              1.45f
#define LC_ASSERTIVE_BLOCK_GAP               28.0f
#define LC_ASSERTIVE_BLOCK_TIME               1.20f
#define LOCAL_AVOID_RANGE                    24.0f
#define LOCAL_AVOID_HORIZON                   1.20f
#define LOCAL_AVOID_COLLISION_MARGIN          1.10f
#define LOCAL_AVOID_FRONT_CLEAR_BONUS        120
#define LOCAL_AVOID_CONNECTOR_BONUS          170
#define LOCAL_AVOID_INSIDE_BOX_BONUS         130
#define LOCAL_AVOID_STOP_BUFFER               2.0f
#define LOCAL_AVOID_IMMEDIATE_OVERLAP_INFLATE 0.08f

// EN: Intelligent anti-stall / anti-overlap / lane-intent model v14.
//     v14 fixes lane-drop deadlocks by treating wide 4->3 / 3->4 mainline
//     changes as straight continuations, adds early non-blocking lane-drop
//     merge preparation, and lets destination links spread across all lanes.
// KO: 지능형 정체/중첩/차선의도 보정 모델 v14입니다. 넓은 본선의 4->3 / 3->4
//     차로수 변화 구간을 회전이 아닌 직진 연속 구간으로 보고, 차로 감소 사전
//     합류와 도착 링크 다차로 분산을 추가해 데드락을 줄입니다.
#define SMART_AI_VERSION                         17
#define SPAWN_RACE_REQUEUE_ENABLED              1
#define SPAWN_RACE_RECENT_WINDOW                0.30f
#define SPAWN_RACE_REQUEUE_FULLSCAN_MAX       4096
#define CONTACT_LONGITUDINAL_REPAIR_GAP         3.20f
#define CONTACT_ROUTE_REPAIR_EXTRA              0.90f
#define CONTACT_CROSS_BACKOFF_MIN               1.50f
#define CONTACT_CROSS_BACKOFF_SPEED_TIME        0.35f
#define CONNECTOR_PROTECTED_PROGRESS_FRAC       0.16f
#define CONNECTOR_PROTECTED_EXIT_FRAC           0.30f
#define CONNECTOR_INBOX_MIN_CLEAR_SPEED_AV      3.40f
#define CONNECTOR_INBOX_MIN_CLEAR_SPEED_HUMAN   2.65f
#define SMART_STALL_SPEED_EPS                   0.42f
#define SMART_STALL_FRONT_GAP                   18.0f
#define SMART_STALL_CLEAR_ACCEL_SCALE           0.62f
#define SMART_STALL_RELEASE_WAIT                0.22f
#define MISSED_TURN_ESCAPE_WAIT                 1.20f
#define MISSED_TURN_ESCAPE_DIST                 18.0f
#define MISSED_TURN_ESCAPE_SPEED_AV             2.80f
#define MISSED_TURN_ESCAPE_SPEED_HUMAN          2.10f
#define LOCAL_AVOID_GRANTED_CONNECTOR_GRACE     1
#define COMPLETE_OVERLAP_RELEASE_ENABLED        1
#define COMPLETE_OVERLAP_RELEASE_DIST           1.20f
#define COMPLETE_OVERLAP_RELEASE_PERIOD         0.55f
#define COMPLETE_OVERLAP_RELEASE_MAX_SPEED      1.35f
#define COMPLETE_OVERLAP_RELEASE_SPEED_AV       5.20f
#define COMPLETE_OVERLAP_RELEASE_SPEED_HUMAN    4.05f
#define COMPLETE_OVERLAP_RELEASE_ACCEL_SCALE    0.78f
#define COMPLETE_OVERLAP_CONTACT_FORWARD_NUDGE  1.15f
#define COMPLETE_OVERLAP_CONTACT_BACKOFF_EXTRA  1.35f
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
#define CRUISE_RANDOM_LANE_CHANGE_PROB            0.22f
#define CRUISE_RANDOM_LANE_UTILITY_TOL            0.32f
#define OPEN_LANE_LC_UTILITY_TOL                  0.22f
#define ROUTE_MISMATCH_REPAIR_ENABLED                1
#define ROUTE_POS_REPAIR_SCAN_MAX                  24
#define UPCOMING_EXIT_LANE_PREP_ENABLED             1
#define UPCOMING_EXIT_LOOKAHEAD_LANES               6
#define UPCOMING_EXIT_PREP_EXTRA_DIST            70.0f
#define UPCOMING_EXIT_PREP_MAX_DIST             540.0f
#define WRONG_LANE_STALL_FORCE_WAIT               1.60f
#define STALE_BRAKE_CLEAR_FRONT_GAP             20.0f
#define STALE_BRAKE_CLEAR_ACCEL_AV               0.72f
#define STALE_BRAKE_CLEAR_ACCEL_HUMAN            0.50f
#define CONTACT_REPAIR_HOLD_ACCEL                0.00f
#define CONTACT_REPAIR_LOSER_SPEED_CAP           0.35f
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
#define LANE_SPREAD_MIN_DIST_TO_NODE            72.0f
#define LANE_SPREAD_MIN_LINK_LENGTH             70.0f
#define LANE_SPREAD_MIN_CURRENT_GAP             10.0f
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
#define MIDROAD_NEG_ACCEL_RELEASE_SCALE            0.42f
#define STOPPED_NEG_ACCEL_ZERO_SPEED              0.08f
#define OPEN_LANE_EMPTIEST_GROUP_SCAN_ENABLED       1
#define OPEN_LANE_EMPTIEST_SCAN_PERIOD             3.0f
#define OPEN_LANE_EMPTIEST_FRONT_GAIN              6.0f
#define OPEN_LANE_EMPTIEST_SCORE_GAIN              7.5f
#define OPEN_LANE_EMPTIEST_MIN_EXIT_DIST         210.0f

// EN: v17 anti-stall and congestion-aware lane choice.  The rightmost lane is
//     treated as a potential bottleneck lane unless an exit is near; through
//     traffic prefers safe inner/open lanes.  A zero-acceleration watchdog and
//     active-lane-change abort/commit guard prevent lane-drop no-start deadlocks.
// KO: v17 정체 기반 차선 선택입니다. 나들목이 가까운 경우가 아니면 우측 끝
//     차로를 병목 차선으로 보고, 직진 차량은 안전한 안쪽/빈 차선으로 분산됩니다.
//     또한 0가속도 감시와 차선변경 중단/완료 가드로 차로 감소 no-start 데드락을 막습니다.
#define MIDROAD_ZERO_ACCEL_WATCHDOG_ENABLED          1
#define MIDROAD_ZERO_ACCEL_CLEAR_SPEED             0.48f
#define MIDROAD_ZERO_ACCEL_CLEAR_FRONT_GAP         22.0f
#define MIDROAD_ZERO_ACCEL_RELEASE_SCALE            0.46f
#define MIDROAD_ZERO_ACCEL_MIN_END_DIST            12.0f
#define RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED          1
#define RIGHT_EDGE_BOTTLENECK_MIN_GROUP              3
#define RIGHT_EDGE_BOTTLENECK_IDX_LIMIT              1
#define RIGHT_EDGE_BOTTLENECK_PENALTY             24.0f
#define RIGHT_EDGE_INNER_BONUS                      5.5f
#define RIGHT_EDGE_SAFE_FRONT_LOSS                  4.0f
#define RIGHT_EDGE_SCAN_SCORE_GAIN                  2.0f
#define LANE_DROP_AMBIGUOUS_RIGHT_EDGE_FALLBACK      1
#define LANE_DROP_ACTIVE_LC_ABORT_T                0.38f
#define LANE_DROP_ACTIVE_LC_COMMIT_T               0.62f
#define LANE_DROP_ACTIVE_LC_BARRIER_EXTRA           1.25f
#define LANE_DROP_ACTIVE_LC_RELEASE_SPEED          1.60f

// EN: Interchange edge-lane rule.  A one-lane ramp/lane-drop attached to a
//     multi-lane mainline must use the nearest physical outside lane.  This
//     prevents ramps from entering/exiting through the center lane when source
//     GIS centerlines meet at the road axis.
// KO: 나들목 최외곽 차로 규칙입니다. 1차로 램프/차로감소 링크가 다차로
//     본선에 붙을 때, 원본 중심선이 도로 중앙에서 만나더라도 실제 차량은 가장
//     가까운 물리적 바깥 차로로만 들어오고 나갑니다.
#define INTERCHANGE_EDGE_ONLY_ENABLED             1
#define INTERCHANGE_RAMP_MAX_GROUP_LANES          1
#define INTERCHANGE_MAIN_MIN_GROUP_LANES          2

// EN: A right turn must begin before the mathematical node point.  If the
//     connector starts only after the car reaches the node, the cubic path has
//     to fold back into the receiving lane and visually invades other lanes.
//     These constants move the connector start a few meters upstream for right
//     turns and fit a short arc through the curb-side corridor.
// KO: 우회전은 수학적 노드 지점에 도달한 뒤 시작하면 안 됩니다. 노드에 닿은
//     뒤에야 connector가 시작되면 경로가 다시 진입 차선으로 접히면서 옆 차선을
//     침범해 보입니다. 아래 상수는 우회전 connector 시작점을 몇 m 앞당겨 실제
//     연석 쪽 corridor를 따라 짧은 원호로 돌게 합니다.
#define RIGHT_TURN_ENTRY_BACKOFF_MIN          6.00f
#define RIGHT_TURN_ENTRY_BACKOFF_MAX         16.00f
#define RIGHT_TURN_ENTRY_BACKOFF_BASE         7.20f
#define RIGHT_TURN_ENTRY_BACKOFF_PER_DEG      0.060f
#define TURN_ARC_STRICT_MAX_SWEEP_RAD         2.05f

// EN: Intersection box and lane-change discipline.  A line-based road graph
//     has no explicit polygon for an intersection, so CUDA approximates the
//     occupied intersection as a square/box depth derived from lane width.
//     Once a vehicle reaches that box, it should clear it instead of stopping
//     on the node.  Lane changes must be completed before this box.
// KO: 교차로 박스와 차선변경 규칙입니다. 선 기반 도로망에는 명시적 교차로 면이
//     없으므로, CUDA에서는 차로 폭을 기준으로 한 사각형/박스 깊이로 교차로 점유
//     구간을 근사합니다. 차량이 이 박스에 들어오면 노드 위에서 멈추지 말고 빠져나가야
//     하며, 차선변경은 이 박스 전에 끝나야 합니다.
#define INTERSECTION_BOX_LANE_WIDTH_MULT       1.55f
#define INTERSECTION_BOX_MIN_DEPTH             5.25f
#define INTERSECTION_BOX_MAX_DEPTH            19.50f
#define INTERSECTION_BOX_ENTRY_MARGIN          0.65f
#define INTERSECTION_BOX_PRIORITY_BONUS       96
#define LANE_CHANGE_BOX_CLEAR_MULT            10.50f
#define STRAIGHT_NO_TURN_SPEED_CAP_DEG        12.0f
#define RIGHT_TURN_CURB_BIAS_FRAC              0.18f

// EN: Lane-change intent, courtesy and front-clear deadlock release tuning.
// KO: 차선변경 의도 표시, 양보/공격성, 앞공간 우선 데드락 해소 튜닝입니다.
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

// EN: v7 intelligent-flow recovery. Keep queues outside/at stop lines, not in
//     the middle of intersections; prefer vehicles with a real front gap.
// KO: v7 지능형 주행 복구. 대기는 교차로 한가운데가 아니라 외부/정지선에서
//     만들고, 실제 앞공간이 있는 차량을 우선 통과시킵니다.
#define SMART_STALL_SPEED                  SMART_STALL_SPEED_EPS
#define SMART_STALL_RELEASE_SPEED_AV       4.9f
#define SMART_STALL_RELEASE_SPEED_HUMAN    3.8f
#define SMART_STALL_RELEASE_ACCEL_SCALE    SMART_STALL_CLEAR_ACCEL_SCALE
#define PRIORITY_GATE_FRONT_CLEAR_OVERRIDE_WAIT 0.85f
#define CONNECTOR_EXIT_SPACE_TIME          0.90f
#define CONNECTOR_EXIT_SPACE_MIN          12.0f
#define LC_ACTIVE_FREEZE_FRONT_MIN          7.0f
#define LC_ACTIVE_FREEZE_REAR_MIN           9.0f
#define LC_ACTIVE_FREEZE_T_MAX              0.62f

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

// ============================================================
// Render
// ============================================================

struct RenderVertex {
    float x, y;
    float r, g, b, a;
    float size;
};

static cudaGraphicsResource* g_render_resource = nullptr;
static int g_render_textured_cars = 0;

// ============================================================
// ECS Component Layout
// ============================================================

struct ECSArrays {
    // Entity
    int* alive;

    // Transform / Kinematics
    float* x;
    float* y;
    float* s;
    float* speed;
    float* accel;
    float* heading;
    float* steer_angle;

    // Vehicle spec
    float* length;
    float* width;

    // Driver component
    int* driver_type;
    float* reaction_time;
    float* min_gap;
    float* aggressiveness;
    float* politeness;
    float* risk_tolerance;
    float* comfort_decel;
    float* desired_speed_factor;

    // Lane component
    int* lane_id;

    // Route component
    int* route_id;
    int* route_pos;
    float* entry_time;

    // State component
    int* vehicle_state;

    // Connector component
    int* connector_from_lane;
    int* connector_to_lane;
    float* connector_s;
    float* connector_length;

    // Lane-change component
    int* lane_change_active;
    int* lane_change_from_lane;
    int* lane_change_to_lane;
    float* lane_change_t;
    float* lane_change_duration;
    float* lc_cooldown;

    // EN: Turn indicator component. -1 left, +1 right, 0 off.
    // KO: 방향지시등 컴포넌트입니다. -1 좌회전, +1 우회전, 0 꺼짐.
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
// Helpers
// ============================================================

__device__ __forceinline__ float clampf_cuda(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ int clampi_cuda(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    if (x == 0u) x = 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ __forceinline__ float rand_uniform(uint32_t& state) {
    return (float)(xorshift32(state) & 0x00FFFFFF) / 16777216.0f;
}

__device__ __forceinline__ uint32_t hash_u32_ecs(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float hash01_ecs(uint32_t x) {
    return (float)(hash_u32_ecs(x) & 0x00FFFFFFu) / 16777216.0f;
}

__device__ __forceinline__ bool timed_pair_random_self_wins_ecs(
    int self,
    int other,
    float current_time,
    float period
) {
    int a = self < other ? self : other;
    int b = self < other ? other : self;
    float safe_period = fmaxf(period, 0.01f);
    uint32_t slot = (uint32_t)floorf(fmaxf(current_time, 0.0f) / safe_period);
    uint32_t h = hash_u32_ecs(
        ((uint32_t)(a + 1) * 747796405u)
        ^ ((uint32_t)(b + 3) * 2891336453u)
        ^ (slot * 277803737u)
    );
    bool lower_id_wins = (h & 1u) == 0u;
    return (self == a) ? lower_id_wins : !lower_id_wins;
}

__device__ __forceinline__ float wrap_pi(float a) {
    while (a > 3.14159265359f) a -= 6.28318530718f;
    while (a < -3.14159265359f) a += 6.28318530718f;
    return a;
}

__device__ __forceinline__ float smoothstep01(float t) {
    t = clampf_cuda(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float unwrap_angle_near(float target, float reference) {
    return reference + wrap_pi(target - reference);
}

__device__ __forceinline__ float advance_heading_limited(
    float current,
    float target,
    float max_yaw_rate,
    float dt
) {
    float delta = wrap_pi(target - current);
    float max_delta = fmaxf(0.01f, max_yaw_rate * fmaxf(dt, 0.001f));
    delta = clampf_cuda(delta, -max_delta, max_delta);
    return wrap_pi(current + delta);
}


__device__ __forceinline__ float vehicle_wheelbase_from_length(float len) {
    return clampf_cuda(len * KIN_WHEELBASE_FACTOR, KIN_MIN_WHEELBASE, KIN_MAX_WHEELBASE);
}

__device__ __forceinline__ float vehicle_yaw_rate_from_steer(
    float speed,
    float steer,
    float wheelbase
) {
    if (speed < KIN_MIN_YAW_SPEED) return 0.0f;
    return speed * tanf(steer) / fmaxf(wheelbase, 0.1f);
}

__device__ __forceinline__ float advance_heading_bicycle_ecs(
    int id,
    ECSArrays ecs,
    float target_heading,
    float speed,
    float dt,
    float& steer_out,
    float& yaw_rate_out
) {
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

    /*
        EN: Pure-pursuit style steering command.  Heading can change only through
        v / wheelbase * tan(steer), so a stopped vehicle cannot spin in place
        and the instantaneous rotation center remains physically outside/near
        the axle geometry instead of being fixed at the vehicle center.
        KO: Pure-pursuit 방식의 조향 명령입니다. heading은 v/wheelbase*tan(steer)를 통해서만
        바뀌므로, 정지 차량이 제자리 회전하지 않고 실제 차량처럼 전진하면서 회전합니다.
    */
    float lookahead = speed * (human ? 1.70f : 1.35f) + wheelbase * 1.85f;
    lookahead = clampf_cuda(lookahead, wheelbase * 1.35f, human ? 26.0f : 22.0f);

    float curvature = 2.0f * sinf(err) / fmaxf(lookahead, 0.1f);
    float steer_cmd = atanf(wheelbase * curvature);
    steer_cmd = clampf_cuda(steer_cmd, -max_steer, max_steer);

    float max_delta_steer = max_steer_rate * fmaxf(dt, 0.001f);
    float steer = prev_steer + clampf_cuda(
        steer_cmd - prev_steer,
        -max_delta_steer,
        max_delta_steer
    );
    steer = clampf_cuda(steer, -max_steer, max_steer);

    float yaw_rate = vehicle_yaw_rate_from_steer(speed, steer, wheelbase);
    yaw_rate = clampf_cuda(yaw_rate, -max_yaw_rate, max_yaw_rate);

    float new_h = wrap_pi(old_h + yaw_rate * dt);

    /*
        Path-locked bicycle pose correction.

        The previous version moved the vehicle center along the lane/connector
        curve but let heading lag too far behind that path.  On tight turns the
        body looked as if it was rotating around a point outside the vehicle, or
        even sliding slightly backward.  We still compute yaw from the steering
        angle, but clamp the visual heading error against the path tangent so the
        body remains aligned with the forward-moving path.  Stopped vehicles
        still cannot spin in place because this block only runs above
        KIN_MIN_YAW_SPEED.
    */
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

__device__ __forceinline__ float enforce_path_heading_error_limit_ecs(
    int id,
    ECSArrays ecs,
    float candidate_heading,
    float path_heading,
    float speed,
    float dt,
    float max_error,
    float& steer_out,
    float& yaw_rate_out
) {
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

__device__ __forceinline__ float sensor_half_fov_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_FRONT_HALF_FOV_HUMAN : SENSOR_FRONT_HALF_FOV_AV;
}

__device__ __forceinline__ float sensor_front_range_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_FRONT_RANGE_HUMAN : SENSOR_FRONT_RANGE_AV;
}

__device__ __forceinline__ float sensor_side_range_for_driver(int dtype) {
    return dtype == HUMAN ? SENSOR_SIDE_RANGE_HUMAN : SENSOR_SIDE_RANGE_AV;
}

__device__ __forceinline__ bool point_in_oriented_cone(
    float ox,
    float oy,
    float heading,
    float px,
    float py,
    float range,
    float half_fov,
    float lateral_margin,
    float& forward,
    float& lateral,
    float& dist
) {
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

__device__ __forceinline__ bool sensor_front_cone_detects_ecs(
    int self,
    int other,
    ECSArrays ecs,
    float range,
    float half_fov,
    float& forward,
    float& lateral,
    float& dist
) {
    float margin =
        SENSOR_CONE_EDGE_MARGIN
        + 0.5f * fmaxf(ecs.width[other], 1.0f)
        + SENSOR_RAY_WIDTH;

    return point_in_oriented_cone(
        ecs.x[self],
        ecs.y[self],
        ecs.heading[self],
        ecs.x[other],
        ecs.y[other],
        range,
        half_fov,
        margin,
        forward,
        lateral,
        dist
    );
}

__device__ __forceinline__ bool sensor_rear_mirror_detects_ecs(
    int self,
    int other,
    ECSArrays ecs,
    float range,
    float& forward,
    float& lateral,
    float& dist
) {
    float margin =
        SENSOR_CONE_EDGE_MARGIN
        + 0.5f * fmaxf(ecs.width[other], 1.0f)
        + SENSOR_RAY_WIDTH;

    return point_in_oriented_cone(
        ecs.x[self],
        ecs.y[self],
        wrap_pi(ecs.heading[self] + 3.14159265359f),
        ecs.x[other],
        ecs.y[other],
        range,
        SENSOR_SIDE_HALF_FOV,
        margin,
        forward,
        lateral,
        dist
    );
}

__device__ __forceinline__ float apply_reaction_delay_accel(
    float prev_accel,
    float target_accel,
    float reaction_time,
    int dtype,
    float dt,
    float* metrics
) {
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
        atomicAdd(&metrics[METRIC_DELAY_SUM], rt);
        atomicAdd(&metrics[METRIC_DELAY_COUNT], 1.0f);
        atomicAdd(&metrics[METRIC_REACTION_SUM], reaction_time);
        atomicAdd(&metrics[METRIC_REACTION_COUNT], 1.0f);
        atomicAdd(&metrics[METRIC_RESPONSE_LAG_SUM], fabsf(target_accel - effective));
        atomicAdd(&metrics[METRIC_RESPONSE_LAG_COUNT], 1.0f);
    }

    return effective;
}

__device__ __forceinline__ int world_cell_index(
    float px,
    float py,
    float world_min_x,
    float world_min_y,
    float world_cell_size,
    int world_grid_w,
    int world_grid_h
) {
    if (!isfinite(px) || !isfinite(py) || world_cell_size <= 0.0f) return -1;

    int cx = (int)floorf((px - world_min_x) / world_cell_size);
    int cy = (int)floorf((py - world_min_y) / world_cell_size);

    if (cx < 0 || cx >= world_grid_w) return -1;
    if (cy < 0 || cy >= world_grid_h) return -1;

    return cy * world_grid_w + cx;
}

__device__ __forceinline__ void lane_dir(
    int lane,
    const RoadNetwork road,
    float& dx,
    float& dy
) {
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

__device__ __forceinline__ float lane_heading(int lane, const RoadNetwork road) {
    float dx, dy;
    lane_dir(lane, road, dx, dy);
    return atan2f(dy, dx);
}

__device__ __forceinline__ void lane_xy_from_s(
    int lane,
    float ss,
    const RoadNetwork road,
    float& ox,
    float& oy
) {
    float L = fmaxf(road.lane_length[lane], 0.1f);
    float t = clampf_cuda(ss / L, 0.0f, 1.0f);

    ox = road.lane_start_x[lane] + t * (road.lane_end_x[lane] - road.lane_start_x[lane]);
    oy = road.lane_start_y[lane] + t * (road.lane_end_y[lane] - road.lane_start_y[lane]);
}

__device__ __forceinline__ void lane_xy_heading_from_s(
    int lane,
    float ss,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    lane_xy_from_s(lane, ss, road, ox, oy);
    oh = lane_heading(lane, road);
}

__device__ __forceinline__ bool lane_connected(
    int a,
    int b,
    const RoadNetwork road
) {
    return road.lane_end_node[a] == road.lane_start_node[b];
}

__device__ __forceinline__ int route_pos_for_lane_ecs(
    int route_id,
    int lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ float turn_angle_deg(
    int a,
    int b,
    const RoadNetwork road
) {
    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);

    float dot = clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
    float cross = ax * by - ay * bx;

    return fabsf(atan2f(cross, dot)) * 57.2957795f;
}


__device__ __forceinline__ float lane_signed_turn_deg(
    int a,
    int b,
    const RoadNetwork road
) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return 0.0f;

    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);

    float dot = clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
    float cross = ax * by - ay * bx;

    return atan2f(cross, dot) * 57.2957795f;
}

__device__ __forceinline__ int turn_code_from_lanes_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (from_lane < 0 || to_lane < 0) return TURN_STRAIGHT;

    float signed_deg = lane_signed_turn_deg(from_lane, to_lane, road);

    /* Keep the same sign convention as Python classify_turn():
       positive mathematical rotation is a left turn, and TURN_LEFT == -1. */
    if (signed_deg > 25.0f) return TURN_LEFT;
    if (signed_deg < -25.0f) return TURN_RIGHT;
    return TURN_STRAIGHT;
}


__device__ __forceinline__ void lane_midpoint_ecs(
    int lane,
    const RoadNetwork road,
    float& mx,
    float& my
) {
    mx = 0.5f * (road.lane_start_x[lane] + road.lane_end_x[lane]);
    my = 0.5f * (road.lane_start_y[lane] + road.lane_end_y[lane]);
}

__device__ __forceinline__ bool valid_lane_ecs(
    int lane,
    const RoadNetwork road
) {
    return lane >= 0 && lane < road.num_lanes;
}

__device__ __forceinline__ bool lanes_share_link_geometry_ecs(
    int a,
    int b,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(a, road) || !valid_lane_ecs(b, road)) return false;

    bool same_nodes =
        road.lane_start_node[a] == road.lane_start_node[b]
        && road.lane_end_node[a] == road.lane_end_node[b];
    if (!same_nodes) return false;

    float ah = lane_heading(a, road);
    float bh = lane_heading(b, road);
    return fabsf(wrap_pi(bh - ah)) < 0.35f;
}

__device__ __forceinline__ float neighbor_lateral_cross_ecs(
    int lane,
    int neighbor,
    const RoadNetwork road
) {
    if (!lanes_share_link_geometry_ecs(lane, neighbor, road)) return 0.0f;

    float dx, dy;
    lane_dir(lane, road, dx, dy);

    float mx, my, nx, ny;
    lane_midpoint_ecs(lane, road, mx, my);
    lane_midpoint_ecs(neighbor, road, nx, ny);

    /*
        Positive cross = neighbor is physically to the left of the current
        travel direction.  Negative cross = physically to the right.
        This is safer than trusting the imported left_lane/right_lane labels,
        because some datasets expose lane order in the opposite convention.
    */
    return dx * (ny - my) - dy * (nx - mx);
}

__device__ __forceinline__ int geometric_left_neighbor_ecs(
    int lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;

    int candidates[2] = { road.left_lane[lane], road.right_lane[lane] };
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

__device__ __forceinline__ int geometric_right_neighbor_ecs(
    int lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;

    int candidates[2] = { road.left_lane[lane], road.right_lane[lane] };
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

__device__ __forceinline__ bool is_leftmost_lane_ecs(
    int lane,
    const RoadNetwork road
) {
    return valid_lane_ecs(lane, road) && geometric_left_neighbor_ecs(lane, road) < 0;
}

__device__ __forceinline__ bool is_rightmost_lane_ecs(
    int lane,
    const RoadNetwork road
) {
    return valid_lane_ecs(lane, road) && geometric_right_neighbor_ecs(lane, road) < 0;
}

__device__ __forceinline__ int rightmost_lane_in_group_ecs(
    int lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int cur = lane;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP; ++k) {
        int nb = geometric_right_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return cur;
}

__device__ __forceinline__ int lane_group_count_and_index_ecs(
    int lane,
    const RoadNetwork road,
    int& out_index
) {
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

__device__ __forceinline__ int lane_at_right_to_left_index_ecs(
    int lane,
    int target_index,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int leftmost_lane_in_group_ecs(
    int lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;
    int cur = lane;
    for (int k = 0; k < CRUISE_RANDOM_LANE_MAX_GROUP; ++k) {
        int nb = geometric_left_neighbor_ecs(cur, road);
        if (!valid_lane_ecs(nb, road)) break;
        cur = nb;
    }
    return cur;
}

__device__ __forceinline__ float lane_endpoint_dist2_ecs(
    int lane,
    float px,
    float py,
    bool at_end,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return 1.0e30f;
    float x = at_end ? road.lane_end_x[lane] : road.lane_start_x[lane];
    float y = at_end ? road.lane_end_y[lane] : road.lane_start_y[lane];
    float dx = x - px;
    float dy = y - py;
    return dx * dx + dy * dy;
}

__device__ __forceinline__ int nearest_outer_lane_to_point_ecs(
    int group_lane,
    float px,
    float py,
    bool at_end,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(group_lane, road)) return -1;
    int right = rightmost_lane_in_group_ecs(group_lane, road);
    int left = leftmost_lane_in_group_ecs(group_lane, road);
    if (!valid_lane_ecs(right, road)) right = group_lane;
    if (!valid_lane_ecs(left, road)) left = group_lane;
    float dr = lane_endpoint_dist2_ecs(right, px, py, at_end, road);
    float dl = lane_endpoint_dist2_ecs(left, px, py, at_end, road);
    return dr <= dl ? right : left;
}

__device__ __forceinline__ bool lane_groups_same_ecs(
    int a,
    int b,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(a, road) || !valid_lane_ecs(b, road)) return false;
    if (a == b) return true;
    if (road.lane_start_node[a] != road.lane_start_node[b]) return false;
    if (road.lane_end_node[a] != road.lane_end_node[b]) return false;
    float ah = lane_heading(a, road);
    float bh = lane_heading(b, road);
    return fabsf(wrap_pi(bh - ah)) < 0.35f;
}

__device__ __forceinline__ int lane_group_count_ecs(
    int lane,
    const RoadNetwork road
) {
    int idx = -1;
    return lane_group_count_and_index_ecs(lane, road, idx);
}

__device__ __forceinline__ bool route_lane_current_compatible_ecs(
    int route_lane,
    int current_lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(route_lane, road) || !valid_lane_ecs(current_lane, road)) return false;
    if (route_lane == current_lane) return true;
    return lane_groups_same_ecs(route_lane, current_lane, road);
}

__device__ __forceinline__ int repair_route_pos_for_current_lane_ecs(
    int current_lane,
    int route_id,
    int route_pos,
    const RoadNetwork road
) {
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
        if (d < 0) d = -d + 3;  // prefer a slightly downstream-compatible match.
        if (d < best_score) {
            best_score = d;
            best = k;
        }
    }
    if (best >= 0) return best;

    // Fallback: keep the current route position if the actual lane can still
    // enter the route's next lane.  This covers random same-link lane changes
    // where the route lane itself is not listed but the connector is valid.
    for (int k = lo; k < hi; ++k) {
        int next_lane = road.route_lanes[ro0 + k + 1];
        if (valid_lane_ecs(next_lane, road) && lane_connected(current_lane, next_lane, road)) {
            return k;
        }
    }
#endif
    return -1;
}


__device__ __forceinline__ int connected_equivalent_lane_from_group_ecs(
    int from_lane,
    int group_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int interchange_receiving_outer_lane_ecs(
    int from_lane,
    int candidate_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int interchange_source_outer_lane_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ bool wide_lane_count_change_continuation_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int effective_turn_code_ecs(
    int from_lane,
    int to_lane,
    int route_turn,
    const RoadNetwork road
) {
    int geom_turn = turn_code_from_lanes_ecs(from_lane, to_lane, road);

    // EN: A 4->3 or 3->4 wide mainline continuation may bend more than the
    //     generic 25 degree turn threshold.  Do not force all vehicles to the
    //     left/right edge lane for that case; it is a lane-count transition,
    //     not an exit turn.
    // KO: 4->3 또는 3->4 본선 연속 구간은 일반 25도 회전 기준보다 크게 휘어
    //     보일 수 있습니다. 이 경우 모든 차량을 좌/우 가장자리 차로로 몰지 말고
    //     차로수 변화 직진 구간으로 처리합니다.
    if (wide_lane_count_change_continuation_ecs(from_lane, to_lane, road)) {
        return TURN_STRAIGHT;
    }

    if (geom_turn != TURN_STRAIGHT || route_turn == TURN_STRAIGHT) return geom_turn;
    return route_turn;
}

__device__ __forceinline__ int lane_count_reduction_drop_side_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (!wide_lane_count_change_continuation_ecs(from_lane, to_lane, road)) return 0;
    int from_count = lane_group_count_ecs(from_lane, road);
    int to_count = lane_group_count_ecs(to_lane, road);
    if (from_count <= to_count) return 0;

    int from_right = rightmost_lane_in_group_ecs(from_lane, road);
    int from_left = leftmost_lane_in_group_ecs(from_lane, road);
    int to_right = rightmost_lane_in_group_ecs(to_lane, road);
    int to_left = leftmost_lane_in_group_ecs(to_lane, road);
    if (!valid_lane_ecs(from_right, road) || !valid_lane_ecs(from_left, road) ||
        !valid_lane_ecs(to_right, road) || !valid_lane_ecs(to_left, road)) {
        return 0;
    }

    float dr = lane_endpoint_dist2_ecs(from_right, road.lane_start_x[to_right], road.lane_start_y[to_right], true, road);
    float dl = lane_endpoint_dist2_ecs(from_left,  road.lane_start_x[to_left],  road.lane_start_y[to_left],  true, road);
    float eps2 = LANE_COUNT_CHANGE_EDGE_ALIGN_EPS * LANE_COUNT_CHANGE_EDGE_ALIGN_EPS;

    if (dr > dl + eps2) return -1;  // right edge moves inward: rightmost lane(s) drop.
    if (dl > dr + eps2) return  1;  // left edge moves inward: leftmost lane(s) drop.
    return 0;
}

__device__ __forceinline__ int lane_count_reduction_step_target_ecs(
    int lane,
    int next_lane,
    const RoadNetwork road
) {
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
    // EN: Some GIS lane-drop geometries keep both outer edges almost aligned,
    //     making the disappearing side ambiguous.  In Korea the right edge is
    //     the frequent bottleneck/merge side, so move only the rightmost extra
    //     lane inward as a conservative fallback.
    // KO: GIS상 양쪽 끝점이 거의 맞아 어느 쪽 차로가 사라지는지 애매한 경우가
    //     있습니다. 우측 차로 병목이 잦으므로, fallback은 우측 끝 여분 차로만
    //     안쪽으로 합류시킵니다.
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

__device__ __forceinline__ int lane_steps_to_specific_lane_ecs(
    int lane,
    int target_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int adjacent_lane_toward_specific_lane_ecs(
    int lane,
    int target_lane,
    const RoadNetwork road
) {
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

__device__ __forceinline__ int random_cruise_lane_step_target_ecs(
    int id,
    int lane,
    ECSArrays ecs,
    const RoadNetwork road
) {
    int cur_index = -1;
    int count = lane_group_count_and_index_ecs(lane, road, cur_index);
    if (count <= 1 || cur_index < 0) return -1;

    /*
        EN: Stable per-vehicle/per-route lane occupancy for through traffic.
            Vehicles not preparing for a left/right exit pick a deterministic
            pseudo-random lane within the current lane bundle and move one
            adjacent lane at a time.
        KO: 좌/우 진출 준비가 아닌 직진 차량은 현재 차로 묶음 안에서 차량/route별
            안정적인 pseudo-random 목표 차로를 고르고, 한 번에 한 차로씩 이동합니다.
    */
    uint32_t key =
        ((uint32_t)(id + 1) * 747796405u)
        ^ ((uint32_t)(ecs.route_id[id] + 101) * 2891336453u)
        ^ ((uint32_t)(ecs.route_pos[id] + 17) * 277803737u)
        ^ ((uint32_t)(road.lane_start_node[lane] + 3) * 1442695041u)
        ^ ((uint32_t)(road.lane_end_node[lane] + 5) * 1597334677u);
    int target_index = (int)(hash_u32_ecs(key) % (uint32_t)count);
    if (target_index == cur_index) return -1;

    if (target_index > cur_index) return geometric_left_neighbor_ecs(lane, road);
    return geometric_right_neighbor_ecs(lane, road);
}

__device__ __forceinline__ bool turn_requires_dedicated_lane_ecs(int turn) {
    return turn == TURN_LEFT || turn == TURN_RIGHT;
}

__device__ __forceinline__ bool lane_legal_for_turn_ecs(
    int lane,
    int turn,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return false;
    if (turn == TURN_LEFT) return is_leftmost_lane_ecs(lane, road);
    if (turn == TURN_RIGHT) return is_rightmost_lane_ecs(lane, road);
    return true;
}

__device__ __forceinline__ int adjacent_lane_toward_turn_lane_ecs(
    int lane,
    int turn,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;
    if (turn == TURN_LEFT) return geometric_left_neighbor_ecs(lane, road);
    if (turn == TURN_RIGHT) return geometric_right_neighbor_ecs(lane, road);
    return -1;
}

__device__ __forceinline__ int receiving_lane_for_turn_ecs(
    int candidate_lane,
    int turn,
    const RoadNetwork road
) {
    /*
        EN: Choose the lane that should be entered immediately after a turn.
            A right turn must enter the physical rightmost lane of the receiving
            road, and a left turn must enter the physical leftmost lane.  Route
            generation already tries to do this, but this CUDA-side guard keeps
            old route caches or noisy lane ordering from forcing a right-turn
            vehicle to cut across several lanes inside the intersection.

        KO: 회전 직후 진입해야 하는 차선을 다시 보정합니다. 우회전은 진입 도로의
            실제 맨 우측 차선으로, 좌회전은 실제 맨 좌측 차선으로 들어가야 합니다.
            Python route도 그렇게 생성하지만, 오래된 cache나 데이터 차선 순서 오류가
            있어도 CUDA 최종 단계에서 교차로 내부 다차선 침범을 막습니다.
    */
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

__device__ __forceinline__ int lane_steps_to_turn_lane_ecs(
    int lane,
    int turn,
    const RoadNetwork road
) {
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

__device__ __forceinline__ float turn_lane_prep_distance_ecs(
    int lane_steps,
    float speed,
    int dtype
) {
    if (lane_steps <= 0) return 0.0f;
    float human_factor = dtype == HUMAN ? 1.28f : 1.0f;
    float dynamic = fmaxf(speed, 4.0f) * (dtype == HUMAN ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV);
    float d = TURN_LANE_PREP_BASE_DIST + TURN_LANE_PREP_PER_LANE_DIST * lane_steps + dynamic;
    return clampf_cuda(d * human_factor, 35.0f, 190.0f);
}

__device__ __forceinline__ float turn_lane_hold_accel_ecs(
    float dist_to_end,
    float speed,
    int dtype
) {
    float hold_dist = fmaxf(0.65f, dist_to_end - (DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER));
    float req = -(speed * speed) / fmaxf(2.0f * hold_dist, 0.5f);
    float max_b = dtype == HUMAN ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    return clampf_cuda(req, -EMERGENCY_DECEL, 0.0f) - 0.05f * max_b;
}

__device__ __forceinline__ int upcoming_exit_lane_step_target_ecs(
    int id,
    int lane,
    ECSArrays ecs,
    const RoadNetwork road,
    float dist_to_end,
    float speed,
    int dtype,
    int* out_event_turn,
    float* out_event_distance
) {
#if UPCOMING_EXIT_LANE_PREP_ENABLED
    if (out_event_turn != nullptr) *out_event_turn = TURN_STRAIGHT;
    if (out_event_distance != nullptr) *out_event_distance = 1.0e9f;
    if (!valid_lane_ecs(lane, road) || ecs.vehicle_state[id] != VEH_ON_LANE) return -1;

    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    if (rid < 0 || rid >= road.num_routes) return -1;

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
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

            int target_edge_on_current = wants_right
                ? rightmost_lane_in_group_ecs(lane, road)
                : leftmost_lane_in_group_ecs(lane, road);
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

__device__ __forceinline__ int route_next_lane_for_vehicle_ecs(
    int id,
    ECSArrays ecs,
    const RoadNetwork road
) {
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

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[id] = repaired_pos;
    }
    if (rpos < 0 || rpos >= route_len) return -1;

    int next_pos = rpos + 1;
    if (next_pos < 0 || next_pos >= route_len) return -1;

    int scan_hi = min(route_len, next_pos + ROUTE_NEXT_LANE_EQUIV_SCAN);
    for (int pos = next_pos; pos < scan_hi; ++pos) {
        int candidate = road.route_lanes[ro0 + pos];
        if (!valid_lane_ecs(candidate, road)) continue;

        int route_turn = road.route_turns[ro0 + max(0, pos - 1)];
        int turn = effective_turn_code_ecs(lane, candidate, route_turn, road);

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

        // EN: The route may name a different lane of the same receiving link.
        //     Try every same-link lane and keep a connector-valid alternative.
        // KO: route가 같은 수신 링크의 다른 차로를 가리키는 경우가 있습니다.
        //     같은 링크 묶음의 차로 중 실제 연결 가능한 차로를 찾아 복구합니다.
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

    return -1;
}


__device__ __forceinline__ int route_turn_for_vehicle_ecs(
    int id,
    ECSArrays ecs,
    const RoadNetwork road
) {
    if (ecs.vehicle_state[id] == VEH_IN_CONNECTOR) {
        return effective_turn_code_ecs(
            ecs.connector_from_lane[id],
            ecs.connector_to_lane[id],
            TURN_STRAIGHT,
            road
        );
    }

    int rid = ecs.route_id[id];
    int rpos = ecs.route_pos[id];
    int lane = ecs.vehicle_state[id] == VEH_IN_CONNECTOR ? ecs.connector_from_lane[id] : ecs.lane_id[id];
    if (rid < 0 || rid >= road.num_routes) return TURN_STRAIGHT;

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
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

__device__ __forceinline__ int indicator_from_lateral_move_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return INDICATOR_NONE;
    float side = neighbor_lateral_cross_ecs(from_lane, to_lane, road);
    if (side > 0.05f) return INDICATOR_LEFT;
    if (side < -0.05f) return INDICATOR_RIGHT;
    return INDICATOR_NONE;
}

__device__ __forceinline__ int indicator_state_ecs(
    int id,
    ECSArrays ecs
) {
    if (ecs.turn_signal == nullptr) return INDICATOR_NONE;
    int v = ecs.turn_signal[id];
    if (v == INDICATOR_LEFT || v == INDICATOR_RIGHT || v == INDICATOR_HAZARD) return v;
    return INDICATOR_NONE;
}

__device__ __forceinline__ bool indicator_active_ecs(
    int id,
    ECSArrays ecs
) {
    return indicator_state_ecs(id, ecs) == INDICATOR_LEFT
        || indicator_state_ecs(id, ecs) == INDICATOR_RIGHT;
}


__device__ __forceinline__ int indicator_target_lane_ecs(
    int lane,
    int signal,
    const RoadNetwork road
) {
    if (!valid_lane_ecs(lane, road)) return -1;
    if (signal == INDICATOR_LEFT) return geometric_left_neighbor_ecs(lane, road);
    if (signal == INDICATOR_RIGHT) return geometric_right_neighbor_ecs(lane, road);
    return -1;
}

__device__ __forceinline__ bool indicator_matches_lateral_move_ecs(
    int from_lane,
    int to_lane,
    ECSArrays ecs,
    int id,
    const RoadNetwork road
) {
    int needed = indicator_from_lateral_move_ecs(from_lane, to_lane, road);
    if (needed == INDICATOR_NONE) return true;
    return indicator_state_ecs(id, ecs) == needed;
}

__device__ __forceinline__ int intended_turn_with_indicator_ecs(
    int id,
    int from_lane,
    int to_lane,
    ECSArrays ecs,
    const RoadNetwork road
) {
    int route_turn = route_turn_for_vehicle_ecs(id, ecs, road);
    int turn = effective_turn_code_ecs(from_lane, to_lane, route_turn, road);

    int sig = indicator_state_ecs(id, ecs);
    // EN: An active indicator is treated as a public intent declaration.
    //     It does not override an already-entered connector, but it lets
    //     other vehicles anticipate left/right movements before entry.
    // KO: 켜진 깜빡이는 공개된 의도 선언으로 취급합니다. 이미 connector에 들어간
    //     차량의 실제 경로를 바꾸지는 않지만, 진입 전 좌/우회전을 미리 예측하게 합니다.
    if (ecs.vehicle_state[id] != VEH_IN_CONNECTOR) {
        if (sig == INDICATOR_LEFT) return TURN_LEFT;
        if (sig == INDICATOR_RIGHT) return TURN_RIGHT;
    }
    return turn;
}

__device__ __forceinline__ float lane_heading_dot_ecs(
    int a,
    int b,
    const RoadNetwork road
) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return 1.0f;

    float ax, ay, bx, by;
    lane_dir(a, road, ax, ay);
    lane_dir(b, road, bx, by);

    return clampf_cuda(ax * bx + ay * by, -1.0f, 1.0f);
}

__device__ __forceinline__ bool same_approach_same_direction_lanes_ecs(
    int a,
    int b,
    const RoadNetwork road
) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return false;
    if (a == b) return true;

    bool same_nodes =
        road.lane_start_node[a] == road.lane_start_node[b]
        && road.lane_end_node[a] == road.lane_end_node[b];

    return same_nodes && lane_heading_dot_ecs(a, b, road) > 0.82f;
}

__device__ __forceinline__ bool opposite_corridor_lanes_ecs(
    int a,
    int b,
    const RoadNetwork road
) {
    if (a < 0 || b < 0 || a >= road.num_lanes || b >= road.num_lanes) return false;

    bool reverse_nodes =
        road.lane_start_node[a] == road.lane_end_node[b]
        && road.lane_end_node[a] == road.lane_start_node[b];

    return reverse_nodes && lane_heading_dot_ecs(a, b, road) < -0.82f;
}


// ============================================================
// Surface-aware connector / conflict geometry
// EN: The road graph is line-based, but vehicles occupy lanes on a road surface.
//     These helpers build a smooth centerline trajectory through the surface
//     of an intersection.  Right turns stay near the right-side lane center;
//     left turns stay near the left-side lane center.
// KO: 입력 도로는 선(LineString)이지만 실제 차량은 폭이 있는 도로 면 위의 차로를 주행합니다.
//     아래 함수들은 교차로 안에서 차선 중심을 따라 자연스러운 원호/완화 곡선을 만듭니다.
//     우회전은 우측 면을, 좌회전은 좌측 면을 따라 돌도록 합니다.
// ============================================================


__device__ __forceinline__ bool right_turn_corner_line_params_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road,
    float& corner_x,
    float& corner_y,
    float& from_param,
    float& to_param
) {
    /*
        EN: Geometry core for a lane-preserving right turn.

        The lane-center lines of the incoming rightmost lane and outgoing
        rightmost lane meet at a physical curb-side corner, not at the abstract
        graph node.  A natural right turn should be tangent to both lane centers
        around this corner.  `from_param` is the signed distance from the incoming
        lane end to that corner along the incoming heading; `to_param` is the
        signed distance from the outgoing lane start to the corner along the
        outgoing heading.

        KO: 차선을 보존하는 우회전의 핵심 기하입니다.

        진입 우측 차선 중심선과 진출 우측 차선 중심선은 추상 graph node가 아니라
        실제 연석 쪽 코너에서 만납니다. 자연스러운 우회전은 이 코너를 기준으로 두
        차선 중심선에 접해야 합니다. `from_param`은 진입 차선 끝에서 진입 heading
        방향으로 코너까지의 signed 거리, `to_param`은 진출 차선 시작에서 진출
        heading 방향으로 코너까지의 signed 거리입니다.
    */
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

    // Solve E + a*t0 = S + b*t1.
    float rx = s1x - e0x;
    float ry = s1y - e0y;
    float det = t0x * t1y - t0y * t1x;
    if (fabsf(det) < 1.0e-4f) return false;

    from_param = (rx * t1y - ry * t1x) / det;
    to_param   = (rx * t0y - ry * t0x) / det;

    if (!isfinite(from_param) || !isfinite(to_param)) return false;
    if (fabsf(from_param) > 32.0f || fabsf(to_param) > 32.0f) return false;

    corner_x = e0x + t0x * from_param;
    corner_y = e0y + t0y * from_param;
    return true;
}

__device__ __forceinline__ float connector_local_handoff_estimate_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (
        from_lane < 0 || from_lane >= road.num_lanes
        || to_lane < 0 || to_lane >= road.num_lanes
    ) {
        return 0.0f;
    }

    float raw_dx = road.lane_start_x[to_lane] - road.lane_end_x[from_lane];
    float raw_dy = road.lane_start_y[to_lane] - road.lane_end_y[from_lane];
    float raw_chord = sqrtf(raw_dx * raw_dx + raw_dy * raw_dy);

    // EN: For right turns, exit farther into the receiving lane so the curve
    //     can follow the rounded curb instead of cutting diagonally across
    //     inner lanes.
    // KO: 우회전은 진출 차선 안쪽으로 더 들어간 지점에서 connector를 끝내야
    //     연석 반경을 따라 돌고 안쪽 차선을 대각선으로 침범하지 않습니다.
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

    // EN: Only exact same-node connectors need an artificial handoff into the target lane.
    // KO: 양쪽 차선 끝점이 거의 같은 경우에만 목표 차선 안쪽으로 짧게 넘겨 줍니다.
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

__device__ __forceinline__ float connector_entry_backoff_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    /*
        EN: Start right-turn connector slightly upstream from the lane end.
            This is the missing piece in the previous builds: the vehicle waited
            until its reference point was at the node and then tried to rotate
            into the receiving lane.  Real cars begin the steering motion while
            still in the rightmost through lane, so the path must start before
            the node.  Left turns keep the old start because they naturally cross
            the intersection box and were visually acceptable in the video.

        KO: 우회전 connector 시작점을 차선 끝보다 조금 앞쪽으로 당깁니다. 이전
            버전은 차량 기준점이 노드에 도착한 뒤 회전을 시작해서 진입 차선으로
            억지로 접히는 느낌이 났습니다. 실제 차량은 맨 우측 직진 차선 안에서
            이미 조향을 시작하므로, 우회전 경로도 노드보다 앞에서 시작해야 합니다.
            좌회전은 교차로 박스를 자연스럽게 가로지르므로 기존 시작점을 유지합니다.
    */
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return 0.0f;
    int turn = turn_code_from_lanes_ecs(from_lane, to_lane, road);
    if (turn != TURN_RIGHT) return 0.0f;

    float deg = fabsf(lane_signed_turn_deg(from_lane, to_lane, road));
    float backoff = RIGHT_TURN_ENTRY_BACKOFF_BASE + RIGHT_TURN_ENTRY_BACKOFF_PER_DEG * deg;

    // EN: Align the connector start with the tangent point of the curb arc.
    //     If the abstract node lies ahead of the physical lane-center corner,
    //     the car must start steering even earlier.
    // KO: connector 시작점을 연석 원호의 접점과 맞춥니다. 추상 node가 실제
    //     차선 중심 코너보다 앞에 있으면 더 일찍 조향을 시작해야 합니다.
    float corner_x, corner_y, from_param, to_param;
    if (right_turn_corner_line_params_ecs(from_lane, to_lane, road, corner_x, corner_y, from_param, to_param)) {
        backoff = fmaxf(backoff, -from_param + DEFAULT_LANE_WIDTH * 1.55f);
    }

    float lane_L = fmaxf(road.lane_length[from_lane], 0.1f);
    backoff = clampf_cuda(backoff, RIGHT_TURN_ENTRY_BACKOFF_MIN, RIGHT_TURN_ENTRY_BACKOFF_MAX);
    return fminf(backoff, fmaxf(0.0f, lane_L * 0.48f));
}

__device__ __forceinline__ void connector_surface_endpoints_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road,
    float& p0x,
    float& p0y,
    float& h0,
    float& p3x,
    float& p3y,
    float& h3,
    float& handoff_s
) {
    h0 = lane_heading(from_lane, road);
    h3 = lane_heading(to_lane, road);

    float entry_backoff = connector_entry_backoff_ecs(from_lane, to_lane, road);
    p0x = road.lane_end_x[from_lane] - cosf(h0) * entry_backoff;
    p0y = road.lane_end_y[from_lane] - sinf(h0) * entry_backoff;

    handoff_s = connector_local_handoff_estimate_ecs(from_lane, to_lane, road);
    p3x = road.lane_start_x[to_lane] + cosf(h3) * handoff_s;
    p3y = road.lane_start_y[to_lane] + sinf(h3) * handoff_s;
}

__device__ __forceinline__ float lane_width_estimate_ecs(
    int lane,
    const RoadNetwork road
) {
    /*
        EN: Estimate lane width from geometric neighbors.  The CUDA network does
            not carry lane_width explicitly, so use the nearest left/right lane
            center distance when available and fall back to DEFAULT_LANE_WIDTH.
        KO: CUDA 네트워크에는 lane_width가 직접 들어오지 않으므로 좌/우 이웃 차로
            중심 간 거리를 이용해 차로 폭을 추정하고, 없으면 DEFAULT_LANE_WIDTH를 사용합니다.
    */
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


__device__ __forceinline__ float intersection_box_depth_ecs(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    /*
        EN: Approximate the intersection entry region with a lane-width based
            rectangular depth along the incoming lane.  This replaces the older
            vague circular/radius approach.  The exact polygon is not available
            in the line network, but a square depth of about one to two lane
            widths gives a clear rule: before the box, obey entry priority; once
            inside the box, clear the intersection.

        KO: 진입 차선 방향으로 차로 폭 기반의 사각형 깊이를 사용해 교차로 진입
            범위를 근사합니다. 예전처럼 애매한 반경으로 보지 않고, 선 기반 도로망에서
            사용 가능한 차선 폭으로 "박스"를 만듭니다. 박스 밖에서는 진입 우선순위를
            지키고, 박스 안에 들어온 차량은 교차로를 비우는 쪽을 우선합니다.
    */
    float lw = lane_width_estimate_ecs(from_lane, road);
    if (valid_lane_ecs(to_lane, road)) {
        lw = fminf(lw, lane_width_estimate_ecs(to_lane, road));
    }
    float entry = connector_entry_backoff_ecs(from_lane, to_lane, road);
    float d = fmaxf(lw * INTERSECTION_BOX_LANE_WIDTH_MULT, entry + lw * 0.35f);
    return clampf_cuda(d, INTERSECTION_BOX_MIN_DEPTH, INTERSECTION_BOX_MAX_DEPTH);
}

__device__ __forceinline__ bool inside_intersection_box_ecs(
    float dist_to_end,
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    return dist_to_end <= intersection_box_depth_ecs(from_lane, to_lane, road) + INTERSECTION_BOX_ENTRY_MARGIN;
}

__device__ __forceinline__ float lane_change_no_start_distance_ecs(
    int lane,
    const RoadNetwork road
) {
    float lw = lane_width_estimate_ecs(lane, road);
    return fmaxf(LANE_CHANGE_NO_START_DIST_TO_NODE, lw * LANE_CHANGE_BOX_CLEAR_MULT);
}

__device__ __forceinline__ float lane_change_finish_distance_ecs(
    int lane,
    const RoadNetwork road
) {
    float lw = lane_width_estimate_ecs(lane, road);
    return fmaxf(LANE_CHANGE_FINISH_BEFORE_NODE, lw * (LANE_CHANGE_BOX_CLEAR_MULT * 0.62f));
}

__device__ __forceinline__ bool connector_surface_arc_params_raw_ecs(
    float p0x,
    float p0y,
    float h0,
    float p3x,
    float p3y,
    float h3,
    float& cx,
    float& cy,
    float& radius,
    float& a0,
    float& sweep,
    float& arc_len
) {
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);

    if (abs_delta < SURFACE_TURN_MIN_DELTA_RAD) return false;

    float sign = delta >= 0.0f ? 1.0f : -1.0f;
    float n0x = -sinf(h0);
    float n0y =  cosf(h0);
    float n1x = -sinf(h3);
    float n1y =  cosf(h3);

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

    // EN: When both tangent centers agree, a true circular arc is used.
    // KO: 시작/종료 접선이 만드는 원 중심이 잘 맞으면 실제 원호를 사용합니다.
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

__device__ __forceinline__ void connector_surface_fallback_bezier_ecs(
    int from_lane,
    int to_lane,
    float u,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);

    float span_dx = p3x - p0x;
    float span_dy = p3y - p0y;
    float span = sqrtf(span_dx * span_dx + span_dy * span_dy);
    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);

    // EN: Short handles prevent the vehicle from cutting to the node center.
    // KO: 핸들을 짧게 잡아 차량이 교차로 중앙으로 파고들어 꺾는 현상을 줄입니다.
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

    ox =
        ww * w * p0x
        + 3.0f * ww * u * p1x
        + 3.0f * w * uu * p2x
        + uu * u * p3x;

    oy =
        ww * w * p0y
        + 3.0f * ww * u * p1y
        + 3.0f * w * uu * p2y
        + uu * u * p3y;

    float dxdt =
        3.0f * ww * (p1x - p0x)
        + 6.0f * w * u * (p2x - p1x)
        + 3.0f * uu * (p3x - p2x);
    float dydt =
        3.0f * ww * (p1y - p0y)
        + 6.0f * w * u * (p2y - p1y)
        + 3.0f * uu * (p3y - p2y);

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


__device__ __forceinline__ bool connector_right_turn_corner_xy_heading_ecs(
    int from_lane,
    int to_lane,
    float u,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    /*
        EN: Right-turn special path, tangent-fillet version.

        A real car turning right from the rightmost lane does not drive to the
        middle of the intersection and then rotate.  It follows the incoming
        lane forward, rounds the curb, and joins the rightmost receiving lane.
        The most stable geometric proxy for that is the intersection point of
        the incoming lane tangent and the backward outgoing-lane tangent.  A
        quadratic Bezier p0 -> q -> p3 is then used as a short fillet.

        This prevents the previous cubic control points from drifting toward
        the node center and invading adjacent lanes during right turns.

        KO: 우회전 전용 경로, 접선 fillet 버전입니다.

        실제 차량은 우회전할 때 교차로 중앙까지 들어갔다가 제자리로 회전하지
        않습니다. 진입 차선을 따라 조금 전진하고, 우측 연석을 둥글게 돌며,
        진출 도로의 맨 우측 차선으로 합류합니다. 이를 가장 안정적으로 표현하기
        위해 진입 차선 접선과 진출 차선 역방향 접선의 교점을 제어점 q로 잡고,
        p0 -> q -> p3 2차 Bezier fillet을 사용합니다.

        이렇게 하면 이전 cubic 제어점이 교차로 중앙 쪽으로 흘러가 우회전 중
        여러 차선을 침범하는 현상을 줄일 수 있습니다.
    */
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return false;

    float p0x, p0y, h0, p3x, p3y, h3, handoff_s;
    connector_surface_endpoints_ecs(from_lane, to_lane, road, p0x, p0y, h0, p3x, p3y, h3, handoff_s);

    float delta = wrap_pi(h3 - h0);
    float abs_delta = fabsf(delta);

    // EN/KO: Only normal right turns use this fillet path.
    if (delta >= -0.18f || abs_delta > 2.30f) return false;

    float t0x = cosf(h0);
    float t0y = sinf(h0);
    float t3x = cosf(h3);
    float t3y = sinf(h3);

    float dx = p3x - p0x;
    float dy = p3y - p0y;
    float chord = sqrtf(dx * dx + dy * dy);
    if (!isfinite(chord) || chord < 0.45f) return false;

    /* Solve p0 + a*t0 = p3 - b*t3.
       EN: q is the corner point where the two lane-center tangents meet.
       KO: q는 두 차선 중심 접선이 만나는 우측 코너 제어점입니다. */
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

    /* EN: Keep q near the start-end chord.  If q is far away due to noisy map
       geometry, the fillet would become a loop or cut across lanes.
       KO: 지도 기하가 불안정해 q가 멀리 튀면 loop나 차선 침범이 되므로 chord 주변으로
       제한합니다. */
    float ux = dx / chord;
    float uy = dy / chord;
    float px = -uy;
    float py = ux;
    float qrx = qx - p0x;
    float qry = qy - p0y;
    float qproj = clampf_cuda(qrx * ux + qry * uy, 0.02f * chord, 0.98f * chord);
    float lane_w = fminf(lane_width_estimate_ecs(from_lane, road), lane_width_estimate_ecs(to_lane, road));
    float qlat_limit = clampf_cuda(
        fmaxf(lane_w * 0.62f, chord * RIGHT_TURN_CHORD_LATERAL_LIMIT),
        1.10f,
        4.40f
    );
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
        float err = clampf_cuda(
            wrap_pi(tan_h - interp_h),
            -RIGHT_TURN_FILLET_MAX_HEADING_ERR,
            RIGHT_TURN_FILLET_MAX_HEADING_ERR
        );
        oh = wrap_pi(interp_h + RIGHT_TURN_FILLET_TANGENT_BLEND * err);
    } else {
        oh = interp_h;
    }

    return true;
}


__device__ __forceinline__ bool connector_right_turn_lane_following_xy_heading_ecs(
    int from_lane,
    int to_lane,
    float u,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    /*
        EN: Curb-arc right turn, lane-preserving version.

        This replaces the previous diagonal/quadratic fallback.  The old path
        still sometimes connected the incoming lane end to the outgoing lane
        start by a chord across the intersection box.  That is why the vehicle
        appeared to invade several lanes while turning right.

        New path:
        1) Find the intersection of the incoming right-lane centerline and the
           outgoing right-lane centerline.  This is the physical curb-side corner.
        2) Use the connector entry backoff and receiving-lane handoff to define
           tangent points on those SAME lane centerlines.
        3) Drive a cubic Hermite/Bezier segment tangent to both centerlines.
           Because endpoints and tangents lie on the rightmost lane centers, the
           vehicle rounds the curb outside the old rectangular node box instead
           of cutting diagonally through inner lanes.

        KO: 차선을 보존하는 연석 원호형 우회전입니다.

        이전 경로는 진입 차선 끝과 진출 차선 시작을 교차로 박스를 가로지르는 chord로
        연결하는 경우가 남아 있었습니다. 그래서 우회전 차량이 여러 차선을 침범하는
        것처럼 보였습니다.

        새 경로는 다음과 같습니다.
        1) 진입 우측 차선 중심선과 진출 우측 차선 중심선의 교점을 찾습니다. 이것이
           실제 연석 쪽 코너입니다.
        2) connector entry backoff와 진출 차선 handoff를 이용해 두 차선 중심선 위의
           접점을 잡습니다.
        3) 두 차선 중심선에 접하는 cubic Hermite/Bezier로 회전합니다. 시작/끝/접선이
           모두 우측 끝 차선 중심에 있기 때문에, 예전 직사각형 node 박스를 대각선으로
           자르지 않고 바깥쪽 연석 반경을 따라 돕니다.
    */
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

    // EN/KO: Force both endpoints to stay on the two lane centerlines around the curb corner.
    float r_start = entry_backoff + from_param;
    float r_exit  = exit_handoff - to_param;
    float radius = fminf(r_start, r_exit);
    radius = clampf_cuda(radius, lane_w * 1.10f, lane_w * 2.75f);

    float ax = corner_x - t0x * radius;
    float ay = corner_y - t0y * radius;
    float bx = corner_x + t3x * radius;
    float by = corner_y + t3y * radius;

    // If the requested entry/exit offsets are longer than the tangent radius,
    // keep short straight lead/exit pieces.  This preserves continuity with the
    // lane motion state and avoids a visual teleport at connector enter/exit.
    float lead_len = fmaxf(0.0f, sqrtf((ax - p0x) * (ax - p0x) + (ay - p0y) * (ay - p0y)));
    float exit_len = fmaxf(0.0f, sqrtf((p3x - bx) * (p3x - bx) + (p3y - by) * (p3y - by)));

    float handle = clampf_cuda(radius * 0.62f, lane_w * 0.65f, lane_w * 2.10f);
    float c1x = ax + t0x * handle;
    float c1y = ay + t0y * handle;
    float c2x = bx - t3x * handle;
    float c2y = by - t3y * handle;

    // EN: Estimate curve length cheaply for progress distribution.
    // KO: 진행률 분배용으로 곡선 길이를 가볍게 근사합니다.
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

__device__ __forceinline__ bool connector_surface_arc_xy_heading_ecs(
    int from_lane,
    int to_lane,
    float u,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    /*
        EN: True single-arc turn path.  When the lane-center endpoints and
            tangents form a reasonable quarter-turn, this is more vehicle-like
            than a free cubic: curvature is monotonic, the heading is exactly
            tangent to the path, and a right turn cannot accidentally produce a
            loop that swings across adjacent lanes.

        KO: 실제 원호 기반 회전 경로입니다. 차선 중심 끝점과 접선이 정상적인
            90도 회전에 가까우면 자유 cubic보다 훨씬 차량답습니다. 곡률이 단조롭고
            차체 heading이 경로 접선과 일치하며, 우회전이 loop를 만들어 옆 차선을
            침범하는 현상을 막습니다.
    */
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

__device__ __forceinline__ void connector_surface_path_xy_heading_ecs(
    int from_lane,
    int to_lane,
    float u,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    /*
        EN: Intersection-corridor connector.

        The map is line-based, but the vehicle must behave as if it is driving
        on a lane surface.  The previous arc fitter could occasionally accept a
        large sweep and make a right/left turn look like it was orbiting the
        whole intersection.  This version deliberately uses a short tangent
        cubic between the source lane center and the receiving lane center.

        - Right turns start from the physical rightmost lane and enter the
          physical rightmost receiving lane, so the path hugs the right-side
          road surface instead of going to the intersection center.
        - Left turns start from the physical leftmost lane and enter the
          physical leftmost receiving lane, so the path crosses the box once,
          without an extra loop.
        - Heading comes from the path tangent but is blended with monotonic
          from-heading to to-heading interpolation to prevent sudden spins.

        KO: 교차로 주행 corridor입니다.

        지도는 선 기반이지만 차량은 차선 폭을 가진 면 위를 주행해야 합니다. 이전
        원호 fitting은 가끔 너무 큰 sweep을 허용해 우회전/좌회전이 교차로 전체를
        빙 도는 것처럼 보였습니다. 이 버전은 출발 차선 중심과 진입 차선 중심을
        짧은 접선 cubic으로 연결합니다.

        - 우회전은 실제 맨 우측 차선에서 시작해 진입 도로의 맨 우측 차선으로
          들어가므로 교차로 중앙까지 들어가지 않고 우측 면을 따라 돕니다.
        - 좌회전은 실제 맨 좌측 차선에서 시작해 맨 좌측 진입 차선으로 들어가며,
          교차로를 한 번만 가로지르고 추가 loop를 만들지 않습니다.
        - heading은 경로 접선을 쓰되, 시작 heading에서 종료 heading으로 단조롭게
          보간한 값과 섞어 급격한 제자리 회전을 막습니다.
    */
    if (
        from_lane < 0 || from_lane >= road.num_lanes
        || to_lane < 0 || to_lane >= road.num_lanes
    ) {
        ox = 0.0f;
        oy = 0.0f;
        oh = 0.0f;
        return;
    }

    // EN: IMPORTANT RIGHT-TURN FIX.
    //     The previous build used a separate tangent-fillet path only for right
    //     turns.  In noisy GIS geometry that special case produced an asymmetric
    //     turn angle and could swing into adjacent lanes.  Right turns now use
    //     exactly the same short corridor-cubic family as left turns, mirrored by
    //     the lane geometry itself.  This makes right-turn yaw and lateral motion
    //     match the natural left-turn behavior while still enforcing rightmost
    //     source/receiving lanes elsewhere.
    // KO: 중요 우회전 수정입니다.
    //     이전 빌드는 우회전에만 별도 tangent-fillet 경로를 썼습니다. GIS 기하가
    //     조금만 불안정해도 우회전 각도가 좌회전과 다르게 비틀리고, 옆 차선을
    //     침범할 수 있었습니다. 이제 우회전도 좌회전과 동일한 짧은 corridor-cubic
    //     계열을 사용하고, 좌/우 차이는 차선 기하가 자연스럽게 반영합니다.
    //     우측 끝 차선에서 출발/진입해야 한다는 규칙은 decision/entry 단계에서
    //     계속 강제합니다.

    // EN/KO: Right turns use a stricter lane-preserving path first.
    if (connector_right_turn_lane_following_xy_heading_ecs(from_lane, to_lane, u, road, ox, oy, oh)) {
        return;
    }

    // EN/KO: Prefer a true arc when the two lane tangents fit.
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

    /* EN: Short handles keep the path inside the intended lane corridor.
       KO: 짧은 핸들은 경로가 의도한 차선 corridor 밖으로 부풀어 오르지 않게 합니다. */
    float handle = fminf(
        fmaxf(chord * 0.36f, SURFACE_TURN_HANDLE_MIN),
        SURFACE_TURN_HANDLE_MAX
    );
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

    /* EN: Remove control point overshoot that can create a loop.
       KO: loop를 만드는 제어점 과대 돌출을 줄입니다. */
    if (chord > 0.25f) {
        float ux = dx / chord;
        float uy = dy / chord;
        float p1proj = (p1x - p0x) * ux + (p1y - p0y) * uy;
        float p2proj = (p2x - p0x) * ux + (p2y - p0y) * uy;
        float lo = -0.10f * chord;
        float hi =  1.10f * chord;
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

    ox =
        ww * w * p0x
        + 3.0f * ww * u * p1x
        + 3.0f * w * uu * p2x
        + uu * u * p3x;
    oy =
        ww * w * p0y
        + 3.0f * ww * u * p1y
        + 3.0f * w * uu * p2y
        + uu * u * p3y;

    float dxdt =
        3.0f * ww * (p1x - p0x)
        + 6.0f * w * u * (p2x - p1x)
        + 3.0f * uu * (p3x - p2x);
    float dydt =
        3.0f * ww * (p1y - p0y)
        + 6.0f * w * u * (p2y - p1y)
        + 3.0f * uu * (p3y - p2y);

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

__device__ __forceinline__ bool connector_swept_paths_overlap_ecs(
    int a_from,
    int a_to,
    int b_from,
    int b_to,
    const RoadNetwork road
) {
    if (
        a_to < 0 || a_to >= road.num_lanes
        || b_to < 0 || b_to >= road.num_lanes
    ) {
        return false;
    }

    if (a_to == b_to) return true;

    const float threshold = SURFACE_TURN_CONFLICT_RADIUS;
    const float threshold2 = threshold * threshold;

    // EN: Three samples are a cheap swept-path proxy for intersection conflicts.
    // KO: 세 지점 샘플로 회전/직진 경로가 실제로 겹치는지 가볍게 판정합니다.
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

__device__ __forceinline__ bool intersection_conflict_relevant_lanes_ecs(
    int self_lane,
    int self_next_lane,
    int other_lane,
    int other_next_lane,
    bool other_in_connector,
    const RoadNetwork road
) {
    if (
        self_lane < 0 || self_lane >= road.num_lanes
        || other_lane < 0 || other_lane >= road.num_lanes
    ) {
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

    bool swept_overlap = connector_swept_paths_overlap_ecs(
        self_lane,
        self_next_lane,
        other_lane,
        other_next_lane,
        road
    );
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

__device__ __forceinline__ bool intersection_conflict_relevant_vehicles_ecs(
    int self,
    int self_lane,
    int self_next_lane,
    int other,
    int other_lane,
    int other_next_lane,
    bool other_in_connector,
    ECSArrays ecs,
    const RoadNetwork road
) {
    if (
        self_lane < 0 || self_lane >= road.num_lanes
        || other_lane < 0 || other_lane >= road.num_lanes
    ) return false;

    if (same_approach_same_direction_lanes_ecs(self_lane, other_lane, road)) return false;

    int self_turn = intended_turn_with_indicator_ecs(self, self_lane, self_next_lane, ecs, road);
    int other_turn = intended_turn_with_indicator_ecs(other, other_lane, other_next_lane, ecs, road);

    bool swept_overlap = connector_swept_paths_overlap_ecs(
        self_lane,
        self_next_lane,
        other_lane,
        other_next_lane,
        road
    );

    // EN: If the observed indicator says the other vehicle is peeling away and
    //     the actual surface paths do not overlap, keep moving.  This prevents
    //     side traffic from freezing a clear straight path.
    // KO: 상대 차량의 깜빡이가 내 경로와 멀어지는 방향이고 실제 도로 면 경로가
    //     겹치지 않으면 그대로 진행합니다. 옆 차 때문에 불필요하게 멈추는 것을 막습니다.
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

__device__ __forceinline__ float directional_attention_range_ecs(
    int self_lane,
    int other_lane,
    int dtype,
    const RoadNetwork road
) {
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

__device__ __forceinline__ bool directional_vehicle_conflict_relevant_ecs(
    int self,
    int self_lane,
    int self_next_lane,
    int other,
    ECSArrays ecs,
    const RoadNetwork road
) {
    if (other == self || ecs.alive[other] != ENTITY_ALIVE) return false;
    if (self_lane < 0 || self_lane >= road.num_lanes) return false;

    bool other_in_connector = ecs.vehicle_state[other] == VEH_IN_CONNECTOR;
    int other_lane = other_in_connector ? ecs.connector_from_lane[other] : ecs.lane_id[other];
    int other_next_lane = other_in_connector
        ? ecs.connector_to_lane[other]
        : route_next_lane_for_vehicle_ecs(other, ecs, road);

    if (other_lane < 0 || other_lane >= road.num_lanes) return false;

    int self_node = road.lane_end_node[self_lane];
    int other_node = road.lane_end_node[other_lane];

    /* Not the same conflict node: ignore same-direction and opposite-side
       vehicles.  This keeps a car from braking for traffic behind a median or
       for vehicles on a nearby parallel road. */
    if (self_node != other_node) {
        return false;
    }

    return intersection_conflict_relevant_vehicles_ecs(
        self,
        self_lane,
        self_next_lane,
        other,
        other_lane,
        other_next_lane,
        other_in_connector,
        ecs,
        road
    );
}

__device__ __forceinline__ float turn_speed_cap(float angle_deg, int dtype) {
    bool human = dtype == HUMAN;

    float straight = human ? TURN_SPEED_STRAIGHT_HUMAN : TURN_SPEED_STRAIGHT_AV;
    float hard     = human ? TURN_SPEED_HARD_HUMAN     : TURN_SPEED_HARD_AV;
    float uturn    = human ? TURN_SPEED_UTURN_HUMAN    : TURN_SPEED_UTURN_AV;

    if (angle_deg < 8.0f) return straight;

    if (angle_deg > 115.0f) {
        float t = clampf_cuda((angle_deg - 115.0f) / 65.0f, 0.0f, 1.0f);
        return hard + (uturn - hard) * smoothstep01(t);
    }

    float t = clampf_cuda((angle_deg - 8.0f) / 107.0f, 0.0f, 1.0f);
    return straight + (hard - straight) * smoothstep01(t);
}

__device__ __forceinline__ float connector_length_between_lanes(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (from_lane < 0 || to_lane < 0) return CONNECTOR_DEFAULT_LEN;
    if (from_lane >= road.num_lanes || to_lane >= road.num_lanes) return CONNECTOR_DEFAULT_LEN;

    /* EN: Measure the same corridor path that rendering/physics uses.  This
       keeps connector_s speed proportional to real traveled distance and avoids
       the old mismatch where length came from an accepted circular arc but the
       pose came from a different fallback curve.
       KO: 렌더링/물리가 실제로 사용하는 corridor 경로를 샘플링해 길이를 잽니다.
       connector_s 속도와 실제 이동 거리가 맞고, 예전처럼 길이와 위치 경로가 달라져
       회전이 이상해지는 일을 막습니다. */
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

__device__ __forceinline__ float connector_geometry_chord(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    if (from_lane < 0 || to_lane < 0) return 1.0e9f;

    float dx = road.lane_start_x[to_lane] - road.lane_end_x[from_lane];
    float dy = road.lane_start_y[to_lane] - road.lane_end_y[from_lane];
    float chord = sqrtf(dx * dx + dy * dy);

    if (!isfinite(chord)) return 1.0e9f;
    return chord;
}

__device__ __forceinline__ bool connector_uses_handoff(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    return connector_geometry_chord(from_lane, to_lane, road) < CONNECTOR_SAME_NODE_EPS;
}

__device__ __forceinline__ float connector_exit_handoff_s(
    int from_lane,
    int to_lane,
    const RoadNetwork road
) {
    /*
        EN: The connector trajectory and the lane handoff must use the exact same
            target-lane S value.  Earlier code only applied the local-handoff
            estimate while drawing/integrating the curved connector, then snapped
            the vehicle to S=0 for non-same-node right turns at completion.  That
            mismatch is the typical cause of the visible upward pop after a
            right turn finishes.
        KO: connector 곡선 끝점과 차선 복귀 S 좌표를 반드시 같은 함수로 계산합니다.
            이전 로직은 우회전 곡선을 목표 차선 안쪽 지점까지 그려 놓고, 완료 시에는
            일부 경우 S=0으로 복귀시켜 차량이 위로 튀는 현상이 발생했습니다.
    */
    return connector_local_handoff_estimate_ecs(from_lane, to_lane, road);
}

__device__ __forceinline__ float connector_route_distance_to_next_lane_s(
    int from_lane,
    int to_lane,
    float remain_from_lane,
    float next_lane_s,
    const RoadNetwork road
) {
    float clen = connector_length_between_lanes(from_lane, to_lane, road);
    float entry_backoff = connector_entry_backoff_ecs(from_lane, to_lane, road);
    float handoff = connector_exit_handoff_s(from_lane, to_lane, road);
    float after_handoff = fmaxf(0.0f, next_lane_s - handoff);
    return fmaxf(0.0f, remain_from_lane - entry_backoff) + clen + after_handoff;
}

__device__ __forceinline__ void connector_xy_heading_from_s(
    int from_lane,
    int to_lane,
    float conn_s,
    float conn_len,
    const RoadNetwork road,
    float& ox,
    float& oy,
    float& oh
) {
    // EN: Map connector progress to the road-surface turn path.
    // KO: connector_s 진행률을 도로 면 위의 회전 경로에 매핑합니다.
    conn_len = fmaxf(conn_len, CONNECTOR_MIN_LEN);
    float u = clampf_cuda(conn_s / conn_len, 0.0f, 1.0f);
    connector_surface_path_xy_heading_ecs(from_lane, to_lane, u, road, ox, oy, oh);
}




// ============================================================
// Signal helpers
// ============================================================

__device__ __forceinline__ bool in_phase(float p, float a, float b) {
    if (a <= b) return p >= a && p < b;
    return p >= a || p < b;
}

__device__ __forceinline__ int signal_state(
    float t,
    float cycle,
    float green_start,
    float green_end,
    float yellow_start,
    float yellow_end
) {
    if (cycle <= 1.0f) return LIGHT_GREEN;

    float p = fmodf(t, cycle);
    if (p < 0.0f) p += cycle;

    if (in_phase(p, green_start, green_end)) return LIGHT_GREEN;
    if (in_phase(p, yellow_start, yellow_end)) return LIGHT_YELLOW;

    return LIGHT_RED;
}

__device__ __forceinline__ bool signal_turn_match(int signal_turn, int turn) {
    if (signal_turn == TURN_ANY) return true;
    if (turn == TURN_LEFT) return signal_turn == TURN_LEFT;
    if (turn == TURN_RIGHT) return signal_turn == TURN_RIGHT || signal_turn == TURN_STRAIGHT;
    return signal_turn == TURN_STRAIGHT;
}

__device__ __forceinline__ int get_signal_for_lane_turn(
    int lane,
    int turn,
    float current_time,
    const RoadNetwork road,
    const Signals signals
) {
    int node = road.lane_end_node[lane];
    int found = LIGHT_GREEN;

    for (int k = 0; k < signals.num_signals; ++k) {
        if (signals.signal_node[k] != node) continue;
        if (!signal_turn_match(signals.signal_turn[k], turn)) continue;

        int st = signal_state(
            current_time,
            signals.signal_cycle[k],
            signals.signal_green_start[k],
            signals.signal_green_end[k],
            signals.signal_yellow_start[k],
            signals.signal_yellow_end[k]
        );

        if (st == LIGHT_RED) return LIGHT_RED;
        if (st == LIGHT_YELLOW) found = LIGHT_YELLOW;
    }

    return found;
}

__device__ __forceinline__ bool node_has_signal_ecs(
    int node,
    const Signals signals
) {
    if (node < 0) return false;
    for (int k = 0; k < signals.num_signals; ++k) {
        if (signals.signal_node[k] == node) return true;
    }
    return false;
}

// ============================================================
// Car-following / driver model
// ============================================================

__device__ __forceinline__ float desired_speed_ecs(
    int id,
    int lane,
    const ECSArrays ecs,
    const RoadNetwork road
) {
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

    if (dtype == AV) {
        return clampf_cuda(limit * factor, 3.0f, 36.0f);
    }

    float human_factor = factor + 0.10f * (aggr - 0.5f);
    return clampf_cuda(limit * human_factor, 3.0f, 38.0f);
}

__device__ __forceinline__ float estimate_follow_accel_ecs(
    float v,
    float desired_v,
    float front_gap,
    float front_v,
    int dtype,
    float min_gap_i,
    float reaction_i,
    float comfort_decel_i,
    float aggressiveness_i,
    float risk_i
) {
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

    float s_star = s0 + fmaxf(
        0.0f,
        v * T + (v * dv) / fmaxf(2.0f * sqrt_ab, 0.1f)
    );

    float free_term = powf(v / fmaxf(desired_v, 0.1f), 4.0f);

    float interact = 0.0f;
    if (front_gap < 1.0e8f) {
        interact = powf(s_star / fmaxf(front_gap, 0.5f), 2.0f);
    }

    float a = max_accel * (1.0f - free_term - interact);
    return clampf_cuda(a, -EMERGENCY_DECEL, max_accel);
}


__device__ __forceinline__ float relative_closing_ttc_accel_limit_ecs(
    int self,
    int other,
    ECSArrays ecs,
    float horizon,
    float* metrics
) {
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

    float combined =
        0.5f * ecs.length[self]
        + 0.5f * ecs.length[other]
        + MIN_BUMPER_GAP;

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
        atomicAdd(&metrics[METRIC_INTERACTION_BRAKE], 1.0f);
    }

    return clampf_cuda(limit, -EMERGENCY_DECEL, 0.0f);
}

__device__ __forceinline__ float interaction_accel_limit_ecs(
    int self,
    int lane,
    int next_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float* metrics
) {
    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return 1000.0f;

    bool human = ecs.driver_type[self] == HUMAN;
    float max_range = human ? DIRECTIONAL_SIDE_RANGE_HUMAN : DIRECTIONAL_SIDE_RANGE_AV;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    int cr = clampi_cuda(
        (int)ceilf(max_range / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    float best_limit = 1000.0f;

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_connector = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_in_connector ? ecs.connector_from_lane[j] : ecs.lane_id[j];

                    if (
                        other_lane >= 0
                        && other_lane < road.num_lanes
                        && directional_vehicle_conflict_relevant_ecs(
                            self,
                            lane,
                            next_lane,
                            j,
                            ecs,
                            road
                        )
                    ) {
                        float rx = ecs.x[j] - ecs.x[self];
                        float ry = ecs.y[j] - ecs.y[self];
                        float dist = sqrtf(fmaxf(rx * rx + ry * ry, 0.001f));

                        float attention_range = directional_attention_range_ecs(
                            lane,
                            other_lane,
                            ecs.driver_type[self],
                            road
                        );

                        if (other_in_connector) {
                            attention_range = fmaxf(attention_range, 26.0f);
                        }

                        if (dist <= attention_range) {
                            float limit = relative_closing_ttc_accel_limit_ecs(
                                self,
                                j,
                                ecs,
                                2.6f,
                                metrics
                            );
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


__device__ __forceinline__ bool local_same_path_following_ecs(
    int a,
    int b,
    ECSArrays ecs
) {
    if (ecs.vehicle_state[a] == VEH_ON_LANE && ecs.vehicle_state[b] == VEH_ON_LANE) {
        if (ecs.lane_id[a] == ecs.lane_id[b]) return true;
        if (ecs.lane_change_active[a] != 0 && (ecs.lane_change_from_lane[a] == ecs.lane_id[b] || ecs.lane_change_to_lane[a] == ecs.lane_id[b])) return true;
        if (ecs.lane_change_active[b] != 0 && (ecs.lane_change_from_lane[b] == ecs.lane_id[a] || ecs.lane_change_to_lane[b] == ecs.lane_id[a])) return true;
    }
    if (ecs.vehicle_state[a] == VEH_IN_CONNECTOR && ecs.vehicle_state[b] == VEH_IN_CONNECTOR) {
        return ecs.connector_from_lane[a] == ecs.connector_from_lane[b]
            && ecs.connector_to_lane[a] == ecs.connector_to_lane[b];
    }
    return false;
}

__device__ __forceinline__ bool local_front_clear_for_id_ecs(
    int id,
    PerceptionSoA perception,
    ECSArrays ecs
) {
    if (id < 0) return true;
    float fg = perception.front_gap != nullptr ? perception.front_gap[id] : 1.0e9f;
    if (!isfinite(fg)) fg = 0.0f;
    float needed = fmaxf(
        FRONT_CLEAR_PRIORITY_MIN_GAP,
        fmaxf(ecs.length[id], 4.0f) + MIN_BUMPER_GAP + fmaxf(0.0f, ecs.speed[id]) * FRONT_CLEAR_PRIORITY_TIME
    );
    return fg > needed;
}

__device__ __forceinline__ int local_avoidance_priority_key_ecs(
    int id,
    PerceptionSoA perception,
    ECSArrays ecs,
    RoadNetwork road
) {
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
        float behavior = clampf_cuda(
            0.55f * ecs.aggressiveness[id]
            + 0.35f * ecs.risk_tolerance[id]
            + 0.10f * (1.0f - ecs.politeness[id]),
            0.0f,
            1.0f
        );
        key -= (int)(behavior * 18.0f);
    }
    return key;
}

__device__ __forceinline__ float local_obstacle_avoidance_accel_limit_ecs(
    int self,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    PerceptionSoA perception,
    int max_entities,
    float dt,
    float* metrics
) {
    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return 1000.0f;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(LOCAL_AVOID_RANGE / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

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

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
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
                        float combined =
                            0.36f * (fmaxf(ecs.length[self], 3.0f) + fmaxf(ecs.length[j], 3.0f))
                            + 0.50f * fmaxf(fmaxf(ecs.width[self], 1.4f), fmaxf(ecs.width[j], 1.4f))
                            + LOCAL_AVOID_COLLISION_MARGIN;
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
        atomicAdd(&metrics[METRIC_ANTI_COLLISION_BRAKE], 1.0f);
        atomicAdd(&metrics[METRIC_INTERACTION_BRAKE], 1.0f);
    }
    return best_limit;
}

__device__ __forceinline__ float intersection_conflict_accel_limit_ecs(
    int self,
    int lane,
    int next_lane,
    float dist_to_end,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float current_time,
    float* metrics
) {
    if (next_lane < 0 || next_lane >= road.num_lanes) return 1000.0f;
    if (dist_to_end > INTERSECTION_APPROACH_RANGE) return 1000.0f;

    int node = road.lane_end_node[lane];
    if (node < 0) return 1000.0f;

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return 1000.0f;

    float self_arrival = current_time + dist_to_end / fmaxf(ecs.speed[self], 0.8f);
    bool human = ecs.driver_type[self] == HUMAN;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    int cr = clampi_cuda(
        (int)ceilf(DIRECTIONAL_SIDE_RANGE_HUMAN / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    float best_limit = 1000.0f;

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_intersection = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_from = other_in_intersection ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_in_intersection
                        ? ecs.connector_to_lane[j]
                        : route_next_lane_for_vehicle_ecs(j, ecs, road);

                    if (other_from >= 0 && other_from < road.num_lanes) {
                        int other_node = road.lane_end_node[other_from];

                        if (
                            other_node == node
                            && intersection_conflict_relevant_vehicles_ecs(
                                self,
                                lane,
                                next_lane,
                                j,
                                other_from,
                                other_next,
                                other_in_intersection,
                                ecs,
                                road
                            )
                        ) {
                            float other_dist = 0.0f;
                            float other_arrival = current_time;

                            if (other_in_intersection) {
                                other_arrival = current_time - 0.25f;
                            } else {
                                other_dist = fmaxf(0.0f, road.lane_length[other_from] - ecs.s[j]);

                                float attention_range = directional_attention_range_ecs(
                                    lane,
                                    other_from,
                                    ecs.driver_type[self],
                                    road
                                );
                                attention_range = fminf(attention_range, INTERSECTION_APPROACH_RANGE);

                                if (other_dist <= attention_range) {
                                    other_arrival = current_time + other_dist / fmaxf(ecs.speed[j], 0.8f);
                                } else {
                                    other_arrival = 1.0e9f;
                                }
                            }

                            /* A fully stopped car far from its own stop line is not an
                               immediate crossing hazard.  Without this guard, one stalled
                               side-street car could freeze the whole intersection approach. */
                            if (
                                !other_in_intersection
                                && ecs.speed[j] < DIRECTIONAL_OTHER_STOP_EPS
                                && other_dist > fmaxf(12.0f, dist_to_end + 4.0f)
                            ) {
                                other_arrival = 1.0e9f;
                            }

                            if (other_arrival < 1.0e8f) {
                                float dt_arrival = other_arrival - self_arrival;
                                bool other_priority =
                                    other_in_intersection
                                    || dt_arrival < -INTERSECTION_PRIORITY_EPS
                                    || (fabsf(dt_arrival) < INTERSECTION_TIME_WINDOW && j < self);

                                if (other_priority) {
                                    float stop_dist = fmaxf(dist_to_end - INTERSECTION_STOP_BUFFER, 0.75f);
                                    float req = -(ecs.speed[self] * ecs.speed[self]) /
                                        fmaxf(2.0f * stop_dist, 0.5f);

                                    /* Keep the yield finite and avoid forcing a car that is
                                       already creeping with a clear front gap into a permanent
                                       zero-acceleration state. */
                                    req = clampf_cuda(req, -EMERGENCY_DECEL, -0.03f);
                                    req = fminf(req, -0.35f * max_decel);
                                    best_limit = fminf(best_limit, req);

                                    if (metrics != nullptr) {
                                        atomicAdd(&metrics[METRIC_CONFLICT_YIELD], 1.0f);
                                        atomicAdd(&metrics[METRIC_COOP_YIELD], 1.0f);
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

__device__ __forceinline__ bool approach_from_self_right_ecs(
    int self_lane,
    int other_lane,
    const RoadNetwork road
) {
    if (self_lane < 0 || other_lane < 0) return false;
    float sx, sy, ox, oy;
    lane_dir(self_lane, road, sx, sy);
    lane_dir(other_lane, road, ox, oy);

    /* Incoming lane directions point toward the same node.  For a vehicle whose
       forward vector is `self`, an approach coming from its physical right has
       the other incoming vector rotated counter-clockwise from self, so the
       cross product is positive. */
    float cross = sx * oy - sy * ox;
    return cross > UNSIGNAL_RIGHT_PRIORITY_CROSS;
}

__device__ __forceinline__ int turn_priority_rank_unsignal_ecs(int turn) {
    if (turn == TURN_STRAIGHT) return 0;
    if (turn == TURN_RIGHT) return 1;
    if (turn == TURN_LEFT) return 2;
    return 1;
}

__device__ __forceinline__ unsigned int deadlock_release_score_ecs(
    int id,
    int node,
    float current_time
) {
    int slot = (int)floorf(current_time / fmaxf(DEADLOCK_RELEASE_PERIOD, 0.25f));
    unsigned int x = (unsigned int)id * 747796405u;
    x ^= (unsigned int)node * 2891336453u;
    x ^= (unsigned int)slot * 277803737u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return x;
}

__device__ __forceinline__ bool unsignal_other_has_priority_ecs(
    int self,
    int other,
    int self_lane,
    int self_next_lane,
    int other_lane,
    int other_next_lane,
    bool other_in_connector,
    float self_arrival,
    float other_arrival,
    const RoadNetwork road
) {
    if (other_in_connector) return true;

    int self_turn = turn_code_from_lanes_ecs(self_lane, self_next_lane, road);
    int other_turn = turn_code_from_lanes_ecs(other_lane, other_next_lane, road);

    bool other_from_right = approach_from_self_right_ecs(self_lane, other_lane, road);
    bool self_from_right  = approach_from_self_right_ecs(other_lane, self_lane, road);

    float dt_arrival = other_arrival - self_arrival;

    float dot = lane_heading_dot_ecs(self_lane, other_lane, road);
    bool crossing_straight_pair =
        self_turn == TURN_STRAIGHT
        && other_turn == TURN_STRAIGHT
        && fabsf(dot) <= DIRECTIONAL_SIDE_DOT_ABS;

    /*
        EN: Unsignal priority rule, closer to real road behavior.
            1) Straight movement has the highest priority over turning movements.
            2) If both vehicles are straight and their approaches are orthogonal,
               the vehicle coming from the right has priority.
            3) For other ambiguous unsignal conflicts, keep right-hand priority,
               then opposing left-turn-yield, then arrival order.

        KO: 무신호 우선권 규칙을 실제 도로 동작에 가깝게 정리합니다.
            1) 직진은 회전 차량보다 최우선입니다.
            2) 둘 다 직진이고 접근 방향이 직교하면 우측 차량이 우선입니다.
            3) 그 외 애매한 충돌은 우측 우선, 마주 오는 좌회전 양보, 도착순으로
               정리합니다.
    */
    if (self_turn == TURN_STRAIGHT && other_turn != TURN_STRAIGHT) return false;
    if (other_turn == TURN_STRAIGHT && self_turn != TURN_STRAIGHT) {
        return dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW + 0.85f;
    }

    if (crossing_straight_pair) {
        if (other_from_right && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
        if (self_from_right && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;
    }

    /* Main unsignal rule: yield to the approach on your right, but do not wait
       forever for a vehicle that will arrive much later. */
    if (other_from_right && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
    if (self_from_right && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;

    /* For opposing approaches, left turns yield to straight/right movements. */
    if (dot < DIRECTIONAL_ONCOMING_DOT) {
        int sr = turn_priority_rank_unsignal_ecs(self_turn);
        int orr = turn_priority_rank_unsignal_ecs(other_turn);
        if (sr > orr && dt_arrival <= UNSIGNAL_RIGHT_PRIORITY_WINDOW) return true;
        if (sr < orr && dt_arrival >= -UNSIGNAL_RIGHT_PRIORITY_WINDOW) return false;
    }

    /* Fallback: arrival order, then deterministic id tie-break. */
    if (dt_arrival < -UNSIGNAL_ARRIVAL_EPS) return true;
    if (dt_arrival >  UNSIGNAL_ARRIVAL_EPS) return false;
    return other < self;
}

__device__ __forceinline__ float unsignal_priority_accel_limit_ecs(
    int self,
    int lane,
    int next_lane,
    float dist_to_end,
    float front_gap,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float current_time,
    float dt,
    float* metrics,
    bool* out_blocked,
    bool* out_release,
    bool* out_conflict_seen
) {
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

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return 1000.0f;

    float self_arrival = current_time +
        fmaxf(0.0f, dist_to_end - DEFAULT_STOP_OFFSET) / fmaxf(ecs.speed[self], 0.8f);

    bool human = ecs.driver_type[self] == HUMAN;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;
    float patience = human ? DEADLOCK_PATIENCE_HUMAN : DEADLOCK_PATIENCE_AV;
    if (indicator_active_ecs(self, ecs)) {
        // EN: A signaled vehicle has declared its path; if that path is physically clear,
        //     release deadlocks earlier so the queue can drain.
        // KO: 깜빡이를 켠 차량은 경로 의도를 공개했으므로 실제 경로가 비어 있으면
        //     교착 상태에서 조금 더 빨리 빠져나오게 합니다.
        patience *= DEADLOCK_INDICATOR_PATIENCE_SCALE;
    } else {
        patience *= DEADLOCK_ESCAPE_PATIENCE_SCALE;
    }

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(UNSIGNAL_PRIORITY_APPROACH_RANGE / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

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

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_in_connector = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_in_connector ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_in_connector
                        ? ecs.connector_to_lane[j]
                        : route_next_lane_for_vehicle_ecs(j, ecs, road);

                    if (
                        other_lane >= 0 && other_lane < road.num_lanes
                        && road.lane_end_node[other_lane] == node
                        && intersection_conflict_relevant_vehicles_ecs(
                            self,
                            lane,
                            next_lane,
                            j,
                            other_lane,
                            other_next,
                            other_in_connector,
                            ecs,
                            road
                        )
                    ) {
                        float other_dist = 0.0f;
                        float other_arrival = current_time - 0.15f;

                        if (!other_in_connector) {
                            other_dist = fmaxf(0.0f, road.lane_length[other_lane] - ecs.s[j]);
                            float attention = directional_attention_range_ecs(
                                lane,
                                other_lane,
                                ecs.driver_type[self],
                                road
                            );
                            attention = fmaxf(attention, UNSIGNAL_PRIORITY_NEAR_LINE_DIST);
                            attention = fminf(attention, UNSIGNAL_PRIORITY_APPROACH_RANGE);
                            if (other_dist > attention) {
                                j = grid.cell_next[j];
                                guard++;
                                continue;
                            }
                            other_arrival = current_time +
                                fmaxf(0.0f, other_dist - DEFAULT_STOP_OFFSET) / fmaxf(ecs.speed[j], 0.8f);

                            /* A stopped vehicle still far from its own stop line should not
                               freeze the node; it will be handled when it creeps closer. */
                            if (
                                ecs.speed[j] < UNSIGNAL_STOPPED_EPS
                                && other_dist > UNSIGNAL_STOPPED_FAR_IGNORE_DIST
                                && other_dist > dist_to_end + 5.0f
                            ) {
                                j = grid.cell_next[j];
                                guard++;
                                continue;
                            }
                        }

                        bool other_priority = unsignal_other_has_priority_ecs(
                            self,
                            j,
                            lane,
                            next_lane,
                            other_lane,
                            other_next,
                            other_in_connector,
                            self_arrival,
                            other_arrival,
                            road
                        );

                        if (other_priority) {
                            must_yield = true;
                            conflict_count++;
                            if (metrics != nullptr) {
                                atomicAdd(&metrics[METRIC_UNSIGNAL_CONFLICT], 1.0f);
                                if (indicator_active_ecs(j, ecs) || indicator_active_ecs(self, ecs)) {
                                    atomicAdd(&metrics[METRIC_INDICATOR_CONFLICT_YIELD], 1.0f);
                                }
                            }
                            if (other_in_connector) connector_priority = true;
                            if (!other_in_connector && ecs.speed[j] > 1.15f && other_arrival <= self_arrival + 0.65f) moving_priority = true;
                        }

                        bool other_stopped_near =
                            !other_in_connector
                            && ecs.speed[j] < UNSIGNAL_STOPPED_EPS
                            && other_dist <= UNSIGNAL_PRIORITY_NEAR_LINE_DIST + 5.0f
                            && ecs.connector_length[j] > patience * 0.35f;

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
        if (wait_time <= 0.01f && metrics != nullptr) atomicAdd(&metrics[METRIC_UNSIGNAL_PRIORITY_GO], 1.0f);
        return 1000.0f;
    }

    if (out_blocked) *out_blocked = true;
    if (out_conflict_seen) *out_conflict_seen = conflict_count > 0;

    float clear_front = fmaxf(UNSIGNAL_RELEASE_FRONT_GAP, ecs.length[self] + MIN_BUMPER_GAP + 5.0f);
    bool front_clear = front_gap > clear_front;
    bool front_super_clear = front_gap > clear_front * FRONT_CLEAR_RELEASE_GAP_MULT;

    // EN: If this car can physically clear the node and the nominal priority
    //     owner is blocked, release it much earlier.  This is the explicit
    //     "front space goes first" deadlock breaker requested for congested
    //     networks.
    // KO: 이 차량 앞쪽이 실제로 비어 있고 명목상 우선권 차량이 막힌 상태라면
    //     훨씬 빠르게 내보냅니다. 정체 네트워크에서 필요한 "앞이 빈 차 우선"
    //     데드락 해소 규칙입니다.
    bool front_clear_assertive_release =
        wait_time >= patience * FRONT_CLEAR_ASSERTIVE_WAIT_SCALE
        && near_stop_line
        && front_super_clear
        && !connector_priority
        && !moving_priority;

    bool front_empty_release =
        wait_time >= fmaxf(0.06f, patience * FRONT_EMPTY_RELEASE_WAIT_SCALE)
        && near_stop_line
        && front_clear
        && !connector_priority
        && (best_release_id == self || front_super_clear);

    bool extended_wait_release =
        wait_time >= patience * 1.25f
        && near_stop_line
        && front_clear
        && !connector_priority
        && (best_release_id == self || front_super_clear);

    bool deadlock_candidate =
        (
            wait_time >= patience
            && near_stop_line
            && front_clear
            && !moving_priority
            && !connector_priority
            && (best_release_id == self || front_clear_assertive_release)
        )
        || extended_wait_release
        || front_empty_release
        || front_clear_assertive_release;

    if (deadlock_candidate) {
        if (out_release) *out_release = true;
        ecs.connector_length[self] = fminf(wait_time, patience + 0.5f);
        if (metrics != nullptr) {
            atomicAdd(&metrics[METRIC_DEADLOCK_RELEASE], 1.0f);
            atomicAdd(&metrics[METRIC_UNSIGNAL_PRIORITY_GO], 1.0f);
            atomicAdd(&metrics[METRIC_DEADLOCK_ESCAPE_GO], 1.0f);
            if (front_empty_release) atomicAdd(&metrics[METRIC_FRONT_SPACE_RELEASE], 1.0f);
            if (indicator_active_ecs(self, ecs)) atomicAdd(&metrics[METRIC_INDICATOR_PRIORITY_GO], 1.0f);
        }
        return 1000.0f;
    }

    if (metrics != nullptr) {
        atomicAdd(&metrics[METRIC_UNSIGNAL_RIGHT_YIELD], 1.0f);
        atomicAdd(&metrics[METRIC_DEADLOCK_WAIT], wait_time > 0.0f ? dt : 0.0f);
    }

    float stop_dist = fmaxf(dist_to_end - DEFAULT_STOP_OFFSET - INTERSECTION_STOP_BUFFER, 0.75f);
    float req = -(ecs.speed[self] * ecs.speed[self]) / fmaxf(2.0f * stop_dist, 0.5f);
    req = clampf_cuda(req, -EMERGENCY_DECEL, -0.05f);
    req = fminf(req, -0.30f * max_decel);
    return req;
}

__device__ __forceinline__ int connector_entry_unique_priority_key_ecs(
    int id,
    int lane,
    int next_lane,
    ECSArrays ecs,
    RoadNetwork road
) {
    /*
        EN: Unique per-vehicle entry priority used only as a final deadlock
            breaker at the connector mouth.  Lower key wins.  It combines:
            - longer waiting time, stored in connector_length while queued,
            - shorter distance to the stop line,
            - turn rank,
            - vehicle id as a deterministic non-overlapping tie-breaker.

        KO: connector 입구에서 최종 deadlock을 깨기 위한 차량별 유일 우선순위입니다.
            작은 key가 먼저 갑니다. 구성 요소는 다음과 같습니다.
            - 대기 시간(connector_length에 임시 누적),
            - 정지선까지 짧은 거리,
            - 회전 종류 rank,
            - 차량 id deterministic tie-breaker.
    */
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
        float behavior = clampf_cuda(
            0.55f * ecs.aggressiveness[id]
            + 0.35f * ecs.risk_tolerance[id]
            + 0.10f * (1.0f - ecs.politeness[id]),
            0.0f,
            1.0f
        );
        behavior_credit = (int)(behavior * 16.0f);
    }
    int raw = 20000 + dist_part + turn_rank - wait_credit - behavior_credit;
    raw = clampi_cuda(raw, 0, 30000);
    return (raw << PRIORITY_GATE_ID_BITS) | (id & PRIORITY_GATE_ID_MASK);
}


__device__ __forceinline__ bool connector_exit_space_clear_ecs(
    int self,
    int from_lane,
    int to_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities
) {
    /*
        EN: Do not enter a connector if the receiving lane immediately after the
            handoff is already occupied.  This moves queues to the stop line and
            prevents cars from stopping in the middle of the intersection box.
        KO: handoff 직후의 진출 차로가 막혀 있으면 connector에 들어가지 않습니다.
            대기열을 교차로 안이 아니라 정지선 쪽에 만들기 위한 장치입니다.
    */
    if (!valid_lane_ecs(from_lane, road) || !valid_lane_ecs(to_lane, road)) return true;

    float hx, hy;
    float handoff_s = connector_exit_handoff_s(from_lane, to_lane, road);
    handoff_s = clampf_cuda(handoff_s, 0.0f, fmaxf(road.lane_length[to_lane], 0.1f));
    lane_xy_from_s(to_lane, handoff_s, road, hx, hy);

    int base = world_cell_index(hx, hy, grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return true;

    float v = fmaxf(0.0f, ecs.speed[self]);
    float required = fmaxf(
        CONNECTOR_EXIT_SPACE_MIN,
        fmaxf(PRIORITY_GATE_EXIT_SPACE, ecs.length[self] + MIN_BUMPER_GAP + v * CONNECTOR_EXIT_SPACE_TIME)
    );

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf((required + 8.0f) / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool on_surface = ecs.lane_id[j] == to_lane;
                    if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR) {
                        on_surface = on_surface || ecs.connector_from_lane[j] == to_lane || ecs.connector_to_lane[j] == to_lane;
                    }
                    if (ecs.lane_change_active[j] != 0) {
                        on_surface = on_surface || ecs.lane_change_from_lane[j] == to_lane || ecs.lane_change_to_lane[j] == to_lane;
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


__device__ __forceinline__ bool zipper_merge_same_receiving_lanes_ecs(
    int a_next,
    int b_next,
    RoadNetwork road
) {
#if ZIPPER_MERGE_ENABLED
    if (!valid_lane_ecs(a_next, road) || !valid_lane_ecs(b_next, road)) return false;
    if (a_next == b_next) return true;
    return same_approach_same_direction_lanes_ecs(a_next, b_next, road);
#else
    return false;
#endif
}

__device__ __forceinline__ bool zipper_merge_self_yields_ecs(
    int self,
    int lane,
    int next_lane,
    int other,
    int other_lane,
    int other_next_lane,
    ECSArrays ecs,
    RoadNetwork road,
    float current_time
) {
#if ZIPPER_MERGE_ENABLED
    if (!valid_lane_ecs(lane, road) || !valid_lane_ecs(other_lane, road)) return false;
    if (!zipper_merge_same_receiving_lanes_ecs(next_lane, other_next_lane, road)) return false;
    if (!same_approach_same_direction_lanes_ecs(lane, other_lane, road) && road.lane_end_node[lane] != road.lane_end_node[other_lane]) return false;

    float dist_self = fmaxf(0.0f, road.lane_length[lane] - ecs.s[self]);
    float dist_other = fmaxf(0.0f, road.lane_length[other_lane] - ecs.s[other]);
    bool one_near_merge = dist_self <= ZIPPER_MERGE_RANGE || dist_other <= ZIPPER_MERGE_RANGE;
    if (!one_near_merge) return false;

    // KO: 병목/합류부에서는 해당 진입 차선 끝에 더 가까운 차량이 먼저 들어갑니다.
    // EN: At a bottleneck/merge mouth, the vehicle closer to its own lane end gets the slot first.
    if (dist_other + ZIPPER_MERGE_CLOSER_EPS < dist_self) return true;
    if (dist_self + ZIPPER_MERGE_CLOSER_EPS < dist_other) return false;

    // KO: 거리 차이가 거의 없으면 한국식 "한 대씩" 합류를 위해 차선별 슬롯을 교대합니다.
    // EN: Near-ties alternate lanes to approximate a one-by-one zipper merge.
    int lo_lane = lane < other_lane ? lane : other_lane;
    int hi_lane = lane < other_lane ? other_lane : lane;
    float period = fmaxf(ZIPPER_MERGE_ALTERNATE_PERIOD, 0.10f);
    uint32_t slot = (uint32_t)floorf(fmaxf(0.0f, current_time) / period);
    int preferred_lane = ((slot & 1u) == 0u) ? lo_lane : hi_lane;
    if (lane != other_lane) {
        if (lane != preferred_lane) return true;
        return false;
    }

    // Same physical lane should not normally get here. Keep a deterministic tie-breaker.
    return self > other;
#else
    return false;
#endif
}

__device__ __forceinline__ bool connector_entry_clear_ecs(
    int self,
    int lane,
    int next_lane,
    ECSArrays ecs,
    DecisionSoA decision,
    RoadNetwork road,
    SpatialGrid grid,
    float current_time,
    int max_entities
) {
    int node = road.lane_end_node[lane];
    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return true;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(CONNECTOR_ENTRY_CLEAR_RADIUS / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    if (!connector_exit_space_clear_ecs(self, lane, next_lane, ecs, road, grid, max_entities)) {
        return false;
    }

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_conn = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_from = other_conn ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_conn ? ecs.connector_to_lane[j] : route_next_lane_for_vehicle_ecs(j, ecs, road);

                    if (other_from >= 0 && other_from < road.num_lanes && road.lane_end_node[other_from] == node) {
                        bool relevant = intersection_conflict_relevant_vehicles_ecs(
                            self,
                            lane,
                            next_lane,
                            j,
                            other_from,
                            other_next,
                            other_conn,
                            ecs,
                            road
                        );

                        float dxp = ecs.x[j] - ecs.x[self];
                        float dyp = ecs.y[j] - ecs.y[self];
                        float d = sqrtf(fmaxf(dxp * dxp + dyp * dyp, 0.001f));

                        if (other_conn && relevant) {
                            float other_len = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                            float other_s = clampf_cuda(ecs.connector_s[j], 0.0f, other_len);
                            bool active_path_already_clear =
                                other_s >= other_len * PRIORITY_GATE_ACTIVE_CLEAR_FRACTION
                                || (other_len - other_s) <= PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST;

                            // EN: Do not block the whole node for a connector that has already
                            //     cleared the crossing zone.  The same-path/front-gap guard still
                            //     prevents rear-end collisions.
                            // KO: 이미 충돌 지점을 지나 출구를 비우는 connector 때문에 노드 전체를
                            //     막지 않습니다. 같은 경로/앞차 gap 검사는 계속 추돌을 막습니다.
                            if (!active_path_already_clear) {
                                return false;
                            }
                        }

                        if (
                            !other_conn
                            && decision.wants_connector[j] != 0
                            && d < CONNECTOR_ENTRY_CLEAR_RADIUS
                        ) {
                            bool same_receiving_merge = zipper_merge_same_receiving_lanes_ecs(
                                next_lane,
                                other_next,
                                road
                            );
                            bool merge_relevant = relevant || same_receiving_merge;
                            if (merge_relevant) {
                                if (same_receiving_merge) {
                                    bool self_yields = zipper_merge_self_yields_ecs(
                                        self,
                                        lane,
                                        next_lane,
                                        j,
                                        other_from,
                                        other_next,
                                        ecs,
                                        road,
                                        current_time
                                    );
                                    bool other_yields = zipper_merge_self_yields_ecs(
                                        j,
                                        other_from,
                                        other_next,
                                        self,
                                        lane,
                                        next_lane,
                                        ecs,
                                        road,
                                        current_time
                                    );
                                    if (self_yields && !other_yields) {
                                        return false;
                                    }
                                    if (!self_yields && other_yields) {
                                        j = grid.cell_next[j];
                                        guard++;
                                        continue;
                                    }
                                }

                                int self_key = connector_entry_unique_priority_key_ecs(
                                    self, lane, next_lane, ecs, road
                                );
                                int other_key = connector_entry_unique_priority_key_ecs(
                                    j, other_from, other_next, ecs, road
                                );

                                /*
                                    EN: Final simultaneous-entry rule. For regular crossing
                                        conflicts this unique key removes all-yield cycles. For
                                        same receiving-lane bottlenecks the zipper helper above
                                        first applies closer-car-first / one-by-one alternation,
                                        then this key only breaks true ties.

                                    KO: 최종 동시 진입 규칙입니다. 일반 교차 충돌은 유일 key로
                                        모두가 멈추는 cycle을 끊고, 같은 진출 차로로 합류하는
                                        병목은 위의 zipper 규칙이 먼저 "가까운 차 우선/한 대씩"을
                                        적용한 뒤 이 key는 완전 동률만 정리합니다.
                                */
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

// ============================================================
// Intersection Priority Gate System
// ============================================================

__device__ __forceinline__ bool intersection_priority_context_ecs(
    int id,
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    float current_time,
    int& lane,
    int& next_lane,
    int& turn,
    int& node,
    float& dist_to_end,
    bool& signal_permits
) {
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

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
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
        // EN: A yellow-light vehicle may continue only if it is already so close
        //     that a comfortable stop is no longer realistic.
        // KO: 황색 신호에서는 정지선에 이미 너무 가까워 편안한 정지가 어려울 때만
        //     우선순위 게이트 후보로 둡니다.
        float v = fmaxf(ecs.speed[id], 0.0f);
        float stop_need = v * fmaxf(ecs.reaction_time[id], 0.25f) + (v * v) / (2.0f * 3.0f) + 2.0f;
        signal_permits = dist_to_end <= stop_need;
    }
    if (st == LIGHT_RED) signal_permits = false;

    // EN: If the vehicle reference point is already inside the lane-width
    //     intersection box, stopping for a signal/priority rule is more dangerous
    //     than clearing the box.  Let the priority gate order in-box vehicles by
    //     unique keys and then force them out.
    // KO: 차량 기준점이 이미 차로 폭 기반 교차로 박스 안에 들어왔다면 신호/우선순위로
    //     그 자리에 멈추는 것이 더 위험합니다. 우선순위 게이트가 박스 안 차량을
    //     유일 key로 정렬한 뒤 빠져나가게 합니다.
    if (inside_box) signal_permits = true;

    return signal_permits;
}

__device__ __forceinline__ float priority_front_clear_gap_ecs(
    int id,
    ECSArrays ecs
) {
    if (id < 0) return FRONT_CLEAR_PRIORITY_MIN_GAP;
    float v = fmaxf(0.0f, ecs.speed[id]);
    float len = fmaxf(ecs.length[id], 4.0f);
    return fmaxf(FRONT_CLEAR_PRIORITY_MIN_GAP, len + MIN_BUMPER_GAP + v * FRONT_CLEAR_PRIORITY_TIME);
}

__device__ __forceinline__ bool priority_front_clear_ecs(
    int id,
    PerceptionSoA perception,
    ECSArrays ecs
) {
    if (id < 0) return true;
    float fg = perception.front_gap != nullptr ? perception.front_gap[id] : 1.0e9f;
    if (!isfinite(fg)) fg = 0.0f;
    return fg > priority_front_clear_gap_ecs(id, ecs);
}

__device__ __forceinline__ int priority_gate_key_ecs(
    int id,
    int lane,
    int next_lane,
    int turn,
    float dist_to_end,
    PerceptionSoA perception,
    ECSArrays ecs,
    RoadNetwork road
) {
    // EN: Lower key wins.  Waiting time dominates, then whether the forward
    //     path is actually clear, then distance/turn class.  This makes the car
    //     that can immediately clear the node go first instead of freezing behind
    //     a blocked candidate.
    // KO: 작은 key가 먼저 통과합니다. 대기 시간 다음으로 "앞이 비었는지"를
    //     크게 반영합니다. 그래서 막힌 후보 뒤에서 노드 전체가 얼어붙지 않고,
    //     실제로 빠져나갈 수 있는 차량이 먼저 나갑니다.
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
        float behavior =
            0.58f * clampf_cuda(ecs.aggressiveness[id], 0.0f, 1.0f)
            + 0.42f * clampf_cuda(ecs.risk_tolerance[id], 0.0f, 1.0f)
            - 0.36f * clampf_cuda(ecs.politeness[id], 0.0f, 1.0f);
        int behavior_bias = clampi_cuda((int)floorf((0.48f - behavior) * PRIORITY_GATE_BEHAVIOR_BIAS_SCALE), -4, 4);
        key += behavior_bias;
    }

    if (indicator_active_ecs(id, ecs) && turn != TURN_STRAIGHT) {
        key -= 1;
    }

    key = clampi_cuda(key + turn_bias, 0, 2047);
    return key;
}

__global__ void clear_intersection_priority_gate_kernel(
    int* priority_table,
    int num_nodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || priority_table == nullptr) return;

    int base = node * PRIORITY_GATE_SLOT_STRIDE;
    priority_table[base + PRIORITY_GATE_SLOT_BEST] = PRIORITY_GATE_EMPTY;
    priority_table[base + PRIORITY_GATE_SLOT_OCCUPIED] = 0;
    priority_table[base + PRIORITY_GATE_SLOT_COUNT] = 0;
    priority_table[base + PRIORITY_GATE_SLOT_GRANTED] = -1;
}

__global__ void mark_intersection_occupancy_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    int* priority_table,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || priority_table == nullptr) return;
    if (ecs.alive[i] != ENTITY_ALIVE || ecs.vehicle_state[i] != VEH_IN_CONNECTOR) return;

    int from_lane = ecs.connector_from_lane[i];
    if (!valid_lane_ecs(from_lane, road)) return;

    int node = road.lane_end_node[from_lane];
    if (node < 0 || node >= road.num_nodes) return;

    int base = node * PRIORITY_GATE_SLOT_STRIDE;
    atomicAdd(&priority_table[base + PRIORITY_GATE_SLOT_OCCUPIED], 1);
}

__global__ void select_intersection_priority_candidates_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    DecisionSoA decision,
    PerceptionSoA perception,
    int* priority_table,
    float* metrics,
    float current_time,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || priority_table == nullptr) return;

    int lane, next_lane, turn, node;
    float dist_to_end;
    bool signal_permits;
    if (!intersection_priority_context_ecs(
            i, ecs, road, signals, current_time,
            lane, next_lane, turn, node, dist_to_end, signal_permits)) {
        return;
    }

    // EN: Only vehicles that are actually trying to clear the node are candidates.
    //     Cars far upstream still receive normal car-following/lane-change logic.
    // KO: 실제로 노드를 비우려는 차량만 후보입니다. 너무 뒤쪽 차량은 일반 추종/차선변경
    //     로직에 맡깁니다.
    if (dist_to_end > PRIORITY_GATE_APPROACH_RANGE) return;

    int key = priority_gate_key_ecs(i, lane, next_lane, turn, dist_to_end, perception, ecs, road);
    int packed = (key << PRIORITY_GATE_ID_BITS) | (i & PRIORITY_GATE_ID_MASK);

    int base = node * PRIORITY_GATE_SLOT_STRIDE;
    atomicMin(&priority_table[base + PRIORITY_GATE_SLOT_BEST], packed);
    atomicAdd(&priority_table[base + PRIORITY_GATE_SLOT_COUNT], 1);

    if (metrics != nullptr) atomicAdd(&metrics[METRIC_PRIORITY_GATE_CANDIDATE], 1.0f);
}


__device__ __forceinline__ bool priority_gate_path_blocked_ecs(
    int self,
    int lane,
    int next_lane,
    int turn,
    int node,
    float dist_to_end,
    int self_packed,
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    SpatialGrid grid,
    PerceptionSoA perception,
    int max_entities,
    float current_time,
    float dt,
    float* metrics,
    bool* out_active_hold,
    bool* out_candidate_hold,
    bool* out_conflict_free
) {
    /*
        EN: Conflict-aware intersection gate.

        The previous gate locked the whole node whenever any vehicle was inside
        or more than one candidate was waiting.  That caused fake deadlocks: a
        vehicle with an empty, non-overlapping path was forced to wait for an
        unrelated connector.  This helper compares only vehicles whose swept
        lane-surface paths actually overlap.  Non-conflicting movements may
        proceed together, while conflicting movements are sorted by a unique
        priority key.

        KO: 충돌 경로 기반 교차로 게이트입니다.

        이전 게이트는 교차로 안에 차량이 하나만 있어도 노드 전체를 잠가서, 실제로
        겹치지 않는 경로와 앞이 빈 차량까지 기다리게 만들었습니다. 이 함수는 도로
        면 위의 swept path가 실제로 겹치는 차량끼리만 비교합니다. 겹치지 않는 차량은
        동시에 빠져나가고, 겹치는 차량은 차량마다 유일한 우선순위 key로 정렬합니다.
    */
    if (out_active_hold) *out_active_hold = false;
    if (out_candidate_hold) *out_candidate_hold = false;
    if (out_conflict_free) *out_conflict_free = true;

    int base_cell = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base_cell < 0) return false;

    int bc_x = base_cell % grid.width;
    int bc_y = base_cell / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(PRIORITY_GATE_PATH_SCAN_RANGE / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    bool blocked = false;

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    bool other_conn = ecs.vehicle_state[j] == VEH_IN_CONNECTOR;
                    int other_lane = other_conn ? ecs.connector_from_lane[j] : ecs.lane_id[j];
                    int other_next = other_conn ? ecs.connector_to_lane[j] : -1;

                    if (valid_lane_ecs(other_lane, road) && road.lane_end_node[other_lane] == node) {
                        if (other_conn) {
                            other_next = ecs.connector_to_lane[j];
                            if (valid_lane_ecs(other_next, road)) {
                                bool relevant = intersection_conflict_relevant_vehicles_ecs(
                                    self,
                                    lane,
                                    next_lane,
                                    j,
                                    other_lane,
                                    other_next,
                                    true,
                                    ecs,
                                    road
                                );

                                if (relevant) {
                                    if (out_conflict_free) *out_conflict_free = false;

                                    float other_len = fmaxf(ecs.connector_length[j], CONNECTOR_MIN_LEN);
                                    float other_s = clampf_cuda(ecs.connector_s[j], 0.0f, other_len);
                                    bool active_path_already_clear =
                                        other_s >= other_len * PRIORITY_GATE_ACTIVE_CLEAR_FRACTION
                                        || (other_len - other_s) <= PRIORITY_GATE_ACTIVE_EXIT_CLEAR_DIST;

                                    if (!active_path_already_clear) {
                                        if (out_active_hold) *out_active_hold = true;
                                        blocked = true;
                                    }
                                }
                            }
                        } else {
                            int olane, onext, oturn, onode;
                            float odist;
                            bool osignal;
                            if (intersection_priority_context_ecs(
                                    j,
                                    ecs,
                                    road,
                                    signals,
                                    current_time,
                                    olane,
                                    onext,
                                    oturn,
                                    onode,
                                    odist,
                                    osignal
                                ) && onode == node) {
                                bool relevant = intersection_conflict_relevant_vehicles_ecs(
                                    self,
                                    lane,
                                    next_lane,
                                    j,
                                    olane,
                                    onext,
                                    false,
                                    ecs,
                                    road
                                );

                                if (relevant) {
                                    if (out_conflict_free) *out_conflict_free = false;

                                    bool pairwise_other_priority = false;
                                    bool unsignal_pair = !node_has_signal_ecs(node, signals);
                                    if (unsignal_pair) {
                                        float self_arrival = dist_to_end / fmaxf(ecs.speed[self], 1.0f);
                                        float other_arrival = odist / fmaxf(ecs.speed[j], 1.0f);
                                        pairwise_other_priority = unsignal_other_has_priority_ecs(
                                            self,
                                            j,
                                            lane,
                                            next_lane,
                                            olane,
                                            onext,
                                            false,
                                            self_arrival,
                                            other_arrival,
                                            road
                                        );
                                    }

                                    int other_key = priority_gate_key_ecs(j, olane, onext, oturn, odist, perception, ecs, road);
                                    int other_packed = (other_key << PRIORITY_GATE_ID_BITS) | (j & PRIORITY_GATE_ID_MASK);

                                    bool self_inside_box = inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
                                    bool other_inside_box = inside_intersection_box_ecs(odist, olane, onext, road);
                                    bool other_goes_first;
                                    if (self_inside_box && other_inside_box) {
                                        // EN: In-box deadlock breaker.  Right-hand priority can form a cycle when
                                        //     four vehicles are already inside/at the box.  At that point the safest
                                        //     rule is deterministic unique-key ordering, not another circular yield.
                                        // KO: 박스 내부 데드락 해소 규칙입니다. 이미 박스 안/입구에 있는 네 차량은
                                        //     우측 우선권만으로 cycle을 만들 수 있으므로, 유일 key 순서로 정렬합니다.
                                        other_goes_first = other_packed < self_packed;
                                    } else if (self_inside_box != other_inside_box) {
                                        // EN/KO: Vehicles already in the intersection box clear before upstream approaches.
                                        other_goes_first = other_inside_box;
                                    } else {
                                        float self_wait = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                        float other_wait = clampf_cuda(ecs.connector_length[j], 0.0f, 60.0f);
                                        bool timed_release_order =
                                            self_wait > 0.75f
                                            || other_wait > 0.75f
                                            || dist_to_end <= PRIORITY_GATE_NEAR_LINE_DIST
                                            || odist <= PRIORITY_GATE_NEAR_LINE_DIST;

                                        // EN: Right-hand priority is human-like while vehicles are still rolling,
                                        // but at the mouth of a congested node it can create a circular wait.
                                        // Switch to the unique time/distance key once either car has queued.
                                        // KO: 접근 중에는 우측 우선권을 따르지만, node 입구에서 대기열이
                                        // 생기면 순환 양보가 발생할 수 있습니다. 한쪽이라도 기다린 뒤에는
                                        // 시간/거리 기반 유일 key로 전환해 반드시 한 대가 빠져나가게 합니다.
                                        other_goes_first =
                                            (unsignal_pair && !timed_release_order)
                                            ? pairwise_other_priority
                                            : (other_packed < self_packed);
                                    }

                                    bool self_front_clear = priority_front_clear_ecs(self, perception, ecs);
                                    bool other_front_clear = priority_front_clear_ecs(j, perception, ecs);
                                    float self_wait_now = clampf_cuda(ecs.connector_length[self], 0.0f, 60.0f);
                                    float other_wait_now = clampf_cuda(ecs.connector_length[j], 0.0f, 60.0f);
                                    if (
                                        other_goes_first
                                        && self_front_clear
                                        && !other_front_clear
                                        && !other_inside_box
                                        && self_wait_now >= PRIORITY_GATE_BLOCKED_OTHER_IGNORE_WAIT
                                        && other_wait_now < self_wait_now + 2.0f
                                    ) {
                                        /* EN: Front-clear priority.  If another candidate has nominal priority
                                           but cannot leave because its own exit is blocked, let the clear car
                                           drain the node first.  This is the GPU equivalent of drivers taking
                                           an open gap instead of all four approaches waiting forever.
                                           KO: 앞이 빈 차량 우선. 명목상 우선권이 있는 후보가 자기 앞이 막혀
                                           빠져나가지 못하면, 앞이 빈 차량이 먼저 노드를 비우게 합니다. */
                                        other_goes_first = false;
                                    }

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
        atomicAdd(&metrics[METRIC_PRIORITY_PATH_BLOCK], 1.0f);
        if (out_active_hold && *out_active_hold) atomicAdd(&metrics[METRIC_PRIORITY_ACTIVE_PATH_HOLD], 1.0f);
        if (ecs.driver_type[self] == HUMAN) atomicAdd(&metrics[METRIC_HUMAN_AI_COURTESY_YIELD], 1.0f);
    }

    return blocked;
}

__global__ void apply_intersection_priority_gate_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    DecisionSoA decision,
    SpatialGrid grid,
    PerceptionSoA perception,
    int* priority_table,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || priority_table == nullptr) return;

    int lane, next_lane, turn, node;
    float dist_to_end;
    bool signal_permits;
    if (!intersection_priority_context_ecs(
            i, ecs, road, signals, current_time,
            lane, next_lane, turn, node, dist_to_end, signal_permits)) {
        return;
    }

    int base = node * PRIORITY_GATE_SLOT_STRIDE;
    int best = priority_table[base + PRIORITY_GATE_SLOT_BEST];
    int count = priority_table[base + PRIORITY_GATE_SLOT_COUNT];
    int occupied = priority_table[base + PRIORITY_GATE_SLOT_OCCUPIED];

    if (best == PRIORITY_GATE_EMPTY || count <= 0) return;

    int key = priority_gate_key_ecs(i, lane, next_lane, turn, dist_to_end, perception, ecs, road);
    int self_packed = (key << PRIORITY_GATE_ID_BITS) | (i & PRIORITY_GATE_ID_MASK);
    int winner_id = best & PRIORITY_GATE_ID_MASK;
    bool is_node_winner = ((i & PRIORITY_GATE_ID_MASK) == winner_id);

    bool active_path_hold = false;
    bool candidate_path_hold = false;
    bool conflict_free = true;
    bool path_blocked = priority_gate_path_blocked_ecs(
        i,
        lane,
        next_lane,
        turn,
        node,
        dist_to_end,
        self_packed,
        ecs,
        road,
        signals,
        grid,
        perception,
        max_entities,
        current_time,
        dt,
        metrics,
        &active_path_hold,
        &candidate_path_hold,
        &conflict_free
    );

    float gate_wait_time = ecs.connector_length[i];
    if (!isfinite(gate_wait_time) || gate_wait_time < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) gate_wait_time = 0.0f;
    bool gate_front_clear = priority_front_clear_ecs(i, perception, ecs);
    if (
        path_blocked
        && candidate_path_hold
        && !active_path_hold
        && gate_front_clear
        && gate_wait_time >= PRIORITY_GATE_FRONT_CLEAR_OVERRIDE_WAIT
    ) {
        // EN/KO: A queued car with a clear exit may take the gap instead of
        // waiting forever behind a blocked/hesitating candidate.
        path_blocked = false;
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_FRONT_SPACE_RELEASE], 1.0f);
    }

    /*
        EN: We no longer hold just because the node has an occupied connector.
        Hold only when that occupied connector or another candidate shares the
        same swept path.  This removes the common "empty road but deadlocked"
        failure while preserving collision prevention.

        KO: 이제 노드에 connector 차량이 있다는 이유만으로 전체를 막지 않습니다.
        실제 swept path가 겹치는 connector 또는 더 높은 우선순위 후보가 있을 때만
        정지합니다. 그래서 앞이 빈데도 멈추는 가짜 deadlock을 줄이고, 충돌 방지는
        유지합니다.
    */
    if (path_blocked) {
        float v = fmaxf(ecs.speed[i], 0.0f);
        float stop_dist = fmaxf(dist_to_end - PRIORITY_GATE_STOP_BUFFER, 0.60f);
        float req = -(v * v) / fmaxf(2.0f * stop_dist, 0.5f);
        req = clampf_cuda(req, -EMERGENCY_DECEL, -0.04f);

        // EN: Very polite human drivers brake a little earlier; aggressive ones
        //     stop later but still never cross a blocked conflict path.
        // KO: 예의성 높은 사람 운전자는 조금 더 일찍 감속하고, 공격적인 운전자는 조금
        //     늦게 서지만, 막힌 충돌 경로를 넘지는 않습니다.
        if (ecs.driver_type[i] == HUMAN) {
            float courtesy = clampf_cuda(ecs.politeness[i], 0.0f, 1.0f);
            req *= (1.0f + HUMAN_AI_COURTESY_HOLD_SCALE * courtesy);
        }

        decision.target_accel[i] = fminf(decision.target_accel[i], req);
        decision.wants_connector[i] = 0;
        decision.connector_target_lane[i] = -1;

        float wait_time = ecs.connector_length[i];
        if (!isfinite(wait_time) || wait_time < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) wait_time = 0.0f;
        if (dist_to_end <= PRIORITY_GATE_NEAR_LINE_DIST + 5.0f && ecs.speed[i] < 1.0f) {
            ecs.connector_length[i] = fminf(wait_time + dt, 45.0f);
        }

        if (metrics != nullptr) {
            atomicAdd(&metrics[METRIC_PRIORITY_GATE_BLOCKED], 1.0f);
            atomicAdd(&metrics[METRIC_ENTRY_QUEUE_HOLD], dt);
            if (active_path_hold || occupied > 0) atomicAdd(&metrics[METRIC_INTERSECTION_OCCUPIED_HOLD], 1.0f);
        }
        return;
    }

    atomicMax(&priority_table[base + PRIORITY_GATE_SLOT_GRANTED], i);

    bool human = ecs.driver_type[i] == HUMAN;
    float target_v = human ? PRIORITY_GATE_RELEASE_SPEED_HUMAN : PRIORITY_GATE_RELEASE_SPEED_AV;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;

    // EN: Human-like GPU driver policy.  A clear, non-overlapping path is taken
    //     more proactively by aggressive/risk-tolerant humans, while cautious
    //     humans still accelerate gently.  This is a per-vehicle AI controller
    //     executed on the GPU, not a CPU-side scripted queue.
    // KO: GPU에서 실행되는 사람다운 운전자 정책입니다. 실제로 비어 있고 경로가 겹치지
    //     않으면 공격적/위험 감수 성향의 사람 운전자는 조금 더 적극적으로 빠져나오고,
    //     신중한 운전자는 부드럽게 가속합니다. CPU 큐가 아니라 차량별 GPU AI입니다.
    float behavior = 0.5f;
    if (human) {
        behavior = clampf_cuda(
            0.52f * ecs.aggressiveness[i]
            + 0.38f * ecs.risk_tolerance[i]
            + 0.20f * (1.0f - ecs.politeness[i]),
            0.0f,
            1.0f
        );
        target_v += HUMAN_AI_ASSERTIVE_BOOST_HUMAN * behavior;
    } else {
        target_v += HUMAN_AI_ASSERTIVE_BOOST_AV;
    }

    float release_a = (target_v - ecs.speed[i]) / fmaxf(dt, 0.01f);
    float release_scale = human ? (0.32f + 0.28f * behavior) : PRIORITY_GATE_MAX_RELEASE_ACCEL;
    release_a = clampf_cuda(release_a, 0.0f, max_accel * release_scale);
    if (release_a > 0.0f) decision.target_accel[i] = fmaxf(decision.target_accel[i], release_a);

    float v_after = fmaxf(0.0f, ecs.speed[i] + decision.target_accel[i] * dt);
    float trigger = fmaxf(
        intersection_box_depth_ecs(lane, next_lane, road),
        v_after * dt + CONNECTOR_TRIGGER_MARGIN + 0.10f
    );
    if (dist_to_end <= trigger) {
        decision.wants_connector[i] = 1;
        decision.connector_target_lane[i] = next_lane;
    }

    if (metrics != nullptr) {
        atomicAdd(&metrics[METRIC_PRIORITY_GATE_GRANTED], 1.0f);
        if (conflict_free) atomicAdd(&metrics[METRIC_PRIORITY_CONFLICT_FREE_GO], 1.0f);
        if (count > 1 && !is_node_winner) atomicAdd(&metrics[METRIC_FRONT_SPACE_RELEASE], 1.0f);
        if (human && release_a > 0.0f) atomicAdd(&metrics[METRIC_HUMAN_AI_ASSERTIVE_GO], 1.0f);
        if (count > 1) {
            atomicAdd(&metrics[METRIC_UNIQUE_PRIORITY_TIE], 1.0f);
            if (is_node_winner || conflict_free) atomicAdd(&metrics[METRIC_DEADLOCK_PRIORITY_RELEASE], 1.0f);
        }
    }
}


// ============================================================
// Base kernels
// ============================================================

__global__ void clear_int_kernel(int* data, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

__global__ void clear_float_kernel(float* data, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

__global__ void clear_decision_kernel(
    DecisionSoA decision,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    decision.desired_speed[i] = 0.0f;
    decision.target_accel[i] = 0.0f;

    decision.wants_lane_change[i] = 0;
    decision.lane_change_target[i] = -1;

    decision.wants_connector[i] = 0;
    decision.connector_target_lane[i] = -1;

    decision.should_exit[i] = 0;
}

// ============================================================
// Spatial Hash System
// ============================================================

__global__ void spatial_hash_build_system(
    ECSArrays ecs,
    SpatialGrid grid,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities) return;

    grid.cell_next[i] = -1;

    if (ecs.alive[i] != ENTITY_ALIVE) return;

    int cell = world_cell_index(
        ecs.x[i],
        ecs.y[i],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (cell < 0) return;

    int old = atomicExch(&grid.cell_head[cell], i);
    grid.cell_next[i] = old;
}

// ============================================================
// Spawn System
// ============================================================

__device__ __forceinline__ void init_driver_personality_ecs(
    int id,
    int dtype,
    uint32_t& rs,
    ECSArrays ecs
) {
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
        /* Human drivers are still heterogeneous, but the default distribution is
           no longer biased toward dangerous gap acceptance. */
        ecs.aggressiveness[id] = clampf_cuda(0.18f + 0.48f * u1, 0.10f, 0.78f);
        ecs.politeness[id] = clampf_cuda(0.24f + 0.48f * u2, 0.12f, 0.82f);
        ecs.risk_tolerance[id] = clampf_cuda(0.12f + 0.52f * u3, 0.05f, 0.72f);
        ecs.comfort_decel[id] = 2.2f + 1.4f * u4;
        ecs.desired_speed_factor[id] = 0.74f + 0.20f * u5;
        ecs.lc_cooldown[id] = 0.0f;
    }
}

__device__ __forceinline__ bool spawn_area_clear(
    int lane,
    float spawn_s,
    float px,
    float py,
    float new_len,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    int exclude
) {
    int base = world_cell_index(
        px,
        py,
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return true;

    float limit = MAX_SPEED_FALLBACK;
    if (lane >= 0 && lane < road.num_lanes) {
        limit = road.lane_speed_limit[lane];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    }

    /* The safer driver model needs enough initial time headway.  The previous
       8 m spawn gap let cars appear inside another car's safe following zone,
       which made the newly spawned group brake to zero immediately. */
    float required_gap = fmaxf(12.0f, new_len + 0.80f * limit);
    float radial_gap = fmaxf(7.0f, 0.65f * required_gap);

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j != exclude && ecs.alive[j] == ENTITY_ALIVE) {
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
                        float gap =
                            fabsf(ecs.s[j] - spawn_s)
                            - 0.5f * ecs.length[j]
                            - 0.5f * new_len;

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

__device__ __forceinline__ bool spawn_area_clear_fullscan(
    int lane,
    float spawn_s,
    float px,
    float py,
    float new_len,
    ECSArrays ecs,
    RoadNetwork road,
    int max_entities,
    int exclude
) {
    float limit = MAX_SPEED_FALLBACK;
    if (lane >= 0 && lane < road.num_lanes) {
        limit = road.lane_speed_limit[lane];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    }

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
            float gap =
                fabsf(ecs.s[j] - spawn_s)
                - 0.5f * ecs.length[j]
                - 0.5f * new_len;

            if (gap < required_gap) return false;
        }
    }

    return true;
}


__device__ __forceinline__ void requeue_spawn_demand_for_vehicle_ecs(
    int lane,
    int route,
    SpawnConfig spawn
) {
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

__global__ void resolve_spawn_overlap_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    SpawnConfig spawn,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    /*
        EN: Resolve same-step spawn races by requeueing the later/newer vehicle
            back into the invisible spawn accumulator.  The previous v5 code kept
            the vehicle alive and only set speed to zero, so two cars could remain
            visually stacked at the entry.  This kernel now guarantees that a car
            born into an occupied area is removed from the rendered network and its
            demand is retried later.

        KO: 같은 step에서 여러 스폰 포인트가 같은 진입부를 동시에 점유하는 경합을
            해결합니다. v5는 겹친 차량을 살려 둔 채 속도만 0으로 만들어 화면에 차가
            포개질 수 있었습니다. 이제는 후순위/신규 차량을 화면에서 제거하고 해당
            수요를 보이지 않는 스폰 대기열로 되돌려 다음 step에 재시도합니다.
    */
    float recent_window = fmaxf(fmaxf(dt * 2.5f, SPAWN_RACE_RECENT_WINDOW), 1.0e-4f);
    bool i_recent = fabsf(ecs.entry_time[i] - current_time) <= recent_window;
    if (!i_recent) return;

    int lane_i = ecs.lane_id[i];
    float limit = MAX_SPEED_FALLBACK;
    if (lane_i >= 0 && lane_i < road.num_lanes) {
        limit = road.lane_speed_limit[lane_i];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;
    }

    float required_gap = fmaxf(14.0f, ecs.length[i] + 0.95f * limit);
    float radial_gap = fmaxf(8.0f, 0.70f * required_gap);

    int base = world_cell_index(
        ecs.x[i],
        ecs.y[i],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(radial_gap / fmaxf(grid.cell_size, 0.1f)) + 1,
        1,
        WORLD_MAX_CELL_RADIUS
    );

    bool delete_i = false;

    for (int dy = -cr; dy <= cr && !delete_i; ++dy) {
        for (int dx = -cr; dx <= cr && !delete_i; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities && !delete_i) {
                if (j != i && ecs.alive[j] == ENTITY_ALIVE) {
                    float dxp = ecs.x[j] - ecs.x[i];
                    float dyp = ecs.y[j] - ecs.y[i];
                    float d2 = dxp * dxp + dyp * dyp;

                    bool same_surface = ecs.lane_id[j] == lane_i;
                    same_surface = same_surface || ecs.connector_from_lane[j] == lane_i;
                    same_surface = same_surface || ecs.connector_to_lane[j] == lane_i;
                    same_surface = same_surface || ecs.lane_change_from_lane[j] == lane_i;
                    same_surface = same_surface || ecs.lane_change_to_lane[j] == lane_i;

                    bool spatial_conflict = d2 < radial_gap * radial_gap;
                    bool lane_conflict = false;
                    if (same_surface) {
                        float gap =
                            fabsf(ecs.s[j] - ecs.s[i])
                            - 0.5f * ecs.length[j]
                            - 0.5f * ecs.length[i];
                        lane_conflict = gap < required_gap;
                    }

                    if (spatial_conflict || lane_conflict) {
                        bool j_recent = fabsf(ecs.entry_time[j] - current_time) <= recent_window;
                        if (!j_recent) {
                            delete_i = true;
                        } else {
                            // EN/KO: Deterministic tie break so only one of two recent cars requeues.
                            delete_i = i > j;
                        }
                    }
                }

                j = grid.cell_next[j];
                guard++;
            }
        }
    }

    if (delete_i) {
        int rid = ecs.route_id[i];
        requeue_spawn_demand_for_vehicle_ecs(lane_i, rid, spawn);
        ecs.alive[i] = ENTITY_FREE;
        ecs.speed[i] = 0.0f;
        ecs.accel[i] = 0.0f;
        if (metrics != nullptr) {
            atomicAdd(&metrics[METRIC_SPAWN_FAIL], 1.0f);
            atomicAdd(&metrics[METRIC_PENETRATION_PREVENTED], 1.0f);
        }
    }
}



__device__ __forceinline__ float spawn_rate_vps_at(
    const SpawnConfig spawn,
    int p,
    float current_time
) {
    float base = 0.0f;
    if (spawn.demand_vps != nullptr && p >= 0 && p < spawn.num_spawn_points) {
        base = fmaxf(spawn.demand_vps[p], 0.0f);
        if (!isfinite(base)) base = 0.0f;
    }

    if (
        spawn.demand_profile_vps == nullptr
        || spawn.demand_profile_has == nullptr
        || spawn.demand_profile_slots <= 0
        || p < 0
        || p >= spawn.num_spawn_points
        || spawn.demand_profile_has[p] == 0
    ) {
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


__global__ void spawn_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    SpawnConfig spawn,
    uint32_t* rng_state,
    float* metrics,
    int* spawn_lane_locks,
    int spawn_lock_count,
    float current_time,
    float dt,
    int max_entities,
    int step_index
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= spawn.num_spawn_points) return;

    int ln = spawn.spawn_lane[p];
    int rid = spawn.spawn_route[p];

    if (ln < 0 || ln >= road.num_lanes || rid < 0 || rid >= road.num_routes) {
        return;
    }

    int initial_route_pos = route_pos_for_lane_ecs(rid, ln, road);
    if (initial_route_pos < 0) {
        atomicAdd(&metrics[METRIC_SPAWN_FAIL], 1.0f);
        return;
    }

    float rate = spawn_rate_vps_at(spawn, p, current_time);

    uint32_t st =
        rng_state[p]
        ^ ((uint32_t)step_index * 747796405u
        +  (uint32_t)p * 2891336453u + 1u);

    float acc = spawn.spawn_accumulator[p] + rate * dt;

    if (!isfinite(acc) || acc < 0.0f) acc = 0.0f;
    acc = fminf(acc, SPAWN_ACCUMULATOR_MAX);

    /* EN: Once demand has crossed an integer vehicle, that vehicle is scheduled.
       Scheduled spawns are no longer dropped just because the entry area is
       congested; otherwise a jammed road would artificially reduce inflow and
       could not reproduce queue growth.
       KO: 누적 수요가 정수 차량 수를 넘으면 그 차량은 예약된 스폰입니다.
       진입부가 막혔다는 이유로 예약 스폰을 버리지 않습니다. 그래야 정체 상황에서도
       유입량이 보존되어 네트워크 정체를 재현할 수 있습니다. */
    int count = clampi_cuda((int)floorf(acc), 0, SPAWN_MAX_PER_POINT_PER_STEP);

    if (count <= 0) {
        spawn.spawn_accumulator[p] = acc;
        rng_state[p] = st;
        return;
    }

    /*
        EN: A spawn point is one producer, but several spawn points may map to
            the same physical lane.  CUDA threads for those points run at the
            same time, so a full-scan clear test can still race before the other
            new vehicle flips from SPAWNING to ALIVE.  Use the already-cleared
            reservation table as a per-step lane mutex; losing producers keep
            their demand in the invisible queue and try on the next frame.
        KO: 스폰 포인트는 여러 개라도 실제 같은 차로에 매칭될 수 있습니다.
            CUDA thread가 동시에 돌면 full-scan 검사 전에 서로가 아직 ALIVE가
            아니라서 같은 위치에 생성될 수 있습니다. 매 step 초기화되는 reservation
            table을 차로별 mutex처럼 사용해, 진 쪽은 화면에 겹치지 않고 보이지 않는
            대기열에 그대로 남겨 다음 프레임에 다시 시도합니다.
    */
    if (spawn_lane_locks != nullptr && spawn_lock_count > 0) {
        int lock_idx = ln % spawn_lock_count;
        int old_owner = atomicCAS(&spawn_lane_locks[lock_idx], RESERVATION_FREE, p);
        if (old_owner != RESERVATION_FREE && old_owner != p) {
            spawn.spawn_accumulator[p] = acc;
            rng_state[p] = st;
            if (metrics != nullptr) atomicAdd(&metrics[METRIC_ENTRY_QUEUE_HOLD], dt);
            return;
        }
    }

    spawn.spawn_accumulator[p] = acc - (float)count;

    int failed = 0;
    int deferred = 0;
    int spawned_now = 0;

    for (int c = 0; c < count; ++c) {
        float len = 4.2f + 0.7f * rand_uniform(st);
        float wid = 1.7f + 0.25f * rand_uniform(st);

        float limit = road.lane_speed_limit[ln];
        if (!isfinite(limit) || limit < 2.0f) limit = MAX_SPEED_FALLBACK;

        /* EN: A scheduled spawn is preserved, but it is not forced into an
           occupied entry cell.  The backlog remains in spawn_accumulator, which
           behaves like an invisible queue outside the network.  When the entry
           opens, cars are released in order instead of being stacked on top of
           each other.
           KO: 예약된 스폰은 보존하지만, 점유된 진입부 안으로 억지로 생성하지
           않습니다. 누적기는 네트워크 밖의 보이지 않는 대기열처럼 남아 있다가,
           진입부가 열리면 순서대로 차량을 내보냅니다. 그래서 스폰 지역 중첩이
           생기지 않고 시간이 뒤로 밀립니다. */
        float required_gap = fmaxf(12.0f, len + 0.80f * limit);
        float center_spacing = required_gap + len + 1.0f;
        float spawn_s = len * 0.5f + 2.0f + (float)spawned_now * center_spacing;

        float max_spawn_s = fmaxf(0.0f, road.lane_length[ln] - len * 0.5f - 0.25f);
        if (spawn_s > max_spawn_s) {
            deferred += count - c;
            break;
        }
        spawn_s = clampf_cuda(spawn_s, 0.0f, max_spawn_s);

        float px, py;
        lane_xy_from_s(ln, spawn_s, road, px, py);

        bool entry_clear = spawn_area_clear(
            ln,
            spawn_s,
            px,
            py,
            len,
            ecs,
            road,
            grid,
            max_entities,
            -1
        );

        if (entry_clear) {
            // EN/KO: Full scan sees vehicles spawned earlier in this same step,
            // which are not present in the spatial grid until the rebuild below.
            entry_clear = spawn_area_clear_fullscan(
                ln,
                spawn_s,
                px,
                py,
                len,
                ecs,
                road,
                max_entities,
                -1
            );
        }

        if (!entry_clear) {
            deferred += count - c;
            break;
        }

        int id = -1;

        // EN/KO: Fast random allocation first.
        for (int tries = 0; tries < 32; ++tries) {
            int cand = clampi_cuda((int)(rand_uniform(st) * max_entities), 0, max_entities - 1);
            if (atomicCAS((unsigned int*)&ecs.alive[cand], ENTITY_FREE, ENTITY_SPAWNING) == ENTITY_FREE) {
                id = cand;
                break;
            }
        }

        // EN/KO: If the random picks missed sparse free slots, scan once so a
        // scheduled spawn is not lost while capacity still exists.
        if (id < 0) {
            int start_id = clampi_cuda((int)(rand_uniform(st) * max_entities), 0, max_entities - 1);
            for (int k = 0; k < max_entities; ++k) {
                int cand = (start_id + k) % max_entities;
                if (atomicCAS((unsigned int*)&ecs.alive[cand], ENTITY_FREE, ENTITY_SPAWNING) == ENTITY_FREE) {
                    id = cand;
                    break;
                }
            }
        }

        if (id < 0) {
            failed += count - c;
            if (metrics != nullptr) atomicAdd(&metrics[METRIC_SPAWN_FAIL], (float)(count - c));
            break;
        }

        uint32_t rs =
            rng_state[id + spawn.num_spawn_points]
            ^ ((uint32_t)step_index + 12345u)
            ^ ((uint32_t)p * 1103515245u);

        int dtype =
            rand_uniform(rs) < clampf_cuda(spawn.av_penetration, 0.0f, 1.0f)
            ? AV
            : HUMAN;

        ecs.x[id] = px;
        ecs.y[id] = py;
        ecs.s[id] = spawn_s;

        // EN/KO: Only physically clear entries spawn.  Therefore every new car
        // can start as normal rolling traffic; congested demand waits invisibly.
        ecs.speed[id] = clampf_cuda(limit * 0.45f, 3.0f, limit * 0.7f);
        ecs.accel[id] = 0.0f;
        ecs.heading[id] = lane_heading(ln, road);
        ecs.steer_angle[id] = 0.0f;

        ecs.length[id] = len;
        ecs.width[id] = wid;

        ecs.driver_type[id] = dtype;
        ecs.reaction_time[id] = dtype == AV ? 0.38f : 1.55f;
        ecs.min_gap[id] = dtype == AV ? SAFE_GAP_AV : SAFE_GAP_HUMAN;

        ecs.lane_id[id] = ln;

        ecs.route_id[id] = rid;
        ecs.route_pos[id] = initial_route_pos;
        ecs.entry_time[id] = current_time;

        ecs.vehicle_state[id] = VEH_ON_LANE;

        ecs.connector_from_lane[id] = -1;
        ecs.connector_to_lane[id] = -1;
        ecs.connector_s[id] = 0.0f;
        ecs.connector_length[id] = 0.0f;

        ecs.lane_change_active[id] = 0;
        ecs.lane_change_from_lane[id] = ln;
        ecs.lane_change_to_lane[id] = ln;
        ecs.lane_change_t[id] = 0.0f;
        ecs.lane_change_duration[id] =
            dtype == AV ? LANE_CHANGE_DURATION_AV : LANE_CHANGE_DURATION_HUMAN;

        // EN: New vehicles start with indicators off.
        // KO: 새로 스폰된 차량은 방향지시등을 끈 상태에서 시작합니다.
        if (ecs.turn_signal != nullptr) ecs.turn_signal[id] = INDICATOR_NONE;
        if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[id] = 0.0f;

        init_driver_personality_ecs(id, dtype, rs, ecs);

        rng_state[id + spawn.num_spawn_points] = rs;

        __threadfence();
        atomicExch((unsigned int*)&ecs.alive[id], (unsigned int)ENTITY_ALIVE);
        __threadfence();

        spawned_now++;
        atomicAdd(&metrics[METRIC_SPAWNED], 1.0f);
    }

    if (failed > 0 || deferred > 0) {
        // EN/KO: Capacity failures and blocked entry queues are retried later.
        spawn.spawn_accumulator[p] = fminf(
            spawn.spawn_accumulator[p] + (float)(failed + deferred),
            SPAWN_ACCUMULATOR_MAX
        );
    }

    rng_state[p] = st;
}

// ============================================================
// Turn Indicator System
// 방향지시등 시스템
// ============================================================

__global__ void turn_signal_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    float* metrics,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities) return;

    if (ecs.alive[i] != ENTITY_ALIVE) {
        if (ecs.turn_signal != nullptr) ecs.turn_signal[i] = INDICATOR_NONE;
        if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = 0.0f;
        return;
    }

    int desired_signal = INDICATOR_NONE;
    int lane = ecs.lane_id[i];
    bool human = ecs.driver_type[i] == HUMAN;

    if (ecs.vehicle_state[i] == VEH_IN_CONNECTOR) {
        int turn = turn_code_from_lanes_ecs(ecs.connector_from_lane[i], ecs.connector_to_lane[i], road);
        if (turn == TURN_LEFT) desired_signal = INDICATOR_LEFT;
        else if (turn == TURN_RIGHT) desired_signal = INDICATOR_RIGHT;
    } else if (valid_lane_ecs(lane, road)) {
        if (ecs.lane_change_active[i] != 0) {
            desired_signal = indicator_from_lateral_move_ecs(
                ecs.lane_change_from_lane[i],
                ecs.lane_change_to_lane[i],
                road
            );
        }

        int rid = ecs.route_id[i];
        int rpos = ecs.route_pos[i];
        if (rid >= 0 && rid < road.num_routes) {
            int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
            if (repaired_pos >= 0) {
                rpos = repaired_pos;
                ecs.route_pos[i] = repaired_pos;
            }

            int ro0 = road.route_offsets[rid];
            int ro1 = road.route_offsets[rid + 1];
            int route_len = ro1 - ro0;
            if (rpos >= 0 && rpos < route_len) {
                int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
                rpos = ecs.route_pos[i];
                bool has_next = valid_lane_ecs(next_lane, road) && lane_connected(lane, next_lane, road);
                int turn = route_turn_for_vehicle_ecs(i, ecs, road);
                if (has_next) {
                    turn = effective_turn_code_ecs(lane, next_lane, turn, road);

                    int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
                    adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
                    adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
                    if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
                        next_lane = adjusted_next;
                    }
                }

                float L = fmaxf(road.lane_length[lane], 0.1f);
                float dist_to_end = fmaxf(0.0f, L - ecs.s[i]);
                float turn_lookahead = human ? INDICATOR_TURN_LOOKAHEAD_HUMAN : INDICATOR_TURN_LOOKAHEAD_AV;
                float lc_lookahead = human ? INDICATOR_LC_LOOKAHEAD_HUMAN : INDICATOR_LC_LOOKAHEAD_AV;

                int interchange_source_lane = has_next ? interchange_source_outer_lane_ecs(lane, next_lane, road) : -1;
                bool interchange_dedicated = has_next && valid_lane_ecs(interchange_source_lane, road) && lane != interchange_source_lane;
                bool ordinary_dedicated = has_next && turn_requires_dedicated_lane_ecs(turn);
                bool dedicated = interchange_dedicated || ordinary_dedicated;
                bool lane_ok = true;
                if (interchange_dedicated) lane_ok = false;
                else if (ordinary_dedicated) lane_ok = lane_legal_for_turn_ecs(lane, turn, road);

                if (interchange_dedicated && !lane_ok && dist_to_end < lc_lookahead) {
                    int target = adjacent_lane_toward_specific_lane_ecs(lane, interchange_source_lane, road);
                    desired_signal = indicator_from_lateral_move_ecs(lane, target, road);
                } else if (ordinary_dedicated && !lane_ok && dist_to_end < lc_lookahead) {
                    int target = adjacent_lane_toward_turn_lane_ecs(lane, turn, road);
                    desired_signal = indicator_from_lateral_move_ecs(lane, target, road);
                } else if (
                    has_next
                    && next_lane != lane
                    && dist_to_end < lc_lookahead
                    && lanes_share_link_geometry_ecs(lane, next_lane, road)
                ) {
                    // EN: Signal before a lateral lane-prep movement, not only
                    //     after the lane-change component has already started.
                    // KO: 차선변경 component가 시작된 뒤가 아니라, 옆 차로로 들어가려는
                    //     준비 단계부터 방향지시등을 켭니다.
                    desired_signal = indicator_from_lateral_move_ecs(lane, next_lane, road);
                } else if (has_next && dist_to_end < turn_lookahead) {
                    if (turn == TURN_LEFT) desired_signal = INDICATOR_LEFT;
                    else if (turn == TURN_RIGHT) desired_signal = INDICATOR_RIGHT;
                }
            }
        }
    }

    int old_signal = ecs.turn_signal != nullptr ? ecs.turn_signal[i] : INDICATOR_NONE;
    if (ecs.turn_signal != nullptr) ecs.turn_signal[i] = desired_signal;

    if (ecs.turn_signal_time != nullptr) {
        if (desired_signal == INDICATOR_NONE) {
            ecs.turn_signal_time[i] = 0.0f;
        } else if (old_signal == desired_signal) {
            ecs.turn_signal_time[i] = fminf(ecs.turn_signal_time[i] + dt, 60.0f);
        } else {
            ecs.turn_signal_time[i] = dt;
        }
    }

    if (desired_signal == INDICATOR_LEFT) atomicAdd(&metrics[METRIC_INDICATOR_LEFT_ON], 1.0f);
    if (desired_signal == INDICATOR_RIGHT) atomicAdd(&metrics[METRIC_INDICATOR_RIGHT_ON], 1.0f);
}

// ============================================================
// Perception System
// ============================================================

__device__ __forceinline__ bool vehicle_on_lane_surface_ecs(
    int id,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ void find_front_on_route_ecs(
    int self,
    int lane,
    int next_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float search_radius,
    float* metrics,
    float& front_gap,
    float& front_speed,
    float& front_s,
    float& front_len,
    int& front_lane
) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;
    front_s = 1.0e9f;
    front_len = 4.5f;
    front_lane = -1;

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    int cr = clampi_cuda(
        (int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    float curr_L = fmaxf(road.lane_length[lane], 0.1f);
    float self_s = ecs.s[self];
    float remain_curr = curr_L - self_s;

    int dtype = ecs.driver_type[self];
    float sense_range = fminf(search_radius, sensor_front_range_for_driver(dtype));
    float sense_fov = sensor_half_fov_for_driver(dtype);

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    float sensor_fwd, sensor_lat, sensor_dist;
                    bool detected = sensor_front_cone_detects_ecs(
                        self,
                        j,
                        ecs,
                        sense_range,
                        sense_fov,
                        sensor_fwd,
                        sensor_lat,
                        sensor_dist
                    );
                    if (detected && metrics != nullptr) {
                        atomicAdd(&metrics[METRIC_SENSOR_DETECTION], 1.0f);
                    }

                    /*
                        EN: Front following must be lane/route based, not purely
                            camera-cone based.  On curved connectors or during a
                            lane change, the vehicle ahead can sit outside the
                            current heading cone for one frame; if ignored, the
                            rear vehicle can jump through it.  We therefore accept
                            any vehicle that lies on the current or next route
                            surface and is ahead in route distance.

                        KO: 앞차 추종은 카메라 cone만이 아니라 차선/경로 거리 기준이어야
                            합니다. 곡선 connector나 차선변경 중에는 한 프레임 동안 앞차가
                            현재 heading cone 밖으로 보일 수 있고, 이때 무시하면 뒤차가
                            뚫고 지나갑니다. 따라서 현재/다음 경로 면에 있고 경로거리상
                            앞에 있는 차량은 항상 앞차 후보로 둡니다.
                    */
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
                        bool on_next =
                            next_lane >= 0
                            && next_lane < road.num_lanes
                            && vehicle_on_lane_surface_ecs(j, next_lane, ecs, road);

                        if (on_curr) {
                            float ds = ecs.s[j] - self_s;
                            if (ds > 0.0f && ds < route_ds) {
                                route_ds = ds;
                                candidate_lane = lane;
                            }
                        }

                        if (on_next) {
                            float ds_next = connector_route_distance_to_next_lane_s(
                                lane,
                                next_lane,
                                remain_curr,
                                ecs.s[j],
                                road
                            );

                            if (ds_next > 0.0f && ds_next < route_ds) {
                                route_ds = ds_next;
                                candidate_lane = next_lane;
                            }
                        }
                    }

                    if (candidate_lane >= 0 && route_ds <= search_radius + 1.0f) {
                        float gap =
                            route_ds
                            - 0.5f * ecs.length[self]
                            - 0.5f * ecs.length[j];

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

                            if (metrics != nullptr) {
                                atomicAdd(&metrics[METRIC_SENSOR_FRONT_HIT], 1.0f);
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



__device__ __forceinline__ void find_lane_neighbors_ecs(
    int self,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float search_radius,
    float* metrics,
    float& front_gap,
    float& front_speed,
    float& rear_gap,
    float& rear_speed
) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;
    rear_gap = 1.0e9f;
    rear_speed = 0.0f;

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    int cr = clampi_cuda(
        (int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    int dtype = ecs.driver_type[self];
    float front_range = fminf(search_radius, sensor_front_range_for_driver(dtype));
    float mirror_range = fminf(search_radius, sensor_side_range_for_driver(dtype));
    float front_fov = sensor_half_fov_for_driver(dtype);

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
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
                                eff_s = fmaxf(road.lane_length[target_lane], 0.1f)
                                      + ecs.connector_s[j];
                            }
                        }

                        float ds = eff_s - ecs.s[self];
                        float sf, sl, sd;
                        bool seen = false;

                        if (ds >= 0.0f) {
                            seen = sensor_front_cone_detects_ecs(
                                self,
                                j,
                                ecs,
                                front_range,
                                front_fov,
                                sf,
                                sl,
                                sd
                            );
                        } else {
                            seen = sensor_rear_mirror_detects_ecs(
                                self,
                                j,
                                ecs,
                                mirror_range,
                                sf,
                                sl,
                                sd
                            );
                        }

                        bool lane_geometry_seen = fabsf(ds) <= search_radius;
                        if (seen || lane_geometry_seen) {
                            if (seen && metrics != nullptr) {
                                atomicAdd(&metrics[METRIC_SENSOR_DETECTION], 1.0f);
                            }

                            float gap =
                                fabsf(ds)
                                - 0.5f * ecs.length[self]
                                - 0.5f * ecs.length[j];

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


__device__ __forceinline__ void indicator_merge_response_accel_ecs(
    int self,
    int lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float current_time,
    float dt,
    float front_gap,
    float* metrics,
    float& yield_limit,
    float& assert_boost
) {
    /*
        EN: Neighbor response to a vehicle that has indicated a lane change into
            this lane.  Most drivers create a gap by easing off/braking.  A small,
            deterministic fraction of aggressive drivers accelerates instead; the
            merging vehicle will see the closing rear gap in MOBIL and will not
            enter.

        KO: 이 차로로 들어오겠다고 깜빡이를 켠 옆 차에 대한 반응입니다. 대부분은
            가속을 멈추거나 약하게 감속해 gap을 만들어 줍니다. 다만 공격적인 일부
            운전자는 오히려 가속합니다. 이 경우 들어오려는 차량은 MOBIL 안전 gap에서
            뒤차 접근을 보고 차선변경을 보류합니다.
    */
    yield_limit = 1000.0f;
    assert_boost = 0.0f;
    if (!valid_lane_ecs(lane, road)) return;

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda(
        (int)ceilf(INDICATOR_MERGE_COURTESY_RANGE / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    bool human = ecs.driver_type[self] == HUMAN;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float yield_decel = human ? INDICATOR_MERGE_YIELD_DECEL_HUMAN : INDICATOR_MERGE_YIELD_DECEL_AV;
    float assert_accel = human ? INDICATOR_MERGE_ASSERT_ACCEL_HUMAN : INDICATOR_MERGE_ASSERT_ACCEL_AV;

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && ecs.vehicle_state[j] == VEH_ON_LANE) {
                    int other_lane = ecs.lane_id[j];
                    int sig = indicator_state_ecs(j, ecs);
                    int target_lane = indicator_target_lane_ecs(other_lane, sig, road);

                    if (
                        target_lane == lane
                        && lanes_share_link_geometry_ecs(other_lane, lane, road)
                        && ecs.turn_signal_time != nullptr
                        && ecs.turn_signal_time[j] >= INDICATOR_MIN_ON_TIME
                    ) {
                        float ds = ecs.s[j] - ecs.s[self];
                        if (ds >= -INDICATOR_MERGE_SIDE_RANGE && ds <= INDICATOR_MERGE_COURTESY_RANGE) {
                            float raw_gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                            float close_gap = fmaxf(INDICATOR_MERGE_SIDE_RANGE, ecs.speed[self] * 0.85f + 5.5f);

                            if (raw_gap < close_gap) {
                                uint32_t slot = (uint32_t)floorf(current_time / 3.0f);
                                float h = hash01_ecs(
                                    ((uint32_t)(self + 1) * 747796405u)
                                    ^ ((uint32_t)(j + 3) * 2891336453u)
                                    ^ (slot * 277803737u)
                                );

                                float assert_chance = INDICATOR_MERGE_ASSERT_RATE;
                                assert_chance *= 0.50f
                                    + 0.75f * clampf_cuda(ecs.aggressiveness[self], 0.0f, 1.0f)
                                    + 0.45f * clampf_cuda(ecs.risk_tolerance[self], 0.0f, 1.0f);
                                assert_chance *= 1.15f - 0.55f * clampf_cuda(ecs.politeness[self], 0.0f, 1.0f);
                                assert_chance = clampf_cuda(assert_chance, 0.02f, 0.28f);

                                float safe_front = fmaxf(ecs.length[self] + MIN_BUMPER_GAP + 4.0f, ecs.speed[self] * 0.75f + 6.0f);
                                bool can_assert = h < assert_chance && front_gap > safe_front;

                                if (can_assert) {
                                    float desired = desired_speed_ecs(self, lane, ecs, road);
                                    float a = (desired - ecs.speed[self]) / fmaxf(dt, 0.01f);
                                    a = clampf_cuda(a, 0.0f, max_accel * assert_accel);
                                    assert_boost = fmaxf(assert_boost, a);
                                    if (metrics != nullptr) atomicAdd(&metrics[METRIC_HUMAN_AI_ASSERTIVE_GO], 1.0f);
                                } else {
                                    float intensity = clampf_cuda((close_gap - raw_gap) / fmaxf(close_gap, 0.5f), 0.0f, 1.0f);
                                    float y = -yield_decel * (0.65f + 0.75f * intensity);
                                    yield_limit = fminf(yield_limit, y);
                                    if (metrics != nullptr) {
                                        atomicAdd(&metrics[METRIC_HUMAN_AI_COURTESY_YIELD], 1.0f);
                                        atomicAdd(&metrics[METRIC_INDICATOR_CONFLICT_YIELD], 1.0f);
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

__global__ void perception_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    PerceptionSoA perception,
    float* metrics,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int lane = ecs.lane_id[i];
    if (lane < 0 || lane >= road.num_lanes) return;

    int rid = ecs.route_id[i];
    int rpos = ecs.route_pos[i];
    if (rid < 0 || rid >= road.num_routes) {
        perception.front_gap[i] = 1.0e9f;
        perception.front_speed[i] = 0.0f;
        perception.front_s[i] = 1.0e9f;
        perception.front_length[i] = 0.0f;
        perception.front_lane[i] = -1;
        perception.target_front_gap[i] = 1.0e9f;
        perception.target_front_speed[i] = 0.0f;
        perception.target_rear_gap[i] = 1.0e9f;
        perception.target_rear_speed[i] = 0.0f;
        return;
    }

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[i] = repaired_pos;
    }

    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;

    if (route_len <= 0 || rpos < 0 || rpos >= route_len) {
        perception.front_gap[i] = 1.0e9f;
        perception.front_speed[i] = 0.0f;
        perception.front_s[i] = 1.0e9f;
        perception.front_length[i] = 0.0f;
        perception.front_lane[i] = -1;
        perception.target_front_gap[i] = 1.0e9f;
        perception.target_front_speed[i] = 0.0f;
        perception.target_rear_gap[i] = 1.0e9f;
        perception.target_rear_speed[i] = 0.0f;
        return;
    }

    int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
    rpos = ecs.route_pos[i];

    bool human = ecs.driver_type[i] == HUMAN;

    find_front_on_route_ecs(
        i,
        lane,
        next_lane,
        ecs,
        road,
        grid,
        max_entities,
        human ? 170.0f : 150.0f,
        nullptr,
        perception.front_gap[i],
        perception.front_speed[i],
        perception.front_s[i],
        perception.front_length[i],
        perception.front_lane[i]
    );


    int ll = geometric_left_neighbor_ecs(lane, road);
    int rr = geometric_right_neighbor_ecs(lane, road);

    float fg, fv, rg, rv;

    int target_lane = -1;
    int turn = TURN_STRAIGHT;
    bool has_next =
        next_lane >= 0
        && next_lane < road.num_lanes
        && lane_connected(lane, next_lane, road);

    if (has_next) {
        int route_turn = (rpos >= 0 && rpos < route_len) ? road.route_turns[ro0 + rpos] : TURN_STRAIGHT;
        turn = effective_turn_code_ecs(lane, next_lane, route_turn, road);

        int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
        adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
        adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
        if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
            next_lane = adjusted_next;
        }
    }

    int interchange_source_lane = has_next ? interchange_source_outer_lane_ecs(lane, next_lane, road) : -1;
    bool interchange_dedicated = has_next && valid_lane_ecs(interchange_source_lane, road) && lane != interchange_source_lane;

    if (interchange_dedicated) {
        int prep_lane = adjacent_lane_toward_specific_lane_ecs(lane, interchange_source_lane, road);
        if (prep_lane >= 0 && prep_lane < road.num_lanes) {
            target_lane = prep_lane;
        }
    } else if (
        has_next
        && turn_requires_dedicated_lane_ecs(turn)
        && !lane_legal_for_turn_ecs(lane, turn, road)
    ) {
        int prep_lane = adjacent_lane_toward_turn_lane_ecs(lane, turn, road);
        if (prep_lane >= 0 && prep_lane < road.num_lanes) {
            target_lane = prep_lane;
        }
    }

    if (target_lane < 0) {
        if (next_lane == ll) target_lane = ll;
        else if (next_lane == rr) target_lane = rr;
    }

    if (target_lane >= 0) {
        find_lane_neighbors_ecs(
            i,
            target_lane,
            ecs,
            road,
            grid,
            max_entities,
            human ? 150.0f : 125.0f,
            nullptr,
            fg,
            fv,
            rg,
            rv
        );

        perception.target_front_gap[i] = fg;
        perception.target_front_speed[i] = fv;
        perception.target_rear_gap[i] = rg;
        perception.target_rear_speed[i] = rv;
    } else {
        perception.target_front_gap[i] = 1.0e9f;
        perception.target_front_speed[i] = 0.0f;
        perception.target_rear_gap[i] = 1.0e9f;
        perception.target_rear_speed[i] = 0.0f;
    }

    /* Coarse, low-contention sensor statistics.  The previous version counted
       every candidate inside the inner neighbor loops, which could create many
       atomics per vehicle per frame in dense traffic. */
    if (metrics != nullptr) {
        bool front_hit = perception.front_gap[i] < 1.0e8f;
        bool lane_hit =
            perception.target_front_gap[i] < 1.0e8f
            || perception.target_rear_gap[i] < 1.0e8f;
        if (front_hit) atomicAdd(&metrics[METRIC_SENSOR_FRONT_HIT], 1.0f);
        if (front_hit || lane_hit) atomicAdd(&metrics[METRIC_SENSOR_DETECTION], 1.0f);
    }
}

// ============================================================
// Reservation System
// ============================================================

__global__ void clear_reservation_system(
    int* reservation_table,
    int total_slots
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_slots) {
        reservation_table[i] = RESERVATION_FREE;
    }
}

__device__ __forceinline__ bool try_reserve_slot_ecs(
    int self,
    int node,
    float arrival_time,
    float crossing_time,
    int* reservation_table,
    int num_nodes,
    float current_time,
    float* metrics
) {
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
            atomicAdd(&metrics[METRIC_RES_REJECT], 1.0f);
            return false;
        }
    }

    for (int k = 0; k < need_slots; ++k) {
        int idx = base + start_slot + k;

        int old = atomicCAS(
            &reservation_table[idx],
            RESERVATION_FREE,
            self
        );

        if (old != RESERVATION_FREE && old != self) {
            atomicAdd(&metrics[METRIC_RES_REJECT], 1.0f);
            return false;
        }
    }

    atomicAdd(&metrics[METRIC_RES_ACCEPT], 1.0f);
    return true;
}

// ============================================================
// Decision System
// ============================================================

__device__ __forceinline__ float signal_accel_limit_ecs(
    int lane,
    int turn,
    float ss,
    float v,
    float reaction_time,
    int dtype,
    float current_time,
    const RoadNetwork road,
    const Signals signals,
    int* out_state,
    float* out_stop_s
) {
    if (out_state) *out_state = LIGHT_GREEN;
    if (out_stop_s) *out_stop_s = 1.0e9f;

    float L = fmaxf(road.lane_length[lane], 0.1f);
    float stop_s = fmaxf(0.0f, L - DEFAULT_STOP_OFFSET);
    float d = stop_s - ss;

    int st = get_signal_for_lane_turn(lane, turn, current_time, road, signals);
    if (out_state) *out_state = st;

    if (d > 190.0f) return 1000.0f;
    if (st == LIGHT_GREEN) return 1000.0f;

    /* Already across the stop line.  The connector gate in the decision stage
       will still prevent a new red entry unless the vehicle was physically
       committed. */
    if (d < -0.25f) return 1000.0f;

    bool human = dtype == HUMAN;
    float rt = fmaxf(reaction_time, human ? 0.75f : 0.12f);
    float comfortable_b = human ? (0.78f * MAX_DECEL_HUMAN) : (0.86f * MAX_DECEL_AV);
    comfortable_b = fmaxf(comfortable_b, 1.2f);

    float comfortable_stop =
        v * rt
        + (v * v) / fmaxf(2.0f * comfortable_b, 0.1f)
        + INTERSECTION_STOP_BUFFER;

    bool stop = false;

    if (st == LIGHT_RED) {
        stop = true;
    } else {
        /* Yellow: stop unless the car is already too close to brake
           comfortably.  This removes the previous "barely pass" behavior. */
        stop = d > comfortable_stop * 0.72f;
    }

    if (!stop) return 1000.0f;

    if (out_stop_s) *out_stop_s = stop_s;

    /*
        If a car has already slowed to a near stop while still several meters
        before the stop line, do not clamp acceleration to exactly zero.  The
        old v^2/(2d) formula returns 0 when v==0, so cars could freeze with no
        vehicle in front and create artificial queues.  Allow low-speed creep;
        the stop-line guard below still prevents crossing on red.
    */
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


__device__ __forceinline__ float effective_s_on_lane_surface_ecs(
    int id,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ bool indicator_targets_lane_ecs(
    int id,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ bool lane_change_assertive_response_ecs(
    int responder,
    int merger,
    float current_time,
    ECSArrays ecs
) {
    float behavior = clampf_cuda(
        0.62f * ecs.aggressiveness[responder]
        + 0.42f * ecs.risk_tolerance[responder]
        - 0.34f * ecs.politeness[responder],
        0.0f,
        1.0f
    );
    float p = clampf_cuda(
        LANE_CHANGE_COOP_ASSERTIVE_PROB * (0.30f + 1.15f * behavior),
        0.015f,
        0.24f
    );
    uint32_t slot = (uint32_t)floorf(current_time * 1.7f);
    uint32_t h = (uint32_t)responder * 1103515245u ^ (uint32_t)merger * 2654435761u ^ slot * 2246822519u;
    return hash01_ecs(h) < p;
}

__device__ __forceinline__ bool lane_change_assertive_blocker_ecs(
    int self,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    float current_time,
    int max_entities,
    float* metrics
) {
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

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
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
                        if (metrics != nullptr) atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);
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

__device__ __forceinline__ float lane_change_courtesy_accel_limit_ecs(
    int self,
    int lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    float current_time,
    int max_entities,
    float* out_boost,
    bool* out_assertive
) {
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

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE && indicator_targets_lane_ecs(j, lane, ecs, road)) {
                    int jl = ecs.lane_id[j];
                    if (valid_lane_ecs(jl, road) && same_approach_same_direction_lanes_ecs(lane, jl, road)) {
                        float ds = ecs.s[j] - ecs.s[self];
                        if (ds <= LC_INDICATOR_SIDE_FRONT && ds >= -LC_INDICATOR_SIDE_REAR) {
                            float gap = fabsf(ds) - 0.5f * ecs.length[self] - 0.5f * ecs.length[j];
                            float zone = fmaxf(
                                LC_ASSERTIVE_BLOCK_GAP * LANE_CHANGE_COOP_ZONE_MULT,
                                ecs.length[self] + ecs.length[j] + MIN_BUMPER_GAP + ecs.speed[self] * 0.85f
                            );
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


__device__ __forceinline__ bool open_lane_candidate_safe_ecs(
    int self,
    float target_front_gap,
    float target_front_speed,
    float target_rear_gap,
    float target_rear_speed,
    ECSArrays ecs,
    bool urgent
) {
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

    return target_front_gap > front_req
        && target_rear_gap > rear_req
        && !(target_rear_speed > v + 1.2f && target_rear_gap < rear_req * 1.45f);
}

__device__ __forceinline__ int pick_open_lane_target_ecs(
    int self,
    int lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    PerceptionSoA perception,
    float current_time,
    int max_entities,
    bool allow_congestion_escape,
    bool allow_spread
) {
    if (!valid_lane_ecs(lane, road) || ecs.vehicle_state[self] != VEH_ON_LANE) return -1;
    if (ecs.lane_change_active[self] != 0) return -1;

    float L = fmaxf(road.lane_length[lane], 0.1f);
    float dist_to_end = fmaxf(0.0f, L - ecs.s[self]);
    float v = fmaxf(0.0f, ecs.speed[self]);
    bool human = ecs.driver_type[self] == HUMAN;

    float current_gap = perception.front_gap != nullptr ? perception.front_gap[self] : 1.0e9f;
    float current_front_speed = perception.front_speed != nullptr ? perception.front_speed[self] : 0.0f;
    if (!isfinite(current_gap)) current_gap = 0.0f;

    float blocked_gap = fmaxf(
        CONGESTION_ESCAPE_CURRENT_GAP,
        fmaxf(ecs.length[self], 4.0f) + MIN_BUMPER_GAP + v * 1.05f
    );
    bool front_lane_blocked =
        current_gap < blocked_gap
        || (current_gap < blocked_gap * 1.7f && current_front_speed + 2.0f < v);

    bool congestion_mode =
#if CONGESTION_ESCAPE_LC_ENABLED
        allow_congestion_escape
        && front_lane_blocked
        && ecs.lc_cooldown[self] <= CONGESTION_ESCAPE_COOLDOWN_READY
        && L >= CONGESTION_ESCAPE_MIN_LINK_LENGTH
        && dist_to_end >= CONGESTION_ESCAPE_MIN_DIST_TO_NODE;
#else
        false;
#endif

    bool spread_mode =
#if LANE_SPREAD_CHANGE_ENABLED
        allow_spread
        && ecs.lc_cooldown[self] <= LANE_SPREAD_COOLDOWN_READY
        && L >= LANE_SPREAD_MIN_LINK_LENGTH
        && dist_to_end >= LANE_SPREAD_MIN_DIST_TO_NODE
        && current_gap >= LANE_SPREAD_MIN_CURRENT_GAP;
#else
        false;
#endif

    if (!congestion_mode && !spread_mode) return -1;

    int cur_group_idx = -1;
    int cur_group_count = lane_group_count_and_index_ecs(lane, road, cur_group_idx);
    bool right_edge_pressure = false;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
    right_edge_pressure =
        allow_spread
        && cur_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP
        && cur_group_idx >= 0
        && cur_group_idx <= RIGHT_EDGE_BOTTLENECK_IDX_LIMIT;
#endif

    int candidates[2] = {
        geometric_left_neighbor_ecs(lane, road),
        geometric_right_neighbor_ecs(lane, road)
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
        find_lane_neighbors_ecs(
            self,
            target,
            ecs,
            road,
            grid,
            max_entities,
            search_r,
            nullptr,
            fg,
            fv,
            rg,
            rv
        );

        bool safe = open_lane_candidate_safe_ecs(self, fg, fv, rg, rv, ecs, congestion_mode);
        if (!safe) continue;

        float front_gain = fg - current_gap;
        float rear_gain = rg - (human ? OPEN_LANE_REAR_GAP_HUMAN : OPEN_LANE_REAR_GAP_AV);
        int target_group_idx = -1;
        int target_group_count = lane_group_count_and_index_ecs(target, road, target_group_idx);
        bool target_is_inner =
            cur_group_count == target_group_count
            && target_group_idx >= 0
            && cur_group_idx >= 0
            && target_group_idx > cur_group_idx;

        bool empty_interval = fg >= LANE_SPREAD_EMPTY_FRONT_GAP && rg >= LANE_SPREAD_EMPTY_REAR_GAP;

        bool good_for_congestion =
            congestion_mode
            && (
                front_gain >= CONGESTION_ESCAPE_FRONT_GAIN
                || fg >= blocked_gap * 1.25f
                || empty_interval
            );

        bool stable_random_preference = false;
        if (spread_mode) {
            uint32_t slot = (uint32_t)floorf(current_time / 5.0f);
            uint32_t h = hash_u32_ecs(
                ((uint32_t)(self + 1) * 747796405u)
                ^ ((uint32_t)(ecs.route_id[self] + 13) * 2891336453u)
                ^ ((uint32_t)(target + 31) * 277803737u)
                ^ slot
            );
            stable_random_preference = (h & 7u) == 0u;
        }

        bool good_for_spread =
            spread_mode
            && (
                front_gain >= LANE_SPREAD_FRONT_GAIN
                || (empty_interval && stable_random_preference)
                || (front_gain >= 6.0f && rear_gain >= LANE_SPREAD_REAR_GAIN && stable_random_preference)
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                || (right_edge_pressure && target_is_inner && front_gain >= -RIGHT_EDGE_SAFE_FRONT_LOSS)
#endif
            );

        if (!good_for_congestion && !good_for_spread) continue;

        float side_hash = hash01_ecs(
            ((uint32_t)(self + 5) * 1103515245u)
            ^ ((uint32_t)(target + 7) * 2654435761u)
        );
        float score = fg + 0.22f * rg - 0.45f * fmaxf(0.0f, rv - v) + side_hash;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
        if (target_group_count == cur_group_count && target_group_idx >= 0 && cur_group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP) {
            score += (float)target_group_idx * RIGHT_EDGE_INNER_BONUS;
            if (target_group_idx == 0) score -= RIGHT_EDGE_BOTTLENECK_PENALTY;
            if (right_edge_pressure && target_group_idx > cur_group_idx) score += RIGHT_EDGE_BOTTLENECK_PENALTY * 0.75f;
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
    if (best_lane < 0 && (spread_mode || congestion_mode)) {
        // EN: If the adjacent lanes are not immediately compelling, scan the
        //     whole current lane bundle, pick the emptiest stable target lane,
        //     and move one adjacent lane toward it.  This keeps vehicles from
        //     clinging to the lane chosen by the route cache when the exit is
        //     not coming soon.
        // KO: 바로 옆 차선만 봐서는 변화가 부족한 경우 현재 도로의 전체 차로 묶음을
        //     훑어 가장 비어 있는 목표 차선을 고르고, 그쪽으로 한 차로씩 이동합니다.
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
            uint32_t slot = (uint32_t)floorf(current_time / OPEN_LANE_EMPTIEST_SCAN_PERIOD);
            int right = rightmost_lane_in_group_ecs(lane, road);
            int cur = right;
            for (int kk = 0; kk < CRUISE_RANDOM_LANE_MAX_GROUP && valid_lane_ecs(cur, road); ++kk) {
                if (cur != lane && same_approach_same_direction_lanes_ecs(lane, cur, road)) {
                    float fg, fv, rg, rv;
                    find_lane_neighbors_ecs(
                        self,
                        cur,
                        ecs,
                        road,
                        grid,
                        max_entities,
                        LANE_SPREAD_SEARCH_RADIUS,
                        nullptr,
                        fg,
                        fv,
                        rg,
                        rv
                    );
                    bool empty_interval = fg >= LANE_SPREAD_EMPTY_FRONT_GAP && rg >= LANE_SPREAD_EMPTY_REAR_GAP;
                    float hash_bias = hash01_ecs(
                        ((uint32_t)(self + 113) * 1103515245u)
                        ^ ((uint32_t)(cur + 19) * 2654435761u)
                        ^ (slot * 747796405u)
                    );
                    float closing_penalty = fmaxf(0.0f, rv - v) * 0.55f;
                    float score = fg + 0.30f * rg - closing_penalty + hash_bias;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                    if (group_count >= RIGHT_EDGE_BOTTLENECK_MIN_GROUP) {
                        score += (float)kk * RIGHT_EDGE_INNER_BONUS;
                        if (kk == 0) score -= RIGHT_EDGE_BOTTLENECK_PENALTY;
                        if (right_edge_pressure && kk > cur_idx) score += RIGHT_EDGE_BOTTLENECK_PENALTY * 0.60f;
                    }
#endif
                    if (empty_interval) score += 18.0f;
                    if (fg > current_gap + OPEN_LANE_EMPTIEST_FRONT_GAIN) score += 7.0f;
                    float gain_need = OPEN_LANE_EMPTIEST_SCORE_GAIN;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                    if (right_edge_pressure && kk > cur_idx) gain_need = RIGHT_EDGE_SCAN_SCORE_GAIN;
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

            int step = valid_lane_ecs(desired_group_lane, road)
                ? adjacent_lane_toward_specific_lane_ecs(lane, desired_group_lane, road)
                : -1;
            if (valid_lane_ecs(step, road) && same_approach_same_direction_lanes_ecs(lane, step, road)) {
                float fg, fv, rg, rv;
                find_lane_neighbors_ecs(
                    self,
                    step,
                    ecs,
                    road,
                    grid,
                    max_entities,
                    LANE_SPREAD_SEARCH_RADIUS,
                    nullptr,
                    fg,
                    fv,
                    rg,
                    rv
                );
                bool safe_step = open_lane_candidate_safe_ecs(self, fg, fv, rg, rv, ecs, congestion_mode);
                bool enough_step_gain = fg > current_gap + 3.0f || fg >= LANE_SPREAD_EMPTY_FRONT_GAP;
#if RIGHT_EDGE_BOTTLENECK_AVOID_ENABLED
                int step_idx = -1;
                lane_group_count_and_index_ecs(step, road, step_idx);
                if (right_edge_pressure && step_idx > cur_idx && fg >= current_gap - RIGHT_EDGE_SAFE_FRONT_LOSS) {
                    enough_step_gain = true;
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

__device__ __forceinline__ bool mobil_decision_ecs(
    int i,
    int target_lane,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    PerceptionSoA perception,
    float* metrics,
    float current_time,
    int max_entities,
    bool mandatory,
    bool opportunistic
) {
    if (target_lane < 0 || target_lane >= road.num_lanes) return false;
    if (ecs.lc_cooldown[i] > 0.0f && !mandatory && !opportunistic) return false;

    // EN: Do not enter when the vehicle in the signaled target lane chooses
    //     the rare assertive response and accelerates to close the gap.
    // KO: 깜빡이를 본 목표 차선 차량이 드물게 가속해 gap을 닫는 경우에는
    //     차선 변경을 시작하지 않습니다.
    if (lane_change_assertive_blocker_ecs(i, target_lane, ecs, road, grid, current_time, max_entities, metrics)) {
        return false;
    }

    bool human = ecs.driver_type[i] == HUMAN;

    atomicAdd(&metrics[METRIC_MOBIL_EVAL], 1.0f);

    float tf_gap = perception.target_front_gap[i];
    float tf_v   = perception.target_front_speed[i];
    float tr_gap = perception.target_rear_gap[i];
    float tr_v   = perception.target_rear_speed[i];

    float rear_required =
        (human ? LANE_CHANGE_REAR_GAP_HUMAN : LANE_CHANGE_REAR_GAP_AV)
        * (1.20f - 0.55f * ecs.risk_tolerance[i]);

    rear_required += tr_v * (human ? 1.0f : 0.65f);

    if (tr_gap < rear_required || (tr_v > ecs.speed[i] + 0.65f && tr_gap < rear_required * 1.55f)) {
        atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);
        return false;
    }

    float front_required =
        (human ? LANE_CHANGE_FRONT_GAP_HUMAN : LANE_CHANGE_FRONT_GAP_AV)
        * (1.15f - 0.45f * ecs.risk_tolerance[i]);

    front_required += ecs.speed[i] * (human ? 0.85f : 0.55f);

    if (tf_gap < front_required) {
        atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);
        return false;
    }

    if (mandatory) {
        atomicAdd(&metrics[METRIC_LC_ACCEPT], 1.0f);
        return true;
    }

    int lane = ecs.lane_id[i];

    float desired_curr = desired_speed_ecs(i, lane, ecs, road);
    float desired_next = desired_speed_ecs(i, target_lane, ecs, road);

    float a_old = estimate_follow_accel_ecs(
        ecs.speed[i],
        desired_curr,
        perception.front_gap[i],
        perception.front_speed[i],
        ecs.driver_type[i],
        ecs.min_gap[i],
        ecs.reaction_time[i],
        ecs.comfort_decel[i],
        ecs.aggressiveness[i],
        ecs.risk_tolerance[i]
    );

    float a_new = estimate_follow_accel_ecs(
        ecs.speed[i],
        desired_next,
        tf_gap,
        tf_v,
        ecs.driver_type[i],
        ecs.min_gap[i],
        ecs.reaction_time[i],
        ecs.comfort_decel[i],
        ecs.aggressiveness[i],
        ecs.risk_tolerance[i]
    );

    float ego_gain = a_new - a_old;

    float threshold = human ? MOBIL_THRESHOLD_HUMAN : MOBIL_THRESHOLD_AV;
    threshold *= 1.30f - 0.55f * ecs.aggressiveness[i];

    float utility = ego_gain - threshold;
    float hysteresis = human ? 0.15f : 0.10f;

    bool accepted = utility > MOBIL_MIN_ADVANTAGE + hysteresis;

    if (!accepted && opportunistic) {
        // EN: For straight-flow spreading and congestion bypasses, allow a
        //     slightly lower utility lane change with a stable per-vehicle
        //     probability.  Safety gaps above are still mandatory.
        // KO: 직진 분산/정체 회피 차선 변경은 안전 gap을 만족한 뒤, 약간 낮은
        //     효용도에서도 차량별 안정 확률로 허용합니다.
        uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
        uint32_t h = hash_u32_ecs(
            ((uint32_t)(i + 1) * 747796405u)
            ^ ((uint32_t)(target_lane + 13) * 2891336453u)
            ^ ((uint32_t)(ecs.route_pos[i] + 17) * 277803737u)
            ^ (slot * 1103515245u)
        );
        float p = CRUISE_RANDOM_LANE_CHANGE_PROB
            * (0.62f + 0.48f * clampf_cuda(ecs.aggressiveness[i], 0.0f, 1.0f)
                    + 0.35f * clampf_cuda(ecs.risk_tolerance[i], 0.0f, 1.0f));
        p = clampf_cuda(p, 0.045f, 0.36f);
        float tol = CRUISE_RANDOM_LANE_UTILITY_TOL;
        accepted = utility > -tol && hash01_ecs(h) < p;
    }

    if (accepted) atomicAdd(&metrics[METRIC_LC_ACCEPT], 1.0f);
    else atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);

    return accepted;
}

__global__ void decision_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    Signals signals,
    SpatialGrid grid,
    PerceptionSoA perception,
    DecisionSoA decision,
    int* reservation_table,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int lane = ecs.lane_id[i];
    if (lane < 0 || lane >= road.num_lanes) {
        decision.should_exit[i] = 1;
        return;
    }

    if (ecs.vehicle_state[i] == VEH_IN_CONNECTOR) {
        decision.target_accel[i] = 0.0f;
        return;
    }

    int rid = ecs.route_id[i];
    int rpos = ecs.route_pos[i];
    if (rid < 0 || rid >= road.num_routes) {
        decision.should_exit[i] = 1;
        return;
    }

    int repaired_pos = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[i] = repaired_pos;
    }

    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;

    if (route_len <= 0 || rpos < 0 || rpos >= route_len) {
        decision.should_exit[i] = 1;
        return;
    }

    int route_lane = road.route_lanes[ro0 + rpos];
    if (!route_lane_current_compatible_ecs(route_lane, lane, road)) {
        int downstream_next = (rpos + 1 < route_len) ? road.route_lanes[ro0 + rpos + 1] : -1;
        if (!valid_lane_ecs(downstream_next, road) || !lane_connected(lane, downstream_next, road)) {
            // EN/KO: Irreparable route/lane mismatch must not leave a vehicle
            // parked in the road.  Remove the broken vehicle instead of feeding
            // invalid route data into the controller forever.
            decision.should_exit[i] = 1;
            return;
        }
    }

    int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
    rpos = ecs.route_pos[i];
    int turn = route_turn_for_vehicle_ecs(i, ecs, road);

    bool has_next =
        next_lane >= 0
        && next_lane < road.num_lanes
        && lane_connected(lane, next_lane, road);

    if (has_next) {
        turn = effective_turn_code_ecs(lane, next_lane, turn, road);

        int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
        adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
        adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
        if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
            next_lane = adjusted_next;
        }
    }

    int interchange_source_lane = has_next ? interchange_source_outer_lane_ecs(lane, next_lane, road) : -1;

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
        return;
    }

    bool interchange_needs_outer_lane =
        has_next
        && valid_lane_ecs(interchange_source_lane, road)
        && lane != interchange_source_lane;
    bool ordinary_turn_needs_dedicated_lane =
        has_next
        && turn_requires_dedicated_lane_ecs(turn);
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
    float turn_prep_dist = turn_lane_prep_distance_ecs(
        turn_lane_steps >= 99 ? 4 : turn_lane_steps,
        v,
        ecs.driver_type[i]
    );
    float lc_no_start_dist = lane_change_no_start_distance_ecs(lane, road);
    float lc_deadline_dist = dist_to_end - lc_no_start_dist;
    float lc_duration_nominal = human ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV;
    bool mandatory_lc_pending =
        turn_needs_dedicated_lane
        && !turn_lane_ok
        && turn_lc_target >= 0
        && turn_lc_target < road.num_lanes;

    if (mandatory_lc_pending) {
        bool too_late_to_start_lc = dist_to_end <= fmaxf(lc_no_start_dist, MISSED_TURN_ESCAPE_DIST);
        float front_open = fmaxf(SMART_STALL_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 8.0f);
        if (too_late_to_start_lc) {
            float w = ecs.connector_length[i];
            if (!isfinite(w) || w < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) w = 0.0f;
            if (ecs.speed[i] < 1.0f || perception.front_gap[i] > front_open) {
                w = fminf(w + dt, 30.0f);
            }
            ecs.connector_length[i] = w;

            // EN: If a mandatory merge was missed, do not let the car park forever
            // in the travel lane.  After a short wait and only with a clear front
            // path, allow a slow late-turn escape.  This sacrifices strict lane
            // compliance only to prevent network-wide deadlock.
            // KO: 회전 전용 차선 진입을 놓쳤다고 주행 차로 한가운데 영구 정차하지
            // 않게 합니다. 짧게 기다린 뒤 앞 경로가 열려 있을 때만 저속 late-turn
            // escape를 허용합니다. 이는 네트워크 전체 데드락 방지를 위한 최후 수단입니다.
            if (w >= MISSED_TURN_ESCAPE_WAIT && perception.front_gap[i] > front_open) {
                turn_lane_ok = true;
                mandatory_lc_pending = false;
                float escape_cap = human ? MISSED_TURN_ESCAPE_SPEED_HUMAN : MISSED_TURN_ESCAPE_SPEED_AV;
                desired_v = fminf(desired_v, escape_cap);
                atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
            }
        }
    }

    if (ecs.lane_change_active[i] != 0) {
        // EN/KO: During a lane change, coast or brake lightly so the vehicle
        // has lateral room instead of accelerating into the side gap.
        desired_v = fminf(desired_v, fmaxf(human ? LANE_CHANGE_PREP_MIN_CAP_HUMAN : LANE_CHANGE_PREP_MIN_CAP_AV, desired_v * LANE_CHANGE_ACTIVE_SPEED_CAP_SCALE));
    }

    if (next_lane >= 0 && next_lane < road.num_lanes) {
        float ctrl_dist = human ? TURN_CONTROL_DIST_HUMAN : TURN_CONTROL_DIST_AV;

        if (dist_to_end < ctrl_dist) {
            float angle = turn_angle_deg(lane, next_lane, road);
            // EN: Do not slow down for a straight connector when the road ahead is clear.
            //     Only real turns get a turn-speed cap.
            // KO: 앞이 비어 있는 직진 교차로에서는 이유 없이 감속하지 않습니다.
            //     실제 회전일 때만 회전 속도 제한을 적용합니다.
            if (angle > STRAIGHT_NO_TURN_SPEED_CAP_DEG || turn != TURN_STRAIGHT) {
                desired_v = fminf(desired_v, turn_speed_cap(angle, ecs.driver_type[i]));
            }
        }
    }

    if (
        turn_needs_dedicated_lane
        && !turn_lane_ok
        && dist_to_end < turn_prep_dist
    ) {
        float wrong_lane_cap = human ? TURN_LANE_WRONG_LANE_SPEED_HUMAN : TURN_LANE_WRONG_LANE_SPEED_AV;
        if (dist_to_end < TURN_LANE_HARD_HOLD_DIST + DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER) {
            wrong_lane_cap = fminf(wrong_lane_cap, human ? 1.4f : 1.8f);
        }
        desired_v = fminf(desired_v, wrong_lane_cap);
        atomicAdd(&metrics[METRIC_TURN_LANE_PREP], 1.0f);
    }

    float acc_cmd = estimate_follow_accel_ecs(
        v,
        desired_v,
        perception.front_gap[i],
        perception.front_speed[i],
        ecs.driver_type[i],
        ecs.min_gap[i],
        ecs.reaction_time[i],
        ecs.comfort_decel[i],
        ecs.aggressiveness[i],
        ecs.risk_tolerance[i]
    );

    if (perception.front_gap[i] < 1.0e8f) {
        float safe =
            (human ? SAFE_GAP_HUMAN : SAFE_GAP_AV)
            + v * (human ? SAFE_TIME_HEADWAY_HUMAN : SAFE_TIME_HEADWAY_AV);

        if (perception.front_gap[i] < safe) {
            float ratio = safe / fmaxf(perception.front_gap[i], 0.5f);
            acc_cmd = fminf(acc_cmd, -max_decel * ratio * ratio);
        }
    }

    if (mandatory_lc_pending && dist_to_end < turn_prep_dist) {
        /*
            EN: Mandatory turn-lane preparation.  Compute the remaining distance
            to the no-start deadline, then stop accelerating or brake to buy time
            for a safe indicated merge.
            KO: 회전 전 차선변경 준비입니다. 현재 위치에서 차선변경 시작 마지노선까지
            남은 거리를 계산하고, 안전 gap이 생기도록 가속을 끊거나 감속합니다.
        */
        float pressure = clampf_cuda(
            (turn_prep_dist - dist_to_end) / fmaxf(turn_prep_dist, 1.0f),
            0.0f,
            1.0f
        );
        float deadline = fmaxf(0.0f, lc_deadline_dist);
        float min_cap = human ? LANE_CHANGE_PREP_MIN_CAP_HUMAN : LANE_CHANGE_PREP_MIN_CAP_AV;
        float time_budget = lc_duration_nominal * (1.20f + 0.45f * pressure);
        float prep_cap = deadline / fmaxf(time_budget, 0.35f);
        prep_cap = clampf_cuda(prep_cap, min_cap, desired_v);

        if (deadline < LANE_CHANGE_PREP_HARD_DIST) {
            prep_cap = fminf(prep_cap, min_cap + 0.45f * deadline / fmaxf(LANE_CHANGE_PREP_HARD_DIST, 1.0f));
        }
        desired_v = fminf(desired_v, prep_cap);

        float prep_a = (prep_cap - v) / fmaxf(dt, 0.01f);
        if (prep_a > LC_PREP_COAST_ACCEL_LIMIT) {
            prep_a = LC_PREP_COAST_ACCEL_LIMIT;
        }
        float max_prep_brake = human ? LC_PREP_MAX_BRAKE_HUMAN : LC_PREP_MAX_BRAKE_AV;
        if (deadline < LANE_CHANGE_PREP_HARD_DIST) {
            max_prep_brake = fminf(max_decel, max_prep_brake * 1.35f);
        }
        prep_a = clampf_cuda(prep_a, -max_prep_brake, LC_PREP_COAST_ACCEL_LIMIT);
        acc_cmd = fminf(acc_cmd, prep_a);
        atomicAdd(&metrics[METRIC_TURN_LANE_PREP], 1.0f);
    }

    if (ecs.lane_change_active[i] != 0) {
        acc_cmd = fminf(acc_cmd, LC_PREP_COAST_ACCEL_LIMIT);
    }

    float courtesy_boost = 0.0f;
    bool courtesy_assertive = false;
    float courtesy_limit = lane_change_courtesy_accel_limit_ecs(
        i,
        lane,
        ecs,
        road,
        grid,
        current_time,
        max_entities,
        &courtesy_boost,
        &courtesy_assertive
    );
    if (courtesy_limit < 999.0f) {
        acc_cmd = fminf(acc_cmd, courtesy_limit);
        atomicAdd(&metrics[METRIC_HUMAN_AI_COURTESY_YIELD], 1.0f);
    }
    if (courtesy_boost > 0.0f && perception.front_gap[i] > fmaxf(ecs.length[i] + MIN_BUMPER_GAP + 6.0f, 12.0f)) {
        acc_cmd = fmaxf(acc_cmd, fminf(courtesy_boost, max_accel * 0.65f));
        if (courtesy_assertive) atomicAdd(&metrics[METRIC_HUMAN_AI_ASSERTIVE_GO], 1.0f);
    }

    int sig_state = LIGHT_GREEN;
    float stop_s = 1.0e9f;

    bool self_inside_intersection_box =
        has_next
        && turn_lane_ok
        && inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);

    float sig_acc = signal_accel_limit_ecs(
        lane,
        turn,
        s_curr,
        v,
        ecs.reaction_time[i],
        ecs.driver_type[i],
        current_time,
        road,
        signals,
        &sig_state,
        &stop_s
    );

    // EN: Once the reference point is inside the lane-width intersection box,
    //     the safe behavior is to clear the box.  A red/yellow stop applies before
    //     the box, not after the car has already entered it.
    // KO: 차량 기준점이 차로 폭 기반 교차로 박스 안에 들어온 뒤에는 박스를 비우는
    //     것이 안전합니다. 적/황색 정지는 박스 진입 전 적용하고, 이미 들어온 차량은
    //     교차로 안에서 멈춰 데드락을 만들지 않게 합니다.
    if (self_inside_intersection_box) {
        sig_acc = 1000.0f;
    }

    acc_cmd = fminf(acc_cmd, sig_acc);

    if (
        self_inside_intersection_box
        && perception.front_gap[i] > fmaxf(ecs.length[i] + MIN_BUMPER_GAP + 3.0f, 7.0f)
    ) {
        // EN/KO: Once physically inside the intersection box, do not allow a
        // stale yield/red command to leave the vehicle stopped in the middle.
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
        if (signal_stop_command) atomicAdd(&metrics[METRIC_RED_LIGHT_STOP], 1.0f);
        if (stop_s < 1.0e8f && s_curr > stop_s + 0.25f && v > 0.2f) {
            atomicAdd(&metrics[METRIC_RED_LIGHT_VIOLATION], 1.0f);
        }
    } else if (sig_state == LIGHT_YELLOW) {
        if (signal_stop_command) atomicAdd(&metrics[METRIC_YELLOW_STOP], 1.0f);
        else atomicAdd(&metrics[METRIC_YELLOW_GO], 1.0f);
    }

    if (signal_stop_command && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        atomicAdd(&metrics[METRIC_INTERSECTION_WAIT], dt);
    }

    float interaction_limit = 1000.0f;
    if (has_next && turn_lane_ok && !self_inside_intersection_box && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        interaction_limit = interaction_accel_limit_ecs(
            i,
            lane,
            next_lane,
            ecs,
            road,
            grid,
            max_entities,
            nullptr
        );
        if (interaction_limit < 999.0f) {
            atomicAdd(&metrics[METRIC_INTERACTION_BRAKE], 1.0f);
        }
    }
    acc_cmd = fminf(acc_cmd, interaction_limit);

    bool reservation_granted = true;
    bool directional_conflict_present = false;

    if (has_next && turn_lane_ok && !self_inside_intersection_box && dist_to_end < INTERSECTION_APPROACH_RANGE) {
        if (unsignal_node) {
            float unsignal_limit = unsignal_priority_accel_limit_ecs(
                i,
                lane,
                next_lane,
                dist_to_end,
                perception.front_gap[i],
                ecs,
                road,
                grid,
                max_entities,
                current_time,
                dt,
                metrics,
                &unsignal_blocked,
                &unsignal_deadlock_release,
                &unsignal_conflict_seen
            );

            directional_conflict_present = unsignal_conflict_seen;

            if (unsignal_blocked && !unsignal_deadlock_release) {
                acc_cmd = fminf(acc_cmd, unsignal_limit);
                atomicAdd(&metrics[METRIC_INTERSECTION_WAIT], dt);
                atomicAdd(&metrics[METRIC_CONFLICT_YIELD], 1.0f);
                atomicAdd(&metrics[METRIC_COOP_YIELD], 1.0f);
            } else if (unsignal_deadlock_release) {
                float release_v = human ? DEADLOCK_RELEASE_CREEP_HUMAN : DEADLOCK_RELEASE_CREEP_AV;
                float release_a = (release_v - v) / fmaxf(dt, 0.01f);
                release_a = clampf_cuda(release_a, 0.0f, max_accel * 0.38f);
                acc_cmd = fmaxf(acc_cmd, release_a);
                acc_cmd = fminf(acc_cmd, max_accel * 0.38f);
                atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
            }
        } else {
            float conflict_limit = intersection_conflict_accel_limit_ecs(
                i,
                lane,
                next_lane,
                dist_to_end,
                ecs,
                road,
                grid,
                max_entities,
                current_time,
                nullptr
            );

            if (conflict_limit < 999.0f) {
                directional_conflict_present = true;
            }

            if (conflict_limit < acc_cmd) {
                acc_cmd = conflict_limit;
                atomicAdd(&metrics[METRIC_INTERSECTION_WAIT], dt);
                atomicAdd(&metrics[METRIC_CONFLICT_YIELD], 1.0f);
                atomicAdd(&metrics[METRIC_COOP_YIELD], 1.0f);
            }
        }
    } else if (ecs.vehicle_state[i] == VEH_ON_LANE) {
        ecs.connector_length[i] = fmaxf(0.0f, ecs.connector_length[i] - dt);
    }

    /* The reservation table is intentionally coarse: it has node/time slots but
       no turn-direction key.  Use it only for signalized/directed conflicts.
       Unsignal intersections use the right-hand-priority rule above; a coarse
       node slot would otherwise override the physical priority owner. */
    if (
        has_next
        && turn_lane_ok
        && !unsignal_node
        && !self_inside_intersection_box
        && directional_conflict_present
        && dist_to_end < 30.0f
    ) {
        int node = road.lane_end_node[lane];

        float arrival_time = current_time + dist_to_end / fmaxf(v, 1.0f);
        float angle = turn_angle_deg(lane, next_lane, road);

        float crossing_time = 1.6f;
        if (angle > 45.0f) crossing_time += 0.8f;
        if (angle > 100.0f) crossing_time += 1.0f;
        crossing_time += human ? 0.45f : 0.20f;

        bool reserved = try_reserve_slot_ecs(
            i,
            node,
            arrival_time,
            crossing_time,
            reservation_table,
            road.num_nodes,
            current_time,
            metrics
        );

        reservation_granted = reserved;

        if (!reserved) {
            float crawl_v = human ? YIELD_CREEP_SPEED_HUMAN : YIELD_CREEP_SPEED_AV;
            float a = (crawl_v - v) / fmaxf(dt, 0.01f);
            float no_front_gap = fmaxf(STALL_RECOVERY_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 5.0f);
            bool can_creep_to_line =
                dist_to_end > DEFAULT_STOP_OFFSET + SIGNAL_CREEP_HOLD_DIST
                && perception.front_gap[i] > no_front_gap;

            if (can_creep_to_line && v < crawl_v) {
                float creep_a = clampf_cuda(a, 0.0f, max_accel * 0.45f);
                acc_cmd = fmaxf(acc_cmd, creep_a);
                acc_cmd = fminf(acc_cmd, max_accel * 0.45f);
            } else {
                acc_cmd = fminf(acc_cmd, clampf_cuda(a, -max_decel, 0.0f));
            }

            atomicAdd(&metrics[METRIC_COOP_YIELD], 1.0f);
            atomicAdd(&metrics[METRIC_INTERSECTION_WAIT], dt);
        }
    }

    float stop_line_dist =
        stop_s < 1.0e8f
        ? stop_s - s_curr
        : dist_to_end - DEFAULT_STOP_OFFSET;

    float rt = fmaxf(ecs.reaction_time[i], human ? 0.75f : 0.12f);
    float comfortable_b = human ? (0.78f * MAX_DECEL_HUMAN) : (0.86f * MAX_DECEL_AV);
    comfortable_b = fmaxf(comfortable_b, 1.2f);
    float comfortable_stop =
        v * rt
        + (v * v) / fmaxf(2.0f * comfortable_b, 0.1f)
        + INTERSECTION_STOP_BUFFER;

    bool yellow_committed =
        sig_state == LIGHT_YELLOW
        && stop_line_dist <= comfortable_stop * 0.72f
        && dist_to_end <= fmaxf(6.0f, v * dt + 2.0f);

    bool signal_permits_connector = sig_state == LIGHT_GREEN || yellow_committed;
    if (sig_state == LIGHT_RED) signal_permits_connector = false;
    // EN/KO: Once already inside the intersection box, keep clearing it even if the light changes.
    if (self_inside_intersection_box) signal_permits_connector = true;

    if (
        sig_state != LIGHT_GREEN
        && !self_inside_intersection_box
        && stop_s < 1.0e8f
        && dist_to_end > 0.0f
    ) {
        float predicted_s = s_curr + fmaxf(0.0f, v + acc_cmd * dt) * dt;

        if (predicted_s > stop_s - 0.15f && !yellow_committed) {
            acc_cmd = -EMERGENCY_DECEL;
        }
    }

    if (
        has_next
        && !self_inside_intersection_box
        && !signal_permits_connector
        && dist_to_end < 8.0f
    ) {
        float stop_dist = fmaxf(dist_to_end - DEFAULT_STOP_OFFSET, 0.55f);
        float req = -(v * v) / fmaxf(2.0f * stop_dist, 0.5f);
        acc_cmd = fminf(acc_cmd, clampf_cuda(req, -EMERGENCY_DECEL, 0.0f));
    }

    if (
        has_next
        && turn_needs_dedicated_lane
        && !turn_lane_ok
    ) {
        atomicAdd(&metrics[METRIC_TURN_LANE_ILLEGAL], 1.0f);

        bool no_adjacent_turn_lane = turn_lc_target < 0 || turn_lc_target >= road.num_lanes;
        bool too_close_for_lane_change =
            dist_to_end <= fmaxf(
                TURN_LANE_MIN_LC_DIST,
                fmaxf(1.0f, v) * (human ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV) * 0.45f
                    + DEFAULT_STOP_OFFSET + TURN_LANE_STOP_BUFFER
            );

        if (no_adjacent_turn_lane || too_close_for_lane_change) {
            acc_cmd = fminf(acc_cmd, turn_lane_hold_accel_ecs(dist_to_end, v, ecs.driver_type[i]));
            atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
        }
    }

    /*
        Last-resort anti-stall recovery.  This is deliberately conservative:
        it only runs when no route-front vehicle is close, the vehicle is
        practically stopped, and there is still room before the hard stop line.
        It fixes the common deadlock where red/yield/reservation logic returns
        zero acceleration forever even though there is no car ahead.
    */
    bool signal_requires_stop =
        sig_state == LIGHT_RED
        || (sig_state == LIGHT_YELLOW && !yellow_committed);
    float stop_line_gap = stop_s < 1.0e8f ? stop_s - s_curr : dist_to_end - DEFAULT_STOP_OFFSET;
    bool hard_signal_hold = signal_requires_stop && stop_line_gap <= SIGNAL_CREEP_HOLD_DIST;
    bool front_clear_for_creep =
        perception.front_gap[i] > fmaxf(STALL_RECOVERY_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 5.0f);
    bool hard_unsignal_hold =
        unsignal_blocked
        && !unsignal_deadlock_release
        && !front_clear_for_creep
        && dist_to_end <= DEFAULT_STOP_OFFSET + UNSIGNAL_PRIORITY_NEAR_LINE_DIST
        && ecs.connector_length[i] < (human ? 0.65f : 0.45f);

    if (
        has_next
        && v < 0.32f
        && acc_cmd <= 0.02f
        && front_clear_for_creep
        && !hard_signal_hold
        && !hard_unsignal_hold
        && !(turn_needs_dedicated_lane && !turn_lane_ok)
        && dist_to_end > STALL_RECOVERY_MIN_END_DIST
    ) {
        float creep_v = unsignal_deadlock_release
            ? (human ? DEADLOCK_RELEASE_CREEP_HUMAN : DEADLOCK_RELEASE_CREEP_AV)
            : (signal_requires_stop
                ? (human ? SIGNAL_CREEP_SPEED_HUMAN : SIGNAL_CREEP_SPEED_AV)
                : (human ? YIELD_CREEP_SPEED_HUMAN : YIELD_CREEP_SPEED_AV));
        float creep_a = (creep_v - v) / fmaxf(dt, 0.01f);

        if (creep_a > 0.0f) {
            creep_a = clampf_cuda(creep_a, 0.0f, max_accel * 0.40f);
            acc_cmd = fmaxf(acc_cmd, creep_a);
            acc_cmd = fminf(acc_cmd, max_accel * 0.40f);
            atomicAdd(&metrics[METRIC_QUEUE_DELAY_SUM], dt);
            atomicAdd(&metrics[METRIC_QUEUE_DELAY_COUNT], 1.0f);
        }
    }

    // EN: v7 smart release layer.  If a car is stopped with a real route-front
    //     gap, prefer moving it through the available space instead of leaving a
    //     zero-acceleration deadlock.  This does not override a hard red stop or
    //     an unresolved mandatory turn-lane hold.
    // KO: v7 지능형 출발 레이어입니다. 도로 좌표 기준 앞공간이 실제로 비어 있는데
    //     0가속도로 멈춰 있으면, 데드락을 유지하지 않고 사용 가능한 공간으로 이동합니다.
    //     단, 적색 정지선 정지와 아직 해결되지 않은 회전 전용차로 대기는 넘지 않습니다.
    bool smart_front_clear = perception.front_gap[i] > fmaxf(
        SMART_STALL_FRONT_GAP,
        ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 0.85f
    );
    bool smart_red_hold = signal_requires_stop
        && !self_inside_intersection_box
        && stop_line_gap <= SIGNAL_CREEP_HOLD_DIST + 1.0f;
    bool smart_wrong_lane_hold = turn_needs_dedicated_lane && !turn_lane_ok;
    float smart_wait_time = ecs.connector_length[i];
    if (!isfinite(smart_wait_time) || smart_wait_time < 0.0f || ecs.vehicle_state[i] != VEH_ON_LANE) {
        smart_wait_time = 0.0f;
    }
    if (
        has_next
        && v < SMART_STALL_SPEED
        && acc_cmd <= 0.04f
        && smart_front_clear
        && !smart_red_hold
        && !smart_wrong_lane_hold
        && (
            self_inside_intersection_box
            || dist_to_end <= DEFAULT_STOP_OFFSET + PRIORITY_GATE_NEAR_LINE_DIST + 2.0f
            || smart_wait_time >= SMART_STALL_RELEASE_WAIT
        )
    ) {
        if (unsignal_blocked && smart_wait_time >= SMART_STALL_RELEASE_WAIT && smart_front_clear) {
            unsignal_deadlock_release = true;
        }
        float smart_v = self_inside_intersection_box
            ? (human ? CONNECTOR_INBOX_MIN_CLEAR_SPEED_HUMAN : CONNECTOR_INBOX_MIN_CLEAR_SPEED_AV)
            : (human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV);
        float smart_a = (smart_v - v) / fmaxf(dt, 0.01f);
        smart_a = clampf_cuda(smart_a, 0.0f, max_accel * SMART_STALL_RELEASE_ACCEL_SCALE);
        if (smart_a > 0.0f) {
            acc_cmd = fmaxf(acc_cmd, smart_a);
            acc_cmd = fminf(acc_cmd, max_accel * SMART_STALL_RELEASE_ACCEL_SCALE);
            atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
        }
    }

#if MIDROAD_NEG_ACCEL_WATCHDOG_ENABLED
    {
        // EN: Mid-link stale-brake watchdog.  Some negative accelerations are
        //     legitimate near a stop line, blocked conflict path, or mandatory
        //     turn lane.  Away from those hard holds, a stopped vehicle with a
        //     clear route-front gap must not keep a stale negative acceleration.
        // KO: 링크 중간 stale brake 감시입니다. 정지선/충돌 경로/필수 진출 차선 대기가
        //     아닌데 앞 경로가 비어 있고 거의 정지한 차량은 음수 가속도를 계속 유지하지
        //     않게 합니다.
        float stale_clear_gap = fmaxf(
            MIDROAD_NEG_ACCEL_CLEAR_FRONT_GAP,
            ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.25f
        );
        bool midroad_clear_brake =
            v <= MIDROAD_NEG_ACCEL_CLEAR_SPEED
            && acc_cmd < -0.05f
            && perception.front_gap[i] > stale_clear_gap
            && dist_to_end > MIDROAD_NEG_ACCEL_CLEAR_MIN_END_DIST
            && !hard_signal_hold
            && !hard_unsignal_hold
            && !smart_wrong_lane_hold
            && !(turn_needs_dedicated_lane && !turn_lane_ok);
        if (midroad_clear_brake) {
            float release_v = human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV;
            float release_a = (release_v - v) / fmaxf(dt, 0.01f);
            release_a = clampf_cuda(release_a, 0.0f, max_accel * MIDROAD_NEG_ACCEL_RELEASE_SCALE);
            acc_cmd = fmaxf(acc_cmd, release_a);
            if (ecs.accel[i] < 0.0f) ecs.accel[i] = 0.0f;
            if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
        }
    }
#endif

#if MIDROAD_ZERO_ACCEL_WATCHDOG_ENABLED
    {
        // EN: v16 cleared stale negative acceleration, but a few lane-drop /
        //     no-start states could still leave target acceleration exactly zero
        //     while the lane ahead was open.  Away from hard red/yield/wrong-lane
        //     holds, restart the car with a small positive command.
        // KO: v16에서 음수 가속도는 지웠지만 차로 감소/no-start 상태에서 앞이
        //     비어 있는데 target accel이 정확히 0으로 굳는 경우가 남았습니다.
        //     적색/양보/필수 진출 차선 대기가 아니면 작은 양수 가속도로 재출발합니다.
        float zero_clear_gap = fmaxf(
            MIDROAD_ZERO_ACCEL_CLEAR_FRONT_GAP,
            ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v) * 1.10f
        );
        bool midroad_zero_hold =
            has_next
            && v <= MIDROAD_ZERO_ACCEL_CLEAR_SPEED
            && acc_cmd >= -0.04f
            && acc_cmd <= 0.035f
            && perception.front_gap[i] > zero_clear_gap
            && dist_to_end > MIDROAD_ZERO_ACCEL_MIN_END_DIST
            && !hard_signal_hold
            && !hard_unsignal_hold
            && !smart_wrong_lane_hold
            && !(turn_needs_dedicated_lane && !turn_lane_ok);
        if (midroad_zero_hold) {
            float release_v = human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV;
            float release_a = (release_v - v) / fmaxf(dt, 0.01f);
            release_a = clampf_cuda(release_a, 0.0f, max_accel * MIDROAD_ZERO_ACCEL_RELEASE_SCALE);
            if (release_a > 0.0f) {
                acc_cmd = fmaxf(acc_cmd, release_a);
                if (ecs.accel[i] < 0.0f) ecs.accel[i] = 0.0f;
                if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
            }
        }
    }
#endif


    int ll = geometric_left_neighbor_ecs(lane, road);
    int rr = geometric_right_neighbor_ecs(lane, road);

    int upcoming_event_turn = TURN_STRAIGHT;
    float upcoming_event_dist = 1.0e9f;
    int lookahead_lc_target = -1;
    if (
        ecs.lane_change_active[i] == 0
        && has_next
        && !turn_needs_dedicated_lane
    ) {
        lookahead_lc_target = upcoming_exit_lane_step_target_ecs(
            i,
            lane,
            ecs,
            road,
            dist_to_end,
            v,
            ecs.driver_type[i],
            &upcoming_event_turn,
            &upcoming_event_dist
        );
    }

    int lane_drop_lc_target = -1;
#if LANE_COUNT_CHANGE_PREP_ENABLED
    if (
        ecs.lane_change_active[i] == 0
        && has_next
        && turn == TURN_STRAIGHT
        && !turn_needs_dedicated_lane
    ) {
        int prep_target = lane_count_reduction_step_target_ecs(lane, next_lane, road);
        if (valid_lane_ecs(prep_target, road)) {
            int cur_idx = -1;
            int from_count = lane_group_count_and_index_ecs(lane, road, cur_idx);
            int to_count = lane_group_count_ecs(next_lane, road);
            int dropped = max(1, from_count - to_count);
            float prep_dist = LANE_COUNT_CHANGE_PREP_MIN_DIST
                + LANE_COUNT_CHANGE_PREP_PER_DROPPED * (float)dropped
                + fmaxf(0.0f, v) * 2.2f;
            prep_dist = clampf_cuda(prep_dist, LANE_COUNT_CHANGE_PREP_MIN_DIST, LANE_COUNT_CHANGE_PREP_MAX_DIST);
            if (
                from_count > to_count
                && dist_to_end <= prep_dist
                && dist_to_end > fmaxf(lc_no_start_dist, LANE_COUNT_CHANGE_PREP_MIN_DIST)
            ) {
                lane_drop_lc_target = prep_target;
            }
        }
    }
#endif

    int lc_target = -1;
    bool mandatory_lc = false;
    bool opportunistic_lc = false;
    if (
        ecs.lane_change_active[i] == 0
        && turn_needs_dedicated_lane
        && !turn_lane_ok
        && turn_lc_target >= 0
        && turn_lc_target < road.num_lanes
        && dist_to_end > fmaxf(TURN_LANE_MIN_LC_DIST, lc_no_start_dist)
    ) {
        lc_target = turn_lc_target;
        mandatory_lc = true;
        atomicAdd(&metrics[METRIC_TURN_LANE_PREP], 1.0f);
    } else if (
        valid_lane_ecs(lookahead_lc_target, road)
        && dist_to_end > lc_no_start_dist
    ) {
        // EN: Prepare early for a downstream exit/interchange edge lane.  The
        //     vehicle changes one lane at a time; open-lane spreading resumes
        //     after it reaches the required side.
        // KO: 다음 램프/좌우 진출이 다가오면 미리 가장자리 차로 쪽으로 한 차로씩
        //     이동합니다. 필요한 쪽에 도달하면 다시 빈 차선 분산 로직이 작동합니다.
        lc_target = lookahead_lc_target;
        mandatory_lc = true;
        atomicAdd(&metrics[METRIC_TURN_LANE_PREP], 1.0f);
    } else if (
        valid_lane_ecs(lane_drop_lc_target, road)
        && dist_to_end > lc_no_start_dist
    ) {
        // EN: Non-blocking lane-drop preparation.  The disappearing edge lane
        //     tries to merge inward early, but failure does not freeze the car;
        //     the connector can still clear the node one-by-one if the gap never
        //     opens before the taper.
        // KO: 차로 감소 사전 합류입니다. 사라지는 가장자리 차로는 미리 안쪽으로
        //     합류를 시도하지만, 실패해도 차량을 멈춰 세우지 않고 노드에서는 한 대씩
        //     통과할 수 있게 둡니다.
        lc_target = lane_drop_lc_target;
        mandatory_lc = true;
        atomicAdd(&metrics[METRIC_TURN_LANE_PREP], 1.0f);
    } else if (
        ecs.lane_change_active[i] == 0
        && next_lane >= 0
        && next_lane != lane
        && dist_to_end > lc_no_start_dist
        && upcoming_event_dist <= OPEN_LANE_EMPTIEST_MIN_EXIT_DIST
    ) {
        // EN/KO: Do not blindly chase a cached route lane far upstream.  Only
        // honor this route-lane step when a downstream exit is near; otherwise
        // the open-lane scan below may use the emptier lane bundle.
        if (ll == next_lane) lc_target = ll;
        else if (rr == next_lane) lc_target = rr;
    } else if (
        ecs.lane_change_active[i] == 0
        && (has_next || lane_group_count_ecs(lane, road) > 1)
        && (turn == TURN_STRAIGHT || upcoming_event_dist > OPEN_LANE_EMPTIEST_MIN_EXIT_DIST)
        && !turn_needs_dedicated_lane
    ) {
        // EN: First try a real open-lane escape/spread.  If the lane ahead is
        //     blocked and an adjacent lane has enough front/rear gap, move there.
        //     This also helps ramp/on-ramp vehicles disperse across the mainline
        //     instead of staying glued to the entry edge or to the center lane.
        // KO: 먼저 실제 빈 차선 회피/분산을 시도합니다. 앞 차선이 막혔고 옆 차선
        //     앞/뒤 gap이 충분하면 그쪽으로 옮깁니다. 나들목에서 들어온 차량도
        //     진입 가장자리나 중앙 차선에 고착되지 않고 본선 여러 차로로 퍼집니다.
        int open_target = pick_open_lane_target_ecs(
            i,
            lane,
            ecs,
            road,
            grid,
            perception,
            current_time,
            max_entities,
            true,
            true
        );
        if (valid_lane_ecs(open_target, road)) {
            lc_target = open_target;
            mandatory_lc = false;
            opportunistic_lc = true;
        }
#if CRUISE_RANDOM_LANE_CHANGE_ENABLED
        else if (
            ecs.lc_cooldown[i] <= CRUISE_RANDOM_LANE_COOLDOWN_READY
            && road.lane_length[lane] >= CRUISE_RANDOM_LANE_MIN_LINK_LENGTH
            && dist_to_end > fmaxf(CRUISE_RANDOM_LANE_MIN_DIST_TO_NODE, lc_no_start_dist + 20.0f)
        ) {
            uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
            uint32_t h = hash_u32_ecs(
                ((uint32_t)(i + 1) * 1103515245u)
                ^ ((uint32_t)(ecs.route_id[i] + 23) * 2654435761u)
                ^ ((uint32_t)(road.lane_start_node[lane] + 31) * 747796405u)
                ^ (slot * 2891336453u)
            );
            float p = CRUISE_RANDOM_LANE_CHANGE_PROB
                * (0.70f + 0.38f * clampf_cuda(ecs.aggressiveness[i], 0.0f, 1.0f));
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
        ecs.lane_change_active[i] == 0
        && !has_next
        && lane_group_count_ecs(lane, road) > 1
        && road.lane_length[lane] >= DESTINATION_SPREAD_MIN_LINK_LENGTH
        && dist_to_end > fmaxf(DESTINATION_SPREAD_MIN_DIST_TO_END, lc_no_start_dist + 20.0f)
#else
        false
#endif
    ) {
        // EN: On the final destination link there is no upcoming turn that
        //     requires vehicles to stay on one edge.  Keep using empty-lane
        //     and stable random spread so arrivals do not all queue on the
        //     side they entered from.
        // KO: 최종 도착 링크에서는 더 이상 특정 가장자리 차선으로 나갈 필요가
        //     없습니다. 빈 차선 탐색과 안정 랜덤 분산을 계속 적용해 도착 차량이
        //     진입한 한쪽 차선에만 줄 서지 않게 합니다.
        int open_target = pick_open_lane_target_ecs(
            i,
            lane,
            ecs,
            road,
            grid,
            perception,
            current_time,
            max_entities,
            true,
            true
        );
        if (valid_lane_ecs(open_target, road)) {
            lc_target = open_target;
            opportunistic_lc = true;
        }
#if CRUISE_RANDOM_LANE_CHANGE_ENABLED
        else if (ecs.lc_cooldown[i] <= CRUISE_RANDOM_LANE_COOLDOWN_READY) {
            uint32_t slot = (uint32_t)floorf(current_time / CRUISE_RANDOM_LANE_DECISION_PERIOD);
            uint32_t h = hash_u32_ecs(
                ((uint32_t)(i + 19) * 1103515245u)
                ^ ((uint32_t)(ecs.route_id[i] + 41) * 2654435761u)
                ^ ((uint32_t)(lane + 97) * 747796405u)
                ^ (slot * 2891336453u)
            );
            float p = clampf_cuda(
                DESTINATION_SPREAD_RANDOM_PROB * (0.65f + 0.45f * clampf_cuda(ecs.aggressiveness[i], 0.0f, 1.0f)),
                0.035f,
                0.42f
            );
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
        // EN/KO: Always refresh target-lane gaps for the exact lane chosen by
        // the decision layer.  This prevents using stale center-lane perception
        // after a random or congestion-escape target was selected.
        find_lane_neighbors_ecs(
            i,
            lc_target,
            ecs,
            road,
            grid,
            max_entities,
            human ? 160.0f : 140.0f,
            nullptr,
            perception.target_front_gap[i],
            perception.target_front_speed[i],
            perception.target_rear_gap[i],
            perception.target_rear_speed[i]
        );

        int lc_sig = indicator_from_lateral_move_ecs(lane, lc_target, road);
        if (lc_sig != INDICATOR_NONE && ecs.turn_signal != nullptr) {
            if (ecs.turn_signal[i] == lc_sig) {
                if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = fminf(ecs.turn_signal_time[i] + dt, 60.0f);
            } else {
                ecs.turn_signal[i] = lc_sig;
                if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = dt;
            }
        }

        bool ok = mobil_decision_ecs(
            i,
            lc_target,
            ecs,
            road,
            grid,
            perception,
            metrics,
            current_time,
            max_entities,
            mandatory_lc,
            opportunistic_lc
        );

        if (ok) {
            decision.wants_lane_change[i] = 1;
            decision.lane_change_target[i] = lc_target;
        }
    }

    float connector_trigger_dist = CONNECTOR_EXIT_EPS;
    if (has_next) {
        float v_after_cmd = fmaxf(0.0f, v + acc_cmd * dt);
        float box_depth = intersection_box_depth_ecs(lane, next_lane, road);
        connector_trigger_dist = fmaxf(
            fmaxf(CONNECTOR_EXIT_EPS, box_depth),
            v_after_cmd * dt + CONNECTOR_TRIGGER_MARGIN
        );
    }

    bool priority_permits_connector =
        self_inside_intersection_box
        || !unsignal_node
        || !unsignal_blocked
        || unsignal_deadlock_release;

    if (
        has_next
        && reservation_granted
        && signal_permits_connector
        && priority_permits_connector
        && turn_lane_ok
        && dist_to_end <= connector_trigger_dist
    ) {
        decision.wants_connector[i] = 1;
        decision.connector_target_lane[i] = next_lane;
    }

    decision.desired_speed[i] = desired_v;
    decision.target_accel[i] = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);
}

// ============================================================
// Lane Change System
// ============================================================

__global__ void lane_change_system_kernel(
    ECSArrays ecs,
    DecisionSoA decision,
    RoadNetwork road,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[i] != VEH_ON_LANE) return;

    bool human = ecs.driver_type[i] == HUMAN;

    if (ecs.lc_cooldown[i] > 0.0f) {
        ecs.lc_cooldown[i] = fmaxf(0.0f, ecs.lc_cooldown[i] - dt);
    }

    if (
        decision.wants_lane_change[i] != 0
        && ecs.lane_change_active[i] == 0
    ) {
        int target = decision.lane_change_target[i];

        if (target >= 0 && target < road.num_lanes) {
            int lane = ecs.lane_id[i];
            if (!valid_lane_ecs(lane, road) || !same_approach_same_direction_lanes_ecs(lane, target, road)) return;
            float dist_to_node = road.lane_length[lane] - ecs.s[i];
            float no_start_dist = lane_change_no_start_distance_ecs(lane, road);
            float finish_dist = lane_change_finish_distance_ecs(lane, road);
            int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
            bool inside_box =
                valid_lane_ecs(next_lane, road)
                && lane_connected(lane, next_lane, road)
                && inside_intersection_box_ecs(dist_to_node, lane, next_lane, road);
            // EN: Never start a lane change in the intersection/connector area.
            //     Cars must prepare on the straight lane-running segment before the node.
            // KO: 교차로/connector 영역에서는 차선변경을 새로 시작하지 않습니다.
            //     차량은 노드에 도착하기 전 차선 주행 모드에서 미리 차선을 맞춥니다.
            if (inside_box || dist_to_node < fmaxf(no_start_dist, finish_dist)) {
                return;
            }

            int sig = indicator_from_lateral_move_ecs(lane, target, road);
            if (sig != INDICATOR_NONE && ecs.turn_signal != nullptr) {
                if (ecs.turn_signal[i] != sig) {
                    ecs.turn_signal[i] = sig;
                    if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = dt;
                    return;
                }
                if (ecs.turn_signal_time != nullptr && ecs.turn_signal_time[i] < LANE_CHANGE_SIGNAL_LEAD_TIME) {
                    ecs.turn_signal_time[i] = fminf(ecs.turn_signal_time[i] + dt, LANE_CHANGE_SIGNAL_LEAD_TIME);
                    return;
                }
            }

            ecs.lane_change_active[i] = 1;
            ecs.lane_change_from_lane[i] = lane;
            ecs.lane_change_to_lane[i] = target;
            ecs.lane_change_t[i] = 0.0f;
            ecs.lane_change_duration[i] =
                human ? LANE_CHANGE_DURATION_HUMAN : LANE_CHANGE_DURATION_AV;
            ecs.lc_cooldown[i] =
                human ? LC_COOLDOWN_HUMAN : LC_COOLDOWN_AV;
            if (sig != INDICATOR_NONE && ecs.turn_signal != nullptr) {
                ecs.turn_signal[i] = sig;
                if (ecs.turn_signal_time != nullptr) {
                    ecs.turn_signal_time[i] = fmaxf(ecs.turn_signal_time[i], LANE_CHANGE_SIGNAL_LEAD_TIME);
                }
            }
        }
    }
}

// ============================================================
// Motion System
// ============================================================

__global__ void motion_system_kernel(
    ECSArrays ecs,
    DecisionSoA decision,
    RoadNetwork road,
    PerceptionSoA perception,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[i] != VEH_ON_LANE) return;

    if (decision.should_exit[i] != 0) {
        ecs.alive[i] = ENTITY_FREE;
        atomicAdd(&metrics[METRIC_EXITED], 1.0f);
        atomicAdd(&metrics[METRIC_TRAVEL_TIME], fmaxf(0.0f, current_time - ecs.entry_time[i]));
        return;
    }

    int lane = ecs.lane_id[i];

    if (lane < 0 || lane >= road.num_lanes) {
        ecs.alive[i] = ENTITY_FREE;
        return;
    }

    bool human = ecs.driver_type[i] == HUMAN;
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;

    float s_curr = ecs.s[i];
    float v = ecs.speed[i];

    float target_acc_cmd = clampf_cuda(
        decision.target_accel[i],
        -EMERGENCY_DECEL,
        max_accel
    );

    float prev_accel_for_delay = ecs.accel[i];
    float L_for_stale = fmaxf(road.lane_length[lane], 0.1f);
    float dist_to_end_for_stale = fmaxf(0.0f, L_for_stale - s_curr);
    bool front_clear_for_stale =
        perception.front_gap[i] > fmaxf(
            STALE_BRAKE_CLEAR_FRONT_GAP,
            ecs.length[i] + MIN_BUMPER_GAP + 6.0f
        );
    if (
        prev_accel_for_delay < -1.5f
        && target_acc_cmd > 0.05f
        && v < SMART_STALL_SPEED_EPS
        && front_clear_for_stale
    ) {
        // EN/KO: Contact repair can leave an emergency-brake value in accel.
        // If the car is now stopped and the route front is clear, do not feed
        // that stale negative value into the reaction-delay filter.
        prev_accel_for_delay = 0.0f;
        float clear_a = (human ? STALE_BRAKE_CLEAR_ACCEL_HUMAN : STALE_BRAKE_CLEAR_ACCEL_AV) * max_accel;
        target_acc_cmd = fmaxf(target_acc_cmd, clear_a);
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
    }

#if MIDROAD_NEG_ACCEL_WATCHDOG_ENABLED
    if (
        prev_accel_for_delay < -0.25f
        && target_acc_cmd < 0.02f
        && v <= MIDROAD_NEG_ACCEL_CLEAR_SPEED
        && front_clear_for_stale
        && dist_to_end_for_stale > MIDROAD_NEG_ACCEL_CLEAR_MIN_END_DIST
    ) {
        // EN/KO: A negative target far from any node with no front car is almost
        // always a stale hold from contact/local-avoidance/priority repair.  Clear
        // it here as a final guard before the reaction-delay filter.
        prev_accel_for_delay = 0.0f;
        float clear_a = max_accel * MIDROAD_NEG_ACCEL_RELEASE_SCALE;
        target_acc_cmd = fmaxf(target_acc_cmd, clear_a);
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
    }
#endif

#if MIDROAD_ZERO_ACCEL_WATCHDOG_ENABLED
    if (
        target_acc_cmd >= -0.04f
        && target_acc_cmd <= 0.035f
        && v <= MIDROAD_ZERO_ACCEL_CLEAR_SPEED
        && front_clear_for_stale
        && dist_to_end_for_stale > fmaxf(28.0f, MIDROAD_ZERO_ACCEL_MIN_END_DIST + 8.0f)
    ) {
        // EN/KO: Final motion-side guard for stale zero-accel states far from
        // any stop line.  The decision kernel is the primary authority for red/
        // yield holds, so this only runs well upstream of a node.
        target_acc_cmd = fmaxf(target_acc_cmd, max_accel * MIDROAD_ZERO_ACCEL_RELEASE_SCALE);
        if (prev_accel_for_delay < 0.0f) prev_accel_for_delay = 0.0f;
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
    }
#endif

    float acc_cmd = apply_reaction_delay_accel(

        prev_accel_for_delay,
        target_acc_cmd,
        ecs.reaction_time[i],
        ecs.driver_type[i],
        dt,
        metrics
    );
    acc_cmd = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);

    if (acc_cmd < -2.0f) {
        atomicAdd(&metrics[METRIC_COMFORT_BRAKE], 1.0f);
    }

    float v_next = fmaxf(0.0f, v + acc_cmd * dt);
    float s_next = s_curr + 0.5f * (v + v_next) * dt;

    if (v <= STOPPED_NEG_ACCEL_ZERO_SPEED && v_next <= STOPPED_NEG_ACCEL_ZERO_SPEED && acc_cmd < 0.0f) {
        // EN/KO: A vehicle that is already stopped has no physical backward
        // acceleration.  Store zero instead of keeping a negative stale value.
        acc_cmd = 0.0f;
    }

    // EN: Hard no-pass guard.  The decision/IDM model tries to slow down,
    //     but this cap prevents a discrete time step from jumping through the
    //     rear bumper of the front vehicle.
    // KO: 앞차 투과 방지 하드 가드입니다. 의사결정/IDM이 감속을 시도하더라도,
    //     한 프레임 이동량이 앞차 뒤 범퍼를 넘어가지 못하게 직접 제한합니다.
    if (perception.front_gap[i] < 1.0e8f) {
        float max_advance = fmaxf(0.0f, perception.front_gap[i] - ANTI_PASS_THROUGH_GAP);
        float attempted = s_next - s_curr;
        if (attempted > max_advance) {
            s_next = s_curr + max_advance;
            v_next = fminf(v_next, max_advance / fmaxf(dt, 0.001f));
            acc_cmd = (v_next - v) / fmaxf(dt, 0.001f);
            acc_cmd = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);
            atomicAdd(&metrics[METRIC_PENETRATION_PREVENTED], 1.0f);
            atomicAdd(&metrics[METRIC_ANTI_COLLISION_BRAKE], 1.0f);
        }
    }

    if (s_next + NO_BACKWARD_EPS < s_curr) {
        s_next = s_curr;
        v_next = 0.0f;
        acc_cmd = 0.0f;
    }

    float L = fmaxf(road.lane_length[lane], 0.1f);
    bool force_lc_finish_now = false;

    if (ecs.lane_change_active[i] != 0) {
        // EN: Finish lane changes before the intersection, never inside the node box.
        //     v17 adds a no-start-barrier escape: if a lane-drop/open-lane change
        //     freezes at the barrier with a closed target gap, either abort back
        //     to the original lane early or commit if already mostly across.
        //     This prevents the common 4->3 deadlock where accel becomes exactly
        //     zero and the vehicle parks at lc_stop_s.
        // KO: 차선변경은 교차로 안이 아니라 노드 이전에서 끝내야 합니다. v17은
        //     no-start 경계에서 target gap이 닫혀 멈춘 경우, 초반이면 원래 차로로
        //     취소하고 이미 많이 넘어갔으면 완료시켜 4->3 구간 0가속도 정지를 막습니다.
        float lc_stop_s = fmaxf(0.0f, L - lane_change_finish_distance_ecs(lane, road));
        int from_ln = ecs.lane_change_from_lane[i];
        int to_ln = ecs.lane_change_to_lane[i];
        float dur = fmaxf(ecs.lane_change_duration[i], 0.1f);
        float t_prog = clampf_cuda(ecs.lane_change_t[i] / dur, 0.0f, 1.0f);
        float freeze_front = fmaxf(LC_ACTIVE_FREEZE_FRONT_MIN, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v_next) * 0.35f);
        float freeze_rear = fmaxf(LC_ACTIVE_FREEZE_REAR_MIN, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, perception.target_rear_speed[i]) * 0.45f);
        bool active_lc_gap_unsafe_pre =
            perception.target_front_gap[i] < freeze_front
            || perception.target_rear_gap[i] < freeze_rear;
        bool at_lc_barrier = s_next > lc_stop_s + LANE_DROP_ACTIVE_LC_BARRIER_EXTRA
            || (s_curr >= lc_stop_s - LANE_DROP_ACTIVE_LC_BARRIER_EXTRA && v_next < 0.75f);

        if (
            at_lc_barrier
            && active_lc_gap_unsafe_pre
            && valid_lane_ecs(from_ln, road)
            && valid_lane_ecs(to_ln, road)
        ) {
            if (t_prog < 0.50f || t_prog <= LANE_DROP_ACTIVE_LC_ABORT_T) {
                ecs.lane_change_active[i] = 0;
                ecs.lane_change_from_lane[i] = from_ln;
                ecs.lane_change_to_lane[i] = from_ln;
                ecs.lane_change_t[i] = 0.0f;
                ecs.lane_id[i] = from_ln;
                lane = from_ln;
                L = fmaxf(road.lane_length[lane], 0.1f);
                s_next = fminf(s_next, L - 0.05f);
                acc_cmd = fmaxf(acc_cmd, 0.0f);
                if (ecs.turn_signal != nullptr) {
                    ecs.turn_signal[i] = INDICATOR_NONE;
                    if (ecs.turn_signal_time != nullptr) ecs.turn_signal_time[i] = 0.0f;
                }
                atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);
            } else {
                force_lc_finish_now = true;
                ecs.lane_change_t[i] = dur;
            }
        }

        if (ecs.lane_change_active[i] != 0 && s_next > lc_stop_s && !force_lc_finish_now) {
            s_next = fmaxf(s_curr, lc_stop_s);
            v_next = fminf(v_next, 0.35f);
            acc_cmd = fminf(acc_cmd, 0.0f);
        }
    }

    if (decision.wants_connector[i] != 0) {
        /*
            Do not crush speed at the lane end when the connector transition is
            already requested.  Let the vehicle carry its current speed into the
            connector and preserve a small overflow as connector_s on entry.
        */
        if (s_next >= L - CONNECTOR_ENTER_EPS) {
            s_next = fmaxf(s_next, L - CONNECTOR_ENTER_EPS);
            s_next = fminf(s_next, L + 0.35f);
        }
    } else {
        if (s_next > L - 0.05f) {
            s_next = L - 0.05f;
            v_next = fminf(v_next, 0.4f);
        }
    }

    ecs.s[i] = s_next;
    ecs.speed[i] = v_next;
    ecs.accel[i] = acc_cmd;

    float px, py, ph;
    lane_xy_heading_from_s(lane, s_next, road, px, py, ph);

    if (ecs.lane_change_active[i] != 0) {
        int from_ln = ecs.lane_change_from_lane[i];
        int to_ln = ecs.lane_change_to_lane[i];

        if (
            from_ln >= 0 && from_ln < road.num_lanes
            && to_ln >= 0 && to_ln < road.num_lanes
        ) {
            float dur = fmaxf(ecs.lane_change_duration[i], 0.1f);
            float t_prev = clampf_cuda(ecs.lane_change_t[i] / dur, 0.0f, 1.0f);
            float t = clampf_cuda((ecs.lane_change_t[i] + dt) / dur, 0.0f, 1.0f);

            // EN: If the target gap closes while the vehicle is still early in the
            // lane change, pause lateral motion and brake/coast instead of sliding
            // into another car.  Once more than half committed, continue clearing.
            // KO: 차선변경 초반에 목표 차로 gap이 닫히면 옆으로 밀고 들어가지 않고
            // lateral 진행을 잠깐 멈춘 뒤 감속/타력 주행합니다. 절반 이상 들어간
            // 경우에는 오히려 빨리 마무리해 겹침을 줄입니다.
            float freeze_front = fmaxf(LC_ACTIVE_FREEZE_FRONT_MIN, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, v_next) * 0.35f);
            float freeze_rear = fmaxf(LC_ACTIVE_FREEZE_REAR_MIN, ecs.length[i] + MIN_BUMPER_GAP + fmaxf(0.0f, perception.target_rear_speed[i]) * 0.45f);
            bool active_lc_gap_unsafe =
                perception.target_front_gap[i] < freeze_front
                || perception.target_rear_gap[i] < freeze_rear;
            if (force_lc_finish_now) {
                t = 1.0f;
            } else if (active_lc_gap_unsafe && t_prev < LC_ACTIVE_FREEZE_T_MAX) {
                t = t_prev;
                v_next = fminf(v_next, fmaxf(0.0f, v - 0.65f * dt));
                acc_cmd = fminf(acc_cmd, -0.65f);
                atomicAdd(&metrics[METRIC_LC_REJECT], 1.0f);
            }

            float ax, ay, ah;
            float bx, by, bh;

            float s_from = clampf_cuda(s_next, 0.0f, road.lane_length[from_ln]);
            float s_to   = clampf_cuda(s_next, 0.0f, road.lane_length[to_ln]);

            lane_xy_heading_from_s(from_ln, s_from, road, ax, ay, ah);
            lane_xy_heading_from_s(to_ln, s_to, road, bx, by, bh);

            float u = smoothstep01(t);

            px = ax + (bx - ax) * u;
            py = ay + (by - ay) * u;
            ph = wrap_pi(ah + wrap_pi(bh - ah) * u);

            ecs.lane_change_t[i] = t * dur;

            if (t >= 1.0f) {
                atomicAdd(&metrics[METRIC_LANE_CHANGE_TIME_SUM], dur);
                atomicAdd(&metrics[METRIC_LANE_CHANGE_TIME_COUNT], 1.0f);

                ecs.lane_id[i] = to_ln;
                int repaired_after_lc = repair_route_pos_for_current_lane_ecs(to_ln, ecs.route_id[i], ecs.route_pos[i], road);
                if (repaired_after_lc >= 0) {
                    ecs.route_pos[i] = repaired_after_lc;
                }
                ecs.lane_change_active[i] = 0;
                ecs.lane_change_from_lane[i] = to_ln;
                ecs.lane_change_to_lane[i] = to_ln;
                ecs.lane_change_t[i] = 0.0f;
            }
        } else {
            ecs.lane_change_active[i] = 0;
        }
    }

    // EN/KO: Active lane-change freeze/abort/commit may adjust local speed or
    // acceleration after the first write above; store the final values so the
    // debug state and the next reaction-delay step do not keep stale zeros.
    ecs.speed[i] = v_next;
    ecs.accel[i] = acc_cmd;

    ecs.x[i] = px;
    ecs.y[i] = py;

    float steer = 0.0f;
    float yaw_rate = 0.0f;
    float new_h = advance_heading_bicycle_ecs(
        i,
        ecs,
        ph,
        v_next,
        dt,
        steer,
        yaw_rate
    );

    ecs.steer_angle[i] = steer;
    ecs.heading[i] = new_h;

    atomicAdd(&metrics[METRIC_STEER_ABS_SUM], fabsf(steer));
    atomicAdd(&metrics[METRIC_STEER_COUNT], 1.0f);
    atomicAdd(&metrics[METRIC_YAW_RATE_ABS_SUM], fabsf(yaw_rate));
    atomicAdd(&metrics[METRIC_YAW_RATE_COUNT], 1.0f);
}

// ============================================================
// Connector System
// ============================================================

__global__ void connector_enter_system_kernel(
    ECSArrays ecs,
    DecisionSoA decision,
    RoadNetwork road,
    SpatialGrid grid,
    float* metrics,
    float current_time,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[i] != VEH_ON_LANE) return;

    if (decision.wants_connector[i] == 0) return;

    int lane = ecs.lane_id[i];
    int next_lane = decision.connector_target_lane[i];

    if (
        lane < 0 || lane >= road.num_lanes
        || next_lane < 0 || next_lane >= road.num_lanes
        || !lane_connected(lane, next_lane, road)
    ) {
        return;
    }

    int turn = turn_code_from_lanes_ecs(lane, next_lane, road);
    int rid = ecs.route_id[i];
    int rpos = ecs.route_pos[i];
    if (rid >= 0 && rid < road.num_routes && rpos >= 0) {
        int ro0 = road.route_offsets[rid];
        int ro1 = road.route_offsets[rid + 1];
        if (ro1 > ro0 && ro0 + rpos < ro1) {
            int route_turn = road.route_turns[ro0 + rpos];
            turn = effective_turn_code_ecs(lane, next_lane, route_turn, road);
        } else {
            turn = effective_turn_code_ecs(lane, next_lane, TURN_STRAIGHT, road);
        }
    } else {
        turn = effective_turn_code_ecs(lane, next_lane, TURN_STRAIGHT, road);
    }

    int adjusted_next = interchange_receiving_outer_lane_ecs(lane, next_lane, road);
    adjusted_next = receiving_lane_for_turn_ecs(adjusted_next, turn, road);
    adjusted_next = interchange_receiving_outer_lane_ecs(lane, adjusted_next, road);
    if (valid_lane_ecs(adjusted_next, road) && lane_connected(lane, adjusted_next, road)) {
        next_lane = adjusted_next;
        decision.connector_target_lane[i] = adjusted_next;
    }

    int interchange_source_lane = interchange_source_outer_lane_ecs(lane, next_lane, road);
    if (valid_lane_ecs(interchange_source_lane, road) && lane != interchange_source_lane) {
        atomicAdd(&metrics[METRIC_TURN_LANE_ILLEGAL], 1.0f);
        bool late_escape = ecs.connector_length[i] >= fmaxf(MISSED_TURN_ESCAPE_WAIT, WRONG_LANE_STALL_FORCE_WAIT);
        if (!late_escape) {
            atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
            return;
        }
        // EN: Last-resort stale-lane repair.  v12 keeps interchanges edge-only;
        //     however, if a legacy/bad lane id has already trapped the vehicle
        //     at the stop line, allow the protected connector to clear the road
        //     after a short wait rather than freezing forever.
        // KO: 최후의 stale-lane 복구입니다. v12 규칙은 나들목 최외곽 차로를 유지하지만,
        //     이미 잘못된 lane id 때문에 정지선에 갇힌 차량은 짧게 기다린 뒤 보호된
        //     connector로 통과시켜 도로 중간 영구정지를 막습니다.
        atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
    }

    if (!lane_legal_for_turn_ecs(lane, turn, road)) {
        atomicAdd(&metrics[METRIC_TURN_LANE_ILLEGAL], 1.0f);
        bool late_escape = ecs.connector_length[i] >= MISSED_TURN_ESCAPE_WAIT;
        if (!late_escape) {
            return;
        }
        // EN/KO: Last-resort missed-turn escape: the decision kernel already
        // verified a clear front gap and waited briefly.  Connector-entry clearance
        // below still prevents collisions.
        atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
    }

    // EN: Never enter an intersection while a mandatory turn-lane change is still in progress.
    // KO: 회전 전용 차선으로 이동하는 차선변경이 끝나지 않은 상태에서는 교차로에 진입하지 않습니다.
    if ((turn_requires_dedicated_lane_ecs(turn) || valid_lane_ecs(interchange_source_lane, road)) && ecs.lane_change_active[i] != 0) {
        atomicAdd(&metrics[METRIC_TURN_LANE_BLOCK], 1.0f);
        return;
    }

    if (!connector_entry_clear_ecs(i, lane, next_lane, ecs, decision, road, grid, current_time, max_entities)) {
        atomicAdd(&metrics[METRIC_CONNECTOR_SAFE_YIELD], 1.0f);
        atomicAdd(&metrics[METRIC_PRIORITY_ENTRY_BLOCK], 1.0f);
        return;
    }

    float lane_L = fmaxf(road.lane_length[lane], 0.1f);
    float entry_backoff = connector_entry_backoff_ecs(lane, next_lane, road);
    float connector_start_s = fmaxf(0.0f, lane_L - entry_backoff);

    if (ecs.s[i] < connector_start_s - CONNECTOR_ENTER_EPS) {
        return;
    }

    float clen = connector_length_between_lanes(lane, next_lane, road);
    float overflow = fmaxf(0.0f, ecs.s[i] - connector_start_s);

    ecs.vehicle_state[i] = VEH_IN_CONNECTOR;
    ecs.connector_from_lane[i] = lane;
    ecs.connector_to_lane[i] = next_lane;
    ecs.connector_length[i] = clen;
    ecs.connector_s[i] = clampf_cuda(
        CONNECTOR_ENTER_EPS + overflow,
        CONNECTOR_ENTER_EPS,
        fmaxf(CONNECTOR_ENTER_EPS, clen - CONNECTOR_EXIT_EPS)
    );

    ecs.s[i] = connector_start_s;

    ecs.lane_change_active[i] = 0;
    ecs.lane_change_from_lane[i] = lane;
    ecs.lane_change_to_lane[i] = lane;
    ecs.lane_change_t[i] = 0.0f;

    atomicAdd(&metrics[METRIC_CONNECTOR_IN], 1.0f);
}


__device__ __forceinline__ void find_front_in_connector_ecs(
    int self,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float search_radius,
    float& front_gap,
    float& front_speed
) {
    front_gap = 1.0e9f;
    front_speed = 0.0f;

    int from_ln = ecs.connector_from_lane[self];
    int to_ln = ecs.connector_to_lane[self];

    int base = world_cell_index(
        ecs.x[self],
        ecs.y[self],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    int cr = clampi_cuda(
        (int)ceilf(search_radius / fmaxf(grid.cell_size, 0.1f)),
        1,
        WORLD_MAX_CELL_RADIUS
    );

    float self_progress = ecs.connector_s[self];
    float clen = fmaxf(ecs.connector_length[self], CONNECTOR_MIN_LEN);

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j != self && ecs.alive[j] == ENTITY_ALIVE) {
                    float ds = 1.0e9f;

                    if (
                        ecs.vehicle_state[j] == VEH_IN_CONNECTOR
                        && ecs.connector_from_lane[j] == from_ln
                        && ecs.connector_to_lane[j] == to_ln
                    ) {
                        ds = ecs.connector_s[j] - self_progress;
                    } else if (
                        ecs.vehicle_state[j] == VEH_ON_LANE
                        && ecs.lane_id[j] == to_ln
                    ) {
                        /*
                            The connector exits at handoff_s on same-node
                            geometry.  Vehicles before that point are behind
                            the connector exit, so they must not make the
                            connector vehicle stop at every new node.
                        */
                        float handoff_s = connector_exit_handoff_s(from_ln, to_ln, road);
                        float lane_ahead_s = ecs.s[j] - handoff_s;

                        if (lane_ahead_s >= 0.0f) {
                            ds = (clen - self_progress) + lane_ahead_s;
                        }
                    }

                    if (ds > 0.0f && ds < 1.0e8f) {
                        float gap =
                            ds
                            - 0.5f * ecs.length[self]
                            - 0.5f * ecs.length[j];

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


__device__ __forceinline__ float connector_cross_conflict_accel_limit_ecs(
    int self,
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    int max_entities,
    float dt,
    float* metrics
) {
    int from_ln = ecs.connector_from_lane[self];
    int to_ln = ecs.connector_to_lane[self];
    if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road)) return 1000.0f;

    int node = road.lane_end_node[from_ln];
    float self_len = fmaxf(ecs.connector_length[self], CONNECTOR_MIN_LEN);
    float self_s = ecs.connector_s[self];
    float self_v = fmaxf(ecs.speed[self], 0.15f);

    // EN: Once the car is committed inside the connector, it must clear the box;
    //     stopping in the middle creates the deadlock reported in dense networks.
    //     Cross conflicts should be handled before entry, not after both cars are
    //     already in the intersection.
    // KO: 차량이 connector 안으로 충분히 들어온 뒤에는 박스를 비워야 합니다.
    //     중간 정지는 고밀도 네트워크에서 데드락을 만듭니다. 교차 충돌은 원칙적으로
    //     진입 전에 막고, 이미 들어온 차량은 빠져나가게 합니다.
    if (self_s > fmaxf(1.5f, self_len * CONNECTOR_PROTECTED_PROGRESS_FRAC)
        || (self_len - self_s) < self_len * CONNECTOR_PROTECTED_EXIT_FRAC) {
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

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
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
                                    if (ddx * ddx + ddy * ddy > CONNECTOR_CROSS_CONFLICT_RADIUS * CONNECTOR_CROSS_CONFLICT_RADIUS) continue;

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
        atomicAdd(&metrics[METRIC_CONNECTOR_CROSS_YIELD], 1.0f);
        atomicAdd(&metrics[METRIC_ANTI_COLLISION_BRAKE], 1.0f);
    }
    return req;
}

__global__ void connector_motion_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[i] != VEH_IN_CONNECTOR) return;

    int from_ln = ecs.connector_from_lane[i];
    int to_ln = ecs.connector_to_lane[i];

    if (
        from_ln < 0 || from_ln >= road.num_lanes
        || to_ln < 0 || to_ln >= road.num_lanes
    ) {
        ecs.alive[i] = ENTITY_FREE;
        return;
    }

    bool human = ecs.driver_type[i] == HUMAN;

    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float max_decel = human ? MAX_DECEL_HUMAN : MAX_DECEL_AV;

    float clen = fmaxf(ecs.connector_length[i], CONNECTOR_MIN_LEN);
    ecs.connector_length[i] = clen;

    float target_v = human ? CONNECTOR_SPEED_HUMAN : CONNECTOR_SPEED_AV;
    float turn_angle = turn_angle_deg(from_ln, to_ln, road);
    float signed_turn = lane_signed_turn_deg(from_ln, to_ln, road);
    if (turn_angle <= STRAIGHT_NO_TURN_SPEED_CAP_DEG || wide_lane_count_change_continuation_ecs(from_ln, to_ln, road)) {
        // EN: A straight connector is just a continuation through the node.
        //     Preserve free-flow speed when there is no front vehicle or signal reason.
        // KO: 직진 connector는 노드를 통과하는 연속 주행입니다. 앞차/신호 이유가 없으면
        //     교차로 통과 중 속도를 떨어뜨리지 않습니다.
        target_v = desired_speed_ecs(i, to_ln, ecs, road);
    } else {
        target_v = fminf(target_v, turn_speed_cap(turn_angle, ecs.driver_type[i]));
        target_v = fmaxf(target_v, human ? 2.4f : 3.0f);
    }
    if (signed_turn < -25.0f && metrics != nullptr) {
        atomicAdd(&metrics[METRIC_RIGHT_TURN_SYMMETRIC_PATH], 1.0f);
    }

    float v = ecs.speed[i];

    float front_gap, front_speed;
    find_front_in_connector_ecs(
        i,
        ecs,
        road,
        grid,
        max_entities,
        55.0f,
        front_gap,
        front_speed
    );

    float acc_cmd =
        max_accel * (1.0f - powf(v / fmaxf(target_v, 0.1f), 4.0f));

    // EN: Once a vehicle has entered the intersection, it owns a protected
    //     clearing movement.  Cross-traffic is stopped by the entry priority gate
    //     before it enters, so connector motion should not create an in-box
    //     mutual-yield deadlock.  We still keep same-path front following below.
    // KO: 차량이 교차로에 진입하면 보호된 통과 동작으로 봅니다. 교차 교통은 진입 전에
    //     우선순위 게이트에서 막으므로, connector 내부에서 서로 양보하며 멈추는
    //     데드락을 만들지 않습니다. 같은 경로 앞차 추종은 아래에서 계속 유지합니다.
    if (metrics != nullptr) atomicAdd(&metrics[METRIC_FORCE_PASS_THROUGH], 1.0f);

    /*
        connector 내부 car-following.
        기존에는 connector 진입 후 모든 차량이 target_v로만 이동해서
        교차로 안에서 서로 겹치는 문제가 발생했다.
    */
    if (front_gap < 1.0e8f) {
        float safe =
            (human ? SAFE_GAP_HUMAN : SAFE_GAP_AV)
            + v * (human ? SAFE_TIME_HEADWAY_HUMAN : SAFE_TIME_HEADWAY_AV);

        float follow_a = estimate_follow_accel_ecs(
            v,
            target_v,
            front_gap,
            front_speed,
            ecs.driver_type[i],
            ecs.min_gap[i],
            ecs.reaction_time[i],
            ecs.comfort_decel[i],
            ecs.aggressiveness[i],
            ecs.risk_tolerance[i]
        );

        acc_cmd = fminf(acc_cmd, follow_a);

        if (front_gap < safe) {
            float ratio = safe / fmaxf(front_gap, 0.5f);
            acc_cmd = fminf(acc_cmd, -max_decel * ratio * ratio);
        }

        if (front_gap < MIN_BUMPER_GAP) {
            acc_cmd = -EMERGENCY_DECEL;
        }
    }

    if (front_gap > fmaxf(CONNECTOR_EXIT_SPACE_MIN * 0.55f, ecs.length[i] + MIN_BUMPER_GAP + 4.0f)
        && v < SMART_STALL_SPEED
        && acc_cmd <= 0.03f) {
        float clear_v = human ? SMART_STALL_RELEASE_SPEED_HUMAN : SMART_STALL_RELEASE_SPEED_AV;
        float clear_a = (clear_v - v) / fmaxf(dt, 0.01f);
        clear_a = clampf_cuda(clear_a, 0.0f, max_accel * SMART_STALL_RELEASE_ACCEL_SCALE);
        acc_cmd = fmaxf(acc_cmd, clear_a);
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_FORCE_PASS_THROUGH], 1.0f);
    }

#if CONNECTOR_NAV_AVOIDANCE_ENABLED
    // EN: Nav-mesh style local avoidance for vehicles already inside the
    //     intersection connector.  The priority gate prevents new conflicts at
    //     entry, but this extra swept-path check handles vehicles that are
    //     already inside the box and would otherwise meet at the same point.
    // KO: 교차로 connector 내부 차량에 대한 네비메시식 국소 회피입니다. 진입
    //     우선순위 게이트가 새 충돌을 막지만, 이미 박스 안에 들어온 차량끼리 같은
    //     지점에서 만나는 경우를 swept-path 시간차로 감속 회피합니다.
    float cross_limit = connector_cross_conflict_accel_limit_ecs(
        i,
        ecs,
        road,
        grid,
        max_entities,
        dt,
        metrics
    );
    if (cross_limit < 999.0f) {
        acc_cmd = fminf(acc_cmd, cross_limit);
    }
#endif

    // EN/KO: Anti-stall inside connector.  If the physical front path is clear,
    // do not let a stale cross-yield command leave a car stopped in the middle.
    if (v < SMART_STALL_SPEED_EPS && front_gap > SMART_STALL_FRONT_GAP) {
        float clear_v = human ? CONNECTOR_INBOX_MIN_CLEAR_SPEED_HUMAN : CONNECTOR_INBOX_MIN_CLEAR_SPEED_AV;
        float clear_a = (clear_v - v) / fmaxf(dt, 0.01f);
        clear_a = clampf_cuda(clear_a, 0.0f, max_accel * SMART_STALL_CLEAR_ACCEL_SCALE);
        acc_cmd = fmaxf(acc_cmd, clear_a);
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
    }

    float target_acc_cmd = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);

    float prev_accel_for_delay = ecs.accel[i];
    if (
        prev_accel_for_delay < -1.5f
        && target_acc_cmd > 0.05f
        && v < SMART_STALL_SPEED_EPS
        && front_gap > fmaxf(STALE_BRAKE_CLEAR_FRONT_GAP, ecs.length[i] + MIN_BUMPER_GAP + 6.0f)
    ) {
        // EN/KO: Clear stale contact-brake memory once the connector path ahead is open.
        prev_accel_for_delay = 0.0f;
        float clear_a = (human ? STALE_BRAKE_CLEAR_ACCEL_HUMAN : STALE_BRAKE_CLEAR_ACCEL_AV) * max_accel;
        target_acc_cmd = fmaxf(target_acc_cmd, clear_a);
        if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
    }

    acc_cmd = apply_reaction_delay_accel(
        prev_accel_for_delay,
        target_acc_cmd,
        ecs.reaction_time[i],
        ecs.driver_type[i],
        dt,
        metrics
    );
    acc_cmd = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);

    if (acc_cmd < -2.0f) {
        atomicAdd(&metrics[METRIC_COMFORT_BRAKE], 1.0f);
    }

    float v_next = fmaxf(0.0f, v + acc_cmd * dt);
    float cs_next = ecs.connector_s[i] + 0.5f * (v + v_next) * dt;

    /*
        앞차가 너무 가까우면 이번 step에서 앞차를 뚫고 지나가지 못하게 제한.
    */
    if (front_gap < 1.0e8f) {
        float max_advance = fmaxf(0.0f, front_gap - MIN_BUMPER_GAP);
        float attempted = cs_next - ecs.connector_s[i];

        if (attempted > max_advance) {
            cs_next = ecs.connector_s[i] + max_advance;
            v_next = fminf(v_next, max_advance / fmaxf(dt, 0.001f));
            acc_cmd = (v_next - v) / fmaxf(dt, 0.001f);
            acc_cmd = clampf_cuda(acc_cmd, -EMERGENCY_DECEL, max_accel);
        }
    }

    atomicAdd(&metrics[METRIC_CONNECTOR_RUN], 1.0f);
    if (v_next < target_v * 0.65f) {
        atomicAdd(&metrics[METRIC_CONNECTOR_DELAY_SUM], dt);
        atomicAdd(&metrics[METRIC_CONNECTOR_DELAY_COUNT], 1.0f);
    }

    int rid = ecs.route_id[i];
    int rpos = ecs.route_pos[i];
    if (rid < 0 || rid >= road.num_routes) {
        ecs.alive[i] = ENTITY_FREE;
        return;
    }

    int repaired_pos = repair_route_pos_for_current_lane_ecs(from_ln, rid, rpos, road);
    if (repaired_pos >= 0) {
        rpos = repaired_pos;
        ecs.route_pos[i] = repaired_pos;
    }

    int ro0 = road.route_offsets[rid];
    int ro1 = road.route_offsets[rid + 1];
    int route_len = ro1 - ro0;
    if (route_len <= 0 || rpos < 0 || rpos >= route_len) {
        ecs.alive[i] = ENTITY_FREE;
        return;
    }

    int next_pos = rpos + 1;

    if (next_pos >= route_len) {
        ecs.alive[i] = ENTITY_FREE;
        atomicAdd(&metrics[METRIC_EXITED], 1.0f);
        atomicAdd(&metrics[METRIC_TRAVEL_TIME], fmaxf(0.0f, current_time - ecs.entry_time[i]));
        return;
    }

    if (cs_next >= clen - CONNECTOR_EXIT_EPS) {
        float overflow = cs_next - clen;

        /*
            출구 직후 next lane 초입에 차량이 너무 가까우면
            connector 안에서 대기시킨다.
        */
        if (front_gap < MIN_BUMPER_GAP + 1.0f) {
            cs_next = fmaxf(0.0f, clen - CONNECTOR_EXIT_EPS - 0.05f);
            v_next = 0.0f;
            acc_cmd = -EMERGENCY_DECEL;
            if (signed_turn < -25.0f && metrics != nullptr) {
                atomicAdd(&metrics[METRIC_RIGHT_TURN_EXIT_GAP_HOLD], 1.0f);
            }
        } else {
            ecs.vehicle_state[i] = VEH_ON_LANE;
            ecs.connector_from_lane[i] = -1;
            ecs.connector_to_lane[i] = -1;
            ecs.connector_s[i] = 0.0f;
            ecs.connector_length[i] = 0.0f;

            ecs.lane_id[i] = to_ln;
            ecs.route_pos[i] = next_pos;
            int repaired_after_connector = repair_route_pos_for_current_lane_ecs(to_ln, rid, ecs.route_pos[i], road);
            if (repaired_after_connector >= 0) {
                ecs.route_pos[i] = repaired_after_connector;
            }

            float handoff_s = connector_exit_handoff_s(from_ln, to_ln, road);
            float lane_s = handoff_s + fmaxf(0.0f, overflow);

            ecs.s[i] = clampf_cuda(
                lane_s,
                0.0f,
                fmaxf(0.0f, road.lane_length[to_ln] - 0.05f)
            );

            ecs.speed[i] = fmaxf(v_next, human ? 1.2f : 1.8f);
            ecs.accel[i] = acc_cmd;

            float px, py, ph;
            lane_xy_heading_from_s(to_ln, ecs.s[i], road, px, py, ph);

            ecs.x[i] = px;
            ecs.y[i] = py;

            float steer = 0.0f;
            float yaw_rate = 0.0f;
            float new_h = advance_heading_bicycle_ecs(
                i,
                ecs,
                ph,
                ecs.speed[i],
                dt,
                steer,
                yaw_rate
            );
            new_h = enforce_path_heading_error_limit_ecs(
                i,
                ecs,
                new_h,
                ph,
                ecs.speed[i],
                dt,
                human ? CONNECTOR_HEADING_LOCK_HUMAN : CONNECTOR_HEADING_LOCK_AV,
                steer,
                yaw_rate
            );

            ecs.steer_angle[i] = steer;
            ecs.heading[i] = new_h;

            atomicAdd(&metrics[METRIC_STEER_ABS_SUM], fabsf(steer));
            atomicAdd(&metrics[METRIC_STEER_COUNT], 1.0f);
            atomicAdd(&metrics[METRIC_YAW_RATE_ABS_SUM], fabsf(yaw_rate));
            atomicAdd(&metrics[METRIC_YAW_RATE_COUNT], 1.0f);

            return;
        }
    }

    ecs.connector_s[i] = clampf_cuda(cs_next, 0.0f, clen);

    ecs.speed[i] = v_next;
    ecs.accel[i] = acc_cmd;

    ecs.lane_id[i] = from_ln;
    {
        float entry_backoff = connector_entry_backoff_ecs(from_ln, to_ln, road);
        ecs.s[i] = fmaxf(0.0f, road.lane_length[from_ln] - entry_backoff);
    }

    float px, py, ph;
    // EN: Position advances along the connector path first; heading follows through the bicycle model.
    // KO: 먼저 차량 위치가 회전 경로를 따라 전진하고, 차체 방향은 bicycle model로 따라갑니다.
    //     따라서 정지 상태에서 제자리로 빙글 도는 회전은 발생하지 않습니다.
    connector_xy_heading_from_s(
        from_ln,
        to_ln,
        ecs.connector_s[i],
        clen,
        road,
        px,
        py,
        ph
    );

    ecs.x[i] = px;
    ecs.y[i] = py;

    float steer = 0.0f;
    float yaw_rate = 0.0f;
    float new_h = advance_heading_bicycle_ecs(
        i,
        ecs,
        ph,
        v_next,
        dt,
        steer,
        yaw_rate
    );
    new_h = enforce_path_heading_error_limit_ecs(
        i,
        ecs,
        new_h,
        ph,
        v_next,
        dt,
        human ? CONNECTOR_HEADING_LOCK_HUMAN : CONNECTOR_HEADING_LOCK_AV,
        steer,
        yaw_rate
    );

    ecs.steer_angle[i] = steer;
    ecs.heading[i] = new_h;

    atomicAdd(&metrics[METRIC_STEER_ABS_SUM], fabsf(steer));
    atomicAdd(&metrics[METRIC_STEER_COUNT], 1.0f);
    atomicAdd(&metrics[METRIC_YAW_RATE_ABS_SUM], fabsf(yaw_rate));
    atomicAdd(&metrics[METRIC_YAW_RATE_COUNT], 1.0f);
}


// ============================================================
// Collision / Safety System
// ============================================================

__device__ __forceinline__ void vehicle_obb_axes(
    float h,
    float& fx,
    float& fy,
    float& sx,
    float& sy
) {
    fx = cosf(h);
    fy = sinf(h);
    sx = -fy;
    sy = fx;
}

__device__ __forceinline__ bool obb_overlap_sat(
    float ax,
    float ay,
    float ah,
    float al,
    float aw,
    float bx,
    float by,
    float bh,
    float bl,
    float bw,
    float inflate
) {
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

    float axes_x[4] = { afx, asx, bfx, bsx };
    float axes_y[4] = { afy, asy, bfy, bsy };

    for (int k = 0; k < 4; ++k) {
        float ux = axes_x[k];
        float uy = axes_y[k];

        float dist = fabsf(dx * ux + dy * uy);

        float ra =
            0.5f * al * fabsf(ux * afx + uy * afy)
            + 0.5f * aw * fabsf(ux * asx + uy * asy);

        float rb =
            0.5f * bl * fabsf(ux * bfx + uy * bfy)
            + 0.5f * bw * fabsf(ux * bsx + uy * bsy);

        if (dist > ra + rb) return false;
    }

    return true;
}

__device__ __forceinline__ bool swept_overlap_ecs(
    float ax,
    float ay,
    float ah,
    float av,
    float al,
    float aw,
    float bx,
    float by,
    float bh,
    float bv,
    float bl,
    float bw,
    float dt,
    float horizon,
    float inflate
) {
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



__device__ __forceinline__ void sync_vehicle_to_path_ecs(
    int id,
    ECSArrays ecs,
    RoadNetwork road
) {
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
            float px, py, ph;
            lane_xy_heading_from_s(ln, ecs.s[id], road, px, py, ph);
            ecs.x[id] = px;
            ecs.y[id] = py;
            ecs.heading[id] = ph;
        }
    }
}


__global__ void route_lane_repair_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    float* metrics,
    float dt,
    int max_entities
) {
#if ROUTE_LANE_RUNTIME_REPAIR_ENABLED
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int state = ecs.vehicle_state[i];
    if (state == VEH_ON_LANE) {
        int lane = ecs.lane_id[i];
        if (!valid_lane_ecs(lane, road)) {
            ecs.alive[i] = ENTITY_FREE;
            return;
        }

        float L = fmaxf(road.lane_length[lane], 0.1f);
        ecs.s[i] = clampf_cuda(ecs.s[i], 0.0f, fmaxf(0.0f, L - 0.05f));

        int rid = ecs.route_id[i];
        int rpos = ecs.route_pos[i];
        if (rid < 0 || rid >= road.num_routes) {
            // Invalid route ids are unrecoverable; remove the bad actor instead of
            // leaving a permanently frozen vehicle in the road.
            ecs.alive[i] = ENTITY_FREE;
            return;
        }

        int repaired = repair_route_pos_for_current_lane_ecs(lane, rid, rpos, road);
        if (repaired >= 0 && repaired != rpos) {
            ecs.route_pos[i] = repaired;
            if (ecs.speed[i] < ROUTE_LANE_REPAIR_SPEED_CAP) {
                ecs.speed[i] = fmaxf(ecs.speed[i], 0.6f);
            }
            if (ecs.accel[i] < 0.0f) ecs.accel[i] = 0.0f;
            if (metrics != nullptr) atomicAdd(&metrics[METRIC_DEADLOCK_CREEP], 1.0f);
        }

        if (ecs.lane_change_active[i] != 0) {
            int from_ln = ecs.lane_change_from_lane[i];
            int to_ln = ecs.lane_change_to_lane[i];
            if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road) || !same_approach_same_direction_lanes_ecs(from_ln, to_ln, road)) {
                ecs.lane_change_active[i] = 0;
                ecs.lane_change_from_lane[i] = lane;
                ecs.lane_change_to_lane[i] = lane;
                ecs.lane_change_t[i] = 0.0f;
            }
        }

        sync_vehicle_to_path_ecs(i, ecs, road);
        return;
    }

    if (state == VEH_IN_CONNECTOR) {
        int from_ln = ecs.connector_from_lane[i];
        int to_ln = ecs.connector_to_lane[i];
        if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road) || !lane_connected(from_ln, to_ln, road)) {
            ecs.alive[i] = ENTITY_FREE;
            return;
        }
        float clen = fmaxf(ecs.connector_length[i], CONNECTOR_MIN_LEN);
        ecs.connector_length[i] = clen;
        ecs.connector_s[i] = clampf_cuda(ecs.connector_s[i], 0.0f, clen);
        int rid = ecs.route_id[i];
        if (rid < 0 || rid >= road.num_routes) {
            ecs.alive[i] = ENTITY_FREE;
            return;
        }
        sync_vehicle_to_path_ecs(i, ecs, road);
    }
#else
    (void)ecs; (void)road; (void)metrics; (void)dt; (void)max_entities;
#endif
}

__device__ __forceinline__ void move_vehicle_back_on_path_ecs(
    int id,
    float back,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ bool repair_pair_same_lane_ecs(
    int a,
    int b,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (ecs.vehicle_state[a] != VEH_ON_LANE || ecs.vehicle_state[b] != VEH_ON_LANE) return false;
    int la = ecs.lane_id[a];
    int lb = ecs.lane_id[b];
    if (!valid_lane_ecs(la, road) || la != lb) return false;

    int front = ecs.s[a] >= ecs.s[b] ? a : b;
    int rear = front == a ? b : a;
    float desired_center_ds = 0.5f * ecs.length[front] + 0.5f * ecs.length[rear]
        + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_rear_s = ecs.s[front] - desired_center_ds;
    if (ecs.s[rear] > target_rear_s) {
        ecs.s[rear] = fmaxf(0.0f, target_rear_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(rear, ecs, road);
        ecs.speed[rear] = fminf(ecs.speed[rear], fmaxf(0.0f, ecs.speed[front] - 0.3f));
        ecs.accel[rear] = CONTACT_REPAIR_HOLD_ACCEL;
    }
    return true;
}

__device__ __forceinline__ bool repair_pair_same_connector_ecs(
    int a,
    int b,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (ecs.vehicle_state[a] != VEH_IN_CONNECTOR || ecs.vehicle_state[b] != VEH_IN_CONNECTOR) return false;
    if (ecs.connector_from_lane[a] != ecs.connector_from_lane[b] || ecs.connector_to_lane[a] != ecs.connector_to_lane[b]) return false;

    int front = ecs.connector_s[a] >= ecs.connector_s[b] ? a : b;
    int rear = front == a ? b : a;
    float desired_center_ds = 0.5f * ecs.length[front] + 0.5f * ecs.length[rear]
        + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_rear_s = ecs.connector_s[front] - desired_center_ds;
    if (ecs.connector_s[rear] > target_rear_s) {
        ecs.connector_s[rear] = fmaxf(0.0f, target_rear_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(rear, ecs, road);
        ecs.speed[rear] = fminf(ecs.speed[rear], fmaxf(0.0f, ecs.speed[front] - 0.3f));
        ecs.accel[rear] = CONTACT_REPAIR_HOLD_ACCEL;
    }
    return true;
}

__device__ __forceinline__ bool repair_connector_to_lane_pair_ecs(
    int conn,
    int lane_vehicle,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (ecs.vehicle_state[conn] != VEH_IN_CONNECTOR || ecs.vehicle_state[lane_vehicle] != VEH_ON_LANE) return false;
    int from_ln = ecs.connector_from_lane[conn];
    int to_ln = ecs.connector_to_lane[conn];
    int lane = ecs.lane_id[lane_vehicle];
    if (!valid_lane_ecs(from_ln, road) || !valid_lane_ecs(to_ln, road) || lane != to_ln) return false;

    float handoff = connector_exit_handoff_s(from_ln, to_ln, road);
    if (ecs.s[lane_vehicle] < handoff) return false;

    float clen = fmaxf(ecs.connector_length[conn], CONNECTOR_MIN_LEN);
    float desired_center_ds = 0.5f * ecs.length[conn] + 0.5f * ecs.length[lane_vehicle]
        + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_conn_s = clen + (ecs.s[lane_vehicle] - handoff) - desired_center_ds;
    if (ecs.connector_s[conn] > target_conn_s) {
        ecs.connector_s[conn] = clampf_cuda(target_conn_s - CONTACT_ROUTE_REPAIR_EXTRA, 0.0f, clen);
        sync_vehicle_to_path_ecs(conn, ecs, road);
        ecs.speed[conn] = fminf(ecs.speed[conn], fmaxf(0.0f, ecs.speed[lane_vehicle] - 0.3f));
        ecs.accel[conn] = CONTACT_REPAIR_HOLD_ACCEL;
    }
    return true;
}

__device__ __forceinline__ bool repair_lane_to_connector_pair_ecs(
    int lane_vehicle,
    int conn,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (ecs.vehicle_state[lane_vehicle] != VEH_ON_LANE || ecs.vehicle_state[conn] != VEH_IN_CONNECTOR) return false;
    int lane = ecs.lane_id[lane_vehicle];
    int from_ln = ecs.connector_from_lane[conn];
    int to_ln = ecs.connector_to_lane[conn];
    if (!valid_lane_ecs(lane, road) || lane != from_ln || !valid_lane_ecs(to_ln, road)) return false;

    float start_s = fmaxf(0.0f, road.lane_length[from_ln] - connector_entry_backoff_ecs(from_ln, to_ln, road));
    float desired_center_ds = 0.5f * ecs.length[lane_vehicle] + 0.5f * ecs.length[conn]
        + CONTACT_LONGITUDINAL_REPAIR_GAP;
    float target_lane_s = start_s + ecs.connector_s[conn] - desired_center_ds;
    if (ecs.s[lane_vehicle] > target_lane_s) {
        ecs.s[lane_vehicle] = fmaxf(0.0f, target_lane_s - CONTACT_ROUTE_REPAIR_EXTRA);
        sync_vehicle_to_path_ecs(lane_vehicle, ecs, road);
        ecs.speed[lane_vehicle] = fminf(ecs.speed[lane_vehicle], fmaxf(0.0f, ecs.speed[conn] - 0.3f));
        ecs.accel[lane_vehicle] = CONTACT_REPAIR_HOLD_ACCEL;
    }
    return true;
}

__device__ __forceinline__ bool repair_route_overlap_ecs(
    int a,
    int b,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (repair_pair_same_lane_ecs(a, b, ecs, road)) return true;
    if (repair_pair_same_connector_ecs(a, b, ecs, road)) return true;
    if (repair_connector_to_lane_pair_ecs(a, b, ecs, road)) return true;
    if (repair_connector_to_lane_pair_ecs(b, a, ecs, road)) return true;
    if (repair_lane_to_connector_pair_ecs(a, b, ecs, road)) return true;
    if (repair_lane_to_connector_pair_ecs(b, a, ecs, road)) return true;
    return false;
}

__device__ __forceinline__ int contact_priority_score_ecs(
    int id,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ void move_vehicle_forward_on_path_ecs(
    int id,
    float forward,
    ECSArrays ecs,
    RoadNetwork road
) {
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

__device__ __forceinline__ void overlap_release_boost_vehicle_ecs(
    int id,
    ECSArrays ecs,
    RoadNetwork road,
    float dt
) {
    if (id < 0 || ecs.alive[id] != ENTITY_ALIVE) return;
    bool human = ecs.driver_type[id] == HUMAN;
    float release_v = human ? COMPLETE_OVERLAP_RELEASE_SPEED_HUMAN : COMPLETE_OVERLAP_RELEASE_SPEED_AV;
    if (ecs.vehicle_state[id] == VEH_ON_LANE && valid_lane_ecs(ecs.lane_id[id], road)) {
        release_v = fminf(release_v, fmaxf(1.0f, desired_speed_ecs(id, ecs.lane_id[id], ecs, road)));
    }
    float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
    float release_a = (release_v - ecs.speed[id]) / fmaxf(dt, 0.01f);
    release_a = clampf_cuda(release_a, 0.0f, max_accel * COMPLETE_OVERLAP_RELEASE_ACCEL_SCALE);

    // EN: Give the randomly selected winner both velocity and a small path
    //     nudge.  Without the nudge, two perfectly overlapping stopped cars can
    //     remain visually coincident until the next full simulation tick.
    // KO: 무작위로 선택된 우선 차량에는 속도뿐 아니라 경로 방향의 작은 전진 보정도
    //     줍니다. 완전히 겹쳐 정지한 두 차량이 다음 tick까지 계속 같은 좌표에 남는
    //     현상을 줄입니다.
    move_vehicle_forward_on_path_ecs(id, COMPLETE_OVERLAP_CONTACT_FORWARD_NUDGE, ecs, road);
    ecs.speed[id] = fmaxf(ecs.speed[id], fminf(release_v, ecs.speed[id] + release_a * fmaxf(dt, 0.01f)));
    ecs.accel[id] = fmaxf(ecs.accel[id], release_a);
}

__global__ void contact_resolve_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int base = world_cell_index(
        ecs.x[i],
        ecs.y[i],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );
    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j > i && ecs.alive[j] == ENTITY_ALIVE) {
                    float ddx = ecs.x[j] - ecs.x[i];
                    float ddy = ecs.y[j] - ecs.y[i];
                    float broad_r =
                        0.5f * sqrtf(ecs.length[i] * ecs.length[i] + ecs.width[i] * ecs.width[i])
                        + 0.5f * sqrtf(ecs.length[j] * ecs.length[j] + ecs.width[j] * ecs.width[j])
                        + CONTACT_RESOLVE_MAX_PUSH + 1.5f;

                    if (ddx * ddx + ddy * ddy <= broad_r * broad_r) {
                        bool hit = obb_overlap_sat(
                            ecs.x[i], ecs.y[i], ecs.heading[i], ecs.length[i], ecs.width[i],
                            ecs.x[j], ecs.y[j], ecs.heading[j], ecs.length[j], ecs.width[j],
                            CONTACT_RESOLVE_INFLATE
                        );

                        if (hit) {
                            bool complete_overlap = false;
#if COMPLETE_OVERLAP_RELEASE_ENABLED
                            complete_overlap =
                                ddx * ddx + ddy * ddy <= COMPLETE_OVERLAP_RELEASE_DIST * COMPLETE_OVERLAP_RELEASE_DIST
                                && ecs.speed[i] <= COMPLETE_OVERLAP_RELEASE_MAX_SPEED
                                && ecs.speed[j] <= COMPLETE_OVERLAP_RELEASE_MAX_SPEED
                                && obb_overlap_sat(
                                    ecs.x[i], ecs.y[i], ecs.heading[i], ecs.length[i], ecs.width[i],
                                    ecs.x[j], ecs.y[j], ecs.heading[j], ecs.length[j], ecs.width[j],
                                    0.0f
                                );
#endif

                            bool route_repaired = false;
                            if (complete_overlap) {
                                bool i_wins = timed_pair_random_self_wins_ecs(
                                    i,
                                    j,
                                    current_time,
                                    COMPLETE_OVERLAP_RELEASE_PERIOD
                                );
                                int winner = i_wins ? i : j;
                                int loser = i_wins ? j : i;
                                float back = CONTACT_CROSS_BACKOFF_MIN + COMPLETE_OVERLAP_CONTACT_BACKOFF_EXTRA + fmaxf(ecs.speed[loser], 0.0f) * CONTACT_CROSS_BACKOFF_SPEED_TIME;
                                back = fminf(CONTACT_RESOLVE_MAX_PUSH, fmaxf(back, CONTACT_RESOLVE_BACKOFF));
                                move_vehicle_back_on_path_ecs(loser, back, ecs, road);
                                overlap_release_boost_vehicle_ecs(winner, ecs, road, dt);
                                route_repaired = true;
                                if (metrics != nullptr) {
                                    atomicAdd(&metrics[METRIC_DEADLOCK_RELEASE], 1.0f);
                                    atomicAdd(&metrics[METRIC_DEADLOCK_ESCAPE_GO], 1.0f);
                                }
                            } else {
                                route_repaired = repair_route_overlap_ecs(i, j, ecs, road);
                            }

                            if (!route_repaired) {
                                int si = contact_priority_score_ecs(i, ecs, road);
                                int sj = contact_priority_score_ecs(j, ecs, road);
                                int loser = j;
                                if (si < sj) loser = i;
                                else if (sj < si) loser = j;
                                else {
                                    // EN/KO: Newer/higher id yields on exact ties.
                                    loser = (ecs.entry_time[i] > ecs.entry_time[j] + 0.001f) ? i : j;
                                    if (fabsf(ecs.entry_time[i] - ecs.entry_time[j]) <= 0.001f) loser = i > j ? i : j;
                                }

                                float back = CONTACT_CROSS_BACKOFF_MIN + fmaxf(ecs.speed[loser], 0.0f) * CONTACT_CROSS_BACKOFF_SPEED_TIME;
                                back = fminf(CONTACT_RESOLVE_MAX_PUSH, fmaxf(back, CONTACT_RESOLVE_BACKOFF));
                                move_vehicle_back_on_path_ecs(loser, back, ecs, road);
                            }

                            if (metrics != nullptr) {
                                atomicAdd(&metrics[METRIC_PENETRATION_PREVENTED], 1.0f);
                                atomicAdd(&metrics[METRIC_ANTI_COLLISION_BRAKE], 1.0f);
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


__global__ void collision_system_kernel(
    ECSArrays ecs,
    SpatialGrid grid,
    float* metrics,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int base = world_cell_index(
        ecs.x[i],
        ecs.y[i],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j > i && ecs.alive[j] == ENTITY_ALIVE) {
                    float ddx = ecs.x[j] - ecs.x[i];
                    float ddy = ecs.y[j] - ecs.y[i];

                    float broad_r =
                        0.5f * sqrtf(ecs.length[i] * ecs.length[i] + ecs.width[i] * ecs.width[i])
                        + 0.5f * sqrtf(ecs.length[j] * ecs.length[j] + ecs.width[j] * ecs.width[j])
                        + fmaxf(ecs.speed[i], ecs.speed[j]) * dt
                        + 3.0f;

                    if (ddx * ddx + ddy * ddy <= broad_r * broad_r) {
                        bool hit = swept_overlap_ecs(
                            ecs.x[i],
                            ecs.y[i],
                            ecs.heading[i],
                            ecs.speed[i],
                            ecs.length[i],
                            ecs.width[i],
                            ecs.x[j],
                            ecs.y[j],
                            ecs.heading[j],
                            ecs.speed[j],
                            ecs.length[j],
                            ecs.width[j],
                            dt,
                            0.35f,
                            0.20f
                        );

                        if (hit) {
                            atomicAdd(&metrics[METRIC_COLLISION], 1.0f);
                        }
                    }
                }

                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}


// ============================================================
// Nav-mesh style local obstacle avoidance before motion
// ============================================================

__device__ __forceinline__ bool local_avoid_same_stream_skip_ecs(
    int a,
    int b,
    ECSArrays ecs,
    RoadNetwork road
) {
    if (ecs.vehicle_state[a] != VEH_ON_LANE || ecs.vehicle_state[b] != VEH_ON_LANE) return false;
    if (ecs.lane_change_active[a] != 0 || ecs.lane_change_active[b] != 0) return false;

    int la = ecs.lane_id[a];
    int lb = ecs.lane_id[b];
    if (!valid_lane_ecs(la, road) || !valid_lane_ecs(lb, road)) return false;
    if (la != lb) return false;

    float dh = fabsf(wrap_pi(ecs.heading[a] - ecs.heading[b]));
    return dh < 0.36f;
}

__device__ __forceinline__ int local_avoid_priority_score_ecs(
    int id,
    ECSArrays ecs,
    RoadNetwork road,
    PerceptionSoA perception
) {
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

__device__ __forceinline__ bool local_avoid_pair_relevant_ecs(
    int a,
    int b,
    ECSArrays ecs,
    RoadNetwork road
) {
    /*
        EN: Filter the nav-mesh avoidance to pairs that can actually interact on
            the road surface.  Without this, nearby but unrelated links (parallel
            lanes, map crossings, overpasses, or cars already separated by a node)
            could make vehicles stop in the middle of an empty road.
        KO: 네비메시식 회피는 실제 도로면에서 상호작용 가능한 쌍에만 적용합니다.
            그렇지 않으면 평행 차로, 지도상 교차하지만 연결되지 않는 도로, 이미 node로
            분리된 차량 때문에 빈 도로 한가운데서 멈추는 현상이 생깁니다.
    */
    if (obb_overlap_sat(
            ecs.x[a], ecs.y[a], ecs.heading[a], ecs.length[a], ecs.width[a],
            ecs.x[b], ecs.y[b], ecs.heading[b], ecs.length[b], ecs.width[b],
            LOCAL_AVOID_IMMEDIATE_OVERLAP_INFLATE)) {
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
            if (!valid_lane_ecs(af, road) || !valid_lane_ecs(at, road) || !valid_lane_ecs(bf, road) || !valid_lane_ecs(bt, road)) return false;
            if (af == bf && at == bt) return true;
            return road.lane_end_node[af] == road.lane_end_node[bf]
                && connector_swept_paths_overlap_ecs(af, at, bf, bt, road);
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
                return intersection_conflict_relevant_vehicles_ecs(
                    c, cf, ct, o, ol, on, false, ecs, road
                );
            }
            return intersection_conflict_relevant_vehicles_ecs(
                o, ol, on, c, cf, ct, true, ecs, road
            );
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
        if (lb == af || lb == at || lanes_share_link_geometry_ecs(lb, at, road) || lanes_share_link_geometry_ecs(lb, af, road)) return true;
    }
    if (ecs.lane_change_active[b] != 0) {
        int bf = ecs.lane_change_from_lane[b];
        int bt = ecs.lane_change_to_lane[b];
        if (la == bf || la == bt || lanes_share_link_geometry_ecs(la, bt, road) || lanes_share_link_geometry_ecs(la, bf, road)) return true;
    }

    if (lanes_share_link_geometry_ecs(la, lb, road)) {
        // Same directed link but different lanes: only lane-change/indicator courtesy handles them.
        return indicator_targets_lane_ecs(a, lb, ecs, road) || indicator_targets_lane_ecs(b, la, ecs, road);
    }

    int na = route_next_lane_for_vehicle_ecs(a, ecs, road);
    int nb = route_next_lane_for_vehicle_ecs(b, ecs, road);
    if (valid_lane_ecs(na, road) && valid_lane_ecs(nb, road) && road.lane_end_node[la] == road.lane_end_node[lb]) {
        float da = fmaxf(0.0f, road.lane_length[la] - ecs.s[a]);
        float db = fmaxf(0.0f, road.lane_length[lb] - ecs.s[b]);
        if (da <= PRIORITY_GATE_PATH_SCAN_RANGE && db <= PRIORITY_GATE_PATH_SCAN_RANGE) {
            return intersection_conflict_relevant_vehicles_ecs(a, la, na, b, lb, nb, false, ecs, road);
        }
    }

    return false;
}

__global__ void local_obstacle_avoidance_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    SpatialGrid grid,
    PerceptionSoA perception,
    DecisionSoA decision,
    float* metrics,
    float current_time,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;
    if (ecs.vehicle_state[i] != VEH_ON_LANE) return;

    int base = world_cell_index(ecs.x[i], ecs.y[i], grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height);
    if (base < 0) return;

    int lane = ecs.lane_id[i];
    int next_lane = route_next_lane_for_vehicle_ecs(i, ecs, road);
    bool self_inside_box = false;
    if (valid_lane_ecs(lane, road) && valid_lane_ecs(next_lane, road)) {
        float dist_to_end = fmaxf(0.0f, road.lane_length[lane] - ecs.s[i]);
        self_inside_box = inside_intersection_box_ecs(dist_to_end, lane, next_lane, road);
    }

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;
    int cr = clampi_cuda((int)ceilf(LOCAL_AVOID_RANGE / fmaxf(grid.cell_size, 0.1f)), 1, WORLD_MAX_CELL_RADIUS);

    int self_score = local_avoid_priority_score_ecs(i, ecs, road, perception);
    bool self_front_clear = priority_front_clear_ecs(i, perception, ecs);

#if LOCAL_AVOID_GRANTED_CONNECTOR_GRACE
    // EN: If the priority gate has already granted connector entry and the
    // forward route is open, do not let the generic local-avoidance layer cancel
    // the connector and create a fake mid-node stop.  Last-resort contact repair
    // still runs after motion.
    // KO: 우선순위 게이트가 connector 진입을 허용했고 앞 경로가 열려 있으면,
    // 일반 국소 회피 레이어가 다시 connector를 취소해 가짜 중간 정지를 만들지
    // 않도록 합니다. 최후 접촉 보정은 motion 뒤에 여전히 실행됩니다.
    if (self_front_clear && (decision.wants_connector[i] != 0 || self_inside_box)) {
        return;
    }
#endif

    float best_req = 1000.0f;
    bool blocked = false;
    bool overlap_release_go = false;

    float self_v = fmaxf(0.0f, ecs.speed[i] + fmaxf(0.0f, decision.target_accel[i]) * dt * 0.35f);

    for (int dy = -cr; dy <= cr; ++dy) {
        for (int dx = -cr; dx <= cr; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;
            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;
            while (j >= 0 && guard < max_entities) {
                if (j != i && ecs.alive[j] == ENTITY_ALIVE) {
                    if (local_avoid_same_stream_skip_ecs(i, j, ecs, road)) {
                        j = grid.cell_next[j];
                        guard++;
                        continue;
                    }

                    if (!local_avoid_pair_relevant_ecs(i, j, ecs, road)) {
                        j = grid.cell_next[j];
                        guard++;
                        continue;
                    }

                    float ddx = ecs.x[j] - ecs.x[i];
                    float ddy = ecs.y[j] - ecs.y[i];
                    float d2 = ddx * ddx + ddy * ddy;
                    float broad_r = LOCAL_AVOID_RANGE
                        + 0.5f * sqrtf(ecs.length[i] * ecs.length[i] + ecs.width[i] * ecs.width[i])
                        + 0.5f * sqrtf(ecs.length[j] * ecs.length[j] + ecs.width[j] * ecs.width[j]);
                    if (d2 <= broad_r * broad_r) {
                        float horizon = fmaxf(LOCAL_AVOID_HORIZON, 0.40f + 0.045f * fmaxf(self_v, ecs.speed[j]));
                        bool hit = swept_overlap_ecs(
                            ecs.x[i],
                            ecs.y[i],
                            ecs.heading[i],
                            self_v,
                            ecs.length[i],
                            ecs.width[i],
                            ecs.x[j],
                            ecs.y[j],
                            ecs.heading[j],
                            fmaxf(0.0f, ecs.speed[j]),
                            ecs.length[j],
                            ecs.width[j],
                            dt,
                            horizon,
                            LOCAL_AVOID_COLLISION_MARGIN
                        );

                        if (hit) {
                            bool complete_overlap = false;
#if COMPLETE_OVERLAP_RELEASE_ENABLED
                            complete_overlap =
                                d2 <= COMPLETE_OVERLAP_RELEASE_DIST * COMPLETE_OVERLAP_RELEASE_DIST
                                && ecs.speed[i] <= COMPLETE_OVERLAP_RELEASE_MAX_SPEED
                                && ecs.speed[j] <= COMPLETE_OVERLAP_RELEASE_MAX_SPEED
                                && obb_overlap_sat(
                                    ecs.x[i], ecs.y[i], ecs.heading[i], ecs.length[i], ecs.width[i],
                                    ecs.x[j], ecs.y[j], ecs.heading[j], ecs.length[j], ecs.width[j],
                                    0.0f
                                );
#endif

                            bool other_front_clear = priority_front_clear_ecs(j, perception, ecs);
                            int other_score = local_avoid_priority_score_ecs(j, ecs, road, perception);

                            bool self_goes_first = false;
                            if (complete_overlap) {
                                self_goes_first = timed_pair_random_self_wins_ecs(
                                    i,
                                    j,
                                    current_time,
                                    COMPLETE_OVERLAP_RELEASE_PERIOD
                                );
                                if (self_goes_first) overlap_release_go = true;
                            } else if (self_front_clear && !other_front_clear) {
                                self_goes_first = true;
                            } else if (!self_front_clear && other_front_clear) {
                                self_goes_first = false;
                            } else if (self_inside_box && ecs.vehicle_state[j] != VEH_IN_CONNECTOR) {
                                self_goes_first = true;
                            } else if (ecs.vehicle_state[j] == VEH_IN_CONNECTOR && !self_inside_box) {
                                self_goes_first = false;
                            } else if (self_score != other_score) {
                                self_goes_first = self_score > other_score;
                            } else {
                                self_goes_first = i < j;
                            }

                            if (!self_goes_first) {
                                float center_dist = sqrtf(fmaxf(d2, 0.01f));
                                float stop_dist = center_dist
                                    - 0.5f * ecs.length[i]
                                    - 0.5f * ecs.length[j]
                                    - LOCAL_AVOID_STOP_BUFFER;
                                stop_dist = fmaxf(stop_dist, 0.55f);
                                float req = -(ecs.speed[i] * ecs.speed[i]) / fmaxf(2.0f * stop_dist, 0.5f);
                                req = clampf_cuda(req, -EMERGENCY_DECEL, -0.04f);
                                best_req = fminf(best_req, req);
                                blocked = true;
                            }
                        }
                    }
                }
                j = grid.cell_next[j];
                guard++;
            }
        }
    }

    if (overlap_release_go && !blocked) {
        bool human = ecs.driver_type[i] == HUMAN;
        float max_accel = human ? MAX_ACCEL_HUMAN : MAX_ACCEL_AV;
        float release_v = human ? COMPLETE_OVERLAP_RELEASE_SPEED_HUMAN : COMPLETE_OVERLAP_RELEASE_SPEED_AV;
        float release_a = (release_v - ecs.speed[i]) / fmaxf(dt, 0.01f);
        release_a = clampf_cuda(release_a, 0.0f, max_accel * COMPLETE_OVERLAP_RELEASE_ACCEL_SCALE);
        decision.target_accel[i] = fmaxf(decision.target_accel[i], release_a);
        if (metrics != nullptr) {
            atomicAdd(&metrics[METRIC_DEADLOCK_RELEASE], 1.0f);
            atomicAdd(&metrics[METRIC_DEADLOCK_ESCAPE_GO], 1.0f);
        }
    }

    if (blocked) {
        decision.target_accel[i] = fminf(decision.target_accel[i], best_req);
        if (!self_inside_box) {
            decision.wants_connector[i] = 0;
            decision.connector_target_lane[i] = -1;
        }
        if (metrics != nullptr) {
            atomicAdd(&metrics[METRIC_ANTI_COLLISION_BRAKE], 1.0f);
            atomicAdd(&metrics[METRIC_PRIORITY_PATH_BLOCK], 1.0f);
        }
    }
}

__global__ void safety_metrics_system_kernel(
    ECSArrays ecs,
    SpatialGrid grid,
    float* metrics,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    int base = world_cell_index(
        ecs.x[i],
        ecs.y[i],
        grid.min_x,
        grid.min_y,
        grid.cell_size,
        grid.width,
        grid.height
    );

    if (base < 0) return;

    int bc_x = base % grid.width;
    int bc_y = base / grid.width;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int cx = bc_x + dx;
            int cy = bc_y + dy;

            if (cx < 0 || cx >= grid.width || cy < 0 || cy >= grid.height) continue;

            int j = grid.cell_head[cy * grid.width + cx];
            int guard = 0;

            while (j >= 0 && guard < max_entities) {
                if (j > i && ecs.alive[j] == ENTITY_ALIVE) {
                    float rx = ecs.x[j] - ecs.x[i];
                    float ry = ecs.y[j] - ecs.y[i];

                    float dist = sqrtf(fmaxf(rx * rx + ry * ry, 0.001f));

                    float vix = cosf(ecs.heading[i]) * ecs.speed[i];
                    float viy = sinf(ecs.heading[i]) * ecs.speed[i];
                    float vjx = cosf(ecs.heading[j]) * ecs.speed[j];
                    float vjy = sinf(ecs.heading[j]) * ecs.speed[j];

                    float rvx = vjx - vix;
                    float rvy = vjy - viy;

                    float closing = -((rx * rvx + ry * rvy) / fmaxf(dist, 0.1f));

                    float combined =
                        0.5f * ecs.length[i]
                        + 0.5f * ecs.length[j]
                        + MIN_BUMPER_GAP;

                    float gap = dist - combined;

                    if (gap < 1.0f) {
                        atomicAdd(&metrics[METRIC_NEAR_MISS], 1.0f);
                    }

                    if (closing > 0.1f && gap > 0.0f) {
                        float ttc = gap / closing;

                        if (ttc < TTC_CRITICAL) {
                            atomicAdd(&metrics[METRIC_TTC_CRITICAL], 1.0f);
                        } else if (ttc < TTC_WARNING) {
                            atomicAdd(&metrics[METRIC_TTC_WARNING], 1.0f);
                        }
                    }
                }

                j = grid.cell_next[j];
                guard++;
            }
        }
    }
}

// ============================================================
// Stats System
// ============================================================

__global__ void stats_system_kernel(
    ECSArrays ecs,
    RoadNetwork road,
    PerceptionSoA perception,
    DecisionSoA decision,
    float* metrics,
    float dt,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities || ecs.alive[i] != ENTITY_ALIVE) return;

    float v = ecs.speed[i];
    float a = ecs.accel[i];

    atomicAdd(&metrics[METRIC_ACTIVE], 1.0f);
    atomicAdd(&metrics[METRIC_SPEED_SUM], v);
    atomicAdd(&metrics[METRIC_SPEED_COUNT], 1.0f);

    if (v < 2.0f) atomicAdd(&metrics[METRIC_SLOW_COUNT], 1.0f);
    if (v < 0.2f) {
        atomicAdd(&metrics[METRIC_STOP_COUNT], 1.0f);
        atomicAdd(&metrics[METRIC_STANDSTILL_TIME], dt);
    }

    if (a > 0.05f) {
        atomicAdd(&metrics[METRIC_ACCEL_COUNT], 1.0f);
        atomicAdd(&metrics[METRIC_ACCEL_SUM], a);
        atomicAdd(&metrics[METRIC_ACCEL_SQ_SUM], a * a);
    } else if (a < -0.05f) {
        float d = -a;
        atomicAdd(&metrics[METRIC_DECEL_COUNT], 1.0f);
        atomicAdd(&metrics[METRIC_DECEL_SUM], d);
        atomicAdd(&metrics[METRIC_DECEL_SQ_SUM], d * d);
    }

    if (a < -3.5f) {
        atomicAdd(&metrics[METRIC_HARD_BRAKE], 1.0f);
    }

    if (perception.front_gap[i] < 1.0e8f) {
        atomicAdd(&metrics[METRIC_MIN_GAP_SUM], perception.front_gap[i]);
        atomicAdd(&metrics[METRIC_MIN_GAP_COUNT], 1.0f);

        if (v > 0.5f) {
            atomicAdd(&metrics[METRIC_HEADWAY_SUM], perception.front_gap[i] / fmaxf(v, 0.5f));
            atomicAdd(&metrics[METRIC_HEADWAY_COUNT], 1.0f);
        }
    }

    float desired = decision.desired_speed[i];
    if (!isfinite(desired) || desired < 0.1f) {
        int lane = ecs.lane_id[i];
        if (lane >= 0 && lane < road.num_lanes) {
            desired = desired_speed_ecs(i, lane, ecs, road);
        } else {
            desired = MAX_SPEED_FALLBACK;
        }
    }

    if (desired > 0.5f) {
        float loss_ratio = clampf_cuda((desired - v) / desired, 0.0f, 1.0f);
        atomicAdd(&metrics[METRIC_TIME_LOSS_SUM], loss_ratio * dt);
        atomicAdd(&metrics[METRIC_TIME_LOSS_COUNT], 1.0f);

        if (loss_ratio > 0.65f && v < 1.0f) {
            atomicAdd(&metrics[METRIC_QUEUE_DELAY_SUM], dt);
            atomicAdd(&metrics[METRIC_QUEUE_DELAY_COUNT], 1.0f);
        }
    }
}

// ============================================================
// Render System
// ============================================================

__device__ __forceinline__ void write_render_quad(
    RenderVertex* out,
    int base,
    float cx,
    float cy,
    float h,
    float L,
    float W,
    float r,
    float g,
    float b,
    float a
) {
    float c = cosf(h);
    float ss = sinf(h);

    float dx = c * L * 0.5f;
    float dy = ss * L * 0.5f;

    float nx = -ss * W * 0.5f;
    float ny =  c * W * 0.5f;

    float x0 = cx - dx - nx;
    float y0 = cy - dy - ny;

    float x1 = cx + dx - nx;
    float y1 = cy + dy - ny;

    float x2 = cx + dx + nx;
    float y2 = cy + dy + ny;

    float x3 = cx - dx + nx;
    float y3 = cy - dy + ny;

    RenderVertex v0 = { x0, y0, r, g, b, a, 1 };
    RenderVertex v1 = { x1, y1, r, g, b, a, 1 };
    RenderVertex v2 = { x2, y2, r, g, b, a, 1 };
    RenderVertex v3 = { x3, y3, r, g, b, a, 1 };

    out[base + 0] = v0;
    out[base + 1] = v1;
    out[base + 2] = v2;
    out[base + 3] = v0;
    out[base + 4] = v2;
    out[base + 5] = v3;
}

__device__ __forceinline__ void write_render_textured_quad(
    RenderVertex* out,
    int base,
    float cx,
    float cy,
    float h,
    float L,
    float W,
    int driver_type,
    int turn_signal,
    float turn_signal_time
) {
    /*
        EN: Textured vehicle quad. The RenderVertex layout is unchanged;
            r/g are reused as texture UV coordinates. The uploaded car image
            faces +X, which matches heading=0 in the simulation.
        KO: 텍스처 차량 사각형입니다. RenderVertex 레이아웃은 그대로 두고
            r/g 값을 텍스처 UV 좌표로 재사용합니다. 업로드 이미지의 앞쪽이 +X를
            향하므로 시뮬레이션 heading=0과 맞습니다.
    */
    float c = cosf(h);
    float ss = sinf(h);

    float dx = c * L * 0.5f;
    float dy = ss * L * 0.5f;

    float nx = -ss * W * 0.5f;
    float ny =  c * W * 0.5f;

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

    // EN: r/g are UV, b is driver flag, a is amber indicator blink strength.
    // KO: r/g는 UV, b는 운전자 flag, a는 황색 방향지시등 점멸 강도입니다.
    RenderVertex v0 = { x0, y0, 0.0f, 0.0f, driver_flag, blink_flag, 1.0f };
    RenderVertex v1 = { x1, y1, 1.0f, 0.0f, driver_flag, blink_flag, 1.0f };
    RenderVertex v2 = { x2, y2, 1.0f, 1.0f, driver_flag, blink_flag, 1.0f };
    RenderVertex v3 = { x3, y3, 0.0f, 1.0f, driver_flag, blink_flag, 1.0f };

    out[base + 0] = v0;
    out[base + 1] = v1;
    out[base + 2] = v2;
    out[base + 3] = v0;
    out[base + 4] = v2;
    out[base + 5] = v3;
}

__global__ void render_textured_body_system_kernel(
    RenderVertex* out,
    ECSArrays ecs,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities) return;

    int base = i * RENDER_BODY_VERTS_PER_VEHICLE;

    if (ecs.alive[i] != ENTITY_ALIVE) {
        RenderVertex z = { 0, 0, 0, 0, 0, 0, 1 };
        for (int k = 0; k < RENDER_BODY_VERTS_PER_VEHICLE; ++k) out[base + k] = z;
        return;
    }

    // EN: Slightly enlarge the textured quad so transparent pixels include mirrors/bumpers.
    // KO: 이미지의 투명 여백 안에 범퍼/사이드미러가 들어가므로 조금 여유 있게 그립니다.
    write_render_textured_quad(
        out,
        base,
        ecs.x[i],
        ecs.y[i],
        ecs.heading[i],
        fmaxf(ecs.length[i] * 1.06f, 2.0f),
        fmaxf(ecs.width[i] * 1.18f, 1.0f),
        ecs.driver_type[i],
        indicator_state_ecs(i, ecs),
        ecs.turn_signal_time != nullptr ? ecs.turn_signal_time[i] : 0.0f
    );
}

__global__ void render_body_system_kernel(
    RenderVertex* out,
    ECSArrays ecs,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities) return;

    int base = i * RENDER_BODY_VERTS_PER_VEHICLE;

    if (ecs.alive[i] != ENTITY_ALIVE) {
        RenderVertex z = { 0, 0, 0, 0, 0, 0, 1 };
        for (int k = 0; k < RENDER_BODY_VERTS_PER_VEHICLE; ++k) out[base + k] = z;
        return;
    }

    // EN: Human vehicles are white; AVs are sky-blue.
    // KO: 사람 운전 차량은 흰색, 자율주행 차량은 하늘색으로 표시합니다.
    float r = ecs.driver_type[i] == AV ? 0.55f : 1.00f;
    float g = ecs.driver_type[i] == AV ? 0.84f : 1.00f;
    float b = ecs.driver_type[i] == AV ? 1.00f : 1.00f;
    int sig = indicator_state_ecs(i, ecs);
    float sig_time = ecs.turn_signal_time != nullptr ? ecs.turn_signal_time[i] : 0.0f;
    if (sig != INDICATOR_NONE && fmodf(fmaxf(sig_time, 0.0f) * 2.25f, 1.0f) < 0.55f) {
        r = 1.00f;
        g = 0.72f;
        b = 0.08f;
    }

    write_render_quad(
        out,
        base,
        ecs.x[i],
        ecs.y[i],
        ecs.heading[i],
        fmaxf(ecs.length[i], 2.0f),
        fmaxf(ecs.width[i], 1.0f),
        r,
        g,
        b,
        1.0f
    );
}

__global__ void render_system_kernel(
    RenderVertex* out,
    ECSArrays ecs,
    int max_entities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_entities) return;

    int base = i * RENDER_FULL_VERTS_PER_VEHICLE;

    if (ecs.alive[i] != ENTITY_ALIVE) {
        RenderVertex z = { 0, 0, 0, 0, 0, 0, 1 };
        for (int k = 0; k < RENDER_FULL_VERTS_PER_VEHICLE; ++k) out[base + k] = z;
        return;
    }

    float cx = ecs.x[i];
    float cy = ecs.y[i];
    float h = ecs.heading[i];

    float L = fmaxf(ecs.length[i], 2.0f);
    float W = fmaxf(ecs.width[i], 1.0f);

    // EN: Human vehicles are white; AVs are sky-blue.
    // KO: 사람 운전 차량은 흰색, 자율주행 차량은 하늘색으로 표시합니다.
    float r = ecs.driver_type[i] == AV ? 0.55f : 1.00f;
    float g = ecs.driver_type[i] == AV ? 0.84f : 1.00f;
    float b = ecs.driver_type[i] == AV ? 1.00f : 1.00f;
    int sig = indicator_state_ecs(i, ecs);
    float sig_time = ecs.turn_signal_time != nullptr ? ecs.turn_signal_time[i] : 0.0f;
    if (sig != INDICATOR_NONE && fmodf(fmaxf(sig_time, 0.0f) * 2.25f, 1.0f) < 0.55f) {
        r = 1.00f;
        g = 0.72f;
        b = 0.08f;
    }

    write_render_quad(out, base, cx, cy, h, L, W, r, g, b, 1.0f);

    float fx = cosf(h);
    float fy = sinf(h);
    float sx = -fy;
    float sy = fx;

    float wheelbase = vehicle_wheelbase_from_length(L);
    float front_off = 0.5f * wheelbase;
    float rear_off = -0.5f * wheelbase;
    float lateral = 0.38f * W;

    float wheel_L = clampf_cuda(0.16f * L, 0.55f, 0.82f);
    float wheel_W = clampf_cuda(0.15f * W, 0.20f, 0.32f);

    float wr = 0.04f;
    float wg = 0.04f;
    float wb = 0.04f;

    float raw_steer = (ecs.steer_angle != nullptr) ? ecs.steer_angle[i] : 0.0f;
    float steer = clampf_cuda(
        raw_steer,
        -(ecs.driver_type[i] == HUMAN ? MAX_STEER_HUMAN : MAX_STEER_AV),
        (ecs.driver_type[i] == HUMAN ? MAX_STEER_HUMAN : MAX_STEER_AV)
    );

    float front_h = wrap_pi(h + steer);
    float rear_h = h;

    float flx = cx + fx * front_off + sx * lateral;
    float fly = cy + fy * front_off + sy * lateral;
    float frx = cx + fx * front_off - sx * lateral;
    float fry = cy + fy * front_off - sy * lateral;
    float rlx = cx + fx * rear_off + sx * lateral;
    float rly = cy + fy * rear_off + sy * lateral;
    float rrx = cx + fx * rear_off - sx * lateral;
    float rry = cy + fy * rear_off - sy * lateral;

    write_render_quad(out, base + 6,  flx, fly, front_h, wheel_L, wheel_W, wr, wg, wb, 1.0f);
    write_render_quad(out, base + 12, frx, fry, front_h, wheel_L, wheel_W, wr, wg, wb, 1.0f);
    write_render_quad(out, base + 18, rlx, rly, rear_h,  wheel_L, wheel_W, wr, wg, wb, 1.0f);
    write_render_quad(out, base + 24, rrx, rry, rear_h,  wheel_L, wheel_W, wr, wg, wb, 1.0f);
}

// ============================================================
// ECS Launcher
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
) {
    if (max_entities <= 0 || road.num_lanes <= 0) return;
    if (grid.cell_size <= 0.1f || grid.width <= 0 || grid.height <= 0) return;
    if (dt <= 0.0f || dt > 0.25f) return;

    int threads = 256;

    int entity_blocks = (max_entities + threads - 1) / threads;
    int world_cells = grid.width * grid.height;
    int world_blocks = (world_cells + threads - 1) / threads;

    int spawn_blocks = (spawn.num_spawn_points + threads - 1) / threads;

    // Step-scoped metrics reset.  Spawned/exited remain cumulative at 0 and 1.
    clear_float_kernel<<<1, 256, 0, stream>>>(metrics + 6, METRICS_SIZE - 6, 0.0f);

    clear_decision_kernel<<<entity_blocks, threads, 0, stream>>>(
        decision,
        max_entities
    );

    int total_slots = (reservation_table != nullptr && road.num_nodes > 0)
        ? road.num_nodes * RES_HORIZON_SLOTS
        : 0;

    if (total_slots > 0) {
        int res_blocks = (total_slots + threads - 1) / threads;

        clear_reservation_system<<<res_blocks, threads, 0, stream>>>(
            reservation_table,
            total_slots
        );
    }

    // Build grid before spawn.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    // Spawn.
    if (spawn.num_spawn_points > 0) {
        spawn_system_kernel<<<spawn_blocks, threads, 0, stream>>>(
            ecs,
            road,
            grid,
            spawn,
            rng_state,
            metrics,
            reservation_table,
            reservation_table != nullptr ? total_slots : 0,
            current_time,
            dt,
            max_entities,
            step_index
        );
    }

    // EN: Spawn lane locks temporarily reuse the reservation table only during
    //     spawn.  Clear it immediately afterward so the signal/priority
    //     reservation logic cannot see a spawn lock as a fake intersection owner.
    // KO: 스폰 차로 lock은 스폰 중에만 reservation table을 임시로 재사용합니다.
    //     스폰 직후 즉시 비워서 신호/우선순위 예약 로직이 스폰 lock을 가짜
    //     교차로 점유자로 오해하지 않게 합니다.
    if (total_slots > 0) {
        int res_blocks = (total_slots + threads - 1) / threads;
        clear_reservation_system<<<res_blocks, threads, 0, stream>>>(
            reservation_table,
            total_slots
        );
    }

    // Rebuild grid after spawn, then damp same-step entry overlaps with a
    // local grid lookup instead of an O(max_entities) full scan.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    resolve_spawn_overlap_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        grid,
        spawn,
        metrics,
        current_time,
        dt,
        max_entities
    );

    // Rebuild once more so forced/queued spawn states are visible to perception.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    // ECS pipeline.
    route_lane_repair_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        metrics,
        dt,
        max_entities
    );

    // Route repair can remove invalid actors or clamp their lane position.
    // Rebuild the grid so perception never sees stale repaired/free vehicles.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    turn_signal_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        metrics,
        dt,
        max_entities
    );

    perception_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        grid,
        perception,
        metrics,
        max_entities
    );

    decision_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        signals,
        grid,
        perception,
        decision,
        reservation_table,
        metrics,
        current_time,
        dt,
        max_entities
    );

    // EN: Reuse the reservation table as a node-scoped priority gate after
    //     decision.  It serializes conflicting starts into the intersection, so
    //     vehicles already inside can keep moving out without in-box deadlock.
    // KO: decision 이후 reservation table을 노드별 우선순위 게이트로 재사용합니다.
    //     교차로 진입 시작을 순서화하여, 이미 들어간 차량은 박스 안에서 멈추지 않고
    //     빠져나갈 수 있습니다.
    if (reservation_table != nullptr && road.num_nodes > 0) {
        int node_blocks = (road.num_nodes + threads - 1) / threads;

        clear_intersection_priority_gate_kernel<<<node_blocks, threads, 0, stream>>>(
            reservation_table,
            road.num_nodes
        );

        mark_intersection_occupancy_kernel<<<entity_blocks, threads, 0, stream>>>(
            ecs,
            road,
            reservation_table,
            max_entities
        );

        select_intersection_priority_candidates_kernel<<<entity_blocks, threads, 0, stream>>>(
            ecs,
            road,
            signals,
            decision,
            perception,
            reservation_table,
            metrics,
            current_time,
            max_entities
        );

        apply_intersection_priority_gate_kernel<<<entity_blocks, threads, 0, stream>>>(
            ecs,
            road,
            signals,
            decision,
            grid,
            perception,
            reservation_table,
            metrics,
            current_time,
            dt,
            max_entities
        );
    }

    // EN/KO: Nav-mesh style local avoidance adjusts acceleration before motion.
    local_obstacle_avoidance_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        grid,
        perception,
        decision,
        metrics,
        current_time,
        dt,
        max_entities
    );

    lane_change_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        decision,
        road,
        dt,
        max_entities
    );

    motion_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        decision,
        road,
        perception,
        metrics,
        current_time,
        dt,
        max_entities
    );

    connector_enter_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        decision,
        road,
        grid,
        metrics,
        current_time,
        max_entities
    );

    // Connector entry changes state and position semantics, so rebuild before
    // connector car-following reads nearby vehicles.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    connector_motion_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        grid,
        metrics,
        current_time,
        dt,
        max_entities
    );

    // Rebuild grid after movement.
    clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
        grid.cell_head,
        world_cells,
        WORLD_CELL_EMPTY
    );

    spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        max_entities
    );

    for (int contact_pass = 0; contact_pass < CONTACT_RESOLVE_PASSES; ++contact_pass) {
        contact_resolve_system_kernel<<<entity_blocks, threads, 0, stream>>>(
            ecs,
            road,
            grid,
            metrics,
            current_time,
            dt,
            max_entities
        );

        clear_int_kernel<<<world_blocks, threads, 0, stream>>>(
            grid.cell_head,
            world_cells,
            WORLD_CELL_EMPTY
        );

        spatial_hash_build_system<<<entity_blocks, threads, 0, stream>>>(
            ecs,
            grid,
            max_entities
        );
    }

    collision_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        metrics,
        dt,
        max_entities
    );

    safety_metrics_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        grid,
        metrics,
        max_entities
    );

    stats_system_kernel<<<entity_blocks, threads, 0, stream>>>(
        ecs,
        road,
        perception,
        decision,
        metrics,
        dt,
        max_entities
    );
}

// ============================================================
// Render Interop
// ============================================================

extern "C" void register_render_vbo_cuda(unsigned int vbo) {
    if (g_render_resource != nullptr) {
        cudaGraphicsUnregisterResource(g_render_resource);
        g_render_resource = nullptr;
    }

    cudaGraphicsGLRegisterBuffer(
        &g_render_resource,
        vbo,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
}

extern "C" void set_vehicle_texture_render_cuda(int enabled) {
    g_render_textured_cars = enabled != 0 ? 1 : 0;
}

extern "C" void launch_render_vbo_cuda_ecs(
    ECSArrays ecs,
    int max_entities,
    cudaStream_t stream
) {
    if (g_render_resource == nullptr || max_entities <= 0) return;

    cudaGraphicsMapResources(1, &g_render_resource, stream);

    RenderVertex* dev_ptr = nullptr;
    size_t size = 0;

    cudaGraphicsResourceGetMappedPointer(
        (void**)&dev_ptr,
        &size,
        g_render_resource
    );

    if (dev_ptr != nullptr) {
        int threads = 256;
        int blocks = (max_entities + threads - 1) / threads;

        size_t full_needed =
            (size_t)max_entities
            * (size_t)RENDER_FULL_VERTS_PER_VEHICLE
            * sizeof(RenderVertex);

        size_t body_needed =
            (size_t)max_entities
            * (size_t)RENDER_BODY_VERTS_PER_VEHICLE
            * sizeof(RenderVertex);

        if (g_render_textured_cars != 0 && size >= body_needed) {
            render_textured_body_system_kernel<<<blocks, threads, 0, stream>>>(
                dev_ptr,
                ecs,
                max_entities
            );
        } else if (size >= full_needed) {
            render_system_kernel<<<blocks, threads, 0, stream>>>(
                dev_ptr,
                ecs,
                max_entities
            );
        } else if (size >= body_needed) {
            render_body_system_kernel<<<blocks, threads, 0, stream>>>(
                dev_ptr,
                ecs,
                max_entities
            );
        }
    }

    cudaGraphicsUnmapResources(1, &g_render_resource, stream);
}

extern "C" void unregister_render_vbo_cuda() {
    if (g_render_resource != nullptr) {
        cudaGraphicsUnregisterResource(g_render_resource);
        g_render_resource = nullptr;
    }
}