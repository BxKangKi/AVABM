
# main.py
# ECS CUDA traffic simulation Python frontend
# EN: Surface-aware traffic frontend with road surfaces, lane markings, and bilingual comments.
# KO: 도로 면, 차선 렌더링, 현실적 회전/우선권 모델을 지원하는 Python 프런트엔드입니다.

import os
import math
import time
import json
import csv
import shutil
import queue
import ctypes
import threading
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import geopandas as gpd
import networkx as nx
import torch
from shapely.geometry import Point, LineString, MultiLineString

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    import pyproj
except Exception:
    pyproj = None


# ============================================================
# Config
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.txt"


def _read_config_txt(path: Path):
    """EN: Read key=value settings from config.txt.
    KO: config.txt에서 key=value 설정을 읽습니다.

    Environment-variable style settings were moved into config.txt so both the
    Python frontend and the batch files use one source of truth.  Keys are case
    insensitive.  Empty strings, None and null can be used for optional values.
    """
    data = {}
    try:
        if not path.exists():
            return data
        with path.open("r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip()
                if not key:
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                else:
                    # Keep Windows paths intact while still allowing trailing comments.
                    for marker in (" #", " ;"):
                        pos = value.find(marker)
                        if pos >= 0:
                            value = value[:pos].rstrip()
                data[key] = value
    except Exception as e:
        print(f"[Config] failed to read {path}: {e}")
    return data


CONFIG = _read_config_txt(CONFIG_PATH)


def cfg(name, default=None):
    return CONFIG.get(str(name).upper(), default)


def cfg_optional(name, default=None):
    value = cfg(name, default)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ["", "none", "null", "nil"]:
        return None
    return value


def cfg_bool(name, default=False):
    value = cfg(name, None)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ["1", "true", "t", "yes", "y", "on", "enable", "enabled"]:
            return True
        if s in ["0", "false", "f", "no", "n", "off", "disable", "disabled", "none", "null", ""]:
            return False
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def cfg_int(name, default=0):
    value = cfg(name, default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def cfg_float(name, default=0.0):
    value = cfg(name, default)
    try:
        return float(value)
    except Exception:
        return float(default)


ROAD_GPKG = cfg("ROAD_GPKG", "data/road_links.gpkg")
SPAWN_GPKG = cfg("SPAWN_GPKG", "data/spawn_points.gpkg")
SIGNAL_GPKG = cfg("SIGNAL_GPKG", "data/signals.gpkg")

GPKG_LAYER = cfg("GPKG_LAYER", "Grid")
SPAWN_GPKG_LAYER = cfg_optional("SPAWN_GPKG_LAYER", GPKG_LAYER)
SIGNAL_GPKG_LAYER = cfg_optional("SIGNAL_GPKG_LAYER", None)

SCREEN_W = cfg_int("SCREEN_W", 1600)
SCREEN_H = cfg_int("SCREEN_H", 900)

MAX_AGENTS = cfg_int("MAX_AGENTS", 120000)
NUM_ROUTES = cfg_int("NUM_ROUTES", 8192)

DT = cfg_float("DT", 0.05)
AV_PENETRATION = cfg_float("AV_PENETRATION", 0.0)

ROUTE_SEED = cfg_int("ROUTE_SEED", 20260529)
MIN_TRIP_DISTANCE = cfg_float("MIN_TRIP_DISTANCE", 3000.0)
MAX_ROUTE_TRIES = cfg_int("MAX_ROUTE_TRIES", 120000)

ROUTE_CACHE_DIR = cfg("ROUTE_CACHE_DIR", "data/route_cache")
REFRESH_ROUTE_CACHE = cfg_bool("REFRESH_ROUTE_CACHE", False)
BACKUP_ROUTE_CACHE = cfg_bool("BACKUP_ROUTE_CACHE", True)
ROUTE_PARALLEL = cfg_bool("ROUTE_PARALLEL", True)
ROUTE_PARALLEL_BACKEND = str(cfg("ROUTE_PARALLEL_BACKEND", "process")).lower()
ROUTE_WORKERS = cfg_int("ROUTE_WORKERS", max(1, min((os.cpu_count() or 4) - 1, 32)))
# EN/KO: v15 caches the final CUDA-ready routes after multi-lane spawn
# expansion and lane safety repair, so the expensive post-processing does not
# run again every time an unchanged network starts.  ROUTE_CACHE_FINAL_READY is
# kept as a backward-compatible alias for ROUTE_READY_CACHE.
ROUTE_READY_CACHE = cfg_bool("ROUTE_READY_CACHE", cfg_bool("ROUTE_CACHE_FINAL_READY", True))
ROUTE_CACHE_FINAL_READY = bool(ROUTE_READY_CACHE)
ROUTE_SPAWN_EXPANSION_PARALLEL = cfg_bool("ROUTE_SPAWN_EXPANSION_PARALLEL", True)
ROUTE_SPAWN_EXPANSION_BACKEND = str(cfg("ROUTE_SPAWN_EXPANSION_BACKEND", "thread")).lower()
_route_spawn_workers_cfg = cfg_int("ROUTE_SPAWN_EXPANSION_WORKERS", 0)
ROUTE_SPAWN_EXPANSION_WORKERS = max(1, int(ROUTE_WORKERS if _route_spawn_workers_cfg <= 0 else _route_spawn_workers_cfg))
ROUTE_SPAWN_EXPANSION_MIN_ROUTES = cfg_int("ROUTE_SPAWN_EXPANSION_MIN_ROUTES", 1024)
ROUTE_CACHE_VERSION = "ecs_v29_spawn_expansion_parallel_ready_cache"

WORLD_CELL_SIZE = cfg_float("WORLD_CELL_SIZE", 20.0)
WORLD_GRID_MAX_W = cfg_int("WORLD_GRID_MAX_W", 2048)
WORLD_GRID_MAX_H = cfg_int("WORLD_GRID_MAX_H", 2048)

MAX_CONFLICT_LANES = 8
CELL_COUNT_PER_LANE = 1
RES_HORIZON_SLOTS = 16
METRICS_SIZE = 112
DEFAULT_LANE_WIDTH = 3.5

# EN: Road direction / one-way handling.  This controls whether a source
#     centerline is split into two opposite-direction carriageways.
#     auto        = read ONEWAY/ONE_WAY fields and OSM other_tags such as
#                   "oneway"=>"yes".
#     force_oneway = never split the centerline; all lanes follow the geometry.
#     force_two_way = always split the centerline into opposing directions.
# KO: 도로 방향/일방통행 처리입니다. 원본 중심선을 양방향 차로로 나눌지
#     결정합니다. Dongbu처럼 OSM other_tags 안에 oneway=yes, lanes=3이
#     들어 있는 레이어는 auto로 두면 자동으로 중앙 분리 없이 일방통행
#     3차로가 생성됩니다.
ROAD_ONEWAY_MODE = str(cfg("ROAD_ONEWAY_MODE", cfg("ONEWAY_MODE", "auto"))).strip().lower()
ROAD_DEFAULT_ONEWAY = cfg_bool("ROAD_DEFAULT_ONEWAY", cfg_bool("DEFAULT_ONEWAY", False))
ROAD_PARSE_OSM_OTHER_TAGS = cfg_bool("ROAD_PARSE_OSM_OTHER_TAGS", True)
# EN/KO: Plain keyword hints for custom rows whose other_tags is not OSM hstore.
# Dongbu has rows marked enterance/sideway/sidway with ROAD_BT=3; these are
# one-lane slip/side roads and should not be split into two opposing lanes.
ROAD_ONEWAY_HINT_KEYWORDS = str(cfg("ROAD_ONEWAY_HINT_KEYWORDS", "enterance,entrance,sideway,sidway,slip"))
# EN/KO: Optional narrow-road fallback. 0 disables it.
ROAD_NARROW_ONEWAY_MAX_WIDTH = cfg_float("ROAD_NARROW_ONEWAY_MAX_WIDTH", 0.0)

RENDER_BODY_VERTS_PER_VEHICLE = 6
RENDER_FULL_VERTS_PER_VEHICLE = 30
RENDER_WHEELS = cfg_bool("RENDER_WHEELS", False)
RENDER_VERTS_PER_VEHICLE = RENDER_FULL_VERTS_PER_VEHICLE if RENDER_WHEELS else RENDER_BODY_VERTS_PER_VEHICLE

# EN: Lane marking render switches.  These were referenced by the render path
#     but were missing from the config block in an earlier build.
# KO: 차선 렌더링 스위치입니다. 이전 빌드에서는 렌더링 함수가 이 값을
#     참조하지만 Config 블록에 정의가 없어 NameError가 날 수 있었습니다.
DRAW_LANE_MARKINGS = cfg_bool("DRAW_LANE_MARKINGS", True)
DRAW_LANE_CENTERLINES = cfg_bool("DRAW_LANE_CENTERLINES", True)
DRAW_LANE_EDGES = cfg_bool("DRAW_LANE_EDGES", True)
LANE_MARKING_WIDTH = cfg_float("LANE_MARKING_WIDTH", 0.5)
LANE_DASH_LENGTH = cfg_float("LANE_DASH_LENGTH", 7.0)
LANE_DASH_GAP = cfg_float("LANE_DASH_GAP", 5.0)
LANE_CENTER_DIVIDER_COLOR = (1.0, 0.58, 0.05, 0.96)
LANE_SEPARATOR_COLOR = (0.96, 0.96, 0.90, 0.80)
LANE_EDGE_COLOR = (0.82, 0.82, 0.78, 0.42)

# EN: Curved intersection asphalt fillets.  The previous semicircle apron could
#     make the node look like a circle at the middle of the intersection.  The
#     renderer now paints compact corner fillets between adjacent road arms,
#     matching the rounded inner curb shape shown in the reference image.
# KO: 교차로 곡선 아스팔트 필렛 설정입니다. 이전 반원 apron은 교차로 중심에
#     원이 있는 것처럼 보일 수 있었습니다. 이제 인접 도로 팔 사이의 안쪽 모서리에
#     작은 곡선 필렛을 그려, 보내준 참고 이미지처럼 코너가 둥글게 이어지게 합니다.
DRAW_INTERSECTION_APRONS = cfg_bool("DRAW_INTERSECTION_APRONS", True)
INTERSECTION_APRON_RADIUS_MULT = cfg_float("INTERSECTION_APRON_RADIUS_MULT", 2.05)
INTERSECTION_APRON_EXTRA = cfg_float("INTERSECTION_APRON_EXTRA", 2.5)
INTERSECTION_APRON_MAX_RADIUS = cfg_float("INTERSECTION_APRON_MAX_RADIUS", 14.0)
INTERSECTION_APRON_SEGMENTS = cfg_int("INTERSECTION_APRON_SEGMENTS", 32)
INTERSECTION_APRON_CENTER_FILL_MULT = cfg_float("INTERSECTION_APRON_CENTER_FILL_MULT", 0.0)
INTERSECTION_APRON_MIN_GAP_DEG = cfg_float("INTERSECTION_APRON_MIN_GAP_DEG", 28.0)
# Kept for compatibility with older config.txt files.  It is ignored by the new
# corner-fillet renderer because the curve is no longer a center semicircle fan.
INTERSECTION_APRON_SEMICIRCLE_FLIP = cfg_bool("INTERSECTION_APRON_SEMICIRCLE_FLIP", True)

# EN: Zoom can now go much deeper for inspecting lane-following right turns and
#     deadlock release behavior.
# KO: 우회전 경로와 데드락 해소 과정을 가까이 볼 수 있도록 최대 확대를 늘렸습니다.
ZOOM_MIN = cfg_float("ZOOM_MIN", 0.05)
ZOOM_MAX = cfg_float("ZOOM_MAX", 120.0)
ZOOM_SPEED = cfg_float("ZOOM_SPEED", 1.14)

# EN: Lane-marking grouping.  Grouping by physical source row/segment is more stable
#     than grouping by sorted node ids when GIS node ordering is irregular.
# KO: 차선 표시 그룹입니다. GIS 노드 번호가 불규칙할 때 sorted node 기준보다
#     원본 도로 row/segment 기준으로 묶는 쪽이 양방향 점선 표시가 안정적입니다.
LANE_MARKING_GROUP_BY_SOURCE = cfg_bool("LANE_MARKING_GROUP_BY_SOURCE", True)

# EN: Textured vehicle rendering. Put the uploaded car image here:
#     assets/car_topdown.png. The image is drawn as one textured quad per car.
# KO: 차량 이미지 렌더링 설정입니다. 업로드한 차량 이미지를
#     assets/car_topdown.png 위치에 넣으면 차량당 하나의 텍스처 사각형으로 그립니다.
ASSET_DIR = Path(cfg("ASSET_DIR", str(PROJECT_DIR / "assets")))
CAR_TEXTURE_PATH = cfg("CAR_TEXTURE_PATH", str(ASSET_DIR / "car_topdown.png"))
USE_TEXTURED_CARS = cfg_bool("USE_TEXTURED_CARS", True)

# EN: Textured cars use the 6-vertex body VBO. Wheel triangles are disabled because
#     the wheel/body shape is already contained in the uploaded image.
# KO: 텍스처 차량은 6정점 차체 VBO를 사용합니다. 바퀴/차체 모양은 이미지 안에
#     이미 들어 있으므로 별도 바퀴 삼각형은 끕니다.
if USE_TEXTURED_CARS:
    RENDER_WHEELS = False
    RENDER_VERTS_PER_VEHICLE = RENDER_BODY_VERTS_PER_VEHICLE

# EN: Signal hover panel settings. The panel is rendered in-screen when the mouse
#     is near a signal point.
# KO: 신호등 hover 패널 설정입니다. 마우스가 신호점 근처에 오면 화면 위에
#     phase 상세 정보를 표시합니다.
SIGNAL_HOVER_RADIUS_PX = cfg_float("SIGNAL_HOVER_RADIUS_PX", 18.0)
SHOW_SIGNAL_HOVER = cfg_bool("SHOW_SIGNAL_HOVER", True)

# EN: Vehicle hover panel.  The nearest active car under the mouse is queried
#     on GPU and only one vehicle record is copied back to CPU.
# KO: 차량 hover 패널입니다. 마우스 아래의 가장 가까운 활성 차량을 GPU에서 찾고,
#     선택된 차량 한 대의 정보만 CPU로 가져옵니다.
SHOW_VEHICLE_HOVER = cfg_bool("SHOW_VEHICLE_HOVER", True)
VEHICLE_HOVER_RADIUS_PX = cfg_float("VEHICLE_HOVER_RADIUS_PX", 24.0)


TURN_LEFT = -1
TURN_STRAIGHT = 0
TURN_RIGHT = 1
TURN_ANY = 99

INDICATOR_NONE = 0
INDICATOR_LEFT = -1
INDICATOR_RIGHT = 1
INDICATOR_HAZARD = 2

LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2

VEH_ON_LANE = 0
VEH_IN_CONNECTOR = 1

BASE_VPS = cfg_float("BASE_VPS", 0.2)
SPAWN_REF_WIDTH = cfg_float("SPAWN_REF_WIDTH", 10.5)
SPAWN_WIDTH_EXP_K = cfg_float("SPAWN_WIDTH_EXP_K", 1.25)
SPAWN_LANE_POWER = cfg_float("SPAWN_LANE_POWER", 1.15)
SPAWN_MAX_MULT = cfg_float("SPAWN_MAX_MULT", 18.0)
MAX_TOTAL_VPS = cfg_float("MAX_TOTAL_VPS", 40.0)

# EN: When a spawn node feeds a multi-lane outgoing link, create spawn slots
#     for every usable lane and divide that spawn point's demand evenly across
#     those lanes.  This prevents all vehicles from appearing in only the
#     route generator's first selected lane.
# KO: 스폰 노드가 여러 차로짜리 진입 링크로 이어질 때, 사용 가능한 모든 차로에
#     스폰 슬롯을 만들고 수요를 균등 분배합니다. 경로 생성기가 처음 고른 한
#     차로에만 차량이 몰리는 현상을 막습니다.
SPAWN_MULTI_LANE_BALANCE = cfg_bool("SPAWN_MULTI_LANE_BALANCE", True)

# EN: Route-lane intent model.  Keep the current relative lane on straight
#     multi-lane arterials, move toward the correct edge before left/right exits,
#     and use a stable per-route random lane for through movements instead of
#     falling back to the center lane every time.
# KO: 경로 차선 의도 모델입니다. 직진 다차로 간선에서는 기존 상대 차선을 유지하고,
#     좌/우 진출 전에는 해당 가장자리 차선으로 이동하며, 단순 직진 차량은 항상
#     가운데가 아니라 route별 안정적인 랜덤 차선을 점유합니다.
ROUTE_KEEP_RELATIVE_LANE = cfg_bool("ROUTE_KEEP_RELATIVE_LANE", True)
ROUTE_DESTINATION_LANE_INTENT = cfg_bool("ROUTE_DESTINATION_LANE_INTENT", True)
ROUTE_STRAIGHT_LANE_RANDOMIZE = cfg_bool("ROUTE_STRAIGHT_LANE_RANDOMIZE", True)
ROUTE_RANDOM_LANE_SALT = cfg_int("ROUTE_RANDOM_LANE_SALT", 91127)
# EN/KO: Drop route-cache entries whose consecutive lane IDs do not connect.
# This prevents a vehicle from being assigned to a lane that cannot reach the
# route's next lane, which previously could leave it stopped in the road.
ROUTE_VALIDATE_LANE_CONNECTIVITY = cfg_bool("ROUTE_VALIDATE_LANE_CONNECTIVITY", True)
ROUTE_DROP_INVALID_LANE_PATHS = cfg_bool("ROUTE_DROP_INVALID_LANE_PATHS", True)

# EN: Multi-lane continuation / destination spread guard.  Some GIS links split
#     a continuous mainline into 4->3 or 3->4 lane-count changes with a curved
#     heading, which can be misread as a left/right turn.  Treat those wide
#     lane-count transitions as straight continuations, and keep destination
#     links from collapsing to a single edge lane.
# KO: 다차로 본선이 4->3 또는 3->4처럼 차로 수만 바뀌며 이어지는 경우, 곡선
#     각도 때문에 좌/우회전으로 오인되어 가장자리 차선 대기가 걸릴 수 있습니다.
#     이런 넓은 차로수 변화 구간은 직진 연속 구간으로 보고, 도착 링크도 한쪽
#     차선으로만 몰리지 않도록 분산합니다.
ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT = cfg_bool("ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT", True)
ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG = cfg_float("ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG", 62.0)
ROUTE_LANE_COUNT_CHANGE_BALANCE = cfg_bool("ROUTE_LANE_COUNT_CHANGE_BALANCE", True)
ROUTE_DESTINATION_LANE_SPREAD = cfg_bool("ROUTE_DESTINATION_LANE_SPREAD", True)

# EN: Route path cost bias.  Pure shortest-path routing can choose narrow side
#     links beside a multi-lane arterial.  These knobs keep Dongbu-like mainline
#     routes on wider/larger-capacity links unless the detour is clearly needed.
# KO: 경로 비용 보정입니다. 단순 최단거리만 쓰면 다차로 본선 옆의 좁은 연결로를
#     고르는 일이 있습니다. 아래 값들은 동부간선도로처럼 넓은 본선 차로를 가능한
#     유지하도록 비용을 보정합니다.
ROUTE_USE_WIDTH_BIASED_COST = cfg_bool("ROUTE_USE_WIDTH_BIASED_COST", True)
ROUTE_WIDE_LINK_REF_LANES = cfg_float("ROUTE_WIDE_LINK_REF_LANES", 3.0)
ROUTE_WIDE_LINK_POWER = cfg_float("ROUTE_WIDE_LINK_POWER", 0.80)
ROUTE_NARROW_LINK_PENALTY = cfg_float("ROUTE_NARROW_LINK_PENALTY", 4.25)
ROUTE_NARROW_WIDTH_MAX = cfg_float("ROUTE_NARROW_WIDTH_MAX", 7.5)
ROUTE_WIDE_LINK_MIN_COST_FACTOR = cfg_float("ROUTE_WIDE_LINK_MIN_COST_FACTOR", 0.75)
ROUTE_WIDE_LINK_MAX_COST_FACTOR = cfg_float("ROUTE_WIDE_LINK_MAX_COST_FACTOR", 5.50)

# EN: Interchange/ramp edge-lane model.  One-lane ramps or lane-drop links
#     connected to a multi-lane mainline must enter/exit through the nearest
#     outside lane, never through the visual/geometry center of the main road.
# KO: 나들목/램프 최외곽 차로 모델입니다. 1차로 램프 또는 차로 감소 링크가
#     다차로 본선에 붙을 때 도로 중앙이 아니라 가장 가까운 바깥 차로로만
#     진입/진출하게 합니다.
INTERCHANGE_EDGE_ONLY = cfg_bool("INTERCHANGE_EDGE_ONLY", True)
INTERCHANGE_RAMP_MAX_LANES = cfg_int("INTERCHANGE_RAMP_MAX_LANES", 1)
INTERCHANGE_MAIN_MIN_LANES = cfg_int("INTERCHANGE_MAIN_MIN_LANES", 2)
INTERCHANGE_ADJUST_RAMP_GEOMETRY = cfg_bool("INTERCHANGE_ADJUST_RAMP_GEOMETRY", True)

# EN: Optional time-of-day spawn demand profile loaded from spawn-point attributes.
#     Fields named SPWN01..SPWN24 (or SPWN00..SPWN23) are treated as hourly
#     traffic volumes and linearly interpolated by simulation time.  Missing
#     fields keep the generated default demand.
# KO: 스폰 포인트 속성에서 시간대별 수요를 읽는 설정입니다.
#     SPWN01..SPWN24 또는 SPWN00..SPWN23 필드는 시간대별 교통량으로 보고
#     시뮬레이션 시간에 따라 선형 보간합니다. 해당 필드가 없으면 기존 기본
#     스폰 수요를 그대로 사용합니다.
SPAWN_PROFILE_SLOTS = cfg_int("SPAWN_PROFILE_SLOTS", 24)
SPAWN_PROFILE_SLOT_SECONDS = cfg_float("SPAWN_PROFILE_SLOT_SECONDS", 3600.0)
SPAWN_PROFILE_FIELD_PREFIX = cfg("SPAWN_PROFILE_FIELD_PREFIX", "SPWN")
SPAWN_PROFILE_UNIT = str(cfg("SPAWN_PROFILE_UNIT", "vph")).strip().lower()
SPAWN_NODE_MATCH_MAX_DIST = cfg_float("SPAWN_NODE_MATCH_MAX_DIST", 150.0)

SPWNTYPE_ORIGIN = 1
SPWNTYPE_DESTINATION = 2
SPWNTYPE_BOTH = SPWNTYPE_ORIGIN | SPWNTYPE_DESTINATION

TURN_LANE_ROUTE_PREP_BASE = cfg_float("TURN_LANE_ROUTE_PREP_BASE", 30.0)
TURN_LANE_ROUTE_PREP_PER_LANE = cfg_float("TURN_LANE_ROUTE_PREP_PER_LANE", 24.0)
STRICT_ROUTE_TURN_LANE_FILTER = cfg_bool("STRICT_ROUTE_TURN_LANE_FILTER", False)

SIGNAL_NODE_MATCH_MAX_DIST = cfg_float("SIGNAL_NODE_MATCH_MAX_DIST", 120.0)
DEBUG_SYNC = cfg_bool("DEBUG_SYNC", False)
METRICS_PATH = cfg("METRICS_PATH", "data/simulation_metrics.csv")

# Performance knobs.  RENDER_INTERVAL=2 updates the vehicle VBO every other rendered frame.
# PHYSICS_STEPS_PER_FRAME runs multiple CUDA physics ticks before each OpenGL redraw.
# This is useful for 24-hour studies where rendering every 0.1 simulated second is too slow.
RENDER_INTERVAL = max(1, cfg_int("RENDER_INTERVAL", 1))
FPS_LIMIT = cfg_int("FPS_LIMIT", 60)
PHYSICS_STEPS_PER_FRAME = max(1, cfg_int("PHYSICS_STEPS_PER_FRAME", cfg_int("FAST_FORWARD_STEPS_PER_FRAME", 1)))
METRICS_INTERVAL = cfg_float("METRICS_INTERVAL", 2.0)

# ============================================================
# Simulation duration / research section outputs
# ============================================================
# EN: Negative duration means run until ESC/window close. Non-negative value auto-stops
#     after that many simulated seconds and saves all section outputs.
# KO: 음수이면 ESC/창닫기까지 실행합니다. 0 이상이면 지정한 시뮬레이션 초만큼
#     실행한 뒤 자동 종료하고 구간별 결과를 저장합니다.
SCENARIO_ID = str(cfg("SCENARIO_ID", "default"))
SCENARIO_SEED = cfg_int("SCENARIO_SEED", ROUTE_SEED)
SIMULATION_DURATION_SECONDS = cfg_float("SIMULATION_DURATION_SECONDS", -1.0)

# EN: SECTION_LENGTH_M > 0 splits each directed road link.  -1 saves one
#     whole-network section.
# KO: SECTION_LENGTH_M > 0이면 방향성 링크를 구간으로 분할하고, -1이면 전체
#     네트워크를 하나의 구간으로 저장합니다.
SECTION_STATS_ENABLED = cfg_bool("SECTION_STATS_ENABLED", True)
SECTION_LENGTH_M = cfg_float("SECTION_LENGTH_M", 250.0)
SECTION_STATS_INTERVAL = max(1.0e-6, cfg_float("SECTION_STATS_INTERVAL", cfg_float("SECTION_METRICS_INTERVAL", 60.0)))
SECTION_OUTPUT_DIR = Path(cfg("SECTION_OUTPUT_DIR", "data"))
SECTION_STATS_CSV = cfg("SECTION_STATS_CSV", cfg("SECTION_TIME_BINS_CSV_PATH", "data/section_time_bins_{section_length_label}.csv"))
SECTION_SUMMARY_CSV = cfg("SECTION_SUMMARY_CSV", cfg("SECTION_SCENARIO_CSV_PATH", "data/scenario_summary_{section_length_label}.csv"))
SECTION_BASE_GPKG_PATH = cfg("SECTION_BASE_GPKG_PATH", cfg("SECTION_ROAD_GPKG_PATH", "data/road_sections_{section_length_label}.gpkg"))
SECTION_FINAL_GPKG_PATH = cfg("SECTION_FINAL_GPKG_PATH", cfg("SECTION_RESULTS_GPKG_PATH", "data/section_results_{section_length_label}.gpkg"))
SECTION_BASE_LAYER = cfg("SECTION_BASE_LAYER", "road_sections")
SECTION_RESULTS_LAYER = cfg("SECTION_RESULTS_LAYER", "section_summary")
SECTION_TIME_LAYER = cfg("SECTION_TIME_LAYER", "section_time_bins")
SECTION_WRITE_BASE_GPKG = cfg_bool("SECTION_WRITE_BASE_GPKG", cfg_bool("SECTION_WRITE_ROAD_GPKG", True))
SECTION_WRITE_TIME_GPKG = cfg_bool("SECTION_WRITE_TIME_GPKG", cfg_bool("SECTION_WRITE_TIME_BIN_GPKG", True))
SECTION_MAX_TIME_GPKG_ROWS = cfg_int("SECTION_MAX_TIME_GPKG_ROWS", cfg_int("SECTION_TIME_BIN_GPKG_MAX_ROWS", 750000))
SECTION_WRITE_EMPTY_ROWS = cfg_bool("SECTION_WRITE_EMPTY_ROWS", cfg_bool("SECTION_SAVE_EMPTY_BINS", True))
SECTION_BACKUP_OUTPUTS = cfg_bool("SECTION_BACKUP_OUTPUTS", cfg_bool("SECTION_STATS_BACKUP", True))
SECTION_FREEFLOW_SPEED_MPS = cfg_float("SECTION_FREEFLOW_SPEED_MPS", -1.0)
SECTION_CONGESTION_SPEED_RATIO = cfg_float("SECTION_CONGESTION_SPEED_RATIO", cfg_float("CONGESTION_SPEED_RATIO", 0.5))
SECTION_QUEUE_SPEED_KMH = cfg_float("SECTION_QUEUE_SPEED_KMH", cfg_float("QUEUE_SPEED_KMH", 10.0))
SECTION_STOP_SPEED_KMH = cfg_float("SECTION_STOP_SPEED_KMH", cfg_float("STOP_SPEED_KMH", 5.0))
SECTION_GO_SPEED_KMH = cfg_float("SECTION_GO_SPEED_KMH", cfg_float("GO_SPEED_KMH", 30.0))
SECTION_HARD_BRAKE_MPS2 = cfg_float("SECTION_HARD_BRAKE_MPS2", cfg_float("HARD_BRAKE_MPS2", -3.0))
SECTION_OCCUPANCY_EFFECTIVE_VEHICLE_LENGTH_M = cfg_float("SECTION_OCCUPANCY_EFFECTIVE_VEHICLE_LENGTH_M", 7.5)
SECTION_VIS_WIDTH_BASE = cfg_float("SECTION_VIS_WIDTH_BASE", 1.0)
SECTION_VIS_WIDTH_ALPHA = cfg_float("SECTION_VIS_WIDTH_ALPHA", 5.0)

# ============================================================
# Utilities
# ============================================================

def format_sim_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["1", "y", "yes", "true", "t", "oneway", "일방", "일방통행"]:
            return True
        if s in ["0", "n", "no", "false", "f", "양방", "양방향"]:
            return False
    try:
        return bool(int(v))
    except Exception:
        return default



def is_missing_value(v) -> bool:
    """EN/KO: Robust missing-value check for pandas/geopandas attribute cells."""
    if v is None:
        return True
    try:
        if hasattr(v, "isna"):
            return bool(v.isna())
    except Exception:
        pass
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def parse_osm_other_tags(v):
    """EN: Parse common OSM hstore-like other_tags strings.
    KO: GeoPackage의 OSM other_tags(예: "oneway"=>"yes")를 dict로 변환합니다.
    """
    if is_missing_value(v):
        return {}
    text = str(v).strip()
    if not text:
        return {}
    tags = {}
    # Standard ogr2ogr OSM hstore representation: "key"=>"value",...
    for m in re.finditer(r'"([^"]+)"\s*=>\s*"([^"]*)"', text):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key:
            tags[key] = val
    if tags:
        return tags
    # Loose fallback for key=>value or key=value lists. Plain labels such as
    # "enterance" and "sideway" are handled separately as one-way hints.
    for m in re.finditer(r'([^,;=><]+)\s*(?:=>|=)\s*([^,;]+)', text):
        key = str(m.group(1)).strip().strip('\"\'').lower()
        val = str(m.group(2)).strip().strip('\"\'')
        if key:
            tags[key] = val
    return tags


def parse_first_number(v, default=None):
    """EN/KO: Extract the first numeric value from GIS/OSM text."""
    if is_missing_value(v):
        return default
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except Exception:
            return default
    s = str(v).strip().replace(',', '')
    if not s:
        return default
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return default
    try:
        f = float(m.group(0))
        return f if math.isfinite(f) else default
    except Exception:
        return default


def parse_first_int(v, default=None):
    f = parse_first_number(v, default=None)
    if f is None:
        return default
    try:
        return int(round(float(f)))
    except Exception:
        return default


def first_tag(tags, names, default=None):
    if not tags:
        return default
    for name in names:
        key = str(name).strip().lower()
        if key in tags and not is_missing_value(tags[key]):
            return tags[key]
    return default


def first_existing_or_tag(row, row_names, tags=None, tag_names=None, default=None):
    v = first_existing(row, row_names, None)
    if not is_missing_value(v):
        return v
    if ROAD_PARSE_OSM_OTHER_TAGS:
        v = first_tag(tags or {}, tag_names or [], None)
        if not is_missing_value(v):
            return v
    return default


def parse_speed_kmh(v, default=60.0):
    if is_missing_value(v):
        return float(default)
    s = str(v).strip().lower()
    if s in ["", "none", "null", "signals", "variable"]:
        return float(default)
    num = parse_first_number(v, default=None)
    if num is None:
        return float(default)
    if "mph" in s:
        num *= 1.609344
    return float(num)


def parse_lane_count_from_row(row, road_width, tags):
    """EN/KO: Prefer explicit GIS/OSM lane counts before width inference."""
    direct = first_existing_or_tag(
        row,
        ["LANES", "LANE_CNT", "LANE_COUNT", "차로수", "차선수"],
        tags=tags,
        tag_names=["lanes", "lanes:total"],
        default=None,
    )
    n = parse_first_int(direct, default=None)
    if n is not None and n > 0:
        return max(1, int(n))

    # Some OSM rows store the two directions separately.
    nf = parse_first_int(first_tag(tags, ["lanes:forward"], None), default=0) or 0
    nb = parse_first_int(first_tag(tags, ["lanes:backward"], None), default=0) or 0
    if nf + nb > 0:
        return max(1, int(nf + nb))

    return max(1, int(round(float(road_width) / DEFAULT_LANE_WIDTH)))


def _oneway_mode_value():
    mode = str(ROAD_ONEWAY_MODE or "auto").strip().lower().replace('-', '_').replace(' ', '_')
    if mode in ["1", "true", "yes", "y", "on", "oneway", "force", "force_oneway", "all_oneway", "일방", "일방통행"]:
        return "force_oneway"
    if mode in ["0", "false", "no", "n", "off", "two_way", "force_two_way", "bidirectional", "양방", "양방향"]:
        return "force_two_way"
    return "auto"


def parse_road_oneway(row, tags, road_width):
    """EN: Decide if one source centerline should be treated as one-way.
    KO: 원본 중심선을 일방통행으로 볼지 결정합니다. auto 모드에서는 GIS 필드,
        OSM other_tags, 사용자 keyword hint, 선택적 폭 기준을 차례로 봅니다.
    """
    mode = _oneway_mode_value()
    if mode == "force_oneway":
        return True
    if mode == "force_two_way":
        return False

    v = first_existing_or_tag(
        row,
        ["ONEWAY", "ONE_WAY", "ONEWAY_YN", "일방통행", "oneway"],
        tags=tags,
        tag_names=["oneway", "oneway:vehicle", "vehicle:oneway"],
        default=None,
    )
    if not is_missing_value(v):
        s = str(v).strip().lower().replace('-', '_').replace(' ', '_')
        if s in ["1", "y", "yes", "true", "t", "on", "oneway", "일방", "일방통행", "reverse", "_1"]:
            return True
        if s in ["0", "n", "no", "false", "f", "off", "both", "two_way", "bidirectional", "양방", "양방향"]:
            return False
        # OSM uses -1 for a reversed one-way relative to the stored geometry.
        # This frontend does not currently reverse geometry for -1, but it must
        # still avoid creating an artificial opposite-direction bundle.
        if str(v).strip() == "-1":
            return True

    if str(first_tag(tags, ["junction"], "")).strip().lower() in ["roundabout", "circular"]:
        return True

    plain = str(first_existing(row, ["other_tags", "OTHER_TAGS", "TAGS", "tags"], "") or "").strip().lower()
    # Apply hint keywords only to custom/plain labels, not to parsed hstore keys
    # unless the explicit oneway field above was absent.
    hints = [h.strip().lower() for h in str(ROAD_ONEWAY_HINT_KEYWORDS or "").split(',') if h.strip()]
    if plain and hints:
        for h in hints:
            if h in plain:
                return True

    try:
        max_w = float(ROAD_NARROW_ONEWAY_MAX_WIDTH)
    except Exception:
        max_w = 0.0
    if max_w > 0.0 and float(road_width) > 0.0 and float(road_width) <= max_w:
        return True

    return bool(ROAD_DEFAULT_ONEWAY)


def parse_spawn_type(v) -> int:
    """EN: Parse SPWNTYPE into origin/destination role bits.

    Accepted examples:
      origin: O, ORIGIN, START, SOURCE, FROM, 출발, 출발지, 1
      destination: D, DEST, DESTINATION, SINK, TO, 도착, 도착지, 2
      both: B, BOTH, OD, ALL, 둘다, 양쪽, 3

    KO: SPWNTYPE 값을 출발/도착 역할 bit로 변환합니다. 값이 없으면 기존과
    호환되도록 출발지이자 도착지로 처리합니다.
    """
    if is_missing_value(v):
        return SPWNTYPE_BOTH
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        if iv == SPWNTYPE_ORIGIN:
            return SPWNTYPE_ORIGIN
        if iv == SPWNTYPE_DESTINATION:
            return SPWNTYPE_DESTINATION
        if iv == SPWNTYPE_BOTH or iv == 0:
            return SPWNTYPE_BOTH
    try:
        fv = float(v)
        if math.isfinite(fv) and abs(fv - round(fv)) < 1.0e-6:
            return parse_spawn_type(int(round(fv)))
    except Exception:
        pass

    s = str(v).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in ["", "none", "null", "nan"]:
        return SPWNTYPE_BOTH
    if s in ["o", "origin", "orig", "start", "source", "src", "from", "departure", "depart", "spawn", "only_origin", "origin_only", "출발", "출발지", "시작", "시작점"]:
        return SPWNTYPE_ORIGIN
    if s in ["d", "dest", "destination", "arrival", "arrive", "sink", "to", "end", "target", "only_dest", "dest_only", "destination_only", "도착", "도착지", "종점", "목적지"]:
        return SPWNTYPE_DESTINATION
    if s in ["b", "both", "all", "any", "od", "origin_dest", "origin_destination", "bi", "bidirectional", "양쪽", "양방향", "둘다", "모두", "출발도착"]:
        return SPWNTYPE_BOTH
    return SPWNTYPE_BOTH


def spawn_type_label(bits: int) -> str:
    bits = int(bits)
    if bits == SPWNTYPE_ORIGIN:
        return "origin"
    if bits == SPWNTYPE_DESTINATION:
        return "destination"
    if bits == SPWNTYPE_BOTH:
        return "both"
    return f"unknown({bits})"


def _spawn_profile_column_map(columns):
    """EN: Detect SPWNxx hourly profile fields and map them to slot indices.

    If SPWN00 exists, suffixes are treated as 0-based hours.  Otherwise
    SPWN01..SPWN24 are treated as 1-based slots and mapped to 0..23.

    KO: SPWNxx 시간대 필드를 찾아 slot index로 매핑합니다. SPWN00이 있으면
    0-based, 없으면 SPWN01..SPWN24를 1-based slot으로 봅니다.
    """
    slots = max(1, int(SPAWN_PROFILE_SLOTS))
    prefix = str(SPAWN_PROFILE_FIELD_PREFIX or "SPWN")
    pattern = re.compile(rf"^{re.escape(prefix)}[_-]?(\d{{1,2}})$", re.IGNORECASE)
    raw = []
    for col in columns:
        m = pattern.match(str(col))
        if not m:
            continue
        try:
            raw.append((col, int(m.group(1))))
        except Exception:
            continue
    if not raw:
        return []

    nums = [n for _, n in raw]
    zero_based = 0 in nums
    mapped = []
    seen = set()
    for col, n in raw:
        idx = n if zero_based else n - 1
        if 0 <= idx < slots and idx not in seen:
            mapped.append((col, int(idx)))
            seen.add(int(idx))
    mapped.sort(key=lambda x: x[1])
    return mapped


def _spawn_attr_value_to_vps(v):
    if is_missing_value(v):
        return math.nan
    try:
        val = float(v)
    except Exception:
        return math.nan
    if not math.isfinite(val) or val < 0.0:
        return math.nan

    unit = str(SPAWN_PROFILE_UNIT or "vph").strip().lower()
    if unit in ["vps", "veh/s", "veh/sec", "vehicle/s", "vehicles/s", "per_s", "per_sec", "초"]:
        return val
    if unit in ["vpm", "veh/min", "vehicle/min", "vehicles/min", "per_min", "분"]:
        return val / 60.0
    if unit in ["auto", "자동"]:
        # EN/KO: Small decimal rates are likely already vehicles/sec; large counts are likely hourly volumes.
        return val if val <= 10.0 else val / 3600.0
    # EN/KO: Default attribute unit is vehicles/hour.
    return val / 3600.0


def _fill_spawn_profile_circular(values):
    """Fill missing hourly slots by circular linear interpolation."""
    arr = np.asarray(values, dtype=np.float32).copy()
    n = int(arr.size)
    if n <= 0:
        return None

    valid = np.where(np.isfinite(arr) & (arr >= 0.0))[0]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        arr[:] = float(arr[int(valid[0])])
        return arr.astype(np.float32, copy=False)

    out = arr.copy()
    for i in range(n):
        if np.isfinite(out[i]) and out[i] >= 0.0:
            continue
        prev_candidates = valid[valid <= i]
        prev_i = int(prev_candidates[-1]) if len(prev_candidates) else int(valid[-1])
        next_candidates = valid[valid >= i]
        next_i = int(next_candidates[0]) if len(next_candidates) else int(valid[0])

        total = (next_i - prev_i) % n
        if total == 0:
            out[i] = float(arr[prev_i])
            continue
        dist = (i - prev_i) % n
        t = float(dist) / float(total)
        out[i] = float(arr[prev_i]) * (1.0 - t) + float(arr[next_i]) * t

    out = np.maximum(out, 0.0)
    return out.astype(np.float32, copy=False)


def spawn_profile_from_row(row):
    col_map = _spawn_profile_column_map(getattr(row, "index", []))
    if not col_map:
        return None
    slots = max(1, int(SPAWN_PROFILE_SLOTS))
    vals = np.full(slots, np.nan, dtype=np.float32)
    for col, idx in col_map:
        try:
            vals[int(idx)] = _spawn_attr_value_to_vps(row[col])
        except Exception:
            pass
    return _fill_spawn_profile_circular(vals)


def interpolate_spawn_demand_np(base_vps, profile_vps, profile_has, current_time, slot_seconds):
    """EN/KO: CPU mirror of the CUDA hourly spawn interpolation, used for UI metrics."""
    base = np.asarray(base_vps, dtype=np.float32)
    if profile_vps is None or profile_has is None:
        return base
    prof = np.asarray(profile_vps, dtype=np.float32)
    has = np.asarray(profile_has, dtype=np.int32).astype(bool)
    if prof.ndim != 2 or prof.shape[0] != base.shape[0] or prof.shape[1] <= 0 or not np.any(has):
        return base
    n = int(prof.shape[1])
    period = max(1.0e-6, float(slot_seconds))
    cycle = period * float(n)
    local = float(current_time) % cycle if cycle > 0.0 else 0.0
    idxf = local / period
    i0 = int(math.floor(idxf)) % n
    i1 = (i0 + 1) % n
    frac = float(idxf - math.floor(idxf))
    out = base.copy()
    out[has] = prof[has, i0] * (1.0 - frac) + prof[has, i1] * frac
    out[~np.isfinite(out)] = base[~np.isfinite(out)]
    return np.maximum(out, 0.0).astype(np.float32, copy=False)


def first_existing(row, names, default=None):
    for name in names:
        if name not in row:
            continue
        v = row[name]
        if v is None:
            continue
        try:
            if np.isnan(v):
                continue
        except Exception:
            pass
        return v
    return default


def ensure_parent(path):
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)


def backup_file_if_exists(path, label="RouteCache"):
    if not path or not os.path.exists(path):
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{ts}"
    try:
        ensure_parent(backup_path)
        shutil.copy2(path, backup_path)
        print(f"[{label}] backup:", backup_path)
        return backup_path
    except Exception as e:
        print(f"[{label}] backup failed:", e)
        return None


def as_contig_i32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int32))


def as_contig_f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

# ============================================================
# Research section/cell outputs
# ============================================================

def section_length_label_value(section_length_m=None):
    try:
        v = float(SECTION_LENGTH_M if section_length_m is None else section_length_m)
    except Exception:
        v = -1.0
    if v < 0.0:
        return "network"
    if abs(v - round(v)) < 1.0e-6:
        return f"{int(round(v))}m"
    return f"{v:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "m"


def expand_output_path(path_like, section_length_m=None):
    label = section_length_label_value(section_length_m)
    try:
        value = str(path_like).format(
            section_length_label=label,
            section_length=label,
            section_len=label,
            section_length_m=("network" if float(section_length_m if section_length_m is not None else SECTION_LENGTH_M) < 0.0 else f"{float(section_length_m if section_length_m is not None else SECTION_LENGTH_M):.3f}".rstrip("0").rstrip(".")),
            scenario_id=str(SCENARIO_ID),
        )
    except Exception:
        value = str(path_like)
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p


def replace_output_file(path_like, label="Output", backup=True):
    p = expand_output_path(path_like)
    ensure_parent(str(p))
    if p.exists():
        if backup:
            backup_file_if_exists(str(p), label=label)
        try:
            p.unlink()
        except Exception as e:
            print(f"[{label}] could not remove old file {p}: {e}")
    return p


def line_subsegment(line, start_m, end_m):
    try:
        length = max(float(line.length), 0.0)
        a = max(0.0, min(float(start_m), length))
        b = max(0.0, min(float(end_m), length))
        if b < a:
            a, b = b, a
        p0 = line.interpolate(a)
        p1 = line.interpolate(b)
        if p0.distance(p1) < 1.0e-9:
            eps = min(max(length * 1.0e-6, 0.01), max(length, 0.01))
            p1 = line.interpolate(min(length, a + eps))
        return LineString([(float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))])
    except Exception:
        try:
            coords = list(line.coords)
            if len(coords) >= 2:
                return LineString([coords[0], coords[-1]])
        except Exception:
            pass
        return LineString([(0.0, 0.0), (0.01, 0.0)])


def build_analysis_sections(links, lanes, network_crs, section_length_m):
    """Build directed-link sections used for V(x,t), K(x,t), Q(x,t) and congestion outputs."""
    lane_to_link = np.full(len(lanes), -1, dtype=np.int32)
    for lane in lanes:
        lid = int(lane.get("lane_id", -1))
        if 0 <= lid < len(lanes):
            lane_to_link[lid] = int(lane.get("link_id", -1))

    link_first = np.zeros(len(links), dtype=np.int32)
    link_count = np.ones(len(links), dtype=np.int32)
    records = []
    geoms = []

    if float(section_length_m) < 0.0:
        total_len = 0.0
        lane_meters = 0.0
        speed_weight = 0.0
        line_geoms = []
        for link in links:
            length = max(0.0, float(link.get("length", 0.0)))
            lanes_here = max(1.0, float(link.get("lane_count", 1)))
            total_len += length
            lane_meters += length * lanes_here
            speed_weight += length * lanes_here * max(1.0, float(link.get("speed_mps", 1.0)))
            geom = link.get("geometry")
            if geom is not None and not geom.is_empty:
                line_geoms.append(geom)
        freeflow = speed_weight / max(lane_meters, 1.0e-6)
        if SECTION_FREEFLOW_SPEED_MPS > 0.0:
            freeflow = float(SECTION_FREEFLOW_SPEED_MPS)
        geom = MultiLineString(line_geoms) if line_geoms else LineString([(0.0, 0.0), (0.01, 0.0)])
        records.append({
            "section_id": 0, "link_id": -1, "link_seg": -1,
            "start_m": 0.0, "end_m": float(total_len), "length_m": float(max(total_len, 1.0e-6)),
            "lane_count": float(max(1.0, lane_meters / max(total_len, 1.0e-6))),
            "lane_meters": float(max(lane_meters, 1.0e-6)),
            "freeflow_mps": float(max(freeflow, 1.0)), "freeflow_kmh": float(max(freeflow, 1.0) * 3.6),
            "source_row": -1, "source_segment": -1, "direction": 0,
        })
        geoms.append(geom)
    else:
        cell_len = max(1.0, float(section_length_m))
        link_first[:] = -1
        link_count[:] = 0
        sid = 0
        for link in links:
            link_id = int(link.get("link_id", sid))
            length = max(0.0, float(link.get("length", 0.0)))
            nseg = max(1, int(math.ceil(max(length, 1.0e-6) / cell_len)))
            if 0 <= link_id < len(link_first):
                link_first[link_id] = sid
                link_count[link_id] = nseg
            lane_count = max(1, int(link.get("lane_count", 1)))
            geom = link.get("geometry")
            if geom is None or geom.is_empty:
                geom = LineString([(0.0, 0.0), (max(length, 0.01), 0.0)])
            ff = float(link.get("speed_mps", 0.0))
            if SECTION_FREEFLOW_SPEED_MPS > 0.0:
                ff = float(SECTION_FREEFLOW_SPEED_MPS)
            ff = max(ff, 1.0)
            for k in range(nseg):
                a = float(k) * cell_len
                b = min(float(k + 1) * cell_len, max(length, 1.0e-6))
                seg_len = max(b - a, 1.0e-6)
                records.append({
                    "section_id": sid, "link_id": link_id, "link_seg": int(k),
                    "start_m": float(a), "end_m": float(b), "length_m": float(seg_len),
                    "lane_count": float(lane_count), "lane_meters": float(seg_len * lane_count),
                    "freeflow_mps": float(ff), "freeflow_kmh": float(ff * 3.6),
                    "source_row": int(link.get("source_row", -1)),
                    "source_segment": int(link.get("source_segment", -1)),
                    "direction": int(link.get("direction", 0)),
                })
                geoms.append(line_subsegment(geom, a, b))
                sid += 1

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=network_crs)
    gdf["section_len_cfg"] = float(section_length_m)
    return {
        "gdf": gdf,
        "lane_to_link": lane_to_link,
        "link_first_section": link_first,
        "link_section_count": link_count,
        "section_length": gdf["length_m"].to_numpy(dtype=np.float32),
        "section_lane_count": gdf["lane_count"].to_numpy(dtype=np.float32),
        "section_lane_meters": gdf["lane_meters"].to_numpy(dtype=np.float32),
        "section_freeflow": gdf["freeflow_mps"].to_numpy(dtype=np.float32),
        "section_link_id": gdf["link_id"].to_numpy(dtype=np.int32),
        "section_link_seg": gdf["link_seg"].to_numpy(dtype=np.int32),
        "section_start_m": gdf["start_m"].to_numpy(dtype=np.float32),
        "section_end_m": gdf["end_m"].to_numpy(dtype=np.float32),
        "whole_network": bool(float(section_length_m) < 0.0),
    }


def clean_number(v, ndigits=6):
    try:
        x = float(v)
        if not math.isfinite(x):
            return ""
        return round(x, ndigits)
    except Exception:
        return ""


class SectionStatsRecorder:
    FIELDNAMES = [
        "scenario_id", "seed", "step", "time_bin_start", "time_bin_end",
        "section_id", "link_id", "link_seg", "start_m", "end_m", "length_m", "lane_count",
        "vehicle_count", "flow_count", "flow_vph",
        "density_veh_per_km", "density_veh_per_lane_km",
        "mean_speed_kmh", "harmonic_speed_kmh", "speed_std_kmh",
        "mean_accel_mps2", "accel_std_mps2", "occupancy",
        "queue_vehicle_count", "queue_length_m", "delay_vehicle_seconds",
        "av_count", "hdv_count", "local_av_penetration",
        "mean_speed_av_kmh", "mean_speed_hdv_kmh", "delta_speed_av_hdv_kmh",
        "congestion", "congestion_intensity", "congestion_area_m_s", "congestion_duration_s",
        "hard_brake_count", "stop_count", "stop_go_events", "go_stop_events",
    ]

    def __init__(self, section_info, max_agents, enabled=True):
        self.enabled = bool(enabled)
        self.info = section_info
        self.sections_gdf = section_info["gdf"].copy()
        self.nsec = int(len(self.sections_gdf))
        self.max_agents = int(max_agents)
        self.csv_path = expand_output_path(SECTION_STATS_CSV, SECTION_LENGTH_M)
        self.summary_path = expand_output_path(SECTION_SUMMARY_CSV, SECTION_LENGTH_M)
        self.base_gpkg_path = expand_output_path(SECTION_BASE_GPKG_PATH, SECTION_LENGTH_M)
        self.final_gpkg_path = expand_output_path(SECTION_FINAL_GPKG_PATH, SECTION_LENGTH_M)
        self.csv_file = None
        self.csv_writer = None
        self.last_sample_time = None
        self.time_rows_for_gpkg = []
        self.time_gpkg_truncated = False
        self.reset_accumulators()
        if not self.enabled:
            print("[SectionStats] disabled")
            return
        self.open_csv(reset_file=True)
        if SECTION_WRITE_BASE_GPKG:
            self.write_base_gpkg()
        print("[SectionStats] enabled:", self.nsec, "sections", "interval_s:", SECTION_STATS_INTERVAL)

    def reset_accumulators(self):
        n = self.nsec
        self.last_vehicle_section = np.full(self.max_agents, -1, dtype=np.int32)
        self.last_vehicle_motion_state = np.zeros(self.max_agents, dtype=np.int8)
        self.samples = np.zeros(n, dtype=np.float64)
        self.nonempty_samples = np.zeros(n, dtype=np.float64)
        self.vehicle_count_sum = np.zeros(n, dtype=np.float64)
        self.flow_count_total = np.zeros(n, dtype=np.float64)
        self.flow_vph_sum = np.zeros(n, dtype=np.float64)
        self.density_sum = np.zeros(n, dtype=np.float64)
        self.density_lane_sum = np.zeros(n, dtype=np.float64)
        self.speed_sum = np.zeros(n, dtype=np.float64)
        self.speed_count = np.zeros(n, dtype=np.float64)
        self.mean_speed_sum = np.zeros(n, dtype=np.float64)
        self.mean_speed_samples = np.zeros(n, dtype=np.float64)
        self.accel_sum = np.zeros(n, dtype=np.float64)
        self.accel_count = np.zeros(n, dtype=np.float64)
        self.occupancy_sum = np.zeros(n, dtype=np.float64)
        self.queue_max = np.zeros(n, dtype=np.float64)
        self.queue_sum = np.zeros(n, dtype=np.float64)
        self.delay_total = np.zeros(n, dtype=np.float64)
        self.av_count_sum = np.zeros(n, dtype=np.float64)
        self.hdv_count_sum = np.zeros(n, dtype=np.float64)
        self.congestion_duration_total = np.zeros(n, dtype=np.float64)
        self.congestion_area_total = np.zeros(n, dtype=np.float64)
        self.congestion_intensity_sum = np.zeros(n, dtype=np.float64)
        self.hard_brake_total = np.zeros(n, dtype=np.float64)
        self.stop_count_sum = np.zeros(n, dtype=np.float64)
        self.stop_go_total = np.zeros(n, dtype=np.float64)
        self.go_stop_total = np.zeros(n, dtype=np.float64)
        self.last_sample_time = None

    def open_csv(self, reset_file=False):
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
        if reset_file:
            replace_output_file(self.csv_path, label="SectionStatsCSV", backup=SECTION_BACKUP_OUTPUTS)
        else:
            ensure_parent(str(self.csv_path))
        self.csv_file = self.csv_path.open("w", encoding="utf-8-sig", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.FIELDNAMES)
        self.csv_writer.writeheader()

    def reset(self):
        if not self.enabled:
            return
        self.reset_accumulators()
        self.time_rows_for_gpkg.clear()
        self.time_gpkg_truncated = False
        self.open_csv(reset_file=True)
        print("[SectionStats] reset")

    def write_base_gpkg(self):
        try:
            path = replace_output_file(self.base_gpkg_path, label="SectionBaseGPKG", backup=SECTION_BACKUP_OUTPUTS)
            gdf = self.sections_gdf.copy()
            gdf["scenario_id"] = str(SCENARIO_ID)
            gdf["section_tag"] = section_length_label_value(SECTION_LENGTH_M)
            gdf.to_file(path, layer=str(SECTION_BASE_LAYER), driver="GPKG")
            print("[SectionStats] base GPKG:", path)
        except Exception as e:
            print("[SectionStats] base GPKG write failed:", e)

    def vehicle_sections(self, lane_ids, s_vals):
        lane_ids = np.asarray(lane_ids, dtype=np.int32)
        s_vals = np.asarray(s_vals, dtype=np.float32)
        out = np.full(lane_ids.shape, -1, dtype=np.int32)
        valid_lane = (lane_ids >= 0) & (lane_ids < len(self.info["lane_to_link"]))
        if not np.any(valid_lane):
            return out
        if self.info["whole_network"]:
            out[valid_lane] = 0
            return out
        link_ids = self.info["lane_to_link"][lane_ids[valid_lane]]
        valid_link = (link_ids >= 0) & (link_ids < len(self.info["link_first_section"]))
        idx_base = np.where(valid_lane)[0]
        if not np.any(valid_link):
            return out
        idx = idx_base[valid_link]
        link_ids = link_ids[valid_link]
        first = self.info["link_first_section"][link_ids]
        count = self.info["link_section_count"][link_ids]
        ok = (first >= 0) & (count > 0)
        if not np.any(ok):
            return out
        idx = idx[ok]
        first = first[ok]
        count = count[ok]
        local = np.floor(np.maximum(s_vals[idx], 0.0) / max(1.0, float(SECTION_LENGTH_M))).astype(np.int32)
        local = np.minimum(np.maximum(local, 0), count - 1)
        out[idx] = first + local
        return out

    def maybe_sample(self, current_time, step, veh, force=False):
        if not self.enabled or veh is None:
            return False
        current_time = float(current_time)
        interval_cfg = max(1.0e-6, float(SECTION_STATS_INTERVAL))
        if self.last_sample_time is not None:
            if not force and current_time < self.last_sample_time + interval_cfg - 1.0e-9:
                return False
            if force and abs(current_time - self.last_sample_time) < 1.0e-9:
                return False
            interval = max(1.0e-6, current_time - self.last_sample_time)
        else:
            if not force and current_time < interval_cfg - 1.0e-9:
                return False
            interval = max(1.0e-6, current_time if current_time > 0.0 else interval_cfg)

        try:
            active = veh["active"].detach().cpu().numpy().astype(np.int32, copy=False)
            active_ids = np.flatnonzero(active != 0)
            lane_all = veh["lane_id"].detach().cpu().numpy().astype(np.int32, copy=False)
            s_all = veh["s"].detach().cpu().numpy().astype(np.float32, copy=False)
            speed_all = veh["speed"].detach().cpu().numpy().astype(np.float32, copy=False)
            accel_all = veh["accel"].detach().cpu().numpy().astype(np.float32, copy=False)
            dtype_all = veh["driver_type"].detach().cpu().numpy().astype(np.int32, copy=False)
            length_all = veh["vehicle_length"].detach().cpu().numpy().astype(np.float32, copy=False)
            vehicle_state = veh["vehicle_state"].detach().cpu().numpy().astype(np.int32, copy=False)
            connector_to = veh["connector_to_lane"].detach().cpu().numpy().astype(np.int32, copy=False)
        except Exception as e:
            print("[SectionStats] tensor copy failed:", e)
            return False

        n = self.nsec
        count = np.zeros(n, dtype=np.float64)
        flow_count = np.zeros(n, dtype=np.float64)
        sum_speed = np.zeros(n, dtype=np.float64)
        sum_speed2 = np.zeros(n, dtype=np.float64)
        sum_inv_speed = np.zeros(n, dtype=np.float64)
        sum_accel = np.zeros(n, dtype=np.float64)
        sum_accel2 = np.zeros(n, dtype=np.float64)
        sum_length = np.zeros(n, dtype=np.float64)
        av_count = np.zeros(n, dtype=np.float64)
        hdv_count = np.zeros(n, dtype=np.float64)
        sum_speed_av = np.zeros(n, dtype=np.float64)
        sum_speed_hdv = np.zeros(n, dtype=np.float64)
        queue_count = np.zeros(n, dtype=np.float64)
        hard_brake_count = np.zeros(n, dtype=np.float64)
        stop_count = np.zeros(n, dtype=np.float64)
        stop_go_events = np.zeros(n, dtype=np.float64)
        go_stop_events = np.zeros(n, dtype=np.float64)
        new_last_section = np.full(self.max_agents, -1, dtype=np.int32)
        new_motion_state = np.zeros(self.max_agents, dtype=np.int8)

        if len(active_ids) > 0:
            lane_active = lane_all[active_ids].copy()
            s_active = s_all[active_ids].astype(np.float32, copy=True)
            conn_mask = (vehicle_state[active_ids] == VEH_IN_CONNECTOR) & (connector_to[active_ids] >= 0)
            lane_active[conn_mask] = connector_to[active_ids][conn_mask]
            # Connector state uses a curved handoff path, not the downstream lane's own s-coordinate.
            # For section aggregation, map connector vehicles to the first cell of their downstream lane
            # so long upstream s-values do not get mis-binned near the end of the downstream link.
            s_active[conn_mask] = 0.0
            sec_all = self.vehicle_sections(lane_active, s_active)
            valid = (sec_all >= 0) & (sec_all < n)
            if np.any(valid):
                ids = active_ids[valid]
                sec = sec_all[valid]
                spd = np.maximum(speed_all[ids].astype(np.float64), 0.0)
                acc = accel_all[ids].astype(np.float64)
                dtype = dtype_all[ids]
                vlen = np.maximum(length_all[ids].astype(np.float64), 0.1)
                count = np.bincount(sec, minlength=n).astype(np.float64)
                sum_speed = np.bincount(sec, weights=spd, minlength=n).astype(np.float64)
                sum_speed2 = np.bincount(sec, weights=spd * spd, minlength=n).astype(np.float64)
                sum_inv_speed = np.bincount(sec, weights=1.0 / np.maximum(spd, 0.1), minlength=n).astype(np.float64)
                sum_accel = np.bincount(sec, weights=acc, minlength=n).astype(np.float64)
                sum_accel2 = np.bincount(sec, weights=acc * acc, minlength=n).astype(np.float64)
                sum_length = np.bincount(sec, weights=vlen, minlength=n).astype(np.float64)
                is_av = dtype == 1
                av_count = np.bincount(sec, weights=is_av.astype(np.float64), minlength=n).astype(np.float64)
                hdv_count = count - av_count
                if np.any(is_av):
                    sum_speed_av = np.bincount(sec[is_av], weights=spd[is_av], minlength=n).astype(np.float64)
                if np.any(~is_av):
                    sum_speed_hdv = np.bincount(sec[~is_av], weights=spd[~is_av], minlength=n).astype(np.float64)
                qmask = spd < max(0.0, float(SECTION_QUEUE_SPEED_KMH)) / 3.6
                smask = spd < max(0.0, float(SECTION_STOP_SPEED_KMH)) / 3.6
                hbmask = acc <= float(SECTION_HARD_BRAKE_MPS2)
                if np.any(qmask):
                    queue_count = np.bincount(sec[qmask], minlength=n).astype(np.float64)
                if np.any(smask):
                    stop_count = np.bincount(sec[smask], minlength=n).astype(np.float64)
                if np.any(hbmask):
                    hard_brake_count = np.bincount(sec[hbmask], minlength=n).astype(np.float64)
                prev_sec = self.last_vehicle_section[ids]
                entered = sec != prev_sec
                if np.any(entered):
                    flow_count = np.bincount(sec[entered], minlength=n).astype(np.float64)
                prev_state = self.last_vehicle_motion_state[ids]
                curr_state = prev_state.copy()
                curr_state[spd < max(0.0, float(SECTION_STOP_SPEED_KMH)) / 3.6] = 1
                curr_state[spd > max(0.0, float(SECTION_GO_SPEED_KMH)) / 3.6] = 2
                sg = (prev_state == 1) & (curr_state == 2)
                gs = (prev_state == 2) & (curr_state == 1)
                if np.any(sg):
                    stop_go_events = np.bincount(sec[sg], minlength=n).astype(np.float64)
                if np.any(gs):
                    go_stop_events = np.bincount(sec[gs], minlength=n).astype(np.float64)
                new_last_section[ids] = sec
                new_motion_state[ids] = curr_state

        self.last_vehicle_section = new_last_section
        self.last_vehicle_motion_state = new_motion_state

        length_m = np.maximum(self.info["section_length"].astype(np.float64), 1.0e-6)
        lane_count_arr = np.maximum(self.info["section_lane_count"].astype(np.float64), 1.0)
        lane_meters = np.maximum(self.info["section_lane_meters"].astype(np.float64), 1.0e-6)
        freeflow = np.maximum(self.info["section_freeflow"].astype(np.float64), 1.0)
        nonzero = count > 0
        mean_speed = np.full(n, np.nan, dtype=np.float64)
        harmonic_speed = np.full(n, np.nan, dtype=np.float64)
        speed_std = np.full(n, np.nan, dtype=np.float64)
        mean_accel = np.full(n, np.nan, dtype=np.float64)
        accel_std = np.full(n, np.nan, dtype=np.float64)
        mean_speed_av = np.full(n, np.nan, dtype=np.float64)
        mean_speed_hdv = np.full(n, np.nan, dtype=np.float64)
        delta_av_hdv = np.full(n, np.nan, dtype=np.float64)
        mean_speed[nonzero] = sum_speed[nonzero] / count[nonzero]
        harmonic_speed[nonzero] = count[nonzero] / np.maximum(sum_inv_speed[nonzero], 1.0e-9)
        speed_std[nonzero] = np.sqrt(np.maximum(0.0, sum_speed2[nonzero] / count[nonzero] - mean_speed[nonzero] ** 2))
        mean_accel[nonzero] = sum_accel[nonzero] / count[nonzero]
        accel_std[nonzero] = np.sqrt(np.maximum(0.0, sum_accel2[nonzero] / count[nonzero] - mean_accel[nonzero] ** 2))
        avnz = av_count > 0
        hdvnz = hdv_count > 0
        mean_speed_av[avnz] = sum_speed_av[avnz] / av_count[avnz]
        mean_speed_hdv[hdvnz] = sum_speed_hdv[hdvnz] / hdv_count[hdvnz]
        both = avnz & hdvnz
        delta_av_hdv[both] = mean_speed_av[both] - mean_speed_hdv[both]

        density = count / np.maximum(length_m / 1000.0, 1.0e-9)
        density_lane = count / np.maximum(lane_meters / 1000.0, 1.0e-9)
        flow_vph = flow_count * 3600.0 / max(interval, 1.0e-6)
        occupancy = np.minimum(1.0, (sum_length + count * float(SECTION_OCCUPANCY_EFFECTIVE_VEHICLE_LENGTH_M)) / lane_meters)
        queue_length = np.minimum(length_m, queue_count * float(SECTION_OCCUPANCY_EFFECTIVE_VEHICLE_LENGTH_M) / lane_count_arr)
        delay = np.zeros(n, dtype=np.float64)
        if len(active_ids) > 0:
            lane_active = lane_all[active_ids].copy()
            s_active = s_all[active_ids].astype(np.float32, copy=True)
            conn_mask = (vehicle_state[active_ids] == VEH_IN_CONNECTOR) & (connector_to[active_ids] >= 0)
            lane_active[conn_mask] = connector_to[active_ids][conn_mask]
            # Connector state uses a curved handoff path, not the downstream lane's own s-coordinate.
            # For section aggregation, map connector vehicles to the first cell of their downstream lane
            # so long upstream s-values do not get mis-binned near the end of the downstream link.
            s_active[conn_mask] = 0.0
            sec_all = self.vehicle_sections(lane_active, s_active)
            valid = (sec_all >= 0) & (sec_all < n)
            if np.any(valid):
                ids = active_ids[valid]
                sec = sec_all[valid]
                spd = np.maximum(speed_all[ids].astype(np.float64), 0.0)
                weights = np.maximum(0.0, 1.0 - spd / np.maximum(freeflow[sec], 1.0e-6)) * interval
                delay = np.bincount(sec, weights=weights, minlength=n).astype(np.float64)
        threshold = freeflow * max(0.0, float(SECTION_CONGESTION_SPEED_RATIO))
        congestion = np.zeros(n, dtype=np.int32)
        congestion[nonzero] = (mean_speed[nonzero] < threshold[nonzero]).astype(np.int32)
        intensity = np.zeros(n, dtype=np.float64)
        intensity[nonzero] = np.maximum(0.0, 1.0 - mean_speed[nonzero] / np.maximum(freeflow[nonzero], 1.0e-6))
        congestion_duration = congestion.astype(np.float64) * interval
        congestion_area = congestion.astype(np.float64) * length_m * interval
        av_pen = np.full(n, np.nan, dtype=np.float64)
        av_pen[nonzero] = av_count[nonzero] / np.maximum(count[nonzero], 1.0)

        row_ids = range(n) if SECTION_WRITE_EMPTY_ROWS else np.flatnonzero(nonzero | (flow_count > 0) | (congestion > 0))
        t0 = max(0.0, current_time - interval)
        for sid in row_ids:
            row = {
                "scenario_id": str(SCENARIO_ID), "seed": int(SCENARIO_SEED), "step": int(step),
                "time_bin_start": round(float(t0), 6), "time_bin_end": round(float(current_time), 6),
                "section_id": int(sid), "link_id": int(self.info["section_link_id"][sid]), "link_seg": int(self.info["section_link_seg"][sid]),
                "start_m": clean_number(self.info["section_start_m"][sid], 3), "end_m": clean_number(self.info["section_end_m"][sid], 3),
                "length_m": clean_number(length_m[sid], 3), "lane_count": clean_number(lane_count_arr[sid], 3),
                "vehicle_count": int(count[sid]), "flow_count": int(flow_count[sid]), "flow_vph": clean_number(flow_vph[sid]),
                "density_veh_per_km": clean_number(density[sid]), "density_veh_per_lane_km": clean_number(density_lane[sid]),
                "mean_speed_kmh": clean_number(mean_speed[sid] * 3.6), "harmonic_speed_kmh": clean_number(harmonic_speed[sid] * 3.6),
                "speed_std_kmh": clean_number(speed_std[sid] * 3.6), "mean_accel_mps2": clean_number(mean_accel[sid]),
                "accel_std_mps2": clean_number(accel_std[sid]), "occupancy": clean_number(occupancy[sid]),
                "queue_vehicle_count": int(queue_count[sid]), "queue_length_m": clean_number(queue_length[sid]),
                "delay_vehicle_seconds": clean_number(delay[sid]), "av_count": int(av_count[sid]), "hdv_count": int(hdv_count[sid]),
                "local_av_penetration": clean_number(av_pen[sid]), "mean_speed_av_kmh": clean_number(mean_speed_av[sid] * 3.6),
                "mean_speed_hdv_kmh": clean_number(mean_speed_hdv[sid] * 3.6), "delta_speed_av_hdv_kmh": clean_number(delta_av_hdv[sid] * 3.6),
                "congestion": int(congestion[sid]), "congestion_intensity": clean_number(intensity[sid]),
                "congestion_area_m_s": clean_number(congestion_area[sid]), "congestion_duration_s": clean_number(congestion_duration[sid]),
                "hard_brake_count": int(hard_brake_count[sid]), "stop_count": int(stop_count[sid]),
                "stop_go_events": int(stop_go_events[sid]), "go_stop_events": int(go_stop_events[sid]),
            }
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
            if SECTION_WRITE_TIME_GPKG and not self.time_gpkg_truncated:
                if len(self.time_rows_for_gpkg) < max(0, int(SECTION_MAX_TIME_GPKG_ROWS)):
                    self.time_rows_for_gpkg.append(dict(row))
                else:
                    self.time_rows_for_gpkg.clear()
                    self.time_gpkg_truncated = True
                    print("[SectionStats] section_time_bins GPKG layer disabled by row limit; CSV still has all rows.")
        if self.csv_file is not None:
            self.csv_file.flush()

        self.samples += 1.0
        self.nonempty_samples += nonzero.astype(np.float64)
        self.vehicle_count_sum += count
        self.flow_count_total += flow_count
        self.flow_vph_sum += flow_vph
        self.density_sum += density
        self.density_lane_sum += density_lane
        self.speed_sum += sum_speed
        self.speed_count += count
        self.mean_speed_sum[nonzero] += mean_speed[nonzero] * 3.6
        self.mean_speed_samples[nonzero] += 1.0
        self.accel_sum += sum_accel
        self.accel_count += count
        self.occupancy_sum += occupancy
        self.queue_max = np.maximum(self.queue_max, queue_length)
        self.queue_sum += queue_length
        self.delay_total += delay
        self.av_count_sum += av_count
        self.hdv_count_sum += hdv_count
        self.congestion_duration_total += congestion_duration
        self.congestion_area_total += congestion_area
        self.congestion_intensity_sum += intensity
        self.hard_brake_total += hard_brake_count
        self.stop_count_sum += stop_count
        self.stop_go_total += stop_go_events
        self.go_stop_total += go_stop_events
        self.last_sample_time = current_time
        return True

    def finalize(self, current_time=0.0, step=0, veh=None, metrics_snapshot=None, exit_reason="normal"):
        if not self.enabled:
            return
        try:
            self.maybe_sample(float(current_time), int(step), veh, force=True)
        except Exception as e:
            print("[SectionStats] final sample failed:", e)
        try:
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()
                self.csv_file = None
        except Exception:
            pass
        self.write_final_outputs(float(current_time), int(step), metrics_snapshot or {}, exit_reason=str(exit_reason))

    def write_final_outputs(self, current_time, step, metrics_snapshot, exit_reason="normal"):
        n = self.nsec
        sample_den = np.maximum(self.samples, 1.0)
        speed_den = np.maximum(self.speed_count, 1.0)
        accel_den = np.maximum(self.accel_count, 1.0)
        nonempty_den = np.maximum(self.mean_speed_samples, 1.0)
        total_type_count = self.av_count_sum + self.hdv_count_sum

        result = self.sections_gdf.copy()
        result["scenario_id"] = str(SCENARIO_ID)
        result["seed"] = int(SCENARIO_SEED)
        result["av_pen_cfg"] = float(AV_PENETRATION)
        result["sim_time_s"] = float(current_time)
        result["samples"] = self.samples
        result["veh_avg"] = self.vehicle_count_sum / sample_den
        result["flow_total"] = self.flow_count_total
        result["flow_vph_avg"] = self.flow_vph_sum / sample_den
        result["density_avg"] = self.density_sum / sample_den
        result["density_ln"] = self.density_lane_sum / sample_den
        result["mean_spd"] = self.speed_sum / speed_den * 3.6
        result["mean_spd_bin"] = self.mean_speed_sum / nonempty_den
        result["mean_accel"] = self.accel_sum / accel_den
        result["occupancy"] = self.occupancy_sum / sample_den
        result["queue_max"] = self.queue_max
        result["queue_avg"] = self.queue_sum / sample_den
        result["delay_s"] = self.delay_total
        result["av_sum"] = self.av_count_sum
        result["hdv_sum"] = self.hdv_count_sum
        result["local_av_p"] = self.av_count_sum / np.maximum(total_type_count, 1.0)
        result["cong_dur"] = self.congestion_duration_total
        result["cong_area"] = self.congestion_area_total
        result["cong_int"] = self.congestion_intensity_sum / sample_den
        result["hard_brake"] = self.hard_brake_total
        result["stop_count"] = self.stop_count_sum
        result["stop_go"] = self.stop_go_total
        result["go_stop"] = self.go_stop_total
        width_metric = self.delay_total.copy()
        denom = float(np.nanmax(width_metric)) if len(width_metric) else 0.0
        if denom <= 1.0e-9:
            width_metric = self.congestion_duration_total.copy()
            denom = float(np.nanmax(width_metric)) if len(width_metric) else 0.0
        if denom <= 1.0e-9:
            width_metric = self.queue_max.copy()
            denom = float(np.nanmax(width_metric)) if len(width_metric) else 0.0
        result["vis_width"] = float(SECTION_VIS_WIDTH_BASE) * (1.0 + float(SECTION_VIS_WIDTH_ALPHA) * width_metric / max(denom, 1.0e-9))
        result["vis_color"] = result["cong_int"]

        final_path = replace_output_file(self.final_gpkg_path, label="SectionResultGPKG", backup=SECTION_BACKUP_OUTPUTS)
        try:
            result.to_file(final_path, layer=str(SECTION_RESULTS_LAYER), driver="GPKG")
            self.sections_gdf.to_file(final_path, layer=str(SECTION_BASE_LAYER), driver="GPKG")
            if SECTION_WRITE_TIME_GPKG and self.time_rows_for_gpkg and not self.time_gpkg_truncated:
                geom_by_sid = list(self.sections_gdf.geometry)
                geoms = [geom_by_sid[int(r["section_id"])] for r in self.time_rows_for_gpkg]
                time_gdf = gpd.GeoDataFrame(self.time_rows_for_gpkg, geometry=geoms, crs=self.sections_gdf.crs)
                time_gdf.to_file(final_path, layer=str(SECTION_TIME_LAYER), driver="GPKG")
            print("[SectionStats] final GPKG:", final_path)
        except Exception as e:
            print("[SectionStats] final GPKG write failed:", e)

        completed = float(metrics_snapshot.get("completed", 0.0) or 0.0)
        travel_sum = float(metrics_snapshot.get("travel_time_sum", 0.0) or 0.0)
        summary = {
            "scenario_id": str(SCENARIO_ID),
            "seed": int(SCENARIO_SEED),
            "exit_reason": str(exit_reason),
            "av_penetration": float(AV_PENETRATION),
            "section_length_m": float(SECTION_LENGTH_M),
            "stats_interval_s": float(SECTION_STATS_INTERVAL),
            "sim_time_s": float(current_time),
            "step": int(step),
            "sections": int(n),
            "total_delay_vehicle_seconds": float(np.nansum(self.delay_total)),
            "total_delay_vehicle_hours": float(np.nansum(self.delay_total) / 3600.0),
            "congestion_area_m_s": float(np.nansum(self.congestion_area_total)),
            "congestion_area_km_h": float(np.nansum(self.congestion_area_total) / 1000.0 / 3600.0),
            "congestion_duration_total_s": float(np.nansum(self.congestion_duration_total)),
            "max_queue_length_m": float(np.nanmax(self.queue_max)) if n else 0.0,
            "total_flow_count": float(np.nansum(self.flow_count_total)),
            "hard_brake_events": float(np.nansum(self.hard_brake_total)),
            "stop_go_events": float(np.nansum(self.stop_go_total)),
            "go_stop_events": float(np.nansum(self.go_stop_total)),
            "completed": completed,
            "spawned": float(metrics_snapshot.get("spawned", 0.0) or 0.0),
            "active_final": float(metrics_snapshot.get("active", 0.0) or 0.0),
            "mean_travel_time_s": (travel_sum / completed) if completed > 0.0 else "",
            "network_mean_speed_kmh": float(np.nansum(self.speed_sum) / max(np.nansum(self.speed_count), 1.0) * 3.6),
            "network_local_av_penetration": float(np.nansum(self.av_count_sum) / max(np.nansum(total_type_count), 1.0)),
            "section_stats_csv": str(self.csv_path),
            "section_base_gpkg": str(self.base_gpkg_path),
            "section_results_gpkg": str(final_path),
        }
        summary_path = replace_output_file(self.summary_path, label="SectionSummary", backup=SECTION_BACKUP_OUTPUTS)
        try:
            with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)
            print("[SectionStats] CSV:", self.csv_path)
            print("[SectionStats] summary:", summary_path)
        except Exception as e:
            print("[SectionStats] summary write failed:", e)


# ============================================================
# Async metrics
# ============================================================

class AsyncMetricsWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.q = queue.Queue(maxsize=1024)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, step, snap):
        try:
            self.q.put_nowait((int(step), dict(snap)))
        except queue.Full:
            pass

    def _run(self):
        header_written = False
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                step, snap = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ensure_parent(self.filepath)
                header = ",".join(["step"] + list(snap.keys()))
                mode = "a"

                if not header_written:
                    existing_ok = False
                    if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
                        try:
                            with open(self.filepath, "r", encoding="utf-8") as rf:
                                existing_ok = rf.readline().strip() == header
                        except Exception:
                            existing_ok = False
                        if not existing_ok:
                            backup_file_if_exists(self.filepath, label="Metrics")
                            mode = "w"
                    else:
                        mode = "w"

                with open(self.filepath, mode, encoding="utf-8") as f:
                    if not header_written:
                        if mode == "w":
                            f.write(header + "\n")
                        header_written = True
                    f.write(",".join([str(step)] + [str(v) for v in snap.values()]) + "\n")
            except Exception as e:
                print("[Metrics] write failed:", e)
            finally:
                self.q.task_done()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)


# ============================================================
# GPKG loading
# ============================================================

def list_gpkg_layers(path):
    if not path or not os.path.exists(path):
        return []
    try:
        import pyogrio
        return [str(x[0]) for x in pyogrio.list_layers(path)]
    except Exception:
        pass
    try:
        import fiona
        return list(fiona.listlayers(path))
    except Exception:
        return []


def read_layer(path, layer=None, required=False, prefer_geometry=None, label="GPKG"):
    prefer_geometry = prefer_geometry or []
    if not path or not os.path.exists(path):
        msg = f"[{label}] file not found: {path}"
        print(msg)
        if required:
            raise FileNotFoundError(msg)
        return gpd.GeoDataFrame(geometry=[]), None

    layers = list_gpkg_layers(path)
    print(f"[{label}] layers:", layers)

    if layer is not None and layer in layers:
        gdf = gpd.read_file(path, layer=layer)
        print(f"[{label}] loaded layer:", layer, "rows:", len(gdf), "crs:", gdf.crs)
        return gdf, layer

    if layer is not None and layer not in layers:
        print(f"[{label}] requested layer not found:", layer)

    prefer_set = set(prefer_geometry)
    for lname in layers:
        try:
            gdf = gpd.read_file(path, layer=lname)
            if len(gdf) == 0:
                continue
            if prefer_set:
                geom_types = set(gdf.geometry.geom_type.dropna().astype(str).tolist())
                if not geom_types.intersection(prefer_set):
                    continue
            print(f"[{label}] auto layer:", lname, "rows:", len(gdf), "crs:", gdf.crs)
            return gdf, lname
        except Exception as e:
            print(f"[{label}] skip layer {lname}:", e)

    if required:
        raise RuntimeError(f"[{label}] no readable layer in {path}")
    return gpd.GeoDataFrame(geometry=[]), None


def load_points(path, layer=None, label="Points"):
    gdf, used = read_layer(path, layer=layer, required=False, prefer_geometry=["Point", "MultiPoint"], label=label)
    pts = []
    if gdf is None or len(gdf) == 0:
        return pts, None
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geoms = [geom] if geom.geom_type == "Point" else list(geom.geoms) if geom.geom_type == "MultiPoint" else []
        for p in geoms:
            pts.append((float(p.x), float(p.y)))
    print(f"[{label}] loaded:", len(pts), "layer:", used)
    return pts, gdf.crs


def load_spawn_records(path, layer=None, label="Spawn"):
    """EN: Load spawn points with SPWNTYPE and SPWNxx hourly demand attributes.

    KO: 스폰 포인트의 위치와 함께 SPWNTYPE, SPWNxx 시간대별 수요 속성을 읽습니다.
    기존 load_points()는 신호등처럼 위치만 필요한 레이어를 위해 그대로 둡니다.
    """
    gdf, used = read_layer(path, layer=layer, required=False, prefer_geometry=["Point", "MultiPoint"], label=label)
    records = []
    if gdf is None or len(gdf) == 0:
        return records, None

    profile_cols = _spawn_profile_column_map(gdf.columns)
    type_fields = ["SPWNTYPE", "SPWN_TYPE", "SPAWNTYPE", "SPAWN_TYPE", "spwntype", "spawn_type", "TYPE", "type"]

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        geoms = [geom] if geom.geom_type == "Point" else list(geom.geoms) if geom.geom_type == "MultiPoint" else []
        if not geoms:
            continue

        spwn_type = parse_spawn_type(first_existing(row, type_fields, None))
        profile = spawn_profile_from_row(row) if profile_cols else None

        for p in geoms:
            records.append({
                "x": float(p.x),
                "y": float(p.y),
                "spwn_type": int(spwn_type),
                "profile_vps": None if profile is None else np.asarray(profile, dtype=np.float32).copy(),
            })

    prof_count = sum(1 for r in records if r.get("profile_vps") is not None)
    role_counts = {
        "origin": sum(1 for r in records if int(r.get("spwn_type", SPWNTYPE_BOTH)) == SPWNTYPE_ORIGIN),
        "destination": sum(1 for r in records if int(r.get("spwn_type", SPWNTYPE_BOTH)) == SPWNTYPE_DESTINATION),
        "both": sum(1 for r in records if int(r.get("spwn_type", SPWNTYPE_BOTH)) == SPWNTYPE_BOTH),
    }
    print(f"[{label}] loaded:", len(records), "layer:", used)
    print(f"[{label}] SPWNTYPE counts:", role_counts, "profile_rows:", prof_count, "profile_fields:", [str(c) for c, _ in profile_cols])
    return records, gdf.crs


# ============================================================
# Network
# ============================================================

def build_graph_and_lanes(gpkg_path, layer=None, tol=1.0):
    gdf, used_layer = read_layer(gpkg_path, layer=layer, required=True, prefer_geometry=["LineString", "MultiLineString"], label="Road")
    print("[Road] using layer:", used_layer)
    print("[Road] oneway mode:", _oneway_mode_value(), "parse_osm_other_tags:", ROAD_PARSE_OSM_OTHER_TAGS)

    node_map = {}
    nodes = []
    links = []
    lanes = []
    oneway_row_count = 0
    twoway_row_count = 0
    parsed_lanes_from_tags = 0

    def key_xy(x, y):
        return round(float(x) / tol), round(float(y) / tol)

    def get_node(x, y, spawn=False):
        k = key_xy(x, y)
        if k not in node_map:
            nid = len(nodes)
            node_map[k] = nid
            nodes.append({"node_id": nid, "geometry": Point(float(x), float(y)), "spawn": bool(spawn)})
        elif spawn:
            nodes[node_map[k]]["spawn"] = True
        return node_map[k]

    for ridx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        if not parts:
            continue

        tags = parse_osm_other_tags(first_existing(row, ["other_tags", "OTHER_TAGS", "TAGS", "tags"], None))

        width_value = first_existing_or_tag(
            row,
            ["ROAD_BT", "ROAD_WIDTH", "WIDTH", "width", "도로폭"],
            tags=tags,
            tag_names=["width"],
            default=10.5,
        )
        road_width = parse_first_number(width_value, default=10.5)
        if road_width is None or road_width <= 0.0:
            road_width = 10.5
        road_width = float(road_width)

        tag_lane_before = first_tag(tags, ["lanes", "lanes:total", "lanes:forward", "lanes:backward"], None)
        total_lanes = parse_lane_count_from_row(row, road_width, tags)
        if tag_lane_before is not None:
            parsed_lanes_from_tags += 1

        speed_value = first_existing_or_tag(
            row,
            ["MAX_SPD", "SPEED", "SPEED_KMH", "제한속도"],
            tags=tags,
            tag_names=["maxspeed", "maxspeed:forward", "maxspeed:backward"],
            default=60.0,
        )
        speed_kmh = parse_speed_kmh(speed_value, default=60.0)
        oneway = parse_road_oneway(row, tags, road_width)
        spawn_flag = parse_bool(first_existing(row, ["SPAWN", "spawn", "Spawn"], False))

        if oneway:
            oneway_row_count += 1
        else:
            twoway_row_count += 1

        # EN: Two-way roads are split into two symmetric direction bundles unless
        #     the one-way resolver says the source line already represents one
        #     carriageway.  Dongbu's OSM rows carry other_tags oneway=yes, so they
        #     stay as one-way 3/4-lane bundles instead of being duplicated into
        #     opposite-direction lanes.  Rows without explicit OSM hstore but with
        #     custom labels such as enterance/sideway/sidway are also kept one-way
        #     through ROAD_ONEWAY_HINT_KEYWORDS.
        # KO: 일방통행으로 판정되지 않은 도로만 양방향 묶음으로 나눕니다. Dongbu
        #     OSM 행의 other_tags(예: oneway=yes, lanes=3)를 읽어 불필요한 반대방향
        #     차로와 주황색 중앙선이 생기지 않게 했습니다.
        if not oneway:
            total_lanes = max(2, int(total_lanes))
            if total_lanes % 2 != 0:
                total_lanes += 1

        lane_width = max(road_width / max(total_lanes, 1), 2.7)
        # EN: Keep the rendered road surface at least as wide as the generated lane bundle.
        # KO: 생성된 차로 묶음보다 도로 면이 좁지 않도록 보정합니다.
        effective_road_width = max(float(road_width), float(lane_width) * float(total_lanes))

        for part_index, g in enumerate(parts):
            coords = list(g.coords)
            if len(coords) < 2:
                continue
            if oneway:
                specs = [(coords, total_lanes, 1, False, False)]
            else:
                lanes_each_dir = max(1, int(total_lanes) // 2)
                specs = [
                    (coords, lanes_each_dir, 1, True, False),
                    (list(reversed(coords)), lanes_each_dir, -1, True, True),
                ]

            seg_count = len(coords) - 1
            for dcoords, lanes_this_dir, direction, bidir, reversed_source in specs:
                for si in range(len(dcoords) - 1):
                    x0, y0 = dcoords[si]
                    x1, y1 = dcoords[si + 1]
                    dx = float(x1) - float(x0)
                    dy = float(y1) - float(y0)
                    seg_len = math.hypot(dx, dy)
                    if seg_len < 0.5:
                        continue

                    # EN: source_segment must identify the same original physical
                    #     segment for forward and reverse lanes.  The old code used
                    #     si after reversing coords, so segment 0 in one direction
                    #     was grouped with the far-end segment in the opposite
                    #     direction.  That produced the stray lane markings shown in
                    #     the Dongbu screenshot.
                    # KO: reverse 방향에서도 원본 물리 segment 번호가 같아야 합니다.
                    #     예전에는 뒤집힌 coords의 si를 그대로 써서 서로 다른 위치의
                    #     segment들이 같은 차선 표시 그룹에 묶였고, 화면 오른쪽처럼
                    #     떨어진 이상 차선이 생겼습니다.
                    source_segment = (seg_count - 1 - si) if reversed_source else si

                    from_node = get_node(x0, y0, spawn_flag)
                    to_node = get_node(x1, y1, spawn_flag)
                    link_id = len(links)
                    seg_geom = LineString([(x0, y0), (x1, y1)])
                    links.append({
                        "link_id": link_id,
                        "from_node": from_node,
                        "to_node": to_node,
                        "geometry": seg_geom,
                        "length": float(seg_len),
                        "width": float(effective_road_width),
                        "raw_width": float(road_width),
                        "lane_count": int(lanes_this_dir),
                        "speed_mps": max(2.0, float(speed_kmh) / 3.6),
                        "source_row": int(ridx),
                        "source_part": int(part_index),
                        "source_segment": int(source_segment),
                        "direction": int(direction),
                        "oneway": bool(oneway),
                    })

                    hd = math.atan2(dy, dx)
                    nx_left = -math.sin(hd)
                    ny_left = math.cos(hd)
                    for li in range(lanes_this_dir):
                        lane_id = len(lanes)
                        offset = -(li + 0.5) * lane_width if bidir else (li - (lanes_this_dir - 1) * 0.5) * lane_width
                        sx = float(x0 + offset * nx_left)
                        sy = float(y0 + offset * ny_left)
                        ex = float(x1 + offset * nx_left)
                        ey = float(y1 + offset * ny_left)
                        lanes.append({
                            "lane_id": lane_id,
                            "link_id": link_id,
                            "lane_index": li,
                            "lane_index_from_right": li,
                            "lane_count": int(lanes_this_dir),
                            "lane_width": float(lane_width),
                            "road_width": float(effective_road_width),
                            "source_row": int(ridx),
                            "source_part": int(part_index),
                            "source_segment": int(source_segment),
                            "direction": int(direction),
                            "oneway": bool(oneway),
                            "length": float(seg_len),
                            "start_x": sx,
                            "start_y": sy,
                            "end_x": ex,
                            "end_y": ey,
                            "from_node": from_node,
                            "to_node": to_node,
                            "geometry": LineString([(sx, sy), (ex, ey)]),
                        })

    if not nodes or not links or not lanes:
        raise RuntimeError("Road network is empty.")
    print("[Road] rows one-way/two-way:", oneway_row_count, "/", twoway_row_count, "tag-lane rows:", parsed_lanes_from_tags)
    return nodes, links, lanes, gdf.crs

def lane_midpoint_for_side_order(lane):
    return (
        0.5 * (float(lane["start_x"]) + float(lane["end_x"])),
        0.5 * (float(lane["start_y"]) + float(lane["end_y"])),
    )


def link_direction_for_side_order(link):
    coords = list(link["geometry"].coords)
    if len(coords) < 2:
        return 1.0, 0.0
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    n = math.hypot(dx, dy)
    if n < 1.0e-9:
        return 1.0, 0.0
    return dx / n, dy / n


def lane_lateral_offset_for_side_order(lane, links_by_id):
    """
    Signed lane position relative to its travel direction.
    Negative means physical right side, positive means physical left side.
    This avoids trusting dataset lane_index order when enforcing turn lanes.
    """
    link = links_by_id.get(int(lane["link_id"]))
    if link is None:
        return float(lane.get("lane_index_from_right", 0))

    coords = list(link["geometry"].coords)
    if len(coords) >= 2:
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        bx = 0.5 * (float(x0) + float(x1))
        by = 0.5 * (float(y0) + float(y1))
    else:
        bx, by = lane_midpoint_for_side_order(lane)

    mx, my = lane_midpoint_for_side_order(lane)
    dx, dy = link_direction_for_side_order(link)
    left_x, left_y = -dy, dx
    return (mx - bx) * left_x + (my - by) * left_y


def sort_lanes_right_to_left(group, links_by_id):
    return sorted(
        group,
        key=lambda z: (
            lane_lateral_offset_for_side_order(z, links_by_id),
            int(z.get("lane_index_from_right", 0)),
            int(z["lane_id"]),
        ),
    )


def _interchange_model_enabled():
    return bool(INTERCHANGE_EDGE_ONLY)


def _group_is_ramp_side(group):
    return bool(group) and len(group) <= max(1, int(INTERCHANGE_RAMP_MAX_LANES))


def _group_is_main_side(group):
    return bool(group) and len(group) >= max(2, int(INTERCHANGE_MAIN_MIN_LANES))


def _lane_endpoint_xy(lane, at_end=False):
    if at_end:
        return float(lane["end_x"]), float(lane["end_y"])
    return float(lane["start_x"]), float(lane["start_y"])


def nearest_outer_lane_to_point(group, px, py, at_end=False):
    """Return the physical right/left edge lane nearest to a ramp endpoint."""
    if not group:
        return -1
    if len(group) == 1:
        return int(group[0]["lane_id"])
    right = group[0]
    left = group[-1]
    rx, ry = _lane_endpoint_xy(right, at_end=at_end)
    lx, ly = _lane_endpoint_xy(left, at_end=at_end)
    dr = (rx - px) * (rx - px) + (ry - py) * (ry - py)
    dl = (lx - px) * (lx - px) + (ly - py) * (ly - py)
    return int(right["lane_id"] if dr <= dl else left["lane_id"])


def interchange_source_outer_lane_id(current_group, next_group):
    """Mainline -> one-lane ramp: choose the nearest outside source lane."""
    if not _interchange_model_enabled():
        return -1
    if not (_group_is_main_side(current_group) and _group_is_ramp_side(next_group)):
        return -1
    ramp = next_group[0]
    px, py = _lane_endpoint_xy(ramp, at_end=False)
    return nearest_outer_lane_to_point(current_group, px, py, at_end=True)


def interchange_receiving_outer_lane_id(previous_group, current_group):
    """One-lane ramp -> mainline: choose the nearest outside receiving lane."""
    if not _interchange_model_enabled():
        return -1
    if not (_group_is_ramp_side(previous_group) and _group_is_main_side(current_group)):
        return -1
    ramp = previous_group[0]
    px, py = _lane_endpoint_xy(ramp, at_end=True)
    return nearest_outer_lane_to_point(current_group, px, py, at_end=False)


def _set_lane_endpoint(lane, x, y, at_end=False):
    if at_end:
        lane["end_x"] = float(x)
        lane["end_y"] = float(y)
    else:
        lane["start_x"] = float(x)
        lane["start_y"] = float(y)
    lane["geometry"] = LineString([(float(lane["start_x"]), float(lane["start_y"])), (float(lane["end_x"]), float(lane["end_y"]))])
    lane["length"] = float(math.hypot(float(lane["end_x"]) - float(lane["start_x"]), float(lane["end_y"]) - float(lane["start_y"])))


def _render_coords_for_link(link):
    geom = link.get("render_geometry", link.get("geometry"))
    coords = list(geom.coords) if geom is not None else list(link["geometry"].coords)
    if len(coords) < 2:
        coords = list(link["geometry"].coords)
    return [(float(x), float(y)) for x, y in coords]


def _set_link_render_endpoint(link, x, y, at_end=False):
    coords = _render_coords_for_link(link)
    if len(coords) < 2:
        return
    if at_end:
        coords[-1] = (float(x), float(y))
    else:
        coords[0] = (float(x), float(y))
    link["render_geometry"] = LineString(coords)


def apply_interchange_outer_edge_geometry(nodes, links, lanes, link_to_lanes):
    """Snap one-lane ramp endpoints to the nearest outside mainline lane.

    EN: Source GIS centerlines often connect ramps to the center of a wide road.
        The simulation uses lane centerlines, so ramp lane endpoints and static
        rendering are shifted to the nearest physical outside lane at the shared
        graph node.  Node IDs remain unchanged, preserving graph connectivity.
    KO: 원본 GIS 중심선은 램프가 넓은 본선의 중앙에 붙는 것처럼 들어오는 경우가
        많습니다. 차량 경로와 렌더링에서 램프 차로의 시작/끝점을 공유 노드의
        가장 가까운 본선 바깥 차로 위치로 이동해, 나들목이 도로 중앙이 아니라
        최외곽 차로에서 나가고 들어오도록 보정합니다.
    """
    if not (INTERCHANGE_EDGE_ONLY and INTERCHANGE_ADJUST_RAMP_GEOMETRY):
        return 0

    outgoing = {}
    incoming = {}
    for link in links:
        outgoing.setdefault(int(link["from_node"]), []).append(int(link["link_id"]))
        incoming.setdefault(int(link["to_node"]), []).append(int(link["link_id"]))

    lane_by_id = {int(l["lane_id"]): l for l in lanes}
    start_candidates = {}
    end_candidates = {}

    def add_candidate(store, ramp_link_id, anchor_x, anchor_y, ref_x, ref_y):
        d2 = (float(anchor_x) - float(ref_x)) ** 2 + (float(anchor_y) - float(ref_y)) ** 2
        old = store.get(int(ramp_link_id))
        if old is None or d2 < old[0]:
            store[int(ramp_link_id)] = (float(d2), float(anchor_x), float(anchor_y))

    for node_id in range(len(nodes)):
        inc_ids = incoming.get(int(node_id), [])
        out_ids = outgoing.get(int(node_id), [])

        # Mainline -> ramp diverge: snap the ramp start to the nearest edge of
        # the incoming multi-lane bundle.
        for main_id in inc_ids:
            main_group = link_to_lanes.get(int(main_id), [])
            if not _group_is_main_side(main_group):
                continue
            for ramp_id in out_ids:
                ramp_group = link_to_lanes.get(int(ramp_id), [])
                if not _group_is_ramp_side(ramp_group):
                    continue
                edge_id = interchange_source_outer_lane_id(main_group, ramp_group)
                edge_lane = lane_by_id.get(int(edge_id))
                ramp_lane = ramp_group[0] if ramp_group else None
                if edge_lane is None or ramp_lane is None:
                    continue
                ax, ay = _lane_endpoint_xy(edge_lane, at_end=True)
                rx, ry = _lane_endpoint_xy(ramp_lane, at_end=False)
                add_candidate(start_candidates, ramp_id, ax, ay, rx, ry)

        # Ramp -> mainline merge: snap the ramp end to the nearest edge of the
        # outgoing multi-lane bundle.
        for ramp_id in inc_ids:
            ramp_group = link_to_lanes.get(int(ramp_id), [])
            if not _group_is_ramp_side(ramp_group):
                continue
            for main_id in out_ids:
                main_group = link_to_lanes.get(int(main_id), [])
                if not _group_is_main_side(main_group):
                    continue
                edge_id = interchange_receiving_outer_lane_id(ramp_group, main_group)
                edge_lane = lane_by_id.get(int(edge_id))
                ramp_lane = ramp_group[0] if ramp_group else None
                if edge_lane is None or ramp_lane is None:
                    continue
                ax, ay = _lane_endpoint_xy(edge_lane, at_end=False)
                rx, ry = _lane_endpoint_xy(ramp_lane, at_end=True)
                add_candidate(end_candidates, ramp_id, ax, ay, rx, ry)

    changed = set()
    for ramp_id, (_, x, y) in start_candidates.items():
        group = link_to_lanes.get(int(ramp_id), [])
        if not group:
            continue
        lane = group[0]
        _set_lane_endpoint(lane, x, y, at_end=False)
        _set_link_render_endpoint(links[int(ramp_id)], x, y, at_end=False)
        changed.add(int(ramp_id))

    for ramp_id, (_, x, y) in end_candidates.items():
        group = link_to_lanes.get(int(ramp_id), [])
        if not group:
            continue
        lane = group[0]
        _set_lane_endpoint(lane, x, y, at_end=True)
        _set_link_render_endpoint(links[int(ramp_id)], x, y, at_end=True)
        changed.add(int(ramp_id))

    for ramp_id in changed:
        group = link_to_lanes.get(int(ramp_id), [])
        if not group:
            continue
        lane = group[0]
        links[int(ramp_id)]["length"] = float(lane.get("length", links[int(ramp_id)].get("length", 0.0)))

    if changed:
        print("[Interchange] outer-edge ramp geometry adjusted:", len(changed), "links")
    return len(changed)


def route_graph_weight_for_link(link):
    length = max(0.01, float(link.get("length", 0.0)))
    if not ROUTE_USE_WIDTH_BIASED_COST:
        return length

    lane_count = max(1.0, float(link.get("lane_count", 1)))
    width = max(1.0, float(link.get("width", link.get("raw_width", SPAWN_REF_WIDTH))))
    raw_width = max(1.0, float(link.get("raw_width", width)))
    speed = max(1.0, float(link.get("speed_mps", 13.9)))
    ref_lanes = max(1.0, float(ROUTE_WIDE_LINK_REF_LANES))
    power = max(0.0, float(ROUTE_WIDE_LINK_POWER))

    # EN: Penalize low-capacity alternatives rather than letting a narrow side
    #     link win merely because it is a few meters shorter.  The effective
    #     width check catches Dongbu-style one-lane slip/side roads even when
    #     lane_count metadata is noisy.
    # KO: 좁은 측도/램프가 몇 m 짧다는 이유로 본선을 이기지 못하게 저용량 링크
    #     비용을 크게 올립니다. raw/effective 폭도 함께 보아 차로 수 메타데이터가
    #     불안정한 Dongbu 형태의 좁은 연결로를 걸러냅니다.
    lane_factor = (ref_lanes / lane_count) ** power
    width_factor = (max(SPAWN_REF_WIDTH, 1.0) / width) ** (power * 0.75)
    speed_factor = (13.9 / speed) ** 0.20
    factor = max(lane_factor, width_factor, speed_factor)

    if lane_count <= 1.25 or raw_width <= float(ROUTE_NARROW_WIDTH_MAX):
        factor *= max(1.0, float(ROUTE_NARROW_LINK_PENALTY))

    factor = max(float(ROUTE_WIDE_LINK_MIN_COST_FACTOR), min(float(ROUTE_WIDE_LINK_MAX_COST_FACTOR), factor))
    return length * factor


def build_static_network(nodes, links, lanes):
    G = nx.DiGraph()
    for link in links:
        u = int(link["from_node"])
        v = int(link["to_node"])
        edge_data = {
            "weight": float(route_graph_weight_for_link(link)),
            "length_m": float(link.get("length", 0.0)),
            "lane_count": int(link.get("lane_count", 1)),
            "width": float(link.get("width", link.get("raw_width", 0.0))),
            "raw_width": float(link.get("raw_width", link.get("width", 0.0))),
            "link_id": int(link["link_id"]),
        }
        if G.has_edge(u, v):
            # EN: Keep the lower-cost/wider parallel edge.  A later one-lane
            #     side road should not overwrite the Dongbu mainline between
            #     the same graph nodes.
            # KO: 같은 노드쌍을 잇는 링크가 여러 개면 낮은 비용/넓은 링크를 유지합니다.
            #     뒤에서 읽힌 1차로 측도가 동부간선도로 본선을 덮어쓰지 않게 합니다.
            old = G[u][v]
            replace = edge_data["weight"] < float(old.get("weight", 1.0e30))
            if abs(edge_data["weight"] - float(old.get("weight", 1.0e30))) <= 1.0e-6:
                replace = edge_data["lane_count"] > int(old.get("lane_count", 1)) or edge_data["width"] > float(old.get("width", 0.0))
            if replace:
                G[u][v].update(edge_data)
        else:
            G.add_edge(u, v, **edge_data)

    links_by_id = {int(l["link_id"]): l for l in links}
    link_to_lanes = {}
    for lane in lanes:
        link_to_lanes.setdefault(int(lane["link_id"]), []).append(lane)
    for lid in link_to_lanes:
        # Geometry-normalized order: index 0 is the physical rightmost lane
        # for this lane group's travel direction; index -1 is the physical
        # leftmost lane.  Do not rely on source lane_index metadata here.
        link_to_lanes[lid] = sort_lanes_right_to_left(link_to_lanes[lid], links_by_id)

    left = np.full(len(lanes), -1, dtype=np.int32)
    right = np.full(len(lanes), -1, dtype=np.int32)
    for group in link_to_lanes.values():
        for i, lane in enumerate(group):
            lane_id = int(lane["lane_id"])
            if i > 0:
                right[lane_id] = int(group[i - 1]["lane_id"])
            if i < len(group) - 1:
                left[lane_id] = int(group[i + 1]["lane_id"])
    return G, link_to_lanes, left, right


# ============================================================
# Spatial matching
# ============================================================

def build_node_tree(nodes, valid_nodes=None):
    if cKDTree is None:
        return None, None
    if valid_nodes is None:
        ids = np.array([int(n["node_id"]) for n in nodes], dtype=np.int32)
    else:
        valid = set(int(x) for x in valid_nodes)
        ids = np.array([int(n["node_id"]) for n in nodes if int(n["node_id"]) in valid], dtype=np.int32)
    if len(ids) == 0:
        return None, None
    coords = np.array([[nodes[int(i)]["geometry"].x, nodes[int(i)]["geometry"].y] for i in ids], dtype=np.float64)
    return cKDTree(coords), ids


def transform_points(points, src_crs, dst_crs):
    if not points:
        return []
    if not src_crs or not dst_crs or str(src_crs) == str(dst_crs) or pyproj is None:
        return [(float(x), float(y)) for x, y in points]
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = []
    for x, y in points:
        try:
            xx, yy = transformer.transform(float(x), float(y))
            out.append((float(xx), float(yy)))
        except Exception:
            pass
    return out


def match_spawn_nodes(nodes, points, point_crs, network_crs, max_dist=150.0):
    if not points:
        matched = {int(n["node_id"]) for n in nodes if n.get("spawn", False)}
        print("[Spawn] matched nodes from road flag:", len(matched))
        return matched

    points = transform_points(points, point_crs, network_crs)
    tree, ids = build_node_tree(nodes)
    matched = set()

    if tree is not None:
        for x, y in points:
            d, idx = tree.query([x, y], k=1)
            if float(d) <= max_dist:
                matched.add(int(ids[int(idx)]))
    else:
        for x, y in points:
            best = -1
            best_d = 1.0e30
            for n in nodes:
                p = n["geometry"]
                d = math.hypot(float(p.x) - x, float(p.y) - y)
                if d < best_d:
                    best_d = d
                    best = int(n["node_id"])
            if best >= 0 and best_d <= max_dist:
                matched.add(best)

    print("[Spawn] matched nodes:", len(matched))
    return matched


def match_spawn_records(nodes, records, point_crs, network_crs, max_dist=None):
    """EN: Match spawn-point records to graph nodes and split them by SPWNTYPE.

    Returns:
      origin_nodes, destination_nodes, all_spawn_nodes, node_profiles

    KO: 스폰 레코드를 그래프 노드에 매칭하고 SPWNTYPE에 따라 출발/도착 노드를
    분리합니다. 시간대별 profile은 출발 가능한 노드에만 붙입니다.
    """
    if max_dist is None:
        max_dist = SPAWN_NODE_MATCH_MAX_DIST

    if not records:
        matched = {int(n["node_id"]) for n in nodes if n.get("spawn", False)}
        print("[Spawn] matched nodes from road flag:", len(matched))
        return set(matched), set(matched), set(matched), {}

    points = [(float(r["x"]), float(r["y"])) for r in records]
    points = transform_points(points, point_crs, network_crs)
    tree, ids = build_node_tree(nodes)

    origin_nodes = set()
    destination_nodes = set()
    all_nodes = set()
    profile_lists = {}

    for rec, (x, y) in zip(records, points):
        nearest = -1
        best_d = 1.0e30

        if tree is not None:
            d, idx = tree.query([x, y], k=1)
            best_d = float(d)
            if best_d <= float(max_dist):
                nearest = int(ids[int(idx)])
        else:
            for n in nodes:
                p = n["geometry"]
                d = math.hypot(float(p.x) - x, float(p.y) - y)
                if d < best_d:
                    best_d = d
                    nearest = int(n["node_id"])
            if best_d > float(max_dist):
                nearest = -1

        if nearest < 0:
            continue

        bits = int(rec.get("spwn_type", SPWNTYPE_BOTH))
        if bits & SPWNTYPE_ORIGIN:
            origin_nodes.add(nearest)
        if bits & SPWNTYPE_DESTINATION:
            destination_nodes.add(nearest)
        all_nodes.add(nearest)

        prof = rec.get("profile_vps", None)
        if prof is not None and (bits & SPWNTYPE_ORIGIN):
            profile_lists.setdefault(nearest, []).append(np.asarray(prof, dtype=np.float32))

    node_profiles = {}
    for nid, plist in profile_lists.items():
        if not plist:
            continue
        arr = np.vstack(plist).astype(np.float32, copy=False)
        with np.errstate(invalid="ignore"):
            node_profiles[int(nid)] = np.nanmean(arr, axis=0).astype(np.float32, copy=False)

    print("[Spawn] matched nodes:", len(all_nodes))
    print("[Spawn] origins:", len(origin_nodes), "destinations:", len(destination_nodes), "profile_nodes:", len(node_profiles))
    return origin_nodes, destination_nodes, all_nodes, node_profiles


# ============================================================
# Routing
# ============================================================

def node_xy(nodes, node_id):
    p = nodes[int(node_id)]["geometry"]
    return float(p.x), float(p.y)


def node_dist(nodes, a, b):
    ax, ay = node_xy(nodes, a)
    bx, by = node_xy(nodes, b)
    return math.hypot(ax - bx, ay - by)


def classify_turn(link_a, link_b):
    c1 = list(link_a["geometry"].coords)
    c2 = list(link_b["geometry"].coords)
    if len(c1) < 2 or len(c2) < 2:
        return TURN_STRAIGHT
    v1 = np.array([c1[-1][0] - c1[-2][0], c1[-1][1] - c1[-2][1]], dtype=np.float64)
    v2 = np.array([c2[1][0] - c2[0][0], c2[1][1] - c2[0][1]], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return TURN_STRAIGHT
    v1 /= n1
    v2 /= n2
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle = math.degrees(math.atan2(cross, dot))
    if angle > 25.0:
        return TURN_LEFT
    if angle < -25.0:
        return TURN_RIGHT
    return TURN_STRAIGHT


def signed_turn_angle_deg_between_links(link_a, link_b):
    c1 = list(link_a["geometry"].coords)
    c2 = list(link_b["geometry"].coords)
    if len(c1) < 2 or len(c2) < 2:
        return 0.0
    v1 = np.array([c1[-1][0] - c1[-2][0], c1[-1][1] - c1[-2][1]], dtype=np.float64)
    v2 = np.array([c2[1][0] - c2[0][0], c2[1][1] - c2[0][1]], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    v1 /= n1
    v2 /= n2
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(math.degrees(math.atan2(cross, dot)))


def is_wide_lane_count_continuation_py(link_id_a, link_id_b, link_to_lanes, links):
    if not ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT:
        return False
    try:
        a = int(link_id_a)
        b = int(link_id_b)
        if not (0 <= a < len(links) and 0 <= b < len(links)):
            return False
        group_a = link_to_lanes.get(a, [])
        group_b = link_to_lanes.get(b, [])
        ca = len(group_a)
        cb = len(group_b)
        if ca < 2 or cb < 2 or ca == cb:
            return False
        if min(ca, cb) <= int(INTERCHANGE_RAMP_MAX_LANES):
            return False
        if int(links[a].get("to_node", -1)) != int(links[b].get("from_node", -2)):
            return False
        angle = abs(signed_turn_angle_deg_between_links(links[a], links[b]))
        return angle <= float(ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG)
    except Exception:
        return False


def route_turn_for_link_transition(link_id_a, link_id_b, link_to_lanes, links):
    if is_wide_lane_count_continuation_py(link_id_a, link_id_b, link_to_lanes, links):
        return TURN_STRAIGHT
    return classify_turn(links[int(link_id_a)], links[int(link_id_b)])


def _balanced_lane_count_change_lane_id(group, previous_group=None, previous_lane_id=-1, key=0):
    """Map lanes through 4->3 / 3->4 style mainline changes without center collapse."""
    if not group:
        return -1
    if not previous_group or not ROUTE_LANE_COUNT_CHANGE_BALANCE:
        return _relative_lane_id(group, previous_group, previous_lane_id, default_key=key)
    n0 = len(previous_group)
    n1 = len(group)
    prev_idx = lane_index_in_group(previous_group, previous_lane_id)
    if prev_idx < 0 or n0 <= 1 or n1 <= 1 or n0 == n1:
        return _relative_lane_id(group, previous_group, previous_lane_id, default_key=key)

    # Keep both outside edges available, but distribute ambiguous middle lanes
    # with a deterministic jitter so 4->3 does not always collapse into one
    # middle lane.  Example 4->3: 0->0, 3->2, while lanes 1/2 are split
    # between neighboring target lanes by route/vehicle key.
    if prev_idx == 0:
        mapped_idx = 0
    elif prev_idx == n0 - 1:
        mapped_idx = n1 - 1
    else:
        jitter = (_stable_lane_mix((int(key) ^ (prev_idx * 2654435761) ^ (n0 * 131071) ^ n1)) & 65535) / 65536.0
        mapped_idx = int(math.floor((float(prev_idx) + jitter) * float(n1) / float(n0)))
    mapped_idx = max(0, min(n1 - 1, int(mapped_idx)))
    return int(group[mapped_idx]["lane_id"])


def _stable_lane_mix(value):
    x = (int(value) ^ int(ROUTE_RANDOM_LANE_SALT)) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return int(x & 0xFFFFFFFF)


def stable_lane_index(group, key=0):
    if not group:
        return -1
    return int(_stable_lane_mix(key) % max(1, len(group)))


def lane_choice_for_turn(group, turn, key=0):
    """Choose a source lane by route intent: right exit from right, left exit from left, through traffic spread."""
    if not group:
        return -1
    if turn == TURN_LEFT:
        return int(group[-1]["lane_id"])
    if turn == TURN_RIGHT:
        return int(group[0]["lane_id"])
    idx = stable_lane_index(group, key) if ROUTE_STRAIGHT_LANE_RANDOMIZE else len(group) // 2
    idx = max(0, min(len(group) - 1, int(idx)))
    return int(group[idx]["lane_id"])


def _relative_lane_id(group, previous_group=None, previous_lane_id=-1, default_key=0):
    if not group:
        return -1
    prev_idx = lane_index_in_group(previous_group or [], previous_lane_id)
    if prev_idx >= 0 and previous_group:
        if len(previous_group) <= 1:
            mapped_idx = stable_lane_index(group, default_key) if ROUTE_STRAIGHT_LANE_RANDOMIZE else len(group) // 2
        elif len(group) <= 1:
            mapped_idx = 0
        else:
            mapped_idx = int(round(float(prev_idx) * float(len(group) - 1) / float(max(1, len(previous_group) - 1))))
        mapped_idx = max(0, min(len(group) - 1, mapped_idx))
        return int(group[mapped_idx]["lane_id"])
    idx = stable_lane_index(group, default_key) if ROUTE_STRAIGHT_LANE_RANDOMIZE else len(group) // 2
    idx = max(0, min(len(group) - 1, int(idx)))
    return int(group[idx]["lane_id"])


def receiving_lane_after_turn(group, previous_turn, fallback_turn=TURN_STRAIGHT, key=0):
    """Lane to enter after an intersection, with upcoming-exit intent and non-center through lanes."""
    if not group:
        return -1

    # Physical right-to-left order: group[0] is rightmost, group[-1] is leftmost.
    # A completed turn enters the legal edge lane first.  For straight-through
    # entries, the upcoming maneuver controls lane intent: right exits prepare on
    # the right, left exits prepare on the left, and ordinary through traffic is
    # distributed instead of always using the center.
    if previous_turn == TURN_LEFT:
        return int(group[-1]["lane_id"])
    if previous_turn == TURN_RIGHT:
        return int(group[0]["lane_id"])
    if ROUTE_DESTINATION_LANE_INTENT:
        if fallback_turn == TURN_LEFT:
            return int(group[-1]["lane_id"])
        if fallback_turn == TURN_RIGHT:
            return int(group[0]["lane_id"])
    return lane_choice_for_turn(group, TURN_STRAIGHT, key=key)


def receiving_lane_after_turn_balanced(group, previous_turn, previous_group=None, previous_lane_id=-1, fallback_turn=TURN_STRAIGHT, key=0):
    """Preserve relative lane on straight arterials; use edge lanes for upcoming exits; randomize through lanes."""
    if not group:
        return -1

    if previous_turn == TURN_LEFT:
        return int(group[-1]["lane_id"])
    if previous_turn == TURN_RIGHT:
        return int(group[0]["lane_id"])

    if ROUTE_DESTINATION_LANE_INTENT:
        if fallback_turn == TURN_LEFT:
            return int(group[-1]["lane_id"])
        if fallback_turn == TURN_RIGHT:
            return int(group[0]["lane_id"])

    if (
        ROUTE_KEEP_RELATIVE_LANE
        and previous_turn == TURN_STRAIGHT
        and previous_group
        and len(previous_group) > 0
    ):
        if ROUTE_LANE_COUNT_CHANGE_BALANCE and len(previous_group) != len(group):
            return _balanced_lane_count_change_lane_id(
                group,
                previous_group=previous_group,
                previous_lane_id=previous_lane_id,
                key=key,
            )
        return _relative_lane_id(group, previous_group, previous_lane_id, default_key=key)

    return receiving_lane_after_turn(group, previous_turn, fallback_turn=fallback_turn, key=key)


def lane_index_in_group(group, lane_id):
    for idx, lane in enumerate(group):
        if int(lane["lane_id"]) == int(lane_id):
            return int(idx)
    return -1


def lane_change_steps_needed(group, from_lane_id, turn):
    if not group or turn not in (TURN_LEFT, TURN_RIGHT):
        return 0
    cur = lane_index_in_group(group, from_lane_id)
    if cur < 0:
        return 0
    target = len(group) - 1 if turn == TURN_LEFT else 0
    return abs(int(target) - int(cur))


def route_link_has_enough_turn_prep(group, entry_lane_id, turn, link_length):
    if not STRICT_ROUTE_TURN_LANE_FILTER:
        return True
    steps = lane_change_steps_needed(group, entry_lane_id, turn)
    if steps <= 0:
        return True
    required = TURN_LANE_ROUTE_PREP_BASE + TURN_LANE_ROUTE_PREP_PER_LANE * steps
    # Do not over-prune tiny map segments; CUDA still handles hard cases by
    # holding before the stop line.  This filter only removes obviously
    # impossible lane-position jumps from cached route generation.
    return float(link_length) >= min(required, 120.0)


def lane_choices_for_spawn_start(group, turn, link_length):
    """Return candidate first lanes for balanced multi-lane spawning.

    Straight routes may start in any lane of the outgoing bundle.  Turning
    routes may also start from neighboring lanes when the route-prep filter
    allows enough distance for a mandatory lane change; otherwise the legal
    edge turn lane remains as a safe fallback.
    """
    if not group:
        return []

    fallback = lane_choice_for_turn(group, turn, key=(int(link_length) ^ (int(turn) * 1009)))
    if not SPAWN_MULTI_LANE_BALANCE or len(group) <= 1:
        return [int(fallback)] if int(fallback) >= 0 else []

    candidates = []
    for lane in group:
        lane_id = int(lane["lane_id"])
        if turn == TURN_STRAIGHT or route_link_has_enough_turn_prep(group, lane_id, turn, link_length):
            candidates.append(lane_id)

    if not candidates and int(fallback) >= 0:
        candidates = [int(fallback)]
    elif int(fallback) >= 0 and int(fallback) not in candidates:
        candidates.append(int(fallback))

    # Keep the same physical right-to-left order used by lane adjacency.
    return [int(x) for x in candidates]


def _make_routes_worker(worker_id, target_routes, max_tries, seed, G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, min_trip_distance):
    rng = np.random.default_rng(int(seed) + int(worker_id) * 100003)
    origin_nodes = np.array(sorted(int(n) for n in origin_nodes if int(n) in G.nodes), dtype=np.int32)
    destination_nodes = np.array(sorted(int(n) for n in destination_nodes if int(n) in G.nodes), dtype=np.int32)
    if len(origin_nodes) < 1 or len(destination_nodes) < 1:
        return [], [], 0, 0

    links_by_id = {int(l["link_id"]): l for l in links}
    route_lane_lists = []
    route_turn_lists = []
    made = 0
    tries = 0
    min_od_euclidean = max(50.0, float(min_trip_distance) * 0.35)

    while made < target_routes and tries < max_tries:
        tries += 1
        origin = int(origin_nodes[rng.integers(0, len(origin_nodes))])
        dest = int(destination_nodes[rng.integers(0, len(destination_nodes))])
        if origin == dest or node_dist(nodes, origin, dest) < min_od_euclidean:
            continue
        try:
            heuristic_scale = float(ROUTE_WIDE_LINK_MIN_COST_FACTOR) if ROUTE_USE_WIDTH_BIASED_COST else 1.0
            path = nx.astar_path(G, origin, dest, heuristic=lambda a, b: node_dist(nodes, a, b) * heuristic_scale, weight="weight")
        except Exception:
            continue
        if len(path) < 2:
            continue

        total_len = 0.0
        link_path = []
        valid = True
        for a, b in zip(path[:-1], path[1:]):
            if not G.has_edge(a, b):
                valid = False
                break
            edge = G[a][b]
            link_id = int(edge["link_id"])
            total_len += float(edge.get("length_m", edge.get("weight", 0.0)))
            link_path.append(link_id)
        if not valid or total_len < min_trip_distance:
            continue

        link_turns = []
        for i, link_id in enumerate(link_path):
            if i < len(link_path) - 1:
                link_turns.append(route_turn_for_link_transition(link_id, link_path[i + 1], link_to_lanes, links))
            else:
                link_turns.append(TURN_STRAIGHT)

        lane_path = []
        turn_path = []
        for i, link_id in enumerate(link_path):
            group = link_to_lanes.get(link_id)
            if not group:
                valid = False
                break

            turn = int(link_turns[i])

            if i == 0:
                # EN: For multi-lane spawn links, do not always choose only the
                #     middle/right/left route-start lane.  Pick among every
                #     usable lane so route caches contain starts from all lanes.
                #     Exception: if the very next link is a one-lane ramp, the
                #     vehicle starts from the nearest outside exit lane.
                # KO: 다차로 스폰 링크에서는 중앙/우측/좌측 한 차로만 고르지 않고,
                #     사용 가능한 모든 차로 중에서 고릅니다. 단, 바로 다음 링크가
                #     1차로 나들목이면 가장 가까운 최외곽 진출 차로에서 시작합니다.
                next_group = link_to_lanes.get(int(link_path[i + 1]), []) if i < len(link_path) - 1 else []
                edge_exit = interchange_source_outer_lane_id(group, next_group)
                if edge_exit >= 0:
                    choices = [int(edge_exit)]
                else:
                    choices = lane_choices_for_spawn_start(group, turn, links_by_id[link_id].get("length", 0.0))
                if not choices:
                    valid = False
                    break
                lane_id = int(choices[int(rng.integers(0, len(choices)))])
            else:
                prev_turn = int(link_turns[i - 1])
                prev_group = link_to_lanes.get(int(link_path[i - 1]), [])
                prev_lane_id = int(lane_path[-1]) if lane_path else -1
                lane_key = (int(worker_id) * 1000003) ^ (int(tries) * 9176) ^ (int(i) * 131071) ^ int(link_id)
                lane_id = receiving_lane_after_turn_balanced(
                    group,
                    prev_turn,
                    previous_group=prev_group,
                    previous_lane_id=prev_lane_id,
                    fallback_turn=turn,
                    key=lane_key,
                )

                # EN/KO: Ramp/on-ramp receiving edge guard.  If a one-lane
                # ramp enters a multi-lane mainline, enter through the nearest
                # outside lane instead of the center/random lane.
                edge_receive = interchange_receiving_outer_lane_id(prev_group, group)
                if edge_receive >= 0:
                    lane_id = int(edge_receive)
                else:
                    # EN/KO: For a mainline link that immediately exits to a
                    # one-lane ramp, keep the route's base lane on the nearest
                    # outside exit edge.  CUDA can still disperse after merges.
                    next_group = link_to_lanes.get(int(link_path[i + 1]), []) if i < len(link_path) - 1 else []
                    edge_exit = interchange_source_outer_lane_id(group, next_group)
                    if edge_exit >= 0:
                        lane_id = int(edge_exit)

            if lane_id < 0:
                valid = False
                break

            if not route_link_has_enough_turn_prep(group, lane_id, turn, links_by_id[link_id].get("length", 0.0)):
                valid = False
                break

            lane_path.append(int(lane_id))
            turn_path.append(int(turn))
        if not valid or not lane_path:
            continue
        route_lane_lists.append(lane_path)
        route_turn_lists.append(turn_path)
        made += 1
    return route_lane_lists, route_turn_lists, made, tries


def _pack_route_lists(route_lane_lists, route_turn_lists):
    route_offsets = [0]
    route_lanes = []
    route_turns = []
    for lane_path, turn_path in zip(route_lane_lists, route_turn_lists):
        if not lane_path:
            continue
        route_lanes.extend([int(x) for x in lane_path])
        route_turns.extend([int(x) for x in turn_path])
        route_offsets.append(len(route_lanes))
    return as_contig_i32(route_offsets), as_contig_i32(route_lanes), as_contig_i32(route_turns)


def retarget_route_start_lane_path(lane_path, turn_path, new_first_lane, link_to_lanes, lanes):
    """Adjust a cloned route so straight-through segments keep lane position."""
    if not lane_path:
        return []

    new_path = [int(x) for x in lane_path]
    new_path[0] = int(new_first_lane)

    for i in range(1, len(new_path)):
        prev_lane = int(new_path[i - 1])
        original_lane = int(lane_path[i])
        if not (0 <= prev_lane < len(lanes) and 0 <= original_lane < len(lanes)):
            continue

        prev_link = int(lanes[prev_lane]["link_id"])
        current_link = int(lanes[original_lane]["link_id"])
        prev_group = link_to_lanes.get(prev_link, [])
        current_group = link_to_lanes.get(current_link, [])

        # EN/KO: If this transition is a mainline -> ramp/narrow interchange
        # exit, the source lane itself must be the nearest outside lane.  Earlier
        # multi-lane spawn expansion could clone the first link onto an inner
        # lane and then leave the vehicle unable to legally enter the ramp.
        edge_exit = interchange_source_outer_lane_id(prev_group, current_group)
        if edge_exit >= 0 and int(new_path[i - 1]) != int(edge_exit):
            new_path[i - 1] = int(edge_exit)
            prev_lane = int(edge_exit)

        previous_turn = int(turn_path[i - 1]) if i - 1 < len(turn_path) else TURN_STRAIGHT
        fallback_turn = int(turn_path[i]) if i < len(turn_path) else TURN_STRAIGHT

        lane_key = (int(new_first_lane) * 1000003) ^ (int(prev_lane) * 9176) ^ (int(i) * 131071) ^ int(current_link)
        lane_id = receiving_lane_after_turn_balanced(
            current_group,
            previous_turn,
            previous_group=prev_group,
            previous_lane_id=prev_lane,
            fallback_turn=fallback_turn,
            key=lane_key,
        )
        edge_receive = interchange_receiving_outer_lane_id(prev_group, current_group)
        if edge_receive >= 0:
            lane_id = int(edge_receive)
        if lane_id >= 0:
            new_path[i] = int(lane_id)

    return new_path


def route_lane_path_has_turn_prep(lane_path, turn_path, link_to_lanes, lanes, links):
    if not STRICT_ROUTE_TURN_LANE_FILTER:
        return True
    for i, lane_id in enumerate(lane_path):
        lane_id = int(lane_id)
        if not (0 <= lane_id < len(lanes)):
            return False
        link_id = int(lanes[lane_id]["link_id"])
        group = link_to_lanes.get(link_id, [])
        turn = int(turn_path[i]) if i < len(turn_path) else TURN_STRAIGHT
        link_len = float(links[link_id].get("length", 0.0)) if 0 <= link_id < len(links) else 0.0
        if not route_link_has_enough_turn_prep(group, lane_id, turn, link_len):
            return False
    return True




def _py_lane_connected(a, b, lanes):
    try:
        a = int(a)
        b = int(b)
        return (
            0 <= a < len(lanes)
            and 0 <= b < len(lanes)
            and int(lanes[a]["to_node"]) == int(lanes[b]["from_node"])
        )
    except Exception:
        return False


def route_lane_path_is_connected(lane_path, turn_path, link_to_lanes, lanes):
    """Return True only if every consecutive lane handoff is physically reachable."""
    if not ROUTE_VALIDATE_LANE_CONNECTIVITY:
        return True
    if not lane_path:
        return False

    for lane_id in lane_path:
        if not (0 <= int(lane_id) < len(lanes)):
            return False

    for i in range(len(lane_path) - 1):
        a = int(lane_path[i])
        b = int(lane_path[i + 1])
        if not _py_lane_connected(a, b, lanes):
            return False

        # EN/KO: Interchange edge rule must hold in cached routes too, not only
        # in the live CUDA guard.  If the route says a mainline lane exits into
        # a one-lane ramp from an inner lane, discard it before spawning.
        a_group = link_to_lanes.get(int(lanes[a]["link_id"]), [])
        b_group = link_to_lanes.get(int(lanes[b]["link_id"]), [])
        edge_exit = interchange_source_outer_lane_id(a_group, b_group)
        if edge_exit >= 0 and int(edge_exit) != a:
            return False
        edge_receive = interchange_receiving_outer_lane_id(a_group, b_group)
        if edge_receive >= 0 and int(edge_receive) != b:
            return False

    return True


def filter_invalid_route_arrays(route_offsets, route_lanes, route_turns, link_to_lanes, lanes):
    """Drop route paths with broken lane connectivity before they can spawn stuck cars."""
    if not ROUTE_DROP_INVALID_LANE_PATHS:
        return as_contig_i32(route_offsets), as_contig_i32(route_lanes), as_contig_i32(route_turns)

    route_offsets = np.asarray(route_offsets, dtype=np.int32)
    route_lanes = np.asarray(route_lanes, dtype=np.int32)
    route_turns = np.asarray(route_turns, dtype=np.int32)

    kept_lanes = []
    kept_turns = []
    dropped = 0
    repaired_edge = 0

    for rid in range(max(0, len(route_offsets) - 1)):
        off0 = int(route_offsets[rid])
        off1 = int(route_offsets[rid + 1])
        if off1 <= off0:
            dropped += 1
            continue
        lane_path = route_lanes[off0:off1].astype(np.int32, copy=True).tolist()
        turn_path = route_turns[off0:off1].astype(np.int32, copy=True).tolist()

        # First try a conservative interchange-edge repair, then validate.
        repaired = False
        for i in range(len(lane_path) - 1):
            a = int(lane_path[i])
            b = int(lane_path[i + 1])
            if not (0 <= a < len(lanes) and 0 <= b < len(lanes)):
                continue
            a_group = link_to_lanes.get(int(lanes[a]["link_id"]), [])
            b_group = link_to_lanes.get(int(lanes[b]["link_id"]), [])
            edge_exit = interchange_source_outer_lane_id(a_group, b_group)
            if edge_exit >= 0 and int(edge_exit) != a and _py_lane_connected(edge_exit, b, lanes):
                lane_path[i] = int(edge_exit)
                repaired = True
            a = int(lane_path[i])
            edge_receive = interchange_receiving_outer_lane_id(a_group, b_group)
            if edge_receive >= 0 and int(edge_receive) != b and _py_lane_connected(a, edge_receive, lanes):
                lane_path[i + 1] = int(edge_receive)
                repaired = True

        if repaired:
            repaired_edge += 1

        if not route_lane_path_is_connected(lane_path, turn_path, link_to_lanes, lanes):
            dropped += 1
            continue
        kept_lanes.append(lane_path)
        kept_turns.append(turn_path)

    if not kept_lanes:
        raise RuntimeError("All generated routes were rejected by lane-connectivity validation.")

    out_offsets, out_lanes, out_turns = _pack_route_lists(kept_lanes, kept_turns)
    if dropped or repaired_edge:
        print(
            "[Routes] lane-connectivity guard:",
            "kept=", int(len(out_offsets) - 1),
            "dropped=", int(dropped),
            "interchange_edge_repaired=", int(repaired_edge),
        )
    return out_offsets, out_lanes, out_turns

def _expand_multilane_spawn_chunk(start_rid, end_rid, route_offsets, route_lanes, route_turns, link_to_lanes, lanes, links):
    """Worker body for multi-lane spawn route-start expansion."""
    expanded_lane_lists = []
    expanded_turn_lists = []
    clone_count = 0
    multi_group_count = 0

    route_count = max(0, len(route_offsets) - 1)
    start_rid = max(0, min(int(start_rid), route_count))
    end_rid = max(start_rid, min(int(end_rid), route_count))

    for rid in range(start_rid, end_rid):
        off0 = int(route_offsets[rid])
        off1 = int(route_offsets[rid + 1])
        if off1 <= off0:
            continue

        lane_path = route_lanes[off0:off1].astype(np.int32, copy=True).tolist()
        turn_path = route_turns[off0:off1].astype(np.int32, copy=True).tolist()
        first_lane = int(lane_path[0])

        choices = [first_lane]
        if 0 <= first_lane < len(lanes):
            first_link = int(lanes[first_lane]["link_id"])
            group = link_to_lanes.get(first_link, [])
            if len(group) > 1:
                turn0 = int(turn_path[0]) if turn_path else TURN_STRAIGHT
                link_len = float(links[first_link].get("length", 0.0)) if 0 <= first_link < len(links) else 0.0

                # EN/KO: If the very first transition is already a ramp/narrow
                # interchange exit, do not clone starts onto inner lanes; those
                # cars would reach the node on a lane that cannot legally enter
                # the ramp and could stall there.
                next_group = []
                if len(lane_path) > 1 and 0 <= int(lane_path[1]) < len(lanes):
                    next_group = link_to_lanes.get(int(lanes[int(lane_path[1])]["link_id"]), [])
                edge_exit = interchange_source_outer_lane_id(group, next_group)
                if edge_exit >= 0:
                    alt = [int(edge_exit)]
                else:
                    alt = lane_choices_for_spawn_start(group, turn0, link_len)
                if alt:
                    choices = alt
                    multi_group_count += 1

        for lane0 in choices:
            if int(lane0) == first_lane:
                new_lane_path = list(lane_path)
            else:
                new_lane_path = retarget_route_start_lane_path(
                    lane_path,
                    turn_path,
                    int(lane0),
                    link_to_lanes,
                    lanes,
                )
                if not route_lane_path_has_turn_prep(new_lane_path, turn_path, link_to_lanes, lanes, links):
                    continue

            expanded_lane_lists.append(new_lane_path)
            expanded_turn_lists.append(list(turn_path))
            if int(lane0) != first_lane:
                clone_count += 1

    return int(start_rid), expanded_lane_lists, expanded_turn_lists, int(clone_count), int(multi_group_count)


def _route_count_from_offsets(route_offsets):
    return max(0, int(len(route_offsets) - 1))


def _chunk_route_ranges(route_count, worker_count):
    route_count = max(0, int(route_count))
    worker_count = max(1, int(worker_count))
    worker_count = min(worker_count, max(1, route_count))
    base = route_count // worker_count
    rem = route_count % worker_count
    ranges = []
    start = 0
    for i in range(worker_count):
        size = base + (1 if i < rem else 0)
        end = start + size
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges



def expand_routes_for_multilane_spawn(route_offsets, route_lanes, route_turns, link_to_lanes, lanes, links):
    """Clone route starts so every lane of a multi-lane spawn link can emit cars.

    The CUDA spawn kernel requires the spawn lane to appear in the selected
    route.  Older route caches often contain only one first lane per outgoing
    link, so this CPU-side expansion duplicates each route with alternative
    first lanes from the same physical lane group.  Only the first lane changes;
    the downstream path and turn metadata are preserved.

    EN: v15 can run the expansion in coarse chunks and then save the expanded
    route set into the ready-route cache.  Coarse thread chunks avoid copying
    shapely/geopandas-derived route metadata and preserve deterministic route
    order after sorting by chunk.
    KO: v15에서는 이 확장을 chunk 단위로 병렬 처리하고, 결과를 ready route
    cache에 저장할 수 있습니다. chunk 결과를 시작 route 번호로 정렬하므로
    병렬 처리해도 route 순서는 안정적으로 유지됩니다.
    """
    if not SPAWN_MULTI_LANE_BALANCE:
        return as_contig_i32(route_offsets), as_contig_i32(route_lanes), as_contig_i32(route_turns)

    t0 = time.perf_counter()
    route_offsets = np.asarray(route_offsets, dtype=np.int32)
    route_lanes = np.asarray(route_lanes, dtype=np.int32)
    route_turns = np.asarray(route_turns, dtype=np.int32)
    route_count = _route_count_from_offsets(route_offsets)

    configured_workers = int(ROUTE_SPAWN_EXPANSION_WORKERS)
    if configured_workers <= 0:
        configured_workers = int(ROUTE_WORKERS)
    worker_count = max(1, min(configured_workers, route_count if route_count > 0 else 1))
    use_parallel = (
        bool(ROUTE_SPAWN_EXPANSION_PARALLEL)
        and worker_count > 1
        and route_count >= max(2, int(ROUTE_SPAWN_EXPANSION_MIN_ROUTES))
    )

    results = []
    if use_parallel:
        ranges = _chunk_route_ranges(route_count, worker_count)
        backend = str(ROUTE_SPAWN_EXPANSION_BACKEND).strip().lower()
        if backend == "process":
            # EN/KO: route expansion shares shapely/geopandas-derived lane/link
            # metadata.  ProcessPool works only after pickling that full metadata
            # into every worker on Windows, which is often slower and less stable
            # than the actual expansion.  Use thread chunks for stable startup.
            print("[Routes] multi-lane spawn expansion: process backend requested; using thread backend for stable shared-memory route data")
            backend = "thread"
        Executor = ThreadPoolExecutor
        print(
            "[Routes] multi-lane spawn expansion parallel:",
            "backend=", backend,
            "workers=", len(ranges),
            "base_routes=", int(route_count),
        )
        try:
            with Executor(max_workers=len(ranges)) as executor:
                futures = [
                    executor.submit(
                        _expand_multilane_spawn_chunk,
                        int(start), int(end),
                        route_offsets, route_lanes, route_turns,
                        link_to_lanes, lanes, links,
                    )
                    for start, end in ranges
                ]
                for fut in as_completed(futures):
                    results.append(fut.result())
            results.sort(key=lambda x: int(x[0]))
        except Exception as e:
            print("[Routes] multi-lane spawn expansion parallel failed; fallback serial:", e)
            results = []

    if not results:
        results = [_expand_multilane_spawn_chunk(0, route_count, route_offsets, route_lanes, route_turns, link_to_lanes, lanes, links)]

    expanded_lane_lists = []
    expanded_turn_lists = []
    clone_count = 0
    multi_group_count = 0
    for _start, lane_lists, turn_lists, clones, hits in results:
        expanded_lane_lists.extend(lane_lists)
        expanded_turn_lists.extend(turn_lists)
        clone_count += int(clones)
        multi_group_count += int(hits)

    if not expanded_lane_lists:
        return as_contig_i32(route_offsets), as_contig_i32(route_lanes), as_contig_i32(route_turns)

    out_offsets, out_lanes, out_turns = _pack_route_lists(expanded_lane_lists, expanded_turn_lists)
    elapsed = time.perf_counter() - t0
    if clone_count > 0:
        print(
            "[Routes] multi-lane spawn expansion:",
            "base_routes=", int(route_count),
            "expanded_routes=", int(len(out_offsets) - 1),
            "cloned_starts=", int(clone_count),
            "multi_lane_route_hits=", int(multi_group_count),
            "elapsed_sec=", f"{elapsed:.3f}",
        )
    else:
        print("[Routes] multi-lane spawn expansion: no additional lanes needed", "elapsed_sec=", f"{elapsed:.3f}")
    return out_offsets, out_lanes, out_turns

def make_routes_single(G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, num_routes):
    print("[Routes] single generation")
    lane_lists, turn_lists, made, tries = _make_routes_worker(
        0, int(num_routes), int(MAX_ROUTE_TRIES), int(ROUTE_SEED), G, nodes, links, link_to_lanes,
        origin_nodes, destination_nodes, float(MIN_TRIP_DISTANCE)
    )
    if made <= 0:
        raise RuntimeError("No valid routes generated.")
    print("[Routes] single made:", made, "tries:", tries)
    return _pack_route_lists(lane_lists, turn_lists)


def make_routes_parallel(G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, num_routes):
    worker_count = max(1, min(int(ROUTE_WORKERS), int(num_routes)))
    if not ROUTE_PARALLEL or worker_count <= 1 or int(num_routes) <= 1:
        return make_routes_single(G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, num_routes)

    base = int(num_routes) // worker_count
    rem = int(num_routes) % worker_count
    targets = [base + (1 if i < rem else 0) for i in range(worker_count)]
    tries_per_worker = max(1000, int(math.ceil(float(MAX_ROUTE_TRIES) / float(worker_count))))
    print("[Routes] parallel generation")
    print("[Routes] backend:", ROUTE_PARALLEL_BACKEND)
    print("[Routes] workers:", worker_count)

    Executor = ProcessPoolExecutor if ROUTE_PARALLEL_BACKEND == "process" else ThreadPoolExecutor
    all_lane_lists = []
    all_turn_lists = []
    total_made = 0

    with Executor(max_workers=worker_count) as executor:
        futures = []
        for worker_id, target in enumerate(targets):
            if target <= 0:
                continue
            futures.append(executor.submit(
                _make_routes_worker, int(worker_id), int(target), int(tries_per_worker), int(ROUTE_SEED),
                G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, float(MIN_TRIP_DISTANCE)
            ))
        for fut in as_completed(futures):
            try:
                lane_lists, turn_lists, made, tries = fut.result()
                all_lane_lists.extend(lane_lists)
                all_turn_lists.extend(turn_lists)
                total_made += int(made)
                print(f"[Routes] worker done: made={made}, tries={tries}, total={total_made}/{num_routes}")
            except Exception as e:
                print("[Routes] worker failed:", e)

    if total_made < int(num_routes):
        remaining = int(num_routes) - int(total_made)
        print("[Routes] fallback single remaining:", remaining)
        try:
            extra_offsets, extra_lanes, extra_turns = make_routes_single(G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, remaining)
            for rid in range(len(extra_offsets) - 1):
                off0 = int(extra_offsets[rid])
                off1 = int(extra_offsets[rid + 1])
                if off1 > off0:
                    all_lane_lists.append(extra_lanes[off0:off1].astype(np.int32).tolist())
                    all_turn_lists.append(extra_turns[off0:off1].astype(np.int32).tolist())
        except Exception as e:
            print("[Routes] fallback failed:", e)

    if not all_lane_lists:
        raise RuntimeError("No valid routes generated.")
    all_lane_lists = all_lane_lists[:int(num_routes)]
    all_turn_lists = all_turn_lists[:int(num_routes)]
    route_offsets, route_lanes, route_turns = _pack_route_lists(all_lane_lists, all_turn_lists)
    print("[Routes] final count:", len(route_offsets) - 1)
    print("[Routes] lane elements:", len(route_lanes))
    return route_offsets, route_lanes, route_turns


def make_route_cache_key(gpkg_path, network_crs, num_routes, min_trip_distance, nodes, links, lanes, origin_nodes=None, destination_nodes=None):
    import hashlib
    h = hashlib.sha1()
    h.update(str(ROUTE_CACHE_VERSION).encode("utf-8"))
    h.update(str(gpkg_path).encode("utf-8", errors="ignore"))
    h.update(str(network_crs).encode("utf-8", errors="ignore"))
    h.update(str(int(num_routes)).encode("utf-8"))
    h.update(str(float(min_trip_distance)).encode("utf-8"))
    h.update(str(len(nodes)).encode("utf-8"))
    h.update(str(len(links)).encode("utf-8"))
    h.update(str(len(lanes)).encode("utf-8"))
    # EN/KO: Include route-shaping config that changes generated lane paths or
    # the final ready-route cache content.  This prevents stale route caches
    # when toggling multi-lane spawn, lane-drop balancing, destination spread,
    # or interchange edge-only behavior.
    for cache_name, cache_value in [
        ("SPAWN_MULTI_LANE_BALANCE", SPAWN_MULTI_LANE_BALANCE),
        ("ROUTE_VALIDATE_LANE_CONNECTIVITY", ROUTE_VALIDATE_LANE_CONNECTIVITY),
        ("ROUTE_DROP_INVALID_LANE_PATHS", ROUTE_DROP_INVALID_LANE_PATHS),
        ("ROUTE_USE_WIDTH_BIASED_COST", ROUTE_USE_WIDTH_BIASED_COST),
        ("ROUTE_KEEP_RELATIVE_LANE", ROUTE_KEEP_RELATIVE_LANE),
        ("ROUTE_DESTINATION_LANE_INTENT", ROUTE_DESTINATION_LANE_INTENT),
        ("ROUTE_STRAIGHT_LANE_RANDOMIZE", ROUTE_STRAIGHT_LANE_RANDOMIZE),
        ("ROUTE_RANDOM_LANE_SALT", ROUTE_RANDOM_LANE_SALT),
        ("ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT", ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT),
        ("ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG", ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG),
        ("ROUTE_LANE_COUNT_CHANGE_BALANCE", ROUTE_LANE_COUNT_CHANGE_BALANCE),
        ("ROUTE_DESTINATION_LANE_SPREAD", ROUTE_DESTINATION_LANE_SPREAD),
        ("INTERCHANGE_EDGE_ONLY", INTERCHANGE_EDGE_ONLY),
        ("INTERCHANGE_RAMP_MAX_LANES", INTERCHANGE_RAMP_MAX_LANES),
        ("INTERCHANGE_MAIN_MIN_LANES", INTERCHANGE_MAIN_MIN_LANES),
        ("INTERCHANGE_ADJUST_RAMP_GEOMETRY", INTERCHANGE_ADJUST_RAMP_GEOMETRY),
    ]:
        h.update(f"{cache_name}={cache_value}".encode("utf-8", errors="ignore"))
    if origin_nodes is not None:
        h.update(b"origins:")
        h.update(",".join(str(int(n)) for n in sorted(origin_nodes)).encode("utf-8"))
    if destination_nodes is not None:
        h.update(b"destinations:")
        h.update(",".join(str(int(n)) for n in sorted(destination_nodes)).encode("utf-8"))
    try:
        p = Path(gpkg_path)
        base = p.with_suffix("")
        for ext in [".gpkg", ".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            fp = base.with_suffix(ext)
            if fp.exists():
                st = fp.stat()
                h.update(fp.name.encode("utf-8", errors="ignore"))
                h.update(str(st.st_mtime_ns).encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
    except Exception:
        pass

    # EN/KO: Route arrays also depend on routing and lane-intent settings.
    # Include them so a config behavior change cannot accidentally reuse an
    # older route cache with incompatible lane choices.
    route_identity = {
        # EN/KO: Do not include CONFIG_VERSION itself.  Only route-shaping
        # values below should invalidate the expensive base/ready route cache.
        "route_seed": int(ROUTE_SEED),
        "route_parallel": bool(ROUTE_PARALLEL),
        "route_parallel_backend": str(ROUTE_PARALLEL_BACKEND),
        "route_workers": int(ROUTE_WORKERS),
        "road_layer": str(GPKG_LAYER),
        "spawn_layer": str(SPAWN_GPKG_LAYER),
        "road_oneway_mode": str(ROAD_ONEWAY_MODE),
        "road_default_oneway": bool(ROAD_DEFAULT_ONEWAY),
        "spawn_node_match_max_dist": float(SPAWN_NODE_MATCH_MAX_DIST),
        "min_trip_distance": float(MIN_TRIP_DISTANCE),
        "width_biased_cost": bool(ROUTE_USE_WIDTH_BIASED_COST),
        "wide_ref_lanes": float(ROUTE_WIDE_LINK_REF_LANES),
        "wide_power": float(ROUTE_WIDE_LINK_POWER),
        "narrow_penalty": float(ROUTE_NARROW_LINK_PENALTY),
        "keep_relative_lane": bool(ROUTE_KEEP_RELATIVE_LANE),
        "destination_lane_intent": bool(ROUTE_DESTINATION_LANE_INTENT),
        "straight_lane_randomize": bool(ROUTE_STRAIGHT_LANE_RANDOMIZE),
        "random_lane_salt": int(ROUTE_RANDOM_LANE_SALT),
        "spawn_multi_lane_balance": bool(SPAWN_MULTI_LANE_BALANCE),
        "interchange_edge_only": bool(INTERCHANGE_EDGE_ONLY),
        "interchange_ramp_max_lanes": int(INTERCHANGE_RAMP_MAX_LANES),
        "interchange_main_min_lanes": int(INTERCHANGE_MAIN_MIN_LANES),
        "wide_lane_change_as_straight": bool(ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT),
        "wide_continuation_max_turn_deg": float(ROUTE_WIDE_CONTINUATION_MAX_TURN_DEG),
        "lane_count_change_balance": bool(ROUTE_LANE_COUNT_CHANGE_BALANCE),
        "destination_lane_spread": bool(ROUTE_DESTINATION_LANE_SPREAD),
    }
    h.update(json.dumps(route_identity, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    return h.hexdigest()[:20]


def get_route_cache_path(gpkg_path, network_crs, num_routes, min_trip_distance, nodes, links, lanes, origin_nodes=None, destination_nodes=None):
    key = make_route_cache_key(gpkg_path, network_crs, num_routes, min_trip_distance, nodes, links, lanes, origin_nodes=origin_nodes, destination_nodes=destination_nodes)
    os.makedirs(ROUTE_CACHE_DIR, exist_ok=True)
    return os.path.join(ROUTE_CACHE_DIR, f"routes_{key}.npz")


def load_route_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        route_offsets = as_contig_i32(data["route_offsets"])
        route_lanes = as_contig_i32(data["route_lanes"])
        route_turns = as_contig_i32(data["route_turns"])
        if len(route_offsets) < 2 or len(route_lanes) != len(route_turns) or int(route_offsets[-1]) != len(route_lanes):
            print("[RouteCache] invalid cache. ignore:", cache_path)
            return None
        print("[RouteCache] loaded:", cache_path)
        print("[RouteCache] route count:", len(route_offsets) - 1)
        print("[RouteCache] lane elements:", len(route_lanes))
        return route_offsets, route_lanes, route_turns
    except Exception as e:
        print("[RouteCache] load failed:", e)
        return None


def save_route_cache(cache_path, route_offsets, route_lanes, route_turns, meta=None):
    try:
        ensure_parent(cache_path)
        if BACKUP_ROUTE_CACHE and os.path.exists(cache_path):
            backup_file_if_exists(cache_path)
        np.savez_compressed(
            cache_path,
            route_offsets=as_contig_i32(route_offsets),
            route_lanes=as_contig_i32(route_lanes),
            route_turns=as_contig_i32(route_turns),
            meta=json.dumps(meta or {}, ensure_ascii=False),
        )
        print("[RouteCache] saved:", cache_path)
    except Exception as e:
        print("[RouteCache] save failed:", e)


def make_routes_cached(G, nodes, links, lanes, link_to_lanes, origin_nodes, destination_nodes, gpkg_path, network_crs, num_routes):
    cache_path = get_route_cache_path(gpkg_path, network_crs, num_routes, MIN_TRIP_DISTANCE, nodes, links, lanes, origin_nodes=origin_nodes, destination_nodes=destination_nodes)
    if not REFRESH_ROUTE_CACHE:
        cached = load_route_cache(cache_path)
        if cached is not None:
            return cached
    print("[RouteCache] base route cache miss or refresh requested.")
    route_offsets, route_lanes, route_turns = make_routes_parallel(G, nodes, links, link_to_lanes, origin_nodes, destination_nodes, num_routes)
    save_route_cache(cache_path, route_offsets, route_lanes, route_turns, meta={
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": ROUTE_CACHE_VERSION,
        "kind": "base_routes",
        "num_routes": int(num_routes),
        "min_trip_distance": float(MIN_TRIP_DISTANCE),
        "nodes": len(nodes),
        "links": len(links),
        "lanes": len(lanes),
        "gpkg_path": str(gpkg_path),
        "network_crs": str(network_crs),
        "origin_nodes": len(origin_nodes),
        "destination_nodes": len(destination_nodes),
    })
    return route_offsets, route_lanes, route_turns


def get_ready_route_cache_path(gpkg_path, network_crs, num_routes, min_trip_distance, nodes, links, lanes, origin_nodes=None, destination_nodes=None):
    key = make_route_cache_key(
        gpkg_path, network_crs, num_routes, min_trip_distance, nodes, links, lanes,
        origin_nodes=origin_nodes, destination_nodes=destination_nodes,
    )
    os.makedirs(ROUTE_CACHE_DIR, exist_ok=True)
    return os.path.join(ROUTE_CACHE_DIR, f"routes_ready_{key}.npz")


def make_routes_ready_cached(G, nodes, links, lanes, link_to_lanes, origin_nodes, destination_nodes, gpkg_path, network_crs, num_routes):
    """Return CUDA-ready route arrays, caching expansion and lane-safety repair.

    EN: The base route cache stores only the requested NUM_ROUTES paths.  With
    multi-lane spawn balancing, those paths are then expanded into additional
    start-lane variants and sanitized before CUDA can use them.  That expansion
    can dominate startup when the base cache already exists, so v15 saves a
    second ready-route cache containing the expanded and repaired arrays.

    KO: base route cache는 NUM_ROUTES개의 기본 경로만 저장합니다. 다차로
    스폰 균등화를 켜면 시작 차로별 clone 확장과 lane 안전 보정을 한 번 더 거쳐야
    CUDA가 사용할 수 있습니다. v15부터 이 최종 결과를 ready route cache로 저장해
    다음 실행에서는 확장 단계를 다시 돌지 않습니다.
    """
    ready_cache_path = get_ready_route_cache_path(
        gpkg_path, network_crs, num_routes, MIN_TRIP_DISTANCE, nodes, links, lanes,
        origin_nodes=origin_nodes, destination_nodes=destination_nodes,
    )
    if ROUTE_READY_CACHE and not REFRESH_ROUTE_CACHE:
        cached = load_route_cache(ready_cache_path)
        if cached is not None:
            print("[RouteCache] ready expanded/sanitized routes loaded")
            return cached

    route_offsets, route_lanes, route_turns = make_routes_cached(
        G, nodes, links, lanes, link_to_lanes, origin_nodes, destination_nodes, gpkg_path, network_crs, num_routes
    )
    route_offsets, route_lanes, route_turns = expand_routes_for_multilane_spawn(
        route_offsets, route_lanes, route_turns, link_to_lanes, lanes, links
    )
    route_offsets, route_lanes, route_turns = sanitize_and_repair_route_arrays(
        route_offsets, route_lanes, route_turns, link_to_lanes, lanes, links
    )
    if ROUTE_READY_CACHE:
        save_route_cache(ready_cache_path, route_offsets, route_lanes, route_turns, meta={
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": ROUTE_CACHE_VERSION,
            "kind": "ready_expanded_sanitized_routes",
            "base_num_routes": int(num_routes),
            "ready_route_count": int(len(route_offsets) - 1),
            "min_trip_distance": float(MIN_TRIP_DISTANCE),
            "spawn_multi_lane_balance": bool(SPAWN_MULTI_LANE_BALANCE),
            "route_validate_lane_connectivity": bool(ROUTE_VALIDATE_LANE_CONNECTIVITY),
            "route_drop_invalid_lane_paths": bool(ROUTE_DROP_INVALID_LANE_PATHS),
            "route_wide_lane_change_as_straight": bool(ROUTE_WIDE_LANE_CHANGE_AS_STRAIGHT),
            "route_lane_count_change_balance": bool(ROUTE_LANE_COUNT_CHANGE_BALANCE),
            "route_destination_lane_spread": bool(ROUTE_DESTINATION_LANE_SPREAD),
            "interchange_edge_only": bool(INTERCHANGE_EDGE_ONLY),
            "nodes": len(nodes),
            "links": len(links),
            "lanes": len(lanes),
            "gpkg_path": str(gpkg_path),
            "network_crs": str(network_crs),
            "origin_nodes": len(origin_nodes),
            "destination_nodes": len(destination_nodes),
        })
    return route_offsets, route_lanes, route_turns


# ============================================================
# Signals
# ============================================================

def make_phase_windows(offset, cycle=90.0):
    straight_green = 32.0
    straight_yellow = 4.0
    left_green = 14.0
    left_yellow = 4.0
    st_g0 = offset % cycle
    st_g1 = (st_g0 + straight_green) % cycle
    st_y0 = st_g1
    st_y1 = (st_y0 + straight_yellow) % cycle
    lt_g0 = (st_y1 + 2.0) % cycle
    lt_g1 = (lt_g0 + left_green) % cycle
    lt_y0 = lt_g1
    lt_y1 = (lt_y0 + left_yellow) % cycle
    return {"cycle": cycle, "straight": (st_g0, st_g1, st_y0, st_y1), "left": (lt_g0, lt_g1, lt_y0, lt_y1)}


def build_signal_records(nodes, lanes, signal_points, signal_crs, network_crs):
    if not signal_points:
        print("[Signals] no signal points.")
        return []
    signal_points = transform_points(signal_points, signal_crs, network_crs)
    control_nodes = sorted({int(l["to_node"]) for l in lanes})
    tree, ids = build_node_tree(nodes, valid_nodes=control_nodes)
    records = []
    seen = set()

    def add(node, turn, win):
        key = (int(node), int(turn))
        if key in seen:
            return
        seen.add(key)
        p = nodes[int(node)]["geometry"]
        g0, g1, y0, y1 = win["left"] if turn == TURN_LEFT else win["straight"]
        records.append({
            "node": int(node), "turn": int(turn), "cycle": float(win["cycle"]),
            "green_start": float(g0), "green_end": float(g1),
            "yellow_start": float(y0), "yellow_end": float(y1),
            "x": float(p.x), "y": float(p.y),
        })

    matched = 0
    for idx, (x, y) in enumerate(signal_points):
        nearest = -1
        if tree is not None:
            d, tidx = tree.query([x, y], k=1)
            if float(d) <= SIGNAL_NODE_MATCH_MAX_DIST:
                nearest = int(ids[int(tidx)])
        if nearest < 0:
            continue
        raw = ((nearest * 1103515245 + idx * 12345) & 0x7fffffff) / float(0x7fffffff)
        win = make_phase_windows(raw * 50.0)
        add(nearest, TURN_STRAIGHT, win)
        add(nearest, TURN_LEFT, win)
        matched += 1
    print("[Signals] matched points:", matched)
    print("[Signals] records:", len(records))
    return records


def signal_records_to_numpy(records):
    if not records:
        return {k: as_contig_i32([]) if k in ["node", "turn"] else as_contig_f32([]) for k in [
            "node", "turn", "cycle", "green_start", "green_end", "yellow_start", "yellow_end"
        ]}
    return {
        "node": as_contig_i32([r["node"] for r in records]),
        "turn": as_contig_i32([r["turn"] for r in records]),
        "cycle": as_contig_f32([r["cycle"] for r in records]),
        "green_start": as_contig_f32([r["green_start"] for r in records]),
        "green_end": as_contig_f32([r["green_end"] for r in records]),
        "yellow_start": as_contig_f32([r["yellow_start"] for r in records]),
        "yellow_end": as_contig_f32([r["yellow_end"] for r in records]),
    }


# ============================================================
# OpenGL helpers
# ============================================================

def compute_map_transform(links, screen_w, screen_h, padding=50):
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for link in links:
        b = link["geometry"].bounds
        min_x = min(min_x, b[0])
        min_y = min(min_y, b[1])
        max_x = max(max_x, b[2])
        max_y = max(max_y, b[3])
    map_w = max(max_x - min_x, 1.0)
    map_h = max(max_y - min_y, 1.0)
    scale = min((screen_w - 2 * padding) / map_w, (screen_h - 2 * padding) / map_h)
    scale = max(scale, 1e-6)
    return min_x, min_y, max_x, max_y, scale, padding


def compute_world_grid(min_x, min_y, max_x, max_y, cell_size):
    margin = 20
    grid_w = int(math.ceil((max_x - min_x) / cell_size)) + margin * 2
    grid_h = int(math.ceil((max_y - min_y) / cell_size)) + margin * 2
    grid_w = max(16, min(grid_w, WORLD_GRID_MAX_W))
    grid_h = max(16, min(grid_h, WORLD_GRID_MAX_H))
    world_min_x = float(min_x) - margin * cell_size
    world_min_y = float(min_y) - margin * cell_size
    return world_min_x, world_min_y, float(cell_size), int(grid_w), int(grid_h)


def create_shader():
    from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
    from OpenGL.GL.shaders import compileProgram, compileShader
    vertex_src = """
    #version 330 core
    layout(location = 0) in vec2 in_pos;
    layout(location = 1) in vec4 in_color;
    layout(location = 2) in float in_size;
    out vec4 v_color;
    uniform vec2 u_cam;
    uniform float u_scale;
    uniform vec2 u_screen;
    uniform float u_padding;
    void main() {
        float sx = u_padding + (in_pos.x - u_cam.x) * u_scale;
        float sy_screen = u_screen.y - (u_padding + (in_pos.y - u_cam.y) * u_scale);
        float ndc_x = 2.0 * sx / u_screen.x - 1.0;
        float ndc_y = 1.0 - 2.0 * sy_screen / u_screen.y;
        gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
        gl_PointSize = max(in_size, 2.0);
        v_color = in_color;
    }
    """
    fragment_src = """
    #version 330 core
    in vec4 v_color;
    out vec4 FragColor;
    uniform int u_is_point;
    void main() {
        if (u_is_point == 1) {
            vec2 c = gl_PointCoord - vec2(0.5);
            if (dot(c, c) > 0.25) discard;
        }
        FragColor = v_color;
    }
    """
    return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))


def create_vehicle_texture_shader():
    """EN: Shader for textured vehicle quads with driver-type tinting.
    KO: 운전자 유형별 색상 tint를 적용하는 차량 텍스처 셰이더입니다.

    The CUDA VBO keeps the same 7-float layout: x, y, attr0, attr1, attr2,
    attr3, size.  attr0/attr1 are UV coordinates, attr2 is a driver flag
    (0 human, 1 AV), and attr3 is the amber indicator blink strength.

    CUDA VBO는 기존 7-float 레이아웃을 유지합니다. attr0/attr1은 UV,
    attr2는 운전자 flag(0 사람, 1 AV), attr3는 황색 방향지시등 점멸 강도입니다.
    사람 차량은 흰색, AV는 하늘색으로 보이고 깜빡이 상태는 amber tint로 표시합니다.
    """
    from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
    from OpenGL.GL.shaders import compileProgram, compileShader
    vertex_src = """
    #version 330 core
    layout(location = 0) in vec2 in_pos;
    layout(location = 1) in vec4 in_uv_misc;
    out vec2 v_uv;
    out float v_driver;
    out float v_signal;
    uniform vec2 u_cam;
    uniform float u_scale;
    uniform vec2 u_screen;
    uniform float u_padding;
    void main() {
        float sx = u_padding + (in_pos.x - u_cam.x) * u_scale;
        float sy_screen = u_screen.y - (u_padding + (in_pos.y - u_cam.y) * u_scale);
        float ndc_x = 2.0 * sx / u_screen.x - 1.0;
        float ndc_y = 1.0 - 2.0 * sy_screen / u_screen.y;
        gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
        v_uv = in_uv_misc.xy;
        v_driver = in_uv_misc.z;
        v_signal = in_uv_misc.w;
    }
    """
    fragment_src = """
    #version 330 core
    in vec2 v_uv;
    in float v_driver;
    in float v_signal;
    out vec4 FragColor;
    uniform sampler2D u_tex;
    void main() {
        vec4 c = texture(u_tex, v_uv);
        if (c.a < 0.02) discard;

        vec3 humanTint = vec3(1.00, 1.00, 1.00);
        vec3 avTint    = vec3(0.55, 0.84, 1.00);
        vec3 tint = mix(humanTint, avTint, step(0.5, v_driver));

        // EN: Preserve texture details by using luminance as shading.
        // KO: 텍스처 디테일은 밝기(luminance)로 살리고 차체색만 운전자 유형별로 바꿉니다.
        float luma = dot(c.rgb, vec3(0.299, 0.587, 0.114));
        float shade = clamp(luma * 1.28 + 0.06, 0.0, 1.0);
        vec3 detail = tint * shade;
        // Keep very dark pixels, such as tires/windows, dark.
        // 타이어/창문처럼 매우 어두운 픽셀은 검게 유지합니다.
        detail = mix(detail, c.rgb * 0.45, smoothstep(0.0, 0.18, 0.18 - luma));
        vec3 amber = vec3(1.00, 0.72, 0.08);
        detail = mix(detail, amber, clamp(v_signal, 0.0, 1.0) * 0.42);
        FragColor = vec4(detail, c.a);
    }
    """
    return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))


def create_ui_texture_shader():
    """EN: Screen-space textured quad shader for small hover panels.
    KO: 신호 hover 패널 같은 화면 좌표 텍스처 사각형용 셰이더입니다.
    """
    from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
    from OpenGL.GL.shaders import compileProgram, compileShader
    vertex_src = """
    #version 330 core
    layout(location = 0) in vec2 in_ndc;
    layout(location = 1) in vec2 in_uv;
    out vec2 v_uv;
    void main() {
        gl_Position = vec4(in_ndc, 0.0, 1.0);
        v_uv = in_uv;
    }
    """
    fragment_src = """
    #version 330 core
    in vec2 v_uv;
    out vec4 FragColor;
    uniform sampler2D u_tex;
    void main() {
        FragColor = texture(u_tex, v_uv);
    }
    """
    return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))


def load_vehicle_texture(texture_path):
    """EN: Load assets/car_topdown.png as an OpenGL texture.
    KO: assets/car_topdown.png 파일을 OpenGL 텍스처로 로드합니다.
    """
    import pygame
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexParameteri, glTexImage2D,
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_LINEAR, GL_CLAMP_TO_EDGE,
        GL_RGBA, GL_UNSIGNED_BYTE,
    )
    path = Path(texture_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"vehicle texture not found: {path}")
    surf = pygame.image.load(str(path)).convert_alpha()
    w, h = surf.get_size()
    # EN: flip=True gives OpenGL bottom-left texture coordinates.
    # KO: flip=True로 OpenGL의 bottom-left 텍스처 좌표계에 맞춥니다.
    data = pygame.image.tostring(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(w), int(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)
    print("[Render] vehicle texture:", str(path), "size:", w, "x", h)
    return tex


def create_ui_text_renderer():
    """EN: Allocate resources for hover text panel rendering.
    KO: hover 텍스트 패널 렌더링 리소스를 생성합니다.
    """
    import pygame
    from OpenGL.GL import (
        glGenVertexArrays, glGenBuffers, glGenTextures, glBindVertexArray, glBindBuffer,
        glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
        glBindTexture, glTexParameteri, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT,
        GL_FALSE, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_LINEAR, GL_CLAMP_TO_EDGE,
    )
    pygame.font.init()
    shader = create_ui_texture_shader()
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    tex = glGenTextures(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    font = pygame.font.SysFont(["malgungothic", "AppleGothic", "NanumGothic", "arial"], 16)
    return {"shader": shader, "vao": vao, "vbo": vbo, "tex": tex, "font": font}


def draw_text_panel(renderer, lines, mouse_xy, screen_w, screen_h):
    """EN: Draw a small multi-line text panel near the mouse.
    KO: 마우스 근처에 여러 줄 텍스트 패널을 그립니다.
    """
    if not lines:
        return
    import pygame
    from OpenGL.GL import (
        glUseProgram, glGetUniformLocation, glUniform1i, glActiveTexture, glBindTexture,
        glTexImage2D, glBindBuffer, glBufferSubData, glBindVertexArray, glDrawArrays,
        GL_TEXTURE0, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_ARRAY_BUFFER, GL_TRIANGLES,
    )
    font = renderer["font"]
    rendered = [font.render(str(line), True, (240, 240, 240, 255)) for line in lines]
    pad = 8
    line_gap = 3
    w = max(max((s.get_width() for s in rendered), default=1) + pad * 2, 220)
    h = sum(s.get_height() for s in rendered) + line_gap * max(0, len(rendered) - 1) + pad * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((20, 22, 24, 220))
    y = pad
    for rs in rendered:
        surf.blit(rs, (pad, y))
        y += rs.get_height() + line_gap
    data = pygame.image.tostring(surf, "RGBA", True)

    mx, my = mouse_xy
    x0 = min(max(int(mx) + 14, 4), max(4, int(screen_w) - w - 4))
    y0 = min(max(int(my) + 18, 4), max(4, int(screen_h) - h - 4))
    x1 = x0 + w
    y1 = y0 + h

    def ndc_x(px):
        return 2.0 * float(px) / float(screen_w) - 1.0
    def ndc_y(py):
        return 1.0 - 2.0 * float(py) / float(screen_h)

    # EN: 6 vertices, each: ndc.xy, uv.xy. KO: 6정점, 각 정점은 ndc.xy, uv.xy.
    verts = np.asarray([
        ndc_x(x0), ndc_y(y1), 0.0, 0.0,
        ndc_x(x1), ndc_y(y1), 1.0, 0.0,
        ndc_x(x1), ndc_y(y0), 1.0, 1.0,
        ndc_x(x0), ndc_y(y1), 0.0, 0.0,
        ndc_x(x1), ndc_y(y0), 1.0, 1.0,
        ndc_x(x0), ndc_y(y0), 0.0, 1.0,
    ], dtype=np.float32)

    glUseProgram(renderer["shader"])
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, renderer["tex"])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(w), int(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glUniform1i(glGetUniformLocation(renderer["shader"], "u_tex"), 0)
    glBindBuffer(GL_ARRAY_BUFFER, renderer["vbo"])
    glBufferSubData(GL_ARRAY_BUFFER, 0, int(verts.nbytes), verts)
    glBindVertexArray(renderer["vao"])
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)


def screen_to_world(mx, my, cam_x, cam_y, scale_zoom, padding, screen_h):
    """EN: Convert screen pixel to map/world coordinates.
    KO: 화면 픽셀 좌표를 지도/world 좌표로 변환합니다.
    """
    return (
        cam_x + (float(mx) - padding) / max(scale_zoom, 1.0e-9),
        cam_y + (float(screen_h) - float(my) - padding) / max(scale_zoom, 1.0e-9),
    )


def world_to_screen(wx, wy, cam_x, cam_y, scale_zoom, padding, screen_h):
    """EN: Convert world coordinates to screen pixels.
    KO: world 좌표를 화면 픽셀 좌표로 변환합니다.
    """
    sx = padding + (float(wx) - cam_x) * scale_zoom
    sy = float(screen_h) - (padding + (float(wy) - cam_y) * scale_zoom)
    return sx, sy


def phase_name(state):
    if state == LIGHT_GREEN:
        return "GREEN/파란불"
    if state == LIGHT_YELLOW:
        return "YELLOW/노란불"
    return "RED/빨간불"


def phase_interval_text(label, start, end, cycle):
    return f"{label}: {start:.1f}s -> {end:.1f}s / cycle {cycle:.1f}s"


def signal_red_interval_texts(r):
    """EN: Red windows are the parts of the cycle outside green/yellow.
    KO: 빨간불 구간은 cycle 중 파란불/노란불이 아닌 구간입니다.
    """
    cycle = max(float(r["cycle"]), 1.0)
    intervals = [
        (float(r["green_start"]) % cycle, float(r["green_end"]) % cycle),
        (float(r["yellow_start"]) % cycle, float(r["yellow_end"]) % cycle),
    ]
    # EN: unwrap intervals into [0, cycle). KO: wrap interval을 [0, cycle)로 풉니다.
    occ = []
    for a, b in intervals:
        if abs(a - b) < 1.0e-6:
            continue
        if a < b:
            occ.append((a, b))
        else:
            occ.append((a, cycle))
            occ.append((0.0, b))
    occ.sort()
    red = []
    cur = 0.0
    for a, b in occ:
        if a > cur + 1.0e-5:
            red.append((cur, a))
        cur = max(cur, b)
    if cur < cycle - 1.0e-5:
        red.append((cur, cycle))
    if not red:
        return ["  red/빨간불: none"]
    return [f"  red/빨간불: {a:.1f}s -> {b:.1f}s / cycle {cycle:.1f}s" for a, b in red]


def signal_time_to_next_change(t, r):
    """EN: Time until this signal record changes phase.
    KO: 해당 신호 record가 다음 phase로 바뀔 때까지 남은 시간입니다.
    """
    cycle = max(float(r["cycle"]), 1.0)
    p = float(t) % cycle
    points = [float(r["green_start"]), float(r["green_end"]), float(r["yellow_start"]), float(r["yellow_end"]), cycle]
    best = cycle
    for q in points:
        dt = (q - p) % cycle
        if dt > 1.0e-4:
            best = min(best, dt)
    return best


def signal_hover_lines(records, mouse_xy, cam_x, cam_y, scale_zoom, padding, screen_w, screen_h, current_time):
    """EN: Return hover panel lines for the nearest signal, or [] if none.
    KO: 가장 가까운 신호등 hover 패널 텍스트를 반환합니다. 없으면 []입니다.
    """
    if not records or not SHOW_SIGNAL_HOVER:
        return []
    mx, my = mouse_xy
    best_node = None
    best_d2 = float("inf")
    for r in records:
        sx, sy = world_to_screen(r["x"], r["y"], cam_x, cam_y, scale_zoom, padding, screen_h)
        dx = sx - float(mx)
        dy = sy - float(my)
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_node = int(r["node"])
    if best_node is None or best_d2 > SIGNAL_HOVER_RADIUS_PX * SIGNAL_HOVER_RADIUS_PX:
        return []
    rows = [r for r in records if int(r["node"]) == best_node]
    rows.sort(key=lambda r: int(r["turn"]))
    lines = [f"Signal node {best_node} / 신호 노드 {best_node}", f"sim time {format_sim_time(current_time)}"]
    for r in rows:
        turn = int(r["turn"])
        label = "LEFT / 좌회전" if turn == TURN_LEFT else "STRAIGHT+RIGHT / 직진+우회전"
        st = signal_state_at(current_time, r["cycle"], r["green_start"], r["green_end"], r["yellow_start"], r["yellow_end"])
        remain = signal_time_to_next_change(current_time, r)
        lines.append(f"{label}: {phase_name(st)}  next {remain:.1f}s")
        lines.append("  " + phase_interval_text("green/파란불", r["green_start"], r["green_end"], r["cycle"]))
        lines.append("  " + phase_interval_text("yellow/노란불", r["yellow_start"], r["yellow_end"], r["cycle"]))
        lines.extend(signal_red_interval_texts(r))
    return lines




def _vehicle_signal_label(sig):
    sig = int(sig)
    if sig == INDICATOR_LEFT:
        return "LEFT / 좌깜빡이"
    if sig == INDICATOR_RIGHT:
        return "RIGHT / 우깜빡이"
    if sig == INDICATOR_HAZARD:
        return "HAZARD / 비상등"
    return "OFF / 꺼짐"


def _vehicle_state_label(st):
    return "CONNECTOR / 교차로 통과" if int(st) == VEH_IN_CONNECTOR else "LANE / 차로 주행"


def vehicle_hover_lines(veh, mouse_xy, cam_x, cam_y, scale_zoom, padding, screen_w, screen_h, current_time):
    """EN: Return hover panel lines for the nearest vehicle.
    KO: 마우스 아래 가장 가까운 차량의 hover 패널 텍스트를 반환합니다.

    The nearest-neighbor query is done as one small torch reduction on GPU, then
    only the selected vehicle's component values are copied to CPU.
    가장 가까운 차량 탐색은 GPU torch reduction으로 수행하고, 선택된 차량 한 대의
    component 값만 CPU로 복사합니다.
    """
    if not SHOW_VEHICLE_HOVER:
        return []
    mx, my = mouse_xy
    wx, wy = screen_to_world(mx, my, cam_x, cam_y, scale_zoom, padding, screen_h)
    radius_world = VEHICLE_HOVER_RADIUS_PX / max(float(scale_zoom), 1.0e-9)
    if radius_world <= 0.0:
        return []
    try:
        with torch.no_grad():
            dx = veh["x"] - float(wx)
            dy = veh["y"] - float(wy)
            d2 = dx * dx + dy * dy
            active = veh["active"] == 1
            masked = torch.where(active, d2, torch.full_like(d2, float("inf")))
            best_d2_t, best_idx_t = torch.min(masked, dim=0)
            best_d2 = float(best_d2_t.detach().item())
            if not math.isfinite(best_d2) or best_d2 > radius_world * radius_world:
                return []
            idx = int(best_idx_t.detach().item())
            vals = {
                "id": idx,
                "x": float(veh["x"][idx].detach().item()),
                "y": float(veh["y"][idx].detach().item()),
                "s": float(veh["s"][idx].detach().item()),
                "speed": float(veh["speed"][idx].detach().item()),
                "accel": float(veh["accel"][idx].detach().item()),
                "heading": float(veh["heading"][idx].detach().item()),
                "steer": float(veh["steer_angle"][idx].detach().item()),
                "length": float(veh["vehicle_length"][idx].detach().item()),
                "width": float(veh["vehicle_width"][idx].detach().item()),
                "lane": int(veh["lane_id"][idx].detach().item()),
                "route": int(veh["route_id"][idx].detach().item()),
                "route_pos": int(veh["route_pos"][idx].detach().item()),
                "state": int(veh["vehicle_state"][idx].detach().item()),
                "driver": int(veh["driver_type"][idx].detach().item()),
                "lc": int(veh["lane_change_active"][idx].detach().item()),
                "lc_to": int(veh["lane_change_to_lane"][idx].detach().item()),
                "conn_from": int(veh["connector_from_lane"][idx].detach().item()),
                "conn_to": int(veh["connector_to_lane"][idx].detach().item()),
                "conn_s": float(veh["connector_s"][idx].detach().item()),
                "signal": int(veh["turn_signal"][idx].detach().item()),
                "signal_time": float(veh["turn_signal_time"][idx].detach().item()),
                "aggr": float(veh["aggressiveness"][idx].detach().item()),
                "polite": float(veh["politeness"][idx].detach().item()),
                "risk": float(veh["risk_tolerance"][idx].detach().item()),
            }
    except Exception:
        return []

    driver = "HUMAN / 사람" if vals["driver"] == 0 else "AV / 자율주행"
    lines = [
        f"Vehicle {vals['id']} / 차량 {vals['id']}",
        f"driver: {driver}",
        f"state: {_vehicle_state_label(vals['state'])}",
        f"speed: {vals['speed']:.2f} m/s  accel: {vals['accel']:.2f} m/s²",
        f"lane: {vals['lane']}  route: {vals['route']}:{vals['route_pos']}  s={vals['s']:.1f}m",
        f"indicator: {_vehicle_signal_label(vals['signal'])}  {vals['signal_time']:.1f}s",
        f"heading: {math.degrees(vals['heading']):.1f}°  steer: {math.degrees(vals['steer']):.1f}°",
        f"size: {vals['length']:.1f}m x {vals['width']:.1f}m",
        f"AI/personality: aggr={vals['aggr']:.2f} polite={vals['polite']:.2f} risk={vals['risk']:.2f}",
    ]
    if vals["lc"] != 0:
        lines.append(f"lane change -> {vals['lc_to']} / 차선변경 중")
    if vals["state"] == VEH_IN_CONNECTOR:
        lines.append(f"connector: {vals['conn_from']} -> {vals['conn_to']}  s={vals['conn_s']:.1f}m")
    lines.append(f"sim time {format_sim_time(current_time)}")
    return lines

def road_vertices_for_chunk(chunk):
    """EN: Build filled road-surface triangles from link width.
    KO: 링크 폭을 이용해 선형 도로를 면(삼각형)으로 렌더링합니다.
    """
    verts = []
    for link in chunk:
        geom = link.get("render_geometry", link.get("geometry"))
        coords = list(geom.coords) if geom is not None else []
        if len(coords) < 2:
            continue
        half_w = float(link["width"]) * 0.5
        for a, b in zip(coords[:-1], coords[1:]):
            dx = float(b[0]) - float(a[0])
            dy = float(b[1]) - float(a[1])
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            nx = -dy / L
            ny = dx / L
            p0 = (a[0] + nx * half_w, a[1] + ny * half_w)
            p1 = (a[0] - nx * half_w, a[1] - ny * half_w)
            p2 = (b[0] + nx * half_w, b[1] + ny * half_w)
            p3 = (b[0] - nx * half_w, b[1] - ny * half_w)
            def add(p):
                verts.extend([p[0], p[1], 0.25, 0.25, 0.25, 1.0, 1.0])
            add(p0); add(p1); add(p2); add(p1); add(p2); add(p3)
    return verts


def intersection_apron_vertices(lanes, nodes):
    """EN: Paint rounded inner curb fillets at intersection nodes.

    The old visual treatment drew semicircle fans from the node center.  That
    made intersections look as if a circular island or disk existed in the
    middle.  This renderer instead finds adjacent road arms and adds only the
    compact corner patch between their lane-surface edges.  Existing road
    rectangles still draw the straight arms and central overlap; these extra
    triangles only round the green/asphalt inside corners like the reference
    image.

    KO: 교차로 node에 안쪽 연석 곡선 필렛을 그립니다.

    이전 방식은 node 중심에서 반원 fan을 그려서 교차로 한가운데 원반이 있는 것처럼
    보였습니다. 새 방식은 인접한 도로 팔을 찾아 두 도로 면의 가장자리 사이 코너에만
    작은 곡선 패치를 추가합니다. 직선 도로와 중앙 겹침은 기존 사각 도로 면이 그리며,
    여기서는 참고 이미지처럼 녹지/아스팔트 모서리만 둥글게 이어 줍니다.
    """
    if not DRAW_INTERSECTION_APRONS or not lanes:
        return []

    node_xy = _node_lookup_from_nodes(nodes)
    raw_incident = {}

    def norm_angle(a):
        a = math.fmod(float(a), 2.0 * math.pi)
        if a < 0.0:
            a += 2.0 * math.pi
        return a

    def angle_diff_abs(a, b):
        d = abs(norm_angle(a - b))
        return min(d, 2.0 * math.pi - d)

    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    def line_intersection(px, py, ux, uy, qx, qy, vx, vy):
        den = cross(ux, uy, vx, vy)
        if abs(den) < 1.0e-6:
            return None
        t = cross(qx - px, qy - py, vx, vy) / den
        return (px + ux * t, py + uy * t)

    # EN/KO: Collect one outward road-arm direction per lane endpoint.
    for lane in lanes:
        try:
            lw = max(2.4, float(lane.get("lane_width", DEFAULT_LANE_WIDTH)))
            half_w = lw * 0.5
            sx0 = float(lane["start_x"])
            sy0 = float(lane["start_y"])
            sx1 = float(lane["end_x"])
            sy1 = float(lane["end_y"])
            h = math.atan2(sy1 - sy0, sx1 - sx0)
            entries = (
                (int(lane["from_node"]), h),
                (int(lane["to_node"]), h + math.pi),
            )
            for nid, outward_h in entries:
                if nid in node_xy:
                    raw_incident.setdefault(nid, []).append((norm_angle(outward_h), half_w))
        except Exception:
            continue

    verts = []
    color = (0.25, 0.25, 0.25, 1.0, 1.0)
    segs_total = max(12, int(INTERSECTION_APRON_SEGMENTS))
    merge_tol = math.radians(14.0)
    min_gap = math.radians(max(5.0, float(INTERSECTION_APRON_MIN_GAP_DEG)))
    max_gap = math.radians(158.0)

    def add_vertex(x, y):
        verts.extend([float(x), float(y), color[0], color[1], color[2], color[3], color[4]])

    def add_tri(a, b, c):
        add_vertex(a[0], a[1])
        add_vertex(b[0], b[1])
        add_vertex(c[0], c[1])

    for nid, vals in raw_incident.items():
        if len(vals) < 2:
            continue
        cx, cy = node_xy[nid]

        # EN: Merge parallel lanes of the same physical arm by vector average.
        # KO: 같은 도로 팔에 속한 평행 차선들을 방향 평균으로 병합합니다.
        vals = sorted(vals, key=lambda z: z[0])
        groups = []
        for h, hw in vals:
            placed = False
            for g in groups:
                gh = math.atan2(g["sy"], g["sx"])
                if angle_diff_abs(h, gh) <= merge_tol:
                    g["sx"] += math.cos(h)
                    g["sy"] += math.sin(h)
                    g["half_w"] = max(g["half_w"], hw)
                    g["count"] += 1
                    placed = True
                    break
            if not placed:
                groups.append({"sx": math.cos(h), "sy": math.sin(h), "half_w": hw, "count": 1})

        arms = []
        for g in groups:
            h = norm_angle(math.atan2(g["sy"], g["sx"]))
            arms.append((h, float(g["half_w"])))
        arms = sorted(arms, key=lambda z: z[0])
        if len(arms) < 2:
            continue

        # Optional micro fill for numerical cracks only.  Default is 0 so no
        # visible center disk is drawn.
        max_half = max(hw for _, hw in arms)
        center_fill = max_half * max(0.0, float(INTERSECTION_APRON_CENTER_FILL_MULT))
        if center_fill > 0.05:
            csegs = max(8, segs_total // 2)
            for k in range(csegs):
                a0 = 2.0 * math.pi * (k / csegs)
                a1 = 2.0 * math.pi * ((k + 1) / csegs)
                add_tri(
                    (cx, cy),
                    (cx + math.cos(a0) * center_fill, cy + math.sin(a0) * center_fill),
                    (cx + math.cos(a1) * center_fill, cy + math.sin(a1) * center_fill),
                )

        n = len(arms)
        for idx in range(n):
            h0, hw0 = arms[idx]
            h1, hw1 = arms[(idx + 1) % n]
            gap = norm_angle(h1 - h0)
            if gap < min_gap or gap > max_gap:
                continue

            u0 = (math.cos(h0), math.sin(h0))
            u1 = (math.cos(h1), math.sin(h1))

            # For the counter-clockwise gap h0->h1, the facing edges are the
            # left edge of arm0 and the right edge of arm1.
            n0 = (-u0[1], u0[0])
            n1 = (u1[1], -u1[0])

            p0_line = (cx + n0[0] * hw0, cy + n0[1] * hw0)
            p1_line = (cx + n1[0] * hw1, cy + n1[1] * hw1)
            inner = line_intersection(
                p0_line[0], p0_line[1], u0[0], u0[1],
                p1_line[0], p1_line[1], u1[0], u1[1],
            )
            if inner is None:
                continue
            ix, iy = inner
            if not (math.isfinite(ix) and math.isfinite(iy)):
                continue
            if math.hypot(ix - cx, iy - cy) > max(60.0, 8.0 * max_half):
                continue

            radius = max(hw0, hw1) * float(INTERSECTION_APRON_RADIUS_MULT) + float(INTERSECTION_APRON_EXTRA)
            radius = max(max(hw0, hw1) * 0.65, radius)
            radius = min(radius, float(INTERSECTION_APRON_MAX_RADIUS))
            if not math.isfinite(radius) or radius <= 0.10:
                continue

            # Tangent points along the two road-edge lines.  The distances are
            # scaled mildly by the angle so skewed Y/T intersections remain compact.
            angle_scale = max(0.65, min(1.35, 1.0 / max(math.sin(gap * 0.5), 0.55)))
            tangent = radius * angle_scale
            a = (ix + u0[0] * tangent, iy + u0[1] * tangent)
            b = (ix + u1[0] * tangent, iy + u1[1] * tangent)

            # Quadratic Bezier control point, pushed outward into the corner gap.
            outx = n0[0] + n1[0]
            outy = n0[1] + n1[1]
            out_len = math.hypot(outx, outy)
            if out_len < 1.0e-6:
                continue
            outx /= out_len
            outy /= out_len
            control_gain = radius * (0.85 + 0.25 * min(1.0, gap / (0.5 * math.pi)))
            ctrl = (ix + outx * control_gain, iy + outy * control_gain)

            arc_segs = max(5, int(segs_total * gap / (2.0 * math.pi)) + 2)
            prev = a
            for k in range(1, arc_segs + 1):
                t = k / arc_segs
                omt = 1.0 - t
                q = (
                    omt * omt * a[0] + 2.0 * omt * t * ctrl[0] + t * t * b[0],
                    omt * omt * a[1] + 2.0 * omt * t * ctrl[1] + t * t * b[1],
                )
                add_tri(inner, prev, q)
                prev = q

    return verts


def _append_render_vertex(verts, x, y, r, g, b, a=1.0, size=1.0):
    """EN: Append one vertex in the shared shader format.
    KO: 공통 셰이더 포맷(x, y, rgba, size)에 맞게 정점 하나를 추가합니다.
    """
    verts.extend([float(x), float(y), float(r), float(g), float(b), float(a), float(size)])


def _append_solid_line(verts, ax, ay, bx, by, color):
    """EN: Append one continuous GL_LINES segment.
    KO: 끊기지 않는 GL_LINES 선분 하나를 추가합니다.
    """
    r, g, b, a = color
    _append_render_vertex(verts, ax, ay, r, g, b, a, 1.0)
    _append_render_vertex(verts, bx, by, r, g, b, a, 1.0)


def _append_dashed_line(verts, ax, ay, bx, by, color, dash_len=None, gap_len=None):
    """EN: Append a dashed road lane line in world-space meters.
    KO: world 좌표 meter 기준의 흰색 점선 차선을 추가합니다.
    """
    dash_len = float(LANE_DASH_LENGTH if dash_len is None else dash_len)
    gap_len = float(LANE_DASH_GAP if gap_len is None else gap_len)
    dx = float(bx) - float(ax)
    dy = float(by) - float(ay)
    L = math.hypot(dx, dy)
    if L < 1.0e-6:
        return
    ux = dx / L
    uy = dy / L
    step = max(0.5, dash_len + gap_len)
    pos = 0.0
    while pos < L:
        e = min(L, pos + dash_len)
        if e > pos + 0.05:
            _append_solid_line(
                verts,
                ax + ux * pos,
                ay + uy * pos,
                ax + ux * e,
                ay + uy * e,
                color,
            )
        pos += step


def _node_lookup_from_nodes(nodes):
    return {int(n["node_id"]): (float(n["geometry"].x), float(n["geometry"].y)) for n in nodes}


def _axis_from_lane_group(group, node_lookup):
    """EN: Return one physical road-axis segment for a lane bundle.
    KO: 같은 물리 도로 구간의 차로 묶음에서 도로 중심축 선분을 반환합니다.
    """
    a = int(group[0]["from_node"])
    b = int(group[0]["to_node"])
    key = tuple(sorted((a, b)))
    if key[0] in node_lookup and key[1] in node_lookup:
        x0, y0 = node_lookup[key[0]]
        x1, y1 = node_lookup[key[1]]
    else:
        # EN: Fallback: average lane endpoints. KO: fallback으로 차로 끝점을 평균냅니다.
        x0 = sum(float(z["start_x"]) for z in group) / max(1, len(group))
        y0 = sum(float(z["start_y"]) for z in group) / max(1, len(group))
        x1 = sum(float(z["end_x"]) for z in group) / max(1, len(group))
        y1 = sum(float(z["end_y"]) for z in group) / max(1, len(group))
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1.0e-6:
        return None
    return x0, y0, x1, y1, dx / L, dy / L, -dy / L, dx / L


def _cluster_offsets(offsets, tol=0.32):
    """EN: Merge nearly identical lane-boundary offsets.
    KO: 서로 거의 같은 차선 경계 offset을 하나로 병합합니다.
    """
    if not offsets:
        return []
    vals = sorted(float(x) for x in offsets if math.isfinite(float(x)))
    if not vals:
        return []
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - (sum(clusters[-1]) / len(clusters[-1]))) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def lane_marking_vertices_for_group(group, node_lookup):
    """EN: Build realistic lane paint for one physical road segment.
    KO: 하나의 물리 도로 구간에 대해 실제 도로와 비슷한 차선 표시를 만듭니다.

    Updated rule / 수정된 규칙:
    - Opposite-direction divider: orange solid line / 반대 방향 중앙선: 주황색 실선
    - Same-direction separators on BOTH directions: white dashed lines
      / 양쪽 진행 방향의 같은 방향 차로 경계는 모두 흰색 점선
    - Road edges: subtle solid white / 도로 외곽선은 연한 흰색 실선

    The previous implementation clustered all boundary offsets first.  With
    asymmetric GIS lane offsets, one side's internal separator could be mistaken
    for an outer edge and not be dashed.  This version sorts lane centers and
    draws the boundary between every adjacent lane pair, so reverse-direction
    lane separators are dashed exactly like forward-direction separators.

    이전 구현은 모든 경계 offset을 먼저 cluster 했기 때문에 GIS 차선 offset이
    비대칭이면 반대편 내부 차선 경계가 외곽선으로 오인되어 점선이 빠질 수
    있었습니다. 이제 차로 중심을 offset 순서로 정렬하고 인접한 두 차로 사이의
    경계를 직접 그려, 양방향 모든 같은 방향 차선 경계가 흰색 점선으로 나옵니다.
    """
    axis = _axis_from_lane_group(group, node_lookup)
    if axis is None:
        return []
    x0, y0, x1, y1, ux, uy, nx, ny = axis
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)

    lanes_info = []
    for lane in group:
        lx = 0.5 * (float(lane["start_x"]) + float(lane["end_x"]))
        ly = 0.5 * (float(lane["start_y"]) + float(lane["end_y"]))
        off = (lx - mx) * nx + (ly - my) * ny
        lw = max(2.4, float(lane.get("lane_width", DEFAULT_LANE_WIDTH)))
        # EN: Prefer the explicit physical direction stored during lane generation.
        #     The old dot-product sign depended on arbitrary node ordering; on some
        #     links one travel direction was misclassified and its separator became
        #     a solid/edge line instead of a white dashed lane line.
        # KO: 차로 생성 때 저장한 실제 진행 방향을 우선 사용합니다. 예전 dot-product
        #     방식은 노드 번호/축 방향에 따라 한쪽 진행 방향이 잘못 분류되어 점선이
        #     빠지는 문제가 있었습니다.
        sign = int(lane.get("direction", 0))
        if sign == 0:
            ldx = float(lane["end_x"]) - float(lane["start_x"])
            ldy = float(lane["end_y"]) - float(lane["start_y"])
            ln = math.hypot(ldx, ldy)
            sign = 1
            if ln > 1.0e-6:
                sign = 1 if (ldx / ln) * ux + (ldy / ln) * uy >= 0.0 else -1
        lanes_info.append({"off": off, "width": lw, "sign": sign})

    if not lanes_info:
        return []
    lanes_info.sort(key=lambda z: z["off"])

    verts = []

    def endpoints_at_offset(off):
        return x0 + nx * off, y0 + ny * off, x1 + nx * off, y1 + ny * off

    # EN/KO: Outer road edges from the outside of the extreme lane surfaces.
    min_edge = min(z["off"] - 0.5 * z["width"] for z in lanes_info)
    max_edge = max(z["off"] + 0.5 * z["width"] for z in lanes_info)
    if DRAW_LANE_EDGES:
        for off in (min_edge, max_edge):
            ax, ay, bx, by = endpoints_at_offset(off)
            _append_solid_line(verts, ax, ay, bx, by, LANE_EDGE_COLOR)

    # EN: Draw a boundary between each adjacent lane center.  Direction change
    #     means the yellow center divider; same direction means dashed white.
    # KO: 인접 차로 중심 사이마다 차선을 그립니다. 진행 방향이 바뀌면 주황색 중앙선,
    #     같은 방향이면 흰색 점선입니다.
    for a, b in zip(lanes_info[:-1], lanes_info[1:]):
        off = 0.5 * (a["off"] + b["off"])
        ax, ay, bx, by = endpoints_at_offset(off)
        if int(a["sign"]) != int(b["sign"]):
            _append_solid_line(verts, ax, ay, bx, by, LANE_CENTER_DIVIDER_COLOR)
        else:
            if DRAW_LANE_CENTERLINES:
                _append_dashed_line(verts, ax, ay, bx, by, LANE_SEPARATOR_COLOR)

    return verts

def build_lane_marking_vertex_array(lanes, nodes):
    """EN: Build one static VBO array for realistic lane markings.
    KO: 실제 도로에 가까운 차선 표시용 정적 VBO 배열을 생성합니다.
    """
    if not lanes or not DRAW_LANE_MARKINGS:
        return np.zeros((0,), dtype=np.float32)
    node_lookup = _node_lookup_from_nodes(nodes)
    groups = {}
    for lane in lanes:
        if LANE_MARKING_GROUP_BY_SOURCE and "source_row" in lane and "source_segment" in lane:
            # EN: Physical road segment key. Both directions created from the same
            #     original row/part/segment share this key. source_part is included
            #     so MultiLineString parts never get merged accidentally.
            # KO: 물리 도로 구간 key입니다. 원본 row/part/segment가 같은 차로만
            #     같은 그룹으로 묶습니다. source_part를 포함해 MultiLineString의
            #     서로 떨어진 part가 섞이는 것도 방지합니다.
            key = (
                "src",
                int(lane.get("source_row", -1)),
                int(lane.get("source_part", 0)),
                int(lane.get("source_segment", -1)),
            )
        else:
            key = tuple(sorted((int(lane["from_node"]), int(lane["to_node"]))))
        groups.setdefault(key, []).append(lane)
    all_verts = []
    for group in groups.values():
        all_verts.extend(lane_marking_vertices_for_group(group, node_lookup))
    return as_contig_f32(all_verts)

def build_road_vertex_array(links, lanes=None, nodes=None):
    if not links:
        return np.zeros((0,), dtype=np.float32)
    workers = max(1, min(os.cpu_count() or 4, 8, len(links)))
    chunk_size = int(math.ceil(len(links) / workers))
    chunks = [links[i:i + chunk_size] for i in range(0, len(links), chunk_size)]
    all_verts = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(road_vertices_for_chunk, c) for c in chunks]
        for fut in as_completed(futures):
            all_verts.extend(fut.result())

    # EN: Add curved intersection asphalt after the straight road strips.
    # KO: 직선 도로 면 위에 곡선형 교차로 apron을 추가합니다.
    if lanes is not None and nodes is not None:
        all_verts.extend(intersection_apron_vertices(lanes, nodes))

    return as_contig_f32(all_verts)


def create_static_vao_vbo(arr):
    from OpenGL.GL import (
        glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer, GL_ARRAY_BUFFER, GL_STATIC_DRAW,
        GL_FLOAT, GL_FALSE,
    )
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    stride = 7 * 4
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, int(arr.nbytes), arr, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo, len(arr) // 7


def create_dynamic_vao_vbo(vertex_count):
    from OpenGL.GL import (
        glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
        GL_FLOAT, GL_FALSE,
    )
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    stride = 7 * 4
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, int(vertex_count) * stride, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo


def signal_state_at(t, cycle, g0, g1, y0, y1):
    if cycle <= 1.0:
        return LIGHT_GREEN
    p = t % cycle
    def inside(x, a, b):
        if a <= b:
            return a <= x < b
        return x >= a or x < b
    if inside(p, g0, g1):
        return LIGHT_GREEN
    if inside(p, y0, y1):
        return LIGHT_YELLOW
    return LIGHT_RED


def update_signal_vbo(vbo, records, current_time):
    from OpenGL.GL import glBindBuffer, glBufferSubData, GL_ARRAY_BUFFER
    verts = []
    for r in records:
        state = signal_state_at(current_time, r["cycle"], r["green_start"], r["green_end"], r["yellow_start"], r["yellow_end"])
        if state == LIGHT_GREEN:
            color = (0.0, 1.0, 0.15, 1.0)
        elif state == LIGHT_YELLOW:
            color = (1.0, 0.85, 0.05, 1.0)
        else:
            color = (1.0, 0.05, 0.05, 1.0)
        size = 10.0 if int(r["turn"]) == TURN_LEFT else 7.0
        verts.extend([r["x"], r["y"], color[0], color[1], color[2], color[3], size])
    arr = as_contig_f32(verts)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    if arr.nbytes > 0:
        glBufferSubData(GL_ARRAY_BUFFER, 0, int(arr.nbytes), arr)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# ============================================================
# Metrics
# ============================================================

def metrics_to_snapshot(m):
    def get(idx, default=0.0):
        if idx < len(m):
            return float(m[idx])
        return float(default)

    def count(idx):
        return max(get(idx), 1.0)

    def avg(sum_idx, count_idx):
        return get(sum_idx) / count(count_idx)

    active = max(get(7), 0.0)

    # Indices 0 and 1 are cumulative. Most per-frame metrics are reset by
    # the CUDA core every step from index 6 onward.
    return {
        "spawned": int(get(0)),
        "completed": int(get(1)),
        "active": int(active),
        "travel_time_sum": get(6),
        "avg_speed": avg(14, 15),
        "avg_accel": avg(9, 8),
        "avg_decel": avg(12, 11),
        "slow_count": int(get(16)),
        "stop_count": int(get(19)),
        "collision": int(get(20)),
        "rejected_spawn": int(get(22)),
        "connector_enter": int(get(25)),
        "connector_active": int(get(26)),
        "reservation_accept": int(get(34)),
        "reservation_reject": int(get(35)),
        "lane_change": int(get(36)),
        "rejected_lane_change": int(get(37)),
        "ttc_critical": int(get(38)),
        "ttc_warning": int(get(39)),
        "hard_brake": int(get(40)),
        "near_miss": int(get(41)),
        "coop_yield": int(get(42)),
        "mobil_eval": int(get(43)),

        # Metrics added by the spawnfix / bicycle-model CUDA core.
        "avg_delay": avg(44, 45),
        "avg_reaction_time": avg(46, 47),
        "avg_response_lag": avg(48, 49),
        "avg_abs_steer": avg(50, 51),
        "avg_abs_yaw_rate": avg(52, 53),
        "avg_headway": avg(54, 55),
        "avg_min_gap": avg(56, 57),
        "intersection_wait": get(58),
        "red_light_stop": int(get(59)),
        "yellow_stop": int(get(60)),
        "yellow_go": int(get(61)),
        "red_light_violation": int(get(62)),
        "sensor_detection": int(get(63)),
        "sensor_front_hit": int(get(64)),
        "conflict_yield": int(get(65)),
        "interaction_brake": int(get(66)),
        "avg_queue_delay": avg(67, 68),
        "avg_lane_change_time": avg(69, 70),
        "avg_connector_delay": avg(71, 72),
        "comfort_brake": int(get(73)),
        "standstill_time": get(74),
        "avg_time_loss": avg(75, 76),
        "turn_lane_prep": int(get(77)),
        "turn_lane_block": int(get(78)),
        "turn_lane_illegal": int(get(79)),
        "unsignal_right_yield": int(get(80)),
        "unsignal_priority_go": int(get(81)),
        "unsignal_conflict": int(get(82)),
        "deadlock_wait": get(83),
        "deadlock_release": int(get(84)),
        "deadlock_creep": int(get(85)),
        "connector_safe_yield": int(get(86)),
        "priority_entry_block": int(get(87)),
        "indicator_left_on": int(get(88)),
        "indicator_right_on": int(get(89)),
        "indicator_conflict_yield": int(get(90)),
        "indicator_priority_go": int(get(91)),
        "anti_collision_brake": int(get(92)),
        "penetration_prevented": int(get(93)),
        "connector_cross_yield": int(get(94)),
        "deadlock_escape_go": int(get(95)),
        "priority_gate_candidate": int(get(96)),
        "priority_gate_granted": int(get(97)),
        "priority_gate_blocked": int(get(98)),
        "intersection_occupied_hold": int(get(99)),
        "force_pass_through": int(get(100)),
        "unique_priority_tie": int(get(101)),
        "deadlock_priority_release": int(get(102)),
        "entry_queue_hold": get(103),
        "priority_conflict_free_go": int(get(104)),
        "priority_path_block": int(get(105)),
        "priority_active_path_hold": int(get(106)),
        "human_ai_assertive_go": int(get(107)),
        "human_ai_courtesy_yield": int(get(108)),
        "right_turn_symmetric_path": int(get(109)),
        "right_turn_exit_gap_hold": int(get(110)),
        "front_space_release": int(get(111)),
    }


def validate_route_arrays(route_offsets_np, route_lanes_np, route_turns_np, num_lanes):
    if len(route_offsets_np) < 2:
        raise RuntimeError("route_offsets must have at least 2 elements")
    if len(route_lanes_np) != len(route_turns_np):
        raise RuntimeError("route_lanes and route_turns length mismatch")
    if int(route_offsets_np[-1]) != len(route_lanes_np):
        raise RuntimeError("route_offsets[-1] must equal len(route_lanes)")
    if len(route_lanes_np) > 0:
        mn = int(np.min(route_lanes_np))
        mx = int(np.max(route_lanes_np))
        if mn < 0 or mx >= num_lanes:
            raise RuntimeError(f"route_lanes out of range: min={mn}, max={mx}, num_lanes={num_lanes}")


def lane_connected_py(lanes, from_lane_id, to_lane_id):
    if not (0 <= int(from_lane_id) < len(lanes) and 0 <= int(to_lane_id) < len(lanes)):
        return False
    return int(lanes[int(from_lane_id)].get("to_node", -999999)) == int(lanes[int(to_lane_id)].get("from_node", 999999))


def same_lane_group_py(lanes, a, b):
    if not (0 <= int(a) < len(lanes) and 0 <= int(b) < len(lanes)):
        return False
    la = lanes[int(a)]
    lb = lanes[int(b)]
    return int(la.get("from_node", -1)) == int(lb.get("from_node", -2)) and int(la.get("to_node", -3)) == int(lb.get("to_node", -4))


def route_lane_repair_candidate(current_lane, desired_next_lane, link_to_lanes, lanes, previous_turn=TURN_STRAIGHT, fallback_turn=TURN_STRAIGHT, key=0):
    """Return a connected receiving lane for a route step, or -1 if impossible.

    EN: This is a CPU-side guard against a corrupt/stale route cache assigning a
    lane that does not physically begin at the current lane's end node.  It first
    tries the intended next link's lane bundle, then falls back to every lane that
    starts at the same node.  The returned lane is always connector-reachable from
    current_lane.

    KO: 오래된 route cache나 차선 보정 과정에서 현재 차로 끝 노드와 물리적으로
    이어지지 않는 lane이 들어가면 차량이 도로 중간/끝에서 멈출 수 있습니다. 먼저
    원래 next link의 차로 묶음에서 고치고, 실패하면 같은 노드에서 시작하는 차로 중
    연결 가능한 차로로 복구합니다. 반환값은 항상 current_lane에서 connector로 진입
    가능한 lane입니다.
    """
    current_lane = int(current_lane)
    desired_next_lane = int(desired_next_lane)
    if not (0 <= current_lane < len(lanes)):
        return -1

    if 0 <= desired_next_lane < len(lanes) and lane_connected_py(lanes, current_lane, desired_next_lane):
        return desired_next_lane

    candidate_groups = []
    if 0 <= desired_next_lane < len(lanes):
        target_link = int(lanes[desired_next_lane].get("link_id", -1))
        group = link_to_lanes.get(target_link, [])
        if group:
            candidate_groups.append(group)

    # Fallback: all outgoing lanes from the current end node, grouped by link.
    end_node = int(lanes[current_lane].get("to_node", -1))
    seen_links = set()
    for lane in lanes:
        if int(lane.get("from_node", -2)) != end_node:
            continue
        lid = int(lane.get("link_id", -1))
        if lid in seen_links:
            continue
        seen_links.add(lid)
        group = link_to_lanes.get(lid, [])
        if group:
            candidate_groups.append(group)

    previous_group = link_to_lanes.get(int(lanes[current_lane].get("link_id", -1)), [])
    for group in candidate_groups:
        edge_receive = interchange_receiving_outer_lane_id(previous_group, group)
        if edge_receive >= 0 and lane_connected_py(lanes, current_lane, edge_receive):
            return int(edge_receive)

        candidate = receiving_lane_after_turn_balanced(
            group,
            previous_turn,
            previous_group=previous_group,
            previous_lane_id=current_lane,
            fallback_turn=fallback_turn,
            key=key,
        )
        if candidate >= 0 and lane_connected_py(lanes, current_lane, candidate):
            return int(candidate)

        for lane in group:
            lid = int(lane.get("lane_id", -1))
            if lane_connected_py(lanes, current_lane, lid):
                return lid

    return -1


def sanitize_and_repair_route_arrays(route_offsets_np, route_lanes_np, route_turns_np, link_to_lanes, lanes, links):
    """Repair or drop impossible lane steps before sending routes to CUDA.

    EN: A vehicle should never receive a route step where lane A cannot connect
    to lane B.  Such a mismatch can make it stop at an artificial lane deadline.
    This pass is intentionally conservative: it repairs the receiving lane inside
    the intended next link when possible, truncates only irreparable tails, and
    drops only routes shorter than two valid lane segments.

    KO: 차량 route에 A 차로에서 B 차로로 물리적으로 갈 수 없는 단계가 들어가면
    주행 중 잘못된 mandatory lane 대기로 멈출 수 있습니다. 이 검사는 CUDA로 넘기기
    전에 next link 내부에서 가능한 수신 차로로 고치고, 불가능한 꼬리만 잘라냅니다.
    """
    route_offsets_np = np.asarray(route_offsets_np, dtype=np.int32)
    route_lanes_np = np.asarray(route_lanes_np, dtype=np.int32)
    route_turns_np = np.asarray(route_turns_np, dtype=np.int32)

    if not ROUTE_VALIDATE_LANE_CONNECTIVITY or not ROUTE_DROP_INVALID_LANE_PATHS:
        print("[Routes] lane safety repair: disabled by config")
        return as_contig_i32(route_offsets_np), as_contig_i32(route_lanes_np), as_contig_i32(route_turns_np)

    repaired_lists = []
    repaired_turns = []
    dropped = 0
    truncated = 0
    fixed_steps = 0
    impossible_steps = 0

    for rid in range(max(0, len(route_offsets_np) - 1)):
        off0 = int(route_offsets_np[rid])
        off1 = int(route_offsets_np[rid + 1])
        if off1 <= off0:
            dropped += 1
            continue

        lane_path = [int(x) for x in route_lanes_np[off0:off1]]
        turn_path = [int(x) for x in route_turns_np[off0:off1]]
        if not lane_path or any((x < 0 or x >= len(lanes)) for x in lane_path):
            dropped += 1
            continue

        out_lanes = [lane_path[0]]
        out_turns = []
        ok = True
        for i in range(len(lane_path) - 1):
            current_lane = int(out_lanes[-1])
            desired_next = int(lane_path[i + 1])
            prev_turn = int(turn_path[i]) if i < len(turn_path) else TURN_STRAIGHT
            fallback_turn = int(turn_path[i + 1]) if i + 1 < len(turn_path) else TURN_STRAIGHT

            current_link_id = int(lanes[current_lane]["link_id"])
            desired_link_id = int(lanes[desired_next]["link_id"]) if 0 <= desired_next < len(lanes) else -1
            current_group = link_to_lanes.get(current_link_id, [])
            desired_group = link_to_lanes.get(desired_link_id, [])

            # EN/KO: Mainline -> ramp transitions are valid only from the
            # nearest outside source lane.  Fix this before checking the normal
            # node-to-node connection so stale route caches cannot place a car
            # on an inner lane and make it wait forever at the ramp mouth.
            edge_exit = interchange_source_outer_lane_id(current_group, desired_group)
            if edge_exit >= 0 and int(edge_exit) != current_lane:
                prev_ok = len(out_lanes) <= 1 or lane_connected_py(lanes, int(out_lanes[-2]), int(edge_exit))
                if prev_ok:
                    out_lanes[-1] = int(edge_exit)
                    current_lane = int(edge_exit)
                    current_link_id = int(lanes[current_lane]["link_id"])
                    current_group = link_to_lanes.get(current_link_id, [])
                    fixed_steps += 1

            next_lane = desired_next

            # EN/KO: Ramp -> mainline transitions must enter through the nearest
            # outside receiving lane.
            edge_receive = interchange_receiving_outer_lane_id(current_group, desired_group)
            if edge_receive >= 0 and lane_connected_py(lanes, current_lane, edge_receive):
                if int(edge_receive) != int(next_lane):
                    fixed_steps += 1
                next_lane = int(edge_receive)

            if not lane_connected_py(lanes, current_lane, next_lane):
                key = (int(rid) * 1000003) ^ (int(i) * 9176) ^ int(current_lane) ^ (int(desired_next) * 131071)
                next_lane = route_lane_repair_candidate(
                    current_lane,
                    desired_next,
                    link_to_lanes,
                    lanes,
                    previous_turn=prev_turn,
                    fallback_turn=fallback_turn,
                    key=key,
                )
                if next_lane >= 0:
                    fixed_steps += 1
                else:
                    impossible_steps += 1
                    truncated += 1
                    break

            next_link_id = int(lanes[int(next_lane)]["link_id"])
            out_turns.append(route_turn_for_link_transition(current_link_id, next_link_id, link_to_lanes, links))
            out_lanes.append(int(next_lane))

        if len(out_lanes) < 2:
            dropped += 1
            continue
        if len(out_turns) < len(out_lanes):
            out_turns.append(TURN_STRAIGHT)
        else:
            out_turns = out_turns[:len(out_lanes)]
        repaired_lists.append(out_lanes)
        repaired_turns.append(out_turns)

    if not repaired_lists:
        raise RuntimeError("No valid routes after route lane safety repair.")

    out_offsets, out_lanes, out_turns = _pack_route_lists(repaired_lists, repaired_turns)
    if fixed_steps or dropped or truncated or impossible_steps:
        print(
            "[Routes] lane safety repair:",
            "routes_in=", int(len(route_offsets_np) - 1),
            "routes_out=", int(len(out_offsets) - 1),
            "fixed_steps=", int(fixed_steps),
            "truncated_routes=", int(truncated),
            "dropped_routes=", int(dropped),
            "impossible_steps=", int(impossible_steps),
        )
    else:
        print("[Routes] lane safety repair: all route lane transitions are connector-valid")
    return out_offsets, out_lanes, out_turns


# ============================================================
# Main
# ============================================================

def main():
    import pygame
    from OpenGL.GL import (
        glViewport, glClearColor, glEnable, glBlendFunc, glUseProgram,
        glGetUniformLocation, glUniform2f, glUniform1f, glUniform1i,
        glClear, glBindVertexArray, glDrawArrays, glLineWidth,
        glActiveTexture, glBindTexture,
        GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_PROGRAM_POINT_SIZE,
        GL_COLOR_BUFFER_BIT, GL_TRIANGLES, GL_POINTS, GL_LINES, GL_TEXTURE0, GL_TEXTURE_2D,
    )
    import avabm_cuda as sim

    if DT <= 0.0 or DT > 0.25:
        raise ValueError("DT must be in (0, 0.25]")
    if not (0.0 <= AV_PENETRATION <= 1.0):
        raise ValueError("AV_PENETRATION must be in [0, 1]")
    if METRICS_SIZE < 112:
        raise ValueError("METRICS_SIZE must be at least 112 for conflict-aware priority / human-AI CUDA core")
    if RENDER_VERTS_PER_VEHICLE not in (RENDER_BODY_VERTS_PER_VEHICLE, RENDER_FULL_VERTS_PER_VEHICLE):
        raise ValueError("invalid RENDER_VERTS_PER_VEHICLE")

    print("[Info] load network")
    nodes, links, lanes, network_crs = build_graph_and_lanes(ROAD_GPKG, layer=GPKG_LAYER)
    num_nodes = len(nodes)
    num_lanes = len(lanes)
    print("[Network] nodes:", num_nodes)
    print("[Network] links:", len(links))
    print("[Network] lanes:", num_lanes)
    print("[Network] crs:", network_crs)

    print("[Info] build static network")
    G, link_to_lanes, left_np, right_np = build_static_network(nodes, links, lanes)
    _interchange_adjusted = apply_interchange_outer_edge_geometry(nodes, links, lanes, link_to_lanes)
    if _interchange_adjusted:
        # Rebuild graph weights and lane adjacency after ramp lane endpoints are
        # snapped from GIS centerlines to the nearest physical outside lane.
        G, link_to_lanes, left_np, right_np = build_static_network(nodes, links, lanes)
    print("[Graph] nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    print("[Network] lane side ordering: geometry-normalized right->left; surface-aware turns enabled; interchange ramps use outer edge lanes")

    spawn_records, spawn_crs = load_spawn_records(SPAWN_GPKG, layer=SPAWN_GPKG_LAYER, label="Spawn")
    origin_nodes, destination_nodes, spawn_nodes, node_spawn_profiles = match_spawn_records(
        nodes, spawn_records, spawn_crs, network_crs
    )
    if not spawn_nodes:
        raise RuntimeError("No spawn nodes.")
    if not origin_nodes:
        raise RuntimeError("No origin-capable spawn nodes. Check SPWNTYPE values.")
    if not destination_nodes:
        raise RuntimeError("No destination-capable spawn nodes. Check SPWNTYPE values.")
    for nid in spawn_nodes:
        nodes[int(nid)]["spawn"] = True
        nodes[int(nid)]["spawn_origin"] = int(nid) in origin_nodes
        nodes[int(nid)]["spawn_destination"] = int(nid) in destination_nodes

    route_offsets_np, route_lanes_np, route_turns_np = make_routes_ready_cached(
        G=G, nodes=nodes, links=links, lanes=lanes, link_to_lanes=link_to_lanes,
        origin_nodes=origin_nodes, destination_nodes=destination_nodes,
        gpkg_path=ROAD_GPKG, network_crs=network_crs, num_routes=NUM_ROUTES,
    )
    validate_route_arrays(route_offsets_np, route_lanes_np, route_turns_np, num_lanes)
    valid_route_count = len(route_offsets_np) - 1
    print("[Routes] valid:", valid_route_count)
    print("[Routes] elements:", len(route_lanes_np))

    lane_length_np = as_contig_f32([l["length"] for l in lanes])
    lane_start_x_np = as_contig_f32([l["start_x"] for l in lanes])
    lane_start_y_np = as_contig_f32([l["start_y"] for l in lanes])
    lane_end_x_np = as_contig_f32([l["end_x"] for l in lanes])
    lane_end_y_np = as_contig_f32([l["end_y"] for l in lanes])
    lane_start_node_np = as_contig_i32([l["from_node"] for l in lanes])
    lane_end_node_np = as_contig_i32([l["to_node"] for l in lanes])
    lane_speed_limit_np = as_contig_f32([float(links[int(l["link_id"])]["speed_mps"]) for l in lanes])
    conflict_lanes_np = np.full((num_lanes, MAX_CONFLICT_LANES), -1, dtype=np.int32)

    section_info = None
    if SECTION_STATS_ENABLED:
        print("[SectionStats] build analysis sections. length_m:", SECTION_LENGTH_M)
        section_info = build_analysis_sections(links, lanes, network_crs, SECTION_LENGTH_M)
        print("[SectionStats] sections:", len(section_info["gdf"]), "label:", section_length_label_value(SECTION_LENGTH_M))
    else:
        print("[SectionStats] disabled")

    print("[Info] build spawn table")
    routes_by_first_lane = {}
    for rid in range(valid_route_count):
        off0 = int(route_offsets_np[rid])
        off1 = int(route_offsets_np[rid + 1])
        if off1 <= off0:
            continue
        first_lane = int(route_lanes_np[off0])
        if 0 <= first_lane < num_lanes:
            routes_by_first_lane.setdefault(first_lane, []).append(rid)
    spawn_lanes = sorted(routes_by_first_lane.keys())
    if not spawn_lanes:
        raise RuntimeError("No spawn lanes from routes.")
    rng = np.random.default_rng(ROUTE_SEED)
    spawn_lane_np = as_contig_i32(spawn_lanes)
    spawn_route_np = np.full(len(spawn_lanes), -1, dtype=np.int32)
    for i, lane_id in enumerate(spawn_lane_np):
        candidates = routes_by_first_lane[int(lane_id)]
        spawn_route_np[i] = int(candidates[int(rng.integers(0, len(candidates)))])
    spawn_route_np = as_contig_i32(spawn_route_np)
    num_spawn_points = int(len(spawn_lane_np))
    print("[Spawn] slots:", num_spawn_points)

    lane_road_width = np.zeros(num_lanes, dtype=np.float32)
    lane_count_dir = np.ones(num_lanes, dtype=np.float32)
    for lane in lanes:
        lid = int(lane["lane_id"])
        link = links[int(lane["link_id"])]
        lane_road_width[lid] = float(link["width"])
        lane_count_dir[lid] = max(1.0, float(link["lane_count"]))

    spawn_group_slots = {}
    spawn_node_slots = {}
    for i, lid in enumerate(spawn_lane_np):
        lane = lanes[int(lid)]
        origin_node = int(lane["from_node"])
        link_id = int(lane["link_id"])
        spawn_group_slots.setdefault((origin_node, link_id), []).append(int(i))
        spawn_node_slots.setdefault(origin_node, []).append(int(i))

    demand_np = np.zeros(num_spawn_points, dtype=np.float32)
    for slots in spawn_group_slots.values():
        if not slots:
            continue
        ref_lid = int(spawn_lane_np[int(slots[0])])
        width = max(1.0, float(lane_road_width[ref_lid]))
        lanes_here = max(1.0, float(lane_count_dir[ref_lid]))
        width_mult = math.exp(SPAWN_WIDTH_EXP_K * (width / max(SPAWN_REF_WIDTH, 1.0) - 1.0))
        width_mult = min(width_mult, SPAWN_MAX_MULT)
        lane_mult = lanes_here ** SPAWN_LANE_POWER
        group_rate = BASE_VPS * width_mult * lane_mult
        per_lane_rate = group_rate / max(1, len(slots)) if SPAWN_MULTI_LANE_BALANCE else group_rate
        for i in slots:
            demand_np[int(i)] = float(per_lane_rate)

    total_vps = float(np.sum(demand_np))
    if total_vps > MAX_TOTAL_VPS and total_vps > 1.0e-9:
        demand_np *= MAX_TOTAL_VPS / total_vps
    demand_np = as_contig_f32(demand_np)

    profile_slots = max(1, int(SPAWN_PROFILE_SLOTS))
    spawn_profile_np = np.zeros((num_spawn_points, profile_slots), dtype=np.float32)
    spawn_profile_has_np = np.zeros(num_spawn_points, dtype=np.int32)
    for origin_node, slots in spawn_node_slots.items():
        prof = node_spawn_profiles.get(int(origin_node))
        if prof is None or not slots:
            continue
        prof = np.asarray(prof, dtype=np.float32)
        if prof.size != profile_slots:
            tmp = np.full(profile_slots, np.nan, dtype=np.float32)
            ncopy = min(profile_slots, int(prof.size))
            if ncopy > 0:
                tmp[:ncopy] = prof[:ncopy]
            prof = _fill_spawn_profile_circular(tmp)
        if prof is None:
            continue
        prof = np.maximum(np.asarray(prof, dtype=np.float32), 0.0)
        per_slot_prof = prof / max(1, len(slots)) if SPAWN_MULTI_LANE_BALANCE else prof
        for i in slots:
            spawn_profile_np[int(i), :] = per_slot_prof
            spawn_profile_has_np[int(i)] = 1

    multi_lane_groups = sum(1 for slots in spawn_group_slots.values() if len(slots) > 1)
    multi_lane_slots = sum(len(slots) for slots in spawn_group_slots.values() if len(slots) > 1)
    if SPAWN_MULTI_LANE_BALANCE:
        print(
            "[Spawn] multi-lane balance:",
            "groups=", int(multi_lane_groups),
            "balanced_slots=", int(multi_lane_slots),
            "node_profile_split=", int(sum(1 for slots in spawn_node_slots.values() if len(slots) > 1)),
        )

    total_vps_host = float(np.sum(demand_np))
    profile_count = int(np.sum(spawn_profile_has_np))
    print("[Demand] default_total_vps:", total_vps_host)
    print("[Demand] default min/max slot vps:", float(np.min(demand_np)), float(np.max(demand_np)))
    if profile_count > 0:
        slot_totals = np.sum(spawn_profile_np[spawn_profile_has_np.astype(bool)], axis=0)
        print("[Demand] profiled slots:", profile_count, "unit:", SPAWN_PROFILE_UNIT, "slot_seconds:", SPAWN_PROFILE_SLOT_SECONDS)
        print("[Demand] profile total_vps min/max:", float(np.min(slot_totals)), float(np.max(slot_totals)))
    else:
        print("[Demand] no SPWNxx profile found; using default demand for all spawn lanes.")

    signal_points, signal_crs = load_points(SIGNAL_GPKG, layer=SIGNAL_GPKG_LAYER, label="Signal")
    signal_records = build_signal_records(nodes, lanes, signal_points, signal_crs, network_crs)
    signal_np = signal_records_to_numpy(signal_records)
    num_signals_cuda = len(signal_np["node"])
    num_signals_render = len(signal_records)
    print("[Signals] cuda:", num_signals_cuda, "render:", num_signals_render)

    min_x, min_y, max_x, max_y, scale, padding = compute_map_transform(links, SCREEN_W, SCREEN_H)
    world_min_x, world_min_y, world_cell_size, world_grid_w, world_grid_h = compute_world_grid(min_x, min_y, max_x, max_y, WORLD_CELL_SIZE)
    print("[WorldGrid] min:", world_min_x, world_min_y)
    print("[WorldGrid] cell:", world_cell_size)
    print("[WorldGrid] size:", world_grid_w, world_grid_h)

    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)
    zoom = 1.0
    zoom_speed = ZOOM_SPEED
    cam_x = mid_x - SCREEN_W / (2.0 * scale * zoom)
    cam_y = mid_y - SCREEN_H / (2.0 * scale * zoom)
    dragging = False
    last_mouse = (0, 0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print("[CUDA] device:", torch.cuda.get_device_name(0))
    print("[Render] vehicle vertices:", RENDER_VERTS_PER_VEHICLE, "wheels:", bool(RENDER_WHEELS), "interval:", RENDER_INTERVAL, "textured:", bool(USE_TEXTURED_CARS))
    print("[Metrics] size:", METRICS_SIZE)
    print("[PriorityGate/Clearance] world_cell_size:", WORLD_CELL_SIZE, "fps_limit:", FPS_LIMIT, "metrics_interval:", METRICS_INTERVAL, "lane_markings:", DRAW_LANE_MARKINGS, "centerlines:", DRAW_LANE_CENTERLINES, "edges:", DRAW_LANE_EDGES, "line_width:", LANE_MARKING_WIDTH, "group_by_source:", LANE_MARKING_GROUP_BY_SOURCE, "vehicle_hover:", SHOW_VEHICLE_HOVER)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("AVABM ECS CUDA Traffic - curved intersections + deadlock escape")

    glViewport(0, 0, SCREEN_W, SCREEN_H)
    glClearColor(0.11, 0.11, 0.11, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_PROGRAM_POINT_SIZE)

    shader = create_shader()
    vehicle_shader = create_vehicle_texture_shader() if USE_TEXTURED_CARS else shader
    vehicle_texture = load_vehicle_texture(CAR_TEXTURE_PATH) if USE_TEXTURED_CARS else None
    ui_renderer = create_ui_text_renderer() if (SHOW_SIGNAL_HOVER or SHOW_VEHICLE_HOVER) else None
    road_arr = build_road_vertex_array(links, lanes, nodes)
    road_vao, road_vbo, road_vertices = create_static_vao_vbo(road_arr)

    # EN: Lane markings are static, so one CPU-built VBO is enough.
    # KO: 차선 표시는 정적이므로 CPU에서 한 번 만든 VBO를 계속 재사용합니다.
    lane_marking_arr = build_lane_marking_vertex_array(lanes, nodes)
    lane_marking_vao, lane_marking_vbo, lane_marking_vertices = create_static_vao_vbo(lane_marking_arr)
    print("[Render] lane marking vertices:", lane_marking_vertices, "enabled:", bool(DRAW_LANE_MARKINGS))

    vehicle_vao, vehicle_vbo = create_dynamic_vao_vbo(MAX_AGENTS * RENDER_VERTS_PER_VEHICLE)
    signal_vao, signal_vbo = create_dynamic_vao_vbo(max(num_signals_render, 1))

    glUseProgram(shader)
    loc_u_cam = glGetUniformLocation(shader, "u_cam")
    loc_u_scale = glGetUniformLocation(shader, "u_scale")
    loc_u_screen = glGetUniformLocation(shader, "u_screen")
    loc_u_padding = glGetUniformLocation(shader, "u_padding")
    loc_u_is_point = glGetUniformLocation(shader, "u_is_point")

    glUseProgram(vehicle_shader)
    vehicle_loc_u_cam = glGetUniformLocation(vehicle_shader, "u_cam")
    vehicle_loc_u_scale = glGetUniformLocation(vehicle_shader, "u_scale")
    vehicle_loc_u_screen = glGetUniformLocation(vehicle_shader, "u_screen")
    vehicle_loc_u_padding = glGetUniformLocation(vehicle_shader, "u_padding")
    vehicle_loc_u_tex = glGetUniformLocation(vehicle_shader, "u_tex")

    def t(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, non_blocking=False)

    veh = {
        "s": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "x": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "y": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "speed": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "accel": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "heading": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "steer_angle": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        "vehicle_length": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "vehicle_width": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "reaction_time": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "min_gap": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        "lane_id": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "active": torch.zeros(MAX_AGENTS, device=device, dtype=torch.int32),
        "driver_type": torch.zeros(MAX_AGENTS, device=device, dtype=torch.int32),
        "route_id": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "route_pos": torch.zeros(MAX_AGENTS, device=device, dtype=torch.int32),

        "vehicle_state": torch.full((MAX_AGENTS,), VEH_ON_LANE, device=device, dtype=torch.int32),
        "connector_from_lane": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "connector_to_lane": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "connector_s": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "connector_length": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        "lane_change_active": torch.zeros(MAX_AGENTS, device=device, dtype=torch.int32),
        "lane_change_from_lane": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "lane_change_to_lane": torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32),
        "lane_change_t": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "lane_change_duration": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        "aggressiveness": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "politeness": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "risk_tolerance": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "comfort_decel": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "desired_speed_factor": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
        "lc_cooldown": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        # EN: Turn indicators are rendered as amber tint and used as public intent for lane-change courtesy.
        # KO: 방향지시등은 황색 tint로 렌더링되며 차선변경 양보 판단에 공개 의도로 사용됩니다.
        "turn_signal": torch.zeros(MAX_AGENTS, device=device, dtype=torch.int32),
        "turn_signal_time": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),

        "entry_time": torch.zeros(MAX_AGENTS, device=device, dtype=torch.float32),
    }

    road = {
        "lane_length": t(lane_length_np),
        "lane_start_x": t(lane_start_x_np),
        "lane_start_y": t(lane_start_y_np),
        "lane_end_x": t(lane_end_x_np),
        "lane_end_y": t(lane_end_y_np),
        "lane_speed_limit": t(lane_speed_limit_np),
        "lane_start_node": t(lane_start_node_np),
        "lane_end_node": t(lane_end_node_np),
        "left_lane": t(left_np.astype(np.int32)),
        "right_lane": t(right_np.astype(np.int32)),
        "conflict_lanes": t(conflict_lanes_np.astype(np.int32)),
    }
    routes = {
        "offsets": t(route_offsets_np.astype(np.int32)),
        "lanes": t(route_lanes_np.astype(np.int32)),
        "turns": t(route_turns_np.astype(np.int32)),
    }

    spawn_accumulator = torch.zeros(num_spawn_points, device=device, dtype=torch.float32)
    demand_vps = t(demand_np.astype(np.float32))
    demand_profile_vps = t(spawn_profile_np.astype(np.float32))
    demand_profile_has = t(spawn_profile_has_np.astype(np.int32))
    spawn_lane = t(spawn_lane_np.astype(np.int32))
    spawn_route = t(spawn_route_np.astype(np.int32))

    lane_cell_head = torch.full((num_lanes * CELL_COUNT_PER_LANE,), -1, device=device, dtype=torch.int32)
    lane_cell_next = torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32)
    world_cell_head = torch.full((world_grid_w * world_grid_h,), -1, device=device, dtype=torch.int32)
    world_cell_next = torch.full((MAX_AGENTS,), -1, device=device, dtype=torch.int32)

    signals = {
        "node": t(signal_np["node"].astype(np.int32)),
        "turn": t(signal_np["turn"].astype(np.int32)),
        "cycle": t(signal_np["cycle"].astype(np.float32)),
        "green_start": t(signal_np["green_start"].astype(np.float32)),
        "green_end": t(signal_np["green_end"].astype(np.float32)),
        "yellow_start": t(signal_np["yellow_start"].astype(np.float32)),
        "yellow_end": t(signal_np["yellow_end"].astype(np.float32)),
    }

    rng_state = torch.randint(1, 1 << 31, (MAX_AGENTS + num_spawn_points + 32,), device=device, dtype=torch.int32)
    metrics = torch.zeros(METRICS_SIZE, device=device, dtype=torch.float32)
    intersection_lock = torch.full((num_nodes,), -1, device=device, dtype=torch.int32)
    reservation_table = torch.full((num_nodes * RES_HORIZON_SLOTS,), -1, device=device, dtype=torch.int32)

    torch.cuda.synchronize()
    sim.register_render_vbo(int(vehicle_vbo))
    if hasattr(sim, "set_vehicle_texture_render"):
        sim.set_vehicle_texture_render(bool(USE_TEXTURED_CARS))

    clock = pygame.time.Clock()
    writer = AsyncMetricsWriter(METRICS_PATH)
    section_collector = SectionStatsRecorder(section_info, MAX_AGENTS, enabled=SECTION_STATS_ENABLED) if section_info is not None else None
    running = True
    exit_reason = "running"
    step = 0
    render_frame = 0
    last_metrics_time = time.perf_counter()
    print("[Runtime] simulation_duration_seconds:", SIMULATION_DURATION_SECONDS, "section_stats:", bool(SECTION_STATS_ENABLED), "section_length_m:", SECTION_LENGTH_M, "section_interval:", SECTION_STATS_INTERVAL, "physics_steps_per_frame:", PHYSICS_STEPS_PER_FRAME)

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_reason = "window_close"
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        exit_reason = "esc"
                        running = False
                    elif event.key == pygame.K_r:
                        for v in veh.values():
                            v.zero_()
                        veh["active"].zero_()
                        veh["lane_id"].fill_(-1)
                        veh["route_id"].fill_(-1)
                        veh["route_pos"].zero_()
                        veh["vehicle_state"].fill_(VEH_ON_LANE)
                        veh["connector_from_lane"].fill_(-1)
                        veh["connector_to_lane"].fill_(-1)
                        veh["connector_s"].zero_()
                        veh["connector_length"].zero_()
                        veh["lane_change_active"].zero_()
                        veh["lane_change_from_lane"].fill_(-1)
                        veh["lane_change_to_lane"].fill_(-1)
                        veh["lane_change_t"].zero_()
                        veh["lane_change_duration"].zero_()
                        veh["lc_cooldown"].zero_()
                        veh["turn_signal"].zero_()
                        veh["turn_signal_time"].zero_()
                        spawn_accumulator.zero_()
                        metrics.zero_()
                        intersection_lock.fill_(-1)
                        reservation_table.fill_(-1)
                        if section_collector is not None:
                            section_collector.reset()
                        step = 0
                        render_frame = 0
                        print("[Reset] ECS simulation reset")
                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    world_x, world_y = screen_to_world(mx, my, cam_x, cam_y, scale * zoom, padding, SCREEN_H)
                    zoom = zoom * zoom_speed if event.y > 0 else zoom / zoom_speed
                    zoom = max(ZOOM_MIN, min(zoom, ZOOM_MAX))
                    cam_x = world_x - (mx - padding) / (scale * zoom)
                    cam_y = world_y - (SCREEN_H - my - padding) / (scale * zoom)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_mouse = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    mx, my = pygame.mouse.get_pos()
                    dx = mx - last_mouse[0]
                    dy = my - last_mouse[1]
                    cam_x -= dx / (scale * zoom)
                    cam_y += dy / (scale * zoom)
                    last_mouse = (mx, my)

            if not running:
                break

            current_time = float(step * DT)
            if SIMULATION_DURATION_SECONDS >= 0.0 and current_time >= float(SIMULATION_DURATION_SECONDS):
                exit_reason = "duration"
                running = False
                break

            for _physics_substep in range(PHYSICS_STEPS_PER_FRAME):
                current_time = float(step * DT)
                if SIMULATION_DURATION_SECONDS >= 0.0 and current_time >= float(SIMULATION_DURATION_SECONDS):
                    exit_reason = "duration"
                    running = False
                    break

                sim.step(
                    veh["s"], veh["x"], veh["y"], veh["speed"], veh["accel"], veh["heading"], veh["steer_angle"],
                    veh["vehicle_length"], veh["vehicle_width"], veh["reaction_time"], veh["min_gap"],
                    veh["lane_id"], veh["active"], veh["driver_type"], veh["route_id"], veh["route_pos"],
                    veh["vehicle_state"], veh["connector_from_lane"], veh["connector_to_lane"], veh["connector_s"], veh["connector_length"],
                    veh["lane_change_active"], veh["lane_change_from_lane"], veh["lane_change_to_lane"], veh["lane_change_t"], veh["lane_change_duration"],
                    veh["aggressiveness"], veh["politeness"], veh["risk_tolerance"], veh["comfort_decel"], veh["desired_speed_factor"], veh["lc_cooldown"],
                    veh["turn_signal"], veh["turn_signal_time"],
                    road["lane_length"], road["lane_start_x"], road["lane_start_y"], road["lane_end_x"], road["lane_end_y"], road["lane_speed_limit"],
                    road["lane_start_node"], road["lane_end_node"], road["left_lane"], road["right_lane"], road["conflict_lanes"],
                    routes["offsets"], routes["lanes"], routes["turns"],
                    spawn_accumulator, demand_vps, demand_profile_vps, demand_profile_has,
                    int(profile_slots), float(SPAWN_PROFILE_SLOT_SECONDS),
                    spawn_lane, spawn_route,
                    veh["entry_time"],
                    lane_cell_head, lane_cell_next,
                    world_cell_head, world_cell_next,
                    float(world_min_x), float(world_min_y), float(world_cell_size), int(world_grid_w), int(world_grid_h),
                    signals["node"], signals["turn"], signals["cycle"], signals["green_start"], signals["green_end"], signals["yellow_start"], signals["yellow_end"],
                    rng_state, metrics,
                    current_time, float(DT), float(AV_PENETRATION), int(MAX_AGENTS), int(num_spawn_points), int(num_lanes), int(num_signals_cuda), int(step),
                    intersection_lock, reservation_table, int(num_nodes),
                )

                if DEBUG_SYNC:
                    torch.cuda.synchronize()

                section_sample_time = float(current_time + DT)
                if section_collector is not None:
                    section_collector.maybe_sample(section_sample_time, step, veh, force=False)
                step += 1

            if not running:
                break

            current_time = float(step * DT)
            do_vehicle_render = (render_frame % RENDER_INTERVAL) == 0
            if do_vehicle_render:
                sim.update_render_vbo(
                    veh["x"], veh["y"], veh["heading"], veh["steer_angle"], veh["active"], veh["driver_type"],
                    veh["vehicle_length"], veh["vehicle_width"], int(MAX_AGENTS),
                )

            now = time.perf_counter()
            if now - last_metrics_time >= METRICS_INTERVAL:
                m = metrics.detach().cpu().numpy()
                snap = metrics_to_snapshot(m)
                total_vps_live = float(np.sum(interpolate_spawn_demand_np(
                    demand_np, spawn_profile_np, spawn_profile_has_np, current_time, SPAWN_PROFILE_SLOT_SECONDS
                )))
                pygame.display.set_caption(
                    f"AVABM ECS | t={format_sim_time(current_time)} | "
                    f"active={snap['active']} spawn={snap['spawned']} done={snap['completed']} "
                    f"v={snap['avg_speed']:.2f} delay={snap['avg_delay']:.2f} headway={snap['avg_headway']:.2f} "
                    f"lc={snap['lane_change']} rej_lc={snap['rejected_lane_change']} "
                    f"res={snap['reservation_accept']}/{snap['reservation_reject']} conn={snap['connector_active']} "
                    f"sensor={snap['sensor_front_hit']}/{snap['sensor_detection']} interact={snap['interaction_brake']} "
                    f"turnprep={snap['turn_lane_prep']} turnblk={snap['turn_lane_block']} "
                    f"prioY={snap['unsignal_right_yield']} prioGo={snap['unsignal_priority_go']} "
                    f"deadRel={snap['deadlock_release']} esc={snap['deadlock_escape_go']} gate={snap['priority_gate_granted']}/{snap['priority_gate_blocked']} "
                    f"freeGo={snap['priority_conflict_free_go']} pathBlk={snap['priority_path_block']} fsRel={snap['front_space_release']} "
                    f"occ={snap['intersection_occupied_hold']} pass={snap['force_pass_through']} qhold={snap['entry_queue_hold']:.1f} "
                    f"sigL/R={snap['indicator_left_on']}/{snap['indicator_right_on']} hAI={snap['human_ai_assertive_go']}/{snap['human_ai_courtesy_yield']} "
                    f"rt={snap['right_turn_symmetric_path']}/{snap['right_turn_exit_gap_hold']} anti={snap['anti_collision_brake']} pen={snap['penetration_prevented']} "
                    f"near={snap['near_miss']} ttc={snap['ttc_critical']}/{snap['ttc_warning']} "
                    f"hb={snap['hard_brake']} col={snap['collision']} rej_sp={snap['rejected_spawn']} vps={total_vps_live:.1f}"
                )
                writer.write(step, snap)
                last_metrics_time = now

            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(shader)
            glUniform2f(loc_u_cam, cam_x, cam_y)
            glUniform1f(loc_u_scale, scale * zoom)
            glUniform2f(loc_u_screen, SCREEN_W, SCREEN_H)
            glUniform1f(loc_u_padding, padding)

            glUniform1i(loc_u_is_point, 0)
            glBindVertexArray(road_vao)
            glDrawArrays(GL_TRIANGLES, 0, road_vertices)

            if lane_marking_vertices > 0:
                # EN: Draw lane markings over the road surface.
                # KO: 도로 면 위에 차선 표시를 덧그립니다.
                glLineWidth(max(0.1, float(LANE_MARKING_WIDTH)))
                glBindVertexArray(lane_marking_vao)
                glDrawArrays(GL_LINES, 0, lane_marking_vertices)

            if num_signals_render > 0:
                glUniform1i(loc_u_is_point, 1)
                update_signal_vbo(signal_vbo, signal_records, current_time)
                glBindVertexArray(signal_vao)
                glDrawArrays(GL_POINTS, 0, num_signals_render)

            # EN: Vehicle draw.  If USE_TEXTURED_CARS=1, the CUDA VBO stores UVs
            #     in the color attribute slots and this shader samples assets/car_topdown.png.
            # KO: 차량 렌더링입니다. USE_TEXTURED_CARS=1이면 CUDA VBO의 색상 attribute 칸에
            #     UV 좌표를 넣고, 이 셰이더가 assets/car_topdown.png를 샘플링합니다.
            if USE_TEXTURED_CARS:
                glUseProgram(vehicle_shader)
                glUniform2f(vehicle_loc_u_cam, cam_x, cam_y)
                glUniform1f(vehicle_loc_u_scale, scale * zoom)
                glUniform2f(vehicle_loc_u_screen, SCREEN_W, SCREEN_H)
                glUniform1f(vehicle_loc_u_padding, padding)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, vehicle_texture)
                glUniform1i(vehicle_loc_u_tex, 0)
                glBindVertexArray(vehicle_vao)
                glDrawArrays(GL_TRIANGLES, 0, MAX_AGENTS * RENDER_VERTS_PER_VEHICLE)
                glBindTexture(GL_TEXTURE_2D, 0)
            else:
                glUseProgram(shader)
                glUniform1i(loc_u_is_point, 0)
                glBindVertexArray(vehicle_vao)
                glDrawArrays(GL_TRIANGLES, 0, MAX_AGENTS * RENDER_VERTS_PER_VEHICLE)

            hover_lines = signal_hover_lines(
                signal_records, pygame.mouse.get_pos(), cam_x, cam_y, scale * zoom,
                padding, SCREEN_W, SCREEN_H, current_time
            )
            if not hover_lines:
                hover_lines = vehicle_hover_lines(
                    veh, pygame.mouse.get_pos(), cam_x, cam_y, scale * zoom,
                    padding, SCREEN_W, SCREEN_H, current_time
                )
            if hover_lines and ui_renderer is not None:
                draw_text_panel(ui_renderer, hover_lines, pygame.mouse.get_pos(), SCREEN_W, SCREEN_H)

            glBindVertexArray(0)

            pygame.display.flip()
            clock.tick(FPS_LIMIT if FPS_LIMIT > 0 else 0)
            render_frame += 1
    finally:
        # Normal shutdown path.  Section outputs are intentionally written here so
        # duration-stop, ESC, window-close, and any other clean exit all persist
        # the final CSV/GPKG research datasets.
        final_step = int(step) if "step" in locals() else 0
        final_sim_time = float(final_step * DT)
        try:
            final_reason = str(exit_reason)
            if final_reason == "running":
                final_reason = "normal"
        except Exception:
            final_reason = "normal"

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        final_metrics = {}
        try:
            if "metrics" in locals():
                final_metrics = metrics_to_snapshot(metrics.detach().cpu().numpy())
        except Exception as e:
            print("[SectionStats] final metrics snapshot failed:", e)
            final_metrics = {}

        try:
            if "section_collector" in locals() and section_collector is not None:
                section_collector.finalize(final_sim_time, final_step, veh, metrics_snapshot=final_metrics, exit_reason=final_reason)
        except Exception as e:
            print("[SectionStats] finalize failed:", e)

        try:
            writer.close()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            sim.unregister_render_vbo()
        except Exception as e:
            print("[Warning] unregister failed:", e)
        pygame.quit()


if __name__ == "__main__":
    main()
