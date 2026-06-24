// Monolithic aggregator kept for CUDA_SINGLE_TU_BUILD=1.
// Default builds compile partitioned .cu files to keep each ptxas input small.
#define AVABM_PART_SKIP_COMMON 1
#include "main_common.cuh"
#include "main_core_grid_spawn.cu"
#include "main_core_decision.cu"
#include "main_core_motion.cu"
#include "main_render_launch.cu"
