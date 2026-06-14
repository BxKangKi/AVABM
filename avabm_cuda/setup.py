from pathlib import Path
import os
import shlex
from setuptools import setup


def _read_config(path: Path):
    cfg = {}
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].split(";", 1)[0].strip().strip('"').strip("'")
        if key:
            cfg[key] = None if value.lower() in {"", "none", "null"} else os.path.expandvars(value)
    return cfg


def _truthy(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "none", ""}


def _int_value(value, default=0):
    """Parse optional integer build settings from config.txt.

    KO: 문법 이유: config.txt 값은 문자열이므로 int(float(...))로 "4"와 "4.0"을
        모두 허용하고, 실패 시 default로 되돌립니다.
    KO: 논리 이유: 빌드 옵션 하나가 잘못 적혀도 전체 확장 빌드가 즉시 죽지 않고
        안전한 기본값으로 진행하게 합니다.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float_value(value, default=0.0):
    """Parse optional floating-point build settings from config.txt."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _split_flags(value):
    if not value:
        return []
    return shlex.split(str(value), posix=(os.name != "nt"))


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG = _read_config(ROOT_DIR / "config.txt")

# The batch file normally exports these, but setup.py also reads config.txt so
# manual builds such as `python setup.py build_ext --inplace` behave the same.
# KO 문법 이유: torch.utils.cpp_extension을 import하기 전에 환경 변수를 먼저 채웁니다.
# KO 논리 이유: PyTorch의 cl 버전 확인 경고는 import 이후 build_ext 단계에서 발생하므로,
#    TORCH_DONT_CHECK_COMPILER_ABI를 미리 설정하면 사용자가 본 불필요한 warning이 사라집니다.
for key in (
    "CUDA_HOME",
    "MAX_JOBS",
    "TORCH_NVCC_FLAGS",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_BUILD_MODE",
    "CUDA_NVCC_THREADS",
    "CUDA_SPLIT_COMPILE",
    "TORCH_DONT_CHECK_COMPILER_ABI",
):
    if CONFIG.get(key) and not os.environ.get(key):
        os.environ[key] = CONFIG[key]
if not os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI"):
    os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"

# Import PyTorch's extension builder only after the ABI-check environment is ready.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

# ---------------------------------------------------------------------------
# Build speed / warning policy
# ---------------------------------------------------------------------------
# KO: CUDA 컴파일은 main.cu가 큰 단일 translation unit이라 원래 오래 걸립니다.
#     v31은 실행 성능 요청을 우선해 config.txt 기본값을 release/O3로 바꿨습니다.
#     재빌드 시간을 다시 줄이고 싶으면 CUDA_BUILD_MODE=fastdev, CUDA_OPT_LEVEL=1로
#     낮출 수 있습니다. MSVC /GL 제거와 가벼운 torch header는 계속 유지됩니다.
# EN: main.cu is a large single CUDA translation unit. v31 defaults config.txt to
#     release/O3 for runtime speed. Switch back to fastdev/O1 when rebuild speed
#     matters more than simulation throughput.
build_mode = str(CONFIG.get("CUDA_BUILD_MODE", "release")).strip().lower()
if build_mode not in {"fastdev", "release", "debug"}:
    build_mode = "release"

# 문법 이유: dict.get(..., default)로 build mode별 기본 최적화 레벨을 한 줄에서
# 선택하고, config.txt의 CUDA_OPT_LEVEL이 있으면 그 값을 우선합니다.
# 논리 이유: v31에서는 실행 속도를 우선해 release/O3를 config 기본값으로 두지만,
# 개발 중에는 fastdev/O1로 되돌려 재컴파일 시간을 줄일 수 있게 분리합니다.
default_opt_level = {"fastdev": "1", "release": "3", "debug": "0"}.get(build_mode, "1")
opt_level = str(CONFIG.get("CUDA_OPT_LEVEL", default_opt_level)).strip().upper().lstrip("O")
if opt_level not in {"0", "1", "2", "3"}:
    opt_level = default_opt_level

cxx_standard = str(CONFIG.get("CUDA_CXX_STANDARD", "17")).strip().lower().replace("c++", "")
if cxx_standard not in {"17", "20"}:
    cxx_standard = "17"

show_warnings = _truthy(CONFIG.get("CUDA_SHOW_WARNINGS"), default=False)
suppress_header_warnings = _truthy(CONFIG.get("CUDA_SUPPRESS_HEADER_WARNINGS"), default=True)
fast_math = _truthy(CONFIG.get("CUDA_FAST_MATH"), default=True)
nvcc_threads = max(0, _int_value(CONFIG.get("CUDA_NVCC_THREADS"), default=0))
split_compile = max(0, _int_value(CONFIG.get("CUDA_SPLIT_COMPILE"), default=0))
disable_msvc_gl = _truthy(CONFIG.get("CUDA_DISABLE_MSVC_GL"), default=True)
use_full_torch_header = _truthy(CONFIG.get("CUDA_USE_FULL_TORCH_EXTENSION_HEADER"), default=False)
fast_equiv_math = _truthy(CONFIG.get("CUDA_FAST_EQUIV_MATH"), default=True)
use_async_memset_clear = _truthy(CONFIG.get("CUDA_USE_ASYNC_MEMSET_CLEAR"), default=True)
spawn_grid_insert_fastpath = _truthy(CONFIG.get("CUDA_SPAWN_GRID_INSERT_FASTPATH"), default=True)
min_cruise_enabled = _truthy(CONFIG.get("SPEED_MIN_CRUISE_ENABLED"), default=True)
min_cruise_kmh = max(0.0, _float_value(CONFIG.get("SPEED_MIN_CRUISE_KMH"), default=40.0))

nvcc_flags = [f"-O{opt_level}", f"-std=c++{cxx_standard}", "--expt-relaxed-constexpr"]
if fast_math:
    nvcc_flags.append("--use_fast_math")
if nvcc_threads > 0:
    # 문법 이유: nvcc 옵션은 ["--threads", "4"]처럼 옵션명과 값을 별도 list
    # 원소로 넣어야 공백/인용 문제가 없습니다.
    # 논리 이유: 큰 main.cu의 device 컴파일 하위 작업을 nvcc 내부 worker thread로
    # 나눠, clean build 시간을 줄입니다. 값 0은 호환성을 위해 옵션을 끕니다.
    nvcc_flags.extend(["--threads", str(nvcc_threads)])
if split_compile > 0:
    # KO: --split-compile은 CUDA 버전/코드 형태에 따라 효과가 다르므로 기본값은 0입니다.
    nvcc_flags.extend(["--split-compile", str(split_compile)])
if not show_warnings:
    # -w suppresses nvcc/device warnings.  Host MSVC warnings are handled with /wd####
    # instead of /w, because /w fought with distutils' /W3 and produced D9025 noise.
    nvcc_flags.extend(["-w"])

# PyTorch does not always propagate TORCH_NVCC_FLAGS into the concrete nvcc
# command for setup.py builds, so copy configured flags explicitly.
for flag in _split_flags(os.environ.get("TORCH_NVCC_FLAGS") or CONFIG.get("TORCH_NVCC_FLAGS")):
    if flag and flag not in nvcc_flags:
        nvcc_flags.append(flag)

for flag in _split_flags(CONFIG.get("CUDA_EXTRA_NVCC_FLAGS")):
    nvcc_flags.append(flag)

if os.name == "nt":
    # KO 문법 이유: MSVC 옵션은 같은 종류가 여러 번 있으면 뒤쪽 옵션이 이깁니다.
    # KO 논리 이유: Python distutils가 기본으로 /O2 /GL을 넣는데, binding.cpp는
    #    PyTorch header 확인이 대부분이라 /Od /GL-가 개발 빌드 시간을 더 줄입니다.
    if build_mode == "release":
        cxx_flags = ["/O2"]
    else:
        cxx_flags = ["/Od", "/Ob0"]
    if disable_msvc_gl:
        cxx_flags.append("/GL-")
    cxx_flags.extend([f"/std:c++{cxx_standard}", "/EHsc", "/bigobj"])
    if suppress_header_warnings:
        warning_disables = [
            "/wd4996", "/wd4819", "/wd4251", "/wd4275", "/wd4244", "/wd4267",
            "/wd4018", "/wd4190", "/wd4624", "/wd4067", "/wd4068", "/wd4273",
        ]
        cxx_flags.extend(warning_disables)
        for wd in warning_disables:
            nvcc_flags.extend(["-Xcompiler", wd])
else:
    # KO 논리 이유: binding.cpp의 런타임 비용은 작고 컴파일 비용이 크므로 fastdev/debug는
    #    -O0으로 빠르게 빌드하고, CUDA kernel(main.cu) 성능은 CUDA_OPT_LEVEL로 별도 제어합니다.
    cxx_flags = ["-O2" if build_mode == "release" else "-O0", f"-std=c++{cxx_standard}"]
    if suppress_header_warnings:
        cxx_flags.extend(["-Wno-deprecated-declarations", "-Wno-unused-parameter"])
        nvcc_flags.append("-Xcompiler=-Wno-deprecated-declarations")

for flag in _split_flags(CONFIG.get("CUDA_EXTRA_CXX_FLAGS")):
    cxx_flags.append(flag)

common_defines = [
    ("_CRT_SECURE_NO_WARNINGS", None),
    ("_SCL_SECURE_NO_WARNINGS", None),
    ("_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS", None),
    ("AVABM_CUDA_BUILD_V31", "1"),
]
if use_full_torch_header:
    # KO: 문제가 생길 경우 config.txt에서 CUDA_USE_FULL_TORCH_EXTENSION_HEADER=1로 바꾸면
    #     기존처럼 torch/extension.h를 사용합니다. 기본은 컴파일이 빠른 light header입니다.
    common_defines.append(("AVABM_USE_FULL_TORCH_EXTENSION_HEADER", "1"))

# KO 문법 이유: define_macros의 값은 컴파일러 명령행의 /D 또는 -D 값이 됩니다.
# KO 논리 이유: 속도 하한과 안전한 memset clear 같은 실행 최적화는 CUDA 코드가
#    컴파일될 때 상수로 박혀야 분기/전역 메모리 읽기 없이 빠르게 동작합니다.
common_defines.extend([
    ("AVABM_FAST_EQUIV_MATH", "1" if fast_equiv_math else "0"),
    ("AVABM_USE_ASYNC_MEMSET_CLEAR", "1" if use_async_memset_clear else "0"),
    ("AVABM_SPAWN_GRID_INSERT_FASTPATH", "1" if spawn_grid_insert_fastpath else "0"),
    ("AVABM_MIN_CRUISE_SPEED_ENABLED", "1" if min_cruise_enabled else "0"),
    ("AVABM_MIN_CRUISE_SPEED_KMH", f"{min_cruise_kmh:.6f}"),
])


_BaseBuildExtension = BuildExtension.with_options(use_ninja=True)


class AVABMBuildExtension(_BaseBuildExtension):
    """BuildExtension with Windows defaults adjusted for faster local builds."""

    def _strip_msvc_slow_defaults(self):
        """Remove MSVC defaults that are useful for release wheels but slow local rebuilds."""
        # KO 문법 이유: distutils의 MSVCCompiler는 initialize()가 끝난 뒤
        #     compile_options/ldflags_shared 같은 list 속성에 기본 플래그를 채웁니다.
        #     따라서 문자열 list를 직접 필터링해야 실제 ninja 명령 앞부분의 /O2 /GL을
        #     제거할 수 있습니다.
        # KO 논리 이유: 사용자가 보낸 로그에서는 우리가 뒤쪽에 /Od /GL-를 넣었지만
        #     앞쪽의 Python 기본값 /O2 /GL이 먼저 남아 D9025 warning을 만들었습니다.
        #     여기서는 기본값 자체를 제거해 warning과 불필요한 LTCG 준비 비용을 줄입니다.
        if os.name != "nt":
            return

        remove_compile = {"/gl"}
        if build_mode != "release":
            remove_compile.add("/o2")

        remove_link = {"/ltcg"}

        for attr in ("compile_options", "compile_options_debug"):
            opts = getattr(self.compiler, attr, None)
            if isinstance(opts, list):
                kept = []
                for opt in opts:
                    key = str(opt).strip().lower()
                    if key in remove_compile:
                        continue
                    kept.append(opt)
                setattr(self.compiler, attr, kept)

        # KO 문법 이유: MSVC 링크 옵션도 list로 보관되므로 같은 방식으로 필터링합니다.
        # KO 논리 이유: /GL을 제거했다면 링크 단계의 /LTCG도 개발 빌드에서는 필요 없습니다.
        for attr in ("ldflags_shared", "ldflags_shared_debug"):
            opts = getattr(self.compiler, attr, None)
            if isinstance(opts, list):
                kept = []
                for opt in opts:
                    key = str(opt).strip().lower()
                    if key in remove_link:
                        continue
                    kept.append(opt)
                setattr(self.compiler, attr, kept)

    def build_extensions(self):
        # KO 문법 이유: PyTorch BuildExtension은 Windows+ninja 경로에서 compile_options를
        #     읽어 build.ninja를 생성합니다. 이 값은 compiler.initialize() 이후 확정되므로,
        #     initialize()를 감싸서 초기화 직후 한 번 더 제거합니다.
        # KO 논리 이유: v29에서는 제거 함수가 너무 이른 시점에 실행되어 /O2 /GL이
        #     실제 명령 앞부분에 남았습니다. v30은 초기화 직후와 super() 호출 직전 두 번
        #     보정해서 PyTorch/Setuptools 버전 차이를 견딥니다.
        if os.name == "nt" and disable_msvc_gl and getattr(self, "compiler", None) is not None:
            original_initialize = getattr(self.compiler, "initialize", None)
            if callable(original_initialize) and not getattr(self.compiler, "_avabm_init_wrapped", False):
                def initialize_and_strip(*args, **kwargs):
                    result = original_initialize(*args, **kwargs)
                    self._strip_msvc_slow_defaults()
                    return result

                self.compiler.initialize = initialize_and_strip
                self.compiler._avabm_init_wrapped = True

            if not getattr(self.compiler, "initialized", False) and callable(getattr(self.compiler, "initialize", None)):
                self.compiler.initialize()
            self._strip_msvc_slow_defaults()

        super().build_extensions()


setup(
    name="avabm_cuda",
    ext_modules=[
        CUDAExtension(
            name="avabm_cuda",
            sources=["binding.cpp", "main.cu"],
            define_macros=common_defines,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": AVABMBuildExtension},
)
