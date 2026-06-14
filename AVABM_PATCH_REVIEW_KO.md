# AVABM CUDA/Deadlock/Spawn Route 재검토 및 패치 메모

## 1. CUDA 컴파일이 오래 걸리는 원인과 적용한 완화

`avabm_cuda/main.cu`가 약 1.3만 줄짜리 큰 단일 CUDA translation unit입니다. 따라서 clean build는 원래 오래 걸릴 수 있습니다. 사용자가 보내준 빌드 로그에서는 CUDA 커널 컴파일 전에 `binding.cpp` C++/PyTorch 확장 컴파일 단계가 먼저 오래 걸리는 모습도 확인되었습니다.

v28 패치에서는 다음을 추가로 적용했습니다.

추가 v29 hotfix에서는 Windows CMD가 UTF-8/Korean batch comments 또는 LF-only batch 파일을 명령 조각으로 잘못 실행하는 문제를 막기 위해 `build.bat`, `run.bat`, `config.txt`를 CRLF + CMD-safe 파서로 다시 작성했습니다.

- `config.txt`
  - `CONFIG_VERSION=31`
  - `TORCH_DONT_CHECK_COMPILER_ABI=1`
  - `CUDA_CXX_STANDARD=17`
  - `CUDA_DISABLE_MSVC_GL=1`
  - `CUDA_USE_FULL_TORCH_EXTENSION_HEADER=0`
  - `CUDA_BUILD_MODE=release`
  - `CUDA_OPT_LEVEL=3`
  - `CUDA_NVCC_THREADS=4`
- `avabm_cuda/setup.py`
  - PyTorch extension builder를 import하기 전에 `TORCH_DONT_CHECK_COMPILER_ABI`를 설정하도록 순서를 바꿨습니다.
  - MSVC 기본 `/GL` link-time code generation을 제거합니다.
  - fastdev/debug에서는 `binding.cpp`를 `/Od /Ob0`로 컴파일해 PyTorch header 확인 시간을 줄입니다.
  - C++20 대신 C++17을 기본값으로 사용합니다.
  - `torch/extension.h` 대신 더 가벼운 header 조합을 기본으로 사용합니다.
  - 문제가 생기면 `CUDA_USE_FULL_TORCH_EXTENSION_HEADER=1`로 기존 전체 header 방식으로 되돌릴 수 있습니다.
- `avabm_cuda/build.bat`
  - `/w`를 쓰지 않도록 바꿨습니다. `/w`는 distutils 기본 `/W3`와 충돌해 `D9025` warning을 만들었습니다.
  - 필요한 경고만 `/wd####`로 끕니다.
  - 새 빌드 옵션을 로그에 출력합니다.
- `avabm_cuda/build_helper.py`
  - 새 빌드 옵션을 fingerprint에 포함했습니다.
  - 소스/빌드 설정이 동일하면 `build_ext` 자체를 건너뜁니다.

주의: 이번 패치처럼 주석/빌드 파일이 바뀌면 첫 1회는 재빌드가 필요합니다. 이후 소스와 빌드 설정이 같으면 skip됩니다.

## 2. 사용자가 보낸 빌드 로그 해석

로그의 핵심은 다음입니다.

```txt
W... Error checking compiler version for cl
subprocess.CalledProcessError: Command 'cl' returned non-zero exit status 2.
```

이 부분은 실제 `cl.exe` 컴파일 실패가 아니라 PyTorch가 `cl`을 인자 없이 실행해 버전 문자열을 확인하려다가 경고를 띄운 것입니다. 바로 아래에서 실제 `binding.cpp` 컴파일 명령이 진행되고 있으므로 fatal error는 아닙니다.

더 중요한 부분은 실제 컴파일 명령에 다음이 들어간 점입니다.

```txt
/O2 /W3 /GL ... binding.cpp ... /O2 /std:c++20 ... /w
```

- `/GL`: link-time code generation입니다. Python extension의 `binding.cpp`에는 이득이 거의 없고 빌드 시간만 늘릴 수 있습니다.
- `/O2`: PyTorch header가 많은 binding 파일에서는 개발 중 컴파일 시간을 늘립니다.
- `/std:c++20`: 이 binding 코드에는 C++20 기능이 필요하지 않습니다.
- `/w`와 `/W3`: 서로 충돌해 `D9025` warning을 만들 수 있습니다.

v28에서는 위 항목을 조정했고, v29에서는 build.bat/run.bat/config.txt 파싱 안정성을 추가로 수정했습니다.

## 3. 데드락 문제 재검토 결과

기존 코드에는 이미 다음 종류의 데드락 완화가 들어 있었습니다.

- unsignal priority deadlock release
- priority gate front-clear release
- connector overlap wait release
- complete overlap release
- lane-change no-start deadlock cleanup
- stale zero-acceleration cleanup

다만 재검토 중 한 가지 실제 위험을 찾았습니다. `connector_enter_system_kernel`은 decision 단계에서 최종 `decision.connector_target_lane`을 보정합니다. 예를 들어 missed-exit straight fallback, 회전 수신 차로 보정, edge-lane 보정이 일어날 수 있습니다. 그런데 `connector_entry_clear_ecs()` 내부의 주변 차량 비교는 on-lane 주변 차량에 대해 아직 raw route next lane을 기준으로 충돌/zipper/deadlock 순서를 비교할 수 있었습니다.

패치 내용:

- `avabm_cuda/main.cu`
  - `connector_entry_clear_ecs()`에서 주변 차량이 아직 ON_LANE이고 이미 connector 의도가 있으면 `decision.connector_target_lane[j]`를 우선 사용합니다.
  - 이 target이 유효하고 실제 lane 연결이 있을 때만 `other_next`를 덮어씁니다.

효과:

- 같은 수신 차로를 향하는 두 차량이 서로 다른 raw route next lane으로 비교되어 zipper/deadlock 순서가 빗나가는 경우를 줄입니다.
- 최종 진입 허가 단계가 decision 단계와 같은 lane target을 기준으로 판단합니다.

## 4. 진입 차량 차선/경로 고정 문제 재검토 결과

이전 구조는 `routes_by_first_lane`에서 lane별 route 후보가 여러 개 있어도 spawn table에는 lane 하나당 route 하나만 들어갔습니다. 즉 다차로 스폰 균등화가 있어도 같은 실제 진입 차로에 나온 차량은 계속 같은 route id를 받을 수 있어, 사용자가 본 “진입 차량 차선 고정/경로 고정” 현상이 남을 수 있었습니다.

패치 내용:

- `main.py`
  - `SPAWN_ROUTE_CHOICES_PER_LANE` 설정을 추가했습니다.
  - 같은 실제 spawn lane을 여러 spawn slot으로 반복하고, 각 slot에 다른 route id를 넣도록 `build_spawn_route_slots()`를 추가했습니다.
  - demand/profile은 반복된 slot 수로 나눠 전체 유입량이 route 후보 수만큼 부풀지 않도록 했습니다.
- `config.txt`
  - `SPAWN_ROUTE_CHOICES_PER_LANE=4` 추가.

효과:

- 실제 진입 차로는 유지되지만 route 후보가 여러 개로 분산됩니다.
- CUDA spawn lane lock은 그대로 유지되므로 같은 tick에 같은 물리 차로로 여러 대가 겹쳐 들어가는 문제는 막습니다.
- spawn 로그에 `lanes`, `slots`, `route_choices_per_lane_max_cfg`, `avg_seen`이 출력되어 실제 route 선택지 수를 확인할 수 있습니다.

## 5. 주석 추가 범위

요청한 “문법 이유”와 “논리적 흐름 이유”를 변경 코드 중심으로 추가했습니다.

- `main.py`: spawn route slot 생성, demand split, 설정값 설명
- `avabm_cuda/main.cu`: connector target 보정 및 deadlock/zipper 판단 이유
- `avabm_cuda/setup.py`: build mode, optimization level, MSVC `/GL` 제거, light torch header fallback 이유
- `avabm_cuda/build_helper.py`: fingerprint 대상 key 선정 이유
- `avabm_cuda/build.bat`: config parsing, skip/rebuild 분기, `cl` warning 억제 이유
- `run.bat`: 실행 전 fingerprint 검증 이유
- `avabm_cuda/binding.cpp`: POD ABI와 tensor check 이유, light torch header 사용 이유
- `avabm_cuda/pyproject.toml`: build-system 선언 이유

## 6. 여기서 수행한 검증

이 패치 작업 환경은 Linux 컨테이너이며 Windows CUDA/MSVC/nvcc 실행 환경이 아니므로 실제 `build.bat` clean build는 실행하지 못했습니다. 대신 다음 검사를 수행했습니다.

- `python -m py_compile main.py avabm_cuda/setup.py avabm_cuda/build_helper.py`
- `python avabm_cuda/build_helper.py check`
- 새 spawn route slot helper의 작은 입력 동작 검사
- 기존 lane당 route 하나 선택 코드가 남아 있는지 grep 검사
- config의 제어문자 backspace 제거 확인

## 7. 사용자 PC에서 권장 실행 순서

1. 기존 실패 중간 산출물이 있으면 한 번만 clean rebuild:

```bat
avabm_cuda\build.bat clean
```

2. 이후 일반 실행:

```bat
run.bat
```

3. 혹시 v28의 light torch header에서 컴파일 오류가 나면 `config.txt`에서 아래처럼 바꾼 뒤 다시 빌드:

```txt
CUDA_USE_FULL_TORCH_EXTENSION_HEADER=1
```

4. 최종 성능 benchmark 때만 `config.txt`에서 다음처럼 바꾸고 재빌드:

```txt
CUDA_BUILD_MODE=release
CUDA_OPT_LEVEL=3
```

5. `CUDA_NVCC_THREADS=4`가 CUDA 환경에서 문제를 만들면 다음처럼 끄고 다시 빌드:

```txt
CUDA_NVCC_THREADS=0
```


## 8. v29 batch/config hotfix

사용자가 보고한 다음 오류는 CUDA 컴파일 에러가 아니라 Windows CMD가 배치/설정 파일 일부를 명령어로 잘못 해석한 증상입니다.

```txt
'LUE를' is not recognized as an internal or external command
'_REBUILD' is not recognized as an internal or external command
'UDA_HOME' is not recognized as an internal or external command
```

원인으로 가장 가능성이 큰 지점은 v28 `build.bat` 안의 UTF-8 Korean `rem` 주석과 LF-only 줄바꿈입니다. 특히 `KEY=VALUE를` 같은 주석 조각이 명령처럼 실행된 흔적이 보였습니다.

v29 조치:

- `avabm_cuda/build.bat`을 CRLF 줄바꿈으로 저장했습니다.
- `run.bat`도 CRLF 줄바꿈으로 다시 저장했습니다.
- 두 batch 파일 안의 Korean 주석을 제거하고 ASCII 주석만 남겼습니다.
- config 로딩은 `for /f eol=# ...` 대신 `findstr`로 `KEY=VALUE` 형식의 ASCII 설정 줄만 읽도록 바꿨습니다.
- `config.txt`도 CRLF 줄바꿈과 ASCII-safe comments로 바꿨습니다. Korean 설명은 이 문서에 유지했습니다.

이제 `LUE를`, `UDA_HOME`, `OBS`, `Activating` 같은 조각이 별도 명령으로 실행되면 안 됩니다.


## v30 build warning cleanup

- 사용자 로그에서 `binding.cpp` 컴파일 명령 앞부분에 Python/MSVC 기본 `/O2 /GL`이 남고, 뒤쪽의 `/Od /GL-`와 충돌해 `D9025` warning이 발생하는 것을 확인했습니다.
- `AVABMBuildExtension`이 MSVC compiler 초기화 직후 `compile_options`에서 `/O2`, `/GL`을 제거하도록 보강했습니다.
- 개발 빌드에서는 링크 단계의 `/LTCG`도 제거해 `/GL` 제거와 일관되게 했습니다.
- 문법상 이유: PyTorch의 Windows+ninja 빌드 경로는 `compiler.initialize()` 이후 생성된 `compile_options`를 읽어 `build.ninja`를 만듭니다. 따라서 초기화 전 제거만으로는 부족할 수 있어, 초기화 함수를 감싸는 방식으로 수정했습니다.
- 논리상 이유: CUDA kernel 성능은 `main.cu`의 nvcc 옵션이 좌우하고, `binding.cpp`는 Python-CUDA 연결층이므로 개발 중에는 `/Od /GL-`이 빠른 재빌드에 더 적합합니다.

## v31 runtime optimization + 40km/h cruise floor

사용자가 보고한 상태는 “데드락은 줄었지만 실행 성능이 너무 느리고, 일부 차량 속도가 지나치게 낮다”였습니다. v31은 기존 deadlock / spawn route / lane balancing 알고리즘 구조를 유지하면서, 계산 비용이 큰 부분을 줄이고 순항 희망속도 하한을 추가했습니다.

### 1. 결과를 바꾸지 않는 런타임 최적화

- `avabm_cuda/binding.cpp`
  - `sim.step()` 호출마다 새로 만들던 perception/decision 임시 tensor 16개를 thread-local cache로 재사용합니다.
  - 문법 이유: `torch::Tensor`를 C++ struct 멤버로 보관하면 reference count가 유지되어 함수가 끝나도 GPU scratch buffer가 살아 있습니다.
  - 논리 이유: CUDA kernel 알고리즘과 입력/출력 tensor는 그대로이고, 매 tick 반복되던 GPU allocator 호출만 제거합니다. 따라서 차량 의사결정 결과는 바뀌지 않고 Python/C++ 경계 overhead가 줄어듭니다.

- `avabm_cuda/main.cu`
  - 매 tick 여러 번 수행되는 world grid / reservation table / metrics clear를 `cudaMemsetAsync` 빠른 경로로 바꿨습니다.
  - 문법 이유: `cudaMemsetAsync`는 byte pattern을 채우므로 0과 -1 패턴에만 사용했습니다. float zero, int zero, int -1은 이 방식이 안전합니다.
  - 논리 이유: 기존 clear kernel과 최종 메모리 값은 같지만, CUDA runtime의 memset path가 더 가볍고 launch overhead도 줄어듭니다.

- `avabm_cuda/main.cu`
  - spawn 중 같은 step에 생성된 차량을 찾기 위한 `max_entities` 전체 full-scan을 기본적으로 피합니다.
  - 문법 이유: 새 차량을 초기화한 뒤 spatial grid에 즉시 CAS 방식으로 삽입해, 다음 spawn 검사에서 기존 `spawn_area_clear()` grid lookup을 그대로 사용합니다.
  - 논리 이유: “진입부가 비었는지 확인한다”는 알고리즘 의도는 유지하면서, 차량 수가 많을 때 spawn 1대마다 150,000 슬롯을 훑는 병목을 제거합니다. 필요하면 `CUDA_SPAWN_GRID_INSERT_FASTPATH=0`으로 이전 full-scan 경로를 다시 켤 수 있습니다.

- `avabm_cuda/main.cu`
  - IDM 식의 `powf(x, 4.0f)`와 `powf(x, 2.0f)`를 고정 차수 곱셈으로 바꿨습니다.
  - 문법 이유: 지수가 항상 2 또는 4로 고정되어 있으므로 일반 지수 함수 호출 대신 `x*x`, `(x*x)*(x*x)`로 쓸 수 있습니다.
  - 논리 이유: 수식 의미는 동일하고, 모든 차량의 car-following 가속도 계산에서 반복되는 비싼 device math 호출을 줄입니다.

- `config.txt`
  - 실행 성능 우선 요청에 맞춰 기본 빌드를 `CUDA_BUILD_MODE=release`, `CUDA_OPT_LEVEL=3`로 변경했습니다.
  - 첫 빌드는 더 오래 걸릴 수 있지만, build fingerprint가 같으면 이후 실행은 빌드를 건너뜁니다.

### 2. 40km/h 이상 순항 희망속도

- `config.txt`
  - `SPEED_MIN_CRUISE_ENABLED=1`
  - `SPEED_MIN_CRUISE_KMH=40.0`

- `avabm_cuda/main.cu`
  - `desired_speed_ecs()`에서 안전 제약이 없는 순항 희망속도의 하한을 40km/h로 적용합니다.
  - 새로 진입하는 차량도 낮은 초기속도로 오래 기어가지 않도록 spawn 초기속도에 같은 하한을 반영했습니다.
  - spawn entry gap 계산도 같은 유효 속도 기준으로 넓혀, 초기속도만 올리고 간격 검사는 그대로 두는 위험한 변경을 피했습니다.
  - 빨간불, 앞차, 교차로 양보, 차선변경 no-start, 급회전 connector 같은 안전 제약은 그대로 우선합니다.

중요: “모든 순간의 실제 속도를 무조건 40km/h 이상으로 강제”하면 빨간불/앞차/급회전에서도 감속하지 못해 충돌과 신호 위반이 생깁니다. v31은 안전을 깨는 hard clamp가 아니라, 기존 알고리즘 안에서 자유 주행 목표 속도를 40km/h 이상으로 올리는 방식입니다. 따라서 정체나 안전 제약이 없는데도 낮은 희망속도 때문에 계속 느리게 가는 차량을 줄이는 것이 목적입니다.

### 3. v31 설정 되돌리기

성능 최적화는 유지하되 40km/h 순항 하한만 끄려면:

```txt
SPEED_MIN_CRUISE_ENABLED=0
```

순항 하한을 다른 값으로 조절하려면:

```txt
SPEED_MIN_CRUISE_KMH=50.0
```

첫 clean build 시간을 다시 줄이고 싶으면:

```txt
CUDA_BUILD_MODE=fastdev
CUDA_OPT_LEVEL=1
```

단, 이 경우 런타임 kernel 성능은 release/O3보다 낮을 수 있습니다.
