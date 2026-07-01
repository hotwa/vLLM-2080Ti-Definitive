#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$ROOT"
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"
LOG_DIR=${LOG_DIR:-"$ROOT/build-logs"}
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/build-$STAMP.log"
FLASHQLA_REPO=${FLASHQLA_REPO:-https://github.com/weicj/FlashQLA-SM70-SM75.git}
FLASHQLA_DIR=${FLASHQLA_DIR:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
CUTLASS_REPO=${CUTLASS_REPO:-https://github.com/nvidia/cutlass.git}
CUTLASS_REVISION=${CUTLASS_REVISION:-v4.4.2}
CUTLASS_DIR=${CUTLASS_DIR:-"$ROOT/.deps/cutlass-src"}
TRITON_REPO=${TRITON_REPO:-https://github.com/triton-lang/triton.git}
TRITON_TAG=${TRITON_TAG:-v3.6.0}
TRITON_KERNELS_DIR=${TRITON_KERNELS_DIR:-"$ROOT/.deps/triton-src"}
BUILD_PYPI_OFFICIAL_INDEX=${BUILD_PYPI_OFFICIAL_INDEX:-https://pypi.org/simple}
BUILD_PYPI_FOREIGN_INDEX=${BUILD_PYPI_FOREIGN_INDEX:-https://pypi.python.org/simple}
BUILD_PYPI_DOMESTIC_INDEX=${BUILD_PYPI_DOMESTIC_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
BUILD_GIT_FOREIGN_REPO_PREFIX=${BUILD_GIT_FOREIGN_REPO_PREFIX:-https://gh-proxy.com/}
BUILD_GIT_DOMESTIC_REPO_PREFIX=${BUILD_GIT_DOMESTIC_REPO_PREFIX:-https://ghfast.top/}
BUILD_GIT_OFFICIAL_PROBE=${BUILD_GIT_OFFICIAL_PROBE:-${FLASHQLA_REPO}/info/refs?service=git-upload-pack}
BUILD_GIT_FOREIGN_PROBE=${BUILD_GIT_FOREIGN_PROBE:-${BUILD_GIT_FOREIGN_REPO_PREFIX}${BUILD_GIT_OFFICIAL_PROBE}}
BUILD_GIT_DOMESTIC_PROBE=${BUILD_GIT_DOMESTIC_PROBE:-${BUILD_GIT_DOMESTIC_REPO_PREFIX}${BUILD_GIT_OFFICIAL_PROBE}}
# Preflight uses small URL samples to choose the fastest route before install.
# Primary timeouts only bound the later real install/clone attempt on that route.
BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS=${BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS:-${BUILD_PREFLIGHT_TIMEOUT_SECONDS:-5}}
BUILD_PREFLIGHT_TIMEOUT_SECONDS=${BUILD_PREFLIGHT_TIMEOUT_SECONDS:-$BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS}
BUILD_PYPI_PRIMARY_TIMEOUT_SECONDS=${BUILD_PYPI_PRIMARY_TIMEOUT_SECONDS:-60}
BUILD_GIT_PRIMARY_TIMEOUT_SECONDS=${BUILD_GIT_PRIMARY_TIMEOUT_SECONDS:-120}
BUILD_GIT_MIRROR_PREFIXES=${BUILD_GIT_MIRROR_PREFIXES:-https://gh-proxy.com/ https://ghfast.top/}
TOTAL_STEPS=15
STEP_INDEX=0
BUILD_STARTED_AT=$(date +%s)
VERSION=${VERSION:-$FORK_RELEASE}

banner() {
  cat <<EOF
============================================================
 $PROJECT_NAME v$VERSION
 One-click source build
 Runtime: $RUNTIME_IDENTITY
 Author: $PROJECT_AUTHOR
============================================================
EOF
}

fail() {
  echo
  echo "BUILD FAILED"
  echo "Log: $LOG"
  echo "$*" >&2
  exit 1
}

format_seconds() {
  local seconds=$1
  printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

detect_cpu_threads() {
  local threads
  threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if ! is_positive_integer "$threads"; then
    threads=$(nproc 2>/dev/null || true)
  fi
  if ! is_positive_integer "$threads"; then
    threads=4
  fi
  echo "$threads"
}

detect_memory_gb() {
  local mem_kb mem_gb
  mem_kb=$(awk '/MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)
  mem_gb=$(( mem_kb / 1024 / 1024 ))
  if (( mem_gb < 1 )); then
    mem_gb=1
  fi
  echo "$mem_gb"
}

select_max_jobs() {
  local threads=$1
  local jobs
  jobs=$((threads / 2))
  (( jobs >= 1 )) || jobs=1
  echo "$jobs"
}

validate_max_jobs_range() {
  local jobs=$1
  local threads=$2
  is_positive_integer "$jobs" || fail "MAX_JOBS must be a positive integer."
  if (( jobs < 1 || jobs > threads )); then
    fail "MAX_JOBS must be between 1 and CPU_THREADS ($threads)."
  fi
}

configure_build_parallelism() {
  local answer

  if [[ "$MAX_JOBS_SOURCE" != "auto" ]]; then
    return 0
  fi
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    return 0
  fi

  cat <<EOF

Build thread selection:
  CPU threads detected: $CPU_THREADS
  Recommended build threads: $MAX_JOBS
  Allowed range: 1-$CPU_THREADS

Press Enter to use the recommended value, or type a build thread count:
EOF

  while true; do
    read -r answer
    if [[ -z "$answer" ]]; then
      return 0
    fi
    if is_positive_integer "$answer" && (( answer >= 1 && answer <= CPU_THREADS )); then
      MAX_JOBS="$answer"
      MAX_JOBS_SOURCE=manual-prompt
      export MAX_JOBS
      return 0
    fi
    echo "Please enter a number from 1 to $CPU_THREADS, or press Enter for $MAX_JOBS:"
  done
}

validate_cuda_dev_files() {
  local lib_dir="$CUDA_HOME/targets/x86_64-linux/lib"
  local missing=()
  local file

  for file in libcudadevrt.a libcudart_static.a libculibos.a; do
    if [[ ! -f "$lib_dir/$file" ]]; then
      missing+=("$lib_dir/$file")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    {
      echo "CUDA toolkit is missing required development static libraries:"
      printf '  %s\n' "${missing[@]}"
      echo
      echo "Install or repair the CUDA ${VALIDATED_CUDA_VERSION} development toolkit."
      echo "On Debian/Ubuntu NVIDIA CUDA repos, this is usually:"
      echo "  sudo apt-get install --reinstall cuda-cudart-dev-12-8"
    } >&2
    fail "Incomplete CUDA development toolkit at CUDA_HOME=$CUDA_HOME"
  fi
}

confirm_install() {
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    fail "Interactive confirmation is required. Re-run with ASSUME_YES=1 for non-interactive install."
  fi

  cat <<EOF

This script will build and install vLLM 2080 Ti Definitive from source into:
  $ROOT/.venv

Expected time:
  30-60 minutes on a typical dual RTX 2080 Ti host.

Build parallelism:
  CPU threads: $CPU_THREADS
  System memory: ${MEMORY_GB}GB
  MAX_JOBS: $MAX_JOBS ($MAX_JOBS_SOURCE)

It will download Python/CUDA dependencies, compile CUDA extensions, and keep
logs under:
  $LOG

Continue? [y/N]:
EOF

  local answer
  while true; do
    read -r answer
    case "$answer" in
      y|Y)
        return 0
        ;;
      n|N|"")
        echo "Build cancelled."
        exit 0
        ;;
      *)
        echo "Please type y to continue or n to exit:"
        ;;
    esac
  done
}

prompt_yes_no_timeout() {
  local prompt=$1
  local timeout_seconds=${2:-10}
  local default_answer=${3:-y}
  local answer=""

  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi

  if [[ ! -t 0 ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi

  printf '%s ' "$prompt"
  if read -r -t "$timeout_seconds" answer; then
    case "$answer" in
      y|Y|yes|YES) printf 'y\n'; return 0 ;;
      n|N|no|NO) printf 'n\n'; return 0 ;;
      *) printf '%s\n' "$default_answer"; return 0 ;;
    esac
  fi

  printf '%s\n' "$default_answer"
}

measure_network_url_ms() {
  local url=$1
  local timeout_seconds=${2:-6}
  local start end elapsed
  start=$(date +%s%3N)

  if command -v curl >/dev/null 2>&1; then
    curl -L --max-time "$timeout_seconds" --connect-timeout "$timeout_seconds" \
      --fail --silent --show-error --output /dev/null "$url" >/dev/null 2>&1 || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -q --timeout="$timeout_seconds" -O /dev/null "$url" >/dev/null 2>&1 || return 1
  else
    return 1
  fi

  end=$(date +%s%3N)
  elapsed=$((end - start))
  (( elapsed > 0 )) || elapsed=1
  printf '%s\n' "$elapsed"
}

pypi_probe_url() {
  local index_url=$1
  index_url=${index_url%/}
  if [[ "$index_url" == */simple ]]; then
    printf '%s/pip/\n' "$index_url"
  else
    printf '%s\n' "$index_url"
  fi
}

probe_download_route() {
  local mode=$1
  local pypi_url=$2
  local git_url=$3
  local timeout_seconds=${BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS:-8}
  local pypi_probe
  local pypi_ms git_ms total_ms

  pypi_probe=$(pypi_probe_url "$pypi_url")
  if ! pypi_ms=$(measure_network_url_ms "$pypi_probe" "$timeout_seconds"); then
    printf 'Preflight: %-8s unavailable at Python index %s\n' "$mode" "$pypi_probe" >&2
    return 1
  fi
  if ! git_ms=$(measure_network_url_ms "$git_url" "$timeout_seconds"); then
    printf 'Preflight: %-8s unavailable at Git probe %s\n' "$mode" "$git_url" >&2
    return 1
  fi
  total_ms=$((pypi_ms + git_ms))
  printf 'Preflight: %-8s route %5sms total  PyPI=%sms  Git=%sms\n' \
    "$mode" "$total_ms" "$pypi_ms" "$git_ms" >&2
  printf '%s\t%s\t%s\t%s\n' "$total_ms" "$mode" "$pypi_ms" "$git_ms"
}

choose_download_mode() {
  local measurements=()
  local line selected_line
  local selected_mode="official"
  local selected_total="unknown"
  local selected_pypi="unknown"
  local selected_git="unknown"
  local probe_tmp
  local -a probe_jobs=()
  local item pid mode

  echo "Preflight: benchmarking download routes..."
  echo "Preflight: sample timeout is ${BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS}s per URL probe."
  if ! is_positive_integer "$BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS"; then
    fail "BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS must be a positive integer."
  fi

  probe_tmp=$(mktemp -d)
  probe_download_route official "$BUILD_PYPI_OFFICIAL_INDEX" "$BUILD_GIT_OFFICIAL_PROBE" \
    >"$probe_tmp/official.out" 2>"$probe_tmp/official.err" &
  probe_jobs+=("$!:official")
  probe_download_route foreign "$BUILD_PYPI_FOREIGN_INDEX" "$BUILD_GIT_FOREIGN_PROBE" \
    >"$probe_tmp/foreign.out" 2>"$probe_tmp/foreign.err" &
  probe_jobs+=("$!:foreign")
  probe_download_route domestic "$BUILD_PYPI_DOMESTIC_INDEX" "$BUILD_GIT_DOMESTIC_PROBE" \
    >"$probe_tmp/domestic.out" 2>"$probe_tmp/domestic.err" &
  probe_jobs+=("$!:domestic")

  for item in "${probe_jobs[@]}"; do
    pid=${item%%:*}
    wait "$pid" || true
  done

  for mode in official foreign domestic; do
    [[ ! -s "$probe_tmp/$mode.err" ]] || cat "$probe_tmp/$mode.err" >&2
    if [[ -s "$probe_tmp/$mode.out" ]]; then
      line=$(head -n 1 "$probe_tmp/$mode.out")
      measurements+=("$line")
    fi
  done

  rm -rf "$probe_tmp"

  if ((${#measurements[@]} == 0)); then
    echo "Preflight: network looks poor; no download route is currently reachable."
    local answer
    answer=$(prompt_yes_no_timeout "Continue anyway? [y/N]:" 10 n)
    [[ "$answer" == "y" ]] || fail "Build cancelled because network preflight failed."
    selected_mode="official"
  else
    selected_line=$(printf '%s\n' "${measurements[@]}" | sort -n -k1,1 | head -n 1)
    selected_mode=$(printf '%s\n' "$selected_line" | awk -F '\t' '{print $2}')
    selected_total=$(printf '%s\n' "$selected_line" | awk -F '\t' '{print $1}')
    selected_pypi=$(printf '%s\n' "$selected_line" | awk -F '\t' '{print $3}')
    selected_git=$(printf '%s\n' "$selected_line" | awk -F '\t' '{print $4}')
  fi

  case "$selected_mode" in
    official)
      BUILD_PYPI_ACTIVE_INDEX="$BUILD_PYPI_OFFICIAL_INDEX"
      BUILD_GIT_ACTIVE_PREFIX=""
      echo "Preflight: selected official download route (${selected_total}ms sample total; PyPI=${selected_pypi}ms; Git=${selected_git}ms)."
      ;;
    foreign)
      BUILD_PYPI_ACTIVE_INDEX="$BUILD_PYPI_FOREIGN_INDEX"
      BUILD_GIT_ACTIVE_PREFIX="$BUILD_GIT_FOREIGN_REPO_PREFIX"
      echo "Preflight: selected foreign mirror route (${selected_total}ms sample total; PyPI=${selected_pypi}ms; Git=${selected_git}ms)."
      local answer
      answer=$(prompt_yes_no_timeout "Continue with mirror route? [Y/n]:" 10 y)
      [[ "$answer" != "n" ]] || fail "Build cancelled by user."
      ;;
    domestic)
      BUILD_PYPI_ACTIVE_INDEX="$BUILD_PYPI_DOMESTIC_INDEX"
      BUILD_GIT_ACTIVE_PREFIX="$BUILD_GIT_DOMESTIC_REPO_PREFIX"
      echo "Preflight: selected domestic mirror route (${selected_total}ms sample total; PyPI=${selected_pypi}ms; Git=${selected_git}ms)."
      local answer
      answer=$(prompt_yes_no_timeout "Continue with domestic mirror route? [Y/n]:" 10 y)
      [[ "$answer" != "n" ]] || fail "Build cancelled by user."
      ;;
  esac

  export BUILD_PYPI_ACTIVE_INDEX BUILD_GIT_ACTIVE_PREFIX
}

check_cpu_cores() {
  local cores=${CPU_THREADS:-$(detect_cpu_threads)}
  if (( cores < 4 )); then
    echo "Preflight: CPU core count is low ($cores cores). Build may be slow."
  else
    echo "Preflight: CPU core count is OK ($cores cores)."
  fi
}

check_memory_headroom() {
  local mem_kb mem_gb
  mem_kb=$(awk '/MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)
  mem_gb=$(( mem_kb / 1024 / 1024 ))
  if (( mem_gb < 16 )); then
    echo "Preflight: system memory is low (${mem_gb}GB). Recommended is 16GB+."
  else
    echo "Preflight: system memory is OK (${mem_gb}GB)."
  fi
}

check_disk_headroom() {
  local free_kb free_gb
  free_kb=$(df -Pk "$ROOT" | awk 'NR == 2 { print $4 }')
  free_gb=$(( free_kb / 1024 / 1024 ))
  if (( free_gb < 80 )); then
    echo "Preflight: free disk under source tree is low (${free_gb}GB). Recommended is 80GB+ for a clean build."
  else
    echo "Preflight: free disk under source tree is OK (${free_gb}GB)."
  fi
}

check_gpu_hardware() {
  local gpu_summary
  local gpu_count
  local high_vram_count

  gpu_summary=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
  if [[ -z "$gpu_summary" ]]; then
    echo "Preflight: no NVIDIA GPU was detected."
    return 0
  fi

  echo "Preflight: detected GPUs:"
  echo "$gpu_summary" | sed 's/^/  - /'

  gpu_count=$(printf '%s\n' "$gpu_summary" | sed '/^[[:space:]]*$/d' | wc -l)
  high_vram_count=$(
    printf '%s\n' "$gpu_summary" |
      awk -F, '
        {
          mem = $2
          gsub(/[^0-9]/, "", mem)
          if (mem + 0 >= 20000) count += 1
        }
        END { print count + 0 }
      '
  )

  if (( gpu_count < 2 || high_vram_count < 2 )); then
    echo "Preflight: recommended target is two RTX 2080 Ti-class GPUs with about 22GB VRAM each."
    echo "Preflight: current GPU layout may still build, but validated profiles may not fit or may run slower."
  else
    echo "Preflight: GPU layout matches the recommended dual 2080 Ti-class target."
  fi
}

check_gpu_topology() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  echo "Preflight: GPU topology:"
  nvidia-smi topo -m 2>/dev/null | sed 's/^/  /' || echo "  topology query unavailable"
}

print_step_header() {
  local title=$1
  STEP_INDEX=$((STEP_INDEX + 1))
  echo
  echo "------------------------------------------------------------"
  echo " Step $STEP_INDEX/$TOTAL_STEPS: $title"
  echo "------------------------------------------------------------"
  echo "[$(date '+%F %T')] Step $STEP_INDEX/$TOTAL_STEPS: $title" >> "$LOG"
}

run_with_progress() {
  local title=$1
  shift

  if run_with_progress_status "$title" "$@"; then
    return 0
  fi

  echo
  echo "Last log lines:"
  tail -n 80 "$LOG" || true
  fail "Step failed: $title"
}

run_with_progress_status() {
  local title=$1
  shift
  local step_start
  local tmp_status
  local pid
  local rc=0
  local elapsed
  local spinner='|/-\'
  local spin_i=0

  if [[ "${REUSE_STEP_HEADER:-0}" == "1" ]]; then
    echo
    echo "  $title"
    echo "[$(date '+%F %T')] $title" >> "$LOG"
  else
    print_step_header "$title"
  fi
  step_start=$(date +%s)
  tmp_status=$(mktemp)

  (
    set +e
    "$@" >>"$LOG" 2>&1
    echo $? > "$tmp_status"
  ) &
  pid=$!

  while kill -0 "$pid" >/dev/null 2>&1; do
    elapsed=$(format_seconds "$(( $(date +%s) - step_start ))")
    if [[ -t 1 ]]; then
      printf '\r[%c] %s running... elapsed %s | log: %s' \
        "${spinner:spin_i%${#spinner}:1}" "$title" "$elapsed" "$LOG"
      spin_i=$((spin_i + 1))
      if [[ -s "$LOG" ]]; then
        printf '\n'
        tail -n "${TAIL_LINES:-3}" "$LOG" | sed 's/^/    /'
        printf '\033[%sA' "$(( ${TAIL_LINES:-3} + 1 ))"
      fi
    else
      echo "[$elapsed] $title still running. Log: $LOG"
      if [[ -s "$LOG" ]]; then
        tail -n "${TAIL_LINES:-3}" "$LOG" | sed 's/^/    /'
      fi
    fi
    sleep "${PROGRESS_INTERVAL:-5}"
  done

  wait "$pid" || true
  if [[ -s "$tmp_status" ]]; then
    rc=$(cat "$tmp_status")
  else
    rc=1
  fi
  rm -f "$tmp_status"

  if [[ -t 1 ]]; then
    printf '\r'
    printf '%*s\r' "$(tput cols 2>/dev/null || echo 120)" ''
  fi
  elapsed=$(format_seconds "$(( $(date +%s) - step_start ))")

  if [[ "$rc" == "0" ]]; then
    echo "OK: $title completed in $elapsed"
    return 0
  else
    echo "FAILED: $title after $elapsed"
    return "$rc"
  fi
}

run_uv_pip_with_mirror_fallback() {
  local title=$1
  shift
  local timeout_seconds=${BUILD_PYPI_PRIMARY_TIMEOUT_SECONDS:-300}
  local primary_index=${BUILD_PYPI_INDEX:-${BUILD_PYPI_ACTIVE_INDEX:-}}
  local mirror_index=${BUILD_PYPI_MIRROR_INDEX:-${BUILD_PYPI_DOMESTIC_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}}
  local wheelhouse_dir=${BUILD_WHEELHOUSE_DIR:-}
  local -a uv_env=(env)

  if [[ -z "$wheelhouse_dir" && -d /data/wheelhouse/cu128 ]]; then
    wheelhouse_dir=/data/wheelhouse/cu128
  fi

  if [[ -n "$wheelhouse_dir" ]]; then
    if [[ -d "$wheelhouse_dir" ]]; then
      uv_env+=("UV_FIND_LINKS=$wheelhouse_dir")
    else
      fail "BUILD_WHEELHOUSE_DIR does not exist: $wheelhouse_dir"
    fi
  fi

  if [[ "${BUILD_PYPI_MIRROR_FALLBACK:-1}" != "1" ]]; then
    run_with_progress "$title" "${uv_env[@]}" uv pip "$@"
    return 0
  fi

  if ! is_positive_integer "$timeout_seconds"; then
    fail "BUILD_PYPI_PRIMARY_TIMEOUT_SECONDS must be a positive integer."
  fi

  echo "Python package mirror fallback: enabled"
  echo "  Selected-route attempt timeout: ${timeout_seconds}s"
  echo "  Mirror index: $mirror_index"
  if [[ -n "$wheelhouse_dir" ]]; then
    echo "  Local wheelhouse: $wheelhouse_dir"
  fi
  {
    echo "Python package mirror fallback: enabled"
    echo "Selected-route attempt timeout: ${timeout_seconds}s"
    echo "Mirror index: $mirror_index"
    if [[ -n "$wheelhouse_dir" ]]; then
      echo "Local wheelhouse: $wheelhouse_dir"
    fi
  } >> "$LOG"

  print_step_header "$title"

  local REUSE_STEP_HEADER=1
  if [[ -n "$primary_index" ]]; then
    if run_with_progress_status "Selected-route index attempt" \
      "${uv_env[@]}" UV_DEFAULT_INDEX="$primary_index" timeout --preserve-status "$timeout_seconds" uv pip "$@"; then
      return 0
    fi
  else
    if run_with_progress_status "Selected-route index attempt" \
      "${uv_env[@]}" timeout --preserve-status "$timeout_seconds" uv pip "$@"; then
      return 0
    fi
  fi

  echo
  echo "Selected-route Python package install was too slow or failed; retrying with mirror."
  echo "Selected-route Python package install was too slow or failed; retrying with mirror." >> "$LOG"
  if run_with_progress_status "Mirror index attempt" \
    "${uv_env[@]}" UV_DEFAULT_INDEX="$mirror_index" UV_INDEX_STRATEGY=unsafe-best-match uv pip "$@"; then
    return 0
  fi

  echo
  echo "Last log lines:"
  tail -n 80 "$LOG" || true
  fail "Step failed: $title"
}

git_url_with_prefix() {
  local prefix=$1
  local repo=$2
  if [[ "$prefix" == *"{}"* ]]; then
    printf '%s\n' "${prefix//\{\}/$repo}"
  else
    printf '%s%s\n' "$prefix" "$repo"
  fi
}

active_git_url() {
  local repo=$1
  if [[ -n "${BUILD_GIT_ACTIVE_PREFIX:-}" ]]; then
    git_url_with_prefix "$BUILD_GIT_ACTIVE_PREFIX" "$repo"
  else
    printf '%s\n' "$repo"
  fi
}

fetch_git_repo_once() {
  local repo=$1
  if [[ -d "$FLASHQLA_DIR/.git" ]]; then
    git -C "$FLASHQLA_DIR" fetch --depth=1 "$repo"
    git -C "$FLASHQLA_DIR" checkout -q FETCH_HEAD
  else
    git clone --depth=1 "$repo" "$FLASHQLA_DIR"
  fi
}

fetch_git_tag_once() {
  local repo=$1
  local dir=$2
  local ref=$3

  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" fetch --depth=1 origin "tag" "$ref"
    git -C "$dir" checkout -q "$ref"
  else
    git clone --depth=1 --branch "$ref" --single-branch "$repo" "$dir"
  fi
}

fetch_git_branch_or_tag_once() {
  local repo=$1
  local dir=$2
  local ref=$3

  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" fetch --depth=1 origin "refs/tags/$ref:refs/tags/$ref"
    git -C "$dir" checkout -q "$ref"
  else
    git clone --depth=1 --branch "$ref" --single-branch "$repo" "$dir"
  fi
}

fetch_flashqla_with_fallback() {
  local timeout_seconds=$BUILD_GIT_PRIMARY_TIMEOUT_SECONDS
  local mirror_prefix
  local mirror_repo
  local primary_repo

  if ! is_positive_integer "$timeout_seconds"; then
    fail "BUILD_GIT_PRIMARY_TIMEOUT_SECONDS must be a positive integer."
  fi

  mkdir -p "$(dirname -- "$FLASHQLA_DIR")"
  if [[ -e "$FLASHQLA_DIR" && ! -d "$FLASHQLA_DIR/.git" ]]; then
    fail "FlashQLA dir exists but is not a git checkout: $FLASHQLA_DIR"
  fi

  echo "Git fetch fallback: enabled"
  echo "  Selected-route attempt timeout: ${timeout_seconds}s"
  echo "  Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  {
    echo "Git fetch fallback: enabled"
    echo "Selected-route attempt timeout: ${timeout_seconds}s"
    echo "Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  } >> "$LOG"

  primary_repo=$(active_git_url "$FLASHQLA_REPO")
  local REUSE_STEP_HEADER=1
  if run_with_progress_status "Selected-route git attempt" \
    env FLASHQLA_DIR="$FLASHQLA_DIR" timeout --preserve-status "$timeout_seconds" bash -c '
      repo=$1
      if [[ -d "$FLASHQLA_DIR/.git" ]]; then
        git -C "$FLASHQLA_DIR" fetch --depth=1 "$repo"
        git -C "$FLASHQLA_DIR" checkout -q FETCH_HEAD
      else
        git clone --depth=1 "$repo" "$FLASHQLA_DIR"
      fi
    ' _ "$primary_repo"; then
    return 0
  fi

  for mirror_prefix in $BUILD_GIT_MIRROR_PREFIXES; do
    [[ -n "$mirror_prefix" ]] || continue
    mirror_repo=$(git_url_with_prefix "$mirror_prefix" "$FLASHQLA_REPO")
    echo
    echo "Selected-route git fetch was too slow or failed; retrying with mirror: $mirror_repo"
    echo "Selected-route git fetch was too slow or failed; retrying with mirror: $mirror_repo" >> "$LOG"
    if run_with_progress_status "Mirror git attempt" fetch_git_repo_once "$mirror_repo"; then
      return 0
    fi
  done

  echo
  echo "Last log lines:"
  tail -n 80 "$LOG" || true
  fail "Step failed: Fetch FlashQLA SM70/SM75 backend"
}

fetch_cutlass_with_fallback() {
  local timeout_seconds=$BUILD_GIT_PRIMARY_TIMEOUT_SECONDS
  local mirror_prefix
  local mirror_repo
  local primary_repo

  if ! is_positive_integer "$timeout_seconds"; then
    fail "BUILD_GIT_PRIMARY_TIMEOUT_SECONDS must be a positive integer."
  fi

  mkdir -p "$(dirname -- "$CUTLASS_DIR")"
  if [[ -e "$CUTLASS_DIR" && ! -d "$CUTLASS_DIR/.git" ]]; then
    fail "CUTLASS dir exists but is not a git checkout: $CUTLASS_DIR"
  fi

  echo "CUTLASS fetch fallback: enabled"
  echo "  Selected-route attempt timeout: ${timeout_seconds}s"
  echo "  Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  {
    echo "CUTLASS fetch fallback: enabled"
    echo "Selected-route attempt timeout: ${timeout_seconds}s"
    echo "Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  } >> "$LOG"

  primary_repo=$(active_git_url "$CUTLASS_REPO")
  local REUSE_STEP_HEADER=1
  if run_with_progress_status "Selected-route CUTLASS attempt" \
    env CUTLASS_DIR="$CUTLASS_DIR" timeout --preserve-status "$timeout_seconds" bash -c '
      repo=$1
      ref=$2
      dir=$3
      if [[ -d "$dir/.git" ]]; then
        git -C "$dir" fetch --depth=1 origin tag "$ref"
        git -C "$dir" checkout -q "$ref"
      else
        git clone --depth=1 --branch "$ref" --single-branch "$repo" "$dir"
      fi
    ' _ "$primary_repo" "$CUTLASS_REVISION" "$CUTLASS_DIR"; then
    return 0
  fi

  for mirror_prefix in $BUILD_GIT_MIRROR_PREFIXES; do
    [[ -n "$mirror_prefix" ]] || continue
    mirror_repo=$(git_url_with_prefix "$mirror_prefix" "$CUTLASS_REPO")
    echo
    echo "Selected-route CUTLASS fetch was too slow or failed; retrying with mirror: $mirror_repo"
    echo "Selected-route CUTLASS fetch was too slow or failed; retrying with mirror: $mirror_repo" >> "$LOG"
    if run_with_progress_status "Mirror CUTLASS attempt" \
      fetch_git_tag_once "$mirror_repo" "$CUTLASS_DIR" "$CUTLASS_REVISION"; then
      return 0
    fi
  done

  echo
  echo "Last log lines:"
  tail -n 80 "$LOG" || true
fail "Step failed: Fetch CUTLASS source"
}

fetch_triton_with_fallback() {
  local timeout_seconds=$BUILD_GIT_PRIMARY_TIMEOUT_SECONDS
  local mirror_prefix
  local mirror_repo
  local primary_repo

  if ! is_positive_integer "$timeout_seconds"; then
    fail "BUILD_GIT_PRIMARY_TIMEOUT_SECONDS must be a positive integer."
  fi

  mkdir -p "$(dirname -- "$TRITON_KERNELS_DIR")"
  if [[ -e "$TRITON_KERNELS_DIR" && ! -d "$TRITON_KERNELS_DIR/.git" ]]; then
    fail "TRITON_KERNELS_DIR exists but is not a git checkout: $TRITON_KERNELS_DIR"
  fi

  echo "Triton fetch fallback: enabled"
  echo "  Selected-route attempt timeout: ${timeout_seconds}s"
  echo "  Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  {
    echo "Triton fetch fallback: enabled"
    echo "Selected-route attempt timeout: ${timeout_seconds}s"
    echo "Mirror prefixes: ${BUILD_GIT_MIRROR_PREFIXES:-none}"
  } >> "$LOG"

  primary_repo=$(active_git_url "$TRITON_REPO")
  local REUSE_STEP_HEADER=1
  if run_with_progress_status "Selected-route Triton attempt" \
    env TRITON_KERNELS_DIR="$TRITON_KERNELS_DIR" timeout --preserve-status "$timeout_seconds" bash -c '
      repo=$1
      ref=$2
      dir=$3
      if [[ -d "$dir/.git" ]]; then
        git -C "$dir" fetch --depth=1 origin "refs/tags/$ref:refs/tags/$ref"
        git -C "$dir" checkout -q "$ref"
      else
        git clone --depth=1 --branch "$ref" --single-branch "$repo" "$dir"
      fi
    ' _ "$primary_repo" "$TRITON_TAG" "$TRITON_KERNELS_DIR"; then
    return 0
  fi

  for mirror_prefix in $BUILD_GIT_MIRROR_PREFIXES; do
    [[ -n "$mirror_prefix" ]] || continue
    mirror_repo=$(git_url_with_prefix "$mirror_prefix" "$TRITON_REPO")
    echo
    echo "Selected-route Triton fetch was too slow or failed; retrying with mirror: $mirror_repo"
    echo "Selected-route Triton fetch was too slow or failed; retrying with mirror: $mirror_repo" >> "$LOG"
    if run_with_progress_status "Mirror Triton attempt" \
      fetch_git_branch_or_tag_once "$mirror_repo" "$TRITON_KERNELS_DIR" "$TRITON_TAG"; then
      return 0
    fi
  done

  echo
  echo "Last log lines:"
  tail -n 80 "$LOG" || true
  fail "Step failed: Fetch Triton kernels source"
}

install_torch_from_wheelhouse() {
  local wheelhouse_dir=${BUILD_WHEELHOUSE_DIR:-}
  if [[ -z "$wheelhouse_dir" && -d /data/wheelhouse/cu128 ]]; then
    wheelhouse_dir=/data/wheelhouse/cu128
  fi
  [[ -n "$wheelhouse_dir" ]] || return 0
  [[ -d "$wheelhouse_dir" ]] || return 0

  run_with_progress "Install torch from local wheelhouse" \
    env UV_NO_INDEX=1 UV_FIND_LINKS="$wheelhouse_dir" UV_LINK_MODE=copy \
    uv pip install --python .venv/bin/python --no-deps --no-index --find-links "$wheelhouse_dir" \
      "torch==${VALIDATED_TORCH_VERSION}"
}

install_build_frontend_requirements() {
  run_uv_pip_with_mirror_fallback "Install build frontend requirements" \
    install --python .venv/bin/python -U \
    "pip" \
    "wheel" \
    "packaging>=24.2" \
    "setuptools>=77.0.3,<81.0.0" \
    "setuptools-scm>=8" \
    "cmake>=3.26.1" \
    "ninja"
}

ensure_build_frontend_tools() {
  .venv/bin/python - <<'PY'
from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

try:
    from packaging.version import Version
except Exception as exc:  # pragma: no cover - hard fail in build script
    raise SystemExit(f"packaging import failed: {exc}") from exc

required_modules = {
    "setuptools": ("77.0.3", "81.0.0"),
    "setuptools_scm": ("8.0.0", None),
}

for module_name, (min_version, max_exclusive) in required_modules.items():
    module = import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        raise SystemExit(f"{module_name} has no __version__")
    parsed = Version(version)
    if parsed < Version(min_version):
        raise SystemExit(f"{module_name} version too old: {version} < {min_version}")
    if max_exclusive is not None and parsed >= Version(max_exclusive):
        raise SystemExit(
            f"{module_name} version too new: {version} >= {max_exclusive}"
        )

cmake_path = Path(sys.prefix) / "bin" / "cmake"
if not cmake_path.is_file():
    raise SystemExit(f"missing venv cmake: {cmake_path}")

output = subprocess.check_output([str(cmake_path), "--version"], text=True)
first_line = output.splitlines()[0].strip()
version_text = first_line.rsplit(" ", 1)[-1]
if Version(version_text) < Version("3.26.1"):
    raise SystemExit(f"cmake version too old: {version_text} < 3.26.1")

print(f"build_frontend_ok setuptools={import_module('setuptools').__version__} "
      f"setuptools_scm={import_module('setuptools_scm').__version__} "
      f"cmake={version_text}")
PY
}

run_step() {
  local title=$1
  shift
  print_step_header "$title"
  "$@" 2>&1 | tee -a "$LOG"
}

mkdir -p "$LOG_DIR"
touch "$LOG"
banner | tee -a "$LOG"

echo "Build log: $LOG"
echo "Source: $ROOT"

CPU_THREADS=${CPU_THREADS:-$(detect_cpu_threads)}
if ! is_positive_integer "$CPU_THREADS"; then
  fail "CPU_THREADS must be a positive integer when set explicitly."
fi
MEMORY_GB=${MEMORY_GB:-$(detect_memory_gb)}
if ! is_positive_integer "$MEMORY_GB"; then
  fail "MEMORY_GB must be a positive integer when set explicitly."
fi
if [[ -z "${MAX_JOBS:-}" ]]; then
  MAX_JOBS=$(select_max_jobs "$CPU_THREADS")
  MAX_JOBS_SOURCE=auto
else
  validate_max_jobs_range "$MAX_JOBS" "$CPU_THREADS"
  MAX_JOBS_SOURCE=manual
fi
export CPU_THREADS
export MEMORY_GB
export MAX_JOBS

cd "$ROOT"

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
  fail "Run this script from the vLLM 2080 Ti Definitive source tree."
fi

check_cpu_cores
check_memory_headroom
check_disk_headroom
check_gpu_hardware
check_gpu_topology
configure_build_parallelism

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    fail "CUDA_HOME is not set and nvcc was not found under /usr/local/cuda*."
  fi
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  fail "nvcc not found at $CUDA_HOME/bin/nvcc."
fi
validate_cuda_dev_files
choose_download_mode
confirm_install

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee -a "$LOG"
  export PATH="$HOME/.local/bin:$PATH"
fi

command -v uv >/dev/null 2>&1 || fail "uv install did not put uv on PATH."

export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu128}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export VLLM_CUTLASS_SRC_DIR=${VLLM_CUTLASS_SRC_DIR:-$CUTLASS_DIR}
export TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR:-"$TRITON_KERNELS_DIR/python/triton_kernels/triton_kernels"}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}
export NVCC_THREADS=${NVCC_THREADS:-1}

cat <<EOF | tee -a "$LOG"

Build settings:
  Fork release=$VERSION
  Base vLLM=$BASE_VLLM_VERSION
  Runtime identity=$RUNTIME_IDENTITY
  Validated CUDA=$VALIDATED_CUDA_VERSION
  Validated torch=$VALIDATED_TORCH_VERSION
  Reference NVIDIA driver=$VALIDATED_NVIDIA_DRIVER_VERSION
  FlashQLA repo=$FLASHQLA_REPO
  FlashQLA dir=$FLASHQLA_DIR
  CUDA_HOME=$CUDA_HOME
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
  UV_TORCH_BACKEND=$UV_TORCH_BACKEND
  CPU_THREADS=$CPU_THREADS
  MEMORY_GB=$MEMORY_GB
  MAX_JOBS=$MAX_JOBS ($MAX_JOBS_SOURCE)
  CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL
  NVCC_THREADS=$NVCC_THREADS
  CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
  VENV=$ROOT/.venv
  Python package mirror fallback=${BUILD_PYPI_MIRROR_FALLBACK:-1}
  Python package mirror index=${BUILD_PYPI_MIRROR_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
  Network preflight sample timeout=${BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS}s per URL probe
  Python package selected-route attempt timeout=${BUILD_PYPI_PRIMARY_TIMEOUT_SECONDS}s
  Python package wheelhouse=${BUILD_WHEELHOUSE_DIR:-auto:/data/wheelhouse/cu128}
  Git selected-route attempt timeout=${BUILD_GIT_PRIMARY_TIMEOUT_SECONDS}s
  Git mirror prefixes=${BUILD_GIT_MIRROR_PREFIXES:-none}
  CUTLASS repo=$CUTLASS_REPO
  CUTLASS ref=$CUTLASS_REVISION
  CUTLASS dir=$CUTLASS_DIR
  Triton repo=$TRITON_REPO
  Triton tag=$TRITON_TAG
  Triton dir=$TRITON_KERNELS_DIR
  Triton kernels dir=$TRITON_KERNELS_SRC_DIR
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! run_step "GPU summary" nvidia-smi; then
    echo "GPU summary warning: nvidia-smi returned a non-zero status." | tee -a "$LOG"
  fi
else
  echo "GPU summary skipped: nvidia-smi not found." | tee -a "$LOG"
fi
run_step "CUDA compiler" "$CUDA_HOME/bin/nvcc" --version

if [[ ! -d .venv ]]; then
  run_with_progress "Create Python virtualenv" uv venv --python "${PYTHON_VERSION:-3.11}" .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  fail ".venv/bin/python was not created."
fi

install_torch_from_wheelhouse

install_build_frontend_requirements
run_with_progress "Validate build frontend tools" ensure_build_frontend_tools

[[ -f requirements/build/cuda.txt ]] || fail "Missing requirements/build/cuda.txt. Re-sync the repository before building."
run_uv_pip_with_mirror_fallback "Install CUDA build requirements" install --python .venv/bin/python -r requirements/build/cuda.txt
run_with_progress "Re-validate build frontend tools" ensure_build_frontend_tools

[[ -f requirements/cuda.txt ]] || fail "Missing requirements/cuda.txt. Re-sync the repository before building."
run_uv_pip_with_mirror_fallback "Install CUDA runtime requirements" install --python .venv/bin/python -r requirements/cuda.txt
run_with_progress "Re-validate build frontend tools after runtime deps" ensure_build_frontend_tools

patch_flashqla_sm75_imports() {
  python - "$FLASHQLA_DIR" "$ROOT/tools/flashqla_sm75_patches" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
patch_root = Path(sys.argv[2])
patches = {
    root / "flash_qla" / "__init__.py": '''# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.0"

try:
    from flash_qla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_fwd,
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule,
    )
except ValueError:
    chunk_gated_delta_rule_fwd = None
    chunk_gated_delta_rule_bwd = None
    chunk_gated_delta_rule = None

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
]
''',
    root / "flash_qla" / "ops" / "__init__.py": '''# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

try:
    from .gated_delta_rule import chunk_gated_delta_rule
except ValueError:
    chunk_gated_delta_rule = None

__all__ = ["chunk_gated_delta_rule"]
''',
    root / "flash_qla" / "ops" / "gated_delta_rule" / "__init__.py": '''# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

try:
    from .chunk import chunk_gated_delta_rule
except ValueError:
    chunk_gated_delta_rule = None

__all__ = ["chunk_gated_delta_rule"]
''',
}

for path, content in patches.items():
    if not path.exists():
        raise SystemExit(f"missing FlashQLA file: {path}")
    path.write_text(content, encoding="utf-8")

legacy_py = root / "flash_qla" / "ops" / "gated_delta_rule" / "legacy" / "sm_legacy.py"
legacy_cu = legacy_py.with_name("csrc") / "gdn_forward.cu"
patch_py = patch_root / "sm_legacy.py"
patch_cu = patch_root / "gdn_forward.cu"
for src, dst in ((patch_py, legacy_py), (patch_cu, legacy_cu)):
    if not src.exists():
        raise SystemExit(f"missing FlashQLA SM75 patch file: {src}")
    if not dst.exists():
        raise SystemExit(f"missing FlashQLA target file: {dst}")
    shutil.copyfile(src, dst)
PY
}

build_flashqla_legacy_extension() {
  env \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_PATH="$CUDA_PATH" \
    CUDACXX="$CUDACXX" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$FLASHQLA_DIR/.torch_extensions_vllm_flashqla_legacy}" \
    .venv/bin/python - "$FLASHQLA_DIR" <<'PY'
from pathlib import Path
import importlib
import os
import sys

flashqla_root = Path(sys.argv[1]).resolve()
legacy_src = (
    flashqla_root
    / "flash_qla"
    / "ops"
    / "gated_delta_rule"
    / "legacy"
    / "csrc"
    / "gdn_forward.cu"
)
if not legacy_src.is_file():
    raise SystemExit(f"missing FlashQLA legacy source: {legacy_src}")

sys.path.insert(0, str(flashqla_root))
legacy = importlib.import_module("flash_qla.ops.gated_delta_rule.legacy.sm_legacy")
ext = legacy._load_ext()
so_path = Path(getattr(ext, "__file__", ""))
if not so_path.is_file():
    raise SystemExit(f"FlashQLA legacy extension did not produce a shared object: {so_path}")

print(f"flash_qla_legacy_extension={so_path}")
print(f"torch_extensions_dir={os.environ.get('TORCH_EXTENSIONS_DIR', '')}")
PY
}

validate_runtime_components() {
  env \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_PATH="$CUDA_PATH" \
    CUDACXX="$CUDACXX" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$FLASHQLA_DIR/.torch_extensions_vllm_flashqla_legacy}" \
    VLLM_CUTLASS_SRC_DIR="$VLLM_CUTLASS_SRC_DIR" \
    TRITON_KERNELS_SRC_DIR="$TRITON_KERNELS_SRC_DIR" \
    FLASHQLA_DIR="$FLASHQLA_DIR" \
    .venv/bin/python "$ROOT/tools/validate_runtime_components.py"
}

run_with_progress "Fetch FlashQLA SM70/SM75 backend" fetch_flashqla_with_fallback
run_with_progress "Patch FlashQLA SM75 legacy imports" patch_flashqla_sm75_imports
run_uv_pip_with_mirror_fallback "Install FlashQLA SM70/SM75 backend" install --python .venv/bin/python --no-deps -e "$FLASHQLA_DIR"
run_with_progress "Build FlashQLA legacy GDN extension" build_flashqla_legacy_extension
run_with_progress "Fetch CUTLASS source" fetch_cutlass_with_fallback
run_with_progress "Fetch Triton kernels source" fetch_triton_with_fallback

run_with_progress "Build and install vLLM 2080 Ti Definitive runtime" \
  env \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_PATH="$CUDA_PATH" \
    CUDACXX="$CUDACXX" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    FLASHINFER_ENABLE_AOT="$FLASHINFER_ENABLE_AOT" \
    VLLM_CUTLASS_SRC_DIR="$VLLM_CUTLASS_SRC_DIR" \
    TRITON_KERNELS_SRC_DIR="$TRITON_KERNELS_SRC_DIR" \
    VLLM_VERSION_OVERRIDE="$VERSION" \
    uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .

run_with_progress "Validate runtime components" validate_runtime_components

echo
echo "BUILD OK"
echo "Total elapsed: $(format_seconds "$(( $(date +%s) - BUILD_STARTED_AT ))")"
echo "Log: $LOG"
echo "Next step:"
echo "  ./launcher.sh"
echo
