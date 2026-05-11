#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build"
NP=2
NT=64
MAX_BYTES=1024
DO_BUILD=1
LAUNCHER="auto"
REQUIRE_GPUS=0

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --build-dir <dir>   Build directory (default: build)"
  echo "  --np <ranks>        MPI ranks for distributed runs (default: 2)"
  echo "  --nt <tasks>        Number of task instances (default: 64)"
  echo "  --max-bytes <size>  Max message size in bytes (default: 1024)"
  echo "  --launcher <name>   Distributed launcher: auto|mpirun|srun (default: auto)"
  echo "  --require-gpus <N>  Fail unless nvidia-smi sees exactly N GPUs"
  echo "  --no-build          Skip build step"
  echo "  -h, --help          Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --np)
      NP="$2"
      shift 2
      ;;
    --nt)
      NT="$2"
      shift 2
      ;;
    --max-bytes)
      MAX_BYTES="$2"
      shift 2
      ;;
    --launcher)
      LAUNCHER="$2"
      shift 2
      ;;
    --require-gpus)
      REQUIRE_GPUS="$2"
      shift 2
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

EXE_AUTO="./${BUILD_DIR}/tests/dsl/ptg/receive_auto"
EXE_MIXED="./${BUILD_DIR}/tests/dsl/ptg/receive_auto_mixed"
GPU_MCA_ARGS=(-- --mca runtime_comm_auto_gpu_enable 1 --mca device_load_balance_allow_cpu 0 --mca runtime_comm_short_limit 0)
DIST_CMD=()
DIST_GPU_MCA_ARGS=()

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

extract_gpu_hits() {
  local output="$1"
  local hits
  hits="$(echo "${output}" | sed -n 's/.*gpu_hits=\([0-9][0-9]*\).*/\1/p' | tail -n1)"
  if [[ -z "${hits}" ]]; then
    echo "Could not parse gpu_hits from output:" >&2
    echo "${output}" >&2
    return 1
  fi
  echo "${hits}"
}

count_visible_nvidia_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found in PATH" >&2
    return 1
  fi
  nvidia-smi -L | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]'
}

run_dist_cmd() {
  "${DIST_CMD[@]}" "$@"
}

if [[ "${LAUNCHER}" == "auto" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    LAUNCHER="srun"
  else
    LAUNCHER="mpirun"
  fi
fi

case "${LAUNCHER}" in
  mpirun)
    DIST_CMD=(env OMPI_MCA_psec=native PMIX_MCA_psec=native mpirun -np "${NP}")
    DIST_GPU_MCA_ARGS=()
    ;;
  srun)
    DIST_CMD=(env OMPI_MCA_psec=native PMIX_MCA_psec=native srun -n "${NP}" --gpus-per-task=1)
    # One CUDA device per rank prevents each rank from reserving all GPUs.
    DIST_GPU_MCA_ARGS=(--mca device_cuda_enabled 1)
    ;;
  *)
    echo "Invalid launcher '${LAUNCHER}'. Use auto|mpirun|srun." >&2
    exit 1
    ;;
esac

if [[ "${REQUIRE_GPUS}" -gt 0 ]]; then
  VISIBLE_GPUS="$(count_visible_nvidia_gpus)"
  if [[ "${VISIBLE_GPUS}" -ne "${REQUIRE_GPUS}" ]]; then
    echo "GPU precheck failed: expected ${REQUIRE_GPUS}, found ${VISIBLE_GPUS}" >&2
    exit 1
  fi
fi

if [[ ${DO_BUILD} -eq 1 ]]; then
  run_cmd cmake --build "${BUILD_DIR}" --target receive_auto receive_auto_mixed -j8
fi

if [[ ! -x "${EXE_AUTO}" || ! -x "${EXE_MIXED}" ]]; then
  echo "Missing test binaries. Expected:" >&2
  echo "  ${EXE_AUTO}" >&2
  echo "  ${EXE_MIXED}" >&2
  echo "Run without --no-build or check --build-dir." >&2
  exit 1
fi

echo
echo "== CPU local runs =="
run_cmd "${EXE_AUTO}" "-n=${NT}" "-b=${MAX_BYTES}"
run_cmd "${EXE_MIXED}" "-n=${NT}" "-b=${MAX_BYTES}"

echo
echo "== CPU distributed runs (${LAUNCHER} ranks=${NP}) =="
run_cmd run_dist_cmd "${EXE_AUTO}" "-n=${NT}" "-b=${MAX_BYTES}"
run_cmd run_dist_cmd "${EXE_MIXED}" "-n=${NT}" "-b=${MAX_BYTES}"

echo
echo "== GPU-forced local runs =="
gpu_out_auto="$("${EXE_AUTO}" "-n=${NT}" "-b=${MAX_BYTES}" "${GPU_MCA_ARGS[@]}")"
echo "${gpu_out_auto}"
gpu_hits_auto="$(extract_gpu_hits "${gpu_out_auto}")"
if [[ "${gpu_hits_auto}" -le 0 ]]; then
  echo "GPU validation failed for receive_auto: gpu_hits=${gpu_hits_auto}" >&2
  exit 1
fi

gpu_out_mixed="$("${EXE_MIXED}" "-n=${NT}" "-b=${MAX_BYTES}" "${GPU_MCA_ARGS[@]}")"
echo "${gpu_out_mixed}"
gpu_hits_mixed="$(extract_gpu_hits "${gpu_out_mixed}")"
if [[ "${gpu_hits_mixed}" -le 0 ]]; then
  echo "GPU validation failed for receive_auto_mixed: gpu_hits=${gpu_hits_mixed}" >&2
  exit 1
fi

echo
echo "== GPU-forced distributed runs (${LAUNCHER} ranks=${NP}) =="
gpu_mpi_out_auto="$(run_dist_cmd "${EXE_AUTO}" "-n=${NT}" "-b=${MAX_BYTES}" "${GPU_MCA_ARGS[@]}" "${DIST_GPU_MCA_ARGS[@]}")"
echo "${gpu_mpi_out_auto}"
gpu_mpi_hits_auto="$(extract_gpu_hits "${gpu_mpi_out_auto}")"
if [[ "${gpu_mpi_hits_auto}" -le 0 ]]; then
  echo "GPU MPI validation failed for receive_auto: gpu_hits=${gpu_mpi_hits_auto}" >&2
  exit 1
fi

gpu_mpi_out_mixed="$(run_dist_cmd "${EXE_MIXED}" "-n=${NT}" "-b=${MAX_BYTES}" "${GPU_MCA_ARGS[@]}" "${DIST_GPU_MCA_ARGS[@]}")"
echo "${gpu_mpi_out_mixed}"
gpu_mpi_hits_mixed="$(extract_gpu_hits "${gpu_mpi_out_mixed}")"
if [[ "${gpu_mpi_hits_mixed}" -le 0 ]]; then
  echo "GPU MPI validation failed for receive_auto_mixed: gpu_hits=${gpu_mpi_hits_mixed}" >&2
  exit 1
fi

echo
echo "All receive_auto tests passed."
