#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage: run_simple_gemm_batch_sweep.sh -e DTD_SIMPLE_GEMM [options]

Run the DTD simple_gemm test on one node while sweeping CUDA batch mode, GPU
count, matrix size, and tile size. The output file is a structured log that can
be consumed by plot_simple_gemm_batch_sweep.py.

Required:
  -e, --exe PATH             Path to the dtd_test_simple_gemm executable.

Options:
  -o, --output PATH          Output log file (default: simple_gemm_sweep.log).
  -g, --gpus LIST            GPU counts to expose, comma or space separated
                             (default: 1). Example: 1,2,4
  -s, --sizes LIST           Cubic matrix sizes M=N=K (default: 4096).
  -t, --tiles LIST           Cubic tile sizes mb=nb=kb (default: 256).
  -m, --modes LIST           Batch modes (default: none,one-by-one,cublas).
  -r, --runs N               Number of measured runs passed to simple_gemm
                             (default: 5; simple_gemm also does one warmup).
      --batch-size N         Maximum number of GEMM tasks collected per batch.
      --batch-slots N        Maximum in-flight cuBLAS-batched submissions per
                             CUDA stream.
      --mpirun PATH          Optional mpirun/mpiexec path.
      --np N                 MPI ranks when --mpirun is used (default: 1).
      --timeout DURATION     Timeout for each individual run, using the
                             timeout command's duration syntax. Example: 10m
      --timeout-cmd PATH     Timeout command to use when --timeout is set
                             (default: timeout).
      --gpu-offset N         First CUDA device id to expose (default: 0).
      --extra-gemm-args STR  Extra arguments before simple_gemm's "--".
      --extra-parsec-args STR
                             Extra PaRSEC MCA arguments after "--".
      --append               Append to the output file instead of replacing it.
      --allow-non-divisible  Run matrix/tile pairs even when size % tile != 0.
      --stop-on-error        Stop the sweep at the first failed run.
  -h, --help                 Show this help.

Environment variables with the same upper-case names can be used for defaults,
for example GPU_COUNTS="1,2,4" MATRIX_SIZES="8192 16384".
EOF
}

split_list() {
    local value="$1"
    value="${value//,/ }"
    # shellcheck disable=SC2086
    printf '%s\n' $value
}

gpu_visible_list() {
    local count="$1"
    local offset="$2"
    local visible=""

    if [[ "${count}" == "all" ]]; then
        printf 'all\n'
        return 0
    fi

    for ((i = 0; i < count; i++)); do
        if [[ -n "${visible}" ]]; then
            visible+=","
        fi
        visible+="$((offset + i))"
    done
    printf '%s\n' "${visible}"
}

quote_command() {
    local item
    for item in "$@"; do
        printf '%q ' "${item}"
    done
}

EXE="${DTD_SIMPLE_GEMM:-}"
OUTPUT="${OUTPUT:-simple_gemm_sweep.log}"
GPU_COUNTS="${GPU_COUNTS:-1}"
MATRIX_SIZES="${MATRIX_SIZES:-4096}"
TILE_SIZES="${TILE_SIZES:-256}"
BATCH_MODES="${BATCH_MODES:-none,one-by-one,cublas}"
RUNS="${RUNS:-5}"
BATCH_SIZE="${BATCH_SIZE:-}"
BATCH_SLOTS="${BATCH_SLOTS:-}"
MPIRUN="${MPIRUN:-}"
NP="${NP:-1}"
RUN_TIMEOUT="${RUN_TIMEOUT:-}"
TIMEOUT_CMD="${TIMEOUT_CMD:-timeout}"
GPU_OFFSET="${GPU_OFFSET:-0}"
EXTRA_GEMM_ARGS="${EXTRA_GEMM_ARGS:-}"
EXTRA_PARSEC_ARGS="${EXTRA_PARSEC_ARGS:-}"
APPEND=0
ALLOW_NON_DIVISIBLE=0
STOP_ON_ERROR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--exe)
            EXE="$2"; shift 2 ;;
        -o|--output)
            OUTPUT="$2"; shift 2 ;;
        -g|--gpus)
            GPU_COUNTS="$2"; shift 2 ;;
        -s|--sizes|--matrix-sizes)
            MATRIX_SIZES="$2"; shift 2 ;;
        -t|--tiles|--tile-sizes)
            TILE_SIZES="$2"; shift 2 ;;
        -m|--modes)
            BATCH_MODES="$2"; shift 2 ;;
        -r|--runs)
            RUNS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --batch-slots)
            BATCH_SLOTS="$2"; shift 2 ;;
        --mpirun)
            MPIRUN="$2"; shift 2 ;;
        --np)
            NP="$2"; shift 2 ;;
        --timeout)
            RUN_TIMEOUT="$2"; shift 2 ;;
        --timeout-cmd)
            TIMEOUT_CMD="$2"; shift 2 ;;
        --gpu-offset)
            GPU_OFFSET="$2"; shift 2 ;;
        --extra-gemm-args)
            EXTRA_GEMM_ARGS="$2"; shift 2 ;;
        --extra-parsec-args)
            EXTRA_PARSEC_ARGS="$2"; shift 2 ;;
        --append)
            APPEND=1; shift ;;
        --allow-non-divisible)
            ALLOW_NON_DIVISIBLE=1; shift ;;
        --stop-on-error)
            STOP_ON_ERROR=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if [[ -z "${EXE}" ]]; then
    echo "Missing required --exe PATH" >&2
    usage >&2
    exit 2
fi

if [[ ! -x "${EXE}" ]]; then
    echo "Executable not found or not executable: ${EXE}" >&2
    exit 2
fi

if [[ -n "${RUN_TIMEOUT}" ]] && ! command -v "${TIMEOUT_CMD}" >/dev/null 2>&1; then
    echo "Timeout command not found: ${TIMEOUT_CMD}" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"

if [[ "${APPEND}" -eq 0 ]]; then
    {
        printf '# parsec simple_gemm batch sweep log v1\n'
        printf '# created_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf '# executable=%s\n' "${EXE}"
        printf '# gpu_counts=%s\n' "${GPU_COUNTS}"
        printf '# matrix_sizes=%s\n' "${MATRIX_SIZES}"
        printf '# tile_sizes=%s\n' "${TILE_SIZES}"
        printf '# batch_modes=%s\n' "${BATCH_MODES}"
        printf '# runs=%s\n' "${RUNS}"
        printf '# timeout=%s\n' "${RUN_TIMEOUT:-none}"
        printf '# timeout_cmd=%s\n' "${TIMEOUT_CMD}"
        printf '# batch_size=%s\n' "${BATCH_SIZE:-default}"
        printf '# batch_slots=%s\n' "${BATCH_SLOTS:-default}"
    } > "${OUTPUT}"
fi

read -r -a extra_gemm_args <<< "${EXTRA_GEMM_ARGS}"
read -r -a extra_parsec_args <<< "${EXTRA_PARSEC_ARGS}"

run_id=0
failed=0
for mode in $(split_list "${BATCH_MODES}"); do
    for gpus in $(split_list "${GPU_COUNTS}"); do
        visible="$(gpu_visible_list "${gpus}" "${GPU_OFFSET}")"
        for matrix in $(split_list "${MATRIX_SIZES}"); do
            for tile in $(split_list "${TILE_SIZES}"); do
                if [[ "${ALLOW_NON_DIVISIBLE}" -eq 0 && "$((matrix % tile))" -ne 0 ]]; then
                    echo "Skipping M=N=K=${matrix}, tile=${tile}: matrix size is not divisible by tile size" >&2
                    continue
                fi

                run_id=$((run_id + 1))
                cmd=("${EXE}"
                     --device GPU
                     --M "${matrix}" --N "${matrix}" --K "${matrix}"
                     --mb "${tile}" --nb "${tile}" --kb "${tile}"
                     --nruns "${RUNS}"
                     --batch-mode "${mode}")
                if [[ -n "${BATCH_SIZE}" ]]; then
                    cmd+=(--batch-size "${BATCH_SIZE}")
                fi
                if [[ -n "${BATCH_SLOTS}" ]]; then
                    cmd+=(--batch-slots "${BATCH_SLOTS}")
                fi
                if [[ "${#extra_gemm_args[@]}" -gt 0 && -n "${extra_gemm_args[0]}" ]]; then
                    cmd+=("${extra_gemm_args[@]}")
                fi
                if [[ "${#extra_parsec_args[@]}" -gt 0 && -n "${extra_parsec_args[0]}" ]]; then
                    cmd+=(-- "${extra_parsec_args[@]}")
                fi

                full_cmd=()
                if [[ -n "${MPIRUN}" ]]; then
                    full_cmd+=("${MPIRUN}" -np "${NP}")
                fi
                if [[ "${visible}" == "all" ]]; then
                    full_cmd+=("${cmd[@]}")
                else
                    full_cmd+=(env "CUDA_VISIBLE_DEVICES=${visible}" "${cmd[@]}")
                fi
                if [[ -n "${RUN_TIMEOUT}" ]]; then
                    full_cmd=("${TIMEOUT_CMD}" "${RUN_TIMEOUT}" "${full_cmd[@]}")
                fi

                echo "Running mode=${mode} gpus=${gpus} matrix=${matrix} tile=${tile}" >&2
                {
                    printf 'PARSEC_SIMPLE_GEMM_RUN_BEGIN run_id=%d mode=%s gpus=%s cuda_visible_devices=%s matrix=%s tile=%s runs=%s timeout=%s utc=%s\n' \
                           "${run_id}" "${mode}" "${gpus}" "${visible}" "${matrix}" "${tile}" "${RUNS}" "${RUN_TIMEOUT:-none}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
                    printf 'PARSEC_SIMPLE_GEMM_COMMAND run_id=%d ' "${run_id}"
                    quote_command "${full_cmd[@]}"
                    printf '\n'
                } >> "${OUTPUT}"

                if "${full_cmd[@]}" >> "${OUTPUT}" 2>&1; then
                    rc=0
                else
                    rc=$?
                    failed=1
                fi

                printf 'PARSEC_SIMPLE_GEMM_RUN_END run_id=%d rc=%d utc=%s\n' \
                       "${run_id}" "${rc}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${OUTPUT}"

                if [[ "${rc}" -ne 0 ]]; then
                    echo "Run ${run_id} failed with rc=${rc}" >&2
                    if [[ "${STOP_ON_ERROR}" -ne 0 ]]; then
                        exit "${rc}"
                    fi
                fi
            done
        done
    done
done

exit "${failed}"
