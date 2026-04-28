#!/bin/bash
#SBATCH --job-name=profile_ssm_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=logs/profile_metrics_%j.out
#SBATCH --error=logs/profile_metrics_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

module purge
module load gcc/13.3.0
module load cuda/12.6.3

D_LIST=(1 16 64 256 512)
L_LIST=(1024 2048 4096 8192 16384 32768 65536 131072)
KERNELS=(warp_shuffle blelloch hillis_steele)

INPUT_DIR="${PROJECT_ROOT}/SyntheticData/inputs"
REF_DIR="${PROJECT_ROOT}/SequentialBaseline/SequentialData"
OUT_ROOT="${PROJECT_ROOT}/Results/profile_run_${SLURM_JOB_ID}"
RAW_DIR="${OUT_ROOT}/raw_ncu"
BIN_DIR="${OUT_ROOT}/bin"
TIMING_CSV="${OUT_ROOT}/timing.csv"

mkdir -p "${OUT_ROOT}" "${RAW_DIR}" "${BIN_DIR}"
rm -f "${TIMING_CSV}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

if [[ -z "${PEAK_BW_GBS:-}" ]]; then
    if [[ "${GPU_NAME}" == *"SXM4"* ]]; then
        PEAK_BW_GBS=2039
    elif [[ "${GPU_NAME}" == *"80GB"* ]]; then
        PEAK_BW_GBS=1935
    else
        PEAK_BW_GBS=1555
    fi
fi

if [[ -z "${PEAK_FP32_GFLOPS:-}" ]]; then
    PEAK_FP32_GFLOPS=19500
fi

echo "Node:  $(hostname)"
echo "GPU:   ${GPU_NAME}"
echo "Start: $(date)"
echo "OUT:   ${OUT_ROOT}"
echo "Peak BW (GB/s): ${PEAK_BW_GBS}"
echo "Peak FP32 (GFLOP/s): ${PEAK_FP32_GFLOPS}"
echo

echo "[Stage 1/3] Generate synthetic inputs"
GEN_BIN="${BIN_DIR}/generate_inputs"
g++ -O2 -std=c++17 -o "${GEN_BIN}" "${PROJECT_ROOT}/SyntheticData/generate_inputs.cpp"
mkdir -p "${INPUT_DIR}"
"${GEN_BIN}" "${INPUT_DIR}" | tee "${OUT_ROOT}/generate_inputs.log"

echo
echo "[Stage 2/3] Run sequential baseline and write references"
REF_BIN="${BIN_DIR}/run_reference"
g++ -O2 -std=c++17 -o "${REF_BIN}" "${PROJECT_ROOT}/SequentialBaseline/run_reference.cpp"
mkdir -p "${REF_DIR}"
"${REF_BIN}" "${INPUT_DIR}" "${REF_DIR}" | tee "${OUT_ROOT}/run_reference.log"

echo
echo "[Stage 3/3] Profile GPU kernels"

AVAIL_METRICS="$(ncu --query-metrics --chip sm_80 --csv | awk -F, 'NR>1 {gsub(/"/,"",$1); print $1}')"

pick_metric() {
    for candidate in "$@"; do
        if grep -Fxq "${candidate}" <<< "${AVAIL_METRICS}"; then
            echo "${candidate}"
            return 0
        fi
    done
    echo ""
    return 1
}

M_GPU_TIME="$(pick_metric \
    gpu__time_duration.sum \
    gpu__time_duration.avg)"

M_SM_UTIL="$(pick_metric \
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
    sm__throughput.avg.pct_of_peak_sustained_active)"

M_DRAM_UTIL="$(pick_metric \
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    dram__throughput.avg.pct_of_peak_sustained_active)"

M_OCC="$(pick_metric \
    sm__warps_active.avg.pct_of_peak_sustained_active \
    sm__warps_active.avg.pct_of_peak_sustained_elapsed)"

M_WARP_RATIO="$(pick_metric \
    smsp__thread_inst_executed_per_inst_executed.ratio)"

M_DRAM_BYTES="$(pick_metric \
    dram__bytes.sum)"

M_DRAM_BYTES_READ="$(pick_metric \
    dram__bytes_read.sum)"

M_DRAM_BYTES_WRITE="$(pick_metric \
    dram__bytes_write.sum)"

M_FADD="$(pick_metric \
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum)"

M_FMUL="$(pick_metric \
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum)"

M_FFMA="$(pick_metric \
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum)"

M_REGS="$(pick_metric \
    launch__registers_per_thread)"

M_SMEM_BLOCK="$(pick_metric \
    launch__shared_mem_per_block_allocated \
    launch__shared_mem_per_block_static)"

M_BLOCK_SIZE="$(pick_metric \
    launch__block_size)"

M_L1_HIT="$(pick_metric \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum)"

M_L1_MISS="$(pick_metric \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum)"

M_L2_HIT="$(pick_metric \
    lts__t_sectors_srcunit_lsu_op_read_lookup_hit.sum)"

M_L2_MISS="$(pick_metric \
    lts__t_sectors_srcunit_lsu_op_read_lookup_miss.sum)"

METRICS=()
for m in \
    "${M_GPU_TIME}" \
    "${M_SM_UTIL}" "${M_DRAM_UTIL}" "${M_OCC}" "${M_WARP_RATIO}" \
    "${M_DRAM_BYTES}" "${M_DRAM_BYTES_READ}" "${M_DRAM_BYTES_WRITE}" \
    "${M_FADD}" "${M_FMUL}" "${M_FFMA}" \
    "${M_REGS}" "${M_SMEM_BLOCK}" "${M_BLOCK_SIZE}" \
    "${M_L1_HIT}" "${M_L1_MISS}" "${M_L2_HIT}" "${M_L2_MISS}"; do
    if [[ -n "${m}" ]]; then
        METRICS+=("${m}")
    fi
done

if [[ ${#METRICS[@]} -eq 0 ]]; then
    echo "No Nsight metrics selected."
    exit 1
fi

METRIC_STR="$(IFS=,; echo "${METRICS[*]}")"

echo "Selected metrics:"
for metric in "${METRICS[@]}"; do
    echo "  - ${metric}"
done
echo

TOTAL=$(( ${#D_LIST[@]} * ${#L_LIST[@]} * ${#KERNELS[@]} ))
COUNT=0

BIN="${BIN_DIR}/profile_driver"
echo "Compiling profile driver (runtime D)"
nvcc -O3 -std=c++17 -arch=sm_80 --maxrregcount=64 \
    -o "${BIN}" "${SCRIPT_DIR}/profile_driver.cu"

for D in "${D_LIST[@]}"; do
    for KERNEL in "${KERNELS[@]}"; do
        for L in "${L_LIST[@]}"; do
            COUNT=$((COUNT + 1))
            TAG="${KERNEL}_D${D}_L${L}"

            INPUT_FILE="${INPUT_DIR}/input_B1_L${L}_D${D}.bin"
            REF_FILE="${REF_DIR}/ref_B1_L${L}_D${D}.bin"
            if [[ ! -f "${INPUT_FILE}" || ! -f "${REF_FILE}" ]]; then
                echo "[${COUNT}/${TOTAL}] Skipping ${TAG} (missing input/ref file)"
                continue
            fi

            echo "[${COUNT}/${TOTAL}] Timing ${TAG}"
            "${BIN}" \
                --kernel "${KERNEL}" \
                --D "${D}" \
                --L "${L}" \
                --input_dir "${INPUT_DIR}" \
                --ref_dir "${REF_DIR}" \
                --warmup 3 \
                --repeat 10 \
                --csv_append "${TIMING_CSV}"

            echo "[${COUNT}/${TOTAL}] Profiling ${TAG} with Nsight Compute"
            ncu \
                --target-processes all \
                --clock-control base \
                --cache-control all \
                --replay-mode kernel \
                --csv \
                --page raw \
                --metrics "${METRIC_STR}" \
                --force-overwrite \
                --log-file "${RAW_DIR}/${TAG}.csv" \
                "${BIN}" \
                    --kernel "${KERNEL}" \
                    --D "${D}" \
                    --L "${L}" \
                    --input_dir "${INPUT_DIR}" \
                    --ref_dir "${REF_DIR}" \
                    --warmup 0 \
                    --repeat 1 \
                    --skip_check \
                    --no_print
        done
    done
done

echo
echo "Running analysis..."
python3 "${SCRIPT_DIR}/analyze_metrics.py" \
    --timing_csv "${TIMING_CSV}" \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${OUT_ROOT}" \
    --peak_bw_gbs "${PEAK_BW_GBS}" \
    --peak_fp32_gflops "${PEAK_FP32_GFLOPS}" | tee "${OUT_ROOT}/analysis_stdout.txt"

echo
echo "Finished: $(date)"
echo "All outputs saved in: ${OUT_ROOT}"
