#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=gpu
#SBATCH --job-name=tinybird-bench
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/home/georgev/TinyBird/logs/tinybird-bench-%j.out
#SBATCH --error=/home/georgev/TinyBird/logs/tinybird-bench-%j.err

set -euo pipefail

# --- modules & env
module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate /home/georgev/.conda/envs/tinybird
pip install -r "$HOME/TinyBird/requirements.txt"

PIRG=${PIRG:-gardnerlab}
MY_SCRATCH=/scratch/${PIRG}/${USER}

# Ensure directories exist
mkdir -p "$MY_SCRATCH/benchmark_results"

# --- Keep libraries from oversubscribing threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

cd "$HOME/TinyBird"
bash shell/classify_detect_bench.sh \
    --spec_root "$MY_SCRATCH/SongMAE_Bench_Data" \
    --results_dir "$MY_SCRATCH/benchmark_results" \
    --num_workers "$SLURM_CPUS_PER_TASK" \
    --batch_size 32 \
    --probe_mode "finetune" \
    --task_mode "classify"

echo "Benchmark job completed. Results in: $MY_SCRATCH/benchmark_results"
