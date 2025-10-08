#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=computelong
#SBATCH --job-name=tinybird-xcl
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/home/georgev/TinyBird/logs/tinybird-xcl-%j.out
#SBATCH --error=/home/georgev/TinyBird/logs/tinybird-xcl-%j.err

set -euo pipefail

# --- modules & env
module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate /home/georgev/.conda/envs/tinybird

# --- scratch on Talapas is /scratch/<PIRG>/<USER>, not /gpfs/scratch/<USER>
PIRG=${PIRG:-gardnerlab}
MY_SCRATCH=/scratch/${PIRG}/${USER}
mkdir -p "$MY_SCRATCH/specs"
chmod -R g+rwX "$MY_SCRATCH"

# --- Hugging Face cache: rely on defaults to keep things simple

# --- keep libraries from oversubscribing threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

# 75% of allocated CPUs for workers (>=1)
workers=$(( SLURM_CPUS_PER_TASK * 75 / 100 ))
if [ "$workers" -lt 1 ]; then workers=1; fi

cd "$HOME/TinyBird"

# Run all splits; change list if XCL only has some splits
python -u src/audio2spec.py \
--birdset "XCM" \
--birdset_split "train" \
--dst_dir "$MY_SCRATCH/specs/" \
--format pt \
--single_threaded false \
--max_workers "$workers"

echo "Done. Specs under: $MY_SCRATCH/specs"
