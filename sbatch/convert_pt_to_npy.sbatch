#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=computelong
#SBATCH --job-name=convert_pt
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/home/georgev/TinyBird/logs/convert_pt-%j.out
#SBATCH --error=/home/georgev/TinyBird/logs/convert_pt-%j.err

set -euo pipefail

# --- modules & env
module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate /home/georgev/.conda/envs/tinybird
pip install -r "$HOME/TinyBird/requirements.txt"

# --- scratch on Talapas is /scratch/<PIRG>/<USER>
PIRG=${PIRG:-gardnerlab}
MY_SCRATCH=/scratch/${PIRG}/${USER}
SPEC_DIR="$MY_SCRATCH/specs"
chmod -R g+rwX "$MY_SCRATCH"

# --- keep libraries from oversubscribing threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

# 75% of allocated CPUs for workers (>=1)
workers=$(( SLURM_CPUS_PER_TASK * 75 / 100 ))
if [ "$workers" -lt 1 ]; then workers=1; fi

cd "$HOME/TinyBird"

# Create destination directory for converted NPY files in scratch
dst="$MY_SCRATCH/specs_XCM"
mkdir -p "$dst"

# Convert PT files to NPY format
python -u scripts/convert_pt_to_npy.py \
  --src_dir "$SPEC_DIR" \
  --dst_dir "$dst"

echo "Done. NPYs under: $dst"