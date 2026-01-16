#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=computelong
#SBATCH --job-name=test_train_split
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/home/georgev/TinyBird/logs/pt_conversion-%j.out
#SBATCH --error=/home/georgev/TinyBird/logs/pt_conversion-%j.err

set -euo pipefail

# --- modules & env
module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate /home/georgev/.conda/envs/tinybird
conda install --file "$HOME/TinyBird/requirements.txt" -y

# --- scratch on Talapas is /scratch/<PIRG>/<USER>
PIRG=${PIRG:-gardnerlab}
MY_SCRATCH=/scratch/${PIRG}/${USER}
SPEC_DIR="$MY_SCRATCH/XCM_Bambird"
ANNOTATION_JSON="$SPEC_DIR/XCM_train_annotations.json"
TRAIN_DIR="$MY_SCRATCH/XCM_Train"
TEST_DIR="$MY_SCRATCH/XCM_Test"
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

# Train/test split (deterministic if --seed is set)
python -u scripts/split_train_test.py \
  --mode split \
  --spec_dir "$SPEC_DIR" \
  --train_dir "$TRAIN_DIR" \
  --test_dir "$TEST_DIR" \
  --annotation_json "$ANNOTATION_JSON" \
  --train_percent 95 \
  --ignore_bird_id \
  --seed 42

echo "Done. Train: $TRAIN_DIR | Test: $TEST_DIR"
