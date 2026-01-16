#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=computelong
#SBATCH --job-name=compute_spec_stats
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

# change to src/compute_statistics_of_spectrograms.py if you want to use the original script
python -u src/compute_statistics_of_spectrograms.py --spec_dir "$SPEC_DIR"

echo "Done. Stats under: $SPEC_DIR/audio_params.json"
