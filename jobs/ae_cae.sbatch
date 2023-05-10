#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=ae_cae.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --mail-type=END

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles

module load cuda/11.3
module load miniconda/3.9

conda activate /tudelft.net/staff-umbrella/StudentsCVlab/smukherjee/envs/phd_assignment

export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=TRUE

cd ../src

srun python train.py --exp_config ../scripts/ae_cae.yaml