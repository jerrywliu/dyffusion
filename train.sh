#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=m4633
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15

source activate dyffusion

# cmd="WAND__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py experiment=navier_stokes_interpolation"
# cmd="WAND__SERVICE_WAIT=300 python run.py experiment=navier_stokes_dyffusion diffusion.interpolator_run_id=dq31fual"

# NSTK: interpolation and dyffusion
cmd="WAND__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py experiment=nstk_interpolation"
# cmd="WAND__SERVICE_WAIT=300 python run.py experiment=nstk_dyffusion"

set -x
srun -l \
    bash -c "
    $cmd
    "
