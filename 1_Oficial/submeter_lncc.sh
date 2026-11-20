#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -p sequana_cpu
#SBATCH -J nextflow_gerente
#SBATCH -o log_python_saida.txt
#SBATCH -e log_python_erro.txt

module purge
module load python/3.7.3_sequana
cd /scratch/rtm-uq/alexandre.lima/Breno
eval "$(conda shell.bash hook)"
conda activate /scratch/rtm-uq/alexandre.lima/Breno/conda_envs/tcc

nextflow run workflow.nf -params-file config.yaml -resume
