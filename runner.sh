#!/bin/sh
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:titanxp:1
#SBATCH -c 16
#SBATCH --mem=60000
#SBATCH -o /datasets/datasets_kleina/slurm_output/logs/%j_%x.txt

tmp_dir=/ssd/${SLURM_JOB_ID}
mkdir ${tmp_dir}
datadir="/datasets/datasets_kleina/bone_seg_ijcars_compressed"
output_dir="/datasets/datasets_kleina/slurm_output"
cp -r ${datadir}/* ${tmp_dir}

CUDA_CACHE_PATH=/ssd/${SLURM_JOB_ID}/nvcache
mkdir -p ${CUDA_CACHE_PATH}
export CUDA_CACHE_PATH

python train.py --base_dir ${output_dir} --data_dir ${tmp_dir}/preprocessed_compressed --data_test_dir ${tmp_dir}/preprocessed_compressed --split_dir ${tmp_dir} --do_batchnorm=True --do_ce_weighting=False --batch_size=8 --patch_size=512 --additional_slices=0 --n_epochs=50 --name=${SLURM_JOB_NAME}

rm -r ${tmp_dir}

# You could pass the job-name to your python script by doing the following:
# python MyPythonScript.py $SLURM_JOB_NAME
# titanxp -> long
# titanx -> short