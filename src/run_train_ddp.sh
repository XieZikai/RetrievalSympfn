#!/bin/bash
#SBATCH --job-name=sympfn_ddp         # 任务名
#SBATCH --nodes=1                     # 使用的节点数 (单机多卡设为 1)
#SBATCH --ntasks-per-node=1           # 每个节点启动一个主进程
#SBATCH --gres=gpu:4                  # 申请的 GPU 数量 (假设这里用 4 张卡)
#SBATCH --cpus-per-task=16            # CPU 核心数 (推荐 = GPU数 * 4)
#SBATCH --partition=a800
#SBATCH --output=/fs0/home/zikaix/Data/zikaix/logs/slurm-sympfn-ddp-%j.out         # 标准输出日志
#SBATCH --error=/fs0/home/zikaix/Data/zikaix/logs/slurm-sympfn-ddp-%j.err          # 错误输出日志
#SBATCH --time=2:00:00               # 限制运行时间

# 1. 激活你的虚拟环境 (替换为你实际的 conda 环境或 venv)
source /fs0/home/zikaix/miniconda3/bin/activate tabpfn_env

# 2. 获取当前分配到的 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果 SLURM 没有设置这个变量，回退读取申请的 gpu 数量
    NUM_GPUS=$SLURM_GPUS_ON_NODE
fi
echo "Starting DDP on $NUM_GPUS GPUs..."

# 3. 使用 torchrun 启动 DDP 训练
# nproc_per_node 必须等于你申请的 GPU 数量
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_ddp.py