#!/bin/bash
#SBATCH --job-name=run_train1    # 作业名，方便识别
#SBATCH --output=train1_%j.log    # 输出日志（%j为作业ID，避免重复）
#SBATCH --error=train1_%j.err     # 错误日志
#SBATCH --nodes=1                   # 用1个节点（单节点足够）
#SBATCH --ntasks-per-node=1         # 单个节点只跑1个任务（代码未并行，多任务会冲突）
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # 给这个任务分配8个CPU核心（根据需求调整，如4/16）
#SBATCH --time=360:00:00             # 运行时间（4小时，根据生成数量调整）
#SBATCH --partition=a800             # GPU分区名（替换为你超算的实际分区，如gpu_v100）

# 1. 加载超算基础环境（若超算无需加载Anaconda，可删此句）
module load anaconda/2024.06

# 2. 激活你的conda环境
source activate tabPFN

# 3. 运行data.py（带参数，指定生成数量、保存目录等，根据需求调整）
python train1.py 

# 4. 运行完成后提示（可选）
echo "✅ 运行完成，日志已保存到 train1_${SLURM_JOB_ID}.log"