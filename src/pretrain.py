import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.io.stata import excessive_string_length_error

from src.generator import Generator
from src.set_encoder import SetEncoder
from tokenizer import FormulaTokenizer
from metrics import FormulaDistanceMetric

import os
import numpy as np
from tqdm import tqdm  # 推荐安装 tqdm 显示生成进度
from torch.utils.data import Dataset, DataLoader


class CacheDataset(Dataset):
    def __init__(self, data_source):
        """
        data_source: 可以是内存中的 dict，也可以是硬盘上的 .pt 文件路径
        """
        if isinstance(data_source, str):
            print(f"Loading data from {data_source}...")
            self.data = torch.load(data_source)
        else:
            self.data = data_source

        self.length = len(self.data['target_sim'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'x1': self.data['x1'][idx],
            'y1': self.data['y1'][idx],
            'f1': self.data['f1_tokens'][idx],  # Query Formula

            'x2': self.data['x2'][idx],
            'y2': self.data['y2'][idx],
            'f2': self.data['f2_tokens'][idx],  # Context Formula

            'target_sim': self.data['target_sim'][idx]
        }


def generate_and_save_data(
        generator,
        metric_calculator,
        cache_size,
        save_path=None,  # [新增] 保存路径
        target_dim = 16
):
    """
    生成数据，计算相似度，并Token化公式。
    如果 save_path 不为空，则保存到硬盘。
    """

    # 容器
    list_x1, list_y1 = [], []
    list_x2, list_y2 = [], []
    list_target_sims = []

    # [新增] 公式 Token 容器
    list_f1_tokens = []
    list_f2_tokens = []

    print(f"Generating data ({cache_size} samples)...")
    i = 0
    while i < cache_size:

        try:

            # 1. 生成 Anchor (Query)
            x_f, y_f, f_expr, f, f_id, input_dimension, org_tree = generator.sample_formula()
            x_padded = np.zeros((x_f.shape[0], target_dim), dtype=np.float32)
            x_padded[:, :x_f.shape[1]] = x_f
            x_f = x_padded

            # 2. 生成 Comparison (Context / Retrieved)
            rand = np.random.rand()
            if rand < 0.15:
                x_g, y_g, g_expr, g, g_id, g_input_dimension, g_org_tree = x_f, y_f, f_expr, f, f_id, input_dimension, org_tree
                x_g, y_g = generator.resample_formula(org_tree, input_dimension)
            elif rand < 0.3:
                x_g, y_g, g_expr, g, g_id, g_input_dimension, g_org_tree = generator.mutate_formula(f, input_dimension, org_tree, edit=3)
            elif rand < 0.5:
                x_g, y_g, g_expr, g, g_id, g_input_dimension, g_org_tree = generator.mutate_formula(f, input_dimension, org_tree, edit=5)
            else:
                x_g, y_g, g_expr, g, g_id, g_input_dimension, g_org_tree = generator.sample_formula()

            x_padded = np.zeros((x_g.shape[0], target_dim), dtype=np.float32)
            x_padded[:, :x_g.shape[1]] = x_g
            x_g = x_padded

            target_sim = metric_calculator.compute_similarity(f_expr, g_expr)

            # Append to lists
            list_x1.append(torch.tensor(x_f, dtype=torch.float32))
            list_y1.append(torch.tensor(y_f, dtype=torch.float32))
            list_x2.append(torch.tensor(x_g, dtype=torch.float32))
            list_y2.append(torch.tensor(y_g, dtype=torch.float32))
            list_target_sims.append(target_sim)

            # 转为 Tensor 并存入列表 (LongTensor)
            list_f1_tokens.append(torch.tensor(f_id, dtype=torch.long))
            list_f2_tokens.append(torch.tensor(g_id, dtype=torch.long))
            # print(f'Dataset {i} generated!')
            i += 1

        except:
            pass

    # 5. 统一堆叠 (Stack)
    data_dict = {
        'x1': torch.stack(list_x1),  # (B, N, D)
        'y1': torch.stack(list_y1),  # (B, N)
        'x2': torch.stack(list_x2),
        'y2': torch.stack(list_y2),
        'target_sim': torch.tensor(list_target_sims, dtype=torch.float32),  # (B,)

        # [新增] 公式序列
        'f1_tokens': torch.stack(list_f1_tokens),  # (B, Max_Len)
        'f2_tokens': torch.stack(list_f2_tokens)  # (B, Max_Len)
    }

    # 6. [新增] 保存到硬盘
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data_dict, save_path)
        print(f"Data saved to {save_path}")

    return data_dict


class StructureMetricTrainer:
    def __init__(self,
                 set_encoder,
                 device,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 alpha=0.5,
                 max_variables=16,
                 max_length=103,
                 batch_size=256,  # 训练时的 Batch Size
                 sample_num=100,
                 save_path='../training_data'
                 ):

        self.tokenizer = FormulaTokenizer(max_variables=max_variables, max_length=max_length)
        self.metric = FormulaDistanceMetric(self.tokenizer)
        self.model = set_encoder
        self.device = device
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.alpha = alpha

        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train_epoch(self, dataloader):
        """
        在一个固定的 Cache 数据集上跑一个 Epoch
        """
        self.model.train()
        total_loss = 0
        count = 0

        for batch in dataloader:
            # 1. 数据搬运到 GPU
            x1 = batch['x1'].to(self.device)
            y1 = batch['y1'].to(self.device)
            x2 = batch['x2'].to(self.device)
            y2 = batch['y2'].to(self.device)
            targets = batch['target_sim'].to(self.device)

            # 2. 清空梯度
            self.optimizer.zero_grad()

            # 3. Forward
            emb1 = F.normalize(self.model(x1, y1), p=2, dim=1)
            emb2 = F.normalize(self.model(x2, y2), p=2, dim=1)

            pred_sims = (emb1 * emb2).sum(dim=1)

            # print(pred_sims)
            # print(targets)

            # 4. Backward
            loss = self.mse_loss(pred_sims, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count

    def run_training_loop(self, generator, num_files=10, cache_size=1000, total_epochs=50):
        """
        Args:
            generator: 公式生成器
            num_files: 总共生成多少个 dataset 文件 (相当于总数据集大小 = num_files * cache_size)
            cache_size: 每个文件包含多少个样本
            total_epochs: 遍历所有文件的次数
        """

        # === Phase 1: 数据预生成 (Offline Generation) ===
        data_dir = os.path.join(self.save_path, "pregenerated_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print(f"=== Phase 1: Generating {num_files} dataset files... ===")
        generated_files = []

        for i in range(num_files):
            # [修复] 文件名带上索引，避免覆盖
            file_name = f"dataset_{i}.pt"
            file_path = os.path.join(data_dir, file_name)
            generated_files.append(file_path)

            if os.path.exists(file_path):
                print(f"File {file_name} already exists. Skipping generation.")
                continue

            # 调用生成函数
            generate_and_save_data(
                generator=generator,
                metric_calculator=self.metric,
                cache_size=cache_size,
                save_path=file_path,
                target_dim=16
            )
            print(f"Generated {i + 1}/{num_files}: {file_name}")

        print("=== Data Generation Complete. Starting Training... ===")

        # === Phase 2: 迭代训练 (Iterative Training) ===
        # 逻辑：每个 Epoch 都要遍历所有的文件

        for epoch in range(total_epochs):
            print(f"\n>>> Global Epoch {epoch + 1}/{total_epochs}")

            epoch_loss = 0
            file_count = 0

            # [优化] 每个 Epoch 开始前打乱文件读取顺序，增加随机性
            # 类似于 Shuffle
            np.random.shuffle(generated_files)

            # 遍历每一个文件
            for file_path in generated_files:
                # 1. 加载当前文件
                # 这里的 IO 可能会成为瓶颈，如果是 SSD 没问题
                # 如果是 HDD，可以使用 torch.utils.data.ChainDataset 或者预加载下个文件
                dataset = CacheDataset(file_path)

                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,  # 文件内部打乱
                    num_workers=0,  # Mac 上用 0 可能更稳，Linux 用 4
                    pin_memory=True
                )

                # 2. 训练这个文件 (只训练 1 次！不要在这里多 Epoch)
                # 我们希望模型快速看一遍这个文件，然后去看下一个文件
                # 这样可以保持梯度的多样性
                loss = self.train_epoch(dataloader)
                epoch_loss += loss
                file_count += 1

                # print(f"   Processed {os.path.basename(file_path)} | Loss: {loss:.4f}")

            avg_epoch_loss = epoch_loss / file_count
            print(f"   [Epoch {epoch + 1} Summary] Avg Loss: {avg_epoch_loss:.6f}")

            # 3. 保存 Checkpoint
            if (epoch + 1) % 5 == 0:
                ckpt_name = f"set_encoder_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), os.path.join(self.save_path, ckpt_name))
                print(f"   Checkpoint saved: {ckpt_name}")


if __name__ == "__main__":
    # 1. 初始化
    generator = Generator()  # 你的生成器已经包含了 RandomState 的修复
    set_encoder = SetEncoder(num_x_features=16)

    # 2. 选择设备 (Mac M系列)
    device = 'cpu'
    print(f"Training on {device}")

    # 3. 初始化 Trainer
    trainer = StructureMetricTrainer(set_encoder=set_encoder, device=device)

    # 4. 开始流程
    # 假设我们生成 50 个文件，每个文件 2000 个样本 => 总共 10万数据
    # 训练 100 个 Epoch
    trainer.run_training_loop(
        generator=generator,
        num_files=10,  # 生成 50 个不同的文件
        cache_size=100,  # 每个文件 2000 条数据
        total_epochs=100  # 总共把这 50 个文件过 100 遍
    )