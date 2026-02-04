import torch
import torch.nn as nn
import torch.nn.functional as F

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
        tokenizer,  # [新增] 需要 tokenizer 把公式转成 ID
        alpha,
        cache_size,
        sample_num,
        save_path=None  # [新增] 保存路径
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
    for _ in tqdm(range(cache_size)):
        # 1. 生成 Anchor (Query)
        f_base = generator.sample_formula()
        x1, y1 = f_base.sample(sample_num)

        # 2. 生成 Comparison (Context / Retrieved)
        rand = np.random.rand()
        if rand < 0.25:
            f_target = f_base  # Dist=0
            x2, y2 = f_base.sample(sample_num)
        elif rand < 0.50:
            f_target = generator.mutate_formula(f_base, edits=1)
            x2, y2 = f_target.sample(sample_num)
        elif rand < 0.75:
            f_target = generator.mutate_formula(f_base, edits=3)
            x2, y2 = f_target.sample(sample_num)
        else:
            f_target = generator.sample_formula()
            x2, y2 = f_target.sample(sample_num)

        # 3. 计算 Target Similarity (用于 SetEncoder)
        base_prefix = f_base.to_prefix_tokens()
        target_prefix = f_target.to_prefix_tokens()

        raw_dist = metric_calculator.compute_distance(base_prefix, target_prefix)
        target_sim = np.exp(-alpha * raw_dist)

        # 4. [新增] Token化公式 (用于 TabPFN)
        # tokenizer.encode 会自动处理 Padding, SOS, EOS, Truncation
        # 返回的是 List[int]
        f1_ids = tokenizer.encode(base_prefix)
        f2_ids = tokenizer.encode(target_prefix)

        # Append to lists
        list_x1.append(x1)
        list_y1.append(y1)
        list_x2.append(x2)
        list_y2.append(y2)
        list_target_sims.append(target_sim)

        # 转为 Tensor 并存入列表 (LongTensor)
        list_f1_tokens.append(torch.tensor(f1_ids, dtype=torch.long))
        list_f2_tokens.append(torch.tensor(f2_ids, dtype=torch.long))

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
            targets = batch['target'].to(self.device)

            # 2. 清空梯度
            self.optimizer.zero_grad()

            # 3. Forward
            emb1 = F.normalize(self.model(x1, y1), p=2, dim=1)
            emb2 = F.normalize(self.model(x2, y2), p=2, dim=1)

            pred_sims = (emb1 * emb2).sum(dim=1)

            # 4. Backward
            loss = self.mse_loss(pred_sims, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count

    def run_training_loop(self, generator, num_cycles=100, cache_size=10000, epochs_per_cycle=20):
        """
        主训练循环
        Args:
            generator: 公式生成器
            num_cycles: 总共进行多少次 "生成-训练" 循环
            cache_size: 每次循环生成的样本数量 (比如 10,000)
            epochs_per_cycle: 每批数据训练几个 Epoch (通常 1 次就够了，避免过拟合这批特定数据)
        """
        print(f"Start Training: {num_cycles} cycles, {cache_size} samples per cycle.")

        for cycle in range(num_cycles):
            dataset_path = os.path.join(self.save_path, f"dataset_generated_{num_cycles}")
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            print(f"\n=== Cycle {cycle + 1}/{num_cycles} ===")

            # 1. 在线生成数据集 (CPU)
            raw_data = generate_and_save_data(
                generator=generator,
                metric_calculator=self.metric,
                alpha=self.alpha,
                cache_size=cache_size,
                sample_num=self.sample_num,
                save_path=os.path.join(dataset_path, "dataset.pt"),
                tokenizer=self.tokenizer
            )

            # 2. 封装成 Dataset 和 DataLoader
            dataset = CacheDataset(raw_data)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,  # 打乱这批数据
                num_workers=4,  # 开启多进程加速数据传输
                pin_memory=True  # 加速 CPU -> GPU 传输
            )

            # 3. 训练 (GPU)
            for epoch in range(epochs_per_cycle):
                avg_loss = self.train_epoch(dataloader)
                print(f"   Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

            # 4. (可选) 保存 Checkpoint
            if (cycle + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"set_encoder_cycle_{cycle + 1}.pth")