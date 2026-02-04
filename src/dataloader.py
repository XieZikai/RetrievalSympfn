import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os


class StaticContrastiveDataset(Dataset):
    def __init__(self, data_root, points_per_view=50):
        """
        data_root: 包含所有 .pt 文件的根目录
        points_per_view: 每个 View 给 SetEncoder 看多少个点
        """
        super().__init__()
        # 1. 扫描所有文件
        # 假设文件结构是 data_root/formula_0.pt, data_root/formula_1.pt ...
        # 或者 data_root/category/formula.pt
        self.file_list = sorted(glob.glob(os.path.join(data_root, "**", "*.pt"), recursive=True))
        self.points_per_view = points_per_view

        print(f"Dataset initialized. Found {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. 加载文件
        path = self.file_list[idx]
        try:
            # 假设 data 是 {'x': [N, D], 'y': [N]}
            data = torch.load(path)
            full_x = data['x']
            full_y = data['y']
        except Exception as e:
            # 容错处理：如果文件坏了，随机换一个
            return self.__getitem__((idx + 1) % len(self))

        total_points = full_x.shape[0]

        # --- 构造 View 1 (Anchor) ---
        # 策略：随机采样 indices
        # replace=False 保证点不重复
        indices1 = torch.randperm(total_points)[:self.points_per_view]
        x1 = full_x[indices1]
        y1 = full_y[indices1]

        # --- 构造 View 2 (Positive) ---
        # 策略 A: 互斥采样 (Disjoint Sampling) - 推荐
        # 尽量让 View 2 的点和 View 1 不一样，迫使模型学“形状”而不是“背诵点”

        # 这里的实现技巧：先生成一个全排列，View1取前50，View2取后50
        all_indices = torch.randperm(total_points)

        # 重新取 View 1
        idx1 = all_indices[:self.points_per_view]
        x1 = full_x[idx1]
        y1 = full_y[idx1]

        # 取 View 2
        idx2 = all_indices[self.points_per_view: self.points_per_view * 2]
        # 如果总点数不够分两份，就允许重复
        if len(idx2) < self.points_per_view:
            idx2 = torch.randint(0, total_points, (self.points_per_view,))

        x2 = full_x[idx2]
        y2 = full_y[idx2]

        # 策略 B: 噪声增强 (Noise Augmentation) - 强烈推荐
        # 即使点不同，加上噪声也能防止过拟合
        noise_level = 0.05
        y2 = y2 + torch.randn_like(y2) * noise_level * (y2.std() + 1e-8)

        return {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2
        }


# --- 训练时的调用 ---
def train():
    # 只需要这就够了！
    dataset = StaticContrastiveDataset("./my_data_folder")

    loader = DataLoader(
        dataset,
        batch_size=256,  # 越大越好！这是提供负样本的关键
        shuffle=True,  # 必须为 True！这是多元化的来源
        num_workers=4,  # 加速读取
        drop_last=True  # 丢弃最后凑不够数的 batch
    )

    for batch in loader:
        x1, y1 = batch['x1'], batch['y1']
        x2, y2 = batch['x2'], batch['y2']

        # 放到 GPU
        # ...

        # 拼接成 (2*Batch, ...) 喂给 SetEncoder
        # 计算 InfoNCE Loss


class ContrastiveFunctionDataset(Dataset):
    def __init__(self, data_root, formula_tokenizer=None, epoch_multiplier=1):
        """
        data_root: 存放数据集文件夹的根目录
        epoch_multiplier: 虚拟扩大数据集长度，让一个 epoch 跑久一点
        """
        self.data_root = data_root
        # 获取所有数据集的路径列表
        # 假设结构: data_root/func_001/data.pt
        self.file_list = glob.glob(os.path.join(data_root, "*", "data.pt"))
        self.epoch_multiplier = epoch_multiplier

        print(f"Found {len(self.file_list)} function datasets.")

    def __len__(self):
        # 虚拟长度，避免频繁重置 DataLoader
        return len(self.file_list) * self.epoch_multiplier

    def __getitem__(self, idx):
        # 1. 真实索引映射
        real_idx = idx % len(self.file_list)
        path = self.file_list[real_idx]

        # 2. 加载数据 (假设存的是 Tensor: [N_total, X_dim], [N_total])
        # 如果是 CSV 就在这里解析
        data = torch.load(path)
        full_x = data['x']  # Shape: [1000, 16]
        full_y = data['y']  # Shape: [1000]

        total_points = full_x.shape[0]
        sample_size = 100  # 我们希望 SetEncoder 看到的点数

        # --- 3. 构造 View 1 (Anchor) ---
        # 随机抽取 indices
        idx1 = torch.randperm(total_points)[:sample_size]
        x1 = full_x[idx1]
        y1 = full_y[idx1]

        # --- 4. 构造 View 2 (Positive) ---
        # 关键：必须与 View 1 不同，但来自同一个分布/公式
        # 策略：抽取完全不同的点，或者加噪声
        idx2 = torch.randperm(total_points)[:sample_size]
        x2 = full_x[idx2]
        y2 = full_y[idx2]

        # 数据增强：对 View 2 的 y 加一点噪声，增加鲁棒性
        noise = torch.randn_like(y2) * 0.05 * y2.std()
        y2 = y2 + noise

        # --- 5. (可选) 加载 Hard Negative ---
        # 如果你已经按 Edit Distance 聚类好了，这里可以加载一个 "hard_neg_path"
        # 但标准做法是只返回 views，靠 Batch 内的其他数据做负样本

        return {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            # 如果需要公式用于 debug 或 metric learning
            # 'formula_seq': data['formula_tokens']
        }


# --- DataLoader 构建 ---
def get_dataloader(data_root, batch_size=64):
    dataset = ContrastiveFunctionDataset(data_root)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 必须 Shuffle！这保证了 Batch 内的数据是随机的负样本
        num_workers=4,  # 并行加载
        pin_memory=True,  # 加速 GPU 传输
        drop_last=True  # 丢弃最后一个不完整的 Batch (这就避免了形状对不齐)
    )
    return loader