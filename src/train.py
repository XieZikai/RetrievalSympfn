import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm

# 假设 tokenizer 和 metrics 已经定义
from tokenizer import FormulaTokenizer
from metrics import FormulaDistanceMetric


# --- 定义 Dataset (负责搬运 .pt 数据) ---
class SymPFNDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path)
        # 直接读取保存好的 Tensor (Query 和 Context 已经配对好了)
        self.q_x = data['query_x']
        self.q_y = data['query_y']
        self.q_f = data['query_f']

        self.c_x = data['ctx_x']
        self.c_y = data['ctx_y']
        self.c_f = data['ctx_f']

    def __len__(self):
        return len(self.q_x)

    def __getitem__(self, idx):
        return {
            'query_x': self.q_x[idx], 'query_y': self.q_y[idx], 'query_f': self.q_f[idx],
            'ctx_x': self.c_x[idx], 'ctx_y': self.c_y[idx], 'ctx_f': self.c_f[idx]
        }


# --- 补全后的 PFNTrainer ---
class PFNTrainer:
    def __init__(self,
                 set_encoder,
                 sympfn_model,
                 device,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 alpha=0.5,
                 max_variables=16,
                 max_length=103,
                 batch_size=256,
                 sample_num=100,
                 save_path='../training_data'  # 这里指向存放 generated .pt 文件的目录
                 ):

        self.tokenizer = FormulaTokenizer(max_variables=max_variables, max_length=max_length)
        self.metric = FormulaDistanceMetric(self.tokenizer)

        self.set_encoder = set_encoder
        self.sympfn_model = sympfn_model
        self.device = device
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.alpha = alpha
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # --- Loss ---
        # 忽略 padding token 的 loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        # --- Optimizer ---
        # 只优化 SymPFN，SetEncoder 保持冻结
        self.optimizer = torch.optim.AdamW(
            self.sympfn_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # --- Freeze SetEncoder ---
        print("Freezing SetEncoder parameters...")
        for param in self.set_encoder.parameters():
            param.requires_grad = False
        self.set_encoder.eval()

    def generate_dataloader(self, file_name):
        """
        加载指定的 .pt 文件并返回 DataLoader
        Args:
            file_name: e.g., "sympfn_training_data_cycle_0.pt"
        """
        full_path = os.path.join(self.save_path, file_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")

        dataset = SymPFNDataset(full_path)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练时必须打乱
            num_workers=4,  # 多进程加速数据读取
            pin_memory=True  # 加速 CPU -> GPU 传输
        )
        return dataloader

    def train_epoch(self, dataloader, epoch_idx):
        """
        训练一个 Epoch
        """
        self.sympfn_model.train()  # 开启 Dropout/BN 更新
        self.set_encoder.eval()  # 保持冻结

        total_loss = 0
        # 使用 tqdm 显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} Training")

        for batch in pbar:
            # 1. 数据上 GPU
            # Query (Target Task)
            q_x = batch['query_x'].to(self.device)
            q_y = batch['query_y'].to(self.device)
            q_f = batch['query_f'].to(self.device)  # Shape: [B, Seq_Len]

            # Context (Reference Info)
            ctx_x = batch['ctx_x'].to(self.device)
            ctx_y = batch['ctx_y'].to(self.device)
            ctx_f = batch['ctx_f'].to(self.device)  # Shape: [B, Seq_Len]

            # 2. 计算 Context Embedding (On-the-fly)
            with torch.no_grad():
                # SetEncoder 只需要处理 Context，因为它是给 SymPFN 提供参考的
                # 输入: [B, N, D] -> 输出: [B, D]
                raw_emb_ctx = self.set_encoder(ctx_x, ctx_y)

                # [关键] 必须做 L2 归一化，因为 FAISS 检索时用的是归一化向量
                # 这样 SymPFN 看到的 Embedding 幅度才是一致的
                emb_ctx = F.normalize(raw_emb_ctx, p=2, dim=1)

                # 维度调整: SymPFN 期望 Context 是序列形式 [Batch, K, Dim]
                # 这里我们每个 Query 只有一个 Context (Top-1)，所以 K=1
                d_train_embs = emb_ctx.unsqueeze(1)  # [B, 1, D]

            # 3. 维度调整 Context Tokens
            # [B, Len] -> [B, 1, Len]
            f_train_tokens = ctx_f.unsqueeze(1)

            # 4. SymPFN Forward
            self.optimizer.zero_grad()

            # 假设你的 SymPFN forward 接受以下参数
            logits = self.sympfn_model(
                xs_test=q_x,
                ys_test=q_y,
                test_mask=None,  # 假设全是有效点
                d_train_embs=d_train_embs,  # 检索到的上下文 Embedding
                f_train_tokens=f_train_tokens,  # 检索到的上下文公式
                f_train_mask=None,  # 假设公式无需 mask (或已 pad)
                target_f_tokens=q_f  # 目标公式 (用于 Teacher Forcing)
            )
            # logits shape: [Batch, Seq_Len - 1, Vocab_Size]

            # 5. 计算 Loss (Shifted Targets)
            # 目标是预测下一个 token
            # 如果 q_f 是 [SOS, A, B, EOS, PAD]
            # targets 应该是 [A, B, EOS, PAD, PAD] (即 q_f[:, 1:])
            # logits 对应预测这些位置

            vocab_size = logits.shape[-1]
            targets = q_f[:, 1:].contiguous().view(-1)
            predictions = logits.view(-1, vocab_size)

            loss = self.ce_loss(predictions, targets)

            # 6. Backward
            loss.backward()

            # 梯度裁剪 (防止 Transformer 训练不稳定)
            torch.nn.utils.clip_grad_norm_(self.sympfn_model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(dataloader)

    def run_training_loop(self, train_files, epochs_per_file=1):
        """
        主训练循环
        Args:
            train_files: List[str]，例如 ["cycle_0.pt", "cycle_1.pt"]
            epochs_per_file: 每个文件训练多少轮 (通常 1 轮即可，因为数据量很大且不重复)
        """
        print(f"Start SymPFN Training on {len(train_files)} dataset files.")

        for file_idx, file_name in enumerate(train_files):
            print(f"\n=== Processing File {file_idx + 1}/{len(train_files)}: {file_name} ===")

            # 1. 加载数据
            try:
                dataloader = self.generate_dataloader(file_name)
            except FileNotFoundError as e:
                print(e)
                continue

            # 2. 训练 Epochs
            for epoch in range(epochs_per_file):
                avg_loss = self.train_epoch(dataloader, epoch_idx=epoch)
                print(f"   [File {file_name} - Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

            # 3. 每个 File 训练完后保存 Checkpoint
            # 这样如果训练中断，可以从下一个 File 继续
            ckpt_name = f"sympfn_checkpoint_file_{file_idx}.pth"
            save_full_path = os.path.join(self.save_path, ckpt_name)
            torch.save(self.sympfn_model.state_dict(), save_full_path)
            print(f"   Saved checkpoint to {ckpt_name}")

        print("All training completed.")