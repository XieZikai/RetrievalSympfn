import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import gc  # 引入垃圾回收模块
from tqdm import tqdm

from set_encoder import SympfnModel, SetEncoder
from tokenizer import FormulaTokenizer

import logging  # <--- 1. 导入
from logging import getLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("train1.log", mode="w"),
        logging.StreamHandler()
    ])
logger = getLogger(__name__)

# --- Dataset 保持不变 ---
class SymPFNDataset(Dataset):
    def __init__(self, data_path):
        # print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path, map_location='cpu')  # 强制确保先加载到内存，不要直接上 GPU
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
            'query_x': self.q_x[idx],
            'query_y': self.q_y[idx],
            'query_f': self.q_f[idx],
            'ctx_x': self.c_x[idx],
            'ctx_y': self.c_y[idx],
            'ctx_f': self.c_f[idx]
        }


# --- 优化后的 Trainer ---
class PFNTrainer:
    def __init__(self,
                 set_encoder,
                 sympfn_model,
                 device,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 batch_size=256,
                 save_path='../training_data',
                 max_variables=16,
                 max_length=103,
                 val_ratio=0.1
                 ):

        self.tokenizer = FormulaTokenizer(max_variables=max_variables, max_length=max_length)
        self.set_encoder = set_encoder
        self.sympfn_model = sympfn_model
        self.device = device
        self.batch_size = batch_size
        self.save_path = save_path
        self.val_ratio = val_ratio

        self.history = {'train_loss': [], 'val_loss': []}

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        self.optimizer = torch.optim.AdamW(
            self.sympfn_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 冻结 SetEncoder
        for param in self.set_encoder.parameters():
            param.requires_grad = False
        self.set_encoder.eval()
        self.set_encoder.to(device)
        self.sympfn_model.to(device)

    def generate_dataloaders(self, file_name):
        full_path = os.path.join(self.save_path, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")

        # 使用 map_location='cpu' 防止加载数据时占用显存
        full_dataset = SymPFNDataset(full_path)

        total_size = len(full_dataset)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - val_size

        logger.info(f"Splitting: Train={train_size}, Val={val_size}")

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        # num_workers=0 在调试显存问题时最安全
        # pin_memory=True 加速数据传输
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader

    def train_epoch(self, dataloader, epoch_idx):
        self.sympfn_model.train()
        self.set_encoder.eval()

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch_idx} [Train]", leave=False)

        for batch in pbar:
            # 1. 搬运数据
            q_x = batch['query_x'].to(self.device, non_blocking=True)
            q_y = batch['query_y'].to(self.device, non_blocking=True)
            q_f = batch['query_f'].to(self.device, non_blocking=True)

            ctx_x = batch['ctx_x'].to(self.device, non_blocking=True)
            ctx_y = batch['ctx_y'].to(self.device, non_blocking=True)
            ctx_f = batch['ctx_f'].to(self.device, non_blocking=True)

            # 2. 计算 Context Embedding (No Grad)
            with torch.no_grad():
                curr_batch_size, k, n_points, n_features = ctx_x.shape

                # Flatten
                flat_ctx_x = ctx_x.view(-1, n_points, n_features)
                flat_ctx_y = ctx_y.view(-1, n_points, 1)

                raw_emb_ctx = self.set_encoder(flat_ctx_x, flat_ctx_y)
                flat_emb_ctx = F.normalize(raw_emb_ctx, p=2, dim=1)

                d_train_embs = flat_emb_ctx.view(curr_batch_size, k, -1)

            # 3. Forward
            self.optimizer.zero_grad()

            logits = self.sympfn_model(
                xs_test=q_x,
                ys_test=q_y,
                test_mask=None,
                d_train_embs=d_train_embs,
                f_train_tokens=ctx_f,
                target_f_tokens=q_f
            )

            # 4. Loss
            targets = q_f[:, 1:].contiguous().view(-1)
            predictions = logits.view(-1, logits.shape[-1])
            loss = self.ce_loss(predictions, targets)

            # 5. Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sympfn_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # [关键修改] 手动删除变量，释放计算图
            del loss, logits, predictions, targets, q_x, q_y, q_f, ctx_x, ctx_y, ctx_f, d_train_embs

        # Epoch 结束清理缓存
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader, epoch_idx):
        self.sympfn_model.eval()
        self.set_encoder.eval()

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch_idx} [Val]  ", leave=False)

        # 验证前先清理一下，确保 Training 留下的垃圾被收走
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in pbar:
                # 数据上 GPU
                q_x = batch['query_x'].to(self.device, non_blocking=True)
                q_y = batch['query_y'].to(self.device, non_blocking=True)
                q_f = batch['query_f'].to(self.device, non_blocking=True)

                ctx_x = batch['ctx_x'].to(self.device, non_blocking=True)
                ctx_y = batch['ctx_y'].to(self.device, non_blocking=True)
                ctx_f = batch['ctx_f'].to(self.device, non_blocking=True)

                # Context Logic
                curr_batch_size, k, n_points, n_features = ctx_x.shape
                flat_ctx_x = ctx_x.view(-1, n_points, n_features)
                flat_ctx_y = ctx_y.view(-1, n_points, 1)

                raw_emb_ctx = self.set_encoder(flat_ctx_x, flat_ctx_y)
                flat_emb_ctx = F.normalize(raw_emb_ctx, p=2, dim=1)
                d_train_embs = flat_emb_ctx.view(curr_batch_size, k, -1)

                # Forward
                logits = self.sympfn_model(
                    xs_test=q_x,
                    ys_test=q_y,
                    test_mask=None,
                    d_train_embs=d_train_embs,
                    f_train_tokens=ctx_f,
                    target_f_tokens=q_f
                )

                targets = q_f[:, 1:].contiguous().view(-1)
                predictions = logits.view(-1, logits.shape[-1])

                # 这里的 loss 是一个标量 Tensor
                loss = self.ce_loss(predictions, targets)

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                # 手动删除
                del loss, logits, predictions, targets, q_x, q_y, q_f, ctx_x, ctx_y, ctx_f, d_train_embs

        # 验证结束后再次清理
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return total_loss / len(dataloader)

    def save_logs(self):
        log_path = os.path.join(self.save_path, "training_logs.json")
        try:
            with open(log_path, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logger.info(f"Warning: Failed to save logs: {e}")

    def run_training_loop(self, train_files, epochs_per_file=1):
        logger.info(f"Start SymPFN Training on {len(train_files)} files.")

        for file_idx, file_name in enumerate(train_files):
            logger.info(f"\n=== Processing {file_name} ({file_idx + 1}/{len(train_files)}) ===")

            # 清理之前的 DataLoader 残留
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            try:
                train_loader, val_loader = self.generate_dataloaders(file_name)
            except FileNotFoundError as e:
                logger.info(e)
                continue

            for epoch in range(epochs_per_file):
                # 训练
                train_loss = self.train_epoch(train_loader, epoch_idx=epoch)

                # 验证
                val_loss = self.validate_epoch(val_loader, epoch_idx=epoch)

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                logger.info(f"   [Epoch {epoch}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # 保存模型
            ckpt_name = f"sympfn_epoch_{file_idx}.pth"
            torch.save(self.sympfn_model.state_dict(), os.path.join(self.save_path, ckpt_name))
            self.save_logs()

            # 文件切换时，彻底删除 loader，防止内存泄漏
            del train_loader, val_loader
            gc.collect()


if __name__ == "__main__":
    # --- 1. 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using Device: {device}")

    # --- 2. SetEncoder ---
    set_encoder = SetEncoder(num_x_features=16)
    output_dir = '~/Data/zikaix/RetrievalSympfn/training_data'
    checkpoint_path = os.path.join(output_dir, "set_encoder_epoch_100.pth")

    if os.path.exists(checkpoint_path):
        ##############修改，读取checkpoion###########################
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        try:
            set_encoder.load_state_dict(state_dict)
            logger.info("SetEncoder weights loaded.")
        except RuntimeError as e:
            logger.info(f"Weight loading error: {e}")
    else:
        logger.info("Warning: SetEncoder weights not found.")

    set_encoder.to(device)
    set_encoder.eval()

    # --- 3. SympfnModel ---
    sympfn_model = SympfnModel(
        set_encoder=set_encoder,
        vocab_size=103
    )
    sympfn_model.to(device)

    # --- 4. Trainer ---
    output_dir = '/fs0/home/zikaix/Data/zikaix/RetrievalSympfn/training_data1'
    training_data_path = os.path.join(output_dir, "sympfn_top20_data.pt")
    data_dir = os.path.dirname(training_data_path)
    data_filename = os.path.basename(training_data_path)

    trainer = PFNTrainer(
        set_encoder=set_encoder,
        sympfn_model=sympfn_model,
        device=device,
        batch_size=16,
        learning_rate=1e-4,
        save_path=data_dir,
        val_ratio=0.1  # 使用 10% 的数据作为验证集
    )

    # --- 5. Start ---
    trainer.run_training_loop(
        train_files=[data_filename],
        epochs_per_file=50
    )
