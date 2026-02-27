import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler  # <--- [新增] 分布式采样器
import torch.distributed as dist  # <--- [新增] 分布式通信库
from torch.nn.parallel import DistributedDataParallel as DDP  # <--- [新增] DDP 包装器

import os
import json
import gc
from tqdm import tqdm

from set_encoder import SympfnModel, SetEncoder
from tokenizer import FormulaTokenizer

import logging
from logging import getLogger


# --- 初始化日志的函数 (修改为只在主进程输出) ---
def setup_logger(rank):
    logger = getLogger(__name__)
    if rank == 0:  # 只让主进程写日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler("train_ddp.log", mode="w"),
                logging.StreamHandler()
            ])
    else:
        # 非主进程不输出 INFO，避免终端被刷屏
        logging.basicConfig(level=logging.WARNING)
    return logger


# --- Dataset 保持不变 ---
class SymPFNDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location='cpu')
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


class PFNTrainer:
    def __init__(self,
                 set_encoder,
                 sympfn_model,
                 local_rank,
                 global_rank,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 batch_size=256,
                 save_path='../training_data',
                 max_variables=16,
                 max_length=103,
                 val_ratio=0.1,
                 resume_checkpoint=None  # <--- [新增] 传入断点文件的路径
                 ):

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.device = torch.device(f"cuda:{local_rank}")

        self.tokenizer = FormulaTokenizer(max_variables=max_variables, max_length=max_length)
        self.save_path = save_path
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.history = {'train_loss': [], 'val_loss': []}

        if self.global_rank == 0 and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        # 1. 模型上卡并冻结 SetEncoder
        self.set_encoder = set_encoder.to(self.device)
        self.sympfn_model = sympfn_model.to(self.device)

        for param in self.set_encoder.parameters():
            param.requires_grad = False
        self.set_encoder.eval()

        # 2. DDP 包装
        self.sympfn_model = DDP(self.sympfn_model, device_ids=[self.local_rank])

        # 3. 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.sympfn_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 4. [新增] 断点续训加载逻辑
        self.start_file_idx = 0
        self.start_epoch = 0

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            if self.global_rank == 0:
                print(f"[*] Resuming training from checkpoint: {resume_checkpoint}")

            # 【重要】DDP 下必须使用 map_location 将模型参数均摊到各个卡上，否则 GPU0 会 OOM
            map_loc = f'cuda:{self.local_rank}'
            checkpoint = torch.load(resume_checkpoint, map_location=map_loc)

            # 1. 获取原始的 state_dict
            raw_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # 2. 清洗 Keys (去除可能存在的 'module.' 前缀)
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    clean_state_dict[k[7:]] = v  # 切掉前 7 个字符 'module.'
                else:
                    clean_state_dict[k] = v

            # 恢复模型权重 (注意 DDP 包装后的 key 带有 'module.' 前缀)
            self.sympfn_model.module.load_state_dict(clean_state_dict)

            # 恢复优化器状态 (动量等)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 恢复进度进度
            self.start_file_idx = checkpoint.get('file_idx', 0)
            self.start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个 epoch 开始

            # 恢复历史 loss 记录
            if 'history' in checkpoint:
                self.history = checkpoint['history']


    def generate_dataloaders(self, file_name, logger):
        full_path = os.path.join(self.save_path, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")

        full_dataset = SymPFNDataset(full_path)
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - val_size

        if self.global_rank == 0:
            logger.info(f"Splitting: Train={train_size}, Val={val_size}")

        # 使用同样的随机种子切分数据集，保证所有 GPU 切分结果一致
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        # [关键] 替换为 DistributedSampler
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.val_sampler = DistributedSampler(val_dataset, shuffle=False)

        # 注意：使用了 Sampler 后，DataLoader 的 shuffle 必须为 False
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            sampler=self.train_sampler, shuffle=False,
            num_workers=4, pin_memory=True  # DDP下可以适当开启 num_workers 加速
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            sampler=self.val_sampler, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, val_loader

    def _reduce_loss(self, loss_tensor):
        """ [新增] 将多张卡上的 loss 汇总求平均 """
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
        return loss_tensor

    def train_epoch(self, dataloader, epoch_idx, logger):
        self.sympfn_model.train()
        self.set_encoder.eval()
        total_loss = 0.0

        # 只在主卡显示进度条
        if self.global_rank == 0:
            pbar = tqdm(dataloader, desc=f"Ep {epoch_idx} [Train]", leave=False)
        else:
            pbar = dataloader

        for batch in pbar:
            q_x = batch['query_x'].to(self.device, non_blocking=True)
            q_y = batch['query_y'].to(self.device, non_blocking=True)
            q_f = batch['query_f'].to(self.device, non_blocking=True)
            ctx_x = batch['ctx_x'].to(self.device, non_blocking=True)
            ctx_y = batch['ctx_y'].to(self.device, non_blocking=True)
            ctx_f = batch['ctx_f'].to(self.device, non_blocking=True)

            with torch.no_grad():
                curr_batch_size, k, n_points, n_features = ctx_x.shape
                flat_ctx_x = ctx_x.view(-1, n_points, n_features)
                flat_ctx_y = ctx_y.view(-1, n_points, 1)
                raw_emb_ctx = self.set_encoder(flat_ctx_x, flat_ctx_y)
                flat_emb_ctx = F.normalize(raw_emb_ctx, p=2, dim=1)
                d_train_embs = flat_emb_ctx.view(curr_batch_size, k, -1)

            self.optimizer.zero_grad()
            logits = self.sympfn_model(
                xs_test=q_x, ys_test=q_y, test_mask=None,
                d_train_embs=d_train_embs, f_train_tokens=ctx_f, target_f_tokens=q_f
            )

            targets = q_f[:, 1:].contiguous().view(-1)
            predictions = logits.view(-1, logits.shape[-1])
            loss = self.ce_loss(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sympfn_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 汇总各卡的 loss (不参与梯度计算，仅用于记录)
            with torch.no_grad():
                reduced_loss = self._reduce_loss(loss.clone())

            total_loss += reduced_loss.item()
            if self.global_rank == 0:
                pbar.set_postfix({'loss': f"{reduced_loss.item():.4f}"})

            del loss, logits, predictions, targets, q_x, q_y, q_f, ctx_x, ctx_y, ctx_f, d_train_embs

        torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader, epoch_idx, logger):
        self.sympfn_model.eval()
        total_loss = 0.0

        if self.global_rank == 0:
            pbar = tqdm(dataloader, desc=f"Ep {epoch_idx} [Val]  ", leave=False)
        else:
            pbar = dataloader

        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in pbar:
                q_x = batch['query_x'].to(self.device, non_blocking=True)
                q_y = batch['query_y'].to(self.device, non_blocking=True)
                q_f = batch['query_f'].to(self.device, non_blocking=True)
                ctx_x = batch['ctx_x'].to(self.device, non_blocking=True)
                ctx_y = batch['ctx_y'].to(self.device, non_blocking=True)
                ctx_f = batch['ctx_f'].to(self.device, non_blocking=True)

                curr_batch_size, k, n_points, n_features = ctx_x.shape
                flat_ctx_x = ctx_x.view(-1, n_points, n_features)
                flat_ctx_y = ctx_y.view(-1, n_points, 1)
                raw_emb_ctx = self.set_encoder(flat_ctx_x, flat_ctx_y)
                flat_emb_ctx = F.normalize(raw_emb_ctx, p=2, dim=1)
                d_train_embs = flat_emb_ctx.view(curr_batch_size, k, -1)

                logits = self.sympfn_model(
                    xs_test=q_x, ys_test=q_y, test_mask=None,
                    d_train_embs=d_train_embs, f_train_tokens=ctx_f, target_f_tokens=q_f
                )

                targets = q_f[:, 1:].contiguous().view(-1)
                predictions = logits.view(-1, logits.shape[-1])
                loss = self.ce_loss(predictions, targets)

                # 汇总各卡 Loss
                reduced_loss = self._reduce_loss(loss.clone())
                total_loss += reduced_loss.item()

                if self.global_rank == 0:
                    pbar.set_postfix({'loss': f"{reduced_loss.item():.4f}"})

                del loss, logits, predictions, targets, q_x, q_y, q_f, ctx_x, ctx_y, ctx_f, d_train_embs

        torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def run_training_loop(self, train_files, epochs_per_file, logger):
        if self.global_rank == 0:
            logger.info(f"Start SymPFN DDP Training on {len(train_files)} files.")

        for file_idx, file_name in enumerate(train_files):
            # [新增] 跳过已经彻底训练完的文件
            if file_idx < self.start_file_idx:
                continue

            if self.global_rank == 0:
                logger.info(f"\n=== Processing {file_name} ({file_idx + 1}/{len(train_files)}) ===")

            torch.cuda.empty_cache()
            gc.collect()

            try:
                train_loader, val_loader = self.generate_dataloaders(file_name, logger)
            except FileNotFoundError as e:
                if self.global_rank == 0: logger.error(e)
                continue

            for epoch in range(epochs_per_file):
                # [新增] 跳过当前文件内已经训练完的 epoch
                if file_idx == self.start_file_idx and epoch < self.start_epoch:
                    continue

                self.train_sampler.set_epoch(epoch)

                train_loss = self.train_epoch(train_loader, epoch_idx=epoch, logger=logger)
                val_loss = self.validate_epoch(val_loader, epoch_idx=epoch, logger=logger)

                if self.global_rank == 0:
                    self.history['train_loss'].append(train_loss)
                    self.history['val_loss'].append(val_loss)
                    logger.info(f"   [Epoch {epoch}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

                    # [新增] 保存最新的全局 Checkpoint 字典
                    latest_ckpt_path = os.path.join(self.save_path, "latest_sympfn_checkpoint.pth")
                    checkpoint_dict = {
                        'model_state_dict': self.sympfn_model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'file_idx': file_idx,
                        'epoch': epoch,
                        'history': self.history
                    }
                    torch.save(checkpoint_dict, latest_ckpt_path)

                    # 保存 loss 记录到 JSON
                    log_path = os.path.join(self.save_path, "training_logs.json")
                    with open(log_path, 'w') as f:
                        json.dump(self.history, f, indent=4)

            # 当一个文件彻底训练完，重置 start_epoch，以便下一个文件从第 0 个 epoch 开始
            self.start_epoch = 0

            del train_loader, val_loader
            gc.collect()


if __name__ == "__main__":
    # --- 1. 初始化 DDP 环境 ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    logger = setup_logger(global_rank)
    if global_rank == 0:
        logger.info(f"DDP Initialized. World Size: {dist.get_world_size()}")

    # --- 2. 准备模型 ---
    set_encoder = SetEncoder(num_x_features=16)
    output_dir = '/fs0/home/zikaix/Data/zikaix/RetrievalSympfn/training_data'
    checkpoint_path = os.path.join(output_dir, "set_encoder_epoch_100.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        try:
            set_encoder.load_state_dict(state_dict)
            if global_rank == 0: logger.info("SetEncoder weights loaded.")
        except RuntimeError as e:
            if global_rank == 0: logger.error(f"Weight loading error: {e}")

    sympfn_model = SympfnModel(set_encoder=set_encoder, vocab_size=103)

    # --- 3. 配置 Trainer ---
    training_data_path = os.path.join(output_dir, "sympfn_top10_data.pt")
    data_dir = os.path.dirname(training_data_path)
    data_filename = os.path.basename(training_data_path)

    # [新增] 自动寻找最新的断点文件
    resume_ckpt_file = os.path.join(data_dir, "latest_sympfn_checkpoint.pth")
    if not os.path.exists(resume_ckpt_file):
        resume_ckpt_file = None  # 如果没有找到，就从零开始

    trainer = PFNTrainer(
        set_encoder=set_encoder,
        sympfn_model=sympfn_model,
        local_rank=local_rank,
        global_rank=global_rank,
        batch_size=16,
        learning_rate=1e-4,
        save_path=data_dir,
        val_ratio=0.1,
        resume_checkpoint=resume_ckpt_file  # <--- 传入断点路径
    )

    trainer.run_training_loop(train_files=[data_filename], epochs_per_file=50, logger=logger)

    # 销毁进程组
    dist.destroy_process_group()