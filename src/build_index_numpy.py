import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import logging  # <--- 1. 导入
from logging import getLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("build_index_numpy.log", mode="w"),
        logging.StreamHandler()
    ])
logger = getLogger(__name__)

def build_database_and_generate_data_topk(
        data_path_list,
        set_encoder,
        output_dir,
        device='cpu',
        batch_size=256,
        k=10,  # Context 的长度 (Support Set Size)
        include_self_prob=0.5  # 50% 的概率包含 Query 自身，50% 概率强制排除自身
):
    """
    生成 Top-K Context 数据集。
    输出的 Context 维度将变为: [Total_Samples, k, Features]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Loading and Merging Data ---
    logger.info(">>> 1. Loading and Merging Data...")
    all_x, all_y, all_f_tokens = [], [], []

    for p in tqdm(data_path_list, desc="Loading"):
        data = torch.load(p, map_location='cpu')
        all_x.extend([data['x1'], data['x2']])
        all_y.extend([data['y1'], data['y2']])
        all_f_tokens.extend([data['f1_tokens'], data['f2_tokens']])

    full_x = torch.cat(all_x, dim=0).float()
    full_y = torch.cat(all_y, dim=0).float()
    full_f = torch.cat(all_f_tokens, dim=0).long()

    total_samples = full_x.shape[0]
    logger.info(f"Total samples: {total_samples}")

    # --- 2. Computing Embeddings ---
    logger.info(">>> 2. Computing Embeddings...")
    dataset = TensorDataset(full_x, full_y)
    # Mac M系列芯片建议 num_workers=0
    num_workers = 0 if device == 'mps' else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    set_encoder.eval()
    set_encoder.to(device)

    embeddings_list = []
    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Encoding"):
            bx, by = bx.to(device), by.to(device)
            raw_emb = set_encoder(bx, by)
            # L2 归一化，保证矩阵乘法等价于余弦相似度
            emb = F.normalize(raw_emb, p=2, dim=1)
            #embeddings_list.append(emb)  # 保持在 GPU/MPS 上
            embeddings_list.append(emb)  # #######################################⭐ 立刻搬回 CPU

    all_embeddings = torch.cat(embeddings_list, dim=0).to(device)  # [Total, Dim]

    # --- 3. Retrieval (Matrix Multiplication) ---
    # 我们需要检索 k+1 个邻居。
    # 因为第 1 个邻居通常是它自己 (Distance=0, Score=1.0)
    # 如果我们要“排除自己”，就需要取 [1:k+1]
    search_k = k + 1

    logger.info(f">>> 3. Retrieving Top-{k} Contexts (Search Window={search_k})...")

    all_context_indices = []
    num_batches = (total_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Retrieving"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)

            batch_query = all_embeddings[start_idx:end_idx]  # [B, Dim]
            curr_batch_size = batch_query.shape[0]

            # 计算相似度矩阵: [B, Total]
            # 注意：如果数据量特别大(>10w)，这里可能需要分块计算以防爆显存
            sim_scores = torch.matmul(batch_query, all_embeddings.T)

            # 获取 Top-(K+1) 的索引
            # indices shape: [B, K+1]
            # 第0列通常是它自己
            _, indices = torch.topk(sim_scores, k=search_k, dim=1)

            # --- 4. 概率性 Self-Inclusion 策略 ---
            # 生成随机掩码: True 代表“排除自己”，False 代表“包含自己”
            # exclude_mask: [B]
            exclude_mask = torch.rand(curr_batch_size, device=device) > include_self_prob

            # 扩展掩码以便进行 gather 或 where 操作: [B, k]
            exclude_mask_expanded = exclude_mask.unsqueeze(1).expand(-1, k)

            # 方案 A: 包含自己 -> 取索引 [0, 1, ..., k-1]
            indices_include = indices[:, 0:k]

            # 方案 B: 排除自己 -> 取索引 [1, 2, ..., k]
            indices_exclude = indices[:, 1:k + 1]

            # 根据掩码选择方案
            # 如果 mask 为 True (排除)，选 indices_exclude；否则选 indices_include
            final_batch_indices = torch.where(exclude_mask_expanded, indices_exclude, indices_include)

            all_context_indices.append(final_batch_indices.cpu())

    # 拼接最终的索引矩阵: [Total, k]
    final_ctx_ids = torch.cat(all_context_indices, dim=0).long()

    # --- 5. Assembling Dataset ---
    logger.info(">>> 5. Assembling Dataset with [Batch, k, Features] shape...")

    # PyTorch 的高级索引会自动处理额外的维度
    # final_ctx_ids 是 [N, k]
    # full_x 是 [N, Feat]
    # full_x[final_ctx_ids] -> [N, k, Feat]
    ctx_x = full_x[final_ctx_ids]
    ctx_y = full_y[final_ctx_ids]
    ctx_f = full_f[final_ctx_ids]

    dataset_dict = {
        # Query (Input Task): [N, Feat]
        'query_x': full_x,
        'query_y': full_y,
        'query_f': full_f,

        # Context (Support Set): [N, k, Feat]
        'ctx_x': ctx_x,
        'ctx_y': ctx_y,
        'ctx_f': ctx_f,
    }

    save_path = os.path.join(output_dir, f"sympfn_top{k}_data.pt")
    torch.save(dataset_dict, save_path)

    # 保存 Embeddings 供后续推理复用
    metadata = {
        'embeddings': all_embeddings.cpu(),
        'x': full_x,  # 原始数据库
        'y': full_y
    }
    torch.save(metadata, os.path.join(output_dir, "metadata_db.pt"))

    logger.info(f"Done! Saved to {save_path}")
    logger.info(f"Shape Check:")
    logger.info(f"  Query X: {full_x.shape}")  # [N, X_Feat]
    logger.info(f"  Context X: {ctx_x.shape}")  # [N, k, X_Feat]
    logger.info(f"  Context F: {ctx_f.shape}")  # [N, k, Seq_Len]


if __name__ == "__main__":
    from set_encoder import SetEncoder

    # 配置
    set_encoder = SetEncoder(num_x_features=16)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # 加载权重 (省略部分与之前相同...)
    # 假设你已经加载好了权重
    output_dir = '/fs0/home/zikaix/Data/zikaix/RetrievalSympfn/training_data1'

    data_path_list = [os.path.join(output_dir, 'pregenerated_data', f)
                      for f in os.listdir(os.path.join(output_dir, 'pregenerated_data')) if f.endswith('.pt')]

    build_database_and_generate_data_topk(
        data_path_list,
        set_encoder,
        output_dir=output_dir,
        device=device,
        batch_size=256,
        k=20,  # 你要求的 10 个 context
        include_self_prob=0.1  # 50% 概率包含自己
    )