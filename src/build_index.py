import torch
import torch.nn.functional as F
import faiss
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def build_database_and_generate_data(
        data_path_list,  # 本地 .pt 文件列表
        set_encoder,  # 预训练好的 SetEncoder
        output_dir,  # 保存目录
        device='cuda',
        batch_size=256,  # 推理和检索的 Batch Size
        k=10,  # 检索 Top-K
        temperature=0.1,  # 采样温度
        force_self_prob=0.25  # 25% 的概率强制使用自己作为 Context (Copy Task)
):
    """
    流程:
    1. 读取本地数据 -> 合并
    2. SetEncoder 计算 Embedding
    3. 建立 FAISS 索引
    4. [新功能] 利用索引对自己进行检索 -> 生成 (Query, Context) 对
    5. 保存最终训练集
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Loading and Merging Data ---
    print(">>> 1. Loading and Merging Data from disk...")
    all_x, all_y, all_f = [], [], []

    for p in tqdm(data_path_list, desc="Loading files"):
        data = torch.load(p)
        # 拆分 x1/x2, y1/y2, f1/f2 并全部作为独立样本
        all_x.extend([data['x1'], data['x2']])
        all_y.extend([data['y1'], data['y2']])
        all_f.extend([data['f1_tokens'], data['f2_tokens']])

    # 拼接成巨大的 Tensor (CPU)
    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_f = torch.cat(all_f, dim=0)

    total_samples = full_x.shape[0]
    print(f"Total samples: {total_samples}")

    # --- 2. Computing Embeddings ---
    print(">>> 2. Computing Embeddings...")
    # 使用 Dataset 封装以利用 DataLoader
    dataset = TensorDataset(full_x, full_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    set_encoder.eval()
    set_encoder.to(device)

    embeddings_list = []

    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Encoding"):
            bx, by = bx.to(device), by.to(device)
            raw_emb = set_encoder(bx, by)
            # L2 归一化 (Cosine Similarity)
            emb = F.normalize(raw_emb, p=2, dim=1)
            embeddings_list.append(emb.cpu().numpy())

    # 拼接 Embedding 矩阵 (Total, Dim)
    embedding_matrix = np.concatenate(embeddings_list, axis=0)
    dimension = embedding_matrix.shape[1]

    # --- 3. Building FAISS Index ---
    print(">>> 3. Building FAISS Index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)
    print(f"Index built. Vectors: {index.ntotal}")

    # --- 4. Generating Contexts (Self-Retrieval) ---
    print(f">>> 4. Generating Contexts (p_self={force_self_prob}, temp={temperature})...")

    # 容器：存放检索到的 Context ID
    all_selected_ids = []

    # 批量检索，避免内存爆炸
    num_batches = (total_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Retrieving"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        # 获取当前 Batch 的 Embedding
        batch_emb = embedding_matrix[start_idx:end_idx]  # numpy array
        curr_batch_size = batch_emb.shape[0]

        # A. FAISS 搜索 Top-K
        # D: scores (Cosine Sim), I: Indices
        scores, indices = index.search(batch_emb, k)

        # B. 概率采样 (Softmax + Multinomial)
        scores_tensor = torch.tensor(scores)  # (B, K)
        probs = F.softmax(scores_tensor / temperature, dim=1)

        # 采样: 从 0~K-1 中选一个下标
        sampled_idx_in_k = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # 映射回全局 ID
        indices_tensor = torch.tensor(indices)  # (B, K)
        # Gather 选中的 ID
        selected_global_ids = torch.gather(indices_tensor, 1, sampled_idx_in_k).squeeze(1)  # (B,)

        # C. 应用混合策略 (Force Self / Copy Mechanism)
        # 生成掩码: True 表示强制使用自己
        force_mask = torch.rand(curr_batch_size) < force_self_prob

        if force_mask.any():
            # 当前 Batch 对应的真实全局 ID
            # 因为是顺序处理的，所以 ID 就是 range(start, end)
            self_ids = torch.arange(start_idx, end_idx)
            # 覆盖
            selected_global_ids[force_mask] = self_ids[force_mask]

        all_selected_ids.append(selected_global_ids)

    # 拼接所有选中的 Context ID
    final_ctx_ids = torch.cat(all_selected_ids, dim=0).long()

    # --- 5. Assembling & Saving ---
    print(">>> 5. Assembling and Saving Dataset...")

    # 利用 Advanced Indexing 直接从 full_x/y/f 中提取 Context
    # full_x 都在 CPU 上，可以直接索引
    ctx_x = full_x[final_ctx_ids]
    ctx_y = full_y[final_ctx_ids]
    ctx_f = full_f[final_ctx_ids]

    # 构建最终字典
    dataset_dict = {
        # Metadata (Database) - 如果你想保留纯净库供以后用
        'db_x': full_x,
        'db_y': full_y,
        'db_f': full_f,

        'ctx_x': ctx_x,  # Context 是检索+策略生成的数据
        'ctx_y': ctx_y,
        'ctx_f': ctx_f,
    }

    # 保存训练集
    save_path = os.path.join(output_dir, "sympfn_training_data.pt")
    torch.save(dataset_dict, save_path)

    # (可选) 保存 FAISS 索引以便将来做推断
    faiss.write_index(index, os.path.join(output_dir, "knn.index"))

    print(f"Done! Dataset saved to {save_path}")
    print(f"Shape Check -> Query: {full_x.shape}, Context: {ctx_x.shape}")


# --- 简单测试调用 ---
if __name__ == "__main__":
    # 假设
    # files = ["cycle_0.pt", "cycle_1.pt"]
    # encoder = ...
    # build_database_and_generate_data(files, encoder, "./output_db")
    pass