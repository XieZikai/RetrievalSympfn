import torch
import torch.nn.functional as F
import faiss
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import faulthandler
faulthandler.enable()

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def build_database_and_generate_data(
        data_path_list,  # 本地 .pt 文件列表
        set_encoder,  # 预训练好的 SetEncoder
        output_dir,  # 保存目录
        device='cuda',  # 如果是 Mac，建议传入 'mps'
        batch_size=256,  # 推理和检索的 Batch Size
        k=10,  # 检索 Top-K
        temperature=0.1,  # 采样温度
        force_self_prob=0.25  # 25% 的概率强制使用自己作为 Context (Copy Task)
):
    """
    流程:
    1. 读取本地数据 -> 合并 x1/x2 作为全量数据库
    2. SetEncoder 计算 Embedding
    3. 建立 FAISS 索引
    4. 利用索引对自己进行检索 -> 生成 (Query, Context) 对
    5. 保存最终训练集 (包含 Token 序列)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Loading and Merging Data ---
    print(">>> 1. Loading and Merging Data from disk...")
    all_x, all_y, all_f = [], [], []

    for p in tqdm(data_path_list, desc="Loading files"):
        # 加载硬盘上的 .pt 文件
        data = torch.load(p, map_location='cpu')  # 先加载到 CPU 防止显存爆

        # 提取数据
        # 注意：我们将 dataset 中的 (x1, y1) 和 (x2, y2) 全部视为独立的样本加入库中
        # 这样可以最大化数据库的容量和多样性
        all_x.extend([data['x1'], data['x2']])
        all_y.extend([data['y1'], data['y2']])

        # 这里 f1_tokens 和 f2_tokens 已经是 [B, Max_Len] 的 LongTensor
        all_f.extend([data['f1_tokens'], data['f2_tokens']])

    # 拼接成巨大的 Tensor
    # cat: [B1, B2, ...] -> [Total_N, ...]
    full_x = torch.cat(all_x, dim=0).float()  # 强制 float32 兼容 MPS
    full_y = torch.cat(all_y, dim=0).float()
    full_f = torch.cat(all_f, dim=0).long()  # 强制 long

    total_samples = full_x.shape[0]
    print(f"Total samples to index: {total_samples}")
    print(f"Formula Token Shape: {full_f.shape}")  # 应该是 [Total, 103]

    # --- 2. Computing Embeddings ---
    print(">>> 2. Computing Embeddings...")
    # 使用 Dataset 封装以利用 DataLoader
    dataset = TensorDataset(full_x, full_y)

    # Mac 上 num_workers=0 可能更稳定，Linux 可设为 4
    num_workers = 0 if device == 'mps' else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    set_encoder.eval()
    set_encoder.to(device)

    embeddings_list = []

    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Encoding"):
            bx, by = bx.to(device), by.to(device)

            # SetEncoder 只看 (x, y)，不看公式 token
            raw_emb = set_encoder(bx, by)

            # [重要] L2 归一化！
            # FAISS 使用 Inner Product (IP) 索引
            # Normalized Vector Dot Product == Cosine Similarity
            emb = F.normalize(raw_emb, p=2, dim=1)

            embeddings_list.append(emb.cpu().numpy())

    # 拼接 Embedding 矩阵
    # Shape: (Total_Samples, Hidden_Dim)
    embedding_matrix = np.concatenate(embeddings_list, axis=0)
    dimension = embedding_matrix.shape[1]

    # --- 3. Building FAISS Index ---
    print(">>> 3. Building FAISS Index...")
    # 使用内积索引 (IndexFlatIP)，对于归一化向量等价于余弦相似度
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    print(f"Index built. Total vectors: {index.ntotal}")

    # --- 4. Generating Contexts (Self-Retrieval) ---
    print(f">>> 4. Generating Contexts (p_self={force_self_prob}, temp={temperature})...")

    all_selected_ids = []

    # 批量检索
    num_batches = (total_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Retrieving"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        batch_emb = embedding_matrix[start_idx:end_idx]  # numpy
        curr_batch_size = batch_emb.shape[0]

        # A. FAISS 搜索 Top-K
        scores, indices = index.search(batch_emb, k)

        # B. 概率采样
        scores_tensor = torch.tensor(scores)
        probs = F.softmax(scores_tensor / temperature, dim=1)
        sampled_idx_in_k = torch.multinomial(probs, num_samples=1)

        indices_tensor = torch.tensor(indices)
        selected_global_ids = torch.gather(indices_tensor, 1, sampled_idx_in_k).squeeze(1)

        # C. Force Self / Copy Mechanism
        force_mask = torch.rand(curr_batch_size) < force_self_prob
        if force_mask.any():
            self_ids = torch.arange(start_idx, end_idx)
            selected_global_ids[force_mask] = self_ids[force_mask]

        all_selected_ids.append(selected_global_ids)

    # 拼接所有 Context ID
    final_ctx_ids = torch.cat(all_selected_ids, dim=0).long()

    # --- 5. Assembling & Saving ---
    print(">>> 5. Assembling and Saving Dataset...")

    # 提取 Context 数据
    ctx_x = full_x[final_ctx_ids]
    ctx_y = full_y[final_ctx_ids]
    ctx_f = full_f[final_ctx_ids]  # 这里直接索引 Token ID 序列

    # 构建最终字典
    # Query: 原始数据 (full_*)
    # Context: 检索到的数据 (ctx_*)
    dataset_dict = {
        # Query (Input Task)
        'query_x': full_x,
        'query_y': full_y,
        'query_f': full_f,  # [Total, Seq_Len] 包含 SOS/EOS/PAD

        # Context (Retrieved Hint)
        'ctx_x': ctx_x,
        'ctx_y': ctx_y,
        'ctx_f': ctx_f,  # [Total, Seq_Len] 包含 SOS/EOS/PAD

        # Metadata (可选：用于后续保存 Database)
        # 'db_x': full_x,
        # 'db_y': full_y,
        # 'db_f': full_f
    }

    save_path = os.path.join(output_dir, "sympfn_training_data.pt")
    torch.save(dataset_dict, save_path)

    # 保存索引和元数据，以便推理时使用
    faiss.write_index(index, os.path.join(output_dir, "knn.index"))

    # 保存元数据供推理使用 (只存必要的)
    metadata = {
        'x': full_x,
        'y': full_y,
        'f_tokens': full_f
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    print(f"Done! Dataset saved to {save_path}")
    print(f"Shape Check -> Query F: {full_f.shape}, Context F: {ctx_f.shape}")


if __name__ == "__main__":
    from set_encoder import SetEncoder

    set_encoder = SetEncoder(num_x_features=16)
    device = 'cpu'
    checkpoint_path = "../training_data/set_encoder_epoch_10.pth"
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # 4. 将权重注入模型
    try:
        set_encoder.load_state_dict(state_dict)
        print("Success: Weights loaded.")
    except RuntimeError as e:
        print("Error: 加载失败，通常是因为模型结构参数不匹配。")
        print(f"详情: {e}")
    # 5. [非常重要] 切换到评估模式
    # 这会固定 BatchNorm 和 Dropout，保证推理结果稳定
    set_encoder.to(device)
    set_encoder.eval()

    data_list = os.listdir('../training_data/pregenerated_data')
    data_path_list = [
        os.path.join('../training_data/pregenerated_data', data)
        for data in data_list
        if data.endswith('.pt')  # [建议] 只读取 .pt 文件
    ]
    output_dir = '../training_data'

    build_database_and_generate_data(
        data_path_list,  # 本地 .pt 文件列表
        set_encoder,  # 预训练好的 SetEncoder
        output_dir,  # 保存目录
        device='cpu',  # 如果是 Mac，建议传入 'mps'
        batch_size=256,  # 推理和检索的 Batch Size
        k=10,  # 检索 Top-K
        temperature=0.1,  # 采样温度
        force_self_prob=0.25  # 25% 的概率强制使用自己作为 Context (Copy Task)
    )