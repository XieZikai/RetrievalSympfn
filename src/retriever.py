import numpy as np
import torch
from torch.nn import functional as F
import os
from src.set_encoder import SetEncoder


##################用来加载x和y的函数##################
def load_xy(folder_path, device="cpu"):
    """
    只读取 x, y
    x: (N, Dx)
    y: (N, 1) or (N,)
    """
    x = np.load(os.path.join(folder_path, "x.npy"))
    y = np.load(os.path.join(folder_path, "y.npy"))

    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


class Retriever:
    def __init__(self, set_encoder, device="cpu", normalize_embedding=True):
        self.set_encoder = set_encoder.to(device)
        self.device = device

        self.normalize_embedding = normalize_embedding

        self.embeddings = None      # (N, D)
        self.dataset_ids = []       # list[str]



    ########################################
    # ===== 单个 embedding（no_grad）
    ########################################
    def get_embedding(self, x, y):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        elif y.dim() == 2:
            y = y.unsqueeze(0)

      
        emb = self.set_encoder(x, y)

        if self.normalize_embedding:
                emb = F.normalize(emb, dim=-1)

        return emb.squeeze(0)

    ########################################
    # ===== 批量 embedding
    ########################################
    def get_embedding_batch(self, x_batch, y_batch):
        """
        x_batch: (B, 100, 16)
        y_batch: (B, 100)
        """
        
        emb = self.set_encoder(x_batch, y_batch)
        if self.normalize_embedding:
                emb = F.normalize(emb, dim=-1)
        return emb

    ########################################
    # ===== 支持 batch_size
    ########################################
    def build_index(self, root_dir, batch_size=11):
        self.set_encoder.eval()

        subdirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        embeddings = []
        self.dataset_ids = []

        total = len(subdirs)
        print(f"📦 Building embeddings for {total} datasets")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_dirs = subdirs[start:end]

            x_list, y_list = [], []

            for dataset_id in batch_dirs:
                folder = os.path.join(root_dir, dataset_id)
                x, y = load_xy(folder, device=self.device)
                x_list.append(x)
                y_list.append(y)
                self.dataset_ids.append(dataset_id)

            x_batch = torch.stack(x_list, dim=0).to(self.device)
            y_batch = torch.stack(y_list, dim=0).to(self.device)

            emb_batch = self.get_embedding_batch(x_batch, y_batch)
            embeddings.append(emb_batch.cpu())

            print(f"  🚀 {end}/{total} processed")

        self.embeddings = torch.cat(embeddings, dim=0)
        print("✅ Embedding index built.")

#################保存总的embeddingsy以及相应的数据集名称##################
    def save_index(self, save_path):
        assert self.embeddings is not None, "Embeddings not built yet!"

        save_obj = {
        "embeddings": self.embeddings.cpu(),  # 🔴  🔴 🔴 🔴 🔴 🔴 🔴 🔴 🔴 🔴存 CPU
        "dataset_ids": self.dataset_ids,
        "normalize_embedding": self.normalize_embedding,
      }

        torch.save(save_obj, save_path)
        print(f"✅ Embedding index saved to: {save_path}")

###################如果已经有embeddings,则直接就加载##################
    def load_index(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device)

        self.embeddings = ckpt["embeddings"].to(self.device)
        self.dataset_ids = ckpt["dataset_ids"]
        self.normalize_embedding = ckpt.get(
        "normalize_embedding", True
        )

        print(
          f"✅ Loaded {self.embeddings.shape[0]} embeddings "
          f"of dim {self.embeddings.shape[1]}"
      )

######################选择前k个最近的dataset_id######################
    def retrieve_top_k_ids(self, x, y, k=10):
        """
        输入新 dataset (x, y)
        输出：最近的 k 个 dataset_id（子文件夹名）
        """
        # (D,)
        query_emb = self.get_embedding(x, y)

        # (1, D)
        query_emb = query_emb.unsqueeze(0)

        # squared L2 distance
        # embeddings: (N, D)
        dist = ((self.embeddings - query_emb) ** 2).sum(dim=1)

        _, topk_idx = torch.topk(dist, k, largest=False)

        topk_ids = [self.dataset_ids[i] for i in topk_idx.tolist()]
        return topk_ids


#####################智能准备 Retriever 的 embedding index#####################
def prepare_retriever_index(retriever, root_dir, index_path):
    """
    智能准备 Retriever 的 embedding index：
    - 如果已有 symtab_embedding_index.pt，直接加载
    - 否则遍历数据集计算 embedding 并保存
    """

    if os.path.exists(index_path):
        print(f"⚡ Found existing embedding index: {index_path}. Loading...")
        retriever.load_index(index_path)
    else:
        print(f"⚠️ No existing index found. Building embeddings from {root_dir} ...")
        retriever.build_index(root_dir)
        retriever.save_index(index_path)
        print(f"✅ Embedding index built and saved to: {index_path}")


# ============================================ 使用示例 ====================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #初始化set encoder和retriever
    set_encoder = SetEncoder(
        num_x_features=16,
        n_out=512,
        nhead=4,
        nhid=1024,
        nlayers=6,
        dropout=0.0,
    )

    retriever = Retriever(
        set_encoder,
        device=device,
        normalize_embedding=True,
    )

    root_dir = "/Users/zikaixie/PycharmProjects/TabPFN/sympfn_data" # 存储多个子数据集的根目录
    index_path = os.path.join(root_dir, "symtab_embedding_index.pt") # embedding index 保存路径

    # 智能加载 / 构建 embedding
    prepare_retriever_index(retriever, root_dir, index_path)

    # 新数据检索
    x_new, y_new = load_xy(
        "/Users/zikaixie/PycharmProjects/TabPFN/sympfn_data/dataset1_17",
        device="cpu"
    )

    nearest_ids = retriever.retrieve_top_k_ids(x_new, y_new, k=3)
    print(nearest_ids)