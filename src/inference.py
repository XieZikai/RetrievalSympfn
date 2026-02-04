import torch
import torch.nn.functional as F
import faiss
import os
import numpy as np
from tokenizer import FormulaTokenizer
from set_encoder import SetEncoder, SympfnModel


class SymbolicRegressor:
    def __init__(self,
                 set_encoder,
                 sympfn_model,
                 index_dir,
                 device='cuda',
                 max_len=103):
        """
        初始化推理引擎
        Args:
            set_encoder: 已加载权重的 SetEncoder 模型
            sympfn_model: 已加载权重的 SymPFN 模型
            index_dir: 包含 knn.index 和 metadata.pt 的目录
        """
        self.device = device
        self.max_len = max_len
        self.tokenizer = FormulaTokenizer()  # 假设使用默认参数

        # 1. 模型准备
        self.set_encoder = set_encoder.to(device).eval()
        self.sympfn = sympfn_model.to(device).eval()

        # 2. 加载检索库 (FAISS + Metadata)
        print(f"Loading retrieval database from {index_dir}...")
        self.index = faiss.read_index(os.path.join(index_dir, "knn.index"))
        self.metadata = torch.load(os.path.join(index_dir, "metadata.pt"), map_location='cpu')

        print("Symbolic Regressor Ready!")

    def predict(self, x, y, beam_width=1):
        """
        对给定的数据集进行符号回归
        Args:
            x: numpy array or tensor, shape (N, D) or (N,)
            y: numpy array or tensor, shape (N,)
        Returns:
            str: 预测的数学公式字符串 (Infix format, e.g., "x_0 + sin(x_1)")
        """
        # --- 1. 数据预处理 ---
        # 确保输入是 Batch 形式: [1, N, D]
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray): y = torch.from_numpy(y).float()

        if x.dim() == 1: x = x.unsqueeze(1)  # (N,) -> (N, 1)
        if x.dim() == 2: x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        if y.dim() == 1: y = y.unsqueeze(0)  # (N,) -> (1, N)

        x = x.to(self.device)
        y = y.to(self.device)

        # --- 2. 检索 Context (RAG) ---
        # 计算 Query Embedding
        with torch.no_grad():
            raw_emb = self.set_encoder(x, y)
            query_emb = F.normalize(raw_emb, p=2, dim=1).cpu().numpy()

        # FAISS 搜索 Top-1
        _, indices = self.index.search(query_emb, k=1)
        retrieved_id = indices[0][0]  # 全局 ID

        # 从 Metadata 获取 Context 数据
        # 注意：这里我们重新计算 context 的 embedding，确保流程一致
        ctx_x = self.metadata['x'][retrieved_id].unsqueeze(0).to(self.device)  # [1, N, D]
        ctx_y = self.metadata['y'][retrieved_id].unsqueeze(0).to(self.device)  # [1, N]
        ctx_f = self.metadata['f_tokens'][retrieved_id].unsqueeze(0).to(self.device)  # [1, L]

        # 计算 Context Embedding 给 SymPFN 用
        with torch.no_grad():
            ctx_raw_emb = self.set_encoder(ctx_x, ctx_y)
            ctx_emb = F.normalize(ctx_raw_emb, p=2, dim=1)  # [1, D]

            # 调整维度以适配 SymPFN: [Batch, K=1, Dim]
            d_train_embs = ctx_emb.unsqueeze(1)
            # 调整 tokens 维度: [Batch, K=1, Len]
            f_train_tokens = ctx_f.unsqueeze(1)

        # --- 3. 自回归生成 (Greedy Search) ---
        # 这一步是推断的核心：一个词一个词地生成
        generated_tokens = self._greedy_decode(
            x_query=x,
            y_query=y,
            d_train_embs=d_train_embs,
            f_train_tokens=f_train_tokens
        )

        # --- 4. 解码为字符串 ---
        # token ids -> list of strings (e.g., ['add', 'x_0', 'sin', ...])
        token_strs = self.tokenizer.decode(generated_tokens)

        # list -> Human Readable String
        # 这里你可以写一个简单的转换器把前缀转中缀，或者直接输出前缀
        return self._prefix_to_infix_simple(token_strs)

    def _greedy_decode(self, x_query, y_query, d_train_embs, f_train_tokens):
        """
        SymPFN 的自回归生成循环
        """
        batch_size = x_query.size(0)

        # 初始化输入: [SOS]
        # shape: [Batch, 1]
        input_seq = torch.full(
            (batch_size, 1),
            self.tokenizer.sos_id,
            dtype=torch.long,
            device=self.device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for _ in range(self.max_len):
                # Forward
                # 注意：target_f_tokens 是当前已生成的部分
                logits = self.sympfn(
                    xs_test=x_query,
                    ys_test=y_query,
                    test_mask=None,
                    d_train_embs=d_train_embs,
                    f_train_tokens=f_train_tokens,
                    f_train_mask=None,
                    target_f_tokens=input_seq
                )

                # 获取最后一个时间步的预测
                # logits: [Batch, Seq_Len, Vocab] -> 取最后一个: [Batch, Vocab]
                next_token_logits = logits[:, -1, :]

                # Greedy: 取概率最大的
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [Batch, 1]

                # 拼接到序列中
                input_seq = torch.cat([input_seq, next_token_id], dim=1)

                # 检查是否遇到 EOS (这里简化处理 Batch=1 的情况)
                if next_token_id.item() == self.tokenizer.eos_id:
                    break

        return input_seq[0].cpu().tolist()

    def _prefix_to_infix_simple(self, token_list):
        """
        简单的可视化辅助函数：将 token 列表转为字符串
        实际应用中建议使用之前的 StructureCanonicalizer 或专业的转换库
        """
        # 简单拼接返回，或者你可以接入你之前的 converter
        return " ".join(token_list)


if __name__ == "__main__":
    # 1. 实例化模型结构
    # 这里的参数必须和训练时完全一致
    set_encoder = SetEncoder(...)
    sympfn_model = SympfnModel(...)

    # 2. 加载训练好的权重
    set_encoder.load_state_dict(torch.load("set_encoder_final.pth"))
    sympfn_model.load_state_dict(torch.load("sympfn_final.pth"))

    # 3. 初始化推理器
    # index_dir 必须包含 knn.index 和 metadata.pt
    predictor = SymbolicRegressor(
        set_encoder=set_encoder,
        sympfn_model=sympfn_model,
        index_dir="./faiss_db_output"
    )

    # 4. 准备新数据 (测试数据)
    # 例如：y = 2 * x0 + 1
    new_x = np.random.uniform(-1, 1, (100, 16))  # 100个点，16个变量
    # 假设只有 x0 有用
    new_y = 2 * new_x[:, 0] + 1

    # 5. 预测
    formula_str = predictor.predict(new_x, new_y)

    print(f"Ground Truth: 2 * x_0 + 1")
    print(f"Predicted:    {formula_str}")