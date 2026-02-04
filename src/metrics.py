import Levenshtein  # pip install Levenshtein


class FormulaDistanceMetric:
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: 你之前定义的 FormulaTokenizer 实例
        """
        self.tokenizer = tokenizer
        # 构建一个从 Token ID 到 Unicode 字符的映射
        # Levenshtein 库处理字符串极快，所以我们将每个 token_id 映射为一个唯一的字符
        # chr(0) 到 chr(vocab_size)
        self.id2char = {i: chr(i) for i in range(len(tokenizer.id2token) + 100)}  # +100 防止溢出

    def _seq_to_str(self, token_ids):
        """将 ID 列表转换为紧凑的字符串"""
        return "".join([self.id2char[tid] for tid in token_ids])

    def compute_distance(self, seq1, seq2):
        """
        计算原始编辑距离 (整数)
        seq1, seq2: Token List (strings) 或 Token ID List (ints)
        """
        # 如果输入是 Token 字符串列表，先转 ID
        if seq1 and isinstance(seq1[0], str):
            seq1 = self.tokenizer.encode(seq1)  # 假设 encode 返回 ID list
            seq2 = self.tokenizer.encode(seq2)

        # 移除 SOS/EOS/PAD 对结构比较的影响 (可选，但推荐)
        # 假设 tokenizer.encode 加上了 SOS(idx 1) 和 EOS(idx 2)
        # 我们这里简单切片去掉头尾，或者你可以手动 filter
        s1_str = self._seq_to_str(seq1)
        s2_str = self._seq_to_str(seq2)

        return Levenshtein.distance(s1_str, s2_str)

    def compute_similarity(self, seq1, seq2):
        """
        计算归一化相似度 [0, 1]
        1.0 表示完全一样
        """
        dist = self.compute_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))

        if max_len == 0:
            return 1.0

        return 1.0 - (dist / max_len)


# ================= 测试 =================
if __name__ == "__main__":
    # 假设你已经有了 tokenizer
    # from tokenizer import FormulaTokenizer
    # tokenizer = FormulaTokenizer()

    # 这里为了演示，mock 一个 tokenizer
    class MockTokenizer:
        def __init__(self):
            self.id2token = ['<PAD>', '<SOS>', '<EOS>', '<C>', 'add', 'mul', 'sin', 'x_0', 'x_1']
            self.token2id = {t: i for i, t in enumerate(self.id2token)}

        def encode(self, lst):
            return [self.token2id[t] for t in lst]


    tokenizer = MockTokenizer()
    metric = FormulaDistanceMetric(tokenizer)

    # Case 1: 细微差别 (x_0 vs x_1)
    f1 = ['add', '<C>', 'x_0']
    f2 = ['add', '<C>', 'x_1']
    sim = metric.compute_similarity(f1, f2)
    print(f"Case 1 (Change Var): Dist={metric.compute_distance(f1, f2)}, Sim={sim:.4f}")
    # Dist should be 1 (replace x_0 with x_1)

    # Case 2: 结构差异 (add vs mul)
    f3 = ['mul', '<C>', 'x_0']
    sim = metric.compute_similarity(f1, f3)
    print(f"Case 2 (Change Op):  Dist={metric.compute_distance(f1, f3)}, Sim={sim:.4f}")

    # Case 3: 复杂差异 (插入节点)
    # f1: C + x0
    # f4: C + sin(x0) -> ['add', '<C>', 'sin', 'x_0']
    f4 = ['add', '<C>', 'sin', 'x_0']
    sim = metric.compute_similarity(f1, f4)
    print(f"Case 3 (Insert Op):  Dist={metric.compute_distance(f1, f4)}, Sim={sim:.4f}")