import torch
from structure_canonicalizer import StructureCanonicalizer, ExpressionConverter


class FormulaTokenizer:
    def __init__(self, max_variables=16, max_length=103):
        # 1. 定义特殊 Token
        # <PAD>: 0 (必须是0，方便 embedding padding_idx)
        # <SOS>: Start of Sentence
        # <EOS>: End of Sentence
        # <C>:   Constant placeholder
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<C>']

        # 2. 定义算子集合 (根据你的需求增减)
        self.operators = [
            'add', 'sub', 'mul', 'div',
            'pow', 'abs', 'max', 'min',
            'sin', 'cos', 'exp', 'log', 'sqrt', 'tan'
        ]

        # 3. 定义变量集合 (x_0 到 x_15)
        self.variables = [f'x_{i}' for i in range(max_variables)]

        # 4. 构建字典
        self.id2token = self.special_tokens + self.operators + self.variables
        self.token2id = {token: idx for idx, token in enumerate(self.id2token)}

        # 缓存关键 ID
        self.pad_id = self.token2id['<PAD>']
        self.sos_id = self.token2id['<SOS>']
        self.eos_id = self.token2id['<EOS>']
        self.c_id = self.token2id['<C>']

        self.max_length = max_length

        print(f"Tokenizer initialized. Vocab Size: {len(self.id2token)}")

    def encode(self, token_list):
        """
        将 token 列表转换为定长的 ID 列表
        Process: [SOS] + Tokens + [EOS] + [PAD]...
        """
        # 1. Start with SOS
        ids = [self.sos_id]

        # 2. Convert Tokens to IDs
        for token in token_list:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                # 遇到未知字符，直接报错，保证数据纯净
                raise ValueError(f"Unknown token encountered: {token}")

        # 3. Add EOS
        ids.append(self.eos_id)

        # --- 长度控制 (核心修改) ---
        current_len = len(ids)

        if current_len > self.max_length:
            # Case A: 序列过长 -> 截断
            # 策略：保留前 max_length - 1 个，强制把最后一位设为 EOS
            # 注意：截断会导致公式语法被破坏，但这是为了保证 tensor 形状的无奈之举
            ids = ids[:self.max_length]
            ids[-1] = self.eos_id
        elif current_len < self.max_length:
            # Case B: 序列过短 -> 填充
            # 计算需要补多少个 0
            pad_len = self.max_length - current_len
            ids.extend([self.pad_id] * pad_len)

        # 此时 len(ids) 必定等于 self.max_length
        return ids

    def decode(self, id_list):
        """
        解码：去除 SOS, PAD, 并在遇到 EOS 时停止
        """
        tokens = []
        for idx in id_list:
            # 兼容 Tensor 输入
            if hasattr(idx, 'item'):
                idx = idx.item()

            if idx == self.pad_id:  # 忽略 PAD
                continue
            if idx == self.sos_id:  # 忽略 SOS
                continue
            if idx == self.eos_id:  # 遇到 EOS 结束
                break

            if 0 <= idx < len(self.id2token):
                tokens.append(self.id2token[idx])
            else:
                tokens.append('<UNK>')
        return tokens

    def filter_dataset(self, dataset_iterator):
        """
        帮你筛选数据集的工具函数
        """
        valid_data = []
        for data in dataset_iterator:
            try:
                # 假设 data 是一个 token list
                # 尝试编码，如果不报错说明符合规范
                _ = self.encode(data)
                valid_data.append(data)
            except ValueError as e:
                # 记录被丢弃的数据，方便排查
                print(f"Dropped invalid data: {data} | Reason: {e}")
        return valid_data


if __name__ == "__main__":
    tokenizer = FormulaTokenizer()
    canon = StructureCanonicalizer()
    converter = ExpressionConverter()
    # l = ['add', '<C>', 'mul', 'cos', 'add', '<C>', 'mul', '<C>', 'x_0', 'pow', 'add', '<C>', 'mul', '<C>', 'x_1', '<C>']
    # print(tokenizer.encode(l))

    import os
    import pandas as pd

    folders = os.listdir("../sympfn_data")
    for folder in folders:
        if 'dataset' in folder:
            metadata = os.path.join("../sympfn_data", folder, 'metadata.csv')
            df = pd.read_csv(metadata)
            f = df['f'].iloc[0]
            result = converter.convert(f)
            print(f"转换结果 (Token List): {result}")
            print(f"\n转换为canonical：{canon.get_canonical_skeleton(result)}")
            print("=========")

