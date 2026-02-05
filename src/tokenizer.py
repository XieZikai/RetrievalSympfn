import torch
from structure_canonicalizer import StructureCanonicalizer, ExpressionConverter

import torch


class FormulaTokenizer:
    def __init__(self, max_variables=16, max_length=103):
        # 1. 定义特殊 Token
        # <PAD>: 0
        # <SOS>: Start of Sentence
        # <EOS>: End of Sentence
        # <C>:   数值常数占位符 (如 1.5, -3.2)
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<C>']

        # 2. 定义数学常数 (根据你的策略，这些保留原名，不转为 <C>)
        self.math_constants = [
            'e', 'pi', 'euler_gamma'
        ]

        # 3. 定义算子集合 (全量覆盖 Generator 和 Converter)
        self.operators = [
            # 基础四则 & 幂
            'add', 'sub', 'mul', 'div', 'pow',

            # 一元负号 & 倒数 & 绝对值 & 符号
            'neg', 'inv', 'abs', 'sign',

            # 比较
            'max', 'min',

            # 指数 & 对数
            'exp', 'log', 'log2', 'log10',

            # 开方 & 特殊幂
            'sqrt', 'pow2', 'pow3',

            # 三角函数
            'sin', 'cos', 'tan',
            'asin', 'acos', 'atan',

            # 双曲函数
            'sinh', 'cosh', 'tanh',
            'asinh', 'acosh', 'atanh'
        ]

        # 4. 定义变量集合 (x_0 到 x_15)
        self.variables = [f'x_{i}' for i in range(max_variables)]

        # 5. 构建完整字典
        # 顺序：特殊 -> 常数 -> 算子 -> 变量
        self.id2token = self.special_tokens + self.math_constants + self.operators + self.variables
        self.token2id = {token: idx for idx, token in enumerate(self.id2token)}

        # 缓存关键 ID，方便快速调用
        self.pad_id = self.token2id['<PAD>']
        self.sos_id = self.token2id['<SOS>']
        self.eos_id = self.token2id['<EOS>']
        self.c_id = self.token2id['<C>']

        self.max_length = max_length

        print(f"Tokenizer initialized. Vocab Size: {len(self.id2token)}")
        # 打印一下验证 neg 是否在里面
        # print(f"Check: 'neg' ID is {self.token2id.get('neg')}, 'pi' ID is {self.token2id.get('pi')}")

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
                # 详细报错，帮助排查生成器是否产生了奇怪的符号
                valid_ops = ", ".join(self.operators)
                raise ValueError(
                    f"Unknown token encountered: '{token}'. \nExpected one of variables, constants, or: {valid_ops}")

        # 3. Add EOS
        ids.append(self.eos_id)

        # --- 长度控制 ---
        current_len = len(ids)

        if current_len > self.max_length:
            # Case A: 序列过长 -> 截断
            ids = ids[:self.max_length]
            ids[-1] = self.eos_id  # 保证最后一位是 EOS
        elif current_len < self.max_length:
            # Case B: 序列过短 -> 填充
            pad_len = self.max_length - current_len
            ids.extend([self.pad_id] * pad_len)

        return ids

    def encode_raw(self, token_list):
        ids = []
        for token in token_list:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                # 详细报错，帮助排查生成器是否产生了奇怪的符号
                valid_ops = ", ".join(self.operators)
                raise ValueError(
                    f"Unknown token encountered: '{token}'. \nExpected one of variables, constants, or: {valid_ops}")
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

