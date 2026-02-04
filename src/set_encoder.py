import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math


class SetEncoder(nn.Module):
    def __init__(self,
                 num_x_features,  # x 的原始特征维度
                 n_out=512,  # 输出 Embedding 维度 (也是内部 Hidden Dim)
                 nhead=4,
                 nhid=1024,
                 nlayers=6,
                 dropout=0.0,
                 activation='gelu',
                 pre_norm=False):
        super().__init__()

        self.ninp = n_out

        # --- 1. 独立的编码器 (TabPFN Style) ---
        # x_encoder: 把 x 从原始维度映射到 Hidden Dim
        self.x_encoder = nn.Sequential(
            nn.Linear(num_x_features, self.ninp, bias=False),
            nn.GELU(),
            nn.Linear(self.ninp, self.ninp)
        )

        # y_encoder: 把 y 从 1维 映射到 Hidden Dim
        self.y_encoder = nn.Linear(1, self.ninp)

        # --- 2. Transformer 主干 ---
        # batch_first=True: 输入形状 (Batch, N, Dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # 遍历 x_encoder 中的每一层
        if isinstance(self.x_encoder, nn.Sequential):
            for layer in self.x_encoder:
                # 只有当这一层是 Linear 层时，才初始化权重
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
        elif isinstance(self.x_encoder, nn.Linear):
            # 如果你改回单层 Linear，这部分代码兼容旧逻辑
            self.x_encoder.weight.data.uniform_(-initrange, initrange)
            self.x_encoder.bias.data.zero_()

        self.y_encoder.weight.data.uniform_(-initrange, initrange)
        self.y_encoder.bias.data.zero_()

    def forward(self, x, y, src_key_padding_mask=None):
        """
        x: (Batch, N, x_dim)
        y: (Batch, N) or (Batch, N, 1)
        src_key_padding_mask: (Batch, N) -> True 表示 Padding，对齐样本数量，代表样本点那个维度的attention mask，不去关注补的点

        训练时随机 Mask 掉一部分点，提升在小样本测试集上的鲁棒性

        for batch in dataloader:
            x, y = batch # (32, 100, ...)

            # 1. 随机生成有效长度
            # 让模型有时看100个点，有时只看20个点
            valid_lengths = torch.randint(low=20, high=101, size=(32,)) # [20, 100]

            # 2. 构造 Mask
            mask = torch.zeros(32, 100, dtype=torch.bool)
            for i in range(32):
                mask[i, valid_lengths[i]:] = True # 把超出长度的部分 Mask 掉

            # 3. 传入模型
            # Set Encoder 会忽略被 Mask 的部分
            # 这样模型就学会了：无论给我 20 个点还是 100 个点，我都能提取出正确的公式特征
            embedding = model(x, y, src_key_padding_mask=mask)
        """

        # --- A. Y-Normalization (Set-wise) ---
        # 这一步依然非常重要，保证进入 y_encoder 的数值分布是稳定的
        if y.dim() == 2:
            y = y.unsqueeze(-1)  # (Batch, N, 1)

        # --- B. 独立编码 (Separate Encoding) ---
        # 1. 编码 X
        # x_src: (Batch, N, Hidden_Dim)
        x_src = self.x_encoder(x)

        # 2. 编码 Y
        # y_src: (Batch, N, Hidden_Dim)
        y_src = self.y_encoder(y)

        # --- C. 特征融合 (TabPFN Style: Addition) ---
        # 这里的逻辑是：一个数据点是由它的位置(x)和它的值(y)共同定义的
        # 直接相加允许 Transformer 像处理 Positional Encoding 一样处理 Y 值
        # input_emb: (Batch, N, Hidden_Dim)
        input_emb = x_src + y_src

        # --- D. Transformer 交互 ---
        # output: (Batch, N, Hidden_Dim)
        output = self.transformer_encoder(input_emb, src_key_padding_mask=src_key_padding_mask)

        # --- E. 聚合 (Pooling) ---
        # 将 N 个点压缩成 1 个 Embedding
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(-1).float()
            output = output * (1.0 - mask)
            embedding = output.sum(dim=1) / (1.0 - mask).sum(dim=1).clamp(min=1)
        else:
            embedding = output.mean(dim=1)

        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (Seq_Len, Batch, Dim) if batch_first=False
        x: (Batch, Seq_Len, Dim) if batch_first=True
        """
        # 这里适配 batch_first=True
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class FormulaEncoder(nn.Module):
    def __init__(self,
                 vocab_size,  # 词表大小 (Token的总种类数)
                 n_out=512,  # 输出 Embedding 维度
                 max_len=103,  # 序列最大长度
                 pad_token_id=0,  # Padding 的 Token ID
                 nhead=4,
                 nhid=1024,
                 nlayers=6,
                 dropout=0.1,
                 activation='gelu',
                 pre_norm=False):
        super().__init__()

        self.ninp = n_out
        self.pad_token_id = pad_token_id

        # 1. Embedding 层
        # padding_idx=pad_token_id 会让该 index 的向量恒为 0，不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, self.ninp, padding_idx=pad_token_id)

        # 2. 位置编码 (必须有!)
        self.pos_encoder = PositionalEncoding(self.ninp, dropout, max_len=max_len)

        # 3. Transformer 主干
        encoder_layer = TransformerEncoderLayer(
            d_model=self.ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # 你的输入是 (Batch, Len)
            norm_first=pre_norm
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # 显式将 padding 的 embedding 归零 (虽然 padding_idx 应该已经做了)
        with torch.no_grad():
            self.embedding.weight[self.pad_token_id].fill_(0)

    def forward(self, f_tokens, src_key_padding_mask=None):
        """
        Args:
            f_tokens: (Batch, f_length) -> LongTensor (整数索引)
            src_key_padding_mask: (Batch, f_length) -> BoolTensor, True 表示 Padding

        Returns:
            embedding: (Batch, n_out)
        """
        # 1. Token Embedding
        # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        src = self.embedding(f_tokens) * math.sqrt(self.ninp)  # 缩放是 Transformer 的标准操作

        # 2. Add Positional Encoding
        src = self.pos_encoder(src)

        # 3. Transformer Encoding
        # Mask 必须传进去，防止 Attention 看到 Padding
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 4. Pooling (聚合)
        # 我们需要把 (Batch, Seq_Len, Dim) 变成 (Batch, Dim)

        if src_key_padding_mask is not None:
            # 这里的逻辑和 SetEncoder 一样：只对非 Padding 的 Token 求平均

            # mask: (Batch, Seq_Len, 1) -> True 为无效
            mask = src_key_padding_mask.unsqueeze(-1).float()
            valid_mask = 1.0 - mask

            # 因为 embedding 层设置了 padding_idx=0，且 bias=False (默认embedding没bias)
            # 所以 padding 位置的值理论上是 0。但经过 layer norm 和 transformer 计算后可能不是 0
            # 所以这里必须显式抹零
            output = output * valid_mask

            sum_output = output.sum(dim=1)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)

            embedding = sum_output / valid_counts
        else:
            # 如果没有 mask，直接平均
            embedding = output.mean(dim=1)

        return embedding


class SympfnModel(nn.Module):
    def __init__(self,
                 set_encoder,  # 预训练好的 SetEncoder 实例
                 vocab_size,  # 公式 Token 总数
                 n_out=512,  # Hidden Dim
                 max_len=103,  # 公式最大长度
                 pad_token_id=0,
                 nhead=4,
                 nhid=1024,
                 nlayers=6,  # Encoder 层数
                 decoder_layers=4,  # Decoder 层数 (生成公式用)
                 dropout=0.1,
                 activation='gelu',
                 pre_norm=False):
        super().__init__()

        self.ninp = n_out
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # --- 1. Set Encoder (Frozen) ---
        self.set_encoder = set_encoder
        # 冻结权重
        for param in self.set_encoder.parameters():
            param.requires_grad = False
        self.set_encoder.eval()  # 设为验证模式 (关闭 Dropout/BN 更新)

        # --- 2. Formula Encoder (用于处理 Context 中的 f_train) ---
        # 注意：这里我们复用你定义的 FormulaEncoder 逻辑，但可能不需要 pooling
        # 为了简单，我们这里假设 f_train 已经被压缩成了 (Batch, 512) 的向量
        # 如果 f_train 是原始 token，我们需要一个 Trainable Formula Encoder
        self.f_context_encoder = FormulaEncoder(
            vocab_size=vocab_size, n_out=n_out, max_len=max_len,
            pad_token_id=pad_token_id, nhead=nhead, nhid=nhid, nlayers=nlayers
        )

        # --- 3. Meta Transformer (Encoder) ---
        # 作用：阅读 Context 和 Query，提取特征
        encoder_layer = TransformerEncoderLayer(
            d_model=self.ninp, nhead=nhead, dim_feedforward=nhid,
            dropout=dropout, activation=activation, batch_first=True, norm_first=pre_norm
        )
        self.meta_transformer = TransformerEncoder(encoder_layer, num_layers=nlayers)

        # --- 4. Formula Decoder (Generator) ---
        # 作用：根据 Meta Transformer 的输出，生成公式
        # 这是一个标准的 GPT-style Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, n_out, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(n_out, dropout, max_len=max_len)

        decoder_layer = TransformerDecoderLayer(
            d_model=n_out, nhead=nhead, dim_feedforward=nhid,
            dropout=dropout, activation=activation, batch_first=True, norm_first=pre_norm
        )
        self.formula_decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # 输出层: 映射回词表大小
        self.output_head = nn.Linear(n_out, vocab_size)

    def _interleave(self, d_emb, f_emb):
        """
        交替拼接 Context
        d_emb: (Batch, K, Dim)
        f_emb: (Batch, K, Dim)
        Returns: (Batch, 2*K, Dim) -> [d1, f1, d2, f2...]
        """
        # stack: (Batch, K, 2, Dim)
        # flatten: (Batch, K*2, Dim)
        return torch.stack([d_emb, f_emb], dim=2).flatten(1, 2)

    def generate_square_subsequent_mask(self, sz):
        """生成因果遮罩 (Triangular Mask)"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                xs_test, ys_test, test_mask,  # Query Dataset
                d_train_embs,  # Context Dataset Embeddings (从数据库读出的) (Batch, K, 512)
                f_train_tokens, f_train_mask=None,  # Context Formula Tokens (Batch, K, Len)
                target_f_tokens=None  # 用于训练时 Teacher Forcing (Batch, Len)
                ):
        """
        Args:
            xs_test, ys_test: 测试集 (Query)
            d_train_embs: 检索到的 Top-K 训练集 Embedding
            f_train_tokens: 检索到的 Top-K 训练集对应的公式
            target_f_tokens: 目标公式 (仅训练时提供)
        """

        # 1. 编码 Query Dataset
        # set_encoder 是冻结的
        with torch.no_grad():
            # q_emb: (Batch, 512) -> (Batch, 1, 512)
            q_emb = self.set_encoder(xs_test, ys_test, src_key_padding_mask=test_mask).unsqueeze(1)

        # 2. 编码 Context Formula
        # 我们需要把 (Batch, K, Len) 的 token 变成 (Batch, K, 512) 的向量
        # 这是一个大的 Batch 处理
        batch_size, k, f_len = f_train_tokens.shape
        # flatten batch & k -> (Batch*K, Len)
        f_train_flat = f_train_tokens.view(-1, f_len)
        f_mask_flat = f_train_mask.view(-1, f_len) if f_train_mask is not None else None

        # f_emb_flat: (Batch*K, 512)
        f_emb_flat = self.f_context_encoder(f_train_flat, src_key_padding_mask=f_mask_flat)
        f_emb = f_emb_flat.view(batch_size, k, -1)  # (Batch, K, 512)

        # 3. 构造 Meta Sequence
        # Context: [d1, f1, d2, f2...]
        context_seq = self._interleave(d_train_embs, f_emb)  # (Batch, 2*K, 512)

        # Full Sequence: [Context, Query]
        # encoder_input: (Batch, 2*K + 1, 512)
        encoder_input = torch.cat([context_seq, q_emb], dim=1)

        # 4. Meta Transformer (Encoder)
        # memory: (Batch, Seq_Len, 512)
        # 这里不需要 Mask，因为所有 Context 对 Query 都是可见的
        memory = self.meta_transformer(encoder_input)

        # 我们只关心 Query 位置的输出，作为 Decoder 的 Condition
        # query_memory: (Batch, 1, 512)
        query_memory = memory[:, -1:, :]

        # --- 5. Decoder (生成公式) ---

        if target_f_tokens is not None:
            # === 训练模式 (Teacher Forcing) ===
            # 输入: <SOS> + f_tokens[:-1] (Shifted Right)
            tgt_input = target_f_tokens[:, :-1]
            tgt_seq_len = tgt_input.shape[1]

            # Embedding & PosEncoding
            tgt_emb = self.decoder_embedding(tgt_input) * math.sqrt(self.ninp)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Causal Mask (必须!)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input.device)

            # Decoder Forward
            # memory 传入 query_memory (作为 Cross Attention 的 Key/Value)
            # tgt 传入目标序列
            decoder_output = self.formula_decoder(tgt=tgt_emb, memory=query_memory, tgt_mask=tgt_mask)

            # Project to Vocab
            logits = self.output_head(decoder_output)  # (Batch, Len, Vocab)
            return logits

        else:
            # === 推断模式 (Autoregressive Generation) ===
            # 贪婪解码 (Greedy Decode) 示例
            curr_token = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long,
                                    device=q_emb.device)  # 假设 0 是 SOS
            # 实际应该用专门的 SOS_ID

            generated_tokens = []

            for i in range(self.max_len):
                tgt_emb = self.decoder_embedding(curr_token) * math.sqrt(self.ninp)
                tgt_emb = self.pos_encoder(tgt_emb)

                # Inference 时不需要 Mask，或者 Mask 也是全 1
                decoder_output = self.formula_decoder(tgt=tgt_emb, memory=query_memory)

                # 取最后一个 token 的输出
                next_token_logits = self.output_head(decoder_output[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_tokens.append(next_token)
                curr_token = torch.cat([curr_token, next_token], dim=1)

            return torch.cat(generated_tokens, dim=1)


def test_pipeline():
    print("=" * 20 + " 开始测试 " + "=" * 20)

    # --- 1. 定义超参数 ---
    BATCH_SIZE = 4
    N_SAMPLES = 100  # 每个数据集的样本数 (N)
    N_FEATURES = 100  # x 的特征维度
    N_OUT = 512  # Hidden Dim / Embedding Dim
    VOCAB_SIZE = 50  # 词表大小
    MAX_LEN = 103  # 公式最大长度
    CONTEXT_K = 3  # 检索到的上下文数量 (Top-K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # --- 2. 实例化模型 ---
    print("\n[1] 实例化模型...")
    set_encoder = SetEncoder(
        num_x_features=N_FEATURES,
        n_out=N_OUT,
        nhead=4,
        nhid=1024,
        nlayers=2  # 测试用，层数少点跑得快
    ).to(device)

    # 这里的 FormulaEncoder 是为了单独测试用的，SympfnModel 内部也会实例化一个
    f_encoder_test = FormulaEncoder(
        vocab_size=VOCAB_SIZE,
        n_out=N_OUT,
        max_len=MAX_LEN
    ).to(device)

    model = SympfnModel(
        set_encoder=set_encoder,
        vocab_size=VOCAB_SIZE,
        n_out=N_OUT,
        max_len=MAX_LEN,
        nlayers=2,  # Encoder 层数
        decoder_layers=2  # Decoder 层数
    ).to(device)

    print("模型实例化成功。SetEncoder 参数已冻结: ", not next(model.set_encoder.parameters()).requires_grad)

    # --- 3. 构造伪数据 ---
    print("\n[2] 构造伪数据...")

    # A. Query 数据 (测试集数据)
    # Shape: (Batch, N, Dim)
    query_xs = torch.randn(BATCH_SIZE, N_SAMPLES, N_FEATURES).to(device)
    query_ys = torch.randn(BATCH_SIZE, N_SAMPLES).to(device)
    # 假设没有 Padding，Mask 全为 False
    query_mask = torch.zeros(BATCH_SIZE, N_SAMPLES, dtype=torch.bool).to(device)

    # B. Context 数据 (检索到的 Top-K 训练数据)
    # 注意：TabPFN 接收的是训练数据的 Embedding，而不是原始 x,y
    # Shape: (Batch, K, Hidden_Dim)
    # 这里模拟从数据库检索出来的 D_train Embedding
    context_d_embs = torch.randn(BATCH_SIZE, CONTEXT_K, N_OUT).to(device)

    # C. Context 公式 (检索到的 Top-K 训练数据对应的公式)
    # Shape: (Batch, K, Max_Len)
    context_f_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_K, MAX_LEN)).to(device)
    # 假设没有 Padding
    context_f_mask = torch.zeros(BATCH_SIZE, CONTEXT_K, MAX_LEN, dtype=torch.bool).to(device)

    # D. Target 公式 (当前 Query 对应的真实公式，用于训练 Loss)
    # Shape: (Batch, Max_Len)
    target_f_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN)).to(device)

    # --- 4. 测试独立 Encoder ---
    print("\n[3] 测试独立组件...")

    # SetEncoder
    d_emb = set_encoder(query_xs, query_ys, src_key_padding_mask=query_mask)
    print(f"SetEncoder Output Shape: {d_emb.shape} (Expected: [{BATCH_SIZE}, {N_OUT}])")
    assert d_emb.shape == (BATCH_SIZE, N_OUT), "SetEncoder shape mismatch!"

    # FormulaEncoder (模拟处理 Context 中的公式)
    # 把它 reshape 成 (Batch * K, Len)以此来测试
    flat_f = context_f_tokens.view(-1, MAX_LEN)
    f_emb = f_encoder_test(flat_f)
    print(f"FormulaEncoder Output Shape: {f_emb.shape} (Expected: [{BATCH_SIZE * CONTEXT_K}, {N_OUT}])")
    assert f_emb.shape == (BATCH_SIZE * CONTEXT_K, N_OUT), "FormulaEncoder shape mismatch!"

    # --- 5. 测试 TabPFN Training Forward (Teacher Forcing) ---
    print("\n[4] 测试 SympfnModel Training Forward...")

    # 传入 target_f_tokens，激活 Decoder 的并行训练模式
    logits = model(
        xs_test=query_xs,
        ys_test=query_ys,
        test_mask=query_mask,
        d_train_embs=context_d_embs,
        f_train_tokens=context_f_tokens,
        f_train_mask=context_f_mask,  # 传入 mask
        target_f_tokens=target_f_tokens
    )

    # 输出应该是 (Batch, Max_Len - 1, Vocab_Size) 或者 (Batch, Max_Len, Vocab_Size)
    # 取决于 Decoder 里的 shift 操作。通常输出长度和 target 长度一致。
    print(f"Training Logits Shape: {logits.shape} (Expected: [{BATCH_SIZE}, {MAX_LEN - 1}, {VOCAB_SIZE}])")

    # 检查 Loss 计算
    # Flatten logits and targets
    # logits: (B, L-1, V), target: (B, L) -> shift target: target[:, 1:]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), target_f_tokens[:, 1:].reshape(-1))
    print(f"Training Loss: {loss.item()}")

    assert not torch.isnan(loss), "Training loss is NaN!"

    # --- 6. 测试 TabPFN Inference (Autoregressive Generation) ---
    print("\n[5] 测试 SympfnModel Inference (Generation)...")

    # 不传入 target_f_tokens，激活自回归生成模式
    with torch.no_grad():
        generated_tokens = model(
            xs_test=query_xs,
            ys_test=query_ys,
            test_mask=query_mask,
            d_train_embs=context_d_embs,
            f_train_tokens=context_f_tokens,
            f_train_mask=context_f_mask,
            target_f_tokens=None
        )

    print(f"Generated Tokens Shape: {generated_tokens.shape} (Expected: [{BATCH_SIZE}, {MAX_LEN}])")
    print(f"Sample Generated Sequence: {generated_tokens[0][:10].tolist()}...")

    assert generated_tokens.shape == (BATCH_SIZE, MAX_LEN), "Inference shape mismatch!"

    print("\n" + "=" * 20 + " 测试全部通过 " + "=" * 20)


if __name__ == "__main__":
    # 确保你的类定义在同一个文件中，或者已经 import 进来
    test_pipeline()