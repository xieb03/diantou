import sys

sys.path.append("../")
from project_utils import *

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy
import pandas as pd
import altair as alt
# spacy是一款工业级自然语言处理工具。spaCy 是一个在 Python 和 Cython 中用于高级自然语言处理的库。它基于最新的研究成果，并从一开始就设计成可用于实际产品。
# spaCy 提供了预训练的流程，并目前支持70多种语言的分词和训练。它具有最先进的速度和神经网络模型，用于标记、解析、命名实体识别、文本分类等任务，
# 还支持使用预训练的转换器（如BERT）进行多任务学习，以及生产就绪的训练系统、简单的模型打包、部署和工作流管理。spaCy 是商业开源软件，采用 MIT 许可证发布。
# noinspection PyUnresolvedReferences
# import GPUtil
import warnings


# http://nlp.seas.harvard.edu/annotated-transformer/
# https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # log(softmax(x)
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, n):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# The encoder is composed of a stack of N = 6 identical layers.
# We employ a residual connection (cite) around each of the two sub-layers, followed by layer normalization (cite).
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n=6):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
# We apply dropout (cite) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter 并将这个 parameter 绑定到这个 module 里面
        # net.parameter() 中就有这个绑定的 parameter，所以在参数优化的时候可以进行优化的，所以经过类型转换这个 self.v 变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ReslayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ReslayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


# Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
# and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_layer_1 = ReslayerConnection(size, dropout)
        self.res_layer_2 = ReslayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.res_layer_1(x, lambda _x: self.self_attn(_x, _x, _x, mask))
        return self.res_layer_2(x, self.feed_forward)


# The decoder is also composed of a stack of N = 6 identical layers.
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, n=6):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer,
# which performs multi-head attention over the output of the encoder stack.
# Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.res_layer_1 = ReslayerConnection(size, dropout)
        self.res_layer_2 = ReslayerConnection(size, dropout)
        self.res_layer_3 = ReslayerConnection(size, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.res_layer_1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.res_layer_2(x, lambda _x: self.src_attn(_x, m, m, src_mask))
        return self.res_layer_3(x, self.feed_forward)


# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
# This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i
# can depend only on the known outputs at positions less than i.
# tensor([[[0, 1, 1],
#          [0, 0, 1],
#          [0, 0, 0]]], dtype=torch.uint8)
# tensor([[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]])
# Q：为什么是 mask 前面而不是后面？
# A: 实际上是 mask 前面的，因为在 == 0 之后，上三角阵的下三角将全部为 True，这样外面再用 .type_as(src.data) 做数据转换的时候，会将 True 转化为 1，因此实际上出来的是下三角。即 mask 前面的
def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    # diagonal = 0，包含主对角线和其上三角的元素，其余元素为 0
    # diagonal = 1，仅包含主对角线其上三角的元素，其余元素为 0
    # 注意，如果是大于2 维度，也只是在最后两个维度保持上三角阵
    # uint8 节省内存
    subsequent_mask_ = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask_ == 0


# Below the attention mask shows the position each tgt word (row) is allowed to look at (column). Words are blocked for attending to future words during training.
def example_mask():
    # noinspection PyUnresolvedReferences
    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    # 不要加 .show()，会报错：altair_viewer._utils.NoMatchingVersions: No matches for version='5.16.3' among ['4.0.2', '4.8.1', '4.17.0'].
    return (
        alt.Chart(ls_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


# An attention function can be described as mapping a query and a set of key-value pairs to an output,
# where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values,
# where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension dk,
# and values of dimension dv. We compute the dot products of the query with all keys, divide each by dk^0.5, and apply a softmax function to obtain the weights on the values.
# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix K.
# The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意是在 score 上面增加 mask，因为后面会有 softmax，因此等于在对应的位置上面将输出设置为 0（因为设置填充值为 -1E9，即非常小的值），同时又几乎不改变其它的概率值（概率和保持为 1）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
# With a single attention head, averaging inhibits this.
# In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel / h = 64.
# Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.
# The Transformer uses multi-head attention in three different ways: previous decoder layer, and the memory keys and values come from the
# 1. In "encoder-decoder attention" layers, the queries come from the output of the encoder.
#   This allows every position in the decoder to attend over all positions in the input sequence.
#   This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as (cite).
#
# 2. The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place,
#   in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
#
# 3. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.
#   We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
#   We implement this inside of scaled dot-product attention by masking out (setting to ) all values in the input of the softmax which correspond to illegal connections.
# 注意这里并不是显式的拼接了 h = 8 个结果，而是利用 dmodel = dk * h 这样的大矩阵直接进行计算，省去了 for 循环
class MultiHeadedAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 注意是生成了 4 个，前 3 个用于 Q, K, V 的生成，最后一个用于结果
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # size 和 shape 一样，只不过一个是函数，另一个是属性，.size() = .shape
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


# Position-wise Feed-Forward Networks
# In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network,
# which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
# While the linear transformations are the same across different positions, they use different parameters from layer to layer.
# Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel=512, and the inner-layer has dimensionality dff=2048
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# Embeddings and Softmax
# Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel.
# We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
# In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation.
# In the embedding layers, we multiply those weights by dmodel^0.5
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Positional Encoding
# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence,
# we must inject some information about the relative or absolute position of the tokens in the sequence.
# To this end, we add “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks.
# The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed.
# There are many choices of positional encodings, learned and fixed.
# In this work, we use sine and cosine functions of different frequencies.
# In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of dropout=0.1.
# We also experimented with using learned positional embeddings (cite) instead, and found that the two versions produced nearly identical results.
# We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.
# todo：这里为什么用 dmodel 而不是 dk
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(max_len * 2) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 通过 register_buffer() 登记过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新。
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


# Here we define a function from hyperparameters to a full model.
def make_model(src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=5000):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout), n),
        decoder=Decoder(
            DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=dropout), n),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Here we make a forward step to generate a prediction of the model. We try to use our transformer to memorize the input.
# As you will see the output is randomly generated due to the fact that the model is not trained yet.
# In the next tutorial we will build the training function and try to train our model to memorize the numbers from 1 to 10.
# Example Untrained Model Prediction: tensor([[0, 3, 7, 5, 4, 3, 3, 3, 3, 3]])
# noinspection PyUnresolvedReferences
def inference_test():
    test_model = make_model(src_vocab=11, tgt_vocab=11, n=2)
    list_model_parameter_summary(test_model)
    print_model_parameter_summary(test_model)
    test_model.eval()

    # torch.int64
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    src_mask = torch.ones(1, 1, 10)

    # torch.Size([1, 10, 512])
    # src_mask 相当于没有用，因为没有 0 元素
    memory = test_model.encode(src, src_mask)
    # tensor([[0]])，初始值，后面每一次循环会递增一个值，例如会变成 tensor([[0, 3]])
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        print(str(i) + "-" * 80)
        # 第 0 次：tensor([[[1]]])
        # 第 1 次：tensor([[[1, 0],
        #                  [1, 1]]])
        # 第 2 次：tensor([[[1, 0, 0],
        #                  [1, 1, 0],
        #                  [1, 1, 1]]])
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)

        out = test_model.decode(memory, src_mask, ys, tgt_mask)
        # 第 0 次：torch.Size([1, 1, 512])
        # 第 1 次：torch.Size([1, 2, 512])
        # 第 2 次：torch.Size([1, 3, 512])
        print(out.shape)
        print(out)
        prob = test_model.generator(out[:, -1])
        print(prob)
        _, next_word = torch.max(prob, dim=1)
        print(torch.max(prob, dim=1))
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        print(ys)

    print("Example Untrained Model Prediction:", ys)


@func_timer(arg=True)
def main():
    fix_all_seed(_simple=False)
    # attn_shape = (1, 3, 3)
    # print(torch.ones(attn_shape))
    # print(torch.triu(torch.ones(attn_shape), diagonal=1))
    # print(torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8))
    # print(torch.ones(attn_shape).masked_fill(torch.triu(torch.ones(attn_shape), diagonal=1) == 0, -1e9))
    # print(torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0)
    # example_mask()
    inference_test()


if __name__ == '__main__':
    main()
