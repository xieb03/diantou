import sys

sys.path.append("../")
from project_utils import *

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import altair as alt
# spacy是一款工业级自然语言处理工具。spaCy 是一个在 Python 和 Cython 中用于高级自然语言处理的库。它基于最新的研究成果，并从一开始就设计成可用于实际产品。
# spaCy 提供了预训练的流程，并目前支持70多种语言的分词和训练。它具有最先进的速度和神经网络模型，用于标记、解析、命名实体识别、文本分类等任务，
# 还支持使用预训练的转换器（如BERT）进行多任务学习，以及生产就绪的训练系统、简单的模型打包、部署和工作流管理。spaCy 是商业开源软件，采用 MIT 许可证发布。
import spacy
# Build a Vocab from an iterator.
from torchtext.vocab import build_vocab_from_iterator
# Convert iterable-style dataset to map-style dataset.
from torchtext.data.functional import to_map_style_dataset
# Sampler that restricts data loading to a subset of the dataset
from torch.utils.data.distributed import DistributedSampler
# Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
import torch.distributed as dist
# Implement distributed data parallelism based on ``torch.distributed`` at module level.
from torch.nn.parallel import DistributedDataParallel as Ddp
import GPUtil
import torch.multiprocessing as mp

# http://nlp.seas.harvard.edu/annotated-transformer/
# https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb
# https://zhuanlan.zhihu.com/p/559495068

"""
Part 1: Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure (cite). 
Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). 
Given z, the decoder then generates an output sequence (y1, ..., yn) of symbols one element at a time. 
At each step the model is auto-regressive (cite), consuming the previously generated symbols as additional input when generating the next.
"""


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
        return self.decode(memory=self.encode(src=src, src_mask=src_mask), src_mask=src_mask, tgt=tgt,
                           tgt_mask=tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# 最后一层，一个全连接，外加 log(softmax(x)) loss
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # log(softmax(x))
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
# Q: Transformer 为什么用 LayerNorm 不使用 BatchNorm？
# 为什么要归一化？
# 使数据的分布稳定，降低数据的方差。将数据化成均值为 0 方差为 1，防止落入激活函数饱和区，训练过程平稳
# 为什么不用BatchNorm？
# 不同归一化方法操作的维度不同，对于输入[N, C, H, W]维度的图片：
#   BatchNorm 在 C 维度上，计算 (N, H, W) 的统计量，拉平各个 C 里面的差异。
#   LayerNorm 在 N 维度上，计算 (C, H, W) 的统计量，拉平各个 N 里面的差异。
# 注意，这个图只是在CV中的例子，在NLP中，LayerNorm的操作对象是：
#   对于输入 [N, L, E] 维度的文本（Batch size, seq len, embedding size）
#   计算 (E) 的统计量，而不是（L, E）
# 从数据的角度解释：CV 通常用 BatchNorm，NLP 通常用 LayerNorm。图像数据一个 Channel 内的关联性比较大，不同 Channel 的信息需要保持差异性。文本数据一个 Batch 内的不同样本关联性不大。
# 从Pad的角度解释：不同句子的长度不同，在句子的末尾归一化会受到 pad 的影响，使得统计量不置信。
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter 并将这个 parameter 绑定到这个 module 里面
        # net.parameter() 中就有这个绑定的 parameter，所以在参数优化的时候可以进行优化的，所以经过类型转换这个 self.v 变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断地修改其值以达到最优化。
        # torch.Size([512])
        self.a_2 = nn.Parameter(torch.ones(features))
        # torch.Size([512])
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# PreNorm 和 PostNorm 的区别，为什么 PreNorm 最终效果不如 PostNorm？
# preNorm: x_t+1 = x_t + Ft(Norm(x_t))
# PostNorm: x_t+1 = Norm(x_t + Ft((x_t))
# 目前比较明确的结论是：同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。Pre Norm更容易训练好理解，因为它的恒等路径更突出，但为什么它效果反而没那么好呢？
# 另外，笔者自己也做过对比实验，显示Post Norm的结构迁移性能更加好，也就是说在Pretraining中，Pre Norm和Post Norm都能做到大致相同的结果，但是Post Norm的Finetune效果明显更好。
# 可能读者会反问《On Layer Normalization in the Transformer Architecture》不是显示Pre Norm要好于Post Norm吗？这是不是矛盾了？
# 其实这篇文章比较的是在完全相同的训练设置下Pre Norm的效果要优于Post Norm，这只能显示出Pre Norm更容易训练，因为Post Norm要达到自己的最优效果，不能用跟Pre Norm一样的训练配置（比如Pre Norm可以不加Warmup但Post Norm通常要加），所以结论并不矛盾。
# 说白了，Pre Norm 结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。
# 而 Post Norm 刚刚相反，它每 Norm 一次就削弱一次恒等分支的权重，所以 Post Norm 反而是更突出残差分支的，因此 Post Norm 中的层数更加 “足秤”，一旦训练好之后效果更优。
# 前段时间号称能训练1000层Transformer的DeepNet想必不少读者都听说过，在其论文《DeepNet: Scaling Transformers to 1,000 Layers》中对Pre Norm的描述是：
# However, the gradients of Pre-LN at bottom layers tend to be larger than at top layers, leading to a degradation in performance compared with Post-LN.
# 不少读者当时可能并不理解这段话的逻辑关系，但看了前一节内容的解释后，想必会有新的理解。
# 简单来说，所谓“the gradients of Pre-LN at bottom layers tend to be larger than at top layers”，就是指Pre Norm结构会过度倾向于恒等分支（bottom layers），
# 从而使得Pre Norm倾向于退化（degradation）为一个“浅而宽”的模型，最终不如同一深度的Post Norm。这跟前面的直观理解本质上是一致的。
# Q: Transformer如何缓解梯度消失？
# 什么是梯度消失？
# 它指的是（主要是在模型的初始阶段）越靠近输入的层梯度越小，趋于零甚至等于零，而我们主要用的是基于梯度的优化器，所以梯度消失意味着我们没有很好的信号去调整优化前面的层。
# 换句话说，前面的层也许几乎没有得到更新，一直保持随机初始化的状态；只有比较靠近输出的层才更新得比较好，但这些层的输入是前面没有更新好的层的输出，所以输入质量可能会很糟糕（因为经过了一个近乎随机的变换），
# 因此哪怕后面的层更新好了，总体效果也不好。最终，我们会观察到很反直觉的现象：模型越深，效果越差，哪怕训练集都如此。
# 如何解决梯度消失？
# 解决梯度消失的一个标准方法就是残差链接，正式提出于 ResNet 中。残差的思想非常简单直接：你不是担心输入的梯度会消失吗？那我直接给它补上一个梯度为常数的项不就行了？最简单地，将模型变成 y=x+F (x)
# 这样一来，由于多了一条“直通”路 x ，就算 F(x) 中的 x 梯度消失了，x 的梯度基本上也能得以保留，从而使得深层模型得到有效的训练。
# LayerNorm 能缓解梯度消失吗？
# 在 BERT 和 Transformer 里边，使用的是 Post Norm 设计，它把 Norm 操作加在了残差之后：
# 我们知道，残差有利于解决梯度消失，但是在 Post Norm 中，残差这条通道被严重削弱了，越靠近输入，削弱得越严重，残差 “名存实亡”。所以说，在 Post Norm 的 BERT 模型中，LN 不仅不能缓解梯度消失，它还是梯度消失的 “元凶” 之一。
# 虽然Post Norm 会带来一定的梯度消失问题，但其实它也有其他方面的好处。最明显的是，它稳定了前向传播的数值，并且保持了每个模块的一致性。其次，梯度消失也不全是 “坏处”，其实对于 Finetune 阶段来说，它反而是好处。
# 在 Finetune 的时候，我们通常希望优先调整靠近输出层的参数，不要过度调整靠近输入层的参数，以免严重破坏预训练效果。而梯度消失意味着越靠近输入层，其结果对最终输出的影响越弱，这正好是 Finetune 时所希望的。
# 所以，预训练好的 Post Norm 模型，往往比 Pre Norm 模型有更好的 Finetune 效果。
# Adam如何解决梯度消失？
# 其实，最关键的原因是，在当前的各种自适应优化技术下，我们已经不大担心梯度消失问题了。
# 这是因为，当前 NLP 中主流的优化器是 Adam 及其变种。对于 Adam 来说，由于包含了动量和二阶矩校正，所以近似来看，它的更新量大致上为
# 可以看到，分子分母是都是同量纲的，因此分式结果其实就是 (1)的量级，而更新量就是 (η)量级。也就是说，理论上只要梯度的绝对值大于随机误差，那么对应的参数都会有常数量级的更新量；
# 这跟 SGD 不一样，SGD 的更新量是正比于梯度的，只要梯度小，更新量也会很小，如果梯度过小，那么参数几乎会没被更新。
# 所以，Post Norm 的残差虽然被严重削弱，但是在 base、large 级别的模型中，它还不至于削弱到小于随机误差的地步，因此配合 Adam 等优化器，它还是可以得到有效更新的，也就有可能成功训练了。
# 当然，只是有可能，事实上越深的 Post Norm 模型确实越难训练，比如要仔细调节学习率和 Warmup 等。
# Warmup 如何解决梯度消失？
# Warmup 是在训练开始阶段，将学习率从 0 缓增到指定大小，而不是一开始从指定大小训练。如果不进行 Wamrup，那么模型一开始就快速地学习，由于梯度消失，模型对越靠后的层越敏感，也就是越靠后的层学习得越快，然后后面的层是以前面的层的输出为输入的，前面的层根本就没学好，所以后面的层虽然学得快，但却是建立在糟糕的输入基础上的。
# 很快地，后面的层以糟糕的输入为基础到达了一个糟糕的局部最优点，此时它的学习开始放缓（因为已经到达了它认为的最优点附近），同时反向传播给前面层的梯度信号进一步变弱，这就导致了前面的层的梯度变得不准。但 Adam 的更新量是常数量级的，梯度不准，但更新量依然是常数量级，意味着可能就是一个常数量级的随机噪声了，于是学习方向开始不合理，前面的输出开始崩盘，导致后面的层也一并崩盘。
# 所以，如果 Post Norm 结构的模型不进行 Wamrup，我们能观察到的现象往往是：loss 快速收敛到一个常数附近，然后再训练一段时间，loss 开始发散，直至 NAN。如果进行 Wamrup，那么留给模型足够多的时间进行 “预热”，在这个过程中，主要是抑制了后面的层的学习速度，并且给了前面的层更多的优化时间，以促进每个层的同步优化。
# 这里的讨论前提是梯度消失，如果是 Pre Norm 之类的结果，没有明显的梯度消失现象，那么不加 Warmup 往往也可以成功训练。
# Q: BERT 权重初始标准差为什么是0.02？
# 喜欢扣细节的同学会留意到，BERT 默认的初始化方法是标准差为 0.02 的截断正态分布，由于是截断正态分布，所以实际标准差会更小，大约是 0.02/1.1368472≈0.0176。这个标准差是大还是小呢？对于 Xavier 初始化来说，一个 n×n 的矩阵应该用 1/n 的方差初始化，而 BERT base 的 n 为 768，算出来的标准差是$1/\sqrt{768}≈0.0361$。这就意味着，这个初始化标准差是明显偏小的，大约只有常见初始化标准差的一半。
# 为什么 BERT 要用偏小的标准差初始化呢？事实上，这还是跟 Post Norm 设计有关，偏小的标准差会导致函数的输出整体偏小，从而使得 Post Norm 设计在初始化阶段更接近于恒等函数，从而更利于优化。具体来说，按照前面的假设，如果 x 的方差是 1，F (x) 的方差是$σ^2$，那么初始化阶段，Norm 操作就相当于除以$\sqrt{1+σ^2}$。如果 σ 比较小，那么残差中的 “直路” 权重就越接近于 1，那么模型初始阶段就越接近一个恒等函数，就越不容易梯度消失。
# 正所谓 “我们不怕梯度消失，但我们也不希望梯度消失”，简单地将初始化标注差设小一点，就可以使得 σ 变小一点，从而在保持 Post Norm 的同时缓解一下梯度消失，何乐而不为？那能不能设置得更小甚至全零？一般来说初始化过小会丧失多样性，缩小了模型的试错空间，也会带来负面效果。综合来看，缩小到标准的 1/2，是一个比较靠谱的选择了。
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
        # torch.Size([1, 1, 512])
        # torch.Size([1, 2, 512])
        # torch.Size([1, 3, 512])
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

    def forward(self, x, src_mask):
        """Follow Figure 1 (left) for connections."""
        # torch.Size([1, 10, 512])
        x = self.res_layer_1(x, lambda _x: self.self_attn(_x, _x, _x, src_mask))
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
        x = self.res_layer_1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        # 注意这里有一个 decoder 和 encoder 的交叉多头注意力，这里的顺序仿照 q = x, k = v = memory，与上面 q = k = v = x 有所区别
        x = self.res_layer_2(x, lambda _x: self.src_attn(_x, memory, memory, src_mask))
        return self.res_layer_3(x, self.feed_forward)


# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
# This masking, combined with fact that the output embeddings are offset by one position,
# ensures that the predictions for position ith can depend only on the known outputs at positions less than ith.
# tensor([[[0, 1, 1],
#          [0, 0, 1],
#          [0, 0, 0]]], dtype=torch.uint8)
# tensor([[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]])
# Q：为什么是 mask 前面而不是后面？
# A: 实际上是 mask 前面的，但最终的效果是 mask 后面的。因为在 == 0 之后，上三角阵的下三角将全部为 True，这样外面再用 .type_as(src.data) 做数据转换的时候，会将 True 转化为 1，因此实际上出来的是下三角。
#    而最终调用 masked_fill(mask == 0, -1e9) 的时候，会把 0 对应的位置 mask 掉，因此会把后面的 mask 掉
# 注意是个方阵，例如输出是 1 * n，那么实际上 mask 是 n * n，因为对于每一位的输出，都要有一个 n 维的 mask 序列
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

    pivot_data = ls_data.pivot(index='Masking', columns='Window')['Subsequent Mask']
    print(pivot_data)
    # 数据的 index 和 columns 分别为 heatmap 的 y 轴方向和 x 轴方向标签
    sns.heatmap(pivot_data, annot=True, fmt="d", cmap='viridis')
    plt.show()

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
# We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension d_k,
# and values of dimension d_v. We compute the dot products of the query with all keys, divide each by d_k^0.5, and apply a softmax function to obtain the weights on the values.
# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix K.
# The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:
# Q: 为什么在进行 softmax 之前需要除以 d_k^0.5
# 论文中的解释是：向量的点积结果会很大，将 softmax 函数 push 到梯度很小的区域，scaled 会缓解这种现象。
# 结论：假设向量 q 和 k 的各个分量 qi, ki 是互相独立的随机变量，均值是 0，方差是 1，那么 q · k 的均值是 0 方差是 d_k。除以 d_k^0.5 可以将方差恢复为1。
# softmax+交叉熵会让输入过大/过小的梯度消失吗？
# 不会。因为交叉熵有一个log。log_softmax的梯度和刚才算出来的不同，就算输入的某一个x过大也不会梯度消失。
# softmax+MSE会有什么问题？为什么我们在分类的时候不使用MSE作为损失函数？
# 刚才的解释就可以说明这个问题。因为MSE中没有log，所以softmax+MSE会造成梯度消失。
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意是在 score 上面增加 mask，因为后面会有 softmax，因此等于在对应的位置上面将输出设置为 0（因为设置填充值为 -1E9，即非常小的值），同时又几乎不改变其它的概率值（概率和保持为 1）
    # 最终 scores.dim = mask.dim
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        assert_equal(scores.dim(), mask.dim())
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
# With a single attention head, averaging inhibits this.
# In this work we employ h = 8 parallel attention layers, or heads. For each of these we use d_k = d_v = d_model / h = 64.
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
# 注意这里并不是显式的拼接了 h = 8 个结果，而是利用 d_k * h 这样的大矩阵直接进行计算，省去了 for 循环，而为了方便，一般直接另 d_k = d_model / h，其中 d_model 是 embedding 的维度，d_k 是 Q/K/V 的维度
# 为什么 Transformer 需要进行 Multi-head Attention？
# 实验证明多头是必要的，8/16个头都可以取得更好的效果，但是超过16个反而效果不好。每个头关注的信息不同，但是头之间的差异随着层数增加而减少。并且不是所有头都有用，有工作尝试剪枝，可以得到更好的表现。
# Transformer 为什么 Q 和 K 使用不同的权重矩阵生成？
# 可以理解为是在不同空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的 attention score 矩阵的泛化能力更高。
# 如果使用相同的 W，attention score 会退化为一个近似对角矩阵。因为在 softmax 之前，对角线上的元素都是通过自身点乘自身得到的，是许多正数的和，所以会非常大，而其他元素有正有负，没有这么大。经过 softmax 之后，对角线上的元素会接近 1。
class MultiHeadedAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 注意是生成了 4 个，前 3 个用于 Q, K, V 的生成，最后一个用于结果
        # torch.Size([512, 512])
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
        # torch.Size([1, 8, 1, 64])
        # torch.Size([1, 8, 2, 64])
        # torch.Size([1, 8, 3, 64])
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # torch.Size([1, 8, 1, 64])
        # torch.Size([1, 8, 2, 64])
        # torch.Size([1, 8, 3, 64])
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        # torch.Size([1, 1, 512])
        # torch.Size([1, 2, 512])
        # torch.Size([1, 3, 512])
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
# Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is d_model=512, and the inner-layer has dimensionality dff=2048
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
# Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.
# We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
# In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation.
# In the embedding layers, we multiply those weights by d_model^0.5
# 为什么要乘以 d_model^0.5
# 如果使用 Xavier 初始化，Embedding 的方差为 1/d_model，当d_model非常大时，矩阵中的每一个值都会减小。通过乘一个 d_model^0.5 可以将方差恢复到1。
# 因为Position Encoding是通过三角函数算出来的，值域为[-1, 1]。所以当加上 Position Encoding 时，需要放大 embedding 的数值，否则规模不一致相加后会丢失信息。
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 11 * 512
        self.lut = nn.Embedding(vocab, d_model)
        # 512
        self.d_model = d_model

    def forward(self, x):
        # torch.Size([1, 1])
        # torch.Size([1, 2])
        # torch.Size([1, 3])
        # print(x.shape)
        # torch.Size([1, 1, 512])
        # torch.Size([1, 2, 512])
        # torch.Size([1, 3, 512])
        return self.lut(x) * math.sqrt(self.d_model)


# Positional Encoding
# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence,
# we must inject some information about the relative or absolute position of the tokens in the sequence.
# To this end, we add “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks.
# The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.
# There are many choices of positional encodings, learned and fixed.
# In this work, we use sine and cosine functions of different frequencies.
# In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of dropout=0.1.
# We also experimented with using learned positional embeddings (cite) instead, and found that the two versions produced nearly identical results.
# We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.
# 注意，d_model 才是 embedding 的维度，而 d_k 只是注意力机制中 Q/K/V 的维度，只是一般为了简化，我们令 d_k = d_model / h，即 h 个头恰好组成 d_model 维度
# 这样处理，在注意力机制中，我们实际上是用 d_k * h 大小的矩阵来直接运算，而不是分别运算 h 个 d_k，再进行 concat
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # torch.Size([5000, 1])
        # tensor([[   0],
        #         [   1],
        #         [   2],
        #         ...,
        #         [4997],
        #         [4998],
        #         [4999]])
        position = torch.arange(0, max_len).unsqueeze(1)
        # torch.Size([256]), 1 / 10000^(2i / 512)
        # tensor([1.0000e+00, 3.9811e-01, 1.5849e-01, 6.3096e-02, 2.5119e-02, 1.0000e-02,
        #         3.9811e-03, 1.5849e-03, 6.3096e-04, 2.5119e-04])
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(max_len * 2) / d_model)
        )
        # tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 0.0000e+00,
        #          0.0000e+00],
        #         [1.0000e+00, 3.9811e-01, 1.5849e-01,  ..., 1.5849e-03, 6.3096e-04,
        #          2.5119e-04],
        #         [2.0000e+00, 7.9621e-01, 3.1698e-01,  ..., 3.1698e-03, 1.2619e-03,
        #          5.0238e-04],
        #         ...,
        #         [4.9970e+03, 1.9893e+03, 7.9197e+02,  ..., 7.9197e+00, 3.1529e+00,
        #          1.2552e+00],
        #         [4.9980e+03, 1.9897e+03, 7.9213e+02,  ..., 7.9213e+00, 3.1535e+00,
        #          1.2554e+00],
        #         [4.9990e+03, 1.9901e+03, 7.9229e+02,  ..., 7.9229e+00, 3.1542e+00,
        #          1.2557e+00]])
        # print(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # torch.Size([1, 5000, 512])
        # 5000 个位置，每个位置是 512 维的向量
        pe = pe.unsqueeze(0)
        # 通过 register_buffer() 登记过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # torch.Size([1, 1, 512])
        # torch.Size([1, 2, 512])
        # torch.Size([1, 3, 512])
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def example_positional():
    pe = PositionalEncoding(20, 0)
    # 100 个位置，d_model = 20
    # torch.Size([1, 100, 20])
    y = pe.forward(torch.zeros(1, 100, 20))

    # 当 dim = 2k 时
    # y(pos, 2k) = sin(pos/10000^(2k/d_model))
    # T = 2pi * 10000^(2k/d_model)
    # 当 2k = 4, d_model = 20 时，T = 40
    # 当 2k = 6, d_model = 20 时，T = 100

    # 当 dim = 2k + 1 时
    # y(pos, 2k + 1) = cos(pos/10000^(2k/d_model))
    # T = 2pi * 10000^(2k/d_model)
    # 当 2k + 1 = 5, d_model = 20 时，T = 40
    # 当 2k + 1 = 7, d_model = 20 时，T = 100

    # 即 2k 和 2k + 1 的周期是一样的，只不过一个是 sin 从 0 开始，一个是 cos 从 1 开始
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

    print(data)
    # UserWarning: The palette list has more values (10) than needed (4), which may not be intended.
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=data, x='position', y='embedding', hue='dimension',
                 palette=sns.color_palette(n_colors=data["dimension"].nunique()))
    plt.xlim(0, 100)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


# Here we define a function from hyperparameters to a full model.
# 注意一共有 3 * n 个多注意力头，分别是 encoder 的、decoder 的，encoder 和 decoder 交叉的，他们之间并不共享权重
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
    # trainable parameter:
    # 	0: encoder.layers.0.self_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	1: encoder.layers.0.self_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	2: encoder.layers.0.self_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	3: encoder.layers.0.self_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	4: encoder.layers.0.self_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	5: encoder.layers.0.self_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	6: encoder.layers.0.self_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	7: encoder.layers.0.self_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	8: encoder.layers.0.feed_forward.w_1.weight, torch.float32, torch.Size([2048, 512]), True
    # 	9: encoder.layers.0.feed_forward.w_1.bias, torch.float32, torch.Size([2048]), True
    # 	10: encoder.layers.0.feed_forward.w_2.weight, torch.float32, torch.Size([512, 2048]), True
    # 	11: encoder.layers.0.feed_forward.w_2.bias, torch.float32, torch.Size([512]), True
    # 	12: encoder.layers.0.res_layer_1.norm.a_2, torch.float32, torch.Size([512]), True
    # 	13: encoder.layers.0.res_layer_1.norm.b_2, torch.float32, torch.Size([512]), True
    # 	14: encoder.layers.0.res_layer_2.norm.a_2, torch.float32, torch.Size([512]), True
    # 	15: encoder.layers.0.res_layer_2.norm.b_2, torch.float32, torch.Size([512]), True
    # 	16: encoder.layers.1.self_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	17: encoder.layers.1.self_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	18: encoder.layers.1.self_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	19: encoder.layers.1.self_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	20: encoder.layers.1.self_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	21: encoder.layers.1.self_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	22: encoder.layers.1.self_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	23: encoder.layers.1.self_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	24: encoder.layers.1.feed_forward.w_1.weight, torch.float32, torch.Size([2048, 512]), True
    # 	25: encoder.layers.1.feed_forward.w_1.bias, torch.float32, torch.Size([2048]), True
    # 	26: encoder.layers.1.feed_forward.w_2.weight, torch.float32, torch.Size([512, 2048]), True
    # 	27: encoder.layers.1.feed_forward.w_2.bias, torch.float32, torch.Size([512]), True
    # 	28: encoder.layers.1.res_layer_1.norm.a_2, torch.float32, torch.Size([512]), True
    # 	29: encoder.layers.1.res_layer_1.norm.b_2, torch.float32, torch.Size([512]), True
    # 	30: encoder.layers.1.res_layer_2.norm.a_2, torch.float32, torch.Size([512]), True
    # 	31: encoder.layers.1.res_layer_2.norm.b_2, torch.float32, torch.Size([512]), True
    # 	32: encoder.norm.a_2, torch.float32, torch.Size([512]), True
    # 	33: encoder.norm.b_2, torch.float32, torch.Size([512]), True
    # 	34: decoder.layers.0.self_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	35: decoder.layers.0.self_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	36: decoder.layers.0.self_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	37: decoder.layers.0.self_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	38: decoder.layers.0.self_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	39: decoder.layers.0.self_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	40: decoder.layers.0.self_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	41: decoder.layers.0.self_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	42: decoder.layers.0.src_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	43: decoder.layers.0.src_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	44: decoder.layers.0.src_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	45: decoder.layers.0.src_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	46: decoder.layers.0.src_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	47: decoder.layers.0.src_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	48: decoder.layers.0.src_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	49: decoder.layers.0.src_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	50: decoder.layers.0.feed_forward.w_1.weight, torch.float32, torch.Size([2048, 512]), True
    # 	51: decoder.layers.0.feed_forward.w_1.bias, torch.float32, torch.Size([2048]), True
    # 	52: decoder.layers.0.feed_forward.w_2.weight, torch.float32, torch.Size([512, 2048]), True
    # 	53: decoder.layers.0.feed_forward.w_2.bias, torch.float32, torch.Size([512]), True
    # 	54: decoder.layers.0.res_layer_1.norm.a_2, torch.float32, torch.Size([512]), True
    # 	55: decoder.layers.0.res_layer_1.norm.b_2, torch.float32, torch.Size([512]), True
    # 	56: decoder.layers.0.res_layer_2.norm.a_2, torch.float32, torch.Size([512]), True
    # 	57: decoder.layers.0.res_layer_2.norm.b_2, torch.float32, torch.Size([512]), True
    # 	58: decoder.layers.0.res_layer_3.norm.a_2, torch.float32, torch.Size([512]), True
    # 	59: decoder.layers.0.res_layer_3.norm.b_2, torch.float32, torch.Size([512]), True
    # 	60: decoder.layers.1.self_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	61: decoder.layers.1.self_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	62: decoder.layers.1.self_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	63: decoder.layers.1.self_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	64: decoder.layers.1.self_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	65: decoder.layers.1.self_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	66: decoder.layers.1.self_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	67: decoder.layers.1.self_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	68: decoder.layers.1.src_attn.linears.0.weight, torch.float32, torch.Size([512, 512]), True
    # 	69: decoder.layers.1.src_attn.linears.0.bias, torch.float32, torch.Size([512]), True
    # 	70: decoder.layers.1.src_attn.linears.1.weight, torch.float32, torch.Size([512, 512]), True
    # 	71: decoder.layers.1.src_attn.linears.1.bias, torch.float32, torch.Size([512]), True
    # 	72: decoder.layers.1.src_attn.linears.2.weight, torch.float32, torch.Size([512, 512]), True
    # 	73: decoder.layers.1.src_attn.linears.2.bias, torch.float32, torch.Size([512]), True
    # 	74: decoder.layers.1.src_attn.linears.3.weight, torch.float32, torch.Size([512, 512]), True
    # 	75: decoder.layers.1.src_attn.linears.3.bias, torch.float32, torch.Size([512]), True
    # 	76: decoder.layers.1.feed_forward.w_1.weight, torch.float32, torch.Size([2048, 512]), True
    # 	77: decoder.layers.1.feed_forward.w_1.bias, torch.float32, torch.Size([2048]), True
    # 	78: decoder.layers.1.feed_forward.w_2.weight, torch.float32, torch.Size([512, 2048]), True
    # 	79: decoder.layers.1.feed_forward.w_2.bias, torch.float32, torch.Size([512]), True
    # 	80: decoder.layers.1.res_layer_1.norm.a_2, torch.float32, torch.Size([512]), True
    # 	81: decoder.layers.1.res_layer_1.norm.b_2, torch.float32, torch.Size([512]), True
    # 	82: decoder.layers.1.res_layer_2.norm.a_2, torch.float32, torch.Size([512]), True
    # 	83: decoder.layers.1.res_layer_2.norm.b_2, torch.float32, torch.Size([512]), True
    # 	84: decoder.layers.1.res_layer_3.norm.a_2, torch.float32, torch.Size([512]), True
    # 	85: decoder.layers.1.res_layer_3.norm.b_2, torch.float32, torch.Size([512]), True
    # 	86: decoder.norm.a_2, torch.float32, torch.Size([512]), True
    # 	87: decoder.norm.b_2, torch.float32, torch.Size([512]), True
    # 	88: src_embed.0.lut.weight, torch.float32, torch.Size([11, 512]), True
    # 	89: tgt_embed.0.lut.weight, torch.float32, torch.Size([11, 512]), True
    # 	90: generator.proj.weight, torch.float32, torch.Size([11, 512]), True
    # 	91: generator.proj.bias, torch.float32, torch.Size([11]), True
    # distrainable parameter:
    # distrainable buffers:
    # 	0: src_embed.1.pe, torch.float32, torch.Size([1, 5000, 512]), False
    # 	1: tgt_embed.1.pe, torch.float32, torch.Size([1, 5000, 512]), False
    # model (<class '__main__.EncoderDecoder'>) has 19851787 parameters, 14731787 (74.21%) are trainable, 占 0.07G 显存.
    list_model_parameter_summary(test_model)
    test_model.eval()

    # torch.int64
    # torch.Size([1, 10])
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    # 相当于全不 mask，即都可以看到，因为这里相当于算出整个的 memory
    # torch.Size([1, 1, 10])
    src_mask = torch.ones(1, 1, 10)

    # torch.Size([1, 10, 512])
    # src_mask 相当于没有用，因为没有 0 元素
    # 一开始先将所有的 memory 算好
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

        out = test_model.decode(memory, src_mask, tgt=ys, tgt_mask=tgt_mask)
        # 第 0 次：torch.Size([1, 1, 512])
        # 第 1 次：torch.Size([1, 2, 512])
        # 第 2 次：torch.Size([1, 3, 512])
        # 注意这里有重复计算，以 torch.Size([1, 3, 512]) 为例，其中 torch.Size([1, 1:2, 512]) 就是上一步算过的
        # print(out.shape)
        # 第 0 次：torch.Size([1, 512])
        # 第 1 次：torch.Size([1, 512])
        # 第 2 次：torch.Size([1, 512])
        # print(out[:, -1].shape)

        # 只考虑最后一次输出的概率（注意这里不是概率，而是 log(softmax(x))
        # 贪婪策略，只选择概率最高的那一个
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        print(ys)

    print("Example Untrained Model Prediction:", ys)


"""
Part 2: Model Training
Training
This section describes the training regime for our models.
"""


# We stop for a quick interlude to introduce some of the tools needed to train a standard encoder decoder model.
# First we define a batch object that holds the src and target sentences for training, as well as constructing the masks.
# src_mask，torch.Size([2, 1, 10])，只 mask 掉 pad
# tgt_mask，torch.Size([2, 9, 9])，同时 mask 掉 pad 和 未来的单词
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, _pad=2):  # 2 = <blank>
        # tensor([[1, 5, 8, 3, 4, 3, 7, 9, 4, 7],
        #         [1, 4, 5, 1, 1, 4, 8, 3, 6, 7]])
        #
        # torch.Size([2, 10])
        self.src = src
        # tensor([[[True, True, True, True, True, True, True, True, True, True]],
        #         [[True, True, True, True, True, True, True, True, True, True]]])
        # torch.Size([2, 10]) -> torch.Size([2, 1, 10])
        self.src_mask = (src != _pad).unsqueeze(-2)
        if tgt is not None:
            # tgt 去掉最后一个数
            # 注意 transformer 模型是要给 y 的初值的，因此这里就是把 tgt[:-1] 作为初值，那么 tgt[1:] 就是 label
            # 在训练的时候，并不会用每个位置预测的 label 作为下一个位置的初始值，而是一直用真值作为初始值，但是在最终预测的时候，会用每个位置的预测 label 作为 下一个位置的初始值
            # tensor([[1, 5, 8, 3, 4, 3, 7, 9, 4],
            #         [1, 4, 5, 1, 1, 4, 8, 3, 6]])
            # # torch.Size([2, 9])
            self.tgt = tgt[:, :-1]
            # tgt 去掉第一个数
            # tensor([[5, 8, 3, 4, 3, 7, 9, 4, 7],
            #         [4, 5, 1, 1, 4, 8, 3, 6, 7]])
            # # torch.Size([2, 9])
            self.tgt_y = tgt[:, 1:]
            # torch.Size([2, 9, 9])
            self.tgt_mask = self.make_std_mask(self.tgt, _pad)
            self.ntokens = (self.tgt_y != _pad).data.sum()

    # noinspection PyUnresolvedReferences
    # 同时考虑 pad_mask 和 subsequent_mask
    @staticmethod
    def make_std_mask(tgt, _pad):
        """Create a mask to hide padding and future words."""
        # 同时 mask 掉 pad 和 未来的单词
        tgt_mask = (tgt != _pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


# Next we create a generic training and scoring function to keep track of loss. We pass in a generic loss compute function that also handles parameter updates.
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    def __str__(self):
        return f"{{step: {self.step}, accum_step: {self.accum_step}, samples: {self.samples}, tokens: {self.tokens}}}"

    def __repr__(self):
        return self.__str__()


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # torch.Size([80, 10])
        # print(batch.src.shape)
        # torch.Size([80, 9])
        # print(batch.tgt.shape)
        # torch.Size([80, 1, 10])
        # print(batch.src_mask.shape)
        # torch.Size([80, 9, 9])
        # print(batch.tgt_mask.shape)
        out = model.forward(
            src=batch.src, tgt=batch.tgt, src_mask=batch.src_mask, tgt_mask=batch.tgt_mask
        )
        # print(str(i) + "-" * 80)
        # torch.Size([80, 9, 512])
        # print(out.shape)
        # torch.Size([80, 9])
        # print(batch.tgt_y.shape)
        # tensor
        # print(batch.ntokens)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# Note: This part is very important. Need to train with this setup of the model.
# This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.
# We used warmup_steps = 4000
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    # step 与 warmup 相等的时候取极值，极值大小为 warmup^-0.5 * model_size^-0.5
    # 即 warmup 越大，lr 的极值越小
    # 即 model_size 越大，lr 的极值越小
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# Example of the curves of this model for different model sizes and for optimization hyperparameters.
def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, (model_size, factor, warmup) in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step_: rate(step_, model_size, factor, warmup)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    print(opts_data)
    sns.lineplot(data=opts_data, x='step', y='Learning Rate', hue='model_size:warmup')
    plt.show()

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


# Label Smoothing
# During training, we employed label smoothing of value e_ls=0.1.
# This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
# We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution,
# we create a distribution that has confidence of the correct word and the rest of the smoothing mass distributed throughout the vocabulary.
# 标签平滑技术通过在训练时对真实标签进行平滑处理，从而减少模型对训练数据中的噪声标签的过度依赖，提高模型的泛化性能。
# 具体而言，标签平滑通过引入一定的噪声或模糊性来减小真实标签的置信度，从而迫使模型在训练时更加关注输入数据的特征，而不是过于依赖标签信息。这种平滑处理可以通过在交叉熵损失函数中引入额外的惩罚项或者对真实标签进行平滑化处理来实现。可以认为是一种正则化。
# 注意，这个时候如果是过于自信的输出，例如非 0 即 1，反而会惩罚，即 loss > 0
# 当 smoothing = 0 的事后，就是原本的 label
class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # KL 散度作为 loss，注意 input 是 log_softmax 之后的结果，而 target 默认是概率，即 log_target = False，没有经过 log，因此在计算 KS 的时候要重新取 log
        # p 可以有 0，但是 q 不能用 0，会自动 变成 0
        # p * log(p / q) = p * (logp - logq)
        # if not log_target: # default
        #     loss_pointwise = target * (target.log() - input)
        # else:
        #     loss_pointwise = target.exp() * (target - input)
        self.criterion = nn.KLDivLoss(reduction="sum", log_target=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 种类
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 用同一个值来填充，下面全都是 self.smoothing / (self.size - 2)
        # tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])
        true_dist.fill_(value=self.smoothing / (self.size - 2))
        # 将 src 中的值根据 dim 和 index 填充到 tensor中。这里有两个关键部分，1）将 src 中的值填充到 tensor中；2）dim 和 index 指定的位置，一个是src中的位置，另一个是tensor中的位置
        # 三维例子
        # self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        # self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
        # 二维例子
        # self[index[i][j]][j] = src[i][j] # if dim == 0
        # self[i][index[i][j]] = src[i][j] # if dim == 1
        # tensor([[0.1333, 0.1333, 0.6000, 0.1333, 0.1333],
        #         [0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        #         [0.6000, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.6000, 0.1333],
        #         [0.1333, 0.1333, 0.1333, 0.6000, 0.1333]])
        # 将目标的对应列索引填成 confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        #         [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        #         [0.0000, 0.1333, 0.1333, 0.1333, 0.1333],
        #         [0.0000, 0.1333, 0.1333, 0.6000, 0.1333],
        #         [0.0000, 0.1333, 0.1333, 0.6000, 0.1333]])
        # padding_idx 置于 0
        true_dist[:, self.padding_idx] = 0
        # noinspection PyTypeChecker
        # 非零元素定位，返回索引位置
        # tensor([[2]])
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 通过按 index 中给出的顺序选择索引，用值 value 填充 self 张量的元素。
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # 如果有 label 是 padding_idx，那么对应的行全都置于 0
        # 可以看到，最后的结果，除了 padding_idx 以外，其它的概率和仍为 1，表现为标签类对应的概率为 (1 - smoothing)，其它类标签对应的概率为 smoothing / (N - 1)
        # 因此 (1 - smoothing) + smoothing / (N - 1) * (N - 1) = 1，其实就是一种软分类，从非 1 即 0 变为 soft
        # 注意，这个时候如果是过于自信的输出，例如非 0 即 1，反而会惩罚，即 loss > 0
        # tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        #         [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        #         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.1333, 0.1333, 0.6000, 0.1333],
        #         [0.0000, 0.1333, 0.1333, 0.6000, 0.1333]])
        self.true_dist = true_dist
        return self.criterion(input=x, target=true_dist.clone().detach())


# Here we can see an example of how the mass is distributed to the words based on confidence.
# Label smoothing actually starts to penalize the model if it gets very confident about a given choice.
def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    # tensor([[   -inf, -1.6094, -0.3567, -2.3026,    -inf],
    #         [   -inf, -1.6094, -0.3567, -2.3026,    -inf],
    #         [   -inf, -1.6094, -0.3567, -2.3026,    -inf],
    #         [   -inf, -1.6094, -0.3567, -2.3026,    -inf],
    #         [   -inf, -1.6094, -0.3567, -2.3026,    -inf]])
    x = predict.log()
    crit(x=x, target=torch.LongTensor([2, 1, 0, 3, 3]))
    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    pivot_data = ls_data.pivot(index='rows', columns='columns')['target distribution']
    print(pivot_data)
    # 数据的 index 和 columns 分别为 heatmap 的 y 轴方向和 x 轴方向标签
    sns.heatmap(pivot_data, annot=True, fmt=".2F", cmap='viridis')
    plt.show()

    return (
        alt.Chart(ls_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


# # Label smoothing actually starts to penalize the model if it gets very confident about a given choice.
def penalization_visualization():
    def loss(x):
        d = x + 3 * 1
        # predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
        predict = torch.FloatTensor([[1.0e-10, x / d, 1 / d, 1 / d, 1 / d]])
        result = crit(x=predict.log(), target=torch.LongTensor([1])).data
        # 当 x = 27 的时候，有 predict ≈ crit.true_dist，因此 KL 散度为 0
        # 当 x 更大的时候，实际上模型有更大的自信，但在 LabelSmoothing 语境下，KL Loss 反而会上升
        # 27 tensor([[1.0000e-10, 9.0000e-01, 3.3333e-02, 3.3333e-02, 3.3333e-02]]) tensor([[0.0000, 0.9000, 0.0333, 0.0333, 0.0333]])
        if result < 1E-9:
            print(x, predict, crit.true_dist)
        return result

    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    print(loss_data)
    loss_data.plot(x="Steps", y="Loss")
    plt.show()

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


# We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.
# 这里 1 是初值，或者说句子的开始
def data_gen(min_value, max_value, batch_size, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.randint(min_value, max_value, size=(batch_size, 10))
        # tensor([[1, 5, 8, 3, 4, 3, 7, 9, 4, 7],
        #         [1, 4, 5, 1, 1, 4, 8, 3, 6, 7]])
        # torch.Size([2, 10])
        data[:, 0] = 1
        # .clone().detach() 等价于完全复制一个 tensor，即解耦了 value，也解耦了梯度
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


# Loss Computation
class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return loss.data * norm, loss


# Greedy Decoding
# This code predicts a translation using greedy decoding for simplicity.
# 等价于上面的 inference_test()
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # 要给定开头第一个字符
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        # noinspection PyUnresolvedReferences
        out = model.decode(
            memory, src_mask, tgt=ys, tgt_mask=subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 只考虑最后一次输出的概率（注意这里不是概率，而是 log(softmax(x))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# noinspection PyMissingConstructor,PyMethodOverriding
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        pass

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


# noinspection PyMissingConstructor,PyMethodOverriding
class DummyScheduler:
    def step(self):
        pass


# Train the simple copy task.
# 'main' spent 55.6565s.
def example_simple_model():
    vocab_size = 11
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, n=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        print(run_epoch(
            data_iter=data_gen(1, vocab_size, batch_size, 20),
            model=model,
            loss_compute=SimpleLossCompute(model.generator, criterion),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode="train",
        ))
        model.eval()
        # print(run_epoch(
        #     data_iter=data_gen(1, vocab_size, batch_size, 5),
        #     model=model,
        #     loss_compute=SimpleLossCompute(model.generator, criterion),
        #     optimizer=DummyOptimizer(),
        #     scheduler=DummyScheduler(),
        #     mode="eval",
        # ))

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    # (tensor(0.1317), {step: 400, accum_step: 400, samples: 32000, tokens: 288000})
    # tensor([[0, 9, 2, 3, 4, 5, 6, 7, 8, 9]])
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


"""
Part 3: A Real World Example
Now we consider a real-world example using the Multi30k German-English Translation task. 
This task is much smaller than the WMT task considered in the paper, 
but it illustrates the whole system. We also show how to use multi-gpu processing to make it really fast.
"""


# Data Loading
# We will load the dataset using torchtext and spacy for tokenization.
# # Load spacy tokenizer models, download them if they haven't been downloaded already.
# spacy 具有每种语言的模型（德语的 de_core_news_sm 和英语的 en_core_web_sm
# 首次运行，会执行 python -m spacy download de_core_news_sm，即安装 package，然后就可以用 import de_core_news_sm 来查看
# Collecting de-core-news-sm==3.7.0
#   Downloading https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl (14.6 MB)
# Installing collected packages: de-core-news-sm
# Successfully installed de-core-news-sm-3.7.0
# ✔ Download and installation successful
# You can now load the package via spacy.load('de_core_news_sm')
def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# noinspection PyTupleAssignmentBalance
# The build_vocab_from_iterator function from torchtext.vocab is used to build the vocabulary with all these components.
# It uses yield_tokens to generate the tokens for each sequence. yield_tokens takes train + val + test,
# which creates a single data iterator with all the sources, the tokenization function for the respective language (tokenize_de or tokenize_en),
# and the appropriate index for the language in the iterator (0 for German and 1 for English). It also requires the minimum frequency and the special tokens.
# The special tokens are
# “<bos>” for the start of sequences, 0
# “<eos>” for the end of sequences, 1
# “<pad>” for the padding, 2
# “<unk>” for tokens that are not present in the vocabulary, 3
def build_vocabulary(spacy_de, spacy_en, min_freq=2):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # UTF-8 error: https://github.com/pytorch/text/issues/2221
    # 下载文件自己解压即可
    print("Building German Vocabulary ...")
    (train, val, test) = Multi30k(root=TORCH_TEXT_DATA_PATH, language_pair=("de", "en"))
    # 29001 1015 1000
    # ZipperIterDataPipe 是一个用于合并多个数据源的迭代器。这个错误表明你正在尝试对一个ZipperIterDataPipe实例进行一个需要知道其长度的操作，但是这个实例在内部没有一个有效的长度值。
    # 尝试先将其转换成一个列表，然后再获取长度。例如：len(list(zipper_pipe))
    # print(len(list(train)), len(list(val)), len(list(test)))
    # tokens for each German sentence (index 0)
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=min_freq,
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = Multi30k(root=TORCH_TEXT_DATA_PATH, language_pair=("de", "en"))
    # tokens for each English sentence (index 1)
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=min_freq,
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    # # set default token for out-of-vocabulary words (OOV)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en, min_freq=2, force=False):
    assert min_freq in {1, 2}
    file_path = os.path.join(SPACY_DATA_PATH, "vocab.pt")
    if force or not is_file_exist(file_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en, min_freq=min_freq)
        torch.save((vocab_src, vocab_tgt), file_path)
    else:
        vocab_src, vocab_tgt = torch.load(file_path)
    if min_freq == 2:
        assert_equal(len(vocab_src), 8316)
        assert_equal(len(vocab_tgt), 6384)
    else:
        assert_equal(len(vocab_src), 19953)
        assert_equal(len(vocab_tgt), 11158)
    return vocab_src, vocab_tgt


# Batching matters a ton for speed. We want to have very evenly divided batches, with absolutely minimal padding.
# To do this we have to hack a bit around the default torchtext batching.
# This code patches their default batching to make sure we search over enough sentences to find tight batches.
def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,
        pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return src, tgt


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<pad>"],
        )

    # noinspection PyTupleAssignmentBalance
    train_iter, valid_iter, test_iter = Multi30k(root=TORCH_TEXT_DATA_PATH, language_pair=("de", "en"))

    train_iter_map = to_map_style_dataset(
        train_iter
    )
    # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        # Initialize the default distributed process group.
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        model = Ddp(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            # noinspection PyUnresolvedReferences
            train_dataloader.sampler.set_epoch(epoch)
            # noinspection PyUnresolvedReferences
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = os.path.join("bigdata", "%s%.2d.pt" % (config["file_prefix"], epoch))
            torch.save(module.state_dict(), file_path)
        # Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in `nvidia-smi`.
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        loss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(loss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = os.path.join("bigdata", "%sfinal.pt" % config["file_prefix"])
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


# 'main' spent 354.0479s.
def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, min_freq=2)

    model_path = os.path.join("bigdata", "multi30k_model_final.pt")
    if not is_file_exist(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    # Copy parameters and buffers from :attr:`state_dict` into this module and its descendants
    model.load_state_dict(torch.load(model_path))
    return model


# Model Averaging: The paper averages the last k checkpoints to create an ensemble effect. We can do this after the fact if we have a bunch of models:
def average(model, models):
    """Average models into model"""
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


# Load data and model for output checks
def check_outputs(
        valid_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        n_examples=15,
        pad_idx=2,
        eos_string="<eos>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
                " ".join(
                    [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                ).split(eos_string, 1)[0]
                + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        # noinspection PyTypeChecker
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


# 'main' spent 5.1898s.
def run_model_example(n_examples=5):
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, min_freq=2)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        0,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), n=6)
    assert not is_model_on_gpu(model)
    model.load_state_dict(torch.load("bigdata/multi30k_model_final.pt"))
    assert not is_model_on_gpu(model)
    model.cuda(0)
    assert is_model_on_gpu(model)

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


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
    # example_positional()
    # inference_test()
    # example_learning_schedule()
    # example_label_smoothing()
    # penalization_visualization()
    # example_simple_model()

    # spacy_de, spacy_en = load_tokenizers()
    # vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, min_freq=2, force=True)

    # load_trained_model()

    run_model_example()

    pass


if __name__ == '__main__':
    main()
